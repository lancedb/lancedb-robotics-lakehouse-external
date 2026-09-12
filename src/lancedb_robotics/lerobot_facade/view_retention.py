"""Published-view end of life and pin safety (backlog 0508).

The 0490/0491 published-view catalog had no end of life and no story for the
Lance versions a view pins surviving retention. This module is that story,
in four cooperating pieces (0111/0140/0143/0145 lifecycle precedents):

- :func:`retire_view` deletes one published view (header, file rows, pointer
  rows) with **protected-view semantics**: the view the ``lerobot_view_latest``
  pointer targets -- or the newest header for its repo_id -- is refused with a
  typed error unless ``force=True``, because retiring it would change what
  ``get_view(repo_id=...)`` (every default ``LeRobotDataset(root=<lake>)``
  open) resolves to. Deletion order is pointer rows, then header, then file
  rows: the header is what makes a view visible to the resolve chain, and a
  crash after the header delete leaves only invisible file rows the orphan
  reconciler reclaims. A postcondition re-read refuses to report success while
  any row survives (BUG-04 rule); re-running converges.
- :func:`plan_view_retention` / :func:`apply_view_retention`: an age +
  retain-N-newest-per-repo policy over the header catalog. **Report-only by
  default**: ``lake maintain`` computes and reports candidates every run, and
  deleting them requires the explicit apply flag -- published views are
  reproducibility contracts, so enforcement is an operator opt-in (the 0111
  "activating a policy is explicit" posture). The newest view per repo is
  never a candidate (``retain_latest_per_repo`` is floored at 1) and neither
  is a current pointer target.
- :func:`reconcile_orphan_view_files`: publish writes file rows first and the
  header last (crash safety), so a crashed publish leaves headerless file
  rows nothing ever reclaims -- the 0146 row-plan-chunk problem in a new
  table. Headerless file rows older than a grace window are deleted in
  bounded chunks; the delete predicate is additionally bounded by the grace
  cutoff so a concurrent re-publish's fresh rows are never swept.
- :func:`view_retention_pin_details` + :func:`view_readiness`: pin safety.
  The pin-details map (same shape as ``snapshot_retention_pin_details``)
  feeds ``maintain_lake``'s tag-before-cleanup step, so versions pinned by
  non-retired views are tagged and version pruning can no longer turn a
  published view into :class:`~.views.StaleViewVersionError`. The readiness
  report (0143 ``curation_replay_readiness`` shape) classifies each pinned
  ``(table, version)`` as protected / unprotected / pruned / unreadable and
  rolls that up per view -- catching views whose pins were already pruned
  before this shipped. :func:`view_pin_conformance` classifies the backend
  (supported / capability-gated / unavailable) so ``db://`` and
  namespace-managed lakes get a typed posture instead of silence (0116
  invariant).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from lancedb_robotics.capability_gates import DIRECT_LANCE, VERSIONING, backend_supports

from .views import (
    VIEW_FILES_TABLE,
    VIEW_LATEST_TABLE,
    VIEWS_TABLE,
    ViewError,
    _latest_pointer_view_id,
    _sql_literal,
    _timestamp_literal,
    _update_latest_pointer,
)

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from lancedb_robotics.lake import Lake

#: Versioned report contracts (0134/0135 convention).
RETENTION_REPORT_VERSION = "lerobot-view-retention/1"
READINESS_SCHEMA_VERSION = "lancedb-robotics/lerobot-view-readiness/v1"

#: Backend conformance classes for pinned published-view opens (0143 vocabulary).
VIEW_PIN_SUPPORTED = "supported"
VIEW_PIN_CAPABILITY_GATED = "capability-gated"
VIEW_PIN_UNAVAILABLE = "unavailable"

#: Per-pin protection/readability states (0143 vocabulary).
PIN_PROTECTED = "protected"
PIN_UNPROTECTED = "unprotected"
PIN_PRUNED = "pruned"
PIN_UNREADABLE = "unreadable"
PIN_BACKEND_GATED = "backend-gated"

#: Overall readiness verdicts.
READY = "ready"
AT_RISK = "at-risk"
BACKEND_GATED = "backend-gated"

_CAPABILITY = "table_versioning"

#: Mirrors ``maintenance._PIN_TAG_PREFIX`` (the managed pin tag written by
#: ``lake maintain``'s tag-before-cleanup step); kept as a local constant so
#: this module never imports maintenance (which imports this module).
#: ``test_pin_tag_prefix_matches_maintenance`` fails if the two ever drift.
_PIN_TAG_PREFIX = "lbr-snapshot-pin-v"

#: Policy defaults: a view is a retention candidate only when it is older than
#: this AND not among the newest ``DEFAULT_RETAIN_LATEST_PER_REPO`` for its
#: repo_id AND not a current pointer target.
DEFAULT_VIEW_RETENTION_AGE = timedelta(days=90)
DEFAULT_RETAIN_LATEST_PER_REPO = 5

#: Headerless file rows younger than this are treated as an in-flight publish
#: (publish stamps every file row with one fresh ``created_at`` and writes the
#: header immediately after the last file flush), never reclaimed.
DEFAULT_ORPHAN_GRACE = timedelta(hours=24)

_SCAN_BATCH_SIZE = 4_096
_KEY_CHUNK = 32
#: Row bound for unordered (dict/list) detection fallbacks (0507 discipline):
#: past it the pass reports itself skipped, never buffers unbounded state.
_MAX_UNORDERED_SCAN_ROWS = 100_000
#: Pointer-target set bound for retention planning; past it planning continues
#: but reports the cross-check truncated (the newest-per-repo rule still
#: protects every healthy pointer target, and the 0507 reconciler heals a
#: dangling pointer left by retiring a stale one).
_MAX_POINTER_TARGETS = 100_000
#: Header rows tolerated by :func:`view_retention_pin_details` before it
#: refuses: silently skipping pins would let cleanup prune versions live views
#: still need, which is exactly the data-loss shape this module exists to
#: prevent -- so past the bound it fails loudly instead of degrading.
_MAX_PIN_VIEWS = 100_000
_MAX_RETENTION_CANDIDATES = 1_000
_MAX_ORPHAN_RECLAIMS_PER_RUN = 10_000
_MAX_READINESS_VIEWS = 10_000
_MAX_CHECKOUT_PROBES = 1_000
#: Sample view ids retained per pin / at-risk entries retained per report so
#: report size stays bounded regardless of catalog size.
_PIN_VIEW_SAMPLE = 4
_MAX_AT_RISK_LISTED = 32


class ViewRetirementError(ViewError):
    """Raised when a retire/retention operation cannot proceed or verify safely."""


class ProtectedViewError(ViewRetirementError):
    """Typed refusal to retire a view the resolve chain still depends on."""


# ---------------------------------------------------------------------------
# retire_view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewRetirementReport:
    """Outcome of one :func:`retire_view` call."""

    view_id: str
    repo_id: str
    #: "retired", "dry-run" (nothing written), or "absent" (no header, file, or
    #: pointer row references this view id -- retire is idempotent, so a re-run
    #: after success or a crash converges here instead of raising).
    status: str
    header_rows_deleted: int = 0
    file_rows_deleted: int = 0
    pointer_rows_deleted: int = 0
    #: "untouched" (pointer never referenced this view), "repointed" (moved to
    #: the newest remaining header), "removed" (no header remains for the
    #: repo_id), or "fallback-scan" (pointer rows deleted but the re-point
    #: could not land; resolves fall back to the ordered catalog read and the
    #: 0507 reconciler converges it -- degraded loudly, never silently).
    pointer_action: str = "untouched"
    forced: bool = False
    detail: str = ""

    def to_params(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "repo_id": self.repo_id,
            "status": self.status,
            "header_rows_deleted": self.header_rows_deleted,
            "file_rows_deleted": self.file_rows_deleted,
            "pointer_rows_deleted": self.pointer_rows_deleted,
            "pointer_action": self.pointer_action,
            "forced": self.forced,
            "detail": self.detail,
        }


def _count_rows(lake: Lake, table_name: str, where: str, column: str) -> int:
    """Bounded streaming count of physical rows matching ``where``."""
    try:
        table = lake.table(table_name)
    except Exception:  # noqa: BLE001 - table absent (pre-0490/0507 lakes).
        return 0
    count = 0
    for batch in (
        table.search().select([column]).where(where).to_batches(batch_size=_SCAN_BATCH_SIZE)
    ):
        count += batch.num_rows
    return count


def _newest_header_for_repo(
    lake: Lake, repo_id: str, *, exclude_view_id: str | None = None
) -> tuple[str, Any] | None:
    """Newest ``(view_id, created_at)`` header for ``repo_id`` via ordered read.

    Same shape as ``view_lifecycle._newest_header_ordered`` with an optional
    exclusion (retire needs "newest remaining"). Returns ``("", None)`` when no
    header matches and ``None`` when the backend cannot order the scan.
    """
    where = f"repo_id = {_sql_literal(repo_id)}"
    if exclude_view_id is not None:
        where += f" AND view_id <> {_sql_literal(exclude_view_id)}"
    try:
        from lancedb.query import ColumnOrdering

        batches = (
            lake.table(VIEWS_TABLE)
            .search()
            .select(["view_id", "created_at"])
            .where(where)
            .order_by(
                [
                    ColumnOrdering(column_name="created_at", ascending=False),
                    ColumnOrdering(column_name="view_id", ascending=False),
                ]
            )
            .to_batches(batch_size=8)
        )
        for batch in batches:
            rows = batch.to_pylist()
            if rows:
                return (str(rows[0]["view_id"]), rows[0]["created_at"])
        return ("", None)
    except Exception:  # noqa: BLE001 - ordering unavailable; caller decides.
        return None


def retire_view(
    lake: Lake,
    view_id: str,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> ViewRetirementReport:
    """Retire one published view: delete its header, file rows, and pointer rows.

    Protected-view semantics: the current ``lerobot_view_latest`` target for
    the view's repo_id and the newest header per repo_id are refused with
    :class:`ProtectedViewError` unless ``force=True`` -- and when the backend
    cannot order the header scan the protection check **fails closed** (it
    refuses rather than guessing). With ``force`` the pointer is repaired to
    the newest remaining header (or removed when none remains).

    Idempotent and crash-convergent: a re-run after any crash window finishes
    the job, and a fully-absent view id returns ``status="absent"`` rather
    than raising (so convergence re-runs and typo'd ids are distinguishable
    only by the report, deliberately -- the CLI prints it).
    """
    view_id = str(view_id)
    literal = _sql_literal(view_id)

    header_rows = _header_copies(lake, view_id)
    header_count = len(header_rows)
    repo_id = str(header_rows[0]["repo_id"]) if header_rows else ""
    file_count = _count_rows(lake, VIEW_FILES_TABLE, f"view_id = {literal}", "file_id")
    pointer_count = _count_rows(lake, VIEW_LATEST_TABLE, f"view_id = {literal}", "repo_id")

    if not header_count and not file_count and not pointer_count:
        return ViewRetirementReport(
            view_id=view_id,
            repo_id="",
            status="absent",
            detail="no header, file, or pointer row references this view id",
        )

    if header_count and not force:
        _require_not_protected(lake, view_id, repo_id)

    if dry_run:
        return ViewRetirementReport(
            view_id=view_id,
            repo_id=repo_id,
            status="dry-run",
            header_rows_deleted=header_count,
            file_rows_deleted=file_count,
            pointer_rows_deleted=pointer_count,
            pointer_action="repointed" if pointer_count else "untouched",
            forced=force,
        )

    # Pointer rows first: while they exist, resolves may still hand out this
    # view id. Deleting them makes resolves fall back to the ordered catalog
    # read, which stops returning the view the moment the header goes.
    if pointer_count:
        lake.table(VIEW_LATEST_TABLE).delete(f"view_id = {literal}")
    if header_count:
        lake.table(VIEWS_TABLE).delete(f"view_id = {literal}")
    if file_count:
        lake.table(VIEW_FILES_TABLE).delete(f"view_id = {literal}")

    # Postcondition (BUG-04 rule): never report success while rows survive.
    surviving_headers = _count_rows(lake, VIEWS_TABLE, f"view_id = {literal}", "view_id")
    surviving_files = _count_rows(lake, VIEW_FILES_TABLE, f"view_id = {literal}", "file_id")
    surviving_pointers = _count_rows(lake, VIEW_LATEST_TABLE, f"view_id = {literal}", "repo_id")
    if surviving_headers or surviving_files or surviving_pointers:
        raise ViewRetirementError(
            f"retire postcondition failed for view {view_id!r}: "
            f"{surviving_headers} header / {surviving_files} file / "
            f"{surviving_pointers} pointer row(s) survived the delete; "
            "re-run retire_view to converge"
        )

    pointer_action = "untouched"
    if pointer_count:
        # Best-effort re-point (0507 publish posture: pointer maintenance
        # degrades loudly in the report, never fails the operation -- resolves
        # fall back to the ordered read and `lake maintain` reconciles).
        pointer_action = "removed"
        newest = _newest_header_for_repo(lake, repo_id) if repo_id else ("", None)
        if newest is None:
            pointer_action = "fallback-scan"
        elif newest[0]:
            try:
                repointed = _update_latest_pointer(
                    lake, repo_id=repo_id, view_id=newest[0], view_created_at=newest[1]
                )
                pointer_action = "repointed" if repointed else "fallback-scan"
            except Exception:  # noqa: BLE001 - degrade loudly, never fail retire.
                pointer_action = "fallback-scan"

    return ViewRetirementReport(
        view_id=view_id,
        repo_id=repo_id,
        status="retired",
        header_rows_deleted=header_count,
        file_rows_deleted=file_count,
        pointer_rows_deleted=pointer_count,
        pointer_action=pointer_action,
        forced=force,
    )


def _header_copies(lake: Lake, view_id: str) -> list[dict[str, Any]]:
    """All physical header copies for ``view_id`` (projected, bounded count)."""
    try:
        table = lake.table(VIEWS_TABLE)
    except Exception:  # noqa: BLE001 - table absent on pre-0490 lakes.
        return []
    rows: list[dict[str, Any]] = []
    for batch in (
        table.search()
        .select(["view_id", "repo_id", "created_at"])
        .where(f"view_id = {_sql_literal(view_id)}")
        .to_batches(batch_size=_SCAN_BATCH_SIZE)
    ):
        rows.extend(batch.to_pylist())
    return rows


def _require_not_protected(lake: Lake, view_id: str, repo_id: str) -> None:
    pointer_target = _latest_pointer_view_id(lake, repo_id)
    if pointer_target == view_id:
        raise ProtectedViewError(
            f"view {view_id!r} is the current lerobot_view_latest target for "
            f"repo_id {repo_id!r} -- retiring it would change what "
            "get_view(repo_id=...) resolves. Publish a newer view first, or pass "
            "force=True (`train view retire --force`) to retire it and re-point "
            "the pointer at the newest remaining view."
        )
    newest = _newest_header_for_repo(lake, repo_id)
    if newest is None:
        # Fail closed: without an ordered read we cannot prove this is not the
        # newest published view for its repo (SKILLS.md: never silently guess).
        raise ProtectedViewError(
            f"cannot verify view {view_id!r} is not the newest published view "
            f"for repo_id {repo_id!r} (backend cannot order the header scan); "
            "pass force=True to retire it anyway"
        )
    if newest[0] == view_id:
        raise ProtectedViewError(
            f"view {view_id!r} is the newest published view for repo_id "
            f"{repo_id!r} -- retiring it would regress every "
            "get_view(repo_id=...) resolve. Publish a newer view first, or pass "
            "force=True to retire it anyway."
        )


# ---------------------------------------------------------------------------
# Retention policy (report-only by default)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewRetentionCandidate:
    """One view the policy would retire."""

    view_id: str
    repo_id: str
    created_at: Any

    def to_params(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "repo_id": self.repo_id,
            "created_at": (
                self.created_at.isoformat()
                if isinstance(self.created_at, datetime)
                else self.created_at
            ),
        }


@dataclass(frozen=True)
class ViewRetentionReport:
    """Result of one retention plan (and optional apply) run."""

    report_version: str
    #: "planned" (report-only), "applied", "absent" (no view catalog), or
    #: "skipped-unordered-over-bound" (backend cannot order the header scan and
    #: the catalog is too large for the bounded fallback).
    status: str
    older_than_seconds: float | None
    retain_latest_per_repo: int
    views_seen: int = 0
    repos_seen: int = 0
    candidates: tuple[ViewRetentionCandidate, ...] = ()
    candidates_remaining: int = 0
    #: Views skipped because their ``created_at`` is null (written outside
    #: publish); age cannot be established so the policy never touches them.
    skipped_null_created_at: int = 0
    #: True when the pointer-target cross-check set hit its bound; the
    #: newest-per-repo rule still protects healthy pointer targets, and the
    #: 0507 reconciler heals a dangling pointer, but the direct check was
    #: truncated -- reported, never silent.
    pointer_targets_truncated: bool = False
    applied: tuple[dict[str, Any], ...] = ()
    skipped_protected: int = 0
    apply_errors: tuple[str, ...] = ()
    detail: str = ""

    def to_params(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "status": self.status,
            "older_than_seconds": self.older_than_seconds,
            "retain_latest_per_repo": self.retain_latest_per_repo,
            "views_seen": self.views_seen,
            "repos_seen": self.repos_seen,
            "candidates": [item.to_params() for item in self.candidates],
            "candidate_count": len(self.candidates),
            "candidates_remaining": self.candidates_remaining,
            "skipped_null_created_at": self.skipped_null_created_at,
            "pointer_targets_truncated": self.pointer_targets_truncated,
            "applied": [dict(item) for item in self.applied],
            "skipped_protected": self.skipped_protected,
            "apply_errors": list(self.apply_errors),
            "detail": self.detail,
        }


def _pointer_targets(lake: Lake) -> tuple[set[str], bool]:
    """Distinct pointed-at view ids, bounded; ``(set, truncated)``."""
    targets: set[str] = set()
    try:
        batches = (
            lake.table(VIEW_LATEST_TABLE)
            .search()
            .select(["view_id"])
            .to_batches(batch_size=_SCAN_BATCH_SIZE)
        )
    except Exception:  # noqa: BLE001 - pointer table absent on pre-0507 lakes.
        return targets, False
    try:
        for batch in batches:
            for row in batch.to_pylist():
                if len(targets) >= _MAX_POINTER_TARGETS:
                    return targets, True
                targets.add(str(row["view_id"]))
    except Exception:  # noqa: BLE001 - mid-scan failure: report truncated.
        return targets, True
    return targets, False


def _iter_headers_repo_ordered(lake: Lake):
    """Headers ordered ``(repo_id asc, created_at desc, view_id desc)``.

    Returns a batch-row iterator or ``None`` when the backend cannot order the
    scan (caller falls back to the bounded unordered path).
    """
    try:
        from lancedb.query import ColumnOrdering

        batches = (
            lake.table(VIEWS_TABLE)
            .search()
            .select(["view_id", "repo_id", "created_at"])
            .order_by(
                [
                    ColumnOrdering(column_name="repo_id", ascending=True),
                    ColumnOrdering(column_name="created_at", ascending=False),
                    ColumnOrdering(column_name="view_id", ascending=False),
                ]
            )
            .to_batches(batch_size=_SCAN_BATCH_SIZE)
        )
        iterator = iter(batches)
        first = next(iterator, None)
    except Exception:  # noqa: BLE001 - ordering unavailable.
        return None

    def _rows():
        if first is not None:
            yield from first.to_pylist()
        for batch in iterator:
            yield from batch.to_pylist()

    return _rows()


def _headers_unordered_grouped(lake: Lake) -> list[dict[str, Any]] | None:
    """Bounded unordered fallback: all headers, sorted client-side.

    Groups per repo and sorts each group ``(created_at desc, view_id desc)``
    with null ``created_at`` sorting oldest, matching the backend ordering the
    primary path requests. Returns ``None`` when the catalog exceeds the loud
    row bound (0507 discipline: never an unbounded in-memory sort).
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    rows_seen = 0
    for batch in (
        lake.table(VIEWS_TABLE)
        .search()
        .select(["view_id", "repo_id", "created_at"])
        .to_batches(batch_size=_SCAN_BATCH_SIZE)
    ):
        for row in batch.to_pylist():
            rows_seen += 1
            if rows_seen > _MAX_UNORDERED_SCAN_ROWS:
                return None
            groups.setdefault(str(row["repo_id"]), []).append(row)
    ordered: list[dict[str, Any]] = []
    for repo_id in sorted(groups):
        # Collapse physical duplicate copies to the newest per view_id BEFORE
        # ranking: a duplicate's older stamp must never demote its view past
        # the retain window (scale-review finding 1).
        newest_by_id: dict[str, dict[str, Any]] = {}
        for row in groups[repo_id]:
            view_id = str(row["view_id"])
            kept = newest_by_id.get(view_id)
            row_key = (row.get("created_at") is not None, row.get("created_at"))
            kept_key = (
                (kept.get("created_at") is not None, kept.get("created_at"))
                if kept is not None
                else None
            )
            if kept is None or row_key > kept_key:
                newest_by_id[view_id] = row
        items = list(newest_by_id.values())
        # reverse=True gives created_at desc / view_id desc; the leading
        # is-not-None flag makes null created_at sort oldest under reverse.
        items.sort(
            key=lambda row: (
                (row.get("created_at") is not None, row.get("created_at")),
                str(row["view_id"]),
            ),
            reverse=True,
        )
        ordered.extend(items)
    return ordered


def plan_view_retention(
    lake: Lake,
    *,
    older_than: timedelta | None = DEFAULT_VIEW_RETENTION_AGE,
    retain_latest_per_repo: int = DEFAULT_RETAIN_LATEST_PER_REPO,
    max_candidates: int = _MAX_RETENTION_CANDIDATES,
    now: datetime | None = None,
) -> ViewRetentionReport:
    """Compute (never delete) the views the retention policy would retire.

    A view is a candidate only when **all** hold: it is not among the newest
    ``retain_latest_per_repo`` headers for its repo_id (floored at 1 -- the
    newest view per repo is categorically never retired by policy), its
    ``created_at`` is older than ``older_than`` (``None`` disables the age
    gate: count-based only), and it is not a current pointer target. Bounded:
    the primary path is a backend-ordered streaming scan with O(1) client
    memory; the unordered fallback is loudly bounded (0507 discipline); at
    most ``max_candidates`` are returned per run and the remainder is counted.
    """
    if older_than is not None and older_than < timedelta(0):
        raise ValueError("older_than must be non-negative or None")
    if retain_latest_per_repo < 1:
        raise ValueError(
            "retain_latest_per_repo must be >= 1: the newest published view per "
            "repo_id is never retired by policy (it is what get_view resolves)"
        )

    try:
        lake.table(VIEWS_TABLE)
    except Exception:  # noqa: BLE001 - no view catalog on this lake.
        return ViewRetentionReport(
            report_version=RETENTION_REPORT_VERSION,
            status="absent",
            older_than_seconds=older_than.total_seconds() if older_than else None,
            retain_latest_per_repo=retain_latest_per_repo,
        )

    moment = now or datetime.now(UTC)
    cutoff = (moment - older_than) if older_than is not None else None
    pointer_targets, targets_truncated = _pointer_targets(lake)

    rows = _iter_headers_repo_ordered(lake)
    status = "planned"
    detail = ""
    if rows is None:
        grouped = _headers_unordered_grouped(lake)
        if grouped is None:
            return ViewRetentionReport(
                report_version=RETENTION_REPORT_VERSION,
                status="skipped-unordered-over-bound",
                older_than_seconds=older_than.total_seconds() if older_than else None,
                retain_latest_per_repo=retain_latest_per_repo,
                pointer_targets_truncated=targets_truncated,
                detail=(
                    "backend cannot order the header scan and the catalog holds "
                    f"more than {_MAX_UNORDERED_SCAN_ROWS} rows; refusing an "
                    "unbounded in-memory sort"
                ),
            )
        rows = iter(grouped)

    views_seen = 0
    repos_seen = 0
    skipped_null = 0
    candidates: list[ViewRetentionCandidate] = []
    remaining = 0
    current_repo: str | None = None
    rank = 0
    # Duplicate physical copies of one view are NOT adjacent under the
    # (created_at desc) ordering -- concurrent identical publishers stamp their
    # own created_at, so copies of view A can straddle a different view B.
    # Ranking a later, older copy again would push a view the policy promised
    # to retain past the retain window (scale-review finding 1), so dedup uses
    # a per-repo id set (reset at each repo boundary, loudly bounded).
    seen_repo_views: set[str] = set()
    scan_error = ""
    try:
        for row in rows:
            repo_id = str(row["repo_id"])
            view_id = str(row["view_id"])
            if repo_id != current_repo:
                current_repo = repo_id
                repos_seen += 1
                rank = 0
                seen_repo_views = set()
            if view_id in seen_repo_views:
                continue
            if len(seen_repo_views) >= _MAX_UNORDERED_SCAN_ROWS:
                return ViewRetentionReport(
                    report_version=RETENTION_REPORT_VERSION,
                    status="skipped-unordered-over-bound",
                    older_than_seconds=older_than.total_seconds() if older_than else None,
                    retain_latest_per_repo=retain_latest_per_repo,
                    views_seen=views_seen,
                    repos_seen=repos_seen,
                    pointer_targets_truncated=targets_truncated,
                    detail=(
                        f"repo {repo_id!r} holds more than {_MAX_UNORDERED_SCAN_ROWS} "
                        "distinct published views; refusing an unbounded per-repo "
                        "dedup set -- retire views for this repo explicitly"
                    ),
                )
            seen_repo_views.add(view_id)
            views_seen += 1
            rank += 1
            if rank <= retain_latest_per_repo:
                continue
            created_at = row.get("created_at")
            if created_at is None:
                skipped_null += 1
                continue
            if cutoff is not None and created_at >= cutoff:
                continue
            if view_id in pointer_targets:
                continue
            if len(candidates) >= max_candidates:
                remaining += 1
                continue
            candidates.append(
                ViewRetentionCandidate(view_id=view_id, repo_id=repo_id, created_at=created_at)
            )
    except Exception as exc:  # noqa: BLE001 - lazy ordering rejection mid-scan.
        # Stop cleanly (0507 posture): candidates found so far are individually
        # valid; the run reports itself aborted and a re-run converges.
        status = "skipped-ordered-scan-aborted"
        scan_error = str(exc)
        detail = scan_error

    return ViewRetentionReport(
        report_version=RETENTION_REPORT_VERSION,
        status=status,
        older_than_seconds=older_than.total_seconds() if older_than else None,
        retain_latest_per_repo=retain_latest_per_repo,
        views_seen=views_seen,
        repos_seen=repos_seen,
        candidates=tuple(candidates),
        candidates_remaining=remaining,
        skipped_null_created_at=skipped_null,
        pointer_targets_truncated=targets_truncated,
        detail=detail,
    )


def apply_view_retention(
    lake: Lake,
    *,
    older_than: timedelta | None = DEFAULT_VIEW_RETENTION_AGE,
    retain_latest_per_repo: int = DEFAULT_RETAIN_LATEST_PER_REPO,
    max_candidates: int = _MAX_RETENTION_CANDIDATES,
    dry_run: bool = True,
    now: datetime | None = None,
) -> ViewRetentionReport:
    """Plan and (only when ``dry_run=False``) retire retention candidates.

    ``dry_run=True`` -- the default, and what ``lake maintain`` runs unless the
    operator passes the apply flag -- returns the plan untouched. Enforcement
    retires each candidate through :func:`retire_view` (never forced): a
    candidate that became protected mid-run (raced by a newer publish making it
    the pointer target) is counted skipped, a candidate whose postcondition
    fails is recorded as an error, and both converge on a re-run.
    """
    plan = plan_view_retention(
        lake,
        older_than=older_than,
        retain_latest_per_repo=retain_latest_per_repo,
        max_candidates=max_candidates,
        now=now,
    )
    if dry_run or plan.status != "planned" or not plan.candidates:
        return plan

    applied: list[dict[str, Any]] = []
    skipped_protected = 0
    errors: list[str] = []
    for candidate in plan.candidates:
        try:
            applied.append(retire_view(lake, candidate.view_id).to_params())
        except ProtectedViewError:
            skipped_protected += 1
        except ViewRetirementError as exc:
            errors.append(f"{candidate.view_id}: {exc}")
    return ViewRetentionReport(
        report_version=plan.report_version,
        status="applied",
        older_than_seconds=plan.older_than_seconds,
        retain_latest_per_repo=plan.retain_latest_per_repo,
        views_seen=plan.views_seen,
        repos_seen=plan.repos_seen,
        candidates=plan.candidates,
        candidates_remaining=plan.candidates_remaining,
        skipped_null_created_at=plan.skipped_null_created_at,
        pointer_targets_truncated=plan.pointer_targets_truncated,
        applied=tuple(applied),
        skipped_protected=skipped_protected,
        apply_errors=tuple(errors),
        detail=plan.detail,
    )


# ---------------------------------------------------------------------------
# Orphan file-row reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrphanViewFileReport:
    """Result of one :func:`reconcile_orphan_view_files` run."""

    #: "reconciled", "absent", "skipped-unordered-over-bound", or
    #: "skipped-ordered-scan-aborted" (deletes already applied are safe; a
    #: re-run converges).
    status: str
    file_rows_seen: int = 0
    distinct_view_ids: int = 0
    orphan_view_ids: int = 0
    orphan_view_ids_reclaimed: int = 0
    #: Verified by a post-delete re-count (BUG-04 rule), except under
    #: ``dry_run`` where it is the planned count.
    file_rows_deleted: int = 0
    #: Rows matching a delete predicate that survived it (a delete that did
    #: nothing, or rows a concurrent writer replaced); a re-run converges.
    file_rows_surviving: int = 0
    in_grace_view_ids: int = 0
    reclaims_remaining: int = 0
    dry_run: bool = False
    detail: str = ""

    def to_params(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "file_rows_seen": self.file_rows_seen,
            "distinct_view_ids": self.distinct_view_ids,
            "orphan_view_ids": self.orphan_view_ids,
            "orphan_view_ids_reclaimed": self.orphan_view_ids_reclaimed,
            "file_rows_deleted": self.file_rows_deleted,
            "file_rows_surviving": self.file_rows_surviving,
            "in_grace_view_ids": self.in_grace_view_ids,
            "reclaims_remaining": self.reclaims_remaining,
            "dry_run": self.dry_run,
            "detail": self.detail,
        }


@dataclass
class _FileGroup:
    view_id: str
    rows: int = 0
    newest: datetime | None = None


def _iter_file_groups_ordered(lake: Lake):
    """Yield per-view_id file groups from a backend-ordered scan (O(1) memory)."""
    table = lake.table(VIEW_FILES_TABLE)
    try:
        from lancedb.query import ColumnOrdering

        batches = (
            table.search()
            .select(["view_id", "created_at"])
            .order_by([ColumnOrdering(column_name="view_id", ascending=True)])
            .to_batches(batch_size=_SCAN_BATCH_SIZE)
        )
        iterator = iter(batches)
        first = next(iterator, None)
    except Exception:  # noqa: BLE001 - ordering unavailable; caller falls back.
        return None

    stats = {"rows": 0, "distinct": 0}

    def _generate():
        current: _FileGroup | None = None
        chunks = [first] if first is not None else []

        def _batches():
            yield from chunks
            yield from iterator

        try:
            for batch in _batches():
                for row in batch.to_pylist():
                    stats["rows"] += 1
                    view_id = str(row["view_id"])
                    if current is None or view_id != current.view_id:
                        if current is not None:
                            yield current
                        stats["distinct"] += 1
                        current = _FileGroup(view_id=view_id)
                    current.rows += 1
                    created = row.get("created_at")
                    if created is not None and (current.newest is None or created > current.newest):
                        current.newest = created
        except Exception as exc:  # noqa: BLE001 - lazy ordering rejection mid-scan.
            stats["scan_error"] = str(exc)
            return
        if current is not None:
            yield current

    return _generate(), stats


def _iter_file_groups_unordered(lake: Lake):
    """Dict fallback under the loud row bound (0507 discipline)."""
    table = lake.table(VIEW_FILES_TABLE)
    groups: dict[str, _FileGroup] = {}
    rows = 0
    for batch in (
        table.search().select(["view_id", "created_at"]).to_batches(batch_size=_SCAN_BATCH_SIZE)
    ):
        for row in batch.to_pylist():
            rows += 1
            if rows > _MAX_UNORDERED_SCAN_ROWS:
                raise ViewRetirementError(
                    "backend cannot order the view_id-keyed file scan and the "
                    f"table holds more than {_MAX_UNORDERED_SCAN_ROWS} rows; "
                    "refusing an unbounded in-memory group map"
                )
            view_id = str(row["view_id"])
            group = groups.get(view_id)
            if group is None:
                group = _FileGroup(view_id=view_id)
                groups[view_id] = group
            group.rows += 1
            created = row.get("created_at")
            if created is not None and (group.newest is None or created > group.newest):
                group.newest = created
    ordered = sorted(groups.values(), key=lambda group: group.view_id)
    return iter(ordered), {"rows": rows, "distinct": len(groups)}


def _headers_exist(lake: Lake, view_ids: list[str]) -> set[str]:
    """Which of ``view_ids`` have a header row (chunked ``IN`` point reads)."""
    present: set[str] = set()
    table = lake.table(VIEWS_TABLE)
    for start in range(0, len(view_ids), _KEY_CHUNK):
        chunk = view_ids[start : start + _KEY_CHUNK]
        literals = ", ".join(_sql_literal(view_id) for view_id in chunk)
        for batch in (
            table.search()
            .select(["view_id"])
            .where(f"view_id IN ({literals})")
            .to_batches(batch_size=_SCAN_BATCH_SIZE)
        ):
            for row in batch.to_pylist():
                present.add(str(row["view_id"]))
    return present


def reconcile_orphan_view_files(
    lake: Lake,
    *,
    grace: timedelta = DEFAULT_ORPHAN_GRACE,
    max_reclaims: int = _MAX_ORPHAN_RECLAIMS_PER_RUN,
    dry_run: bool = False,
    now: datetime | None = None,
) -> OrphanViewFileReport:
    """Reclaim ``lerobot_view_files`` rows whose view has no header row.

    Publish writes file rows first and the header last, so a crashed publish
    leaves headerless (invisible, but never reclaimed) file rows; a crashed
    :func:`retire_view` leaves the same shape. Groups the file table by
    ``view_id`` (backend-ordered primary path, loudly-bounded dict fallback),
    checks header existence in chunked ``IN`` point reads, and deletes
    headerless groups whose newest ``created_at`` is older than ``grace``.
    The delete predicate is itself bounded by the grace cutoff, so a
    concurrent re-publish of the same view (which refreshes ``created_at`` on
    the rows it upserts) never has fresh rows swept out from under its own
    postcondition check. Rows with null ``created_at`` were not written by
    publish (it always stamps one) and are reclaimed. Bounded work per run;
    re-running converges.
    """
    if grace < timedelta(0):
        raise ValueError("grace must be non-negative")
    try:
        lake.table(VIEW_FILES_TABLE)
    except Exception:  # noqa: BLE001 - table absent on pre-0491 lakes.
        return OrphanViewFileReport(status="absent", dry_run=dry_run)

    moment = now or datetime.now(UTC)
    cutoff = moment - grace
    cutoff_literal = _timestamp_literal(cutoff)

    ordered = _iter_file_groups_ordered(lake)
    if ordered is not None:
        groups, stats = ordered
    else:
        try:
            groups, stats = _iter_file_groups_unordered(lake)
        except ViewRetirementError as exc:
            return OrphanViewFileReport(
                status="skipped-unordered-over-bound", dry_run=dry_run, detail=str(exc)
            )

    orphan_ids = 0
    reclaimed = 0
    rows_deleted = 0
    rows_surviving = 0
    in_grace = 0
    remaining = 0
    pending: list[tuple[str, int]] = []  # (view_id, planned row deletions)

    def _flush(batch: list[_FileGroup]) -> None:
        nonlocal orphan_ids, reclaimed, rows_deleted, in_grace, remaining, pending
        if not batch:
            return
        present = _headers_exist(lake, [group.view_id for group in batch])
        for group in batch:
            if group.view_id in present:
                continue
            if group.newest is not None and group.newest > cutoff:
                in_grace += 1
                continue
            orphan_ids += 1
            if reclaimed + len(pending) >= max_reclaims:
                remaining += 1
                continue
            pending.append((group.view_id, group.rows))
        if len(pending) >= _KEY_CHUNK:
            _delete_pending()

    def _delete_pending() -> None:
        nonlocal reclaimed, rows_deleted, rows_surviving, pending
        if not pending:
            return
        literals = ", ".join(_sql_literal(view_id) for view_id, _rows in pending)
        # Bounded by the grace cutoff too: a re-publish racing this sweep
        # stamps its upserted rows fresh, keeping them out of the predicate.
        predicate = (
            f"view_id IN ({literals}) AND "
            f"(created_at <= {cutoff_literal} OR created_at IS NULL)"
        )
        planned = sum(rows for _view_id, rows in pending)
        if dry_run:
            rows_deleted += planned
        else:
            lake.table(VIEW_FILES_TABLE).delete(predicate)
            # BUG-04 rule: count what actually went, never report the plan as
            # the outcome. Survivors (a delete that did nothing) are reported
            # and a re-run converges on them.
            survivors = _count_rows(lake, VIEW_FILES_TABLE, predicate, "file_id")
            rows_deleted += max(planned - survivors, 0)
            rows_surviving += survivors
        reclaimed += len(pending)
        pending = []

    batch: list[_FileGroup] = []
    for group in groups:
        batch.append(group)
        if len(batch) >= _KEY_CHUNK:
            _flush(batch)
            batch = []
    _flush(batch)
    _delete_pending()

    scan_error = str(stats.get("scan_error") or "")
    status = "skipped-ordered-scan-aborted" if scan_error else "reconciled"
    return OrphanViewFileReport(
        status=status,
        file_rows_seen=int(stats["rows"]),
        distinct_view_ids=int(stats["distinct"]),
        orphan_view_ids=orphan_ids,
        orphan_view_ids_reclaimed=reclaimed,
        file_rows_deleted=rows_deleted,
        file_rows_surviving=rows_surviving,
        in_grace_view_ids=in_grace,
        reclaims_remaining=remaining,
        dry_run=dry_run,
        detail=scan_error,
    )


# ---------------------------------------------------------------------------
# Pin protection (feeds maintain_lake's tag-before-cleanup step)
# ---------------------------------------------------------------------------


def view_retention_pin_details(
    lake: Lake, *, max_views: int = _MAX_PIN_VIEWS
) -> dict[str, dict[int, dict[str, Any]]]:
    """Table-version pins implied by non-retired published views.

    Same ``{table: {version: detail}}`` shape as
    ``lineage.snapshot_retention_pin_details``, merged by ``maintain_lake``
    into the pin map its tag-before-cleanup step protects -- so version
    pruning can never eat a version a live published view pins. Streams the
    header catalog with a light projection; memory is bounded by distinct
    ``(table, version)`` pairs (views over an unchanged lake share pins).

    Past ``max_views`` this **raises** rather than degrades: silently skipping
    pins would let cleanup prune versions live views still need -- the exact
    data-loss shape this exists to prevent. The remediation is retiring views
    (0508 policy) or raising the bound deliberately.
    """
    from lancedb_robotics.lineage import _empty_pin_detail, _merge_pin_detail

    try:
        table = lake.table(VIEWS_TABLE)
    except Exception:  # noqa: BLE001 - no view catalog: nothing to pin.
        return {}

    pins: dict[str, dict[int, dict[str, Any]]] = {}
    views_seen = 0
    previous_view: str | None = None
    for batch in (
        table.search()
        .select(["view_id", "repo_id", "table_versions"])
        .to_batches(batch_size=_SCAN_BATCH_SIZE)
    ):
        for row in batch.to_pylist():
            view_id = str(row["view_id"])
            if view_id == previous_view:
                continue  # cheap adjacent-duplicate skip; full dedup not needed.
            previous_view = view_id
            views_seen += 1
            if views_seen > max_views:
                raise ViewRetirementError(
                    f"more than {max_views} published views hold version pins; "
                    "refusing to build an unbounded pin map. Retire old views "
                    "(`train view retire`, or `lake maintain "
                    "--apply-lerobot-view-retention`) or raise the bound "
                    "deliberately via maintain_lake(lerobot_view_pin_max_views=...)"
                )
            repo_id = str(row.get("repo_id") or "")
            for entry in row.get("table_versions") or ():
                pinned_table = str(entry.get("table") or "")
                version = entry.get("version")
                if not pinned_table or version is None:
                    continue
                detail = _empty_pin_detail()
                detail["reasons"].add(f"lerobot-view:{repo_id}")
                detail["categories"].update({"lerobot-view", "training-reproducibility"})
                detail["artifact_ids"].add(view_id)
                pins.setdefault(pinned_table, {}).setdefault(int(version), _empty_pin_detail())
                _merge_pin_detail(pins[pinned_table][int(version)], detail)
    return pins


# ---------------------------------------------------------------------------
# Backend conformance + readiness (0143 shape)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewPinConformance:
    """Whether pinned published-view opens work on the resolved backend."""

    backend_kind: str
    data_plane: str
    status: str
    capability: str
    advertised: bool
    namespace_managed_versioning: bool
    fallbacks: tuple[str, ...]
    reason: str | None
    suggested_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind,
            "data_plane": self.data_plane,
            "status": self.status,
            "capability": self.capability,
            "advertised": self.advertised,
            "namespace_managed_versioning": self.namespace_managed_versioning,
            "fallbacks": list(self.fallbacks),
            "reason": self.reason,
            "suggested_action": self.suggested_action,
        }


def view_pin_conformance(lake: Lake) -> ViewPinConformance:
    """Classify pinned published-view open support for ``lake``'s backend.

    ``supported`` -- the backend advertises ``table_versioning`` and does not
    delegate version lifecycle to a namespace: ``PinnedLake.table`` checkout
    works and ``lake maintain`` can tag/protect the pinned versions.
    ``capability-gated`` -- the backend does not advertise it (a ``db://``
    remote DB by default): every pinned open raises the typed
    ``StaleViewVersionError`` rather than silently serving the live version
    (0116 invariant), and the remediation is a direct-IO plane.
    ``unavailable`` -- version lifecycle is namespace-managed: the SDK cannot
    protect a pinned version with its own tags, so pinned opens depend on the
    namespace retaining those versions.
    """
    spec = getattr(lake, "connection_spec", None)
    if spec is None:
        return ViewPinConformance(
            backend_kind="unclassified",
            data_plane="unclassified",
            status=VIEW_PIN_SUPPORTED,
            capability=_CAPABILITY,
            advertised=True,
            namespace_managed_versioning=False,
            fallbacks=(),
            reason=None,
            suggested_action="none; pinned opens run in-process against the dataset",
        )

    capabilities = spec.capabilities
    managed = bool(getattr(capabilities, "namespace_managed_versioning", False))
    advertised = backend_supports(spec, VERSIONING)
    fallbacks = ("object_store_lancedb_oss", "pylance_direct_namespace")

    if managed:
        return ViewPinConformance(
            backend_kind=spec.kind,
            data_plane=spec.data_plane,
            status=VIEW_PIN_UNAVAILABLE,
            capability=_CAPABILITY,
            advertised=advertised,
            namespace_managed_versioning=True,
            fallbacks=(),
            reason=(
                "namespace manages table versioning; the SDK cannot pin or "
                "protect the historical versions a published view records, so "
                "pinned opens depend on the namespace retaining and serving them"
            ),
            suggested_action=(
                "ask the namespace to retain the view-pinned table versions, or "
                "publish views against an object-store copy that owns its own "
                "version history"
            ),
        )
    if advertised:
        return ViewPinConformance(
            backend_kind=spec.kind,
            data_plane=spec.data_plane,
            status=VIEW_PIN_SUPPORTED,
            capability=_CAPABILITY,
            advertised=True,
            namespace_managed_versioning=False,
            fallbacks=(),
            reason=None,
            suggested_action="none; run `lake maintain` to keep view pins protected",
        )
    return ViewPinConformance(
        backend_kind=spec.kind,
        data_plane=spec.data_plane,
        status=VIEW_PIN_CAPABILITY_GATED,
        capability=_CAPABILITY,
        advertised=False,
        namespace_managed_versioning=False,
        fallbacks=fallbacks,
        reason=(
            f"backend {spec.kind!r} does not advertise the {_CAPABILITY!r} "
            "capability required to open a published view at its pinned versions"
        ),
        suggested_action=(
            "point the data plane at " + " or ".join(fallbacks) + " for pinned "
            "opens, or advertise the capability via "
            f"remote_capabilities={{{_CAPABILITY!r}: True}} if the deployment supports it"
        ),
    )


@dataclass(frozen=True)
class ViewPinStatus:
    """Protection/readability of one view-pinned ``(table, version)``."""

    table: str
    version: int
    status: str
    on_disk: bool
    tagged: bool
    readable: bool | None
    current_version: int | None
    view_count: int
    view_sample: tuple[str, ...]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "version": self.version,
            "status": self.status,
            "on_disk": self.on_disk,
            "tagged": self.tagged,
            "readable": self.readable,
            "current_version": self.current_version,
            "view_count": self.view_count,
            "view_sample": list(self.view_sample),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ViewReadinessReport:
    """Openability of every published view's pinned table versions."""

    schema_version: str
    lake_uri: str
    backend: ViewPinConformance
    views_checked: int
    views_remaining: int
    views_ready: int
    views_at_risk: int
    pins: tuple[ViewPinStatus, ...]
    at_risk_views: tuple[dict[str, Any], ...]
    status: str
    ready: bool
    suggested_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lake_uri": self.lake_uri,
            "backend": self.backend.to_dict(),
            "views_checked": self.views_checked,
            "views_remaining": self.views_remaining,
            "views_ready": self.views_ready,
            "views_at_risk": self.views_at_risk,
            "pins": [pin.to_dict() for pin in self.pins],
            "at_risk_pins": [
                pin.to_dict()
                for pin in self.pins
                if pin.status in (PIN_PRUNED, PIN_UNREADABLE, PIN_UNPROTECTED)
            ],
            "at_risk_views": [dict(item) for item in self.at_risk_views],
            "status": self.status,
            "ready": self.ready,
            "suggested_actions": list(self.suggested_actions),
        }


def view_readiness(
    lake: Lake,
    *,
    check_readability: bool = True,
    max_views: int = _MAX_READINESS_VIEWS,
    max_checkout_probes: int = _MAX_CHECKOUT_PROBES,
) -> ViewReadinessReport:
    """Report whether every published view can still open at its pinned versions.

    The 0143 ``curation_replay_readiness`` shape applied to the view catalog:
    each distinct pinned ``(table, version)`` is classified ``protected``
    (managed pin tag, or current), ``unprotected`` (present but nothing stops a
    later cleanup pruning it -- run ``lake maintain``), ``pruned`` (gone: every
    open of the referencing views raises ``StaleViewVersionError``; re-publish
    them), or ``unreadable`` (present but checkout fails). On a
    ``capability-gated``/``unavailable`` backend pins are ``backend-gated`` and
    the verdict never falls through to ``ready`` (readiness never checked is
    not readiness -- SKILLS.md silent-degrade rule). Bounded: at most
    ``max_views`` views and ``max_checkout_probes`` checkout probes per run,
    the remainder reported.
    """
    # Deliberate reuse of the 0143 per-table meta + checkout probe helpers --
    # they are backend mechanics, not curation semantics. A unit test pins the
    # import so a refactor there fails loudly here.
    from lancedb_robotics.curation_replay_retention import (
        _checkout_readable,
        _table_version_meta,
    )

    conformance = view_pin_conformance(lake)
    spec = getattr(lake, "connection_spec", None)
    can_inspect = backend_supports(spec, DIRECT_LANCE)
    can_read = check_readability and can_inspect and conformance.status == VIEW_PIN_SUPPORTED

    # One streaming pass over the headers: per-view pin keys (bounded by
    # max_views) and the deduped (table, version) -> referencing-views map.
    views: list[tuple[str, str, tuple[tuple[str, int], ...]]] = []
    pin_views: dict[tuple[str, int], list[str]] = {}
    views_remaining = 0
    seen: set[str] = set()
    try:
        header_table = lake.table(VIEWS_TABLE)
    except Exception:  # noqa: BLE001 - no view catalog: trivially ready.
        header_table = None
    if header_table is not None:
        for batch in (
            header_table.search()
            .select(["view_id", "repo_id", "table_versions"])
            .to_batches(batch_size=_SCAN_BATCH_SIZE)
        ):
            for row in batch.to_pylist():
                view_id = str(row["view_id"])
                if view_id in seen:
                    continue  # physical duplicate copies count once.
                if len(seen) >= max_views:
                    views_remaining += 1
                    continue
                seen.add(view_id)
                keys: list[tuple[str, int]] = []
                for entry in row.get("table_versions") or ():
                    table = str(entry.get("table") or "")
                    version = entry.get("version")
                    if not table or version is None:
                        continue
                    key = (table, int(version))
                    keys.append(key)
                    holders = pin_views.setdefault(key, [])
                    holders.append(view_id)
                views.append((view_id, str(row.get("repo_id") or ""), tuple(keys)))

    statuses: dict[tuple[str, int], ViewPinStatus] = {}
    tables = sorted({table for (table, _version) in pin_views})
    per_table = {table: _table_version_meta(lake, table) for table in tables} if can_inspect else {}
    probes_used = 0

    for (table, version), holders in sorted(pin_views.items()):
        sample = tuple(sorted(holders)[:_PIN_VIEW_SAMPLE])
        if not can_inspect:
            statuses[(table, version)] = ViewPinStatus(
                table=table,
                version=version,
                status=PIN_BACKEND_GATED,
                on_disk=False,
                tagged=False,
                readable=None,
                current_version=None,
                view_count=len(holders),
                view_sample=sample,
                detail=(
                    "backend cannot inspect on-disk versions; "
                    f"{conformance.suggested_action}"
                ),
            )
            continue
        meta = per_table[table]
        on_disk = meta.available is None or version in meta.available
        tagged = f"{_PIN_TAG_PREFIX}{version}" in meta.tags
        current = meta.current
        readable: bool | None = None
        if not on_disk:
            status = PIN_PRUNED
            detail = (
                f"view-pinned {table}@{version} is no longer on disk; every open "
                "of the referencing views raises StaleViewVersionError -- "
                "re-publish them against the current lake"
            )
        else:
            protected = tagged or (current is not None and version == current)
            if can_read and probes_used < max_checkout_probes:
                probes_used += 1
                readable = _checkout_readable(lake, table, version)
            if readable is False:
                status = PIN_UNREADABLE
                detail = f"{table}@{version} is present but cannot be checked out on this backend"
            elif protected:
                status = PIN_PROTECTED
                detail = (
                    f"{table}@{version} is protected by a managed pin tag"
                    if tagged
                    else f"{table}@{version} is the current version"
                )
            else:
                status = PIN_UNPROTECTED
                detail = (
                    f"{table}@{version} is present but untagged; a version cleanup "
                    "could prune it -- run `lake maintain` to tag view pins"
                )
        statuses[(table, version)] = ViewPinStatus(
            table=table,
            version=version,
            status=status,
            on_disk=on_disk,
            tagged=tagged,
            readable=readable,
            current_version=current,
            view_count=len(holders),
            view_sample=sample,
            detail=detail,
        )

    at_risk_states = (PIN_PRUNED, PIN_UNREADABLE, PIN_UNPROTECTED)
    views_ready = 0
    at_risk_views: list[dict[str, Any]] = []
    views_at_risk = 0
    for view_id, repo_id, keys in views:
        bad = [
            f"{table}@{version}:{statuses[(table, version)].status}"
            for (table, version) in keys
            if statuses[(table, version)].status in at_risk_states
        ]
        if bad:
            views_at_risk += 1
            if len(at_risk_views) < _MAX_AT_RISK_LISTED:
                at_risk_views.append(
                    {"view_id": view_id, "repo_id": repo_id, "at_risk_pins": bad}
                )
        else:
            views_ready += 1

    suggested: list[str] = []
    if conformance.status != VIEW_PIN_SUPPORTED or not can_inspect:
        # Same rule as 0143: pins that were never verified must not report
        # ready -- a backend can advertise versioning yet gate direct IO.
        overall = BACKEND_GATED
        ready = False
        if conformance.status != VIEW_PIN_SUPPORTED:
            suggested.append(conformance.suggested_action)
        else:
            suggested.append(
                f"backend {conformance.backend_kind!r} advertises versioning but "
                "not direct object IO, so view pins cannot be verified here; "
                "verify from object_store_lancedb_oss or pylance_direct_namespace"
            )
    elif views_at_risk or views_remaining:
        overall = AT_RISK
        ready = False
        if any(status.status == PIN_PRUNED for status in statuses.values()):
            suggested.append(
                "a view-pinned table version has been pruned; re-publish the "
                "affected views (their manifests can no longer be served)"
            )
        if any(status.status == PIN_UNPROTECTED for status in statuses.values()):
            suggested.append(
                "unprotected view pins exist; run `lake maintain` (with published-"
                "view pin protection on) before any version cleanup"
            )
        if views_remaining:
            suggested.append(
                f"{views_remaining} view(s) were not checked (over the "
                f"{max_views}-view bound); retire old views or raise max_views"
            )
    else:
        overall = READY
        ready = True

    return ViewReadinessReport(
        schema_version=READINESS_SCHEMA_VERSION,
        lake_uri=lake.uri,
        backend=conformance,
        views_checked=len(views),
        views_remaining=views_remaining,
        views_ready=views_ready,
        views_at_risk=views_at_risk,
        pins=tuple(statuses[key] for key in sorted(statuses)),
        at_risk_views=tuple(at_risk_views),
        status=overall,
        ready=ready,
        suggested_actions=tuple(suggested),
    )
