"""Scalable curation row-plan catalog and chunked targets (backlog 0146).

Backlog 0084 compiles row-grain membership decisions into a zero-copy row plan
and can freeze it as a ``lineage_artifacts`` row. That shape is fine for a
preview over a handful of scenarios, but it carried the *entire* target-id list
three ways at once: in the returned dataclass, in the ``transform_runs`` params
JSON (via the compile report), and in the frozen artifact's ``row_ids``
``list<string>`` cell. A robotics program compiling millions of observation,
aligned-frame, or episode targets per branch cannot use any of those three.

This module is the scalable storage and query layer underneath that compiler:

- ``curation_row_plans`` is the manifest/header: one row per compiled plan with
  every identity and count promoted into a dedicated, scalar-indexable column
  (plan, view, grain, source snapshot, transform lineage, ``created_at``,
  lifecycle state). ``summary_json`` is a deliberately *bounded* summary --
  counts plus capped conflict / rejected / label-intent samples -- so a plan
  header never grows with the target count.
- ``curation_row_plan_chunks`` holds the ordered target membership in bounded
  chunks (``target_ids`` plus the resolved ``lance_row_ids``), written in bounded
  batches so a multi-million-target plan never stages one oversized commit
  (BUG-02) and read back in stable ``start_ordinal`` order without ever
  materializing the whole membership in Python (BUG-06).
- The compact ``lineage_artifacts`` row stays the canonical lineage handle. Below
  :data:`INLINE_TARGET_LIMIT` the artifact keeps its inline ``row_ids`` exactly as
  0084 wrote them; above it, the artifact carries a *pointer* to chunked storage
  in its metadata instead of an unbounded id cell.
- Retention (:func:`prune_row_plans`) compacts superseded **dry-run** plans only.
  A frozen plan, or any plan referenced by a training run or a training report,
  is protected and never pruned. Pruning is safe-delete: the bounded summary and
  every promoted column survive as audit evidence; only the chunk rows and the
  summary body are cleared.

Every read here follows the repo's bounded-streaming discipline (SKILLS.md §2):
a required projection, batched ``to_batches`` folds, engine ``order_by`` keyset
pagination with a bounded top-``page_size+1`` heap fallback, and never a
``to_arrow().to_pylist()`` over an unbounded scan. Writes are atomic
``merge_insert`` upserts with a bounded commit-conflict retry (BUG-04), ordered
chunks-first / header-last so a crash mid-write leaves an unpublished plan rather
than a header promising rows that were never written (backlog 0141).
"""

from __future__ import annotations

import base64
import hashlib
import heapq
import json
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime, timedelta
from itertools import islice
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from lancedb_robotics.schemas import (
    CURATION_ROW_PLAN_CHUNKS_SCHEMA,
    CURATION_ROW_PLANS_SCHEMA,
)

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from lancedb_robotics.lake import Lake

ROW_PLAN_CATALOG_SCHEMA_VERSION = "lancedb-robotics/curation-row-plan-catalog/v1"

_PLAN_TABLE = "curation_row_plans"
_CHUNK_TABLE = "curation_row_plan_chunks"

#: Tables whose ``row_plan_id`` column pins a compiled plan as training or
#: evaluation evidence. A plan named by any of these is never prunable.
REFERENCING_TABLES: tuple[tuple[str, str], ...] = (
    ("training_runs", "row_plan_id"),
    ("training_reports", "row_plan_id"),
)

#: Plan lifecycle states.
STATE_ACTIVE = "active"
STATE_SUPERSEDED = "superseded"
STATE_PRUNED = "pruned"
STATES: tuple[str, ...] = (STATE_ACTIVE, STATE_SUPERSEDED, STATE_PRUNED)

#: At or below this target count the plan's ids stay inline on the compact
#: lineage handle (0084 behaviour, byte-identical). Above it the frozen artifact
#: carries a pointer to chunked storage instead of an unbounded ``row_ids`` cell.
#: Matches ``curate._VIEW_INLINE_SCENARIO_ID_LIMIT`` (backlog 0081).
INLINE_TARGET_LIMIT = 1024

#: Targets per chunk row. Wide enough that a million-target plan is ~244 chunks,
#: narrow enough that one chunk cell stays small.
CHUNK_SIZE = 4096

#: Above this target count the compiled result stops materializing the parallel
#: ``target_ids`` / ``lance_row_ids`` tuples in the returned dataclass and the
#: compile report; callers page through :class:`CurationRowPlanTargets` instead.
#: Mirrors ``curate._VIEW_MEMBERSHIP_MATERIALIZE_SOFT_LIMIT`` (backlog 0141).
MATERIALIZE_SOFT_LIMIT = 50_000

#: How long a chunk row with no plan header is presumed to belong to an in-flight
#: compile rather than to a crashed one. Chunks are written before the header, so a
#: large compile spends real time in that state; reclaiming inside this window
#: would delete rows a live writer is about to publish a header for.
ORPHAN_GRACE = timedelta(hours=6)

#: Cap on each diagnostic sample list carried in ``summary_json`` / the compile
#: report. Counts are always exact; the samples are truncated with an explicit
#: ``*_truncated`` flag so a reader is never silently shown a partial list.
SAMPLE_LIMIT = 200

#: Deterministic page ordering for plan listing.
HISTORY_ORDER_COLUMNS: tuple[str, str] = ("created_at", "plan_id")

DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 10_000
DEFAULT_TARGET_PAGE_SIZE = 1000
MAX_TARGET_PAGE_SIZE = 100_000

_SCAN_BATCH = 2048
#: Bounded per-commit chunk batch so a 10^6-target plan never stages one
#: oversized commit (BUG-02).
_CHUNK_WRITE_BATCH = 32
#: Chunk rows fetched per windowed read when paging targets.
_CHUNK_SCAN_BATCH = 32
_MERGE_INSERT_ATTEMPTS = 3
#: Chunk width for ``<id> IN (...)`` reads/deletes (BUG-13 frontier width).
_IN_CHUNK = 512
_RETENTION_FLUSH = 512

#: Every header column, in schema order.
_PLAN_ROW_KEYS: tuple[str, ...] = tuple(f.name for f in CURATION_ROW_PLANS_SCHEMA)

#: Header columns that carry a JSON body or a nested/lifecycle payload. Anything
#: listed here is excluded from the light projections below, so "never pull a body
#: on a scan that visits every plan" is enforced by subtraction rather than by a
#: human keeping two 36-line lists in step.
_PLAN_BODY_COLUMNS: frozenset[str] = frozenset(
    {
        "summary_json",
        "table_versions",
        "target_fragment_ids",
        "superseded_at",
        "pruned_at",
        "retention_policy_json",
    }
)

#: Light projection for plan listing / retention -- never includes a JSON body.
PLAN_SUMMARY_COLUMNS: tuple[str, ...] = tuple(
    name for name in _PLAN_ROW_KEYS if name not in _PLAN_BODY_COLUMNS
)

#: Columns read for the retention *scan* (adds lifecycle metadata to the summary
#: set). Stays JSON-free and narrow: the scan visits every non-pruned plan, so it
#: must never pull a body. The state flip re-reads the full rows for its bounded
#: flush batch (see :func:`_apply_state`) rather than rewriting a partial row.
_RETENTION_COLUMNS: tuple[str, ...] = PLAN_SUMMARY_COLUMNS + (
    "superseded_at",
    "pruned_at",
    "retention_policy_json",
)

#: Chunk projection for target paging -- never the digest/lineage columns.
_CHUNK_READ_COLUMNS: tuple[str, ...] = (
    "plan_id",
    "chunk_index",
    "start_ordinal",
    "end_ordinal",
    "target_ids",
    "lance_row_ids",
    "target_count",
)


class RowPlanCatalogError(Exception):
    """Raised when a row plan cannot be written, read, or pruned as requested."""


# --------------------------------------------------------------------------- #
# Small shared helpers (mirrors materialization_rollups / curate conventions).
# --------------------------------------------------------------------------- #


def _digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _is_retryable_commit_conflict(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "commit conflict" in text or "retryable" in text or "concurrent" in text


def _merge_insert_with_retry(
    table: Any,
    key_column: str,
    data: pa.Table,
    *,
    update_matched: bool,
) -> None:
    """Single-commit upsert (BUG-04 shape) with bounded commit-conflict retry.

    ``update_matched=False`` is insert-only, so two writers persisting the same
    content-addressed chunk rows converge without duplicating; ``True`` replaces
    the matched row in one commit (header publish, lifecycle flips).
    """
    if data.num_rows == 0:
        return
    last_error: BaseException | None = None
    for _ in range(_MERGE_INSERT_ATTEMPTS):
        builder = table.merge_insert(key_column)
        if update_matched:
            builder = builder.when_matched_update_all()
        try:
            builder.when_not_matched_insert_all().execute(data)
            return
        except Exception as exc:  # noqa: BLE001 - retry only the retryable conflict.
            if not _is_retryable_commit_conflict(exc):
                raise
            last_error = exc
    raise RowPlanCatalogError(
        f"merge_insert on {key_column!r} kept hitting commit conflicts "
        f"after {_MERGE_INSERT_ATTEMPTS} attempts: {last_error}"
    )


def _record_batch(stats: Any, table: str, count: int) -> None:
    if stats is not None and hasattr(stats, "record_batch"):
        stats.record_batch(table, count)


def _record_full_scan(stats: Any, table: str, count: int) -> None:
    if stats is not None and hasattr(stats, "record_full_scan"):
        stats.record_full_scan(table, count)


def _stream_rows(
    lake: Lake,
    table: str,
    *,
    columns: Sequence[str] | None = None,
    where_sql: str | None = None,
    batch_size: int = _SCAN_BATCH,
    stats: Any = None,
) -> Iterable[list[dict[str, Any]]]:
    """Yield bounded row batches with projection + filter pushdown (SKILLS.md §2).

    **At-most-once.** The materialized fallback fires only while *nothing has been
    yielded yet* — either the query could not be built, or it failed on its very
    first batch. Once a row has been handed to the caller, a later read error
    propagates instead of restarting the scan from the top.

    That guarantee is load-bearing, not defensive. Three consumers here are
    order- or count-sensitive: :func:`_iter_plans_for_retention` (a re-yielded row
    shifts the series rank, so retention prunes the wrong plan),
    :func:`_read_plans_bounded_heap` (double-counts ``qualifying``, so ``has_more``
    lies), and :func:`_iter_chunk_rows` (duplicates a target ordinal, breaking the
    plan's ordered-target contract and its digest). Re-reading under them silently
    corrupts the answer rather than failing.

    ``materialization_rollups._stream_rows`` still carries the restart-on-error
    shape this function deliberately does not.
    """
    handle = lake.table(table)
    available = set(handle.schema.names)
    projected = [column for column in (columns or ()) if column in available]
    batches = None
    try:
        query = handle.search()
        if projected:
            query = query.select(projected)
        if where_sql:
            query = query.where(where_sql)
        batches = query.to_batches(batch_size=batch_size)
    except Exception:
        batches = None
    if batches is not None:
        yielded = False
        try:
            for batch in batches:
                rows = batch.to_pylist()
                _record_batch(stats, table, len(rows))
                yielded = True
                yield rows
            return
        except Exception:
            if yielded:
                # Mid-stream failure: the caller already has rows, so restarting
                # would deliver some of them twice. Fail instead.
                raise
    handle_rows = handle.to_arrow()
    if projected:
        keep = [name for name in projected if name in handle_rows.schema.names]
        handle_rows = handle_rows.select(keep)
    rows = handle_rows.to_pylist()
    _record_full_scan(stats, table, len(rows))
    yield rows


def _chunked(seq: Sequence[str], size: int = _IN_CHUNK) -> Iterable[list[str]]:
    items = list(seq)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _in_clause(column: str, ids: Sequence[str]) -> str:
    return f"{column} IN (" + ", ".join(_sql_literal(x) for x in ids) + ")"


# --------------------------------------------------------------------------- #
# Cursor codecs.
# --------------------------------------------------------------------------- #


def _encode_plan_cursor(created_at: datetime | None, plan_id: str) -> str:
    payload = {
        "created_at": created_at.isoformat() if created_at else "",
        "plan_id": plan_id,
    }
    return base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True).encode()).decode()


def _decode_plan_cursor(token: str) -> tuple[datetime | None, str]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
        created = payload.get("created_at") or ""
        return (_coerce_dt(created) if created else None, str(payload["plan_id"]))
    except Exception as exc:  # noqa: BLE001 - a bad cursor is a caller error.
        raise RowPlanCatalogError(f"invalid row-plan cursor: {token!r}") from exc


def encode_target_cursor(start_ordinal: int) -> str:
    """Opaque, stable cursor for the next target page (ordinal keyset)."""
    raw = json.dumps({"start_ordinal": int(start_ordinal)}, sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_target_cursor(token: str) -> int:
    try:
        payload = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
        return int(payload["start_ordinal"])
    except Exception as exc:  # noqa: BLE001 - a bad cursor is a caller error.
        raise RowPlanCatalogError(f"invalid row-plan target cursor: {token!r}") from exc


def _plan_row_key(row: dict[str, Any]) -> tuple[datetime, str]:
    created = _coerce_dt(row.get("created_at")) or datetime.min.replace(tzinfo=UTC)
    return (created, str(row.get("plan_id") or ""))


# --------------------------------------------------------------------------- #
# Public value types.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CurationRowPlanEntry:
    """One compiled row plan's promoted header columns (no id lists, no body)."""

    plan_id: str
    view_id: str
    view_name: str
    target_grain: str
    target_table: str
    source_snapshot_name: str
    base_policy: str
    conflict_policy: str
    storage_kind: str
    chunk_table: str
    chunk_size: int
    chunk_count: int
    target_count: int
    candidate_count: int
    selected_count: int
    rejected_count: int
    conflict_count: int
    label_intent_count: int
    scenario_count: int
    membership_transform_count: int
    superseded_membership_count: int
    lance_row_ids_present: bool
    target_ids_digest: str
    plan_digest: str
    artifact_id: str
    frozen: bool
    metadata_only: bool
    payload_copy_policy: str
    copied_payload_bytes: int
    transform_id: str
    source_view_transform_id: str
    source_snapshot_transform_id: str
    state: str
    summary_available: bool
    created_by: str
    created_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        # Every field is a str/int/bool except ``created_at``, so this is exactly the
        # hand-written mapping it replaces -- and a new header column no longer needs
        # a matching edit here.
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        payload["created_at"] = self.created_at.isoformat() if self.created_at else ""
        return payload


@dataclass(frozen=True)
class CurationRowPlanPage:
    """One bounded, ordered page of row-plan headers."""

    records: tuple[CurationRowPlanEntry, ...]
    page_size: int
    page_index: int
    cursor: str | None
    next_cursor: str | None
    has_more: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "record_count": len(self.records),
            "page_size": self.page_size,
            "page_index": self.page_index,
            "cursor": self.cursor or "",
            "next_cursor": self.next_cursor or "",
            "has_more": self.has_more,
            "order": list(HISTORY_ORDER_COLUMNS),
        }


@dataclass(frozen=True)
class CurationRowPlanTarget:
    """One plan target at a stable ordinal, with its resolved Lance row id."""

    ordinal: int
    target_id: str
    lance_row_id: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "target_id": self.target_id,
            "lance_row_id": self.lance_row_id,
        }


@dataclass(frozen=True)
class CurationRowPlanTargetPage:
    """One bounded page of plan targets in ascending ordinal order."""

    plan_id: str
    targets: tuple[CurationRowPlanTarget, ...]
    page_size: int
    start_ordinal: int
    next_cursor: str | None
    has_more: bool
    target_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "targets": [target.to_dict() for target in self.targets],
            "target_ids": [target.target_id for target in self.targets],
            "lance_row_ids": [target.lance_row_id for target in self.targets],
            "returned": len(self.targets),
            "page_size": self.page_size,
            "start_ordinal": self.start_ordinal,
            "next_cursor": self.next_cursor or "",
            "has_more": self.has_more,
            "target_count": self.target_count,
            "order": "start_ordinal",
        }


@dataclass(frozen=True)
class CurationRowPlanTargets:
    """Lazy, ordered view over a compiled plan's targets.

    Holds only the plan header's storage descriptor; iteration and paging read
    bounded windows of ``curation_row_plan_chunks`` (or replay the inline ids for
    a small plan) so a multi-million-target plan is never materialized in Python.
    """

    lake: Lake
    plan_id: str
    target_grain: str
    target_table: str
    storage_kind: str
    target_count: int
    chunk_size: int
    chunk_count: int
    inline_targets: tuple[tuple[str, int | None], ...] | None = None
    page_size: int = DEFAULT_TARGET_PAGE_SIZE
    #: ``False`` when the target table has moved past the version the plan pinned,
    #: so the stored ``lance_row_ids`` (fragment addresses) may have been remapped
    #: by compaction. The ``target_id`` values are always valid; a consumer must
    #: resolve by id rather than ``take_row_ids`` when this is ``False``.
    row_ids_trustworthy: bool = True
    #: Version the plan pinned for ``target_table``, and the table's version now.
    pinned_table_version: int | None = None
    current_table_version: int | None = None

    def __len__(self) -> int:
        return int(self.target_count)

    def iter_targets(
        self,
        *,
        start_ordinal: int = 0,
        stats: Any = None,
    ) -> Iterator[CurationRowPlanTarget]:
        """Stream targets in ascending ordinal order from ``start_ordinal``."""
        if self.storage_kind == "inline":
            for ordinal, (target_id, row_id) in enumerate(self.inline_targets or ()):
                if ordinal < start_ordinal:
                    continue
                yield CurationRowPlanTarget(ordinal, target_id, row_id)
            return
        yield from _iter_chunked_targets(
            self.lake,
            self.plan_id,
            target_count=self.target_count,
            chunk_size=self.chunk_size,
            start_ordinal=start_ordinal,
            stats=stats,
        )

    def page(
        self,
        *,
        page_size: int | None = None,
        cursor: str | None = None,
        stats: Any = None,
    ) -> CurationRowPlanTargetPage:
        """One bounded page of targets; ``next_cursor`` resumes deterministically."""
        limit = _normalize_page_size(page_size or self.page_size, DEFAULT_TARGET_PAGE_SIZE, MAX_TARGET_PAGE_SIZE)
        start = decode_target_cursor(cursor) if cursor else 0
        if start < 0:
            raise RowPlanCatalogError(f"row-plan target cursor ordinal must be >= 0, got {start}")
        collected: list[CurationRowPlanTarget] = []
        for target in self.iter_targets(start_ordinal=start, stats=stats):
            collected.append(target)
            if len(collected) >= limit:
                break
        next_ordinal = start + len(collected)
        has_more = next_ordinal < int(self.target_count)
        return CurationRowPlanTargetPage(
            plan_id=self.plan_id,
            targets=tuple(collected),
            page_size=limit,
            start_ordinal=start,
            next_cursor=encode_target_cursor(next_ordinal) if has_more else None,
            has_more=has_more,
            target_count=int(self.target_count),
        )

    def iter_pages(
        self,
        *,
        page_size: int | None = None,
        cursor: str | None = None,
        stats: Any = None,
    ) -> Iterator[CurationRowPlanTargetPage]:
        """Walk every page from ``cursor`` forward in a **single** pass.

        Deliberately not a loop over :meth:`page`. Each ``page`` call restarts
        ``iter_targets``, and the chunk reader fetches a whole ordinal *window*
        (``chunk_size * _CHUNK_SCAN_BATCH`` targets) per restart -- so paging a
        large plan that way re-reads and re-sorts the same window once per page.
        Holding one iterator and slicing it keeps a full walk at one pass over the
        chunk table. :meth:`page` stays the stateless, cursor-addressable entry
        point for callers that need to resume across processes.
        """
        limit = _normalize_page_size(
            page_size or self.page_size, DEFAULT_TARGET_PAGE_SIZE, MAX_TARGET_PAGE_SIZE
        )
        start = decode_target_cursor(cursor) if cursor else 0
        if start < 0:
            raise RowPlanCatalogError(
                f"row-plan target cursor ordinal must be >= 0, got {start}"
            )
        total = int(self.target_count)
        stream = self.iter_targets(start_ordinal=start, stats=stats)
        ordinal = start
        while True:
            collected = tuple(islice(stream, limit))
            next_ordinal = ordinal + len(collected)
            has_more = next_ordinal < total
            yield CurationRowPlanTargetPage(
                plan_id=self.plan_id,
                targets=collected,
                page_size=limit,
                start_ordinal=ordinal,
                next_cursor=encode_target_cursor(next_ordinal) if has_more else None,
                has_more=has_more,
                target_count=total,
            )
            if not has_more:
                return
            ordinal = next_ordinal

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "target_grain": self.target_grain,
            "target_table": self.target_table,
            "storage_kind": self.storage_kind,
            "target_count": int(self.target_count),
            "chunk_size": int(self.chunk_size),
            "chunk_count": int(self.chunk_count),
            "page_size": int(self.page_size),
            "order": "start_ordinal",
            "row_ids_trustworthy": bool(self.row_ids_trustworthy),
            "pinned_table_version": self.pinned_table_version,
            "current_table_version": self.current_table_version,
        }


@dataclass(frozen=True)
class RowPlanRetentionReport:
    """Outcome of a row-plan retention/compaction pass."""

    dry_run: bool
    retain_latest: int
    older_than: datetime | None
    scanned_count: int
    protected_count: int
    superseded_plan_ids: tuple[str, ...] = field(default_factory=tuple)
    pruned_plan_ids: tuple[str, ...] = field(default_factory=tuple)
    protected_plan_ids: tuple[str, ...] = field(default_factory=tuple)
    #: Plans whose chunk delete failed or could not be verified. Their headers are
    #: deliberately left untouched so a later pass retries them -- but they appear in
    #: neither ``pruned`` nor ``protected``, so they are reported explicitly rather
    #: than vanishing from the report while the command exits 0.
    delete_failed_plan_ids: tuple[str, ...] = field(default_factory=tuple)
    chunks_deleted: int = 0
    #: Bytes of ``summary_json`` actually cleared by this pass. Named for what it
    #: measures: an earlier ``before``/``after`` pair reported "after" as a constant
    #: 0, which reads like a measurement and would tell an operator either that
    #: nothing was reclaimed or that the accounting is broken.
    summary_bytes_cleared: int = 0

    @property
    def superseded_count(self) -> int:
        return len(self.superseded_plan_ids)

    @property
    def pruned_count(self) -> int:
        return len(self.pruned_plan_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "retain_latest": self.retain_latest,
            "older_than": self.older_than.isoformat() if self.older_than else "",
            "scanned_count": self.scanned_count,
            "protected_count": self.protected_count,
            "superseded_count": self.superseded_count,
            "pruned_count": self.pruned_count,
            "superseded_plan_ids": list(self.superseded_plan_ids),
            "pruned_plan_ids": list(self.pruned_plan_ids),
            "protected_plan_ids": list(self.protected_plan_ids),
            "delete_failed_plan_ids": list(self.delete_failed_plan_ids),
            "delete_failed_count": len(self.delete_failed_plan_ids),
            "chunks_deleted": self.chunks_deleted,
            "summary_bytes_cleared": self.summary_bytes_cleared,
        }


@dataclass(frozen=True)
class RowPlanValidationIssue:
    """One structural problem found in a plan's chunked target storage."""

    plan_id: str
    code: str
    detail: str
    chunk_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "code": self.code,
            "detail": self.detail,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class RowPlanValidationReport:
    """Result of validating chunked plan storage against its header."""

    scanned_plan_ids: tuple[str, ...]
    issues: tuple[RowPlanValidationIssue, ...]
    orphan_chunk_plan_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.issues and not self.orphan_chunk_plan_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_plan_count": len(self.scanned_plan_ids),
            "scanned_plan_ids": list(self.scanned_plan_ids),
            "issues": [issue.to_dict() for issue in self.issues],
            "issue_count": len(self.issues),
            "orphan_chunk_plan_ids": list(self.orphan_chunk_plan_ids),
            "ok": self.ok,
        }


# --------------------------------------------------------------------------- #
# Digests, samples, and the storage descriptor.
# --------------------------------------------------------------------------- #


class TargetIdsDigest:
    """Streaming digest over an ordered target-id sequence.

    0084 fed the whole ``target_ids`` list into the plan digest, which meant the
    plan id could not be computed without holding every id in memory. Folding the
    ids through a streaming sha1 keeps ``plan_id`` content-addressed over exactly
    the same ordered content at O(1) memory.
    """

    def __init__(self) -> None:
        self._hasher = hashlib.sha1()
        self.count = 0

    def update(self, target_id: str) -> None:
        self._hasher.update(str(target_id).encode())
        self._hasher.update(b"\n")
        self.count += 1

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


def target_ids_digest(target_ids: Iterable[str]) -> str:
    """Digest of an ordered target-id sequence (same fold as :class:`TargetIdsDigest`)."""
    digest = TargetIdsDigest()
    for target_id in target_ids:
        digest.update(target_id)
    return digest.hexdigest()


def sample(items: Sequence[Any], *, limit: int | None = None) -> tuple[list[Any], bool]:
    """Bounded sample of ``items`` plus whether it was truncated.

    Slices rather than copying the input first: the callers feed it lists that can
    hold one entry per row-grain decision on the branch, and copying all of them to
    keep 200 defeats the point of sampling.
    """
    cap = SAMPLE_LIMIT if limit is None else int(limit)
    kept = list(islice(iter(items), cap))
    return kept, len(items) > cap


def storage_payload(
    *,
    plan_id: str,
    target_count: int,
    chunk_size: int | None = None,
    inline_target_limit: int | None = None,
    target_ids_digest: str = "",
) -> dict[str, Any]:
    """Storage descriptor for a compiled plan's target membership.

    ``inline`` keeps the ids on the compact lineage handle exactly as 0084 wrote
    them; ``chunked`` moves them into ``curation_row_plan_chunks`` and leaves the
    artifact holding a pointer.

    ``chunk_size`` / ``inline_target_limit`` default to the module tunables read
    at *call* time, so an operator (or a test) can adjust them without the value
    having been frozen into a default argument at import.
    """
    count = int(target_count)
    limit = INLINE_TARGET_LIMIT if inline_target_limit is None else int(inline_target_limit)
    chunked = count > limit
    width = max(1, CHUNK_SIZE if chunk_size is None else int(chunk_size))
    return {
        "kind": "chunked" if chunked else "inline",
        "table": _CHUNK_TABLE if chunked else "",
        "plan_id": plan_id,
        "target_count": count,
        "inline_target_count": 0 if chunked else count,
        "chunk_size": width if chunked else 0,
        "chunk_count": ((count + width - 1) // width) if chunked else 0,
        "order": "start_ordinal" if chunked else "compile-order",
        "target_ids_digest": target_ids_digest,
    }


# --------------------------------------------------------------------------- #
# Write path: chunks first (bounded batches), header last.
# --------------------------------------------------------------------------- #


def _chunk_row(
    *,
    plan_id: str,
    chunk_index: int,
    start_ordinal: int,
    target_ids: Sequence[str],
    lance_row_ids: Sequence[int | None],
    created_by: str,
    transform_id: str,
    created_at: datetime,
) -> dict[str, Any]:
    chunk_digest = _digest(
        {
            "plan_id": plan_id,
            "chunk_index": chunk_index,
            "start_ordinal": start_ordinal,
            "target_ids": list(target_ids),
        }
    )
    return {
        "chunk_id": f"rowplanchunk-{chunk_digest}",
        "plan_id": plan_id,
        "chunk_index": int(chunk_index),
        "start_ordinal": int(start_ordinal),
        "end_ordinal": int(start_ordinal) + len(target_ids) - 1,
        "target_ids": [str(item) for item in target_ids],
        "lance_row_ids": [None if item is None else int(item) for item in lance_row_ids],
        "target_count": len(target_ids),
        "chunk_digest": chunk_digest,
        "created_by": created_by,
        "transform_id": transform_id,
        "created_at": created_at,
    }


def write_row_plan_chunks(
    lake: Lake,
    plan_id: str,
    targets: Iterable[tuple[str, int | None]],
    *,
    chunk_size: int | None = None,
    created_by: str = "lancedb-robotics",
    transform_id: str = "",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Persist a plan's ordered targets as bounded chunk rows.

    Streams ``targets`` -- never materializes the full membership -- sealing a
    chunk every ``chunk_size`` targets and committing every
    ``_CHUNK_WRITE_BATCH`` chunks so a multi-million-target plan is a sequence of
    bounded commits rather than one oversized write (BUG-02). Chunk rows are
    content-addressed, so a retried write converges instead of duplicating
    (BUG-04). Returns the storage descriptor with the streamed count/digest.
    """
    width = max(1, CHUNK_SIZE if chunk_size is None else int(chunk_size))
    now = created_at or datetime.now(UTC)
    handle = lake.table(_CHUNK_TABLE)
    digest = TargetIdsDigest()
    pending_rows: list[dict[str, Any]] = []
    buffer_ids: list[str] = []
    buffer_row_ids: list[int | None] = []
    chunk_index = 0
    start_ordinal = 0

    def _flush_pending(force: bool = False) -> None:
        if not pending_rows:
            return
        if not force and len(pending_rows) < _CHUNK_WRITE_BATCH:
            return
        _merge_insert_with_retry(
            handle,
            "chunk_id",
            pa.Table.from_pylist(pending_rows, schema=CURATION_ROW_PLAN_CHUNKS_SCHEMA),
            update_matched=False,
        )
        pending_rows.clear()

    def _seal_chunk() -> None:
        nonlocal chunk_index, start_ordinal
        if not buffer_ids:
            return
        pending_rows.append(
            _chunk_row(
                plan_id=plan_id,
                chunk_index=chunk_index,
                start_ordinal=start_ordinal,
                target_ids=buffer_ids,
                lance_row_ids=buffer_row_ids,
                created_by=created_by,
                transform_id=transform_id,
                # Stamp each chunk when it is sealed, not once before the first
                # commit. Orphan reclamation uses the newest chunk's age to tell an
                # in-flight write from a crashed one; a single up-front timestamp
                # means a compile running longer than the grace window presents as
                # fully aged-out and gets reclaimed mid-write.
                created_at=now if created_at is not None else datetime.now(UTC),
            )
        )
        chunk_index += 1
        start_ordinal += len(buffer_ids)
        buffer_ids.clear()
        buffer_row_ids.clear()
        _flush_pending()

    for target_id, row_id in targets:
        digest.update(target_id)
        buffer_ids.append(str(target_id))
        buffer_row_ids.append(None if row_id is None else int(row_id))
        if len(buffer_ids) >= width:
            _seal_chunk()
    _seal_chunk()
    _flush_pending(force=True)
    return {
        "kind": "chunked",
        "table": _CHUNK_TABLE,
        "plan_id": plan_id,
        "target_count": digest.count,
        "inline_target_count": 0,
        "chunk_size": width,
        "chunk_count": chunk_index,
        "order": "start_ordinal",
        "target_ids_digest": digest.hexdigest(),
    }


def persisted_chunk_shape(lake: Lake, plan_id: str) -> tuple[int, int]:
    """``(chunk_count, target_count)`` actually stored for ``plan_id``.

    Read back through an indexed ``plan_id`` predicate with a narrow projection --
    never the id payload -- so a caller can verify a chunk write landed before
    publishing a header that promises those rows.
    """
    chunks = 0
    targets = 0
    where_sql = f"plan_id = {_sql_literal(plan_id)}"
    for batch in _stream_rows(
        lake, _CHUNK_TABLE, columns=("plan_id", "chunk_index", "target_count"), where_sql=where_sql
    ):
        for row in batch:
            if str(row.get("plan_id") or "") != plan_id:
                continue
            chunks += 1
            targets += _as_int(row.get("target_count"))
    return chunks, targets


def plan_header_row(
    *,
    plan_id: str,
    view_id: str,
    view_name: str,
    target_grain: str,
    target_table: str,
    source_snapshot_name: str,
    base_policy: str,
    conflict_policy: str,
    storage: dict[str, Any],
    counts: dict[str, int],
    plan_digest: str,
    artifact_id: str,
    frozen: bool,
    transform_id: str,
    source_view_transform_id: str,
    source_snapshot_transform_id: str,
    table_versions: Sequence[tuple[str, int]],
    summary: dict[str, Any],
    created_by: str,
    created_at: datetime | None = None,
    state: str = STATE_ACTIVE,
    target_fragment_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build the ``curation_row_plans`` header row for a compiled plan."""
    summary_json = json.dumps(summary, sort_keys=True, default=str)
    return {
        "plan_id": plan_id,
        "view_id": view_id,
        "view_name": view_name,
        "target_grain": target_grain,
        "target_table": target_table,
        "source_snapshot_name": source_snapshot_name,
        "base_policy": base_policy,
        "conflict_policy": conflict_policy,
        "storage_kind": str(storage.get("kind") or "inline"),
        "chunk_table": str(storage.get("table") or ""),
        "chunk_size": _as_int(storage.get("chunk_size")),
        "chunk_count": _as_int(storage.get("chunk_count")),
        "target_count": _as_int(storage.get("target_count")),
        "candidate_count": _as_int(counts.get("candidate_count")),
        "selected_count": _as_int(counts.get("selected_count")),
        "rejected_count": _as_int(counts.get("rejected_count")),
        "conflict_count": _as_int(counts.get("conflict_count")),
        "label_intent_count": _as_int(counts.get("label_intent_count")),
        "scenario_count": _as_int(counts.get("scenario_count")),
        "membership_transform_count": _as_int(counts.get("membership_transform_count")),
        "superseded_membership_count": _as_int(counts.get("superseded_membership_count")),
        "lance_row_ids_present": bool(counts.get("lance_row_ids_present", True)),
        "target_fragment_ids": list(target_fragment_ids or ()),
        "target_ids_digest": str(storage.get("target_ids_digest") or ""),
        "plan_digest": plan_digest,
        "artifact_id": artifact_id,
        "frozen": bool(frozen),
        "metadata_only": True,
        "payload_copy_policy": "logical-reference",
        "copied_payload_bytes": 0,
        "transform_id": transform_id,
        "source_view_transform_id": source_view_transform_id,
        "source_snapshot_transform_id": source_snapshot_transform_id,
        "table_versions": [
            {"table": table, "version": int(version), "tag": ""}
            for table, version in table_versions
        ],
        "state": state,
        "superseded_at": None,
        "pruned_at": None,
        "retention_policy_json": "",
        "summary_available": True,
        "summary_json": summary_json,
        "created_by": created_by,
        "created_at": created_at or datetime.now(UTC),
    }


def publish_row_plan(lake: Lake, header: dict[str, Any]) -> dict[str, Any]:
    """Publish (upsert) the plan header as the *final* commit of a plan write.

    Chunks are written first: a crash between the two leaves unreferenced chunk
    rows -- which the next identical compile converges onto, and which
    :func:`validate_row_plan_storage` reports as orphans -- rather than a header
    promising targets that were never persisted (backlog 0141).
    """
    row = {key: header.get(key) for key in _PLAN_ROW_KEYS}
    _merge_insert_with_retry(
        lake.table(_PLAN_TABLE),
        "plan_id",
        pa.Table.from_pylist([row], schema=CURATION_ROW_PLANS_SCHEMA),
        update_matched=True,
    )
    return row


# --------------------------------------------------------------------------- #
# Read path: bounded windowed chunk reads.
# --------------------------------------------------------------------------- #


def _iter_chunk_rows(
    lake: Lake,
    plan_id: str,
    *,
    target_count: int,
    chunk_size: int,
    scan_from_ordinal: int = 0,
    stats: Any = None,
) -> Iterable[dict[str, Any]]:
    """Yield a plan's chunk rows in ascending ``start_ordinal`` order.

    Reads a bounded *window* of ordinals at a time (``chunk_size *
    _CHUNK_SCAN_BATCH`` targets) via an indexed predicate, so client memory is
    O(window) regardless of plan size and a page starting deep in a large plan
    never reads the chunks before it (mirrors ``curate._stream_view_chunk_rows``).
    """
    width = max(1, int(chunk_size))
    total = max(0, int(target_count))
    window_ids = width * _CHUNK_SCAN_BATCH
    plan_literal = _sql_literal(plan_id)
    lo = max(0, (int(scan_from_ordinal) // width) * width)
    while lo < total:
        hi = lo + window_ids
        where_sql = (
            f"plan_id = {plan_literal} "
            f"AND start_ordinal >= {lo} AND start_ordinal < {hi}"
        )
        window: list[dict[str, Any]] = []
        for batch in _stream_rows(
            lake,
            _CHUNK_TABLE,
            columns=_CHUNK_READ_COLUMNS,
            where_sql=where_sql,
            batch_size=_CHUNK_SCAN_BATCH,
            stats=stats,
        ):
            for row in batch:
                # The materialized fallback ignores the pushdown predicate, so
                # re-apply it here rather than trusting the scan.
                if str(row.get("plan_id") or "") != plan_id:
                    continue
                start = _as_int(row.get("start_ordinal"))
                if start < lo or start >= hi:
                    continue
                window.append(row)
        window.sort(key=lambda row: (_as_int(row.get("start_ordinal")), _as_int(row.get("chunk_index"))))
        yield from window
        lo = hi


def _iter_chunked_targets(
    lake: Lake,
    plan_id: str,
    *,
    target_count: int,
    chunk_size: int,
    start_ordinal: int = 0,
    stats: Any = None,
) -> Iterator[CurationRowPlanTarget]:
    """Stream a chunked plan's targets, validating contiguity as it goes."""
    expected_ordinal = max(0, (int(start_ordinal) // max(1, int(chunk_size))) * max(1, int(chunk_size)))
    emitted = 0
    wanted_from = int(start_ordinal)
    for row in _iter_chunk_rows(
        lake,
        plan_id,
        target_count=target_count,
        chunk_size=chunk_size,
        scan_from_ordinal=start_ordinal,
        stats=stats,
    ):
        start = _as_int(row.get("start_ordinal"))
        if start != expected_ordinal:
            raise RowPlanCatalogError(
                f"row plan {plan_id!r} chunk storage is non-contiguous: expected "
                f"start_ordinal {expected_ordinal}, found {start}"
            )
        target_ids = list(row.get("target_ids") or ())
        row_ids = list(row.get("lance_row_ids") or ())
        for offset, target_id in enumerate(target_ids):
            ordinal = start + offset
            if ordinal < wanted_from:
                continue
            lance_row_id = row_ids[offset] if offset < len(row_ids) else None
            yield CurationRowPlanTarget(ordinal, str(target_id), lance_row_id)
            emitted += 1
        expected_ordinal = start + len(target_ids)
    remaining = max(0, int(target_count) - wanted_from)
    if emitted < remaining:
        raise RowPlanCatalogError(
            f"row plan {plan_id!r} chunk storage is short: read {emitted} of "
            f"{remaining} targets from ordinal {wanted_from}"
        )


def _entry_from_row(row: dict[str, Any]) -> CurationRowPlanEntry:
    return CurationRowPlanEntry(
        plan_id=str(row.get("plan_id") or ""),
        view_id=str(row.get("view_id") or ""),
        view_name=str(row.get("view_name") or ""),
        target_grain=str(row.get("target_grain") or ""),
        target_table=str(row.get("target_table") or ""),
        source_snapshot_name=str(row.get("source_snapshot_name") or ""),
        base_policy=str(row.get("base_policy") or ""),
        conflict_policy=str(row.get("conflict_policy") or ""),
        storage_kind=str(row.get("storage_kind") or "inline"),
        chunk_table=str(row.get("chunk_table") or ""),
        chunk_size=_as_int(row.get("chunk_size")),
        chunk_count=_as_int(row.get("chunk_count")),
        target_count=_as_int(row.get("target_count")),
        candidate_count=_as_int(row.get("candidate_count")),
        selected_count=_as_int(row.get("selected_count")),
        rejected_count=_as_int(row.get("rejected_count")),
        conflict_count=_as_int(row.get("conflict_count")),
        label_intent_count=_as_int(row.get("label_intent_count")),
        scenario_count=_as_int(row.get("scenario_count")),
        membership_transform_count=_as_int(row.get("membership_transform_count")),
        superseded_membership_count=_as_int(row.get("superseded_membership_count")),
        lance_row_ids_present=bool(row.get("lance_row_ids_present")),
        target_ids_digest=str(row.get("target_ids_digest") or ""),
        plan_digest=str(row.get("plan_digest") or ""),
        artifact_id=str(row.get("artifact_id") or ""),
        frozen=bool(row.get("frozen")),
        metadata_only=bool(row.get("metadata_only", True)),
        payload_copy_policy=str(row.get("payload_copy_policy") or "logical-reference"),
        copied_payload_bytes=_as_int(row.get("copied_payload_bytes")),
        transform_id=str(row.get("transform_id") or ""),
        source_view_transform_id=str(row.get("source_view_transform_id") or ""),
        source_snapshot_transform_id=str(row.get("source_snapshot_transform_id") or ""),
        state=str(row.get("state") or STATE_ACTIVE),
        summary_available=bool(row.get("summary_available")),
        created_by=str(row.get("created_by") or ""),
        created_at=_coerce_dt(row.get("created_at")),
    )


def _plan_row(
    lake: Lake,
    plan_id: str,
    *,
    columns: Sequence[str] | None = None,
    stats: Any = None,
) -> dict[str, Any] | None:
    """Fetch one plan header by id via an indexed predicate (never a full scan)."""
    where_sql = f"plan_id = {_sql_literal(plan_id)}"
    for batch in _stream_rows(
        lake, _PLAN_TABLE, columns=columns or PLAN_SUMMARY_COLUMNS, where_sql=where_sql, stats=stats
    ):
        for row in batch:
            if str(row.get("plan_id") or "") == plan_id:
                return row
    return None


def read_row_plan(lake: Lake, plan_id: str, *, stats: Any = None) -> CurationRowPlanEntry:
    """One compiled plan's promoted header columns, by id."""
    row = _plan_row(lake, plan_id, stats=stats)
    if row is None:
        raise RowPlanCatalogError(f"no curation row plan {plan_id!r} in {lake.uri}")
    return _entry_from_row(row)


def row_plan_summary(lake: Lake, plan_id: str, *, stats: Any = None) -> dict[str, Any]:
    """Bounded compile summary for a plan: counts plus capped diagnostic samples.

    The samples are explicitly truncated at :data:`SAMPLE_LIMIT` with
    ``*_truncated`` flags, so a caller reading a million-target plan's conflicts
    is never handed an unbounded list and never silently shown a partial one.
    """
    row = _plan_row(
        lake,
        plan_id,
        columns=(*PLAN_SUMMARY_COLUMNS, "summary_json"),
        stats=stats,
    )
    if row is None:
        raise RowPlanCatalogError(f"no curation row plan {plan_id!r} in {lake.uri}")
    entry = _entry_from_row(row)
    body: dict[str, Any] = {}
    raw = str(row.get("summary_json") or "")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            body = parsed
    # ``summary_available`` comes from the promoted column, which ``_apply_state``
    # maintains -- recomputing it as ``bool(raw)`` here gave two answers to one
    # question, and disagreed with what ``list_row_plans`` reports.
    return {
        **entry.to_dict(),
        "schema_version": ROW_PLAN_CATALOG_SCHEMA_VERSION,
        "summary": body,
    }


def open_row_plan_targets(
    lake: Lake,
    plan_id: str,
    *,
    page_size: int | None = None,
    stats: Any = None,
) -> CurationRowPlanTargets:
    """Lazy, ordered handle over a stored plan's targets (bounded reads only).

    Also resolves whether the plan's stored ``lance_row_ids`` are still meaningful:
    they are fragment addresses, so a ``lake maintain --compact`` on the target
    table since the compile can have remapped them. The handle reports that as
    ``row_ids_trustworthy`` rather than letting a consumer ``take_row_ids`` into a
    raw Lance error -- the ``target_id`` values are always valid.
    """
    # ``summary_json`` is projected up front so an inline plan resolves in ONE header
    # read: it holds the inline ids, and inline is the common case (any plan at or
    # below INLINE_TARGET_LIMIT).
    row = _plan_row(
        lake,
        plan_id,
        columns=(
            *PLAN_SUMMARY_COLUMNS,
            "table_versions",
            "target_fragment_ids",
            "summary_json",
        ),
        stats=stats,
    )
    if row is None:
        raise RowPlanCatalogError(f"no curation row plan {plan_id!r} in {lake.uri}")
    entry = _entry_from_row(row)
    if entry.state == STATE_PRUNED:
        raise RowPlanCatalogError(
            f"row plan {plan_id!r} was pruned by retention; its chunked targets are "
            "no longer stored (recompile the view to rebuild it)"
        )
    inline: tuple[tuple[str, int | None], ...] | None = None
    if entry.storage_kind == "inline":
        inline = _inline_targets_from_row(row)
    stale, staleness = row_ids_are_stale(lake, row)
    trustworthy = not stale
    pinned = staleness["pinned_version"]
    current = staleness["current_version"]
    return CurationRowPlanTargets(
        lake=lake,
        plan_id=plan_id,
        target_grain=entry.target_grain,
        target_table=entry.target_table,
        storage_kind=entry.storage_kind,
        target_count=entry.target_count,
        chunk_size=entry.chunk_size or CHUNK_SIZE,
        chunk_count=entry.chunk_count,
        inline_targets=inline,
        page_size=DEFAULT_TARGET_PAGE_SIZE if page_size is None else int(page_size),
        row_ids_trustworthy=trustworthy,
        pinned_table_version=pinned,
        current_table_version=current,
    )


def _inline_targets_from_row(row: dict[str, Any]) -> tuple[tuple[str, int | None], ...]:
    """Inline plan targets, decoded from a header row's bounded summary.

    Pure row -> tuple, so the caller that already holds the header does not read it
    again. Only reachable for ``storage_kind == "inline"`` plans, i.e. at most
    :data:`INLINE_TARGET_LIMIT` ids -- bounded by construction.
    """
    try:
        body = json.loads(str(row.get("summary_json") or "") or "{}")
    except json.JSONDecodeError:
        body = {}
    target_ids = [str(item) for item in (body.get("target_ids") or ())]
    row_ids = list(body.get("lance_row_ids") or ())
    return tuple(
        (target_id, row_ids[index] if index < len(row_ids) else None)
        for index, target_id in enumerate(target_ids)
    )


# --------------------------------------------------------------------------- #
# Plan listing (keyset paged).
# --------------------------------------------------------------------------- #


def _normalize_page_size(page_size: int | None, default: int, maximum: int) -> int:
    if page_size is None:
        return default
    size = int(page_size)
    if size <= 0:
        raise RowPlanCatalogError(f"page_size must be > 0, got {page_size}")
    return min(size, maximum)


def _plan_equality_filters(**candidates: str | None) -> dict[str, str]:
    """The requested ``column = value`` header filters, dropping unset ones.

    Shared by the pushdown builder and the client-side matcher so the two cannot
    disagree about which columns a filter applies to -- they previously each
    carried their own copy of the same seven-pair list.
    """
    return {column: str(value) for column, value in candidates.items() if value}


def _plan_filter_sql(
    *,
    view_id: str | None,
    view_name: str | None,
    target_grain: str | None,
    source_snapshot_name: str | None,
    storage_kind: str | None,
    state: str | None,
    frozen: bool | None,
    transform_id: str | None,
    cursor_key: tuple[datetime, str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> str | None:
    equality = _plan_equality_filters(
        view_id=view_id,
        view_name=view_name,
        target_grain=target_grain,
        source_snapshot_name=source_snapshot_name,
        storage_kind=storage_kind,
        state=state,
        transform_id=transform_id,
    )
    clauses: list[str] = []
    for column, value in equality.items():
        clauses.append(f"{column} = {_sql_literal(str(value))}")
    if frozen is not None:
        clauses.append(f"frozen = {'true' if frozen else 'false'}")
    # Push the keyset lower bound down so page N seeks on the ``created_at`` BTREE
    # instead of ordering the whole filtered set and discarding the prefix. The
    # bound is inclusive on ``created_at`` because the tie-break on ``plan_id`` is
    # still applied client-side -- a page boundary that lands inside a timestamp
    # tie must not skip its siblings.
    lower = cursor_key[0] if cursor_key is not None else None
    if since is not None and (lower is None or since > lower):
        lower = since
    if lower is not None:
        clauses.append(f"created_at >= {_ts_literal(lower)}")
    if until is not None:
        clauses.append(f"created_at <= {_ts_literal(until)}")
    return " AND ".join(clauses) if clauses else None


def _ts_literal(value: datetime) -> str:
    """Timestamp literal for a pushdown predicate.

    Must be a typed ``timestamp '...'`` literal, not a quoted string: Lance
    refuses to resolve ``created_at >= '<str>'`` against a timestamp column, which
    would make the whole ordered read fall back to the bounded heap. Mirrors
    ``aligned_tick_lifecycle._ts_literal``.
    """
    coerced = value if value.tzinfo else value.replace(tzinfo=UTC)
    return "timestamp '" + coerced.astimezone(UTC).isoformat() + "'"


def _plan_matcher(
    *,
    view_id: str | None,
    view_name: str | None,
    target_grain: str | None,
    source_snapshot_name: str | None,
    storage_kind: str | None,
    state: str | None,
    frozen: bool | None,
    transform_id: str | None,
    since: datetime | None,
    until: datetime | None,
) -> Callable[[dict[str, Any]], bool]:
    equality = _plan_equality_filters(
        view_id=view_id,
        view_name=view_name,
        target_grain=target_grain,
        source_snapshot_name=source_snapshot_name,
        storage_kind=storage_kind,
        state=state,
        transform_id=transform_id,
    )

    def _matches(row: dict[str, Any]) -> bool:
        for column, value in equality.items():
            if str(row.get(column) or "") != str(value):
                return False
        if frozen is not None and bool(row.get("frozen")) is not bool(frozen):
            return False
        created = _plan_row_key(row)[0]
        if since is not None and created < since:
            return False
        if until is not None and created > until:
            return False
        return True

    return _matches


def _read_plans_ordered(
    handle: Any,
    *,
    where_sql: str | None,
    cursor_key: tuple[datetime, str] | None,
    since: datetime | None,
    until: datetime | None,
    page_size: int,
    stats: Any,
) -> tuple[list[dict[str, Any]], bool] | None:
    want = page_size + 1
    try:
        from lancedb.query import ColumnOrdering

        query = handle.search()
        if where_sql:
            query = query.where(where_sql)
        query = query.select(list(PLAN_SUMMARY_COLUMNS))
        query = query.order_by(
            [
                ColumnOrdering(column_name="created_at", ascending=True),
                ColumnOrdering(column_name="plan_id", ascending=True),
            ]
        )
        batches = query.to_batches(batch_size=_SCAN_BATCH)
    except Exception:
        return None
    collected: list[dict[str, Any]] = []
    has_more = False
    try:
        stop = False
        for batch in batches:
            rows = batch.to_pylist()
            _record_batch(stats, _PLAN_TABLE, len(rows))
            for row in rows:
                key = _plan_row_key(row)
                if until is not None and key[0] > until:
                    stop = True
                    break
                if since is not None and key[0] < since:
                    continue
                if cursor_key is not None and key <= cursor_key:
                    continue
                collected.append(row)
                if len(collected) >= want:
                    has_more = True
                    stop = True
                    break
            if stop:
                break
    except Exception:
        return None
    if has_more:
        collected = collected[:page_size]
    return collected, has_more


def _read_plans_bounded_heap(
    lake: Lake,
    *,
    where_sql: str | None,
    matcher: Callable[[dict[str, Any]], bool],
    cursor_key: tuple[datetime, str] | None,
    page_size: int,
    stats: Any,
) -> tuple[list[dict[str, Any]], bool]:
    """The page's rows via a bounded top-k, for a backend that cannot order.

    Uses :func:`heapq.nsmallest` rather than a hand-rolled heap. That is not just
    less code: keeping the ``k`` smallest keys requires evicting the current
    *largest* on overflow, and getting that backwards silently drops the row that
    belongs on page one. ``nsmallest`` gets it right by construction, in O(k)
    memory, and already breaks ties toward the earliest-seen row.
    """
    want = page_size + 1
    qualifying = 0

    def _candidates() -> Iterator[dict[str, Any]]:
        nonlocal qualifying
        for batch in _stream_rows(
            lake, _PLAN_TABLE, columns=PLAN_SUMMARY_COLUMNS, where_sql=where_sql, stats=stats
        ):
            for row in batch:
                if not matcher(row):
                    continue
                if cursor_key is not None and _plan_row_key(row) <= cursor_key:
                    continue
                qualifying += 1
                yield row

    ordered = heapq.nsmallest(want, _candidates(), key=_plan_row_key)
    has_more = qualifying > page_size
    return ordered[:page_size], has_more


def list_row_plans(
    lake: Lake,
    *,
    view_id: str | None = None,
    view_name: str | None = None,
    target_grain: str | None = None,
    source_snapshot_name: str | None = None,
    storage_kind: str | None = None,
    state: str | None = None,
    frozen: bool | None = None,
    transform_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    page_size: int | None = None,
    cursor: str | None = None,
    page_index: int = 0,
    stats: Any = None,
) -> CurationRowPlanPage:
    """One deterministic, bounded page of row-plan headers, ascending key.

    Filters push down to indexed header columns; rows are ordered by
    ``(created_at, plan_id)`` in the engine. Client memory is bounded to
    ``page_size`` (on a backend that cannot order, a top-``page_size+1`` heap
    produces the same page in O(page_size) memory and warns rather than silently
    taking the costlier path).
    """
    limit = _normalize_page_size(page_size, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
    cursor_key: tuple[datetime, str] | None = None
    if cursor:
        created, plan_id = _decode_plan_cursor(cursor)
        cursor_key = (created or datetime.min.replace(tzinfo=UTC), plan_id)
    where_sql = _plan_filter_sql(
        view_id=view_id,
        view_name=view_name,
        target_grain=target_grain,
        source_snapshot_name=source_snapshot_name,
        storage_kind=storage_kind,
        state=state,
        frozen=frozen,
        transform_id=transform_id,
        cursor_key=cursor_key,
        since=since,
        until=until,
    )
    matcher = _plan_matcher(
        view_id=view_id,
        view_name=view_name,
        target_grain=target_grain,
        source_snapshot_name=source_snapshot_name,
        storage_kind=storage_kind,
        state=state,
        frozen=frozen,
        transform_id=transform_id,
        since=since,
        until=until,
    )
    ordered = _read_plans_ordered(
        lake.table(_PLAN_TABLE),
        where_sql=where_sql,
        cursor_key=cursor_key,
        since=since,
        until=until,
        page_size=limit,
        stats=stats,
    )
    if ordered is None:
        warnings.warn(
            "row-plan listing could not push ordering into the engine; using a "
            "bounded top-page heap instead (same page, O(page_size) memory)",
            RuntimeWarning,
            stacklevel=2,
        )
        rows, has_more = _read_plans_bounded_heap(
            lake,
            where_sql=where_sql,
            matcher=matcher,
            cursor_key=cursor_key,
            page_size=limit,
            stats=stats,
        )
    else:
        rows, has_more = ordered
        rows = [row for row in rows if matcher(row)]
    records = tuple(_entry_from_row(row) for row in rows)
    next_cursor = None
    if has_more and records:
        last = records[-1]
        next_cursor = _encode_plan_cursor(last.created_at, last.plan_id)
    return CurationRowPlanPage(
        records=records,
        page_size=limit,
        page_index=int(page_index),
        cursor=cursor,
        next_cursor=next_cursor,
        has_more=bool(has_more),
    )


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #


def validate_row_plan_storage(
    lake: Lake,
    *,
    plan_id: str | None = None,
    include_orphans: bool = True,
    stats: Any = None,
) -> RowPlanValidationReport:
    """Check chunked plan storage against its header (counts, bounds, order).

    Scans one plan when ``plan_id`` is given, otherwise every chunked plan, and
    reads each plan's chunks through the same bounded windowed reader used by the
    target pager -- never the whole chunk table at once.

    ``include_orphans=False`` skips the orphan sweep (a whole-chunk-table scan plus
    a whole-header scan) for a caller that is about to run
    :func:`compact_row_plan_chunks`, which derives the same set -- and more, since
    it also separates in-flight writes.
    """
    issues: list[RowPlanValidationIssue] = []
    scanned: list[str] = []
    known: set[str] = set()
    # Row-id staleness is a property of the *target table*, not of each plan, and
    # only a handful of distinct tables exist across a whole catalog. Resolving the
    # version + fragment set once per table keeps validation from opening the same
    # table and re-sorting its fragment ids once per plan.
    table_state: dict[str, tuple[int | None, list[int] | None]] = {}
    where_sql = "storage_kind = 'chunked'"
    if plan_id:
        where_sql = f"{where_sql} AND plan_id = {_sql_literal(plan_id)}"
    columns = (
        "plan_id",
        "storage_kind",
        "target_count",
        "chunk_size",
        "chunk_count",
        "target_ids_digest",
        "state",
        # Needed for the row-id staleness check; ``table_versions`` is a small
        # per-plan struct list, not a body.
        "target_table",
        "lance_row_ids_present",
        "table_versions",
        "target_fragment_ids",
    )
    for batch in _stream_rows(lake, _PLAN_TABLE, columns=columns, where_sql=where_sql, stats=stats):
        for row in batch:
            if str(row.get("storage_kind") or "") != "chunked":
                continue
            current = str(row.get("plan_id") or "")
            if plan_id and current != plan_id:
                continue
            known.add(current)
            if str(row.get("state") or "") == STATE_PRUNED:
                continue
            scanned.append(current)
            issues.extend(_validate_one_plan(lake, row, stats=stats, table_state=table_state))
    orphans: list[str] = []
    if include_orphans and not plan_id:
        seen_chunk_plans: set[str] = set()
        for batch in _stream_rows(lake, _CHUNK_TABLE, columns=("plan_id",), stats=stats):
            for row in batch:
                seen_chunk_plans.add(str(row.get("plan_id") or ""))
        header_plans = _all_plan_ids(lake, stats=stats)
        orphans = sorted(seen_chunk_plans - header_plans - {""})
    return RowPlanValidationReport(
        scanned_plan_ids=tuple(scanned),
        issues=tuple(issues),
        orphan_chunk_plan_ids=tuple(orphans),
    )


def _all_plan_ids(lake: Lake, *, stats: Any = None) -> set[str]:
    plan_ids: set[str] = set()
    for batch in _stream_rows(lake, _PLAN_TABLE, columns=("plan_id",), stats=stats):
        for row in batch:
            plan_ids.add(str(row.get("plan_id") or ""))
    return plan_ids


def _validate_one_plan(
    lake: Lake,
    header: dict[str, Any],
    *,
    stats: Any = None,
    table_state: dict[str, tuple[int | None, list[int] | None]] | None = None,
) -> list[RowPlanValidationIssue]:
    plan_id = str(header.get("plan_id") or "")
    expected_count = _as_int(header.get("target_count"))
    expected_chunks = _as_int(header.get("chunk_count"))
    width = _as_int(header.get("chunk_size")) or CHUNK_SIZE
    issues: list[RowPlanValidationIssue] = []
    seen_chunks = 0
    seen_targets = 0
    expected_ordinal = 0
    seen_indexes: set[int] = set()
    digest = TargetIdsDigest()
    for row in _iter_chunk_rows(
        lake, plan_id, target_count=expected_count, chunk_size=width, stats=stats
    ):
        index = _as_int(row.get("chunk_index"))
        start = _as_int(row.get("start_ordinal"))
        target_ids = list(row.get("target_ids") or ())
        if index in seen_indexes:
            issues.append(
                RowPlanValidationIssue(plan_id, "duplicate-chunk", f"chunk_index {index} seen twice", index)
            )
            continue
        seen_indexes.add(index)
        if start != expected_ordinal:
            issues.append(
                RowPlanValidationIssue(
                    plan_id,
                    "non-contiguous",
                    f"chunk {index} starts at {start}, expected {expected_ordinal}",
                    index,
                )
            )
        if _as_int(row.get("end_ordinal")) != start + len(target_ids) - 1:
            issues.append(
                RowPlanValidationIssue(
                    plan_id, "chunk-bounds-mismatch", f"chunk {index} end_ordinal disagrees with its ids", index
                )
            )
        if _as_int(row.get("target_count")) != len(target_ids):
            issues.append(
                RowPlanValidationIssue(
                    plan_id, "chunk-count-mismatch", f"chunk {index} target_count disagrees with its ids", index
                )
            )
        for target_id in target_ids:
            digest.update(str(target_id))
        seen_chunks += 1
        seen_targets += len(target_ids)
        expected_ordinal = start + len(target_ids)
    if seen_targets != expected_count:
        issues.append(
            RowPlanValidationIssue(
                plan_id,
                "target-count-mismatch",
                f"header claims {expected_count} targets, chunks hold {seen_targets}",
            )
        )
    if expected_chunks and seen_chunks != expected_chunks:
        issues.append(
            RowPlanValidationIssue(
                plan_id,
                "chunk-count-mismatch",
                f"header claims {expected_chunks} chunks, found {seen_chunks}",
            )
        )
    expected_digest = str(header.get("target_ids_digest") or "")
    if expected_digest and seen_targets == expected_count and digest.hexdigest() != expected_digest:
        issues.append(
            RowPlanValidationIssue(
                plan_id, "digest-mismatch", "recomputed target-id digest disagrees with the header"
            )
        )
    issues.extend(_row_id_staleness_issues(lake, header, table_state=table_state))
    return issues


def row_ids_are_stale(
    lake: Lake,
    header: dict[str, Any],
    *,
    table_state: dict[str, tuple[int | None, list[int] | None]] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Whether a plan's stored ``lance_row_ids`` may have been remapped.

    Lance row ids are fragment addresses, not stable keys: ``lake maintain
    --compact`` rewrites fragments and remaps them, so a plan compiled before a
    compaction can hold addresses that no longer resolve. The target ids are always
    still valid, so this is a degradation to report, not corruption.

    The signal is the target table's **fragment identity**, not its version number.
    A plain append also bumps the version but remaps nothing; keying on the version
    would flag every plan in any actively-ingesting lake, permanently pushing
    consumers off the ``take_row_ids`` fast path that BUG-06 round 2 exists to
    provide -- and burying real issues under the noise.
    """
    target_table = str(header.get("target_table") or "")
    detail: dict[str, Any] = {
        "pinned_version": _pinned_version(header, target_table),
        "current_version": None,
        "pinned_fragments": _pinned_fragments(header),
        "current_fragments": None,
    }
    if not target_table or not bool(header.get("lance_row_ids_present", True)):
        return False, detail
    version, current_fragments = _target_table_state(lake, target_table, cache=table_state)
    if version is None:
        return False, detail
    detail["current_version"] = version
    detail["current_fragments"] = current_fragments
    pinned_fragments = detail["pinned_fragments"]
    if pinned_fragments is None or current_fragments is None:
        # No fragment signature to compare (plan predates the signature, or the
        # backend does not expose fragments). Fall back to the version check, which
        # is conservative but only fires when the version actually moved.
        pinned = detail["pinned_version"]
        if pinned is None or detail["current_version"] == pinned:
            return False, detail
        return True, detail
    # Compaction rewrites the fragment set; an append only extends it. Existing row
    # addresses survive an extension.
    return not set(pinned_fragments).issubset(set(current_fragments)), detail


def _row_id_staleness_issues(
    lake: Lake,
    header: dict[str, Any],
    *,
    table_state: dict[str, tuple[int | None, list[int] | None]] | None = None,
) -> list[RowPlanValidationIssue]:
    """Emit a ``row-ids-possibly-stale`` issue when the row ids may be remapped."""
    stale, detail = row_ids_are_stale(lake, header, table_state=table_state)
    if not stale:
        return []
    return [
        RowPlanValidationIssue(
            str(header.get("plan_id") or ""),
            "row-ids-possibly-stale",
            f"{header.get('target_table')} fragments changed since the plan was "
            f"compiled (pinned version {detail['pinned_version']}, now "
            f"{detail['current_version']}); stored lance_row_ids are fragment "
            "addresses and may have been remapped by compaction. Read targets by "
            "target_id, or recompile the plan to re-resolve row ids.",
        )
    ]


def _pinned_version(header: dict[str, Any], table: str) -> int | None:
    for entry in header.get("table_versions") or ():
        if isinstance(entry, dict) and str(entry.get("table") or "") == table:
            return _as_int(entry.get("version"))
    return None


def _pinned_fragments(header: dict[str, Any]) -> list[int] | None:
    """Fragment ids the plan recorded for its target table, if any."""
    raw = header.get("target_fragment_ids")
    if raw is None:
        return None
    try:
        return [int(item) for item in raw]
    except (TypeError, ValueError):
        return None


def _target_table_state(
    lake: Lake,
    table: str,
    *,
    cache: dict[str, tuple[int | None, list[int] | None]] | None = None,
) -> tuple[int | None, list[int] | None]:
    """``(version, fragment_ids)`` for ``table``, memoized per validation pass.

    Both values are properties of the table, so a whole-catalog validation resolves
    each distinct target table once instead of once per plan.
    """
    if cache is not None and table in cache:
        return cache[table]
    try:
        version: int | None = int(lake.table(table).version)
    except Exception:
        version = None
    fragments = _fragment_signature(lake, table) if version is not None else None
    state = (version, fragments)
    if cache is not None:
        cache[table] = state
    return state


def current_fragment_ids(lake: Lake, table: str) -> list[int] | None:
    """Fragment ids of ``table`` right now, for pinning into a plan header."""
    return _fragment_signature(lake, table)


def _fragment_signature(lake: Lake, table: str) -> list[int] | None:
    """Current fragment ids for ``table``, or ``None`` if not exposed."""
    try:
        handle = lake.table(table)
        dataset = getattr(handle, "to_lance", None)
        fragments = dataset().get_fragments() if callable(dataset) else None
        if fragments is None:
            return None
        return sorted(int(fragment.fragment_id) for fragment in fragments)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Retention / compaction.
# --------------------------------------------------------------------------- #


def referenced_plan_ids(
    lake: Lake, plan_ids: Sequence[str], *, stats: Any = None
) -> set[str]:
    """Plan ids named by a training run or training report (chunked IN reads).

    A plan pinned as training or evaluation evidence is never prunable: dropping
    its chunks would break the manifest's claim that the run's row membership can
    be replayed.
    """
    wanted = {str(plan_id) for plan_id in plan_ids if plan_id}
    if not wanted:
        return set()
    found: set[str] = set()
    for table, column in REFERENCING_TABLES:
        try:
            lake.table(table)
        except Exception:
            continue
        for group in _chunked(sorted(wanted)):
            where_sql = _in_clause(column, group)
            try:
                for batch in _stream_rows(
                    lake, table, columns=(column,), where_sql=where_sql, stats=stats
                ):
                    for row in batch:
                        value = str(row.get(column) or "")
                        if value in wanted:
                            found.add(value)
            except Exception:  # noqa: BLE001 - a missing/odd table must not unprotect.
                # Fail closed: if we cannot prove a plan is unreferenced, treat
                # every candidate as referenced rather than pruning evidence.
                return set(wanted)
    return found


def _is_prunable(row: dict[str, Any]) -> bool:
    """A superseded, unfrozen plan is safe to compact; frozen evidence is not."""
    if bool(row.get("frozen")):
        return False
    if _as_int(row.get("copied_payload_bytes")) > 0:
        return False
    if str(row.get("artifact_id") or ""):
        return False
    return True


def _plan_series_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("view_id") or ""),
        str(row.get("target_grain") or ""),
        str(row.get("source_snapshot_name") or ""),
        str(row.get("base_policy") or ""),
    )


def _iter_plans_for_retention(
    lake: Lake, *, where_sql: str | None, stats: Any
) -> Iterable[dict[str, Any]]:
    """Stream retention candidates in stable series order.

    Prefers engine ordering so the series grouping never needs the whole catalog
    in memory; warns and sorts a bounded candidate set in Python only when the
    backend cannot order (never silently degrades).
    """
    handle = lake.table(_PLAN_TABLE)
    # Build the ordered query inside the try, but iterate it OUTSIDE: retention is
    # rank-sensitive, so a mid-stream read error must fail the pass, never restart
    # it. The shared ``_stream_rows`` fallback re-reads the source from the top,
    # which for this consumer means already-yielded rows are counted twice, the
    # series rank shifts, and the *newest* (rank 0) plan gets pruned while the
    # report says success.
    batches = None
    try:
        from lancedb.query import ColumnOrdering

        query = handle.search()
        if where_sql:
            query = query.where(where_sql)
        query = query.select(list(_RETENTION_COLUMNS)).order_by(
            [
                ColumnOrdering(column_name="view_id", ascending=True),
                ColumnOrdering(column_name="target_grain", ascending=True),
                ColumnOrdering(column_name="source_snapshot_name", ascending=True),
                ColumnOrdering(column_name="base_policy", ascending=True),
                ColumnOrdering(column_name="created_at", ascending=False),
                ColumnOrdering(column_name="plan_id", ascending=False),
            ]
        )
        batches = query.to_batches(batch_size=_SCAN_BATCH)
    except Exception:
        batches = None
    if batches is not None:
        for batch in batches:
            rows = batch.to_pylist()
            _record_batch(stats, _PLAN_TABLE, len(rows))
            yield from rows
        return
    warnings.warn(
        "row-plan retention could not push series ordering into the engine; "
        "sorting the candidate set in Python instead (same order, but the "
        "candidate set is held in memory for the sort)",
        RuntimeWarning,
        stacklevel=2,
    )
    collected = [
        row
        for batch in _stream_rows(
            lake, _PLAN_TABLE, columns=_RETENTION_COLUMNS, where_sql=where_sql, stats=stats
        )
        for row in batch
    ]
    # Must match the engine ordering exactly: series ascending, then created_at and
    # plan_id *descending*, so rank 0 is the NEWEST plan in the series. Sorting
    # ascending here would invert ``retain_latest`` and prune the freshest plan while
    # keeping the oldest. Two stable passes express that without a custom
    # reversed-comparison key: sort by the descending part first, then re-sort by
    # series, which Python's stable sort preserves within each group.
    collected.sort(key=_plan_row_key, reverse=True)
    collected.sort(key=_plan_series_key)
    yield from collected


def prune_row_plans(
    lake: Lake,
    *,
    retain_latest: int = 1,
    older_than: datetime | timedelta | None = None,
    dry_run: bool = False,
    created_by: str = "lancedb-robotics",
    stats: Any = None,
) -> RowPlanRetentionReport:
    """Compact superseded, unreferenced row plans; keep frozen/pinned evidence.

    Within each ``(view, grain, source snapshot, base policy)`` series the newest
    ``retain_latest`` plans stay ``active``; older ones become ``superseded`` and,
    when they are prunable *and* older than ``older_than``, ``pruned``. A plan is
    protected -- never touched beyond a state flip -- when it is frozen, carries a
    lineage artifact id, or is named by ``training_runs.row_plan_id`` /
    ``training_reports.row_plan_id``.

    Pruning is safe-delete and write-ahead: chunk rows and the summary body are
    cleared first (both idempotent), then the header state flip is the final
    commit, so a crash mid-prune converges on the next pass instead of leaving a
    header that claims chunks which no longer exist.
    """
    if int(retain_latest) < 1:
        raise RowPlanCatalogError(f"retain_latest must be >= 1, got {retain_latest}")
    now = datetime.now(UTC)
    cutoff: datetime | None
    if isinstance(older_than, timedelta):
        cutoff = now - older_than
    else:
        cutoff = _coerce_dt(older_than) if older_than is not None else None
    policy = json.dumps(
        {
            "policy": "row-plan-retention/v1",
            "retain_latest": int(retain_latest),
            "older_than": cutoff.isoformat() if cutoff else "",
            "applied_at": now.isoformat(),
            "applied_by": created_by,
        },
        sort_keys=True,
    )

    scanned = 0
    protected: list[str] = []
    supersede_ids: list[str] = []
    prune_ids: list[str] = []
    supersede_batch: list[str] = []
    # ``(plan_id, current_state)`` so a deferred supersede knows whether the row is
    # already superseded -- rewriting it anyway churns a table version per protected
    # plan per pass and misreports a flip that did not happen.
    prune_batch: list[tuple[str, str]] = []
    delete_failed: list[str] = []
    chunks_deleted = 0
    bytes_cleared = 0
    series_seen: dict[tuple[str, str, str, str], int] = {}

    def _defer_supersede(plan_id: str, current_state: str) -> None:
        if current_state == STATE_SUPERSEDED:
            return
        supersede_ids.append(plan_id)
        supersede_batch.append(plan_id)
        if len(supersede_batch) >= _RETENTION_FLUSH:
            _flush_supersede()

    def _flush_supersede() -> None:
        if supersede_batch and not dry_run:
            _apply_state(lake, supersede_batch, state=STATE_SUPERSEDED, now=now, policy=policy)
        supersede_batch.clear()

    def _flush_prune() -> None:
        nonlocal chunks_deleted, bytes_cleared
        if not prune_batch:
            return
        batch_ids = [plan_id for plan_id, _ in prune_batch]
        # Re-check protection for THIS batch immediately before deleting anything.
        # Checking after the delete would make the protection decorative: a plan a
        # training run pins would lose its chunks and still be reported protected.
        referenced = referenced_plan_ids(lake, batch_ids, stats=stats)
        deletable = [plan_id for plan_id in batch_ids if plan_id not in referenced]
        if referenced:
            # Drop them from the pruned set in one pass; ``list.remove`` per id would
            # be O(prune_ids) each.
            prune_ids[:] = [plan_id for plan_id in prune_ids if plan_id not in referenced]
            for plan_id, current_state in prune_batch:
                if plan_id in referenced:
                    protected.append(plan_id)
                    _defer_supersede(plan_id, current_state)
        if deletable and not dry_run:
            # Write-ahead safe-delete: clear the chunk rows first (idempotent and
            # verified), then flip the header as the final commit. A plan whose
            # chunks could not be deleted keeps its header, so the next pass
            # rescans and retries it instead of stranding unreachable rows.
            deleted, failed = _delete_plan_chunks(lake, deletable)
            chunks_deleted += deleted
            if failed:
                failed_set = set(failed)
                delete_failed.extend(failed)
                prune_ids[:] = [
                    plan_id for plan_id in prune_ids if plan_id not in failed_set
                ]
                deletable = [
                    plan_id for plan_id in deletable if plan_id not in failed_set
                ]
            bytes_cleared += _apply_state(
                lake, deletable, state=STATE_PRUNED, now=now, policy=policy
            )
        prune_batch.clear()

    where_sql = f"state != {_sql_literal(STATE_PRUNED)}"
    for row in _iter_plans_for_retention(lake, where_sql=where_sql, stats=stats):
        current_state = str(row.get("state") or "")
        if current_state == STATE_PRUNED:
            continue
        scanned += 1
        key = _plan_series_key(row)
        rank = series_seen.get(key, 0)
        series_seen[key] = rank + 1
        plan_id = str(row.get("plan_id") or "")
        if rank < int(retain_latest):
            continue
        if not _is_prunable(row):
            protected.append(plan_id)
            _defer_supersede(plan_id, current_state)
            continue
        # No cutoff at all means soft-retire only: nothing is ever deleted unless
        # the caller explicitly asks for an age bound.
        if cutoff is None or _plan_row_key(row)[0] > cutoff:
            _defer_supersede(plan_id, current_state)
            continue
        prune_ids.append(plan_id)
        prune_batch.append((plan_id, current_state))
        if len(prune_batch) >= _RETENTION_FLUSH:
            _flush_prune()
    # Prune first: ``_flush_prune`` can add protected plans to ``supersede_batch``,
    # so flushing supersedes first would drop the final batch's protected flips.
    _flush_prune()
    _flush_supersede()

    if dry_run:
        # A dry run must preview the same decisions the real run would make,
        # protection included -- otherwise the preview cannot warn about a pinned
        # plan that a real run would (correctly) refuse to touch.
        referenced = referenced_plan_ids(lake, prune_ids, stats=stats)
        if referenced:
            prune_ids = [plan_id for plan_id in prune_ids if plan_id not in referenced]
            protected.extend(sorted(referenced))
    return RowPlanRetentionReport(
        dry_run=bool(dry_run),
        retain_latest=int(retain_latest),
        older_than=cutoff,
        scanned_count=scanned,
        protected_count=len(set(protected)),
        superseded_plan_ids=tuple(dict.fromkeys(supersede_ids)),
        pruned_plan_ids=tuple(dict.fromkeys(prune_ids)),
        protected_plan_ids=tuple(sorted(set(protected))),
        delete_failed_plan_ids=tuple(dict.fromkeys(delete_failed)),
        chunks_deleted=chunks_deleted,
        summary_bytes_cleared=bytes_cleared,
    )


def _apply_state(
    lake: Lake,
    plan_ids: Sequence[str],
    *,
    state: str,
    now: datetime,
    policy: str,
) -> int:
    """Flip the lifecycle state of the named plans in one bounded commit.

    Re-reads the *full* header rows for this bounded batch before writing. The
    retention scan projects a narrow, JSON-free column set (it visits every
    non-pruned plan), and ``merge_insert(...).when_matched_update_all()`` writes
    every column of the row it is given -- so rewriting a scan row would null the
    columns the scan did not read, destroying ``table_versions`` (the plan's
    version pinning) even on the soft-retire path that deletes nothing.

    Returns the number of ``summary_json`` bytes cleared, so the caller gets its
    byte accounting without a per-candidate read of the body inside the scan loop.
    """
    wanted = [str(plan_id) for plan_id in plan_ids if plan_id]
    if not wanted:
        return 0
    cleared = 0
    for group in _chunked(wanted):
        rows = _fetch_rows_by_ids(lake, _PLAN_TABLE, "plan_id", group, columns=_PLAN_ROW_KEYS)
        updated: list[dict[str, Any]] = []
        for plan_id in group:
            row = rows.get(plan_id)
            if row is None:
                continue
            merged = {key: row.get(key) for key in _PLAN_ROW_KEYS}
            merged["state"] = state
            merged["retention_policy_json"] = policy
            if state == STATE_SUPERSEDED:
                merged["superseded_at"] = merged.get("superseded_at") or now
            if state == STATE_PRUNED:
                merged["superseded_at"] = merged.get("superseded_at") or now
                merged["pruned_at"] = now
                cleared += len(str(merged.get("summary_json") or "").encode())
                merged["summary_available"] = False
                merged["summary_json"] = ""
                merged["chunk_count"] = 0
            updated.append(merged)
        _merge_insert_with_retry(
            lake.table(_PLAN_TABLE),
            "plan_id",
            pa.Table.from_pylist(updated, schema=CURATION_ROW_PLANS_SCHEMA),
            update_matched=True,
        )
    return cleared


def _delete_plan_chunks(lake: Lake, plan_ids: Sequence[str]) -> tuple[int, list[str]]:
    """Delete chunk rows for the given plans; return (deleted, failed plan ids).

    Verifies the delete by re-counting instead of trusting the pre-count, and
    reports the plans whose chunks are still present so the caller can leave their
    headers alone. Flipping a header to ``pruned`` after a failed delete would
    strand the chunk rows permanently: the retention scan skips pruned plans,
    validation skips them, and orphan compaction only reclaims chunks whose header
    is *missing*.
    """
    wanted = [str(plan_id) for plan_id in plan_ids if plan_id]
    deleted = 0
    failed: list[str] = []
    if not wanted:
        return 0, []
    try:
        handle = lake.table(_CHUNK_TABLE)
    except Exception:
        return 0, list(wanted)
    for group in _chunked(wanted):
        clause = _in_clause("plan_id", group)
        try:
            before = int(handle.count_rows(clause))
        except Exception:
            before = None
        try:
            handle.delete(clause)
        except Exception:  # noqa: BLE001 - report the failure; never claim success.
            failed.extend(group)
            continue
        try:
            after: int | None = int(handle.count_rows(clause))
        except Exception:
            after = None
        if after is None:
            # Fail closed: an unverifiable delete is NOT a verified one. Reading it
            # as "all rows gone" would let the caller mark the plan pruned with its
            # chunks possibly intact -- the exact stranding this verification
            # exists to prevent.
            failed.extend(group)
            continue
        if after:
            # The delete reported success but rows survive (a concurrent writer
            # re-added them, or the predicate did not match them all). Do not let
            # the caller mark these plans pruned.
            failed.extend(group)
            continue
        # Only count what we can actually attribute; an unknown ``before`` means the
        # rows are gone but the number is not reportable.
        if before is not None:
            deleted += max(0, before - after)
    return deleted, failed


def _fetch_rows_by_ids(
    lake: Lake,
    table: str,
    id_column: str,
    ids: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch only the rows whose ``id_column`` is in ``ids`` via chunked IN reads.

    Bounds the read to the requested id set instead of scanning the whole table
    (SKILLS.md §2). Deduplicates ids and keeps the last row seen per id.
    """
    wanted = list(dict.fromkeys(str(item) for item in ids))
    out: dict[str, dict[str, Any]] = {}
    if not wanted:
        return out
    # Set membership, not `in wanted` on the list: the fallback scan re-applies this
    # per returned row, and a list makes that O(ids) per row.
    wanted_set = set(wanted)
    for group in _chunked(wanted):
        where_sql = _in_clause(id_column, group)
        for batch in _stream_rows(lake, table, columns=columns, where_sql=where_sql):
            for row in batch:
                key = str(row.get(id_column) or "")
                if key in wanted_set:
                    out[key] = row
    return out


def compact_row_plan_chunks(
    lake: Lake,
    *,
    dry_run: bool = False,
    grace: timedelta | None = None,
    stats: Any = None,
) -> dict[str, Any]:
    """Delete orphan chunk rows whose plan header no longer exists.

    Orphans are the expected residue of a crash between the chunk writes and the
    header publish. Removing them is safe *once the write is definitely not still
    in flight*: chunks-first ordering means a compile that has written its chunks
    but not yet published its header looks exactly like an orphan, and a
    multi-million-target compile sits in that window for minutes. Chunks younger
    than ``grace`` (default :data:`ORPHAN_GRACE`) are therefore left alone and
    reported separately, so a concurrent compile is never sabotaged.

    Reads only ``plan_id``/``created_at`` from the chunk table -- never the id
    payload -- so reclaiming orphans does not re-read the whole catalog's
    membership.
    """
    window = ORPHAN_GRACE if grace is None else grace
    now = datetime.now(UTC)
    cutoff = now - window
    header_plans = _all_plan_ids(lake, stats=stats)
    orphan_ages: dict[str, datetime] = {}
    for batch in _stream_rows(
        lake, _CHUNK_TABLE, columns=("plan_id", "created_at"), stats=stats
    ):
        for row in batch:
            plan_id = str(row.get("plan_id") or "")
            if not plan_id or plan_id in header_plans:
                continue
            created = _coerce_dt(row.get("created_at")) or now
            previous = orphan_ages.get(plan_id)
            # Track the NEWEST chunk per plan: a plan is only safely orphaned once
            # every one of its chunks has aged out of the in-flight window.
            if previous is None or created > previous:
                orphan_ages[plan_id] = created
    reclaimable = sorted(plan_id for plan_id, ts in orphan_ages.items() if ts < cutoff)
    in_flight = sorted(plan_id for plan_id, ts in orphan_ages.items() if ts >= cutoff)
    deleted = 0
    failed: list[str] = []
    if reclaimable and not dry_run:
        # Re-read the header set immediately before deleting: a header published
        # between the scan above and this delete would otherwise leave a live plan
        # whose chunks we remove.
        published_since = _all_plan_ids(lake, stats=stats)
        raced = sorted(plan_id for plan_id in reclaimable if plan_id in published_since)
        if raced:
            reclaimable = [plan_id for plan_id in reclaimable if plan_id not in set(raced)]
            in_flight = sorted(set(in_flight) | set(raced))
    if reclaimable and not dry_run:
        deleted, failed = _delete_plan_chunks(lake, reclaimable)
    return {
        "dry_run": bool(dry_run),
        "grace_seconds": int(window.total_seconds()),
        "orphan_plan_ids": reclaimable,
        "orphan_plan_count": len(reclaimable),
        "skipped_in_flight_plan_ids": in_flight,
        "skipped_in_flight_count": len(in_flight),
        "chunks_deleted": deleted,
        "delete_failed_plan_ids": failed,
    }
