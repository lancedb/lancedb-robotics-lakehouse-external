"""Physical duplicate compaction for the published-view catalog (backlog 0507).

Concurrent identical publishers can land benign duplicate rows per key in the
view catalog: Lance arbitrates truly simultaneous same-key ``merge_insert``
inserts as non-conflicting appends (verified empirically in the 0490/0491
scale review). Every read path deduplicates by key, so the duplicates are
semantically invisible -- but they accumulate physically with no reclamation
path. :func:`compact_view_catalog` is that path, wired into ``lake maintain``
after the 0140 pattern.

Design, per table (``lerobot_views`` keyed by ``view_id``,
``lerobot_view_files`` by ``file_id``, ``lerobot_view_latest`` by ``repo_id``):

- **Bounded duplicate detection.** The primary path orders the scan by key in
  the backend, so duplicate copies are adjacent and detection needs O(1)
  client memory regardless of table size. When the backend cannot order the
  scan, a dict fallback is used under a loud row bound -- past it the table is
  *reported skipped*, never silently half-processed.
- **Canonical copy = newest by the table's semantic order.** For the header
  and file tables that is ``created_at``. For the pointer table it is
  ``(view_created_at, view_id, created_at)`` -- write-time ``created_at``
  alone could keep a *later-written duplicate of an older view* and regress
  the pointer.
- **Tier 1 (unique newest copy).** Delete strictly-older copies by key +
  order predicate. The newest copy is never touched, so a key can never drop
  to zero rows; a crash mid-delete leaves stragglers a re-run collapses.
- **Tier 2 (copies tied at the newest order value).** SQL cannot separate
  identical rows, so the 0135 aligned-tick pattern applies: re-add the
  canonical row stamped strictly newer (last order column + 1 microsecond),
  then delete ``key AND order < marker``. A crash after the add leaves an
  extra copy (re-run collapses it); a crash mid-delete leaves stragglers
  (re-run collapses them); the key never has zero rows. The <=1 microsecond
  ``created_at`` nudge on a view header is documented: it can only reorder
  against an exact-timestamp tie of a *different* view, where ordering was
  already arbitrary.
- **Postcondition (BUG-04 rule).** Every processed key is re-read (chunked
  ``IN`` predicates, 0145 discipline): a key with zero remaining rows raises
  :class:`ViewCatalogCompactionError` -- that is data loss, never reported as
  success. A key still holding >1 rows (a concurrent publisher re-raced the
  window) is *reported*, not raised: re-running converges.
- **Bounded work per run.** At most ``max_keys_per_run`` duplicated keys are
  processed per invocation; the report says how many remain, and re-running
  converges (resumable-by-rerun, no cursor state to persist).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from lancedb_robotics.schemas import (
    LEROBOT_VIEW_FILES_SCHEMA,
    LEROBOT_VIEW_LATEST_SCHEMA,
    LEROBOT_VIEWS_SCHEMA,
)

from .views import (
    _POINTER_UPDATE_CONDITION,
    VIEW_FILES_TABLE,
    VIEW_LATEST_TABLE,
    VIEWS_TABLE,
    ViewError,
    _merge_insert_with_retry,
    _sql_literal,
    _timestamp_literal,
)

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from lancedb_robotics.lake import Lake

#: Versioned report contract (0134/0135 convention).
COMPACTION_REPORT_VERSION = "lerobot-view-catalog-compaction/1"

#: Arrow batch size for the bounded detection scans.
_SCAN_BATCH_SIZE = 4_096

#: Duplicated keys processed per run; the report counts the remainder and a
#: re-run converges (bounded work per invocation, resumable by re-running).
_MAX_DUPLICATE_KEYS_PER_RUN = 10_000

#: Keys per chunked delete predicate / verification ``IN`` list.
_KEY_CHUNK = 32

#: Copies tracked per key before the key is reported skipped-oversized (a key
#: with this many physical duplicates is not a benign race artifact).
_MAX_COPIES_PER_KEY = 4_096

#: Row bound for the unordered (dict) detection fallback. Past it the table is
#: reported skipped rather than buffering an unbounded key map.
_MAX_UNORDERED_SCAN_ROWS = 100_000

#: Distinct repo_ids tracked per pointer-reconcile run; the remainder is
#: reported and a re-run converges.
_MAX_RECONCILE_REPOS = 100_000

#: (table, key column, semantic order columns, schema). Order columns choose
#: the canonical copy; see the module docstring for why the pointer table
#: orders by the pointed-at view before write time.
_TABLE_SPECS: tuple[tuple[str, str, tuple[str, ...], pa.Schema], ...] = (
    (VIEWS_TABLE, "view_id", ("created_at",), LEROBOT_VIEWS_SCHEMA),
    (VIEW_FILES_TABLE, "file_id", ("created_at",), LEROBOT_VIEW_FILES_SCHEMA),
    (
        VIEW_LATEST_TABLE,
        "repo_id",
        ("view_created_at", "view_id", "created_at"),
        LEROBOT_VIEW_LATEST_SCHEMA,
    ),
)


class ViewCatalogCompactionError(ViewError):
    """Raised when compaction cannot proceed safely or a postcondition fails."""


@dataclass(frozen=True)
class ViewCatalogTableCompaction:
    """Per-table outcome of one :func:`compact_view_catalog` run."""

    table: str
    key_column: str
    #: "compacted", "clean" (no duplicates), "absent" (table not in the lake),
    #: "skipped-unordered-over-bound" (backend cannot order the scan and the
    #: table is too large for the bounded dict fallback), or
    #: "skipped-ordered-scan-aborted" (the ordered scan failed mid-run; deletes
    #: already applied are safe and a re-run converges).
    status: str
    physical_rows_before: int = 0
    distinct_keys_seen: int = 0
    duplicate_keys_found: int = 0
    duplicate_keys_processed: int = 0
    duplicate_keys_remaining: int = 0
    rows_deleted: int = 0
    canonical_rows_readded: int = 0
    keys_still_duplicated: int = 0
    keys_skipped_oversized: int = 0
    keys_skipped_null_order: int = 0
    detail: str = ""

    def to_params(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "key_column": self.key_column,
            "status": self.status,
            "physical_rows_before": self.physical_rows_before,
            "distinct_keys_seen": self.distinct_keys_seen,
            "duplicate_keys_found": self.duplicate_keys_found,
            "duplicate_keys_processed": self.duplicate_keys_processed,
            "duplicate_keys_remaining": self.duplicate_keys_remaining,
            "rows_deleted": self.rows_deleted,
            "canonical_rows_readded": self.canonical_rows_readded,
            "keys_still_duplicated": self.keys_still_duplicated,
            "keys_skipped_oversized": self.keys_skipped_oversized,
            "keys_skipped_null_order": self.keys_skipped_null_order,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LatestPointerReconciliation:
    """Outcome of the pointer-vs-catalog reconcile pass (scale-review H1).

    A crash between a publish's catalog write and its pointer update -- or a
    failed pointer update whose stale-row removal also failed -- leaves
    ``lerobot_view_latest`` pointing at a *previous* view. That pointer is
    valid (the header exists), so the resolve chain would serve it silently
    forever. This pass compares every pointer row against a backend-ordered
    newest read of ``lerobot_views`` and repairs the losers.

    ``status``: "reconciled", "absent" (no pointer table), or
    "skipped-unordered" (the backend cannot order the header scan; a per-repo
    guarded scan inside maintenance would be unbounded, so the pass reports
    itself skipped instead).
    """

    status: str
    pointers_checked: int = 0
    pointers_repaired: int = 0
    dangling_pointers_removed: int = 0
    repairs_remaining: int = 0
    detail: str = ""

    def to_params(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "pointers_checked": self.pointers_checked,
            "pointers_repaired": self.pointers_repaired,
            "dangling_pointers_removed": self.dangling_pointers_removed,
            "repairs_remaining": self.repairs_remaining,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ViewCatalogCompactionReport:
    """Result of one :func:`compact_view_catalog` run."""

    report_version: str
    lake_uri: str
    dry_run: bool
    tables: tuple[ViewCatalogTableCompaction, ...]
    latest_pointer_reconciliation: LatestPointerReconciliation

    @property
    def rows_deleted(self) -> int:
        return sum(item.rows_deleted for item in self.tables)

    @property
    def converged(self) -> bool:
        """True when a re-run has nothing left to do."""
        return (
            all(
                item.duplicate_keys_remaining == 0
                and item.keys_still_duplicated == 0
                and item.keys_skipped_oversized == 0
                and not item.status.startswith("skipped")
                for item in self.tables
            )
            and self.latest_pointer_reconciliation.repairs_remaining == 0
        )

    def to_params(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "lake_uri": self.lake_uri,
            "dry_run": self.dry_run,
            "rows_deleted": self.rows_deleted,
            "converged": self.converged,
            "tables": [item.to_params() for item in self.tables],
            "latest_pointer_reconciliation": self.latest_pointer_reconciliation.to_params(),
        }


def _order_key(row: dict[str, Any], order_columns: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row[column] for column in order_columns)


def _order_literal(column: str, value: Any) -> str:
    if isinstance(value, datetime):
        return _timestamp_literal(value)
    return _sql_literal(value)


def _older_than_predicate(
    key_column: str, key: str, order_columns: tuple[str, ...], bound: tuple[Any, ...]
) -> str:
    """``key = K AND (order tuple) < (bound)``, lexicographic, typed literals."""
    branches: list[str] = []
    for depth, column in enumerate(order_columns):
        parts = [
            f"{order_columns[i]} = {_order_literal(order_columns[i], bound[i])}"
            for i in range(depth)
        ]
        parts.append(f"{column} < {_order_literal(column, bound[depth])}")
        branches.append("(" + " AND ".join(parts) + ")")
    ordered = " OR ".join(branches)
    return f"{key_column} = {_sql_literal(key)} AND ({ordered})"


def _null_safe_sort_key(copy: tuple[Any, ...]) -> tuple[tuple[bool, Any], ...]:
    # Rows written outside publish can carry null order values; they sort as
    # oldest instead of crashing tuple comparison. Keys whose *newest* copy has
    # a null order value are skipped (no typed SQL bound can be built).
    return tuple((value is not None, value) for value in copy)


@dataclass
class _DuplicatedKey:
    key: str
    copies: list[tuple[Any, ...]]
    oversized: bool = False

    @property
    def newest(self) -> tuple[Any, ...]:
        return max(self.copies, key=_null_safe_sort_key)

    @property
    def newest_tied(self) -> bool:
        newest = self.newest
        return sum(1 for copy in self.copies if copy == newest) > 1


def _iter_duplicated_keys_ordered(
    table: Any, key_column: str, order_columns: tuple[str, ...]
):
    """Yield duplicated keys from a backend key-ordered scan; O(1) client memory.

    Returns ``(iterator, stats)`` where stats is mutated during iteration, or
    ``None`` when the backend cannot order the scan.
    """
    try:
        from lancedb.query import ColumnOrdering

        batches = (
            table.search()
            .select([key_column, *order_columns])
            .order_by([ColumnOrdering(column_name=key_column, ascending=True)])
            .to_batches(batch_size=_SCAN_BATCH_SIZE)
        )
        iterator = iter(batches)
        first = next(iterator, None)
    except Exception:  # noqa: BLE001 - ordering unavailable; caller falls back.
        return None

    stats = {"physical_rows": 0, "distinct_keys": 0}

    def _generate():
        current: _DuplicatedKey | None = None

        def _flush(entry: _DuplicatedKey | None):
            if entry is not None and (len(entry.copies) > 1 or entry.oversized):
                return entry
            return None

        chunks = [first] if first is not None else []

        def _batches():
            yield from chunks
            yield from iterator

        try:
            for batch in _batches():
                for row in batch.to_pylist():
                    stats["physical_rows"] += 1
                    key = str(row[key_column])
                    if current is None or key != current.key:
                        flushed = _flush(current)
                        if flushed is not None:
                            yield flushed
                        stats["distinct_keys"] += 1
                        current = _DuplicatedKey(key=key, copies=[])
                    if len(current.copies) >= _MAX_COPIES_PER_KEY:
                        current.oversized = True
                    else:
                        current.copies.append(_order_key(row, order_columns))
        except Exception as exc:  # noqa: BLE001 - lazy ordering rejection mid-scan.
            # Stop cleanly and let the caller report the table skipped instead
            # of tracebacking out of maintenance; deletes already flushed are
            # individually safe and a re-run converges.
            stats["scan_error"] = str(exc)
            return
        flushed = _flush(current)
        if flushed is not None:
            yield flushed

    return _generate(), stats


def _iter_duplicated_keys_unordered(
    table: Any, key_column: str, order_columns: tuple[str, ...]
):
    """Dict-based duplicate detection under a loud row bound (order fallback)."""
    tracked: dict[str, _DuplicatedKey] = {}
    physical_rows = 0
    for batch in (
        table.search()
        .select([key_column, *order_columns])
        .to_batches(batch_size=_SCAN_BATCH_SIZE)
    ):
        for row in batch.to_pylist():
            physical_rows += 1
            if physical_rows > _MAX_UNORDERED_SCAN_ROWS:
                raise ViewCatalogCompactionError(
                    f"backend cannot order the {key_column}-keyed scan and the "
                    f"table holds more than {_MAX_UNORDERED_SCAN_ROWS} rows; "
                    "refusing an unbounded in-memory key map"
                )
            key = str(row[key_column])
            entry = tracked.get(key)
            if entry is None:
                entry = _DuplicatedKey(key=key, copies=[])
                tracked[key] = entry
            if len(entry.copies) >= _MAX_COPIES_PER_KEY:
                entry.oversized = True
            else:
                entry.copies.append(_order_key(row, order_columns))
    duplicated = [
        entry for entry in tracked.values() if len(entry.copies) > 1 or entry.oversized
    ]
    duplicated.sort(key=lambda entry: entry.key)
    stats = {"physical_rows": physical_rows, "distinct_keys": len(tracked)}
    return iter(duplicated), stats


def _bump_last_order_value(bound: tuple[Any, ...]) -> tuple[Any, ...]:
    last = bound[-1]
    if not isinstance(last, datetime):  # pragma: no cover - specs end in timestamps.
        raise ViewCatalogCompactionError(
            "tier-2 dedup requires the last order column to be a timestamp"
        )
    return (*bound[:-1], last + timedelta(microseconds=1))


def _readd_canonical_row(
    table: Any,
    schema: pa.Schema,
    key_column: str,
    key: str,
    order_columns: tuple[str, ...],
    newest: tuple[Any, ...],
) -> tuple[Any, ...]:
    """0135 tier-2 shape: append the canonical row stamped strictly newer.

    Reads one full copy at the tied-newest order value (bounded: one row, the
    only full-row read compaction ever does), bumps the last order column by
    one microsecond, and appends it. Returns the marker order tuple; the
    caller deletes ``key AND order < marker`` afterwards.
    """
    match = [f"{key_column} = {_sql_literal(key)}"]
    match.extend(
        f"{column} = {_order_literal(column, newest[i])}"
        for i, column in enumerate(order_columns)
    )
    rows = table.search().where(" AND ".join(match)).limit(1).to_arrow().to_pylist()
    if not rows:
        raise ViewCatalogCompactionError(
            f"canonical copy for key {key!r} vanished mid-compaction; "
            "re-run compaction"
        )
    canonical = dict(rows[0])
    marker = _bump_last_order_value(newest)
    canonical[order_columns[-1]] = marker[-1]
    table.add(pa.Table.from_pylist([canonical], schema=schema))
    return marker


def _verify_processed_keys(
    table: Any, key_column: str, keys: list[str]
) -> tuple[int, list[str]]:
    """Per-key surviving-row counts for processed keys (chunked IN reads).

    Returns ``(still_duplicated, lost_keys)``; the caller raises on any lost
    key -- a key with zero surviving rows is data loss, never a success.
    """
    still_duplicated = 0
    lost: list[str] = []
    for start in range(0, len(keys), _KEY_CHUNK):
        chunk = keys[start : start + _KEY_CHUNK]
        literals = ", ".join(_sql_literal(key) for key in chunk)
        counts = dict.fromkeys(chunk, 0)
        for batch in (
            table.search()
            .select([key_column])
            .where(f"{key_column} IN ({literals})")
            .to_batches(batch_size=_SCAN_BATCH_SIZE)
        ):
            for row in batch.to_pylist():
                key = str(row[key_column])
                if key in counts:
                    counts[key] += 1
        for key, count in counts.items():
            if count == 0:
                lost.append(key)
            elif count > 1:
                still_duplicated += 1
    return still_duplicated, lost


def _newest_header_ordered(lake: Lake, repo_id: str) -> tuple[str, Any] | None:
    """Newest ``(view_id, created_at)`` header for ``repo_id`` via ordered read.

    Returns ``("", None)`` when the repo has no headers, and ``None`` when the
    backend cannot order the scan (reconcile then reports itself skipped -- a
    per-repo guarded scan inside maintenance would be unbounded).
    """
    try:
        from lancedb.query import ColumnOrdering

        batches = (
            lake.table(VIEWS_TABLE)
            .search()
            .select(["view_id", "created_at"])
            .where(f"repo_id = {_sql_literal(repo_id)}")
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
    except Exception:  # noqa: BLE001 - ordering unavailable; caller skips.
        return None


def _reconcile_latest_pointers(
    lake: Lake, *, dry_run: bool, max_repairs: int
) -> LatestPointerReconciliation:
    """Repair ``lerobot_view_latest`` rows that disagree with the header catalog.

    See :class:`LatestPointerReconciliation` for why this exists (scale-review
    H1: a stale-but-valid pointer is served silently by the resolve chain).
    Bounded: pointer rows stream in batches, one backend-ordered top-1 header
    read per distinct repo_id, at most ``max_repairs`` writes per run. A
    pointer older than the newest header is repaired with the same newest-wins
    conditional upsert publish uses (safe under concurrent publishes); a
    pointer *newer than every header* references a view the catalog does not
    hold, so its rows are deleted and resolves fall back to the ordered read.
    """
    try:
        pointer_table = lake.table(VIEW_LATEST_TABLE)
    except Exception:  # noqa: BLE001 - pointer table absent on pre-0507 lakes.
        return LatestPointerReconciliation(status="absent")

    checked = repaired = dangling = remaining = 0
    seen: set[str] = set()
    for batch in (
        pointer_table.search()
        .select(["repo_id", "view_id", "view_created_at"])
        .to_batches(batch_size=_SCAN_BATCH_SIZE)
    ):
        for row in batch.to_pylist():
            repo_id = str(row["repo_id"])
            if repo_id in seen:
                continue
            if len(seen) >= _MAX_RECONCILE_REPOS:
                remaining += 1
                continue
            seen.add(repo_id)
            checked += 1
            newest = _newest_header_ordered(lake, repo_id)
            if newest is None:
                return LatestPointerReconciliation(
                    status="skipped-unordered",
                    pointers_checked=checked,
                    pointers_repaired=repaired,
                    dangling_pointers_removed=dangling,
                    repairs_remaining=remaining,
                    detail="backend cannot order the header scan",
                )
            newest_id, newest_created = newest
            pointer_key = (row["view_created_at"], str(row["view_id"]))
            header_key = (newest_created, newest_id)
            if newest_id and header_key == pointer_key:
                continue
            if repaired + dangling >= max_repairs:
                remaining += 1
                continue
            if not newest_id or header_key < pointer_key:
                # No headers at all, or the pointer claims something newer than
                # every header: it references a view the catalog cannot serve.
                if not dry_run:
                    pointer_table.delete(f"repo_id = {_sql_literal(repo_id)}")
                dangling += 1
                continue
            if not dry_run:
                corrected = pa.Table.from_pylist(
                    [
                        {
                            "repo_id": repo_id,
                            "view_id": newest_id,
                            "view_created_at": newest_created,
                            "created_at": datetime.now(UTC),
                        }
                    ],
                    schema=LEROBOT_VIEW_LATEST_SCHEMA,
                )
                _merge_insert_with_retry(
                    pointer_table,
                    "repo_id",
                    corrected,
                    update_condition=_POINTER_UPDATE_CONDITION,
                )
            repaired += 1
    return LatestPointerReconciliation(
        status="reconciled",
        pointers_checked=checked,
        pointers_repaired=repaired,
        dangling_pointers_removed=dangling,
        repairs_remaining=remaining,
    )


def _compact_table(
    lake: Lake,
    table_name: str,
    key_column: str,
    order_columns: tuple[str, ...],
    schema: pa.Schema,
    *,
    dry_run: bool,
    max_keys_per_run: int,
) -> ViewCatalogTableCompaction:
    try:
        table = lake.table(table_name)
    except Exception:  # noqa: BLE001 - pointer table absent on pre-0507 lakes.
        return ViewCatalogTableCompaction(
            table=table_name, key_column=key_column, status="absent"
        )

    ordered = _iter_duplicated_keys_ordered(table, key_column, order_columns)
    if ordered is not None:
        duplicated_iter, stats = ordered
    else:
        try:
            duplicated_iter, stats = _iter_duplicated_keys_unordered(
                table, key_column, order_columns
            )
        except ViewCatalogCompactionError as exc:
            return ViewCatalogTableCompaction(
                table=table_name,
                key_column=key_column,
                status="skipped-unordered-over-bound",
                detail=str(exc),
            )

    duplicate_keys_found = 0
    processed_keys: list[str] = []
    keys_skipped_oversized = 0
    keys_skipped_null_order = 0
    rows_deleted = 0
    canonical_rows_readded = 0
    remaining = 0
    # Chunked tier-1 deletes: (predicate, planned-deletions) buffered per chunk
    # so each table.delete commit stays bounded.
    pending_predicates: list[str] = []
    pending_deletions = 0

    def _flush_deletes():
        nonlocal pending_predicates, pending_deletions, rows_deleted
        if pending_predicates:
            if not dry_run:
                table.delete(" OR ".join(f"({p})" for p in pending_predicates))
            rows_deleted += pending_deletions
            pending_predicates, pending_deletions = [], 0

    for entry in duplicated_iter:
        duplicate_keys_found += 1
        if entry.oversized:
            keys_skipped_oversized += 1
            continue
        newest = entry.newest
        if any(value is None for value in newest):
            # A copy stamped outside publish carries a null order value: no
            # typed SQL bound can be built, so the key is skipped and reported.
            keys_skipped_null_order += 1
            continue
        if len(processed_keys) >= max_keys_per_run:
            remaining += 1
            continue
        processed_keys.append(entry.key)
        if entry.newest_tied:
            # Tier 2: identical newest copies -- re-add strictly newer, then
            # delete everything older than the marker.
            if not dry_run:
                marker = _readd_canonical_row(
                    table, schema, key_column, entry.key, order_columns, newest
                )
            else:
                marker = _bump_last_order_value(newest)
            canonical_rows_readded += 1
            pending_predicates.append(
                _older_than_predicate(key_column, entry.key, order_columns, marker)
            )
            pending_deletions += len(entry.copies)
        else:
            # Tier 1: strictly-older copies only; the newest copy is untouched.
            pending_predicates.append(
                _older_than_predicate(key_column, entry.key, order_columns, newest)
            )
            pending_deletions += len(entry.copies) - 1
        if len(pending_predicates) >= _KEY_CHUNK:
            _flush_deletes()
    _flush_deletes()

    still_duplicated = 0
    if processed_keys and not dry_run:
        still_duplicated, lost = _verify_processed_keys(
            table, key_column, processed_keys
        )
        if lost:
            raise ViewCatalogCompactionError(
                f"compaction postcondition failed on {table_name!r}: keys "
                f"{lost[:8]!r} have zero surviving rows. This should be "
                "impossible (deletes are bounded strictly below the kept "
                "copy); re-publish the affected views before trusting this "
                "catalog"
            )

    scan_error = str(stats.get("scan_error") or "")
    if scan_error:
        status = "skipped-ordered-scan-aborted"
    else:
        status = "compacted" if duplicate_keys_found else "clean"
    return ViewCatalogTableCompaction(
        table=table_name,
        key_column=key_column,
        status=status,
        physical_rows_before=int(stats["physical_rows"]),
        distinct_keys_seen=int(stats["distinct_keys"]),
        duplicate_keys_found=duplicate_keys_found,
        duplicate_keys_processed=len(processed_keys),
        duplicate_keys_remaining=remaining,
        rows_deleted=rows_deleted,
        canonical_rows_readded=canonical_rows_readded,
        keys_still_duplicated=still_duplicated,
        keys_skipped_oversized=keys_skipped_oversized,
        keys_skipped_null_order=keys_skipped_null_order,
        detail=scan_error,
    )


def compact_view_catalog(
    lake: Lake,
    *,
    dry_run: bool = False,
    max_keys_per_run: int = _MAX_DUPLICATE_KEYS_PER_RUN,
) -> ViewCatalogCompactionReport:
    """Collapse physical duplicate rows in the published-view catalog tables.

    Safe by construction: only rows *strictly older than the kept canonical
    copy* of a duplicated key are ever deleted, distinct keys are asserted to
    survive (zero-row keys raise), and every mechanism converges under crashes
    and concurrent publishers by re-running. ``dry_run`` reports the same plan
    without writing. See the module docstring for the full design.
    """
    tables = tuple(
        _compact_table(
            lake,
            table_name,
            key_column,
            order_columns,
            schema,
            dry_run=dry_run,
            max_keys_per_run=max_keys_per_run,
        )
        for table_name, key_column, order_columns, schema in _TABLE_SPECS
    )
    # After physical dedup, repair pointers a crash window left stale
    # (scale-review H1); dedup first so reconcile sees one row per repo_id.
    reconciliation = _reconcile_latest_pointers(
        lake, dry_run=dry_run, max_repairs=max_keys_per_run
    )
    return ViewCatalogCompactionReport(
        report_version=COMPACTION_REPORT_VERSION,
        lake_uri=lake.uri,
        dry_run=dry_run,
        tables=tables,
        latest_pointer_reconciliation=reconciliation,
    )
