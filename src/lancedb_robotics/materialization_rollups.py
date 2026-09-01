"""Scalable curation materialization accounting catalog and rollups (backlog 0145).

Backlog 0083 stores one copy-accounting report per projection / live / plan /
export operation in ``curation_materializations``, keyed on a content-addressed
``materialization_id`` with the full accounting body in a ``report_json`` cell.
That keeps the vertical slice simple, but a large curation program generates many
repeated plan comparisons, export attempts, benchmark runs, and external handoffs
across teams and snapshots. At hundreds of thousands of reports the compare
"materialization" summary streamed *every* source row and parsed ``report_json``
per row just to recover planned bytes -- an unbounded, JSON-bound scan.

This module is the scalable layer on top of that compat table:

- ``curation_materialization_rollups`` promotes every byte/count field and every
  identity column (dataset, snapshot/branch, target format, mode, transform
  lineage, ``created_at``) into dedicated, scalar-indexable columns. Snapshot,
  branch, and format rollups plus paged history queries push their filters down
  and aggregate over promoted columns -- never parsing a JSON blob in client
  memory. The catalog is fully rebuildable from ``curation_materializations``
  (:func:`sync_materialization_rollups`), and every ``materialization_report``
  write emits the matching rollup row inline (0098 zero-divergence).
- ``curation_materialization_files`` holds optional per-output-file accounting
  chunks for large exports, derived from the 0144 object-store reconciliation
  object list and written in bounded batches so a big file list never becomes one
  oversized commit. It keeps the rollup row light while making per-file cost
  queryable.
- Retention (:func:`prune_materialization_rollups`) compacts *superseded plan /
  dry-run reports only*. Completed export evidence is never pruned. Pruning is
  safe-delete: the source ``report_json`` and the per-file chunks are cleared,
  while the promoted rollup columns and ``report_sha1`` survive as audit
  evidence.

Every read here follows the repo's bounded-streaming discipline (SKILLS.md §2):
a required projection, batched ``to_batches`` folds, engine ``order_by`` keyset
pagination with a bounded top-``page_size+1`` heap fallback, and never a
``to_arrow().to_pylist()`` over an unbounded scan. Writes are atomic
``merge_insert`` upserts with a bounded commit-conflict retry (BUG-04).
"""

from __future__ import annotations

import base64
import hashlib
import heapq
import json
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from lancedb_robotics.schemas import (
    CURATION_MATERIALIZATION_FILES_SCHEMA,
    CURATION_MATERIALIZATION_ROLLUPS_SCHEMA,
)

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from lancedb_robotics.lake import Lake

ROLLUP_SCHEMA_VERSION = "lancedb-robotics/curation-materialization-rollup/v1"

_SOURCE_TABLE = "curation_materializations"
_ROLLUP_TABLE = "curation_materialization_rollups"
_FILES_TABLE = "curation_materialization_files"

#: Rollup lifecycle states (kept off the compat source row -- see retention).
STATE_ACTIVE = "active"
STATE_SUPERSEDED = "superseded"
STATE_PRUNED = "pruned"
STATES: tuple[str, ...] = (STATE_ACTIVE, STATE_SUPERSEDED, STATE_PRUNED)

#: Reports eligible for retention/compaction. Completed export evidence and
#: logical-reference projection records are protected; only plan/dry-run reports
#: (which are re-generated repeatedly during candidate comparison) are pruned.
PRUNABLE_MODES: frozenset[str] = frozenset({"plan", "dry-run"})

#: Deterministic page ordering for history queries.
HISTORY_ORDER_COLUMNS: tuple[str, str] = ("created_at", "materialization_id")

_DEFAULT_PAGE_SIZE = 50
_MAX_PAGE_SIZE = 1000
_SCAN_BATCH = 2048
#: Bounded per-commit file-chunk batch so a 10^5-10^6-object export never stages
#: one oversized commit (BUG-02).
_FILE_CHUNK_BATCH = 512
_MERGE_INSERT_ATTEMPTS = 3

#: Light projection for summary/rollup aggregation -- never includes a JSON body.
_ROLLUP_SUMMARY_COLUMNS: tuple[str, ...] = (
    "materialization_id",
    "dataset_id",
    "snapshot_name",
    "target_format",
    "output_uri",
    "mode",
    "payload_copy_policy",
    "reconciliation_status",
    "state",
    "selected_scenario_count",
    "selected_observation_count",
    "total_payload_bytes",
    "copied_payload_bytes",
    "logical_reference_bytes",
    "planned_payload_bytes",
    "metadata_bytes_written",
    "copy_ratio",
    "output_file_count",
    "output_file_bytes",
    "captured_file_count",
    "report_sha1",
    "report_bytes",
    "source_report_available",
    "projection_transform_id",
    "transform_id",
    "superseded_by",
    "created_by",
    "created_at",
)

#: Columns read for retention (adds pruned/superseded metadata to the summary set).
_RETENTION_COLUMNS: tuple[str, ...] = _ROLLUP_SUMMARY_COLUMNS + (
    "superseded_at",
    "pruned_at",
    "retention_policy_json",
    "source_table_versions",
)


class MaterializationRollupError(Exception):
    """Raised when a rollup catalog query or retention operation cannot proceed."""


# --------------------------------------------------------------------------- #
# Small house-convention helpers.
# --------------------------------------------------------------------------- #


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _digest(payload: Any) -> str:
    return hashlib.sha1(_json_dumps(payload).encode()).hexdigest()[:20]


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _coerce_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


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

    ``update_matched=False`` is insert-only so concurrent writers writing the
    same content-addressed rows converge without duplicates; ``update_matched=
    True`` replaces the matched row in one commit (lifecycle / clear-body flips).
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
    raise MaterializationRollupError(
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

    Falls back to a single materialized scan only when the streaming query cannot
    be built; callers re-apply their own row predicate so the fallback stays
    correct. Mirrors ``curate._stream_table_rows``.
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
        try:
            for batch in batches:
                rows = batch.to_pylist()
                _record_batch(stats, table, len(rows))
                yield rows
            return
        except Exception:
            pass
    rows = handle.to_arrow().to_pylist()
    if projected:
        wanted = set(projected)
        rows = [{k: v for k, v in row.items() if k in wanted} for row in rows]
    _record_full_scan(stats, table, len(rows))
    yield rows


def _table_has_rows(lake: Lake, table: str, where_sql: str | None = None) -> bool:
    handle = lake.table(table)
    try:
        return int(handle.count_rows(where_sql) if where_sql else handle.count_rows()) > 0
    except Exception:
        for batch in _stream_rows(lake, table, columns=("materialization_id",), where_sql=where_sql):
            if batch:
                return True
        return False


def _count_rows(lake: Lake, table: str, where_sql: str | None = None) -> int:
    handle = lake.table(table)
    try:
        return int(handle.count_rows(where_sql) if where_sql else handle.count_rows())
    except Exception:
        total = 0
        for batch in _stream_rows(lake, table, columns=("materialization_id",), where_sql=where_sql):
            total += len(batch)
        return total


#: Chunk width for ``<id> IN (...)`` reads/deletes. Matches the BUG-13 lineage
#: frontier chunk width: wide enough to amortize per-query cost, narrow enough
#: that the ``IsIn`` predicate never blows up the query planner (SKILLS.md §2).
_IN_CHUNK = 512


def _chunked(seq: Sequence[str], size: int = _IN_CHUNK) -> Iterable[list[str]]:
    items = list(seq)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _in_clause(column: str, ids: Sequence[str]) -> str:
    return f"{column} IN (" + ", ".join(_sql_literal(x) for x in ids) + ")"


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
    wanted = list(dict.fromkeys(str(i) for i in ids))
    out: dict[str, dict[str, Any]] = {}
    if not wanted:
        return out
    wanted_set = set(wanted)
    for group in _chunked(wanted):
        clause = _in_clause(id_column, group)
        for batch in _stream_rows(lake, table, columns=columns, where_sql=clause):
            for row in batch:
                rid = str(row.get(id_column) or "")
                if rid in wanted_set:
                    out[rid] = row
    return out


# --------------------------------------------------------------------------- #
# Dataclasses.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MaterializationRollupEntry:
    """One indexed rollup catalog row (light: promoted columns, no JSON body)."""

    materialization_id: str
    dataset_id: str
    snapshot_name: str
    target_format: str
    output_uri: str
    mode: str
    payload_copy_policy: str
    reconciliation_status: str
    state: str
    selected_scenario_count: int
    selected_observation_count: int
    total_payload_bytes: int
    copied_payload_bytes: int
    logical_reference_bytes: int
    planned_payload_bytes: int
    metadata_bytes_written: int
    copy_ratio: float
    output_file_count: int
    output_file_bytes: int
    captured_file_count: int
    report_sha1: str
    report_bytes: int
    source_report_available: bool
    projection_transform_id: str
    transform_id: str
    superseded_by: str
    created_by: str
    created_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "materialization_id": self.materialization_id,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "target_format": self.target_format,
            "output_uri": self.output_uri,
            "mode": self.mode,
            "payload_copy_policy": self.payload_copy_policy,
            "reconciliation_status": self.reconciliation_status,
            "state": self.state,
            "selected_scenario_count": self.selected_scenario_count,
            "selected_observation_count": self.selected_observation_count,
            "total_payload_bytes": self.total_payload_bytes,
            "copied_payload_bytes": self.copied_payload_bytes,
            "logical_reference_bytes": self.logical_reference_bytes,
            "planned_payload_bytes": self.planned_payload_bytes,
            "metadata_bytes_written": self.metadata_bytes_written,
            "copy_ratio": self.copy_ratio,
            "output_file_count": self.output_file_count,
            "output_file_bytes": self.output_file_bytes,
            "captured_file_count": self.captured_file_count,
            "report_sha1": self.report_sha1,
            "report_bytes": self.report_bytes,
            "source_report_available": self.source_report_available,
            "projection_transform_id": self.projection_transform_id,
            "transform_id": self.transform_id,
            "superseded_by": self.superseded_by,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass(frozen=True)
class MaterializationRollupPage:
    """One deterministic page of rollup entries, keyset-paged by created_at+id."""

    records: tuple[MaterializationRollupEntry, ...]
    page_size: int
    page_index: int
    cursor: str | None
    next_cursor: str | None
    has_more: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [entry.to_dict() for entry in self.records],
            "count": len(self.records),
            "page_size": self.page_size,
            "page_index": self.page_index,
            "cursor": self.cursor,
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
        }


@dataclass(frozen=True)
class MaterializationRollupBucket:
    """Aggregated copy-cost for one grouping key (branch / format / mode / day)."""

    key: str
    materialization_count: int
    total_payload_bytes: int
    copied_payload_bytes: int
    logical_reference_bytes: int
    planned_payload_bytes: int
    metadata_bytes_written: int
    copy_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "materialization_count": self.materialization_count,
            "total_payload_bytes": self.total_payload_bytes,
            "copied_payload_bytes": self.copied_payload_bytes,
            "logical_reference_bytes": self.logical_reference_bytes,
            "planned_payload_bytes": self.planned_payload_bytes,
            "metadata_bytes_written": self.metadata_bytes_written,
            "copy_ratio": self.copy_ratio,
        }


@dataclass(frozen=True)
class MaterializationRollup:
    """Aggregate copy-cost rollup over the indexed catalog (no JSON parsed)."""

    scope: dict[str, Any]
    materialization_count: int
    total_payload_bytes: int
    copied_payload_bytes: int
    logical_reference_bytes: int
    planned_payload_bytes: int
    metadata_bytes_written: int
    copy_ratio: float
    output_file_count: int
    output_file_bytes: int
    group_by: str | None
    buckets: tuple[MaterializationRollupBucket, ...]
    bounded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "materialization_count": self.materialization_count,
            "total_payload_bytes": self.total_payload_bytes,
            "copied_payload_bytes": self.copied_payload_bytes,
            "logical_reference_bytes": self.logical_reference_bytes,
            "planned_payload_bytes": self.planned_payload_bytes,
            "metadata_bytes_written": self.metadata_bytes_written,
            "copy_ratio": self.copy_ratio,
            "output_file_count": self.output_file_count,
            "output_file_bytes": self.output_file_bytes,
            "group_by": self.group_by,
            "buckets": [bucket.to_dict() for bucket in self.buckets],
            "bounded": self.bounded,
        }


@dataclass(frozen=True)
class MaterializationOutputFile:
    """One per-output-file accounting chunk row."""

    file_accounting_id: str
    materialization_id: str
    dataset_id: str
    chunk_index: int
    relative_path: str
    uri: str
    content_length: int
    classification: str
    container: str
    compression: str
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_accounting_id": self.file_accounting_id,
            "materialization_id": self.materialization_id,
            "dataset_id": self.dataset_id,
            "chunk_index": self.chunk_index,
            "relative_path": self.relative_path,
            "uri": self.uri,
            "content_length": self.content_length,
            "classification": self.classification,
            "container": self.container,
            "compression": self.compression,
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class MaterializationRetentionReport:
    """Outcome of a rollup retention / plan-compaction pass."""

    dry_run: bool
    retain_latest: int
    older_than: str | None
    scanned_count: int
    protected_count: int
    active_ids: tuple[str, ...] = field(default_factory=tuple)
    superseded_ids: tuple[str, ...] = field(default_factory=tuple)
    pruned_ids: tuple[str, ...] = field(default_factory=tuple)
    body_bytes_before: int = 0
    body_bytes_after: int = 0
    file_chunks_deleted: int = 0

    @property
    def superseded_count(self) -> int:
        return len(self.superseded_ids)

    @property
    def pruned_count(self) -> int:
        return len(self.pruned_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "retain_latest": self.retain_latest,
            "older_than": self.older_than,
            "scanned_count": self.scanned_count,
            "protected_count": self.protected_count,
            "active_ids": list(self.active_ids),
            "superseded_ids": list(self.superseded_ids),
            "pruned_ids": list(self.pruned_ids),
            "superseded_count": self.superseded_count,
            "pruned_count": self.pruned_count,
            "body_bytes_before": self.body_bytes_before,
            "body_bytes_after": self.body_bytes_after,
            "file_chunks_deleted": self.file_chunks_deleted,
        }


@dataclass(frozen=True)
class MaterializationRollupSyncReport:
    """Outcome of rebuilding the rollup catalog from the compat source table."""

    source_count: int
    rebuilt_count: int
    preserved_pruned_count: int
    indexes_built: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_count": self.source_count,
            "rebuilt_count": self.rebuilt_count,
            "preserved_pruned_count": self.preserved_pruned_count,
            "indexes_built": list(self.indexes_built),
        }


# --------------------------------------------------------------------------- #
# Cursor codec (mirrors 0142 membership history).
# --------------------------------------------------------------------------- #


def _encode_cursor(created_at: datetime | None, materialization_id: str) -> str:
    payload = {
        "created_at": created_at.isoformat() if created_at else "",
        "materialization_id": materialization_id,
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _decode_cursor(token: str) -> tuple[datetime | None, str]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
        created = payload.get("created_at") or ""
        return (_coerce_dt(created) if created else None, str(payload["materialization_id"]))
    except Exception as exc:  # noqa: BLE001 - a bad cursor is a caller error.
        raise MaterializationRollupError(f"invalid materialization history cursor: {token!r}") from exc


def _row_key(row: dict[str, Any]) -> tuple[datetime, str]:
    created = _coerce_dt(row.get("created_at")) or datetime.min.replace(tzinfo=UTC)
    return (created, str(row.get("materialization_id") or ""))


# --------------------------------------------------------------------------- #
# Row derivation from the compat source table.
# --------------------------------------------------------------------------- #


def _derive_reconciliation(report: dict[str, Any], reconciliation: dict[str, Any] | None) -> dict[str, Any]:
    if reconciliation:
        return reconciliation
    embedded = report.get("reconciliation")
    return embedded if isinstance(embedded, dict) else {}


def _planned_bytes(report: dict[str, Any]) -> int:
    accounting = report.get("accounting") or {}
    return _as_int(
        accounting.get("payload_bytes_planned")
        or report.get("payload_bytes_planned")
        or report.get("planned_payload_bytes")
    )


def rollup_row_from_source(
    source_row: dict[str, Any],
    *,
    reconciliation: dict[str, Any] | None = None,
    existing: dict[str, Any] | None = None,
    created_by: str = "lancedb-robotics",
    captured_file_count: int | None = None,
) -> dict[str, Any]:
    """Build a rollup catalog row dict from a ``curation_materializations`` row.

    Parses ``report_json`` exactly once (here at write/sync time) to lift planned
    bytes, copy policy, and reconciliation status into promoted columns, so every
    later summary/rollup read is JSON-free. ``existing`` preserves the lifecycle
    (``state``/``superseded_*``/``pruned_at``/retention policy) and the original
    ``created_by`` across a resync so a rebuild never resurrects a pruned body or
    resets a superseded plan.
    """
    report_json_str = str(source_row.get("report_json") or "")
    try:
        report = json.loads(report_json_str) if report_json_str else {}
    except json.JSONDecodeError:
        report = {}
    accounting = report.get("accounting") or {}
    recon = _derive_reconciliation(report, reconciliation)

    total = _as_int(source_row.get("total_payload_bytes"))
    copied = _as_int(source_row.get("copied_payload_bytes"))
    logical = _as_int(source_row.get("logical_reference_bytes"))
    planned = _planned_bytes(report)
    metadata = _as_int(source_row.get("metadata_bytes_written"))
    copy_ratio = _as_float(source_row.get("copy_ratio"))

    payload_copy_policy = str(
        accounting.get("payload_copy_policy")
        or report.get("payload_copy_policy")
        or (
            "would-copy-payloads"
            if str(source_row.get("mode")) in {"plan", "dry-run"} and planned
            else ("materialized-copy" if copied else "logical-reference")
        )
    )
    reconciliation_status = str(
        report.get("reconciliation_status") or recon.get("status") or ""
    )
    output_file_count = _as_int(recon.get("object_count"))
    output_file_bytes = _as_int(recon.get("total_object_bytes"))

    source_report_available = bool(report_json_str)
    report_sha1 = _sha1_text(report_json_str) if source_report_available else str(
        (existing or {}).get("report_sha1") or ""
    )
    report_bytes = (
        len(report_json_str.encode())
        if source_report_available
        else _as_int((existing or {}).get("report_bytes"))
    )

    existing = existing or {}
    state = str(existing.get("state") or STATE_ACTIVE)
    if state not in STATES:
        state = STATE_ACTIVE
    resolved_captured = (
        captured_file_count
        if captured_file_count is not None
        else _as_int(existing.get("captured_file_count"))
    )
    return {
        "materialization_id": str(source_row.get("materialization_id") or ""),
        "dataset_id": str(source_row.get("dataset_id") or ""),
        "snapshot_name": str(source_row.get("snapshot_name") or ""),
        "target_format": str(source_row.get("target_format") or ""),
        "output_uri": str(source_row.get("output_uri") or ""),
        "mode": str(source_row.get("mode") or ""),
        "payload_copy_policy": payload_copy_policy,
        "reconciliation_status": reconciliation_status,
        "state": state,
        "selected_scenario_count": _as_int(source_row.get("selected_scenario_count")),
        "selected_observation_count": _as_int(source_row.get("selected_observation_count")),
        "total_payload_bytes": total,
        "copied_payload_bytes": copied,
        "logical_reference_bytes": logical,
        "planned_payload_bytes": planned,
        "metadata_bytes_written": metadata,
        "copy_ratio": copy_ratio,
        "output_file_count": output_file_count,
        "output_file_bytes": output_file_bytes,
        "captured_file_count": resolved_captured,
        "report_sha1": report_sha1,
        "report_bytes": report_bytes,
        "source_report_available": source_report_available,
        "source_table_versions": list(source_row.get("source_table_versions") or ()),
        "projection_transform_id": str(source_row.get("projection_transform_id") or ""),
        "transform_id": str(source_row.get("transform_id") or ""),
        "superseded_by": str(existing.get("superseded_by") or ""),
        "superseded_at": _coerce_dt(existing.get("superseded_at")),
        "pruned_at": _coerce_dt(existing.get("pruned_at")),
        "retention_policy_json": str(existing.get("retention_policy_json") or ""),
        "created_by": str(existing.get("created_by") or source_row.get("created_by") or created_by),
        "created_at": _coerce_dt(source_row.get("created_at")),
    }


def _entry_from_row(row: dict[str, Any]) -> MaterializationRollupEntry:
    return MaterializationRollupEntry(
        materialization_id=str(row.get("materialization_id") or ""),
        dataset_id=str(row.get("dataset_id") or ""),
        snapshot_name=str(row.get("snapshot_name") or ""),
        target_format=str(row.get("target_format") or ""),
        output_uri=str(row.get("output_uri") or ""),
        mode=str(row.get("mode") or ""),
        payload_copy_policy=str(row.get("payload_copy_policy") or ""),
        reconciliation_status=str(row.get("reconciliation_status") or ""),
        state=str(row.get("state") or STATE_ACTIVE),
        selected_scenario_count=_as_int(row.get("selected_scenario_count")),
        selected_observation_count=_as_int(row.get("selected_observation_count")),
        total_payload_bytes=_as_int(row.get("total_payload_bytes")),
        copied_payload_bytes=_as_int(row.get("copied_payload_bytes")),
        logical_reference_bytes=_as_int(row.get("logical_reference_bytes")),
        planned_payload_bytes=_as_int(row.get("planned_payload_bytes")),
        metadata_bytes_written=_as_int(row.get("metadata_bytes_written")),
        copy_ratio=_as_float(row.get("copy_ratio")),
        output_file_count=_as_int(row.get("output_file_count")),
        output_file_bytes=_as_int(row.get("output_file_bytes")),
        captured_file_count=_as_int(row.get("captured_file_count")),
        report_sha1=str(row.get("report_sha1") or ""),
        report_bytes=_as_int(row.get("report_bytes")),
        source_report_available=bool(row.get("source_report_available")),
        projection_transform_id=str(row.get("projection_transform_id") or ""),
        transform_id=str(row.get("transform_id") or ""),
        superseded_by=str(row.get("superseded_by") or ""),
        created_by=str(row.get("created_by") or ""),
        created_at=_coerce_dt(row.get("created_at")),
    )


# --------------------------------------------------------------------------- #
# Per-output-file accounting chunks.
# --------------------------------------------------------------------------- #


def output_files_from_reconciliation(
    reconciliation: dict[str, Any] | None,
    *,
    materialization_id: str,
    dataset_id: str,
) -> list[dict[str, Any]]:
    """Build per-file accounting rows from a 0144 reconciliation object list.

    Only the objects actually carried in the reconciliation dict are captured
    (the durable materialization row embeds a bounded sample, cap 1000); the true
    total lives in the rollup ``output_file_count``. Rows are content-addressed by
    ``(materialization_id, relative_path)`` so a re-run is idempotent.
    """
    if not reconciliation:
        return []
    objects = reconciliation.get("objects")
    if not isinstance(objects, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        relative_path = str(obj.get("relative_path") or obj.get("uri") or index)
        rows.append(
            {
                "file_accounting_id": "matf-" + _digest(
                    {"materialization_id": materialization_id, "relative_path": relative_path}
                ),
                "materialization_id": materialization_id,
                "dataset_id": dataset_id,
                "chunk_index": index // _FILE_CHUNK_BATCH,
                "relative_path": relative_path,
                "uri": str(obj.get("uri") or ""),
                "content_length": _as_int(obj.get("content_length")),
                "classification": str(obj.get("classification") or ""),
                "container": str(obj.get("container") or ""),
                "compression": str(obj.get("compression") or ""),
                "checksum": str(obj.get("checksum") or ""),
            }
        )
    return rows


def _write_file_chunks(
    lake: Lake,
    file_rows: list[dict[str, Any]],
    *,
    created_at: datetime,
    transform_id: str,
) -> int:
    """Write per-file accounting rows in bounded merge_insert batches."""
    if not file_rows:
        return 0
    table = lake.table(_FILES_TABLE)
    written = 0
    for start in range(0, len(file_rows), _FILE_CHUNK_BATCH):
        batch = file_rows[start : start + _FILE_CHUNK_BATCH]
        payload = [
            {**row, "transform_id": transform_id, "created_at": created_at} for row in batch
        ]
        data = pa.Table.from_pylist(payload, schema=CURATION_MATERIALIZATION_FILES_SCHEMA)
        _merge_insert_with_retry(table, "file_accounting_id", data, update_matched=True)
        written += len(batch)
    return written


# --------------------------------------------------------------------------- #
# Inline write (called by curate.materialization_report).
# --------------------------------------------------------------------------- #


def write_rollup(
    lake: Lake,
    source_row: dict[str, Any],
    *,
    reconciliation: dict[str, Any] | None = None,
    created_by: str = "lancedb-robotics",
) -> dict[str, Any]:
    """Emit the rollup row (and any per-file chunks) for one source report inline.

    Idempotent: keyed on the content-addressed ``materialization_id``. Preserves
    an existing lifecycle so re-recording the same report never resurrects a
    pruned body. Best-effort per-file chunk capture from a supplied reconciliation
    object list. Returns the rollup row dict written.
    """
    materialization_id = str(source_row.get("materialization_id") or "")
    dataset_id = str(source_row.get("dataset_id") or "")
    existing = _fetch_rollup_row(lake, materialization_id)
    created_at = _coerce_dt(source_row.get("created_at")) or datetime.now(UTC)

    file_rows = output_files_from_reconciliation(
        reconciliation, materialization_id=materialization_id, dataset_id=dataset_id
    )
    captured = 0
    if file_rows:
        captured = _write_file_chunks(
            lake,
            file_rows,
            created_at=created_at,
            transform_id=str(source_row.get("transform_id") or ""),
        )

    row = rollup_row_from_source(
        source_row,
        reconciliation=reconciliation,
        existing=existing,
        created_by=created_by,
        captured_file_count=captured if file_rows else None,
    )
    table = lake.table(_ROLLUP_TABLE)
    data = pa.Table.from_pylist([row], schema=CURATION_MATERIALIZATION_ROLLUPS_SCHEMA)
    _merge_insert_with_retry(table, "materialization_id", data, update_matched=True)
    return row


def _fetch_rollup_row(lake: Lake, materialization_id: str) -> dict[str, Any] | None:
    if not materialization_id:
        return None
    where = f"materialization_id = {_sql_literal(materialization_id)}"
    for batch in _stream_rows(lake, _ROLLUP_TABLE, columns=_RETENTION_COLUMNS, where_sql=where):
        for row in batch:
            if str(row.get("materialization_id") or "") == materialization_id:
                return row
    return None


# --------------------------------------------------------------------------- #
# Sync / backfill from the compat source table.
# --------------------------------------------------------------------------- #


def sync_materialization_rollups(
    lake: Lake,
    *,
    build_indexes: bool = True,
    created_by: str = "lancedb-robotics",
) -> MaterializationRollupSyncReport:
    """Rebuild ``curation_materialization_rollups`` from ``curation_materializations``.

    Streams the source rows in bounded batches, re-derives each rollup row
    deterministically, and preserves the lifecycle metadata (state / supersession
    / pruned body) and original ``created_by`` for rows a prior retention pass
    already acted on. Run once on a lake that recorded materializations before
    0145, or any time to repair catalog drift; ``materialization_report`` keeps
    the catalog current automatically.

    Bounded memory: the source is streamed in batches and the existing rollup
    rows needed to preserve lifecycle are fetched per batch via a chunked
    ``materialization_id IN (...)`` lookup -- neither the source nor the rollup
    catalog is ever collected whole (SKILLS.md §2).
    """
    source_count = 0
    preserved_pruned = 0
    for batch in _stream_rows(lake, _SOURCE_TABLE):
        if not batch:
            continue
        batch_ids = [str(source_row.get("materialization_id") or "") for source_row in batch]
        existing_by_id = _fetch_rows_by_ids(
            lake, _ROLLUP_TABLE, "materialization_id", batch_ids, columns=_RETENTION_COLUMNS
        )
        rebuilt: list[dict[str, Any]] = []
        for source_row in batch:
            source_count += 1
            existing = existing_by_id.get(str(source_row.get("materialization_id") or ""))
            row = rollup_row_from_source(
                source_row, existing=existing, created_by=created_by
            )
            if row["state"] == STATE_PRUNED:
                preserved_pruned += 1
            rebuilt.append(row)
        _flush_rollup_rows(lake, rebuilt)

    indexes_built: tuple[dict[str, Any], ...] = ()
    if build_indexes:
        indexes_built = _build_rollup_indexes(lake)
    return MaterializationRollupSyncReport(
        source_count=source_count,
        rebuilt_count=source_count,
        preserved_pruned_count=preserved_pruned,
        indexes_built=indexes_built,
    )


def _flush_rollup_rows(lake: Lake, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    table = lake.table(_ROLLUP_TABLE)
    data = pa.Table.from_pylist(rows, schema=CURATION_MATERIALIZATION_ROLLUPS_SCHEMA)
    _merge_insert_with_retry(table, "materialization_id", data, update_matched=True)


def _build_rollup_indexes(lake: Lake) -> tuple[dict[str, Any], ...]:
    try:
        from lancedb_robotics.indexing import (
            build_curation_materialization_rollup_predicate_indexes,
        )

        results = build_curation_materialization_rollup_predicate_indexes(lake, replace=False)
        return tuple(result.to_params() for result in results)
    except Exception:  # noqa: BLE001 - index build is best-effort (unsupported backends).
        return ()


# --------------------------------------------------------------------------- #
# Summary / rollups (JSON-free aggregation over promoted columns).
# --------------------------------------------------------------------------- #


def _scope_where(dataset_id: str | None, snapshot_name: str | None) -> str | None:
    clauses: list[str] = []
    if dataset_id:
        clauses.append(f"dataset_id = {_sql_literal(dataset_id)}")
    if snapshot_name:
        clauses.append(f"snapshot_name = {_sql_literal(snapshot_name)}")
    return " AND ".join(clauses) if clauses else None


def _day_bucket(created_at: datetime | None) -> str:
    if created_at is None:
        return ""
    return created_at.date().isoformat()


_GROUP_BY_KEYS: dict[str, Callable[[dict[str, Any]], str]] = {
    "snapshot": lambda row: str(row.get("snapshot_name") or ""),
    "branch": lambda row: str(row.get("snapshot_name") or ""),
    "format": lambda row: str(row.get("target_format") or ""),
    "mode": lambda row: str(row.get("mode") or ""),
    "policy": lambda row: str(row.get("payload_copy_policy") or ""),
    "transform": lambda row: str(row.get("transform_id") or ""),
    "day": lambda row: _day_bucket(_coerce_dt(row.get("created_at"))),
}


def _empty_accumulator() -> dict[str, int]:
    return {
        "materialization_count": 0,
        "total_payload_bytes": 0,
        "copied_payload_bytes": 0,
        "logical_reference_bytes": 0,
        "planned_payload_bytes": 0,
        "metadata_bytes_written": 0,
        "output_file_count": 0,
        "output_file_bytes": 0,
    }


def _accumulate(acc: dict[str, int], row: dict[str, Any]) -> None:
    acc["materialization_count"] += 1
    acc["total_payload_bytes"] += _as_int(row.get("total_payload_bytes"))
    acc["copied_payload_bytes"] += _as_int(row.get("copied_payload_bytes"))
    acc["logical_reference_bytes"] += _as_int(row.get("logical_reference_bytes"))
    acc["planned_payload_bytes"] += _as_int(row.get("planned_payload_bytes"))
    acc["metadata_bytes_written"] += _as_int(row.get("metadata_bytes_written"))
    acc["output_file_count"] += _as_int(row.get("output_file_count"))
    acc["output_file_bytes"] += _as_int(row.get("output_file_bytes"))


def _ratio(copied: int, total: int) -> float:
    return copied / total if total else 0.0


def materialization_rollup_summary(
    lake: Lake,
    *,
    dataset_id: str | None = None,
    snapshot_name: str | None = None,
    include_pruned: bool = True,
    group_by: str | None = None,
    stats: Any = None,
    batch_size: int = _SCAN_BATCH,
) -> MaterializationRollup:
    """Aggregate copy-cost over the indexed rollup catalog -- never parses JSON.

    Streams only the promoted numeric/identity columns for the scoped rows and
    folds them into a grand total plus, when ``group_by`` is set, per-key buckets
    (``snapshot``/``branch``, ``format``, ``mode``, ``policy``, ``transform``,
    ``day``) -- the branch-comparison and benchmark copy-cost trend surface.
    """
    if group_by is not None and group_by not in _GROUP_BY_KEYS:
        raise MaterializationRollupError(
            f"unknown group_by {group_by!r}; expected one of {sorted(_GROUP_BY_KEYS)}"
        )
    where = _scope_where(dataset_id, snapshot_name)
    grand = _empty_accumulator()
    buckets: dict[str, dict[str, int]] = {}
    key_fn = _GROUP_BY_KEYS.get(group_by) if group_by else None
    bounded = True
    for batch in _stream_rows(
        lake, _ROLLUP_TABLE, columns=_ROLLUP_SUMMARY_COLUMNS, where_sql=where, batch_size=batch_size, stats=stats
    ):
        for row in batch:
            if dataset_id and str(row.get("dataset_id") or "") != dataset_id:
                continue
            if snapshot_name and str(row.get("snapshot_name") or "") != snapshot_name:
                continue
            if not include_pruned and str(row.get("state")) == STATE_PRUNED:
                continue
            _accumulate(grand, row)
            if key_fn is not None:
                bucket = buckets.setdefault(key_fn(row), _empty_accumulator())
                _accumulate(bucket, row)
    if stats is not None and getattr(stats, "materialized_tables", None):
        if _ROLLUP_TABLE in getattr(stats, "materialized_tables", ()):  # pragma: no cover - honesty flag
            bounded = False
    ordered_buckets = tuple(
        MaterializationRollupBucket(
            key=key,
            materialization_count=acc["materialization_count"],
            total_payload_bytes=acc["total_payload_bytes"],
            copied_payload_bytes=acc["copied_payload_bytes"],
            logical_reference_bytes=acc["logical_reference_bytes"],
            planned_payload_bytes=acc["planned_payload_bytes"],
            metadata_bytes_written=acc["metadata_bytes_written"],
            copy_ratio=_ratio(acc["copied_payload_bytes"], acc["total_payload_bytes"]),
        )
        for key, acc in sorted(buckets.items())
    )
    return MaterializationRollup(
        scope={"dataset_id": dataset_id or "", "snapshot_name": snapshot_name or "", "include_pruned": include_pruned},
        materialization_count=grand["materialization_count"],
        total_payload_bytes=grand["total_payload_bytes"],
        copied_payload_bytes=grand["copied_payload_bytes"],
        logical_reference_bytes=grand["logical_reference_bytes"],
        planned_payload_bytes=grand["planned_payload_bytes"],
        metadata_bytes_written=grand["metadata_bytes_written"],
        copy_ratio=_ratio(grand["copied_payload_bytes"], grand["total_payload_bytes"]),
        output_file_count=grand["output_file_count"],
        output_file_bytes=grand["output_file_bytes"],
        group_by=group_by,
        buckets=ordered_buckets,
        bounded=bounded,
    )


def dataset_summary_compat(
    lake: Lake,
    dataset_id: str,
    *,
    stats: Any = None,
    batch_size: int = _SCAN_BATCH,
) -> dict[str, Any] | None:
    """Return the legacy ``_materialization_summary`` dict from the rollup catalog.

    Returns ``None`` when the dataset has no rollup rows yet (an un-synced old
    lake), so the caller can fall back to the compat source-streaming path. This
    is the fast path behind the compare "materialization" metric: it reads only
    promoted columns and never parses ``report_json``. It preserves the legacy
    summary shape, so the ``reports`` list is O(dataset reports) -- the bounded
    surfaces for a large program are :func:`materialization_rollup_summary`
    (folded totals + group-by trends) and :func:`list_materialization_history`
    (paged); folding this per-report list into a bounded preview is follow-up
    0486.
    """
    where = f"dataset_id = {_sql_literal(dataset_id)}"
    rows: list[dict[str, Any]] = []
    for batch in _stream_rows(
        lake, _ROLLUP_TABLE, columns=_ROLLUP_SUMMARY_COLUMNS, where_sql=where, batch_size=batch_size, stats=stats
    ):
        for row in batch:
            if str(row.get("dataset_id") or "") == dataset_id:
                rows.append(row)
    if not rows:
        return None
    rows.sort(key=_row_key)
    total = sum(_as_int(r.get("total_payload_bytes")) for r in rows)
    copied = sum(_as_int(r.get("copied_payload_bytes")) for r in rows)
    logical = sum(_as_int(r.get("logical_reference_bytes")) for r in rows)
    planned = sum(_as_int(r.get("planned_payload_bytes")) for r in rows)
    metadata = sum(_as_int(r.get("metadata_bytes_written")) for r in rows)
    return {
        "materialization_count": len(rows),
        "total_payload_bytes": total,
        "copied_payload_bytes": copied,
        "logical_reference_bytes": logical,
        "planned_payload_bytes": planned,
        "metadata_bytes_written": metadata,
        "copy_ratio": _ratio(copied, total),
        "reports": [
            {
                "materialization_id": str(r.get("materialization_id") or ""),
                "target_format": str(r.get("target_format") or ""),
                "output_uri": str(r.get("output_uri") or ""),
                "mode": str(r.get("mode") or ""),
                "copied_payload_bytes": _as_int(r.get("copied_payload_bytes")),
                "logical_reference_bytes": _as_int(r.get("logical_reference_bytes")),
                "planned_payload_bytes": _as_int(r.get("planned_payload_bytes")),
                "metadata_bytes_written": _as_int(r.get("metadata_bytes_written")),
                "copy_ratio": _as_float(r.get("copy_ratio")),
                "transform_id": str(r.get("transform_id") or ""),
            }
            for r in rows
        ],
    }


# --------------------------------------------------------------------------- #
# Paginated history query (engine order_by keyset + bounded heap fallback).
# --------------------------------------------------------------------------- #


def _history_where(
    *,
    dataset_id: str | None,
    snapshot_name: str | None,
    target_format: str | None,
    mode: str | None,
    payload_copy_policy: str | None,
    state: str | None,
) -> str | None:
    clauses: list[str] = []
    if dataset_id:
        clauses.append(f"dataset_id = {_sql_literal(dataset_id)}")
    if snapshot_name:
        clauses.append(f"snapshot_name = {_sql_literal(snapshot_name)}")
    if target_format:
        clauses.append(f"target_format = {_sql_literal(target_format)}")
    if mode:
        clauses.append(f"mode = {_sql_literal(mode)}")
    if payload_copy_policy:
        clauses.append(f"payload_copy_policy = {_sql_literal(payload_copy_policy)}")
    if state:
        clauses.append(f"state = {_sql_literal(state)}")
    return " AND ".join(clauses) if clauses else None


def _matcher(
    *,
    dataset_id: str | None,
    snapshot_name: str | None,
    target_format: str | None,
    mode: str | None,
    payload_copy_policy: str | None,
    state: str | None,
    since: datetime | None,
    until: datetime | None,
) -> Callable[[dict[str, Any]], bool]:
    def _match(row: dict[str, Any]) -> bool:
        if dataset_id and str(row.get("dataset_id") or "") != dataset_id:
            return False
        if snapshot_name and str(row.get("snapshot_name") or "") != snapshot_name:
            return False
        if target_format and str(row.get("target_format") or "") != target_format:
            return False
        if mode and str(row.get("mode") or "") != mode:
            return False
        if payload_copy_policy and str(row.get("payload_copy_policy") or "") != payload_copy_policy:
            return False
        if state and str(row.get("state") or "") != state:
            return False
        created = _coerce_dt(row.get("created_at"))
        if since is not None and (created is None or created < since):
            return False
        if until is not None and (created is None or created > until):
            return False
        return True

    return _match


def _read_history_ordered(
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
        query = query.select(list(_ROLLUP_SUMMARY_COLUMNS))
        query = query.order_by(
            [
                ColumnOrdering(column_name="created_at", ascending=True),
                ColumnOrdering(column_name="materialization_id", ascending=True),
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
            _record_batch(stats, _ROLLUP_TABLE, len(rows))
            for row in rows:
                key = _row_key(row)
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


@dataclass(order=True)
class _HeapKey:
    created_at: datetime
    materialization_id: str


def _read_history_bounded_heap(
    lake: Lake,
    *,
    where_sql: str | None,
    matcher: Callable[[dict[str, Any]], bool],
    cursor_key: tuple[datetime, str] | None,
    page_size: int,
    stats: Any,
) -> tuple[list[dict[str, Any]], bool]:
    want = page_size + 1
    heap: list[tuple[_HeapKey, int, dict[str, Any]]] = []
    seq = 0
    qualifying = 0
    for batch in _stream_rows(
        lake, _ROLLUP_TABLE, columns=_ROLLUP_SUMMARY_COLUMNS, where_sql=where_sql, stats=stats
    ):
        for row in batch:
            if not matcher(row):
                continue
            key = _row_key(row)
            if cursor_key is not None and key <= cursor_key:
                continue
            qualifying += 1
            entry = (_HeapKey(key[0], key[1]), seq, row)
            seq += 1
            if len(heap) < want:
                heapq.heappush(heap, entry)
            elif key < (heap[0][0].created_at, heap[0][0].materialization_id):
                heapq.heapreplace(heap, entry)
    ordered = [entry[2] for entry in sorted(heap, key=lambda item: _row_key(item[2]))]
    has_more = qualifying > page_size
    return ordered[:page_size], has_more


def list_materialization_history(
    lake: Lake,
    *,
    dataset_id: str | None = None,
    snapshot_name: str | None = None,
    target_format: str | None = None,
    mode: str | None = None,
    payload_copy_policy: str | None = None,
    state: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    page_size: int | None = None,
    cursor: str | None = None,
    page_index: int = 0,
    stats: Any = None,
) -> MaterializationRollupPage:
    """One deterministic, bounded page of materialization history, ascending key.

    Filters push down to indexed rollup columns; rows are ordered by
    ``(created_at, materialization_id)`` in the engine. Client memory is bounded
    to ``page_size`` (the page is capped and, on an unorderable backend, a top-
    ``page_size+1`` heap produces the same page in O(page_size) memory -- never a
    silent costlier path). The cursor is applied *client-side* (the keyset is not
    pushed into the WHERE clause, mirroring 0142's created_at handling), so a
    page at depth N still *scans* the O(N*page_size) consumed prefix even though
    it only *collects* one page -- deep paging is O(offset) rows read per page.
    Scope the query with filters for the fast path; engine-side keyset pushdown
    for flat deep paging is follow-up 0484.
    """
    if state is not None and state not in STATES:
        raise MaterializationRollupError(f"unknown state {state!r}; expected one of {list(STATES)}")
    resolved_page = _DEFAULT_PAGE_SIZE if page_size is None else int(page_size)
    if resolved_page <= 0:
        raise MaterializationRollupError("page_size must be positive")
    resolved_page = min(resolved_page, _MAX_PAGE_SIZE)
    cursor_key = _decode_cursor(cursor) if cursor else None
    where_sql = _history_where(
        dataset_id=dataset_id,
        snapshot_name=snapshot_name,
        target_format=target_format,
        mode=mode,
        payload_copy_policy=payload_copy_policy,
        state=state,
    )
    match = _matcher(
        dataset_id=dataset_id,
        snapshot_name=snapshot_name,
        target_format=target_format,
        mode=mode,
        payload_copy_policy=payload_copy_policy,
        state=state,
        since=since,
        until=until,
    )
    cursor_tuple = None
    if cursor_key is not None:
        cursor_tuple = (cursor_key[0] or datetime.min.replace(tzinfo=UTC), cursor_key[1])

    handle = lake.table(_ROLLUP_TABLE)
    ordered = _read_history_ordered(
        handle,
        where_sql=where_sql,
        cursor_key=cursor_tuple,
        since=since,
        until=until,
        page_size=resolved_page,
        stats=stats,
    )
    if ordered is not None:
        rows, has_more = ordered
    else:
        warnings.warn(
            "materialization history paging could not order the scan in the backend; "
            "using a bounded heap over a scoped scan (deterministic, O(page_size) client "
            "memory, but O(scope) rows scanned per page). Run `lake maintain` to build the "
            "rollup predicate indexes.",
            RuntimeWarning,
            stacklevel=2,
        )
        rows, has_more = _read_history_bounded_heap(
            lake,
            where_sql=where_sql,
            matcher=match,
            cursor_key=cursor_tuple,
            page_size=resolved_page,
            stats=stats,
        )
    records = tuple(_entry_from_row(row) for row in rows)
    next_cursor = None
    if has_more and records:
        last = records[-1]
        next_cursor = _encode_cursor(last.created_at, last.materialization_id)
    return MaterializationRollupPage(
        records=records,
        page_size=resolved_page,
        page_index=page_index,
        cursor=cursor,
        next_cursor=next_cursor,
        has_more=has_more,
    )


def iter_materialization_history(
    lake: Lake,
    *,
    cursor: str | None = None,
    **kwargs: Any,
) -> Iterable[MaterializationRollupPage]:
    """Yield successive history pages until the catalog is exhausted."""
    page_index = 0
    current = cursor
    while True:
        page = list_materialization_history(
            lake, cursor=current, page_index=page_index, **kwargs
        )
        yield page
        if not page.has_more or not page.next_cursor:
            return
        current = page.next_cursor
        page_index += 1


# --------------------------------------------------------------------------- #
# Retention / plan compaction.
# --------------------------------------------------------------------------- #


def _resolve_older_than(older_than: datetime | timedelta | None) -> datetime | None:
    if older_than is None:
        return None
    if isinstance(older_than, timedelta):
        return datetime.now(UTC) - older_than
    return _coerce_dt(older_than)


def _is_prunable_plan(row: dict[str, Any]) -> bool:
    """A plan/dry-run report with no copied payload bytes -- safe to compact.

    Completed export evidence (mode ``export`` or any copied payload bytes) and
    logical-reference projection records are protected and never touched.
    """
    if str(row.get("mode") or "") not in PRUNABLE_MODES:
        return False
    if _as_int(row.get("copied_payload_bytes")) > 0:
        return False
    return True


def _series_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset_id") or ""),
        str(row.get("target_format") or ""),
        str(row.get("output_uri") or ""),
        str(row.get("mode") or ""),
    )


#: Bounded batch for retention state flips / source-body clears.
_RETENTION_FLUSH = 512

#: Order that groups rows by plan series and, within a series, newest-first, so a
#: streamed pass keeps only the newest ``retain_latest`` in memory.
_SERIES_ORDER_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("dataset_id", True),
    ("target_format", True),
    ("output_uri", True),
    ("mode", True),
    ("created_at", False),
    ("materialization_id", False),
)


def _iter_plan_rows_ordered(lake: Lake, plan_where: str) -> Iterable[dict[str, Any]]:
    """Stream prunable plan rows grouped by series, newest-first within a series.

    Primary path pushes the ``mode IN (plan,dry-run)`` predicate and the
    series/created_at ``order_by`` into the engine so completed export evidence
    never enters the working set and rows arrive already grouped -- the caller
    holds only the newest ``retain_latest`` of each series. If the backend cannot
    order the scan, falls back to loading just the plan-report subset (bounded to
    plan/dry-run rows via the pushed predicate, never the whole catalog) and
    sorting in Python, with a non-silent warning (SKILLS.md §1: never a silent
    costlier path).
    """
    handle = lake.table(_ROLLUP_TABLE)
    try:
        from lancedb.query import ColumnOrdering

        query = handle.search().where(plan_where).select(list(_RETENTION_COLUMNS))
        query = query.order_by(
            [ColumnOrdering(column_name=col, ascending=asc) for col, asc in _SERIES_ORDER_COLUMNS]
        )
        batches = query.to_batches(batch_size=_SCAN_BATCH)
    except Exception:
        batches = None
    if batches is not None:
        try:
            for batch in batches:
                yield from batch.to_pylist()
            return
        except Exception:
            pass
    warnings.warn(
        "materialization retention could not order the scan in the backend; loading the "
        "plan/dry-run report subset into memory to group by series (bounded to plan rows "
        "via the pushed mode predicate, never the whole catalog). Run `lake maintain` to "
        "build the rollup predicate indexes.",
        RuntimeWarning,
        stacklevel=2,
    )
    rows: list[dict[str, Any]] = []
    for batch in _stream_rows(lake, _ROLLUP_TABLE, columns=_RETENTION_COLUMNS, where_sql=plan_where):
        rows.extend(batch)
    rows.sort(key=_row_key, reverse=True)  # (created_at, id) descending
    rows.sort(key=_series_key)  # stable: group by series, newest-first within
    yield from rows


def prune_materialization_rollups(
    lake: Lake,
    *,
    retain_latest: int = 1,
    older_than: datetime | timedelta | None = None,
    dry_run: bool = False,
    created_by: str = "lancedb-robotics",
) -> MaterializationRetentionReport:
    """Compact superseded plan / dry-run reports (active -> superseded -> pruned).

    Within each ``(dataset, format, output_uri, mode)`` plan series the newest
    ``retain_latest`` reports stay ``active``. Older ones are marked
    ``superseded`` (source body retained). Superseded reports that predate
    ``older_than`` are ``pruned``: the source ``report_json`` and the per-file
    accounting chunks are cleared, and the promoted rollup row + ``report_sha1``
    survive as audit evidence. Completed export evidence is never eligible. With
    no ``older_than`` nothing is deleted (safe soft-retire); ``dry_run`` reports
    the same sets without writing.

    Bounded memory (SKILLS.md §2): only plan/dry-run rows are read (pushed
    predicate), they are streamed grouped by series, and only the newest
    ``retain_latest`` of the current series plus a bounded write batch are held --
    memory is independent of catalog size even when one series holds millions of
    repeated plan reports. Crash-convergent (SKILLS.md §1): for each pruned batch
    the source body + file chunks are cleared (idempotently) *before* the rollup
    state is flipped to ``pruned``, so an interrupted run leaves the row still
    prunable and a retry converges rather than orphaning a body.
    """
    if retain_latest < 0:
        raise MaterializationRollupError("retain_latest must be >= 0")
    cutoff = _resolve_older_than(older_than)
    plan_where = _in_clause("mode", sorted(PRUNABLE_MODES))
    total_count = _count_rows(lake, _ROLLUP_TABLE)
    now = datetime.now(UTC)
    policy = _json_dumps(
        {
            "operation": "materialization-rollup-retention",
            "retain_latest": retain_latest,
            "older_than": cutoff.isoformat() if cutoff else None,
            "created_by": created_by,
        }
    )

    superseded_ids: list[str] = []
    pruned_ids: list[str] = []
    body_bytes_before = 0
    pruned_bytes = 0
    scanned = 0
    file_chunks_deleted = 0
    supersede_batch: list[dict[str, Any]] = []
    prune_batch: list[dict[str, Any]] = []

    def _flush_supersede() -> None:
        if supersede_batch and not dry_run:
            _apply_state(lake, supersede_batch, state=STATE_SUPERSEDED, now=now, policy=policy)
        supersede_batch.clear()

    def _flush_prune() -> None:
        nonlocal file_chunks_deleted
        if prune_batch and not dry_run:
            # Write-ahead safe-delete: clear the source body + file chunks first
            # (both idempotent), then flip the rollup state as the final commit.
            file_chunks_deleted += _prune_source_bodies(
                lake, [r["materialization_id"] for r in prune_batch]
            )
            _apply_state(lake, prune_batch, state=STATE_PRUNED, now=now, policy=policy)
        prune_batch.clear()

    current_key: tuple[str, str, str, str] | None = None
    position = 0
    newest_id = ""
    for row in _iter_plan_rows_ordered(lake, plan_where):
        if not _is_prunable_plan(row):
            continue
        scanned += 1
        key = _series_key(row)
        if key != current_key:
            current_key = key
            position = 0
            newest_id = str(row.get("materialization_id") or "")
        else:
            position += 1
        if row.get("source_report_available"):
            body_bytes_before += _as_int(row.get("report_bytes"))
        if position < retain_latest:
            continue  # newest retain_latest of the series stay active
        mat_id = str(row.get("materialization_id") or "")
        tagged = {**row, "superseded_by": newest_id}
        created = _row_key(row)[0]
        if cutoff is not None and created < cutoff:
            if str(row.get("state")) != STATE_PRUNED:
                pruned_ids.append(mat_id)
                if row.get("source_report_available"):
                    pruned_bytes += _as_int(row.get("report_bytes"))
                prune_batch.append(tagged)
                if len(prune_batch) >= _RETENTION_FLUSH:
                    _flush_prune()
        elif str(row.get("state")) != STATE_SUPERSEDED:
            superseded_ids.append(mat_id)
            supersede_batch.append(tagged)
            if len(supersede_batch) >= _RETENTION_FLUSH:
                _flush_supersede()
    _flush_supersede()
    _flush_prune()

    return MaterializationRetentionReport(
        dry_run=dry_run,
        retain_latest=retain_latest,
        older_than=cutoff.isoformat() if cutoff else None,
        scanned_count=scanned,
        protected_count=max(0, total_count - scanned),
        active_ids=(),
        superseded_ids=tuple(superseded_ids),
        pruned_ids=tuple(pruned_ids),
        body_bytes_before=body_bytes_before,
        body_bytes_after=body_bytes_before - pruned_bytes,
        file_chunks_deleted=file_chunks_deleted,
    )


def _apply_state(
    lake: Lake,
    rows: list[dict[str, Any]],
    *,
    state: str,
    now: datetime,
    policy: str,
) -> None:
    """Flip the rollup lifecycle state for ``rows`` in one atomic merge_insert."""
    payload: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["state"] = state
        updated["retention_policy_json"] = policy
        if state == STATE_SUPERSEDED:
            updated["superseded_at"] = now
        elif state == STATE_PRUNED:
            updated["pruned_at"] = now
            updated["superseded_at"] = updated.get("superseded_at") or now
            updated["source_report_available"] = False
        # Normalize types the schema requires.
        updated["created_at"] = _coerce_dt(updated.get("created_at"))
        updated["superseded_at"] = _coerce_dt(updated.get("superseded_at"))
        updated["pruned_at"] = _coerce_dt(updated.get("pruned_at"))
        updated["source_table_versions"] = list(updated.get("source_table_versions") or ())
        payload.append({k: updated.get(k) for k in _ROLLUP_ROW_KEYS})
    table = lake.table(_ROLLUP_TABLE)
    data = pa.Table.from_pylist(payload, schema=CURATION_MATERIALIZATION_ROLLUPS_SCHEMA)
    _merge_insert_with_retry(table, "materialization_id", data, update_matched=True)


def _prune_source_bodies(lake: Lake, materialization_ids: Sequence[str]) -> int:
    """Clear source ``report_json`` and delete per-file chunks for pruned reports.

    Safe-delete: only the heavy body and file chunks go; the source row's identity
    and byte columns stay, and the rollup row survives as audit evidence. Bounded:
    the source rows to clear are fetched by the pruned id set via chunked
    ``materialization_id IN (...)`` reads (never a full-table blob scan), and file
    chunks are counted + deleted by the same chunked predicate rather than one
    commit per id (SKILLS.md §2).
    """
    ids = list(dict.fromkeys(str(m) for m in materialization_ids))
    if not ids:
        return 0
    source = lake.table(_SOURCE_TABLE)
    existing = _fetch_rows_by_ids(lake, _SOURCE_TABLE, "materialization_id", ids)
    cleared: list[dict[str, Any]] = []
    for row in existing.values():
        updated = dict(row)
        updated["report_json"] = ""
        updated["source_table_versions"] = list(updated.get("source_table_versions") or ())
        updated["created_at"] = _coerce_dt(updated.get("created_at"))
        cleared.append(updated)
    if cleared:
        data = pa.Table.from_pylist(cleared, schema=_source_schema())
        _merge_insert_with_retry(source, "materialization_id", data, update_matched=True)

    files = lake.table(_FILES_TABLE)
    deleted = 0
    for group in _chunked(ids):
        clause = _in_clause("materialization_id", group)
        try:
            deleted += int(files.count_rows(clause))
        except Exception:  # noqa: BLE001 - count is best-effort telemetry.
            pass
        try:
            files.delete(clause)
        except Exception:  # noqa: BLE001 - a backend without delete leaves chunks (best-effort).
            pass
    return deleted


def _source_schema() -> pa.Schema:
    from lancedb_robotics.schemas import CURATION_MATERIALIZATIONS_SCHEMA

    return CURATION_MATERIALIZATIONS_SCHEMA


#: Column order used when writing rollup rows (matches the schema field order).
_ROLLUP_ROW_KEYS: tuple[str, ...] = tuple(
    field.name for field in CURATION_MATERIALIZATION_ROLLUPS_SCHEMA
)
