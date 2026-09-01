"""Lake-wide ``aligned_ticks`` migration and validation at scale (backlog 0134).

Backlog 0078 materialized recorded alignments into the ``aligned_ticks`` JSONB
training table and shipped a *single-job* backfill helper
(``lake.training.backfill_aligned_ticks``). Production lakes accumulated many
recorded alignment jobs before that landed, in mixed old/new schema states, and
on remote deployments where a failed job must be auditable rather than rerun by
hand. This module is the production-scale path:

- **Batch**: one call migrates all (or selected) recorded alignment jobs.
- **Bounded**: source ``aligned_frames`` rows are read in tick-index windows;
  no per-job collect-all, no giant ``IN (...)`` predicates (range predicates
  only), so peak memory is bounded by ``tick_window`` regardless of lake size.
  Sparse tick ranges are handled by folding the set of *non-empty* windows
  during the bounds scan, so a job never pays for empty range queries.
- **Resumable + idempotent**: ``aligned_tick_id`` is content-addressed by
  ``(alignment_id, tick_index)``, so a rerun writes only the ticks that are
  missing. Progress is effectively keyed by alignment id and tick range, and
  staleness is detected via the recipe digest carried on every tick row.
- **Validated**: every window compares a per-tick metadata signature of the
  source frame rows against the stored tick rows, round-trips the four JSONB
  columns, and re-derives the typed summary columns. Failures carry the
  alignment id, tick index, stream key, and mismatched column.
- **Explicit at the edges**: a missing ``aligned_ticks`` table is only created
  with ``create_missing_table=True``, and on classified remote/namespace
  backends creation is capability-gated (schema family, backlog 0128) — the
  job is *skipped with a reason*, never half-written.
- **Auditable**: each migrated job records a ``transform_runs`` row
  (kind ``aligned-ticks-migration``, idempotent by content digest) and emits
  inline lineage, so the migration is visible to ``lake.lineage`` consumers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from lancedb_robotics.capability_gates import MAINTENANCE, SCHEMA, lake_capability_reason
from lancedb_robotics.schemas import ALIGNED_TICKS_SCHEMA, TRANSFORM_RUNS_SCHEMA

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from lancedb_robotics.lake import Lake

#: Versioned report contract, mirroring ``training-loader-report/v1`` (0124).
MIGRATION_REPORT_VERSION = "aligned-tick-migration/1"

#: ``transform_runs.kind`` for per-job migration lineage rows.
MIGRATION_TRANSFORM_KIND = "aligned-ticks-migration"

#: Ticks per scan/verify window. Bounds peak memory to roughly
#: ``tick_window * len(streams)`` source rows per job regardless of lake size.
DEFAULT_TICK_WINDOW = 1024

#: Rows per ``aligned_ticks.add`` append inside a window.
DEFAULT_WRITE_BATCH_SIZE = 512

#: Arrow batch size for bounded source/verify scans.
_SCAN_BATCH_SIZE = 4096

#: Per-job cap on recorded validation-failure details. Counts stay exact; only
#: the detail list stops growing so a pathological job cannot balloon the
#: report (or the migrator's memory).
MAX_VALIDATION_DETAILS_PER_JOB = 20

_JOB_COLUMNS = (
    "alignment_id",
    "name",
    "streams",
    "recipe",
    "output_table",
    "input_versions",
    "transform_id",
    "created_at",
)

_JOB_MIGRATED_STATUSES = {"migrated", "resumed"}


#: merge_insert attempts before surfacing a Lance optimistic-concurrency
#: commit conflict. Migration writes are idempotent, so a preempted commit
#: converges on retry.
_MERGE_INSERT_ATTEMPTS = 3


class AlignedTickMigrationError(Exception):
    """Raised when the migration cannot be planned or executed."""


def _is_retryable_commit_conflict(exc: BaseException) -> bool:
    """True for Lance's retryable optimistic-concurrency commit conflict.

    Mirrors enrich's `_upsert_rows` guard: the preempted writer re-reads the
    latest version and re-runs; because these writes are idempotent by key,
    retrying converges.
    """
    return "commit conflict" in str(exc).lower()


def _merge_insert_with_retry(
    table: Any,
    key_column: str,
    data: pa.Table,
    *,
    update_matched: bool = False,
) -> None:
    """Single-commit upsert (BUG-04 fix shape) with bounded conflict retry.

    ``update_matched=False`` is insert-only: matched keys are left untouched,
    which is what makes concurrent migrators writing the same content-addressed
    rows converge without duplicates.
    """
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
    raise AlignedTickMigrationError(
        f"merge_insert on {key_column!r} kept hitting commit conflicts "
        f"after {_MERGE_INSERT_ATTEMPTS} attempts: {last_error}"
    )


def migrate_aligned_ticks(
    lake: Lake,
    *,
    alignments: Sequence[str] | None = None,
    dry_run: bool = False,
    verify: bool = True,
    replace: bool = False,
    create_missing_table: bool = False,
    tick_window: int = DEFAULT_TICK_WINDOW,
    batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
    created_by: str = "aligned-tick-migration",
) -> dict[str, Any]:
    """Migrate recorded alignment jobs into ``aligned_ticks`` and validate them.

    ``alignments`` selects jobs by alignment id or name (``None`` means every
    recorded, materialized alignment job). ``dry_run`` produces the same
    job/tick plan — jobs scanned, ticks expected, ticks that would be written —
    without writing rows or lineage. ``replace`` deletes a job's existing tick
    rows first, which is also the remediation for stale or failed-validation
    rows. In the report, ``ticks_written`` means "would be written" for
    dry-run jobs (status ``planned``). Returns a
    :data:`MIGRATION_REPORT_VERSION` report dict.
    """
    if tick_window <= 0:
        raise AlignedTickMigrationError("tick_window must be positive")
    if batch_size <= 0:
        raise AlignedTickMigrationError("batch_size must be positive")

    jobs = _select_alignment_jobs(lake, alignments)
    started_at = datetime.now(UTC)
    table_state = _aligned_ticks_table_state(
        lake,
        create_missing_table=create_missing_table,
        dry_run=dry_run,
    )

    job_reports: list[dict[str, Any]] = []
    for job in jobs:
        if table_state["skip_reason"] is not None:
            job_reports.append(
                _job_report(job, status="skipped", reason=table_state["skip_reason"])
            )
            continue
        try:
            job_reports.append(
                _migrate_job(
                    lake,
                    job,
                    dry_run=dry_run,
                    verify=verify,
                    replace=replace,
                    tick_window=tick_window,
                    batch_size=batch_size,
                    ticks_table_exists=table_state["exists"],
                    created_by=created_by,
                )
            )
        except AlignedTickMigrationError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad job must not abort the sweep.
            job_reports.append(_job_report(job, status="failed", reason=f"unexpected error: {exc}"))

    # A mass migration appends many small fragments and leaves new rows outside
    # the existing scalar indexes (BUG-14/BUG-15 shape); close the sweep with a
    # scoped compact + index refresh, capability-gated with an explicit reason.
    maintenance: dict[str, Any] | None = None
    total_written = sum(int(job.get("ticks_written") or 0) for job in job_reports)
    if not dry_run and total_written:
        maintenance = _post_migration_maintenance(lake)

    finished_at = datetime.now(UTC)
    return _build_report(
        lake,
        job_reports,
        dry_run=dry_run,
        verify=verify,
        replace=replace,
        table_created=table_state["created"],
        table_would_create=table_state["would_create"],
        maintenance=maintenance,
        started_at=started_at,
        finished_at=finished_at,
    )


def _select_alignment_jobs(
    lake: Lake,
    alignments: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Return the latest materialized job row per alignment id, bounded scan."""
    # Membership probe rather than a broad open/except: a transient backend
    # error must surface as itself, not masquerade as "table absent".
    if "alignment_jobs" not in set(lake.table_names()):
        raise AlignedTickMigrationError(
            "this lake has no alignment_jobs table; initialize the lake and record "
            "an alignment before migrating"
        )
    table = lake.table("alignment_jobs")
    latest: dict[str, dict[str, Any]] = {}
    query = table.search().select(list(_JOB_COLUMNS))
    for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
        for row in batch.to_pylist():
            if row.get("output_table") not in {"aligned_frames", "aligned_ticks"}:
                continue
            alignment_id = str(row["alignment_id"])
            current = latest.get(alignment_id)
            if current is None or row["created_at"] > current["created_at"]:
                latest[alignment_id] = row
    jobs = sorted(latest.values(), key=lambda row: str(row["alignment_id"]))
    if alignments is None:
        return jobs
    requested = [str(item) for item in alignments if str(item)]
    if not requested:
        raise AlignedTickMigrationError("alignments must name at least one alignment id or name")
    by_id = {str(job["alignment_id"]): job for job in jobs}
    by_name: dict[str, dict[str, Any]] = {}
    for job in jobs:
        by_name.setdefault(str(job["name"]), job)
    selected: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []
    for item in requested:
        job = by_id.get(item) or by_name.get(item)
        if job is None:
            unknown.append(item)
        else:
            selected[str(job["alignment_id"])] = job
    if unknown:
        raise AlignedTickMigrationError(
            f"unknown alignment ids or names {unknown}; "
            "list recorded alignments via the alignment_jobs table"
        )
    return sorted(selected.values(), key=lambda row: str(row["alignment_id"]))


def _aligned_ticks_table_state(
    lake: Lake,
    *,
    create_missing_table: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Resolve the target table once. ``skip_reason`` set => all jobs skip."""
    state = {
        "exists": False,
        "created": False,
        "would_create": False,
        "skip_reason": None,
    }
    # Membership probe rather than open/except: a transient backend error must
    # surface as itself, not be misread as "table absent".
    if "aligned_ticks" in set(lake.table_names()):
        state["exists"] = True
        return state
    if not create_missing_table:
        state["skip_reason"] = (
            "aligned_ticks table is absent in this lake; re-run with "
            "create_missing_table=True (CLI: --create-table) to create it explicitly"
        )
        return state
    capability_reason = lake_capability_reason(lake, SCHEMA)
    if capability_reason is not None:
        state["skip_reason"] = f"cannot create aligned_ticks table: {capability_reason}"
        return state
    if dry_run:
        state["would_create"] = True
        return state
    # Race-safe against a concurrent migrator: exist_ok tolerates the loser,
    # and the re-open below is the single source of truth.
    lake._db.create_table("aligned_ticks", schema=ALIGNED_TICKS_SCHEMA, exist_ok=True)
    lake.table("aligned_ticks")
    state["exists"] = True
    state["created"] = True
    return state


def _migrate_job(
    lake: Lake,
    job: Mapping[str, Any],
    *,
    dry_run: bool,
    verify: bool,
    replace: bool,
    tick_window: int,
    batch_size: int,
    ticks_table_exists: bool,
    created_by: str,
) -> dict[str, Any]:
    from lancedb_robotics.training import _normalize_aligned_streams, _sql_literal

    streams = _normalize_aligned_streams(job, None)
    alignment_id = str(job["alignment_id"])
    source_version_before = int(lake.table("aligned_frames").version)
    bounds = _tick_index_bounds(lake, alignment_id, streams, tick_window=tick_window)
    if bounds is None:
        return _job_report(
            job,
            status="skipped",
            reason="no aligned_frames rows recorded for this alignment",
        )
    lo, hi, source_frame_rows, windows = bounds
    if lo < 0:
        # Alignment tick indexes start at 0 by construction; a negative index
        # would fall outside the truncating window math and silently escape
        # migration, so fail the job loudly instead.
        return _job_report(
            job,
            status="failed",
            reason=(
                f"aligned_frames rows carry a negative tick_index ({lo}); this job "
                "cannot be windowed safely and needs manual inspection"
            ),
            source_frame_rows=source_frame_rows,
        )

    if ticks_table_exists and not replace:
        stale = _stale_tick_state(lake, job)
        if stale is not None:
            return _job_report(
                job,
                status="skipped",
                reason=(
                    f"existing aligned_ticks rows carry recipe digest {stale!r} but the "
                    f"recorded job implies {_recipe_digest(job)!r}; re-run with "
                    "replace=True to rewrite them"
                ),
                source_frame_rows=source_frame_rows,
            )

    if ticks_table_exists and replace and not dry_run:
        lake.table("aligned_ticks").delete(f"alignment_id = {_sql_literal(alignment_id)}")

    totals = {
        "ticks_expected": 0,
        "ticks_existing": 0,
        "ticks_written": 0,
        "frame_rows_read": 0,
    }
    failures: list[dict[str, Any]] = []
    failure_counts = {"metadata_mismatches": 0, "jsonb_failures": 0, "summary_mismatches": 0}

    for window_id in windows:
        window = (window_id * tick_window, (window_id + 1) * tick_window)
        _migrate_window(
            lake,
            job,
            streams=streams,
            window=window,
            dry_run=dry_run,
            verify=verify,
            batch_size=batch_size,
            ticks_table_exists=ticks_table_exists,
            replaced=replace,
            totals=totals,
            failures=failures,
            failure_counts=failure_counts,
        )

    validation_failed = any(failure_counts.values())
    if dry_run:
        status = "planned"
    elif validation_failed:
        status = "failed"
    elif totals["ticks_written"] == 0:
        status = "already-migrated"
    elif totals["ticks_existing"] > 0:
        status = "resumed"
    else:
        status = "migrated"
    reason = None
    if validation_failed:
        reason = (
            "validation failed; inspect validation_failures and re-run with "
            "replace=True to rewrite this alignment"
        )

    report = _job_report(
        job,
        status=status,
        reason=reason,
        source_frame_rows=source_frame_rows,
    )
    source_version_after = int(lake.table("aligned_frames").version)
    report.update(
        {
            "tick_index_range": [lo, hi],
            "ticks_expected": totals["ticks_expected"],
            "ticks_existing": totals["ticks_existing"],
            "ticks_written": totals["ticks_written"],
            "frame_rows_read": totals["frame_rows_read"],
            "replaced": bool(replace),
            # Table versions are lake-global, so drift here can be another
            # alignment's write; it is recorded for auditability (re-run this
            # job if its own frames changed), not treated as a failure.
            "source_table_version": source_version_before,
            "source_table_version_after": source_version_after,
            "source_version_drift": source_version_after != source_version_before,
            "validation": dict(failure_counts),
            "validation_failures": failures,
            "validation_failures_truncated": (sum(failure_counts.values()) > len(failures)),
        }
    )
    if not dry_run and (status in _JOB_MIGRATED_STATUSES or status == "failed"):
        report["transform_id"] = _record_job_transform(lake, job, report, created_by=created_by)
    return report


def _migrate_window(
    lake: Lake,
    job: Mapping[str, Any],
    *,
    streams: tuple[str, ...],
    window: tuple[int, int],
    dry_run: bool,
    verify: bool,
    batch_size: int,
    ticks_table_exists: bool,
    replaced: bool,
    totals: dict[str, int],
    failures: list[dict[str, Any]],
    failure_counts: dict[str, int],
) -> None:
    from lancedb_robotics.training import (
        _aligned_rows_by_tick,
        _aligned_tick_storage_row_from_frame_rows,
        _chunks,
    )

    frame_rows = _scan_frame_window(lake, job, streams, window)
    totals["frame_rows_read"] += len(frame_rows)
    rows_by_tick = _aligned_rows_by_tick(frame_rows, streams)
    if not rows_by_tick:
        return
    totals["ticks_expected"] += len(rows_by_tick)
    # Under replace the job's rows are deleted up front (or would be, in a
    # dry-run), so every expected tick is a write; otherwise resume by writing
    # only ticks that are not already stored.
    existing_ticks: set[int] = set()
    if ticks_table_exists and not replaced:
        existing_ticks = _existing_tick_indices(lake, job, window)
    totals["ticks_existing"] += len(existing_ticks & set(rows_by_tick))
    missing = sorted(set(rows_by_tick) - existing_ticks)
    if dry_run:
        totals["ticks_written"] += len(missing)
        if verify and ticks_table_exists and not replaced:
            # Dry-run audits only rows that already exist; planned writes are
            # not corruption and must not surface as mismatches.
            stored = {tick: rows_by_tick[tick] for tick in existing_ticks & set(rows_by_tick)}
            if stored:
                _verify_window(
                    lake,
                    job,
                    streams=streams,
                    window=window,
                    rows_by_tick=stored,
                    failures=failures,
                    failure_counts=failure_counts,
                )
        return
    if missing:
        created_at = datetime.now(UTC)
        storage_rows = [
            _aligned_tick_storage_row_from_frame_rows(
                job,
                tick_index=tick,
                rows_by_stream=rows_by_tick[tick],
                streams=streams,
                created_at=created_at,
            )
            for tick in missing
        ]
        table = lake.table("aligned_ticks")
        for chunk in _chunks(storage_rows, batch_size):
            # Insert-only merge keyed on the content-addressed id: a racing
            # migrator that already wrote a tick matches and is skipped, so
            # concurrent runs converge without duplicate rows (BUG-04 shape;
            # aligned_ticks has no blob columns, so merge_insert is permitted).
            _merge_insert_with_retry(
                table,
                "aligned_tick_id",
                pa.Table.from_pylist(list(chunk), schema=ALIGNED_TICKS_SCHEMA),
            )
        totals["ticks_written"] += len(missing)
    if verify:
        _verify_window(
            lake,
            job,
            streams=streams,
            window=window,
            rows_by_tick=rows_by_tick,
            failures=failures,
            failure_counts=failure_counts,
        )


def _scan_frame_window(
    lake: Lake,
    job: Mapping[str, Any],
    streams: tuple[str, ...],
    window: tuple[int, int],
) -> list[dict[str, Any]]:
    from lancedb_robotics.training import _ALIGNED_FRAME_SCAN_COLUMNS, _sql_predicate

    predicate = " AND ".join(
        [
            _sql_predicate("alignment_id", str(job["alignment_id"])),
            _sql_predicate("stream", streams),
            f"tick_index >= {int(window[0])}",
            f"tick_index < {int(window[1])}",
        ]
    )
    query = (
        lake.table("aligned_frames")
        .search()
        .select(list(_ALIGNED_FRAME_SCAN_COLUMNS))
        .where(predicate)
    )
    rows: list[dict[str, Any]] = []
    for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
        rows.extend(batch.to_pylist())
    return rows


def _scan_tick_window(
    lake: Lake,
    job: Mapping[str, Any],
    window: tuple[int, int],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    from lancedb_robotics.training import _sql_predicate

    predicate = " AND ".join(
        [
            _sql_predicate("alignment_id", str(job["alignment_id"])),
            f"tick_index >= {int(window[0])}",
            f"tick_index < {int(window[1])}",
        ]
    )
    query = lake.table("aligned_ticks").search().select(list(columns)).where(predicate)
    rows: list[dict[str, Any]] = []
    for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
        rows.extend(batch.to_pylist())
    return rows


def _existing_tick_indices(
    lake: Lake,
    job: Mapping[str, Any],
    window: tuple[int, int],
) -> set[int]:
    rows = _scan_tick_window(lake, job, window, ("tick_index",))
    return {int(row["tick_index"]) for row in rows}


def _tick_index_bounds(
    lake: Lake,
    alignment_id: str,
    streams: tuple[str, ...],
    *,
    tick_window: int,
) -> tuple[int, int, int, tuple[int, ...]] | None:
    """Fold (min, max, row count, non-empty window ids) over aligned_frames.

    Bounded memory: the fold keeps one int per *non-empty* window, never the
    tick list itself, so a sparse tick range costs O(occupied windows), not
    O(range width) empty queries later.
    """
    from lancedb_robotics.training import _sql_predicate

    predicate = " AND ".join(
        [
            _sql_predicate("alignment_id", alignment_id),
            _sql_predicate("stream", streams),
        ]
    )
    query = lake.table("aligned_frames").search().select(["tick_index"]).where(predicate)
    lo: int | None = None
    hi: int | None = None
    count = 0
    windows: set[int] = set()
    window_size = pa.scalar(tick_window, pa.int64())
    for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
        column = batch.column("tick_index")
        if len(column) == 0:
            continue
        count += len(column)
        column = column.cast(pa.int64())
        stats = pc.min_max(column).as_py()
        batch_lo, batch_hi = int(stats["min"]), int(stats["max"])
        lo = batch_lo if lo is None else min(lo, batch_lo)
        hi = batch_hi if hi is None else max(hi, batch_hi)
        for window_id in pc.unique(pc.divide(column, window_size)).to_pylist():
            windows.add(int(window_id))
    if lo is None or hi is None:
        return None
    return lo, hi, count, tuple(sorted(windows))


def _stale_tick_state(lake: Lake, job: Mapping[str, Any]) -> str | None:
    """Return a stale recipe digest carried by existing tick rows, if any.

    Pushed down as a ``recipe_digest <> expected`` predicate with ``limit(1)``
    (alignment_id and recipe_digest are indexed hot columns), so freshness
    costs one indexed probe instead of streaming every tick row's digest.
    """
    from lancedb_robotics.training import _sql_literal, _sql_predicate

    expected = _recipe_digest(job)
    predicate = " AND ".join(
        [
            _sql_predicate("alignment_id", str(job["alignment_id"])),
            f"recipe_digest <> {_sql_literal(expected)}",
        ]
    )
    rows = (
        lake.table("aligned_ticks")
        .search()
        .select(["recipe_digest"])
        .where(predicate)
        .limit(1)
        .to_arrow()
        .to_pylist()
    )
    if rows:
        return str(rows[0]["recipe_digest"])
    return None


def _verify_window(
    lake: Lake,
    job: Mapping[str, Any],
    *,
    streams: tuple[str, ...],
    window: tuple[int, int],
    rows_by_tick: Mapping[int, Mapping[str, Mapping[str, Any]]],
    failures: list[dict[str, Any]],
    failure_counts: dict[str, int],
) -> None:
    from lancedb_robotics.training import (
        _ALIGNED_TICK_SCAN_COLUMNS,
        TrainingError,
        _aligned_metadata_signature,
        _aligned_tick_masks,
        _aligned_tick_stream_detail,
        _validate_aligned_tick_summary,
    )

    def _record(kind_counter: str, detail: dict[str, Any]) -> None:
        failure_counts[kind_counter] += 1
        if len(failures) < MAX_VALIDATION_DETAILS_PER_JOB:
            failures.append(detail)

    stored_rows = _scan_tick_window(lake, job, window, _ALIGNED_TICK_SCAN_COLUMNS)
    stored_by_tick: dict[int, dict[str, dict[str, Any]]] = {}
    for row in stored_rows:
        tick = int(row["tick_index"])
        if tick not in rows_by_tick:
            continue
        try:
            stream_detail = _aligned_tick_stream_detail(row)
            masks = _aligned_tick_masks(row)
        except TrainingError as exc:
            _record(
                "jsonb_failures",
                {
                    "kind": "jsonb-round-trip",
                    "alignment_id": str(job["alignment_id"]),
                    "tick_index": tick,
                    "aligned_tick_id": row.get("aligned_tick_id"),
                    "detail": str(exc),
                },
            )
            continue
        try:
            _validate_aligned_tick_summary(row, stream_detail, masks)
        except TrainingError as exc:
            _record(
                "summary_mismatches",
                {
                    "kind": "summary-mismatch",
                    "alignment_id": str(job["alignment_id"]),
                    "tick_index": tick,
                    "aligned_tick_id": row.get("aligned_tick_id"),
                    "detail": str(exc),
                },
            )
            continue
        stored_by_tick[tick] = stream_detail
    for tick in sorted(rows_by_tick):
        expected = _aligned_metadata_signature(tick, rows_by_tick[tick], streams)
        actual = _aligned_metadata_signature(tick, stored_by_tick.get(tick, {}), streams)
        if expected == actual:
            continue
        diffs = _signature_diff(expected, actual) or [("*", "*", None, None)]
        for stream, column, expected_value, actual_value in diffs:
            _record(
                "metadata_mismatches",
                {
                    "kind": "metadata-mismatch",
                    "alignment_id": str(job["alignment_id"]),
                    "tick_index": tick,
                    "stream": stream,
                    "column": column,
                    "expected": expected_value,
                    "actual": actual_value,
                },
            )


def _signature_diff(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> list[tuple[str, str, Any, Any]]:
    diffs: list[tuple[str, str, Any, Any]] = []
    expected_streams = expected.get("streams") or {}
    actual_streams = actual.get("streams") or {}
    for stream in expected_streams:
        expected_columns = expected_streams.get(stream) or {}
        actual_columns = actual_streams.get(stream) or {}
        for column in expected_columns:
            if expected_columns.get(column) != actual_columns.get(column):
                diffs.append(
                    (
                        str(stream),
                        str(column),
                        expected_columns.get(column),
                        actual_columns.get(column),
                    )
                )
    return diffs


def _recipe_digest(job: Mapping[str, Any]) -> str:
    from lancedb_robotics.training import _alignment_recipe_digest

    return _alignment_recipe_digest(job)


def _job_report(
    job: Mapping[str, Any],
    *,
    status: str,
    reason: str | None,
    source_frame_rows: int = 0,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "alignment_id": str(job["alignment_id"]),
        "alignment_name": str(job["name"]),
        "recipe_digest": _recipe_digest(job),
        "status": status,
        "source_frame_rows": int(source_frame_rows),
    }
    if reason is not None:
        report["reason"] = reason
    return report


def _post_migration_maintenance(lake: Lake) -> dict[str, Any]:
    """Compact ``aligned_ticks`` and refresh its scalar indexes after a sweep.

    Never fails the migration: unsupported backends record an explicit
    ``skipped`` reason (0128 gate / 0129 managed-versioning guard) and engine
    errors record ``failed`` with the message — the written tick rows stand
    either way.
    """
    from lancedb_robotics.indexing import (
        ALIGNED_TICK_PREDICATE_INDEX_COLUMNS,
        build_scalar_indexes,
    )
    from lancedb_robotics.pylance_execution import require_namespace_write_supported

    result: dict[str, Any] = {
        "table": "aligned_ticks",
        "status": "completed",
        "reason": None,
        "compaction": None,
        "indexes": [],
    }
    capability_reason = lake_capability_reason(lake, MAINTENANCE)
    if capability_reason is not None:
        result["status"] = "skipped"
        result["reason"] = capability_reason
        return result
    try:
        require_namespace_write_supported(lake.connection_spec, "aligned_ticks")
    except Exception as exc:  # noqa: BLE001 - explicit skip, not a migration failure.
        result["status"] = "skipped"
        result["reason"] = str(exc)
        return result
    try:
        metrics = lake.table("aligned_ticks").to_lance().optimize.compact_files()
        result["compaction"] = {
            name: int(getattr(metrics, name))
            for name in ("fragments_removed", "fragments_added", "files_removed", "files_added")
            if getattr(metrics, name, None) is not None
        }
        result["indexes"] = [
            index.to_params()
            for index in build_scalar_indexes(
                lake,
                table="aligned_ticks",
                columns=ALIGNED_TICK_PREDICATE_INDEX_COLUMNS,
                replace=True,
            )
        ]
    except Exception as exc:  # noqa: BLE001 - maintenance is best-effort post-sweep.
        result["status"] = "failed"
        result["reason"] = f"post-migration maintenance failed: {exc}"
    return result


def _build_report(
    lake: Lake,
    job_reports: list[dict[str, Any]],
    *,
    dry_run: bool,
    verify: bool,
    replace: bool,
    table_created: bool,
    table_would_create: bool,
    maintenance: dict[str, Any] | None,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    from lancedb_robotics.training import _stable_digest

    counts = {
        "jobs_scanned": len(job_reports),
        "jobs_migrated": sum(1 for job in job_reports if job["status"] in _JOB_MIGRATED_STATUSES),
        "jobs_already_migrated": sum(
            1 for job in job_reports if job["status"] == "already-migrated"
        ),
        "jobs_planned": sum(1 for job in job_reports if job["status"] == "planned"),
        "jobs_skipped": sum(1 for job in job_reports if job["status"] == "skipped"),
        "jobs_failed": sum(1 for job in job_reports if job["status"] == "failed"),
        "aligned_ticks_written": sum(int(job.get("ticks_written") or 0) for job in job_reports),
        "source_aligned_frame_rows": sum(
            int(job.get("source_frame_rows") or 0) for job in job_reports
        ),
    }
    validation = {"metadata_mismatches": 0, "jsonb_failures": 0, "summary_mismatches": 0}
    for job in job_reports:
        for key in validation:
            validation[key] += int((job.get("validation") or {}).get(key) or 0)
    payload = {
        "jobs": [
            {
                "alignment_id": job["alignment_id"],
                "recipe_digest": job["recipe_digest"],
                "status": job["status"],
            }
            for job in job_reports
        ],
        "dry_run": dry_run,
        "replace": replace,
    }
    return {
        "report_version": MIGRATION_REPORT_VERSION,
        "migration_id": "atm-" + _stable_digest(payload),
        "lake_uri": getattr(lake, "uri", None),
        "dry_run": bool(dry_run),
        "verify": bool(verify),
        "replace": bool(replace),
        "aligned_ticks_table_created": bool(table_created),
        "aligned_ticks_table_would_create": bool(table_would_create),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        **counts,
        "validation": validation,
        "maintenance": maintenance,
        "jobs": job_reports,
    }


def _record_job_transform(
    lake: Lake,
    job: Mapping[str, Any],
    job_report: Mapping[str, Any],
    *,
    created_by: str,
) -> str:
    """Record one idempotent transform_runs row for this job's migration state.

    ``transform_id`` is content-addressed by (alignment, recipe digest, tick
    schema version), so re-running the migration replaces the same row instead
    of accumulating duplicates, mirroring how alignment views record
    themselves in ``_record_alignment_job``.
    """
    from lancedb_robotics.lineage import emit_transform_lineage
    from lancedb_robotics.training import (
        ALIGNED_TICKS_SCHEMA_VERSION,
        _alignment_input_versions,
        _stable_digest,
    )

    transform_id = "tfm-atm-" + _stable_digest(
        {
            "alignment_id": job_report["alignment_id"],
            "recipe_digest": job_report["recipe_digest"],
            "schema_version": ALIGNED_TICKS_SCHEMA_VERSION,
        }
    )
    now = datetime.now(UTC)
    params = {
        key: job_report.get(key)
        for key in (
            "alignment_id",
            "alignment_name",
            "recipe_digest",
            "status",
            "tick_index_range",
            "ticks_expected",
            "ticks_existing",
            "ticks_written",
            "source_frame_rows",
            "replaced",
            "validation",
        )
    }
    params["report_version"] = MIGRATION_REPORT_VERSION
    row = {
        "transform_id": transform_id,
        "kind": MIGRATION_TRANSFORM_KIND,
        "source_id": None,
        "input_uris": [],
        "input_table_versions": list(_alignment_input_versions(job)),
        "output_tables": ["aligned_ticks"],
        "params": json.dumps(params, sort_keys=True),
        "status": "completed" if job_report["status"] != "failed" else "failed",
        "error": job_report.get("reason"),
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    # Single-commit upsert rather than delete+add: a crash between the two
    # commits would lose the prior audit row, and concurrent migrators could
    # double-insert (BUG-04).
    _merge_insert_with_retry(
        lake.table("transform_runs"),
        "transform_id",
        pa.Table.from_pylist([row], schema=TRANSFORM_RUNS_SCHEMA),
        update_matched=True,
    )
    emit_transform_lineage(lake, row)
    return transform_id
