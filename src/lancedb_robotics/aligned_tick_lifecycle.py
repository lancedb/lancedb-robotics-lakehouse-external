"""``aligned_ticks`` compaction and retention lifecycle at scale (backlog 0135).

Backlog 0078 made recorded alignments write the ``aligned_ticks`` JSONB training
table (one row per policy tick), keeping the normalized ``aligned_frames`` rows
as a compatibility surface, and 0134 shipped the lake-wide, resumable migration
that populates ``aligned_ticks`` from ``aligned_frames``. Repeated backfills,
recipe revisions, and interrupted/concurrent writers accumulate three kinds of
cruft on the hot training surface over time:

- **Duplicate rows** — more than one physical row sharing an ``aligned_tick_id``
  (content-addressed by ``(alignment_id, tick_index)``). A recipe revision or a
  re-materialized job appends a fresh row for a tick that already has one; a
  crash between a legacy delete+add leaves stragglers. Duplicates inflate scans
  and can shadow the current sample with a stale copy.
- **Stale rows** — tick rows whose ``recipe_digest`` no longer matches the
  recorded alignment job's current recipe.
- **Orphan rows** — rows for an ``alignment_id`` that no longer has a recorded
  ``alignment_jobs`` row at all.

This module is the retention/compaction lifecycle for that surface:

- **Diagnose** (:func:`diagnose_aligned_ticks`): report table size, row counts,
  duplicate counts, stale-row counts, and per-alignment posture for
  ``aligned_ticks`` and the compatibility ``aligned_frames`` rows, plus the
  ``aligned_ticks``/``aligned_frames`` table versions currently pinned by
  dataset snapshots, lineage references, and active retention/evidence holds.
- **Clean up** (:func:`cleanup_aligned_ticks`): dry-run by default. Collapses
  duplicate ids to a single canonical row (preferring the row matching the
  current recipe), optionally removes orphan-alignment rows, then compacts and
  refreshes the aligned scalar indexes, recording an auditable ``transform_runs``
  row + inline lineage on apply.

Two invariants make this safe at scale (see ``SKILLS.md``):

- **Row-level only; never prune versions.** Every removal is a delete on the
  *current* table version, which creates a new version and leaves prior
  versions (and their rows) intact. Any dataset snapshot, lineage reference, or
  active retention/evidence hold that pins an older ``aligned_ticks``/
  ``aligned_frames`` version therefore keeps its exact rows — this lifecycle
  never runs ``cleanup_old_versions`` (that stays with ``lake maintain``, which
  already tags pinned versions before pruning). Pinned/held versions are
  surfaced in the report as proof they are preserved, not touched.
- **Crash-safe, converging dedup.** Collapsing duplicate physical rows that
  share a key cannot be a single Lance ``merge_insert`` (a key with duplicate
  target rows is not collapsed by an upsert), so dedup re-adds the canonical row
  stamped with a per-window ``created_at`` marker strictly greater than every
  existing copy, then deletes the older copies by ``created_at < marker``. A
  crash after the add leaves an extra copy (a re-run collapses it) and a crash
  mid-delete leaves stragglers (a re-run collapses them) — a tick never drops to
  zero rows, and two concurrent runs converge.

Bounded memory throughout: both tables are read in tick-index windows via range
predicates (never a per-alignment collect-all, never a wide id ``IN`` list); the
full-row scan needed to rebuild a canonical row is taken only for the windows
that actually contain duplicates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from lancedb_robotics.capability_gates import MAINTENANCE, lake_capability_reason
from lancedb_robotics.schemas import (
    ALIGNED_FRAMES_SCHEMA,
    ALIGNED_TICKS_SCHEMA,
    TRANSFORM_RUNS_SCHEMA,
)

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from lancedb_robotics.lake import Lake

#: Versioned report contract, mirroring ``aligned-tick-migration/1`` (0134).
LIFECYCLE_REPORT_VERSION = "aligned-tick-lifecycle/1"

#: ``transform_runs.kind`` for an applied lifecycle run.
LIFECYCLE_TRANSFORM_KIND = "aligned-ticks-lifecycle"

#: Ticks per scan/dedup window. Bounds peak memory to roughly ``tick_window``
#: rows (light columns) per alignment regardless of lake size.
DEFAULT_TICK_WINDOW = 1024

#: Rows per re-add append inside a window.
DEFAULT_WRITE_BATCH_SIZE = 512

#: Arrow batch size for bounded scans.
_SCAN_BATCH_SIZE = 4096

#: Per-alignment cap on recorded per-recipe digest breakdown so a pathological
#: job cannot balloon the report; counts stay exact.
_MAX_RECIPE_DIGESTS_PER_ALIGNMENT = 16

_JOB_COLUMNS = (
    "alignment_id",
    "name",
    "recipe",
    "output_table",
    "input_versions",
    "transform_id",
    "created_at",
)

#: Light columns sufficient to plan dedup/staleness without touching JSONB.
_TICK_PLAN_COLUMNS = ("aligned_tick_id", "tick_index", "recipe_digest", "created_at")
_FRAME_PLAN_COLUMNS = ("aligned_frame_id", "tick_index", "transform_id", "created_at")

_LIFECYCLE_TABLES = ("aligned_ticks", "aligned_frames")


class AlignedTickLifecycleError(Exception):
    """Raised when the lifecycle cannot be planned or executed."""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def diagnose_aligned_ticks(
    lake: Lake,
    *,
    alignments: Sequence[str] | None = None,
    include_frames: bool = True,
    tick_window: int = DEFAULT_TICK_WINDOW,
) -> dict[str, Any]:
    """Report ``aligned_ticks``/``aligned_frames`` size and retention posture.

    ``alignments`` selects jobs by alignment id or name (``None`` means every
    alignment present in the recorded jobs and in the two tables). The returned
    :data:`LIFECYCLE_REPORT_VERSION` report carries, per alignment, the tick/frame
    row counts, distinct-id counts, duplicate counts, stale-row counts, the count
    of ticks that carry only stale rows (and so need re-materialization), and
    whether the alignment is orphaned. Table-level totals plus the
    ``aligned_ticks``/``aligned_frames`` versions pinned by snapshots, lineage,
    or active retention/evidence holds are included. This call never writes.

    Memory is bounded (one tick-index window at a time); wall-clock scales with
    the number of alignments times rows because each alignment is scanned under
    an ``alignment_id`` predicate. That predicate is cheap once
    ``build_aligned_training_predicate_indexes`` (the BTREE on ``alignment_id``)
    has run — which the apply path's post-cleanup maintenance builds/refreshes.
    """
    if tick_window <= 0:
        raise AlignedTickLifecycleError("tick_window must be positive")
    generated_at = datetime.now(UTC)
    plan = _plan_lifecycle(
        lake,
        alignments=alignments,
        include_frames=include_frames,
        tick_window=tick_window,
    )
    return _diagnose_report(lake, plan, generated_at=generated_at, tick_window=tick_window)


def cleanup_aligned_ticks(
    lake: Lake,
    *,
    alignments: Sequence[str] | None = None,
    dry_run: bool = True,
    remove_duplicates: bool = True,
    remove_orphans: bool = False,
    include_frames: bool = True,
    tick_window: int = DEFAULT_TICK_WINDOW,
    batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
    created_by: str = "aligned-tick-lifecycle",
) -> dict[str, Any]:
    """Compact and retention-clean ``aligned_ticks`` (+compatibility frames).

    Dry-run by default: the returned report always carries the plan (removable
    duplicate/orphan row counts, skipped protected alignments). When
    ``dry_run=False`` the plan is applied — duplicates are collapsed to their
    canonical row, orphan rows removed when ``remove_orphans=True``, and the
    tables compacted with their aligned scalar indexes refreshed — and one
    idempotent ``transform_runs`` row + inline lineage is recorded.

    Row removal happens only on the current table version; prior versions (and
    any snapshot/lineage/hold that pins them) keep their rows untouched. This
    lifecycle never prunes versions — use ``lake maintain`` for that.
    """
    if tick_window <= 0:
        raise AlignedTickLifecycleError("tick_window must be positive")
    if batch_size <= 0:
        raise AlignedTickLifecycleError("batch_size must be positive")

    started_at = datetime.now(UTC)
    plan = _plan_lifecycle(
        lake,
        alignments=alignments,
        include_frames=include_frames,
        tick_window=tick_window,
    )
    report = _diagnose_report(lake, plan, generated_at=started_at, tick_window=tick_window)
    report.update(
        {
            "dry_run": bool(dry_run),
            "remove_duplicates": bool(remove_duplicates),
            "remove_orphans": bool(remove_orphans),
            "include_frames": bool(include_frames),
        }
    )
    report["plan"] = _plan_summary(
        plan,
        remove_duplicates=remove_duplicates,
        remove_orphans=remove_orphans,
        include_frames=include_frames,
    )

    if dry_run:
        return report

    applied = _apply_lifecycle(
        lake,
        plan,
        remove_duplicates=remove_duplicates,
        remove_orphans=remove_orphans,
        include_frames=include_frames,
        tick_window=tick_window,
        batch_size=batch_size,
        started_at=started_at,
        created_by=created_by,
    )
    report["applied"] = applied
    return report


# --------------------------------------------------------------------------- #
# Planning (read-only)
# --------------------------------------------------------------------------- #
def _plan_lifecycle(
    lake: Lake,
    *,
    alignments: Sequence[str] | None,
    include_frames: bool,
    tick_window: int,
) -> dict[str, Any]:
    """Build the read-only lifecycle plan shared by diagnose and cleanup."""
    table_names = set(lake.table_names())
    ticks_exists = "aligned_ticks" in table_names
    frames_exists = include_frames and "aligned_frames" in table_names

    jobs = _recorded_jobs(lake) if "alignment_jobs" in table_names else {}
    held_versions = _held_table_versions(lake)

    # Union of alignment ids seen in the jobs table and in the two lifecycle
    # tables (bounded by alignment count, not row count).
    present: set[str] = set(jobs)
    if ticks_exists:
        present |= _distinct_alignment_ids(lake, "aligned_ticks")
    if frames_exists:
        present |= _distinct_alignment_ids(lake, "aligned_frames")

    selected = _select_alignment_ids(present, jobs, alignments)

    alignments_plan: list[dict[str, Any]] = []
    for alignment_id in sorted(selected):
        job = jobs.get(alignment_id)
        alignments_plan.append(
            _plan_alignment(
                lake,
                alignment_id=alignment_id,
                job=job,
                ticks_exists=ticks_exists,
                frames_exists=frames_exists,
                tick_window=tick_window,
            )
        )
    return {
        "ticks_exists": ticks_exists,
        "frames_exists": frames_exists,
        "held_versions": held_versions,
        "alignments": alignments_plan,
    }


def _plan_alignment(
    lake: Lake,
    *,
    alignment_id: str,
    job: Mapping[str, Any] | None,
    ticks_exists: bool,
    frames_exists: bool,
    tick_window: int,
) -> dict[str, Any]:
    current_recipe = _alignment_recipe_digest(job) if job is not None else None
    current_transform = str(job["transform_id"]) if job and job.get("transform_id") else None

    tick_stats = (
        _scan_key_windows(
            lake,
            table="aligned_ticks",
            alignment_id=alignment_id,
            key_column="aligned_tick_id",
            plan_columns=_TICK_PLAN_COLUMNS,
            current_marker=current_recipe,
            marker_column="recipe_digest",
            tick_window=tick_window,
        )
        if ticks_exists
        else _empty_key_stats()
    )
    frame_stats = (
        _scan_key_windows(
            lake,
            table="aligned_frames",
            alignment_id=alignment_id,
            key_column="aligned_frame_id",
            plan_columns=_FRAME_PLAN_COLUMNS,
            current_marker=current_transform,
            marker_column="transform_id",
            tick_window=tick_window,
        )
        if frames_exists
        else _empty_key_stats()
    )
    return {
        "alignment_id": alignment_id,
        "alignment_name": str(job["name"]) if job and job.get("name") else None,
        "orphan": job is None,
        "current_recipe_digest": current_recipe,
        "current_transform_id": current_transform,
        "ticks": tick_stats,
        "frames": frame_stats,
    }


def _scan_key_windows(
    lake: Lake,
    *,
    table: str,
    alignment_id: str,
    key_column: str,
    plan_columns: Sequence[str],
    current_marker: str | None,
    marker_column: str,
    tick_window: int,
) -> dict[str, Any]:
    """Fold duplicate/stale posture for one alignment over tick-index windows.

    Bounded memory: only ``tick_window`` light rows are held at a time, and the
    per-window ``created_at`` maxima needed later for crash-safe dedup are kept
    per *occupied* window, never the full row set.
    """
    bounds = _tick_index_bounds(lake, table, alignment_id, tick_window=tick_window)
    stats = _empty_key_stats()
    if bounds is None:
        return stats
    lo, hi, _row_count, windows = bounds
    stats["tick_index_range"] = [lo, hi]

    duplicate_windows: list[list[int]] = []
    recipe_counts: dict[str, int] = {}
    for window_id in windows:
        window = (window_id * tick_window, (window_id + 1) * tick_window)
        rows = _scan_window(lake, table, alignment_id, window, plan_columns)
        if not rows:
            continue
        stats["rows"] += len(rows)
        # Group by the content-addressed key, not tick_index: aligned_frames has
        # one row per (tick, stream), so several distinct keys share a tick.
        by_key: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_key.setdefault(str(row[key_column]), []).append(row)
            marker_value = row.get(marker_column)
            if marker_column == "recipe_digest" and marker_value is not None:
                recipe_counts[str(marker_value)] = recipe_counts.get(str(marker_value), 0) + 1
        stats["distinct_ids"] += len(by_key)
        window_has_duplicate = False
        for key_rows in by_key.values():
            if current_marker is not None:
                fresh = sum(1 for r in key_rows if str(r.get(marker_column)) == current_marker)
                stats["fresh_rows"] += fresh
                stats["stale_rows"] += len(key_rows) - fresh
                if fresh == 0:
                    stats["ids_needing_rematerialize"] += 1
            if len(key_rows) > 1:
                stats["duplicate_rows"] += len(key_rows) - 1
                window_has_duplicate = True
        if window_has_duplicate:
            duplicate_windows.append([window[0], window[1]])
    stats["duplicate_windows"] = duplicate_windows
    stats["recipe_digests"] = dict(
        sorted(recipe_counts.items(), key=lambda item: (-item[1], item[0]))[
            :_MAX_RECIPE_DIGESTS_PER_ALIGNMENT
        ]
    )
    return stats


def _empty_key_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "distinct_ids": 0,
        "duplicate_rows": 0,
        "fresh_rows": 0,
        "stale_rows": 0,
        "ids_needing_rematerialize": 0,
        "tick_index_range": None,
        "duplicate_windows": [],
        "recipe_digests": {},
    }


# --------------------------------------------------------------------------- #
# Apply (write)
# --------------------------------------------------------------------------- #
def _apply_lifecycle(
    lake: Lake,
    plan: Mapping[str, Any],
    *,
    remove_duplicates: bool,
    remove_orphans: bool,
    include_frames: bool,
    tick_window: int,
    batch_size: int,
    started_at: datetime,
    created_by: str,
) -> dict[str, Any]:
    tick_version_before = int(lake.table("aligned_ticks").version) if plan["ticks_exists"] else None
    frame_version_before = (
        int(lake.table("aligned_frames").version)
        if include_frames and plan["frames_exists"]
        else None
    )

    removed = {
        "duplicate_tick_rows_removed": 0,
        "duplicate_frame_rows_removed": 0,
        "orphan_tick_rows_removed": 0,
        "orphan_frame_rows_removed": 0,
        "orphan_alignments_removed": 0,
    }
    for alignment in plan["alignments"]:
        alignment_id = str(alignment["alignment_id"])
        if remove_orphans and alignment["orphan"]:
            removed["orphan_tick_rows_removed"] += _delete_alignment_rows(
                lake, "aligned_ticks", alignment_id
            ) if plan["ticks_exists"] else 0
            if include_frames and plan["frames_exists"]:
                removed["orphan_frame_rows_removed"] += _delete_alignment_rows(
                    lake, "aligned_frames", alignment_id
                )
            if alignment["ticks"]["rows"] or alignment["frames"]["rows"]:
                removed["orphan_alignments_removed"] += 1
            continue
        if not remove_duplicates:
            continue
        if plan["ticks_exists"]:
            removed["duplicate_tick_rows_removed"] += _dedup_alignment(
                lake,
                table="aligned_ticks",
                alignment_id=alignment_id,
                key_column="aligned_tick_id",
                schema=ALIGNED_TICKS_SCHEMA,
                current_marker=alignment["current_recipe_digest"],
                marker_column="recipe_digest",
                duplicate_windows=alignment["ticks"]["duplicate_windows"],
                batch_size=batch_size,
            )
        if include_frames and plan["frames_exists"]:
            removed["duplicate_frame_rows_removed"] += _dedup_alignment(
                lake,
                table="aligned_frames",
                alignment_id=alignment_id,
                key_column="aligned_frame_id",
                schema=ALIGNED_FRAMES_SCHEMA,
                current_marker=alignment["current_transform_id"],
                marker_column="transform_id",
                duplicate_windows=alignment["frames"]["duplicate_windows"],
                batch_size=batch_size,
            )

    rows_changed = any(
        removed[key]
        for key in (
            "duplicate_tick_rows_removed",
            "duplicate_frame_rows_removed",
            "orphan_tick_rows_removed",
            "orphan_frame_rows_removed",
        )
    )
    maintenance = _post_cleanup_maintenance(lake, include_frames=include_frames) if rows_changed else None

    applied: dict[str, Any] = dict(removed)
    applied["tick_version_before"] = tick_version_before
    applied["tick_version_after"] = (
        int(lake.table("aligned_ticks").version) if plan["ticks_exists"] else None
    )
    applied["frame_version_before"] = frame_version_before
    applied["frame_version_after"] = (
        int(lake.table("aligned_frames").version)
        if include_frames and plan["frames_exists"]
        else None
    )
    applied["maintenance"] = maintenance
    # Record an audit row only when the lifecycle actually changed rows. A
    # no-op run (nothing to remove) must not append a transform_runs row or emit
    # lineage on every invocation -- that would churn the lineage graph forever
    # for a scheduled/retried cleanup that already converged.
    applied["transform_id"] = (
        _record_transform(
            lake,
            plan,
            removed=removed,
            maintenance=maintenance,
            remove_duplicates=remove_duplicates,
            remove_orphans=remove_orphans,
            include_frames=include_frames,
            started_at=started_at,
            created_by=created_by,
        )
        if rows_changed
        else None
    )
    return applied


def _dedup_alignment(
    lake: Lake,
    *,
    table: str,
    alignment_id: str,
    key_column: str,
    schema: pa.Schema,
    current_marker: str | None,
    marker_column: str,
    duplicate_windows: Sequence[Sequence[int]],
    batch_size: int,
) -> int:
    """Collapse duplicate-key rows to one canonical row per id, crash-safely.

    For each window with duplicates, the canonical row per id (the row matching
    the current recipe/transform where present, else the newest) is re-added
    with ``created_at`` set to a per-window marker strictly greater than every
    existing copy, then the older copies are deleted by ``created_at < marker``.
    A crash or a concurrent run converges (a tick never drops to zero rows).
    """
    handle = lake.table(table)
    column_names = [f.name for f in schema]
    removed = 0
    for window_bounds in duplicate_windows:
        window = (int(window_bounds[0]), int(window_bounds[1]))
        rows = _scan_window(lake, table, alignment_id, window, column_names)
        by_key: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_key.setdefault(str(row[key_column]), []).append(row)
        canonical_rows: list[dict[str, Any]] = []
        max_created: datetime | None = None
        for key_rows in by_key.values():
            if len(key_rows) <= 1:
                continue
            removed += len(key_rows) - 1
            canonical_rows.append(_canonical_row(key_rows, current_marker, marker_column))
            for row in key_rows:
                created = _as_dt(row.get("created_at"))
                if created is not None and (max_created is None or created > max_created):
                    max_created = created
        if not canonical_rows:
            continue
        # A single marker strictly greater than every existing copy in the
        # window: each re-added canonical (stamped == marker) survives the
        # `< marker` delete while every prior copy (< marker) is removed. Falls
        # back to now() when no copy carried a timestamp.
        base = max_created or datetime.now(UTC)
        marker = base + timedelta(microseconds=1)
        dup_ids = [str(row[key_column]) for row in canonical_rows]
        for row in canonical_rows:
            row["created_at"] = marker
        for chunk in _chunks(canonical_rows, batch_size):
            handle.add(pa.Table.from_pylist(list(chunk), schema=schema))
        # The just-added canonicals carry created_at == marker; every prior copy
        # is strictly older (< marker) or has a NULL created_at (a writer left
        # it unset). `NULL < ts` is NULL in SQL, so the NULL branch is explicit
        # -- otherwise a NULL-timestamp straggler would survive forever and the
        # run would report a false-positive removal while never converging.
        predicate = " AND ".join(
            [
                _sql_predicate("alignment_id", alignment_id),
                _sql_predicate(key_column, dup_ids),
                f"(created_at < {_ts_literal(marker)} OR created_at IS NULL)",
            ]
        )
        handle.delete(predicate)
    return removed


def _canonical_row(
    rows: Sequence[Mapping[str, Any]],
    current_marker: str | None,
    marker_column: str,
) -> dict[str, Any]:
    """Pick the surviving row: current recipe/transform first, then newest."""

    def _sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        matches_current = current_marker is not None and str(row.get(marker_column)) == current_marker
        created = _as_dt(row.get("created_at"))
        created_key = created.timestamp() if created is not None else float("-inf")
        return (1 if matches_current else 0, created_key, str(row.get("transform_id") or ""))

    return dict(max(rows, key=_sort_key))


def _delete_alignment_rows(lake: Lake, table: str, alignment_id: str) -> int:
    handle = lake.table(table)
    before = int(handle.count_rows(_sql_predicate("alignment_id", alignment_id)))
    if before == 0:
        return 0
    handle.delete(_sql_predicate("alignment_id", alignment_id))
    return before


def _post_cleanup_maintenance(lake: Lake, *, include_frames: bool) -> dict[str, Any]:
    """Compact the touched tables and refresh their aligned scalar indexes.

    Never fails the lifecycle: unsupported backends record an explicit
    ``skipped`` reason (0128 gate / 0129 managed-versioning guard); engine
    errors record ``failed`` with the message — the row removals stand either
    way. Follows the compact -> index order (no version prune).
    """
    from lancedb_robotics.indexing import build_aligned_training_predicate_indexes
    from lancedb_robotics.pylance_execution import require_namespace_write_supported

    result: dict[str, Any] = {
        "status": "completed",
        "reason": None,
        "tables": [],
        "indexes": [],
    }
    capability_reason = lake_capability_reason(lake, MAINTENANCE)
    if capability_reason is not None:
        result["status"] = "skipped"
        result["reason"] = capability_reason
        return result
    tables = ["aligned_ticks"] + (["aligned_frames"] if include_frames else [])
    tables = [table for table in tables if table in set(lake.table_names())]
    try:
        for table in tables:
            require_namespace_write_supported(lake.connection_spec, table)
    except Exception as exc:  # noqa: BLE001 - explicit skip, not a failure.
        result["status"] = "skipped"
        result["reason"] = str(exc)
        return result
    try:
        for table in tables:
            metrics = lake.table(table).to_lance().optimize.compact_files()
            result["tables"].append(
                {
                    "table": table,
                    "compaction": {
                        name: int(getattr(metrics, name))
                        for name in (
                            "fragments_removed",
                            "fragments_added",
                            "files_removed",
                            "files_added",
                        )
                        if getattr(metrics, name, None) is not None
                    },
                }
            )
        result["indexes"] = [
            index.to_params()
            for index in build_aligned_training_predicate_indexes(
                lake,
                include_frames=include_frames,
                replace=True,
            )
        ]
    except Exception as exc:  # noqa: BLE001 - best-effort post-cleanup.
        result["status"] = "failed"
        result["reason"] = f"post-cleanup maintenance failed: {exc}"
    return result


def _record_transform(
    lake: Lake,
    plan: Mapping[str, Any],
    *,
    removed: Mapping[str, int],
    maintenance: Mapping[str, Any] | None,
    remove_duplicates: bool,
    remove_orphans: bool,
    include_frames: bool,
    started_at: datetime,
    created_by: str,
) -> str:
    """Record one content-addressed ``transform_runs`` row + inline lineage."""
    from lancedb_robotics.lineage import emit_transform_lineage

    now = datetime.now(UTC)
    output_tables = ["aligned_ticks"] + (["aligned_frames"] if include_frames else [])
    output_tables = [table for table in output_tables if table in set(lake.table_names())]
    params = {
        "report_version": LIFECYCLE_REPORT_VERSION,
        "remove_duplicates": bool(remove_duplicates),
        "remove_orphans": bool(remove_orphans),
        "include_frames": bool(include_frames),
        "alignment_ids": sorted(str(a["alignment_id"]) for a in plan["alignments"]),
        "removed": dict(removed),
        "maintenance": maintenance,
    }
    # Content-addressed by what changed (not wall-clock): two runs that remove
    # the same rows for the same alignments upsert the same row rather than
    # accumulating duplicates, mirroring the 0134 migration's transform id.
    transform_id = "tfm-atl-" + _stable_digest(
        {
            "alignment_ids": params["alignment_ids"],
            "removed": params["removed"],
            "output_tables": output_tables,
        }
    )
    row = {
        "transform_id": transform_id,
        "kind": LIFECYCLE_TRANSFORM_KIND,
        "source_id": None,
        "input_uris": [],
        "input_table_versions": [
            {"table": table, "version": int(lake.table(table).version), "tag": ""}
            for table in output_tables
        ],
        "output_tables": output_tables,
        "params": json.dumps(params, sort_keys=True, default=str),
        "status": "completed",
        "error": None,
        "started_at": started_at,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    # Single-commit upsert (not delete+add): concurrent runs converge and a
    # crash cannot lose a prior audit row.
    _merge_insert_transform_with_retry(lake.table("transform_runs"), row)
    emit_transform_lineage(lake, row)
    return transform_id


#: merge_insert attempts before surfacing a Lance optimistic-concurrency
#: commit conflict; the upsert is idempotent so a preempted commit converges.
_MERGE_INSERT_ATTEMPTS = 3


def _merge_insert_transform_with_retry(table: Any, row: Mapping[str, Any]) -> None:
    data = pa.Table.from_pylist([dict(row)], schema=TRANSFORM_RUNS_SCHEMA)
    last_error: BaseException | None = None
    for _ in range(_MERGE_INSERT_ATTEMPTS):
        try:
            (
                table.merge_insert("transform_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(data)
            )
            return
        except Exception as exc:  # noqa: BLE001 - retry only the retryable conflict.
            if "commit conflict" not in str(exc).lower():
                raise
            last_error = exc
    raise AlignedTickLifecycleError(
        f"transform_runs upsert kept hitting commit conflicts after "
        f"{_MERGE_INSERT_ATTEMPTS} attempts: {last_error}"
    )


# --------------------------------------------------------------------------- #
# Report shaping
# --------------------------------------------------------------------------- #
def _diagnose_report(
    lake: Lake,
    plan: Mapping[str, Any],
    *,
    generated_at: datetime,
    tick_window: int,
) -> dict[str, Any]:
    totals = {
        "aligned_tick_rows": 0,
        "distinct_tick_ids": 0,
        "duplicate_tick_rows": 0,
        "stale_tick_rows": 0,
        "ticks_needing_rematerialize": 0,
        "aligned_frame_rows": 0,
        "distinct_frame_ids": 0,
        "duplicate_frame_rows": 0,
        "stale_frame_rows": 0,
        "orphan_alignments": 0,
        "orphan_tick_rows": 0,
        "orphan_frame_rows": 0,
    }
    alignment_reports: list[dict[str, Any]] = []
    for alignment in plan["alignments"]:
        ticks = alignment["ticks"]
        frames = alignment["frames"]
        totals["aligned_tick_rows"] += ticks["rows"]
        totals["distinct_tick_ids"] += ticks["distinct_ids"]
        totals["duplicate_tick_rows"] += ticks["duplicate_rows"]
        totals["stale_tick_rows"] += ticks["stale_rows"]
        totals["ticks_needing_rematerialize"] += ticks["ids_needing_rematerialize"]
        totals["aligned_frame_rows"] += frames["rows"]
        totals["distinct_frame_ids"] += frames["distinct_ids"]
        totals["duplicate_frame_rows"] += frames["duplicate_rows"]
        totals["stale_frame_rows"] += frames["stale_rows"]
        if alignment["orphan"]:
            totals["orphan_alignments"] += 1
            totals["orphan_tick_rows"] += ticks["rows"]
            totals["orphan_frame_rows"] += frames["rows"]
        alignment_reports.append(_alignment_report(alignment))

    return {
        "report_version": LIFECYCLE_REPORT_VERSION,
        "lake_uri": getattr(lake, "uri", None),
        "generated_at": generated_at.isoformat(),
        "tick_window": tick_window,
        "tables": _tables_report(lake, plan),
        "totals": totals,
        "alignments": alignment_reports,
        "version_cleanup": (
            "not-performed (row-level lifecycle only; run `lake maintain` for version "
            "pruning, which tags snapshot/lineage/hold-pinned versions before pruning)"
        ),
    }


def _alignment_report(alignment: Mapping[str, Any]) -> dict[str, Any]:
    ticks = alignment["ticks"]
    frames = alignment["frames"]
    return {
        "alignment_id": alignment["alignment_id"],
        "alignment_name": alignment["alignment_name"],
        "orphan": alignment["orphan"],
        "current_recipe_digest": alignment["current_recipe_digest"],
        "current_transform_id": alignment["current_transform_id"],
        "tick_index_range": ticks["tick_index_range"],
        "aligned_tick_rows": ticks["rows"],
        "distinct_tick_ids": ticks["distinct_ids"],
        "duplicate_tick_rows": ticks["duplicate_rows"],
        "fresh_tick_rows": ticks["fresh_rows"],
        "stale_tick_rows": ticks["stale_rows"],
        "ticks_needing_rematerialize": ticks["ids_needing_rematerialize"],
        "recipe_digests": ticks["recipe_digests"],
        "aligned_frame_rows": frames["rows"],
        "distinct_frame_ids": frames["distinct_ids"],
        "duplicate_frame_rows": frames["duplicate_rows"],
        "stale_frame_rows": frames["stale_rows"],
    }


def _tables_report(lake: Lake, plan: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for table in _LIFECYCLE_TABLES:
        exists = plan["ticks_exists"] if table == "aligned_ticks" else plan["frames_exists"]
        if not exists:
            report[table] = {"exists": False}
            continue
        report[table] = {
            "exists": True,
            "rows": int(lake.table(table).count_rows()),
            "fragments": _fragment_count(lake, table),
            "pinned_versions": list(plan["held_versions"].get(table, ())),
        }
    return report


def _plan_summary(
    plan: Mapping[str, Any],
    *,
    remove_duplicates: bool,
    remove_orphans: bool,
    include_frames: bool,
) -> dict[str, Any]:
    dup_ticks = 0
    dup_frames = 0
    orphan_alignments = 0
    orphan_tick_rows = 0
    orphan_frame_rows = 0
    for alignment in plan["alignments"]:
        if alignment["orphan"]:
            if remove_orphans:
                orphan_alignments += 1
                orphan_tick_rows += alignment["ticks"]["rows"]
                orphan_frame_rows += alignment["frames"]["rows"] if include_frames else 0
            continue
        if remove_duplicates:
            dup_ticks += alignment["ticks"]["duplicate_rows"]
            if include_frames:
                dup_frames += alignment["frames"]["duplicate_rows"]
    return {
        "duplicate_tick_rows_removable": dup_ticks,
        "duplicate_frame_rows_removable": dup_frames,
        "orphan_alignments_removable": orphan_alignments,
        "orphan_tick_rows_removable": orphan_tick_rows,
        "orphan_frame_rows_removable": orphan_frame_rows,
        "orphan_alignments_retained": sum(
            1 for a in plan["alignments"] if a["orphan"] and not remove_orphans
        ),
    }


# --------------------------------------------------------------------------- #
# Bounded scan helpers
# --------------------------------------------------------------------------- #
def _distinct_alignment_ids(lake: Lake, table: str) -> set[str]:
    """Stream ``alignment_id`` and return the distinct set (bounded by count)."""
    ids: set[str] = set()
    query = lake.table(table).search().select(["alignment_id"])
    for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
        for value in batch.column("alignment_id").to_pylist():
            if value is not None:
                ids.add(str(value))
    return ids


def _tick_index_bounds(
    lake: Lake,
    table: str,
    alignment_id: str,
    *,
    tick_window: int,
) -> tuple[int, int, int, tuple[int, ...]] | None:
    """Fold (min, max, row count, occupied window ids) over one alignment.

    Bounded memory: keeps one int per *occupied* window, never the tick list.
    """
    query = (
        lake.table(table)
        .search()
        .select(["tick_index"])
        .where(_sql_predicate("alignment_id", alignment_id))
    )
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


def _scan_window(
    lake: Lake,
    table: str,
    alignment_id: str,
    window: tuple[int, int],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    predicate = " AND ".join(
        [
            _sql_predicate("alignment_id", alignment_id),
            f"tick_index >= {int(window[0])}",
            f"tick_index < {int(window[1])}",
        ]
    )
    query = lake.table(table).search().select(list(columns)).where(predicate)
    rows: list[dict[str, Any]] = []
    for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
        rows.extend(batch.to_pylist())
    return rows


def _fragment_count(lake: Lake, table: str) -> int:
    try:
        return len(lake.table(table).to_lance().get_fragments())
    except Exception:  # noqa: BLE001 - fragment count is diagnostic, not load-bearing.
        return -1


# --------------------------------------------------------------------------- #
# Jobs, holds, selection
# --------------------------------------------------------------------------- #
def _recorded_jobs(lake: Lake) -> dict[str, dict[str, Any]]:
    """Return the latest recorded job row per alignment id (bounded scan)."""
    latest: dict[str, dict[str, Any]] = {}
    query = lake.table("alignment_jobs").search().select(list(_JOB_COLUMNS))
    for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
        for row in batch.to_pylist():
            alignment_id = row.get("alignment_id")
            if not alignment_id:
                continue
            alignment_id = str(alignment_id)
            current = latest.get(alignment_id)
            if current is None or row["created_at"] > current["created_at"]:
                latest[alignment_id] = row
    return latest


def _held_table_versions(lake: Lake) -> dict[str, tuple[dict[str, Any], ...]]:
    """Return the ``aligned_ticks``/``aligned_frames`` versions pinned by
    dataset snapshots or active retention/evidence holds.

    These versions are preserved by construction (this lifecycle never prunes
    versions); they are surfaced so the report proves what is protected. The
    lookup is bounded — a streamed ``dataset_snapshots`` scan and a
    ``kind='table-version' AND table_name IN (...)`` predicate read of
    ``lineage_artifacts`` (bounded by version count) — so it never loads the
    whole lineage graph the way the prune-time ``lineage_retention_pin_details``
    does (that stays with ``lake maintain``; here it would be BUG-13 shape).
    """
    table_names = set(lake.table_names())
    pins: dict[str, dict[int, dict[str, set[str]]]] = {
        table: {} for table in _LIFECYCLE_TABLES
    }

    def _add(table: str, version: int, reason: str) -> None:
        if table not in pins:
            return
        detail = pins[table].setdefault(int(version), {"reasons": set()})
        detail["reasons"].add(reason)

    if "dataset_snapshots" in table_names:
        try:
            query = (
                lake.table("dataset_snapshots").search().select(["name", "table_versions"])
            )
            for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
                for row in batch.to_pylist():
                    label = row.get("name") or "dataset-snapshot"
                    for version in row.get("table_versions") or []:
                        table = version.get("table")
                        if table in pins and version.get("version") is not None:
                            _add(table, int(version["version"]), f"dataset-snapshot:{label}")
        except Exception:  # noqa: BLE001 - pins are advisory diagnostics here.
            pass

    if "lineage_artifacts" in table_names:
        try:
            from lancedb_robotics.lineage import _retention_hold_from_artifact_row

            predicate = " AND ".join(
                [
                    _sql_predicate("kind", "table-version"),
                    _sql_predicate("table_name", list(_LIFECYCLE_TABLES)),
                ]
            )
            query = (
                lake.table("lineage_artifacts")
                .search()
                .select(["artifact_id", "table_name", "table_version", "metadata"])
                .where(predicate)
            )
            for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
                for row in batch.to_pylist():
                    table = row.get("table_name")
                    version = row.get("table_version")
                    if table not in pins or version is None:
                        continue
                    hold = _retention_hold_from_artifact_row(getattr(lake, "uri", ""), row)
                    if hold is not None and hold.active:
                        _add(table, int(version), f"retention-hold:{hold.reason or 'active'}")
        except Exception:  # noqa: BLE001 - pins are advisory diagnostics here.
            pass

    held: dict[str, tuple[dict[str, Any], ...]] = {}
    for table in _LIFECYCLE_TABLES:
        if pins[table]:
            held[table] = tuple(
                {"version": version, "reasons": sorted(detail["reasons"])}
                for version, detail in sorted(pins[table].items())
            )
    return held


def _select_alignment_ids(
    present: set[str],
    jobs: Mapping[str, Mapping[str, Any]],
    alignments: Sequence[str] | None,
) -> set[str]:
    if alignments is None:
        return set(present)
    requested = [str(item) for item in alignments if str(item)]
    if not requested:
        raise AlignedTickLifecycleError(
            "alignments must name at least one alignment id or name"
        )
    by_name: dict[str, str] = {}
    for alignment_id, job in jobs.items():
        name = job.get("name")
        if name:
            by_name.setdefault(str(name), alignment_id)
    selected: set[str] = set()
    unknown: list[str] = []
    for item in requested:
        if item in present:
            selected.add(item)
        elif item in by_name:
            selected.add(by_name[item])
        else:
            unknown.append(item)
    if unknown:
        raise AlignedTickLifecycleError(
            f"unknown alignment ids or names {unknown}; list recorded alignments "
            "via the alignment_jobs table"
        )
    return selected


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _alignment_recipe_digest(job: Mapping[str, Any]) -> str:
    from lancedb_robotics.training import _alignment_recipe_digest as _digest

    return _digest(job)


def _stable_digest(payload: Any) -> str:
    from lancedb_robotics.training import _stable_digest as _digest

    return _digest(payload)


def _sql_predicate(column: str, expected: Any) -> str:
    from lancedb_robotics.training import _sql_predicate as _predicate

    return _predicate(column, expected)


def _ts_literal(value: datetime) -> str:
    """Timestamp SQL literal for a ``created_at`` comparison."""
    return "timestamp '" + value.astimezone(UTC).isoformat() + "'"


def _as_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return None


def _chunks(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]
