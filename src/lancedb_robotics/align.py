"""Multi-rate temporal alignment over canonical observation rows (backlog 0031)."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
from lancedb.query import ColumnOrdering

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.schemas import (
    ALIGNED_FRAMES_SCHEMA,
    ALIGNED_TICKS_SCHEMA,
    ALIGNMENT_JOBS_SCHEMA,
    TRANSFORM_RUNS_SCHEMA,
)

SUPPORTED_INTERPOLATION = ("nearest", "previous", "linear")
SUB_FRAME_BOUND_NS = 1_000_000

_CLOCK_FIELDS = {
    "timestamp_ns": "timestamp_ns",
    "robot_time_ns": "timestamp_ns",
    "header_time_ns": "timestamp_ns",
    "hardware_time_ns": "timestamp_ns",
    "receive_time_ns": "raw_log_time_ns",
    "raw_log_time_ns": "raw_log_time_ns",
}
_ROW_ID_COLUMN = "_rowid"
_ALIGNMENT_METADATA_COLUMNS = (
    "observation_id",
    "run_id",
    "timestamp_ns",
    "raw_log_time_ns",
    "raw_sequence",
    "topic",
    "sensor_id",
    "modality",
    "raw_channel",
)
_SOURCE_VALUE_COLUMNS = (
    "observation_id",
    "state_vector",
    "action_vector",
    "payload_json",
)
DEFAULT_BATCH_SIZE = 8192


class AlignmentError(Exception):
    """Raised when an aligned view cannot be created."""


@dataclass(frozen=True)
class AlignmentView:
    """Deterministic aligned view on a common clock.

    ``quality_summary["sub_frame_bound_ns"]`` documents the numerical bound used
    by correctness tests: every joined source error is expected to be no larger
    than this bound, and linear interpolation lands exactly on the query clock.
    """

    lake_uri: str
    name: str
    alignment_id: str
    transform_id: str
    rate_hz: float
    clock: str
    streams: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    quality_summary: dict[str, Any]
    quality_flags: tuple[str, ...]
    output_table: str


@dataclass(frozen=True)
class _TimedRow:
    time_ns: int
    row: dict[str, Any]
    latency_ns: int
    row_id: int


@dataclass
class _StreamingMetrics:
    segment_count: int = 0
    metadata_rows_scanned: int = 0
    source_rows_hydrated: int = 0
    output_rows_written: int = 0
    compatibility_frame_rows_written: int = 0
    max_in_memory_segment_size: int = 0
    row_count: int = 0
    confidence_sum: float = 0.0
    min_row_confidence: float | None = None
    quality_flags: set[str] = field(default_factory=set)

    def observe_segment(self, rows: Sequence[dict[str, Any]]) -> None:
        self.segment_count += 1
        self.max_in_memory_segment_size = max(self.max_in_memory_segment_size, len(rows))
        for row in rows:
            confidence = float(row["confidence"])
            self.row_count += 1
            self.confidence_sum += confidence
            if self.min_row_confidence is None:
                self.min_row_confidence = confidence
            else:
                self.min_row_confidence = min(self.min_row_confidence, confidence)
            self.quality_flags.update(row.get("quality_flags") or [])


class LakeAlign:
    """Convenience namespace exposed as ``lake.align``."""

    def __init__(self, lake: Lake) -> None:
        self._lake = lake

    def create_view(
        self,
        name: str,
        *,
        source: str = "observations",
        clock: str = "timestamp_ns",
        rate_hz: float,
        streams: Sequence[str],
        tolerance_ms: float | None = None,
        interpolation: Mapping[str, str] | str | None = None,
        latency_ns: Mapping[str, int] | None = None,
        run_id: str | None = None,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
        created_by: str = "lancedb-robotics",
        materialize: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> AlignmentView:
        """Resample named observation streams onto one query clock and record lineage."""

        return create_alignment_view(
            self._lake,
            name=name,
            source=source,
            clock=clock,
            rate_hz=rate_hz,
            streams=streams,
            tolerance_ms=tolerance_ms,
            interpolation=interpolation,
            latency_ns=latency_ns,
            run_id=run_id,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            created_by=created_by,
            materialize=materialize,
            batch_size=batch_size,
            record=True,
        )

    def window(
        self,
        *,
        name: str,
        rate_hz: float,
        streams: Sequence[str],
        run_id: str | None = None,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
        clock: str = "timestamp_ns",
        tolerance_ms: float | None = None,
        interpolation: Mapping[str, str] | str | None = None,
        latency_ns: Mapping[str, int] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> AlignmentView:
        """Compute an aligned view without writing lineage rows.

        This is the read-path used by ``Episode.window``. Public workflows that
        need durable recipe lineage should call :meth:`create_view`.
        """

        return create_alignment_view(
            self._lake,
            name=name,
            source="observations",
            clock=clock,
            rate_hz=rate_hz,
            streams=streams,
            tolerance_ms=tolerance_ms,
            interpolation=interpolation,
            latency_ns=latency_ns,
            run_id=run_id,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            materialize=False,
            batch_size=batch_size,
            record=False,
        )


def create_alignment_view(
    lake: Lake,
    *,
    name: str,
    source: str = "observations",
    clock: str = "timestamp_ns",
    rate_hz: float,
    streams: Sequence[str],
    tolerance_ms: float | None = None,
    interpolation: Mapping[str, str] | str | None = None,
    latency_ns: Mapping[str, int] | None = None,
    run_id: str | None = None,
    start_time_ns: int | None = None,
    end_time_ns: int | None = None,
    created_by: str = "lancedb-robotics",
    materialize: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    record: bool = True,
) -> AlignmentView:
    """Build an aligned observation view at ``rate_hz``.

    The v0 engine is intentionally explicit: it reads canonical ``observations``
    rows, distinguishes hardware/header timestamps from receive timestamps via
    ``clock``, applies per-stream latency correction, and emits deterministic
    quality diagnostics for tolerance misses, clock drift, and dropped frames.
    """

    if not name:
        raise AlignmentError("name is required")
    if source != "observations":
        raise AlignmentError("only source='observations' is supported in this release")
    if rate_hz <= 0:
        raise AlignmentError("rate_hz must be positive")
    if batch_size <= 0:
        raise AlignmentError("batch_size must be positive")
    if not streams:
        raise AlignmentError("at least one stream is required")
    clock_field = _clock_field(clock)
    selected_streams = _normalize_streams(streams)
    interpolation_by_stream = _normalize_interpolation(selected_streams, interpolation)
    latency_by_stream = _normalize_latency(selected_streams, latency_ns or {})
    tolerance_ns = _tolerance_ns(tolerance_ms)
    step_ns = max(1, round(1_000_000_000 / float(rate_hz)))
    sub_frame_bound_ns = _sub_frame_bound(step_ns, tolerance_ns)

    if record and materialize:
        return _create_streaming_materialized_alignment_view(
            lake,
            name=name,
            source=source,
            clock=clock,
            clock_field=clock_field,
            rate_hz=rate_hz,
            selected_streams=selected_streams,
            interpolation_by_stream=interpolation_by_stream,
            latency_by_stream=latency_by_stream,
            tolerance_ms=tolerance_ms,
            tolerance_ns=tolerance_ns,
            run_id=run_id,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            created_by=created_by,
            batch_size=batch_size,
            step_ns=step_ns,
            sub_frame_bound_ns=sub_frame_bound_ns,
        )

    observations = _observation_refs(
        lake,
        run_id=run_id,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        batch_size=batch_size,
    )
    rows_by_stream = {
        stream: _timed_rows(
            observations,
            stream=stream,
            clock_field=clock_field,
            latency_ns=latency_by_stream[stream],
        )
        for stream in selected_streams
    }
    resolved_start, resolved_end = _resolve_bounds(
        lake,
        rows_by_stream,
        run_id=run_id,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
    )
    ticks = _query_ticks(resolved_start, resolved_end, step_ns)

    stream_stats = {
        stream: _initial_stream_stats(rows_by_stream[stream], step_ns, tolerance_ns)
        for stream in selected_streams
    }
    aligned_rows: list[dict[str, Any]] = []
    for index, timestamp_ns in enumerate(ticks):
        aligned_streams: dict[str, Any] = {}
        row_flags: set[str] = set()
        row_confidences: list[float] = []
        for stream in selected_streams:
            result = _align_one_stream(
                stream=stream,
                target_ns=timestamp_ns,
                timed_rows=rows_by_stream[stream],
                method=interpolation_by_stream[stream],
                tolerance_ns=tolerance_ns,
                step_ns=step_ns,
                stats=stream_stats[stream],
            )
            aligned_streams[stream] = result
            row_flags.update(result["quality_flags"])
            row_confidences.append(float(result["confidence"]))
        aligned_rows.append(
            {
                "index": index,
                "timestamp_ns": timestamp_ns,
                "streams": aligned_streams,
                "quality_flags": sorted(row_flags),
                "confidence": min(row_confidences) if row_confidences else 0.0,
            }
        )
    aligned_rows = _hydrate_aligned_rows(lake, aligned_rows, batch_size=batch_size)

    quality_summary, quality_flags = _quality_summary(
        rows=aligned_rows,
        stream_stats=stream_stats,
        sub_frame_bound_ns=sub_frame_bound_ns,
    )
    recipe = {
        "name": name,
        "source": source,
        "clock": clock,
        "clock_field": clock_field,
        "rate_hz": float(rate_hz),
        "streams": list(selected_streams),
        "tolerance_ms": tolerance_ms,
        "interpolation": interpolation_by_stream,
        "latency_ns": latency_by_stream,
        "run_id": run_id,
        "start_time_ns": resolved_start,
        "end_time_ns": resolved_end,
        "sub_frame_bound_ns": sub_frame_bound_ns,
        "execution": "row-id-plan",
        "materialize": bool(record and materialize),
        "batch_size": int(batch_size),
    }
    input_versions = _table_versions(lake, ("runs", "observations"))
    alignment_id = f"aln-{_digest({'recipe': recipe, 'input_versions': input_versions})}"
    transform_id = f"tfm-align-{_digest({'alignment_id': alignment_id, 'recipe': recipe})}"
    output_table = "aligned_ticks" if record and materialize else f"virtual:alignment_view:{name}"
    view = AlignmentView(
        lake_uri=lake.uri,
        name=name,
        alignment_id=alignment_id,
        transform_id=transform_id,
        rate_hz=float(rate_hz),
        clock=clock,
        streams=selected_streams,
        rows=tuple(aligned_rows),
        quality_summary={
            **quality_summary,
            "alignment_id": alignment_id,
            "transform_id": transform_id,
            "source_rows_read": _source_rows_read(aligned_rows),
            "execution": "row-id-plan",
            "output_table": output_table,
        },
        quality_flags=tuple(quality_flags),
        output_table=output_table,
    )
    if record:
        if materialize:
            _materialize_aligned_frames(
                lake,
                view=view,
                transform_id=transform_id,
                batch_size=batch_size,
            )
        _record_alignment_job(
            lake,
            view=view,
            recipe=recipe,
            input_versions=input_versions,
            created_by=created_by,
        )
    return view


def _create_streaming_materialized_alignment_view(
    lake: Lake,
    *,
    name: str,
    source: str,
    clock: str,
    clock_field: str,
    rate_hz: float,
    selected_streams: tuple[str, ...],
    interpolation_by_stream: dict[str, str],
    latency_by_stream: dict[str, int],
    tolerance_ms: float | None,
    tolerance_ns: int | None,
    run_id: str | None,
    start_time_ns: int | None,
    end_time_ns: int | None,
    created_by: str,
    batch_size: int,
    step_ns: int,
    sub_frame_bound_ns: int,
) -> AlignmentView:
    resolved_start, resolved_end = _resolve_streaming_bounds(
        lake,
        selected_streams=selected_streams,
        clock_field=clock_field,
        latency_by_stream=latency_by_stream,
        run_id=run_id,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        batch_size=batch_size,
    )
    recipe = {
        "name": name,
        "source": source,
        "clock": clock,
        "clock_field": clock_field,
        "rate_hz": float(rate_hz),
        "streams": list(selected_streams),
        "tolerance_ms": tolerance_ms,
        "interpolation": interpolation_by_stream,
        "latency_ns": latency_by_stream,
        "run_id": run_id,
        "start_time_ns": resolved_start,
        "end_time_ns": resolved_end,
        "sub_frame_bound_ns": sub_frame_bound_ns,
        "execution": "streaming-row-id-plan",
        "materialize": True,
        "batch_size": int(batch_size),
    }
    input_versions = _table_versions(lake, ("runs", "observations"))
    alignment_id = f"aln-{_digest({'recipe': recipe, 'input_versions': input_versions})}"
    transform_id = f"tfm-align-{_digest({'alignment_id': alignment_id, 'recipe': recipe})}"
    output_table = "aligned_ticks"
    recipe_digest = _recipe_digest(recipe)
    stream_stats, metadata_rows_scanned = _streaming_stream_stats(
        lake,
        selected_streams=selected_streams,
        clock_field=clock_field,
        latency_by_stream=latency_by_stream,
        run_id=run_id,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        step_ns=step_ns,
        tolerance_ns=tolerance_ns,
        batch_size=batch_size,
    )
    metrics = _StreamingMetrics(metadata_rows_scanned=metadata_rows_scanned)
    preview_rows: list[dict[str, Any]] = []

    frames_table = lake.table("aligned_frames")
    ticks_table = lake.table("aligned_ticks")
    frames_table.delete(f"alignment_id = {_sql_literal(alignment_id)}")
    ticks_table.delete(f"alignment_id = {_sql_literal(alignment_id)}")
    now = datetime.now(UTC)
    for tick_start_index, ticks in _iter_tick_segments(
        resolved_start,
        resolved_end,
        step_ns,
        batch_size,
    ):
        segment_refs, segment_scanned = _segment_observation_refs(
            lake,
            selected_streams=selected_streams,
            clock_field=clock_field,
            latency_by_stream=latency_by_stream,
            run_id=run_id,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            segment_start_ns=ticks[0],
            segment_end_ns=ticks[-1],
            batch_size=batch_size,
        )
        metrics.metadata_rows_scanned += segment_scanned
        rows_by_stream = {
            stream: _timed_rows(
                segment_refs,
                stream=stream,
                clock_field=clock_field,
                latency_ns=latency_by_stream[stream],
            )
            for stream in selected_streams
        }
        segment_rows: list[dict[str, Any]] = []
        for offset, timestamp_ns in enumerate(ticks):
            aligned_streams: dict[str, Any] = {}
            row_flags: set[str] = set()
            row_confidences: list[float] = []
            for stream in selected_streams:
                result = _align_one_stream(
                    stream=stream,
                    target_ns=timestamp_ns,
                    timed_rows=rows_by_stream[stream],
                    method=interpolation_by_stream[stream],
                    tolerance_ns=tolerance_ns,
                    step_ns=step_ns,
                    stats=stream_stats[stream],
                )
                aligned_streams[stream] = result
                row_flags.update(result["quality_flags"])
                row_confidences.append(float(result["confidence"]))
            segment_rows.append(
                {
                    "index": tick_start_index + offset,
                    "timestamp_ns": timestamp_ns,
                    "streams": aligned_streams,
                    "quality_flags": sorted(row_flags),
                    "confidence": min(row_confidences) if row_confidences else 0.0,
                }
            )
        segment_rows = _hydrate_aligned_rows(
            lake,
            segment_rows,
            batch_size=batch_size,
            metrics=metrics,
        )
        metrics.observe_segment(segment_rows)
        metrics.output_rows_written += _append_aligned_tick_rows(
            ticks_table,
            alignment_id=alignment_id,
            alignment_name=name,
            recipe_digest=recipe_digest,
            transform_id=transform_id,
            rows=segment_rows,
            batch_size=batch_size,
            created_at=now,
        )
        metrics.compatibility_frame_rows_written += _append_aligned_frame_rows(
            frames_table,
            alignment_id=alignment_id,
            transform_id=transform_id,
            rows=segment_rows,
            batch_size=batch_size,
            created_at=now,
        )
        if len(preview_rows) < batch_size:
            remaining = batch_size - len(preview_rows)
            preview_rows.extend(segment_rows[:remaining])

    quality_summary, quality_flags = _quality_summary(
        rows=preview_rows,
        stream_stats=stream_stats,
        sub_frame_bound_ns=sub_frame_bound_ns,
        row_metrics=metrics,
    )
    view = AlignmentView(
        lake_uri=lake.uri,
        name=name,
        alignment_id=alignment_id,
        transform_id=transform_id,
        rate_hz=float(rate_hz),
        clock=clock,
        streams=selected_streams,
        rows=tuple(preview_rows),
        quality_summary={
            **quality_summary,
            "alignment_id": alignment_id,
            "transform_id": transform_id,
            "source_rows_read": metrics.source_rows_hydrated,
            "source_rows_hydrated": metrics.source_rows_hydrated,
            "execution": "streaming-row-id-plan",
            "segment_count": metrics.segment_count,
            "metadata_rows_scanned": metrics.metadata_rows_scanned,
            "output_rows_written": metrics.output_rows_written,
            "compatibility_frame_rows_written": metrics.compatibility_frame_rows_written,
            "max_in_memory_segment_size": metrics.max_in_memory_segment_size,
            "preview_rows": len(preview_rows),
            "output_table": output_table,
        },
        quality_flags=tuple(quality_flags),
        output_table=output_table,
    )
    _record_alignment_job(
        lake,
        view=view,
        recipe=recipe,
        input_versions=input_versions,
        created_by=created_by,
    )
    return view


def _clock_field(clock: str) -> str:
    normalized = clock.strip()
    try:
        return _CLOCK_FIELDS[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(_CLOCK_FIELDS))
        raise AlignmentError(f"unknown clock {clock!r}; expected one of {choices}") from exc


def _normalize_streams(streams: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(stream) for stream in streams if str(stream))
    if not selected:
        raise AlignmentError("at least one stream is required")
    if len(set(selected)) != len(selected):
        raise AlignmentError("streams must be unique")
    return selected


def _normalize_interpolation(
    streams: tuple[str, ...],
    interpolation: Mapping[str, str] | str | None,
) -> dict[str, str]:
    if interpolation is None:
        return {stream: "nearest" for stream in streams}
    if isinstance(interpolation, str):
        method = _validate_method(interpolation)
        return {stream: method for stream in streams}

    values: dict[str, str] = {}
    normalized_keys = {_normalize_stream_key(key): value for key, value in interpolation.items()}
    for stream in streams:
        raw = interpolation.get(stream)
        if raw is None:
            raw = normalized_keys.get(_normalize_stream_key(stream), "nearest")
        values[stream] = _validate_method(raw)
    return values


def _validate_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in SUPPORTED_INTERPOLATION:
        choices = ", ".join(SUPPORTED_INTERPOLATION)
        raise AlignmentError(f"unsupported interpolation {method!r}; expected one of {choices}")
    return normalized


def _normalize_latency(streams: tuple[str, ...], values: Mapping[str, int]) -> dict[str, int]:
    normalized_keys = {_normalize_stream_key(key): int(value) for key, value in values.items()}
    latency: dict[str, int] = {}
    for stream in streams:
        raw = values.get(stream)
        if raw is None:
            raw = normalized_keys.get(_normalize_stream_key(stream), 0)
        latency[stream] = int(raw)
    return latency


def _tolerance_ns(tolerance_ms: float | None) -> int | None:
    if tolerance_ms is None:
        return None
    if tolerance_ms < 0:
        raise AlignmentError("tolerance_ms must be non-negative")
    return int(round(float(tolerance_ms) * 1_000_000))


def _observation_refs(
    lake: Lake,
    *,
    run_id: str | None,
    start_time_ns: int | None,
    end_time_ns: int | None,
    batch_size: int,
) -> list[dict[str, Any]]:
    if start_time_ns is not None and end_time_ns is not None and end_time_ns < start_time_ns:
        raise AlignmentError("end_time_ns must be greater than or equal to start_time_ns")
    filter_sql = _observation_filter(run_id, start_time_ns, end_time_ns)
    rows = []
    query = lake.table("observations").search().select(list(_ALIGNMENT_METADATA_COLUMNS))
    if filter_sql:
        query = query.where(filter_sql)
    batches = (
        query.with_row_id(True)
        .order_by(
            [
                ColumnOrdering(column_name="timestamp_ns"),
                ColumnOrdering(column_name="raw_sequence"),
                ColumnOrdering(column_name="topic"),
                ColumnOrdering(column_name="observation_id"),
            ]
        )
        .to_batches(batch_size=batch_size)
    )
    for batch in batches:
        for row in batch.to_pylist():
            row_id = row.get(_ROW_ID_COLUMN)
            if row_id is None:
                raise AlignmentError("Lance scan did not return row ids for observations")
            rows.append({**row, _ROW_ID_COLUMN: int(row_id)})
    rows.sort(key=lambda row: _observation_sort_key(row))
    return rows


def _observation_filter(
    run_id: str | None,
    start_time_ns: int | None,
    end_time_ns: int | None,
) -> str | None:
    clauses = []
    if run_id is not None:
        clauses.append(f"run_id = {_sql_literal(run_id)}")
    if start_time_ns is not None:
        clauses.append(f"timestamp_ns >= {int(start_time_ns)}")
    if end_time_ns is not None:
        clauses.append(f"timestamp_ns <= {int(end_time_ns)}")
    return " AND ".join(clauses) if clauses else None


def _resolve_streaming_bounds(
    lake: Lake,
    *,
    selected_streams: tuple[str, ...],
    clock_field: str,
    latency_by_stream: dict[str, int],
    run_id: str | None,
    start_time_ns: int | None,
    end_time_ns: int | None,
    batch_size: int,
) -> tuple[int, int]:
    if run_id is not None or (start_time_ns is not None and end_time_ns is not None):
        return _resolve_bounds(
            lake,
            {stream: () for stream in selected_streams},
            run_id=run_id,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
        )

    min_time: int | None = start_time_ns
    max_time: int | None = end_time_ns
    filter_sql = _observation_filter(run_id, start_time_ns, end_time_ns)
    for row in _iter_observation_refs(
        lake,
        filter_sql=filter_sql,
        clock_field=clock_field,
        ascending=True,
        batch_size=batch_size,
    ):
        for stream in selected_streams:
            if not _matches_stream(row, stream):
                continue
            time_ns = _metadata_time_ns(
                row,
                clock_field=clock_field,
                latency_ns=latency_by_stream[stream],
            )
            min_time = time_ns if min_time is None else min(min_time, time_ns)
            max_time = time_ns if max_time is None else max(max_time, time_ns)
    if min_time is None or max_time is None:
        raise AlignmentError("no observations matched the requested streams")
    if max_time < min_time:
        raise AlignmentError("end_time_ns must be greater than or equal to start_time_ns")
    return int(min_time), int(max_time)


def _streaming_stream_stats(
    lake: Lake,
    *,
    selected_streams: tuple[str, ...],
    clock_field: str,
    latency_by_stream: dict[str, int],
    run_id: str | None,
    start_time_ns: int | None,
    end_time_ns: int | None,
    step_ns: int,
    tolerance_ns: int | None,
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    stats = {stream: _empty_stream_stats() for stream in selected_streams}
    previous_times: dict[str, int | None] = {stream: None for stream in selected_streams}
    intervals: dict[str, list[int]] = {stream: [] for stream in selected_streams}
    latency_ranges: dict[str, list[int]] = {stream: [] for stream in selected_streams}
    metadata_rows_scanned = 0
    filter_sql = _observation_filter(run_id, start_time_ns, end_time_ns)
    for row in _iter_observation_refs(
        lake,
        filter_sql=filter_sql,
        clock_field=clock_field,
        ascending=True,
        batch_size=batch_size,
    ):
        metadata_rows_scanned += 1
        for stream in selected_streams:
            if not _matches_stream(row, stream):
                continue
            time_ns = _metadata_time_ns(
                row,
                clock_field=clock_field,
                latency_ns=latency_by_stream[stream],
            )
            stats[stream]["observations"] += 1
            previous = previous_times[stream]
            if previous is not None and time_ns > previous:
                intervals[stream].append(time_ns - previous)
            previous_times[stream] = time_ns
            raw_log_time_ns = row.get("raw_log_time_ns")
            timestamp_ns = row.get("timestamp_ns")
            if raw_log_time_ns is not None and timestamp_ns is not None:
                latency_ranges[stream].append(int(raw_log_time_ns) - int(timestamp_ns))
    for stream in selected_streams:
        _finalize_streaming_stream_stats(
            stats[stream],
            intervals=intervals[stream],
            latencies=latency_ranges[stream],
            step_ns=step_ns,
            tolerance_ns=tolerance_ns,
        )
    return stats, metadata_rows_scanned


def _empty_stream_stats() -> dict[str, Any]:
    return {
        "observations": 0,
        "aligned_count": 0,
        "missing_count": 0,
        "tolerance_exceeded_count": 0,
        "source_errors_ns": None,
        "source_error_count": 0,
        "source_error_sum_ns": 0,
        "max_abs_error_ns": None,
        "median_interval_ns": None,
        "dropped_frame_count": 0,
        "latency_drift_ns": 0,
        "clock_drift_detected": False,
    }


def _finalize_streaming_stream_stats(
    stats: dict[str, Any],
    *,
    intervals: Sequence[int],
    latencies: Sequence[int],
    step_ns: int,
    tolerance_ns: int | None,
) -> None:
    median_interval_ns = int(statistics.median(intervals)) if intervals else None
    dropped_frame_count = 0
    if median_interval_ns and median_interval_ns > 0:
        threshold = median_interval_ns * 1.5
        for interval in intervals:
            if interval > threshold:
                dropped_frame_count += max(1, round(interval / median_interval_ns) - 1)
    latency_drift_ns = max(latencies) - min(latencies) if len(latencies) >= 2 else 0
    drift_threshold_ns = tolerance_ns if tolerance_ns is not None else max(1, step_ns // 2)
    stats["median_interval_ns"] = median_interval_ns
    stats["dropped_frame_count"] = int(dropped_frame_count)
    stats["latency_drift_ns"] = int(latency_drift_ns)
    stats["clock_drift_detected"] = latency_drift_ns > drift_threshold_ns


def _segment_observation_refs(
    lake: Lake,
    *,
    selected_streams: tuple[str, ...],
    clock_field: str,
    latency_by_stream: dict[str, int],
    run_id: str | None,
    start_time_ns: int | None,
    end_time_ns: int | None,
    segment_start_ns: int,
    segment_end_ns: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    clock_column = _metadata_clock_column(clock_field)
    min_latency = min(latency_by_stream.values()) if latency_by_stream else 0
    max_latency = max(latency_by_stream.values()) if latency_by_stream else 0
    base_filter = _observation_filter(run_id, start_time_ns, end_time_ns)
    main_filter = _combine_filters(
        base_filter,
        f"{clock_column} >= {int(segment_start_ns + min_latency)}",
        f"{clock_column} <= {int(segment_end_ns + max_latency)}",
    )
    rows_by_id: dict[int, dict[str, Any]] = {}
    scanned = 0
    for row in _iter_observation_refs(
        lake,
        filter_sql=main_filter,
        clock_field=clock_field,
        ascending=True,
        batch_size=batch_size,
    ):
        scanned += 1
        if _metadata_row_in_segment(
            row,
            selected_streams=selected_streams,
            clock_field=clock_field,
            latency_by_stream=latency_by_stream,
            segment_start_ns=segment_start_ns,
            segment_end_ns=segment_end_ns,
        ):
            rows_by_id[int(row[_ROW_ID_COLUMN])] = row

    before_rows, before_scanned = _boundary_observation_refs(
        lake,
        selected_streams=selected_streams,
        clock_field=clock_field,
        latency_by_stream=latency_by_stream,
        run_id=run_id,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        boundary_ns=segment_start_ns,
        before=True,
        batch_size=batch_size,
    )
    after_rows, after_scanned = _boundary_observation_refs(
        lake,
        selected_streams=selected_streams,
        clock_field=clock_field,
        latency_by_stream=latency_by_stream,
        run_id=run_id,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        boundary_ns=segment_end_ns,
        before=False,
        batch_size=batch_size,
    )
    scanned += before_scanned + after_scanned
    for row in before_rows + after_rows:
        rows_by_id[int(row[_ROW_ID_COLUMN])] = row
    return sorted(rows_by_id.values(), key=_observation_sort_key), scanned


def _metadata_row_in_segment(
    row: dict[str, Any],
    *,
    selected_streams: tuple[str, ...],
    clock_field: str,
    latency_by_stream: dict[str, int],
    segment_start_ns: int,
    segment_end_ns: int,
) -> bool:
    for stream in selected_streams:
        if not _matches_stream(row, stream):
            continue
        time_ns = _metadata_time_ns(
            row,
            clock_field=clock_field,
            latency_ns=latency_by_stream[stream],
        )
        if segment_start_ns <= time_ns <= segment_end_ns:
            return True
    return False


def _boundary_observation_refs(
    lake: Lake,
    *,
    selected_streams: tuple[str, ...],
    clock_field: str,
    latency_by_stream: dict[str, int],
    run_id: str | None,
    start_time_ns: int | None,
    end_time_ns: int | None,
    boundary_ns: int,
    before: bool,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    clock_column = _metadata_clock_column(clock_field)
    min_latency = min(latency_by_stream.values()) if latency_by_stream else 0
    max_latency = max(latency_by_stream.values()) if latency_by_stream else 0
    boundary_filter = (
        f"{clock_column} < {int(boundary_ns + max_latency)}"
        if before
        else f"{clock_column} > {int(boundary_ns + min_latency)}"
    )
    filter_sql = _combine_filters(
        _observation_filter(run_id, start_time_ns, end_time_ns),
        boundary_filter,
    )
    found: dict[str, dict[str, Any]] = {}
    scanned = 0
    for row in _iter_observation_refs(
        lake,
        filter_sql=filter_sql,
        clock_field=clock_field,
        ascending=not before,
        batch_size=batch_size,
    ):
        scanned += 1
        for stream in selected_streams:
            if stream in found or not _matches_stream(row, stream):
                continue
            time_ns = _metadata_time_ns(
                row,
                clock_field=clock_field,
                latency_ns=latency_by_stream[stream],
            )
            if (before and time_ns < boundary_ns) or (not before and time_ns > boundary_ns):
                found[stream] = row
        if len(found) == len(selected_streams):
            break
    return list(found.values()), scanned


def _iter_observation_refs(
    lake: Lake,
    *,
    filter_sql: str | None,
    clock_field: str,
    ascending: bool,
    batch_size: int,
):
    query = lake.table("observations").search().select(list(_ALIGNMENT_METADATA_COLUMNS))
    if filter_sql:
        query = query.where(filter_sql)
    for batch in (
        query.with_row_id(True)
        .order_by(_metadata_ordering(clock_field, ascending=ascending))
        .to_batches(batch_size=batch_size)
    ):
        for row in batch.to_pylist():
            row_id = row.get(_ROW_ID_COLUMN)
            if row_id is None:
                raise AlignmentError("Lance scan did not return row ids for observations")
            yield {**row, _ROW_ID_COLUMN: int(row_id)}


def _metadata_ordering(clock_field: str, *, ascending: bool) -> list[ColumnOrdering]:
    clock_column = _metadata_clock_column(clock_field)
    return [
        ColumnOrdering(column_name=clock_column, ascending=ascending),
        ColumnOrdering(column_name="raw_sequence", ascending=ascending),
        ColumnOrdering(column_name="topic", ascending=ascending),
        ColumnOrdering(column_name="observation_id", ascending=ascending),
    ]


def _metadata_clock_column(clock_field: str) -> str:
    return "raw_log_time_ns" if clock_field == "raw_log_time_ns" else "timestamp_ns"


def _metadata_time_ns(row: dict[str, Any], *, clock_field: str, latency_ns: int) -> int:
    clock_value = row.get(clock_field)
    if clock_value is None:
        clock_value = row.get("timestamp_ns")
    if clock_value is None:
        raise AlignmentError("observation metadata is missing a timestamp")
    return int(clock_value) - int(latency_ns)


def _combine_filters(*filters: str | None) -> str | None:
    clauses = [f"({value})" for value in filters if value]
    return " AND ".join(clauses) if clauses else None


def _iter_tick_segments(
    start_time_ns: int,
    end_time_ns: int,
    step_ns: int,
    batch_size: int,
):
    tick_index = 0
    timestamp_ns = start_time_ns
    while timestamp_ns <= end_time_ns:
        segment: list[int] = []
        segment_start_index = tick_index
        while len(segment) < batch_size and timestamp_ns <= end_time_ns:
            segment.append(timestamp_ns)
            timestamp_ns += step_ns
            tick_index += 1
        yield segment_start_index, tuple(segment)


def _timed_rows(
    observations: Sequence[dict[str, Any]],
    *,
    stream: str,
    clock_field: str,
    latency_ns: int,
) -> tuple[_TimedRow, ...]:
    rows = []
    for row in observations:
        if not _matches_stream(row, stream):
            continue
        clock_value = row.get(clock_field)
        if clock_value is None:
            clock_value = row.get("timestamp_ns")
        if clock_value is None:
            continue
        rows.append(
            _TimedRow(
                time_ns=int(clock_value) - latency_ns,
                row=row,
                latency_ns=latency_ns,
                row_id=int(row[_ROW_ID_COLUMN]),
            )
        )
    return tuple(sorted(rows, key=lambda item: (item.time_ns, _observation_sort_key(item.row))))


def _resolve_bounds(
    lake: Lake,
    rows_by_stream: Mapping[str, tuple[_TimedRow, ...]],
    *,
    run_id: str | None,
    start_time_ns: int | None,
    end_time_ns: int | None,
) -> tuple[int, int]:
    if start_time_ns is not None and end_time_ns is not None:
        return int(start_time_ns), int(end_time_ns)

    run_start: int | None = None
    run_end: int | None = None
    if run_id is not None:
        for batch in (
            lake.table("runs")
            .search()
            .where(f"run_id = {_sql_literal(run_id)}")
            .select(["run_id", "start_time_ns", "end_time_ns"])
            .to_batches(batch_size=1)
        ):
            for row in batch.to_pylist():
                run_start = int(row["start_time_ns"])
                run_end = int(row["end_time_ns"])
                break
            if run_start is not None:
                break

    all_times = [item.time_ns for rows in rows_by_stream.values() for item in rows]
    if not all_times and (run_start is None or run_end is None):
        raise AlignmentError("no observations matched the requested streams")

    resolved_start = int(
        start_time_ns
        if start_time_ns is not None
        else run_start
        if run_start is not None
        else min(all_times)
    )
    resolved_end = int(
        end_time_ns
        if end_time_ns is not None
        else run_end
        if run_end is not None
        else max(all_times)
    )
    if resolved_end < resolved_start:
        raise AlignmentError("end_time_ns must be greater than or equal to start_time_ns")
    return resolved_start, resolved_end


def _query_ticks(start_time_ns: int, end_time_ns: int, step_ns: int) -> tuple[int, ...]:
    if start_time_ns == end_time_ns:
        return (start_time_ns,)
    ticks = []
    timestamp_ns = start_time_ns
    while timestamp_ns <= end_time_ns:
        ticks.append(timestamp_ns)
        timestamp_ns += step_ns
    return tuple(ticks)


def _sub_frame_bound(step_ns: int, tolerance_ns: int | None) -> int:
    candidates = [max(1, step_ns // 2)]
    if tolerance_ns is not None:
        candidates.append(max(1, tolerance_ns))
    candidates.append(SUB_FRAME_BOUND_NS)
    return min(candidates)


def _initial_stream_stats(
    timed_rows: tuple[_TimedRow, ...],
    step_ns: int,
    tolerance_ns: int | None,
) -> dict[str, Any]:
    times = [item.time_ns for item in timed_rows]
    intervals = [
        later - earlier for earlier, later in zip(times, times[1:], strict=False) if later > earlier
    ]
    median_interval_ns = int(statistics.median(intervals)) if intervals else None
    dropped_frame_count = 0
    if median_interval_ns and median_interval_ns > 0:
        threshold = median_interval_ns * 1.5
        for interval in intervals:
            if interval > threshold:
                dropped_frame_count += max(1, round(interval / median_interval_ns) - 1)

    latencies = []
    for item in timed_rows:
        raw_log_time_ns = item.row.get("raw_log_time_ns")
        timestamp_ns = item.row.get("timestamp_ns")
        if raw_log_time_ns is not None and timestamp_ns is not None:
            latencies.append(int(raw_log_time_ns) - int(timestamp_ns))
    latency_drift_ns = max(latencies) - min(latencies) if len(latencies) >= 2 else 0
    drift_threshold_ns = tolerance_ns if tolerance_ns is not None else max(1, step_ns // 2)

    return {
        "observations": len(timed_rows),
        "aligned_count": 0,
        "missing_count": 0,
        "tolerance_exceeded_count": 0,
        "source_errors_ns": [],
        "source_error_count": 0,
        "source_error_sum_ns": 0,
        "max_abs_error_ns": None,
        "median_interval_ns": median_interval_ns,
        "dropped_frame_count": int(dropped_frame_count),
        "latency_drift_ns": int(latency_drift_ns),
        "clock_drift_detected": latency_drift_ns > drift_threshold_ns,
    }


def _record_source_error(stats: dict[str, Any], abs_error_ns: int) -> None:
    if stats.get("source_errors_ns") is not None:
        stats["source_errors_ns"].append(int(abs_error_ns))
    stats["source_error_count"] = int(stats.get("source_error_count") or 0) + 1
    stats["source_error_sum_ns"] = int(stats.get("source_error_sum_ns") or 0) + int(abs_error_ns)
    current_max = stats.get("max_abs_error_ns")
    stats["max_abs_error_ns"] = (
        int(abs_error_ns) if current_max is None else max(int(current_max), int(abs_error_ns))
    )


def _align_one_stream(
    *,
    stream: str,
    target_ns: int,
    timed_rows: tuple[_TimedRow, ...],
    method: str,
    tolerance_ns: int | None,
    step_ns: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    if not timed_rows:
        stats["missing_count"] += 1
        return _missing_result(stream, target_ns, method, "stream-missing")
    if method == "previous":
        candidate = _previous_row(timed_rows, target_ns)
        if candidate is None:
            stats["missing_count"] += 1
            return _missing_result(stream, target_ns, method, "no-previous-sample")
        return _source_result(
            stream,
            target_ns,
            method,
            candidate,
            tolerance_ns,
            step_ns,
            stats,
        )
    if method == "linear":
        exact = _exact_row(timed_rows, target_ns)
        if exact is not None:
            return _source_result(
                stream,
                target_ns,
                method,
                exact,
                tolerance_ns,
                step_ns,
                stats,
            )
        before, after = _bracketing_rows(timed_rows, target_ns)
        if before is None or after is None:
            stats["missing_count"] += 1
            return _missing_result(stream, target_ns, method, "no-linear-bracket")
        if tolerance_ns is not None and (
            target_ns - before.time_ns > tolerance_ns or after.time_ns - target_ns > tolerance_ns
        ):
            stats["missing_count"] += 1
            stats["tolerance_exceeded_count"] += 1
            return _out_of_tolerance_result(
                stream,
                target_ns,
                method,
                nearest=min((before, after), key=lambda row: abs(row.time_ns - target_ns)),
                error_ns=max(target_ns - before.time_ns, after.time_ns - target_ns),
                tolerance_ns=tolerance_ns,
            )
        stats["aligned_count"] += 1
        _record_source_error(stats, 0)
        return {
            "status": "aligned",
            "run_id": before.row.get("run_id"),
            "stream": stream,
            "timestamp_ns": target_ns,
            "interpolation": "linear",
            "observation_id": None,
            "source_observation_ids": [
                before.row.get("observation_id"),
                after.row.get("observation_id"),
            ],
            "source_row_ids": [before.row_id, after.row_id],
            "source_times_ns": [before.time_ns, after.time_ns],
            "source_timestamp_ns": target_ns,
            "source_time_ns": target_ns,
            "receive_time_ns": None,
            "latency_ns": None,
            "error_ns": 0,
            "absolute_error_ns": 0,
            "confidence": 1.0,
            "quality_flags": [],
            "value": None,
        }

    candidate = min(
        timed_rows,
        key=lambda item: (abs(item.time_ns - target_ns), item.time_ns, _observation_sort_key(item.row)),
    )
    return _source_result(
        stream,
        target_ns,
        method,
        candidate,
        tolerance_ns,
        step_ns,
        stats,
    )


def _source_result(
    stream: str,
    target_ns: int,
    method: str,
    candidate: _TimedRow,
    tolerance_ns: int | None,
    step_ns: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    error_ns = candidate.time_ns - target_ns
    abs_error_ns = abs(error_ns)
    if tolerance_ns is not None and abs_error_ns > tolerance_ns:
        stats["missing_count"] += 1
        stats["tolerance_exceeded_count"] += 1
        return _out_of_tolerance_result(
            stream,
            target_ns,
            method,
            nearest=candidate,
            error_ns=abs_error_ns,
            tolerance_ns=tolerance_ns,
        )

    stats["aligned_count"] += 1
    _record_source_error(stats, abs_error_ns)
    return {
        "status": "aligned",
        "run_id": candidate.row.get("run_id"),
        "stream": stream,
        "timestamp_ns": target_ns,
        "interpolation": method,
        "observation_id": candidate.row.get("observation_id"),
        "source_observation_ids": [candidate.row.get("observation_id")],
        "source_row_ids": [candidate.row_id],
        "source_times_ns": [candidate.time_ns],
        "source_timestamp_ns": candidate.row.get("timestamp_ns"),
        "source_time_ns": candidate.time_ns,
        "receive_time_ns": candidate.row.get("raw_log_time_ns"),
        "latency_ns": candidate.latency_ns,
        "error_ns": error_ns,
        "absolute_error_ns": abs_error_ns,
        "confidence": _confidence(abs_error_ns, tolerance_ns, step_ns),
        "quality_flags": [],
        "value": None,
    }


def _missing_result(stream: str, target_ns: int, method: str, reason: str) -> dict[str, Any]:
    return {
        "status": "missing",
        "run_id": None,
        "stream": stream,
        "timestamp_ns": target_ns,
        "interpolation": method,
        "observation_id": None,
        "source_observation_ids": [],
        "source_row_ids": [],
        "source_times_ns": [],
        "source_timestamp_ns": None,
        "source_time_ns": None,
        "receive_time_ns": None,
        "latency_ns": None,
        "error_ns": None,
        "absolute_error_ns": None,
        "confidence": 0.0,
        "quality_flags": [_flag(stream, reason)],
        "value": None,
    }


def _out_of_tolerance_result(
    stream: str,
    target_ns: int,
    method: str,
    *,
    nearest: _TimedRow,
    error_ns: int,
    tolerance_ns: int,
) -> dict[str, Any]:
    return {
        "status": "out_of_tolerance",
        "run_id": nearest.row.get("run_id"),
        "stream": stream,
        "timestamp_ns": target_ns,
        "interpolation": method,
        "observation_id": None,
        "nearest_observation_id": nearest.row.get("observation_id"),
        "source_observation_ids": [],
        "source_row_ids": [],
        "source_times_ns": [],
        "nearest_row_id": nearest.row_id,
        "source_timestamp_ns": nearest.row.get("timestamp_ns"),
        "source_time_ns": nearest.time_ns,
        "receive_time_ns": nearest.row.get("raw_log_time_ns"),
        "latency_ns": nearest.latency_ns,
        "error_ns": nearest.time_ns - target_ns,
        "absolute_error_ns": int(error_ns),
        "confidence": 0.0,
        "quality_flags": [_flag(stream, "tolerance-exceeded")],
        "value": None,
        "tolerance_ns": tolerance_ns,
    }


def _previous_row(timed_rows: tuple[_TimedRow, ...], target_ns: int) -> _TimedRow | None:
    candidate = None
    for item in timed_rows:
        if item.time_ns > target_ns:
            break
        candidate = item
    return candidate


def _exact_row(timed_rows: tuple[_TimedRow, ...], target_ns: int) -> _TimedRow | None:
    for item in timed_rows:
        if item.time_ns == target_ns:
            return item
        if item.time_ns > target_ns:
            break
    return None


def _bracketing_rows(
    timed_rows: tuple[_TimedRow, ...],
    target_ns: int,
) -> tuple[_TimedRow | None, _TimedRow | None]:
    before = None
    after = None
    for item in timed_rows:
        if item.time_ns < target_ns:
            before = item
            continue
        if item.time_ns > target_ns:
            after = item
            break
    return before, after


def _interpolated_value(
    before: dict[str, Any],
    after: dict[str, Any],
    target_ns: int,
    before_time_ns: int,
    after_time_ns: int,
) -> Any:
    before_value = _sample_value(before)
    after_value = _sample_value(after)
    if before_value is None or after_value is None or after_time_ns == before_time_ns:
        return None
    ratio = (target_ns - before_time_ns) / (after_time_ns - before_time_ns)
    if _is_number(before_value) and _is_number(after_value):
        return float(before_value) + (float(after_value) - float(before_value)) * ratio
    if isinstance(before_value, list) and isinstance(after_value, list):
        if len(before_value) != len(after_value):
            return None
        if not all(_is_number(item) for item in before_value + after_value):
            return None
        return [
            float(left) + (float(right) - float(left)) * ratio
            for left, right in zip(before_value, after_value, strict=True)
        ]
    return None


def _sample_value(row: dict[str, Any]) -> Any:
    for key in ("state_vector", "action_vector"):
        value = row.get(key)
        if value:
            return [float(item) for item in value]
    payload = row.get("payload_json")
    if not payload:
        return None
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if _is_number(decoded):
        return float(decoded)
    if isinstance(decoded, list) and all(_is_number(item) for item in decoded):
        return [float(item) for item in decoded]
    if isinstance(decoded, dict):
        for key in ("value", "state", "action"):
            value = decoded.get(key)
            if _is_number(value):
                return float(value)
            if isinstance(value, list) and all(_is_number(item) for item in value):
                return [float(item) for item in value]
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _confidence(abs_error_ns: int, tolerance_ns: int | None, step_ns: int) -> float:
    if abs_error_ns == 0:
        return 1.0
    if tolerance_ns is not None:
        if tolerance_ns == 0:
            return 0.0
        return max(0.0, 1.0 - (abs_error_ns / tolerance_ns))
    return 1.0 / (1.0 + (abs_error_ns / max(1, step_ns)))


def _hydrate_aligned_rows(
    lake: Lake,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    metrics: _StreamingMetrics | None = None,
) -> list[dict[str, Any]]:
    source_row_ids = sorted(
        {
            int(row_id)
            for row in rows
            for result in row["streams"].values()
            for row_id in result.get("source_row_ids", [])
        }
    )
    source_rows = _source_rows_by_observation_id(
        lake,
        source_row_ids,
        batch_size=batch_size,
        metrics=metrics,
    )
    for row in rows:
        for result in row["streams"].values():
            if result["status"] != "aligned":
                continue
            ids = [value for value in result.get("source_observation_ids", []) if value]
            if result["interpolation"] == "linear" and len(ids) == 2 and result["observation_id"] is None:
                before = source_rows.get(ids[0])
                after = source_rows.get(ids[1])
                if before is None or after is None:
                    result["quality_flags"] = sorted(
                        set(result["quality_flags"]) | {_flag(result["stream"], "source-row-missing")}
                    )
                    continue
                result["value"] = _interpolated_value(
                    before,
                    after,
                    int(result["timestamp_ns"]),
                    int(result["source_times_ns"][0]),
                    int(result["source_times_ns"][1]),
                )
                if result["value"] is None:
                    result["quality_flags"] = sorted(
                        set(result["quality_flags"]) | {_flag(result["stream"], "linear-value-missing")}
                    )
                continue
            if ids:
                source = source_rows.get(ids[0])
                if source is None:
                    result["quality_flags"] = sorted(
                        set(result["quality_flags"]) | {_flag(result["stream"], "source-row-missing")}
                    )
                    continue
                result["value"] = _sample_value(source)
        row_flags = {
            flag for result in row["streams"].values() for flag in result.get("quality_flags", [])
        }
        row["quality_flags"] = sorted(row_flags)
    return rows


def _source_rows_by_observation_id(
    lake: Lake,
    row_ids: Sequence[int],
    *,
    batch_size: int,
    metrics: _StreamingMetrics | None = None,
) -> dict[str, dict[str, Any]]:
    if not row_ids:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    table = lake.table("observations")
    for chunk in _chunks([int(row_id) for row_id in row_ids], batch_size):
        reader = table.take_row_ids(chunk).select(list(_SOURCE_VALUE_COLUMNS)).to_batches(
            max_batch_length=batch_size
        )
        for batch in reader:
            for row in batch.to_pylist():
                if metrics is not None:
                    metrics.source_rows_hydrated += 1
                observation_id = row.get("observation_id")
                if observation_id:
                    rows[observation_id] = row
    return rows


def _source_rows_read(rows: Sequence[dict[str, Any]]) -> int:
    return len(
        {
            int(row_id)
            for row in rows
            for result in row["streams"].values()
            for row_id in result.get("source_row_ids", [])
        }
    )


def _quality_summary(
    *,
    rows: Sequence[dict[str, Any]],
    stream_stats: dict[str, dict[str, Any]],
    sub_frame_bound_ns: int,
    row_metrics: _StreamingMetrics | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    flags: set[str] = set(row_metrics.quality_flags) if row_metrics else set()
    if row_metrics is None:
        for row in rows:
            flags.update(row.get("quality_flags") or [])
    stream_summary: dict[str, dict[str, Any]] = {}
    all_error_maxes = []
    for stream, stats in stream_stats.items():
        errors = [int(value) for value in stats.get("source_errors_ns") or []]
        error_count = int(stats.get("source_error_count") or len(errors))
        error_sum = int(stats.get("source_error_sum_ns") or sum(errors))
        max_abs_error_ns = stats.get("max_abs_error_ns")
        if max_abs_error_ns is None and errors:
            max_abs_error_ns = max(errors)
        if max_abs_error_ns is not None:
            all_error_maxes.append(int(max_abs_error_ns))
        stream_flags = []
        if stats["observations"] == 0:
            stream_flags.append(_flag(stream, "stream-missing"))
        if stats["tolerance_exceeded_count"]:
            stream_flags.append(_flag(stream, "tolerance-exceeded"))
        if stats["dropped_frame_count"]:
            stream_flags.append(_flag(stream, "dropped-frames"))
        if stats["clock_drift_detected"]:
            stream_flags.append(_flag(stream, "clock-drift"))
        flags.update(stream_flags)
        stream_summary[stream] = {
            "observations": stats["observations"],
            "aligned_count": stats["aligned_count"],
            "missing_count": stats["missing_count"],
            "tolerance_exceeded_count": stats["tolerance_exceeded_count"],
            "max_abs_error_ns": max_abs_error_ns,
            "mean_abs_error_ns": (error_sum / error_count) if error_count else None,
            "median_interval_ns": stats["median_interval_ns"],
            "dropped_frame_count": stats["dropped_frame_count"],
            "latency_drift_ns": stats["latency_drift_ns"],
            "clock_drift_detected": stats["clock_drift_detected"],
            "quality_flags": sorted(stream_flags),
        }
    row_confidences = [float(row["confidence"]) for row in rows] if row_metrics is None else []
    row_count = row_metrics.row_count if row_metrics else len(rows)
    confidence = (
        row_metrics.confidence_sum / row_metrics.row_count
        if row_metrics and row_metrics.row_count
        else statistics.fmean(row_confidences)
        if row_confidences
        else 0.0
    )
    min_row_confidence = (
        row_metrics.min_row_confidence
        if row_metrics and row_metrics.min_row_confidence is not None
        else min(row_confidences)
        if row_confidences
        else 0.0
    )
    summary = {
        "rows": row_count,
        "streams": stream_summary,
        "sub_frame_bound_ns": sub_frame_bound_ns,
        "max_abs_error_ns": max(all_error_maxes) if all_error_maxes else None,
        "confidence": confidence,
        "min_row_confidence": min_row_confidence,
        "quality_flags": sorted(flags),
    }
    return summary, tuple(sorted(flags))


def _record_alignment_job(
    lake: Lake,
    *,
    view: AlignmentView,
    recipe: dict[str, Any],
    input_versions: list[dict[str, Any]],
    created_by: str,
) -> None:
    now = datetime.now(UTC)
    jobs = lake.table("alignment_jobs")
    jobs.delete(f"alignment_id = '{view.alignment_id}'")
    jobs.add(
        pa.Table.from_pylist(
            [
                {
                    "alignment_id": view.alignment_id,
                    "name": view.name,
                    "input_tables": ["runs", "observations"],
                    "input_versions": input_versions,
                    "streams": list(view.streams),
                    "clock": view.clock,
                    "rate_hz": float(view.rate_hz),
                    "tolerance_ms": recipe["tolerance_ms"],
                    "recipe": json.dumps(recipe, sort_keys=True),
                    "output_table": view.output_table,
                    "quality_summary": json.dumps(view.quality_summary, sort_keys=True),
                    "quality_flags": list(view.quality_flags),
                    "transform_id": view.transform_id,
                    "created_at": now,
                }
            ],
            schema=ALIGNMENT_JOBS_SCHEMA,
        )
    )

    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{view.transform_id}'")
    transform_row = {
        "transform_id": view.transform_id,
        "kind": "alignment-view",
        "source_id": None,
        "input_uris": [],
        "input_table_versions": input_versions,
        "output_tables": _alignment_output_tables(view),
        "params": json.dumps(
            {
                "alignment_id": view.alignment_id,
                "name": view.name,
                "recipe": recipe,
                "quality_summary": view.quality_summary,
            },
            sort_keys=True,
        ),
        "status": "completed",
        "error": None,
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): the alignment-view slice without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)


def _alignment_output_tables(view: AlignmentView) -> list[str]:
    tables = ["alignment_jobs"]
    if view.output_table == "aligned_ticks":
        tables.extend(["aligned_ticks", "aligned_frames"])
    elif view.output_table == "aligned_frames":
        tables.append("aligned_frames")
    return tables


def _materialize_aligned_frames(
    lake: Lake,
    *,
    view: AlignmentView,
    transform_id: str,
    batch_size: int,
) -> None:
    table = lake.table("aligned_frames")
    table.delete(f"alignment_id = {_sql_literal(view.alignment_id)}")
    _append_aligned_frame_rows(
        table,
        alignment_id=view.alignment_id,
        transform_id=transform_id,
        rows=view.rows,
        batch_size=batch_size,
        created_at=datetime.now(UTC),
    )


def _append_aligned_tick_rows(
    table,
    *,
    alignment_id: str,
    alignment_name: str,
    recipe_digest: str,
    transform_id: str,
    rows: Sequence[dict[str, Any]],
    batch_size: int,
    created_at: datetime,
) -> int:
    materialized_rows = [
        _aligned_tick_row(
            alignment_id=alignment_id,
            alignment_name=alignment_name,
            recipe_digest=recipe_digest,
            transform_id=transform_id,
            row=row,
            created_at=created_at,
        )
        for row in rows
    ]
    if not materialized_rows:
        return 0
    for chunk in _chunks(materialized_rows, batch_size):
        table.add(pa.Table.from_pylist(chunk, schema=ALIGNED_TICKS_SCHEMA))
    return len(materialized_rows)


def _aligned_tick_row(
    *,
    alignment_id: str,
    alignment_name: str,
    recipe_digest: str,
    transform_id: str,
    row: Mapping[str, Any],
    created_at: datetime,
) -> dict[str, Any]:
    tick_index = int(row["index"])
    timestamp_ns = int(row["timestamp_ns"])
    stream_detail: dict[str, dict[str, Any]] = {}
    masks = {
        "valid": {},
        "missing": {},
        "interpolated": {},
        "out_of_tolerance": {},
    }
    stream_values: dict[str, Any] = {}
    lineage = {
        "aligned_frame_ids": {},
        "source_observation_ids": {},
        "source_row_ids": {},
    }
    available_streams: list[str] = []
    missing_streams: list[str] = []
    interpolated_streams: list[str] = []
    out_of_tolerance_streams: list[str] = []
    confidences: list[float] = []
    quality_flags = set(row.get("quality_flags") or [])
    run_id: str | None = None
    for stream, result in row["streams"].items():
        aligned_frame_id = "af-" + _digest(
            {
                "alignment_id": alignment_id,
                "tick_index": tick_index,
                "stream": stream,
            }
        )
        source_observation_ids = list(result.get("source_observation_ids") or [])
        source_row_ids = [int(row_id) for row_id in result.get("source_row_ids") or []]
        confidence = float(result.get("confidence") or 0.0)
        status = result["status"]
        interpolation = result.get("interpolation")
        value_json = _json_or_none(result.get("value"))
        missing = status == "missing" or not source_observation_ids
        interpolated = (
            status == "aligned"
            and interpolation == "linear"
            and len(source_observation_ids) > 1
        )
        out_of_tolerance = status == "out_of_tolerance"
        if missing:
            missing_streams.append(stream)
        else:
            available_streams.append(stream)
        if interpolated:
            interpolated_streams.append(stream)
        if out_of_tolerance:
            out_of_tolerance_streams.append(stream)
        if result.get("run_id") and run_id is None:
            run_id = str(result["run_id"])
        confidences.append(confidence)
        quality_flags.update(result.get("quality_flags") or [])
        stream_detail[stream] = {
            "stream": stream,
            "run_id": result.get("run_id"),
            "timestamp_ns": timestamp_ns,
            "status": status,
            "interpolation": interpolation,
            "observation_id": result.get("observation_id"),
            "source_observation_ids": source_observation_ids,
            "source_row_ids": source_row_ids,
            "source_timestamp_ns": result.get("source_timestamp_ns"),
            "source_time_ns": result.get("source_time_ns"),
            "receive_time_ns": result.get("receive_time_ns"),
            "latency_ns": result.get("latency_ns"),
            "error_ns": result.get("error_ns"),
            "absolute_error_ns": result.get("absolute_error_ns"),
            "confidence": confidence,
            "value_json": value_json,
            "quality_flags": list(result.get("quality_flags") or []),
            "aligned_frame_id": aligned_frame_id,
            "transform_id": transform_id,
        }
        masks["valid"][stream] = status == "aligned"
        masks["missing"][stream] = missing
        masks["interpolated"][stream] = interpolated
        masks["out_of_tolerance"][stream] = out_of_tolerance
        stream_values[stream] = result.get("value")
        lineage["aligned_frame_ids"][stream] = aligned_frame_id
        lineage["source_observation_ids"][stream] = source_observation_ids
        lineage["source_row_ids"][stream] = source_row_ids
    return {
        "aligned_tick_id": "at-" + _digest(
            {"alignment_id": alignment_id, "tick_index": tick_index}
        ),
        "alignment_id": alignment_id,
        "alignment_name": alignment_name,
        "recipe_digest": recipe_digest,
        "run_id": run_id,
        "tick_index": tick_index,
        "timestamp_ns": timestamp_ns,
        "available_streams": available_streams,
        "missing_streams": missing_streams,
        "interpolated_streams": interpolated_streams,
        "out_of_tolerance_streams": out_of_tolerance_streams,
        "has_missing": bool(missing_streams),
        "has_out_of_tolerance": bool(out_of_tolerance_streams),
        "min_confidence": min(confidences) if confidences else 0.0,
        "quality_flags": sorted(quality_flags),
        "stream_detail_json": _json_or_none(stream_detail),
        "masks_json": _json_or_none(masks),
        "stream_values_json": _json_or_none(stream_values),
        "lineage_json": _json_or_none(lineage),
        "transform_id": transform_id,
        "created_at": created_at,
    }


def _append_aligned_frame_rows(
    table,
    *,
    alignment_id: str,
    transform_id: str,
    rows: Sequence[dict[str, Any]],
    batch_size: int,
    created_at: datetime,
) -> int:
    materialized_rows = []
    for row in rows:
        tick_index = int(row["index"])
        timestamp_ns = int(row["timestamp_ns"])
        for stream, result in row["streams"].items():
            materialized_rows.append(
                {
                    "aligned_frame_id": "af-"
                    + _digest(
                        {
                            "alignment_id": alignment_id,
                            "tick_index": tick_index,
                            "stream": stream,
                        }
                    ),
                    "alignment_id": alignment_id,
                    "run_id": result.get("run_id"),
                    "tick_index": tick_index,
                    "timestamp_ns": timestamp_ns,
                    "stream": stream,
                    "status": result["status"],
                    "interpolation": result["interpolation"],
                    "observation_id": result.get("observation_id"),
                    "source_observation_ids": list(result.get("source_observation_ids") or []),
                    "source_row_ids": [int(row_id) for row_id in result.get("source_row_ids") or []],
                    "source_timestamp_ns": result.get("source_timestamp_ns"),
                    "source_time_ns": result.get("source_time_ns"),
                    "receive_time_ns": result.get("receive_time_ns"),
                    "latency_ns": result.get("latency_ns"),
                    "error_ns": result.get("error_ns"),
                    "absolute_error_ns": result.get("absolute_error_ns"),
                    "confidence": float(result.get("confidence") or 0.0),
                    "value_json": _json_or_none(result.get("value")),
                    "quality_flags": list(result.get("quality_flags") or []),
                    "transform_id": transform_id,
                    "created_at": created_at,
                }
            )
    if not materialized_rows:
        return 0
    for chunk in _chunks(materialized_rows, batch_size):
        table.add(pa.Table.from_pylist(chunk, schema=ALIGNED_FRAMES_SCHEMA))
    return len(materialized_rows)


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _recipe_digest(recipe: Mapping[str, Any]) -> str:
    return "recipe-" + _digest(recipe)


def _matches_stream(row: dict[str, Any], stream: str) -> bool:
    expected = _normalize_stream_key(stream)
    values = (
        row.get("topic"),
        row.get("sensor_id"),
        row.get("modality"),
        row.get("raw_channel"),
    )
    return any(_normalize_stream_key(value) == expected for value in values if value)


def _normalize_stream_key(value: Any) -> str:
    return str(value).strip().strip("/").lower().replace("_", "-")


def _observation_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("timestamp_ns") or 0),
        int(row.get("raw_sequence") or 0),
        row.get("topic") or "",
        row.get("sensor_id") or "",
        row.get("observation_id") or "",
    )


def _flag(stream: str, reason: str) -> str:
    return f"alignment:{_normalize_stream_key(stream)}:{reason}"


def _chunks(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _table_versions(lake: Lake, tables: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"table": table, "version": int(lake.table(table).version), "tag": ""}
        for table in tables
    ]


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]
