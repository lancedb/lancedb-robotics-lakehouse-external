"""First-class episode derivation, import, and frame/window access."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.schemas import (
    EPISODES_SCHEMA,
    TRANSFORM_RUNS_SCHEMA,
    VIDEOS_SCHEMA,
)

DEFAULT_START_EVENT_TYPES = ("episode_start", "teleop_start", "start")
DEFAULT_STOP_EVENT_TYPES = (
    "episode_end",
    "episode_stop",
    "teleop_stop",
    "success",
    "failure",
    "episode_success",
    "episode_failure",
)
DEFAULT_OUTCOME_BY_EVENT_TYPE = {
    "success": "success",
    "episode_success": "success",
    "failure": "failure",
    "episode_failure": "failure",
}

_OBSERVATION_EPISODE_FIELD_TYPES = {
    "episode_id": pa.string(),
    "episode_index": pa.int64(),
    "frame_index": pa.int64(),
    "robot_id": pa.string(),
    "site_id": pa.string(),
    "task_id": pa.string(),
    "software_version": pa.string(),
    "outcome": pa.string(),
}
_TIME_UNIT_MULTIPLIERS = {
    "ns": Decimal(1),
    "us": Decimal(1_000),
    "ms": Decimal(1_000_000),
    "s": Decimal(1_000_000_000),
}
_START_TIME_NS_FIELDS = ("from_timestamp_ns", "start_time_ns", "start_timestamp_ns")
_END_TIME_NS_FIELDS = ("to_timestamp_ns", "end_time_ns", "end_timestamp_ns")
_START_TIME_FIELDS = ("from_timestamp", "start_timestamp", "start_time", "start", "from")
_END_TIME_FIELDS = ("to_timestamp", "end_timestamp", "end_time", "end", "to")
_EPISODE_OBSERVATION_COLUMNS = (
    "observation_id",
    "run_id",
    "timestamp_ns",
    "sensor_id",
    "topic",
    "modality",
    "raw_uri",
    "raw_sequence",
    "message_encoding",
    "schema_encoding",
)
_SOURCE_TABLES = {"events", "labels", "model_outputs"}
_DERIVATION_KIND = "episode-derivation"
_LIFECYCLE_KIND = "episode-derivation-lifecycle"
_OVERLAP_POLICIES = {"error", "replace", "supersede", "preserve"}


class EpisodeError(Exception):
    """Raised when episodes cannot be derived or resolved."""


@dataclass(frozen=True)
class EpisodeBuildReport:
    """Summary of one episode derivation transform."""

    lake_uri: str
    transform_id: str
    boundary_source: str
    episodes_written: int
    videos_written: int
    frames_tagged: int
    episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class EpisodePredicateInterval:
    """One planned interval from predicate mining."""

    run_id: str
    from_timestamp_ns: int
    to_timestamp_ns: int
    frame_count: int
    observation_ids: tuple[str, ...]
    source_observation_ids: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    source_label_ids: tuple[str, ...] = ()
    source_model_output_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "from_timestamp_ns": self.from_timestamp_ns,
            "to_timestamp_ns": self.to_timestamp_ns,
            "frame_count": self.frame_count,
            "observation_ids": list(self.observation_ids),
        }
        if self.source_observation_ids:
            payload["source_observation_ids"] = list(self.source_observation_ids)
        if self.source_event_ids:
            payload["source_event_ids"] = list(self.source_event_ids)
        if self.source_label_ids:
            payload["source_label_ids"] = list(self.source_label_ids)
        if self.source_model_output_ids:
            payload["source_model_output_ids"] = list(self.source_model_output_ids)
        return payload


@dataclass(frozen=True)
class EpisodeOverlapConflict:
    """One planned frame annotation that conflicts with an existing owner."""

    observation_id: str
    existing_episode_id: str
    existing_transform_id: str
    planned_episode_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "existing_episode_id": self.existing_episode_id,
            "existing_transform_id": self.existing_transform_id,
            "planned_episode_id": self.planned_episode_id,
        }


@dataclass(frozen=True)
class EpisodeMiningReport:
    """Report for predicate mining, including dry-run planned intervals."""

    lake_uri: str
    transform_id: str
    boundary_source: str
    dry_run: bool
    intervals_planned: int
    frames_planned: int
    episodes_written: int
    videos_written: int
    frames_tagged: int
    episode_ids: tuple[str, ...]
    intervals: tuple[EpisodePredicateInterval, ...]
    overlap_policy: str = "error"
    overlap_conflicts: tuple[EpisodeOverlapConflict, ...] = ()


@dataclass(frozen=True)
class EpisodeDerivationDryRun:
    """Dry-run report for a planned episode derivation."""

    lake_uri: str
    transform_id: str
    boundary_source: str
    overlap_policy: str
    episodes_planned: int
    videos_planned: int
    frames_planned: int
    episode_ids: tuple[str, ...]
    overlap_conflicts: tuple[EpisodeOverlapConflict, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "transform_id": self.transform_id,
            "boundary_source": self.boundary_source,
            "overlap_policy": self.overlap_policy,
            "episodes_planned": self.episodes_planned,
            "videos_planned": self.videos_planned,
            "frames_planned": self.frames_planned,
            "episode_ids": list(self.episode_ids),
            "overlap_conflicts": [conflict.to_dict() for conflict in self.overlap_conflicts],
        }


@dataclass(frozen=True)
class EpisodeDerivationSummary:
    """Registry row for one episode derivation transform."""

    transform_id: str
    kind: str
    params_hash: str
    episode_ids: tuple[str, ...]
    frame_count: int
    status: str
    superseded_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "kind": self.kind,
            "params_hash": self.params_hash,
            "episode_ids": list(self.episode_ids),
            "frame_count": self.frame_count,
            "status": self.status,
            "superseded_by": self.superseded_by,
        }


@dataclass(frozen=True)
class EpisodeDerivationDetail:
    """Detailed registry view for one derivation."""

    summary: EpisodeDerivationSummary
    transform: dict[str, Any]
    params: dict[str, Any]
    episodes: tuple[dict[str, Any], ...]
    videos: tuple[dict[str, Any], ...]
    lifecycle_actions: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "transform": self.transform,
            "params": self.params,
            "episodes": list(self.episodes),
            "videos": list(self.videos),
            "lifecycle_actions": list(self.lifecycle_actions),
        }


@dataclass(frozen=True)
class EpisodeLifecycleReport:
    """Summary of a lifecycle action over an episode derivation."""

    lake_uri: str
    action: str
    target_transform_id: str
    lifecycle_transform_id: str
    replacement_transform_id: str | None = None
    episodes_removed: int = 0
    videos_removed: int = 0
    frames_untagged: int = 0
    episode_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "action": self.action,
            "target_transform_id": self.target_transform_id,
            "lifecycle_transform_id": self.lifecycle_transform_id,
            "replacement_transform_id": self.replacement_transform_id,
            "episodes_removed": self.episodes_removed,
            "videos_removed": self.videos_removed,
            "frames_untagged": self.frames_untagged,
            "episode_ids": list(self.episode_ids),
        }


@dataclass(frozen=True)
class IntervalManifest:
    """Parsed interval manifest and source identity for replayable imports."""

    records: tuple[dict[str, Any], ...]
    format: str
    source_uri: str
    sha256: str


@dataclass(frozen=True)
class EpisodeWindow:
    """Deterministic aligned view returned by :meth:`Episode.window`."""

    episode_id: str
    rate_hz: float
    streams: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _EpisodeBound:
    run_id: str
    start_time_ns: int
    end_time_ns: int
    boundary_source: str
    outcome: str | None
    source_event_ids: tuple[str, ...] = ()
    source_scenario_ids: tuple[str, ...] = ()
    source_label_ids: tuple[str, ...] = ()
    source_model_output_ids: tuple[str, ...] = ()
    source_observation_ids: tuple[str, ...] | None = None
    observation_ids: tuple[str, ...] | None = None
    source_query: dict[str, Any] | None = None
    task_id: str | None = None
    external_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None
    source_provenance: dict[str, Any] | None = None
    source_payload: dict[str, Any] | None = None
    clipped: dict[str, dict[str, int]] | None = None


@dataclass(frozen=True)
class _ValidatedInterval:
    run_id: str
    start_time_ns: int
    end_time_ns: int
    original_start_time_ns: int
    original_end_time_ns: int
    external_id: str | None
    outcome: str | None
    task_id: str | None
    tags: tuple[str, ...]
    metadata: dict[str, Any]
    source_provenance: dict[str, Any]
    source_payload: dict[str, Any]
    clipped: dict[str, dict[str, int]]


@dataclass(frozen=True)
class _PredicateHit:
    run_id: str
    start_time_ns: int
    end_time_ns: int
    observation_ids: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()
    label_ids: tuple[str, ...] = ()
    model_output_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FrameOwner:
    episode_id: str
    transform_id: str


class Episode:
    """A first-class episode row with frame and window helpers."""

    def __init__(self, lake: Lake, row: dict[str, Any]) -> None:
        self._lake = lake
        self.row = row

    @property
    def episode_id(self) -> str:
        return self.row["episode_id"]

    @property
    def episode_index(self) -> int:
        return int(self.row["episode_index"])

    def frames(self, streams: Sequence[str] | None = None) -> tuple[dict[str, Any], ...]:
        """Return this episode's observations in stable ``frame_index`` order."""
        rows = [
            row
            for row in self._lake.table("observations").to_arrow().to_pylist()
            if row.get("episode_id") == self.episode_id
        ]
        if streams:
            selected = tuple(streams)
            rows = [row for row in rows if any(_matches_stream(row, stream) for stream in selected)]
        return tuple(sorted(rows, key=_frame_sort_key))

    def window(self, rate_hz: float, streams: Sequence[str]) -> EpisodeWindow:
        """Return a deterministic multi-stream slice over this episode.

        The 0031 alignment engine owns common-clock resampling. Episode windows
        are read-only views, so this delegates to ``lake.align.window`` without
        writing an ``alignment_jobs`` lineage row for each read.
        """
        if rate_hz <= 0:
            raise EpisodeError("rate_hz must be positive")
        selected_streams = tuple(streams)
        if not selected_streams:
            raise EpisodeError("at least one stream is required")

        view = self._lake.align.window(
            name=f"episode-{self.episode_id}",
            rate_hz=rate_hz,
            streams=selected_streams,
            run_id=self.row["run_id"],
            start_time_ns=int(self.row["from_timestamp_ns"]),
            end_time_ns=int(self.row["to_timestamp_ns"]),
        )
        return EpisodeWindow(
            episode_id=self.episode_id,
            rate_hz=float(rate_hz),
            streams=selected_streams,
            rows=view.rows,
        )


class LakeEpisodes:
    """Convenience namespace exposed as ``lake.episodes``."""

    def __init__(self, lake: Lake) -> None:
        self._lake = lake

    def from_markers(
        self,
        *,
        start_event_types: Sequence[str] | None = None,
        stop_event_types: Sequence[str] | None = None,
        outcome_by_event_type: Mapping[str, str] | None = None,
        overlap_policy: str = "error",
        created_by: str = "lancedb-robotics",
    ) -> EpisodeBuildReport:
        return build_episodes_from_markers(
            self._lake,
            start_event_types=start_event_types,
            stop_event_types=stop_event_types,
            outcome_by_event_type=outcome_by_event_type,
            overlap_policy=overlap_policy,
            created_by=created_by,
        )

    def from_query(
        self,
        *,
        event_type: str,
        before_ns: int,
        after_ns: int,
        overlap_policy: str = "error",
        created_by: str = "lancedb-robotics",
    ) -> EpisodeBuildReport:
        return build_episodes_from_query(
            self._lake,
            event_type=event_type,
            before_ns=before_ns,
            after_ns=after_ns,
            overlap_policy=overlap_policy,
            created_by=created_by,
        )

    def from_predicate(
        self,
        *,
        where: str | None = None,
        source_table: str | None = None,
        source_where: str | None = None,
        before_ns: int = 0,
        after_ns: int = 0,
        merge_gap_ns: int | None = None,
        min_duration_ns: int | None = None,
        max_duration_ns: int | None = None,
        topics: Sequence[str] | None = None,
        modalities: Sequence[str] | None = None,
        dry_run: bool = False,
        overlap_policy: str = "error",
        created_by: str = "lancedb-robotics",
    ) -> EpisodeMiningReport:
        return build_episodes_from_predicate(
            self._lake,
            where=where,
            source_table=source_table,
            source_where=source_where,
            before_ns=before_ns,
            after_ns=after_ns,
            merge_gap_ns=merge_gap_ns,
            min_duration_ns=min_duration_ns,
            max_duration_ns=max_duration_ns,
            topics=topics,
            modalities=modalities,
            dry_run=dry_run,
            overlap_policy=overlap_policy,
            created_by=created_by,
        )

    def from_scenarios(
        self,
        *,
        scenario_ids: Sequence[str] | None = None,
        snapshot_name: str | None = None,
        outcome: str | None = None,
        outcome_by_coverage_tag: Mapping[str, str] | None = None,
        outcome_by_summary: Mapping[str, str] | None = None,
        task_id: str | None = None,
        task_id_by_coverage_tag: Mapping[str, str] | None = None,
        task_id_by_summary: Mapping[str, str] | None = None,
        overlap_policy: str = "error",
        created_by: str = "lancedb-robotics",
    ) -> EpisodeBuildReport:
        return build_episodes_from_scenarios(
            self._lake,
            scenario_ids=scenario_ids,
            snapshot_name=snapshot_name,
            outcome=outcome,
            outcome_by_coverage_tag=outcome_by_coverage_tag,
            outcome_by_summary=outcome_by_summary,
            task_id=task_id,
            task_id_by_coverage_tag=task_id_by_coverage_tag,
            task_id_by_summary=task_id_by_summary,
            overlap_policy=overlap_policy,
            created_by=created_by,
        )

    def from_intervals(
        self,
        intervals: Sequence[Mapping[str, Any]],
        *,
        time_unit: str = "ns",
        allow_clipped: bool = False,
        allow_empty: bool = False,
        source_uri: str | None = None,
        source_sha256: str | None = None,
        overlap_policy: str = "error",
        created_by: str = "lancedb-robotics",
    ) -> EpisodeBuildReport:
        return build_episodes_from_intervals(
            self._lake,
            intervals,
            time_unit=time_unit,
            allow_clipped=allow_clipped,
            allow_empty=allow_empty,
            source_uri=source_uri,
            source_sha256=source_sha256,
            overlap_policy=overlap_policy,
            created_by=created_by,
        )

    def get(self, episode_id: str) -> Episode:
        rows = [
            row
            for row in self._lake.table("episodes").to_arrow().to_pylist()
            if row["episode_id"] == episode_id
        ]
        if not rows:
            raise EpisodeError(f"no episode {episode_id!r} in {self._lake.uri}")
        return Episode(self._lake, rows[0])

    def list_derivations(self) -> tuple[EpisodeDerivationSummary, ...]:
        """Return the computed episode-derivation registry."""
        return list_episode_derivations(self._lake)

    def show_derivation(self, transform_id: str) -> EpisodeDerivationDetail:
        """Return detailed recipe, output, and lifecycle state for a derivation."""
        return show_episode_derivation(self._lake, transform_id)

    def dry_run(
        self,
        recipe: Mapping[str, Any] | str,
        *,
        overlap_policy: str = "error",
    ) -> EpisodeDerivationDryRun | EpisodeMiningReport:
        """Plan a recorded or ad hoc derivation recipe without writing tables."""
        return dry_run_episode_derivation(
            self._lake,
            recipe,
            overlap_policy=overlap_policy,
        )

    def rebuild(
        self,
        transform_id: str,
        *,
        overlap_policy: str = "replace",
        created_by: str = "lancedb-robotics",
    ) -> EpisodeBuildReport | EpisodeMiningReport:
        """Replay a recorded derivation recipe."""
        return rebuild_episode_derivation(
            self._lake,
            transform_id,
            overlap_policy=overlap_policy,
            created_by=created_by,
        )

    def supersede(
        self,
        old_transform_id: str,
        recipe: Mapping[str, Any] | str,
        *,
        created_by: str = "lancedb-robotics",
    ) -> EpisodeBuildReport | EpisodeMiningReport:
        """Replace an older active derivation with a new recorded recipe."""
        return supersede_episode_derivation(
            self._lake,
            old_transform_id,
            recipe,
            created_by=created_by,
        )

    def clear(
        self,
        transform_id: str,
        *,
        created_by: str = "lancedb-robotics",
    ) -> EpisodeLifecycleReport:
        """Remove a derivation's rows and safely unset its current frame owners."""
        return clear_episode_derivation(
            self._lake,
            transform_id,
            created_by=created_by,
        )


def list_episode_derivations(lake: Lake) -> tuple[EpisodeDerivationSummary, ...]:
    """Return the computed episode-derivation registry."""
    transform_rows = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row.get("kind") == _DERIVATION_KIND
    ]
    episodes_by_transform: dict[str, list[dict[str, Any]]] = {}
    for row in lake.table("episodes").to_arrow().to_pylist():
        episodes_by_transform.setdefault(row["transform_id"], []).append(row)

    statuses = _derivation_statuses(lake)
    summaries: list[EpisodeDerivationSummary] = []
    for row in sorted(transform_rows, key=lambda item: (item.get("created_at"), item["transform_id"])):
        params = _transform_params(row)
        current_episodes = episodes_by_transform.get(row["transform_id"], [])
        episode_ids = tuple(
            sorted(
                {
                    *(str(value) for value in params.get("episode_ids") or []),
                    *(episode["episode_id"] for episode in current_episodes),
                }
            )
        )
        status, superseded_by = statuses.get(row["transform_id"], ("active", None))
        summaries.append(
            EpisodeDerivationSummary(
                transform_id=row["transform_id"],
                kind=_recipe_boundary_source(params),
                params_hash=_recipe_params_hash(params),
                episode_ids=episode_ids,
                frame_count=sum(int(episode.get("frame_count") or 0) for episode in current_episodes),
                status=status,
                superseded_by=superseded_by,
            )
        )
    return tuple(summaries)


def show_episode_derivation(lake: Lake, transform_id: str) -> EpisodeDerivationDetail:
    """Return detailed recipe, output, and lifecycle state for one derivation."""
    transform = _derivation_transform_row(lake, transform_id)
    summaries = {row.transform_id: row for row in list_episode_derivations(lake)}
    summary = summaries.get(transform_id)
    if summary is None:  # pragma: no cover - guarded by _derivation_transform_row
        raise EpisodeError(f"no episode derivation transform {transform_id!r}")
    episodes = tuple(
        sorted(
            [
                row
                for row in lake.table("episodes").to_arrow().to_pylist()
                if row["transform_id"] == transform_id
            ],
            key=lambda row: (row["episode_index"], row["episode_id"]),
        )
    )
    videos = tuple(
        sorted(
            [
                row
                for row in lake.table("videos").to_arrow().to_pylist()
                if row["transform_id"] == transform_id
            ],
            key=lambda row: (row["episode_index"], row["camera_key"], row["video_id"]),
        )
    )
    actions = tuple(
        row
        for row in _lifecycle_action_rows(lake)
        if _lifecycle_params(row).get("target_transform_id") == transform_id
        or _lifecycle_params(row).get("replacement_transform_id") == transform_id
    )
    return EpisodeDerivationDetail(
        summary=summary,
        transform=transform,
        params=_transform_params(transform),
        episodes=episodes,
        videos=videos,
        lifecycle_actions=actions,
    )


def dry_run_episode_derivation(
    lake: Lake,
    recipe: Mapping[str, Any] | str,
    *,
    overlap_policy: str = "error",
) -> EpisodeDerivationDryRun | EpisodeMiningReport:
    """Plan an episode derivation recipe without writing tables."""
    return _run_episode_recipe(
        lake,
        _resolve_episode_recipe(lake, recipe),
        overlap_policy=overlap_policy,
        dry_run=True,
    )


def rebuild_episode_derivation(
    lake: Lake,
    transform_id: str,
    *,
    overlap_policy: str = "replace",
    created_by: str = "lancedb-robotics",
) -> EpisodeBuildReport | EpisodeMiningReport:
    """Replay a recorded episode derivation recipe."""
    row = _derivation_transform_row(lake, transform_id)
    report = _run_episode_recipe(
        lake,
        _transform_params(row),
        overlap_policy=overlap_policy,
        dry_run=False,
        created_by=created_by,
    )
    if isinstance(report, EpisodeDerivationDryRun):  # pragma: no cover - dry_run=False guard
        raise EpisodeError("rebuild unexpectedly returned a dry-run report")
    if report.transform_id != transform_id:
        raise EpisodeError(
            f"recorded recipe rebuilt to {report.transform_id!r}, expected {transform_id!r}"
        )
    _record_lifecycle_action(
        lake,
        action="rebuild",
        target_transform_id=transform_id,
        replacement_transform_id=transform_id,
        episode_ids=report.episode_ids,
        frames_affected=report.frames_tagged,
        created_by=created_by,
    )
    return report


def supersede_episode_derivation(
    lake: Lake,
    old_transform_id: str,
    recipe: Mapping[str, Any] | str,
    *,
    created_by: str = "lancedb-robotics",
) -> EpisodeBuildReport | EpisodeMiningReport:
    """Supersede an older derivation with a new recipe."""
    _derivation_transform_row(lake, old_transform_id)
    report = _run_episode_recipe(
        lake,
        _resolve_episode_recipe(lake, recipe),
        overlap_policy="supersede",
        dry_run=False,
        created_by=created_by,
        supersede_transform_ids=(old_transform_id,),
    )
    if isinstance(report, EpisodeDerivationDryRun):  # pragma: no cover - dry_run=False guard
        raise EpisodeError("supersede unexpectedly returned a dry-run report")
    if report.transform_id == old_transform_id:
        raise EpisodeError("supersede requires a replacement recipe, not the same transform")
    return report


def clear_episode_derivation(
    lake: Lake,
    transform_id: str,
    *,
    created_by: str = "lancedb-robotics",
) -> EpisodeLifecycleReport:
    """Clear a derivation's output rows and safe current frame annotations."""
    _derivation_transform_row(lake, transform_id)
    target_episodes = [
        row
        for row in lake.table("episodes").to_arrow().to_pylist()
        if row["transform_id"] == transform_id
    ]
    target_videos = [
        row
        for row in lake.table("videos").to_arrow().to_pylist()
        if row["transform_id"] == transform_id
    ]
    episode_ids = tuple(sorted(row["episode_id"] for row in target_episodes))
    clear_updates = _clear_updates_for_episode_ids(lake, episode_ids)

    lake.table("episodes").delete(f"transform_id = '{transform_id}'")
    lake.table("videos").delete(f"transform_id = '{transform_id}'")
    _apply_observation_episode_columns(lake, clear_updates)
    lifecycle_id = _record_lifecycle_action(
        lake,
        action="clear",
        target_transform_id=transform_id,
        episode_ids=episode_ids,
        frames_affected=len(clear_updates),
        created_by=created_by,
    )
    return EpisodeLifecycleReport(
        lake_uri=lake.uri,
        action="clear",
        target_transform_id=transform_id,
        lifecycle_transform_id=lifecycle_id,
        episodes_removed=len(target_episodes),
        videos_removed=len(target_videos),
        frames_untagged=len(clear_updates),
        episode_ids=episode_ids,
    )


def load_interval_manifest(
    path: str | Path,
    *,
    format: str | None = None,
) -> IntervalManifest:
    """Load a JSONL or CSV interval manifest and return raw records plus source hash."""
    source = Path(path)
    data = source.read_bytes()
    manifest_format = _interval_manifest_format(source, format)
    if manifest_format == "jsonl":
        records = load_intervals_jsonl(source)
    elif manifest_format == "csv":
        records = load_intervals_csv(source)
    else:  # pragma: no cover - guarded by _interval_manifest_format
        raise EpisodeError(f"unsupported interval manifest format {manifest_format!r}")
    return IntervalManifest(
        records=records,
        format=manifest_format,
        source_uri=str(source),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def load_intervals_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load JSON Lines interval records from ``path``."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EpisodeError(f"invalid JSONL interval at line {line_number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise EpisodeError(f"JSONL interval line {line_number} must be an object")
            records.append(dict(value))
    return tuple(records)


def load_intervals_csv(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load CSV interval records from ``path``."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise EpisodeError("CSV interval manifest must include a header row")
        for row_number, row in enumerate(reader, start=2):
            cleaned = {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key is not None and str(key).strip()
            }
            if any(value not in (None, "") for value in cleaned.values()):
                records.append(cleaned)
            elif row:
                raise EpisodeError(f"CSV interval row {row_number} is empty")
    return tuple(records)


def _run_episode_recipe(
    lake: Lake,
    recipe: Mapping[str, Any],
    *,
    overlap_policy: str,
    dry_run: bool,
    created_by: str = "lancedb-robotics",
    supersede_transform_ids: Sequence[str] = (),
) -> EpisodeBuildReport | EpisodeDerivationDryRun | EpisodeMiningReport:
    kind = _recipe_boundary_source(recipe)
    if kind == "markers":
        return build_episodes_from_markers(
            lake,
            start_event_types=_optional_sequence(recipe.get("start_event_types")),
            stop_event_types=_optional_sequence(recipe.get("stop_event_types")),
            outcome_by_event_type=_optional_mapping(recipe.get("outcome_by_event_type")),
            overlap_policy=overlap_policy,
            dry_run=dry_run,
            created_by=created_by,
            supersede_transform_ids=supersede_transform_ids,
        )
    if kind == "query":
        return build_episodes_from_query(
            lake,
            event_type=str(recipe.get("event_type") or ""),
            before_ns=int(recipe.get("before_ns") or 0),
            after_ns=int(recipe.get("after_ns") or 0),
            overlap_policy=overlap_policy,
            dry_run=dry_run,
            created_by=created_by,
            supersede_transform_ids=supersede_transform_ids,
        )
    if kind == "predicate":
        return build_episodes_from_predicate(
            lake,
            where=_optional_str(recipe.get("where")),
            source_table=_optional_str(recipe.get("source_table")),
            source_where=_optional_str(recipe.get("source_where")),
            before_ns=int(recipe.get("before_ns") or 0),
            after_ns=int(recipe.get("after_ns") or 0),
            merge_gap_ns=_optional_int(recipe.get("merge_gap_ns")),
            min_duration_ns=_optional_int(recipe.get("min_duration_ns")),
            max_duration_ns=_optional_int(recipe.get("max_duration_ns")),
            topics=_optional_sequence(recipe.get("topics")),
            modalities=_optional_sequence(recipe.get("modalities")),
            dry_run=dry_run,
            overlap_policy=overlap_policy,
            created_by=created_by,
            supersede_transform_ids=supersede_transform_ids,
        )
    if kind == "scenarios":
        snapshot_name = _optional_str(recipe.get("snapshot_name"))
        return build_episodes_from_scenarios(
            lake,
            scenario_ids=None if snapshot_name else _optional_sequence(recipe.get("scenario_ids")),
            snapshot_name=snapshot_name,
            outcome=_optional_str(recipe.get("outcome")),
            outcome_by_coverage_tag=_optional_mapping(recipe.get("outcome_by_coverage_tag")),
            outcome_by_summary=_optional_mapping(recipe.get("outcome_by_summary")),
            task_id=_optional_str(recipe.get("task_id")),
            task_id_by_coverage_tag=_optional_mapping(recipe.get("task_id_by_coverage_tag")),
            task_id_by_summary=_optional_mapping(recipe.get("task_id_by_summary")),
            overlap_policy=overlap_policy,
            dry_run=dry_run,
            created_by=created_by,
            supersede_transform_ids=supersede_transform_ids,
        )
    if kind == "intervals":
        intervals = _recipe_interval_records(recipe)
        return build_episodes_from_intervals(
            lake,
            intervals,
            time_unit=str(recipe.get("time_unit") or "ns"),
            allow_clipped=bool(recipe.get("allow_clipped")),
            allow_empty=bool(recipe.get("allow_empty")),
            source_uri=_optional_str(recipe.get("source_uri")),
            source_sha256=_optional_str(recipe.get("source_sha256")),
            overlap_policy=overlap_policy,
            dry_run=dry_run,
            created_by=created_by,
            supersede_transform_ids=supersede_transform_ids,
        )
    raise EpisodeError(f"unsupported episode derivation kind {kind!r}")


def _resolve_episode_recipe(lake: Lake, recipe: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(recipe, Mapping):
        return dict(recipe)
    value = str(recipe).strip()
    if value.startswith("tfm-"):
        return _transform_params(_derivation_transform_row(lake, value))
    return {"kind": value}


def _derivation_transform_row(lake: Lake, transform_id: str) -> dict[str, Any]:
    rows = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row.get("kind") == _DERIVATION_KIND and row.get("transform_id") == transform_id
    ]
    if not rows:
        raise EpisodeError(f"no episode derivation transform {transform_id!r}")
    return max(rows, key=lambda row: (row.get("created_at"), row["transform_id"]))


def _transform_params(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("params") or "{}")
    except json.JSONDecodeError as exc:
        raise EpisodeError(f"transform {row.get('transform_id')!r} params are not valid JSON") from exc
    if not isinstance(value, dict):
        raise EpisodeError(f"transform {row.get('transform_id')!r} params must be a JSON object")
    return value


def _recipe_boundary_source(params: Mapping[str, Any]) -> str:
    raw = str(params.get("boundary_source") or params.get("kind") or "").strip().lower()
    raw = raw.removeprefix("episode-")
    aliases = {
        "marker": "markers",
        "markers": "markers",
        "query": "query",
        "scenario": "scenarios",
        "scenarios": "scenarios",
        "interval": "intervals",
        "intervals": "intervals",
        "predicate": "predicate",
    }
    kind = aliases.get(raw)
    if kind is None:
        raise EpisodeError(f"episode derivation recipe is missing a supported kind: {raw!r}")
    return kind


def _recipe_params_hash(params: Mapping[str, Any]) -> str:
    ignored = {"boundary_source", "episode_ids", "overlap_policy", "supersedes_transform_ids"}
    return _digest({key: value for key, value in params.items() if key not in ignored})


def _recipe_interval_records(recipe: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    records = recipe.get("records")
    if records is None:
        records = recipe.get("intervals")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise EpisodeError("interval recipe requires interval records")
    normalized = []
    for index, item in enumerate(records, start=1):
        if not isinstance(item, Mapping):
            raise EpisodeError(f"interval recipe item {index} must be an object")
        payload = item.get("source_payload") if isinstance(item.get("source_payload"), Mapping) else item
        normalized.append(dict(payload))
    return tuple(normalized)


def _optional_sequence(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise EpisodeError("recipe value must be a sequence")
    return tuple(str(item) for item in value if str(item))


def _optional_mapping(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise EpisodeError("recipe value must be an object")
    return {str(key): str(item) for key, item in value.items()}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def build_episodes_from_markers(
    lake: Lake,
    *,
    start_event_types: Sequence[str] | None = None,
    stop_event_types: Sequence[str] | None = None,
    outcome_by_event_type: Mapping[str, str] | None = None,
    overlap_policy: str = "error",
    dry_run: bool = False,
    created_by: str = "lancedb-robotics",
    supersede_transform_ids: Sequence[str] = (),
) -> EpisodeBuildReport | EpisodeDerivationDryRun:
    """Build episodes from explicit start/stop marker events."""
    starts = _normalize_event_types(start_event_types or DEFAULT_START_EVENT_TYPES)
    outcomes = dict(DEFAULT_OUTCOME_BY_EVENT_TYPE)
    if outcome_by_event_type:
        outcomes.update({str(key): str(value) for key, value in outcome_by_event_type.items()})
    stops = _normalize_event_types(stop_event_types or DEFAULT_STOP_EVENT_TYPES)
    if not starts:
        raise EpisodeError("at least one start event type is required")
    if not stops:
        raise EpisodeError("at least one stop event type is required")

    params = {
        "kind": "episode-markers",
        "start_event_types": starts,
        "stop_event_types": stops,
        "outcome_by_event_type": dict(sorted(outcomes.items())),
    }
    transform_id = f"tfm-episodes-markers-{_digest(params)}"
    run_rows = _rows_by_id(lake, "runs", "run_id")
    events_by_run = _events_by_run(lake)
    bounds: list[_EpisodeBound] = []

    for run_id in sorted(events_by_run):
        if run_id not in run_rows:
            continue
        open_marker: dict[str, Any] | None = None
        for event in events_by_run[run_id]:
            event_type = event.get("event_type")
            if event_type in starts:
                open_marker = event
                continue
            if event_type not in stops or open_marker is None:
                continue
            start_ns = _event_start_ns(open_marker)
            end_ns = _event_end_ns(event)
            if end_ns < start_ns:
                raise EpisodeError(
                    f"marker end event {event['event_id']!r} is earlier than its start"
                )
            bounds.append(
                _EpisodeBound(
                    run_id=run_id,
                    start_time_ns=start_ns,
                    end_time_ns=end_ns,
                    boundary_source="markers",
                    outcome=_marker_outcome(event, outcomes),
                    source_event_ids=(open_marker["event_id"], event["event_id"]),
                    source_query=params,
                )
            )
            open_marker = None

    if dry_run:
        return _dry_run_episodes(
            lake,
            bounds,
            transform_id=transform_id,
            boundary_source="markers",
            params=params,
            overlap_policy=overlap_policy,
        )

    return _materialize_episodes(
        lake,
        bounds,
        transform_id=transform_id,
        boundary_source="markers",
        params=params,
        overlap_policy=overlap_policy,
        created_by=created_by,
        supersede_transform_ids=supersede_transform_ids,
    )


def build_episodes_from_query(
    lake: Lake,
    *,
    event_type: str,
    before_ns: int,
    after_ns: int,
    overlap_policy: str = "error",
    dry_run: bool = False,
    created_by: str = "lancedb-robotics",
    supersede_transform_ids: Sequence[str] = (),
) -> EpisodeBuildReport | EpisodeDerivationDryRun:
    """Mine episodes around events matching ``event_type``."""
    if not event_type:
        raise EpisodeError("event_type is required")
    if before_ns < 0 or after_ns < 0:
        raise EpisodeError("before_ns and after_ns must be non-negative")

    params = {
        "kind": "episode-query",
        "event_type": event_type,
        "before_ns": int(before_ns),
        "after_ns": int(after_ns),
    }
    transform_id = f"tfm-episodes-query-{_digest(params)}"
    runs = _rows_by_id(lake, "runs", "run_id")
    bounds: list[_EpisodeBound] = []

    for event in sorted(
        lake.table("events").to_arrow().to_pylist(),
        key=lambda row: (row.get("run_id") or "", _event_start_ns(row), row["event_id"]),
    ):
        if event.get("event_type") != event_type:
            continue
        run = runs.get(event["run_id"])
        if run is None:
            continue
        timestamp_ns = int(event.get("timestamp_ns") or event.get("start_time_ns") or 0)
        start_ns = max(int(run["start_time_ns"]), timestamp_ns - int(before_ns))
        end_ns = min(int(run["end_time_ns"]), timestamp_ns + int(after_ns))
        if end_ns < start_ns:
            continue
        bounds.append(
            _EpisodeBound(
                run_id=event["run_id"],
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                boundary_source="query",
                outcome=event.get("severity") or event_type,
                source_event_ids=(event["event_id"],),
                source_query=params,
            )
        )

    if dry_run:
        return _dry_run_episodes(
            lake,
            bounds,
            transform_id=transform_id,
            boundary_source="query",
            params=params,
            overlap_policy=overlap_policy,
        )

    return _materialize_episodes(
        lake,
        bounds,
        transform_id=transform_id,
        boundary_source="query",
        params=params,
        overlap_policy=overlap_policy,
        created_by=created_by,
        supersede_transform_ids=supersede_transform_ids,
    )


def build_episodes_from_predicate(
    lake: Lake,
    *,
    where: str | None = None,
    source_table: str | None = None,
    source_where: str | None = None,
    before_ns: int = 0,
    after_ns: int = 0,
    merge_gap_ns: int | None = None,
    min_duration_ns: int | None = None,
    max_duration_ns: int | None = None,
    topics: Sequence[str] | None = None,
    modalities: Sequence[str] | None = None,
    dry_run: bool = False,
    overlap_policy: str = "error",
    created_by: str = "lancedb-robotics",
    supersede_transform_ids: Sequence[str] = (),
) -> EpisodeMiningReport:
    """Mine physical episodes from observation or auxiliary-table predicates."""
    normalized_where = _normalize_predicate(where)
    normalized_source_table = _normalize_source_table(source_table)
    normalized_source_where = _normalize_predicate(source_where)
    if normalized_source_table is None and normalized_where is None:
        raise EpisodeError("where is required when source_table is not provided")
    if normalized_source_table is not None and normalized_source_where is None:
        raise EpisodeError("source_where is required when source_table is provided")
    if before_ns < 0 or after_ns < 0:
        raise EpisodeError("before_ns and after_ns must be non-negative")
    if merge_gap_ns is not None and merge_gap_ns < 0:
        raise EpisodeError("merge_gap_ns must be non-negative")
    if min_duration_ns is not None and min_duration_ns < 0:
        raise EpisodeError("min_duration_ns must be non-negative")
    if max_duration_ns is not None and max_duration_ns < 0:
        raise EpisodeError("max_duration_ns must be non-negative")
    if (
        min_duration_ns is not None
        and max_duration_ns is not None
        and max_duration_ns < min_duration_ns
    ):
        raise EpisodeError("max_duration_ns must be greater than or equal to min_duration_ns")

    normalized_topics = _normalize_filter_values(topics, "topics")
    normalized_modalities = _normalize_filter_values(modalities, "modalities")
    observation_filter = _observation_predicate_filter(
        normalized_where,
        topics=normalized_topics,
        modalities=normalized_modalities,
    )
    params = {
        "kind": "episode-predicate",
        "where": normalized_where,
        "source_table": normalized_source_table,
        "source_where": normalized_source_where,
        "before_ns": int(before_ns),
        "after_ns": int(after_ns),
        "merge_gap_ns": int(merge_gap_ns or 0),
        "min_duration_ns": int(min_duration_ns) if min_duration_ns is not None else None,
        "max_duration_ns": int(max_duration_ns) if max_duration_ns is not None else None,
        "topics": list(normalized_topics),
        "modalities": list(normalized_modalities),
    }
    transform_id = f"tfm-episodes-predicate-{_digest(params)}"

    runs = _rows_by_id(lake, "runs", "run_id")
    observations_by_run = _observations_by_run(
        lake,
        where_sql=observation_filter,
    )
    if normalized_source_table is None:
        hits = _predicate_hits_from_observations(observations_by_run)
    else:
        hits = _predicate_hits_from_source(
            lake,
            normalized_source_table,
            normalized_source_where or "",
        )

    planned = _predicate_intervals(
        hits,
        runs,
        observations_by_run,
        before_ns=int(before_ns),
        after_ns=int(after_ns),
        merge_gap_ns=int(merge_gap_ns or 0),
        min_duration_ns=min_duration_ns,
        max_duration_ns=max_duration_ns,
    )
    bounds = [
        _EpisodeBound(
            run_id=interval.run_id,
            start_time_ns=interval.from_timestamp_ns,
            end_time_ns=interval.to_timestamp_ns,
            boundary_source="predicate",
            outcome=None,
            source_event_ids=interval.source_event_ids,
            source_label_ids=interval.source_label_ids,
            source_model_output_ids=interval.source_model_output_ids,
            source_observation_ids=interval.source_observation_ids,
            observation_ids=interval.observation_ids,
            source_query=params,
            metadata={
                "frame_count": interval.frame_count,
            },
        )
        for interval in planned
    ]

    frames_planned = sum(interval.frame_count for interval in planned)
    if dry_run:
        dry = _dry_run_episodes(
            lake,
            bounds,
            transform_id=transform_id,
            boundary_source="predicate",
            params=params,
            overlap_policy=overlap_policy,
            input_tables=_predicate_input_tables(normalized_source_table),
            observations_by_run=observations_by_run,
        )
        return EpisodeMiningReport(
            lake_uri=lake.uri,
            transform_id=transform_id,
            boundary_source="predicate",
            dry_run=True,
            intervals_planned=len(planned),
            frames_planned=frames_planned,
            episodes_written=0,
            videos_written=0,
            frames_tagged=0,
            episode_ids=dry.episode_ids,
            intervals=tuple(planned),
            overlap_policy=dry.overlap_policy,
            overlap_conflicts=dry.overlap_conflicts,
        )

    build_report = _materialize_episodes(
        lake,
        bounds,
        transform_id=transform_id,
        boundary_source="predicate",
        params=params,
        overlap_policy=overlap_policy,
        created_by=created_by,
        input_tables=_predicate_input_tables(normalized_source_table),
        observations_by_run=observations_by_run,
        supersede_transform_ids=supersede_transform_ids,
    )
    return EpisodeMiningReport(
        lake_uri=build_report.lake_uri,
        transform_id=build_report.transform_id,
        boundary_source=build_report.boundary_source,
        dry_run=False,
        intervals_planned=len(planned),
        frames_planned=frames_planned,
        episodes_written=build_report.episodes_written,
        videos_written=build_report.videos_written,
        frames_tagged=build_report.frames_tagged,
        episode_ids=build_report.episode_ids,
        intervals=tuple(planned),
        overlap_policy=overlap_policy,
    )


def build_episodes_from_scenarios(
    lake: Lake,
    *,
    scenario_ids: Sequence[str] | None = None,
    snapshot_name: str | None = None,
    outcome: str | None = None,
    outcome_by_coverage_tag: Mapping[str, str] | None = None,
    outcome_by_summary: Mapping[str, str] | None = None,
    task_id: str | None = None,
    task_id_by_coverage_tag: Mapping[str, str] | None = None,
    task_id_by_summary: Mapping[str, str] | None = None,
    overlap_policy: str = "error",
    dry_run: bool = False,
    created_by: str = "lancedb-robotics",
    supersede_transform_ids: Sequence[str] = (),
) -> EpisodeBuildReport | EpisodeDerivationDryRun:
    """Promote curated scenario windows or a snapshot selection into episodes."""
    if bool(scenario_ids) == bool(snapshot_name):
        raise EpisodeError("provide exactly one of scenario_ids or snapshot_name")

    source_snapshot = None
    snapshot_table_versions: list[dict[str, Any]] = []
    if snapshot_name:
        source_snapshot = _snapshot_row(lake, snapshot_name)
        selected_ids = _snapshot_scenario_ids(source_snapshot)
        snapshot_table_versions = list(source_snapshot.get("table_versions") or [])
        scenarios = _scenarios_as_of(
            lake,
            _snapshot_table_version(source_snapshot, "scenarios"),
        )
    else:
        selected_ids = _normalize_scenario_ids(scenario_ids or ())
        scenarios = _rows_by_id(lake, "scenarios", "scenario_id")

    if not selected_ids:
        raise EpisodeError("no scenarios selected for promotion")
    unknown = [scenario_id for scenario_id in selected_ids if scenario_id not in scenarios]
    if unknown:
        raise EpisodeError(f"unknown scenario ids: {unknown}")

    outcome_by_coverage_tag = _normalize_mapping(outcome_by_coverage_tag, "outcome_by_coverage_tag")
    outcome_by_summary = _normalize_mapping(outcome_by_summary, "outcome_by_summary")
    task_id_by_coverage_tag = _normalize_mapping(task_id_by_coverage_tag, "task_id_by_coverage_tag")
    task_id_by_summary = _normalize_mapping(task_id_by_summary, "task_id_by_summary")
    runs = _rows_by_id(lake, "runs", "run_id")

    params = {
        "kind": "episode-scenarios",
        "scenario_ids": sorted(selected_ids),
        "snapshot_name": source_snapshot.get("name") if source_snapshot else None,
        "snapshot_dataset_id": source_snapshot.get("dataset_id") if source_snapshot else None,
        "snapshot_table_versions": snapshot_table_versions,
        "outcome": outcome,
        "outcome_by_coverage_tag": dict(sorted(outcome_by_coverage_tag.items())),
        "outcome_by_summary": dict(sorted(outcome_by_summary.items())),
        "task_id": task_id,
        "task_id_by_coverage_tag": dict(sorted(task_id_by_coverage_tag.items())),
        "task_id_by_summary": dict(sorted(task_id_by_summary.items())),
    }
    transform_id = f"tfm-episodes-scenarios-{_digest(params)}"

    bounds: list[_EpisodeBound] = []
    for scenario in sorted(
        (scenarios[scenario_id] for scenario_id in selected_ids), key=_scenario_sort_key
    ):
        run_id = scenario["run_id"]
        run = runs.get(run_id)
        if run is None:
            raise EpisodeError(
                f"scenario {scenario['scenario_id']!r} references unknown run {run_id!r}"
            )
        start_ns = int(scenario["start_time_ns"])
        end_ns = int(scenario["end_time_ns"])
        if end_ns < start_ns:
            raise EpisodeError(
                f"scenario {scenario['scenario_id']!r} has end_time_ns earlier than start_time_ns"
            )
        trigger_event_id = str(scenario.get("trigger_event_id") or "")
        bounds.append(
            _EpisodeBound(
                run_id=run_id,
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                boundary_source="scenarios",
                outcome=_scenario_outcome(
                    scenario,
                    outcome=outcome,
                    outcome_by_coverage_tag=outcome_by_coverage_tag,
                    outcome_by_summary=outcome_by_summary,
                ),
                source_event_ids=(trigger_event_id,) if trigger_event_id else (),
                source_scenario_ids=(scenario["scenario_id"],),
                observation_ids=_scenario_observation_ids(scenario),
                source_query=params,
                task_id=_scenario_task_id(
                    scenario,
                    run,
                    task_id=task_id,
                    task_id_by_coverage_tag=task_id_by_coverage_tag,
                    task_id_by_summary=task_id_by_summary,
                ),
            )
        )

    input_tables: tuple[str, ...] = ("runs", "observations", "scenarios")
    if source_snapshot:
        input_tables = (*input_tables, "dataset_snapshots")
    if dry_run:
        return _dry_run_episodes(
            lake,
            bounds,
            transform_id=transform_id,
            boundary_source="scenarios",
            params=params,
            overlap_policy=overlap_policy,
            input_tables=input_tables,
        )
    return _materialize_episodes(
        lake,
        bounds,
        transform_id=transform_id,
        boundary_source="scenarios",
        params=params,
        overlap_policy=overlap_policy,
        created_by=created_by,
        input_tables=input_tables,
        supersede_transform_ids=supersede_transform_ids,
    )


def build_episodes_from_intervals(
    lake: Lake,
    intervals: Sequence[Mapping[str, Any]],
    *,
    time_unit: str = "ns",
    allow_clipped: bool = False,
    allow_empty: bool = False,
    source_uri: str | None = None,
    source_sha256: str | None = None,
    overlap_policy: str = "error",
    dry_run: bool = False,
    created_by: str = "lancedb-robotics",
    supersede_transform_ids: Sequence[str] = (),
) -> EpisodeBuildReport | EpisodeDerivationDryRun:
    """Import externally authored interval records as first-class episodes."""
    normalized_unit = _normalize_time_unit(time_unit)
    validated = _validate_interval_records(
        lake,
        intervals,
        time_unit=normalized_unit,
        allow_clipped=allow_clipped,
        allow_empty=allow_empty,
    )
    params = {
        "kind": "episode-intervals",
        "time_unit": normalized_unit,
        "allow_clipped": bool(allow_clipped),
        "allow_empty": bool(allow_empty),
        "source_uri": source_uri,
        "source_sha256": source_sha256,
        "intervals": [_interval_params(interval) for interval in validated],
    }
    transform_id = f"tfm-episodes-intervals-{_digest(params)}"
    bounds = [
        _EpisodeBound(
            run_id=interval.run_id,
            start_time_ns=interval.start_time_ns,
            end_time_ns=interval.end_time_ns,
            boundary_source="intervals",
            outcome=interval.outcome,
            source_query=params,
            task_id=interval.task_id,
            external_id=interval.external_id,
            tags=interval.tags,
            metadata=interval.metadata,
            source_provenance=interval.source_provenance,
            source_payload=interval.source_payload,
            clipped=interval.clipped or None,
        )
        for interval in validated
    ]

    if dry_run:
        return _dry_run_episodes(
            lake,
            bounds,
            transform_id=transform_id,
            boundary_source="intervals",
            params=params,
            overlap_policy=overlap_policy,
            input_tables=("runs", "observations"),
            input_uris=(source_uri,) if source_uri else (),
        )

    return _materialize_episodes(
        lake,
        bounds,
        transform_id=transform_id,
        boundary_source="intervals",
        params=params,
        overlap_policy=overlap_policy,
        created_by=created_by,
        input_tables=("runs", "observations"),
        input_uris=(source_uri,) if source_uri else (),
        supersede_transform_ids=supersede_transform_ids,
    )


def _dry_run_episodes(
    lake: Lake,
    bounds: list[_EpisodeBound],
    *,
    transform_id: str,
    boundary_source: str,
    params: dict[str, Any],
    overlap_policy: str,
    input_tables: Sequence[str] = ("runs", "observations", "events"),
    input_uris: Sequence[str] = (),
    observations_by_run: dict[str, list[dict[str, Any]]] | None = None,
) -> EpisodeDerivationDryRun:
    del input_tables, input_uris, params
    policy = _normalize_overlap_policy(overlap_policy)
    runs = _rows_by_id(lake, "runs", "run_id")
    observations_by_run = observations_by_run or _observations_by_run(lake)
    now = datetime.now(UTC)
    candidate = _episode_output_rows(bounds, runs, observations_by_run, transform_id, now)
    conflicts = _overlap_conflicts(lake, candidate["frame_updates"], transform_id)
    output = candidate
    if policy == "preserve" and conflicts:
        output = _episode_output_rows(
            bounds,
            runs,
            observations_by_run,
            transform_id,
            now,
            excluded_observation_ids={conflict.observation_id for conflict in conflicts},
        )
    return EpisodeDerivationDryRun(
        lake_uri=lake.uri,
        transform_id=transform_id,
        boundary_source=boundary_source,
        overlap_policy=policy,
        episodes_planned=len(output["episodes"]),
        videos_planned=len(output["videos"]),
        frames_planned=len(output["frame_updates"]),
        episode_ids=tuple(sorted(row["episode_id"] for row in output["episodes"])),
        overlap_conflicts=tuple(conflicts),
    )


def _materialize_episodes(
    lake: Lake,
    bounds: list[_EpisodeBound],
    *,
    transform_id: str,
    boundary_source: str,
    params: dict[str, Any],
    overlap_policy: str,
    created_by: str,
    input_tables: Sequence[str] = ("runs", "observations", "events"),
    input_uris: Sequence[str] = (),
    observations_by_run: dict[str, list[dict[str, Any]]] | None = None,
    supersede_transform_ids: Sequence[str] = (),
) -> EpisodeBuildReport:
    policy = _normalize_overlap_policy(overlap_policy)
    input_versions = _table_versions(lake, input_tables)
    runs = _rows_by_id(lake, "runs", "run_id")
    observations_by_run = observations_by_run or _observations_by_run(lake)
    now = datetime.now(UTC)
    output = _episode_output_rows(bounds, runs, observations_by_run, transform_id, now)
    conflicts = _overlap_conflicts(lake, output["frame_updates"], transform_id)
    explicit_supersede_ids = tuple(dict.fromkeys(str(value) for value in supersede_transform_ids))

    if policy == "preserve" and conflicts:
        output = _episode_output_rows(
            bounds,
            runs,
            observations_by_run,
            transform_id,
            now,
            excluded_observation_ids={conflict.observation_id for conflict in conflicts},
        )
        conflicts = ()
    elif policy == "supersede":
        if explicit_supersede_ids:
            unexpected = [
                conflict
                for conflict in conflicts
                if conflict.existing_transform_id not in explicit_supersede_ids
            ]
            if unexpected:
                raise _conflict_error(policy, unexpected)
            superseded_ids = explicit_supersede_ids
        else:
            superseded_ids = tuple(
                sorted({conflict.existing_transform_id for conflict in conflicts})
            )
    else:
        if conflicts:
            raise _conflict_error(policy, conflicts)
        superseded_ids = ()

    if policy != "supersede":
        superseded_ids = ()

    clear_transform_ids = tuple(dict.fromkeys((transform_id, *superseded_ids)))
    observation_updates = _clear_updates_for_transforms(lake, clear_transform_ids)
    observation_updates.update(output["frame_updates"])

    episodes = lake.table("episodes")
    episodes.delete(f"transform_id = '{transform_id}'")
    if output["episodes"]:
        episodes.add(pa.Table.from_pylist(output["episodes"], schema=EPISODES_SCHEMA))

    videos = lake.table("videos")
    videos.delete(f"transform_id = '{transform_id}'")
    if output["videos"]:
        videos.add(pa.Table.from_pylist(output["videos"], schema=VIDEOS_SCHEMA))

    _apply_observation_episode_columns(lake, observation_updates)
    output_episode_ids = sorted(row["episode_id"] for row in output["episodes"])
    record_params = {
        **params,
        "episode_ids": output_episode_ids,
        "overlap_policy": policy,
    }
    if superseded_ids:
        record_params["supersedes_transform_ids"] = sorted(superseded_ids)
    _record_transform(
        lake,
        transform_id=transform_id,
        boundary_source=boundary_source,
        params=record_params,
        input_table_versions=input_versions,
        input_uris=input_uris,
        created_by=created_by,
    )
    for superseded_id in superseded_ids:
        _record_lifecycle_action(
            lake,
            action="supersede",
            target_transform_id=superseded_id,
            replacement_transform_id=transform_id,
            episode_ids=output_episode_ids,
            frames_affected=len(observation_updates),
            created_by=created_by,
        )

    return EpisodeBuildReport(
        lake_uri=lake.uri,
        transform_id=transform_id,
        boundary_source=boundary_source,
        episodes_written=len(output["episodes"]),
        videos_written=len(output["videos"]),
        frames_tagged=len(output["frame_updates"]),
        episode_ids=tuple(output_episode_ids),
    )


def _episode_output_rows(
    bounds: list[_EpisodeBound],
    runs: dict[str, dict[str, Any]],
    observations_by_run: dict[str, list[dict[str, Any]]],
    transform_id: str,
    now: datetime,
    *,
    excluded_observation_ids: set[str] | None = None,
) -> dict[str, Any]:
    episode_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    frame_updates: dict[str, dict[str, Any]] = {}
    excluded_observation_ids = excluded_observation_ids or set()

    sorted_bounds = sorted(
        bounds,
        key=lambda bound: (
            bound.run_id,
            bound.start_time_ns,
            bound.end_time_ns,
            bound.boundary_source,
            bound.source_event_ids,
            bound.source_scenario_ids,
            bound.source_label_ids,
            bound.source_model_output_ids,
        ),
    )
    for episode_index, bound in enumerate(sorted_bounds):
        run = runs[bound.run_id]
        frames = [
            frame
            for frame in _frames_for_bound(bound, observations_by_run)
            if frame["observation_id"] not in excluded_observation_ids
        ]
        if not frames and excluded_observation_ids:
            continue
        episode_id = _episode_id(bound, transform_id)
        camera_frames = [obs for obs in frames if _is_camera_observation(obs)]
        provenance = {
            "source_event_ids": list(bound.source_event_ids),
            "source_scenario_ids": list(bound.source_scenario_ids),
            "source_query": bound.source_query,
            "run_id": bound.run_id,
            "from_timestamp_ns": bound.start_time_ns,
            "to_timestamp_ns": bound.end_time_ns,
        }
        source_observation_ids = (
            bound.source_observation_ids
            if bound.source_observation_ids is not None
            else bound.observation_ids
        )
        if source_observation_ids is not None:
            provenance["source_observation_ids"] = list(source_observation_ids)
        if bound.source_label_ids:
            provenance["source_label_ids"] = list(bound.source_label_ids)
        if bound.source_model_output_ids:
            provenance["source_model_output_ids"] = list(bound.source_model_output_ids)
        if bound.external_id is not None:
            provenance["external_id"] = bound.external_id
        if bound.tags:
            provenance["tags"] = list(bound.tags)
        if bound.metadata:
            provenance["metadata"] = bound.metadata
        if bound.source_provenance:
            provenance["source_provenance"] = bound.source_provenance
        if bound.source_payload is not None:
            provenance["source_payload"] = bound.source_payload
        if bound.clipped:
            provenance["clipped"] = bound.clipped
        task_id = bound.task_id if bound.task_id is not None else run.get("task_id")
        episode_rows.append(
            {
                "episode_id": episode_id,
                "run_id": bound.run_id,
                "episode_index": episode_index,
                "from_timestamp_ns": bound.start_time_ns,
                "to_timestamp_ns": bound.end_time_ns,
                "boundary_source": bound.boundary_source,
                "outcome": bound.outcome,
                "frame_count": len(frames),
                "camera_blobs": [obs["observation_id"] for obs in camera_frames],
                "task_id": task_id,
                "embedding": None,
                "provenance": json.dumps(provenance, sort_keys=True),
                "transform_id": transform_id,
                "created_at": now,
            }
        )

        for frame_index, obs in enumerate(frames):
            frame_updates.setdefault(
                obs["observation_id"],
                {
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "robot_id": run.get("robot_id"),
                    "site_id": run.get("site_id"),
                    "task_id": task_id,
                    "software_version": run.get("software_version"),
                    "outcome": bound.outcome,
                },
            )

        video_rows.extend(
            _video_rows_for_episode(
                episode_id=episode_id,
                episode_index=episode_index,
                run_id=bound.run_id,
                frames=camera_frames,
                transform_id=transform_id,
                now=now,
            )
        )

    return {"episodes": episode_rows, "videos": video_rows, "frame_updates": frame_updates}


def _video_rows_for_episode(
    *,
    episode_id: str,
    episode_index: int,
    run_id: str,
    frames: list[dict[str, Any]],
    transform_id: str,
    now: datetime,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        grouped.setdefault(_camera_key(frame), []).append(frame)

    rows = []
    for camera_key in sorted(grouped):
        camera_frames = sorted(grouped[camera_key], key=_frame_sort_key)
        first = camera_frames[0]
        video_id = "vid-" + _digest(
            {
                "episode_id": episode_id,
                "camera_key": camera_key,
                "observation_ids": [row["observation_id"] for row in camera_frames],
            }
        )
        rows.append(
            {
                "video_id": video_id,
                "run_id": run_id,
                "episode_id": episode_id,
                "episode_index": episode_index,
                "camera_key": camera_key,
                "sensor_id": first.get("sensor_id"),
                "topic": first.get("topic"),
                "from_timestamp_ns": int(camera_frames[0]["timestamp_ns"]),
                "to_timestamp_ns": int(camera_frames[-1]["timestamp_ns"]),
                "frame_count": len(camera_frames),
                "observation_ids": [row["observation_id"] for row in camera_frames],
                "raw_uri": first.get("raw_uri"),
                "codec": first.get("message_encoding") or first.get("schema_encoding"),
                "uri": None,
                "transform_id": transform_id,
                "created_at": now,
            }
        )
    return rows


def _normalize_overlap_policy(policy: str) -> str:
    value = str(policy or "error").strip().lower().replace("_", "-")
    aliases = {"fail": "error", "same-recipe": "replace"}
    value = aliases.get(value, value)
    if value not in _OVERLAP_POLICIES:
        allowed = ", ".join(sorted(_OVERLAP_POLICIES))
        raise EpisodeError(f"overlap_policy must be one of: {allowed}")
    return value


def _frame_owners(lake: Lake) -> dict[str, _FrameOwner]:
    episodes = {
        row["episode_id"]: row["transform_id"]
        for row in lake.table("episodes").to_arrow().to_pylist()
    }
    owners: dict[str, _FrameOwner] = {}
    for row in lake.table("observations").to_arrow().to_pylist():
        episode_id = row.get("episode_id")
        if not episode_id:
            continue
        owners[row["observation_id"]] = _FrameOwner(
            episode_id=episode_id,
            transform_id=episodes.get(episode_id, ""),
        )
    return owners


def _overlap_conflicts(
    lake: Lake,
    updates: dict[str, dict[str, Any]],
    transform_id: str,
) -> list[EpisodeOverlapConflict]:
    owners = _frame_owners(lake)
    conflicts: list[EpisodeOverlapConflict] = []
    for observation_id in sorted(updates):
        owner = owners.get(observation_id)
        if owner is None or owner.transform_id == transform_id:
            continue
        conflicts.append(
            EpisodeOverlapConflict(
                observation_id=observation_id,
                existing_episode_id=owner.episode_id,
                existing_transform_id=owner.transform_id or "unknown",
                planned_episode_id=str(updates[observation_id].get("episode_id") or ""),
            )
        )
    return conflicts


def _conflict_error(
    policy: str,
    conflicts: Sequence[EpisodeOverlapConflict],
) -> EpisodeError:
    sample = ", ".join(
        f"{conflict.observation_id}:{conflict.existing_transform_id}"
        for conflict in list(conflicts)[:5]
    )
    suffix = "..." if len(conflicts) > 5 else ""
    return EpisodeError(
        f"episode derivation overlap_policy={policy!r} conflicts with "
        f"{len(conflicts)} already-owned frame(s): {sample}{suffix}"
    )


def _clear_updates_for_transforms(
    lake: Lake,
    transform_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    selected = set(transform_ids)
    if not selected:
        return {}
    owners = _frame_owners(lake)
    return {
        observation_id: _empty_episode_update()
        for observation_id, owner in owners.items()
        if owner.transform_id in selected
    }


def _clear_updates_for_episode_ids(
    lake: Lake,
    episode_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    selected = set(episode_ids)
    if not selected:
        return {}
    return {
        row["observation_id"]: _empty_episode_update()
        for row in lake.table("observations").to_arrow().to_pylist()
        if row.get("episode_id") in selected
    }


def _empty_episode_update() -> dict[str, Any]:
    return {name: None for name in _OBSERVATION_EPISODE_FIELD_TYPES}


def _apply_observation_episode_columns(
    lake: Lake,
    updates: dict[str, dict[str, Any]],
) -> None:
    if not updates:
        return
    table = lake.table("observations")
    dataset = table.to_lance()
    schema_names = set(table.schema.names)
    projected = ["observation_id"] + [
        name for name in _OBSERVATION_EPISODE_FIELD_TYPES if name in schema_names
    ]
    current = dataset.to_table(columns=projected)
    if current.num_rows == 0:
        return

    ids = current["observation_id"].to_pylist()
    values_by_column: dict[str, list[Any]] = {}
    for name in _OBSERVATION_EPISODE_FIELD_TYPES:
        values_by_column[name] = (
            current[name].to_pylist() if name in current.column_names else [None] * len(ids)
        )

    for index, observation_id in enumerate(ids):
        update = updates.get(observation_id)
        if update is None:
            continue
        for name, value in update.items():
            values_by_column[name][index] = value

    existing = [name for name in _OBSERVATION_EPISODE_FIELD_TYPES if name in schema_names]
    if existing:
        dataset.drop_columns(existing)
    dataset.add_columns(
        pa.table(
            {
                name: pa.array(values, type=_OBSERVATION_EPISODE_FIELD_TYPES[name])
                for name, values in values_by_column.items()
            }
        )
    )


def _record_transform(
    lake: Lake,
    *,
    transform_id: str,
    boundary_source: str,
    params: dict[str, Any],
    input_table_versions: list[dict[str, Any]],
    created_by: str,
    input_uris: Sequence[str] = (),
) -> None:
    now = datetime.now(UTC)
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{transform_id}'")
    transform_row = {
        "transform_id": transform_id,
        "kind": _DERIVATION_KIND,
        "source_id": None,
        "input_uris": list(input_uris),
        "input_table_versions": input_table_versions,
        "output_tables": ["episodes", "videos", "observations"],
        "params": json.dumps(
            {
                "boundary_source": boundary_source,
                **params,
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
    # Emit lineage inline (backlog 0098): the episode-derivation slice without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)


def _record_lifecycle_action(
    lake: Lake,
    *,
    action: str,
    target_transform_id: str,
    created_by: str,
    replacement_transform_id: str | None = None,
    episode_ids: Sequence[str] = (),
    frames_affected: int = 0,
    details: Mapping[str, Any] | None = None,
) -> str:
    now = datetime.now(UTC)
    params = {
        "action": action,
        "target_transform_id": target_transform_id,
        "replacement_transform_id": replacement_transform_id,
        "episode_ids": list(episode_ids),
        "frames_affected": int(frames_affected),
    }
    if details:
        params["details"] = _jsonable_mapping(details)
    lifecycle_transform_id = "tfm-episode-lifecycle-" + _digest(
        {
            **params,
            "created_at": now.isoformat(),
        }
    )
    transform_row = {
        "transform_id": lifecycle_transform_id,
        "kind": _LIFECYCLE_KIND,
        "source_id": None,
        "input_uris": [],
        "input_table_versions": _table_versions(
            lake,
            ("episodes", "videos", "observations", "transform_runs"),
        ),
        "output_tables": ["transform_runs", "episodes", "videos", "observations"],
        "params": json.dumps(params, sort_keys=True),
        "status": "completed",
        "error": None,
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    lake.table("transform_runs").add(
        pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA)
    )
    # Emit lineage inline (backlog 0098): the episode-lifecycle slice without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
    return lifecycle_transform_id


def _lifecycle_action_rows(lake: Lake) -> tuple[dict[str, Any], ...]:
    return tuple(
        sorted(
            [
                row
                for row in lake.table("transform_runs").to_arrow().to_pylist()
                if row.get("kind") == _LIFECYCLE_KIND
            ],
            key=lambda row: (row.get("created_at"), row["transform_id"]),
        )
    )


def _lifecycle_params(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("params") or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _derivation_statuses(lake: Lake) -> dict[str, tuple[str, str | None]]:
    statuses: dict[str, tuple[str, str | None]] = {}
    for row in _lifecycle_action_rows(lake):
        params = _lifecycle_params(row)
        target = params.get("target_transform_id")
        if not target:
            continue
        action = params.get("action")
        if action == "clear":
            statuses[str(target)] = ("cleared", None)
        elif action == "supersede":
            replacement = params.get("replacement_transform_id")
            statuses[str(target)] = (
                "superseded",
                str(replacement) if replacement else None,
            )
        elif action == "rebuild":
            statuses[str(target)] = ("active", None)
    return statuses


def _rows_by_id(lake: Lake, table_name: str, id_column: str) -> dict[str, dict[str, Any]]:
    return {row[id_column]: row for row in lake.table(table_name).to_arrow().to_pylist()}


def _rows_by_id_as_of(
    lake: Lake,
    table_name: str,
    id_column: str,
    version: int | None,
) -> dict[str, dict[str, Any]]:
    table = lake.table(table_name)
    if version is not None:
        table.checkout(int(version))
    try:
        return {row[id_column]: row for row in table.to_arrow().to_pylist()}
    finally:
        if version is not None:
            table.checkout_latest()


def _events_by_run(lake: Lake) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for event in lake.table("events").to_arrow().to_pylist():
        rows.setdefault(event["run_id"], []).append(event)
    for run_events in rows.values():
        run_events.sort(key=lambda row: (_event_start_ns(row), row["event_id"]))
    return rows


def _observations_by_run(
    lake: Lake,
    where_sql: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for observation in _scan_projected_rows(
        lake,
        "observations",
        _EPISODE_OBSERVATION_COLUMNS,
        where_sql=where_sql,
    ):
        rows.setdefault(observation["run_id"], []).append(observation)
    for run_observations in rows.values():
        run_observations.sort(key=_frame_sort_key)
    return rows


def _table_versions(lake: Lake, tables: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"table": table, "version": int(lake.table(table).version), "tag": ""} for table in tables
    ]


def _snapshot_row(lake: Lake, snapshot_name: str) -> dict[str, Any]:
    rows = [
        row
        for row in lake.table("dataset_snapshots").to_arrow().to_pylist()
        if row["name"] == snapshot_name or row["dataset_id"] == snapshot_name
    ]
    if not rows:
        raise EpisodeError(f"no snapshot named {snapshot_name!r} in {lake.uri}")
    return max(rows, key=lambda row: (row["created_at"], row["dataset_id"]))


def _snapshot_scenario_ids(snapshot: dict[str, Any]) -> tuple[str, ...]:
    query_spec = _json_object(snapshot.get("query_spec"), "query_spec")
    return _normalize_scenario_ids(query_spec.get("scenario_ids") or ())


def _snapshot_table_version(snapshot: dict[str, Any], table_name: str) -> int | None:
    for item in snapshot.get("table_versions") or []:
        if item.get("table") == table_name:
            return int(item["version"])
    return None


def _scenarios_as_of(lake: Lake, version: int | None) -> dict[str, dict[str, Any]]:
    return _rows_by_id_as_of(lake, "scenarios", "scenario_id", version)


def _json_object(raw: Any, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise EpisodeError(f"snapshot {field_name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise EpisodeError(f"snapshot {field_name} must be a JSON object")
    return value


def _normalize_event_types(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))


def _normalize_scenario_ids(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))


def _normalize_mapping(
    mapping: Mapping[str, str] | None,
    field_name: str,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_value in (mapping or {}).items():
        key = str(raw_key).strip()
        value = str(raw_value).strip()
        if not key or not value:
            raise EpisodeError(f"{field_name} cannot contain empty keys or values")
        normalized[key] = value
    return normalized


def _normalize_predicate(predicate: str | None) -> str | None:
    if predicate is None:
        return None
    text = str(predicate).strip()
    return text or None


def _normalize_source_table(source_table: str | None) -> str | None:
    if source_table is None:
        return None
    value = str(source_table).strip().lower().replace("-", "_")
    aliases = {
        "event": "events",
        "label": "labels",
        "model_output": "model_outputs",
        "model_outputs": "model_outputs",
    }
    value = aliases.get(value, value)
    if value not in _SOURCE_TABLES:
        allowed = ", ".join(sorted(_SOURCE_TABLES))
        raise EpisodeError(f"source_table must be one of: {allowed}")
    return value


def _normalize_filter_values(values: Sequence[str] | None, field_name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values or ():
        text = str(value).strip()
        if not text:
            raise EpisodeError(f"{field_name} cannot contain empty values")
        normalized.append(text)
    return tuple(dict.fromkeys(normalized))


def _observation_predicate_filter(
    where_sql: str | None,
    *,
    topics: Sequence[str],
    modalities: Sequence[str],
) -> str | None:
    clauses = []
    if where_sql:
        clauses.append(f"({where_sql})")
    if topics:
        clauses.append(_column_in_filter("topic", topics))
    if modalities:
        clauses.append(_column_in_filter("modality", modalities))
    return " AND ".join(clauses) if clauses else None


def _column_in_filter(column: str, values: Sequence[str]) -> str:
    if len(values) == 1:
        return f"{column} = {_sql_literal(values[0])}"
    return f"{column} IN ({', '.join(_sql_literal(value) for value in values)})"


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _scan_projected_rows(
    lake: Lake,
    table_name: str,
    columns: Sequence[str],
    *,
    where_sql: str | None = None,
) -> list[dict[str, Any]]:
    table = lake.table(table_name)
    available = set(table.schema.names)
    projected = [column for column in columns if column in available]
    if not projected:
        raise EpisodeError(f"no projected columns are present on {table_name!r}")
    try:
        query = table.search().select(projected)
        if where_sql:
            query = query.where(where_sql)
        rows: list[dict[str, Any]] = []
        for batch in query.to_batches(batch_size=4096):
            rows.extend(batch.to_pylist())
        return rows
    except Exception as exc:
        predicate = f" with predicate {where_sql!r}" if where_sql else ""
        raise EpisodeError(f"cannot scan {table_name}{predicate}: {exc}") from exc


def _predicate_hits_from_observations(
    observations_by_run: dict[str, list[dict[str, Any]]],
) -> list[_PredicateHit]:
    hits = []
    for run_id in sorted(observations_by_run):
        for observation in observations_by_run[run_id]:
            timestamp_ns = int(observation["timestamp_ns"])
            hits.append(
                _PredicateHit(
                    run_id=run_id,
                    start_time_ns=timestamp_ns,
                    end_time_ns=timestamp_ns,
                    observation_ids=(observation["observation_id"],),
                )
            )
    return hits


def _predicate_hits_from_source(
    lake: Lake,
    source_table: str,
    source_where: str,
) -> list[_PredicateHit]:
    source_rows = _scan_projected_rows(
        lake,
        source_table,
        _source_columns(source_table),
        where_sql=source_where,
    )
    maps = _predicate_lookup_maps(lake)
    hits = [
        _predicate_hit_from_source_row(row, source_table=source_table, maps=maps)
        for row in source_rows
    ]
    return sorted(
        hits,
        key=lambda hit: (
            hit.run_id,
            hit.start_time_ns,
            hit.end_time_ns,
            hit.observation_ids,
            hit.event_ids,
            hit.label_ids,
            hit.model_output_ids,
        ),
    )


def _source_columns(source_table: str) -> tuple[str, ...]:
    if source_table == "events":
        return ("event_id", "run_id", "timestamp_ns", "start_time_ns", "end_time_ns")
    if source_table == "labels":
        return ("label_id", "run_id", "observation_id", "scenario_id", "event_id")
    return ("model_output_id", "run_id", "observation_id", "scenario_id")


def _predicate_lookup_maps(lake: Lake) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "runs": _rows_by_id(lake, "runs", "run_id"),
        "observations": {
            row["observation_id"]: row
            for row in _scan_projected_rows(lake, "observations", _EPISODE_OBSERVATION_COLUMNS)
        },
        "events": {
            row["event_id"]: row
            for row in _scan_projected_rows(
                lake,
                "events",
                ("event_id", "run_id", "timestamp_ns", "start_time_ns", "end_time_ns"),
            )
        },
        "scenarios": {
            row["scenario_id"]: row
            for row in _scan_projected_rows(
                lake,
                "scenarios",
                ("scenario_id", "run_id", "start_time_ns", "end_time_ns", "observation_ids"),
            )
        },
    }


def _predicate_hit_from_source_row(
    row: dict[str, Any],
    *,
    source_table: str,
    maps: dict[str, dict[str, dict[str, Any]]],
) -> _PredicateHit:
    if source_table == "events":
        return _PredicateHit(
            run_id=_required_source_id(row, "run_id", "event"),
            start_time_ns=_event_start_ns(row),
            end_time_ns=_event_end_ns(row),
            event_ids=(row["event_id"],),
        )

    source_id = _required_source_id(
        row,
        "label_id" if source_table == "labels" else "model_output_id",
        source_table[:-1],
    )
    source_kwargs: dict[str, tuple[str, ...]] = (
        {"label_ids": (source_id,)}
        if source_table == "labels"
        else {"model_output_ids": (source_id,)}
    )
    explicit_run_id = row.get("run_id") or None
    observation_id = row.get("observation_id") or None
    if observation_id:
        observation = maps["observations"].get(observation_id)
        if observation is None:
            raise EpisodeError(f"{source_table} row {source_id!r} references unknown observation")
        run_id = _coalesced_run_id(explicit_run_id, observation["run_id"], source_id)
        timestamp_ns = int(observation["timestamp_ns"])
        return _PredicateHit(
            run_id=run_id,
            start_time_ns=timestamp_ns,
            end_time_ns=timestamp_ns,
            observation_ids=(observation_id,),
            **source_kwargs,
        )

    event_id = row.get("event_id") or None
    if event_id:
        event = maps["events"].get(event_id)
        if event is None:
            raise EpisodeError(f"{source_table} row {source_id!r} references unknown event")
        run_id = _coalesced_run_id(explicit_run_id, event["run_id"], source_id)
        return _PredicateHit(
            run_id=run_id,
            start_time_ns=_event_start_ns(event),
            end_time_ns=_event_end_ns(event),
            event_ids=(event_id,),
            **source_kwargs,
        )

    scenario_id = row.get("scenario_id") or None
    if scenario_id:
        scenario = maps["scenarios"].get(scenario_id)
        if scenario is None:
            raise EpisodeError(f"{source_table} row {source_id!r} references unknown scenario")
        run_id = _coalesced_run_id(explicit_run_id, scenario["run_id"], source_id)
        return _PredicateHit(
            run_id=run_id,
            start_time_ns=int(scenario["start_time_ns"]),
            end_time_ns=int(scenario["end_time_ns"]),
            observation_ids=tuple(str(value) for value in scenario.get("observation_ids") or ()),
            **source_kwargs,
        )

    if explicit_run_id:
        run = maps["runs"].get(explicit_run_id)
        if run is None:
            raise EpisodeError(f"{source_table} row {source_id!r} references unknown run")
        return _PredicateHit(
            run_id=explicit_run_id,
            start_time_ns=int(run["start_time_ns"]),
            end_time_ns=int(run["end_time_ns"]),
            **source_kwargs,
        )

    raise EpisodeError(
        f"{source_table} row {source_id!r} must reference an observation, scenario, event, or run"
    )


def _required_source_id(row: dict[str, Any], key: str, noun: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise EpisodeError(f"{noun} row is missing {key}")
    return value


def _coalesced_run_id(explicit: str | None, implied: str, source_id: str) -> str:
    if explicit is not None and explicit != implied:
        raise EpisodeError(
            f"source row {source_id!r} run_id {explicit!r} does not match target run_id {implied!r}"
        )
    return implied


def _predicate_intervals(
    hits: Sequence[_PredicateHit],
    runs: dict[str, dict[str, Any]],
    observations_by_run: dict[str, list[dict[str, Any]]],
    *,
    before_ns: int,
    after_ns: int,
    merge_gap_ns: int,
    min_duration_ns: int | None,
    max_duration_ns: int | None,
) -> list[EpisodePredicateInterval]:
    expanded: list[dict[str, Any]] = []
    for hit in hits:
        run = runs.get(hit.run_id)
        if run is None:
            continue
        start_ns = max(int(run["start_time_ns"]), int(hit.start_time_ns) - before_ns)
        end_ns = min(int(run["end_time_ns"]), int(hit.end_time_ns) + after_ns)
        if end_ns < start_ns:
            continue
        expanded.append(
            {
                "run_id": hit.run_id,
                "start_time_ns": start_ns,
                "end_time_ns": end_ns,
                "source_observation_ids": list(hit.observation_ids),
                "source_event_ids": list(hit.event_ids),
                "source_label_ids": list(hit.label_ids),
                "source_model_output_ids": list(hit.model_output_ids),
            }
        )

    merged = _merge_predicate_segments(expanded, merge_gap_ns=merge_gap_ns)
    intervals: list[EpisodePredicateInterval] = []
    for segment in merged:
        duration_ns = int(segment["end_time_ns"]) - int(segment["start_time_ns"])
        if min_duration_ns is not None and duration_ns < min_duration_ns:
            continue
        if max_duration_ns is not None and duration_ns > max_duration_ns:
            continue
        frames = [
            observation
            for observation in observations_by_run.get(segment["run_id"], [])
            if int(segment["start_time_ns"])
            <= int(observation["timestamp_ns"])
            <= int(segment["end_time_ns"])
        ]
        if not frames:
            continue
        intervals.append(
            EpisodePredicateInterval(
                run_id=segment["run_id"],
                from_timestamp_ns=int(segment["start_time_ns"]),
                to_timestamp_ns=int(segment["end_time_ns"]),
                frame_count=len(frames),
                observation_ids=tuple(observation["observation_id"] for observation in frames),
                source_observation_ids=tuple(segment["source_observation_ids"]),
                source_event_ids=tuple(segment["source_event_ids"]),
                source_label_ids=tuple(segment["source_label_ids"]),
                source_model_output_ids=tuple(segment["source_model_output_ids"]),
            )
        )
    return intervals


def _merge_predicate_segments(
    segments: list[dict[str, Any]],
    *,
    merge_gap_ns: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for segment in sorted(
        segments,
        key=lambda item: (
            item["run_id"],
            item["start_time_ns"],
            item["end_time_ns"],
            item["source_observation_ids"],
            item["source_event_ids"],
            item["source_label_ids"],
            item["source_model_output_ids"],
        ),
    ):
        if (
            merged
            and merged[-1]["run_id"] == segment["run_id"]
            and int(segment["start_time_ns"]) <= int(merged[-1]["end_time_ns"]) + merge_gap_ns
        ):
            current = merged[-1]
            current["end_time_ns"] = max(int(current["end_time_ns"]), int(segment["end_time_ns"]))
            for key in (
                "source_observation_ids",
                "source_event_ids",
                "source_label_ids",
                "source_model_output_ids",
            ):
                current[key] = _extend_unique(current[key], segment[key])
            continue
        merged.append(
            {
                **segment,
                "source_observation_ids": list(segment["source_observation_ids"]),
                "source_event_ids": list(segment["source_event_ids"]),
                "source_label_ids": list(segment["source_label_ids"]),
                "source_model_output_ids": list(segment["source_model_output_ids"]),
            }
        )
    return merged


def _extend_unique(existing: Sequence[str], incoming: Sequence[str]) -> list[str]:
    values = list(existing)
    seen = set(values)
    for value in incoming:
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def _predicate_input_tables(source_table: str | None) -> tuple[str, ...]:
    tables = ["runs", "observations"]
    if source_table is not None:
        tables.append(source_table)
    if source_table in {"labels", "model_outputs"}:
        tables.extend(["events", "scenarios"])
    return tuple(dict.fromkeys(tables))


def _interval_manifest_format(path: Path, requested: str | None) -> str:
    value = (requested or path.suffix.lstrip(".") or "jsonl").strip().lower()
    if value == "json":
        value = "jsonl"
    if value not in {"jsonl", "csv"}:
        raise EpisodeError("interval manifest format must be one of: jsonl, csv")
    return value


def _validate_interval_records(
    lake: Lake,
    intervals: Sequence[Mapping[str, Any]],
    *,
    time_unit: str,
    allow_clipped: bool,
    allow_empty: bool,
) -> tuple[_ValidatedInterval, ...]:
    records = list(intervals)
    if not records:
        raise EpisodeError("no interval records supplied")

    runs = _rows_by_id(lake, "runs", "run_id")
    observations_by_run = _observations_by_run(lake)
    external_ids: dict[str, int] = {}
    validated: list[_ValidatedInterval] = []

    for index, raw_record in enumerate(records, start=1):
        if not isinstance(raw_record, Mapping):
            raise EpisodeError(f"interval {index} must be an object")
        record = dict(raw_record)
        source_payload = _jsonable_mapping(record)
        external_id = _optional_text(
            record,
            ("external_id", "external_episode_id", "id"),
        )
        label = _interval_label(index, external_id)
        if external_id is not None:
            prior = external_ids.get(external_id)
            if prior is not None:
                raise EpisodeError(
                    f"{label} reuses external_id {external_id!r}; already seen in interval {prior}"
                )
            external_ids[external_id] = index

        run_id = _required_text(record, ("run_id",), "run_id", label)
        start_ns = _timestamp_ns(
            record,
            ns_fields=_START_TIME_NS_FIELDS,
            fields=_START_TIME_FIELDS,
            field_name="from_timestamp_ns",
            time_unit=time_unit,
            label=label,
        )
        end_ns = _timestamp_ns(
            record,
            ns_fields=_END_TIME_NS_FIELDS,
            fields=_END_TIME_FIELDS,
            field_name="to_timestamp_ns",
            time_unit=time_unit,
            label=label,
        )
        if end_ns < start_ns:
            raise EpisodeError(f"{label} has to_timestamp_ns earlier than from_timestamp_ns")

        run = runs.get(run_id)
        if run is None:
            raise EpisodeError(f"{label} references unknown run_id {run_id!r}")

        run_start = int(run["start_time_ns"])
        run_end = int(run["end_time_ns"])
        effective_start = start_ns
        effective_end = end_ns
        clipped: dict[str, dict[str, int]] = {}
        if effective_start < run_start:
            if not allow_clipped:
                raise EpisodeError(f"{label} starts before run {run_id!r} bounds")
            clipped["from_timestamp_ns"] = {"original": effective_start, "effective": run_start}
            effective_start = run_start
        if effective_end > run_end:
            if not allow_clipped:
                raise EpisodeError(f"{label} ends after run {run_id!r} bounds")
            clipped["to_timestamp_ns"] = {"original": effective_end, "effective": run_end}
            effective_end = run_end
        if effective_end < effective_start:
            raise EpisodeError(f"{label} is outside run {run_id!r} after clipping")

        has_observation = any(
            effective_start <= int(observation["timestamp_ns"]) <= effective_end
            for observation in observations_by_run.get(run_id, [])
        )
        if not has_observation and not allow_empty:
            raise EpisodeError(f"{label} contains no observations")

        validated.append(
            _ValidatedInterval(
                run_id=run_id,
                start_time_ns=effective_start,
                end_time_ns=effective_end,
                original_start_time_ns=start_ns,
                original_end_time_ns=end_ns,
                external_id=external_id,
                outcome=_optional_text(record, ("outcome", "result")),
                task_id=_optional_text(record, ("task_id", "task")),
                tags=_tags_value(record, label),
                metadata=_object_value(record, "metadata", label),
                source_provenance=_object_value(record, "provenance", label),
                source_payload=source_payload,
                clipped=clipped,
            )
        )
    return tuple(validated)


def _interval_params(interval: _ValidatedInterval) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": interval.run_id,
        "from_timestamp_ns": interval.original_start_time_ns,
        "to_timestamp_ns": interval.original_end_time_ns,
        "effective_from_timestamp_ns": interval.start_time_ns,
        "effective_to_timestamp_ns": interval.end_time_ns,
        "source_payload": interval.source_payload,
    }
    if interval.external_id is not None:
        payload["external_id"] = interval.external_id
    if interval.outcome is not None:
        payload["outcome"] = interval.outcome
    if interval.task_id is not None:
        payload["task_id"] = interval.task_id
    if interval.tags:
        payload["tags"] = list(interval.tags)
    if interval.metadata:
        payload["metadata"] = interval.metadata
    if interval.source_provenance:
        payload["provenance"] = interval.source_provenance
    if interval.clipped:
        payload["clipped"] = interval.clipped
    return payload


def _normalize_time_unit(time_unit: str) -> str:
    unit = str(time_unit).strip().lower()
    if unit not in _TIME_UNIT_MULTIPLIERS:
        raise EpisodeError("time_unit must be one of: ns, us, ms, s")
    return unit


def _timestamp_ns(
    record: Mapping[str, Any],
    *,
    ns_fields: Sequence[str],
    fields: Sequence[str],
    field_name: str,
    time_unit: str,
    label: str,
) -> int:
    for key in ns_fields:
        if _has_value(record.get(key)):
            return _parse_timestamp_ns(record[key], unit="ns", field_name=key, label=label)
    for key in fields:
        if _has_value(record.get(key)):
            return _parse_timestamp_ns(record[key], unit=time_unit, field_name=key, label=label)
    raise EpisodeError(f"{label} is missing {field_name}")


def _parse_timestamp_ns(value: Any, *, unit: str, field_name: str, label: str) -> int:
    if isinstance(value, bool) or not _has_value(value):
        raise EpisodeError(f"{label} has invalid {field_name}")
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise EpisodeError(f"{label} has invalid {field_name}") from exc
    nanoseconds = amount * _TIME_UNIT_MULTIPLIERS[unit]
    integral = nanoseconds.to_integral_value()
    if integral != nanoseconds:
        raise EpisodeError(f"{label} {field_name} does not resolve to whole nanoseconds")
    return int(integral)


def _required_text(
    record: Mapping[str, Any],
    keys: Sequence[str],
    field_name: str,
    label: str,
) -> str:
    value = _optional_text(record, keys)
    if value is None:
        raise EpisodeError(f"{label} is missing {field_name}")
    return value


def _optional_text(record: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if _has_value(value):
            return str(value).strip()
    return None


def _tags_value(record: Mapping[str, Any], label: str) -> tuple[str, ...]:
    raw = record.get("tags")
    if not _has_value(raw):
        raw = record.get("tag")
    if not _has_value(raw):
        return ()

    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                values = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EpisodeError(f"{label} tags must be a JSON list or comma list") from exc
        else:
            values = re.split(r"[,;]", stripped)
    else:
        values = raw

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise EpisodeError(f"{label} tags must be a list")
    normalized = [str(value).strip() for value in values if _has_value(value)]
    return tuple(dict.fromkeys(value for value in normalized if value))


def _object_value(record: Mapping[str, Any], field_name: str, label: str) -> dict[str, Any]:
    value = record.get(field_name)
    if not _has_value(value):
        parsed: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        parsed = _jsonable_mapping(value)
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise EpisodeError(f"{label} {field_name} must be a JSON object") from exc
        if not isinstance(decoded, Mapping):
            raise EpisodeError(f"{label} {field_name} must be a JSON object")
        parsed = _jsonable_mapping(decoded)
    else:
        raise EpisodeError(f"{label} {field_name} must be a JSON object")

    for key, prefixed_value in _prefixed_values(record, f"{field_name}.").items():
        parsed[key] = prefixed_value
    return parsed


def _prefixed_values(record: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {
        str(key)[len(prefix) :]: _jsonable(value)
        for key, value in record.items()
        if str(key).startswith(prefix) and str(key)[len(prefix) :] and _has_value(value)
    }


def _jsonable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(item) for key, item in value.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _has_value(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or value.strip() != "")


def _interval_label(index: int, external_id: str | None) -> str:
    if external_id is None:
        return f"interval {index}"
    return f"interval {index} ({external_id})"


def _event_start_ns(event: dict[str, Any]) -> int:
    return int(event.get("start_time_ns") or event.get("timestamp_ns") or 0)


def _event_end_ns(event: dict[str, Any]) -> int:
    return int(event.get("end_time_ns") or event.get("timestamp_ns") or _event_start_ns(event))


def _marker_outcome(event: dict[str, Any], outcome_by_event_type: Mapping[str, str]) -> str | None:
    event_type = event.get("event_type")
    if event_type in outcome_by_event_type:
        return outcome_by_event_type[event_type]
    text = str(event.get("notes") or "").strip().lower()
    if text in {"success", "succeeded", "pass", "passed"}:
        return "success"
    if text in {"failure", "failed", "fail"}:
        return "failure"
    return None


def _scenario_outcome(
    scenario: dict[str, Any],
    *,
    outcome: str | None,
    outcome_by_coverage_tag: Mapping[str, str],
    outcome_by_summary: Mapping[str, str],
) -> str | None:
    if outcome is not None:
        return str(outcome)
    for tag in scenario.get("coverage_tags") or []:
        if tag in outcome_by_coverage_tag:
            return outcome_by_coverage_tag[tag]
    summary = str(scenario.get("summary") or "").strip()
    if summary in outcome_by_summary:
        return outcome_by_summary[summary]
    return None


def _scenario_task_id(
    scenario: dict[str, Any],
    run: dict[str, Any],
    *,
    task_id: str | None,
    task_id_by_coverage_tag: Mapping[str, str],
    task_id_by_summary: Mapping[str, str],
) -> str | None:
    if task_id is not None:
        return str(task_id)
    for tag in scenario.get("coverage_tags") or []:
        if tag in task_id_by_coverage_tag:
            return task_id_by_coverage_tag[tag]
    summary = str(scenario.get("summary") or "").strip()
    if summary in task_id_by_summary:
        return task_id_by_summary[summary]
    if summary:
        return summary
    return run.get("task_id")


def _scenario_observation_ids(scenario: dict[str, Any]) -> tuple[str, ...] | None:
    values = [str(value) for value in scenario.get("observation_ids") or [] if str(value)]
    if not values:
        return None
    return tuple(dict.fromkeys(values))


def _scenario_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("run_id") or ""),
        int(row.get("start_time_ns") or 0),
        int(row.get("end_time_ns") or 0),
        str(row.get("scenario_id") or ""),
    )


def _frames_for_bound(
    bound: _EpisodeBound,
    observations_by_run: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    run_observations = observations_by_run.get(bound.run_id, [])
    if bound.observation_ids is not None:
        by_id = {row["observation_id"]: row for row in run_observations}
        missing = [
            observation_id
            for observation_id in bound.observation_ids
            if observation_id not in by_id
        ]
        if missing:
            raise EpisodeError(
                f"scenario promotion references observations not present in run "
                f"{bound.run_id!r}: {missing}"
            )
        return [by_id[observation_id] for observation_id in bound.observation_ids]

    return [
        obs
        for obs in run_observations
        if bound.start_time_ns <= int(obs["timestamp_ns"]) <= bound.end_time_ns
    ]


def _episode_id(bound: _EpisodeBound, transform_id: str) -> str:
    payload = {
        "transform_id": transform_id,
        "run_id": bound.run_id,
        "start_time_ns": bound.start_time_ns,
        "end_time_ns": bound.end_time_ns,
        "boundary_source": bound.boundary_source,
        "source_event_ids": bound.source_event_ids,
    }
    if bound.source_scenario_ids:
        payload["source_scenario_ids"] = bound.source_scenario_ids
    if bound.source_observation_ids:
        payload["source_observation_ids"] = bound.source_observation_ids
    if bound.source_label_ids:
        payload["source_label_ids"] = bound.source_label_ids
    if bound.source_model_output_ids:
        payload["source_model_output_ids"] = bound.source_model_output_ids
    return "ep-" + _digest(payload)


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _frame_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    frame_index = row.get("frame_index")
    return (
        frame_index if frame_index is not None else 10**18,
        int(row.get("timestamp_ns") or 0),
        int(row.get("raw_sequence") or 0),
        str(row.get("topic") or ""),
        str(row.get("observation_id") or ""),
    )


def _is_camera_observation(obs: dict[str, Any]) -> bool:
    modality = str(obs.get("modality") or "").lower()
    topic = str(obs.get("topic") or "").lower()
    sensor = str(obs.get("sensor_id") or "").lower()
    return modality in {"image", "camera", "video"} or "camera" in topic or "camera" in sensor


def _camera_key(obs: dict[str, Any]) -> str:
    base = str(obs.get("topic") or obs.get("sensor_id") or "camera")
    key = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    return key or "camera"


def _matches_stream(row: dict[str, Any], stream: str) -> bool:
    return stream in {
        str(row.get("topic") or ""),
        str(row.get("sensor_id") or ""),
        str(row.get("modality") or ""),
    }


def _nearest_frame(frames: tuple[dict[str, Any], ...], timestamp_ns: int) -> dict[str, Any] | None:
    if not frames:
        return None
    return min(
        frames,
        key=lambda row: (
            abs(int(row["timestamp_ns"]) - timestamp_ns),
            int(row.get("frame_index") or 0),
            str(row["observation_id"]),
        ),
    )
