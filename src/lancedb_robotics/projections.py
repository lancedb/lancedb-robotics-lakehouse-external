"""External training projection registry and manifest contract.

``lake.training`` is the Lance-native training surface. This module owns
boundary-format projections from a pinned snapshot into external conventions
such as LeRobot and RLDS. A projection can be live, materialized, or a dry-run
plan, but every mode returns the same lineage-bearing manifest shape.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.lineage_hooks import (
    attach_lineage_context_to_params,
    begin_lineage_execution,
    normalize_lineage_context,
)
from lancedb_robotics.materialization import (
    ProjectionAccounting,
    json_metadata_bytes,
    metadata_bytes_written,
    payload_size,
)
from lancedb_robotics.schemas import TRANSFORM_RUNS_SCHEMA

if TYPE_CHECKING:
    from lancedb_robotics.lake import Lake
    from lancedb_robotics.training import LanceTrainingDataset


PROJECTION_MANIFEST_FILENAME = "projection_manifest.json"
DEFAULT_WEBDATASET_SHARD_SIZE = 1000
DEFAULT_WEBDATASET_COMPRESSION = "none"
LEROBOT_REQUIRED_COLUMNS = (
    "observation_id",
    "episode_id",
    "scenario_id",
    "run_id",
    "split",
    "episode_index",
    "frame_index",
    "timestamp_ns",
    "relative_time_s",
    "sensor_id",
    "topic",
    "modality",
    "state_vector",
    "action_vector",
    "caption",
    "payload",
    "raw_uri",
    "raw_channel",
    "raw_sequence",
)
WEBDATASET_REQUIRED_COLUMNS = (
    "observation_id",
    "episode_id",
    "scenario_id",
    "run_id",
    "split",
    "episode_index",
    "frame_index",
    "timestamp_ns",
    "relative_time_s",
    "sensor_id",
    "topic",
    "modality",
    "state_vector",
    "action_vector",
    "caption",
    "payload",
    "raw_uri",
    "raw_channel",
    "raw_sequence",
    "message_encoding",
    "schema_encoding",
)
RLDS_REQUIRED_COLUMNS = (
    "observation_id",
    "episode_id",
    "scenario_id",
    "run_id",
    "split",
    "episode_index",
    "frame_index",
    "timestamp_ns",
    "relative_time_s",
    "sensor_id",
    "topic",
    "modality",
    "state_vector",
    "action_vector",
    "caption",
    "payload_json",
    "payload",
    "raw_uri",
    "raw_channel",
    "raw_sequence",
    "message_encoding",
    "schema_encoding",
)


class ProjectionMode(StrEnum):
    """Supported projection execution modes."""

    LIVE = "live"
    EXPORT = "export"
    PLAN = "plan"


class ProjectionError(Exception):
    """Raised when a projection cannot be planned or run."""


class ProjectionDependencyError(ProjectionError):
    """Raised when a required optional projection dependency is missing."""


@dataclass(frozen=True)
class ProjectionManifest:
    """Shared manifest for live, materialized, and dry-run projections."""

    lake_uri: str
    source_snapshot_id: str
    snapshot_name: str
    table_versions: tuple[dict[str, Any], ...]
    format: str
    format_version: str
    mode: ProjectionMode | str
    feature_schema: dict[str, Any]
    lossiness: tuple[str, ...]
    media_policy: dict[str, Any]
    output_paths: tuple[str, ...]
    live_adapter: dict[str, Any]
    content_hashes: dict[str, str]
    transform_id: str
    transform_lineage_id: str
    accounting: dict[str, Any] = field(default_factory=dict)
    lineage_context: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the schema-level requirements shared by all projection modes."""
        required = {
            "lake_uri": self.lake_uri,
            "source_snapshot_id": self.source_snapshot_id,
            "snapshot_name": self.snapshot_name,
            "format": self.format,
            "format_version": self.format_version,
            "transform_id": self.transform_id,
            "transform_lineage_id": self.transform_lineage_id,
        }
        for name, value in required.items():
            if not value:
                raise ValueError(f"projection manifest requires {name}")

        mode = _mode(self.mode)
        if not self.table_versions:
            raise ValueError("projection manifest requires pinned table_versions")
        for version in self.table_versions:
            if not version.get("table") or version.get("version") is None:
                raise ValueError("each table version must include table and version")
        if not isinstance(self.feature_schema, dict) or not self.feature_schema:
            raise ValueError("projection manifest requires a non-empty feature_schema")
        if not isinstance(self.media_policy, dict):
            raise ValueError("projection manifest media_policy must be an object")
        if mode == ProjectionMode.LIVE and not self.live_adapter:
            raise ValueError("live projection manifest requires live_adapter identity")
        if mode == ProjectionMode.EXPORT:
            if not self.output_paths:
                raise ValueError("export projection manifest requires output_paths")
            if not self.content_hashes:
                raise ValueError("export projection manifest requires content_hashes")
        if self.accounting:
            accounting = ProjectionAccounting.from_dict(self.accounting)
            if accounting.source_snapshot_id != self.source_snapshot_id:
                raise ValueError("projection accounting source_snapshot_id must match manifest")
            if accounting.source_snapshot_name != self.snapshot_name:
                raise ValueError("projection accounting source_snapshot_name must match manifest")
            if accounting.target_format != self.format:
                raise ValueError("projection accounting target_format must match manifest")
            if accounting.projection_transform_id and (
                accounting.projection_transform_id != self.transform_id
            ):
                raise ValueError("projection accounting transform id must match manifest")
            for name in (
                "logical_row_count",
                "payload_bytes_referenced",
                "payload_bytes_copied",
                "payload_bytes_planned",
                "metadata_bytes_written",
            ):
                if int(accounting.to_dict()[name]) < 0:
                    raise ValueError(f"projection accounting {name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "lake_uri": self.lake_uri,
            "source_snapshot_id": self.source_snapshot_id,
            "snapshot_name": self.snapshot_name,
            "table_versions": [dict(version) for version in self.table_versions],
            "format": self.format,
            "format_version": self.format_version,
            "mode": _mode(self.mode).value,
            "feature_schema": self.feature_schema,
            "lossiness": list(self.lossiness),
            "media_policy": self.media_policy,
            "output_paths": list(self.output_paths),
            "live_adapter": self.live_adapter,
            "content_hashes": dict(self.content_hashes),
            "transform_id": self.transform_id,
            "transform_lineage_id": self.transform_lineage_id,
            "accounting": dict(self.accounting),
            "manifest": PROJECTION_MANIFEST_FILENAME,
        }
        if self.lineage_context:
            payload["lineage_context"] = dict(self.lineage_context)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectionManifest:
        return cls(
            lake_uri=str(payload.get("lake_uri") or ""),
            source_snapshot_id=str(payload.get("source_snapshot_id") or ""),
            snapshot_name=str(payload.get("snapshot_name") or ""),
            table_versions=tuple(dict(item) for item in payload.get("table_versions") or ()),
            format=str(payload.get("format") or ""),
            format_version=str(payload.get("format_version") or ""),
            mode=_mode(payload.get("mode") or ""),
            feature_schema=dict(payload.get("feature_schema") or {}),
            lossiness=tuple(str(item) for item in payload.get("lossiness") or ()),
            media_policy=dict(payload.get("media_policy") or {}),
            output_paths=tuple(str(item) for item in payload.get("output_paths") or ()),
            live_adapter=dict(payload.get("live_adapter") or {}),
            content_hashes={
                str(key): str(value)
                for key, value in (payload.get("content_hashes") or {}).items()
            },
            transform_id=str(payload.get("transform_id") or ""),
            transform_lineage_id=str(payload.get("transform_lineage_id") or ""),
            accounting=dict(payload.get("accounting") or {}),
            lineage_context=dict(payload.get("lineage_context") or {}),
        )


@dataclass(frozen=True)
class ProjectionSpec:
    """Registry entry for one boundary projection format."""

    name: str
    format_version: str


class ProjectionRegistry:
    """Format registry separate from the Lance-native training API."""

    def __init__(self) -> None:
        self._specs: dict[str, ProjectionSpec] = {}

    def register(self, spec: ProjectionSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ProjectionSpec:
        key = name.lower().replace("-", "_")
        spec = self._specs.get(key)
        if spec is None:
            raise ProjectionError(
                f"unsupported projection format {name!r}; choose from "
                f"{', '.join(self.formats())}"
            )
        return spec

    def formats(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))


class ProjectionLiveDataset:
    """Manifest-bearing live adapter over a Lance-native training dataset."""

    def __init__(
        self,
        *,
        format: str,
        manifest: ProjectionManifest,
        native_dataset: LanceTrainingDataset,
    ) -> None:
        self.format = format
        self.manifest = manifest
        self.native_dataset = native_dataset

    def __len__(self) -> int:
        return len(self.native_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.native_dataset[index]
        return {
            **sample,
            "_projection": {
                "format": self.manifest.format,
                "mode": self.manifest.mode.value,
                "transform_id": self.manifest.transform_id,
                "snapshot_id": self.manifest.source_snapshot_id,
            },
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]


@dataclass(frozen=True)
class LeRobotMetadata:
    """LeRobot-style metadata exposed by the live adapter."""

    info: dict[str, Any]
    tasks: tuple[dict[str, Any], ...]
    episodes: tuple[dict[str, Any], ...]
    frames: tuple[dict[str, Any], ...]
    videos: tuple[dict[str, Any], ...]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "info": self.info,
            "tasks": [dict(row) for row in self.tasks],
            "episodes": [dict(row) for row in self.episodes],
            "frames": [dict(row) for row in self.frames],
            "videos": [dict(row) for row in self.videos],
            "stats": self.stats,
        }


class LeRobotLiveDataset(ProjectionLiveDataset):
    """LeRobot-compatible live adapter over a pinned Lance snapshot.

    The adapter keeps Lance as the source of truth and projects samples into the
    common LeRobot keys used by policy code: ``observation.state``, ``action``,
    ``timestamp``, ``task``, and ``observation.images.<camera>``. Media values
    follow the selected native training media policy, so the default is a lazy
    Lance media handle rather than a repacked image/video file.
    """

    def __init__(
        self,
        *,
        manifest: ProjectionManifest,
        native_dataset: LanceTrainingDataset,
        metadata: LeRobotMetadata,
    ) -> None:
        super().__init__(
            format="lerobot",
            manifest=manifest,
            native_dataset=native_dataset,
        )
        self.repo_id = f"lancedb-robotics/{manifest.snapshot_name}"
        self.root = None
        self.meta = metadata
        self.features = metadata.info["features"]
        self.fps = metadata.info["fps"]
        self.num_episodes = metadata.info["total_episodes"]
        self.num_frames = len(native_dataset)
        self.episode_data_index = native_dataset.episode_data_index
        self.hf_dataset = None
        self._camera_keys = tuple(
            row["key"] for row in manifest.feature_schema["features"].get("cameras", ())
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        normalized = index + len(self) if index < 0 else index
        sample = self.native_dataset[index]
        return _lerobot_live_sample(self, normalized, sample)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self[index] for index in indices]


class WebDatasetLiveDataset(ProjectionLiveDataset):
    """WebDataset-shaped live iterable over a pinned Lance snapshot.

    Samples expose the standard WebDataset ``__key__`` plus a JSON metadata
    payload and, when present, one media payload/handle keyed by extension
    (``jpg``, ``png``, ``webp``, or ``bin``). The values come from the native
    row plan, so live mode keeps the Lance snapshot as source of truth instead
    of first writing tar shards.
    """

    def __init__(
        self,
        *,
        manifest: ProjectionManifest,
        native_dataset: LanceTrainingDataset,
    ) -> None:
        super().__init__(
            format="webdataset",
            manifest=manifest,
            native_dataset=native_dataset,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        normalized = index + len(self) if index < 0 else index
        sample = self.native_dataset[index]
        return _webdataset_live_sample(self, normalized, sample)


@dataclass(frozen=True)
class _RLDSEpisodeEntry:
    episode: Any
    sample_indices: tuple[int, ...]


class RLDSLiveDataset(ProjectionLiveDataset):
    """RLDS/TFDS-shaped live episode iterator over a pinned Lance snapshot.

    Iteration yields episode dictionaries with ``steps`` records using the
    standard RLDS keys: ``observation``, ``action``, ``reward``, ``discount``,
    ``is_first``, ``is_last``, and ``is_terminal``. Camera payload values follow
    the native training media policy, so the default is a lazy Lance media handle
    rather than materialized image bytes.
    """

    def __init__(
        self,
        *,
        manifest: ProjectionManifest,
        native_dataset: LanceTrainingDataset,
    ) -> None:
        super().__init__(
            format="rlds",
            manifest=manifest,
            native_dataset=native_dataset,
        )
        self.features = manifest.feature_schema["features"]
        self._episode_entries = _rlds_live_episode_entries(native_dataset)
        self.num_episodes = len(self._episode_entries)
        self.num_steps = len(native_dataset)

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, index: int) -> dict[str, Any]:
        normalized = index + len(self) if index < 0 else index
        entry = self._episode_entries[normalized]
        return _rlds_live_episode(self, normalized, entry)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def steps(self) -> Iterator[dict[str, Any]]:
        """Yield flattened RLDS steps while keeping episode iteration primary."""
        for episode in self:
            yield from episode["steps"]


class LakeProjectionBinding:
    """One ``lake.projections.<format>`` binding."""

    def __init__(self, lake: Lake, spec: ProjectionSpec) -> None:
        self._lake = lake
        self._spec = spec

    def dataset(
        self,
        snapshot_name: str,
        *,
        mode: str | ProjectionMode = ProjectionMode.LIVE,
        require_native: bool = False,
        shard_size: int = DEFAULT_WEBDATASET_SHARD_SIZE,
        compression: str = DEFAULT_WEBDATASET_COMPRESSION,
        lineage_context: Any | None = None,
        **training_options: Any,
    ) -> ProjectionLiveDataset | ProjectionManifest:
        selected = _mode(mode)
        if selected == ProjectionMode.LIVE:
            return live_projection(
                self._lake,
                snapshot_name,
                fmt=self._spec.name,
                require_native=require_native,
                lineage_context=lineage_context,
                **training_options,
            )
        if selected == ProjectionMode.PLAN:
            return plan_projection(
                self._lake,
                snapshot_name,
                fmt=self._spec.name,
                require_native=require_native,
                shard_size=shard_size,
                compression=compression,
                lineage_context=lineage_context,
            )
        raise ProjectionError(
            "dataset(..., mode='export') needs an output directory; "
            "call export(snapshot, out=...) instead"
        )

    def plan(
        self,
        snapshot_name: str,
        *,
        require_native: bool = False,
        shard_size: int = DEFAULT_WEBDATASET_SHARD_SIZE,
        compression: str = DEFAULT_WEBDATASET_COMPRESSION,
        lineage_context: Any | None = None,
    ) -> ProjectionManifest:
        return plan_projection(
            self._lake,
            snapshot_name,
            fmt=self._spec.name,
            require_native=require_native,
            shard_size=shard_size,
            compression=compression,
            lineage_context=lineage_context,
        )

    def export(
        self,
        snapshot_name: str,
        *,
        out: str | Path,
        require_native: bool = False,
        shard_size: int = DEFAULT_WEBDATASET_SHARD_SIZE,
        compression: str = DEFAULT_WEBDATASET_COMPRESSION,
        lineage_context: Any | None = None,
    ) -> ProjectionManifest:
        return export_projection(
            self._lake,
            snapshot_name,
            fmt=self._spec.name,
            out_dir=out,
            require_native=require_native,
            shard_size=shard_size,
            compression=compression,
            lineage_context=lineage_context,
        )


class LakeProjections:
    """Convenience namespace exposed as ``lake.projections``."""

    def __init__(self, lake: Lake, registry: ProjectionRegistry | None = None) -> None:
        self._lake = lake
        self._registry = registry or DEFAULT_PROJECTION_REGISTRY

    def __getattr__(self, name: str) -> LakeProjectionBinding:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            spec = self._registry.get(name)
        except ProjectionError as exc:
            raise AttributeError(str(exc)) from exc
        return LakeProjectionBinding(self._lake, spec)

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *self._registry.formats()})

    def format(self, name: str) -> LakeProjectionBinding:
        return LakeProjectionBinding(self._lake, self._registry.get(name))

    def dataset(
        self,
        format: str,
        snapshot_name: str,
        *,
        mode: str | ProjectionMode = ProjectionMode.LIVE,
        require_native: bool = False,
        shard_size: int = DEFAULT_WEBDATASET_SHARD_SIZE,
        compression: str = DEFAULT_WEBDATASET_COMPRESSION,
        lineage_context: Any | None = None,
        **training_options: Any,
    ) -> ProjectionLiveDataset | ProjectionManifest:
        return self.format(format).dataset(
            snapshot_name,
            mode=mode,
            require_native=require_native,
            shard_size=shard_size,
            compression=compression,
            lineage_context=lineage_context,
            **training_options,
        )

    def plan(
        self,
        format: str,
        snapshot_name: str,
        *,
        require_native: bool = False,
        shard_size: int = DEFAULT_WEBDATASET_SHARD_SIZE,
        compression: str = DEFAULT_WEBDATASET_COMPRESSION,
        lineage_context: Any | None = None,
    ) -> ProjectionManifest:
        return self.format(format).plan(
            snapshot_name,
            require_native=require_native,
            shard_size=shard_size,
            compression=compression,
            lineage_context=lineage_context,
        )

    def export(
        self,
        format: str,
        snapshot_name: str,
        *,
        out: str | Path,
        require_native: bool = False,
        shard_size: int = DEFAULT_WEBDATASET_SHARD_SIZE,
        compression: str = DEFAULT_WEBDATASET_COMPRESSION,
        lineage_context: Any | None = None,
    ) -> ProjectionManifest:
        return self.format(format).export(
            snapshot_name,
            out=out,
            require_native=require_native,
            shard_size=shard_size,
            compression=compression,
            lineage_context=lineage_context,
        )

    def materialization_summary(
        self,
        snapshot_name: str,
        *,
        formats: Sequence[str] | None = None,
        require_native: bool = False,
        shard_size: int = DEFAULT_WEBDATASET_SHARD_SIZE,
        compression: str = DEFAULT_WEBDATASET_COMPRESSION,
        lineage_context: Any | None = None,
    ) -> dict[str, Any]:
        names = tuple(formats or self._registry.formats())
        rows: list[dict[str, Any]] = []
        for name in names:
            manifest = self.plan(
                name,
                snapshot_name,
                require_native=require_native,
                shard_size=shard_size,
                compression=compression,
                lineage_context=lineage_context,
            )
            accounting = ProjectionAccounting.from_dict(manifest.accounting).to_dict()
            rows.append(
                {
                    "format": manifest.format,
                    "format_version": manifest.format_version,
                    "mode": manifest.mode.value,
                    "transform_id": manifest.transform_id,
                    "accounting": accounting,
                    "lossiness": list(manifest.lossiness),
                }
            )
        rows.sort(
            key=lambda row: (
                int(row["accounting"].get("payload_bytes_planned") or 0),
                int(row["accounting"].get("metadata_bytes_written") or 0),
                row["format"],
            )
        )
        return {
            "snapshot_name": snapshot_name,
            "formats": rows,
        }


DEFAULT_PROJECTION_REGISTRY = ProjectionRegistry()
DEFAULT_PROJECTION_REGISTRY.register(ProjectionSpec("lerobot", "lerobot-v3.0"))
DEFAULT_PROJECTION_REGISTRY.register(ProjectionSpec("rlds", "rlds-tfds-style-v0"))
DEFAULT_PROJECTION_REGISTRY.register(ProjectionSpec("webdataset", "webdataset-tar-v0"))


def register_projection(spec: ProjectionSpec) -> None:
    """Register a projection format for future ``lake.projections`` lookups."""
    DEFAULT_PROJECTION_REGISTRY.register(spec)


def plan_projection(
    lake: Lake,
    snapshot_name: str,
    *,
    fmt: str,
    require_native: bool = False,
    shard_size: int = DEFAULT_WEBDATASET_SHARD_SIZE,
    compression: str = DEFAULT_WEBDATASET_COMPRESSION,
    created_by: str = "lancedb-robotics",
    lineage_context: Any | None = None,
) -> ProjectionManifest:
    """Validate a projection and return a manifest without external writes."""
    spec = DEFAULT_PROJECTION_REGISTRY.get(fmt)
    native_loader = _native_loader_status(spec)
    _require_optional_dependencies(spec, native_loader, require_native=require_native)
    projection_options = _projection_options(
        spec,
        shard_size=shard_size,
        compression=compression,
    )
    handle = begin_lineage_execution(
        lineage_context,
        operation="projection-plan",
        params={"snapshot_name": snapshot_name, "format": spec.name},
    )
    context = handle.finish(status="planned")
    plan = _projection_plan(lake, snapshot_name, spec, **projection_options)
    manifest = _manifest(
        lake,
        plan,
        spec,
        mode=ProjectionMode.PLAN,
        media_policy=_media_policy(
            "plan",
            native_loader=native_loader,
            projection_options=projection_options,
        ),
        output_paths=(),
        live_adapter={},
        content_hashes={},
        payload_bytes_copied=0,
        planned_payload_bytes=plan.payload_bytes_to_copy,
        lineage_context=context,
    )
    manifest = _manifest(
        lake,
        plan,
        spec,
        mode=ProjectionMode.PLAN,
        media_policy=manifest.media_policy,
        output_paths=(),
        live_adapter={},
        content_hashes={},
        metadata_bytes=json_metadata_bytes(manifest.to_dict()),
        payload_bytes_copied=0,
        planned_payload_bytes=plan.payload_bytes_to_copy,
        lineage_context=context,
    )
    _record_projection_transform(lake, manifest, created_by=created_by, status="planned")
    _record_projection_accounting(lake, manifest, created_by=created_by)
    return manifest


def live_projection(
    lake: Lake,
    snapshot_name: str,
    *,
    fmt: str,
    require_native: bool = False,
    created_by: str = "lancedb-robotics",
    lineage_context: Any | None = None,
    **training_options: Any,
) -> ProjectionLiveDataset:
    """Return a live adapter over Lance plus a projection manifest."""
    from lancedb_robotics.training import training_dataset

    spec = DEFAULT_PROJECTION_REGISTRY.get(fmt)
    native_loader = _native_loader_status(spec)
    _require_optional_dependencies(spec, native_loader, require_native=require_native)
    handle = begin_lineage_execution(
        lineage_context,
        operation="projection-live",
        params={"snapshot_name": snapshot_name, "format": spec.name},
    )
    context = handle.finish(status="completed")
    native_dataset = training_dataset(
        lake,
        snapshot_name,
        **_training_options_for_projection(spec, training_options),
    )
    plan = _projection_plan(lake, snapshot_name, spec)
    adapter_identity = {
        "class": _live_adapter_class_name(spec),
        "native_access_pattern": native_dataset.manifest.access_pattern,
        "row_plan_id": native_dataset.manifest.row_plan_id,
        "epoch_plan_id": native_dataset.manifest.epoch_plan_id,
    }
    if spec.name == "lerobot":
        adapter_identity["protocol"] = "torch.utils.data.Dataset"
    elif spec.name == "rlds":
        adapter_identity["protocol"] = "rlds-episode-iterator"
    elif spec.name == "webdataset":
        adapter_identity["protocol"] = "webdataset-iterable"
    media_policy = _media_policy(
        "live",
        native_loader=native_loader,
        training_manifest=native_dataset.manifest.to_dict(),
    )
    manifest = _manifest(
        lake,
        plan,
        spec,
        mode=ProjectionMode.LIVE,
        media_policy=media_policy,
        output_paths=(),
        live_adapter=adapter_identity,
        content_hashes={},
        payload_bytes_copied=0,
        lineage_context=context,
    )
    manifest = _manifest(
        lake,
        plan,
        spec,
        mode=ProjectionMode.LIVE,
        media_policy=media_policy,
        output_paths=(),
        live_adapter=adapter_identity,
        content_hashes={},
        metadata_bytes=json_metadata_bytes(manifest.to_dict()),
        payload_bytes_copied=0,
        lineage_context=context,
    )
    _record_projection_transform(lake, manifest, created_by=created_by)
    _record_projection_accounting(lake, manifest, created_by=created_by)
    if spec.name == "lerobot":
        return LeRobotLiveDataset(
            manifest=manifest,
            native_dataset=native_dataset,
            metadata=_lerobot_metadata(native_dataset, manifest),
        )
    if spec.name == "rlds":
        return RLDSLiveDataset(
            manifest=manifest,
            native_dataset=native_dataset,
        )
    if spec.name == "webdataset":
        return WebDatasetLiveDataset(
            manifest=manifest,
            native_dataset=native_dataset,
        )
    return ProjectionLiveDataset(
        format=spec.name,
        manifest=manifest,
        native_dataset=native_dataset,
    )


def export_projection(
    lake: Lake,
    snapshot_name: str,
    *,
    fmt: str,
    out_dir: str | Path,
    require_native: bool = False,
    shard_size: int = DEFAULT_WEBDATASET_SHARD_SIZE,
    compression: str = DEFAULT_WEBDATASET_COMPRESSION,
    created_by: str = "lancedb-robotics",
    lineage_context: Any | None = None,
) -> ProjectionManifest:
    """Materialize a projection and write its shared projection manifest."""
    import lancedb_robotics.dataset_export as dataset_export

    spec = DEFAULT_PROJECTION_REGISTRY.get(fmt)
    native_loader = _native_loader_status(spec)
    _require_optional_dependencies(spec, native_loader, require_native=require_native)
    projection_options = _projection_options(
        spec,
        shard_size=shard_size,
        compression=compression,
    )
    handle = begin_lineage_execution(
        lineage_context,
        operation="projection-export",
        params={"snapshot_name": snapshot_name, "format": spec.name},
    )
    context = handle.finish(status="completed")

    dataset_manifest = dataset_export.export_dataset_snapshot(
        lake,
        snapshot_name,
        out_dir=out_dir,
        fmt=spec.name,
        require_native=require_native,
        **projection_options,
        created_by=created_by,
        record_materialization=False,
    )
    plan = _projection_plan(lake, snapshot_name, spec, **projection_options)
    root = Path(dataset_manifest.out_dir)
    projection_manifest_path = root / PROJECTION_MANIFEST_FILENAME
    output_paths = tuple(
        str(root / rel)
        for rel in (
            *dataset_manifest.data_files,
            dataset_export.DATASET_EXPORT_MANIFEST_FILENAME,
        )
    ) + (str(projection_manifest_path),)
    media_policy = _media_policy(
        "export",
        native_loader=dataset_manifest.native_loader,
        projection_options=projection_options,
    )
    metadata_bytes = 0
    manifest = _manifest(
        lake,
        plan,
        spec,
        mode=ProjectionMode.EXPORT,
        media_policy=media_policy,
        output_paths=output_paths,
        live_adapter={},
        content_hashes={"dataset": dataset_manifest.content_hash},
        metadata_bytes=metadata_bytes,
        payload_bytes_copied=plan.payload_bytes_to_copy,
        planned_payload_bytes=plan.payload_bytes_to_copy,
        target_path=str(root),
        lineage_context=context,
    )
    for _ in range(3):
        _write_projection_manifest(root, manifest)
        observed_metadata_bytes = metadata_bytes_written(
            output_paths,
            payload_bytes_copied=plan.payload_bytes_to_copy,
        )
        if observed_metadata_bytes == metadata_bytes:
            break
        metadata_bytes = observed_metadata_bytes
        manifest = _manifest(
            lake,
            plan,
            spec,
            mode=ProjectionMode.EXPORT,
            media_policy=media_policy,
            output_paths=output_paths,
            live_adapter={},
            content_hashes={"dataset": dataset_manifest.content_hash},
            metadata_bytes=metadata_bytes,
            payload_bytes_copied=plan.payload_bytes_to_copy,
            planned_payload_bytes=plan.payload_bytes_to_copy,
            target_path=str(root),
            lineage_context=context,
        )
    _write_projection_manifest(root, manifest)
    _record_projection_transform(lake, manifest, created_by=created_by)
    _record_projection_accounting(lake, manifest, created_by=created_by)
    return manifest


def _mode(value: str | ProjectionMode) -> ProjectionMode:
    if isinstance(value, ProjectionMode):
        return value
    try:
        return ProjectionMode(str(value).lower())
    except ValueError as exc:
        raise ProjectionError(
            f"unsupported projection mode {value!r}; choose from live, export, plan"
        ) from exc


def _training_options_for_projection(
    spec: ProjectionSpec,
    training_options: dict[str, Any],
) -> dict[str, Any]:
    options = dict(training_options)
    if spec.name not in {"lerobot", "rlds", "webdataset"}:
        return options

    requested = tuple(options.get("columns") or ())
    required = {
        "lerobot": LEROBOT_REQUIRED_COLUMNS,
        "rlds": RLDS_REQUIRED_COLUMNS,
        "webdataset": WEBDATASET_REQUIRED_COLUMNS,
    }[spec.name]
    if requested:
        columns = tuple(dict.fromkeys((*requested, *required)))
    else:
        columns = required
    options["columns"] = columns
    options.setdefault("media", "bytes" if spec.name == "webdataset" else "metadata")
    return options


def _projection_options(
    spec: ProjectionSpec,
    *,
    shard_size: int,
    compression: str,
) -> dict[str, Any]:
    if spec.name != "webdataset":
        return {}
    import lancedb_robotics.dataset_export as dataset_export

    return {
        "shard_size": dataset_export._normalize_webdataset_shard_size(shard_size),
        "compression": dataset_export._normalize_webdataset_compression(compression),
    }


def _live_adapter_class_name(spec: ProjectionSpec) -> str:
    if spec.name == "lerobot":
        return "LeRobotLiveDataset"
    if spec.name == "rlds":
        return "RLDSLiveDataset"
    if spec.name == "webdataset":
        return "WebDatasetLiveDataset"
    return "ProjectionLiveDataset"


@dataclass(frozen=True)
class _ProjectionPlan:
    dataset_id: str
    snapshot_name: str
    table_versions: tuple[dict[str, Any], ...]
    feature_schema: dict[str, Any]
    lossiness: tuple[str, ...]
    selected_scenario_count: int
    selected_observation_count: int
    logical_row_count: int
    payload_bytes_referenced: int
    payload_bytes_to_copy: int


def _projection_plan(
    lake: Lake,
    snapshot_name: str,
    spec: ProjectionSpec,
    **projection_options: Any,
) -> _ProjectionPlan:
    import lancedb_robotics.dataset_export as dataset_export

    context = dataset_export._snapshot_context(
        lake,
        snapshot_name,
        include_payload_blobs=True,
        include_video_encoding_blobs=False,
    )
    episodes = dataset_export._episodes(context)
    camera_topics = dataset_export._camera_topics(episodes)
    step_count = sum(len(episode.observations) for episode in episodes)
    selected_observations = tuple(
        obs for episode in episodes for obs in episode.observations
    )
    feature_schema, lossiness = dataset_export._feature_spec(
        context,
        episodes,
        camera_topics,
        format=spec.name,
        step_count=step_count,
        shard_size=projection_options.get("shard_size"),
        compression=projection_options.get("compression"),
    )
    return _ProjectionPlan(
        dataset_id=context.dataset_id,
        snapshot_name=context.snapshot_name,
        table_versions=context.table_versions,
        feature_schema=feature_schema,
        lossiness=tuple(lossiness),
        selected_scenario_count=len(context.scenario_ids),
        selected_observation_count=len(selected_observations),
        logical_row_count=step_count,
        payload_bytes_referenced=_payload_bytes_for_observations(
            context,
            selected_observations,
        ),
        payload_bytes_to_copy=_payload_bytes_for_observations(
            context,
            selected_observations,
            camera_only=True,
        ),
    )


def _payload_bytes_for_observations(
    context: Any,
    observations: tuple[dict[str, Any], ...],
    *,
    camera_only: bool = False,
) -> int:
    import lancedb_robotics.dataset_export as dataset_export

    total = 0
    for obs in observations:
        if camera_only and not dataset_export._is_camera_observation(obs):
            continue
        observation_id = str(obs.get("observation_id") or "")
        if observation_id in context.payload_blobs:
            total += len(context.payload_blobs.get(observation_id) or b"")
        else:
            total += payload_size(obs.get("payload_blob"))
    return total


def _lerobot_metadata(
    native_dataset: LanceTrainingDataset,
    manifest: ProjectionManifest,
) -> LeRobotMetadata:
    import lancedb_robotics.dataset_export as dataset_export

    episodes = native_dataset._episodes
    camera_topics = dataset_export._camera_topics(episodes)
    tasks = tuple(dataset_export._task_rows(episodes))
    episode_rows = tuple(_lerobot_live_episode_rows(episodes))
    frame_rows = tuple(_lerobot_live_frame_rows(native_dataset))
    video_rows = tuple(_lerobot_live_video_rows(native_dataset, camera_topics))
    info = {
        "codebase_version": manifest.format_version.removeprefix("lerobot-"),
        "format": "lerobot",
        "dataset_id": manifest.source_snapshot_id,
        "fps": int(round(native_dataset.fps or 1)),
        "features": _lerobot_features(manifest.feature_schema),
        "total_episodes": len(episodes),
        "total_frames": len(native_dataset),
        "total_tasks": len(tasks),
        "data_path": "live://lance/{row_plan_id}/{index}",
        "video_path": "live://lance/{video_key}/{episode_index}/{frame_index}",
    }
    return LeRobotMetadata(
        info=info,
        tasks=tasks,
        episodes=episode_rows,
        frames=frame_rows,
        videos=video_rows,
        stats=dataset_export._stats(episodes),
    )


def _lerobot_features(feature_schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = feature_schema.get("features", {})
    state_shape = list(features.get("observation.state", {}).get("shape") or [0])
    action_shape = list(features.get("action", {}).get("shape") or [0])
    projected: dict[str, dict[str, Any]] = {
        "index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "observation.state": {"dtype": "float32", "shape": state_shape},
        "action": {"dtype": "float32", "shape": action_shape},
        "task": {"dtype": "string", "shape": [1]},
        "language_instruction": {"dtype": "string", "shape": [1]},
    }
    for camera in features.get("cameras", ()):
        key = camera["key"]
        projected[f"observation.images.{key}"] = {
            "dtype": "image",
            "shape": [3, 0, 0],
            "source": camera.get("source", "observations.payload_blob"),
            "topic": camera.get("topic", ""),
        }
    return projected


def _lerobot_live_episode_rows(episodes: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for episode in episodes:
        length = len(episode.observations)
        rows.append(
            {
                "episode_index": episode.index,
                "episode_id": episode.episode_id,
                "scenario_id": episode.scenario["scenario_id"],
                "run_id": episode.scenario["run_id"],
                "split": episode.split,
                "tasks": [episode.task],
                "length": length,
                "dataset_from_index": offset,
                "dataset_to_index": offset + length,
                "task_index": episode.task_index,
                "task": episode.task,
                "source_table": "episodes" if episode.physical_episode else "scenarios",
            }
        )
        offset += length
    return rows


def _lerobot_live_frame_rows(native_dataset: LanceTrainingDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, ref in enumerate(native_dataset._frame_refs):
        obs = ref.observation
        rows.append(
            {
                "index": index,
                "episode_index": ref.episode.index,
                "frame_index": ref.frame_index,
                "observation_id": obs["observation_id"],
                "episode_id": obs.get("episode_id") or ref.episode.episode_id,
                "scenario_id": ref.episode.scenario["scenario_id"],
                "run_id": ref.episode.scenario["run_id"],
                "timestamp_ns": int(obs["timestamp_ns"]),
                "topic": obs.get("topic"),
                "modality": obs.get("modality"),
            }
        )
    return rows


def _lerobot_live_video_rows(
    native_dataset: LanceTrainingDataset,
    camera_topics: dict[str, str],
) -> list[dict[str, Any]]:
    context = native_dataset._context
    selected_episode_ids = {episode.episode_id for episode in native_dataset._episodes}
    selected_videos = [
        row for row in context.videos.values() if row.get("episode_id") in selected_episode_ids
    ]
    if selected_videos:
        return [
            {
                "video_id": row["video_id"],
                "episode_id": row["episode_id"],
                "episode_index": row["episode_index"],
                "camera_key": row["camera_key"],
                "topic": row.get("topic"),
                "frame_count": row.get("frame_count"),
                "uri": row.get("uri"),
                "source_table": "videos",
            }
            for row in sorted(
                selected_videos,
                key=lambda item: (
                    int(item.get("episode_index") or 0),
                    str(item.get("camera_key") or ""),
                    str(item.get("video_id") or ""),
                ),
            )
        ]

    rows: list[dict[str, Any]] = []
    for episode in native_dataset._episodes:
        for camera_key, topic in camera_topics.items():
            frame_count = sum(
                1
                for obs in episode.observations
                if _camera_key_from_sample(obs) == camera_key
            )
            if frame_count:
                rows.append(
                    {
                        "video_id": f"live-{episode.episode_id}-{camera_key}",
                        "episode_id": episode.episode_id,
                        "episode_index": episode.index,
                        "camera_key": camera_key,
                        "topic": topic,
                        "frame_count": frame_count,
                        "uri": None,
                        "source_table": "observations",
                    }
                )
    return rows


def _lerobot_live_sample(
    adapter: LeRobotLiveDataset,
    index: int,
    sample: dict[str, Any],
) -> dict[str, Any]:
    payload = sample.get("payload")
    task = sample.get("caption") or ""
    projected = {
        "index": index,
        "episode_index": sample.get("episode_index"),
        "frame_index": sample.get("frame_index"),
        "timestamp": sample.get("relative_time_s"),
        "timestamp_ns": sample.get("timestamp_ns"),
        "task_index": _task_index(adapter, sample),
        "task": task,
        "language_instruction": task,
        "observation.state": sample.get("state_vector") or [],
        "action": sample.get("action_vector") or [],
        "observation_id": sample.get("observation_id"),
        "episode_id": sample.get("episode_id"),
        "scenario_id": sample.get("scenario_id"),
        "run_id": sample.get("run_id"),
        "topic": sample.get("topic"),
        "modality": sample.get("modality"),
    }
    for camera_key in adapter._camera_keys:
        projected[f"observation.images.{camera_key}"] = None
    if payload is not None and _is_camera_sample(sample):
        projected[f"observation.images.{_camera_key_from_sample(sample)}"] = payload
    projected["_lineage"] = _projection_sample_lineage(adapter, index, sample)
    projected["_projection"] = {
        "format": adapter.manifest.format,
        "mode": adapter.manifest.mode.value,
        "transform_id": adapter.manifest.transform_id,
        "snapshot_id": adapter.manifest.source_snapshot_id,
    }
    if "_media" in sample:
        projected["_media"] = sample["_media"]
    return projected


def _webdataset_live_sample(
    adapter: WebDatasetLiveDataset,
    index: int,
    sample: dict[str, Any],
) -> dict[str, Any]:
    import lancedb_robotics.dataset_export as dataset_export

    key = dataset_export._webdataset_sample_key_from_parts(
        episode_index=int(sample.get("episode_index") or 0),
        frame_index=int(sample.get("frame_index") or 0),
        observation_id=str(sample.get("observation_id") or index),
    )
    caption = sample.get("caption") or ""
    metadata = {
        "__key__": key,
        "sample_id": sample.get("observation_id"),
        "observation_id": sample.get("observation_id"),
        "episode_id": sample.get("episode_id"),
        "scenario_id": sample.get("scenario_id"),
        "run_id": sample.get("run_id"),
        "split": sample.get("split"),
        "episode_index": sample.get("episode_index"),
        "frame_index": sample.get("frame_index"),
        "timestamp_ns": sample.get("timestamp_ns"),
        "relative_time_s": sample.get("relative_time_s"),
        "sensor_id": sample.get("sensor_id"),
        "topic": sample.get("topic"),
        "modality": sample.get("modality"),
        "state_vector": sample.get("state_vector") or [],
        "action_vector": sample.get("action_vector") or [],
        "caption": caption,
        "task": caption,
        "media": {
            "key": None,
            "camera_key": _camera_key_from_sample(sample) if _is_camera_sample(sample) else None,
            "source": "observations.payload_blob" if _is_camera_sample(sample) else None,
        },
    }
    projected: dict[str, Any] = {
        "__key__": key,
        "json": metadata,
        "txt": caption.encode(),
        "_lineage": _projection_sample_lineage(adapter, index, sample),
        "_projection": {
            "format": adapter.manifest.format,
            "mode": adapter.manifest.mode.value,
            "transform_id": adapter.manifest.transform_id,
            "snapshot_id": adapter.manifest.source_snapshot_id,
        },
    }
    payload = sample.get("payload")
    if payload is not None and _is_camera_sample(sample):
        ext = dataset_export._webdataset_media_extension(sample)
        metadata["media"]["key"] = ext
        metadata["media"]["payload_bytes"] = len(payload) if isinstance(payload, bytes) else None
        projected[ext] = payload
    if "_media" in sample:
        projected["_media"] = sample["_media"]
    return projected


def _rlds_live_episode_entries(
    native_dataset: LanceTrainingDataset,
) -> tuple[_RLDSEpisodeEntry, ...]:
    episodes_by_index = {episode.index: episode for episode in native_dataset._episodes}
    grouped: dict[int, list[tuple[int, int]]] = {}
    for sample_index, ref in enumerate(native_dataset._frame_refs):
        grouped.setdefault(ref.episode.index, []).append((sample_index, ref.frame_index))

    entries: list[_RLDSEpisodeEntry] = []
    for episode_index in sorted(grouped):
        ordered = sorted(grouped[episode_index], key=lambda item: (item[1], item[0]))
        entries.append(
            _RLDSEpisodeEntry(
                episode=episodes_by_index[episode_index],
                sample_indices=tuple(sample_index for sample_index, _ in ordered),
            )
        )
    return tuple(entries)


def _rlds_live_episode(
    adapter: RLDSLiveDataset,
    index: int,
    entry: _RLDSEpisodeEntry,
) -> dict[str, Any]:
    steps = tuple(
        _rlds_live_step(
            adapter,
            sample_index,
            step_index=step_index,
            step_count=len(entry.sample_indices),
        )
        for step_index, sample_index in enumerate(entry.sample_indices)
    )
    metadata = _rlds_episode_metadata(entry.episode, step_count=len(steps))
    return {
        "episode_id": entry.episode.episode_id,
        "episode_metadata": metadata,
        "steps": steps,
        "_lineage": _rlds_episode_lineage(adapter, index, entry),
        "_projection": {
            "format": adapter.manifest.format,
            "mode": adapter.manifest.mode.value,
            "transform_id": adapter.manifest.transform_id,
            "snapshot_id": adapter.manifest.source_snapshot_id,
        },
    }


def _rlds_episode_metadata(episode: Any, *, step_count: int) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "episode_index": episode.index,
        "scenario_id": episode.scenario["scenario_id"],
        "run_id": episode.scenario["run_id"],
        "split": episode.split,
        "task_index": episode.task_index,
        "task": episode.task,
        "length": step_count,
        "start_time_ns": int(episode.scenario["start_time_ns"]),
        "end_time_ns": int(episode.scenario["end_time_ns"]),
        "source_table": "episodes" if episode.physical_episode else "scenarios",
    }


def _rlds_live_step(
    adapter: RLDSLiveDataset,
    sample_index: int,
    *,
    step_index: int,
    step_count: int,
) -> dict[str, Any]:
    sample = adapter.native_dataset[sample_index]
    is_last = step_index == step_count - 1
    observation: dict[str, Any] = {
        "state": sample.get("state_vector") or [],
        "caption": sample.get("caption") or "",
        "payload_json": sample.get("payload_json"),
    }
    payload = sample.get("payload")
    if payload is not None and _is_camera_sample(sample):
        observation["image"] = {_camera_key_from_sample(sample): payload}

    projected = {
        "observation": observation,
        "action": sample.get("action_vector") or [],
        "reward": 0.0,
        "discount": 0.0 if is_last else 1.0,
        "is_first": step_index == 0,
        "is_last": is_last,
        "is_terminal": is_last,
        "metadata": _rlds_step_metadata(sample, step_index=step_index),
        "_lineage": _projection_sample_lineage(adapter, sample_index, sample),
    }
    if "_media" in sample:
        projected["_media"] = sample["_media"]
    return projected


def _rlds_step_metadata(sample: dict[str, Any], *, step_index: int) -> dict[str, Any]:
    return {
        "step_index": step_index,
        "observation_id": sample.get("observation_id"),
        "episode_id": sample.get("episode_id"),
        "scenario_id": sample.get("scenario_id"),
        "run_id": sample.get("run_id"),
        "split": sample.get("split"),
        "episode_index": sample.get("episode_index"),
        "frame_index": sample.get("frame_index"),
        "timestamp_ns": sample.get("timestamp_ns"),
        "relative_time_s": sample.get("relative_time_s"),
        "sensor_id": sample.get("sensor_id"),
        "topic": sample.get("topic"),
        "modality": sample.get("modality"),
        "caption": sample.get("caption") or "",
        "raw_uri": sample.get("raw_uri"),
        "raw_channel": sample.get("raw_channel"),
        "raw_sequence": sample.get("raw_sequence"),
        "message_encoding": sample.get("message_encoding"),
        "schema_encoding": sample.get("schema_encoding"),
    }


def _rlds_episode_lineage(
    adapter: RLDSLiveDataset,
    index: int,
    entry: _RLDSEpisodeEntry,
) -> dict[str, Any]:
    plan_indices = [
        int(adapter.native_dataset.epoch_plan.sample_indices[sample_index])
        for sample_index in entry.sample_indices
    ]
    return {
        "snapshot_id": adapter.manifest.source_snapshot_id,
        "table_versions": list(adapter.manifest.table_versions),
        "row_plan_id": adapter.native_dataset.row_plan.plan_id,
        "epoch_plan_id": adapter.native_dataset.epoch_plan.plan_id,
        "episode_sample_index": index,
        "episode_id": entry.episode.episode_id,
        "episode_index": entry.episode.index,
        "step_count": len(entry.sample_indices),
        "row_ids": [
            adapter.native_dataset.row_plan.row_ids[plan_index]
            for plan_index in plan_indices
        ],
        "frame_ids": [
            adapter.native_dataset.row_plan.frame_ids[plan_index]
            for plan_index in plan_indices
        ],
        "source": {
            "scenario_id": entry.episode.scenario["scenario_id"],
            "run_id": entry.episode.scenario["run_id"],
        },
    }


def _task_index(adapter: LeRobotLiveDataset, sample: dict[str, Any]) -> int:
    episode_index = sample.get("episode_index")
    if episode_index is None:
        return 0
    for episode in adapter.meta.episodes:
        if episode["episode_index"] == episode_index:
            return int(episode["task_index"])
    return 0


def _projection_sample_lineage(
    adapter: ProjectionLiveDataset,
    index: int,
    sample: dict[str, Any],
) -> dict[str, Any]:
    plan_index = int(adapter.native_dataset.epoch_plan.sample_indices[index])
    row_id = adapter.native_dataset.row_plan.row_ids[plan_index]
    frame_id = adapter.native_dataset.row_plan.frame_ids[plan_index]
    return {
        "snapshot_id": adapter.manifest.source_snapshot_id,
        "table_versions": list(adapter.manifest.table_versions),
        "row_plan_id": adapter.native_dataset.row_plan.plan_id,
        "epoch_plan_id": adapter.native_dataset.epoch_plan.plan_id,
        "sample_index": index,
        "plan_index": plan_index,
        "row_id": row_id,
        "frame_id": frame_id,
        "observation_id": sample.get("observation_id"),
        "episode_id": sample.get("episode_id"),
        "frame_index": sample.get("frame_index"),
        "source": {
            "run_id": sample.get("run_id"),
            "raw_uri": sample.get("raw_uri"),
            "raw_channel": sample.get("raw_channel"),
            "raw_sequence": sample.get("raw_sequence"),
        },
    }


def _is_camera_sample(sample: dict[str, Any]) -> bool:
    modality = str(sample.get("modality") or "").lower()
    topic = str(sample.get("topic") or "").lower()
    sensor = str(sample.get("sensor_id") or "").lower()
    return modality in {"image", "camera", "video"} or "camera" in topic or "camera" in sensor


def _camera_key_from_sample(sample: dict[str, Any]) -> str:
    import lancedb_robotics.dataset_export as dataset_export

    return dataset_export._camera_key(sample)


def _manifest(
    lake: Lake,
    plan: _ProjectionPlan,
    spec: ProjectionSpec,
    *,
    mode: ProjectionMode,
    media_policy: dict[str, Any],
    output_paths: tuple[str, ...],
    live_adapter: dict[str, Any],
    content_hashes: dict[str, str],
    metadata_bytes: int = 0,
    payload_bytes_copied: int | None = None,
    planned_payload_bytes: int | None = None,
    target_path: str = "",
    lineage_context: Any | None = None,
) -> ProjectionManifest:
    lineage_context = normalize_lineage_context(lineage_context)
    transform_id = _transform_id(
        plan,
        spec,
        mode,
        media_policy,
        content_hashes,
        lineage_context=lineage_context,
    )
    copied = (
        plan.payload_bytes_to_copy
        if payload_bytes_copied is None
        else int(payload_bytes_copied)
    )
    planned = (
        copied
        if planned_payload_bytes is None
        else int(planned_payload_bytes)
    )
    accounting = ProjectionAccounting(
        logical_row_count=plan.logical_row_count,
        selected_scenario_count=plan.selected_scenario_count,
        selected_observation_count=plan.selected_observation_count,
        payload_bytes_referenced=plan.payload_bytes_referenced,
        payload_bytes_copied=copied,
        metadata_bytes_written=int(metadata_bytes),
        target_format=spec.name,
        target_path=target_path,
        projection_transform_id=transform_id,
        source_snapshot_id=plan.dataset_id,
        source_snapshot_name=plan.snapshot_name,
        source_table_versions=plan.table_versions,
        mode=mode.value,
        payload_copy_policy=_payload_copy_policy(mode, copied, planned),
        dry_run=mode == ProjectionMode.PLAN,
        payload_bytes_planned=planned,
    ).to_dict()
    manifest = ProjectionManifest(
        lake_uri=lake.uri,
        source_snapshot_id=plan.dataset_id,
        snapshot_name=plan.snapshot_name,
        table_versions=plan.table_versions,
        format=spec.name,
        format_version=spec.format_version,
        mode=mode,
        feature_schema=plan.feature_schema,
        lossiness=plan.lossiness,
        media_policy=media_policy,
        output_paths=output_paths,
        live_adapter=live_adapter,
        content_hashes=content_hashes,
        transform_id=transform_id,
        transform_lineage_id=transform_id,
        accounting=accounting,
        lineage_context=lineage_context.to_dict() if lineage_context else {},
    )
    manifest.validate()
    return manifest


def _payload_copy_policy(mode: ProjectionMode, copied: int, planned: int = 0) -> str:
    if mode == ProjectionMode.LIVE:
        return "logical-reference"
    if mode == ProjectionMode.PLAN:
        return "would-copy-payloads" if planned else "logical-reference"
    return "materialized-copy" if copied else "metadata-only"


def _native_loader_status(spec: ProjectionSpec) -> dict[str, Any]:
    import lancedb_robotics.dataset_export as dataset_export

    return dataset_export.native_loader_status(spec.name)


def _require_optional_dependencies(
    spec: ProjectionSpec,
    native_loader: dict[str, Any],
    *,
    require_native: bool,
) -> None:
    if require_native and not native_loader["available"]:
        missing = ", ".join(native_loader["missing"]) or "unknown optional dependency"
        raise ProjectionDependencyError(
            f"{spec.name} projection requires its native loader: missing {missing}; "
            f"install {native_loader['install']}"
        )


def _media_policy(
    mode: str,
    *,
    native_loader: dict[str, Any],
    training_manifest: dict[str, Any] | None = None,
    projection_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "mode": mode,
        "native_loader": native_loader,
    }
    if projection_options:
        policy["projection_options"] = dict(projection_options)
    if training_manifest:
        policy["payloads"] = training_manifest.get("media", {}).get("policy", "metadata")
        policy["training"] = {
            "columns": training_manifest.get("columns", []),
            "filters": training_manifest.get("filters", {}),
            "shuffle": training_manifest.get("shuffle", False),
            "worker": training_manifest.get("worker", {}),
            "media": training_manifest.get("media", {}),
        }
    elif mode == "export":
        policy["payloads"] = "materialized"
    else:
        policy["payloads"] = "planned"
    return policy


def _transform_id(
    plan: _ProjectionPlan,
    spec: ProjectionSpec,
    mode: ProjectionMode,
    media_policy: dict[str, Any],
    content_hashes: dict[str, str],
    *,
    lineage_context: Any | None = None,
) -> str:
    payload = {
        "dataset_id": plan.dataset_id,
        "format": spec.name,
        "format_version": spec.format_version,
        "mode": mode.value,
        "table_versions": list(plan.table_versions),
        "media_policy": _stable_media_policy(media_policy),
        "content_hashes": content_hashes,
    }
    lineage_params = attach_lineage_context_to_params({}, lineage_context)
    if lineage_params:
        payload["lineage_context"] = lineage_params
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return f"tfm-projection-{mode.value}-{spec.name}-{digest}"


def _stable_media_policy(media_policy: dict[str, Any]) -> dict[str, Any]:
    stable = dict(media_policy)
    native_loader = dict(stable.get("native_loader") or {})
    native_loader.pop("missing", None)
    native_loader["available"] = bool(native_loader.get("available"))
    stable["native_loader"] = native_loader
    return stable


def _write_projection_manifest(root: Path, manifest: ProjectionManifest) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / PROJECTION_MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n"
    )


def _record_projection_accounting(
    lake: Lake,
    manifest: ProjectionManifest,
    *,
    created_by: str,
) -> None:
    accounting = ProjectionAccounting.from_dict(manifest.accounting)
    lake.curate.materialization_report(
        manifest.snapshot_name,
        target_format=manifest.format,
        output_uri=accounting.target_path,
        mode=manifest.mode.value,
        copied_payload_bytes=accounting.payload_bytes_copied,
        metadata_bytes_written=accounting.metadata_bytes_written,
        planned_payload_bytes=accounting.payload_bytes_planned,
        projection_transform_id=manifest.transform_id,
        created_by=created_by,
    )


def _record_projection_transform(
    lake: Lake,
    manifest: ProjectionManifest,
    *,
    created_by: str,
    status: str = "completed",
    error: str = "",
) -> None:
    now = datetime.now(UTC)
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{manifest.transform_id}'")
    transform_row = {
        "transform_id": manifest.transform_id,
        "kind": "projection",
        "source_id": manifest.source_snapshot_id,
        "input_uris": [
            f"{lake.uri}#dataset_snapshots/{manifest.source_snapshot_id}"
        ],
        "input_table_versions": list(manifest.table_versions),
        "output_tables": [],
        "params": json.dumps(
            attach_lineage_context_to_params(
                manifest.to_dict(),
                manifest.lineage_context,
            ),
            sort_keys=True,
        ),
        "status": status,
        "error": error,
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): projection transform without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
