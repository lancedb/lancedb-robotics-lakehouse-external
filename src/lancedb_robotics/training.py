"""Lance-native training access over dataset snapshots.

Proves a curated snapshot (backlog 0010 / Feature Set 7) can feed training
without repacking the corpus: :func:`load_snapshot_preview` reads the snapshot's
selected ``scenarios`` rows *as of the table versions the snapshot pinned* and
returns deterministic, framework-agnostic sample dictionaries. No new shard
layout is written — only existing rows are read at their pinned version.

PyTorch is an optional dependency. :func:`to_torch_dataset` adapts the same
samples into a ``torch.utils.data.Dataset`` when torch is importable, and raises
a friendly :class:`TrainingError` when it is not — so the dict preview always
works even in a torch-free environment.

Backlog 0039 extends the native dataset with framework-neutral batch collation
and PyTorch map/iterable adapters. Backlog 0051 adds the same PyTorch bridge for
aligned policy-tick datasets. These wrappers keep Lance as the source of truth:
they batch samples without repacking, and preserve row/tick-plan, epoch-plan,
media/feature-policy, and source-lineage metadata in every batch.
"""

import hashlib
import importlib.util
import io
import json
import random
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa

from lancedb_robotics.blob import (
    PAYLOAD_BLOB_COLUMN,
    fetch_blob,
    fetch_blobs,
    fetch_blobs_by_row_id,
)
from lancedb_robotics.dataset_export import (
    _camera_key,
    _Episode,
    _episodes,
    _infer_fps,
    _is_camera_observation,
    _relative_seconds,
    _rows_by_id_take,
    _snapshot_context,
    _SnapshotContext,
    _vector,
)
from lancedb_robotics.indexing import (
    ALIGNED_FRAME_PREDICATE_INDEX_COLUMNS,
    ALIGNED_TICK_PREDICATE_INDEX_COLUMNS,
    SCALAR_INDEX_TYPE,
    build_aligned_training_predicate_indexes,
    describe_scalar_indexes,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.materialization import (
    ProjectionAccounting,
    json_metadata_bytes,
    payload_size,
)
from lancedb_robotics.schemas import ALIGNED_TICKS_SCHEMA
from lancedb_robotics.training_plan_artifacts import (
    DEFAULT_PLAN_PAGE_SIZE,
    SERVER_SIDE_PLAN_KIND,
    InMemoryServerSidePlanStore,
    LanceTablePlanPageStore,
    ServerSidePlanArtifact,
    ServerSidePlanError,
    ServerSidePlanStore,
    build_server_side_row_plan,
    open_server_side_row_plan,
)
from lancedb_robotics.training_prewarm_jobs import (
    DEFAULT_PREWARM_JOB_TTL_S,
    ERROR_STATUSES,
    PrewarmJobCoordinator,
    PrewarmJobRunError,
    open_prewarm_job_store,
    resolve_prewarm_job_store,
    worker_label_from_request,
)
from lancedb_robotics.training_prewarm_planner import (
    DEFAULT_PREWARM_VARIABLE_WIDTH_BYTES,
    PrewarmPlan,
    PrewarmPlannerOptions,
    TableMetadata,
    build_page_cache_prewarm_plan,
    resolve_prewarm_database,
)
from lancedb_robotics.training_query_warm import (
    DEFAULT_QUERY_WARM_CHUNK_SIZE,
    QueryWarmTableSpec,
    TableIndexPrecondition,
    build_query_warm_plan,
    warm_id_column,
    warm_query_cache,
)
from lancedb_robotics.training_report_schema import (
    REPORT_REDACTION_MARKER,
    ReportValidation,
    is_secret_report_key,
    validate_training_loader_report,
)
from lancedb_robotics.video import (
    VIDEO_ENCODING_BLOB_COLUMN,
    VideoError,
    decode_frame_from_encoding,
)

# Sample fields available from a snapshot's scenario rows, and the default
# projection (scenario-level handles plus the demo embedding vector).
ALL_FIELDS = (
    "scenario_id",
    "run_id",
    "split",
    "start_time_ns",
    "end_time_ns",
    "topics",
    "summary",
    "observation_count",
    "embedding",
)
DEFAULT_PREVIEW_COLUMNS = ("scenario_id", "split", "summary", "topics", "embedding")
DEFAULT_BATCH_SIZE = 4
TORCH_INSTALL_GUIDANCE = (
    "PyTorch is not installed; install `lancedb-robotics[torch]` to use "
    "PyTorch dataloaders (the framework-neutral batch iterator does not require it)"
)

DEFAULT_TRAINING_COLUMNS = (
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
    "payload_size",
)
TRAINING_COLUMNS = (
    *DEFAULT_TRAINING_COLUMNS,
    "video_frame",
    "task",
    "task_index",
    "quality_flags",
    "raw_uri",
    "raw_channel",
    "raw_sequence",
    "message_encoding",
    "schema_encoding",
    "payload",
)
DEFAULT_SHUFFLE_SEED = 0
ROW_ID_COLUMN = "_rowid"
DEFAULT_MEDIA_POLICY = "metadata"
MEDIA_POLICIES = ("metadata", "bytes", "array", "tensor", "uri")
DEFAULT_MEDIA_CACHE_POLICY = "none"
MEDIA_CACHE_POLICIES = ("none", "bounded", "epoch")
DEFAULT_MEDIA_CACHE_SIZE = 128
DEFAULT_TRAINING_BACKEND = "auto"
TRAINING_BACKENDS = ("auto", "local", "enterprise")
EPOCH_BACKEND_PYTHON = "python"
EPOCH_BACKEND_LANCEDB_PERMUTATION = "lancedb_permutation"
EPOCH_BACKEND_DIRECT_LANCE = "direct_lance"
EPOCH_BACKEND_SERVER_SIDE_PLAN = "server_side_plan"
EPOCH_BACKEND_KIND = "lancedb-robotics/epoch-execution-backend/v1"
EPOCH_PERMUTATION_TABLE_PREFIX = "__lancedb_robotics_epoch_perm_"
DEFAULT_ENTERPRISE_FALLBACK_POLICY = "fail"
ENTERPRISE_FALLBACK_POLICIES = ("fail", "warn", "direct", "local")
ENTERPRISE_TRAINING_CONNECTION_KINDS = frozenset(
    {"lancedb_remote_db", "rest_namespace_lancedb", "namespace_lancedb"}
)
ENTERPRISE_TRAINING_CAPABILITIES = (
    "db_remote_connection",
    "remote_scan",
    "remote_take",
    "remote_filtered_read",
    "plan_executor_cache_metrics",
    "page_cache_prewarm",
    "page_cache_status",
    "namespace_direct_object_io",
    "managed_versioning",
    "blob_or_video_remote_hydration",
    "server_side_row_plan",
)
DEFAULT_ENTERPRISE_CACHE_POLICY = "none"
ENTERPRISE_CACHE_POLICIES = ("none", "lazy", "epoch", "snapshot")
ENTERPRISE_PREWARM_POLICIES = frozenset({"epoch", "snapshot"})
DEFAULT_PREWARM_MAX_ROWS = 1_000_000
DEFAULT_PREWARM_MAX_BYTES = 1 << 30
DEFAULT_PREWARM_MAX_FRAGMENTS = 10_000
DEFAULT_PREWARM_TIMEOUT_S = 300.0
DEFAULT_PREWARM_CONCURRENCY = 1
DECODED_MEDIA_POLICIES = ("array", "tensor")
TRAINING_LOADER_REPORT_KIND = "lancedb-robotics/training-loader-report/v1"
_HEAVY_MEDIA_COLUMNS = {"payload", "payload_size", "video_frame"}
_PREWARM_HEAVY_TRAINING_COLUMNS = {"payload", "video_frame"}
_PREWARM_HEAVY_SOURCE_COLUMNS = {PAYLOAD_BLOB_COLUMN, VIDEO_ENCODING_BLOB_COLUMN}


def _lake_hook(lake: Any, *names: str) -> Any:
    """Read the first attached lake hook among ``names`` (canonical name first).

    Backlog 0345 renamed the client-facing injection hooks toward the
    query-node model — the client talks to the query node, never a plan executor.
    The pre-0345 ``plan_executor_*`` attribute names are still honored so existing
    deployments that set them keep working; new code and docs use the query-node
    names (``query_node_client``, ``page_cache_prewarm``/``page_cache_prewarm_status``,
    ``query_node_cache_telemetry``).
    """
    for name in names:
        value = getattr(lake, name, None)
        if value is not None:
            return value
    return None
DEFAULT_ALIGNED_TRAINING_COLUMNS = (
    "alignment_id",
    "tick_index",
    "timestamp_ns",
    "run_id",
    "streams",
    "masks",
    "quality_flags",
    "lineage",
)
ALIGNED_TRAINING_COLUMNS = (
    *DEFAULT_ALIGNED_TRAINING_COLUMNS,
    "alignment_name",
)
_ALIGNED_FRAME_SCAN_COLUMNS = (
    "aligned_frame_id",
    "alignment_id",
    "run_id",
    "tick_index",
    "timestamp_ns",
    "stream",
    "status",
    "interpolation",
    "observation_id",
    "source_observation_ids",
    "source_row_ids",
    "source_timestamp_ns",
    "source_time_ns",
    "receive_time_ns",
    "latency_ns",
    "error_ns",
    "absolute_error_ns",
    "confidence",
    "value_json",
    "quality_flags",
    "transform_id",
)
_ALIGNED_TICK_SCAN_COLUMNS = (
    "aligned_tick_id",
    "alignment_id",
    "alignment_name",
    "recipe_digest",
    "run_id",
    "tick_index",
    "timestamp_ns",
    "available_streams",
    "missing_streams",
    "interpolated_streams",
    "out_of_tolerance_streams",
    "has_missing",
    "has_out_of_tolerance",
    "min_confidence",
    "quality_flags",
    "stream_detail_json",
    "masks_json",
    "stream_values_json",
    "lineage_json",
    "transform_id",
)
DEFAULT_ALIGNED_FEATURE_POLICY = "metadata"
ALIGNED_FEATURE_POLICIES = ("metadata", "value", "bytes", "array", "tensor", "uri")
_ALIGNED_PAYLOAD_POLICIES = {"bytes", "array", "tensor"}
ALIGNED_TICKS_STORAGE_BACKEND = "aligned_ticks-jsonb"
ALIGNED_FRAMES_STORAGE_BACKEND = "aligned_frames-pivot"
ALIGNED_TICKS_SCHEMA_VERSION = "1"
_ALIGNED_OBSERVATION_FEATURE_COLUMNS = (
    "observation_id",
    "run_id",
    "episode_id",
    "episode_index",
    "frame_index",
    "timestamp_ns",
    "sensor_id",
    "topic",
    "modality",
    "raw_uri",
    "raw_channel",
    "raw_sequence",
    "payload_json",
    "message_encoding",
    "schema_encoding",
    "decode_status",
    "state_vector",
    "action_vector",
    "quality_flags",
)

_PLAN_REQUIRED_COLUMNS = (
    "observation_id",
    "run_id",
    "episode_id",
    "episode_index",
    "frame_index",
    "timestamp_ns",
    "topic",
    "modality",
)
_OBSERVATION_COLUMNS = {
    "observation_id",
    "run_id",
    "episode_id",
    "episode_index",
    "frame_index",
    "timestamp_ns",
    "sensor_id",
    "topic",
    "modality",
    "robot_id",
    "site_id",
    "task_id",
    "software_version",
    "outcome",
    "raw_uri",
    "raw_channel",
    "raw_log_time_ns",
    "raw_sequence",
    "payload_json",
    "message_encoding",
    "schema_encoding",
    "decode_status",
    "decode_error",
    "state_vector",
    "action_vector",
    "caption",
    "quality_flags",
    "transform_id",
    "created_at",
}
_TRAINING_TO_OBSERVATION_COLUMN = {
    "observation_id": "observation_id",
    "run_id": "run_id",
    "episode_id": "episode_id",
    "episode_index": "episode_index",
    "frame_index": "frame_index",
    "timestamp_ns": "timestamp_ns",
    "sensor_id": "sensor_id",
    "topic": "topic",
    "modality": "modality",
    "state_vector": "state_vector",
    "action_vector": "action_vector",
    "caption": "caption",
    "payload_json": "payload_json",
    "quality_flags": "quality_flags",
    "raw_uri": "raw_uri",
    "raw_channel": "raw_channel",
    "raw_sequence": "raw_sequence",
    "message_encoding": "message_encoding",
    "schema_encoding": "schema_encoding",
    "task": "task_id",
}
_REMOTE_DATASET_SNAPSHOT_COLUMNS = (
    "dataset_id",
    "name",
    "kind",
    "query_spec",
    "table_versions",
    "tag",
    "split",
    "balance_report",
    "coverage_report",
    "created_by",
    "transform_id",
    "created_at",
)
_REMOTE_SCENARIO_COLUMNS = (
    "scenario_id",
    "run_id",
    "start_time_ns",
    "end_time_ns",
    "window_ns",
    "is_partial",
    "topics",
    "observation_ids",
    "observation_count",
    "scenario_type",
    "trigger_event_id",
    "source",
    "parent_scenario_id",
    "coverage_tags",
    "summary",
    "transform_id",
    "created_at",
)
_REMOTE_RUN_COLUMNS = (
    "run_id",
    "run_kind",
    "source",
    "source_id",
    "raw_uri",
    "robot_id",
    "site_id",
    "task_id",
    "start_time_ns",
    "end_time_ns",
    "duration_ns",
    "software_version",
    "hardware_version",
    "calibration_version",
    "model_version",
    "metadata",
    "quality_flags",
    "transform_id",
    "created_at",
)
_REMOTE_OBSERVATION_COLUMNS = (
    "observation_id",
    "run_id",
    "episode_id",
    "episode_index",
    "frame_index",
    "timestamp_ns",
    "sensor_id",
    "topic",
    "modality",
    "robot_id",
    "site_id",
    "task_id",
    "software_version",
    "outcome",
    "raw_uri",
    "raw_channel",
    "raw_log_time_ns",
    "raw_sequence",
    "payload_json",
    "message_encoding",
    "schema_encoding",
    "decode_status",
    "decode_error",
    "state_vector",
    "action_vector",
    "caption",
    "quality_flags",
    "transform_id",
    "created_at",
)
_REMOTE_EPISODE_COLUMNS = (
    "episode_id",
    "run_id",
    "episode_index",
    "from_timestamp_ns",
    "to_timestamp_ns",
    "boundary_source",
    "outcome",
    "frame_count",
    "camera_blobs",
    "task_id",
    "embedding",
    "provenance",
    "transform_id",
    "created_at",
)
_REMOTE_VIDEO_COLUMNS = (
    "video_id",
    "run_id",
    "episode_id",
    "episode_index",
    "camera_key",
    "sensor_id",
    "topic",
    "from_timestamp_ns",
    "to_timestamp_ns",
    "frame_count",
    "observation_ids",
    "raw_uri",
    "codec",
    "uri",
    "transform_id",
    "created_at",
)
_REMOTE_VIDEO_ENCODING_COLUMNS = (
    "encoding_id",
    "video_id",
    "run_id",
    "episode_id",
    "episode_index",
    "camera_key",
    "codec",
    "gop_size",
    "resolution",
    "fps",
    "frame_count",
    "keyframe_map_ref",
    "keyframe_map_json",
    "nvdec_compatible",
    "source_size_bytes",
    "encoded_size_bytes",
    "transform_id",
    "created_at",
)
_PUSHDOWN_FILTER_COLUMNS = {
    "observation_id": "observation_id",
    "run_id": "run_id",
    "episode_id": "episode_id",
    "episode_index": "episode_index",
    "frame_index": "frame_index",
    "timestamp_ns": "timestamp_ns",
    "sensor_id": "sensor_id",
    "topic": "topic",
    "modality": "modality",
    "raw_uri": "raw_uri",
    "raw_channel": "raw_channel",
    "raw_sequence": "raw_sequence",
}


class TrainingError(Exception):
    """Raised when a snapshot preview cannot be produced."""


class EnterpriseTrainingError(TrainingError):
    """Base class for Enterprise remote training setup failures."""


class MissingEnterpriseAuthError(EnterpriseTrainingError):
    """Raised when Enterprise training cannot find runtime auth material."""


class EnterpriseCapabilityError(EnterpriseTrainingError):
    """Raised when the resolved lake cannot satisfy a required capability."""

    def __init__(
        self,
        message: str,
        *,
        missing_capabilities: Sequence[str] = (),
        remediation: str | None = None,
    ) -> None:
        detail = message
        if missing_capabilities:
            detail += f" Missing capabilities: {', '.join(missing_capabilities)}."
        if remediation:
            detail += f" Remediation: {remediation}"
        super().__init__(detail)
        self.missing_capabilities = tuple(missing_capabilities)
        self.remediation = remediation


class UnsupportedRemoteOperationError(EnterpriseCapabilityError):
    """Raised when remote scan/take/filtered-read support is unavailable."""


class QueryNodeUnavailableError(EnterpriseCapabilityError):
    """Raised when a plan-executor-backed Enterprise path is required but absent."""


class ServerSidePlanUnavailableError(EnterpriseCapabilityError):
    """Raised when a server-side row-plan artifact is requested but unsupported."""


class PrewarmUnavailableError(EnterpriseCapabilityError):
    """Raised when page-cache prewarm is required but unsupported or timed out."""


class CacheMetricsUnavailableError(EnterpriseCapabilityError):
    """Raised when cache-metric reporting is required but unavailable."""


class RemoteQueryNodeError(EnterpriseTrainingError):
    """Raised when a live plan-executor remote scan/take/filtered-read fails.

    Carries the failing operation, table, pinned version, and the request id
    returned (or synthesized) for the request envelope, plus capability/fallback
    remediation so a caller can decide whether to retry, fall back, or abort.
    """

    def __init__(
        self,
        *,
        operation: str,
        table: str,
        version: int | None,
        request_id: str | None,
        reason: str,
        remediation: str | None = None,
    ) -> None:
        guidance = remediation or (
            "verify the plan-executor deployment is reachable and the pinned "
            "table version is still served, or set fallback='local'/'direct' to "
            "authorize a non-plan-executor read path"
        )
        version_text = "latest" if version is None else str(version)
        request_text = request_id or "unknown"
        super().__init__(
            f"live plan-executor {operation} failed for table {table!r} "
            f"(version {version_text}, request {request_text}): {reason}. "
            f"Remediation: {guidance}"
        )
        self.operation = operation
        self.table = table
        self.version = version
        self.request_id = request_id
        self.reason = reason
        self.remediation = guidance


class MetadataOnlyViolationError(EnterpriseTrainingError):
    """Raised when a metadata-only loader would read payload/blob columns.

    A metadata-only or uri media policy must never project or take heavy
    payload/blob columns through the live plan-executor client. This guardrail
    fires before any remote request is issued.
    """

    def __init__(self, *, operation: str, table: str, columns: Sequence[str]) -> None:
        blocked = ", ".join(sorted(set(columns)))
        super().__init__(
            f"metadata-only training must not {operation} payload/blob "
            f"columns from table {table!r}: {blocked}. Use media='bytes'/'array'/"
            "'tensor' (or feature policy equivalent) to authorize payload reads."
        )
        self.operation = operation
        self.table = table
        self.columns = tuple(columns)


class StaleTableVersionError(EnterpriseTrainingError):
    """Raised when a pinned snapshot/table version cannot be checked out."""

    def __init__(
        self,
        *,
        table: str,
        requested_version: int,
        snapshot_id: str | None,
        reason: str,
    ) -> None:
        snapshot_text = f" for snapshot {snapshot_id}" if snapshot_id else ""
        super().__init__(
            f"stale table version{snapshot_text}: table {table!r} requested "
            f"version {requested_version}, but checkout failed ({reason}). "
            "Remediation: refresh or recreate the training snapshot against an "
            "available managed table version before starting the loader."
        )
        self.table = table
        self.requested_version = requested_version
        self.snapshot_id = snapshot_id
        self.reason = reason


class NamespaceCredentialExpiredError(EnterpriseTrainingError):
    """Raised when namespace-vended credentials are expired before training starts."""


class WorkerResumeMismatchError(EnterpriseTrainingError):
    """Raised when worker resume metadata does not match the planned epoch."""


@dataclass(frozen=True)
class SnapshotPreview:
    """A deterministic, framework-agnostic batch previewed from a snapshot."""

    lake_uri: str
    dataset_id: str
    name: str
    tag: str
    split_by: str
    total_scenarios: int
    columns: tuple[str, ...]
    samples: list[dict]


@dataclass(frozen=True)
class TrainingBackendReport:
    """Resolved training backend and remote-execution capability report."""

    requested_backend: str
    resolved_backend: str
    execution_mode: str
    connection_kind: str
    display_uri: str
    request_routing: dict[str, Any]
    capabilities: dict[str, Any]
    plan_executor: dict[str, Any]
    cache: dict[str, Any]
    fallback_policy: str = DEFAULT_ENTERPRISE_FALLBACK_POLICY
    fallback: dict[str, Any] | None = None
    fallback_events: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "execution_mode": self.execution_mode,
            "connection_kind": self.connection_kind,
            "display_uri": self.display_uri,
            "request_routing": _jsonable(self.request_routing),
            "capabilities": _jsonable(self.capabilities),
            "plan_executor": _jsonable(self.plan_executor),
            "cache": _jsonable(self.cache),
            "fallback_policy": self.fallback_policy,
            "warnings": list(self.warnings),
            "metrics": _jsonable(self.metrics),
        }
        if self.fallback is not None:
            result["fallback"] = _jsonable(self.fallback)
        if self.fallback_events:
            result["fallback_events"] = _jsonable(list(self.fallback_events))
        return result


@dataclass(frozen=True)
class TrainingLoaderReport:
    """JSON-serializable, secret-free report for one training loader run."""

    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _redact_report(self.payload)

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)

    def write_json(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
        sort_keys: bool = True,
    ) -> None:
        Path(path).write_text(self.to_json(indent=indent, sort_keys=sort_keys) + "\n")


@dataclass(frozen=True)
class TrainingPrewarmOptions:
    """Policy knobs for Enterprise plan-executor cache prewarm requests."""

    columns: tuple[str, ...] | None = None
    include_heavy: bool = False
    max_rows: int | None = DEFAULT_PREWARM_MAX_ROWS
    max_bytes: int | None = DEFAULT_PREWARM_MAX_BYTES
    max_fragments: int | None = DEFAULT_PREWARM_MAX_FRAGMENTS
    timeout_s: float | None = DEFAULT_PREWARM_TIMEOUT_S
    concurrency: int = DEFAULT_PREWARM_CONCURRENCY
    wait: bool = False
    on_error: str = "warn"

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns) if self.columns is not None else None,
            "include_heavy": self.include_heavy,
            "max_rows": self.max_rows,
            "max_bytes": self.max_bytes,
            "max_fragments": self.max_fragments,
            "timeout_s": self.timeout_s,
            "concurrency": self.concurrency,
            "wait": self.wait,
            "on_error": self.on_error,
        }


@dataclass(frozen=True)
class LanceTrainingManifest:
    """In-memory manifest for a Lance-native training dataset view."""

    lake_uri: str
    dataset_id: str
    snapshot_name: str
    access_pattern: str
    table_versions: tuple[dict[str, Any], ...]
    columns: tuple[str, ...]
    filters: dict[str, Any]
    shuffle: bool
    shuffle_seed: int | None
    episode_count: int
    total_frames: int
    selected_frames: int
    fps: float | None
    time_windows: dict[str, tuple[float, ...]]
    row_plan_id: str
    epoch_plan_id: str
    epoch: int
    ordering_policy: str
    worker_id: int
    num_workers: int
    resume_from: int
    media_policy: str
    decoder: str
    cache_policy: str
    cache_size: int
    accounting: dict[str, Any] = field(default_factory=dict)
    epoch_backend: dict[str, Any] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=dict)
    server_side_plan: dict[str, Any] | None = None

    def to_dict(self, *, include_loader_report: bool = True) -> dict[str, Any]:
        result = {
            "lake_uri": self.lake_uri,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "access_pattern": self.access_pattern,
            "table_versions": list(self.table_versions),
            "columns": list(self.columns),
            "filters": self.filters,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "episode_count": self.episode_count,
            "total_frames": self.total_frames,
            "selected_frames": self.selected_frames,
            "fps": self.fps,
            "time_windows": {
                key: list(deltas) for key, deltas in self.time_windows.items()
            },
            "row_plan_id": self.row_plan_id,
            "epoch_plan_id": self.epoch_plan_id,
            "epoch": self.epoch,
            "ordering_policy": self.ordering_policy,
            "worker": {
                "id": self.worker_id,
                "num_workers": self.num_workers,
                "resume_from": self.resume_from,
            },
            "epoch_backend": dict(self.epoch_backend),
            "media": {
                "policy": self.media_policy,
                "decoder": self.decoder,
                "cache": {
                    "policy": self.cache_policy,
                    "max_entries": self.cache_size,
                },
            },
            "backend": dict(self.backend),
            "accounting": dict(self.accounting),
        }
        if self.server_side_plan is not None:
            result["server_side_plan"] = _jsonable(self.server_side_plan)
        if include_loader_report:
            result["loader_report"] = _native_loader_report_payload(result)
        return result


@dataclass(frozen=True)
class AlignedTrainingManifest:
    """In-memory manifest for a policy-tick training view over aligned frames."""

    lake_uri: str
    alignment_id: str
    alignment_name: str
    access_pattern: str
    storage_backend: str
    schema_version: str
    recipe_digest: str
    output_table: str
    table_versions: tuple[dict[str, Any], ...]
    read_table_versions: tuple[dict[str, Any], ...]
    streams: tuple[str, ...]
    columns: tuple[str, ...]
    quality_policy: dict[str, Any]
    total_ticks: int
    selected_ticks: int
    tick_plan_id: str
    epoch_plan_id: str
    epoch: int
    ordering_policy: str
    worker_id: int
    num_workers: int
    resume_from: int
    feature_policy: str
    decoder: str
    cache_policy: str
    cache_size: int
    predicate_indexes: tuple[dict[str, Any], ...] = ()
    epoch_backend: dict[str, Any] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_loader_report: bool = True) -> dict[str, Any]:
        result = {
            "lake_uri": self.lake_uri,
            "alignment_id": self.alignment_id,
            "alignment_name": self.alignment_name,
            "access_pattern": self.access_pattern,
            "storage_backend": self.storage_backend,
            "schema_version": self.schema_version,
            "recipe_digest": self.recipe_digest,
            "output_table": self.output_table,
            "table_versions": list(self.table_versions),
            "read_table_versions": list(self.read_table_versions),
            "streams": list(self.streams),
            "columns": list(self.columns),
            "quality_policy": _jsonable(self.quality_policy),
            "total_ticks": self.total_ticks,
            "selected_ticks": self.selected_ticks,
            "tick_plan_id": self.tick_plan_id,
            "epoch_plan_id": self.epoch_plan_id,
            "epoch": self.epoch,
            "ordering_policy": self.ordering_policy,
            "worker": {
                "id": self.worker_id,
                "num_workers": self.num_workers,
                "resume_from": self.resume_from,
            },
            "epoch_backend": dict(self.epoch_backend),
            "features": {
                "policy": self.feature_policy,
                "decoder": self.decoder,
                "cache": {
                    "policy": self.cache_policy,
                    "max_entries": self.cache_size,
                },
            },
            "predicate_indexes": [_jsonable(index) for index in self.predicate_indexes],
        }
        if self.backend:
            result["backend"] = dict(self.backend)
        if include_loader_report:
            result["loader_report"] = _aligned_loader_report_payload(result)
        return result


@dataclass(frozen=True)
class _TrainingFrameRef:
    linear_index: int
    episode: _Episode
    frame_index: int
    observation: dict[str, Any]
    row_id: int | None = None


@dataclass(frozen=True)
class TrainingRowPlan:
    """Version-pinned frame selection before epoch ordering is applied."""

    plan_id: str
    dataset_id: str
    snapshot_name: str
    table_versions: tuple[dict[str, Any], ...]
    columns: tuple[str, ...]
    filters: dict[str, Any]
    scan: dict[str, Any]
    frame_ids: tuple[str, ...]
    row_ids: tuple[int | None, ...]
    total_frames: int
    selected_frames: int
    materialization_policies: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "table_versions": list(self.table_versions),
            "columns": list(self.columns),
            "filters": _jsonable(self.filters),
            "scan": _jsonable(self.scan),
            "frame_ids": list(self.frame_ids),
            "row_ids": list(self.row_ids),
            "total_frames": self.total_frames,
            "selected_frames": self.selected_frames,
            "materialization_policies": dict(self.materialization_policies),
        }


@dataclass(frozen=True)
class AlignedTrainingTickPlan:
    """Version-pinned policy-tick selection before epoch ordering is applied."""

    plan_id: str
    alignment_id: str
    alignment_name: str
    table_versions: tuple[dict[str, Any], ...]
    read_table_versions: tuple[dict[str, Any], ...]
    storage_backend: str
    schema_version: str
    streams: tuple[str, ...]
    columns: tuple[str, ...]
    quality_policy: dict[str, Any]
    scan: dict[str, Any]
    tick_indices: tuple[int, ...]
    aligned_frame_ids: tuple[tuple[str, ...], ...]
    source_row_ids: tuple[tuple[int, ...], ...]
    row_ids: tuple[int | None, ...]
    total_ticks: int
    selected_ticks: int
    total_frames: int
    selected_frames: int
    materialization_policies: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "alignment_id": self.alignment_id,
            "alignment_name": self.alignment_name,
            "table_versions": list(self.table_versions),
            "read_table_versions": list(self.read_table_versions),
            "storage_backend": self.storage_backend,
            "schema_version": self.schema_version,
            "streams": list(self.streams),
            "columns": list(self.columns),
            "quality_policy": _jsonable(self.quality_policy),
            "scan": _jsonable(self.scan),
            "tick_indices": list(self.tick_indices),
            "aligned_frame_ids": [list(values) for values in self.aligned_frame_ids],
            "source_row_ids": [list(values) for values in self.source_row_ids],
            "row_ids": list(self.row_ids),
            "total_ticks": self.total_ticks,
            "selected_ticks": self.selected_ticks,
            "materialization_policies": dict(self.materialization_policies),
        }


@dataclass(frozen=True)
class EpochExecutionBackend:
    """Execution descriptor for an epoch order over a row/tick plan."""

    kind: str
    execution_mode: str
    row_plan_id: str
    snapshot_id: str | None
    table_versions: tuple[dict[str, Any], ...]
    shuffle_seed: int | None
    epoch: int
    worker_id: int
    num_workers: int
    resume_from: int
    permutation_table: str | None = None
    permutation_ref: str | None = None
    epoch_plan_id: str | None = None
    supported: bool = True
    selected: bool = True
    reason: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "descriptor_kind": EPOCH_BACKEND_KIND,
            "execution_mode": self.execution_mode,
            "row_plan_id": self.row_plan_id,
            "epoch_plan_id": self.epoch_plan_id,
            "snapshot_id": self.snapshot_id,
            "table_versions": list(self.table_versions),
            "shuffle_seed": self.shuffle_seed,
            "epoch": self.epoch,
            "worker": {
                "id": self.worker_id,
                "num_workers": self.num_workers,
                "resume_from": self.resume_from,
            },
            "supported": self.supported,
            "selected": self.selected,
            "capabilities": _jsonable(self.capabilities),
            "warnings": list(self.warnings),
        }
        if self.permutation_table is not None:
            result["permutation_table"] = self.permutation_table
        if self.permutation_ref is not None:
            result["permutation_ref"] = self.permutation_ref
        if self.reason is not None:
            result["reason"] = self.reason
        return result


@dataclass(frozen=True)
class EpochPlan:
    """Deterministic sample order and worker partition for one epoch."""

    plan_id: str
    row_plan_id: str
    shuffle: bool
    shuffle_seed: int | None
    epoch: int
    ordering_policy: str
    global_order: tuple[int, ...]
    worker_id: int
    num_workers: int
    worker_order: tuple[int, ...]
    resume_from: int
    sample_indices: tuple[int, ...]
    backend: EpochExecutionBackend

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "row_plan_id": self.row_plan_id,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "epoch": self.epoch,
            "ordering_policy": self.ordering_policy,
            "global_order": list(self.global_order),
            "worker": {
                "id": self.worker_id,
                "num_workers": self.num_workers,
                "order": list(self.worker_order),
                "resume_from": self.resume_from,
                "sample_indices": list(self.sample_indices),
            },
            "backend": self.backend.to_dict(),
        }


@dataclass(frozen=True)
class TrainingMediaHandle:
    """Stable lazy reference to one heavy training media field."""

    media_id: str
    field: str
    kind: str
    observation_id: str
    source_table: str
    source_column: str
    source_id: str
    row_id: int | None
    raw_uri: str | None
    raw_channel: str | None
    raw_sequence: int | None
    episode_id: str
    episode_index: int
    frame_index: int
    camera_key: str | None = None
    video_id: str | None = None
    byte_range: tuple[int, int] | None = None
    gop_index: int | None = None
    decoder: str = "auto"
    _resolver: Any = field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_id": self.media_id,
            "field": self.field,
            "kind": self.kind,
            "observation_id": self.observation_id,
            "source_table": self.source_table,
            "source_column": self.source_column,
            "source_id": self.source_id,
            "row_id": self.row_id,
            "raw_uri": self.raw_uri,
            "raw_channel": self.raw_channel,
            "raw_sequence": self.raw_sequence,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "camera_key": self.camera_key,
            "video_id": self.video_id,
            "byte_range": list(self.byte_range) if self.byte_range is not None else None,
            "gop_index": self.gop_index,
            "decoder": self.decoder,
            "lance_uri": self.lance_uri,
        }

    @property
    def lance_uri(self) -> str:
        return f"lance://{self.source_table}/{self.source_id}/{self.source_column}"

    def uri_ref(self) -> dict[str, Any]:
        return {
            "lance_uri": self.lance_uri,
            "raw_uri": self.raw_uri,
            "raw_channel": self.raw_channel,
            "raw_sequence": self.raw_sequence,
            "frame_index": self.frame_index,
        }

    def read_bytes(self) -> bytes | None:
        if self._resolver is None:
            raise TrainingError("media handle is detached from its training dataset")
        return self._resolver.read_handle_bytes(self)[0]

    def to_array(self) -> Any:
        if self._resolver is None:
            raise TrainingError("media handle is detached from its training dataset")
        return self._resolver.decode_handle_array(self)

    def to_tensor(self) -> Any:
        if self._resolver is None:
            raise TrainingError("media handle is detached from its training dataset")
        return self._resolver.decode_handle_tensor(self)


class _MediaCache:
    def __init__(self, policy: str, max_entries: int) -> None:
        self.policy = policy
        self.max_entries = max(1, int(max_entries))
        self._items: OrderedDict[tuple[Any, ...], bytes | None] = OrderedDict()

    def get(
        self,
        key: tuple[Any, ...],
        loader: Callable[[], bytes | None],
    ) -> tuple[bytes | None, bool]:
        value, cache_hit = self.get_cached(key)
        if cache_hit:
            return value, True
        value = loader()
        self.put(key, value)
        return value, False

    def get_cached(self, key: tuple[Any, ...]) -> tuple[bytes | None, bool]:
        if self.policy == "none":
            return None, False
        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value, True
        return None, False

    def put(self, key: tuple[Any, ...], value: bytes | None) -> None:
        if self.policy == "none":
            return
        self._items[key] = value
        if self.policy == "bounded":
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)


# --- 0119: live plan-executor client contract -------------------------------

_REMOTE_PAYLOAD_COLUMNS = frozenset({PAYLOAD_BLOB_COLUMN, VIDEO_ENCODING_BLOB_COLUMN})


def _is_remote_payload_column(column: str) -> bool:
    """True when ``column`` names a heavy payload/blob column a metadata-only
    loader must never read through the live plan-executor client.

    Recognizes bare (``payload_blob``, ``data``) and table-qualified
    (``video_encodings.data``) forms.
    """
    token = str(column)
    leaf = token.rsplit(".", 1)[-1]
    return token in _REMOTE_PAYLOAD_COLUMNS or leaf in _REMOTE_PAYLOAD_COLUMNS


@dataclass(frozen=True)
class QueryNodeRequest:
    """A version-pinned request envelope for one plan-executor operation.

    Every remote scan / take / filtered-read carries the table, the pinned table
    version, the manifest e-tag (when the deployment exposes one), the projected
    columns, and either the coalesced row ids or the predicate/range being read.
    The envelope is what a live client receives and what the loader records for
    lineage; it never contains auth material.
    """

    operation: str
    table: str
    version: int | None
    columns: tuple[str, ...]
    coalescing_window: str
    metadata_only: bool = False
    row_ids: tuple[int, ...] | None = None
    where_sql: str | None = None
    limit: int | None = None
    with_row_id: bool = False
    manifest_etag: str | None = None
    blob_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operation": self.operation,
            "table": self.table,
            "version": self.version,
            "columns": list(self.columns),
            "coalescing_window": self.coalescing_window,
            "metadata_only": self.metadata_only,
        }
        if self.row_ids is not None:
            payload["row_ids"] = list(self.row_ids)
        if self.where_sql is not None:
            payload["where_sql"] = self.where_sql
        if self.limit is not None:
            payload["limit"] = self.limit
        if self.with_row_id:
            payload["with_row_id"] = True
        if self.manifest_etag is not None:
            payload["manifest_etag"] = self.manifest_etag
        if self.blob_column is not None:
            payload["blob_column"] = self.blob_column
        return payload


@dataclass
class QueryNodeResponse:
    """The result of a live plan-executor request.

    ``rows`` carries dict rows for scan/filtered-read/row take (in request
    order); ``blobs`` carries raw bytes aligned to the request row ids for blob
    take. The remaining fields carry the observability metadata a plan-executor
    deployment returns: a request id, the echoed manifest e-tag, Sophon cache
    metrics (accepts ``x-cache-*`` headers or ``RequestCacheMetrics`` shapes),
    the participating plan-executor addresses, and bytes read.
    """

    rows: list[dict[str, Any]] | None = None
    blobs: list[bytes | None] | None = None
    request_id: str | None = None
    manifest_etag: str | None = None
    cache_metrics: Mapping[str, Any] | None = None
    pe_addrs: tuple[str, ...] = ()
    bytes_read: int | None = None


class QueryNodeClient(Protocol):
    """The live plan-executor client contract behind the 0071 boundary.

    A deployment attaches an object implementing :meth:`execute` to
    ``lake.plan_executor_client``. The training loader routes every remote
    operation through it; when no client is attached the loader falls back to the
    hermetic Lance-backed path used by the 0071 fake remote.
    """

    def execute(
        self, request: QueryNodeRequest
    ) -> "QueryNodeResponse | Mapping[str, Any]": ...


def _coerce_query_node_response(
    value: QueryNodeResponse | Mapping[str, Any],
) -> QueryNodeResponse:
    if isinstance(value, QueryNodeResponse):
        return value
    if isinstance(value, Mapping):
        rows = value.get("rows")
        blobs = value.get("blobs")
        pe_addrs = value.get("pe_addrs") or value.get("plan_executors") or ()
        return QueryNodeResponse(
            rows=list(rows) if rows is not None else None,
            blobs=list(blobs) if blobs is not None else None,
            request_id=value.get("request_id") or value.get("x-request-id"),
            manifest_etag=value.get("manifest_etag") or value.get("etag"),
            cache_metrics=(
                value.get("cache_metrics") or value.get("cache") or value.get("headers")
            ),
            pe_addrs=tuple(str(addr) for addr in pe_addrs),
            bytes_read=value.get("bytes_read"),
        )
    raise TypeError(
        f"plan-executor client returned an unsupported response type: {type(value)!r}"
    )


def _lance_take_rows(
    lake: Lake,
    table_name: str,
    row_ids: Sequence[int],
    *,
    columns: Sequence[str],
    version: int | None,
) -> list[dict[str, Any]]:
    """Take rows by row id at a pinned version and return them in request order."""
    table = lake.table(table_name)
    if version is not None:
        table.checkout(version)
    try:
        dataset = table.to_lance()
        rows = dataset.take(list(row_ids), columns=list(columns)).to_pylist()
    finally:
        if version is not None:
            table.checkout_latest()
    return [
        {**dict(row), ROW_ID_COLUMN: int(row_id)}
        for row_id, row in zip(row_ids, rows, strict=True)
    ]


def _lance_take_blobs(
    lake: Lake,
    table_name: str,
    blob_column: str,
    row_ids: Sequence[int],
    *,
    version: int | None,
) -> list[bytes | None]:
    """Take blob bytes by row id at a pinned version, aligned to request order."""
    table = lake.table(table_name)
    if version is not None:
        table.checkout(version)
    try:
        dataset = table.to_lance()
        blob_files = dataset.take_blobs(blob_column, ids=list(row_ids))
        return [blob_file.read() for blob_file in blob_files]
    finally:
        if version is not None:
            table.checkout_latest()


def _lance_filtered_read(
    lake: Lake,
    table_name: str,
    *,
    columns: Sequence[str],
    where_sql: str,
    version: int | None,
    with_row_id: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Bounded predicate/scan read at a pinned version."""
    table = lake.table(table_name)
    if version is not None:
        table.checkout(version)
    try:
        query = table.search().select(list(columns))
        if where_sql:
            query = query.where(where_sql)
        if with_row_id:
            query = query.with_row_id(True)
        rows: list[dict[str, Any]] = []
        for batch in query.to_batches(batch_size=4096):
            rows.extend(batch.to_pylist())
            if limit is not None and len(rows) >= limit:
                rows = rows[:limit]
                break
    finally:
        if version is not None:
            table.checkout_latest()
    return rows


class RemoteQueryNodeClient:
    """Concrete Enterprise/Sophon-backed plan-executor client.

    Dispatches remote scan / take / filtered-read for an Enterprise db:// or
    namespace-backed lake and normalizes the deployment's cache telemetry into a
    :class:`QueryNodeResponse`. Data-plane reads use the live lake's Lance
    tables (the same client that resolved the connection, so auth material is
    honored at runtime and never persisted). Cache hits/misses, request ids,
    plan-executor fanout, and bytes-read come from ``metrics_source`` — a mapping
    or callable a deployment wires to Sophon ``RequestCacheMetrics`` /
    ``x-cache-*`` response headers. Manifest e-tags are echoed from the request
    envelope so callers can confirm the pinned manifest was served.
    """

    def __init__(
        self,
        lake: Lake,
        *,
        metrics_source: Any | None = None,
        address: str = "plan-executor",
    ) -> None:
        self.lake = lake
        self.metrics_source = (
            metrics_source
            if metrics_source is not None
            else _lake_hook(lake, "query_node_cache_telemetry", "plan_executor_cache_metrics")
        )
        self.address = address
        self._seq = 0

    def execute(self, request: QueryNodeRequest) -> QueryNodeResponse:
        self._seq += 1
        fallback_request_id = f"pe-{request.operation}-{self._seq:06d}"
        telemetry = self._telemetry(request, fallback_request_id)
        if request.operation == "remote_take" and request.blob_column is not None:
            blobs = _lance_take_blobs(
                self.lake,
                request.table,
                request.blob_column,
                request.row_ids or (),
                version=request.version,
            )
            bytes_read = sum(len(value or b"") for value in blobs)
            return QueryNodeResponse(
                blobs=blobs,
                request_id=telemetry["request_id"],
                manifest_etag=request.manifest_etag,
                cache_metrics=telemetry["cache"],
                pe_addrs=telemetry["pe_addrs"],
                bytes_read=bytes_read,
            )
        if request.operation == "remote_take":
            rows = _lance_take_rows(
                self.lake,
                request.table,
                request.row_ids or (),
                columns=request.columns,
                version=request.version,
            )
            return QueryNodeResponse(
                rows=rows,
                request_id=telemetry["request_id"],
                manifest_etag=request.manifest_etag,
                cache_metrics=telemetry["cache"],
                pe_addrs=telemetry["pe_addrs"],
                bytes_read=telemetry["bytes_read"],
            )
        rows = _lance_filtered_read(
            self.lake,
            request.table,
            columns=request.columns,
            where_sql=request.where_sql or "",
            version=request.version,
            with_row_id=request.with_row_id,
            limit=request.limit,
        )
        return QueryNodeResponse(
            rows=rows,
            request_id=telemetry["request_id"],
            manifest_etag=request.manifest_etag,
            cache_metrics=telemetry["cache"],
            pe_addrs=telemetry["pe_addrs"],
            bytes_read=telemetry["bytes_read"],
        )

    def _telemetry(
        self, request: QueryNodeRequest, fallback_request_id: str
    ) -> dict[str, Any]:
        source = self.metrics_source
        cache: Mapping[str, Any] | None = None
        if callable(source):
            try:
                cache = source(
                    operation=request.operation,
                    table=request.table,
                    request=request.to_dict(),
                )
            except TypeError:
                cache = source(request.operation, request.table, request.to_dict())
        elif isinstance(source, Mapping):
            value = source.get(request.operation, source)
            cache = value if isinstance(value, Mapping) else None
        request_id = fallback_request_id
        bytes_read: int | None = None
        pe_addrs: tuple[str, ...] = (self.address,)
        if isinstance(cache, Mapping):
            request_id = str(cache.get("request_id") or fallback_request_id)
            raw_bytes = cache.get("bytes_read")
            bytes_read = int(raw_bytes) if raw_bytes is not None else None
            per = cache.get("per_addr") or cache.get("by_addr") or cache.get("by_pe")
            if isinstance(per, Mapping) and per:
                pe_addrs = tuple(str(addr) for addr in per)
        return {
            "request_id": request_id,
            "cache": cache,
            "bytes_read": bytes_read,
            "pe_addrs": pe_addrs,
        }


# --- Backlog 0345 deprecation aliases --------------------------------------
# The client is a query-node reader, not a plan-executor client. The pre-0345
# public names below remain importable for back-compat and will be removed in a
# future release; new code should use the QueryNode* names.
PlanExecutorRequest = QueryNodeRequest
PlanExecutorResponse = QueryNodeResponse
PlanExecutorClient = QueryNodeClient
RemotePlanExecutorClient = RemoteQueryNodeClient
RemotePlanExecutorError = RemoteQueryNodeError
PlanExecutorUnavailableError = QueryNodeUnavailableError


@dataclass
class _QueryNodeHydrationStats:
    requests: int = 0
    remote_scans: int = 0
    remote_takes: int = 0
    remote_filtered_reads: int = 0
    row_ids_requested: int = 0
    row_ids_unique: int = 0
    rows_returned: int = 0
    bytes_read: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    pe_fanout: int = 0
    live_requests: int = 0
    operations: list[dict[str, Any]] = field(default_factory=list)
    per_addr: dict[str, dict[str, int]] = field(default_factory=dict)
    per_request: dict[str, dict[str, int]] = field(default_factory=dict)
    request_ids: list[str] = field(default_factory=list)
    manifest_etags: dict[str, str] = field(default_factory=dict)

    def record(
        self,
        *,
        operation: str,
        table: str,
        columns: Sequence[str],
        requested_row_ids: int = 0,
        unique_row_ids: int = 0,
        rows_returned: int = 0,
        bytes_read: int = 0,
        cache_metrics: Mapping[str, Any] | None = None,
        version: int | None = None,
        coalescing_window: str | None = None,
        request_id: str | None = None,
        manifest_etag: str | None = None,
        pe_addrs: Sequence[str] = (),
        live: bool = False,
    ) -> None:
        self.requests += 1
        if live:
            self.live_requests += 1
        if operation == "remote_scan":
            self.remote_scans += 1
        elif operation == "remote_take":
            self.remote_takes += 1
        elif operation == "remote_filtered_read":
            self.remote_filtered_reads += 1
        self.row_ids_requested += int(requested_row_ids)
        self.row_ids_unique += int(unique_row_ids)
        self.rows_returned += int(rows_returned)
        self.bytes_read += int(bytes_read)

        normalized_metrics = _normalize_plan_executor_cache_metrics(cache_metrics)
        self.cache_hits += normalized_metrics["hits"]
        self.cache_misses += normalized_metrics["misses"]
        for addr, values in normalized_metrics["per_addr"].items():
            entry = self.per_addr.setdefault(addr, {"hits": 0, "misses": 0})
            entry["hits"] += int(values.get("hits") or 0)
            entry["misses"] += int(values.get("misses") or 0)
        # A live response can name the participating plan executors even when the
        # cache metrics are not broken down per address; count them for fanout.
        for addr in pe_addrs:
            self.per_addr.setdefault(str(addr), {"hits": 0, "misses": 0})
        self.pe_fanout = max(self.pe_fanout, len(self.per_addr))

        if request_id is not None:
            self.request_ids.append(request_id)
            entry = self.per_request.setdefault(
                request_id, {"hits": 0, "misses": 0}
            )
            entry["hits"] += normalized_metrics["hits"]
            entry["misses"] += normalized_metrics["misses"]
        if manifest_etag is not None:
            self.manifest_etags[table] = manifest_etag

        self.operations.append(
            {
                "operation": operation,
                "table": table,
                "version": version,
                "columns": list(columns),
                "row_ids_requested": int(requested_row_ids),
                "row_ids_unique": int(unique_row_ids),
                "rows_returned": int(rows_returned),
                "bytes_read": int(bytes_read),
                "request_id": request_id,
                "manifest_etag": manifest_etag,
                "plan_executors": list(pe_addrs),
                "live": bool(live),
                "cache": {
                    "hits": normalized_metrics["hits"],
                    "misses": normalized_metrics["misses"],
                    "per_addr": normalized_metrics["per_addr"],
                },
                "coalescing_window": coalescing_window,
            }
        )

    def to_metrics(self) -> dict[str, Any]:
        return {
            "hydration_requests": self.requests,
            "live_hydration_requests": self.live_requests,
            "remote_scan_requests": self.remote_scans,
            "remote_take_requests": self.remote_takes,
            "remote_filtered_read_requests": self.remote_filtered_reads,
            "row_ids_requested": self.row_ids_requested,
            "row_ids_unique": self.row_ids_unique,
            "row_ids_coalesced": self.row_ids_requested - self.row_ids_unique,
            "rows_returned": self.rows_returned,
            "bytes_read": self.bytes_read,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "pe_fanout": self.pe_fanout,
            "cache_by_plan_executor": dict(self.per_addr),
            "cache_by_request": dict(self.per_request),
            "request_ids": list(self.request_ids[-50:]),
            "manifest_etags": dict(self.manifest_etags),
            "operations": list(self.operations[-20:]),
        }


class _QueryNodeHydrationExecutor:
    """Remote-hydration facade for Enterprise training loaders (0071/0119).

    Builds a version-pinned request envelope (table, version, columns, coalesced
    row ids or predicate/range, manifest e-tag) for every remote scan / take /
    filtered-read. When a live plan-executor client is attached at
    ``lake.plan_executor_client`` (0119) each operation routes through it and the
    real response metadata — request id, echoed manifest e-tag, Sophon cache
    metrics, plan-executor fanout, bytes read — is normalized into loader
    metrics; a client failure is re-raised as a typed
    :class:`RemoteQueryNodeError`. With no client attached the loader falls
    back to the hermetic Lance-backed path used by the 0071 fake remote and reads
    cache metrics from ``lake.plan_executor_cache_metrics``. Metadata-only loaders
    are guarded so payload/blob columns are never taken through the client.
    """

    def __init__(
        self,
        lake: Lake,
        backend_report: TrainingBackendReport,
        *,
        table_versions: Sequence[Mapping[str, Any]],
        manifest_backend: dict[str, Any] | None = None,
        coalescing_window: str,
        metadata_only: bool = False,
    ) -> None:
        self.lake = lake
        self.backend_report = backend_report
        self.table_versions = {
            str(item["table"]): int(item["version"])
            for item in table_versions
            if item.get("table") is not None and item.get("version") is not None
        }
        self._table_etags = {
            str(item["table"]): str(
                item.get("manifest_etag")
                if item.get("manifest_etag") is not None
                else item.get("etag")
            )
            for item in table_versions
            if item.get("table") is not None
            and (item.get("manifest_etag") is not None or item.get("etag") is not None)
        }
        self.manifest_backend = manifest_backend
        self.coalescing_window = coalescing_window
        self.metadata_only = bool(metadata_only)
        self.live_client = _lake_hook(lake, "query_node_client", "plan_executor_client")
        self._etag_source = getattr(lake, "manifest_etags", None)
        self._request_seq = 0
        self.enabled = (
            backend_report.resolved_backend == "enterprise"
            and bool(backend_report.plan_executor.get("available"))
        )
        self.stats = _QueryNodeHydrationStats()

    @property
    def live_client_attached(self) -> bool:
        return self.live_client is not None

    def take_rows(
        self,
        table_name: str,
        row_ids: Sequence[int],
        *,
        columns: Sequence[str],
        version: int | None = None,
    ) -> dict[int, dict[str, Any]]:
        wanted = [int(row_id) for row_id in row_ids]
        unique_row_ids = list(dict.fromkeys(wanted))
        if not unique_row_ids:
            return {}
        self._guard_metadata_only("take", table_name, columns)
        table_version = self._version(table_name, version)
        request = self._build_request(
            "remote_take",
            table_name,
            version=table_version,
            columns=columns,
            row_ids=unique_row_ids,
        )
        response = self._execute(request)
        rows_list = response.rows or []
        result = {
            int(row_id): row
            for row_id, row in zip(unique_row_ids, rows_list, strict=True)
        }
        self._record_response(
            request,
            response,
            requested_row_ids=len(wanted),
            unique_row_ids=len(unique_row_ids),
            rows_returned=len(rows_list),
        )
        return result

    def take_blobs(
        self,
        table_name: str,
        blob_column: str,
        row_ids: Sequence[int],
        *,
        version: int | None = None,
    ) -> dict[int, bytes | None]:
        wanted = [int(row_id) for row_id in row_ids]
        unique_row_ids = list(dict.fromkeys(wanted))
        if not unique_row_ids:
            return {}
        self._guard_metadata_only("take", table_name, [blob_column])
        table_version = self._version(table_name, version)
        request = self._build_request(
            "remote_take",
            table_name,
            version=table_version,
            columns=(blob_column,),
            row_ids=unique_row_ids,
            blob_column=blob_column,
        )
        response = self._execute(request)
        blobs = response.blobs or []
        result = {
            int(row_id): value
            for row_id, value in zip(unique_row_ids, blobs, strict=True)
        }
        self._record_response(
            request,
            response,
            requested_row_ids=len(wanted),
            unique_row_ids=len(unique_row_ids),
            rows_returned=len(blobs),
        )
        return result

    def filtered_read(
        self,
        table_name: str,
        *,
        columns: Sequence[str],
        where_sql: str,
        version: int | None = None,
        with_row_id: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._guard_metadata_only("filtered-read", table_name, columns)
        table_version = self._version(table_name, version)
        operation = "remote_filtered_read" if where_sql else "remote_scan"
        request = self._build_request(
            operation,
            table_name,
            version=table_version,
            columns=columns,
            where_sql=where_sql,
            limit=limit,
            with_row_id=with_row_id,
        )
        response = self._execute(request)
        rows = response.rows or []
        self._record_response(
            request,
            response,
            requested_row_ids=0,
            unique_row_ids=0,
            rows_returned=len(rows),
        )
        return rows

    def to_metrics(self) -> dict[str, Any]:
        return self.stats.to_metrics()

    def _version(self, table_name: str, version: int | None) -> int | None:
        if version is not None:
            return int(version)
        return self.table_versions.get(table_name)

    def _manifest_etag(self, table_name: str) -> str | None:
        source = self._etag_source
        value: Any = None
        if source is not None:
            if callable(source):
                try:
                    value = source(table_name)
                except TypeError:
                    try:
                        value = source(table=table_name)
                    except TypeError:
                        value = None
            elif isinstance(source, Mapping):
                value = source.get(table_name)
            if value is not None:
                return str(value)
        embedded = self._table_etags.get(table_name)
        return str(embedded) if embedded is not None else None

    def _guard_metadata_only(
        self, operation: str, table_name: str, columns: Sequence[str]
    ) -> None:
        if not self.metadata_only:
            return
        blocked = [column for column in columns if _is_remote_payload_column(column)]
        if blocked:
            raise MetadataOnlyViolationError(
                operation=operation, table=table_name, columns=blocked
            )

    def _build_request(
        self,
        operation: str,
        table_name: str,
        *,
        version: int | None,
        columns: Sequence[str],
        row_ids: Sequence[int] | None = None,
        where_sql: str | None = None,
        limit: int | None = None,
        with_row_id: bool = False,
        blob_column: str | None = None,
    ) -> QueryNodeRequest:
        return QueryNodeRequest(
            operation=operation,
            table=table_name,
            version=version,
            columns=tuple(columns),
            coalescing_window=self.coalescing_window,
            metadata_only=self.metadata_only,
            row_ids=tuple(int(r) for r in row_ids) if row_ids is not None else None,
            where_sql=where_sql,
            limit=limit,
            with_row_id=with_row_id,
            manifest_etag=self._manifest_etag(table_name),
            blob_column=blob_column,
        )

    def _execute(self, request: QueryNodeRequest) -> QueryNodeResponse:
        self._request_seq += 1
        synthetic_id = f"loc-{request.operation}-{self._request_seq:06d}"
        if self.live_client is not None:
            try:
                raw = self.live_client.execute(request)
            except RemoteQueryNodeError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalized to a typed diagnostic
                raise RemoteQueryNodeError(
                    operation=request.operation,
                    table=request.table,
                    version=request.version,
                    request_id=None,
                    reason=str(exc) or exc.__class__.__name__,
                ) from exc
            response = _coerce_query_node_response(raw)
            if response.request_id is None:
                response.request_id = synthetic_id
            if response.manifest_etag is None:
                response.manifest_etag = request.manifest_etag
            return response
        cache = self._cache_metrics_for(
            request.operation, table=request.table, request=request.to_dict()
        )
        return self._execute_local(request, request_id=synthetic_id, cache=cache)

    def _execute_local(
        self,
        request: QueryNodeRequest,
        *,
        request_id: str,
        cache: Mapping[str, Any] | None,
    ) -> QueryNodeResponse:
        if request.operation == "remote_take" and request.blob_column is not None:
            blobs = _lance_take_blobs(
                self.lake,
                request.table,
                request.blob_column,
                request.row_ids or (),
                version=request.version,
            )
            return QueryNodeResponse(
                blobs=blobs,
                request_id=request_id,
                manifest_etag=request.manifest_etag,
                cache_metrics=cache,
                bytes_read=sum(len(value or b"") for value in blobs),
            )
        if request.operation == "remote_take":
            rows = _lance_take_rows(
                self.lake,
                request.table,
                request.row_ids or (),
                columns=request.columns,
                version=request.version,
            )
            return QueryNodeResponse(
                rows=rows,
                request_id=request_id,
                manifest_etag=request.manifest_etag,
                cache_metrics=cache,
            )
        rows = _lance_filtered_read(
            self.lake,
            request.table,
            columns=request.columns,
            where_sql=request.where_sql or "",
            version=request.version,
            with_row_id=request.with_row_id,
            limit=request.limit,
        )
        return QueryNodeResponse(
            rows=rows,
            request_id=request_id,
            manifest_etag=request.manifest_etag,
            cache_metrics=cache,
        )

    def _record_response(
        self,
        request: QueryNodeRequest,
        response: QueryNodeResponse,
        *,
        requested_row_ids: int,
        unique_row_ids: int,
        rows_returned: int,
    ) -> None:
        self.stats.record(
            operation=request.operation,
            table=request.table,
            version=request.version,
            columns=request.columns,
            requested_row_ids=requested_row_ids,
            unique_row_ids=unique_row_ids,
            rows_returned=rows_returned,
            bytes_read=int(response.bytes_read or 0),
            cache_metrics=response.cache_metrics,
            coalescing_window=self.coalescing_window,
            request_id=response.request_id,
            manifest_etag=response.manifest_etag,
            pe_addrs=response.pe_addrs,
            live=self.live_client is not None,
        )
        self._sync_manifest_metrics()

    def _cache_metrics_for(
        self,
        operation: str,
        *,
        table: str,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        source = _lake_hook(self.lake, "query_node_cache_telemetry", "plan_executor_cache_metrics")
        if source is None:
            return None
        if callable(source):
            try:
                return source(operation=operation, table=table, request=dict(request))
            except TypeError:
                return source(operation, table, dict(request))
        if isinstance(source, Mapping):
            value = source.get(operation, source)
            return value if isinstance(value, Mapping) else None
        return None

    def _sync_manifest_metrics(self) -> None:
        if self.manifest_backend is None:
            return
        metrics = self.manifest_backend.setdefault("metrics", {})
        metrics.update(self.stats.to_metrics())


def _first_present(metrics: Mapping[str, Any], *keys: str) -> Any:
    """Return the first non-None value among ``keys`` in ``metrics``.

    Accepts the loader-native (``hits``/``misses``), catalog (``cache_hits``/
    ``cache_misses``), and Sophon response-header (``x-cache-hits``/
    ``x-cache-misses``) spellings.
    """
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _normalize_plan_executor_cache_metrics(
    metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not metrics:
        return {"hits": 0, "misses": 0, "per_addr": {}}
    per_addr_raw = (
        metrics.get("per_addr") or metrics.get("by_addr") or metrics.get("by_pe") or {}
    )
    per_addr: dict[str, dict[str, int]] = {}
    if isinstance(per_addr_raw, Mapping):
        for addr, values in per_addr_raw.items():
            if not isinstance(values, Mapping):
                continue
            per_addr[str(addr)] = {
                "hits": int(
                    _first_present(values, "hits", "cache_hits", "x-cache-hits") or 0
                ),
                "misses": int(
                    _first_present(values, "misses", "cache_misses", "x-cache-misses")
                    or 0
                ),
            }
    hits = _first_present(metrics, "hits", "cache_hits", "x-cache-hits")
    misses = _first_present(metrics, "misses", "cache_misses", "x-cache-misses")
    total_hits = int(hits if hits is not None else sum(v["hits"] for v in per_addr.values()))
    total_misses = int(
        misses if misses is not None else sum(v["misses"] for v in per_addr.values())
    )
    return {"hits": total_hits, "misses": total_misses, "per_addr": per_addr}


def _prewarm_metadata_fn(lake: Lake):
    """Return an advisory table-metadata accessor for the prewarm planner.

    Backlog 0122: schema and row count are cheap metadata calls that are safe over
    both a local lake and the Enterprise query node — the only two things a client
    may talk to. The planner never reads fragments, row groups, or any plan-executor
    internals; fragment / PE fanout is decided by the query node and surfaced in the
    prewarm response, not computed here. Every access is best-effort — advisory
    estimates must never fail plan construction, so all failures degrade to ``None``.
    """

    def _metadata(table_name: str, version: int | None) -> TableMetadata | None:
        try:
            table = lake.table(table_name)
        except Exception:
            return None
        schema = None
        total_rows = None
        try:
            schema = table.schema
        except Exception:
            schema = None
        try:
            total_rows = int(table.count_rows())
        except Exception:
            total_rows = None
        return TableMetadata(schema=schema, total_rows=total_rows)

    return _metadata


def _query_warm_index_checker(lake: Lake):
    """Return an index-precondition checker for the query-warm planner (backlog 0348).

    `where(<id> IN ...)` only warms a useful index (and avoids a full scan) when the id
    column is scalar-indexed; this reuses the read-only `describe_scalar_indexes` probe
    so the plan can warn when it is not (BUG-15). Never mutates the table.
    """

    def _check(table: str, id_column: str) -> TableIndexPrecondition | None:
        try:
            results = describe_scalar_indexes(lake, table=table, columns=[id_column])
        except Exception as exc:
            return TableIndexPrecondition(
                table=table,
                id_column=id_column,
                indexed=False,
                status="unknown",
                note=f"index check failed: {exc}",
            )
        if not results:
            return None
        result = results[0]
        return TableIndexPrecondition(
            table=table,
            id_column=id_column,
            indexed=result.status == "already_present",
            status=str(result.status),
            note=str(getattr(result, "reason", None) or result.status),
        )

    return _check


class _PageCachePrewarmExecutor:
    """Build and submit Enterprise page-cache prewarm request envelopes."""

    def __init__(
        self,
        lake: Lake,
        backend_report: TrainingBackendReport,
        *,
        manifest_backend: dict[str, Any],
    ) -> None:
        self.lake = lake
        self.backend_report = backend_report
        self.manifest_backend = manifest_backend
        self._last_request: dict[str, Any] | None = None
        self._last_status: dict[str, Any] = self._cache_state().get(
            "prewarm_status_detail",
            {"status": "not-requested"},
        )
        # Backlog 0121: opt-in durable JobRun lifecycle. With no store attached the
        # prewarm path is byte-identical to the 0072 process-local envelope.
        self._job_store = resolve_prewarm_job_store(lake)
        self._coordinator: PrewarmJobCoordinator | None = None

    def maybe_prewarm_training(
        self,
        row_plan: TrainingRowPlan,
        epoch_plan: EpochPlan,
        *,
        refs: Sequence[_TrainingFrameRef],
        options: TrainingPrewarmOptions,
    ) -> dict[str, Any]:
        request = _training_prewarm_request(
            self.lake,
            self.backend_report,
            row_plan,
            epoch_plan,
            refs=refs,
            options=options,
        )
        return self._submit_or_record(request, options=options)

    def maybe_prewarm_aligned(
        self,
        tick_plan: AlignedTrainingTickPlan,
        epoch_plan: EpochPlan,
        *,
        options: TrainingPrewarmOptions,
    ) -> dict[str, Any]:
        request = _aligned_prewarm_request(
            self.lake,
            self.backend_report,
            tick_plan,
            epoch_plan,
            options=options,
        )
        return self._submit_or_record(request, options=options)

    def status(
        self,
        *,
        wait: bool = False,
        fail_on_error: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        if self._last_request is None:
            return dict(self._last_status)
        status = self._poll_status(
            self._last_request,
            wait=wait,
            timeout_s=timeout_s,
        )
        self._update_status(status)
        if fail_on_error and status.get("status") in {
            "failed",
            "unsupported",
            "timeout",
            "timed-out",
        }:
            raise PrewarmUnavailableError(
                "Enterprise cache prewarm did not complete: "
                f"{status.get('reason') or status.get('status')}",
                missing_capabilities=("page_cache_prewarm",),
                remediation="Retry with a longer timeout or use fallback='warn' for lazy-cache execution.",
            )
        return dict(status)

    def page_cache_plan(
        self,
        *,
        request: Mapping[str, Any] | None = None,
        concurrency: int | None = None,
        estimate: bool = True,
        variable_width_bytes: int = DEFAULT_PREWARM_VARIABLE_WIDTH_BYTES,
        heavy_bytes_per_row: int | None = None,
    ) -> PrewarmPlan:
        """Build a query-node page-cache prewarm plan for the current request.

        Backlog 0122: projects the 0072 prewarm request onto the real Enterprise
        ``PageCacheBeginPrewarmRequest`` shape (``{db, table, columns, table_version,
        concurrency}``) and attaches advisory cost estimates. This never contacts a
        plan executor and never emits placement / row-id / fragment routing; fragment
        and PE fanout are the query node's concern.
        """
        req = request if request is not None else self._last_request
        if not req or not req.get("requested"):
            return PrewarmPlan(
                prewarm_id=str((req or {}).get("prewarm_id") or ""),
                scope=str((req or {}).get("scope") or ""),
                policy=str((req or {}).get("policy") or ""),
                database=None,
                applicable=False,
                reason=(req or {}).get("reason")
                or "cache policy does not request prewarm",
                tables=(),
            )
        database = resolve_prewarm_database(
            uri=self.lake.uri,
            connection_kind=self.backend_report.connection_kind,
            namespace_properties=getattr(
                getattr(self.lake, "connection_spec", None),
                "namespace_client_properties",
                None,
            ),
        )
        not_applicable_reason = None
        if self.backend_report.resolved_backend != "enterprise":
            not_applicable_reason = (
                "page-cache prewarm applies only to the Enterprise query-node backend"
            )
        options = PrewarmPlannerOptions(
            concurrency=concurrency,
            variable_width_bytes=variable_width_bytes,
            heavy_bytes_per_row=heavy_bytes_per_row,
            estimate=estimate,
        )
        return build_page_cache_prewarm_plan(
            req,
            database=database,
            heavy_columns=tuple(_PREWARM_HEAVY_SOURCE_COLUMNS),
            metadata_fn=_prewarm_metadata_fn(self.lake) if estimate else None,
            options=options,
            not_applicable_reason=not_applicable_reason,
        )

    def _submit_or_record(
        self,
        request: dict[str, Any],
        *,
        options: TrainingPrewarmOptions,
    ) -> dict[str, Any]:
        self._last_request = request
        cache_state = self._cache_state()
        cache_state["prewarm_requested"] = bool(request["requested"])
        cache_state["prewarm_id"] = request.get("prewarm_id")
        cache_state["prewarm_requests"] = [request] if request["requested"] else []
        cache_state["prewarm_limits"] = request.get("limits", {})
        cache_state["prewarm_executed"] = False
        cache_state["prewarm_status"] = "not-requested"
        cache_state["prewarm_status_detail"] = {"status": "not-requested"}
        self._sync_prewarm_metrics(request, {"status": "not-requested"})

        if not request["requested"]:
            status = {"status": "not-requested", "reason": request.get("reason")}
            self._update_status(status)
            return status
        if request.get("status") == "skipped":
            status = {
                "status": "skipped",
                "reason": request.get("skip_reason"),
                "prewarm_id": request.get("prewarm_id"),
            }
            self._update_status(status)
            return status
        if self.backend_report.resolved_backend != "enterprise":
            status = {
                "status": "not-applicable",
                "reason": "prewarm only executes for Enterprise training backends",
                "prewarm_id": request.get("prewarm_id"),
            }
            self._update_status(status)
            return status
        if not self.backend_report.plan_executor.get("prewarm_supported"):
            status = {
                "status": "unsupported",
                "reason": "plan-executor prewarm capability is not available",
                "prewarm_id": request.get("prewarm_id"),
            }
            self._handle_error_status(status, options)
            return status

        if self._job_store is not None:
            return self._submit_via_jobrun(request, options=options)

        source = _lake_hook(self.lake, "page_cache_prewarm", "plan_executor_prewarm")
        if source is None:
            status = {
                "status": "planned",
                "reason": "no live plan-executor prewarm client is attached",
                "prewarm_id": request.get("prewarm_id"),
            }
            _append_backend_warning(
                self.manifest_backend,
                "Enterprise cache prewarm request was planned but not submitted; "
                "attach lake.plan_executor_prewarm or use the live Enterprise "
                "client follow-up path.",
            )
            self._update_status(status)
            return status

        started = time.perf_counter()
        try:
            response = self._call_submit(source, request)
        except Exception as exc:  # pragma: no cover - exercised by tests with hooks.
            status = {
                "status": "failed",
                "reason": str(exc),
                "prewarm_id": request.get("prewarm_id"),
            }
            self._handle_error_status(status, options)
            return status
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        status = _normalize_prewarm_status(
            response,
            request=request,
            default_status="submitted",
            submit_duration_ms=elapsed_ms,
        )
        self._cache_state()["prewarm_executed"] = True
        self._update_status(status)
        if options.wait:
            status = self.status(
                wait=True,
                fail_on_error=options.on_error == "raise",
                timeout_s=options.timeout_s,
            )
        return status

    def _build_coordinator(self) -> PrewarmJobCoordinator:
        source = _lake_hook(self.lake, "page_cache_prewarm", "plan_executor_prewarm")
        status_source = _lake_hook(self.lake, "page_cache_prewarm_status", "plan_executor_prewarm_status")
        submit_fn = (
            (lambda req: self._call_submit(source, req)) if source is not None else None
        )
        status_fn = (
            (lambda **kw: self._call_status(status_source, **kw))
            if status_source is not None
            else None
        )
        kwargs: dict[str, Any] = {
            "submit_fn": submit_fn,
            "status_fn": status_fn,
            "ttl_s": getattr(self.lake, "prewarm_job_ttl_s", DEFAULT_PREWARM_JOB_TTL_S),
        }
        clock = getattr(self.lake, "prewarm_clock", None)
        if callable(clock):
            kwargs["now_fn"] = clock
        return PrewarmJobCoordinator(self._job_store, **kwargs)

    def _job_labels(self, request: Mapping[str, Any]) -> dict[str, Any]:
        routing = request.get("routing") or {}
        job = (
            getattr(self.lake, "training_job_label", None)
            or routing.get("job")
            or request.get("dataset_id")
            or request.get("alignment_id")
            or "default"
        )
        return {
            "job": str(job),
            "caller": getattr(self.lake, "training_caller_label", None),
        }

    def _submit_via_jobrun(
        self,
        request: dict[str, Any],
        *,
        options: TrainingPrewarmOptions,
    ) -> dict[str, Any]:
        coordinator = self._build_coordinator()
        self._coordinator = coordinator
        labels = self._job_labels(request)
        result = coordinator.submit_or_attach(
            request,
            job_label=labels["job"],
            caller_label=labels["caller"],
            worker_label=worker_label_from_request(request),
            wait=False,
        )
        status = result.status
        source = _lake_hook(self.lake, "page_cache_prewarm", "plan_executor_prewarm")
        if source is None:
            _append_backend_warning(
                self.manifest_backend,
                "Enterprise cache prewarm request was recorded as a durable JobRun but "
                "not submitted; attach lake.plan_executor_prewarm or use the live "
                "Enterprise client follow-up path.",
            )
            self._update_status(status)
            return status
        if status.get("status") in {"active", "complete"}:
            self._cache_state()["prewarm_executed"] = True
        self._update_status(status)
        if status.get("status") in ERROR_STATUSES:
            self._handle_error_status(status, options)
            return dict(self._last_status)
        if options.wait:
            status = self.status(
                wait=True,
                fail_on_error=options.on_error == "raise",
                timeout_s=options.timeout_s,
            )
        return status

    def _poll_status(
        self,
        request: Mapping[str, Any],
        *,
        wait: bool,
        timeout_s: float | None,
    ) -> dict[str, Any]:
        if self._job_store is not None and self._coordinator is not None:
            record = self._coordinator.poll(
                str(request["prewarm_id"]), request, wait=wait, timeout_s=timeout_s
            )
            return record.status_dict()
        source = _lake_hook(self.lake, "page_cache_prewarm_status", "plan_executor_prewarm_status")
        if source is None:
            return dict(self._last_status)
        started = time.perf_counter()
        try:
            response = self._call_status(
                source,
                prewarm_id=str(request["prewarm_id"]),
                request=request,
                wait=wait,
                timeout_s=timeout_s,
            )
        except Exception as exc:  # pragma: no cover - exercised by tests with hooks.
            return {
                "status": "failed",
                "reason": str(exc),
                "prewarm_id": request.get("prewarm_id"),
            }
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return _normalize_prewarm_status(
            response,
            request=request,
            default_status=self._last_status.get("status", "submitted"),
            status_duration_ms=elapsed_ms,
        )

    def _call_submit(self, source: Any, request: Mapping[str, Any]) -> Any:
        if callable(source):
            try:
                return source(request=dict(request))
            except TypeError:
                return source(dict(request))
        if isinstance(source, list):
            source.append(dict(request))
            return {"status": "submitted"}
        if isinstance(source, Mapping):
            return source
        return None

    def _call_status(
        self,
        source: Any,
        *,
        prewarm_id: str,
        request: Mapping[str, Any],
        wait: bool,
        timeout_s: float | None,
    ) -> Any:
        if callable(source):
            try:
                return source(
                    prewarm_id=prewarm_id,
                    request=dict(request),
                    wait=wait,
                    timeout_s=timeout_s,
                )
            except TypeError:
                try:
                    return source(prewarm_id, dict(request), wait, timeout_s)
                except TypeError:
                    return source(prewarm_id)
        if isinstance(source, Mapping):
            return source.get(prewarm_id, source)
        return None

    def _cache_state(self) -> dict[str, Any]:
        return self.manifest_backend.setdefault("cache", {})

    def _update_status(self, status: Mapping[str, Any]) -> None:
        normalized = dict(status)
        self._last_status = normalized
        cache_state = self._cache_state()
        cache_state["prewarm_status"] = str(normalized.get("status", "unknown"))
        cache_state["prewarm_status_detail"] = _jsonable(normalized)
        self._sync_prewarm_metrics(self._last_request or {}, normalized)

    def _handle_error_status(
        self,
        status: dict[str, Any],
        options: TrainingPrewarmOptions,
    ) -> None:
        self._update_status(status)
        if options.on_error == "raise":
            raise PrewarmUnavailableError(
                "Enterprise cache prewarm failed: "
                f"{status.get('reason') or status.get('status')}",
                missing_capabilities=("page_cache_prewarm",),
                remediation="Use prewarm_options={'on_error': 'warn'} or fallback='warn' when lazy-cache execution is acceptable.",
            )
        _append_backend_warning(
            self.manifest_backend,
            "Enterprise cache prewarm did not complete: "
            f"{status.get('reason') or status.get('status')}",
        )

    def _sync_prewarm_metrics(
        self,
        request: Mapping[str, Any],
        status: Mapping[str, Any],
    ) -> None:
        metrics = self.manifest_backend.setdefault("metrics", {})
        table_requests = request.get("tables") if isinstance(request, Mapping) else None
        metrics["prewarm_requests"] = len(table_requests or [])
        metrics["prewarm_policy"] = request.get("policy")
        metrics["prewarm_row_count"] = request.get("row_count")
        metrics["prewarm_projected_columns"] = request.get("projected_columns")
        metrics["prewarm_status"] = status.get("status")
        for source_key, target_key in (
            ("pe_fanout", "prewarm_pe_fanout"),
            ("completed_executors", "prewarm_completed_executors"),
            ("failed_executors", "prewarm_failed_executors"),
            ("cache_hits", "prewarm_cache_hits"),
            ("cache_misses", "prewarm_cache_misses"),
            ("warm_bytes", "prewarm_warm_bytes"),
            ("cold_bytes", "prewarm_cold_bytes"),
            ("duration_ms", "prewarm_duration_ms"),
            ("submit_duration_ms", "prewarm_submit_duration_ms"),
            ("status_duration_ms", "prewarm_status_duration_ms"),
        ):
            if status.get(source_key) is not None:
                metrics[target_key] = status[source_key]


def _epoch_scope_id(epoch_plan: EpochPlan) -> str:
    """Worker-invariant id for an epoch's prewarm scope.

    Backlog 0121: the epoch prewarm target is the whole epoch, identical for every
    worker. This digest deliberately excludes ``worker_id``/``num_workers``/
    ``resume_from`` (which ``epoch_plan.plan_id`` includes) so all workers opening
    the same snapshot/seed/epoch dedup to one JobRun.
    """
    return "epoch-scope-" + _stable_digest(
        {
            "row_plan_id": epoch_plan.row_plan_id,
            "shuffle": epoch_plan.shuffle,
            "shuffle_seed": epoch_plan.shuffle_seed,
            "epoch": epoch_plan.epoch,
            "ordering_policy": epoch_plan.ordering_policy,
            "global_order": list(epoch_plan.global_order),
        }
    )


def _training_prewarm_request(
    lake: Lake,
    backend_report: TrainingBackendReport,
    row_plan: TrainingRowPlan,
    epoch_plan: EpochPlan,
    *,
    refs: Sequence[_TrainingFrameRef],
    options: TrainingPrewarmOptions,
) -> dict[str, Any]:
    policy = str(backend_report.cache.get("policy", DEFAULT_ENTERPRISE_CACHE_POLICY))
    requested = policy in ENTERPRISE_PREWARM_POLICIES
    if not requested:
        return {
            "kind": "lancedb-robotics/training-prewarm/v1",
            "requested": False,
            "policy": policy,
            "reason": f"cache_policy={policy!r} does not request prewarm",
        }

    logical_columns = tuple(options.columns or row_plan.columns)
    source_columns, skipped_columns = _native_prewarm_source_columns(
        logical_columns,
        include_heavy=options.include_heavy,
    )
    row_ids = tuple(int(ref.row_id) for ref in refs if ref.row_id is not None)
    table_version = _version_from_table_versions(row_plan.table_versions, "observations")
    estimated_bytes = (
        sum(payload_size(ref.observation.get("payload_blob")) for ref in refs)
        if PAYLOAD_BLOB_COLUMN in source_columns
        else 0
    )
    scope = "snapshot" if policy == "snapshot" else "epoch"
    # Backlog 0121: the epoch/snapshot prewarm target is the whole epoch, not a
    # worker shard. Key the id on a worker-invariant epoch scope id (not the
    # per-worker ``epoch_plan.plan_id``) and the epoch-global row set, so every
    # worker opening the same snapshot/epoch collapses to one JobRun.
    epoch_scope_id = _epoch_scope_id(epoch_plan) if policy == "epoch" else None
    prewarm_id = "prewarm-" + _stable_digest(
        {
            "policy": policy,
            "scope": scope,
            "row_plan_id": row_plan.plan_id,
            "epoch_scope_id": epoch_scope_id,
            "columns": source_columns,
            "row_ranges": _row_id_ranges(row_ids),
            "table_version": table_version,
        }
    )
    request = {
        "kind": "lancedb-robotics/training-prewarm/v1",
        "requested": True,
        "prewarm_id": prewarm_id,
        "policy": policy,
        "scope": scope,
        "dataset_id": row_plan.dataset_id,
        "snapshot_name": row_plan.snapshot_name,
        "row_plan_id": row_plan.plan_id,
        "epoch_plan_id": epoch_plan.plan_id,
        "epoch_scope_id": epoch_scope_id,
        "worker": {
            "id": epoch_plan.worker_id,
            "num_workers": epoch_plan.num_workers,
            "sample_count": len(epoch_plan.sample_indices),
        },
        "routing": dict(backend_report.request_routing),
        "table_uri": lake.uri,
        "tables": [
            {
                "table": "observations",
                "uri": lake.uri,
                "version": table_version,
                "label": _prewarm_table_label(
                    backend_report.display_uri,
                    "observations",
                    table_version,
                ),
                "projected_columns": list(source_columns),
                "logical_columns": list(logical_columns),
                "row_id_ranges": _row_id_ranges(row_ids),
                "row_count": len(row_ids),
                "fragments": [],
                "row_groups": [],
                "placement": {
                    "mode": "query-node-routed",
                    "fragment_ranges_available": False,
                    "row_group_ranges_available": False,
                },
            }
        ],
        "projected_columns": list(source_columns),
        "logical_columns": list(logical_columns),
        "excluded_columns": skipped_columns,
        "row_count": len(row_ids),
        "estimated_bytes": estimated_bytes,
        "limits": _prewarm_limits_dict(options),
        "concurrency": options.concurrency,
        "timeout_s": options.timeout_s,
        "include_heavy": options.include_heavy,
    }
    return _apply_prewarm_limits(request, options)


def _aligned_prewarm_request(
    lake: Lake,
    backend_report: TrainingBackendReport,
    tick_plan: AlignedTrainingTickPlan,
    epoch_plan: EpochPlan,
    *,
    options: TrainingPrewarmOptions,
) -> dict[str, Any]:
    policy = str(backend_report.cache.get("policy", DEFAULT_ENTERPRISE_CACHE_POLICY))
    requested = policy in ENTERPRISE_PREWARM_POLICIES
    if not requested:
        return {
            "kind": "lancedb-robotics/training-prewarm/v1",
            "requested": False,
            "policy": policy,
            "reason": f"cache_policy={policy!r} does not request prewarm",
        }

    logical_columns = tuple(options.columns or tick_plan.columns)
    source_columns, skipped_columns = _aligned_prewarm_source_columns(
        logical_columns,
        feature_policy=tick_plan.materialization_policies.get("payload"),
        storage_backend=tick_plan.storage_backend,
        include_heavy=options.include_heavy,
    )
    # Backlog 0121: epoch prewarm warms the whole epoch (worker-invariant), so use
    # the epoch-global order rather than this worker's ``sample_indices`` shard.
    plan_indices = (
        tuple(epoch_plan.global_order)
        if policy == "epoch"
        else tuple(range(len(tick_plan.tick_indices)))
    )
    aligned_row_ids = tuple(
        int(tick_plan.row_ids[index])
        for index in plan_indices
        if tick_plan.row_ids[index] is not None
    )
    source_row_ids = tuple(
        int(row_id)
        for index in plan_indices
        for row_id in tick_plan.source_row_ids[index]
    )
    aligned_table = str(tick_plan.scan.get("table") or "aligned_frames")
    aligned_version = _version_from_table_versions(tick_plan.read_table_versions, aligned_table)
    observation_version = _version_from_table_versions(
        tick_plan.table_versions,
        "observations",
    )
    tables = [
        {
            "table": aligned_table,
            "uri": lake.uri,
            "version": aligned_version,
            "label": _prewarm_table_label(
                backend_report.display_uri,
                aligned_table,
                aligned_version,
            ),
            "projected_columns": list(source_columns[aligned_table]),
            "logical_columns": list(logical_columns),
            "row_id_ranges": _row_id_ranges(aligned_row_ids),
            "row_count": len(aligned_row_ids),
            "fragments": [],
            "row_groups": [],
            "placement": {
                "mode": "query-node-routed",
                "fragment_ranges_available": False,
                "row_group_ranges_available": False,
            },
        }
    ]
    if source_columns["observations"]:
        tables.append(
            {
                "table": "observations",
                "uri": lake.uri,
                "version": observation_version,
                "label": _prewarm_table_label(
                    backend_report.display_uri,
                    "observations",
                    observation_version,
                ),
                "projected_columns": list(source_columns["observations"]),
                "logical_columns": list(logical_columns),
                "row_id_ranges": _row_id_ranges(source_row_ids),
                "row_count": len(source_row_ids),
                "fragments": [],
                "row_groups": [],
                "placement": {
                    "mode": "query-node-routed",
                    "fragment_ranges_available": False,
                    "row_group_ranges_available": False,
                },
            }
        )
    scope = "snapshot" if policy == "snapshot" else "epoch"
    epoch_scope_id = _epoch_scope_id(epoch_plan) if policy == "epoch" else None
    prewarm_id = "prewarm-" + _stable_digest(
        {
            "policy": policy,
            "scope": scope,
            "tick_plan_id": tick_plan.plan_id,
            "epoch_scope_id": epoch_scope_id,
            "tables": tables,
        }
    )
    projected_columns = tuple(
        dict.fromkeys(
            column
            for table in tables
            for column in table["projected_columns"]
        )
    )
    request = {
        "kind": "lancedb-robotics/training-prewarm/v1",
        "requested": True,
        "prewarm_id": prewarm_id,
        "policy": policy,
        "scope": scope,
        "alignment_id": tick_plan.alignment_id,
        "alignment_name": tick_plan.alignment_name,
        "tick_plan_id": tick_plan.plan_id,
        "epoch_plan_id": epoch_plan.plan_id,
        "epoch_scope_id": epoch_scope_id,
        "worker": {
            "id": epoch_plan.worker_id,
            "num_workers": epoch_plan.num_workers,
            "sample_count": len(epoch_plan.sample_indices),
        },
        "routing": dict(backend_report.request_routing),
        "table_uri": lake.uri,
        "tables": tables,
        "projected_columns": list(projected_columns),
        "logical_columns": list(logical_columns),
        "excluded_columns": skipped_columns,
        "row_count": sum(int(table["row_count"]) for table in tables),
        "estimated_bytes": 0,
        "limits": _prewarm_limits_dict(options),
        "concurrency": options.concurrency,
        "timeout_s": options.timeout_s,
        "include_heavy": options.include_heavy,
    }
    return _apply_prewarm_limits(request, options)


def _native_prewarm_source_columns(
    columns: Sequence[str],
    *,
    include_heavy: bool,
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    projected: list[str] = []
    skipped: list[dict[str, str]] = []
    for column in columns:
        if column in _PREWARM_HEAVY_TRAINING_COLUMNS and not include_heavy:
            skipped.append(
                {
                    "column": column,
                    "reason": "heavy media prewarm requires include_heavy=True",
                }
            )
            continue
        if column == "payload":
            projected.append(PAYLOAD_BLOB_COLUMN)
            continue
        if column == "video_frame":
            skipped.append(
                {
                    "column": column,
                    "reason": "video frame prewarm needs codec placement metadata",
                }
            )
            continue
        mapped = _TRAINING_TO_OBSERVATION_COLUMN.get(column)
        if mapped:
            projected.append(mapped)
        elif column in _OBSERVATION_COLUMNS:
            projected.append(column)
        else:
            skipped.append({"column": column, "reason": "derived or scenario-level field"})
    return tuple(dict.fromkeys(projected)), skipped


def _aligned_prewarm_source_columns(
    columns: Sequence[str],
    *,
    feature_policy: str | None,
    storage_backend: str,
    include_heavy: bool,
) -> tuple[dict[str, tuple[str, ...]], list[dict[str, str]]]:
    aligned_table = (
        "aligned_ticks" if storage_backend == ALIGNED_TICKS_STORAGE_BACKEND else "aligned_frames"
    )
    aligned_columns = (
        ["aligned_tick_id", "alignment_id", "tick_index", "stream_detail_json", "masks_json"]
        if aligned_table == "aligned_ticks"
        else ["aligned_frame_id", "alignment_id", "tick_index", "stream", "status"]
    )
    observation_columns: list[str] = []
    skipped: list[dict[str, str]] = []
    if "lineage" in columns:
        if aligned_table == "aligned_ticks":
            aligned_columns.extend(["lineage_json"])
        else:
            aligned_columns.extend(["source_observation_ids", "source_row_ids"])
    if "streams" in columns:
        if aligned_table == "aligned_ticks":
            aligned_columns.extend(["stream_detail_json", "stream_values_json"])
        else:
            aligned_columns.extend(["source_observation_ids", "source_row_ids", "value_json"])
        if feature_policy and feature_policy.startswith(("bytes:", "array:", "tensor:")):
            if include_heavy:
                observation_columns.append(PAYLOAD_BLOB_COLUMN)
            else:
                skipped.append(
                    {
                        "column": "payload",
                        "reason": "heavy media prewarm requires include_heavy=True",
                    }
                )
    return (
        {
            aligned_table: tuple(dict.fromkeys(aligned_columns)),
            "observations": tuple(dict.fromkeys(observation_columns)),
        },
        skipped,
    )


def _apply_prewarm_limits(
    request: dict[str, Any],
    options: TrainingPrewarmOptions,
) -> dict[str, Any]:
    if not request.get("projected_columns"):
        request["status"] = "skipped"
        request["skip_reason"] = "no projectable non-heavy columns selected for prewarm"
        return request
    row_count = int(request.get("row_count") or 0)
    estimated_bytes = int(request.get("estimated_bytes") or 0)
    limit_failures: list[str] = []
    if options.max_rows is not None and row_count > options.max_rows:
        limit_failures.append(f"row_count {row_count} exceeds max_rows {options.max_rows}")
    if options.max_bytes is not None and estimated_bytes > options.max_bytes:
        limit_failures.append(
            f"estimated_bytes {estimated_bytes} exceeds max_bytes {options.max_bytes}"
        )
    fragment_count = request.get("fragment_count")
    if (
        options.max_fragments is not None
        and fragment_count is not None
        and int(fragment_count) > options.max_fragments
    ):
        limit_failures.append(
            f"fragment_count {fragment_count} exceeds max_fragments {options.max_fragments}"
        )
    if limit_failures:
        request["status"] = "skipped"
        request["skip_reason"] = "; ".join(limit_failures)
    return request


def _normalize_prewarm_status(
    response: Any,
    *,
    request: Mapping[str, Any],
    default_status: str,
    submit_duration_ms: float | None = None,
    status_duration_ms: float | None = None,
) -> dict[str, Any]:
    if isinstance(response, Mapping):
        status = dict(response)
    else:
        status = {}
    status.setdefault("status", default_status)
    status.setdefault("prewarm_id", request.get("prewarm_id"))
    if status.get("pe_fanout") is None:
        completed = int(status.get("completed_executors") or 0)
        failed = int(status.get("failed_executors") or 0)
        if completed or failed:
            status["pe_fanout"] = completed + failed
    if submit_duration_ms is not None:
        status["submit_duration_ms"] = submit_duration_ms
    if status_duration_ms is not None:
        status["status_duration_ms"] = status_duration_ms
    return _jsonable(status)


def _prewarm_limits_dict(options: TrainingPrewarmOptions) -> dict[str, Any]:
    return {
        "max_rows": options.max_rows,
        "max_bytes": options.max_bytes,
        "max_fragments": options.max_fragments,
        "timeout_s": options.timeout_s,
    }


def _row_id_ranges(row_ids: Sequence[int]) -> list[dict[str, int]]:
    if not row_ids:
        return []
    ordered = sorted(dict.fromkeys(int(row_id) for row_id in row_ids))
    ranges: list[dict[str, int]] = []
    start = prev = ordered[0]
    for row_id in ordered[1:]:
        if row_id == prev + 1:
            prev = row_id
            continue
        ranges.append({"start": start, "end": prev + 1, "count": prev - start + 1})
        start = prev = row_id
    ranges.append({"start": start, "end": prev + 1, "count": prev - start + 1})
    return ranges


def _version_from_table_versions(
    table_versions: Sequence[Mapping[str, Any]],
    table_name: str,
) -> int | None:
    for item in table_versions:
        if str(item.get("table")) == table_name and item.get("version") is not None:
            return int(item["version"])
    return None


def _prewarm_table_label(display_uri: str, table_name: str, version: int | None) -> str:
    suffix = f"@v{version}" if version is not None else ""
    return f"{display_uri}/{table_name}{suffix}"


def _append_backend_warning(backend: dict[str, Any], warning: str) -> None:
    warnings = list(backend.get("warnings") or [])
    if warning not in warnings:
        warnings.append(warning)
    backend["warnings"] = warnings


class _TrainingMediaResolver:
    def __init__(
        self,
        lake: Lake,
        context: Any,
        *,
        media_policy: str,
        decoder: str,
        cache_policy: str,
        cache_size: int,
        hydration_executor: _QueryNodeHydrationExecutor | None = None,
    ) -> None:
        self.lake = lake
        self.context = context
        self.media_policy = media_policy
        self.decoder = decoder
        self.cache_policy = cache_policy
        self.cache_size = cache_size
        self.cache = _MediaCache(cache_policy, cache_size)
        self.hydration_executor = hydration_executor
        self._prefetched_bytes: dict[tuple[Any, ...], bytes | None] = {}

    def value_for_column(
        self,
        ref: _TrainingFrameRef,
        column: str,
        columns: tuple[str, ...],
    ) -> tuple[Any, dict[str, Any] | None]:
        if column not in _HEAVY_MEDIA_COLUMNS:
            return _column_value(self.context, ref, column), None

        handle = self._handle_for_column(ref, column, columns)
        if handle is None:
            return None if column != "payload_size" else 0, self._missing_audit(column)
        if column == "payload_size":
            return self._payload_size_value(handle)
        return self._media_value(handle)

    def prehydrate_refs(
        self,
        refs: Sequence[_TrainingFrameRef],
        columns: tuple[str, ...],
    ) -> None:
        """Coalesce native media hydration for a batch-sized row-plan chunk."""
        self._prefetched_bytes.clear()
        if self.media_policy in {"metadata", "uri"} or self.hydration_executor is None:
            return
        if not self.hydration_executor.enabled:
            return
        handles: list[TrainingMediaHandle] = []
        for ref in refs:
            for column in columns:
                if column not in _HEAVY_MEDIA_COLUMNS:
                    continue
                handle = self._handle_for_column(ref, column, columns)
                if handle is not None and handle.kind == "payload_blob":
                    handles.append(handle)
        row_ids = [
            int(handle.row_id)
            for handle in handles
            if handle.row_id is not None
        ]
        if not row_ids:
            return
        payloads = self.hydration_executor.take_blobs(
            "observations",
            PAYLOAD_BLOB_COLUMN,
            row_ids,
            version=_table_version(self.context, "observations"),
        )
        for handle in handles:
            if handle.row_id is None:
                continue
            self._prefetched_bytes[self._cache_key(handle)] = payloads.get(
                int(handle.row_id)
            )

    def clear_prefetch(self) -> None:
        self._prefetched_bytes.clear()

    def read_handle_bytes(self, handle: TrainingMediaHandle) -> tuple[bytes | None, dict[str, Any]]:
        key = self._cache_key(handle)
        data, cache_hit = self.cache.get(key, lambda: self._load_handle_bytes(handle))
        audit = self._handle_audit(handle)
        audit.update(
            {
                "materialized": True,
                "cache_hit": cache_hit,
                "bytes_read": len(data or b""),
            }
        )
        return data, audit

    def decode_handle_array(self, handle: TrainingMediaHandle) -> Any:
        self._require_optional("PIL", extra="media", purpose="media='array'")
        self._require_optional("numpy", extra="media", purpose="media='array'")
        import numpy as np
        from PIL import Image

        data, _ = self.read_handle_bytes(handle)
        if not data:
            return None
        try:
            return np.asarray(Image.open(io.BytesIO(data)))
        except Exception as exc:
            raise TrainingError(
                f"cannot decode media {handle.media_id!r} as an image array; "
                "request media='bytes' to inspect the original payload"
            ) from exc

    def decode_handle_tensor(self, handle: TrainingMediaHandle) -> Any:
        self._require_optional("torch", extra="torch", purpose="media='tensor'")
        import torch

        array = self.decode_handle_array(handle)
        return None if array is None else torch.as_tensor(array)

    def _media_value(self, handle: TrainingMediaHandle) -> tuple[Any, dict[str, Any]]:
        if self.media_policy == "metadata":
            return handle, self._handle_audit(handle)
        if self.media_policy == "uri":
            return handle.uri_ref(), self._handle_audit(handle)
        if self.media_policy == "bytes":
            return self.read_handle_bytes(handle)
        if self.media_policy == "array":
            value = self.decode_handle_array(handle)
            return value, self._decoded_audit(handle)
        if self.media_policy == "tensor":
            value = self.decode_handle_tensor(handle)
            return value, self._decoded_audit(handle)
        raise TrainingError(f"unknown media policy {self.media_policy!r}")

    def _payload_size_value(self, handle: TrainingMediaHandle) -> tuple[int | None, dict[str, Any]]:
        if self.media_policy in {"metadata", "uri"}:
            return None, self._handle_audit(handle)
        data, audit = self.read_handle_bytes(handle)
        return len(data or b""), audit

    def _handle_for_column(
        self,
        ref: _TrainingFrameRef,
        column: str,
        columns: tuple[str, ...],
    ) -> TrainingMediaHandle | None:
        if column == "video_frame":
            return self._video_handle(ref) or self._payload_handle(ref, field=column)
        if column == "payload_size" and "video_frame" in columns:
            return self._video_handle(ref) or self._payload_handle(ref, field=column)
        return self._payload_handle(ref, field=column)

    def _payload_handle(self, ref: _TrainingFrameRef, *, field: str) -> TrainingMediaHandle | None:
        obs = ref.observation
        observation_id = obs.get("observation_id")
        if not observation_id:
            return None
        return TrainingMediaHandle(
            media_id=f"media-{_stable_digest({'field': field, 'observation_id': observation_id})}",
            field=field,
            kind="payload_blob",
            observation_id=observation_id,
            source_table="observations",
            source_column=PAYLOAD_BLOB_COLUMN,
            source_id=observation_id,
            row_id=ref.row_id,
            raw_uri=obs.get("raw_uri"),
            raw_channel=obs.get("raw_channel"),
            raw_sequence=obs.get("raw_sequence"),
            episode_id=obs.get("episode_id") or ref.episode.episode_id,
            episode_index=int(obs.get("episode_index") if obs.get("episode_index") is not None else ref.episode.index),
            frame_index=int(obs.get("frame_index") if obs.get("frame_index") is not None else ref.frame_index),
            camera_key=_camera_key(obs) if _is_camera_observation(obs) else None,
            decoder=self.decoder,
            _resolver=self,
        )

    def _video_handle(self, ref: _TrainingFrameRef) -> TrainingMediaHandle | None:
        obs = ref.observation
        if not _is_camera_observation(obs):
            return None
        frame_index = obs.get("frame_index")
        if frame_index is None:
            frame_index = ref.frame_index
        frame_index = int(frame_index)
        episode_id = obs.get("episode_id") or ref.episode.episode_id
        camera_key = _camera_key(obs)
        candidates = [
            row
            for row in getattr(self.context, "video_encodings", {}).values()
            if row.get("episode_id") == episode_id
            and row.get("camera_key") == camera_key
            and _encoding_contains_frame(row, frame_index)
        ]
        if not candidates:
            return None
        row = sorted(
            candidates,
            key=lambda item: (item.get("created_at"), item.get("encoding_id")),
        )[-1]
        entry = _encoding_frame_entry(row, frame_index)
        return TrainingMediaHandle(
            media_id=f"media-{_stable_digest({'encoding_id': row['encoding_id'], 'frame': frame_index})}",
            field="video_frame",
            kind="codec_video_frame",
            observation_id=obs["observation_id"],
            source_table="video_encodings",
            source_column=VIDEO_ENCODING_BLOB_COLUMN,
            source_id=row["encoding_id"],
            row_id=None,
            raw_uri=obs.get("raw_uri"),
            raw_channel=obs.get("raw_channel"),
            raw_sequence=obs.get("raw_sequence"),
            episode_id=episode_id,
            episode_index=int(row["episode_index"]),
            frame_index=frame_index,
            camera_key=camera_key,
            video_id=row.get("video_id"),
            byte_range=(int(entry["byte_start"]), int(entry["byte_end"])),
            gop_index=int(entry["gop_index"]),
            decoder=self.decoder,
            _resolver=self,
        )

    def _load_handle_bytes(self, handle: TrainingMediaHandle) -> bytes | None:
        key = self._cache_key(handle)
        if key in self._prefetched_bytes:
            return self._prefetched_bytes[key]
        if handle.kind == "codec_video_frame":
            encoded = self._fetch_blob_as_of(
                "video_encodings",
                VIDEO_ENCODING_BLOB_COLUMN,
                handle.source_id,
                id_column="encoding_id",
            )
            if not encoded:
                return self._fallback_payload_bytes(handle)
            row = self.context.video_encodings.get(handle.source_id)
            if row is None:
                return self._fallback_payload_bytes(handle)
            try:
                decoded = decode_frame_from_encoding(
                    row,
                    encoded,
                    handle.frame_index,
                    decoder=self.decoder,
                )
            except VideoError as exc:
                raise TrainingError(
                    f"cannot decode video frame {handle.frame_index} from "
                    f"encoding {handle.source_id!r}: {exc}"
                ) from exc
            return decoded.frame

        if (
            self.hydration_executor is not None
            and self.hydration_executor.enabled
            and handle.row_id is not None
        ):
            payloads = self.hydration_executor.take_blobs(
                "observations",
                PAYLOAD_BLOB_COLUMN,
                [int(handle.row_id)],
                version=_table_version(self.context, "observations"),
            )
            return payloads.get(int(handle.row_id))

        return self._fetch_blob_as_of(
            "observations",
            PAYLOAD_BLOB_COLUMN,
            handle.observation_id,
            id_column="observation_id",
        )

    def _fallback_payload_bytes(self, handle: TrainingMediaHandle) -> bytes | None:
        return self._fetch_blob_as_of(
            "observations",
            PAYLOAD_BLOB_COLUMN,
            handle.observation_id,
            id_column="observation_id",
        )

    def _fetch_blob_as_of(
        self,
        table_name: str,
        blob_column: str,
        row_id: str,
        *,
        id_column: str,
    ) -> bytes | None:
        table = self.lake.table(table_name)
        version = _table_version(self.context, table_name)
        if version is not None:
            table.checkout(version)
        try:
            return fetch_blob(table, blob_column, row_id, id_column=id_column)
        finally:
            if version is not None:
                table.checkout_latest()

    def _cache_key(self, handle: TrainingMediaHandle) -> tuple[Any, ...]:
        return (
            handle.kind,
            handle.source_table,
            handle.source_id,
            handle.observation_id,
            handle.frame_index,
            self.decoder if handle.kind == "codec_video_frame" else "",
        )

    def _handle_audit(self, handle: TrainingMediaHandle) -> dict[str, Any]:
        audit = handle.to_dict()
        audit.update(
            {
                "policy": self.media_policy,
                "materialized": False,
                "cache_policy": self.cache_policy,
            }
        )
        return audit

    def _decoded_audit(self, handle: TrainingMediaHandle) -> dict[str, Any]:
        audit = self._handle_audit(handle)
        audit.update({"materialized": True, "decoded": self.media_policy})
        return audit

    def _missing_audit(self, column: str) -> dict[str, Any]:
        return {
            "field": column,
            "policy": self.media_policy,
            "materialized": False,
            "missing": True,
            "cache_policy": self.cache_policy,
        }

    def _require_optional(self, module: str, *, extra: str, purpose: str) -> None:
        if importlib.util.find_spec(module) is None:
            raise TrainingError(
                f"{purpose} requires optional dependency {module!r}; install "
                f"`lancedb-robotics[{extra}]` or request media='bytes'"
            )


class _AlignedFeatureResolver:
    def __init__(
        self,
        lake: Lake,
        job: Mapping[str, Any],
        *,
        feature_policy: str,
        decoder: str,
        cache_policy: str,
        cache_size: int,
        hydration_executor: _QueryNodeHydrationExecutor | None = None,
    ) -> None:
        self.lake = lake
        self.job = job
        self.feature_policy = feature_policy
        self.decoder = decoder
        self.cache_policy = cache_policy
        self.cache_size = cache_size
        self.cache = _MediaCache(cache_policy, cache_size)
        self.hydration_executor = hydration_executor
        self.observations_version = _alignment_table_version(job, "observations")
        self._video_encodings: dict[str, dict[str, Any]] | None = None

    def hydrate_streams(
        self,
        stream_samples: Mapping[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return self.hydrate_stream_batches([stream_samples])[0]

    def hydrate_stream_batches(
        self,
        stream_sample_batches: Sequence[Mapping[str, dict[str, Any]]],
    ) -> list[dict[str, dict[str, Any]]]:
        hydrated_batches = [
            {stream: dict(sample) for stream, sample in stream_samples.items()}
            for stream_samples in stream_sample_batches
        ]
        metadata_by_row_id: dict[int, dict[str, Any]] = {}
        metadata_by_observation_id: dict[str, dict[str, Any]] = {}
        if self.feature_policy in {"uri", "bytes", "array", "tensor"}:
            source_row_ids = sorted(
                {
                    int(row_id)
                    for hydrated in hydrated_batches
                    for sample in hydrated.values()
                    for row_id in sample.get("source_row_ids") or []
                }
            )
            source_observation_ids = sorted(
                {
                    str(observation_id)
                    for hydrated in hydrated_batches
                    for sample in hydrated.values()
                    for observation_id in sample.get("source_observation_ids") or []
                    if observation_id
                }
            )
            metadata_by_row_id = self._observation_metadata_by_row_id(source_row_ids)
            metadata_by_observation_id = {
                row["observation_id"]: row for row in metadata_by_row_id.values()
            }
            missing_observation_ids = [
                observation_id
                for observation_id in source_observation_ids
                if observation_id not in metadata_by_observation_id
            ]
            metadata_by_observation_id.update(
                self._observation_metadata_by_observation_id(missing_observation_ids)
            )

        refs_by_batch: list[dict[str, list[dict[str, Any]]]] = []
        for hydrated in hydrated_batches:
            refs_by_batch.append(
                {
                    stream: self._source_refs_for_stream(
                        sample,
                        metadata_by_row_id=metadata_by_row_id,
                        metadata_by_observation_id=metadata_by_observation_id,
                    )
                    for stream, sample in hydrated.items()
                }
            )
        if self.feature_policy in _ALIGNED_PAYLOAD_POLICIES:
            self._materialize_refs(
                [
                    ref
                    for refs_by_stream in refs_by_batch
                    for refs in refs_by_stream.values()
                    for ref in refs
                ]
            )

        for hydrated, refs_by_stream in zip(hydrated_batches, refs_by_batch, strict=True):
            for stream, sample in hydrated.items():
                refs = refs_by_stream.get(stream, [])
                sample["feature"] = self._feature_for_stream(sample, refs)
                sample["quality_flags"] = sorted(
                    set(sample.get("quality_flags") or [])
                    | {
                        f"training:{_aligned_stream_key(stream)}:missing-payload"
                        for ref in refs
                        if ref.get("missing_payload")
                    }
                )
        return hydrated_batches

    def _feature_for_stream(
        self,
        sample: Mapping[str, Any],
        refs: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        feature = {
            "policy": self.feature_policy,
            "decoder": self.decoder,
            "cache_policy": self.cache_policy,
            "value": sample.get("value") if self.feature_policy != "metadata" else None,
            "source_rows": [],
        }
        source_rows: list[dict[str, Any]] = []
        for ref in refs:
            entry = self._source_row_feature(ref)
            if self.feature_policy == "uri":
                entry["uri"] = _aligned_uri_ref(ref)
            elif self.feature_policy == "bytes":
                entry["payload"] = ref.get("payload")
            elif self.feature_policy == "array":
                entry["array"] = ref.get("array")
            elif self.feature_policy == "tensor":
                entry["tensor"] = ref.get("tensor")
            source_rows.append(entry)
        feature["source_rows"] = source_rows
        return feature

    def _source_row_feature(self, ref: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "row_id": ref.get("row_id"),
            "observation_id": ref.get("observation_id"),
            "kind": ref.get("kind"),
            "source_table": ref.get("source_table"),
            "source_column": ref.get("source_column"),
            "source_id": ref.get("source_id"),
            "lance_uri": ref.get("lance_uri"),
            "raw_uri": ref.get("raw_uri"),
            "raw_channel": ref.get("raw_channel"),
            "raw_sequence": ref.get("raw_sequence"),
            "frame_index": ref.get("frame_index"),
            "video_id": ref.get("video_id"),
            "byte_range": ref.get("byte_range"),
            "gop_index": ref.get("gop_index"),
            "materialized": bool(ref.get("materialized")),
            "cache_hit": bool(ref.get("cache_hit")),
            "bytes_read": int(ref.get("bytes_read") or 0),
            "missing_payload": bool(ref.get("missing_payload")),
            "hydration": ref.get("hydration"),
            "decoder": ref.get("decoder"),
        }

    def _source_refs_for_stream(
        self,
        sample: Mapping[str, Any],
        *,
        metadata_by_row_id: Mapping[int, dict[str, Any]],
        metadata_by_observation_id: Mapping[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        row_ids = [int(row_id) for row_id in sample.get("source_row_ids") or []]
        observation_ids = [
            str(observation_id)
            for observation_id in sample.get("source_observation_ids") or []
            if observation_id
        ]
        refs: list[dict[str, Any]] = []
        for offset in range(max(len(row_ids), len(observation_ids))):
            row_id = row_ids[offset] if offset < len(row_ids) else None
            observation_id = observation_ids[offset] if offset < len(observation_ids) else None
            metadata = (
                metadata_by_row_id.get(row_id)
                if row_id is not None
                else None
            ) or (
                metadata_by_observation_id.get(observation_id)
                if observation_id is not None
                else None
            )
            if metadata is None:
                metadata = {
                    "observation_id": observation_id,
                    ROW_ID_COLUMN: row_id,
                    "raw_uri": None,
                    "raw_channel": None,
                    "raw_sequence": None,
                }
            ref = self._video_ref(metadata, row_id=row_id)
            if ref is None:
                ref = self._payload_ref(metadata, row_id=row_id)
            refs.append(ref)
        return refs

    def _payload_ref(
        self,
        metadata: Mapping[str, Any],
        *,
        row_id: int | None,
    ) -> dict[str, Any]:
        observation_id = metadata.get("observation_id")
        return {
            "row_id": row_id,
            "observation_id": observation_id,
            "kind": "payload_blob",
            "source_table": "observations",
            "source_column": PAYLOAD_BLOB_COLUMN,
            "source_id": observation_id,
            "lance_uri": _aligned_lance_uri(
                "observations",
                PAYLOAD_BLOB_COLUMN,
                row_id=row_id,
                source_id=observation_id,
            ),
            "raw_uri": metadata.get("raw_uri"),
            "raw_channel": metadata.get("raw_channel"),
            "raw_sequence": metadata.get("raw_sequence"),
            "frame_index": metadata.get("frame_index"),
            "message_encoding": metadata.get("message_encoding"),
            "schema_encoding": metadata.get("schema_encoding"),
            "decode_status": metadata.get("decode_status"),
            "materialized": False,
            "cache_hit": False,
            "bytes_read": 0,
            "missing_payload": False,
            "hydration": "row-id" if row_id is not None else "observation-id",
            "decoder": self.decoder,
        }

    def _video_ref(
        self,
        metadata: Mapping[str, Any],
        *,
        row_id: int | None,
    ) -> dict[str, Any] | None:
        if not _is_camera_observation(dict(metadata)):
            return None
        episode_id = metadata.get("episode_id")
        frame_index = metadata.get("frame_index")
        if episode_id is None or frame_index is None:
            return None
        camera_key = _camera_key(dict(metadata))
        candidates = [
            row
            for row in self._video_encoding_rows().values()
            if row.get("episode_id") == episode_id
            and row.get("camera_key") == camera_key
            and _encoding_contains_frame(row, int(frame_index))
        ]
        if not candidates:
            return None
        row = sorted(
            candidates,
            key=lambda item: (item.get("created_at"), item.get("encoding_id")),
        )[-1]
        entry = _encoding_frame_entry(row, int(frame_index))
        return {
            "row_id": row_id,
            "observation_id": metadata.get("observation_id"),
            "kind": "codec_video_frame",
            "source_table": "video_encodings",
            "source_column": VIDEO_ENCODING_BLOB_COLUMN,
            "source_id": row["encoding_id"],
            "lance_uri": _aligned_lance_uri(
                "video_encodings",
                VIDEO_ENCODING_BLOB_COLUMN,
                row_id=None,
                source_id=row["encoding_id"],
            ),
            "raw_uri": metadata.get("raw_uri"),
            "raw_channel": metadata.get("raw_channel"),
            "raw_sequence": metadata.get("raw_sequence"),
            "frame_index": int(frame_index),
            "video_id": row.get("video_id"),
            "byte_range": (int(entry["byte_start"]), int(entry["byte_end"])),
            "gop_index": int(entry["gop_index"]),
            "materialized": False,
            "cache_hit": False,
            "bytes_read": 0,
            "missing_payload": False,
            "hydration": "codec-video",
            "decoder": self.decoder,
        }

    def _materialize_refs(self, refs: Sequence[dict[str, Any]]) -> None:
        refs_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for ref in refs:
            refs_by_key.setdefault(self._cache_key(ref), []).append(ref)

        refs_to_load: list[dict[str, Any]] = []
        for key, grouped_refs in refs_by_key.items():
            cached, cache_hit = self.cache.get_cached(key)
            if cache_hit:
                for ref in grouped_refs:
                    self._apply_materialized_bytes(ref, cached, cache_hit=True)
            else:
                refs_to_load.append(grouped_refs[0])

        payloads = [ref for ref in refs_to_load if ref.get("kind") == "payload_blob"]
        payload_load_refs = payloads
        if self.hydration_executor is not None and self.hydration_executor.enabled:
            payload_load_refs = [
                ref
                for grouped_refs in refs_by_key.values()
                for ref in grouped_refs
                if ref.get("kind") == "payload_blob"
            ]
        payload_results = self._load_payload_refs(payload_load_refs)
        for ref in payloads:
            data = payload_results.get(id(ref))
            key = self._cache_key(ref)
            self.cache.put(key, data)
            for grouped_ref in refs_by_key[key]:
                self._apply_materialized_bytes(grouped_ref, data, cache_hit=False)

        for ref in refs_to_load:
            if ref.get("kind") != "codec_video_frame":
                continue
            data = self._load_video_ref(ref)
            key = self._cache_key(ref)
            self.cache.put(key, data)
            for grouped_ref in refs_by_key[key]:
                self._apply_materialized_bytes(grouped_ref, data, cache_hit=False)

    def _load_payload_refs(
        self,
        refs: Sequence[dict[str, Any]],
    ) -> dict[int, bytes | None]:
        if not refs:
            return {}
        loaded: dict[int, bytes | None] = {}
        row_refs = [ref for ref in refs if ref.get("row_id") is not None]
        fallback_refs: list[dict[str, Any]] = []
        if row_refs:
            row_ids = [int(ref["row_id"]) for ref in row_refs]
            try:
                by_row_id = self._fetch_payloads_by_row_id(row_ids)
            except Exception:
                by_row_id = {}
                fallback_refs.extend(row_refs)
            else:
                for ref in row_refs:
                    row_id = int(ref["row_id"])
                    if row_id in by_row_id:
                        loaded[id(ref)] = by_row_id[row_id]
                        ref["hydration"] = "row-id"
                    else:
                        fallback_refs.append(ref)

        fallback_refs.extend(ref for ref in refs if ref.get("row_id") is None)
        fallback_ids = [
            str(ref["observation_id"])
            for ref in fallback_refs
            if ref.get("observation_id")
        ]
        by_observation_id = self._fetch_payloads_by_observation_id(fallback_ids)
        for ref in fallback_refs:
            observation_id = ref.get("observation_id")
            if observation_id and observation_id in by_observation_id:
                loaded[id(ref)] = by_observation_id[str(observation_id)]
                ref["hydration"] = "observation-id"
            elif id(ref) not in loaded:
                loaded[id(ref)] = None
                ref["hydration"] = "missing"
        return loaded

    def _load_video_ref(self, ref: Mapping[str, Any]) -> bytes | None:
        table = self.lake.table("video_encodings")
        encoded = fetch_blobs(
            table,
            VIDEO_ENCODING_BLOB_COLUMN,
            [str(ref["source_id"])],
            id_column="encoding_id",
        ).get(str(ref["source_id"]))
        if not encoded:
            payload_ref = self._payload_ref(ref, row_id=ref.get("row_id"))
            payloads = self._load_payload_refs([payload_ref])
            return next(iter(payloads.values()), None)
        row = self._video_encoding_rows().get(str(ref["source_id"]))
        if row is None:
            return None
        try:
            decoded = decode_frame_from_encoding(
                row,
                encoded,
                int(ref["frame_index"]),
                decoder=self.decoder,
            )
        except VideoError as exc:
            raise TrainingError(
                f"cannot decode aligned video frame {ref['frame_index']} from "
                f"encoding {ref['source_id']!r}: {exc}"
            ) from exc
        return decoded.frame

    def _apply_materialized_bytes(
        self,
        ref: dict[str, Any],
        data: bytes | None,
        *,
        cache_hit: bool,
    ) -> None:
        ref["materialized"] = True
        ref["cache_hit"] = cache_hit
        ref["bytes_read"] = len(data or b"")
        ref["missing_payload"] = not bool(data)
        if self.feature_policy == "bytes":
            ref["payload"] = data
            return
        if self.feature_policy == "array":
            ref["array"] = self._decode_array(data, ref)
            return
        if self.feature_policy == "tensor":
            ref["tensor"] = self._decode_tensor(data, ref)

    def _decode_array(self, data: bytes | None, ref: Mapping[str, Any]) -> Any:
        if not data:
            return None
        self._require_optional("PIL", extra="media", purpose="features='array'")
        self._require_optional("numpy", extra="media", purpose="features='array'")
        import numpy as np
        from PIL import Image

        try:
            return np.asarray(Image.open(io.BytesIO(data)))
        except Exception as exc:
            raise TrainingError(
                f"cannot decode aligned feature payload {ref.get('source_id')!r} "
                "as an image array; request features='bytes' to inspect the original payload"
            ) from exc

    def _decode_tensor(self, data: bytes | None, ref: Mapping[str, Any]) -> Any:
        if not data:
            return None
        self._require_optional("torch", extra="torch", purpose="features='tensor'")
        import torch

        array = self._decode_array(data, ref)
        return None if array is None else torch.as_tensor(array)

    def _fetch_payloads_by_row_id(self, row_ids: Sequence[int]) -> dict[int, bytes]:
        if (
            self.hydration_executor is not None
            and self.hydration_executor.enabled
        ):
            return {
                row_id: value or b""
                for row_id, value in self.hydration_executor.take_blobs(
                    "observations",
                    PAYLOAD_BLOB_COLUMN,
                    row_ids,
                    version=self.observations_version,
                ).items()
            }
        table = self.lake.table("observations")
        if self.observations_version is not None:
            table.checkout(self.observations_version)
        try:
            return fetch_blobs_by_row_id(table, PAYLOAD_BLOB_COLUMN, row_ids)
        finally:
            if self.observations_version is not None:
                table.checkout_latest()

    def _fetch_payloads_by_observation_id(
        self,
        observation_ids: Sequence[str],
    ) -> dict[str, bytes]:
        if not observation_ids:
            return {}
        table = self.lake.table("observations")
        if self.observations_version is not None:
            table.checkout(self.observations_version)
        try:
            return fetch_blobs(
                table,
                PAYLOAD_BLOB_COLUMN,
                observation_ids,
                id_column="observation_id",
            )
        finally:
            if self.observations_version is not None:
                table.checkout_latest()

    def _observation_metadata_by_row_id(
        self,
        row_ids: Sequence[int],
    ) -> dict[int, dict[str, Any]]:
        wanted = {int(row_id) for row_id in row_ids}
        if not wanted:
            return {}
        if (
            self.hydration_executor is not None
            and self.hydration_executor.enabled
        ):
            return self.hydration_executor.take_rows(
                "observations",
                sorted(wanted),
                columns=_ALIGNED_OBSERVATION_FEATURE_COLUMNS,
                version=self.observations_version,
            )
        table = self.lake.table("observations")
        if self.observations_version is not None:
            table.checkout(self.observations_version)
        try:
            dataset = table.to_lance()
            rows = dataset.to_table(
                columns=list(_ALIGNED_OBSERVATION_FEATURE_COLUMNS),
                with_row_id=True,
            ).to_pylist()
        finally:
            if self.observations_version is not None:
                table.checkout_latest()
        return {
            int(row[ROW_ID_COLUMN]): row
            for row in rows
            if row.get(ROW_ID_COLUMN) is not None and int(row[ROW_ID_COLUMN]) in wanted
        }

    def _observation_metadata_by_observation_id(
        self,
        observation_ids: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        ids = tuple(dict.fromkeys(str(observation_id) for observation_id in observation_ids))
        if not ids:
            return {}
        predicate = _sql_predicate("observation_id", ids)
        if (
            self.hydration_executor is not None
            and self.hydration_executor.enabled
        ):
            rows = self.hydration_executor.filtered_read(
                "observations",
                columns=_ALIGNED_OBSERVATION_FEATURE_COLUMNS,
                where_sql=predicate,
                version=self.observations_version,
                with_row_id=True,
            )
        else:
            rows = _scan_observations_as_of(
                self.lake,
                self.observations_version,
                _ALIGNED_OBSERVATION_FEATURE_COLUMNS,
                predicate,
            )
        return {str(row["observation_id"]): row for row in rows if row.get("observation_id")}

    def _video_encoding_rows(self) -> dict[str, dict[str, Any]]:
        if self._video_encodings is None:
            self._video_encodings = {
                row["encoding_id"]: row
                for row in self.lake.table("video_encodings").to_arrow().to_pylist()
                if row.get("encoding_id")
            }
        return self._video_encodings

    def _cache_key(self, ref: Mapping[str, Any]) -> tuple[Any, ...]:
        if ref.get("kind") == "codec_video_frame":
            return (
                "codec_video_frame",
                ref.get("source_id"),
                ref.get("frame_index"),
                self.decoder,
            )
        return (
            "payload_blob",
            self.observations_version,
            ref.get("row_id"),
            ref.get("observation_id"),
        )

    def _require_optional(self, module: str, *, extra: str, purpose: str) -> None:
        if importlib.util.find_spec(module) is None:
            raise TrainingError(
                f"{purpose} requires optional dependency {module!r}; install "
                f"`lancedb-robotics[{extra}]` or request features='bytes'"
            )


class LanceTrainingDataset:
    """Random-access, Lance-native training view over a pinned snapshot.

    This is intentionally not a LeRobot/WebDataset facade. Samples expose the
    lake's canonical observation/scenario/run fields, with projection,
    deterministic global shuffle, filter, and temporal-window controls that map
    to the capabilities the Lance substrate is meant to optimize.
    """

    def __init__(
        self,
        lake: Lake,
        snapshot_name: str,
        *,
        columns: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        shuffle: bool = False,
        shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
        epoch: int = 0,
        worker_id: int = 0,
        num_workers: int = 1,
        resume_from: int = 0,
        time_windows: Mapping[str, Sequence[float]] | None = None,
        media: str | bool = DEFAULT_MEDIA_POLICY,
        payloads: str | bool | None = None,
        decoder: str = "auto",
        media_cache: str = DEFAULT_MEDIA_CACHE_POLICY,
        media_cache_size: int = DEFAULT_MEDIA_CACHE_SIZE,
        backend: str = DEFAULT_TRAINING_BACKEND,
        cache_policy: str = DEFAULT_ENTERPRISE_CACHE_POLICY,
        prewarm: bool = False,
        prewarm_options: Mapping[str, Any] | None = None,
        allow_fallback: bool = False,
        fallback: str | bool | None = None,
    ) -> None:
        self.lake = lake
        self.snapshot_name = snapshot_name
        self.columns = tuple(columns or DEFAULT_TRAINING_COLUMNS)
        self.filters = dict(filters or {})
        self.shuffle = bool(shuffle)
        self.shuffle_seed = shuffle_seed
        self.time_windows = {
            key: tuple(float(delta) for delta in deltas)
            for key, deltas in (time_windows or {}).items()
        }
        self.epoch = int(epoch)
        self.worker_id = int(worker_id)
        self.num_workers = int(num_workers)
        self.resume_from = int(resume_from)
        self.media_policy = _resolve_media_policy(media, payloads)
        self.decoder = _validate_decoder(decoder)
        self.media_cache = _validate_media_cache(media_cache)
        self.media_cache_size = int(media_cache_size)
        self.prewarm_options = _normalize_prewarm_options(prewarm_options)
        self.fallback_policy = _validate_enterprise_fallback_policy(
            fallback,
            allow_fallback=allow_fallback,
        )
        self.backend_report = _training_backend_report(
            lake,
            backend=backend,
            cache_policy=cache_policy,
            prewarm=prewarm,
            fallback_policy=self.fallback_policy,
            required_capabilities=_native_required_enterprise_capabilities(
                self.columns,
                media_policy=self.media_policy,
                cache_policy=cache_policy,
            ),
        )
        _validate_columns(self.columns)
        _validate_filter_keys(self.filters)
        _validate_window_keys(self.time_windows)
        _validate_epoch_args(
            epoch=self.epoch,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            resume_from=self.resume_from,
        )
        _validate_media_cache_size(self.media_cache_size)

        self._context = _training_snapshot_context(
            lake,
            snapshot_name,
            remote_safe=self.backend_report.resolved_backend == "enterprise",
            observation_columns=_native_context_observation_columns(
                self.columns, self.filters, self.time_windows
            ),
        )
        self._episodes = _episodes(self._context)
        all_refs = _frame_refs(self._episodes)
        self._refs_by_episode = _refs_by_episode(all_refs)
        self.row_plan, self._planned_refs = _build_row_plan(
            lake,
            self._context,
            all_refs,
            columns=self.columns,
            filters=self.filters,
            media_policy=self.media_policy,
            decoder=self.decoder,
            cache_policy=self.media_cache,
        )
        self.epoch_plan = _build_epoch_plan(
            self.row_plan,
            lake=lake,
            enable_lancedb_backend=True,
            shuffle=self.shuffle,
            shuffle_seed=self.shuffle_seed,
            epoch=self.epoch,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            resume_from=self.resume_from,
        )
        self._frame_refs = tuple(self._planned_refs[index] for index in self.epoch_plan.sample_indices)

        # Backlog 0117: an Enterprise dataset can express its epoch order as a
        # server-side row-plan artifact -- a version-pinned handle the query node
        # persists once and workers page. The handle summary is advertised eagerly
        # (O(1), no I/O); the pages materialize lazily on ``row_plan_handle`` access
        # so forked worker reconstruction never writes the store.
        self._server_side_plan_capable = bool(
            self.backend_report.resolved_backend == "enterprise"
            and self.backend_report.capabilities.get("server_side_row_plan")
        )
        self._server_side_plan: ServerSidePlanArtifact | None = None
        self._server_side_plan_summary = _server_side_plan_summary(self)

        self.fps = _infer_fps(self._episodes)
        self.num_episodes = len(self._episodes)
        self.num_frames = len(self._frame_refs)
        self.total_frames = len(all_refs)
        self.features = _training_features()
        self.episode_data_index = _episode_data_index(self._episodes)
        accounting = ProjectionAccounting(
            logical_row_count=self.num_frames,
            selected_scenario_count=len(self._context.scenario_ids),
            selected_observation_count=self.num_frames,
            payload_bytes_referenced=sum(
                payload_size(ref.observation.get("payload_blob"))
                for ref in self._frame_refs
            ),
            payload_bytes_copied=0,
            metadata_bytes_written=0,
            target_format=(
                "lance-native-enterprise-training"
                if self.backend_report.resolved_backend == "enterprise"
                else "lance-native-training"
            ),
            target_path=lake.uri,
            projection_transform_id=self.row_plan.plan_id,
            source_snapshot_id=self._context.dataset_id,
            source_snapshot_name=self._context.snapshot_name,
            source_table_versions=self._context.table_versions,
            mode="live",
            payload_copy_policy="logical-reference",
            dry_run=False,
        ).to_dict()
        self.backend_report = _training_backend_report(
            lake,
            backend=backend,
            cache_policy=cache_policy,
            prewarm=prewarm,
            fallback_policy=self.fallback_policy,
            required_capabilities=_native_required_enterprise_capabilities(
                self.columns,
                media_policy=self.media_policy,
                cache_policy=cache_policy,
            ),
            row_plan=self.row_plan,
            selected_frames=self.num_frames,
        )
        self.manifest = LanceTrainingManifest(
            lake_uri=lake.uri,
            dataset_id=self._context.dataset_id,
            snapshot_name=self._context.snapshot_name,
            access_pattern=(
                "enterprise-remote-snapshot"
                if self.backend_report.resolved_backend == "enterprise"
                else "lance-native-snapshot"
            ),
            table_versions=self._context.table_versions,
            columns=self.columns,
            filters=self.filters,
            shuffle=self.shuffle,
            shuffle_seed=self.shuffle_seed if self.shuffle else None,
            episode_count=self.num_episodes,
            total_frames=self.total_frames,
            selected_frames=self.num_frames,
            fps=self.fps,
            time_windows=self.time_windows,
            row_plan_id=self.row_plan.plan_id,
            epoch_plan_id=self.epoch_plan.plan_id,
            epoch=self.epoch_plan.epoch,
            ordering_policy=self.epoch_plan.ordering_policy,
            worker_id=self.epoch_plan.worker_id,
            num_workers=self.epoch_plan.num_workers,
            resume_from=self.epoch_plan.resume_from,
            media_policy=self.media_policy,
            decoder=self.decoder,
            cache_policy=self.media_cache,
            cache_size=self.media_cache_size,
            accounting=accounting,
            epoch_backend=self.epoch_plan.backend.to_dict(),
            backend=self.backend_report.to_dict(),
            server_side_plan=self._server_side_plan_summary,
        )
        accounting["metadata_bytes_written"] = json_metadata_bytes(self.manifest.to_dict())
        self._hydration_executor = _QueryNodeHydrationExecutor(
            lake,
            self.backend_report,
            table_versions=self._context.table_versions,
            manifest_backend=self.manifest.backend,
            coalescing_window="native-training-batch",
            metadata_only=self.media_policy in {"metadata", "uri"},
        )
        self._prewarm_executor = _PageCachePrewarmExecutor(
            lake,
            self.backend_report,
            manifest_backend=self.manifest.backend,
        )
        # Backlog 0121: epoch prewarm warms the whole epoch, not this worker's
        # shard, so every worker derives the same request/JobRun. ``snapshot``
        # already spans the full plan; ``epoch`` reads the epoch-global order.
        prewarm_refs = (
            self._planned_refs
            if self.backend_report.cache["policy"] == "snapshot"
            else tuple(self._planned_refs[index] for index in self.epoch_plan.global_order)
        )
        self._prewarm_executor.maybe_prewarm_training(
            self.row_plan,
            self.epoch_plan,
            refs=prewarm_refs,
            options=self.prewarm_options,
        )
        accounting["metadata_bytes_written"] = json_metadata_bytes(self.manifest.to_dict())
        self._media_resolver = _TrainingMediaResolver(
            lake,
            self._context,
            media_policy=self.media_policy,
            decoder=self.decoder,
            cache_policy=self.media_cache,
            cache_size=self.media_cache_size,
            hydration_executor=self._hydration_executor,
        )

    def __len__(self) -> int:
        return len(self._frame_refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ref = self._frame_refs[index]
        sample = _sample_for_ref(self._context, ref, self.columns, self._media_resolver)
        if self.time_windows:
            sample["windows"] = {
                key: self._window(ref, key, deltas)
                for key, deltas in self.time_windows.items()
            }
        return sample

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        normalized = [index + len(self) if index < 0 else int(index) for index in indices]
        for index in normalized:
            if index < 0 or index >= len(self):
                raise IndexError(index)
        refs = [self._frame_refs[index] for index in normalized]
        self._media_resolver.prehydrate_refs(refs, self.columns)
        try:
            return [self[index] for index in normalized]
        finally:
            self._media_resolver.clear_prefetch()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def prewarm_status(
        self,
        *,
        wait: bool = False,
        fail_on_error: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Return or refresh the Enterprise cache prewarm status for this dataset."""
        return self._prewarm_executor.status(
            wait=wait,
            fail_on_error=fail_on_error,
            timeout_s=timeout_s,
        )

    def page_cache_prewarm_plan(
        self,
        *,
        concurrency: int | None = None,
        estimate: bool = True,
    ) -> dict[str, Any]:
        """Return the Enterprise query-node page-cache prewarm plan (backlog 0122).

        Emits a valid ``PageCacheBeginPrewarmRequest`` (``{db, table, columns,
        table_version, concurrency}``) per table plus advisory cost estimates
        (selected vs total rows, over-warm ratio, per-column byte estimate, local
        fragment estimate). The plan targets the query node only and contains no
        plan-executor placement, ``pe_fanout``, or row/fragment routing.
        """
        return self._prewarm_executor.page_cache_plan(
            concurrency=concurrency,
            estimate=estimate,
        ).to_dict()

    def _build_query_warm_plan(
        self,
        *,
        chunk_size: int,
        include_heavy: bool,
        columns: Sequence[str] | None,
    ):
        logical = tuple(columns or self.row_plan.columns)
        source_columns, _skipped = _native_prewarm_source_columns(
            logical, include_heavy=include_heavy
        )
        table = str(self.row_plan.scan.get("table") or "observations")
        id_column = warm_id_column(table) or "observation_id"
        version = _version_from_table_versions(self.row_plan.table_versions, table)
        # Always warm the id column too, and guarantee a non-empty SELECT.
        select_columns = tuple(dict.fromkeys((id_column, *source_columns)))
        spec = QueryWarmTableSpec(
            table=table,
            version=version,
            id_column=id_column,
            id_values=self.row_plan.frame_ids,
            columns=select_columns,
        )
        return build_query_warm_plan(
            snapshot_name=self.snapshot_name,
            scope="row-plan",
            specs=[spec],
            chunk_size=chunk_size,
            index_checker=_query_warm_index_checker(self.lake),
        )

    def query_warm_plan(
        self,
        *,
        chunk_size: int = DEFAULT_QUERY_WARM_CHUNK_SIZE,
        include_heavy: bool = False,
        columns: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Build a query-driven cache-warm plan for this dataset (backlog 0348).

        Emits bounded ``where(<stable id> IN (...))`` warm queries over the epoch's
        observation ids and the projected training columns, so the query node warms
        exactly the rows this run reads (not the whole table). Filters on a stable
        data-id column (``observation_id``), never ``_rowid``, and contains no
        plan-executor placement. Warns when the id column is not scalar-indexed (the
        warm would otherwise degrade to a full scan).
        """
        return self._build_query_warm_plan(
            chunk_size=chunk_size, include_heavy=include_heavy, columns=columns
        ).to_dict()

    def warm_query_cache(
        self,
        *,
        chunk_size: int = DEFAULT_QUERY_WARM_CHUNK_SIZE,
        include_heavy: bool = False,
        columns: Sequence[str] | None = None,
        row_limit_per_query: int | None = None,
    ) -> dict[str, Any]:
        """Execute the query-driven warm plan to warm the query-node cache (backlog 0348).

        Runs each bounded warm query through the sanctioned ``search().where(...)`` path
        and drains it (bounded memory) so the query node caches the touched pages.
        Best-effort per query; never contacts a plan executor.
        """
        plan = self._build_query_warm_plan(
            chunk_size=chunk_size, include_heavy=include_heavy, columns=columns
        )
        result = warm_query_cache(
            self.lake, plan, row_limit_per_query=row_limit_per_query
        )
        result["plan"] = plan.to_dict()
        return result

    def loader_report(
        self,
        *,
        training_run_id: str | None = None,
        model_run_id: str | None = None,
        model_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> TrainingLoaderReport:
        """Return the current structured loader report for this dataset."""
        return TrainingLoaderReport(
            _native_loader_report_payload(
                self.manifest.to_dict(include_loader_report=False),
                run=_loader_report_run_hooks(
                    training_run_id=training_run_id,
                    model_run_id=model_run_id,
                    model_id=model_id,
                    extra=extra,
                ),
            )
        )

    def loader_config(self) -> dict[str, Any]:
        """Return a lightweight, picklable config for worker-side reconstruction."""
        return {
            "lake_uri": self.lake.uri,
            "snapshot_name": self.snapshot_name,
            "columns": list(self.columns),
            "filters": _jsonable(self.filters),
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "epoch": self.epoch,
            "worker_id": self.worker_id,
            "num_workers": self.num_workers,
            "resume_from": self.resume_from,
            "time_windows": {
                key: list(deltas) for key, deltas in self.time_windows.items()
            },
            "media": self.media_policy,
            "decoder": self.decoder,
            "media_cache": self.media_cache,
            "media_cache_size": self.media_cache_size,
            "backend": self.backend_report.requested_backend,
            "cache_policy": self.backend_report.cache["policy"],
            "prewarm": self.backend_report.cache["prewarm_requested"],
            "prewarm_options": self.prewarm_options.to_dict(),
            "allow_fallback": self.backend_report.fallback is not None,
            "fallback": self.fallback_policy,
            "connection": _connection_loader_config(self.lake),
        }

    def server_side_plan(
        self,
        *,
        page_size: int = DEFAULT_PLAN_PAGE_SIZE,
        store: str = "auto",
    ) -> ServerSidePlanArtifact:
        """Build (once) and return the server-side row-plan artifact for this dataset.

        Only available for the resolved Enterprise backend with the
        ``server_side_row_plan`` capability; raises
        :class:`ServerSidePlanUnavailableError` otherwise (backlog 0117). The
        artifact is a version-pinned, paginated, serializable handle whose epoch
        order equals this dataset's global order.
        """
        if self._server_side_plan is not None and page_size == self._server_side_plan.page_size:
            return self._server_side_plan
        artifact = _build_dataset_server_side_plan(self, page_size=page_size, store=store)
        self._server_side_plan = artifact
        return artifact

    @property
    def row_plan_handle(self) -> ServerSidePlanArtifact | None:
        """The server-side plan artifact for Enterprise datasets, else ``None``."""
        if not self._server_side_plan_capable or self._server_side_plan_summary is None:
            return None
        return self.server_side_plan()

    def torch(
        self,
        *,
        iterable: bool | None = None,
        adapter: str | None = None,
        include_lineage: bool = True,
    ):
        """Wrap this native training dataset as a PyTorch dataset."""
        if adapter is None:
            adapter = "iterable" if iterable is not False else "map"
        elif iterable is not None:
            requested = "iterable" if iterable else "map"
            if str(adapter).lower().replace("_", "-") not in {requested, requested + "-style"}:
                raise TrainingError("pass either iterable or adapter, not conflicting values")
        selected = str(adapter).lower().replace("_", "-")
        if selected in {"map", "map-style", "random-access"}:
            return to_torch_map_dataset(self, include_lineage=include_lineage)
        if selected in {"iter", "iterable", "stream", "streaming"}:
            return to_torch_iterable_dataset(self, include_lineage=include_lineage)
        raise TrainingError("adapter must be one of map or iterable")

    def torch_dataloader(
        self,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = 0,
        adapter: str = "auto",
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        include_lineage: bool = True,
        **kwargs: Any,
    ):
        """Build a PyTorch DataLoader over this native training dataset."""
        return to_torch_dataloader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            adapter=adapter,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            include_lineage=include_lineage,
            **kwargs,
        )

    def _window(
        self,
        ref: _TrainingFrameRef,
        key: str,
        deltas: tuple[float, ...],
    ) -> list[dict[str, Any]]:
        episode_refs = self._refs_by_episode[ref.episode.index]
        base_ts = int(ref.observation["timestamp_ns"])
        values: list[dict[str, Any]] = []
        for delta in deltas:
            target_ns = base_ts + round(delta * 1_000_000_000)
            aligned = min(
                episode_refs,
                key=lambda candidate: (
                    abs(int(candidate.observation["timestamp_ns"]) - target_ns),
                    candidate.frame_index,
                ),
            )
            values.append(
                {
                    "delta_s": delta,
                    "index": aligned.linear_index,
                    "frame_index": aligned.frame_index,
                    "timestamp_ns": int(aligned.observation["timestamp_ns"]),
                    "value": _column_value(self._context, aligned, key),
                }
            )
        return values


class AlignedFrameTrainingDataset:
    """Random-access, framework-neutral training view over materialized alignment rows."""

    def __init__(
        self,
        lake: Lake,
        alignment: str | None = None,
        *,
        alignment_id: str | None = None,
        name: str | None = None,
        streams: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        statuses: Sequence[str] | str | None = None,
        require_streams: bool | Sequence[str] = False,
        allow_missing: bool = True,
        min_confidence: float | None = None,
        shuffle: bool = False,
        shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
        epoch: int = 0,
        worker_id: int = 0,
        num_workers: int = 1,
        resume_from: int = 0,
        features: str | bool = DEFAULT_ALIGNED_FEATURE_POLICY,
        decoder: str = "auto",
        feature_cache: str = DEFAULT_MEDIA_CACHE_POLICY,
        feature_cache_size: int = DEFAULT_MEDIA_CACHE_SIZE,
        backend: str = DEFAULT_TRAINING_BACKEND,
        cache_policy: str = DEFAULT_ENTERPRISE_CACHE_POLICY,
        prewarm: bool = False,
        prewarm_options: Mapping[str, Any] | None = None,
        allow_fallback: bool = False,
        fallback: str | bool | None = None,
    ) -> None:
        if alignment is not None and (alignment_id is not None or name is not None):
            raise TrainingError("pass either alignment or alignment_id/name, not both")
        self.lake = lake
        self._job = _resolve_alignment_job(
            lake,
            alignment=alignment,
            alignment_id=alignment_id,
            name=name,
        )
        self.alignment_id = str(self._job["alignment_id"])
        self.alignment_name = str(self._job["name"])
        self.columns = tuple(columns or DEFAULT_ALIGNED_TRAINING_COLUMNS)
        self.streams = _normalize_aligned_streams(self._job, streams)
        self.statuses = _normalize_aligned_statuses(statuses)
        self.require_streams = _normalize_required_streams(require_streams, self.streams)
        self.allow_missing = bool(allow_missing)
        self.min_confidence = _validate_min_confidence(min_confidence)
        self.shuffle = bool(shuffle)
        self.shuffle_seed = shuffle_seed
        self.epoch = int(epoch)
        self.worker_id = int(worker_id)
        self.num_workers = int(num_workers)
        self.resume_from = int(resume_from)
        self.feature_policy = _resolve_aligned_feature_policy(features)
        self.decoder = _validate_decoder(decoder)
        self.feature_cache = _validate_media_cache(feature_cache)
        self.feature_cache_size = int(feature_cache_size)
        self.prewarm_options = _normalize_prewarm_options(prewarm_options)
        self.fallback_policy = _validate_enterprise_fallback_policy(
            fallback,
            allow_fallback=allow_fallback,
        )
        self.backend_report = _training_backend_report(
            lake,
            backend=backend,
            cache_policy=cache_policy,
            prewarm=prewarm,
            fallback_policy=self.fallback_policy,
            required_capabilities=_aligned_required_enterprise_capabilities(
                self.feature_policy,
                cache_policy=cache_policy,
            ),
        )
        _validate_aligned_columns(self.columns)
        _validate_epoch_args(
            epoch=self.epoch,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            resume_from=self.resume_from,
        )
        _validate_media_cache_size(self.feature_cache_size)

        self.quality_policy = {
            "allow_missing": self.allow_missing,
            "min_confidence": self.min_confidence,
            "require_streams": list(self.require_streams),
            "statuses": list(self.statuses),
        }
        (
            self.tick_plan,
            self._rows_by_tick,
            self._filtered_ticks,
        ) = _build_aligned_tick_plan(
            lake,
            self._job,
            streams=self.streams,
            columns=self.columns,
            quality_policy=self.quality_policy,
            feature_policy=self.feature_policy,
            decoder=self.decoder,
            cache_policy=self.feature_cache,
        )
        self.epoch_plan = _build_epoch_plan(
            self.tick_plan,
            shuffle=self.shuffle,
            shuffle_seed=self.shuffle_seed,
            epoch=self.epoch,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            resume_from=self.resume_from,
        )
        self._tick_indices = tuple(
            self.tick_plan.tick_indices[index] for index in self.epoch_plan.sample_indices
        )
        self.backend_report = _training_backend_report(
            lake,
            backend=backend,
            cache_policy=cache_policy,
            prewarm=prewarm,
            fallback_policy=self.fallback_policy,
            required_capabilities=_aligned_required_enterprise_capabilities(
                self.feature_policy,
                cache_policy=cache_policy,
            ),
            selected_frames=len(self._tick_indices),
            plan_id=self.tick_plan.plan_id,
            table_versions=self.tick_plan.read_table_versions,
        )
        self._hydration_executor = _QueryNodeHydrationExecutor(
            lake,
            self.backend_report,
            table_versions=self.tick_plan.table_versions,
            coalescing_window="aligned-training-batch",
            metadata_only=self.feature_policy not in _ALIGNED_PAYLOAD_POLICIES,
        )
        self._feature_resolver = _AlignedFeatureResolver(
            lake,
            self._job,
            feature_policy=self.feature_policy,
            decoder=self.decoder,
            cache_policy=self.feature_cache,
            cache_size=self.feature_cache_size,
            hydration_executor=self._hydration_executor,
        )
        self.num_ticks = len(self._tick_indices)
        self.total_ticks = self.tick_plan.total_ticks
        self.manifest = AlignedTrainingManifest(
            lake_uri=lake.uri,
            alignment_id=self.alignment_id,
            alignment_name=self.alignment_name,
            access_pattern=(
                "lance-native-aligned-ticks"
                if self.tick_plan.storage_backend == ALIGNED_TICKS_STORAGE_BACKEND
                else "lance-native-aligned-frames"
            ),
            storage_backend=self.tick_plan.storage_backend,
            schema_version=self.tick_plan.schema_version,
            recipe_digest=_alignment_recipe_digest(self._job),
            output_table=str(self.tick_plan.scan.get("table") or self._job["output_table"]),
            table_versions=_alignment_input_versions(self._job),
            read_table_versions=self.tick_plan.read_table_versions,
            streams=self.streams,
            columns=self.columns,
            quality_policy=self.quality_policy,
            total_ticks=self.tick_plan.total_ticks,
            selected_ticks=self.num_ticks,
            tick_plan_id=self.tick_plan.plan_id,
            epoch_plan_id=self.epoch_plan.plan_id,
            epoch=self.epoch_plan.epoch,
            ordering_policy=self.epoch_plan.ordering_policy,
            worker_id=self.epoch_plan.worker_id,
            num_workers=self.epoch_plan.num_workers,
            resume_from=self.epoch_plan.resume_from,
            feature_policy=self.feature_policy,
            decoder=self.decoder,
            cache_policy=self.feature_cache,
            cache_size=self.feature_cache_size,
            predicate_indexes=tuple(self.tick_plan.scan.get("predicate_indexes") or ()),
            epoch_backend=self.epoch_plan.backend.to_dict(),
            backend=self.backend_report.to_dict(),
        )
        self._hydration_executor.manifest_backend = self.manifest.backend
        self._prewarm_executor = _PageCachePrewarmExecutor(
            lake,
            self.backend_report,
            manifest_backend=self.manifest.backend,
        )
        self._prewarm_executor.maybe_prewarm_aligned(
            self.tick_plan,
            self.epoch_plan,
            options=self.prewarm_options,
        )

    def __len__(self) -> int:
        return len(self._tick_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        plan_index = int(self.epoch_plan.sample_indices[index])
        tick_index = self.tick_plan.tick_indices[plan_index]
        sample = _aligned_sample_for_tick(
            self._job,
            tick_index,
            self._rows_by_tick.get(tick_index, {}),
            streams=self.streams,
            columns=self.columns,
            quality_policy=self.quality_policy,
            manifest=self.manifest,
            tick_plan=self.tick_plan,
            epoch_plan=self.epoch_plan,
            sample_index=index,
            plan_index=plan_index,
            feature_resolver=self._feature_resolver,
        )
        return sample

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        normalized = [index + len(self) if index < 0 else int(index) for index in indices]
        for index in normalized:
            if index < 0 or index >= len(self):
                raise IndexError(index)
        plans = [
            (
                index,
                int(self.epoch_plan.sample_indices[index]),
                int(self.tick_plan.tick_indices[int(self.epoch_plan.sample_indices[index])]),
            )
            for index in normalized
        ]
        stream_sample_batches = [
            {
                stream: _aligned_stream_sample(
                    stream,
                    self._rows_by_tick.get(tick_index, {}).get(stream),
                )
                for stream in self.streams
            }
            for _, _, tick_index in plans
        ]
        hydrated_batches = self._feature_resolver.hydrate_stream_batches(
            stream_sample_batches
        )
        return [
            _aligned_sample_from_streams(
                self._job,
                tick_index,
                hydrated_streams,
                columns=self.columns,
                quality_policy=self.quality_policy,
                manifest=self.manifest,
                tick_plan=self.tick_plan,
                epoch_plan=self.epoch_plan,
                sample_index=index,
                plan_index=plan_index,
            )
            for (index, plan_index, tick_index), hydrated_streams in zip(
                plans, hydrated_batches, strict=True
            )
        ]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def prewarm_status(
        self,
        *,
        wait: bool = False,
        fail_on_error: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Return or refresh the Enterprise cache prewarm status for this dataset."""
        return self._prewarm_executor.status(
            wait=wait,
            fail_on_error=fail_on_error,
            timeout_s=timeout_s,
        )

    def page_cache_prewarm_plan(
        self,
        *,
        concurrency: int | None = None,
        estimate: bool = True,
    ) -> dict[str, Any]:
        """Return the Enterprise query-node page-cache prewarm plan (backlog 0122).

        Emits a valid ``PageCacheBeginPrewarmRequest`` (``{db, table, columns,
        table_version, concurrency}``) per table plus advisory cost estimates. The
        plan targets the query node only and contains no plan-executor placement,
        ``pe_fanout``, or row/fragment routing.
        """
        return self._prewarm_executor.page_cache_plan(
            concurrency=concurrency,
            estimate=estimate,
        ).to_dict()

    def loader_report(
        self,
        *,
        training_run_id: str | None = None,
        model_run_id: str | None = None,
        model_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> TrainingLoaderReport:
        """Return the current structured loader report for this aligned dataset."""
        return TrainingLoaderReport(
            _aligned_loader_report_payload(
                self.manifest.to_dict(include_loader_report=False),
                run=_loader_report_run_hooks(
                    training_run_id=training_run_id,
                    model_run_id=model_run_id,
                    model_id=model_id,
                    extra=extra,
                ),
            )
        )

    def loader_config(self) -> dict[str, Any]:
        """Return a lightweight, picklable config for aligned worker reconstruction."""
        return {
            "lake_uri": self.lake.uri,
            "alignment_id": self.alignment_id,
            "alignment_name": self.alignment_name,
            "streams": list(self.streams),
            "columns": list(self.columns),
            "statuses": list(self.statuses),
            "require_streams": list(self.require_streams),
            "allow_missing": self.allow_missing,
            "min_confidence": self.min_confidence,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "epoch": self.epoch,
            "worker_id": self.worker_id,
            "num_workers": self.num_workers,
            "resume_from": self.resume_from,
            "features": self.feature_policy,
            "decoder": self.decoder,
            "feature_cache": self.feature_cache,
            "feature_cache_size": self.feature_cache_size,
            "backend": self.backend_report.requested_backend,
            "cache_policy": self.backend_report.cache["policy"],
            "prewarm": self.backend_report.cache["prewarm_requested"],
            "prewarm_options": self.prewarm_options.to_dict(),
            "allow_fallback": self.backend_report.fallback is not None,
            "fallback": self.fallback_policy,
            "connection": _connection_loader_config(self.lake),
        }

    def torch(
        self,
        *,
        adapter: str = "iterable",
        include_lineage: bool = True,
    ):
        """Wrap this aligned dataset as a PyTorch dataset."""
        return to_torch_aligned_dataset(
            self,
            adapter=adapter,
            include_lineage=include_lineage,
        )

    def torch_dataloader(
        self,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = 0,
        adapter: str = "auto",
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        include_lineage: bool = True,
        **kwargs: Any,
    ):
        """Build a PyTorch DataLoader over this aligned dataset."""
        return to_torch_aligned_dataloader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            adapter=adapter,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            include_lineage=include_lineage,
            **kwargs,
        )


class LakeTraining:
    """Convenience namespace exposed as ``lake.training``."""

    def __init__(self, lake: Lake) -> None:
        self._lake = lake

    def record_run(
        self,
        snapshot: str | None = None,
        **kwargs: Any,
    ):
        """Record a training run manifest for a pinned dataset snapshot."""
        from lancedb_robotics.run_manifests import record_training_run

        return record_training_run(self._lake, snapshot=snapshot, **kwargs)

    def record_checkpoint(
        self,
        **kwargs: Any,
    ):
        """Record a checkpoint/model artifact produced by a training run."""
        from lancedb_robotics.run_manifests import record_checkpoint

        return record_checkpoint(self._lake, **kwargs)

    def attach_external_refs(
        self,
        training_run_id: str,
        refs: Mapping[str, Any],
        *,
        replace: bool = False,
    ):
        """Attach MLflow/W&B/etc. references to an existing training manifest."""
        from lancedb_robotics.run_manifests import attach_training_external_refs

        return attach_training_external_refs(
            self._lake,
            training_run_id,
            refs,
            replace=replace,
        )

    def attach_model_external_refs(
        self,
        model_artifact_id: str,
        refs: Mapping[str, Any],
        *,
        replace: bool = False,
    ):
        """Attach external tracker/artifact references to an existing model row."""
        from lancedb_robotics.run_manifests import attach_model_external_refs

        return attach_model_external_refs(
            self._lake,
            model_artifact_id,
            refs,
            replace=replace,
        )

    def runs(self, **kwargs: Any):
        """Bounded/indexed training-run query with an execution plan (backlog 0100)."""
        from lancedb_robotics.run_manifests import query_training_runs

        return query_training_runs(self._lake, **kwargs)

    def checkpoints(self, **kwargs: Any):
        """Bounded/indexed model-artifact query (by training run, alias, checksum...)."""
        from lancedb_robotics.run_manifests import query_model_artifacts

        return query_model_artifacts(self._lake, **kwargs)

    def list_runs(self, **kwargs: Any):
        """Deterministic paged training-run listing with continuation tokens."""
        from lancedb_robotics.run_manifests import list_training_runs

        return list_training_runs(self._lake, **kwargs)

    def list_checkpoints(self, **kwargs: Any):
        """Deterministic paged model-artifact listing with continuation tokens."""
        from lancedb_robotics.run_manifests import list_model_artifacts

        return list_model_artifacts(self._lake, **kwargs)

    def retention_plan(self, **kwargs: Any):
        """Report which training/eval manifests are protected vs safe to expire."""
        from lancedb_robotics.run_manifests import plan_manifest_retention

        return plan_manifest_retention(self._lake, **kwargs)

    def expire_run(self, training_run_id: str, *, force: bool = False):
        """Delete a training run (refused if protected unless ``force``)."""
        from lancedb_robotics.run_manifests import delete_manifest

        return delete_manifest(
            self._lake, kind="training-run", manifest_id=training_run_id, force=force
        )

    def expire_checkpoint(self, model_artifact_id: str, *, force: bool = False):
        """Delete a checkpoint/model artifact (refused if protected unless ``force``)."""
        from lancedb_robotics.run_manifests import delete_manifest

        return delete_manifest(
            self._lake, kind="model-artifact", manifest_id=model_artifact_id, force=force
        )

    def record_report(
        self,
        report: Any | None = None,
        *,
        dataset: Any | None = None,
        **kwargs: Any,
    ):
        """Persist an Enterprise training loader/backend report (backlog 0115).

        Pass a ``dataset`` (its ``loader_report`` + ``manifest.backend`` are
        read) or a ``report`` (``TrainingLoaderReport`` / payload mapping).
        Idempotent by report content digest.
        """
        from lancedb_robotics.run_manifests import record_training_report

        return record_training_report(self._lake, report, dataset=dataset, **kwargs)

    def reports(self, **kwargs: Any):
        """Deterministic paged listing of persisted training-report history.

        Returns summary rows (no full payload bodies); reload one full report
        with :meth:`get_report`.
        """
        from lancedb_robotics.run_manifests import list_training_reports

        return list_training_reports(self._lake, **kwargs)

    def query_reports(self, **kwargs: Any):
        """Bounded training-report query with an execution plan (by run/backend/fallback)."""
        from lancedb_robotics.run_manifests import query_training_reports

        return query_training_reports(self._lake, **kwargs)

    def get_report(self, report_id: str | None = None, **kwargs: Any):
        """Reload the full backend report for a report id or run/epoch/worker."""
        from lancedb_robotics.run_manifests import get_training_report

        return get_training_report(self._lake, report_id=report_id, **kwargs)

    def report_metrics(self, **kwargs: Any):
        """Sum cache hits/misses, bytes read, and PE fanout across matched reports."""
        from lancedb_robotics.run_manifests import aggregate_training_report_metrics

        return aggregate_training_report_metrics(self._lake, **kwargs)

    def report_retention_plan(self, **kwargs: Any):
        """Report which persisted training reports are protected vs safe to expire."""
        from lancedb_robotics.run_manifests import plan_manifest_retention

        return plan_manifest_retention(self._lake, kinds=("training-report",), **kwargs)

    def expire_report(self, report_id: str, *, force: bool = False):
        """Delete a persisted training report (refused if protected unless ``force``)."""
        from lancedb_robotics.run_manifests import delete_manifest

        return delete_manifest(
            self._lake, kind="training-report", manifest_id=report_id, force=force
        )

    def aggregate_reports(self, reports: Any, *, job_id: str | None = None):
        """Combine per-worker loader reports into one job-level report (backlog 0123).

        ``reports`` is any iterable of :class:`TrainingLoaderReport` (or objects
        with ``to_dict()``) or payload mappings, one per distributed worker/epoch.
        Sums cache hits/misses, bytes read, row counts, and operation counts across
        workers; deduplicates shared prewarm events by ``prewarm_id``; preserves
        per-worker/per-epoch drill-downs; and surfaces mixed-backend or fallback
        states as job-level warnings. The result is deterministic (order-independent)
        and lake-independent — this method is a convenience wrapper for discovery.
        """
        from lancedb_robotics.training_report_aggregation import (
            aggregate_training_loader_reports,
        )

        return aggregate_training_loader_reports(reports, job_id=job_id)

    def validate_report(self, report: Any) -> ReportValidation:
        """Validate a loader report against the ``.../v1`` schema + redaction contract (0124).

        ``report`` is a :class:`TrainingLoaderReport`, a payload mapping, or any
        object with ``to_dict()``. Returns a :class:`ReportValidation` whose
        ``ok`` is ``True`` only when the JSON conforms to the versioned schema
        *and* carries no unredacted credential. Lake-independent — exposed here
        for discovery alongside :meth:`aggregate_reports` and :meth:`record_report`.
        """
        return validate_training_loader_report(report)

    def epoch_backend_capability(self) -> dict[str, Any]:
        """Report available epoch-order execution backends for this lake."""
        return _epoch_backend_capability(self._lake)

    def permutation_capability(self) -> dict[str, Any]:
        """Report whether the native LanceDB ``Permutation`` reader can back a plan.

        Backlog 0120 adopts ``lancedb.permutation.Permutation`` as a read backend over
        the single-table native observation path (lazy ``select_columns`` projection and
        ``torch_col``/other native formats). This reports module/connection availability
        and whether ``torch_col`` is limited to scalar numeric columns on this stack.
        """
        from lancedb_robotics.training_permutation import native_permutation_capability

        return native_permutation_capability(self._lake)

    def permutation_plan(
        self,
        snapshot_name: str,
        *,
        columns: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        shuffle: bool = True,
        shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
        epoch: int = 0,
        worker_id: int = 0,
        num_workers: int = 1,
        resume_from: int = 0,
        time_windows: Mapping[str, Sequence[float]] | None = None,
        output_format: str = "arrow",
        verify: bool = True,
    ) -> Any:
        """Back a version-pinned snapshot plan with the native ``Permutation`` reader.

        Builds the normal row/epoch plan (which, under backlog 0077, already persists the
        deterministic ``(row_id, split_id)`` ordering table) and reads through it with the
        native ``lancedb.permutation.Permutation`` reader: ``columns`` are projected via
        :meth:`Permutation.select_columns` and ``output_format`` selects a native batch
        format (``"torch_col"`` requires scalar numeric columns). Returns a
        :class:`~lancedb_robotics.training_permutation.NativePermutationPlan` handle whose
        ``.reader()`` streams batches. Aligned grouped-source snapshots are not eligible;
        use :meth:`aligned_dataset` (executor-backed).
        """
        from lancedb_robotics.training_permutation import build_native_permutation_plan

        dataset = LanceTrainingDataset(
            self._lake,
            snapshot_name,
            columns=columns,
            filters=filters,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            epoch=epoch,
            worker_id=worker_id,
            num_workers=num_workers,
            resume_from=resume_from,
            time_windows=time_windows,
        )
        return build_native_permutation_plan(
            self._lake,
            dataset.row_plan,
            dataset.epoch_plan,
            columns=columns,
            output_format=output_format,
            verify=verify,
        )

    def server_side_row_plan(
        self,
        snapshot_name: str,
        *,
        columns: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        shuffle: bool = False,
        shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
        epoch: int = 0,
        backend: str = "enterprise",
        cache_policy: str = DEFAULT_ENTERPRISE_CACHE_POLICY,
        allow_fallback: bool = False,
        fallback: str | bool | None = None,
        page_size: int = DEFAULT_PLAN_PAGE_SIZE,
        store: str = "auto",
    ) -> dict[str, Any]:
        """Build a version-pinned server-side row-plan artifact and return its handle.

        The query node persists the epoch order once as paginated pages and returns a
        small, secret-free, serializable handle (backlog 0117). Raises
        :class:`ServerSidePlanUnavailableError` when the resolved backend lacks the
        ``server_side_row_plan`` capability, pointing at local fallback / capability
        negotiation. Page the returned handle with :meth:`row_plan_page`.
        """
        dataset = LanceTrainingDataset(
            self._lake,
            snapshot_name,
            columns=columns,
            filters=filters,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            epoch=epoch,
            worker_id=0,
            num_workers=1,
            resume_from=0,
            backend=backend,
            cache_policy=cache_policy,
            allow_fallback=allow_fallback,
            fallback=fallback,
        )
        artifact = dataset.server_side_plan(page_size=page_size, store=store)
        return artifact.to_dict()

    def page_cache_prewarm_plan(
        self,
        snapshot_name: str,
        *,
        columns: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        shuffle: bool = False,
        shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
        epoch: int = 0,
        backend: str = "enterprise",
        cache_policy: str = "epoch",
        include_heavy: bool = False,
        allow_fallback: bool = False,
        fallback: str | bool | None = None,
        concurrency: int | None = None,
        estimate: bool = True,
    ) -> dict[str, Any]:
        """Build the Enterprise query-node page-cache prewarm plan for a snapshot (0122).

        Returns a valid ``PageCacheBeginPrewarmRequest`` (``{db, table, columns,
        table_version, concurrency}``) per table the training snapshot/epoch reads,
        plus advisory cost estimates. Defaults to ``cache_policy="epoch"`` so a plan
        is produced; the plan targets the query node only and never a plan executor.
        """
        dataset = LanceTrainingDataset(
            self._lake,
            snapshot_name,
            columns=columns,
            filters=filters,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            epoch=epoch,
            worker_id=0,
            num_workers=1,
            resume_from=0,
            backend=backend,
            cache_policy=cache_policy,
            prewarm_options={"include_heavy": include_heavy},
            allow_fallback=allow_fallback,
            fallback=fallback,
        )
        return dataset.page_cache_prewarm_plan(concurrency=concurrency, estimate=estimate)

    def query_warm_plan(
        self,
        snapshot_name: str,
        *,
        columns: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        shuffle: bool = False,
        shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
        epoch: int = 0,
        chunk_size: int = DEFAULT_QUERY_WARM_CHUNK_SIZE,
        include_heavy: bool = False,
        backend: str = "auto",
        allow_fallback: bool = False,
        fallback: str | bool | None = None,
    ) -> dict[str, Any]:
        """Build a query-driven cache-warm plan for a snapshot (backlog 0348).

        Emits bounded ``where(<stable id> IN (...))`` warm queries over the epoch's
        stable ids so the query node warms exactly what training reads. Works for any
        backend (the plan is just the reads the loader will run); never references a
        plan executor and never uses ``_rowid``.
        """
        dataset = LanceTrainingDataset(
            self._lake,
            snapshot_name,
            columns=columns,
            filters=filters,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            epoch=epoch,
            worker_id=0,
            num_workers=1,
            resume_from=0,
            backend=backend,
            allow_fallback=allow_fallback,
            fallback=fallback,
        )
        return dataset.query_warm_plan(
            chunk_size=chunk_size, include_heavy=include_heavy
        )

    def row_plan_page(
        self,
        handle: Mapping[str, Any],
        *,
        worker_id: int = 0,
        num_workers: int = 1,
        resume_from: int = 0,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one bounded page of a server-side row-plan handle.

        Reopens the handle against its durable page store and returns either the page
        addressed by ``page_token`` or this worker's first page for
        ``(worker_id, num_workers, resume_from)`` -- with a ``next_page_token`` for
        deterministic worker handoff. No snapshot table object is required.
        """
        store = _plan_page_store_for_handle(self._lake, handle)
        artifact = open_server_side_row_plan(handle, store=store)
        if page_token is not None:
            return artifact.page_from_token(page_token, resume_from=resume_from).to_dict()
        for page in artifact.iter_pages(
            worker_id=worker_id, num_workers=num_workers, resume_from=resume_from
        ):
            return page.to_dict()
        return {
            "plan_handle_id": artifact.plan_handle_id,
            "worker": {"id": worker_id, "num_workers": num_workers},
            "resume_from": resume_from,
            "page_index": None,
            "page_token": None,
            "next_page_token": None,
            "start_offset": None,
            "row_ids": [],
            "frame_ids": [],
            "size": 0,
        }

    def row_plan_pages(
        self,
        handle: Mapping[str, Any],
        *,
        worker_id: int = 0,
        num_workers: int = 1,
        resume_from: int = 0,
    ) -> list[dict[str, Any]]:
        """Return all of a worker's bounded pages for a server-side row-plan handle."""
        store = _plan_page_store_for_handle(self._lake, handle)
        artifact = open_server_side_row_plan(handle, store=store)
        return [
            page.to_dict()
            for page in artifact.iter_pages(
                worker_id=worker_id, num_workers=num_workers, resume_from=resume_from
            )
        ]

    def conformance_matrix(self, *, include_local_endpoint: bool = False):
        """Return the Enterprise-training compatibility matrix (backlog 0116).

        A deterministic, data-free classification of each backend scenario and
        injected fault as ``supported`` / ``fallback`` / ``unsupported`` using
        the production backend resolver. Use :meth:`run_conformance` to replay a
        real snapshot through the same cases.
        """
        from lancedb_robotics.enterprise_conformance import compatibility_matrix

        return compatibility_matrix(include_local_endpoint=include_local_endpoint)

    def run_conformance(
        self,
        snapshot: str,
        *,
        include_local_endpoint: bool = False,
        strict: bool = False,
        **kwargs: Any,
    ):
        """Replay ``snapshot`` through every backend case and injected fault.

        Proves each degradation yields a typed error or an explicit fallback
        report (never silent local materialization), that ``host_override``
        routing survives the worker handoff without leaking API keys, and that
        local and (faked) Enterprise paths emit equivalent sample/row/table-version
        lineage and batch schemas. Set ``strict=True`` to raise on any violation.
        """
        from lancedb_robotics.enterprise_conformance import run_conformance

        return run_conformance(
            self._lake,
            snapshot,
            include_local_endpoint=include_local_endpoint,
            strict=strict,
            **kwargs,
        )

    def query_node_conformance(
        self,
        snapshot: str,
        *,
        include_local_endpoint: bool = False,
        strict: bool = False,
    ):
        """Prove the live query-node read-client contract for ``snapshot`` (0119).

        Runs the native fixture against both the local Lance-backed reader and an
        attached high-fidelity fake query-node client, then asserts local/live
        payload + lineage equivalence, real cache hits/misses aggregated by
        request (server-reported telemetry), manifest e-tags carried in every
        request envelope, request-id recording, typed
        :class:`RemoteQueryNodeError` on a forced failure, and the metadata-only
        guardrail. ``include_local_endpoint`` records a gated local Sophon
        query-node endpoint when available. Set ``strict=True`` to raise on any
        violation.
        """
        from lancedb_robotics.enterprise_conformance import run_query_node_conformance

        return run_query_node_conformance(
            self._lake,
            snapshot,
            include_local_endpoint=include_local_endpoint,
            strict=strict,
        )

    # Backlog 0345 deprecation alias: the client talks to the query node.
    plan_executor_conformance = query_node_conformance

    def prewarm_jobs(
        self,
        *,
        status: str | None = None,
        policy: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List durable Enterprise cache-prewarm JobRuns for this lake (backlog 0121).

        Returns the deduplicated JobRun lifecycle records -- one per ``prewarm_id``
        across workers, retries, and repeated job starts -- most-recently-updated
        first. Returns ``[]`` when no durable JobRun store is attached or present.
        """
        store = open_prewarm_job_store(self._lake)
        if store is None:
            return []
        return [record.to_dict() for record in store.list(status=status, policy=policy, limit=limit)]

    def prewarm_job(self, prewarm_id: str) -> dict[str, Any] | None:
        """Return a single durable prewarm JobRun by id (with full status history)."""
        store = open_prewarm_job_store(self._lake)
        if store is None:
            return None
        record = store.get(prewarm_id)
        return record.to_dict() if record is not None else None

    def prewarm_job_status(self, prewarm_id: str) -> dict[str, Any] | None:
        """Return the 0072-shaped status envelope for a durable prewarm JobRun."""
        store = open_prewarm_job_store(self._lake)
        if store is None:
            return None
        record = store.get(prewarm_id)
        return record.status_dict() if record is not None else None

    def retry_prewarm_job(self, prewarm_id: str) -> dict[str, Any]:
        """Re-submit a failed/canceled/expired prewarm JobRun (increments retry count).

        Raises :class:`PrewarmJobRunError` when the JobRun is unknown or is still
        warm/in-flight (only terminal-not-warm JobRuns can be retried).
        """
        coordinator = self._prewarm_coordinator(prewarm_id)
        return coordinator.retry(prewarm_id).status_dict()

    def cancel_prewarm_job(self, prewarm_id: str, *, reason: str | None = None) -> dict[str, Any]:
        """Cancel a no-longer-needed prewarm JobRun; complete JobRuns are left as-is."""
        coordinator = self._prewarm_coordinator(prewarm_id)
        return coordinator.cancel(prewarm_id, reason=reason).status_dict()

    def expire_prewarm_jobs(self) -> list[dict[str, Any]]:
        """Sweep and mark TTL-elapsed prewarm JobRuns as ``expired`` (maintenance)."""
        store = open_prewarm_job_store(self._lake)
        if store is None:
            return []
        coordinator = PrewarmJobCoordinator(store)
        return [record.to_dict() for record in coordinator.expire_due()]

    def _prewarm_coordinator(self, prewarm_id: str) -> PrewarmJobCoordinator:
        store = open_prewarm_job_store(self._lake)
        if store is None:
            raise PrewarmJobRunError(
                f"no durable prewarm JobRun store for id {prewarm_id!r}; attach "
                "lake.prewarm_job_store or set lake.prewarm_jobs_durable=True"
            )
        source = _lake_hook(self._lake, "page_cache_prewarm", "plan_executor_prewarm")
        status_source = _lake_hook(self._lake, "page_cache_prewarm_status", "plan_executor_prewarm_status")
        return PrewarmJobCoordinator(
            store,
            submit_fn=(lambda req: source(request=dict(req))) if callable(source) else None,
            status_fn=(
                (lambda **kw: status_source(**kw)) if callable(status_source) else None
            ),
            ttl_s=getattr(self._lake, "prewarm_job_ttl_s", DEFAULT_PREWARM_JOB_TTL_S),
        )

    def dataset(
        self,
        snapshot_name: str,
        *,
        columns: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        shuffle: bool = False,
        shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
        epoch: int = 0,
        worker_id: int = 0,
        num_workers: int = 1,
        resume_from: int = 0,
        time_windows: Mapping[str, Sequence[float]] | None = None,
        media: str | bool = DEFAULT_MEDIA_POLICY,
        payloads: str | bool | None = None,
        decoder: str = "auto",
        media_cache: str = DEFAULT_MEDIA_CACHE_POLICY,
        media_cache_size: int = DEFAULT_MEDIA_CACHE_SIZE,
        backend: str = DEFAULT_TRAINING_BACKEND,
        cache_policy: str = DEFAULT_ENTERPRISE_CACHE_POLICY,
        prewarm: bool = False,
        prewarm_options: Mapping[str, Any] | None = None,
        allow_fallback: bool = False,
        fallback: str | bool | None = None,
    ) -> LanceTrainingDataset:
        return LanceTrainingDataset(
            self._lake,
            snapshot_name,
            columns=columns,
            filters=filters,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            epoch=epoch,
            worker_id=worker_id,
            num_workers=num_workers,
            resume_from=resume_from,
            time_windows=time_windows,
            media=media,
            payloads=payloads,
            decoder=decoder,
            media_cache=media_cache,
            media_cache_size=media_cache_size,
            backend=backend,
            cache_policy=cache_policy,
            prewarm=prewarm,
            prewarm_options=prewarm_options,
            allow_fallback=allow_fallback,
            fallback=fallback,
        )

    def aligned_dataset(
        self,
        alignment: str | None = None,
        *,
        alignment_id: str | None = None,
        name: str | None = None,
        streams: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        statuses: Sequence[str] | str | None = None,
        require_streams: bool | Sequence[str] = False,
        allow_missing: bool = True,
        min_confidence: float | None = None,
        shuffle: bool = False,
        shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
        epoch: int = 0,
        worker_id: int = 0,
        num_workers: int = 1,
        resume_from: int = 0,
        features: str | bool = DEFAULT_ALIGNED_FEATURE_POLICY,
        decoder: str = "auto",
        feature_cache: str = DEFAULT_MEDIA_CACHE_POLICY,
        feature_cache_size: int = DEFAULT_MEDIA_CACHE_SIZE,
        backend: str = DEFAULT_TRAINING_BACKEND,
        cache_policy: str = DEFAULT_ENTERPRISE_CACHE_POLICY,
        prewarm: bool = False,
        prewarm_options: Mapping[str, Any] | None = None,
        allow_fallback: bool = False,
        fallback: str | bool | None = None,
    ) -> AlignedFrameTrainingDataset:
        return AlignedFrameTrainingDataset(
            self._lake,
            alignment,
            alignment_id=alignment_id,
            name=name,
            streams=streams,
            columns=columns,
            statuses=statuses,
            require_streams=require_streams,
            allow_missing=allow_missing,
            min_confidence=min_confidence,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            epoch=epoch,
            worker_id=worker_id,
            num_workers=num_workers,
            resume_from=resume_from,
            features=features,
            decoder=decoder,
            feature_cache=feature_cache,
            feature_cache_size=feature_cache_size,
            backend=backend,
            cache_policy=cache_policy,
            prewarm=prewarm,
            prewarm_options=prewarm_options,
            allow_fallback=allow_fallback,
            fallback=fallback,
        )

    def backfill_aligned_ticks(
        self,
        alignment: str | None = None,
        *,
        alignment_id: str | None = None,
        name: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        replace: bool = True,
        verify: bool = True,
    ) -> dict[str, Any]:
        return backfill_aligned_ticks(
            self._lake,
            alignment,
            alignment_id=alignment_id,
            name=name,
            batch_size=batch_size,
            replace=replace,
            verify=verify,
        )

    def index_aligned_predicates(
        self,
        *,
        include_frames: bool = True,
        refresh: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Build scalar indexes for aligned training hot predicates."""
        return tuple(
            result.to_params()
            for result in build_aligned_training_predicate_indexes(
                self._lake,
                include_frames=include_frames,
                replace=refresh,
            )
        )


def training_dataset(
    lake: Lake,
    snapshot_name: str,
    *,
    columns: Sequence[str] | None = None,
    filters: Mapping[str, Any] | None = None,
    shuffle: bool = False,
    shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
    epoch: int = 0,
    worker_id: int = 0,
    num_workers: int = 1,
    resume_from: int = 0,
    time_windows: Mapping[str, Sequence[float]] | None = None,
    media: str | bool = DEFAULT_MEDIA_POLICY,
    payloads: str | bool | None = None,
    decoder: str = "auto",
    media_cache: str = DEFAULT_MEDIA_CACHE_POLICY,
    media_cache_size: int = DEFAULT_MEDIA_CACHE_SIZE,
    backend: str = DEFAULT_TRAINING_BACKEND,
    cache_policy: str = DEFAULT_ENTERPRISE_CACHE_POLICY,
    prewarm: bool = False,
    prewarm_options: Mapping[str, Any] | None = None,
    allow_fallback: bool = False,
    fallback: str | bool | None = None,
) -> LanceTrainingDataset:
    """Build a Lance-native training dataset over ``snapshot_name``."""
    return LanceTrainingDataset(
        lake,
        snapshot_name,
        columns=columns,
        filters=filters,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        epoch=epoch,
        worker_id=worker_id,
        num_workers=num_workers,
        resume_from=resume_from,
        time_windows=time_windows,
        media=media,
        payloads=payloads,
        decoder=decoder,
        media_cache=media_cache,
        media_cache_size=media_cache_size,
        backend=backend,
        cache_policy=cache_policy,
        prewarm=prewarm,
        prewarm_options=prewarm_options,
        allow_fallback=allow_fallback,
        fallback=fallback,
    )


def aligned_training_dataset(
    lake: Lake,
    alignment: str | None = None,
    *,
    alignment_id: str | None = None,
    name: str | None = None,
    streams: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    statuses: Sequence[str] | str | None = None,
    require_streams: bool | Sequence[str] = False,
    allow_missing: bool = True,
    min_confidence: float | None = None,
    shuffle: bool = False,
    shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
    epoch: int = 0,
    worker_id: int = 0,
    num_workers: int = 1,
    resume_from: int = 0,
    features: str | bool = DEFAULT_ALIGNED_FEATURE_POLICY,
    decoder: str = "auto",
    feature_cache: str = DEFAULT_MEDIA_CACHE_POLICY,
    feature_cache_size: int = DEFAULT_MEDIA_CACHE_SIZE,
    backend: str = DEFAULT_TRAINING_BACKEND,
    cache_policy: str = DEFAULT_ENTERPRISE_CACHE_POLICY,
    prewarm: bool = False,
    prewarm_options: Mapping[str, Any] | None = None,
    allow_fallback: bool = False,
    fallback: str | bool | None = None,
) -> AlignedFrameTrainingDataset:
    """Build a Lance-native training dataset over recorded alignment ticks."""
    return AlignedFrameTrainingDataset(
        lake,
        alignment,
        alignment_id=alignment_id,
        name=name,
        streams=streams,
        columns=columns,
        statuses=statuses,
        require_streams=require_streams,
        allow_missing=allow_missing,
        min_confidence=min_confidence,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        epoch=epoch,
        worker_id=worker_id,
        num_workers=num_workers,
        resume_from=resume_from,
        features=features,
        decoder=decoder,
        feature_cache=feature_cache,
        feature_cache_size=feature_cache_size,
        backend=backend,
        cache_policy=cache_policy,
        prewarm=prewarm,
        prewarm_options=prewarm_options,
        allow_fallback=allow_fallback,
        fallback=fallback,
    )


def index_aligned_training_predicates(
    lake: Lake,
    *,
    include_frames: bool = True,
    refresh: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Build scalar indexes for aligned training hot predicates."""
    return tuple(
        result.to_params()
        for result in build_aligned_training_predicate_indexes(
            lake,
            include_frames=include_frames,
            replace=refresh,
        )
    )


def backfill_aligned_ticks(
    lake: Lake,
    alignment: str | None = None,
    *,
    alignment_id: str | None = None,
    name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    replace: bool = True,
    verify: bool = True,
) -> dict[str, Any]:
    """Derive ``aligned_ticks`` rows from legacy ``aligned_frames`` rows."""
    if batch_size <= 0:
        raise TrainingError("batch_size must be positive")
    job = _resolve_alignment_job(
        lake,
        alignment=alignment,
        alignment_id=alignment_id,
        name=name,
    )
    streams = _normalize_aligned_streams(job, None)
    frame_scan = _planned_aligned_frame_scan(
        lake,
        str(job["alignment_id"]),
        streams,
        statuses=(),
        min_confidence=None,
    )
    rows_by_tick = _aligned_rows_by_tick(frame_scan["rows"], streams)
    created_at = datetime.now(UTC)
    tick_rows = [
        _aligned_tick_storage_row_from_frame_rows(
            job,
            tick_index=tick,
            rows_by_stream=rows_by_tick[tick],
            streams=streams,
            created_at=created_at,
        )
        for tick in sorted(rows_by_tick)
    ]
    table = _ensure_aligned_ticks_table(lake)
    if replace:
        table.delete(f"alignment_id = {_sql_literal(job['alignment_id'])}")
    for chunk in _chunks(tick_rows, batch_size):
        table.add(pa.Table.from_pylist(chunk, schema=ALIGNED_TICKS_SCHEMA))
    verified = False
    if verify:
        tick_scan = _planned_aligned_tick_scan(lake, str(job["alignment_id"]))
        tick_rows_by_tick = {
            int(row["tick_index"]): _aligned_tick_stream_detail(row)
            for row in tick_scan["rows"]
            if int(row["tick_index"]) in rows_by_tick
        }
        before = [
            _aligned_metadata_signature(tick, rows_by_tick[tick], streams)
            for tick in sorted(rows_by_tick)
        ]
        after = [
            _aligned_metadata_signature(tick, tick_rows_by_tick.get(tick, {}), streams)
            for tick in sorted(rows_by_tick)
        ]
        verified = before == after
        if not verified:
            raise TrainingError(
                f"backfilled aligned_ticks for {job['alignment_id']!r} did not match "
                "aligned_frames metadata samples"
            )
    return {
        "alignment_id": str(job["alignment_id"]),
        "storage_backend": ALIGNED_TICKS_STORAGE_BACKEND,
        "schema_version": ALIGNED_TICKS_SCHEMA_VERSION,
        "aligned_ticks_written": len(tick_rows),
        "source_aligned_frame_rows": int(frame_scan["row_count"]),
        "metadata_samples_verified": verified,
        "replaced": bool(replace),
    }


def iter_aligned_training_batches(
    dataset: AlignedFrameTrainingDataset,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    drop_last: bool = False,
    collate_fn: Callable[..., dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield framework-neutral batches from an aligned policy-tick dataset."""
    if batch_size < 1:
        raise TrainingError("batch_size must be positive")
    collate = collate_fn or collate_aligned_training_samples
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        if len(indices) < batch_size and drop_last:
            continue
        samples = dataset.__getitems__(indices)
        yield collate(samples, dataset=dataset, indices=indices)


def iter_training_batches(
    dataset: LanceTrainingDataset,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    drop_last: bool = False,
    collate_fn: Callable[..., dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield framework-neutral batches from a native training dataset.

    The returned batches are plain Python containers suitable for Arrow/NumPy/JAX
    handoff, and include a ``_lineage`` block that keeps snapshot/table-version,
    row-plan, epoch-plan, row-id/frame-id, and source provenance visible.
    """
    if batch_size < 1:
        raise TrainingError("batch_size must be positive")
    collate = collate_fn or collate_training_samples
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        if len(indices) < batch_size and drop_last:
            continue
        samples = dataset.__getitems__(indices)
        yield collate(samples, dataset=dataset, indices=indices)


def collate_training_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    dataset: LanceTrainingDataset | None = None,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Collate native training samples into a framework-neutral batch dictionary."""
    if not samples:
        raise TrainingError("cannot collate an empty training batch")
    if dataset is not None and indices is None:
        raise TrainingError("indices are required when collating with dataset lineage")
    if indices is not None and len(indices) != len(samples):
        raise TrainingError("indices length must match samples length")

    materialized = [dict(sample) for sample in samples]
    media_audits: list[dict[str, Any] | None] = []
    windows_by_sample: list[dict[str, Any] | None] = []
    lineages: list[dict[str, Any]] = []
    for offset, sample in enumerate(materialized):
        media_audits.append(sample.pop("_media", None))
        windows_by_sample.append(sample.pop("windows", None))
        existing_lineage = sample.pop("_lineage", None)
        if dataset is not None:
            assert indices is not None
            lineages.append(_sample_lineage(dataset, int(indices[offset]), sample))
        elif isinstance(existing_lineage, Mapping):
            lineages.append(dict(existing_lineage))

    batch: dict[str, Any] = {}
    schema_columns: list[str] = []
    vector_schema: dict[str, Any] = {}
    for column in _ordered_sample_columns(materialized):
        values = [sample.get(column) for sample in materialized]
        if _is_numeric_vector_column(values):
            padded, mask = _pad_numeric_vectors(values)
            batch[column] = padded
            mask_column = f"{column}_mask"
            batch[mask_column] = mask
            schema_columns.extend([column, mask_column])
            vector_schema[column] = {
                "dtype": "float32",
                "shape": [len(samples), len(padded[0]) if padded else 0],
                "mask_column": mask_column,
            }
        else:
            batch[column] = values
            schema_columns.append(column)

    windows, window_schema = _collate_windows(windows_by_sample)
    if windows:
        batch["windows"] = windows
        schema_columns.append("windows")

    media = _collate_media_audits(media_audits)
    if media:
        batch["_media"] = media
    if lineages:
        batch["_lineage"] = _collate_lineage(lineages)
    batch["_schema"] = {
        "size": len(samples),
        "columns": schema_columns,
        "vectors": vector_schema,
        "windows": window_schema,
    }
    return batch


def collate_aligned_training_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    dataset: AlignedFrameTrainingDataset | None = None,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Collate aligned policy-tick samples into a framework-neutral batch."""
    if not samples:
        raise TrainingError("cannot collate an empty aligned training batch")
    if dataset is not None and indices is None:
        raise TrainingError("indices are required when collating with dataset lineage")
    if indices is not None and len(indices) != len(samples):
        raise TrainingError("indices length must match samples length")

    materialized = [dict(sample) for sample in samples]
    streams_by_sample: list[Mapping[str, Any] | None] = []
    masks_by_sample: list[Mapping[str, Any] | None] = []
    lineages: list[dict[str, Any]] = []
    for offset, sample in enumerate(materialized):
        streams = sample.pop("streams", None)
        streams_by_sample.append(streams if isinstance(streams, Mapping) else None)
        masks = sample.pop("masks", None)
        masks_by_sample.append(masks if isinstance(masks, Mapping) else None)
        existing_lineage = sample.pop("_lineage", None)
        if existing_lineage is None:
            existing_lineage = sample.pop("lineage", None)
        else:
            sample.pop("lineage", None)
        if isinstance(existing_lineage, Mapping):
            lineages.append(dict(existing_lineage))
        elif dataset is not None:
            assert indices is not None
            lineages.append(_aligned_lineage_for_dataset(dataset, int(indices[offset])))

    batch: dict[str, Any] = {}
    schema_columns: list[str] = []
    vector_schema: dict[str, Any] = {}
    for column in _ordered_sample_columns(materialized):
        values = [sample.get(column) for sample in materialized]
        if _is_numeric_vector_column(values):
            padded, mask = _pad_numeric_vectors(values)
            batch[column] = padded
            mask_column = f"{column}_mask"
            batch[mask_column] = mask
            schema_columns.extend([column, mask_column])
            vector_schema[column] = {
                "dtype": "float32",
                "shape": [len(samples), len(padded[0]) if padded else 0],
                "mask_column": mask_column,
            }
        else:
            batch[column] = values
            schema_columns.append(column)

    streams, stream_schema = _collate_aligned_streams(streams_by_sample)
    if streams:
        batch["streams"] = streams
        schema_columns.append("streams")

    masks = _collate_aligned_masks(masks_by_sample, tuple(streams))
    if masks:
        batch["masks"] = masks
        schema_columns.append("masks")

    if lineages:
        batch["_lineage"] = _collate_aligned_lineage(lineages)
    batch["_schema"] = {
        "size": len(samples),
        "columns": schema_columns,
        "vectors": vector_schema,
        "streams": stream_schema,
    }
    return batch


def collate_torch_training_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    dataset: LanceTrainingDataset | None = None,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Collate native samples and convert regular numeric fields to torch tensors."""
    _require_torch()
    import torch

    batch = collate_training_samples(samples, dataset=dataset, indices=indices)
    converted: dict[str, Any] = {}
    for key, value in batch.items():
        if key.startswith("_"):
            converted[key] = value
        elif key == "windows":
            converted[key] = {
                name: {
                    field: _torchify_regular_numeric(value, torch)
                    for field, value in window.items()
                }
                for name, window in value.items()
            }
        else:
            converted[key] = _torchify_regular_numeric(value, torch)
    return converted


def collate_torch_aligned_training_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    dataset: AlignedFrameTrainingDataset | None = None,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Collate aligned samples and convert tensor-friendly numeric fields."""
    _require_torch()
    import torch

    batch = collate_aligned_training_samples(samples, dataset=dataset, indices=indices)
    converted: dict[str, Any] = {}
    for key, value in batch.items():
        if key.startswith("_"):
            converted[key] = value
        elif key == "streams":
            converted[key] = _torchify_aligned_streams(value, torch)
        elif key == "masks":
            converted[key] = _torchify_aligned_masks(value, torch)
        else:
            converted[key] = _torchify_regular_numeric(value, torch)
    return converted


def to_torch_map_dataset(
    dataset: LanceTrainingDataset,
    *,
    include_lineage: bool = True,
):
    """Wrap a native dataset as a PyTorch map-style ``Dataset``."""
    cls = _torch_map_dataset_cls()
    return cls(dataset, include_lineage=include_lineage)


def to_torch_iterable_dataset(
    dataset: LanceTrainingDataset,
    *,
    include_lineage: bool = True,
):
    """Wrap a native dataset as a PyTorch ``IterableDataset``.

    Worker processes rebuild the Lance dataset from ``dataset.loader_config()``
    and pass PyTorch's ``worker_id``/``num_workers`` into the existing epoch-plan
    sharding logic, so one global epoch is covered exactly once.
    """
    cls = _torch_iterable_dataset_cls()
    return cls(dataset.loader_config(), include_lineage=include_lineage)


def to_torch_dataloader(
    dataset: LanceTrainingDataset,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    adapter: str = "auto",
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    include_lineage: bool = True,
    **kwargs: Any,
):
    """Build a PyTorch ``DataLoader`` with native-training collation.

    ``adapter='auto'`` uses map-style access for single-process loading and the
    iterable, worker-partitioned adapter when ``num_workers`` is positive.
    """
    _require_torch()
    from torch.utils.data import DataLoader

    if batch_size < 1:
        raise TrainingError("batch_size must be positive")
    if num_workers < 0:
        raise TrainingError("num_workers must be non-negative")
    selected = str(adapter).lower().replace("_", "-")
    if selected == "auto":
        selected = "iterable" if num_workers else "map"
    if selected in {"map", "map-style", "random-access"}:
        torch_dataset = to_torch_map_dataset(dataset, include_lineage=include_lineage)
    elif selected in {"iter", "iterable", "stream", "streaming"}:
        torch_dataset = to_torch_iterable_dataset(dataset, include_lineage=include_lineage)
    else:
        raise TrainingError("adapter must be one of auto, map, or iterable")

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_torch_training_samples,
    }
    if prefetch_factor is not None:
        if num_workers < 1:
            raise TrainingError("prefetch_factor requires num_workers > 0")
        loader_kwargs["prefetch_factor"] = prefetch_factor
    if persistent_workers:
        if num_workers < 1:
            raise TrainingError("persistent_workers requires num_workers > 0")
        loader_kwargs["persistent_workers"] = persistent_workers
    loader_kwargs.update(kwargs)
    return DataLoader(torch_dataset, **loader_kwargs)


def to_torch_aligned_dataset(
    dataset: AlignedFrameTrainingDataset,
    *,
    adapter: str = "iterable",
    include_lineage: bool = True,
):
    """Wrap an aligned policy-tick dataset as a PyTorch dataset."""
    selected = str(adapter).lower().replace("_", "-")
    if selected in {"map", "map-style", "random-access"}:
        return to_torch_aligned_map_dataset(dataset, include_lineage=include_lineage)
    if selected in {"iter", "iterable", "stream", "streaming"}:
        return to_torch_aligned_iterable_dataset(dataset, include_lineage=include_lineage)
    raise TrainingError("adapter must be one of map or iterable")


def to_torch_aligned_map_dataset(
    dataset: AlignedFrameTrainingDataset,
    *,
    include_lineage: bool = True,
):
    """Wrap an aligned policy-tick dataset as a PyTorch map-style ``Dataset``."""
    cls = _torch_aligned_map_dataset_cls()
    return cls(dataset, include_lineage=include_lineage)


def to_torch_aligned_iterable_dataset(
    dataset: AlignedFrameTrainingDataset,
    *,
    include_lineage: bool = True,
):
    """Wrap an aligned policy-tick dataset as a PyTorch ``IterableDataset``.

    Worker processes rebuild the aligned dataset from ``dataset.loader_config()``
    and pass PyTorch's ``worker_id``/``num_workers`` into the existing epoch-plan
    sharding logic, so one global tick epoch is covered exactly once.
    """
    cls = _torch_aligned_iterable_dataset_cls()
    return cls(dataset.loader_config(), include_lineage=include_lineage)


def to_torch_aligned_dataloader(
    dataset: AlignedFrameTrainingDataset,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    adapter: str = "auto",
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    include_lineage: bool = True,
    **kwargs: Any,
):
    """Build a PyTorch ``DataLoader`` for aligned policy-tick training."""
    _require_torch()
    from torch.utils.data import DataLoader

    if batch_size < 1:
        raise TrainingError("batch_size must be positive")
    if num_workers < 0:
        raise TrainingError("num_workers must be non-negative")
    selected = str(adapter).lower().replace("_", "-")
    if selected == "auto":
        selected = "iterable" if num_workers else "map"
    if selected in {"map", "map-style", "random-access"}:
        torch_dataset = to_torch_aligned_map_dataset(
            dataset,
            include_lineage=include_lineage,
        )
    elif selected in {"iter", "iterable", "stream", "streaming"}:
        torch_dataset = to_torch_aligned_iterable_dataset(
            dataset,
            include_lineage=include_lineage,
        )
    else:
        raise TrainingError("adapter must be one of auto, map, or iterable")

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_torch_aligned_training_samples,
    }
    if prefetch_factor is not None:
        if num_workers < 1:
            raise TrainingError("prefetch_factor requires num_workers > 0")
        loader_kwargs["prefetch_factor"] = prefetch_factor
    if persistent_workers:
        if num_workers < 1:
            raise TrainingError("persistent_workers requires num_workers > 0")
        loader_kwargs["persistent_workers"] = persistent_workers
    loader_kwargs.update(kwargs)
    return DataLoader(torch_dataset, **loader_kwargs)


def _sample_with_lineage(
    dataset: LanceTrainingDataset,
    index: int,
    *,
    include_lineage: bool,
) -> dict[str, Any]:
    normalized = index + len(dataset) if index < 0 else index
    sample = dict(dataset[index])
    if include_lineage:
        sample["_lineage"] = _sample_lineage(dataset, normalized, sample)
    return sample


def _sample_lineage(
    dataset: LanceTrainingDataset,
    index: int,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    if index < 0 or index >= len(dataset):
        raise TrainingError(f"sample index out of range: {index}")
    plan_index = int(dataset.epoch_plan.sample_indices[index])
    row_id = dataset.row_plan.row_ids[plan_index]
    frame_id = dataset.row_plan.frame_ids[plan_index]
    ref = dataset._frame_refs[index]
    obs = ref.observation
    return {
        "snapshot_id": dataset.manifest.dataset_id,
        "dataset_id": dataset.manifest.dataset_id,
        "snapshot_name": dataset.manifest.snapshot_name,
        "table_versions": list(dataset.manifest.table_versions),
        "row_plan_id": dataset.row_plan.plan_id,
        "epoch_plan_id": dataset.epoch_plan.plan_id,
        "backend": dict(dataset.manifest.backend),
        "epoch_backend": dataset.epoch_plan.backend.to_dict(),
        "epoch": dataset.epoch_plan.epoch,
        "worker_id": dataset.epoch_plan.worker_id,
        "num_workers": dataset.epoch_plan.num_workers,
        "sample_index": index,
        "plan_index": plan_index,
        "row_id": row_id,
        "frame_id": frame_id,
        "observation_id": sample.get("observation_id") or obs.get("observation_id"),
        "episode_id": sample.get("episode_id") or obs.get("episode_id") or ref.episode.episode_id,
        "frame_index": sample.get("frame_index", ref.frame_index),
        "source": {
            "run_id": sample.get("run_id") or obs.get("run_id") or ref.episode.scenario["run_id"],
            "raw_uri": sample.get("raw_uri") or obs.get("raw_uri"),
            "raw_channel": sample.get("raw_channel") or obs.get("raw_channel"),
            "raw_sequence": sample.get("raw_sequence") or obs.get("raw_sequence"),
        },
    }


def _connection_loader_config(lake: Lake) -> dict[str, Any]:
    """Secret-free connection settings needed to reopen a lake in workers."""
    spec = getattr(lake, "connection_spec", None)
    if spec is None:
        return {}
    result: dict[str, Any] = {
        "kind": getattr(spec, "kind", None),
        "display_uri": getattr(spec, "display_uri", lake.uri),
        "auth_refs": {
            key: value
            for key, value in getattr(spec, "auth_refs", {}).items()
            if value
        },
    }
    if spec.kind == "lancedb_remote_db":
        kwargs = dict(getattr(spec, "lancedb_connect_kwargs", {}) or {})
        remote = {
            "remote_auth_ref": result["auth_refs"].get("remote"),
            "region": kwargs.get("region"),
            "host_override": kwargs.get("host_override"),
            "client_config": kwargs.get("client_config"),
        }
        result["remote"] = {key: value for key, value in remote.items() if value is not None}
    elif getattr(spec, "namespace_client_impl", None):
        result["namespace"] = {
            "namespace_client_impl": spec.namespace_client_impl,
            "namespace_client_properties": _safe_worker_namespace_properties(
                getattr(spec, "namespace_client_properties", {}) or {}
            ),
            "namespace_client_pushdown_operations": list(
                getattr(spec, "namespace_client_pushdown_operations", ()) or ()
            ),
            "namespace_auth_ref": result["auth_refs"].get("namespace"),
            "storage_auth_ref": result["auth_refs"].get("storage"),
        }
    return _jsonable(result)


def _open_lake_from_loader_config(config: Mapping[str, Any]) -> Lake:
    connection = config.get("connection")
    if not isinstance(connection, Mapping):
        return Lake.open(config["lake_uri"])
    if connection.get("kind") == "lancedb_remote_db":
        remote = connection.get("remote")
        remote = remote if isinstance(remote, Mapping) else {}
        return Lake.open(
            config["lake_uri"],
            remote_auth_ref=remote.get("remote_auth_ref"),
            region=remote.get("region"),
            host_override=remote.get("host_override"),
            client_config=remote.get("client_config"),
        )
    namespace = connection.get("namespace")
    if isinstance(namespace, Mapping) and namespace.get("namespace_client_impl"):
        return Lake.open(
            config.get("lake_uri"),
            namespace_auth_ref=namespace.get("namespace_auth_ref"),
            storage_auth_ref=namespace.get("storage_auth_ref"),
            namespace_client_impl=namespace.get("namespace_client_impl"),
            namespace_client_properties=namespace.get("namespace_client_properties"),
            namespace_client_pushdown_operations=namespace.get(
                "namespace_client_pushdown_operations"
            ),
        )
    return Lake.open(config["lake_uri"])


def _safe_worker_namespace_properties(properties: Mapping[str, Any]) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in properties.items():
        name = str(key)
        lowered = name.lower()
        if lowered.startswith("header.authorization"):
            continue
        if any(token in lowered for token in ("password", "secret", "token", "api_key")):
            continue
        if lowered.startswith("credential_vendor."):
            continue
        safe[name] = str(value)
    return safe


def _dataset_from_loader_config(
    config: Mapping[str, Any],
    *,
    worker_id: int | None = None,
    num_workers: int | None = None,
    resume_from: int | None = None,
) -> LanceTrainingDataset:
    lake = _open_lake_from_loader_config(config)
    return training_dataset(
        lake,
        str(config["snapshot_name"]),
        columns=config.get("columns"),
        filters=config.get("filters"),
        shuffle=bool(config.get("shuffle", False)),
        shuffle_seed=config.get("shuffle_seed"),
        epoch=int(config.get("epoch", 0)),
        worker_id=int(config.get("worker_id", 0) if worker_id is None else worker_id),
        num_workers=int(config.get("num_workers", 1) if num_workers is None else num_workers),
        resume_from=int(config.get("resume_from", 0) if resume_from is None else resume_from),
        time_windows=config.get("time_windows"),
        media=config.get("media", DEFAULT_MEDIA_POLICY),
        decoder=str(config.get("decoder", "auto")),
        media_cache=str(config.get("media_cache", DEFAULT_MEDIA_CACHE_POLICY)),
        media_cache_size=int(config.get("media_cache_size", DEFAULT_MEDIA_CACHE_SIZE)),
        backend=str(config.get("backend", DEFAULT_TRAINING_BACKEND)),
        cache_policy=str(config.get("cache_policy", DEFAULT_ENTERPRISE_CACHE_POLICY)),
        prewarm=bool(config.get("prewarm", False)),
        prewarm_options=config.get("prewarm_options"),
        allow_fallback=bool(config.get("allow_fallback", False)),
        fallback=config.get("fallback"),
    )


def _aligned_sample_with_lineage(
    dataset: AlignedFrameTrainingDataset,
    index: int,
    *,
    include_lineage: bool,
) -> dict[str, Any]:
    normalized = index + len(dataset) if index < 0 else index
    sample = dict(dataset[index])
    if include_lineage:
        lineage = sample.get("lineage")
        sample["_lineage"] = (
            dict(lineage)
            if isinstance(lineage, Mapping)
            else _aligned_lineage_for_dataset(dataset, normalized)
        )
    return sample


def _aligned_lineage_for_dataset(
    dataset: AlignedFrameTrainingDataset,
    index: int,
) -> dict[str, Any]:
    if index < 0 or index >= len(dataset):
        raise TrainingError(f"aligned sample index out of range: {index}")
    plan_index = int(dataset.epoch_plan.sample_indices[index])
    tick_index = int(dataset.tick_plan.tick_indices[plan_index])
    sample = _aligned_sample_for_tick(
        dataset._job,
        tick_index,
        dataset._rows_by_tick.get(tick_index, {}),
        streams=dataset.streams,
        columns=("lineage",),
        quality_policy=dataset.quality_policy,
        manifest=dataset.manifest,
        tick_plan=dataset.tick_plan,
        epoch_plan=dataset.epoch_plan,
        sample_index=index,
        plan_index=plan_index,
        feature_resolver=dataset._feature_resolver,
    )
    return dict(sample["lineage"])


def _aligned_dataset_from_loader_config(
    config: Mapping[str, Any],
    *,
    worker_id: int | None = None,
    num_workers: int | None = None,
    resume_from: int | None = None,
) -> AlignedFrameTrainingDataset:
    lake = _open_lake_from_loader_config(config)
    return aligned_training_dataset(
        lake,
        alignment_id=str(config["alignment_id"]),
        streams=config.get("streams"),
        columns=config.get("columns"),
        statuses=config.get("statuses"),
        require_streams=config.get("require_streams", False),
        allow_missing=bool(config.get("allow_missing", True)),
        min_confidence=config.get("min_confidence"),
        shuffle=bool(config.get("shuffle", False)),
        shuffle_seed=config.get("shuffle_seed"),
        epoch=int(config.get("epoch", 0)),
        worker_id=int(config.get("worker_id", 0) if worker_id is None else worker_id),
        num_workers=int(config.get("num_workers", 1) if num_workers is None else num_workers),
        resume_from=int(config.get("resume_from", 0) if resume_from is None else resume_from),
        features=config.get("features", DEFAULT_ALIGNED_FEATURE_POLICY),
        decoder=str(config.get("decoder", "auto")),
        feature_cache=str(config.get("feature_cache", DEFAULT_MEDIA_CACHE_POLICY)),
        feature_cache_size=int(config.get("feature_cache_size", DEFAULT_MEDIA_CACHE_SIZE)),
        backend=str(config.get("backend", DEFAULT_TRAINING_BACKEND)),
        cache_policy=str(config.get("cache_policy", DEFAULT_ENTERPRISE_CACHE_POLICY)),
        prewarm=bool(config.get("prewarm", False)),
        prewarm_options=config.get("prewarm_options"),
        allow_fallback=bool(config.get("allow_fallback", False)),
        fallback=config.get("fallback"),
    )


_TORCH_MAP_DATASET_CLS: Any = None
_TORCH_ITERABLE_DATASET_CLS: Any = None
_TORCH_ALIGNED_MAP_DATASET_CLS: Any = None
_TORCH_ALIGNED_ITERABLE_DATASET_CLS: Any = None


def _torch_map_dataset_cls():
    global _TORCH_MAP_DATASET_CLS
    _require_torch()
    if _TORCH_MAP_DATASET_CLS is not None:
        return _TORCH_MAP_DATASET_CLS

    from torch.utils.data import Dataset

    class TorchMapTrainingDataset(Dataset):
        def __init__(
            self,
            dataset_or_config: LanceTrainingDataset | Mapping[str, Any],
            *,
            include_lineage: bool = True,
        ) -> None:
            if isinstance(dataset_or_config, LanceTrainingDataset):
                self._dataset = dataset_or_config
                self._config = dataset_or_config.loader_config()
            else:
                self._dataset = None
                self._config = dict(dataset_or_config)
            self.include_lineage = include_lineage
            self.collate_fn = collate_torch_training_samples

        @property
        def dataset(self) -> LanceTrainingDataset:
            if self._dataset is None:
                self._dataset = _dataset_from_loader_config(self._config)
            return self._dataset

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return _sample_with_lineage(
                self.dataset,
                int(index),
                include_lineage=self.include_lineage,
            )

        def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
            return [self[index] for index in indices]

        def __reduce__(self):
            return (
                _rebuild_torch_map_training_dataset,
                (self._config, self.include_lineage),
            )

    TorchMapTrainingDataset.__module__ = __name__
    globals()["TorchMapTrainingDataset"] = TorchMapTrainingDataset
    _TORCH_MAP_DATASET_CLS = TorchMapTrainingDataset
    return _TORCH_MAP_DATASET_CLS


def _torch_iterable_dataset_cls():
    global _TORCH_ITERABLE_DATASET_CLS
    _require_torch()
    if _TORCH_ITERABLE_DATASET_CLS is not None:
        return _TORCH_ITERABLE_DATASET_CLS

    import torch
    from torch.utils.data import IterableDataset

    class TorchIterableTrainingDataset(IterableDataset):
        def __init__(
            self,
            config: Mapping[str, Any],
            *,
            include_lineage: bool = True,
        ) -> None:
            self._config = dict(config)
            self.include_lineage = include_lineage
            self.collate_fn = collate_torch_training_samples

        def __iter__(self) -> Iterator[dict[str, Any]]:
            worker = torch.utils.data.get_worker_info()
            if worker is None:
                worker_id = int(self._config.get("worker_id", 0))
                num_workers = int(self._config.get("num_workers", 1))
            else:
                worker_id = int(worker.id)
                num_workers = int(worker.num_workers)
            dataset = _dataset_from_loader_config(
                self._config,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            for index in range(len(dataset)):
                yield _sample_with_lineage(
                    dataset,
                    index,
                    include_lineage=self.include_lineage,
                )

        def __len__(self) -> int:
            return len(_dataset_from_loader_config(self._config))

        def __reduce__(self):
            return (
                _rebuild_torch_iterable_training_dataset,
                (self._config, self.include_lineage),
            )

    TorchIterableTrainingDataset.__module__ = __name__
    globals()["TorchIterableTrainingDataset"] = TorchIterableTrainingDataset
    _TORCH_ITERABLE_DATASET_CLS = TorchIterableTrainingDataset
    return _TORCH_ITERABLE_DATASET_CLS


def _torch_aligned_map_dataset_cls():
    global _TORCH_ALIGNED_MAP_DATASET_CLS
    _require_torch()
    if _TORCH_ALIGNED_MAP_DATASET_CLS is not None:
        return _TORCH_ALIGNED_MAP_DATASET_CLS

    from torch.utils.data import Dataset

    class TorchMapAlignedTrainingDataset(Dataset):
        def __init__(
            self,
            dataset_or_config: AlignedFrameTrainingDataset | Mapping[str, Any],
            *,
            include_lineage: bool = True,
        ) -> None:
            if isinstance(dataset_or_config, AlignedFrameTrainingDataset):
                self._dataset = dataset_or_config
                self._config = dataset_or_config.loader_config()
            else:
                self._dataset = None
                self._config = dict(dataset_or_config)
            self.include_lineage = include_lineage
            self.collate_fn = collate_torch_aligned_training_samples

        @property
        def dataset(self) -> AlignedFrameTrainingDataset:
            if self._dataset is None:
                self._dataset = _aligned_dataset_from_loader_config(self._config)
            return self._dataset

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return _aligned_sample_with_lineage(
                self.dataset,
                int(index),
                include_lineage=self.include_lineage,
            )

        def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
            return [self[index] for index in indices]

        def __reduce__(self):
            return (
                _rebuild_torch_aligned_map_training_dataset,
                (self._config, self.include_lineage),
            )

    TorchMapAlignedTrainingDataset.__module__ = __name__
    globals()["TorchMapAlignedTrainingDataset"] = TorchMapAlignedTrainingDataset
    _TORCH_ALIGNED_MAP_DATASET_CLS = TorchMapAlignedTrainingDataset
    return _TORCH_ALIGNED_MAP_DATASET_CLS


def _torch_aligned_iterable_dataset_cls():
    global _TORCH_ALIGNED_ITERABLE_DATASET_CLS
    _require_torch()
    if _TORCH_ALIGNED_ITERABLE_DATASET_CLS is not None:
        return _TORCH_ALIGNED_ITERABLE_DATASET_CLS

    import torch
    from torch.utils.data import IterableDataset

    class TorchIterableAlignedTrainingDataset(IterableDataset):
        def __init__(
            self,
            config: Mapping[str, Any],
            *,
            include_lineage: bool = True,
        ) -> None:
            self._config = dict(config)
            self.include_lineage = include_lineage
            self.collate_fn = collate_torch_aligned_training_samples

        def __iter__(self) -> Iterator[dict[str, Any]]:
            worker = torch.utils.data.get_worker_info()
            if worker is None:
                worker_id = int(self._config.get("worker_id", 0))
                num_workers = int(self._config.get("num_workers", 1))
            else:
                worker_id = int(worker.id)
                num_workers = int(worker.num_workers)
            dataset = _aligned_dataset_from_loader_config(
                self._config,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            for index in range(len(dataset)):
                yield _aligned_sample_with_lineage(
                    dataset,
                    index,
                    include_lineage=self.include_lineage,
                )

        def __len__(self) -> int:
            return len(_aligned_dataset_from_loader_config(self._config))

        def __reduce__(self):
            return (
                _rebuild_torch_aligned_iterable_training_dataset,
                (self._config, self.include_lineage),
            )

    TorchIterableAlignedTrainingDataset.__module__ = __name__
    globals()["TorchIterableAlignedTrainingDataset"] = TorchIterableAlignedTrainingDataset
    _TORCH_ALIGNED_ITERABLE_DATASET_CLS = TorchIterableAlignedTrainingDataset
    return _TORCH_ALIGNED_ITERABLE_DATASET_CLS


def _rebuild_torch_map_training_dataset(
    config: Mapping[str, Any],
    include_lineage: bool,
):
    return _torch_map_dataset_cls()(config, include_lineage=include_lineage)


def _rebuild_torch_iterable_training_dataset(
    config: Mapping[str, Any],
    include_lineage: bool,
):
    return _torch_iterable_dataset_cls()(config, include_lineage=include_lineage)


def _rebuild_torch_aligned_map_training_dataset(
    config: Mapping[str, Any],
    include_lineage: bool,
):
    return _torch_aligned_map_dataset_cls()(config, include_lineage=include_lineage)


def _rebuild_torch_aligned_iterable_training_dataset(
    config: Mapping[str, Any],
    include_lineage: bool,
):
    return _torch_aligned_iterable_dataset_cls()(config, include_lineage=include_lineage)


def _require_torch() -> None:
    if not torch_available():
        raise TrainingError(TORCH_INSTALL_GUIDANCE)


def _ordered_sample_columns(samples: Sequence[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        for column in sample:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return columns


def _is_numeric_vector_column(values: Sequence[Any]) -> bool:
    return any(_is_numeric_sequence(value) for value in values)


def _is_numeric_sequence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes, bytearray)):
        return False
    return all(_is_numeric_scalar(item) for item in value)


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, bool)


def _pad_numeric_vectors(values: Sequence[Any]) -> tuple[list[list[float]], list[list[bool]]]:
    numeric = [list(value) if _is_numeric_sequence(value) else [] for value in values]
    width = max((len(value) for value in numeric), default=0)
    padded: list[list[float]] = []
    mask: list[list[bool]] = []
    for value in numeric:
        row = [float(item) for item in value] + [0.0] * (width - len(value))
        row_mask = [True] * len(value) + [False] * (width - len(value))
        padded.append(row)
        mask.append(row_mask)
    return padded, mask


def _collate_aligned_streams(
    streams_by_sample: Sequence[Mapping[str, Any] | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stream_names: list[str] = []
    for streams in streams_by_sample:
        if not streams:
            continue
        for stream in streams:
            if stream not in stream_names:
                stream_names.append(str(stream))
    if not stream_names:
        return {}, {}

    collated: dict[str, Any] = {}
    schema: dict[str, Any] = {}
    preferred_fields = [
        "status",
        "interpolation",
        "observation_id",
        "timestamp_ns",
        "source_observation_ids",
        "source_row_ids",
        "source_timestamp_ns",
        "source_time_ns",
        "receive_time_ns",
        "latency_ns",
        "error_ns",
        "absolute_error_ns",
        "confidence",
        "value",
        "quality_flags",
        "aligned_frame_id",
        "transform_id",
        "feature",
    ]
    for stream in stream_names:
        per_stream = [
            dict(streams.get(stream, {})) if streams and isinstance(streams.get(stream), Mapping) else {}
            for streams in streams_by_sample
        ]
        field_names: list[str] = []
        for field_name in preferred_fields:
            if any(field_name in sample for sample in per_stream):
                field_names.append(field_name)
        for sample in per_stream:
            for field_name in sample:
                if field_name != "stream" and field_name not in field_names:
                    field_names.append(field_name)

        stream_batch: dict[str, Any] = {
            "stream": stream,
            "present": [bool(sample) for sample in per_stream],
        }
        stream_schema: dict[str, Any] = {}
        for field_name in field_names:
            values = [sample.get(field_name) for sample in per_stream]
            if field_name == "value" and _is_numeric_vector_column(values):
                padded, value_mask = _pad_numeric_vectors(values)
                stream_batch["value"] = padded
                stream_batch["value_mask"] = value_mask
                stream_batch["value_present"] = [value is not None for value in values]
                stream_schema["value"] = {
                    "dtype": "float32",
                    "shape": [len(per_stream), len(padded[0]) if padded else 0],
                    "mask_column": "value_mask",
                    "present_column": "value_present",
                }
            elif field_name in {"source_observation_ids", "source_row_ids", "quality_flags"}:
                stream_batch[field_name] = [list(value or []) for value in values]
            elif field_name == "feature":
                stream_batch[field_name] = [dict(value) if isinstance(value, Mapping) else value for value in values]
            else:
                stream_batch[field_name] = values

        observation_sources = stream_batch.get("source_observation_ids", [[] for _ in per_stream])
        row_sources = stream_batch.get("source_row_ids", [[] for _ in per_stream])
        source_lengths = [
            max(len(observation_sources[index]), len(row_sources[index]))
            for index in range(len(per_stream))
        ]
        source_width = max(source_lengths, default=0)
        stream_batch["source_count"] = source_lengths
        stream_batch["source_mask"] = [
            [offset < source_lengths[index] for offset in range(source_width)]
            for index in range(len(per_stream))
        ]
        stream_schema["sources"] = {
            "shape": [len(per_stream), source_width],
            "mask_column": "source_mask",
            "count_column": "source_count",
        }
        collated[stream] = stream_batch
        schema[stream] = stream_schema
    return collated, schema


def _collate_aligned_masks(
    masks_by_sample: Sequence[Mapping[str, Any] | None],
    stream_names: tuple[str, ...],
) -> dict[str, Any]:
    mask_names: list[str] = []
    for masks in masks_by_sample:
        if not masks:
            continue
        for mask_name in masks:
            if mask_name not in mask_names:
                mask_names.append(str(mask_name))
    if not mask_names:
        return {}
    collated: dict[str, Any] = {}
    for mask_name in mask_names:
        by_stream: dict[str, list[bool]] = {}
        for stream in stream_names:
            by_stream[stream] = [
                bool((masks or {}).get(mask_name, {}).get(stream, False))
                for masks in masks_by_sample
            ]
        collated[mask_name] = by_stream
    return collated


def _collate_windows(
    windows_by_sample: Sequence[Mapping[str, Any] | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    names: list[str] = []
    for windows in windows_by_sample:
        if not windows:
            continue
        for name in windows:
            if name not in names:
                names.append(name)

    collated: dict[str, Any] = {}
    schema: dict[str, Any] = {}
    for name in names:
        per_sample = [
            list(windows.get(name, [])) if windows and name in windows else []
            for windows in windows_by_sample
        ]
        length = max((len(items) for items in per_sample), default=0)
        value_width = _window_value_width(per_sample)
        value_is_vector = value_width > 0
        window_batch = {
            "delta_s": [],
            "frame_index": [],
            "timestamp_ns": [],
            "value": [],
            "mask": [],
        }
        for items in per_sample:
            delta_row: list[float] = []
            frame_row: list[int] = []
            timestamp_row: list[int] = []
            value_row: list[Any] = []
            mask_row: list[bool] = []
            for offset in range(length):
                item = items[offset] if offset < len(items) else None
                value = item.get("value") if isinstance(item, Mapping) else None
                valid = isinstance(item, Mapping) and value is not None
                delta_row.append(float(item.get("delta_s", 0.0)) if valid else 0.0)
                frame_row.append(int(item.get("frame_index", -1)) if valid else -1)
                timestamp_row.append(int(item.get("timestamp_ns", 0)) if valid else 0)
                if value_is_vector:
                    value_row.append(_pad_window_vector(value, value_width))
                else:
                    value_row.append(value if valid else None)
                mask_row.append(valid)
            window_batch["delta_s"].append(delta_row)
            window_batch["frame_index"].append(frame_row)
            window_batch["timestamp_ns"].append(timestamp_row)
            window_batch["value"].append(value_row)
            window_batch["mask"].append(mask_row)
        collated[name] = window_batch
        schema[name] = {
            "length": length,
            "value_shape": [value_width] if value_is_vector else [],
            "mask_column": "mask",
        }
    return collated, schema


def _window_value_width(per_sample: Sequence[Sequence[Mapping[str, Any]]]) -> int:
    width = 0
    for items in per_sample:
        for item in items:
            value = item.get("value") if isinstance(item, Mapping) else None
            if _is_numeric_sequence(value):
                width = max(width, len(value))
    return width


def _pad_window_vector(value: Any, width: int) -> list[float]:
    if not _is_numeric_sequence(value):
        return [0.0] * width
    row = [float(item) for item in value]
    return row + [0.0] * (width - len(row))


def _collate_media_audits(media_audits: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    present = [dict(audit) for audit in media_audits if audit]
    if not present:
        return {}
    first = present[0]
    fields: dict[str, list[Any]] = {}
    field_names: list[str] = []
    for audit in media_audits:
        for field_name in (audit or {}).get("fields", {}):
            if field_name not in field_names:
                field_names.append(field_name)
    for field_name in field_names:
        fields[field_name] = [
            (audit or {}).get("fields", {}).get(field_name) for audit in media_audits
        ]
    return {
        "policy": first.get("policy"),
        "decoder": first.get("decoder"),
        "cache": first.get("cache"),
        "fields": fields,
    }


def _collate_lineage(lineages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = dict(lineages[0])
    return {
        "snapshot_id": first.get("snapshot_id"),
        "dataset_id": first.get("dataset_id"),
        "snapshot_name": first.get("snapshot_name"),
        "table_versions": first.get("table_versions", []),
        "row_plan_id": first.get("row_plan_id"),
        "epoch_plan_id": first.get("epoch_plan_id"),
        "backend": first.get("backend"),
        "epoch_backend": first.get("epoch_backend"),
        "epoch": first.get("epoch"),
        "worker": {
            "id": first.get("worker_id"),
            "num_workers": first.get("num_workers"),
        },
        "sample_indices": [lineage.get("sample_index") for lineage in lineages],
        "plan_indices": [lineage.get("plan_index") for lineage in lineages],
        "row_ids": [lineage.get("row_id") for lineage in lineages],
        "frame_ids": [lineage.get("frame_id") for lineage in lineages],
        "observation_ids": [lineage.get("observation_id") for lineage in lineages],
        "episode_ids": [lineage.get("episode_id") for lineage in lineages],
        "frame_indices": [lineage.get("frame_index") for lineage in lineages],
        "source": [lineage.get("source") for lineage in lineages],
    }


def _collate_aligned_lineage(lineages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = dict(lineages[0])
    stream_names = _aligned_lineage_stream_names(lineages)
    return {
        "alignment_job": first.get("alignment_job"),
        "backend": first.get("backend"),
        "epoch_backend": first.get("epoch_backend"),
        "transform_id": first.get("transform_id"),
        "transform_ids": [lineage.get("transform_id") for lineage in lineages],
        "input_table_versions": first.get("input_table_versions", []),
        "read_table_versions": first.get("read_table_versions", []),
        "tick_plan_id": first.get("tick_plan_id"),
        "epoch_plan_id": first.get("epoch_plan_id"),
        "epoch": first.get("epoch"),
        "worker": {
            "id": first.get("worker_id"),
            "num_workers": first.get("num_workers"),
        },
        "sample_indices": [lineage.get("sample_index") for lineage in lineages],
        "plan_indices": [lineage.get("plan_index") for lineage in lineages],
        "tick_indices": [lineage.get("tick_index") for lineage in lineages],
        "quality_policy": first.get("quality_policy"),
        "aligned_frame_ids": {
            stream: [
                (lineage.get("aligned_frame_ids") or {}).get(stream)
                for lineage in lineages
            ]
            for stream in stream_names
        },
        "source_observation_ids": {
            stream: [
                list((lineage.get("source_observation_ids") or {}).get(stream) or [])
                for lineage in lineages
            ]
            for stream in stream_names
        },
        "source_row_ids": {
            stream: [
                list((lineage.get("source_row_ids") or {}).get(stream) or [])
                for lineage in lineages
            ]
            for stream in stream_names
        },
        "features": [lineage.get("features") for lineage in lineages],
    }


def _aligned_lineage_stream_names(lineages: Sequence[Mapping[str, Any]]) -> list[str]:
    stream_names: list[str] = []
    for lineage in lineages:
        for key in ("aligned_frame_ids", "source_observation_ids", "source_row_ids"):
            for stream in lineage.get(key) or {}:
                if stream not in stream_names:
                    stream_names.append(str(stream))
    return stream_names


def _torchify_regular_numeric(value: Any, torch: Any) -> Any:
    if not isinstance(value, list) or not value:
        return value
    if not _regular_numeric_nested(value):
        return value
    dtype = _torch_dtype(value, torch)
    try:
        return torch.tensor(value, dtype=dtype)
    except (TypeError, ValueError):
        return value


def _regular_numeric_nested(value: Any) -> bool:
    if isinstance(value, list):
        if not value:
            return True
        lengths = [len(item) for item in value if isinstance(item, list)]
        if lengths and any(not isinstance(item, list) for item in value):
            return False
        if lengths and any(length != lengths[0] for length in lengths):
            return False
        return all(_regular_numeric_nested(item) for item in value)
    return isinstance(value, (int, float, bool)) and value is not None


def _torch_dtype(value: Any, torch: Any) -> Any:
    leaves = list(_numeric_leaves(value))
    if all(isinstance(item, bool) for item in leaves):
        return torch.bool
    if all(isinstance(item, int) and not isinstance(item, bool) for item in leaves):
        return torch.int64
    return torch.float32


def _numeric_leaves(value: Any) -> Iterator[Any]:
    if isinstance(value, list):
        for item in value:
            yield from _numeric_leaves(item)
    else:
        yield value


def _torchify_aligned_streams(streams: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    preserved = {
        "source_observation_ids",
        "source_row_ids",
        "quality_flags",
        "feature",
    }
    converted: dict[str, Any] = {}
    for stream, batch in streams.items():
        if not isinstance(batch, Mapping):
            converted[stream] = batch
            continue
        stream_batch: dict[str, Any] = {}
        for field_name, value in batch.items():
            if field_name in preserved:
                stream_batch[field_name] = value
            else:
                stream_batch[field_name] = _torchify_regular_numeric(value, torch)
        converted[stream] = stream_batch
    return converted


def _torchify_aligned_masks(masks: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for mask_name, by_stream in masks.items():
        if not isinstance(by_stream, Mapping):
            converted[mask_name] = by_stream
            continue
        converted[mask_name] = {
            stream: torch.tensor(values, dtype=torch.bool)
            if isinstance(values, list)
            else values
            for stream, values in by_stream.items()
        }
    return converted


def torch_available() -> bool:
    """Whether PyTorch can be imported in this environment."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _training_snapshot_context(
    lake: Lake,
    snapshot_name: str,
    *,
    remote_safe: bool,
    observation_columns: Sequence[str] | None = None,
) -> _SnapshotContext:
    """Build the snapshot context used by native training datasets."""
    if not remote_safe:
        # BUG-06: read only the snapshot's referenced observations, projected to the
        # columns this dataset actually uses, and streamed -- never the whole corpus
        # eagerly materialized to Python. Blob bytes hydrate lazily by id through the
        # media resolver; the projected payload_blob is only its size descriptor.
        return _snapshot_context(
            lake,
            snapshot_name,
            include_payload_blobs=False,
            include_video_encoding_blobs=False,
            observation_columns=observation_columns,
            scope_observations_to_scenarios=True,
        )
    return _remote_training_snapshot_context(lake, snapshot_name)


def _remote_training_snapshot_context(lake: Lake, snapshot_name: str) -> _SnapshotContext:
    row = _latest_remote_snapshot_row(lake, snapshot_name)
    query_spec = json.loads(row["query_spec"] or "{}")
    split_payload = json.loads(row["split"] or "{}")
    versions = {tv["table"]: int(tv["version"]) for tv in row["table_versions"]}

    scenario_ids = tuple(sorted(query_spec.get("scenario_ids", [])))
    scenario_rows = _scan_remote_rows_by_values(
        lake,
        "scenarios",
        versions.get("scenarios"),
        _REMOTE_SCENARIO_COLUMNS,
        "scenario_id",
        scenario_ids,
        snapshot_id=row["dataset_id"],
    )
    scenarios = {scenario["scenario_id"]: scenario for scenario in scenario_rows}
    missing_scenarios = [sid for sid in scenario_ids if sid not in scenarios]
    if missing_scenarios:
        raise TrainingError(
            f"snapshot {snapshot_name!r} references scenarios not present at the "
            f"pinned version: {missing_scenarios}"
        )

    observation_rows = _remote_observation_rows_for_scenarios(
        lake,
        versions.get("observations"),
        scenario_rows,
        snapshot_id=row["dataset_id"],
    )
    observations = {obs["observation_id"]: obs for obs in observation_rows}
    missing_observations = _missing_snapshot_observations(scenario_rows, observations)
    if missing_observations:
        raise TrainingError(
            f"snapshot {snapshot_name!r} references observations not present at "
            f"the pinned version: {missing_observations}"
        )

    episode_ids = sorted(
        {
            str(obs["episode_id"])
            for obs in observation_rows
            if obs.get("episode_id")
        }
    )
    episode_rows = (
        _scan_remote_rows_by_values(
            lake,
            "episodes",
            versions["episodes"],
            _REMOTE_EPISODE_COLUMNS,
            "episode_id",
            episode_ids,
            snapshot_id=row["dataset_id"],
        )
        if "episodes" in versions and episode_ids
        else []
    )
    episodes = {episode["episode_id"]: episode for episode in episode_rows}

    run_ids = sorted(
        {
            str(row["run_id"])
            for row in (*scenario_rows, *observation_rows, *episode_rows)
            if row.get("run_id")
        }
    )
    run_rows = _scan_remote_rows_by_values(
        lake,
        "runs",
        versions.get("runs"),
        _REMOTE_RUN_COLUMNS,
        "run_id",
        run_ids,
        snapshot_id=row["dataset_id"],
    )

    video_rows = (
        _scan_remote_rows_by_values(
            lake,
            "videos",
            versions["videos"],
            _REMOTE_VIDEO_COLUMNS,
            "episode_id",
            episode_ids,
            snapshot_id=row["dataset_id"],
        )
        if "videos" in versions and episode_ids
        else []
    )
    video_encoding_rows = (
        _scan_remote_rows_by_values(
            lake,
            "video_encodings",
            versions["video_encodings"],
            _REMOTE_VIDEO_ENCODING_COLUMNS,
            "episode_id",
            episode_ids,
            snapshot_id=row["dataset_id"],
        )
        if "video_encodings" in versions and episode_ids
        else []
    )

    return _SnapshotContext(
        row=row,
        dataset_id=row["dataset_id"],
        snapshot_name=row["name"],
        scenario_ids=scenario_ids,
        split_assignments=dict(split_payload.get("assignments", {})),
        table_versions=tuple(
            {
                "table": tv["table"],
                "version": int(tv["version"]),
                "tag": tv.get("tag") or "",
            }
            for tv in row["table_versions"]
        ),
        scenarios=scenarios,
        episodes=episodes,
        observations=observations,
        videos={video["video_id"]: video for video in video_rows},
        video_encodings={item["encoding_id"]: item for item in video_encoding_rows},
        runs={run["run_id"]: run for run in run_rows},
        payload_blobs={},
        video_encoding_blobs={},
    )


def _latest_remote_snapshot_row(lake: Lake, name: str) -> dict[str, Any]:
    rows = _scan_remote_table_rows(
        lake,
        "dataset_snapshots",
        None,
        _REMOTE_DATASET_SNAPSHOT_COLUMNS,
        _sql_predicate("name", name),
    )
    if not rows:
        raise TrainingError(f"no snapshot named {name!r} in {lake.uri}")
    return max(rows, key=lambda row: (row["created_at"], row["dataset_id"]))


def _remote_observation_rows_for_scenarios(
    lake: Lake,
    version: int | None,
    scenario_rows: Sequence[Mapping[str, Any]],
    *,
    snapshot_id: str | None = None,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    requested_ids = sorted(
        {
            str(observation_id)
            for scenario in scenario_rows
            for observation_id in (scenario.get("observation_ids") or [])
            if observation_id
        }
    )
    for row in _scan_remote_rows_by_values(
        lake,
        "observations",
        version,
        _REMOTE_OBSERVATION_COLUMNS,
        "observation_id",
        requested_ids,
        snapshot_id=snapshot_id,
    ):
        by_id[row["observation_id"]] = row

    windowed_scenarios = [
        scenario for scenario in scenario_rows if not (scenario.get("observation_ids") or [])
    ]
    if windowed_scenarios:
        run_ids = sorted({str(scenario["run_id"]) for scenario in windowed_scenarios})
        rows = _scan_remote_rows_by_values(
            lake,
            "observations",
            version,
            _REMOTE_OBSERVATION_COLUMNS,
            "run_id",
            run_ids,
            snapshot_id=snapshot_id,
        )
        for row in rows:
            if _observation_in_any_scenario_window(row, windowed_scenarios):
                by_id[row["observation_id"]] = row

    return sorted(
        by_id.values(),
        key=lambda row: (
            str(row.get("run_id") or ""),
            int(row.get("timestamp_ns") or 0),
            str(row.get("topic") or ""),
            str(row["observation_id"]),
        ),
    )


def _observation_in_any_scenario_window(
    observation: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
) -> bool:
    run_id = observation.get("run_id")
    timestamp_ns = observation.get("timestamp_ns")
    if run_id is None or timestamp_ns is None:
        return False
    ts = int(timestamp_ns)
    return any(
        scenario.get("run_id") == run_id
        and int(scenario["start_time_ns"]) <= ts <= int(scenario["end_time_ns"])
        for scenario in scenarios
    )


def _missing_snapshot_observations(
    scenarios: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    return [
        str(observation_id)
        for scenario in scenarios
        for observation_id in (scenario.get("observation_ids") or [])
        if str(observation_id) not in observations
    ]


#: Above this many id values, fetch by random-access ``take_row_ids`` instead of an
#: ``id IN (...)`` predicate. The id-pinned ``observation_id`` set of a real snapshot
#: is ~178K values, which blows up the planner natively (BUG-06); the windowed
#: ``run_id`` set is a handful, so it stays on the cheaper predicate path.
_REMOTE_TAKE_THRESHOLD = 1024


def _scan_remote_rows_by_values(
    lake: Lake,
    table_name: str,
    version: int | None,
    columns: Sequence[str],
    id_column: str,
    values: Sequence[Any],
    *,
    snapshot_id: str | None = None,
) -> list[dict[str, Any]]:
    unique_values = tuple(dict.fromkeys(value for value in values if value is not None))
    if not unique_values:
        return []
    if len(unique_values) > _REMOTE_TAKE_THRESHOLD:
        return _take_remote_rows_by_id(
            lake,
            table_name,
            version,
            columns,
            id_column,
            unique_values,
            snapshot_id=snapshot_id,
        )
    return _scan_remote_table_rows(
        lake,
        table_name,
        version,
        columns,
        _sql_predicate(id_column, unique_values),
        snapshot_id=snapshot_id,
    )


def _take_remote_rows_by_id(
    lake: Lake,
    table_name: str,
    version: int | None,
    columns: Sequence[str],
    id_column: str,
    values: Sequence[Any],
    *,
    snapshot_id: str | None = None,
) -> list[dict[str, Any]]:
    """Remote twin of the BUG-06 fix: fetch a large id set by ``take_row_ids`` rather
    than a multi-MB ``id IN (...)`` predicate sent to the remote planner. Shares the
    version-checkout/error contract of :func:`_scan_remote_table_rows`."""
    table = lake.table(table_name)
    if version is not None:
        try:
            table.checkout(int(version))
        except Exception as exc:
            raise StaleTableVersionError(
                table=table_name,
                requested_version=int(version),
                snapshot_id=snapshot_id,
                reason=str(exc),
            ) from exc
    try:
        return _rows_by_id_take(
            table,
            columns=columns,
            id_column=id_column,
            id_values=values,
        )
    except Exception as exc:
        raise TrainingError(
            f"cannot take {table_name!r} rows through the remote training planner: {exc}"
        ) from exc
    finally:
        if version is not None:
            table.checkout_latest()


def _scan_remote_table_rows(
    lake: Lake,
    table_name: str,
    version: int | None,
    columns: Sequence[str],
    where_sql: str,
    *,
    snapshot_id: str | None = None,
) -> list[dict[str, Any]]:
    table = lake.table(table_name)
    if version is not None:
        try:
            table.checkout(int(version))
        except Exception as exc:
            raise StaleTableVersionError(
                table=table_name,
                requested_version=int(version),
                snapshot_id=snapshot_id,
                reason=str(exc),
            ) from exc
    try:
        query = table.search().select(list(columns))
        if where_sql:
            query = query.where(where_sql)
        rows: list[dict[str, Any]] = []
        for batch in query.to_batches(batch_size=4096):
            rows.extend(batch.to_pylist())
        return rows
    except Exception as exc:
        raise TrainingError(
            f"cannot scan {table_name!r} through the remote training planner: {exc}"
        ) from exc
    finally:
        if version is not None:
            table.checkout_latest()


def _latest_snapshot_row(lake: Lake, name: str) -> dict:
    rows = [
        row for row in lake.table("dataset_snapshots").to_arrow().to_pylist() if row["name"] == name
    ]
    if not rows:
        raise TrainingError(f"no snapshot named {name!r} in {lake.uri}")
    # Most recently created wins, with a stable dataset_id tie-break.
    return max(rows, key=lambda row: (row["created_at"], row["dataset_id"]))


def _scenarios_as_of(lake: Lake, version: int) -> dict[str, dict]:
    """Read scenario rows at a pinned Lance version, keyed by scenario_id."""
    table = lake.table("scenarios")
    table.checkout(version)
    try:
        return {row["scenario_id"]: row for row in table.to_arrow().to_pylist()}
    finally:
        table.checkout_latest()


def _record(row: dict, split: str) -> dict:
    embedding = row.get("embedding")
    return {
        "scenario_id": row["scenario_id"],
        "run_id": row["run_id"],
        "split": split,
        "start_time_ns": row["start_time_ns"],
        "end_time_ns": row["end_time_ns"],
        "topics": list(row["topics"] or []),
        "summary": row.get("summary"),
        "observation_count": row["observation_count"],
        "embedding": list(embedding) if embedding is not None else None,
    }


def load_snapshot_preview(
    lake: Lake,
    name: str,
    *,
    columns: list[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> SnapshotPreview:
    """Preview the first ``batch_size`` samples of snapshot ``name``.

    Samples are read as of the snapshot's pinned table versions and ordered by
    ``scenario_id`` so the batch is deterministic for fixed inputs.
    """
    selected_columns = tuple(columns) if columns else DEFAULT_PREVIEW_COLUMNS
    unknown = [column for column in selected_columns if column not in ALL_FIELDS]
    if unknown:
        raise TrainingError(
            f"unknown preview columns {unknown}; choose from {', '.join(ALL_FIELDS)}"
        )

    row = _latest_snapshot_row(lake, name)
    query_spec = json.loads(row["query_spec"])
    split_payload = json.loads(row["split"])
    assignments = split_payload.get("assignments", {})
    versions = {tv["table"]: tv["version"] for tv in row["table_versions"]}

    scenario_ids = sorted(query_spec.get("scenario_ids", []))
    scenarios = _scenarios_as_of(lake, versions["scenarios"])
    missing = [sid for sid in scenario_ids if sid not in scenarios]
    if missing:
        raise TrainingError(
            f"snapshot {name!r} references scenarios not present at the pinned version: {missing}"
        )

    samples = []
    for sid in scenario_ids[:batch_size]:
        record = _record(scenarios[sid], assignments.get(sid, "train"))
        samples.append({column: record[column] for column in selected_columns})

    return SnapshotPreview(
        lake_uri=lake.uri,
        dataset_id=row["dataset_id"],
        name=row["name"],
        tag=row["tag"],
        split_by=split_payload.get("by", "run"),
        total_scenarios=len(scenario_ids),
        columns=selected_columns,
        samples=samples,
    )


def to_torch_dataset(preview: SnapshotPreview):
    """Adapt a preview into a ``torch.utils.data.Dataset`` (numeric lists → tensors).

    Raises :class:`TrainingError` with install guidance when torch is absent.
    """
    if not torch_available():
        raise TrainingError(
            "PyTorch is not installed; install `lancedb-robotics[torch]` to build "
            "tensor batches (the dictionary preview does not require it)"
        )
    import torch
    from torch.utils.data import Dataset

    samples = preview.samples

    class _SnapshotDataset(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int) -> dict:
            out = {}
            for key, value in samples[index].items():
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(item, (int, float)) for item in value)
                ):
                    out[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    out[key] = value
            return out

    return _SnapshotDataset()


def _frame_refs(episodes: tuple[_Episode, ...]) -> tuple[_TrainingFrameRef, ...]:
    refs: list[_TrainingFrameRef] = []
    linear_index = 0
    for episode in episodes:
        for frame_index, observation in enumerate(episode.observations):
            refs.append(
                _TrainingFrameRef(
                    linear_index=linear_index,
                    episode=episode,
                    frame_index=frame_index,
                    observation=observation,
                )
            )
            linear_index += 1
    return tuple(refs)


def _native_context_observation_columns(
    columns: Sequence[str],
    filters: Mapping[str, Any],
    time_windows: Mapping[str, Any],
) -> tuple[str, ...]:
    """Observation columns the native context must read for ``columns``/filters.

    BUG-06: the context projects only the columns episode-building, sampling, and
    accounting actually touch -- never the heavy ``payload_json``/vector columns
    unless they are requested -- so a trivial projection over a large snapshot
    stays small instead of materializing the whole corpus. ``payload_blob`` is
    included as its cheap ``{position, size}`` descriptor (a blob scan reads no
    bytes) so the manifest's payload accounting stays accurate.
    """
    selected: set[str] = set(_PLAN_REQUIRED_COLUMNS)
    # sensor_id feeds camera-key/is-camera checks; payload_blob descriptor feeds
    # payload-byte accounting. Both are structural regardless of requested columns.
    selected.update({"sensor_id", PAYLOAD_BLOB_COLUMN})
    for column in columns:
        physical = _TRAINING_TO_OBSERVATION_COLUMN.get(column)
        if physical:
            selected.add(physical)
    for key in (*filters, *time_windows):
        physical = _TRAINING_TO_OBSERVATION_COLUMN.get(key)
        if physical:
            selected.add(physical)
    return tuple(sorted(selected))


def _build_row_plan(
    lake: Lake,
    context: Any,
    all_refs: tuple[_TrainingFrameRef, ...],
    *,
    columns: tuple[str, ...],
    filters: Mapping[str, Any],
    media_policy: str,
    decoder: str,
    cache_policy: str,
) -> tuple[TrainingRowPlan, tuple[_TrainingFrameRef, ...]]:
    scan = _planned_observation_scan(lake, context, all_refs, columns=columns, filters=filters)
    row_ids_by_observation = {
        row["observation_id"]: int(row[ROW_ID_COLUMN])
        for row in scan["rows"]
        if row.get("observation_id") and row.get(ROW_ID_COLUMN) is not None
    }
    scanned_observation_ids = set(row_ids_by_observation)
    selected_refs = tuple(
        replace(ref, row_id=row_ids_by_observation.get(ref.observation["observation_id"]))
        for ref in all_refs
        if ref.observation["observation_id"] in scanned_observation_ids
        and _matches_filters(context, ref, filters)
    )
    frame_ids = tuple(ref.observation["observation_id"] for ref in selected_refs)
    row_ids = tuple(ref.row_id for ref in selected_refs)
    materialization_policies = _materialization_policies(
        columns,
        media_policy=media_policy,
        decoder=decoder,
        cache_policy=cache_policy,
    )
    plan_payload = {
        "dataset_id": context.dataset_id,
        "snapshot_name": context.snapshot_name,
        "table_versions": context.table_versions,
        "columns": columns,
        "filters": filters,
        "scan": {key: value for key, value in scan.items() if key != "rows"},
        "frame_ids": frame_ids,
        "row_ids": row_ids,
        "materialization_policies": materialization_policies,
    }
    plan = TrainingRowPlan(
        plan_id="rowplan-" + _stable_digest(plan_payload),
        dataset_id=context.dataset_id,
        snapshot_name=context.snapshot_name,
        table_versions=context.table_versions,
        columns=columns,
        filters=dict(filters),
        scan={key: value for key, value in scan.items() if key != "rows"},
        frame_ids=frame_ids,
        row_ids=row_ids,
        total_frames=len(all_refs),
        selected_frames=len(selected_refs),
        materialization_policies=materialization_policies,
    )
    return plan, selected_refs


def _planned_observation_scan(
    lake: Lake,
    context: Any,
    all_refs: tuple[_TrainingFrameRef, ...],
    *,
    columns: tuple[str, ...],
    filters: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_refs = _logical_candidate_refs(context, all_refs, filters)
    candidate_ids = tuple(ref.observation["observation_id"] for ref in candidate_refs)
    scan_columns = _scan_columns(columns, filters, candidate_refs)
    pushdown_predicates: list[str] = []
    pushed_filters: dict[str, str] = {}

    # BUG-06: do NOT push observation_id IN (<candidate_ids>) -- a real snapshot pins
    # ~178K ids, and that predicate blows up the query planner natively. Push only the
    # (small) filter predicates; restrict to the candidate observations by id-set
    # membership while streaming the scan (see ``_scan_observations_as_of``).
    for key, expected in filters.items():
        column = _pushdown_filter_column(key, expected, candidate_refs)
        if column is None:
            continue
        pushdown_predicates.append(_sql_predicate(column, expected))
        pushed_filters[key] = column

    where_sql = " AND ".join(pushdown_predicates)
    rows = (
        _scan_observations_as_of(
            lake,
            _table_version(context, "observations"),
            scan_columns,
            where_sql,
            keep_ids=set(candidate_ids),
        )
        if candidate_ids
        else []
    )
    return {
        "table": "observations",
        "version": _table_version(context, "observations"),
        "columns": scan_columns,
        "filter_predicate": where_sql,
        "logical_predicates": [_logical_predicate(key, expected) for key, expected in filters.items()],
        "pushed_filters": pushed_filters,
        "candidate_frame_ids": list(candidate_ids),
        "row_count": len(rows),
        "rows": rows,
    }


def _logical_candidate_refs(
    context: Any,
    refs: tuple[_TrainingFrameRef, ...],
    filters: Mapping[str, Any],
) -> tuple[_TrainingFrameRef, ...]:
    logical_only = [
        key
        for key, expected in filters.items()
        if _pushdown_filter_column(key, expected, refs) is None
    ]
    if not logical_only:
        return refs
    return tuple(
        ref
        for ref in refs
        if all(
            _filter_matches(_column_value(context, ref, key), filters[key])
            for key in logical_only
        )
    )


def _scan_columns(
    columns: tuple[str, ...],
    filters: Mapping[str, Any],
    candidate_refs: tuple[_TrainingFrameRef, ...],
) -> list[str]:
    selected = set(_PLAN_REQUIRED_COLUMNS)
    for column in columns:
        physical = _TRAINING_TO_OBSERVATION_COLUMN.get(column)
        if physical in _OBSERVATION_COLUMNS:
            selected.add(physical)
    for key, expected in filters.items():
        physical = _pushdown_filter_column(key, expected, candidate_refs)
        if physical in _OBSERVATION_COLUMNS:
            selected.add(physical)
    return [column for column in _PLAN_REQUIRED_COLUMNS if column in selected] + sorted(
        selected - set(_PLAN_REQUIRED_COLUMNS)
    )


def _pushdown_filter_column(
    key: str,
    expected: Any,
    refs: tuple[_TrainingFrameRef, ...],
) -> str | None:
    if key == "task":
        if any(_filter_matches(ref.observation.get("task_id"), expected) for ref in refs):
            return "task_id"
        return None
    return _PUSHDOWN_FILTER_COLUMNS.get(key)


def _scan_observations_as_of(
    lake: Lake,
    version: int | None,
    columns: Sequence[str],
    where_sql: str,
    *,
    keep_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    table = lake.table("observations")
    if version is not None:
        table.checkout(version)
    try:
        query = table.search().select(list(columns))
        if where_sql:
            query = query.where(where_sql)
        rows: list[dict[str, Any]] = []
        for batch in query.with_row_id(True).to_batches(batch_size=4096):
            for row in batch.to_pylist():
                # BUG-06: membership in the snapshot's referenced observations is
                # enforced here, streaming, rather than via a giant observation_id
                # IN (<ids>) predicate that blows up the planner. Only candidate rows
                # are retained, so the result holds the referenced set, not the corpus.
                if keep_ids is not None and row.get("observation_id") not in keep_ids:
                    continue
                row_id = row.get(ROW_ID_COLUMN)
                if row_id is None:
                    raise TrainingError("Lance observation scan did not return row ids")
                rows.append({**row, ROW_ID_COLUMN: int(row_id)})
        return rows
    except Exception as exc:
        raise TrainingError(f"cannot build training row plan from Lance scan: {exc}") from exc
    finally:
        if version is not None:
            table.checkout_latest()


def _seeded_global_order(
    plan_id: str,
    selected_frames: int,
    *,
    shuffle: bool,
    shuffle_seed: int | None,
    epoch: int,
) -> list[int]:
    """Deterministic epoch order over frame indices.

    Single source of truth for both the in-memory local planner and the server-side
    plan artifact so the two orderings cannot drift (backlog 0117 equivalence).
    """

    order = list(range(selected_frames))
    if not shuffle:
        return order
    seed = shuffle_seed if shuffle_seed is not None else DEFAULT_SHUFFLE_SEED
    rng = random.Random(f"{plan_id}:{seed}")
    rng.shuffle(order)
    if order:
        offset = epoch % len(order)
        order = order[offset:] + order[:offset]
    return order


_SERVER_SIDE_PLAN_CAPABILITY_KEYS = (
    "server_side_row_plan",
    "remote_scan",
    "remote_filtered_read",
    "managed_versioning",
)


def _server_side_plan_inputs(dataset: Any) -> dict[str, Any] | None:
    """Assemble the ordered row/frame id set + metadata for the plan artifact.

    Returns ``None`` when the row plan has unresolved row ids (nothing to persist as
    a server-side artifact). The ordered indices come from the dataset's epoch plan,
    so the artifact reproduces the exact epoch order the local planner computed.
    """

    row_plan = dataset.row_plan
    global_order = dataset.epoch_plan.global_order
    row_ids = row_plan.row_ids
    frame_ids = row_plan.frame_ids
    ordered_row_ids: list[int] = []
    ordered_frame_ids: list[str] = []
    for index in global_order:
        row_id = row_ids[index]
        if row_id is None:
            return None
        ordered_row_ids.append(int(row_id))
        ordered_frame_ids.append(str(frame_ids[index]))
    report = dataset.backend_report
    capabilities = {
        key: bool(report.capabilities.get(key))
        for key in _SERVER_SIDE_PLAN_CAPABILITY_KEYS
    }
    scan = row_plan.scan or {}
    return {
        "row_plan_id": row_plan.plan_id,
        "snapshot_id": row_plan.dataset_id,
        "snapshot_name": row_plan.snapshot_name,
        "table_versions": list(row_plan.table_versions),
        "columns": list(row_plan.columns),
        "display_uri": report.display_uri,
        "connection_kind": report.connection_kind,
        "ordered_row_ids": ordered_row_ids,
        "ordered_frame_ids": ordered_frame_ids,
        "ordering_policy": dataset.epoch_plan.ordering_policy,
        "shuffle": dataset.epoch_plan.shuffle,
        "shuffle_seed": dataset.epoch_plan.shuffle_seed,
        "epoch": dataset.epoch_plan.epoch,
        "pushed_filters": dict(scan.get("pushed_filters", {})),
        "logical_predicates": list(scan.get("logical_predicates", [])),
        "capabilities": capabilities,
    }


def _server_side_plan_summary(dataset: Any) -> dict[str, Any] | None:
    """A lightweight, O(1) advertisement of the plan artifact for the manifest.

    Records availability and pagination shape without hashing the ordered ids or
    persisting pages (those happen lazily via ``dataset.server_side_plan()``).
    """

    if not getattr(dataset, "_server_side_plan_capable", False):
        return None
    total_rows = len(dataset.epoch_plan.global_order)
    if any(dataset.row_plan.row_ids[index] is None for index in dataset.epoch_plan.global_order):
        return {
            "available": False,
            "reason": "row plan has unresolved row ids; server-side artifact not built",
            "kind": SERVER_SIDE_PLAN_KIND,
        }
    page_size = DEFAULT_PLAN_PAGE_SIZE
    num_pages = (total_rows + page_size - 1) // page_size
    store_kind = _plan_page_store_kind(dataset.lake, "auto")
    return {
        "available": True,
        "kind": SERVER_SIDE_PLAN_KIND,
        "ordering_policy": dataset.epoch_plan.ordering_policy,
        "total_rows": total_rows,
        "page_size": page_size,
        "num_pages": num_pages,
        "store_kind": store_kind,
        "display_uri": dataset.backend_report.display_uri,
    }


def _plan_page_store_kind(lake: Lake, store: str) -> str:
    if store == "memory":
        return InMemoryServerSidePlanStore.kind
    if store == "lancedb":
        return LanceTablePlanPageStore.kind
    db = getattr(lake, "_db", None)
    if db is not None and all(
        hasattr(db, name) for name in ("create_table", "open_table", "list_tables")
    ):
        return LanceTablePlanPageStore.kind
    return InMemoryServerSidePlanStore.kind


def _dataset_plan_page_store(lake: Lake, store: str) -> ServerSidePlanStore:
    if _plan_page_store_kind(lake, store) == LanceTablePlanPageStore.kind:
        return LanceTablePlanPageStore(getattr(lake, "_db", None))
    return InMemoryServerSidePlanStore()


def _raise_server_side_plan_unavailable(reason: str) -> None:
    raise ServerSidePlanUnavailableError(
        f"Enterprise server-side row-plan artifact is unavailable: {reason}",
        missing_capabilities=["server_side_row_plan"],
        remediation=(
            "Open a db:// or REST Namespace Enterprise lake with server-side query "
            "support (server_side_row_plan capability), pass fallback='local' for "
            "in-process local planning, or negotiate the capability with the "
            "deployment before requesting a server-side plan handle."
        ),
    )


def _build_dataset_server_side_plan(
    dataset: Any,
    *,
    page_size: int = DEFAULT_PLAN_PAGE_SIZE,
    store: str = "auto",
) -> ServerSidePlanArtifact:
    if not getattr(dataset, "_server_side_plan_capable", False):
        _raise_server_side_plan_unavailable(
            "the resolved backend does not expose the server_side_row_plan capability"
        )
    inputs = _server_side_plan_inputs(dataset)
    if inputs is None:
        _raise_server_side_plan_unavailable(
            "the row plan has unresolved row ids for the selected snapshot"
        )
    store_obj = _dataset_plan_page_store(dataset.lake, store)
    return build_server_side_row_plan(page_size=page_size, store=store_obj, **inputs)


def _plan_page_store_for_handle(lake: Lake, handle: Mapping[str, Any]) -> ServerSidePlanStore:
    store_kind = str(handle.get("store_kind", ""))
    if store_kind == LanceTablePlanPageStore.kind:
        db = getattr(lake, "_db", None)
        if db is None:
            raise ServerSidePlanError(
                "handle references a LanceDB-backed plan store but the lake exposes "
                "no LanceDB connection"
            )
        return LanceTablePlanPageStore(db)
    raise ServerSidePlanError(
        f"cannot reopen a {store_kind!r} plan-page store from a handle alone; "
        "in-memory stores are process-local -- rebuild the handle against a "
        "durable (lancedb) store for cross-process paging"
    )


def _build_epoch_plan(
    row_plan: TrainingRowPlan,
    *,
    lake: Lake | None = None,
    enable_lancedb_backend: bool = False,
    shuffle: bool,
    shuffle_seed: int | None,
    epoch: int,
    worker_id: int,
    num_workers: int,
    resume_from: int,
) -> EpochPlan:
    ordering_policy = "seeded-global-permutation" if shuffle else "snapshot-frame-order"
    seed = shuffle_seed if shuffle_seed is not None else DEFAULT_SHUFFLE_SEED
    global_order = _seeded_global_order(
        row_plan.plan_id,
        row_plan.selected_frames,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        epoch=epoch,
    )
    backend, global_order = _resolve_epoch_execution_backend(
        lake,
        row_plan,
        tuple(global_order),
        shuffle=shuffle,
        shuffle_seed=seed if shuffle else None,
        epoch=epoch,
        worker_id=worker_id,
        num_workers=num_workers,
        resume_from=resume_from,
        enable_lancedb_backend=enable_lancedb_backend,
    )
    if resume_from > len(global_order):
        raise WorkerResumeMismatchError(
            "worker resume offset is beyond the planned epoch: "
            f"row_plan_id={row_plan.plan_id}, resume_from={resume_from}, "
            f"epoch_size={len(global_order)}. Remediation: restart the worker "
            "from a fresh loader config or resume with an offset produced by the "
            "same snapshot, epoch, shuffle seed, and worker partition."
        )
    resumed_global_order = global_order[resume_from:]
    worker_order = tuple(resumed_global_order[worker_id::num_workers])
    sample_indices = worker_order
    plan_payload = {
        "row_plan_id": row_plan.plan_id,
        "shuffle": shuffle,
        "shuffle_seed": seed if shuffle else None,
        "epoch": epoch,
        "ordering_policy": ordering_policy,
        "global_order": global_order,
        "resumed_global_order": resumed_global_order,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "resume_from": resume_from,
        "backend": backend.to_dict(),
    }
    plan_id = "epoch-" + _stable_digest(plan_payload)
    backend = replace(backend, epoch_plan_id=plan_id)
    return EpochPlan(
        plan_id=plan_id,
        row_plan_id=row_plan.plan_id,
        shuffle=shuffle,
        shuffle_seed=seed if shuffle else None,
        epoch=epoch,
        ordering_policy=ordering_policy,
        global_order=tuple(global_order),
        worker_id=worker_id,
        num_workers=num_workers,
        worker_order=worker_order,
        resume_from=resume_from,
        sample_indices=tuple(sample_indices),
        backend=backend,
    )


def _resolve_epoch_execution_backend(
    lake: Lake | None,
    row_plan: Any,
    global_order: tuple[int, ...],
    *,
    shuffle: bool,
    shuffle_seed: int | None,
    epoch: int,
    worker_id: int,
    num_workers: int,
    resume_from: int,
    enable_lancedb_backend: bool,
) -> tuple[EpochExecutionBackend, tuple[int, ...]]:
    capabilities = _epoch_backend_capability(lake)
    common = {
        "row_plan_id": row_plan.plan_id,
        "snapshot_id": getattr(row_plan, "dataset_id", None),
        "table_versions": tuple(dict(item) for item in getattr(row_plan, "table_versions", ())),
        "shuffle_seed": shuffle_seed,
        "epoch": epoch,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "resume_from": resume_from,
        "capabilities": capabilities,
    }
    if not shuffle:
        return (
            EpochExecutionBackend(
                kind=EPOCH_BACKEND_PYTHON,
                execution_mode="python-snapshot-order",
                reason="shuffle is disabled; snapshot row order does not need a persisted permutation",
                **common,
            ),
            global_order,
        )
    if not enable_lancedb_backend:
        return (
            EpochExecutionBackend(
                kind=EPOCH_BACKEND_PYTHON,
                execution_mode="python-in-memory",
                reason="LanceDB permutation backend is not enabled for this plan type",
                **common,
            ),
            global_order,
        )

    lancedb_capability = capabilities[EPOCH_BACKEND_LANCEDB_PERMUTATION]
    if not lancedb_capability["supported"]:
        return (
            EpochExecutionBackend(
                kind=EPOCH_BACKEND_PYTHON,
                execution_mode="python-in-memory",
                reason=lancedb_capability["reason"],
                **common,
            ),
            global_order,
        )

    try:
        descriptor, ordered_indices = _persist_lancedb_epoch_permutation(
            lake,
            row_plan,
            global_order,
            shuffle_seed=shuffle_seed,
            epoch=epoch,
            worker_id=worker_id,
            num_workers=num_workers,
            resume_from=resume_from,
            capabilities=capabilities,
        )
    except Exception as exc:
        return (
            EpochExecutionBackend(
                kind=EPOCH_BACKEND_PYTHON,
                execution_mode="python-in-memory",
                reason=f"LanceDB permutation backend probe failed: {exc}",
                warnings=(str(exc),),
                **common,
            ),
            global_order,
        )
    return descriptor, ordered_indices


def _persist_lancedb_epoch_permutation(
    lake: Lake | None,
    row_plan: Any,
    global_order: tuple[int, ...],
    *,
    shuffle_seed: int | None,
    epoch: int,
    worker_id: int,
    num_workers: int,
    resume_from: int,
    capabilities: dict[str, Any],
) -> tuple[EpochExecutionBackend, tuple[int, ...]]:
    if lake is None:
        raise TrainingError("no lake is available for LanceDB permutation planning")
    db = getattr(lake, "_db", None)
    if db is None:
        raise TrainingError("lake does not expose a LanceDB connection")
    row_ids = tuple(getattr(row_plan, "row_ids", ()))
    if len(row_ids) < len(global_order) or any(row_ids[index] is None for index in global_order):
        raise TrainingError("row plan has missing row ids and cannot be persisted as a permutation")
    ordered_row_ids = tuple(int(row_ids[index]) for index in global_order)
    permutation_table = _epoch_permutation_table_name(
        row_plan,
        ordered_row_ids,
        shuffle_seed=shuffle_seed,
        epoch=epoch,
    )
    table = _create_or_reuse_epoch_permutation_table(db, permutation_table, ordered_row_ids)
    row_id_to_index = {int(row_id): index for index, row_id in enumerate(row_ids) if row_id is not None}
    ordered_indices = _indices_from_lancedb_permutation_table(table, row_id_to_index)
    if ordered_indices != global_order:
        raise TrainingError("LanceDB permutation table returned an order different from the row plan")
    descriptor = EpochExecutionBackend(
        kind=EPOCH_BACKEND_LANCEDB_PERMUTATION,
        execution_mode="lancedb-permutation-table",
        row_plan_id=row_plan.plan_id,
        snapshot_id=getattr(row_plan, "dataset_id", None),
        table_versions=tuple(dict(item) for item in getattr(row_plan, "table_versions", ())),
        shuffle_seed=shuffle_seed,
        epoch=epoch,
        worker_id=worker_id,
        num_workers=num_workers,
        resume_from=resume_from,
        permutation_table=permutation_table,
        permutation_ref=f"lancedb://{permutation_table}",
        reason="ordered row ids are persisted in a LanceDB permutation table",
        capabilities=capabilities,
    )
    return descriptor, ordered_indices


def _epoch_permutation_table_name(
    row_plan: Any,
    ordered_row_ids: tuple[int, ...],
    *,
    shuffle_seed: int | None,
    epoch: int,
) -> str:
    payload = {
        "row_plan_id": row_plan.plan_id,
        "snapshot_id": getattr(row_plan, "dataset_id", None),
        "table_versions": getattr(row_plan, "table_versions", ()),
        "shuffle_seed": shuffle_seed,
        "epoch": epoch,
        "ordered_row_ids": ordered_row_ids,
    }
    return EPOCH_PERMUTATION_TABLE_PREFIX + _stable_digest(payload)


def _create_or_reuse_epoch_permutation_table(
    db: Any,
    table_name: str,
    ordered_row_ids: tuple[int, ...],
) -> Any:
    table_names = _db_table_names(db)
    if table_name in table_names:
        table = db.open_table(table_name)
        stored = tuple(int(row["row_id"]) for row in table.to_arrow().to_pylist())
        if stored == ordered_row_ids:
            return table
    data = pa.table(
        {
            "row_id": pa.array(ordered_row_ids, type=pa.uint64()),
            "split_id": pa.array([0] * len(ordered_row_ids), type=pa.uint64()),
        }
    )
    mode = "overwrite" if table_name in table_names else "create"
    return db.create_table(table_name, data=data, mode=mode)


def _indices_from_lancedb_permutation_table(
    table: Any,
    row_id_to_index: Mapping[int, int],
) -> tuple[int, ...]:
    indices: list[int] = []
    for row in table.to_arrow().to_pylist():
        row_id = int(row["row_id"])
        if row_id not in row_id_to_index:
            raise TrainingError(f"permutation row id {row_id} is not in the row plan")
        indices.append(row_id_to_index[row_id])
    return tuple(indices)


def _db_table_names(db: Any) -> set[str]:
    response = db.list_tables()
    tables = getattr(response, "tables", response)
    return {str(name) for name in (tables or [])}


def _resolve_alignment_job(
    lake: Lake,
    *,
    alignment: str | None,
    alignment_id: str | None,
    name: str | None,
) -> dict[str, Any]:
    if alignment is None and alignment_id is None and name is None:
        raise TrainingError("alignment_id or name is required")
    jobs = lake.table("alignment_jobs").to_arrow().to_pylist()
    selected: list[dict[str, Any]] = []
    if alignment_id is not None:
        selected = [row for row in jobs if row["alignment_id"] == alignment_id]
    elif name is not None:
        selected = [row for row in jobs if row["name"] == name]
    elif alignment is not None:
        selected = [row for row in jobs if row["alignment_id"] == alignment]
        if not selected:
            selected = [row for row in jobs if row["name"] == alignment]
    if name is not None and alignment_id is not None:
        selected = [
            row
            for row in selected
            if row["alignment_id"] == alignment_id and row["name"] == name
        ]
    if not selected:
        target = alignment_id or name or alignment
        raise TrainingError(f"no recorded alignment job named or identified by {target!r}")
    job = max(selected, key=lambda row: (row["created_at"], row["alignment_id"]))
    if job.get("output_table") not in {"aligned_frames", "aligned_ticks"}:
        raise TrainingError(
            f"alignment {job['alignment_id']!r} was not materialized to a training table"
        )
    return job


def _normalize_aligned_streams(
    job: Mapping[str, Any],
    streams: Sequence[str] | None,
) -> tuple[str, ...]:
    available = tuple(str(stream) for stream in (job.get("streams") or ()))
    selected = tuple(str(stream) for stream in (streams or available) if str(stream))
    if not selected:
        raise TrainingError("at least one aligned stream is required")
    if len(set(selected)) != len(selected):
        raise TrainingError("aligned streams must be unique")
    unknown = [stream for stream in selected if stream not in available]
    if unknown:
        raise TrainingError(
            f"unknown aligned streams {unknown}; choose from {', '.join(available)}"
        )
    return selected


def _normalize_aligned_statuses(statuses: Sequence[str] | str | None) -> tuple[str, ...]:
    if statuses is None:
        return ()
    if isinstance(statuses, str):
        values = (statuses,)
    else:
        values = tuple(str(status) for status in statuses)
    normalized = tuple(status for status in values if status)
    if len(set(normalized)) != len(normalized):
        raise TrainingError("aligned status filters must be unique")
    return normalized


def _normalize_required_streams(
    require_streams: bool | Sequence[str],
    streams: tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(require_streams, bool):
        return streams if require_streams else ()
    required = tuple(str(stream) for stream in require_streams if str(stream))
    unknown = [stream for stream in required if stream not in streams]
    if unknown:
        raise TrainingError(
            f"required streams {unknown} are not in the selected streams {list(streams)}"
        )
    return required


def _validate_min_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    confidence = float(value)
    if confidence < 0.0 or confidence > 1.0:
        raise TrainingError("min_confidence must be between 0.0 and 1.0")
    return confidence


def _validate_aligned_columns(columns: Sequence[str]) -> None:
    unknown = [column for column in columns if column not in ALIGNED_TRAINING_COLUMNS]
    if unknown:
        raise TrainingError(
            f"unknown aligned training columns {unknown}; choose from "
            f"{', '.join(ALIGNED_TRAINING_COLUMNS)}"
        )


def _build_aligned_tick_plan(
    lake: Lake,
    job: Mapping[str, Any],
    *,
    streams: tuple[str, ...],
    columns: tuple[str, ...],
    quality_policy: Mapping[str, Any],
    feature_policy: str,
    decoder: str,
    cache_policy: str,
) -> tuple[AlignedTrainingTickPlan, dict[int, dict[str, dict[str, Any]]], set[int]]:
    tick_plan = _build_aligned_ticks_jsonb_plan(
        lake,
        job,
        streams=streams,
        columns=columns,
        quality_policy=quality_policy,
        feature_policy=feature_policy,
        decoder=decoder,
        cache_policy=cache_policy,
    )
    if tick_plan is not None:
        return tick_plan
    return _build_aligned_frame_pivot_plan(
        lake,
        job,
        streams=streams,
        columns=columns,
        quality_policy=quality_policy,
        feature_policy=feature_policy,
        decoder=decoder,
        cache_policy=cache_policy,
    )


def _build_aligned_frame_pivot_plan(
    lake: Lake,
    job: Mapping[str, Any],
    *,
    streams: tuple[str, ...],
    columns: tuple[str, ...],
    quality_policy: Mapping[str, Any],
    feature_policy: str,
    decoder: str,
    cache_policy: str,
) -> tuple[AlignedTrainingTickPlan, dict[int, dict[str, dict[str, Any]]], set[int]]:
    all_tick_indices = _scan_aligned_tick_indices(lake, str(job["alignment_id"]), streams)
    scan = _planned_aligned_frame_scan(
        lake,
        str(job["alignment_id"]),
        streams,
        statuses=tuple(quality_policy.get("statuses") or ()),
        min_confidence=quality_policy.get("min_confidence"),
    )
    rows = scan["rows"]
    rows_by_tick = _aligned_rows_by_tick(rows, streams)
    filtered_ticks = _filtered_aligned_ticks(
        all_tick_indices,
        rows_by_tick,
        streams=streams,
        require_streams=tuple(quality_policy.get("require_streams") or ()),
        allow_missing=bool(quality_policy.get("allow_missing", True)),
    )
    tick_indices = tuple(tick for tick in all_tick_indices if tick not in filtered_ticks)
    aligned_frame_ids = tuple(
        tuple(
            rows_by_tick.get(tick, {}).get(stream, {}).get("aligned_frame_id")
            for stream in streams
            if rows_by_tick.get(tick, {}).get(stream, {}).get("aligned_frame_id")
        )
        for tick in tick_indices
    )
    source_row_ids = tuple(
        tuple(
            row_id
            for stream in streams
            for row_id in rows_by_tick.get(tick, {}).get(stream, {}).get("source_row_ids")
            or []
        )
        for tick in tick_indices
    )
    row_ids = tuple(
        _first_aligned_frame_row_id(rows_by_tick.get(tick, {}).get(stream) for stream in streams)
        for tick in tick_indices
    )
    read_versions = _current_table_versions(lake, ("alignment_jobs", "aligned_frames"))
    materialization_policies = _aligned_materialization_policies(
        feature_policy,
        decoder=decoder,
        cache_policy=cache_policy,
        storage_backend=ALIGNED_FRAMES_STORAGE_BACKEND,
    )
    plan_payload = {
        "alignment_id": job["alignment_id"],
        "alignment_name": job["name"],
        "storage_backend": ALIGNED_FRAMES_STORAGE_BACKEND,
        "schema_version": "1",
        "recipe_digest": _alignment_recipe_digest(job),
        "table_versions": _alignment_input_versions(job),
        "read_table_versions": read_versions,
        "streams": streams,
        "columns": columns,
        "quality_policy": quality_policy,
        "scan": {key: value for key, value in scan.items() if key != "rows"},
        "tick_indices": tick_indices,
        "aligned_frame_ids": aligned_frame_ids,
        "source_row_ids": source_row_ids,
        "materialization_policies": materialization_policies,
    }
    plan = AlignedTrainingTickPlan(
        plan_id="tickplan-" + _stable_digest(plan_payload),
        alignment_id=str(job["alignment_id"]),
        alignment_name=str(job["name"]),
        table_versions=_alignment_input_versions(job),
        read_table_versions=read_versions,
        storage_backend=ALIGNED_FRAMES_STORAGE_BACKEND,
        schema_version="1",
        streams=streams,
        columns=columns,
        quality_policy=dict(quality_policy),
        scan={key: value for key, value in scan.items() if key != "rows"},
        tick_indices=tick_indices,
        aligned_frame_ids=aligned_frame_ids,
        source_row_ids=source_row_ids,
        row_ids=row_ids,
        total_ticks=len(all_tick_indices),
        selected_ticks=len(tick_indices),
        total_frames=len(all_tick_indices),
        selected_frames=len(tick_indices),
        materialization_policies=materialization_policies,
    )
    return plan, rows_by_tick, filtered_ticks


def _build_aligned_ticks_jsonb_plan(
    lake: Lake,
    job: Mapping[str, Any],
    *,
    streams: tuple[str, ...],
    columns: tuple[str, ...],
    quality_policy: Mapping[str, Any],
    feature_policy: str,
    decoder: str,
    cache_policy: str,
) -> tuple[AlignedTrainingTickPlan, dict[int, dict[str, dict[str, Any]]], set[int]] | None:
    scan = _planned_aligned_tick_scan(
        lake,
        str(job["alignment_id"]),
        quality_policy=quality_policy,
    )
    rows = scan["rows"]
    if not rows:
        return None
    statuses = tuple(quality_policy.get("statuses") or ())
    min_confidence = quality_policy.get("min_confidence")
    rows_by_tick: dict[int, dict[str, dict[str, Any]]] = {}
    row_by_tick: dict[int, dict[str, Any]] = {}
    for row in rows:
        stream_detail = _aligned_tick_stream_detail(row)
        masks = _aligned_tick_masks(row)
        _validate_aligned_tick_summary(row, stream_detail, masks)
        tick = int(row["tick_index"])
        row_by_tick[tick] = row
        rows_by_tick[tick] = {
            stream: stream_row
            for stream in streams
            if (stream_row := stream_detail.get(stream)) is not None
            and _aligned_tick_stream_matches_policy(
                stream_row,
                statuses=statuses,
                min_confidence=min_confidence,
            )
        }
    all_tick_indices = tuple(sorted(row_by_tick))
    filtered_ticks = _filtered_aligned_ticks(
        all_tick_indices,
        rows_by_tick,
        streams=streams,
        require_streams=tuple(quality_policy.get("require_streams") or ()),
        allow_missing=bool(quality_policy.get("allow_missing", True)),
    )
    tick_indices = tuple(tick for tick in all_tick_indices if tick not in filtered_ticks)
    aligned_frame_ids = tuple(
        tuple(
            rows_by_tick.get(tick, {}).get(stream, {}).get("aligned_frame_id")
            for stream in streams
            if rows_by_tick.get(tick, {}).get(stream, {}).get("aligned_frame_id")
        )
        for tick in tick_indices
    )
    source_row_ids = tuple(
        tuple(
            row_id
            for stream in streams
            for row_id in rows_by_tick.get(tick, {}).get(stream, {}).get("source_row_ids")
            or []
        )
        for tick in tick_indices
    )
    row_ids = tuple(
        int(row_by_tick[tick][ROW_ID_COLUMN])
        if row_by_tick[tick].get(ROW_ID_COLUMN) is not None
        else None
        for tick in tick_indices
    )
    read_versions = _current_table_versions(lake, ("alignment_jobs", "aligned_ticks"))
    materialization_policies = _aligned_materialization_policies(
        feature_policy,
        decoder=decoder,
        cache_policy=cache_policy,
        storage_backend=ALIGNED_TICKS_STORAGE_BACKEND,
    )
    plan_payload = {
        "alignment_id": job["alignment_id"],
        "alignment_name": job["name"],
        "storage_backend": ALIGNED_TICKS_STORAGE_BACKEND,
        "schema_version": ALIGNED_TICKS_SCHEMA_VERSION,
        "recipe_digest": _alignment_recipe_digest(job),
        "table_versions": _alignment_input_versions(job),
        "read_table_versions": read_versions,
        "streams": streams,
        "columns": columns,
        "quality_policy": quality_policy,
        "scan": {key: value for key, value in scan.items() if key != "rows"},
        "tick_indices": tick_indices,
        "aligned_frame_ids": aligned_frame_ids,
        "source_row_ids": source_row_ids,
        "materialization_policies": materialization_policies,
    }
    plan = AlignedTrainingTickPlan(
        plan_id="tickplan-" + _stable_digest(plan_payload),
        alignment_id=str(job["alignment_id"]),
        alignment_name=str(job["name"]),
        table_versions=_alignment_input_versions(job),
        read_table_versions=read_versions,
        storage_backend=ALIGNED_TICKS_STORAGE_BACKEND,
        schema_version=ALIGNED_TICKS_SCHEMA_VERSION,
        streams=streams,
        columns=columns,
        quality_policy=dict(quality_policy),
        scan={key: value for key, value in scan.items() if key != "rows"},
        tick_indices=tick_indices,
        aligned_frame_ids=aligned_frame_ids,
        source_row_ids=source_row_ids,
        row_ids=row_ids,
        total_ticks=len(all_tick_indices),
        selected_ticks=len(tick_indices),
        total_frames=len(all_tick_indices),
        selected_frames=len(tick_indices),
        materialization_policies=materialization_policies,
    )
    return plan, rows_by_tick, filtered_ticks


def _predicate_index_params(
    lake: Lake,
    *,
    table: str,
    columns: Sequence[str],
    filter_columns: Sequence[str],
    diagnostic_columns: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    filter_set = set(filter_columns)
    diagnostic_set = set(diagnostic_columns)
    params: list[dict[str, Any]] = []
    try:
        results = describe_scalar_indexes(lake, table=table, columns=columns)
    except Exception as exc:  # noqa: BLE001 - index diagnostics must not block reads
        results = ()
        for column in columns:
            payload = {
                "table": table,
                "column": column,
                "status": "failed",
                "index_type": SCALAR_INDEX_TYPE,
                "num_rows": None,
                "reason": f"cannot inspect scalar indexes for {table!r}: {exc}",
            }
            payload["used_in_filter"] = column in filter_set
            payload["predicate_role"] = (
                "filter"
                if column in filter_set
                else "quality-diagnostic"
                if column in diagnostic_set
                else "hot-column"
            )
            params.append(payload)
    for result in results:
        payload = result.to_params()
        column = result.column
        payload["used_in_filter"] = column in filter_set
        payload["predicate_role"] = (
            "filter"
            if column in filter_set
            else "quality-diagnostic"
            if column in diagnostic_set
            else "hot-column"
        )
        params.append(payload)
    return tuple(params)


def _aligned_tick_diagnostic_index_columns(
    quality_policy: Mapping[str, Any],
) -> tuple[str, ...]:
    columns: list[str] = []
    if quality_policy.get("min_confidence") is not None:
        columns.append("min_confidence")
    if quality_policy.get("require_streams") or not bool(
        quality_policy.get("allow_missing", True)
    ):
        columns.append("has_missing")
    return tuple(columns)


def _planned_aligned_tick_scan(
    lake: Lake,
    alignment_id: str,
    *,
    quality_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = quality_policy or {}
    predicate_indexes = _predicate_index_params(
        lake,
        table="aligned_ticks",
        columns=ALIGNED_TICK_PREDICATE_INDEX_COLUMNS,
        filter_columns=("alignment_id",),
        diagnostic_columns=_aligned_tick_diagnostic_index_columns(policy),
    )
    try:
        table = lake.table("aligned_ticks")
    except Exception:
        return {
            "table": "aligned_ticks",
            "version": None,
            "columns": list(_ALIGNED_TICK_SCAN_COLUMNS),
            "filter_predicate": _sql_predicate("alignment_id", alignment_id),
            "predicate_indexes": predicate_indexes,
            "row_count": 0,
            "rows": [],
            "fallback_reason": "aligned_ticks table is absent",
        }
    where_sql = _sql_predicate("alignment_id", alignment_id)
    query = (
        table.search()
        .select(list(_ALIGNED_TICK_SCAN_COLUMNS))
        .where(where_sql)
        .with_row_id(True)
    )
    rows: list[dict[str, Any]] = []
    for batch in query.to_batches(batch_size=4096):
        for row in batch.to_pylist():
            row_id = row.get(ROW_ID_COLUMN)
            rows.append({**row, ROW_ID_COLUMN: int(row_id) if row_id is not None else None})
    rows.sort(key=lambda row: (int(row["tick_index"]), row["aligned_tick_id"]))
    return {
        "table": "aligned_ticks",
        "version": int(table.version),
        "columns": list(_ALIGNED_TICK_SCAN_COLUMNS),
        "filter_predicate": where_sql,
        "predicate_indexes": predicate_indexes,
        "post_filter": "stream_detail_json status/min_confidence per selected stream",
        "row_count": len(rows),
        "rows": rows,
    }


def _scan_aligned_tick_indices(
    lake: Lake,
    alignment_id: str,
    streams: tuple[str, ...],
) -> tuple[int, ...]:
    predicate = " AND ".join(
        [
            _sql_predicate("alignment_id", alignment_id),
            _sql_predicate("stream", streams),
        ]
    )
    rows = (
        lake.table("aligned_frames")
        .search()
        .select(["tick_index"])
        .where(predicate)
        .to_arrow()
        .to_pylist()
    )
    return tuple(sorted({int(row["tick_index"]) for row in rows}))


def _planned_aligned_frame_scan(
    lake: Lake,
    alignment_id: str,
    streams: tuple[str, ...],
    *,
    statuses: tuple[str, ...],
    min_confidence: float | None,
) -> dict[str, Any]:
    predicates = [
        _sql_predicate("alignment_id", alignment_id),
        _sql_predicate("stream", streams),
    ]
    if statuses:
        predicates.append(_sql_predicate("status", statuses))
    if min_confidence is not None:
        predicates.append(f"confidence >= {float(min_confidence)}")
    where_sql = " AND ".join(predicates)
    predicate_indexes = _predicate_index_params(
        lake,
        table="aligned_frames",
        columns=ALIGNED_FRAME_PREDICATE_INDEX_COLUMNS,
        filter_columns=(
            "alignment_id",
            "stream",
            *(("status",) if statuses else ()),
            *(("confidence",) if min_confidence is not None else ()),
        ),
    )
    query = (
        lake.table("aligned_frames")
        .search()
        .select(list(_ALIGNED_FRAME_SCAN_COLUMNS))
        .where(where_sql)
        .with_row_id(True)
    )
    rows: list[dict[str, Any]] = []
    for batch in query.to_batches(batch_size=4096):
        for row in batch.to_pylist():
            row_id = row.get(ROW_ID_COLUMN)
            rows.append({**row, ROW_ID_COLUMN: int(row_id) if row_id is not None else None})
    rows.sort(
        key=lambda row: (
            int(row["tick_index"]),
            streams.index(row["stream"]) if row["stream"] in streams else len(streams),
            row["aligned_frame_id"],
        )
    )
    return {
        "table": "aligned_frames",
        "version": int(lake.table("aligned_frames").version),
        "columns": list(_ALIGNED_FRAME_SCAN_COLUMNS),
        "filter_predicate": where_sql,
        "predicate_indexes": predicate_indexes,
        "row_count": len(rows),
        "rows": rows,
    }


def _ensure_aligned_ticks_table(lake: Lake):
    try:
        return lake.table("aligned_ticks")
    except Exception:
        lake._db.create_table("aligned_ticks", schema=ALIGNED_TICKS_SCHEMA, exist_ok=True)
        return lake.table("aligned_ticks")


def _aligned_tick_storage_row_from_frame_rows(
    job: Mapping[str, Any],
    *,
    tick_index: int,
    rows_by_stream: Mapping[str, Mapping[str, Any]],
    streams: tuple[str, ...],
    created_at: datetime,
) -> dict[str, Any]:
    stream_samples = {
        stream: _aligned_stream_sample(stream, rows_by_stream.get(stream))
        for stream in streams
    }
    masks = _aligned_masks(stream_samples)
    quality_flags = sorted(
        {
            flag
            for sample in stream_samples.values()
            for flag in sample.get("quality_flags") or []
        }
    )
    stream_detail: dict[str, dict[str, Any]] = {}
    stream_values: dict[str, Any] = {}
    lineage = {
        "aligned_frame_ids": {},
        "source_observation_ids": {},
        "source_row_ids": {},
    }
    for stream, sample in stream_samples.items():
        detail = {
            key: value
            for key, value in sample.items()
            if key not in {"feature", "value"}
        }
        stream_detail[stream] = detail
        stream_values[stream] = sample.get("value")
        lineage["aligned_frame_ids"][stream] = sample.get("aligned_frame_id")
        lineage["source_observation_ids"][stream] = list(
            sample.get("source_observation_ids") or []
        )
        lineage["source_row_ids"][stream] = list(sample.get("source_row_ids") or [])
    missing_streams = [
        stream for stream in streams if bool((masks.get("missing") or {}).get(stream))
    ]
    interpolated_streams = [
        stream for stream in streams if bool((masks.get("interpolated") or {}).get(stream))
    ]
    out_of_tolerance_streams = [
        stream
        for stream in streams
        if bool((masks.get("out_of_tolerance") or {}).get(stream))
    ]
    confidences = [float(sample.get("confidence") or 0.0) for sample in stream_samples.values()]
    return {
        "aligned_tick_id": "at-"
        + _stable_digest({"alignment_id": job["alignment_id"], "tick_index": tick_index}),
        "alignment_id": str(job["alignment_id"]),
        "alignment_name": str(job["name"]),
        "recipe_digest": _alignment_recipe_digest(job),
        "run_id": _aligned_tick_run_id(stream_samples),
        "tick_index": int(tick_index),
        "timestamp_ns": _aligned_tick_timestamp(stream_samples),
        "available_streams": [
            stream for stream in streams if stream not in set(missing_streams)
        ],
        "missing_streams": missing_streams,
        "interpolated_streams": interpolated_streams,
        "out_of_tolerance_streams": out_of_tolerance_streams,
        "has_missing": bool(missing_streams),
        "has_out_of_tolerance": bool(out_of_tolerance_streams),
        "min_confidence": min(confidences) if confidences else 0.0,
        "quality_flags": quality_flags,
        "stream_detail_json": _jsonb_dumps(stream_detail),
        "masks_json": _jsonb_dumps(masks),
        "stream_values_json": _jsonb_dumps(stream_values),
        "lineage_json": _jsonb_dumps(lineage),
        "transform_id": str(job["transform_id"]),
        "created_at": created_at,
    }


def _aligned_metadata_signature(
    tick_index: int,
    rows_by_stream: Mapping[str, Mapping[str, Any]],
    streams: tuple[str, ...],
) -> dict[str, Any]:
    stream_samples = {
        stream: _aligned_stream_sample(stream, rows_by_stream.get(stream))
        for stream in streams
    }
    return {
        "tick_index": int(tick_index),
        "streams": {
            stream: {
                key: stream_samples[stream].get(key)
                for key in (
                    "status",
                    "observation_id",
                    "source_observation_ids",
                    "source_row_ids",
                    "source_timestamp_ns",
                    "source_time_ns",
                    "receive_time_ns",
                    "latency_ns",
                    "error_ns",
                    "absolute_error_ns",
                    "confidence",
                    "value",
                    "quality_flags",
                    "aligned_frame_id",
                    "transform_id",
                )
            }
            for stream in streams
        },
        "masks": _aligned_masks(stream_samples),
    }


def _aligned_tick_stream_detail(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    detail = _jsonb_dict(row, "stream_detail_json")
    normalized: dict[str, dict[str, Any]] = {}
    for stream, payload in detail.items():
        if not isinstance(payload, Mapping):
            raise TrainingError(
                f"aligned_ticks row {row.get('aligned_tick_id')!r} has invalid "
                f"stream detail for {stream!r}"
            )
        stream_row = dict(payload)
        stream_row.setdefault("stream", stream)
        stream_row["source_observation_ids"] = list(
            stream_row.get("source_observation_ids") or []
        )
        stream_row["source_row_ids"] = [
            int(row_id) for row_id in stream_row.get("source_row_ids") or []
        ]
        stream_row["quality_flags"] = list(stream_row.get("quality_flags") or [])
        stream_row["confidence"] = float(stream_row.get("confidence") or 0.0)
        normalized[str(stream)] = stream_row
    return normalized


def _aligned_tick_masks(row: Mapping[str, Any]) -> dict[str, dict[str, bool]]:
    raw = _jsonb_dict(row, "masks_json")
    masks: dict[str, dict[str, bool]] = {}
    for name, values in raw.items():
        if isinstance(values, Mapping):
            masks[str(name)] = {str(stream): bool(value) for stream, value in values.items()}
    return masks


def _validate_aligned_tick_summary(
    row: Mapping[str, Any],
    stream_detail: Mapping[str, Mapping[str, Any]],
    masks: Mapping[str, Mapping[str, bool]],
) -> None:
    missing = {
        stream
        for stream, detail in stream_detail.items()
        if bool((masks.get("missing") or {}).get(stream))
        or detail.get("status") in {"missing", "filtered"}
        or not detail.get("source_observation_ids")
    }
    interpolated = {
        stream
        for stream, detail in stream_detail.items()
        if bool((masks.get("interpolated") or {}).get(stream))
        or (
            detail.get("status") == "aligned"
            and detail.get("interpolation") == "linear"
            and len(detail.get("source_observation_ids") or []) > 1
        )
    }
    out_of_tolerance = {
        stream
        for stream, detail in stream_detail.items()
        if bool((masks.get("out_of_tolerance") or {}).get(stream))
        or detail.get("status") == "out_of_tolerance"
    }
    available = set(stream_detail) - missing
    expected = {
        "available_streams": available,
        "missing_streams": missing,
        "interpolated_streams": interpolated,
        "out_of_tolerance_streams": out_of_tolerance,
    }
    for column, values in expected.items():
        actual = {str(value) for value in row.get(column) or []}
        if actual != values:
            raise TrainingError(
                f"aligned_ticks row {row.get('aligned_tick_id')!r} has {column}={sorted(actual)} "
                f"but JSONB stream detail implies {sorted(values)}"
            )
    if bool(row.get("has_missing")) != bool(missing):
        raise TrainingError(
            f"aligned_ticks row {row.get('aligned_tick_id')!r} has has_missing="
            f"{row.get('has_missing')!r} but JSONB stream detail implies {bool(missing)!r}"
        )
    if bool(row.get("has_out_of_tolerance")) != bool(out_of_tolerance):
        raise TrainingError(
            f"aligned_ticks row {row.get('aligned_tick_id')!r} has has_out_of_tolerance="
            f"{row.get('has_out_of_tolerance')!r} but JSONB stream detail implies "
            f"{bool(out_of_tolerance)!r}"
        )
    confidences = [float(detail.get("confidence") or 0.0) for detail in stream_detail.values()]
    expected_min = min(confidences) if confidences else 0.0
    actual_min = float(row.get("min_confidence") or 0.0)
    if abs(actual_min - expected_min) > 1e-9:
        raise TrainingError(
            f"aligned_ticks row {row.get('aligned_tick_id')!r} has min_confidence="
            f"{actual_min!r} but JSONB stream detail implies {expected_min!r}"
        )


def _aligned_tick_stream_matches_policy(
    row: Mapping[str, Any],
    *,
    statuses: tuple[str, ...],
    min_confidence: float | None,
) -> bool:
    if statuses and row.get("status") not in statuses:
        return False
    if min_confidence is not None and float(row.get("confidence") or 0.0) < float(
        min_confidence
    ):
        return False
    return True


def _jsonb_dict(row: Mapping[str, Any], column: str) -> dict[str, Any]:
    value = row.get(column)
    if value in (None, ""):
        return {}
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TrainingError(
                f"aligned_ticks row {row.get('aligned_tick_id')!r} has invalid {column}"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise TrainingError(
                f"aligned_ticks row {row.get('aligned_tick_id')!r} has non-object {column}"
            )
        return {str(key): item for key, item in parsed.items()}
    raise TrainingError(
        f"aligned_ticks row {row.get('aligned_tick_id')!r} has unsupported {column} value"
    )


def _jsonb_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True)


def _chunks(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _aligned_rows_by_tick(
    rows: Sequence[dict[str, Any]],
    streams: tuple[str, ...],
) -> dict[int, dict[str, dict[str, Any]]]:
    stream_order = {stream: index for index, stream in enumerate(streams)}
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            int(item["tick_index"]),
            stream_order.get(str(item["stream"]), len(stream_order)),
            item["aligned_frame_id"],
        ),
    ):
        tick = int(row["tick_index"])
        stream = str(row["stream"])
        grouped.setdefault(tick, {})
        grouped[tick].setdefault(stream, row)
    return grouped


def _filtered_aligned_ticks(
    all_tick_indices: Sequence[int],
    rows_by_tick: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    streams: tuple[str, ...],
    require_streams: tuple[str, ...],
    allow_missing: bool,
) -> set[int]:
    filtered: set[int] = set()
    required = require_streams or (() if allow_missing else streams)
    for tick in all_tick_indices:
        rows_by_stream = rows_by_tick.get(tick, {})
        if required and any(not _aligned_stream_is_valid(rows_by_stream.get(stream)) for stream in required):
            filtered.add(tick)
    return filtered


def _aligned_stream_is_valid(row: Mapping[str, Any] | None) -> bool:
    return bool(row) and row.get("status") == "aligned"


def _first_aligned_frame_row_id(rows: Any) -> int | None:
    for row in rows:
        if row and row.get(ROW_ID_COLUMN) is not None:
            return int(row[ROW_ID_COLUMN])
    return None


def _aligned_sample_for_tick(
    job: Mapping[str, Any],
    tick_index: int,
    rows_by_stream: Mapping[str, dict[str, Any]],
    *,
    streams: tuple[str, ...],
    columns: tuple[str, ...],
    quality_policy: Mapping[str, Any],
    manifest: AlignedTrainingManifest,
    tick_plan: AlignedTrainingTickPlan,
    epoch_plan: EpochPlan,
    sample_index: int,
    plan_index: int,
    feature_resolver: _AlignedFeatureResolver | None = None,
) -> dict[str, Any]:
    stream_samples = {
        stream: _aligned_stream_sample(stream, rows_by_stream.get(stream))
        for stream in streams
    }
    if feature_resolver is not None:
        stream_samples = feature_resolver.hydrate_streams(stream_samples)
    return _aligned_sample_from_streams(
        job,
        tick_index,
        stream_samples,
        columns=columns,
        quality_policy=quality_policy,
        manifest=manifest,
        tick_plan=tick_plan,
        epoch_plan=epoch_plan,
        sample_index=sample_index,
        plan_index=plan_index,
    )


def _aligned_sample_from_streams(
    job: Mapping[str, Any],
    tick_index: int,
    stream_samples: Mapping[str, dict[str, Any]],
    *,
    columns: tuple[str, ...],
    quality_policy: Mapping[str, Any],
    manifest: AlignedTrainingManifest,
    tick_plan: AlignedTrainingTickPlan,
    epoch_plan: EpochPlan,
    sample_index: int,
    plan_index: int,
) -> dict[str, Any]:
    timestamp_ns = _aligned_tick_timestamp(stream_samples)
    run_id = _aligned_tick_run_id(stream_samples)
    quality_flags = sorted(
        {
            flag
            for stream_sample in stream_samples.values()
            for flag in stream_sample.get("quality_flags") or []
        }
    )
    values = {
        "alignment_id": job["alignment_id"],
        "alignment_name": job["name"],
        "tick_index": int(tick_index),
        "timestamp_ns": timestamp_ns,
        "run_id": run_id,
        "streams": stream_samples,
        "masks": _aligned_masks(stream_samples),
        "quality_flags": quality_flags,
        "lineage": _aligned_sample_lineage(
            job,
            stream_samples,
            manifest=manifest,
            tick_plan=tick_plan,
            epoch_plan=epoch_plan,
            sample_index=sample_index,
            plan_index=plan_index,
            tick_index=int(tick_index),
            quality_policy=quality_policy,
        ),
    }
    return {column: values[column] for column in columns}


def _aligned_stream_sample(stream: str, row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "stream": stream,
            "run_id": None,
            "timestamp_ns": None,
            "status": "filtered",
            "interpolation": None,
            "observation_id": None,
            "source_observation_ids": [],
            "source_row_ids": [],
            "source_timestamp_ns": None,
            "source_time_ns": None,
            "receive_time_ns": None,
            "latency_ns": None,
            "error_ns": None,
            "absolute_error_ns": None,
            "confidence": 0.0,
            "value_json": None,
            "value": None,
            "quality_flags": [f"training:{_aligned_stream_key(stream)}:filtered"],
            "aligned_frame_id": None,
            "transform_id": None,
        }
    return {
        "stream": stream,
        "run_id": row.get("run_id"),
        "timestamp_ns": row.get("timestamp_ns"),
        "status": row.get("status"),
        "interpolation": row.get("interpolation"),
        "observation_id": row.get("observation_id"),
        "source_observation_ids": list(row.get("source_observation_ids") or []),
        "source_row_ids": [int(row_id) for row_id in row.get("source_row_ids") or []],
        "source_timestamp_ns": row.get("source_timestamp_ns"),
        "source_time_ns": row.get("source_time_ns"),
        "receive_time_ns": row.get("receive_time_ns"),
        "latency_ns": row.get("latency_ns"),
        "error_ns": row.get("error_ns"),
        "absolute_error_ns": row.get("absolute_error_ns"),
        "confidence": float(row.get("confidence") or 0.0),
        "value_json": row.get("value_json"),
        "value": _parse_aligned_value_json(row.get("value_json")),
        "quality_flags": list(row.get("quality_flags") or []),
        "aligned_frame_id": row.get("aligned_frame_id"),
        "transform_id": row.get("transform_id"),
    }


def _aligned_tick_timestamp(stream_samples: Mapping[str, Mapping[str, Any]]) -> int | None:
    for sample in stream_samples.values():
        value = sample.get("timestamp_ns")
        if value is not None:
            return int(value)
    for sample in stream_samples.values():
        value = sample.get("source_time_ns")
        if value is not None:
            return int(value) - int(sample.get("error_ns") or 0)
    return None


def _aligned_tick_run_id(stream_samples: Mapping[str, Mapping[str, Any]]) -> str | None:
    # Materialized aligned rows carry run_id only for streams with a concrete source row.
    for sample in stream_samples.values():
        run_id = sample.get("run_id")
        if run_id:
            return str(run_id)
    for sample in stream_samples.values():
        source_ids = sample.get("source_observation_ids") or []
        if source_ids:
            return None
    return None


def _aligned_masks(stream_samples: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    masks = {
        "valid": {},
        "missing": {},
        "payload_missing": {},
        "decoder_unsupported": {},
        "stale": {},
        "interpolated": {},
        "out_of_tolerance": {},
    }
    for stream, sample in stream_samples.items():
        status = sample.get("status")
        interpolation = sample.get("interpolation")
        missing = status in {"missing", "filtered"} or not sample.get("source_observation_ids")
        out_of_tolerance = status == "out_of_tolerance"
        interpolated = (
            status == "aligned"
            and interpolation == "linear"
            and len(sample.get("source_observation_ids") or []) > 1
        )
        masks["valid"][stream] = status == "aligned"
        masks["missing"][stream] = missing
        source_rows = (sample.get("feature") or {}).get("source_rows") or []
        masks["payload_missing"][stream] = any(
            bool(row.get("missing_payload")) for row in source_rows
        )
        masks["decoder_unsupported"][stream] = any(
            bool(row.get("decoder_unsupported")) for row in source_rows
        )
        masks["stale"][stream] = out_of_tolerance
        masks["interpolated"][stream] = interpolated
        masks["out_of_tolerance"][stream] = out_of_tolerance
    return masks


def _aligned_sample_lineage(
    job: Mapping[str, Any],
    stream_samples: Mapping[str, Mapping[str, Any]],
    *,
    manifest: AlignedTrainingManifest,
    tick_plan: AlignedTrainingTickPlan,
    epoch_plan: EpochPlan,
    sample_index: int,
    plan_index: int,
    tick_index: int,
    quality_policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "alignment_job": {
            "alignment_id": job["alignment_id"],
            "name": job["name"],
            "output_table": job["output_table"],
            "recipe_digest": manifest.recipe_digest,
        },
        "storage_backend": manifest.storage_backend,
        "schema_version": manifest.schema_version,
        "backend": dict(manifest.backend),
        "epoch_backend": epoch_plan.backend.to_dict(),
        "transform_id": job["transform_id"],
        "input_table_versions": list(manifest.table_versions),
        "read_table_versions": list(manifest.read_table_versions),
        "tick_plan_id": tick_plan.plan_id,
        "epoch_plan_id": epoch_plan.plan_id,
        "epoch": epoch_plan.epoch,
        "worker_id": epoch_plan.worker_id,
        "num_workers": epoch_plan.num_workers,
        "sample_index": sample_index,
        "plan_index": plan_index,
        "tick_index": tick_index,
        "quality_policy": _jsonable(quality_policy),
        "aligned_frame_ids": {
            stream: sample.get("aligned_frame_id")
            for stream, sample in stream_samples.items()
        },
        "source_observation_ids": {
            stream: list(sample.get("source_observation_ids") or [])
            for stream, sample in stream_samples.items()
        },
        "source_row_ids": {
            stream: list(sample.get("source_row_ids") or [])
            for stream, sample in stream_samples.items()
        },
        "features": _aligned_feature_lineage(stream_samples),
    }


def _parse_aligned_value_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _aligned_feature_lineage(
    stream_samples: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    policies = {
        (sample.get("feature") or {}).get("policy")
        for sample in stream_samples.values()
        if sample.get("feature")
    }
    decoders = {
        (sample.get("feature") or {}).get("decoder")
        for sample in stream_samples.values()
        if sample.get("feature")
    }
    cache_policies = {
        (sample.get("feature") or {}).get("cache_policy")
        for sample in stream_samples.values()
        if sample.get("feature")
    }
    return {
        "policy": next(iter(policies), None),
        "decoder": next(iter(decoders), None),
        "cache_policy": next(iter(cache_policies), None),
        "streams": {
            stream: {
                "value_materialized": (sample.get("feature") or {}).get("value") is not None,
                "source_rows": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"payload", "array", "tensor"}
                    }
                    for row in (sample.get("feature") or {}).get("source_rows") or []
                ],
            }
            for stream, sample in stream_samples.items()
        },
    }


def _aligned_uri_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lance_uri": ref.get("lance_uri"),
        "raw_uri": ref.get("raw_uri"),
        "raw_channel": ref.get("raw_channel"),
        "raw_sequence": ref.get("raw_sequence"),
        "frame_index": ref.get("frame_index"),
        "video_id": ref.get("video_id"),
        "byte_range": ref.get("byte_range"),
        "gop_index": ref.get("gop_index"),
    }


def _aligned_lance_uri(
    table: str,
    column: str,
    *,
    row_id: int | None,
    source_id: Any,
) -> str:
    if row_id is not None:
        return f"lance://{table}/_rowid/{row_id}/{column}"
    return f"lance://{table}/{source_id}/{column}"


def _aligned_stream_key(stream: str) -> str:
    return str(stream).strip().strip("/").lower().replace("_", "-").replace("/", "-")


def _alignment_input_versions(job: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in (job.get("input_versions") or ()))


def _alignment_table_version(job: Mapping[str, Any], table: str) -> int | None:
    for item in job.get("input_versions") or ():
        if item.get("table") == table:
            return int(item["version"])
    return None


def _current_table_versions(lake: Lake, tables: Sequence[str]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"table": table, "version": int(lake.table(table).version), "tag": ""}
        for table in tables
    )


def _alignment_recipe_digest(job: Mapping[str, Any]) -> str:
    recipe = job.get("recipe") or ""
    try:
        payload: Any = json.loads(recipe)
    except json.JSONDecodeError:
        payload = recipe
    return "recipe-" + _stable_digest(payload)


def _materialization_policies(
    columns: tuple[str, ...],
    *,
    media_policy: str,
    decoder: str,
    cache_policy: str,
) -> dict[str, str]:
    policies: dict[str, str] = {}
    if "payload" in columns or "payload_size" in columns:
        policies["payload"] = f"{media_policy}:observations.payload_blob"
    if "video_frame" in columns:
        policies["video_frame"] = f"{media_policy}:codec-aware-frame:{decoder}"
    if policies:
        policies["cache"] = cache_policy
    return policies


def _aligned_materialization_policies(
    feature_policy: str,
    *,
    decoder: str,
    cache_policy: str,
    storage_backend: str,
) -> dict[str, str]:
    policies = {"features": f"{feature_policy}:{storage_backend}"}
    if feature_policy == "value":
        policies["value"] = "value_json"
    elif feature_policy == "uri":
        policies["payload"] = "uri:observations.payload_blob-or-video_encodings.data"
    elif feature_policy in _ALIGNED_PAYLOAD_POLICIES:
        policies["payload"] = (
            f"{feature_policy}:observations.payload_blob-or-video_encodings.data:{decoder}"
        )
    if feature_policy != "metadata":
        policies["cache"] = cache_policy
    return policies


def _table_version(context: Any, table: str) -> int | None:
    for item in context.table_versions:
        if item["table"] == table:
            return int(item["version"])
    return None


def _refs_by_episode(
    refs: tuple[_TrainingFrameRef, ...],
) -> dict[int, tuple[_TrainingFrameRef, ...]]:
    grouped: dict[int, list[_TrainingFrameRef]] = {}
    for ref in refs:
        grouped.setdefault(ref.episode.index, []).append(ref)
    return {episode: tuple(items) for episode, items in grouped.items()}


def _episode_data_index(episodes: tuple[_Episode, ...]) -> dict[str, list[int]]:
    starts: list[int] = []
    stops: list[int] = []
    index = 0
    for episode in episodes:
        starts.append(index)
        index += len(episode.observations)
        stops.append(index)
    return {"from": starts, "to": stops}


def _resolve_media_policy(media: str | bool, payloads: str | bool | None) -> str:
    media_policy = _coerce_media_policy(media, parameter="media")
    if payloads is None:
        return media_policy
    payload_policy = _coerce_media_policy(payloads, parameter="payloads")
    if media_policy != DEFAULT_MEDIA_POLICY and media_policy != payload_policy:
        raise TrainingError(
            f"conflicting media policies: media={media_policy!r}, payloads={payload_policy!r}"
        )
    return payload_policy


def _resolve_aligned_feature_policy(value: str | bool) -> str:
    if isinstance(value, bool):
        return "bytes" if value else "metadata"
    policy = str(value).lower().replace("_", "-")
    aliases = {
        "none": "metadata",
        "metadata-only": "metadata",
        "values": "value",
        "json": "value",
        "byte": "bytes",
        "payload": "bytes",
        "payloads": "bytes",
        "arrays": "array",
        "tensors": "tensor",
        "uris": "uri",
    }
    policy = aliases.get(policy, policy)
    if policy not in ALIGNED_FEATURE_POLICIES:
        raise TrainingError(
            f"features must be one of {', '.join(ALIGNED_FEATURE_POLICIES)}; got {value!r}"
        )
    return policy


def _coerce_media_policy(value: str | bool, *, parameter: str) -> str:
    if isinstance(value, bool):
        return "bytes" if value else "metadata"
    policy = str(value).lower().replace("_", "-")
    aliases = {
        "none": "metadata",
        "metadata-only": "metadata",
        "handles": "metadata",
        "byte": "bytes",
        "arrays": "array",
        "tensors": "tensor",
        "uris": "uri",
    }
    policy = aliases.get(policy, policy)
    if policy not in MEDIA_POLICIES:
        raise TrainingError(
            f"{parameter} must be one of {', '.join(MEDIA_POLICIES)}; got {value!r}"
        )
    return policy


def _validate_decoder(decoder: str) -> str:
    if decoder not in {"auto", "cpu", "nvdec"}:
        raise TrainingError("decoder must be one of auto, cpu, nvdec")
    return decoder


def _validate_media_cache(media_cache: str) -> str:
    policy = str(media_cache).lower().replace("_", "-")
    aliases = {
        "no-cache": "none",
        "off": "none",
        "local": "bounded",
        "bounded-local": "bounded",
        "epoch-local": "epoch",
    }
    policy = aliases.get(policy, policy)
    if policy not in MEDIA_CACHE_POLICIES:
        raise TrainingError(
            f"media_cache must be one of {', '.join(MEDIA_CACHE_POLICIES)}; got {media_cache!r}"
        )
    return policy


def _validate_training_backend(backend: str) -> str:
    selected = str(backend).lower().replace("_", "-")
    aliases = {
        "native": "local",
        "lance": "local",
        "lance-native": "local",
        "oss": "local",
        "remote": "enterprise",
        "db": "enterprise",
        "db-uri": "enterprise",
        "lancedb-enterprise": "enterprise",
    }
    selected = aliases.get(selected, selected)
    if selected not in TRAINING_BACKENDS:
        raise TrainingError(
            f"backend must be one of {', '.join(TRAINING_BACKENDS)}; got {backend!r}"
        )
    return selected


def _validate_enterprise_fallback_policy(
    fallback: str | bool | None,
    *,
    allow_fallback: bool,
) -> str:
    if fallback is None:
        return "local" if allow_fallback else DEFAULT_ENTERPRISE_FALLBACK_POLICY
    if isinstance(fallback, bool):
        return "local" if fallback else "fail"
    selected = str(fallback).lower().replace("_", "-")
    aliases = {
        "error": "fail",
        "raise": "fail",
        "strict": "fail",
        "warn-and-continue": "warn",
        "warning": "warn",
        "continue": "warn",
        "degrade": "warn",
        "direct-data-plane": "direct",
        "object-store": "direct",
        "pylance": "direct",
        "dev": "local",
        "test": "local",
        "local-dev": "local",
        "allow": "local",
    }
    selected = aliases.get(selected, selected)
    if selected not in ENTERPRISE_FALLBACK_POLICIES:
        raise TrainingError(
            "fallback must be one of "
            f"{', '.join(ENTERPRISE_FALLBACK_POLICIES)}; got {fallback!r}"
        )
    return selected


def _validate_enterprise_cache_policy(cache_policy: str) -> str:
    selected = str(cache_policy).lower().replace("_", "-")
    aliases = {
        "off": "none",
        "no-cache": "none",
        "report": "lazy",
        "report-only": "lazy",
        "warm": "epoch",
        "warm-epoch": "epoch",
        "per-worker": "epoch",
        "worker": "epoch",
        "job": "snapshot",
        "run": "snapshot",
        "snapshot-wide": "snapshot",
    }
    selected = aliases.get(selected, selected)
    if selected not in ENTERPRISE_CACHE_POLICIES:
        raise TrainingError(
            "cache_policy must be one of "
            f"{', '.join(ENTERPRISE_CACHE_POLICIES)}; got {cache_policy!r}"
        )
    return selected


def _native_required_enterprise_capabilities(
    columns: Sequence[str],
    *,
    media_policy: str,
    cache_policy: str,
) -> tuple[str, ...]:
    required = {
        "db_remote_connection",
        "remote_scan",
        "remote_filtered_read",
    }
    selected_cache_policy = _validate_enterprise_cache_policy(cache_policy)
    required.update(_cache_required_enterprise_capabilities(selected_cache_policy))
    materializes_heavy = media_policy in DECODED_MEDIA_POLICIES or media_policy == "bytes"
    if materializes_heavy and any(column in _HEAVY_MEDIA_COLUMNS for column in columns):
        required.update({"remote_take", "blob_or_video_remote_hydration"})
    return tuple(capability for capability in ENTERPRISE_TRAINING_CAPABILITIES if capability in required)


def _aligned_required_enterprise_capabilities(
    feature_policy: str,
    *,
    cache_policy: str,
) -> tuple[str, ...]:
    required = {
        "db_remote_connection",
        "remote_scan",
        "remote_filtered_read",
    }
    selected_cache_policy = _validate_enterprise_cache_policy(cache_policy)
    required.update(_cache_required_enterprise_capabilities(selected_cache_policy))
    if feature_policy in _ALIGNED_PAYLOAD_POLICIES:
        required.update({"remote_take", "blob_or_video_remote_hydration"})
    return tuple(capability for capability in ENTERPRISE_TRAINING_CAPABILITIES if capability in required)


def _cache_required_enterprise_capabilities(cache_policy: str) -> set[str]:
    if cache_policy == "none":
        return set()
    required = {"plan_executor_cache_metrics"}
    if cache_policy in ENTERPRISE_PREWARM_POLICIES:
        required.update({"page_cache_prewarm", "page_cache_status"})
    return required


def _normalize_prewarm_options(
    prewarm_options: Mapping[str, Any] | None,
) -> TrainingPrewarmOptions:
    raw = dict(prewarm_options or {})
    unknown = sorted(
        set(raw)
        - {
            "columns",
            "include_heavy",
            "max_rows",
            "max_bytes",
            "max_fragments",
            "timeout_s",
            "concurrency",
            "wait",
            "on_error",
        }
    )
    if unknown:
        raise TrainingError(f"unknown prewarm_options keys {unknown}")
    columns_value = raw.get("columns")
    if columns_value is None:
        columns = None
    elif isinstance(columns_value, str):
        columns = tuple(part.strip() for part in columns_value.split(",") if part.strip())
    else:
        columns = tuple(str(column) for column in columns_value)

    on_error = str(raw.get("on_error", "warn")).lower().replace("_", "-")
    on_error_aliases = {"warning": "warn", "fail": "raise", "error": "raise"}
    on_error = on_error_aliases.get(on_error, on_error)
    if on_error not in {"warn", "raise"}:
        raise TrainingError("prewarm_options['on_error'] must be 'warn' or 'raise'")

    max_rows = _optional_positive_int(
        raw.get("max_rows", DEFAULT_PREWARM_MAX_ROWS),
        "prewarm_options['max_rows']",
    )
    max_bytes = _optional_positive_int(
        raw.get("max_bytes", DEFAULT_PREWARM_MAX_BYTES),
        "prewarm_options['max_bytes']",
    )
    max_fragments = _optional_positive_int(
        raw.get("max_fragments", DEFAULT_PREWARM_MAX_FRAGMENTS),
        "prewarm_options['max_fragments']",
    )
    timeout_s = _optional_positive_float(
        raw.get("timeout_s", DEFAULT_PREWARM_TIMEOUT_S),
        "prewarm_options['timeout_s']",
    )
    concurrency = _optional_positive_int(
        raw.get("concurrency", DEFAULT_PREWARM_CONCURRENCY),
        "prewarm_options['concurrency']",
    )
    assert concurrency is not None
    return TrainingPrewarmOptions(
        columns=columns,
        include_heavy=bool(raw.get("include_heavy", False)),
        max_rows=max_rows,
        max_bytes=max_bytes,
        max_fragments=max_fragments,
        timeout_s=timeout_s,
        concurrency=concurrency,
        wait=bool(raw.get("wait", False)),
        on_error=on_error,
    )


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 1:
        raise TrainingError(f"{name} must be positive or None")
    return parsed


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0.0:
        raise TrainingError(f"{name} must be positive or None")
    return parsed


def _epoch_backend_capability(
    lake: Lake | None,
    *,
    lake_capabilities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    capabilities = (
        dict(lake_capabilities)
        if lake_capabilities is not None
        else _lake_capabilities_dict(lake)
        if lake is not None
        else {}
    )
    db = getattr(lake, "_db", None) if lake is not None else None
    permutation_module = _lancedb_permutation_module()
    lancedb_supported = (
        permutation_module is not None
        and db is not None
        and all(hasattr(db, name) for name in ("create_table", "open_table", "list_tables"))
    )
    if permutation_module is None:
        lancedb_reason = "lancedb.permutation is not importable"
    elif db is None:
        lancedb_reason = "lake does not expose a LanceDB connection"
    elif not all(hasattr(db, name) for name in ("create_table", "open_table", "list_tables")):
        lancedb_reason = "LanceDB connection cannot create/open/list permutation tables"
    else:
        lancedb_reason = "lancedb.permutation and table creation APIs are available"

    direct_lance_supported = (
        importlib.util.find_spec("lance") is not None
        and bool(capabilities.get("direct_object_io"))
    )
    direct_lance_reason = (
        "pylance and direct object IO are available"
        if direct_lance_supported
        else "direct pylance ordered-row backend is not available for this lake"
    )
    server_side_supported = bool(capabilities.get("server_side_query"))
    server_side_reason = (
        "server-side query-node planning is available; the query node can build a "
        "paginated row-plan artifact"
        if server_side_supported
        else "server-side query-node planning is not available for this lake"
    )
    return {
        EPOCH_BACKEND_PYTHON: {
            "supported": True,
            "execution_mode": "python-in-memory",
            "reason": "compatible deterministic ordering in the SDK process",
        },
        EPOCH_BACKEND_LANCEDB_PERMUTATION: {
            "supported": lancedb_supported,
            "execution_mode": "lancedb-permutation-table",
            "reason": lancedb_reason,
        },
        EPOCH_BACKEND_DIRECT_LANCE: {
            "supported": direct_lance_supported,
            "execution_mode": "direct-lance-ordered-take",
            "reason": direct_lance_reason,
        },
        EPOCH_BACKEND_SERVER_SIDE_PLAN: {
            "supported": server_side_supported,
            "execution_mode": "server-side-plan-artifact",
            "reason": server_side_reason,
        },
    }


def _lancedb_permutation_module() -> Any | None:
    if importlib.util.find_spec("lancedb.permutation") is None:
        return None
    try:
        import lancedb.permutation as permutation
    except Exception:
        return None
    if not hasattr(permutation, "Permutation") or not hasattr(
        permutation, "permutation_builder"
    ):
        return None
    return permutation


def _training_backend_report(
    lake: Lake,
    *,
    backend: str,
    cache_policy: str,
    prewarm: bool,
    fallback_policy: str,
    required_capabilities: Sequence[str] = (),
    row_plan: TrainingRowPlan | None = None,
    selected_frames: int | None = None,
    plan_id: str | None = None,
    table_versions: Sequence[Mapping[str, Any]] | None = None,
) -> TrainingBackendReport:
    requested = _validate_training_backend(backend)
    selected_cache_policy = _validate_enterprise_cache_policy(cache_policy)
    requested_cache_policy = selected_cache_policy
    fallback_policy = _validate_enterprise_fallback_policy(
        fallback_policy,
        allow_fallback=False,
    )
    spec = getattr(lake, "connection_spec", None)
    connection_kind = str(getattr(spec, "kind", "local_path") or "local_path")
    display_uri = str(getattr(spec, "display_uri", lake.uri) or lake.uri)
    connect_kwargs = dict(getattr(spec, "lancedb_connect_kwargs", {}) or {})
    host_override = connect_kwargs.get("host_override")
    namespace_properties = dict(getattr(spec, "namespace_client_properties", {}) or {})
    namespace_endpoint = namespace_properties.get("uri")
    capabilities = _lake_capabilities_dict(lake)
    capabilities["epoch_backends"] = _epoch_backend_capability(
        lake,
        lake_capabilities=capabilities,
    )
    auth_refs = dict(getattr(spec, "auth_refs", {}) or {})
    is_enterprise_connection = connection_kind in ENTERPRISE_TRAINING_CONNECTION_KINDS

    fallback: dict[str, Any] | None = None
    fallback_events: list[dict[str, Any]] = []
    warnings: list[str] = []
    if requested == "auto":
        resolved = "enterprise" if is_enterprise_connection else "local"
    elif requested == "enterprise" and not is_enterprise_connection:
        if fallback_policy == "local":
            resolved = "local"
            fallback = {
                "from": "enterprise",
                "to": "local",
                "reason": f"lake connection kind {connection_kind!r} is not Enterprise remote",
            }
            fallback_events.append(fallback)
            warnings.append(
                "enterprise backend request fell back to local Lance-native execution"
            )
        elif fallback_policy == "direct" and _direct_fallback_allowed(
            spec,
            capabilities,
            connection_kind=connection_kind,
        ):
            resolved = "local"
            fallback = {
                "from": "enterprise",
                "to": "direct-data-plane",
                "reason": f"lake connection kind {connection_kind!r} is not Enterprise remote",
                "policy": "direct",
                "preserves_versions": True,
            }
            fallback_events.append(fallback)
            warnings.append(
                "enterprise backend request fell back to direct Lance data-plane execution"
            )
        else:
            raise EnterpriseCapabilityError(
                "enterprise training backend requires a db:// or namespace-backed lake; "
                f"opened {connection_kind!r}. Pass fallback='local' for explicit "
                "local test/dev execution, or fallback='direct' when a direct "
                "Lance data-plane path is authorized."
            )
    else:
        resolved = requested

    enterprise_capabilities = _enterprise_training_capabilities(
        lake,
        spec,
        capabilities,
        connection_kind=connection_kind,
        resolved_backend=resolved,
        is_enterprise_connection=is_enterprise_connection,
    )
    capabilities = {**capabilities, **enterprise_capabilities}

    if resolved == "enterprise":
        _check_enterprise_training_prerequisites(
            lake,
            spec,
            connection_kind=connection_kind,
            connect_kwargs=connect_kwargs,
            auth_refs=auth_refs,
        )
        missing = [
            capability
            for capability in required_capabilities
            if not bool(capabilities.get(capability))
        ]
        if missing:
            negotiation = _negotiate_missing_enterprise_capabilities(
                missing,
                fallback_policy=fallback_policy,
                spec=spec,
                capabilities=capabilities,
                connection_kind=connection_kind,
                selected_cache_policy=selected_cache_policy,
                requested_cache_policy=requested_cache_policy,
            )
            resolved = negotiation["resolved_backend"]
            selected_cache_policy = negotiation["cache_policy"]
            fallback_events.extend(negotiation["fallback_events"])
            warnings.extend(negotiation["warnings"])
            if fallback is None and fallback_events:
                fallback = fallback_events[0]

    if resolved == "enterprise":
        execution_mode = (
            "enterprise-db-query-node"
            if connection_kind == "lancedb_remote_db"
            else "enterprise-namespace-query-node"
        )
    else:
        execution_mode = "local-lance-native"

    if connection_kind == "lancedb_remote_db":
        http_endpoint = host_override
        routing_mode = "host-override" if host_override else "regional-default"
    elif connection_kind in {"rest_namespace_lancedb", "namespace_lancedb"}:
        http_endpoint = namespace_endpoint
        routing_mode = "namespace-endpoint"
    else:
        http_endpoint = None
        routing_mode = "local"
    if resolved == "enterprise" and connection_kind == "lancedb_remote_db" and not host_override:
        warnings.append(
            "host_override is not set; Enterprise requests use the LanceDB client "
            "default regional endpoint instead of an explicit HTTP endpoint"
        )
    if bool(prewarm) and requested_cache_policy not in ENTERPRISE_PREWARM_POLICIES:
        warnings.append(
            f"prewarm=True is ignored for cache_policy={requested_cache_policy!r}; "
            "use cache_policy='epoch' or cache_policy='snapshot' to request prewarm"
        )

    query_node_available = bool(
        resolved == "enterprise"
        and capabilities.get("remote_scan")
        and capabilities.get("remote_take")
        and capabilities.get("remote_filtered_read")
    )
    live_client_attached = _lake_hook(lake, "query_node_client", "plan_executor_client") is not None
    if not query_node_available:
        integration_status = "not-available-for-resolved-backend"
    elif live_client_attached:
        integration_status = "live-client-attached"
    else:
        integration_status = "capability-reported"
    plan_executor = {
        "requested": resolved == "enterprise",
        "available": query_node_available,
        "remote_scan": bool(resolved == "enterprise" and capabilities.get("remote_scan")),
        "remote_take": bool(resolved == "enterprise" and capabilities.get("remote_take")),
        "remote_filtered_read": bool(
            resolved == "enterprise" and capabilities.get("remote_filtered_read")
        ),
        "cache_metrics": bool(
            resolved == "enterprise" and capabilities.get("plan_executor_cache_metrics")
        ),
        "prewarm_supported": bool(
            resolved == "enterprise" and capabilities.get("page_cache_prewarm")
        ),
        "live_client_attached": bool(query_node_available and live_client_attached),
        "integration_status": integration_status,
    }
    prewarm_requested = selected_cache_policy in ENTERPRISE_PREWARM_POLICIES
    cache = {
        "policy": selected_cache_policy,
        "scope": "plan-executor" if resolved == "enterprise" else "not-applicable",
        "prewarm_requested": prewarm_requested,
        "prewarm_executed": False,
        "prewarm_id": None,
        "prewarm_status": "not-requested",
        "prewarm_status_detail": {"status": "not-requested"},
        "prewarm_requests": [],
        "prewarm_limits": {},
    }
    if selected_cache_policy != requested_cache_policy:
        cache["requested_policy"] = requested_cache_policy
    request_routing = {
        "mode": routing_mode,
        "http_endpoint": http_endpoint,
        "host_override": host_override,
        "region": connect_kwargs.get("region"),
        "all_requests_use_host_override": bool(
            resolved == "enterprise"
            and connection_kind == "lancedb_remote_db"
            and host_override
        ),
    }
    resolved_plan_id = plan_id or (row_plan.plan_id if row_plan is not None else None)
    resolved_table_versions = (
        list(table_versions)
        if table_versions is not None
        else list(row_plan.table_versions)
        if row_plan is not None
        else None
    )
    metrics = {
        "rows_planned": selected_frames,
        "row_plan_id": resolved_plan_id,
        "cache_hits": None,
        "cache_misses": None,
        "bytes_read": None,
        "pe_fanout": None,
    }
    if selected_frames is not None:
        metrics["row_count"] = selected_frames
    if resolved_table_versions is not None:
        metrics["table_versions"] = list(resolved_table_versions)

    return TrainingBackendReport(
        requested_backend=requested,
        resolved_backend=resolved,
        execution_mode=execution_mode,
        connection_kind=connection_kind,
        display_uri=display_uri,
        request_routing=request_routing,
        capabilities=capabilities,
        plan_executor=plan_executor,
        cache=cache,
        fallback_policy=fallback_policy,
        fallback=fallback,
        fallback_events=tuple(fallback_events),
        warnings=tuple(warnings),
        metrics=metrics,
    )


def _enterprise_training_capabilities(
    lake: Lake,
    spec: Any,
    base_capabilities: Mapping[str, Any],
    *,
    connection_kind: str,
    resolved_backend: str,
    is_enterprise_connection: bool,
) -> dict[str, bool]:
    remote = resolved_backend == "enterprise"
    server_side_query = bool(base_capabilities.get("server_side_query"))
    blob_fetch_remote = bool(base_capabilities.get("blob_fetch_remote"))
    direct_object_io = _direct_fallback_allowed(
        spec,
        base_capabilities,
        connection_kind=connection_kind,
    )
    matrix = {
        "db_remote_connection": bool(is_enterprise_connection),
        "remote_scan": bool(remote and server_side_query),
        "remote_take": bool(remote and server_side_query and blob_fetch_remote),
        "remote_filtered_read": bool(remote and server_side_query),
        "plan_executor_cache_metrics": bool(remote and server_side_query and blob_fetch_remote),
        "page_cache_prewarm": bool(remote and server_side_query and blob_fetch_remote),
        "page_cache_status": bool(remote and server_side_query and blob_fetch_remote),
        "namespace_direct_object_io": bool(
            connection_kind in {"rest_namespace_lancedb", "namespace_lancedb"}
            and direct_object_io
        ),
        "managed_versioning": bool(
            getattr(spec, "managed_versioning", False)
            or base_capabilities.get("namespace_managed_versioning")
        ),
        "blob_or_video_remote_hydration": bool(remote and blob_fetch_remote),
        "server_side_row_plan": bool(remote and server_side_query),
    }
    overrides = getattr(lake, "enterprise_training_capabilities", None)
    if overrides is None:
        overrides = getattr(spec, "enterprise_training_capabilities", None)
    if isinstance(overrides, Mapping):
        for key, value in overrides.items():
            name = str(key)
            if name in matrix:
                matrix[name] = bool(value)
    return matrix


def _check_enterprise_training_prerequisites(
    lake: Lake,
    spec: Any,
    *,
    connection_kind: str,
    connect_kwargs: Mapping[str, Any],
    auth_refs: Mapping[str, Any],
) -> None:
    if getattr(lake, "enterprise_extra_available", True) is False:
        raise MissingEnterpriseAuthError(
            "Enterprise training requires the LanceDB Enterprise client extra. "
            "Remediation: install the Enterprise-enabled LanceDB client package "
            "and retry the loader setup."
        )
    if _namespace_credentials_expired(lake, spec):
        refreshed = _attempt_namespace_credential_refresh(lake, spec)
        if _namespace_credentials_expired(lake, spec):
            attempted = (
                " A single automatic credential refresh was attempted and the "
                "vendor still reports the credentials as expired."
                if refreshed
                else ""
            )
            raise NamespaceCredentialExpiredError(
                "namespace credentials are expired before Enterprise training "
                f"starts.{attempted} Remediation: refresh the namespace credential "
                "vendor or reopen the lake with a valid "
                "namespace_auth_ref/storage_auth_ref."
            )
    if connection_kind != "lancedb_remote_db":
        return
    if connect_kwargs.get("api_key") or connect_kwargs.get("client_config"):
        return
    if any(value for value in auth_refs.values()):
        return
    raise MissingEnterpriseAuthError(
        "Enterprise db:// training requires a runtime API token or remote_auth_ref. "
        "Remediation: pass remote_auth_ref to Lake.open/connect, configure a "
        "runtime auth provider, or set an Enterprise API token for the current process."
    )


def _namespace_credentials_expired(lake: Lake, spec: Any) -> bool:
    return bool(
        getattr(lake, "namespace_credentials_expired", False)
        or getattr(spec, "namespace_credentials_expired", False)
    )


def _attempt_namespace_credential_refresh(lake: Lake, spec: Any) -> bool:
    """Invoke a single namespace-credential refresh hook if one is registered.

    Enterprise deployments vend short-lived scoped credentials; a loader that
    starts just after expiry should be given exactly one chance to refresh
    before failing with a targeted diagnostic. Returns ``True`` when a refresh
    hook was found and invoked (regardless of whether it cleared the expiry),
    ``False`` when no hook is registered so callers can keep the legacy
    fail-fast behavior. The hook is looked up on the lake first, then the
    connection spec, and any hook exception is swallowed so it collapses into
    the standard :class:`NamespaceCredentialExpiredError` diagnostic.
    """

    hook = getattr(lake, "namespace_credential_refresh", None)
    if hook is None:
        hook = getattr(spec, "namespace_credential_refresh", None)
    if not callable(hook):
        return False
    try:
        hook()
    except Exception:  # noqa: BLE001 - degrade to the expired diagnostic below
        return True
    return True


def _direct_fallback_allowed(
    spec: Any,
    capabilities: Mapping[str, Any],
    *,
    connection_kind: str,
) -> bool:
    if bool(getattr(spec, "direct_object_io_allowed", False)):
        return True
    if bool(capabilities.get("direct_object_io")):
        return True
    return connection_kind == "local_path"


def _negotiate_missing_enterprise_capabilities(
    missing: Sequence[str],
    *,
    fallback_policy: str,
    spec: Any,
    capabilities: Mapping[str, Any],
    connection_kind: str,
    selected_cache_policy: str,
    requested_cache_policy: str,
) -> dict[str, Any]:
    missing = tuple(dict.fromkeys(str(item) for item in missing))
    cache_only = all(item in _CACHE_FALLBACK_CAPABILITIES for item in missing)
    fallback_events: list[dict[str, Any]] = []
    warnings: list[str] = []

    if fallback_policy == "warn" and cache_only:
        cache_policy = selected_cache_policy
        prewarm_missing = [
            item for item in missing if item in {"page_cache_prewarm", "page_cache_status"}
        ]
        metrics_missing = [
            item for item in missing if item == "plan_executor_cache_metrics"
        ]
        if prewarm_missing:
            cache_policy = "lazy"
            event = {
                "from": f"cache_policy={requested_cache_policy}",
                "to": "lazy-cache",
                "policy": "warn",
                "reason": "page-cache prewarm/status capability is unavailable",
                "missing_capabilities": prewarm_missing,
                "preserves_versions": True,
            }
            fallback_events.append(event)
            warnings.append(
                "Enterprise cache prewarm is unavailable; continuing in lazy-cache mode"
            )
        if metrics_missing:
            event = {
                "from": "plan-executor-cache-metrics",
                "to": "remote-execution-without-live-cache-metrics",
                "policy": "warn",
                "reason": "cache metrics capability is unavailable",
                "missing_capabilities": metrics_missing,
                "preserves_versions": True,
            }
            fallback_events.append(event)
            warnings.append(
                "Enterprise cache metrics are unavailable; loader report will mark "
                "cache counters as absent until a live metrics client is attached"
            )
        return {
            "resolved_backend": "enterprise",
            "cache_policy": cache_policy,
            "fallback_events": fallback_events,
            "warnings": warnings,
        }

    if fallback_policy == "direct":
        if not _direct_fallback_allowed(
            spec,
            capabilities,
            connection_kind=connection_kind,
        ):
            raise UnsupportedRemoteOperationError(
                "Enterprise direct fallback was requested, but the resolved lake "
                "does not authorize direct Lance/object-store data-plane access.",
                missing_capabilities=missing,
                remediation=(
                    "Open a namespace-backed lake with direct_object_io_allowed, "
                    "or use fallback='fail' until remote scan/take support is enabled."
                ),
            )
        event = {
            "from": "enterprise-remote-query-node",
            "to": "direct-data-plane",
            "policy": "direct",
            "reason": "required Enterprise remote capabilities are unavailable",
            "missing_capabilities": list(missing),
            "preserves_versions": True,
        }
        return {
            "resolved_backend": "local",
            "cache_policy": selected_cache_policy,
            "fallback_events": [event],
            "warnings": [
                "Enterprise training fell back to direct Lance data-plane access "
                "with an explicit report entry"
            ],
        }

    if fallback_policy == "local":
        event = {
            "from": "enterprise-remote-query-node",
            "to": "local",
            "policy": "local",
            "reason": "required Enterprise remote capabilities are unavailable",
            "missing_capabilities": list(missing),
            "preserves_versions": True,
        }
        return {
            "resolved_backend": "local",
            "cache_policy": selected_cache_policy,
            "fallback_events": [event],
            "warnings": [
                "Enterprise training fell back to explicit local test/dev execution"
            ],
        }

    if fallback_policy == "warn" and not cache_only:
        raise UnsupportedRemoteOperationError(
            "fallback='warn' can only degrade optional cache/prewarm capabilities; "
            "remote data-plane capabilities must use fallback='direct', "
            "fallback='local', or fail fast.",
            missing_capabilities=missing,
            remediation=(
                "Enable remote scan/take/filtered-read/blob hydration, or select "
                "fallback='direct' only when direct object IO is authorized."
            ),
        )

    _raise_capability_error(missing)


_CACHE_FALLBACK_CAPABILITIES = frozenset(
    {"plan_executor_cache_metrics", "page_cache_prewarm", "page_cache_status"}
)


def _raise_capability_error(missing: Sequence[str]) -> None:
    missing = tuple(dict.fromkeys(str(item) for item in missing))
    remediation = (
        "Open a db:// or REST Namespace lake with Enterprise support, configure "
        "remote_auth_ref/namespace_auth_ref credentials, or choose an explicit "
        "fallback policy when local/direct execution is intended."
    )
    if any(item in {"remote_scan", "remote_take", "remote_filtered_read"} for item in missing):
        raise UnsupportedRemoteOperationError(
            "Enterprise training backend requires remote query-node planning; "
            "a required remote operation is unsupported.",
            missing_capabilities=missing,
            remediation=remediation,
        )
    if "page_cache_prewarm" in missing or "page_cache_status" in missing:
        raise PrewarmUnavailableError(
            "Enterprise training backend cannot start because page-cache prewarm "
            "or status is required but unavailable.",
            missing_capabilities=missing,
            remediation="Use fallback='warn' to continue in lazy-cache mode, or enable page-cache prewarm/status.",
        )
    if "plan_executor_cache_metrics" in missing:
        raise CacheMetricsUnavailableError(
            "Enterprise training backend cannot start because cache metrics are "
            "required by the selected cache policy but unavailable.",
            missing_capabilities=missing,
            remediation="Use fallback='warn' to record the metrics gap, or enable plan-executor cache metrics.",
        )
    if "blob_or_video_remote_hydration" in missing:
        raise QueryNodeUnavailableError(
            "Enterprise training backend cannot hydrate blob/video payloads through "
            "the remote plan-executor path.",
            missing_capabilities=missing,
            remediation=remediation,
        )
    raise EnterpriseCapabilityError(
        "Enterprise training backend cannot start because required capabilities are missing.",
        missing_capabilities=missing,
        remediation=remediation,
    )


def _native_loader_report_payload(
    manifest: Mapping[str, Any],
    *,
    run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    backend = _mapping_dict(manifest.get("backend"))
    return _redact_report(
        _jsonable(
            {
                "kind": TRAINING_LOADER_REPORT_KIND,
                "loader": {
                    "kind": "native-training",
                    "access_pattern": manifest.get("access_pattern"),
                },
                "lake": _loader_report_lake(manifest, backend),
                "snapshot": {
                    "id": manifest.get("dataset_id"),
                    "name": manifest.get("snapshot_name"),
                },
                "table_versions": list(manifest.get("table_versions") or []),
                "plans": {
                    "row_plan_id": manifest.get("row_plan_id"),
                    "epoch_plan_id": manifest.get("epoch_plan_id"),
                    "epoch": manifest.get("epoch"),
                    "ordering_policy": manifest.get("ordering_policy"),
                    "shuffle": manifest.get("shuffle"),
                    "shuffle_seed": manifest.get("shuffle_seed"),
                    "epoch_backend": _mapping_dict(manifest.get("epoch_backend")),
                    "worker": _mapping_dict(manifest.get("worker")),
                    "selected_rows": manifest.get("selected_frames"),
                    "total_rows": manifest.get("total_frames"),
                },
                "policies": {
                    "columns": list(manifest.get("columns") or []),
                    "filters": _mapping_dict(manifest.get("filters")),
                    "time_windows": _mapping_dict(manifest.get("time_windows")),
                    "media": _mapping_dict(manifest.get("media")),
                    "enterprise_cache": _mapping_dict(backend.get("cache")),
                },
                "remote_execution": _remote_execution_report(backend),
                "metrics": _loader_report_metrics(
                    backend,
                    worker=_mapping_dict(manifest.get("worker")),
                    epoch=manifest.get("epoch"),
                ),
                "fallback_events": _fallback_events(backend),
                "disabled_capabilities": _disabled_capabilities(backend),
                "run": dict(run or {}),
            }
        )
    )


def _aligned_loader_report_payload(
    manifest: Mapping[str, Any],
    *,
    run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    backend = _mapping_dict(manifest.get("backend"))
    return _redact_report(
        _jsonable(
            {
                "kind": TRAINING_LOADER_REPORT_KIND,
                "loader": {
                    "kind": "aligned-training",
                    "access_pattern": manifest.get("access_pattern"),
                },
                "lake": _loader_report_lake(manifest, backend),
                "alignment": {
                    "id": manifest.get("alignment_id"),
                    "name": manifest.get("alignment_name"),
                    "recipe_digest": manifest.get("recipe_digest"),
                    "output_table": manifest.get("output_table"),
                },
                "table_versions": list(manifest.get("table_versions") or []),
                "read_table_versions": list(manifest.get("read_table_versions") or []),
                "plans": {
                    "tick_plan_id": manifest.get("tick_plan_id"),
                    "epoch_plan_id": manifest.get("epoch_plan_id"),
                    "epoch": manifest.get("epoch"),
                    "ordering_policy": manifest.get("ordering_policy"),
                    "epoch_backend": _mapping_dict(manifest.get("epoch_backend")),
                    "worker": _mapping_dict(manifest.get("worker")),
                    "selected_rows": manifest.get("selected_ticks"),
                    "total_rows": manifest.get("total_ticks"),
                },
                "policies": {
                    "columns": list(manifest.get("columns") or []),
                    "streams": list(manifest.get("streams") or []),
                    "quality_policy": _mapping_dict(manifest.get("quality_policy")),
                    "features": _mapping_dict(manifest.get("features")),
                    "enterprise_cache": _mapping_dict(backend.get("cache")),
                },
                "remote_execution": _remote_execution_report(backend),
                "metrics": _loader_report_metrics(
                    backend,
                    worker=_mapping_dict(manifest.get("worker")),
                    epoch=manifest.get("epoch"),
                ),
                "fallback_events": _fallback_events(backend),
                "disabled_capabilities": _disabled_capabilities(backend),
                "run": dict(run or {}),
            }
        )
    )


def _loader_report_run_hooks(
    *,
    training_run_id: str | None = None,
    model_run_id: str | None = None,
    model_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run: dict[str, Any] = {}
    if training_run_id is not None:
        run["training_run_id"] = training_run_id
    if model_run_id is not None:
        run["model_run_id"] = model_run_id
    if model_id is not None:
        run["model_id"] = model_id
    if extra:
        run.update(dict(extra))
    return run


def _loader_report_lake(
    manifest: Mapping[str, Any],
    backend: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "uri": manifest.get("lake_uri"),
        "display_uri": backend.get("display_uri") or manifest.get("lake_uri"),
        "backend_kind": backend.get("resolved_backend"),
        "connection_kind": backend.get("connection_kind"),
        "execution_mode": backend.get("execution_mode"),
        "request_routing": _mapping_dict(backend.get("request_routing")),
    }


def _remote_execution_report(backend: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "requested_backend": backend.get("requested_backend"),
        "resolved_backend": backend.get("resolved_backend"),
        "execution_mode": backend.get("execution_mode"),
        "connection_kind": backend.get("connection_kind"),
        "display_uri": backend.get("display_uri"),
        "request_routing": _mapping_dict(backend.get("request_routing")),
        "capabilities": _mapping_dict(backend.get("capabilities")),
        "plan_executor": _mapping_dict(backend.get("plan_executor")),
        "cache": _mapping_dict(backend.get("cache")),
        "fallback_policy": backend.get("fallback_policy"),
        "warnings": list(backend.get("warnings") or []),
    }


def _loader_report_metrics(
    backend: Mapping[str, Any],
    *,
    worker: Mapping[str, Any],
    epoch: Any,
) -> dict[str, Any]:
    metrics = _mapping_dict(backend.get("metrics"))
    operations = [
        dict(operation)
        for operation in metrics.get("operations") or []
        if isinstance(operation, Mapping)
    ]
    hits = _metric_int(metrics, "cache_hits")
    misses = _metric_int(metrics, "cache_misses")
    return {
        "summary": {
            key: metrics.get(key)
            for key in (
                "row_plan_id",
                "rows_planned",
                "row_count",
                "hydration_requests",
                "row_ids_requested",
                "row_ids_unique",
                "row_ids_coalesced",
                "rows_returned",
                "bytes_read",
                "pe_fanout",
                "prewarm_policy",
                "prewarm_status",
                "prewarm_row_count",
                "prewarm_projected_columns",
                "prewarm_completed_executors",
                "prewarm_failed_executors",
                "prewarm_warm_bytes",
                "prewarm_cold_bytes",
                "prewarm_duration_ms",
            )
            if key in metrics
        },
        "operations_by_type": {
            "remote_scan": _metric_int(metrics, "remote_scan_requests"),
            "remote_take": _metric_int(metrics, "remote_take_requests"),
            "remote_filtered_read": _metric_int(
                metrics,
                "remote_filtered_read_requests",
            ),
            "prewarm": _metric_int(metrics, "prewarm_requests"),
        },
        "cache": {
            "hits": hits,
            "misses": misses,
            "by_plan_executor": _mapping_dict(metrics.get("cache_by_plan_executor")),
            "by_operation": _cache_counts_by_key(operations, "operation"),
            "by_batch": _cache_counts_by_key(operations, "coalescing_window"),
            "by_worker": _cache_counts_for_worker(worker, hits=hits, misses=misses),
            "by_epoch": _cache_counts_for_epoch(epoch, hits=hits, misses=misses),
            "prewarm": {
                "hits": _metric_int(metrics, "prewarm_cache_hits"),
                "misses": _metric_int(metrics, "prewarm_cache_misses"),
            },
        },
        "operations": operations,
    }


def _cache_counts_by_key(
    operations: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for operation in operations:
        label = operation.get(key)
        if label is None:
            continue
        cache = _mapping_dict(operation.get("cache"))
        entry = counts.setdefault(str(label), {"hits": 0, "misses": 0})
        entry["hits"] += _metric_int(cache, "hits")
        entry["misses"] += _metric_int(cache, "misses")
    return counts


def _cache_counts_for_worker(
    worker: Mapping[str, Any],
    *,
    hits: int,
    misses: int,
) -> dict[str, dict[str, int]]:
    if not worker:
        return {}
    worker_id = worker.get("id")
    num_workers = worker.get("num_workers")
    if worker_id is None or num_workers is None:
        return {}
    return {f"{worker_id}/{num_workers}": {"hits": hits, "misses": misses}}


def _cache_counts_for_epoch(
    epoch: Any,
    *,
    hits: int,
    misses: int,
) -> dict[str, dict[str, int]]:
    if epoch is None:
        return {}
    return {str(epoch): {"hits": hits, "misses": misses}}


def _fallback_events(backend: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = [
        dict(event)
        for event in backend.get("fallback_events") or []
        if isinstance(event, Mapping)
    ]
    if events:
        return events
    fallback = backend.get("fallback")
    return [dict(fallback)] if isinstance(fallback, Mapping) else []


def _disabled_capabilities(backend: Mapping[str, Any]) -> list[str]:
    disabled: set[str] = set()
    capabilities = _mapping_dict(backend.get("capabilities"))
    disabled.update(str(key) for key, value in capabilities.items() if value is False)
    plan_executor = _mapping_dict(backend.get("plan_executor"))
    disabled.update(
        f"plan_executor.{key}"
        for key, value in plan_executor.items()
        if value is False
    )
    return sorted(disabled)


def _metric_int(metrics: Mapping[str, Any], key: str) -> int:
    value = metrics.get(key)
    if value is None:
        return 0
    return int(value)


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _redact_report(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            redacted[name] = (
                REPORT_REDACTION_MARKER if _secret_report_key(name) else _redact_report(item)
            )
        return redacted
    if isinstance(value, tuple):
        return [_redact_report(item) for item in value]
    if isinstance(value, list):
        return [_redact_report(item) for item in value]
    if isinstance(value, str) and _secret_report_value(value):
        return REPORT_REDACTION_MARKER
    return value


def _secret_report_key(key: str) -> bool:
    # Delegate to the single source of truth for the secret-key contract so the
    # redactor and the 0124 conformance scanner can never disagree.
    return is_secret_report_key(key)


def _secret_report_value(value: str) -> bool:
    stripped = value.strip().lower()
    return stripped.startswith(("bearer ", "basic "))


def _lake_capabilities_dict(lake: Lake) -> dict[str, Any]:
    capabilities = getattr(lake, "capabilities", None)
    if capabilities is None:
        spec = getattr(lake, "connection_spec", None)
        capabilities = getattr(spec, "capabilities", None)
    if capabilities is None:
        return {
            "server_side_query": False,
            "direct_object_io": True,
            "namespace_resolution": False,
            "namespace_managed_versioning": False,
            "geneva_worker_specs": False,
            "blob_fetch_remote": False,
        }
    if hasattr(capabilities, "__dict__"):
        return dict(capabilities.__dict__)
    return dict(capabilities)


def _validate_media_cache_size(media_cache_size: int) -> None:
    if media_cache_size < 1:
        raise TrainingError("media_cache_size must be positive")


def _validate_columns(columns: Sequence[str]) -> None:
    unknown = [column for column in columns if column not in TRAINING_COLUMNS]
    if unknown:
        raise TrainingError(
            f"unknown training columns {unknown}; choose from {', '.join(TRAINING_COLUMNS)}"
        )


def _validate_filter_keys(filters: Mapping[str, Any]) -> None:
    unknown = [column for column in filters if column not in TRAINING_COLUMNS]
    if unknown:
        raise TrainingError(
            f"unknown training filters {unknown}; choose from {', '.join(TRAINING_COLUMNS)}"
        )


def _validate_window_keys(time_windows: Mapping[str, Sequence[float]]) -> None:
    allowed = {"state_vector", "action_vector", "payload_json", "caption"}
    unknown = [column for column in time_windows if column not in allowed]
    if unknown:
        raise TrainingError(
            f"unknown time window columns {unknown}; choose from {', '.join(sorted(allowed))}"
        )


def _validate_epoch_args(
    *,
    epoch: int,
    worker_id: int,
    num_workers: int,
    resume_from: int,
) -> None:
    if epoch < 0:
        raise TrainingError("epoch must be non-negative")
    if num_workers < 1:
        raise TrainingError("num_workers must be at least 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise TrainingError("worker_id must be between 0 and num_workers - 1")
    if resume_from < 0:
        raise TrainingError("resume_from must be non-negative")


def _training_features() -> dict[str, dict[str, str]]:
    return {
        "observation_id": {"source": "observations.observation_id", "dtype": "string"},
        "episode_id": {"source": "episodes.episode_id", "dtype": "string"},
        "scenario_id": {"source": "scenarios.scenario_id", "dtype": "string"},
        "run_id": {"source": "runs.run_id", "dtype": "string"},
        "timestamp_ns": {"source": "observations.timestamp_ns", "dtype": "int64"},
        "state_vector": {"source": "observations.state_vector", "dtype": "float32[]"},
        "action_vector": {"source": "observations.action_vector", "dtype": "float32[]"},
        "payload": {"source": "observations.payload_blob", "dtype": "bytes"},
        "video_frame": {"source": "video_encodings.data", "dtype": "bytes"},
    }


def _matches_filters(
    context: Any,
    ref: _TrainingFrameRef,
    filters: Mapping[str, Any],
) -> bool:
    return all(
        _filter_matches(_column_value(context, ref, key), expected)
        for key, expected in filters.items()
    )


def _filter_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, (set, tuple, list, frozenset)):
        return value in expected
    return value == expected


def _sample_for_ref(
    context: Any,
    ref: _TrainingFrameRef,
    columns: tuple[str, ...],
    media_resolver: _TrainingMediaResolver,
) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    media_fields: dict[str, Any] = {}
    for column in columns:
        value, audit = media_resolver.value_for_column(ref, column, columns)
        sample[column] = value
        if audit is not None:
            media_fields[column] = audit
    if media_fields:
        sample["_media"] = {
            "policy": media_resolver.media_policy,
            "decoder": media_resolver.decoder,
            "cache": {
                "policy": media_resolver.cache_policy,
                "max_entries": media_resolver.cache_size,
            },
            "fields": media_fields,
        }
    return sample


def _column_value(context: Any, ref: _TrainingFrameRef, column: str) -> Any:
    obs = ref.observation
    episode = ref.episode
    scenario = episode.scenario
    if column == "observation_id":
        return obs["observation_id"]
    if column == "episode_id":
        return episode.episode_id
    if column == "scenario_id":
        return scenario["scenario_id"]
    if column == "run_id":
        return scenario["run_id"]
    if column == "split":
        return episode.split
    if column == "episode_index":
        return episode.index
    if column == "frame_index":
        return ref.frame_index
    if column == "timestamp_ns":
        return int(obs["timestamp_ns"])
    if column == "relative_time_s":
        return _relative_seconds(obs["timestamp_ns"], scenario["start_time_ns"])
    if column == "sensor_id":
        return obs.get("sensor_id")
    if column == "topic":
        return obs.get("topic")
    if column == "modality":
        return obs.get("modality")
    if column == "state_vector":
        return _vector(obs.get("state_vector"))
    if column == "action_vector":
        return _vector(obs.get("action_vector"))
    if column == "caption":
        return obs.get("caption") or episode.task
    if column == "payload_json":
        return obs.get("payload_json")
    if column == "payload_size":
        payload = _payload(context, obs)
        return len(payload) if payload else 0
    if column == "task":
        return obs.get("task_id") or episode.task
    if column == "task_index":
        return episode.task_index
    if column == "quality_flags":
        return list(obs.get("quality_flags") or [])
    if column == "raw_uri":
        return obs.get("raw_uri")
    if column == "raw_channel":
        return obs.get("raw_channel")
    if column == "raw_sequence":
        return obs.get("raw_sequence")
    if column == "message_encoding":
        return obs.get("message_encoding")
    if column == "schema_encoding":
        return obs.get("schema_encoding")
    if column == "payload":
        return _payload(context, obs)
    if column == "video_frame":
        return _video_frame(context, ref)
    raise TrainingError(f"unknown training column {column!r}")


def _payload(context: Any, obs: dict[str, Any]) -> bytes | None:
    if context is None or not _is_camera_observation(obs):
        return None
    payload = context.payload_blobs.get(obs["observation_id"], b"")
    return payload or None


def _video_frame(context: Any, ref: _TrainingFrameRef) -> bytes | None:
    obs = ref.observation
    if context is None or not _is_camera_observation(obs):
        return None

    frame_index = obs.get("frame_index")
    if frame_index is None:
        frame_index = ref.frame_index
    frame_index = int(frame_index)
    episode_id = obs.get("episode_id") or ref.episode.episode_id
    camera_key = _camera_key(obs)
    candidates = [
        row
        for row in getattr(context, "video_encodings", {}).values()
        if row.get("episode_id") == episode_id
        and row.get("camera_key") == camera_key
        and _encoding_contains_frame(row, frame_index)
    ]
    if not candidates:
        return _payload(context, obs)

    row = sorted(
        candidates,
        key=lambda item: (item.get("created_at"), item.get("encoding_id")),
    )[-1]
    encoded = getattr(context, "video_encoding_blobs", {}).get(row["encoding_id"], b"")
    if not encoded:
        return _payload(context, obs)
    return decode_frame_from_encoding(row, encoded, frame_index, decoder="auto").frame


def _encoding_contains_frame(row: dict[str, Any], frame_index: int) -> bool:
    try:
        _encoding_frame_entry(row, frame_index)
    except TrainingError:
        return False
    return True


def _encoding_frame_entry(row: dict[str, Any], frame_index: int) -> dict[str, Any]:
    for entry in json.loads(row.get("keyframe_map_json") or "[]"):
        if int(entry["first_frame_index"]) <= frame_index <= int(entry["last_frame_index"]):
            return entry
    raise TrainingError(
        f"encoding {row.get('encoding_id')!r} has no GOP containing frame {frame_index}"
    )


def _sql_predicate(column: str, expected: Any) -> str:
    if isinstance(expected, (set, tuple, list, frozenset)):
        values = list(expected)
        if not values:
            return "false"
        return f"{column} IN ({', '.join(_sql_literal(value) for value in values)})"
    return f"{column} = {_sql_literal(expected)}"


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _logical_predicate(column: str, expected: Any) -> str:
    return _sql_predicate(column, expected)


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
