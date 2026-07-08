"""LeRobot dataset adapter: inspect structured episodes and frames.

LeRobot datasets are already columnar training datasets, so this adapter is a
schema mapper rather than a message decoder. It reads local LeRobot-style
directories directly from their JSON/Parquet metadata and resolves HF Hub repo
ids through the optional LeRobot/Hugging Face stack when installed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import re
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lancedb_robotics._mp4 import Mp4MetadataError, inspect_mp4_video
from lancedb_robotics.adapters import AdapterError, AdapterInfo
from lancedb_robotics.lerobot_object_store_manifest import (
    LeRobotObjectStoreManifestCache,
    resolve_lerobot_object_store_manifest_cache,
)
from lancedb_robotics.lerobot_object_store_validation import (
    DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
    LeRobotObjectStoreValidationObject,
    object_metadata_fingerprint,
    resolve_lerobot_object_store_validation_config,
    validate_lerobot_object_store_source,
    validation_checksum_material,
)
from lancedb_robotics.sources import file_checksum
from lancedb_robotics.storage import (
    StorageConfigError,
    is_object_store_uri,
    join_uri,
    list_uri,
    open_binary_uri,
    read_text_uri,
    source_uri,
    uri_info,
)

_SUPPORTED_CODEBASE_VERSIONS = ("v2.0", "v2.1", "v3.0")
_INSTALL_HINT = "lancedb-robotics[lerobot]"
_DEFAULT_MEDIA_INSPECTION_WORKERS = 4
_DEFAULT_MEDIA_INSPECTION_RETRY_BACKOFF_SECONDS = 0.0
_DEFAULT_MEDIA_INSPECTION_EXECUTION_MODE = "thread"
_MEDIA_INSPECTION_EXECUTION_MODES = ("thread", "process")
_MEDIA_INSPECTION_START_METHOD_ENV = "LANCEDB_ROBOTICS_LEROBOT_MEDIA_INSPECTION_START_METHOD"


@dataclass(frozen=True)
class _LeRobotRoot:
    """Runtime-only access wrapper for local/HF paths and object-store roots."""

    uri: str
    path: Path | None = None
    storage_options: Mapping[str, Any] | None = None
    auth_ref: str | None = None
    manifest_cache: LeRobotObjectStoreManifestCache | None = None

    @property
    def is_remote(self) -> bool:
        return is_object_store_uri(self.uri)

    def __str__(self) -> str:
        return str(self.path) if self.path is not None else self.uri


@dataclass(frozen=True)
class LeRobotSource:
    """Resolved LeRobot dataset source and stable content identity."""

    uri: str
    root: _LeRobotRoot
    checksum: str
    digest: str
    input_uris: tuple[str, ...]
    repo_id: str | None = None
    revision: str | None = None
    identity_kind: str = "content-sha256-media-stat"
    object_store_validation: dict[str, Any] | None = None
    kind: str = "dataset"
    storage_identifier: str = "lerobot"


@dataclass(frozen=True)
class LeRobotDataset:
    """Parsed LeRobot metadata and frame rows used by ingest."""

    source: LeRobotSource
    info: dict[str, Any]
    codebase_version: str
    tasks: tuple[dict[str, Any], ...]
    episodes: tuple[dict[str, Any], ...]
    frames: tuple[dict[str, Any], ...]
    camera_keys: tuple[str, ...]
    video_files: tuple[dict[str, Any], ...]
    media_inspection: dict[str, Any]
    data_files: tuple[str, ...]
    native_loader: dict[str, Any]


@dataclass(frozen=True)
class LeRobotFrameBatch:
    """One bounded batch of normalized LeRobot frame rows."""

    data_file: str
    row_group: int
    batch_index: int
    rows: tuple[dict[str, Any], ...]
    row_count: int
    bytes_scanned: int


@dataclass(frozen=True)
class _FrameScanStats:
    frame_count: int
    start_time_ns: int
    end_time_ns: int
    episodes: tuple[dict[str, Any], ...]
    data_files: tuple[dict[str, Any], ...]


class LeRobotAdapter:
    info = AdapterInfo(name="lerobot", format="lerobot", capabilities=("inspect", "ingest"))

    def availability(self) -> dict[str, Any]:
        """Return native LeRobot/HF dependency status without importing it eagerly."""
        missing = [
            module
            for module in ("lerobot", "huggingface_hub")
            if importlib.util.find_spec(module) is None
        ]
        return {
            "available": not missing,
            "modules": ["lerobot", "huggingface_hub"],
            "missing": missing,
            "install": _INSTALL_HINT,
        }

    def inspect(
        self,
        source: str | Path,
        *,
        inspect_videos: bool = True,
        media_inspection_workers: int | None = None,
        media_inspection_timeout_seconds: float | None = None,
        media_inspection_retries: int = 0,
        media_inspection_retry_backoff_seconds: float = _DEFAULT_MEDIA_INSPECTION_RETRY_BACKOFF_SECONDS,
        media_inspection_execution_mode: str = _DEFAULT_MEDIA_INSPECTION_EXECUTION_MODE,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
        source_manifest_cache: LeRobotObjectStoreManifestCache | str | Path | None = None,
        object_store_validation_policy: str = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
        object_store_validation_sample_count: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
        object_store_validation_sample_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
        object_store_validation_strict_max_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
        **_: Any,
    ) -> dict[str, Any]:
        """Describe a LeRobot dataset without writing to the lake."""
        manifest_cache = resolve_lerobot_object_store_manifest_cache(source_manifest_cache)
        dataset = self.dataset(
            source,
            include_frames=False,
            inspect_videos=inspect_videos,
            media_inspection_workers=media_inspection_workers,
            media_inspection_timeout_seconds=media_inspection_timeout_seconds,
            media_inspection_retries=media_inspection_retries,
            media_inspection_retry_backoff_seconds=media_inspection_retry_backoff_seconds,
            media_inspection_execution_mode=media_inspection_execution_mode,
            storage_options=storage_options,
            auth_ref=auth_ref,
            source_manifest_cache=manifest_cache,
            object_store_validation_policy=object_store_validation_policy,
            object_store_validation_sample_count=object_store_validation_sample_count,
            object_store_validation_sample_bytes=object_store_validation_sample_bytes,
            object_store_validation_strict_max_bytes=object_store_validation_strict_max_bytes,
        )
        feature_schema = dict(dataset.info.get("features") or {})
        scan = _scan_frame_stats(
            dataset.source.root,
            info=dataset.info,
            tasks=dataset.tasks,
            camera_keys=dataset.camera_keys,
        )
        frame_count = scan.frame_count
        episodes = tuple(dataset.episodes or scan.episodes)
        topics = [
            {
                "topic": key,
                "message_encoding": "parquet",
                "schema_name": spec.get("dtype") if isinstance(spec, dict) else type(spec).__name__,
                "schema_encoding": "lerobot-feature",
                "message_count": frame_count,
                "start_time_ns": None,
                "end_time_ns": None,
                "can_decode": True,
            }
            for key, spec in sorted(feature_schema.items())
        ]
        if not topics:
            topics = [
                {
                    "topic": "frames",
                    "message_encoding": "parquet",
                    "schema_name": "lerobot-frame",
                    "schema_encoding": "lerobot-feature",
                    "message_count": frame_count,
                    "start_time_ns": None,
                    "end_time_ns": None,
                    "can_decode": True,
                }
            ]
        start = scan.start_time_ns
        end = scan.end_time_ns
        return {
            "adapter": self.info.name,
            "path": dataset.source.uri,
            "profile": "lerobot",
            "library": f"lerobot/{_package_version('lerobot')}",
            "native_loader": dataset.native_loader,
            "codebase_version": dataset.codebase_version,
            "fps": dataset.info.get("fps"),
            "robot_type": dataset.info.get("robot_type"),
            "message_count": frame_count,
            "frame_count": frame_count,
            "episode_count": len(episodes),
            "task_count": len(dataset.tasks),
            "schema_count": len(feature_schema),
            "channel_count": len(feature_schema),
            "chunk_count": len(dataset.data_files),
            "start_time_ns": start,
            "end_time_ns": end,
            "duration_ns": max(0, end - start),
            "indexed": True,
            "topics": topics,
            "features": feature_schema,
            "camera_keys": list(dataset.camera_keys),
            "tasks": [dict(row) for row in dataset.tasks],
            "episodes": [dict(row) for row in episodes],
            "video_files": [dict(row) for row in dataset.video_files],
            "diagnostics": _video_diagnostics(dataset.video_files),
            "media_inspection": dict(dataset.media_inspection),
            "data_files": list(dataset.data_files),
            "data_file_stats": [dict(row) for row in scan.data_files],
            "source_identity": _source_identity(dataset.source),
            "object_store_manifest": _object_store_manifest_report(dataset.source.root),
            "object_store_validation": dataset.source.object_store_validation,
            "attachments": [],
            "metadata": [],
        }

    def ingest(
        self,
        source: str | Path,
        *,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
        source_manifest_cache: LeRobotObjectStoreManifestCache | str | Path | None = None,
        object_store_validation_policy: str = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
        object_store_validation_sample_count: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
        object_store_validation_sample_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
        object_store_validation_strict_max_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
        **_: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield normalized LeRobot frame rows.

        The high-level lake ingest path consumes :meth:`dataset` so it can write
        episode/video metadata as well; this iterator exists to satisfy the
        adapter capability contract and for lightweight conformance checks.
        """
        yield from self.dataset(
            source,
            include_frames=True,
            storage_options=storage_options,
            auth_ref=auth_ref,
            source_manifest_cache=source_manifest_cache,
            object_store_validation_policy=object_store_validation_policy,
            object_store_validation_sample_count=object_store_validation_sample_count,
            object_store_validation_sample_bytes=object_store_validation_sample_bytes,
            object_store_validation_strict_max_bytes=object_store_validation_strict_max_bytes,
        ).frames

    def dataset(
        self,
        source: str | Path,
        *,
        include_frames: bool = True,
        inspect_videos: bool = True,
        media_inspection_cache: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        media_inspection_workers: int | None = None,
        media_inspection_timeout_seconds: float | None = None,
        media_inspection_retries: int = 0,
        media_inspection_retry_backoff_seconds: float = _DEFAULT_MEDIA_INSPECTION_RETRY_BACKOFF_SECONDS,
        media_inspection_execution_mode: str = _DEFAULT_MEDIA_INSPECTION_EXECUTION_MODE,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
        source_manifest_cache: LeRobotObjectStoreManifestCache | str | Path | None = None,
        object_store_validation_policy: str = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
        object_store_validation_sample_count: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
        object_store_validation_sample_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
        object_store_validation_strict_max_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
    ) -> LeRobotDataset:
        manifest_cache = resolve_lerobot_object_store_manifest_cache(source_manifest_cache)
        resolved = self.source(
            source,
            storage_options=storage_options,
            auth_ref=auth_ref,
            source_manifest_cache=manifest_cache,
            object_store_validation_policy=object_store_validation_policy,
            object_store_validation_sample_count=object_store_validation_sample_count,
            object_store_validation_sample_bytes=object_store_validation_sample_bytes,
            object_store_validation_strict_max_bytes=object_store_validation_strict_max_bytes,
        )
        info = _read_info(resolved.root)
        version = _normalize_codebase_version(info.get("codebase_version"))
        feature_schema = dict(info.get("features") or {})
        camera_keys = tuple(_camera_keys(feature_schema))
        tasks = tuple(_read_tasks(resolved.root))
        raw_frames = _read_frame_rows(resolved.root) if include_frames else []
        frames = tuple(_normalize_frames(raw_frames, info=info, tasks=tasks, camera_keys=camera_keys))
        episodes = tuple(_read_or_synthesize_episodes(resolved.root, frames, tasks))
        video_files, media_inspection = _video_files(
            resolved.root,
            info,
            camera_keys,
            episodes,
            inspect_videos=inspect_videos,
            media_inspection_cache=tuple(media_inspection_cache),
            max_workers=media_inspection_workers,
            timeout_seconds=media_inspection_timeout_seconds,
            retry_count=media_inspection_retries,
            retry_backoff_seconds=media_inspection_retry_backoff_seconds,
            execution_mode=media_inspection_execution_mode,
        )
        return LeRobotDataset(
            source=resolved,
            info=info,
            codebase_version=version,
            tasks=tasks,
            episodes=episodes,
            frames=frames,
            camera_keys=camera_keys,
            video_files=video_files,
            media_inspection=media_inspection,
            data_files=tuple(_relative_path(resolved.root, path) for path in _data_files(resolved.root)),
            native_loader=self.availability(),
        )

    def inspect_media(
        self,
        dataset: LeRobotDataset,
        *,
        media_inspection_cache: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        media_inspection_workers: int | None = None,
        media_inspection_timeout_seconds: float | None = None,
        media_inspection_retries: int = 0,
        media_inspection_retry_backoff_seconds: float = _DEFAULT_MEDIA_INSPECTION_RETRY_BACKOFF_SECONDS,
        media_inspection_execution_mode: str = _DEFAULT_MEDIA_INSPECTION_EXECUTION_MODE,
    ) -> LeRobotDataset:
        """Inspect LeRobot MP4 metadata with bounded workers and optional cache reuse."""
        video_files = tuple(_video_files_from_existing(dataset.video_files, dataset.source.root))
        inspected, report = _inspect_video_rows(
            dataset.source.root,
            video_files,
            media_inspection_cache=tuple(media_inspection_cache),
            max_workers=media_inspection_workers,
            timeout_seconds=media_inspection_timeout_seconds,
            retry_count=media_inspection_retries,
            retry_backoff_seconds=media_inspection_retry_backoff_seconds,
            execution_mode=media_inspection_execution_mode,
        )
        return replace(dataset, video_files=inspected, media_inspection=report)

    def iter_frame_batches(
        self,
        source: str | Path | LeRobotDataset,
        *,
        batch_size: int = 1024,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
        source_manifest_cache: LeRobotObjectStoreManifestCache | str | Path | None = None,
        object_store_validation_policy: str = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
        object_store_validation_sample_count: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
        object_store_validation_sample_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
        object_store_validation_strict_max_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
    ) -> Iterator[LeRobotFrameBatch]:
        """Yield normalized frame rows in bounded Parquet batches."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        dataset = (
            source
            if isinstance(source, LeRobotDataset)
            else self.dataset(
                source,
                include_frames=False,
                storage_options=storage_options,
                auth_ref=auth_ref,
                source_manifest_cache=source_manifest_cache,
                object_store_validation_policy=object_store_validation_policy,
                object_store_validation_sample_count=object_store_validation_sample_count,
                object_store_validation_sample_bytes=object_store_validation_sample_bytes,
                object_store_validation_strict_max_bytes=object_store_validation_strict_max_bytes,
            )
        )
        yield from _iter_normalized_frame_batches(
            dataset.source.root,
            info=dataset.info,
            tasks=dataset.tasks,
            camera_keys=dataset.camera_keys,
            batch_size=batch_size,
        )

    def source(
        self,
        source: str | Path,
        *,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
        source_manifest_cache: LeRobotObjectStoreManifestCache | str | Path | None = None,
        object_store_validation_policy: str = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
        object_store_validation_sample_count: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
        object_store_validation_sample_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
        object_store_validation_strict_max_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
    ) -> LeRobotSource:
        manifest_cache = resolve_lerobot_object_store_manifest_cache(source_manifest_cache)
        validation_config = resolve_lerobot_object_store_validation_config(
            object_store_validation_policy,
            sample_count=object_store_validation_sample_count,
            sample_bytes=object_store_validation_sample_bytes,
            strict_max_bytes=object_store_validation_strict_max_bytes,
        )
        root, uri, repo_id, revision = _resolve_source(
            source,
            storage_options=storage_options,
            auth_ref=auth_ref,
            source_manifest_cache=manifest_cache,
        )
        if not _is_file(root, "meta/info.json"):
            raise AdapterError(
                f"not a LeRobot dataset: {uri} is missing meta/info.json"
            )
        if (
            root.is_remote
            and root.manifest_cache is None
            and validation_config.policy != DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY
        ):
            root = replace(root, manifest_cache=LeRobotObjectStoreManifestCache())
        files = _content_files(root)
        identity_kind = "content-sha256-media-stat"
        if repo_id:
            revision = _snapshot_revision(root) or revision
            checksum = f"hf:{repo_id}@{revision or 'unresolved'}"
            identity_kind = "hf-revision" if revision else "hf-repo"
        else:
            metadata_by_rel: dict[str, dict[str, Any]] | None = {} if root.is_remote else None
            checksum = _combined_checksum(root, files, metadata_by_rel=metadata_by_rel)
            if root.is_remote:
                identity_kind = "object-store-metadata"
        if repo_id:
            metadata_by_rel = None
        object_store_validation = None
        if root.is_remote:
            try:
                object_store_validation = validate_lerobot_object_store_source(
                    root.uri,
                    objects=_object_store_validation_objects(
                        root,
                        files,
                        metadata_by_rel=metadata_by_rel,
                    ),
                    config=validation_config,
                    storage_options=root.storage_options,
                    auth_ref=root.auth_ref,
                )
            except StorageConfigError as exc:
                raise AdapterError(str(exc)) from exc
            checksum_material = validation_checksum_material(object_store_validation)
            if checksum_material is not None:
                checksum = _checksum_with_validation(checksum, checksum_material)
                identity_kind = {
                    "sampled-validation": "object-store-sampled-validation",
                    "strict-content-hash": "object-store-strict-content-hash",
                }.get(validation_config.policy, identity_kind)
        digest = hashlib.sha256(checksum.encode()).hexdigest()[:16]
        input_uris = tuple(_file_uri(path) for path in files)
        if repo_id:
            source_ref = f"hf://{repo_id}" + (f"@{revision}" if revision else "")
            input_uris = (source_ref, *input_uris)
        return LeRobotSource(
            uri=uri,
            root=root,
            checksum=checksum,
            digest=digest,
            input_uris=input_uris,
            repo_id=repo_id,
            revision=revision,
            identity_kind=identity_kind,
            object_store_validation=object_store_validation,
        )


def _resolve_source(
    source: str | Path,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
    source_manifest_cache: LeRobotObjectStoreManifestCache | None = None,
) -> tuple[_LeRobotRoot, str, str | None, str | None]:
    value = str(source)
    if is_object_store_uri(value):
        uri = value.rstrip("/")
        root = _LeRobotRoot(
            uri=uri,
            path=None,
            storage_options=dict(storage_options or {}),
            auth_ref=auth_ref,
            manifest_cache=source_manifest_cache,
        )
        if not _is_file(root, "meta/info.json"):
            raise AdapterError(f"not a LeRobot dataset: {uri} is missing meta/info.json")
        return root, uri, None, None

    path = Path(value).expanduser()
    if path.exists():
        root = path.resolve()
        if root.is_file():
            raise AdapterError(f"LeRobot source must be a dataset directory, got file: {root}")
        uri = source_uri(root)
        return _LeRobotRoot(uri=uri, path=root), uri, None, None

    parsed = _parse_hf_source(value)
    if parsed is not None:
        repo_id, revision = parsed
        root = _download_hf_dataset(repo_id, revision=revision)
        uri = f"hf://{repo_id}" + (f"@{revision}" if revision else "")
        return _LeRobotRoot(uri=source_uri(root), path=root), uri, repo_id, revision

    raise AdapterError(
        f"no such LeRobot dataset path or HF Hub repo id: {value}; "
        "pass a local dataset directory or install the LeRobot extra for HF Hub ids"
    )


def _looks_like_hf_repo_id(value: str) -> bool:
    return _parse_hf_source(value) is not None


def _parse_hf_source(value: str) -> tuple[str, str | None] | None:
    match = re.match(
        r"^(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:@(?P<revision>[A-Za-z0-9_.\-/]+))?$",
        value,
    )
    if not match:
        return None
    return match.group("repo"), match.group("revision")


def _download_hf_dataset(repo_id: str, *, revision: str | None = None) -> Path:
    if importlib.util.find_spec("huggingface_hub") is None:
        raise AdapterError(
            "LeRobot HF Hub ingest requires the optional LeRobot/Hugging Face "
            f"dependencies; install `{_INSTALL_HINT}` and retry"
        )
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision)
        ).resolve()
    except Exception as exc:  # noqa: BLE001 - external client errors need a stable adapter message.
        raise AdapterError(f"cannot download LeRobot dataset {repo_id!r} from HF Hub: {exc}") from exc


def _snapshot_revision(root: _LeRobotRoot) -> str | None:
    name = Path(str(root)).name
    return name if re.match(r"^[0-9a-f]{40}$", name) else None


def _read_info(root: _LeRobotRoot) -> dict[str, Any]:
    try:
        payload = json.loads(_read_text(root, "meta/info.json"))
    except json.JSONDecodeError as exc:
        raise AdapterError(f"invalid LeRobot meta/info.json in {root}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdapterError(f"invalid LeRobot meta/info.json in {root}: expected object")
    return payload


def _normalize_codebase_version(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise AdapterError(
            "LeRobot dataset declares no codebase_version in meta/info.json; "
            f"supported versions are {', '.join(_SUPPORTED_CODEBASE_VERSIONS)}"
        )
    normalized = raw if raw.startswith("v") else f"v{raw}"
    if normalized not in _SUPPORTED_CODEBASE_VERSIONS:
        raise AdapterError(
            f"unsupported LeRobot codebase_version {raw!r}; supported versions are "
            f"{', '.join(_SUPPORTED_CODEBASE_VERSIONS)}"
        )
    return normalized


def _read_tasks(root: _LeRobotRoot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jsonl = "meta/tasks.jsonl"
    if _is_file(root, jsonl):
        rows.extend(_read_jsonl(root, jsonl))
    parquet = "meta/tasks.parquet"
    if not rows and _is_file(root, parquet):
        rows.extend(_read_parquet_rows(root, parquet))
    normalized: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        task = row.get("task") or row.get("name") or row.get("__index_level_0__")
        task_index = row.get("task_index", row.get("index", ordinal))
        if task is None:
            continue
        normalized.append({"task_index": int(task_index or 0), "task": str(task)})
    return sorted(_dedupe_dicts(normalized, "task_index"), key=lambda row: row["task_index"])


def _read_or_synthesize_episodes(
    root: _LeRobotRoot,
    frames: tuple[dict[str, Any], ...],
    tasks: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _glob(root, "meta/episodes/**/*.parquet"):
        rows.extend(_read_parquet_rows(root, path))
    jsonl = "meta/episodes.jsonl"
    if not rows and _is_file(root, jsonl):
        rows.extend(_read_jsonl(root, jsonl))
    if rows:
        return [_normalize_episode_row(row, frames, tasks) for row in rows]
    return [_synthesized_episode(index, rows, tasks) for index, rows in _frames_by_episode(frames).items()]


def _normalize_episode_row(
    row: dict[str, Any],
    frames: tuple[dict[str, Any], ...],
    tasks: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    episode_index = int(row.get("episode_index") or 0)
    episode_frames = [frame for frame in frames if frame["_episode_index"] == episode_index]
    task = _episode_task(row, episode_frames, tasks)
    length = int(row.get("length") or len(episode_frames))
    return {
        "episode_index": episode_index,
        "episode_id": str(row.get("episode_id") or f"episode-{episode_index:06d}"),
        "scenario_id": str(row.get("scenario_id") or f"lerobot-episode-{episode_index:06d}"),
        "run_id": str(row.get("run_id") or ""),
        "split": str(row.get("split") or "train"),
        "tasks": [task] if task else [str(item) for item in row.get("tasks") or []],
        "length": length,
        "dataset_from_index": _optional_int(row.get("dataset_from_index")),
        "dataset_to_index": _optional_int(row.get("dataset_to_index")),
    }


def _synthesized_episode(
    episode_index: int,
    frames: list[dict[str, Any]],
    tasks: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    task = _episode_task({}, frames, tasks)
    return {
        "episode_index": episode_index,
        "episode_id": f"episode-{episode_index:06d}",
        "scenario_id": f"lerobot-episode-{episode_index:06d}",
        "run_id": "",
        "split": "train",
        "tasks": [task],
        "length": len(frames),
        "dataset_from_index": None,
        "dataset_to_index": None,
    }


def _episode_task(
    row: dict[str, Any],
    frames: list[dict[str, Any]],
    tasks: tuple[dict[str, Any], ...],
) -> str:
    declared = row.get("tasks")
    if isinstance(declared, list) and declared:
        return str(declared[0])
    if row.get("task"):
        return str(row["task"])
    if frames:
        frame_task = frames[0].get("_task")
        if frame_task:
            return str(frame_task)
        task_index = frames[0].get("_task_index")
        by_index = {task["task_index"]: task["task"] for task in tasks}
        if task_index in by_index:
            return str(by_index[task_index])
    return "unknown task"


def _read_frame_rows(root: _LeRobotRoot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _data_files(root):
        rel = _relative_path(root, path)
        for row in _read_parquet_rows(root, path):
            rows.append({**row, "_source_parquet": rel})
    if not rows:
        raise AdapterError(f"LeRobot dataset {root} has no Parquet frame files under data/")
    return rows


def _normalize_frames(
    rows: list[dict[str, Any]],
    *,
    info: dict[str, Any],
    tasks: tuple[dict[str, Any], ...],
    camera_keys: tuple[str, ...],
    state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    fps = float(info.get("fps") or 1.0)
    task_by_index = {int(task["task_index"]): str(task["task"]) for task in tasks}
    normalized: list[dict[str, Any]] = []
    if state is None:
        state = {"ordinal": 0, "per_episode_frame": {}}
    per_episode_frame: dict[int, int] = state.setdefault("per_episode_frame", {})
    for row in rows:
        ordinal = int(state.get("ordinal") or 0)
        state["ordinal"] = ordinal + 1
        episode_index = int(row.get("episode_index") or _episode_index_from_path(row) or 0)
        fallback_frame = per_episode_frame.get(episode_index, 0)
        per_episode_frame[episode_index] = fallback_frame + 1
        frame_index = int(row.get("frame_index") if row.get("frame_index") is not None else fallback_frame)
        task_index = _optional_int(row.get("task_index"))
        task = str(row.get("task") or task_by_index.get(task_index or 0) or "unknown task")
        timestamp_ns = _timestamp_ns(row, episode_index=episode_index, frame_index=frame_index, fps=fps)
        images = {
            key: _jsonable(_feature_value(row, f"observation.images.{key}"))
            for key in camera_keys
            if _feature_value(row, f"observation.images.{key}") is not None
        }
        recognized = _recognized_frame_keys(camera_keys)
        unmapped = {
            key: _jsonable(value)
            for key, value in row.items()
            if key not in recognized and not key.startswith("_")
        }
        normalized.append(
            {
                **row,
                "_global_index": int(row.get("index") if row.get("index") is not None else ordinal),
                "_episode_index": episode_index,
                "_frame_index": frame_index,
                "_timestamp_ns": timestamp_ns,
                "_timestamp_s": float(row.get("timestamp") or 0.0),
                "_state_vector": _float_vector(_feature_value(row, "observation.state")),
                "_action_vector": _float_vector(_feature_value(row, "action")),
                "_task_index": task_index if task_index is not None else 0,
                "_task": task,
                "_caption": str(row.get("language_instruction") or row.get("caption") or task),
                "_images": images,
                "_unmapped": unmapped,
            }
        )
    return normalized


def _iter_normalized_frame_batches(
    root: _LeRobotRoot,
    *,
    info: dict[str, Any],
    tasks: tuple[dict[str, Any], ...],
    camera_keys: tuple[str, ...],
    batch_size: int,
) -> Iterator[LeRobotFrameBatch]:
    state: dict[str, Any] = {"ordinal": 0, "per_episode_frame": {}}
    for path in _data_files(root):
        rel = _relative_path(root, path)
        with _open_binary(root, path) as stream:
            parquet = pq.ParquetFile(stream)
            for row_group in range(parquet.num_row_groups):
                metadata = parquet.metadata.row_group(row_group)
                bytes_scanned = int(metadata.total_byte_size)
                for batch_index, record_batch in enumerate(
                    parquet.iter_batches(batch_size=batch_size, row_groups=[row_group])
                ):
                    raw_rows = [{**row, "_source_parquet": rel} for row in record_batch.to_pylist()]
                    rows = _normalize_frames(
                        raw_rows,
                        info=info,
                        tasks=tasks,
                        camera_keys=camera_keys,
                        state=state,
                    )
                    yield LeRobotFrameBatch(
                        data_file=rel,
                        row_group=row_group,
                        batch_index=batch_index,
                        rows=tuple(rows),
                        row_count=len(rows),
                        bytes_scanned=bytes_scanned,
                    )


def _scan_frame_stats(
    root: _LeRobotRoot,
    *,
    info: dict[str, Any],
    tasks: tuple[dict[str, Any], ...],
    camera_keys: tuple[str, ...],
) -> _FrameScanStats:
    frame_count = 0
    start_time_ns: int | None = None
    end_time_ns: int | None = None
    per_episode: dict[int, dict[str, Any]] = {}
    data_files: list[dict[str, Any]] = []
    for batch in _iter_normalized_frame_batches(
        root,
        info=info,
        tasks=tasks,
        camera_keys=camera_keys,
        batch_size=4096,
    ):
        data_files.append(
            {
                "path": batch.data_file,
                "row_group": batch.row_group,
                "batch_index": batch.batch_index,
                "rows": batch.row_count,
                "bytes_scanned": batch.bytes_scanned,
            }
        )
        for frame in batch.rows:
            frame_count += 1
            timestamp_ns = int(frame["_timestamp_ns"])
            start_time_ns = timestamp_ns if start_time_ns is None else min(start_time_ns, timestamp_ns)
            end_time_ns = timestamp_ns if end_time_ns is None else max(end_time_ns, timestamp_ns)
            episode_index = int(frame["_episode_index"])
            episode = per_episode.setdefault(
                episode_index,
                {
                    "episode_index": episode_index,
                    "episode_id": f"episode-{episode_index:06d}",
                    "scenario_id": f"lerobot-episode-{episode_index:06d}",
                    "run_id": "",
                    "split": "train",
                    "tasks": [str(frame["_task"])],
                    "length": 0,
                    "dataset_from_index": int(frame["_global_index"]),
                    "dataset_to_index": int(frame["_global_index"]) + 1,
                },
            )
            episode["length"] = int(episode["length"]) + 1
            episode["dataset_from_index"] = min(
                int(episode["dataset_from_index"]), int(frame["_global_index"])
            )
            episode["dataset_to_index"] = max(
                int(episode["dataset_to_index"]), int(frame["_global_index"]) + 1
            )
    if frame_count == 0:
        raise AdapterError(f"LeRobot dataset {root} has no Parquet frame files under data/")
    return _FrameScanStats(
        frame_count=frame_count,
        start_time_ns=start_time_ns or 0,
        end_time_ns=end_time_ns or 0,
        episodes=tuple(per_episode[index] for index in sorted(per_episode)),
        data_files=tuple(data_files),
    )


def _timestamp_ns(row: dict[str, Any], *, episode_index: int, frame_index: int, fps: float) -> int:
    if row.get("timestamp_ns") is not None:
        return int(row["timestamp_ns"])
    episode_offset_ns = int(episode_index) * 1_000_000_000_000
    if row.get("timestamp") is not None:
        return episode_offset_ns + int(round(float(row["timestamp"]) * 1_000_000_000))
    frame_delta = int(round(1_000_000_000 / max(fps, 1.0)))
    return episode_offset_ns + frame_index * frame_delta


def _feature_value(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    parts = key.split(".")
    value: Any = row
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _camera_keys(features: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for name in features:
        if name.startswith("observation.images."):
            keys.append(name.removeprefix("observation.images."))
        elif name.startswith("observation.image."):
            keys.append(name.removeprefix("observation.image."))
    return sorted(dict.fromkeys(keys))


def _video_files(
    root: _LeRobotRoot,
    info: dict[str, Any],
    camera_keys: tuple[str, ...],
    episodes: tuple[dict[str, Any], ...],
    *,
    inspect_videos: bool = True,
    media_inspection_cache: tuple[dict[str, Any], ...] = (),
    max_workers: int | None = None,
    timeout_seconds: float | None = None,
    retry_count: int = 0,
    retry_backoff_seconds: float = _DEFAULT_MEDIA_INSPECTION_RETRY_BACKOFF_SECONDS,
    execution_mode: str = _DEFAULT_MEDIA_INSPECTION_EXECUTION_MODE,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    by_key_episode: dict[tuple[str, int], str | Path] = {}
    expected_frames = {
        int(episode["episode_index"]): _optional_int(episode.get("length"))
        for episode in episodes
    }
    template = info.get("video_path")
    chunks_size = int(info.get("chunks_size") or 1000)
    if isinstance(template, str) and template:
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            chunk_index = episode_index // max(chunks_size, 1)
            for camera_key in camera_keys:
                video_key = f"observation.images.{camera_key}"
                try:
                    rel = template.format(
                        chunk_index=chunk_index,
                        episode_chunk=chunk_index,
                        file_index=0,
                        episode_index=episode_index,
                        video_key=video_key,
                        camera_key=camera_key,
                    )
                except KeyError:
                    continue
                by_key_episode[(camera_key, episode_index)] = _join(root, rel)

    for path in _glob(root, "videos/**/*"):
        if not _is_file(root, path):
            continue
        camera_key = _infer_camera_key(path, camera_keys)
        if camera_key is None:
            continue
        episode_index = _episode_index_from_name(path)
        if episode_index is None:
            continue
        by_key_episode.setdefault((camera_key, episode_index), path)

    rows: list[dict[str, Any]] = []
    for (camera_key, episode_index), path in sorted(by_key_episode.items()):
        rows.append(
            _video_file_row(
                root,
                path,
                camera_key=camera_key,
                episode_index=episode_index,
                expected_frame_count=expected_frames.get(episode_index),
                dataset_fps=_optional_float(info.get("fps")),
            )
        )
    if not inspect_videos:
        return tuple(rows), _media_inspection_report(
            rows,
            max_workers=_bounded_media_workers(max_workers),
            inspect_videos=False,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            execution_mode=execution_mode,
        )
    return _inspect_video_rows(
        root,
        tuple(rows),
        media_inspection_cache=media_inspection_cache,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        retry_backoff_seconds=retry_backoff_seconds,
        execution_mode=execution_mode,
    )


def _video_file_row(
    root: _LeRobotRoot,
    path: str | Path,
    *,
    camera_key: str,
    episode_index: int,
    expected_frame_count: int | None,
    dataset_fps: float | None,
) -> dict[str, Any]:
    rel = _relative_path(root, path)
    uri = _file_uri(path)
    diagnostics: list[dict[str, Any]] = []
    codec = _codec_from_path(path)
    exists = _is_file(root, path)
    metadata = _safe_info(root, path) if exists else {}
    size = _metadata_size(metadata)

    if not exists:
        diagnostics.append(
            _video_diagnostic(
                "missing-video",
                "error",
                camera_key=camera_key,
                episode_index=episode_index,
                path=rel,
                message=f"declared LeRobot video file is missing: {rel}",
            )
        )

    return {
        "camera_key": camera_key,
        "episode_index": episode_index,
        "path": rel,
        "uri": uri,
        "codec": codec,
        "codec_tag": None,
        "codec_profile": None,
        "width": None,
        "height": None,
        "resolution": None,
        "fps": dataset_fps,
        "frame_count": None,
        "expected_frame_count": expected_frame_count,
        "gop_size": None,
        "duration_seconds": None,
        "keyframe_map": [],
        "diagnostics": diagnostics,
        "size": size,
        "inspection_status": "missing" if diagnostics else "pending",
        "inspection_fingerprint": _video_inspection_fingerprint(
            uri=uri,
            size=size,
            metadata=_fingerprint_metadata(metadata),
            expected_frame_count=expected_frame_count,
        ),
        "object_metadata": _fingerprint_metadata(metadata),
        "inspection_bytes_read": 0,
        "inspection_duration_ms": 0.0,
        "inspection_attempts": 0,
        "inspection_retries": 0,
        "inspection_timeouts": 0,
        "inspection_error_class": None,
        "inspection_attempt_errors": [],
        "inspection_reused": False,
        "inspection_reused_from": None,
        "inspection_error": None,
        "inspection_execution": None,
        "inspection_worker_killed": False,
    }


def _video_files_from_existing(
    video_files: tuple[dict[str, Any], ...],
    root: _LeRobotRoot,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for video in video_files:
        row = dict(video)
        if "inspection_fingerprint" not in row:
            path = _join(root, str(row["path"]))
            metadata = _safe_info(root, path) if _is_file(root, path) else {}
            row["inspection_fingerprint"] = _video_inspection_fingerprint(
                uri=str(row.get("uri") or _file_uri(path)),
                size=int(row.get("size") or _metadata_size(metadata)),
                metadata=_fingerprint_metadata(metadata),
                expected_frame_count=_optional_int(row.get("expected_frame_count")),
            )
        rows.append(row)
    return tuple(rows)


def _inspect_video_rows(
    root: _LeRobotRoot,
    rows: tuple[dict[str, Any], ...],
    *,
    media_inspection_cache: tuple[dict[str, Any], ...] = (),
    max_workers: int | None = None,
    timeout_seconds: float | None = None,
    retry_count: int = 0,
    retry_backoff_seconds: float = _DEFAULT_MEDIA_INSPECTION_RETRY_BACKOFF_SECONDS,
    execution_mode: str = _DEFAULT_MEDIA_INSPECTION_EXECUTION_MODE,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    timeout_seconds = _normalize_timeout_seconds(timeout_seconds)
    retry_count = max(0, int(retry_count))
    retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
    execution_mode = _normalize_media_inspection_execution_mode(execution_mode)
    if not rows:
        return (), _media_inspection_report(
            (),
            max_workers=_bounded_media_workers(max_workers),
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
            execution_mode=execution_mode,
        )
    cache = _media_cache_by_fingerprint(media_inspection_cache)
    worker_count = min(_bounded_media_workers(max_workers), max(len(rows), 1))
    inspected: list[dict[str, Any] | None] = [None] * len(rows)
    if execution_mode == "process":
        pending_process_rows = _pending_process_video_rows(rows, cache, inspected)
        _inspect_video_rows_with_process_timeout(
            root,
            pending_process_rows,
            cache=cache,
            inspected=inspected,
            worker_count=worker_count,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    elif timeout_seconds is None:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _inspect_video_row,
                    root,
                    row,
                    cache,
                    retry_count=retry_count,
                    retry_backoff_seconds=retry_backoff_seconds,
                ): index
                for index, row in enumerate(rows)
            }
            for future in as_completed(futures):
                inspected[futures[future]] = future.result()
    else:
        _inspect_video_rows_with_timeout(
            root,
            rows,
            cache=cache,
            inspected=inspected,
            worker_count=worker_count,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    completed = tuple(row for row in inspected if row is not None)
    return completed, _media_inspection_report(
        completed,
        max_workers=worker_count,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        retry_backoff_seconds=retry_backoff_seconds,
        execution_mode=execution_mode,
    )


def _inspect_video_rows_with_timeout(
    root: _LeRobotRoot,
    rows: tuple[dict[str, Any], ...],
    *,
    cache: dict[str, dict[str, Any]],
    inspected: list[dict[str, Any] | None],
    worker_count: int,
    timeout_seconds: float,
    retry_count: int,
    retry_backoff_seconds: float,
) -> None:
    pending = list(enumerate(rows))
    futures: dict[Any, tuple[int, dict[str, Any], float, float]] = {}
    executor = ThreadPoolExecutor(max_workers=worker_count)

    def submit_available() -> None:
        while pending and len(futures) < worker_count:
            index, row = pending.pop(0)
            started = time.perf_counter()
            future = executor.submit(
                _inspect_video_row,
                root,
                row,
                cache,
                retry_count=retry_count,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            futures[future] = (index, row, started, started + timeout_seconds)

    try:
        submit_available()
        while futures:
            ready = {future for future in futures if future.done()}
            if not ready:
                now = time.perf_counter()
                next_deadline = min(deadline for _, _, _, deadline in futures.values())
                timeout = max(0.0, next_deadline - now)
                ready, _ = wait(tuple(futures), timeout=timeout, return_when=FIRST_COMPLETED)
            for future in tuple(ready):
                if future not in futures:
                    continue
                index, _row, _started, _deadline = futures.pop(future)
                inspected[index] = future.result()
            now = time.perf_counter()
            expired = [
                future
                for future, (_index, _row, _started, deadline) in futures.items()
                if deadline <= now
            ]
            for future in expired:
                index, row, started, _deadline = futures.pop(future)
                future.cancel()
                inspected[index] = _timeout_video_row(
                    row,
                    timeout_seconds=timeout_seconds,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                )
            submit_available()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _inspect_video_rows_with_process_timeout(
    root: _LeRobotRoot,
    rows: list[tuple[int, dict[str, Any]]],
    *,
    cache: dict[str, dict[str, Any]],
    inspected: list[dict[str, Any] | None],
    worker_count: int,
    timeout_seconds: float | None,
    retry_count: int,
    retry_backoff_seconds: float,
) -> None:
    context = _media_inspection_process_context()
    root_payload = _media_inspection_root_payload(root)
    pending = list(rows)
    active: dict[int, tuple[Any, Any, dict[str, Any], float, float | None]] = {}

    def close_queue(result_queue: Any) -> None:
        cancel_join = getattr(result_queue, "cancel_join_thread", None)
        if callable(cancel_join):
            cancel_join()
        close = getattr(result_queue, "close", None)
        if callable(close):
            close()

    def submit_available() -> None:
        while pending and len(active) < worker_count:
            index, row = pending.pop(0)
            result_queue = context.Queue(maxsize=1)
            process = context.Process(
                target=_inspect_video_row_process_target,
                args=(
                    result_queue,
                    root_payload,
                    row,
                    cache,
                    retry_count,
                    retry_backoff_seconds,
                ),
            )
            process.daemon = True
            started = time.perf_counter()
            process.start()
            deadline = started + timeout_seconds if timeout_seconds is not None else None
            active[index] = (process, result_queue, row, started, deadline)

    submit_available()
    while active:
        progressed = False
        for index, (process, result_queue, row, started, _deadline) in tuple(active.items()):
            try:
                message = result_queue.get_nowait()
            except queue.Empty:
                message = None
            if message is not None:
                process.join(timeout=0.1)
                close_queue(result_queue)
                active.pop(index, None)
                inspected[index] = _process_result_video_row(
                    message,
                    row,
                    started=started,
                )
                progressed = True
                continue
            if process.is_alive():
                continue
            process.join(timeout=0.1)
            close_queue(result_queue)
            active.pop(index, None)
            inspected[index] = _process_error_video_row(
                row,
                error_class="ProcessExitError",
                error=f"media inspection worker exited with code {process.exitcode}",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
            progressed = True

        now = time.perf_counter()
        for index, (process, result_queue, row, started, deadline) in tuple(active.items()):
            if deadline is None or deadline > now:
                continue
            active.pop(index, None)
            worker_killed = _terminate_media_inspection_process(process)
            close_queue(result_queue)
            inspected[index] = _timeout_video_row(
                row,
                timeout_seconds=float(timeout_seconds),
                duration_ms=(time.perf_counter() - started) * 1000.0,
                execution_mode="process",
                worker_killed=worker_killed,
            )
            progressed = True

        submit_available()
        if active and not progressed:
            deadlines = [deadline for *_rest, deadline in active.values() if deadline is not None]
            sleep_for = 0.01
            if deadlines:
                sleep_for = min(sleep_for, max(0.0, min(deadlines) - time.perf_counter()))
            time.sleep(sleep_for)


def _inspect_video_row_process_target(
    result_queue: Any,
    root_payload: Mapping[str, Any],
    row: Mapping[str, Any],
    cache: Mapping[str, Mapping[str, Any]],
    retry_count: int,
    retry_backoff_seconds: float,
) -> None:
    try:
        root = _media_inspection_root_from_payload(root_payload)
        inspected = _inspect_video_row(
            root,
            dict(row),
            {str(key): dict(value) for key, value in cache.items()},
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        inspected["inspection_execution"] = "process"
        inspected.setdefault("inspection_worker_killed", False)
        result_queue.put(("ok", inspected))
    except BaseException as exc:  # pragma: no cover - parent converts this to a diagnostic.
        result_queue.put(("error", exc.__class__.__name__, str(exc)))


def _media_inspection_process_context():
    try:
        methods = mp.get_all_start_methods()
    except RuntimeError:  # pragma: no cover - platform-specific multiprocessing state.
        methods = []
    requested = os.environ.get(_MEDIA_INSPECTION_START_METHOD_ENV)
    if requested:
        method = requested.strip().lower()
        if method not in methods:
            expected = ", ".join(methods) or "<none>"
            raise ValueError(
                f"{_MEDIA_INSPECTION_START_METHOD_ENV}={requested!r} is not supported; "
                f"available multiprocessing start methods: {expected}"
            )
        return mp.get_context(method)
    if "spawn" in methods:
        return mp.get_context("spawn")
    return mp.get_context()


def _media_inspection_root_payload(root: _LeRobotRoot) -> dict[str, Any]:
    return {
        "uri": root.uri,
        "path": str(root.path) if root.path is not None else None,
        "storage_options": dict(root.storage_options or {}),
        "auth_ref": root.auth_ref,
    }


def _media_inspection_root_from_payload(payload: Mapping[str, Any]) -> _LeRobotRoot:
    path_value = payload.get("path")
    return _LeRobotRoot(
        uri=str(payload["uri"]),
        path=Path(str(path_value)) if path_value else None,
        storage_options=dict(payload.get("storage_options") or {}),
        auth_ref=payload.get("auth_ref"),
    )


def _terminate_media_inspection_process(process: Any) -> bool:
    killed = False
    if process.is_alive():
        process.terminate()
        killed = True
        process.join(timeout=0.25)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        killed = True
        process.join(timeout=0.25)
    if not process.is_alive():
        process.join(timeout=0.1)
    return killed


def _process_result_video_row(
    message: Any,
    row: dict[str, Any],
    *,
    started: float,
) -> dict[str, Any]:
    if isinstance(message, tuple) and len(message) == 2 and message[0] == "ok":
        inspected = dict(message[1])
        inspected["inspection_execution"] = "process"
        inspected.setdefault("inspection_worker_killed", False)
        return inspected
    if isinstance(message, tuple) and len(message) >= 3 and message[0] == "error":
        return _process_error_video_row(
            row,
            error_class=str(message[1]),
            error=str(message[2]),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
    return _process_error_video_row(
        row,
        error_class="ProcessResultError",
        error="media inspection worker returned an unrecognized result",
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def _process_error_video_row(
    row: dict[str, Any],
    *,
    error_class: str,
    error: str,
    duration_ms: float,
) -> dict[str, Any]:
    diagnostics = [dict(item) for item in row.get("diagnostics") or []]
    diagnostics.append(
        _video_diagnostic(
            "corrupt-video",
            "error",
            camera_key=str(row["camera_key"]),
            episode_index=int(row["episode_index"]),
            path=str(row["path"]),
            message=f"media inspection worker failed for {row['path']}: {error}",
            error_class=error_class,
            attempts=1,
        )
    )
    inspected = {
        **row,
        "diagnostics": diagnostics,
        "inspection_status": "failed",
        "inspection_bytes_read": 0,
        "inspection_duration_ms": duration_ms,
        "inspection_attempts": 1,
        "inspection_retries": 0,
        "inspection_timeouts": 0,
        "inspection_error_class": error_class,
        "inspection_attempt_errors": [
            {
                "attempt": 1,
                "error_class": error_class,
                "error": error,
                "duration_ms": duration_ms,
            }
        ],
        "inspection_reused": False,
        "inspection_reused_from": None,
        "inspection_error": error,
        "inspection_execution": "process",
        "inspection_worker_killed": False,
    }
    inspected["inspection_id"] = _video_inspection_id(inspected)
    return inspected


def _pending_process_video_rows(
    rows: tuple[dict[str, Any], ...],
    cache: dict[str, dict[str, Any]],
    inspected: list[dict[str, Any] | None],
) -> list[tuple[int, dict[str, Any]]]:
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        resolved = _cached_or_terminal_video_row(row, cache, execution_mode="cache")
        if resolved is None:
            pending.append((index, row))
        else:
            inspected[index] = resolved
    return pending


def _cached_or_terminal_video_row(
    row: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    *,
    execution_mode: str,
) -> dict[str, Any] | None:
    cached = cache.get(str(row.get("inspection_fingerprint") or ""))
    if cached is not None:
        reused = dict(cached)
        reused["inspection_reused"] = True
        reused["inspection_reused_from"] = cached.get("inspection_id")
        reused["inspection_duration_ms"] = 0.0
        reused["inspection_bytes_read"] = 0
        reused["inspection_attempts"] = 0
        reused["inspection_retries"] = 0
        reused["inspection_timeouts"] = 0
        reused["inspection_error_class"] = None
        reused["inspection_attempt_errors"] = []
        reused["inspection_error"] = None
        reused["inspection_execution"] = execution_mode
        reused["inspection_worker_killed"] = False
        return reused
    if row.get("inspection_status") == "completed":
        completed = dict(row)
        completed.setdefault("inspection_id", _video_inspection_id(completed))
        completed["inspection_reused"] = bool(completed.get("inspection_reused"))
        completed["inspection_reused_from"] = completed.get("inspection_reused_from")
        completed.setdefault("inspection_attempts", 0)
        completed.setdefault("inspection_retries", 0)
        completed.setdefault("inspection_timeouts", 0)
        completed.setdefault("inspection_error_class", None)
        completed.setdefault("inspection_attempt_errors", [])
        completed.setdefault("inspection_execution", execution_mode)
        completed.setdefault("inspection_worker_killed", False)
        return completed
    if row.get("inspection_status") == "missing":
        missing = dict(row)
        missing.setdefault("inspection_execution", None)
        missing.setdefault("inspection_worker_killed", False)
        return missing
    return None


def _inspect_video_row(
    root: _LeRobotRoot,
    row: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    *,
    retry_count: int = 0,
    retry_backoff_seconds: float = _DEFAULT_MEDIA_INSPECTION_RETRY_BACKOFF_SECONDS,
) -> dict[str, Any]:
    path = _join(root, str(row["path"]))
    cached_or_terminal = _cached_or_terminal_video_row(row, cache, execution_mode="thread")
    if cached_or_terminal is not None:
        return cached_or_terminal

    started = time.perf_counter()
    diagnostics = [dict(item) for item in row.get("diagnostics") or []]
    status = "completed"
    error: str | None = None
    bytes_read = 0
    codec = str(row.get("codec") or _codec_from_path(path))
    codec_tag = row.get("codec_tag")
    codec_profile = row.get("codec_profile")
    width = row.get("width")
    height = row.get("height")
    resolution = row.get("resolution")
    fps = row.get("fps")
    frame_count = row.get("frame_count")
    gop_size = row.get("gop_size")
    duration_seconds = row.get("duration_seconds")
    keyframe_map: list[dict[str, Any]] = list(row.get("keyframe_map") or [])
    attempts = 0
    attempt_errors: list[dict[str, Any]] = []
    error_class: str | None = None
    metadata = None
    max_attempts = max(1, 1 + max(0, int(retry_count)))
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        attempt_started = time.perf_counter()
        try:
            metadata = inspect_mp4_video(
                str(row.get("uri") or _file_uri(path)),
                storage_options=dict(root.storage_options or {}),
                auth_ref=root.auth_ref,
            )
        except (OSError, Mp4MetadataError) as exc:
            error = str(exc)
            error_class = exc.__class__.__name__
            attempt_errors.append(
                {
                    "attempt": attempt,
                    "error_class": error_class,
                    "error": str(exc),
                    "duration_ms": (time.perf_counter() - attempt_started) * 1000.0,
                }
            )
            if attempt < max_attempts:
                if retry_backoff_seconds > 0:
                    time.sleep(retry_backoff_seconds)
                continue
            status = "failed"
            diagnostics.append(
                _video_diagnostic(
                    "corrupt-video",
                    "error",
                    camera_key=str(row["camera_key"]),
                    episode_index=int(row["episode_index"]),
                    path=str(row["path"]),
                    message=(
                        f"could not read MP4 sample metadata for {row['path']} "
                        f"after {attempt} attempt(s): {exc}"
                    ),
                    error_class=error_class,
                    attempts=attempt,
                )
            )
        break

    if metadata is not None:
        error = None
        error_class = None
        bytes_read = int(metadata.bytes_read)
        codec = metadata.codec
        codec_tag = metadata.codec_tag
        codec_profile = metadata.codec_profile
        width = metadata.width
        height = metadata.height
        resolution = metadata.resolution
        fps = metadata.fps if metadata.fps is not None else fps
        frame_count = metadata.frame_count
        gop_size = metadata.gop_size
        duration_seconds = metadata.duration_seconds
        keyframe_map = [dict(entry) for entry in metadata.keyframe_map]
        if codec not in {"h264", "h265", "av1", "mp4v", "mjpeg"}:
            diagnostics.append(
                _video_diagnostic(
                    "unsupported-codec",
                    "warning",
                    camera_key=str(row["camera_key"]),
                    episode_index=int(row["episode_index"]),
                    path=str(row["path"]),
                    message=f"MP4 codec {codec!r} is not a known accelerated training codec",
                    codec=codec,
                )
            )

    expected_frame_count = _optional_int(row.get("expected_frame_count"))
    if expected_frame_count is not None and frame_count is not None:
        if int(expected_frame_count) != int(frame_count):
            diagnostics.append(
                _video_diagnostic(
                    "frame-count-mismatch",
                    "error",
                    camera_key=str(row["camera_key"]),
                    episode_index=int(row["episode_index"]),
                    path=str(row["path"]),
                    message=(
                        f"LeRobot metadata declares {expected_frame_count} frames but "
                        f"MP4 sample table has {frame_count}"
                    ),
                    expected_frame_count=int(expected_frame_count),
                    actual_frame_count=int(frame_count),
                )
            )

    duration_ms = (time.perf_counter() - started) * 1000.0
    inspected = {
        **row,
        "codec": codec,
        "codec_tag": codec_tag,
        "codec_profile": codec_profile,
        "width": width,
        "height": height,
        "resolution": resolution,
        "fps": fps,
        "frame_count": frame_count,
        "gop_size": gop_size,
        "duration_seconds": duration_seconds,
        "keyframe_map": keyframe_map,
        "diagnostics": diagnostics,
        "inspection_status": status,
        "inspection_bytes_read": bytes_read,
        "inspection_duration_ms": duration_ms,
        "inspection_attempts": attempts,
        "inspection_retries": max(0, attempts - 1),
        "inspection_timeouts": 0,
        "inspection_error_class": error_class,
        "inspection_attempt_errors": attempt_errors,
        "inspection_reused": False,
        "inspection_reused_from": None,
        "inspection_error": error,
        "inspection_execution": "thread",
        "inspection_worker_killed": False,
    }
    inspected["inspection_id"] = _video_inspection_id(inspected)
    return inspected


def _timeout_video_row(
    row: dict[str, Any],
    *,
    timeout_seconds: float,
    duration_ms: float,
    execution_mode: str = "thread",
    worker_killed: bool = False,
) -> dict[str, Any]:
    diagnostics = [dict(item) for item in row.get("diagnostics") or []]
    message = f"timed out reading MP4 sample metadata for {row['path']} after {timeout_seconds:g}s"
    diagnostics.append(
        _video_diagnostic(
            "timeout-video",
            "error",
            camera_key=str(row["camera_key"]),
            episode_index=int(row["episode_index"]),
            path=str(row["path"]),
            message=message,
            timeout_seconds=float(timeout_seconds),
            attempts=1,
            execution_mode=execution_mode,
            worker_killed=bool(worker_killed),
        )
    )
    inspected = {
        **row,
        "diagnostics": diagnostics,
        "inspection_status": "timeout",
        "inspection_bytes_read": 0,
        "inspection_duration_ms": duration_ms,
        "inspection_attempts": 1,
        "inspection_retries": 0,
        "inspection_timeouts": 1,
        "inspection_error_class": "TimeoutError",
        "inspection_attempt_errors": [
            {
                "attempt": 1,
                "error_class": "TimeoutError",
                "error": message,
                "duration_ms": duration_ms,
            }
        ],
        "inspection_reused": False,
        "inspection_reused_from": None,
        "inspection_error": message,
        "inspection_execution": execution_mode,
        "inspection_worker_killed": bool(worker_killed),
    }
    inspected["inspection_id"] = _video_inspection_id(inspected)
    return inspected


def _media_cache_by_fingerprint(
    rows: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("inspection_status") != "completed":
            continue
        fingerprint = str(row.get("inspection_fingerprint") or "")
        if fingerprint:
            cache[fingerprint] = dict(row)
    return cache


def _normalize_timeout_seconds(value: float | int | None) -> float | None:
    if value is None:
        return None
    seconds = float(value)
    if seconds <= 0:
        return None
    return seconds


def _normalize_media_inspection_execution_mode(value: str | None) -> str:
    mode = (value or _DEFAULT_MEDIA_INSPECTION_EXECUTION_MODE).strip().lower()
    if mode not in _MEDIA_INSPECTION_EXECUTION_MODES:
        expected = ", ".join(_MEDIA_INSPECTION_EXECUTION_MODES)
        raise ValueError(f"unknown LeRobot media inspection execution mode {value!r}; expected {expected}")
    return mode


def _media_inspection_report(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    max_workers: int,
    inspect_videos: bool = True,
    timeout_seconds: float | None = None,
    retry_count: int = 0,
    retry_backoff_seconds: float = _DEFAULT_MEDIA_INSPECTION_RETRY_BACKOFF_SECONDS,
    execution_mode: str = _DEFAULT_MEDIA_INSPECTION_EXECUTION_MODE,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    codec_counts: dict[str, int] = {}
    diagnostic_counts: dict[str, int] = {}
    reused_count = 0
    total_bytes_read = 0
    total_duration_ms = 0.0
    total_attempts = 0
    total_retries = 0
    total_timeouts = 0
    killed_worker_count = 0
    videos: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("inspection_status") or "pending")
        status_counts[status] = status_counts.get(status, 0) + 1
        codec = str(row.get("codec") or "unknown")
        codec_counts[codec] = codec_counts.get(codec, 0) + 1
        if row.get("inspection_reused"):
            reused_count += 1
        total_bytes_read += int(row.get("inspection_bytes_read") or 0)
        total_duration_ms += float(row.get("inspection_duration_ms") or 0.0)
        total_attempts += int(row.get("inspection_attempts") or 0)
        total_retries += int(row.get("inspection_retries") or 0)
        total_timeouts += int(row.get("inspection_timeouts") or 0)
        if row.get("inspection_worker_killed"):
            killed_worker_count += 1
        for diagnostic in row.get("diagnostics") or []:
            code = str(diagnostic.get("code") or "unknown")
            diagnostic_counts[code] = diagnostic_counts.get(code, 0) + 1
        videos.append(_media_inspection_summary(row))
    return {
        "version": 1,
        "mode": "bounded-metadata-inspection" if inspect_videos else "deferred",
        "execution_mode": execution_mode if inspect_videos else "none",
        "max_workers": int(max_workers),
        "timeout_seconds": timeout_seconds,
        "retry_count": int(max(0, retry_count)),
        "retry_backoff_seconds": float(max(0.0, retry_backoff_seconds)),
        "video_count": len(videos),
        "reused_count": reused_count,
        "total_bytes_read": total_bytes_read,
        "total_duration_ms": total_duration_ms,
        "total_attempts": total_attempts,
        "total_retries": total_retries,
        "total_timeouts": total_timeouts,
        "killed_worker_count": killed_worker_count,
        "status_counts": dict(sorted(status_counts.items())),
        "codec_counts": dict(sorted(codec_counts.items())),
        "diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        "videos": videos,
    }


def _media_inspection_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "inspection_id",
        "inspection_status",
        "inspection_fingerprint",
        "inspection_bytes_read",
        "inspection_duration_ms",
        "inspection_attempts",
        "inspection_retries",
        "inspection_timeouts",
        "inspection_error_class",
        "inspection_attempt_errors",
        "inspection_reused",
        "inspection_reused_from",
        "inspection_error",
        "inspection_execution",
        "inspection_worker_killed",
        "object_metadata",
        "camera_key",
        "episode_index",
        "path",
        "uri",
        "size",
        "codec",
        "codec_tag",
        "codec_profile",
        "width",
        "height",
        "resolution",
        "fps",
        "frame_count",
        "expected_frame_count",
        "gop_size",
        "duration_seconds",
        "keyframe_map",
        "diagnostics",
    )
    return {key: row.get(key) for key in keys if key in row}


def _bounded_media_workers(value: int | None) -> int:
    if value is None:
        return _DEFAULT_MEDIA_INSPECTION_WORKERS
    return max(1, min(int(value), 32))


def _video_inspection_id(row: dict[str, Any]) -> str:
    return "media-inspection-" + hashlib.sha256(
        json.dumps(
            {
                "fingerprint": row.get("inspection_fingerprint"),
                "path": row.get("path"),
                "status": row.get("inspection_status"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]


def _video_inspection_fingerprint(
    *,
    uri: str,
    size: int,
    metadata: dict[str, Any],
    expected_frame_count: int | None,
) -> str:
    payload = {
        "uri": uri,
        "size": int(size),
        "metadata": dict(metadata),
        "expected_frame_count": expected_frame_count,
    }
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _video_mtime_ns(path: Path) -> int | None:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return None


def _video_diagnostic(
    code: str,
    severity: str,
    *,
    camera_key: str,
    episode_index: int,
    path: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "camera_key": camera_key,
        "episode_index": int(episode_index),
        "path": path,
        "message": message,
        **details,
    }


def _video_diagnostics(video_files: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for video in video_files:
        diagnostics.extend(dict(item) for item in video.get("diagnostics") or [])
    return diagnostics


def _infer_camera_key(path: str | Path, camera_keys: tuple[str, ...]) -> str | None:
    path_text = _path_text(path)
    haystack = path_text.lower()
    for key in camera_keys:
        if key.lower() in haystack:
            return key
    match = re.search(r"observation\.images\.([^/]+)", path_text)
    if match:
        return match.group(1)
    return camera_keys[0] if len(camera_keys) == 1 else None


def _episode_index_from_name(path: str | Path) -> int | None:
    match = re.search(r"episode[_-](\d+)", Path(_path_text(path)).name)
    return int(match.group(1)) if match else None


def _episode_index_from_path(row: dict[str, Any]) -> int | None:
    source = str(row.get("_source_parquet") or "")
    match = re.search(r"episode[_-](\d+)", source)
    return int(match.group(1)) if match else None


def _codec_from_path(path: str | Path) -> str:
    suffix = Path(_path_text(path)).suffix.lower().lstrip(".")
    return suffix or "unknown"


def _data_files(root: _LeRobotRoot) -> list[str | Path]:
    return _glob(root, "data/**/*.parquet")


def _content_files(root: _LeRobotRoot) -> list[str | Path]:
    patterns = (
        "meta/**/*.json",
        "meta/**/*.jsonl",
        "meta/**/*.parquet",
        "data/**/*.parquet",
        "videos/**/*",
        "images/**/*",
    )
    files: dict[str, str | Path] = {}
    for pattern in patterns:
        for path in _glob(root, pattern):
            if _is_file(root, path):
                files[_relative_path(root, path)] = path
    return [files[key] for key in sorted(files)]


def _combined_checksum(
    root: _LeRobotRoot,
    files: list[str | Path],
    *,
    metadata_by_rel: dict[str, dict[str, Any]] | None = None,
) -> str:
    digest = hashlib.sha256()
    for path in files:
        rel = _relative_path(root, path)
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(
            _content_file_fingerprint(
                root,
                path,
                rel=rel,
                metadata_by_rel=metadata_by_rel,
            ).encode()
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _checksum_with_validation(checksum: str, validation_material: str) -> str:
    digest = hashlib.sha256()
    digest.update(checksum.encode())
    digest.update(b"\0")
    digest.update(validation_material.encode())
    return digest.hexdigest()


def _object_store_validation_objects(
    root: _LeRobotRoot,
    files: list[str | Path],
    *,
    metadata_by_rel: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[LeRobotObjectStoreValidationObject, ...]:
    manifest_entries: dict[str, Any] = {}
    if root.manifest_cache is not None:
        manifest = root.manifest_cache.manifest(
            root.uri,
            storage_options=root.storage_options,
            auth_ref=root.auth_ref,
        )
        manifest_entries = {entry.relative_path: entry for entry in manifest.entries}
    objects: list[LeRobotObjectStoreValidationObject] = []
    for path in files:
        relative = _relative_path(root, path)
        manifest_entry = manifest_entries.get(relative)
        if manifest_entry is not None:
            objects.append(
                LeRobotObjectStoreValidationObject(
                    uri=manifest_entry.uri,
                    relative_path=manifest_entry.relative_path,
                    info=dict(manifest_entry.info),
                    metadata_fingerprint=manifest_entry.fingerprint,
                )
            )
            continue
        info = dict((metadata_by_rel or {}).get(relative) or _safe_info(root, path))
        objects.append(
            LeRobotObjectStoreValidationObject(
                uri=_file_uri(path),
                relative_path=relative,
                info=info,
                metadata_fingerprint=object_metadata_fingerprint(info),
            )
        )
    return tuple(objects)


def _content_file_fingerprint(
    root: _LeRobotRoot,
    path: str | Path,
    *,
    rel: str | None = None,
    metadata_by_rel: dict[str, dict[str, Any]] | None = None,
) -> str:
    relative = rel or _relative_path(root, path)
    metadata = _safe_info(root, path)
    if metadata_by_rel is not None:
        metadata_by_rel[relative] = dict(metadata)
    if root.is_remote:
        fingerprint_metadata = _fingerprint_metadata(metadata)
        if fingerprint_metadata:
            return json.dumps(
                {
                    "kind": "object-metadata-v1",
                    **fingerprint_metadata,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
    if relative.startswith(("videos/", "images/")):
        return json.dumps(
            {
                "kind": "media-stat-v1",
                "size": _metadata_size(metadata),
                "mtime_ns": _metadata_mtime_ns(metadata),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    if root.storage_options or root.auth_ref:
        checksum = file_checksum(
            _file_uri(path),
            storage_options=dict(root.storage_options or {}),
            auth_ref=root.auth_ref,
        )
    else:
        checksum = file_checksum(_file_uri(path))
    return "sha256:" + checksum


def _read_jsonl(root: _LeRobotRoot, path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(_read_text(root, path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"invalid JSONL in {path}:{line_number}: {exc}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _frames_by_episode(frames: tuple[dict[str, Any], ...]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for frame in frames:
        grouped.setdefault(int(frame["_episode_index"]), []).append(frame)
    return dict(sorted(grouped.items()))


def _recognized_frame_keys(camera_keys: tuple[str, ...]) -> set[str]:
    keys = {
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "timestamp_ns",
        "task_index",
        "task",
        "language_instruction",
        "caption",
        "observation.state",
        "action",
        "observation_id",
        "episode_id",
        "scenario_id",
        "run_id",
        "topic",
        "modality",
    }
    keys.update(f"observation.images.{key}" for key in camera_keys)
    keys.update(f"observation.image.{key}" for key in camera_keys)
    return keys


def _float_vector(value: Any) -> list[float]:
    if value is None:
        return []
    return [float(item) for item in value]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dedupe_dicts(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    by_key: dict[Any, dict[str, Any]] = {}
    for row in rows:
        by_key.setdefault(row.get(key), row)
    return list(by_key.values())


def _join(root: _LeRobotRoot, rel: str | Path) -> str | Path:
    value = str(rel)
    if root.is_remote:
        if is_object_store_uri(value):
            return value
        return join_uri(root.uri, value)
    if root.path is None:
        return value
    path = Path(value)
    return path if path.is_absolute() else root.path / path


def _glob(root: _LeRobotRoot, pattern: str) -> list[str | Path]:
    try:
        if root.is_remote:
            if root.manifest_cache is not None:
                return root.manifest_cache.list_paths(
                    root.uri,
                    pattern=pattern,
                    storage_options=root.storage_options,
                    auth_ref=root.auth_ref,
                )
            return list_uri(
                root.uri,
                pattern=pattern,
                storage_options=root.storage_options,
                auth_ref=root.auth_ref,
            )
        if root.path is None:
            return []
        return sorted(path.resolve() for path in root.path.glob(pattern) if path.is_file())
    except StorageConfigError as exc:
        raise AdapterError(str(exc)) from exc


def _is_file(root: _LeRobotRoot, path: str | Path) -> bool:
    target = _join(root, path)
    if is_object_store_uri(str(target)):
        try:
            info = uri_info(
                str(target),
                storage_options=root.storage_options,
                auth_ref=root.auth_ref,
            )
        except StorageConfigError:
            return False
        return str(info.get("type") or "file").lower() != "directory"
    return Path(target).is_file()


def _read_text(root: _LeRobotRoot, path: str | Path) -> str:
    target = _join(root, path)
    try:
        return read_text_uri(
            target,
            storage_options=root.storage_options,
            auth_ref=root.auth_ref,
        )
    except StorageConfigError as exc:
        raise AdapterError(str(exc)) from exc


@contextmanager
def _open_binary(root: _LeRobotRoot, path: str | Path):
    target = _join(root, path)
    try:
        with open_binary_uri(
            target,
            storage_options=root.storage_options,
            auth_ref=root.auth_ref,
        ) as stream:
            yield stream
    except StorageConfigError as exc:
        raise AdapterError(str(exc)) from exc


def _read_parquet_rows(root: _LeRobotRoot, path: str | Path) -> list[dict[str, Any]]:
    with _open_binary(root, path) as stream:
        return pq.read_table(stream).to_pylist()


def _safe_info(root: _LeRobotRoot, path: str | Path) -> dict[str, Any]:
    target = _join(root, path)
    if root.is_remote and root.manifest_cache is not None:
        cached = root.manifest_cache.cached_info(
            root.uri,
            str(target),
            storage_options=root.storage_options,
            auth_ref=root.auth_ref,
        )
        if cached is not None:
            return cached
    try:
        return uri_info(
            target,
            storage_options=root.storage_options,
            auth_ref=root.auth_ref,
        )
    except StorageConfigError:
        return {}


def _object_store_manifest_report(root: _LeRobotRoot) -> dict[str, Any] | None:
    if not root.is_remote or root.manifest_cache is None:
        return None
    return root.manifest_cache.last_report(
        root.uri,
        storage_options=root.storage_options,
        auth_ref=root.auth_ref,
    )


def _metadata_size(metadata: Mapping[str, Any]) -> int:
    for key in ("size", "Size", "ContentLength", "content_length"):
        if metadata.get(key) is not None:
            return int(metadata[key])
    return 0


def _metadata_mtime_ns(metadata: Mapping[str, Any]) -> int | None:
    if metadata.get("mtime_ns") is not None:
        return int(metadata["mtime_ns"])
    value = metadata.get("mtime") or metadata.get("last_modified") or metadata.get("LastModified")
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        return int(value.timestamp() * 1_000_000_000)
    try:
        return int(float(value) * 1_000_000_000)
    except (TypeError, ValueError):
        return None


def _fingerprint_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "etag",
        "ETag",
        "version_id",
        "VersionId",
        "generation",
        "Generation",
        "size",
        "Size",
        "last_modified",
        "LastModified",
        "mtime",
        "mtime_ns",
    )
    result: dict[str, Any] = {}
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        normalized = {
            "ETag": "etag",
            "VersionId": "version_id",
            "Generation": "generation",
            "Size": "size",
            "LastModified": "last_modified",
        }.get(key, key)
        result[normalized] = value.isoformat() if hasattr(value, "isoformat") else value
    if result.get("size") is not None:
        result["size"] = int(result["size"])
    return result


def _file_uri(path: str | Path) -> str:
    value = str(path)
    return value if is_object_store_uri(value) else source_uri(value)


def _path_text(path: str | Path) -> str:
    return path.as_posix() if isinstance(path, Path) else str(path)


def _relative_path(root: _LeRobotRoot, path: str | Path) -> str:
    if root.is_remote:
        value = str(path)
        prefix = root.uri.rstrip("/") + "/"
        return value[len(prefix) :] if value.startswith(prefix) else value
    target = Path(path)
    try:
        if root.path is None:
            return target.as_posix()
        return target.resolve().relative_to(root.path.resolve()).as_posix()
    except ValueError:
        return target.resolve().as_posix()


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _source_identity(source: LeRobotSource) -> dict[str, Any]:
    identity = {
        "kind": source.identity_kind,
        "checksum": source.checksum,
        "digest": source.digest,
        "repo_id": source.repo_id,
        "revision": source.revision,
    }
    if source.object_store_validation is not None:
        identity["object_store_validation"] = dict(source.object_store_validation)
    return identity
