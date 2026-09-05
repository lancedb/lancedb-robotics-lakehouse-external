"""Derive a lerobot-loadable ``meta/`` manifest from a live facade.

Upstream ``huggingface/lerobot`` PR #4363 (merged to ``main`` 2026-08-28, not yet in
any PyPI release) made ``LeRobotDataset`` dispatch storage format polymorphically:
before any reader is chosen, ``LeRobotDatasetMetadata`` loads a small,
storage-format-neutral manifest (``meta/info.json``, ``meta/episodes/*.parquet``
whenever ``info.json`` declares any episodes, and ``meta/stats.json`` when present)
from ``root``. This module derives exactly those files for one
:class:`~lancedb_robotics.lerobot_facade.LiveLeRobotFacade` configuration, plus a
companion ``lancedb_robotics_source.json`` recording how
``dataset_reader.LancedbRoboticsDatasetReader`` reopens the live facade on read.

Since backlogs 0490/0491 the derive step also computes **normalization statistics**
(``meta/stats.json``, see :mod:`.stats`) and records **pinned table versions**, and
the public entry point moved from hand-run local directories to
:func:`lancedb_robotics.lerobot_facade.views.publish_view`, which stores the derived
files in the lake itself (``lerobot_view_files``) so clients materialize them through
the same Lance connection instead of a hand-distributed side channel.
:func:`write_dataset_manifest` remains only as a deprecated local-directory writer.

Deliberately dependency-light: only ``pyarrow``/``numpy`` (already hard dependencies
of this repo) are needed -- no ``lerobot``, ``pandas``, or ``torch`` import here.
``meta/tasks.parquet`` is skipped entirely (``total_tasks=0`` in ``info.json``): the
reader returns each frame's ``task`` as a plain string directly from the facade, so
nothing ever reads ``meta.tasks`` for this storage format.

Camera features are declared ``dtype: "image"``, not ``"video"``: the facade
already produces one decoded-per-tick JPEG per camera frame
(``VideoIndex.decode_batch``), not an mp4 file requiring timestamp-windowed
decode -- ``"image"`` is the accurate description and avoids fabricating
``videos/<key>/chunk_index``/``file_index``/``from_timestamp`` episode columns
that are private to the mp4-backed readers and never read by this storage format.
"""

from __future__ import annotations

import io
import json
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lancedb_robotics.lake import Lake

from ._reader_core import SOURCE_MANIFEST_FILENAME, STORAGE_FORMAT
from .mapping import CanonicalVectorMapping
from .reader import LiveLeRobotFacade
from .stats import compute_view_stats, serialize_stats

CODEBASE_VERSION = "v3.0"

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
SOURCE_MANIFEST_PATH = f"meta/{SOURCE_MANIFEST_FILENAME}"

# Bounded episode-index rows AND bytes per parquet file so a very large
# multi-episode view never becomes one oversized file/cell (the 0144
# scale-review failure shape; commits are bounded in bytes, not just rows —
# BUG-02). Files sort lexicographically (file-00000, file-00001, ...), which is
# exactly the order a parquet-directory read concatenates them in; 5-digit
# padding keeps that true to 100k files.
EPISODES_ROWS_PER_FILE = 100_000
EPISODES_TARGET_FILE_BYTES = 8 * 1024 * 1024

# Resolved episode ids are inlined into the companion source manifest only up
# to this bound (0146 discipline: inline small id lists, never unbounded ones).
# Beyond it the manifest stores null and readers re-derive episode membership
# from the pinned tables, which is deterministic by construction.
EPISODE_IDS_INLINE_LIMIT = 100_000

_DEFAULT_FEATURES: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
}


class ManifestDeriveError(ValueError):
    """Raised when a manifest cannot be derived for a facade configuration.

    Subclasses ``ValueError`` because that is what the original
    ``write_dataset_manifest`` raised for an unusable configuration — callers
    (and tests) written against that contract keep working.
    """


def _safe_relative_path(path: str) -> Path:
    """Validate a manifest-relative path; never trust one read from a table."""
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ManifestDeriveError(f"unsafe manifest file path {path!r}")
    return candidate


@dataclass(frozen=True)
class DerivedManifest:
    """One view's derived metadata files, ready to publish or write locally.

    ``files`` is a single-use stream: the episode-index parquet chunks are
    produced lazily, so a consumer that writes each file as it arrives holds at
    most one bounded file in memory — never the whole derived ``meta/`` tree
    (SKILLS.md §0 invariant 1, applied to the publish side too).
    """

    files: Iterator[tuple[str, bytes]] = field(repr=False)  # (relative path, content)
    total_frames: int = 0
    total_episodes: int = 0
    features: dict[str, dict[str, Any]] = field(repr=False, default_factory=dict)
    stats_sampling: dict[str, Any] | None = None


def derive_manifest(
    facade: LiveLeRobotFacade,
    *,
    repo_id: str,
    fps: int,
    statuses: Sequence[str] | str | None = None,
    min_confidence: float | None = None,
    require_streams: bool | Sequence[str] = True,
    robot_type: str | None = None,
    lake_uri: str | None = None,
    lake_open_kwargs: Mapping[str, Any] | None = None,
    view_id: str | None = None,
    table_versions: Sequence[Mapping[str, Any]] | None = None,
    include_stats: bool = True,
    stats_options: Mapping[str, Any] | None = None,
) -> DerivedManifest:
    """Derive every ``meta/`` file for ``facade``'s exact resolved tick set.

    ``fps`` is a nominal/target frame rate for lerobot-native tooling that paces
    playback by it (e.g. Foxglove's uniform-grid fallback for readers without
    ``hf_dataset``) -- aligned ticks are event-synchronized, not necessarily
    fixed-rate, so this is not derived from real inter-tick deltas; supply the
    alignment's intended tick rate.

    ``lake_uri`` is what the companion source manifest records for reopening the
    facade; :func:`views.publish_view` passes ``None`` (the client's own root is
    injected at materialization time), the deprecated local writer passes the
    publishing lake's URI.

    ``include_stats`` computes ``meta/stats.json`` in the same pass (backlog
    0490): a bounded-memory full scan for vector/scalar features plus a bounded
    deterministic sample for camera features (see :mod:`.stats`).
    """
    if len(facade) == 0:
        raise ManifestDeriveError(
            "resolved zero frames for this facade configuration; nothing to derive"
        )
    mapping = facade.mapping

    episode_boundaries: list[tuple[int, int]] = []
    cursor = 0
    for episode in facade._episodes:
        start = cursor
        cursor += len(episode.tick_indices)
        episode_boundaries.append((start, cursor))

    sample = facade[0]
    features = dict(_DEFAULT_FEATURES)
    if mapping.state_streams:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [len(sample["observation.state"])],
            "names": None,
        }
    if mapping.action_streams:
        features["action"] = {
            "dtype": "float32",
            "shape": [len(sample["action"])],
            "names": None,
        }
    camera_keys = sorted(key for key in sample if key.startswith("observation.images."))
    for camera_key in camera_keys:
        features[camera_key] = {"dtype": "image", "shape": None, "names": None}

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "features": features,
        "total_episodes": len(facade._episodes),
        "total_frames": len(facade),
        "total_tasks": 0,
        "storage_format": STORAGE_FORMAT,
        "robot_type": robot_type,
    }

    stats_bytes: bytes | None = None
    stats_sampling: dict[str, Any] | None = None
    if include_stats:
        stats, stats_sampling = compute_view_stats(
            facade, camera_keys=camera_keys, **dict(stats_options or {})
        )
        missing = [key for key in features if key not in stats]
        if missing:
            raise ManifestDeriveError(
                f"statistics missing for declared features {missing!r}; refusing to "
                "publish a manifest whose stats.json disagrees with info.json"
            )
        stats_bytes = _json_bytes(serialize_stats(stats))

    # Resolved episode ids are bounded-inline only (0146 discipline); a larger
    # view stores null and readers re-derive membership from the pinned tables.
    inline_episode_ids: list[str] | None = None
    if len(facade._episodes) <= EPISODE_IDS_INLINE_LIMIT:
        inline_episode_ids = [episode.episode_id for episode in facade._episodes]
    source = {
        "lake_uri": lake_uri,
        "lake_open_kwargs": dict(lake_open_kwargs or {}),
        "alignment_id": facade._dataset.alignment_id,
        "mapping": {
            "state_streams": list(mapping.state_streams),
            "action_streams": list(mapping.action_streams),
            "camera_streams": list(mapping.camera_streams),
        },
        "statuses": list(statuses)
        if isinstance(statuses, Sequence) and not isinstance(statuses, str)
        else statuses,
        "min_confidence": min_confidence,
        "require_streams": require_streams,
        "episode_ids": inline_episode_ids,
        "repo_id": repo_id,
        "view_id": view_id,
        "table_versions": [dict(item) for item in (table_versions or [])],
        "total_frames": len(facade),
        "total_episodes": len(facade._episodes),
        "stats_sampling": stats_sampling,
    }

    def _iter_files() -> Iterator[tuple[str, bytes]]:
        yield (INFO_PATH, _json_bytes(info))
        yield from _episode_index_files(facade, episode_boundaries)
        if stats_bytes is not None:
            yield (STATS_PATH, stats_bytes)
        yield (SOURCE_MANIFEST_PATH, _json_bytes(source))

    return DerivedManifest(
        files=_iter_files(),
        total_frames=len(facade),
        total_episodes=len(facade._episodes),
        features=features,
        stats_sampling=stats_sampling,
    )


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _episode_index_files(
    facade: LiveLeRobotFacade,
    episode_boundaries: list[tuple[int, int]],
) -> Iterator[tuple[str, bytes]]:
    """Stream bounded parquet files for the episode index.

    Each file is capped both by rows (``EPISODES_ROWS_PER_FILE``) and by
    estimated serialized bytes (``EPISODES_TARGET_FILE_BYTES``, dominated by the
    free-form task strings), so no single Lance cell or commit is ever
    oversized regardless of episode count or task-string length.
    """
    episodes = facade._episodes

    def _flush(rows, bounds, file_index) -> tuple[str, bytes]:
        table = pa.table(
            {
                "episode_index": pa.array(
                    [episode.episode_index for episode in rows], type=pa.int64()
                ),
                "dataset_from_index": pa.array([b[0] for b in bounds], type=pa.int64()),
                "dataset_to_index": pa.array([b[1] for b in bounds], type=pa.int64()),
                "length": pa.array([b[1] - b[0] for b in bounds], type=pa.int64()),
                "tasks": pa.array(
                    [[episode.task] for episode in rows], type=pa.list_(pa.string())
                ),
            }
        )
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        return (
            f"meta/episodes/chunk-000/file-{file_index:05d}.parquet",
            buffer.getvalue(),
        )

    rows: list[Any] = []
    bounds: list[tuple[int, int]] = []
    estimated_bytes = 0
    file_index = 0
    for episode, bound in zip(episodes, episode_boundaries, strict=True):
        rows.append(episode)
        bounds.append(bound)
        estimated_bytes += 64 + len(episode.task or "")
        if len(rows) >= EPISODES_ROWS_PER_FILE or estimated_bytes >= EPISODES_TARGET_FILE_BYTES:
            yield _flush(rows, bounds, file_index)
            rows, bounds, estimated_bytes = [], [], 0
            file_index += 1
    if rows:
        yield _flush(rows, bounds, file_index)


def write_manifest_files(root: Path | str, files: Iterable[tuple[str, bytes]]) -> Path:
    """Write derived manifest files under ``root`` (paths validated, dirs created).

    Consumes ``files`` one at a time — safe for the streaming form
    :func:`derive_manifest` returns.
    """
    root = Path(root)
    for path, content in files:
        target = root / _safe_relative_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return root


def write_dataset_manifest(
    lake: Lake,
    root: Path | str,
    *,
    repo_id: str,
    fps: int,
    mapping: CanonicalVectorMapping,
    alignment: str | None = None,
    alignment_id: str | None = None,
    name: str | None = None,
    episode_ids: Sequence[str] | None = None,
    statuses: Sequence[str] | str | None = None,
    min_confidence: float | None = None,
    require_streams: bool | Sequence[str] = True,
    robot_type: str | None = None,
    lake_open_kwargs: Mapping[str, Any] | None = None,
    include_stats: bool = True,
    stats_options: Mapping[str, Any] | None = None,
) -> Path:
    """Deprecated: write a local, *unpinned* manifest directory for one facade.

    Superseded by :func:`lancedb_robotics.lerobot_facade.views.publish_view`
    (backlog 0491), which pins table versions and stores the derived files in
    the lake so clients need no hand-distributed directory. This writer reads
    the lake live and records no version pins: the resulting manifest is only
    valid until the next ingest/re-alignment/quality change, and the reader's
    frame-count guard will refuse it loudly once the lake has advanced.
    """
    warnings.warn(
        "write_dataset_manifest is deprecated; use "
        "lancedb_robotics.lerobot_facade.views.publish_view (backlog 0491) — "
        "published views are version-pinned and need no hand-distributed directory",
        DeprecationWarning,
        stacklevel=2,
    )
    facade = LiveLeRobotFacade(
        lake,
        alignment=alignment,
        alignment_id=alignment_id,
        name=name,
        mapping=mapping,
        episode_ids=episode_ids,
        statuses=statuses,
        min_confidence=min_confidence,
        require_streams=require_streams,
    )
    derived = derive_manifest(
        facade,
        repo_id=repo_id,
        fps=fps,
        statuses=statuses,
        min_confidence=min_confidence,
        require_streams=require_streams,
        robot_type=robot_type,
        lake_uri=lake.uri,
        lake_open_kwargs=lake_open_kwargs,
        include_stats=include_stats,
        stats_options=stats_options,
    )
    return write_manifest_files(root, derived.files)
