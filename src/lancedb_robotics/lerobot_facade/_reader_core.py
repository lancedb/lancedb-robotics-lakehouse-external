"""Storage-format-neutral logic behind ``dataset_reader.LancedbRoboticsDatasetReader``.

Deliberately holds no import of ``lerobot`` itself: everything a
``lerobot.datasets.dataset_reader.BaseDatasetReader`` subclass needs from this
module -- parsing the companion source manifest, computing episode-boundary
index math, lazily (re)opening the live facade, and converting one facade item
into the lerobot item contract (tensorize, decode camera JPEG bytes) -- is
plain ``numpy``/``torch``/this package's own code. ``dataset_reader.py`` itself
hard-imports ``lerobot.datasets.dataset_reader.BaseDatasetReader`` at module
level (that class does not exist in any released ``lerobot`` version yet, only
on upstream ``main`` post PR #4363), so it cannot be imported at all in the
default dev/CI environment. Splitting the actual logic out here means it stays
unit-testable without the dev-only, commit-pinned ``lerobot-main-dev`` extra --
only the thin ABC wiring in ``dataset_reader.py`` needs that gated lane.
"""

from __future__ import annotations

import importlib.util
import io
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lancedb_robotics.lake import Lake

from .mapping import CanonicalVectorMapping
from .reader import LiveLeRobotFacade

STORAGE_FORMAT = "lancedb_robotics"
SOURCE_MANIFEST_FILENAME = "lancedb_robotics_source.json"


class ManifestDriftError(Exception):
    """The live facade disagrees with what this manifest recorded at publish time.

    Mirrors upstream ``LanceDatasetReader.__init__``'s own loud guards (frame
    count vs ``meta.total_frames``; episode ranges tiling ``[0, total_frames)``):
    silent index skew inside a training run is the one failure this storage
    format must never allow (backlog 0491).
    """


class SourceManifest:
    """Parsed ``meta/lancedb_robotics_source.json`` (see ``manifest.py``)."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.lake_uri: str | None = raw.get("lake_uri")
        self.lake_open_kwargs: dict[str, Any] = dict(raw.get("lake_open_kwargs") or {})
        self.alignment_id: str = raw["alignment_id"]
        self.mapping = CanonicalVectorMapping(
            state_streams=tuple(raw["mapping"]["state_streams"]),
            action_streams=tuple(raw["mapping"]["action_streams"]),
            camera_streams=tuple(raw["mapping"]["camera_streams"]),
        )
        self.statuses: Sequence[str] | str | None = raw.get("statuses")
        self.min_confidence: float | None = raw.get("min_confidence")
        self.require_streams: bool | Sequence[str] = raw.get("require_streams", True)
        # None when the resolved episode list exceeded the manifest's inline
        # bound (EPISODE_IDS_INLINE_LIMIT): membership is re-derived from the
        # pinned tables, which is deterministic by construction.
        self.episode_ids: list[str] | None = raw.get("episode_ids")
        # Backlog 0491 fields; absent on manifests written before publish-view
        # existed (those open live and rely on the frame-count guard alone).
        self.view_id: str | None = raw.get("view_id")
        self.table_versions: list[dict[str, Any]] = [
            dict(item) for item in (raw.get("table_versions") or [])
        ]
        self.total_frames: int | None = (
            int(raw["total_frames"]) if raw.get("total_frames") is not None else None
        )
        self.total_episodes: int | None = (
            int(raw["total_episodes"]) if raw.get("total_episodes") is not None else None
        )


def load_source_manifest(root: Path) -> SourceManifest:
    payload = json.loads((root / "meta" / SOURCE_MANIFEST_FILENAME).read_text())
    return SourceManifest(payload)


def episode_frame_bounds(
    dataset_from_index: np.ndarray,
    dataset_to_index: np.ndarray,
    episodes: list[int] | None,
) -> tuple[np.ndarray | None, dict[int, int] | None]:
    """Return ``(rel_to_abs, absolute_to_relative_idx)`` for an episode filter.

    ``episodes`` must already be sorted. Mirrors upstream
    ``LanceDatasetReader.__init__``'s own technique (lines ~172-183 of
    ``lance_backend.py``): frames are served in storage order (ascending
    episode index), so the relative position lerobot hands the reader is
    exactly the position in this concatenation -- both ``(None, None)`` when
    ``episodes`` is ``None`` (no translation needed; ``meta.total_frames``
    answers ``num_frames`` directly).
    """
    if episodes is None:
        return None, None
    if not episodes:
        return np.array([], dtype=np.int64), {}
    rel_to_abs = np.concatenate(
        [np.arange(dataset_from_index[i], dataset_to_index[i]) for i in episodes]
    )
    absolute_to_relative_idx = {int(abs_idx): rel_idx for rel_idx, abs_idx in enumerate(rel_to_abs)}
    return rel_to_abs, absolute_to_relative_idx


def validate_episode_tiling(
    dataset_from_index: np.ndarray,
    dataset_to_index: np.ndarray,
    total_frames: int,
) -> None:
    """Raise unless episode ranges exactly tile ``[0, total_frames)``.

    The second of upstream ``LanceDatasetReader.__init__``'s two guards: an
    episode index that overlaps, gaps, or overshoots the frame count would make
    ``EpisodeAwareSampler`` emit indices pointing at frames nobody intended.
    """
    if len(dataset_from_index) != len(dataset_to_index):
        raise ManifestDriftError(
            "episode index is corrupt: dataset_from_index and dataset_to_index "
            f"disagree in length ({len(dataset_from_index)} vs {len(dataset_to_index)})"
        )
    if len(dataset_from_index) == 0:
        if total_frames != 0:
            raise ManifestDriftError(
                f"episode index is empty but the manifest declares {total_frames} frames"
            )
        return
    cursor = 0
    for index, (start, end) in enumerate(
        zip(dataset_from_index.tolist(), dataset_to_index.tolist(), strict=True)
    ):
        if start != cursor or end < start:
            raise ManifestDriftError(
                f"episode ranges do not tile [0, {total_frames}): episode {index} "
                f"spans [{start}, {end}) but the previous episode ended at {cursor}"
            )
        cursor = end
    if cursor != total_frames:
        raise ManifestDriftError(
            f"episode ranges end at {cursor} but the manifest declares "
            f"{total_frames} frames"
        )


def open_facade(
    manifest: SourceManifest,
    episodes: list[int] | None,
    *,
    expected_frames: int | None = None,
) -> LiveLeRobotFacade:
    """(Re)open the live facade this manifest records, pinned when possible.

    A manifest published by ``views.publish_view`` carries ``table_versions``;
    the facade then reads through a :class:`~.views.PinnedLake` at exactly those
    versions, so what it serves is byte-identical to what was published no
    matter how far the live lake has advanced. Legacy manifests (no pins) open
    live and rely on the frame-count guard below to turn drift into a loud
    error instead of silent index skew — upstream ``LanceDatasetReader``'s own
    ``count_rows() != meta.total_frames`` behavior.
    """
    if manifest.lake_uri is None:
        raise ManifestDriftError(
            "source manifest carries no lake_uri; it was materialized without a "
            "root (re-materialize the view via localize_root / materialize_view)"
        )
    lake: Any = Lake.open(manifest.lake_uri, **manifest.lake_open_kwargs)
    if manifest.table_versions:
        from .views import PinnedLake, _versions_map  # local import: views imports manifest

        lake = PinnedLake(lake, _versions_map(manifest.table_versions))

    def _build(episode_ids: list[str] | None) -> LiveLeRobotFacade:
        return LiveLeRobotFacade(
            lake,
            alignment_id=manifest.alignment_id,
            mapping=manifest.mapping,
            episode_ids=episode_ids,
            statuses=manifest.statuses,
            min_confidence=manifest.min_confidence,
            require_streams=manifest.require_streams,
        )

    episode_ids: list[str] | None = None
    if episodes is not None:
        known_ids = manifest.episode_ids
        if known_ids is None:
            # The manifest's episode list exceeded its inline bound: resolve
            # the (deterministic, pinned) episode order once, then re-open
            # filtered. Costs one extra unfiltered open, only on huge views.
            known_ids = [episode.episode_id for episode in _build(None)._episodes]
        episode_ids = [known_ids[i] for i in episodes]
    facade = _build(episode_ids)
    if expected_frames is None and episodes is None:
        expected_frames = manifest.total_frames
    if expected_frames is not None and len(facade) != expected_frames:
        raise ManifestDriftError(
            f"live facade resolves {len(facade)} frames but this manifest recorded "
            f"{expected_frames}; the lake has advanced past what the manifest "
            "describes. Re-publish the view (or re-run write_dataset_manifest) "
            "instead of training on silently skewed indices."
        )
    return facade


def facade_item_to_lerobot_item(
    item: dict[str, Any],
    *,
    return_uint8: bool,
    image_transforms: Callable | None,
) -> dict[str, Any]:
    """Convert one ``LiveLeRobotFacade`` item dict to the lerobot item contract.

    Tensorizes numeric vectors (mirrors ``reader.py``'s own ``_tensorize``) and
    decodes ``observation.images.<key>`` JPEG bytes into ``(C, H, W)`` pixel
    tensors -- ``LiveLeRobotFacade`` itself returns raw bytes (see its own
    module docstring: "this repo carries no image-decode dependency"), but
    lerobot-native consumers (e.g. Foxglove's ``arr.numpy()``) expect decoded
    pixel data, not encoded bytes.
    """
    import torch

    result: dict[str, Any] = {
        "episode_index": torch.tensor(item["episode_index"]),
        "frame_index": torch.tensor(item["frame_index"]),
        "index": torch.tensor(item["index"]),
        "timestamp": torch.tensor(item["timestamp"], dtype=torch.float32),
        "task": item["task"],
    }
    for key in ("observation.state", "action"):
        if key in item:
            result[key] = torch.tensor(item[key], dtype=torch.float32)
    for key, value in item.items():
        if key.startswith("observation.images."):
            result[key] = decode_image_bytes(
                value, return_uint8=return_uint8, image_transforms=image_transforms
            )
    return result


def localize_root(
    repo_id: str | None,
    root: str | Path,
    revision: str | None = None,
    *,
    token: str | bool | None = None,
    force_cache_sync: bool = False,
) -> Path:
    """Resolve ``root`` to a local ``lancedb_robotics`` manifest directory.

    Three cases, in order (backlog 0491):

    1. ``root`` is already a materialized manifest directory
       (``meta/info.json`` present): returned as-is.
    2. ``root`` is a lake — a local lake directory or an object-store URI like
       ``s3://acme/lake.lance`` — holding a view published under ``repo_id``
       (``revision`` selects an exact ``view_id``): the view's ``meta/`` files
       are materialized from the lake's ``lerobot_view_files`` rows into a
       local per-view cache directory (atomic, concurrency-safe; see
       ``views.materialize_view``) and that directory is returned. Nothing is
       hand-distributed to the client; auth resolves from the ambient
       environment exactly as any other ``Lake.open`` does.
    3. Anything else raises ``FileNotFoundError``, per the per-format probe
       loop in ``lerobot.datasets.storage.localize_remote_root`` — other
       registered formats get their turn.

    Has no ``lerobot`` dependency itself, but is only ever called through that
    probe loop — ``dataset_reader.py`` re-exports it as the module-level
    attribute upstream's ``storage.py`` resolves ``localize_root`` on for this
    format, exactly like upstream's own ``lance_backend.py`` re-exports its own
    from ``lance_utils.py``.
    """
    root_str = str(root)
    if "://" not in root_str:
        root_path = Path(root)
        if (root_path / "meta" / "info.json").exists():
            return root_path
    # Lazy import: views imports manifest which imports this module.
    from lancedb_robotics.lake import LakeError

    from .views import ViewError, ViewNotFoundError, resolve_view_root

    try:
        return resolve_view_root(
            repo_id, root_str, revision=revision, force_cache_sync=force_cache_sync
        )
    except LakeError as exc:
        raise FileNotFoundError(
            f"no {STORAGE_FORMAT!r} manifest found at {root_str} and it is not a "
            f"lancedb-robotics lake ({exc})"
        ) from exc
    except ViewNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc
    except ViewError as exc:
        raise FileNotFoundError(
            f"cannot resolve a published view for {repo_id!r} at {root_str}: {exc}"
        ) from exc


def decode_image_bytes(
    data: bytes,
    *,
    return_uint8: bool,
    image_transforms: Callable | None = None,
):
    """Decode one JPEG-encoded camera frame into a ``(C, H, W)`` torch tensor."""
    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError(
            "decoding observation.images.* requires optional dependency 'PIL'; "
            "install `lancedb-robotics[media]`"
        )
    import torch
    from PIL import Image

    array = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))  # (H, W, C) uint8
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()  # (C, H, W)
    if not return_uint8:
        tensor = tensor.to(torch.float32) / 255.0
    if image_transforms is not None:
        tensor = image_transforms(tensor)
    return tensor
