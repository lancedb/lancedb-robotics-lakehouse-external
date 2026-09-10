"""``BaseDatasetReader`` implementation serving ``LiveLeRobotFacade`` data.

Registers the ``"lancedb_robotics"`` storage format with upstream
``lerobot.datasets.storage`` (``huggingface/lerobot`` PR #4363, merged to
``main`` 2026-08-28 -- not yet in any PyPI release) so a
``meta/info.json`` written by :func:`lancedb_robotics.lerobot_facade.manifest.
write_dataset_manifest` can be opened as a real
``lerobot.datasets.lerobot_dataset.LeRobotDataset``, making first-party
lerobot tooling (``lerobot-dataset-viz --display-mode foxglove``,
``EpisodeAwareSampler``, ...) work directly against a live lakehouse
alignment, with no export step.

This module is **not** imported by ``lancedb_robotics.lerobot_facade``'s
package ``__init__`` -- it hard-depends on the ``lerobot`` storage-format
registry, which does not exist in any released ``lerobot`` version yet.
Import it explicitly to opt in::

    import lancedb_robotics.lerobot_facade.dataset_reader  # registers "lancedb_robotics"

All the actual logic (manifest parsing, episode-boundary index math, lazily
(re)opening the live facade, item tensorization/image decode) lives in
``_reader_core.py``, which has no ``lerobot`` dependency and is unit-tested
directly without the dev-only ``lerobot-main-dev`` extra; this module is only
the thin wiring that satisfies the real ``BaseDatasetReader`` ABC, verified
against real upstream code by the ``lerobot_main_dev``-marked integration
test.

``delta_timestamps`` (backlog 0509) reuses upstream's own validation and
seconds-to-frames conversion (``check_delta_timestamps``/``get_delta_indices``
from ``lerobot.datasets.feature_utils``, exactly as ``LanceDatasetReader``
does), then hands integer frame deltas to ``_reader_core.plan_windows``/
``hydrate_windowed_items`` — clamping, ``<key>_is_pad`` masks, and stacked
window shapes all mirror upstream ``LanceDatasetReader._plan_batch``. The one
deliberate deviation: a delta key the view's mapping cannot serve raises
``ValueError`` at construction instead of upstream's silent mask-only items.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import BaseDatasetReader
from lerobot.datasets.feature_utils import check_delta_timestamps, get_delta_indices
from lerobot.datasets.storage import register_dataset_reader

from . import _reader_core
from ._reader_core import STORAGE_FORMAT, localize_root  # noqa: F401  (re-exported for storage.py)

register_dataset_reader(STORAGE_FORMAT, "lancedb_robotics.lerobot_facade.dataset_reader")


class LancedbRoboticsDatasetReader(BaseDatasetReader):
    """Reader serving datasets whose ``meta/info.json`` declares
    ``"storage_format": "lancedb_robotics"``: a live lakehouse alignment read
    through :class:`LiveLeRobotFacade`, reopened from the companion
    ``meta/lancedb_robotics_source.json`` manifest (see
    :mod:`lancedb_robotics.lerobot_facade.manifest`).
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        return_uint8: bool = False,
        depth_output_unit: str = "mm",
        token: str | bool | None = None,
    ) -> None:
        self.meta = meta
        self.delta_indices: dict[str, list[int]] | None = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)
        self.return_uint8 = return_uint8
        self.set_image_transforms(image_transforms)

        # Non-default formats receive the *raw* storage root here (upstream
        # passes `root=self._storage_root or root` so a backend can read data
        # in place), while the localized `meta/` — including our companion
        # source manifest — lives at `meta.root`. Prefer an explicit local
        # root that actually carries the manifest (the legacy hand-materialized
        # flow); otherwise read it from the localized directory.
        root_path = Path(meta.root)
        if (
            root is not None
            and "://" not in str(root)
            and (Path(root) / "meta" / _reader_core.SOURCE_MANIFEST_FILENAME).exists()
        ):
            root_path = Path(root)
        self._manifest = _reader_core.load_source_manifest(root_path)
        if self.delta_indices is not None:
            _reader_core.validate_delta_keys(list(self.delta_indices), self._manifest.mapping)
        self.episodes: list[int] | None = sorted(episodes) if episodes is not None else None
        self._facade = None

        ep_from = self._episode_numpy("dataset_from_index")
        ep_to = self._episode_numpy("dataset_to_index")
        # Guard (backlog 0491, mirrors upstream LanceDatasetReader.__init__):
        # a manifest whose episode ranges don't tile [0, total_frames) would
        # make EpisodeAwareSampler silently index the wrong frames.
        _reader_core.validate_episode_tiling(ep_from, ep_to, int(meta.total_frames))
        self._ep_from, self._ep_to = ep_from, ep_to
        self._rel_to_abs, self._absolute_to_relative_idx = _reader_core.episode_frame_bounds(
            ep_from, ep_to, self.episodes
        )
        self._num_frames = (
            int(len(self._rel_to_abs)) if self._rel_to_abs is not None else meta.total_frames
        )

    def _episode_numpy(self, column: str):
        import numpy as np

        return (
            self.meta.episodes.data.column(column)
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )

    # ── BaseDatasetReader contract ─────────────────────────────────────

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        return self._absolute_to_relative_idx

    def get_item(self, idx: int) -> dict:
        if self.delta_indices is not None:
            return self._get_windowed_items([idx])[0]
        return _reader_core.facade_item_to_lerobot_item(
            self._ensure_facade()[idx],
            return_uint8=self.return_uint8,
            image_transforms=self._image_transforms,
        )

    def get_items(self, indices: list[int]) -> list[dict]:
        if self.delta_indices is not None:
            return self._get_windowed_items(list(indices))
        facade = self._ensure_facade()
        return [
            _reader_core.facade_item_to_lerobot_item(
                item, return_uint8=self.return_uint8, image_transforms=self._image_transforms
            )
            for item in facade.__getitems__(list(indices))
        ]

    def _get_windowed_items(self, indices: list[int]) -> list[dict]:
        facade = self._ensure_facade()
        assert self.delta_indices is not None
        plans = _reader_core.plan_windows(
            [self._resolve_abs_idx(idx) for idx in indices],
            self.delta_indices,
            self._ep_from,
            self._ep_to,
        )
        return _reader_core.hydrate_windowed_items(
            facade,
            plans,
            abs_to_facade=self._absolute_to_relative_idx,
            return_uint8=self.return_uint8,
            image_transforms=self._image_transforms,
        )

    def _resolve_abs_idx(self, idx: int) -> int:
        # Mirrors upstream LanceDatasetReader._resolve_abs_idx: lerobot hands
        # this reader *relative* indices (positions within the episode filter).
        idx = int(idx)
        if idx < 0:
            idx += self._num_frames
        if not 0 <= idx < self._num_frames:
            raise IndexError(
                f"Index {idx} is out of range for a dataset of {self._num_frames} frames."
            )
        return int(self._rel_to_abs[idx]) if self._rel_to_abs is not None else idx

    # ── Lazy facade (picklable: never carries a live connection) ──────

    def _ensure_facade(self):
        if self._facade is None:
            # expected_frames is the second 0491 guard: the reopened (pinned
            # when published, live for legacy manifests) facade must resolve
            # exactly the frame count this manifest recorded, or fail loudly.
            self._facade = _reader_core.open_facade(
                self._manifest, self.episodes, expected_frames=self._num_frames
            )
        return self._facade

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_facade"] = None
        return state


# The class lerobot.datasets.storage instantiates for storage_format "lancedb_robotics".
DATASET_READER = LancedbRoboticsDatasetReader
