"""Streaming normalization statistics for published LeRobot views (backlog 0490).

LeRobot policies build their normalization buffers from ``dataset.meta.stats``
(per-feature ``mean``/``std``/``min``/``max``, plus ``q01``/``q10``/``q50``/
``q90``/``q99`` for quantile-normalizing policies), loaded from
``meta/stats.json`` by ``lerobot.datasets.io_utils.load_stats`` /
``cast_stats_to_numpy``: a plain-JSON nested dict ``{feature: {stat: [...]}}``.
This module computes those statistics for one
:class:`~lancedb_robotics.lerobot_facade.LiveLeRobotFacade` — i.e. for exactly
the ticks one published view resolves — in a single bounded-memory pass.

Contract mirrored from real upstream sources at the adapter's pinned
verification commit ``3f2c29ef`` (``datasets/compute_stats.py``):

- vector features carry per-dimension arrays of shape ``(D,)``; scalar features
  shape ``(1,)``; image features per-channel ``(C, 1, 1)`` with values divided
  by 255 into ``[0, 1]``; ``count`` is always shape ``(1,)``.
- quantiles are estimated with an adaptive fixed-bin histogram (upstream's
  ``RunningQuantileStats`` technique, ``num_quantile_bins=5000``), re-binned
  when the observed range expands.
- image features are sampled, not exhaustively decoded: a deterministic
  ``linspace`` over the view of ``min(max(100, N**0.75), 10000)`` frames
  (upstream ``estimate_num_samples``/``sample_indices``), each frame
  stride-downsampled when larger than 300px (upstream
  ``auto_downsample_height_width``).

Differences from upstream, both deliberate:

- mean/variance use a float64 Chan/Welford batch merge instead of upstream's
  running mean-of-squares — strictly better numerics, explicitly allowed by
  backlog 0490 ("Welford, or a shifted-data two-pass if numerical accuracy
  demands it").
- no ``lerobot`` import anywhere: like ``manifest.py``, this module must be
  importable with no ``lerobot`` install. The one optional dependency is PIL,
  required only when camera streams are mapped, and its absence is a loud
  typed error (user decision, 2026-09-04): a published view must never carry
  silently incomplete stats.
"""

from __future__ import annotations

import importlib.util
import io
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .reader import LiveLeRobotFacade

DEFAULT_QUANTILES: tuple[float, ...] = (0.01, 0.10, 0.50, 0.90, 0.99)
NUM_QUANTILE_BINS = 5000

DEFAULT_TABULAR_BATCH_SIZE = 1024
DEFAULT_CAMERA_DECODE_BATCH_SIZE = 64

# Upstream estimate_num_samples defaults (compute_stats.py).
CAMERA_MIN_SAMPLES = 100
CAMERA_MAX_SAMPLES = 10_000
CAMERA_SAMPLE_POWER = 0.75

# Upstream auto_downsample_height_width defaults.
CAMERA_DOWNSAMPLE_TARGET = 150
CAMERA_DOWNSAMPLE_THRESHOLD = 300


class ViewStatsError(Exception):
    """Raised when normalization statistics cannot be computed for a view."""


def quantile_key(quantile: float) -> str:
    """Upstream's stat key for one quantile: ``0.01 -> "q01"``."""
    return f"q{int(quantile * 100):02d}"


class StreamingFeatureStats:
    """Bounded-memory per-dimension statistics over batches of vectors.

    Accumulates count/mean/M2 (Chan parallel-variance batch merge, float64),
    running min/max, and an adaptive fixed-bin histogram per dimension for
    quantile estimates. Memory is O(dims × num_bins), independent of how many
    rows are streamed through — SKILLS.md §0 invariant 1.
    """

    def __init__(
        self,
        *,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        num_bins: int = NUM_QUANTILE_BINS,
    ) -> None:
        self._quantiles = tuple(quantiles)
        self._num_bins = int(num_bins)
        self._count = 0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: np.ndarray | None = None  # (dims, num_bins) float64
        self._edges: np.ndarray | None = None  # (dims, num_bins + 1) float64

    @property
    def count(self) -> int:
        return self._count

    def update(self, batch: np.ndarray) -> None:
        """Fold one ``(rows, dims)`` batch into the running statistics."""
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        if batch.ndim != 2:
            raise ViewStatsError(f"expected a (rows, dims) batch, got shape {batch.shape}")
        if batch.shape[0] == 0:
            return
        if not np.isfinite(batch).all():
            raise ViewStatsError(
                "non-finite value (nan/inf) in a feature batch; refusing to publish "
                "statistics a policy would silently train against"
            )
        rows, dims = batch.shape

        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)
        batch_min = batch.min(axis=0)
        batch_max = batch.max(axis=0)

        if self._count == 0:
            self._mean = batch_mean
            self._m2 = batch_m2
            self._min = batch_min
            self._max = batch_max
            self._count = rows
            self._init_histograms(dims)
            self._fold_histogram(batch)
            return

        assert self._mean is not None and self._m2 is not None
        assert self._min is not None and self._max is not None
        if dims != self._mean.size:
            raise ViewStatsError(
                f"feature width changed mid-stream: {self._mean.size} -> {dims}"
            )

        new_min = np.minimum(self._min, batch_min)
        new_max = np.maximum(self._max, batch_max)
        if (new_min < self._min).any() or (new_max > self._max).any():
            self._min = new_min
            self._max = new_max
            self._rebin()
        # Chan's parallel update: merge (count, mean, M2) of the batch into ours.
        total = self._count + rows
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (rows / total)
        self._m2 = self._m2 + batch_m2 + (delta**2) * (self._count * rows / total)
        self._count = total
        self._fold_histogram(batch)

    def statistics(self) -> dict[str, np.ndarray]:
        """Finalized ``{mean, std, min, max, count, qXX...}`` arrays, shape ``(dims,)``."""
        if self._count == 0 or self._mean is None:
            raise ViewStatsError("no rows were accumulated for this feature")
        assert self._m2 is not None and self._min is not None and self._max is not None
        variance = self._m2 / self._count
        stats: dict[str, np.ndarray] = {
            "min": self._min.copy(),
            "max": self._max.copy(),
            "mean": self._mean.copy(),
            "std": np.sqrt(np.maximum(0.0, variance)),
            "count": np.array([self._count], dtype=np.int64),
        }
        for quantile in self._quantiles:
            stats[quantile_key(quantile)] = self._estimate_quantile(quantile)
        return stats

    # -- histogram quantiles (upstream RunningQuantileStats technique) --------

    def _init_histograms(self, dims: int) -> None:
        assert self._min is not None and self._max is not None
        self._histograms = np.zeros((dims, self._num_bins), dtype=np.float64)
        self._edges = np.empty((dims, self._num_bins + 1), dtype=np.float64)
        for dim in range(dims):
            self._edges[dim] = np.linspace(
                self._min[dim] - 1e-10, self._max[dim] + 1e-10, self._num_bins + 1
            )

    def _rebin(self) -> None:
        """Widen bin edges to the new range, redistributing existing counts."""
        assert self._histograms is not None and self._edges is not None
        assert self._min is not None and self._max is not None
        dims = self._histograms.shape[0]
        new_edges = np.empty_like(self._edges)
        new_histograms = np.zeros_like(self._histograms)
        for dim in range(dims):
            new_edges[dim] = np.linspace(
                min(self._min[dim] - 1e-10, self._edges[dim][0]),
                max(self._max[dim] + 1e-10, self._edges[dim][-1]),
                self._num_bins + 1,
            )
            occupied = self._histograms[dim] > 0
            if occupied.any():
                centers = (self._edges[dim][:-1] + self._edges[dim][1:]) / 2.0
                target = np.clip(
                    np.searchsorted(new_edges[dim], centers[occupied], side="right") - 1,
                    0,
                    self._num_bins - 1,
                )
                np.add.at(new_histograms[dim], target, self._histograms[dim][occupied])
        self._edges = new_edges
        self._histograms = new_histograms

    def _fold_histogram(self, batch: np.ndarray) -> None:
        assert self._histograms is not None and self._edges is not None
        for dim in range(batch.shape[1]):
            counts, _ = np.histogram(batch[:, dim], bins=self._edges[dim])
            self._histograms[dim] += counts

    def _estimate_quantile(self, quantile: float) -> np.ndarray:
        assert self._histograms is not None and self._edges is not None
        dims = self._histograms.shape[0]
        out = np.empty(dims, dtype=np.float64)
        for dim in range(dims):
            histogram = self._histograms[dim]
            total = histogram.sum()
            if total <= 0:
                out[dim] = float(self._mean[dim]) if self._mean is not None else 0.0
                continue
            cumulative = np.cumsum(histogram)
            rank = quantile * total
            bin_index = int(np.searchsorted(cumulative, rank, side="left"))
            bin_index = min(bin_index, self._num_bins - 1)
            left = self._edges[dim][bin_index]
            right = self._edges[dim][bin_index + 1]
            in_bin = histogram[bin_index]
            below = cumulative[bin_index] - in_bin
            fraction = 0.5 if in_bin <= 0 else np.clip((rank - below) / in_bin, 0.0, 1.0)
            out[dim] = left + (right - left) * fraction
        return out


def estimate_camera_samples(
    view_len: int,
    *,
    min_samples: int = CAMERA_MIN_SAMPLES,
    max_samples: int = CAMERA_MAX_SAMPLES,
    power: float = CAMERA_SAMPLE_POWER,
) -> int:
    """Upstream ``estimate_num_samples``: ``min(max(min, N**power), max)``, capped at N."""
    if view_len < min_samples:
        min_samples = view_len
    return max(min_samples, min(int(view_len**power), max_samples))


def camera_sample_indices(view_len: int, num_samples: int) -> list[int]:
    """Upstream ``sample_indices``: deterministic rounded linspace over the view."""
    if view_len <= 0 or num_samples <= 0:
        return []
    return np.round(np.linspace(0, view_len - 1, num_samples)).astype(int).tolist()


def downsample_image(
    image: np.ndarray,
    *,
    target_size: int = CAMERA_DOWNSAMPLE_TARGET,
    max_size_threshold: int = CAMERA_DOWNSAMPLE_THRESHOLD,
) -> np.ndarray:
    """Upstream ``auto_downsample_height_width`` for one ``(C, H, W)`` frame."""
    _, height, width = image.shape
    if max(width, height) < max_size_threshold:
        return image
    factor = int(width / target_size) if width > height else int(height / target_size)
    factor = max(factor, 1)
    return image[:, ::factor, ::factor]


def _require_pil() -> Any:
    if importlib.util.find_spec("PIL") is None:
        raise ViewStatsError(
            "computing camera normalization statistics requires optional "
            "dependency 'PIL'; install `lancedb-robotics[media]`. A view with "
            "camera streams is never published with incomplete stats."
        )
    from PIL import Image

    return Image


def _decode_jpeg_chw(image_module: Any, data: bytes) -> np.ndarray:
    array = np.asarray(image_module.open(io.BytesIO(data)).convert("RGB"))  # (H, W, C)
    return np.transpose(array, (2, 0, 1))  # (C, H, W) uint8


def _batched(indices: Sequence[int], batch_size: int) -> Iterator[Sequence[int]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def compute_view_stats(
    facade: LiveLeRobotFacade,
    *,
    camera_keys: Sequence[str] = (),
    batch_size: int = DEFAULT_TABULAR_BATCH_SIZE,
    camera_decode_batch_size: int = DEFAULT_CAMERA_DECODE_BATCH_SIZE,
    camera_min_samples: int = CAMERA_MIN_SAMPLES,
    camera_max_samples: int = CAMERA_MAX_SAMPLES,
    camera_sample_power: float = CAMERA_SAMPLE_POWER,
    camera_downsample_target: int = CAMERA_DOWNSAMPLE_TARGET,
    camera_downsample_threshold: int = CAMERA_DOWNSAMPLE_THRESHOLD,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Compute per-feature normalization statistics for every frame of ``facade``.

    Returns ``(stats, sampling_record)``: ``stats`` maps every feature the
    manifest declares to its finalized stat arrays; ``sampling_record``
    documents the camera sampling policy actually applied (recorded in the
    published view row so the numbers are reproducible and auditable).

    ``camera_keys`` is the manifest's own declared ``observation.images.*``
    feature list — the manifest derive step decides which features exist, and
    stats cover exactly those, so ``info.json`` and ``stats.json`` can never
    disagree about the feature set.

    The tabular pass streams the *entire* view in bounded batches through the
    underlying :class:`AlignedFrameTrainingDataset` batch reader — deliberately
    not ``facade[i]``, which would decode every camera JPEG. The camera pass
    decodes only the bounded deterministic sample.
    """
    if batch_size <= 0 or camera_decode_batch_size <= 0:
        raise ViewStatsError("batch sizes must be positive")
    total = len(facade)
    if total == 0:
        raise ViewStatsError("view resolves zero frames; nothing to compute stats over")

    camera_keys = tuple(camera_keys)
    if camera_keys:
        image_module = _require_pil()

    vector_features = [
        key for key in ("observation.state", "action") if facade.has_feature(key)
    ]
    scalar_features = ("timestamp", "frame_index", "episode_index", "index")
    accumulators: dict[str, StreamingFeatureStats] = {
        key: StreamingFeatureStats() for key in (*vector_features, *scalar_features)
    }

    for start in range(0, total, batch_size):
        indices = range(start, min(start + batch_size, total))
        batch = facade.tabular_feature_batch(indices)
        for key in vector_features:
            rows = [row for row in batch[key] if row is not None]
            if rows:
                accumulators[key].update(np.asarray(rows, dtype=np.float64))
        for key in scalar_features:
            accumulators[key].update(np.asarray(batch[key], dtype=np.float64).reshape(-1, 1))

    stats: dict[str, dict[str, np.ndarray]] = {}
    for key in vector_features:
        if accumulators[key].count == 0:
            raise ViewStatsError(
                f"feature {key!r} is declared by this view's mapping but no resolved "
                "frame carried a complete vector for it; refusing to publish"
            )
        stats[key] = accumulators[key].statistics()
    for key in scalar_features:
        stats[key] = accumulators[key].statistics()

    sampling_record: dict[str, Any] = {
        "tabular": {"method": "full-scan", "batch_size": batch_size, "frames": total},
    }

    if camera_keys:
        num_samples = estimate_camera_samples(
            total,
            min_samples=camera_min_samples,
            max_samples=camera_max_samples,
            power=camera_sample_power,
        )
        sampled = camera_sample_indices(total, num_samples)
        camera_accumulators = {key: StreamingFeatureStats() for key in camera_keys}
        images_seen = {key: 0 for key in camera_keys}
        for chunk in _batched(sampled, camera_decode_batch_size):
            for item in facade.__getitems__(list(chunk)):
                for key in camera_keys:
                    data = item.get(key)
                    if data is None:
                        continue
                    frame = _decode_jpeg_chw(image_module, data)
                    frame = downsample_image(
                        frame,
                        target_size=camera_downsample_target,
                        max_size_threshold=camera_downsample_threshold,
                    )
                    channels = frame.shape[0]
                    pixels = frame.reshape(channels, -1).T  # (H*W, C)
                    camera_accumulators[key].update(pixels.astype(np.float64))
                    images_seen[key] += 1
        for key in camera_keys:
            if images_seen[key] == 0:
                raise ViewStatsError(
                    f"camera feature {key!r} is declared by this view but none of the "
                    f"{len(sampled)} sampled frames carried a decodable frame for it; "
                    "refusing to publish incomplete stats"
                )
            channel_stats = camera_accumulators[key].statistics()
            reshaped: dict[str, np.ndarray] = {}
            for stat_key, value in channel_stats.items():
                if stat_key == "count":
                    # Upstream counts sampled images, not pixels.
                    reshaped[stat_key] = np.array([images_seen[key]], dtype=np.int64)
                else:
                    reshaped[stat_key] = (value / 255.0).reshape(-1, 1, 1)
            stats[key] = reshaped
        sampling_record["camera"] = {
            "method": "linspace",
            "num_samples": len(sampled),
            "images_seen": images_seen,
            "min_samples": camera_min_samples,
            "max_samples": camera_max_samples,
            "power": camera_sample_power,
            "downsample_target": camera_downsample_target,
            "downsample_threshold": camera_downsample_threshold,
            "decode_batch_size": camera_decode_batch_size,
        }

    return stats, sampling_record


def serialize_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
    """Plain-JSON form of ``stats`` — exactly what upstream ``load_stats`` reads back.

    ``write_stats`` upstream applies ``serialize_dict`` (arrays -> lists) and
    ``load_stats`` applies ``cast_stats_to_numpy`` (lists -> arrays) over the
    nested dict; emitting nested plain lists here round-trips through both.
    """
    serialized: dict[str, dict[str, Any]] = {}
    for feature, feature_stats in stats.items():
        serialized[feature] = {
            key: np.asarray(value).tolist() for key, value in feature_stats.items()
        }
    return serialized
