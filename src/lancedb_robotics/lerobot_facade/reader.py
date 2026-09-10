"""Live LeRobot-shaped frame reader over a materialized alignment.

Wraps :class:`~lancedb_robotics.training.AlignedFrameTrainingDataset` --
which already resolves an alignment, filters ticks by quality policy, and
hydrates per-stream samples with caching/batching -- adding the two things it
does not do: episode/frame-index segmentation (``aligned_ticks`` has no
episode concept) and canonical ``observation.state``/``action`` vector
composition (backlog-0254-style tuple concatenation, applied at this layer).

Framework-neutral by default (plain dicts, plain ``list[float]`` vectors,
``observation.images.<key>`` as raw JPEG ``bytes``), matching
``AlignedFrameTrainingDataset``'s own design and the repo's established
pattern of deferring PyTorch into a thin, lazily-built wrapper (see
``training.py``'s ``_torch_map_dataset_cls``) -- importing this module never
requires the optional ``torch`` extra; only :func:`to_torch_dataset` does.
Camera bytes stay raw bytes even through ``to_torch_dataset`` -- this repo
carries no image-decode dependency (no Pillow), matching ``video.py``'s own
``VideoFrame.frame: bytes`` convention; a training loop decodes with
whatever image library it already uses.

``delta_timestamps`` windowing itself lives one layer up, in
``_reader_core.plan_windows``/``hydrate_windowed_items`` (backlog 0509) --
this facade stays per-frame and framework-neutral, contributing only the
:meth:`LiveLeRobotFacade.hydrate_batch` seam those helpers batch through.
There are no changes to ``dataset_export.py``/the CLI export path -- this
reads live, with no export step at all.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from typing import Any

from lancedb_robotics.lake import Lake
from lancedb_robotics.training import (
    TORCH_INSTALL_GUIDANCE,
    _resolve_alignment_job,
    torch_available,
)

from .doctor import preflight
from .episodes import FacadeEpisode, build_episode_index
from .mapping import CanonicalVectorMapping, compose_vector, validate_mapping
from .videos import VideoIndex, camera_feature_key


class LiveLeRobotFacade:
    """Random-access, episode-structured, LeRobot-shaped view over one alignment.

    Each item is a plain dict: ``{"episode_index", "frame_index", "index",
    "timestamp", "task", "observation.state"?, "action"?}``, with vectors as
    ``list[float]``. Use :func:`to_torch_dataset` to get ``torch.Tensor``
    values and a real ``torch.utils.data.Dataset``.
    """

    def __init__(
        self,
        lake: Lake,
        *,
        alignment: str | None = None,
        alignment_id: str | None = None,
        name: str | None = None,
        mapping: CanonicalVectorMapping,
        episode_ids: Sequence[str] | None = None,
        statuses: Sequence[str] | str | None = None,
        min_confidence: float | None = None,
        require_streams: bool | Sequence[str] = True,
    ) -> None:
        job = _resolve_alignment_job(
            lake, alignment=alignment, alignment_id=alignment_id, name=name
        )
        validate_mapping(mapping, job.get("streams") or ())
        self.mapping = mapping
        self._dataset = lake.training.aligned_dataset(
            alignment_id=str(job["alignment_id"]),
            streams=mapping.streams,
            shuffle=False,
            statuses=statuses,
            min_confidence=min_confidence,
            require_streams=require_streams,
        )
        self._episodes = build_episode_index(
            lake,
            job,
            episode_ids=episode_ids,
            allowed_tick_indices=self._dataset.tick_plan.tick_indices,
        )
        self.doctor_report = preflight(job, mapping, self._episodes)
        self.doctor_report.raise_if_unusable()
        self._dataset_index_by_tick = {
            self._dataset.tick_plan.tick_indices[plan_index]: dataset_index
            for dataset_index, plan_index in enumerate(self._dataset.epoch_plan.sample_indices)
        }
        self._frame_locations: tuple[tuple[int, int], ...] = tuple(
            (episode.episode_index, position)
            for episode in self._episodes
            for position in range(len(episode.tick_indices))
        )
        self._video_index: VideoIndex | None = None
        if mapping.camera_streams:
            run_id = json.loads(job["recipe"] or "{}").get("run_id")
            self._video_index = VideoIndex(lake, run_id=run_id)

    def __len__(self) -> int:
        return len(self._frame_locations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = self._normalize_index(index)
        episode_index, position = self._frame_locations[index]
        episode = self._episodes[episode_index]
        tick_index = episode.tick_indices[position]
        sample = self._dataset[self._dataset_index_by_tick[tick_index]]
        camera_frames = None
        if self._video_index is not None:
            requests = self._camera_requests(sample, episode)
            decoded = self._video_index.decode_batch(list(requests.values()))
            camera_frames = {key: decoded[location] for key, location in requests.items()}
        return self._to_item(sample, episode, position, index, camera_frames)

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        return self.hydrate_batch(indices)

    def hydrate_batch(
        self,
        indices: Sequence[int],
        *,
        camera_keys_per_frame: Sequence[Collection[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Batched hydration with per-frame camera-decode selection.

        ``camera_keys_per_frame`` carries one collection of
        ``observation.images.<key>`` feature keys per requested index, naming
        which camera features to decode for that frame (an empty collection
        decodes none); ``None`` decodes every mapped camera for every frame —
        exactly ``__getitems__``'s behavior, which delegates here. Costs stay
        batch-shaped either way: one ``AlignedFrameTrainingDataset.
        __getitems__`` call hydrates every tabular tick and one
        :meth:`VideoIndex.decode_batch` call serves every requested camera
        frame (each GOP blob fetched/un-zlipped once). The selective form
        exists for temporal-window assembly (backlog 0509): a window over
        ``action`` alone must not decode a camera frame per window member,
        and a windowed camera must decode only at its own window rows.
        """
        if camera_keys_per_frame is not None and len(camera_keys_per_frame) != len(indices):
            raise ValueError(
                f"camera_keys_per_frame has {len(camera_keys_per_frame)} entries "
                f"for {len(indices)} indices; pass exactly one collection per index"
            )
        normalized = [self._normalize_index(index) for index in indices]
        plans = [(index, *self._frame_locations[index]) for index in normalized]
        dataset_indices = [
            self._dataset_index_by_tick[self._episodes[episode_index].tick_indices[position]]
            for _, episode_index, position in plans
        ]
        samples = self._dataset.__getitems__(dataset_indices)

        camera_frames_per_item: list[dict[str, bytes] | None] = [None] * len(plans)
        if self._video_index is not None:
            per_item_requests = [
                self._camera_requests(
                    sample,
                    self._episodes[episode_index],
                    feature_keys=(
                        None if camera_keys_per_frame is None else camera_keys_per_frame[position_in_batch]
                    ),
                )
                for position_in_batch, ((_, episode_index, _), sample) in enumerate(
                    zip(plans, samples, strict=True)
                )
            ]
            all_requests = [
                location for requests in per_item_requests for location in requests.values()
            ]
            decoded = self._video_index.decode_batch(all_requests)
            camera_frames_per_item = [
                {key: decoded[location] for key, location in requests.items()}
                for requests in per_item_requests
            ]

        return [
            self._to_item(sample, self._episodes[episode_index], position, index, camera_frames)
            for (index, episode_index, position), sample, camera_frames in zip(
                plans, samples, camera_frames_per_item, strict=True
            )
        ]

    def has_feature(self, key: str) -> bool:
        """True when this facade's mapping declares the vector feature ``key``."""
        if key == "observation.state":
            return bool(self.mapping.state_streams)
        if key == "action":
            return bool(self.mapping.action_streams)
        return False

    def tabular_feature_batch(self, indices: Sequence[int]) -> dict[str, list[Any]]:
        """Vector/scalar feature values for ``indices``, with NO camera decode.

        The bounded-batch seam the view-publish statistics pass (backlog 0490)
        streams through: identical index math and vector composition to
        ``__getitems__``, but reads only the underlying tabular dataset —
        never :class:`VideoIndex`, so a full-view scan decodes zero JPEGs.
        """
        normalized = [self._normalize_index(index) for index in indices]
        plans = [(index, *self._frame_locations[index]) for index in normalized]
        dataset_indices = [
            self._dataset_index_by_tick[self._episodes[episode_index].tick_indices[position]]
            for _, episode_index, position in plans
        ]
        samples = self._dataset.__getitems__(dataset_indices)
        batch: dict[str, list[Any]] = {
            "index": [index for index, _, _ in plans],
            "episode_index": [episode_index for _, episode_index, _ in plans],
            "frame_index": [position for _, _, position in plans],
            "timestamp": [sample["timestamp_ns"] / 1_000_000_000.0 for sample in samples],
        }
        if self.mapping.state_streams:
            batch["observation.state"] = [
                compose_vector(sample["streams"], self.mapping.state_streams)
                for sample in samples
            ]
        if self.mapping.action_streams:
            batch["action"] = [
                compose_vector(sample["streams"], self.mapping.action_streams)
                for sample in samples
            ]
        return batch

    def _camera_requests(
        self,
        sample: dict[str, Any],
        episode: FacadeEpisode,
        feature_keys: Collection[str] | None = None,
    ) -> dict[str, tuple[str, int]]:
        """Resolve this sample's camera streams to ``{feature_key: (encoding_id, frame_index)}``.

        ``feature_keys`` (full ``observation.images.<key>`` names) restricts
        which cameras resolve; ``None`` resolves all mapped camera streams.
        """
        assert self._video_index is not None
        requests: dict[str, tuple[str, int]] = {}
        for stream in self.mapping.camera_streams:
            key = camera_feature_key(stream)
            if feature_keys is not None and f"observation.images.{key}" not in feature_keys:
                continue
            stream_sample = sample["streams"].get(stream)
            observation_id = stream_sample.get("observation_id") if stream_sample else None
            location = self._video_index.locate(
                camera_key=key, episode_id=episode.episode_id, observation_id=observation_id
            )
            if location is not None:
                requests[key] = location
        return requests

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return index

    def _to_item(
        self,
        sample: dict[str, Any],
        episode: FacadeEpisode,
        frame_index: int,
        absolute_index: int,
        camera_frames: dict[str, bytes] | None,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "episode_index": episode.episode_index,
            "frame_index": frame_index,
            "index": absolute_index,
            "timestamp": sample["timestamp_ns"] / 1_000_000_000.0,
            "task": episode.task,
        }
        state = compose_vector(sample["streams"], self.mapping.state_streams)
        if state is not None:
            item["observation.state"] = state
        action = compose_vector(sample["streams"], self.mapping.action_streams)
        if action is not None:
            item["action"] = action
        for key, frame_bytes in (camera_frames or {}).items():
            item[f"observation.images.{key}"] = frame_bytes
        return item


_TORCH_FACADE_CLS: Any = None


def to_torch_dataset(facade: LiveLeRobotFacade) -> Any:
    """Wrap ``facade`` as a ``torch.utils.data.Dataset`` yielding float32 tensors.

    Deferred/lazy, mirroring ``training.py``'s ``_torch_map_dataset_cls``:
    importing ``lerobot_facade.reader`` never requires the optional ``torch``
    extra -- only calling this function does.
    """
    global _TORCH_FACADE_CLS
    if not torch_available():
        raise RuntimeError(TORCH_INSTALL_GUIDANCE)
    if _TORCH_FACADE_CLS is None:
        from torch.utils.data import Dataset

        class _TorchLiveLeRobotFacade(Dataset):
            def __init__(self, inner: LiveLeRobotFacade) -> None:
                self._inner = inner

            def __len__(self) -> int:
                return len(self._inner)

            def __getitem__(self, index: int) -> dict[str, Any]:
                return _tensorize(self._inner[index])

            def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
                return [_tensorize(item) for item in self._inner.__getitems__(indices)]

        _TorchLiveLeRobotFacade.__module__ = __name__
        _TORCH_FACADE_CLS = _TorchLiveLeRobotFacade
    return _TORCH_FACADE_CLS(facade)


def _tensorize(item: dict[str, Any]) -> dict[str, Any]:
    import torch

    result = dict(item)
    for key in ("observation.state", "action"):
        if key in result:
            result[key] = torch.tensor(result[key], dtype=torch.float32)
    return result
