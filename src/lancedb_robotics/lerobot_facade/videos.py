"""Live video/camera frame resolution for the LeRobot facade.

Resolve-at-open, mirroring how the tabular side resolves ``aligned_ticks``
once at construction rather than per-frame: one scan of ``videos`` +
``video_encodings`` for the facade's run builds an in-memory
``(camera_key, episode_id) -> encoding + ordered observation_ids`` index.
Per-frame reads batch by ``encoding_id`` (:func:`VideoIndex.decode_batch`
fetches/un-zlips each GOP blob once even when several requested frames share
it) via ``fetch_blob``/``decode_frame_from_encoding`` directly -- never
``seek_video_frame``'s per-call full ``video_encodings`` table scan
(``video.py:278-306``/``:1013-1031``).

Requires ``video_encodings`` to already hold genuinely decoded image bytes
(``video.py``'s ``_encoding_row`` redecodes via ``schema_registry`` as of
this session's fix) -- this module does not itself decode ROS envelopes, it
only resolves *which* bytes to fetch.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lancedb_robotics.blob import fetch_blob
from lancedb_robotics.lake import Lake
from lancedb_robotics.video import decode_frame_from_encoding


def camera_feature_key(stream: str) -> str:
    """Derive a LeRobot ``observation.images.<key>`` key from a stream/topic name."""
    key = re.sub(r"[^a-z0-9]+", "_", stream.lower()).strip("_")
    return key or "camera"


@dataclass(frozen=True)
class _CameraVideo:
    encoding_id: str
    observation_ids: tuple[str, ...]  # ordered; position is the video-local frame index


class VideoIndex:
    """Resolve-at-open ``(camera_key, episode_id)`` lookup for one run's videos."""

    def __init__(self, lake: Lake, *, run_id: str) -> None:
        self._lake = lake
        videos = (
            lake.table("videos")
            .search()
            .where(f"run_id = {_sql_literal(run_id)}")
            .to_arrow()
            .to_pylist()
        )
        encodings = (
            lake.table("video_encodings")
            .search()
            .where(f"run_id = {_sql_literal(run_id)}")
            .to_arrow()
            .to_pylist()
        )
        self._encoding_row_by_id: dict[str, dict[str, Any]] = {
            row["encoding_id"]: row for row in encodings
        }
        # Newest encoding wins per video (mirrors video.py's _select_encoding_row
        # sort-by-created_at-then-take-last convention).
        latest_encoding_by_video: dict[str, dict[str, Any]] = {}
        for row in sorted(encodings, key=lambda r: (r["created_at"], r["encoding_id"])):
            latest_encoding_by_video[row["video_id"]] = row
        self._video_by_key: dict[tuple[str, str], _CameraVideo] = {}
        for video in videos:
            encoding = latest_encoding_by_video.get(video["video_id"])
            if encoding is None:
                continue
            self._video_by_key[(video["camera_key"], video["episode_id"])] = _CameraVideo(
                encoding_id=encoding["encoding_id"],
                observation_ids=tuple(video.get("observation_ids") or ()),
            )

    def locate(
        self, *, camera_key: str, episode_id: str, observation_id: str | None
    ) -> tuple[str, int] | None:
        """Resolve a specific camera observation to ``(encoding_id, frame_index)``.

        Returns ``None`` when the stream had no aligned observation for this
        tick (unaligned/out-of-tolerance) or no video was ever encoded for
        this ``(camera_key, episode_id)`` -- callers should omit the feature
        for that frame, the same "None means absent" convention as
        :func:`mapping.compose_vector`.
        """
        if not observation_id:
            return None
        video = self._video_by_key.get((camera_key, episode_id))
        if video is None or observation_id not in video.observation_ids:
            return None
        return video.encoding_id, video.observation_ids.index(observation_id)

    def decode_batch(self, requests: Sequence[tuple[str, int]]) -> dict[tuple[str, int], bytes]:
        """Decode every requested ``(encoding_id, frame_index)``.

        Fetches/un-zlips each distinct ``encoding_id``'s GOP-packed blob once,
        regardless of how many requested frames it contains -- the batch-then-
        fetch discipline, at the encoding level (not per source blob, since a
        GOP-decode is cheap in-memory work once the bytes are in hand).
        """
        by_encoding: dict[str, list[int]] = {}
        for encoding_id, frame_index in requests:
            by_encoding.setdefault(encoding_id, []).append(frame_index)
        frames: dict[tuple[str, int], bytes] = {}
        for encoding_id, frame_indices in by_encoding.items():
            row = self._encoding_row_by_id[encoding_id]
            encoded = fetch_blob(
                self._lake.table("video_encodings"),
                "data",
                encoding_id,
                id_column="encoding_id",
                connection_spec=self._lake.connection_spec,
            )
            for frame_index in frame_indices:
                decoded = decode_frame_from_encoding(row, encoded, frame_index)
                frames[(encoding_id, frame_index)] = decoded.frame
        return frames


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"
