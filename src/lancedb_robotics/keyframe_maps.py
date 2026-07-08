"""Content-addressed keyframe-map artifact helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

KEYFRAME_MAP_REF_PREFIX = "sha256:"
KEYFRAME_MAP_ARTIFACT_ID_PREFIX = "kfmap-"


class KeyframeMapError(Exception):
    """Raised when an offloaded keyframe map cannot be resolved."""


def keyframe_map_content_sha256(keyframe_json: str) -> str:
    """Return the canonical JSON content digest for a keyframe map body."""

    return hashlib.sha256(keyframe_json.encode()).hexdigest()


def keyframe_map_ref(keyframe_json: str) -> str:
    """Return the stable ref stored on video encoding rows."""

    return KEYFRAME_MAP_REF_PREFIX + keyframe_map_content_sha256(keyframe_json)


def keyframe_map_artifact_id(content_sha256: str) -> str:
    """Return the catalog artifact id for a content digest."""

    return KEYFRAME_MAP_ARTIFACT_ID_PREFIX + content_sha256


def keyframe_map_entries_from_json(keyframe_json: str | None) -> list[dict[str, Any]]:
    if not keyframe_json:
        return []
    return [dict(entry) for entry in json.loads(keyframe_json)]


def keyframe_map_json_for_encoding(lake: Any, row: dict[str, Any]) -> str:
    """Return inline or offloaded keyframe-map JSON for a video encoding row."""

    inline = row.get("keyframe_map_json")
    if inline:
        return str(inline)
    ref = str(row.get("keyframe_map_ref") or "")
    if not ref:
        return "[]"
    return load_keyframe_map_json(lake, ref)


def keyframe_map_entries_for_encoding(lake: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    return keyframe_map_entries_from_json(keyframe_map_json_for_encoding(lake, row))


def load_keyframe_map_json(lake: Any, ref_or_artifact_id: str) -> str:
    """Load one keyframe-map body from the canonical artifact catalog."""

    ref = _normalize_keyframe_map_ref(ref_or_artifact_id)
    artifact_id = (
        ref_or_artifact_id
        if ref_or_artifact_id.startswith(KEYFRAME_MAP_ARTIFACT_ID_PREFIX)
        else keyframe_map_artifact_id(ref.removeprefix(KEYFRAME_MAP_REF_PREFIX))
    )
    for row in lake.table("keyframe_map_artifacts").to_arrow().to_pylist():
        if row.get("keyframe_map_ref") == ref or row.get("artifact_id") == artifact_id:
            keyframe_json = row.get("keyframe_map_json")
            if keyframe_json:
                return str(keyframe_json)
            raise KeyframeMapError(f"keyframe-map artifact {artifact_id!r} has no JSON body")
    raise KeyframeMapError(f"missing keyframe-map artifact for {ref_or_artifact_id!r}")


def should_inline_keyframe_map(
    keyframe_json: str,
    *,
    threshold_bytes: int | None,
    threshold_frames: int | None,
) -> bool:
    """Return true when a keyframe map should stay inline on the encoding row."""

    if keyframe_json == "[]":
        return True
    if threshold_bytes is not None and len(keyframe_json.encode()) > int(threshold_bytes):
        return False
    if threshold_frames is not None:
        frame_count, _ = keyframe_map_shape(keyframe_map_entries_from_json(keyframe_json))
        if frame_count > int(threshold_frames):
            return False
    return True


def keyframe_map_shape(entries: list[dict[str, Any]]) -> tuple[int, int]:
    """Return ``(frame_count, gop_count)`` for either encoded-GOP or MP4 maps."""

    frame_count = 0
    for entry in entries:
        if entry.get("frames"):
            frame_count += len(entry.get("frames") or [])
            continue
        if entry.get("frame_count") is not None:
            frame_count += int(entry["frame_count"])
            continue
        first = entry.get("first_frame_index")
        last = entry.get("last_frame_index")
        if first is not None and last is not None:
            frame_count += max(0, int(last) - int(first) + 1)
    return frame_count, len(entries)


def keyframe_map_artifact_row(
    keyframe_json: str,
    *,
    source_video_fingerprint: str | None,
    inspection_id: str | None,
    source_uri: str | None,
    source_path: str | None,
    encoding_id: str | None,
    video_id: str | None,
    run_id: str | None,
    episode_id: str | None,
    episode_index: int | None,
    camera_key: str | None,
    transform_id: str,
    created_at: datetime,
) -> dict[str, Any]:
    """Build one canonical ``keyframe_map_artifacts`` row."""

    content_sha = keyframe_map_content_sha256(keyframe_json)
    entries = keyframe_map_entries_from_json(keyframe_json)
    frame_count, gop_count = keyframe_map_shape(entries)
    return {
        "artifact_id": keyframe_map_artifact_id(content_sha),
        "keyframe_map_ref": KEYFRAME_MAP_REF_PREFIX + content_sha,
        "content_sha256": content_sha,
        "json_size_bytes": len(keyframe_json.encode()),
        "frame_count": frame_count,
        "gop_count": gop_count,
        "source_video_fingerprint": source_video_fingerprint,
        "inspection_id": inspection_id,
        "source_uri": source_uri,
        "source_path": source_path,
        "encoding_id": encoding_id,
        "video_id": video_id,
        "run_id": run_id,
        "episode_id": episode_id,
        "episode_index": episode_index,
        "camera_key": camera_key,
        "keyframe_map_json": keyframe_json,
        "transform_id": transform_id,
        "created_at": created_at,
    }


def _normalize_keyframe_map_ref(value: str) -> str:
    if value.startswith(KEYFRAME_MAP_REF_PREFIX):
        return value
    if value.startswith(KEYFRAME_MAP_ARTIFACT_ID_PREFIX):
        return KEYFRAME_MAP_REF_PREFIX + value.removeprefix(KEYFRAME_MAP_ARTIFACT_ID_PREFIX)
    raise KeyframeMapError(f"unsupported keyframe-map ref {value!r}")
