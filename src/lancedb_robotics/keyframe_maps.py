"""Content-addressed keyframe-map artifact helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

KEYFRAME_MAP_REF_PREFIX = "sha256:"
KEYFRAME_MAP_ARTIFACT_ID_PREFIX = "kfmap-"
KEYFRAME_MAP_REFERRER_ID_PREFIX = "kfmap-ref-"


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


def keyframe_map_artifact_referrer_id(
    *,
    artifact_id: str,
    referrer_table: str,
    encoding_id: str | None,
    video_id: str | None,
    run_id: str | None,
    episode_id: str | None,
    episode_index: int | None,
    camera_key: str | None,
    source_video_fingerprint: str | None,
    inspection_id: str | None,
) -> str:
    """Return the stable usage id for one keyframe-map artifact referrer."""

    payload = {
        "artifact_id": artifact_id,
        "referrer_table": referrer_table,
        "encoding_id": encoding_id,
        "video_id": video_id,
        "run_id": run_id,
        "episode_id": episode_id,
        "episode_index": episode_index,
        "camera_key": camera_key,
        "source_video_fingerprint": source_video_fingerprint,
        "inspection_id": inspection_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return KEYFRAME_MAP_REFERRER_ID_PREFIX + digest


def keyframe_map_entries_from_json(keyframe_json: str | None) -> list[dict[str, Any]]:
    if not keyframe_json:
        return []
    return [dict(entry) for entry in json.loads(keyframe_json)]


def keyframe_map_json_for_encoding(
    lake: Any,
    row: dict[str, Any],
    *,
    artifact_table_version: int | None = None,
) -> str:
    """Return inline or offloaded keyframe-map JSON for a video encoding row."""

    inline = row.get("keyframe_map_json")
    if inline:
        return str(inline)
    ref = str(row.get("keyframe_map_ref") or "")
    if not ref:
        return "[]"
    return load_keyframe_map_json(lake, ref, table_version=artifact_table_version)


def keyframe_map_entries_for_encoding(
    lake: Any,
    row: dict[str, Any],
    *,
    artifact_table_version: int | None = None,
) -> list[dict[str, Any]]:
    return keyframe_map_entries_from_json(
        keyframe_map_json_for_encoding(
            lake,
            row,
            artifact_table_version=artifact_table_version,
        )
    )


def load_keyframe_map_json(
    lake: Any,
    ref_or_artifact_id: str,
    *,
    table_version: int | None = None,
) -> str:
    """Load one keyframe-map body from the canonical artifact catalog."""

    row = load_keyframe_map_artifact(
        lake,
        ref_or_artifact_id,
        table_version=table_version,
    )
    return str(row["keyframe_map_json"])


def load_keyframe_map_artifact(
    lake: Any,
    ref_or_artifact_id: str,
    *,
    table_version: int | None = None,
) -> dict[str, Any]:
    """Load one keyframe-map artifact row, optionally at a pinned table version."""

    ref = _normalize_keyframe_map_ref(ref_or_artifact_id)
    artifact_id = (
        ref_or_artifact_id
        if ref_or_artifact_id.startswith(KEYFRAME_MAP_ARTIFACT_ID_PREFIX)
        else keyframe_map_artifact_id(ref.removeprefix(KEYFRAME_MAP_REF_PREFIX))
    )
    table = lake.table("keyframe_map_artifacts")
    if table_version is not None:
        try:
            table.checkout(int(table_version))
        except Exception as exc:  # noqa: BLE001 - expose table/version context
            raise KeyframeMapError(
                "cannot read keyframe-map artifacts at "
                f"version {int(table_version)}; the version may have been pruned: {exc}"
            ) from exc
    try:
        for row in table.to_arrow().to_pylist():
            if row.get("keyframe_map_ref") == ref or row.get("artifact_id") == artifact_id:
                return _validated_keyframe_map_artifact(
                    dict(row),
                    ref=ref,
                    artifact_id=artifact_id,
                    table_version=table_version,
                )
    finally:
        if table_version is not None:
            table.checkout_latest()
    raise KeyframeMapError(
        "missing keyframe-map artifact "
        f"{artifact_id!r} for {ref!r} in keyframe_map_artifacts{_version_phrase(table_version)}"
    )


def list_keyframe_map_referrers(
    lake: Any,
    ref_or_artifact_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """List usage rows for keyframe-map artifact bodies."""

    rows = [
        dict(row) for row in lake.table("keyframe_map_artifact_referrers").to_arrow().to_pylist()
    ]
    if ref_or_artifact_id:
        ref = _normalize_keyframe_map_ref(ref_or_artifact_id)
        artifact_id = (
            ref_or_artifact_id
            if ref_or_artifact_id.startswith(KEYFRAME_MAP_ARTIFACT_ID_PREFIX)
            else keyframe_map_artifact_id(ref.removeprefix(KEYFRAME_MAP_REF_PREFIX))
        )
        rows = [
            row
            for row in rows
            if row.get("artifact_id") == artifact_id or row.get("keyframe_map_ref") == ref
        ]
    return tuple(sorted(rows, key=_referrer_sort_key))


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


def keyframe_map_artifact_referrer_row(
    artifact: dict[str, Any],
    *,
    referrer_table: str = "video_encodings",
    referrer_kind: str = "video_encoding",
    referrer_table_version: int | None = None,
) -> dict[str, Any]:
    """Build one deterministic usage row for a keyframe-map artifact body."""

    artifact_id = str(artifact["artifact_id"])
    row = {
        "referrer_id": keyframe_map_artifact_referrer_id(
            artifact_id=artifact_id,
            referrer_table=referrer_table,
            encoding_id=artifact.get("encoding_id"),
            video_id=artifact.get("video_id"),
            run_id=artifact.get("run_id"),
            episode_id=artifact.get("episode_id"),
            episode_index=artifact.get("episode_index"),
            camera_key=artifact.get("camera_key"),
            source_video_fingerprint=artifact.get("source_video_fingerprint"),
            inspection_id=artifact.get("inspection_id"),
        ),
        "artifact_id": artifact_id,
        "keyframe_map_ref": artifact.get("keyframe_map_ref"),
        "content_sha256": artifact.get("content_sha256"),
        "referrer_kind": referrer_kind,
        "referrer_table": referrer_table,
        "referrer_table_version": referrer_table_version,
        "source_video_fingerprint": artifact.get("source_video_fingerprint"),
        "inspection_id": artifact.get("inspection_id"),
        "source_uri": artifact.get("source_uri"),
        "source_path": artifact.get("source_path"),
        "encoding_id": artifact.get("encoding_id"),
        "video_id": artifact.get("video_id"),
        "run_id": artifact.get("run_id"),
        "episode_id": artifact.get("episode_id"),
        "episode_index": artifact.get("episode_index"),
        "camera_key": artifact.get("camera_key"),
        "transform_id": artifact.get("transform_id"),
        "created_at": artifact.get("created_at"),
    }
    return row


def _referrer_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("artifact_id") or ""),
        str(row.get("run_id") or ""),
        int(row["episode_index"]) if row.get("episode_index") is not None else -1,
        str(row.get("camera_key") or ""),
        str(row.get("encoding_id") or ""),
        str(row.get("referrer_id") or ""),
    )


def _validated_keyframe_map_artifact(
    row: dict[str, Any],
    *,
    ref: str,
    artifact_id: str,
    table_version: int | None,
) -> dict[str, Any]:
    row_ref = str(row.get("keyframe_map_ref") or "")
    row_artifact_id = str(row.get("artifact_id") or "")
    expected_sha = ref.removeprefix(KEYFRAME_MAP_REF_PREFIX)
    row_sha = str(row.get("content_sha256") or expected_sha)
    if row_ref != ref or row_artifact_id != artifact_id or row_sha != expected_sha:
        raise KeyframeMapError(
            "keyframe-map artifact identity mismatch "
            f"for {artifact_id!r}{_version_phrase(table_version)}: "
            f"expected ref={ref!r}, content_sha256={expected_sha!r}; "
            f"got artifact_id={row_artifact_id!r}, ref={row_ref!r}, content_sha256={row_sha!r}"
        )
    keyframe_json = row.get("keyframe_map_json")
    if not keyframe_json:
        raise KeyframeMapError(
            f"keyframe-map artifact {artifact_id!r}{_version_phrase(table_version)} "
            "has no JSON body"
        )
    actual_sha = keyframe_map_content_sha256(str(keyframe_json))
    if actual_sha != expected_sha:
        raise KeyframeMapError(
            "keyframe-map artifact content mismatch "
            f"for {artifact_id!r}{_version_phrase(table_version)}: "
            f"expected sha256 {expected_sha}, got {actual_sha}"
        )
    row["keyframe_map_json"] = str(keyframe_json)
    return row


def _version_phrase(table_version: int | None) -> str:
    if table_version is None:
        return " at current table head"
    return f" at version {int(table_version)}"


def _normalize_keyframe_map_ref(value: str) -> str:
    if value.startswith(KEYFRAME_MAP_REF_PREFIX):
        return value
    if value.startswith(KEYFRAME_MAP_ARTIFACT_ID_PREFIX):
        return KEYFRAME_MAP_REF_PREFIX + value.removeprefix(KEYFRAME_MAP_ARTIFACT_ID_PREFIX)
    raise KeyframeMapError(f"unsupported keyframe-map ref {value!r}")
