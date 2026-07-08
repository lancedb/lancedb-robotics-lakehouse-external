"""Codec-aware GOP video encoding and frame seek (backlog 0030)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import struct
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from lancedb_robotics._mp4 import Mp4MetadataError, read_mp4_frame_sample
from lancedb_robotics.blob import fetch_blob, fetch_blobs
from lancedb_robotics.keyframe_maps import (
    KeyframeMapError,
    keyframe_map_entries_for_encoding,
    keyframe_map_json_for_encoding,
    keyframe_map_ref,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.schemas import TRANSFORM_RUNS_SCHEMA, VIDEO_ENCODINGS_SCHEMA

VIDEO_ENCODING_BLOB_COLUMN = "data"
DEFAULT_CODEC = "lrb-gop-zlib"
DEFAULT_GOP_SIZE = 2
DEFAULT_RESOLUTION = "unknown"
_FRAME_LEN = struct.Struct(">Q")
_SOURCE_MP4_DECODER_BACKENDS: dict[str, Any] = {}


class VideoError(Exception):
    """Raised when codec-aware video rows cannot be encoded or decoded."""


@dataclass(frozen=True)
class VideoEncodingReport:
    """Summary of a video encode transform."""

    lake_uri: str
    transform_id: str
    codec: str
    gop_size: int
    encodings_written: int
    encoding_ids: tuple[str, ...]


@dataclass(frozen=True)
class VideoFrame:
    """One decoded frame plus seek-cost metadata."""

    encoding_id: str
    video_id: str
    episode_id: str
    episode_index: int
    frame_index: int
    camera_key: str
    frame: bytes
    decoder: str
    bytes_read: int
    encoded_size_bytes: int
    byte_range: tuple[int, int]
    gop_index: int
    gop_first_frame_index: int
    gop_last_frame_index: int


@dataclass(frozen=True)
class VideoConformanceReport:
    """Decoded source-video conformance summary."""

    lake_uri: str
    transform_id: str
    decoder: str
    status: str
    results: tuple[dict[str, Any], ...]
    status_counts: dict[str, int]
    codec_counts: dict[str, int]
    backend_versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "transform_id": self.transform_id,
            "decoder": self.decoder,
            "status": self.status,
            "results": [dict(row) for row in self.results],
            "status_counts": dict(self.status_counts),
            "codec_counts": dict(self.codec_counts),
            "backend_versions": dict(self.backend_versions),
        }


@dataclass(frozen=True)
class _DecodedSourceFrame:
    frame: bytes
    backend: str
    version: str
    seek_strategy: str
    frames_decoded: int
    seek_frame_index: int | None = None
    fallback_reason: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] = ()


class _SourceMp4DecoderUnavailable(VideoError):
    """Raised when an optional source MP4 decoder backend is not installed."""


class LakeVideo:
    """Convenience namespace exposed as ``lake.video``."""

    def __init__(self, lake: Lake) -> None:
        self._lake = lake

    def encode(
        self,
        *,
        video_id: str | None = None,
        episode_id: str | None = None,
        camera_key: str | None = None,
        codec: str = DEFAULT_CODEC,
        gop_size: int = DEFAULT_GOP_SIZE,
        resolution: str | None = None,
        fps: float | None = None,
        nvdec_compatible: bool = False,
        created_by: str = "lancedb-robotics",
    ) -> VideoEncodingReport:
        return encode_videos(
            self._lake,
            video_id=video_id,
            episode_id=episode_id,
            camera_key=camera_key,
            codec=codec,
            gop_size=gop_size,
            resolution=resolution,
            fps=fps,
            nvdec_compatible=nvdec_compatible,
            created_by=created_by,
        )

    def seek(
        self,
        episode: str | int | Any,
        frame: int,
        *,
        camera_key: str | None = None,
        encoding_id: str | None = None,
        decoder: str = "auto",
    ) -> VideoFrame:
        return seek_video_frame(
            self._lake,
            episode,
            frame,
            camera_key=camera_key,
            encoding_id=encoding_id,
            decoder=decoder,
        )

    def encodings(self, *, video_id: str | None = None) -> tuple[dict[str, Any], ...]:
        rows = self._lake.table("video_encodings").to_arrow().to_pylist()
        if video_id is not None:
            rows = [row for row in rows if row["video_id"] == video_id]
        return tuple(sorted(rows, key=_encoding_sort_key))

    def conform_source(
        self,
        samples: Iterable[Mapping[str, Any]] = (),
        *,
        decoder: str = "auto",
        fail_on_mismatch: bool = False,
        created_by: str = "lancedb-robotics",
    ) -> VideoConformanceReport:
        return conform_source_mp4_frames(
            self._lake,
            samples=samples,
            decoder=decoder,
            fail_on_mismatch=fail_on_mismatch,
            created_by=created_by,
        )


def encode_videos(
    lake: Lake,
    *,
    video_id: str | None = None,
    episode_id: str | None = None,
    camera_key: str | None = None,
    codec: str = DEFAULT_CODEC,
    gop_size: int = DEFAULT_GOP_SIZE,
    resolution: str | None = None,
    fps: float | None = None,
    nvdec_compatible: bool = False,
    created_by: str = "lancedb-robotics",
) -> VideoEncodingReport:
    """Encode selected ``videos`` rows into GOP-addressable ``video_encodings``."""
    if gop_size <= 0:
        raise VideoError("gop_size must be positive")
    if codec != DEFAULT_CODEC:
        raise VideoError(f"unsupported codec {codec!r}; expected {DEFAULT_CODEC!r}")

    videos = _select_video_rows(lake, video_id=video_id, episode_id=episode_id, camera_key=camera_key)
    transform_id = "tfm-video-encode-" + _digest(
        {
            "video_ids": [row["video_id"] for row in videos],
            "codec": codec,
            "gop_size": int(gop_size),
            "resolution": resolution or DEFAULT_RESOLUTION,
            "fps": fps,
            "nvdec_compatible": bool(nvdec_compatible),
        }
    )
    now = datetime.now(UTC)
    encoding_rows = [
        _encoding_row(
            lake,
            video,
            codec=codec,
            gop_size=int(gop_size),
            resolution=resolution or DEFAULT_RESOLUTION,
            fps=fps,
            nvdec_compatible=bool(nvdec_compatible),
            transform_id=transform_id,
            now=now,
        )
        for video in videos
    ]

    table = lake.table("video_encodings")
    for row in encoding_rows:
        table.delete(f"encoding_id = '{row['encoding_id']}'")
    if encoding_rows:
        table.add(pa.Table.from_pylist(encoding_rows, schema=VIDEO_ENCODINGS_SCHEMA))

    _record_transform(
        lake,
        transform_id=transform_id,
        params={
            "video_ids": [row["video_id"] for row in videos],
            "encoding_ids": [row["encoding_id"] for row in encoding_rows],
            "codec": codec,
            "gop_size": int(gop_size),
            "resolution": resolution or DEFAULT_RESOLUTION,
            "fps": fps,
            "nvdec_compatible": bool(nvdec_compatible),
        },
        created_by=created_by,
    )

    return VideoEncodingReport(
        lake_uri=lake.uri,
        transform_id=transform_id,
        codec=codec,
        gop_size=int(gop_size),
        encodings_written=len(encoding_rows),
        encoding_ids=tuple(row["encoding_id"] for row in encoding_rows),
    )


def seek_video_frame(
    lake: Lake,
    episode: str | int | Any,
    frame: int,
    *,
    camera_key: str | None = None,
    encoding_id: str | None = None,
    decoder: str = "auto",
) -> VideoFrame:
    """Seek to one frame via the nearest enclosing GOP byte range."""
    if frame < 0:
        raise VideoError("frame must be non-negative")
    row = _select_encoding_row(
        lake,
        episode=episode,
        frame_index=int(frame),
        camera_key=camera_key,
        encoding_id=encoding_id,
    )
    encoded = fetch_blob(
        lake.table("video_encodings"),
        VIDEO_ENCODING_BLOB_COLUMN,
        row["encoding_id"],
        id_column="encoding_id",
    )
    if not encoded:
        return _seek_video_reference_frame(lake, row, int(frame), decoder=decoder)
    return decode_frame_from_encoding(row, encoded, int(frame), decoder=decoder)


def decode_frame_from_encoding(
    row: dict[str, Any],
    encoded: bytes,
    frame_index: int,
    *,
    decoder: str = "auto",
) -> VideoFrame:
    """Decode ``frame_index`` from an already-fetched encoded byte stream."""
    entry = _keyframe_entry(row, frame_index)
    start = int(entry["byte_start"])
    end = int(entry["byte_end"])
    gop_bytes = encoded[start:end]
    decoder_used = _decoder_for(row, decoder)
    if decoder_used == "nvdec":
        frames = _decode_gop_nvdec(gop_bytes)
    else:
        frames = _decode_gop_cpu(gop_bytes)

    local_index = frame_index - int(entry["first_frame_index"])
    try:
        frame = frames[local_index]
    except IndexError as exc:
        raise VideoError(
            f"keyframe map for {row['encoding_id']!r} does not contain frame {frame_index}"
        ) from exc

    return VideoFrame(
        encoding_id=row["encoding_id"],
        video_id=row["video_id"],
        episode_id=row["episode_id"],
        episode_index=int(row["episode_index"]),
        frame_index=int(frame_index),
        camera_key=row["camera_key"],
        frame=frame,
        decoder=decoder_used,
        bytes_read=len(gop_bytes),
        encoded_size_bytes=int(row["encoded_size_bytes"]),
        byte_range=(start, end),
        gop_index=int(entry["gop_index"]),
        gop_first_frame_index=int(entry["first_frame_index"]),
        gop_last_frame_index=int(entry["last_frame_index"]),
    )


def conform_source_mp4_frames(
    lake: Lake,
    *,
    samples: Iterable[Mapping[str, Any]] = (),
    decoder: str = "auto",
    fail_on_mismatch: bool = False,
    created_by: str = "lancedb-robotics",
) -> VideoConformanceReport:
    """Verify optional decoded-frame conformance for source MP4 references."""

    sample_rows = [dict(sample) for sample in samples] or _default_source_conformance_samples(lake)
    started = datetime.now(UTC)
    results = tuple(
        _conform_source_mp4_sample(lake, sample, decoder=decoder)
        for sample in sample_rows
    )
    status_counts = _count_by(results, "status")
    codec_counts = _count_by(results, "codec")
    backend_versions = {
        str(result["decoder_backend"]): str(result["decoder_version"])
        for result in results
        if result.get("decoder_backend") and result.get("decoder_version")
    }
    status = _conformance_status(status_counts)
    transform_id = "tfm-video-conformance-" + _digest(
        {
            "decoder": decoder,
            "samples": [
                {
                    "episode": sample.get("episode", sample.get("episode_index")),
                    "frame_index": sample.get("frame_index"),
                    "camera_key": sample.get("camera_key"),
                    "encoding_id": sample.get("encoding_id"),
                    "expected_sha256": _expected_frame_sha256(sample),
                }
                for sample in sample_rows
            ],
        }
    )
    report = VideoConformanceReport(
        lake_uri=lake.uri,
        transform_id=transform_id,
        decoder=decoder,
        status=status,
        results=results,
        status_counts=status_counts,
        codec_counts=codec_counts,
        backend_versions=backend_versions,
    )
    _record_conformance_transform(
        lake,
        transform_id=transform_id,
        report=report,
        started_at=started,
        finished_at=datetime.now(UTC),
        created_by=created_by,
    )
    if fail_on_mismatch and status in {"failed", "error"}:
        raise VideoError(f"source MP4 conformance {transform_id} finished with status {status}")
    return report


def _conform_source_mp4_sample(
    lake: Lake,
    sample: dict[str, Any],
    *,
    decoder: str,
) -> dict[str, Any]:
    frame_index = int(sample.get("frame_index", 0))
    row = _select_encoding_row(
        lake,
        episode=sample.get("episode", sample.get("episode_index", 0)),
        frame_index=frame_index,
        camera_key=sample.get("camera_key"),
        encoding_id=sample.get("encoding_id"),
    )
    entry = _keyframe_entry(row, frame_index, lake=lake)
    frame_entry = _sample_frame_entry(entry, frame_index)
    video = _video_row_for_encoding(lake, row)
    expected_sha256 = _expected_frame_sha256(sample)
    byte_range = _frame_byte_range(entry, frame_entry)
    base = {
        "encoding_id": row["encoding_id"],
        "video_id": row["video_id"],
        "episode_id": row["episode_id"],
        "episode_index": int(row["episode_index"]),
        "frame_index": frame_index,
        "camera_key": row["camera_key"],
        "codec": str(row.get("codec") or "unknown"),
        "gop_index": int(entry["gop_index"]),
        "keyframe_frame_index": int(entry["keyframe_frame_index"]),
        "byte_range": list(byte_range) if byte_range else None,
        "expected_sha256": expected_sha256,
        "decoder_requested": decoder,
        "decoder_backend": None,
        "decoder_version": None,
        "seek_strategy": None,
        "frames_decoded": 0,
        "seek_frame_index": None,
        "fallback_reason": None,
        "decoded_sha256": None,
        "status": "skipped",
        "reason": None,
        "error": None,
    }
    if row.get("data") is not None:
        return {**base, "reason": "not-source-mp4-reference"}
    if not expected_sha256:
        return {**base, "reason": "missing-expected-frame"}
    try:
        decoded = _decode_source_mp4_frame(lake, row, video, frame_index, decoder=decoder)
    except _SourceMp4DecoderUnavailable as exc:
        return {**base, "reason": "decoder-unavailable", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - conformance should report per-sample failures.
        return {**base, "status": "error", "reason": "decoder-error", "error": str(exc)}

    decoded_sha256 = hashlib.sha256(decoded.frame).hexdigest()
    passed = decoded_sha256 == expected_sha256
    return {
        **base,
        "decoder_backend": decoded.backend,
        "decoder_version": decoded.version,
        "seek_strategy": decoded.seek_strategy,
        "frames_decoded": decoded.frames_decoded,
        "seek_frame_index": decoded.seek_frame_index,
        "fallback_reason": decoded.fallback_reason,
        "decoded_sha256": decoded_sha256,
        "status": "passed" if passed else "failed",
        "reason": None if passed else "decoded-frame-mismatch",
    }


def _decode_source_mp4_frame(
    lake: Lake,
    row: dict[str, Any],
    video: dict[str, Any],
    frame_index: int,
    *,
    decoder: str,
) -> _DecodedSourceFrame:
    codec = str(row.get("codec") or video.get("codec") or "unknown").lower()
    backend = _source_mp4_decoder_backend(codec, decoder)
    entry = _keyframe_entry(row, frame_index, lake=lake)
    frame_entry = _sample_frame_entry(entry, frame_index)
    uri = str(video.get("raw_uri") or video.get("uri") or "")
    if not uri:
        raise VideoError(f"video {row['video_id']!r} has no source URI for conformance")
    if backend in _SOURCE_MP4_DECODER_BACKENDS:
        decoded = _SOURCE_MP4_DECODER_BACKENDS[backend](
            lake=lake,
            row=row,
            video=video,
            uri=uri,
            entry=entry,
            frame_entry=frame_entry,
            frame_index=frame_index,
            codec=codec,
        )
        return _normalize_decoded_source_frame(decoded, backend=backend)
    if backend == "pillow-mjpeg":
        keyframe_json = keyframe_map_json_for_encoding(lake, row)
        return _decode_source_mp4_frame_pillow(uri, keyframe_json, frame_index)
    if backend == "pyav":
        return _decode_source_mp4_frame_pyav(uri, frame_index, entry=entry, frame_entry=frame_entry)
    raise _SourceMp4DecoderUnavailable(f"unknown source MP4 decoder backend {backend!r}")


def _source_mp4_decoder_backend(codec: str, requested: str) -> str:
    if requested in _SOURCE_MP4_DECODER_BACKENDS:
        return requested
    if requested == "pillow-mjpeg":
        _require_optional_module("PIL", extra="media", backend=requested)
        if codec not in {"mjpeg", "mjpg"}:
            raise _SourceMp4DecoderUnavailable(
                f"decoder {requested!r} only supports MJPEG source samples, got codec {codec!r}"
            )
        return requested
    if requested == "pyav":
        _require_optional_module("av", extra="video-decode", backend=requested)
        return requested
    if requested != "auto":
        raise VideoError(
            "source MP4 conformance decoder must be auto, pyav, pillow-mjpeg, "
            "or a registered test/backend name"
        )
    if codec in {"mjpeg", "mjpg"} and importlib.util.find_spec("PIL") is not None:
        return "pillow-mjpeg"
    if importlib.util.find_spec("av") is not None:
        return "pyav"
    raise _SourceMp4DecoderUnavailable(
        "source MP4 decoded-frame conformance requires optional backend 'av' "
        "(install the video-decode extra) or Pillow for MJPEG samples"
    )


def _require_optional_module(module: str, *, extra: str, backend: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise _SourceMp4DecoderUnavailable(
            f"decoder backend {backend!r} requires optional dependency {module!r}; "
            f"install lancedb-robotics[{extra}]"
        )


def _decode_source_mp4_frame_pillow(
    uri: str,
    keyframe_json: str,
    frame_index: int,
) -> _DecodedSourceFrame:
    from io import BytesIO

    from PIL import Image

    sample = read_mp4_frame_sample(uri, keyframe_json, frame_index)
    with Image.open(BytesIO(sample)) as image:
        frame = image.convert("RGB").tobytes()
    return _DecodedSourceFrame(
        frame=frame,
        backend="pillow-mjpeg",
        version=_package_version("pillow"),
        seek_strategy="sample-byte-range",
        frames_decoded=1,
    )


def _decode_source_mp4_frame_pyav(
    uri: str,
    frame_index: int,
    *,
    entry: Mapping[str, Any] | None = None,
    frame_entry: Mapping[str, Any] | None = None,
) -> _DecodedSourceFrame:
    fallback_reason = _pyav_seek_fallback_reason(entry, frame_entry)
    if fallback_reason is None:
        try:
            return _decode_source_mp4_frame_pyav_seek(
                uri,
                frame_index,
                entry=entry or {},
                frame_entry=frame_entry or {},
            )
        except Exception as exc:  # noqa: BLE001 - fall back and report why in conformance output.
            fallback_reason = f"pyav-seek-failed:{type(exc).__name__}: {exc}"

    return _decode_source_mp4_frame_pyav_sequential(
        uri,
        frame_index,
        fallback_reason=fallback_reason,
    )


def _decode_source_mp4_frame_pyav_seek(
    uri: str,
    frame_index: int,
    *,
    entry: Mapping[str, Any],
    frame_entry: Mapping[str, Any],
) -> _DecodedSourceFrame:
    import av

    keyframe_time_units = int(entry["keyframe_time_units"])
    target_time_units = int(frame_entry["sample_time_units"])
    seek_frame_index = int(entry.get("keyframe_frame_index", entry.get("first_frame_index", 0)))
    target_local_index = int(frame_index) - seek_frame_index
    container = av.open(uri)
    try:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise VideoError(f"source MP4 {uri!r} has no video stream")
        stream = streams[0]
        container.seek(keyframe_time_units, stream=stream, any_frame=False, backward=True)
        for decoded_count, frame in enumerate(container.decode(stream), start=1):
            if frame.pts is not None:
                if int(frame.pts) != target_time_units:
                    continue
            elif decoded_count - 1 != target_local_index:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            return _DecodedSourceFrame(
                frame=rgb.tobytes(),
                backend="pyav",
                version=_package_version("av"),
                seek_strategy="nearest-keyframe",
                frames_decoded=decoded_count,
                seek_frame_index=seek_frame_index,
                dtype="uint8",
                shape=tuple(int(value) for value in rgb.shape),
            )
    finally:
        container.close()
    raise VideoError(
        f"source MP4 {uri!r} did not decode frame {frame_index} after seeking to "
        f"keyframe {seek_frame_index}"
    )


def _decode_source_mp4_frame_pyav_sequential(
    uri: str,
    frame_index: int,
    *,
    fallback_reason: str | None,
) -> _DecodedSourceFrame:
    import av

    container = av.open(uri)
    try:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise VideoError(f"source MP4 {uri!r} has no video stream")
        for index, frame in enumerate(container.decode(streams[0])):
            if index != frame_index:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            return _DecodedSourceFrame(
                frame=rgb.tobytes(),
                backend="pyav",
                version=_package_version("av"),
                seek_strategy="sequential" if fallback_reason is None else "sequential-fallback",
                frames_decoded=index + 1,
                seek_frame_index=0,
                fallback_reason=fallback_reason,
                dtype="uint8",
                shape=tuple(int(value) for value in rgb.shape),
            )
    finally:
        container.close()
    raise VideoError(f"source MP4 {uri!r} did not decode frame {frame_index}")


def _pyav_seek_fallback_reason(
    entry: Mapping[str, Any] | None,
    frame_entry: Mapping[str, Any] | None,
) -> str | None:
    if not entry:
        return "missing-keyframe-entry"
    if frame_entry is None:
        return "missing-frame-entry"
    if entry.get("keyframe_time_units") is None:
        return "missing-keyframe-time-units"
    if frame_entry.get("sample_time_units") is None:
        return "missing-frame-time-units"
    return None


def _normalize_decoded_source_frame(decoded: Any, *, backend: str) -> _DecodedSourceFrame:
    if isinstance(decoded, _DecodedSourceFrame):
        return decoded
    if isinstance(decoded, bytes):
        return _DecodedSourceFrame(
            frame=decoded,
            backend=backend,
            version="test",
            seek_strategy="registered",
            frames_decoded=1,
            dtype="uint8",
        )
    if isinstance(decoded, Mapping):
        frame = decoded.get("frame")
        if not isinstance(frame, bytes):
            raise VideoError(f"registered decoder {backend!r} did not return frame bytes")
        return _DecodedSourceFrame(
            frame=frame,
            backend=str(decoded.get("backend") or backend),
            version=str(decoded.get("version") or "unknown"),
            seek_strategy=str(decoded.get("seek_strategy") or "registered"),
            frames_decoded=int(decoded.get("frames_decoded") or 1),
            seek_frame_index=(
                int(decoded["seek_frame_index"])
                if decoded.get("seek_frame_index") is not None
                else None
            ),
            fallback_reason=(
                str(decoded["fallback_reason"])
                if decoded.get("fallback_reason") is not None
                else None
            ),
            dtype=str(decoded.get("dtype") or "") or None,
            shape=tuple(int(value) for value in (decoded.get("shape") or ())),
        )
    raise VideoError(f"registered decoder {backend!r} returned unsupported result")


def _default_source_conformance_samples(lake: Lake) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in lake.table("video_encodings").to_arrow().to_pylist():
        if row.get("data") is not None:
            continue
        try:
            keyframes = keyframe_map_entries_for_encoding(lake, row)
        except KeyframeMapError:
            continue
        if not keyframes:
            continue
        samples.append(
            {
                "encoding_id": row["encoding_id"],
                "frame_index": int(keyframes[0]["first_frame_index"]),
            }
        )
    return samples


def _expected_frame_sha256(sample: Mapping[str, Any]) -> str | None:
    expected = sample.get("expected_frame")
    if expected is None:
        expected = sample.get("expected_bytes")
    if isinstance(expected, str):
        expected = expected.encode()
    if isinstance(expected, bytes):
        return hashlib.sha256(expected).hexdigest()
    expected_sha256 = sample.get("expected_sha256")
    return str(expected_sha256) if expected_sha256 else None


def _frame_byte_range(
    entry: dict[str, Any],
    frame_entry: dict[str, Any] | None,
) -> tuple[int, int] | None:
    if frame_entry is not None:
        start = frame_entry.get("byte_start")
        end = frame_entry.get("byte_end")
    else:
        start = entry.get("byte_start")
        end = entry.get("byte_end")
    if start is None or end is None:
        return None
    return int(start), int(end)


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _conformance_status(status_counts: Mapping[str, int]) -> str:
    if status_counts.get("failed"):
        return "failed"
    if status_counts.get("error"):
        return "error"
    if status_counts.get("passed"):
        return "passed"
    return "skipped"


def _record_conformance_transform(
    lake: Lake,
    *,
    transform_id: str,
    report: VideoConformanceReport,
    started_at: datetime,
    finished_at: datetime,
    created_by: str,
) -> None:
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{transform_id}'")
    input_uris = sorted(
        {
            str(video.get("raw_uri") or video.get("uri"))
            for video in lake.table("videos").to_arrow().to_pylist()
            if video.get("raw_uri") or video.get("uri")
        }
    )
    row = {
        "transform_id": transform_id,
        "kind": "video-conformance",
        "source_id": None,
        "input_uris": input_uris,
        "input_table_versions": [
            {"table": "videos", "version": int(lake.table("videos").version), "tag": ""},
            {
                "table": "video_encodings",
                "version": int(lake.table("video_encodings").version),
                "tag": "",
            },
        ],
        "output_tables": [],
        "params": json.dumps(report.to_dict(), sort_keys=True),
        "status": "completed",
        "error": None if report.status not in {"failed", "error"} else report.status,
        "started_at": started_at,
        "finished_at": finished_at,
        "created_by": created_by,
        "created_at": finished_at,
    }
    transforms.add(pa.Table.from_pylist([row], schema=TRANSFORM_RUNS_SCHEMA))
    emit_transform_lineage(lake, row)


def _select_video_rows(
    lake: Lake,
    *,
    video_id: str | None,
    episode_id: str | None,
    camera_key: str | None,
) -> list[dict[str, Any]]:
    rows = lake.table("videos").to_arrow().to_pylist()
    if video_id is not None:
        rows = [row for row in rows if row["video_id"] == video_id]
    if episode_id is not None:
        rows = [row for row in rows if row["episode_id"] == episode_id]
    if camera_key is not None:
        rows = [row for row in rows if row["camera_key"] == camera_key]
    rows = sorted(rows, key=lambda row: (row.get("episode_index") or 0, row["camera_key"], row["video_id"]))
    if not rows:
        raise VideoError("no videos matched the requested selection")
    return rows


def _encoding_row(
    lake: Lake,
    video: dict[str, Any],
    *,
    codec: str,
    gop_size: int,
    resolution: str,
    fps: float | None,
    nvdec_compatible: bool,
    transform_id: str,
    now: datetime,
) -> dict[str, Any]:
    observation_ids = [str(value) for value in (video.get("observation_ids") or [])]
    if not observation_ids:
        raise VideoError(f"video {video['video_id']!r} has no observation_ids")

    payloads = fetch_blobs(
        lake.table("observations"),
        "payload_blob",
        observation_ids,
        id_column="observation_id",
    )
    missing = [obs_id for obs_id in observation_ids if not payloads.get(obs_id)]
    if missing:
        raise VideoError(
            f"video {video['video_id']!r} cannot be encoded; missing frame bytes for {missing}"
        )

    frames = [payloads[obs_id] for obs_id in observation_ids]
    encoded, keyframe_map = _encode_gops(frames, gop_size)
    keyframe_json = json.dumps(keyframe_map, sort_keys=True, separators=(",", ":"))
    source_hash = _sha256_join(frames)
    encoding_id = "venc-" + _digest(
        {
            "video_id": video["video_id"],
            "observation_ids": observation_ids,
            "source_hash": source_hash,
            "codec": codec,
            "gop_size": int(gop_size),
            "resolution": resolution,
            "fps": fps if fps is not None else _infer_fps(lake, video),
            "nvdec_compatible": bool(nvdec_compatible),
        }
    )
    effective_fps = float(fps) if fps is not None else _infer_fps(lake, video)

    return {
        "encoding_id": encoding_id,
        "video_id": video["video_id"],
        "run_id": video["run_id"],
        "episode_id": video["episode_id"],
        "episode_index": int(video["episode_index"]),
        "camera_key": video["camera_key"],
        "codec": codec,
        "gop_size": int(gop_size),
        "resolution": resolution,
        "fps": effective_fps,
        "frame_count": len(frames),
        "keyframe_map_ref": keyframe_map_ref(keyframe_json),
        "keyframe_map_json": keyframe_json,
        "nvdec_compatible": bool(nvdec_compatible),
        "source_size_bytes": sum(len(frame) for frame in frames),
        "encoded_size_bytes": len(encoded),
        "data": encoded,
        "transform_id": transform_id,
        "created_at": now,
    }


def _encode_gops(frames: list[bytes], gop_size: int) -> tuple[bytes, list[dict[str, int]]]:
    chunks: list[bytes] = []
    keyframe_map: list[dict[str, int]] = []
    offset = 0
    for gop_index, first in enumerate(range(0, len(frames), gop_size)):
        gop_frames = frames[first : first + gop_size]
        raw = _pack_frames(gop_frames)
        encoded = zlib.compress(raw)
        start = offset
        offset += len(encoded)
        end = offset
        chunks.append(encoded)
        keyframe_map.append(
            {
                "gop_index": gop_index,
                "keyframe_frame_index": first,
                "first_frame_index": first,
                "last_frame_index": first + len(gop_frames) - 1,
                "frame_count": len(gop_frames),
                "byte_start": start,
                "byte_end": end,
                "encoded_size_bytes": len(encoded),
                "raw_size_bytes": len(raw),
            }
        )
    return b"".join(chunks), keyframe_map


def _pack_frames(frames: list[bytes]) -> bytes:
    out = bytearray()
    for frame in frames:
        out += _FRAME_LEN.pack(len(frame))
        out += frame
    return bytes(out)


def _unpack_frames(data: bytes) -> list[bytes]:
    frames: list[bytes] = []
    offset = 0
    while offset < len(data):
        if offset + _FRAME_LEN.size > len(data):
            raise VideoError("corrupt GOP payload: truncated frame length")
        (size,) = _FRAME_LEN.unpack(data[offset : offset + _FRAME_LEN.size])
        offset += _FRAME_LEN.size
        end = offset + int(size)
        if end > len(data):
            raise VideoError("corrupt GOP payload: truncated frame bytes")
        frames.append(data[offset:end])
        offset = end
    return frames


def _decode_gop_cpu(gop_bytes: bytes) -> list[bytes]:
    return _unpack_frames(zlib.decompress(gop_bytes))


def _decode_gop_nvdec(gop_bytes: bytes) -> list[bytes]:
    # The package-level contract is parity with CPU. The optional dependency seam
    # can replace this body with a hardware decoder without changing callers.
    return _decode_gop_cpu(gop_bytes)


def _decoder_for(row: dict[str, Any], requested: str) -> str:
    if requested not in {"auto", "cpu", "nvdec"}:
        raise VideoError("decoder must be one of auto, cpu, nvdec")
    if requested == "cpu":
        return "cpu"
    if bool(row.get("nvdec_compatible")) and _nvdec_available():
        return "nvdec"
    return "cpu"


def _nvdec_available() -> bool:
    return (
        importlib.util.find_spec("PyNvVideoCodec") is not None
        or importlib.util.find_spec("decord") is not None
    )


def _select_encoding_row(
    lake: Lake,
    *,
    episode: str | int | Any,
    frame_index: int,
    camera_key: str | None,
    encoding_id: str | None,
) -> dict[str, Any]:
    rows = lake.table("video_encodings").to_arrow().to_pylist()
    if encoding_id is not None:
        rows = [row for row in rows if row["encoding_id"] == encoding_id]
    else:
        rows = [row for row in rows if _matches_episode(row, episode)]
        rows = [row for row in rows if _has_frame(lake, row, frame_index)]
        if camera_key is not None:
            rows = [row for row in rows if row["camera_key"] == camera_key]
    if not rows:
        raise VideoError("no video encoding matched the requested episode/frame")
    return sorted(rows, key=_encoding_sort_key)[-1]


def _matches_episode(row: dict[str, Any], episode: str | int | Any) -> bool:
    if hasattr(episode, "episode_id"):
        episode = episode.episode_id
    if isinstance(episode, int):
        return int(row["episode_index"]) == episode
    value = str(episode)
    return row["episode_id"] == value or str(row["episode_index"]) == value


def _has_frame(lake: Lake, row: dict[str, Any], frame_index: int) -> bool:
    try:
        _keyframe_entry(row, frame_index, lake=lake)
    except VideoError:
        return False
    return True


def _keyframe_entry(
    row: dict[str, Any],
    frame_index: int,
    *,
    lake: Lake | None = None,
) -> dict[str, Any]:
    if lake is None:
        keyframe_entries = json.loads(row.get("keyframe_map_json") or "[]")
    else:
        try:
            keyframe_entries = keyframe_map_entries_for_encoding(lake, row)
        except KeyframeMapError as exc:
            raise VideoError(str(exc)) from exc
    for entry in keyframe_entries:
        if int(entry["first_frame_index"]) <= frame_index <= int(entry["last_frame_index"]):
            return entry
    raise VideoError(
        f"encoding {row['encoding_id']!r} has no GOP containing frame {frame_index}"
    )


def _seek_video_reference_frame(
    lake: Lake,
    row: dict[str, Any],
    frame_index: int,
    *,
    decoder: str,
) -> VideoFrame:
    if decoder not in {"auto", "cpu"}:
        raise VideoError("source MP4 reference seek supports decoder auto or cpu")
    entry = _keyframe_entry(row, frame_index, lake=lake)
    frame_entry = _sample_frame_entry(entry, frame_index)
    if not frame_entry:
        raise VideoError(
            f"encoding {row['encoding_id']!r} has no MP4 sample range for frame {frame_index}"
        )
    video = _video_row_for_encoding(lake, row)
    uri = str(video.get("raw_uri") or video.get("uri") or "")
    if not uri:
        raise VideoError(f"video {row['video_id']!r} has no source URI for reference seek")
    try:
        keyframe_json = keyframe_map_json_for_encoding(lake, row)
        sample = read_mp4_frame_sample(uri, keyframe_json, frame_index)
    except (OSError, Mp4MetadataError) as exc:
        raise VideoError(f"cannot read source MP4 sample for {row['encoding_id']!r}: {exc}") from exc
    start = int(frame_entry["byte_start"])
    end = int(frame_entry["byte_end"])
    return VideoFrame(
        encoding_id=row["encoding_id"],
        video_id=row["video_id"],
        episode_id=row["episode_id"],
        episode_index=int(row["episode_index"]),
        frame_index=int(frame_index),
        camera_key=row["camera_key"],
        frame=sample,
        decoder="mp4-sample",
        bytes_read=len(sample),
        encoded_size_bytes=int(row["encoded_size_bytes"] or len(sample)),
        byte_range=(start, end),
        gop_index=int(entry["gop_index"]),
        gop_first_frame_index=int(entry["first_frame_index"]),
        gop_last_frame_index=int(entry["last_frame_index"]),
    )


def _sample_frame_entry(entry: dict[str, Any], frame_index: int) -> dict[str, Any] | None:
    for frame in entry.get("frames") or []:
        if int(frame.get("frame_index", -1)) == int(frame_index):
            return frame
    return None


def _video_row_for_encoding(lake: Lake, row: dict[str, Any]) -> dict[str, Any]:
    for video in lake.table("videos").to_arrow().to_pylist():
        if video["video_id"] == row["video_id"]:
            return video
    raise VideoError(f"encoding {row['encoding_id']!r} references missing video {row['video_id']!r}")


def _infer_fps(lake: Lake, video: dict[str, Any]) -> float:
    ids = set(video.get("observation_ids") or [])
    timestamps = sorted(
        int(row["timestamp_ns"])
        for row in lake.table("observations").to_arrow().to_pylist()
        if row["observation_id"] in ids
    )
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False) if b > a]
    if not deltas:
        return 0.0
    median_delta = sorted(deltas)[len(deltas) // 2]
    return 1_000_000_000 / median_delta


def _record_transform(
    lake: Lake,
    *,
    transform_id: str,
    params: dict[str, Any],
    created_by: str,
) -> None:
    now = datetime.now(UTC)
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{transform_id}'")
    transform_row = {
        "transform_id": transform_id,
        "kind": "video-encoding",
        "source_id": None,
        "input_uris": [],
        "input_table_versions": [
            {"table": "videos", "version": int(lake.table("videos").version), "tag": ""},
            {
                "table": "observations",
                "version": int(lake.table("observations").version),
                "tag": "",
            },
        ],
        "output_tables": ["video_encodings"],
        "params": json.dumps(params, sort_keys=True),
        "status": "completed",
        "error": None,
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): video-encoding transform without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)


def _encoding_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("created_at") or datetime.min.replace(tzinfo=UTC),
        row.get("encoding_id") or "",
    )


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _sha256_join(chunks: list[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(_FRAME_LEN.pack(len(chunk)))
        digest.update(chunk)
    return digest.hexdigest()


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"
