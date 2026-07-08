"""Dependency-light MP4 metadata and sample-table helpers."""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from lancedb_robotics.storage import StorageConfigError, open_binary_uri


class Mp4MetadataError(Exception):
    """Raised when an MP4 file cannot provide the expected sample metadata."""


@dataclass(frozen=True)
class Mp4VideoMetadata:
    """Parsed metadata for one video track inside an MP4 container."""

    codec: str
    codec_tag: str
    codec_profile: str | None
    width: int | None
    height: int | None
    fps: float | None
    frame_count: int
    gop_size: int | None
    keyframe_map: tuple[dict[str, Any], ...]
    duration_seconds: float | None
    bytes_read: int

    @property
    def resolution(self) -> str | None:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return None


@dataclass
class _Mp4Track:
    handler_type: str | None = None
    timescale: int | None = None
    duration: int | None = None
    width: int | None = None
    height: int | None = None
    codec_tag: str | None = None
    codec_profile: str | None = None
    stts: list[tuple[int, int]] = field(default_factory=list)
    ctts: list[tuple[int, int]] = field(default_factory=list)
    stss: list[int] = field(default_factory=list)
    stsz: list[int] = field(default_factory=list)
    stsc: list[tuple[int, int, int]] = field(default_factory=list)
    chunk_offsets: list[int] = field(default_factory=list)


@dataclass
class _Mp4State:
    brands: list[str] = field(default_factory=list)
    tracks: list[_Mp4Track] = field(default_factory=list)
    bytes_read: int = 0


_CONTAINER_BOXES = {
    b"moov",
    b"mdia",
    b"minf",
    b"stbl",
    b"dinf",
    b"edts",
    b"mvex",
}

_CODEC_NAMES = {
    "avc1": "h264",
    "avc3": "h264",
    "hev1": "h265",
    "hvc1": "h265",
    "av01": "av1",
    "mp4v": "mp4v",
    "mjpg": "mjpeg",
}


def inspect_mp4_video(
    path: str | Path,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> Mp4VideoMetadata:
    """Parse MP4 video-track metadata without decoding or reading media payloads."""

    try:
        with open_binary_uri(path, storage_options=storage_options, auth_ref=auth_ref) as fh:
            size = _stream_size(fh)
            if size <= 0:
                raise Mp4MetadataError("empty MP4 file")

            state = _Mp4State()
            _parse_boxes(fh, 0, size, state, current_track=None)
    except StorageConfigError as exc:
        raise Mp4MetadataError(str(exc)) from exc

    if size <= 0:
        raise Mp4MetadataError("empty MP4 file")

    if not state.brands:
        raise Mp4MetadataError("missing ftyp box")
    track = _select_video_track(state)
    frame_count = _frame_count(track)
    if frame_count <= 0:
        raise Mp4MetadataError("video track has no samples")

    sample_ranges = _sample_byte_ranges(track)
    sample_timings = _sample_timings(track)
    sync_frames = _sync_frame_indices(track, frame_count)
    keyframe_map = tuple(_build_keyframe_map(frame_count, sync_frames, sample_ranges, sample_timings))
    timescale = int(track.timescale or 0)
    sample_duration = sum(count * delta for count, delta in track.stts)
    fps = _fps(frame_count, timescale, sample_duration, track.duration)
    duration_seconds = (
        float(track.duration) / timescale
        if track.duration is not None and timescale > 0
        else (float(sample_duration) / timescale if sample_duration and timescale > 0 else None)
    )
    codec_tag = track.codec_tag or "unknown"

    return Mp4VideoMetadata(
        codec=_CODEC_NAMES.get(codec_tag, codec_tag),
        codec_tag=codec_tag,
        codec_profile=track.codec_profile,
        width=track.width,
        height=track.height,
        fps=fps,
        frame_count=frame_count,
        gop_size=_infer_gop_size(sync_frames, frame_count),
        keyframe_map=keyframe_map,
        duration_seconds=duration_seconds,
        bytes_read=state.bytes_read,
    )


def read_mp4_frame_sample(
    path: str | Path,
    keyframe_map: str | list[dict[str, Any]] | tuple[dict[str, Any], ...],
    frame_index: int,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> bytes:
    """Read the encoded MP4 sample bytes for ``frame_index`` from a keyframe map."""

    entries = json.loads(keyframe_map) if isinstance(keyframe_map, str) else list(keyframe_map)
    for entry in entries:
        for frame in entry.get("frames") or []:
            if int(frame.get("frame_index", -1)) != int(frame_index):
                continue
            start = frame.get("byte_start")
            end = frame.get("byte_end")
            if start is None or end is None:
                raise Mp4MetadataError(f"frame {frame_index} has no MP4 sample byte range")
            return _read_range(
                path,
                int(start),
                int(end),
                storage_options=storage_options,
                auth_ref=auth_ref,
            )
    raise Mp4MetadataError(f"keyframe map does not contain frame {frame_index}")


def _stream_size(fh: BinaryIO) -> int:
    try:
        current = fh.tell()
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(current)
    except (OSError, ValueError) as exc:
        raise Mp4MetadataError("MP4 source is not seekable") from exc
    return int(size)


def _parse_boxes(
    fh: BinaryIO,
    start: int,
    end: int,
    state: _Mp4State,
    *,
    current_track: _Mp4Track | None,
) -> None:
    offset = start
    while offset + 8 <= end:
        fh.seek(offset)
        header = fh.read(8)
        state.bytes_read += len(header)
        if len(header) < 8:
            return
        size32, box_type = struct.unpack(">I4s", header)
        header_size = 8
        if size32 == 1:
            extended = fh.read(8)
            state.bytes_read += len(extended)
            if len(extended) < 8:
                raise Mp4MetadataError("truncated extended MP4 box header")
            (box_size,) = struct.unpack(">Q", extended)
            header_size = 16
        elif size32 == 0:
            box_size = end - offset
        else:
            box_size = int(size32)

        payload_start = offset + header_size
        box_end = offset + box_size
        if box_size < header_size or box_end > end:
            raise Mp4MetadataError(f"invalid MP4 box {box_type.decode('latin1', 'replace')}")

        if box_type == b"ftyp":
            state.brands = _parse_ftyp(_read_payload(fh, payload_start, box_end, state))
        elif box_type == b"trak":
            track = _Mp4Track()
            state.tracks.append(track)
            _parse_boxes(fh, payload_start, box_end, state, current_track=track)
        elif box_type in _CONTAINER_BOXES:
            _parse_boxes(fh, payload_start, box_end, state, current_track=current_track)
        elif current_track is not None:
            payload = _read_payload(fh, payload_start, box_end, state)
            _parse_track_leaf(current_track, box_type, payload)

        offset = box_end


def _read_payload(fh: BinaryIO, start: int, end: int, state: _Mp4State) -> bytes:
    fh.seek(start)
    payload = fh.read(max(0, end - start))
    state.bytes_read += len(payload)
    return payload


def _parse_ftyp(payload: bytes) -> list[str]:
    brands: list[str] = []
    if len(payload) >= 4:
        brands.append(payload[:4].decode("latin1", "replace").strip())
    for offset in range(8, len(payload) - 3, 4):
        brand = payload[offset : offset + 4].decode("latin1", "replace").strip()
        if brand:
            brands.append(brand)
    return brands


def _parse_track_leaf(track: _Mp4Track, box_type: bytes, payload: bytes) -> None:
    if box_type == b"hdlr":
        track.handler_type = _parse_handler(payload)
    elif box_type == b"tkhd":
        width, height = _parse_tkhd_dimensions(payload)
        track.width = width or track.width
        track.height = height or track.height
    elif box_type == b"mdhd":
        track.timescale, track.duration = _parse_mdhd(payload)
    elif box_type == b"stsd":
        codec_tag, width, height, profile = _parse_stsd(payload)
        track.codec_tag = codec_tag or track.codec_tag
        track.codec_profile = profile or track.codec_profile
        track.width = width or track.width
        track.height = height or track.height
    elif box_type == b"stts":
        track.stts = _parse_stts(payload)
    elif box_type == b"ctts":
        track.ctts = _parse_ctts(payload)
    elif box_type == b"stss":
        track.stss = _parse_stss(payload)
    elif box_type == b"stsz":
        track.stsz = _parse_stsz(payload)
    elif box_type == b"stsc":
        track.stsc = _parse_stsc(payload)
    elif box_type == b"stco":
        track.chunk_offsets = _parse_stco(payload)
    elif box_type == b"co64":
        track.chunk_offsets = _parse_co64(payload)


def _parse_handler(payload: bytes) -> str | None:
    if len(payload) < 12:
        return None
    return payload[8:12].decode("latin1", "replace").strip() or None


def _parse_tkhd_dimensions(payload: bytes) -> tuple[int | None, int | None]:
    if len(payload) < 8:
        return None, None
    width = _fixed_16_16(payload[-8:-4])
    height = _fixed_16_16(payload[-4:])
    return width, height


def _parse_mdhd(payload: bytes) -> tuple[int | None, int | None]:
    if len(payload) < 20:
        return None, None
    version = payload[0]
    if version == 1:
        if len(payload) < 32:
            return None, None
        return _u32(payload, 20), _u64(payload, 24)
    return _u32(payload, 12), _u32(payload, 16)


def _parse_stsd(payload: bytes) -> tuple[str | None, int | None, int | None, str | None]:
    if len(payload) < 16:
        return None, None, None, None
    count = _u32(payload, 4)
    offset = 8
    for _ in range(count):
        if offset + 16 > len(payload):
            return None, None, None, None
        entry_size = _u32(payload, offset)
        if entry_size < 16 or offset + entry_size > len(payload):
            return None, None, None, None
        codec_tag = payload[offset + 4 : offset + 8].decode("latin1", "replace").strip()
        entry = payload[offset + 8 : offset + entry_size]
        width = _u16(entry, 24) if len(entry) >= 28 else None
        height = _u16(entry, 26) if len(entry) >= 28 else None
        profile = _parse_codec_profile(entry[78:]) if len(entry) > 78 else None
        return codec_tag or None, width, height, profile
    return None, None, None, None


def _parse_codec_profile(payload: bytes) -> str | None:
    offset = 0
    while offset + 8 <= len(payload):
        size = _u32(payload, offset)
        box_type = payload[offset + 4 : offset + 8]
        if size < 8 or offset + size > len(payload):
            return None
        data = payload[offset + 8 : offset + size]
        if box_type == b"avcC" and len(data) >= 2:
            return f"avc-profile-{data[1]}"
        if box_type == b"hvcC" and data:
            return f"hvc-profile-{data[0] & 0x1f}"
        offset += size
    return None


def _parse_stts(payload: bytes) -> list[tuple[int, int]]:
    if len(payload) < 8:
        return []
    entries: list[tuple[int, int]] = []
    count = _u32(payload, 4)
    offset = 8
    for _ in range(count):
        if offset + 8 > len(payload):
            break
        entries.append((_u32(payload, offset), _u32(payload, offset + 4)))
        offset += 8
    return entries


def _parse_ctts(payload: bytes) -> list[tuple[int, int]]:
    if len(payload) < 8:
        return []
    version = payload[0]
    entries: list[tuple[int, int]] = []
    count = _u32(payload, 4)
    offset = 8
    for _ in range(count):
        if offset + 8 > len(payload):
            break
        sample_count = _u32(payload, offset)
        if version == 1:
            sample_offset = struct.unpack_from(">i", payload, offset + 4)[0]
        else:
            sample_offset = _u32(payload, offset + 4)
        entries.append((sample_count, sample_offset))
        offset += 8
    return entries


def _parse_stss(payload: bytes) -> list[int]:
    if len(payload) < 8:
        return []
    values: list[int] = []
    count = _u32(payload, 4)
    offset = 8
    for _ in range(count):
        if offset + 4 > len(payload):
            break
        values.append(_u32(payload, offset))
        offset += 4
    return values


def _parse_stsz(payload: bytes) -> list[int]:
    if len(payload) < 12:
        return []
    sample_size = _u32(payload, 4)
    count = _u32(payload, 8)
    if sample_size:
        return [sample_size] * count
    sizes: list[int] = []
    offset = 12
    for _ in range(count):
        if offset + 4 > len(payload):
            break
        sizes.append(_u32(payload, offset))
        offset += 4
    return sizes


def _parse_stsc(payload: bytes) -> list[tuple[int, int, int]]:
    if len(payload) < 8:
        return []
    entries: list[tuple[int, int, int]] = []
    count = _u32(payload, 4)
    offset = 8
    for _ in range(count):
        if offset + 12 > len(payload):
            break
        entries.append((_u32(payload, offset), _u32(payload, offset + 4), _u32(payload, offset + 8)))
        offset += 12
    return entries


def _parse_stco(payload: bytes) -> list[int]:
    if len(payload) < 8:
        return []
    offsets: list[int] = []
    count = _u32(payload, 4)
    offset = 8
    for _ in range(count):
        if offset + 4 > len(payload):
            break
        offsets.append(_u32(payload, offset))
        offset += 4
    return offsets


def _parse_co64(payload: bytes) -> list[int]:
    if len(payload) < 8:
        return []
    offsets: list[int] = []
    count = _u32(payload, 4)
    offset = 8
    for _ in range(count):
        if offset + 8 > len(payload):
            break
        offsets.append(_u64(payload, offset))
        offset += 8
    return offsets


def _select_video_track(state: _Mp4State) -> _Mp4Track:
    video_tracks = [track for track in state.tracks if track.handler_type == "vide"]
    candidates = video_tracks or [track for track in state.tracks if track.codec_tag]
    if not candidates:
        raise Mp4MetadataError("missing video track")
    return candidates[0]


def _frame_count(track: _Mp4Track) -> int:
    if track.stsz:
        return len(track.stsz)
    if track.stts:
        return sum(count for count, _delta in track.stts)
    return 0


def _fps(
    frame_count: int,
    timescale: int,
    sample_duration: int,
    track_duration: int | None,
) -> float | None:
    duration = sample_duration or int(track_duration or 0)
    if frame_count <= 0 or timescale <= 0 or duration <= 0:
        return None
    return float(frame_count) * float(timescale) / float(duration)


def _sample_byte_ranges(track: _Mp4Track) -> list[tuple[int, int] | None]:
    sample_count = _frame_count(track)
    ranges: list[tuple[int, int] | None] = [None] * sample_count
    if not track.stsz or not track.chunk_offsets or not track.stsc:
        return ranges

    stsc_entries = sorted(track.stsc, key=lambda item: item[0])
    sample_index = 0
    for chunk_index, chunk_offset in enumerate(track.chunk_offsets, start=1):
        samples_per_chunk = _samples_per_chunk(stsc_entries, chunk_index)
        offset = int(chunk_offset)
        for _ in range(samples_per_chunk):
            if sample_index >= sample_count:
                return ranges
            sample_size = int(track.stsz[sample_index])
            ranges[sample_index] = (offset, offset + sample_size)
            offset += sample_size
            sample_index += 1
    return ranges


def _sample_timings(track: _Mp4Track) -> list[dict[str, Any] | None]:
    sample_count = _frame_count(track)
    if sample_count <= 0 or not track.stts or not track.timescale:
        return [None] * sample_count

    durations: list[int] = []
    for count, delta in track.stts:
        durations.extend([int(delta)] * int(count))
        if len(durations) >= sample_count:
            break
    if len(durations) < sample_count:
        durations.extend([0] * (sample_count - len(durations)))

    composition_offsets: list[int] = []
    for count, offset in track.ctts:
        composition_offsets.extend([int(offset)] * int(count))
        if len(composition_offsets) >= sample_count:
            break
    if len(composition_offsets) < sample_count:
        composition_offsets.extend([0] * (sample_count - len(composition_offsets)))

    timescale = int(track.timescale)
    decode_time = 0
    timings: list[dict[str, Any] | None] = []
    for sample_index in range(sample_count):
        duration = int(durations[sample_index])
        presentation_time = int(decode_time + composition_offsets[sample_index])
        timings.append(
            {
                "sample_time_units": presentation_time,
                "sample_decode_time_units": int(decode_time),
                "sample_duration_units": duration,
                "sample_time_seconds": float(presentation_time) / float(timescale),
                "timebase_num": 1,
                "timebase_den": timescale,
            }
        )
        decode_time += duration
    return timings


def _samples_per_chunk(entries: list[tuple[int, int, int]], chunk_index: int) -> int:
    active = entries[0]
    for entry in entries:
        if int(entry[0]) > chunk_index:
            break
        active = entry
    return int(active[1])


def _sync_frame_indices(track: _Mp4Track, frame_count: int) -> list[int]:
    if not track.stss:
        return list(range(frame_count))
    return sorted({sample_number - 1 for sample_number in track.stss if 1 <= sample_number <= frame_count})


def _build_keyframe_map(
    frame_count: int,
    sync_frames: list[int],
    sample_ranges: list[tuple[int, int] | None],
    sample_timings: list[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    if not sync_frames:
        sync_frames = [0]
    entries: list[dict[str, Any]] = []
    for gop_index, first_frame in enumerate(sync_frames):
        next_first = sync_frames[gop_index + 1] if gop_index + 1 < len(sync_frames) else frame_count
        last_frame = max(first_frame, min(frame_count - 1, next_first - 1))
        frames: list[dict[str, Any]] = []
        starts: list[int] = []
        ends: list[int] = []
        for frame_index in range(first_frame, last_frame + 1):
            sample_range = sample_ranges[frame_index] if frame_index < len(sample_ranges) else None
            frame: dict[str, Any] = {
                "frame_index": frame_index,
                "sample_index": frame_index + 1,
                "is_keyframe": frame_index == first_frame,
            }
            sample_timing = sample_timings[frame_index] if frame_index < len(sample_timings) else None
            if sample_timing is not None:
                frame.update(sample_timing)
            if sample_range is not None:
                start, end = sample_range
                frame["byte_start"] = int(start)
                frame["byte_end"] = int(end)
                starts.append(int(start))
                ends.append(int(end))
            frames.append(frame)
        entry = {
            "source": "mp4-sample-table",
            "reencoded": False,
            "gop_index": gop_index,
            "keyframe_frame_index": first_frame,
            "first_frame_index": first_frame,
            "last_frame_index": last_frame,
            "frame_count": len(frames),
            "byte_start": min(starts) if starts else None,
            "byte_end": max(ends) if ends else None,
            "frames": frames,
        }
        keyframe_timing = sample_timings[first_frame] if first_frame < len(sample_timings) else None
        if keyframe_timing is not None:
            entry.update(
                {
                    "keyframe_time_units": keyframe_timing["sample_time_units"],
                    "keyframe_time_seconds": keyframe_timing["sample_time_seconds"],
                    "timebase_num": keyframe_timing["timebase_num"],
                    "timebase_den": keyframe_timing["timebase_den"],
                }
            )
        entries.append(entry)
    return entries


def _infer_gop_size(sync_frames: list[int], frame_count: int) -> int | None:
    if frame_count <= 0:
        return None
    if len(sync_frames) < 2:
        return frame_count
    deltas = [b - a for a, b in zip(sync_frames, sync_frames[1:], strict=False) if b > a]
    if not deltas:
        return frame_count
    return sorted(deltas)[len(deltas) // 2]


def _read_range(
    path: str | Path,
    start: int,
    end: int,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> bytes:
    if start < 0 or end < start:
        raise Mp4MetadataError(f"invalid MP4 byte range {start}:{end}")
    try:
        with open_binary_uri(path, storage_options=storage_options, auth_ref=auth_ref) as fh:
            fh.seek(start)
            data = fh.read(end - start)
    except StorageConfigError as exc:
        raise Mp4MetadataError(str(exc)) from exc
    if len(data) != end - start:
        raise Mp4MetadataError(f"truncated MP4 byte range {start}:{end}")
    return data


def _fixed_16_16(data: bytes) -> int | None:
    if len(data) != 4:
        return None
    raw = _u32(data, 0)
    value = raw >> 16
    return value or None


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from(">Q", data, offset)[0]
