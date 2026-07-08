"""Real LeRobot/source-MP4 decoded-frame conformance lane (backlog 0366)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest
from conftest import require_video_decode
from test_lerobot_adapter import (
    _lerobot_fixture,
    _lerobot_ingest_params,
    _require_decoded_frame_conformance,
    _write_json,
    _write_parquet,
)

from lancedb_robotics._mp4 import inspect_mp4_video
from lancedb_robotics.ingest import ingest_lerobot
from lancedb_robotics.lake import Lake
from lancedb_robotics.video import _decode_source_mp4_frame_pyav

_REQUIRE_VIDEO_DECODE_ENV = "LANCEDB_ROBOTICS_REQUIRE_VIDEO_DECODE"
_ARTIFACT_DIR_ENV = "LANCEDB_ROBOTICS_VIDEO_DECODE_ARTIFACT_DIR"
_FRAME_SIZE = 16
_BASELINE_CODEC = "mpeg4"
_BASELINE_CODEC_NAME = "mp4v"
_OPTIONAL_ACCELERATED_CODECS = (
    ("libx264", "h264"),
    ("libx265", "h265"),
    ("libaom-av1", "av1"),
)


def _skip_or_fail_required(reason: str) -> None:
    if os.environ.get(_REQUIRE_VIDEO_DECODE_ENV) == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _require_encoder(av: ModuleType, codec: str) -> None:
    try:
        av.codec.Codec(codec, "w")
    except Exception as exc:  # noqa: BLE001 - PyAV reports codec availability through native errors.
        _skip_or_fail_required(
            f"PyAV encoder {codec!r} is unavailable; cannot create decode fixture "
            f"({type(exc).__name__}: {exc})"
        )


def _skip_if_optional_encoder_missing(av: ModuleType, codec: str) -> None:
    try:
        av.codec.Codec(codec, "w")
    except Exception as exc:  # noqa: BLE001 - optional host codec matrix.
        pytest.skip(
            f"optional PyAV encoder {codec!r} is unavailable on this host "
            f"({type(exc).__name__}: {exc})"
        )


def _write_pyav_mp4(
    path: Path,
    *,
    av: ModuleType,
    np: ModuleType,
    codec: str,
    colors: tuple[tuple[int, int, int], ...],
    gop_size: int | None = None,
    options: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    try:
        stream = container.add_stream(codec, rate=10)
        stream.width = _FRAME_SIZE
        stream.height = _FRAME_SIZE
        stream.pix_fmt = "yuv420p"
        if gop_size is not None:
            stream.gop_size = int(gop_size)
        if options:
            stream.options = dict(options)
        for frame_index, color in enumerate(colors):
            pixels = np.zeros((_FRAME_SIZE, _FRAME_SIZE, 3), dtype=np.uint8)
            pixels[:, :] = color
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = frame_index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception as exc:  # noqa: BLE001 - encoder failures are native and host-specific.
        _skip_or_fail_required(
            f"PyAV encoder {codec!r} failed while creating decode fixture "
            f"({type(exc).__name__}: {exc})"
        )
    finally:
        container.close()


def _decoded_rgb_hashes(path: Path, *, av: ModuleType) -> tuple[list[str], list[list[int]]]:
    container = av.open(str(path))
    try:
        hashes: list[str] = []
        shapes: list[list[int]] = []
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            shapes.append([int(value) for value in rgb.shape])
        return hashes, shapes
    finally:
        container.close()


def _write_artifact(name: str, payload: dict) -> None:
    artifact_dir = os.environ.get(_ARTIFACT_DIR_ENV)
    if not artifact_dir:
        return
    path = Path(artifact_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _single_episode_lerobot_fixture(root: Path, *, frame_count: int) -> Path:
    source = _lerobot_fixture(root, version="v3.0")
    info = json.loads((source / "meta/info.json").read_text())
    info["total_episodes"] = 1
    info["total_frames"] = int(frame_count)
    _write_json(source / "meta/info.json", info)
    _write_parquet(
        source / "meta/episodes/chunk-000/file-000.parquet",
        [
            {
                "episode_index": 0,
                "episode_id": "lerobot-ep-0",
                "scenario_id": "source-scn-0",
                "tasks": ["pick cube"],
                "length": int(frame_count),
                "dataset_from_index": 0,
                "dataset_to_index": int(frame_count),
                "split": "train",
            }
        ],
    )
    rows = [
        {
            "index": index,
            "episode_index": 0,
            "frame_index": index,
            "timestamp": float(index) / 10.0,
            "task_index": 0,
            "task": "pick cube",
            "observation.state": [float(index), float(index) + 0.5],
            "action": [float(index) / 10.0, float(index) / 10.0 + 0.05],
            "observation.images.front": (
                "videos/chunk-000/observation.images.front/episode_000000.mp4"
            ),
            "vendor.force": float(index),
        }
        for index in range(frame_count)
    ]
    _write_parquet(source / "data/chunk-000/file-000.parquet", rows)
    extra_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    if extra_video.exists():
        extra_video.unlink()
    return source


def test_require_video_decode_converts_unexpected_skip_to_failure(monkeypatch):
    """The video-decode CI lane must fail, not pass with all tests skipped."""

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name in {"av", "numpy"}:
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.delenv(_REQUIRE_VIDEO_DECODE_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_video_decode()

    monkeypatch.setenv(_REQUIRE_VIDEO_DECODE_ENV, "1")
    with pytest.raises(pytest.fail.Exception):
        require_video_decode()


@pytest.mark.video_decode
def test_lerobot_ingest_and_source_conformance_decode_real_mp4_pixels(tmp_path):
    av, np = require_video_decode()
    _require_encoder(av, _BASELINE_CODEC)
    source = _lerobot_fixture(tmp_path / "lerobot-real-video-decode", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_pyav_mp4(
        first_video,
        av=av,
        np=np,
        codec=_BASELINE_CODEC,
        colors=((255, 0, 0), (0, 255, 0)),
    )
    _write_pyav_mp4(
        second_video,
        av=av,
        np=np,
        codec=_BASELINE_CODEC,
        colors=((0, 0, 255),),
    )
    expected_hashes, expected_shapes = _decoded_rgb_hashes(first_video, av=av)
    assert expected_shapes == [[_FRAME_SIZE, _FRAME_SIZE, 3], [_FRAME_SIZE, _FRAME_SIZE, 3]]

    lake = Lake.init(tmp_path / "lake")
    report = ingest_lerobot(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
        decoded_frame_conformance={
            "enabled": True,
            "backend": "pyav",
            "samples": [
                {
                    "camera_key": "front",
                    "episode_index": 0,
                    "frame_index": 1,
                    "expected_pixel_sha256": expected_hashes[1],
                }
            ],
        },
    )

    assert report.rows_added["observations"] == 3
    ingest_conformance = _require_decoded_frame_conformance(_lerobot_ingest_params(lake))
    _write_artifact("lerobot-ingest-decoded-frame-conformance.json", ingest_conformance)
    assert ingest_conformance["status"] == "passed"
    assert ingest_conformance["frames_checked"] == 1
    assert ingest_conformance["backend"]["name"] == "pyav"
    assert ingest_conformance["backend"]["available"] is True
    assert ingest_conformance["codec_coverage"] == {
        _BASELINE_CODEC_NAME: {"supported": True, "frames_checked": 1, "failures": 0}
    }
    check = ingest_conformance["checks"][0]
    assert check["codec"] == _BASELINE_CODEC_NAME
    assert check["shape"] == [_FRAME_SIZE, _FRAME_SIZE, 3]
    assert check["dtype"] == "uint8"
    assert check["pixel_sha256"] == expected_hashes[1]
    assert check["expected_pixel_sha256"] == expected_hashes[1]
    assert check["decoded_frame_count"] >= 1
    assert isinstance(check["seek_frame_index"], int)

    source_conformance = lake.video.conform_source(
        [
            {
                "camera_key": "front",
                "episode_index": 0,
                "frame_index": 1,
                "expected_sha256": expected_hashes[1],
            }
        ],
        decoder="pyav",
    )
    _write_artifact("lake-video-source-conformance.json", source_conformance.to_dict())

    assert source_conformance.status == "passed"
    assert source_conformance.status_counts == {"passed": 1}
    assert source_conformance.codec_counts == {_BASELINE_CODEC_NAME: 1}
    assert source_conformance.backend_versions.get("pyav")
    result = source_conformance.results[0]
    assert result["decoded_sha256"] == expected_hashes[1]
    assert result["decoder_backend"] == "pyav"
    assert result["frames_decoded"] >= 1
    assert result["byte_range"][0] < result["byte_range"][1]


@pytest.mark.video_decode
def test_lerobot_pyav_conformance_seeks_from_nearest_keyframe(tmp_path):
    av, np = require_video_decode()
    _require_encoder(av, "libx264")
    source = _single_episode_lerobot_fixture(
        tmp_path / "lerobot-nearest-keyframe-decode",
        frame_count=8,
    )
    video_path = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    colors = tuple(
        ((index * 30) % 255, (index * 50) % 255, (index * 70) % 255)
        for index in range(8)
    )
    _write_pyav_mp4(
        video_path,
        av=av,
        np=np,
        codec="libx264",
        colors=colors,
        gop_size=3,
        options={
            "g": "3",
            "keyint_min": "3",
            "sc_threshold": "0",
            "bf": "0",
            "preset": "ultrafast",
            "tune": "zerolatency",
        },
    )
    expected_hashes, _shapes = _decoded_rgb_hashes(video_path, av=av)

    metadata = inspect_mp4_video(video_path)
    assert metadata.codec == "h264"
    assert [(entry["keyframe_frame_index"], entry["last_frame_index"]) for entry in metadata.keyframe_map] == [
        (0, 2),
        (3, 5),
        (6, 7),
    ]
    target_entry = metadata.keyframe_map[1]
    assert target_entry["keyframe_time_units"] == target_entry["frames"][0]["sample_time_units"]
    assert target_entry["frames"][2]["frame_index"] == 5
    assert target_entry["frames"][2]["sample_time_units"] > target_entry["keyframe_time_units"]

    lake = Lake.init(tmp_path / "lake")
    ingest_lerobot(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
        decoded_frame_conformance={
            "enabled": True,
            "backend": "pyav",
            "samples": [
                {
                    "camera_key": "front",
                    "episode_index": 0,
                    "frame_index": 5,
                    "expected_pixel_sha256": expected_hashes[5],
                }
            ],
        },
    )

    ingest_conformance = _require_decoded_frame_conformance(_lerobot_ingest_params(lake))
    check = ingest_conformance["checks"][0]
    assert ingest_conformance["status"] == "passed"
    assert check["seek_strategy"] == "nearest-keyframe"
    assert check["seek_frame_index"] == 3
    assert check["decoded_frame_count"] == 3
    assert check["fallback_reason"] is None

    source_conformance = lake.video.conform_source(
        [
            {
                "camera_key": "front",
                "episode_index": 0,
                "frame_index": 5,
                "expected_sha256": expected_hashes[5],
            }
        ],
        decoder="pyav",
    )
    result = source_conformance.results[0]
    assert source_conformance.status == "passed"
    assert result["seek_strategy"] == "nearest-keyframe"
    assert result["seek_frame_index"] == 3
    assert result["frames_decoded"] == 3
    assert result["frames_decoded"] < 6
    assert result["fallback_reason"] is None


@pytest.mark.video_decode
def test_pyav_decode_reports_sequential_fallback_without_seek_timestamps(tmp_path):
    av, np = require_video_decode()
    _require_encoder(av, _BASELINE_CODEC)
    path = tmp_path / "fallback.mp4"
    _write_pyav_mp4(
        path,
        av=av,
        np=np,
        codec=_BASELINE_CODEC,
        colors=((255, 0, 0), (0, 255, 0)),
    )
    expected_hashes, _shapes = _decoded_rgb_hashes(path, av=av)
    metadata = inspect_mp4_video(path)
    entry = dict(metadata.keyframe_map[1])
    entry.pop("keyframe_time_units", None)
    frame_entry = dict(entry["frames"][0])
    frame_entry.pop("sample_time_units", None)

    decoded = _decode_source_mp4_frame_pyav(path.as_posix(), 1, entry=entry, frame_entry=frame_entry)

    assert hashlib.sha256(decoded.frame).hexdigest() == expected_hashes[1]
    assert decoded.seek_strategy == "sequential-fallback"
    assert decoded.seek_frame_index == 0
    assert decoded.frames_decoded == 2
    assert decoded.fallback_reason == "missing-keyframe-time-units"


@pytest.mark.video_decode
@pytest.mark.parametrize(("encoder", "expected_codec"), _OPTIONAL_ACCELERATED_CODECS)
def test_optional_accelerated_mp4_codecs_decode_when_pyav_encoder_exists(
    tmp_path,
    encoder,
    expected_codec,
):
    av, np = require_video_decode()
    _skip_if_optional_encoder_missing(av, encoder)
    path = tmp_path / f"{expected_codec}.mp4"
    _write_pyav_mp4(
        path,
        av=av,
        np=np,
        codec=encoder,
        colors=((32, 64, 96), (96, 64, 32)),
    )

    metadata = inspect_mp4_video(path)
    hashes, shapes = _decoded_rgb_hashes(path, av=av)
    _write_artifact(
        f"optional-codec-{expected_codec}.json",
        {
            "encoder": encoder,
            "codec": metadata.codec,
            "codec_tag": metadata.codec_tag,
            "codec_profile": metadata.codec_profile,
            "frame_count": metadata.frame_count,
            "gop_size": metadata.gop_size,
            "decoded_pixel_sha256": hashes,
            "decoded_shapes": shapes,
        },
    )

    assert metadata.codec == expected_codec
    assert metadata.frame_count == 2
    assert len(hashes) == 2
    assert shapes == [[_FRAME_SIZE, _FRAME_SIZE, 3], [_FRAME_SIZE, _FRAME_SIZE, 3]]
