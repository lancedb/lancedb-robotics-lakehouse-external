"""Codec-aware video encoding and GOP seek tests (backlog 0030)."""

import json
from datetime import UTC, datetime

import pyarrow as pa
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import (
    EVENTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    RUNS_SCHEMA,
    SCENARIOS_SCHEMA,
)

runner = CliRunner()


def _frame_bytes(index: int) -> bytes:
    marker = f"frame-{index:02d}|".encode()
    return marker * 128 + bytes([65 + index]) * 4096


def _video_lake(path, *, frame_count: int = 8):
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-video",
                    "run_kind": "teleop",
                    "source": "synthetic",
                    "source_id": "src-video",
                    "raw_uri": "memory://video",
                    "robot_id": "robot-video",
                    "site_id": "lab-video",
                    "task_id": "inspect bin",
                    "start_time_ns": 0,
                    "end_time_ns": (frame_count - 1) * 1_000_000_000,
                    "duration_ns": (frame_count - 1) * 1_000_000_000,
                    "software_version": "sw-video",
                    "hardware_version": "hw-video",
                    "calibration_version": "cal-video",
                    "model_version": "",
                    "metadata": [],
                    "quality_flags": [],
                    "transform_id": "tfm-source",
                    "created_at": now,
                }
            ],
            schema=RUNS_SCHEMA,
        )
    )

    frames = {index: _frame_bytes(index) for index in range(frame_count)}
    lake.table("observations").add(
        pa.Table.from_pylist(
            [
                {
                    "observation_id": f"obs-video-{index}",
                    "run_id": "run-video",
                    "episode_id": None,
                    "episode_index": None,
                    "frame_index": None,
                    "timestamp_ns": index * 1_000_000_000,
                    "sensor_id": "camera_front",
                    "topic": "/camera/front",
                    "modality": "image",
                    "robot_id": None,
                    "site_id": None,
                    "task_id": None,
                    "software_version": None,
                    "outcome": None,
                    "raw_uri": "memory://video",
                    "raw_channel": "/camera/front",
                    "raw_log_time_ns": index * 1_000_000_000,
                    "raw_sequence": index,
                    "payload_json": None,
                    "payload_blob": frames[index],
                    "message_encoding": "jpeg",
                    "schema_encoding": "jpeg",
                    "decode_status": "decoded",
                    "decode_error": "",
                    "state_vector": [float(index)],
                    "action_vector": [float(index) + 0.5],
                    "caption": f"frame {index}",
                    "quality_flags": [],
                    "transform_id": "tfm-ingest",
                    "created_at": now,
                }
                for index in range(frame_count)
            ],
            schema=OBSERVATIONS_SCHEMA,
        )
    )
    lake.table("events").add(
        pa.Table.from_pylist(
            [
                {
                    "event_id": "evt-start",
                    "run_id": "run-video",
                    "timestamp_ns": 0,
                    "start_time_ns": 0,
                    "end_time_ns": 0,
                    "event_type": "teleop_start",
                    "severity": "",
                    "source": "synthetic",
                    "notes": "",
                    "linked_incident_id": "",
                    "transform_id": "tfm-events",
                    "created_at": now,
                },
                {
                    "event_id": "evt-stop",
                    "run_id": "run-video",
                    "timestamp_ns": (frame_count - 1) * 1_000_000_000,
                    "start_time_ns": (frame_count - 1) * 1_000_000_000,
                    "end_time_ns": (frame_count - 1) * 1_000_000_000,
                    "event_type": "success",
                    "severity": "",
                    "source": "synthetic",
                    "notes": "",
                    "linked_incident_id": "",
                    "transform_id": "tfm-events",
                    "created_at": now,
                },
            ],
            schema=EVENTS_SCHEMA,
        )
    )
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-video",
                    "run_id": "run-video",
                    "start_time_ns": 0,
                    "end_time_ns": (frame_count - 1) * 1_000_000_000,
                    "window_ns": (frame_count - 1) * 1_000_000_000,
                    "is_partial": False,
                    "topics": ["/camera/front"],
                    "observation_ids": [f"obs-video-{index}" for index in range(frame_count)],
                    "observation_count": frame_count,
                    "scenario_type": "teleop",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": ["video"],
                    "summary": "inspect bin",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                }
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )
    lake.episodes.from_markers(
        start_event_types=["teleop_start"],
        stop_event_types=["success"],
    )
    return lake, frames


def _only_video_id(lake: Lake) -> str:
    rows = lake.table("videos").to_arrow().to_pylist()
    assert len(rows) == 1
    return rows[0]["video_id"]


def _only_episode_id(lake: Lake) -> str:
    rows = lake.table("episodes").to_arrow().to_pylist()
    assert len(rows) == 1
    return rows[0]["episode_id"]


def test_encode_writes_video_encoding_row_and_keyframe_map(tmp_path):
    lake, _ = _video_lake(tmp_path / "robot.lance")

    report = lake.video.encode(
        video_id=_only_video_id(lake),
        gop_size=2,
        codec="lrb-gop-zlib",
        resolution="640x480",
        nvdec_compatible=True,
    )

    assert report.encodings_written == 1
    row = lake.table("video_encodings").to_arrow().to_pylist()[0]
    assert row["encoding_id"] == report.encoding_ids[0]
    assert row["codec"] == "lrb-gop-zlib"
    assert row["gop_size"] == 2
    assert row["resolution"] == "640x480"
    assert row["fps"] == 1.0
    assert row["nvdec_compatible"] is True
    assert row["keyframe_map_ref"].startswith("sha256:")
    assert row["source_size_bytes"] > row["encoded_size_bytes"]

    keyframes = json.loads(row["keyframe_map_json"])
    assert [entry["keyframe_frame_index"] for entry in keyframes] == [0, 2, 4, 6]
    assert all(entry["byte_end"] > entry["byte_start"] for entry in keyframes)


def test_seek_returns_correct_frame_from_bounded_gop(tmp_path):
    lake, frames = _video_lake(tmp_path / "robot.lance")
    lake.video.encode(video_id=_only_video_id(lake), gop_size=2)

    decoded = lake.video.seek(_only_episode_id(lake), 5)

    assert decoded.frame == frames[5]
    assert decoded.decoder == "cpu"
    assert decoded.gop_first_frame_index == 4
    assert decoded.bytes_read < decoded.encoded_size_bytes
    assert decoded.bytes_read == decoded.byte_range[1] - decoded.byte_range[0]


def test_gop_size_knob_changes_footprint_and_seek_cost(tmp_path):
    lake, _ = _video_lake(tmp_path / "robot.lance")
    video_id = _only_video_id(lake)

    small_gop = lake.video.encode(video_id=video_id, gop_size=2).encoding_ids[0]
    wide_gop = lake.video.encode(video_id=video_id, gop_size=8).encoding_ids[0]

    rows = {
        row["encoding_id"]: row for row in lake.table("video_encodings").to_arrow().to_pylist()
    }
    assert rows[wide_gop]["encoded_size_bytes"] <= rows[small_gop]["encoded_size_bytes"]

    episode_id = _only_episode_id(lake)
    small_seek = lake.video.seek(episode_id, 5, encoding_id=small_gop)
    wide_seek = lake.video.seek(episode_id, 5, encoding_id=wide_gop)

    assert wide_seek.bytes_read > small_seek.bytes_read


def test_auto_decoder_uses_nvdec_when_available_and_matches_cpu(tmp_path, monkeypatch):
    lake, _ = _video_lake(tmp_path / "robot.lance")
    lake.video.encode(video_id=_only_video_id(lake), gop_size=2, nvdec_compatible=True)
    episode_id = _only_episode_id(lake)

    cpu = lake.video.seek(episode_id, 3, decoder="cpu")
    monkeypatch.setattr("lancedb_robotics.video._nvdec_available", lambda: True)
    nvdec = lake.video.seek(episode_id, 3, decoder="auto")

    assert nvdec.decoder == "nvdec"
    assert nvdec.frame == cpu.frame


def test_video_cli_encodes_and_inspects(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _video_lake(lake_path)

    encoded = runner.invoke(
        app,
        [
            "video",
            "encode",
            "--lake",
            str(lake_path),
            "--gop-size",
            "2",
            "--resolution",
            "640x480",
            "--nvdec-compatible",
        ],
    )

    assert encoded.exit_code == 0
    assert "encodings: 1" in encoded.output
    assert "gop size: 2" in encoded.output

    inspected = runner.invoke(app, ["video", "inspect", "--lake", str(lake_path)])

    assert inspected.exit_code == 0
    assert "codec: lrb-gop-zlib" in inspected.output
    assert "resolution: 640x480" in inspected.output
    assert "keyframes: 4" in inspected.output


def test_training_dataset_reads_video_frame_through_codec_path(tmp_path):
    lake, frames = _video_lake(tmp_path / "robot.lance")
    lake.video.encode(video_id=_only_video_id(lake), gop_size=2)
    create_snapshot(
        lake,
        name="video-demo",
        scenario_ids=["scn-video"],
        split_by="scenario",
    )

    dataset = lake.training.dataset(
        "video-demo",
        columns=["observation_id", "video_frame", "payload_size"],
        media="bytes",
        media_cache="bounded",
        media_cache_size=2,
    )

    assert dataset[4]["observation_id"] == "obs-video-4"
    assert dataset[4]["video_frame"] == frames[4]
    assert dataset[4]["payload_size"] == len(frames[4])
    media = dataset[4]["_media"]["fields"]["video_frame"]
    assert media["kind"] == "codec_video_frame"
    assert media["source_table"] == "video_encodings"
    assert media["byte_range"][1] > media["byte_range"][0]
    assert media["gop_index"] == 2
    assert dataset.manifest.to_dict()["media"]["policy"] == "bytes"
    assert dataset.features["video_frame"]["source"] == "video_encodings.data"
