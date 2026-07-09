"""Replay bundle exporter tests (backlog 0107)."""

import copy
import hashlib
import json
import struct
import zlib
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import pyarrow as pa
import pytest
from mcap.reader import make_reader
from typer.testing import CliRunner

from lancedb_robotics.blob import fetch_blob
from lancedb_robotics.cli import app
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.evidence import EvidencePackError
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.keyframe_maps import keyframe_map_artifact_row
from lancedb_robotics.lake import Lake
from lancedb_robotics.replay import REPLAY_BUNDLE_SCHEMA, build_replay_bundle
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.schemas import (
    EVENTS_SCHEMA,
    KEYFRAME_MAP_ARTIFACTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    RUNS_SCHEMA,
    SCENARIOS_SCHEMA,
    VIDEO_ENCODINGS_SCHEMA,
)
from lancedb_robotics.video import VIDEO_ENCODING_BLOB_COLUMN
from lancedb_robotics.writeback import ingest_model_outputs

runner = CliRunner()

_FRAME_LEN = struct.Struct(">Q")


def _trace_lake(tmp_path, fixtures_dir, fixture_name="sample.mcap"):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / fixture_name)
    create_scenario_windows(lake, window_ns=100_000_000)
    scenarios = sorted(
        lake.table("scenarios").to_arrow().to_pylist(),
        key=lambda row: (row["start_time_ns"], row["scenario_id"]),
    )
    manifest = create_snapshot(
        lake,
        name="demo-v1",
        tag="training-demo",
        scenario_ids=[row["scenario_id"] for row in scenarios],
    )
    ingest_model_outputs(
        lake,
        {
            "model_output_id": "out-regression",
            "observation_id": scenarios[0]["observation_ids"][0],
            "scenario_id": scenarios[0]["scenario_id"],
            "dataset_id": manifest.dataset_id,
            "model_version": "policy@abc123",
            "prediction": "regressed",
            "score": 0.12,
            "producer_run_id": "checkpoint-abc123",
        },
        source="trainer",
    )
    return lake, manifest


def _table_versions_and_counts(lake):
    return {
        name: (int(lake.table(name).version), lake.table(name).count_rows())
        for name in lake.table_names()
    }


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(uri)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# --- MCAP slice bundles -----------------------------------------------------


def test_replay_bundle_mcap_slice_round_trips_source_coordinates(tmp_path, fixtures_dir):
    lake, manifest = _trace_lake(tmp_path, fixtures_dir)
    pack = lake.lineage.trace_checkpoint("checkpoint-abc123").evidence_pack()
    coords = [c for c in pack.manifest["source_coordinates"] if c.get("log_time_ns") is not None]
    assert coords

    report = lake.lineage.replay_bundle(
        "checkpoint-abc123", checkpoint=True, output_dir=str(tmp_path / "bundle")
    )

    assert report.manifest["schema_version"] == REPLAY_BUNDLE_SCHEMA
    assert report.parent_evidence_pack_digest == pack.manifest_digest
    assert report.file_count == len(report.manifest["mcap_slices"]) >= 1
    # Bundle manifest carries the parent digest and the source table-version pins.
    assert report.manifest["parent_evidence_pack"]["manifest_digest"] == pack.manifest_digest
    assert {
        (row["table"], row["version"]) for row in report.manifest["source_table_versions"]
    } >= set(manifest.table_versions)

    # Every emitted object has a byte count and sha256 that match bytes on disk.
    bundle_dir = Path(report.output_dir)
    for file_record in report.files:
        payload = (bundle_dir / file_record["path"]).read_bytes()
        assert file_record["bytes"] == len(payload)
        assert file_record["sha256"] == _sha256(payload)

    # Each referenced source coordinate round-trips into a slice.
    present = set()
    for sl in report.manifest["mcap_slices"]:
        with (bundle_dir / sl["path"]).open("rb") as handle:
            for _schema, channel, message in make_reader(handle).iter_messages():
                present.add((channel.topic, message.log_time))
    for coord in coords:
        assert (coord["channel"], coord["log_time_ns"]) in present


def test_replay_bundle_is_deterministic_and_read_only(tmp_path, fixtures_dir):
    lake, _ = _trace_lake(tmp_path, fixtures_dir)
    before = _table_versions_and_counts(lake)

    first = lake.lineage.replay_bundle(
        "checkpoint-abc123", checkpoint=True, output_dir=str(tmp_path / "a")
    )
    second = lake.lineage.replay_bundle(
        "checkpoint-abc123", checkpoint=True, output_dir=str(tmp_path / "b")
    )

    assert first.bundle_digest == second.bundle_digest
    assert [f["sha256"] for f in first.files] == [f["sha256"] for f in second.files]
    # A manifest.json is written and equals the reported manifest.
    assert json.loads(Path(first.manifest_path).read_text()) == first.manifest
    # Exporting is read-only against the lake.
    assert _table_versions_and_counts(lake) == before


def test_replay_bundle_external_links_are_bundle_relative(tmp_path, fixtures_dir):
    lake, _ = _trace_lake(tmp_path, fixtures_dir)
    report = lake.lineage.replay_bundle(
        "checkpoint-abc123",
        checkpoint=True,
        output_dir=str(tmp_path / "bundle"),
        viewer_formats=("foxglove", "rerun"),
    )
    sl = report.manifest["mcap_slices"][0]
    tools = {link["tool"] for link in sl["external_links"]}
    assert tools == {"foxglove", "rerun"}
    for link in sl["external_links"]:
        # Relative to the bundle root, not an absolute path that would leak tmp_path.
        assert link["target"] == sl["path"]
        assert str(tmp_path) not in link["target"]


# --- Video clip / GOP bundles ----------------------------------------------


def _decode_gop(payload: bytes) -> list[bytes]:
    raw = zlib.decompress(payload)
    frames: list[bytes] = []
    offset = 0
    while offset < len(raw):
        (size,) = _FRAME_LEN.unpack(raw[offset : offset + _FRAME_LEN.size])
        offset += _FRAME_LEN.size
        frames.append(raw[offset : offset + size])
        offset += size
    return frames


def _frame_bytes(index: int) -> bytes:
    return f"frame-{index:02d}|".encode() * 128 + bytes([65 + index]) * 4096


def _video_lake(path, *, frame_count: int = 6):
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
    lake.episodes.from_markers(start_event_types=["teleop_start"], stop_event_types=["success"])
    return lake, frames


def _video_pack_lake(tmp_path, frame_count=6):
    lake, frames = _video_lake(tmp_path / "robot.lance", frame_count=frame_count)
    video_id = lake.table("videos").to_arrow().to_pylist()[0]["video_id"]
    lake.video.encode(video_id=video_id, gop_size=2)
    scenario = lake.table("scenarios").to_arrow().to_pylist()[0]
    manifest = create_snapshot(
        lake, name="vid-v1", tag="vid-tag", scenario_ids=[scenario["scenario_id"]]
    )
    ingest_model_outputs(
        lake,
        {
            "model_output_id": "out-vid",
            "observation_id": scenario["observation_ids"][0],
            "scenario_id": scenario["scenario_id"],
            "dataset_id": manifest.dataset_id,
            "model_version": "m@1",
            "prediction": "x",
            "producer_run_id": "ckpt-vid",
        },
        source="trainer",
    )
    return lake, frames


def _offloaded_video_pack_lake(tmp_path, frame_count=6):
    lake, frames = _video_lake(tmp_path / "robot.lance", frame_count=frame_count)
    video = lake.table("videos").to_arrow().to_pylist()[0]
    lake.video.encode(video_id=video["video_id"], gop_size=2)
    artifact = _offload_keyframe_map(lake, video)
    scenario = lake.table("scenarios").to_arrow().to_pylist()[0]
    manifest = create_snapshot(
        lake, name="vid-v1", tag="vid-tag", scenario_ids=[scenario["scenario_id"]]
    )
    ingest_model_outputs(
        lake,
        {
            "model_output_id": "out-vid",
            "observation_id": scenario["observation_ids"][0],
            "scenario_id": scenario["scenario_id"],
            "dataset_id": manifest.dataset_id,
            "model_version": "m@1",
            "prediction": "x",
            "producer_run_id": "ckpt-vid",
        },
        source="trainer",
    )
    return lake, frames, artifact


def _offload_keyframe_map(lake, video):
    row = lake.table("video_encodings").to_arrow().to_pylist()[0]
    keyframe_json = row["keyframe_map_json"]
    assert keyframe_json
    artifact = keyframe_map_artifact_row(
        keyframe_json,
        source_video_fingerprint="synthetic-video-fingerprint",
        inspection_id="inspection-video",
        source_uri=video.get("raw_uri") or video.get("uri"),
        source_path=None,
        encoding_id=row["encoding_id"],
        video_id=row["video_id"],
        run_id=row.get("run_id"),
        episode_id=row.get("episode_id"),
        episode_index=row.get("episode_index"),
        camera_key=row.get("camera_key"),
        transform_id="tfm-keyframe-map-offload",
        created_at=row["created_at"],
    )
    lake.table("keyframe_map_artifacts").add(
        pa.Table.from_pylist([artifact], schema=KEYFRAME_MAP_ARTIFACTS_SCHEMA)
    )
    offloaded = dict(row)
    offloaded["keyframe_map_json"] = None
    offloaded[VIDEO_ENCODING_BLOB_COLUMN] = fetch_blob(
        lake.table("video_encodings"),
        VIDEO_ENCODING_BLOB_COLUMN,
        row["encoding_id"],
        id_column="encoding_id",
    )
    (
        lake.table("video_encodings")
        .merge_insert("encoding_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(pa.Table.from_pylist([offloaded], schema=VIDEO_ENCODINGS_SCHEMA))
    )
    return artifact


def _corrupt_keyframe_map_artifact_head(lake, artifact):
    corrupt = dict(artifact)
    corrupt["keyframe_map_json"] = "[]"
    (
        lake.table("keyframe_map_artifacts")
        .merge_insert("artifact_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(pa.Table.from_pylist([corrupt], schema=KEYFRAME_MAP_ARTIFACTS_SCHEMA))
    )


def test_replay_bundle_video_gops_reconstruct_and_account_hashes(tmp_path):
    lake, frames = _video_pack_lake(tmp_path)
    pack = lake.lineage.trace_checkpoint("ckpt-vid").evidence_pack()
    assert pack.manifest["video_encoding_refs"]

    report = lake.lineage.replay_bundle(
        "ckpt-vid",
        checkpoint=True,
        output_dir=str(tmp_path / "vb"),
        include_mcap=False,
        include_video=True,
    )
    clip = report.manifest["video_clips"][0]
    bundle_dir = Path(report.output_dir)

    # Hash accounting: every clip/GOP file matches its recorded hash and size.
    for file_record in report.files:
        payload = (bundle_dir / file_record["path"]).read_bytes()
        assert file_record["bytes"] == len(payload)
        assert file_record["sha256"] == _sha256(payload)

    # GOP byte slices concatenate back to the full clip blob...
    ordered = sorted(clip["gops"], key=lambda g: g["gop_index"])
    concat = b"".join((bundle_dir / g["path"]).read_bytes() for g in ordered)
    clip_bytes = (bundle_dir / clip["clip"]["path"]).read_bytes()
    assert concat == clip_bytes

    # ...and decode back to the exact source frames.
    recovered: list[bytes] = []
    for g in ordered:
        recovered.extend(_decode_gop((bundle_dir / g["path"]).read_bytes()))
    assert recovered == [frames[i] for i in range(len(frames))]


def test_replay_bundle_video_gop_bytes_are_deterministic(tmp_path):
    lake, _ = _video_pack_lake(tmp_path)
    first = lake.lineage.replay_bundle(
        "ckpt-vid",
        checkpoint=True,
        output_dir=str(tmp_path / "v1"),
        include_mcap=False,
        include_video=True,
    )
    second = lake.lineage.replay_bundle(
        "ckpt-vid",
        checkpoint=True,
        output_dir=str(tmp_path / "v2"),
        include_mcap=False,
        include_video=True,
    )
    assert first.bundle_digest == second.bundle_digest
    assert [f["sha256"] for f in first.files] == [f["sha256"] for f in second.files]


def test_replay_bundle_resolves_offloaded_keyframe_maps_at_pinned_artifact_version(tmp_path):
    lake, frames, artifact = _offloaded_video_pack_lake(tmp_path)
    pack = lake.lineage.trace_checkpoint("ckpt-vid").evidence_pack()
    versions = {row["table"]: row["version"] for row in pack.manifest["table_versions"]}
    pinned_version = versions["keyframe_map_artifacts"]
    video_ref = pack.manifest["video_encoding_refs"][0]
    assert video_ref["keyframe_map_ref"] == artifact["keyframe_map_ref"]
    assert video_ref["keyframe_map_storage"] == "artifact"
    assert video_ref["keyframe_map_artifact_id"] == artifact["artifact_id"]
    assert video_ref["keyframe_map_artifact_table_version"] == pinned_version

    _corrupt_keyframe_map_artifact_head(lake, artifact)
    latest_version = int(lake.table("keyframe_map_artifacts").version)
    assert latest_version > pinned_version

    report = build_replay_bundle(
        lake,
        pack.manifest,
        output_dir=str(tmp_path / "pinned"),
        include_mcap=False,
        include_video=True,
    )
    clip = report.manifest["video_clips"][0]
    assert clip["keyframe_map_ref"] == artifact["keyframe_map_ref"]
    assert clip["keyframe_map_artifact_table_version"] == pinned_version
    bundle_dir = Path(report.output_dir)
    recovered: list[bytes] = []
    for gop in sorted(clip["gops"], key=lambda item: item["gop_index"]):
        recovered.extend(_decode_gop((bundle_dir / gop["path"]).read_bytes()))
    assert recovered == [frames[i] for i in range(len(frames))]

    bad_manifest = copy.deepcopy(pack.manifest)
    for row in bad_manifest["table_versions"]:
        if row["table"] == "keyframe_map_artifacts":
            row["version"] = latest_version
    for ref in bad_manifest["video_encoding_refs"]:
        ref["keyframe_map_artifact_table_version"] = latest_version
    with pytest.raises(EvidencePackError, match="content mismatch"):
        build_replay_bundle(
            lake,
            bad_manifest,
            output_dir=str(tmp_path / "latest"),
            include_mcap=False,
            include_video=True,
        )


def test_replay_bundle_can_skip_per_gop_files(tmp_path):
    lake, _ = _video_pack_lake(tmp_path)
    report = lake.lineage.replay_bundle(
        "ckpt-vid",
        checkpoint=True,
        output_dir=str(tmp_path / "vb"),
        include_mcap=False,
        include_video=True,
        include_gops=False,
    )
    clip = report.manifest["video_clips"][0]
    assert clip["gops"] == []
    assert {f["kind"] for f in report.files} == {"video-clip"}


# --- Limits and error paths -------------------------------------------------


def test_replay_bundle_enforces_max_files_without_partial_output(tmp_path, fixtures_dir):
    lake, _ = _trace_lake(tmp_path, fixtures_dir)
    out = tmp_path / "bounded"
    with pytest.raises(EvidencePackError, match="max_files"):
        lake.lineage.replay_bundle(
            "checkpoint-abc123", checkpoint=True, output_dir=str(out), max_files=0
        )
    assert not out.exists()


def test_replay_bundle_enforces_max_bytes_without_partial_output(tmp_path):
    lake, _ = _video_pack_lake(tmp_path)
    out = tmp_path / "bounded"
    with pytest.raises(EvidencePackError, match="max_bytes"):
        lake.lineage.replay_bundle(
            "ckpt-vid",
            checkpoint=True,
            output_dir=str(out),
            include_mcap=False,
            include_video=True,
            max_bytes=1,
        )
    assert not out.exists()


def test_replay_bundle_missing_source_bytes_fails(tmp_path, fixtures_dir):
    lake, _ = _trace_lake(tmp_path, fixtures_dir)
    pack = lake.lineage.trace_checkpoint("checkpoint-abc123").evidence_pack()
    manifest = dict(pack.manifest)
    manifest["source_coordinates"] = [
        {
            "uri": str(tmp_path / "does-not-exist.mcap"),
            "channel": "/imu",
            "offset": 0,
            "log_time_ns": 1,
            "observation_ids": [],
        }
    ]
    manifest["payload_refs"] = []
    with pytest.raises(EvidencePackError, match="missing source bytes"):
        build_replay_bundle(lake, manifest, output_dir=str(tmp_path / "bundle"))


def test_replay_bundle_rejects_non_evidence_pack_manifest(tmp_path, fixtures_dir):
    lake, _ = _trace_lake(tmp_path, fixtures_dir)
    with pytest.raises(EvidencePackError, match="unsupported evidence-pack schema"):
        build_replay_bundle(lake, {"schema_version": "not-a-pack"}, output_dir=str(tmp_path / "b"))


def test_replay_bundle_requires_at_least_one_exporter(tmp_path, fixtures_dir):
    lake, _ = _trace_lake(tmp_path, fixtures_dir)
    pack = lake.lineage.trace_checkpoint("checkpoint-abc123").evidence_pack()
    with pytest.raises(EvidencePackError, match="nothing to export"):
        build_replay_bundle(
            lake,
            pack.manifest,
            output_dir=str(tmp_path / "b"),
            include_mcap=False,
            include_video=False,
        )


# --- CLI --------------------------------------------------------------------


def test_cli_export_replay_writes_bundle(tmp_path, fixtures_dir):
    lake, _ = _trace_lake(tmp_path, fixtures_dir)
    out = tmp_path / "bundle"
    result = runner.invoke(
        app,
        [
            "lineage",
            "export-replay",
            "checkpoint-abc123",
            "--lake",
            str(lake.uri),
            "--output-dir",
            str(out),
            "--checkpoint",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["manifest"]["schema_version"] == REPLAY_BUNDLE_SCHEMA
    assert payload["file_count"] >= 1
    assert (out / "manifest.json").exists()


def test_cli_export_replay_requires_an_exporter(tmp_path, fixtures_dir):
    lake, _ = _trace_lake(tmp_path, fixtures_dir)
    result = runner.invoke(
        app,
        [
            "lineage",
            "export-replay",
            "checkpoint-abc123",
            "--lake",
            str(lake.uri),
            "--output-dir",
            str(tmp_path / "bundle"),
            "--checkpoint",
            "--no-mcap",
        ],
    )
    assert result.exit_code == 2
    assert "at least one" in result.output
