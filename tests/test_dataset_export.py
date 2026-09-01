"""LeRobot/RLDS dataset export tests (backlog 0026)."""

import importlib.util
import json
import os
import tarfile
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.dataset_export import (
    DATASET_EXPORT_MANIFEST_FILENAME,
    DatasetExportError,
    export_dataset_snapshot,
    native_loader_status,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA, SCENARIOS_SCHEMA

_REQUIRE_RLDS_NATIVE = os.getenv("LANCEDB_ROBOTICS_REQUIRE_RLDS_NATIVE") == "1"


def _require_rlds_native_stack() -> None:
    status = native_loader_status("rlds")
    if status["available"]:
        return

    missing = ", ".join(status["missing"])
    reason = (
        "RLDS optional dependency is not installed or unsupported on this "
        f"platform; missing {missing}; install {status['install']}"
    )
    if _REQUIRE_RLDS_NATIVE:
        pytest.fail(reason)
    pytest.skip(reason)


def _exportable_lake(path):
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-export",
                    "run_kind": "demo",
                    "source": "synthetic",
                    "source_id": "src-export",
                    "raw_uri": "memory://run-export",
                    "robot_id": "robot-1",
                    "site_id": "lab",
                    "task_id": "pick the cube",
                    "start_time_ns": 1_000,
                    "end_time_ns": 2_000,
                    "duration_ns": 1_000,
                    "software_version": "test",
                    "hardware_version": "test",
                    "calibration_version": "test",
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
    lake.table("observations").add(
        pa.Table.from_pylist(
            [
                {
                    "observation_id": "obs-camera-0",
                    "run_id": "run-export",
                    "timestamp_ns": 1_000,
                    "sensor_id": "camera_front",
                    "topic": "/camera/front",
                    "modality": "image",
                    "raw_uri": "memory://run-export",
                    "raw_channel": "/camera/front",
                    "raw_log_time_ns": 1_000,
                    "raw_sequence": 0,
                    "payload_json": None,
                    "payload_blob": b"frame-0",
                    "message_encoding": "jpeg",
                    "schema_encoding": "jpeg",
                    "decode_status": "decoded",
                    "decode_error": "",
                    "state_vector": [1.0, 2.0],
                    "action_vector": [0.25, -0.5],
                    "caption": "reach toward the cube",
                    "quality_flags": [],
                    "transform_id": "tfm-ingest",
                    "created_at": now,
                },
                {
                    "observation_id": "obs-state-1",
                    "run_id": "run-export",
                    "timestamp_ns": 2_000,
                    "sensor_id": "joint_state",
                    "topic": "/joint_states",
                    "modality": "state",
                    "raw_uri": "memory://run-export",
                    "raw_channel": "/joint_states",
                    "raw_log_time_ns": 2_000,
                    "raw_sequence": 1,
                    "payload_json": "{\"joint\":1}",
                    "payload_blob": None,
                    "message_encoding": "json",
                    "schema_encoding": "json",
                    "decode_status": "decoded",
                    "decode_error": "",
                    "state_vector": [1.5, 2.5],
                    "action_vector": [0.5, -0.25],
                    "caption": "close the gripper",
                    "quality_flags": [],
                    "transform_id": "tfm-ingest",
                    "created_at": now,
                },
            ],
            schema=OBSERVATIONS_SCHEMA,
        )
    )
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-export-a",
                    "run_id": "run-export",
                    "start_time_ns": 1_000,
                    "end_time_ns": 1_000,
                    "window_ns": 0,
                    "is_partial": False,
                    "topics": ["/camera/front"],
                    "observation_ids": ["obs-camera-0"],
                    "observation_count": 1,
                    "scenario_type": "demo",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": ["camera"],
                    "summary": "pick the cube",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                },
                {
                    "scenario_id": "scn-export-b",
                    "run_id": "run-export",
                    "start_time_ns": 2_000,
                    "end_time_ns": 2_000,
                    "window_ns": 0,
                    "is_partial": False,
                    "topics": ["/joint_states"],
                    "observation_ids": ["obs-state-1"],
                    "observation_count": 1,
                    "scenario_type": "demo",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": ["state"],
                    "summary": "pick the cube",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                },
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )
    create_snapshot(
        lake,
        name="demo-v1",
        scenario_ids=["scn-export-a", "scn-export-b"],
        split_by="scenario",
    )
    return lake


@pytest.fixture
def lake(tmp_path):
    return _exportable_lake(tmp_path / "robot.lance")


def test_lerobot_export_writes_episode_parquet_metadata_and_camera_blob(lake, tmp_path):
    out = tmp_path / "lerobot"
    manifest = export_dataset_snapshot(lake, "demo-v1", out_dir=out, fmt="lerobot")

    assert manifest.format == "lerobot"
    assert manifest.format_version == "lerobot-v3.0"
    assert manifest.episode_count == 2
    assert manifest.step_count == 2
    assert len(manifest.content_hash) == 64

    written = json.loads((out / DATASET_EXPORT_MANIFEST_FILENAME).read_text())
    assert written["dataset_id"] == manifest.dataset_id
    assert written["table_versions"]
    assert written["feature_spec"]["features"]["cameras"][0]["key"] == "camera_front"

    info = json.loads((out / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["data_path"] == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    assert info["features"]["observation.state"]["shape"] == [2]
    assert info["features"]["observation.images.camera_front"]["dtype"] == "image"
    assert (out / "meta/tasks.parquet").is_file()
    episode_rows = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert episode_rows[0]["dataset_from_index"] == 0
    assert episode_rows[0]["dataset_to_index"] == 1
    # This fixture's observation carries a synthetic `payload_blob` with a
    # fake `raw_uri` ("memory://run-export") that can't actually be
    # re-opened, so the exporter correctly can't resolve a real image for it
    # (see test_lerobot_export_resolves_real_camera_bytes_from_mcap_source for
    # the real-source path) — no file is written rather than one holding the
    # raw, undecoded payload_blob bytes.
    assert not (out / "images/camera_front/episode_000000/frame_000000.bin").exists()

    rows = pq.read_table(out / "data/chunk-000/file-000.parquet").to_pylist()
    assert rows[0]["observation.state"] == pytest.approx([1.0, 2.0])
    assert rows[0]["action"] == pytest.approx([0.25, -0.5])
    assert rows[0]["language_instruction"] == "reach toward the cube"
    assert rows[0]["observation.images.camera_front"] is None


def _write_camera_mcap(path, frames: list[bytes]) -> None:
    """A minimal real MCAP with one cbor `/camera/front` message per entry in
    ``frames``. cbor is self-describing (no schema definition needed to
    decode, unlike ros1/ros2/protobuf), so this is a real, exercisable decode
    without needing a full ROS message-definition fixture. Each frame is made
    large enough to cross ``DEFAULT_BLOB_THRESHOLD`` so the decoder actually
    hoists it as a blob field rather than base64-inlining it, matching how a
    real compressed-image payload behaves.
    """
    import cbor2
    from mcap.writer import CompressionType, Writer

    with open(path, "wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start(profile="", library="lancedb-robotics-test")
        schema_id = writer.register_schema(
            name="test.CompressedImage",
            encoding="cbor",
            data=cbor2.dumps({"fields": ["format", "data"]}),
        )
        channel_id = writer.register_channel(
            topic="/camera/front", message_encoding="cbor", schema_id=schema_id
        )
        for i, frame in enumerate(frames):
            writer.add_message(
                channel_id=channel_id,
                log_time=1_000_000_000 + i * 100_000_000,
                publish_time=1_000_000_000 + i * 100_000_000,
                data=cbor2.dumps({"format": "jpeg", "data": frame}),
            )
        writer.finish()


def test_decode_camera_frames_from_source_reads_real_image_bytes(tmp_path):
    from lancedb_robotics.dataset_export import _decode_camera_frames_from_source

    frame_0 = b"\xff\xd8\xff" + b"\x00" * 3000
    frame_1 = b"\xff\xd8\xff" + b"\x01" * 3000
    mcap_path = tmp_path / "camera.mcap"
    _write_camera_mcap(mcap_path, [frame_0, frame_1])

    resolved = _decode_camera_frames_from_source(
        str(mcap_path),
        {("/camera/front", 0): "obs-a", ("/camera/front", 1): "obs-b"},
        storage_options=None,
        auth_ref=None,
    )

    assert resolved == {"obs-a": frame_0, "obs-b": frame_1}


def test_decode_camera_frames_from_source_unresolvable_source_returns_empty(tmp_path):
    from lancedb_robotics.dataset_export import _decode_camera_frames_from_source

    resolved = _decode_camera_frames_from_source(
        str(tmp_path / "does-not-exist.mcap"),
        {("/camera/front", 0): "obs-a"},
        storage_options=None,
        auth_ref=None,
    )

    assert resolved == {}


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="LeRobot optional dependency is not installed",
)
def test_lerobot_export_metadata_loads_with_official_lerobot_package(lake, tmp_path):
    out = tmp_path / "lerobot"
    manifest = export_dataset_snapshot(lake, "demo-v1", out_dir=out, fmt="lerobot")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id="local/demo-v1",
        root=out,
        download_videos=False,
    )

    assert dataset.num_frames == manifest.step_count
    assert dataset.num_episodes == manifest.episode_count
    assert "observation.state" in dataset.features


def _lake_with_action_vectors(path, action_vectors: list[list[float] | None]) -> Lake:
    """A minimal exportable lake with one run/scenario and one state-only
    observation per entry in ``action_vectors`` (state_vector stays constant so
    only the 'action' feature's shape behavior is under test)."""
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    observation_ids = [f"obs-{i}" for i in range(len(action_vectors))]
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-export",
                    "run_kind": "demo",
                    "source": "synthetic",
                    "source_id": "src-export",
                    "raw_uri": "memory://run-export",
                    "robot_id": "robot-1",
                    "site_id": "lab",
                    "task_id": "pick the cube",
                    "start_time_ns": 1_000,
                    "end_time_ns": 1_000 + len(action_vectors),
                    "duration_ns": len(action_vectors),
                    "software_version": "test",
                    "hardware_version": "test",
                    "calibration_version": "test",
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
    lake.table("observations").add(
        pa.Table.from_pylist(
            [
                {
                    "observation_id": observation_id,
                    "run_id": "run-export",
                    "timestamp_ns": 1_000 + i,
                    "sensor_id": "joint_state",
                    "topic": "/joint_states",
                    "modality": "state",
                    "raw_uri": "memory://run-export",
                    "raw_channel": "/joint_states",
                    "raw_log_time_ns": 1_000 + i,
                    "raw_sequence": i,
                    "payload_json": "{\"joint\":1}",
                    "payload_blob": None,
                    "message_encoding": "json",
                    "schema_encoding": "json",
                    "decode_status": "decoded",
                    "decode_error": "",
                    "state_vector": [1.0, 2.0],
                    "action_vector": action_vector,
                    "caption": "close the gripper",
                    "quality_flags": [],
                    "transform_id": "tfm-ingest",
                    "created_at": now,
                }
                for i, (observation_id, action_vector) in enumerate(
                    zip(observation_ids, action_vectors, strict=True)
                )
            ],
            schema=OBSERVATIONS_SCHEMA,
        )
    )
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-export-a",
                    "run_id": "run-export",
                    "start_time_ns": 1_000,
                    "end_time_ns": 1_000 + len(action_vectors),
                    "window_ns": 0,
                    "is_partial": False,
                    "topics": ["/joint_states"],
                    "observation_ids": observation_ids,
                    "observation_count": len(observation_ids),
                    "scenario_type": "demo",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": ["state"],
                    "summary": "pick the cube",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                }
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )
    create_snapshot(lake, name="demo-v1", scenario_ids=["scn-export-a"], split_by="scenario")
    return lake


def test_lerobot_export_omits_feature_with_no_data(tmp_path):
    lake = _lake_with_action_vectors(tmp_path / "robot.lance", [None, None])
    out = tmp_path / "lerobot"
    export_dataset_snapshot(lake, "demo-v1", out_dir=out, fmt="lerobot")

    info = json.loads((out / "meta/info.json").read_text())
    assert "action" not in info["features"]
    assert "observation.state" in info["features"]

    # The declared schema must match the physical parquet columns exactly, or
    # a real LeRobot/`datasets` load fails with a column-mismatch CastError
    # even though the export itself "succeeded" (regression: the omission
    # above used to apply only to meta/info.json, not to the parquet writer).
    columns = set(pq.read_schema(out / "data/chunk-000/file-000.parquet").names)
    assert "action" not in columns
    assert "observation.state" in columns


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="LeRobot optional dependency is not installed",
)
def test_lerobot_export_with_no_action_data_loads_with_official_lerobot_package(tmp_path):
    lake = _lake_with_action_vectors(tmp_path / "robot.lance", [None, None])
    out = tmp_path / "lerobot"
    manifest = export_dataset_snapshot(lake, "demo-v1", out_dir=out, fmt="lerobot")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="local/demo-v1", root=out, download_videos=False)

    assert dataset.num_frames == manifest.step_count
    assert "action" not in dataset.features
    assert "observation.state" in dataset.features


def test_lerobot_export_rejects_inconsistent_vector_lengths(tmp_path):
    lake = _lake_with_action_vectors(tmp_path / "robot.lance", [[0.1, 0.2], [0.1, 0.2, 0.3]])
    out = tmp_path / "lerobot"

    with pytest.raises(DatasetExportError, match="inconsistent vector lengths"):
        export_dataset_snapshot(lake, "demo-v1", out_dir=out, fmt="lerobot")


def test_lerobot_export_rejects_partial_vector_coverage(tmp_path):
    lake = _lake_with_action_vectors(tmp_path / "robot.lance", [[0.1, 0.2], None])
    out = tmp_path / "lerobot"

    with pytest.raises(DatasetExportError, match="missing this feature"):
        export_dataset_snapshot(lake, "demo-v1", out_dir=out, fmt="lerobot")


def test_rlds_export_writes_episode_step_structure(lake, tmp_path):
    out = tmp_path / "rlds"
    manifest = export_dataset_snapshot(lake, "demo-v1", out_dir=out, fmt="rlds")

    assert manifest.format == "rlds"
    assert manifest.format_version == "rlds-tfds-style-v0"
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["features"]["step"]["count"] == 2
    assert info["features"]["rlds"]["steps_key"] == "steps"
    assert info["features"]["reward"]["value"] == 0.0
    assert "RLDS reward is synthesized" in " ".join(info["lossy_mapping"])

    rows = pq.read_table(out / "episodes/episode_000000/steps.parquet").to_pylist()
    assert rows[0]["episode_id"] == "scn-export-a"
    assert rows[0]["is_first"] is True
    assert rows[0]["is_last"] is True
    assert rows[0]["is_terminal"] is True
    assert rows[0]["observation.state"] == pytest.approx([1.0, 2.0])
    assert rows[0]["action"] == pytest.approx([0.25, -0.5])
    assert rows[0]["relative_time_s"] == pytest.approx(0.0)
    assert rows[0]["split"] == "train"
    assert rows[0]["task"] == "pick the cube"
    assert rows[0]["observation.sensor_id"] == "camera_front"
    assert rows[0]["lineage.raw_uri"] == "memory://run-export"
    assert json.loads(rows[0]["lineage.table_versions_json"])
    assert (
        rows[0]["observation.image.camera_front"]
        == "images/camera_front/episode_000000/frame_000000.bin"
    )


@pytest.mark.rlds_native
def test_rlds_native_loader_stack_imports_when_requested():
    _require_rlds_native_stack()
    import reverb
    import rlds
    import tensorflow
    import tensorflow_datasets

    status = native_loader_status("rlds")
    assert status["available"] is True
    assert status["missing"] == []
    assert set(status["modules"]) == {
        "rlds",
        "tensorflow",
        "tensorflow_datasets",
        "reverb",
    }
    assert rlds
    assert reverb
    assert tensorflow
    assert tensorflow_datasets


def test_webdataset_export_writes_deterministic_tar_shards(lake, tmp_path, monkeypatch):
    import lancedb_robotics.dataset_export as dataset_export

    monkeypatch.setattr(dataset_export, "_module_available", lambda module: False)

    out = tmp_path / "webdataset"
    manifest = export_dataset_snapshot(
        lake,
        "demo-v1",
        out_dir=out,
        fmt="webdataset",
        shard_size=1,
    )

    assert manifest.format == "webdataset"
    assert manifest.format_version == "webdataset-tar-v0"
    assert manifest.episode_count == 2
    assert manifest.step_count == 2
    assert len(manifest.content_hash) == 64
    assert "shards/shard-000000.tar" in manifest.data_files
    assert "shards/shard-000001.tar" in manifest.data_files

    written = json.loads((out / DATASET_EXPORT_MANIFEST_FILENAME).read_text())
    assert written["feature_spec"]["dry_run"]["shard_count"] == 2
    assert written["feature_spec"]["dry_run"]["estimated_bytes"] > 0
    assert written["native_loader"]["missing"] == ["webdataset"]

    with tarfile.open(out / "shards/shard-000000.tar") as tar:
        names = sorted(tar.getnames())
        assert names == [
            "episode-000000/frame-000000-obs-camera-0.jpg",
            "episode-000000/frame-000000-obs-camera-0.json",
            "episode-000000/frame-000000-obs-camera-0.txt",
        ]
        metadata = json.loads(
            tar.extractfile("episode-000000/frame-000000-obs-camera-0.json")
            .read()
            .decode()
        )
        media = tar.extractfile("episode-000000/frame-000000-obs-camera-0.jpg").read()

    assert metadata["sample_id"] == "obs-camera-0"
    assert metadata["state_vector"] == pytest.approx([1.0, 2.0])
    assert metadata["action_vector"] == pytest.approx([0.25, -0.5])
    assert metadata["caption"] == "reach toward the cube"
    assert metadata["lineage"]["table_versions"]
    assert media == b"frame-0"

    accounting = manifest.accounting
    assert accounting["payload_bytes_referenced"] == len(b"frame-0")
    assert accounting["payload_bytes_copied"] == len(b"frame-0")
    assert accounting["logical_reference_bytes"] == 0
    assert accounting["metadata_bytes_written"] > 0
    assert accounting["copy_ratio"] == pytest.approx(1.0)

    materializations = [
        row
        for row in lake.table("curation_materializations").to_arrow().to_pylist()
        if row["projection_transform_id"] == manifest.transform_id
    ]
    assert len(materializations) == 1
    assert materializations[0]["target_format"] == "webdataset"
    assert materializations[0]["copied_payload_bytes"] == len(b"frame-0")
    assert materializations[0]["logical_reference_bytes"] == 0


@pytest.mark.skipif(
    importlib.util.find_spec("webdataset") is None,
    reason="WebDataset optional dependency is not installed",
)
def test_webdataset_export_reads_with_official_webdataset_package(lake, tmp_path):
    out = tmp_path / "webdataset"
    export_dataset_snapshot(lake, "demo-v1", out_dir=out, fmt="webdataset")

    import webdataset as wds

    samples = list(wds.WebDataset(str(out / "shards/shard-000000.tar")))

    assert samples[0]["__key__"] == "episode-000000/frame-000000-obs-camera-0"
    assert json.loads(samples[0]["json"].decode())["sample_id"] == "obs-camera-0"


def test_dataset_export_content_hash_is_reproducible(lake, tmp_path):
    first = export_dataset_snapshot(lake, "demo-v1", out_dir=tmp_path / "a", fmt="lerobot")
    second = export_dataset_snapshot(lake, "demo-v1", out_dir=tmp_path / "b", fmt="lerobot")

    assert second.content_hash == first.content_hash


def test_dataset_export_records_transform_lineage(lake, tmp_path):
    manifest = export_dataset_snapshot(lake, "demo-v1", out_dir=tmp_path / "lerobot", fmt="lerobot")

    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == manifest.transform_id
    )
    assert transform["kind"] == "dataset-export"
    assert transform["source_id"] == manifest.dataset_id
    assert {tv["table"] for tv in transform["input_table_versions"]} >= {
        "scenarios",
        "runs",
        "observations",
    }
    params = json.loads(transform["params"])
    assert params["format"] == "lerobot"
    assert params["content_hash"] == manifest.content_hash


def test_dataset_export_reports_missing_native_loader(lake, tmp_path, monkeypatch):
    import lancedb_robotics.dataset_export as dataset_export

    monkeypatch.setattr(dataset_export, "_module_available", lambda module: False)

    manifest = export_dataset_snapshot(lake, "demo-v1", out_dir=tmp_path / "lerobot", fmt="lerobot")
    assert manifest.native_loader["available"] is False
    assert manifest.native_loader["missing"] == ["lerobot"]

    with pytest.raises(DatasetExportError, match="native loader unavailable"):
        export_dataset_snapshot(
            lake,
            "demo-v1",
            out_dir=tmp_path / "rlds",
            fmt="rlds",
            require_native=True,
        )


def test_dataset_export_errors_for_unknown_snapshot_and_format(lake, tmp_path):
    with pytest.raises(DatasetExportError):
        export_dataset_snapshot(lake, "ghost", out_dir=tmp_path / "x", fmt="lerobot")

    with pytest.raises(DatasetExportError):
        export_dataset_snapshot(lake, "demo-v1", out_dir=tmp_path / "x", fmt="unknown")
