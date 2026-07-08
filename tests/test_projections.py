"""Projection contract tests (backlog 0040)."""

import json
import os
from datetime import UTC, datetime
from types import SimpleNamespace

import pyarrow as pa
import pytest

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.lake import Lake
from lancedb_robotics.projections import (
    PROJECTION_MANIFEST_FILENAME,
    LeRobotLiveDataset,
    ProjectionDependencyError,
    ProjectionManifest,
    ProjectionMode,
    RLDSLiveDataset,
    WebDatasetLiveDataset,
)
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA, SCENARIOS_SCHEMA

_REQUIRE_RLDS_NATIVE = os.getenv("LANCEDB_ROBOTICS_REQUIRE_RLDS_NATIVE") == "1"


def _require_rlds_native_stack() -> None:
    import lancedb_robotics.dataset_export as dataset_export

    status = dataset_export.native_loader_status("rlds")
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


def _projection_lake(path):
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-projection",
                    "run_kind": "demo",
                    "source": "synthetic",
                    "source_id": "src-projection",
                    "raw_uri": "memory://run-projection",
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
                    "run_id": "run-projection",
                    "timestamp_ns": 1_000,
                    "sensor_id": "camera_front",
                    "topic": "/camera/front",
                    "modality": "image",
                    "raw_uri": "memory://run-projection",
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
                    "run_id": "run-projection",
                    "timestamp_ns": 2_000,
                    "sensor_id": "joint_state",
                    "topic": "/joint_states",
                    "modality": "state",
                    "raw_uri": "memory://run-projection",
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
                    "scenario_id": "scn-projection-a",
                    "run_id": "run-projection",
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
                    "scenario_id": "scn-projection-b",
                    "run_id": "run-projection",
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
        scenario_ids=["scn-projection-a", "scn-projection-b"],
        split_by="scenario",
    )
    return lake


@pytest.fixture
def lake(tmp_path):
    return _projection_lake(tmp_path / "robot.lance")


def test_projection_manifest_schema_validates_required_contract():
    manifest = ProjectionManifest(
        lake_uri="memory://lake",
        source_snapshot_id="ds-1",
        snapshot_name="demo-v1",
        table_versions=({"table": "scenarios", "version": 1, "tag": ""},),
        format="lerobot",
        format_version="lerobot-v3.0",
        mode=ProjectionMode.LIVE,
        feature_schema={"features": {"step": {"count": 2}}},
        lossiness=(),
        media_policy={"payloads": "lazy"},
        output_paths=(),
        live_adapter={"class": "ProjectionLiveDataset"},
        content_hashes={},
        transform_id="tfm-projection-live-demo",
        transform_lineage_id="tfm-projection-live-demo",
        accounting={
            "version": "projection-accounting-v1",
            "logical_row_count": 2,
            "selected_scenario_count": 2,
            "selected_observation_count": 2,
            "payload_bytes_referenced": 7,
            "payload_bytes_copied": 0,
            "metadata_bytes_written": 0,
            "target_format": "lerobot",
            "target_path": "",
            "projection_transform_id": "tfm-projection-live-demo",
            "source_snapshot_id": "ds-1",
            "source_snapshot_name": "demo-v1",
            "source_table_versions": [{"table": "scenarios", "version": 1, "tag": ""}],
            "mode": "live",
            "payload_copy_policy": "logical-reference",
            "dry_run": False,
        },
    )

    manifest.validate()
    payload = manifest.to_dict()
    round_tripped = ProjectionManifest.from_dict(payload)
    assert round_tripped == manifest

    broken = {**payload, "source_snapshot_id": ""}
    with pytest.raises(ValueError, match="source_snapshot_id"):
        ProjectionManifest.from_dict(broken).validate()


def test_lake_projection_live_adapter_dispatches_through_registry(lake):
    assert hasattr(lake, "projections")
    assert not hasattr(lake, "facades")

    adapter = lake.projections.lerobot.dataset("demo-v1", mode="live")

    assert isinstance(adapter, LeRobotLiveDataset)
    assert adapter.manifest.mode == ProjectionMode.LIVE
    assert adapter.manifest.format == "lerobot"
    assert adapter.manifest.format_version == "lerobot-v3.0"
    assert adapter.manifest.live_adapter["class"] == "LeRobotLiveDataset"
    assert adapter.manifest.live_adapter["protocol"] == "torch.utils.data.Dataset"
    assert len(adapter) == 2
    sample = adapter[0]
    assert sample["observation_id"] == "obs-camera-0"
    assert sample["observation.state"] == [1.0, 2.0]
    assert sample["action"] == [0.25, -0.5]
    assert sample["task"] == "reach toward the cube"
    assert sample["observation.images.camera_front"].observation_id == "obs-camera-0"
    assert sample["_lineage"]["snapshot_id"] == adapter.manifest.source_snapshot_id
    assert sample["_lineage"]["row_plan_id"] == adapter.native_dataset.row_plan.plan_id
    assert sample["_lineage"]["frame_id"] == "obs-camera-0"
    assert adapter.manifest.table_versions
    assert adapter.features["observation.images.camera_front"]["dtype"] == "image"
    assert adapter.meta.episodes[0]["length"] == 1
    assert adapter.meta.frames[0]["observation_id"] == "obs-camera-0"
    assert adapter.meta.videos[0]["camera_key"] == "camera_front"

    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == adapter.manifest.transform_id
    )
    assert transform["kind"] == "projection"
    assert json.loads(transform["params"])["mode"] == "live"


def test_projection_plan_is_manifest_only(lake):
    manifest = lake.projections.lerobot.plan("demo-v1")

    assert manifest.mode == ProjectionMode.PLAN
    assert manifest.format_version == "lerobot-v3.0"
    assert manifest.output_paths == ()
    assert manifest.content_hashes == {}
    assert manifest.feature_schema["format"] == "lerobot"
    assert manifest.feature_schema["features"]["observation.state"]["shape"] == [2]
    assert manifest.feature_schema["features"]["cameras"][0]["key"] == "camera_front"
    assert manifest.accounting["dry_run"] is True
    assert manifest.accounting["logical_row_count"] == 2
    assert manifest.accounting["payload_bytes_referenced"] == len(b"frame-0")
    assert manifest.accounting["payload_bytes_copied"] == 0
    assert manifest.accounting["payload_bytes_planned"] == len(b"frame-0")
    assert manifest.accounting["logical_reference_bytes"] == len(b"frame-0")
    assert manifest.accounting["copy_ratio"] == 0.0
    assert manifest.accounting["payload_copy_policy"] == "would-copy-payloads"


def test_webdataset_projection_plan_reports_shards_schema_and_dependency(lake, monkeypatch):
    import lancedb_robotics.dataset_export as dataset_export

    monkeypatch.setattr(dataset_export, "_module_available", lambda module: False)

    manifest = lake.projections.webdataset.plan(
        "demo-v1",
        shard_size=1,
        compression="gzip",
    )

    assert manifest.mode == ProjectionMode.PLAN
    assert manifest.format == "webdataset"
    assert manifest.format_version == "webdataset-tar-v0"
    assert manifest.media_policy["projection_options"] == {
        "shard_size": 1,
        "compression": "gzip",
    }
    dry_run = manifest.feature_schema["dry_run"]
    assert dry_run["shard_count"] == 2
    assert dry_run["sample_count"] == 2
    assert dry_run["compression"] == "gzip"
    assert dry_run["sample_schema"]["__key__"]
    assert manifest.media_policy["native_loader"]["missing"] == ["webdataset"]


def test_lerobot_live_adapter_preserves_native_row_plan_sample_ids(lake):
    adapter = lake.projections.lerobot.dataset("demo-v1", mode="live")

    assert [sample["_lineage"]["frame_id"] for sample in adapter] == list(
        adapter.native_dataset.row_plan.frame_ids
    )
    assert [sample["observation_id"] for sample in adapter] == [
        "obs-camera-0",
        "obs-state-1",
    ]


def test_webdataset_live_adapter_shapes_samples_and_media(lake):
    adapter = lake.projections.webdataset.dataset("demo-v1", mode="live")

    assert isinstance(adapter, WebDatasetLiveDataset)
    assert adapter.manifest.live_adapter["class"] == "WebDatasetLiveDataset"
    assert adapter.manifest.live_adapter["protocol"] == "webdataset-iterable"
    assert [sample["_lineage"]["frame_id"] for sample in adapter] == list(
        adapter.native_dataset.row_plan.frame_ids
    )
    sample = adapter[0]
    assert sample["__key__"] == "episode-000000/frame-000000-obs-camera-0"
    assert sample["json"]["sample_id"] == "obs-camera-0"
    assert sample["json"]["state_vector"] == [1.0, 2.0]
    assert sample["json"]["action_vector"] == [0.25, -0.5]
    assert sample["json"]["caption"] == "reach toward the cube"
    assert sample["jpg"] == b"frame-0"
    assert sample["txt"] == b"reach toward the cube"
    assert sample["_lineage"]["row_plan_id"] == adapter.native_dataset.row_plan.plan_id


def test_rlds_live_adapter_iterates_episodes_and_steps(lake):
    adapter = lake.projections.rlds.dataset("demo-v1", mode="live")

    assert isinstance(adapter, RLDSLiveDataset)
    assert adapter.manifest.mode == ProjectionMode.LIVE
    assert adapter.manifest.format == "rlds"
    assert adapter.manifest.format_version == "rlds-tfds-style-v0"
    assert adapter.manifest.live_adapter["class"] == "RLDSLiveDataset"
    assert adapter.manifest.live_adapter["protocol"] == "rlds-episode-iterator"
    assert len(adapter) == 2
    assert adapter.num_steps == 2

    episode = adapter[0]
    assert episode["episode_id"] == "scn-projection-a"
    assert episode["episode_metadata"]["source_table"] == "scenarios"
    assert episode["episode_metadata"]["task"] == "pick the cube"
    assert episode["_lineage"]["frame_ids"] == ["obs-camera-0"]

    step = episode["steps"][0]
    assert step["observation"]["state"] == [1.0, 2.0]
    assert step["observation"]["caption"] == "reach toward the cube"
    assert step["observation"]["image"]["camera_front"].observation_id == "obs-camera-0"
    assert step["action"] == [0.25, -0.5]
    assert step["reward"] == 0.0
    assert step["discount"] == 0.0
    assert step["is_first"] is True
    assert step["is_last"] is True
    assert step["is_terminal"] is True
    assert step["metadata"]["observation_id"] == "obs-camera-0"
    assert step["metadata"]["raw_uri"] == "memory://run-projection"
    assert step["_lineage"]["snapshot_id"] == adapter.manifest.source_snapshot_id
    assert step["_lineage"]["row_plan_id"] == adapter.native_dataset.row_plan.plan_id

    flattened = list(adapter.steps())
    assert [item["metadata"]["observation_id"] for item in flattened] == [
        "obs-camera-0",
        "obs-state-1",
    ]


@pytest.mark.rlds_native
def test_rlds_live_adapter_uses_standard_rlds_keys_when_available():
    _require_rlds_native_stack()
    import rlds

    manifest = ProjectionManifest(
        lake_uri="memory://lake",
        source_snapshot_id="ds-native-rlds",
        snapshot_name="demo-v1",
        table_versions=({"table": "observations", "version": 1, "tag": ""},),
        format="rlds",
        format_version="rlds-tfds-style-v0",
        mode=ProjectionMode.LIVE,
        feature_schema={"features": {}},
        lossiness=(),
        media_policy={},
        output_paths=(),
        live_adapter={"class": "RLDSLiveDataset"},
        content_hashes={},
        transform_id="tfm-native-rlds",
        transform_lineage_id="tfm-native-rlds",
    )
    episode_ref = SimpleNamespace(
        episode_id="episode-0",
        index=0,
        scenario={
            "scenario_id": "scn-0",
            "run_id": "run-0",
            "start_time_ns": 1_000,
            "end_time_ns": 1_000,
        },
        split="train",
        task_index=0,
        task="pick the cube",
        physical_episode=None,
    )

    class _NativeDataset:
        _episodes = (episode_ref,)
        _frame_refs = (SimpleNamespace(episode=episode_ref, frame_index=0),)
        row_plan = SimpleNamespace(
            plan_id="row-plan-native-rlds",
            row_ids=[0],
            frame_ids=["obs-native-rlds"],
        )
        epoch_plan = SimpleNamespace(plan_id="epoch-plan-native-rlds", sample_indices=[0])

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "observation_id": "obs-native-rlds",
                "episode_id": "episode-0",
                "scenario_id": "scn-0",
                "run_id": "run-0",
                "split": "train",
                "episode_index": 0,
                "frame_index": 0,
                "timestamp_ns": 1_000,
                "relative_time_s": 0.0,
                "sensor_id": "joint_state",
                "topic": "/joint_states",
                "modality": "state",
                "state_vector": [1.0, 2.0],
                "action_vector": [0.25, -0.5],
                "caption": "reach toward the cube",
                "payload_json": "{\"joint\":1}",
                "raw_uri": "memory://run-0",
                "raw_channel": "/joint_states",
                "raw_sequence": 0,
                "message_encoding": "json",
                "schema_encoding": "json",
            }

    adapter = RLDSLiveDataset(manifest=manifest, native_dataset=_NativeDataset())
    episode = adapter[0]
    step = episode[getattr(rlds, "STEPS", "steps")][0]

    assert getattr(rlds, "OBSERVATION", "observation") in step
    assert getattr(rlds, "ACTION", "action") in step
    assert getattr(rlds, "REWARD", "reward") in step
    assert getattr(rlds, "DISCOUNT", "discount") in step
    assert getattr(rlds, "IS_FIRST", "is_first") in step
    assert getattr(rlds, "IS_LAST", "is_last") in step
    assert getattr(rlds, "IS_TERMINAL", "is_terminal") in step


def test_projection_export_writes_manifest_and_is_idempotent(lake, tmp_path):
    first = lake.projections.lerobot.export("demo-v1", out=tmp_path / "lerobot")
    second = lake.projections.lerobot.export("demo-v1", out=tmp_path / "lerobot")

    assert first.transform_id == second.transform_id
    assert first.content_hashes == second.content_hashes
    assert (tmp_path / "lerobot" / PROJECTION_MANIFEST_FILENAME).is_file()
    assert (tmp_path / "lerobot" / "dataset_export_manifest.json").is_file()
    assert first.accounting["target_path"] == str(tmp_path / "lerobot")
    assert first.accounting["logical_row_count"] == 2
    assert first.accounting["payload_bytes_referenced"] == len(b"frame-0")
    assert first.accounting["payload_bytes_copied"] == len(b"frame-0")
    assert first.accounting["metadata_bytes_written"] > 0
    assert first.accounting["source_snapshot_id"] == first.source_snapshot_id

    rows = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == first.transform_id
    ]
    assert len(rows) == 1
    assert json.loads(rows[0]["params"])["mode"] == "export"
    materializations = [
        row
        for row in lake.table("curation_materializations").to_arrow().to_pylist()
        if row["projection_transform_id"] == first.transform_id
    ]
    assert len(materializations) == 1
    assert materializations[0]["copied_payload_bytes"] == len(b"frame-0")
    assert materializations[0]["metadata_bytes_written"] == first.accounting["metadata_bytes_written"]


def test_scoped_lineage_context_flows_to_training_and_projection_export(lake, tmp_path):
    with lake.lineage.context(
        {
            "provider": "dagster",
            "external_run_id": "dagster-run-0113",
            "external_job_id": "asset-job",
            "external_refs": {"dagster_asset_key": "robotics/demo"},
        }
    ):
        training = lake.training.record_run(
            "demo-v1",
            training_run_id="train-scoped-0113",
            hyperparameters={"lr": 0.001},
        )
        exported = lake.projections.lerobot.export(
            "demo-v1",
            out=tmp_path / "scoped-lerobot",
        )
        with lake.lineage.context(
            {
                "external_run_id": "dagster-child-0113",
                "external_refs": {"step": "override"},
            }
        ):
            child = lake.training.record_run(
                "demo-v1",
                training_run_id="train-child-0113",
            )
        explicit = lake.training.record_run(
            "demo-v1",
            training_run_id="train-explicit-0113",
            lineage_context={
                "external_run_id": "manual-override-0113",
                "external_refs": {"manual": "yes"},
            },
        )

    assert training.lineage_context["provider"] == "dagster"
    assert training.lineage_context["external_run_id"] == "dagster-run-0113"
    assert training.external_refs["dagster_run_id"] == "dagster-run-0113"
    assert training.external_refs["dagster_asset_key"] == "robotics/demo"
    assert exported.lineage_context["external_run_id"] == "dagster-run-0113"
    assert exported.lineage_context["external_refs"]["dagster_asset_key"] == "robotics/demo"
    assert child.lineage_context["provider"] == "dagster"
    assert child.lineage_context["external_run_id"] == "dagster-child-0113"
    assert child.lineage_context["external_refs"] == {
        "dagster_asset_key": "robotics/demo",
        "step": "override",
    }
    assert explicit.lineage_context["provider"] == "dagster"
    assert explicit.lineage_context["external_run_id"] == "manual-override-0113"
    assert explicit.lineage_context["external_refs"]["manual"] == "yes"
    assert explicit.lineage_context["external_refs"]["dagster_asset_key"] == "robotics/demo"
    assert lake.lineage.current_context() is None

    transforms = {
        row["transform_id"]: json.loads(row["params"])
        for row in lake.table("transform_runs").to_arrow().to_pylist()
    }
    assert transforms[training.transform_id]["lineage_context"]["external_run_id"] == "dagster-run-0113"
    assert transforms[exported.transform_id]["lineage_context"]["external_run_id"] == "dagster-run-0113"


def test_webdataset_projection_export_writes_deterministic_shards(lake, tmp_path):
    plan = lake.projections.webdataset.plan(
        "demo-v1",
        shard_size=1,
    )
    first = lake.projections.webdataset.export(
        "demo-v1",
        out=tmp_path / "webdataset",
        shard_size=1,
    )
    second = lake.projections.webdataset.export(
        "demo-v1",
        out=tmp_path / "webdataset",
        shard_size=1,
    )

    assert first.content_hashes == second.content_hashes
    assert first.feature_schema["dry_run"]["shard_count"] == 2
    assert plan.accounting["dry_run"] is True
    assert first.accounting["dry_run"] is False
    assert plan.accounting["payload_bytes_referenced"] == first.accounting["payload_bytes_referenced"]
    assert plan.accounting["payload_bytes_copied"] == 0
    assert plan.accounting["logical_reference_bytes"] == plan.accounting["payload_bytes_referenced"]
    assert plan.accounting["payload_bytes_planned"] == first.accounting["payload_bytes_copied"]
    assert plan.accounting["copy_ratio"] == 0.0
    assert plan.accounting["logical_row_count"] == first.accounting["logical_row_count"]
    assert (tmp_path / "webdataset" / PROJECTION_MANIFEST_FILENAME).is_file()
    assert (tmp_path / "webdataset" / "shards/shard-000000.tar").is_file()

    plan_materializations = [
        row
        for row in lake.table("curation_materializations").to_arrow().to_pylist()
        if row["projection_transform_id"] == plan.transform_id
    ]
    assert len(plan_materializations) == 1
    assert plan_materializations[0]["copied_payload_bytes"] == 0
    assert plan_materializations[0]["logical_reference_bytes"] == len(b"frame-0")
    plan_report = json.loads(plan_materializations[0]["report_json"])
    assert plan_report["accounting"]["payload_bytes_planned"] == len(b"frame-0")


def test_rlds_projection_plan_and_export_share_episode_step_mapping(lake, tmp_path, monkeypatch):
    import lancedb_robotics.dataset_export as dataset_export

    monkeypatch.setattr(dataset_export, "_module_available", lambda module: False)

    plan = lake.projections.rlds.plan("demo-v1")
    assert plan.mode == ProjectionMode.PLAN
    assert plan.format == "rlds"
    assert plan.feature_schema["features"]["rlds"]["steps_key"] == "steps"
    assert plan.feature_schema["features"]["reward"]["value"] == 0.0
    assert "RLDS reward is synthesized" in " ".join(plan.lossiness)
    assert plan.media_policy["native_loader"]["missing"] == [
        "rlds",
        "tensorflow",
        "tensorflow_datasets",
        "reverb",
    ]

    live = lake.projections.rlds.dataset("demo-v1")
    export = lake.projections.rlds.export("demo-v1", out=tmp_path / "rlds")

    dataset_manifest = json.loads((tmp_path / "rlds" / "dataset_export_manifest.json").read_text())
    info = json.loads((tmp_path / "rlds" / "dataset_info.json").read_text())
    assert dataset_manifest["episode_count"] == len(live)
    assert dataset_manifest["step_count"] == live.num_steps
    assert info["features"]["rlds"]["step_fields"] == [
        "observation",
        "action",
        "reward",
        "discount",
        "is_first",
        "is_last",
        "is_terminal",
        "metadata",
    ]
    assert export.feature_schema == plan.feature_schema
    assert (tmp_path / "rlds" / PROJECTION_MANIFEST_FILENAME).is_file()


def test_projection_missing_native_dependency_prevents_partial_output(lake, tmp_path, monkeypatch):
    import lancedb_robotics.dataset_export as dataset_export

    monkeypatch.setattr(dataset_export, "_module_available", lambda module: False)
    out = tmp_path / "blocked"

    with pytest.raises(ProjectionDependencyError, match="lerobot"):
        lake.projections.lerobot.export("demo-v1", out=out, require_native=True)

    assert not out.exists()

    rlds_out = tmp_path / "blocked-rlds"
    with pytest.raises(ProjectionDependencyError, match="rlds"):
        lake.projections.rlds.export("demo-v1", out=rlds_out, require_native=True)

    assert not rlds_out.exists()
