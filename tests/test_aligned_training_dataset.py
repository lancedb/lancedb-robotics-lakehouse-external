"""Aligned-frame training dataset tests (backlog 0049)."""

import json
from datetime import UTC, datetime

import pyarrow as pa
import pytest
from conftest import require_start_method, require_torch_loader

import lancedb_robotics.training as training_mod
from lancedb_robotics.connections import LakeCapabilities, LakeConnectionSpec
from lancedb_robotics.indexing import ScalarIndexResult
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import ALIGNED_TICKS_SCHEMA, OBSERVATIONS_SCHEMA, RUNS_SCHEMA
from lancedb_robotics.training import (
    TrainingError,
    aligned_training_dataset,
    collate_torch_aligned_training_samples,
    index_aligned_training_predicates,
    iter_aligned_training_batches,
    to_torch_aligned_dataloader,
    to_torch_aligned_map_dataset,
)


def _aligned_training_lake(path):
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-aligned-training",
                    "run_kind": "teleop",
                    "source": "synthetic",
                    "source_id": "src-aligned-training",
                    "raw_uri": "memory://aligned-training",
                    "robot_id": "robot-arm",
                    "site_id": "lab-a",
                    "task_id": "insert peg",
                    "start_time_ns": 0,
                    "end_time_ns": 100_000_000,
                    "duration_ns": 100_000_000,
                    "software_version": "sw-1",
                    "hardware_version": "hw-1",
                    "calibration_version": "cal-1",
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
    rows = []

    def observation(
        observation_id: str,
        topic: str,
        timestamp_ns: int,
        sequence: int,
        *,
        state_vector: list[float] | None = None,
        action_vector: list[float] | None = None,
        payload_blob: bytes | None = None,
    ) -> dict:
        return {
            "observation_id": observation_id,
            "run_id": "run-aligned-training",
            "episode_id": None,
            "episode_index": None,
            "frame_index": None,
            "timestamp_ns": timestamp_ns,
            "sensor_id": topic.strip("/").replace("/", "_"),
            "topic": topic,
            "modality": "action" if topic == "/action" else "state",
            "robot_id": "robot-arm",
            "site_id": "lab-a",
            "task_id": "insert peg",
            "software_version": "sw-1",
            "outcome": "",
            "raw_uri": "memory://aligned-training",
            "raw_channel": topic,
            "raw_log_time_ns": timestamp_ns,
            "raw_sequence": sequence,
            "payload_json": None,
            "payload_blob": payload_blob,
            "message_encoding": "json",
            "schema_encoding": "json",
            "decode_status": "decoded",
            "decode_error": "",
            "state_vector": state_vector,
            "action_vector": action_vector,
            "caption": "",
            "quality_flags": [],
            "transform_id": "tfm-ingest",
            "created_at": now,
        }

    for index, timestamp_ns in enumerate([0, 50_000_000, 100_000_000]):
        rows.append(
            observation(
                f"joint-{index}",
                "/joint_states",
                timestamp_ns,
                index,
                state_vector=[float(index * 50)],
            )
        )
    rows.append(
        observation(
            "action-0",
            "/action",
            0,
            10,
            action_vector=[0.0],
            payload_blob=b"action-payload-0",
        )
    )
    rows.append(
        observation(
            "action-1",
            "/action",
            100_000_000,
            11,
            action_vector=[1.0],
            payload_blob=b"action-payload-1",
        )
    )
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))
    view = lake.align.create_view(
        "policy_bridge",
        run_id="run-aligned-training",
        rate_hz=20.0,
        streams=["/joint_states", "/action", "/force_torque"],
        tolerance_ms=100.0,
        interpolation={
            "/joint_states": "nearest",
            "/action": "nearest",
            "/force_torque": "nearest",
        },
    )
    return lake, view


def _mark_enterprise_lake(
    lake,
    *,
    uri="db://robotics",
    server_side_query=True,
    blob_fetch_remote=True,
):
    lake.connection_spec = LakeConnectionSpec(
        kind="lancedb_remote_db",
        uri=uri,
        display_uri=uri,
        lancedb_connect_kwargs={
            "api_key": "secret-api-key",
            "region": "us-west-2",
            "host_override": "https://phalanx.acme.internal",
        },
        auth_refs={"remote": "enterprise-prod"},
        capabilities=LakeCapabilities(
            server_side_query=server_side_query,
            blob_fetch_remote=blob_fetch_remote,
        ),
    )
    lake.capabilities = lake.connection_spec.capabilities
    lake.uri = uri
    return lake


def test_aligned_dataset_pivots_recorded_alignment_into_tick_samples(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")

    dataset = lake.training.aligned_dataset(name="policy_bridge")

    assert len(dataset) == 3
    assert dataset.manifest.access_pattern == "lance-native-aligned-ticks"
    assert dataset.manifest.storage_backend == "aligned_ticks-jsonb"
    assert dataset.manifest.schema_version == "1"
    assert dataset.manifest.output_table == "aligned_ticks"
    assert dataset.manifest.alignment_id == view.alignment_id
    assert dataset.manifest.alignment_name == "policy_bridge"
    assert dataset.manifest.recipe_digest
    assert dataset.manifest.tick_plan_id == dataset.tick_plan.plan_id
    assert dataset.manifest.to_dict()["quality_policy"] == {
        "allow_missing": True,
        "min_confidence": None,
        "require_streams": [],
        "statuses": [],
    }

    sample = dataset[1]
    assert sample["alignment_id"] == view.alignment_id
    assert sample["tick_index"] == 1
    assert sample["timestamp_ns"] == 50_000_000
    assert sample["run_id"] == "run-aligned-training"
    assert list(sample["streams"]) == ["/joint_states", "/action", "/force_torque"]

    joint = sample["streams"]["/joint_states"]
    assert joint["status"] == "aligned"
    assert joint["observation_id"] == "joint-1"
    assert joint["source_observation_ids"] == ["joint-1"]
    assert joint["source_row_ids"]
    assert joint["source_timestamp_ns"] == 50_000_000
    assert joint["value"] == [50.0]

    force = sample["streams"]["/force_torque"]
    assert force["status"] == "missing"
    assert sample["masks"]["valid"]["/joint_states"] is True
    assert sample["masks"]["missing"]["/force_torque"] is True
    assert sample["masks"]["interpolated"]["/joint_states"] is False

    lineage = sample["lineage"]
    assert lineage["alignment_job"]["alignment_id"] == view.alignment_id
    assert lineage["storage_backend"] == "aligned_ticks-jsonb"
    assert lineage["transform_id"] == view.transform_id
    assert lineage["source_observation_ids"]["/joint_states"] == ["joint-1"]
    assert {item["table"] for item in lineage["input_table_versions"]} == {
        "runs",
        "observations",
    }


def test_aligned_dataset_manifest_records_predicate_index_status(tmp_path):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")

    build_results = lake.training.index_aligned_predicates(include_frames=False)
    dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        min_confidence=0.75,
        require_streams=True,
    )
    manifest = dataset.manifest.to_dict()
    indexes = {
        entry["column"]: entry
        for entry in manifest["predicate_indexes"]
        if entry["table"] == "aligned_ticks"
    }

    assert {result["status"] for result in build_results} == {"built"}
    assert indexes["alignment_id"]["status"] == "already_present"
    assert indexes["alignment_id"]["used_in_filter"] is True
    assert indexes["alignment_id"]["predicate_role"] == "filter"
    assert indexes["min_confidence"]["status"] == "already_present"
    assert indexes["min_confidence"]["predicate_role"] == "quality-diagnostic"
    assert indexes["has_missing"]["status"] == "already_present"
    assert indexes["has_missing"]["predicate_role"] == "quality-diagnostic"
    assert list(dataset.tick_plan.scan["predicate_indexes"]) == manifest["predicate_indexes"]


def test_aligned_dataset_records_skipped_predicate_indexes_without_blocking_reads(
    tmp_path, monkeypatch
):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")

    def fake_describe_scalar_indexes(lake, *, table, columns):
        return tuple(
            ScalarIndexResult(
                table=table,
                column=column,
                status="skipped",
                reason="test backend does not expose create_scalar_index",
            )
            for column in columns
        )

    monkeypatch.setattr(
        training_mod,
        "describe_scalar_indexes",
        fake_describe_scalar_indexes,
    )

    dataset = lake.training.aligned_dataset(name="policy_bridge")

    assert len(dataset) == 3
    assert {
        entry["status"] for entry in dataset.manifest.to_dict()["predicate_indexes"]
    } == {"skipped"}


def test_recorded_alignment_writes_tick_grain_jsonb_rows(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")

    rows = [
        row
        for row in lake.table("aligned_ticks").to_arrow().to_pylist()
        if row["alignment_id"] == view.alignment_id
    ]

    assert len(rows) == 3
    middle = next(row for row in rows if row["tick_index"] == 1)
    assert middle["alignment_name"] == "policy_bridge"
    assert middle["recipe_digest"].startswith("recipe-")
    assert middle["available_streams"] == ["/joint_states", "/action"]
    assert middle["missing_streams"] == ["/force_torque"]
    assert middle["has_missing"] is True
    assert middle["has_out_of_tolerance"] is False

    detail = json.loads(middle["stream_detail_json"])
    masks = json.loads(middle["masks_json"])
    lineage = json.loads(middle["lineage_json"])
    assert list(detail) == ["/action", "/force_torque", "/joint_states"]
    assert detail["/joint_states"]["status"] == "aligned"
    assert detail["/joint_states"]["source_row_ids"]
    assert detail["/joint_states"]["value_json"] == "[50.0]"
    assert detail["/force_torque"]["status"] == "missing"
    assert masks["missing"]["/force_torque"] is True
    assert lineage["source_observation_ids"]["/joint_states"] == ["joint-1"]


def test_aligned_dataset_falls_back_to_frames_pivot_when_tick_rows_absent(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    lake.table("aligned_ticks").delete(f"alignment_id = '{view.alignment_id}'")

    dataset = lake.training.aligned_dataset(
        alignment_id=view.alignment_id,
        streams=["/joint_states", "/action"],
        statuses=["aligned"],
        min_confidence=0.75,
    )

    assert dataset.manifest.access_pattern == "lance-native-aligned-frames"
    assert dataset.manifest.storage_backend == "aligned_frames-pivot"
    assert dataset.manifest.output_table == "aligned_frames"
    assert dataset.tick_plan.scan["table"] == "aligned_frames"
    assert "stream IN ('/joint_states', '/action')" in dataset.tick_plan.scan["filter_predicate"]
    assert "status IN ('aligned')" in dataset.tick_plan.scan["filter_predicate"]
    assert "confidence >= 0.75" in dataset.tick_plan.scan["filter_predicate"]
    predicate_indexes = {
        entry["column"]: entry for entry in dataset.manifest.to_dict()["predicate_indexes"]
    }
    assert predicate_indexes["stream"]["used_in_filter"] is True
    assert predicate_indexes["status"]["used_in_filter"] is True
    assert predicate_indexes["confidence"]["used_in_filter"] is True
    assert [sample["tick_index"] for sample in dataset] == [0, 1, 2]
    assert dataset[1]["streams"]["/action"]["status"] == "filtered"


def test_aligned_predicate_index_top_level_function_matches_lake_namespace(tmp_path):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")

    first = index_aligned_training_predicates(lake, include_frames=False)
    second = lake.training.index_aligned_predicates(include_frames=False)

    assert {result["status"] for result in first} == {"built"}
    assert {result["status"] for result in second} == {"already_present"}


def test_backfill_aligned_ticks_from_frames_preserves_metadata_samples(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    before = lake.training.aligned_dataset(name="policy_bridge")
    expected = [
        {
            "tick_index": sample["tick_index"],
            "streams": {
                stream: {
                    "status": payload["status"],
                    "observation_id": payload["observation_id"],
                    "source_observation_ids": payload["source_observation_ids"],
                    "source_row_ids": payload["source_row_ids"],
                    "value": payload["value"],
                }
                for stream, payload in sample["streams"].items()
            },
            "masks": sample["masks"],
        }
        for sample in before
    ]
    lake.table("aligned_ticks").delete(f"alignment_id = '{view.alignment_id}'")
    fallback = lake.training.aligned_dataset(name="policy_bridge")
    assert fallback.manifest.storage_backend == "aligned_frames-pivot"

    result = lake.training.backfill_aligned_ticks(name="policy_bridge")
    after = lake.training.aligned_dataset(name="policy_bridge")
    actual = [
        {
            "tick_index": sample["tick_index"],
            "streams": {
                stream: {
                    "status": payload["status"],
                    "observation_id": payload["observation_id"],
                    "source_observation_ids": payload["source_observation_ids"],
                    "source_row_ids": payload["source_row_ids"],
                    "value": payload["value"],
                }
                for stream, payload in sample["streams"].items()
            },
            "masks": sample["masks"],
        }
        for sample in after
    ]

    assert result["aligned_ticks_written"] == 3
    assert result["source_aligned_frame_rows"] == 9
    assert result["metadata_samples_verified"] is True
    assert after.manifest.storage_backend == "aligned_ticks-jsonb"
    assert actual == expected


def test_aligned_tick_summary_mismatch_is_rejected(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    row = next(
        row
        for row in lake.table("aligned_ticks").to_arrow().to_pylist()
        if row["alignment_id"] == view.alignment_id and row["tick_index"] == 1
    )
    lake.table("aligned_ticks").delete(f"aligned_tick_id = '{row['aligned_tick_id']}'")
    row["missing_streams"] = []
    lake.table("aligned_ticks").add(pa.Table.from_pylist([row], schema=ALIGNED_TICKS_SCHEMA))

    with pytest.raises(TrainingError, match="missing_streams"):
        lake.training.aligned_dataset(name="policy_bridge")


def test_aligned_dataset_stream_status_confidence_filters_affect_plan_and_samples(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")

    dataset = lake.training.aligned_dataset(
        alignment_id=view.alignment_id,
        streams=["/joint_states", "/action"],
        statuses=["aligned"],
        min_confidence=0.75,
    )

    assert len(dataset) == 3
    assert dataset.manifest.streams == ("/joint_states", "/action")
    assert dataset.manifest.storage_backend == "aligned_ticks-jsonb"
    assert dataset.tick_plan.scan["table"] == "aligned_ticks"
    assert dataset.tick_plan.scan["post_filter"] == (
        "stream_detail_json status/min_confidence per selected stream"
    )
    middle = dataset[1]
    assert middle["streams"]["/action"]["status"] == "filtered"
    assert middle["masks"]["missing"]["/action"] is True
    assert middle["masks"]["valid"]["/joint_states"] is True

    required = lake.training.aligned_dataset(
        alignment_id=view.alignment_id,
        streams=["/joint_states", "/action"],
        min_confidence=0.75,
        require_streams=True,
    )
    assert [sample["tick_index"] for sample in required] == [0, 2]


def test_aligned_dataset_function_matches_lake_namespace_and_projects_columns(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")

    direct = aligned_training_dataset(
        lake,
        name="policy_bridge",
        streams=["/joint_states"],
        columns=["tick_index", "streams"],
        shuffle=True,
        shuffle_seed=17,
    )
    via_lake = lake.training.aligned_dataset(
        alignment_id=view.alignment_id,
        streams=["/joint_states"],
        columns=["tick_index", "streams"],
        shuffle=True,
        shuffle_seed=17,
    )

    assert [sample["tick_index"] for sample in direct] == [
        sample["tick_index"] for sample in via_lake
    ]
    assert direct.epoch_plan.plan_id == via_lake.epoch_plan.plan_id
    assert set(direct[0]) == {"tick_index", "streams"}


def test_aligned_dataset_metadata_features_do_not_read_payload_blobs(
    tmp_path, monkeypatch
):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")

    def fail_row_id_fetch(*args, **kwargs):
        raise AssertionError("metadata feature policy should not fetch payload blobs")

    def fail_observation_id_fetch(*args, **kwargs):
        raise AssertionError("metadata feature policy should not fetch payload blobs")

    monkeypatch.setattr(training_mod, "fetch_blobs_by_row_id", fail_row_id_fetch)
    monkeypatch.setattr(training_mod, "fetch_blobs", fail_observation_id_fetch)

    dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/action"],
        features="metadata",
    )
    sample = dataset[0]
    action = sample["streams"]["/action"]

    assert action["feature"]["policy"] == "metadata"
    assert action["feature"]["source_rows"][0]["materialized"] is False
    assert action["feature"]["source_rows"][0]["row_id"] in action["source_row_ids"]
    assert dataset.manifest.to_dict()["features"] == {
        "policy": "metadata",
        "decoder": "auto",
        "cache": {"policy": "none", "max_entries": 128},
    }
    assert dataset.tick_plan.materialization_policies["features"] == (
        "metadata:aligned_ticks-jsonb"
    )


def test_aligned_dataset_value_features_parse_value_json_without_payload_reads(
    tmp_path, monkeypatch
):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")

    def fail_row_id_fetch(*args, **kwargs):
        raise AssertionError("value feature policy should not fetch payload blobs")

    monkeypatch.setattr(training_mod, "fetch_blobs_by_row_id", fail_row_id_fetch)

    dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/joint_states"],
        features="value",
    )
    sample = dataset[1]
    joint = sample["streams"]["/joint_states"]

    assert joint["value"] == [50.0]
    assert joint["feature"]["policy"] == "value"
    assert joint["feature"]["value"] == [50.0]
    assert sample["lineage"]["features"]["policy"] == "value"
    assert sample["lineage"]["features"]["streams"]["/joint_states"][
        "value_materialized"
    ] is True


def test_aligned_dataset_bytes_features_coalesce_source_row_fetches(
    tmp_path, monkeypatch
):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    calls: list[tuple[int, ...]] = []
    original = training_mod.fetch_blobs_by_row_id

    def counted_fetch_blobs_by_row_id(handle, blob_column, row_ids):
        calls.append(tuple(row_ids))
        return original(handle, blob_column, row_ids)

    monkeypatch.setattr(training_mod, "fetch_blobs_by_row_id", counted_fetch_blobs_by_row_id)

    dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/action"],
        features="bytes",
        feature_cache="epoch",
    )
    first = dataset[0]["streams"]["/action"]["feature"]["source_rows"][0]
    second = dataset[1]["streams"]["/action"]["feature"]["source_rows"][0]

    assert first["payload"] == b"action-payload-0"
    assert second["payload"] == b"action-payload-0"
    assert first["row_id"] == second["row_id"]
    assert calls == [(first["row_id"],)]
    assert second["cache_hit"] is True
    assert dataset.tick_plan.materialization_policies["payload"].startswith(
        "bytes:observations.payload_blob-or-video_encodings.data"
    )
    assert dataset.manifest.to_dict()["features"]["cache"]["policy"] == "epoch"


def test_enterprise_aligned_batch_hydration_coalesces_grouped_source_rows(tmp_path):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    _mark_enterprise_lake(lake)
    lake.query_node_cache_telemetry = {
        "remote_take": {"per_addr": {"pe-a": {"hits": 3, "misses": 1}}}
    }

    dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/action"],
        features="bytes",
        backend="enterprise",
    )
    batch = next(iter_aligned_training_batches(dataset, batch_size=2))

    features = batch["streams"]["/action"]["feature"]
    source_rows = [feature["source_rows"] for feature in features]
    assert batch["tick_index"] == [0, 1]
    assert source_rows[0][0]["payload"] == b"action-payload-0"
    assert source_rows[1][0]["payload"] == b"action-payload-0"
    assert source_rows[0][0]["row_id"] == source_rows[1][0]["row_id"]
    metrics = dataset.manifest.backend["metrics"]
    payload_ops = [
        operation
        for operation in metrics["operations"]
        if operation["columns"] == ["payload_blob"]
    ]
    assert len(payload_ops) == 1
    assert payload_ops[0]["row_ids_requested"] == 2
    assert payload_ops[0]["row_ids_unique"] == 1
    assert payload_ops[0]["coalescing_window"] == "aligned-training-batch"
    assert metrics["row_ids_coalesced"] >= 1
    assert metrics["cache_hits"] >= 3
    assert batch["_lineage"]["backend"]["resolved_backend"] == "enterprise"


def test_enterprise_aligned_loader_report_includes_tick_plan_and_cache(tmp_path):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    _mark_enterprise_lake(lake)
    lake.query_node_cache_telemetry = {
        "remote_take": {"per_addr": {"pe-a": {"hits": 2, "misses": 1}}}
    }

    dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/action"],
        features="bytes",
        backend="enterprise",
        cache_policy="lazy",
    )
    next(iter_aligned_training_batches(dataset, batch_size=2))
    report = dataset.loader_report(model_id="policy-v2").to_dict()

    assert report["loader"] == {
        "kind": "aligned-training",
        "access_pattern": "lance-native-aligned-ticks",
    }
    assert report["alignment"]["name"] == "policy_bridge"
    assert report["plans"]["tick_plan_id"] == dataset.tick_plan.plan_id
    assert report["plans"]["epoch_plan_id"] == dataset.epoch_plan.plan_id
    assert report["policies"]["features"]["policy"] == "bytes"
    assert report["policies"]["enterprise_cache"]["policy"] == "lazy"
    assert report["metrics"]["operations_by_type"]["remote_take"] >= 1
    assert report["metrics"]["cache"]["by_epoch"]["0"]["hits"] >= 2
    assert report["run"] == {"model_id": "policy-v2"}
    assert dataset.manifest.to_dict()["loader_report"]["alignment"]["name"] == "policy_bridge"


def test_enterprise_aligned_epoch_prewarm_uses_tick_plan(tmp_path):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    _mark_enterprise_lake(lake)
    requests = []
    lake.page_cache_prewarm = requests

    dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/action"],
        backend="enterprise",
        cache_policy="epoch",
    )

    assert len(requests) == 1
    request = requests[0]
    assert request["policy"] == "epoch"
    assert request["tick_plan_id"] == dataset.tick_plan.plan_id
    assert request["epoch_plan_id"] == dataset.epoch_plan.plan_id
    assert request["row_count"] == 3
    assert request["tables"][0]["table"] == "aligned_ticks"
    assert request["tables"][0]["version"] is not None
    assert "stream_detail_json" in request["tables"][0]["projected_columns"]
    assert dataset.manifest.backend["cache"]["prewarm_status"] == "submitted"


def test_aligned_dataset_decoded_features_missing_dependencies_are_targeted(
    tmp_path, monkeypatch
):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    original_find_spec = training_mod.importlib.util.find_spec

    def missing_pillow(module):
        if module == "PIL":
            return None
        return original_find_spec(module)

    monkeypatch.setattr(training_mod.importlib.util, "find_spec", missing_pillow)

    metadata = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/action"],
        features="metadata",
    )
    assert metadata[0]["streams"]["/action"]["feature"]["policy"] == "metadata"

    bytes_dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/action"],
        features="bytes",
    )
    assert bytes_dataset[0]["streams"]["/action"]["feature"]["source_rows"][0][
        "payload"
    ] == b"action-payload-0"

    decoded = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/action"],
        features="array",
    )
    with pytest.raises(TrainingError, match="features='array'.*PIL"):
        decoded[0]


def test_aligned_dataset_feature_masks_distinguish_payload_stream_and_tolerance(
    tmp_path,
):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")

    default = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/joint_states", "/force_torque"],
        features="bytes",
    )
    sample = default[1]

    assert sample["masks"]["payload_missing"]["/joint_states"] is True
    assert sample["masks"]["missing"]["/force_torque"] is True
    assert sample["masks"]["out_of_tolerance"]["/joint_states"] is False
    assert "training:joint-states:missing-payload" in sample["quality_flags"]
    assert sample["lineage"]["features"]["streams"]["/joint_states"]["source_rows"][
        0
    ]["missing_payload"] is True

    strict = lake.align.create_view(
        "strict_policy_bridge",
        run_id="run-aligned-training",
        rate_hz=20.0,
        streams=["/action"],
        tolerance_ms=10.0,
        interpolation={"/action": "nearest"},
    )
    strict_dataset = lake.training.aligned_dataset(
        alignment_id=strict.alignment_id,
        streams=["/action"],
        features="metadata",
    )

    assert strict_dataset[1]["streams"]["/action"]["status"] == "out_of_tolerance"
    assert strict_dataset[1]["masks"]["out_of_tolerance"]["/action"] is True
    assert strict_dataset[1]["masks"]["stale"]["/action"] is True


def test_aligned_batch_iterator_collates_masks_lineage_and_variable_sources(tmp_path):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    view = lake.align.create_view(
        "linear_policy_bridge",
        run_id="run-aligned-training",
        rate_hz=40.0,
        streams=["/joint_states"],
        tolerance_ms=100.0,
        interpolation={"/joint_states": "linear"},
    )

    dataset = lake.training.aligned_dataset(
        alignment_id=view.alignment_id,
        streams=["/joint_states"],
        features="value",
    )

    batch = next(iter_aligned_training_batches(dataset, batch_size=3))

    assert batch["tick_index"] == [0, 1, 2]
    joint = batch["streams"]["/joint_states"]
    assert joint["value"] == [[0.0], [25.0], [50.0]]
    assert joint["value_mask"] == [[True], [True], [True]]
    assert joint["source_count"] == [1, 2, 1]
    assert joint["source_mask"] == [[True, False], [True, True], [True, False]]
    assert batch["masks"]["interpolated"]["/joint_states"] == [False, True, False]
    assert batch["_lineage"]["alignment_job"]["alignment_id"] == view.alignment_id
    assert batch["_lineage"]["tick_indices"] == [0, 1, 2]
    assert batch["_schema"]["streams"]["/joint_states"]["sources"]["shape"] == [3, 2]


def test_aligned_torch_adapter_missing_torch_does_not_block_neutral_iteration(
    tmp_path, monkeypatch
):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    dataset = lake.training.aligned_dataset(name="policy_bridge")
    original_find_spec = training_mod.importlib.util.find_spec

    def missing_torch(module):
        if module == "torch":
            return None
        return original_find_spec(module)

    monkeypatch.setattr(training_mod.importlib.util, "find_spec", missing_torch)

    assert [sample["tick_index"] for sample in dataset] == [0, 1, 2]
    with pytest.raises(TrainingError, match=r"lancedb-robotics\[torch\]"):
        to_torch_aligned_map_dataset(dataset)


@pytest.mark.torch_loader
def test_torch_aligned_map_dataset_collates_grouped_policy_ticks(tmp_path):
    require_torch_loader()

    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/joint_states", "/action", "/force_torque"],
        features="value",
    )
    torch_dataset = to_torch_aligned_map_dataset(dataset)

    batch = collate_torch_aligned_training_samples(
        [torch_dataset[0], torch_dataset[1]]
    )

    import torch

    assert torch.equal(batch["tick_index"], torch.tensor([0, 1]))
    assert tuple(batch["streams"]["/joint_states"]["value"].shape) == (2, 1)
    assert torch.equal(
        batch["streams"]["/joint_states"]["value_mask"],
        torch.ones((2, 1), dtype=torch.bool),
    )
    assert torch.equal(
        batch["masks"]["missing"]["/force_torque"],
        torch.ones((2,), dtype=torch.bool),
    )
    assert batch["_lineage"]["tick_indices"] == [0, 1]
    assert batch["_lineage"]["source_observation_ids"]["/joint_states"] == [
        ["joint-0"],
        ["joint-1"],
    ]


@pytest.mark.torch_loader
def test_torch_aligned_iterable_dataloader_multi_worker_covers_ticks_once(tmp_path):
    require_torch_loader()

    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        columns=["tick_index", "streams", "masks", "lineage"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
    )
    require_start_method("fork")
    loader = to_torch_aligned_dataloader(
        dataset,
        batch_size=1,
        num_workers=2,
        adapter="iterable",
        multiprocessing_context="fork",
    )

    seen = []
    worker_ids = set()
    for batch in loader:
        seen.extend(batch["tick_index"].tolist())
        worker_ids.add(batch["_lineage"]["worker"]["id"])

    expected = [sample["tick_index"] for sample in dataset]
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == len(expected)
    assert worker_ids == {0, 1}


@pytest.mark.torch_loader
def test_torch_aligned_dataloader_smoke_training_loop(tmp_path):
    require_torch_loader()

    import torch

    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        streams=["/joint_states", "/action"],
        features="value",
    )
    loader = to_torch_aligned_dataloader(
        dataset,
        batch_size=2,
        num_workers=0,
        adapter="iterable",
    )
    weight = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float32))
    optimizer = torch.optim.SGD([weight], lr=0.001)
    ticks_seen = 0

    for batch in loader:
        state = batch["streams"]["/joint_states"]["value"]
        action = batch["streams"]["/action"]["value"]
        valid = (
            batch["masks"]["valid"]["/joint_states"]
            & batch["masks"]["valid"]["/action"]
        )
        prediction = state @ weight
        loss = ((prediction - action) ** 2)[valid].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ticks_seen += int(valid.sum())

    assert ticks_seen == len(dataset)
    assert torch.isfinite(weight).all()
