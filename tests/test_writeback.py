"""Closed-loop writeback tests (backlog 0027)."""

from datetime import UTC, datetime

import pyarrow as pa
import pytest

from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import (
    DATASET_SNAPSHOTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    RUNS_SCHEMA,
    SCENARIOS_SCHEMA,
    TRANSFORM_RUNS_SCHEMA,
)
from lancedb_robotics.writeback import (
    MAX_JSON_FIELD_BYTES,
    WritebackError,
    downstream_for_run,
    import_labels,
    ingest_model_outputs,
    record_feedback,
    trace_model_output,
)


def _seed_lake(path):
    lake = Lake.init(path)
    now = datetime.now(UTC)
    lake.table("transform_runs").add(
        pa.Table.from_pylist(
            [
                {
                    "transform_id": "tfm-source",
                    "kind": "ingest",
                    "source_id": "src-1",
                    "input_uris": ["file:///sample.mcap"],
                    "input_table_versions": [],
                    "output_tables": ["runs", "observations"],
                    "params": "{}",
                    "status": "completed",
                    "started_at": now,
                    "finished_at": now,
                    "created_by": "test",
                    "created_at": now,
                },
                {
                    "transform_id": "tfm-scenarios",
                    "kind": "scenario-windowing",
                    "source_id": "scenario-windowing",
                    "input_uris": [],
                    "input_table_versions": [],
                    "output_tables": ["scenarios"],
                    "params": "{}",
                    "status": "completed",
                    "started_at": now,
                    "finished_at": now,
                    "created_by": "test",
                    "created_at": now,
                },
                {
                    "transform_id": "tfm-snapshot",
                    "kind": "dataset-snapshot",
                    "source_id": "dataset",
                    "input_uris": [],
                    "input_table_versions": [],
                    "output_tables": ["dataset_snapshots"],
                    "params": "{}",
                    "status": "completed",
                    "started_at": now,
                    "finished_at": now,
                    "created_by": "test",
                    "created_at": now,
                },
            ],
            schema=TRANSFORM_RUNS_SCHEMA,
        )
    )
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-1",
                    "run_kind": "mcap",
                    "source": "mcap",
                    "source_id": "src-1",
                    "raw_uri": "file:///sample.mcap",
                    "robot_id": "robot-a",
                    "site_id": "site-a",
                    "task_id": "pick",
                    "start_time_ns": 0,
                    "end_time_ns": 100,
                    "duration_ns": 100,
                    "software_version": "sw",
                    "hardware_version": "hw",
                    "calibration_version": "cal",
                    "model_version": "baseline",
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
                    "observation_id": "obs-1",
                    "run_id": "run-1",
                    "timestamp_ns": 10,
                    "sensor_id": "camera",
                    "topic": "/camera",
                    "modality": "image",
                    "raw_uri": "file:///sample.mcap",
                    "raw_channel": "1",
                    "raw_log_time_ns": 10,
                    "raw_sequence": 1,
                    "payload_json": "{}",
                    "payload_blob": None,
                    "message_encoding": "json",
                    "schema_encoding": "jsonschema",
                    "decode_status": "decoded",
                    "decode_error": None,
                    "state_vector": None,
                    "action_vector": None,
                    "caption": None,
                    "quality_flags": [],
                    "transform_id": "tfm-source",
                    "created_at": now,
                }
            ],
            schema=OBSERVATIONS_SCHEMA,
        )
    )
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-1",
                    "run_id": "run-1",
                    "start_time_ns": 0,
                    "end_time_ns": 100,
                    "window_ns": 100,
                    "is_partial": False,
                    "topics": ["/camera"],
                    "observation_ids": ["obs-1"],
                    "observation_count": 1,
                    "scenario_type": "fixed-window",
                    "trigger_event_id": None,
                    "source": "scenario-windowing",
                    "parent_scenario_id": None,
                    "coverage_tags": ["demo"],
                    "summary": "camera window",
                    "transform_id": "tfm-scenarios",
                    "created_at": now,
                }
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )
    lake.table("dataset_snapshots").add(
        pa.Table.from_pylist(
            [
                {
                    "dataset_id": "ds-1",
                    "name": "demo",
                    "kind": "scenario-snapshot",
                    "query_spec": '{"scenario_ids":["scn-1"]}',
                    "table_versions": [],
                    "tag": "demo",
                    "split": "{}",
                    "balance_report": None,
                    "coverage_report": None,
                    "created_by": "test",
                    "transform_id": "tfm-snapshot",
                    "created_at": now,
                }
            ],
            schema=DATASET_SNAPSHOTS_SCHEMA,
        )
    )
    return lake


@pytest.fixture
def closed_loop_lake(tmp_path):
    return _seed_lake(tmp_path / "robot.lance")


def test_label_writeback_adds_source_row_lineage_and_denormalized_columns(closed_loop_lake):
    report = import_labels(
        closed_loop_lake,
        [
            {
                "observation_id": "obs-1",
                "scenario_id": "scn-1",
                "label_type": "class",
                "label": "missed-pedestrian",
                "confidence": 0.9,
                "reviewer": "qa@example.com",
            }
        ],
        source="label-studio",
    )

    assert report.rows_written == 1
    assert report.output_tables == ("labels", "observations", "scenarios")
    label = closed_loop_lake.table("labels").to_arrow().to_pylist()[0]
    assert label["label"] == "missed-pedestrian"
    assert label["transform_id"] == report.transform_id

    observation = closed_loop_lake.table("observations").to_arrow().to_pylist()[0]
    scenario = closed_loop_lake.table("scenarios").to_arrow().to_pylist()[0]
    assert observation["label"] == "missed-pedestrian"
    assert scenario["label"] == "missed-pedestrian"

    transform = [
        row
        for row in closed_loop_lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == report.transform_id
    ][0]
    assert transform["kind"] == "label-writeback"
    assert "tfm-source" in transform["params"]
    assert "tfm-scenarios" in transform["params"]


def test_model_outputs_and_feedback_trace_end_to_end(closed_loop_lake):
    output_report = ingest_model_outputs(
        closed_loop_lake,
        [
            {
                "model_output_id": "out-1",
                "observation_id": "obs-1",
                "scenario_id": "scn-1",
                "dataset_id": "ds-1",
                "model_version": "policy@abc123",
                "prediction": "pedestrian",
                "score": 0.42,
                "producer_run_id": "wandb-run-7",
            }
        ],
        source="wandb",
    )

    feedback_report = record_feedback(
        closed_loop_lake,
        [
            {
                "model_output_id": "out-1",
                "feedback_type": "field_failure",
                "severity": "high",
                "linked_incident_id": "inc-77",
                "notes": "missed pedestrian at crossing",
            }
        ],
        source="fleet-ops",
    )

    assert output_report.rows_written == 1
    assert feedback_report.rows_written == 1
    observation = closed_loop_lake.table("observations").to_arrow().to_pylist()[0]
    scenario = closed_loop_lake.table("scenarios").to_arrow().to_pylist()[0]
    assert observation["prediction"] == "pedestrian"
    assert observation["prediction_score"] == pytest.approx(0.42)
    assert scenario["feedback_severity"] == "high"

    trace = trace_model_output(closed_loop_lake, "out-1")
    assert trace["model_output"]["model_output_id"] == "out-1"
    assert trace["observation"]["observation_id"] == "obs-1"
    assert trace["scenario"]["scenario_id"] == "scn-1"
    assert trace["dataset_snapshot"]["dataset_id"] == "ds-1"
    assert trace["source_run"]["run_id"] == "run-1"
    assert {row["kind"] for row in trace["transform_runs"]} >= {
        "ingest",
        "scenario-windowing",
        "dataset-snapshot",
        "model-output-writeback",
    }

    downstream = downstream_for_run(closed_loop_lake, "run-1")
    assert [row["model_output_id"] for row in downstream["model_outputs"]] == ["out-1"]
    assert [row["feedback_id"] for row in downstream["feedback"]]


def test_unknown_target_fails_before_partial_write(tmp_path):
    lake = _seed_lake(tmp_path / "robot.lance")

    with pytest.raises(WritebackError, match="unknown observation_id"):
        import_labels(
            lake,
            [{"observation_id": "missing", "label_type": "class", "label": "pedestrian"}],
        )

    assert lake.table("labels").count_rows() == 0
    assert "label" not in lake.table("observations").schema.names


def test_oversized_model_output_fails_before_partial_write(tmp_path):
    lake = _seed_lake(tmp_path / "robot.lance")
    oversized = "x" * (MAX_JSON_FIELD_BYTES + 1)

    with pytest.raises(WritebackError, match="maximum JSON field size"):
        ingest_model_outputs(
            lake,
            [
                {
                    "observation_id": "obs-1",
                    "model_version": "policy@abc123",
                    "output": oversized,
                }
            ],
        )

    assert lake.table("model_outputs").count_rows() == 0
    assert "prediction" not in lake.table("observations").schema.names
