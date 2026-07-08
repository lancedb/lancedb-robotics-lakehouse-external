"""First-class episode table and frame metadata tests (backlog 0029)."""

import json
from datetime import UTC, datetime

import pyarrow as pa
import pytest
from typer.testing import CliRunner

from lancedb_robotics import episodes as episodes_module
from lancedb_robotics.cli import app
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.episodes import EpisodeError, load_interval_manifest
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import (
    EVENTS_SCHEMA,
    MODEL_OUTPUTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    RUNS_SCHEMA,
    SCENARIOS_SCHEMA,
)

runner = CliRunner()


def _episode_lake(path):
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-episodes",
                    "run_kind": "teleop",
                    "source": "synthetic",
                    "source_id": "src-episodes",
                    "raw_uri": "memory://episodes",
                    "robot_id": "robot-arm-1",
                    "site_id": "lab-a",
                    "task_id": "pick cube",
                    "start_time_ns": 0,
                    "end_time_ns": 3_000_000_000,
                    "duration_ns": 3_000_000_000,
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
    observation_rows = []
    for index, (timestamp_ns, topic, modality) in enumerate(
        [
            (0, "/camera/front", "image"),
            (500_000_000, "/joint_states", "state"),
            (1_000_000_000, "/camera/front", "image"),
            (2_000_000_000, "/camera/front", "image"),
            (2_500_000_000, "/joint_states", "state"),
            (3_000_000_000, "/camera/front", "image"),
        ]
    ):
        observation_rows.append(
            {
                "observation_id": f"obs-{index}",
                "run_id": "run-episodes",
                "episode_id": None,
                "episode_index": None,
                "frame_index": None,
                "timestamp_ns": timestamp_ns,
                "sensor_id": "camera_front" if "camera" in topic else "joint_state",
                "topic": topic,
                "modality": modality,
                "robot_id": None,
                "site_id": None,
                "task_id": None,
                "software_version": None,
                "outcome": None,
                "raw_uri": "memory://episodes",
                "raw_channel": topic,
                "raw_log_time_ns": timestamp_ns,
                "raw_sequence": index,
                "payload_json": None if modality == "image" else '{"joint": 1}',
                "payload_blob": f"frame-{index}".encode() if modality == "image" else None,
                "message_encoding": "jpeg" if modality == "image" else "json",
                "schema_encoding": "jpeg" if modality == "image" else "json",
                "decode_status": "decoded",
                "decode_error": "",
                "state_vector": [float(index)],
                "action_vector": [float(index) + 0.25],
                "caption": f"observation {index}",
                "quality_flags": [],
                "transform_id": "tfm-ingest",
                "created_at": now,
            }
        )
    lake.table("observations").add(
        pa.Table.from_pylist(observation_rows, schema=OBSERVATIONS_SCHEMA)
    )
    lake.table("events").add(
        pa.Table.from_pylist(
            [
                {
                    "event_id": "evt-stop-1",
                    "run_id": "run-episodes",
                    "timestamp_ns": 3_000_000_000,
                    "start_time_ns": 3_000_000_000,
                    "end_time_ns": 3_000_000_000,
                    "event_type": "failure",
                    "severity": "critical",
                    "source": "teleop",
                    "notes": "",
                    "linked_incident_id": "",
                    "transform_id": "tfm-events",
                    "created_at": now,
                },
                {
                    "event_id": "evt-start-0",
                    "run_id": "run-episodes",
                    "timestamp_ns": 0,
                    "start_time_ns": 0,
                    "end_time_ns": 0,
                    "event_type": "teleop_start",
                    "severity": "",
                    "source": "teleop",
                    "notes": "",
                    "linked_incident_id": "",
                    "transform_id": "tfm-events",
                    "created_at": now,
                },
                {
                    "event_id": "evt-disengage",
                    "run_id": "run-episodes",
                    "timestamp_ns": 2_500_000_000,
                    "start_time_ns": 2_500_000_000,
                    "end_time_ns": 2_500_000_000,
                    "event_type": "disengagement",
                    "severity": "critical",
                    "source": "fleet",
                    "notes": "",
                    "linked_incident_id": "",
                    "transform_id": "tfm-events",
                    "created_at": now,
                },
                {
                    "event_id": "evt-stop-0",
                    "run_id": "run-episodes",
                    "timestamp_ns": 1_000_000_000,
                    "start_time_ns": 1_000_000_000,
                    "end_time_ns": 1_000_000_000,
                    "event_type": "success",
                    "severity": "",
                    "source": "teleop",
                    "notes": "",
                    "linked_incident_id": "",
                    "transform_id": "tfm-events",
                    "created_at": now,
                },
                {
                    "event_id": "evt-start-1",
                    "run_id": "run-episodes",
                    "timestamp_ns": 2_000_000_000,
                    "start_time_ns": 2_000_000_000,
                    "end_time_ns": 2_000_000_000,
                    "event_type": "teleop_start",
                    "severity": "",
                    "source": "teleop",
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
                    "scenario_id": "scn-episode-0",
                    "run_id": "run-episodes",
                    "start_time_ns": 0,
                    "end_time_ns": 1_000_000_000,
                    "window_ns": 1_000_000_000,
                    "is_partial": False,
                    "topics": ["/camera/front", "/joint_states"],
                    "observation_ids": ["obs-0", "obs-1", "obs-2"],
                    "observation_count": 3,
                    "scenario_type": "teleop",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": ["episode"],
                    "summary": "pick cube",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                },
                {
                    "scenario_id": "scn-episode-1",
                    "run_id": "run-episodes",
                    "start_time_ns": 2_000_000_000,
                    "end_time_ns": 3_000_000_000,
                    "window_ns": 1_000_000_000,
                    "is_partial": False,
                    "topics": ["/camera/front", "/joint_states"],
                    "observation_ids": ["obs-3", "obs-4", "obs-5"],
                    "observation_count": 3,
                    "scenario_type": "teleop",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": ["episode", "result:failure"],
                    "summary": "place cube",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                },
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )
    return lake


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _add_model_outputs(lake):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lake.table("model_outputs").add(
        pa.Table.from_pylist(
            [
                {
                    "model_output_id": "out-pedestrian",
                    "run_id": "run-episodes",
                    "observation_id": "obs-2",
                    "scenario_id": None,
                    "dataset_id": None,
                    "model_version": "detector-v1",
                    "output_type": "detection",
                    "prediction": "pedestrian",
                    "output_json": '{"class": "pedestrian"}',
                    "score": 0.95,
                    "producer_run_id": "model-run-1",
                    "source": "unit-test",
                    "metadata": [],
                    "transform_id": "tfm-model-output",
                    "created_at": now,
                },
                {
                    "model_output_id": "out-car",
                    "run_id": "run-episodes",
                    "observation_id": "obs-4",
                    "scenario_id": None,
                    "dataset_id": None,
                    "model_version": "detector-v1",
                    "output_type": "detection",
                    "prediction": "car",
                    "output_json": '{"class": "car"}',
                    "score": 0.4,
                    "producer_run_id": "model-run-1",
                    "source": "unit-test",
                    "metadata": [],
                    "transform_id": "tfm-model-output",
                    "created_at": now,
                },
            ],
            schema=MODEL_OUTPUTS_SCHEMA,
        )
    )


def _episode_fingerprint(lake):
    return [
        (
            row["from_timestamp_ns"],
            row["to_timestamp_ns"],
            row["outcome"],
            row["task_id"],
            row["frame_count"],
        )
        for row in sorted(
            lake.table("episodes").to_arrow().to_pylist(),
            key=lambda item: item["episode_index"],
        )
    ]


def test_from_markers_builds_episodes_tags_frames_and_records_lineage(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")

    report = lake.episodes.from_markers(
        start_event_types=["teleop_start"],
        stop_event_types=["success", "failure"],
    )

    assert report.boundary_source == "markers"
    assert report.episodes_written == 2
    assert report.frames_tagged == 6
    assert report.videos_written == 2

    episodes = sorted(
        lake.table("episodes").to_arrow().to_pylist(),
        key=lambda row: row["episode_index"],
    )
    assert [row["outcome"] for row in episodes] == ["success", "failure"]
    assert [(row["from_timestamp_ns"], row["to_timestamp_ns"]) for row in episodes] == [
        (0, 1_000_000_000),
        (2_000_000_000, 3_000_000_000),
    ]
    assert [row["frame_count"] for row in episodes] == [3, 3]
    assert episodes[0]["camera_blobs"] == ["obs-0", "obs-2"]

    observations = {
        row["observation_id"]: row for row in lake.table("observations").to_arrow().to_pylist()
    }
    assert [observations[f"obs-{index}"]["frame_index"] for index in range(3)] == [0, 1, 2]
    assert observations["obs-1"]["episode_id"] == episodes[0]["episode_id"]
    assert observations["obs-1"]["robot_id"] == "robot-arm-1"
    assert observations["obs-1"]["task_id"] == "pick cube"
    assert observations["obs-1"]["outcome"] == "success"

    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == report.transform_id
    )
    assert transform["kind"] == "episode-derivation"
    assert set(transform["output_tables"]) == {"episodes", "videos", "observations"}
    params = json.loads(transform["params"])
    assert params["boundary_source"] == "markers"
    assert params["episode_ids"] == sorted(report.episode_ids)


def test_from_query_carves_event_window_and_is_idempotent(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")

    first = lake.episodes.from_query(
        event_type="disengagement",
        before_ns=250_000_000,
        after_ns=250_000_000,
    )
    second = lake.episodes.from_query(
        event_type="disengagement",
        before_ns=250_000_000,
        after_ns=250_000_000,
    )

    assert second.transform_id == first.transform_id
    episodes = lake.table("episodes").to_arrow().to_pylist()
    assert len(episodes) == 1
    assert episodes[0]["boundary_source"] == "query"
    assert episodes[0]["from_timestamp_ns"] == 2_250_000_000
    assert episodes[0]["to_timestamp_ns"] == 2_750_000_000
    assert episodes[0]["frame_count"] == 1

    observations = lake.table("observations").to_arrow().to_pylist()
    tagged = [row for row in observations if row["episode_id"] == episodes[0]["episode_id"]]
    assert [row["observation_id"] for row in tagged] == ["obs-4"]

    transforms = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == first.transform_id
    ]
    assert len(transforms) == 1


def test_from_predicate_mines_observation_windows_merges_and_records_lineage(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")
    lake.episodes.from_markers(
        start_event_types=["teleop_start"],
        stop_event_types=["success", "failure"],
    )

    separate = lake.episodes.from_predicate(
        where="task_id = 'pick cube' AND outcome = 'failure'",
        modalities=["image"],
        dry_run=True,
    )
    merged = lake.episodes.from_predicate(
        where="task_id = 'pick cube' AND outcome = 'failure'",
        modalities=["image"],
        merge_gap_ns=1_000_000_000,
        dry_run=True,
    )
    report = lake.episodes.from_predicate(
        where="task_id = 'pick cube' AND outcome = 'failure'",
        modalities=["image"],
        merge_gap_ns=1_000_000_000,
        overlap_policy="supersede",
    )

    assert separate.dry_run is True
    assert separate.intervals_planned == 2
    assert merged.intervals_planned == 1
    assert merged.frames_planned == 2
    assert report.boundary_source == "predicate"
    assert report.episodes_written == 1
    assert report.frames_tagged == 2
    assert report.videos_written == 1
    assert report.intervals[0].observation_ids == ("obs-3", "obs-5")

    episode = next(
        row
        for row in lake.table("episodes").to_arrow().to_pylist()
        if row["transform_id"] == report.transform_id
    )
    assert episode["from_timestamp_ns"] == 2_000_000_000
    assert episode["to_timestamp_ns"] == 3_000_000_000
    assert episode["frame_count"] == 2
    provenance = json.loads(episode["provenance"])
    assert provenance["source_query"]["where"] == "task_id = 'pick cube' AND outcome = 'failure'"
    assert provenance["source_query"]["modalities"] == ["image"]
    assert provenance["source_observation_ids"] == ["obs-3", "obs-5"]

    observations = {
        row["observation_id"]: row for row in lake.table("observations").to_arrow().to_pylist()
    }
    assert observations["obs-3"]["episode_id"] == episode["episode_id"]
    assert observations["obs-5"]["episode_id"] == episode["episode_id"]
    assert observations["obs-4"]["episode_id"] != episode["episode_id"]

    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == report.transform_id
    )
    params = json.loads(transform["params"])
    assert params["boundary_source"] == "predicate"
    assert params["where"] == "task_id = 'pick cube' AND outcome = 'failure'"
    assert params["merge_gap_ns"] == 1_000_000_000
    assert params["episode_ids"] == sorted(report.episode_ids)


def test_from_predicate_mines_model_output_source_table_and_tags_frames(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")
    _add_model_outputs(lake)

    report = lake.episodes.from_predicate(
        source_table="model_outputs",
        source_where="prediction = 'pedestrian' AND score >= 0.9",
        before_ns=500_000_000,
        after_ns=500_000_000,
    )

    assert report.episodes_written == 1
    assert report.frames_tagged == 2
    assert report.intervals[0].source_observation_ids == ("obs-2",)
    assert report.intervals[0].source_model_output_ids == ("out-pedestrian",)
    episode = lake.table("episodes").to_arrow().to_pylist()[0]
    assert episode["from_timestamp_ns"] == 500_000_000
    assert episode["to_timestamp_ns"] == 1_500_000_000
    assert episode["frame_count"] == 2
    provenance = json.loads(episode["provenance"])
    assert provenance["source_model_output_ids"] == ["out-pedestrian"]
    assert provenance["source_observation_ids"] == ["obs-2"]
    assert provenance["source_query"]["source_table"] == "model_outputs"
    assert provenance["source_query"]["source_where"] == "prediction = 'pedestrian' AND score >= 0.9"

    observations = {
        row["observation_id"]: row for row in lake.table("observations").to_arrow().to_pylist()
    }
    assert observations["obs-1"]["episode_id"] == episode["episode_id"]
    assert observations["obs-2"]["episode_id"] == episode["episode_id"]
    assert observations["obs-4"]["episode_id"] != episode["episode_id"]
    transform = lake.table("transform_runs").to_arrow().to_pylist()[0]
    assert {"observations", "model_outputs"} <= {
        item["table"] for item in transform["input_table_versions"]
    }


def test_from_predicate_dry_run_has_no_table_side_effects(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")
    _add_model_outputs(lake)
    before_versions = {
        name: int(lake.table(name).version)
        for name in ("episodes", "videos", "observations", "transform_runs")
    }

    report = lake.episodes.from_predicate(
        source_table="model_outputs",
        source_where="prediction = 'pedestrian'",
        before_ns=500_000_000,
        after_ns=500_000_000,
        dry_run=True,
    )

    assert report.dry_run is True
    assert report.intervals_planned == 1
    assert report.frames_planned == 2
    assert report.episodes_written == 0
    assert {
        name: int(lake.table(name).version)
        for name in ("episodes", "videos", "observations", "transform_runs")
    } == before_versions
    assert lake.table("episodes").to_arrow().num_rows == 0
    assert lake.table("videos").to_arrow().num_rows == 0
    assert lake.table("transform_runs").to_arrow().num_rows == 0
    assert {row["episode_id"] for row in lake.table("observations").to_arrow().to_pylist()} == {
        None
    }


def test_from_predicate_observation_scans_do_not_project_payload_blob(
    tmp_path,
    monkeypatch,
):
    lake = _episode_lake(tmp_path / "robot.lance")
    lake.episodes.from_markers(
        start_event_types=["teleop_start"],
        stop_event_types=["success", "failure"],
    )
    observed_columns = []
    original_scan = episodes_module._scan_projected_rows

    def wrapped_scan(lake_arg, table_name, columns, *, where_sql=None):
        if table_name == "observations":
            observed_columns.append(tuple(columns))
        return original_scan(lake_arg, table_name, columns, where_sql=where_sql)

    monkeypatch.setattr(episodes_module, "_scan_projected_rows", wrapped_scan)

    lake.episodes.from_predicate(
        where="outcome = 'failure'",
        modalities=["image"],
        dry_run=True,
    )

    assert observed_columns
    assert all("payload_blob" not in columns for columns in observed_columns)
    assert all("payload_json" not in columns for columns in observed_columns)


def test_episode_derivation_overlap_policies_and_registry(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")
    query = lake.episodes.from_query(
        event_type="disengagement",
        before_ns=250_000_000,
        after_ns=250_000_000,
    )

    with pytest.raises(EpisodeError, match="already-owned"):
        lake.episodes.from_markers(
            start_event_types=["teleop_start"],
            stop_event_types=["success", "failure"],
        )

    dry_run = lake.episodes.dry_run(
        {
            "kind": "markers",
            "start_event_types": ["teleop_start"],
            "stop_event_types": ["success", "failure"],
        }
    )
    assert dry_run.episodes_planned == 2
    assert dry_run.frames_planned == 6
    assert [conflict.observation_id for conflict in dry_run.overlap_conflicts] == ["obs-4"]

    markers = lake.episodes.from_markers(
        start_event_types=["teleop_start"],
        stop_event_types=["success", "failure"],
        overlap_policy="preserve",
    )

    observations = {
        row["observation_id"]: row for row in lake.table("observations").to_arrow().to_pylist()
    }
    assert markers.frames_tagged == 5
    assert observations["obs-4"]["episode_id"] == query.episode_ids[0]
    assert observations["obs-3"]["episode_id"] in markers.episode_ids
    assert observations["obs-5"]["episode_id"] in markers.episode_ids

    summaries = {row.transform_id: row for row in lake.episodes.list_derivations()}
    assert summaries[query.transform_id].status == "active"
    assert summaries[query.transform_id].kind == "query"
    assert summaries[markers.transform_id].status == "active"
    assert summaries[markers.transform_id].kind == "markers"
    assert summaries[markers.transform_id].frame_count == 5
    assert summaries[markers.transform_id].params_hash

    detail = lake.episodes.show_derivation(markers.transform_id)
    assert detail.summary == summaries[markers.transform_id]
    assert detail.params["boundary_source"] == "markers"
    assert len(detail.episodes) == 2


def test_episode_derivation_supersede_and_clear_preserve_newer_owner(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")
    query = lake.episodes.from_query(
        event_type="disengagement",
        before_ns=250_000_000,
        after_ns=250_000_000,
    )

    replacement = lake.episodes.supersede(
        query.transform_id,
        {
            "kind": "predicate",
            "where": "observation_id = 'obs-4'",
        },
    )

    observations = {
        row["observation_id"]: row for row in lake.table("observations").to_arrow().to_pylist()
    }
    assert observations["obs-4"]["episode_id"] == replacement.episode_ids[0]
    summaries = {row.transform_id: row for row in lake.episodes.list_derivations()}
    assert summaries[query.transform_id].status == "superseded"
    assert summaries[query.transform_id].superseded_by == replacement.transform_id
    assert summaries[replacement.transform_id].status == "active"

    clear = lake.episodes.clear(query.transform_id)

    observations = {
        row["observation_id"]: row for row in lake.table("observations").to_arrow().to_pylist()
    }
    assert clear.episodes_removed == 1
    assert clear.frames_untagged == 0
    assert observations["obs-4"]["episode_id"] == replacement.episode_ids[0]
    assert {
        row["transform_id"] for row in lake.table("episodes").to_arrow().to_pylist()
    } == {replacement.transform_id}


def test_episode_derivation_rebuild_is_deterministic_after_clear(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")
    report = lake.episodes.from_markers(
        start_event_types=["teleop_start"],
        stop_event_types=["success", "failure"],
    )
    original_fingerprint = [
        (
            row["observation_id"],
            row["episode_id"],
            row["episode_index"],
            row["frame_index"],
        )
        for row in sorted(
            lake.table("observations").to_arrow().to_pylist(),
            key=lambda item: item["observation_id"],
        )
    ]

    clear = lake.episodes.clear(report.transform_id)
    rebuilt = lake.episodes.rebuild(report.transform_id)

    assert clear.frames_untagged == 6
    assert rebuilt.transform_id == report.transform_id
    assert rebuilt.episode_ids == report.episode_ids
    assert [
        (
            row["observation_id"],
            row["episode_id"],
            row["episode_index"],
            row["frame_index"],
        )
        for row in sorted(
            lake.table("observations").to_arrow().to_pylist(),
            key=lambda item: item["observation_id"],
        )
    ] == original_fingerprint
    summary = {
        row.transform_id: row for row in lake.episodes.list_derivations()
    }[report.transform_id]
    assert summary.status == "active"
    actions = [
        json.loads(row["params"])["action"]
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["kind"] == "episode-derivation-lifecycle"
    ]
    assert actions == ["clear", "rebuild"]


def test_from_scenarios_promotes_curated_ids_with_provenance(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")

    report = lake.episodes.from_scenarios(
        scenario_ids=["scn-episode-1", "scn-episode-0"],
        outcome_by_coverage_tag={"result:failure": "failure"},
    )

    assert report.boundary_source == "scenarios"
    assert report.episodes_written == 2
    assert report.frames_tagged == 6
    assert report.videos_written == 2

    episodes = sorted(
        lake.table("episodes").to_arrow().to_pylist(),
        key=lambda row: row["episode_index"],
    )
    assert [(row["from_timestamp_ns"], row["to_timestamp_ns"]) for row in episodes] == [
        (0, 1_000_000_000),
        (2_000_000_000, 3_000_000_000),
    ]
    assert [row["frame_count"] for row in episodes] == [3, 3]
    assert [row["task_id"] for row in episodes] == ["pick cube", "place cube"]
    assert [row["outcome"] for row in episodes] == [None, "failure"]

    provenance = [json.loads(row["provenance"]) for row in episodes]
    assert [item["source_scenario_ids"] for item in provenance] == [
        ["scn-episode-0"],
        ["scn-episode-1"],
    ]
    assert provenance[0]["source_observation_ids"] == ["obs-0", "obs-1", "obs-2"]

    observations = {
        row["observation_id"]: row for row in lake.table("observations").to_arrow().to_pylist()
    }
    assert [observations[f"obs-{index}"]["frame_index"] for index in range(3)] == [0, 1, 2]
    assert observations["obs-4"]["episode_id"] == episodes[1]["episode_id"]
    assert observations["obs-4"]["task_id"] == "place cube"
    assert observations["obs-4"]["outcome"] == "failure"


def test_from_intervals_jsonl_imports_episodes_frames_provenance_and_is_idempotent(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")
    manifest_path = tmp_path / "intervals.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "external_id": "incident-1",
                "run_id": "run-episodes",
                "from_timestamp_ns": 0,
                "to_timestamp_ns": 1_000_000_000,
                "task_id": "pick cube",
                "outcome": "success",
                "tags": ["partner", "review"],
                "metadata": {"reviewer": "ada"},
                "provenance": {"tool": "review-ui", "batch": "b1"},
            },
            {
                "external_id": "incident-2",
                "run_id": "run-episodes",
                "from_timestamp_ns": 2_000_000_000,
                "to_timestamp_ns": 3_000_000_000,
                "task_id": "place cube",
                "outcome": "failure",
            },
        ],
    )
    manifest = load_interval_manifest(manifest_path, format="jsonl")

    first = lake.episodes.from_intervals(
        manifest.records,
        source_uri=manifest.source_uri,
        source_sha256=manifest.sha256,
    )
    second = lake.episodes.from_intervals(
        manifest.records,
        source_uri=manifest.source_uri,
        source_sha256=manifest.sha256,
    )

    assert second.transform_id == first.transform_id
    assert first.boundary_source == "intervals"
    assert second.episodes_written == 2
    assert second.frames_tagged == 6
    assert second.videos_written == 2

    episodes = sorted(
        lake.table("episodes").to_arrow().to_pylist(),
        key=lambda row: row["episode_index"],
    )
    assert len(episodes) == 2
    assert [(row["from_timestamp_ns"], row["to_timestamp_ns"]) for row in episodes] == [
        (0, 1_000_000_000),
        (2_000_000_000, 3_000_000_000),
    ]
    assert [row["outcome"] for row in episodes] == ["success", "failure"]
    assert [row["task_id"] for row in episodes] == ["pick cube", "place cube"]
    assert [row["frame_count"] for row in episodes] == [3, 3]

    provenance = json.loads(episodes[0]["provenance"])
    assert provenance["external_id"] == "incident-1"
    assert provenance["tags"] == ["partner", "review"]
    assert provenance["metadata"] == {"reviewer": "ada"}
    assert provenance["source_provenance"] == {"batch": "b1", "tool": "review-ui"}
    assert provenance["source_payload"]["external_id"] == "incident-1"
    assert provenance["source_query"]["source_sha256"] == manifest.sha256

    observations = {
        row["observation_id"]: row for row in lake.table("observations").to_arrow().to_pylist()
    }
    assert [observations[f"obs-{index}"]["frame_index"] for index in range(3)] == [0, 1, 2]
    assert observations["obs-1"]["episode_id"] == episodes[0]["episode_id"]
    assert observations["obs-4"]["episode_id"] == episodes[1]["episode_id"]
    assert observations["obs-4"]["outcome"] == "failure"

    transforms = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == first.transform_id
    ]
    assert len(transforms) == 1
    assert transforms[0]["input_uris"] == [str(manifest_path)]
    params = json.loads(transforms[0]["params"])
    assert params["boundary_source"] == "intervals"
    assert params["source_uri"] == str(manifest_path)
    assert params["source_sha256"] == manifest.sha256
    assert params["intervals"][0]["external_id"] == "incident-1"
    assert params["episode_ids"] == sorted(first.episode_ids)


def test_interval_csv_loader_normalizes_time_unit_and_metadata(tmp_path):
    json_lake = _episode_lake(tmp_path / "json.lance")
    csv_lake = _episode_lake(tmp_path / "csv.lance")
    json_lake.episodes.from_intervals(
        [
            {
                "external_id": "csv-1",
                "run_id": "run-episodes",
                "from_timestamp_ns": 0,
                "to_timestamp_ns": 1_000_000_000,
                "task_id": "pick cube",
                "outcome": "success",
            },
            {
                "external_id": "csv-2",
                "run_id": "run-episodes",
                "from_timestamp_ns": 2_000_000_000,
                "to_timestamp_ns": 3_000_000_000,
                "task_id": "place cube",
                "outcome": "failure",
            },
        ]
    )
    csv_path = tmp_path / "intervals.csv"
    csv_path.write_text(
        "external_id,run_id,from_timestamp,to_timestamp,outcome,task_id,tags,"
        "metadata.reviewer,provenance\n"
        'csv-1,run-episodes,0,1,success,pick cube,"partner;review",ada,'
        '"{""tool"": ""sheet""}"\n'
        "csv-2,run-episodes,2,3,failure,place cube,,,\n"
    )
    manifest = load_interval_manifest(csv_path, format="csv")

    csv_lake.episodes.from_intervals(
        manifest.records,
        time_unit="s",
        source_uri=manifest.source_uri,
        source_sha256=manifest.sha256,
    )

    assert _episode_fingerprint(csv_lake) == _episode_fingerprint(json_lake)
    first = min(
        csv_lake.table("episodes").to_arrow().to_pylist(),
        key=lambda row: row["episode_index"],
    )
    provenance = json.loads(first["provenance"])
    assert provenance["tags"] == ["partner", "review"]
    assert provenance["metadata"] == {"reviewer": "ada"}
    assert provenance["source_provenance"] == {"tool": "sheet"}


@pytest.mark.parametrize(
    ("bad_row", "match"),
    [
        (
            {
                "external_id": "missing-run",
                "run_id": "missing",
                "from_timestamp_ns": 0,
                "to_timestamp_ns": 1_000_000_000,
            },
            "unknown run_id",
        ),
        (
            {
                "external_id": "inverted",
                "run_id": "run-episodes",
                "from_timestamp_ns": 2_000_000_000,
                "to_timestamp_ns": 1_000_000_000,
            },
            "earlier",
        ),
        (
            {
                "external_id": "empty",
                "run_id": "run-episodes",
                "from_timestamp_ns": 1_250_000_000,
                "to_timestamp_ns": 1_500_000_000,
            },
            "no observations",
        ),
    ],
)
def test_from_intervals_rejects_invalid_intervals_atomically(tmp_path, bad_row, match):
    lake = _episode_lake(tmp_path / "robot.lance")
    valid = {
        "external_id": "valid",
        "run_id": "run-episodes",
        "from_timestamp_ns": 0,
        "to_timestamp_ns": 1_000_000_000,
    }

    with pytest.raises(EpisodeError, match=match):
        lake.episodes.from_intervals([valid, bad_row])

    assert lake.table("episodes").to_arrow().num_rows == 0
    assert not lake.table("transform_runs").to_arrow().to_pylist()
    assert {row["episode_id"] for row in lake.table("observations").to_arrow().to_pylist()} == {
        None
    }


def test_from_intervals_rejects_duplicate_external_ids_atomically(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")

    with pytest.raises(EpisodeError, match="reuses external_id"):
        lake.episodes.from_intervals(
            [
                {
                    "external_id": "duplicate",
                    "run_id": "run-episodes",
                    "from_timestamp_ns": 0,
                    "to_timestamp_ns": 1_000_000_000,
                },
                {
                    "external_id": "duplicate",
                    "run_id": "run-episodes",
                    "from_timestamp_ns": 2_000_000_000,
                    "to_timestamp_ns": 3_000_000_000,
                },
            ]
        )

    assert lake.table("episodes").to_arrow().num_rows == 0
    assert not lake.table("transform_runs").to_arrow().to_pylist()


def test_from_intervals_allow_clipped_records_original_bounds_in_provenance(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")

    report = lake.episodes.from_intervals(
        [
            {
                "external_id": "clipped",
                "run_id": "run-episodes",
                "from_timestamp_ns": -500_000_000,
                "to_timestamp_ns": 3_500_000_000,
            }
        ],
        allow_clipped=True,
    )

    assert report.episodes_written == 1
    episode = lake.table("episodes").to_arrow().to_pylist()[0]
    assert episode["from_timestamp_ns"] == 0
    assert episode["to_timestamp_ns"] == 3_000_000_000
    assert episode["frame_count"] == 6
    provenance = json.loads(episode["provenance"])
    assert provenance["clipped"] == {
        "from_timestamp_ns": {"original": -500_000_000, "effective": 0},
        "to_timestamp_ns": {"original": 3_500_000_000, "effective": 3_000_000_000},
    }
    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == report.transform_id
    )
    params = json.loads(transform["params"])
    assert params["allow_clipped"] is True
    assert params["intervals"][0]["clipped"] == provenance["clipped"]


def test_from_scenarios_snapshot_uses_pinned_selection_and_is_idempotent(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")
    create_snapshot(
        lake,
        name="curated-v1",
        scenario_ids=["scn-episode-1", "scn-episode-0"],
        split_by="scenario",
    )
    original = next(
        row
        for row in lake.table("scenarios").to_arrow().to_pylist()
        if row["scenario_id"] == "scn-episode-0"
    )
    mutated = {
        **original,
        "end_time_ns": 3_000_000_000,
        "observation_ids": [f"obs-{index}" for index in range(6)],
        "observation_count": 6,
        "summary": "mutated after snapshot",
    }
    scenarios = lake.table("scenarios")
    scenarios.delete("scenario_id = 'scn-episode-0'")
    scenarios.add(pa.Table.from_pylist([mutated], schema=SCENARIOS_SCHEMA))

    first = lake.episodes.from_scenarios(snapshot_name="curated-v1")
    second = lake.episodes.from_scenarios(snapshot_name="curated-v1")

    assert second.transform_id == first.transform_id
    episodes = sorted(
        [
            row
            for row in lake.table("episodes").to_arrow().to_pylist()
            if row["transform_id"] == first.transform_id
        ],
        key=lambda row: row["episode_index"],
    )
    assert len(episodes) == 2
    assert [(row["from_timestamp_ns"], row["to_timestamp_ns"]) for row in episodes] == [
        (0, 1_000_000_000),
        (2_000_000_000, 3_000_000_000),
    ]
    assert [row["frame_count"] for row in episodes] == [3, 3]

    transforms = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == first.transform_id
    ]
    assert len(transforms) == 1
    params = json.loads(transforms[0]["params"])
    assert params["boundary_source"] == "scenarios"
    assert params["snapshot_dataset_id"].startswith("ds-")
    assert params["scenario_ids"] == ["scn-episode-0", "scn-episode-1"]


def test_from_scenarios_honors_observation_id_order_and_time_fallback(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-reordered",
                    "run_id": "run-episodes",
                    "start_time_ns": 0,
                    "end_time_ns": 1_000_000_000,
                    "window_ns": 1_000_000_000,
                    "is_partial": False,
                    "topics": ["/camera/front", "/joint_states"],
                    "observation_ids": ["obs-2", "obs-0", "obs-1"],
                    "observation_count": 3,
                    "scenario_type": "teleop",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": [],
                    "summary": "",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                },
                {
                    "scenario_id": "scn-fallback",
                    "run_id": "run-episodes",
                    "start_time_ns": 2_000_000_000,
                    "end_time_ns": 3_000_000_000,
                    "window_ns": 1_000_000_000,
                    "is_partial": False,
                    "topics": ["/camera/front", "/joint_states"],
                    "observation_ids": None,
                    "observation_count": 3,
                    "scenario_type": "teleop",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": [],
                    "summary": "",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                },
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )

    lake.episodes.from_scenarios(scenario_ids=["scn-reordered", "scn-fallback"])

    episodes_by_scenario = {
        json.loads(row["provenance"])["source_scenario_ids"][0]: row
        for row in lake.table("episodes").to_arrow().to_pylist()
    }
    observations = lake.table("observations").to_arrow().to_pylist()
    reordered = sorted(
        [
            row
            for row in observations
            if row["episode_id"] == episodes_by_scenario["scn-reordered"]["episode_id"]
        ],
        key=lambda row: row["frame_index"],
    )
    fallback = sorted(
        [
            row
            for row in observations
            if row["episode_id"] == episodes_by_scenario["scn-fallback"]["episode_id"]
        ],
        key=lambda row: row["frame_index"],
    )

    assert [row["observation_id"] for row in reordered] == ["obs-2", "obs-0", "obs-1"]
    assert [row["observation_id"] for row in fallback] == ["obs-3", "obs-4", "obs-5"]


def test_episode_resolves_ordered_frames_and_deterministic_window(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")
    lake.episodes.from_markers(
        start_event_types=["teleop_start"],
        stop_event_types=["success", "failure"],
    )
    first = min(
        lake.table("episodes").to_arrow().to_pylist(),
        key=lambda row: row["episode_index"],
    )
    episode = lake.episodes.get(first["episode_id"])

    assert [row["observation_id"] for row in episode.frames()] == ["obs-0", "obs-1", "obs-2"]
    window = episode.window(rate_hz=1.0, streams=["/camera/front", "/joint_states"])
    again = episode.window(rate_hz=1.0, streams=["/camera/front", "/joint_states"])

    assert window == again
    assert [row["timestamp_ns"] for row in window.rows] == [0, 1_000_000_000]
    assert window.rows[0]["streams"]["/camera/front"]["observation_id"] == "obs-0"
    assert window.rows[0]["streams"]["/joint_states"]["observation_id"] == "obs-1"
    assert window.rows[1]["streams"]["/camera/front"]["observation_id"] == "obs-2"


def test_episodes_from_query_cli_reports_transform(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _episode_lake(lake_path)

    result = runner.invoke(
        app,
        [
            "episodes",
            "from-query",
            "--lake",
            str(lake_path),
            "--event-type",
            "disengagement",
            "--before",
            "250ms",
            "--after",
            "250ms",
        ],
    )

    assert result.exit_code == 0
    assert "boundary source: query" in result.output
    assert "episodes: 1" in result.output
    assert "frames tagged: 1" in result.output


def test_episodes_mine_cli_reports_dry_run_intervals(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _episode_lake(lake_path)
    _add_model_outputs(lake)

    result = runner.invoke(
        app,
        [
            "episodes",
            "mine",
            "--lake",
            str(lake_path),
            "--source-table",
            "model_outputs",
            "--source-where",
            "prediction = 'pedestrian'",
            "--before",
            "500ms",
            "--after",
            "500ms",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "boundary source: predicate" in result.output
    assert "dry-run: true" in result.output
    assert "planned intervals: 1" in result.output
    assert "planned frames: 2" in result.output
    assert '"source_model_output_ids": ["out-pedestrian"]' in result.output
    assert lake.table("episodes").to_arrow().num_rows == 0


def test_episodes_from_scenarios_cli_reports_transform(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _episode_lake(lake_path)

    result = runner.invoke(
        app,
        [
            "episodes",
            "from-scenarios",
            "--lake",
            str(lake_path),
            "--scenario-id",
            "scn-episode-0",
            "--outcome",
            "success",
        ],
    )

    assert result.exit_code == 0
    assert "boundary source: scenarios" in result.output
    assert "episodes: 1" in result.output
    assert "frames tagged: 3" in result.output
    assert "transform: tfm-episodes-scenarios-" in result.output


def test_episodes_import_intervals_cli_reports_transform(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _episode_lake(lake_path)
    intervals = tmp_path / "intervals.jsonl"
    _write_jsonl(
        intervals,
        [
            {
                "external_id": "cli-1",
                "run_id": "run-episodes",
                "from_timestamp_ns": 0,
                "to_timestamp_ns": 1_000_000_000,
                "outcome": "success",
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "episodes",
            "import-intervals",
            "--lake",
            str(lake_path),
            "--file",
            str(intervals),
            "--format",
            "jsonl",
        ],
    )

    assert result.exit_code == 0
    assert "boundary source: intervals" in result.output
    assert "episodes: 1" in result.output
    assert "frames tagged: 3" in result.output
    assert "transform: tfm-episodes-intervals-" in result.output


def test_episodes_derivations_cli_lists_shows_dry_runs_and_clears(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _episode_lake(lake_path)
    report = lake.episodes.from_query(
        event_type="disengagement",
        before_ns=250_000_000,
        after_ns=250_000_000,
    )

    listed = runner.invoke(
        app,
        [
            "episodes",
            "derivations",
            "list",
            "--lake",
            str(lake_path),
        ],
    )
    shown = runner.invoke(
        app,
        [
            "episodes",
            "derivations",
            "show",
            report.transform_id,
            "--lake",
            str(lake_path),
        ],
    )
    dry_run = runner.invoke(
        app,
        [
            "episodes",
            "derivations",
            "dry-run",
            "markers",
            "--lake",
            str(lake_path),
            "--start-event",
            "teleop_start",
            "--stop-event",
            "success",
            "--stop-event",
            "failure",
        ],
    )
    cleared = runner.invoke(
        app,
        [
            "episodes",
            "derivations",
            "clear",
            report.transform_id,
            "--lake",
            str(lake_path),
        ],
    )

    assert listed.exit_code == 0
    assert '"kind": "query"' in listed.output
    assert shown.exit_code == 0
    assert f'"transform_id": "{report.transform_id}"' in shown.output
    assert dry_run.exit_code == 0
    assert "overlap conflicts: 1" in dry_run.output
    assert cleared.exit_code == 0
    assert '"action": "clear"' in cleared.output
    assert '"frames_untagged": 1' in cleared.output


def test_training_dataset_prefers_physical_episodes_when_pinned(tmp_path):
    lake = _episode_lake(tmp_path / "robot.lance")
    report = lake.episodes.from_markers(
        start_event_types=["teleop_start"],
        stop_event_types=["success", "failure"],
    )
    first = min(
        lake.table("episodes").to_arrow().to_pylist(),
        key=lambda row: row["episode_index"],
    )
    create_snapshot(
        lake,
        name="episode-demo",
        scenario_ids=["scn-episode-0"],
        split_by="scenario",
    )

    dataset = lake.training.dataset("episode-demo")

    assert dataset.num_episodes == 1
    assert len(dataset) == 3
    assert first["episode_id"] in report.episode_ids
    assert dataset[0]["episode_id"] == first["episode_id"]
    assert dataset[0]["scenario_id"] == "scn-episode-0"
    assert dataset.manifest.total_frames == 3
    assert {"episodes", "videos"} <= {item["table"] for item in dataset.manifest.table_versions}
