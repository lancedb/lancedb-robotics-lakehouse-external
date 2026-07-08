"""Scenario windowing tests (backlog 0006)."""

import json

import pytest

from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.scenarios import (
    ScenarioError,
    create_scenario_windows,
    parse_duration_ns,
)

BASE_NS = 1_700_000_000_000_000_000


@pytest.fixture
def lake(tmp_path, fixtures_dir):
    opened = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(opened, fixtures_dir / "sample.mcap")
    return opened


def _scenario_rows(lake):
    return sorted(lake.table("scenarios").to_arrow().to_pylist(), key=lambda r: r["start_time_ns"])


def test_parse_duration_ns_accepts_cli_units():
    assert parse_duration_ns("5s") == 5_000_000_000
    assert parse_duration_ns("100ms") == 100_000_000
    assert parse_duration_ns("250000ns") == 250_000


def test_parse_duration_ns_rejects_invalid_values():
    with pytest.raises(ScenarioError):
        parse_duration_ns("five")
    with pytest.raises(ScenarioError):
        parse_duration_ns("0s")


def test_dense_streams_produce_deterministic_boundaries(lake):
    report = create_scenario_windows(lake, window_ns=100_000_000)

    assert report.rows_added == 2
    rows = _scenario_rows(lake)
    assert [(r["start_time_ns"], r["end_time_ns"]) for r in rows] == [
        (BASE_NS, BASE_NS + 100_000_000),
        (BASE_NS + 100_000_000, BASE_NS + 200_000_000),
    ]
    assert [r["observation_count"] for r in rows] == [2, 3]
    assert rows[-1]["observation_ids"][-1].endswith("/imu:000002")


def test_sparse_topic_filter_still_uses_run_anchored_boundaries(lake):
    report = create_scenario_windows(lake, window_ns=75_000_000, topics=("/camera/front",))

    assert report.rows_added == 2
    rows = _scenario_rows(lake)
    assert [(r["start_time_ns"], r["end_time_ns"]) for r in rows] == [
        (BASE_NS, BASE_NS + 75_000_000),
        (BASE_NS + 150_000_000, BASE_NS + 200_000_000),
    ]
    assert [r["topics"] for r in rows] == [["/camera/front"], ["/camera/front"]]
    assert [r["observation_count"] for r in rows] == [1, 1]


def test_partial_final_window_behavior_is_explicit(lake):
    create_scenario_windows(lake, window_ns=75_000_000, include_partial=True)
    rows = _scenario_rows(lake)

    assert [(r["end_time_ns"] - r["start_time_ns"], r["is_partial"]) for r in rows] == [
        (75_000_000, False),
        (75_000_000, False),
        (50_000_000, True),
    ]
    assert "partial:true" in rows[-1]["coverage_tags"]


def test_drop_partial_final_window(lake):
    report = create_scenario_windows(lake, window_ns=75_000_000, include_partial=False)

    assert report.rows_added == 2
    assert [r["is_partial"] for r in _scenario_rows(lake)] == [False, False]


def test_scenario_rows_preserve_lineage_to_source_observations(lake):
    report = create_scenario_windows(lake, window_ns=100_000_000)

    scenario = _scenario_rows(lake)[0]
    assert scenario["transform_id"] == report.transform_id
    assert scenario["source"] == "scenario-windowing"
    assert scenario["scenario_type"] == "fixed-window"
    assert scenario["topics"] == ["/camera/front", "/imu"]
    assert scenario["observation_count"] == len(scenario["observation_ids"])
    observation_ids = {
        row["observation_id"] for row in lake.table("observations").to_arrow().to_pylist()
    }
    assert set(scenario["observation_ids"]).issubset(observation_ids)


def test_transform_lineage_records_windowing_params(lake):
    report = create_scenario_windows(
        lake, window_ns=100_000_000, topics=("/imu",), include_partial=False
    )

    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == report.transform_id
    )
    assert transform["kind"] == "scenario-windowing"
    assert transform["output_tables"] == ["scenarios"]
    params = json.loads(transform["params"])
    assert params["include_partial"] is False
    assert params["kind"] == "scenario-windowing"
    assert params["topics"] == ["/imu"]
    assert params["window_ns"] == 100_000_000
    assert params["rows_added"] == report.rows_added
    assert len(params["scenario_ids"]) == report.rows_added


def test_rerun_keeps_matching_transform_rows_in_place(lake):
    first = create_scenario_windows(lake, window_ns=100_000_000)
    before = {row["scenario_id"]: row for row in lake.table("scenarios").to_arrow().to_pylist()}
    second = create_scenario_windows(lake, window_ns=100_000_000)

    assert second.transform_id == first.transform_id
    assert second.rows_added == 0
    assert second.rows_replaced == 0
    assert lake.table("scenarios").count_rows() == 2
    after = {row["scenario_id"]: row for row in lake.table("scenarios").to_arrow().to_pylist()}
    assert after == before
    matching_transforms = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == first.transform_id
    ]
    assert len(matching_transforms) == 1


def test_create_scenario_windows_appends_new_runs_without_touching_existing_rows(
    tmp_path, fixtures_dir
):
    lake = Lake.init(tmp_path / "robot.lance")
    first = ingest_mcap(lake, fixtures_dir / "sample.mcap")
    initial = create_scenario_windows(lake, window_ns=100_000_000)
    before = {row["scenario_id"]: row for row in lake.table("scenarios").to_arrow().to_pylist()}

    second = ingest_mcap(lake, fixtures_dir / "records.mcap")
    appended = create_scenario_windows(lake, window_ns=100_000_000)

    assert second.run_id != first.run_id
    assert appended.rows_replaced == 0
    assert appended.rows_added == 2
    after = {row["scenario_id"]: row for row in lake.table("scenarios").to_arrow().to_pylist()}
    for scenario_id, row in before.items():
        assert after[scenario_id] == row
    assert lake.table("scenarios").count_rows() == initial.rows_added + appended.rows_added
    assert {row["run_id"] for row in after.values()} == {first.run_id, second.run_id}
