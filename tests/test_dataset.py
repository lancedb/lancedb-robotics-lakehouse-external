"""Dataset snapshot tests (backlog 0009)."""

import json

import pytest

from lancedb_robotics.dataset import (
    DatasetError,
    create_snapshot,
)
from lancedb_robotics.enrich import enrich_scenarios
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.search import HYBRID, last_search, record_search, search_scenarios


def _build_lake(path):
    lake = Lake.init(path)
    return lake


def _windowed_enriched(path, fixtures_dir):
    lake = Lake.init(path)
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=50_000_000)
    enrich_scenarios(lake)
    return lake


@pytest.fixture
def lake(tmp_path, fixtures_dir):
    return _windowed_enriched(tmp_path / "robot.lance", fixtures_dir)


def _all_scenario_ids(lake):
    return sorted(r["scenario_id"] for r in lake.table("scenarios").to_arrow().to_pylist())


def _snapshot_row(lake, dataset_id):
    return next(
        r
        for r in lake.table("dataset_snapshots").to_arrow().to_pylist()
        if r["dataset_id"] == dataset_id
    )


# --- manifest contents ------------------------------------------------------


def test_create_snapshot_from_explicit_ids_records_manifest(lake):
    ids = _all_scenario_ids(lake)
    manifest = create_snapshot(lake, name="demo-v1", scenario_ids=ids)

    assert manifest.name == "demo-v1"
    assert manifest.dataset_id.startswith("ds-")
    assert set(manifest.scenario_ids) == set(ids)

    row = _snapshot_row(lake, manifest.dataset_id)
    assert row["name"] == "demo-v1"
    assert row["transform_id"] == manifest.transform_id
    spec = json.loads(row["query_spec"])
    assert sorted(spec["scenario_ids"]) == ids
    tables = {tv["table"] for tv in row["table_versions"]}
    assert {"scenarios", "runs", "observations", "keyframe_map_artifacts"} <= tables


def test_snapshot_records_transform_lineage(lake):
    manifest = create_snapshot(lake, name="demo-v1", scenario_ids=_all_scenario_ids(lake))

    transform = next(
        r
        for r in lake.table("transform_runs").to_arrow().to_pylist()
        if r["transform_id"] == manifest.transform_id
    )
    assert transform["kind"] == "dataset-snapshot"
    assert transform["output_tables"] == ["dataset_snapshots"]


# --- reproducibility + split stability --------------------------------------


def test_snapshot_is_reproducible_for_fixed_inputs(tmp_path, fixtures_dir):
    lake_a = _windowed_enriched(tmp_path / "a.lance", fixtures_dir)
    lake_b = _windowed_enriched(tmp_path / "b.lance", fixtures_dir)

    man_a = create_snapshot(lake_a, name="demo-v1", scenario_ids=_all_scenario_ids(lake_a))
    man_b = create_snapshot(lake_b, name="demo-v1", scenario_ids=_all_scenario_ids(lake_b))

    assert man_a.dataset_id == man_b.dataset_id
    assert man_a.split_assignments == man_b.split_assignments


def test_split_by_run_keeps_a_run_together(lake):
    manifest = create_snapshot(
        lake, name="demo", scenario_ids=_all_scenario_ids(lake), split_by="run"
    )
    # The fixture is a single run, so every scenario lands in one split.
    assert len(set(manifest.split_assignments.values())) == 1


def test_split_by_scenario_is_stable(tmp_path, fixtures_dir):
    lake_a = _windowed_enriched(tmp_path / "a.lance", fixtures_dir)
    lake_b = _windowed_enriched(tmp_path / "b.lance", fixtures_dir)

    man_a = create_snapshot(
        lake_a, name="d", scenario_ids=_all_scenario_ids(lake_a), split_by="scenario"
    )
    man_b = create_snapshot(
        lake_b, name="d", scenario_ids=_all_scenario_ids(lake_b), split_by="scenario"
    )

    assert man_a.split_assignments == man_b.split_assignments
    assert sum(man_a.split_counts.values()) == len(man_a.scenario_ids)


def test_snapshot_rerun_is_idempotent(lake):
    ids = _all_scenario_ids(lake)
    first = create_snapshot(lake, name="demo-v1", scenario_ids=ids)
    second = create_snapshot(lake, name="demo-v1", scenario_ids=ids)

    assert second.dataset_id == first.dataset_id
    rows = [
        r
        for r in lake.table("dataset_snapshots").to_arrow().to_pylist()
        if r["dataset_id"] == first.dataset_id
    ]
    assert len(rows) == 1


# --- selection errors -------------------------------------------------------


def test_create_snapshot_rejects_unknown_scenarios(lake):
    with pytest.raises(DatasetError):
        create_snapshot(lake, name="x", scenario_ids=["scn-does-not-exist"])


def test_create_snapshot_requires_a_selection(lake):
    with pytest.raises(DatasetError):
        create_snapshot(lake, name="x", scenario_ids=[])


# --- from-search-last provenance --------------------------------------------


def test_record_and_resolve_last_search(lake):
    text_ids = [r.scenario_id for r in search_scenarios(lake, mode="text", query="imu")]
    record_search(lake, mode="text", query="imu", where=None, limit=10, scenario_ids=text_ids)
    hybrid_ids = [r.scenario_id for r in search_scenarios(lake, mode=HYBRID, query="camera")]
    record_search(lake, mode=HYBRID, query="camera", where=None, limit=10, scenario_ids=hybrid_ids)

    recorded = last_search(lake)
    assert recorded["mode"] == HYBRID
    assert recorded["query"] == "camera"
    assert recorded["scenario_ids"] == hybrid_ids


def test_snapshot_from_last_search_uses_recorded_ids(lake):
    ids = [r.scenario_id for r in search_scenarios(lake, mode="text", query="camera")]
    record_search(lake, mode="text", query="camera", where=None, limit=10, scenario_ids=ids)
    recorded = last_search(lake)

    manifest = create_snapshot(
        lake, name="demo-v1", scenario_ids=recorded["scenario_ids"], source=recorded
    )

    assert set(manifest.scenario_ids) == set(ids)
    spec = json.loads(_snapshot_row(lake, manifest.dataset_id)["query_spec"])
    assert spec["source"]["query"] == "camera"
