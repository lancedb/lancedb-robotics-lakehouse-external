"""Content-addressed, portable run/source ids (backlog 0013).

``run_id``/``source_id`` derive from file *content* (checksum), not the
absolute source path, so the same bytes ingested from any location curate to
the same ids and the same train/val/test split bucket — reproducible across
machines. The raw path is still recorded as ``raw_uri`` provenance.
"""

import shutil

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.enrich import enrich_scenarios
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.scenarios import create_scenario_windows


def _copy_fixture(fixtures_dir, dest_dir, name):
    """Copy the sample MCAP to ``dest_dir/<name>/sample.mcap`` (distinct path, same bytes)."""
    target_dir = dest_dir / name
    target_dir.mkdir(parents=True)
    target = target_dir / "sample.mcap"
    shutil.copyfile(fixtures_dir / "sample.mcap", target)
    return target


def _windowed_enriched(lake_path, mcap_path):
    lake = Lake.init(lake_path)
    report = ingest_mcap(lake, mcap_path)
    create_scenario_windows(lake, window_ns=50_000_000)
    enrich_scenarios(lake)
    return lake, report


def _ids(lake):
    obs = sorted(r["observation_id"] for r in lake.table("observations").to_arrow().to_pylist())
    scn = sorted(r["scenario_id"] for r in lake.table("scenarios").to_arrow().to_pylist())
    return obs, scn


def test_run_and_derived_ids_are_path_independent(tmp_path, fixtures_dir):
    """Same bytes from two absolute paths → identical run/observation/scenario ids."""
    path_a = _copy_fixture(fixtures_dir, tmp_path / "loc_a", "alpha")
    path_b = _copy_fixture(fixtures_dir, tmp_path / "loc_b", "beta")
    assert path_a != path_b  # genuinely different absolute paths

    lake_a, report_a = _windowed_enriched(tmp_path / "a.lance", path_a)
    lake_b, report_b = _windowed_enriched(tmp_path / "b.lance", path_b)

    assert report_a.run_id == report_b.run_id
    assert report_a.source.source_id == report_b.source.source_id

    obs_a, scn_a = _ids(lake_a)
    obs_b, scn_b = _ids(lake_b)
    assert obs_a == obs_b
    assert scn_a == scn_b


def test_raw_uri_still_records_the_actual_source_path(tmp_path, fixtures_dir):
    """Path stays out of the id key but is preserved as ``raw_uri`` provenance."""
    path_a = _copy_fixture(fixtures_dir, tmp_path / "loc_a", "alpha")
    path_b = _copy_fixture(fixtures_dir, tmp_path / "loc_b", "beta")

    lake_a, report_a = _windowed_enriched(tmp_path / "a.lance", path_a)
    lake_b, report_b = _windowed_enriched(tmp_path / "b.lance", path_b)

    assert report_a.source.uri == str(path_a.resolve())
    assert report_b.source.uri == str(path_b.resolve())
    run_a = lake_a.table("runs").to_arrow().to_pylist()[0]
    run_b = lake_b.table("runs").to_arrow().to_pylist()[0]
    assert run_a["raw_uri"] == str(path_a.resolve())
    assert run_b["raw_uri"] == str(path_b.resolve())


def test_snapshot_dataset_id_and_split_are_cross_machine_reproducible(tmp_path, fixtures_dir):
    """Snapshot built from each location → identical dataset_id AND split assignment."""
    path_a = _copy_fixture(fixtures_dir, tmp_path / "loc_a", "alpha")
    path_b = _copy_fixture(fixtures_dir, tmp_path / "loc_b", "beta")

    lake_a, _ = _windowed_enriched(tmp_path / "a.lance", path_a)
    lake_b, _ = _windowed_enriched(tmp_path / "b.lance", path_b)

    ids_a = sorted(r["scenario_id"] for r in lake_a.table("scenarios").to_arrow().to_pylist())
    ids_b = sorted(r["scenario_id"] for r in lake_b.table("scenarios").to_arrow().to_pylist())

    man_a = create_snapshot(lake_a, name="demo-v1", scenario_ids=ids_a)
    man_b = create_snapshot(lake_b, name="demo-v1", scenario_ids=ids_b)

    assert man_a.dataset_id == man_b.dataset_id
    assert man_a.split_assignments == man_b.split_assignments
