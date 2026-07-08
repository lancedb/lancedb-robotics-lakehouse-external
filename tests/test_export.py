"""Export manifest + MCAP clip export tests (backlog 0011)."""

import json
import shutil

import pytest
from mcap.reader import make_reader

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.enrich import enrich_scenarios
from lancedb_robotics.export import (
    MANIFEST_FILENAME,
    ExportError,
    export_snapshot,
)
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.scenarios import create_scenario_windows


def _snapshot_lake(lake_path, source_mcap, *, name="demo-v1"):
    lake = Lake.init(lake_path)
    ingest_mcap(lake, source_mcap)
    create_scenario_windows(lake, window_ns=50_000_000)
    enrich_scenarios(lake)
    ids = sorted(r["scenario_id"] for r in lake.table("scenarios").to_arrow().to_pylist())
    create_snapshot(lake, name=name, scenario_ids=ids)
    return lake


@pytest.fixture
def lake(tmp_path, fixtures_dir):
    return _snapshot_lake(tmp_path / "robot.lance", fixtures_dir / "sample.mcap")


# --- manifest + clip export -------------------------------------------------


def test_export_manifest_lists_clips_with_lineage(lake, tmp_path):
    out = tmp_path / "clips"
    manifest = export_snapshot(lake, "demo-v1", out_dir=out)

    assert manifest.dataset_id.startswith("ds-")
    assert manifest.format == "mcap"
    assert len(manifest.clips) == 4
    for clip in manifest.clips:
        assert clip.scenario_id.startswith("scn-")
        assert clip.observation_ids  # lineage back to observations
        assert clip.source_uri.endswith("sample.mcap")
        assert clip.end_time_ns >= clip.start_time_ns
        assert clip.status == "exported"
        assert clip.lossiness == "lossless-slice"
    # Manifest file is written into the output directory.
    written = json.loads((out / MANIFEST_FILENAME).read_text())
    assert written["dataset_id"] == manifest.dataset_id
    assert len(written["clips"]) == 4


def test_exported_clips_are_valid_mcap_within_window(lake, tmp_path):
    manifest = export_snapshot(lake, "demo-v1", out_dir=tmp_path / "clips")

    for clip in manifest.clips:
        # One exported message per source observation (reversible slice).
        assert clip.message_count == len(clip.observation_ids)
        with open(clip.out_path, "rb") as handle:
            messages = [(c.topic, m.log_time) for _, c, m in make_reader(handle).iter_messages()]
        assert messages  # non-empty clip
        for topic, log_time in messages:
            assert topic in clip.topics
            assert clip.start_time_ns <= log_time <= clip.end_time_ns


def test_export_records_transform_lineage(lake, tmp_path):
    manifest = export_snapshot(lake, "demo-v1", out_dir=tmp_path / "clips")

    transform = next(
        r
        for r in lake.table("transform_runs").to_arrow().to_pylist()
        if r["transform_id"] == manifest.transform_id
    )
    assert transform["kind"] == "export"
    params = json.loads(transform["params"])
    assert params["dataset_id"] == manifest.dataset_id
    assert params["format"] == "mcap"


def test_clips_carry_optional_external_links(lake, tmp_path):
    manifest = export_snapshot(lake, "demo-v1", out_dir=tmp_path / "clips")
    for clip in manifest.clips:
        assert isinstance(clip.external_links, list)
        assert clip.external_links  # at least one replay-tool hint


# --- plan-only + skipped capability -----------------------------------------


def test_plan_only_writes_manifest_without_clips(lake, tmp_path):
    out = tmp_path / "clips"
    manifest = export_snapshot(lake, "demo-v1", out_dir=out, plan_only=True)

    assert manifest.exported == 0
    assert manifest.planned == 4
    for clip in manifest.clips:
        assert clip.status == "planned"
        assert clip.lossiness == "plan-only"
    # Manifest exists, but no clip files were written.
    assert (out / MANIFEST_FILENAME).is_file()
    assert not list(out.glob("*.mcap"))


def test_skipped_capability_when_source_unreachable(tmp_path, fixtures_dir):
    # Ingest from a copy, then remove it so the raw source is unreachable.
    source = tmp_path / "moved.mcap"
    shutil.copy(fixtures_dir / "sample.mcap", source)
    lake = _snapshot_lake(tmp_path / "robot.lance", source)
    source.unlink()

    manifest = export_snapshot(lake, "demo-v1", out_dir=tmp_path / "clips")

    assert manifest.exported == 0
    assert manifest.skipped == 4
    for clip in manifest.clips:
        assert clip.status == "skipped"
        assert "source" in clip.reason.lower()
        assert clip.out_path is None


# --- errors -----------------------------------------------------------------


def test_export_unknown_snapshot_raises(lake, tmp_path):
    with pytest.raises(ExportError):
        export_snapshot(lake, "ghost", out_dir=tmp_path / "clips")


def test_export_unknown_format_raises(lake, tmp_path):
    with pytest.raises(ExportError):
        export_snapshot(lake, "demo-v1", out_dir=tmp_path / "clips", fmt="rosbag")
