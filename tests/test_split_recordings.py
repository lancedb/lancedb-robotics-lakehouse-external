"""Split / multi-file (directory) recording tests (backlog 0019).

A rosbag2-style recording split into a directory of shards must become **one
logical run**: one content-addressed ``run_id`` over the ordered shard
checksums, per-topic ``sequence`` continuous across shard boundaries, and each
observation's ``raw_uri`` pointing at the specific shard. The fixture
(tests/fixtures/make_split_recording.py) is three shards of nine ``/imu`` +
nine ``/gps`` json messages, so every count below is pinned to that layout.
"""

import importlib.util
import shutil
from pathlib import Path

import pytest
from mcap.reader import make_reader

from lancedb_robotics.adapters import AdapterError, get_adapter
from lancedb_robotics.export import export_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.recordings import (
    export_window,
    inspect_recording,
    inspect_source,
    resolve_recording,
    resolve_shards,
)
from lancedb_robotics.sources import recording_content_key


def _load_make():
    path = Path(__file__).parent / "fixtures" / "make_split_recording.py"
    spec = importlib.util.spec_from_file_location("make_split_recording", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MK = _load_make()
BASE_NS = MK.BASE_NS
STEP_NS = MK.STEP_NS
TOTAL_TICKS = MK.TOTAL_TICKS  # 9
FULL_START = BASE_NS
FULL_END = BASE_NS + (TOTAL_TICKS - 1) * STEP_NS


@pytest.fixture
def recording_dir(fixtures_dir) -> Path:
    return fixtures_dir / "split_recording"


@pytest.fixture
def bare_dir(recording_dir, tmp_path) -> Path:
    """The same shards copied without a metadata.yaml (name-ordering fallback)."""
    dest = tmp_path / "bare_recording"
    dest.mkdir()
    for name in MK.SHARD_NAMES:
        shutil.copyfile(recording_dir / name, dest / name)
    return dest


@pytest.fixture
def combined_mcap(tmp_path) -> Path:
    """A single file holding the same nine ticks the recording is split across."""
    path = tmp_path / "combined.mcap"
    path.write_bytes(MK.build_combined_bytes())
    return path


@pytest.fixture
def lake(tmp_path) -> Lake:
    return Lake.init(tmp_path / "robot.lance")


# --- resolution -------------------------------------------------------------


def test_directory_resolves_to_ordered_shards(recording_dir):
    plan = resolve_shards(recording_dir)
    assert plan.is_split is True
    assert [p.name for p in plan.paths] == MK.SHARD_NAMES
    assert plan.declared_start_ns == FULL_START
    assert plan.declared_end_ns == FULL_END
    assert plan.storage_identifier == "mcap"


def test_metadata_yaml_path_resolves_like_its_directory(recording_dir):
    by_file = resolve_shards(recording_dir / "metadata.yaml")
    by_dir = resolve_shards(recording_dir)
    assert by_file.paths == by_dir.paths
    assert by_file.is_split is True


def test_bare_directory_falls_back_to_name_ordering(bare_dir):
    plan = resolve_shards(bare_dir)
    assert plan.is_split is True
    assert [p.name for p in plan.paths] == MK.SHARD_NAMES  # lexicographic == declared
    assert plan.metadata_path is None
    assert plan.declared_start_ns is None  # no manifest -> no declared range


def test_single_file_is_not_split(fixtures_dir):
    plan = resolve_shards(fixtures_dir / "sample.mcap")
    assert plan.is_split is False
    assert len(plan.paths) == 1


def test_missing_path_and_empty_directory_raise(tmp_path):
    with pytest.raises(AdapterError):
        resolve_shards(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AdapterError):
        resolve_shards(empty)


# --- inspect aggregate ------------------------------------------------------


def test_inspect_reports_merged_topics_counts_and_range(recording_dir):
    report = inspect_source(recording_dir)
    assert report["is_split"] is True
    assert report["shard_count"] == 3
    assert report["message_count"] == TOTAL_TICKS * 2  # 18
    assert report["channel_count"] == 2
    assert report["schema_count"] == 2
    assert report["start_time_ns"] == FULL_START
    assert report["end_time_ns"] == FULL_END
    by_topic = {t["topic"]: t for t in report["topics"]}
    assert by_topic["/imu"]["message_count"] == TOTAL_TICKS
    assert by_topic["/gps"]["message_count"] == TOTAL_TICKS
    assert by_topic["/imu"]["start_time_ns"] == FULL_START
    assert by_topic["/imu"]["end_time_ns"] == FULL_END


def test_inspect_aggregate_equals_single_file_concatenation(recording_dir, combined_mcap):
    """Acceptance: directory aggregates equal a single-file inspect of the concat."""
    merged = inspect_recording(recording_dir)
    single = get_adapter("mcap").inspect(combined_mcap)

    assert merged["message_count"] == single["message_count"]
    assert merged["start_time_ns"] == single["start_time_ns"]
    assert merged["end_time_ns"] == single["end_time_ns"]
    assert merged["duration_ns"] == single["duration_ns"]
    assert merged["channel_count"] == single["channel_count"]
    assert merged["schema_count"] == single["schema_count"]

    def topic_view(report):
        return {
            t["topic"]: (t["message_count"], t["start_time_ns"], t["end_time_ns"])
            for t in report["topics"]
        }

    assert topic_view(merged) == topic_view(single)


def test_inspect_reports_shard_inventory_and_boundary_gaps(recording_dir):
    report = inspect_source(recording_dir)
    shards = report["shards"]
    assert [s["name"] for s in shards] == MK.SHARD_NAMES
    assert all(s["readable"] for s in shards)
    assert [s["message_count"] for s in shards] == [6, 6, 6]  # 3 imu + 3 gps each
    # Three shards -> two boundaries; the inter-shard step is one tick.
    gaps = report["gaps"]
    assert [g["kind"] for g in gaps] == ["gap", "gap"]
    assert [g["delta_ns"] for g in gaps] == [STEP_NS, STEP_NS]
    assert [(g["after_shard"], g["before_shard"]) for g in gaps] == [(0, 1), (1, 2)]


def test_inspect_source_leaves_single_file_unchanged(fixtures_dir):
    report = inspect_source(fixtures_dir / "sample.mcap")
    assert "is_split" not in report
    assert report == get_adapter("mcap").inspect(fixtures_dir / "sample.mcap")


# --- ingest as one run ------------------------------------------------------


def test_ingest_directory_is_exactly_one_run(lake, recording_dir):
    report = ingest_mcap(lake, recording_dir)
    assert report.already_ingested is False
    assert lake.table("runs").count_rows() == 1
    assert report.rows_added["runs"] == 1
    assert report.rows_added["observations"] == TOTAL_TICKS * 2
    assert report.observations_by_topic == {"/gps": TOTAL_TICKS, "/imu": TOTAL_TICKS}
    run = lake.table("runs").to_arrow().to_pylist()[0]
    assert run["run_id"] == report.run_id
    assert run["start_time_ns"] == FULL_START
    assert run["end_time_ns"] == FULL_END


def test_run_id_is_content_addressed_over_ordered_shards(lake, recording_dir):
    report = ingest_mcap(lake, recording_dir)
    recording = resolve_recording(recording_dir)
    expected = f"run-{recording_content_key(recording.checksums)[:16]}"
    assert report.run_id == expected
    assert report.source.source_id == f"src-{recording_content_key(recording.checksums)[:16]}"


def test_per_topic_sequence_is_continuous_across_shard_boundaries(lake, recording_dir):
    report = ingest_mcap(lake, recording_dir)
    rows = lake.table("observations").to_arrow().to_pylist()
    for topic in ("/imu", "/gps"):
        seqs = sorted(r["raw_sequence"] for r in rows if r["topic"] == topic)
        assert seqs == list(range(TOTAL_TICKS))  # 0..8, no resets at boundaries
    ids = [r["observation_id"] for r in rows]
    assert len(set(ids)) == len(ids) == TOTAL_TICKS * 2  # unique within the one run
    assert {r["run_id"] for r in rows} == {report.run_id}


def test_each_observation_raw_uri_points_at_its_shard(lake, recording_dir):
    ingest_mcap(lake, recording_dir)
    rows = lake.table("observations").to_arrow().to_pylist()
    # Tick t lives in shard t // 3; its raw_uri must name that shard file.
    for row in rows:
        tick = (row["raw_log_time_ns"] - BASE_NS) // STEP_NS
        assert Path(row["raw_uri"]).name == MK.SHARD_NAMES[tick // 3]
    # The run row's raw_uri is the recording root (the directory) for export.
    run = lake.table("runs").to_arrow().to_pylist()[0]
    assert run["raw_uri"] == str(recording_dir.resolve())


def test_reingesting_the_directory_is_idempotent(lake, recording_dir):
    first = ingest_mcap(lake, recording_dir)
    second = ingest_mcap(lake, recording_dir)
    assert second.already_ingested is True
    assert second.run_id == first.run_id
    assert second.rows_added["transform_runs"] == 1
    assert all(
        count == 0 for table, count in second.rows_added.items() if table != "transform_runs"
    )
    assert lake.table("runs").count_rows() == 1


def test_run_id_is_independent_of_manifest_and_listing_order(tmp_path, recording_dir, bare_dir):
    """Same shards, with vs without a manifest, curate to the same run_id."""
    lake_meta = Lake.init(tmp_path / "meta.lance")
    lake_bare = Lake.init(tmp_path / "bare.lance")
    assert ingest_mcap(lake_meta, recording_dir).run_id == ingest_mcap(lake_bare, bare_dir).run_id


def test_source_is_registered_as_one_recording(lake, recording_dir):
    report = ingest_mcap(lake, recording_dir)
    sources = lake.table("integration_sources").to_arrow().to_pylist()
    assert len(sources) == 1
    row = sources[0]
    assert row["source_id"] == report.source.source_id
    assert row["kind"] == "recording"
    assert row["uri"] == str(recording_dir.resolve())
    metadata = {entry["key"]: entry["value"] for entry in row["metadata"]}
    assert metadata["shard_count"] == "3"
    assert metadata["schema:/imu"] == "sample.Imu:jsonschema"
    # Each shard is recorded with its checksum so the run's provenance is exact.
    for index, name in enumerate(MK.SHARD_NAMES):
        assert metadata[f"shard:{index}"].startswith(f"{name}:")


def test_ingest_records_shard_inventory_in_transform_lineage(lake, recording_dir):
    import json

    ingest_mcap(lake, recording_dir)
    ingest = next(
        t for t in lake.table("transform_runs").to_arrow().to_pylist() if t["kind"] == "ingest"
    )
    params = json.loads(ingest["params"])
    assert params["recording"]["shard_count"] == 3
    assert [s["name"] for s in params["recording"]["shards"]] == MK.SHARD_NAMES
    assert len(ingest["input_uris"]) == 3  # one per shard


# --- export across shard boundaries -----------------------------------------


def test_export_window_spanning_two_shards_is_one_clip(recording_dir, tmp_path):
    """Acceptance: a window spanning a shard boundary yields one valid clip."""
    plan = resolve_shards(recording_dir)
    # Ticks 1-4: ticks 1,2 are in shard 0, ticks 3,4 are in shard 1.
    start, end = BASE_NS + STEP_NS, BASE_NS + 4 * STEP_NS
    out = tmp_path / "clip.mcap"
    result = export_window(
        plan.paths, start_time_ns=start, end_time_ns=end, out_path=out, topics=("/imu", "/gps")
    )
    with out.open("rb") as handle:
        messages = [(c.topic, m.log_time) for _, c, m in make_reader(handle).iter_messages()]
    # Four ticks * two topics, all within the window, spanning the boundary.
    assert result["message_count"] == len(messages) == 8
    log_times = {t for _, t in messages}
    assert all(start <= t <= end for t in log_times)
    assert min(log_times) < BASE_NS + 3 * STEP_NS <= max(log_times)  # truly crosses shard 0->1


def test_export_snapshot_from_split_recording_produces_clips(lake, recording_dir, tmp_path):
    """End-to-end narrative: a split recording snapshots and exports as one run."""
    from lancedb_robotics.dataset import create_snapshot
    from lancedb_robotics.enrich import enrich_scenarios
    from lancedb_robotics.scenarios import create_scenario_windows

    ingest_mcap(lake, recording_dir)
    create_scenario_windows(lake, window_ns=FULL_END - FULL_START)  # one window over the whole run
    enrich_scenarios(lake)
    ids = sorted(r["scenario_id"] for r in lake.table("scenarios").to_arrow().to_pylist())
    create_snapshot(lake, name="split-v1", scenario_ids=ids)

    manifest = export_snapshot(lake, "split-v1", out_dir=tmp_path / "clips")
    assert manifest.exported >= 1
    spanning = None
    for clip in manifest.clips:
        assert clip.source_uri == str(recording_dir.resolve())
        assert clip.status == "exported"
        with open(clip.out_path, "rb") as handle:
            times = [m.log_time for _, _, m in make_reader(handle).iter_messages()]
        assert times  # a non-empty, valid clip
        assert all(clip.start_time_ns <= t <= clip.end_time_ns for t in times)
        if min(times) <= FULL_START and max(times) >= FULL_END:
            spanning = clip
    # At least one exported clip spans the full run (i.e. crosses every shard).
    assert spanning is not None


# --- integrity across shards ------------------------------------------------


def test_damaged_shard_quarantines_the_one_run_keeping_recovered(lake, fixtures_dir, tmp_path):
    """A truncated shard mid-recording quarantines the run but keeps the prefix."""
    rec = tmp_path / "damaged_recording"
    rec.mkdir()
    shutil.copyfile(fixtures_dir / "split_recording" / "recording_0.mcap", rec / "recording_0.mcap")
    shutil.copyfile(fixtures_dir / "truncated.mcap", rec / "recording_1.mcap")

    report = ingest_mcap(lake, rec)
    assert lake.table("runs").count_rows() == 1  # still exactly one run
    assert report.quarantined is True
    assert report.recovered_count > 0
    # The healthy shard's messages survive alongside the recovered prefix.
    assert report.message_count >= 6
    run = lake.table("runs").to_arrow().to_pylist()[0]
    assert "quarantined" in run["quality_flags"]
    # The run's range covers every observation it actually wrote.
    times = [o["raw_log_time_ns"] for o in lake.table("observations").to_arrow().to_pylist()]
    assert run["start_time_ns"] <= min(times)
    assert run["end_time_ns"] >= max(times)
