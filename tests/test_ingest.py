"""MCAP ingest mapping and provenance tests (backlog 0004).

The fixture (tests/fixtures/make_sample_mcap.py) carries 3 `/imu` json
messages and 2 `/camera/front` cbor messages, so every count below is
pinned to that layout.
"""

import json

import pytest

from lancedb_robotics.adapters import AdapterError
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake

BASE_NS = 1_700_000_000_000_000_000  # matches tests/fixtures/make_sample_mcap.py


@pytest.fixture
def sample_mcap(fixtures_dir):
    return fixtures_dir / "sample.mcap"


@pytest.fixture
def lake(tmp_path):
    return Lake.init(tmp_path / "robot.lance")


@pytest.fixture
def report(lake, sample_mcap):
    return ingest_mcap(lake, sample_mcap)


def test_ingest_writes_canonical_rows(report, lake):
    assert report.already_ingested is False
    assert report.rows_added == {
        "integration_sources": 1,
        "runs": 1,
        "observations": 5,
        # sample.mcap carries no attachments (backlog 0016): zero rows, no write.
        "attachments": 0,
        "events": 2,
        "transform_runs": 2,
    }
    for table, expected in report.rows_added.items():
        if table == "transform_runs":
            # ingest writes 2 (inspect + ingest); the default end-of-ingest
            # compaction records one more `maintenance` row (backlog 0180 / BUG-14).
            assert lake.table(table).count_rows() == expected + 1
            continue
        assert lake.table(table).count_rows() == expected


def test_ingest_reports_observation_counts_by_topic(report):
    assert report.observations_by_topic == {"/camera/front": 2, "/imu": 3}


def test_ingest_run_row_covers_the_log_time_range(report, lake):
    runs = lake.table("runs").to_arrow().to_pylist()
    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == report.run_id
    assert run["source"] == "mcap"
    assert run["source_id"] == report.source.source_id
    assert run["raw_uri"] == report.source.uri
    assert run["start_time_ns"] == BASE_NS
    assert run["end_time_ns"] == BASE_NS + 200_000_000
    assert run["duration_ns"] == 200_000_000


def test_ingest_observations_are_topic_and_time_indexed(report, lake):
    rows = lake.table("observations").to_arrow().to_pylist()
    assert len(rows) == 5
    imu = sorted((r for r in rows if r["topic"] == "/imu"), key=lambda r: r["timestamp_ns"])
    assert [r["timestamp_ns"] for r in imu] == [
        BASE_NS,
        BASE_NS + 100_000_000,
        BASE_NS + 200_000_000,
    ]
    assert [r["raw_sequence"] for r in imu] == [0, 1, 2]
    assert {r["modality"] for r in imu} == {"imu"}
    cam = [r for r in rows if r["topic"] == "/camera/front"]
    assert {r["modality"] for r in cam} == {"image"}
    ids = [r["observation_id"] for r in rows]
    assert len(set(ids)) == 5  # unique within the run


def test_ingest_extracts_run_boundary_events(report, lake):
    events = sorted(lake.table("events").to_arrow().to_pylist(), key=lambda r: r["timestamp_ns"])
    assert [e["event_type"] for e in events] == ["run_start", "run_end"]
    assert events[0]["timestamp_ns"] == BASE_NS
    assert events[1]["timestamp_ns"] == BASE_NS + 200_000_000
    assert {e["run_id"] for e in events} == {report.run_id}


def test_ingest_records_inspect_and_ingest_transform_lineage(report, lake):
    transforms = {t["kind"]: t for t in lake.table("transform_runs").to_arrow().to_pylist()}
    # inspect + ingest from the ingest itself; `maintenance` from the default
    # end-of-ingest compaction (backlog 0180 / BUG-14), which carries no source.
    assert set(transforms) == {"inspect", "ingest", "maintenance"}
    for kind in ("inspect", "ingest"):
        transform = transforms[kind]
        assert transform["status"] == "completed"
        assert transform["source_id"] == report.source.source_id
        assert transform["input_uris"] == [report.source.uri]
        json.loads(transform["params"])  # params must be valid JSON
    assert transforms["ingest"]["output_tables"] == ["runs", "observations", "events"]


def test_ingest_rows_trace_back_to_source_and_transform(report, lake):
    ingest_transform = next(
        t for t in lake.table("transform_runs").to_arrow().to_pylist() if t["kind"] == "ingest"
    )
    for table in ("runs", "observations", "events"):
        for row in lake.table(table).to_arrow().to_pylist():
            assert row["transform_id"] == ingest_transform["transform_id"]
    for row in lake.table("observations").to_arrow().to_pylist():
        assert row["run_id"] == report.run_id
        assert row["raw_uri"] == report.source.uri


def test_reingest_is_a_no_op(report, lake, sample_mcap):
    before_transforms = lake.table("transform_runs").count_rows()
    second = ingest_mcap(lake, sample_mcap)
    assert second.already_ingested is True
    assert second.run_id == report.run_id
    assert second.rows_added["transform_runs"] == 1
    assert sum(count for table, count in second.rows_added.items() if table != "transform_runs") == 0
    for table, expected in report.rows_added.items():
        if table == "transform_runs":
            assert lake.table(table).count_rows() == before_transforms + 1
            continue
        assert lake.table(table).count_rows() == expected
    duplicate = lake.table("transform_runs").to_arrow().to_pylist()[-1]
    assert duplicate["kind"] == "ingest"
    assert duplicate["status"] == "skipped-duplicate"
    params = json.loads(duplicate["params"])
    assert params["run_id"] == report.run_id
    assert params["rows_added"]["observations"] == 0


def test_incremental_ingest_appends_new_log_without_touching_existing_rows(
    lake, sample_mcap, fixtures_dir
):
    first = ingest_mcap(lake, sample_mcap)
    first_runs = {
        row["run_id"]: row for row in lake.table("runs").to_arrow().to_pylist()
    }
    first_observations = {
        row["observation_id"]: row for row in lake.table("observations").to_arrow().to_pylist()
    }

    second = ingest_mcap(lake, fixtures_dir / "records.mcap")

    assert second.already_ingested is False
    assert second.run_id != first.run_id
    assert lake.table("runs").count_rows() == 2
    assert lake.table("observations").count_rows() == first.message_count + second.message_count
    after_runs = {row["run_id"]: row for row in lake.table("runs").to_arrow().to_pylist()}
    after_observations = {
        row["observation_id"]: row for row in lake.table("observations").to_arrow().to_pylist()
    }
    assert after_runs[first.run_id] == first_runs[first.run_id]
    for observation_id, row in first_observations.items():
        assert after_observations[observation_id] == row
    assert all(
        row["run_id"] in {first.run_id, second.run_id}
        for row in lake.table("observations").to_arrow().to_pylist()
    )


def test_ingest_invalid_file_writes_nothing(lake, tmp_path):
    bogus = tmp_path / "bogus.mcap"
    bogus.write_bytes(b"this is not an mcap file")
    with pytest.raises(AdapterError):
        ingest_mcap(lake, bogus)
    for table in ("integration_sources", "runs", "observations", "events", "transform_runs"):
        assert lake.table(table).count_rows() == 0
