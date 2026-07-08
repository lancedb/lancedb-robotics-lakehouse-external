"""Summary-less / unindexed MCAP support (backlog 0018).

A live, append-only, or never-finalized recording has no summary section: the
seeking index is gone, but the bytes are valid and the messages are readable by
a linear scan. The adapter must inspect, ingest, and export such a file exactly
like its finalized twin (counts, time range, topics, canonical rows), marking
the scan-derived stats ``indexed: False`` -- distinct from truncation (0017),
which is still rejected/quarantined.

Fixtures: tests/fixtures/make_summaryless_mcap.py writes summaryless.mcap and
its finalized twin summaryless_indexed.mcap from the same messages.
"""

import pytest
from mcap.reader import make_reader

from lancedb_robotics.adapters import AdapterError, get_adapter
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake

BASE_NS = 1_700_000_000_000_000_000  # matches make_summaryless_mcap.py


@pytest.fixture
def summaryless(fixtures_dir):
    return fixtures_dir / "summaryless.mcap"


@pytest.fixture
def indexed_twin(fixtures_dir):
    return fixtures_dir / "summaryless_indexed.mcap"


# --- inspect ----------------------------------------------------------------


def test_summaryless_fixture_really_has_no_summary(summaryless, indexed_twin):
    # Guard the premise: the fixture is genuinely unindexed, the twin is not.
    with summaryless.open("rb") as handle:
        assert make_reader(handle).get_summary() is None
    with indexed_twin.open("rb") as handle:
        assert make_reader(handle).get_summary() is not None


def test_inspect_summaryless_matches_indexed_twin(summaryless, indexed_twin):
    adapter = get_adapter("mcap")
    scanned = adapter.inspect(summaryless)
    indexed = adapter.inspect(indexed_twin)

    # Stats are scan-derived vs summary-derived, but otherwise identical.
    assert scanned["indexed"] is False
    assert indexed["indexed"] is True
    for key in (
        "message_count",
        "schema_count",
        "channel_count",
        "start_time_ns",
        "end_time_ns",
        "duration_ns",
    ):
        assert scanned[key] == indexed[key], key
    # Topics carry encoding, schema, per-topic counts, time ranges, and decode
    # capability -- all recovered from the scan must match the indexed copy.
    assert scanned["topics"] == indexed["topics"]


def test_inspect_summaryless_reports_expected_metadata(summaryless):
    report = get_adapter("mcap").inspect(summaryless)
    assert report["message_count"] == 5
    assert report["schema_count"] == 2
    assert report["channel_count"] == 2
    assert report["start_time_ns"] == BASE_NS
    assert report["end_time_ns"] == BASE_NS + 200_000_000
    assert report["duration_ns"] == 200_000_000

    by_topic = {t["topic"]: t for t in report["topics"]}
    assert set(by_topic) == {"/imu", "/camera/front"}
    assert by_topic["/imu"]["message_count"] == 3
    assert by_topic["/imu"]["can_decode"] is True
    assert by_topic["/camera/front"]["message_count"] == 2
    # cbor is decoded schema-free now (backlog 0020); the dev/CI env has the extra.
    assert by_topic["/camera/front"]["can_decode"] is True


def test_inspect_summaryless_omits_chunk_offsets(summaryless, indexed_twin):
    # Chunk offsets need the summary index: unavailable -> empty, not an error.
    scanned = get_adapter("mcap").inspect(summaryless)
    assert scanned["chunk_count"] == 0
    assert scanned["chunks"] == []
    # The finalized twin, by contrast, does report chunk offsets.
    indexed = get_adapter("mcap").inspect(indexed_twin)
    assert indexed["chunk_count"] >= 1
    assert indexed["chunks"]


def test_inspect_summaryless_is_deterministic(summaryless):
    import json

    adapter = get_adapter("mcap")
    first = json.dumps(adapter.inspect(summaryless), sort_keys=True)
    second = json.dumps(adapter.inspect(summaryless), sort_keys=True)
    assert first == second


# --- ingest -----------------------------------------------------------------


def _ingest(tmp_path, name, source):
    lake = Lake.init(tmp_path / name)
    report = ingest_mcap(lake, source)
    return lake, report


def _observation_fingerprint(lake):
    """Canonical observation content, free of run-id/path/time-of-ingest fields."""
    rows = lake.table("observations").to_arrow().to_pylist()
    return sorted(
        (
            r["topic"],
            r["timestamp_ns"],
            r["raw_sequence"],
            r["modality"],
            r["message_encoding"],
            r["schema_encoding"],
            r["decode_status"],
            r["payload_json"],
        )
        for r in rows
    )


def test_ingest_summaryless_produces_correct_rows(tmp_path, summaryless):
    lake, report = _ingest(tmp_path, "robot.lance", summaryless)
    assert report.already_ingested is False
    assert report.message_count == 5
    assert report.observations_by_topic == {"/camera/front": 2, "/imu": 3}
    assert lake.table("observations").count_rows() == 5
    # A clean (if unindexed) read is not quarantined.
    assert report.integrity_status == "complete"
    assert report.quarantined is False
    # Time coverage is recovered from the scan, not a summary.
    assert report.start_time_ns == BASE_NS
    assert report.end_time_ns == BASE_NS + 200_000_000


def test_ingest_summaryless_matches_indexed_twin(tmp_path, summaryless, indexed_twin):
    # "A log pulled straight off a robot mid-mission ingests the same as a
    # cleanly-closed one -- no summary required."
    scanned_lake, scanned_report = _ingest(tmp_path, "scanned.lance", summaryless)
    indexed_lake, indexed_report = _ingest(tmp_path, "indexed.lance", indexed_twin)

    assert scanned_report.observations_by_topic == indexed_report.observations_by_topic
    assert scanned_report.message_count == indexed_report.message_count
    assert _observation_fingerprint(scanned_lake) == _observation_fingerprint(indexed_lake)


# --- export -----------------------------------------------------------------


def test_export_from_summaryless_produces_indexed_clip(tmp_path, summaryless):
    out_path = tmp_path / "clip.mcap"
    result = get_adapter("mcap").export(
        summaryless,
        start_time_ns=BASE_NS,
        end_time_ns=BASE_NS + 200_000_000,
        out_path=out_path,
    )
    assert result["message_count"] == 5
    assert set(result["topics"]) == {"/imu", "/camera/front"}

    # The exported clip is re-indexed: it now has a summary and inspects as
    # indexed, even though the source did not.
    with out_path.open("rb") as handle:
        assert make_reader(handle).get_summary() is not None
    clip_report = get_adapter("mcap").inspect(out_path)
    assert clip_report["indexed"] is True
    assert clip_report["message_count"] == 5


def test_export_window_from_summaryless_respects_bounds(tmp_path, summaryless):
    out_path = tmp_path / "clip.mcap"
    # Window [BASE, BASE+50ms] keeps /imu@0 and /camera/front@50ms only.
    result = get_adapter("mcap").export(
        summaryless,
        start_time_ns=BASE_NS,
        end_time_ns=BASE_NS + 50_000_000,
        out_path=out_path,
    )
    assert result["message_count"] == 2
    with out_path.open("rb") as handle:
        log_times = [m.log_time for _, _, m in make_reader(handle).iter_messages()]
    assert all(BASE_NS <= t <= BASE_NS + 50_000_000 for t in log_times)


# --- genuine-error path is preserved ----------------------------------------


def test_zero_byte_file_still_raises(tmp_path):
    empty = tmp_path / "empty.mcap"
    empty.write_bytes(b"")
    with pytest.raises(AdapterError):
        get_adapter("mcap").inspect(empty)


def test_zero_byte_file_does_not_silently_ingest(tmp_path):
    empty = tmp_path / "empty.mcap"
    empty.write_bytes(b"")
    lake = Lake.init(tmp_path / "robot.lance")
    with pytest.raises(AdapterError):
        ingest_mcap(lake, empty)
    for table in ("runs", "observations", "events", "transform_runs"):
        assert lake.table(table).count_rows() == 0


def test_garbage_file_still_raises(tmp_path):
    bogus = tmp_path / "bogus.mcap"
    bogus.write_bytes(b"this is not an mcap file at all")
    with pytest.raises(AdapterError):
        get_adapter("mcap").inspect(bogus)
