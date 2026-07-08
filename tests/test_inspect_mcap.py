"""MCAP fixture inspection contract tests (backlog 0003).

These pin the inspect payload shape that downstream ingest (backlog 0004)
will consume: topics, encodings, schemas, counts, time ranges, chunk offsets,
and decode capability.
"""

import pytest

from lancedb_robotics.adapters import get_adapter
from lancedb_robotics.adapters.mcap_adapter import AdapterError

BASE_NS = 1_700_000_000_000_000_000  # matches tests/fixtures/make_sample_mcap.py


@pytest.fixture
def sample_mcap(fixtures_dir):
    return fixtures_dir / "sample.mcap"


@pytest.fixture
def report(sample_mcap):
    return get_adapter("mcap").inspect(sample_mcap)


def test_inspect_reports_file_level_metadata(report, sample_mcap):
    assert report["adapter"] == "mcap"
    assert report["path"] == str(sample_mcap)
    assert report["profile"] == ""
    assert report["library"] == "lancedb-robotics-fixture"
    assert report["message_count"] == 5
    assert report["schema_count"] == 2
    assert report["channel_count"] == 2
    assert report["chunk_count"] == 1
    assert report["start_time_ns"] == BASE_NS
    assert report["end_time_ns"] == BASE_NS + 200_000_000
    assert report["duration_ns"] == 200_000_000


def test_inspect_topics_are_sorted_and_complete(report):
    topics = report["topics"]
    assert [t["topic"] for t in topics] == ["/camera/front", "/imu"]

    cam, imu = topics
    assert imu["message_encoding"] == "json"
    assert imu["schema_name"] == "sample.Imu"
    assert imu["schema_encoding"] == "jsonschema"
    assert imu["message_count"] == 3
    assert imu["start_time_ns"] == BASE_NS
    assert imu["end_time_ns"] == BASE_NS + 200_000_000

    assert cam["message_encoding"] == "cbor"
    assert cam["schema_name"] == "sample.CompressedImage"
    assert cam["schema_encoding"] == "cbor"
    assert cam["message_count"] == 2
    assert cam["start_time_ns"] == BASE_NS + 50_000_000
    assert cam["end_time_ns"] == BASE_NS + 150_000_000


def test_inspect_reports_decode_capability(report):
    by_topic = {t["topic"]: t for t in report["topics"]}
    # json decodes with the stdlib; cbor is self-describing binary, decoded once
    # the cbor extra is present (backlog 0020), which the dev/CI env installs.
    assert by_topic["/imu"]["can_decode"] is True
    assert by_topic["/camera/front"]["can_decode"] is True


def test_inspect_reports_chunk_offsets(report):
    chunks = report["chunks"]
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["offset"] > 0
    assert chunk["length"] > 0
    assert chunk["message_start_time_ns"] == BASE_NS
    assert chunk["message_end_time_ns"] == BASE_NS + 200_000_000


def test_inspect_is_deterministic(sample_mcap):
    import json

    adapter = get_adapter("mcap")
    first = json.dumps(adapter.inspect(sample_mcap), sort_keys=True)
    second = json.dumps(adapter.inspect(sample_mcap), sort_keys=True)
    assert first == second


def test_inspect_missing_file_raises(tmp_path):
    with pytest.raises(AdapterError):
        get_adapter("mcap").inspect(tmp_path / "nope.mcap")


def test_inspect_non_mcap_file_raises(tmp_path):
    bogus = tmp_path / "bogus.mcap"
    bogus.write_bytes(b"this is not an mcap file")
    with pytest.raises(AdapterError):
        get_adapter("mcap").inspect(bogus)
