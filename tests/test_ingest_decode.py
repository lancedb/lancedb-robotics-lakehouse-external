"""Ingest decode mapping tests (backlog 0014, extended for 0020).

The shared ``sample.mcap`` fixture carries 3 json ``/imu`` messages (decodable
with the stdlib) and 2 cbor ``/camera/front`` messages. cbor is self-describing
binary, decoded schema-free once the ``cbor`` extra is present (backlog 0020),
which the dev/CI env installs -- so both channels land ``decoded`` with populated
``payload_json``, every row kept and envelope/provenance unchanged.
"""

import json

import pytest

from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake


@pytest.fixture
def sample_mcap(fixtures_dir):
    return fixtures_dir / "sample.mcap"


@pytest.fixture
def lake(tmp_path):
    return Lake.init(tmp_path / "robot.lance")


@pytest.fixture
def rows(lake, sample_mcap):
    ingest_mcap(lake, sample_mcap)
    return lake.table("observations").to_arrow().to_pylist()


def test_json_observations_are_decoded(rows):
    imu = [r for r in rows if r["topic"] == "/imu"]
    assert len(imu) == 3
    for r in imu:
        assert r["decode_status"] == "decoded"
        assert r["decode_error"] is None
        assert r["message_encoding"] == "json"
        assert r["schema_encoding"] == "jsonschema"
        payload = json.loads(r["payload_json"])
        assert "gyro_z" in payload  # the fixture's json field
        assert r["payload_blob"] is None  # scalar message


def test_cbor_observations_are_decoded_schema_free(rows):
    cam = [r for r in rows if r["topic"] == "/camera/front"]
    assert len(cam) == 2  # not dropped
    for r in cam:
        assert r["decode_status"] == "decoded"
        assert r["decode_error"] is None
        assert r["message_encoding"] == "cbor"
        assert r["schema_encoding"] == "cbor"
        payload = json.loads(r["payload_json"])
        # the fixture's real cbor camera map decodes to its fields
        assert payload["format"] == "jpeg"
        assert payload["frame_id"] == "cam_front"
        # bytes stay recoverable via pointer provenance
        assert r["raw_uri"] and r["raw_channel"] == "/camera/front"


def test_envelope_and_provenance_unchanged_by_decoding(rows):
    # Row count and the existing envelope/provenance fields are untouched (v1 -> v2
    # adds columns, it does not change or drop existing ones).
    assert len(rows) == 5
    for r in rows:
        assert r["raw_uri"]
        assert r["raw_channel"] == r["topic"]
        assert r["raw_log_time_ns"] == r["timestamp_ns"]
        assert r["raw_sequence"] >= 0


def test_decode_coverage_recorded_in_transform_params(lake, sample_mcap):
    ingest_mcap(lake, sample_mcap)
    ingest = next(
        t for t in lake.table("transform_runs").to_arrow().to_pylist() if t["kind"] == "ingest"
    )
    params = json.loads(ingest["params"])
    assert "decode_by_status" in params
    assert "decode_by_encoding" in params
    # 5 messages total accounted for across statuses and encodings.
    assert sum(params["decode_by_status"].values()) == 5
    assert sum(params["decode_by_encoding"].values()) == 5
    assert params["decode_by_status"].get("decoded", 0) >= 3  # the 3 json messages
    assert params["decode_by_encoding"]["json"] == 3
    # Remaining-raw coverage is reported and empty here (everything decoded).
    assert params["decode_raw_by_encoding"] == {}
