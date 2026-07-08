"""Compression-codec coverage on real corpus slices (backlog 0017).

Two tiny fixtures carry genuine compressed bytes sliced from the corpora with the
lossless ``export``: ``slice_ros1_zstd.mcap`` (ros1msg + zstd, from demo/didi)
and ``slice_mixed_lz4.mcap`` (json + protobuf + lz4, from nuScenes). They prove
both codecs inspect and ingest correctly, that decode runs for real (not
probe-only), and that an unavailable codec fails with a clear, actionable error
rather than a stack trace.
"""

import json

import pytest

from lancedb_robotics.adapters import CodecUnavailableError, get_adapter
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake


@pytest.fixture
def adapter():
    return get_adapter("mcap")


@pytest.fixture
def zstd_slice(fixtures_dir):
    return fixtures_dir / "slice_ros1_zstd.mcap"


@pytest.fixture
def lz4_slice(fixtures_dir):
    return fixtures_dir / "slice_mixed_lz4.mcap"


@pytest.fixture
def lake(tmp_path):
    return Lake.init(tmp_path / "robot.lance")


# --- zstd / ros1 ----------------------------------------------------------


def test_zstd_slice_inspects_as_ros1_zstd(adapter, zstd_slice):
    info = adapter.inspect(zstd_slice)
    assert {c["compression"] for c in info["chunks"]} == {"zstd"}
    assert info["topics"], "slice should carry topics"
    for topic in info["topics"]:
        assert topic["message_encoding"] == "ros1"
        assert topic["schema_encoding"] == "ros1msg"


def test_zstd_slice_ingests_and_decodes_ros1(lake, zstd_slice):
    pytest.importorskip("mcap_ros1")
    report = ingest_mcap(lake, zstd_slice)
    assert report.quarantined is False
    rows = lake.table("observations").to_arrow().to_pylist()
    assert rows and all(r["decode_status"] == "decoded" for r in rows)
    # Decoded ros1 payloads are real JSON dicts (not raw passthrough).
    assert all(json.loads(r["payload_json"]) for r in rows)


def test_zstd_decode_parity_with_uncompressed_copy(adapter, zstd_slice, tmp_path):
    """Re-exporting the same messages uncompressed yields byte-identical decodes."""
    plain = tmp_path / "plain.mcap"
    info = adapter.inspect(zstd_slice)
    adapter.export(
        zstd_slice,
        start_time_ns=info["start_time_ns"],
        end_time_ns=info["end_time_ns"],
        out_path=plain,
        compression="none",
    )
    assert {c["compression"] for c in adapter.inspect(plain)["chunks"]} == {""}

    def decoded(path):
        return [
            (m["topic"], m["sequence"], m["decode_status"], m["payload_json"])
            for m in adapter.ingest(path)
        ]

    assert decoded(zstd_slice) == decoded(plain)


# --- lz4 / mixed encodings ------------------------------------------------


def test_lz4_slice_inspects_as_mixed_lz4(adapter, lz4_slice):
    info = adapter.inspect(lz4_slice)
    assert {c["compression"] for c in info["chunks"]} == {"lz4"}
    encodings = {t["message_encoding"] for t in info["topics"]}
    assert {"json", "protobuf"} <= encodings  # one file, more than one encoding


def test_lz4_slice_ingests_json(lake, lz4_slice):
    # json decodes with only the stdlib -- no extra needed.
    report = ingest_mcap(lake, lz4_slice)
    assert report.quarantined is False
    rows = {r["topic"]: r for r in lake.table("observations").to_arrow().to_pylist()}
    imu = rows["/imu"]
    assert imu["decode_status"] == "decoded"
    assert imu["modality"] == "imu"


def test_lz4_slice_decodes_protobuf(lake, lz4_slice):
    pytest.importorskip("mcap_protobuf")
    ingest_mcap(lake, lz4_slice)
    rows = {r["topic"]: r for r in lake.table("observations").to_arrow().to_pylist()}
    gps = rows["/gps"]
    assert gps["decode_status"] == "decoded"
    assert gps["modality"] == "gps"  # foxglove.LocationFix -> gps (typed extraction)


def test_lz4_decode_coverage_spans_both_encodings(lake, lz4_slice):
    report = ingest_mcap(lake, lz4_slice)
    assert report.decode_by_encoding.get("json", 0) >= 1
    assert report.decode_by_encoding.get("protobuf", 0) >= 1


# --- missing codec --------------------------------------------------------


def test_missing_zstd_codec_raises_actionable_error(adapter, zstd_slice, monkeypatch):
    # Simulate an environment without the zstd codec.
    monkeypatch.setattr("mcap.stream_reader.zstandard", None)
    with pytest.raises(CodecUnavailableError) as excinfo:
        list(adapter.ingest(zstd_slice))
    message = str(excinfo.value)
    assert "zstandard" in message
    assert "install" in message.lower()
    assert excinfo.value.codec == "zstandard"


def test_missing_lz4_codec_raises_actionable_error(adapter, lz4_slice, monkeypatch):
    monkeypatch.setattr("mcap.stream_reader.lz4", None)
    with pytest.raises(CodecUnavailableError) as excinfo:
        list(adapter.ingest(lz4_slice))
    assert "lz4" in str(excinfo.value)
    assert excinfo.value.codec == "lz4"


def test_missing_codec_fails_ingest_mcap_as_hard_error(lake, zstd_slice, monkeypatch):
    # A missing codec is fixable by installing it, so it is never quarantined --
    # it aborts ingest before any run row is written.
    monkeypatch.setattr("mcap.stream_reader.zstandard", None)
    with pytest.raises(CodecUnavailableError):
        ingest_mcap(lake, zstd_slice)
    assert lake.table("runs").count_rows() == 0
