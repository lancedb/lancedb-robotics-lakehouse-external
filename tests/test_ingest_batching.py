"""Streaming/batched ingest + per-run decode summary (backlog 0017).

Observations stream to the lake in batches so a multi-GB log ingests with bounded
memory. The batch size is a pure performance knob: it must never change which
rows land or their contents. The per-run decode summary (how messages resolved,
which encodings appeared) is surfaced on the report and the ingest lineage.
"""

import json

import pytest

from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake

_PROJECTION = (
    "observation_id",
    "topic",
    "raw_sequence",
    "timestamp_ns",
    "modality",
    "decode_status",
    "payload_json",
    "state_vector",
    "action_vector",
)


@pytest.fixture
def sample_mcap(fixtures_dir):
    return fixtures_dir / "sample.mcap"


def _rows(tmp_path, name, source, **kwargs):
    lake = Lake.init(tmp_path / name)
    ingest_mcap(lake, source, **kwargs)
    rows = lake.table("observations").to_arrow().to_pylist()
    rows.sort(key=lambda r: r["observation_id"])
    return [{k: r[k] for k in _PROJECTION} for r in rows]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 1000])
def test_batch_size_does_not_change_rows(tmp_path, sample_mcap, batch_size):
    baseline = _rows(tmp_path, "baseline.lance", sample_mcap, batch_size=1024)
    batched = _rows(tmp_path, f"batched-{batch_size}.lance", sample_mcap, batch_size=batch_size)
    assert batched == baseline


def test_all_messages_land_regardless_of_batch_size(tmp_path, sample_mcap):
    lake = Lake.init(tmp_path / "robot.lance")
    report = ingest_mcap(lake, sample_mcap, batch_size=1)
    assert lake.table("observations").count_rows() == report.message_count == 5


def test_batch_size_must_be_positive(tmp_path, sample_mcap):
    lake = Lake.init(tmp_path / "robot.lance")
    with pytest.raises(ValueError, match="batch_size"):
        ingest_mcap(lake, sample_mcap, batch_size=0)


def test_batch_size_recorded_in_transform_params(tmp_path, sample_mcap):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, sample_mcap, batch_size=2)
    ingest = next(
        t for t in lake.table("transform_runs").to_arrow().to_pylist() if t["kind"] == "ingest"
    )
    assert json.loads(ingest["params"])["batch_size"] == 2


# --- per-run decode summary -----------------------------------------------


def test_decode_summary_on_report(tmp_path, sample_mcap):
    lake = Lake.init(tmp_path / "robot.lance")
    report = ingest_mcap(lake, sample_mcap)
    # sample.mcap: 3 json + 2 cbor, all decoded now that cbor is self-describing
    # (backlog 0020). Nothing remains raw, so the raw-by-encoding map is empty.
    assert report.decode_by_status == {"decoded": 5}
    assert report.decode_by_encoding == {"cbor": 2, "json": 3}
    assert report.decode_raw_by_encoding == {}


def test_undecodable_messages_do_not_quarantine(tmp_path, sample_mcap):
    # `raw` is a normal outcome (unsupported encoding), not corruption -- the run
    # must not be quarantined for it.
    lake = Lake.init(tmp_path / "robot.lance")
    report = ingest_mcap(lake, sample_mcap)
    assert report.integrity_status == "complete"
    assert report.quarantined is False
