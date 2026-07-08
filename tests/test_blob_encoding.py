"""Blob-encoded payload columns + additive (versioned) mutation (backlog 0035).

Decision 0024 makes Lance the index *and* the fast-access layer: the heavy
payload columns (``observations.payload_blob``, ``attachments.data``) are
blob-encoded, so a scan that does not project them reads no blob bytes and a
row's bytes are fetched lazily by id; quality verdicts on the blob-encoded
``observations`` table are written additively (``drop_columns`` + ``add_columns``),
never in-place ``Table.update`` (which raises on a table with a blob column).

These tests build observation/attachment rows directly through the canonical
schema so the blob bytes are real and controlled -- the small MCAP fixtures all
fall under the 2048-byte hoist threshold, so ingest alone never produces a blob.
``test_reingest_is_idempotent`` covers the real ingest path for acceptance #4.
"""

import hashlib

import lance
import pyarrow as pa
import pytest

from lancedb_robotics.blob import (
    ATTACHMENT_DATA_COLUMN,
    PAYLOAD_BLOB_COLUMN,
    fetch_blob,
    fetch_blobs,
)
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.quality import (
    RuleResult,
    RunQualityReport,
    apply_quality_results,
    resolve_profile,
)
from lancedb_robotics.schemas import (
    ATTACHMENTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    RUNS_SCHEMA,
)

LAZY_REF = pa.struct([pa.field("position", pa.uint64()), pa.field("size", pa.uint64())])

RUN_ID = "run-blobs"
# Distinct, well-above-noise payloads so byte-counter deltas are unambiguous.
PAYLOADS = {
    "run-blobs:/camera/front:000000": b"\x89PNG\r\n" + b"front-pixels;" * 4000,
    "run-blobs:/camera/front:000001": b"\x89PNG\r\n" + b"front-pixels;" * 6000,
    "run-blobs:/lidar/top:000000": b"PCD" + b"\x01\x02\x03\x04" * 5000,
}
TOTAL_BLOB_BYTES = sum(len(v) for v in PAYLOADS.values())


def _obs_row(observation_id, topic, *, blob=None, sequence=0):
    return {
        "observation_id": observation_id,
        "run_id": RUN_ID,
        "timestamp_ns": 1_700_000_000_000_000_000 + sequence,
        "topic": topic,
        "modality": "image" if "camera" in topic else "pointcloud",
        "raw_uri": "/archival/source.mcap",
        "raw_channel": topic,
        "raw_sequence": sequence,
        "payload_blob": blob,
        "decode_status": "decoded",
    }


@pytest.fixture
def blob_lake(tmp_path):
    """A lake whose observations carry real blob bytes, plus a matching run row."""
    lake = Lake.init(tmp_path / "blobs.lance")
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": RUN_ID, "run_kind": "drive", "raw_uri": "/archival/source.mcap"}],
            schema=RUNS_SCHEMA,
        )
    )
    rows = [
        _obs_row(
            "run-blobs:/camera/front:000000",
            "/camera/front",
            blob=PAYLOADS["run-blobs:/camera/front:000000"],
            sequence=0,
        ),
        _obs_row(
            "run-blobs:/camera/front:000001",
            "/camera/front",
            blob=PAYLOADS["run-blobs:/camera/front:000001"],
            sequence=1,
        ),
        _obs_row(
            "run-blobs:/lidar/top:000000",
            "/lidar/top",
            blob=PAYLOADS["run-blobs:/lidar/top:000000"],
            sequence=0,
        ),
        # A scalar /imu row with no blob -- the metadata-only neighbour.
        _obs_row("run-blobs:/imu:000000", "/imu", blob=None, sequence=0),
    ]
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))
    return lake


# --- acceptance #1: blob-encoded, metadata scan reads no blob bytes -----------


def test_payload_columns_are_lazy_blob_refs(blob_lake):
    """A full read returns payload_blob as a Lance lazy ref, not materialized bytes."""
    table = blob_lake.table("observations").to_arrow()
    assert table.column("payload_blob").type == LAZY_REF


def test_metadata_only_scan_reads_no_blob_bytes(blob_lake):
    """Projecting only metadata columns reads far fewer bytes than the blobs hold."""
    dataset = blob_lake.table("observations").to_lance()
    before = lance.bytes_read_counter()
    scanned = dataset.to_table(columns=["observation_id", "topic", "quality_flags"])
    meta_bytes = lance.bytes_read_counter() - before

    assert scanned.num_rows == 4
    assert set(scanned.column_names) == {"observation_id", "topic", "quality_flags"}
    # The metadata scan must not have pulled the blob payloads.
    assert meta_bytes < TOTAL_BLOB_BYTES


# --- acceptance #2: fetch blob bytes lazily by id, round-trip -----------------


def test_take_blob_by_id_round_trips_byte_for_byte(blob_lake):
    table = blob_lake.table("observations")
    blobs = fetch_blobs(table, PAYLOAD_BLOB_COLUMN, list(PAYLOADS), id_column="observation_id")
    assert blobs == PAYLOADS


def test_blob_fetch_reads_only_the_requested_bytes(blob_lake):
    """Fetching one blob reads ~that blob, not the whole column."""
    table = blob_lake.table("observations")
    target = "run-blobs:/camera/front:000001"
    before = lance.bytes_read_counter()
    data = fetch_blob(table, PAYLOAD_BLOB_COLUMN, target, id_column="observation_id")
    fetched_bytes = lance.bytes_read_counter() - before

    assert data == PAYLOADS[target]
    assert len(data) <= fetched_bytes < TOTAL_BLOB_BYTES


def test_fetch_blobs_handles_missing_and_null(blob_lake):
    table = blob_lake.table("observations")
    blobs = fetch_blobs(
        table,
        PAYLOAD_BLOB_COLUMN,
        ["run-blobs:/imu:000000", "does-not-exist", "run-blobs:/lidar/top:000000"],
        id_column="observation_id",
    )
    assert blobs["run-blobs:/imu:000000"] == b""  # row exists, blob is NULL
    assert "does-not-exist" not in blobs  # absent id omitted
    assert blobs["run-blobs:/lidar/top:000000"] == PAYLOADS["run-blobs:/lidar/top:000000"]


def test_attachment_data_round_trips_by_id(tmp_path):
    lake = Lake.init(tmp_path / "att.lance")
    payloads = {"a:att:0000": b"calib" * 2000, "a:att:0001": b"\x00\x01" * 3000}
    rows = [
        {
            "attachment_id": aid,
            "run_id": "a",
            "name": aid,
            "media_type": "application/octet-stream",
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "data": data,
        }
        for aid, data in payloads.items()
    ]
    lake.table("attachments").add(pa.Table.from_pylist(rows, schema=ATTACHMENTS_SCHEMA))

    table = lake.table("attachments")
    assert table.to_arrow().column("data").type == LAZY_REF  # lazy ref, no bytes
    fetched = fetch_blobs(table, ATTACHMENT_DATA_COLUMN, list(payloads), id_column="attachment_id")
    assert fetched == payloads


# --- acceptance #4: stable id across enrichment; re-ingest idempotent ---------


def test_observation_id_stable_across_add_columns_enrichment(blob_lake):
    """An add_columns enrichment pass keeps observation_id and the blobs intact."""
    dataset = blob_lake.table("observations").to_lance()
    before = dataset.to_table(columns=["observation_id"]).column("observation_id").to_pylist()

    # Append a typed enrichment column (the SILA "append a column" pattern).
    dataset.add_columns({"enriched_tag": "'reviewed'"})

    after = blob_lake.table("observations")
    assert after.to_arrow().column("observation_id").to_pylist() == before
    assert "enriched_tag" in after.schema.names
    # Blobs survive the enrichment, fetchable by the same ids.
    assert (
        fetch_blobs(after, PAYLOAD_BLOB_COLUMN, list(PAYLOADS), id_column="observation_id")
        == PAYLOADS
    )


def test_reingest_is_idempotent(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "idem.lance")
    first = ingest_mcap(lake, fixtures_dir / "sample.mcap")
    ids_first = sorted(
        o["observation_id"] for o in lake.table("observations").to_arrow().to_pylist()
    )

    second = ingest_mcap(lake, fixtures_dir / "sample.mcap")
    ids_second = sorted(
        o["observation_id"] for o in lake.table("observations").to_arrow().to_pylist()
    )

    assert second.already_ingested
    assert first.run_id == second.run_id
    assert ids_first == ids_second  # no duplicate rows, ids stable (content-addressed)


# --- acceptance #3: quality verdict additive + versioned + blob-safe ----------


def _imu_failure_report():
    return RunQualityReport(
        run_id=RUN_ID,
        profile="demo",
        rules=(
            RuleResult("monotonic-timestamps", "failed", ("not monotonic",), ("/imu",)),
            RuleResult("required-topics", "passed"),
        ),
    )


def test_quality_verdict_is_additive_versioned_and_blob_safe(blob_lake):
    profile = resolve_profile("demo")
    version_before = blob_lake.table("observations").to_lance().version

    apply_quality_results(blob_lake, [_imu_failure_report()], profile)

    observations = blob_lake.table("observations")
    # Versioned/additive: the write advanced the table version (no in-place edit).
    assert observations.to_lance().version > version_before

    rows = {r["observation_id"]: r for r in observations.to_arrow().to_pylist()}
    assert rows["run-blobs:/imu:000000"]["quality_flags"] == ["quality:failed:monotonic-timestamps"]
    for camera_id in ("run-blobs:/camera/front:000000", "run-blobs:/lidar/top:000000"):
        assert not rows[camera_id]["quality_flags"]  # passing topics cleared to NULL

    # The critical proof: the blob bytes survived the verdict write byte-for-byte.
    # (Had quality used in-place Table.update, the apply would have *raised*.)
    assert (
        fetch_blobs(observations, PAYLOAD_BLOB_COLUMN, list(PAYLOADS), id_column="observation_id")
        == PAYLOADS
    )

    # Run-level gate verdict is unchanged (quarantine still set on the run).
    run = next(r for r in blob_lake.table("runs").to_arrow().to_pylist() if r["run_id"] == RUN_ID)
    assert "quarantined" in run["quality_flags"]


def test_quality_revalidation_is_latest_wins(blob_lake):
    profile = resolve_profile("demo")
    apply_quality_results(blob_lake, [_imu_failure_report()], profile)
    version_after_first = blob_lake.table("observations").to_lance().version

    apply_quality_results(blob_lake, [_imu_failure_report()], profile)
    observations = blob_lake.table("observations")

    assert observations.to_lance().version > version_after_first  # each pass is a new version
    imu = next(
        r
        for r in observations.to_arrow().to_pylist()
        if r["observation_id"] == "run-blobs:/imu:000000"
    )
    assert imu["quality_flags"] == ["quality:failed:monotonic-timestamps"]  # no stacking
