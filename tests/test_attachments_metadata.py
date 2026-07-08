"""MCAP attachment + metadata record capture (backlog 0016).

"Full MCAP parsing" is more than message payloads: MCAP defines two other
first-class record types a message-only reader ignores — attachments (embedded
files like calibration/intrinsics) and log-level metadata records (run context
distinct from per-channel metadata). These tests pin both, on inspect and on
ingest, and pin that a log carrying neither is unaffected.

The positive path runs against ``records.mcap`` (tests/fixtures/make_records_mcap.py),
a synthetic fixture whose metadata record mirrors the single ``scene-info``
record real nuScenes mini files carry; no corpus ships attachments, so the
fixture is the only attachment coverage. The regression path reuses
``sample.mcap``, which has zero of either record type.
"""

import hashlib
import json

import pytest

from lancedb_robotics.adapters import get_adapter
from lancedb_robotics.blob import ATTACHMENT_DATA_COLUMN, fetch_blobs
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake

# Mirrors tests/fixtures/make_records_mcap.py.
EXPECTED_SCENE_INFO = {
    "description": "Parked truck, construction, intersection, turn left",
    "name": "scene-0061",
    "location": "singapore-onenorth",
    "vehicle": "n015",
    "date_captured": "2018-07-24",
}
EXPECTED_INTRINSICS = bytes(range(64))


@pytest.fixture
def records_mcap(fixtures_dir):
    return fixtures_dir / "records.mcap"


@pytest.fixture
def sample_mcap(fixtures_dir):
    return fixtures_dir / "sample.mcap"


# --- inspect -----------------------------------------------------------------


def test_inspect_lists_metadata_records(records_mcap):
    report = get_adapter("mcap").inspect(records_mcap)
    assert report["metadata"] == [
        {"name": "scene-info", "keys": sorted(EXPECTED_SCENE_INFO)}
    ]


def test_inspect_lists_attachment_summaries(records_mcap):
    report = get_adapter("mcap").inspect(records_mcap)
    attachments = report["attachments"]
    # Sorted by (log_time, offset, name): calibration was logged first.
    assert [a["name"] for a in attachments] == ["calibration.json", "intrinsics.bin"]
    cal, intr = attachments
    assert cal["media_type"] == "application/json"
    assert intr["media_type"] == "application/octet-stream"
    assert intr["size"] == len(EXPECTED_INTRINSICS)
    # Summaries come from the index, so offsets are real positions in the file.
    assert all(a["offset"] > 0 for a in attachments)
    assert all(a["log_time_ns"] > 0 for a in attachments)


def test_inspect_reports_empty_records_for_plain_log(sample_mcap):
    report = get_adapter("mcap").inspect(sample_mcap)
    assert report["attachments"] == []
    assert report["metadata"] == []


# --- adapter generators ------------------------------------------------------


def test_adapter_attachments_are_hashed_and_recoverable(records_mcap):
    attachments = list(get_adapter("mcap").attachments(records_mcap))
    assert [a["name"] for a in attachments] == ["calibration.json", "intrinsics.bin"]
    for att in attachments:
        # Self-consistent content hash over the recovered bytes.
        assert att["sha256"] == hashlib.sha256(att["data"]).hexdigest()
        assert att["size"] == len(att["data"])
    cal, intr = attachments
    assert json.loads(cal["data"]) == {"camera_matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    assert intr["data"] == EXPECTED_INTRINSICS


def test_adapter_metadata_records_carry_keys_and_values(records_mcap):
    records = list(get_adapter("mcap").metadata_records(records_mcap))
    assert records == [{"name": "scene-info", "metadata": EXPECTED_SCENE_INFO}]


def test_adapter_records_empty_for_plain_log(sample_mcap):
    adapter = get_adapter("mcap")
    assert list(adapter.attachments(sample_mcap)) == []
    assert list(adapter.metadata_records(sample_mcap)) == []


# --- ingest ------------------------------------------------------------------


@pytest.fixture
def lake(tmp_path):
    return Lake.init(tmp_path / "robot.lance")


def _run_metadata(lake, run_id):
    rows = lake.table("runs").to_arrow().to_pylist()
    (run,) = [r for r in rows if r["run_id"] == run_id]
    return {kv["key"]: kv["value"] for kv in run["metadata"]}


def _ingest_transform_params(lake, run_id):
    rows = lake.table("transform_runs").to_arrow().to_pylist()
    (ingest,) = [r for r in rows if r["kind"] == "ingest" and run_id in r["params"]]
    return ingest, json.loads(ingest["params"])


def test_ingest_merges_metadata_record_namespaced(lake, records_mcap):
    report = ingest_mcap(lake, records_mcap)
    metadata = _run_metadata(lake, report.run_id)
    # File-level keys still present, distinct from the metadata record.
    assert "profile" in metadata
    assert "library" in metadata
    # scene-info record merged in, namespaced by record name.
    assert metadata["scene-info.location"] == "singapore-onenorth"
    assert metadata["scene-info.vehicle"] == "n015"
    assert metadata["scene-info.date_captured"] == "2018-07-24"
    assert all(k.startswith("scene-info.") for k in metadata if k.endswith("location"))


def test_ingest_captures_attachments_recoverably(lake, records_mcap):
    report = ingest_mcap(lake, records_mcap)
    assert report.rows_added["attachments"] == 2

    table = lake.table("attachments")
    rows = table.to_arrow().to_pylist()
    rows.sort(key=lambda r: r["attachment_id"])
    assert [r["name"] for r in rows] == ["calibration.json", "intrinsics.bin"]
    assert all(r["run_id"] == report.run_id for r in rows)
    # ``data`` is Lance blob-encoded (backlog 0035): the metadata scan above reads
    # no attachment bytes (the column comes back as a lazy ref), so the bytes are
    # fetched lazily by id and verified byte-for-byte against the stored hash/size.
    blobs = fetch_blobs(
        table,
        ATTACHMENT_DATA_COLUMN,
        [r["attachment_id"] for r in rows],
        id_column="attachment_id",
    )
    for r in rows:
        data = blobs[r["attachment_id"]]
        assert r["sha256"] == hashlib.sha256(data).hexdigest()
        assert r["size"] == len(data)
    intr = next(r for r in rows if r["name"] == "intrinsics.bin")
    assert blobs[intr["attachment_id"]] == EXPECTED_INTRINSICS


def test_ingest_records_record_counts_in_transform_params(lake, records_mcap):
    report = ingest_mcap(lake, records_mcap)
    transform, params = _ingest_transform_params(lake, report.run_id)
    assert params["attachment_count"] == 2
    assert params["metadata_record_count"] == 1
    assert "attachments" in transform["output_tables"]


def test_zero_records_log_ingests_unchanged(lake, sample_mcap):
    report = ingest_mcap(lake, sample_mcap)

    # No attachment rows, table left empty.
    assert report.rows_added["attachments"] == 0
    assert lake.table("attachments").count_rows() == 0

    # runs.metadata carries the file-level keys + the integrity verdict (backlog
    # 0017), but no namespaced metadata-record keys.
    metadata = _run_metadata(lake, report.run_id)
    assert set(metadata) == {"profile", "library", "integrity.status"}
    assert metadata["integrity.status"] == "complete"

    # Counts recorded as zero; attachments not declared as an output table.
    transform, params = _ingest_transform_params(lake, report.run_id)
    assert params["attachment_count"] == 0
    assert params["metadata_record_count"] == 0
    assert transform["output_tables"] == ["runs", "observations", "events"]
