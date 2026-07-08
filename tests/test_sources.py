"""Source registration contract tests (backlog 0004).

Registration is the provenance anchor: every ingested row must trace back to
an `integration_sources` row identified by (URI, checksum). Re-registering an
unchanged file is a no-op; a changed file at the same URI is a new source.
"""

import pytest

from lancedb_robotics.lake import Lake
from lancedb_robotics.sources import file_checksum, register_source


@pytest.fixture
def sample_mcap(fixtures_dir):
    return fixtures_dir / "sample.mcap"


@pytest.fixture
def lake(tmp_path):
    return Lake.init(tmp_path / "robot.lance")


def _write_minimal_mcap(path):
    from mcap.writer import Writer

    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start(profile="", library="test-changed-content")
        writer.finish()


def test_file_checksum_is_sha256_hex(sample_mcap):
    checksum = file_checksum(sample_mcap)
    assert len(checksum) == 64
    assert checksum == file_checksum(sample_mcap)  # deterministic


def test_register_source_writes_one_row(lake, sample_mcap):
    reg = register_source(lake, sample_mcap, adapter="mcap")
    assert reg.created is True
    assert reg.source_id.startswith("src-")
    assert reg.checksum == file_checksum(sample_mcap)
    assert lake.table("integration_sources").count_rows() == 1


def test_register_source_is_idempotent_by_uri_and_checksum(lake, sample_mcap):
    first = register_source(lake, sample_mcap, adapter="mcap")
    second = register_source(lake, sample_mcap, adapter="mcap")
    assert second.created is False
    assert second.source_id == first.source_id
    assert lake.table("integration_sources").count_rows() == 1


def test_register_source_with_changed_content_is_a_new_source(lake, sample_mcap, tmp_path):
    moved = tmp_path / "sample.mcap"
    moved.write_bytes(sample_mcap.read_bytes())
    first = register_source(lake, moved, adapter="mcap")
    _write_minimal_mcap(moved)  # valid MCAP, different content, same URI
    second = register_source(lake, moved, adapter="mcap")
    assert second.created is True
    assert second.source_id != first.source_id
    assert lake.table("integration_sources").count_rows() == 2


def test_register_source_records_checksum_adapter_and_schema_fingerprints(lake, sample_mcap):
    reg = register_source(lake, sample_mcap, adapter="mcap")
    rows = lake.table("integration_sources").to_arrow().to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert row["source_id"] == reg.source_id
    assert row["kind"] == "file"
    assert row["uri"] == reg.uri
    metadata = {entry["key"]: entry["value"] for entry in row["metadata"]}
    assert metadata["checksum"] == reg.checksum
    assert metadata["adapter"] == "mcap"
    # Schema fingerprints come from inspection: one per topic.
    assert metadata["schema:/imu"] == "sample.Imu:jsonschema"
    assert metadata["schema:/camera/front"] == "sample.CompressedImage:cbor"
