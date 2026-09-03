"""Schema registry: persist real message-definition bytes at ingest.

Root cause this closes: ``adapters/decoders.py``'s ``PayloadDecoder.decode``
needs a real schema object (``.name``/``.encoding``/``.data``) to re-decode
ros1/cdr/protobuf/flatbuffer payloads, but ``OBSERVATIONS_SCHEMA`` (pre-v5)
only ever persisted ``schema_encoding`` (a descriptive string) -- never the
actual definition bytes. These tests use the real, checked-in
``slice_ros1_zstd.mcap`` fixture (genuine ros1msg content, not a synthetic
schema-free stand-in) so the digest/registry plumbing is proven against an
actual ROS message definition, not a toy.
"""

from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.schema_registry import fetch_registered_schema, schema_digest


def test_ingest_populates_schema_registry_and_observation_digests(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    report = ingest_mcap(lake, fixtures_dir / "slice_ros1_zstd.mcap")
    assert report.quarantined is False

    observations = (
        lake.table("observations")
        .search()
        .select(["topic", "message_encoding", "schema_encoding", "schema_digest"])
        .to_arrow()
        .to_pylist()
    )
    assert observations, "fixture should decode to at least one observation"
    assert all(row["message_encoding"] == "ros1" for row in observations)
    # Every ros1 row gets a schema_digest -- this is the field that didn't
    # exist before v5 and is what makes payload_blob re-decodable.
    assert all(row["schema_digest"] for row in observations)

    registry = lake.table("schema_registry").to_arrow().to_pylist()
    assert registry, "distinct schemas seen during ingest must be registered"
    # One registry row per distinct topic schema (3 topics in this fixture:
    # diagnostics/range/tracks), not one per message -- content-addressed dedup.
    topics = {row["topic"] for row in observations}
    assert len(registry) == len(topics)

    registered_digests = {row["schema_digest"] for row in registry}
    assert registered_digests == {row["schema_digest"] for row in observations}

    for row in registry:
        assert row["data"], "the real message-definition bytes must be persisted"
        assert row["schema_digest"] == schema_digest(
            name=row["schema_name"], encoding=row["schema_encoding"], data=row["data"]
        )


def test_fetch_registered_schema_returns_byte_identical_definition(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "slice_ros1_zstd.mcap")

    obs_row = (
        lake.table("observations")
        .search()
        .select(["schema_digest", "schema_encoding"])
        .limit(1)
        .to_arrow()
        .to_pylist()[0]
    )
    registry_row = [
        row
        for row in lake.table("schema_registry").to_arrow().to_pylist()
        if row["schema_digest"] == obs_row["schema_digest"]
    ][0]

    schema = fetch_registered_schema(lake, obs_row["schema_digest"])
    assert schema is not None
    assert schema.name == registry_row["schema_name"]
    assert schema.encoding == obs_row["schema_encoding"] == registry_row["schema_encoding"]
    assert schema.data == registry_row["data"]  # real .msg definition bytes, byte-identical

    # Cache is honored (no second table read needed for a repeat lookup).
    cache: dict = {}
    first = fetch_registered_schema(lake, obs_row["schema_digest"], cache=cache)
    second = fetch_registered_schema(lake, obs_row["schema_digest"], cache=cache)
    assert first is second


def test_fetch_registered_schema_missing_digest_returns_none(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    assert fetch_registered_schema(lake, "does-not-exist") is None


def test_redecode_payload_blob_reports_raw_without_schema_digest():
    from lancedb_robotics.schema_registry import redecode_payload_blob

    class _NoLake:
        def table(self, name):  # pragma: no cover - never reached, asserts it isn't
            raise AssertionError("should not query the lake when schema_digest is absent")

    result = redecode_payload_blob(
        _NoLake(),
        {"schema_digest": None, "message_encoding": "ros1", "payload_blob": b"x" * 10},
    )
    assert result.status == "raw"
