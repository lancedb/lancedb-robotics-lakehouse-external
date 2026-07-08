"""Regenerate tests/fixtures/summaryless.mcap and summaryless_indexed.mcap (backlog 0018).

Run: uv run python tests/fixtures/make_summaryless_mcap.py

Two byte-stable fixtures written from the *same* messages:

- ``summaryless.mcap`` -- a legitimate unindexed log: chunked data, but the
  writer emits no summary section (no statistics, no chunk/attachment/metadata
  indexes, no repeated schemas/channels), so the footer's ``summary_start`` is 0
  and ``get_summary()`` returns ``None``. This is the shape of a live/streaming
  recording, an append-only writer, or a recording whose summary was never
  flushed -- the bytes are valid, just unindexed. Distinct from truncation
  (backlog 0017): nothing is damaged.
- ``summaryless_indexed.mcap`` -- its finalized twin, written with the same
  records but a full summary + index. Inspecting/ingesting both must yield the
  same counts, time range, topics, and schema inventory: "a log pulled straight
  off a robot mid-mission reads the same as a cleanly-closed one."

Both carry two decodable channels exercising two decode families -- a ``json``
``/imu`` channel and a self-describing ``cbor`` ``/camera/front`` channel
(backlog 0020) -- mirroring sample.mcap, so decode-capability reporting is
exercised. All timestamps are fixed so the fixtures are deterministic across
regenerations.
"""

import json
from pathlib import Path

import cbor2
from mcap.writer import CompressionType, IndexType, Writer

BASE_NS = 1_700_000_000_000_000_000  # same fixed epoch as make_sample_mcap.py

SUMMARYLESS = Path(__file__).parent / "summaryless.mcap"
INDEXED = Path(__file__).parent / "summaryless_indexed.mcap"


def _populate(writer: Writer) -> None:
    """Write the shared record set: 3 json /imu + 2 cbor /camera/front messages."""
    writer.start(profile="", library="lancedb-robotics-fixture")

    imu_schema = writer.register_schema(
        name="sample.Imu",
        encoding="jsonschema",
        data=json.dumps({"type": "object", "properties": {"gyro_z": {"type": "number"}}}).encode(),
    )
    imu_channel = writer.register_channel(
        topic="/imu", message_encoding="json", schema_id=imu_schema
    )
    for i in range(3):
        writer.add_message(
            channel_id=imu_channel,
            log_time=BASE_NS + i * 100_000_000,
            publish_time=BASE_NS + i * 100_000_000,
            data=json.dumps({"gyro_z": 0.1 * i}).encode(),
        )

    cam_schema = writer.register_schema(
        name="sample.CompressedImage",
        encoding="cbor",
        data=cbor2.dumps({"fields": ["format", "frame_id", "data"]}),
    )
    cam_channel = writer.register_channel(
        topic="/camera/front", message_encoding="cbor", schema_id=cam_schema
    )
    for i in range(2):
        writer.add_message(
            channel_id=cam_channel,
            log_time=BASE_NS + 50_000_000 + i * 100_000_000,
            publish_time=BASE_NS + 50_000_000 + i * 100_000_000,
            data=cbor2.dumps(
                {"format": "jpeg", "frame_id": "cam_front", "data": b"\xff\xd8\xff" + bytes([i])}
            ),
        )

    writer.finish()


def _write_summaryless() -> None:
    with SUMMARYLESS.open("wb") as stream:
        # Chunked data (so the scan path must break chunks up) but no summary:
        # disabling statistics, all indexes, summary offsets, and repeated
        # schemas/channels leaves the summary section empty, so finish() writes a
        # footer with summary_start == 0.
        writer = Writer(
            stream,
            compression=CompressionType.NONE,
            use_chunking=True,
            use_statistics=False,
            use_summary_offsets=False,
            repeat_schemas=False,
            repeat_channels=False,
            index_types=IndexType.NONE,
        )
        _populate(writer)


def _write_indexed() -> None:
    with INDEXED.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)  # defaults: full summary + index
        _populate(writer)


def main() -> None:
    _write_summaryless()
    _write_indexed()

    # Verify the fixtures do what they claim: one has no summary, the twin does,
    # and both inspect to the same message-level metadata.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mcap.reader import make_reader

    from lancedb_robotics.adapters import get_adapter

    with SUMMARYLESS.open("rb") as handle:
        assert make_reader(handle).get_summary() is None, "summaryless.mcap should have no summary"
    with INDEXED.open("rb") as handle:
        assert make_reader(handle).get_summary() is not None, "twin should have a summary"

    adapter = get_adapter("mcap")
    scanned = adapter.inspect(SUMMARYLESS)
    indexed = adapter.inspect(INDEXED)
    assert scanned["indexed"] is False and indexed["indexed"] is True
    assert scanned["message_count"] == indexed["message_count"] == 5
    assert scanned["topics"] == indexed["topics"], "scan-derived topics must match the twin"

    for path in (SUMMARYLESS, INDEXED):
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
