"""Regenerate tests/fixtures/crc_corrupt.mcap (backlog 0017).

Run: uv run python tests/fixtures/make_crc_corrupt_mcap.py

A structurally complete MCAP whose chunk CRC no longer matches its contents: one
data byte in a middle chunk is flipped after the file is written, so the summary
and footer stay valid but the chunk's CRC fails. Inspect (which does not validate
CRCs) still succeeds, so a run is created; ingest validates CRCs as it reads,
yields the messages from the good leading chunks, then raises and quarantines the
run -- silent corruption is never passed through.

Uncompressed chunks are used so the single flipped byte deterministically breaks
the uncompressed CRC (rather than the zstd/lz4 decompressor) -- CRC validation is
codec-independent. The base is deterministic and the flip offset is fixed, so the
fixture is byte-stable.
"""

import io
from pathlib import Path

from mcap.writer import CompressionType, Writer

OUT = Path(__file__).parent / "crc_corrupt.mcap"
BASE_NS = 1_700_000_000_000_000_000
N_MESSAGES = 20
PAYLOAD = b"y" * 120


def _build_base() -> bytes:
    buf = io.BytesIO()
    # compression=NONE: the chunk records hold the raw message bytes, so a single
    # flipped data byte breaks the uncompressed CRC cleanly (and reproducibly).
    writer = Writer(buf, compression=CompressionType.NONE, enable_crcs=True, chunk_size=256)
    writer.start(profile="", library="lancedb-robotics-fixture")
    schema = writer.register_schema(
        name="sample.Reading", encoding="jsonschema", data=b'{"type":"object"}'
    )
    channel = writer.register_channel(topic="/sensor", message_encoding="json", schema_id=schema)
    for i in range(N_MESSAGES):
        writer.add_message(
            channel_id=channel,
            log_time=BASE_NS + i * 1_000_000,
            publish_time=BASE_NS + i * 1_000_000,
            data=b'{"i":%d,"pad":"%s"}' % (i, PAYLOAD),
        )
    writer.finish()
    return buf.getvalue()


def build_bytes() -> bytes:
    """Return the CRC-corrupt fixture bytes (deterministic; used by main + the test)."""
    base = bytearray(_build_base())
    # Flip a byte in a middle chunk's data region (well past the header, well
    # before the trailing summary), so leading chunks validate and yield a prefix.
    base[len(base) // 2] ^= 0xFF
    return bytes(base)


def main() -> None:
    OUT.write_bytes(build_bytes())

    # Verify: inspect succeeds (summary intact); validated ingest quarantines;
    # unvalidated ingest reads silently (the corruption only surfaces with CRCs).
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from lancedb_robotics.adapters import CorruptMcapError, get_adapter

    adapter = get_adapter("mcap")
    adapter.inspect(OUT)  # must not raise -- summary/footer are intact

    recovered = 0
    status = None
    try:
        for _ in adapter.ingest(OUT, validate_crcs=True):
            recovered += 1
    except CorruptMcapError as exc:
        status = exc.status
    assert status == "crc-mismatch", f"expected crc-mismatch, got {status!r}"

    silent = sum(1 for _ in adapter.ingest(OUT, validate_crcs=False))
    assert silent == N_MESSAGES, f"unvalidated read should see all {N_MESSAGES}, saw {silent}"
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes): "
        f"validated read recovers {recovered} then quarantines (status={status}); "
        f"unvalidated read silently passes all {silent}"
    )


if __name__ == "__main__":
    main()
