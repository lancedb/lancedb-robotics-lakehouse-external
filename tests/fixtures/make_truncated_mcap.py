"""Regenerate tests/fixtures/truncated.mcap (backlog 0017).

Run: uv run python tests/fixtures/make_truncated_mcap.py

A deliberately truncated MCAP: a complete, multi-chunk zstd file with its tail
(summary, footer, and part of the data section) chopped off. The seeking reader
cannot find the footer/index, so ingest falls back to a forward streaming pass
that recovers the readable prefix and quarantines the rest -- never a hard crash
on the whole file.

The base is built deterministically (fixed messages, small chunk size to force
many chunks) and cut at a fixed fraction of its length, so the fixture is
byte-stable. The cut lands mid-data, so recovery yields *some but not all*
messages -- the point of the fixture.
"""

import io
from pathlib import Path

from mcap.writer import CompressionType, Writer

OUT = Path(__file__).parent / "truncated.mcap"
BASE_NS = 1_700_000_000_000_000_000
N_MESSAGES = 30
PAYLOAD = b"x" * 300  # big enough, with the small chunk size, to force many chunks
KEEP_FRACTION = (55, 100)  # keep 55% of the bytes -> mid-data cut


def _build_base() -> bytes:
    buf = io.BytesIO()
    writer = Writer(buf, compression=CompressionType.ZSTD, enable_crcs=True, chunk_size=512)
    writer.start(profile="", library="lancedb-robotics-fixture")
    schema = writer.register_schema(
        name="sample.Reading",
        encoding="jsonschema",
        data=b'{"type":"object"}',
    )
    channel = writer.register_channel(
        topic="/sensor", message_encoding="json", schema_id=schema
    )
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
    """Return the truncated fixture bytes (deterministic; used by main + the test)."""
    base = _build_base()
    cut = len(base) * KEEP_FRACTION[0] // KEEP_FRACTION[1]
    return base[:cut]


def main() -> None:
    OUT.write_bytes(build_bytes())

    # Verify the fixture does what it claims: recovery yields a partial prefix.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from lancedb_robotics.adapters import CorruptMcapError, get_adapter

    adapter = get_adapter("mcap")
    recovered = 0
    status = None
    try:
        for _ in adapter.ingest(OUT):
            recovered += 1
    except CorruptMcapError as exc:
        status = exc.status
    assert status == "truncated", f"expected truncated, got {status!r}"
    assert 0 < recovered < N_MESSAGES, f"expected a partial prefix, recovered {recovered}"
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes): "
        f"recovers {recovered}/{N_MESSAGES} messages then quarantines (status={status})"
    )


if __name__ == "__main__":
    main()
