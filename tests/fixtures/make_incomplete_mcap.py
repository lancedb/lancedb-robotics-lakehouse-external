"""Regenerate tests/fixtures/incomplete.mcap.

Run: uv run python tests/fixtures/make_incomplete_mcap.py

The intentionally bad sibling of sample.mcap, built to fail the `demo`
validation profile deterministically in three ways:

- `/imu` has only 2 messages (profile requires 3) and both share one log_time,
  so timestamps are not strictly increasing.
- `/imu` uses the `lcm` message encoding -- a real robotics serialization
  (Lightweight Communications and Marshalling) that is *outside* the MCAP
  well-known registry, so the package has no decoder for it and the
  decodable-streams rule fails. An out-of-registry encoding is used on purpose:
  every registry encoding (json/ros1/cdr/protobuf/flatbuffer/cbor/msgpack) is now
  decodable when its extra is installed (backlog 0020), so only a genuinely
  unknown encoding stays undecodable regardless of which extras are present.
- `/camera/front` is absent entirely.

The time-range-overlap rule still passes (only one required topic is present),
so reports show a mix of failed and passed rules.
"""

import json
from pathlib import Path

from mcap.writer import CompressionType, Writer

BASE_NS = 1_700_000_000_000_000_000  # same fixed epoch as make_sample_mcap.py

OUT = Path(__file__).parent / "incomplete.mcap"


def main() -> None:
    with OUT.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start(profile="", library="lancedb-robotics-fixture")

        imu_schema = writer.register_schema(
            name="sample/Imu",
            encoding="lcm",
            data=json.dumps({"note": "stand-in; lcm is outside the MCAP registry"}).encode(),
        )
        imu_channel = writer.register_channel(
            topic="/imu",
            message_encoding="lcm",
            schema_id=imu_schema,
        )
        for i in range(2):
            writer.add_message(
                channel_id=imu_channel,
                log_time=BASE_NS,  # deliberately identical: monotonicity must fail
                publish_time=BASE_NS,
                data=b"\x00" + bytes([i]),
            )

        writer.finish()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
