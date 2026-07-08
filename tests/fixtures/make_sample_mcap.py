"""Regenerate tests/fixtures/sample.mcap.

Run: uv run python tests/fixtures/make_sample_mcap.py

All timestamps are fixed so inspection metadata is stable across regenerations.
The file carries two decodable channels exercising two decode families: a ``json``
``/imu`` channel (stdlib) and a ``cbor`` ``/camera/front`` channel (self-describing
binary, backlog 0020). cbor is decoded schema-free, so both channels land
``decoded`` once the ``cbor`` extra is present; the camera payload is a real cbor
map so the decode is meaningful rather than a stand-in. The deterministically
*undecodable* fixture is incomplete.mcap (an ``lcm`` channel, outside the MCAP
well-known registry), which the decodable-streams quality rule keys on.
"""

import json
from pathlib import Path

import cbor2
from mcap.writer import CompressionType, Writer

BASE_NS = 1_700_000_000_000_000_000  # fixed epoch so the fixture is deterministic

OUT = Path(__file__).parent / "sample.mcap"


def main() -> None:
    with OUT.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start(profile="", library="lancedb-robotics-fixture")

        imu_schema = writer.register_schema(
            name="sample.Imu",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {"gyro_z": {"type": "number"}},
                }
            ).encode(),
        )
        imu_channel = writer.register_channel(
            topic="/imu",
            message_encoding="json",
            schema_id=imu_schema,
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
            # cbor is self-describing; the schema record is informational only.
            data=cbor2.dumps({"fields": ["format", "frame_id", "data"]}),
        )
        cam_channel = writer.register_channel(
            topic="/camera/front",
            message_encoding="cbor",
            schema_id=cam_schema,
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
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
