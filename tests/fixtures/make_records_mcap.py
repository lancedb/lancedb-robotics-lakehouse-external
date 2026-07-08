"""Regenerate tests/fixtures/records.mcap.

Run: uv run python tests/fixtures/make_records_mcap.py

The sibling of sample.mcap that carries MCAP's *other* first-class record types
(backlog 0016): one log-level metadata record and two attachments, alongside a
normal decodable message stream.

- One metadata record named ``scene-info`` whose key/values mirror the shape of
  the single metadata record real nuScenes files carry (description / name /
  location / vehicle / date_captured), so the ingest path is exercised against
  a realistic record without needing the multi-GB corpus.
- Two attachments with distinct media types (a JSON calibration blob and a
  binary intrinsics blob), so attachment capture, content hashing, and
  deterministic ordering are all exercised. No real corpus ships attachments, so
  this synthetic fixture is the only coverage until one arrives.
- One ``/imu`` json channel with three messages, so the file also ingests normal
  observations: metadata/attachment capture must be additive, never replacing
  the message path.

All timestamps are fixed so inspection metadata is stable across regenerations.
"""

import json
from pathlib import Path

from mcap.writer import CompressionType, Writer

BASE_NS = 1_700_000_000_000_000_000  # same fixed epoch as make_sample_mcap.py

OUT = Path(__file__).parent / "records.mcap"

# Mirrors the single metadata record nuScenes mini files carry (name scene-info).
SCENE_INFO = {
    "description": "Parked truck, construction, intersection, turn left",
    "name": "scene-0061",
    "location": "singapore-onenorth",
    "vehicle": "n015",
    "date_captured": "2018-07-24",
}

CALIBRATION = json.dumps({"camera_matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]}).encode()
INTRINSICS = bytes(range(64))  # opaque binary stand-in for a calibration blob


def main() -> None:
    with OUT.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start(profile="", library="lancedb-robotics-fixture")

        imu_schema = writer.register_schema(
            name="sample.Imu",
            encoding="jsonschema",
            data=json.dumps(
                {"type": "object", "properties": {"gyro_z": {"type": "number"}}}
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

        # Log-level metadata record (distinct from channel metadata).
        writer.add_metadata(name="scene-info", data=SCENE_INFO)

        # Two attachments with distinct media types; log_time order is the tie
        # break the adapter sorts on, so give them increasing log_times.
        writer.add_attachment(
            create_time=BASE_NS,
            log_time=BASE_NS + 10_000_000,
            name="calibration.json",
            media_type="application/json",
            data=CALIBRATION,
        )
        writer.add_attachment(
            create_time=BASE_NS,
            log_time=BASE_NS + 20_000_000,
            name="intrinsics.bin",
            media_type="application/octet-stream",
            data=INTRINSICS,
        )

        writer.finish()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
