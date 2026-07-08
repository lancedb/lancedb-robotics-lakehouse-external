"""Regenerate tests/fixtures/split_recording/ (backlog 0019).

Run: uv run python tests/fixtures/make_split_recording.py

A rosbag2-style *split recording*: one logical run cut into a directory of three
shards plus a ``metadata.yaml`` that records the ordered shard list and the
recording's overall time range::

    split_recording/
      metadata.yaml
      recording_0.mcap
      recording_1.mcap
      recording_2.mcap

The shards tile one continuous timeline. Every "tick" carries one ``/imu`` and
one ``/gps`` json message at the same ``log_time``; ticks 0-2 live in shard 0,
3-5 in shard 1, 6-8 in shard 2. So each topic has nine messages whose per-topic
sequence must stay continuous (0..8) across shard boundaries, and the merged
inspection of the directory must equal a single-file inspect of the same nine
ticks concatenated (``build_combined_bytes``).

Shard names sort lexicographically into the same order the ``metadata.yaml``
declares, so a bare directory (no manifest) resolves the identical shard order —
and therefore the identical content-addressed ``run_id``.

All timestamps are fixed, the writer uses no compression, so every shard is
byte-stable across regenerations (verified by tests/test_fixture_tooling.py).
"""

import json
from pathlib import Path

from mcap.writer import CompressionType, Writer

BASE_NS = 1_700_000_000_000_000_000  # same fixed epoch as make_sample_mcap.py
STEP_NS = 10_000_000  # 10 ms between ticks
N_SHARDS = 3
TICKS_PER_SHARD = 3
TOTAL_TICKS = N_SHARDS * TICKS_PER_SHARD

OUT_DIR = Path(__file__).parent / "split_recording"
SHARD_NAMES = [f"recording_{i}.mcap" for i in range(N_SHARDS)]
METADATA_NAME = "metadata.yaml"

_IMU_SCHEMA = json.dumps({"type": "object", "properties": {"gyro_z": {"type": "number"}}}).encode()
_GPS_SCHEMA = json.dumps({"type": "object", "properties": {"lat": {"type": "number"}}}).encode()


def _tick_time(tick: int) -> int:
    return BASE_NS + tick * STEP_NS


def _write_ticks(stream, ticks: range) -> None:
    """Write one /imu + one /gps json message per tick into an open MCAP writer."""
    writer = Writer(stream, compression=CompressionType.NONE)
    writer.start(profile="", library="lancedb-robotics-fixture")
    imu_schema = writer.register_schema(name="sample.Imu", encoding="jsonschema", data=_IMU_SCHEMA)
    gps_schema = writer.register_schema(name="sample.Gps", encoding="jsonschema", data=_GPS_SCHEMA)
    imu_channel = writer.register_channel(topic="/imu", message_encoding="json", schema_id=imu_schema)
    gps_channel = writer.register_channel(topic="/gps", message_encoding="json", schema_id=gps_schema)
    for tick in ticks:
        log_time = _tick_time(tick)
        writer.add_message(
            channel_id=imu_channel,
            log_time=log_time,
            publish_time=log_time,
            data=json.dumps({"gyro_z": 0.1 * tick}).encode(),
        )
        writer.add_message(
            channel_id=gps_channel,
            log_time=log_time,
            publish_time=log_time,
            data=json.dumps({"lat": 1.0 * tick}).encode(),
        )
    writer.finish()


def _shard_ticks(shard_index: int) -> range:
    start = shard_index * TICKS_PER_SHARD
    return range(start, start + TICKS_PER_SHARD)


def build_shard_bytes(shard_index: int) -> bytes:
    """Deterministic bytes for one shard (used by the generator and the test)."""
    import io

    buf = io.BytesIO()
    _write_ticks(buf, _shard_ticks(shard_index))
    return buf.getvalue()


def build_combined_bytes() -> bytes:
    """All ticks in a single file — the single-file equivalent of the recording."""
    import io

    buf = io.BytesIO()
    _write_ticks(buf, range(TOTAL_TICKS))
    return buf.getvalue()


def build_metadata_yaml() -> str:
    """A rosbag2-style metadata.yaml declaring the ordered shards and time range."""
    start_ns = _tick_time(0)
    end_ns = _tick_time(TOTAL_TICKS - 1)
    shard_lines = "\n".join(f"    - {name}" for name in SHARD_NAMES)
    return (
        "rosbag2_bagfile_information:\n"
        "  version: 5\n"
        "  storage_identifier: mcap\n"
        "  duration:\n"
        f"    nanoseconds: {end_ns - start_ns}\n"
        "  starting_time:\n"
        f"    nanoseconds_since_epoch: {start_ns}\n"
        f"  message_count: {TOTAL_TICKS * 2}\n"
        "  relative_file_paths:\n"
        f"{shard_lines}\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(SHARD_NAMES):
        (OUT_DIR / name).write_bytes(build_shard_bytes(index))
    (OUT_DIR / METADATA_NAME).write_text(build_metadata_yaml())
    total = sum((OUT_DIR / name).stat().st_size for name in SHARD_NAMES)
    print(f"wrote {OUT_DIR} ({N_SHARDS} shards, {total} bytes + {METADATA_NAME})")


if __name__ == "__main__":
    main()
