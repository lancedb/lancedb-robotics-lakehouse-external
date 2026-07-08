"""Regenerate tests/fixtures/ros2_imu.mcap (backlog 0017).

Run: uv run python tests/fixtures/make_ros2_imu_mcap.py

No corpus file uses ros2/cdr, so this fixture is *generated* with
``mcap-ros2-support``'s ``Writer`` rather than sliced. It carries three
``sensor_msgs/msg/Imu`` messages, which round-trip to ``message_encoding='cdr'``
and ``schema_encoding='ros2msg'`` -- the one decode path the real corpora cannot
provide. All timestamps are fixed so the metadata is stable across regenerations.

Requires the ``ros2`` extra (``mcap-ros2-support``).
"""

from pathlib import Path

from mcap_ros2.writer import Writer as Ros2Writer

OUT = Path(__file__).parent / "ros2_imu.mcap"

_IMU_DEF = """
geometry_msgs/Quaternion orientation
geometry_msgs/Vector3 angular_velocity
geometry_msgs/Vector3 linear_acceleration
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
"""

BASE_NS = 1_700_000_000_000_000_000  # fixed epoch so the fixture is deterministic


def build_bytes() -> bytes:
    """Return the fixture bytes (deterministic; used by main and the tooling test)."""
    import io

    buf = io.BytesIO()
    writer = Ros2Writer(buf)
    imu_schema = writer.register_msgdef("sensor_msgs/msg/Imu", _IMU_DEF)
    for i in range(3):
        writer.write_message(
            topic="/vehicle/imu/data_raw",
            schema=imu_schema,
            message={
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "angular_velocity": {"x": 0.1 * i, "y": 0.2, "z": 0.3},
                "linear_acceleration": {"x": 9.8, "y": 0.0, "z": 0.1 * i},
            },
            log_time=BASE_NS + i * 100_000_000,
            publish_time=BASE_NS + i * 100_000_000,
        )
    writer.finish()
    return buf.getvalue()


def main() -> None:
    OUT.write_bytes(build_bytes())
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
