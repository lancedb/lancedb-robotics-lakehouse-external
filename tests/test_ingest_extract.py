"""Ingest typed-extraction integration tests (backlog 0015).

End-to-end: a small ros2 fixture carrying an Imu, a TwistStamped, and a
NavSatFix is decoded (0014) and then typed-extracted (0015), so the persisted
``observations`` rows have populated ``state_vector`` / ``action_vector`` and a
type-derived ``modality`` -- the training-preview consumers see real values, not
NULLs. Needs the ``mcap_ros2`` extra; skipped when absent (same pattern as the
0014 cdr tests).
"""

import io
import json

import pytest

from lancedb_robotics.extract import LAYOUT_VERSION
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake

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

_TWIST_STAMPED_DEF = """
geometry_msgs/Twist twist
================================================================================
MSG: geometry_msgs/Twist
geometry_msgs/Vector3 linear
geometry_msgs/Vector3 angular
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
"""

_NAVSAT_DEF = """
float64 latitude
float64 longitude
float64 altitude
"""


def _write_fixture(path):
    from mcap_ros2.writer import Writer as Ros2Writer

    buf = io.BytesIO()
    writer = Ros2Writer(buf)
    imu_schema = writer.register_msgdef("sensor_msgs/msg/Imu", _IMU_DEF)
    twist_schema = writer.register_msgdef("geometry_msgs/msg/TwistStamped", _TWIST_STAMPED_DEF)
    navsat_schema = writer.register_msgdef("sensor_msgs/msg/NavSatFix", _NAVSAT_DEF)
    writer.write_message(
        topic="/vehicle/imu/data_raw",
        schema=imu_schema,
        message={
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "angular_velocity": {"x": 0.1, "y": 0.2, "z": 0.3},
            "linear_acceleration": {"x": 9.8, "y": 0.0, "z": 0.0},
        },
        log_time=10,
        publish_time=10,
    )
    writer.write_message(
        topic="/vehicle/twist",
        schema=twist_schema,
        message={
            "twist": {
                "linear": {"x": 1.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.5},
            }
        },
        log_time=20,
        publish_time=20,
    )
    writer.write_message(
        topic="/vehicle/gps/fix",
        schema=navsat_schema,
        message={"latitude": 37.4, "longitude": -122.1, "altitude": 30.5},
        log_time=30,
        publish_time=30,
    )
    writer.finish()
    path.write_bytes(buf.getvalue())
    return path


@pytest.fixture
def fixture_mcap(tmp_path):
    pytest.importorskip("mcap_ros2")
    return _write_fixture(tmp_path / "ros2_typed.mcap")


@pytest.fixture
def rows(tmp_path, fixture_mcap):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixture_mcap)
    return {
        r["topic"]: r for r in lake.table("observations").to_arrow().to_pylist()
    }


def test_imu_observation_has_documented_state_vector(rows):
    imu = rows["/vehicle/imu/data_raw"]
    assert imu["modality"] == "imu"
    assert imu["state_vector"] == pytest.approx(
        [0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 9.8, 0.0, 0.0]
    )
    assert imu["action_vector"] is None


def test_twist_observation_has_documented_action_vector(rows):
    twist = rows["/vehicle/twist"]
    assert twist["modality"] == "twist"
    assert twist["action_vector"] == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    assert twist["state_vector"] is None


def test_navsat_observation_has_lat_lon_alt_state_vector(rows):
    gps = rows["/vehicle/gps/fix"]
    assert gps["modality"] == "gps"
    assert gps["state_vector"] == pytest.approx([37.4, -122.1, 30.5])


def test_extraction_coverage_recorded_in_transform_params(tmp_path, fixture_mcap):
    lake = Lake.init(tmp_path / "robot2.lance")
    ingest_mcap(lake, fixture_mcap)
    ingest = next(
        t for t in lake.table("transform_runs").to_arrow().to_pylist() if t["kind"] == "ingest"
    )
    params = json.loads(ingest["params"])
    assert params["extract_layout_version"] == LAYOUT_VERSION
    assert params["extracted_by_modality"] == {"gps": 1, "imu": 1, "twist": 1}
