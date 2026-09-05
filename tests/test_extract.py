"""Typed field extraction tests (backlog 0015).

These pin the schema-name -> vector-layout contract: a decoded message dict in,
the documented ``state_vector`` / ``action_vector`` / ``modality`` out. Extraction
runs on the canonical decoded dict produced by 0014 (``payload_json``), so a
single normalized shape covers all encoding families -- and equivalent types
across families (``sensor_msgs/Imu`` vs jsonschema ``IMU``; ``sensor_msgs/
NavSatFix`` vs ``foxglove.LocationFix``) must land the same layout + modality.

The mapping is versioned: ``LAYOUT_VERSION`` and ``LAYOUTS`` document each
vector's component order so the numbers stay interpretable across runs.
"""

import pytest

from lancedb_robotics.extract import (
    LAYOUT_VERSION,
    LAYOUTS,
    extract,
)

# --- canonical decoded message dicts (as 0014's payload_json would hold) ----

# ROS sensor_msgs/Imu: nested vec3/quaternion substructs.
ROS_IMU = {
    "header": {"frame_id": "imu_link", "stamp": {"sec": 1, "nanosec": 0}},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    "angular_velocity": {"x": 0.1, "y": 0.2, "z": 0.3},
    "linear_acceleration": {"x": 9.8, "y": 0.0, "z": 0.0},
}
# foxglove jsonschema IMU: q / rotation_rate / linear_accel keys.
JSON_IMU = {
    "q": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    "rotation_rate": {"x": 0.1, "y": 0.2, "z": 0.3},
    "linear_accel": {"x": 9.8, "y": 0.0, "z": 0.0},
}
# Both IMU forms must yield the same state vector.
IMU_STATE = [0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 9.8, 0.0, 0.0]

ROS_NAVSAT = {
    "header": {"frame_id": "gps"},
    "status": {"status": 0, "service": 1},
    "latitude": 37.4,
    "longitude": -122.1,
    "altitude": 30.5,
}
FG_LOCATIONFIX = {
    "frame_id": "gps",
    "latitude": 37.4,
    "longitude": -122.1,
    "altitude": 30.5,
}
GPS_STATE = [37.4, -122.1, 30.5]

ROS_TWIST_STAMPED = {
    "header": {"frame_id": "base_link"},
    "twist": {
        "linear": {"x": 1.0, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.5},
    },
}
ROS_TWIST = {
    "linear": {"x": 1.0, "y": 0.0, "z": 0.0},
    "angular": {"x": 0.0, "y": 0.0, "z": 0.5},
}
TWIST_ACTION = [1.0, 0.0, 0.0, 0.0, 0.0, 0.5]

ROS_ODOM = {
    "header": {"frame_id": "odom"},
    "child_frame_id": "base_link",
    "pose": {
        "pose": {
            "position": {"x": 2.0, "y": 3.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "covariance": [0.0] * 36,
    },
    "twist": {
        "twist": {
            "linear": {"x": 1.0, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": 0.5},
        },
        "covariance": [0.0] * 36,
    },
}
ODOM_STATE = [2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0]

FG_POSE_IN_FRAME = {
    "timestamp": {"sec": 1, "nsec": 0},
    "frame_id": "map",
    "pose": {
        "position": {"x": 2.0, "y": 3.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    },
}
JSON_POSE = {
    "pos": {"x": 2.0, "y": 3.0, "z": 0.0},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    "vel": {"x": 1.0, "y": 0.0, "z": 0.0},
    "accel": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation_rate": {"x": 0.0, "y": 0.0, "z": 0.5},
}
POSE_STATE = [2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0]

ROS_RANGE = {
    "header": {"frame_id": "sonar"},
    "radiation_type": 0,
    "field_of_view": 0.5,
    "min_range": 0.02,
    "max_range": 4.0,
    "range": 1.25,
}
RANGE_STATE = [1.25, 0.02, 4.0, 0.5]

# XDOF ABC-130k protobuf RobotState/GripperState (real field shapes, see
# docs/YAM_DATA_FORMAT.md): observed streams carry position/velocity/torque,
# commanded (action) streams carry position only -- same schema name either
# way, so extract() types both identically (see LAYOUTS' "joint_state"/
# "gripper" comments for why that's fine: a facade separates observed vs.
# commanded by stream/topic name, not by extract.py's output field).
YAM_ROBOT_STATE = {
    "timestamp": {"seconds": 1751060476, "nanos": 85970432},
    "position": [-0.57, 0.84, 0.68, -0.12, 0.27, -0.81],
    "velocity": [-2.34, 1.11, 1.08, 0.02, -1.07, 0.75, -0.14],
    "torque": [-2.78, 0.69, 11.98, 2.34, -0.01, 0.25, 0.15],
}
JOINT_STATE = [-0.57, 0.84, 0.68, -0.12, 0.27, -0.81]
YAM_ROBOT_ACTION = {
    "timestamp": {"seconds": 1751060476, "nanos": 85970432},
    "position": [-0.60, 0.87, 0.71, -0.11, 0.24, -0.79],
}
JOINT_ACTION = [-0.60, 0.87, 0.71, -0.11, 0.24, -0.79]

YAM_GRIPPER_STATE = {
    "timestamp": {"seconds": 1751060476, "nanos": 85970432},
    "position": [0.957],
    "velocity": [0.0],
    "torque": [0.1],
}
GRIPPER_STATE = [0.957]
YAM_GRIPPER_ACTION = {
    "timestamp": {"seconds": 1751060476, "nanos": 85970432},
    "position": [0.947],
}
GRIPPER_ACTION = [0.947]

# Voxel51 RoboLab-EgoX protobuf RobotState (real field values, decoded
# directly from a real episode.fo.mcap): the *same bare schema name*
# "RobotState" as XDOF ABC-130k above -- a genuine cross-dataset collision,
# confirmed against real data from both producers. Disambiguated by field
# length (13 on /joint-positions, 8 on /actions), never by content/topic.
ROBOLAB_JOINT_STATE = {
    "timestamp": {"seconds": 1000000000},
    "position": [
        -0.295654296875,
        0.401611328125,
        -0.25732421875,
        -1.6611328125,
        0.192138671875,
        1.9775390625,
        0.2802734375,
        1.1920928955078125e-07,
        4.1723251342773438e-07,
        -3.2067298889160156e-05,
        3.2603740692138672e-05,
        2.9385089874267578e-05,
        2.6822090148925781e-05,
    ],
}
ROBOLAB_JOINT_STATE_VECTOR = ROBOLAB_JOINT_STATE["position"]
ROBOLAB_ACTION = {
    "timestamp": {"seconds": 1000000000},
    "position": [
        -0.311767578125,
        0.455322265625,
        -0.26220703125,
        -1.6123046875,
        0.20361328125,
        1.9658203125,
        0.29833984375,
        0.0,
    ],
}
ROBOLAB_ACTION_VECTOR = ROBOLAB_ACTION["position"]


# --- per-type extraction (table-driven over the mapped type set) ------------


@pytest.mark.parametrize(
    "schema_name,payload,modality,state,action",
    [
        ("sensor_msgs/Imu", ROS_IMU, "imu", IMU_STATE, None),
        ("sensor_msgs/msg/Imu", ROS_IMU, "imu", IMU_STATE, None),
        ("IMU", JSON_IMU, "imu", IMU_STATE, None),
        ("sensor_msgs/NavSatFix", ROS_NAVSAT, "gps", GPS_STATE, None),
        ("sensor_msgs/msg/NavSatFix", ROS_NAVSAT, "gps", GPS_STATE, None),
        ("foxglove.LocationFix", FG_LOCATIONFIX, "gps", GPS_STATE, None),
        ("geometry_msgs/TwistStamped", ROS_TWIST_STAMPED, "twist", None, TWIST_ACTION),
        ("geometry_msgs/msg/TwistStamped", ROS_TWIST_STAMPED, "twist", None, TWIST_ACTION),
        ("geometry_msgs/Twist", ROS_TWIST, "twist", None, TWIST_ACTION),
        ("nav_msgs/Odometry", ROS_ODOM, "odometry", ODOM_STATE, TWIST_ACTION),
        ("foxglove.PoseInFrame", FG_POSE_IN_FRAME, "pose", POSE_STATE, None),
        ("Pose", JSON_POSE, "pose", POSE_STATE, None),
        ("sensor_msgs/Range", ROS_RANGE, "range", RANGE_STATE, None),
        ("RobotState", YAM_ROBOT_STATE, "joint_state", JOINT_STATE, None),
        ("GripperState", YAM_GRIPPER_STATE, "gripper", GRIPPER_STATE, None),
        ("RobotState", ROBOLAB_JOINT_STATE, "robolab_joint_state", ROBOLAB_JOINT_STATE_VECTOR, None),
        ("RobotState", ROBOLAB_ACTION, "robolab_action", ROBOLAB_ACTION_VECTOR, None),
    ],
)
def test_typed_extraction(schema_name, payload, modality, state, action):
    result = extract(schema_name, payload)
    assert result.modality == modality
    assert result.state_vector == (state if state is None else pytest.approx(state))
    assert result.action_vector == (action if action is None else pytest.approx(action))


@pytest.mark.parametrize(
    "schema_name,modality",
    [
        ("sensor_msgs/Image", "image"),
        ("sensor_msgs/msg/Image", "image"),
        ("sensor_msgs/CompressedImage", "image"),
        ("foxglove.CompressedImage", "image"),
        ("sensor_msgs/PointCloud2", "pointcloud"),
        ("foxglove.PointCloud", "pointcloud"),
        ("radar_driver/RadarTracks", "radar"),
        ("diagnostic_msgs/DiagnosticArray", "diagnostic"),
    ],
)
def test_binary_and_aggregate_types_get_modality_without_vectors(schema_name, modality):
    # Image/pointcloud/radar/diagnostic are typed by modality but carry no
    # state/action vector: the heavy/structured payload stays in payload_json.
    result = extract(schema_name, {"data": b"", "width": 1, "height": 1, "encoding": "rgb8"})
    assert result.modality == modality
    assert result.state_vector is None
    assert result.action_vector is None


# --- cross-family equivalence ----------------------------------------------


def test_imu_cross_family_equivalence():
    ros = extract("sensor_msgs/Imu", ROS_IMU)
    fg = extract("IMU", JSON_IMU)
    assert ros.modality == fg.modality == "imu"
    assert ros.state_vector == pytest.approx(fg.state_vector)


def test_gps_cross_family_equivalence():
    ros = extract("sensor_msgs/NavSatFix", ROS_NAVSAT)
    fg = extract("foxglove.LocationFix", FG_LOCATIONFIX)
    assert ros.modality == fg.modality == "gps"
    assert ros.state_vector == pytest.approx(fg.state_vector)


def test_yam_robot_state_schema_is_topic_agnostic_state_or_action():
    # RobotState/GripperState carry no topic identity of their own -- the same
    # schema serves both an observed (*-state) and a commanded (*-action)
    # topic, and extract() has no topic argument to tell them apart. It
    # always types both as a "state_vector" (never "action_vector"); a
    # facade's CanonicalVectorMapping is what separates observed vs.
    # commanded, by declaring which *stream names* (topics) are state vs.
    # action -- not by which extract.py field got populated.
    action_result = extract("RobotState", YAM_ROBOT_ACTION)
    assert action_result.modality == "joint_state"
    assert action_result.state_vector == pytest.approx(JOINT_ACTION)
    assert action_result.action_vector is None

    gripper_action_result = extract("GripperState", YAM_GRIPPER_ACTION)
    assert gripper_action_result.modality == "gripper"
    assert gripper_action_result.state_vector == pytest.approx(GRIPPER_ACTION)
    assert gripper_action_result.action_vector is None


def test_robotstate_schema_name_collision_disambiguated_by_field_length():
    # XDOF ABC-130k (6-length position) and Voxel51 RoboLab-EgoX (13- and
    # 8-length position) both use the bare schema name "RobotState" --
    # extract() must not silently conflate them just because the name matches.
    yam = extract("RobotState", YAM_ROBOT_STATE)
    robolab_state = extract("RobotState", ROBOLAB_JOINT_STATE)
    robolab_action = extract("RobotState", ROBOLAB_ACTION)
    assert yam.modality == "joint_state"
    assert robolab_state.modality == "robolab_joint_state"
    assert robolab_action.modality == "robolab_action"
    assert yam.state_vector == pytest.approx(JOINT_STATE)
    assert robolab_state.state_vector == pytest.approx(ROBOLAB_JOINT_STATE_VECTOR)
    assert robolab_action.state_vector == pytest.approx(ROBOLAB_ACTION_VECTOR)


def test_robotstate_with_unrecognized_length_yields_no_modality():
    # A hypothetical 4th "RobotState" producer with an unrecognized length
    # must not be silently guessed as one of the 3 known shapes.
    result = extract("RobotState", {"position": [1.0, 2.0]})
    assert result.modality is None
    assert result.state_vector is None


def test_robotstate_with_no_payload_yields_no_modality():
    # Unlike single-candidate schema names (see
    # test_missing_payload_still_types_modality_from_schema_name below),
    # "RobotState" is genuinely ambiguous without a payload to check length
    # against -- a raw/failed row with this schema name honestly can't be
    # resolved to one of the 3 real shapes, so modality stays None rather
    # than guessing.
    result = extract("RobotState", None)
    assert result.modality is None
    assert result.state_vector is None


# --- structural-matcher fallback (unknown schema name, known shape) ---------


def test_structural_fallback_routes_imu_shaped_unknown_schema():
    # A vendor schema name we never registered, but an Imu-shaped payload.
    result = extract("acme_msgs/CustomImu", ROS_IMU)
    assert result.modality == "imu"
    assert result.state_vector == pytest.approx(IMU_STATE)


def test_structural_fallback_routes_json_imu_shape():
    result = extract("vendor/Telemetry", JSON_IMU)
    assert result.modality == "imu"
    assert result.state_vector == pytest.approx(IMU_STATE)


def test_structural_fallback_routes_navsat_shape():
    result = extract("acme/GnssReport", FG_LOCATIONFIX)
    assert result.modality == "gps"
    assert result.state_vector == pytest.approx(GPS_STATE)


# --- unmapped / malformed: NULL vectors, best-effort modality, no crash -----


def test_unmapped_type_yields_null_vectors_and_no_modality():
    result = extract("acme_msgs/Mystery", {"foo": 1, "bar": [2, 3]})
    assert result.modality is None  # caller falls back to a topic-based guess
    assert result.state_vector is None
    assert result.action_vector is None


def test_missing_payload_still_types_modality_from_schema_name():
    # raw/failed rows have no decoded payload, but the schema name alone is
    # enough to assign a modality (vectors stay NULL, never fabricated).
    result = extract("sensor_msgs/Imu", None)
    assert result.modality == "imu"
    assert result.state_vector is None
    assert result.action_vector is None


def test_malformed_known_type_does_not_crash_and_yields_null_vector():
    # Right schema name, wrong shape (missing the substructs): no exception,
    # modality still assigned, vector NULL rather than fabricated.
    result = extract("sensor_msgs/Imu", {"orientation": "not-a-vector"})
    assert result.modality == "imu"
    assert result.state_vector is None


def test_none_schema_and_none_payload_is_inert():
    result = extract(None, None)
    assert result.modality is None
    assert result.state_vector is None
    assert result.action_vector is None


# --- versioned, documented layout table -------------------------------------


def test_layout_table_documents_every_emitted_vector():
    assert LAYOUT_VERSION  # a non-empty version string
    # Every layout the extractor can emit is documented with an ordered list of
    # component names whose length matches the vectors produced above.
    assert len(LAYOUTS["imu"]) == len(IMU_STATE)
    assert len(LAYOUTS["gps"]) == len(GPS_STATE)
    assert len(LAYOUTS["pose"]) == len(POSE_STATE)
    assert len(LAYOUTS["twist"]) == len(TWIST_ACTION)
    assert len(LAYOUTS["range"]) == len(RANGE_STATE)
    assert len(LAYOUTS["joint_state"]) == len(JOINT_STATE)
    assert len(LAYOUTS["gripper"]) == len(GRIPPER_STATE)
    assert len(LAYOUTS["robolab_joint_state"]) == len(ROBOLAB_JOINT_STATE_VECTOR)
    assert len(LAYOUTS["robolab_action"]) == len(ROBOLAB_ACTION_VECTOR)
