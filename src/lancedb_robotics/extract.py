"""Typed robotics field extraction (backlog 0015).

0014 decodes each MCAP message into a canonical JSON dict (``payload_json``).
That dict is still generic; this module turns the known robot message types into
the model-ready fields the downstream loop consumes: ``state_vector`` (sensor
state), ``action_vector`` (control/command), and a type-derived ``modality``
(replacing the topic-substring guess in ``ingest._modality``).

Because 0014 already normalized every encoding family (ros1/ros2 ``__slots__``,
protobuf ``MessageToDict``, flatbuffer ``.bfbs`` reflection, jsonschema dicts)
into one snake-case dict shape, extraction runs on a **single dict shape**.
Equivalent types across families therefore collapse to the same layout
automatically: ``sensor_msgs/Imu`` and the foxglove jsonschema ``IMU`` both yield
the ``imu`` layout; ``sensor_msgs/NavSatFix`` and ``foxglove.LocationFix`` both
yield ``gps``. In particular a flatbuffer ``foxglove.*`` message routes through
the same :data:`_BY_SCHEMA` name as its protobuf twin and yields the identical
vector -- encoding is an implementation detail (backlog 0020). The flatbuffer
decoder emits non-zero schema defaults (foxglove ``Vector3``/``Quaternion``
default components to 1.0), so a flatbuffer vector left at its default still
matches the protobuf twin, whose proto3 1.0s are always written explicitly.

Routing is exact-schema-name first (:data:`_BY_SCHEMA`), then a structural
fallback (:data:`_STRUCTURAL`, ordered most-specific-first) that types an unknown
schema name by the *shape* of its decoded fields -- so a vendor ``acme/CustomImu``
still lands as ``imu``. Anything that matches nothing returns ``modality=None``
(the caller falls back to a topic-based guess) with NULL vectors.

The vector layouts are a **versioned, documented contract** (:data:`LAYOUTS` /
:data:`LAYOUT_VERSION`): each emitted vector's component order is fixed so the
numbers stay interpretable and stable across runs. Missing or malformed fields
yield NULL vectors -- never a fabricated number and never an exception.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Bump when a layout's component order/length changes. Vectors written under a
# given version are interpretable against the LAYOUTS table for that version.
LAYOUT_VERSION = "1"

# modality -> ordered component names. The single source of truth for what each
# emitted vector means; documented in docs and asserted by tests.
LAYOUTS: dict[str, tuple[str, ...]] = {
    # state vectors
    "imu": (
        "orientation_x", "orientation_y", "orientation_z", "orientation_w",
        "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
        "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
    ),
    "gps": ("latitude", "longitude", "altitude"),
    "pose": (
        "position_x", "position_y", "position_z",
        "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    ),
    "range": ("range", "min_range", "max_range", "field_of_view"),
    # action vector
    "twist": (
        "linear_x", "linear_y", "linear_z",
        "angular_x", "angular_y", "angular_z",
    ),
}


@dataclass(frozen=True)
class ExtractResult:
    """Typed fields extracted from one decoded message.

    ``modality`` is ``None`` when the message could not be typed (the caller then
    falls back to a topic-based guess). ``state_vector`` / ``action_vector`` are
    ``None`` when the type has no such vector or its fields were absent/malformed.
    """

    modality: str | None = None
    state_vector: list[float] | None = None
    action_vector: list[float] | None = None


VectorFn = Callable[[dict], list[float] | None]
"""decoded message dict -> a numeric vector, or None if its fields are absent."""

MatchFn = Callable[[dict], bool]
"""decoded message dict -> True if its field shape matches this type."""


# --- field accessors --------------------------------------------------------
#
# Decoded values come from typed messages, so they are numbers; proto3 may omit
# fields left at their default, so absent scalars read as 0.0. A non-dict or a
# non-numeric value raises, which the extract() wrapper turns into a NULL vector.


def _num(value: object) -> float:
    return float(value)  # type: ignore[arg-type]  # non-numeric -> caught upstream


def _vec3(d: object) -> list[float]:
    assert isinstance(d, dict)
    return [_num(d.get("x", 0.0)), _num(d.get("y", 0.0)), _num(d.get("z", 0.0))]


def _quat(d: object) -> list[float]:
    assert isinstance(d, dict)
    return [
        _num(d.get("x", 0.0)),
        _num(d.get("y", 0.0)),
        _num(d.get("z", 0.0)),
        _num(d.get("w", 1.0)),
    ]


def _twist_vec(twist: object) -> list[float]:
    """``{linear: vec3, angular: vec3}`` -> the 6-D twist action vector."""
    assert isinstance(twist, dict)
    return _vec3(twist["linear"]) + _vec3(twist["angular"])


def _pose_vec(pose: object) -> list[float]:
    """``{position: vec3, orientation: quat}`` -> the 7-D pose state vector."""
    assert isinstance(pose, dict)
    return _vec3(pose["position"]) + _quat(pose["orientation"])


# --- per-type vector builders ----------------------------------------------


def _imu_ros_state(msg: dict) -> list[float]:
    return (
        _quat(msg["orientation"])
        + _vec3(msg["angular_velocity"])
        + _vec3(msg["linear_acceleration"])
    )


def _imu_json_state(msg: dict) -> list[float]:
    return _quat(msg["q"]) + _vec3(msg["rotation_rate"]) + _vec3(msg["linear_accel"])


def _gps_state(msg: dict) -> list[float]:
    return [_num(msg["latitude"]), _num(msg["longitude"]), _num(msg.get("altitude", 0.0))]


def _twist_stamped_action(msg: dict) -> list[float]:
    return _twist_vec(msg["twist"])


def _twist_action(msg: dict) -> list[float]:
    return _twist_vec(msg)


def _odom_state(msg: dict) -> list[float]:
    return _pose_vec(msg["pose"]["pose"])


def _odom_action(msg: dict) -> list[float]:
    return _twist_vec(msg["twist"]["twist"])


def _pose_in_frame_state(msg: dict) -> list[float]:
    return _pose_vec(msg["pose"])


def _json_pose_state(msg: dict) -> list[float]:
    return _vec3(msg["pos"]) + _quat(msg["orientation"])


def _range_state(msg: dict) -> list[float]:
    return [
        _num(msg["range"]),
        _num(msg.get("min_range", 0.0)),
        _num(msg.get("max_range", 0.0)),
        _num(msg.get("field_of_view", 0.0)),
    ]


# --- type handlers ----------------------------------------------------------


@dataclass(frozen=True)
class _Type:
    """One robot message type: its modality and how to build its vectors.

    ``match`` is the structural predicate used when an unknown schema name has
    this type's field shape; ``None`` means the type is exact-name-only.
    """

    modality: str
    state: VectorFn | None = None
    action: VectorFn | None = None
    match: MatchFn | None = None


def _has(msg: dict, *names: str) -> bool:
    return all(n in msg for n in names)


_IMU_ROS = _Type(
    "imu", state=_imu_ros_state,
    match=lambda m: _has(m, "orientation", "angular_velocity", "linear_acceleration"),
)
_IMU_JSON = _Type(
    "imu", state=_imu_json_state,
    match=lambda m: _has(m, "q", "rotation_rate", "linear_accel"),
)
_NAVSAT = _Type(
    "gps", state=_gps_state,
    match=lambda m: _has(m, "latitude", "longitude", "altitude"),
)
_TWIST_STAMPED = _Type(
    "twist", action=_twist_stamped_action,
    match=lambda m: isinstance(m.get("twist"), dict) and _has(m["twist"], "linear", "angular"),
)
_TWIST = _Type(
    "twist", action=_twist_action,
    match=lambda m: _has(m, "linear", "angular"),
)
_ODOM = _Type(
    "odometry", state=_odom_state, action=_odom_action,
    match=lambda m: _has(m, "pose", "twist", "child_frame_id"),
)
_POSE_IN_FRAME = _Type(
    "pose", state=_pose_in_frame_state,
    match=lambda m: isinstance(m.get("pose"), dict)
    and _has(m["pose"], "position", "orientation")
    and _has(m, "frame_id"),
)
_JSON_POSE = _Type(
    "pose", state=_json_pose_state,
    match=lambda m: _has(m, "pos", "orientation"),
)
_RANGE = _Type(
    "range", state=_range_state,
    match=lambda m: _has(m, "range", "radiation_type", "field_of_view", "min_range", "max_range"),
)
# Binary / aggregate types: typed by modality, no state/action vector (the
# structured payload stays in payload_json). Ordered before the scalar types
# in _STRUCTURAL so e.g. a PointCloud isn't mistaken for something else.
_IMAGE = _Type("image", match=lambda m: _has(m, "data", "width", "height", "encoding"))
_PC_ROS = _Type("pointcloud", match=lambda m: _has(m, "data", "fields", "point_step"))
_PC_FG = _Type("pointcloud", match=lambda m: _has(m, "data", "fields", "point_stride"))
_COMPRESSED_IMAGE = _Type(
    "image", match=lambda m: _has(m, "data", "format") and "is_key_frame" not in m
)
_RADAR = _Type("radar")
_DIAGNOSTIC = _Type("diagnostic")


# Exact schema-name -> handler. Both the legacy ``pkg/Type`` and the ROS2
# ``pkg/msg/Type`` spellings map to the same handler.
_BY_SCHEMA: dict[str, _Type] = {
    "sensor_msgs/Imu": _IMU_ROS,
    "sensor_msgs/msg/Imu": _IMU_ROS,
    "IMU": _IMU_JSON,
    "sensor_msgs/NavSatFix": _NAVSAT,
    "sensor_msgs/msg/NavSatFix": _NAVSAT,
    "foxglove.LocationFix": _NAVSAT,
    "geometry_msgs/TwistStamped": _TWIST_STAMPED,
    "geometry_msgs/msg/TwistStamped": _TWIST_STAMPED,
    "geometry_msgs/Twist": _TWIST,
    "geometry_msgs/msg/Twist": _TWIST,
    "nav_msgs/Odometry": _ODOM,
    "nav_msgs/msg/Odometry": _ODOM,
    "foxglove.PoseInFrame": _POSE_IN_FRAME,
    "Pose": _JSON_POSE,
    "sensor_msgs/Range": _RANGE,
    "sensor_msgs/msg/Range": _RANGE,
    "sensor_msgs/Image": _IMAGE,
    "sensor_msgs/msg/Image": _IMAGE,
    "sensor_msgs/CompressedImage": _IMAGE,
    "sensor_msgs/msg/CompressedImage": _IMAGE,
    "foxglove.CompressedImage": _IMAGE,
    "foxglove.CompressedVideo": _IMAGE,
    "sensor_msgs/PointCloud2": _PC_ROS,
    "sensor_msgs/msg/PointCloud2": _PC_ROS,
    "foxglove.PointCloud": _PC_FG,
    "radar_driver/RadarTracks": _RADAR,
    "diagnostic_msgs/DiagnosticArray": _DIAGNOSTIC,
    "diagnostic_msgs/msg/DiagnosticArray": _DIAGNOSTIC,
}

# Structural fallback, ordered most-specific-first. Binary/aggregate shapes are
# tried before the scalar sensor shapes; TwistStamped (nested ``twist``) before
# bare Twist; ros-Imu before the leaner json-Imu.
_STRUCTURAL: tuple[_Type, ...] = (
    _IMAGE,
    _PC_ROS,
    _PC_FG,
    _COMPRESSED_IMAGE,
    _RANGE,
    _ODOM,
    _POSE_IN_FRAME,
    _TWIST_STAMPED,
    _IMU_ROS,
    _IMU_JSON,
    _NAVSAT,
    _JSON_POSE,
    _TWIST,
)


def _structural_match(payload: dict) -> _Type | None:
    for handler in _STRUCTURAL:
        if handler.match is None:
            continue
        try:
            if handler.match(payload):
                return handler
        except Exception:  # noqa: BLE001 - a predicate must never crash extraction
            continue
    return None


def _safe(fn: VectorFn | None, payload: dict | None) -> list[float] | None:
    if fn is None or not isinstance(payload, dict):
        return None
    try:
        return fn(payload)
    except Exception:  # noqa: BLE001 - malformed fields yield NULL, never a crash
        return None


def extract(schema_name: str | None, payload: object) -> ExtractResult:
    """Type a decoded message into ``(modality, state_vector, action_vector)``.

    ``schema_name`` routes by exact name first; an unknown name with a decoded
    ``payload`` (dict) falls back to structural shape matching. ``payload`` may be
    ``None`` (raw/failed rows): the modality is still assigned from the schema
    name when known, with NULL vectors. Returns ``modality=None`` only when the
    type could not be determined at all.
    """
    handler = _BY_SCHEMA.get(schema_name or "")
    if handler is None and isinstance(payload, dict):
        handler = _structural_match(payload)
    if handler is None:
        return ExtractResult()
    return ExtractResult(
        modality=handler.modality,
        state_vector=_safe(handler.state, payload if isinstance(payload, dict) else None),
        action_vector=_safe(handler.action, payload if isinstance(payload, dict) else None),
    )


__all__ = [
    "LAYOUT_VERSION",
    "LAYOUTS",
    "ExtractResult",
    "extract",
]
