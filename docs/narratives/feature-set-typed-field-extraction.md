# Typed Robotics Field Extraction: Decoded Payloads Become State/Action Vectors

## What changed

`ingest` now turns the canonical decoded payload from 0014 (`payload_json`) into
the model-ready fields the downstream loop consumes:

- `state_vector` — sensor state for IMU / GPS / pose / range types.
- `action_vector` — control/command for twist (and odometry) types.
- `modality` — type-derived (e.g. `imu`, `gps`, `twist`), replacing the
  topic-substring guess. The topic guess remains only as the fallback for
  messages whose type is unknown.

Extraction lives in `src/lancedb_robotics/extract.py` and runs on the single
normalized dict shape 0014 already produced, so **equivalent types across
families collapse to one layout**: `sensor_msgs/Imu` and the foxglove jsonschema
`IMU` both yield the `imu` layout; `sensor_msgs/NavSatFix` and
`foxglove.LocationFix` both yield `gps`.

Routing is exact-schema-name first, then a structural fallback that types an
unknown schema name by the *shape* of its decoded fields (so a vendor
`acme/CustomImu` still lands as `imu`). Unmapped types keep `state_vector` /
`action_vector` NULL and a best-effort modality — never a fabricated number,
never an exception.

A bare schema name is not always unique across producers: XDOF ABC-130k and
Voxel51 RoboLab-EgoX both independently chose the generic protobuf message
name `RobotState` for real, differently-shaped robots. For a name known to
collide this way, `_BY_SCHEMA` maps it to several candidate types instead of
one, each disambiguated by real field shape (here, `position` length) using
the same `match` predicate the structural fallback already uses for unknown
names — see the `joint_state`/`robolab_joint_state`/`robolab_action` rows
below. An exact name with no matching candidate — including a raw/failed row
with no payload to check shape against — yields `modality=None` rather than
guessing.

## Versioned layout mapping

The vector layouts are a versioned contract: `extract.LAYOUT_VERSION` stamps the
ingest `transform_runs` params, and `extract.LAYOUTS` is the single source of
truth for each vector's component order. **Current version: `1`.**

| modality | vector | components (in order) |
| --- | --- | --- |
| `imu` | state (10) | `orientation_{x,y,z,w}`, `angular_velocity_{x,y,z}`, `linear_acceleration_{x,y,z}` |
| `gps` | state (3) | `latitude`, `longitude`, `altitude` |
| `pose` | state (7) | `position_{x,y,z}`, `orientation_{x,y,z,w}` |
| `range` | state (4) | `range`, `min_range`, `max_range`, `field_of_view` |
| `joint_state` | state (6) | `joint1` … `joint6` |
| `gripper` | state (1) | `aperture` |
| `robolab_joint_state` | state (13) | `joint1` … `joint13` |
| `robolab_action` | state (8) | `joint1` … `joint8` |
| `twist` | action (6) | `linear_{x,y,z}`, `angular_{x,y,z}` |

Type → layout/modality coverage:

| schema name(s) | modality | state | action |
| --- | --- | --- | --- |
| `sensor_msgs/Imu` (+ `/msg/`), jsonschema `IMU` | `imu` | imu | — |
| `sensor_msgs/NavSatFix` (+ `/msg/`), `foxglove.LocationFix` | `gps` | gps | — |
| `geometry_msgs/TwistStamped` (+ `/msg/`), `geometry_msgs/Twist` | `twist` | — | twist |
| `nav_msgs/Odometry` (+ `/msg/`) | `odometry` | pose | twist |
| `foxglove.PoseInFrame`, jsonschema `Pose` | `pose` | pose | — |
| `sensor_msgs/Range` (+ `/msg/`) | `range` | range | — |
| `RobotState` (XDOF ABC-130k, protobuf; 6-length `position`) | `joint_state` | joint_state | — |
| `GripperState` (XDOF ABC-130k, protobuf) | `gripper` | gripper | — |
| `RobotState` (Voxel51 RoboLab-EgoX, protobuf; 13-length `position`) | `robolab_joint_state` | robolab_joint_state | — |
| `RobotState` (Voxel51 RoboLab-EgoX, protobuf; 8-length `position`) | `robolab_action` | robolab_action | — |
| `sensor_msgs/Image`/`CompressedImage`, `foxglove.CompressedImage`/`CompressedVideo` | `image` | — | — |
| `sensor_msgs/PointCloud2`, `foxglove.PointCloud` | `pointcloud` | — | — |
| `radar_driver/RadarTracks` | `radar` | — | — |
| `diagnostic_msgs/DiagnosticArray` (+ `/msg/`) | `diagnostic` | — | — |

Image / pointcloud / radar / diagnostic are typed by modality but carry no
state/action vector — their structured payload stays in `payload_json`.

`RobotState`/`GripperState` carry no topic identity of their own — the same
schema serves both an observed (`*-state`) and a commanded (`*-action`) topic
in XDOF's ABC-130k dataset, and `extract()` has no topic argument to tell
them apart, so both always land in `state_vector`, never `action_vector`. A
facade's `CanonicalVectorMapping` (see `lerobot_facade/mapping.py`) is what
actually separates observed from commanded, by declaring which stream names
(topics) are state vs. action — not by which `extract.py` field got
populated. Voxel51's RoboLab-EgoX dataset reuses the exact same bare schema
name `RobotState` for its own, unrelated, differently-shaped robot (13-length
`position` on `/joint-positions`, 8-length on `/actions`) — a real
cross-dataset name collision, not a hypothetical one. `extract()` resolves it
by real field length (see `_BY_SCHEMA["RobotState"]`'s three candidates in
`extract.py`), never by schema digest or topic, keeping `extract()`'s
signature unchanged; the same observed/commanded-by-topic note applies to
`robolab_joint_state`/`robolab_action` as to `joint_state` above.

Changing any layout's component order or length is a breaking change and must
bump `LAYOUT_VERSION`.

## Contract

- Vectors are `list<float32>`; component order is fixed by `LAYOUTS` for the
  stamped `LAYOUT_VERSION`.
- A message with no known type (and no matching shape) leaves both vectors NULL
  and `modality` falls back to the topic-based guess.
- A known type with absent/malformed fields yields a NULL vector (not a partial
  or fabricated one) and never raises.
- The ingest `transform_runs` params record `extracted_by_modality` (counts) and
  `extract_layout_version`.

## Encoding is an implementation detail (backlog 0020)

0020 closed the decode gap so **all seven** MCAP registry message encodings are
decoded (given the optional extra): `flatbuffer` via the channel schema's
embedded `.bfbs` reflection (`adapters/flatbuffer.py`, `flatbuffers` runtime), and
`cbor`/`msgpack` as self-describing binary (`cbor2`/`msgpack`). Because the
flatbuffer decode normalizes to the same snake-case dict shape as protobuf, a
flatbuffer `foxglove.*` message routes through the same `_BY_SCHEMA` name as its
protobuf twin and yields the **identical vector and modality** — a flatbuffer
`foxglove.LocationFix` lands the same `gps` row as the protobuf one. The decoder
emits non-zero schema defaults (foxglove `Vector3`/`Quaternion` default to 1.0)
so a value left at its default still matches the protobuf twin, whose proto3 1.0s
are always explicit.

`cbor`/`msgpack` are schema-free, so they get no typed mapping unless the decoded
shape matches a structural matcher. The only `raw`-by-design cases left are a
missing decoder extra, a schemaless channel for an encoding that needs a schema,
and the IDL schema-encoding tail (`ros2idl`/`omgidl`, unsupported upstream); the
ingest `transform_runs` params now record `decode_raw_by_encoding` so the
remaining gap is visible per run.
