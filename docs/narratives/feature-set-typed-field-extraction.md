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
| `sensor_msgs/Image`/`CompressedImage`, `foxglove.CompressedImage`/`CompressedVideo` | `image` | — | — |
| `sensor_msgs/PointCloud2`, `foxglove.PointCloud` | `pointcloud` | — | — |
| `radar_driver/RadarTracks` | `radar` | — | — |
| `diagnostic_msgs/DiagnosticArray` (+ `/msg/`) | `diagnostic` | — | — |

Image / pointcloud / radar / diagnostic are typed by modality but carry no
state/action vector — their structured payload stays in `payload_json`.

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
