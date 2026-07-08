"""Regenerate tests/fixtures/flatbuffer_foxglove.mcap (backlog 0020).

Run: uv run python tests/fixtures/make_flatbuffer_foxglove_mcap.py

No real corpus uses flatbuffer, so this fixture is *generated* from the official
Foxglove flatbuffer schemas (``foxglove-schemas-flatbuffer``): three messages --
``foxglove.LocationFix`` (gps), ``foxglove.PoseInFrame`` (pose), and
``foxglove.CompressedImage`` (image) -- each written with ``message_encoding=
'flatbuffer'`` and a ``flatbuffer`` schema whose ``data`` is the type's binary
``.bfbs`` reflection. These are the flatbuffer twins of the protobuf
``foxglove.*`` types the typed extractor (0015) already maps, so ingest must land
the same vectors/modality regardless of which wire format was used.

The field values are chosen so a flatbuffer message at its non-zero schema
default (foxglove ``Vector3``/``Quaternion`` default to 1.0) is exercised: the
identity quaternion's ``w`` is the schema default and must still decode to 1.0.

Requires the ``flatbuffer`` extra (``flatbuffers``) and
``foxglove-schemas-flatbuffer``. All timestamps are fixed so the fixture is
byte-stable across regenerations (asserted by tests/test_fixture_tooling.py).
"""

import io
from pathlib import Path

import flatbuffers
from foxglove_schemas_flatbuffer import (
    CompressedImage,
    LocationFix,
    Pose,
    PoseInFrame,
    Quaternion,
    Vector3,
    get_schema,
)
from mcap.writer import CompressionType, Writer

OUT = Path(__file__).parent / "flatbuffer_foxglove.mcap"

BASE_NS = 1_700_000_000_000_000_000  # fixed epoch so the fixture is deterministic


def _location_fix() -> bytes:
    b = flatbuffers.Builder(256)
    frame = b.CreateString("gps")
    LocationFix.Start(b)
    LocationFix.AddFrameId(b, frame)
    LocationFix.AddLatitude(b, 37.4)
    LocationFix.AddLongitude(b, -122.1)
    LocationFix.AddAltitude(b, 30.5)
    b.Finish(LocationFix.End(b))
    return bytes(b.Output())


def _pose_in_frame() -> bytes:
    b = flatbuffers.Builder(256)
    frame = b.CreateString("map")
    Vector3.Start(b)
    Vector3.AddX(b, 2.0)
    Vector3.AddY(b, 3.0)
    Vector3.AddZ(b, 4.0)
    position = Vector3.End(b)
    # Identity quaternion: x/y/z = 0.0, w = 1.0 (w is the schema default, so it is
    # omitted from the buffer and must be recovered as 1.0 by the decoder).
    Quaternion.Start(b)
    Quaternion.AddX(b, 0.0)
    Quaternion.AddY(b, 0.0)
    Quaternion.AddZ(b, 0.0)
    Quaternion.AddW(b, 1.0)
    orientation = Quaternion.End(b)
    Pose.Start(b)
    Pose.AddPosition(b, position)
    Pose.AddOrientation(b, orientation)
    pose = Pose.End(b)
    PoseInFrame.Start(b)
    PoseInFrame.AddFrameId(b, frame)
    PoseInFrame.AddPose(b, pose)
    b.Finish(PoseInFrame.End(b))
    return bytes(b.Output())


def _compressed_image() -> bytes:
    b = flatbuffers.Builder(256)
    frame = b.CreateString("cam_front")
    fmt = b.CreateString("jpeg")
    data = b.CreateByteVector(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    CompressedImage.Start(b)
    CompressedImage.AddFrameId(b, frame)
    CompressedImage.AddData(b, data)
    CompressedImage.AddFormat(b, fmt)
    b.Finish(CompressedImage.End(b))
    return bytes(b.Output())


# (topic, fully-qualified flatbuffer type name, message bytes) in log order.
_MESSAGES = (
    ("/gps/fix", "foxglove.LocationFix", _location_fix),
    ("/pose", "foxglove.PoseInFrame", _pose_in_frame),
    ("/camera/front", "foxglove.CompressedImage", _compressed_image),
)


def build_bytes() -> bytes:
    """Return the fixture bytes (deterministic; used by main and the tooling test)."""
    buf = io.BytesIO()
    writer = Writer(buf, compression=CompressionType.NONE)
    writer.start(profile="", library="lancedb-robotics-fixture")
    for index, (topic, type_name, builder) in enumerate(_MESSAGES):
        schema_id = writer.register_schema(
            name=type_name, encoding="flatbuffer", data=get_schema(type_name.split(".")[-1])
        )
        channel_id = writer.register_channel(
            topic=topic, message_encoding="flatbuffer", schema_id=schema_id
        )
        writer.add_message(
            channel_id=channel_id,
            log_time=BASE_NS + index * 100_000_000,
            publish_time=BASE_NS + index * 100_000_000,
            data=builder(),
        )
    writer.finish()
    return buf.getvalue()


def main() -> None:
    OUT.write_bytes(build_bytes())
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
