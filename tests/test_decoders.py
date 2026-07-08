"""Decoder dispatch tests (backlog 0014, completed in 0020).

These pin the contract that dispatch is exhaustive over the registry encodings
and that every non-decodable outcome is data, not a crash: a missing decoder
yields ``raw``; a decoder that raises yields ``failed``; in both cases the
payload is absent but no exception escapes. The json path needs no extra and runs
everywhere; ros2/cdr-, flatbuffer-, cbor-, and msgpack-backed behaviour is
exercised when the matching extra is importable (the dev/CI env installs all of
them) and the missing-extra degrade is simulated with monkeypatch so it is
covered in every environment.
"""

import base64
import io
import json

import pytest

from lancedb_robotics.adapters.decoders import (
    DEFAULT_BLOB_THRESHOLD,
    PayloadDecoder,
    can_decode,
)


class FakeSchema:
    """Minimal stand-in for an MCAP schema (``.encoding``/``.id``/``.data``)."""

    def __init__(self, encoding: str, *, id: int = 1, data: bytes = b"") -> None:
        self.encoding = encoding
        self.id = id
        self.data = data


# --- dispatch resolution -------------------------------------------------


def test_json_decoder_resolves_without_any_extra():
    decoder = PayloadDecoder()
    assert decoder.decoder_for("json", FakeSchema("jsonschema")) is not None
    # json is self-describing: it resolves even on a schemaless channel.
    assert decoder.decoder_for("json", None) is not None


@pytest.mark.parametrize("encoding", ["cbor", "msgpack"])
def test_self_describing_binary_resolves_without_a_schema(encoding):
    # cbor/msgpack are self-describing: a decoder resolves even on a schemaless
    # channel, provided the optional lib is installed (it is in dev/CI).
    pytest.importorskip({"cbor": "cbor2", "msgpack": "msgpack"}[encoding])
    assert can_decode(encoding) is True
    assert PayloadDecoder().decoder_for(encoding, None) is not None


def test_unknown_encoding_resolves_to_no_decoder():
    assert can_decode("totally-unknown") is False
    assert PayloadDecoder().decoder_for("totally-unknown", FakeSchema("x")) is None


# --- decode outcomes that need no extra ----------------------------------


def test_json_payload_decodes_to_canonical_json():
    result = PayloadDecoder().decode("json", FakeSchema("jsonschema"), b'{"gyro_z": 0.5}')
    assert result.status == "decoded"
    assert json.loads(result.payload_json) == {"gyro_z": 0.5}
    assert result.payload_blob is None  # scalar message: no blob
    assert result.error is None


def test_bad_json_payload_is_failed_not_crash():
    result = PayloadDecoder().decode("json", FakeSchema("jsonschema"), b"this is not json")
    assert result.status == "failed"
    assert result.payload_json is None
    assert result.error  # carries the exception text


def test_decoder_resolution_error_is_failed_not_a_crash():
    # Upstream factories parse the schema descriptor when resolving a decoder, so
    # a corrupt schema raises during decoder_for(), not during the decode call.
    # That must surface as `failed`, never propagate and crash ingest. (Regression
    # for the real DecodeError seen on sample.mcap's stand-in protobuf descriptor.)
    decoder = PayloadDecoder()

    def boom(message_encoding, schema):
        raise ValueError("corrupt schema descriptor")

    decoder.decoder_for = boom  # type: ignore[method-assign]
    result = decoder.decode("protobuf", FakeSchema("protobuf"), b"\x08\x01")
    assert result.status == "failed"
    assert result.payload_json is None
    assert "decoder-init" in result.error
    assert "corrupt schema descriptor" in result.error


# --- self-describing binary: cbor / msgpack (backlog 0020) ---------------


def test_cbor_payload_decodes_schema_free():
    pytest.importorskip("cbor2")
    import cbor2

    payload = cbor2.dumps({"format": "jpeg", "seq": 3})
    result = PayloadDecoder().decode("cbor", FakeSchema("cbor"), payload)
    assert result.status == "decoded"
    assert json.loads(result.payload_json) == {"format": "jpeg", "seq": 3}
    assert result.error is None


def test_msgpack_payload_decodes_schema_free():
    pytest.importorskip("msgpack")
    import msgpack

    payload = msgpack.packb({"format": "h264", "seq": 7})
    result = PayloadDecoder().decode("msgpack", FakeSchema("msgpack"), payload)
    assert result.status == "decoded"
    assert json.loads(result.payload_json) == {"format": "h264", "seq": 7}


def test_corrupt_cbor_payload_is_failed_not_crash():
    pytest.importorskip("cbor2")
    # A truncated cbor item (array header claims 2 elements, only 1 present): the
    # decoder ran and raised, so this is `failed` (not `raw`), and the exception
    # is recorded as data rather than escaping.
    result = PayloadDecoder().decode("cbor", FakeSchema("cbor"), b"\x82\x01")
    assert result.status == "failed"
    assert result.payload_json is None
    assert result.error


@pytest.mark.parametrize(
    "encoding,requirement", [("cbor", "cbor2"), ("msgpack", "msgpack")]
)
def test_self_describing_binary_degrades_to_raw_when_lib_absent(monkeypatch, encoding, requirement):
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec
    monkeypatch.setattr(
        importlib_util,
        "find_spec",
        lambda name, *a, **k: None if name == requirement else real_find_spec(name, *a, **k),
    )
    assert can_decode(encoding) is False
    result = PayloadDecoder().decode(encoding, FakeSchema(encoding), b"\x00\x01\x02")
    assert result.status == "raw"
    assert "not installed" in result.error


# --- flatbuffer via embedded .bfbs reflection (backlog 0020) -------------


def test_flatbuffer_message_decodes_from_embedded_reflection():
    pytest.importorskip("flatbuffers")
    get_schema = pytest.importorskip("foxglove_schemas_flatbuffer").get_schema
    LocationFix = pytest.importorskip("foxglove_schemas_flatbuffer.LocationFix")
    import flatbuffers

    builder = flatbuffers.Builder(256)
    LocationFix.Start(builder)
    LocationFix.AddLatitude(builder, 37.4)
    LocationFix.AddLongitude(builder, -122.1)
    LocationFix.AddAltitude(builder, 30.5)
    builder.Finish(LocationFix.End(builder))
    schema = FakeSchema("flatbuffer", id=11, data=get_schema("LocationFix"))
    result = PayloadDecoder().decode("flatbuffer", schema, bytes(builder.Output()))
    assert result.status == "decoded"
    decoded = json.loads(result.payload_json)
    assert decoded["latitude"] == pytest.approx(37.4)
    assert decoded["longitude"] == pytest.approx(-122.1)
    assert decoded["altitude"] == pytest.approx(30.5)


def test_flatbuffer_without_schema_routes_to_raw():
    pytest.importorskip("flatbuffers")
    # flatbuffer is not self-describing: a schemaless channel cannot be decoded.
    result = PayloadDecoder().decode("flatbuffer", None, b"\x00\x01\x02\x03")
    assert result.status == "raw"
    assert "no schema" in result.error


def test_flatbuffer_degrades_to_raw_when_lib_absent(monkeypatch):
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec
    monkeypatch.setattr(
        importlib_util,
        "find_spec",
        lambda name, *a, **k: None if name == "flatbuffers" else real_find_spec(name, *a, **k),
    )
    assert can_decode("flatbuffer") is False
    result = PayloadDecoder().decode("flatbuffer", FakeSchema("flatbuffer", data=b"x"), b"\x00")
    assert result.status == "raw"
    assert "not installed" in result.error


def test_corrupt_flatbuffer_schema_is_failed_not_crash():
    pytest.importorskip("flatbuffers")
    # The .bfbs is unparseable: resolving the decoder raises, which surfaces as
    # `failed` (decoder-init), never an ingest crash.
    schema = FakeSchema("flatbuffer", id=99, data=b"not a real bfbs")
    result = PayloadDecoder().decode("flatbuffer", schema, b"\x00\x01\x02\x03")
    assert result.status == "failed"
    assert result.payload_json is None
    assert result.error


# --- ros2idl schema encoding: documented raw (backlog 0020) --------------


def test_ros2idl_schema_encoding_routes_to_raw_with_reason():
    pytest.importorskip("mcap_ros2")
    # mcap-ros2-support decodes ros2msg schemas only; a ros2idl (IDL) schema on a
    # cdr channel is declined by the factory and recorded as raw with a reason.
    schema = FakeSchema("ros2idl", id=3, data=b"module m { struct T { double v; }; };")
    result = PayloadDecoder().decode("cdr", schema, b"\x00\x01\x02\x03")
    assert result.status == "raw"
    assert result.payload_json is None
    assert "ros2idl" in result.error


def test_missing_decoder_extra_is_raw_when_absent(monkeypatch):
    # When a decoder extra is not installed, a message of that encoding is raw,
    # not a crash. Backlog 0017 makes CI install every decoder, which would
    # permanently skip this path -- so simulate the extra being absent (find_spec
    # returns None for it) to cover it in every environment.
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec
    monkeypatch.setattr(
        importlib_util,
        "find_spec",
        lambda name, *a, **k: None if name == "mcap_protobuf" else real_find_spec(name, *a, **k),
    )
    assert can_decode("protobuf") is False  # the extra now looks absent
    result = PayloadDecoder().decode("protobuf", FakeSchema("protobuf"), b"\x08\x01")
    assert result.status == "raw"
    assert "not installed" in result.error


# --- ros2/cdr-backed behaviour (needs the ros2 extra) --------------------


def _write_ros2(messages):
    """Return MCAP bytes for ``messages`` = list of (msgdef_name, msgdef, dict)."""
    from mcap_ros2.writer import Writer as Ros2Writer

    buf = io.BytesIO()
    writer = Ros2Writer(buf)
    for name, msgdef, payload in messages:
        schema = writer.register_msgdef(name, msgdef)
        writer.write_message(topic="/t", schema=schema, message=payload, log_time=1, publish_time=1)
    writer.finish()
    return buf.getvalue()


_IMU_DEF = """
geometry_msgs/Vector3 angular_velocity
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
"""

_IMAGE_DEF = """
uint32 height
uint32 width
uint8[] data
"""


def _decode_first_ros2(mcap_bytes, decoder):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    reader = make_reader(io.BytesIO(mcap_bytes), decoder_factories=[DecoderFactory()])
    for schema, channel, message in reader.iter_messages():
        return decoder.decode(channel.message_encoding, schema, message.data)
    raise AssertionError("no message in fixture")


def test_cdr_message_decodes_to_payload_json():
    pytest.importorskip("mcap_ros2")
    mcap_bytes = _write_ros2(
        [("sensor_msgs/msg/Imu", _IMU_DEF, {"angular_velocity": {"x": 0.1, "y": 0.2, "z": 0.3}})]
    )
    result = _decode_first_ros2(mcap_bytes, PayloadDecoder())
    assert result.status == "decoded"
    decoded = json.loads(result.payload_json)
    assert decoded["angular_velocity"] == {"x": 0.1, "y": 0.2, "z": 0.3}
    assert result.payload_blob is None  # scalar Imu: no hoisted blob


def test_large_binary_message_hoists_blob_and_elides_json():
    pytest.importorskip("mcap_ros2")
    big = bytes(DEFAULT_BLOB_THRESHOLD + 16)  # over the hoist threshold
    mcap_bytes = _write_ros2(
        [("sensor_msgs/msg/Image", _IMAGE_DEF, {"height": 4, "width": 4, "data": big})]
    )
    result = _decode_first_ros2(mcap_bytes, PayloadDecoder())
    assert result.status == "decoded"
    assert result.payload_blob is not None  # large data hoisted out
    decoded = json.loads(result.payload_json)
    # The big field is elided from JSON (recorded as a length marker), not inlined.
    assert decoded["data"] == {"__elided_bytes__": len(big)}
    assert decoded["height"] == 4


def test_small_binary_is_inlined_not_hoisted():
    pytest.importorskip("mcap_ros2")
    small = b"\x01\x02\x03\x04"
    mcap_bytes = _write_ros2(
        [("sensor_msgs/msg/Image", _IMAGE_DEF, {"height": 1, "width": 1, "data": small})]
    )
    result = _decode_first_ros2(mcap_bytes, PayloadDecoder())
    assert result.status == "decoded"
    assert result.payload_blob is None  # under threshold: stays inline
    decoded = json.loads(result.payload_json)
    assert decoded["data"] == {"__b64__": base64.b64encode(small).decode("ascii")}


def test_schemaless_ros_channel_routes_to_raw():
    pytest.importorskip("mcap_ros2")
    # cdr needs a schema to decode; a schemaless channel must route to raw, not crash.
    result = PayloadDecoder().decode("cdr", None, b"\x00\x01\x02\x03")
    assert result.status == "raw"
    assert result.payload_json is None
    assert "no schema" in result.error
