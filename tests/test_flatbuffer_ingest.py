"""Flatbuffer ingest + protobuf-parity tests (backlog 0020).

The headline of 0020: a flatbuffer-schema Foxglove clip ingests to the same
decoded fields and typed vectors as its protobuf twin -- encoding is an
implementation detail. ``flatbuffer_foxglove.mcap`` carries a LocationFix (gps),
a PoseInFrame (pose), and a CompressedImage (image), each ``message_encoding=
'flatbuffer'`` with the type's ``.bfbs`` reflection as its schema. Needs the
``flatbuffer`` extra (``flatbuffers``); skipped when absent.
"""

import json

import pytest

from lancedb_robotics.extract import extract
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake


@pytest.fixture
def flatbuffer_mcap(fixtures_dir):
    pytest.importorskip("flatbuffers")
    return fixtures_dir / "flatbuffer_foxglove.mcap"


@pytest.fixture
def rows(tmp_path, flatbuffer_mcap):
    lake = Lake.init(tmp_path / "robot.lance")
    report = ingest_mcap(lake, flatbuffer_mcap)
    assert report.decode_by_status == {"decoded": 3}
    assert report.decode_by_encoding == {"flatbuffer": 3}
    return {r["topic"]: r for r in lake.table("observations").to_arrow().to_pylist()}


def test_flatbuffer_locationfix_decodes_and_types_as_gps(rows):
    gps = rows["/gps/fix"]
    assert gps["message_encoding"] == "flatbuffer"
    assert gps["decode_status"] == "decoded"
    assert gps["modality"] == "gps"
    payload = json.loads(gps["payload_json"])
    assert payload["latitude"] == pytest.approx(37.4)
    assert gps["state_vector"] == pytest.approx([37.4, -122.1, 30.5])


def test_flatbuffer_pose_recovers_default_quaternion_w(rows):
    pose = rows["/pose"]
    assert pose["modality"] == "pose"
    # position (2,3,4) + identity quaternion -- w is the schema default (1.0) and
    # is omitted from the buffer, so this asserts the decoder recovers it.
    assert pose["state_vector"] == pytest.approx([2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0])


def test_flatbuffer_compressed_image_types_as_image(rows):
    image = rows["/camera/front"]
    assert image["modality"] == "image"
    payload = json.loads(image["payload_json"])
    assert payload["format"] == "jpeg"


# --- parity vs the protobuf twin -------------------------------------------


def test_flatbuffer_locationfix_extracts_same_as_protobuf_twin(rows):
    # The protobuf MessageToDict of foxglove.LocationFix (proto3: zeros omitted)
    # vs. the flatbuffer decode must yield the identical typed gps vector.
    flatbuffer_payload = json.loads(rows["/gps/fix"]["payload_json"])
    protobuf_twin = {"latitude": 37.4, "longitude": -122.1, "altitude": 30.5}
    fb = extract("foxglove.LocationFix", flatbuffer_payload)
    pb = extract("foxglove.LocationFix", protobuf_twin)
    assert fb.modality == pb.modality == "gps"
    assert fb.state_vector == pytest.approx(pb.state_vector)


def test_flatbuffer_pose_extracts_same_as_protobuf_twin(rows):
    flatbuffer_payload = json.loads(rows["/pose"]["payload_json"])
    # proto3 twin: position 1.0s/zeros omitted... here all non-zero, so explicit.
    protobuf_twin = {
        "frame_id": "map",
        "pose": {
            "position": {"x": 2.0, "y": 3.0, "z": 4.0},
            "orientation": {"w": 1.0},
        },
    }
    fb = extract("foxglove.PoseInFrame", flatbuffer_payload)
    pb = extract("foxglove.PoseInFrame", protobuf_twin)
    assert fb.modality == pb.modality == "pose"
    assert fb.state_vector == pytest.approx(pb.state_vector)
