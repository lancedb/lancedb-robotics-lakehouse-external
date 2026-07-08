"""ROS bag adapter tests (backlog 0025)."""

import importlib.util
import json

import pytest
from typer.testing import CliRunner

from lancedb_robotics.adapters import AdapterError, get_adapter
from lancedb_robotics.cli import app
from lancedb_robotics.ingest import ingest_mcap, ingest_rosbag
from lancedb_robotics.lake import Lake

runner = CliRunner()


@pytest.fixture
def ros1_bag(fixtures_dir):
    return fixtures_dir / "ros1_string.bag"


@pytest.fixture
def ros1_mcap(fixtures_dir):
    return fixtures_dir / "ros1_string.mcap"


@pytest.fixture
def ros2_db3(fixtures_dir):
    return fixtures_dir / "ros2_string_bag" / "ros2_string_bag.db3"


@pytest.fixture
def ros2_dir(fixtures_dir):
    return fixtures_dir / "ros2_string_bag"


@pytest.fixture
def ros2_mcap(fixtures_dir):
    return fixtures_dir / "ros2_string.mcap"


@pytest.fixture
def lake(tmp_path):
    return Lake.init(tmp_path / "robot.lance")


def _observations(lake):
    rows = lake.table("observations").to_arrow().to_pylist()
    return sorted(rows, key=lambda row: (row["topic"], row["timestamp_ns"], row["raw_sequence"]))


def _canonical_observations(lake):
    return [
        {
            "topic": row["topic"],
            "timestamp_ns": row["timestamp_ns"],
            "raw_sequence": row["raw_sequence"],
            "payload_json": row["payload_json"],
            "message_encoding": row["message_encoding"],
            "schema_encoding": row["schema_encoding"],
            "decode_status": row["decode_status"],
            "modality": row["modality"],
        }
        for row in _observations(lake)
    ]


def test_inspect_ros1_bag_reports_topic_and_decode_probe(ros1_bag):
    pytest.importorskip("rosbags")
    pytest.importorskip("mcap_ros1")
    report = get_adapter("rosbag").inspect(ros1_bag)
    assert report["adapter"] == "rosbag"
    assert report["profile"] == "ros1"
    assert report["storage_identifier"] == "rosbag1"
    assert report["message_count"] == 2
    assert report["topics"] == [
        {
            "topic": "/chatter",
            "message_encoding": "ros1",
            "schema_name": "std_msgs/String",
            "schema_encoding": "ros1msg",
            "message_count": 2,
            "start_time_ns": 1_700_000_000_500_000_000,
            "end_time_ns": 1_700_000_000_600_000_000,
            "can_decode": True,
        }
    ]


def test_inspect_ros2_db3_and_directory_report_same_topics(ros2_db3, ros2_dir):
    pytest.importorskip("rosbags")
    by_file = get_adapter("rosbag").inspect(ros2_db3)
    by_dir = get_adapter("rosbag").inspect(ros2_dir)
    assert by_file["profile"] == "ros2"
    assert by_file["storage_identifier"] == "sqlite3"
    assert by_file["topics"] == by_dir["topics"]
    assert by_file["message_count"] == by_dir["message_count"] == 2


def test_ingest_ros1_bag_decodes_payloads(lake, ros1_bag):
    pytest.importorskip("rosbags")
    pytest.importorskip("mcap_ros1")
    report = ingest_rosbag(lake, ros1_bag)
    assert report.decode_by_status == {"decoded": 2}
    assert report.decode_by_encoding == {"ros1": 2}
    assert report.observations_by_topic == {"/chatter": 2}
    rows = _observations(lake)
    assert [json.loads(row["payload_json"])["data"] for row in rows] == [
        "legacy hello",
        "legacy world",
    ]
    assert [row["raw_sequence"] for row in rows] == [0, 1]
    assert {row["decode_status"] for row in rows} == {"decoded"}
    assert {row["message_encoding"] for row in rows} == {"ros1"}
    assert {row["schema_encoding"] for row in rows} == {"ros1msg"}


def test_ingest_ros2_db3_decodes_payloads(lake, ros2_db3):
    pytest.importorskip("rosbags")
    pytest.importorskip("mcap_ros2")
    report = ingest_rosbag(lake, ros2_db3)
    assert report.decode_by_status == {"decoded": 2}
    assert report.decode_by_encoding == {"cdr": 2}
    rows = _observations(lake)
    assert [json.loads(row["payload_json"])["data"] for row in rows] == [
        "legacy hello",
        "legacy world",
    ]
    assert {row["message_encoding"] for row in rows} == {"cdr"}
    assert {row["schema_encoding"] for row in rows} == {"ros2msg"}


def test_ros1_bag_and_mcap_twin_land_equivalent_observation_rows(tmp_path, ros1_bag, ros1_mcap):
    pytest.importorskip("rosbags")
    pytest.importorskip("mcap_ros1")
    bag_lake = Lake.init(tmp_path / "bag.lance")
    mcap_lake = Lake.init(tmp_path / "mcap.lance")
    ingest_rosbag(bag_lake, ros1_bag)
    ingest_mcap(mcap_lake, ros1_mcap)
    assert _canonical_observations(bag_lake) == _canonical_observations(mcap_lake)


def test_ros2_db3_and_mcap_twin_land_equivalent_observation_rows(tmp_path, ros2_db3, ros2_mcap):
    pytest.importorskip("rosbags")
    pytest.importorskip("mcap_ros2")
    bag_lake = Lake.init(tmp_path / "bag.lance")
    mcap_lake = Lake.init(tmp_path / "mcap.lance")
    ingest_rosbag(bag_lake, ros2_db3)
    ingest_mcap(mcap_lake, ros2_mcap)
    assert _canonical_observations(bag_lake) == _canonical_observations(mcap_lake)


def test_missing_rosbags_extra_reports_unavailable(monkeypatch, ros1_bag):
    import lancedb_robotics.adapters.rosbag_adapter as rosbag_adapter

    original = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "rosbags.highlevel":
            return None
        return original(name)

    monkeypatch.setattr(rosbag_adapter.importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(AdapterError, match="optional 'rosbag' extra"):
        get_adapter("rosbag").inspect(ros1_bag)


def test_cli_inspect_rosbag_json(ros1_bag):
    pytest.importorskip("rosbags")
    result = runner.invoke(app, ["inspect", "rosbag", str(ros1_bag)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["adapter"] == "rosbag"
    assert payload["topics"][0]["topic"] == "/chatter"


def test_cli_ingest_rosbag_reports_rows(tmp_path, ros2_db3):
    pytest.importorskip("rosbags")
    pytest.importorskip("mcap_ros2")
    lake_path = tmp_path / "robot.lance"
    init = runner.invoke(app, ["lake", "init", "--lake", str(lake_path)])
    assert init.exit_code == 0
    result = runner.invoke(app, ["ingest", "rosbag", str(ros2_db3), "--lake", str(lake_path)])
    assert result.exit_code == 0
    assert "runs +1" in result.output
    assert "observations +2" in result.output
    assert "/chatter\t2" in result.output
