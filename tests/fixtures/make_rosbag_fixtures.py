"""Regenerate ROS bag adapter fixtures (backlog 0025).

Run: uv run --extra dev python tests/fixtures/make_rosbag_fixtures.py

The fixtures use the same two `std_msgs/String` messages across ROS1 bag,
ROS2 sqlite bag, and MCAP twins. Tests then assert that the container format is
an implementation detail: the canonical observation rows match after ingest.
"""

from pathlib import Path
from shutil import rmtree

from mcap.writer import Writer as McapWriter
from rosbags.rosbag1 import Writer as Rosbag1Writer
from rosbags.rosbag2 import Writer as Rosbag2Writer
from rosbags.typesys import Stores, get_typestore

ROOT = Path(__file__).parent
ROS1_BAG = ROOT / "ros1_string.bag"
ROS1_MCAP = ROOT / "ros1_string.mcap"
ROS2_DIR = ROOT / "ros2_string_bag"
ROS2_DB3 = ROS2_DIR / "ros2_string_bag.db3"
ROS2_MCAP = ROOT / "ros2_string.mcap"

TOPIC = "/chatter"
MSG_DEF = "string data\n"
BASE_NS = 1_700_000_000_500_000_000
MESSAGES = (
    (BASE_NS, "legacy hello"),
    (BASE_NS + 100_000_000, "legacy world"),
)


def _ros1_payloads() -> list[tuple[int, bytes]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    String = typestore.types["std_msgs/msg/String"]
    return [
        (timestamp, bytes(typestore.serialize_ros1(String(text), String.__msgtype__)))
        for timestamp, text in MESSAGES
    ]


def _ros2_payloads() -> list[tuple[int, bytes]]:
    typestore = get_typestore(Stores.ROS2_FOXY)
    String = typestore.types["std_msgs/msg/String"]
    return [
        (timestamp, bytes(typestore.serialize_cdr(String(text), String.__msgtype__)))
        for timestamp, text in MESSAGES
    ]


def build_ros1_bag() -> None:
    ROS1_BAG.unlink(missing_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    String = typestore.types["std_msgs/msg/String"]
    with Rosbag1Writer(ROS1_BAG) as writer:
        connection = writer.add_connection(TOPIC, String.__msgtype__, typestore=typestore)
        for timestamp, payload in _ros1_payloads():
            writer.write(connection, timestamp, payload)


def build_ros2_bag() -> None:
    if ROS2_DIR.exists():
        rmtree(ROS2_DIR)
    typestore = get_typestore(Stores.ROS2_FOXY)
    String = typestore.types["std_msgs/msg/String"]
    with Rosbag2Writer(ROS2_DIR, version=9) as writer:
        connection = writer.add_connection(TOPIC, String.__msgtype__, typestore=typestore)
        for timestamp, payload in _ros2_payloads():
            writer.write(connection, timestamp, payload)


def build_mcap(
    path: Path,
    *,
    profile: str,
    schema_name: str,
    schema_encoding: str,
    message_encoding: str,
    payloads: list[tuple[int, bytes]],
) -> None:
    path.unlink(missing_ok=True)
    with path.open("wb") as stream:
        writer = McapWriter(stream)
        writer.start(profile=profile, library="lancedb-robotics-test")
        schema_id = writer.register_schema(
            name=schema_name,
            encoding=schema_encoding,
            data=MSG_DEF.encode("utf-8"),
        )
        channel_id = writer.register_channel(
            topic=TOPIC,
            message_encoding=message_encoding,
            schema_id=schema_id,
        )
        for sequence, (timestamp, payload) in enumerate(payloads):
            writer.add_message(
                channel_id=channel_id,
                log_time=timestamp,
                publish_time=timestamp,
                sequence=sequence,
                data=payload,
            )
        writer.finish()


def main() -> None:
    ros1_payloads = _ros1_payloads()
    ros2_payloads = _ros2_payloads()
    build_ros1_bag()
    build_ros2_bag()
    build_mcap(
        ROS1_MCAP,
        profile="ros1",
        schema_name="std_msgs/String",
        schema_encoding="ros1msg",
        message_encoding="ros1",
        payloads=ros1_payloads,
    )
    build_mcap(
        ROS2_MCAP,
        profile="ros2",
        schema_name="std_msgs/msg/String",
        schema_encoding="ros2msg",
        message_encoding="cdr",
        payloads=ros2_payloads,
    )
    for path in (ROS1_BAG, ROS1_MCAP, ROS2_DB3, ROS2_DIR / "metadata.yaml", ROS2_MCAP):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
