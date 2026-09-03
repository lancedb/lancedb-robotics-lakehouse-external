"""Live LeRobot facade v1 tests (tabular-only, aligned-tick-backed).

Regression target: MCAP/ROS ingest writes per-message state/action vectors
whose width depends on the source topic's message type (see
``extract.py``'s per-topic layouts). Pooling heterogeneous topics into one
LeRobot episode previously produced frames with mismatched vector widths
(``DatasetExportError`` in ``dataset_export.py`` on real data mixing e.g.
3/4/7/10-float topics). This facade avoids that entirely by reading from an
*aligned* view (one synchronized row per tick, built by ``align.py``) and
composing a declared, fixed-width vector from named streams -- these tests
build a lake with two different-width topics feeding one ``observation.state``
and assert the composed width is exactly the sum of the declared components,
for every frame, across multiple episodes.
"""

from datetime import UTC, datetime

import pyarrow as pa
import pytest
from conftest import require_torch_loader

from lancedb_robotics.lake import Lake
from lancedb_robotics.lerobot_facade import (
    CanonicalVectorMapping,
    LeRobotFacadeError,
    LiveLeRobotFacade,
    compose_vector,
    to_torch_dataset,
    validate_mapping,
)
from lancedb_robotics.schemas import EPISODES_SCHEMA, OBSERVATIONS_SCHEMA, RUNS_SCHEMA

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_RUN_ID = "run-facade"


def _observation(
    observation_id: str,
    topic: str,
    timestamp_ns: int,
    sequence: int,
    *,
    state_vector: list[float] | None = None,
    action_vector: list[float] | None = None,
) -> dict:
    return {
        "observation_id": observation_id,
        "run_id": _RUN_ID,
        "episode_id": None,
        "episode_index": None,
        "frame_index": None,
        "timestamp_ns": timestamp_ns,
        "sensor_id": topic.strip("/").replace("/", "_"),
        "topic": topic,
        "modality": "action" if topic == "/action" else "state",
        "robot_id": "robot-facade",
        "site_id": "lab-facade",
        "task_id": "",
        "software_version": "sw-1",
        "outcome": "",
        "raw_uri": "memory://facade",
        "raw_channel": topic,
        "raw_log_time_ns": timestamp_ns,
        "raw_sequence": sequence,
        "payload_json": None,
        "payload_blob": None,
        "message_encoding": "json",
        "schema_encoding": "json",
        "decode_status": "decoded",
        "decode_error": "",
        "state_vector": state_vector,
        "action_vector": action_vector,
        "caption": "",
        "quality_flags": [],
        "transform_id": "tfm-ingest",
        "created_at": _NOW,
    }


def _episode_row(episode_id: str, index: int, from_ns: int, to_ns: int, task_id: str) -> dict:
    return {
        "episode_id": episode_id,
        "run_id": _RUN_ID,
        "episode_index": index,
        "from_timestamp_ns": from_ns,
        "to_timestamp_ns": to_ns,
        "boundary_source": "test",
        "outcome": "",
        "frame_count": None,
        "camera_blobs": [],
        "task_id": task_id,
        "embedding": None,
        "provenance": "",
        "transform_id": "tfm-episode",
        "created_at": _NOW,
    }


def _two_episode_lake(path):
    """Two physical episodes, two heterogeneous-width state topics + one action topic.

    ``/gps`` yields a 3-float state component, ``/imu`` a 4-float one (deliberately
    different widths, like the real gps=3/imu=10/pose=7/range=4 layouts in
    ``extract.py``) -- ``observation.state`` is declared as their concatenation,
    so every resolved frame must be exactly 7 floats wide.
    """
    lake = Lake.init(path)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": _RUN_ID,
                    "run_kind": "teleop",
                    "source": "synthetic",
                    "source_id": "src-facade",
                    "raw_uri": "memory://facade",
                    "robot_id": "robot-facade",
                    "site_id": "lab-facade",
                    "task_id": "fallback task",
                    "start_time_ns": 0,
                    "end_time_ns": 400_000_000,
                    "duration_ns": 400_000_000,
                    "software_version": "sw-1",
                    "hardware_version": "hw-1",
                    "calibration_version": "cal-1",
                    "model_version": "",
                    "metadata": [],
                    "quality_flags": [],
                    "transform_id": "tfm-source",
                    "created_at": _NOW,
                }
            ],
            schema=RUNS_SCHEMA,
        )
    )
    lake.table("episodes").add(
        pa.Table.from_pylist(
            [
                # Boundaries deliberately sit *off* the 50ms (20 Hz) tick grid so
                # each episode captures exactly its own 3 ticks -- an episode end
                # landing exactly on a tick timestamp would also capture that
                # tick (inclusive bounds, matching dataset_export.py's convention).
                _episode_row("ep-0", 0, 0, 149_000_000, "pick"),
                _episode_row("ep-1", 1, 199_000_000, 349_000_000, "place"),
            ],
            schema=EPISODES_SCHEMA,
        )
    )

    rows = []
    seq = 0
    for episode_start in (0, 200_000_000):
        for offset in (0, 50_000_000, 100_000_000):
            ts = episode_start + offset
            rows.append(_observation(f"gps-{ts}", "/gps", ts, seq, state_vector=[1.0, 2.0, 3.0]))
            seq += 1
            rows.append(
                _observation(f"imu-{ts}", "/imu", ts, seq, state_vector=[10.0, 20.0, 30.0, 40.0])
            )
            seq += 1
            rows.append(_observation(f"action-{ts}", "/action", ts, seq, action_vector=[9.0]))
            seq += 1
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))

    lake.align.create_view(
        "facade_view",
        run_id=_RUN_ID,
        rate_hz=20.0,
        streams=["/gps", "/imu", "/action"],
        tolerance_ms=100.0,
        interpolation={"/gps": "nearest", "/imu": "nearest", "/action": "nearest"},
    )
    return lake


# ---------------------------------------------------------------------------
# mapping.py -- pure unit tests, no lake needed
# ---------------------------------------------------------------------------


def test_compose_vector_concatenates_declared_streams_in_order():
    streams = {
        "/gps": {"value": [1.0, 2.0, 3.0]},
        "/imu": {"value": [10.0, 20.0, 30.0, 40.0]},
    }
    assert compose_vector(streams, ("/gps", "/imu")) == [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0]
    assert compose_vector(streams, ("/imu", "/gps")) == [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0]


def test_compose_vector_returns_none_on_missing_component():
    streams = {"/gps": {"value": None}, "/imu": {"value": [1.0]}}
    assert compose_vector(streams, ("/gps", "/imu")) is None
    assert compose_vector(streams, ()) is None


def test_mapping_requires_at_least_one_component():
    with pytest.raises(ValueError):
        CanonicalVectorMapping()


def test_validate_mapping_rejects_stream_not_in_alignment():
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/lidar"))
    with pytest.raises(ValueError, match=r"/lidar"):
        validate_mapping(mapping, ["/gps", "/imu"])


def test_mapping_streams_property_dedupes_preserving_order():
    mapping = CanonicalVectorMapping(
        state_streams=("/gps", "/imu"), action_streams=("/imu", "/action")
    )
    assert mapping.streams == ("/gps", "/imu", "/action")


# ---------------------------------------------------------------------------
# LiveLeRobotFacade -- integration, real lake + real alignment
# ---------------------------------------------------------------------------


def test_facade_composes_fixed_width_frames_across_heterogeneous_topics(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))

    facade = LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)

    assert len(facade) == 6  # 2 episodes x 3 ticks each
    for item in (facade[i] for i in range(len(facade))):
        assert len(item["observation.state"]) == 7  # 3 (gps) + 4 (imu), never mismatched
        assert len(item["action"]) == 1
        assert item["episode_index"] in (0, 1)

    # frame_index resets and is monotonic within each episode
    by_episode: dict[int, list[int]] = {0: [], 1: []}
    for i in range(len(facade)):
        item = facade[i]
        by_episode[item["episode_index"]].append(item["frame_index"])
    assert by_episode[0] == [0, 1, 2]
    assert by_episode[1] == [0, 1, 2]

    assert facade[0]["task"] == "pick"
    assert facade[3]["task"] == "place"


def test_facade_getitems_batches_match_single_item_reads(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"))

    facade = LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)
    indices = list(range(len(facade)))
    batched = facade.__getitems__(indices)
    singles = [facade[i] for i in indices]
    assert batched == singles


def test_facade_rejects_stream_not_part_of_the_alignment(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/lidar"))

    with pytest.raises(ValueError, match=r"/lidar"):
        LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)


def test_facade_raises_on_episode_ids_outside_the_aligned_time_range(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    lake.table("episodes").add(
        pa.Table.from_pylist(
            [_episode_row("ep-empty", 2, 10_000_000_000, 10_100_000_000, "unreachable")],
            schema=EPISODES_SCHEMA,
        )
    )
    mapping = CanonicalVectorMapping(state_streams=("/gps",))

    with pytest.raises(LeRobotFacadeError, match="ep-empty"):
        LiveLeRobotFacade(
            lake, name="facade_view", mapping=mapping, episode_ids=["ep-0", "ep-empty"]
        )


def test_facade_episode_selection_filters_to_requested_ids(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps",))

    facade = LiveLeRobotFacade(lake, name="facade_view", mapping=mapping, episode_ids=["ep-1"])

    assert len(facade) == 3
    assert all(facade[i]["episode_index"] == 0 for i in range(len(facade)))
    assert facade[0]["task"] == "place"


def test_facade_to_torch_dataset_yields_float32_tensors(tmp_path):
    require_torch_loader()
    import torch

    lake = _two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))
    facade = LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)

    torch_dataset = to_torch_dataset(facade)
    assert len(torch_dataset) == len(facade)
    item = torch_dataset[0]
    assert isinstance(item["observation.state"], torch.Tensor)
    assert item["observation.state"].dtype == torch.float32
    assert item["observation.state"].shape == (7,)
