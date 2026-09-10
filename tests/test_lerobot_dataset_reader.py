"""``LancedbRoboticsDatasetReader`` (lerobot ``BaseDatasetReader`` adapter) tests.

Two tiers, split because ``dataset_reader.py`` hard-imports
``lerobot.datasets.dataset_reader.BaseDatasetReader``/``lerobot.datasets.storage``
-- upstream ``huggingface/lerobot`` PR #4363's storage-format registry, merged to
``main`` 2026-08-28 but not in any PyPI release yet, so the module cannot be
imported at all against this repo's default ``lerobot>=0.4.0`` pin:

* Always-on: ``_reader_core.py`` (manifest parsing, episode-boundary index math,
  item tensorization/image decode) and ``manifest.write_dataset_manifest`` have
  no ``lerobot`` dependency and are exercised directly against a real fixture
  lake -- the same ``_two_episode_lake`` fixture ``test_lerobot_facade.py`` uses,
  so parity assertions compare against the exact same ``LiveLeRobotFacade``
  these tests are adapting.
* ``lerobot_main_dev``-marked: registers the real reader against real upstream
  ``lerobot.datasets.storage``/``LeRobotDataset`` and asserts parity end to end.
  Skips cleanly without the dev-only, commit-pinned ``lancedb-robotics
  [lerobot-main-dev]`` extra (see ``require_lerobot_main_dev`` in conftest.py).
"""

import io
import json
import warnings
from datetime import UTC, datetime

import numpy as np
import pytest
from conftest import require_lerobot_main_dev, require_torch_loader
from test_lerobot_facade import _two_episode_lake

from lancedb_robotics.lerobot_facade import CanonicalVectorMapping, LiveLeRobotFacade
from lancedb_robotics.lerobot_facade import _reader_core as core
from lancedb_robotics.lerobot_facade.manifest import write_dataset_manifest


def _write_manifest(*args, **kwargs):
    """Call the deprecated legacy writer without tripping DeprecationWarning.

    These tests deliberately keep exercising the unpinned local-directory path
    (still what `localize_root` returns for an already-materialized root); the
    supported publish flow is covered by ``test_lerobot_view_publish.py``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return write_dataset_manifest(*args, **kwargs)

# ---------------------------------------------------------------------------
# episode_frame_bounds -- pure numpy, no lake needed
# ---------------------------------------------------------------------------


def test_episode_frame_bounds_returns_none_when_unfiltered():
    rel_to_abs, abs_to_rel = core.episode_frame_bounds(
        np.array([0, 3]), np.array([3, 6]), episodes=None
    )
    assert rel_to_abs is None
    assert abs_to_rel is None


def test_episode_frame_bounds_maps_filtered_episodes_in_the_given_order():
    # The reader sorts `episodes` before calling this (ascending episode_index,
    # matching storage order); passing them pre-sorted here documents that
    # this function itself just concatenates in whatever order it is given.
    dataset_from_index = np.array([0, 3, 7])
    dataset_to_index = np.array([3, 7, 10])

    rel_to_abs, abs_to_rel = core.episode_frame_bounds(
        dataset_from_index, dataset_to_index, episodes=[0, 2]
    )

    assert rel_to_abs.tolist() == [0, 1, 2, 7, 8, 9]
    assert abs_to_rel == {0: 0, 1: 1, 2: 2, 7: 3, 8: 4, 9: 5}


def test_episode_frame_bounds_empty_episode_list():
    rel_to_abs, abs_to_rel = core.episode_frame_bounds(
        np.array([0, 3]), np.array([3, 6]), episodes=[]
    )
    assert rel_to_abs.tolist() == []
    assert abs_to_rel == {}


# ---------------------------------------------------------------------------
# decode_image_bytes -- pure, no lake needed
# ---------------------------------------------------------------------------


def _jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (4, 6)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_decode_image_bytes_returns_chw_float_tensor_by_default():
    require_torch_loader()
    import torch

    tensor = core.decode_image_bytes(
        _jpeg_bytes((255, 0, 0), size=(4, 6)), return_uint8=False, image_transforms=None
    )
    assert tensor.shape == (3, 6, 4)  # (C, H, W)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


def test_decode_image_bytes_return_uint8_skips_normalization():
    require_torch_loader()
    import torch

    tensor = core.decode_image_bytes(
        _jpeg_bytes((0, 255, 0)), return_uint8=True, image_transforms=None
    )
    assert tensor.dtype == torch.uint8


def test_decode_image_bytes_applies_image_transforms():
    require_torch_loader()

    tensor = core.decode_image_bytes(
        _jpeg_bytes((0, 0, 255)),
        return_uint8=False,
        image_transforms=lambda t: t * 0.0,
    )
    assert float(tensor.max()) == 0.0


# ---------------------------------------------------------------------------
# manifest.write_dataset_manifest -- real fixture lake, no lerobot needed
# ---------------------------------------------------------------------------


def test_write_dataset_manifest_matches_facade_episode_boundaries(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))
    root = tmp_path / "manifest"

    _write_manifest(
        lake, root, repo_id="lancedb-robotics/facade-test", fps=20, mapping=mapping, name="facade_view"
    )

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["storage_format"] == "lancedb_robotics"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 6
    assert info["total_tasks"] == 0
    assert info["features"]["observation.state"]["shape"] == [7]
    assert info["features"]["action"]["shape"] == [1]

    import pyarrow.parquet as pq

    episodes = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-00000.parquet")
    rows = episodes.to_pylist()
    assert [row["episode_index"] for row in rows] == [0, 1]
    assert [row["dataset_from_index"] for row in rows] == [0, 3]
    assert [row["dataset_to_index"] for row in rows] == [3, 6]
    assert [row["tasks"] for row in rows] == [["pick"], ["place"]]

    source = json.loads((root / "meta" / core.SOURCE_MANIFEST_FILENAME).read_text())
    assert source["lake_uri"] == lake.uri
    assert source["episode_ids"] == ["ep-0", "ep-1"]
    assert source["mapping"] == {
        "state_streams": ["/gps", "/imu"],
        "action_streams": ["/action"],
        "camera_streams": [],
    }


def test_write_dataset_manifest_rejects_zero_frame_configuration(tmp_path):
    # An episode_ids filter matching no known episode resolves an empty
    # FacadeEpisode tuple (build_episode_index simply has nothing to bound),
    # which is *not* what doctor.raise_if_unusable() checks (it only flags an
    # episode that resolves zero ticks while still being requested) -- so this
    # is the one path that actually reaches write_dataset_manifest's own guard.
    lake = _two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps",))

    with pytest.raises(ValueError, match="zero frames"):
        _write_manifest(
            lake,
            tmp_path / "manifest",
            repo_id="r",
            fps=20,
            mapping=mapping,
            name="facade_view",
            episode_ids=["ep-nonexistent"],
        )


# ---------------------------------------------------------------------------
# Round trip: manifest -> _reader_core pipeline == LiveLeRobotFacade directly
# ---------------------------------------------------------------------------


def test_reader_core_pipeline_matches_live_facade_items(tmp_path):
    require_torch_loader()

    lake = _two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))
    root = tmp_path / "manifest"
    _write_manifest(
        lake, root, repo_id="r", fps=20, mapping=mapping, name="facade_view"
    )

    manifest = core.load_source_manifest(root)
    facade = core.open_facade(manifest, episodes=None)
    reference = LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)
    assert len(facade) == len(reference)

    for idx in range(len(reference)):
        expected = reference[idx]
        got = core.facade_item_to_lerobot_item(facade[idx], return_uint8=False, image_transforms=None)
        assert got["episode_index"].item() == expected["episode_index"]
        assert got["frame_index"].item() == expected["frame_index"]
        assert got["task"] == expected["task"]
        assert got["observation.state"].tolist() == pytest.approx(expected["observation.state"])
        assert got["action"].tolist() == pytest.approx(expected["action"])


def test_reader_core_pipeline_respects_episode_filter(tmp_path):
    require_torch_loader()

    lake = _two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps",))
    root = tmp_path / "manifest"
    _write_manifest(
        lake, root, repo_id="r", fps=20, mapping=mapping, name="facade_view"
    )

    manifest = core.load_source_manifest(root)
    facade = core.open_facade(manifest, episodes=[1])
    assert len(facade) == 3
    item = core.facade_item_to_lerobot_item(facade[0], return_uint8=False, image_transforms=None)
    assert item["task"] == "place"


# ---------------------------------------------------------------------------
# delta_timestamps windowing (backlog 0509) -- plan_windows is pure numpy;
# hydrate_windowed_items runs against the real fixture lake, no lerobot needed
# ---------------------------------------------------------------------------

# The fixture's episode layout: two episodes of three 20 Hz ticks each.
_EP_FROM = np.array([0, 3])
_EP_TO = np.array([3, 6])


def test_plan_windows_clamps_to_episode_bounds_and_pads():
    # Forward window overrunning episode 0's end: clamp to the last frame,
    # pad exactly the overrun positions (upstream _plan_batch's formulas).
    plans = core.plan_windows([1], {"action": [0, 1, 2]}, _EP_FROM, _EP_TO)
    assert plans[0].abs_idx == 1
    assert plans[0].windows["action"] == [1, 2, 2]
    assert plans[0].padding["action_is_pad"] == [False, False, True]

    # Backward window underrunning episode 1's start (frame 3 is ep-1's
    # first): clamps to 3, never bleeds into episode 0's frames.
    plans = core.plan_windows([3], {"observation.state": [-2, -1, 0]}, _EP_FROM, _EP_TO)
    assert plans[0].windows["observation.state"] == [3, 3, 3]
    assert plans[0].padding["observation.state_is_pad"] == [True, True, False]


def test_plan_windows_row_union_dedups_across_keys():
    plans = core.plan_windows(
        [4], {"action": [0, 1], "observation.state": [-1, 0]}, _EP_FROM, _EP_TO
    )
    assert plans[0].rows == frozenset({3, 4, 5})


def test_validate_delta_keys_rejects_keys_the_mapping_cannot_serve():
    mapping = CanonicalVectorMapping(state_streams=("/gps",), action_streams=())
    core.validate_delta_keys(["observation.state"], mapping)  # served: no raise
    with pytest.raises(ValueError, match="action"):
        core.validate_delta_keys(["action"], mapping)
    with pytest.raises(ValueError, match="next.reward"):
        core.validate_delta_keys(["next.reward"], mapping)


def _varying_two_episode_lake(path, *, camera_topics: tuple[str, ...] = ()):
    """`_two_episode_lake`'s layout with per-tick distinct vector values.

    The shared fixture's action is 9.0 at every tick, which cannot tell window
    members apart; here every vector value encodes its tick's sequence number
    so a stacked window asserts exactly which frames landed where. Each topic
    in ``camera_topics`` (e.g. ``/cam``) joins the alignment as a vectorless
    stream (observation ids ``<key>-<ts>``) for the stub-VideoIndex
    camera-window tests.
    """
    import pyarrow as pa
    from test_lerobot_facade import _RUN_ID, _episode_row, _observation

    from lancedb_robotics.lake import Lake
    from lancedb_robotics.schemas import EPISODES_SCHEMA, OBSERVATIONS_SCHEMA, RUNS_SCHEMA

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
                    "created_at": datetime(2026, 1, 1, tzinfo=UTC),
                }
            ],
            schema=RUNS_SCHEMA,
        )
    )
    lake.table("episodes").add(
        pa.Table.from_pylist(
            [
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
            base = float(seq)
            rows.append(
                _observation(
                    f"gps-{ts}", "/gps", ts, seq, state_vector=[base, base + 0.1, base + 0.2]
                )
            )
            seq += 1
            rows.append(
                _observation(
                    f"imu-{ts}",
                    "/imu",
                    ts,
                    seq,
                    state_vector=[base + 0.3, base + 0.4, base + 0.5, base + 0.6],
                )
            )
            seq += 1
            rows.append(_observation(f"action-{ts}", "/action", ts, seq, action_vector=[base]))
            seq += 1
            for topic in camera_topics:
                rows.append(_observation(f"{topic.strip('/')}-{ts}", topic, ts, seq))
                seq += 1
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))

    streams = ["/gps", "/imu", "/action", *camera_topics]
    lake.align.create_view(
        "facade_view",
        run_id=_RUN_ID,
        rate_hz=20.0,
        streams=streams,
        tolerance_ms=100.0,
        interpolation={stream: "nearest" for stream in streams},
    )
    return lake


class _StubVideoIndex:
    """`VideoIndex` stand-in: real JPEG bytes, recorded locate/decode calls.

    The GOP decode chain has its own conformance suite
    (`test_lerobot_video_decode_conformance.py` and the realcorpus facade
    test); what the windowing tests need from a video index is only its
    contract -- `locate` and batch-deduped `decode_batch` -- plus call
    recording, so decode *bounds* can be asserted, not just pixel values.
    """

    def __init__(self, locations: dict[str, tuple[str, int]]):
        self._locations = locations
        self.decode_requests: list[list[tuple[str, int]]] = []

    def locate(self, *, camera_key, episode_id, observation_id):
        del camera_key, episode_id
        if not observation_id:
            return None
        return self._locations.get(observation_id)

    def decode_batch(self, requests):
        self.decode_requests.append(list(requests))
        return {request: self.jpeg_for(request) for request in requests}

    @staticmethod
    def jpeg_for(request: tuple[str, int]) -> bytes:
        encoding_id, frame_index = request
        red = 200 if encoding_id.endswith("ep-1") else 50
        return _jpeg_bytes((red, (frame_index * 60) % 256, 0))


def _camera_facade(tmp_path, *, camera_topics: tuple[str, ...] = ("/cam",)):
    lake = _varying_two_episode_lake(tmp_path / "robot.lance", camera_topics=camera_topics)
    mapping = CanonicalVectorMapping(
        state_streams=("/gps", "/imu"), action_streams=("/action",), camera_streams=camera_topics
    )
    facade = LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)
    locations = {}
    for topic in camera_topics:
        key = topic.strip("/")
        for episode_start, episode_id in ((0, "ep-0"), (200_000_000, "ep-1")):
            for frame_index, offset in enumerate((0, 50_000_000, 100_000_000)):
                locations[f"{key}-{episode_start + offset}"] = (
                    f"enc-{key}-{episode_id}",
                    frame_index,
                )
    stub = _StubVideoIndex(locations)
    facade._video_index = stub
    return facade, stub


def test_hydrate_windowed_items_stacks_vectors_with_boundary_padding(tmp_path):
    require_torch_loader()

    lake = _varying_two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))
    facade = LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)
    reference = [facade[i] for i in range(len(facade))]

    plans = core.plan_windows(
        [0, 2], {"action": [0, 1], "observation.state": [-1, 0]}, _EP_FROM, _EP_TO
    )
    items = core.hydrate_windowed_items(
        facade, plans, abs_to_facade=None, return_uint8=False, image_transforms=None
    )

    import torch

    # Frame 0: action window [0, 1] real, state window clamps [-1] -> frame 0.
    assert items[0]["action"].shape == (2, 1)
    assert items[0]["action"].dtype == torch.float32
    assert items[0]["action"].tolist() == [reference[0]["action"], reference[1]["action"]]
    assert items[0]["action_is_pad"].tolist() == [False, False]
    assert items[0]["observation.state"].shape == (2, 7)
    torch.testing.assert_close(
        items[0]["observation.state"],
        torch.tensor(
            [reference[0]["observation.state"], reference[0]["observation.state"]],
            dtype=torch.float32,
        ),
    )
    assert items[0]["observation.state_is_pad"].tolist() == [True, False]

    # Frame 2 is episode 0's last: the forward window repeats it, padded --
    # and never bleeds into episode 1's frame 3.
    assert items[1]["action"].tolist() == [reference[2]["action"], reference[2]["action"]]
    assert items[1]["action_is_pad"].tolist() == [False, True]
    torch.testing.assert_close(
        items[1]["observation.state"],
        torch.tensor(
            [reference[1]["observation.state"], reference[2]["observation.state"]],
            dtype=torch.float32,
        ),
    )
    assert items[1]["observation.state_is_pad"].tolist() == [False, False]

    # Base per-frame fields are untouched by windowing.
    assert items[1]["episode_index"].item() == 0
    assert items[1]["frame_index"].item() == 2
    assert items[1]["task"] == "pick"


def test_hydrate_windowed_items_reads_the_row_union_once(tmp_path):
    require_torch_loader()

    lake = _varying_two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))
    facade = LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)

    hydrate_calls: list[int] = []
    original_hydrate = facade.hydrate_batch

    def _spying_hydrate(indices, **kwargs):
        hydrate_calls.append(len(list(indices)))
        return original_hydrate(indices, **kwargs)

    facade.hydrate_batch = _spying_hydrate

    # Three overlapping k=2 windows over frames 0..2 stay inside episode 0
    # (frame 2's forward member clamps onto itself), touching rows {0, 1, 2}:
    # one hydrate call for the deduplicated union -- 3 rows, not 3 * 2 = 6.
    plans = core.plan_windows([0, 1, 2], {"action": [0, 1]}, _EP_FROM, _EP_TO)
    items = core.hydrate_windowed_items(
        facade, plans, abs_to_facade=None, return_uint8=False, image_transforms=None
    )
    assert len(items) == 3
    assert hydrate_calls == [3]


def test_windowed_camera_stacks_frames_with_one_gop_batched_decode(tmp_path, monkeypatch):
    require_torch_loader()

    facade, stub = _camera_facade(tmp_path)

    # Pin the per-(frame, camera) JPEG-decode cache: dropping it would leave
    # every value and decode_batch assertion green while multiplying CPU
    # decode by the window-overlap factor (SKILLS.md: pin every guardrail).
    real_decode = core.decode_image_bytes
    decode_calls: list[bytes] = []

    def _counting_decode(data, **kwargs):
        decode_calls.append(data)
        return real_decode(data, **kwargs)

    monkeypatch.setattr(core, "decode_image_bytes", _counting_decode)

    plans = core.plan_windows([0, 1], {"observation.images.cam": [-1, 0]}, _EP_FROM, _EP_TO)
    items = core.hydrate_windowed_items(
        facade, plans, abs_to_facade=None, return_uint8=False, image_transforms=None
    )

    import torch

    key = "observation.images.cam"
    # Frame 0's backward window clamps [-1] onto frame 0 itself: two stacked
    # copies of frame 0's pixels, first position padded.
    assert items[0][key].shape[0] == 2  # (k, C, H, W)
    frame0 = real_decode(
        stub.jpeg_for(("enc-cam-ep-0", 0)), return_uint8=False, image_transforms=None
    )
    frame1 = real_decode(
        stub.jpeg_for(("enc-cam-ep-0", 1)), return_uint8=False, image_transforms=None
    )
    torch.testing.assert_close(items[0][key][0], frame0)
    torch.testing.assert_close(items[0][key][1], frame0)
    assert items[0][f"{key}_is_pad"].tolist() == [True, False]
    torch.testing.assert_close(items[1][key][0], frame0)
    torch.testing.assert_close(items[1][key][1], frame1)
    assert items[1][f"{key}_is_pad"].tolist() == [False, False]

    # One decode_batch call for the whole batch, over the deduplicated
    # window-row union: frames {0, 1}, not 2 samples * k=2 = 4 requests.
    assert stub.decode_requests == [[("enc-cam-ep-0", 0), ("enc-cam-ep-0", 1)]]
    # And exactly one PIL decode per distinct (frame, camera) despite the
    # windows overlapping across both samples.
    assert len(decode_calls) == 2


def test_single_delta_camera_window_squeezes_like_upstream(tmp_path):
    require_torch_loader()

    facade, stub = _camera_facade(tmp_path)
    plans = core.plan_windows([1], {"observation.images.cam": [0]}, _EP_FROM, _EP_TO)
    items = core.hydrate_windowed_items(
        facade, plans, abs_to_facade=None, return_uint8=False, image_transforms=None
    )
    # Upstream squeezes a k=1 video window to (C, H, W); tabular k=1 windows
    # keep their leading dim (both per LanceDatasetReader._build_item).
    assert items[0]["observation.images.cam"].dim() == 3
    assert items[0]["observation.images.cam_is_pad"].tolist() == [False]


def test_action_window_never_multiplies_camera_decode(tmp_path):
    require_torch_loader()

    facade, stub = _camera_facade(tmp_path)
    # k=3 action window from frame 0 hydrates rows {0, 1, 2}, but the camera
    # is not windowed: exactly one decoded frame -- the base frame -- not 3.
    plans = core.plan_windows([0], {"action": [0, 1, 2]}, _EP_FROM, _EP_TO)
    items = core.hydrate_windowed_items(
        facade, plans, abs_to_facade=None, return_uint8=False, image_transforms=None
    )
    assert stub.decode_requests == [[("enc-cam-ep-0", 0)]]
    assert items[0]["observation.images.cam"].dim() == 3  # plain per-frame tensor
    assert items[0]["action"].shape == (3, 1)


def test_mixed_windowed_and_plain_cameras_partition_decode(tmp_path):
    require_torch_loader()

    # Camera A windowed, camera B not: A decodes at exactly its window-row
    # union, B at exactly the base frames -- the set arithmetic behind
    # base_camera_keys / camera_keys_per_frame, pinned with two cameras so a
    # regression to decode-everything-everywhere cannot hide in a
    # single-camera fixture.
    facade, stub = _camera_facade(tmp_path, camera_topics=("/cam", "/cam2"))
    plans = core.plan_windows([1], {"observation.images.cam": [-1, 0]}, _EP_FROM, _EP_TO)
    items = core.hydrate_windowed_items(
        facade, plans, abs_to_facade=None, return_uint8=False, image_transforms=None
    )

    assert len(stub.decode_requests) == 1
    assert sorted(stub.decode_requests[0]) == [
        ("enc-cam-ep-0", 0),  # windowed camera, window row 0
        ("enc-cam-ep-0", 1),  # windowed camera, window row 1 (base)
        ("enc-cam2-ep-0", 1),  # plain camera, base frame only
    ]
    assert items[0]["observation.images.cam"].dim() == 4  # (k, C, H, W)
    assert items[0]["observation.images.cam2"].dim() == 3  # (C, H, W)


def test_windowed_camera_missing_frame_fails_loudly(tmp_path):
    require_torch_loader()

    facade, stub = _camera_facade(tmp_path)
    del stub._locations["cam-50000000"]  # frame 1's camera never encoded
    plans = core.plan_windows([0], {"observation.images.cam": [0, 1]}, _EP_FROM, _EP_TO)
    with pytest.raises(core.WindowHydrationError, match="observation.images.cam"):
        core.hydrate_windowed_items(
            facade, plans, abs_to_facade=None, return_uint8=False, image_transforms=None
        )


def test_hydrate_windowed_items_respects_episode_filter_remapping(tmp_path):
    require_torch_loader()

    lake = _varying_two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))
    root = tmp_path / "manifest"
    _write_manifest(lake, root, repo_id="r", fps=20, mapping=mapping, name="facade_view")

    manifest = core.load_source_manifest(root)
    # Episode filter [1]: the facade serves only episode 1's three frames, so
    # absolute window rows (3..5) must remap through absolute_to_relative_idx.
    facade = core.open_facade(manifest, episodes=[1])
    unfiltered = core.open_facade(manifest, episodes=None)
    _, abs_to_rel = core.episode_frame_bounds(_EP_FROM, _EP_TO, [1])

    plans = core.plan_windows([5], {"action": [0, 1]}, _EP_FROM, _EP_TO)
    items = core.hydrate_windowed_items(
        facade, plans, abs_to_facade=abs_to_rel, return_uint8=False, image_transforms=None
    )
    assert items[0]["action"].tolist() == [unfiltered[5]["action"], unfiltered[5]["action"]]
    assert items[0]["action_is_pad"].tolist() == [False, True]


# ---------------------------------------------------------------------------
# Real upstream lerobot integration (gated -- see require_lerobot_main_dev)
# ---------------------------------------------------------------------------


@pytest.mark.lerobot_main_dev
def test_registered_reader_opens_as_real_lerobot_dataset(tmp_path):
    require_lerobot_main_dev()
    require_torch_loader()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import lancedb_robotics.lerobot_facade.dataset_reader  # noqa: F401  (registers the format)

    lake = _two_episode_lake(tmp_path / "robot.lance")
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))
    root = tmp_path / "manifest"
    _write_manifest(
        lake, root, repo_id="lancedb-robotics/facade-test", fps=20, mapping=mapping, name="facade_view"
    )
    reference = LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)

    dataset = LeRobotDataset("lancedb-robotics/facade-test", root=root)

    assert len(dataset) == len(reference)
    assert dataset.num_episodes == 2
    for idx in range(len(dataset)):
        item = dataset[idx]
        expected = reference[idx]
        assert item["episode_index"].item() == expected["episode_index"]
        assert item["observation.state"].tolist() == pytest.approx(expected["observation.state"])
        assert item["action"].tolist() == pytest.approx(expected["action"])
        assert item["task"] == expected["task"]

    # picklable: DataLoader workers reopen their own connection.
    import pickle

    restored = pickle.loads(pickle.dumps(dataset.reader))
    assert restored.get_item(0)["task"] == reference[0]["task"]

    # delta_timestamps through the real LeRobotDataset stack (backlog 0509):
    # fps=20, so [0.0, 0.05] is frame deltas [0, 1] and [-0.05, 0.0] is
    # [-1, 0]; masks/values follow upstream's episode-boundary padding.
    windowed = LeRobotDataset(
        "lancedb-robotics/facade-test",
        root=root,
        delta_timestamps={"action": [0.0, 0.05], "observation.state": [-0.05, 0.0]},
    )
    import torch

    first = windowed[0]
    assert first["action"].shape == (2, 1)
    assert first["observation.state"].shape == (2, 7)
    torch.testing.assert_close(
        first["action"],
        torch.tensor([reference[0]["action"], reference[1]["action"]], dtype=torch.float32),
    )
    assert first["action_is_pad"].tolist() == [False, False]
    assert first["observation.state_is_pad"].tolist() == [True, False]
    episode_end = windowed[2]  # episode 0's last frame
    torch.testing.assert_close(
        episode_end["action"],
        torch.tensor([reference[2]["action"], reference[2]["action"]], dtype=torch.float32),
    )
    assert episode_end["action_is_pad"].tolist() == [False, True]

    # A key the view cannot serve fails at construction, not mid-training.
    with pytest.raises(ValueError, match="next.reward"):
        LeRobotDataset(
            "lancedb-robotics/facade-test", root=root, delta_timestamps={"next.reward": [0.0]}
        )


@pytest.mark.lerobot_main_dev
def test_published_view_resolves_stats_and_normalization_end_to_end(tmp_path, monkeypatch):
    """0490/0491 acceptance against real upstream lerobot.

    Opens a *published* view through the real ``LeRobotDataset`` with
    ``root=<lake path>`` (no hand-distributed directory): ``localize_root``
    materializes the view's ``meta/`` from the lake's ``lerobot_view_files``
    rows into the per-view cache. Asserts ``meta.stats`` is populated for every
    declared feature, that a real ``NormalizerProcessorStep`` builds
    normalization buffers from it (MEAN_STD and QUANTILES modes -- the
    quantile keys are why 0490 publishes q01..q99), and that after the lake
    advances the same root still serves the pinned view with episode
    boundaries that agree with the frames -- the exact silent-skew case.
    """
    require_lerobot_main_dev()
    require_torch_loader()

    import torch
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    import lancedb_robotics.lerobot_facade.dataset_reader  # noqa: F401  (registers the format)
    from lancedb_robotics.lerobot_facade import publish_view

    monkeypatch.setenv("LANCEDB_ROBOTICS_VIEW_CACHE", str(tmp_path / "view-cache"))
    lake_path = tmp_path / "robot.lance"
    lake = _two_episode_lake(lake_path)
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))
    publish_view(
        lake, repo_id="acme/pick-place-v1", fps=20, mapping=mapping, name="facade_view"
    )
    reference = LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)

    dataset = LeRobotDataset("acme/pick-place-v1", root=f"file://{lake_path}")
    assert len(dataset) == len(reference) == 6

    # meta.stats populated for every declared feature, straight from the lake.
    stats = dataset.meta.stats
    assert stats is not None
    for feature in dataset.meta.features:
        assert feature in stats, f"stats missing for declared feature {feature!r}"
        for key in ("mean", "std", "min", "max", "count", "q01", "q50", "q99"):
            assert key in stats[feature]
    state = np.array([reference[i]["observation.state"] for i in range(len(reference))])
    np.testing.assert_allclose(stats["observation.state"]["mean"], state.mean(axis=0))

    # Real normalization buffers from meta.stats: MEAN_STD for state and the
    # QUANTILES mode (needs q01/q99) for action.
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state.shape[1],)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.QUANTILES,
    }
    normalizer = NormalizerProcessorStep.from_lerobot_dataset(dataset, features, norm_map)
    item = dataset[0]
    from lerobot.lerobot_types import TransitionKey

    transition = normalizer(
        {
            TransitionKey.OBSERVATION: {"observation.state": item["observation.state"]},
            TransitionKey.ACTION: item["action"],
        }
    )
    normalized = transition[TransitionKey.OBSERVATION]["observation.state"]
    expected = (item["observation.state"] - torch.tensor(stats["observation.state"]["mean"])) / (
        torch.tensor(stats["observation.state"]["std"]) + 1e-8
    )
    torch.testing.assert_close(normalized.to(torch.float64), expected.to(torch.float64))

    # Lake advances after publish; the same root still serves the pinned view,
    # and episode boundaries agree with the frames actually served.
    import pyarrow as pa
    from test_lerobot_facade import _episode_row

    from lancedb_robotics.schemas import EPISODES_SCHEMA

    lake.table("episodes").add(
        pa.Table.from_pylist(
            [_episode_row("ep-2", 2, 301_000_000, 360_000_000, "stack")],
            schema=EPISODES_SCHEMA,
        )
    )
    assert len(LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)) > 6

    reopened = LeRobotDataset("acme/pick-place-v1", root=f"file://{lake_path}")
    assert len(reopened) == 6
    episodes = reopened.meta.episodes.data.to_pylist()
    for row in episodes:
        first = reopened[int(row["dataset_from_index"])]
        last = reopened[int(row["dataset_to_index"]) - 1]
        assert first["episode_index"].item() == row["episode_index"]
        assert last["episode_index"].item() == row["episode_index"]


@pytest.mark.lerobot_main_dev
def test_real_policy_builds_normalization_buffers_and_forward_passes(tmp_path, monkeypatch):
    """A real upstream policy (ACT, tiny state-only config) trains one step
    against a published lakehouse view: normalization buffers come from
    ``meta.stats`` and a forward pass completes (0490 acceptance)."""
    require_lerobot_main_dev()
    require_torch_loader()

    import torch
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    import lancedb_robotics.lerobot_facade.dataset_reader  # noqa: F401
    from lancedb_robotics.lerobot_facade import publish_view

    monkeypatch.setenv("LANCEDB_ROBOTICS_VIEW_CACHE", str(tmp_path / "view-cache"))
    lake_path = tmp_path / "robot.lance"
    lake = _two_episode_lake(lake_path)
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))
    publish_view(
        lake, repo_id="acme/pick-place-v1", fps=20, mapping=mapping, name="facade_view"
    )
    dataset = LeRobotDataset("acme/pick-place-v1", root=f"file://{lake_path}")

    chunk = 2
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            # ACT requires an image or environment-state input; feed the same
            # composed state vector as env state so no vision backbone loads.
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(7,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        },
        chunk_size=chunk,
        n_action_steps=1,
        # vision_backbone keeps its default; with no VISUAL input features the
        # backbone is never instantiated (modeling_act gates it on
        # config.image_features), so nothing is downloaded.
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
    )
    policy = ACTPolicy(config)

    normalizer = NormalizerProcessorStep(
        features={**config.input_features, **config.output_features},
        norm_map=config.normalization_mapping,
        stats=dataset.meta.stats,
    )
    items = [dataset[i] for i in range(chunk)]
    batch = {
        "observation.state": torch.stack([items[0]["observation.state"]]),
        "action": torch.stack([torch.stack([item["action"] for item in items])]),
        "action_is_pad": torch.zeros(1, chunk, dtype=torch.bool),
    }
    from lerobot.lerobot_types import TransitionKey

    normalized_observation = normalizer(
        {TransitionKey.OBSERVATION: {"observation.state": batch["observation.state"]}}
    )[TransitionKey.OBSERVATION]
    batch["observation.state"] = normalized_observation["observation.state"]
    batch["observation.environment_state"] = batch["observation.state"]

    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss)


@pytest.mark.lerobot_main_dev
def test_act_chunking_trains_a_step_via_delta_timestamps(tmp_path, monkeypatch):
    """0509 acceptance: a real ACT policy with a nonzero action horizon trains
    one optimizer step against a published view, with the action chunk and its
    ``action_is_pad`` mask produced by the reader's ``delta_timestamps``
    windowing -- not hand-assembled by the test (contrast the 0490 test
    above, written when the reader still raised ``NotImplementedError``)."""
    require_lerobot_main_dev()
    require_torch_loader()

    import torch
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    import lancedb_robotics.lerobot_facade.dataset_reader  # noqa: F401
    from lancedb_robotics.lerobot_facade import publish_view

    monkeypatch.setenv("LANCEDB_ROBOTICS_VIEW_CACHE", str(tmp_path / "view-cache"))
    lake_path = tmp_path / "robot.lance"
    lake = _two_episode_lake(lake_path)
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))
    publish_view(lake, repo_id="acme/pick-place-v1", fps=20, mapping=mapping, name="facade_view")

    chunk = 2
    fps = 20
    dataset = LeRobotDataset(
        "acme/pick-place-v1",
        root=f"file://{lake_path}",
        delta_timestamps={"action": [i / fps for i in range(chunk)]},
    )

    # The reader assembles the horizon: (chunk, action_dim) plus the pad mask,
    # padded exactly at each episode's tail frame.
    item = dataset[0]
    assert item["action"].shape == (chunk, 1)
    assert item["action_is_pad"].tolist() == [False, False]
    assert dataset[2]["action_is_pad"].tolist() == [False, True]

    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(7,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        },
        chunk_size=chunk,
        n_action_steps=1,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
    )
    policy = ACTPolicy(config)
    normalizer = NormalizerProcessorStep(
        features={**config.input_features, **config.output_features},
        norm_map=config.normalization_mapping,
        stats=dataset.meta.stats,
    )

    from lerobot.lerobot_types import TransitionKey

    items = [dataset[i] for i in (0, 3)]  # one sample per episode
    batch = {
        "observation.state": torch.stack([item["observation.state"] for item in items]),
        "action": torch.stack([item["action"] for item in items]),
        "action_is_pad": torch.stack([item["action_is_pad"] for item in items]),
    }
    normalized_observation = normalizer(
        {TransitionKey.OBSERVATION: {"observation.state": batch["observation.state"]}}
    )[TransitionKey.OBSERVATION]
    batch["observation.state"] = normalized_observation["observation.state"]
    batch["observation.environment_state"] = batch["observation.state"]

    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()  # a real optimizer step, not just a forward pass
