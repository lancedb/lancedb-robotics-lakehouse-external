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

    with pytest.raises(NotImplementedError):
        LeRobotDataset(
            "lancedb-robotics/facade-test", root=root, delta_timestamps={"observation.state": [0.0]}
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
