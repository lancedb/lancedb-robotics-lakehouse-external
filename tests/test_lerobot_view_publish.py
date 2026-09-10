"""Published, version-pinned LeRobot views + normalization statistics (0490/0491).

Three always-on tiers, none needing ``lerobot``:

* Pure-numpy statistics unit tests: the streaming accumulator against brute
  force (including the adversarial expanding-range rebin path), the
  upstream-mirrored camera sampling policy, and the bounded-batch mechanism
  (SKILLS.md: assert the read *bound* and the mechanism, not just the answer).
* ``compute_view_stats`` against a stub facade: camera per-channel stats over
  real JPEG bytes, PIL-missing loud failure, sampling-record contents.
* Real fixture-lake integration over ``test_lerobot_facade._two_episode_lake``:
  publish -> catalog rows -> materialize -> stats.json vs brute force; pins
  surviving a lake advance (the exact silent-skew case 0491 exists to kill);
  legacy unpinned manifests failing loudly instead; republish-after-advance
  creating a new view while the old one stays reproducible; concurrent
  materialization converging on one cache dir; ``localize_root`` resolving a
  lake root end to end.

The gated ``lerobot_main_dev`` tier (real ``LeRobotDataset``/policy
normalization) lives in ``test_lerobot_dataset_reader.py``.
"""

import hashlib
import io
import json
import threading
import warnings
from datetime import UTC, datetime

import numpy as np
import pytest
from test_lerobot_facade import _episode_row, _two_episode_lake

from lancedb_robotics.lerobot_facade import (
    CanonicalVectorMapping,
    LiveLeRobotFacade,
    PinnedLake,
    StaleViewVersionError,
    ViewNotFoundError,
    ViewStatsError,
    get_view,
    list_views,
    materialize_view,
    open_published_facade,
    publish_view,
)
from lancedb_robotics.lerobot_facade import _reader_core as core
from lancedb_robotics.lerobot_facade import stats as stats_mod
from lancedb_robotics.lerobot_facade.stats import (
    StreamingFeatureStats,
    camera_sample_indices,
    compute_view_stats,
    downsample_image,
    estimate_camera_samples,
    serialize_stats,
)
from lancedb_robotics.lerobot_facade.views import VIEW_FILES_TABLE, VIEWS_TABLE
from lancedb_robotics.schemas import EPISODES_SCHEMA

_MAPPING = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))


# ---------------------------------------------------------------------------
# StreamingFeatureStats -- pure numpy, no lake
# ---------------------------------------------------------------------------


def _stream_in_batches(data: np.ndarray, batch_sizes: list[int]) -> StreamingFeatureStats:
    acc = StreamingFeatureStats()
    start = 0
    for size in batch_sizes:
        acc.update(data[start : start + size])
        start += size
    assert start == len(data)
    return acc


def test_streaming_stats_match_brute_force_across_uneven_batches():
    rng = np.random.default_rng(7)
    data = rng.normal(loc=[0.0, 100.0, -5.0], scale=[1.0, 30.0, 0.01], size=(5000, 3))
    acc = _stream_in_batches(data, [1, 7, 250, 1024, 3000, 718])

    stats = acc.statistics()
    np.testing.assert_allclose(stats["mean"], data.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(stats["std"], data.std(axis=0), rtol=1e-9)
    np.testing.assert_allclose(stats["min"], data.min(axis=0))
    np.testing.assert_allclose(stats["max"], data.max(axis=0))
    assert stats["count"].tolist() == [5000]
    # Histogram quantiles: correct to roughly a bin width, per dimension.
    tolerance = (data.max(axis=0) - data.min(axis=0)) / 500
    for quantile, key in ((0.01, "q01"), (0.5, "q50"), (0.99, "q99")):
        np.testing.assert_allclose(
            stats[key], np.quantile(data, quantile, axis=0), atol=tolerance.max()
        )


def test_streaming_stats_rebin_when_later_batches_expand_the_range():
    rng = np.random.default_rng(11)
    narrow = rng.uniform(-1.0, 1.0, size=(2000, 2))
    wide = rng.uniform(-1000.0, 1000.0, size=(2000, 2))
    data = np.concatenate([narrow, wide])

    acc = StreamingFeatureStats()
    acc.update(narrow)  # histograms initialized over [-1, 1]
    acc.update(wide)  # range expands 1000x -> rebin path

    stats = acc.statistics()
    np.testing.assert_allclose(stats["mean"], data.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(stats["std"], data.std(axis=0), rtol=1e-9)
    np.testing.assert_allclose(
        stats["q50"], np.quantile(data, 0.5, axis=0), atol=2000 / 500
    )


def test_streaming_stats_reject_non_finite_and_width_changes():
    acc = StreamingFeatureStats()
    acc.update(np.ones((4, 2)))
    with pytest.raises(ViewStatsError, match="non-finite"):
        acc.update(np.array([[1.0, np.nan]]))
    with pytest.raises(ViewStatsError, match="width changed"):
        acc.update(np.ones((4, 3)))
    with pytest.raises(ViewStatsError, match="no rows"):
        StreamingFeatureStats().statistics()


def test_camera_sampling_policy_mirrors_upstream():
    # Values straight from upstream estimate_num_samples' own docstring.
    assert estimate_camera_samples(50) == 50
    assert estimate_camera_samples(400) == 100
    assert estimate_camera_samples(1000) == 177
    assert estimate_camera_samples(10_000) == 1000
    assert estimate_camera_samples(10_000_000) == 10_000
    indices = camera_sample_indices(1000, 177)
    assert indices == camera_sample_indices(1000, 177)  # deterministic
    assert indices[0] == 0 and indices[-1] == 999 and len(indices) == 177


def test_downsample_image_strides_only_large_frames():
    small = np.zeros((3, 100, 200), dtype=np.uint8)
    assert downsample_image(small).shape == small.shape
    large = np.zeros((3, 300, 600), dtype=np.uint8)
    downsampled = downsample_image(large)
    assert max(downsampled.shape[1:]) <= 300
    assert downsampled.shape[2] == 150


def test_serialize_stats_produces_plain_json_lists():
    acc = StreamingFeatureStats()
    acc.update(np.arange(20, dtype=np.float64).reshape(10, 2))
    payload = serialize_stats({"observation.state": acc.statistics()})
    text = json.dumps(payload)  # must be JSON-serializable as-is
    parsed = json.loads(text)
    assert isinstance(parsed["observation.state"]["mean"], list)
    assert parsed["observation.state"]["count"] == [10]


# ---------------------------------------------------------------------------
# compute_view_stats -- stub facade (bounded-batch mechanism + camera path)
# ---------------------------------------------------------------------------


def _jpeg(color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class _StubFacade:
    """Minimal facade surface compute_view_stats touches, with spies."""

    def __init__(self, frames: int, camera_color=(200, 40, 90)) -> None:
        self._frames = frames
        self._camera_bytes = _jpeg(camera_color)
        self.tabular_calls: list[int] = []
        self.camera_decodes = 0

    def __len__(self) -> int:
        return self._frames

    def has_feature(self, key: str) -> bool:
        return key == "observation.state"

    def tabular_feature_batch(self, indices) -> dict:
        indices = list(indices)
        self.tabular_calls.append(len(indices))
        return {
            "index": indices,
            "episode_index": [0] * len(indices),
            "frame_index": indices,
            "timestamp": [i / 20.0 for i in indices],
            "observation.state": [[float(i), float(i) * -2.0] for i in indices],
        }

    def __getitems__(self, indices) -> list[dict]:
        self.camera_decodes += len(indices)
        return [
            {"index": i, "observation.images.cam": self._camera_bytes} for i in indices
        ]


def test_compute_view_stats_streams_bounded_batches_and_samples_cameras():
    facade = _StubFacade(frames=5000)
    stats, record = compute_view_stats(
        facade, camera_keys=("observation.images.cam",), batch_size=256
    )

    # Mechanism, not just answers (SKILLS.md testing rule): the full scan went
    # through the no-camera-decode batch reader in bounded chunks, and camera
    # decode touched exactly the deterministic sample, never the whole view.
    assert max(facade.tabular_calls) <= 256
    assert sum(facade.tabular_calls) == 5000
    expected_samples = estimate_camera_samples(5000)
    assert facade.camera_decodes == expected_samples
    assert record["camera"]["num_samples"] == expected_samples
    assert record["camera"]["method"] == "linspace"
    assert record["tabular"] == {"method": "full-scan", "batch_size": 256, "frames": 5000}

    values = np.array([[float(i), float(i) * -2.0] for i in range(5000)])
    np.testing.assert_allclose(stats["observation.state"]["mean"], values.mean(axis=0))
    np.testing.assert_allclose(stats["observation.state"]["std"], values.std(axis=0), rtol=1e-9)
    assert stats["observation.state"]["count"].tolist() == [5000]
    assert stats["timestamp"]["mean"].shape == (1,)
    assert stats["index"]["max"].tolist() == [4999.0]

    camera = stats["observation.images.cam"]
    assert camera["mean"].shape == (3, 1, 1)
    assert camera["count"].tolist() == [expected_samples]
    # Solid-color JPEG: per-channel mean lands on the color / 255 (with a small
    # compression tolerance), already normalized into [0, 1].
    np.testing.assert_allclose(
        camera["mean"].reshape(3), np.array([200, 40, 90]) / 255.0, atol=3 / 255
    )
    assert float(camera["max"].max()) <= 1.0


def test_compute_view_stats_fails_loudly_without_pil_when_cameras_mapped(monkeypatch):
    facade = _StubFacade(frames=32)
    monkeypatch.setattr(stats_mod.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ViewStatsError, match=r"lancedb-robotics\[media\]"):
        compute_view_stats(facade, camera_keys=("observation.images.cam",))
    # No cameras mapped -> PIL never consulted, stats still computed.
    stats, _ = compute_view_stats(facade, camera_keys=())
    assert "observation.state" in stats


def test_compute_view_stats_rejects_feature_with_no_complete_frames():
    facade = _StubFacade(frames=8)

    def all_missing(indices):
        batch = _StubFacade.tabular_feature_batch(facade, indices)
        batch["observation.state"] = [None] * len(batch["index"])
        return batch

    facade.tabular_feature_batch = all_missing
    with pytest.raises(ViewStatsError, match="no resolved frame"):
        compute_view_stats(facade)


# ---------------------------------------------------------------------------
# Episode-tiling guard -- pure numpy
# ---------------------------------------------------------------------------


def test_validate_episode_tiling_accepts_exact_tiling_and_rejects_everything_else():
    core.validate_episode_tiling(np.array([0, 3]), np.array([3, 6]), 6)
    core.validate_episode_tiling(np.array([], dtype=np.int64), np.array([], dtype=np.int64), 0)
    with pytest.raises(core.ManifestDriftError, match="do not tile"):
        core.validate_episode_tiling(np.array([0, 4]), np.array([3, 6]), 6)  # gap
    with pytest.raises(core.ManifestDriftError, match="do not tile"):
        core.validate_episode_tiling(np.array([0, 2]), np.array([3, 6]), 6)  # overlap
    with pytest.raises(core.ManifestDriftError, match="end at 5"):
        core.validate_episode_tiling(np.array([0, 3]), np.array([3, 5]), 6)  # short
    with pytest.raises(core.ManifestDriftError, match="declares 6"):
        core.validate_episode_tiling(np.array([], dtype=np.int64), np.array([], dtype=np.int64), 6)


# ---------------------------------------------------------------------------
# publish_view / materialize / pinning -- real fixture lake
# ---------------------------------------------------------------------------


def _advance_lake(lake) -> None:
    """Append a third episode that captures one more aligned tick (350ms).

    The alignment materialized ticks across the whole run; the two fixture
    episodes bucket only six of them. Adding an episode window over 350ms makes
    the *live* facade resolve one more frame -- exactly the post-publish drift
    0491's pins and guards exist for.
    """
    import pyarrow as pa

    lake.table("episodes").add(
        pa.Table.from_pylist(
            [_episode_row("ep-2", 2, 301_000_000, 360_000_000, "stack")],
            schema=EPISODES_SCHEMA,
        )
    )


def _publish(lake, **overrides):
    kwargs = dict(
        repo_id="acme/pick-place-v1",
        fps=20,
        mapping=_MAPPING,
        name="facade_view",
        robot_type="test-arm",
        created_by="tests",
    )
    kwargs.update(overrides)
    return publish_view(lake, **kwargs)


def test_publish_view_writes_catalog_rows_and_manifest_files(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)

    assert published.total_frames == 6
    assert published.total_episodes == 2
    header = get_view(lake, repo_id="acme/pick-place-v1")
    assert header["view_id"] == published.view_id
    assert header["file_count"] == published.file_count
    pinned_tables = {item["table"] for item in header["table_versions"]}
    assert {"aligned_ticks", "episodes", "runs", "observations"} <= pinned_tables

    dest = materialize_view(lake, header, tmp_path / "materialized", lake_uri=lake.uri)
    info = json.loads((dest / "meta" / "info.json").read_text())
    assert info["total_frames"] == 6
    stats = json.loads((dest / "meta" / "stats.json").read_text())
    assert set(info["features"]) <= set(stats)
    for feature_stats in stats.values():
        assert {"mean", "std", "min", "max", "count", "q01", "q50", "q99"} <= set(feature_stats)

    # Brute force over the exact frames the facade serves (0490 acceptance).
    facade = LiveLeRobotFacade(lake, name="facade_view", mapping=_MAPPING)
    state = np.array([facade[i]["observation.state"] for i in range(len(facade))])
    np.testing.assert_allclose(stats["observation.state"]["mean"], state.mean(axis=0))
    np.testing.assert_allclose(stats["observation.state"]["std"], state.std(axis=0), atol=1e-12)
    np.testing.assert_allclose(stats["observation.state"]["min"], state.min(axis=0))
    np.testing.assert_allclose(stats["observation.state"]["max"], state.max(axis=0))
    assert stats["observation.state"]["count"] == [6]

    source = json.loads((dest / "meta" / core.SOURCE_MANIFEST_FILENAME).read_text())
    assert source["lake_uri"] == lake.uri
    assert source["view_id"] == published.view_id
    assert source["total_frames"] == 6
    assert {item["table"] for item in source["table_versions"]} == pinned_tables


def test_publish_view_is_idempotent_for_unchanged_lake_and_definition(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    first = _publish(lake)
    second = _publish(lake)
    assert first.view_id == second.view_id
    assert lake.table(VIEWS_TABLE).count_rows() == 1
    assert lake.table(VIEW_FILES_TABLE).count_rows() == first.file_count


def test_published_view_pins_survive_lake_advance(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    header = get_view(lake, view_id=published.view_id)

    before = open_published_facade(lake, header)
    frames_before = [before[i] for i in range(len(before))]

    _advance_lake(lake)
    live = LiveLeRobotFacade(lake, name="facade_view", mapping=_MAPPING)
    assert len(live) > 6  # the lake really advanced

    after = open_published_facade(lake, header)
    assert len(after) == 6
    for expected, got in zip(frames_before, [after[i] for i in range(len(after))], strict=True):
        assert got == expected  # byte-identical reads after the advance (0491)


def test_open_facade_guard_raises_for_legacy_unpinned_manifest_after_advance(tmp_path):
    from lancedb_robotics.lerobot_facade.manifest import write_dataset_manifest

    lake = _two_episode_lake(tmp_path / "robot.lance")
    root = tmp_path / "manifest"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        write_dataset_manifest(
            lake, root, repo_id="r", fps=20, mapping=_MAPPING, name="facade_view"
        )
    manifest = core.load_source_manifest(root)
    manifest.table_versions = []  # legacy shape: no pins recorded

    _advance_lake(lake)
    with pytest.raises(core.ManifestDriftError, match="advanced past"):
        core.open_facade(manifest, episodes=None)


def test_republish_after_advance_creates_new_view_and_keeps_old_reproducible(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    first = _publish(lake)
    _advance_lake(lake)
    second = _publish(lake)

    assert first.view_id != second.view_id
    assert second.total_frames > first.total_frames
    latest = get_view(lake, repo_id="acme/pick-place-v1")
    assert latest["view_id"] == second.view_id  # newest wins for repo_id
    assert [row["view_id"] for row in list_views(lake, repo_id="acme/pick-place-v1")] == [
        second.view_id,
        first.view_id,
    ]

    old = open_published_facade(lake, get_view(lake, view_id=first.view_id))
    assert len(old) == first.total_frames  # first view untouched by the republish


def test_pinned_lake_raises_typed_errors_for_missing_or_pruned_versions(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    with pytest.raises(StaleViewVersionError, match="no pinned version"):
        PinnedLake(lake, {"runs": 1}).table("episodes")
    with pytest.raises(StaleViewVersionError, match="checkout failed"):
        PinnedLake(lake, {"episodes": 999_999}).table("episodes")


def test_concurrent_materialization_converges_on_one_complete_cache_dir(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    header = get_view(lake, view_id=published.view_id)
    dest = tmp_path / "cache" / "view"

    errors: list[BaseException] = []

    def worker():
        try:
            materialize_view(lake, header, dest, lake_uri=lake.uri)
        except BaseException as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert (dest / "meta" / "info.json").exists()
    leftovers = [p for p in dest.parent.iterdir() if p.name != dest.name]
    assert leftovers == []  # every losing temp dir was cleaned up
    manifest = core.load_source_manifest(dest)
    facade = core.open_facade(manifest, episodes=None)
    assert len(facade) == 6


def test_materialize_view_verifies_content_hashes(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    header = dict(get_view(lake, view_id=published.view_id))

    # Corrupt one stored file row (simulates torn storage): sha check must refuse.
    files_table = lake.table(VIEW_FILES_TABLE)
    victim = f"{published.view_id}/meta/info.json"
    import pyarrow as pa

    from lancedb_robotics.schemas import LEROBOT_VIEW_FILES_SCHEMA

    corrupt = pa.Table.from_pylist(
        [
            {
                "file_id": victim,
                "view_id": published.view_id,
                "path": "meta/info.json",
                "content": b"{}",
                "sha256": hashlib.sha256(b"not-the-content").hexdigest(),
                "size_bytes": 2,
                "created_at": datetime.now(UTC),
            }
        ],
        schema=LEROBOT_VIEW_FILES_SCHEMA,
    )
    files_table.merge_insert("file_id").when_matched_update_all().when_not_matched_insert_all().execute(corrupt)

    with pytest.raises(Exception, match="verification failed"):
        materialize_view(lake, header, tmp_path / "materialized")
    assert not (tmp_path / "materialized").exists()


def test_two_views_with_different_selections_publish_different_correct_stats(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    both = _publish(lake)
    only_first = _publish(lake, repo_id="acme/pick-only", episode_ids=["ep-0"])
    assert both.view_id != only_first.view_id

    def stats_of(view_id):
        header = get_view(lake, view_id=view_id)
        dest = materialize_view(lake, header, tmp_path / f"m-{view_id}", lake_uri=lake.uri)
        return json.loads((dest / "meta" / "stats.json").read_text())

    stats_both, stats_first = stats_of(both.view_id), stats_of(only_first.view_id)
    # The fixture's state vectors are constant, so the per-view difference shows
    # up in the frame-indexed features; each is individually correct for
    # exactly the ticks its view resolves (0490 acceptance).
    assert stats_both["index"]["max"] == [5.0]
    assert stats_first["index"]["max"] == [2.0]
    assert stats_both["timestamp"]["mean"] != stats_first["timestamp"]["mean"]
    assert stats_first["episode_index"]["max"] == [0.0]


def test_localize_root_resolves_local_lake_and_manifest_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("LANCEDB_ROBOTICS_VIEW_CACHE", str(tmp_path / "view-cache"))
    lake_path = tmp_path / "robot.lance"
    lake = _two_episode_lake(lake_path)
    published = _publish(lake)

    # Case 2: root is the lake itself -> materialized per-view cache dir.
    resolved = core.localize_root("acme/pick-place-v1", str(lake_path))
    assert (resolved / "meta" / "info.json").exists()
    assert str(resolved).startswith(str(tmp_path / "view-cache"))
    assert published.view_id in resolved.name
    source = json.loads((resolved / "meta" / core.SOURCE_MANIFEST_FILENAME).read_text())
    assert source["lake_uri"] == str(lake_path)  # the client's own root, injected
    facade = core.open_facade(core.load_source_manifest(resolved), episodes=None)
    assert len(facade) == 6

    # revision selects an exact view id.
    assert core.localize_root("ignored", str(lake_path), revision=published.view_id) == resolved

    # file://<lake path> — the form lerobot's is_remote_uri routes through its
    # per-format probe for a local lake — resolves to the same published view.
    file_resolved = core.localize_root("acme/pick-place-v1", f"file://{lake_path}")
    assert (file_resolved / "meta" / "info.json").exists()
    file_source = json.loads(
        (file_resolved / "meta" / core.SOURCE_MANIFEST_FILENAME).read_text()
    )
    assert file_source["lake_uri"] == str(lake_path)  # normalized to a plain path

    # Case 1: an already-materialized manifest directory passes through.
    assert core.localize_root("acme/pick-place-v1", str(resolved)) == resolved

    # Case 3: neither a manifest dir nor a lake -> FileNotFoundError (probe loop).
    with pytest.raises(FileNotFoundError):
        core.localize_root("acme/pick-place-v1", str(tmp_path / "not-a-lake"))
    # A lake with no such published view also refuses, naming the fix.
    with pytest.raises(FileNotFoundError, match="publish"):
        core.localize_root("acme/unknown", str(lake_path))


def test_get_view_raises_view_not_found_with_remediation(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    with pytest.raises(ViewNotFoundError, match="view publish"):
        get_view(lake, repo_id="acme/never-published")


# ---------------------------------------------------------------------------
# CLI: `train view publish|list|materialize`
# ---------------------------------------------------------------------------


def test_train_view_cli_publish_list_materialize(tmp_path):
    from typer.testing import CliRunner

    from lancedb_robotics.cli import app

    runner = CliRunner()
    lake_path = tmp_path / "robot.lance"
    _two_episode_lake(lake_path)

    publish = runner.invoke(
        app,
        [
            "train", "view", "publish",
            "--lake", str(lake_path),
            "--repo-id", "acme/cli-view",
            "--fps", "20",
            "--name", "facade_view",
            "--state-stream", "/gps",
            "--state-stream", "/imu",
            "--action-stream", "/action",
            "--format", "json",
        ],
    )
    assert publish.exit_code == 0, publish.output
    payload = json.loads(publish.output)
    assert payload["total_frames"] == 6

    listing = runner.invoke(
        app, ["train", "view", "list", "--lake", str(lake_path), "--format", "json"]
    )
    assert listing.exit_code == 0, listing.output
    rows = json.loads(listing.output)
    assert [row["view_id"] for row in rows] == [payload["view_id"]]

    dest = tmp_path / "materialized"
    materialize = runner.invoke(
        app,
        [
            "train", "view", "materialize", str(dest),
            "--lake", str(lake_path),
            "--repo-id", "acme/cli-view",
        ],
    )
    assert materialize.exit_code == 0, materialize.output
    assert (dest / "meta" / "stats.json").exists()

    missing = runner.invoke(
        app,
        ["train", "view", "materialize", str(tmp_path / "x"), "--lake", str(lake_path),
         "--repo-id", "acme/never-published"],
    )
    assert missing.exit_code == 1
    assert "publish" in missing.output


# ---------------------------------------------------------------------------
# Guardrails from the scale review (SKILLS.md: pin every guardrail with a test)
# ---------------------------------------------------------------------------


def test_concurrent_same_definition_publishes_converge(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")

    results: list = []
    errors: list[BaseException] = []

    def worker():
        try:
            results.append(_publish(lake))
        except BaseException as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len({published.view_id for published in results}) == 1
    view_id = results[0].view_id
    # Truly simultaneous same-key merge_inserts can land benign byte-identical
    # duplicate rows (Lance treats them as non-conflicting appends); convergence
    # is semantic: reads dedupe by key and every consumer sees one view.
    headers = lake.table(VIEWS_TABLE).search().select(["view_id"]).to_arrow().to_pylist()
    assert {row["view_id"] for row in headers} == {view_id}
    file_rows = (
        lake.table(VIEW_FILES_TABLE).search().select(["file_id"]).to_arrow().to_pylist()
    )
    assert len({row["file_id"] for row in file_rows}) == results[0].file_count
    header = get_view(lake, view_id=view_id)
    assert header["file_count"] == results[0].file_count
    assert [row["view_id"] for row in list_views(lake)] == [view_id]  # deduped listing
    dest = materialize_view(lake, header, tmp_path / "after-race", lake_uri=lake.uri)
    assert (dest / "meta" / "stats.json").exists()


def test_header_definition_never_inlines_resolved_episode_ids(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    whole = _publish(lake)
    header = get_view(lake, view_id=whole.view_id)
    definition = json.loads(header["definition_json"])
    assert definition["episode_selection"] == {"kind": "all"}
    assert "ep-0" not in header["definition_json"]  # resolved ids live in the file row

    explicit = _publish(lake, repo_id="acme/explicit", episode_ids=["ep-0", "ep-1"])
    definition = json.loads(get_view(lake, view_id=explicit.view_id)["definition_json"])
    assert definition["episode_selection"]["kind"] == "explicit"
    assert definition["episode_selection"]["count"] == 2
    assert definition["episode_selection"]["ids"] == ["ep-0", "ep-1"]
    assert "ids_sha256" in definition["episode_selection"]


def test_list_views_projects_bounded_columns_only(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish(lake)
    rows = list_views(lake)
    assert rows and "definition_json" not in rows[0] and "stats_sampling_json" not in rows[0]
    # get_view point-reads the full row for the resolved view only.
    full = get_view(lake, repo_id="acme/pick-place-v1")
    assert "definition_json" in full and "table_versions" in full


def test_catalog_scan_guard_raises_loudly_when_exceeded(tmp_path, monkeypatch):
    from lancedb_robotics.lerobot_facade import views as views_mod

    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    _publish(lake, repo_id="acme/second")
    _publish(lake, repo_id="acme/third")

    monkeypatch.setattr(views_mod, "_MAX_CATALOG_SCAN_ROWS", 2)
    with pytest.raises(views_mod.ViewError, match="keyset"):
        list_views(lake)
    # The exact-view point read stays under the guard and keeps working.
    assert get_view(lake, view_id=published.view_id)["view_id"] == published.view_id


def test_publish_refuses_oversized_explicit_episode_selection(tmp_path, monkeypatch):
    from lancedb_robotics.lerobot_facade import manifest as manifest_mod

    lake = _two_episode_lake(tmp_path / "robot.lance")
    monkeypatch.setattr(manifest_mod, "EPISODE_IDS_INLINE_LIMIT", 1)
    from lancedb_robotics.lerobot_facade.views import ViewPublishError

    with pytest.raises(ViewPublishError, match="quality policy"):
        _publish(lake, episode_ids=["ep-0", "ep-1"])


def test_episode_index_chunks_into_multiple_bounded_files(tmp_path, monkeypatch):
    from lancedb_robotics.lerobot_facade import manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "EPISODES_ROWS_PER_FILE", 1)
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    dest = materialize_view(
        lake, get_view(lake, view_id=published.view_id), tmp_path / "m", lake_uri=lake.uri
    )
    files = sorted((dest / "meta" / "episodes" / "chunk-000").iterdir())
    assert [f.name for f in files] == ["file-00000.parquet", "file-00001.parquet"]

    # Concatenated in lexicographic file order, the chunks tile [0, total_frames)
    # exactly — the property the reader guard enforces.
    import pyarrow.parquet as pq

    rows = [row for f in files for row in pq.read_table(f).to_pylist()]
    core.validate_episode_tiling(
        np.array([row["dataset_from_index"] for row in rows]),
        np.array([row["dataset_to_index"] for row in rows]),
        published.total_frames,
    )
    facade = core.open_facade(core.load_source_manifest(dest), episodes=None)
    assert len(facade) == published.total_frames


class _FlakyTable:
    """merge_insert stub: raises the given errors in order, then succeeds."""

    def __init__(self, errors):
        self._errors = list(errors)
        self.commits = 0

    def merge_insert(self, key):
        return self

    def when_matched_update_all(self, *, where=None):
        return self

    def when_not_matched_insert_all(self):
        return self

    def execute(self, data):
        if self._errors:
            raise self._errors.pop(0)
        self.commits += 1


def test_merge_insert_retry_retries_only_commit_conflicts():
    import pyarrow as pa

    from lancedb_robotics.lerobot_facade.views import (
        ViewPublishError,
        _is_retryable_commit_conflict,
        _merge_insert_with_retry,
    )

    assert _is_retryable_commit_conflict(ValueError("Retryable commit conflict for version 7"))
    # Bare mentions of concurrency are NOT retryable (matches enrich.py's
    # reference classifier); retrying them would mask the original error.
    assert not _is_retryable_commit_conflict(ValueError("concurrent schema change"))

    data = pa.table({"file_id": ["a"]})
    flaky = _FlakyTable([ValueError("Retryable commit conflict for version 7")])
    _merge_insert_with_retry(flaky, "file_id", data)
    assert flaky.commits == 1

    with pytest.raises(RuntimeError, match="disk full"):
        _merge_insert_with_retry(_FlakyTable([RuntimeError("disk full")]), "file_id", data)

    exhausted = _FlakyTable([ValueError("commit conflict")] * 99)
    with pytest.raises(ViewPublishError, match="commit conflicts"):
        _merge_insert_with_retry(exhausted, "file_id", data)


def test_materialize_fetches_content_one_file_at_a_time(tmp_path):
    """Structural guard: the listing scan never projects `content`; bytes are
    point-read per file_id (peak memory = largest single file)."""
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    header = get_view(lake, view_id=published.view_id)

    select_calls: list[list[str]] = []
    real_table = lake.table

    class _SpyQuery:
        def __init__(self, query):
            self._query = query

        def select(self, columns):
            select_calls.append(list(columns))
            return _SpyQuery(self._query.select(columns))

        def __getattr__(self, name):
            attr = getattr(self._query, name)
            if callable(attr):
                def wrapped(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    return _SpyQuery(result) if type(result).__name__.endswith("Query") or hasattr(result, "select") and hasattr(result, "to_arrow") else result
                return wrapped
            return attr

    class _SpyTable:
        def __init__(self, table):
            self._table = table

        def search(self):
            return _SpyQuery(self._table.search())

        def __getattr__(self, name):
            return getattr(self._table, name)

    class _SpyLake:
        def __init__(self, lake):
            self._lake = lake

        def table(self, name):
            return _SpyTable(real_table(name))

        def __getattr__(self, name):
            return getattr(self._lake, name)

    materialize_view(_SpyLake(lake), header, tmp_path / "m", lake_uri=lake.uri)
    content_projections = [cols for cols in select_calls if "content" in cols]
    assert content_projections and all(cols == ["content"] for cols in content_projections)
    # The listing projection carries integrity columns only, never content.
    listing = [cols for cols in select_calls if "path" in cols]
    assert listing and all("content" not in cols for cols in listing)


def test_pinned_lake_marks_handles_and_blob_route_honors_the_pin(tmp_path, monkeypatch):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    versions = {row["table"]: row["version"] for row in
                __import__("lancedb_robotics.lerobot_facade.views", fromlist=["capture_table_versions"]).capture_table_versions(lake)}
    pinned = PinnedLake(lake, versions)
    handle = pinned.table("episodes")
    assert handle._lancedb_robotics_pinned_version == versions["episodes"]

    # The namespace-direct route must forward the pin as version=.
    from lancedb_robotics import blob as blob_mod

    captured: dict = {}

    monkeypatch.setattr(blob_mod.pylance_execution, "has_pylance_access", lambda spec: True)

    def fake_open_direct_dataset(spec, name, **kwargs):
        captured.update(kwargs, table=name)
        return "dataset"

    monkeypatch.setattr(blob_mod.pylance_execution, "open_direct_dataset", fake_open_direct_dataset)
    assert blob_mod._to_dataset(handle, object(), table_name="episodes") == "dataset"
    assert captured["version"] == versions["episodes"]

    captured.clear()
    unpinned_handle = lake.table("episodes")
    assert blob_mod._to_dataset(unpinned_handle, object(), table_name="episodes") == "dataset"
    assert "version" not in captured
