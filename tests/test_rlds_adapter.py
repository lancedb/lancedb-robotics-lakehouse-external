"""RLDS/TFDS source adapter and canonical ingest tests (backlog 0258)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lancedb_robotics import ingest as ingest_module
from lancedb_robotics.adapters import AdapterError, get_adapter, list_adapters
from lancedb_robotics.adapters import rlds_adapter as rlds_module
from lancedb_robotics.adapters.rlds_adapter import (
    RldsFieldMapping,
    decode_rlds_observation_bundle,
)
from lancedb_robotics.blob import PAYLOAD_BLOB_COLUMN, fetch_blob
from lancedb_robotics.cli import app
from lancedb_robotics.ingest import ingest_rlds
from lancedb_robotics.lake import Lake
from lancedb_robotics.rlds_ingest import _rlds_ingest_claim

runner = CliRunner()


class FakeArray:
    def __init__(self, values, *, shape, dtype="uint8"):
        self._values = values
        self.shape = shape
        self.dtype = dtype
        self.size = 1
        for value in shape:
            self.size *= value

    def tobytes(self):
        return bytes(self._values)

    def tolist(self):
        return self._values


class FakeReadConfig:
    def __init__(
        self,
        *,
        add_tfds_id=False,
        skip_prefetch=False,
        try_autocache=True,
        interleave_cycle_length=None,
        interleave_block_length=None,
        num_parallel_calls_for_decode=None,
        num_parallel_calls_for_interleave_files=None,
    ):
        self.kwargs = {
            "add_tfds_id": add_tfds_id,
            "skip_prefetch": skip_prefetch,
            "try_autocache": try_autocache,
            "interleave_cycle_length": interleave_cycle_length,
            "interleave_block_length": interleave_block_length,
            "num_parallel_calls_for_decode": num_parallel_calls_for_decode,
            "num_parallel_calls_for_interleave_files": num_parallel_calls_for_interleave_files,
        }


class FakeReadInstruction:
    def __init__(self, split, *, from_, to, unit):
        self.split = split
        self.from_ = from_
        self.to = to
        self.unit = unit


class FakeSplitInfo:
    def __init__(self, filepaths, episode_count):
        self.filepaths = tuple(filepaths)
        self.num_shards = len(filepaths)
        self.shard_lengths = tuple(1 for _ in filepaths)
        self.file_instructions = tuple(
            SimpleNamespace(filename=path) for path in filepaths
        )
        self.num_examples = episode_count
        self.num_bytes = sum(Path(path).stat().st_size for path in filepaths)


class FakeBuilder:
    def __init__(self, root: Path, episodes_by_shard):
        filepaths = sorted(str(path) for path in root.glob("*.tfrecord-*"))
        self.info = SimpleNamespace(
            name="tiny_open_x",
            version="1.0.0",
            features={"steps": object(), "episode_metadata": object()},
            splits={
                "train": FakeSplitInfo(
                    filepaths,
                    sum(len(rows) for rows in episodes_by_shard.values()),
                )
            },
        )
        self.episodes_by_shard = episodes_by_shard
        self.read_calls = []

    def as_dataset(self, *, split, shuffle_files, read_config):
        assert isinstance(split, FakeReadInstruction)
        assert split.unit == "shard"
        assert split.to == split.from_ + 1
        assert shuffle_files is False
        self.read_calls.append((split.split, split.from_, read_config))
        return iter(self.episodes_by_shard.get(split.from_, ()))


class FakeTfds:
    __version__ = "4.9.10"
    ReadConfig = FakeReadConfig
    core = SimpleNamespace(ReadInstruction=FakeReadInstruction)

    def __init__(self, builder):
        self.builder = builder

    def builder_from_directory(self, builder_dir=None):
        assert Path(builder_dir).name
        return self.builder


def _prepared_fixture(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "dataset_info.json").write_text(
        json.dumps({"name": "tiny_open_x", "version": "1.0.0"}) + "\n"
    )
    (root / "features.json").write_text(json.dumps({"steps": "Dataset"}) + "\n")
    (root / "tiny_open_x-train.tfrecord-00000-of-00002").write_bytes(b"shard-zero")
    (root / "tiny_open_x-train.tfrecord-00001-of-00002").write_bytes(b"shard-one")
    return root


def _step(
    index: int,
    *,
    last: bool,
    terminal: bool | None,
    task: bytes,
    include_action: bool = True,
):
    row = {
        "observation": {
            "state": [float(index), float(index + 1)],
            "natural_language_instruction": task,
            "image": FakeArray([1, 2, 3, 4], shape=(2, 2)),
            "vendor_scalar": 7,
            "timestamp": index / 10,
        },
        "reward": float(index),
        "discount": 0.0 if last else 1.0,
        "is_first": index == 0,
        "is_last": last,
    }
    if terminal is not None:
        row["is_terminal"] = terminal
    if include_action:
        row["action"] = {
            "gripper": [0.25],
            "world_vector": [0.1, 0.2, 0.3],
        }
    return row


def _episodes():
    return {
        0: [
            {
                "tfds_id": b"unstable-upstream-id-0",
                "episode_metadata": {"source": "demo"},
                "steps": iter(
                    [
                        _step(0, last=False, terminal=False, task=b"pick the cube"),
                        _step(1, last=True, terminal=True, task=b"pick the cube"),
                    ]
                ),
            }
        ],
        1: [
            {
                "tfds_id": b"unstable-upstream-id-1",
                "steps": iter(
                    [
                        _step(
                            0,
                            last=True,
                            terminal=False,
                            task=b"place the cube",
                            include_action=False,
                        )
                    ]
                ),
            }
        ],
    }


@pytest.fixture
def fake_rlds(monkeypatch, tmp_path):
    source = _prepared_fixture(tmp_path / "prepared")
    builder = FakeBuilder(source, _episodes())
    fake_tfds = FakeTfds(builder)
    monkeypatch.setattr(rlds_module, "_load_tfds", lambda: fake_tfds)
    adapter = get_adapter("rlds")
    monkeypatch.setattr(
        adapter,
        "availability",
        lambda: {
            "available": True,
            "modules": ["tensorflow", "tensorflow_datasets"],
            "missing": [],
            "install": "lancedb-robotics[tfds]",
        },
    )
    return source, builder


def _lake(tmp_path) -> Lake:
    return Lake.init(tmp_path / "robot.lance")


def _rows(lake: Lake, table: str):
    return lake.table(table).to_arrow().to_pylist()


def test_registry_discovers_rlds_without_importing_tensorflow():
    info = get_adapter("rlds").info
    assert info.name == "rlds"
    assert info.format == "rlds"
    assert {"inspect", "ingest"} <= set(info.capabilities)
    assert "rlds" in [entry.name for entry in list_adapters()]


def test_missing_optional_dependencies_are_actionable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        get_adapter("rlds"),
        "availability",
        lambda: {
            "available": False,
            "missing": ["tensorflow", "tensorflow_datasets"],
            "install": "lancedb-robotics[tfds] (or lancedb-robotics[rlds])",
        },
    )
    with pytest.raises(AdapterError, match=r"lancedb-robotics\[tfds\]"):
        ingest_rlds(_lake(tmp_path), tmp_path / "anything")


def test_source_identity_is_content_addressed_and_path_portable(tmp_path):
    first = _prepared_fixture(tmp_path / "first")
    second = tmp_path / "second"
    shutil.copytree(first, second)
    adapter = get_adapter("rlds")
    first_source = adapter.source(first)
    second_source = adapter.source(second)
    assert first_source.checksum == second_source.checksum
    assert first_source.digest == second_source.digest
    assert first_source.uri != second_source.uri

    shard = second / "tiny_open_x-train.tfrecord-00001-of-00002"
    shard.write_bytes(b"changed!!")
    assert adapter.source(second).digest != first_source.digest


def test_custom_parquet_rlds_export_is_rejected_clearly(tmp_path):
    root = tmp_path / "custom-export"
    root.mkdir()
    (root / "dataset_info.json").write_text(
        json.dumps({"format_version": "rlds-tfds-style-v0"})
    )
    with pytest.raises(AdapterError, match="Parquet.*not a prepared TensorFlow"):
        get_adapter("rlds").source(root)


def test_remote_tfds_transport_boundary_is_explicit():
    with pytest.raises(AdapterError, match="Application Default Credentials"):
        rlds_module._validate_tfds_source_request(
            "gs://open-x/example/1.0.0",
            {"token": "must-not-be-forwarded"},
        )
    with pytest.raises(AdapterError, match="supported through.*gs://"):
        rlds_module._validate_tfds_source_request("s3://bucket/example/1.0.0", None)
    assert rlds_module._tfds_uri("gcs://bucket/example/1.0.0") == (
        "gs://bucket/example/1.0.0"
    )


def test_inspect_and_ingest_map_canonical_rows_shard_by_shard(fake_rlds, tmp_path):
    source, builder = fake_rlds
    adapter = get_adapter("rlds")
    report = adapter.inspect(source)
    assert report["dataset_name"] == "tiny_open_x"
    assert report["episode_count"] == 2
    assert report["shard_count"] == 2
    assert report["source_identity"] == {
        "kind": "metadata-only",
        "checksum": None,
        "artifact_count": None,
    }

    lake = _lake(tmp_path)
    result = ingest_rlds(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )
    assert result.rows_added["episodes"] == 2
    assert result.rows_added["observations"] == 3
    assert result.rows_added["scenarios"] == 2
    assert len(builder.read_calls) == 2
    assert [call[1] for call in builder.read_calls] == [0, 1]
    for _, _, read_config in builder.read_calls:
        assert read_config.kwargs == {
            "add_tfds_id": True,
            "skip_prefetch": True,
            "try_autocache": False,
            "interleave_cycle_length": 1,
            "interleave_block_length": 1,
            "num_parallel_calls_for_decode": 1,
            "num_parallel_calls_for_interleave_files": 1,
        }

    episodes = sorted(_rows(lake, "episodes"), key=lambda row: row["episode_index"])
    observations = sorted(
        _rows(lake, "observations"),
        key=lambda row: (row["episode_index"], row["frame_index"]),
    )
    assert [row["frame_count"] for row in episodes] == [2, 1]
    assert [row["boundary_source"] for row in episodes] == ["rlds-flags"] * 2
    assert [row["outcome"] for row in episodes] == ["terminal", "truncated"]
    assert observations[0]["state_vector"] == pytest.approx([0.0, 1.0])
    assert observations[0]["action_vector"] == pytest.approx([0.25, 0.1, 0.2, 0.3])
    assert observations[0]["task_id"] == "pick the cube"
    assert observations[2]["action_vector"] is None
    assert observations[2]["task_id"] == "place the cube"
    assert observations[0]["raw_uri"].endswith("tfrecord-00000-of-00002")
    assert observations[2]["raw_uri"].endswith("tfrecord-00001-of-00002")
    payload = json.loads(observations[0]["payload_json"])
    assert payload["split"] == "train"
    assert payload["shard_index"] == 0
    assert payload["reward"] == 0.0
    payload_blob = fetch_blob(
        lake.table("observations"),
        PAYLOAD_BLOB_COLUMN,
        observations[0]["observation_id"],
        id_column="observation_id",
    )
    assert payload_blob is not None
    decoded = decode_rlds_observation_bundle(payload_blob)
    assert decoded["format"] == "rlds-observation-bundle-v1"
    assert decoded["fields"]["observation.image"] == bytes([1, 2, 3, 4])


def test_standalone_inspect_does_not_read_or_hash_shards(fake_rlds, monkeypatch):
    source, _ = fake_rlds

    def reject_blob_read(*args, **kwargs):
        del args, kwargs
        raise AssertionError("metadata-only inspect must not open a shard")

    monkeypatch.setattr(rlds_module, "open_binary_uri", reject_blob_read)
    report = get_adapter("rlds").inspect(source)
    assert report["episode_count"] == 2
    assert report["source_identity"]["kind"] == "metadata-only"


def test_reingest_relocated_source_is_a_noop(fake_rlds, tmp_path, monkeypatch):
    source, builder = fake_rlds
    lake = _lake(tmp_path)
    first = ingest_rlds(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )
    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    relocated_builder = FakeBuilder(relocated, _episodes())
    monkeypatch.setattr(rlds_module, "_load_tfds", lambda: FakeTfds(relocated_builder))
    second = ingest_rlds(
        lake,
        relocated,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )
    assert second.already_ingested is True
    assert second.run_id == first.run_id
    assert lake.table("runs").count_rows() == 1
    assert lake.table("observations").count_rows() == 3
    assert relocated_builder.read_calls == []


def test_effective_split_selection_has_one_canonical_identity(fake_rlds, tmp_path):
    source, builder = fake_rlds
    lake = _lake(tmp_path)
    implicit = ingest_rlds(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    for selected in (("train",), ("train", "train")):
        builder.episodes_by_shard = _episodes()
        duplicate = ingest_rlds(
            lake,
            source,
            splits=selected,
            compact=False,
            prune_versions=False,
            index_predicates=False,
        )
        assert duplicate.already_ingested is True
        assert duplicate.run_id == implicit.run_id

    assert lake.table("runs").count_rows() == 1
    assert lake.table("observations").count_rows() == 3


def test_run_identity_includes_mapping_contract(fake_rlds, tmp_path):
    source, builder = fake_rlds
    lake = _lake(tmp_path)
    first = ingest_rlds(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )
    builder.episodes_by_shard = _episodes()
    alternate_mapping = RldsFieldMapping(action_key="optional_action_not_present")
    second = ingest_rlds(
        lake,
        source,
        mapping=alternate_mapping,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )
    assert second.already_ingested is False
    assert second.run_id != first.run_id
    assert lake.table("integration_sources").count_rows() == 1
    assert lake.table("runs").count_rows() == 2
    assert lake.table("observations").count_rows() == 6

    duplicate = ingest_rlds(
        lake,
        source,
        mapping=alternate_mapping,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )
    assert duplicate.already_ingested is True
    assert duplicate.run_id == second.run_id


@pytest.mark.parametrize(
    "steps, message",
    [
        (
            [{**_step(0, last=True, terminal=True, task=b"bad"), "is_first": False}],
            "must begin with is_first=True",
        ),
        (
            [
                _step(0, last=False, terminal=False, task=b"bad"),
                {**_step(1, last=False, terminal=True, task=b"bad"), "is_first": False},
            ],
            "is_last=True on the final step",
        ),
    ],
)
def test_malformed_markers_fail_and_cleanup(fake_rlds, tmp_path, steps, message):
    source, builder = fake_rlds
    builder.episodes_by_shard = {0: [{"steps": iter(steps)}], 1: []}
    lake = _lake(tmp_path)
    with pytest.raises(AdapterError, match=message):
        ingest_rlds(
            lake,
            source,
            batch_size=1,
            compact=False,
            prune_versions=False,
            index_predicates=False,
        )
    assert lake.table("observations").count_rows() == 0
    assert lake.table("episodes").count_rows() == 0
    assert lake.table("runs").count_rows() == 0


def test_partial_step_failure_cleans_up_and_retry_converges(fake_rlds, tmp_path):
    source, builder = fake_rlds

    def broken_steps():
        yield _step(0, last=False, terminal=False, task=b"retry")
        yield _step(1, last=False, terminal=False, task=b"retry")
        raise AdapterError("injected shard failure")

    builder.episodes_by_shard = {0: [{"steps": broken_steps()}], 1: []}
    lake = _lake(tmp_path)
    with pytest.raises(AdapterError, match="injected shard failure"):
        ingest_rlds(
            lake,
            source,
            batch_size=1,
            compact=False,
            prune_versions=False,
            index_predicates=False,
        )
    assert lake.table("observations").count_rows() == 0

    builder.episodes_by_shard = {
        0: [
            {
                "steps": iter(
                    [
                        _step(0, last=False, terminal=False, task=b"retry"),
                        _step(1, last=True, terminal=True, task=b"retry"),
                    ]
                )
            }
        ],
        1: [],
    }
    result = ingest_rlds(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )
    assert result.rows_added["observations"] == 2
    ids = [row["observation_id"] for row in _rows(lake, "observations")]
    assert len(ids) == len(set(ids)) == 2


def test_finalize_failure_cleans_completion_rows_and_retry_converges(
    fake_rlds, tmp_path, monkeypatch
):
    source, builder = fake_rlds
    lake = _lake(tmp_path)
    original_finalize = ingest_module._finalize_ingest

    def fail_finalize(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected finalize failure")

    monkeypatch.setattr(ingest_module, "_finalize_ingest", fail_finalize)
    with pytest.raises(RuntimeError, match="injected finalize failure"):
        ingest_rlds(lake, source, compact=False, prune_versions=False)
    for table_name in (
        "runs",
        "observations",
        "episodes",
        "scenarios",
        "events",
        "transform_runs",
    ):
        assert lake.table(table_name).count_rows() == 0

    monkeypatch.setattr(ingest_module, "_finalize_ingest", original_finalize)
    builder.episodes_by_shard = _episodes()
    retried = ingest_rlds(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )
    assert retried.rows_added["observations"] == 3
    assert lake.table("runs").count_rows() == 1


def test_in_process_concurrent_ingests_converge(fake_rlds, tmp_path, monkeypatch):
    source, _ = fake_rlds
    lake = _lake(tmp_path)
    # Each adapter pass needs fresh nested iterators.  The in-process claim lock
    # ensures only the winner consumes them; the second caller observes the run.
    builders = []

    def load_tfds():
        builder = FakeBuilder(source, _episodes())
        builders.append(builder)
        return FakeTfds(builder)

    monkeypatch.setattr(rlds_module, "_load_tfds", load_tfds)
    results = []
    errors = []

    def run():
        try:
            results.append(
                ingest_rlds(
                    lake,
                    source,
                    compact=False,
                    prune_versions=False,
                    index_predicates=False,
                )
            )
        except Exception as exc:  # noqa: BLE001 - thread evidence is asserted below.
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sorted(result.already_ingested for result in results) == [False, True]
    ids = [row["observation_id"] for row in _rows(lake, "observations")]
    assert len(ids) == len(set(ids)) == 3


def test_lake_resident_claim_blocks_a_second_connection_before_writes(tmp_path):
    lake_path = tmp_path / "claims.lance"
    first = Lake.init(lake_path)
    second = Lake.open(lake_path)

    with _rlds_ingest_claim(first, run_id="run-first", claimed_by="worker-a"):
        with pytest.raises(AdapterError, match="already holds the lake-wide write claim"):
            with _rlds_ingest_claim(
                second,
                run_id="run-second",
                claimed_by="worker-b",
            ):
                raise AssertionError("a second connection must not enter the write section")

    gate = first.table("rlds_ingest_claims").to_arrow().to_pylist()
    assert len(gate) == 1
    assert gate[0]["owner_token"] is None
    assert gate[0]["run_id"] is None


def test_corrupt_duplicate_claim_rows_fail_before_claiming(tmp_path):
    lake = Lake.init(tmp_path / "duplicate-claims.lance")
    claims = lake.table("rlds_ingest_claims")
    claims.add(claims.to_arrow())

    with pytest.raises(AdapterError, match="must contain exactly one `global` gate row"):
        with _rlds_ingest_claim(lake, run_id="run-test", claimed_by="worker"):
            raise AssertionError("corrupt coordination must not enter the write section")

    assert claims.count_rows("owner_token IS NOT NULL") == 0


def test_release_failure_does_not_mask_body_failure(tmp_path, monkeypatch):
    lake = Lake.init(tmp_path / "release-failure.lance")
    claims = lake.table("rlds_ingest_claims")
    original_update = claims.update
    update_calls = 0

    def fail_release(*args, **kwargs):
        nonlocal update_calls
        update_calls += 1
        if update_calls == 2:
            raise RuntimeError("injected release failure")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(claims, "update", fail_release)
    original_table = lake.table
    monkeypatch.setattr(
        lake,
        "table",
        lambda name: claims if name == "rlds_ingest_claims" else original_table(name),
    )

    with pytest.raises(ValueError, match="original ingest failure"):
        with _rlds_ingest_claim(lake, run_id="run-test", claimed_by="worker"):
            raise ValueError("original ingest failure")


def test_cli_ingest_rlds_uses_mapping_and_reports_rows(fake_rlds, tmp_path):
    source, _ = fake_rlds
    lake_path = tmp_path / "cli.lance"
    Lake.init(lake_path)
    result = runner.invoke(
        app,
        [
            "ingest",
            "rlds",
            str(source),
            "--lake",
            str(lake_path),
            "--split",
            "train",
            "--batch-size",
            "1",
            "--no-compact",
            "--no-prune-versions",
            "--no-index-predicates",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "episodes +2" in result.output
    assert "observations +3" in result.output
    assert "rlds.steps\t3" in result.output


def test_cli_inspect_rlds_text(fake_rlds):
    source, _ = fake_rlds
    result = runner.invoke(app, ["inspect", "rlds", str(source), "--format", "text"])
    assert result.exit_code == 0, result.output
    assert "RLDS/TFDS tiny_open_x@1.0.0: 2 episodes, 2 shards" in result.output


@pytest.mark.rlds_native
def test_native_prepared_tfds_dataset_ingests(tmp_path):
    """Exercise a real tfds.features.Dataset nested-step TFRecord fixture."""
    try:
        import numpy as np
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except Exception as exc:  # noqa: BLE001 - optional native stack is host-specific.
        reason = f"TFDS ingest stack unavailable; install lancedb-robotics[tfds] ({exc})"
        if os.getenv("LANCEDB_ROBOTICS_REQUIRE_RLDS_NATIVE") == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    class TinyRldsBuilder(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")

        def _info(self):
            return self.dataset_info_from_configs(
                disable_shuffling=True,
                features=tfds.features.FeaturesDict(
                    {
                        "episode_metadata": {"episode_id": tfds.features.Text()},
                        "steps": tfds.features.Dataset(
                            {
                                "observation": {
                                    "state": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                                    "image": tfds.features.Image(shape=(2, 2, 3)),
                                    "natural_language_instruction": tfds.features.Text(),
                                    "timestamp": np.float64,
                                },
                                "action": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                                "is_first": np.bool_,
                                "is_last": np.bool_,
                                "is_terminal": np.bool_,
                            }
                        ),
                    }
                )
            )

        def _split_generators(self, dl_manager):
            del dl_manager
            return {"train": self._generate_examples()}

        def _generate_examples(self):
            image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
            for episode_index, length in enumerate((2, 1)):
                steps = []
                for step_index in range(length):
                    steps.append(
                        {
                            "observation": {
                                "state": np.array(
                                    [episode_index, step_index], dtype=np.float32
                                ),
                                "image": image,
                                "natural_language_instruction": "pick the cube",
                                "timestamp": float(step_index) / 10,
                            },
                            "action": np.array([0.1, 0.2], dtype=np.float32),
                            "is_first": step_index == 0,
                            "is_last": step_index == length - 1,
                            "is_terminal": step_index == length - 1,
                        }
                    )
                yield episode_index, {
                    "episode_metadata": {"episode_id": f"native-{episode_index}"},
                    "steps": steps,
                }

    # A builder declared in a pytest module has no importable package resource
    # root. Point TFDS at the isolated fixture directory explicitly.
    TinyRldsBuilder.pkg_dir_path = tmp_path
    builder = TinyRldsBuilder(data_dir=tmp_path / "tfds-data")
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(num_shards=2, try_download_gcs=False)
    )
    prepared = Path(builder.data_dir)
    assert tfds.builder_from_directory(prepared).info.splits["train"].num_shards == 2

    lake_path = tmp_path / "native.lance"
    ingest_script = (
        "import sys\n"
        "import tensorflow as tf\n"
        "from lancedb_robotics.ingest import ingest_rlds\n"
        "from lancedb_robotics.lake import Lake\n"
        "lake = Lake.init(sys.argv[2])\n"
        "report = ingest_rlds(lake, sys.argv[1], split='train', compact=False, "
        "prune_versions=False, index_predicates=False)\n"
        "assert report.rows_added['episodes'] == 2\n"
        "assert report.rows_added['observations'] == 3\n"
        "assert tf.executing_eagerly()\n"
    )
    child_env = dict(os.environ)
    child_env["TFDS_DISABLE_GCS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", ingest_script, str(prepared), str(lake_path)],
        capture_output=True,
        check=False,
        env=child_env,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    lake = Lake.open(lake_path)
    rows = sorted(
        lake.table("observations").to_arrow().to_pylist(),
        key=lambda row: (row["episode_index"], row["frame_index"]),
    )
    assert rows[0]["state_vector"] == pytest.approx([0.0, 0.0])
    assert rows[0]["action_vector"] == pytest.approx([0.1, 0.2])
    assert rows[0]["task_id"] == "pick the cube"
    assert rows[0]["raw_uri"].endswith(".tfrecord-00000-of-00002")
    assert tf.executing_eagerly()
