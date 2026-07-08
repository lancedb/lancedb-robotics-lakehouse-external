"""Native LanceDB ``Permutation`` read-backend tests (backlog 0120).

These prove that a version-pinned robotics ``TrainingRowPlan`` / ``EpochPlan`` can be
backed by LanceDB's native ``lancedb.permutation.Permutation`` reader without losing
snapshot lineage, without forcing payload/blob materialization, and that aligned
multi-source tick plans fall back explicitly to the executor-backed path.
"""

import pytest
from test_native_training_dataset import _training_lake

import lancedb_robotics.training as training_mod
import lancedb_robotics.training_permutation as perm_mod
from lancedb_robotics.blob import PAYLOAD_BLOB_COLUMN
from lancedb_robotics.training import (
    AlignedTrainingTickPlan,
    EpochExecutionBackend,
    EpochPlan,
)
from lancedb_robotics.training_permutation import (
    AlignedPermutationUnsupportedError,
    NativePermutationPlan,
    NativePermutationUnavailableError,
    TorchColUnsupportedError,
    build_native_permutation_plan,
    native_permutation_capability,
)


@pytest.fixture
def lake(tmp_path):
    # A handful of frames so the epoch order is non-trivial after shuffling.
    return _training_lake(tmp_path / "robot.lance", frame_count=6)


def _expected_frame_order(dataset, *, indices=None):
    order = dataset.epoch_plan.sample_indices if indices is None else indices
    return [dataset.row_plan.frame_ids[index] for index in order]


def _read_observation_ids(plan):
    ids = []
    for batch in plan.reader().select_columns(["observation_id"]).with_format("arrow").iter(
        1024, skip_last_batch=False
    ):
        ids.extend(batch.column("observation_id").to_pylist())
    return ids


# ---------------------------------------------------------------------------
# Test-first plan item 1: identical observation ids for a fixed seed/epoch.
# ---------------------------------------------------------------------------
def test_native_permutation_plan_matches_row_plan_order(lake):
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=17,
        epoch=2,
    )
    plan = build_native_permutation_plan(
        lake,
        dataset.row_plan,
        dataset.epoch_plan,
        columns=["observation_id"],
        output_format="arrow",
    )

    assert isinstance(plan, NativePermutationPlan)
    assert plan.plan_kind == perm_mod.NATIVE_PERMUTATION_PLAN_KIND
    assert plan.base_table == "observations"
    assert plan.row_plan_id == dataset.row_plan.plan_id
    assert plan.epoch_plan_id == dataset.epoch_plan.plan_id
    assert plan.num_rows == len(dataset.epoch_plan.sample_indices)

    # The native reader yields the exact frame order of the epoch plan.
    expected = _expected_frame_order(dataset)
    assert _read_observation_ids(plan) == expected
    assert plan.equivalence["matches"] is True
    assert plan.equivalence["checked_rows"] >= 1


def test_native_permutation_plan_reuses_0077_permutation_table(lake):
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=5,
        epoch=0,
    )
    # 0077 selected its own permutation table for a single-worker global order.
    assert dataset.epoch_plan.backend.kind == training_mod.EPOCH_BACKEND_LANCEDB_PERMUTATION
    plan = build_native_permutation_plan(
        lake, dataset.row_plan, dataset.epoch_plan, columns=["observation_id"]
    )
    # 0120 reads through the *same* table 0077 built -- no duplicate ordering artifact.
    assert plan.permutation_source == "reused-0077-table"
    assert plan.permutation_table == dataset.epoch_plan.backend.permutation_table


# ---------------------------------------------------------------------------
# Test-first plan item 2: tensor-only projection does not read payload/blob.
# ---------------------------------------------------------------------------
def test_native_permutation_projection_excludes_payload_and_blob(lake):
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["state_vector", "action_vector"],
        shuffle=True,
        shuffle_seed=3,
    )
    plan = build_native_permutation_plan(
        lake,
        dataset.row_plan,
        dataset.epoch_plan,
        columns=["state_vector", "action_vector"],
        output_format="arrow",
    )
    assert plan.projection == ("state_vector", "action_vector")
    assert PAYLOAD_BLOB_COLUMN not in plan.projection
    assert "payload_json" not in plan.projection

    # The native reader only reads the projected columns (no blob column materialized).
    batch = next(iter(plan.reader().iter(1024, skip_last_batch=False)))
    assert batch.schema.names == ["state_vector", "action_vector"]
    assert PAYLOAD_BLOB_COLUMN not in batch.schema.names


# ---------------------------------------------------------------------------
# Test-first plan item 3: efficient (torch_col) format preserves order + lineage.
# ---------------------------------------------------------------------------
def test_native_permutation_torch_col_scalar_columns(lake):
    pytest.importorskip("torch")
    # timestamp_ns / raw_sequence are non-null scalar numeric columns -- torch_col also
    # requires no nulls (DLPack rejects them), which frame_index would violate here.
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["timestamp_ns", "raw_sequence"],
        shuffle=True,
        shuffle_seed=9,
    )
    plan = build_native_permutation_plan(
        lake,
        dataset.row_plan,
        dataset.epoch_plan,
        columns=["timestamp_ns", "raw_sequence"],
        output_format="torch_col",
    )
    assert plan.output_format == "torch_col"
    assert plan.projection == ("timestamp_ns", "raw_sequence")
    assert plan.torch_col["supported"] is True

    import torch

    tensor = plan.take([0, 1, 2])
    assert isinstance(tensor, torch.Tensor)
    # torch_col: first dim indexes columns -> column ordering is preserved.
    assert tensor.shape[0] == len(plan.projection)

    # Lineage travels with the plan even though the tensor batch carries no metadata.
    assert plan.lineage["snapshot_name"] == "demo-v1"
    assert plan.lineage["row_plan_id"] == dataset.row_plan.plan_id
    assert plan.lineage["table_versions"]
    assert plan.lineage["permutation_ref"].startswith("lancedb://")


def test_native_permutation_torch_col_rejects_list_columns(lake):
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["state_vector"],
        shuffle=True,
        shuffle_seed=1,
    )
    with pytest.raises(TorchColUnsupportedError) as excinfo:
        build_native_permutation_plan(
            lake,
            dataset.row_plan,
            dataset.epoch_plan,
            columns=["state_vector"],
            output_format="torch_col",
        )
    message = str(excinfo.value)
    assert "state_vector" in message
    assert "torch_col" in message


# ---------------------------------------------------------------------------
# Test-first plan item 4: aligned grouped-source falls back explicitly.
# ---------------------------------------------------------------------------
def _minimal_aligned_tick_plan():
    return AlignedTrainingTickPlan(
        plan_id="tickplan-test",
        alignment_id="align-test",
        alignment_name="policy_bridge",
        table_versions=({"table": "aligned_frames", "version": 1},),
        read_table_versions=({"table": "observations", "version": 1},),
        storage_backend="grain",
        schema_version="v1",
        streams=("camera", "state"),
        columns=("state_vector",),
        quality_policy={},
        scan={},
        tick_indices=(0, 1),
        aligned_frame_ids=(("a", "b"), ("c", "d")),
        source_row_ids=((0, 1), (2, 3)),
        row_ids=(None, None),
        total_ticks=2,
        selected_ticks=2,
        total_frames=4,
        selected_frames=4,
    )


def _minimal_epoch_plan(row_plan_id):
    backend = EpochExecutionBackend(
        kind=training_mod.EPOCH_BACKEND_PYTHON,
        execution_mode="python-in-memory",
        row_plan_id=row_plan_id,
        snapshot_id=None,
        table_versions=(),
        shuffle_seed=None,
        epoch=0,
        worker_id=0,
        num_workers=1,
        resume_from=0,
    )
    return EpochPlan(
        plan_id="epoch-test",
        row_plan_id=row_plan_id,
        shuffle=False,
        shuffle_seed=None,
        epoch=0,
        ordering_policy="snapshot-frame-order",
        global_order=(0, 1),
        worker_id=0,
        num_workers=1,
        worker_order=(0, 1),
        resume_from=0,
        sample_indices=(0, 1),
        backend=backend,
    )


def test_native_permutation_rejects_aligned_grouped_source(lake):
    tick_plan = _minimal_aligned_tick_plan()
    epoch_plan = _minimal_epoch_plan(tick_plan.plan_id)
    with pytest.raises(AlignedPermutationUnsupportedError):
        build_native_permutation_plan(lake, tick_plan, epoch_plan)


def test_aligned_dataset_stays_executor_backed(lake):
    # The aligned path is already executor-backed (enable_lancedb_backend=False);
    # the native single-table permutation simply does not apply to it.
    # No alignment fixture here -- assert the design invariant via the tick plan guard.
    tick_plan = _minimal_aligned_tick_plan()
    supported, reason = perm_mod.native_permutation_supported_for_plan(tick_plan)
    assert supported is False
    assert "aligned" in reason.lower()


# ---------------------------------------------------------------------------
# Capability probe.
# ---------------------------------------------------------------------------
def test_native_permutation_capability_reports_supported(lake):
    cap = lake.training.permutation_capability()
    assert cap["supported"] is True
    assert cap["execution_mode"] == perm_mod.NATIVE_PERMUTATION_EXECUTION_MODE
    assert "torch_col" in cap["output_formats"]
    assert isinstance(cap["torch_available"], bool)
    assert cap["torch_col_scalar_only"] is True


def test_native_permutation_unavailable_without_module(lake, monkeypatch):
    monkeypatch.setattr(perm_mod, "_native_permutation_module", lambda: None)
    cap = native_permutation_capability(lake)
    assert cap["supported"] is False

    dataset = lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=2
    )
    with pytest.raises(NativePermutationUnavailableError):
        build_native_permutation_plan(
            lake, dataset.row_plan, dataset.epoch_plan, columns=["observation_id"]
        )


# ---------------------------------------------------------------------------
# Resume-from suffix equivalence.
# ---------------------------------------------------------------------------
def test_native_permutation_resume_offset_matches_suffix(lake):
    full = lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=8, epoch=1
    )
    resumed = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=8,
        epoch=1,
        resume_from=2,
    )
    resumed_plan = build_native_permutation_plan(
        lake, resumed.row_plan, resumed.epoch_plan, columns=["observation_id"]
    )
    # Resuming at offset 2 yields the suffix of the fresh global epoch order.
    expected_suffix = _expected_frame_order(full, indices=full.epoch_plan.global_order[2:])
    assert _read_observation_ids(resumed_plan) == expected_suffix


# ---------------------------------------------------------------------------
# CLI surface.
# ---------------------------------------------------------------------------
def test_cli_permutation_capability_and_build(tmp_path):
    import json

    from typer.testing import CliRunner

    from lancedb_robotics.cli.train import train_app

    lake_path = tmp_path / "robot.lance"
    _training_lake(lake_path, frame_count=6)
    runner = CliRunner()

    cap = runner.invoke(
        train_app, ["permutation", "capability", "--lake", str(lake_path), "--format", "json"]
    )
    assert cap.exit_code == 0, cap.stdout
    assert json.loads(cap.stdout)["supported"] is True

    build = runner.invoke(
        train_app,
        [
            "permutation",
            "build",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
            "--columns",
            "state_vector,action_vector",
            "--read-format",
            "arrow",
            "--shuffle-seed",
            "4",
            "--format",
            "json",
        ],
    )
    assert build.exit_code == 0, build.stdout
    handle = json.loads(build.stdout)
    assert handle["plan_kind"] == perm_mod.NATIVE_PERMUTATION_PLAN_KIND
    assert handle["projection"] == ["state_vector", "action_vector"]
    assert handle["equivalence"]["matches"] is True
