"""Epoch-permutation artifact lifecycle tests (backlog 0131).

Backlog 0077 persists a deterministic epoch order as an internal LanceDB
``(row_id, split_id)`` permutation table, deterministically named and reused.
0131 adds a lifecycle layer on top of those tables *without* changing the 0077
naming or ``lake.training.dataset(...)`` arguments:

* a durable catalog that records each internal permutation artifact with its
  owner row/epoch plan ids, snapshot id, table versions, seed, epoch, worker
  partition, created/last-used timestamps, use count, and retention policy;
* reuse-vs-newly-created reporting in dataset accounting;
* a cleanup API/CLI that drops unreferenced internal permutation tables while
  preserving tables referenced by retained training runs/reports (or pinned by
  a ``keep`` retention policy), and is safe to run repeatedly.

These tests are the test-first plan for 0131.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from test_native_training_dataset import _training_lake

import lancedb_robotics.training as training_mod
from lancedb_robotics.training import EPOCH_PERMUTATION_TABLE_PREFIX
from lancedb_robotics.training_permutation_catalog import (
    PERMUTATION_CATALOG_TABLE,
    PermutationArtifactCatalog,
    open_permutation_catalog,
)


@pytest.fixture
def lake(tmp_path):
    return _training_lake(tmp_path / "robot.lance", frame_count=6)


def _perm_tables(lake) -> set[str]:
    response = lake._db.list_tables()
    tables = getattr(response, "tables", response)
    return {
        str(name)
        for name in (tables or [])
        if str(name).startswith(EPOCH_PERMUTATION_TABLE_PREFIX)
    }


# ---------------------------------------------------------------------------
# Item 1: two identical shuffled datasets reuse one cataloged permutation
# artifact and update its last-used metadata.
# ---------------------------------------------------------------------------
def test_identical_datasets_reuse_one_cataloged_artifact(lake):
    first = lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=17, epoch=1
    )
    assert first.epoch_plan.backend.kind == training_mod.EPOCH_BACKEND_LANCEDB_PERMUTATION

    artifacts = lake.training.permutation_artifacts()
    assert len(artifacts) == 1
    record = artifacts[0]
    assert record["permutation_table"].startswith(EPOCH_PERMUTATION_TABLE_PREFIX)
    assert record["row_plan_id"] == first.row_plan.plan_id
    assert record["epoch_plan_id"] == first.epoch_plan.plan_id
    assert record["dataset_id"] == first.manifest.dataset_id
    assert record["shuffle_seed"] == 17
    assert record["epoch"] == 1
    assert record["use_count"] == 1
    assert record["row_count"] == len(first.epoch_plan.global_order)
    assert record["retention_policy"] == "auto"
    created_at = record["created_at"]

    # First build materialized the ordering table.
    assert first.manifest.accounting["permutation_artifact"]["reused"] is False

    # An identical dataset (same snapshot/seed/epoch) reuses the same table and
    # bumps the cataloged use count + last-used timestamp; it does not fork a
    # second permutation artifact.
    second = lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=17, epoch=1
    )
    assert second.manifest.accounting["permutation_artifact"]["reused"] is True

    artifacts = lake.training.permutation_artifacts()
    assert len(artifacts) == 1
    reused = artifacts[0]
    assert reused["permutation_table"] == record["permutation_table"]
    assert reused["use_count"] == 2
    assert reused["created_at"] == created_at
    assert reused["last_used_at"] >= created_at
    assert len(_perm_tables(lake)) == 1


# ---------------------------------------------------------------------------
# Item 2: cleanup keeps an artifact referenced by a recorded training run.
# ---------------------------------------------------------------------------
def test_cleanup_keeps_artifact_referenced_by_training_run(lake):
    dataset = lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=3, epoch=0
    )
    perm_table = dataset.epoch_plan.backend.permutation_table
    assert perm_table in _perm_tables(lake)

    # A retained training run pins this dataset's row/epoch plan.
    lake.training.record_run(dataset=dataset, status="completed")

    report = lake.training.cleanup_permutation_artifacts(dry_run=False)
    assert perm_table not in report["removed"]
    assert perm_table in report["kept"]
    assert report["removed_count"] == 0
    assert perm_table in _perm_tables(lake)
    # The catalog entry survives, too.
    assert lake.training.permutation_artifact(perm_table) is not None


# ---------------------------------------------------------------------------
# Item 3: cleanup removes an unreferenced internal permutation table.
# ---------------------------------------------------------------------------
def test_cleanup_removes_unreferenced_artifact(lake):
    dataset = lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=9, epoch=4
    )
    perm_table = dataset.epoch_plan.backend.permutation_table
    assert perm_table in _perm_tables(lake)
    assert lake.training.permutation_artifact(perm_table) is not None

    # No training run references it -> cleanup reclaims it.
    report = lake.training.cleanup_permutation_artifacts(dry_run=False)
    assert perm_table in report["removed"]
    assert report["removed_count"] == 1
    assert report["reclaimed_bytes"] >= 0
    assert perm_table not in _perm_tables(lake)
    assert lake.training.permutation_artifact(perm_table) is None

    # Idempotent: a second cleanup finds nothing left to remove.
    again = lake.training.cleanup_permutation_artifacts(dry_run=False)
    assert again["removed"] == []
    assert again["removed_count"] == 0


def test_cleanup_dry_run_reports_without_deleting(lake):
    dataset = lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=2, epoch=0
    )
    perm_table = dataset.epoch_plan.backend.permutation_table

    report = lake.training.cleanup_permutation_artifacts(dry_run=True)
    assert report["dry_run"] is True
    assert perm_table in report["removed"]
    # Dry run must not touch the physical table or the catalog entry.
    assert perm_table in _perm_tables(lake)
    assert lake.training.permutation_artifact(perm_table) is not None


def test_keep_retention_policy_preserves_unreferenced_artifact(lake):
    dataset = lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=6, epoch=0
    )
    perm_table = dataset.epoch_plan.backend.permutation_table

    catalog = open_permutation_catalog(lake)
    assert isinstance(catalog, PermutationArtifactCatalog)
    catalog.set_retention_policy(perm_table, "keep")

    report = lake.training.cleanup_permutation_artifacts(dry_run=False)
    assert perm_table in report["kept"]
    assert perm_table not in report["removed"]
    assert perm_table in _perm_tables(lake)


# ---------------------------------------------------------------------------
# Item 4: accounting reports internal permutation artifact count/bytes.
# ---------------------------------------------------------------------------
def test_accounting_reports_artifact_count_and_bytes(lake):
    assert lake.training.permutation_artifact_accounting()["artifact_count"] == 0

    lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=1, epoch=0
    )
    lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=2, epoch=0
    )

    accounting = lake.training.permutation_artifact_accounting()
    assert accounting["artifact_count"] == 2
    assert accounting["total_rows"] == 2 * len(range(6))
    assert accounting["estimated_bytes"] > 0
    assert accounting["by_backend"][training_mod.EPOCH_BACKEND_LANCEDB_PERMUTATION] == 2


def test_catalog_prefix_matches_training_constant():
    # The catalog keeps a local copy of the artifact prefix (no training.py import
    # cycle). Pin it equal to the source of truth so a future rename of the 0077
    # constant can't silently make the catalog's prefix scan miss every artifact.
    from lancedb_robotics.training_permutation_catalog import (
        EPOCH_PERMUTATION_TABLE_PREFIX as CATALOG_PREFIX,
    )

    assert CATALOG_PREFIX == EPOCH_PERMUTATION_TABLE_PREFIX


def test_catalog_table_is_not_itself_a_permutation_artifact(lake):
    lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=1, epoch=0
    )
    # The catalog's own table must not be mistaken for a cleanable permutation
    # artifact (prefix isolation).
    assert not PERMUTATION_CATALOG_TABLE.startswith(EPOCH_PERMUTATION_TABLE_PREFIX)
    tables = [r["permutation_table"] for r in lake.training.permutation_artifacts()]
    assert PERMUTATION_CATALOG_TABLE not in tables
    report = lake.training.cleanup_permutation_artifacts(dry_run=True)
    assert PERMUTATION_CATALOG_TABLE not in report["removed"]


def test_cli_permutation_artifacts_and_cleanup(tmp_path):
    import json

    from typer.testing import CliRunner

    from lancedb_robotics.cli.train import train_app

    lake_path = tmp_path / "robot.lance"
    built = _training_lake(lake_path, frame_count=6)
    built.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=8, epoch=0
    )
    runner = CliRunner()

    listing = runner.invoke(
        train_app, ["permutation", "artifacts", "--lake", str(lake_path), "--format", "json"]
    )
    assert listing.exit_code == 0, listing.stdout
    payload = json.loads(listing.stdout)
    assert payload["accounting"]["artifact_count"] == 1
    assert len(payload["artifacts"]) == 1

    # Dry-run is the default; it must not delete.
    dry = runner.invoke(
        train_app, ["permutation", "cleanup", "--lake", str(lake_path), "--format", "json"]
    )
    assert dry.exit_code == 0, dry.stdout
    dry_report = json.loads(dry.stdout)
    assert dry_report["dry_run"] is True
    assert dry_report["removed_count"] == 1

    # --apply reclaims the unreferenced table.
    applied = runner.invoke(
        train_app,
        ["permutation", "cleanup", "--lake", str(lake_path), "--apply", "--format", "json"],
    )
    assert applied.exit_code == 0, applied.stdout
    applied_report = json.loads(applied.stdout)
    assert applied_report["dry_run"] is False
    assert applied_report["removed_count"] == 1

    after = runner.invoke(
        train_app, ["permutation", "artifacts", "--lake", str(lake_path), "--format", "json"]
    )
    assert json.loads(after.stdout)["accounting"]["artifact_count"] == 0


def _make_orphan(lake, suffix: str) -> str:
    orphan = EPOCH_PERMUTATION_TABLE_PREFIX + suffix
    lake._db.create_table(
        orphan,
        data=pa.table(
            {
                "row_id": pa.array([0, 1, 2], type=pa.uint64()),
                "split_id": pa.array([0, 0, 0], type=pa.uint64()),
            }
        ),
        mode="create",
    )
    return orphan


def test_uncataloged_permutation_table_is_discoverable_and_conservatively_kept(lake):
    # A stray internal permutation table with no catalog metadata (e.g. left by a
    # pre-catalog writer, or by a crash/lost-race between the physical-table write
    # and the catalog write) is discovered by prefix but CANNOT be reference-
    # checked. Default cleanup keeps it (never risks dropping a table a retained
    # run references); only an explicit --include-uncataloged reclaims it.
    orphan = _make_orphan(lake, "deadbeefdeadbeef")
    artifacts = {r["permutation_table"]: r for r in lake.training.permutation_artifacts()}
    assert orphan in artifacts
    assert artifacts[orphan]["cataloged"] is False

    default_report = lake.training.cleanup_permutation_artifacts(dry_run=False)
    assert orphan not in default_report["removed"]
    assert orphan in default_report["skipped_uncataloged"]
    assert orphan in _perm_tables(lake)

    forced = lake.training.cleanup_permutation_artifacts(
        dry_run=False, include_uncataloged=True
    )
    assert orphan in forced["removed"]
    assert orphan not in _perm_tables(lake)


def test_catalog_write_survives_concurrent_upserts(lake):
    # The catalog is one shared write target: many workers/experiments upsert
    # concurrently. The bounded retry on retryable commit conflicts must converge
    # rather than silently drop a loser's row (SKILLS.md concurrent-writer rule).
    from datetime import UTC, datetime

    from lancedb_robotics.training_permutation_catalog import (
        LanceTablePermutationArtifactStore,
        _is_retryable_commit_conflict,
        build_permutation_artifact_record,
    )

    assert _is_retryable_commit_conflict(RuntimeError("Retryable commit conflict for version 3"))
    assert not _is_retryable_commit_conflict(RuntimeError("schema mismatch"))

    store = LanceTablePermutationArtifactStore(lake._db)
    calls = {"n": 0}

    class _FlakyTable:
        def __init__(self, inner):
            self._inner = inner

        def merge_insert(self, *a, **k):
            builder = self._inner.merge_insert(*a, **k)

            class _B:
                def when_matched_update_all(self):
                    builder.when_matched_update_all()
                    return self

                def when_not_matched_insert_all(self):
                    builder.when_not_matched_insert_all()
                    return self

                def execute(self, data):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        raise RuntimeError("Retryable commit conflict for version 1")
                    return builder.execute(data)

            return _B()

    orig_ensure = store._ensure_table
    store._ensure_table = lambda: _FlakyTable(orig_ensure())  # type: ignore[method-assign]

    dataset = lake.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=11, epoch=0
    )
    perm_table = dataset.epoch_plan.backend.permutation_table
    record = build_permutation_artifact_record(
        permutation_table=perm_table,
        permutation_ref=f"lancedb://{perm_table}",
        backend_kind="lancedb_permutation",
        permutation_source="materialized",
        row_plan_id="rp",
        epoch_plan_id="ep",
        dataset_id="ds",
        snapshot_name="demo-v1",
        table_versions=[],
        shuffle_seed=11,
        epoch=0,
        worker_id=0,
        num_workers=1,
        resume_from=0,
        row_count=3,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    store.put(record)  # first execute raises retryable -> retried -> converges
    assert calls["n"] >= 2
    assert store.get(perm_table) is not None


def test_uncataloged_permutation_table_reclaimed_when_forced(lake):
    orphan = _make_orphan(lake, "cafecafecafecafe")
    report = lake.training.cleanup_permutation_artifacts(
        dry_run=False, include_uncataloged=True
    )
    assert orphan in report["removed"]
    assert orphan not in _perm_tables(lake)
