"""Indexed run-manifest query, paged listing, metric lookup, and retention (0100).

These cover the read/retention surfaces layered over the 0062 training/evaluation
manifests: bounded/indexed query on large synthetic histories, materialized
metric-key lookup, deterministic cursor pagination, lineage/feedback/snapshot
retention protection, and query behavior on a namespace-like backend that cannot
push down server-side queries.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.indexing import build_run_manifest_predicate_indexes, has_scalar_index
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.run_manifests import (
    RunManifestError,
    delete_manifest,
    list_training_runs,
    manifest_protection,
    plan_manifest_retention,
    query_eval_metrics,
    query_evaluation_runs,
    sync_evaluation_run_metrics,
)
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.schemas import EVALUATION_RUNS_SCHEMA, TRAINING_RUNS_SCHEMA

_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _manifest_lake(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    scenarios = sorted(
        lake.table("scenarios").to_arrow().to_pylist(),
        key=lambda row: (row["start_time_ns"], row["scenario_id"]),
    )
    manifest = create_snapshot(
        lake,
        name="demo-v1",
        tag="training-demo",
        scenario_ids=[row["scenario_id"] for row in scenarios],
    )
    return lake, manifest


def _eval_row(index: int, *, model_artifact_id: str, dataset_id: str) -> dict:
    return {
        "eval_run_id": f"eval-{index:06d}",
        "model_artifact_id": model_artifact_id,
        "training_run_id": f"train-{index % 7}",
        "dataset_id": dataset_id,
        "snapshot_name": dataset_id,
        "snapshot_tag": "",
        "table_versions": [],
        "metrics_json": json.dumps({"success_rate": (index % 100) / 100.0}),
        "slice_metrics_json": json.dumps({"night/rain": {"success_rate": (index % 50) / 100.0}}),
        "failure_outputs_json": "{}",
        "code_ref": "git:eval",
        "package_versions_json": "{}",
        "environment_json": "{}",
        "hardware_json": "{}",
        "runtime_json": "{}",
        "external_refs": [],
        "status": "completed",
        "manifest_digest": f"dig-{index}",
        "transform_id": f"tfm-eval-{index}",
        "created_by": "lancedb-robotics",
        "created_at": _BASE + timedelta(microseconds=index),
    }


def _training_row(index: int, *, dataset_id: str) -> dict:
    return {
        "training_run_id": f"train-{index:03d}",
        "dataset_id": dataset_id,
        "snapshot_name": dataset_id,
        "snapshot_tag": "",
        "table_versions": [],
        "row_plan_id": None,
        "epoch_plan_id": None,
        "projection_manifest_ids": [],
        "code_ref": "git:train",
        "package_versions_json": "{}",
        "environment_json": "{}",
        "hardware_json": "{}",
        "runtime_json": "{}",
        "hyperparameters_json": "{}",
        "random_seeds_json": "{}",
        "split_policy_json": "{}",
        "external_refs": [],
        "status": "completed",
        "manifest_digest": f"dig-{index}",
        "transform_id": f"tfm-train-{index}",
        "created_by": "lancedb-robotics",
        "created_at": _BASE + timedelta(microseconds=index),
    }


def test_query_evaluations_by_model_artifact_and_snapshot_uses_bounded_plan(tmp_path):
    lake = Lake.init(tmp_path / "scale.lance")
    total = 6000
    rows = [
        _eval_row(i, model_artifact_id=f"model-{i % 6}", dataset_id=f"dataset-{i % 6}")
        for i in range(total)
    ]
    lake.table("evaluation_runs").add(pa.Table.from_pylist(rows, schema=EVALUATION_RUNS_SCHEMA))

    results = build_run_manifest_predicate_indexes(lake, include_metrics=False)
    assert any(r.status == "built" and r.column == "model_artifact_id" for r in results)
    assert has_scalar_index(lake.table("evaluation_runs"), "model_artifact_id")

    result = query_evaluation_runs(lake, model_artifact_id="model-3", dataset_id="dataset-3")

    expected = total // 6
    assert len(result.rows) == expected
    assert all(row["model_artifact_id"] == "model-3" for row in result.rows)
    assert all(row["dataset_id"] == "dataset-3" for row in result.rows)
    # Bounded pushdown: the engine narrowed the scan to matching rows, not the
    # whole 6000-row history, and the plan reports it honestly.
    assert result.plan.bounded is True
    assert result.plan.scanned_rows == result.plan.matched_rows == expected
    assert result.plan.scanned_rows < total


def test_metric_key_lookup_reports_materialized_index(tmp_path, fixtures_dir):
    lake, _snapshot = _manifest_lake(tmp_path, fixtures_dir)
    training = lake.training.record_run("demo-v1", code_ref="git:train")
    checkpoint = lake.training.record_checkpoint(
        training_run_id=training.training_run_id,
        model_artifact_id="policy-checkpoint",
        artifact_uri="s3://models/policy.ckpt",
    )
    lake.eval.record_run(
        "demo-v1",
        model_artifact_id=checkpoint.model_artifact_id,
        metrics={"success_rate": 0.9},
        slice_metrics={
            "night/rain": {"success_rate": 0.6},
            "day/clear": {"success_rate": 0.95},
        },
    )

    # Inline emission (0100) materializes the metric surface on the eval write,
    # so a lookup pushes down to the indexed table.
    result = lake.eval.metrics(metric="success_rate")
    assert result.materialized is True
    assert result.plan is not None and result.plan.bounded is True
    keys = {row["metric_key"] for row in result.rows}
    assert {"success_rate", "night/rain.success_rate", "day/clear.success_rate"} <= keys

    # "all evaluations where night/rain.success_rate < 0.8" narrows by value range.
    below = lake.eval.metrics(metric="success_rate", max_score=0.8)
    below_keys = {row["metric_key"] for row in below.rows}
    assert "night/rain.success_rate" in below_keys
    assert "day/clear.success_rate" not in below_keys

    # Fallback: with no materialized surface the lookup still returns the right
    # rows by parsing the manifest JSON, and says so (materialized=False).
    lake.table("evaluation_run_metrics").delete("true")
    fallback = query_eval_metrics(lake, metric="success_rate", max_score=0.8)
    assert fallback.materialized is False
    assert {row["metric_key"] for row in fallback.rows} == below_keys

    # Rebuild is deterministic and restores the materialized path.
    report = sync_evaluation_run_metrics(lake)
    assert report.metric_rows == 3
    assert query_eval_metrics(lake, metric="success_rate", max_score=0.8).materialized is True


def test_retention_refuses_protected_checkpoint_unless_forced(tmp_path, fixtures_dir):
    lake, _snapshot = _manifest_lake(tmp_path, fixtures_dir)
    training = lake.training.record_run("demo-v1", code_ref="git:train")
    checkpoint = lake.training.record_checkpoint(
        training_run_id=training.training_run_id,
        model_artifact_id="policy-checkpoint",
        artifact_uri="s3://models/policy.ckpt",
    )
    lake.eval.record_run(
        "demo-v1",
        model_artifact_id=checkpoint.model_artifact_id,
        metrics={"success_rate": 0.75},
    )
    lake.lineage.refresh_graph()

    protection = manifest_protection(
        lake, kind="model-artifact", manifest_id=checkpoint.model_artifact_id
    )
    assert protection.protected is True
    assert "lineage" in protection.categories
    assert any(reason.startswith("evaluation-run:") for reason in protection.reasons)

    plan = plan_manifest_retention(lake, kinds=("model-artifact",))
    assert checkpoint.model_artifact_id in {item.manifest_id for item in plan.protected}
    assert checkpoint.model_artifact_id not in {item.manifest_id for item in plan.deletable}

    with pytest.raises(RunManifestError, match="protected"):
        delete_manifest(lake, kind="model-artifact", manifest_id=checkpoint.model_artifact_id)
    # Unchanged after a refused delete.
    assert lake.table("model_artifacts").count_rows() == 1

    forced = delete_manifest(
        lake, kind="model-artifact", manifest_id=checkpoint.model_artifact_id, force=True
    )
    assert forced["deleted"] is True and forced["forced"] is True
    assert lake.table("model_artifacts").count_rows() == 0


def test_retention_plan_lists_unreferenced_manifest_as_deletable(tmp_path):
    lake = Lake.init(tmp_path / "retention.lance")
    # Directly-inserted orphan run: no snapshot, no downstream, no lineage edges.
    lake.table("training_runs").add(
        pa.Table.from_pylist([_training_row(0, dataset_id="orphan-dataset")], schema=TRAINING_RUNS_SCHEMA)
    )

    plan = plan_manifest_retention(lake, kinds=("training-run",))
    assert {item.manifest_id for item in plan.deletable} == {"train-000"}
    assert plan.protected == ()

    result = delete_manifest(lake, kind="training-run", manifest_id="train-000")
    assert result["deleted"] is True and result["forced"] is False
    assert lake.table("training_runs").count_rows() == 0


def test_paged_manifest_listing_is_deterministic(tmp_path):
    lake = Lake.init(tmp_path / "paged.lance")
    count = 12
    rows = [_training_row(i, dataset_id="pg-dataset") for i in range(count)]
    lake.table("training_runs").add(pa.Table.from_pylist(rows, schema=TRAINING_RUNS_SCHEMA))
    expected_ids = [f"train-{i:03d}" for i in range(count)]

    # Walk every page via the continuation token.
    seen: list[str] = []
    token = None
    pages = 0
    while True:
        page = list_training_runs(lake, page_size=5, page_token=token, dataset_id="pg-dataset")
        pages += 1
        seen.extend(row["training_run_id"] for row in page.rows)
        assert page.total_count == count
        token = page.next_page_token
        if token is None:
            break
        assert pages <= count  # guard against a non-terminating cursor

    assert seen == expected_ids  # stable order, no gaps, no duplicates

    # Repeated identical calls are deterministic (same rows, same next token).
    first_a = list_training_runs(lake, page_size=5, dataset_id="pg-dataset")
    first_b = list_training_runs(lake, page_size=5, dataset_id="pg-dataset")
    assert [r["training_run_id"] for r in first_a.rows] == [r["training_run_id"] for r in first_b.rows]
    assert first_a.next_page_token == first_b.next_page_token

    # A token is bound to its query: reusing it under different filters is rejected.
    with pytest.raises(RunManifestError, match="does not match this query"):
        list_training_runs(
            lake, page_size=5, page_token=first_a.next_page_token, dataset_id="other-dataset"
        )


def test_manifest_query_matches_across_local_and_namespace_like_backend(tmp_path, monkeypatch):
    """A namespace backend without server-side query pushdown returns the same rows."""
    lake = Lake.init(tmp_path / "compat.lance")
    rows = [
        _eval_row(i, model_artifact_id=f"model-{i % 3}", dataset_id=f"dataset-{i % 3}")
        for i in range(60)
    ]
    lake.table("evaluation_runs").add(pa.Table.from_pylist(rows, schema=EVALUATION_RUNS_SCHEMA))

    local = query_evaluation_runs(lake, model_artifact_id="model-1", dataset_id="dataset-1")
    assert local.plan.bounded is True
    local_ids = [row["eval_run_id"] for row in local.rows]
    assert local_ids  # sanity: the filter matched something

    # Simulate a namespace/remote backend that cannot build a server-side query:
    # ``.search()`` raises, so the query falls back to a client-side scan.
    handle_type = type(lake.table("evaluation_runs"))

    def _no_server_side_search(self, *args, **kwargs):
        raise RuntimeError("server-side query not available on this backend")

    monkeypatch.setattr(handle_type, "search", _no_server_side_search)

    remote = query_evaluation_runs(lake, model_artifact_id="model-1", dataset_id="dataset-1")
    assert remote.plan.bounded is False  # fell back to the client-side path
    assert [row["eval_run_id"] for row in remote.rows] == local_ids  # identical results
