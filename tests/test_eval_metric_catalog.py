"""Eval metric catalog, indexes, and retention tests (backlog 0095)."""

import json
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest
from test_curate import NOW, _build_curation_lake
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.curate import CurationError
from lancedb_robotics.indexing import (
    PREDICATE_INDEX_COLUMNS_BY_TABLE,
    build_eval_metric_catalog_predicate_indexes,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import FEEDBACK_SCHEMA, MODEL_OUTPUTS_SCHEMA

runner = CliRunner()


def _snapshot(lake: Lake, name: str, scenario_ids: list[str]) -> None:
    lake.curate.workbench(scope=scenario_ids).snapshot(name=name, split_by="scenario")


def _import_metrics(
    lake: Lake,
    snapshot: str,
    *,
    evaluation_run: str,
    model_version: str = "policy-v2",
    training_run: str = "train-1",
    metrics: list[dict] | None = None,
):
    return lake.curate.feedback_from_eval(
        snapshot,
        training_run_id=training_run,
        model_version=model_version,
        evaluation_run_id=evaluation_run,
        regression_threshold=0.05,
        metrics=metrics
        or [
            {
                "metric": "success_rate",
                "output_type": "eval_success",
                "slice": "site_id=site-b|object_category=cup",
                "score": 0.45,
                "baseline_score": 0.70,
                "scenario_ids": ["scn-site-b-cup"],
            },
            {
                "metric": "success_rate",
                "output_type": "eval_success",
                "score": 0.80,
                "baseline_score": 0.75,
                "scenario_ids": ["scn-anchor", "scn-site-b-cup"],
            },
        ],
    )


def _add_non_eval_model_outputs(lake: Lake, count: int) -> None:
    lake.table("model_outputs").add(
        pa.Table.from_pylist(
            [
                {
                    "model_output_id": f"det-{index}",
                    "run_id": "run-a",
                    "observation_id": "obs-scn-anchor",
                    "scenario_id": "scn-anchor",
                    "dataset_id": "",
                    "model_version": "detector-v1",
                    "output_type": "detection",
                    "prediction": "cup",
                    "output_json": json.dumps({"boxes": [index]}),
                    "score": 0.9,
                    "producer_run_id": "",
                    "source": "inference",
                    "metadata": [],
                    "transform_id": "",
                    "created_at": NOW,
                }
                for index in range(count)
            ],
            schema=MODEL_OUTPUTS_SCHEMA,
        )
    )


def test_eval_metric_listing_reads_catalog_not_model_outputs(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _snapshot(lake, "candidate-b", ["scn-anchor", "scn-site-b-cup"])
    _add_non_eval_model_outputs(lake, 50)
    feedback = _import_metrics(lake, "candidate-b", evaluation_run="eval-1")

    listing = lake.curate.list_eval_metrics(snapshot="candidate-b")
    assert listing.total_count == 2
    assert not listing.truncated
    assert {entry.model_output_id for entry in listing.entries} == set(
        feedback.metric_output_ids
    )
    assert all(entry.state == "active" for entry in listing.entries)
    assert all(entry.evaluation_run_id == "eval-1" for entry in listing.entries)
    # Only feedback-loop metric rows are cataloged; the 50 detection outputs are not.
    assert lake.table("eval_metric_catalog").count_rows() == 2

    # The listing never touches model_outputs: deleting every source row still
    # lists the same catalog entries.
    lake.table("model_outputs").delete("model_output_id IS NOT NULL")
    relisted = lake.curate.list_eval_metrics(snapshot="candidate-b")
    assert relisted.total_count == 2
    assert {entry.metric for entry in relisted.entries} == {"success_rate"}


def test_eval_metric_filters_are_deterministic(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _snapshot(lake, "candidate-b", ["scn-anchor", "scn-site-b-cup"])
    _import_metrics(lake, "candidate-b", evaluation_run="eval-1")
    _import_metrics(lake, "candidate-b", evaluation_run="eval-2")
    _import_metrics(
        lake,
        "candidate-b",
        evaluation_run="eval-3",
        model_version="policy-v3",
        metrics=[
            {
                "metric": "success_rate",
                "slice": "site_id=site-b|object_category=cup",
                "score": 0.90,
                "baseline_score": 0.45,
                "scenario_ids": ["scn-site-b-cup"],
            }
        ],
    )

    sliced = lake.curate.list_eval_metrics(
        metric="success_rate",
        model_version="policy-v2",
        slice_label="site_id=site-b|object_category=cup",
    )
    assert sliced.total_count == 2
    assert [entry.evaluation_run_id for entry in sliced.entries] == ["eval-2", "eval-1"]
    assert sliced.entries[0].state == "active"
    assert sliced.entries[1].state == "superseded"
    assert sliced.entries[1].superseded_by == sliced.entries[0].model_output_id
    again = lake.curate.list_eval_metrics(
        metric="success_rate",
        model_version="policy-v2",
        slice_label="site_id=site-b|object_category=cup",
    )
    assert [entry.model_output_id for entry in again.entries] == [
        entry.model_output_id for entry in sliced.entries
    ]

    latest = lake.curate.list_eval_metrics(
        metric="success_rate",
        slice_label="site_id=site-b|object_category=cup",
        latest_only=True,
    )
    assert latest.total_count == 2  # one series per model version
    assert {entry.evaluation_run_id for entry in latest.entries} == {"eval-2", "eval-3"}

    regressed = lake.curate.list_eval_metrics(regressed_only=True, latest_only=True)
    assert all(entry.regressed for entry in regressed.entries)
    v3 = lake.curate.list_eval_metrics(model_version="policy-v3")
    assert v3.total_count == 1
    assert v3.entries[0].improvement == pytest.approx(0.45)
    assert v3.entries[0].slice_values == {
        "site_id": "site-b",
        "object_category": "cup",
    }


def test_eval_metric_staleness_fires_after_source_rows_advance(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _snapshot(lake, "candidate-b", ["scn-anchor", "scn-site-b-cup"])
    feedback = _import_metrics(lake, "candidate-b", evaluation_run="eval-1")
    metric_id = feedback.metric_output_ids[0]

    fresh = lake.curate.eval_metric_staleness(metric_id)
    assert fresh.stale is False
    assert fresh.advanced_tables == ()
    assert dict(fresh.recorded_table_versions) == dict(fresh.current_table_versions)

    _add_non_eval_model_outputs(lake, 1)
    lake.table("feedback").add(
        pa.Table.from_pylist(
            [
                {
                    "feedback_id": "fb-1",
                    "run_id": "run-b",
                    "scenario_id": "scn-site-b-cup",
                    "model_output_id": metric_id,
                    "feedback_type": "regression-review",
                    "severity": "high",
                    "notes": "investigate",
                    "source": "fixture",
                    "status": "open",
                    "metadata": [],
                    "transform_id": "",
                    "created_at": NOW,
                }
            ],
            schema=FEEDBACK_SCHEMA,
        )
    )
    stale = lake.curate.eval_metric_staleness(metric_id)
    assert stale.stale is True
    assert {item["table"] for item in stale.advanced_tables} == {
        "model_outputs",
        "feedback",
    }
    for item in stale.advanced_tables:
        assert item["current_version"] > item["recorded_version"]

    by_run = lake.curate.eval_metric_staleness("eval-1")
    assert by_run.stale is True
    assert by_run.evaluation_run_id == "eval-1"

    try:
        lake.curate.eval_metric_staleness("missing-token")
    except CurationError as exc:
        assert "missing-token" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected CurationError for unknown metric token")


def test_retention_dry_run_reports_and_apply_protects_promoted_evidence(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _snapshot(lake, "candidate-b", ["scn-anchor", "scn-site-b-cup"])
    protected_feedback = _import_metrics(lake, "candidate-b", evaluation_run="eval-1")
    _import_metrics(lake, "candidate-b", evaluation_run="eval-2")
    _import_metrics(lake, "candidate-b", evaluation_run="eval-3")
    lake.curate.promote_snapshot(
        "candidate-b",
        decision="promote",
        reason="eval-1 metrics reviewed and accepted",
        evaluation_run_id="eval-1",
        model_version="policy-v2",
        metrics=protected_feedback.report["metrics"],
    )

    cutoff = datetime.now(UTC) + timedelta(hours=1)
    dry = lake.curate.prune_eval_metrics(older_than=cutoff, dry_run=True)
    assert dry.dry_run is True
    assert set(dry.protected_ids) == set(protected_feedback.metric_output_ids)
    eval_2_ids = {
        entry.model_output_id
        for entry in lake.curate.list_eval_metrics(evaluation_run="eval-2").entries
    }
    assert set(dry.pruned_ids) == eval_2_ids
    assert len(dry.retained_ids) == 2  # newest import per series stays active
    # Dry run wrote nothing: every source row is still present.
    source_ids = {
        row["model_output_id"]
        for row in lake.table("model_outputs").to_arrow().to_pylist()
    }
    assert eval_2_ids <= source_ids
    assert set(protected_feedback.metric_output_ids) <= source_ids

    applied = lake.curate.prune_eval_metrics(older_than=cutoff)
    assert set(applied.pruned_ids) == eval_2_ids
    assert set(applied.protected_ids) == set(protected_feedback.metric_output_ids)
    assert applied.transform_id
    source_ids = {
        row["model_output_id"]
        for row in lake.table("model_outputs").to_arrow().to_pylist()
    }
    assert not (eval_2_ids & source_ids)
    assert set(protected_feedback.metric_output_ids) <= source_ids

    pruned = lake.curate.list_eval_metrics(state="pruned")
    assert {entry.model_output_id for entry in pruned.entries} == eval_2_ids
    assert all(not entry.source_available for entry in pruned.entries)
    assert all(entry.retention_policy["retain_latest"] == 1 for entry in pruned.entries)
    protected = lake.curate.list_eval_metrics(evaluation_run="eval-1")
    assert all(entry.state == "superseded" for entry in protected.entries)
    assert all(entry.source_available for entry in protected.entries)
    active = lake.curate.list_eval_metrics(state="active")
    assert {entry.evaluation_run_id for entry in active.entries} == {"eval-3"}


def test_sync_rebuilds_catalog_and_preserves_pruned_audit_rows(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _snapshot(lake, "candidate-b", ["scn-anchor", "scn-site-b-cup"])
    _import_metrics(lake, "candidate-b", evaluation_run="eval-1")
    _import_metrics(lake, "candidate-b", evaluation_run="eval-2")
    before = {
        entry.model_output_id: entry.state
        for entry in lake.curate.list_eval_metrics().entries
    }

    # Simulate a pre-0095 lake: metrics exist in model_outputs, catalog empty.
    lake.table("eval_metric_catalog").delete("model_output_id IS NOT NULL")
    assert lake.curate.list_eval_metrics().total_count == 0

    report = lake.curate.sync_eval_metric_catalog()
    assert report.cataloged == 4
    assert report.active == 2
    assert report.superseded == 2
    assert report.scanned_model_outputs == lake.table("model_outputs").count_rows()
    after = {
        entry.model_output_id: entry.state
        for entry in lake.curate.list_eval_metrics().entries
    }
    assert after == before
    if report.index_results:
        assert all(
            result["status"] in {"built", "already_present", "skipped"}
            for result in report.index_results
        )

    # Pruned audit rows survive a rebuild even though their source is gone.
    cutoff = datetime.now(UTC) + timedelta(hours=1)
    applied = lake.curate.prune_eval_metrics(older_than=cutoff)
    assert applied.pruned_ids
    resynced = lake.curate.sync_eval_metric_catalog()
    assert resynced.preserved_pruned == len(applied.pruned_ids)
    pruned = lake.curate.list_eval_metrics(state="pruned")
    assert {entry.model_output_id for entry in pruned.entries} == set(applied.pruned_ids)


def test_eval_metric_catalog_predicate_indexes_registered(tmp_path):
    assert "eval_metric_catalog" in PREDICATE_INDEX_COLUMNS_BY_TABLE
    indexed_columns = {
        column for column, _ in PREDICATE_INDEX_COLUMNS_BY_TABLE["eval_metric_catalog"]
    }
    assert {
        "series_key",
        "snapshot_name",
        "evaluation_run_id",
        "model_version",
        "metric",
        "slice_label",
        "state",
    } <= indexed_columns
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _snapshot(lake, "candidate-b", ["scn-anchor", "scn-site-b-cup"])
    _import_metrics(lake, "candidate-b", evaluation_run="eval-1")
    results = build_eval_metric_catalog_predicate_indexes(lake)
    assert results
    assert all(result.status in {"built", "already_present", "skipped"} for result in results)


def test_curate_cli_eval_metric_commands(tmp_path):
    lake_path = str(tmp_path / "robot.lance")
    lake = _build_curation_lake(lake_path)
    _snapshot(lake, "candidate-b", ["scn-anchor", "scn-site-b-cup"])
    _import_metrics(lake, "candidate-b", evaluation_run="eval-1")
    _import_metrics(lake, "candidate-b", evaluation_run="eval-2")

    listing = runner.invoke(
        app,
        [
            "curate",
            "eval-metrics",
            "--lake",
            lake_path,
            "--snapshot",
            "candidate-b",
            "--limit",
            "1",
            "--json",
        ],
    )
    assert listing.exit_code == 0, listing.output
    payload = json.loads(listing.output)
    assert payload["count"] == 4
    assert payload["preview_limit"] == 1
    assert payload["truncated"] is True
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["evaluation_run_id"] == "eval-2"
    metric_id = payload["entries"][0]["model_output_id"]

    human = runner.invoke(
        app,
        [
            "curate",
            "eval-metrics",
            "--lake",
            lake_path,
            "--snapshot",
            "candidate-b",
            "--latest-only",
        ],
    )
    assert human.exit_code == 0, human.output
    assert "metrics: 2" in human.output
    assert "showing: 2" in human.output

    staleness = runner.invoke(
        app,
        ["curate", "eval-metric-staleness", "--lake", lake_path, metric_id, "--json"],
    )
    assert staleness.exit_code == 0, staleness.output
    staleness_payload = json.loads(staleness.output)
    assert staleness_payload["model_output_id"] == metric_id
    assert staleness_payload["stale"] is False

    prune = runner.invoke(
        app,
        ["curate", "prune-eval-metrics", "--lake", lake_path, "--dry-run", "--json"],
    )
    assert prune.exit_code == 0, prune.output
    prune_payload = json.loads(prune.output)
    assert prune_payload["dry_run"] is True
    assert prune_payload["pruned_ids"] == []  # no --older-than-days: soft-retire only

    sync = runner.invoke(
        app,
        ["curate", "sync-eval-metrics", "--lake", lake_path, "--json"],
    )
    assert sync.exit_code == 0, sync.output
    sync_payload = json.loads(sync.output)
    assert sync_payload["cataloged"] == 4
    assert sync_payload["active"] == 2
