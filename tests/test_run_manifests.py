"""Training and evaluation run manifest tests (backlog 0062)."""

import json

import pytest

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import (
    model_artifact_lineage_id,
    snapshot_artifact_id,
    training_run_artifact_id,
)
from lancedb_robotics.lineage_hooks import LineageHookError, lineage_context_from_adapter
from lancedb_robotics.run_manifests import RunManifestError
from lancedb_robotics.scenarios import create_scenario_windows


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


def test_record_training_run_manifest_is_stable_and_pins_snapshot_versions(
    tmp_path,
    fixtures_dir,
):
    lake, snapshot = _manifest_lake(tmp_path, fixtures_dir)
    dataset = lake.training.dataset("demo-v1", shuffle=True, shuffle_seed=17)

    first = lake.training.record_run(
        dataset=dataset,
        code_ref="git:abc123",
        package_versions={"lancedb-robotics": "0.0.1"},
        environment={"container_image": "policy-train@sha256:abc"},
        hardware={"accelerator": "cpu"},
        runtime={"python": "3.11"},
        hyperparameters={"lr": 0.001, "batch_size": 8},
        random_seeds={"python": 17},
        external_refs={"mlflow_run_id": "mlflow-123"},
    )
    second = lake.training.record_run(
        dataset=dataset,
        code_ref="git:abc123",
        package_versions={"lancedb-robotics": "0.0.1"},
        environment={"container_image": "policy-train@sha256:abc"},
        hardware={"accelerator": "cpu"},
        runtime={"python": "3.11"},
        hyperparameters={"lr": 0.001, "batch_size": 8},
        random_seeds={"python": 17},
        external_refs={"mlflow_run_id": "mlflow-123"},
    )

    assert second.training_run_id == first.training_run_id
    assert second.manifest_digest == first.manifest_digest
    assert first.dataset_id == snapshot.dataset_id
    assert first.table_versions == tuple(
        {"table": table, "version": version, "tag": ""}
        for table, version in snapshot.table_versions
    )
    assert first.row_plan_id == dataset.row_plan.plan_id
    assert first.epoch_plan_id == dataset.epoch_plan.plan_id

    rows = lake.table("training_runs").to_arrow().to_pylist()
    assert len(rows) == 1
    assert rows[0]["training_run_id"] == first.training_run_id
    assert json.loads(rows[0]["hyperparameters_json"])["lr"] == 0.001
    assert {item["key"]: item["value"] for item in rows[0]["external_refs"]} == {
        "mlflow_run_id": "mlflow-123"
    }


def test_record_checkpoint_traces_to_training_snapshot_code_and_environment(
    tmp_path,
    fixtures_dir,
):
    lake, snapshot = _manifest_lake(tmp_path, fixtures_dir)
    training = lake.training.record_run(
        "demo-v1",
        code_ref="git:train-sha",
        environment={"image_digest": "sha256:trainer"},
        hyperparameters={"epochs": 2},
    )
    checkpoint = lake.training.record_checkpoint(
        training_run_id=training.training_run_id,
        artifact_uri="s3://models/policy.ckpt",
        checksum="sha256:checkpoint",
        aliases=["candidate", "policy@train-sha"],
        framework="torch",
        epoch=2,
        step=128,
        metrics={"train_loss": 0.12},
        external_refs={"wandb_run_id": "wandb-123"},
    )

    trace = lake.lineage.trace_checkpoint(checkpoint.model_artifact_id, where="topic = '/imu'", limit=1)

    assert trace.dataset_snapshot["dataset_id"] == snapshot.dataset_id
    assert trace.training_run["training_run_id"] == training.training_run_id
    assert trace.training_run["code_ref"] == "git:train-sha"
    assert trace.model_artifacts[0]["model_artifact_id"] == checkpoint.model_artifact_id
    assert trace.rows[0]["topic"] == "/imu"
    assert [(item["table"], item["version"]) for item in trace.table_versions] == list(
        snapshot.table_versions
    )

    lake.lineage.refresh_graph()
    graph = lake.lineage.trace(model_artifact_lineage_id(checkpoint.model_artifact_id))
    artifact_ids = {row["artifact_id"] for row in graph.artifacts}
    assert snapshot_artifact_id(snapshot.dataset_id) in artifact_ids
    assert training_run_artifact_id(training.training_run_id) in artifact_ids
    assert {"trained-on", "produced-model"} <= {row["edge_type"] for row in graph.edges}
    executions = {row["execution_id"]: row for row in graph.executions}
    training_execution = executions[f"lancedb-robotics:execution:{training.training_run_id}"]
    assert training_execution["code_ref"] == "git:train-sha"
    assert json.loads(training_execution["params_json"])["hyperparameters"]["epochs"] == 2
    assert json.loads(training_execution["environment_json"])["environment"] == {
        "image_digest": "sha256:trainer"
    }


def test_record_eval_metrics_and_retrieve_by_model_artifact_and_snapshot(
    tmp_path,
    fixtures_dir,
):
    lake, snapshot = _manifest_lake(tmp_path, fixtures_dir)
    training = lake.training.record_run("demo-v1", code_ref="git:train")
    checkpoint = lake.training.record_checkpoint(
        training_run_id=training.training_run_id,
        model_artifact_id="policy-checkpoint",
        artifact_uri="s3://models/policy.ckpt",
    )

    report = lake.eval.record_run(
        "demo-v1",
        model_artifact_id=checkpoint.model_artifact_id,
        metrics={"success_rate": 0.75},
        slice_metrics={"task=pick": {"success_rate": 0.5}},
        failure_outputs={"scenario_ids": ["scn-regression"]},
        code_ref="git:eval",
        external_refs={"mlflow_run_id": "eval-123"},
    )
    rows = lake.eval.runs(
        model_artifact_id=checkpoint.model_artifact_id,
        dataset_id=snapshot.dataset_id,
    )

    assert len(rows) == 1
    assert rows[0]["eval_run_id"] == report.eval_run_id
    assert json.loads(rows[0]["metrics_json"]) == {"success_rate": 0.75}
    assert json.loads(rows[0]["slice_metrics_json"]) == {
        "task=pick": {"success_rate": 0.5}
    }
    assert json.loads(rows[0]["failure_outputs_json"]) == {
        "scenario_ids": ["scn-regression"]
    }

    lake.lineage.refresh_graph()
    graph = lake.lineage.trace(model_artifact_lineage_id(checkpoint.model_artifact_id))
    downstream = lake.lineage.impact(model_artifact_lineage_id(checkpoint.model_artifact_id))
    assert "evaluated-on" not in {row["edge_type"] for row in graph.edges}
    assert "evaluated-model" in {row["edge_type"] for row in downstream.edges}


def test_lineage_context_is_recorded_on_training_checkpoint_and_eval(
    tmp_path,
    fixtures_dir,
):
    lake, _snapshot = _manifest_lake(tmp_path, fixtures_dir)
    context = {
        "provider": "airflow",
        "run_id": "dag-run-0068",
        "job_id": "train-task",
        "code_version": "git:lineage-context",
        "environment_digest": "sha256:context-env",
        "environment": {"container_image": "trainer@sha256:context"},
        "artifact_refs": [{"kind": "dataset", "uri": "s3://features/manifest.json"}],
    }

    training = lake.training.record_run("demo-v1", lineage_context=context)
    checkpoint = lake.training.record_checkpoint(
        training_run_id=training.training_run_id,
        model_artifact_id="policy-context",
        artifact_uri="s3://models/context.ckpt",
        lineage_context=context,
    )
    evaluation = lake.eval.record_run(
        "demo-v1",
        model_artifact_id=checkpoint.model_artifact_id,
        metrics={"success_rate": 1.0},
        lineage_context=context,
    )

    assert training.code_ref == "git:lineage-context"
    assert training.environment["container_image"] == "trainer@sha256:context"
    assert training.external_refs["airflow_run_id"] == "dag-run-0068"
    assert training.external_refs["external_job_id"] == "train-task"
    assert checkpoint.external_refs["airflow_run_id"] == "dag-run-0068"
    assert evaluation.external_refs["external_environment_digest"] == "sha256:context-env"

    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == training.transform_id
    )
    params = json.loads(transform["params"])
    assert params["lineage_context"]["external_run_id"] == "dag-run-0068"
    assert params["external_artifacts"][0]["uri"] == "s3://features/manifest.json"

    lake.lineage.refresh_graph()
    graph = lake.lineage.trace(training.training_run_id, kind="training-run")
    execution = next(
        row
        for row in graph.executions
        if row["execution_id"] == f"lancedb-robotics:execution:{training.training_run_id}"
    )
    metadata = {item["key"]: item["value"] for item in execution["metadata"]}
    assert metadata["airflow_run_id"] == "dag-run-0068"
    assert execution["code_ref"] == "git:lineage-context"


def test_missing_lineage_hook_adapter_names_extra_or_plugin():
    with pytest.raises(LineageHookError, match=r"optional extra/plugin 'definitely-missing-0068'"):
        lineage_context_from_adapter("definitely-missing-0068")


def test_manifest_validation_errors_do_not_partially_write(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    before = _counts(lake)

    with pytest.raises(RunManifestError, match="unknown dataset snapshot"):
        lake.training.record_run("missing")

    assert _counts(lake) == before

    lake, _snapshot = _manifest_lake(tmp_path / "with-data", fixtures_dir)
    before = _counts(lake)
    with pytest.raises(RunManifestError, match="unknown model_artifact_id"):
        lake.eval.record_run(
            "demo-v1",
            model_artifact_id="missing-model",
            metrics={"success_rate": 0.0},
        )
    assert _counts(lake) == before

    with pytest.raises(RunManifestError, match="unknown training_run_id"):
        lake.training.record_checkpoint(
            training_run_id="missing-run",
            artifact_uri="s3://models/missing.ckpt",
        )
    assert _counts(lake) == before


def _counts(lake):
    return {
        table: lake.table(table).count_rows()
        for table in (
            "training_runs",
            "model_artifacts",
            "evaluation_runs",
            "transform_runs",
        )
    }
