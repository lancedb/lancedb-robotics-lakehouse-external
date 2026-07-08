"""External lineage and metadata-system integration tests (backlog 0064)."""

import json

import pytest

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage_integrations import (
    LineageIntegrationError,
    artifact_id_from_external_urn,
)
from lancedb_robotics.scenarios import create_scenario_windows


class _FakeOpenLineageClient:
    def __init__(self, *, fail_first: bool = False):
        self.fail_first = fail_first
        self.calls = 0
        self.events = []

    def emit(self, payload):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("temporary marquez outage")
        self.events.append(payload)
        return {"remote_id": payload["run"]["runId"]}


class _FakeDataHubClient:
    def __init__(self):
        self.edges = []

    def emit(self, payload):
        self.edges.append(payload)
        return {"urn": payload["downstreamUrn"], "remote_id": f"datahub-{len(self.edges)}"}


def _pipeline_lake(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    scenarios = sorted(
        lake.table("scenarios").to_arrow().to_pylist(),
        key=lambda row: (row["start_time_ns"], row["scenario_id"]),
    )
    snapshot = create_snapshot(
        lake,
        name="ol-demo",
        tag="openlineage-demo",
        scenario_ids=[row["scenario_id"] for row in scenarios],
    )
    training = lake.training.record_run(
        "ol-demo",
        code_ref="git:trainer",
        hyperparameters={"lr": 0.001},
    )
    return lake, snapshot, training


def test_openlineage_export_emits_core_events_and_lance_facets(tmp_path, fixtures_dir):
    lake, snapshot, training = _pipeline_lake(tmp_path, fixtures_dir)

    report = lake.lineage.export_openlineage()

    assert report.dry_run is True
    assert report.events
    event_kinds = {
        event["run"]["facets"]["lancedb_robotics_execution"]["kind"]
        for event in report.events
    }
    assert {"ingest", "dataset-snapshot", "training-run"} <= event_kinds

    training_event = next(
        event
        for event in report.events
        if event["run"]["facets"]["lancedb_robotics_execution"]["execution_id"]
        == f"lancedb-robotics:execution:{training.training_run_id}"
    )
    assert training_event["eventType"] == "COMPLETE"
    assert training_event["job"]["namespace"] == "lancedb-robotics"
    assert training_event["inputs"]
    assert training_event["outputs"]
    dataset_facets = [
        dataset["facets"]["lancedb_robotics_artifact"]
        for dataset in training_event["inputs"] + training_event["outputs"]
    ]
    assert {facet["artifact_id"] for facet in dataset_facets} >= {
        f"lancedb-robotics:snapshot:{snapshot.dataset_id}",
        f"lancedb-robotics:training-run:{training.training_run_id}",
    }
    assert all(facet["artifact_urn"].startswith("urn:lancedb-robotics:artifact:") for facet in dataset_facets)


def test_external_urn_round_trips_to_canonical_artifact_id(tmp_path, fixtures_dir):
    lake, snapshot, _training = _pipeline_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    artifact_id = f"lancedb-robotics:snapshot:{snapshot.dataset_id}"

    openlineage_urn = lake.lineage.external_urn(artifact_id)
    datahub_urn = lake.lineage.external_urn(artifact_id, backend="datahub")

    assert artifact_id_from_external_urn(openlineage_urn) == artifact_id
    assert artifact_id_from_external_urn(datahub_urn) == artifact_id
    assert lake.lineage.resolve_external_urn(openlineage_urn) == artifact_id
    assert lake.lineage.resolve_external_urn(datahub_urn) == artifact_id


def test_attach_external_refs_updates_training_model_and_eval_manifests(
    tmp_path,
    fixtures_dir,
):
    lake, _snapshot, training = _pipeline_lake(tmp_path, fixtures_dir)
    checkpoint = lake.training.record_checkpoint(
        training_run_id=training.training_run_id,
        model_artifact_id="policy-openlineage",
        artifact_uri="s3://models/policy.ckpt",
    )
    evaluation = lake.eval.record_run(
        "ol-demo",
        model_artifact_id=checkpoint.model_artifact_id,
        metrics={"success_rate": 0.5},
    )

    training_refs = lake.training.attach_external_refs(
        training.training_run_id,
        {"mlflow_run_id": "mlflow-0064"},
    )
    model_refs = lake.training.attach_model_external_refs(
        checkpoint.model_artifact_id,
        {"wandb_artifact_id": "wandb-artifact-0064"},
    )
    eval_refs = lake.eval.attach_external_refs(
        evaluation.eval_run_id,
        {"wandb_run_id": "wandb-0064"},
    )

    assert training_refs.external_refs["mlflow_run_id"] == "mlflow-0064"
    assert model_refs.external_refs["wandb_artifact_id"] == "wandb-artifact-0064"
    assert eval_refs.external_refs["wandb_run_id"] == "wandb-0064"

    lake.lineage.refresh_graph()
    graph = lake.lineage.trace(training.training_run_id, kind="training-run")
    execution = next(
        row
        for row in graph.executions
        if row["execution_id"] == f"lancedb-robotics:execution:{training.training_run_id}"
    )
    assert {item["key"]: item["value"] for item in execution["metadata"]}["mlflow_run_id"] == "mlflow-0064"


def test_datahub_export_uses_same_lineage_edges_and_round_trip_urns(tmp_path, fixtures_dir):
    lake, snapshot, training = _pipeline_lake(tmp_path, fixtures_dir)

    report = lake.lineage.export_datahub()

    assert report.edges
    assert "trained-on" in {edge["edge_type"] for edge in report.edges}
    training_edge = next(
        edge
        for edge in report.edges
        if edge["edge_type"] == "trained-on"
        and edge["downstream"]["artifact_id"]
        == f"lancedb-robotics:training-run:{training.training_run_id}"
    )
    assert training_edge["upstream"]["artifact_id"] == f"lancedb-robotics:snapshot:{snapshot.dataset_id}"
    assert artifact_id_from_external_urn(training_edge["upstreamUrn"]) == training_edge["upstream"]["artifact_id"]
    assert artifact_id_from_external_urn(training_edge["downstreamUrn"]) == training_edge["downstream"]["artifact_id"]


def test_missing_optional_adapter_names_extra_or_plugin(tmp_path):
    with pytest.raises(LineageIntegrationError, match=r"optional extra/plugin 'definitely-missing-0064'"):
        Lake.init(tmp_path / "robot.lance").lineage.require_integration_adapter(
            "definitely-missing-0064"
        )


def test_openlineage_emit_records_attempts_and_retry_skips_delivered_payloads(
    tmp_path,
    fixtures_dir,
):
    lake, _snapshot, _training = _pipeline_lake(tmp_path, fixtures_dir)
    client = _FakeOpenLineageClient(fail_first=True)

    first = lake.lineage.emit_openlineage(client=client, target="fake-marquez")
    retry = lake.lineage.retry_lineage_delivery(
        "openlineage",
        client=client,
        target="fake-marquez",
    )

    expected_events = list(lake.lineage.export_openlineage(refresh=False).events)
    event_ids = {
        event["run"]["facets"]["lancedb_robotics_execution"]["execution_id"]
        for event in client.events
    }
    expected_event_ids = {
        event["run"]["facets"]["lancedb_robotics_execution"]["execution_id"]
        for event in expected_events
    }
    assert event_ids == expected_event_ids
    assert first.status == "partial"
    assert first.failed_count == 1
    assert retry.delivered_count == 1
    assert retry.already_delivered_count == len(expected_events) - 1

    persisted = lake.lineage.lineage_delivery_attempts(
        backend="openlineage",
        target="fake-marquez",
    )
    delivered = [attempt for attempt in persisted if attempt.status == "delivered"]
    failed = [attempt for attempt in persisted if attempt.status == "failed"]
    assert len(delivered) == len(expected_events)
    assert len(failed) == 1
    assert all(attempt.remote_response_ids for attempt in delivered)


def test_datahub_emit_records_response_ids_and_payload_digests(tmp_path, fixtures_dir):
    lake, _snapshot, _training = _pipeline_lake(tmp_path, fixtures_dir)
    client = _FakeDataHubClient()

    report = lake.lineage.emit_datahub(client=client, target="fake-datahub")

    assert report.status == "delivered"
    assert client.edges
    assert all(edge["upstreamUrn"].startswith("urn:li:dataset:") for edge in client.edges)
    assert all(edge["downstreamUrn"].startswith("urn:li:dataset:") for edge in client.edges)

    attempts = lake.lineage.lineage_delivery_attempts(
        backend="datahub",
        target="fake-datahub",
    )
    assert len(attempts) == len(client.edges)
    assert {attempt.payload_digest for attempt in attempts}
    assert all(attempt.payload_digest.startswith("sha256:") for attempt in attempts)
    assert all(attempt.remote_response_ids for attempt in attempts)


def test_emit_missing_dependency_names_optional_extra_and_adapter(tmp_path, fixtures_dir):
    lake, _snapshot, _training = _pipeline_lake(tmp_path, fixtures_dir)

    with pytest.raises(
        LineageIntegrationError,
        match=r"optional extra/plugin 'definitely-missing-0104'",
    ):
        lake.lineage.emit_openlineage(
            adapter="definitely-missing-0104",
            target="missing",
        )


def test_export_reports_are_json_ready(tmp_path, fixtures_dir):
    lake, _snapshot, _training = _pipeline_lake(tmp_path, fixtures_dir)

    payload = lake.lineage.export_openlineage().to_dict()

    assert json.loads(json.dumps(payload))["event_count"] == len(payload["events"])
