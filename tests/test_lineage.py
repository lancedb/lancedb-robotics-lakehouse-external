"""Regression tracing tests (backlog 0033 / 0065)."""

import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from mcap.reader import make_reader

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.enrich import enrich_scenarios
from lancedb_robotics.indexing import IndexSpec
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import (
    LINEAGE_REPORT_VERSION,
    artifact_id,
    evaluation_run_artifact_id,
    model_artifact_lineage_id,
    snapshot_artifact_id,
    table_version_artifact_id,
    training_run_artifact_id,
)
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.writeback import ingest_model_outputs, record_feedback


def _trace_lake(tmp_path, fixtures_dir, fixture_name="sample.mcap"):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / fixture_name)
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
    ingest_model_outputs(
        lake,
        {
            "model_output_id": "out-regression",
            "observation_id": scenarios[0]["observation_ids"][0],
            "scenario_id": scenarios[0]["scenario_id"],
            "dataset_id": manifest.dataset_id,
            "model_version": "policy@abc123",
            "prediction": "regressed",
            "score": 0.12,
            "producer_run_id": "checkpoint-abc123",
        },
        source="trainer",
    )
    return lake, manifest


def test_trace_checkpoint_returns_snapshot_versions_slice_and_source_logs(tmp_path, fixtures_dir):
    lake, manifest = _trace_lake(tmp_path, fixtures_dir)

    trace = lake.lineage.trace_checkpoint("checkpoint-abc123", where="topic = '/imu'")

    assert trace.dataset_snapshot["dataset_id"] == manifest.dataset_id
    assert [(item["table"], item["version"]) for item in trace.table_versions] == list(
        manifest.table_versions
    )
    assert {row["topic"] for row in trace.rows} == {"/imu"}
    assert {row["dataset_id"] for row in trace.rows} == {manifest.dataset_id}
    assert trace.source_logs
    assert trace.to_source_logs() == tuple(coord.as_tuple() for coord in trace.source_logs)

    coord = trace.source_logs[0]
    uri, channel, offset = coord.as_tuple()
    messages = _messages_by_channel(_local_path(uri), channel)
    assert offset is not None
    assert messages[offset] == (channel, coord.log_time_ns)


def test_trace_checkpoint_resolves_snapshot_from_dataset_tag_metadata(tmp_path, fixtures_dir):
    lake, manifest = _trace_lake(tmp_path, fixtures_dir)
    scenario = lake.table("scenarios").to_arrow().to_pylist()[0]

    ingest_model_outputs(
        lake,
        {
            "model_output_id": "out-tagged",
            "observation_id": scenario["observation_ids"][0],
            "scenario_id": scenario["scenario_id"],
            "model_version": "policy@tagged",
            "prediction": "regressed",
            "producer_run_id": "checkpoint-tagged",
            "metadata": {"dataset_tag": manifest.tag},
        },
        source="trainer",
    )

    trace = lake.lineage.trace_checkpoint("checkpoint-tagged", limit=1)

    assert trace.dataset_snapshot["dataset_id"] == manifest.dataset_id
    assert len(trace.rows) == 1


def test_trace_checkpoint_is_read_only_and_deterministic(tmp_path, fixtures_dir):
    lake, _manifest = _trace_lake(tmp_path, fixtures_dir)
    before = _table_versions_and_counts(lake)

    first = lake.lineage.trace_checkpoint("checkpoint-abc123", where="topic IN ('/imu')")
    second = lake.lineage.trace_checkpoint("checkpoint-abc123", where="topic IN ('/imu')")

    assert first.as_dict() == second.as_dict()
    assert _table_versions_and_counts(lake) == before


def test_trace_evidence_pack_plan_is_deterministic_and_metadata_only(tmp_path, fixtures_dir):
    lake, manifest = _trace_lake(tmp_path, fixtures_dir)
    trace = lake.lineage.trace_checkpoint("checkpoint-abc123", where="topic = '/imu'", limit=2)
    before = _table_versions_and_counts(lake)

    first = trace.evidence_pack()
    second = trace.evidence_pack()

    assert first.mode == "plan"
    assert first.bytes_copied == 0
    assert first.files == ()
    assert first.manifest_digest == second.manifest_digest
    assert first.manifest == second.manifest
    assert first.manifest["schema_version"] == "lancedb-robotics/evidence-pack/v1"
    assert first.manifest["subject"]["model_run_id"] == "checkpoint-abc123"
    assert first.manifest["dataset_snapshot"]["dataset_id"] == manifest.dataset_id
    assert first.manifest["source_coordinates"]
    assert first.manifest["payload_refs"]
    assert first.manifest["rows"]["model_outputs"][0]["model_output_id"] == "out-regression"
    assert {
        (item["table"], item["version"]) for item in first.manifest["table_versions"]
    } >= set(manifest.table_versions)
    assert _table_versions_and_counts(lake) == before


def test_evidence_pack_materializes_blob_hashes_deterministically(tmp_path, fixtures_dir):
    lake, _manifest = _trace_lake(tmp_path, fixtures_dir, fixture_name="records.mcap")
    trace = lake.lineage.trace_checkpoint("checkpoint-abc123", limit=1)

    first = trace.evidence_pack(
        output_dir=tmp_path / "evidence-a",
        materialize=True,
        include_attachments=True,
    )
    second = trace.evidence_pack(
        output_dir=tmp_path / "evidence-b",
        materialize=True,
        include_attachments=True,
    )

    assert first.mode == "materialized"
    assert first.bytes_copied > 0
    assert first.manifest_digest == second.manifest_digest
    assert Path(first.manifest_path).name == "manifest.json"
    assert json.loads(Path(first.manifest_path).read_text()) == first.manifest
    payload_file = first.files[0]
    assert payload_file["kind"] == "attachment"
    materialized = tmp_path / "evidence-a" / payload_file["path"]
    assert materialized.exists()
    assert payload_file["bytes"] == materialized.stat().st_size
    assert payload_file["sha256"] == _sha256(materialized.read_bytes())


def test_evidence_pack_from_feedback_event_lists_writeback_and_sources(tmp_path, fixtures_dir):
    lake, _manifest = _trace_lake(tmp_path, fixtures_dir)
    record_feedback(
        lake,
        {
            "feedback_id": "fb-regression",
            "model_output_id": "out-regression",
            "feedback_type": "field_failure",
            "severity": "high",
            "linked_incident_id": "inc-0065",
        },
        source="fleet-ops",
    )

    pack = lake.lineage.evidence_pack("fb-regression", kind="feedback")
    manifest = pack.manifest

    assert pack.mode == "plan"
    assert pack.bytes_copied == 0
    assert manifest["subject"]["handle"] == "fb-regression"
    assert manifest["rows"]["feedback"][0]["feedback_id"] == "fb-regression"
    assert manifest["rows"]["model_outputs"][0]["model_output_id"] == "out-regression"
    assert manifest["source_coordinates"]
    assert manifest["verification"]["source_coordinate_hashes"]


def test_refresh_graph_projects_snapshot_model_output_and_source_lineage(
    tmp_path, fixtures_dir
):
    lake, manifest = _trace_lake(tmp_path, fixtures_dir)

    report = lake.lineage.refresh_graph()

    assert report.artifacts > 0
    assert report.executions > 0
    assert report.edges > 0
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)
    model_output_id = artifact_id(
        "row",
        table_name="model_outputs",
        row_id="out-regression",
        table_version=int(lake.table("model_outputs").version),
    )

    artifact_rows = lake.table("lineage_artifacts").to_arrow().to_pylist()
    assert snapshot_id in {row["artifact_id"] for row in artifact_rows}
    assert model_output_id in {row["artifact_id"] for row in artifact_rows}
    assert "source" in {row["kind"] for row in artifact_rows}

    upstream_snapshot = lake.lineage.trace(snapshot_id)
    assert {"selected-from", "source-coordinate", "version-pinned"} <= {
        row["edge_type"] for row in upstream_snapshot.edges
    }

    upstream_model_output = lake.lineage.trace(model_output_id)
    assert snapshot_id in {row["artifact_id"] for row in upstream_model_output.artifacts}

    downstream_snapshot = lake.lineage.impact(snapshot_id)
    assert model_output_id in {row["artifact_id"] for row in downstream_snapshot.artifacts}
    assert "evaluated-on" in {row["edge_type"] for row in downstream_snapshot.edges}


def test_trace_resolves_snapshot_tag_and_filters_target_kinds_without_writes(
    tmp_path, fixtures_dir
):
    lake, manifest = _trace_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    before = _table_versions_and_counts(lake)

    graph = lake.lineage.trace(
        manifest.tag,
        kind="snapshot",
        target_kinds=("source",),
        edge_types=("source-coordinate",),
    )

    assert graph.resolved_handle == manifest.tag
    assert graph.root_artifact_ids == (snapshot_artifact_id(manifest.dataset_id),)
    assert {row["kind"] for row in graph.artifacts} == {"dataset-snapshot", "source"}
    assert {row["edge_type"] for row in graph.edges} == {"source-coordinate"}
    assert _table_versions_and_counts(lake) == before


def test_impact_resolves_raw_uri_and_run_id_to_downstream_artifacts(tmp_path, fixtures_dir):
    lake, manifest = _trace_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    run = lake.table("runs").to_arrow().to_pylist()[0]

    source_impact = lake.lineage.impact(
        run["raw_uri"],
        kind="source",
        target_kinds=("dataset-snapshot", "model-output"),
    )
    assert snapshot_artifact_id(manifest.dataset_id) in {
        row["artifact_id"] for row in source_impact.artifacts
    }
    assert {
        row["edge_type"]
        for row in source_impact.edges
    } & {"source-coordinate", "selected-from", "evaluated-on"}

    run_impact = lake.lineage.impact(
        run["run_id"],
        kind="run",
        target_kinds=("scenario", "dataset-snapshot"),
    )
    assert "windowed-run" in {row["edge_type"] for row in run_impact.edges}
    assert snapshot_artifact_id(manifest.dataset_id) in {
        row["artifact_id"] for row in run_impact.artifacts
    }


def test_rebuild_plan_records_source_invalidation_and_orders_downstream_actions(
    tmp_path, fixtures_dir
):
    lake, manifest = _trace_lake(tmp_path, fixtures_dir)
    training = lake.training.record_run("demo-v1", training_run_id="train-source-0066")
    checkpoint = lake.training.record_checkpoint(
        training_run_id=training.training_run_id,
        model_artifact_id="policy-source-0066",
        artifact_uri="s3://models/policy-source-0066.ckpt",
    )
    evaluation = lake.eval.record_run(
        "demo-v1",
        model_artifact_id=checkpoint.model_artifact_id,
        eval_run_id="eval-source-0066",
        metrics={"success_rate": 0.5},
    )
    lake.lineage.refresh_graph()
    run = lake.table("runs").to_arrow().to_pylist()[0]

    plan = lake.lineage.rebuild_plan(
        run["raw_uri"],
        kind="source",
        reason="camera calibration bug",
        severity="high",
        discovered_by="qa",
        actor="robotics-ops",
        record_invalidation=True,
    )
    payload = plan.as_dict()

    actions = {item["artifact_id"]: item for item in payload["actions"]}
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)
    training_id = training_run_artifact_id(training.training_run_id)
    model_id = model_artifact_lineage_id(checkpoint.model_artifact_id)
    eval_id = evaluation_run_artifact_id(evaluation.eval_run_id)
    model_output_id = artifact_id(
        "row",
        table_name="model_outputs",
        row_id="out-regression",
        table_version=int(lake.table("model_outputs").version),
    )

    assert payload["invalidation"]["reason"] == "camera calibration bug"
    assert payload["invalidation"]["severity"] == "high"
    assert actions[snapshot_id]["action"] == "resnapshot"
    assert actions[training_id]["action"] == "retrain"
    assert actions[model_id]["action"] == "retrain"
    assert actions[eval_id]["action"] == "re-evaluate"
    assert actions[model_output_id]["action"] == "re-evaluate"
    assert actions[snapshot_id]["step"] < actions[training_id]["step"]
    assert actions[training_id]["step"] < actions[model_id]["step"]
    assert actions[model_id]["step"] < actions[eval_id]["step"]

    invalidation_edges = [
        row
        for row in lake.table("lineage_edges").to_arrow().to_pylist()
        if row["edge_type"] == "invalidates"
    ]
    assert invalidation_edges


def test_rebuild_plan_for_embedding_provider_marks_search_curation_and_training(
    tmp_path, fixtures_dir
):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    enrichment = enrich_scenarios(lake, index=IndexSpec())
    curated = lake.curate.workbench().dedup(near_duplicate_threshold=0.999)
    snapshot = curated.snapshot(name="provider-curated", split_by="scenario")
    training = lake.training.record_run("provider-curated", training_run_id="train-provider-0066")

    plan = lake.lineage.rebuild_plan(
        provider=enrichment.embedding_provider,
        provider_version=enrichment.embedding_provider_version,
        embedding_column=enrichment.embedding_column,
        reason="embedding provider regression",
        severity="medium",
    )

    actions = plan.as_dict()["actions"]
    artifact_ids = {item["artifact_id"] for item in actions}
    assert f"lancedb-robotics:execution:{enrichment.transform_id}" in artifact_ids
    assert snapshot_artifact_id(snapshot.dataset_id) in artifact_ids
    assert training_run_artifact_id(training.training_run_id) in artifact_ids
    assert any(item["kind"] == "search-index" for item in actions)
    assert any(
        item["kind"] == "transform" and item["metadata"].get("kind", "").startswith("curation-")
        for item in actions
    )


def test_trace_supports_table_version_and_created_at_filters(tmp_path):
    lake = Lake.init(tmp_path / "manual-filtered.lance")
    raw = lake.lineage.record_artifact(
        kind="source",
        source_uri="s3://bucket/run.mcap",
        source_id="source-demo",
    )
    snapshot = lake.lineage.record_artifact(
        kind="dataset-snapshot",
        name="demo",
        table_name="dataset_snapshots",
        table_version=7,
        row_ids=["ds-demo"],
        metadata={"tag": "demo-tag"},
    )
    lake.lineage.record_edge(
        edge_type="selected-from",
        from_artifact_id=raw.artifact_id,
        to_artifact_id=snapshot.artifact_id,
    )

    matching = lake.lineage.trace(
        "demo-tag",
        kind="snapshot",
        table_versions={"dataset_snapshots": 7},
    )
    excluded = lake.lineage.trace(
        "demo-tag",
        kind="snapshot",
        table_versions={"dataset_snapshots": 99},
    )

    assert raw.artifact_id in {row["artifact_id"] for row in matching.artifacts}
    assert {row["artifact_id"] for row in excluded.artifacts} == {snapshot.artifact_id}


def test_lineage_audit_reports_missing_sources_versions_and_edges(tmp_path):
    lake = Lake.init(tmp_path / "manual-audit.lance")
    missing_source = tmp_path / "missing.mcap"
    source = lake.lineage.record_artifact(
        kind="source",
        source_uri=str(missing_source),
        source_id="source-missing-0067",
    )
    missing_version = lake.lineage.record_artifact(
        kind="table-version",
        artifact_id=table_version_artifact_id("scenarios", 999),
        name="scenarios@999",
        table_name="scenarios",
        table_version=999,
    )
    lake.lineage.record_edge(
        edge_type="version-pinned",
        from_artifact_id=missing_version.artifact_id,
        to_artifact_id=source.artifact_id,
    )
    lake.lineage.record_edge(
        edge_type="selected-from",
        from_artifact_id=source.artifact_id,
        to_artifact_id="lancedb-robotics:missing:artifact",
    )

    report = lake.lineage.audit(source.artifact_id, refresh=False)
    payload = report.as_dict()

    assert payload["subject"] == source.artifact_id
    assert {
        item["artifact_id"]
        for item in payload["missing_sources"]
    } == {source.artifact_id}
    assert {
        (item["table"], item["version"])
        for item in payload["missing_table_versions"]
    } == {("scenarios", 999)}
    assert {
        item["missing_artifact_id"]
        for item in payload["unresolved_references"]
    } == {"lancedb-robotics:missing:artifact"}
    assert payload["status"] == "failed"
    assert payload["summary"]["missing_sources"] == 1
    assert any(
        row["validator"] == "local-source-existence" and row["status"] == "failed"
        for row in payload["validator_statuses"]
    )


def test_lineage_audit_pages_findings_with_stable_tokens(tmp_path):
    lake, source = _audit_catalog_lake(tmp_path)

    first = lake.lineage.audit(source.artifact_id, refresh=False, page_size=1)
    first_payload = first.as_dict()
    second = lake.lineage.audit(
        source.artifact_id,
        refresh=False,
        page_size=1,
        page_token=first_payload["page"]["next_page_token"],
    )
    second_payload = second.as_dict()

    assert first_payload["status"] == "failed"
    assert first_payload["summary"]["unresolved_references"] == 1
    assert first_payload["summary"]["missing_sources"] == 1
    assert first_payload["summary"]["missing_table_versions"] == 1
    assert first_payload["page"]["total_findings"] >= 3
    assert first_payload["page"]["returned_findings"] == 1
    assert first_payload["page"]["truncated"] is True
    assert second_payload["page"]["returned_findings"] == 1
    assert _audit_finding_types(first_payload) != _audit_finding_types(second_payload)


def test_lineage_audit_catalog_roundtrip_and_finding_export(tmp_path):
    lake, _source = _audit_catalog_lake(tmp_path)
    lake.lineage.record_artifact(
        kind="source",
        source_uri="s3://robotics-bucket/missing.mcap",
        source_id="remote-source-0112",
    )
    report = lake.lineage.audit(refresh=False)

    entry = lake.lineage.record_audit_report(
        report,
        metadata={"task": "0112"},
        created_by="pytest",
    )
    loaded_entry, loaded_report = lake.lineage.get_audit_report(entry.report_digest)
    page = lake.lineage.audit_reports(status="failed", page_size=1)
    findings = lake.lineage.audit_findings(entry.report_id, page_size=2)
    source_lines = list(
        lake.lineage.iter_audit_findings_ndjson(
            entry.report_id,
            finding_type="missing_sources",
            include_summary=True,
        )
    )

    assert entry.report_id == entry.report_digest
    assert entry.status == "failed"
    assert entry.finding_count >= 3
    assert entry.missing_source_count == 1
    assert loaded_entry.report_id == entry.report_id
    assert loaded_report["report_digest"] == entry.report_digest
    assert loaded_report["graph_snapshot"]["tables"]
    assert page.reports[0].report_id == entry.report_id
    assert findings.next_cursor is not None
    assert {row["record_type"] for row in findings.findings} == {"finding"}
    assert json.loads(source_lines[0])["finding_type"] == "missing_sources"
    assert json.loads(source_lines[-1])["record_type"] == "summary"
    assert any(
        row["validator"] == "object-store-source-existence" and row["status"] == "skipped"
        for row in loaded_report["validator_statuses"]
    )


def _audit_catalog_lake(tmp_path):
    lake = Lake.init(tmp_path / "audit-catalog.lance")
    source = lake.lineage.record_artifact(
        kind="source",
        source_uri=str(tmp_path / "missing-audit-catalog.mcap"),
        source_id="source-missing-0112",
    )
    missing_version = lake.lineage.record_artifact(
        kind="table-version",
        artifact_id=table_version_artifact_id("scenarios", 999),
        name="scenarios@999",
        table_name="scenarios",
        table_version=999,
    )
    lake.lineage.record_edge(
        edge_type="version-pinned",
        from_artifact_id=missing_version.artifact_id,
        to_artifact_id=source.artifact_id,
    )
    lake.lineage.record_edge(
        edge_type="selected-from",
        from_artifact_id=source.artifact_id,
        to_artifact_id="lancedb-robotics:missing:audit-catalog",
    )
    return lake, source


def _audit_finding_types(payload):
    return tuple(
        key
        for key in (
            "unresolved_references",
            "missing_sources",
            "missing_table_versions",
            "stale_external_links",
            "retained_versions",
            "retention_holds",
            "cleanup_candidates",
        )
        if payload[key]
    )


def test_refresh_graph_is_idempotent(tmp_path, fixtures_dir):
    lake, _manifest = _trace_lake(tmp_path, fixtures_dir)

    first = lake.lineage.refresh_graph()
    first_counts = _lineage_counts(lake)
    second = lake.lineage.refresh_graph()

    assert second.artifacts == first.artifacts
    assert second.executions == first.executions
    assert second.edges == first.edges
    assert _lineage_counts(lake) == first_counts


def test_manual_lineage_recording_and_bounded_traversal(tmp_path):
    lake = Lake.init(tmp_path / "manual.lance")
    raw = lake.lineage.record_artifact(
        kind="source",
        source_uri="s3://bucket/run.mcap",
        source_id="source-demo",
    )
    snapshot = lake.lineage.record_artifact(kind="dataset-snapshot", name="demo", row_ids=["ds-demo"])
    execution = lake.lineage.record_execution(
        kind="curation",
        transform_id="tfm-demo",
        input_artifact_ids=[raw.artifact_id],
        output_artifact_ids=[snapshot.artifact_id],
        params={"query": "hard-braking"},
        code_ref="git:abc123",
        provider="unit-test",
        status="completed",
    )
    lake.lineage.record_edge(
        edge_type="selected-from",
        from_artifact_id=raw.artifact_id,
        to_artifact_id=snapshot.artifact_id,
        execution_id=execution.execution_id,
    )

    graph = lake.lineage.trace(snapshot.artifact_id, max_depth=1)

    assert {row["artifact_id"] for row in graph.artifacts} == {
        raw.artifact_id,
        snapshot.artifact_id,
    }
    assert [row["edge_type"] for row in graph.edges] == ["selected-from"]
    assert graph.executions[0]["code_ref"] == "git:abc123"
    assert graph.executions[0]["provider"] == "unit-test"


def test_lineage_graph_report_contract_caps_and_evidence(tmp_path):
    lake = Lake.init(tmp_path / "report-contract.lance")
    raw = lake.lineage.record_artifact(
        kind="source",
        source_uri="s3://bucket/run.mcap",
        source_id="source-report",
        digest="sha256:report",
    )
    table_version = lake.lineage.record_artifact(
        kind="table-version",
        name="observations@7",
        table_name="observations",
        table_version=7,
    )
    snapshot = lake.lineage.record_artifact(
        kind="dataset-snapshot",
        name="demo",
        row_ids=["ds-report"],
    )
    lake.lineage.record_edge(
        edge_type="selected-from",
        from_artifact_id=raw.artifact_id,
        to_artifact_id=snapshot.artifact_id,
    )
    lake.lineage.record_edge(
        edge_type="version-pinned",
        from_artifact_id=table_version.artifact_id,
        to_artifact_id=snapshot.artifact_id,
    )

    graph = lake.lineage.trace(
        snapshot.artifact_id,
        max_depth=1,
        edge_types=("selected-from", "version-pinned"),
    )
    payload = graph.as_dict(include_evidence=False, max_edges=1)

    assert payload["report_version"] == LINEAGE_REPORT_VERSION
    assert payload["report_type"] == "lineage-graph"
    assert payload["controls"]["traversal"]["max_depth"] == 1
    assert payload["controls"]["traversal"]["edge_types"] == [
        "selected-from",
        "version-pinned",
    ]
    assert payload["controls"]["report"]["include_evidence"] is False
    assert payload["controls"]["report"]["max_edges"] == 1
    assert len(payload["edges"]) == 1
    assert any(warning["code"] == "edges-truncated" for warning in payload["warnings"])
    assert payload["evidence"] == {
        "included": False,
        "source_coordinates": [],
        "table_versions": [],
    }
    assert all("source_uri" not in row for row in payload["artifacts"])
    assert all("table_version" not in row for row in payload["artifacts"])

    expanded = graph.as_dict(include_evidence=True)
    assert any(
        row["source_uri"] == "s3://bucket/run.mcap"
        for row in expanded["evidence"]["source_coordinates"]
    )
    assert any(
        row["table_name"] == "observations" and row["table_version"] == 7
        for row in expanded["evidence"]["table_versions"]
    )
    ndjson_records = graph.iter_ndjson_records(include_evidence=False, max_edges=1)
    assert ndjson_records[0]["record_type"] == "report"
    assert len([row for row in ndjson_records if row["record_type"] == "edge"]) == 1


def _table_versions_and_counts(lake):
    return {
        name: (int(lake.table(name).version), lake.table(name).count_rows())
        for name in lake.table_names()
    }


def _lineage_counts(lake):
    return {
        name: lake.table(name).count_rows()
        for name in ("lineage_artifacts", "lineage_executions", "lineage_edges")
    }


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(uri)


def _messages_by_channel(path: Path, channel: str) -> list[tuple[str, int]]:
    with path.open("rb") as handle:
        return [
            (mcap_channel.topic, message.log_time)
            for _schema, mcap_channel, message in make_reader(handle).iter_messages(
                log_time_order=True
            )
            if mcap_channel.topic == channel
        ]


def _sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()
