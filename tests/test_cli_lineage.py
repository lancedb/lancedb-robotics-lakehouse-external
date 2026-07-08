"""CLI tests for `lancedb-robotics lineage` (backlog 0033)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import LINEAGE_REPORT_VERSION, artifact_id, snapshot_artifact_id
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.writeback import ingest_model_outputs

runner = CliRunner()


def _json_post_server():
    posts: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            posts.append(payload)
            body = json.dumps({"remote_id": f"cli-remote-{len(posts)}"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002,N802 - stdlib handler API
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, posts, f"http://127.0.0.1:{server.server_port}/lineage"


def _checkpoint_lake(tmp_path, fixtures_dir):
    lake_path = tmp_path / "robot.lance"
    lake = Lake.init(lake_path)
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    scenario = sorted(
        lake.table("scenarios").to_arrow().to_pylist(),
        key=lambda row: (row["start_time_ns"], row["scenario_id"]),
    )[0]
    snapshot = create_snapshot(
        lake,
        name="cli-demo",
        scenario_ids=[scenario["scenario_id"]],
        tag="cli-training",
    )
    ingest_model_outputs(
        lake,
        {
            "model_output_id": "out-cli-regression",
            "observation_id": scenario["observation_ids"][0],
            "scenario_id": scenario["scenario_id"],
            "dataset_id": snapshot.dataset_id,
            "model_version": "policy@cli",
            "prediction": "regressed",
            "producer_run_id": "checkpoint-cli",
        },
    )
    return lake_path


def test_lineage_trace_cli_outputs_checkpoint_report(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "lineage",
            "trace",
            "checkpoint-cli",
            "--lake",
            str(lake_path),
            "--checkpoint",
            "--where",
            "topic = '/imu'",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model_run_id"] == "checkpoint-cli"
    assert payload["dataset_snapshot"]["name"] == "cli-demo"
    assert payload["rows"][0]["topic"] == "/imu"
    assert payload["source_log_tuples"]


def test_lineage_export_evidence_cli_outputs_plan_manifest(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "lineage",
            "export-evidence",
            "checkpoint-cli",
            "--lake",
            str(lake_path),
            "--checkpoint",
            "--where",
            "topic = '/imu'",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "plan"
    assert payload["bytes_copied"] == 0
    assert payload["manifest"]["schema_version"] == "lancedb-robotics/evidence-pack/v1"
    assert payload["manifest"]["subject"]["model_run_id"] == "checkpoint-cli"
    assert payload["manifest"]["source_coordinates"]
    assert payload["manifest"]["payload_refs"]


def test_lineage_trace_cli_reports_missing_checkpoint(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "lineage",
            "checkpoint",
            "missing-checkpoint",
            "--lake",
            str(lake_path),
        ],
    )

    assert result.exit_code == 1
    assert "model_outputs" in result.output


def test_lineage_refresh_trace_and_impact_cli(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)
    lake = Lake.open(lake_path)
    snapshot = lake.table("dataset_snapshots").to_arrow().to_pylist()[0]
    run = lake.table("runs").to_arrow().to_pylist()[0]
    snapshot_id = snapshot_artifact_id(snapshot["dataset_id"])
    model_output_id = artifact_id(
        "row",
        table_name="model_outputs",
        row_id="out-cli-regression",
        table_version=int(lake.table("model_outputs").version),
    )

    refresh = runner.invoke(app, ["lineage", "refresh", "--lake", str(lake_path)])
    assert refresh.exit_code == 0, refresh.output
    assert json.loads(refresh.output)["edges"] > 0

    trace = runner.invoke(
        app,
        [
            "lineage",
            "trace",
            snapshot_id,
            "--lake",
            str(lake_path),
            "--format",
            "json",
        ],
    )
    assert trace.exit_code == 0, trace.output
    trace_payload = json.loads(trace.output)
    assert "source-coordinate" in {edge["edge_type"] for edge in trace_payload["edges"]}

    tree = runner.invoke(
        app,
        [
            "lineage",
            "trace",
            snapshot["tag"],
            "--kind",
            "snapshot",
            "--lake",
            str(lake_path),
            "--target-kind",
            "source",
            "--evidence",
        ],
    )
    assert tree.exit_code == 0, tree.output
    assert "dataset-snapshot" in tree.output
    assert "source-coordinate" in tree.output

    impact = runner.invoke(
        app,
        [
            "lineage",
            "impact",
            run["raw_uri"],
            "--kind",
            "source",
            "--target-kind",
            "model-output",
            "--lake",
            str(lake_path),
            "--format",
            "json",
        ],
    )
    assert impact.exit_code == 0, impact.output
    impact_payload = json.loads(impact.output)
    assert model_output_id in {artifact["artifact_id"] for artifact in impact_payload["artifacts"]}


def test_lineage_rebuild_plan_cli_outputs_machine_readable_json(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)
    lake = Lake.open(lake_path)
    run = lake.table("runs").to_arrow().to_pylist()[0]

    result = runner.invoke(
        app,
        [
            "lineage",
            "rebuild-plan",
            run["raw_uri"],
            "--kind",
            "source",
            "--lake",
            str(lake_path),
            "--reason",
            "bad calibration",
            "--record-invalidation",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "lancedb-robotics/rebuild-plan/v1"
    assert payload["invalidation"]["reason"] == "bad calibration"
    assert any(item["action"] == "resnapshot" for item in payload["actions"])
    assert any(item["action"] == "re-evaluate" for item in payload["actions"])
    # Backlog 0110 additive summary block.
    assert payload["summary"]["policy"] == "default"
    assert payload["summary"]["action_count"] == len(payload["actions"])


def test_lineage_rebuild_plan_cli_summary_and_pagination(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)
    run = Lake.open(lake_path).table("runs").to_arrow().to_pylist()[0]
    base = ["lineage", "rebuild-plan", run["raw_uri"], "--kind", "source", "--lake", str(lake_path)]

    summary = runner.invoke(app, [*base, "--summary"])
    assert summary.exit_code == 0, summary.output
    summary_payload = json.loads(summary.output)
    assert summary_payload["actions"] == []
    assert summary_payload["page"]["summary_only"] is True
    total = summary_payload["summary"]["action_count"]
    assert total >= 1

    # Page through the same plan and confirm the union covers every action.
    seen: list[int] = []
    token = None
    for _ in range(total + 1):
        args = [*base, "--page-size", "2"]
        if token:
            args += ["--page-token", token]
        page = runner.invoke(app, args)
        assert page.exit_code == 0, page.output
        page_payload = json.loads(page.output)
        assert page_payload["page"]["total_actions"] == total
        seen.extend(a["step"] for a in page_payload["actions"])
        token = page_payload["page"]["next_page_token"]
        if not token:
            break
    assert seen == list(range(1, total + 1))


def test_lineage_rebuild_plan_cli_action_policy_and_guardrail(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)
    run = Lake.open(lake_path).table("runs").to_arrow().to_pylist()[0]
    base = ["lineage", "rebuild-plan", run["raw_uri"], "--kind", "source", "--lake", str(lake_path)]

    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps({"name": "notify-sources", "kind_actions": {"source": "notify-only"}}))
    policy = runner.invoke(app, [*base, "--action-policy", str(policy_file)])
    assert policy.exit_code == 0, policy.output
    policy_payload = json.loads(policy.output)
    assert policy_payload["summary"]["policy"] == "notify-sources"
    source_actions = [a for a in policy_payload["actions"] if a["kind"] == "source"]
    assert source_actions and all(a["action"] == "notify-only" for a in source_actions)

    # Guardrail exits non-zero with an actionable message.
    guarded = runner.invoke(app, [*base, "--max-actions", "0"])
    assert guarded.exit_code == 1
    assert "max_actions" in guarded.output


def test_lineage_audit_cli_outputs_machine_readable_json(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = Lake.init(lake_path)
    missing_source = tmp_path / "missing-source.mcap"
    source = lake.lineage.record_artifact(
        kind="source",
        source_uri=str(missing_source),
        source_id="cli-missing-source-0067",
    )

    result = runner.invoke(
        app,
        [
            "lineage",
            "audit",
            source.artifact_id,
            "--kind",
            "source",
            "--lake",
            str(lake_path),
            "--no-refresh",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["subject"] == source.artifact_id
    assert payload["missing_sources"][0]["artifact_id"] == source.artifact_id
    assert payload["missing_sources"][0]["source_uri"] == str(missing_source)


def test_lineage_audit_cli_records_lists_reloads_and_exports_findings(tmp_path):
    lake_path = tmp_path / "audit-catalog-cli.lance"
    lake = Lake.init(lake_path)
    missing_source = tmp_path / "missing-source-0112.mcap"
    source = lake.lineage.record_artifact(
        kind="source",
        source_uri=str(missing_source),
        source_id="cli-missing-source-0112",
    )

    recorded = runner.invoke(
        app,
        [
            "lineage",
            "audit",
            source.artifact_id,
            "--kind",
            "source",
            "--lake",
            str(lake_path),
            "--no-refresh",
            "--record",
            "--created-by",
            "pytest",
            "--format",
            "json",
        ],
    )

    assert recorded.exit_code == 0, recorded.output
    recorded_payload = json.loads(recorded.output)
    report_id = recorded_payload["catalog_entry"]["report_id"]
    assert recorded_payload["catalog_entry"]["status"] == "failed"

    listed = runner.invoke(
        app,
        [
            "lineage",
            "audit-reports",
            "--lake",
            str(lake_path),
            "--status",
            "failed",
            "--format",
            "json",
        ],
    )
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.output)["reports"][0]["report_id"] == report_id

    reloaded = runner.invoke(
        app,
        ["lineage", "get-audit-report", report_id, "--lake", str(lake_path)],
    )
    assert reloaded.exit_code == 0, reloaded.output
    assert json.loads(reloaded.output)["report"]["report_id"] == report_id

    exported = runner.invoke(
        app,
        [
            "lineage",
            "export-audit-findings",
            report_id,
            "--lake",
            str(lake_path),
            "--finding-type",
            "missing_sources",
            "--format",
            "ndjson",
            "--summary",
        ],
    )
    assert exported.exit_code == 0, exported.output
    lines = [json.loads(line) for line in exported.output.splitlines()]
    assert lines[0]["finding_type"] == "missing_sources"
    assert lines[-1]["record_type"] == "summary"


def test_lineage_export_openlineage_resolve_urn_and_attach_refs_cli(
    tmp_path,
    fixtures_dir,
):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)
    lake = Lake.open(lake_path)
    training = lake.training.record_run("cli-demo", code_ref="git:cli-train")

    attach = runner.invoke(
        app,
        [
            "lineage",
            "attach-refs",
            "training-run",
            training.training_run_id,
            "--lake",
            str(lake_path),
            "--mlflow-run-id",
            "mlflow-cli-0064",
        ],
    )
    assert attach.exit_code == 0, attach.output
    attach_payload = json.loads(attach.output)
    assert attach_payload["external_refs"]["mlflow_run_id"] == "mlflow-cli-0064"

    export = runner.invoke(
        app,
        [
            "lineage",
            "export-openlineage",
            "--lake",
            str(lake_path),
        ],
    )
    assert export.exit_code == 0, export.output
    export_payload = json.loads(export.output)
    assert export_payload["event_count"] > 0
    training_event = next(
        event
        for event in export_payload["events"]
        if event["run"]["facets"]["lancedb_robotics_execution"]["execution_id"]
        == f"lancedb-robotics:execution:{training.training_run_id}"
    )
    urn = training_event["outputs"][0]["name"]

    resolved = runner.invoke(
        app,
        [
            "lineage",
            "resolve-urn",
            urn,
            "--lake",
            str(lake_path),
        ],
    )
    assert resolved.exit_code == 0, resolved.output
    assert json.loads(resolved.output)["artifact_id"].startswith(
        "lancedb-robotics:training-run:"
    )


def test_lineage_export_openlineage_emit_cli_records_delivery_status(
    tmp_path,
    fixtures_dir,
):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)
    server, posts, endpoint = _json_post_server()
    try:
        emit = runner.invoke(
            app,
            [
                "lineage",
                "export-openlineage",
                "--lake",
                str(lake_path),
                "--emit",
                "--endpoint-url",
                endpoint,
                "--target",
                "cli-marquez",
            ],
        )
        retry = runner.invoke(
            app,
            [
                "lineage",
                "export-openlineage",
                "--lake",
                str(lake_path),
                "--retry",
                "--endpoint-url",
                endpoint,
                "--target",
                "cli-marquez",
            ],
        )
    finally:
        server.shutdown()
        server.server_close()

    assert emit.exit_code == 0, emit.output
    emit_payload = json.loads(emit.output)
    assert emit_payload["status"] == "delivered"
    assert emit_payload["delivered_count"] == len(posts) > 0
    assert all(post["schemaURL"].endswith("OpenLineage.json#/$defs/RunEvent") for post in posts)

    assert retry.exit_code == 0, retry.output
    retry_payload = json.loads(retry.output)
    assert retry_payload["status"] == "already-delivered"
    assert retry_payload["already_delivered_count"] == emit_payload["delivered_count"]
    assert len(posts) == emit_payload["delivered_count"]

    attempts = Lake.open(lake_path).lineage.lineage_delivery_attempts(
        backend="openlineage",
        target="cli-marquez",
    )
    assert len(attempts) == emit_payload["delivered_count"]
    assert {attempt.status for attempt in attempts} == {"delivered"}
    assert all(attempt.remote_response_ids for attempt in attempts)


def test_lineage_check_adapter_cli_reports_missing_optional_plugin():
    result = runner.invoke(app, ["lineage", "check-adapter", "definitely-missing-0064"])

    assert result.exit_code == 1
    assert "optional extra/plugin 'definitely-missing-0064'" in result.output


def test_lineage_refresh_plan_cli_reports_source_versions(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)
    # First materialize the graph so a watermark exists.
    assert runner.invoke(app, ["lineage", "refresh", "--lake", str(lake_path)]).exit_code == 0

    result = runner.invoke(app, ["lineage", "refresh", "--lake", str(lake_path), "--plan"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    plan = payload["plan"]
    assert plan["dry_run"] is True
    assert {row["table"] for row in plan["source_tables"]} >= {"observations", "model_outputs"}
    # Nothing changed since the first refresh -> the plan skips.
    assert plan["action"] == "skipped-unchanged"


def test_lineage_index_plan_cli_reports_clean_after_refresh(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)
    assert runner.invoke(app, ["lineage", "refresh", "--lake", str(lake_path)]).exit_code == 0

    result = runner.invoke(app, ["lineage", "index-plan", "--lake", str(lake_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["all_present"] is True
    assert payload["missing"] == []


def test_lineage_impact_cli_paginates(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)
    assert runner.invoke(app, ["lineage", "refresh", "--lake", str(lake_path)]).exit_code == 0
    snapshot_handle = "cli-training"

    result = runner.invoke(
        app,
        [
            "lineage",
            "trace",
            snapshot_handle,
            "--lake",
            str(lake_path),
            "--kind",
            "snapshot",
            "--page-size",
            "1",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["page"]["page_size"] == 1
    assert payload["page"]["total_artifacts"] >= 1


def test_lineage_trace_cli_json_report_contract_and_output_file(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)
    assert runner.invoke(app, ["lineage", "refresh", "--lake", str(lake_path)]).exit_code == 0
    output_path = tmp_path / "trace-report.json"

    result = runner.invoke(
        app,
        [
            "lineage",
            "trace",
            "cli-training",
            "--lake",
            str(lake_path),
            "--kind",
            "snapshot",
            "--format",
            "json",
            "--max-edges",
            "1",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output == ""
    payload = json.loads(output_path.read_text())
    assert payload["report_version"] == LINEAGE_REPORT_VERSION
    assert payload["controls"]["traversal"]["kind"] == "snapshot"
    assert payload["controls"]["report"]["max_edges"] == 1
    assert payload["warnings"][0]["code"] == "edges-truncated"
    assert "artifacts" in payload and "edges" in payload and "executions" in payload


def test_lineage_impact_cli_ndjson_report_is_bounded(tmp_path, fixtures_dir):
    lake_path = _checkpoint_lake(tmp_path, fixtures_dir)
    lake = Lake.open(lake_path)
    run = lake.table("runs").to_arrow().to_pylist()[0]
    assert runner.invoke(app, ["lineage", "refresh", "--lake", str(lake_path)]).exit_code == 0

    result = runner.invoke(
        app,
        [
            "lineage",
            "impact",
            run["raw_uri"],
            "--kind",
            "source",
            "--target-kind",
            "model-output",
            "--lake",
            str(lake_path),
            "--format",
            "ndjson",
            "--max-edges",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.output.splitlines()]
    assert records[0]["record_type"] == "report"
    assert records[0]["report_version"] == LINEAGE_REPORT_VERSION
    assert records[0]["warnings"][0]["code"] == "edges-truncated"
    assert len([record for record in records if record["record_type"] == "edge"]) == 1
    assert any(record["record_type"] == "artifact" for record in records)


def test_lineage_resolve_cli_source_uri_json(tmp_path, fixtures_dir):
    """`lineage resolve <uri> --format json` emits deterministic multi-root output."""
    lake_path = tmp_path / "robot.lance"
    lake = Lake.init(lake_path)
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    lake.lineage.refresh_graph()
    source_uri = str((fixtures_dir / "sample.mcap").resolve())

    result = runner.invoke(
        app,
        ["lineage", "resolve", source_uri, "--lake", str(lake_path), "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "resolved"
    assert payload["multi_root"] is True
    assert payload["root_count"] == len(payload["artifact_ids"]) > 1
    # Deterministic ordering: artifact ids sorted, candidates aligned.
    assert payload["artifact_ids"] == sorted(payload["artifact_ids"])
    assert [c["artifact_id"] for c in payload["candidates"]] == sorted(payload["artifact_ids"])
    assert {c["kind"] for c in payload["candidates"]} == {"source"}
    assert payload["graph_fresh"] is True


def test_lineage_resolve_cli_ambiguous(tmp_path, fixtures_dir):
    lake_path = tmp_path / "robot.lance"
    lake = Lake.init(lake_path)
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    scenario_ids = [
        row["scenario_id"] for row in lake.table("scenarios").to_arrow().to_pylist()
    ]
    create_snapshot(lake, name="dup", tag="dup", scenario_ids=scenario_ids)
    create_snapshot(lake, name="dup", tag="dup", scenario_ids=scenario_ids[:1])
    lake.lineage.refresh_graph()

    result = runner.invoke(
        app,
        ["lineage", "resolve", "dup", "--lake", str(lake_path), "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ambiguous"
    assert payload["root_count"] == 2
    assert any(hint["flag"] == "artifact-id" for hint in payload["disambiguation_hints"])
