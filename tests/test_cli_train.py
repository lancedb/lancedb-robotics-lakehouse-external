"""CLI tests for `lancedb-robotics train preview torch` (backlog 0010)."""

import json

from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.training import torch_available

runner = CliRunner()


def _snapshot_lake(tmp_path, fixtures_dir, *, name="demo-v1"):
    lake_path = tmp_path / "robot.lance"
    assert runner.invoke(app, ["lake", "init", "--lake", str(lake_path)]).exit_code == 0
    assert (
        runner.invoke(
            app, ["ingest", "mcap", str(fixtures_dir / "sample.mcap"), "--lake", str(lake_path)]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app, ["scenarios", "create", "--lake", str(lake_path), "--window", "50ms"]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["scenarios", "enrich", "--lake", str(lake_path)]).exit_code == 0
    assert runner.invoke(app, ["search", "hybrid", "imu", "--lake", str(lake_path)]).exit_code == 0
    assert (
        runner.invoke(
            app,
            [
                "dataset",
                "snapshot",
                "create",
                "--lake",
                str(lake_path),
                "--from-search",
                "last",
                "--name",
                name,
            ],
        ).exit_code
        == 0
    )
    return lake_path


def test_train_preview_torch_prints_deterministic_samples(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app, ["train", "preview", "torch", "--lake", str(lake_path), "--snapshot", "demo-v1"]
    )

    assert result.exit_code == 0
    assert "snapshot: demo-v1 (ds-" in result.output
    assert "scenarios:" in result.output
    assert "sample 1:" in result.output


def test_train_preview_torch_reports_dependency_state(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app, ["train", "preview", "torch", "--lake", str(lake_path), "--snapshot", "demo-v1"]
    )

    assert result.exit_code == 0
    # Friendly, explicit messaging either way; preview still prints.
    if torch_available():
        assert "torch" in result.output.lower()
    else:
        assert "torch not installed" in result.output.lower()


def test_train_preview_torch_column_projection(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "train",
            "preview",
            "torch",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
            "--columns",
            "scenario_id,split",
        ],
    )

    assert result.exit_code == 0
    assert "columns: scenario_id, split" in result.output


def test_train_preview_torch_missing_lake_exits_one(tmp_path):
    result = runner.invoke(
        app,
        [
            "train",
            "preview",
            "torch",
            "--lake",
            str(tmp_path / "nope.lance"),
            "--snapshot",
            "demo-v1",
        ],
    )

    assert result.exit_code == 1
    assert "lake init" in result.output


def test_train_preview_torch_unknown_snapshot_exits_one(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app, ["train", "preview", "torch", "--lake", str(lake_path), "--snapshot", "ghost"]
    )

    assert result.exit_code == 1
    assert "snapshot" in result.output.lower()


def test_train_remote_report_requires_enterprise_or_fallback(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "train",
            "remote",
            "report",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
        ],
    )

    assert result.exit_code == 1
    assert "requires a db:// or namespace-backed lake" in result.output


def test_train_remote_report_json_records_explicit_fallback(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "train",
            "remote",
            "report",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
            "--allow-fallback",
            "--cache-policy",
            "epoch",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["backend"]["requested_backend"] == "enterprise"
    assert payload["backend"]["resolved_backend"] == "local"
    assert payload["backend"]["fallback"]["to"] == "local"
    assert payload["manifest"]["backend"]["cache"]["policy"] == "epoch"
    assert payload["loader_report"]["fallback_events"][0]["to"] == "local"


def test_train_remote_report_writes_loader_report_json(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)
    report_path = tmp_path / "training_loader_report.json"

    result = runner.invoke(
        app,
        [
            "train",
            "remote",
            "report",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
            "--allow-fallback",
            "--cache-policy",
            "epoch",
            "--report-out",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(report_path.read_text())
    assert payload["kind"] == "lancedb-robotics/training-loader-report/v1"
    assert payload["snapshot"]["name"] == "demo-v1"
    assert payload["policies"]["enterprise_cache"]["policy"] == "epoch"
    assert payload["fallback_events"][0]["to"] == "local"


def _seed_prewarm_jobrun(lake_path, prewarm_id, status, *, reason=None):
    from datetime import UTC, datetime

    from lancedb_robotics.lake import Lake
    from lancedb_robotics.training_prewarm_jobs import (
        LanceTablePrewarmJobRunStore,
        build_prewarm_job_run,
    )

    lake = Lake.open(str(lake_path))
    store = LanceTablePrewarmJobRunStore(lake._db)
    now = datetime(2026, 7, 6, tzinfo=UTC)
    record = build_prewarm_job_run(
        {
            "prewarm_id": prewarm_id,
            "policy": "epoch",
            "scope": "epoch",
            "snapshot_name": "demo-v1",
            "routing": {"mode": "host-override"},
            "tables": [{"table": "observations", "version": 1}],
            "projected_columns": ["observation_id"],
            "logical_columns": ["observation_id"],
            "row_count": 3,
        },
        job_label="demo-v1",
        caller_label=None,
        now=now,
        ttl_s=3600.0,
        store_kind="lancedb-table",
        store_ref="",
    )
    store.put(record.attach("worker-0/2", now).with_status(status, now, reason=reason))


def test_train_prewarm_list_status_cancel_retry(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)
    _seed_prewarm_jobrun(lake_path, "prewarm-done", "complete")
    _seed_prewarm_jobrun(lake_path, "prewarm-bad", "failed", reason="pe down")

    listing = runner.invoke(
        app, ["train", "prewarm", "list", "--lake", str(lake_path), "--format", "json"]
    )
    assert listing.exit_code == 0, listing.output
    jobs = {job["prewarm_id"]: job for job in json.loads(listing.output)}
    assert set(jobs) == {"prewarm-done", "prewarm-bad"}
    assert jobs["prewarm-bad"]["terminal_reason"] == "pe down"

    status = runner.invoke(
        app,
        ["train", "prewarm", "status", "--lake", str(lake_path), "--id", "prewarm-bad"],
    )
    assert status.exit_code == 0
    assert "failed" in status.output

    # Cancel is a no-op on a complete (warm) JobRun.
    cancel = runner.invoke(
        app,
        ["train", "prewarm", "cancel", "--lake", str(lake_path), "--id", "prewarm-done"],
    )
    assert cancel.exit_code == 0
    assert "complete" in cancel.output

    # Retry a failed JobRun -> re-submitted (retry_count increments).
    retry = runner.invoke(
        app,
        [
            "train", "prewarm", "retry", "--lake", str(lake_path),
            "--id", "prewarm-bad", "--format", "json",
        ],
    )
    assert retry.exit_code == 0, retry.output
    assert json.loads(retry.output)["retry_count"] == 1


def test_train_prewarm_list_empty(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)
    result = runner.invoke(app, ["train", "prewarm", "list", "--lake", str(lake_path)])
    assert result.exit_code == 0
    assert "no prewarm JobRuns recorded" in result.output


def test_train_conformance_query_node_command_registered_with_deprecated_alias():
    """Backlog 0345: `train conformance query-node` is the command; `plan-executor`
    stays as a hidden deprecated alias."""
    # The new command name is advertised in help; the old one is hidden.
    top = runner.invoke(app, ["train", "conformance", "--help"])
    assert top.exit_code == 0
    assert "query-node" in top.output
    assert "plan-executor" not in top.output

    # Both names resolve to a real command (missing --lake/--snapshot → usage error,
    # exit code 2 — proves the command exists and parses, not "no such command").
    new = runner.invoke(app, ["train", "conformance", "query-node", "--help"])
    assert new.exit_code == 0
    old = runner.invoke(app, ["train", "conformance", "plan-executor", "--help"])
    assert old.exit_code == 0
    assert "Deprecated alias" in old.output


# --- Backlog 0124: `train report validate` -----------------------------------


def _write_native_report(tmp_path, **loader_report_kwargs):
    import test_native_training_dataset as native_mod

    lake = native_mod._training_lake(tmp_path / "native.lance")
    dataset = lake.training.dataset("demo-v1", columns=["observation_id"])
    payload = dataset.loader_report(**loader_report_kwargs).to_dict()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload))
    return path, payload


def test_report_validate_accepts_clean_report(tmp_path):
    path, _payload = _write_native_report(tmp_path)
    result = runner.invoke(app, ["train", "report", "validate", str(path)])
    assert result.exit_code == 0
    assert "OK" in result.output
    assert "lancedb-robotics/training-loader-report/v1" in result.output


def test_report_validate_rejects_credential_bearing_report(tmp_path):
    # A hand-tampered report with an unredacted credential must be refused.
    path, payload = _write_native_report(tmp_path)
    payload["run"]["authorization"] = "Bearer sk-live-abc"
    del payload["metrics"]
    path.write_text(json.dumps(payload))
    result = runner.invoke(app, ["train", "report", "validate", str(path)])
    assert result.exit_code == 1
    assert "FAILED" in result.output
    assert "redaction" in result.output
    assert "schema" in result.output


def test_report_validate_rejects_malformed_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json")
    result = runner.invoke(app, ["train", "report", "validate", str(path)])
    assert result.exit_code == 1
    assert "invalid JSON" in result.output


def test_report_validate_json_output_gates_on_worst_report(tmp_path):
    good, _ = _write_native_report(tmp_path)
    bad_payload = json.loads(good.read_text())
    bad_payload["run"]["api_key"] = "AKIAIOSFODNN7EXAMPLE"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bad_payload))
    result = runner.invoke(
        app, ["train", "report", "validate", str(good), str(bad), "--format", "json"]
    )
    assert result.exit_code == 1
    parsed = json.loads(result.output)
    assert parsed["ok"] is False
    by_path = {entry["path"]: entry for entry in parsed["reports"]}
    assert by_path[str(good)]["ok"] is True
    assert by_path[str(bad)]["ok"] is False
