"""CLI tests for `lancedb-robotics export mcap` (backlog 0011)."""

import json

from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage_hooks import LINEAGE_CONTEXT_ENV

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


def test_export_mcap_writes_clips_and_manifest(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)
    out = tmp_path / "clips"

    result = runner.invoke(
        app,
        ["export", "mcap", "--lake", str(lake_path), "--snapshot", "demo-v1", "--out", str(out)],
    )

    assert result.exit_code == 0
    assert "format: mcap" in result.output
    assert "manifest:" in result.output
    assert "exported" in result.output
    assert (out / "export_manifest.json").is_file()
    assert list(out.glob("*.mcap"))


def test_export_mcap_reads_lineage_context_from_environment(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)
    out = tmp_path / "clips"
    context = {
        "provider": "airflow",
        "run_id": "cli-run-0068",
        "job_id": "export-task",
        "code_version": "git:cli-export",
        "environment_digest": "sha256:cli-env",
    }

    result = runner.invoke(
        app,
        ["export", "mcap", "--lake", str(lake_path), "--snapshot", "demo-v1", "--out", str(out)],
        env={LINEAGE_CONTEXT_ENV: json.dumps(context)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((out / "export_manifest.json").read_text())
    assert payload["lineage_context"]["external_run_id"] == "cli-run-0068"
    assert payload["lineage_context"]["external_job_id"] == "export-task"

    lake = Lake.open(lake_path)
    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == payload["transform_id"]
    )
    params = json.loads(transform["params"])
    assert params["external_refs"]["airflow_run_id"] == "cli-run-0068"
    assert params["code_ref"] == "git:cli-export"


def test_export_mcap_plan_only(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)
    out = tmp_path / "clips"

    result = runner.invoke(
        app,
        [
            "export",
            "mcap",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
            "--out",
            str(out),
            "--plan-only",
        ],
    )

    assert result.exit_code == 0
    assert "planned" in result.output
    assert not list(out.glob("*.mcap"))


def test_export_mcap_missing_lake_exits_one(tmp_path):
    result = runner.invoke(
        app,
        [
            "export",
            "mcap",
            "--lake",
            str(tmp_path / "nope.lance"),
            "--snapshot",
            "demo-v1",
            "--out",
            str(tmp_path / "clips"),
        ],
    )

    assert result.exit_code == 1
    assert "lake init" in result.output


def test_export_mcap_unknown_snapshot_exits_one(tmp_path, fixtures_dir):
    lake_path = _snapshot_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "export",
            "mcap",
            "--lake",
            str(lake_path),
            "--snapshot",
            "ghost",
            "--out",
            str(tmp_path / "clips"),
        ],
    )

    assert result.exit_code == 1
    assert "snapshot" in result.output.lower()
