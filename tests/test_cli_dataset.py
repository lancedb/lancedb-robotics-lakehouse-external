"""CLI tests for `lancedb-robotics dataset snapshot create` (backlog 0009)."""

import json

from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.lake import Lake

runner = CliRunner()


def _searchable_lake(tmp_path, fixtures_dir):
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
    return lake_path


def test_dataset_snapshot_from_search_last(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)
    # A recorded search becomes the "last" selection.
    assert (
        runner.invoke(
            app, ["search", "hybrid", "imu observations", "--lake", str(lake_path)]
        ).exit_code
        == 0
    )

    result = runner.invoke(
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
            "demo-v1",
        ],
    )

    assert result.exit_code == 0
    assert "dataset: demo-v1 (ds-" in result.output
    assert "scenarios:" in result.output
    assert "table versions:" in result.output
    assert "transform: tfm-snapshot-" in result.output
    # backlog 0098: the write command surfaces the inline-emitted lineage ids.
    assert "lineage: execution=" in result.output
    assert "artifacts=" in result.output


def test_dataset_snapshot_from_explicit_scenario_ids(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)
    # Discover a scenario id via a search to feed explicitly.
    listed = runner.invoke(app, ["search", "scalar", "--lake", str(lake_path)])
    assert listed.exit_code == 0
    scenario_id = next(tok for tok in listed.output.split() if tok.startswith("scn-"))

    result = runner.invoke(
        app,
        [
            "dataset",
            "snapshot",
            "create",
            "--lake",
            str(lake_path),
            "--scenario-id",
            scenario_id,
            "--name",
            "one",
        ],
    )

    assert result.exit_code == 0
    assert "dataset: one (ds-" in result.output
    assert "scenarios: 1" in result.output


def test_dataset_snapshot_missing_lake_exits_one(tmp_path):
    result = runner.invoke(
        app,
        [
            "dataset",
            "snapshot",
            "create",
            "--lake",
            str(tmp_path / "nope.lance"),
            "--from-search",
            "last",
            "--name",
            "x",
        ],
    )

    assert result.exit_code == 1
    assert "lake init" in result.output


def test_dataset_snapshot_from_search_last_without_recorded_search_exits_one(
    tmp_path, fixtures_dir
):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
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
            "x",
        ],
    )

    assert result.exit_code == 1
    assert "search" in result.output


def test_dataset_snapshot_requires_a_selection(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app, ["dataset", "snapshot", "create", "--lake", str(lake_path), "--name", "x"]
    )

    assert result.exit_code == 1


def test_dataset_export_cli_writes_manifest(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)
    assert (
        runner.invoke(
            app, ["search", "hybrid", "imu observations", "--lake", str(lake_path)]
        ).exit_code
        == 0
    )
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
                "demo-v1",
            ],
        ).exit_code
        == 0
    )

    out = tmp_path / "rlds"
    result = runner.invoke(
        app,
        [
            "dataset",
            "export",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
            "--format",
            "rlds",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert "format: rlds" in result.output
    assert "content hash:" in result.output
    assert (out / "dataset_export_manifest.json").is_file()


def test_dataset_project_plan_cli_reports_projection_manifest(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)
    assert (
        runner.invoke(
            app, ["search", "hybrid", "imu observations", "--lake", str(lake_path)]
        ).exit_code
        == 0
    )
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
                "demo-v1",
            ],
        ).exit_code
        == 0
    )

    result = runner.invoke(
        app,
        [
            "dataset",
            "project",
            "lerobot",
            "--mode",
            "plan",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
        ],
    )

    assert result.exit_code == 0
    assert "mode: plan" in result.output
    assert "format: lerobot" in result.output
    assert "payload bytes: referenced=" in result.output
    assert "logical=" in result.output
    assert "planned_copy=" in result.output
    assert "copy_ratio=" in result.output
    assert "materialization: dry-run estimate" in result.output
    assert "transform: tfm-projection-" in result.output


def test_dataset_materialization_summary_cli_compares_projection_formats(
    tmp_path, fixtures_dir
):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)
    assert (
        runner.invoke(
            app, ["search", "hybrid", "imu observations", "--lake", str(lake_path)]
        ).exit_code
        == 0
    )
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
                "demo-v1",
            ],
        ).exit_code
        == 0
    )

    result = runner.invoke(
        app,
        [
            "dataset",
            "materialization-summary",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
            "--format",
            "lerobot",
            "--format",
            "webdataset",
            "--shard-size",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "snapshot: demo-v1" in result.output
    assert "format: lerobot" in result.output
    assert "format: webdataset" in result.output
    assert "copied_payload_bytes: 0" in result.output
    assert "logical_reference_bytes:" in result.output
    assert "planned_copy_bytes:" in result.output
    assert "metadata_bytes_written:" in result.output
    assert "copy_ratio: 0.000000" in result.output


def test_dataset_project_reads_lineage_context_from_json_file(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)
    assert (
        runner.invoke(
            app, ["search", "hybrid", "imu observations", "--lake", str(lake_path)]
        ).exit_code
        == 0
    )
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
                "demo-v1",
            ],
        ).exit_code
        == 0
    )
    context_path = tmp_path / "lineage-context.json"
    context_path.write_text(
        json.dumps(
            {
                "provider": "dagster",
                "run_id": "asset-run-0068",
                "job_id": "projection-plan",
                "code_ref": "git:projection",
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "dataset",
            "project",
            "lerobot",
            "--mode",
            "plan",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
            "--lineage-context",
            str(context_path),
        ],
    )

    assert result.exit_code == 0, result.output
    transform_id = next(line.split(":", 1)[1].strip() for line in result.output.splitlines() if line.startswith("transform:"))
    lake = Lake.open(lake_path)
    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == transform_id
    )
    params = json.loads(transform["params"])
    assert params["lineage_context"]["external_run_id"] == "asset-run-0068"
    assert params["external_refs"]["dagster_job_id"] == "projection-plan"


def test_dataset_project_webdataset_plan_cli_reports_shards_and_dependency(
    tmp_path, fixtures_dir, monkeypatch
):
    import lancedb_robotics.dataset_export as dataset_export

    monkeypatch.setattr(dataset_export, "_module_available", lambda module: False)

    lake_path = _searchable_lake(tmp_path, fixtures_dir)
    assert (
        runner.invoke(
            app, ["search", "hybrid", "imu observations", "--lake", str(lake_path)]
        ).exit_code
        == 0
    )
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
                "demo-v1",
            ],
        ).exit_code
        == 0
    )

    result = runner.invoke(
        app,
        [
            "dataset",
            "project",
            "webdataset",
            "--mode",
            "plan",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
            "--shard-size",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "format: webdataset" in result.output
    assert "planned shards:" in result.output
    assert "estimated bytes:" in result.output
    assert "sample schema: __key__, json, media, txt" in result.output
    assert "install: lancedb-robotics[webdataset] (missing webdataset)" in result.output


def test_dataset_export_accepts_projection_format_argument(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)
    assert (
        runner.invoke(
            app, ["search", "hybrid", "imu observations", "--lake", str(lake_path)]
        ).exit_code
        == 0
    )
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
                "demo-v1",
            ],
        ).exit_code
        == 0
    )

    out = tmp_path / "lerobot"
    result = runner.invoke(
        app,
        [
            "dataset",
            "export",
            "lerobot",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert "mode: export" in result.output
    assert "projection manifest:" in result.output
    assert "payload bytes: referenced=" in result.output
    assert (out / "projection_manifest.json").is_file()


def test_dataset_export_webdataset_cli_writes_tar_and_projection_manifest(
    tmp_path, fixtures_dir
):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)
    assert (
        runner.invoke(
            app, ["search", "hybrid", "imu observations", "--lake", str(lake_path)]
        ).exit_code
        == 0
    )
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
                "demo-v1",
            ],
        ).exit_code
        == 0
    )

    out = tmp_path / "webdataset"
    result = runner.invoke(
        app,
        [
            "dataset",
            "export",
            "webdataset",
            "--lake",
            str(lake_path),
            "--snapshot",
            "demo-v1",
            "--out",
            str(out),
            "--shard-size",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "format: webdataset" in result.output
    assert "planned shards:" in result.output
    assert "projection manifest:" in result.output
    assert (out / "projection_manifest.json").is_file()
    assert any((out / "shards").glob("*.tar"))
