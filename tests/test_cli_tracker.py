"""CLI tests for `train tracker` experiment-tracker sync (backlog 0101)."""

import json

from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.lake import Lake

runner = CliRunner()


def _bundle() -> dict:
    return {
        "source": "generic",
        "training_runs": [
            {
                "external_run_id": "run-1",
                "dataset_id": "ds-ext-1",
                "snapshot_name": "ext-snap",
                "table_versions": [{"table": "observations", "version": 2, "tag": ""}],
                "code_ref": "git:v1",
                "hyperparameters": {"lr": 0.001},
            }
        ],
        "model_artifacts": [
            {
                "external_artifact_id": "ckpt-1",
                "external_run_id": "run-1",
                "artifact_uri": "s3://models/policy.ckpt",
                "checksum": "sha256:ckpt",
            }
        ],
    }


def _write_bundle(tmp_path, bundle) -> str:
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    return str(path)


def test_cli_tracker_import_export_drift_roundtrip(tmp_path):
    lake_path = str(tmp_path / "robot.lance")
    Lake.init(lake_path)
    bundle_path = _write_bundle(tmp_path, _bundle())

    imported = runner.invoke(
        app,
        ["train", "tracker", "import", "--lake", lake_path, "--from", bundle_path,
         "--source", "generic", "--format", "json"],
    )
    assert imported.exit_code == 0, imported.output
    payload = json.loads(imported.output)
    assert payload["counts"]["created"] == 2
    assert payload["transform_id"]

    # Export back out to a bundle file.
    out_path = str(tmp_path / "exported.json")
    exported = runner.invoke(
        app,
        ["train", "tracker", "export", "--lake", lake_path, "--out", out_path, "--format", "json"],
    )
    assert exported.exit_code == 0, exported.output
    exported_payload = json.loads(exported.output)
    assert exported_payload["counts"]["training_runs"] == 1

    # Drift check with a changed bundle reports a conflict.
    drift_bundle = _bundle()
    drift_bundle["training_runs"][0]["code_ref"] = "git:v2"
    drift_path = _write_bundle(tmp_path, drift_bundle)
    drift = runner.invoke(
        app,
        ["train", "tracker", "drift", "--lake", lake_path, "--from", drift_path, "--format", "json"],
    )
    assert drift.exit_code == 0, drift.output
    drift_payload = json.loads(drift.output)
    assert drift_payload["has_drift"] is True
    assert drift_payload["counts"]["conflicts"] == 1


def test_cli_tracker_import_missing_mlflow_extra(tmp_path):
    lake_path = str(tmp_path / "robot.lance")
    Lake.init(lake_path)
    result = runner.invoke(
        app,
        ["train", "tracker", "import", "--lake", lake_path, "--source", "mlflow"],
    )
    assert result.exit_code == 1
    assert "mlflow" in result.output
