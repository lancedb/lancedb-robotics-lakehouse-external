"""CLI tests for `lancedb-robotics writeback` (backlog 0027)."""

import json

from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.lake import Lake

runner = CliRunner()


def _windowed_lake(tmp_path, fixtures_dir):
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
    return lake_path


def test_writeback_cli_imports_outputs_feedback_and_traces(tmp_path, fixtures_dir):
    lake_path = _windowed_lake(tmp_path, fixtures_dir)
    lake = Lake.open(lake_path)
    observation_id = lake.table("observations").to_arrow().to_pylist()[0]["observation_id"]
    scenario_id = lake.table("scenarios").to_arrow().to_pylist()[0]["scenario_id"]

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "observation_id": observation_id,
                        "scenario_id": scenario_id,
                        "label_type": "class",
                        "label": "pedestrian",
                    }
                ]
            }
        )
    )
    result = runner.invoke(
        app,
        [
            "writeback",
            "labels",
            str(labels_path),
            "--lake",
            str(lake_path),
            "--source",
            "label-studio",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "kind: label-writeback" in result.output
    assert "rows: 1" in result.output

    outputs_path = tmp_path / "outputs.json"
    outputs_path.write_text(
        json.dumps(
            [
                {
                    "model_output_id": "out-cli-1",
                    "observation_id": observation_id,
                    "scenario_id": scenario_id,
                    "model_version": "policy@cli",
                    "prediction": "pedestrian",
                    "score": 0.8,
                }
            ]
        )
    )
    result = runner.invoke(
        app,
        ["writeback", "model-outputs", str(outputs_path), "--lake", str(lake_path)],
    )
    assert result.exit_code == 0, result.output
    assert "kind: model-output-writeback" in result.output

    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "feedback": [
                    {
                        "model_output_id": "out-cli-1",
                        "feedback_type": "field_failure",
                        "severity": "medium",
                        "linked_incident_id": "inc-cli",
                    }
                ]
            }
        )
    )
    result = runner.invoke(
        app,
        ["writeback", "feedback", str(feedback_path), "--lake", str(lake_path)],
    )
    assert result.exit_code == 0, result.output
    assert "kind: feedback-writeback" in result.output

    result = runner.invoke(
        app, ["writeback", "trace-model-output", "out-cli-1", "--lake", str(lake_path)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model_output"]["model_output_id"] == "out-cli-1"
    assert payload["source_run"]["run_id"].startswith("run-")

    result = runner.invoke(
        app,
        ["writeback", "downstream-run", payload["source_run"]["run_id"], "--lake", str(lake_path)],
    )
    assert result.exit_code == 0, result.output
    assert "out-cli-1" in result.output
    assert "inc-cli" in result.output


def test_writeback_cli_rejects_malformed_json(tmp_path, fixtures_dir):
    lake_path = _windowed_lake(tmp_path, fixtures_dir)
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not-json")

    result = runner.invoke(
        app, ["writeback", "labels", str(bad_path), "--lake", str(lake_path)]
    )

    assert result.exit_code == 1
    assert "not valid JSON" in result.output
