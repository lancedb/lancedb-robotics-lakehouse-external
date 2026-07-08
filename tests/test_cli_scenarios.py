"""CLI tests for `lancedb-robotics scenarios create` (backlog 0006)."""

from typer.testing import CliRunner

from lancedb_robotics.cli import app

runner = CliRunner()


def _lake_with_sample(tmp_path, fixtures_dir):
    lake_path = tmp_path / "robot.lance"
    result = runner.invoke(app, ["lake", "init", "--lake", str(lake_path)])
    assert result.exit_code == 0
    result = runner.invoke(
        app, ["ingest", "mcap", str(fixtures_dir / "sample.mcap"), "--lake", str(lake_path)]
    )
    assert result.exit_code == 0
    return lake_path


def test_scenarios_create_reports_created_windows(tmp_path, fixtures_dir):
    lake_path = _lake_with_sample(tmp_path, fixtures_dir)

    result = runner.invoke(
        app, ["scenarios", "create", "--lake", str(lake_path), "--window", "100ms"]
    )

    assert result.exit_code == 0
    assert "window: 100ms (100000000 ns)" in result.output
    assert "topics: all topics" in result.output
    assert "partial final window: included" in result.output
    assert "runs: 1" in result.output
    assert "scenarios: 2 created" in result.output
    assert "windows" in result.output


def test_scenarios_create_supports_topic_filters(tmp_path, fixtures_dir):
    lake_path = _lake_with_sample(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "scenarios",
            "create",
            "--lake",
            str(lake_path),
            "--window",
            "75ms",
            "--topic",
            "/camera/front",
        ],
    )

    assert result.exit_code == 0
    assert "topics: /camera/front" in result.output
    assert "scenarios: 2 created" in result.output


def test_scenarios_create_can_drop_partial_final_window(tmp_path, fixtures_dir):
    lake_path = _lake_with_sample(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "scenarios",
            "create",
            "--lake",
            str(lake_path),
            "--window",
            "75ms",
            "--drop-partial",
        ],
    )

    assert result.exit_code == 0
    assert "partial final window: dropped" in result.output
    assert "scenarios: 2 created" in result.output


def test_scenarios_create_missing_lake_exits_one(tmp_path):
    result = runner.invoke(
        app, ["scenarios", "create", "--lake", str(tmp_path / "nope.lance"), "--window", "5s"]
    )

    assert result.exit_code == 1
    assert "lake init" in result.output


def test_scenarios_create_invalid_window_exits_one(tmp_path, fixtures_dir):
    lake_path = _lake_with_sample(tmp_path, fixtures_dir)

    result = runner.invoke(
        app, ["scenarios", "create", "--lake", str(lake_path), "--window", "five"]
    )

    assert result.exit_code == 1
    assert "window" in result.output
