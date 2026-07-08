"""CLI tests for `lancedb-robotics ingest mcap` (backlog 0004)."""

import pytest
from typer.testing import CliRunner

from lancedb_robotics.cli import app

runner = CliRunner()


@pytest.fixture
def sample_mcap(fixtures_dir):
    return fixtures_dir / "sample.mcap"


@pytest.fixture
def lake_path(tmp_path):
    path = tmp_path / "robot.lance"
    result = runner.invoke(app, ["lake", "init", "--lake", str(path)])
    assert result.exit_code == 0
    return path


def test_ingest_mcap_reports_stable_row_counts(lake_path, sample_mcap):
    result = runner.invoke(app, ["ingest", "mcap", str(sample_mcap), "--lake", str(lake_path)])
    assert result.exit_code == 0
    for line in (
        "runs +1",
        "observations +5",
        "events +2",
        "transform_runs +2",
        "/camera/front\t2",
        "/imu\t3",
    ):
        assert line in result.output
    # backlog 0098: ingest emits lineage inline and the CLI surfaces the ids.
    assert "lineage: execution=" in result.output


def test_ingest_mcap_twice_reports_already_ingested(lake_path, sample_mcap):
    first = runner.invoke(app, ["ingest", "mcap", str(sample_mcap), "--lake", str(lake_path)])
    second = runner.invoke(app, ["ingest", "mcap", str(sample_mcap), "--lake", str(lake_path)])
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "already ingested" in second.output
    # Same run id reported both times: ingest is content-addressed.
    run_line = next(line for line in first.output.splitlines() if line.startswith("run:"))
    assert run_line.split()[1] in second.output


def test_ingest_mcap_into_missing_lake_fails_cleanly(tmp_path, sample_mcap):
    result = runner.invoke(
        app, ["ingest", "mcap", str(sample_mcap), "--lake", str(tmp_path / "nope.lance")]
    )
    assert result.exit_code != 0
    assert "lake init" in result.output


def test_ingest_mcap_invalid_file_fails_cleanly(lake_path, tmp_path):
    bogus = tmp_path / "bogus.mcap"
    bogus.write_bytes(b"this is not an mcap file")
    result = runner.invoke(app, ["ingest", "mcap", str(bogus), "--lake", str(lake_path)])
    assert result.exit_code != 0
    assert "error" in result.output
