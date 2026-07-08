"""CLI tests for `lancedb-robotics quality validate` (backlog 0005).

Exit-code contract under test: 0 = every validated run passed, 1 = operational
error (missing lake, unknown profile, unknown run), 2 = at least one run
failed validation.
"""

import pytest
from typer.testing import CliRunner

from lancedb_robotics.cli import app

runner = CliRunner()


@pytest.fixture
def lake_path(tmp_path):
    path = tmp_path / "robot.lance"
    result = runner.invoke(app, ["lake", "init", "--lake", str(path)])
    assert result.exit_code == 0
    return path


def _ingest(lake_path, mcap_path):
    result = runner.invoke(app, ["ingest", "mcap", str(mcap_path), "--lake", str(lake_path)])
    assert result.exit_code == 0
    return next(line.split()[1] for line in result.output.splitlines() if line.startswith("run:"))


@pytest.fixture
def good_lake(lake_path, fixtures_dir):
    _ingest(lake_path, fixtures_dir / "sample.mcap")
    return lake_path


@pytest.fixture
def mixed_lake(lake_path, fixtures_dir):
    good = _ingest(lake_path, fixtures_dir / "sample.mcap")
    bad = _ingest(lake_path, fixtures_dir / "incomplete.mcap")
    return lake_path, good, bad


def _validate(lake_path, *extra):
    return runner.invoke(
        app, ["quality", "validate", "--lake", str(lake_path), "--profile", "demo", *extra]
    )


def test_all_passing_lake_exits_zero(good_lake):
    result = _validate(good_lake)
    assert result.exit_code == 0
    assert "passed" in result.output
    assert "1 validated, 1 passed, 0 failed, 0 quarantined" in result.output


def test_failing_run_exits_two_and_reports_details(mixed_lake):
    lake_path, good, bad = mixed_lake
    result = _validate(lake_path)
    assert result.exit_code == 2
    assert f"run {bad}: FAILED (quarantined)" in result.output
    assert f"run {good}: passed" in result.output
    assert "required-topics: failed" in result.output
    assert "/camera/front" in result.output
    assert "2 validated, 1 passed, 1 failed, 1 quarantined" in result.output


def test_validation_writes_quarantine_back_to_lake(mixed_lake):
    lake_path, _good, bad = mixed_lake
    _validate(lake_path)
    from lancedb_robotics.lake import Lake
    from lancedb_robotics.quality import quarantined_run_ids

    assert quarantined_run_ids(Lake.open(lake_path)) == [bad]


def test_dry_run_reports_without_writing(mixed_lake):
    lake_path, _good, _bad = mixed_lake
    result = _validate(lake_path, "--dry-run")
    assert result.exit_code == 2
    assert "dry-run" in result.output
    from lancedb_robotics.lake import Lake
    from lancedb_robotics.quality import quarantined_run_ids

    lake = Lake.open(lake_path)
    assert quarantined_run_ids(lake) == []
    for row in lake.table("runs").to_arrow().to_pylist():
        assert not row["quality_flags"]


def test_single_run_selection(mixed_lake):
    lake_path, good, _bad = mixed_lake
    result = _validate(lake_path, "--run", good)
    assert result.exit_code == 0
    assert "1 validated, 1 passed" in result.output


def test_unknown_run_exits_one(good_lake):
    result = _validate(good_lake, "--run", "run-nope")
    assert result.exit_code == 1
    assert "error" in result.output


def test_missing_lake_exits_one(tmp_path):
    result = _validate(tmp_path / "nope.lance")
    assert result.exit_code == 1
    assert "lake init" in result.output


def test_unknown_profile_exits_one(good_lake):
    result = runner.invoke(
        app, ["quality", "validate", "--lake", str(good_lake), "--profile", "nope"]
    )
    assert result.exit_code == 1
    assert "demo" in result.output  # error names the known profiles


def test_empty_lake_validates_zero_runs(lake_path):
    result = _validate(lake_path)
    assert result.exit_code == 0
    assert "0 validated" in result.output


def test_help_documents_exit_codes():
    result = runner.invoke(app, ["quality", "validate", "--help"])
    assert result.exit_code == 0
    assert "Exit codes" in result.output
