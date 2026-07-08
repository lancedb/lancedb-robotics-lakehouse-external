"""CLI tests for the 0021 embedding-provider and index flags on `scenarios`."""

import pytest
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.embeddings import embedding_extra_available

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
            app, ["scenarios", "create", "--lake", str(lake_path), "--window", "100ms"]
        ).exit_code
        == 0
    )
    return lake_path


def test_enrich_selects_a_real_content_provider(tmp_path, fixtures_dir):
    lake_path = _windowed_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app, ["scenarios", "enrich", "--lake", str(lake_path), "--provider", "hashed-text"]
    )

    assert result.exit_code == 0
    assert "embedding provider: hashed-text-v1 (dim 16)" in result.output
    assert "warning:" not in result.stderr  # no fallback for a dependency-free provider


def test_enrich_with_missing_extra_falls_back_with_warning(tmp_path, fixtures_dir):
    # The 'embeddings' extra is absent in CI: --provider clip degrades to demo
    # with a loud warning routed to stderr, and the run still succeeds.
    if embedding_extra_available("clip"):
        pytest.skip("clip extra is installed; the missing-extra fallback path does not apply")
    lake_path = _windowed_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app, ["scenarios", "enrich", "--lake", str(lake_path), "--provider", "clip"]
    )

    assert result.exit_code == 0
    assert "warning:" in result.stderr
    assert "embeddings" in result.stderr
    assert "embedding provider: demo-hash-v1" in result.output


def test_enrich_index_flag_reports_skip_below_floor(tmp_path, fixtures_dir):
    lake_path = _windowed_lake(tmp_path, fixtures_dir)

    result = runner.invoke(app, ["scenarios", "enrich", "--lake", str(lake_path), "--index"])

    assert result.exit_code == 0
    assert "vector index: skipped" in result.output


def test_scenarios_index_command_runs(tmp_path, fixtures_dir):
    lake_path = _windowed_lake(tmp_path, fixtures_dir)
    assert runner.invoke(app, ["scenarios", "enrich", "--lake", str(lake_path)]).exit_code == 0

    result = runner.invoke(app, ["scenarios", "index", "--lake", str(lake_path)])

    assert result.exit_code == 0
    assert "table: scenarios" in result.output
    assert "column: embedding" in result.output
    assert "vector index: skipped" in result.output  # 2 rows < 256 training floor


def test_scenarios_index_fts_command_runs(tmp_path, fixtures_dir):
    lake_path = _windowed_lake(tmp_path, fixtures_dir)
    assert (
        runner.invoke(
            app,
            ["scenarios", "enrich", "--lake", str(lake_path), "--no-fts-index"],
        ).exit_code
        == 0
    )

    result = runner.invoke(app, ["scenarios", "index-fts", "--lake", str(lake_path)])

    assert result.exit_code == 0
    assert "table: scenarios" in result.output
    assert "column: summary" in result.output
    assert "fts index: built" in result.output
