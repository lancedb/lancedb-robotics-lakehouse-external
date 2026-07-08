"""CLI tests for `lancedb-robotics scenarios enrich` (backlog 0007)."""

from typer.testing import CliRunner

from lancedb_robotics.cli import app

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


def test_scenarios_enrich_reports_providers_and_count(tmp_path, fixtures_dir):
    lake_path = _windowed_lake(tmp_path, fixtures_dir)

    result = runner.invoke(app, ["scenarios", "enrich", "--lake", str(lake_path)])

    assert result.exit_code == 0
    assert "caption provider: demo-template-v1" in result.output
    assert "embedding provider: demo-hash-v1 (dim 16)" in result.output
    assert "scenarios enriched: 2" in result.output
    assert "transform: tfm-enrich-" in result.output
    assert "fts index: built" in result.output


def test_scenarios_enrich_can_skip_embeddings(tmp_path, fixtures_dir):
    lake_path = _windowed_lake(tmp_path, fixtures_dir)

    result = runner.invoke(app, ["scenarios", "enrich", "--lake", str(lake_path), "--no-embed"])

    assert result.exit_code == 0
    assert "embedding provider: none" in result.output
    assert "scenarios enriched: 2" in result.output
    assert "fts index: built" in result.output


def test_scenarios_enrich_selects_caption_provider(tmp_path, fixtures_dir):
    lake_path = _windowed_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "scenarios",
            "enrich",
            "--lake",
            str(lake_path),
            "--caption-provider",
            "image-stats",
            "--no-embed",
        ],
    )

    assert result.exit_code == 0
    assert "caption provider: image-statistics-v1" in result.output
    assert "embedding provider: none" in result.output


def test_scenarios_enrich_caption_provider_falls_back_with_warning(tmp_path, fixtures_dir):
    lake_path = _windowed_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app, ["scenarios", "enrich", "--lake", str(lake_path), "--caption-provider", "vlm-api"]
    )

    assert result.exit_code == 0
    assert "warning:" in result.stderr
    assert "vlm-api" in result.stderr
    assert "caption provider: demo-template-v1" in result.output


def test_scenarios_enrich_strict_fails_when_provider_unavailable(tmp_path, fixtures_dir):
    # BUG-11: --strict must fail instead of silently degrading to demo when a
    # requested real provider is unavailable (vlm-api needs an endpoint absent here).
    lake_path = _windowed_lake(tmp_path, fixtures_dir)

    strict = runner.invoke(
        app,
        ["scenarios", "enrich", "--lake", str(lake_path), "--caption-provider", "vlm-api", "--strict"],
    )
    assert strict.exit_code == 1
    assert "error" in strict.output.lower()

    # contrast: without --strict the same provider degrades to demo (fail-open default)
    lax = runner.invoke(
        app, ["scenarios", "enrich", "--lake", str(lake_path), "--caption-provider", "vlm-api"]
    )
    assert lax.exit_code == 0
    assert "caption provider: demo-template-v1" in lax.output


def test_scenarios_enrich_writes_named_embedding_column(tmp_path, fixtures_dir):
    # BUG-08: --embedding-column lets a second vector space coexist with 'embedding'.
    lake_path = _windowed_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        ["scenarios", "enrich", "--lake", str(lake_path),
         "--embedding-column", "embedding_alt", "--dimension", "32"],
    )

    assert result.exit_code == 0
    assert "embedding column: embedding_alt" in result.output


def test_scenarios_enrich_dimension_change_requires_replace(tmp_path, fixtures_dir):
    # BUG-08: a dimension change on the same column fails fast without
    # --replace-embedding, and the error signposts both escape hatches.
    lake_path = _windowed_lake(tmp_path, fixtures_dir)
    assert (
        runner.invoke(
            app, ["scenarios", "enrich", "--lake", str(lake_path), "--dimension", "16"]
        ).exit_code
        == 0
    )

    failed = runner.invoke(
        app, ["scenarios", "enrich", "--lake", str(lake_path), "--dimension", "32"]
    )
    assert failed.exit_code == 1
    assert "--replace-embedding" in failed.output
    assert "--embedding-column" in failed.output

    migrated = runner.invoke(
        app,
        ["scenarios", "enrich", "--lake", str(lake_path),
         "--dimension", "32", "--replace-embedding"],
    )
    assert migrated.exit_code == 0
    assert "(dim 32)" in migrated.output


def test_scenarios_enrich_missing_lake_exits_one(tmp_path):
    result = runner.invoke(app, ["scenarios", "enrich", "--lake", str(tmp_path / "nope.lance")])

    assert result.exit_code == 1
    assert "lake init" in result.output


def test_scenarios_enrich_without_windows_exits_one(tmp_path, fixtures_dir):
    lake_path = tmp_path / "robot.lance"
    assert runner.invoke(app, ["lake", "init", "--lake", str(lake_path)]).exit_code == 0
    assert (
        runner.invoke(
            app, ["ingest", "mcap", str(fixtures_dir / "sample.mcap"), "--lake", str(lake_path)]
        ).exit_code
        == 0
    )

    result = runner.invoke(app, ["scenarios", "enrich", "--lake", str(lake_path)])

    assert result.exit_code == 1
    assert "scenarios" in result.output
