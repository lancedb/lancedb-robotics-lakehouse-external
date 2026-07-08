"""CLI tests for `lancedb-robotics search` subcommands (backlog 0008)."""

from typer.testing import CliRunner

from lancedb_robotics.cli import app

runner = CliRunner()


def _searchable_lake(tmp_path, fixtures_dir, *, enrich=True):
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
    if enrich:
        assert runner.invoke(app, ["scenarios", "enrich", "--lake", str(lake_path)]).exit_code == 0
    return lake_path


def test_search_scalar_reports_filtered_results(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app, ["search", "scalar", "--lake", str(lake_path), "--where", "observation_count >= 2"]
    )

    assert result.exit_code == 0
    assert "mode: scalar" in result.output
    assert "results: 1" in result.output
    assert "scn-" in result.output


def test_search_text_shows_scores_and_source(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)

    result = runner.invoke(app, ["search", "text", "camera", "--lake", str(lake_path)])

    assert result.exit_code == 0
    assert "mode: text" in result.output
    assert "text=" in result.output
    assert "source:" in result.output


def test_search_vector_runs(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)

    result = runner.invoke(app, ["search", "vector", "imu", "--lake", str(lake_path)])

    assert result.exit_code == 0
    assert "mode: vector" in result.output
    assert "vector_distance=" in result.output


def test_search_hybrid_shows_all_components(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir)

    result = runner.invoke(app, ["search", "hybrid", "imu observations", "--lake", str(lake_path)])

    assert result.exit_code == 0
    assert "mode: hybrid" in result.output
    assert "text=" in result.output
    assert "vector_distance=" in result.output
    assert "relevance=" in result.output
    assert "source:" in result.output


def test_search_vector_accepts_provider_and_diversify_flags(tmp_path, fixtures_dir):
    # BUG-07 wiring: --provider embeds the query in the column's space; the column
    # was enriched with the default (demo), so a matching --provider demo runs clean.
    lake_path = _searchable_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        ["search", "vector", "imu", "--lake", str(lake_path),
         "--provider", "demo", "--no-diversify"],
    )

    assert result.exit_code == 0
    assert "mode: vector" in result.output
    assert "vector_distance=" in result.output


def test_search_vector_accepts_named_column(tmp_path, fixtures_dir):
    # BUG-08: enrich a second embedding column, then query it by --column. The query
    # dimension is resolved from the selected column, so a 32-dim column embeds the
    # query at 32-dim (a default-column query would mis-size against it).
    lake_path = _searchable_lake(tmp_path, fixtures_dir)
    assert (
        runner.invoke(
            app,
            ["scenarios", "enrich", "--lake", str(lake_path),
             "--embedding-column", "embedding_alt", "--dimension", "32"],
        ).exit_code
        == 0
    )

    result = runner.invoke(
        app, ["search", "vector", "imu", "--lake", str(lake_path), "--column", "embedding_alt"]
    )

    assert result.exit_code == 0
    assert "mode: vector" in result.output
    assert "vector_distance=" in result.output


def test_search_vector_missing_named_column_exits_one(tmp_path, fixtures_dir):
    # Querying a column that was never enriched is a clean error, not a crash.
    lake_path = _searchable_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app, ["search", "vector", "imu", "--lake", str(lake_path), "--column", "missing_col"]
    )

    assert result.exit_code == 1
    assert "missing_col" in result.output


def test_search_missing_lake_exits_one(tmp_path):
    result = runner.invoke(app, ["search", "text", "imu", "--lake", str(tmp_path / "nope.lance")])

    assert result.exit_code == 1
    assert "lake init" in result.output


def test_search_vector_without_embeddings_exits_one(tmp_path, fixtures_dir):
    lake_path = _searchable_lake(tmp_path, fixtures_dir, enrich=False)

    result = runner.invoke(app, ["search", "vector", "imu", "--lake", str(lake_path)])

    assert result.exit_code == 1
    assert "enrich" in result.output


def test_search_vector_strict_fails_when_provider_unavailable(tmp_path, fixtures_dir, monkeypatch):
    # BUG-11: `search vector --strict` must refuse to embed the query with a demo
    # stand-in when the requested --provider is unavailable (a mismatched query space
    # ranks by noise). Force a model-backed provider unavailable, deterministically.
    import lancedb_robotics.embeddings as emb

    lake_path = _searchable_lake(tmp_path, fixtures_dir)  # enriched with the demo embedder

    def boom(dimension=None):
        raise emb.EmbeddingExtraMissing(
            "needs the optional dependency 'x'; install `lancedb-robotics[embeddings]`"
        )

    monkeypatch.setitem(
        emb.PROVIDER_REGISTRY,
        "sentence-transformers",
        emb.ProviderInfo(boom, requires_extra=True, modality="text"),
    )

    strict = runner.invoke(
        app,
        ["search", "vector", "imu", "--lake", str(lake_path),
         "--provider", "sentence-transformers", "--strict"],
    )
    assert strict.exit_code == 1
    assert "error" in strict.output.lower()

    # contrast: without --strict the query embeds with a demo fallback (a notice), exit 0
    lax = runner.invoke(
        app,
        ["search", "vector", "imu", "--lake", str(lake_path),
         "--provider", "sentence-transformers"],
    )
    assert lax.exit_code == 0
