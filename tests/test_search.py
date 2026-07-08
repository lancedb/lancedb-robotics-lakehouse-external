"""Baseline search tests over enriched scenario rows (backlog 0008)."""

import pyarrow as pa
import pytest

from lancedb_robotics.embeddings import HashedTextEmbeddingProvider
from lancedb_robotics.enrich import CaptionProvider, ScenarioContext, enrich_scenarios
from lancedb_robotics.indexing import (
    MIN_INDEX_ROWS,
    build_fts_index,
    build_vector_index,
    has_vector_index,
    record_fts_index_transform,
    record_index_transform,
)
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.schemas import RUNS_SCHEMA, SCENARIOS_SCHEMA
from lancedb_robotics.search import (
    HYBRID,
    SCALAR,
    TEXT,
    VECTOR,
    SearchError,
    search_scenarios,
)


@pytest.fixture
def search_lake(tmp_path, fixtures_dir):
    """A lake windowed at 50ms (4 topic-varied scenarios) and enriched."""
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=50_000_000)
    enrich_scenarios(lake)
    return lake


@pytest.fixture
def windowed_only_lake(tmp_path, fixtures_dir):
    """Windowed but NOT enriched: no summary/embedding to search."""
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=50_000_000)
    return lake


@pytest.fixture
def search_lake_without_fts(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=50_000_000)
    enrich_scenarios(lake, fts_index=False)
    return lake


def _rows_by_id(lake):
    return {r["scenario_id"]: r for r in lake.table("scenarios").to_arrow().to_pylist()}


# --- scalar -----------------------------------------------------------------


def test_scalar_search_with_no_filter_returns_all_in_stable_order(search_lake):
    results = search_scenarios(search_lake, mode=SCALAR)

    assert len(results) == 4
    keys = [(r.start_time_ns, r.scenario_id) for r in results]
    assert keys == sorted(keys)


def test_scalar_search_filters_by_metadata(search_lake):
    results = search_scenarios(search_lake, mode=SCALAR, where="observation_count >= 2")

    by_id = _rows_by_id(search_lake)
    assert len(results) == 1
    assert by_id[results[0].scenario_id]["observation_count"] >= 2


# --- text -------------------------------------------------------------------


def test_text_search_matches_summaries(search_lake):
    results = search_scenarios(search_lake, mode=TEXT, query="camera")

    assert results
    for r in results:
        assert "camera" in r.summary.lower()
        assert r.text_score is not None
        assert r.vector_distance is None


def test_text_search_is_deterministic(search_lake):
    first = [r.scenario_id for r in search_scenarios(search_lake, mode=TEXT, query="imu")]
    second = [r.scenario_id for r in search_scenarios(search_lake, mode=TEXT, query="imu")]
    assert first == second


def test_text_ranking_breaks_score_ties_by_scenario_id(search_lake):
    # Two scenarios share an identical "/imu" summary -> identical BM25 score.
    # Ordering must still be deterministic: score desc, then scenario_id asc.
    results = search_scenarios(search_lake, mode=TEXT, query="imu")
    ranked = [(r.text_score, r.scenario_id) for r in results]
    assert ranked == sorted(ranked, key=lambda pair: (-pair[0], pair[1]))


def test_text_search_requires_managed_fts_index(search_lake_without_fts):
    with pytest.raises(SearchError, match="persistent FTS index"):
        search_scenarios(search_lake_without_fts, mode=TEXT, query="camera")


def test_text_search_reuses_prebuilt_fts_index(search_lake, monkeypatch):
    table_cls = type(search_lake.table("scenarios"))
    calls = 0

    def spy(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("search must not rebuild the FTS index")

    monkeypatch.setattr(table_cls, "create_fts_index", spy)

    for _ in range(3):
        assert search_scenarios(search_lake, mode=TEXT, query="imu")
    assert calls == 0


def test_text_search_rejects_stale_fts_index_until_refreshed(search_lake):
    table = search_lake.table("scenarios")
    row = table.to_arrow().to_pylist()[0]
    row["scenario_id"] = "scn-new"
    row["summary"] = "fresh pedestrian crosswalk caption"
    table.add(pa.Table.from_pylist([row], schema=table.schema))

    with pytest.raises(SearchError, match="stale"):
        search_scenarios(search_lake, mode=TEXT, query="crosswalk")

    refreshed = build_fts_index(search_lake, table="scenarios", column="summary")
    record_fts_index_transform(search_lake, refreshed)
    assert search_scenarios(search_lake, mode=TEXT, query="crosswalk")[0].scenario_id == "scn-new"


class _RunTokenCaptionProvider(CaptionProvider):
    name = "run-token-captions"

    def caption(self, ctx: ScenarioContext) -> str:
        token = "secondlogtoken" if ctx.topics == ("/imu",) else "firstlogtoken"
        return f"{token} {ctx.run_id} {ctx.scenario_id}"


def test_incremental_append_refreshes_search_indexes_for_new_rows(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "incremental.lance")
    first = ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    provider = HashedTextEmbeddingProvider(dimension=64)
    enrich_scenarios(
        lake,
        caption_provider=_RunTokenCaptionProvider(),
        embedding_provider=provider,
    )
    before = {row["scenario_id"]: row for row in lake.table("scenarios").to_arrow().to_pylist()}

    second = ingest_mcap(lake, fixtures_dir / "records.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    report = enrich_scenarios(
        lake,
        caption_provider=_RunTokenCaptionProvider(),
        embedding_provider=provider,
    )

    assert first.run_id != second.run_id
    assert report.scenarios_enriched == 2
    after = {row["scenario_id"]: row for row in lake.table("scenarios").to_arrow().to_pylist()}
    for scenario_id, row in before.items():
        assert after[scenario_id] == row

    text_hits = search_scenarios(lake, mode=TEXT, query="secondlogtoken")
    assert text_hits
    assert {hit.run_id for hit in text_hits} == {second.run_id}

    vector_hits = search_scenarios(
        lake,
        mode=VECTOR,
        query="secondlogtoken",
        embedding_provider=provider,
        limit=2,
    )
    assert vector_hits
    assert vector_hits[0].run_id == second.run_id


# --- vector -----------------------------------------------------------------


def test_vector_search_orders_by_distance(search_lake):
    results = search_scenarios(search_lake, mode=VECTOR, query="imu")

    assert len(results) == 4
    distances = [r.vector_distance for r in results]
    assert all(d is not None for d in distances)
    assert distances == sorted(distances)
    assert all(r.text_score is None for r in results)


def test_vector_search_is_deterministic(search_lake):
    first = [
        (r.scenario_id, r.vector_distance)
        for r in search_scenarios(search_lake, mode=VECTOR, query="imu")
    ]
    second = [
        (r.scenario_id, r.vector_distance)
        for r in search_scenarios(search_lake, mode=VECTOR, query="imu")
    ]
    assert first == second


# --- hybrid -----------------------------------------------------------------


def test_hybrid_search_exposes_transparent_score_components(search_lake):
    results = search_scenarios(search_lake, mode=HYBRID, query="imu observations")

    assert results
    for r in results:
        assert r.text_score is not None
        assert r.vector_distance is not None
        assert r.relevance_score is not None
    relevances = [r.relevance_score for r in results]
    assert relevances == sorted(relevances, reverse=True)


def test_hybrid_search_is_deterministic(search_lake):
    first = [r.scenario_id for r in search_scenarios(search_lake, mode=HYBRID, query="imu")]
    second = [r.scenario_id for r in search_scenarios(search_lake, mode=HYBRID, query="imu")]
    assert first == second


def test_vector_and_hybrid_name_column_when_lake_has_multiple_embeddings(tmp_path):
    """Multi-embedding lakes must search the requested vector column explicitly."""
    lake = Lake.init(tmp_path / "multivec.lance")
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": "run-multivec", "run_kind": "drive", "raw_uri": "/multivec.mcap"}],
            schema=RUNS_SCHEMA,
        )
    )
    rows = [
        {
            "scenario_id": f"scn-{i:04d}",
            "run_id": "run-multivec",
            "start_time_ns": i * 10,
            "end_time_ns": i * 10 + 10,
            "window_ns": 10,
            "is_partial": False,
            "topics": ["/imu"],
            "observation_ids": [],
            "observation_count": 1,
        }
        for i in range(MIN_INDEX_ROWS)
    ]
    lake.table("scenarios").add(pa.Table.from_pylist(rows, schema=SCENARIOS_SCHEMA))

    provider = HashedTextEmbeddingProvider(dimension=64)
    enrich_scenarios(
        lake,
        embedding_provider=provider,
        embedding_column="embedding",
        auto_index=False,
        fts_index=True,
    )
    enrich_scenarios(
        lake,
        embedding_provider=provider,
        embedding_column="embedding_clip",
        auto_index=False,
        fts_index=False,
    )
    record_index_transform(lake, build_vector_index(lake, table="scenarios", column="embedding"))
    record_fts_index_transform(lake, build_fts_index(lake, table="scenarios", column="summary"))

    scenarios = lake.table("scenarios")
    assert {"embedding", "embedding_clip"} <= {field.name for field in scenarios.schema}
    assert has_vector_index(scenarios, "embedding")

    vector_hits = search_scenarios(
        lake,
        mode=VECTOR,
        query="imu",
        embedding_column="embedding",
        embedding_provider=provider,
    )
    hybrid_hits = search_scenarios(
        lake,
        mode=HYBRID,
        query="imu observations",
        embedding_column="embedding",
        embedding_provider=provider,
    )

    assert vector_hits
    assert hybrid_hits
    assert all(hit.relevance_score is not None for hit in hybrid_hits)


class _FixtureCaptionProvider(CaptionProvider):
    name = "fixture-content-captions"

    def caption(self, ctx: ScenarioContext) -> str:
        if ctx.scenario_id == "scn-crosswalk":
            return "pedestrian near a crosswalk in a bright camera scene"
        return "empty warehouse aisle with no pedestrians"


def test_hybrid_search_ranks_content_matching_caption_first(tmp_path):
    lake = Lake.init(tmp_path / "content.lance")
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": "run-content", "run_kind": "drive", "raw_uri": "/content.mcap"}],
            schema=RUNS_SCHEMA,
        )
    )
    rows = [
        {
            "scenario_id": "scn-crosswalk",
            "run_id": "run-content",
            "start_time_ns": 0,
            "end_time_ns": 10,
            "window_ns": 10,
            "is_partial": False,
            "topics": ["/camera/front"],
            "observation_ids": [],
            "observation_count": 1,
        },
        {
            "scenario_id": "scn-warehouse",
            "run_id": "run-content",
            "start_time_ns": 10,
            "end_time_ns": 20,
            "window_ns": 10,
            "is_partial": False,
            "topics": ["/camera/front"],
            "observation_ids": [],
            "observation_count": 1,
        },
    ]
    lake.table("scenarios").add(pa.Table.from_pylist(rows, schema=SCENARIOS_SCHEMA))
    enrich_scenarios(
        lake,
        caption_provider=_FixtureCaptionProvider(),
        embedding_provider=HashedTextEmbeddingProvider(dimension=64),
    )

    results = search_scenarios(
        lake,
        mode=HYBRID,
        query="pedestrian crosswalk",
        embedding_provider=HashedTextEmbeddingProvider(dimension=64),
    )

    assert results[0].scenario_id == "scn-crosswalk"
    assert results[0].text_score is not None
    assert results[0].vector_distance is not None
    assert results[0].relevance_score is not None


# --- result shape, limits, and errors --------------------------------------


def test_results_carry_ids_window_and_source_links(search_lake):
    result = search_scenarios(search_lake, mode=HYBRID, query="imu")[0]

    assert result.scenario_id.startswith("scn-")
    assert result.run_id.startswith("run-")
    assert result.end_time_ns >= result.start_time_ns
    assert result.topics
    assert result.source_uri and result.source_uri.endswith("sample.mcap")


def test_limit_caps_result_count(search_lake):
    assert len(search_scenarios(search_lake, mode=VECTOR, query="imu", limit=1)) == 1


def test_text_vector_hybrid_require_a_query(search_lake):
    for mode in (TEXT, VECTOR, HYBRID):
        with pytest.raises(SearchError):
            search_scenarios(search_lake, mode=mode)


def test_vector_and_hybrid_require_embeddings(windowed_only_lake):
    for mode in (VECTOR, HYBRID):
        with pytest.raises(SearchError):
            search_scenarios(windowed_only_lake, mode=mode, query="imu")


def test_unknown_mode_raises(search_lake):
    with pytest.raises(SearchError):
        search_scenarios(search_lake, mode="fuzzy", query="imu")


# --- scene diversification: top-k spans distinct scenes (BUG-07) ------------


def _two_scene_lake(tmp_path):
    """Two scenes (runs), each with two window-scenarios sharing one description.

    Because a run's windows share its ``scene-info.description``, they get an
    identical dense ``embed_text`` and embed to the *same* vector -- the exact
    within-scene near-duplicate that floods the top-k once the embedded text is
    description-forward (BUG-07).
    """
    lake = Lake.init(tmp_path / "scenes.lance")
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-alpha",
                    "run_kind": "drive",
                    "raw_uri": "/alpha.mcap",
                    "metadata": [
                        {"key": "scene-info.description",
                         "value": "alpha quiet parking lot with parked cars"}
                    ],
                },
                {
                    "run_id": "run-beta",
                    "run_kind": "drive",
                    "raw_uri": "/beta.mcap",
                    "metadata": [
                        {"key": "scene-info.description",
                         "value": "beta busy highway tunnel at night"}
                    ],
                },
            ],
            schema=RUNS_SCHEMA,
        )
    )
    scenarios = [
        {
            "scenario_id": f"{run_id}-w{w}",
            "run_id": run_id,
            "start_time_ns": w * 10,
            "end_time_ns": w * 10 + 10,
            "window_ns": 10,
            "is_partial": False,
            "topics": ["/cam/front", "/lidar/top"],
            "observation_ids": [],
            "observation_count": 0,
        }
        for run_id in ("run-alpha", "run-beta")
        for w in (0, 1)
    ]
    lake.table("scenarios").add(pa.Table.from_pylist(scenarios, schema=SCENARIOS_SCHEMA))
    return lake


def test_vector_search_diversifies_near_duplicate_scene_windows(tmp_path):
    lake = _two_scene_lake(tmp_path)
    provider = HashedTextEmbeddingProvider(dimension=64)
    enrich_scenarios(lake, embedding_provider=provider)

    results = search_scenarios(
        lake, mode=VECTOR, query="parking lot parked cars", embedding_provider=provider
    )
    # Top-k spans distinct scenes: one representative window per scene, alpha first.
    assert [r.run_id for r in results] == ["run-alpha", "run-beta"]


def test_vector_search_without_diversify_returns_every_window(tmp_path):
    lake = _two_scene_lake(tmp_path)
    provider = HashedTextEmbeddingProvider(dimension=64)
    enrich_scenarios(lake, embedding_provider=provider)

    raw = search_scenarios(
        lake,
        mode=VECTOR,
        query="parking lot parked cars",
        embedding_provider=provider,
        diversify=False,
    )
    assert len(raw) == 4  # both windows of both scenes
    # The matched scene's two near-duplicate windows lead the ranking.
    assert sorted(r.scenario_id for r in raw[:2]) == ["run-alpha-w0", "run-alpha-w1"]


def test_hybrid_search_diversifies_near_duplicate_scene_windows(tmp_path):
    lake = _two_scene_lake(tmp_path)
    provider = HashedTextEmbeddingProvider(dimension=64)
    enrich_scenarios(lake, embedding_provider=provider)

    results = search_scenarios(
        lake, mode=HYBRID, query="parking lot parked cars", embedding_provider=provider
    )
    assert [r.run_id for r in results] == ["run-alpha", "run-beta"]
