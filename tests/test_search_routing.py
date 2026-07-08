"""Query-time ANN routing controls for scenario search (backlog 0185).

`search_scenarios` gains the same routing contract 0187 gave `search_observations`
(decision 20260702T043600Z): `route="auto"` rides the index with tunable
`nprobes`/`refine_factor`, `route="exact"` bypasses any index (correct by
construction), `route="ann"` requires one. The CLI threads `--nprobes/
--refine-factor/--exact/--ann` through `search vector|hybrid`. Fixtures use
deterministic distinct unit vectors (the `test_indexing.py` pattern) so exact
top-k has an unambiguous numpy ground truth.
"""

import hashlib
import math

import pyarrow as pa
import pytest
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.enrich import EmbeddingProvider, ScenarioContext, enrich_scenarios
from lancedb_robotics.indexing import IndexSpec, build_vector_index, has_vector_index
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import SCENARIOS_SCHEMA
from lancedb_robotics.search import VECTOR, SearchError, search_scenarios

runner = CliRunner()

_DIM = 16
_QUERY = "find the anomaly window"


class _SeededVectorProvider(EmbeddingProvider):
    """Deterministic distinct unit vectors per seed (test_indexing.py pattern)."""

    name = "seeded-routing-test"

    def __init__(self, dimension: int = _DIM) -> None:
        self.dimension = dimension

    def embed(self, ctx: ScenarioContext) -> list[float]:
        return self._vector(ctx.scenario_id)

    def embed_text(self, text: str) -> list[float]:
        return self._vector(f"query::{text}")

    def _vector(self, seed: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
            for offset in range(0, len(digest), 4):
                if len(values) >= self.dimension:
                    break
                word = int.from_bytes(digest[offset : offset + 4], "big")
                values.append(word / 0xFFFFFFFF * 2.0 - 1.0)
            counter += 1
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]


def _seeded_lake(path, *, n: int = 300) -> Lake:
    lake = Lake.init(path)
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": f"scn-{i:04d}",
                    "run_id": "run-synthetic",
                    "start_time_ns": i * 1000,
                    "end_time_ns": i * 1000 + 500,
                    "window_ns": 500,
                    "is_partial": False,
                    "topics": ["/imu"],
                    "observation_ids": [f"obs-{i}"],
                    "observation_count": 1,
                }
                for i in range(n)
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )
    enrich_scenarios(lake, embedding_provider=_SeededVectorProvider(), auto_index=False)
    return lake


def _l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _ground_truth_top(lake: Lake, k: int) -> list[str]:
    provider = _SeededVectorProvider()
    query_vector = provider.embed_text(_QUERY)
    table = lake.table("scenarios").to_arrow()
    pairs = zip(
        table["scenario_id"].to_pylist(), table["embedding"].to_pylist(), strict=True
    )
    ranked = sorted(pairs, key=lambda sv: (_l2(query_vector, sv[1]), sv[0]))
    return [scenario_id for scenario_id, _ in ranked[:k]]


def _search_ids(lake: Lake, *, route: str, **kwargs) -> list[str]:
    results = search_scenarios(
        lake,
        mode=VECTOR,
        query=_QUERY,
        limit=5,
        embedding_provider=_SeededVectorProvider(),
        route=route,
        diversify=False,
        **kwargs,
    )
    return [r.scenario_id for r in results]


def test_exact_matches_brute_force_with_and_without_index(tmp_path):
    """`route="exact"` returns the numpy ground truth, index present or not (AC)."""
    lake = _seeded_lake(tmp_path / "robot.lance")
    truth = _ground_truth_top(lake, 5)

    before_index = _search_ids(lake, route="exact")
    assert before_index == truth

    result = build_vector_index(
        lake, table="scenarios", column="embedding", spec=IndexSpec(index_type="IVF_PQ")
    )
    assert result.status == "built"
    assert has_vector_index(lake.table("scenarios"), "embedding")

    after_index = _search_ids(lake, route="exact")
    assert after_index == truth  # the index is bypassed, not consulted


def test_auto_with_default_refine_matches_exact_on_indexed_lake(tmp_path):
    """The default route keeps its engine defaults (refine_factor=50) -- on this
    fixture that is enough for the indexed path to agree with ground truth."""
    lake = _seeded_lake(tmp_path / "robot.lance")
    build_vector_index(
        lake, table="scenarios", column="embedding", spec=IndexSpec(index_type="IVF_FLAT")
    )
    assert _search_ids(lake, route="auto") == _ground_truth_top(lake, 5)


def test_ann_route_requires_index(tmp_path):
    lake = _seeded_lake(tmp_path / "robot.lance", n=20)
    with pytest.raises(SearchError, match="no vector index"):
        _search_ids(lake, route="ann")


def test_unknown_route_rejected(tmp_path):
    lake = _seeded_lake(tmp_path / "robot.lance", n=20)
    with pytest.raises(SearchError, match="unknown route"):
        _search_ids(lake, route="fastest")


def test_nprobes_refine_and_bypass_reach_the_query_builder(tmp_path, monkeypatch):
    """The knobs are real plumbing: capture what reaches the LanceDB builder."""
    from lancedb.query import LanceVectorQueryBuilder

    lake = _seeded_lake(tmp_path / "robot.lance")
    build_vector_index(
        lake, table="scenarios", column="embedding", spec=IndexSpec(index_type="IVF_PQ")
    )

    seen = {}
    original_nprobes = LanceVectorQueryBuilder.nprobes
    original_refine = LanceVectorQueryBuilder.refine_factor
    original_bypass = LanceVectorQueryBuilder.bypass_vector_index

    def spy_nprobes(self, value):
        seen["nprobes"] = value
        return original_nprobes(self, value)

    def spy_refine(self, value):
        seen["refine_factor"] = value
        return original_refine(self, value)

    def spy_bypass(self):
        seen["bypassed"] = True
        return original_bypass(self)

    monkeypatch.setattr(LanceVectorQueryBuilder, "nprobes", spy_nprobes)
    monkeypatch.setattr(LanceVectorQueryBuilder, "refine_factor", spy_refine)
    monkeypatch.setattr(LanceVectorQueryBuilder, "bypass_vector_index", spy_bypass)

    _search_ids(lake, route="auto", nprobes=7, refine_factor=3)
    assert seen == {"nprobes": 7, "refine_factor": 3}

    seen.clear()
    _search_ids(lake, route="exact")
    assert seen == {"bypassed": True}  # no probe/refine tuning on the bypass path


# --- CLI threading -----------------------------------------------------------


def test_cli_vector_threads_flags_into_search_scenarios(tmp_path, monkeypatch):
    import lancedb_robotics.search as search_module

    lake_path = tmp_path / "robot.lance"
    _seeded_lake(lake_path, n=20)

    captured = {}

    def fake_search_scenarios(lake, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(search_module, "search_scenarios", fake_search_scenarios)
    result = runner.invoke(
        app,
        ["search", "vector", _QUERY, "--lake", str(lake_path), "--no-record",
         "--nprobes", "7", "--refine-factor", "3", "--exact"],
    )

    assert result.exit_code == 0, result.output
    assert captured["nprobes"] == 7
    assert captured["refine_factor"] == 3
    assert captured["route"] == "exact"


def test_cli_hybrid_accepts_routing_flags(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _seeded_lake(lake_path, n=20)  # enrich builds the FTS index hybrid needs

    result = runner.invoke(
        app,
        ["search", "hybrid", _QUERY, "--lake", str(lake_path), "--no-record",
         "--exact", "--limit", "3"],
    )

    assert result.exit_code == 0, result.output
    assert "results:" in result.output


def test_cli_exact_and_ann_are_mutually_exclusive(tmp_path):
    result = runner.invoke(
        app,
        ["search", "vector", _QUERY, "--lake", str(tmp_path / "x.lance"),
         "--exact", "--ann"],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_cli_ann_without_index_errors(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _seeded_lake(lake_path, n=20)  # un-indexed

    result = runner.invoke(
        app,
        ["search", "vector", _QUERY, "--lake", str(lake_path), "--no-record", "--ann"],
    )

    assert result.exit_code == 1
    assert "no vector index" in result.output
