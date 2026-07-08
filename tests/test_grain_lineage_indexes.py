"""Grain + lineage scalar predicate indexes (backlog 0181 / BUG-15).

The scalar-index lifecycle used to stop at aligned_*/curation_*, leaving the two
largest tables in any real lake -- observations (run_id/observation_id/topic/
timestamp_ns) and the lineage graph (edge endpoints, artifact ids) -- unindexed,
so the filters behind BUG-10/BUG-13 were full scans. These tests pin that a normal
ingest indexes the observations grain, a lineage refresh indexes the graph,
`lake maintain` upgrades a pre-existing un-indexed lake, the index types follow the
cardinality guidance (BITMAP for low-card topic/edge_type, BTREE otherwise), and
appended rows stay covered.
"""

import pytest

from lancedb_robotics.indexing import (
    build_lineage_predicate_indexes,
    build_observation_predicate_indexes,
)
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.maintenance import maintain_lake


@pytest.fixture
def sample_mcap(fixtures_dir):
    return fixtures_dir / "sample.mcap"


def _index_types(lake, table):
    """Map indexed column -> index type string (e.g. 'BTree', 'Bitmap')."""
    out = {}
    for info in lake.table(table).list_indices():
        cols = getattr(info, "columns", None) or []
        if cols:
            out[cols[0]] = str(getattr(info, "index_type", ""))
    return out


def test_default_ingest_indexes_observations_grain(tmp_path, sample_mcap):
    lake = Lake.init(tmp_path / "l.lance")
    report = ingest_mcap(lake, sample_mcap)
    types = _index_types(lake, "observations")
    assert set(types) == {"observation_id", "run_id", "topic", "timestamp_ns"}
    # cardinality-appropriate types: low-card topic -> BITMAP, ids/range -> BTREE
    assert "Bitmap" in types["topic"]
    assert "BTree" in types["run_id"]
    assert "BTree" in types["observation_id"]
    assert report.compaction["indexes"] == [
        "observation_id",
        "run_id",
        "timestamp_ns",
        "topic",
    ]


def test_no_index_predicates_skips_then_maintain_creates(tmp_path, sample_mcap):
    """A lake ingested without indexing (or built before 0181) is upgraded by maintain."""
    lake = Lake.init(tmp_path / "l.lance")
    ingest_mcap(lake, sample_mcap, compact=False, prune_versions=False, index_predicates=False)
    assert _index_types(lake, "observations") == {}  # nothing indexed yet
    maintain_lake(lake, tables=("observations",), compact=True, cleanup_older_than=None)
    assert set(_index_types(lake, "observations")) == {
        "observation_id",
        "run_id",
        "topic",
        "timestamp_ns",
    }


def test_lineage_refresh_indexes_graph_tables(tmp_path, sample_mcap):
    lake = Lake.init(tmp_path / "l.lance")
    ingest_mcap(lake, sample_mcap)
    lake.lineage.refresh_graph()
    edges = _index_types(lake, "lineage_edges")
    assert set(edges) == {"from_artifact_id", "to_artifact_id", "edge_type"}
    assert "BTree" in edges["from_artifact_id"]  # endpoints BTREE for IN-frontier lookups
    assert "BTree" in edges["to_artifact_id"]
    assert "Bitmap" in edges["edge_type"]  # low-card -> BITMAP
    artifacts = _index_types(lake, "lineage_artifacts")
    assert "artifact_id" in artifacts and "BTree" in artifacts["artifact_id"]


def test_run_id_filter_is_correct_and_indexed(tmp_path, sample_mcap):
    lake = Lake.init(tmp_path / "l.lance")
    report = ingest_mcap(lake, sample_mcap)
    obs = lake.table("observations")
    assert obs.count_rows(f"run_id = '{report.run_id}'") == obs.count_rows()  # single run
    # the planner picks the scalar index for the run_id filter (not a full scan)
    plan = obs.to_lance().scanner(
        filter=f"run_id = '{report.run_id}'", columns=["observation_id"]
    ).explain_plan(True)
    assert "ScalarIndex" in plan or "MaterializeIndex" in plan or "index_query" in plan.lower()


def test_appended_rows_stay_covered_by_index(tmp_path, sample_mcap, fixtures_dir):
    """A second ingest's rows are folded into the existing index (no unindexed tail)."""
    lake = Lake.init(tmp_path / "l.lance")
    ingest_mcap(lake, sample_mcap)
    ingest_mcap(lake, fixtures_dir / "records.mcap")
    ds = lake.table("observations").to_lance()
    stats = ds.stats.index_stats("run_id_idx")
    assert stats["num_indexed_rows"] == ds.count_rows()
    assert stats["num_unindexed_rows"] == 0


def test_builders_are_idempotent(tmp_path, sample_mcap):
    lake = Lake.init(tmp_path / "l.lance")
    ingest_mcap(lake, sample_mcap)  # already builds observation indexes
    again = build_observation_predicate_indexes(lake, replace=False)
    assert {r.status for r in again} == {"already_present"}
    lake.lineage.refresh_graph()
    lineage_again = build_lineage_predicate_indexes(lake, replace=False)
    assert {r.status for r in lineage_again} == {"already_present"}
