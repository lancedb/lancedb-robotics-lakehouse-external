"""Lineage trace/impact frontier expansion over indexed edges (backlog 0182 / BUG-13).

`trace`/`impact` used to load the whole lineage_edges (1.29M) + lineage_artifacts
(704K) tables into Python on every call to build an in-memory adjacency index --
O(total edges), ~22s on a real graph, OOM at didi scale. The traversal now expands
one BFS level at a time, fetching only the edges incident to the current frontier
via a chunked indexed ``endpoint IN (frontier)`` read (BUG-15 BTREE) and artifacts/
executions on demand by id. These tests pin that the result is identical to a
ground-truth whole-graph BFS, that work is proportional to the *visited* subgraph
(not the whole edge set), and that the bounded-traversal filters still hold.
"""

from collections import deque

import pytest

from lancedb_robotics import lineage as lineage_mod
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake


@pytest.fixture
def graph_lake(tmp_path, fixtures_dir):
    """A two-run lineage graph (enough fan-out to exercise multi-level expansion)."""
    lake = Lake.init(tmp_path / "l.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    ingest_mcap(lake, fixtures_dir / "records.mcap")
    lake.lineage.refresh_graph()
    edges = lake.table("lineage_edges").to_arrow().to_pylist()
    artifact_ids = {
        a["artifact_id"] for a in lake.table("lineage_artifacts").to_arrow().to_pylist()
    }
    return lake, edges, artifact_ids


def _ground_truth(edges, root, direction):
    """Reachable artifact ids + edge ids from a whole-graph BFS (no index, no filter)."""
    source_key = "to_artifact_id" if direction == "upstream" else "from_artifact_id"
    next_key = "from_artifact_id" if direction == "upstream" else "to_artifact_id"
    adjacency: dict[str, list[dict]] = {}
    for edge in edges:
        adjacency.setdefault(edge[source_key], []).append(edge)
    seen, seen_edges, queue = {root}, set(), deque([root])
    while queue:
        for edge in adjacency.get(queue.popleft(), ()):
            seen_edges.add(edge["edge_id"])
            nxt = edge[next_key]
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen, seen_edges


def _downstream_reach(edges, root):
    return len(_ground_truth(edges, root, "downstream")[0])


@pytest.mark.parametrize("direction", ["downstream", "upstream"])
def test_traversal_matches_ground_truth(graph_lake, direction):
    lake, edges, artifact_ids = graph_lake
    roots = sorted({e["from_artifact_id"] for e in edges} | {e["to_artifact_id"] for e in edges})
    fn = lake.lineage.impact if direction == "downstream" else lake.lineage.trace
    for root in roots:
        graph = fn(root)
        exp_artifacts, exp_edges = _ground_truth(edges, root, direction)
        assert {a["artifact_id"] for a in graph.artifacts} == (exp_artifacts & artifact_ids)
        assert {e["edge_id"] for e in graph.edges} == exp_edges


def test_traversal_reads_only_the_visited_subgraph(graph_lake, monkeypatch):
    """The per-level edge fetch returns work proportional to the visited subgraph."""
    lake, edges, _ = graph_lake
    root = max((e["from_artifact_id"] for e in edges), key=lambda r: _downstream_reach(edges, r))
    fetched: list[int] = []
    original = lineage_mod._fetch_edges_incident

    def spy(*args, **kwargs):
        rows = original(*args, **kwargs)
        fetched.append(len(rows))
        return rows

    monkeypatch.setattr(lineage_mod, "_fetch_edges_incident", spy)
    graph = lake.lineage.impact(root)
    # each incident edge is read at most once per level it borders; total stays bounded
    # by the visited subgraph and is strictly less than the whole edge table.
    assert sum(fetched) <= 2 * len(graph.edges) + len(graph.artifacts) + 1
    assert sum(fetched) < len(edges)
    assert len(graph.edges) > 1  # this root actually fans out (meaningful assertion)


def test_max_depth_bounds_expansion(graph_lake):
    lake, edges, _ = graph_lake
    root = max((e["from_artifact_id"] for e in edges), key=lambda r: _downstream_reach(edges, r))
    full = lake.lineage.impact(root)
    shallow = lake.lineage.impact(root, max_depth=1)
    assert len(shallow.artifacts) <= len(full.artifacts)
    assert len(shallow.edges) <= len(full.edges)
    with pytest.raises(lineage_mod.LineageError):
        lake.lineage.impact(root, max_depth=-1)


def test_edge_type_filter_pushed_into_indexed_read(graph_lake):
    lake, edges, _ = graph_lake
    root = max((e["from_artifact_id"] for e in edges), key=lambda r: _downstream_reach(edges, r))
    full = lake.lineage.impact(root)
    edge_types = sorted({e["edge_type"] for e in full.edges})
    if len(edge_types) > 1:
        keep = edge_types[0]
        filtered = lake.lineage.impact(root, edge_types=[keep])
        assert {e["edge_type"] for e in filtered.edges} <= {keep}


def test_direct_artifact_id_resolves_without_full_resolver_scan(graph_lake, monkeypatch):
    """A direct artifact_id handle uses the indexed fast path, never the full table load."""
    lake, edges, artifact_ids = graph_lake
    root = edges[0]["from_artifact_id"]
    # The resolver's slow path loads every lineage_artifacts row; a direct handle must
    # not reach it. Spy on the chunked-IN reader the fast path uses.
    used_fast_path = {"hit": False}
    original = lineage_mod._fetch_rows_by_id_in

    def spy(lake_, table_name, id_column, ids, **kwargs):
        if table_name == "lineage_artifacts" and id_column == "artifact_id" and list(ids) == [root]:
            used_fast_path["hit"] = True
        return original(lake_, table_name, id_column, ids, **kwargs)

    monkeypatch.setattr(lineage_mod, "_fetch_rows_by_id_in", spy)
    graph = lake.lineage.impact(root)
    assert used_fast_path["hit"]
    assert graph.root_artifact_id == root
    exp_artifacts, _ = _ground_truth(edges, root, "downstream")
    assert {a["artifact_id"] for a in graph.artifacts} == (exp_artifacts & artifact_ids)
