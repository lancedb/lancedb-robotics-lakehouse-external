"""Optional lance-graph Cypher lineage backend spike (backlog 0099).

These tests prove the optional backend:

- degrades with an actionable missing-extra error when ``lance_graph`` is absent
  (runs regardless of install state, via monkeypatch);
- maps the canonical lineage tables into a property graph whose Cypher upstream /
  downstream reachability *matches* ``lake.lineage.trace`` / ``lake.lineage.impact``;
- reports whether lance-graph's native (CSR) traversal strategy is usable yet --
  the input to the 0097 use/reject decision;
- and emits SDK-vs-Cypher timings on a high-fan-out synthetic graph.

The parity/benchmark tests skip cleanly when the optional ``graph`` extra is not
installed (mirrors the embeddings extra convention).
"""

from __future__ import annotations

import importlib.util
import time
from datetime import UTC, datetime

import pytest

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import (
    LINEAGE_ARTIFACTS_SCHEMA,
    LINEAGE_EDGES_SCHEMA,
    LineageArtifact,
    LineageEdge,
    _replace_rows,
    artifact_id,
    snapshot_artifact_id,
)
from lancedb_robotics.lineage_graph import (
    LineageGraphBackend,
    LineageGraphExtraMissing,
    lance_graph_available,
    require_lance_graph,
)
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.writeback import ingest_model_outputs

requires_graph = pytest.mark.skipif(
    not lance_graph_available(),
    reason="optional 'graph' extra (lance-graph) is not installed",
)


def _trace_lake(tmp_path, fixtures_dir, fixture_name="sample.mcap"):
    """Baseline lineage fixture: ingest -> window -> snapshot -> model output."""

    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / fixture_name)
    create_scenario_windows(lake, window_ns=100_000_000)
    scenarios = sorted(
        lake.table("scenarios").to_arrow().to_pylist(),
        key=lambda row: (row["start_time_ns"], row["scenario_id"]),
    )
    manifest = create_snapshot(
        lake,
        name="demo-v1",
        tag="training-demo",
        scenario_ids=[row["scenario_id"] for row in scenarios],
    )
    ingest_model_outputs(
        lake,
        {
            "model_output_id": "out-regression",
            "observation_id": scenarios[0]["observation_ids"][0],
            "scenario_id": scenarios[0]["scenario_id"],
            "dataset_id": manifest.dataset_id,
            "model_version": "policy@abc123",
            "prediction": "regressed",
            "score": 0.12,
            "producer_run_id": "checkpoint-abc123",
        },
        source="trainer",
    )
    return lake, manifest


# --- missing-extra degradation (runs even when the extra IS installed) --------


def test_require_lance_graph_raises_actionable_error_when_absent(tmp_path, monkeypatch):
    """The helper raises a clear missing-extra error when lance_graph is absent."""

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(LineageGraphExtraMissing) as excinfo:
        require_lance_graph()
    message = str(excinfo.value)
    assert "lance_graph" in message
    assert "lancedb-robotics[graph]" in message

    # And the SDK surface surfaces the same actionable error, still a LineageError.
    from lancedb_robotics.lineage import LineageError

    lake = Lake.init(tmp_path / "robot.lance")
    with pytest.raises(LineageError) as sdk_exc:
        lake.lineage.cypher("MATCH (a:Artifact) RETURN a.artifact_id AS id")
    assert "lancedb-robotics[graph]" in str(sdk_exc.value)


# --- Cypher parity with the SDK traversal APIs --------------------------------


@requires_graph
def test_cypher_upstream_matches_trace(tmp_path, fixtures_dir):
    lake, manifest = _trace_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)

    report = lake.lineage.compare_cypher_traversal(snapshot_id, direction="upstream")

    assert report.matches, report.as_dict()
    assert report.sdk_ids  # the snapshot genuinely has upstream sources/versions
    # Spot-check against the raw SDK API too (roots excluded on both sides).
    sdk = {
        row["artifact_id"]
        for row in lake.lineage.trace(snapshot_id).artifacts
    } - {snapshot_id}
    assert set(report.cypher_ids) == sdk


@requires_graph
def test_cypher_downstream_matches_impact(tmp_path, fixtures_dir):
    lake, manifest = _trace_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)
    model_output_id = artifact_id(
        "row",
        table_name="model_outputs",
        row_id="out-regression",
        table_version=int(lake.table("model_outputs").version),
    )

    report = lake.lineage.compare_cypher_traversal(snapshot_id, direction="downstream")

    assert report.matches, report.as_dict()
    assert model_output_id in set(report.cypher_ids)
    sdk = {
        row["artifact_id"]
        for row in lake.lineage.impact(snapshot_id).artifacts
    } - {snapshot_id}
    assert set(report.cypher_ids) == sdk


@requires_graph
def test_cypher_convenience_query_returns_edge_properties(tmp_path, fixtures_dir):
    lake, manifest = _trace_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)

    rows = lake.lineage.cypher(
        "MATCH (a:Artifact)-[r:DEPENDS_ON]->(b:Artifact {artifact_id:$root}) "
        "RETURN a.artifact_id AS upstream, r.edge_type AS edge_type",
        parameters={"root": snapshot_id},
    )
    assert rows
    assert "selected-from" in {row["edge_type"] for row in rows}


@requires_graph
def test_deep_variable_length_path_raises_actionable_error(tmp_path, fixtures_dir):
    """Beyond lance-graph's 20-hop cap, point callers back to the unbounded SDK API."""

    from lancedb_robotics.lineage_graph import LineageGraphError

    lake, manifest = _trace_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)
    backend = LineageGraphBackend(lake)
    with pytest.raises(LineageGraphError) as excinfo:
        backend.impact_ids(snapshot_id, max_depth=40)
    message = str(excinfo.value)
    assert "20 hops" in message
    assert "lake.lineage.impact" in message


@requires_graph
def test_native_strategy_probe_reports_availability(tmp_path, fixtures_dir):
    """Records whether native/CSR traversal is usable (input to the 0097 decision)."""

    lake, _ = _trace_lake(tmp_path, fixtures_dir)
    probe = LineageGraphBackend(lake).probe_native_strategy()
    assert "LanceNative" in probe.strategy_names
    # Whatever the outcome, the probe must be conclusive with a non-empty detail.
    assert probe.detail
    if not probe.available:
        # 0.5.4 raises NotImplementedError for the native path; guard the finding.
        assert "not yet implemented" in probe.detail.lower() or "not" in probe.detail.lower()


# --- high fan-out synthetic benchmark ----------------------------------------


def _build_fanout_graph(lake, *, depth: int, branching: int) -> tuple[str, int]:
    """Write a balanced fan-out artifact tree straight into the lineage tables.

    Returns ``(root_id, downstream_node_count)``. Edges point from parent
    (upstream) -> child (downstream), matching the SDK edge convention.
    """

    now = datetime.now(UTC)
    artifacts: list[dict] = []
    edges: list[dict] = []
    root_id = "syn:root"
    artifacts.append(
        LineageArtifact(artifact_id=root_id, kind="source", name="root").as_row(now)
    )
    frontier = [root_id]
    counter = 0
    downstream = 0
    for _level in range(depth):
        next_frontier: list[str] = []
        for parent in frontier:
            for _ in range(branching):
                child = f"syn:n{counter}"
                counter += 1
                downstream += 1
                artifacts.append(
                    LineageArtifact(artifact_id=child, kind="row-set", name=child).as_row(now)
                )
                edges.append(
                    LineageEdge(
                        edge_id=f"syn:e{counter}",
                        edge_type="produced",
                        from_artifact_id=parent,
                        to_artifact_id=child,
                    ).as_row(now)
                )
                next_frontier.append(child)
        frontier = next_frontier
    _replace_rows(lake, "lineage_artifacts", "artifact_id", artifacts, LINEAGE_ARTIFACTS_SCHEMA)
    _replace_rows(lake, "lineage_edges", "edge_id", edges, LINEAGE_EDGES_SCHEMA)
    return root_id, downstream


@requires_graph
def test_fanout_benchmark_sdk_vs_cypher_parity_and_timings(tmp_path, capsys):
    lake = Lake.init(tmp_path / "robot.lance")
    depth, branching = 3, 8
    root_id, expected_downstream = _build_fanout_graph(lake, depth=depth, branching=branching)

    backend = LineageGraphBackend(lake)

    t0 = time.perf_counter()
    sdk_graph = lake.lineage.impact(root_id)
    sdk_secs = time.perf_counter() - t0
    sdk_ids = {row["artifact_id"] for row in sdk_graph.artifacts} - {root_id}

    t1 = time.perf_counter()
    cypher_ids = set(backend.impact_ids(root_id, max_depth=depth + 1))
    cypher_secs = time.perf_counter() - t1

    # Parity: both reach every downstream node.
    assert len(sdk_ids) == expected_downstream
    assert cypher_ids == sdk_ids

    # Probe native once for the record.
    probe = backend.probe_native_strategy()

    with capsys.disabled():
        print(
            f"\n[0099 fan-out benchmark] depth={depth} branching={branching} "
            f"downstream_nodes={expected_downstream}\n"
            f"  SDK impact BFS      : {sdk_secs * 1000:.1f} ms\n"
            f"  Cypher DataFusion   : {cypher_secs * 1000:.1f} ms "
            f"(*1..{depth + 1} join expansion)\n"
            f"  native strategy     : available={probe.available} "
            f"({probe.detail})"
        )
