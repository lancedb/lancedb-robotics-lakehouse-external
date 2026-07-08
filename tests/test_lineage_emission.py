"""Automatic lineage emission for SDK write paths (backlog 0098).

Robustness follow-up to 0060/0097: every canonical SDK write path emits its
lineage graph slice (an execution + input/output artifacts + edges) as part of
the same operation, so ``refresh_graph()`` becomes a reconciliation/backfill
tool rather than the only way to build graph state. These tests pin the 0098
acceptance criteria:

- ingest -> scenarios -> snapshot produces lineage rows *without* calling
  ``refresh_graph()`` (AC#1);
- a failed transform records its execution + consumed inputs but no produced
  output artifacts/edges (AC#2);
- re-running a deterministic operation does not duplicate graph rows (AC#3);
- a full refresh after inline emission reports no divergence, and inline-only
  emission is a consistent subset of the projection (AC#4).
"""

from datetime import UTC, datetime

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import (
    execution_artifact_id,
    snapshot_artifact_id,
)
from lancedb_robotics.scenarios import create_scenario_windows


def _seed_lake(tmp_path, fixtures_dir, fixture_name="sample.mcap"):
    """Ingest -> scenario-window -> snapshot, with NO explicit refresh_graph()."""
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
    return lake, manifest


def _artifacts(lake):
    return {
        row["artifact_id"]: row
        for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
    }


def _executions(lake):
    return {
        row["execution_id"]: row
        for row in lake.table("lineage_executions").to_arrow().to_pylist()
    }


def _edges(lake):
    return lake.table("lineage_edges").to_arrow().to_pylist()


# --- AC#1: write paths emit lineage without a refresh ---------------------------


def test_ingest_scenarios_snapshot_emits_without_refresh(tmp_path, fixtures_dir):
    lake, manifest = _seed_lake(tmp_path, fixtures_dir)

    # No refresh_graph() has run, yet the graph tables carry rows.
    execs = _executions(lake)
    arts = _artifacts(lake)
    assert execs, "expected inline-emitted executions before any refresh_graph()"
    assert arts, "expected inline-emitted artifacts before any refresh_graph()"

    # The three write paths each emitted an execution keyed by their transform id.
    kinds = {row["kind"] for row in execs.values()}
    assert "dataset-snapshot" in kinds
    assert any(kind == "scenario-windowing" for kind in kinds)
    assert any(kind in {"ingest", "inspect"} for kind in kinds)

    # The snapshot artifact and its producing execution were emitted inline.
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)
    assert snapshot_id in arts
    snapshot_exec = execution_artifact_id(manifest.transform_id)
    assert snapshot_exec in execs
    assert execs[snapshot_exec]["status"] == "completed"

    # The snapshot execution pins its inputs and outputs; edges connect them.
    snap = execs[snapshot_exec]
    assert snapshot_id in set(snap["output_artifact_ids"])
    assert snap["input_artifact_ids"], "snapshot should record consumed inputs"
    assert _edges(lake), "expected inline-emitted edges"


def test_emitted_snapshot_reaches_ingest_upstream_without_refresh(tmp_path, fixtures_dir):
    lake, manifest = _seed_lake(tmp_path, fixtures_dir)
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)

    # Walk the emitted edges upstream from the snapshot (from -> to means
    # "upstream produced downstream"); the snapshot must reach an ``observations``
    # table-version, which ingest produced -- all from inline emission.
    edges = _edges(lake)
    upstream = {}
    for edge in edges:
        upstream.setdefault(edge["to_artifact_id"], set()).add(edge["from_artifact_id"])

    seen = set()
    frontier = [snapshot_id]
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        frontier.extend(upstream.get(node, ()))

    arts = _artifacts(lake)
    reached_tables = {
        arts[a]["table_name"] for a in seen if a in arts and arts[a].get("table_name")
    }
    assert "observations" in reached_tables
    assert "runs" in reached_tables


# --- AC#2: failed write records status without produced-output edges -------------


def test_failed_transform_records_execution_without_output_edges(tmp_path, fixtures_dir):
    lake, _manifest = _seed_lake(tmp_path, fixtures_dir)
    now = datetime.now(UTC)

    failed = {
        "transform_id": "tfm-failed-enrich",
        "kind": "enrichment",
        "source_id": None,
        "input_uris": [],
        "input_table_versions": [
            {"table": "observations", "version": int(lake.table("observations").version), "tag": ""}
        ],
        "output_tables": ["observations"],
        "params": "{}",
        "status": "failed",
        "error": "boom",
        "started_at": now,
        "finished_at": now,
        "created_by": "test",
        "created_at": now,
    }
    emitted = lake.lineage.emit_transform(failed)

    execs = _executions(lake)
    exec_id = execution_artifact_id("tfm-failed-enrich")
    assert exec_id in execs
    assert execs[exec_id]["status"] == "failed"
    # A failed transform records its consumed inputs...
    assert execs[exec_id]["input_artifact_ids"], "failed exec should still pin inputs"
    # ...but asserts no produced outputs.
    assert execs[exec_id]["output_artifact_ids"] == []
    assert emitted.produced_outputs is False

    # No "produced" edge originates from a failed execution.
    produced = [
        edge
        for edge in _edges(lake)
        if edge.get("execution_id") == exec_id and edge["edge_type"] == "produced"
    ]
    assert produced == []
    # The consumed-by edge from the input to the transform is present.
    consumed = [
        edge
        for edge in _edges(lake)
        if edge.get("execution_id") == exec_id and edge["edge_type"] == "consumed-by"
    ]
    assert consumed


# --- AC#3: idempotent retry does not duplicate graph rows ------------------------


def _counts(lake):
    return (
        lake.table("lineage_artifacts").count_rows(),
        lake.table("lineage_executions").count_rows(),
        lake.table("lineage_edges").count_rows(),
    )


def test_repeated_emission_is_idempotent(tmp_path, fixtures_dir):
    lake, manifest = _seed_lake(tmp_path, fixtures_dir)
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)
    snapshot_exec = execution_artifact_id(manifest.transform_id)

    row = next(
        r
        for r in lake.table("transform_runs").to_arrow().to_pylist()
        if r["transform_id"] == manifest.transform_id
    )

    # Emission upserts on stable ids: re-emitting the same transform row (same lake
    # state) touches the same graph ids and never appends duplicates.
    before = _counts(lake)
    lake.lineage.emit_transform(row)
    lake.lineage.emit_transform(row)
    assert _counts(lake) == before

    # The stable snapshot artifact and its execution each appear exactly once.
    snapshot_rows = [
        a
        for a in lake.table("lineage_artifacts").to_arrow().to_pylist()
        if a["artifact_id"] == snapshot_id
    ]
    assert len(snapshot_rows) == 1
    exec_rows = [
        e
        for e in lake.table("lineage_executions").to_arrow().to_pylist()
        if e["execution_id"] == snapshot_exec
    ]
    assert len(exec_rows) == 1


# --- AC#4: refresh reconciles emitted rows and reports divergence ---------------


def test_refresh_after_emission_reports_no_divergence(tmp_path, fixtures_dir):
    lake, _manifest = _seed_lake(tmp_path, fixtures_dir)

    lake.lineage.refresh_graph(force_full=True)

    divergence = lake.lineage.emission_divergence()
    assert divergence.consistent is True, divergence.as_dict()
    assert divergence.changed == {}
    assert divergence.missing_from_graph == {}


def test_inline_emission_needs_refresh_backfill(tmp_path, fixtures_dir):
    lake, _manifest = _seed_lake(tmp_path, fixtures_dir)

    # Before any refresh: inline emission has recorded the transform-slice graph
    # but the entity-level nodes are still owned by refresh, so the report is not
    # yet consistent and names what refresh must backfill.
    divergence = lake.lineage.emission_divergence()
    assert divergence.consistent is False
    assert any(divergence.missing_from_graph.values())

    # A refresh reconciles the emitted rows: nothing the current projection needs
    # is left missing or content-mismatched.
    lake.lineage.refresh_graph(force_full=True)
    reconciled = lake.lineage.emission_divergence()
    assert reconciled.consistent is True, reconciled.as_dict()
