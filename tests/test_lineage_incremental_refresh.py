"""Incremental lineage refresh, stale reconciliation, index plan, paging (0097).

Scale/robustness follow-up to 0060's canonical lineage graph. These tests pin the
0097 acceptance criteria:

- refresh is watermark-aware: a re-refresh with no source-table change is skipped
  (touches zero rows), and a refresh after a source change reports why a full
  re-projection is required and which tables changed (AC#1);
- the refresh plan records source table versions and graph-row counts (AC#5);
- re-refreshing after a canonical row is deleted retires the now-stale graph rows
  and reports them, while respecting retention holds (AC#3);
- a missing-index plan is available and actionable, and is clean once the lineage
  predicate indexes are built (AC#2);
- ``trace``/``impact`` return bounded pages with stable total counts and a
  continuation handle (AC#4).
"""

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import snapshot_artifact_id
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.writeback import ingest_model_outputs


def _seed_lake(tmp_path, fixtures_dir, fixture_name="sample.mcap"):
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


def _graph_versions(lake):
    return {
        name: int(lake.table(name).version)
        for name in ("lineage_artifacts", "lineage_executions", "lineage_edges")
    }


# --- AC#5: refresh plan records source table versions + graph-row counts --------


def test_refresh_plan_records_source_versions_and_graph_counts(tmp_path, fixtures_dir):
    lake, _manifest = _seed_lake(tmp_path, fixtures_dir)

    report = lake.lineage.refresh_graph()
    plan = report.plan

    # Plan records a per-source-table version status the audit trail can pin.
    by_table = {row["table"]: row for row in plan.as_dict()["source_tables"]}
    assert "observations" in by_table
    assert "model_outputs" in by_table
    assert by_table["model_outputs"]["current_version"] == int(
        lake.table("model_outputs").version
    )
    # Graph-row counts on the plan agree with the report and the materialized tables.
    assert plan.artifacts == report.artifacts == lake.table("lineage_artifacts").count_rows() - (
        1 if _has_refresh_state(lake) else 0
    )
    assert plan.edges == report.edges
    assert plan.as_dict()["action"] in {"initial-refresh", "full-refresh"}


def _has_refresh_state(lake):
    return any(
        row.get("kind") == "lineage-refresh-state"
        for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
    )


# --- AC#1: incremental / watermark behavior -------------------------------------


def test_second_refresh_with_no_source_change_is_skipped(tmp_path, fixtures_dir):
    lake, _manifest = _seed_lake(tmp_path, fixtures_dir)

    lake.lineage.refresh_graph()
    versions_after_first = _graph_versions(lake)

    report = lake.lineage.refresh_graph()

    assert report.plan.action == "skipped-unchanged"
    assert report.plan.full_scan is False
    assert report.plan.changed_tables == ()
    # A skipped refresh touches no graph rows: table versions are unchanged.
    assert _graph_versions(lake) == versions_after_first


def test_refresh_after_source_append_reports_full_scan_and_changed_tables(
    tmp_path, fixtures_dir
):
    lake, manifest = _seed_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()

    scenario = lake.table("scenarios").to_arrow().to_pylist()[0]
    ingest_model_outputs(
        lake,
        {
            "model_output_id": "out-second",
            "observation_id": scenario["observation_ids"][0],
            "scenario_id": scenario["scenario_id"],
            "dataset_id": manifest.dataset_id,
            "model_version": "policy@def456",
            "prediction": "ok",
            "producer_run_id": "checkpoint-def456",
        },
        source="trainer",
    )

    report = lake.lineage.refresh_graph()

    assert report.plan.full_scan is True
    assert "model_outputs" in report.plan.changed_tables
    assert report.plan.full_scan_reason
    assert "out-second" in {
        tuple(row["row_ids"])[0]
        for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
        if row.get("row_grain") == "model_outputs" and row.get("row_ids")
    }


def test_dry_run_plan_does_not_mutate_graph(tmp_path, fixtures_dir):
    lake, _manifest = _seed_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    versions = _graph_versions(lake)

    plan = lake.lineage.plan_refresh()

    assert plan.dry_run is True
    assert _graph_versions(lake) == versions  # planning never writes


# --- AC#3: stale-edge reconciliation --------------------------------------------


def test_refresh_retires_stale_rows_after_snapshot_delete(tmp_path, fixtures_dir):
    lake, manifest = _seed_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)

    before_artifacts = {
        row["artifact_id"] for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
    }
    assert snapshot_id in before_artifacts

    # Delete the canonical snapshot row out from under the graph.
    lake.table("dataset_snapshots").delete(f"dataset_id = '{manifest.dataset_id}'")

    report = lake.lineage.refresh_graph()

    assert report.plan.retired_artifacts >= 1
    stale_ids = {row["artifact_id"] for row in report.plan.as_dict()["stale_artifacts"]}
    assert snapshot_id in stale_ids
    remaining = {
        row["artifact_id"] for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
    }
    assert snapshot_id not in remaining
    # Its incident edges are retired too (no dangling selected-from/version-pinned).
    edge_endpoints = {
        endpoint
        for row in lake.table("lineage_edges").to_arrow().to_pylist()
        for endpoint in (row["from_artifact_id"], row["to_artifact_id"])
    }
    assert snapshot_id not in edge_endpoints


def test_reconciliation_preserves_held_stale_artifact(tmp_path, fixtures_dir):
    lake, manifest = _seed_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)

    lake.lineage.retain(snapshot_id, legal_hold=True, reason="audit evidence")
    lake.table("dataset_snapshots").delete(f"dataset_id = '{manifest.dataset_id}'")

    report = lake.lineage.refresh_graph()

    held = {row["artifact_id"] for row in report.plan.as_dict()["held_stale_artifacts"]}
    assert snapshot_id in held
    remaining = {
        row["artifact_id"] for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
    }
    assert snapshot_id in remaining  # a held artifact is not retired


# --- AC#2: missing-index plan ---------------------------------------------------


def test_index_plan_reports_missing_then_clean_after_refresh(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")

    # Graph tables exist but are empty and unindexed before the first refresh.
    plan = lake.lineage.index_plan()
    missing_tables = {row["table"] for row in plan["missing"]}
    assert {"lineage_edges", "lineage_artifacts"} <= missing_tables
    assert plan["all_present"] is False
    # Every missing entry is actionable: it names the column, type, and build call.
    assert all(row.get("build_action") for row in plan["missing"])

    lake.lineage.refresh_graph()

    after = lake.lineage.index_plan()
    assert after["all_present"] is True
    assert after["missing"] == []


def test_refresh_plan_surfaces_missing_indexes(tmp_path, fixtures_dir):
    lake, _manifest = _seed_lake(tmp_path, fixtures_dir)
    # A dry-run plan computed before indexes exist should surface the gap.
    plan = lake.lineage.plan_refresh()
    assert isinstance(plan.as_dict()["missing_indexes"], list)


# --- AC#4: bounded / paged trace and impact -------------------------------------


def test_trace_paginates_with_stable_counts_and_continuation(tmp_path, fixtures_dir):
    lake, manifest = _seed_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)

    full = lake.lineage.trace(snapshot_id)
    total = len(full.artifacts)
    assert total > 2  # enough to force multiple pages

    seen: list[str] = []
    token = None
    pages = 0
    while True:
        page = lake.lineage.trace(snapshot_id, page_size=2, page_token=token)
        pages += 1
        assert page.total_artifacts == total  # stable across pages
        assert len(page.artifacts) <= 2
        seen.extend(row["artifact_id"] for row in page.artifacts)
        if page.next_page_token is None:
            assert page.truncated is False
            break
        assert page.truncated is True
        token = page.next_page_token
        assert pages < 50  # guard against a non-terminating cursor

    assert pages > 1
    assert set(seen) == {row["artifact_id"] for row in full.artifacts}
    assert len(seen) == len(set(seen))  # no overlap between pages
    # The page payload advertises its size and carries the continuation handle.
    assert lake.lineage.trace(snapshot_id, page_size=2).as_dict()["page"]["page_size"] == 2


def test_page_token_is_bound_to_its_query(tmp_path, fixtures_dir):
    import pytest

    from lancedb_robotics.lineage import LineageError

    lake, manifest = _seed_lake(tmp_path, fixtures_dir)
    lake.lineage.refresh_graph()
    snapshot_id = snapshot_artifact_id(manifest.dataset_id)

    first = lake.lineage.trace(snapshot_id, page_size=2)
    assert first.next_page_token is not None
    # A token minted for an upstream trace must not be replayed against a
    # downstream impact of the same root.
    with pytest.raises(LineageError):
        lake.lineage.impact(snapshot_id, page_size=2, page_token=first.next_page_token)
