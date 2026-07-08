"""Tests for the query-driven cache-warm planner (backlog 0348).

Warm exactly what training reads by replaying its queries: bounded
`WHERE <stable id> IN (...)` over the epoch subset, filtering on a stable data-id
column (never `_rowid`), with an index precondition check. No plan executor, no
placement, no whole-table warm.
"""

import json

import pytest

from lancedb_robotics.training_query_warm import (
    QueryWarmError,
    QueryWarmTableSpec,
    TableIndexPrecondition,
    build_query_warm_plan,
    warm_id_column,
)


def _indexed(table, id_column):
    return TableIndexPrecondition(
        table=table, id_column=id_column, indexed=True, status="already_present", note="ok"
    )


def _unindexed(table, id_column):
    return TableIndexPrecondition(
        table=table, id_column=id_column, indexed=False, status="would_create", note="no index"
    )


# --------------------------------------------------------------------------- #
# Module-level planner
# --------------------------------------------------------------------------- #


def test_bounded_chunked_predicates_on_stable_id():
    spec = QueryWarmTableSpec(
        table="observations",
        version=3,
        id_column="observation_id",
        id_values=["a", "b", "c", "d", "e"],
        columns=["observation_id", "state_vector"],
    )
    plan = build_query_warm_plan(
        snapshot_name="demo-v1", scope="row-plan", specs=[spec], chunk_size=2
    ).to_dict()
    assert plan["metrics"]["queries"] == 3  # 5 ids / chunk 2 -> 3 queries
    assert plan["metrics"]["total_ids"] == 5
    queries = plan["tables"][0]["queries"]
    assert queries[0]["where"] == "observation_id IN ('a', 'b')"
    assert queries[2]["where"] == "observation_id IN ('e')"
    assert queries[0]["select"] == ["observation_id", "state_vector"]
    # Never leaks internal row ids anywhere.
    assert "_rowid" not in json.dumps(plan)


def test_rejects_internal_rowid():
    spec = QueryWarmTableSpec(
        table="observations",
        version=None,
        id_column="_rowid",
        id_values=[1, 2, 3],
        columns=["state_vector"],
    )
    with pytest.raises(QueryWarmError, match="_rowid"):
        build_query_warm_plan(snapshot_name="s", scope="row-plan", specs=[spec])


def test_index_precondition_warns_when_unindexed():
    spec = QueryWarmTableSpec(
        table="observations",
        version=1,
        id_column="observation_id",
        id_values=["a", "b"],
        columns=["observation_id"],
    )
    plan = build_query_warm_plan(
        snapshot_name="s", scope="row-plan", specs=[spec], index_checker=_unindexed
    ).to_dict()
    assert plan["metrics"]["unindexed_tables"] == 1
    assert any("not scalar-indexed" in w for w in plan["warnings"])
    assert plan["tables"][0]["precondition"]["indexed"] is False


def test_index_precondition_clean_when_indexed():
    spec = QueryWarmTableSpec(
        table="observations",
        version=1,
        id_column="observation_id",
        id_values=["a", "b"],
        columns=["observation_id"],
    )
    plan = build_query_warm_plan(
        snapshot_name="s", scope="row-plan", specs=[spec], index_checker=_indexed
    ).to_dict()
    assert plan["metrics"]["unindexed_tables"] == 0
    assert plan["warnings"] == []
    assert plan["tables"][0]["precondition"]["indexed"] is True


def test_deterministic_and_deduped():
    spec = QueryWarmTableSpec(
        table="observations",
        version=2,
        id_column="observation_id",
        id_values=["a", "a", "b"],  # duplicate collapses
        columns=["observation_id"],
    )
    first = build_query_warm_plan(snapshot_name="s", scope="row-plan", specs=[spec]).to_dict()
    second = build_query_warm_plan(snapshot_name="s", scope="row-plan", specs=[spec]).to_dict()
    assert first == second
    assert first["metrics"]["total_ids"] == 2


def test_warm_id_column_mapping():
    assert warm_id_column("observations") == "observation_id"
    assert warm_id_column("episodes") == "episode_id"
    assert warm_id_column("nonexistent") is None


# --------------------------------------------------------------------------- #
# End-to-end through a training dataset (local lake)
# --------------------------------------------------------------------------- #


def _lake(tmp_path):
    from test_native_training_dataset import _training_lake

    return _training_lake(tmp_path / "robot.lance")


def test_dataset_query_warm_plan_uses_observation_id(tmp_path):
    lake = _lake(tmp_path)
    dataset = lake.training.dataset("demo-v1", columns=["observation_id", "state_vector"])
    plan = dataset.query_warm_plan(chunk_size=2)

    assert plan["snapshot_name"] == "demo-v1"
    table = plan["tables"][0]
    assert table["table"] == "observations"
    assert table["id_column"] == "observation_id"
    # ids are the snapshot's observation ids, not row addresses.
    assert table["total_ids"] == 3
    q0 = table["queries"][0]
    assert q0["where"].startswith("observation_id IN (")
    assert "observation_id" in q0["select"]
    assert "_rowid" not in json.dumps(plan)
    # fresh lake: observation_id is not scalar-indexed -> precondition warns.
    assert table["precondition"]["indexed"] is False
    assert any("not scalar-indexed" in w for w in plan["warnings"])


def test_dataset_warm_query_cache_executes_and_drains(tmp_path):
    lake = _lake(tmp_path)
    dataset = lake.training.dataset("demo-v1", columns=["observation_id", "state_vector"])
    result = dataset.warm_query_cache(chunk_size=2)

    assert result["queries_run"] >= 1
    assert result["queries_failed"] == 0
    assert result["rows_touched"] == 3  # the 3 snapshot observations
    assert result["errors"] == []
    assert result["plan"]["tables"][0]["id_column"] == "observation_id"


def test_index_precondition_clears_after_building_scalar_index(tmp_path):
    from lancedb_robotics.indexing import build_scalar_index

    lake = _lake(tmp_path)
    dataset = lake.training.dataset("demo-v1", columns=["observation_id"])
    assert dataset.query_warm_plan()["tables"][0]["precondition"]["indexed"] is False

    build_scalar_index(lake, table="observations", column="observation_id")
    refreshed = lake.training.dataset("demo-v1", columns=["observation_id"]).query_warm_plan()
    assert refreshed["tables"][0]["precondition"]["indexed"] is True
    assert refreshed["warnings"] == []


def test_lake_training_query_warm_plan_helper(tmp_path):
    lake = _lake(tmp_path)
    plan = lake.training.query_warm_plan("demo-v1", columns=["observation_id"])
    assert plan["tables"][0]["table"] == "observations"
    assert plan["scope"] == "row-plan"
