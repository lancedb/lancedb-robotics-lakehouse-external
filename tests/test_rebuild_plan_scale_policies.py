"""Backlog 0110: scalable rebuild-plan pagination, guardrails, and action policies.

These tests build synthetic lineage graphs with the manual recording API so they
stay fast and deterministic and do not depend on ingest fixtures. They lock the
0110 contract: configurable action policies that never change traversal, plan-size
guardrails with actionable errors, bounded-memory summary/pagination with stable
continuation tokens, and the benchmark report shape.
"""

from __future__ import annotations

import pytest

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import (
    DEFAULT_ACTION_POLICY,
    ActionContext,
    CallableActionPolicy,
    MappingActionPolicy,
    RebuildPlanError,
    RebuildPlanTooLarge,
)


def _fanout_lake(tmp_path, *, fanout: int = 25, noise: int = 200):
    """A source fanning out to ``fanout`` snapshots, plus ``noise`` unrelated sources.

    The affected downstream set from the source is exactly ``source + fanout``; the
    noise artifacts are unreachable, so a bounded traversal must never touch them.
    """

    lake = Lake.init(tmp_path / "fanout.lance")
    source = lake.lineage.record_artifact(
        kind="source", source_uri="s3://bucket/run.mcap", source_id="src-0110"
    )
    snapshot_ids = []
    for i in range(fanout):
        snap = lake.lineage.record_artifact(
            kind="dataset-snapshot", name=f"snap-{i:03d}", row_ids=[f"ds-{i:03d}"]
        )
        lake.lineage.record_edge(
            edge_type="selected-from",
            from_artifact_id=source.artifact_id,
            to_artifact_id=snap.artifact_id,
        )
        snapshot_ids.append(snap.artifact_id)
    for j in range(noise):
        # Unconnected sources that inflate the total graph but are not reachable.
        lake.lineage.record_artifact(
            kind="source", source_uri=f"s3://bucket/noise-{j}.mcap", source_id=f"noise-{j}"
        )
    return lake, source, snapshot_ids


# --- Action policies --------------------------------------------------------


def test_default_policy_matches_builtin_classification(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=5, noise=0)

    baseline = lake.lineage.rebuild_plan(source.artifact_id, refresh=False)
    explicit_default = lake.lineage.rebuild_plan(
        source.artifact_id, refresh=False, action_policy=DEFAULT_ACTION_POLICY
    )
    identity = lake.lineage.rebuild_plan(
        source.artifact_id,
        refresh=False,
        action_policy=CallableActionPolicy(lambda ctx: None),
    )

    def labels(plan):
        return {a.artifact_id: a.action for a in plan.actions}

    assert labels(baseline) == labels(explicit_default) == labels(identity)
    assert baseline.actions_by_type == {"quarantine": 1, "resnapshot": 5}
    assert baseline.policy_name == "default"


def test_custom_policy_remaps_kinds_without_changing_traversal(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=5, noise=0)

    baseline = lake.lineage.rebuild_plan(source.artifact_id, refresh=False)
    remapped = lake.lineage.rebuild_plan(
        source.artifact_id,
        refresh=False,
        action_policy=MappingActionPolicy(kind_actions={"dataset-snapshot": "notify-only"}),
    )

    # The impacted artifact set (and its ordering) is identical -- only labels move.
    assert [a.artifact_id for a in baseline.actions] == [a.artifact_id for a in remapped.actions]
    assert remapped.actions_by_type == {"quarantine": 1, "notify-only": 5}
    assert remapped.policy_name == "mapping"


def test_policy_can_key_on_incoming_edge_type(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=3, noise=0)

    plan = lake.lineage.rebuild_plan(
        source.artifact_id,
        refresh=False,
        action_policy=MappingActionPolicy(edge_actions={"selected-from": "re-export"}),
    )
    downstream = [a for a in plan.actions if a.artifact_id != source.artifact_id]
    assert downstream and all(a.action == "re-export" for a in downstream)


def test_severity_override_applies_blanket_action(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=4, noise=0)

    plan = lake.lineage.rebuild_plan(
        source.artifact_id,
        refresh=False,
        severity="low",
        action_policy=MappingActionPolicy(severity_actions={"low": "notify-only"}),
    )
    assert plan.actions_by_type == {"notify-only": 5}


def test_policy_emitting_unknown_action_is_rejected(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=2, noise=0)

    with pytest.raises(RebuildPlanError, match="unknown action"):
        lake.lineage.rebuild_plan(
            source.artifact_id,
            refresh=False,
            action_policy=CallableActionPolicy(lambda ctx: "bogus"),
        )


def test_mapping_policy_validates_actions_at_construction():
    with pytest.raises(RebuildPlanError, match="unknown action"):
        MappingActionPolicy(kind_actions={"source": "obliterate"})


def test_action_context_exposes_default_and_edges(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=2, noise=0)
    seen: list[ActionContext] = []

    def spy(ctx: ActionContext) -> str | None:
        seen.append(ctx)
        return None

    lake.lineage.rebuild_plan(
        source.artifact_id, refresh=False, action_policy=CallableActionPolicy(spy)
    )
    by_kind = {ctx.kind: ctx for ctx in seen}
    assert by_kind["source"].default_action == "quarantine"
    assert by_kind["dataset-snapshot"].default_action == "resnapshot"
    assert "selected-from" in by_kind["dataset-snapshot"].incoming_edge_types


# --- Plan-size guardrails ---------------------------------------------------


def test_max_affected_artifacts_guardrail(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=25, noise=0)

    with pytest.raises(RebuildPlanTooLarge, match="max_affected_artifacts"):
        lake.lineage.rebuild_plan(source.artifact_id, refresh=False, max_affected_artifacts=5)


def test_max_actions_guardrail(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=25, noise=0)

    with pytest.raises(RebuildPlanTooLarge, match="max_actions"):
        lake.lineage.rebuild_plan(source.artifact_id, refresh=False, max_actions=5)


def test_missing_indexes_raise_actionable_error(tmp_path, monkeypatch):
    lake, source, _ = _fanout_lake(tmp_path, fanout=2, noise=0)

    import lancedb_robotics.lineage as lineage_mod

    monkeypatch.setattr(
        lineage_mod,
        "_lineage_index_plan",
        lambda _lake: {
            "missing": [
                {"table": "lineage_edges", "column": "from_artifact_id", "index_type": "BTREE"}
            ]
        },
    )
    with pytest.raises(RebuildPlanError, match="missing"):
        lake.lineage.rebuild_plan(source.artifact_id, refresh=False, require_indexes=True)


def test_require_indexes_passes_when_present(tmp_path, monkeypatch):
    lake, source, _ = _fanout_lake(tmp_path, fanout=2, noise=0)

    import lancedb_robotics.lineage as lineage_mod

    monkeypatch.setattr(lineage_mod, "_lineage_index_plan", lambda _lake: {"missing": []})
    plan = lake.lineage.rebuild_plan(source.artifact_id, refresh=False, require_indexes=True)
    assert plan.action_count == 3


# --- Summary + pagination ---------------------------------------------------


def test_summary_mode_returns_aggregates_only(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=25, noise=0)

    summary = lake.lineage.rebuild_plan_summary(source.artifact_id, refresh=False)
    assert summary.actions == ()
    assert summary.summary_only is True
    assert summary.total_actions == 26
    assert summary.affected_artifact_count == 26
    assert summary.actions_by_type == {"quarantine": 1, "resnapshot": 25}
    assert summary.affected_by_kind == {"source": 1, "dataset-snapshot": 25}
    payload = summary.as_dict()
    assert payload["actions"] == []
    assert payload["page"]["summary_only"] is True
    assert payload["summary"]["action_count"] == 26


def test_pagination_is_stable_and_covers_full_plan(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=25, noise=0)

    full = lake.lineage.rebuild_plan(source.artifact_id, refresh=False)
    assert full.action_count == 26

    collected = []
    token = None
    pages = 0
    while True:
        page = lake.lineage.rebuild_plan(
            source.artifact_id, refresh=False, page_size=10, page_token=token
        )
        pages += 1
        assert len(page.actions) <= 10
        assert page.total_actions == 26
        collected.extend(page.actions)
        token = page.next_page_token
        if token is None:
            break
        assert pages < 10  # guard against a non-terminating loop

    assert pages == 3
    # Union of pages equals the full ordered plan, no gaps or overlaps.
    assert [a.step for a in collected] == [a.step for a in full.actions]
    assert [a.artifact_id for a in collected] == [a.artifact_id for a in full.actions]


def test_page_token_is_bound_to_its_plan(tmp_path):
    lake, source, snapshot_ids = _fanout_lake(tmp_path, fanout=25, noise=0)

    page = lake.lineage.rebuild_plan(source.artifact_id, refresh=False, page_size=10)
    assert page.next_page_token is not None

    # A token minted for the source plan is not valid for a different root.
    with pytest.raises(RebuildPlanError, match="does not match this plan"):
        lake.lineage.rebuild_plan(
            snapshot_ids[0], refresh=False, page_size=10, page_token=page.next_page_token
        )


def test_page_token_requires_page_size(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=3, noise=0)

    with pytest.raises(RebuildPlanError, match="page_token requires page_size"):
        lake.lineage.rebuild_plan(source.artifact_id, refresh=False, page_token="whatever")


# --- Bounded memory + benchmark ---------------------------------------------


def test_traversal_is_bounded_by_affected_subgraph_not_total_graph(tmp_path):
    # 205 total source-like artifacts, but only 6 are reachable from ``source``.
    lake, source, _ = _fanout_lake(tmp_path, fanout=5, noise=200)

    plan = lake.lineage.rebuild_plan(source.artifact_id, refresh=False)
    # Work is proportional to the visited subgraph, independent of the 200 noise rows.
    assert plan.affected_artifact_count == 6
    assert plan.action_count == 6


def test_benchmark_report_shape_and_bounded_memory(tmp_path):
    lake, source, _ = _fanout_lake(tmp_path, fanout=10, noise=100)

    report = lake.lineage.benchmark_rebuild_plan(source.artifact_id, repeat=2, refresh=False)
    assert report["schema_version"] == "lancedb-robotics/rebuild-plan-benchmark/v1"
    assert report["affected_artifact_count"] == 11
    assert report["action_count"] == 11
    assert report["repeat"] == 2
    assert len(report["traversal_seconds"]) == 2
    assert report["traversal_seconds_best"] <= report["traversal_seconds_mean"] + 1e-9
    assert isinstance(report["peak_memory_bytes"], int) and report["peak_memory_bytes"] > 0
    assert report["actions_by_type"] == {"quarantine": 1, "resnapshot": 10}
