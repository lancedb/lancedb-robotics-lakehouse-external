"""Backlog 0141: lazy curation selection iteration and snapshot planning.

Exercises the ``CurationSelectionMembership`` lazy surface layered over 0081's
chunked view storage: deterministic ordinal paging with bounded reads, streaming
decision application, snapshot planning that preserves identity, and backward
compatibility for small inline/derived selections.
"""

import json

import pytest
from test_curate import _add_dense_cluster, _build_curation_lake
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.curate import (
    _VIEW_MEMBERSHIP_CHUNK_SCAN_BATCH,
    CurationError,
    _ComparisonExecutionStats,
    _decode_membership_cursor,
    _digest,
    _encode_membership_cursor,
)

runner = CliRunner()


def _large_chunked_view(tmp_path, *, count=200, chunk_size=4, name="big-view"):
    """Build a lake with a large chunked saved view; return (lake, ids, view)."""
    lake = _build_curation_lake(tmp_path / "robot.lance")
    cluster = _add_dense_cluster(lake, prefix="big", count=count)
    selection = lake.curate.workbench(scope=cluster)
    view = selection.save_view(
        name, inline_scenario_limit=chunk_size, membership_chunk_size=chunk_size
    )
    assert view.membership_storage == "chunked"
    return lake, selection.scenario_ids, view


# --- AC#1: iterate a chunked view by pages without loading all ids ----------


def test_view_membership_pages_reproduce_deterministic_order(tmp_path):
    lake, eager_ids, _ = _large_chunked_view(tmp_path)
    membership = lake.curate.view_membership("big-view", page_size=7)

    assert membership.is_chunked
    assert membership.total_count == len(eager_ids)
    # Full materialize and page-concatenation both reproduce the eager order.
    assert membership.materialize() == eager_ids
    paged = tuple(
        sid for page in membership.iter_pages(page_size=7) for sid in page.scenario_ids
    )
    assert paged == eager_ids


def test_view_membership_cursor_resume_reproduces_tail(tmp_path):
    lake, eager_ids, _ = _large_chunked_view(tmp_path)
    membership = lake.curate.view_membership("big-view", page_size=7)

    first = membership.page(page_size=7)
    assert first.scenario_ids == eager_ids[:7]
    assert first.has_more

    collected = list(first.scenario_ids)
    cursor = first.next_cursor
    while cursor:
        page = membership.page(page_size=7, cursor=cursor)
        collected.extend(page.scenario_ids)
        cursor = page.next_cursor
    assert tuple(collected) == eager_ids

    # A mid-chunk resume ordinal (page_size not a multiple of chunk size) is exact.
    resume = membership.page(page_size=10, cursor=_encode_membership_cursor(23))
    assert resume.scenario_ids == eager_ids[23:33]


def test_view_membership_reads_are_bounded(tmp_path):
    lake, eager_ids, view = _large_chunked_view(tmp_path, count=200, chunk_size=4)
    membership = lake.curate.view_membership("big-view")
    chunk_count = membership.chunk_count
    assert chunk_count > _VIEW_MEMBERSHIP_CHUNK_SCAN_BATCH  # forces multiple windows

    stats = _ComparisonExecutionStats(batch_size=4096)
    streamed = tuple(membership.iter_scenario_ids(stats=stats))

    assert streamed == eager_ids
    # Bound (SKILLS.md): pushdown used (no full-table scan) and each window scan
    # holds at most one ordinal window of chunk rows, strictly fewer than all.
    assert stats.materialized_tables == set()
    assert stats.peak_batch_rows <= _VIEW_MEMBERSHIP_CHUNK_SCAN_BATCH
    assert stats.peak_batch_rows < chunk_count
    # Every chunk scanned exactly once across disjoint windows.
    assert stats.total_scanned_rows == chunk_count


def test_truncated_view_membership_raises_loudly(tmp_path):
    # A header advertising the full count over a truncated chunk prefix (e.g. a
    # crash mid-write) must fail loudly, not silently return fewer ids.
    lake, eager_ids, _ = _large_chunked_view(tmp_path, count=40, chunk_size=4)
    lake.table("curation_view_membership_chunks").delete("start_ordinal >= 32")

    membership = lake.curate.view_membership("big-view")
    with pytest.raises(CurationError, match="truncated"):
        membership.materialize()
    # A page that stops before the missing tail is fine (early break, not exhausted).
    assert membership.page(page_size=8).scenario_ids == eager_ids[:8]


def test_membership_scan_fallback_emits_warning(tmp_path, monkeypatch):
    from lancedb_robotics import curate as curate_mod

    lake, eager_ids, _ = _large_chunked_view(tmp_path, count=8, chunk_size=4)
    membership = lake.curate.view_membership("big-view")

    def fake_stream(lk, table, *, columns=None, where_sql=None, batch_size=4096, stats=None):
        # Simulate a backend that cannot push the predicate: honest full scan.
        rows = lk.table(table).to_arrow().to_pylist()
        if stats is not None:
            stats.record_full_scan(table, len(rows))
        yield rows

    monkeypatch.setattr(curate_mod, "_stream_table_rows", fake_stream)
    with pytest.warns(RuntimeWarning, match="full table scan"):
        materialized = membership.materialize()
    assert materialized == eager_ids


def test_decision_stream_result_caches_latest_decisions(tmp_path):
    lake, eager_ids, _ = _large_chunked_view(tmp_path)
    lake.curate.workbench(scope="big-view").record_decisions(
        decision="exclude", scenario_ids=[eager_ids[0]], view_name="big-view"
    )
    result = lake.curate.view_membership("big-view").apply_decisions()
    # Re-streaming (e.g. save_view's two passes) reuses one resolved decision map.
    assert result._latest() is result._latest()


def test_membership_cursor_codec_roundtrip_and_validation():
    assert _decode_membership_cursor(None) == 0
    assert _decode_membership_cursor("") == 0
    for ordinal in (0, 1, 42, 1_000_000):
        assert _decode_membership_cursor(_encode_membership_cursor(ordinal)) == ordinal
    with pytest.raises(CurationError):
        _decode_membership_cursor("not-a-valid-cursor!!")


# --- AC#2: apply_decisions can stream a large view -------------------------


def test_streaming_apply_decisions_matches_eager(tmp_path):
    lake, eager_ids, _ = _large_chunked_view(tmp_path)
    excluded = [eager_ids[1], eager_ids[5], eager_ids[9], eager_ids[40]]
    lake.curate.workbench(scope="big-view").record_decisions(
        decision="exclude", scenario_ids=excluded, view_name="big-view"
    )

    eager = lake.curate.workbench(scope="big-view").apply_decisions(view_name="big-view")
    streamed = lake.curate.view_membership("big-view").apply_decisions()

    assert streamed.materialize() == eager.scenario_ids
    assert streamed.input_count == len(eager_ids)
    assert streamed.output_count == len(eager_ids) - len(excluded)
    assert streamed.removed_by_decision == {"exclude": len(excluded)}
    # The streamed input read is bounded and pushed down, not a full scan.
    read = streamed.report["membership_diagnostics"]["streamed_read"]
    assert read["materialized_tables"] == []
    assert read["peak_batch_rows"] <= _VIEW_MEMBERSHIP_CHUNK_SCAN_BATCH


def test_streaming_apply_decisions_persist_view_roundtrips_and_is_idempotent(tmp_path):
    lake, eager_ids, _ = _large_chunked_view(tmp_path)
    lake.curate.workbench(scope="big-view").record_decisions(
        decision="exclude", scenario_ids=[eager_ids[3]], view_name="big-view"
    )
    expected = lake.curate.workbench(scope="big-view").apply_decisions(
        view_name="big-view"
    ).scenario_ids

    streamed = lake.curate.view_membership("big-view").apply_decisions()
    saved = streamed.save_view("big-view-applied", chunk_size=4)
    assert saved.materialize() == expected

    reopened = lake.curate.view_membership("big-view-applied")
    assert reopened.materialize() == expected
    assert reopened.is_chunked

    # Content-addressed + insert-only: re-persisting the same membership converges
    # on the same view id without duplicating chunk rows.
    again = (
        lake.curate.view_membership("big-view").apply_decisions().save_view(
            "big-view-applied", chunk_size=4
        )
    )
    assert again.view_id == saved.view_id
    chunk_ids = [
        row["chunk_id"]
        for row in lake.table("curation_view_membership_chunks").to_arrow().to_pylist()
        if row["view_id"] == saved.view_id
    ]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_streaming_apply_decisions_removing_everything_raises(tmp_path):
    lake, eager_ids, _ = _large_chunked_view(tmp_path, count=6, chunk_size=2)
    lake.curate.workbench(scope="big-view").record_decisions(
        decision="exclude", scenario_ids=list(eager_ids), view_name="big-view"
    )
    with pytest.raises(CurationError):
        lake.curate.view_membership("big-view").apply_decisions()


# --- AC#3: snapshot planning preserves ordered/deterministic lineage --------


def test_snapshot_plan_preserves_identity_and_records_lineage(tmp_path):
    lake, eager_ids, view = _large_chunked_view(tmp_path)

    eager_snapshot = lake.curate.workbench(scope="big-view").snapshot(name="snap-a")
    plan = lake.curate.view_membership("big-view").plan_snapshot(name="snap-a")

    # The plan records deterministic source lineage without materializing ids:
    # it reuses the membership digest 0081 stored at save time.
    assert plan.scenario_count == len(eager_ids)
    assert plan.membership_digest == _digest({"scenario_ids": list(eager_ids)})
    assert any(table == "scenarios" for table, _ in plan.table_versions)
    assert plan.to_dict()["membership_digest"] == plan.membership_digest

    # Freezing preserves snapshot identity (0141 non-goal to change it).
    plan_snapshot = plan.create_snapshot()
    assert plan_snapshot.dataset_id == eager_snapshot.dataset_id
    # Re-planning the same view is deterministic.
    assert (
        lake.curate.view_membership("big-view").plan_snapshot(name="snap-a").plan_id
        == plan.plan_id
    )


# --- AC#4: small inline / derived selections stay backward compatible -------


def test_inline_small_view_membership_is_backward_compatible(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).save_view(
        "small-inline", inline_scenario_limit=100
    )
    membership = lake.curate.view_membership("small-inline")

    assert membership.is_chunked is False
    assert membership.materialize() == ("scn-anchor", "scn-neighbor")
    assert membership.scenario_ids == ("scn-anchor", "scn-neighbor")
    assert [page.scenario_ids for page in membership.iter_pages(page_size=1)] == [
        ("scn-anchor",),
        ("scn-neighbor",),
    ]
    diagnostics = membership.diagnostics()
    assert diagnostics["storage_kind"] == "inline"
    assert diagnostics["row_count"] == 2
    assert diagnostics["materialize_recommended"] is True


def test_selection_membership_from_derived_selection(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    selection = lake.curate.workbench()
    membership = selection.membership(page_size=2)

    assert membership.is_chunked is False
    assert membership.materialize() == selection.scenario_ids
    reassembled = tuple(
        sid for page in membership.iter_pages(page_size=2) for sid in page.scenario_ids
    )
    assert reassembled == selection.scenario_ids


def test_selection_membership_from_chunked_view_is_lazy(tmp_path):
    lake, eager_ids, _ = _large_chunked_view(tmp_path)
    selection = lake.curate.view("big-view")
    membership = selection.membership()

    assert membership.is_chunked
    assert membership.total_count == len(eager_ids)
    assert membership.materialize() == eager_ids


# --- CLI -------------------------------------------------------------------


def test_cli_view_membership_pages_and_reports_diagnostics(tmp_path):
    lake, eager_ids, _ = _large_chunked_view(tmp_path, count=20, chunk_size=4)
    lake_path = str(lake.uri)

    result = runner.invoke(
        app,
        [
            "curate",
            "view-membership",
            "--lake",
            lake_path,
            "--view",
            "big-view",
            "--page-size",
            "5",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["diagnostics"]["row_count"] == len(eager_ids)
    assert payload["diagnostics"]["lazy"] is True
    assert payload["pages"][0]["scenario_ids"] == list(eager_ids[:5])
    assert payload["pages"][0]["has_more"] is True

    all_result = runner.invoke(
        app,
        [
            "curate",
            "view-membership",
            "--lake",
            lake_path,
            "--view",
            "big-view",
            "--page-size",
            "5",
            "--all",
            "--json",
        ],
    )
    assert all_result.exit_code == 0, all_result.output
    all_payload = json.loads(all_result.output)
    streamed = tuple(
        sid for page in all_payload["pages"] for sid in page["scenario_ids"]
    )
    assert streamed == eager_ids
    assert all_payload["pages"][-1]["has_more"] is False
