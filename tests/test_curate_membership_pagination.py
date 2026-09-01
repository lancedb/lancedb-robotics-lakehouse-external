"""Backlog 0142: indexed paginated curation replay for large membership histories.

Covers the bounded, resumable ``resolve_membership_pages`` audit replay that pages
the append-only ``curation_memberships`` decision log in deterministic
``(created_at, membership_id)`` order without materializing unrelated targets, while
leaving the small-history ``resolve_membership`` convenience API (0082) intact.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest
from test_curate import _build_curation_lake
from typer.testing import CliRunner

from lancedb_robotics import curate as curate_mod
from lancedb_robotics.cli import app
from lancedb_robotics.curate import (
    CurationError,
    _ComparisonExecutionStats,
    _decode_membership_history_cursor,
    _encode_membership_history_cursor,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import CURATION_MEMBERSHIPS_SCHEMA

runner = CliRunner()
BASE = datetime(2026, 6, 16, tzinfo=UTC)


def _membership_row(
    *,
    membership_id: str,
    target_id: str,
    created_at: datetime,
    view_id: str = "view-focus",
    scenario_id: str | None = None,
    target_grain: str = "scenario",
    decision: str = "include",
    source: str = "human",
    reviewer: str = "qa-a",
    supersedes: str = "",
    reason_code: str = "",
) -> dict:
    return {
        "membership_id": membership_id,
        "view_id": view_id,
        "target_grain": target_grain,
        "target_id": target_id,
        "scenario_id": scenario_id if scenario_id is not None else target_id,
        "decision": decision,
        "reason_code": reason_code,
        "reason": "",
        "note": "",
        "reviewer": reviewer,
        "queue": "",
        "priority": 0,
        "score": None,
        "metadata": [],
        "source": source,
        "supersedes_membership_id": supersedes,
        "created_by": "test",
        "transform_id": "tfm-test",
        "created_at": created_at,
    }


def _seed_history(
    lake: Lake,
    *,
    focus_target: str = "scn-neighbor",
    focus_count: int = 12,
    other_targets: tuple[str, ...] = (),
    other_count: int = 0,
) -> list[dict]:
    """Append a synthetic decision history: ``focus_count`` rows on one target plus
    ``other_count`` unrelated rows across ``other_targets``. Returns the focus rows in
    deterministic ``(created_at, membership_id)`` order."""
    sources = ["human", "model", "rule", "dedup"]
    reviewers = ["qa-a", "qa-b", "qa-c"]
    decisions = ["include", "exclude", "defer"]
    focus_rows = []
    for index in range(focus_count):
        focus_rows.append(
            _membership_row(
                membership_id=f"mem-focus-{index:04d}",
                target_id=focus_target,
                created_at=BASE + timedelta(seconds=index),
                decision=decisions[index % len(decisions)],
                source=sources[index % len(sources)],
                reviewer=reviewers[index % len(reviewers)],
                supersedes=(f"mem-focus-{index - 1:04d}" if index else ""),
                reason_code=f"pass-{index}",
            )
        )
    other_rows = []
    for index in range(other_count):
        target = other_targets[index % len(other_targets)] if other_targets else "scn-anchor"
        other_rows.append(
            _membership_row(
                membership_id=f"mem-other-{index:05d}",
                target_id=target,
                created_at=BASE + timedelta(seconds=index),
                reason_code=f"other-{index}",
            )
        )
    lake.table("curation_memberships").add(
        pa.Table.from_pylist(focus_rows + other_rows, schema=CURATION_MEMBERSHIPS_SCHEMA)
    )
    return sorted(focus_rows, key=lambda row: (row["created_at"], row["membership_id"]))


def _pages_of(history, *, page_size):
    return list(history.iter_pages(page_size=page_size))


def test_membership_history_pages_reproduce_deterministic_order(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    focus = _seed_history(lake, focus_count=13)
    expected = [row["membership_id"] for row in focus]

    history = lake.curate.resolve_membership_pages(target_ids=["scn-neighbor"], page_size=4)
    pages = _pages_of(history, page_size=4)

    collected = [row["membership_id"] for page in pages for row in page.records]
    assert collected == expected
    # deterministic disjoint page boundaries
    assert [len(page.records) for page in pages] == [4, 4, 4, 1]
    assert [page.has_more for page in pages] == [True, True, True, False]
    assert pages[-1].next_cursor == ""


def test_membership_history_cursor_resume_reproduces_tail(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    focus = _seed_history(lake, focus_count=10)
    expected = [row["membership_id"] for row in focus]

    history = lake.curate.resolve_membership_pages(target_ids=["scn-neighbor"])
    first = history.page(page_size=3)
    assert [row["membership_id"] for row in first.records] == expected[:3]
    assert first.has_more

    collected = list(first.records)
    cursor = first.next_cursor
    while cursor:
        page = history.page(page_size=3, cursor=cursor)
        collected.extend(page.records)
        cursor = page.next_cursor
    assert [row["membership_id"] for row in collected] == expected
    # no row appears twice
    assert len({row["membership_id"] for row in collected}) == len(expected)

    # a mid-stream cursor reproduces exactly the tail after it
    resume = history.page(page_size=100, cursor=first.next_cursor)
    assert [row["membership_id"] for row in resume.records] == expected[3:]
    assert resume.has_more is False


def test_membership_history_reads_are_bounded(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    focus_count = 12
    other_count = 400
    _seed_history(
        lake,
        focus_count=focus_count,
        other_targets=("scn-anchor", "scn-duplicate", "scn-site-b-cup"),
        other_count=other_count,
    )
    total_rows = lake.table("curation_memberships").count_rows()
    assert total_rows == focus_count + other_count

    history = lake.curate.resolve_membership_pages(target_ids=["scn-neighbor"], page_size=5)
    stats = _ComparisonExecutionStats(batch_size=4096)
    page = history.page(page_size=5, stats=stats)

    assert len(page.records) == 5
    # scope predicate pushed down: no full-table materialization, and the scan never
    # touches the ~400 unrelated-target rows.
    assert stats.materialized_tables == set()
    assert stats.streamed_tables == {"curation_memberships"}
    assert stats.total_scanned_rows <= focus_count + 5
    assert stats.total_scanned_rows < total_rows


def test_membership_history_cursor_codec_roundtrip_and_validation():
    created = datetime(2026, 6, 16, 12, 30, tzinfo=UTC)
    token = _encode_membership_history_cursor(created, "mem-focus-0007")
    decoded = _decode_membership_history_cursor(token)
    assert decoded is not None
    assert decoded[0] == created
    assert decoded[1] == "mem-focus-0007"
    assert _decode_membership_history_cursor(None) is None
    assert _decode_membership_history_cursor("") is None
    with pytest.raises(CurationError, match="invalid curation membership history cursor"):
        _decode_membership_history_cursor("!!not-base64!!")


def test_membership_history_audit_slice_filters(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _seed_history(lake, focus_count=12)

    model_only = lake.curate.resolve_membership_pages(
        target_ids=["scn-neighbor"], sources=["model"], page_size=100
    ).page(page_size=100)
    assert model_only.records
    assert {row["source"] for row in model_only.records} == {"model"}

    excludes = lake.curate.resolve_membership_pages(
        target_ids=["scn-neighbor"], decisions=["exclude"], page_size=100
    ).page(page_size=100)
    assert excludes.records
    assert {row["decision"] for row in excludes.records} == {"exclude"}

    reviewer_b = lake.curate.resolve_membership_pages(
        target_ids=["scn-neighbor"], reviewers=["qa-b"], page_size=100
    ).page(page_size=100)
    assert reviewer_b.records
    assert {row["reviewer"] for row in reviewer_b.records} == {"qa-b"}


def test_membership_history_as_of_bound_pages(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    focus = _seed_history(lake, focus_count=10)
    cutoff = focus[4]["created_at"]  # inclusive upper bound

    history = lake.curate.resolve_membership_pages(
        target_ids=["scn-neighbor"], as_of=cutoff, page_size=3
    )
    collected = [row["membership_id"] for page in history.iter_pages(page_size=3) for row in page.records]
    assert collected == [row["membership_id"] for row in focus[:5]]


def test_resolve_membership_pages_matches_recorded_history(tmp_path):
    """The paged API over a view-recorded history matches resolve_membership (0082)."""
    lake = _build_curation_lake(tmp_path / "robot.lance")
    workbench = lake.curate.workbench()
    workbench.save_view("audit-replay")
    for decision, code in (
        ("include", "initial"),
        ("exclude", "safety"),
        ("include", "cleared"),
    ):
        workbench.record_decisions(
            view_name="audit-replay",
            decision=decision,
            scenario_ids=["scn-neighbor"],
            reason_code=code,
        )

    eager = lake.curate.resolve_membership(
        view_name="audit-replay",
        target_grain="scenario",
        target_ids=["scn-neighbor"],
        superseded_policy="history",
    )
    history = lake.curate.resolve_membership_pages(
        view_name="audit-replay", target_grain="scenario", target_ids=["scn-neighbor"], page_size=2
    )
    paged_rows = [row for page in history.iter_pages(page_size=2) for row in page.records]

    assert [row["decision"] for row in paged_rows] == [
        row["decision"] for row in eager.membership_history
    ]
    assert [row["membership_id"] for row in paged_rows] == [
        row["membership_id"] for row in eager.membership_history
    ]
    # manifest surfaces scope + read versions without scanning the decision log
    manifest = history.manifest()
    assert manifest["kind"] == "curation-membership-history"
    assert manifest["ordering"] == ["created_at", "membership_id"]
    assert manifest["view"]["name"] == "audit-replay"
    # 0143: the audit envelope also records the pinned chunk-table version, so
    # chunked saved-view replay is reproducible from the manifest alone.
    assert {row["table"] for row in manifest["read_table_versions"]} == {
        "curation_views",
        "curation_view_membership_chunks",
        "curation_memberships",
    }


def test_membership_history_page_size_guardrails(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _seed_history(lake, focus_count=3)
    history = lake.curate.resolve_membership_pages(target_ids=["scn-neighbor"])
    with pytest.raises(CurationError, match="page_size must be positive"):
        history.page(page_size=0)
    with pytest.raises(CurationError, match="page_size must be an integer"):
        history.page(page_size="lots")  # type: ignore[arg-type]


def test_cli_membership_history_pages_and_resumes(tmp_path):
    import json

    lake_path = tmp_path / "robot.lance"
    lake = _build_curation_lake(lake_path)
    _seed_history(lake, focus_count=7)

    first = runner.invoke(
        app,
        [
            "curate",
            "membership-history",
            "--lake",
            str(lake_path),
            "--target-id",
            "scn-neighbor",
            "--page-size",
            "3",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    assert payload["manifest"]["kind"] == "curation-membership-history"
    assert payload["manifest"]["page_size"] == 3
    assert len(payload["pages"]) == 1
    page = payload["pages"][0]
    assert page["record_count"] == 3
    assert page["has_more"] is True
    assert page["next_cursor"]
    assert page["first_key"]["membership_id"] == "mem-focus-0000"

    resume = runner.invoke(
        app,
        [
            "curate",
            "membership-history",
            "--lake",
            str(lake_path),
            "--target-id",
            "scn-neighbor",
            "--page-size",
            "3",
            "--cursor",
            page["next_cursor"],
            "--json",
        ],
    )
    assert resume.exit_code == 0, resume.output
    resume_page = json.loads(resume.output)["pages"][0]
    assert resume_page["first_key"]["membership_id"] == "mem-focus-0003"

    all_pages = runner.invoke(
        app,
        [
            "curate",
            "membership-history",
            "--lake",
            str(lake_path),
            "--target-id",
            "scn-neighbor",
            "--page-size",
            "3",
            "--all",
            "--json",
        ],
    )
    assert all_pages.exit_code == 0, all_pages.output
    pages = json.loads(all_pages.output)["pages"]
    collected = [row["membership_id"] for page in pages for row in page["records"]]
    assert collected == [f"mem-focus-{index:04d}" for index in range(7)]


def test_cli_membership_history_legacy_default_unchanged(tmp_path):
    """Without paging flags the command still emits the flat 0082 resolution report."""
    import json

    lake_path = tmp_path / "robot.lance"
    lake = _build_curation_lake(lake_path)
    workbench = lake.curate.workbench()
    workbench.save_view("audit-replay")
    workbench.record_decisions(
        view_name="audit-replay",
        decision="include",
        scenario_ids=["scn-neighbor"],
        reason_code="initial",
    )

    result = runner.invoke(
        app,
        [
            "curate",
            "membership-history",
            "--lake",
            str(lake_path),
            "--view",
            "audit-replay",
            "--scenario-id",
            "scn-neighbor",
            "--superseded-policy",
            "history",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    # legacy shape: flat resolution report, not the paged {manifest, pages} envelope
    assert "manifest" not in report
    assert report["latest_decisions"][0]["decision"] == "include"
    assert report["membership_history"][0]["decision"] == "include"


def test_membership_history_order_unavailable_fallback_is_bounded_and_warns(monkeypatch, tmp_path):
    """When the backend cannot order the scan, the bounded-heap fallback returns the
    same deterministic page, stays O(page_size) memory (streamed, not materialized),
    and warns -- the previously-silent degradation must be surfaced (SKILLS.md sec 1)."""
    lake = _build_curation_lake(tmp_path / "robot.lance")
    focus = _seed_history(lake, focus_count=9)
    expected = [row["membership_id"] for row in focus]

    # Force only the ordered path unavailable; leave the real (streamed) row source.
    monkeypatch.setattr(
        curate_mod,
        "_read_membership_history_ordered",
        lambda *args, **kwargs: None,
    )

    history = lake.curate.resolve_membership_pages(target_ids=["scn-neighbor"], page_size=4)
    stats = _ComparisonExecutionStats(batch_size=4096)
    with pytest.warns(RuntimeWarning, match="could not order the scan"):
        first = history.page(page_size=4, stats=stats)
    # streamed pushdown row source -> bounded memory, not a full materialization
    assert stats.materialized_tables == set()
    assert stats.streamed_tables == {"curation_memberships"}
    collected = [row["membership_id"] for row in first.records]
    cursor = first.next_cursor
    while cursor:
        page = history.page(page_size=4, cursor=cursor)
        collected.extend(row["membership_id"] for row in page.records)
        cursor = page.next_cursor
    assert collected == expected


def test_membership_history_unpushable_predicate_streams_bounded_and_warns(monkeypatch, tmp_path):
    """A backend that cannot push the scope predicate still streams (bounded memory,
    matcher filters) rather than materializing the whole table, and warns."""
    lake = _build_curation_lake(tmp_path / "robot.lance")
    focus = _seed_history(
        lake, focus_count=6, other_targets=("scn-anchor", "scn-duplicate"), other_count=50
    )
    expected = [row["membership_id"] for row in focus]

    monkeypatch.setattr(
        curate_mod, "_read_membership_history_ordered", lambda *args, **kwargs: None
    )

    real_source = curate_mod._membership_history_row_source

    def unpushable_source(handle, *, where_sql, stats):
        # Simulate a backend that rejects the where predicate: drop it, so the row
        # source streams the whole table unfiltered (matcher then filters).
        rows_iter, _mode = real_source(handle, where_sql=None, stats=stats)
        return rows_iter, ("unfiltered" if where_sql else "pushdown")

    monkeypatch.setattr(curate_mod, "_membership_history_row_source", unpushable_source)

    history = lake.curate.resolve_membership_pages(target_ids=["scn-neighbor"], page_size=4)
    stats = _ComparisonExecutionStats(batch_size=4096)
    with pytest.warns(RuntimeWarning, match="could not push the scope predicate"):
        first = history.page(page_size=4, stats=stats)
    # streamed (bounded memory) even though it scanned the whole table
    assert stats.materialized_tables == set()
    collected = [row["membership_id"] for row in first.records]
    cursor = first.next_cursor
    while cursor:
        page = history.page(page_size=4, cursor=cursor)
        collected.extend(row["membership_id"] for row in page.records)
        cursor = page.next_cursor
    assert collected == expected
