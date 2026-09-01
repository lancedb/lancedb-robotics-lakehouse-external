"""Backlog 0140: curation view membership chunk migration, validation, compaction."""

import json

import pyarrow as pa
from test_curate import NOW, _build_curation_lake
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.curate import (
    ViewMembershipCompactionReport,
    ViewMembershipMigrationReport,
    ViewMembershipValidationReport,
    compact_view_membership,
    migrate_view_membership,
    validate_view_membership,
)
from lancedb_robotics.maintenance import maintain_lake
from lancedb_robotics.schemas import CURATION_VIEW_MEMBERSHIP_CHUNKS_SCHEMA

_LARGE_SCOPE = (
    "scn-site-b-box-extra",
    "scn-anchor",
    "scn-neighbor",
    "scn-site-b-cup",
    "scn-duplicate",
)


def _view_row(lake, view_id):
    return next(
        row
        for row in lake.table("curation_views").to_arrow().to_pylist()
        if row["view_id"] == view_id
    )


def _chunk_rows(lake, view_id):
    return sorted(
        (
            row
            for row in lake.table("curation_view_membership_chunks").to_arrow().to_pylist()
            if row["view_id"] == view_id
        ),
        key=lambda row: row["start_ordinal"],
    )


def _save_inline_large_view(lake, name="legacy-inline"):
    """Persist a view inline even though it is 'large' (high inline limit)."""
    workbench = lake.curate.workbench(scope=_LARGE_SCOPE)
    expected_order = workbench.scenario_ids
    view = workbench.save_view(name, inline_scenario_limit=100, membership_chunk_size=2)
    assert view.membership_storage == "inline"
    return view, expected_order


def test_migrate_legacy_inline_view_to_chunks_roundtrip(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    view, expected_order = _save_inline_large_view(lake)
    original_table_versions = view.table_versions

    report = migrate_view_membership(lake, inline_scenario_limit=2, chunk_size=2)
    assert isinstance(report, ViewMembershipMigrationReport)
    migrated = [r for r in report.results if r.view_id == view.view_id]
    assert len(migrated) == 1
    assert migrated[0].status == "migrated"
    assert migrated[0].scenario_count == len(expected_order)
    assert migrated[0].chunk_count == 3

    # View identity is preserved; inline column cleared; storage flipped.
    row = _view_row(lake, view.view_id)
    assert row["view_id"] == view.view_id
    assert row["scenario_ids"] == []
    query_spec = json.loads(row["query_spec"])
    assert query_spec["membership_storage"]["kind"] == "chunked"
    assert query_spec["membership_storage"]["chunk_count"] == 3

    # Chunk rows reconstruct the exact ordered membership.
    chunks = _chunk_rows(lake, view.view_id)
    assert [row["scenario_ids"] for row in chunks] == [
        list(expected_order[:2]),
        list(expected_order[2:4]),
        list(expected_order[4:]),
    ]

    # Reopen resolves identical ordered ids from chunks, keeps pinned versions.
    reopened = lake.curate.view("legacy-inline")
    assert reopened.scenario_ids == expected_order
    assert reopened.report["membership_storage"]["kind"] == "chunked"
    assert (
        tuple(
            (item["table"], item["version"])
            for item in reopened.report["table_versions"]
        )
        == original_table_versions
    )


def test_migrate_is_idempotent(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    view, _ = _save_inline_large_view(lake)

    first = migrate_view_membership(lake, inline_scenario_limit=2, chunk_size=2)
    assert [r.status for r in first.results if r.view_id == view.view_id] == ["migrated"]
    chunks_after_first = _chunk_rows(lake, view.view_id)

    second = migrate_view_membership(lake, inline_scenario_limit=2, chunk_size=2)
    result = next(r for r in second.results if r.view_id == view.view_id)
    assert result.status == "already-chunked"
    chunks_after_second = _chunk_rows(lake, view.view_id)
    assert [r["chunk_id"] for r in chunks_after_first] == [
        r["chunk_id"] for r in chunks_after_second
    ]
    assert len(chunks_after_second) == 3


def test_migrate_chunk_write_is_idempotent_under_concurrent_replay(tmp_path):
    """Reproduces the concurrent-migrator shape (SKILLS.md BUG-04): the same
    content-addressed chunk rows written more than once must collapse to one row
    each -- never duplicate -- so the view stays readable. A delete-then-add or a
    bare append write path would fail this under a real race."""
    from lancedb_robotics.curate import (
        _merge_insert_view_rows_with_retry,
        _view_membership_chunk_rows,
    )

    lake = _build_curation_lake(tmp_path / "robot.lance")
    view, expected = _save_inline_large_view(lake)
    migrate_view_membership(lake, inline_scenario_limit=2, chunk_size=2)
    baseline = _chunk_rows(lake, view.view_id)
    assert len({row["chunk_id"] for row in baseline}) == len(baseline)  # unique

    # A second racing migrator replays the identical production chunk write.
    replay_rows = _view_membership_chunk_rows(
        view_id=view.view_id,
        scenario_ids=expected,
        chunk_size=2,
        created_by="racer",
        transform_id="tfm-racer",
        created_at=NOW,
    )
    _merge_insert_view_rows_with_retry(
        lake.table("curation_view_membership_chunks"),
        "chunk_id",
        pa.Table.from_pylist(replay_rows, schema=CURATION_VIEW_MEMBERSHIP_CHUNKS_SCHEMA),
        update_matched=False,
    )
    after = _chunk_rows(lake, view.view_id)
    assert len(after) == len(baseline)  # no duplicate chunk rows
    assert lake.curate.view(view.name).scenario_ids == expected  # still readable
    assert validate_view_membership(lake).status == "ok"


def test_migrate_recovers_from_chunks_written_before_header_flip(tmp_path):
    """Crash-before-flip recovery: if a prior attempt wrote chunk rows but never
    flipped the (still-inline) header, a re-run must converge -- flip the header
    and leave exactly one deterministic chunk set, not duplicates."""
    from lancedb_robotics.curate import (
        _merge_insert_view_rows_with_retry,
        _view_membership_chunk_rows,
    )

    lake = _build_curation_lake(tmp_path / "robot.lance")
    view, expected = _save_inline_large_view(lake)
    # Simulate a crashed prior attempt: chunks present, header still inline.
    pre_rows = _view_membership_chunk_rows(
        view_id=view.view_id,
        scenario_ids=expected,
        chunk_size=2,
        created_by="prior",
        transform_id="tfm-prior",
        created_at=NOW,
    )
    _merge_insert_view_rows_with_retry(
        lake.table("curation_view_membership_chunks"),
        "chunk_id",
        pa.Table.from_pylist(pre_rows, schema=CURATION_VIEW_MEMBERSHIP_CHUNKS_SCHEMA),
        update_matched=False,
    )
    assert json.loads(_view_row(lake, view.view_id)["query_spec"]).get(
        "membership_storage", {}
    ).get("kind", "inline") == "inline"

    report = migrate_view_membership(lake, inline_scenario_limit=2, chunk_size=2)
    assert next(r for r in report.results if r.view_id == view.view_id).status == "migrated"
    chunks = _chunk_rows(lake, view.view_id)
    assert len(chunks) == 3
    assert len({row["chunk_id"] for row in chunks}) == 3  # no duplicates
    assert lake.curate.view(view.name).scenario_ids == expected


def test_migrate_dry_run_writes_nothing(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    view, _ = _save_inline_large_view(lake)

    report = migrate_view_membership(lake, inline_scenario_limit=2, chunk_size=2, dry_run=True)
    result = next(r for r in report.results if r.view_id == view.view_id)
    assert result.status == "would-migrate"
    assert result.chunk_count == 3
    # No writes: still inline, no chunk rows.
    row = _view_row(lake, view.view_id)
    assert row["scenario_ids"] != []
    assert _chunk_rows(lake, view.view_id) == []


def test_migrate_skips_below_threshold(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    view, _ = _save_inline_large_view(lake)

    report = migrate_view_membership(lake, inline_scenario_limit=100, chunk_size=2)
    result = next(r for r in report.results if r.view_id == view.view_id)
    assert result.status == "skipped-below-threshold"
    assert _chunk_rows(lake, view.view_id) == []


def test_validate_healthy_chunked_view_reports_ok(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=_LARGE_SCOPE).save_view(
        "chunked", inline_scenario_limit=2, membership_chunk_size=2
    )

    report = validate_view_membership(lake)
    assert isinstance(report, ViewMembershipValidationReport)
    assert report.status == "ok"
    assert report.chunked_views == 1
    assert report.healthy_views == 1
    assert report.issues == ()


def test_validate_detects_partial_and_corrupt_chunks(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    view = lake.curate.workbench(scope=_LARGE_SCOPE).save_view(
        "chunked", inline_scenario_limit=2, membership_chunk_size=2
    )
    # Corrupt: drop the last chunk (partial membership) => count + digest issues.
    chunks = _chunk_rows(lake, view.view_id)
    last = chunks[-1]
    lake.table("curation_view_membership_chunks").delete(
        f"chunk_id = '{last['chunk_id']}'"
    )

    report = validate_view_membership(lake)
    assert report.status == "issues"
    codes = {issue.code for issue in report.issues}
    assert "total-count-mismatch" in codes
    # Every issue offers actionable repair guidance.
    assert all(issue.repair for issue in report.issues)
    assert all(issue.view_id == view.view_id for issue in report.issues)


def test_validate_and_compact_detect_and_remove_orphan_chunks(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=_LARGE_SCOPE).save_view(
        "chunked", inline_scenario_limit=2, membership_chunk_size=2
    )
    # Plant an orphan chunk (view_id with no header).
    orphan_row = {
        "chunk_id": "viewchunk-orphan-1",
        "view_id": "view-orphan",
        "chunk_index": 0,
        "start_ordinal": 0,
        "end_ordinal": 1,
        "scenario_ids": ["scn-anchor"],
        "scenario_count": 1,
        "chunk_digest": "deadbeef",
        "created_by": "test",
        "transform_id": "tfm-test",
        "created_at": NOW,
    }
    lake.table("curation_view_membership_chunks").add(
        pa.Table.from_pylist([orphan_row], schema=CURATION_VIEW_MEMBERSHIP_CHUNKS_SCHEMA)
    )

    validated = validate_view_membership(lake)
    assert "view-orphan" in validated.orphan_view_ids

    # Dry-run lists but does not delete.
    dry = compact_view_membership(lake, dry_run=True)
    assert "view-orphan" in dry.orphan_view_ids
    assert dry.orphan_chunks_removed == 1
    assert any(
        row["view_id"] == "view-orphan"
        for row in lake.table("curation_view_membership_chunks").to_arrow().to_pylist()
    )

    # Real compaction removes the orphan rows.
    report = compact_view_membership(lake)
    assert isinstance(report, ViewMembershipCompactionReport)
    assert report.orphan_chunks_removed == 1
    assert not any(
        row["view_id"] == "view-orphan"
        for row in lake.table("curation_view_membership_chunks").to_arrow().to_pylist()
    )
    # The healthy view is untouched.
    assert validate_view_membership(lake).status == "ok"


def test_compact_superseded_is_opt_in(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    # Two revisions of the same view name -> distinct content-addressed view ids.
    first = lake.curate.workbench(scope=_LARGE_SCOPE).save_view(
        "rolling", inline_scenario_limit=2, membership_chunk_size=2
    )
    second = lake.curate.workbench(scope=_LARGE_SCOPE[:4]).save_view(
        "rolling", inline_scenario_limit=2, membership_chunk_size=2
    )
    assert first.view_id != second.view_id

    # Default compaction leaves superseded chunks in place.
    default = compact_view_membership(lake)
    assert default.superseded_chunks_removed == 0
    assert _chunk_rows(lake, first.view_id)

    # Opt-in removes the superseded revision's chunks; latest revision retained.
    opt_in = compact_view_membership(lake, include_superseded=True)
    assert first.view_id in opt_in.superseded_view_ids
    assert opt_in.superseded_chunks_removed > 0
    assert _chunk_rows(lake, first.view_id) == []
    assert _chunk_rows(lake, second.view_id)


def test_compact_retains_snapshot_referenced_views(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    first = lake.curate.workbench(scope=_LARGE_SCOPE).save_view(
        "rolling", inline_scenario_limit=2, membership_chunk_size=2
    )
    # Snapshot from the first revision references its view_id in source metadata.
    lake.curate.view("rolling").snapshot(name="pinned-snap", split_by="scenario")
    # New revision supersedes it by name.
    lake.curate.workbench(scope=_LARGE_SCOPE[:4]).save_view(
        "rolling", inline_scenario_limit=2, membership_chunk_size=2
    )

    report = compact_view_membership(lake, include_superseded=True)
    assert first.view_id in report.retained_snapshot_pinned_view_ids
    assert first.view_id not in report.superseded_view_ids
    assert _chunk_rows(lake, first.view_id)


def test_maintenance_surfaces_curation_chunk_section_and_reclaims_orphans(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=_LARGE_SCOPE).save_view(
        "chunked", inline_scenario_limit=2, membership_chunk_size=2
    )
    lake.table("curation_view_membership_chunks").add(
        pa.Table.from_pylist(
            [
                {
                    "chunk_id": "viewchunk-orphan-2",
                    "view_id": "view-orphan",
                    "chunk_index": 0,
                    "start_ordinal": 0,
                    "end_ordinal": 1,
                    "scenario_ids": ["scn-anchor"],
                    "scenario_count": 1,
                    "chunk_digest": "deadbeef",
                    "created_by": "test",
                    "transform_id": "tfm-test",
                    "created_at": NOW,
                }
            ],
            schema=CURATION_VIEW_MEMBERSHIP_CHUNKS_SCHEMA,
        )
    )

    report = maintain_lake(
        lake,
        tables=("curation_view_membership_chunks",),
        cleanup_older_than=None,
        protect_lineage=False,
        refresh_lineage=False,
    )
    section = report.curation_membership_chunks
    assert section is not None
    assert "validation" in section and "compaction" in section
    assert section["compaction"]["orphan_chunks_removed"] == 1
    assert not any(
        row["view_id"] == "view-orphan"
        for row in lake.table("curation_view_membership_chunks").to_arrow().to_pylist()
    )


def test_cli_migrate_validate_compact(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _save_inline_large_view(lake)
    runner = CliRunner()

    migrated = runner.invoke(
        app,
        [
            "curate",
            "migrate-views",
            "--lake",
            str(tmp_path / "robot.lance"),
            "--inline-limit",
            "2",
            "--chunk-size",
            "2",
            "--json",
        ],
    )
    assert migrated.exit_code == 0, migrated.output
    payload = json.loads(migrated.output)
    assert payload["migrated"] == 1

    validated = runner.invoke(
        app,
        ["curate", "validate-chunks", "--lake", str(tmp_path / "robot.lance"), "--json"],
    )
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.output)["status"] == "ok"

    compacted = runner.invoke(
        app,
        [
            "curate",
            "compact-chunks",
            "--lake",
            str(tmp_path / "robot.lance"),
            "--dry-run",
            "--json",
        ],
    )
    assert compacted.exit_code == 0, compacted.output
    assert json.loads(compacted.output)["orphan_chunks_removed"] == 0
