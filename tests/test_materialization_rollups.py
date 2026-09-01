"""Tests for the scalable materialization rollup catalog (backlog 0145).

Covers the two new tables (``curation_materialization_rollups`` +
``curation_materialization_files``), the JSON-free rollup summary, keyset-paged
history, plan-compaction retention with safe-delete, and sync idempotency. The
bound assertions check the *mechanism* (SKILLS.md §3): summary/history read only
promoted columns and never project ``report_json``, and completed export
evidence is never eligible for retention.
"""

import json
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest
from test_curate import _build_curation_lake

from lancedb_robotics import materialization_rollups as mr
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import (
    CANONICAL_TABLES,
    CURATION_MATERIALIZATIONS_SCHEMA,
)

BASE = datetime(2026, 7, 1, tzinfo=UTC)


def _source_row(
    mid,
    *,
    fmt="webdataset",
    mode="export",
    copied=0,
    planned=0,
    output_uri="s3://exports/x",
    dataset="ds-1",
    snapshot="cand-a",
    ts=None,
    objects=None,
):
    recon = None
    report = {"accounting": {"payload_bytes_planned": planned}}
    if objects is not None:
        total_obj = sum(int(o["content_length"]) for o in objects)
        recon = {
            "status": "verified",
            "object_count": len(objects),
            "total_object_bytes": total_obj,
            "objects": objects,
        }
        report["reconciliation"] = {
            "status": "verified",
            "object_count": len(objects),
            "total_object_bytes": total_obj,
        }
        report["reconciliation_status"] = "verified"
    row = {
        "materialization_id": mid,
        "dataset_id": dataset,
        "snapshot_name": snapshot,
        "target_format": fmt,
        "output_uri": output_uri,
        "mode": mode,
        "selected_scenario_count": 2,
        "selected_observation_count": 10,
        "total_payload_bytes": 1000,
        "copied_payload_bytes": copied,
        "logical_reference_bytes": 1000 - copied,
        "metadata_bytes_written": 50,
        "copy_ratio": copied / 1000,
        "source_table_versions": [],
        "report_json": json.dumps(report, sort_keys=True),
        "projection_transform_id": "ptf-" + fmt,
        "created_by": "tester",
        "transform_id": "tfm-" + mid,
        "created_at": ts or BASE,
    }
    return row, recon


def _seed(lake, rows):
    """Write each (row, recon) into the source table and emit its rollup inline."""
    src = lake.table("curation_materializations")
    for row, recon in rows:
        src.add(pa.Table.from_pylist([row], schema=CURATION_MATERIALIZATIONS_SCHEMA))
        mr.write_rollup(lake, row, reconciliation=recon, created_by="tester")


def _bare_lake(tmp_path):
    return Lake.init(tmp_path / "robot.lance")


# --------------------------------------------------------------------------- #
# Registration / schema invariants.
# --------------------------------------------------------------------------- #


def test_tables_are_canonical_and_indexed():
    from lancedb_robotics.indexing import PREDICATE_INDEX_COLUMNS_BY_TABLE

    assert "curation_materialization_rollups" in CANONICAL_TABLES
    assert "curation_materialization_files" in CANONICAL_TABLES
    assert "curation_materialization_rollups" in PREDICATE_INDEX_COLUMNS_BY_TABLE
    assert "curation_materialization_files" in PREDICATE_INDEX_COLUMNS_BY_TABLE


def test_summary_projection_excludes_report_json():
    # Structural guard: the promoted-column projections must never pull the heavy
    # JSON body -- that is the entire point of the rollup catalog (AC1).
    assert "report_json" not in mr._ROLLUP_SUMMARY_COLUMNS
    assert "report_json" not in mr._RETENTION_COLUMNS


# --------------------------------------------------------------------------- #
# Inline write + per-file chunks.
# --------------------------------------------------------------------------- #


def test_write_rollup_promotes_columns_and_writes_file_chunks(tmp_path):
    lake = _bare_lake(tmp_path)
    objects = [
        {
            "relative_path": f"shard-{i}.tar",
            "uri": f"s3://exports/x/shard-{i}.tar",
            "content_length": 100 + i,
            "classification": "payload",
            "container": "tar",
            "compression": "",
            "checksum": f"ck{i}",
        }
        for i in range(3)
    ]
    _seed(lake, [_source_row("mat-exp1", copied=800, objects=objects)])

    entry = mr._fetch_rollup_row(lake, "mat-exp1")
    assert entry["copied_payload_bytes"] == 800
    assert entry["logical_reference_bytes"] == 200
    assert entry["payload_copy_policy"] == "materialized-copy"
    assert entry["reconciliation_status"] == "verified"
    assert entry["state"] == mr.STATE_ACTIVE
    assert entry["output_file_count"] == 3
    assert entry["output_file_bytes"] == 303
    assert entry["captured_file_count"] == 3
    assert entry["source_report_available"] is True

    files = lake.table("curation_materialization_files")
    assert files.count_rows() == 3
    assert files.count_rows("classification = 'payload'") == 3


def test_write_rollup_is_idempotent(tmp_path):
    lake = _bare_lake(tmp_path)
    row, recon = _source_row("mat-exp1", copied=800)
    _seed(lake, [(row, recon)])
    _seed(lake, [(row, recon)])  # re-record same content-addressed report
    rollups = lake.table("curation_materialization_rollups")
    assert rollups.count_rows("materialization_id = 'mat-exp1'") == 1


# --------------------------------------------------------------------------- #
# Rollup summary (JSON-free aggregation).
# --------------------------------------------------------------------------- #


def test_rollup_summary_aggregates_promoted_columns(tmp_path):
    lake = _bare_lake(tmp_path)
    _seed(
        lake,
        [
            _source_row("mat-exp1", fmt="webdataset", mode="export", copied=800, ts=BASE),
            _source_row("mat-plan1", fmt="lerobot", mode="plan", planned=500, ts=BASE + timedelta(days=1)),
            _source_row("mat-plan2", fmt="lerobot", mode="plan", planned=600, ts=BASE + timedelta(days=2)),
        ],
    )
    summary = mr.materialization_rollup_summary(lake, dataset_id="ds-1")
    assert summary.materialization_count == 3
    assert summary.copied_payload_bytes == 800
    assert summary.planned_payload_bytes == 1100
    assert summary.total_payload_bytes == 3000
    assert summary.copy_ratio == pytest.approx(800 / 3000)


def test_rollup_summary_reads_only_promoted_columns(tmp_path, monkeypatch):
    lake = _bare_lake(tmp_path)
    _seed(lake, [_source_row("mat-exp1", copied=800)])

    seen_columns: list[tuple] = []
    original = mr._stream_rows

    def _spy(lake_arg, table, *, columns=None, **kwargs):
        if table == "curation_materialization_rollups":
            seen_columns.append(tuple(columns or ()))
        return original(lake_arg, table, columns=columns, **kwargs)

    monkeypatch.setattr(mr, "_stream_rows", _spy)
    mr.materialization_rollup_summary(lake, dataset_id="ds-1")
    assert seen_columns, "summary should stream the rollup table"
    for cols in seen_columns:
        assert cols, "summary must request an explicit projection, never a full scan"
        assert "report_json" not in cols


def test_rollup_group_by_buckets(tmp_path):
    lake = _bare_lake(tmp_path)
    _seed(
        lake,
        [
            _source_row("mat-a", fmt="webdataset", mode="export", copied=800, snapshot="branch-1"),
            _source_row("mat-b", fmt="lerobot", mode="export", copied=400, snapshot="branch-1"),
            _source_row("mat-c", fmt="lerobot", mode="export", copied=100, snapshot="branch-2"),
        ],
    )
    by_format = mr.materialization_rollup_summary(lake, dataset_id="ds-1", group_by="format")
    fmt_map = {b.key: b for b in by_format.buckets}
    assert fmt_map["webdataset"].copied_payload_bytes == 800
    assert fmt_map["lerobot"].copied_payload_bytes == 500
    assert fmt_map["lerobot"].materialization_count == 2

    by_branch = mr.materialization_rollup_summary(lake, dataset_id="ds-1", group_by="branch")
    branch_map = {b.key: b for b in by_branch.buckets}
    assert branch_map["branch-1"].materialization_count == 2
    assert branch_map["branch-2"].copied_payload_bytes == 100


def test_rollup_summary_rejects_unknown_group_by(tmp_path):
    lake = _bare_lake(tmp_path)
    with pytest.raises(mr.MaterializationRollupError):
        mr.materialization_rollup_summary(lake, group_by="nonsense")


# --------------------------------------------------------------------------- #
# Paged history.
# --------------------------------------------------------------------------- #


def test_history_pagination_is_stable_and_resumable(tmp_path):
    lake = _bare_lake(tmp_path)
    _seed(
        lake,
        [
            _source_row(f"mat-{i:02d}", mode="export", copied=100 + i, ts=BASE + timedelta(days=i))
            for i in range(5)
        ],
    )
    page1 = mr.list_materialization_history(lake, dataset_id="ds-1", page_size=2)
    assert [e.materialization_id for e in page1.records] == ["mat-00", "mat-01"]
    assert page1.has_more and page1.next_cursor

    page2 = mr.list_materialization_history(
        lake, dataset_id="ds-1", page_size=2, cursor=page1.next_cursor
    )
    assert [e.materialization_id for e in page2.records] == ["mat-02", "mat-03"]

    page3 = mr.list_materialization_history(
        lake, dataset_id="ds-1", page_size=2, cursor=page2.next_cursor
    )
    assert [e.materialization_id for e in page3.records] == ["mat-04"]
    assert not page3.has_more and page3.next_cursor is None

    # iter_pages covers the same set without overlap.
    seen = [e.materialization_id for page in mr.iter_materialization_history(lake, dataset_id="ds-1", page_size=2) for e in page.records]
    assert seen == ["mat-00", "mat-01", "mat-02", "mat-03", "mat-04"]


def test_history_filters_push_down(tmp_path):
    lake = _bare_lake(tmp_path)
    _seed(
        lake,
        [
            _source_row("mat-exp", fmt="webdataset", mode="export", copied=800),
            _source_row("mat-plan", fmt="lerobot", mode="plan", planned=500),
        ],
    )
    plans = mr.list_materialization_history(lake, dataset_id="ds-1", mode="plan", page_size=10)
    assert [e.materialization_id for e in plans.records] == ["mat-plan"]

    webdataset = mr.list_materialization_history(
        lake, dataset_id="ds-1", target_format="webdataset", page_size=10
    )
    assert [e.materialization_id for e in webdataset.records] == ["mat-exp"]


def test_history_rejects_bad_cursor_and_state(tmp_path):
    lake = _bare_lake(tmp_path)
    _seed(lake, [_source_row("mat-exp", copied=1)])
    with pytest.raises(mr.MaterializationRollupError):
        mr.list_materialization_history(lake, cursor="not-a-valid-cursor")
    with pytest.raises(mr.MaterializationRollupError):
        mr.list_materialization_history(lake, state="bogus")


# --------------------------------------------------------------------------- #
# Retention / plan compaction.
# --------------------------------------------------------------------------- #


def test_retention_prunes_superseded_plans_and_protects_exports(tmp_path):
    lake = _bare_lake(tmp_path)
    _seed(
        lake,
        [
            _source_row("mat-exp1", mode="export", copied=800, output_uri="s3://exp/a", ts=BASE),
            _source_row("mat-plan1", mode="plan", planned=500, output_uri="s3://plan/a", ts=BASE + timedelta(days=1)),
            _source_row("mat-plan2", mode="plan", planned=600, output_uri="s3://plan/a", ts=BASE + timedelta(days=2)),
        ],
    )
    report = mr.prune_materialization_rollups(
        lake, retain_latest=1, older_than=BASE + timedelta(days=1, hours=12)
    )
    assert report.protected_count == 1  # the export
    assert report.pruned_ids == ("mat-plan1",)
    assert report.body_bytes_after < report.body_bytes_before

    # Export evidence untouched, still holds its body.
    export = mr._fetch_rollup_row(lake, "mat-exp1")
    assert export["state"] == mr.STATE_ACTIVE
    assert export["source_report_available"] is True

    # Pruned plan: rollup row survives as audit metadata; source body cleared.
    plan1 = mr._fetch_rollup_row(lake, "mat-plan1")
    assert plan1["state"] == mr.STATE_PRUNED
    assert plan1["source_report_available"] is False
    assert plan1["superseded_by"] == "mat-plan2"
    assert plan1["report_sha1"]  # digest survives for audit
    src_rows = {
        r["materialization_id"]: r
        for batch in mr._stream_rows(lake, "curation_materializations")
        for r in batch
    }
    assert src_rows["mat-plan1"]["report_json"] == ""
    assert src_rows["mat-exp1"]["report_json"] != ""


def test_retention_soft_retire_without_cutoff_prunes_nothing(tmp_path):
    lake = _bare_lake(tmp_path)
    _seed(
        lake,
        [
            _source_row("mat-plan1", mode="plan", planned=500, output_uri="s3://plan/a", ts=BASE),
            _source_row("mat-plan2", mode="plan", planned=600, output_uri="s3://plan/a", ts=BASE + timedelta(days=1)),
        ],
    )
    report = mr.prune_materialization_rollups(lake, retain_latest=1)
    assert report.pruned_count == 0
    assert report.superseded_ids == ("mat-plan1",)
    assert mr._fetch_rollup_row(lake, "mat-plan1")["state"] == mr.STATE_SUPERSEDED
    assert mr._fetch_rollup_row(lake, "mat-plan2")["state"] == mr.STATE_ACTIVE


def test_retention_dry_run_writes_nothing(tmp_path):
    lake = _bare_lake(tmp_path)
    _seed(
        lake,
        [
            _source_row("mat-plan1", mode="plan", planned=500, output_uri="s3://plan/a", ts=BASE),
            _source_row("mat-plan2", mode="plan", planned=600, output_uri="s3://plan/a", ts=BASE + timedelta(days=1)),
        ],
    )
    report = mr.prune_materialization_rollups(
        lake, retain_latest=1, older_than=timedelta(seconds=1), dry_run=True
    )
    assert report.dry_run is True
    assert report.pruned_count == 1
    # Nothing actually mutated.
    assert mr._fetch_rollup_row(lake, "mat-plan1")["state"] == mr.STATE_ACTIVE


# --------------------------------------------------------------------------- #
# Sync / backfill.
# --------------------------------------------------------------------------- #


def test_sync_rebuilds_from_source_and_preserves_pruned(tmp_path):
    lake = _bare_lake(tmp_path)
    _seed(
        lake,
        [
            _source_row("mat-plan1", mode="plan", planned=500, output_uri="s3://plan/a", ts=BASE),
            _source_row("mat-plan2", mode="plan", planned=600, output_uri="s3://plan/a", ts=BASE + timedelta(days=1)),
        ],
    )
    mr.prune_materialization_rollups(lake, retain_latest=1, older_than=timedelta(seconds=1))
    assert mr._fetch_rollup_row(lake, "mat-plan1")["state"] == mr.STATE_PRUNED

    report = mr.sync_materialization_rollups(lake, build_indexes=False)
    assert report.source_count == 2
    assert report.preserved_pruned_count == 1
    # A resync must never resurrect a pruned body / reset a pruned state.
    assert mr._fetch_rollup_row(lake, "mat-plan1")["state"] == mr.STATE_PRUNED


def test_prune_scans_only_plan_reports_never_the_whole_catalog(tmp_path, monkeypatch):
    # Bound assertion (SKILLS.md §3): retention must never pull completed export
    # evidence into its working set, and must never full-scan the source table to
    # clear a pruned body.
    lake = _bare_lake(tmp_path)
    rows = []
    for i in range(4):  # completed exports -- protected, must not be scanned
        rows.append(_source_row(f"mat-exp{i}", mode="export", copied=500, output_uri=f"s3://exp/{i}"))
    for i in range(3):  # plan reports in one series -- the only prunable set
        rows.append(
            _source_row(f"mat-plan{i}", mode="plan", planned=100, output_uri="s3://plan/a", ts=BASE + timedelta(days=i))
        )
    _seed(lake, rows)

    source_reads: list[str | None] = []
    original = mr._stream_rows

    def _spy(lake_arg, table, *, columns=None, where_sql=None, **kwargs):
        if table == "curation_materializations":
            source_reads.append(where_sql)
        return original(lake_arg, table, columns=columns, where_sql=where_sql, **kwargs)

    monkeypatch.setattr(mr, "_stream_rows", _spy)
    report = mr.prune_materialization_rollups(
        lake, retain_latest=1, older_than=BASE + timedelta(days=5)
    )
    # Only the 3 plan rows entered the working set; the 4 exports are protected.
    assert report.scanned_count == 3
    assert report.protected_count == 4
    # Any source-table read while clearing bodies was scoped by an IN predicate --
    # never an unscoped full-table (blob) scan.
    assert source_reads, "pruning should have cleared at least one source body"
    for clause in source_reads:
        assert clause and "materialization_id IN" in clause


def test_sync_never_preloads_the_whole_rollup_catalog(tmp_path, monkeypatch):
    lake = _bare_lake(tmp_path)
    _seed(lake, [_source_row(f"mat-{i:02d}", copied=10 + i) for i in range(6)])

    rollup_reads: list[str | None] = []
    original = mr._stream_rows

    def _spy(lake_arg, table, *, columns=None, where_sql=None, **kwargs):
        if table == "curation_materialization_rollups":
            rollup_reads.append(where_sql)
        return original(lake_arg, table, columns=columns, where_sql=where_sql, **kwargs)

    monkeypatch.setattr(mr, "_stream_rows", _spy)
    mr.sync_materialization_rollups(lake, build_indexes=False)
    # Every rollup read during sync is a bounded per-batch id lookup, never an
    # unscoped full-catalog preload.
    assert rollup_reads, "sync should look up existing rollup rows"
    for clause in rollup_reads:
        assert clause and "materialization_id IN" in clause


def test_sync_backfills_catalog_for_preexisting_source_rows(tmp_path):
    lake = _bare_lake(tmp_path)
    # Simulate an old lake: source rows written WITHOUT the inline rollup emit.
    src = lake.table("curation_materializations")
    for row, _ in [_source_row("mat-old1", copied=700), _source_row("mat-old2", mode="plan", planned=300)]:
        src.add(pa.Table.from_pylist([row], schema=CURATION_MATERIALIZATIONS_SCHEMA))
    assert lake.table("curation_materialization_rollups").count_rows() == 0

    report = mr.sync_materialization_rollups(lake, build_indexes=False)
    assert report.rebuilt_count == 2
    assert lake.table("curation_materialization_rollups").count_rows() == 2


# --------------------------------------------------------------------------- #
# Integration through lake.curate.materialization_report + compare rewire.
# --------------------------------------------------------------------------- #


def test_materialization_report_emits_rollup_inline(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).snapshot(
        name="candidate-a", split_by="scenario"
    )
    report = lake.curate.materialization_report(
        "candidate-a",
        target_format="webdataset",
        output_uri="s3://exports/candidate-a",
        copied_payload_bytes=0,
        metadata_bytes_written=64,
    )
    entry = mr._fetch_rollup_row(lake, report.materialization_id)
    assert entry is not None
    assert entry["dataset_id"] == report.dataset_id
    assert entry["target_format"] == "webdataset"
    assert entry["metadata_bytes_written"] == 64
    assert entry["state"] == mr.STATE_ACTIVE

    # New SDK surfaces work on the real lake.
    rollup = lake.curate.materialization_rollup(dataset_id=report.dataset_id)
    assert rollup.materialization_count == 1
    history = lake.curate.materialization_history(dataset_id=report.dataset_id)
    assert [e.materialization_id for e in history.records] == [report.materialization_id]


def test_compare_materialization_metric_uses_rollup_catalog(tmp_path, monkeypatch):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).snapshot(
        name="candidate-a", split_by="scenario"
    )
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b", split_by="scenario"
    )
    lake.curate.materialization_report(
        "candidate-b",
        target_format="webdataset",
        output_uri="s3://exports/candidate-b",
        copied_payload_bytes=0,
        metadata_bytes_written=128,
    )
    # The compare "materialization" summary must be served from the rollup catalog
    # (JSON-free) rather than the compat source-streaming fallback. Spy on the
    # rollup fast-path entry point: it must be called and must return a non-None
    # summary (i.e. the fallback source-stream path was never reached).
    from lancedb_robotics import curate as curate_mod

    served: list[bool] = []
    original = curate_mod._rollup_dataset_summary

    def _spy(lake_arg, dataset_id, **kwargs):
        result = original(lake_arg, dataset_id, **kwargs)
        served.append(result is not None)
        return result

    monkeypatch.setattr(curate_mod, "_rollup_dataset_summary", _spy)
    comparison = lake.curate.compare(
        "candidate-a", "candidate-b", metrics=["materialization"]
    )
    assert comparison.report["materialization"]["right"]["materialization_count"] == 1
    # candidate-b's summary came from the rollup catalog (fast path).
    assert served and any(served), "materialization summary must use the rollup catalog"
