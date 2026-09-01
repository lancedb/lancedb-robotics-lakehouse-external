"""aligned_ticks compaction and retention lifecycle tests (backlog 0135)."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest
from test_aligned_training_dataset import _aligned_training_lake, _mark_enterprise_lake
from typer.testing import CliRunner

from lancedb_robotics.aligned_tick_lifecycle import (
    LIFECYCLE_REPORT_VERSION,
    LIFECYCLE_TRANSFORM_KIND,
    AlignedTickLifecycleError,
)
from lancedb_robotics.cli import app
from lancedb_robotics.schemas import (
    ALIGNED_FRAMES_SCHEMA,
    ALIGNED_TICKS_SCHEMA,
    DATASET_SNAPSHOTS_SCHEMA,
)


def _add_second_alignment(lake):
    return lake.align.create_view(
        "policy_bridge_25hz",
        run_id="run-aligned-training",
        rate_hz=25.0,
        streams=["/joint_states", "/action"],
        tolerance_ms=100.0,
        interpolation={"/joint_states": "nearest", "/action": "nearest"},
    )


def _tick_rows(lake, alignment_id):
    return [
        row
        for row in lake.table("aligned_ticks").to_arrow().to_pylist()
        if row["alignment_id"] == alignment_id
    ]


def _frame_rows(lake, alignment_id):
    return [
        row
        for row in lake.table("aligned_frames").to_arrow().to_pylist()
        if row["alignment_id"] == alignment_id
    ]


def _inject_tick_duplicate(lake, alignment_id, tick_index, *, recipe_digest=None, created_delta=None):
    """Append a second physical row for one tick_index (a duplicate id)."""
    row = next(
        r for r in _tick_rows(lake, alignment_id) if r["tick_index"] == tick_index
    )
    dup = dict(row)
    if recipe_digest is not None:
        dup["recipe_digest"] = recipe_digest
    if created_delta is not None:
        dup["created_at"] = dup["created_at"] + created_delta
    lake.table("aligned_ticks").add(pa.Table.from_pylist([dup], schema=ALIGNED_TICKS_SCHEMA))
    return dup


def _inject_frame_duplicate(lake, alignment_id, tick_index):
    row = next(
        r for r in _frame_rows(lake, alignment_id) if r["tick_index"] == tick_index
    )
    dup = dict(row)
    dup["created_at"] = dup["created_at"] - timedelta(days=1)
    lake.table("aligned_frames").add(pa.Table.from_pylist([dup], schema=ALIGNED_FRAMES_SCHEMA))
    return dup


def _sample_signature(dataset):
    return [
        {
            "tick_index": sample["tick_index"],
            "streams": {
                stream: {
                    "status": payload["status"],
                    "observation_id": payload["observation_id"],
                    "value": payload["value"],
                }
                for stream, payload in sample["streams"].items()
            },
            "masks": sample["masks"],
        }
        for sample in dataset
    ]


# --------------------------------------------------------------------------- #
# Diagnose
# --------------------------------------------------------------------------- #
def test_diagnose_clean_lake_reports_no_cruft(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")

    report = lake.training.diagnose_aligned_ticks()

    assert report["report_version"] == LIFECYCLE_REPORT_VERSION
    totals = report["totals"]
    assert totals["duplicate_tick_rows"] == 0
    assert totals["stale_tick_rows"] == 0
    assert totals["duplicate_frame_rows"] == 0
    assert totals["orphan_alignments"] == 0
    assert totals["aligned_tick_rows"] == totals["distinct_tick_ids"] > 0
    assert totals["aligned_frame_rows"] == totals["distinct_frame_ids"] > 0
    tick_table = report["tables"]["aligned_ticks"]
    assert tick_table["exists"] is True
    assert tick_table["rows"] == totals["aligned_tick_rows"]
    assert "version_cleanup" in report and "lake maintain" in report["version_cleanup"]
    assert report["alignments"][0]["current_recipe_digest"].startswith("recipe-")


def test_diagnose_missing_tables_reports_absent(tmp_path):
    from lancedb_robotics.lake import Lake

    lake = Lake.init(tmp_path / "empty.lance")

    report = lake.training.diagnose_aligned_ticks()

    assert report["tables"]["aligned_ticks"] == {"exists": False} or (
        report["tables"]["aligned_ticks"]["rows"] == 0
    )
    assert report["totals"]["aligned_tick_rows"] == 0
    assert report["alignments"] == []


def test_diagnose_counts_duplicates_without_changing_distinct(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _inject_tick_duplicate(lake, view.alignment_id, 0, created_delta=-timedelta(days=1))
    _inject_tick_duplicate(lake, view.alignment_id, 1, created_delta=-timedelta(days=1))

    report = lake.training.diagnose_aligned_ticks()

    alignment = report["alignments"][0]
    assert alignment["aligned_tick_rows"] == 5
    assert alignment["distinct_tick_ids"] == 3
    assert alignment["duplicate_tick_rows"] == 2
    assert report["totals"]["duplicate_tick_rows"] == 2


# --------------------------------------------------------------------------- #
# Cleanup: dry-run vs apply
# --------------------------------------------------------------------------- #
def test_cleanup_dry_run_reports_plan_writes_nothing(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _inject_tick_duplicate(lake, view.alignment_id, 0, created_delta=-timedelta(days=1))
    rows_before = lake.table("aligned_ticks").count_rows()

    report = lake.training.cleanup_aligned_ticks(dry_run=True)

    assert report["dry_run"] is True
    assert report["plan"]["duplicate_tick_rows_removable"] == 1
    assert "applied" not in report
    assert lake.table("aligned_ticks").count_rows() == rows_before
    assert not [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["kind"] == LIFECYCLE_TRANSFORM_KIND
    ]


def test_cleanup_collapses_duplicates_preserving_samples(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _add_second_alignment(lake)
    expected_first = _sample_signature(lake.training.aligned_dataset(name="policy_bridge"))
    expected_second = _sample_signature(
        lake.training.aligned_dataset(name="policy_bridge_25hz")
    )
    for tick in (0, 1, 2):
        _inject_tick_duplicate(lake, view.alignment_id, tick, created_delta=-timedelta(hours=1))

    report = lake.training.cleanup_aligned_ticks(dry_run=False)

    applied = report["applied"]
    assert applied["duplicate_tick_rows_removed"] == 3
    assert applied["transform_id"].startswith("tfm-atl-")
    rows = _tick_rows(lake, view.alignment_id)
    assert len(rows) == len({row["aligned_tick_id"] for row in rows}) == 3
    # Compatibility samples for both alignments are unchanged.
    assert _sample_signature(lake.training.aligned_dataset(name="policy_bridge")) == expected_first
    assert (
        _sample_signature(lake.training.aligned_dataset(name="policy_bridge_25hz"))
        == expected_second
    )
    # Auditable transform_runs row recorded.
    lifecycle_runs = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["kind"] == LIFECYCLE_TRANSFORM_KIND
    ]
    assert len(lifecycle_runs) == 1


def test_cleanup_keeps_current_recipe_over_newer_stale_copy(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    current = lake.training.diagnose_aligned_ticks()["alignments"][0]["current_recipe_digest"]
    # A stale-recipe copy written *after* the fresh row must not win canonical.
    _inject_tick_duplicate(
        lake,
        view.alignment_id,
        0,
        recipe_digest="recipe-staleaaaaaaaa",
        created_delta=timedelta(days=1),
    )

    diag = lake.training.diagnose_aligned_ticks()["alignments"][0]
    assert diag["duplicate_tick_rows"] == 1
    assert diag["stale_tick_rows"] == 1

    lake.training.cleanup_aligned_ticks(dry_run=False)

    survivors = [row for row in _tick_rows(lake, view.alignment_id) if row["tick_index"] == 0]
    assert len(survivors) == 1
    assert survivors[0]["recipe_digest"] == current


def test_cleanup_is_idempotent(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _inject_tick_duplicate(lake, view.alignment_id, 0, created_delta=-timedelta(days=1))

    first = lake.training.cleanup_aligned_ticks(dry_run=False)
    assert first["applied"]["duplicate_tick_rows_removed"] == 1

    second = lake.training.cleanup_aligned_ticks(dry_run=False)
    assert second["applied"]["duplicate_tick_rows_removed"] == 0
    assert second["applied"]["maintenance"] is None  # nothing changed => no compaction


def _lifecycle_transform_rows(lake):
    return [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["kind"] == LIFECYCLE_TRANSFORM_KIND
    ]


def test_noop_apply_writes_no_transform_row(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")

    # Nothing to remove: an apply must not append an audit row or churn lineage.
    report = lake.training.cleanup_aligned_ticks(dry_run=False)
    assert report["applied"]["transform_id"] is None
    assert _lifecycle_transform_rows(lake) == []

    # A real removal records exactly one content-addressed row.
    _inject_tick_duplicate(lake, view.alignment_id, 0, created_delta=-timedelta(days=1))
    lake.training.cleanup_aligned_ticks(dry_run=False)
    assert len(_lifecycle_transform_rows(lake)) == 1

    # A subsequent no-op apply does not add another row.
    lake.training.cleanup_aligned_ticks(dry_run=False)
    assert len(_lifecycle_transform_rows(lake)) == 1


def test_cleanup_converges_with_null_created_at_duplicate(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    # A writer that left created_at unset must still be collapsed, not reported
    # as removed while physically surviving.
    row = next(r for r in _tick_rows(lake, view.alignment_id) if r["tick_index"] == 0)
    dup = dict(row)
    dup["created_at"] = None
    lake.table("aligned_ticks").add(pa.Table.from_pylist([dup], schema=ALIGNED_TICKS_SCHEMA))
    assert lake.table("aligned_ticks").count_rows(f"alignment_id = '{view.alignment_id}'") == 4

    report = lake.training.cleanup_aligned_ticks(dry_run=False)
    assert report["applied"]["duplicate_tick_rows_removed"] == 1

    survivors = [r for r in _tick_rows(lake, view.alignment_id) if r["tick_index"] == 0]
    assert len(survivors) == 1  # physically converged, not a false-positive removal
    # Idempotent: a re-run finds nothing.
    assert lake.training.cleanup_aligned_ticks(dry_run=False)["applied"][
        "duplicate_tick_rows_removed"
    ] == 0


def test_cleanup_converges_on_repeated_stragglers(tmp_path):
    """Simulates the crash/second-writer window: a straggler appearing after a
    prior collapse is still collapsed on the next run, never dropping to zero."""
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    for _ in range(3):
        _inject_tick_duplicate(lake, view.alignment_id, 0, created_delta=-timedelta(hours=1))
        lake.training.cleanup_aligned_ticks(dry_run=False)
        survivors = [r for r in _tick_rows(lake, view.alignment_id) if r["tick_index"] == 0]
        assert len(survivors) == 1


def test_diagnose_does_not_load_whole_lineage_graph(tmp_path, monkeypatch):
    """The bounded-memory guarantee (no BUG-13 whole-graph load) is pinned: the
    diagnose/cleanup pin lookup must not call the prune-time whole-graph helper."""
    import lancedb_robotics.lineage as lineage_mod

    lake, view = _aligned_training_lake(tmp_path / "robot.lance")

    def _boom(*args, **kwargs):
        raise AssertionError("lineage_retention_pin_details loads the whole graph")

    monkeypatch.setattr(lineage_mod, "lineage_retention_pin_details", _boom)
    # Must succeed without touching the whole-graph helper.
    report = lake.training.diagnose_aligned_ticks()
    assert report["report_version"] == LIFECYCLE_REPORT_VERSION


def test_cleanup_deduplicates_compatibility_frames(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _inject_frame_duplicate(lake, view.alignment_id, 0)
    assert lake.training.diagnose_aligned_ticks()["totals"]["duplicate_frame_rows"] == 1

    report = lake.training.cleanup_aligned_ticks(dry_run=False)

    assert report["applied"]["duplicate_frame_rows_removed"] == 1
    frames = _frame_rows(lake, view.alignment_id)
    assert len(frames) == len({row["aligned_frame_id"] for row in frames})


def test_cleanup_no_include_frames_leaves_frames_untouched(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _inject_frame_duplicate(lake, view.alignment_id, 0)
    frames_before = lake.table("aligned_frames").count_rows()

    report = lake.training.cleanup_aligned_ticks(dry_run=False, include_frames=False)

    assert report["applied"]["duplicate_frame_rows_removed"] == 0
    assert lake.table("aligned_frames").count_rows() == frames_before


# --------------------------------------------------------------------------- #
# Orphans
# --------------------------------------------------------------------------- #
def test_orphans_conservative_by_default(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    lake.table("alignment_jobs").delete(f"alignment_id = '{view.alignment_id}'")

    diag = lake.training.diagnose_aligned_ticks()
    assert diag["totals"]["orphan_alignments"] == 1
    assert diag["alignments"][0]["orphan"] is True

    plan = lake.training.cleanup_aligned_ticks(dry_run=True)["plan"]
    assert plan["orphan_tick_rows_removable"] == 0
    assert plan["orphan_alignments_retained"] == 1
    # Default apply must not remove orphan rows.
    applied = lake.training.cleanup_aligned_ticks(dry_run=False)["applied"]
    assert applied["orphan_tick_rows_removed"] == 0
    assert lake.table("aligned_ticks").count_rows(f"alignment_id = '{view.alignment_id}'") == 3


def test_remove_orphans_opt_in_removes_rows(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    lake.table("alignment_jobs").delete(f"alignment_id = '{view.alignment_id}'")

    report = lake.training.cleanup_aligned_ticks(dry_run=False, remove_orphans=True)

    applied = report["applied"]
    assert applied["orphan_tick_rows_removed"] == 3
    assert applied["orphan_frame_rows_removed"] == 9
    assert applied["orphan_alignments_removed"] == 1
    assert lake.table("aligned_ticks").count_rows(f"alignment_id = '{view.alignment_id}'") == 0


# --------------------------------------------------------------------------- #
# Selection + protection
# --------------------------------------------------------------------------- #
def test_alignment_selection_by_name_and_unknown_error(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _add_second_alignment(lake)

    report = lake.training.diagnose_aligned_ticks(alignments=["policy_bridge"])
    assert [a["alignment_name"] for a in report["alignments"]] == ["policy_bridge"]

    with pytest.raises(AlignedTickLifecycleError):
        lake.training.diagnose_aligned_ticks(alignments=["nope"])


def test_pinned_versions_surfaced_and_never_pruned(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _inject_tick_duplicate(lake, view.alignment_id, 0, created_delta=-timedelta(days=1))
    pinned_version = int(lake.table("aligned_ticks").version)
    # A dataset snapshot pinning the current aligned_ticks version.
    lake.table("dataset_snapshots").add(
        pa.Table.from_pylist(
            [
                {
                    "dataset_id": "ds-pin",
                    "name": "pinned",
                    "kind": "search",
                    "query_spec": "{}",
                    "table_versions": [
                        {"table": "aligned_ticks", "version": pinned_version, "tag": ""}
                    ],
                    "tag": "",
                    "split": "",
                    "balance_report": "",
                    "coverage_report": "",
                    "created_by": "test",
                    "transform_id": "tfm-x",
                    "created_at": datetime(2026, 1, 2, tzinfo=UTC),
                }
            ],
            schema=DATASET_SNAPSHOTS_SCHEMA,
        )
    )

    report = lake.training.cleanup_aligned_ticks(dry_run=False)

    pinned = report["tables"]["aligned_ticks"]["pinned_versions"]
    assert any(entry["version"] == pinned_version for entry in pinned)
    # Row-level cleanup happened, but the pinned version's rows remain readable.
    pinned_rows = (
        lake.table("aligned_ticks")
        .to_lance()
        .checkout_version(pinned_version)
        .to_table()
        .to_pylist()
    )
    assert len([r for r in pinned_rows if r["alignment_id"] == view.alignment_id]) == 4


# --------------------------------------------------------------------------- #
# Maintenance integration
# --------------------------------------------------------------------------- #
def test_maintenance_surfaces_lifecycle_diagnostics(tmp_path):
    from lancedb_robotics.maintenance import maintain_lake

    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _inject_tick_duplicate(lake, view.alignment_id, 0, created_delta=-timedelta(days=1))

    report = maintain_lake(
        lake,
        tables=("aligned_ticks",),
        cleanup_older_than=None,
    )

    assert report.aligned_tick_lifecycle is not None
    assert report.aligned_tick_lifecycle["totals"]["duplicate_tick_rows"] == 1


def test_maintenance_can_disable_lifecycle_diagnostics(tmp_path):
    from lancedb_robotics.maintenance import maintain_lake

    lake, view = _aligned_training_lake(tmp_path / "robot.lance")

    report = maintain_lake(
        lake,
        tables=("aligned_ticks",),
        cleanup_older_than=None,
        aligned_tick_lifecycle_diagnostics=False,
    )

    assert report.aligned_tick_lifecycle is None


# --------------------------------------------------------------------------- #
# Remote/enterprise capability gating
# --------------------------------------------------------------------------- #
def test_cleanup_maintenance_skips_on_remote_backend(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _inject_tick_duplicate(lake, view.alignment_id, 0, created_delta=-timedelta(days=1))
    _mark_enterprise_lake(lake)

    report = lake.training.cleanup_aligned_ticks(dry_run=False)

    # Rows were still deduplicated, but compaction/index refresh is skipped with
    # an explicit reason rather than failing.
    assert report["applied"]["duplicate_tick_rows_removed"] == 1
    maintenance = report["applied"]["maintenance"]
    assert maintenance["status"] == "skipped"
    assert maintenance["reason"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_diagnose_and_cleanup_ticks(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _inject_tick_duplicate(lake, view.alignment_id, 0, created_delta=-timedelta(days=1))
    runner = CliRunner()

    diagnosed = runner.invoke(
        app, ["align", "diagnose-ticks", "--lake", str(tmp_path / "robot.lance"), "--json"]
    )
    assert diagnosed.exit_code == 0, diagnosed.output
    assert "aligned-tick-lifecycle/1" in diagnosed.output

    # Dry-run (default) leaves rows in place.
    dry = runner.invoke(
        app, ["align", "cleanup-ticks", "--lake", str(tmp_path / "robot.lance")]
    )
    assert dry.exit_code == 0, dry.output
    assert "(dry-run)" in dry.output
    assert lake.table("aligned_ticks").count_rows() == 4

    applied = runner.invoke(
        app, ["align", "cleanup-ticks", "--lake", str(tmp_path / "robot.lance"), "--apply"]
    )
    assert applied.exit_code == 0, applied.output
    assert "(applied)" in applied.output
    assert lake.table("aligned_ticks").count_rows() == 3
