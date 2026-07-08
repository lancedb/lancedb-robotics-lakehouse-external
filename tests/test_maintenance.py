"""Lake maintenance tests (backlog 0024)."""

import json
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.indexing import build_scalar_index
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import table_version_artifact_id
from lancedb_robotics.maintenance import MaintenanceError, maintain_lake
from lancedb_robotics.schemas import (
    ALIGNED_TICKS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    RUNS_SCHEMA,
    SCENARIOS_SCHEMA,
    TRAINING_RUNS_SCHEMA,
)


def _add_scenario_fragment(lake: Lake, scenario_id: str, run_id: str = "run-maint") -> None:
    index = int(scenario_id.rsplit("-", 1)[-1])
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": scenario_id,
                    "run_id": run_id,
                    "start_time_ns": index * 10,
                    "end_time_ns": index * 10 + 5,
                    "window_ns": 5,
                    "is_partial": False,
                    "topics": ["/imu"],
                    "observation_ids": [f"obs-{index}"],
                    "observation_count": 1,
                    "summary": f"maintenance scenario {index}",
                }
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )


def _version_exists(lake: Lake, table_name: str, version: int) -> bool:
    assert _version_is_readable(lake, table_name, version)
    return True


def _version_is_readable(lake: Lake, table_name: str, version: int) -> bool:
    table = lake.table(table_name)
    try:
        table.checkout(version)
        table.to_arrow()
        return True
    except Exception:
        return False
    finally:
        try:
            table.checkout_latest()
        except Exception:
            pass


def test_maintenance_compacts_and_preserves_snapshot_pinned_versions(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": "run-maint", "run_kind": "drive", "raw_uri": "/maint.mcap"}],
            schema=RUNS_SCHEMA,
        )
    )
    lake.table("observations").add(
        pa.Table.from_pylist(
            [
                {
                    "observation_id": "obs-0",
                    "run_id": "run-maint",
                    "timestamp_ns": 0,
                    "topic": "/imu",
                    "modality": "imu",
                    "decode_status": "decoded",
                    "raw_sequence": 0,
                }
            ],
            schema=OBSERVATIONS_SCHEMA,
        )
    )
    for index in range(5):
        _add_scenario_fragment(lake, f"scn-{index}")
    snapshot = create_snapshot(
        lake, name="maint-v1", scenario_ids=["scn-0", "scn-1"], split_by="scenario"
    )
    pinned_versions = dict(snapshot.table_versions)
    before_fragments = len(lake.table("scenarios").to_lance().get_fragments())

    report = maintain_lake(
        lake,
        tables=("scenarios",),
        cleanup_older_than=timedelta(seconds=0),
        retain_versions=1,
        delete_unverified=True,
    )

    after_fragments = len(lake.table("scenarios").to_lance().get_fragments())
    assert report.tables["scenarios"].fragments_removed > 0
    assert after_fragments < before_fragments
    assert _version_exists(lake, "scenarios", pinned_versions["scenarios"])

    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == report.transform_id
    )
    assert transform["kind"] == "maintenance"
    params = json.loads(transform["params"])
    assert params["tables"]["scenarios"]["pinned_versions"] == [pinned_versions["scenarios"]]
    assert params["tables"]["scenarios"]["cleanup"]["old_versions"] >= 1


def test_maintenance_preserves_lineage_only_table_version_references(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    _add_scenario_fragment(lake, "scn-0")
    pinned_version = int(lake.table("scenarios").version)
    _add_scenario_fragment(lake, "scn-1")

    table_artifact = lake.lineage.record_artifact(
        kind="table-version",
        artifact_id=table_version_artifact_id("scenarios", pinned_version),
        name=f"scenarios@{pinned_version}",
        table_name="scenarios",
        table_version=pinned_version,
    )
    evidence = lake.lineage.record_artifact(
        kind="evidence-pack",
        name="lineage-only-evidence",
        metadata={"reason": "source replay audit"},
    )
    lake.lineage.record_edge(
        edge_type="version-pinned",
        from_artifact_id=table_artifact.artifact_id,
        to_artifact_id=evidence.artifact_id,
        metadata={"reason": "evidence-pack"},
    )

    report = maintain_lake(
        lake,
        tables=("scenarios",),
        compact=False,
        refresh_indexes=False,
        cleanup_older_than=timedelta(seconds=0),
        retain_versions=1,
        delete_unverified=True,
    )

    table_report = report.tables["scenarios"]
    assert pinned_version in table_report.lineage_pinned_versions
    assert pinned_version in table_report.pinned_versions
    assert table_report.warnings
    assert _version_exists(lake, "scenarios", pinned_version)


def test_maintenance_warns_on_pruned_pinned_versions_without_crashing(tmp_path):
    """BUG-16: lineage can pin a table version that cleanup already pruned."""
    lake = Lake.init(tmp_path / "robot.lance")
    _add_scenario_fragment(lake, "scn-0")
    defunct_version = int(lake.table("scenarios").version)
    _add_scenario_fragment(lake, "scn-1")
    _add_scenario_fragment(lake, "scn-2")

    lake.table("scenarios").to_lance().cleanup_old_versions(
        older_than=timedelta(seconds=0),
        retain_versions=1,
        delete_unverified=True,
    )
    assert not _version_is_readable(lake, "scenarios", defunct_version)

    table_artifact = lake.lineage.record_artifact(
        kind="table-version",
        artifact_id=table_version_artifact_id("scenarios", defunct_version),
        name=f"scenarios@{defunct_version}",
        table_name="scenarios",
        table_version=defunct_version,
    )
    evidence = lake.lineage.record_artifact(
        kind="evidence-pack",
        name="pruned-version-evidence",
        metadata={"reason": "pruned source replay"},
    )
    lake.lineage.record_edge(
        edge_type="version-pinned",
        from_artifact_id=table_artifact.artifact_id,
        to_artifact_id=evidence.artifact_id,
        metadata={"reason": "evidence-pack"},
    )

    report = maintain_lake(
        lake,
        tables=("scenarios",),
        compact=False,
        refresh_indexes=False,
        cleanup_older_than=timedelta(seconds=0),
        retain_versions=1,
        delete_unverified=True,
    )

    table_report = report.tables["scenarios"]
    assert defunct_version in table_report.lineage_pinned_versions
    assert defunct_version in table_report.pinned_versions
    assert f"lbr-snapshot-pin-v{defunct_version}" not in table_report.pinned_tags
    assert any(
        "skipped tagging pruned pinned" in warning and str(defunct_version) in warning
        for warning in table_report.warnings
    )
    assert lake.table("scenarios").count_rows() == 3


def test_maintenance_classifies_training_manifest_reproducibility_pins(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    _add_scenario_fragment(lake, "scn-0")
    pinned_version = int(lake.table("scenarios").version)
    _add_scenario_fragment(lake, "scn-1")

    lake.table("training_runs").add(
        pa.Table.from_pylist(
            [
                {
                    "training_run_id": "train-retention-0067",
                    "dataset_id": "ds-retention-0067",
                    "snapshot_name": "retention-manual",
                    "snapshot_tag": "retention-manual",
                    "table_versions": [
                        {"table": "scenarios", "version": pinned_version, "tag": ""}
                    ],
                    "status": "completed",
                    "manifest_digest": "sha256:retention-0067",
                    "created_by": "test",
                    "created_at": datetime.now(UTC),
                }
            ],
            schema=TRAINING_RUNS_SCHEMA,
        )
    )

    report = maintain_lake(
        lake,
        tables=("scenarios",),
        compact=False,
        refresh_indexes=False,
        cleanup_older_than=timedelta(seconds=0),
        retain_versions=1,
        delete_unverified=True,
    )

    table_report = report.tables["scenarios"]
    retention = {
        item["version"]: set(item["categories"])
        for item in table_report.retention_reasons
    }
    assert pinned_version in table_report.lineage_pinned_versions
    assert "training-reproducibility" in retention[pinned_version]
    assert _version_exists(lake, "scenarios", pinned_version)


def test_maintenance_refreshes_existing_scalar_predicate_indexes(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    now = datetime.now(UTC)
    lake.table("aligned_ticks").add(
        pa.Table.from_pylist(
            [
                {
                    "aligned_tick_id": "tick-0",
                    "alignment_id": "align-0",
                    "alignment_name": "policy_bridge",
                    "recipe_digest": "recipe-0",
                    "run_id": "run-0",
                    "tick_index": 0,
                    "timestamp_ns": 0,
                    "available_streams": [],
                    "missing_streams": [],
                    "interpolated_streams": [],
                    "out_of_tolerance_streams": [],
                    "has_missing": False,
                    "has_out_of_tolerance": False,
                    "min_confidence": 1.0,
                    "quality_flags": [],
                    "stream_detail_json": "{}",
                    "masks_json": "{}",
                    "stream_values_json": "{}",
                    "lineage_json": "{}",
                    "transform_id": "tfm-align-0",
                    "created_at": now,
                }
            ],
            schema=ALIGNED_TICKS_SCHEMA,
        )
    )
    assert (
        build_scalar_index(
            lake,
            table="aligned_ticks",
            column="alignment_id",
        ).status
        == "built"
    )

    report = maintain_lake(
        lake,
        tables=("aligned_ticks",),
        compact=False,
        refresh_indexes=True,
        protect_lineage=False,
        cleanup_older_than=None,
    )

    refreshed = report.tables["aligned_ticks"].indexes_refreshed
    assert refreshed == (
        {
            "table": "aligned_ticks",
            "column": "alignment_id",
            "status": "built",
            "index_type": "BTREE",
            "num_rows": 1,
            "reason": None,
        },
    )


def test_retention_hold_prevents_cleanup_until_expiry(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    _add_scenario_fragment(lake, "scn-0")
    held_version = int(lake.table("scenarios").version)
    _add_scenario_fragment(lake, "scn-1")

    artifact = lake.lineage.record_artifact(
        kind="table-version",
        artifact_id=table_version_artifact_id("scenarios", held_version),
        name=f"scenarios@{held_version}",
        table_name="scenarios",
        table_version=held_version,
    )
    hold = lake.lineage.retain(
        artifact.artifact_id,
        retain_until=datetime.now(UTC) + timedelta(days=1),
        owner="qa",
        reason="audit replay window",
        refresh=False,
    )

    protected = maintain_lake(
        lake,
        tables=("scenarios",),
        compact=False,
        refresh_indexes=False,
        cleanup_older_than=timedelta(seconds=0),
        retain_versions=1,
        delete_unverified=True,
    )

    assert hold.active
    assert held_version in protected.tables["scenarios"].retention_hold_versions
    assert _version_exists(lake, "scenarios", held_version)

    expired = lake.lineage.retain(
        artifact.artifact_id,
        retain_until=datetime.now(UTC) - timedelta(seconds=1),
        owner="qa",
        reason="audit replay window expired",
        refresh=False,
    )
    released = maintain_lake(
        lake,
        tables=("scenarios",),
        compact=False,
        refresh_indexes=False,
        cleanup_older_than=timedelta(seconds=0),
        retain_versions=1,
        delete_unverified=True,
    )

    assert not expired.active
    assert held_version not in released.tables["scenarios"].pinned_versions
    assert not _version_is_readable(lake, "scenarios", held_version)


def test_maintenance_can_require_recent_passed_lineage_audit(tmp_path):
    lake = Lake.init(tmp_path / "audit-gated-maintenance.lance")

    with pytest.raises(MaintenanceError, match="recent passed lineage audit"):
        maintain_lake(
            lake,
            tables=("scenarios",),
            compact=False,
            refresh_indexes=False,
            refresh_lineage=False,
            cleanup_older_than=timedelta(seconds=0),
            require_recent_audit=True,
        )

    report = lake.lineage.audit(refresh=False)
    entry = lake.lineage.record_audit_report(report, created_by="pytest")
    gated = maintain_lake(
        lake,
        tables=("scenarios",),
        compact=False,
        refresh_indexes=False,
        refresh_lineage=False,
        cleanup_older_than=timedelta(seconds=0),
        require_recent_audit=True,
    )

    assert entry.status == "passed"
    assert gated.required_audit_report["report_id"] == entry.report_id
