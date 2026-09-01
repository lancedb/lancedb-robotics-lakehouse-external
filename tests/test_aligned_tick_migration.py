"""Batch aligned_ticks migration and validation tests (backlog 0134)."""

import json

import pyarrow as pa
import pytest
from test_aligned_training_dataset import _aligned_training_lake, _mark_enterprise_lake

from lancedb_robotics.aligned_tick_migration import (
    MIGRATION_REPORT_VERSION,
    MIGRATION_TRANSFORM_KIND,
    AlignedTickMigrationError,
    migrate_aligned_ticks,
)
from lancedb_robotics.schemas import ALIGNED_TICKS_SCHEMA


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


def _delete_ticks(lake, alignment_id):
    lake.table("aligned_ticks").delete(f"alignment_id = '{alignment_id}'")


def _sample_signature(dataset):
    return [
        {
            "tick_index": sample["tick_index"],
            "streams": {
                stream: {
                    "status": payload["status"],
                    "observation_id": payload["observation_id"],
                    "source_observation_ids": payload["source_observation_ids"],
                    "source_row_ids": payload["source_row_ids"],
                    "value": payload["value"],
                }
                for stream, payload in sample["streams"].items()
            },
            "masks": sample["masks"],
        }
        for sample in dataset
    ]


def test_migrate_aligned_ticks_batch_migrates_legacy_jobs(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    second = _add_second_alignment(lake)
    expected_first = _sample_signature(lake.training.aligned_dataset(name="policy_bridge"))
    expected_second = _sample_signature(lake.training.aligned_dataset(name="policy_bridge_25hz"))
    _delete_ticks(lake, view.alignment_id)
    _delete_ticks(lake, second.alignment_id)

    report = lake.training.migrate_aligned_ticks()

    assert report["report_version"] == MIGRATION_REPORT_VERSION
    assert report["jobs_scanned"] == 2
    assert report["jobs_migrated"] == 2
    assert report["jobs_failed"] == 0
    assert report["aligned_ticks_written"] > 0
    assert report["validation"] == {
        "metadata_mismatches": 0,
        "jsonb_failures": 0,
        "summary_mismatches": 0,
    }
    statuses = {job["alignment_id"]: job["status"] for job in report["jobs"]}
    assert statuses == {view.alignment_id: "migrated", second.alignment_id: "migrated"}

    after_first = lake.training.aligned_dataset(name="policy_bridge")
    after_second = lake.training.aligned_dataset(name="policy_bridge_25hz")
    assert after_first.manifest.storage_backend == "aligned_ticks-jsonb"
    assert after_second.manifest.storage_backend == "aligned_ticks-jsonb"
    assert _sample_signature(after_first) == expected_first
    assert _sample_signature(after_second) == expected_second


def test_migrate_aligned_ticks_dry_run_writes_nothing(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _delete_ticks(lake, view.alignment_id)

    plan = lake.training.migrate_aligned_ticks(dry_run=True)

    assert plan["dry_run"] is True
    assert plan["jobs_planned"] == 1
    assert plan["aligned_ticks_written"] == 3
    assert _tick_rows(lake, view.alignment_id) == []
    assert not [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["kind"] == MIGRATION_TRANSFORM_KIND
    ]

    report = lake.training.migrate_aligned_ticks()
    assert report["aligned_ticks_written"] == plan["aligned_ticks_written"]
    assert report["jobs"][0]["ticks_expected"] == plan["jobs"][0]["ticks_expected"]


def test_migrate_aligned_ticks_resumes_partial_migration(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    surviving = [row for row in _tick_rows(lake, view.alignment_id) if row["tick_index"] == 0]
    lake.table("aligned_ticks").delete(f"alignment_id = '{view.alignment_id}' AND tick_index >= 1")

    report = lake.training.migrate_aligned_ticks()

    job = report["jobs"][0]
    assert job["status"] == "resumed"
    assert job["ticks_existing"] == 1
    assert job["ticks_written"] == 2
    rows = _tick_rows(lake, view.alignment_id)
    assert len(rows) == 3
    preserved = next(row for row in rows if row["tick_index"] == 0)
    assert preserved["aligned_tick_id"] == surviving[0]["aligned_tick_id"]
    assert preserved["created_at"] == surviving[0]["created_at"]


def test_migrate_aligned_ticks_second_run_is_idempotent(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _delete_ticks(lake, view.alignment_id)
    first = lake.training.migrate_aligned_ticks()
    assert first["jobs_migrated"] == 1

    second = lake.training.migrate_aligned_ticks()

    assert second["jobs_migrated"] == 0
    assert second["jobs_already_migrated"] == 1
    assert second["aligned_ticks_written"] == 0
    assert len(_tick_rows(lake, view.alignment_id)) == 3


def test_migrate_aligned_ticks_selects_requested_alignments(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    second = _add_second_alignment(lake)
    _delete_ticks(lake, view.alignment_id)
    _delete_ticks(lake, second.alignment_id)

    report = lake.training.migrate_aligned_ticks(alignments=["policy_bridge_25hz"])

    assert report["jobs_scanned"] == 1
    assert report["jobs"][0]["alignment_id"] == second.alignment_id
    assert _tick_rows(lake, view.alignment_id) == []

    with pytest.raises(AlignedTickMigrationError, match="unknown alignment"):
        lake.training.migrate_aligned_ticks(alignments=["no-such-alignment"])


def test_migrate_aligned_ticks_missing_table_requires_explicit_creation(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    lake._db.drop_table("aligned_ticks")

    report = lake.training.migrate_aligned_ticks()

    job = report["jobs"][0]
    assert job["status"] == "skipped"
    assert "create_missing_table=True" in job["reason"]
    assert "aligned_ticks" not in lake.table_names()

    created = lake.training.migrate_aligned_ticks(create_missing_table=True)
    assert created["aligned_ticks_table_created"] is True
    assert created["jobs_migrated"] == 1
    assert len(_tick_rows(lake, view.alignment_id)) == 3


def test_migrate_aligned_ticks_capability_gates_remote_table_creation(tmp_path):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    lake._db.drop_table("aligned_ticks")
    _mark_enterprise_lake(lake)

    report = lake.training.migrate_aligned_ticks(create_missing_table=True)

    job = report["jobs"][0]
    assert job["status"] == "skipped"
    assert "schema_evolution" in job["reason"]
    assert report["jobs_failed"] == 0
    assert "aligned_ticks" not in lake.table_names()


def test_migrate_aligned_ticks_validation_reports_corrupt_rows(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    row = next(row for row in _tick_rows(lake, view.alignment_id) if row["tick_index"] == 1)
    lake.table("aligned_ticks").delete(f"aligned_tick_id = '{row['aligned_tick_id']}'")
    corrupted = dict(row)
    corrupted["missing_streams"] = []
    lake.table("aligned_ticks").add(pa.Table.from_pylist([corrupted], schema=ALIGNED_TICKS_SCHEMA))

    report = lake.training.migrate_aligned_ticks()

    job = report["jobs"][0]
    assert job["status"] == "failed"
    assert job["validation"]["summary_mismatches"] == 1
    failure = job["validation_failures"][0]
    assert failure["alignment_id"] == view.alignment_id
    assert failure["tick_index"] == 1
    assert "missing_streams" in failure["detail"]
    assert "replace=True" in job["reason"]

    repaired = lake.training.migrate_aligned_ticks(replace=True)
    assert repaired["jobs_migrated"] == 1
    assert repaired["validation"] == {
        "metadata_mismatches": 0,
        "jsonb_failures": 0,
        "summary_mismatches": 0,
    }


def test_migrate_aligned_ticks_validation_reports_jsonb_failures(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    row = next(row for row in _tick_rows(lake, view.alignment_id) if row["tick_index"] == 2)
    lake.table("aligned_ticks").delete(f"aligned_tick_id = '{row['aligned_tick_id']}'")
    # Lance validates JSONB syntax on write, so malformed JSON cannot even be
    # stored; a valid-JSON non-object payload is the storable corruption that
    # must fail the object round-trip.
    corrupted = dict(row)
    corrupted["stream_detail_json"] = "[1, 2, 3]"
    lake.table("aligned_ticks").add(pa.Table.from_pylist([corrupted], schema=ALIGNED_TICKS_SCHEMA))

    report = lake.training.migrate_aligned_ticks()

    job = report["jobs"][0]
    assert job["status"] == "failed"
    assert job["validation"]["jsonb_failures"] == 1
    failure = next(
        item for item in job["validation_failures"] if item["kind"] == "jsonb-round-trip"
    )
    assert failure["tick_index"] == 2
    assert failure["aligned_tick_id"] == row["aligned_tick_id"]


def test_migrate_aligned_ticks_skips_stale_recipe_rows_without_replace(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    row = next(row for row in _tick_rows(lake, view.alignment_id) if row["tick_index"] == 0)
    lake.table("aligned_ticks").delete(f"aligned_tick_id = '{row['aligned_tick_id']}'")
    stale = dict(row)
    stale["recipe_digest"] = "recipe-stale"
    lake.table("aligned_ticks").add(pa.Table.from_pylist([stale], schema=ALIGNED_TICKS_SCHEMA))

    report = lake.training.migrate_aligned_ticks()

    job = report["jobs"][0]
    assert job["status"] == "skipped"
    assert "recipe-stale" in job["reason"]
    assert "replace=True" in job["reason"]

    repaired = lake.training.migrate_aligned_ticks(replace=True)
    assert repaired["jobs_migrated"] == 1
    rows = _tick_rows(lake, view.alignment_id)
    assert len(rows) == 3
    assert {row["recipe_digest"] for row in rows} != {"recipe-stale"}


def test_migrate_aligned_ticks_records_transform_lineage(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _delete_ticks(lake, view.alignment_id)

    report = lake.training.migrate_aligned_ticks()

    job = report["jobs"][0]
    transform_rows = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["kind"] == MIGRATION_TRANSFORM_KIND
    ]
    assert len(transform_rows) == 1
    transform = transform_rows[0]
    assert transform["transform_id"] == job["transform_id"]
    assert transform["status"] == "completed"
    assert transform["output_tables"] == ["aligned_ticks"]
    params = json.loads(transform["params"])
    assert params["alignment_id"] == view.alignment_id
    assert params["ticks_written"] == 3
    assert params["report_version"] == MIGRATION_REPORT_VERSION

    lake.training.migrate_aligned_ticks(replace=True)
    transform_rows = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["kind"] == MIGRATION_TRANSFORM_KIND
    ]
    assert len(transform_rows) == 1


def test_migrate_aligned_ticks_bounded_windows_cover_all_ticks(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _delete_ticks(lake, view.alignment_id)

    report = lake.training.migrate_aligned_ticks(tick_window=1, batch_size=1)

    job = report["jobs"][0]
    assert job["status"] == "migrated"
    assert job["ticks_written"] == 3
    assert job["validation"] == {
        "metadata_mismatches": 0,
        "jsonb_failures": 0,
        "summary_mismatches": 0,
    }
    assert len(_tick_rows(lake, view.alignment_id)) == 3


def test_migrate_aligned_ticks_concurrent_write_race_leaves_no_duplicates(tmp_path, monkeypatch):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _delete_ticks(lake, view.alignment_id)
    import lancedb_robotics.aligned_tick_migration as migration_mod

    # The BUG-04 interleaving: both migrators read "nothing stored yet" before
    # either writes. The insert-only merge keyed on the content-addressed
    # aligned_tick_id must make the loser's writes no-ops.
    monkeypatch.setattr(migration_mod, "_existing_tick_indices", lambda lake, job, window: set())

    first = lake.training.migrate_aligned_ticks()
    second = lake.training.migrate_aligned_ticks()

    rows = _tick_rows(lake, view.alignment_id)
    assert len(rows) == 3
    assert len({row["aligned_tick_id"] for row in rows}) == 3
    assert first["jobs_failed"] == 0
    assert second["jobs_failed"] == 0


def test_migrate_aligned_ticks_reads_frames_in_bounded_windows(tmp_path, monkeypatch):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _delete_ticks(lake, view.alignment_id)
    import lancedb_robotics.aligned_tick_migration as migration_mod

    windows: list[tuple[int, int]] = []
    real_scan = migration_mod._scan_frame_window

    def spy(lake, job, streams, window):
        windows.append(window)
        return real_scan(lake, job, streams, window)

    monkeypatch.setattr(migration_mod, "_scan_frame_window", spy)

    report = lake.training.migrate_aligned_ticks(tick_window=2)

    # Ticks 0..2 with a 2-tick window => exactly the two occupied windows, so
    # the mechanism (windowed range scans, never a whole-job scan) is pinned.
    assert windows == [(0, 2), (2, 4)]
    assert report["jobs"][0]["ticks_written"] == 3


def test_migrate_aligned_ticks_runs_post_sweep_maintenance(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _delete_ticks(lake, view.alignment_id)

    report = lake.training.migrate_aligned_ticks()

    maintenance = report["maintenance"]
    assert maintenance["status"] == "completed"
    assert maintenance["table"] == "aligned_ticks"
    assert maintenance["compaction"] is not None
    assert maintenance["indexes"]
    assert {index["status"] for index in maintenance["indexes"]} <= {"built", "already_present"}

    # A sweep that writes nothing has nothing to compact.
    again = lake.training.migrate_aligned_ticks()
    assert again["maintenance"] is None


def test_migrate_aligned_ticks_maintenance_skips_with_reason_on_gated_backend(tmp_path):
    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _delete_ticks(lake, view.alignment_id)
    _mark_enterprise_lake(lake)

    report = lake.training.migrate_aligned_ticks()

    assert report["jobs_migrated"] == 1
    maintenance = report["maintenance"]
    assert maintenance["status"] == "skipped"
    assert "direct_object_io" in maintenance["reason"]


def test_migrate_aligned_ticks_rejects_invalid_bounds(tmp_path):
    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")

    with pytest.raises(AlignedTickMigrationError, match="tick_window"):
        migrate_aligned_ticks(lake, tick_window=0)
    with pytest.raises(AlignedTickMigrationError, match="batch_size"):
        migrate_aligned_ticks(lake, batch_size=0)


def test_migrate_ticks_cli_reports_and_json(tmp_path):
    from typer.testing import CliRunner

    from lancedb_robotics.cli import app

    lake, view = _aligned_training_lake(tmp_path / "robot.lance")
    _delete_ticks(lake, view.alignment_id)
    runner = CliRunner()

    dry = runner.invoke(
        app,
        ["align", "migrate-ticks", "--lake", str(tmp_path / "robot.lance"), "--dry-run"],
    )
    assert dry.exit_code == 0, dry.output
    assert "planned" in dry.output
    assert _tick_rows(lake, view.alignment_id) == []

    result = runner.invoke(
        app,
        ["align", "migrate-ticks", "--lake", str(tmp_path / "robot.lance"), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["report_version"] == MIGRATION_REPORT_VERSION
    assert payload["jobs_migrated"] == 1
    assert len(_tick_rows(lake, view.alignment_id)) == 3
