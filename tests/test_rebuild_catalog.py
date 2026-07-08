"""Durable rebuild-plan catalog, approvals, and orchestrator handoff tests (0109)."""

import json

import pytest
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.rebuild_catalog import (
    CATALOG_SCHEMA_VERSION,
    CATALOG_TABLE,
    DISPATCH_SCHEMA_VERSION,
    EVENTS_TABLE,
    RebuildPlanCatalogError,
    RebuildPlanConflict,
    export_rebuild_plan_dispatch,
    get_rebuild_plan,
    rebuild_plan_events,
    rebuild_plans,
    record_rebuild_plan,
    update_rebuild_plan_status,
)
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.writeback import ingest_model_outputs

runner = CliRunner()


def _lake(tmp_path, fixtures_dir, fixture_name="sample.mcap"):
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
    training = lake.training.record_run("demo-v1", training_run_id="train-0109")
    lake.training.record_checkpoint(
        training_run_id=training.training_run_id,
        model_artifact_id="policy-0109",
        artifact_uri="s3://models/policy-0109.ckpt",
    )
    lake.eval.record_run(
        "demo-v1",
        model_artifact_id="policy-0109",
        eval_run_id="eval-0109",
        metrics={"success_rate": 0.5},
    )
    lake.lineage.refresh_graph()
    return lake, manifest


def _source_uri(lake):
    return lake.table("runs").to_arrow().to_pylist()[0]["raw_uri"]


def _plan(lake, *, record_invalidation=False, reason="camera calibration bug"):
    return lake.lineage.rebuild_plan(
        _source_uri(lake),
        kind="source",
        reason=reason,
        severity="high",
        discovered_by="qa",
        actor="robotics-ops",
        record_invalidation=record_invalidation,
    )


# --- Save / reload / digest stability ---------------------------------------


def test_record_reload_and_digest_are_stable(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    plan = _plan(lake)

    entry = record_rebuild_plan(lake, plan, actor="ops")
    assert entry.plan_id == entry.plan_digest
    assert entry.status == "draft"
    assert entry.revision == 0
    assert entry.action_count == len(plan.actions)
    assert entry.catalog_schema_version == CATALOG_SCHEMA_VERSION

    reloaded, payload = get_rebuild_plan(lake, entry.plan_id)
    assert reloaded.plan_id == entry.plan_id
    assert payload["schema_version"] == "lancedb-robotics/rebuild-plan/v1"
    assert "graph" not in payload  # bounded payload drops the impact graph
    assert len(payload["actions"]) == entry.action_count

    # Re-recording the identical plan is idempotent and preserves the row.
    again = record_rebuild_plan(lake, plan, actor="someone-else")
    assert again.plan_id == entry.plan_id
    assert again.created_at == entry.created_at
    assert lake.table(CATALOG_TABLE).count_rows() == 1


def test_digest_independent_of_invalidation_timestamp(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)

    # Both plans compute their actions from the same graph (the marker is written
    # only after the actions are built), so they differ only in the timestamped
    # ``.invalidation`` field that record_invalidation attaches.
    plan_report = _plan(lake, record_invalidation=False)
    plan_marked = _plan(lake, record_invalidation=True)
    assert plan_report.invalidation is None
    assert plan_marked.invalidation is not None
    assert plan_marked.invalidation.created_at is not None
    assert [a.as_dict() for a in plan_report.actions] == [a.as_dict() for a in plan_marked.actions]

    entry_report = record_rebuild_plan(lake, plan_report)
    entry_marked = record_rebuild_plan(lake, plan_marked)
    # The plan digest is content-addressed on the actions and excludes the
    # invalidation marker, so both hash to one idempotent row.
    assert entry_report.plan_digest == entry_marked.plan_digest
    assert lake.table(CATALOG_TABLE).count_rows() == 1


def test_record_accepts_plan_mapping(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    plan = _plan(lake)
    from_object = record_rebuild_plan(lake, plan)
    from_mapping = record_rebuild_plan(lake, plan.as_dict())
    assert from_object.plan_digest == from_mapping.plan_digest


def test_record_rejects_unsupported_schema(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    with pytest.raises(RebuildPlanCatalogError):
        record_rebuild_plan(lake, {"schema_version": "bogus/v9", "actions": []})


def test_report_only_rebuild_plan_writes_no_catalog_row(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    plan = lake.lineage.rebuild_plan(_source_uri(lake), kind="source", reason="x")
    assert plan.actions  # still a report-only RebuildPlan
    assert lake.table(CATALOG_TABLE).count_rows() == 0


# --- Lifecycle / approvals / optimistic concurrency -------------------------


def test_status_transitions_bump_revision_and_append_events(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake))

    approved = update_rebuild_plan_status(lake, entry.plan_id, "approved", approver="lead")
    assert approved.status == "approved"
    assert approved.revision == 1
    assert approved.approved_at is not None

    dispatched = update_rebuild_plan_status(
        lake, entry.plan_id, "dispatched", expected_revision=1
    )
    assert dispatched.status == "dispatched"
    assert dispatched.revision == 2
    assert dispatched.dispatched_at is not None

    completed = update_rebuild_plan_status(lake, entry.plan_id, "completed")
    assert completed.status == "completed"
    assert completed.completed_at is not None

    events = rebuild_plan_events(lake, plan_id=entry.plan_id)
    types = [event["event_type"] for event in events]
    assert types.count("recorded") == 1
    assert types.count("status-changed") == 3
    # append-only log preserves the full transition history
    to_statuses = [event["to_status"] for event in events if event["event_type"] == "status-changed"]
    assert to_statuses == ["approved", "dispatched", "completed"]


def test_approval_requires_approver(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake))
    with pytest.raises(RebuildPlanCatalogError, match="approver"):
        update_rebuild_plan_status(lake, entry.plan_id, "approved")


def test_illegal_transition_reports_allowed_states(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake))
    with pytest.raises(RebuildPlanCatalogError, match="illegal rebuild-plan transition"):
        update_rebuild_plan_status(lake, entry.plan_id, "completed")


def test_stale_revision_is_rejected(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake))
    update_rebuild_plan_status(lake, entry.plan_id, "approved", approver="lead")
    # A caller holding the pre-approval revision (0) is now stale.
    with pytest.raises(RebuildPlanConflict, match="revision"):
        update_rebuild_plan_status(
            lake, entry.plan_id, "dispatched", expected_revision=0
        )


def test_stale_status_is_rejected(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake))
    update_rebuild_plan_status(lake, entry.plan_id, "approved", approver="lead")
    with pytest.raises(RebuildPlanConflict, match="status"):
        update_rebuild_plan_status(
            lake, entry.plan_id, "dispatched", expected_status="draft"
        )


def test_same_status_update_is_rejected(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake))
    with pytest.raises(RebuildPlanCatalogError, match="already in status"):
        update_rebuild_plan_status(lake, entry.plan_id, "draft")


# --- Orchestrator handoff ---------------------------------------------------


def test_dispatch_export_has_stable_ids_and_dependencies(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake))

    dispatch = export_rebuild_plan_dispatch(lake, entry.plan_id)
    assert dispatch.schema_version == DISPATCH_SCHEMA_VERSION
    assert dispatch.actions

    by_artifact = {action["target_artifact_id"]: action for action in dispatch.actions}
    for action in dispatch.actions:
        assert action["action_id"].startswith("act-")
        assert action["target_artifact_id"]
        assert action["external_run_ref"].startswith("rebuild:")
        # dependency artifact ids resolve to concrete action ids within the plan
        for dep_artifact, dep_action in zip(
            action["depends_on_artifact_ids"], action["depends_on"], strict=True
        ):
            assert dep_action == by_artifact[dep_artifact]["action_id"]

    # at least one action carries a real dependency (source -> snapshot -> ...)
    assert any(action["depends_on"] for action in dispatch.actions)


def test_reexporting_an_unchanged_plan_is_identical(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake))

    first = export_rebuild_plan_dispatch(lake, entry.plan_id, orchestrator="airflow")
    second = export_rebuild_plan_dispatch(lake, entry.plan_id, orchestrator="airflow")
    assert first.as_dict() == second.as_dict()
    assert first.as_ndjson() == second.as_ndjson()
    assert len(first.as_ndjson().splitlines()) == len(first.actions)


def test_dry_run_export_does_not_change_status(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake))
    export_rebuild_plan_dispatch(lake, entry.plan_id, dry_run=True)
    reloaded, _ = get_rebuild_plan(lake, entry.plan_id)
    assert reloaded.status == "draft"
    assert reloaded.revision == 0


def test_dispatch_requires_approval_then_is_idempotent(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake))

    with pytest.raises(RebuildPlanCatalogError, match="must be approved"):
        export_rebuild_plan_dispatch(lake, entry.plan_id, dry_run=False)

    update_rebuild_plan_status(lake, entry.plan_id, "approved", approver="lead")
    first = export_rebuild_plan_dispatch(lake, entry.plan_id, dry_run=False)
    after = get_rebuild_plan(lake, entry.plan_id)[0]
    assert after.status == "dispatched"

    # re-dispatch is an idempotent no-op returning an identical payload
    second = export_rebuild_plan_dispatch(lake, entry.plan_id, dry_run=False)
    assert first.as_dict() == second.as_dict()
    assert get_rebuild_plan(lake, entry.plan_id)[0].revision == after.revision

    dispatch_events = rebuild_plan_events(lake, plan_id=entry.plan_id, event_type="dispatch-exported")
    assert len(dispatch_events) == 1


# --- Listing / filtering / pagination ---------------------------------------


def test_list_filters_and_paginates(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake, reason="reason-a"))
    other = record_rebuild_plan(lake, _plan(lake, reason="reason-b"))
    assert entry.plan_id != other.plan_id

    update_rebuild_plan_status(lake, other.plan_id, "approved", approver="lead")

    approved = rebuild_plans(lake, status="approved")
    assert [p.plan_id for p in approved.plans] == [other.plan_id]

    page = rebuild_plans(lake, page_size=1)
    assert len(page.plans) == 1
    assert page.next_cursor is not None
    page2 = rebuild_plans(lake, page_size=1, cursor=page.next_cursor)
    assert len(page2.plans) == 1
    assert page.plans[0].plan_id != page2.plans[0].plan_id

    root = entry.root_artifact_ids[0]
    by_root = rebuild_plans(lake, root_artifact_id=root)
    assert {p.plan_id for p in by_root.plans} == {entry.plan_id, other.plan_id}


# --- CLI --------------------------------------------------------------------


def test_cli_save_list_show_status_export_flow(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    lake_path = str(lake.uri)
    source = _source_uri(lake)

    save = runner.invoke(
        app,
        [
            "lineage", "save-rebuild-plan", source,
            "--lake", lake_path,
            "--kind", "source",
            "--reason", "cli calibration bug",
            "--no-refresh",
        ],
    )
    assert save.exit_code == 0, save.output
    entry = json.loads(save.output)
    plan_id = entry["plan_id"]
    assert entry["status"] == "draft"

    listing = runner.invoke(app, ["lineage", "list-rebuild-plans", "--lake", lake_path])
    assert listing.exit_code == 0, listing.output
    assert plan_id in listing.output

    show = runner.invoke(
        app, ["lineage", "show-rebuild-plan", plan_id, "--lake", lake_path, "--include-plan"]
    )
    assert show.exit_code == 0, show.output
    assert json.loads(show.output)["plan"]["actions"]

    approve = runner.invoke(
        app,
        [
            "lineage", "set-rebuild-status", plan_id, "approved",
            "--lake", lake_path, "--approver", "lead",
        ],
    )
    assert approve.exit_code == 0, approve.output
    assert json.loads(approve.output)["status"] == "approved"

    export = runner.invoke(
        app,
        ["lineage", "export-rebuild-plan", plan_id, "--lake", lake_path, "--orchestrator", "dagster"],
    )
    assert export.exit_code == 0, export.output
    payload = json.loads(export.output)
    assert payload["orchestrator"] == "dagster"
    assert all(action["action_id"].startswith("act-") for action in payload["actions"])

    dispatch = runner.invoke(
        app, ["lineage", "export-rebuild-plan", plan_id, "--lake", lake_path, "--dispatch"]
    )
    assert dispatch.exit_code == 0, dispatch.output

    events = runner.invoke(app, ["lineage", "rebuild-plan-events", "--lake", lake_path])
    assert events.exit_code == 0, events.output
    assert "dispatch-exported" in events.output


def test_cli_stale_status_update_exits_with_conflict(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    entry = record_rebuild_plan(lake, _plan(lake))
    lake_path = str(lake.uri)

    result = runner.invoke(
        app,
        [
            "lineage", "set-rebuild-status", entry.plan_id, "approved",
            "--lake", lake_path, "--approver", "lead",
            "--expected-revision", "5",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "conflict" in result.output.lower()


def test_events_table_is_registered_and_created(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    # both catalog tables exist on a freshly initialised lake
    assert lake.table(CATALOG_TABLE).count_rows() == 0
    assert lake.table(EVENTS_TABLE).count_rows() == 0
