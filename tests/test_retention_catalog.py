"""Durable retention-policy catalog and governance-hook tests (backlog 0111)."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import (
    lineage_retention_pin_details,
    merge_retention_pin_details,
    retention_pin_rows,
    snapshot_retention_pin_details,
)
from lancedb_robotics.retention_catalog import (
    CATALOG_SCHEMA_VERSION,
    CATALOG_TABLE,
    GOVERNANCE_SCHEMA_VERSION,
    CollectingGovernanceSink,
    RetentionPolicyConflict,
    RetentionPolicyError,
    RetentionPolicyTooLarge,
    apply_retention_policy,
    build_retention_policy,
    export_retention_policy_state,
    get_retention_policy,
    project_retention_state,
    record_retention_policy,
    release_retention_policy,
    resolve_retention_holds,
    retention_expiration_notices,
    retention_policies,
    retention_policy_events,
    update_retention_policy_status,
)
from lancedb_robotics.scenarios import create_scenario_windows

runner = CliRunner()


def _lake(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    scenarios = lake.table("scenarios").to_arrow().to_pylist()
    create_snapshot(
        lake,
        name="demo-v1",
        tag="training-demo",
        scenario_ids=[row["scenario_id"] for row in scenarios],
    )
    lake.lineage.refresh_graph()
    return lake


def _policy(**overrides):
    base = dict(
        name="observations-audit",
        kinds=("table-version",),
        tables=("observations",),
        audit_hold=True,
        owner="data-governance",
        reason_template="regulatory audit retention",
    )
    base.update(overrides)
    return build_retention_policy(**base)


def _observations_artifact_id(lake):
    for row in lake.table("lineage_artifacts").to_arrow().to_pylist():
        if row.get("kind") == "table-version" and row.get("table_name") == "observations":
            return row["artifact_id"]
    raise AssertionError("no observations table-version artifact")


# --- Record / reload / idempotency ------------------------------------------


def test_record_reload_and_digest_stable(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    entry = record_retention_policy(lake, _policy(), actor="ops")
    assert entry.policy_id == entry.policy_digest
    assert entry.status == "draft"
    assert entry.revision == 0
    assert entry.catalog_schema_version == CATALOG_SCHEMA_VERSION
    assert entry.audit_hold is True
    assert "kinds=[table-version]" in entry.scope_summary

    reloaded, definition = get_retention_policy(lake, entry.policy_id)
    assert reloaded.policy_id == entry.policy_id
    assert definition["scope"]["tables"] == ["observations"]
    assert definition["rules"]["audit_hold"] is True

    again = record_retention_policy(lake, _policy(), actor="someone-else")
    assert again.policy_id == entry.policy_id
    assert again.created_at == entry.created_at
    assert lake.table(CATALOG_TABLE).count_rows() == 1


def test_content_change_yields_new_policy_id(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    a = record_retention_policy(lake, _policy(), actor="ops")
    b = record_retention_policy(lake, _policy(version="2"), actor="ops")
    assert a.policy_id != b.policy_id
    assert lake.table(CATALOG_TABLE).count_rows() == 2


def test_record_active_requires_approver(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    with pytest.raises(RetentionPolicyError, match="requires approver"):
        record_retention_policy(lake, _policy(), status="active")


# --- Validation -------------------------------------------------------------


def test_empty_scope_rejected():
    with pytest.raises(RetentionPolicyError, match="at least one selector"):
        build_and_check(name="x", audit_hold=True)


def test_ruleless_policy_rejected():
    with pytest.raises(RetentionPolicyError, match="at least one hold"):
        build_and_check(name="x", kinds=("table-version",))


def test_negative_retain_days_rejected():
    with pytest.raises(RetentionPolicyError, match="positive integer"):
        build_and_check(name="x", kinds=("table-version",), retain_for_days=0)


def build_and_check(**kwargs):
    # Validation happens on record; canonicalize via a throwaway in-memory check.
    from lancedb_robotics.retention_catalog import _canonicalize_policy

    _canonicalize_policy(build_retention_policy(**kwargs))


# --- Lifecycle + optimistic concurrency -------------------------------------


def test_lifecycle_transitions_and_conflicts(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    entry = record_retention_policy(lake, _policy(), actor="ops")

    with pytest.raises(RetentionPolicyError, match="requires approver"):
        update_retention_policy_status(lake, entry.policy_id, "active")

    active = update_retention_policy_status(
        lake, entry.policy_id, "active", approver="dpo", expected_revision=0
    )
    assert active.status == "active"
    assert active.revision == 1
    assert active.activated_at is not None

    # Stale revision guard.
    with pytest.raises(RetentionPolicyConflict):
        update_retention_policy_status(lake, entry.policy_id, "suspended", expected_revision=0)

    # Illegal transition.
    with pytest.raises(RetentionPolicyError, match="illegal"):
        update_retention_policy_status(lake, entry.policy_id, "draft")

    suspended = update_retention_policy_status(lake, entry.policy_id, "suspended")
    assert suspended.status == "suspended" and suspended.revision == 2
    archived = update_retention_policy_status(lake, entry.policy_id, "archived")
    assert archived.status == "archived"
    with pytest.raises(RetentionPolicyError, match="terminal"):
        update_retention_policy_status(lake, entry.policy_id, "active", approver="dpo")


# --- List / filter / page ---------------------------------------------------


def test_list_filter_and_page(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    record_retention_policy(lake, _policy(name="a"), actor="ops")
    record_retention_policy(lake, _policy(name="b"), actor="ops")
    third = record_retention_policy(lake, _policy(name="c"), actor="ops")
    update_retention_policy_status(lake, third.policy_id, "active", approver="dpo")

    page = retention_policies(lake, page_size=2)
    assert len(page.policies) == 2
    assert page.next_cursor is not None
    page2 = retention_policies(lake, page_size=2, cursor=page.next_cursor)
    assert len(page2.policies) == 1 and page2.next_cursor is None

    active = retention_policies(lake, status="active")
    assert [p.name for p in active.policies] == ["c"]
    named = retention_policies(lake, name="b")
    assert [p.name for p in named.policies] == ["b"]


# --- Apply / dry-run / same-shape-as-maintenance ----------------------------


def test_apply_dry_run_writes_nothing(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    entry = record_retention_policy(lake, _policy(), actor="ops", approver="dpo", status="active")
    preview = apply_retention_policy(lake, entry.policy_id, dry_run=True)
    assert preview.dry_run is True
    assert preview.matched_count >= 1
    assert preview.applied_count == preview.matched_count
    # No hold materialized.
    resolution = resolve_retention_holds(lake)
    assert resolution.policy_hold_count == 0


def test_apply_requires_active(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    entry = record_retention_policy(lake, _policy(), actor="ops")  # draft
    with pytest.raises(RetentionPolicyError, match="must be active"):
        apply_retention_policy(lake, entry.policy_id, dry_run=False)


def test_apply_materializes_holds_and_matches_maintenance_shape(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    entry = record_retention_policy(lake, _policy(), actor="ops", approver="dpo", status="active")
    result = apply_retention_policy(lake, entry.policy_id, dry_run=False)
    assert result.applied_count >= 1

    obs_id = _observations_artifact_id(lake)
    assert obs_id in result.applied_artifact_ids

    # AC1: matching artifacts now carry a 0067 hold with no manual edit.
    audit = lake.lineage.audit(refresh=False)
    held_ids = {h["artifact_ids"][0] for h in audit.retention_holds if h.get("artifact_ids")}
    assert obs_id in held_ids

    # AC2: inherited holds resolve to the SAME pin shape maintenance consumes.
    resolution = resolve_retention_holds(lake)
    expected = retention_pin_rows(
        merge_retention_pin_details(
            lineage_retention_pin_details(lake),
            snapshot_retention_pin_details(lake),
        )
    )
    assert resolution.pins == expected
    obs_pins = [p for p in resolution.pins if p["table"] == "observations"]
    assert obs_pins and any("retention-hold" in p["categories"] for p in obs_pins)
    assert resolution.policy_hold_count >= 1

    # Re-apply is idempotent: no new conflicts, still applied.
    again = apply_retention_policy(lake, entry.policy_id, dry_run=False)
    assert again.conflict_count == 0
    assert again.applied_count == result.applied_count


def test_retain_for_days_is_relative_to_artifact(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    policy = _policy(audit_hold=False, retain_for_days=30, reason_template="30-day window")
    entry = record_retention_policy(lake, policy, actor="ops", approver="dpo", status="active")
    result = apply_retention_policy(lake, entry.policy_id, dry_run=False)
    assert result.applied_count >= 1
    hold = result.holds[0]
    assert hold["retain_until"] is not None
    assert hold["audit_hold"] is False


# --- Artifact-local override + conflicts ------------------------------------


def test_artifact_local_hold_overrides_policy(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    obs_id = _observations_artifact_id(lake)
    # A manual (artifact-local) hold set the 0067 way.
    lake.lineage.retain(obs_id, legal_hold=True, reason="manual legal hold", refresh=False)

    entry = record_retention_policy(lake, _policy(), actor="ops", approver="dpo", status="active")
    result = apply_retention_policy(lake, entry.policy_id, dry_run=False)
    assert obs_id not in result.applied_artifact_ids
    conflict_ids = {c["artifact_id"] for c in result.conflicts}
    assert obs_id in conflict_ids
    conflict = next(c for c in result.conflicts if c["artifact_id"] == obs_id)
    assert conflict["existing_source"] == "artifact-local"

    # The manual hold is untouched (no policy marker in its reason).
    resolution = resolve_retention_holds(lake)
    obs_hold = next(h for h in resolution.holds if h["artifact_id"] == obs_id)
    assert obs_hold["source"] == "artifact-local"
    assert obs_hold["policy_id"] is None
    # Deterministic shadowing diagnostic surfaces the active policy.
    assert any(c["artifact_id"] == obs_id for c in resolution.conflicts)


def test_release_only_clears_policy_holds(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    # Manual hold on runs; policy hold on observations.
    runs_id = next(
        r["artifact_id"]
        for r in lake.table("lineage_artifacts").to_arrow().to_pylist()
        if r.get("kind") == "table-version" and r.get("table_name") == "runs"
    )
    lake.lineage.retain(runs_id, audit_hold=True, reason="manual", refresh=False)

    entry = record_retention_policy(lake, _policy(), actor="ops", approver="dpo", status="active")
    apply_retention_policy(lake, entry.policy_id, dry_run=False)

    preview = release_retention_policy(lake, entry.policy_id, dry_run=True)
    assert preview.matched_count >= 1 and preview.applied_count == 0

    released = release_retention_policy(lake, entry.policy_id, dry_run=False)
    assert released.applied_count >= 1

    resolution = resolve_retention_holds(lake)
    held_ids = {h["artifact_id"] for h in resolution.holds}
    assert runs_id in held_ids  # manual hold survives
    assert _observations_artifact_id(lake) not in held_ids  # policy hold released


# --- Events -----------------------------------------------------------------


def test_events_append_safe_and_survive_archive(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    entry = record_retention_policy(lake, _policy(), actor="ops")
    update_retention_policy_status(lake, entry.policy_id, "active", approver="dpo")
    apply_retention_policy(lake, entry.policy_id, dry_run=False)
    release_retention_policy(lake, entry.policy_id, dry_run=False)
    update_retention_policy_status(lake, entry.policy_id, "archived")

    events = retention_policy_events(lake, policy_id=entry.policy_id)
    types = [e["event_type"] for e in events]
    assert types[0] == "recorded"
    assert "status-changed" in types
    assert "applied" in types
    assert "released" in types
    # Archival did not delete the history.
    applied = retention_policy_events(lake, policy_id=entry.policy_id, event_type="applied")
    assert applied and applied[0]["artifact_count"] >= 1


# --- Governance projection --------------------------------------------------


def test_governance_projection_is_deterministic_and_secret_free(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    entry = record_retention_policy(lake, _policy(), actor="ops", approver="dpo", status="active")
    apply_retention_policy(lake, entry.policy_id, dry_run=False)

    projection = export_retention_policy_state(lake)
    assert projection.schema_version == GOVERNANCE_SCHEMA_VERSION
    assert len(projection.policies) == 1
    assert len(projection.holds) >= 1
    # Deterministic given lake state.
    assert projection.as_dict() == export_retention_policy_state(lake).as_dict()
    # NDJSON parses line-by-line.
    lines = [json.loads(line) for line in projection.as_ndjson().splitlines()]
    assert {row["record"] for row in lines} == {"policy", "hold"}

    sink = CollectingGovernanceSink()
    receipt = project_retention_state(lake, sink)
    assert receipt["projected_policies"] == 1
    assert len(sink.projections) == 1
    assert sink.projections[0].lake_uri == lake.uri


# --- Guardrails / expiration -------------------------------------------------


def test_apply_guardrail_too_large(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    # Scope every table-version artifact, then bound to 0.
    policy = _policy(name="all-versions", tables=(), audit_hold=True)
    entry = record_retention_policy(lake, policy, actor="ops", approver="dpo", status="active")
    with pytest.raises(RetentionPolicyTooLarge):
        apply_retention_policy(lake, entry.policy_id, dry_run=False, max_artifacts=0)


def test_expiration_notices(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    past = datetime.now(UTC) - timedelta(days=1)
    policy = _policy(name="short", audit_hold=False, retain_until=past.isoformat())
    entry = record_retention_policy(lake, policy, actor="ops", approver="dpo", status="active")
    apply_retention_policy(lake, entry.policy_id, dry_run=False)

    notices = retention_expiration_notices(lake, notify=True)
    assert notices and all(n["expired"] for n in notices)
    assert any(n["policy_id"] == entry.policy_id for n in notices)
    notified = retention_policy_events(lake, policy_id=entry.policy_id, event_type="expiration-notified")
    assert notified and notified[0]["artifact_count"] >= 1


def test_indefinite_holds_never_expire(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    entry = record_retention_policy(lake, _policy(), actor="ops", approver="dpo", status="active")
    apply_retention_policy(lake, entry.policy_id, dry_run=False)
    # audit_hold policy -> indefinite -> not an expiration candidate.
    assert retention_expiration_notices(lake, within=timedelta(days=3650)) == []


# --- Backward compatibility --------------------------------------------------


def test_existing_retain_metadata_behavior_unchanged(tmp_path, fixtures_dir):
    lake = _lake(tmp_path, fixtures_dir)
    obs_id = _observations_artifact_id(lake)
    hold = lake.lineage.retain(obs_id, audit_hold=True, reason="manual", refresh=False)
    assert hold.active is True
    resolution = resolve_retention_holds(lake)
    obs_hold = next(h for h in resolution.holds if h["artifact_id"] == obs_id)
    assert obs_hold["source"] == "artifact-local"
    cleared = lake.lineage.clear_retention(obs_id, refresh=False)
    assert cleared.active is False
    assert resolve_retention_holds(lake).hold_count == 0


# --- CLI smoke --------------------------------------------------------------


def test_cli_save_apply_and_export(tmp_path, fixtures_dir):
    _lake(tmp_path, fixtures_dir)
    lake_path = str(tmp_path / "robot.lance")

    saved = runner.invoke(
        app,
        [
            "lineage", "save-retention-policy", "--lake", lake_path,
            "--name", "cli-audit", "--kind", "table-version", "--table", "observations",
            "--audit-hold", "--owner", "gov", "--reason", "cli audit",
            "--status", "active", "--approver", "dpo",
        ],
    )
    assert saved.exit_code == 0, saved.output
    policy_id = json.loads(saved.output)["policy_id"]

    listed = runner.invoke(app, ["lineage", "list-retention-policies", "--lake", lake_path, "--status", "active"])
    assert listed.exit_code == 0
    assert json.loads(listed.output)["count"] == 1

    applied = runner.invoke(app, ["lineage", "apply-retention-policy", policy_id, "--lake", lake_path, "--apply"])
    assert applied.exit_code == 0, applied.output
    assert json.loads(applied.output)["applied_count"] >= 1

    resolved = runner.invoke(app, ["lineage", "resolve-retention-holds", "--lake", lake_path])
    assert resolved.exit_code == 0
    assert json.loads(resolved.output)["policy_hold_count"] >= 1

    exported = runner.invoke(app, ["lineage", "export-retention-state", "--lake", lake_path, "--ndjson"])
    assert exported.exit_code == 0
    records = [json.loads(line) for line in exported.output.splitlines() if line.strip()]
    assert any(r["record"] == "policy" for r in records)

    events = runner.invoke(app, ["lineage", "retention-policy-events", "--lake", lake_path, "--policy-id", policy_id])
    assert events.exit_code == 0
    assert any(e["event_type"] == "applied" for e in json.loads(events.output))
