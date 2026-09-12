"""Published-view lifecycle, pin protection, and readiness (backlog 0508).

Four surfaces, each pinned with its mechanism and degradation path (SKILLS.md:
assert the bound and the mechanism, not just the answer):

* ``retire_view``: protected-view refusal (pointer target / newest per repo)
  with ``force`` override and pointer re-point, BUG-04 postcondition, and
  crash-window convergence (header gone, files remaining).
* ``plan_view_retention`` / ``apply_view_retention``: age +
  retain-N-newest-per-repo candidates, report-only default inside
  ``lake maintain``, explicit apply opt-in.
* ``reconcile_orphan_view_files``: headerless file rows reclaimed past a grace
  window, in-flight (young) rows and live views untouched.
* ``view_retention_pin_details`` + ``view_readiness`` +
  ``view_pin_conformance``: view pins join maintain's tag-before-cleanup pin
  map (version pruning can no longer break a published view), readiness
  classifies already-broken views, and pinned opens over gated backends are a
  typed error or a working pin -- never a silent latest-read (0116).
"""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest
from test_lerobot_facade import _two_episode_lake

from lancedb_robotics.connections import LakeCapabilities, LakeConnectionSpec
from lancedb_robotics.lerobot_facade import (
    CanonicalVectorMapping,
    ProtectedViewError,
    StaleViewVersionError,
    ViewNotFoundError,
    apply_view_retention,
    get_view,
    open_published_facade,
    plan_view_retention,
    publish_view,
    reconcile_orphan_view_files,
    retire_view,
    view_pin_conformance,
    view_readiness,
    view_retention_pin_details,
)
from lancedb_robotics.lerobot_facade.view_retention import (
    AT_RISK,
    BACKEND_GATED,
    PIN_PROTECTED,
    PIN_PRUNED,
    READY,
    VIEW_PIN_CAPABILITY_GATED,
    VIEW_PIN_SUPPORTED,
    VIEW_PIN_UNAVAILABLE,
)
from lancedb_robotics.lerobot_facade.views import (
    VIEW_FILES_TABLE,
    VIEW_LATEST_TABLE,
    VIEWS_TABLE,
    _latest_pointer_view_id,
)
from lancedb_robotics.maintenance import _PIN_TAG_PREFIX as MAINT_PIN_TAG_PREFIX
from lancedb_robotics.maintenance import maintain_lake
from lancedb_robotics.schemas import LEROBOT_VIEW_FILES_SCHEMA

_MAPPING = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))


def _publish(lake, **overrides):
    kwargs = dict(
        repo_id="acme/pick-place-v1",
        fps=20,
        mapping=_MAPPING,
        name="facade_view",
        robot_type="test-arm",
        created_by="tests",
    )
    kwargs.update(overrides)
    return publish_view(lake, **kwargs)


def _rows(lake, table, columns, where=None):
    query = lake.table(table).search().select(columns)
    if where:
        query = query.where(where)
    return query.to_arrow().to_pylist()


def _view_ids(lake):
    return {row["view_id"] for row in _rows(lake, VIEWS_TABLE, ["view_id"])}


def _spec_lake(spec):
    class _SpecLake:
        connection_spec = spec
        uri = "mem://conformance"

    return _SpecLake()


def _remote_db_spec(**control_plane):
    return LakeConnectionSpec(
        kind="lancedb_remote_db",
        uri="db://robotics",
        display_uri="db://robotics",
        capabilities=LakeCapabilities(
            server_side_query=True, blob_fetch_remote=True, **control_plane
        ),
    )


def _managed_namespace_spec():
    return LakeConnectionSpec(
        kind="rest_namespace_lancedb",
        uri=None,
        display_uri="namespace://robotics",
        capabilities=LakeCapabilities(
            direct_object_io=True,
            namespace_resolution=True,
            namespace_managed_versioning=True,
            table_versioning=True,
        ),
    )


# --------------------------------------------------------------------------- #
# retire_view
# --------------------------------------------------------------------------- #


def test_retire_older_view_keeps_newest_resolvable(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    older = _publish(lake, fps=20)
    newest = _publish(lake, fps=30)

    report = retire_view(lake, older.view_id)

    assert report.status == "retired"
    assert report.header_rows_deleted >= 1
    assert report.file_rows_deleted >= 1
    assert report.pointer_rows_deleted == 0  # the pointer targeted the newest view
    assert report.pointer_action == "untouched"
    assert _view_ids(lake) == {newest.view_id}
    assert not _rows(lake, VIEW_FILES_TABLE, ["file_id"], f"view_id = '{older.view_id}'")
    resolved = get_view(lake, repo_id="acme/pick-place-v1")
    assert resolved["view_id"] == newest.view_id


def test_retire_protected_view_refused_then_forced_repoints(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    older = _publish(lake, fps=20)
    newest = _publish(lake, fps=30)

    with pytest.raises(ProtectedViewError):
        retire_view(lake, newest.view_id)
    assert newest.view_id in _view_ids(lake)

    report = retire_view(lake, newest.view_id, force=True)
    assert report.status == "retired"
    assert report.forced is True
    assert report.pointer_rows_deleted >= 1
    assert report.pointer_action == "repointed"
    assert _view_ids(lake) == {older.view_id}
    assert _latest_pointer_view_id(lake, "acme/pick-place-v1") == older.view_id
    assert get_view(lake, repo_id="acme/pick-place-v1")["view_id"] == older.view_id


def test_retire_only_view_with_force_removes_pointer(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)

    report = retire_view(lake, published.view_id, force=True)

    assert report.status == "retired"
    assert report.pointer_action == "removed"
    assert not _view_ids(lake)
    assert not _rows(lake, VIEW_LATEST_TABLE, ["repo_id"])
    with pytest.raises(ViewNotFoundError):
        get_view(lake, repo_id="acme/pick-place-v1")


def test_retire_dry_run_and_absent(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    older = _publish(lake, fps=20)
    _publish(lake, fps=30)

    dry = retire_view(lake, older.view_id, dry_run=True)
    assert dry.status == "dry-run"
    assert dry.file_rows_deleted >= 1
    assert older.view_id in _view_ids(lake)  # nothing written

    absent = retire_view(lake, "lrv-doesnotexist")
    assert absent.status == "absent"


def test_retire_converges_after_crash_window(tmp_path):
    # Simulate a crash after the header delete: invisible file rows remain.
    lake = _two_episode_lake(tmp_path / "robot.lance")
    older = _publish(lake, fps=20)
    _publish(lake, fps=30)
    lake.table(VIEWS_TABLE).delete(f"view_id = '{older.view_id}'")
    assert _rows(lake, VIEW_FILES_TABLE, ["file_id"], f"view_id = '{older.view_id}'")

    report = retire_view(lake, older.view_id)

    assert report.status == "retired"
    assert report.header_rows_deleted == 0
    assert report.file_rows_deleted >= 1
    assert not _rows(lake, VIEW_FILES_TABLE, ["file_id"], f"view_id = '{older.view_id}'")


# --------------------------------------------------------------------------- #
# Retention policy
# --------------------------------------------------------------------------- #


def test_plan_view_retention_respects_retain_and_age(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    a_old = _publish(lake, repo_id="acme/a", fps=20)
    a_mid = _publish(lake, repo_id="acme/a", fps=30)
    a_new = _publish(lake, repo_id="acme/a", fps=40)
    b_only = _publish(lake, repo_id="acme/b", fps=20)

    future = datetime.now(UTC) + timedelta(days=200)
    plan = plan_view_retention(
        lake, older_than=timedelta(days=90), retain_latest_per_repo=1, now=future
    )

    assert plan.status == "planned"
    assert plan.views_seen == 4
    assert plan.repos_seen == 2
    candidate_ids = {candidate.view_id for candidate in plan.candidates}
    # Newest per repo (a_new, b_only) are never candidates; a_new is also the
    # pointer target for acme/a.
    assert candidate_ids == {a_old.view_id, a_mid.view_id}
    assert a_new.view_id not in candidate_ids
    assert b_only.view_id not in candidate_ids
    # Plan is read-only.
    assert _view_ids(lake) == {a_old.view_id, a_mid.view_id, a_new.view_id, b_only.view_id}


def test_plan_view_retention_age_gate_keeps_young_views(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish(lake, repo_id="acme/a", fps=20)
    _publish(lake, repo_id="acme/a", fps=30)

    plan = plan_view_retention(lake, older_than=timedelta(days=90), retain_latest_per_repo=1)
    assert plan.status == "planned"
    assert not plan.candidates  # both views were just published


def test_retain_latest_per_repo_floor_is_one(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    with pytest.raises(ValueError):
        plan_view_retention(lake, retain_latest_per_repo=0)


def test_apply_view_retention_retires_candidates(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    a_old = _publish(lake, repo_id="acme/a", fps=20)
    a_new = _publish(lake, repo_id="acme/a", fps=30)

    future = datetime.now(UTC) + timedelta(days=200)
    report = apply_view_retention(
        lake,
        older_than=timedelta(days=90),
        retain_latest_per_repo=1,
        dry_run=False,
        now=future,
    )

    assert report.status == "applied"
    assert [item["view_id"] for item in report.applied] == [a_old.view_id]
    assert not report.apply_errors
    assert _view_ids(lake) == {a_new.view_id}
    assert get_view(lake, repo_id="acme/a")["view_id"] == a_new.view_id


def test_apply_view_retention_default_is_dry_run(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish(lake, repo_id="acme/a", fps=20)
    _publish(lake, repo_id="acme/a", fps=30)

    future = datetime.now(UTC) + timedelta(days=200)
    report = apply_view_retention(lake, retain_latest_per_repo=1, now=future)

    assert report.status == "planned"
    assert len(report.candidates) == 1
    assert not report.applied
    assert len(_view_ids(lake)) == 2  # nothing deleted


# --------------------------------------------------------------------------- #
# Orphan file-row reconciliation
# --------------------------------------------------------------------------- #


def _add_orphan_files(lake, view_id, created_at, count=2):
    rows = [
        {
            "file_id": f"{view_id}/meta/orphan-{index}.json",
            "view_id": view_id,
            "path": f"meta/orphan-{index}.json",
            "content": b"{}",
            "sha256": "0" * 64,
            "size_bytes": 2,
            "created_at": created_at,
        }
        for index in range(count)
    ]
    lake.table(VIEW_FILES_TABLE).add(
        pa.Table.from_pylist(rows, schema=LEROBOT_VIEW_FILES_SCHEMA)
    )


def test_reconcile_orphan_view_files_reclaims_old_keeps_young_and_live(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    live = _publish(lake)
    now = datetime.now(UTC)
    _add_orphan_files(lake, "lrv-orphan-old", now - timedelta(days=3))
    _add_orphan_files(lake, "lrv-orphan-young", now)

    report = reconcile_orphan_view_files(lake, grace=timedelta(hours=24), now=now)

    assert report.status == "reconciled"
    assert report.orphan_view_ids_reclaimed == 1
    assert report.file_rows_deleted == 2
    assert report.in_grace_view_ids == 1
    assert not _rows(lake, VIEW_FILES_TABLE, ["file_id"], "view_id = 'lrv-orphan-old'")
    assert _rows(lake, VIEW_FILES_TABLE, ["file_id"], "view_id = 'lrv-orphan-young'")
    # Live view untouched; a pinned open still works end to end.
    assert _rows(lake, VIEW_FILES_TABLE, ["file_id"], f"view_id = '{live.view_id}'")
    open_published_facade(lake, get_view(lake, view_id=live.view_id))


def test_reconcile_orphan_view_files_dry_run_deletes_nothing(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    now = datetime.now(UTC)
    _add_orphan_files(lake, "lrv-orphan-old", now - timedelta(days=3))

    report = reconcile_orphan_view_files(lake, now=now, dry_run=True)

    assert report.dry_run is True
    assert report.orphan_view_ids_reclaimed == 1  # planned, not applied
    assert _rows(lake, VIEW_FILES_TABLE, ["file_id"], "view_id = 'lrv-orphan-old'")


# --------------------------------------------------------------------------- #
# Pin protection in lake maintain
# --------------------------------------------------------------------------- #


def test_view_retention_pin_details_shape_and_retire_drops_pins(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)

    pins = view_retention_pin_details(lake)

    pinned = {entry["table"]: entry["version"] for entry in published.table_versions}
    for table, version in pinned.items():
        detail = pins[table][int(version)]
        assert published.view_id in detail["artifact_ids"]
        assert "lerobot-view" in detail["categories"]
        assert any(reason.startswith("lerobot-view:") for reason in detail["reasons"])

    retire_view(lake, published.view_id, force=True)
    assert view_retention_pin_details(lake) == {}


def test_maintain_protects_view_pinned_versions(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    # Advance a pinned table so the pinned version is no longer current.
    lake.table("runs").delete("run_id = 'no-such-row'")

    maintain_lake(lake, cleanup_older_than=timedelta(0))

    # Mechanism: the pinned runs version carries the managed pin tag.
    pinned_runs = next(
        int(entry["version"]) for entry in published.table_versions if entry["table"] == "runs"
    )
    tags = set(lake.table("runs").to_lance().tags.list())
    assert f"{MAINT_PIN_TAG_PREFIX}{pinned_runs}" in tags
    # Outcome: the pinned open still serves every table at its pinned version.
    facade = open_published_facade(lake, get_view(lake, view_id=published.view_id))
    assert len(facade) > 0


def test_cleanup_without_view_protection_breaks_pinned_open(tmp_path):
    # The negative arm proving the mechanism is load-bearing: with view-pin
    # protection off, version cleanup prunes the pinned version and the open
    # raises the typed StaleViewVersionError -- never a silent latest-read.
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    lake.table("runs").delete("run_id = 'no-such-row'")

    maintain_lake(
        lake,
        cleanup_older_than=timedelta(0),
        protect_lineage=False,
        refresh_lineage=False,
        protect_lerobot_views=False,
        lerobot_view_readiness=False,
    )

    with pytest.raises(StaleViewVersionError):
        open_published_facade(lake, get_view(lake, view_id=published.view_id))


def test_maintain_view_retention_is_report_only_by_default(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    old = _publish(lake, fps=20)
    new = _publish(lake, fps=30)
    # Make both views old enough for the age gate.
    report = maintain_lake(
        lake,
        tables=("lerobot_views",),
        cleanup_older_than=None,
        lerobot_view_retention_older_than=timedelta(0),
        lerobot_view_retain_latest_per_repo=1,
    )

    section = report.lerobot_view_retention
    assert section is not None and section.get("applied") is False
    policy = section["policy"]
    assert policy["status"] == "planned"
    assert policy["candidate_count"] == 1
    assert policy["candidates"][0]["view_id"] == old.view_id
    # Report-only: nothing was deleted.
    assert _view_ids(lake) == {old.view_id, new.view_id}

    applied = maintain_lake(
        lake,
        tables=("lerobot_views",),
        cleanup_older_than=None,
        lerobot_view_retention_older_than=timedelta(0),
        lerobot_view_retain_latest_per_repo=1,
        lerobot_view_retention_apply=True,
    )
    applied_policy = applied.lerobot_view_retention["policy"]
    assert applied_policy["status"] == "applied"
    assert [item["view_id"] for item in applied_policy["applied"]] == [old.view_id]
    assert _view_ids(lake) == {new.view_id}


def test_maintain_reports_view_readiness_section(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish(lake)

    report = maintain_lake(lake, tables=("lerobot_views",), cleanup_older_than=None)

    section = report.lerobot_view_readiness
    assert section is not None
    assert section["schema_version"].endswith("lerobot-view-readiness/v1")
    assert section["views_checked"] == 1


# --------------------------------------------------------------------------- #
# Readiness + backend conformance (0116 invariant)
# --------------------------------------------------------------------------- #


def test_pin_tag_prefix_matches_maintenance():
    # Readiness recognizes protection via the managed pin tag maintenance
    # writes; if the prefixes drift, protection silently stops being detected.
    from lancedb_robotics.lerobot_facade.view_retention import _PIN_TAG_PREFIX

    assert _PIN_TAG_PREFIX == MAINT_PIN_TAG_PREFIX


def test_view_readiness_ready_after_maintain(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    lake.table("runs").delete("run_id = 'no-such-row'")
    maintain_lake(lake, cleanup_older_than=timedelta(0))

    report = view_readiness(lake)

    assert report.status == READY
    assert report.ready is True
    assert report.views_checked == 1
    assert report.views_at_risk == 0
    pinned_runs = next(
        int(entry["version"]) for entry in published.table_versions if entry["table"] == "runs"
    )
    runs_pin = next(
        pin for pin in report.pins if pin.table == "runs" and pin.version == pinned_runs
    )
    assert runs_pin.status == PIN_PROTECTED
    assert runs_pin.tagged is True


def test_view_readiness_reports_pruned_view_at_risk(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    lake.table("runs").delete("run_id = 'no-such-row'")
    maintain_lake(
        lake,
        cleanup_older_than=timedelta(0),
        protect_lineage=False,
        refresh_lineage=False,
        protect_lerobot_views=False,
        lerobot_view_readiness=False,
    )

    report = view_readiness(lake)

    assert report.status == AT_RISK
    assert report.ready is False
    assert report.views_at_risk == 1
    assert any(pin.status == PIN_PRUNED for pin in report.pins)
    assert report.at_risk_views[0]["view_id"] == published.view_id
    assert any("re-publish" in action for action in report.suggested_actions)


def test_view_readiness_backend_gated_never_reports_ready(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish(lake)
    # A db:// backend advertising versioning but not direct object IO: pins
    # cannot be verified here, so the verdict must not fall through to ready.
    lake.connection_spec = _remote_db_spec(table_versioning=True)

    report = view_readiness(lake)

    assert report.status == BACKEND_GATED
    assert report.ready is False
    assert all(pin.status == "backend-gated" for pin in report.pins)
    assert report.suggested_actions


def test_view_pin_conformance_classification():
    from lancedb_robotics.connections import resolve_lake_connection

    local = view_pin_conformance(_spec_lake(resolve_lake_connection("/tmp/l.lance")))
    assert local.status == VIEW_PIN_SUPPORTED

    gated = view_pin_conformance(_spec_lake(_remote_db_spec()))
    assert gated.status == VIEW_PIN_CAPABILITY_GATED
    assert "object_store_lancedb_oss" in gated.fallbacks
    assert gated.suggested_action

    advertised = view_pin_conformance(_spec_lake(_remote_db_spec(table_versioning=True)))
    assert advertised.status == VIEW_PIN_SUPPORTED

    managed = view_pin_conformance(_spec_lake(_managed_namespace_spec()))
    assert managed.status == VIEW_PIN_UNAVAILABLE
    assert managed.namespace_managed_versioning is True


def test_pinned_open_checkout_failure_is_typed_never_silent(tmp_path, monkeypatch):
    # 0116 invariant on the real open path: when the backend cannot check out a
    # pinned version, the open raises the typed StaleViewVersionError with a
    # re-publish remediation -- it never silently serves the live version.
    from lancedb_robotics.lake import Lake

    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    view = get_view(lake, view_id=published.view_id)

    original = Lake.table

    def _no_checkout_table(self, name):
        handle = original(self, name)

        def _raise(_version):
            raise RuntimeError("checkout unsupported on this backend")

        monkeypatch.setattr(handle, "checkout", _raise, raising=False)
        return handle

    monkeypatch.setattr(Lake, "table", _no_checkout_table)
    with pytest.raises(StaleViewVersionError) as excinfo:
        open_published_facade(lake, view)
    assert "Re-publish the view" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Guardrail pins (scale-review findings 1, 2, 4)
# --------------------------------------------------------------------------- #


def _append_header_copy(lake, view_id, created_at):
    """Hand-append a physical duplicate header copy with a different stamp."""
    from lancedb_robotics.schemas import LEROBOT_VIEWS_SCHEMA

    rows = (
        lake.table(VIEWS_TABLE)
        .search()
        .where(f"view_id = '{view_id}'")
        .limit(1)
        .to_arrow()
        .to_pylist()
    )
    assert rows
    copy = dict(rows[0])
    copy["created_at"] = created_at
    lake.table(VIEWS_TABLE).add(pa.Table.from_pylist([copy], schema=LEROBOT_VIEWS_SCHEMA))


def test_duplicate_header_copy_never_demotes_a_retained_view(tmp_path, monkeypatch):
    # Scale-review finding 1 (the shape, per SKILLS.md §3): concurrent identical
    # publishers stamp their own created_at, so duplicate copies of one view are
    # NOT adjacent under the (created_at desc) ordering. A duplicate of view B
    # older than view A must never rank B again past the retain window.
    lake = _two_episode_lake(tmp_path / "robot.lance")
    a_oldest = _publish(lake, fps=20)
    b_middle = _publish(lake, fps=30)
    _publish(lake, fps=40)  # newest; pointer target
    # Duplicate copy of B stamped older than everything else.
    _append_header_copy(lake, b_middle.view_id, datetime.now(UTC) - timedelta(days=400))

    future = datetime.now(UTC) + timedelta(days=200)
    plan = plan_view_retention(
        lake, older_than=timedelta(days=90), retain_latest_per_repo=2, now=future
    )
    candidate_ids = {candidate.view_id for candidate in plan.candidates}
    assert candidate_ids == {a_oldest.view_id}
    assert b_middle.view_id not in candidate_ids  # rank 2: inside the retain window

    # Same guarantee through the unordered fallback path.
    from lancedb_robotics.lerobot_facade import view_retention as vr

    monkeypatch.setattr(vr, "_iter_headers_repo_ordered", lambda _lake: None)
    fallback = plan_view_retention(
        lake, older_than=timedelta(days=90), retain_latest_per_repo=2, now=future
    )
    assert {candidate.view_id for candidate in fallback.candidates} == {a_oldest.view_id}


def test_pin_details_bound_raises_loudly_and_fails_maintenance(tmp_path):
    # Silently skipping pins and then pruning is the data-loss shape 0508
    # prevents; past the bound the pin map must raise, and maintain must fail
    # BEFORE any cleanup (scale-review finding 2 / BUG-11 pinning).
    from lancedb_robotics.lerobot_facade import ViewRetirementError

    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish(lake, fps=20)
    _publish(lake, fps=30)

    with pytest.raises(ViewRetirementError, match="pin map"):
        view_retention_pin_details(lake, max_views=1)

    from lancedb_robotics.maintenance import MaintenanceError

    with pytest.raises(MaintenanceError, match="published-view version pins"):
        maintain_lake(lake, cleanup_older_than=timedelta(0), lerobot_view_pin_max_views=1)
    # Nothing was pruned: both views still open pinned.
    for row in _rows(lake, VIEWS_TABLE, ["view_id"]):
        open_published_facade(lake, get_view(lake, view_id=row["view_id"]))


def test_protection_check_fails_closed_without_ordered_scan(tmp_path, monkeypatch):
    # "Never silently guess": when the newest-header read cannot be ordered,
    # retire refuses non-forced retirement instead of assuming safety.
    from lancedb_robotics.lerobot_facade import view_retention as vr

    lake = _two_episode_lake(tmp_path / "robot.lance")
    older = _publish(lake, fps=20)
    _publish(lake, fps=30)
    monkeypatch.setattr(vr, "_newest_header_for_repo", lambda *args, **kwargs: None)

    with pytest.raises(ProtectedViewError, match="cannot verify"):
        retire_view(lake, older.view_id)
    assert retire_view(lake, older.view_id, force=True).status == "retired"


def test_plan_candidate_bound_and_pointer_target_truncation(tmp_path, monkeypatch):
    from lancedb_robotics.lerobot_facade import view_retention as vr

    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish(lake, fps=20)
    _publish(lake, fps=30)
    _publish(lake, fps=40)

    future = datetime.now(UTC) + timedelta(days=200)
    plan = plan_view_retention(
        lake, retain_latest_per_repo=1, max_candidates=1, now=future
    )
    assert len(plan.candidates) == 1
    assert plan.candidates_remaining == 1

    monkeypatch.setattr(vr, "_MAX_POINTER_TARGETS", 0)
    truncated = plan_view_retention(lake, retain_latest_per_repo=1, now=future)
    assert truncated.pointer_targets_truncated is True


def test_plan_unordered_over_bound_reports_skipped(tmp_path, monkeypatch):
    from lancedb_robotics.lerobot_facade import view_retention as vr

    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish(lake, fps=20)
    _publish(lake, fps=30)
    monkeypatch.setattr(vr, "_iter_headers_repo_ordered", lambda _lake: None)
    monkeypatch.setattr(vr, "_MAX_UNORDERED_SCAN_ROWS", 1)

    plan = plan_view_retention(lake)

    assert plan.status == "skipped-unordered-over-bound"
    assert not plan.candidates


def test_orphan_reclaim_bound_and_unordered_guard(tmp_path, monkeypatch):
    from lancedb_robotics.lerobot_facade import view_retention as vr

    lake = _two_episode_lake(tmp_path / "robot.lance")
    now = datetime.now(UTC)
    _add_orphan_files(lake, "lrv-orphan-a", now - timedelta(days=3))
    _add_orphan_files(lake, "lrv-orphan-b", now - timedelta(days=3))

    bounded = reconcile_orphan_view_files(lake, max_reclaims=1, now=now)
    assert bounded.orphan_view_ids_reclaimed == 1
    assert bounded.reclaims_remaining == 1
    # Re-run converges on the remainder.
    rest = reconcile_orphan_view_files(lake, now=now)
    assert rest.orphan_view_ids_reclaimed == 1
    assert not _rows(lake, VIEW_FILES_TABLE, ["file_id"], "view_id = 'lrv-orphan-b'")

    _add_orphan_files(lake, "lrv-orphan-c", now - timedelta(days=3))
    monkeypatch.setattr(vr, "_iter_file_groups_ordered", lambda _lake: None)
    monkeypatch.setattr(vr, "_MAX_UNORDERED_SCAN_ROWS", 1)
    skipped = reconcile_orphan_view_files(lake, now=now)
    assert skipped.status == "skipped-unordered-over-bound"
    assert _rows(lake, VIEW_FILES_TABLE, ["file_id"], "view_id = 'lrv-orphan-c'")


def test_orphan_delete_predicate_carries_grace_cutoff(tmp_path, monkeypatch):
    # Mechanism pin (BUG-06 round-2 pattern): the delete predicate itself must
    # be bounded by the grace cutoff so a racing re-publish's fresh rows are
    # structurally outside it -- not merely skipped by group-level checks.
    from lancedb_robotics.lake import Lake

    lake = _two_episode_lake(tmp_path / "robot.lance")
    now = datetime.now(UTC)
    _add_orphan_files(lake, "lrv-orphan-old", now - timedelta(days=3))

    predicates: list[str] = []
    original = Lake.table

    def _recording_table(self, name):
        handle = original(self, name)
        if name == VIEW_FILES_TABLE:
            inner_delete = handle.delete

            def _record(predicate):
                predicates.append(predicate)
                return inner_delete(predicate)

            monkeypatch.setattr(handle, "delete", _record, raising=False)
        return handle

    monkeypatch.setattr(Lake, "table", _recording_table)
    report = reconcile_orphan_view_files(lake, now=now)

    assert report.orphan_view_ids_reclaimed == 1
    assert report.file_rows_surviving == 0
    assert predicates, "reconcile deleted nothing"
    assert all("created_at <= " in predicate for predicate in predicates)
    assert all("created_at IS NULL" in predicate for predicate in predicates)
