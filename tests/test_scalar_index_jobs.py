"""Unit tests for the durable scalar predicate index job lifecycle (backlog 0136)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import lancedb
import pytest

from lancedb_robotics.connections import LakeCapabilities, LakeConnectionSpec
from lancedb_robotics.indexing import ScalarIndexResult
from lancedb_robotics.scalar_index_jobs import (
    MODE_ASYNCHRONOUS,
    MODE_PERMISSION_DENIED,
    MODE_SYNCHRONOUS,
    MODE_UNSUPPORTED,
    SCALAR_INDEX_JOB_TABLE,
    InMemoryScalarIndexJobStore,
    LanceTableScalarIndexJobStore,
    ScalarIndexJob,
    ScalarIndexJobCoordinator,
    ScalarIndexJobError,
    ScalarIndexJobRequest,
    build_scalar_index_job,
    probe_scalar_index_capability,
    request_scalar_index,
)


class FakeLake:
    """Minimal lake stand-in for capability probing and store resolution."""

    def __init__(
        self,
        *,
        connection_spec=None,
        db=None,
        submit=None,
        status=None,
        permission_denied=False,
        store=None,
    ) -> None:
        self.connection_spec = connection_spec
        self._db = db
        if submit is not None:
            self.scalar_index_job_submit = submit
        if status is not None:
            self.scalar_index_job_status = status
        if permission_denied:
            self.scalar_index_permission_denied = True
        if store is not None:
            self.scalar_index_job_store = store

    def table(self, _name):  # no version pin in unit tests
        raise LookupError("no table")


def _remote_spec(**caps) -> LakeConnectionSpec:
    return LakeConnectionSpec(
        kind="lancedb_remote_db",
        uri="db://robotics",
        display_uri="db://robotics",
        capabilities=LakeCapabilities(server_side_query=True, **caps),
    )


def _request(column="alignment_id", version=3) -> ScalarIndexJobRequest:
    return ScalarIndexJobRequest(
        table="aligned_ticks", column=column, index_type="BTREE", table_version=version
    )


def _fixed_now(moment=None):
    moment = moment or datetime(2026, 7, 13, tzinfo=UTC)
    return lambda: moment


# --------------------------------------------------------------------------- #
# Capability probe: the four modes                                             #
# --------------------------------------------------------------------------- #


def test_probe_reports_synchronous_for_bare_lake():
    assert probe_scalar_index_capability(FakeLake()).mode == MODE_SYNCHRONOUS


def test_probe_reports_synchronous_when_index_management_advertised():
    lake = FakeLake(connection_spec=_remote_spec(index_management=True))
    cap = probe_scalar_index_capability(lake)
    assert cap.mode == MODE_SYNCHRONOUS
    assert cap.buildable


def test_probe_reports_unsupported_without_index_management():
    lake = FakeLake(connection_spec=_remote_spec(index_management=False))
    cap = probe_scalar_index_capability(lake)
    assert cap.mode == MODE_UNSUPPORTED
    assert not cap.buildable
    assert cap.reason and "index_management" in cap.reason
    assert cap.fallbacks


def test_probe_reports_asynchronous_with_submit_hook():
    lake = FakeLake(
        connection_spec=_remote_spec(index_management=True),
        submit=lambda request: {"status": "active"},
    )
    cap = probe_scalar_index_capability(lake)
    assert cap.mode == MODE_ASYNCHRONOUS
    assert cap.is_async and cap.buildable


def test_probe_reports_permission_denied_flag():
    lake = FakeLake(connection_spec=_remote_spec(index_management=True), permission_denied=True)
    cap = probe_scalar_index_capability(lake)
    assert cap.mode == MODE_PERMISSION_DENIED
    assert not cap.buildable
    assert "authorized" in cap.reason


# --------------------------------------------------------------------------- #
# Idempotent request reuse (synchronous inline build)                          #
# --------------------------------------------------------------------------- #


def test_request_builds_once_and_reuses_completed_job():
    store = InMemoryScalarIndexJobStore()
    calls: list[str] = []

    def build_fn(req: ScalarIndexJobRequest) -> ScalarIndexResult:
        calls.append(req.column)
        return ScalarIndexResult(table=req.table, column=req.column, status="built", num_rows=42)

    coordinator = ScalarIndexJobCoordinator(store, build_fn=build_fn, now_fn=_fixed_now())
    first = coordinator.request(_request(), build_mode=MODE_SYNCHRONOUS, backend_kind="local_path")
    assert first.created and first.job.status == "complete"
    assert first.job.num_rows == 42
    assert first.index_result.status == "built"

    second = coordinator.request(_request(), build_mode=MODE_SYNCHRONOUS, backend_kind="local_path")
    assert second.reused and not second.created
    assert second.job.job_id == first.job.job_id
    assert len(calls) == 1  # the completed job is reused, not rebuilt
    assert second.job.request_count == 2


def test_different_version_is_a_distinct_job():
    store = InMemoryScalarIndexJobStore()
    coordinator = ScalarIndexJobCoordinator(
        store,
        build_fn=lambda req: ScalarIndexResult(table=req.table, column=req.column, status="built"),
        now_fn=_fixed_now(),
    )
    a = coordinator.request(_request(version=3), build_mode=MODE_SYNCHRONOUS, backend_kind="local")
    b = coordinator.request(_request(version=4), build_mode=MODE_SYNCHRONOUS, backend_kind="local")
    assert a.job.job_id != b.job.job_id


def test_replace_flag_does_not_change_job_id():
    assert (
        ScalarIndexJobRequest(table="t", column="c", table_version=1, replace=False).job_id()
        == ScalarIndexJobRequest(table="t", column="c", table_version=1, replace=True).job_id()
    )


# --------------------------------------------------------------------------- #
# Permission-denied and unsupported: ephemeral, persist nothing                #
# --------------------------------------------------------------------------- #


def test_permission_denied_request_is_ephemeral_and_persists_nothing():
    store = InMemoryScalarIndexJobStore()
    lake = FakeLake(
        connection_spec=_remote_spec(index_management=True),
        permission_denied=True,
        store=store,
    )
    result = request_scalar_index(lake, table="aligned_ticks", column="alignment_id")
    assert result.job.build_mode == MODE_PERMISSION_DENIED
    assert result.job.status == "skipped"
    assert result.index_result.status == "skipped"
    assert "authorized" in result.job.terminal_reason
    assert store.list() == []  # no dead job row that would block recovery


def test_unsupported_request_is_ephemeral():
    store = InMemoryScalarIndexJobStore()
    lake = FakeLake(connection_spec=_remote_spec(index_management=False), store=store)
    result = request_scalar_index(lake, table="aligned_ticks", column="run_id")
    assert result.job.build_mode == MODE_UNSUPPORTED
    assert result.job.status == "skipped"
    assert store.list() == []


# --------------------------------------------------------------------------- #
# Asynchronous submit + poll + reconcile                                       #
# --------------------------------------------------------------------------- #


def test_async_submit_records_pending_then_reconcile_completes():
    store = InMemoryScalarIndexJobStore()
    submitted: list[dict] = []

    def submit_fn(request):
        submitted.append(request)
        return {"status": "active"}

    def status_fn(*, job_id, request):
        return {"status": "complete", "num_rows": 1000}

    coordinator = ScalarIndexJobCoordinator(
        store, submit_fn=submit_fn, status_fn=status_fn, now_fn=_fixed_now()
    )
    result = coordinator.request(_request(), build_mode=MODE_ASYNCHRONOUS, backend_kind="db")
    assert result.job.status == "active"
    assert len(submitted) == 1

    changed = coordinator.reconcile()
    assert len(changed) == 1
    assert changed[0].status == "complete"
    assert changed[0].num_rows == 1000


def test_async_without_client_records_planned_submitted():
    store = InMemoryScalarIndexJobStore()
    coordinator = ScalarIndexJobCoordinator(store, now_fn=_fixed_now())
    result = coordinator.request(_request(), build_mode=MODE_ASYNCHRONOUS, backend_kind="db")
    assert result.job.status == "submitted"
    assert "no async" in result.job.status_history[-1]["reason"]


def test_async_submit_permission_error_marks_failed():
    store = InMemoryScalarIndexJobStore()

    def submit_fn(request):
        raise PermissionError("not authorized to build indexes")

    coordinator = ScalarIndexJobCoordinator(store, submit_fn=submit_fn, now_fn=_fixed_now())
    result = coordinator.request(_request(), build_mode=MODE_ASYNCHRONOUS, backend_kind="db")
    assert result.job.status == "failed"
    assert "permission denied" in result.job.terminal_reason


# --------------------------------------------------------------------------- #
# Retry / cancel / expire                                                      #
# --------------------------------------------------------------------------- #


def test_failed_job_is_retryable_and_resubmits():
    store = InMemoryScalarIndexJobStore()
    attempts: list[int] = []

    def build_fn(req):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("transient backend error")
        return ScalarIndexResult(table=req.table, column=req.column, status="built")

    coordinator = ScalarIndexJobCoordinator(store, build_fn=build_fn, now_fn=_fixed_now())
    first = coordinator.request(_request(), build_mode=MODE_SYNCHRONOUS, backend_kind="local")
    assert first.job.status == "failed"

    retried = coordinator.retry(first.job.job_id)
    assert retried.status == "complete"
    assert retried.retry_count == 1
    assert len(attempts) == 2


def test_retry_refuses_complete_job():
    store = InMemoryScalarIndexJobStore()
    coordinator = ScalarIndexJobCoordinator(
        store,
        build_fn=lambda req: ScalarIndexResult(table=req.table, column=req.column, status="built"),
        now_fn=_fixed_now(),
    )
    done = coordinator.request(_request(), build_mode=MODE_SYNCHRONOUS, backend_kind="local")
    with pytest.raises(ScalarIndexJobError):
        coordinator.retry(done.job.job_id)


def test_cancel_marks_canceled_and_is_resubmittable():
    store = InMemoryScalarIndexJobStore()
    coordinator = ScalarIndexJobCoordinator(store, now_fn=_fixed_now())
    result = coordinator.request(_request(), build_mode=MODE_ASYNCHRONOUS, backend_kind="db")
    canceled = coordinator.cancel(result.job.job_id, reason="no longer needed")
    assert canceled.status == "canceled"
    # A canceled job is resubmitted (not reused) on the next request.
    again = coordinator.request(_request(), build_mode=MODE_ASYNCHRONOUS, backend_kind="db")
    assert again.created


def test_ttl_expiry_marks_expired_then_next_request_resubmits():
    store = InMemoryScalarIndexJobStore()
    t0 = datetime(2026, 7, 13, tzinfo=UTC)
    clock = {"now": t0}
    coordinator = ScalarIndexJobCoordinator(
        store, ttl_s=60.0, now_fn=lambda: clock["now"]
    )
    result = coordinator.request(_request(), build_mode=MODE_ASYNCHRONOUS, backend_kind="db")
    assert result.job.status == "submitted"
    clock["now"] = t0 + timedelta(seconds=120)
    expired = coordinator.expire_due()
    assert expired and expired[0].status == "expired"
    resubmitted = coordinator.request(_request(), build_mode=MODE_ASYNCHRONOUS, backend_kind="db")
    assert resubmitted.created and resubmitted.job.retry_count == 1


# --------------------------------------------------------------------------- #
# Record serialization + durable LanceDB store                                 #
# --------------------------------------------------------------------------- #


def test_record_round_trips():
    record = build_scalar_index_job(
        _request(),
        build_mode=MODE_SYNCHRONOUS,
        backend_kind="local_path",
        now=datetime(2026, 7, 13, tzinfo=UTC),
        ttl_s=3600.0,
        store_kind="in-memory",
        store_ref="memory://x",
    )
    restored = ScalarIndexJob.from_dict(record.to_dict())
    assert restored.job_id == record.job_id
    assert restored.table == "aligned_ticks"
    assert restored.column == "alignment_id"
    assert restored.table_version == 3


def test_durable_store_dedups_and_survives_reopen(tmp_path):
    db = lancedb.connect(str(tmp_path / "jobs.lance"))
    store = LanceTableScalarIndexJobStore(db)
    coordinator = ScalarIndexJobCoordinator(
        store,
        build_fn=lambda req: ScalarIndexResult(table=req.table, column=req.column, status="built"),
        now_fn=_fixed_now(),
    )
    first = coordinator.request(_request(), build_mode=MODE_SYNCHRONOUS, backend_kind="local_path")
    coordinator.request(_request(), build_mode=MODE_SYNCHRONOUS, backend_kind="local_path")

    reopened = LanceTableScalarIndexJobStore(lancedb.connect(str(tmp_path / "jobs.lance")))
    job = reopened.get(first.job.job_id)
    assert job is not None
    assert job.status == "complete"
    assert job.request_count == 2
    assert len(reopened.list()) == 1  # deduplicated to one row
    assert SCALAR_INDEX_JOB_TABLE in reopened._table_names()


def test_durable_store_race_safe_claim(tmp_path):
    db = lancedb.connect(str(tmp_path / "jobs.lance"))
    store = LanceTableScalarIndexJobStore(db)
    now = _fixed_now()
    a = build_scalar_index_job(
        _request(), build_mode=MODE_SYNCHRONOUS, backend_kind="local",
        now=now(), ttl_s=None, store_kind="lancedb-table", store_ref="ref",
    )
    b = build_scalar_index_job(
        _request(), build_mode=MODE_SYNCHRONOUS, backend_kind="local",
        now=now(), ttl_s=None, store_kind="lancedb-table", store_ref="ref",
    )
    _, won_a = store.claim(a)
    _, won_b = store.claim(b)
    assert won_a and not won_b  # exactly one winner despite the same job id


def test_durable_claim_insert_only_does_not_overwrite_concurrent_winner(tmp_path):
    # Simulates the interleaving where two workers both pass the read-None check
    # before either commits: the claim path must be insert-if-absent, so the second
    # writer no-ops and the first writer's owner_nonce survives (real CAS, not
    # last-writer-wins). Otherwise both would believe they won and build twice.
    db = lancedb.connect(str(tmp_path / "jobs.lance"))
    store = LanceTableScalarIndexJobStore(db)
    now = _fixed_now()
    a = build_scalar_index_job(
        _request(), build_mode=MODE_ASYNCHRONOUS, backend_kind="db",
        now=now(), ttl_s=None, store_kind="lancedb-table", store_ref="ref",
    )
    b = build_scalar_index_job(
        _request(), build_mode=MODE_ASYNCHRONOUS, backend_kind="db",
        now=now(), ttl_s=None, store_kind="lancedb-table", store_ref="ref",
    )
    assert a.owner_nonce != b.owner_nonce
    store._insert_if_absent(a)
    store._insert_if_absent(b)  # same job_id already present -> must no-op
    winner = store.get(a.job_id)
    assert winner.owner_nonce == a.owner_nonce  # first writer wins, not overwritten
    assert len(store.list()) == 1


def test_reconcile_only_scans_in_flight_jobs():
    # A completed job must not be re-polled by reconcile (would be wasted work and,
    # at scale, would scan the whole terminal-dominated table). status_fn raising
    # proves reconcile never touches the completed job.
    store = InMemoryScalarIndexJobStore()
    _put_job(store, column="alignment_id", status="complete")

    def exploding_status(**_kwargs):
        raise AssertionError("reconcile must not poll a terminal job")

    coordinator = ScalarIndexJobCoordinator(
        store, status_fn=exploding_status, now_fn=_fixed_now()
    )
    assert coordinator.reconcile() == []


# --------------------------------------------------------------------------- #
# Manifest surfacing + strict-mode gate (training-layer integration)           #
# --------------------------------------------------------------------------- #


def _put_job(store, *, column, status, mode=MODE_SYNCHRONOUS, reason=None):
    now = datetime(2026, 7, 13, tzinfo=UTC)
    record = build_scalar_index_job(
        _request(column=column),
        build_mode=mode,
        backend_kind="db",
        now=now,
        ttl_s=None,
        store_kind="in-memory",
        store_ref="memory://x",
    ).with_status(status, now, reason=reason)
    store.put(record)


def test_manifest_merges_pending_job_reference():
    from lancedb_robotics.training import _merge_scalar_index_job_refs

    store = InMemoryScalarIndexJobStore()
    _put_job(store, column="alignment_id", status="submitted", mode=MODE_ASYNCHRONOUS)
    lake = FakeLake(store=store)
    params = (
        {"table": "aligned_ticks", "column": "alignment_id", "status": "skipped",
         "used_in_filter": True, "predicate_role": "filter"},
        {"table": "aligned_ticks", "column": "run_id", "status": "skipped",
         "used_in_filter": False, "predicate_role": "hot-column"},
    )
    merged = _merge_scalar_index_job_refs(lake, "aligned_ticks", params)
    by_col = {p["column"]: p for p in merged}
    assert by_col["alignment_id"]["job_status"] == "submitted"
    assert by_col["alignment_id"]["job_build_mode"] == MODE_ASYNCHRONOUS
    assert "job_status" not in by_col["run_id"]  # no job for this column


def test_manifest_merge_is_noop_without_job_table():
    from lancedb_robotics.training import _merge_scalar_index_job_refs

    lake = FakeLake()  # no store attached -> open returns None -> read-only no-op
    params = ({"table": "aligned_ticks", "column": "alignment_id", "status": "skipped"},)
    assert _merge_scalar_index_job_refs(lake, "aligned_ticks", params) == params


def test_strict_mode_raises_when_filter_predicate_unbacked():
    from lancedb_robotics.scalar_index_jobs import ScalarIndexRequiredError
    from lancedb_robotics.training import _enforce_predicate_index_requirement

    predicate_indexes = (
        {"column": "alignment_id", "status": "skipped", "used_in_filter": True,
         "predicate_role": "filter"},
    )
    with pytest.raises(ScalarIndexRequiredError):
        _enforce_predicate_index_requirement("aligned_ticks", predicate_indexes)


def test_strict_mode_accepts_present_index_or_completed_job():
    from lancedb_robotics.training import _enforce_predicate_index_requirement

    # present index -> ok
    _enforce_predicate_index_requirement(
        "aligned_ticks",
        ({"column": "alignment_id", "status": "already_present", "used_in_filter": True,
          "predicate_role": "filter"},),
    )
    # completed job -> ok
    _enforce_predicate_index_requirement(
        "aligned_ticks",
        ({"column": "alignment_id", "status": "skipped", "job_status": "complete",
          "used_in_filter": True, "predicate_role": "filter"},),
    )
    # non-filter roles never gate a read
    _enforce_predicate_index_requirement(
        "aligned_ticks",
        ({"column": "min_confidence", "status": "skipped", "used_in_filter": False,
          "predicate_role": "quality-diagnostic"},),
    )
