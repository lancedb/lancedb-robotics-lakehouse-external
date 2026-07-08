"""Unit tests for the durable prewarm JobRun store and coordinator (backlog 0121)."""

from datetime import UTC, datetime, timedelta

import lancedb
import pytest

from lancedb_robotics.training_prewarm_jobs import (
    PREWARM_JOB_TABLE,
    InMemoryPrewarmJobRunStore,
    LanceTablePrewarmJobRunStore,
    PrewarmJobCoordinator,
    PrewarmJobRun,
    PrewarmJobRunError,
    build_prewarm_job_run,
)


def _request(prewarm_id="prewarm-abc", **overrides):
    request = {
        "kind": "lancedb-robotics/training-prewarm/v1",
        "requested": True,
        "prewarm_id": prewarm_id,
        "policy": "epoch",
        "scope": "epoch",
        "snapshot_name": "demo",
        "dataset_id": "ds-1",
        "row_plan_id": "row-1",
        "epoch_plan_id": "epoch-1",
        "routing": {"mode": "host-override", "host_override": "https://phalanx"},
        "table_uri": "db://robotics",
        "tables": [
            {"table": "observations", "version": 3, "label": "db://robotics/observations@v3"}
        ],
        "projected_columns": ["observation_id"],
        "logical_columns": ["observation_id"],
        "excluded_columns": [],
        "row_count": 3,
        "estimated_bytes": 0,
        "limits": {"max_rows": 10},
        "worker": {"id": 0, "num_workers": 2},
    }
    request.update(overrides)
    return request


def _coordinator(store, *, submit=None, status=None, now=None, ttl_s=3600.0):
    submits = []

    def default_submit(req):
        submits.append(req)
        return {"status": "active", "pe_fanout": 2}

    def default_status(**kwargs):
        return {"status": "complete", "completed_executors": 2, "failed_executors": 0}

    coordinator = PrewarmJobCoordinator(
        store,
        submit_fn=submit if submit is not None else default_submit,
        status_fn=status if status is not None else default_status,
        now_fn=now if now is not None else (lambda: datetime(2026, 7, 6, tzinfo=UTC)),
        ttl_s=ttl_s,
    )
    return coordinator, submits


def test_two_workers_share_one_jobrun_and_two_status_reads():
    store = InMemoryPrewarmJobRunStore()
    coordinator, submits = _coordinator(store)

    result0 = coordinator.submit_or_attach(_request(), worker_label="worker-0/2", wait=True)
    result1 = coordinator.submit_or_attach(_request(), worker_label="worker-1/2")

    assert result0.created is True
    assert result1.attached is True and result1.created is False
    assert len(submits) == 1
    run = store.get("prewarm-abc")
    assert run.attach_count == 2
    assert run.workers == ("worker-0/2", "worker-1/2")
    assert run.status == "complete"


def test_failed_jobrun_records_reason_and_is_resubmittable():
    store = InMemoryPrewarmJobRunStore()

    def bad(req):
        raise RuntimeError("plan executor down")

    coordinator, _ = _coordinator(store, submit=bad, status=None)
    result = coordinator.submit_or_attach(_request(), worker_label="w0")
    assert result.record.status == "failed"
    assert "plan executor down" in result.record.terminal_reason

    # Reopening a failed JobRun re-submits (a fresh attempt, retry_count++).
    calls = []

    def ok(req):
        calls.append(req)
        return {"status": "complete"}

    coordinator2 = PrewarmJobCoordinator(store, submit_fn=ok, now_fn=lambda: datetime(2026, 7, 6, tzinfo=UTC))
    result2 = coordinator2.submit_or_attach(_request(), worker_label="w0")
    assert result2.record.status == "complete"
    assert store.get("prewarm-abc").retry_count == 1
    assert len(calls) == 1


def test_retry_refuses_warm_and_in_flight_jobruns():
    store = InMemoryPrewarmJobRunStore()
    coordinator, _ = _coordinator(store)
    coordinator.submit_or_attach(_request(), worker_label="w0", wait=True)
    # Complete (warm) -> retry is refused.
    with pytest.raises(PrewarmJobRunError):
        coordinator.retry("prewarm-abc")
    with pytest.raises(PrewarmJobRunError):
        coordinator.retry("prewarm-unknown")


def test_canceled_jobrun_is_not_warm_and_resubmits():
    store = InMemoryPrewarmJobRunStore()
    coordinator, submits = _coordinator(store, status=None)  # stays active
    coordinator.submit_or_attach(_request(), worker_label="w0")
    canceled = coordinator.cancel("prewarm-abc", reason="not needed")
    assert canceled.status == "canceled"
    assert not canceled.is_warm(datetime(2026, 7, 6, tzinfo=UTC))

    coordinator.submit_or_attach(_request(), worker_label="w0")
    assert len(submits) == 2
    assert store.get("prewarm-abc").retry_count == 1


def test_ttl_expiry_marks_expired_and_next_open_resubmits():
    store = InMemoryPrewarmJobRunStore()
    clock = {"t": datetime(2026, 7, 6, tzinfo=UTC)}
    submits = []

    def submit(req):
        submits.append(req)
        return {"status": "complete"}

    coordinator, _ = _coordinator(
        store, submit=submit, status=None, now=lambda: clock["t"], ttl_s=60.0
    )
    coordinator.submit_or_attach(_request(), worker_label="w0")
    assert store.get("prewarm-abc").is_warm(clock["t"]) is True

    clock["t"] = clock["t"] + timedelta(seconds=120)
    expired = coordinator.expire_due()
    assert [run.prewarm_id for run in expired] == ["prewarm-abc"]
    assert store.get("prewarm-abc").status == "expired"

    coordinator.submit_or_attach(_request(), worker_label="w0")
    assert len(submits) == 2
    assert store.get("prewarm-abc").status == "complete"


def test_jobrun_record_round_trips_and_is_secret_free():
    record = build_prewarm_job_run(
        _request(),
        job_label="ds-1",
        caller_label="trainer-42",
        now=datetime(2026, 7, 6, tzinfo=UTC),
        ttl_s=3600.0,
        store_kind="in-memory",
        store_ref="memory://x",
    )
    payload = record.to_dict()  # asserts secret-free internally
    assert "api_key" not in payload
    restored = PrewarmJobRun.from_dict(payload)
    assert restored.prewarm_id == record.prewarm_id
    assert restored.table_versions == record.table_versions
    assert restored.limits == record.limits


def test_secret_free_assertion_rejects_credentials():
    record = build_prewarm_job_run(
        _request(routing={"api_key": "secret"}),
        job_label="ds-1",
        caller_label=None,
        now=datetime(2026, 7, 6, tzinfo=UTC),
        ttl_s=None,
        store_kind="in-memory",
        store_ref="memory://x",
    )
    with pytest.raises(PrewarmJobRunError):
        record.to_dict()


def test_durable_store_dedups_and_survives_reopen(tmp_path):
    db = lancedb.connect(str(tmp_path / "jobs.lance"))
    store = LanceTablePrewarmJobRunStore(db)
    coordinator, submits = _coordinator(store)

    coordinator.submit_or_attach(_request(), worker_label="worker-0/2", wait=True)
    coordinator.submit_or_attach(_request(), worker_label="worker-1/2")
    assert len(submits) == 1

    # Fresh connection == fresh process: the JobRun is still visible.
    reopened = LanceTablePrewarmJobRunStore(lancedb.connect(str(tmp_path / "jobs.lance")))
    run = reopened.get("prewarm-abc")
    assert run.attach_count == 2
    assert run.status == "complete"
    assert len(reopened.list(status="complete")) == 1
    assert PREWARM_JOB_TABLE in reopened._table_names()
