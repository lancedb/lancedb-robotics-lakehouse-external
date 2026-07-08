"""Tests for the Enterprise training report catalog and run history (backlog 0115).

Covers the durable ``training_reports`` catalog: idempotent recording keyed by
content digest, bounded queries by fallback reason / backend transition,
deterministic summary paging, full-report reload, cross-worker/epoch metric
aggregation, reference-based retention, and CLI JSON output. The synthetic-
payload tests pin exact semantics; the dataset-driven tests prove the real
``lake.training.record_report(dataset=...)`` integration and the
backward-compatibility guarantee for ``dataset.manifest.backend``.
"""

import json

import pytest

# Reuse the native-training lake fixtures (real snapshot + enterprise marking).
from test_native_training_dataset import _mark_enterprise_lake, _training_lake
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.run_manifests import (
    TRAINING_REPORTS_TABLE,
    RunManifestError,
)

runner = CliRunner()

LOADER_REPORT_KIND = "lancedb-robotics/training-loader-report/v1"


def _payload(
    lake_uri,
    *,
    epoch=0,
    worker_id=0,
    num_workers=1,
    cache_policy="lazy",
    hits=0,
    misses=0,
    bytes_read=100,
    fallback=False,
    training_run_id="train-abc",
    resolved_backend="enterprise",
):
    """A synthetic native-training loader report payload (0073 shape)."""
    return {
        "kind": LOADER_REPORT_KIND,
        "loader": {"kind": "native-training", "access_pattern": "row"},
        "lake": {"uri": lake_uri, "backend_kind": resolved_backend},
        "snapshot": {"id": "ds-1", "name": "pick-place-v17"},
        "table_versions": [{"table": "observations", "version": 3, "tag": ""}],
        "plans": {
            "row_plan_id": "rp-1",
            "epoch_plan_id": "ep-1",
            "epoch": epoch,
            "worker": {"id": worker_id, "num_workers": num_workers},
        },
        "policies": {"enterprise_cache": {"policy": cache_policy}},
        "remote_execution": {
            "requested_backend": "enterprise",
            "resolved_backend": resolved_backend,
            "connection_kind": "lancedb_remote_db",
            "execution_mode": "enterprise-remote",
            "cache": {"policy": cache_policy},
        },
        "metrics": {
            "summary": {
                "bytes_read": bytes_read,
                "rows_returned": 10,
                "pe_fanout": 2,
                "prewarm_status": "completed" if cache_policy == "prewarm" else "not-requested",
            },
            "cache": {"hits": hits, "misses": misses},
            "operations_by_type": {"prewarm": 1 if cache_policy == "prewarm" else 0},
        },
        "fallback_events": (
            [{"from": "enterprise", "to": "local", "reason": "no plan-executor in region"}]
            if fallback
            else []
        ),
        "run": {"training_run_id": training_run_id},
    }


@pytest.fixture
def lake(tmp_path):
    return _training_lake(tmp_path / "robot.lance")


# --- Idempotency & identity -------------------------------------------------

def test_two_epochs_distinct_rows_with_stable_digests(lake):
    first = lake.training.record_report(_payload(lake.uri, epoch=0, cache_policy="lazy"))
    second = lake.training.record_report(_payload(lake.uri, epoch=1, cache_policy="prewarm"))

    assert first.report_id != second.report_id
    assert lake.table(TRAINING_REPORTS_TABLE).count_rows() == 2

    # Re-recording identical content is a no-op replace with a stable digest.
    again = lake.training.record_report(_payload(lake.uri, epoch=0, cache_policy="lazy"))
    assert again.report_id == first.report_id
    assert again.report_digest == first.report_digest
    assert lake.table(TRAINING_REPORTS_TABLE).count_rows() == 2


def test_scalar_dimensions_extracted_from_payload(lake):
    record = lake.training.record_report(
        _payload(lake.uri, epoch=2, worker_id=1, num_workers=4, cache_policy="prewarm")
    )
    assert record.resolved_backend == "enterprise"
    assert record.cache_policy == "prewarm"
    assert record.prewarm_requested is True
    assert record.prewarm_status == "completed"
    assert record.epoch == 2
    assert record.worker_id == 1
    assert record.num_workers == 4
    assert record.snapshot_name == "pick-place-v17"
    assert record.training_run_id == "train-abc"


# --- Query by fallback reason / backend transition --------------------------

def test_query_by_fallback_reason_returns_only_fallback(lake):
    lake.training.record_report(_payload(lake.uri, epoch=0, fallback=False))
    lake.training.record_report(_payload(lake.uri, epoch=1, fallback=True))

    only_fallback = lake.training.query_reports(fallback=True)
    assert len(only_fallback.rows) == 1
    assert only_fallback.rows[0]["fallback_reason"] == "no plan-executor in region"
    assert only_fallback.plan.bounded is True

    by_reason = lake.training.query_reports(fallback_reason="no plan-executor in region")
    assert len(by_reason.rows) == 1

    by_transition = lake.training.query_reports(fallback_to_backend="local")
    assert len(by_transition.rows) == 1

    none_match = lake.training.query_reports(fallback_reason="does-not-exist")
    assert len(none_match.rows) == 0


# --- Aggregation ------------------------------------------------------------

def test_aggregate_sums_cache_hits_misses_by_worker_and_epoch(lake):
    # Two epochs x two workers of the same run, with known cache counters.
    lake.training.record_report(_payload(lake.uri, epoch=0, worker_id=0, num_workers=2, hits=5, misses=3))
    lake.training.record_report(_payload(lake.uri, epoch=0, worker_id=1, num_workers=2, hits=2, misses=4))
    lake.training.record_report(_payload(lake.uri, epoch=1, worker_id=0, num_workers=2, hits=9, misses=1))
    lake.training.record_report(_payload(lake.uri, epoch=1, worker_id=1, num_workers=2, hits=8, misses=0))

    agg = lake.training.report_metrics(training_run_id="train-abc")
    assert agg.report_count == 4
    assert agg.totals["cache_hits"] == 24
    assert agg.totals["cache_misses"] == 8
    # Worker 0 across both epochs: 5+9 hits, 3+1 misses.
    assert agg.by_worker["0/2"]["cache_hits"] == 14
    assert agg.by_worker["0/2"]["cache_misses"] == 4
    # Epoch 1 across both workers: 9+8 hits, 1+0 misses.
    assert agg.by_epoch["1"]["cache_hits"] == 17
    assert agg.by_epoch["1"]["cache_misses"] == 1


# --- Paging & summary projection --------------------------------------------

def test_reports_paging_is_deterministic_and_omits_payloads(lake):
    for index in range(5):
        lake.training.record_report(_payload(lake.uri, epoch=index))

    seen = []
    token = None
    pages = 0
    while True:
        page = lake.training.reports(page_size=2, page_token=token)
        assert page.table == TRAINING_REPORTS_TABLE
        # Summary rows must never carry the large payload columns.
        for row in page.rows:
            assert "report_json" not in row
            assert "backend_json" not in row
        seen.extend(row["report_id"] for row in page.rows)
        pages += 1
        token = page.next_page_token
        if token is None:
            break

    assert pages == 3
    assert len(seen) == 5
    assert len(set(seen)) == 5

    # Re-walking the pages yields identical ordering.
    again = []
    token = None
    while True:
        page = lake.training.reports(page_size=2, page_token=token)
        again.extend(row["report_id"] for row in page.rows)
        token = page.next_page_token
        if token is None:
            break
    assert again == seen


# --- Full report reload -----------------------------------------------------

def test_get_report_reloads_full_payload_by_id_and_handle(lake):
    record = lake.training.record_report(_payload(lake.uri, epoch=3, worker_id=2, num_workers=4))

    by_id = lake.training.get_report(record.report_id)
    assert by_id.report["kind"] == LOADER_REPORT_KIND
    assert by_id.report["plans"]["epoch"] == 3
    assert by_id.backend  # remote-execution projection stored as backend

    by_handle = lake.training.get_report(training_run_id="train-abc", epoch=3, worker_id=2)
    assert by_handle.report_id == record.report_id

    with pytest.raises(RunManifestError, match="no training report"):
        lake.training.get_report("trpt-nonexistent")


# --- Retention (reference-based) --------------------------------------------

def test_reports_of_live_runs_are_protected(lake):
    live = lake.training.record_run(snapshot="demo-v1", training_run_id="train-live")
    protected = lake.training.record_report(
        _payload(lake.uri, training_run_id=live.training_run_id)
    )
    orphan = lake.training.record_report(
        _payload(lake.uri, epoch=7, training_run_id="train-ghost")
    )

    plan = lake.training.report_retention_plan()
    protected_ids = {item.manifest_id for item in plan.protected}
    deletable_ids = {item.manifest_id for item in plan.deletable}
    assert protected.report_id in protected_ids
    assert orphan.report_id in deletable_ids

    # Protected report refuses deletion without force; orphan deletes freely.
    with pytest.raises(RunManifestError, match="protected"):
        lake.training.expire_report(protected.report_id)
    forced = lake.training.expire_report(protected.report_id, force=True)
    assert forced["deleted"] is True
    dropped = lake.training.expire_report(orphan.report_id)
    assert dropped["deleted"] is True
    assert lake.table(TRAINING_REPORTS_TABLE).count_rows() == 0


# --- Dataset integration + backward compatibility ---------------------------

def test_record_report_from_local_fallback_dataset(lake):
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        allow_fallback=True,
    )
    record = lake.training.record_report(dataset=dataset, training_run_id="train-fallback")

    assert record.requested_backend == "enterprise"
    assert record.resolved_backend == "local"
    assert record.fallback is True
    assert record.fallback_from_backend == "enterprise"
    assert record.fallback_to_backend == "local"
    assert record.training_run_id == "train-fallback"
    # The in-memory manifest.backend shape stays readable and unchanged (AC).
    assert dataset.manifest.to_dict()["backend"]["resolved_backend"] == "local"


def test_record_report_from_enterprise_dataset(lake):
    _mark_enterprise_lake(lake)
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        cache_policy="epoch",
        prewarm=True,
    )
    record = lake.training.record_report(dataset=dataset, training_run_id="train-ent")

    assert record.resolved_backend == "enterprise"
    assert record.connection_kind == "lancedb_remote_db"
    assert record.cache_policy == "epoch"
    assert record.prewarm_requested is True
    # No secrets leak into the persisted payloads.
    persisted = json.dumps(record.report) + json.dumps(record.backend)
    assert "secret-api-key" not in persisted


def test_backward_compatible_empty_catalog(lake):
    # A lake with no recorded reports lists an empty page without error (AC).
    page = lake.training.reports()
    assert page.rows == ()
    assert page.total_count == 0
    # And the manifest.backend shape is still produced by datasets.
    dataset = lake.training.dataset("demo-v1")
    assert "backend" in dataset.manifest.to_dict()


# --- CLI --------------------------------------------------------------------

def test_cli_report_list_get_metrics_json(lake, tmp_path):
    lake_uri = str(lake.uri)
    lake.training.record_report(_payload(lake.uri, epoch=0, hits=5, misses=3))
    lake.training.record_report(_payload(lake.uri, epoch=1, hits=9, misses=1, fallback=True))

    listing = runner.invoke(app, ["train", "report", "list", "--lake", lake_uri, "--format", "json"])
    assert listing.exit_code == 0, listing.output
    payload = json.loads(listing.output)
    assert payload["table"] == TRAINING_REPORTS_TABLE
    assert len(payload["rows"]) == 2

    # Deterministic: identical invocation yields identical JSON.
    listing2 = runner.invoke(app, ["train", "report", "list", "--lake", lake_uri, "--format", "json"])
    assert listing2.output == listing.output

    only_fallback = runner.invoke(
        app, ["train", "report", "list", "--lake", lake_uri, "--fallback", "--format", "json"]
    )
    assert only_fallback.exit_code == 0
    assert len(json.loads(only_fallback.output)["rows"]) == 1

    report_id = payload["rows"][0]["report_id"]
    got = runner.invoke(
        app, ["train", "report", "get", "--lake", lake_uri, "--id", report_id, "--format", "json"]
    )
    assert got.exit_code == 0, got.output
    assert json.loads(got.output)["report_id"] == report_id
    assert "report" in json.loads(got.output)

    metrics = runner.invoke(
        app,
        ["train", "report", "metrics", "--lake", lake_uri, "--training-run", "train-abc", "--format", "json"],
    )
    assert metrics.exit_code == 0, metrics.output
    agg = json.loads(metrics.output)
    assert agg["report_count"] == 2
    assert agg["totals"]["cache_hits"] == 14
