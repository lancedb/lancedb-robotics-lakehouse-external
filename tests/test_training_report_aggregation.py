"""Backlog 0123: multi-worker training loader report aggregation tests."""

import json

import pytest

from lancedb_robotics.training_report_aggregation import (
    TRAINING_LOADER_REPORT_AGGREGATE_KIND,
    AggregatedTrainingReport,
    TrainingReportAggregationError,
    aggregate_training_loader_reports,
    load_report_payloads,
)


def _worker_report(
    *,
    worker_id,
    num_workers=4,
    epoch=0,
    resolved_backend="enterprise",
    requested_backend="enterprise",
    loader_kind="native-training",
    cache_hits=0,
    cache_misses=0,
    bytes_read=0,
    rows_returned=0,
    row_ids_coalesced=0,
    remote_scan=0,
    remote_take=0,
    remote_filtered_read=0,
    by_plan_executor=None,
    prewarm_id=None,
    prewarm_status=None,
    prewarm_warm_bytes=0,
    prewarm_cold_bytes=0,
    prewarm_requests=0,
    fallback_events=None,
    disabled_capabilities=None,
    training_run_id="train-run-1",
    dataset_id="ds-1",
    snapshot_name="demo-v1",
    row_plan_id="row-plan-1",
    epoch_plan_id="epoch-plan-1",
):
    """A shape-accurate per-worker loader report payload (see training._*_loader_report_payload)."""
    summary = {
        "bytes_read": bytes_read,
        "rows_returned": rows_returned,
        "row_ids_coalesced": row_ids_coalesced,
        "pe_fanout": len(by_plan_executor or {}),
    }
    if prewarm_id is not None:
        summary["prewarm_status"] = prewarm_status
        summary["prewarm_warm_bytes"] = prewarm_warm_bytes
        summary["prewarm_cold_bytes"] = prewarm_cold_bytes
    return {
        "kind": "lancedb-robotics/training-loader-report/v1",
        "loader": {"kind": loader_kind, "access_pattern": "enterprise-remote-snapshot"},
        "lake": {
            "uri": "db://robotics",
            "connection_kind": "lancedb_remote_db",
            "execution_mode": "enterprise-remote",
        },
        "snapshot": {"id": dataset_id, "name": snapshot_name},
        "table_versions": [{"table": "observations", "version": 7}],
        "plans": {
            "row_plan_id": row_plan_id,
            "epoch_plan_id": epoch_plan_id,
            "epoch": epoch,
            "worker": {"id": worker_id, "num_workers": num_workers, "resume_from": 0},
        },
        "remote_execution": {
            "requested_backend": requested_backend,
            "resolved_backend": resolved_backend,
            "connection_kind": "lancedb_remote_db",
            "execution_mode": "enterprise-remote",
        },
        "policies": {
            "enterprise_cache": {
                "policy": "lazy",
                "prewarm_id": prewarm_id,
                "prewarm_requested": prewarm_id is not None,
                "prewarm_status": prewarm_status,
            }
        },
        "metrics": {
            "summary": summary,
            "operations_by_type": {
                "remote_scan": remote_scan,
                "remote_take": remote_take,
                "remote_filtered_read": remote_filtered_read,
                "prewarm": prewarm_requests,
            },
            "cache": {
                "hits": cache_hits,
                "misses": cache_misses,
                "by_plan_executor": by_plan_executor or {},
            },
        },
        "fallback_events": fallback_events or [],
        "disabled_capabilities": disabled_capabilities or [],
        "run": {"training_run_id": training_run_id},
    }


def test_two_workers_aggregate_to_one_job_level_summary():
    reports = [
        _worker_report(
            worker_id=0,
            cache_hits=10,
            cache_misses=2,
            bytes_read=1000,
            rows_returned=50,
            remote_take=5,
            by_plan_executor={"pe-a": {"hits": 6, "misses": 1}, "pe-b": {"hits": 4, "misses": 1}},
        ),
        _worker_report(
            worker_id=1,
            cache_hits=7,
            cache_misses=3,
            bytes_read=500,
            rows_returned=40,
            remote_take=4,
            by_plan_executor={"pe-a": {"hits": 3, "misses": 2}, "pe-c": {"hits": 4, "misses": 1}},
        ),
    ]

    aggregate = aggregate_training_loader_reports(reports)
    result = aggregate.to_dict()

    assert result["kind"] == TRAINING_LOADER_REPORT_AGGREGATE_KIND
    assert result["job"]["report_count"] == 2
    assert result["job"]["worker_count"] == 2
    assert result["job"]["workers"] == ["0/4", "1/4"]

    # Client-observable byte + row + operation totals sum across workers.
    assert result["totals"]["bytes_read"] == 1500
    assert result["totals"]["rows_hydrated"] == 90
    assert result["totals"]["operations"]["remote_take"] == 9
    # Server-internal metrics are NOT client-visible over the query-node protocol
    # and must not appear anywhere in the aggregate (backlog 0345): no cache
    # hits/misses/warm/cold, no plan-executor fanout, no per-PE breakdown.
    assert "cache_hits" not in result["totals"]
    assert "cache_misses" not in result["totals"]
    assert "cache" not in result
    assert "pe_fanout" not in result
    assert "plan_executors" not in result

    # Per-worker and per-epoch drill-downs keep the client-observable signals only.
    assert result["by_worker"]["0/4"]["bytes_read"] == 1000
    assert result["by_worker"]["1/4"]["rows_hydrated"] == 40
    assert "cache_warm" not in result["by_worker"]["0/4"]
    assert result["by_epoch"]["0"]["bytes_read"] == 1500
    assert result["by_epoch"]["0"]["rows_hydrated"] == 90
    assert "cache_warm" not in result["by_epoch"]["0"]
    assert result["warnings"] == []


def test_no_server_internal_cache_or_pe_surface():
    # A client over the query-node protocol gets no cache telemetry at all —
    # there is no cache block and no plan-executor surface, only
    # client-observable totals.
    reports = [
        _worker_report(worker_id=0, bytes_read=1000, rows_returned=50, remote_take=5),
        _worker_report(worker_id=1, bytes_read=500, rows_returned=40, remote_take=4),
    ]
    result = aggregate_training_loader_reports(reports).to_dict()

    assert "cache" not in result
    assert "pe_fanout" not in result
    assert "plan_executors" not in result
    # Client-observable totals are still there.
    assert result["totals"]["bytes_read"] == 1500
    assert result["totals"]["operations"]["remote_take"] == 9


def test_duplicate_prewarm_id_counts_once():
    # All four workers share one epoch-scoped prewarm (same prewarm_id, backlog 0121).
    # The client only knows the request identity + status (never server-internal
    # warmed-byte counts), and that identity must be deduplicated to one.
    reports = [
        _worker_report(
            worker_id=worker_id,
            prewarm_id="prewarm-shared-abc",
            prewarm_status="submitted",
            prewarm_requests=1,
        )
        for worker_id in range(4)
    ]

    result = aggregate_training_loader_reports(reports).to_dict()

    assert result["prewarm"]["unique_prewarm_ids"] == 1
    assert result["prewarm"]["statuses"] == {"submitted": 1}
    assert len(result["prewarm"]["requests"]) == 1
    entry = result["prewarm"]["requests"][0]
    assert entry["prewarm_id"] == "prewarm-shared-abc"
    assert sorted(entry["observed_by_workers"]) == ["0/4", "1/4", "2/4", "3/4"]
    # No server-internal warmed-byte / executor telemetry leaks into the section.
    assert "warm_bytes" not in result["prewarm"]
    assert "cold_bytes" not in result["prewarm"]
    assert "warm_bytes" not in entry
    assert "completed_executors" not in entry
    # The raw per-worker request count is still summed as an operation count.
    assert result["prewarm"]["requests_submitted"] == 4
    assert result["totals"]["operations"]["prewarm"] == 4


def test_distinct_prewarm_ids_dedup_independently():
    reports = [
        _worker_report(
            worker_id=0, epoch=0, prewarm_id="prewarm-epoch-0", prewarm_status="submitted"
        ),
        _worker_report(
            worker_id=0, epoch=1, prewarm_id="prewarm-epoch-1", prewarm_status="submitted"
        ),
    ]

    result = aggregate_training_loader_reports(reports).to_dict()

    assert result["prewarm"]["unique_prewarm_ids"] == 2
    assert {r["prewarm_id"] for r in result["prewarm"]["requests"]} == {
        "prewarm-epoch-0",
        "prewarm-epoch-1",
    }


def test_mixed_backend_and_fallback_produce_warnings():
    reports = [
        _worker_report(worker_id=0, resolved_backend="enterprise"),
        _worker_report(
            worker_id=1,
            resolved_backend="local",
            fallback_events=[
                {"from": "enterprise", "to": "local", "reason": "capability missing"}
            ],
        ),
    ]

    result = aggregate_training_loader_reports(reports).to_dict()

    warnings = "\n".join(result["warnings"])
    assert "mixed resolved backends" in warnings
    assert "enterprise" in warnings and "local" in warnings
    assert "1 of 2 workers fell back" in warnings

    # Each fallback event is tagged with its worker for drill-down.
    assert len(result["fallback_events"]) == 1
    assert result["fallback_events"][0]["worker"] == "1/4"
    assert result["fallback_events"][0]["resolved_backend"] == "local"
    assert result["by_worker"]["1/4"]["fell_back"] is True
    assert result["by_worker"]["0/4"]["fell_back"] is False


def test_mixed_loader_kinds_warn():
    reports = [
        _worker_report(worker_id=0, loader_kind="native-training"),
        _worker_report(worker_id=1, loader_kind="aligned-training"),
    ]
    result = aggregate_training_loader_reports(reports).to_dict()
    assert any("mixed loader kinds" in w for w in result["warnings"])


def test_disabled_capabilities_union_and_warning():
    reports = [
        _worker_report(worker_id=0, disabled_capabilities=["plan_executor.remote_take"]),
        _worker_report(worker_id=1, disabled_capabilities=["page_cache_prewarm"]),
    ]
    result = aggregate_training_loader_reports(reports).to_dict()
    assert result["disabled_capabilities"] == [
        "page_cache_prewarm",
        "plan_executor.remote_take",
    ]
    assert any("disabled capabilities" in w for w in result["warnings"])


def test_aggregation_is_order_independent_and_deterministic():
    reports = [
        _worker_report(worker_id=0, cache_hits=3, bytes_read=100),
        _worker_report(worker_id=1, cache_hits=5, bytes_read=200),
        _worker_report(worker_id=2, cache_hits=7, bytes_read=300),
    ]
    a = aggregate_training_loader_reports(reports).to_dict()
    b = aggregate_training_loader_reports(list(reversed(reports))).to_dict()
    assert a == b
    assert a["job"]["job_id"].startswith("trjob-")


def test_explicit_job_id_overrides_digest():
    reports = [_worker_report(worker_id=0)]
    result = aggregate_training_loader_reports(reports, job_id="my-job").to_dict()
    assert result["job"]["job_id"] == "my-job"


def test_empty_reports_raise():
    with pytest.raises(TrainingReportAggregationError):
        aggregate_training_loader_reports([])


def test_non_report_input_raises():
    with pytest.raises(TrainingReportAggregationError):
        aggregate_training_loader_reports([{"not": "a report"}])


def test_cannot_aggregate_an_aggregate():
    reports = [_worker_report(worker_id=0)]
    aggregate = aggregate_training_loader_reports(reports).to_dict()
    with pytest.raises(TrainingReportAggregationError):
        aggregate_training_loader_reports([aggregate])


def test_accepts_training_loader_report_objects():
    class _Report:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self):
            return self._payload

    reports = [_Report(_worker_report(worker_id=0, rows_returned=4))]
    result = aggregate_training_loader_reports(reports).to_dict()
    assert result["job"]["worker_count"] == 1
    assert result["totals"]["rows_hydrated"] == 4


def test_secrets_in_run_block_are_redacted():
    report = _worker_report(worker_id=0)
    report["run"]["authorization_header"] = "Bearer super-secret-token"
    result = aggregate_training_loader_reports([report]).to_dict()
    assert "super-secret-token" not in json.dumps(result)


def test_write_json_and_load_round_trip(tmp_path):
    reports = [
        _worker_report(worker_id=0, cache_hits=1),
        _worker_report(worker_id=1, cache_hits=2),
    ]
    aggregate = aggregate_training_loader_reports(reports)
    out = tmp_path / "agg.json"
    aggregate.write_json(out)
    reloaded = json.loads(out.read_text())
    assert reloaded == aggregate.to_dict()


def test_load_report_payloads_rejects_bad_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(TrainingReportAggregationError):
        load_report_payloads([bad])


def test_aggregated_report_properties():
    reports = [_worker_report(worker_id=0)]
    aggregate = aggregate_training_loader_reports(reports)
    assert isinstance(aggregate, AggregatedTrainingReport)
    assert aggregate.report_count == 1
    assert aggregate.job_id.startswith("trjob-")
    assert aggregate.warnings == ()


# --- shape-drift guard: aggregate reports produced by the real loader path ----


def test_aggregates_real_loader_reports_from_distinct_workers(tmp_path):
    from test_native_training_dataset import _mark_enterprise_lake, _training_lake

    lake = _mark_enterprise_lake(_training_lake(tmp_path / "robot.lance"))
    lake.query_node_cache_telemetry = {
        "remote_take": {"per_addr": {"pe-a": {"hits": 4, "misses": 1}}}
    }
    worker_reports = []
    for worker_id in range(2):
        dataset = lake.training.dataset(
            "demo-v1",
            columns=["observation_id", "payload"],
            media="bytes",
            backend="enterprise",
            cache_policy="lazy",
            worker_id=worker_id,
            num_workers=2,
        )
        dataset.__getitems__([0])
        worker_reports.append(dataset.loader_report(training_run_id="run-xyz"))

    result = lake.training.aggregate_reports(worker_reports).to_dict()

    assert result["kind"] == TRAINING_LOADER_REPORT_AGGREGATE_KIND
    assert result["job"]["report_count"] == 2
    assert sorted(result["job"]["workers"]) == ["0/2", "1/2"]
    assert result["job"]["training_run_ids"] == ["run-xyz"]
    # Real reports carry a resolved enterprise backend and client-observable totals.
    assert result["job"]["backends"]["resolved"] == ["enterprise"]
    assert result["totals"]["bytes_read"] >= 0
    # The read path never surfaces server-internal cache state or PE fanout to the
    # client, even when the loader injected a fake cache-metrics hook (backlog 0345).
    assert "cache" not in result
    assert "pe_fanout" not in result
    assert set(result["by_worker"]) == {"0/2", "1/2"}


# --- CLI: merge report JSON files (lake-free) ---------------------------------


def test_cli_report_merge_writes_deterministic_json(tmp_path):
    from typer.testing import CliRunner

    from lancedb_robotics.cli import app

    runner = CliRunner()
    paths = []
    for worker_id in range(2):
        report = _worker_report(
            worker_id=worker_id,
            cache_hits=5 + worker_id,
            cache_misses=1,
            bytes_read=1000,
            prewarm_id="prewarm-shared",
            prewarm_status="submitted",
            prewarm_warm_bytes=2048,
            prewarm_requests=1,
        )
        path = tmp_path / f"worker-{worker_id}.json"
        path.write_text(json.dumps(report, sort_keys=True))
        paths.append(str(path))

    out = tmp_path / "aggregate.json"
    args = ["train", "report", "merge", "--out", str(out), "--format", "json"]
    for path in paths:
        args += ["--report", path]

    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output

    written = json.loads(out.read_text())
    assert written["kind"] == TRAINING_LOADER_REPORT_AGGREGATE_KIND
    assert written["totals"]["bytes_read"] == 2000
    assert "cache" not in written
    assert "pe_fanout" not in written
    # Shared prewarm counted once even though two workers reported it.
    assert written["prewarm"]["unique_prewarm_ids"] == 1
    assert written["prewarm"]["requests_submitted"] == 2

    # Deterministic: re-running produces byte-identical output.
    out2 = tmp_path / "aggregate2.json"
    args2 = ["train", "report", "merge", "--out", str(out2), "--format", "json"]
    for path in paths:
        args2 += ["--report", path]
    assert runner.invoke(app, args2).exit_code == 0
    assert out2.read_text() == out.read_text()


def test_cli_report_merge_bad_file_exits_one(tmp_path):
    from typer.testing import CliRunner

    from lancedb_robotics.cli import app

    runner = CliRunner()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    result = runner.invoke(app, ["train", "report", "merge", "--report", str(bad)])
    assert result.exit_code == 1
    assert "error:" in result.output
