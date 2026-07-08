"""Backlog 0123: multi-worker training loader report aggregation.

Backlog 0073 makes every native or aligned loader emit a structured,
JSON-serializable :class:`~lancedb_robotics.training.TrainingLoaderReport`. In a
distributed PyTorch / Ray / orchestrated job, *each worker* (a PyTorch
``DataLoader`` worker, or any process handed an explicit ``worker_id``/
``num_workers``) emits its own partial report. Production teams need one
job-level view that combines what those workers observed without double-counting
events that are shared across workers (most importantly the epoch-scoped prewarm
from backlog 0121, whose ``prewarm_id`` is identical for every worker opening the
same snapshot/seed/epoch).

:func:`aggregate_training_loader_reports` reduces a set of per-worker reports into
one deterministic, secret-free :class:`AggregatedTrainingReport`.

Client model (backlog 0345). A training worker is a **client** of the
query node over HTTP; it does not talk to a plan executor and cannot
read server-internal metrics. This aggregate therefore reports **only what the
client itself produces or receives**:

* rows planned/requested/coalesced/returned (the client builds the plan and counts
  what comes back);
* the count of read/prewarm requests the client issued;
* ``bytes_read`` — the volume the client pulled (populated only when byte
  accounting is wired; ``0`` otherwise);
* prewarm identity and *status*: which ``prewarm_id`` values the job submitted, the
  status its own control-plane call returned, and which workers shared each one.

It deliberately does **not** report any *server-internal cache state* — no cache
hits/misses / warm/cold, no plan-executor fanout or per-PE breakdown, and no
prewarm warmed-byte / executor counts. Those live behind the query node / plan
executors (their page cache, their Prometheus/OTel), not on the client data path,
so a worker cannot retrieve them. (Earlier revisions surfaced ``pe_fanout``,
``by_plan_executor`` and a ``cache`` warm/cold block; both were the over-modeling
0345 exists to remove.)

Prewarm request/status is deduplicated by ``prewarm_id`` so a shared epoch prewarm
counts once; per-worker and per-epoch drill-downs are preserved; and
mixed-backend, mixed-loader, and fallback states surface as job-level warnings.

The output is JSON-serializable and passed through the same redaction as the
per-worker reports, so it stays compatible with a future 0115-style report
catalog persistence (a non-goal here).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lancedb_robotics.training import (
    _jsonable,
    _mapping_dict,
    _redact_report,
    _stable_digest,
)

TRAINING_LOADER_REPORT_AGGREGATE_KIND = (
    "lancedb-robotics/training-loader-report-aggregate/v1"
)

# Client-observable summary metrics that are additive across workers: the client
# builds the plan (so it knows how many rows it planned/requested/coalesced) and
# counts what comes back and how many bytes it pulled. It does NOT include cache
# hits/misses or any server-internal counter.
_SUMMABLE_SUMMARY_KEYS: tuple[str, ...] = (
    "rows_planned",
    "row_count",
    "hydration_requests",
    "row_ids_requested",
    "row_ids_unique",
    "row_ids_coalesced",
    "rows_returned",
    "bytes_read",
)

_OPERATION_KEYS: tuple[str, ...] = (
    "remote_scan",
    "remote_take",
    "remote_filtered_read",
    "prewarm",
)


class TrainingReportAggregationError(Exception):
    """Raised when reports cannot be combined into a job-level view."""


@dataclass(frozen=True)
class AggregatedTrainingReport:
    """Deterministic, JSON-serializable job-level roll-up of worker reports."""

    payload: Mapping[str, Any]

    @property
    def job_id(self) -> str:
        job = self.payload.get("job")
        return str(job.get("job_id")) if isinstance(job, Mapping) else ""

    @property
    def report_count(self) -> int:
        job = self.payload.get("job")
        return int(job.get("report_count", 0)) if isinstance(job, Mapping) else 0

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self.payload.get("warnings") or ())

    def to_dict(self) -> dict[str, Any]:
        # Inputs are already redacted, but redact again defensively in case a raw
        # (unredacted) report payload was passed directly.
        return _redact_report(_jsonable(dict(self.payload)))

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)

    def write_json(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
        sort_keys: bool = True,
    ) -> None:
        Path(path).write_text(self.to_json(indent=indent, sort_keys=sort_keys) + "\n")


def _coerce_report(report: Any) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        payload = report.to_dict()
    elif isinstance(report, Mapping):
        payload = dict(report)
    else:
        raise TrainingReportAggregationError(
            "each report must be a TrainingLoaderReport, an object with to_dict(), "
            "or a payload mapping"
        )
    if not isinstance(payload, Mapping):
        raise TrainingReportAggregationError("report payload must be a JSON object")
    kind = str(payload.get("kind") or "")
    if kind == TRAINING_LOADER_REPORT_AGGREGATE_KIND:
        raise TrainingReportAggregationError(
            "cannot aggregate an already-aggregated report; pass per-worker "
            "TrainingLoaderReport payloads"
        )
    if "metrics" not in payload and "loader" not in payload:
        raise TrainingReportAggregationError(
            "report payload does not look like a training loader report "
            "(missing 'loader'/'metrics')"
        )
    return dict(payload)


def _agg_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _worker_key(worker: Mapping[str, Any]) -> str | None:
    worker_id = worker.get("id")
    if worker_id is None:
        return None
    num_workers = worker.get("num_workers")
    return f"{worker_id}/{num_workers}" if num_workers is not None else str(worker_id)


def aggregate_training_loader_reports(
    reports: Iterable[Any],
    *,
    job_id: str | None = None,
) -> AggregatedTrainingReport:
    """Combine per-worker loader reports into one deterministic job-level report.

    ``reports`` is any iterable of :class:`TrainingLoaderReport` (or objects with
    ``to_dict()``) or already-serialized payload mappings, one per worker/epoch.
    The result is order-independent: the same set of reports always produces the
    same ``job_id`` and body.
    """
    payloads = [_coerce_report(report) for report in reports]
    if not payloads:
        raise TrainingReportAggregationError(
            "aggregate_training_loader_reports requires at least one report"
        )

    totals = {key: 0 for key in _SUMMABLE_SUMMARY_KEYS}
    operations = {key: 0 for key in _OPERATION_KEYS}

    by_worker: dict[str, dict[str, Any]] = {}
    by_epoch: dict[str, dict[str, int]] = {}

    # Prewarm dedup: prewarm_id -> merged client-known request identity + status.
    # No warmed-byte / executor counters: those are server-internal cache telemetry.
    prewarm_by_id: dict[str, dict[str, Any]] = {}

    loader_kinds: set[str] = set()
    requested_backends: set[str] = set()
    resolved_backends: set[str] = set()
    connection_kinds: set[str] = set()
    execution_modes: set[str] = set()
    training_run_ids: set[str] = set()
    model_run_ids: set[str] = set()
    model_ids: set[str] = set()
    dataset_ids: set[str] = set()
    snapshot_names: set[str] = set()
    alignment_ids: set[str] = set()
    alignment_names: set[str] = set()
    row_plan_ids: set[str] = set()
    tick_plan_ids: set[str] = set()
    epoch_plan_ids: set[str] = set()
    epochs: set[int] = set()
    worker_keys: set[str] = set()
    lake_uris: set[str] = set()
    table_versions: dict[str, dict[str, Any]] = {}
    disabled_capabilities: set[str] = set()

    fallback_events: list[dict[str, Any]] = []
    fell_back_workers: set[str] = set()

    for payload in payloads:
        loader = _mapping_dict(payload.get("loader"))
        lake = _mapping_dict(payload.get("lake"))
        snapshot = _mapping_dict(payload.get("snapshot"))
        alignment = _mapping_dict(payload.get("alignment"))
        plans = _mapping_dict(payload.get("plans"))
        worker = _mapping_dict(plans.get("worker"))
        remote = _mapping_dict(payload.get("remote_execution"))
        metrics = _mapping_dict(payload.get("metrics"))
        summary = _mapping_dict(metrics.get("summary"))
        ops_by_type = _mapping_dict(metrics.get("operations_by_type"))
        enterprise_cache = _mapping_dict(
            _mapping_dict(payload.get("policies")).get("enterprise_cache")
        )
        run = _mapping_dict(payload.get("run"))

        # --- identity dimensions -------------------------------------------
        if loader.get("kind"):
            loader_kinds.add(str(loader.get("kind")))
        for holder, target in (
            (remote.get("requested_backend"), requested_backends),
            (remote.get("resolved_backend"), resolved_backends),
            (remote.get("connection_kind") or lake.get("connection_kind"), connection_kinds),
            (remote.get("execution_mode") or lake.get("execution_mode"), execution_modes),
            (run.get("training_run_id"), training_run_ids),
            (run.get("model_run_id"), model_run_ids),
            (run.get("model_id"), model_ids),
            (snapshot.get("id"), dataset_ids),
            (snapshot.get("name"), snapshot_names),
            (alignment.get("id"), alignment_ids),
            (alignment.get("name"), alignment_names),
            (plans.get("row_plan_id"), row_plan_ids),
            (plans.get("tick_plan_id"), tick_plan_ids),
            (plans.get("epoch_plan_id"), epoch_plan_ids),
            (lake.get("uri"), lake_uris),
        ):
            if holder is not None:
                target.add(str(holder))
        epoch = plans.get("epoch")
        if epoch is not None:
            epochs.add(int(epoch))
        for version in payload.get("table_versions") or []:
            version_map = _mapping_dict(version)
            name = version_map.get("table") or version_map.get("name")
            if name is not None:
                table_versions[str(name)] = dict(version_map)
        for capability in payload.get("disabled_capabilities") or []:
            disabled_capabilities.add(str(capability))

        # --- client-observable summable metrics ----------------------------
        for key in _SUMMABLE_SUMMARY_KEYS:
            totals[key] += _agg_int(summary.get(key))
        for op in _OPERATION_KEYS:
            operations[op] += _agg_int(ops_by_type.get(op))

        # --- per-worker drill-down -----------------------------------------
        worker_key = _worker_key(worker)
        if worker_key is not None:
            worker_keys.add(worker_key)
            entry = by_worker.setdefault(
                worker_key,
                {
                    "bytes_read": 0,
                    "rows_hydrated": 0,
                    "reports": 0,
                    "epochs": set(),
                    "resolved_backends": set(),
                    "fell_back": False,
                },
            )
            entry["bytes_read"] += _agg_int(summary.get("bytes_read"))
            entry["rows_hydrated"] += _agg_int(summary.get("rows_returned"))
            entry["reports"] += 1
            if epoch is not None:
                entry["epochs"].add(int(epoch))
            if remote.get("resolved_backend") is not None:
                entry["resolved_backends"].add(str(remote.get("resolved_backend")))

        # --- per-epoch drill-down ------------------------------------------
        if epoch is not None:
            epoch_bucket = by_epoch.setdefault(
                str(epoch), {"bytes_read": 0, "rows_hydrated": 0}
            )
            epoch_bucket["bytes_read"] += _agg_int(summary.get("bytes_read"))
            epoch_bucket["rows_hydrated"] += _agg_int(summary.get("rows_returned"))

        # --- prewarm dedup by prewarm_id (identity + status only) ----------
        prewarm_id = enterprise_cache.get("prewarm_id")
        prewarm_status = summary.get("prewarm_status") or enterprise_cache.get(
            "prewarm_status"
        )
        prewarm_requested = bool(enterprise_cache.get("prewarm_requested")) or (
            prewarm_status is not None and str(prewarm_status) != "not-requested"
        )
        if prewarm_id is not None and prewarm_requested:
            merged = prewarm_by_id.setdefault(
                str(prewarm_id),
                {
                    "prewarm_id": str(prewarm_id),
                    "statuses": set(),
                    "observed_by_workers": set(),
                    # ``row_count`` is the size of the client's own warm request.
                    "row_count": 0,
                },
            )
            merged["row_count"] = max(
                merged["row_count"], _agg_int(summary.get("prewarm_row_count"))
            )
            if prewarm_status is not None:
                merged["statuses"].add(str(prewarm_status))
            if worker_key is not None:
                merged["observed_by_workers"].add(worker_key)

        # --- fallback events (tag each with its worker for drill-down) -----
        report_events = [
            e for e in (payload.get("fallback_events") or []) if isinstance(e, Mapping)
        ]
        if report_events:
            if worker_key is not None:
                fell_back_workers.add(worker_key)
                if worker_key in by_worker:
                    by_worker[worker_key]["fell_back"] = True
            for event in report_events:
                tagged = dict(event)
                tagged.setdefault("worker", worker_key)
                tagged.setdefault("resolved_backend", remote.get("resolved_backend"))
                fallback_events.append(tagged)

    # --- prewarm roll-up (unique ids only) ---------------------------------
    prewarm_requests: list[dict[str, Any]] = []
    prewarm_status_counts: dict[str, int] = {}
    prewarm_row_count = 0
    for merged in prewarm_by_id.values():
        prewarm_row_count += int(merged["row_count"])
        for status in merged["statuses"]:
            prewarm_status_counts[status] = prewarm_status_counts.get(status, 0) + 1
        prewarm_requests.append(
            {
                "prewarm_id": merged["prewarm_id"],
                "statuses": sorted(merged["statuses"]),
                "observed_by_workers": sorted(merged["observed_by_workers"]),
                "row_count": int(merged["row_count"]),
            }
        )
    prewarm_requests.sort(key=lambda item: item["prewarm_id"])

    # --- warnings ----------------------------------------------------------
    warnings = _build_warnings(
        loader_kinds=loader_kinds,
        resolved_backends=resolved_backends,
        worker_keys=worker_keys,
        fell_back_workers=fell_back_workers,
        disabled_capabilities=disabled_capabilities,
    )

    resolved_job_id = job_id or _job_identity_digest(
        training_run_ids=training_run_ids,
        dataset_ids=dataset_ids,
        alignment_ids=alignment_ids,
        row_plan_ids=row_plan_ids,
        tick_plan_ids=tick_plan_ids,
        epoch_plan_ids=epoch_plan_ids,
    )

    payload = {
        "kind": TRAINING_LOADER_REPORT_AGGREGATE_KIND,
        "job": {
            "job_id": resolved_job_id,
            "report_count": len(payloads),
            "worker_count": len(worker_keys),
            "loader_kinds": sorted(loader_kinds),
            "training_run_ids": sorted(training_run_ids),
            "model_run_ids": sorted(model_run_ids),
            "model_ids": sorted(model_ids),
            "dataset_ids": sorted(dataset_ids),
            "snapshot_names": sorted(snapshot_names),
            "alignment_ids": sorted(alignment_ids),
            "alignment_names": sorted(alignment_names),
            "row_plan_ids": sorted(row_plan_ids),
            "tick_plan_ids": sorted(tick_plan_ids),
            "epoch_plan_ids": sorted(epoch_plan_ids),
            "epochs": sorted(epochs),
            "workers": sorted(worker_keys),
            "lake_uris": sorted(lake_uris),
            "backends": {
                "requested": sorted(requested_backends),
                "resolved": sorted(resolved_backends),
                "connection_kinds": sorted(connection_kinds),
                "execution_modes": sorted(execution_modes),
            },
            "table_versions": [table_versions[name] for name in sorted(table_versions)],
        },
        "totals": {
            "bytes_read": totals["bytes_read"],
            "rows_hydrated": totals["rows_returned"],
            "rows_returned": totals["rows_returned"],
            "rows_planned": totals["rows_planned"],
            "row_count": totals["row_count"],
            "hydration_requests": totals["hydration_requests"],
            "row_ids_requested": totals["row_ids_requested"],
            "row_ids_unique": totals["row_ids_unique"],
            "row_ids_coalesced": totals["row_ids_coalesced"],
            "operations": operations,
        },
        "prewarm": {
            "unique_prewarm_ids": len(prewarm_by_id),
            "requests_submitted": operations["prewarm"],
            "row_count": prewarm_row_count,
            "statuses": {
                status: prewarm_status_counts[status]
                for status in sorted(prewarm_status_counts)
            },
            "requests": prewarm_requests,
        },
        "by_worker": _finalize_worker_breakdown(by_worker),
        "by_epoch": {epoch: by_epoch[epoch] for epoch in sorted(by_epoch)},
        "fallback_events": fallback_events,
        "disabled_capabilities": sorted(disabled_capabilities),
        "warnings": warnings,
    }
    return AggregatedTrainingReport(_redact_report(_jsonable(payload)))


def _finalize_worker_breakdown(
    by_worker: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(by_worker):
        entry = by_worker[key]
        resolved = sorted(entry["resolved_backends"])
        result[key] = {
            "bytes_read": int(entry["bytes_read"]),
            "rows_hydrated": int(entry["rows_hydrated"]),
            "reports": int(entry["reports"]),
            "epochs": sorted(entry["epochs"]),
            "resolved_backends": resolved,
            "resolved_backend": resolved[0] if len(resolved) == 1 else None,
            "fell_back": bool(entry.get("fell_back")),
        }
    return result


def _build_warnings(
    *,
    loader_kinds: set[str],
    resolved_backends: set[str],
    worker_keys: set[str],
    fell_back_workers: set[str],
    disabled_capabilities: set[str],
) -> list[str]:
    warnings: list[str] = []
    if len(loader_kinds) > 1:
        warnings.append(
            "mixed loader kinds across workers: " + ", ".join(sorted(loader_kinds))
        )
    if len(resolved_backends) > 1:
        warnings.append(
            "mixed resolved backends across workers: "
            + ", ".join(sorted(resolved_backends))
        )
    if fell_back_workers:
        total = len(worker_keys) if worker_keys else len(fell_back_workers)
        warnings.append(
            f"{len(fell_back_workers)} of {total} workers fell back: "
            + ", ".join(sorted(fell_back_workers))
        )
    if disabled_capabilities:
        warnings.append(
            "disabled capabilities across the job: "
            + ", ".join(sorted(disabled_capabilities))
        )
    return warnings


def _job_identity_digest(
    *,
    training_run_ids: set[str],
    dataset_ids: set[str],
    alignment_ids: set[str],
    row_plan_ids: set[str],
    tick_plan_ids: set[str],
    epoch_plan_ids: set[str],
) -> str:
    return "trjob-" + _stable_digest(
        {
            "training_run_ids": sorted(training_run_ids),
            "dataset_ids": sorted(dataset_ids),
            "alignment_ids": sorted(alignment_ids),
            "row_plan_ids": sorted(row_plan_ids),
            "tick_plan_ids": sorted(tick_plan_ids),
            "epoch_plan_ids": sorted(epoch_plan_ids),
        }
    )


def load_report_payloads(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load report JSON files from disk for CLI merge; validates each is a mapping."""
    payloads: list[dict[str, Any]] = []
    for path in paths:
        raw = Path(path).read_text()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TrainingReportAggregationError(
                f"{path}: not valid JSON ({exc})"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise TrainingReportAggregationError(
                f"{path}: expected a JSON object, got {type(parsed).__name__}"
            )
        payloads.append(dict(parsed))
    return payloads
