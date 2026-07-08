# Journey: compare Enterprise training runs over time

**Scenario.** A platform engineer wants to compare last night's Enterprise
training run against this morning's: *did the second run's epochs hit the
plan-executor cache more often, how many bytes did each worker hydrate, and did
any worker fall back off the Enterprise path because prewarm was unavailable in
its region?*

The Enterprise training loader (backlog 0069/0073) already produces a
secret-free `TrainingLoaderReport` and a lower-level backend report for each
loader run — but those live in memory and vanish when the process exits. This
journey **persists** those reports into a durable, queryable catalog
(`training_reports`) so run history survives, and lets you slice it by snapshot,
run, backend, fallback reason, and cache policy without re-running the loader.

The catalog is a *derived* record: the in-memory `dataset.manifest.backend`
shape stays the source of truth and is unchanged. Lakes with no recorded reports
simply return an empty history.

The flow is: **record → list → reload → aggregate**.

## 1. Record a report at the end of a loader run

Hand a training dataset (or a `TrainingLoaderReport`) to the catalog. Recording
is idempotent by a content digest over the report and backend payloads, so
re-recording the same report is a no-op; two runs, epochs, or workers with
different content each get their own row.

```python
dataset = lake.training.dataset(
    "pick-place-v17", columns=["observation_id"],
    backend="enterprise", cache_policy="epoch", prewarm=True,
)
record = lake.training.record_report(dataset=dataset, training_run_id="train-2026-07-05")
print(record.report_id, record.resolved_backend, record.cache_policy)
```

Reproducibility identity (snapshot id/name, table versions, row/epoch plan ids,
epoch, worker) and the queryable backend dimensions (backend kind, connection
kind, cache policy, prewarm status, fallback reason and backend transition) are
extracted into scalar columns; the full report and backend payloads are stored
as JSON for exact reload. Resolved secrets and object-store credentials are
never persisted.

## 2. List the report history

Summary rows page deterministically (ordered by `created_at`, id) and **never
materialize the full report bodies** — the large JSON payload columns are
excluded from the listing projection.

```bash
lancedb-robotics train report list \
  --lake db://robotics --snapshot pick-place-v17 --format json
```

```python
page = lake.training.reports(snapshot="pick-place-v17", page_size=50)
for row in page.rows:
    print(row["report_id"], row["epoch"], row["cache_hits"], row["cache_misses"])
```

Filter by any dimension — for the "which worker fell back?" question, query by
fallback:

```bash
lancedb-robotics train report list --lake db://robotics --fallback --format json
```

```python
fell_back = lake.training.query_reports(fallback=True, fallback_reason="no plan-executor in region")
```

## 3. Reload one full report

When you need the complete backend report for a specific run/epoch/worker (or an
exact report id), reload it — this is the only step that reads the full payload.

```bash
lancedb-robotics train report get --lake db://robotics \
  --training-run train-2026-07-05 --epoch 1 --worker 0 --report-out report.json
```

```python
full = lake.training.get_report(training_run_id="train-2026-07-05", epoch=1, worker_id=0)
print(full.report["metrics"]["cache"], full.backend["plan_executor"])
```

## 4. Aggregate cache metrics across workers and epochs

Sum cache hits/misses, bytes read, and plan-executor fanout across all matching
report rows — this reads only the metric and grouping columns (a bounded
projection, not a payload scan) and breaks totals down per worker and per epoch
so cold vs. warm epochs are directly comparable.

```bash
lancedb-robotics train report metrics --lake db://robotics --training-run train-2026-07-05 --format json
```

```python
agg = lake.training.report_metrics(training_run_id="train-2026-07-05")
print(agg.totals, agg.by_worker, agg.by_epoch)
```

## 5. Merge raw worker reports without a catalog (distributed jobs)

Steps 1–4 assume you first *recorded* reports into the catalog. In a distributed
PyTorch, Ray, or orchestrated job, each worker emits its own `TrainingLoaderReport`
— often just written to a JSON file next to the job's output — and you want one
job-level roll-up *before* (or instead of) persisting anything. `aggregate_reports`
combines a set of per-worker reports directly, with no lake state:

```python
worker_reports = [w0.loader_report(), w1.loader_report(), w2.loader_report(), w3.loader_report()]
job = lake.training.aggregate_reports(worker_reports)      # or aggregate_training_loader_reports(...)
print(job.job_id, job.report_count, job.warnings)
job.write_json("training_loader_report.aggregate.json")
```

The CLI merges JSON files the workers wrote — useful for benchmark output:

```bash
lancedb-robotics train report merge \
  --report worker-0.json --report worker-1.json \
  --report worker-2.json --report worker-3.json \
  --out training_loader_report.aggregate.json --format json
```

The aggregate is **deterministic** (order-independent; same reports → same
`job_id` and body) and **secret-free**. It sums the **client-observable** signals
across workers — bytes read, row counts (planned/requested/coalesced/returned),
and the count of read/prewarm requests each worker issued — and keeps
`by_worker`/`by_epoch` drill-downs.

What it does **not** report matters (backlog 0345): a training worker
is a *client* of the query node over HTTP and has no access to
server-internal metrics. So the aggregate reports **only what the client itself
produces or receives** — bytes pulled, row counts, request counts, and the status
of its own prewarm submissions — and never reports cache hit/miss (warm/cold),
plan-executor fanout, a per-PE breakdown, or prewarm warmed-byte/executor counts.
Those are page-cache and plan-executor internals behind the query node, not on the
client data path, so there is no `cache` block and no `pe_fanout` at all.

**Prewarm** submissions are deduplicated by `prewarm_id`. Because the epoch prewarm
(backlog 0121) is worker-invariant, all four workers report the same `prewarm_id`;
it is counted **once** in the `prewarm` section, carrying only what the client
knows — the id, the status its own control-plane call returned, the requested
`row_count`, and which workers shared it — while `requests_submitted` still
reflects the raw per-worker count.

Heterogeneity surfaces as explicit `warnings`: mixed resolved backends, mixed
loader kinds, `"N of M workers fell back"` (with the worker ids), and disabled
capabilities. Each fallback event is tagged with the worker that emitted it. The
output is JSON-serializable and redacted, so it is compatible with future catalog
persistence.

## Retention

A report is protected while the training run it describes still exists — you
cannot GC the observability record of a live run without `force`. Reports of
expired or unknown runs are freely deletable. This reuses the shared manifest
retention machinery.

```python
plan = lake.training.report_retention_plan()          # protected vs deletable
lake.training.expire_report(report_id)                # refused if run still exists
lake.training.expire_report(report_id, force=True)    # removes it
```

## What's next (not yet built)

`reports()` and `query_reports()` filter over a scan of matching rows rather
than a pushed-down scalar index, aggregation is a Python reduction over the
matched projection, and recording is a per-report call rather than an inline
emission from the loader. Scalar indexes on the report dimensions, streaming
aggregation, and inline emission with reconciliation are filed follow-ons — see
the 0115 follow-on backlog items.
