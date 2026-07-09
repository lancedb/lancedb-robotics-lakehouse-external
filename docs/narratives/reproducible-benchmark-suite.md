# Reproducible Benchmark Suite

Backlog 0034 turns the PRD performance proof points into a checked harness
rather than a slide claim. The entry point is:

```bash
lancedb-robotics lake init ./droid-100-bench.robot.lance
lancedb-robotics bench prepare-lerobot \
  --lake ./droid-100-bench.robot.lance \
  --source lerobot/droid_100 \
  --revision <hf-revision-or-commit> \
  --out ./benchmarks/droid-100-prepare.json
```

The prepare step ingests the public LeRobot corpus, freezes a benchmark snapshot,
and records the dataset id, revision, source URI, size tier, storage tier, row
counts, and snapshot identity. The default source is `lerobot/droid_100`, the
small public DROID smoke tier; larger public LeRobot corpora can use the same
flow by changing `--source`, `--size-tier`, and `--storage-tier`.

```bash
lancedb-robotics bench run \
  --lake ./droid-100-bench.robot.lance \
  --snapshot lerobot-droid-100-benchmark \
  --formats lance,lerobot-default,lerobot-native,webdataset,deeplake \
  --source-dataset-id lerobot/droid_100 \
  --source-dataset-revision <hf-revision-or-commit> \
  --source-dataset-uri hf://lerobot/droid_100 \
  --source-size-tier droid-100 \
  --storage-tier hf-cache \
  --random-frame-samples 64 \
  --frames-per-clip 4 \
  --out ./benchmarks/droid-100-report.json
```

The report is JSON and records:

- dataset identity: lake URI, snapshot name, dataset id, split payload, and
  pinned table versions, plus the source corpus descriptor for public LeRobot
  benchmark runs;
- parameters: requested formats, sample limit, random-access probe count,
  random-frame workload shape, seed, query limit, and fixed-quality policy;
- storage tiers: the tier actually measured and explicit skip notes for local or
  object-store tiers that were not run in this invocation;
- hardware: platform, Python, CPU count, and GPU utilization when `nvidia-smi`
  is available;
- format versions and native-loader availability;
- one entry per requested format, either `completed` with metrics or `skipped`
  with an explicit reason;
- `comparison_table`, a flat row-per-format view of the metric values and units
  for copying into notebooks or benchmark dashboards.

## Retained Public LeRobot Runs

For reportable public numbers, run the retained LeRobot workflow instead of
keeping one-off local JSON files:

```bash
lancedb-robotics bench run-public-lerobot \
  --lake ./droid-100-bench.robot.lance \
  --artifact-root ./benchmarks/public/lerobot \
  --source lerobot/droid_100 \
  --revision <hf-revision-or-commit> \
  --report-id droid-100-<hf-revision-or-commit>-<git-sha> \
  --formats lance,lerobot-default,lerobot-native,webdataset,deeplake \
  --size-tier droid-100 \
  --storage-tier hf-cache
```

The command refuses unpinned HF sources. Each retained run writes:

- `runs/<report-id>/reports/prepare.json`
- `runs/<report-id>/reports/benchmark.json`
- `runs/<report-id>/reports/capacity.json`
- `runs/<report-id>/artifact-manifest.json`
- `runs/<report-id>/logs/run.log`
- `index.json`
- `dashboard.md`

`artifact-manifest.json` is the claim anchor: it records the report id, commit,
dataset id/revision/source URI, snapshot, hardware, storage tier, format status,
skipped arms, comparison table, and file inventory. Rebuild the history view from
the retained manifests without rerunning measurements:

```bash
lancedb-robotics bench public-dashboard \
  --artifact-root ./benchmarks/public/lerobot
```

External performance claims should cite the report id plus the retained manifest
path. If a GPU, object-store tier, Enterprise endpoint, or optional format was
not run, the report/dashboard keeps it as an explicit skipped entry.

Before quoting retained public LeRobot numbers externally, add an explicit
claim manifest and validate it:

```bash
lancedb-robotics bench validate-public-lerobot \
  --artifact-root ./benchmarks/public/lerobot \
  --claims ./benchmarks/public/lerobot-claims.json \
  --out ./benchmarks/public/lerobot-claim-validation.json
```

The claim manifest uses
`lancedb-robotics-public-lerobot-benchmark-claims-v0` and lists the claim id,
report id, commit, pinned dataset revision, storage tier, format, and metric.
The validator fails if the retained manifest or index is stale, if provenance is
missing, or if a skipped format is presented as a measured result.

## Metrics

Every completed non-Enterprise format emits the same five metric keys:

- `dataloader_throughput`: samples per second for bounded sequential sample
  materialization.
- `random_access_latency`: seeded random-index probe latency in ms/sample.
- `random_frame_sampling`: seeded N-frames-per-clip windows, reported as
  frames/s with latency percentiles and payload bytes materialized.
- `query_to_dataset_curation`: time to turn the benchmark query into a dataset
  artifact. For Lance this re-freezes selected scenario ids into a
  version-pinned snapshot; for LeRobot-default this materializes the projection;
  for LeRobot-native this includes the official loader open/materialization path.
- `storage_footprint`: local stored bytes when available, plus payload/video
  byte details. The fixed-quality policy is source payload bytes: no
  transcoding or quality reduction.

## Determinism

The harness uses the snapshot's pinned table versions as the source of truth.
Random access uses Python `Random(seed)`, and the curation query selects the
first `query_limit` sorted scenario ids from the base snapshot. Re-running with
the same lake, snapshot, and parameters creates the same benchmark snapshot name
and structured report shape, with only wall-clock timings and `created_at`
changing.

## Live Enterprise Endpoint Conformance and Calibration

Backlog 0074 (above) proves the `enterprise-lance` report shape against a
fake-local `db://` fixture: `_run_enterprise_lance_format` wraps the same local
Lance tables in a simulated Enterprise connection report, so CI can exercise
cache phases and filter-change measurement without any live endpoint. That
fixture cannot prove production network, query-node placement, or real cache
behavior. Backlog 0125 adds an opt-in **live mode** that runs the same harness
against an already-opened `db://` or REST Namespace lake and calibrates the two
results against each other.

```bash
lancedb-robotics bench run \
  --lake db://your-enterprise-lake \
  --snapshot lerobot-droid-100-benchmark \
  --formats enterprise-lance \
  --enterprise-live \
  --out ./benchmarks/enterprise-live-report.json
```

`--lake` must already resolve to a live Enterprise connection (`db://` remote
DB or a REST Namespace/namespace URI) -- `--enterprise-live` does not open one
for you. Combining `--enterprise-live` with `--enterprise-fixture-uri`, or
passing `--enterprise-live` against a local/non-Enterprise lake, fails fast
with a diagnostic instead of silently falling back to a skipped report entry;
see `_enterprise_benchmark_lake` in `src/lancedb_robotics/benchmark.py`.

### Report profile and confidence labeling

Every `enterprise-lance` report entry now carries a `remote_endpoint.profile`
and a `confidence` label so a report can never be misread as production
evidence:

| Profile | How it's produced | Confidence |
| --- | --- | --- |
| `fake-local-db` | `--enterprise-fixture-uri` (or `enterprise_fixture_uri=...`) | `sdk-contract-only` |
| `live-db` | `--enterprise-live` against a `db://` remote DB lake | `production-calibrated` |
| `live-namespace` | `--enterprise-live` against a REST Namespace / namespace lake | `production-calibrated` |

### Capability preflight and degraded phases

Before running any phase, the live path preflights the endpoint's capability
negotiation using the exact same matrix the training loader itself uses
(`training._enterprise_training_capabilities`), so the preflight can never
drift from what the loader will actually do. `remote_scan` is the one
capability the benchmark cannot run without at all -- if it's unavailable the
whole `enterprise-lance` format is skipped with an explicit reason, the same as
missing endpoint config today.

Every other capability (`remote_take`, `blob_or_video_remote_hydration`,
`plan_executor_cache_metrics`, `page_cache_prewarm`, `page_cache_status`) is
optional: if the endpoint doesn't support it, only the phases that depend on it
report `"status": "degraded"` with a `degraded_reasons` list, and the format
still completes. In particular, if `plan_executor_cache_metrics` is
unsupported, `phase["cache"]["hits"]`/`["misses"]` are `null` -- **unavailable,
never a synthesized zero** -- because a missing counter and a real zero are not
the same fact. This is the direct fix for the 2026-07-07 reframe at the top of
backlog 0125: a benchmark client is a query-node client and cannot observe
server-internal page-cache state, so cache metrics are either what the
endpoint explicitly reports or explicitly absent, never assumed.

Only the `fake-local-db` fixture synthesizes cache hits/misses and a prewarm
completion (to keep the CI-only report shape deterministic); `live-db` and
`live-namespace` runs always read whatever the endpoint's loader report
actually contains.

### Redaction

`remote_endpoint` only ever copies the `region`, `host_override`, and
`cache_policy` keys out of the connection's `lancedb_connect_kwargs` -- never
the raw mapping -- so an `api_key` or other credential passed to `Lake.open`
cannot reach the benchmark report even by accident. `capabilities` and
`capability_checks` are booleans/labels only. Run
`lancedb_robotics.training_report_schema.scan_report_secrets(report)` over a
saved report before sharing it externally; it independently scans every key and
string value for secret-shaped fields as a second line of defense.

### Calibrating fake-local against live

`compare_enterprise_benchmark_results(fake_report, live_report)` takes two
`run_benchmark_suite` reports (or their `formats["enterprise-lance"]` entries)
over the same logical sample set -- one `fake-local-db`, one live -- and returns
the latency/throughput deltas between them plus each side's confidence label
and any degraded capabilities on the live side. It refuses two reports of the
same profile, so a calibration result always compares SDK-contract-only numbers
against production-calibrated ones, never mixes two of the same kind:

```python
from lancedb_robotics.benchmark import compare_enterprise_benchmark_results

calibration = compare_enterprise_benchmark_results(fake_report, live_report)
```

### Minimum live setup

- The opened lake must be a `db://` remote DB or REST Namespace/namespace
  connection with `capabilities.server_side_query` enabled; there is no
  minimum row count, but a snapshot with at least a handful of scenarios is
  needed for the shuffled-epoch and random-access measurements to be
  meaningful rather than trivially fast.
- Record `LANCEDB_ROBOTICS_ENTERPRISE_INSTANCE_CLASS` in the environment
  running the benchmark to have the report's `hardware_class` field describe
  the plan-executor/query-node instance class; it is omitted (not guessed) when
  unset.
- Software versions (`lancedb-robotics`, `lancedb`, `pylance`) are always
  recorded from the local client environment, not the endpoint, since the
  client cannot introspect server-side versions.

## Comparison Formats

The Lance path uses `lake.training.dataset(...)` directly over the pinned
snapshot. The LeRobot-default comparison runs end to end through the existing
dependency-light projection writer and reads the generated parquet rows
directly, while still recording whether the native `lerobot` package is
available.

The LeRobot-native comparison is opt-in via `--formats lerobot-native` or the
alias `lerobot-v3`. It opens the benchmark source or deterministic projection
through the official LeRobotDataset API and measures the same throughput,
random-access, random-frame, curation, and storage metric keys when the optional
LeRobot/Torch/decode stack is installed. The supported install policy is the
single `lerobot-native-bench` extra (`uv sync --extra dev --extra
lerobot-native-bench` for repo work, or `pip install
'lancedb-robotics[lerobot-native-bench]'` for package installs). It declares
LeRobot, torch, NumPy, and PyAV. TorchCodec is probed first when a host or
LeRobot install provides it, but it is not a direct dependency; PyAV is the
stable declared decode backend for CPU CI and portable local runs. Missing
`lerobot`, `torch`, TorchCodec, or PyAV support is reported as a skipped arm with
per-metric skip entries, so the dependency-light `lerobot-default` baseline
remains the stable CI path.

The dedicated `lerobot_native_benchmark` CI lane installs the native stack and
sets `LANCEDB_ROBOTICS_REQUIRE_LEROBOT_NATIVE_BENCHMARK=1`. That converts
dependency, import-path, DataLoader, sample-access, decode-backend, and metric
conformance skips into failures. The lane creates a tiny local v3 dataset with
the official `LeRobotDataset.create` API, reopens it through the benchmark
`lerobot-native` arm, and uploads dependency/version/decode diagnostics. Set
`LANCEDB_ROBOTICS_LEROBOT_NATIVE_PUBLIC_REPO_ID` plus
`LANCEDB_ROBOTICS_LEROBOT_NATIVE_PUBLIC_REVISION` to add a pinned public-source
smoke fixture when network and credentials allow.

Native source resolution is explicit in the report. `lerobot-native` records the
resolver `source_mode` (`auto`, `source`, or `projection`), `cache_mode` (`auto`,
`cache-only`, or `download`), whether a local root/prepared cache was used, and a
preflight block with sample budgets and expected implicit Hub download behavior.
Use `--lerobot-native-source-mode source --lerobot-native-cache-mode cache-only`
when a scheduled run must prove it used a prepared local root or cache instead of
downloading from the Hub. Use `--lerobot-native-source-mode projection` to force
the deterministic projection fallback even when a source descriptor is present.
`--lerobot-native-episode-limit N` passes the first N episodes to the official
loader when the selected LeRobotDataset API supports episode filters.

WebDataset is materialized through deterministic tar shards when requested. Deep
Lake is represented as an explicit skipped entry until its native adapter lands.
A skipped entry is intentional evidence in the report: it means the format or
storage tier was requested or expected and not covered, not silently omitted.
