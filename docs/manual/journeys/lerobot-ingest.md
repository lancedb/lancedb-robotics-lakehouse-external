# Journey: bring a LeRobot dataset into the lake

**Scenario.** You already have policy-training data in LeRobot format, either as
a local directory or as a Hugging Face dataset repo. Ingest it once, then treat it
as first-class lake data: episodes, frame observations, video references, lineage,
snapshots, and LeRobot projection/export all point back to the same source.

## 1. Inspect the dataset

Check the declared LeRobot version, task list, camera keys, fps, frame counts,
and per-camera video metadata without writing anything:

```bash
lancedb-robotics inspect lerobot ./koch_pick_place --format text
```

The inspect report refuses unsupported `codebase_version` values instead of
guessing. HF Hub repo ids are accepted when the optional LeRobot/Hugging Face
stack is installed:

```bash
lancedb-robotics inspect lerobot lerobot/aloha_static_coffee --format json
```

The JSON report also includes `source_identity` and per-data-file row-group
stats, so large dataset planning can see which Parquet chunks and fingerprints
will drive ingest before any lake writes occur. When MP4 camera streams are
readable, a bounded media inspector reads MP4 box/sample-table metadata without
staging or decoding the media payload. `video_files` includes codec, codec
tag/profile, resolution, fps, frame count, GOP size, keyframe map, inspection
status, metadata bytes read, parse duration, and retry reuse fields. The
top-level `media_inspection` object summarizes counts by status, codec, and
diagnostic code. Missing, corrupt, mismatched, or unsupported video files appear
as typed diagnostics while the structured frame rows remain inspectable.

Object-store LeRobot roots are inspected in place. The adapter supports
`s3://`, `gs://` / `gcs://`, `az://`, `abfs://`, and `abfss://` roots through
the matching fsspec backend (`s3fs`, `gcsfs`, or `adlfs`), so a bucket prefix can
be planned without staging every Parquet or MP4 file locally:

```bash
lancedb-robotics inspect lerobot s3://robotics-raw/koch_pick_place \
  --auth-ref raw-lerobot \
  --storage-option region=us-west-2 \
  --format json
```

Python callers pass the same raw-source storage options directly:

```python
from lancedb_robotics.adapters import get_adapter

report = get_adapter("lerobot").inspect(
    "s3://robotics-raw/koch_pick_place",
    auth_ref="raw-lerobot",
    storage_options={"region": "us-west-2"},
)
```

The resolved credentials are used only while reading source metadata and media.
The lake stores the object URI and `auth_ref`, not the underlying secret. For
remote media, `video_files[].uri` remains the original object URI and
`object_metadata` records provider fingerprints such as etag, version id,
generation, size, and last-modified time when the backend exposes them.

Before a large object-store ingest, run the provider conformance matrix from the
same host or CI worker that will read the raw corpus. It verifies storage option
resolution, stat/list/read/open behavior, LeRobot inspect, and the streaming
preflight path that ingest uses, without writing lake rows:

```bash
lancedb-robotics inspect lerobot-object-store-conformance \
  --root s3://robotics-raw/koch_pick_place \
  --auth-ref raw-lerobot \
  --storage-option region=us-west-2 \
  --format text \
  --strict
```

The report emits one case per provider/auth mode (`explicit-options`,
`auth-ref-env`, and provider defaults unless `--no-provider-default` is used).
JSON output includes only option key names, auth refs, operation status, and
provider metadata fields; resolved credential values are never printed. Use this
as a cheap preflight when moving the same LeRobot dataset between local MinIO,
AWS S3, GCS, Azure Blob/Data Lake, and managed CI credentials.

Raw-source provider credential notes:

- S3 or S3-compatible (`s3://`): install `s3fs`. Use AWS environment/config,
  IAM, explicit options such as `region`, `endpoint_url`, `profile`, `key`,
  `secret`, or `token`, or
  `LANCEDB_ROBOTICS_AUTH_<AUTH_REF>_STORAGE_OPTIONS_JSON`.
- Google Cloud Storage (`gs://` or `gcs://`): install `gcsfs`. Use Application
  Default Credentials, workload identity, or explicit/auth-ref JSON options such
  as `token` and `project`.
- Azure Blob/Data Lake (`az://`, `abfs://`, or `abfss://`): install `adlfs`.
  Use Azure environment variables, managed identity, or explicit/auth-ref JSON
  options such as `account_name`, `credential`, `connection_string`,
  `tenant_id`, and `client_id`.

Large object-store LeRobot roots can contain many Parquet, metadata, and media
objects. When a prefix is stable during a backfill, use a source manifest cache
to list it once, reuse the listing across repeated inspect/ingest steps, and
avoid repeated recursive provider globs:

```bash
lancedb-robotics inspect lerobot s3://robotics-raw/koch_pick_place \
  --auth-ref raw-lerobot \
  --source-manifest-cache ./cache/koch_pick_place.manifest.json \
  --format json

lancedb-robotics ingest lerobot s3://robotics-raw/koch_pick_place \
  --lake ./robot.lance \
  --source-auth-ref raw-lerobot \
  --source-manifest-cache ./cache/koch_pick_place.manifest.json
```

The cache records object URIs, provider metadata fingerprints, option key names,
listing metrics, and cache hit/miss state; it does not persist resolved
credential values. Cached listings are revalidated against provider metadata
before reuse and rebuilt when relevant metadata changes. Omit
`--source-manifest-cache` for highly mutable raw buckets or when a provider does
not expose stable size/etag/version/generation/last-modified metadata.

Object-store source identity defaults to `metadata-only`, which preserves the
bounded behavior used by earlier LeRobot ingests: the adapter fingerprints
provider metadata instead of hashing large media or Parquet objects. For higher
assurance roots, select a stronger policy before inspect or ingest:

```bash
lancedb-robotics inspect lerobot s3://robotics-raw/koch_pick_place \
  --auth-ref raw-lerobot \
  --source-manifest-cache ./cache/koch_pick_place.manifest.json \
  --source-validation-policy sampled-validation \
  --source-validation-sample-count 16 \
  --source-validation-sample-bytes 4096 \
  --format json

lancedb-robotics ingest lerobot s3://robotics-raw/koch_pick_place \
  --lake ./robot.lance \
  --source-auth-ref raw-lerobot \
  --source-manifest-cache ./cache/koch_pick_place.manifest.json \
  --source-validation-policy sampled-validation
```

`sampled-validation` deterministically chooses objects from the source manifest
and records bounded byte-range hashes in `source_identity`; this can detect
sampled content changes even when object-store metadata is weak. Use
`strict-content-hash` for small or regulated corpora where reading every source
object is acceptable, and bound the cost with `--strict-content-hash-max-bytes`.
The selected policy, assurance level, sample evidence digest, warning count, and
any weak-provider-metadata warnings are recorded in inspect JSON, ingest
checkpoints, transform params, and source/run metadata. Resolved storage options
and credential values are never written to those artifacts.

For object-store corpora, tune the media inspector separately from frame ingest
so a slow or wedged MP4 object does not block the whole dataset. `inspect
lerobot` and `ingest lerobot` accept the same per-video policy:

```bash
lancedb-robotics inspect lerobot s3://robotics-raw/koch_pick_place \
  --auth-ref raw-lerobot \
  --media-inspection-workers 4 \
  --media-inspection-timeout-seconds 30 \
  --media-inspection-execution-mode process \
  --media-inspection-retries 2 \
  --media-inspection-retry-backoff-seconds 1 \
  --format json
```

```python
report = get_adapter("lerobot").inspect(
    "s3://robotics-raw/koch_pick_place",
    auth_ref="raw-lerobot",
    media_inspection_workers=4,
    media_inspection_timeout_seconds=30,
    media_inspection_execution_mode="process",
    media_inspection_retries=2,
    media_inspection_retry_backoff_seconds=1,
)
```

Timed-out media produces a `timeout-video` diagnostic and an
`inspection_status` of `timeout` for that video. Transient read failures are
retried before becoming `corrupt-video` diagnostics. In both cases, frame-row
inspect/ingest continues; the media report records `total_attempts`,
`total_retries`, `total_timeouts`, final error class, and per-attempt errors so
operators can distinguish a bad file from a slow object store.

The default media-inspection execution mode is `thread`, which has the lowest
overhead and is usually right for local files and well-behaved object-store
clients. A thread timeout records `timeout-video` and lets the dataset continue,
but Python cannot forcibly stop a wedged thread. Use
`--media-inspection-execution-mode process` for hostile networks or storage
gateways: each video read runs in an isolated child process, the scheduler keeps
at most `--media-inspection-workers` children active, and timed-out children are
terminated before the next video is scheduled. Reports add `execution_mode`,
`killed_worker_count`, and per-video `inspection_worker_killed` so operators can
see when hard kills were required. Process mode prefers the `spawn`
multiprocessing start method; set
`LANCEDB_ROBOTICS_LEROBOT_MEDIA_INSPECTION_START_METHOD` only when a platform or
test harness requires another supported method.

## 2. Ingest into canonical rows

Initialize or open a lake, then ingest the dataset:

```bash
lancedb-robotics lake init ./robot.lance
lancedb-robotics ingest lerobot ./koch_pick_place --lake ./robot.lance
```

```python
from lancedb_robotics.ingest import ingest_lerobot
from lancedb_robotics.lake import Lake

lake = Lake.init("./robot.lance")
report = ingest_lerobot(lake, "./koch_pick_place")
```

For an object-store LeRobot root, give source credentials separately from lake
storage credentials when those planes differ:

```bash
lancedb-robotics ingest lerobot s3://robotics-raw/koch_pick_place \
  --lake s3://robotics-lakes/policies.robot.lance \
  --storage-auth-ref lake-writer \
  --source-auth-ref raw-lerobot \
  --source-storage-option region=us-west-2
```

```python
report = ingest_lerobot(
    lake,
    "s3://robotics-raw/koch_pick_place",
    auth_ref="raw-lerobot",
    storage_options={"region": "us-west-2"},
    media_inspection_timeout_seconds=30,
    media_inspection_retries=2,
    media_inspection_retry_backoff_seconds=1,
)
```

For pixel-level decoder checks, opt into the optional conformance lane and
install `lancedb-robotics[video-decode]` on the host that will decode source
MP4s. The base install still performs metadata and sample-byte checks without
PyAV, OpenCV, or ffmpeg imports.

```python
report = ingest_lerobot(
    lake,
    "./koch_pick_place",
    decoded_frame_conformance={
        "enabled": True,
        "backend": "pyav",
        "samples": [
            {
                "camera_key": "front",
                "episode_index": 0,
                "frame_index": 42,
                "expected_pixel_sha256": "<expected-rgb-frame-sha256>",
            }
        ],
    },
)
```

The ingest writes:

- one `runs` row for the LeRobot dataset source;
- one `episodes` row per authored LeRobot episode;
- one `observations` row per frame, with `observation.state`, `action`, `task`,
  `episode_index`, `frame_index`, timestamps, and unmapped feature payloads;
- one authored `scenarios` row per episode so the data can be searched,
  snapshotted, curated, and exported through existing flows;
- `videos` / `video_encodings` references for per-camera MP4 streams, with
  source MP4 sample-table keyframe maps and no re-encoding or video-byte copy.

Small source-MP4 keyframe maps stay inline in `video_encodings.keyframe_map_json`
for simple local workflows. Long clips or dense multi-camera episodes can
produce large maps, so LeRobot ingest offloads maps whose canonical JSON exceeds
64 KiB or 4,096 frames into the content-addressed `keyframe_map_artifacts`
catalog. `video_encodings.keyframe_map_ref` always carries the stable
`sha256:<digest>` ref; offloaded rows leave `keyframe_map_json` empty and point
at the catalog row instead. Tune the thresholds when needed:

```bash
lancedb-robotics ingest lerobot ./long_horizon_policy \
  --lake ./robot.lance \
  --keyframe-map-inline-threshold-bytes 131072 \
  --keyframe-map-inline-threshold-frames 8192
```

```python
report = ingest_lerobot(
    lake,
    "./long_horizon_policy",
    keyframe_map_inline_threshold_bytes=131_072,
    keyframe_map_inline_threshold_frames=8_192,
)
```

Checkpoint progress and transform params record compact artifact refs, JSON
sizes, frame counts, and source video fingerprints instead of repeating the map
body. That keeps long-clip job histories scan-friendly while preserving a
single deduplicated map body per content hash.

Frame rows are streamed from LeRobot Parquet files by row group and record batch.
If an ingest is interrupted after some `observations` were written, a retry skips
the already-present deterministic `run:episode:frame` observation ids and
continues without duplicating rows. Completed media inspections are reused on
retry when the video URI, size, mtime, and expected frame count are unchanged, so
a retry does not reread already-validated MP4 metadata. Each attempt also writes
an append-only `lerobot_ingest_checkpoints` record at claim, media-inspection,
frame-batch, metadata-ready, failed, and completed phases, so operators can
inspect the last durable checkpoint before the final `transform_runs` row exists.
HF Hub sources keep both the requested revision and the resolved snapshot
revision/cache path in that checkpoint ledger; local directories hash manifest
and Parquet files while large media contributes stat fingerprints instead of
full payload checksums. Object-store sources use backend metadata fingerprints
where possible, preferring etag/version/generation style identifiers over local
mtime-style fingerprints. The resulting `videos.raw_uri` / `videos.uri` values
stay as `s3://`, `gs://`, or Azure object URIs so later MP4 sample reads can
hydrate exactly the original source object.

Recommended media-inspection timeout defaults:

- Local smoke tests: leave timeout disabled and retries at `0` to surface bad
  fixtures quickly.
- CI object-store fixtures: `--media-inspection-timeout-seconds 10` and
  `--media-inspection-retries 1`.
- Large remote corpora: start with `30` to `60` seconds and `2` retries, then
  inspect `media_inspection.total_timeouts` and
  `media_inspection.total_retries` in checkpoint progress before widening.
- Provider/client socket timeouts still live in `storage_options` /
  `--source-storage-option` when the fsspec backend supports them; the media
  timeout is the outer per-video wall-clock guard.

After a backfill has written durable checkpoints, ask the lake to recommend the
next timeout and retry settings instead of widening by feel:

```bash
lancedb-robotics ingest lerobot-media-inspection-timeout-plan \
  --lake ./robot.lance \
  --format text
```

JSON mode is suitable for automation:

```bash
lancedb-robotics ingest lerobot-media-inspection-timeout-plan \
  --lake ./robot.lance \
  --source-id src-abc123 \
  --format json
```

The report reads only `lerobot_ingest_checkpoints.progress_json` plus completed
LeRobot `transform_runs.params`. It summarizes configured timeout/retry values,
`total_timeouts`, `total_retries`, process worker kills, and per-video
`inspection_duration_ms` p50/p95/p99 distributions, then emits `apply_args` such
as `--media-inspection-timeout-seconds 40 --media-inspection-retries 2`. Reused
media-inspection cache hits are counted but excluded from duration percentiles.
Recommendations are also grouped by storage tier, provider, and corpus-size lane
so a local smoke corpus does not hide an S3 mid-corpus timeout problem.

For an incident review or CI fixture that has exported checkpoint rows but no
openable lake, run the same planner offline:

```bash
lancedb-robotics ingest lerobot-media-inspection-timeout-plan \
  --checkpoint-rows-json ./reports/lerobot-checkpoint-rows.json \
  --format json
```

Use the lanes this way:

- Local lane: keep timeout disabled or near `1` second and retries at `0` unless
  fixture MP4 metadata reads are intentionally slow.
- CI lane: run the planner after object-store fixtures and fail only on repeated
  `timeout-policy-too-aggressive` recommendations.
- Mid-corpus lane: apply the planner's `apply_args` between retry windows, then
  compare the next report's p95/p99 and timeout counts.
- Full-corpus lane: filter by `--source-id`, `--storage-tier`, or `--provider`
  before applying a global setting so one storage backend does not overfit the
  whole run.

Recent LeRobot jobs are listable without scanning frame rows:

```bash
lancedb-robotics ingest lerobot-jobs --lake ./robot.lance --format text
lancedb-robotics ingest lerobot-job <job-id> --lake ./robot.lance --format json
```

Running jobs carry claim lease metadata in `progress.claim`: owner, token,
generation, heartbeat interval, last heartbeat, and `claim_expires_at`. Normal
ingest extends that lease whenever it appends a durable running checkpoint. A
second worker is refused while the latest checkpoint is still `running`, even if
the recorded lease looks expired, because takeover is an operator decision rather
than an automatic side effect of retrying ingest.

Run the dry-run watchdog from cron, CI, or a backfill operator console to find
claims that need human review before a readiness window slips:

```bash
lancedb-robotics ingest lerobot-claim-watchdog \
  --lake ./robot.lance \
  --stale-after-seconds 21600 \
  --out-json ./reports/lerobot-stale-claims.json \
  --out-markdown ./reports/lerobot-stale-claims.md \
  --fail-on-stale \
  --format json
```

The watchdog reads only `lerobot_ingest_checkpoints`. It reports stale running
claims, live running claims, and inactive latest jobs separately, including job
id, source id, previous owner/token, last heartbeat, expiration time, row
progress, and a suggested `lerobot-claim-recover` command. `--fail-on-stale`
still writes the JSON/Markdown reports before exiting nonzero for CI. Use
shorter fallback windows for disposable CI lakes and the default six-hour window
for large HF backfills unless the ingest workers are configured with a different
lease.

Use explicit claim recovery after confirming the prior worker is gone:

```bash
lancedb-robotics ingest lerobot-claim-recover <job-id> \
  --lake ./robot.lance \
  --action abandon \
  --new-owner backfill-worker-2 \
  --expected-latest-checkpoint-id <checkpoint-id-from-watchdog> \
  --expected-latest-claim-token <claim-token-from-watchdog> \
  --expected-checkpoint-index <checkpoint-index-from-watchdog> \
  --stale-after-seconds 21600 \
  --format json
```

Recovery appends an auditable `abandoned` checkpoint with
`progress.claim_recovery.previous_owner`, previous token, new owner/token,
staleness inputs, and recovery time. A later `ingest lerobot` run for the same
source can then claim the job and resume using deterministic observation ids, so
already-written frames are skipped instead of duplicated. Use `--action steal`
when the handoff should explicitly say the next owner took over the stale claim;
it still records a recovery checkpoint before the next ingest writes a fresh
running claim.

Use the `--expected-*` flags emitted by the watchdog whenever recovery is based
on a previously observed stale row. They are optimistic compare-and-swap guards:
the command verifies that the latest checkpoint id, claim token, and checkpoint
index still match before appending recovery. If another operator or scheduler
already wrote a newer checkpoint, recovery fails with the winning latest
checkpoint details and tells you to inspect `lerobot-job` before retrying.

The same guard is available on `ingest lerobot` for post-recovery retries:

```bash
lancedb-robotics ingest lerobot ./koch_pick_place \
  --lake ./robot.lance \
  --expected-latest-checkpoint-id <abandoned-checkpoint-id> \
  --expected-latest-claim-token <recovery-claim-token> \
  --expected-checkpoint-index <abandoned-checkpoint-index>
```

This is a portable optimistic consistency model for local paths, object stores,
and remote DB lakes; it does not assume unsupported cross-process transactions
or locks. The guard prevents stale operators from acting on a row they no longer
observe as latest, while LanceDB still owns the actual append commit.

Before a long backfill, simulate claim recovery chaos against either the current
checkpoint catalog or synthetic corpus assumptions:

```bash
lancedb-robotics ingest lerobot-claim-recovery-simulate \
  --scenario mid-corpus \
  --synthetic-sources 4 \
  --synthetic-completed-jobs-per-source 8 \
  --synthetic-failed-jobs-per-source 1 \
  --synthetic-running-jobs-per-source 3 \
  --synthetic-checkpoints-per-job 12 \
  --synthetic-stale-running-fraction 0.25 \
  --synthetic-missing-lease-fraction 0.05 \
  --source-size-frames 250000 \
  --batch-size 2048 \
  --retry-owner-count 2 \
  --format json

lancedb-robotics ingest lerobot-claim-recovery-simulate \
  --lake ./robot.lance \
  --scenario full-public-corpus \
  --source-id <source-id> \
  --format json
```

The simulator does not mutate the lake. It models crashes before claim, after
claim, during media inspection, during frame batches, and after metadata-ready
checkpoints. The report includes stale/live/inactive claim counts, CAS conflict
estimates when multiple operators recover the same observed row, recovery
latency, checkpoint-row growth, duplicate-protection checks for observations,
episodes, videos, events, runs, and transform rows, and recommended
lease/heartbeat/watchdog defaults for the selected profile. Use it before
scheduling the watchdog so local smoke tests, CI lakes, mid-corpus runs, and
full public-corpus backfills start with realistic claim windows.

High-volume HF backfills can append one checkpoint row per inspected media phase
and frame batch. Before a multi-day run, use the non-mutating scale planner to
compare retention settings against either synthetic corpus assumptions or the
checkpoint rows already in a lake:

```bash
lancedb-robotics ingest lerobot-checkpoint-retention-plan \
  --scenario full-public-corpus \
  --synthetic-sources 12 \
  --synthetic-completed-jobs-per-source 40 \
  --synthetic-failed-jobs-per-source 4 \
  --synthetic-running-jobs-per-source 1 \
  --synthetic-checkpoints-per-job 24 \
  --format json

lancedb-robotics ingest lerobot-checkpoint-retention-plan \
  --lake ./robot.lance \
  --scenario mid-corpus \
  --source-id <source-id> \
  --format json
```

The plan never mutates checkpoint rows. It compares `local-smoke`,
`ci-disposable`, `mid-corpus`, `full-public-corpus`, and `audit-window` policies
and reports projected row deletion, row/version/fragment deltas, protected-job
counts by reason, hold-protected jobs for observed lakes, and the recommended
policy for the selected scenario. Use it to choose retention settings before
enabling scheduled backfill maintenance.

Use checkpoint retention to preview or apply a safe summary pass:

```bash
lancedb-robotics ingest lerobot-checkpoint-retention \
  --lake ./robot.lance \
  --older-than-days 30 \
  --retain-completed-per-source 10 \
  --retain-failed-per-source 10 \
  --format json

lancedb-robotics ingest lerobot-checkpoint-retention \
  --lake ./robot.lance \
  --apply
```

The policy keeps running jobs fully expanded, keeps recent terminal jobs fully
expanded, and keeps the newest completed/failed histories per source. Older
terminal histories are reduced to the latest terminal checkpoint row, which still
contains the terminal status/error, final row counts, HF requested/resolved
revision, cache path, source identity, manifest fingerprints, and progress JSON.
Checkpoint ids referenced by active lineage retention holds are preserved. For
operator-owned governance windows, prefer catalog-backed LeRobot checkpoint
holds so the policy can preserve selected jobs without hand-creating lineage
artifacts.

```bash
lancedb-robotics ingest lerobot-checkpoint-hold \
  --lake ./robot.lance \
  --job-id <job-id> \
  --legal-hold \
  --owner governance \
  --reason "litigation hold" \
  --format json

lancedb-robotics ingest lerobot-checkpoint-hold \
  --lake ./robot.lance \
  --source-id <source-id> \
  --hf-repo-id lerobot/droid_100 \
  --status completed \
  --updated-after 2026-01-01T00:00:00Z \
  --reason "audit sample"

lancedb-robotics ingest lerobot-checkpoint-release-hold <hold-id> \
  --lake ./robot.lance \
  --released-by governance
```

Holds are stored in `lerobot_checkpoint_holds` and can select checkpoint rows by
checkpoint id, job id, source id, HF repo/revision, terminal status, and updated
time window. A hold can be a legal, audit, promotion, or time-limited retention
rule. Active holds are re-resolved during retention, and retention reports mark
protected jobs with `reason="retention-hold"` plus the hold ids and reasons that
kept those rows expanded. Releasing a hold marks it inactive while preserving the
hold record for audit.

For recurring backfills, run the scheduler-friendly hook from cron, systemd, a
Kubernetes CronJob, or benchmark automation. It executes one retention pass and
emits dashboard-ready telemetry; it does not sleep or own the scheduling loop:

```bash
lancedb-robotics ingest lerobot-checkpoint-retention-schedule \
  --lake ./robot.lance \
  --config-json '{
    "schedule_id": "nightly-hf-backfill-retention",
    "every_minutes": 1440,
    "older_than_days": 30,
    "retain_completed_per_source": 10,
    "retain_failed_per_source": 10,
    "dry_run": true,
    "max_rows": 500000,
    "max_rows_per_source": 50000,
    "max_version_delta": 25
  }' \
  --format json
```

The schedule report wraps the normal retention report with `schedule_id`,
`started_at`, `finished_at`, `next_run_after`, threshold settings, per-source
row/job telemetry, reason counts, hold-protected job counts, version and fragment
deltas, and warning alerts for row-count or version-growth thresholds. Keep
`dry_run` true during audit windows; add `--apply` only for automation that is
allowed to delete summarized checkpoint rows. Threshold alerts are emitted in
the report while the command still exits 0 unless config parsing or retention
execution fails; route those alerts through the scheduler or monitoring policy
that owns paging.

`lake maintain` runs the same retention pass before compacting and cleaning the
`lerobot_ingest_checkpoints` table:

```bash
lancedb-robotics lake maintain --lake ./robot.lance
lancedb-robotics lake maintain --lake ./robot.lance --no-lerobot-checkpoint-retention
```

Recommended defaults:

- Local smoke tests: `--older-than-days 0 --retain-completed-per-source 1`.
- CI: dry-run first with `--format json`; apply with `--older-than-days 0` only
  on disposable lakes.
- Large HF backfills: keep the defaults, then run `lake maintain` after each
  corpus tranche.
- Audit windows: place a checkpoint hold, then run retention normally. Use
  `--no-lerobot-checkpoint-retention` only as a coarse maintenance pause.

Python callers can use the same planner:

```python
from datetime import timedelta
from lancedb_robotics.ingest import (
    apply_lerobot_checkpoint_retention,
    hold_lerobot_checkpoints,
    plan_lerobot_checkpoint_retention_scale,
    release_lerobot_checkpoint_hold,
    run_lerobot_checkpoint_retention_schedule,
    simulate_lerobot_claim_recovery_chaos,
    watch_lerobot_ingest_claims,
)

watchdog = watch_lerobot_ingest_claims(
    lake,
    stale_after=timedelta(hours=6),
    recovery_action="abandon",
    new_owner="backfill-worker-2",
)
for finding in watchdog.stale_claims:
    print(finding.job_id, finding.suggested_recovery_command)

chaos = simulate_lerobot_claim_recovery_chaos(
    lake,
    scenario="mid-corpus",
    source_size_frames=250_000,
    batch_size=2048,
    retry_owner_count=2,
    stale_after=timedelta(hours=6),
)
print(chaos.passed, chaos.recommendations["lease_timeout_seconds"])

plan = plan_lerobot_checkpoint_retention_scale(
    scenario="full-public-corpus",
    synthetic_sources=12,
    synthetic_completed_jobs_per_source=40,
    synthetic_failed_jobs_per_source=4,
    synthetic_running_jobs_per_source=1,
    synthetic_checkpoints_per_job=24,
)
print(plan.recommended_policy, plan.policies[-1].rows_after)

checkpoint_rows = lake.table("lerobot_ingest_checkpoints").to_arrow().to_pylist()
row_plan = plan_lerobot_checkpoint_retention_scale(checkpoint_rows=checkpoint_rows)

hold = hold_lerobot_checkpoints(
    lake,
    job_id="lerobot-job-2026-01-15",
    legal_hold=True,
    owner="governance",
    reason="litigation hold",
)

report = apply_lerobot_checkpoint_retention(
    lake,
    older_than=timedelta(days=30),
    retain_completed_per_source=10,
    retain_failed_per_source=10,
    dry_run=True,
)
print(report.rows_before, report.rows_after, report.jobs[0].hold_reasons)

scheduled = run_lerobot_checkpoint_retention_schedule(
    lake,
    schedule_id="nightly-hf-backfill-retention",
    interval=timedelta(days=1),
    older_than=timedelta(days=30),
    max_rows=500000,
    max_rows_per_source=50000,
    dry_run=True,
)
print(scheduled.telemetry["rows_deleted"], scheduled.alerts)

release_lerobot_checkpoint_hold(lake, hold.hold_id, released_by="governance")
```

For readable LeRobot MP4s, the keyframe map resolves policy frame indices to
source MP4 sample byte ranges. Small clips carry that map inline in
`video_encodings.keyframe_map_json`; offloaded clips carry the same content
behind `video_encodings.keyframe_map_ref` in `keyframe_map_artifacts`.
`lake.video.seek(...)` resolves either form and fetches the exact source sample
bytes for a frame. A later explicit video encoding job can still materialize
Lance-native GOP blobs when a training policy opts into that storage tradeoff.

When decoded pixels matter, for example validating a new decoder stack before a
training run, `lake.video.conform_source(...)` reads selected source MP4 frames
through an optional decoder backend and records a `video-conformance` transform:

```python
conformance = lake.video.conform_source(
    [
        {
            "camera_key": "front",
            "episode_index": 0,
            "frame_index": 42,
            "expected_sha256": "<expected-rgb-frame-sha256>",
        }
    ],
    decoder="pyav",
)
```

The same source-frame checks are available from the CLI for benchmark scripts
and operations runs. `--samples` accepts either a JSON object with `samples`, a
JSON array, a single JSON sample object, or newline-delimited JSON rows:

```bash
lancedb-robotics video conform-source \
  --lake ./koch.lance \
  --samples source-frame-samples.jsonl \
  --decoder pyav \
  --format json \
  --fail-on-mismatch
```

The conformance report records status counts, codec coverage, backend versions,
decoded frame hashes, keyframe/GOP context, and per-frame failures. Missing
decoder dependencies are reported as skipped or unsupported results with an
install hint instead of making ordinary LeRobot ingest depend on media stacks.
When LeRobot MP4 inspection recorded keyframe time-base metadata, the built-in
PyAV backend seeks to the nearest keyframe before decoding the target frame and
reports `seek_strategy`, `seek_frame_index`, `frames_decoded`, and any
`fallback_reason`. If older inline keyframe maps or unusual containers lack
timestamp metadata, the backend records an explicit sequential fallback instead
of silently reporting a fast seek.
The project CI lane that proves this path installs `dev + video-decode` and runs
`LANCEDB_ROBOTICS_REQUIRE_VIDEO_DECODE=1 pytest -m video_decode -v`; with that
flag set, missing PyAV/NumPy, unavailable required fixture encoders, decoder
import failures, or RGB hash mismatches fail the lane instead of producing an
all-skipped pass. The lane uploads a `video-decode-conformance` artifact with
the ingest/source conformance JSON, backend versions, codec coverage, decoded
RGB hashes, and optional codec summaries. H.264, H.265, and AV1 fixtures run
when the host PyAV build advertises matching encoders; unsupported optional
encoders remain explicit skips.

## 3. Snapshot and project back to LeRobot

After ingest, episode scenarios can become a reproducible training slice:

```python
from lancedb_robotics.dataset import SPLIT_BY_SCENARIO, create_snapshot

scenario_ids = [row["scenario_id"] for row in lake.table("scenarios").to_arrow().to_pylist()]
create_snapshot(
    lake,
    name="koch-pick-place-v1",
    scenario_ids=scenario_ids,
    split_by=SPLIT_BY_SCENARIO,
)

projection = lake.projections.lerobot.export("koch-pick-place-v1", out="./koch-pick-place-export")
```

```bash
lancedb-robotics dataset export lerobot koch-pick-place-v1 \
  --lake ./robot.lance --out ./koch-pick-place-export
```

## 4. Prepare a public benchmark corpus

To compare Lance-native training access against the LeRobot-default projection on
a realistic public corpus, prepare the default DROID-100 smoke tier:

```bash
lancedb-robotics lake init ./droid-100-bench.robot.lance
lancedb-robotics bench prepare-lerobot \
  --lake ./droid-100-bench.robot.lance \
  --source lerobot/droid_100 \
  --revision <hf-revision-or-commit> \
  --out ./benchmarks/droid-100-prepare.json
```

The prepare report records the source dataset id, revision, source URI, size
tier, storage tier, ingest run, snapshot name, scenario count, and rows added.
Use those descriptor fields in the benchmark run:

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

The JSON report includes a `comparison_table`, source-corpus metadata, hardware
and library versions, the measured storage tier, explicit local/object-store skip
notes when a tier was not run, and a `random_frame_sampling` metric for
N-frames-per-clip access.

Use `lerobot-default` for the dependency-light projection baseline. Add
`lerobot-native` or `lerobot-v3` when you want the official LeRobotDataset v3
loader path measured too. That arm is optional: it completes when LeRobot, torch,
and a video decode backend are importable, and otherwise appears as an explicit
skipped format with native version/decode metadata.

The supported install target for that native benchmark arm is the dedicated
`lerobot-native-bench` extra:

```bash
uv sync --extra dev --extra lerobot-native-bench
```

For package installs, use `pip install 'lancedb-robotics[lerobot-native-bench]'`.
The extra keeps the base install light and declares LeRobot, torch, NumPy, and
PyAV. TorchCodec is intentionally not a direct dependency: the benchmark probes it
first when the host or LeRobot stack provides it, then falls back to PyAV. Read
native benchmark numbers with that in mind: `native_loader.dependency_status`
records `decode_backend.selected`, backend versions, and the install policy used
to interpret decode behavior.

Platform notes: the native benchmark lane is proven on Linux/Python 3.11 with CPU
torch wheels (`UV_TORCH_BACKEND=cpu`). macOS and newer Python versions can still
use the extra for local development, but native media libraries may fail at import
time; those failures are reported as explicit skipped benchmark formats unless
the required-lane environment flag is set.

Run the native CI lane locally when you want the official loader checks to be
hard failures instead of dependency-light skips:

```bash
uv sync --extra dev --extra lerobot-native-bench
LANCEDB_ROBOTICS_REQUIRE_LEROBOT_NATIVE_BENCHMARK=1 \
LANCEDB_ROBOTICS_LEROBOT_NATIVE_ARTIFACT_DIR=artifacts/lerobot-native-benchmark \
uv run --no-sync pytest -m lerobot_native_benchmark -v
```

That lane creates a tiny local v3 fixture with the official `LeRobotDataset`
writer, reopens it through `lerobot-native`, exercises DataLoader iteration,
random sample access, random-frame metric conformance, and records the resolved
LeRobot/Torch/decode versions as artifacts. To add a public-source smoke fixture
to the same lane, set `LANCEDB_ROBOTICS_LEROBOT_NATIVE_PUBLIC_REPO_ID` and
`LANCEDB_ROBOTICS_LEROBOT_NATIVE_PUBLIC_REVISION` to a pinned, small LeRobot
dataset; `LANCEDB_ROBOTICS_LEROBOT_NATIVE_PUBLIC_SOURCE_URI` can override the
default `hf://<repo>` source URI when a prewarmed local cache or private mirror is
used.

For scheduled or public runs, pin the native source policy so the report says
exactly what the official loader measured. `source` mode refuses projection
fallbacks, `projection` mode forces the deterministic exported projection, and
`cache-only` refuses implicit Hub downloads unless `--source-dataset-uri` points
at an existing local root or prepared cache:

```bash
lancedb-robotics bench run \
  --lake ./droid-100-bench.robot.lance \
  --snapshot lerobot-droid-100-benchmark \
  --formats lerobot-native \
  --source-dataset-id lerobot/droid_100 \
  --source-dataset-revision <hf-revision-or-commit> \
  --source-dataset-uri ./hf-cache/lerobot/droid_100 \
  --lerobot-native-source-mode source \
  --lerobot-native-cache-mode cache-only \
  --lerobot-native-episode-limit 8 \
  --out ./benchmarks/droid-100-native-cache-only.json
```

The native report stores `native_loader.source.resolver`,
`native_loader.source.preflight`, and `native_loader.source.episode_filter` so a
researcher can audit whether a number came from the original source corpus, a
prepared cache, or a materialized projection.

For retained public benchmark evidence, use the combined command. It prepares
the pinned public source, runs the benchmark, stores the prepare/report JSON,
writes an artifact manifest and run log, then refreshes `index.json` and
`dashboard.md` under the artifact root:

```bash
lancedb-robotics bench run-public-lerobot \
  --lake ./droid-100-bench.robot.lance \
  --artifact-root ./benchmarks/public/lerobot \
  --source lerobot/droid_100 \
  --revision <hf-revision-or-commit> \
  --report-id droid-100-<hf-revision-or-commit>-<git-sha> \
  --formats lance,lerobot-default,lerobot-native,webdataset,deeplake \
  --size-tier droid-100 \
  --storage-tier hf-cache \
  --random-frame-samples 64 \
  --frames-per-clip 4
```

Every retained public run writes `reports/capacity.json` before prepare starts.
The default `droid-100` tier is allowed without extra budget flags. Mid and full
tiers require explicit capacity budgets so scheduled jobs do not accidentally
download or decode a much larger corpus on an undersized runner:

```bash
lancedb-robotics bench run-public-lerobot \
  --lake ./droid-100-bench.robot.lance \
  --artifact-root ./benchmarks/public/lerobot \
  --source lerobot/droid_100 \
  --revision <hf-revision-or-commit> \
  --report-id mid-<hf-revision-or-commit>-<git-sha> \
  --size-tier mid \
  --storage-tier object-store \
  --capacity-max-source-bytes 600000000000 \
  --capacity-max-artifact-bytes 30000000000 \
  --capacity-time-budget-seconds 20000 \
  --capacity-require-object-store \
  --capacity-publication-destination s3://robotics-benchmarks/lerobot
```

Use `--capacity-dry-run` to retain a plan without preparing or benchmarking the
corpus. If the selected tier is skipped, the command still writes
`artifact-manifest.json`, `reports/capacity.json`, `logs/run.log`, `index.json`,
and `dashboard.md`. The manifest and index record run-level `status`,
`capacity_status`, and `capacity_skip_reasons`, while format-arm skips such as
`deeplake` or missing `lerobot-native` dependencies remain under
`format_statuses` and `skipped`. That separation lets public benchmark claims
distinguish "this size tier was not within configured capacity" from "this
optional loader arm was unsupported on the runner."

CI can configure the same gate with
`LANCEDB_ROBOTICS_PUBLIC_LEROBOT_MAX_SOURCE_BYTES`,
`LANCEDB_ROBOTICS_PUBLIC_LEROBOT_MAX_ARTIFACT_BYTES`,
`LANCEDB_ROBOTICS_PUBLIC_LEROBOT_TIME_BUDGET_SECONDS`,
`LANCEDB_ROBOTICS_PUBLIC_LEROBOT_REQUIRE_GPU`,
`LANCEDB_ROBOTICS_PUBLIC_LEROBOT_REQUIRE_OBJECT_STORE`, and
`LANCEDB_ROBOTICS_PUBLIC_LEROBOT_PUBLICATION_DESTINATION`.

Rebuild the generated history view without rerunning measurements:

```bash
lancedb-robotics bench public-dashboard \
  --artifact-root ./benchmarks/public/lerobot
```

Publish the retained run layout to a durable static-site or object-store target
after the local run completes:

```bash
lancedb-robotics bench publish-public-lerobot \
  --artifact-root ./benchmarks/public/lerobot \
  --destination s3://robotics-benchmarks/lerobot \
  --report-id droid-100-<hf-revision-or-commit>-<git-sha> \
  --retain-latest 20
```

Publication keeps `runs/<report-id>/...` immutable: if a destination already
has different content for a report file, the command fails instead of silently
overwriting it. The published manifest records artifact checksums, publication
targets, retention class, and whether each report is protected by the latest-N
window or by future claim references. The top-level `index.json` and
`dashboard.md` remain mutable static-site entry points and are refreshed on
each publication.

Before quoting public benchmark numbers in docs, release notes, model cards, or
dashboards, validate the retained evidence and the explicit claim references:

```json
{
  "schema_version": "lancedb-robotics-public-lerobot-benchmark-claims-v0",
  "claims": [
    {
      "id": "release-lance-throughput",
      "report_id": "droid-100-<hf-revision-or-commit>-<git-sha>",
      "commit": "<git-sha>",
      "dataset_revision": "<hf-revision-or-commit>",
      "storage_tier": "hf-cache",
      "format": "lance",
      "metric": "dataloader_throughput"
    }
  ]
}
```

```bash
lancedb-robotics bench validate-public-lerobot \
  --artifact-root ./benchmarks/public/lerobot \
  --claims ./benchmarks/public/lerobot-claims.json \
  --out ./benchmarks/public/lerobot-claim-validation.json
```

The validator checks every retained manifest and the history index for schema
and provenance evidence. Claim entries must reference an existing report id and
match the recorded commit, pinned dataset revision, storage tier, format status,
and metric row. A claim that presents a skipped or unsupported format as a
measured result exits nonzero with diagnostics suitable for CI logs.

Treat external benchmark claims as pointers to a retained report id, commit,
dataset revision, and hardware/storage tier, not as hand-entered numbers copied
from a terminal.

## Audit

The inspect and ingest attempts are recorded in `transform_runs` and emitted into
the lineage graph. Source rows keep the LeRobot directory or HF repo provenance,
the dataset checksum, the declared codebase version, feature fingerprints, and
the files that contributed to the content identity.

The ingest transform params still include the final `progress` object with the
batch size, source identity, `media_inspection`, optional
`decoded_frame_conformance`, bytes scanned, rows seen/written/skipped, last
checkpoint, and a data-file -> row-group -> batch tree. The top-level transform
params also expose `media_inspection`, optional `decoded_frame_conformance`,
`video_files`, and `video_diagnostics` for dashboards that do not want to parse
the nested progress object. The
`lerobot_ingest_checkpoints` table mirrors that progress incrementally, including
HF requested/resolved revision, cache path, manifest fingerprints, claim owner,
claim token, status, phase, and any failure error. When retention summarizes old
terminal histories, the terminal checkpoint row remains the durable job evidence
that job-list, retry, and audit workflows read.

## What's next

The remaining scale work is external governance policy projection for checkpoint
holds, stale selector audits, Markdown claim extraction, object-store capacity
probe hardening, wider pinned public-codec fixture assets, video-decode
capability manifests, and hardware-accelerated decode backends for long clips.
