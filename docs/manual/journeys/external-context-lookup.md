# Journey: find a run by its external ID (and share it safely)

**Scenario.** An operator has a Weights & Biases (or Airflow, or MLflow) run ID
from a dashboard and needs to answer: *which canonical Lance training run is that,
what did it produce, and can I hand an auditor an evidence pack without leaking our
environment secrets?*

Backlog 0068 already records external run/job/code/environment context inside
transform params and manifest fields, and 0098 projects executions into
`lineage_executions`. This journey turns that scattered context into a **queryable
catalog** and applies a **redaction policy** before anything is exported. The Lance
execution/artifact IDs stay the source of truth; the catalog only indexes external
handles that point at them.

The flow has three steps: **backfill → find → export redacted**.

## 1. Backfill the external-context catalog

Index the external handles already present on your canonical rows into the
`external_contexts` catalog. The scan is bounded (a fixed number of rows in memory
per batch) and idempotent — re-running never duplicates rows, because each row's
identity is a content digest over its source row and external handle fields.

```bash
lancedb-robotics lineage backfill-external-context \
  --lake s3://bucket/robot.lance --batch-size 512
# → prints {scanned, recorded, updated, skipped, batches, sources}
```

```python
report = lake.lineage.backfill_external_contexts(batch_size=512)
```

By default it scans `lineage_executions` (authoritative — it already resolves
canonical execution and artifact IDs) and `transform_runs` (for transforms not yet
projected). Restrict with `--source`/`sources=(...)` if you only want one.

## 2. Find the canonical execution and artifacts

Look up by any external handle — provider, run ID, job ID, parent run ID, code
ref, environment digest, or external artifact URI/URN. Results page
deterministically with an opaque cursor, so a large catalog never loads at once.

```bash
lancedb-robotics lineage find-external-context \
  --lake s3://bucket/robot.lance --provider wandb --run-id RUN-1 --page-size 50
```

```python
page = lake.lineage.find_external_context(provider="wandb", external_run_id="RUN-1")
for ctx in page.contexts:
    print(ctx.provider, ctx.external_run_id, "→",
          ctx.execution_id, ctx.artifact_ids, ctx.transform_id)
```

Two providers that happen to share a run ID resolve to **distinct namespaced
records** — the `provider` column keeps them apart and each points at its own
canonical execution. Each entry carries `execution_id`, `artifact_ids`, and
`transform_id`, so you can hand straight off to `lake.lineage.trace(...)` /
`impact(...)` to walk upstream or downstream.

## 3. Export an evidence pack with environment secrets redacted

A redaction policy combines three controls: an **allowlist** of top-level context
keys, a **denylist** of case-insensitive key fragments (`secret`, `token`,
`password`, `credential`, `api_key`, ... by default), and opt-in **secret-value
patterns** (AWS/GitHub/JWT/PEM) that mask a value in place. The policy is applied
**before the pack digest is computed and any bytes are materialized**, so denied
keys never reach the written manifest.

```bash
lancedb-robotics lineage record-evidence lancedb-robotics:model:m1 \
  --lake s3://bucket/robot.lance --redact --redact-detect-secrets
```

```python
from lancedb_robotics.redaction import ContextRedactionPolicy

policy = ContextRedactionPolicy(name="audit-safe")
pack = lake.lineage.evidence_pack("lancedb-robotics:model:m1",
                                  redaction_policy=policy,
                                  output_dir="/tmp/pack", materialize=True)
```

The same policy can be applied at backfill time (`--redact` on
`backfill-external-context`) so the catalog's stored `context_json` is redacted at
rest too.

## Retention and holds

Recorded contexts carry retention governance: mark one protected, set an
`expires_at`, or place a legal/audit hold. Expiry is force-gated when a row is
held, and a plan reports which rows are held versus safe to expire.

```python
lake.lineage.set_external_context_retention(cid, legal_hold=True)
lake.lineage.expire_external_context(cid)              # refused: row is held
lake.lineage.expire_external_context(cid, force=True)  # removes it

plan = lake.lineage.external_context_retention_plan()
print(plan["held_count"], plan["expirable_count"])
```

## Audit

Backfill, retention updates, redaction, and expiry are written to an append-only
event log that survives row expiry:

```bash
lancedb-robotics lineage external-context-events \
  --lake s3://bucket/robot.lance --context-id <context_id>
```

## What's next (not yet built)

`find-external-context` currently filters over the scan of matching rows rather
than a pushed-down scalar index, and the backfill's within-run dedup set is
memory-resident. Scalar indexes on the external-handle columns and a fully
streaming dedup are filed follow-ons — see the external-context-catalog follow-on
backlog items.
