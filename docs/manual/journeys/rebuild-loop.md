# Journey: the rebuild loop

**Scenario.** You discover that a source is bad — a camera calibration was wrong,
a source bag was corrupted, or an embedding provider had a bug. Everything
downstream (snapshots, training runs, model outputs, evaluations) may now be
tainted. This journey walks from "mark it bad" to "hand an approved, ordered
rebuild to whatever runs your jobs."

The loop has five steps: **invalidate → plan → save → approve → dispatch** (with
reconcile as the follow-on that closes it). Each step is available both as a
`lake.lineage.*` API and as a `lineage` CLI command.

## 1. Invalidate the bad artifact

Record a durable invalidation marker. This writes an `invalidation` artifact and
`invalidates` edges from every affected artifact, so the fact survives restarts.

```bash
lancedb-robotics lineage invalidate s3://bucket/run.mcap \
  --lake s3://bucket/robot.lance --kind source \
  --reason "camera calibration bug" --severity high --discovered-by qa
```

```python
lake.lineage.invalidate("s3://bucket/run.mcap", kind="source",
                        reason="camera calibration bug", severity="high")
```

## 2. Plan the rebuild

Compute the ordered downstream impact. The plan classifies each affected artifact
into an action (`resnapshot`, `retrain`, `re-evaluate`, `re-export`, `recompute`,
`quarantine`, `notify-only`) and orders them by semantic dependency edges:

```python
plan = lake.lineage.rebuild_plan("s3://bucket/run.mcap", kind="source",
                                 reason="camera calibration bug")
for action in plan.actions:
    print(action.step, action.action, action.artifact_id)
```

### Scaling and configuring the plan (large lakes)

On a large lake a single invalidation can touch millions of artifacts, and
different teams want different responses. The planner keeps traversal
bounded to the *affected* subgraph and adds four levers on top:

- **Summary first, then drill down.** Ask for aggregates only, then page the
  ordered actions with a stable continuation token:

  ```python
  summary = lake.lineage.rebuild_plan_summary("s3://bucket/run.mcap", kind="source")
  print(summary.affected_artifact_count, summary.actions_by_type)

  page = lake.lineage.rebuild_plan("s3://bucket/run.mcap", kind="source", page_size=500)
  while page.next_page_token:
      handle(page.actions)
      page = lake.lineage.rebuild_plan("s3://bucket/run.mcap", kind="source",
                                       page_size=500, page_token=page.next_page_token)
  ```

  ```bash
  lancedb-robotics lineage rebuild-plan s3://bucket/run.mcap --kind source \
    --lake s3://bucket/robot.lance --summary
  lancedb-robotics lineage rebuild-plan s3://bucket/run.mcap --kind source \
    --lake s3://bucket/robot.lance --page-size 500 --page-token <token>
  ```

- **Guardrails.** Fail fast with an actionable error instead of building an
  unbounded plan, and refuse to full-scan when the traversal indexes are missing:

  ```python
  lake.lineage.rebuild_plan("s3://bucket/run.mcap", kind="source",
                            max_affected_artifacts=100_000, max_actions=100_000,
                            require_indexes=True)   # raises RebuildPlanTooLarge / RebuildPlanError
  ```

- **Action policies.** Remap artifacts to actions without changing *which*
  artifacts are impacted — the default policy reproduces the backlog 0066
  classification exactly. Key on kind, table, incoming edge type, or severity:

  ```python
  from lancedb_robotics.lineage import MappingActionPolicy

  policy = MappingActionPolicy(
      kind_actions={"evaluation-run": "re-evaluate"},
      severity_actions={"low": "notify-only"},   # only notify on low-severity finds
  )
  plan = lake.lineage.rebuild_plan("s3://bucket/run.mcap", kind="source",
                                   action_policy=policy)
  ```

  ```bash
  lancedb-robotics lineage rebuild-plan s3://bucket/run.mcap --kind source \
    --lake s3://bucket/robot.lance --action-policy policy.json
  ```

  A policy may only emit the known action vocabulary; anything else is rejected
  at construction or plan time.

- **Benchmark it.** `lake.lineage.benchmark_rebuild_plan(...)` reports the
  affected-artifact count, action count, traversal time, and peak memory over a
  synthetic or real graph.

## 3. Save the plan to the durable catalog

A plan on its own is a transient report. Recording it makes it durable and gives
it a stable identity — a content digest over its roots, reason, severity, and
ordered actions. The digest deliberately **excludes** the invalidation timestamp
and the impact graph, so re-planning the same invalidation on the same lake state
records to the same row (idempotent), and the bounded plan payload is stored
inline so it reloads without re-tracing the graph.

```bash
lancedb-robotics lineage save-rebuild-plan s3://bucket/run.mcap \
  --lake s3://bucket/robot.lance --kind source \
  --reason "camera calibration bug"
# → prints the catalog entry, including plan_id (== plan_digest) and status "draft"
```

```python
entry = lake.lineage.record_rebuild_plan(plan)          # status "draft", revision 0
again = lake.lineage.record_rebuild_plan(plan)           # idempotent: same plan_id
reloaded, payload = lake.lineage.get_rebuild_plan(entry.plan_id)  # no re-trace
```

List and reload later:

```bash
lancedb-robotics lineage list-rebuild-plans --lake s3://bucket/robot.lance --status draft
lancedb-robotics lineage show-rebuild-plan <plan_id> --lake s3://bucket/robot.lance --include-plan
```

## 4. Approve it

A rebuild can cost real GPU-hours, so it moves through a lifecycle:
`draft → approved → dispatched → completed`/`failed` (plus `abandoned`).
Transitions are validated, and each bumps a `revision` counter used for
optimistic concurrency. Approving requires an approver.

```bash
lancedb-robotics lineage set-rebuild-status <plan_id> approved \
  --lake s3://bucket/robot.lance --approver lead
```

```python
lake.lineage.update_rebuild_plan_status(plan_id, "approved", approver="lead")
```

If someone else changed the plan since you last read it, guard the update and get
an actionable conflict instead of clobbering:

```python
lake.lineage.update_rebuild_plan_status(
    plan_id, "dispatched", expected_revision=1)   # raises RebuildPlanConflict if stale
```

## 5. Dispatch to an orchestrator (or run it yourself)

Export a **deterministic** handoff payload: each action gets a stable `action_id`,
its dependencies resolved to action ids, the target artifact id, table-version
pins, and a retry-safe `external_run_ref`. Re-exporting an unchanged plan yields a
byte-identical payload, so a consumer can dedupe on it.

```bash
# dry-run preview (no state change):
lancedb-robotics lineage export-rebuild-plan <plan_id> \
  --lake s3://bucket/robot.lance --orchestrator dagster
# record the dispatch (requires the plan to be approved; idempotent re-dispatch):
lancedb-robotics lineage export-rebuild-plan <plan_id> \
  --lake s3://bucket/robot.lance --dispatch
# one JSON object per action, for a pipeline to consume:
lancedb-robotics lineage export-rebuild-plan <plan_id> \
  --lake s3://bucket/robot.lance --ndjson
```

```python
dispatch = lake.lineage.export_rebuild_plan_dispatch(plan_id, orchestrator="dagster")
for action in dispatch.actions:
    print(action["action_id"], action["action"], action["external_run_ref"],
          "after", action["depends_on"])
```

Nothing here *requires* an external engine. `--orchestrator` is just a label; the
payload is plain data. You can walk the actions yourself and run each with the
lake's own commands (`create_snapshot`, `lake.training.record_run`,
`lake.eval.record_run`, ...), then mark the plan `completed`.

## Audit

Every recording, status transition, and dispatch is written to an append-only
event log that survives status churn:

```bash
lancedb-robotics lineage rebuild-plan-events --lake s3://bucket/robot.lance --plan-id <plan_id>
```

## What's next (not yet built)

The v1 catalog tracks **whole-plan** status, not per-action outcomes. Closing the
loop — recording each action's result as it runs and deriving plan status from
them (`reconcile`) — is a filed follow-on, and works with local CLIs, no
orchestrator required. See the rebuild-catalog follow-on backlog items.
