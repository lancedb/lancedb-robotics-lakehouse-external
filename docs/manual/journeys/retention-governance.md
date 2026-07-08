# Journey: retention policy and governance

**Scenario.** Compliance says every artifact tied to a regulated dataset must be
held for audit for seven years, and legal occasionally freezes a specific run.
You do not want to hand-edit retention on thousands of artifacts, and you need to
show an auditor — and your enterprise governance / DLP / SIEM system — what is
held, why, and under whose authority.

Backlog 0067 already lets you set a hold on one artifact with
`lake.lineage.retain(...)`. This journey adds the layer above it: **reusable,
versioned policy definitions** that *expand* to those same holds across a whole
scope, an approval lifecycle, an append-safe history, and a dependency-free way to
project the state out. The loop is **define → activate → apply → resolve →
release/expire → project**. Each step is a `lake.lineage.*` API and a `lineage`
CLI command.

A policy never stores per-artifact state. It expands into ordinary 0067 holds, so
maintenance, `lake.lineage.audit`, and reconciliation treat a policy-applied hold
exactly like a manual one. A manually set (artifact-local) hold is always the
lowest-level override: a policy never clobbers it.

## 1. Define a policy

A policy has a name/version, a **scope** (selectors over artifact kind, table,
owner, source, dataset, model, deployment, project, name prefix, or explicit id),
and **rules** (an absolute `retain_until`, a relative `retain_for_days`, and/or an
indefinite legal/audit/promotion hold). Within a selector category the values OR;
across categories they AND. Identity is the content digest of the definition, so
re-recording the same policy is idempotent and changing it mints a new immutable
id.

```bash
lancedb-robotics lineage save-retention-policy --lake s3://bucket/robot.lance \
  --name observations-audit --kind table-version --table observations \
  --audit-hold --owner data-governance --reason "regulatory audit retention" \
  --status draft
```

```python
from lancedb_robotics.retention_catalog import build_retention_policy

policy = build_retention_policy(
    name="observations-audit", kinds=["table-version"], tables=["observations"],
    audit_hold=True, owner="data-governance", reason_template="regulatory audit retention")
entry = lake.lineage.record_retention_policy(policy, actor="ops")
```

## 2. Activate it (approval)

A policy only enforces holds once it is `active`, and activating requires an
`approver`. The lifecycle is `draft → active → suspended` (plus terminal
`archived`), guarded by a `revision` optimistic-concurrency counter so a stale
update is rejected instead of clobbering.

```python
lake.lineage.update_retention_policy_status(
    entry.policy_id, "active", approver="dpo", expected_revision=0)
```

```bash
lancedb-robotics lineage set-retention-status <policy_id> active \
  --lake s3://bucket/robot.lance --approver dpo --expected-revision 0
```

## 3. Apply it (expand to holds)

Applying resolves the scope against `lineage_artifacts` and writes the policy's
hold onto every match — no per-artifact edits. The default is a **dry run** that
previews matches and conflicts without writing; pass `--apply` to materialize.
Artifacts that already carry a human or other-policy hold are **conflicts**:
reported and left untouched. `--max-artifacts` bounds the match before any write.

```bash
# preview:
lancedb-robotics lineage apply-retention-policy <policy_id> --lake s3://bucket/robot.lance
# write the holds:
lancedb-robotics lineage apply-retention-policy <policy_id> --lake s3://bucket/robot.lance --apply
```

```python
result = lake.lineage.apply_retention_policy(policy_id, dry_run=False)
print(result.applied_count, result.conflict_count)
for conflict in result.conflicts:
    print(conflict["artifact_id"], conflict["existing_source"])  # e.g. "artifact-local"
```

## 4. Resolve — the same shape maintenance uses

`resolve_retention_holds` merges policy-applied and artifact-local holds into the
**exact table-version pin shape** `lake maintain` consumes, so what you preview is
what maintenance will protect. Each hold is classified by source
(`policy:<id>` vs `artifact-local`), and active policies shadowed by an
artifact-local override are surfaced as deterministic conflicts.

```python
resolution = lake.lineage.resolve_retention_holds()
print(resolution.policy_hold_count, resolution.artifact_local_count)
for pin in resolution.pins:          # identical to maintenance's retention pins
    print(pin["table"], pin["version"], pin["categories"])
```

```bash
lancedb-robotics lineage resolve-retention-holds --lake s3://bucket/robot.lance
```

## 5. Release or expire

Releasing clears only the holds *this* policy applied — a manual hold or another
policy's hold is never touched. Time-based holds that have passed (or are about to)
surface as expiration notices; indefinite legal/audit/promotion holds never
expire.

```bash
lancedb-robotics lineage release-retention-policy <policy_id> --lake s3://bucket/robot.lance --release
lancedb-robotics lineage retention-expiration-notices --lake s3://bucket/robot.lance --within-days 30 --notify
```

## 6. Project to a governance system

Export policy + resolved-hold state as JSON or NDJSON for an enterprise
governance / DLP / SIEM / records system. The projection is deterministic given
lake state, carries no secrets, and adds **no mandatory dependency**: your sink
owns its transport and credentials at runtime, and nothing from it is persisted.

```bash
lancedb-robotics lineage export-retention-state --lake s3://bucket/robot.lance --ndjson
```

```python
from lancedb_robotics.retention_catalog import CollectingGovernanceSink

sink = CollectingGovernanceSink()          # or your own object with .project(projection)
receipt = lake.lineage.project_retention_state(sink)
print(receipt["projected_policies"], receipt["projected_holds"])
```

## Audit

Every recording, status transition, apply, release, and expiration notice is
written to an append-only event log that survives status churn and policy
archival:

```bash
lancedb-robotics lineage retention-policy-events --lake s3://bucket/robot.lance --policy-id <policy_id>
```

## What's next (not yet built)

Scope resolution pushes down a coarse kind/table/id predicate and matches the rest
in Python; a policy scoped only by owner/project still scans the artifact table.
Bounded, indexed scope resolution and validator/notification plugins are filed
follow-ons — see the retention-catalog follow-on backlog items.
