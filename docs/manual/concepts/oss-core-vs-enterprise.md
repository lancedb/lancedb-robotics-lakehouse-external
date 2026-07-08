# OSS core vs. enterprise/plugin

A recurring architectural line runs through the whole lakehouse, so it is worth
stating once. There are two tiers:

1. **OSS core** — the open-source `lancedb_robotics` package in this repo. It is
   the shared, Lance-native substrate: the canonical tables, the bounded/streaming
   read and write paths, the lineage graph, and CLIs/APIs that run **standalone**.
   Core stays generic and unopinionated.
2. **Enterprise / plugin layers** — deployment-specific code layered on top:
   external integrations, authorization policy, remote/namespace connectivity, and
   heavy distributed compute.

Core stops at producing durable records and emitting well-defined payloads; it
does **not** execute your compute, run a scheduler, or decide org policy. This
keeps the open engine portable and lets each organization bolt on its own rules
without forking core.

## What that looks like in practice

- **Optional dependencies stay optional.** Extras such as the `graph` (lance-graph
  Cypher) backend and embedding/RLDS providers are probed, not required. Metadata
  integrations (OpenLineage, DataHub, MLflow, W&B) go through a plugin contract
  that uses `find_spec` and never imports an optional dependency at module load.
- **Auth is never persisted in core.** Lineage delivery and remote access take an
  auth reference at runtime (env, headers, injected client); credentials are never
  written into lake rows.
- **Heavy compute is a separate layer.** Distributed column materialization is
  handed to the compute engine (Geneva/Ray), not run inside core.
- **Policy is deferred.** Where core needs a decision it cannot own, it exposes
  generic metadata and a seam. For example, the rebuild-plan approval lifecycle
  records a free-form `approver` and only enforces that approving requires *an*
  approver — *who* may approve, and quorum rules, are left to an approval-policy
  plugin.

## Why the rebuild loop hands off instead of executing

The clearest illustration is the [rebuild loop](../journeys/rebuild-loop.md). Core
records a plan, gates its approval, and emits a **deterministic, dependency-ordered
dispatch payload** with stable action ids and retry-safe run references. It does
**not** require Airflow / Dagster / Ray / Slurm — those are just examples of
engines that can consume the payload. You can equally drive the whole lifecycle by
hand with the CLI, and run each action with the lake's own commands
(`create_snapshot`, `lake.training.record_run`, and so on). Core emits the plan;
*how* it runs is your choice.
