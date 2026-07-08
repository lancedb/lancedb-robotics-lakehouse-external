# Journey: prove Enterprise training behaves the same everywhere

**Scenario.** Before shipping a new Enterprise training release, a platform team
needs to trust that `backend="enterprise"` behaves the same across a `db://`
remote DB, a REST Namespace query node, a local test double, and *degraded*
deployments — and that when a deployment can't do something, the loader fails
loudly with a typed error or falls back *explicitly*, never quietly reading the
data locally and pretending it ran on the query node.

This journey runs one snapshot through every backend configuration and every
known failure mode, checks the central invariant on each, and emits a
compatibility matrix an operator can read at a glance. It needs **no Enterprise
secrets** — each case reconfigures a local lake to simulate the target backend
or degradation — so it runs in ordinary CI. An optional, opt-in pass exercises a
real HTTP endpoint served by the local LanceDB Enterprise CLI.

The central invariant: *every degradation is a typed error or an explicit
fallback report — never silent local materialization.*

The flow is: **read the matrix → run the harness → (optionally) hit a real endpoint**.

## 1. Read the compatibility matrix

The matrix is a deterministic, data-free classification of each backend scenario
and injected fault into `supported` / `fallback` / `unsupported`, produced by the
*same* backend resolver the loader uses — so it can never drift from real
behavior. No lake is required.

```bash
lancedb-robotics train conformance matrix --format markdown
```

```python
matrix = lake.training.conformance_matrix()
for row in matrix.rows:
    print(row.name, row.category, row.error_type or row.fallback_to or row.resolved_backend)
```

The generated matrix is also committed as
[reference/enterprise-training-compatibility.generated.md](../reference/enterprise-training-compatibility.generated.md)
and refreshed by `scripts/gen_docs_reference.py`. There is deliberately **no
"silent local" column** — that outcome is a conformance *failure*, never a
documented cell.

## 2. Run the harness against a real snapshot

Replay a snapshot through each backend case and injected fault. For each case the
harness builds a local *twin* of the same request and asserts the (faked)
Enterprise path produces **equivalent sample ids, row ids, table-version lineage,
and batch schema**, that worker handoff keeps `host_override` routing intact
without serializing API keys, and that every degradation is typed or explicitly
reported.

```bash
lancedb-robotics train conformance run --lake ./robot.lance --snapshot demo-v1 --strict
```

```python
report = lake.training.run_conformance("demo-v1", strict=True)
print(report.summary())               # {'total': ..., 'passed': ..., 'failed': 0, ...}
for outcome in report.failures():      # empty on a conforming build
    print(outcome.name, outcome.failures)
```

Injected faults include missing auth, expired scoped credentials (with a single
automatic refresh attempt before failing), unsupported remote scan/take/blob
hydration, unavailable page-cache prewarm and cache metrics, stale pinned table
versions, direct-data-plane and explicit-local fallbacks, and worker
resume-offset mismatches. Each maps to a typed `EnterpriseTrainingError` subclass
or an explicit fallback event — see the matrix for the full list.

## 3. (Optional) exercise a real local Enterprise endpoint

For a first real HTTP `host_override` + query-node registration pass, opt in with
an environment flag and the LanceDB Enterprise CLI binary present. The harness
spawns `lancedb server` and `lancedb pe`, connects through the live
`host_override`, and asserts all training requests route to that endpoint. The
pass is skipped (never failed) when the flag or binary is absent, so secret-free
CI is unaffected.

```bash
LANCEDB_ROBOTICS_ENTERPRISE_ENDPOINT=1 \
  lancedb-robotics train conformance run --lake ./robot.lance --snapshot demo-v1 \
  --include-local-endpoint --strict
```

```python
report = lake.training.run_conformance("demo-v1", include_local_endpoint=True, strict=True)
print(report.local_endpoint)   # {'http_endpoint': 'http://127.0.0.1:...', 'server_running': True, ...}
```

## Deprecated plan-executor names (backlog 0345)

A training client talks only to object storage or the **query node** —
never to a plan executor, and it cannot read plan-executor internals. The SDK
surface was renamed accordingly. The pre-0345 names still work (importable
classes, honored lake hooks, a hidden CLI alias) and will be removed in a future
release; new code should use the query-node names.

| Deprecated (pre-0345) | Use instead |
| --- | --- |
| `RemotePlanExecutorClient`, `PlanExecutorClient` | `RemoteQueryNodeClient`, `QueryNodeClient` |
| `PlanExecutorRequest` / `PlanExecutorResponse` | `QueryNodeRequest` / `QueryNodeResponse` |
| `RemotePlanExecutorError`, `PlanExecutorUnavailableError` | `RemoteQueryNodeError`, `QueryNodeUnavailableError` |
| `FakePlanExecutorClient`, `PlanExecutorConformanceReport` | `FakeQueryNodeClient`, `QueryNodeConformanceReport` |
| `run_plan_executor_conformance`, `lake.training.plan_executor_conformance` | `run_query_node_conformance`, `lake.training.query_node_conformance` |
| `lake.plan_executor_client` | `lake.query_node_client` |
| `lake.plan_executor_prewarm` / `lake.plan_executor_prewarm_status` | `lake.page_cache_prewarm` / `lake.page_cache_prewarm_status` |
| `lake.plan_executor_cache_metrics` | `lake.query_node_cache_telemetry` |
| `train conformance plan-executor` | `train conformance query-node` |

Wire/persisted names are intentionally **unchanged** for back-compat and because
they are server-reported telemetry, not client inputs: the report keys
`plan_executor` (capability block) and `cache_by_plan_executor`, the `pe_fanout`
column and `pe_addrs` response field, and the `plan-executor-conformance/v1`
report schema. `pe_fanout` / `pe_addrs` are only ever populated from a server
response — the client never sends or routes on them.

## What's next (not yet built)

The fake-remote cases simulate capability negotiation and routing but do not read
data over a real query node (that is 0070–0072's remote execution path), so
lineage equivalence is asserted against a local twin rather than a live remote
stream. The compatibility matrix is a fixed case registry rather than a
per-deployment probe, and the local-endpoint pass asserts routing/handoff rather
than a full remote data read. Extending the matrix across other SDK workflows is
0076's remote-compatibility audit; the scale and depth follow-ons are filed as
this task's backlog items.
