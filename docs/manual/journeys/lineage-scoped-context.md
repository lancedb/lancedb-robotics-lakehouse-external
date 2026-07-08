# Journey: scoped lineage context and hook plugins

**Scenario.** A training job is launched by Dagster, Airflow, Ray, Slurm, or a
Kubeflow/MLMD pipeline. You want every LanceDB Robotics operation inside that job
to carry the orchestrator run handle, but canonical Lance IDs must remain the
source of truth and worker processes must not receive secrets by accident.

Scoped lineage context solves that handoff. The context is stored with training
manifests, projection manifests, and transform params; it does not replace
`dataset_id`, `training_run_id`, `transform_id`, or Lance table-version pins.

## 1. Scope a train/export workflow

```python
with lake.lineage.context(
    {
        "provider": "dagster",
        "external_run_id": "dagster-run-42",
        "external_job_id": "curated-train-asset",
        "external_refs": {"dagster_asset_key": "robotics/demo"},
    }
):
    training = lake.training.record_run("demo-v1", hyperparameters={"lr": 0.001})
    projection = lake.projections.lerobot.export("demo-v1", out="./lerobot-demo")
```

Both operations inherit the same external run handle. Explicit per-call context
still wins when a nested operation needs a different run id:

```python
with lake.lineage.context({"provider": "airflow", "external_run_id": "dag-outer"}):
    with lake.lineage.context({"external_run_id": "task-inner"}):
        lake.training.record_run("demo-v1")

    lake.training.record_run(
        "demo-v1",
        lineage_context={"external_run_id": "manual-override"},
    )
```

Nested scopes merge maps and inherit missing scalar fields from the parent. When a
child provides the same scalar field, such as `external_run_id`, the child value
wins.

## 2. Propagate context to workers explicitly

```python
with lake.lineage.context(
    {
        "provider": "ray",
        "external_run_id": "ray-job-42",
        "environment": {"cluster": "research-gpu", "secret_token": "..."},
        "external_refs": {"ray_job_id": "job-42", "api_key": "..."},
    }
):
    worker_payload = lake.lineage.worker_context()
    worker_env = lake.lineage.worker_env()
```

`worker_context()` returns a JSON-ready payload with approved top-level keys only
and removes sensitive nested keys such as tokens, passwords, credentials, API
keys, access keys, and auth material. Pass that payload through your executor,
then open it in the worker:

```python
def worker_main(payload):
    with lake.lineage.context(payload):
        lake.projections.webdataset.plan("demo-v1")
```

This is deliberately explicit for multiprocessing, Ray, Slurm batch scripts, and
remote task runners. Context does not jump across a process boundary unless you
choose what to pass.

## 3. Validate a custom hook

Third-party hooks implement two callbacks:

```python
class DagsterLineageHook:
    def before_execution(self, operation, params=None):
        return {
            "provider": "dagster",
            "external_run_id": "...",
            "external_job_id": "...",
        }

    def after_execution(self, operation, context, *, status, error=None):
        return {"status": status}
```

Run the dependency-free conformance harness before shipping the hook:

```python
report = lake.lineage.check_hook_conformance(DagsterLineageHook())
assert report.passed, report.to_dict()
```

The harness checks callback presence, context normalization, JSON readiness, and
that `after_execution` returns an override rather than mutating the context object
it was given. It does not import Airflow, Dagster, Ray, Slurm, Kubeflow, MLflow,
W&B, or OpenLineage.

## 4. Discover adapter install guidance

```python
for spec in lake.lineage.hook_adapters():
    print(spec.to_dict())

lake.lineage.require_hook_adapter("kubeflow")
```

Built-in guidance covers Airflow, Dagster, Ray, Slurm, Kubeflow/MLMD, MLflow,
W&B, and OpenLineage. Missing adapters raise an actionable error naming the
expected module and optional extra or plugin name.

## Example provider hooks

Airflow usually maps a DAG/task instance to `external_job_id` and
`external_run_id`, with the task key in `external_refs`.

Dagster usually maps an asset job run id to `external_run_id`, an asset key to
`external_refs`, and code location/repository information to `code_ref` or
`facets`.

Ray usually maps a Ray job id or task id to `external_run_id`, with cluster and
runtime-env metadata in `environment`.

Slurm usually maps `SLURM_JOB_ID` to `external_run_id` and `SLURM_ARRAY_TASK_ID`
to `external_refs`; pass only approved environment keys to workers.

Kubeflow/MLMD usually maps pipeline run id to `external_run_id`, component id to
`external_job_id`, and MLMD context/artifact ids to `external_refs`.

## What's next

The scoped API is local and dependency-light. Follow-on scale work is entry-point
auto-discovery for packaged hooks, executor-native propagation helpers for
distributed workers, and stress tests for hook conformance in large multi-worker
training jobs.
