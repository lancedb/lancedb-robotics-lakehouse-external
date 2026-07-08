# Lance-native training datasets

Backlog 0028 is now the direct training path over the lake. The point is not to
look like LeRobot or WebDataset first; the point is to keep training connected to
the same Lance data that powers search, curation, versioning, and lineage.

Use `lake.training.dataset(...)` when a model job should train from a pinned
snapshot without rebuilding shards:

```python
from lancedb_robotics.lake import Lake

lake = Lake.open("./demo.robot.lance")

dataset = lake.training.dataset(
    "demo-v1",
    columns=[
        "observation_id",
        "timestamp_ns",
        "state_vector",
        "action_vector",
        "payload_size",
    ],
    filters={"split": "train", "modality": "image"},
    shuffle=True,
    shuffle_seed=17,
    time_windows={"state_vector": [-0.2, 0.0, 0.2]},
)

sample = dataset[0]
```

Samples use canonical lake fields:

```python
{
    "observation_id": "obs-...",
    "scenario_id": "scn-...",
    "run_id": "run-...",
    "split": "train",
    "timestamp_ns": 1700000000,
    "state_vector": [...],
    "action_vector": [...],
    "payload_size": 12345,
    "windows": {
        "state_vector": [
            {"delta_s": -0.2, "timestamp_ns": ..., "frame_index": 0, "value": [...]},
            {"delta_s": 0.0, "timestamp_ns": ..., "frame_index": 1, "value": [...]},
            {"delta_s": 0.2, "timestamp_ns": ..., "frame_index": 2, "value": [...]},
        ]
    },
}
```

The dataset exposes a manifest with the snapshot id, pinned table versions,
projection, filters, shuffle settings, window settings, and frame counts. That is
the core contract: training samples remain traceable to the source lake.

## Aligned Tick Datasets

Use `lake.training.aligned_dataset(...)` when a policy should train from a
recorded multi-rate alignment job. New recorded alignments write `aligned_ticks`
as the preferred training surface: one Lance row per policy tick, with stable
top-level predicate columns such as `alignment_id`, `run_id`, `tick_index`,
`timestamp_ns`, `has_missing`, `has_out_of_tolerance`, and `min_confidence`.
Dynamic per-stream detail lives in JSONB columns (`stream_detail_json`,
`masks_json`, `stream_values_json`, and `lineage_json`) so recipes can carry
arbitrary stream names, masks, source row ids, values, and lineage without
forcing a new fixed Arrow struct for every alignment shape.

```python
dataset = lake.training.aligned_dataset(
    name="policy_bridge",
    streams=["/joint_states", "/action"],
    min_confidence=0.75,
)

sample = dataset[0]
manifest = dataset.manifest.to_dict()
assert manifest["storage_backend"] in {
    "aligned_ticks-jsonb",
    "aligned_frames-pivot",
}
```

The intended researcher experience is automatic: alignment materialization,
background maintenance, or Enterprise index jobs should create and refresh
useful predicate indexes without requiring the training author to understand
LanceDB index mechanics. Explicit APIs remain available for operators,
benchmarks, and deterministic setup:

```python
index_results = lake.training.index_aligned_predicates(include_frames=True)
assert all(result["status"] in {"built", "already_present"} for result in index_results)
```

`aligned_dataset(...)` does not block on ad hoc index builds during a read. It
records `manifest["predicate_indexes"]` with one entry per hot predicate column,
including `status` (`already_present`, `skipped`, or `failed`), `used_in_filter`,
and `predicate_role`. Unsupported or still-building backends keep using
predicate pushdown scans and record a reason instead of blocking training reads.

The compatibility `aligned_frames` table remains available for per-stream debug
and old lakes. If `aligned_ticks` rows are absent, `aligned_dataset(...)` falls
back to the 0049 pivot path and records `storage_backend:
aligned_frames-pivot` in the manifest. To migrate an older alignment job, run:

```python
result = lake.training.backfill_aligned_ticks(name="policy_bridge")
assert result["metadata_samples_verified"]
```

Schema rule: hot predicates graduate to typed top-level columns where LanceDB
can filter and index them; dynamic stream maps stay JSONB until a predicate is
common enough to deserve additive schema evolution. Fixed Arrow arrays,
fixed-size lists, and structs are reserved for stable feature shapes such as
known tensors, not arbitrary stream maps.

## Run Manifests

Once a trainer consumes a native dataset, record the actual run and checkpoint
back into the lake:

```python
training = lake.training.record_run(
    dataset=dataset,
    code_ref="git:abc123",
    environment={"container_image": "policy-train@sha256:..."},
    hyperparameters={"lr": 0.001},
    random_seeds={"python": 17},
    external_refs={"wandb_run_id": "abc"},
)

checkpoint = lake.training.record_checkpoint(
    training_run_id=training.training_run_id,
    artifact_uri="s3://models/policy.ckpt",
    checksum="sha256:...",
    aliases=["candidate"],
)

lake.eval.record_run(
    "demo-v1",
    model_artifact_id=checkpoint.model_artifact_id,
    metrics={"success_rate": 0.92},
    slice_metrics={"night/rain": {"success_rate": 0.81}},
)
```

These calls write `training_runs`, `model_artifacts`, and `evaluation_runs`.
They keep external tracker IDs as optional `external_refs`; MLflow and W&B are
integrations, not requirements. `lake.lineage.refresh_graph()` projects the
rows into traceable training/model/evaluation artifacts so
`lake.lineage.trace_checkpoint(checkpoint.model_artifact_id)` can recover the
input snapshot, table versions, code ref, params, environment, and source rows.

## Boundary Projections

Format integrations live under `lake.projections`, not `lake.facades`. The
native path stays the default when a job can train directly from Lance:

```python
dataset = lake.training.dataset("demo-v1")
```

Use a live projection when downstream code expects a boundary format identity but
you still want to avoid repacking the snapshot:

```python
adapter = lake.projections.lerobot.dataset("demo-v1", mode="live")
sample = adapter[0]
manifest = adapter.manifest
```

The LeRobot live adapter is a thin compatibility layer over the same native row
plan. It exposes LeRobot-style metadata (`adapter.meta.episodes`,
`adapter.meta.frames`, `adapter.meta.videos`, `adapter.features`,
`adapter.num_frames`) and sample keys such as `observation.state`, `action`,
`timestamp`, `task`, `language_instruction`, and
`observation.images.<camera>`. Camera values follow the native media policy, so
the default is a lazy Lance media handle rather than a repacked image/video
file. Each sample also carries projection lineage back to the snapshot id,
table-version pins, row plan, row/frame id, and raw source provenance.

The adapter carries a projection manifest with the source snapshot id, pinned
table versions, projection format/version, feature schema, lossiness, media
policy, live-adapter identity, and transform lineage id. A dry run validates the
same contract without writing an external layout:

```python
manifest = lake.projections.rlds.plan("demo-v1")
```

Materialized exports are the portability and anti-lock-in path:

```python
manifest = lake.projections.lerobot.export("demo-v1", out="./demo-lerobot")
```

The LeRobot export records `lerobot-v3.0` in the projection manifest and writes
the current chunked layout (`meta/info.json`, `meta/tasks.parquet`,
`meta/episodes/chunk-000/file-000.parquet`, `data/chunk-000/file-000.parquet`)
plus `dataset_export_manifest.json` and `projection_manifest.json`.

RLDS follows the same source-of-truth rule but is episode-oriented in live mode:

```python
episodes = lake.projections.rlds.dataset("demo-v1", mode="live")
episode = next(iter(episodes))
step = episode["steps"][0]
```

Each live episode exposes `episode_metadata` plus `steps`; each step carries
`observation`, `action`, `reward`, `discount`, `is_first`, `is_last`,
`is_terminal`, and metadata/lineage back to the Lance row plan. Rewards are
synthesized as `0.0` and discounts from terminal position until canonical reward
signals exist, and that lossiness is recorded in the projection manifest.
Materialized RLDS export writes the same episode/step mapping to
`dataset_info.json`, `features.json`, `episodes.jsonl`, and per-episode Parquet
step files for portability.

WebDataset follows the same boundary rule. Live mode returns a WebDataset-shaped
iterable whose samples expose `__key__`, `json`, `txt`, and media extension keys
such as `jpg` or `bin`, backed by the native row plan rather than tar shards:

```python
stream = lake.projections.webdataset.dataset("demo-v1", mode="live")
sample = next(iter(stream))
```

Dry-run planning reports deterministic shard ranges, estimated bytes, sample
schema, and optional `webdataset` dependency status without writing output:

```python
plan = lake.projections.webdataset.plan("demo-v1", shard_size=1024)
```

Materialized export writes deterministic `.tar` or `.tar.gz` shards using the
same sample keys and metadata mapping as live mode, plus the shared projection
manifest. It is the portability path for downstream stacks that require shard
files, not the native training surface:

```python
manifest = lake.projections.webdataset.export(
    "demo-v1",
    out="./demo-webdataset",
    shard_size=1024,
)
```

The CLI mirrors those modes:

```bash
lancedb-robotics dataset project lerobot --mode live --lake ./demo.robot.lance --snapshot demo-v1
lancedb-robotics dataset project lerobot --mode plan --lake ./demo.robot.lance --snapshot demo-v1
lancedb-robotics dataset export lerobot --lake ./demo.robot.lance --snapshot demo-v1 --out ./demo-lerobot
lancedb-robotics dataset project rlds --mode live --lake ./demo.robot.lance --snapshot demo-v1
lancedb-robotics dataset export rlds --lake ./demo.robot.lance --snapshot demo-v1 --out ./demo-rlds
lancedb-robotics dataset project webdataset --mode plan --lake ./demo.robot.lance --snapshot demo-v1 --shard-size 1024
lancedb-robotics dataset export webdataset --lake ./demo.robot.lance --snapshot demo-v1 --out ./demo-webdataset
```

Every projection mode records a `transform_runs` row with a deterministic
`tfm-projection-*` id. Re-running the same snapshot/options updates the same
projection lineage row, while materialized exports also write
`projection_manifest.json` next to the format-specific output manifest.
