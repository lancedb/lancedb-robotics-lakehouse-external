# Overview: the lake, tables, and lineage

This chapter is the mental model the rest of the manual builds on. Read it, then
the companion concept chapters — [architecture](architecture.md) and
[storage and auth](storage-and-auth.md) — before diving into a journey.

## The lake

Everything lives in a **lake** — a single LanceDB database at a local path, an
object-store URI, or an enterprise/namespace URI. You open it once and reach every
capability through **namespaces** on the handle:

```python
from lancedb_robotics.lake import Lake

lake = Lake.open("s3://bucket/robot.lance")   # or Lake.init(path) for a new one
lake.lineage.trace(...)                        # namespaces expose the API
```

The namespaces are the map of what the system does:

| Namespace | What it covers |
| --- | --- |
| `lake.ingest` / `lake.inspect` * | register and ingest raw logs; inspect them first |
| `lake.video` | codec-aware video encoding and frame access |
| `lake.align` | multi-rate temporal alignment of streams |
| `lake.episodes` | first-class episodes: creation, derivation, frames |
| `lake.embeddings` | build embedding columns (text/image providers) |
| `lake.curate` | saved views, dedup/diversity, mining, review queues, comparisons |
| `lake.distributions` | distribution specs, balance reports, gap analysis |
| `lake.training` | version-pinned snapshots, dataloaders, training runs |
| `lake.projections` | LeRobot / RLDS / WebDataset live adapters and export |
| `lake.eval` (`lake.evaluation`) | evaluation runs and metrics |
| `lake.tracker` | external experiment-tracker (MLflow/W&B) manifest sync |
| `lake.lineage` | provenance, invalidation, rebuild planning, evidence, audit |

\* ingest/inspect are exposed as CLI groups and top-level `Lake` methods rather
than a namespace object. The same surface is available from the CLI as
`lancedb-robotics <group> <command>`. For the exact, always-current list of every
method and command, see the generated [Python API](../reference/api.generated.md)
and [CLI](../reference/cli.generated.md) reference.

## Canonical tables

`Lake.init` creates a fixed set of **canonical tables** — the schema of the
lakehouse. They span the whole pipeline:

- **raw ingest** — `integration_sources`, `runs`, `observations`, `events`,
  `attachments`, `videos`, `video_encodings`;
- **curation & search** — `scenarios`, `curation_views`, `curation_memberships`,
  `distribution_catalog`, `labels`;
- **datasets & training** — `dataset_snapshots`, `training_runs`,
  `model_artifacts`, `evaluation_runs`, `episodes`, `alignment_jobs`,
  `aligned_ticks`;
- **closed-loop & lineage** — `model_outputs`, `feedback`, `transform_runs`,
  `lineage_artifacts`, `lineage_edges`, `evidence_packs`, `rebuild_plans`.

The complete list, with columns and schema versions, is in the generated
[tables reference](../reference/tables.generated.md).

Two properties hold across all of them:

- **The lake is the working set, not just an index.** Heavy payloads (sensor
  bytes, video) are first-class blob-encoded columns; search, curation, and
  training are served from Lance, and enrichment is *additive* (new columns / new
  versions), never an in-place rewrite. See [architecture](architecture.md).
- **It scales past memory, without you managing it.** The corpus never has to fit
  in RAM: queries stream, a dataset snapshot is a version-pinned *selection* rather
  than a copy, and large operations resume rather than restart. You work with
  terabyte-scale data the same way you'd work with a small sample.

## The lineage graph

As data flows through the lake, each transform emits its slice of a **lineage
graph**: `lineage_artifacts` (sources, snapshots, model outputs, table versions,
invalidation markers, ...) connected by typed `lineage_edges` (`trained-on`,
`selected-from`, `produced-model`, `invalidates`, ...). This graph is what makes
the closed-loop operations possible:

- **trace / impact** — what produced this artifact, and what depends on it.
- **invalidation** — mark an artifact bad (a corrupted source, a buggy embedding
  provider) and record it durably.
- **rebuild planning** — from an invalidation, compute the ordered set of things
  to recompute / resnapshot / retrain / re-evaluate.
- **evidence packs** — a durable, replayable bundle of everything a checkpoint
  depended on, for audit.

Lineage is **opt-in**: curation, training, evaluation, and search all work without
it; lineage adds traceability on top. The
[rebuild-loop journey](../journeys/rebuild-loop.md) walks the operational path end
to end.

## Where to go next

- [Architecture](architecture.md) — why Lance is the working set, the no-join
  model, blob-safe additive enrichment, and content-addressed IDs.
- [Storage and auth](storage-and-auth.md) — local, object-store, `db://`, and REST
  namespace lakes, and how credentials resolve.
- [OSS core vs. enterprise/plugin](oss-core-vs-enterprise.md) — where the open
  substrate ends and deployment-specific layers begin.
