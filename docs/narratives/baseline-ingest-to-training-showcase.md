# Baseline showcase: raw robot logs → searchable, snapshot-ready training data

This is the full baseline story for `lancedb-robotics`, run end to end as a
single integration scenario. It takes one raw robot log and walks it through
every demo-spine command group until it is validated, searchable, frozen into a
reproducible training slice, previewed as training samples, and projected back
out to MCAP clips — **without ever losing the path back to existing replay
tools**.

> Raw robot logs become validated, searchable, snapshot-ready training data
> without losing replay/export paths.

The commands below are kept in lock-step with the runnable integration test in
[`tests/test_integration_showcase.py`](../../tests/test_integration_showcase.py):
the test executes the same command sequence and asserts the summaries quoted
here, so this narrative cannot silently drift from the code.

## The problem

A robotics team has terabytes of raw logs (MCAP, ROS bags, fleet uploads). The
useful training data — the few seconds around a missed pick, a protective stop,
an interesting maneuver — is buried inside those logs, described by no index.
Today teams stitch together one-off converters, a vector database, a metadata
warehouse, label exports, and replay tools just to ask "show me clips like
this." `lancedb-robotics` makes the raw log itself queryable from one surface
and keeps lineage from each answer back to the original bytes.

## The raw input

One small, deterministic fixture stands in for a real log:
[`tests/fixtures/sample.mcap`](../../tests/fixtures/sample.mcap) — 5 messages
across 2 topics over 0.2 seconds:

- `/imu` — `json` encoding, `sample.Imu` schema, 3 messages, **decodable** with
  the stdlib.
- `/camera/front` — `protobuf` encoding, `sample.CompressedImage` schema, 2
  messages, **no decoder** installed (capability is probed, never assumed).

## Run it

```bash
lancedb-robotics lake init --lake ./demo.robot.lance
lancedb-robotics inspect mcap ./fixtures/sample.mcap --format text
lancedb-robotics ingest mcap ./fixtures/sample.mcap --lake ./demo.robot.lance
lancedb-robotics quality validate --lake ./demo.robot.lance --profile demo
lancedb-robotics scenarios create --lake ./demo.robot.lance --window 50ms
lancedb-robotics scenarios enrich --lake ./demo.robot.lance
lancedb-robotics search hybrid "imu observations" --lake ./demo.robot.lance
lancedb-robotics dataset snapshot create --lake ./demo.robot.lance --from-search last --name demo-v1
lancedb-robotics train preview torch --lake ./demo.robot.lance --snapshot demo-v1
lancedb-robotics export mcap --lake ./demo.robot.lance --snapshot demo-v1 --out ./demo-clips
```

> Two notes if you are coming from an earlier draft of the showcase script:
> `inspect mcap` takes `--format text` or `--format json` (there is no `table`
> format), and `scenarios enrich` is a required step before search — it is what
> writes the summary text and embedding vectors that text/vector/hybrid search
> read.

## Step by step

Each step names the canonical lake tables it touches and the summary it prints.
Every derived row carries a `transform_id`, and every command that changes the
lake records a row in `transform_runs`, so the whole pipeline is auditable.

### 1. `lake init` — create the canonical tables

Creates the seven canonical tables (idempotently). Schema versions are persisted
in Arrow metadata and reported back:

```
  integration_sources (v1)
  runs (v1)
  observations (v1)
  events (v1)
  scenarios (v3)
  dataset_snapshots (v1)
  transform_runs (v1)
```

### 2. `inspect mcap` — look before you move data

Reads the file and reports topics, encodings, schemas, counts, time range, and
decode capability **without ingesting anything**. No lake tables are touched.

```
  5 messages, 2 topics, 0.2s (1700000000000000000 .. 1700000000200000000)
  /camera/front	protobuf	sample.CompressedImage	2 msgs	no decoder
  /imu	json	sample.Imu	3 msgs	decodable
```

### 3. `ingest mcap` — raw log → canonical rows

Registers the source and ingests messages into canonical rows, recording pointer
provenance (URI / channel / log-time / sequence) so the original bytes stay
recoverable. **Tables written:** `integration_sources`, `runs`, `observations`,
`events`, `transform_runs` (an `inspect` and an `ingest` lineage row).

```
rows added:
  integration_sources +1
  runs +1
  observations +5
  events +2
  transform_runs +2
observations by topic:
  /camera/front	2
  /imu	3
```

### 4. `quality validate` — the gate before search

Validates each run against the `demo` profile (required topics + minimum counts,
strictly monotonic timestamps, time-range overlap, decode capability). It writes
the verdict to `runs.quality_flags`, flags failing runs `quarantined` so later
steps can exclude them, and records one `quality` lineage row per run. The
fixture passes cleanly:

```
profile: demo
runs: 1 validated, 1 passed, 0 failed, 0 quarantined
```

A failing run would instead be marked `quarantined` (and the command exits `2`),
keeping bad training candidates out of search results and snapshots.

### 5. `scenarios create` — raw stream → searchable windows

Cuts each run into deterministic fixed-duration windows anchored to the run's
start/end, preserving temporal lineage back to the source observations.
**Tables written:** `scenarios`, `transform_runs` (`scenario-windowing`).

```
window: 50ms (50000000 ns)
topics: all topics
partial final window: included
runs: 1
scenarios: 4 created
```

### 6. `scenarios enrich` — stable semantic handles

Attaches deterministic, local demo captions and embedding vectors to each
scenario so they can be searched before any real perception model exists. The
caption is template text; the embedding is a hashed unit vector (a real model
plugs into the same provider contract later). Enrichment also builds the
persistent FTS index over `scenarios.summary`, so search reuses the index instead
of rebuilding it on each query. **Tables written:** `scenarios` (adds `summary`
+ an `embedding` vector column), `transform_runs` (`enrichment`).

```
caption provider: demo-template-v1
embedding provider: demo-hash-v1 (dim 16)
scenarios enriched: 4
fts index: built (FTS over scenarios.summary, 4 rows)
```

### 7. `search hybrid` — one query surface

Searches the same canonical `scenarios` rows by full text (BM25 over the
summaries), vector similarity (over the demo embeddings), and a hybrid of the
two fused by reciprocal rank fusion — all native LanceDB, with transparent score
components and a source link back to the raw log on every hit. Ranking is made
deterministic with an explicit `scenario_id` tie-break.

```
mode: hybrid
query: "imu observations"
results: 4
```

The three `/imu` windows rank above the camera-only window because the query
terms (`imu`, `observations`) appear in their summaries; each result shows its
`text=` (BM25), `vector_distance=`, and `relevance=` (RRF) components so the
ranking is never a black box.

### 8. `dataset snapshot create` — freeze a reproducible slice

Freezes the recorded search result into a `dataset_snapshots` row **without
repacking the corpus**: it pins the selected `scenario_id`s, the originating
search spec, a deterministic train/val/test split (assigned by run id to avoid
leakage), and the **source table versions** so the slice can be re-read exactly.
**Tables written:** `dataset_snapshots`, `transform_runs` (`dataset-snapshot`).

```
dataset: demo-v1 (ds-…)
tag: demo-v1
source: search (hybrid "imu observations")
scenarios: 4
split by run: train=4 val=0 test=0
table versions: scenarios@… runs@… observations@…
transform: tfm-snapshot-…
```

All four windows belong to a single run, so a run-keyed split keeps them
together in one split (avoiding train/val leakage across a run). Which split
they land in is a deterministic function of the run id — and since the run id is
content-addressed from the file bytes (not the source path), the named split is
reproducible across machines: this fixture always lands in `train`.

### 9. `train preview torch` — feed training, no conversion step

Reads the snapshot's scenario rows **as of their pinned table versions** and
returns deterministic sample dictionaries. PyTorch is optional: with it
installed the samples adapt to tensors; without it the dict preview still works
and says so.

```
snapshot: demo-v1 (ds-…)
tag: demo-v1
split by: run
scenarios: 4
columns: scenario_id, split, summary, topics, embedding
framework: torch not installed (showing dict preview); install lancedb-robotics[torch] for tensor batches
```

### 10. `export mcap` — project back into the replay loop

Projects the selected windows back out as one lossless MCAP clip per scenario
(raw messages copied verbatim, so they open in any MCAP tool), plus an
`export_manifest.json` recording row IDs, time windows, source URIs, lossiness,
and optional Foxglove/Rerun replay links. When a source is unreachable or an
adapter cannot slice, the manifest still records the plan with a clear skip
reason. No lake tables change; a `transform_runs` (`export`) row records the
lineage.

```
format: mcap
clips: 4 (exported 4, skipped 0, planned 0)
```

## End-to-end lineage

After the run, `transform_runs` holds one row per pipeline stage — the complete
audit trail from raw bytes to clips:

```
inspect · ingest · quality · scenario-windowing · enrichment · search · dataset-snapshot · export
```

## How this differs from MCAP / Foxglove / Rerun

`lancedb-robotics` is **not** another log format or viewer, and it does not
replace them:

- **MCAP** is the on-disk log container. The lakehouse *ingests* MCAP into
  queryable canonical rows and *exports* selections back to MCAP clips — MCAP
  stays the interchange format.
- **Foxglove / Rerun** are visualization and replay tools. The lakehouse emits
  `external_links` pointing chosen clips back into them — they stay the way
  humans watch the data.
- What none of those provide, and what this layer adds, is the **selection,
  search, versioning, and lineage surface**: one queryable place that joins
  metadata, full text, vectors, time windows, and provenance, turns a query into
  a reproducible dataset snapshot, and traces every derived row back to the
  bytes it came from.

The lakehouse is the selection-and-lineage layer; MCAP, Foxglove, and Rerun
remain in the loop.

## Run it as a test

This whole sequence runs as one deterministic integration test:

```bash
uv run pytest tests/test_integration_showcase.py
```

It asserts the summaries above, the final row counts, the full set of
`transform_runs` kinds, and that the exported clips are valid MCAP — and it
checks that the command block in this document stays in sync with the commands
the test runs.
