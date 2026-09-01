# Bring an RLDS / TFDS dataset into the lake

Use this workflow when an Open-X Embodiment, DROID, or vendor dataset has
already been prepared as a TensorFlow Datasets version directory. Ingest turns
RLDS episodes and their nested steps into the same canonical runs, episodes,
observations, scenarios, events, and lineage used by the rest of the lakehouse.
The result is immediately available to curation, snapshot, training-loader, and
lineage workflows without making TensorFlow part of those downstream paths.

## Install the ingest reader

TensorFlow is intentionally optional. Install the maintained reader lane on a
supported Python host:

```bash
pip install 'lancedb-robotics[tfds]'
```

The older `[rlds]` extra remains a native-export conformance lane. New ingest
workloads should use `[tfds]`, which reads RLDS structure directly through TFDS
without requiring the archived `rlds` or Reverb packages.

## Start with inspect

Pass the generated TFDS **version directory** containing `dataset_info.json`,
feature metadata, and TFRecord shards—not a TFDS download/cache parent and not
this project's `rlds-tfds-style-v0` Parquet export.

```bash
lancedb-robotics inspect rlds ./bridge_dataset/1.0.0 --format text
```

JSON output is useful for automation:

```bash
lancedb-robotics inspect rlds ./bridge_dataset/1.0.0 --format json
```

Inspect reads TFDS metadata only. It reports the dataset name/version, split
names, declared episode and shard counts, and feature structure without opening
or hashing the TFRecord shards. Ingest performs the full streamed content hash
needed for portable, content-addressed IDs before it writes canonical rows.

## Ingest all splits or select a subset

Create or open the lake normally, then ingest the prepared directory:

```bash
lancedb-robotics ingest rlds ./bridge_dataset/1.0.0 \
  --lake ./robotics.lance
```

Repeat `--split` when only selected TFDS splits belong in this lake:

```bash
lancedb-robotics ingest rlds ./bridge_dataset/1.0.0 \
  --lake ./robotics.lance \
  --split train \
  --split validation
```

The adapter reads one physical TFRecord shard at a time, iterates each nested
episode step stream, and flushes canonical rows in bounded batches. Peak Python
state is bounded by one episode (including the observation IDs needed for its
scenario) plus one write batch, rather than total corpus size. Lower
`--batch-size` for unusually large per-step payloads. Final compaction, scalar
index maintenance, and safe version pruning are enabled by default and can use
the standard ingest flags.

RLDS `is_first` and `is_last` markers are validated, rather than guessed.
`is_terminal=false` on a final step remains distinguishable as a truncation.
Invalid RLDS episodes marked by their episode metadata are skipped. A malformed
episode fails the ingest and removes run-scoped partial batches so a corrected
retry converges cleanly.

## Map an Open-X variant

RLDS datasets commonly agree on boundary markers but vary in where they store
state, action, language, and time. Use dotted keys to describe those fields:

```bash
lancedb-robotics ingest rlds ./custom_robot/2.1.0 \
  --lake ./robotics.lance \
  --state-key proprio.joints \
  --action-key action.world_vector \
  --language-key observation.natural_language_instruction \
  --timestamp-key observation.timestamp
```

When no timestamp exists, `--fps 30` derives nanosecond timestamps from the
step ordinal. Without either a timestamp or FPS, the ordinal itself preserves
ordering but does not invent a physical sample rate. Nested numeric state and
action mappings are flattened in stable key order.

The equivalent Python API is:

```python
from lancedb_robotics.adapters.rlds_adapter import RldsFieldMapping
from lancedb_robotics.ingest import ingest_rlds
from lancedb_robotics.lake import Lake

lake = Lake.open("./robotics.lance")
report = ingest_rlds(
    lake,
    "./custom_robot/2.1.0",
    splits=["train"],
    mapping=RldsFieldMapping(
        state_key="proprio.joints",
        action_key="action.world_vector",
        timestamp_key="observation.timestamp",
        fps=30,
    ),
)
print(report.run_id, report.rows_added)
```

## Read directly from Google Cloud Storage

TFDS can open a prepared `gs://` version directory directly:

```bash
lancedb-robotics inspect rlds \
  gs://my-open-x-mirror/bridge_dataset/1.0.0 --format text

lancedb-robotics ingest rlds \
  gs://my-open-x-mirror/bridge_dataset/1.0.0 \
  --lake s3://my-curated-lake
```

The raw TFDS reader uses TensorFlow's Google Cloud Application Default
Credentials. Configure ADC in the process environment or workload identity.
Arbitrary `--source-storage-option` credentials cannot be forwarded into TFDS,
so the command rejects them for `gs://` instead of implying they are honored.
Lake storage credentials remain independent and continue to use the normal
lake options.

## What lands in the lake

- One content-addressed integration source and run. Moving byte-identical TFDS
  files to another path produces the same IDs and an audited no-op on re-ingest.
- One canonical episode and authored scenario per valid RLDS episode.
- One observation per RLDS step, with source split, physical shard, episode, and
  step coordinates preserved in provenance.
- Language instructions in canonical `task_id`/caption fields, numeric state and
  action vectors, and reward/discount/terminal semantics in `payload_json`.
- Large image/array fields in Lance blob storage. Small scalar metadata remains
  queryable without hydrating the heavy bytes.
- Inspect/ingest transform runs, run-boundary events, and lineage edges for audit
  and reproducibility.

The source identity hashes sorted relative artifact names and bytes, never the
absolute source root. IDs supplied by TFDS are retained as provenance only; they
are not trusted as stable canonical identity.

Current ingestion acquires a lake-resident compare-and-swap gate before cleanup
or canonical writes. A second process or container fails with an actionable
claim diagnostic instead of racing the first writer, and ordinary success or
failure releases the gate. A hard process crash deliberately leaves the gate
fail-closed. Durable per-shard checkpoints, leases, stale-claim inspection, and
audited recovery are tracked in backlog 0475 for production schedulers.
