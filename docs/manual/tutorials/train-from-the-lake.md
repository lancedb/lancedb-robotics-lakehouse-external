# Tutorial: from a raw log to a training batch

This is the shortest honest path from a robot log you have never seen to a batch
of tensors a policy can train on — and then the same lake, unchanged, opened a
second time as a real `LeRobotDataset`.

Everything here runs on a clone of this repository against a bundled 9 KB
fixture, so you can follow it start to finish without a corpus, a bucket, or a
network connection. The [variants](#variants-your-own-data) at the end swap the
fixture for your own data.

**What you will have built by the end**

- a lake holding one teleoperation run: 240 raw observations, two labeled
  episodes, and 80 synchronized 20 Hz ticks;
- **Path A** — a `torch.utils.data.DataLoader` over those ticks, with real
  worker processes, a deterministic shuffle, and mid-epoch resume;
- **Path B** — a published, version-pinned LeRobot view of the *same* ticks,
  opened as a `LeRobotDataset` and sampled with an ACT-style action chunk.

The two paths are the point. Nothing is exported, converted, or copied between
them: one set of canonical rows serves a native loader and first-party LeRobot
tooling at the same time.

## Before you start

```bash
uv sync --extra dev --extra torch
```

That is enough for the whole of Path A. Path B's *publish* step also runs on
this install; only the final step — opening the view through upstream lerobot's
own `LeRobotDataset` — needs more, and
[Path B says exactly what](#3-open-it-as-a-real-lerobotdataset) when it gets
there.

Commands below are written as `lancedb-robotics …`. Inside a clone without an
activated virtualenv, prefix them with `uv run`.

---

## Build the lake

Five commands. Run them from the repository root.

```bash
lancedb-robotics lake init --lake ./robot.lance
lancedb-robotics ingest mcap tests/fixtures/teleop.mcap --lake ./robot.lance
lancedb-robotics quality validate --lake ./robot.lance --profile tests/fixtures/teleop-profile.json
lancedb-robotics episodes import-intervals --lake ./robot.lance --file tests/fixtures/teleop-episodes.jsonl
lancedb-robotics align create teleop_20hz --lake ./robot.lance --rate-hz 20 --run-id run-be4244bb87aeb236 --stream /joint-positions --stream /gripper --stream /action
```

The rest of this section explains what each one did. Skip ahead to
[Path A](#path-a-a-native-torch-dataloader) if you would rather see the payoff
first and come back.

### 1. Ingest

`tests/fixtures/teleop.mcap` is four seconds of a 6-DoF arm at 20 Hz: observed
joint positions on `/joint-positions`, gripper aperture on `/gripper`, and the
commanded joint targets on `/action`. Ingest decodes each message, types it, and
writes one `observations` row per message — 240 in total:

```console
$ lancedb-robotics ingest mcap tests/fixtures/teleop.mcap --lake ./robot.lance
run: run-be4244bb87aeb236 (ingested)
...
  observations +240
```

That run id is worth a moment. It is **content-addressed**: a digest of the log's
bytes, not of its path or filename. Copy the fixture somewhere else, rename it,
ingest it on another machine — you get `run-be4244bb87aeb236` every time. That is
why this tutorial can hand you the id as a literal to paste in step 4, and why
the episode manifest in step 3 can reference it.

Ingest also derived typed vectors from the payloads. `/joint-positions` and
`/action` both carry a `RobotState` message, so each lands a 6-float
`state_vector`; `/gripper` carries a `GripperState` and lands a 1-float one.
Note that the *commanded* stream also populates `state_vector` — extraction
types a message by its schema, and it has no way to know a topic is a command.
Observed and commanded get separated later, by stream name, in step 4 and in
Path B's mapping.

### 2. Validate

A quality profile says what a healthy run of this kind looks like. The built-in
`demo` profile describes a different fixture, so this tutorial passes its own:

```json
{
  "name": "teleop",
  "required_topics": [
    { "topic": "/joint-positions", "min_count": 20 },
    { "topic": "/action", "min_count": 20 }
  ],
  "decodable_topics": ["/joint-positions", "/gripper", "/action"],
  "require_monotonic": true,
  "require_overlap": true
}
```

`quality validate` checks required topics, decodability, timestamp
monotonicity, stream overlap, and byte integrity, then writes the verdict back
onto the run. A run that fails is **quarantined** and downstream steps can
exclude it. Our run passes:

```console
run run-be4244bb87aeb236: passed
runs: 1 validated, 1 passed, 0 failed, 0 quarantined
```

Writing a profile for your own logs is the same shape: a JSON file, passed to
`--profile`.

### 3. Label the episodes

An MCAP file is a stream of messages; it has no notion of "this is where the
first demonstration ends." Something has to say so. `episodes import-intervals`
takes that boundary list from wherever you already keep it — a labeling tool, a
teleop session log, a spreadsheet — as JSONL or CSV:

```jsonl
{"run_id": "run-be4244bb87aeb236", "external_id": "teleop-take-1", "task_id": "pick the cube", "outcome": "success", "from_timestamp_ns": 1700000000000000000, "to_timestamp_ns": 1700000001950000000}
{"run_id": "run-be4244bb87aeb236", "external_id": "teleop-take-2", "task_id": "pick the cube", "outcome": "success", "from_timestamp_ns": 1700000002000000000, "to_timestamp_ns": 1700000003950000000}
```

Two takes of the same task, each two seconds. The import validates every
interval against the run's real bounds, refuses overlaps, keeps your
`external_id` so you can trace a row back to the tool that produced it, and tags
the observations that fall inside each one.

```console
boundary source: intervals
episodes: 2
frames tagged: 240
```

Episodes matter more than they look. They are what stops a temporal window in
Path B from bleeding across a cut between two unrelated demonstrations.

### 4. Align

Raw rows are not trainable. Each topic arrives on its own schedule, with its own
vector width, and a policy needs *one row per instant* across every stream it
consumes. That reconciliation is what `align create` does:

```bash
lancedb-robotics align create teleop_20hz --lake ./robot.lance --rate-hz 20 --run-id run-be4244bb87aeb236 --stream /joint-positions --stream /gripper --stream /action
```

It lays a 20 Hz query clock over the run and, for each tick, resolves every
declared stream to a value — recording per-stream whether that value was exact,
interpolated, out of tolerance, or missing, along with the source rows it came
from. The result is 80 rows in `aligned_ticks`:

```console
view: teleop_20hz
alignment: aln-...
rows: 80
streams: /joint-positions, /gripper, /action
confidence: 1.000000
```

`--run-id` is not optional here even though the lake holds exactly one run: the
LeRobot facade in Path B requires a single-run alignment and will refuse one
built across several. Passing it now saves rebuilding the alignment later.

---

## Path A: a native torch DataLoader

`lake.training.aligned_dataset` opens those ticks as a random-access, shuffled,
resumable training set. No new file layout, no shard directory, no export:

```python
from lancedb_robotics import Lake

lake = Lake.open("./robot.lance")

dataset = lake.training.aligned_dataset(
    "teleop_20hz",
    streams=["/joint-positions", "/gripper", "/action"],
    require_streams=True,     # skip any tick missing one of them
    shuffle=True,
    shuffle_seed=17,
)

len(dataset)                  # 80
sample = dataset[0]
sorted(sample["streams"])     # ['/action', '/gripper', '/joint-positions']
sample["streams"]["/joint-positions"]["value"]   # [6 floats]
```

Each sample carries the per-stream values plus the masks (`valid`, `missing`,
`interpolated`, `out_of_tolerance`), the quality flags, and the lineage of the
source rows — so a training loop can *see* which frames were interpolated rather
than discovering it in a loss curve. `require_streams=True` filters incomplete
ticks out of the read plan entirely, upstream of any of that.

### Real workers

```python
loader = dataset.torch_dataloader(
    batch_size=8,
    num_workers=2,
    adapter="iterable",
    multiprocessing_context="spawn",
)

for batch in loader:
    batch["tick_index"]       # tensor of 8 tick indices
    break
```

`multiprocessing_context="spawn"` is not decoration. Lance readers are **not
fork-safe**, so forked workers can inherit a handle they must not share; a spawn
worker rebuilds the dataset from its configuration in a fresh interpreter
instead. With that set, the two workers partition the epoch: across the whole
loader you see all 80 ticks, each exactly once.

### Deterministic shuffle and resume

The shuffle is a seeded permutation of the read plan, not a random walk, which
makes an epoch reproducible and *resumable*:

```python
def order(**kwargs):
    ds = lake.training.aligned_dataset(
        "teleop_20hz",
        streams=["/joint-positions", "/gripper", "/action"],
        require_streams=True,
        shuffle=True,
        **kwargs,
    )
    return [s["tick_index"] for s in ds]

order(shuffle_seed=17) == order(shuffle_seed=17)          # True  -- reproducible
order(shuffle_seed=17) == order(shuffle_seed=99)          # False -- seed matters
order(shuffle_seed=17, resume_from=40) == order(shuffle_seed=17)[40:]   # True
```

`resume_from` is *global*, not per-worker: a job that died 40 frames into an
epoch restarts on frame 41 of the same order, whatever worker count it comes
back with.

---

## Path B: the same ticks, as a real LeRobotDataset

Path A read the lake with this project's own loader. Path B hands the exact same
aligned ticks to first-party [LeRobot](https://github.com/huggingface/lerobot)
tooling, with no export step and no separate metadata directory to distribute.

### 1. Publish a view

Publishing is the deliberate act that makes lake data visible to LeRobot
clients. It resolves one facade configuration against the lake **at pinned table
versions**, captured before anything is read:

```bash
lancedb-robotics train view publish --lake ./robot.lance \
  --repo-id demo/teleop-v1 --fps 20 --name teleop_20hz \
  --state-stream /joint-positions --state-stream /gripper \
  --action-stream /action --robot-type so100
```

```console
published view: lrv-...
repo_id: demo/teleop-v1
frames: 80  episodes: 2
files: 4 (13037 bytes)
```

This is where observed and commanded finally part ways. `--state-stream` is
repeatable and **ordered**: `observation.state` is the concatenation of
`/joint-positions` (6) and `/gripper` (1), so every frame is exactly 7 floats
wide, while `action` is `/action`'s 6. Composing a fixed-width vector from named
streams is precisely what keeps heterogeneous per-topic widths from producing
ragged frames.

Publishing also derived `meta/stats.json` — the per-feature `mean`/`std`/
`min`/`max`/`count` and `q01`–`q99` quantiles that LeRobot policies build their
normalization buffers from — in one bounded-memory pass over the same ticks.

The printed view id is a content digest over the definition *and* the pinned
table versions. Republishing an unchanged definition over an unchanged lake is
an idempotent no-op; publishing after the lake has moved creates a *new* view
while every previously published one keeps serving exactly the frames it served
at publish time.

### 2. Look at what you published

```bash
lancedb-robotics train view list --lake ./robot.lance --repo-id demo/teleop-v1
lancedb-robotics train view materialize ./inspect-here --lake ./robot.lance --repo-id demo/teleop-v1
```

`materialize` writes the derived `meta/` directory to a path you can read with
your own eyes — `info.json`, the chunked episode index, and `stats.json` —
verifying each file's sha256 on the way out. Nothing downstream needs this step;
it is here because seeing the manifest once makes the rest obvious.

### 3. Open it as a real `LeRobotDataset`

This is the one step with an environment caveat, and it is worth stating plainly
rather than burying.

The reader this project ships is a `BaseDatasetReader` against lerobot's
**storage-format registry**, and it declares itself in the
`lerobot.dataset_readers` entry-point group. No PyPI release of lerobot reads
that group yet ([huggingface/lerobot#4576](https://github.com/huggingface/lerobot/pull/4576)
is the upstream change). So today this step needs the dev-only, commit-pinned
extra, and one registering import:

It also needs its **own environment**. The `lerobot` and `lerobot-main-dev`
extras both pin `lerobot` from different sources — a PyPI release floor versus
one git commit — so `pyproject.toml` declares them mutually exclusive and they
cannot be resolved together. Build a separate venv for this step (lerobot
requires Python 3.12+):

```bash
uv venv .venv-lerobot --python 3.12
VIRTUAL_ENV=.venv-lerobot uv pip install -e ".[media,lerobot-main-dev]"
```

Then, in that environment:

```python
import lancedb_robotics.lerobot_facade.dataset_reader   # registers "lancedb_robotics"
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("demo/teleop-v1", root="file:///absolute/path/to/robot.lance")

len(dataset)                                       # 80
dataset.meta.stats["observation.state"]["mean"]    # from the lake, 7 values
item = dataset[0]
item["observation.state"].shape                    # (7,)
item["action"].shape                               # (6,)
```

Once upstream entry-point discovery ships in a release, the import disappears
and installing `lancedb-robotics` is enough for any lerobot process to resolve
the format. Until then, treat the import as required. The
[published-views journey](../journeys/lerobot-published-views.md) tracks that
retirement.

Everything *before* this step — ingest, validation, episodes, alignment,
publish, and the whole of Path A — works on released lerobot, or with no lerobot
installed at all.

### 4. Train on an action chunk

ACT-style policies do not predict one action; they predict a *chunk* of them.
Upstream's `delta_timestamps` asks for that window and the reader assembles it:

```python
dataset = LeRobotDataset(
    "demo/teleop-v1",
    root="file:///absolute/path/to/robot.lance",
    delta_timestamps={"action": [i / 20 for i in range(8)]},   # 8 steps @ 20 fps
)

item = dataset[0]
item["action"].shape        # (8, 6) -- stacked window
item["action_is_pad"]       # bool (8,) -- True where t+delta left the episode
```

Two properties are worth knowing before you trust the output:

- **A window never crosses an episode boundary.** Ask for 8 steps starting at
  frame 36 of a 40-frame episode and the last four clamp to frame 39, repeating
  it, with `action_is_pad` marking exactly those positions. This is why step 3 of
  the lake build mattered: with the wrong episode boundaries, a chunk quietly
  trains on the start of an unrelated take.
- **Reads stay batch-shaped, not window-shaped.** A batch hydrates the
  deduplicated *union* of its window rows in one aligned-tick read, so an 8-step
  chunk does not cost 8× the I/O, and a window over `action` alone never
  multiplies camera decode.

Offsets must be multiples of `1/fps`. A `delta_timestamps` key the view's mapping
cannot serve raises at construction rather than returning something plausible.

### 5. Watch an episode play

```console
$ lancedb-robotics-lerobot-viz \
    --repo-id demo/teleop-v1 \
    --root file:///absolute/path/to/robot.lance \
    --episode-index 0 \
    --display-mode foxglove
# then connect the Foxglove app to ws://127.0.0.1:8765
```

Run it from the same `.venv-lerobot` as steps 3 and 4. This console script is a
temporary shim, not API. Lerobot's own
`lerobot-dataset-viz` constructs the dataset directly, so there is no seam for
the registering import; the shim adds it, localizes a URI `--root` that the
upstream CLI would otherwise mangle into a `Path`, and forwards everything else
unchanged. It retires alongside the import in step 3.

---

## Variants: your own data

**A public LeRobot dataset by Hugging Face repo id.** Needs network and
`uv sync --extra lerobot`; everything after ingest is identical, minus the
episode-import step, because LeRobot datasets already carry episode boundaries:

```bash
lancedb-robotics inspect lerobot lerobot/pusht --format json
lancedb-robotics ingest lerobot lerobot/pusht --lake ./pusht.robot.lance
```

**Your own MCAP or ROS bag.** Substitute the file in step 1 and use
`ingest rosbag` for `.bag` / `.db3` containers. You will need your own quality
profile, your own episode intervals, and your own stream names in
`align create` — which is exactly the three things this tutorial made explicit
rather than hiding.

**An object-store lake.** Pass an `s3://`, `gs://`, `az://`, or `db://` URI as
`--lake` with `--storage-option` or `--auth-ref`. Raw bytes stay where they
live; credentials are resolved in memory and never written to lake tables.

## Where to go next

- [Publish a version-pinned LeRobot view](../journeys/lerobot-published-views.md)
  — the reference treatment of Path B: catalog paging, retirement, retention,
  and what `lake maintain` does to protect a published view's pinned versions.
- [Bring a LeRobot dataset into the lake](../journeys/lerobot-ingest.md) — the
  full ingest surface behind the variant above.
- [The training loader report contract](../concepts/training-loader-report.md) —
  what a loader reports back about what it actually read.
- [CLI reference](../reference/cli.generated.md) and
  [Python API reference](../reference/api.generated.md) for every option the
  five commands above did not use.
