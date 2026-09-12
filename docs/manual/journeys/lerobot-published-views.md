# Journey: publish a version-pinned LeRobot view (with normalization stats)

**Scenario.** A robot-learning team keeps their fleet's data in one lake and
trains with first-party [LeRobot](https://github.com/huggingface/lerobot)
tooling. They want `LeRobotDataset("acme/pick-place-v1", root="s3://acme/robot.lance")`
to *just work* — no export step, no hand-distributed metadata directory — and
they need it to be **reproducible**: a dataset opened mid-training must keep
serving exactly the frames it served at publish time, even while ingest,
re-alignment, and quality flags keep advancing the live lake underneath.

A *published view* is the deliberate act that makes lake data visible to
LeRobot clients. Publishing:

- resolves one facade configuration (alignment + canonical vector mapping +
  quality policy + episode selection) against the lake **at pinned table
  versions**, captured before anything is read;
- derives the LeRobot `meta/` manifest — `info.json`, the chunked episode
  index, and **`meta/stats.json`**, the per-feature normalization statistics
  (`mean`/`std`/`min`/`max`/`count` plus `q01`–`q99` quantiles) LeRobot
  policies build their normalization buffers from — in one bounded-memory
  pass;
- stores the definition and the derived files in two canonical tables
  (`lerobot_views`, `lerobot_view_files`), so any client materializes them
  through the same Lance connection it already has.

The flow is: **publish → open from lerobot → advance the lake without fear →
train with temporal windows**.

## 1. Publish the view

```bash
lancedb-robotics train view publish --lake ./robot.lance \
  --repo-id acme/pick-place-v1 --fps 20 --name pick_place_alignment \
  --state-stream /joints --state-stream /gripper \
  --action-stream /action --camera-stream /cam/wrist \
  --robot-type so100
```

```python
from lancedb_robotics.lerobot_facade import CanonicalVectorMapping, publish_view

published = publish_view(
    lake,
    repo_id="acme/pick-place-v1",
    fps=20,
    name="pick_place_alignment",
    mapping=CanonicalVectorMapping(
        state_streams=("/joints", "/gripper"),
        action_streams=("/action",),
        camera_streams=("/cam/wrist",),
    ),
    robot_type="so100",
)
print(published.view_id, published.total_frames, published.table_versions[:2])
```

The `view_id` is a content digest over the definition **and** the pinned table
versions: republishing an unchanged definition over an unchanged lake is an
idempotent no-op, and publishing after the lake advanced creates a *new* view
while every previously published one stays intact and reproducible.

Statistics ride the same publish pass. Vector and scalar features stream
through the aligned-tick batch reader (bounded memory, no camera decode);
camera features decode a bounded, deterministic sample (`linspace`,
`min(max(100, N^0.75), 10000)` frames, large frames stride-downsampled) whose
parameters are recorded on the view row, so the numbers are auditable. Camera
stats require the `media` extra (PIL); publishing a camera-mapped view without
it fails loudly rather than shipping a view whose `meta.stats` is silently
incomplete. `--no-stats` skips the pass explicitly — policies then cannot
build normalization buffers from the view.

## 2. Open it from lerobot

A client points at the lake — an object-store URI, or `file:///…` for a local
lake — and the newest view published under that `repo_id` resolves
automatically. From Python, register the adapter first (one import):

```python
import lancedb_robotics.lerobot_facade.dataset_reader  # registers "lancedb_robotics"
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("acme/pick-place-v1", root="s3://acme/robot.lance")
dataset.meta.stats["observation.state"]["mean"]  # populated, from the lake
```

The package also declares the format in the `lerobot.dataset_readers`
entry-point group (see `pyproject.toml`). No released lerobot reads that group
yet; once upstream entry-point discovery
([huggingface/lerobot#4576](https://github.com/huggingface/lerobot/pull/4576))
ships in a release, the manual import above becomes unnecessary — installing
`lancedb-robotics` is enough for any lerobot process, console scripts
included, to resolve the format with no configuration.

### Foxglove playback (interim shim)

Until that discovery ships, lerobot's own console scripts have no seam for
the registering import — `lerobot-dataset-viz` constructs the dataset
directly — so Foxglove playback goes through a **temporary** shim. It
registers the format, localizes a URI `--root` into the per-view cache
(the upstream CLI parses `--root` as a `Path`, which mangles URI schemes
before its own remote-root detection can run), and hands everything else
unchanged to `lerobot-dataset-viz`:

```console
$ lancedb-robotics-lerobot-viz \
    --repo-id acme/pick-place-v1 \
    --root file:///path/to/robot.lance \
    --episode-index 0 \
    --display-mode foxglove
# then connect the Foxglove app to ws://127.0.0.1:8765
```

The shim needs a lerobot with the storage-format registry (the dev-only,
commit-pinned `lerobot-main-dev` extra — no PyPI release carries it yet) and
fails with exactly that instruction otherwise. It is interim wiring, not API:
backlog item 0510 retires it — together with the manual import —
once a lerobot release discovers the entry-point group.

`localize_root` opens the lake, reads the view's file rows, verifies each
file's sha256, and materializes `meta/` into a per-view cache directory
(atomic temp-dir rename — N concurrent DataLoader workers converge on one
complete copy). The cache key includes the `view_id`, so re-publishing never
invalidates a directory another run is still using. Pass `revision=<view_id>`
to pin an exact published view instead of the newest.

## 3. Advance the lake without fear

Every read the reopened facade makes goes through a pinned lake: each
canonical table is checked out at the version recorded at publish time. Two
opens of the same view return identical frames no matter how far the live
lake has advanced. Two guards (mirroring upstream's own Lance backend) turn
any residual drift into a loud failure instead of silent index skew:

- the reopened facade must resolve exactly the frame count the manifest
  recorded (`ManifestDriftError` otherwise — e.g. a legacy unpinned manifest
  after new ingest);
- episode ranges must tile `[0, total_frames)` exactly at reader
  construction, so `EpisodeAwareSampler` indices always agree with the frames
  served.

A pinned version that no longer exists (pruned by retention/compaction, or a
backend without version checkout) raises `StaleViewVersionError` naming the
table and the remedy: re-publish the view.

That error should never arise from routine maintenance: `lake maintain` tags
every table version pinned by a non-retired published view (the same managed
pin tags that protect snapshot and lineage pins) before its version cleanup
runs, so pruning skips them. `train view readiness` — also a `lake maintain`
report section — classifies every view-pinned `(table, version)` as
`protected` / `unprotected` / `pruned` / `unreadable`, rolls that up per view,
and states the backend's pinned-open conformance (`supported` locally and on
object stores; `capability-gated` on a `db://` remote without version
checkout, where a pinned open is a typed error, never a silent latest-read;
`unavailable` where a namespace manages versioning).

## 4. Train with temporal windows (`delta_timestamps`)

Policies with horizons — ACT-style action chunking (`action[t..t+k]`),
observation history (`observation[t-1, t]`) — configure upstream's
`delta_timestamps` and the reader assembles the windows:

```python
dataset = LeRobotDataset(
    "acme/pick-place-v1",
    root="s3://acme/robot.lance",
    delta_timestamps={"action": [i / 20 for i in range(50)]},  # 50-step chunk @ 20 fps
)
item = dataset[0]
item["action"].shape        # (50, action_dim) — stacked window
item["action_is_pad"]       # bool (50,) — True where t+delta left the episode
```

Semantics match upstream's own Lance backend exactly: offsets must be
multiples of `1/fps` (± `tolerance_s`); each window member clamps to its
episode's bounds (a window never bleeds into a neighboring episode — the
boundary frame repeats instead); `<key>_is_pad` marks exactly the positions
whose unclamped offset fell outside the episode. Windowed vector features
stack to `(k, dim)`; a windowed camera stacks decoded frames to
`(k, C, H, W)`.

Read costs stay batch-shaped, not window-shaped: each batch hydrates the
*deduplicated union* of window rows in one aligned-tick read, and camera
decode happens only where the batch actually needs pixels — a window over
`action` alone never multiplies camera decode by the chunk size, and a
windowed camera reuses the GOP-batched decoder across overlapping windows.

Two loud guards replace upstream's silent edge behaviors: a `delta_timestamps`
key the view's mapping cannot serve raises `ValueError` at construction, and a
window member whose feature genuinely cannot be hydrated (an unaligned tick
under a permissive quality policy, a camera with no encoded video) raises
`WindowHydrationError` instead of fabricating padding.

## Inspecting and operating

```bash
lancedb-robotics train view list --lake ./robot.lance --repo-id acme/pick-place-v1
lancedb-robotics train view list --lake ./robot.lance --page-size 100   # keyset paging
lancedb-robotics train view materialize ./inspect-here --lake ./robot.lance \
  --repo-id acme/pick-place-v1
lancedb-robotics train view compact-catalog --lake ./robot.lance --dry-run
lancedb-robotics train view readiness --lake ./robot.lance
lancedb-robotics train view retire lrv-0123456789abcdef --lake ./robot.lance --dry-run
```

The plain listing is a bounded convenience surface: past 10k matching views it
raises loudly instead of degrading. `--page-size`/`--cursor` (the
`list_view_pages` API) walk a catalog of any size with a stable newest-first
keyset cursor — views published mid-walk never shift later pages.

Resolving the newest view for a `repo_id` — what every
`LeRobotDataset(root=<lake>)` open does — reads one `lerobot_view_latest`
pointer row, maintained newest-wins by publish, instead of scanning headers;
pre-0507 lakes without a pointer fall back to a backend-ordered top-1 read and
only then to the guarded scan.

Truly simultaneous identical publishes can land benign duplicate catalog rows
(reads deduplicate by key). `train view compact-catalog` — also run by `lake
maintain` — collapses them, keeping the newest copy per key and asserting no
key loses its last row; the same pass reconciles latest-view pointers a crash
between a publish's catalog write and its pointer update left stale.

`lake maintain` builds the view catalog's scalar indexes (`repo_id`,
`view_id`, `file_id`) alongside every other managed predicate index. Old
lakes gain the three catalog tables with one `lancedb-robotics lake init`.

### End of life: retire and retention

`train view retire <view_id>` deletes a view's header, file rows, and pointer
rows. The view the `lerobot_view_latest` pointer targets — or the newest view
for its `repo_id` — is refused with a typed error unless `--force`, because
retiring it changes what every `LeRobotDataset(root=<lake>)` open resolves;
with `--force` the pointer is re-pointed at the newest remaining view. Retire
is idempotent: re-running after a crash converges, and once a view's header
and pins are gone, `lake maintain`'s version cleanup is free to reclaim the
table versions only that view was holding.

A retention *policy* (age plus retain-N-newest-per-repo, defaults 90 days /
5 views, the newest view per repo never eligible) is **report-only by
default**: every `lake maintain` run reports the current candidates, and
nothing is deleted until an operator passes
`--apply-lerobot-view-retention` — published views are reproducibility
contracts, so enforcement is a deliberate act, not a side effect.

Publish writes file rows first and the header last (its crash-safety order),
so a crashed publish leaves invisible headerless file rows; `lake maintain`
reclaims them once they age past a grace window (24 h) that keeps in-flight
publishes safe.

**Audit note.** The view row carries the full definition JSON, the pinned
version of every canonical table, totals, the camera-stats sampling
parameters, `created_by`, and `created_at`; the file rows carry per-file
sha256 and sizes. Materialization refuses hash mismatches outright.

**What's next.** Enterprise `db://` version checkout stays capability-gated —
`view_pin_conformance` / `train view readiness` state the posture per backend
and a pinned open is always a typed error or a working pin (never a silent
latest-read); the registering import and the viz shim retire once upstream
entry-point discovery ships (backlog 0510).
