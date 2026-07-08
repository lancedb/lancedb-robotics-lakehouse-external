# Episode creation and re-creation

Backlog 0029 made episodes first-class lake rows. The important distinction is:
an **episode** is the training/curation unit with stable frame indices, while a
**scenario** remains an analytical window or curation result. Scenarios can feed
snapshots, and old snapshots can still be interpreted as episodes, but only the
episode creation APIs write physical rows to `episodes`/`videos` and annotate
`observations` with `episode_id`, `episode_index`, and `frame_index`.

## Supported now

### From markers

Use this when collection already emitted explicit boundaries: teleop start/stop,
success/failure events, reset markers, or similar.

```bash
lancedb-robotics episodes from-markers \
  --lake ./robot.lance \
  --start-event teleop_start \
  --stop-event success \
  --stop-event failure
```

```python
from lancedb_robotics.lake import Lake

lake = Lake.open("./robot.lance")

report = lake.episodes.from_markers(
    start_event_types=["teleop_start"],
    stop_event_types=["success", "failure"],
)
```

The derivation pairs start and stop events per run, writes one `episodes` row per
interval, writes camera-frame handle rows to `videos`, annotates matching
observations, and records the recipe in `transform_runs`.

### From an event query

Use this for continuous logs where a firing event defines a clip, such as a
disengagement, intervention, collision, or high-severity incident.

```bash
lancedb-robotics episodes from-query \
  --lake ./robot.lance \
  --event-type disengagement \
  --before 2s \
  --after 5s
```

```python
report = lake.episodes.from_query(
    event_type="disengagement",
    before_ns=2_000_000_000,
    after_ns=5_000_000_000,
)
```

The derivation clips each matching event by the requested before/after window,
bounded by the parent run's time range.

### From scenarios or a snapshot

Use this when curation already produced analytical `scenarios` rows, or when a
dataset snapshot pins a curated scenario selection that should graduate into
physical training episodes.

```bash
lancedb-robotics episodes from-scenarios \
  --lake ./robot.lance \
  --scenario-id scn-123 \
  --scenario-id scn-456
```

```bash
lancedb-robotics episodes from-scenarios \
  --lake ./robot.lance \
  --snapshot hard-negatives-v1 \
  --outcome failure
```

```python
report = lake.episodes.from_scenarios(
    scenario_ids=["scn-123", "scn-456"],
    outcome_by_coverage_tag={"result:failure": "failure"},
)

snapshot_report = lake.episodes.from_scenarios(snapshot_name="hard-negatives-v1")
```

The derivation keeps `scenarios` immutable, writes physical `episodes` and
`videos`, preserves source scenario ids in episode provenance, and annotates
observations with stable `episode_id`, `episode_index`, and `frame_index`
columns. If a scenario carries `observation_ids`, that order becomes frame order;
otherwise frames are selected by the scenario's run/time bounds. Snapshot
promotion reads the snapshot's pinned scenario selection so split assignments do
not affect episode ordering.

### Reading episodes

```python
episode = lake.episodes.get(report.episode_ids[0])
frames = episode.frames()
window = episode.window(rate_hz=10, streams=["/camera/front", "/joint_states"])
```

`frames()` returns observations in stable `frame_index` order. `window(...)`
returns a deterministic nearest-frame view today; backlog 0031 replaces the
inside with the sub-frame alignment engine.

## Re-creating episodes today

Re-running the same marker, query, scenario, interval, or predicate recipe is
idempotent. The transform id is derived from the recipe parameters, so the same
recipe replaces its logical episode/video outputs and leaves one lineage row for
that recipe.

Every writer accepts an overlap policy:

- `error` rejects frames already owned by another derivation.
- `replace` allows the same recipe to rebuild its own annotations.
- `preserve` writes only currently unowned frames.
- `supersede` clears an older recipe, writes the replacement, and records
  lineage in `transform_runs`.

Use the lifecycle registry to audit and operate recipes:

```bash
lancedb-robotics episodes derivations list --lake ./robot.lance
lancedb-robotics episodes derivations show <transform-id> --lake ./robot.lance
lancedb-robotics episodes derivations dry-run predicate --lake ./robot.lance \
  --where "outcome = 'failure'"
lancedb-robotics episodes derivations rebuild <transform-id> --lake ./robot.lance
lancedb-robotics episodes derivations supersede <old-transform-id> \
  --with predicate --lake ./robot.lance --where "outcome = 'failure'"
lancedb-robotics episodes derivations clear <transform-id> --lake ./robot.lance
```

## Planned creation paths

The current API intentionally shipped the two core regimes first: marker
boundaries and event-window mining. Scenario promotion has since landed, and the
remaining paths are tracked explicitly:

- **Explicit-interval import**: import JSONL/CSV intervals from labeling,
  review, partner, or sim tools.
- **Predicate-based episode mining**: mine episodes from general predicates
  over observations/events/labels/model outputs.
- **Derivation lifecycle and rebuilds**: implemented lifecycle controls for
  listing, dry-running, rebuilding, clearing, and superseding derivation
  recipes safely.

Until those land, use `from_markers(...)` when boundaries are already present in
`events`, use `from_query(...)` when one event type defines the clip, and keep
scenario/snapshot-derived episodes as a training/export fallback rather than a
physical episode creation path.
