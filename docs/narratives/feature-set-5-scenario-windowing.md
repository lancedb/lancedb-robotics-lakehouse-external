# Feature Set 5a: The Raw Stream Becomes Searchable Behavior Windows

## What changed

`lancedb-robotics scenarios create` now turns ingested runs into deterministic
fixed-duration scenario rows. Boundaries are anchored to the run start/end time,
topic filters select which observations participate, and every scenario stores
its source observation IDs so downstream search and snapshots can trace clips
back to raw messages.

The first implementation covers windowing only. Captions and embeddings remain
the next feature set.

## How to see it

```bash
uv sync --extra dev
uv run lancedb-robotics lake init --lake examples/demo.robot.lance
uv run lancedb-robotics ingest mcap tests/fixtures/sample.mcap --lake examples/demo.robot.lance
uv run lancedb-robotics scenarios create --lake examples/demo.robot.lance --window 100ms
```

Expected output includes the lake URI, resolved window size, topic scope,
partial-final-window behavior, run count, created scenario count, and per-run
window counts.

## Contract

- Window membership is half-open (`start <= timestamp < end`) except the final
  materialized window includes observations exactly at the run end timestamp.
- Partial final windows are included by default and can be skipped with
  `--drop-partial`.
- Scenario rows include `topics`, `observation_ids`, `observation_count`,
  `window_ns`, `is_partial`, `coverage_tags`, and `transform_id`.
- Re-running the same windowing transform replaces matching scenario rows
  instead of duplicating them.
