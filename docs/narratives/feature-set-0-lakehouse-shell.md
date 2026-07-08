# Feature Set 0: The Lakehouse Shell Exists

## What changed

The repo went from docs-first to executable. `import lancedb_robotics` works,
and the `lancedb-robotics` CLI exposes the nine command groups that will become
the demo spine:

```text
$ lancedb-robotics --help
Commands:
  lake       Create and manage a LanceDB robotics lake.
  inspect    Inspect robot log files (MCAP first) without ingesting.
  ingest     Register sources and ingest robot logs into canonical lake rows.
  quality    Validate required streams and manage quarantine results.
  scenarios  Create searchable scenario/clip windows over ingested runs.
  search     Search scenarios with scalar, text, vector, and hybrid queries.
  dataset    Create reproducible dataset snapshots from search results.
  train      Preview dataset snapshots as training datasets.
  export     Export selected clips back to MCAP and replay-tool workflows.
```

Every group exposes help; none has real subcommands yet. Each later feature
set fills in its group: `lake init` (FS1), `inspect mcap` (FS2), `ingest mcap`
(FS3), `quality validate` (FS4), `scenarios create` (FS5), `search hybrid`
(FS6), `dataset snapshot` / `train preview` (FS7), `export mcap` (FS8).

## How to see it

```bash
uv sync --extra dev
uv run pytest                       # 28 tests: import, smoke, group-help contract, fixtures
uv run lancedb-robotics --help      # the demo spine
```

The integration check runs `python -m lancedb_robotics --help` from a clean
temporary directory, proving the shell works outside the repo checkout.

## Harness conventions established

- Test-first: failing import/smoke/contract tests preceded the implementation.
- `tests/fixtures/` holds tiny deterministic fixtures
  (`mini_run_manifest.json` is the first).
- `tests/snapshots/` holds golden CLI output; regenerate intentionally with
  `UPDATE_SNAPSHOTS=1 uv run pytest`.
- The command-group list is pinned in both `lancedb_robotics.cli.COMMAND_GROUPS`
  and the test contract, so adding or renaming a group is an explicit decision.
- `examples/` is the gitignored working directory for showcase data and
  generated lake artifacts.
