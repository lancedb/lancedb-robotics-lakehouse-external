# SKILLS.md — Engineering skills for building at lakehouse scale

This is the technical-standards companion to [AGENTS.md](AGENTS.md). AGENTS.md
governs *workflow* (task records, worktrees, handoffs); this file governs what a
feature has to be true of — for usability, resource usage, scalability to
billions-to-trillions of rows, and robustness under concurrent access or partial
failure — before it's done. It is not a `.miagent/skills/*.md` recipe (those are
bounded to one repeatable task-shaped procedure); this is the standing checklist
every design, implementation, and test plan is measured against.

**North star:** no one who touches this repo — today, or in five years, whether
or not they've read this file — should be able to ship a feature that works on a
laptop-sized sample and OOMs, stalls, silently corrupts data, or quietly degrades
the moment it meets real scale or a second concurrent writer. Every rule below
exists because someone already shipped exactly that, once. The fix is now cheap
to apply and, if you skip it, expensive to rediscover.

Every rule below is grounded in either a load-bearing architecture decision under
[`docs/decisions/`](docs/decisions/) or a real, fixed production-shaped bug from
this repo's history (cited inline). If you find a new failure mode that these
rules wouldn't have caught, add it — this file rots the moment it stops matching
reality.

## 0. The four invariants (non-negotiable)

Every feature, no matter how small, has to fit inside these four. They are why
the API looks the way it does; see
[architecture](docs/manual/concepts/architecture.md) for the full rationale.

1. **Lance is the index *and* the fast-access layer** (decision
   [0024](docs/decisions/0024-lance-is-the-index-and-fast-access-layer.md)).
   Heavy payloads are blob-encoded columns *in* Lance, not external pointers.
   Violate this and you've built a second store that can drift from the lake.
2. **One wide table per grain, no joins** (decision
   [0025](docs/decisions/0025-denormalized-enrich-as-column-data-modeling.md)).
   Relations are reached by point `take` by id or lineage traversal, never
   `join()`. Violate this and hot-path predicates start needing a join the SDK
   doesn't support.
3. **Enrichment is additive, never an in-place rewrite** (decision
   [0026](docs/decisions/0026-blob-safe-additive-write-mechanism.md)). New
   information is a new column / new table version via `add_columns`.
   `Table.update()`/`merge_insert()` raise on blob-encoded tables on purpose.
   Violate this and old snapshots pinned to prior columns silently break.
4. **Content-addressed, portable IDs** (decision
   [0023](docs/decisions/0023-content-addressed-portable-run-ids.md)). IDs derive
   from content hashes, not paths — re-ingesting the same bytes is idempotent and
   reproducible across machines. Violate this and dataset splits/lineage stop
   being reproducible across environments.

## 1. Design skills

### Usability
- Before adding a new canonical table, check
  [`docs/product/open-questions.md`](docs/product/open-questions.md) — it has
  already settled "is this a new entity or a denormalized column?" for several
  cases (e.g. augmentation/reconstruction lineage stayed a `parent_id` column +
  `transform_runs`, not a new table). Re-litigating a settled question is a
  design smell.
- The CLI's top-level command groups (`COMMAND_GROUPS` in
  [`cli/__init__.py`](src/lancedb_robotics/cli/__init__.py)) are a pinned
  contract — tests pin the list. Adding or renaming a group is a product
  decision, not a refactor.
- Prefer a typed error with a clear remediation over a stack trace. Precedent: a
  missing object-store backend (`s3fs`/`gcsfs`/`adlfs`) surfaces a clear install
  error, not an import traceback (see
  [storage-and-auth](docs/manual/concepts/storage-and-auth.md)).
- Never let an operation silently degrade to a slower/costlier path without
  telling the caller. The enterprise-conformance invariant is: **typed error OR
  explicit fallback, never silent local materialization** (backlog
  [0116](.miagent/backlog/0116-enterprise-training-conformance-and-fault-injection.md) /
  `enterprise_conformance.py`,
  `MetadataOnlyViolationError` in `training.py`). A user who thinks they got a
  metadata-only op and actually triggered a full remote fetch has been lied to.
- A performance or scale claim is a usability contract with the reader — see
  §3's benchmarking rule before writing one down anywhere user-facing.

### Resource usage
- Secrets never live in the lake. Design every new connection/credential surface
  around an *auth-reference name*, resolved at runtime per plane
  (`storage_auth_ref` / `remote_auth_ref` / `namespace_auth_ref` /
  `source_auth_ref`) — see
  [storage-and-auth](docs/manual/concepts/storage-and-auth.md). A persisted row
  should never be able to leak a credential.
- Vended enterprise credentials are short-lived by design. A long-running worker
  probes expiry and re-requests; it does not cache a secret past its lifetime.
- Clients and workers talk to object storage or the query node — **never
  directly to a plan executor**. `pe_fanout`/`num_fragments`-shaped fields are
  server *response* telemetry, never client inputs. If a design has a worker
  reaching past the query node to a backend-internal component, that's the
  wrong boundary.

### Scalability
- Any new listing/export/traversal surface must be designed paginated and
  resumable from the first draft, with a bounded page size and a stable
  cursor/token — not bolted on after a full-scan version ships and someone hits
  a wall. This is a recurring theme across the lineage/evidence/rebuild-plan
  catalogs (see the paging-and-indexing backlog items under
  `.miagent/backlog/`).
- Any long-running or batch-shaped operation must be idempotent and resumable by
  a stable digest/id — a restart must not redo completed work or double-write.
  Precedents: content-addressed ingest ids, prewarm `JobRun` dedup keyed on
  `prewarm_id`, the server-side training plan's `plan_handle_id`.
- At true fleet/epoch scale, don't design "build the full row order in the SDK
  process, hand every worker a copy." Build it **once** server-side as a
  version-pinned plan artifact and have workers claim disjoint, bounded pages —
  see the [server-side plan journey](docs/manual/journeys/server-side-plan.md).
  This is additive: small/local datasets keep the simple in-process planner.
- Assume two processes can run the same write path at the same time without
  coordinating — a retried job, a second ingest/enrich/writeback invocation
  against the same table, a second engineer running the same command. Design
  the write as an atomic upsert (Lance `merge_insert`) with a bounded retry on
  retryable commit conflicts, not a hand-rolled external lock or a multi-step
  delete-then-add sequence a crash or a race can leave half-done.

## 2. Implementation skills

### Usability
- Match errors to the boundary that raises them: a missing optional dependency
  should name the extra to install; a capability not available on the resolved
  backend should point at the fallback or the capability-negotiation path (see
  the server-side plan doc's "typed diagnostic" behavior).
- If a design decision changed in `docs/product/open-questions.md`, update the
  registry in the same change — it's the single source of truth, not a diary.

### Resource usage — bounded-streaming discipline
This is the single most repeated lesson in this repo's bug history. **Assume the
corpus never fits in RAM.** Concretely, on every new read path:
- **Project only what you need.** Never select a blob column unless you're
  actually about to fetch bytes — a scan that doesn't project a blob column
  reads zero blob bytes (decision 0024). Skipping this is how "just count rows"
  ends up paying for payload bytes.
- **Stream, don't collect.** Iterate `to_batches()`/`iter_batches()`; never
  `to_arrow().to_pylist()` or `to_pandas()` over an unbounded scan. **BUG-06**
  (`lake.training.dataset(...)` / `bench run --formats lance`): constructing a
  training context over one trivial column OOM'd at 22 GB RSS / 128 GB peak
  because the whole corpus materialized eagerly via `to_arrow().to_pylist()`.
  The fix scoped reads to the snapshot's *referenced* rows, made projection
  request-aware, and switched to `to_batches()`.
- **Scope to what's referenced, not the whole grain.** A snapshot over 20
  scenarios should touch ~178K referenced observations, not the full
  `observations` table. Any new feature that reads a grain table through a
  snapshot/selection must inherit that scoping, not re-derive its own full scan.
- **Never do O(n) or O(n²) work inside a per-row loop.** **BUG-01**
  (`scenarios enrich`): a helper rebuilt `set(all_ids)` *inside* a per-row loop,
  making the scan O(observations × referenced_ids); it stalled past 600s on a
  352K-observation corpus. The fix hoisted the membership set out of the loop
  and pushed the projection into the Lance scan itself. If you're computing
  something once per input row that doesn't depend on that row, hoist it —
  or better, push it into the scan as a projection/predicate.
- **Resolve to row ids and `take`; don't build a giant `IN (...)` at all.**
  BUG-06 recurred after its first fix landed: `lake.training.dataset(...)` still
  turned a pinned snapshot's ~178K observation ids into a multi-MB
  `observation_id IN (<178K literals>)` predicate, which blew up the
  Lance/DataFusion query planner itself (killed past 6 GB on the real corpus) —
  independent of whether a scalar index existed. **Chunking the `IN` list is
  not the fix.** The shipped fix replaced the predicate entirely with Lance's
  random-access `take_row_ids` by resolved row id — the same pattern already
  used for blob bytes — so read cost is proportional to the referenced set
  (~376 MB heap / 1.3 GB RSS on the same corpus, down from an OOM). If you find
  yourself building a wide `IN` clause from an id list you already hold, resolve
  row ids and `take` instead of growing the predicate.

### Robustness — concurrent writers and partial failure
- **A write path must converge or fail loudly, never silently corrupt.** BUG-04:
  two concurrent `scenarios enrich` processes against the same lake raced a
  multi-commit delete-then-add sequence, leaving the `embedding` column added
  but every summary/embedding **null** — no lock, no error, silent data loss.
  The fix leaned on Lance's own atomicity instead of inventing coordination:
  both the scenario and `transform_runs` writes became single atomic
  `merge_insert` upserts (never a multi-step delete+add a crash can interrupt
  mid-way), the writer retries Lance's *retryable* commit conflict with a bound
  (so concurrent writers converge instead of racing forever or failing
  outright), and a post-write check refuses to report success if any written
  row is null. For any new write path, ask: if two of these run at once, or one
  crashes mid-write, does the table end up fully written, unchanged, or loudly
  errored — never partially written and silently reported as success?
- **Batch large commits; don't stage one oversized write.** BUG-02: a single
  oversized commit deterministically panicked Lance's Rust-layer mini-block
  encoder. The fix batched the write into bounded `merge_insert` calls rather
  than one unbounded commit — the same "bounded, not one-shot" discipline as
  the streaming-writer compaction rule below, applied to commit size instead of
  fragment count.

### Scalability — physical layout and indexing
- **The write-time ordering that matters: compact → build/refresh indexes →
  prune old versions.** Getting this order wrong reintroduces the bug it exists
  to fix (pruning before compacting can drop the fragments compaction needed;
  indexing before compacting indexes fragments about to be rewritten). This is
  the order `maintain_lake` uses and the order ingest now calls it in (see
  **BUG-14**/**BUG-15** below).
- **A streaming writer must compact, not rely on per-flush behavior.** Each
  streaming `_flush` is one `table.add()` = one fragment + one version. **BUG-14**:
  a freshly-ingested `observations` grain sat at `batch_size` rows/fragment —
  3–4 orders of magnitude below Lance's healthy ~1M-row-per-fragment target —
  imposing a measured **~79x** per-row scan tax on every later read, plus
  unbounded version churn. Fix: compact once at the *end* of ingest (not
  per-flush) and prune per-flush versions afterward, both on by default.
- **New filter/join-key columns need a scalar index, matched to cardinality.**
  **BUG-15**: the two largest tables in any real lake — `observations` (352K
  rows: `run_id`/`observation_id`/`topic`/`timestamp_ns`) and the lineage graph
  (`lineage_edges`, 1.29M rows: endpoints) — had zero scalar indexes, so the
  filters behind other fixes were full scans. Use BTREE for high-cardinality
  ids/ranges, BITMAP for low-cardinality categoricals (`topic`, `edge_type`).
- **Index coverage does not auto-extend to new fragments.** Appends land
  unindexed until the next refresh/optimize call. A new write path that adds
  rows to an indexed table must call the refresh — don't assume yesterday's
  index still covers today's rows.
- **Don't trust default ANN knobs at your row count.** Lance intentionally
  skips building a vector index below `MIN_INDEX_ROWS` (256 rows) and falls
  back to exact brute-force — check which regime a table is actually in before
  reasoning about latency. When an index *is* built, measure recall@k against
  the brute-force baseline before trusting it: a real run against this corpus
  saw near-zero recall at default IVF_PQ params, and the lever that mattered was
  `refine_factor`, not `nprobes` (tracked in
  [backlog 0183](.miagent/backlog/0183-ann-vector-index-validation-at-scale.md)).
  A vector index with unvalidated recall is worse than exact scan, because it
  looks like it's working.

### Scalability — bounded graph and lineage traversal
- **Expand a bounded frontier; never load the whole graph to answer a local
  question.** BUG-13: `lineage trace`/`impact` loaded the *entire*
  `lineage_edges` (1.29M rows) and `lineage_artifacts` (704K rows) into Python
  on every call to build an in-memory adjacency index — cost O(total edges), a
  ~193 ms floor even when the answer was 3 nodes, and it OOM'd at didi scale.
  The fix was a level-by-level BFS that fetches only the edges incident to the
  *visited* frontier via chunked, indexed `endpoint IN (frontier)` reads
  (reusing the BUG-15 BTREE on edge endpoints, chunked at 512 ids per `IN`),
  plus on-demand artifact fetch by id — reads scale with the visited subgraph,
  not total graph size (~26x faster on a small-result query, and flat as the
  graph grows). Any new traversal over `lineage_edges`/`lineage_artifacts` — or
  any future graph-shaped table — inherits this shape: expand the frontier via
  indexed lookups, never `SELECT *` the whole edge table to build an index
  yourself in Python.

## 3. Testing skills

- **Tier tests by cost, gate with markers — don't skip writing them.** Fast
  synthetic-fixture unit tests always run. `realcorpus` / `slow` /
  `torch_loader` / `video_decode` / `rlds_native` (see `pyproject.toml`) gate
  tests that need real corpora or optional heavy dependencies; they skip
  cleanly when the corpus/extra is absent so the default suite stays green and
  fast. The dedicated CI lane for each flips the skip into a hard failure via
  `LANCEDB_ROBOTICS_REQUIRE_*`, so a genuinely broken optional path can't hide
  behind a permanent skip.
- **Regression-test the *shape* of the scale bug, not today's row count.**
  BUG-01 shipped with a 100k-observation perf regression test; BUG-06 shipped
  with a scope/projection/bounded-memory regression test. A fix without a test
  that reproduces the pathological shape (many small fragments, a wide `IN`
  list, an O(n²) loop, an unindexed hot predicate) regresses silently the next
  time someone touches the same code.
- **Never quote a performance number without a retained artifact.** The
  benchmark suite's discipline exists precisely so external claims are checked,
  not asserted: `bench prepare` → `bench run` (or `run-public-lerobot`) →
  `validate-public-lerobot` against an explicit claim manifest with a pinned
  dataset revision and commit. The validator fails on a stale manifest or on a
  skipped-but-claimed format. If you don't have a report id + manifest path,
  you don't have a number — see
  [reproducible-benchmark-suite](docs/narratives/reproducible-benchmark-suite.md).
- **Docs are part of the test surface.** Any change to the CLI, `lake.*`
  surface, or a table schema requires regenerating
  `docs/manual/reference/*.generated.md` via `scripts/gen_docs_reference.py` —
  the drift test fails otherwise. Treat "docs still true" as a correctness
  check, not a chore.
- **Enterprise-surfaced features get checked against both backends.** For any
  training/data-access surface exposed over `db://`, run the conformance
  harness (`enterprise_conformance.py`) against the local/fake backend and,
  opt-in, the live backend. The thing under test is the invariant itself — typed
  error or explicit fallback — not just "does it return data."
- **Validate ANN recall numerically wherever a vector index is exercised.**
  recall@k vs. brute-force, at the row count and params the feature actually
  ships with, not the defaults.
- **Assert the read/write bound — and the mechanism — not just correctness.**
  Correctness alone can pass a regression that happens to also do the naive
  whole-table thing, just more slowly. BUG-13's frontier-expansion suite
  (`tests/test_lineage_frontier_expansion.py`) pairs ground-truth-BFS
  equivalence with a spy on the fetch helper asserting reads never exceed the
  visited frontier. BUG-06 round 2 went further: `test_native_training_dataset.py`
  added a *structural* guard that asserts `take_row_ids` is actually used and no
  `observation_id IN (...)` predicate is built — not just that the query is fast
  today. Write that second assertion whenever a fix's whole point is "uses the
  right primitive," not only "produces the right answer."
- **Pin every guardrail with a regression test, or it will regress.** BUG-11:
  the `--strict` no-silent-fallback escape hatch shipped once, then a later,
  unrelated embeddings refactor quietly dropped it — reintroducing exactly the
  silent-degradation failure mode §1 warns against, undetected until someone
  hit it again. A rule in this document that isn't backed by a test enforcing
  it is a rule a future refactor can delete for free.

## 4. Known failure modes → the rule that catches them

| Bug | What happened | Rule |
| --- | --- | --- |
| BUG-01 | `enrich()` rebuilt `set(all_ids)` inside a per-row loop — O(observations × referenced_ids); stalled >600s at 352K observations | Never recompute per-row-invariant state inside the loop; push filters into the scan |
| BUG-02 | A single oversized commit deterministically panicked Lance's Rust-layer mini-block encoder during `lineage refresh` | Batch large writes into bounded `merge_insert` calls, not one unbounded commit |
| BUG-04 | Two concurrent `enrich` writers raced a delete-then-add sequence; the embedding column landed with every row **null** — silent data loss, no error | Write as an atomic `merge_insert` upsert with bounded retry; validate postconditions before reporting success |
| BUG-06 (round 1) | Training dataset / bench materialized the whole corpus via `to_arrow().to_pylist()` for one column — 22 GB RSS / 128 GB peak | Scope to referenced rows, project only needed columns, stream via `to_batches()` |
| BUG-06 (round 2) | Even after round 1's fix, a 178K-term `observation_id IN (...)` predicate blew up the query planner itself past 6 GB | Resolve to row ids and `take_row_ids`; a giant `IN` list is fixed by not building it, not by chunking it |
| BUG-11 | A shipped `--strict` no-silent-fallback flag was silently dropped by a later, unrelated refactor | Every guardrail needs a regression test, or a future refactor deletes it for free |
| BUG-13 | `lineage trace`/`impact` loaded the whole 1.29M-edge graph into Python on every call, even for a 3-node answer; OOM'd at didi scale | Expand a bounded frontier via chunked indexed reads; never load the whole graph to answer a local query |
| BUG-14 | Per-flush ingest = one fragment + one version; ~100 rows/fragment left an ~79x per-row scan tax | Compact once at end of ingest, not per flush; compact → index → prune ordering |
| BUG-15 | `observations` (352K rows) and `lineage_edges` (1.29M rows) had zero scalar indexes; filters were full scans | New filter/join-key columns need a scalar index (BTREE high-card / BITMAP low-card), refreshed on every append |
| backlog 0183 | IVF_PQ recall was ~0 at default params; the lever was `refine_factor`, not `nprobes`; index also correctly no-ops below `MIN_INDEX_ROWS`=256 | Measure recall@k explicitly; know your knob and your floor before trusting an ANN index |

## 5. Pre-merge checklist

- [ ] Does this add a join, or a second source of truth for something that
      should be a denormalized column? Check `docs/product/open-questions.md`.
- [ ] Does every new read path project only needed columns and stream
      (`to_batches`), with no `to_pandas()`/`to_pylist()` over an unbounded scan?
- [ ] Does every per-row/per-id loop avoid recomputing loop-invariant state, and
      does every wide id-list read resolve to row ids and `take` rather than
      growing an `IN (...)` predicate?
- [ ] Do new high-/low-cardinality filter columns get scalar indexes, wired into
      `lake maintain` in the compact → index → prune order?
- [ ] Is any new long-running or listing operation paginated and resumable by a
      stable id/offset? Does any new graph/lineage-shaped traversal expand a
      bounded frontier via indexed reads, rather than loading the whole
      edge/artifact table?
- [ ] If this adds or touches a write path, does a concurrent second writer (or
      a crash mid-write) converge or fail loudly — never partially write and
      still report success? Is the commit itself bounded (no oversized
      one-shot write)?
- [ ] Are secrets kept out of persisted rows (auth-ref indirection only), and are
      vended credentials re-requested rather than cached past expiry?
- [ ] Is there a test that reproduces the *shape* of the scale risk (fragment
      count, list width, loop complexity, index coverage), and does it assert
      the read/write *bound* and not just correctness?
- [ ] If this change encodes a "never do X" guardrail, is there a regression
      test that fails if a future refactor silently drops it?
- [ ] If this touches CLI/API/schema, did `scripts/gen_docs_reference.py` run and
      does the drift test pass?
- [ ] If this claims a performance number, is it backed by a retained bench
      report id + validated claim manifest?
