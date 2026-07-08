# Architecture: why the lake is shaped this way

Four load-bearing decisions define the whole system. Understanding them explains
why the API looks the way it does — why there is no `join()`, why enrichment adds
a column instead of updating a row, and why IDs are hashes. Each links to its
decision record for the full rationale.

## 1. Lance is the index *and* the fast-access layer

Heavy payloads — sensor bytes, images, video — are **first-class blob-encoded
columns** in Lance, not plain binary and not external pointers you have to chase.
The consequence: search, curation, and training are served *from Lance itself*.
The raw MCAP / rosbag2 file is archival truth, reachable by a foreign-key +
byte-offset pointer, but it is never a read or training dependency.

Two properties fall out of blob encoding:

- **Projection is cheap.** A scan that does not select the blob column reads zero
  blob bytes. Filtering millions of frames by metadata never touches the payloads.
- **Late materialization.** Training fetches only the bytes it needs, by row id,
  at load time — not by pre-materializing a column into RAM.

This is the foundational choice; the other three follow from it. See
[decision 0024 — Lance is the index and fast-access layer](../../decisions/0024-lance-is-the-index-and-fast-access-layer.md).

## 2. Denormalized, one wide table per grain — no joins

There is **no `join()` in the SDK**. Each entity grain is one wide canonical table
(`observations` is the frame grain, `scenarios` the window grain, and so on), and
anything you filter or train on lives on the grain row itself. Run- and
episode-level scalars (`robot_id`, `site_id`, `task_id`, `outcome`) are
*denormalized* onto the frame so hot-path predicates never need a join. Related
rows are reached by a point `take` by id, not a relational join.

This is not a storage micro-optimization — it is the natural fit for Lance's
columnar, random-access model and for the way training and curation actually query
(filter the grain, then take the payloads). See
[decision 0025 — denormalized, enrich-as-column data modeling](../../decisions/0025-denormalized-enrich-as-column-data-modeling.md).

## 3. Enrichment is additive, never an in-place rewrite

On a blob-encoded table, `Table.update()` and `Table.merge_insert()` both raise.
That is deliberate. Every enrichment — quality verdicts, embeddings, captions,
labels, predictions — is applied as a **new column** (a new table version), not a
mutation of existing rows. The mechanism is a row-aligned `add_columns` on the
underlying Lance dataset.

Why this matters to you as a user:

- **Re-embedding with a new model is a new column, side by side** with the old
  one — not a destructive overwrite. Snapshots pinned to the old column stay
  valid.
- **A row is stable across versions.** An `observation_id` refers to the same
  logical sample no matter how many enrichment columns have been added since.

See [decision 0026 — blob-safe additive write mechanism](../../decisions/0026-blob-safe-additive-write-mechanism.md).

## 4. Content-addressed, portable IDs

`run_id` and `source_id` derive from the **content** of the source file (a
checksum), not its path or URI. Ingesting the same bytes from a laptop, from S3,
or from a teammate's machine produces the **same IDs** — so dataset splits and
lineage are reproducible across environments, and re-ingesting the same log is
idempotent (no duplicate rows).

The trade-off to know: switching an old, path-keyed lake to content-addressed IDs
is a breaking migration (re-ingest into a fresh lake); you cannot mix the two ID
schemes in one lake. See
[decision 0023 — content-addressed portable run IDs](../../decisions/0023-content-addressed-portable-run-ids.md).

## How the four fit together

A raw log is ingested once into content-addressed rows (3); its payloads become
blob columns you can scan past for free and fetch lazily (1); you enrich it by
adding columns rather than rewriting (2, 3); and you query the wide grain directly
without joins (2). Snapshots then pin exact table versions, so a training run is
reproducible down to the column — and because reads stream and a snapshot is a pin
rather than a copy, this all holds whether the corpus is a gigabyte or a petabyte,
without you managing memory. The Operations part (forthcoming) covers keeping the
physical layout healthy at scale.

## Caveats

- The no-join model means genuinely relational questions ("every scenario that
  shares a source with this one") are expressed as point takes + lineage
  traversal, not SQL joins. This is intentional but is a mindset shift coming from
  a warehouse.
- Additive enrichment means table versions accumulate; the Operations part
  (forthcoming) covers compaction and retention to keep the physical layout
  healthy.
