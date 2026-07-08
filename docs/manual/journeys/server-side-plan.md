# Journey: page a fleet-scale training epoch from a server-side plan

**Scenario.** A trainer requests a remote dataset over `db://robotics` with a
snapshot that pins millions of frames. The local planner (see
[OSS core vs Enterprise](../concepts/oss-core-vs-enterprise.md)) would assemble the
*entire* shuffled row order in the SDK process and hand every PyTorch worker a copy
to slice — fine for a laptop, wasteful at fleet scale. Instead the query node
builds the row order **once** as a version-pinned *plan artifact* and returns a
small handle; four workers then claim deterministic, non-overlapping bounded pages
and stream samples, and any worker can resume from a global sample offset after a
restart — without rebuilding or downloading the whole plan.

This is additive: small and local datasets keep the in-process planner unchanged.
The server-side artifact is available only for the resolved **Enterprise** backend
with the `server_side_row_plan` capability; asking for it anywhere else raises a
typed diagnostic that points at local fallback or capability negotiation.

The flow is: **build the handle → page it per worker → resume by global offset**.

## 1. Build the plan handle

The query node persists the epoch order as paginated pages keyed by a stable
`plan_handle_id` (a digest of the row-plan id, pinned table versions, shuffle seed,
epoch, and page size), and returns a small, secret-free, serializable handle. The
handle records the snapshot id, pinned table versions, row-plan id, projected
columns, pushed predicates, and the backend display URI — and never an API key.

```bash
lancedb-robotics train plan build --lake db://robotics --snapshot demo-v1 \
  --columns observation_id --shuffle --shuffle-seed 23 --epoch 2 --page-size 4096 \
  --out plan-handle.json
```

```python
handle = lake.training.server_side_row_plan(
    "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=23, epoch=2,
    page_size=4096,
)
print(handle["plan_handle_id"], handle["num_pages"], handle["store_ref"])
# every Enterprise dataset also exposes its handle directly:
dataset = lake.training.dataset("demo-v1", shuffle=True, shuffle_seed=23, epoch=2,
                                backend="enterprise")
artifact = dataset.row_plan_handle          # None for the local backend
```

The pages persist in a durable, query-node-side store (an internal LanceDB table),
so the handle survives process restarts for resume, retry, and audit. The dataset
manifest and loader report advertise the handle's shape (`server_side_plan`) so a
report reader can see the artifact was used.

## 2. Page it, one worker at a time

Each worker claims whole pages `p` where `p % num_workers == worker_id` — disjoint
across workers, covering the epoch exactly once — and reads only its pages. It
never materializes the full row order. Each page carries a `next_page_token` for
deterministic handoff.

```bash
# worker 0 of 4 — every page it owns, as JSON
lancedb-robotics train plan page --lake db://robotics --handle plan-handle.json \
  --worker 0 --num-workers 4 --all
```

```python
for worker_id in range(4):
    for page in lake.training.row_plan_pages(handle, worker_id=worker_id, num_workers=4):
        row_ids, frame_ids = page["row_ids"], page["frame_ids"]
        # ... hydrate + train on this bounded page ...

# or drive it one page at a time via tokens (worker handoff / checkpointing):
first = lake.training.row_plan_page(handle, worker_id=0, num_workers=4)
nxt = lake.training.row_plan_page(handle, page_token=first["next_page_token"])
```

The server-side order is **equivalent** to the local planner's: with
`num_workers=1` the pages concatenate to the byte-identical ordered ids a local
dataset produces for the same snapshot, seed, epoch, and filters; across workers
the union covers that epoch exactly once with no duplicates.

## 3. Resume from a global sample offset

`resume_from` is a *global* epoch offset, not a per-worker one. Each worker trims
the prefix of its first claimed page that falls before the offset; the union of all
workers' pages, in page order, is exactly the suffix `G[resume_from:]` of the
original global order — through the *same* plan handle, with no rebuild.

```bash
lancedb-robotics train plan page --lake db://robotics --handle plan-handle.json \
  --worker 0 --num-workers 4 --resume-from 500000 --all
```

```python
pages = lake.training.row_plan_pages(handle, worker_id=2, num_workers=4,
                                     resume_from=500_000)
```

## Audit note

The handle is asserted secret-free before it is ever returned or written: bearer
tokens and `api_key`/credential-shaped fields are rejected, `*_auth_ref` names are
preserved (they are references, not secrets). The `train plan page` command reopens
the handle against its durable store using only the serialized metadata — no
snapshot table object is required, so a worker in a fresh process can page a plan it
holds only as JSON.

## What's next (not yet built)

The artifact is built by the SDK acting as the query node and persisted to an
internal Lance table; a true remote query-node build/serve API, aligned-tick plan
artifacts for policy-tick datasets, plan retention/expiry, and cross-run plan reuse
are filed as this task's scale/robustness backlog follow-ons. The page order is
reproduced from the same seeded permutation the local planner uses, so it inherits
the local shuffle's `O(N)` one-time build cost at the query node (paid once, not per
worker).
