# LanceDB Robotics Lakehouse — Manual

A task-oriented manual for the AI researcher and platform engineer using the
lakehouse: how to get raw robotics logs in, curate and version them into training
data, close the loop with model outputs and feedback, and operate the whole thing
with lineage, evidence, and rebuild planning.

This is the *learn-and-use* view. It is deliberately organized by **what you are
trying to do**, not by the order features were built.

## How this manual is organized

- **Concepts** — the mental model you need before anything else: the lake and its
  canonical tables, the lineage graph, and the line between the open-source core
  and enterprise/plugin layers.
  - [Overview](concepts/overview.md)
  - [Architecture: why the lake is shaped this way](concepts/architecture.md)
  - [Storage and auth](concepts/storage-and-auth.md)
  - [OSS core vs. enterprise/plugin](concepts/oss-core-vs-enterprise.md)
  - [The training loader report contract](concepts/training-loader-report.md)
- **Journeys** — end-to-end walkthroughs of a real task.
  - [Bring a LeRobot dataset into the lake](journeys/lerobot-ingest.md)
  - [The rebuild loop: invalidate → plan → approve → dispatch](journeys/rebuild-loop.md)
  - [Retention policy and governance: define → activate → apply → project](journeys/retention-governance.md)
  - [Lineage audit reports and cleanup gates](journeys/lineage-audit-reports.md)
  - [Scoped lineage context and hook plugins](journeys/lineage-scoped-context.md)
  - [Find a run by its external ID (and share it safely)](journeys/external-context-lookup.md)
  - [Compare Enterprise training runs over time](journeys/training-report-history.md)
  - [Prove Enterprise training behaves the same everywhere](journeys/enterprise-training-conformance.md)
  - [Page a fleet-scale training epoch from a server-side plan](journeys/server-side-plan.md)
- **Reference** — complete, **auto-generated** surface docs. Do not edit these by
  hand; they are regenerated from the code.
  - [CLI reference](reference/cli.generated.md)
  - [Python API reference (`lake.*`)](reference/api.generated.md)
  - [Canonical tables](reference/tables.generated.md)
  - [Enterprise training compatibility matrix](reference/enterprise-training-compatibility.generated.md)

## Generated vs. hand-written

The `reference/*.generated.md` pages are produced by
`scripts/gen_docs_reference.py` from the live Typer command tree, the `lake.*`
namespace objects, and `TABLE_SCHEMAS`. `tests/test_docs_reference_current.py`
fails the suite if they drift from the code, so the reference is always true to the
shipped surface. The Concepts and Journeys chapters are hand-written and kept
current manually as the code changes.

To regenerate the reference after a code change:

```bash
python scripts/gen_docs_reference.py
```

To preview the manual locally (optional, requires `mkdocs` + `mkdocs-material`):

```bash
mkdocs serve
```
