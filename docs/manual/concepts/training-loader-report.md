# The training loader report contract

Every native or aligned training loader emits a **training loader report**: a
redacted, JSON-serializable record of *what a training run read and how it was
routed* — the resolved backend, the row/tick and epoch plans, the pinned table
versions, per-request metrics, and any fallbacks or disabled capabilities. It is
produced by `dataset.loader_report()` and written with `--report-out` on
`lancedb-robotics train remote report`.

Because that JSON travels — into benchmark bundles, the
[report catalog](../journeys/training-report-history.md), experiment trackers, and
support tickets — it needs a contract that consumers can rely on and that provably
never leaks a credential. That contract is versioned and enforced.

## The versioned schema

The report carries its version in its `kind` string:

```json
{ "kind": "lancedb-robotics/training-loader-report/v1", "...": "..." }
```

The committed JSON Schema (Draft 2020-12) for v1 lives at
[`reference/training_loader_report.v1.schema.json`](../reference/training_loader_report.v1.schema.json)
and is generated from
`lancedb_robotics.training_report_schema.TRAINING_LOADER_REPORT_SCHEMA` (a drift
test keeps the two identical). Both native and aligned reports share one envelope;
the `loader.kind` field discriminates the two:

| `loader.kind` | Additional required fields |
| --- | --- |
| `native-training` | `snapshot` (`id`, `name`) |
| `aligned-training` | `alignment` (`id`, `name`), `read_table_versions` |

The stable envelope every v1 report guarantees: `kind`, `loader`, `lake`,
`table_versions`, `plans`, `policies`, `remote_execution`, `metrics`,
`fallback_events`, `disabled_capabilities`, and `run`.

## Compatibility rules

The report is a long-lived contract. The rules for evolving it:

1. **The `kind` suffix is the version.** A consumer must branch on it, not on the
   presence of individual fields.
2. **v1 validation is open.** Unknown object properties are *allowed and ignored*.
   A future minor revision may **add** fields — a new metric, a new policy knob —
   to a v1 report, and every existing v1 validator (including the one shipped in
   this repo) must keep passing. Do not write a consumer that rejects a v1 report
   for carrying a field it does not recognize.
3. **Breaking changes require a new version.** Renaming or removing a required
   field, changing a field's type, or changing what a field *means* is a breaking
   change. It requires a new `kind` suffix (`.../v2`) and a new committed schema
   document. v1 consumers keep validating v1 reports unchanged; a v2-aware
   consumer branches on the `kind`.

In short: **additive within a version, new version for anything breaking.**

## The redaction contract

A conformant report never contains a *resolved* credential, while it *does* keep
the safe fields that make it useful. The emitter redacts recursively as the report
is built; validation independently proves the result.

**Removed** (replaced with `<redacted>`) — any field whose name looks like a
secret: `authorization`, `api_key`/`apikey`, `access_key`/`secret_key`,
`session_token`, `password`, `secret`, `credential`, `bearer`, `token`, and their
`namespace_*` credential-vending variants. Any string value shaped like a live
credential (a `Bearer …`/`Basic …` header, an AWS/GitHub/Slack key, a JWT, or a
PEM private key) is caught even under an innocuous key.

**Preserved** — the fields a report exists to carry: `display_uri`, endpoint /
profile / backend labels, capability flags, and any `*_auth_ref` name. An
*auth reference* is a pointer to a credential (a vault path, a profile name), never
the secret itself, so it is always kept.

## Validating a report

Gate a report before you attach it to a benchmark or persist it to a catalog. The
CLI validates one or more files and exits non-zero on any malformed, off-schema,
or credential-bearing report:

```bash
lancedb-robotics train report validate training_loader_report.json
# training_loader_report.json: OK (lancedb-robotics/training-loader-report/v1)

lancedb-robotics train report validate *.json --format json
```

The same check is available in the SDK:

```python
from lancedb_robotics.training_report_schema import validate_training_loader_report

validation = lake.training.validate_report(dataset.loader_report())
if not validation.ok:
    for problem in validation.schema_errors + validation.secret_findings:
        print(problem)
    validation.raise_for_status()  # raises ReportValidationError
```

`validate_report` accepts a `TrainingLoaderReport`, a payload mapping, or anything
with `to_dict()`. It returns a `ReportValidation` with `schema_errors` (shape) and
`secret_findings` (redaction) lists; `ok` is true only when both are empty.

## What's next

- **Aggregate** per-worker reports into a job-level view before validating the
  merged result — see `train report merge` and the
  [training report history](../journeys/training-report-history.md) journey.
- **Persist** validated reports to the durable catalog with
  `lake.training.record_report(...)`.
