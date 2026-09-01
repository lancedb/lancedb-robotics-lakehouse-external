# Storage and auth

A lake is addressed by a URI, and the URI's scheme decides how it connects and how
credentials resolve. This chapter covers the connection kinds and the one auth
principle that runs through all of them. The step-by-step how-to lives in the
Getting Started "connecting" chapter (forthcoming); the implemented resolver paths
are documented in
[enterprise remote & namespace paths](../../product/enterprise-remote-namespace-paths.md).

## The connection kinds

`Lake.init(uri, ...)` and `Lake.open(uri, ...)` accept four kinds of location:

| Kind | Example URI | Notes |
| --- | --- | --- |
| Local path | `/data/robot.lance` | files on disk; nothing to authenticate |
| Object store | `s3://bucket/robot.lance`, `gs://…`, `az://…` | streamed directly; needs the matching cloud SDK + credentials |
| LanceDB Enterprise | `db://database` | managed remote database; queries go through the enterprise query node |
| REST namespace | (namespace-resolved) | a namespace service vends the storage location + short-lived credentials |

For object-store and namespace lakes, the data plane can also be reached directly
through `pylance` once the location/credentials are resolved — that is how workers
read blobs at scale without routing bytes through a control service.

## Credentials never live in the lake

This is the rule to internalize: **secrets are resolved at runtime and are never
persisted in canonical tables.** A lake row stores only a logical URI and an
*auth-reference name* — a key that tells the resolver where to look — not an API
key, token, or connection string.

`Lake.init` / `Lake.open` expose the reference names as parameters, one per plane:

- `storage_auth_ref` — object-store credentials for the lake itself.
- `remote_auth_ref` — LanceDB Enterprise (`db://`) API key/region/host.
- `namespace_auth_ref` — REST namespace service credentials.
- `source_auth_ref` — object-store credentials for reading *raw source* logs
  and datasets (MCAP/ROS/video/LeRobot), which may differ from the lake's own
  storage.

You also pass `storage_options` (a dict of backend options) for object-store
tuning. At connect time the resolver takes the reference name, pulls the actual
credential from the environment / config / an injected client, uses it in memory,
and discards it — the persisted rows carry only the reference name. For enterprise
namespaces, vended credentials are short-lived; long-running workers probe expiry
and re-request rather than caching a secret.

## Why this shape

Keeping auth out of the lake is what lets the same lake definition be shared,
versioned, and audited without leaking secrets, and lets different environments
(laptop, CI, a training cluster) resolve the *same* reference name to *their* own
credentials. It is the storage-and-auth half of the
[OSS-core vs. enterprise/plugin](oss-core-vs-enterprise.md) split: the open core
defines the reference contract; how a deployment resolves a reference (a secrets
manager, a namespace service, an env chain) is a deployment concern.

## Exporting to an object store, and reconciliation

Dataset exports and projections can materialize to an object store as well as a
local directory. `lake.projections.<format>.export(snapshot, out="s3://…")`,
`export_dataset_snapshot(..., out_dir="gs://…")`, and the
`lancedb-robotics dataset export`/`dataset project --mode export` commands all
accept an `s3://` / `gs://` / `az://` destination. Object-store credentials
resolve the same way as everywhere else — pass `storage_options` (or repeatable
`--storage-option key=value`) and/or an `auth_ref` (`--auth-ref`); nothing is
persisted in the lake.

An object-store export stages the projection locally, publishes each file, then
**reconciles** the written objects: it stats every object, compares its content
length against what was staged, and records a stable, secret-free provider
fingerprint (ETag / version / generation). The export **fails with a typed error**
if an expected object is missing or its size does not match — so the accounting
report reflects the boundary artifacts actually written, not just the local plan.
Because accounting is computed from the staged tree, a local export and an
object-store export of the same snapshot produce identical payload/media,
logical-reference, and copy-ratio accounting. The per-object detail (URI, content
length, checksum, payload-vs-metadata classification, container/compression
notes) is written into the export and projection manifests and into the
`curation_materializations` report body.

Manifest-only projection **plans** (`lake.projections.<format>.plan(...)`) and
live projections never resolve a destination: they stay zero-copy and require no
object-store credentials or backend.

## Caveats

- Object-store URIs need the matching fsspec backend installed (`s3fs` for
  `s3://`, `gcsfs` for `gs://` / `gcs://`, `adlfs` for Azure schemes); a missing
  backend surfaces a clear install error, not a stack trace.
- `db://` and REST-namespace features are enterprise paths; a purely local or
  object-store lake uses only `storage_auth_ref` / `storage_options`.
- Export reconciliation verifies **object presence and size** from provider
  metadata; it does not download and re-hash object bytes. Byte-level content
  verification is the strict-content-hash validation policy used on the ingest
  side, not the default export path.
