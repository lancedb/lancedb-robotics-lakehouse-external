"""Object-store-aware reconciliation for materialized projection exports.

Backlog 0083 made curation materialization accounting automatic, but the
projection writers (:mod:`lancedb_robotics.dataset_export` and
:mod:`lancedb_robotics.projections`) only ever wrote to a local ``pathlib.Path``
and measured bytes with ``Path.stat().st_size``. Backlog 0144 extends that to the
object stores real exports land on (``s3://``, ``gs://``, ``az://``) and adds a
post-write reconciliation step.

The design keeps every existing format writer operating on a *local* directory:
the final directory for a local export, or a temporary staging directory for an
object-store export. Accounting (byte counts, ``content_hash``, the
metadata-bytes fixpoint) is therefore computed from the local tree in both cases,
so a local export and an equivalent object-store export produce byte-identical
payload/metadata/logical-reference/copy-ratio accounting. The only remote
operations are:

* **publish** -- upload each staged file with :func:`storage.write_binary_uri`;
* **reconcile** -- stat each written object with :func:`storage.uri_info`, compare
  its content length against what was staged, and record a stable, secret-free
  provider-metadata fingerprint (ETag/version/generation via
  :func:`object_metadata_fingerprint`).

Manifest-only plans and live projections never resolve a destination, so they
stay zero-copy and require no object-store credentials or the ``[object-store]``
extra.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lancedb_robotics.lerobot_object_store_validation import (
    object_metadata_fingerprint,
)
from lancedb_robotics.storage import (
    StorageConfigError,
    is_object_store_uri,
    join_uri,
    uri_info,
    uri_scheme,
    write_binary_uri,
)

EXPORT_RECONCILIATION_VERSION = "export-reconciliation-v1"

#: Per-object payload-vs-metadata classification values.
PAYLOAD_CLASS = "payload"
METADATA_CLASS = "metadata"
MIXED_CLASS = "mixed"

#: Bound the number of files a single export reconciles so a pathological export
#: can't turn reconciliation into an unbounded stat storm without saying so. A
#: bounded export is proportional to what it wrote; this cap only trips on a
#: runaway file explosion, and it fails loudly *before* the expensive hashing and
#: upload rather than silently skipping objects (SKILLS.md: never silently
#: degrade). Very large exports are a listing-sweep follow-up (backlog 0481).
MAX_RECONCILED_OBJECTS = 200_000

#: Cap on the per-object records embedded in a manifest / report body. The full
#: set can reach 10^5-10^6 objects for an image export; embedding all of them in
#: a manifest JSON (re-serialized in the metadata fixpoint) or a single Lance
#: cell is the BUG-02 oversized-write shape. Above this, the manifest carries a
#: bounded sample plus the full counts; the durable curation row carries only the
#: summary (no per-object array).
MAX_EMBEDDED_OBJECTS = 1_000

#: Bounded chunk for streaming file hashes so a large media object never
#: materializes whole just to be fingerprinted.
_HASH_CHUNK_BYTES = 1 << 20


class ExportReconciliationError(Exception):
    """Raised when a materialized export does not match its planned objects."""


@dataclass(frozen=True)
class ExportObject:
    """One materialized output object and its planned (local) accounting."""

    relative_path: str
    uri: str
    content_length: int
    checksum: str
    classification: str
    container: str
    compression: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "uri": self.uri,
            "content_length": int(self.content_length),
            "checksum": self.checksum,
            "classification": self.classification,
            "container": self.container,
            "compression": self.compression,
        }


@dataclass(frozen=True)
class ExportReconciliationReport:
    """Result of reconciling a set of planned objects against a destination."""

    backend: str
    destination: str
    status: str
    checked: bool
    object_count: int
    verified_object_count: int
    unverified_object_count: int
    total_object_bytes: int
    payload_object_bytes: int
    metadata_object_bytes: int
    mixed_object_bytes: int
    objects: tuple[dict[str, Any], ...] = ()
    missing: tuple[str, ...] = ()
    mismatched: tuple[dict[str, Any], ...] = ()

    def _summary(self) -> dict[str, Any]:
        return {
            "version": EXPORT_RECONCILIATION_VERSION,
            "backend": self.backend,
            "destination": self.destination,
            "status": self.status,
            "checked": bool(self.checked),
            "object_count": int(self.object_count),
            "verified_object_count": int(self.verified_object_count),
            "unverified_object_count": int(self.unverified_object_count),
            "total_object_bytes": int(self.total_object_bytes),
            "payload_object_bytes": int(self.payload_object_bytes),
            "metadata_object_bytes": int(self.metadata_object_bytes),
            "mixed_object_bytes": int(self.mixed_object_bytes),
            "missing": list(self.missing),
            "mismatched": [dict(item) for item in self.mismatched],
        }

    def summary_dict(self) -> dict[str, Any]:
        """Reconciliation summary with no per-object array.

        Used for the durable ``curation_materializations`` row so a huge export's
        per-object detail never lands in a single Lance cell (BUG-02 shape).
        """
        return self._summary()

    def to_dict(self) -> dict[str, Any]:
        """Full reconciliation block for a manifest, with a bounded object sample.

        The per-object array is capped at :data:`MAX_EMBEDDED_OBJECTS`; above that
        the manifest carries a sample plus ``objects_truncated`` and the full
        ``object_count`` so the manifest JSON (re-serialized in the metadata
        fixpoint) stays bounded.
        """
        payload = self._summary()
        total = len(self.objects)
        embedded = [dict(obj) for obj in self.objects[:MAX_EMBEDDED_OBJECTS]]
        payload["objects"] = embedded
        payload["objects_embedded"] = len(embedded)
        payload["objects_truncated"] = total > MAX_EMBEDDED_OBJECTS
        return payload


def classify_relative_path(relative_path: str) -> tuple[str, str, str]:
    """Return ``(classification, container, compression)`` for one output file.

    Classification is derived from the export layout, which is deterministic:
    camera payload bytes are written as ``images/**/*.bin``; WebDataset shards
    (``*.tar``/``*.tar.gz``) interleave media and metadata, so they are ``mixed``;
    everything else (parquet tables, JSON/JSONL manifests) is control metadata.
    """
    lower = relative_path.lower()
    if lower.endswith((".tar.gz", ".tgz")):
        return MIXED_CLASS, "tar", "gzip"
    if lower.endswith(".tar"):
        return MIXED_CLASS, "tar", "none"
    if lower.endswith(".parquet"):
        return METADATA_CLASS, "parquet", "none"
    if lower.endswith(".jsonl"):
        return METADATA_CLASS, "jsonl", "none"
    if lower.endswith(".json"):
        return METADATA_CLASS, "json", "none"
    if lower.endswith(".bin"):
        return PAYLOAD_CLASS, "image-bytes", "none"
    if lower.endswith(".gz"):
        return METADATA_CLASS, "binary", "gzip"
    return METADATA_CLASS, "binary", "none"


def _hash_file(path: Path) -> tuple[int, str]:
    """Return ``(size, "sha256:<hex>")`` streaming the file in bounded chunks."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, "sha256:" + digest.hexdigest()


class ExportTarget:
    """Resolve an ``out_dir`` to a local working root plus a publish destination.

    * A local ``out_dir`` writes in place: ``local_root`` is ``out_dir`` and
      ``publish`` is a no-op.
    * An object-store ``out_dir`` stages into a temporary local directory, and
      ``publish`` uploads each staged file to the destination URI.

    Pass ``staging_root`` to reuse a caller-managed staging directory (the
    projection layer does this so it can add its own manifest before cleanup);
    in that case the target does not own or remove the directory.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        storage_options: Mapping[str, Any] | None = None,
        auth_ref: str | None = None,
        staging_root: str | Path | None = None,
    ) -> None:
        raw = str(out_dir)
        self.storage_options = dict(storage_options or {})
        self.auth_ref = auth_ref
        if is_object_store_uri(raw):
            self.is_remote = True
            self.destination = raw.rstrip("/")
            self.backend = uri_scheme(raw)
            if staging_root is not None:
                self.local_root = Path(staging_root)
                self._owns_local = False
            else:
                self.local_root = Path(
                    tempfile.mkdtemp(prefix="lancedb-robotics-export-")
                )
                self._owns_local = True
        else:
            self.is_remote = False
            self.local_root = Path(raw)
            self.destination = str(self.local_root)
            self.backend = "local"
            self._owns_local = False
        self.local_root.mkdir(parents=True, exist_ok=True)

    def uri_for(self, relative_path: str) -> str:
        """Return the destination URI (or local path) for one relative output."""
        return join_uri(self.destination, relative_path)

    def _storage_options(self) -> dict[str, Any] | None:
        return self.storage_options or None

    def publish(self, relative_paths: Sequence[str]) -> int:
        """Upload staged files to the destination. No-op for local targets.

        Reads one file into memory at a time (bounded per file, matching the
        existing in-memory blob write path). Overwrites on re-publish so a
        retried export converges on identical bytes.
        """
        if not self.is_remote:
            return 0
        published = 0
        for relative_path in relative_paths:
            data = (self.local_root / relative_path).read_bytes()
            write_binary_uri(
                self.uri_for(relative_path),
                data,
                storage_options=self._storage_options(),
                auth_ref=self.auth_ref,
            )
            published += 1
        return published

    def build_objects(
        self,
        relative_paths: Sequence[str],
        *,
        compute_checksums: bool = False,
    ) -> list[ExportObject]:
        """Build planned per-object accounting from the staged local files.

        Content length comes from ``stat`` (no read). Reconciliation verifies
        object presence + size + provider fingerprint, so the per-object content
        checksum is opt-in (``compute_checksums``) -- computing it would re-read
        every file whole on top of the whole-export ``content_hash`` that already
        does (SKILLS.md: project only what you need).
        """
        objects: list[ExportObject] = []
        for relative_path in sorted(dict.fromkeys(relative_paths)):
            path = self.local_root / relative_path
            if compute_checksums:
                size, checksum = _hash_file(path)
            else:
                size, checksum = path.stat().st_size, ""
            classification, container, compression = classify_relative_path(relative_path)
            objects.append(
                ExportObject(
                    relative_path=relative_path,
                    uri=self.uri_for(relative_path),
                    content_length=size,
                    checksum=checksum,
                    classification=classification,
                    container=container,
                    compression=compression,
                )
            )
        return objects

    def reconcile(
        self,
        objects: Sequence[ExportObject],
    ) -> ExportReconciliationReport:
        """Stat each written object and confirm it matches its planned size.

        Raises :class:`ExportReconciliationError` when any expected object is
        missing or its content length does not match what was staged. The
        provider metadata fingerprint (ETag/version/generation) is recorded per
        object so a later re-reconciliation can detect a same-size mutation.
        """
        _check_object_bound(len(objects))
        reconciled: list[dict[str, Any]] = []
        missing: list[str] = []
        mismatched: list[dict[str, Any]] = []
        total = payload_bytes = metadata_bytes = mixed_bytes = 0
        verified = unverified = 0
        for obj in objects:
            record = obj.to_dict()
            try:
                info = uri_info(
                    obj.uri,
                    storage_options=self._storage_options(),
                    auth_ref=self.auth_ref,
                )
            except StorageConfigError as exc:
                record["status"] = "missing"
                record["detail"] = str(exc)
                missing.append(obj.relative_path)
                reconciled.append(record)
                continue
            actual_size = _info_size(info)
            record["provider_fingerprint"] = object_metadata_fingerprint(info)
            record["reconciled_size"] = actual_size
            if actual_size is not None and int(actual_size) != int(obj.content_length):
                record["status"] = "size-mismatch"
                record["detail"] = (
                    f"expected {obj.content_length} bytes, found {int(actual_size)}"
                )
                mismatched.append(
                    {
                        "relative_path": obj.relative_path,
                        "uri": obj.uri,
                        "expected": int(obj.content_length),
                        "actual": int(actual_size),
                    }
                )
                reconciled.append(record)
                continue
            total += obj.content_length
            if obj.classification == PAYLOAD_CLASS:
                payload_bytes += obj.content_length
            elif obj.classification == MIXED_CLASS:
                mixed_bytes += obj.content_length
            else:
                metadata_bytes += obj.content_length
            if actual_size is None:
                # Present but the backend returned no size -- do not claim it as
                # verified; surface it rather than folding it into "passed".
                record["status"] = "unverified-size"
                unverified += 1
            else:
                record["status"] = "ok"
                verified += 1
            reconciled.append(record)

        if missing or mismatched:
            status = "failed"
        elif unverified:
            status = "passed-unverified"
        else:
            status = "passed"
        report = ExportReconciliationReport(
            backend=self.backend,
            destination=self.destination,
            status=status,
            checked=True,
            object_count=len(objects),
            verified_object_count=verified,
            unverified_object_count=unverified,
            total_object_bytes=total,
            payload_object_bytes=payload_bytes,
            metadata_object_bytes=metadata_bytes,
            mixed_object_bytes=mixed_bytes,
            objects=tuple(reconciled),
            missing=tuple(missing),
            mismatched=tuple(mismatched),
        )
        if status == "failed":
            raise ExportReconciliationError(_reconciliation_failure_message(report))
        return report

    def cleanup(self) -> None:
        """Remove the staging directory when this target created it."""
        if self._owns_local and self.local_root.exists():
            shutil.rmtree(self.local_root, ignore_errors=True)


def publish_and_reconcile(
    target: ExportTarget,
    relative_paths: Sequence[str],
) -> ExportReconciliationReport:
    """Publish ``relative_paths`` from the target's staging root and reconcile.

    Convenience wrapper for callers that publish and reconcile the same file set
    in one step (the data-file layer of an export). The object-count bound is
    checked *before* any hashing or upload so a runaway export fails fast.
    """
    _check_object_bound(len(list(dict.fromkeys(relative_paths))))
    objects = target.build_objects(relative_paths)
    target.publish(relative_paths)
    return target.reconcile(objects)


def _check_object_bound(count: int) -> None:
    if count > MAX_RECONCILED_OBJECTS:
        raise ExportReconciliationError(
            f"export produced {count} objects, exceeding the reconciliation "
            f"bound of {MAX_RECONCILED_OBJECTS}; split the export into smaller "
            "snapshots, or (backlog 0481) reconcile via a paginated listing "
            "sweep and raise MAX_RECONCILED_OBJECTS deliberately"
        )


def _info_size(info: Mapping[str, Any]) -> int | None:
    for key in ("size", "Size", "ContentLength", "content_length"):
        value = info.get(key)
        if value is not None:
            return int(value)
    return None


def _reconciliation_failure_message(report: ExportReconciliationReport) -> str:
    parts = [
        f"export reconciliation failed against {report.destination} "
        f"({report.backend}): {len(report.missing)} missing, "
        f"{len(report.mismatched)} size-mismatched of {report.object_count} objects"
    ]
    for relative_path in list(report.missing)[:5]:
        parts.append(f"  missing: {relative_path}")
    for item in list(report.mismatched)[:5]:
        parts.append(
            f"  size-mismatch: {item['relative_path']} "
            f"expected {item['expected']} found {item['actual']}"
        )
    parts.append(
        "re-run the export to overwrite the destination, or verify the "
        "object-store credentials and that no other writer mutated the prefix"
    )
    return "\n".join(parts)


def empty_reconciliation(destination: str = "") -> dict[str, Any]:
    """Reconciliation block for modes that write no objects (plan/live)."""
    return ExportReconciliationReport(
        backend="none",
        destination=destination,
        status="not-applicable",
        checked=False,
        object_count=0,
        verified_object_count=0,
        unverified_object_count=0,
        total_object_bytes=0,
        payload_object_bytes=0,
        metadata_object_bytes=0,
        mixed_object_bytes=0,
    ).to_dict()


__all__ = [
    "EXPORT_RECONCILIATION_VERSION",
    "ExportObject",
    "ExportReconciliationError",
    "ExportReconciliationReport",
    "ExportTarget",
    "MAX_RECONCILED_OBJECTS",
    "MIXED_CLASS",
    "METADATA_CLASS",
    "PAYLOAD_CLASS",
    "classify_relative_path",
    "empty_reconciliation",
    "publish_and_reconcile",
]
