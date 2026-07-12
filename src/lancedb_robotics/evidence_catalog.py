"""Durable evidence-pack catalog, retention, and scalable materialization (0108).

Backlog 0065 makes evidence packs deterministic and 0107 turns a pack into
viewer bundles, but both stop at one synchronous local export at a time with no
durable record. Production audit systems need to look a pack up later, enforce
retention/redaction policy, and materialize large packs under byte/file limits
without buffering everything in memory. This module adds:

- a persisted catalog table (:data:`CATALOG_TABLE`) with one idempotent row per
  pack, keyed by ``manifest_digest`` (== ``pack_id``). The metadata-first v1
  manifest is stored inline, so a pack reloads by digest or subject handle
  without re-tracing the lineage graph;
- an append-only audit log (:data:`EVENTS_TABLE`) for pack creation,
  materialization, retention changes, expiry, and redaction denials;
- retention/protection metadata that is queryable and survives catalog reloads,
  plus a force-gated expiry;
- scalable materialization: a chunked, bounded-memory copy plan that enforces
  ``max_bytes`` / ``max_files`` before copying, is idempotent/resumable, and can
  write to a local directory or an object-store destination.

Manifest schema changes are versioned: :func:`is_supported_evidence_schema`
gates every load/materialize path, and old v1 manifests remain loadable.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa

from lancedb_robotics.blob import (
    ATTACHMENT_DATA_COLUMN,
    PAYLOAD_BLOB_COLUMN,
    fetch_blobs_by_row_id,
)
from lancedb_robotics.capability_gates import BLOB, require_lake_capability
from lancedb_robotics.evidence import (
    SUPPORTED_EVIDENCE_PACK_SCHEMAS,
    EvidencePackError,
    EvidencePackReport,
    _json_ready,
    _safe_name,
    _stable_sha256,
    is_supported_evidence_schema,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import (
    EVIDENCE_PACK_EVENTS_SCHEMA,
    EVIDENCE_PACKS_SCHEMA,
)
from lancedb_robotics.storage import (
    join_uri,
    uri_exists,
    write_binary_uri,
)
from lancedb_robotics.video import VIDEO_ENCODING_BLOB_COLUMN

CATALOG_TABLE = "evidence_packs"
EVENTS_TABLE = "evidence_pack_events"

#: Catalog row contract version (independent of the pack manifest schema).
CATALOG_SCHEMA_VERSION = "lancedb-robotics/evidence-pack-catalog/v1"

DEFAULT_RETENTION_POLICY = "default"
DEFAULT_CHUNK_SIZE = 64
_PROGRESS_SIDECAR = "_materialization.json"
_MANIFEST_NAME = "manifest.json"

# (rel-dir, table, id_column, blob_column, ref-list-key, ref-id-key, size-key)
_MATERIALIZE_SPECS: tuple[tuple[str, str, str, str, str, str, str | None], ...] = (
    ("payloads", "observations", "observation_id", PAYLOAD_BLOB_COLUMN, "payload_refs", "row_id", None),
    ("attachments", "attachments", "attachment_id", ATTACHMENT_DATA_COLUMN, "attachment_refs", "attachment_id", "size"),
    (
        "video_encodings",
        "video_encodings",
        "encoding_id",
        VIDEO_ENCODING_BLOB_COLUMN,
        "video_encoding_refs",
        "encoding_id",
        "encoded_size_bytes",
    ),
)


def _require_supported_schema(manifest: Mapping[str, Any]) -> str:
    schema = manifest.get("schema_version")
    if not is_supported_evidence_schema(schema):
        raise EvidencePackError(
            f"unsupported evidence-pack schema {schema!r}; "
            f"expected one of {SUPPORTED_EVIDENCE_PACK_SCHEMAS}"
        )
    return str(schema)


# --- Catalog entry / reports ------------------------------------------------


@dataclass(frozen=True)
class EvidencePackCatalogEntry:
    """A single durable evidence-pack catalog row (manifest kept separate)."""

    pack_id: str
    manifest_digest: str
    catalog_schema_version: str
    manifest_schema_version: str
    subject_kind: str
    subject_handle: str
    lake_uri: str
    mode: str
    materialization_status: str
    output_uri: str
    bytes_total: int
    file_count: int
    row_id_count: int
    source_coordinate_hashes: tuple[str, ...]
    table_version_pins: tuple[dict[str, Any], ...]
    retention_policy: str
    protected: bool
    expires_at: datetime | None
    redaction_policy: str
    redacted: bool
    sensitive_sources: tuple[str, ...]
    metadata: dict[str, str]
    created_by: str
    created_at: datetime
    updated_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "manifest_digest": self.manifest_digest,
            "catalog_schema_version": self.catalog_schema_version,
            "manifest_schema_version": self.manifest_schema_version,
            "subject_kind": self.subject_kind,
            "subject_handle": self.subject_handle,
            "lake_uri": self.lake_uri,
            "mode": self.mode,
            "materialization_status": self.materialization_status,
            "output_uri": self.output_uri,
            "bytes_total": self.bytes_total,
            "file_count": self.file_count,
            "row_id_count": self.row_id_count,
            "source_coordinate_hashes": list(self.source_coordinate_hashes),
            "table_version_pins": [dict(row) for row in self.table_version_pins],
            "retention_policy": self.retention_policy,
            "protected": self.protected,
            "expires_at": _iso_or_none(self.expires_at),
            "redaction_policy": self.redaction_policy,
            "redacted": self.redacted,
            "sensitive_sources": list(self.sensitive_sources),
            "metadata": dict(self.metadata),
            "created_by": self.created_by,
            "created_at": _iso_or_none(self.created_at),
            "updated_at": _iso_or_none(self.updated_at),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class EvidencePackListPage:
    """A bounded page of catalog entries plus an opaque continuation cursor."""

    packs: tuple[EvidencePackCatalogEntry, ...]
    next_cursor: str | None
    page_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "packs": [entry.as_dict() for entry in self.packs],
            "next_cursor": self.next_cursor,
            "page_size": self.page_size,
            "count": len(self.packs),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class MaterializationReport:
    """Result of a scalable evidence-pack materialization run."""

    lake_uri: str
    pack_id: str
    manifest_digest: str
    output_uri: str
    manifest_path: str
    status: str
    bytes_total: int
    file_count: int
    copied_count: int
    skipped_count: int
    files: tuple[dict[str, Any], ...]
    chunk_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "pack_id": self.pack_id,
            "manifest_digest": self.manifest_digest,
            "output_uri": self.output_uri,
            "manifest_path": self.manifest_path,
            "status": self.status,
            "bytes_total": self.bytes_total,
            "file_count": self.file_count,
            "copied_count": self.copied_count,
            "skipped_count": self.skipped_count,
            "chunk_count": self.chunk_count,
            "files": list(self.files),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


# --- Record / load / list ---------------------------------------------------


def record_evidence_pack(
    lake: Lake,
    pack: EvidencePackReport | Mapping[str, Any],
    *,
    retention_policy: str = DEFAULT_RETENTION_POLICY,
    protected: bool = False,
    expires_at: datetime | None = None,
    redaction_policy: str = "",
    sensitive_source_patterns: Sequence[str] = (),
    on_sensitive: str = "flag",
    metadata: Mapping[str, Any] | None = None,
    created_by: str | None = None,
) -> EvidencePackCatalogEntry:
    """Record an evidence pack in the durable catalog, idempotent by digest.

    ``pack`` is an :class:`EvidencePackReport` or a reloaded v1 manifest mapping.
    The pack identity is the manifest digest; re-recording the same pack updates
    the row in place and preserves the original ``created_at``. When
    ``sensitive_source_patterns`` match a source URI in the pack, the row is
    flagged (``on_sensitive="flag"``) or refused (``on_sensitive="deny"``,
    raising :class:`EvidencePackError`).
    """

    manifest = _manifest_of(pack)
    schema_version = _require_supported_schema(manifest)
    digest = _stable_sha256(manifest)
    subject = manifest.get("subject") or {}

    matched = _match_sensitive_sources(manifest, sensitive_source_patterns)
    if matched and on_sensitive == "deny":
        _emit_event(
            lake,
            pack_id=digest,
            manifest_digest=digest,
            event_type="redaction-denied",
            manifest=manifest,
            status="denied",
            detail="sensitive source(s): " + ", ".join(matched),
            created_by=created_by,
            metadata={"sensitive_sources": ";".join(matched)},
        )
        raise EvidencePackError(
            "refusing to record evidence pack: source(s) match the sensitive "
            f"deny list: {', '.join(matched)}"
        )

    existing = _load_row(lake, pack_id=digest)
    now = datetime.now(UTC)
    created_at = _row_timestamp(existing, "created_at", now)

    row = {
        "pack_id": digest,
        "manifest_digest": digest,
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "manifest_schema_version": schema_version,
        "subject_kind": str(subject.get("type") or ""),
        "subject_handle": _subject_handle(subject),
        "lake_uri": str(manifest.get("lake_uri") or lake.uri),
        "mode": str(manifest.get("mode") or "plan"),
        "materialization_status": _initial_status(manifest),
        "output_uri": str(manifest.get("output_dir") or ""),
        "bytes_total": int(_verification_bytes(manifest)),
        "file_count": len(manifest.get("materialized_files") or []),
        "row_id_count": _row_id_count(manifest),
        "source_coordinate_hashes": _coordinate_hashes(manifest),
        "table_version_pins": _version_pins(manifest),
        "retention_policy": str(retention_policy or DEFAULT_RETENTION_POLICY),
        "protected": bool(protected),
        "expires_at": expires_at,
        "redaction_policy": str(redaction_policy or ""),
        "redacted": bool(matched),
        "sensitive_sources": list(matched),
        "manifest_json": _encode_manifest(manifest),
        "metadata": _kv_items(metadata),
        "created_by": created_by or "",
        "created_at": created_at,
        "updated_at": now,
    }
    _upsert_row(lake, row)
    _emit_event(
        lake,
        pack_id=digest,
        manifest_digest=digest,
        event_type="created",
        manifest=manifest,
        status="ok",
        detail="recorded",
        created_by=created_by,
        metadata={"redacted": "true"} if matched else {},
    )
    return _row_to_entry(row)


def load_evidence_pack(
    lake: Lake,
    *,
    digest: str | None = None,
    subject: str | None = None,
) -> tuple[EvidencePackCatalogEntry, dict[str, Any]]:
    """Reload a catalog entry and its manifest by digest or subject handle.

    Exactly one of ``digest`` / ``subject`` is required. When several packs share
    a subject handle, the most recently created pack is returned (ties broken by
    digest) so the call stays deterministic.
    """

    if bool(digest) == bool(subject):
        raise EvidencePackError("pass exactly one of digest= or subject=")

    if digest:
        row = _load_row(lake, pack_id=str(digest))
        if row is None:
            raise EvidencePackError(f"no evidence pack recorded for digest {digest!r}")
    else:
        rows = _load_rows_where(lake, f"subject_handle = {_sql_literal(str(subject))}")
        if not rows:
            raise EvidencePackError(f"no evidence pack recorded for subject {subject!r}")
        rows.sort(key=lambda item: (_as_dt(item.get("created_at")), str(item.get("pack_id"))))
        row = rows[-1]

    manifest = _decode_manifest(row)
    _require_supported_schema(manifest)
    return _row_to_entry(row), manifest


def list_evidence_packs(
    lake: Lake,
    *,
    subject_kind: str | None = None,
    subject_handle: str | None = None,
    materialization_status: str | None = None,
    protected: bool | None = None,
    page_size: int = 50,
    cursor: str | None = None,
) -> EvidencePackListPage:
    """List catalog entries newest-first, filtered and bounded by ``page_size``.

    Filters push down to SQL where possible; results are ordered by
    ``(created_at desc, pack_id desc)`` and paged with an opaque cursor so large
    catalogs never load in one shot.
    """

    if page_size < 1:
        raise EvidencePackError("page_size must be >= 1")

    predicates: list[str] = []
    if subject_kind is not None:
        predicates.append(f"subject_kind = {_sql_literal(subject_kind)}")
    if subject_handle is not None:
        predicates.append(f"subject_handle = {_sql_literal(subject_handle)}")
    if materialization_status is not None:
        predicates.append(f"materialization_status = {_sql_literal(materialization_status)}")
    if protected is not None:
        predicates.append(f"protected = {'true' if protected else 'false'}")

    rows = _load_rows_where(lake, " AND ".join(predicates) if predicates else None)
    ordered = sorted(
        rows,
        key=lambda item: (_as_dt(item.get("created_at")), str(item.get("pack_id"))),
        reverse=True,
    )

    start = _decode_cursor(cursor) if cursor else 0
    window = ordered[start : start + page_size]
    next_cursor = (
        _encode_cursor(start + page_size) if start + page_size < len(ordered) else None
    )
    return EvidencePackListPage(
        packs=tuple(_row_to_entry(row) for row in window),
        next_cursor=next_cursor,
        page_size=page_size,
    )


# --- Retention --------------------------------------------------------------


def set_evidence_retention(
    lake: Lake,
    digest: str,
    *,
    protected: bool | None = None,
    retention_policy: str | None = None,
    expires_at: datetime | None = None,
    clear_expires_at: bool = False,
    redaction_policy: str | None = None,
    created_by: str | None = None,
) -> EvidencePackCatalogEntry:
    """Update retention/protection metadata for a recorded pack."""

    row = _load_row(lake, pack_id=str(digest))
    if row is None:
        raise EvidencePackError(f"no evidence pack recorded for digest {digest!r}")

    if protected is not None:
        row["protected"] = bool(protected)
    if retention_policy is not None:
        row["retention_policy"] = str(retention_policy)
    if clear_expires_at:
        row["expires_at"] = None
    elif expires_at is not None:
        row["expires_at"] = expires_at
    if redaction_policy is not None:
        row["redaction_policy"] = str(redaction_policy)
    row["updated_at"] = datetime.now(UTC)

    _upsert_row(lake, row)
    _emit_event(
        lake,
        pack_id=row["pack_id"],
        manifest_digest=row["manifest_digest"],
        event_type="retention-updated",
        row=row,
        status="ok",
        detail=f"protected={row['protected']} policy={row['retention_policy']}",
        created_by=created_by,
    )
    return _row_to_entry(row)


def evidence_retention_plan(
    lake: Lake,
    *,
    older_than: timedelta | datetime | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Report which recorded packs are protected vs safe to expire, and why."""

    reference = now or datetime.now(UTC)
    cutoff = _retention_cutoff(older_than, reference)
    rows = _load_rows_where(lake, None)

    protected: list[dict[str, Any]] = []
    expirable: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item.get("pack_id"))):
        entry = _row_to_entry(row)
        expires_at = entry.expires_at
        expired = expires_at is not None and expires_at <= reference
        older = cutoff is not None and entry.created_at <= cutoff
        summary = {
            "pack_id": entry.pack_id,
            "subject_handle": entry.subject_handle,
            "created_at": _iso_or_none(entry.created_at),
            "expires_at": _iso_or_none(expires_at),
            "bytes_total": entry.bytes_total,
        }
        if entry.protected:
            protected.append({**summary, "reason": "protected"})
        elif expired or older or (cutoff is None and older_than is None):
            summary["reason"] = "expired" if expired else ("older-than-cutoff" if older else "unconstrained")
            expirable.append(summary)
        else:
            protected.append({**summary, "reason": "within-retention"})

    return {
        "reference_time": _iso_or_none(reference),
        "cutoff": _iso_or_none(cutoff),
        "protected": protected,
        "expirable": expirable,
        "protected_count": len(protected),
        "expirable_count": len(expirable),
    }


def expire_evidence_pack(
    lake: Lake,
    digest: str,
    *,
    force: bool = False,
    created_by: str | None = None,
) -> dict[str, Any]:
    """Delete a catalog entry, refused if protected unless ``force`` is set.

    Only the catalog row is removed; any materialized bytes on disk / object
    store are left for the caller to garbage-collect.
    """

    row = _load_row(lake, pack_id=str(digest))
    if row is None:
        raise EvidencePackError(f"no evidence pack recorded for digest {digest!r}")
    if row.get("protected") and not force:
        raise EvidencePackError(
            f"evidence pack {digest!r} is protected; pass force=True to expire it"
        )

    lake.table(CATALOG_TABLE).delete(f"pack_id = {_sql_literal(str(digest))}")
    _emit_event(
        lake,
        pack_id=str(digest),
        manifest_digest=str(row.get("manifest_digest") or digest),
        event_type="expired",
        row=row,
        status="ok",
        detail="forced" if force else "expired",
        created_by=created_by,
    )
    return {
        "pack_id": str(digest),
        "expired": True,
        "forced": bool(force),
        "was_protected": bool(row.get("protected")),
    }


# --- Scalable materialization ----------------------------------------------


def plan_materialization(
    lake: Lake,
    pack: EvidencePackReport | Mapping[str, Any],
    *,
    include_payloads: bool = False,
    include_attachments: bool = False,
    include_video: bool = False,
    max_bytes: int | None = None,
    max_files: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Build a bounded, chunked copy plan and enforce limits before any copy.

    Returns the ordered object list, per-table chunks, and planned totals. The
    file limit is exact; the byte limit is checked against the sizes known from
    the manifest (attachments / video encodings) up front and again per object
    during the copy, so an over-limit run fails before writing bytes.
    """

    manifest = _manifest_of(pack)
    _require_supported_schema(manifest)
    if chunk_size < 1:
        raise EvidencePackError("chunk_size must be >= 1")
    if not (include_payloads or include_attachments or include_video):
        raise EvidencePackError(
            "nothing to materialize: enable include_payloads / include_attachments / include_video"
        )

    selectors = {
        "payloads": include_payloads,
        "attachments": include_attachments,
        "video_encodings": include_video,
    }
    objects: list[dict[str, Any]] = []
    known_bytes = 0
    for rel_dir, table, id_column, column, refs_key, ref_id_key, size_key in _MATERIALIZE_SPECS:
        if not selectors[rel_dir]:
            continue
        for ref in manifest.get(refs_key) or []:
            row_id = ref.get(ref_id_key)
            if not row_id:
                continue
            size = _maybe_int(ref.get(size_key)) if size_key else None
            if size:
                known_bytes += size
            objects.append(
                {
                    "kind": _kind_for(rel_dir),
                    "table": table,
                    "id_column": id_column,
                    "column": column,
                    "row_id": str(row_id),
                    "path": f"{rel_dir}/{_safe_name(str(row_id))}.bin",
                    "known_bytes": size,
                    "table_version": _table_version(manifest, table),
                }
            )

    objects.sort(key=lambda item: (item["kind"], item["path"]))
    if not objects:
        raise EvidencePackError("evidence pack has no selected blob refs to materialize")

    if max_files is not None and len(objects) > max_files:
        raise EvidencePackError(
            f"materialization would emit {len(objects)} files, exceeding max_files={max_files}"
        )
    if max_bytes is not None and known_bytes > max_bytes:
        raise EvidencePackError(
            f"materialization would copy at least {known_bytes} known bytes, "
            f"exceeding max_bytes={max_bytes}"
        )

    chunks = [objects[i : i + chunk_size] for i in range(0, len(objects), chunk_size)]
    return {
        "objects": objects,
        "chunks": chunks,
        "object_count": len(objects),
        "chunk_count": len(chunks),
        "known_bytes": known_bytes,
        "limits": {"max_bytes": max_bytes, "max_files": max_files},
    }


def materialize_evidence_pack(
    lake: Lake,
    pack: EvidencePackReport | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    output_uri: str | None = None,
    include_payloads: bool = False,
    include_attachments: bool = False,
    include_video: bool = False,
    max_bytes: int | None = None,
    max_files: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    resume: bool = True,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
    record: bool = True,
    retention_policy: str = DEFAULT_RETENTION_POLICY,
    protected: bool = False,
    sensitive_source_patterns: Sequence[str] = (),
    on_sensitive: str = "flag",
    created_by: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> MaterializationReport:
    """Materialize a pack's blobs to a local or object-store destination.

    The copy is chunked (bounded to ``chunk_size`` blobs in memory at a time),
    idempotent and resumable via a ``_materialization.json`` sidecar: an object
    whose bytes already exist is skipped, so a re-run reports identical hashes and
    an interrupted run resumes without re-copying completed objects.
    """

    manifest = _manifest_of(pack)
    _require_supported_schema(manifest)
    destination = _resolve_destination(output_dir, output_uri)
    is_object_dest = output_uri is not None

    matched = _match_sensitive_sources(manifest, sensitive_source_patterns)
    if matched and on_sensitive == "deny":
        raise EvidencePackError(
            "refusing to materialize: source(s) match the sensitive deny list: "
            + ", ".join(matched)
        )

    plan = plan_materialization(
        lake,
        manifest,
        include_payloads=include_payloads,
        include_attachments=include_attachments,
        include_video=include_video,
        max_bytes=max_bytes,
        max_files=max_files,
        chunk_size=chunk_size,
    )

    sidecar = _read_sidecar(destination, storage_options=storage_options, auth_ref=auth_ref) if resume else {}
    files: list[dict[str, Any]] = []
    running_bytes = 0
    copied = 0
    skipped = 0

    for chunk_index, chunk in enumerate(plan["chunks"]):
        # Group the chunk's objects by (table, version) so a single checkout /
        # rowid index / take_blobs call serves the whole chunk (bounded memory).
        chunk_by_table: dict[tuple[str, str, str, int | None], list[dict[str, Any]]] = {}
        for obj in chunk:
            key = (obj["table"], obj["id_column"], obj["column"], obj["table_version"])
            chunk_by_table.setdefault(key, []).append(obj)

        for (table, id_column, column, version), members in chunk_by_table.items():
            done_paths = {
                obj["path"]
                for obj in members
                if _already_done(obj, sidecar, destination, storage_options, auth_ref)
            }
            pending = [obj for obj in members if obj["path"] not in done_paths]
            blobs = _fetch_chunk_blobs(lake, table, id_column, column, [obj["row_id"] for obj in pending], version)
            for obj in members:
                if obj["path"] in done_paths:
                    prior = sidecar[obj["path"]]
                    files.append({**_file_record(obj, int(prior["bytes"]), str(prior["sha256"])), "reused": True})
                    running_bytes += int(prior["bytes"])
                    skipped += 1
                    continue
                payload = blobs.get(obj["row_id"])
                if payload is None:
                    raise EvidencePackError(
                        f"missing source bytes for {obj['table']} {obj['row_id']!r}"
                    )
                running_bytes += len(payload)
                if max_bytes is not None and running_bytes > max_bytes:
                    raise EvidencePackError(
                        f"materialization exceeded max_bytes={max_bytes} at {running_bytes} bytes"
                    )
                sha = hashlib.sha256(payload).hexdigest()
                write_binary_uri(
                    join_uri(destination, obj["path"]),
                    payload,
                    storage_options=storage_options,
                    auth_ref=auth_ref,
                )
                sidecar[obj["path"]] = {"bytes": len(payload), "sha256": sha}
                files.append({**_file_record(obj, len(payload), sha), "reused": False})
                copied += 1

        _write_sidecar(destination, sidecar, storage_options=storage_options, auth_ref=auth_ref)
        if progress is not None:
            progress(
                {
                    "chunk": chunk_index + 1,
                    "chunks": plan["chunk_count"],
                    "copied": copied,
                    "skipped": skipped,
                    "bytes": running_bytes,
                }
            )

    files.sort(key=lambda item: (item["kind"], item["path"]))
    out_manifest = _materialized_manifest(manifest, files, destination)
    manifest_path = join_uri(destination, _MANIFEST_NAME)
    write_binary_uri(
        manifest_path,
        (json.dumps(_json_ready(out_manifest), indent=2, sort_keys=True) + "\n").encode(),
        storage_options=storage_options,
        auth_ref=auth_ref,
    )

    digest = _stable_sha256(manifest)
    status = "materialized"
    if record:
        _upsert_materialized_row(
            lake,
            manifest,
            digest=digest,
            output_uri=str(destination),
            bytes_total=running_bytes,
            file_count=len(files),
            status=status,
            retention_policy=retention_policy,
            protected=protected,
            matched=matched,
            created_by=created_by,
        )
    _emit_event(
        lake,
        pack_id=digest,
        manifest_digest=digest,
        event_type="materialized",
        manifest=manifest,
        status="ok",
        detail=f"copied={copied} skipped={skipped} bytes={running_bytes}",
        created_by=created_by,
        metadata={"output_uri": str(destination), "object_store": "true" if is_object_dest else "false"},
        bytes_total=running_bytes,
        file_count=len(files),
        output_uri=str(destination),
    )

    return MaterializationReport(
        lake_uri=str(manifest.get("lake_uri") or lake.uri),
        pack_id=digest,
        manifest_digest=digest,
        output_uri=str(destination),
        manifest_path=str(manifest_path),
        status=status,
        bytes_total=running_bytes,
        file_count=len(files),
        copied_count=copied,
        skipped_count=skipped,
        files=tuple(files),
        chunk_count=plan["chunk_count"],
    )


# --- Table IO ---------------------------------------------------------------


def _upsert_row(lake: Lake, row: Mapping[str, Any]) -> None:
    table = pa.Table.from_pylist([dict(row)], schema=EVIDENCE_PACKS_SCHEMA)
    (
        lake.table(CATALOG_TABLE)
        .merge_insert("pack_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(table)
    )


def _upsert_materialized_row(
    lake: Lake,
    manifest: Mapping[str, Any],
    *,
    digest: str,
    output_uri: str,
    bytes_total: int,
    file_count: int,
    status: str,
    retention_policy: str,
    protected: bool,
    matched: Sequence[str],
    created_by: str | None,
) -> None:
    existing = _load_row(lake, pack_id=digest)
    now = datetime.now(UTC)
    subject = manifest.get("subject") or {}
    row = {
        "pack_id": digest,
        "manifest_digest": digest,
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "manifest_schema_version": str(manifest.get("schema_version")),
        "subject_kind": str(subject.get("type") or ""),
        "subject_handle": _subject_handle(subject),
        "lake_uri": str(manifest.get("lake_uri") or lake.uri),
        "mode": "materialized",
        "materialization_status": status,
        "output_uri": output_uri,
        "bytes_total": int(bytes_total),
        "file_count": int(file_count),
        "row_id_count": _row_id_count(manifest),
        "source_coordinate_hashes": _coordinate_hashes(manifest),
        "table_version_pins": _version_pins(manifest),
        "retention_policy": (
            str(existing.get("retention_policy")) if existing else str(retention_policy or DEFAULT_RETENTION_POLICY)
        ),
        "protected": bool(existing.get("protected")) if existing else bool(protected),
        "expires_at": _as_dt(existing.get("expires_at")) if existing else None,
        "redaction_policy": str(existing.get("redaction_policy") or "") if existing else "",
        "redacted": bool(matched) or (bool(existing.get("redacted")) if existing else False),
        "sensitive_sources": list(matched) or (list(existing.get("sensitive_sources") or []) if existing else []),
        "manifest_json": _encode_manifest(manifest),
        "metadata": list(existing.get("metadata") or []) if existing else [],
        "created_by": (str(existing.get("created_by")) if existing else (created_by or "")),
        "created_at": _row_timestamp(existing, "created_at", now),
        "updated_at": now,
    }
    _upsert_row(lake, row)


def _emit_event(
    lake: Lake,
    *,
    pack_id: str,
    manifest_digest: str,
    event_type: str,
    status: str,
    detail: str,
    created_by: str | None,
    manifest: Mapping[str, Any] | None = None,
    row: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    bytes_total: int = 0,
    file_count: int = 0,
    output_uri: str = "",
) -> None:
    now = datetime.now(UTC)
    if manifest is not None:
        subject = manifest.get("subject") or {}
        subject_kind = str(subject.get("type") or "")
        subject_handle = _subject_handle(subject)
        mode = str(manifest.get("mode") or "")
    elif row is not None:
        subject_kind = str(row.get("subject_kind") or "")
        subject_handle = str(row.get("subject_handle") or "")
        mode = str(row.get("mode") or "")
    else:
        subject_kind = subject_handle = mode = ""

    event_id = hashlib.sha256(
        json.dumps(
            [pack_id, event_type, status, now.isoformat(), dict(metadata or {})],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    event = {
        "event_id": event_id,
        "pack_id": pack_id,
        "manifest_digest": manifest_digest,
        "event_type": event_type,
        "subject_kind": subject_kind,
        "subject_handle": subject_handle,
        "mode": mode,
        "status": status,
        "bytes_total": int(bytes_total),
        "file_count": int(file_count),
        "output_uri": output_uri,
        "detail": detail,
        "metadata": _kv_items(metadata),
        "created_by": created_by or "",
        "created_at": now,
    }
    table = pa.Table.from_pylist([event], schema=EVIDENCE_PACK_EVENTS_SCHEMA)
    lake.table(EVENTS_TABLE).add(table)


def evidence_pack_events(
    lake: Lake,
    *,
    pack_id: str | None = None,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return audit events, optionally filtered by pack id / event type."""
    predicates: list[str] = []
    if pack_id is not None:
        predicates.append(f"pack_id = {_sql_literal(pack_id)}")
    if event_type is not None:
        predicates.append(f"event_type = {_sql_literal(event_type)}")
    table = lake.table(EVENTS_TABLE)
    handle = table.to_lance()
    where = " AND ".join(predicates) if predicates else None
    arrow = handle.to_table(filter=where) if where else handle.to_table()
    rows = arrow.to_pylist()
    rows.sort(key=lambda item: (_as_dt(item.get("created_at")), str(item.get("event_id"))))
    return _json_ready(rows)


def _load_row(lake: Lake, *, pack_id: str) -> dict[str, Any] | None:
    rows = _load_rows_where(lake, f"pack_id = {_sql_literal(pack_id)}")
    return rows[0] if rows else None


def _load_rows_where(lake: Lake, where: str | None) -> list[dict[str, Any]]:
    handle = lake.table(CATALOG_TABLE).to_lance()
    arrow = handle.to_table(filter=where) if where else handle.to_table()
    return arrow.to_pylist()


# --- Materialization helpers ------------------------------------------------


def _fetch_chunk_blobs(
    lake: Lake,
    table: str,
    id_column: str,
    column: str,
    ids: Sequence[str],
    version: int | None,
) -> dict[str, bytes]:
    if not ids:
        return {}
    handle = lake.table(table)
    if version is not None:
        handle.checkout(int(version))
    try:
        require_lake_capability(lake, BLOB, operation="blob hydration")
        dataset = handle.to_lance()
        index = dataset.to_table(columns=[id_column], with_row_id=True)
        rowid_by_id = dict(
            zip(index[id_column].to_pylist(), index["_rowid"].to_pylist(), strict=True)
        )
        wanted = {str(row_id): rowid_by_id.get(str(row_id)) for row_id in ids}
        present = {rid: rowid for rid, rowid in wanted.items() if rowid is not None}
        by_rowid = fetch_blobs_by_row_id(dataset, column, list(present.values()))
        return {rid: by_rowid.get(rowid, b"") for rid, rowid in present.items()}
    finally:
        if version is not None:
            handle.checkout_latest()


def _already_done(
    obj: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    destination: str,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> bool:
    prior = sidecar.get(obj["path"])
    if not prior:
        return False
    return uri_exists(
        join_uri(destination, obj["path"]),
        storage_options=storage_options,
        auth_ref=auth_ref,
    )


def _read_sidecar(
    destination: str,
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> dict[str, Any]:
    from lancedb_robotics.storage import open_binary_uri

    uri = join_uri(destination, _PROGRESS_SIDECAR)
    if not uri_exists(uri, storage_options=storage_options, auth_ref=auth_ref):
        return {}
    try:
        with open_binary_uri(uri, storage_options=storage_options, auth_ref=auth_ref) as stream:
            data = stream.read()
    except Exception:  # noqa: BLE001 - a corrupt sidecar just restarts the copy
        return {}
    try:
        decoded = json.loads(data)
    except (ValueError, TypeError):
        return {}
    return decoded.get("files", {}) if isinstance(decoded, dict) else {}


def _write_sidecar(
    destination: str,
    sidecar: Mapping[str, Any],
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> None:
    payload = json.dumps({"files": dict(sorted(sidecar.items()))}, sort_keys=True).encode()
    write_binary_uri(
        join_uri(destination, _PROGRESS_SIDECAR),
        payload,
        storage_options=storage_options,
        auth_ref=auth_ref,
    )


def _materialized_manifest(
    manifest: Mapping[str, Any],
    files: Sequence[Mapping[str, Any]],
    destination: str,
) -> dict[str, Any]:
    out = dict(_json_ready(manifest))
    out["mode"] = "materialized"
    out["output_uri"] = destination
    out["materialized_files"] = [
        {k: v for k, v in dict(record).items() if k != "reused"} for record in files
    ]
    verification = dict(out.get("verification") or {})
    verification["materialized_file_hashes"] = [
        {"path": record["path"], "sha256": record["sha256"], "bytes": record["bytes"]}
        for record in files
    ]
    out["verification"] = verification
    return out


def _resolve_destination(output_dir: str | Path | None, output_uri: str | None) -> str:
    if (output_dir is None) == (output_uri is None):
        raise EvidencePackError("pass exactly one of output_dir= or output_uri=")
    if output_uri is not None:
        return str(output_uri)
    return str(Path(output_dir))


def _file_record(obj: Mapping[str, Any], size: int, sha: str) -> dict[str, Any]:
    return {
        "kind": obj["kind"],
        "table": obj["table"],
        "row_id": obj["row_id"],
        "path": obj["path"],
        "bytes": int(size),
        "sha256": sha,
    }


# --- Manifest / row parsing helpers -----------------------------------------


def _manifest_of(pack: EvidencePackReport | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(pack, EvidencePackReport):
        return dict(pack.manifest)
    if isinstance(pack, Mapping):
        if "manifest" in pack and "schema_version" not in pack:
            return dict(pack["manifest"])
        return dict(pack)
    raise EvidencePackError("pack must be an EvidencePackReport or a manifest mapping")


def _encode_manifest(manifest: Mapping[str, Any]) -> str:
    return json.dumps(_json_ready(manifest), sort_keys=True, separators=(",", ":"))


def _decode_manifest(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("manifest_json")
    if not raw:
        raise EvidencePackError(
            f"catalog row {row.get('pack_id')!r} has no stored manifest"
        )
    return json.loads(raw)


def _subject_handle(subject: Mapping[str, Any]) -> str:
    return str(subject.get("handle") or subject.get("model_run_id") or "")


def _coordinate_hashes(manifest: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(coord["coordinate_hash"])
        for coord in manifest.get("source_coordinates") or []
        if coord.get("coordinate_hash")
    )


def _version_pins(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"table": str(row.get("table")), "version": int(row.get("version")), "tag": str(row.get("tag") or "")}
        for row in manifest.get("table_versions") or []
        if row.get("table") is not None and row.get("version") is not None
    ]


def _row_id_count(manifest: Mapping[str, Any]) -> int:
    return sum(len(ids) for ids in (manifest.get("row_ids") or {}).values())


def _verification_bytes(manifest: Mapping[str, Any]) -> int:
    files = manifest.get("materialized_files") or []
    return sum(int(row.get("bytes") or 0) for row in files)


def _initial_status(manifest: Mapping[str, Any]) -> str:
    if manifest.get("materialized_files"):
        return "materialized"
    return "planned"


def _table_version(manifest: Mapping[str, Any], table_name: str) -> int | None:
    for row in manifest.get("table_versions") or []:
        if row.get("table") == table_name and row.get("version") is not None:
            return int(row["version"])
    return None


def _match_sensitive_sources(
    manifest: Mapping[str, Any],
    patterns: Sequence[str],
) -> list[str]:
    if not patterns:
        return []
    uris: set[str] = set()
    for coord in manifest.get("source_coordinates") or []:
        if coord.get("uri"):
            uris.add(str(coord["uri"]))
    for ref in manifest.get("payload_refs") or []:
        if ref.get("raw_uri"):
            uris.add(str(ref["raw_uri"]))
    for ref in manifest.get("video_refs") or []:
        for key in ("uri", "raw_uri"):
            if ref.get(key):
                uris.add(str(ref[key]))
    matched: set[str] = set()
    for uri in uris:
        for pattern in patterns:
            if fnmatch.fnmatch(uri, pattern) or str(pattern) in uri:
                matched.add(uri)
                break
    return sorted(matched)


def _row_to_entry(row: Mapping[str, Any]) -> EvidencePackCatalogEntry:
    return EvidencePackCatalogEntry(
        pack_id=str(row.get("pack_id") or ""),
        manifest_digest=str(row.get("manifest_digest") or ""),
        catalog_schema_version=str(row.get("catalog_schema_version") or CATALOG_SCHEMA_VERSION),
        manifest_schema_version=str(row.get("manifest_schema_version") or ""),
        subject_kind=str(row.get("subject_kind") or ""),
        subject_handle=str(row.get("subject_handle") or ""),
        lake_uri=str(row.get("lake_uri") or ""),
        mode=str(row.get("mode") or ""),
        materialization_status=str(row.get("materialization_status") or ""),
        output_uri=str(row.get("output_uri") or ""),
        bytes_total=int(row.get("bytes_total") or 0),
        file_count=int(row.get("file_count") or 0),
        row_id_count=int(row.get("row_id_count") or 0),
        source_coordinate_hashes=tuple(str(v) for v in row.get("source_coordinate_hashes") or []),
        table_version_pins=tuple(dict(v) for v in row.get("table_version_pins") or []),
        retention_policy=str(row.get("retention_policy") or DEFAULT_RETENTION_POLICY),
        protected=bool(row.get("protected")),
        expires_at=_as_dt(row.get("expires_at")),
        redaction_policy=str(row.get("redaction_policy") or ""),
        redacted=bool(row.get("redacted")),
        sensitive_sources=tuple(str(v) for v in row.get("sensitive_sources") or []),
        metadata=_kv_to_dict(row.get("metadata")),
        created_by=str(row.get("created_by") or ""),
        created_at=_as_dt(row.get("created_at")) or datetime.now(UTC),
        updated_at=_as_dt(row.get("updated_at")) or datetime.now(UTC),
    )


# --- Small shared utilities -------------------------------------------------


def _kind_for(rel_dir: str) -> str:
    return {
        "payloads": "payload",
        "attachments": "attachment",
        "video_encodings": "video-encoding",
    }[rel_dir]


def _retention_cutoff(
    older_than: timedelta | datetime | None,
    reference: datetime,
) -> datetime | None:
    if older_than is None:
        return None
    if isinstance(older_than, timedelta):
        return reference - older_than
    return older_than


def _row_timestamp(row: Mapping[str, Any] | None, key: str, default: datetime) -> datetime:
    if row is None:
        return default
    return _as_dt(row.get(key)) or default


def _as_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _maybe_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _kv_items(metadata: Mapping[str, Any] | None) -> list[dict[str, str]]:
    if not metadata:
        return []
    return [
        {"key": str(key), "value": "" if value is None else str(value)}
        for key, value in sorted(metadata.items())
    ]


def _kv_to_dict(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value or []:
        if isinstance(item, Mapping) and item.get("key") is not None:
            result[str(item["key"])] = "" if item.get("value") is None else str(item["value"])
    return result


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"offset": int(offset)}).encode()).decode()


def _decode_cursor(cursor: str) -> int:
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        return max(0, int(decoded["offset"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise EvidencePackError(f"invalid cursor {cursor!r}") from exc
