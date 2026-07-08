"""Queryable external-context catalog, indexing, and retention (backlog 0114).

Backlog 0068 stores external run/job/code/environment context inside manifest and
transform JSON fields, and 0098 projects executions into ``lineage_executions``.
That is enough for v1 traceability, but platform teams need to *look up* a
canonical execution by an external run/job handle, govern which context is
retained, and redact environment metadata before sharing audits. This module
adds, on top of those existing rows (never a second source of truth):

- a persisted catalog table (:data:`CATALOG_TABLE`) with one idempotent row per
  external context observed on a canonical execution or transform, keyed by a
  content digest so re-running the backfill never duplicates rows;
- an append-only audit log (:data:`EVENTS_TABLE`) for backfill, retention, and
  redaction/expiry events;
- :func:`backfill_external_contexts`, a bounded-memory batched scan of
  ``lineage_executions`` and ``transform_runs`` that extracts external handles
  and links them to canonical execution/artifact IDs;
- :func:`find_external_context`, a paged lookup by provider / external run ID /
  external job ID / code ref / environment digest / artifact URI that resolves
  to the canonical Lance execution and artifacts;
- retention/redaction governance (protection, expiry, legal/audit holds) that is
  queryable and survives catalog reloads.

Canonical execution/artifact IDs in Lance stay authoritative; this catalog only
indexes external handles pointing at them. Existing ``lineage_context`` JSON stays
readable for lakes that never build the catalog.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage_hooks import normalize_lineage_context
from lancedb_robotics.redaction import ContextRedactionPolicy
from lancedb_robotics.schemas import (
    EXTERNAL_CONTEXT_EVENTS_SCHEMA,
    EXTERNAL_CONTEXTS_SCHEMA,
)

CATALOG_TABLE = "external_contexts"
EVENTS_TABLE = "external_context_events"

#: Catalog row contract version.
CATALOG_SCHEMA_VERSION = "lancedb-robotics/external-context-catalog/v1"
#: Stored context-payload contract version.
CONTEXT_SCHEMA_VERSION = "lancedb-robotics/external-context/v1"

DEFAULT_RETENTION_POLICY = "default"
DEFAULT_BATCH_SIZE = 512

#: Source tables scanned by the backfill and the columns each contributes.
_EXECUTION_COLUMNS = (
    "execution_id",
    "provider",
    "code_ref",
    "params_json",
    "environment_json",
    "input_artifact_ids",
    "output_artifact_ids",
    "transform_id",
    "status",
)
_TRANSFORM_COLUMNS = ("transform_id", "params", "status")
BACKFILL_SOURCE_TABLES: tuple[str, ...] = ("lineage_executions", "transform_runs")


class ExternalContextError(Exception):
    """Raised for external-context catalog contract violations."""


# --- Catalog entry / reports ------------------------------------------------


@dataclass(frozen=True)
class ExternalContextEntry:
    """A single durable external-context catalog row."""

    context_id: str
    catalog_schema_version: str
    context_schema_version: str
    lake_uri: str
    provider: str
    external_run_id: str
    external_job_id: str
    external_parent_run_id: str
    external_url: str
    code_ref: str
    environment_digest: str
    artifact_uris: tuple[str, ...]
    source_table: str
    source_id: str
    execution_id: str
    artifact_ids: tuple[str, ...]
    transform_id: str
    status: str
    redacted: bool
    redaction_policy: str
    context: dict[str, Any]
    retention_policy: str
    protected: bool
    expires_at: datetime | None
    legal_hold: bool
    audit_hold: bool
    metadata: dict[str, str]
    created_by: str
    created_at: datetime
    updated_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "catalog_schema_version": self.catalog_schema_version,
            "context_schema_version": self.context_schema_version,
            "lake_uri": self.lake_uri,
            "provider": self.provider,
            "external_run_id": self.external_run_id,
            "external_job_id": self.external_job_id,
            "external_parent_run_id": self.external_parent_run_id,
            "external_url": self.external_url,
            "code_ref": self.code_ref,
            "environment_digest": self.environment_digest,
            "artifact_uris": list(self.artifact_uris),
            "source_table": self.source_table,
            "source_id": self.source_id,
            "execution_id": self.execution_id,
            "artifact_ids": list(self.artifact_ids),
            "transform_id": self.transform_id,
            "status": self.status,
            "redacted": self.redacted,
            "redaction_policy": self.redaction_policy,
            "context": dict(self.context),
            "retention_policy": self.retention_policy,
            "protected": self.protected,
            "expires_at": _iso_or_none(self.expires_at),
            "legal_hold": self.legal_hold,
            "audit_hold": self.audit_hold,
            "metadata": dict(self.metadata),
            "created_by": self.created_by,
            "created_at": _iso_or_none(self.created_at),
            "updated_at": _iso_or_none(self.updated_at),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class ExternalContextPage:
    """A bounded page of catalog entries plus an opaque continuation cursor."""

    contexts: tuple[ExternalContextEntry, ...]
    next_cursor: str | None
    page_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "contexts": [entry.as_dict() for entry in self.contexts],
            "next_cursor": self.next_cursor,
            "page_size": self.page_size,
            "count": len(self.contexts),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class BackfillReport:
    """Result of a bounded external-context backfill run."""

    lake_uri: str
    scanned: int
    recorded: int
    updated: int
    skipped: int
    batches: int
    sources: tuple[str, ...]
    redaction_policy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "scanned": self.scanned,
            "recorded": self.recorded,
            "updated": self.updated,
            "skipped": self.skipped,
            "batches": self.batches,
            "sources": list(self.sources),
            "redaction_policy": self.redaction_policy,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


# --- Record / backfill ------------------------------------------------------


def record_external_context(
    lake: Lake,
    *,
    provider: str | None = None,
    external_run_id: str | None = None,
    external_job_id: str | None = None,
    external_parent_run_id: str | None = None,
    external_url: str | None = None,
    code_ref: str | None = None,
    environment_digest: str | None = None,
    artifact_uris: Iterable[str] = (),
    execution_id: str | None = None,
    artifact_ids: Iterable[str] = (),
    transform_id: str | None = None,
    status: str | None = None,
    context: Mapping[str, Any] | None = None,
    source_table: str = "manual",
    source_id: str | None = None,
    redaction_policy: ContextRedactionPolicy | None = None,
    retention_policy: str = DEFAULT_RETENTION_POLICY,
    protected: bool = False,
    expires_at: datetime | None = None,
    legal_hold: bool = False,
    audit_hold: bool = False,
    metadata: Mapping[str, Any] | None = None,
    created_by: str | None = None,
    now: datetime | None = None,
) -> ExternalContextEntry:
    """Record one external-context row, idempotent by content digest.

    The stored ``context`` payload has ``redaction_policy`` applied when one is
    supplied, so no denied/secret keys are persisted. Re-recording identical
    inputs maps to the same ``context_id`` and preserves ``created_at``.
    """

    extracted = _ExtractedContext(
        provider=_norm(provider),
        external_run_id=_norm(external_run_id),
        external_job_id=_norm(external_job_id),
        external_parent_run_id=_norm(external_parent_run_id),
        external_url=_norm(external_url),
        code_ref=_norm(code_ref),
        environment_digest=_norm(environment_digest),
        artifact_uris=tuple(sorted({str(u) for u in artifact_uris if u})),
        execution_id=_norm(execution_id),
        artifact_ids=tuple(sorted({str(a) for a in artifact_ids if a})),
        transform_id=_norm(transform_id),
        status=_norm(status),
        context=dict(context or {}),
    )
    if not extracted.has_external_handle:
        raise ExternalContextError(
            "record_external_context requires at least one external handle "
            "(provider / external_run_id / external_job_id / code_ref / "
            "environment_digest / artifact_uris)"
        )

    row, was_update = _build_row(
        lake,
        extracted,
        source_table=source_table,
        source_id=_norm(source_id) or extracted.source_fallback,
        redaction_policy=redaction_policy,
        retention_policy=retention_policy,
        protected=protected,
        expires_at=expires_at,
        legal_hold=legal_hold,
        audit_hold=audit_hold,
        metadata=metadata,
        created_by=created_by,
        now=now or datetime.now(UTC),
    )
    _upsert_row(lake, row)
    _emit_event(
        lake,
        context_id=row["context_id"],
        event_type="recorded",
        provider=row["provider"],
        external_run_id=row["external_run_id"],
        detail="updated" if was_update else "recorded",
        created_by=created_by,
        metadata={"redacted": "true"} if row["redacted"] else {},
    )
    return _row_to_entry(row)


def backfill_external_contexts(
    lake: Lake,
    *,
    sources: Sequence[str] = BACKFILL_SOURCE_TABLES,
    redaction_policy: ContextRedactionPolicy | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retention_policy: str = DEFAULT_RETENTION_POLICY,
    created_by: str | None = None,
    now: datetime | None = None,
) -> BackfillReport:
    """Extract external context from canonical rows into the catalog (idempotent).

    Scans each source table in bounded batches (``batch_size`` rows at a time),
    never loading a whole table into memory, and upserts one row per external
    handle. Re-running produces no duplicate rows because ``context_id`` is a
    content digest. ``lineage_executions`` is authoritative (it already resolves
    canonical execution/artifact IDs); ``transform_runs`` fills in transforms not
    yet projected.
    """

    if batch_size < 1:
        raise ExternalContextError("batch_size must be >= 1")

    reference = now or datetime.now(UTC)
    scanned = recorded = updated = skipped = batches = 0
    seen: set[str] = set()

    for source in sources:
        if source not in BACKFILL_SOURCE_TABLES:
            raise ExternalContextError(
                f"unsupported backfill source {source!r}; expected one of {BACKFILL_SOURCE_TABLES}"
            )
        columns = list(_EXECUTION_COLUMNS if source == "lineage_executions" else _TRANSFORM_COLUMNS)
        for batch in _iter_row_batches(lake, source, columns, batch_size):
            batches += 1
            pending: list[dict[str, Any]] = []
            for raw in batch:
                scanned += 1
                extracted = (
                    _extract_from_execution(raw)
                    if source == "lineage_executions"
                    else _extract_from_transform(raw)
                )
                if extracted is None or not extracted.has_external_handle:
                    skipped += 1
                    continue
                row, _ = _build_row(
                    lake,
                    extracted,
                    source_table=source,
                    source_id=extracted.source_fallback,
                    redaction_policy=redaction_policy,
                    retention_policy=retention_policy,
                    protected=False,
                    expires_at=None,
                    legal_hold=False,
                    audit_hold=False,
                    metadata=None,
                    created_by=created_by,
                    now=reference,
                )
                if row["context_id"] in seen:
                    skipped += 1
                    continue
                seen.add(row["context_id"])
                pending.append(row)

            if not pending:
                continue
            fresh, existing = _partition_by_existing(lake, pending)
            _upsert_rows(lake, pending)
            recorded += len(fresh)
            updated += len(existing)

    _emit_event(
        lake,
        context_id="",
        event_type="backfilled",
        provider="",
        external_run_id="",
        detail=f"scanned={scanned} recorded={recorded} updated={updated} skipped={skipped}",
        created_by=created_by,
        metadata={"sources": ",".join(sources)},
    )
    return BackfillReport(
        lake_uri=str(lake.uri),
        scanned=scanned,
        recorded=recorded,
        updated=updated,
        skipped=skipped,
        batches=batches,
        sources=tuple(sources),
        redaction_policy=redaction_policy.name if redaction_policy else "",
    )


# --- Lookup / load ----------------------------------------------------------


def find_external_context(
    lake: Lake,
    *,
    provider: str | None = None,
    external_run_id: str | None = None,
    external_job_id: str | None = None,
    external_parent_run_id: str | None = None,
    code_ref: str | None = None,
    environment_digest: str | None = None,
    artifact_uri: str | None = None,
    source_table: str | None = None,
    page_size: int = 50,
    cursor: str | None = None,
) -> ExternalContextPage:
    """Look up external contexts by handle, paged and deterministic.

    Scalar filters push down to SQL; ``artifact_uri`` filters the (list-valued)
    ``artifact_uris`` column after the scan. Results are ordered by
    ``(created_at desc, context_id desc)`` and paged by an opaque cursor. Each
    entry carries the canonical ``execution_id`` / ``artifact_ids`` /
    ``transform_id`` it resolves to.
    """

    if page_size < 1:
        raise ExternalContextError("page_size must be >= 1")

    predicates: list[str] = []
    for column, value in (
        ("provider", provider),
        ("external_run_id", external_run_id),
        ("external_job_id", external_job_id),
        ("external_parent_run_id", external_parent_run_id),
        ("code_ref", code_ref),
        ("environment_digest", environment_digest),
        ("source_table", source_table),
    ):
        if value is not None:
            predicates.append(f"{column} = {_sql_literal(value)}")

    rows = _load_rows_where(lake, " AND ".join(predicates) if predicates else None)
    if artifact_uri is not None:
        needle = str(artifact_uri)
        rows = [row for row in rows if needle in (list(row.get("artifact_uris") or []))]

    ordered = sorted(
        rows,
        key=lambda item: (_as_dt(item.get("created_at")) or _EPOCH, str(item.get("context_id"))),
        reverse=True,
    )
    start = _decode_cursor(cursor) if cursor else 0
    window = ordered[start : start + page_size]
    next_cursor = (
        _encode_cursor(start + page_size) if start + page_size < len(ordered) else None
    )
    return ExternalContextPage(
        contexts=tuple(_row_to_entry(row) for row in window),
        next_cursor=next_cursor,
        page_size=page_size,
    )


def get_external_context(lake: Lake, context_id: str) -> ExternalContextEntry:
    """Reload a single catalog entry by ``context_id``."""

    row = _load_row(lake, context_id=str(context_id))
    if row is None:
        raise ExternalContextError(f"no external context recorded for id {context_id!r}")
    return _row_to_entry(row)


# --- Retention / governance -------------------------------------------------


def set_external_context_retention(
    lake: Lake,
    context_id: str,
    *,
    protected: bool | None = None,
    retention_policy: str | None = None,
    expires_at: datetime | None = None,
    clear_expires_at: bool = False,
    legal_hold: bool | None = None,
    audit_hold: bool | None = None,
    created_by: str | None = None,
) -> ExternalContextEntry:
    """Update retention/protection/hold metadata for a recorded context row."""

    row = _load_row(lake, context_id=str(context_id))
    if row is None:
        raise ExternalContextError(f"no external context recorded for id {context_id!r}")

    if protected is not None:
        row["protected"] = bool(protected)
    if retention_policy is not None:
        row["retention_policy"] = str(retention_policy)
    if clear_expires_at:
        row["expires_at"] = None
    elif expires_at is not None:
        row["expires_at"] = expires_at
    if legal_hold is not None:
        row["legal_hold"] = bool(legal_hold)
    if audit_hold is not None:
        row["audit_hold"] = bool(audit_hold)
    row["updated_at"] = datetime.now(UTC)

    _upsert_row(lake, row)
    _emit_event(
        lake,
        context_id=str(row["context_id"]),
        event_type="retention-updated",
        provider=str(row.get("provider") or ""),
        external_run_id=str(row.get("external_run_id") or ""),
        detail=(
            f"protected={row['protected']} legal_hold={row['legal_hold']} "
            f"audit_hold={row['audit_hold']} policy={row['retention_policy']}"
        ),
        created_by=created_by,
    )
    return _row_to_entry(row)


def external_context_retention_plan(
    lake: Lake,
    *,
    older_than: timedelta | datetime | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Report which recorded contexts are held vs safe to expire, and why."""

    reference = now or datetime.now(UTC)
    cutoff = _retention_cutoff(older_than, reference)
    rows = _load_rows_where(lake, None)

    held: list[dict[str, Any]] = []
    expirable: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item.get("context_id"))):
        entry = _row_to_entry(row)
        expires_at = entry.expires_at
        expired = expires_at is not None and expires_at <= reference
        older = cutoff is not None and entry.created_at <= cutoff
        summary = {
            "context_id": entry.context_id,
            "provider": entry.provider,
            "external_run_id": entry.external_run_id,
            "created_at": _iso_or_none(entry.created_at),
            "expires_at": _iso_or_none(expires_at),
        }
        if entry.protected or entry.legal_hold or entry.audit_hold:
            reason = "protected" if entry.protected else ("legal-hold" if entry.legal_hold else "audit-hold")
            held.append({**summary, "reason": reason})
        elif expired or older or (cutoff is None and older_than is None):
            summary["reason"] = "expired" if expired else ("older-than-cutoff" if older else "unconstrained")
            expirable.append(summary)
        else:
            held.append({**summary, "reason": "within-retention"})

    return {
        "reference_time": _iso_or_none(reference),
        "cutoff": _iso_or_none(cutoff),
        "held": held,
        "expirable": expirable,
        "held_count": len(held),
        "expirable_count": len(expirable),
    }


def expire_external_context(
    lake: Lake,
    context_id: str,
    *,
    force: bool = False,
    created_by: str | None = None,
) -> dict[str, Any]:
    """Delete a catalog row, refused if held unless ``force`` is set."""

    row = _load_row(lake, context_id=str(context_id))
    if row is None:
        raise ExternalContextError(f"no external context recorded for id {context_id!r}")
    held = bool(row.get("protected") or row.get("legal_hold") or row.get("audit_hold"))
    if held and not force:
        raise ExternalContextError(
            f"external context {context_id!r} is held (protected/legal/audit); "
            "pass force=True to expire it"
        )

    lake.table(CATALOG_TABLE).delete(f"context_id = {_sql_literal(str(context_id))}")
    _emit_event(
        lake,
        context_id=str(context_id),
        event_type="expired",
        provider=str(row.get("provider") or ""),
        external_run_id=str(row.get("external_run_id") or ""),
        detail="forced" if force else "expired",
        created_by=created_by,
    )
    return {
        "context_id": str(context_id),
        "expired": True,
        "forced": bool(force),
        "was_held": held,
    }


def external_context_events(
    lake: Lake,
    *,
    context_id: str | None = None,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return audit events, optionally filtered by context id / event type."""

    predicates: list[str] = []
    if context_id is not None:
        predicates.append(f"context_id = {_sql_literal(context_id)}")
    if event_type is not None:
        predicates.append(f"event_type = {_sql_literal(event_type)}")
    handle = lake.table(EVENTS_TABLE).to_lance()
    where = " AND ".join(predicates) if predicates else None
    arrow = handle.to_table(filter=where) if where else handle.to_table()
    rows = arrow.to_pylist()
    rows.sort(key=lambda item: (_as_dt(item.get("created_at")) or _EPOCH, str(item.get("event_id"))))
    return _json_ready_events(rows)


# --- Extraction -------------------------------------------------------------


@dataclass(frozen=True)
class _ExtractedContext:
    provider: str
    external_run_id: str
    external_job_id: str
    external_parent_run_id: str
    external_url: str
    code_ref: str
    environment_digest: str
    artifact_uris: tuple[str, ...]
    execution_id: str
    artifact_ids: tuple[str, ...]
    transform_id: str
    status: str
    context: dict[str, Any]

    @property
    def has_external_handle(self) -> bool:
        return bool(
            self.provider
            or self.external_run_id
            or self.external_job_id
            or self.code_ref
            or self.environment_digest
            or self.artifact_uris
        )

    @property
    def source_fallback(self) -> str:
        return self.execution_id or self.transform_id or self.external_run_id


def _extract_from_execution(row: Mapping[str, Any]) -> _ExtractedContext | None:
    params = _load_json(row.get("params_json"))
    context = normalize_lineage_context(params.get("lineage_context") or params)
    payload = context.to_dict() if context else {}
    env_digest = (payload.get("environment_digest") or "") or _env_digest(row.get("environment_json"))
    outputs = tuple(sorted(str(a) for a in (row.get("output_artifact_ids") or []) if a))
    return _ExtractedContext(
        provider=_norm(payload.get("provider")) or _norm(row.get("provider")),
        external_run_id=_norm(payload.get("external_run_id")),
        external_job_id=_norm(payload.get("external_job_id")),
        external_parent_run_id=_norm(payload.get("external_parent_run_id")),
        external_url=_norm(payload.get("external_url")),
        code_ref=_norm(payload.get("code_ref")) or _norm(row.get("code_ref")),
        environment_digest=_norm(env_digest),
        artifact_uris=_artifact_uris(payload),
        execution_id=_norm(row.get("execution_id")),
        artifact_ids=outputs,
        transform_id=_norm(row.get("transform_id")),
        status=_norm(payload.get("status")) or _norm(row.get("status")),
        context=payload,
    )


def _extract_from_transform(row: Mapping[str, Any]) -> _ExtractedContext | None:
    params = _load_json(row.get("params"))
    context = normalize_lineage_context(params.get("lineage_context") or params)
    payload = context.to_dict() if context else {}
    return _ExtractedContext(
        provider=_norm(payload.get("provider")),
        external_run_id=_norm(payload.get("external_run_id")),
        external_job_id=_norm(payload.get("external_job_id")),
        external_parent_run_id=_norm(payload.get("external_parent_run_id")),
        external_url=_norm(payload.get("external_url")),
        code_ref=_norm(payload.get("code_ref")),
        environment_digest=_norm(payload.get("environment_digest")),
        artifact_uris=_artifact_uris(payload),
        execution_id="",
        artifact_ids=(),
        transform_id=_norm(row.get("transform_id")),
        status=_norm(payload.get("status")) or _norm(row.get("status")),
        context=payload,
    )


def _build_row(
    lake: Lake,
    extracted: _ExtractedContext,
    *,
    source_table: str,
    source_id: str,
    redaction_policy: ContextRedactionPolicy | None,
    retention_policy: str,
    protected: bool,
    expires_at: datetime | None,
    legal_hold: bool,
    audit_hold: bool,
    metadata: Mapping[str, Any] | None,
    created_by: str | None,
    now: datetime,
) -> tuple[dict[str, Any], bool]:
    context_id = _context_digest(extracted, source_table, source_id)
    stored_context = extracted.context
    redacted = False
    if redaction_policy is not None:
        redacted = redaction_policy.redacts(extracted.context)
        stored_context = redaction_policy.redact_context(extracted.context)

    existing = _load_row(lake, context_id=context_id)
    created_at = _row_timestamp(existing, "created_at", now)
    row = {
        "context_id": context_id,
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "lake_uri": str(lake.uri),
        "provider": extracted.provider,
        "external_run_id": extracted.external_run_id,
        "external_job_id": extracted.external_job_id,
        "external_parent_run_id": extracted.external_parent_run_id,
        "external_url": extracted.external_url,
        "code_ref": extracted.code_ref,
        "environment_digest": extracted.environment_digest,
        "artifact_uris": list(extracted.artifact_uris),
        "source_table": source_table,
        "source_id": source_id,
        "execution_id": extracted.execution_id,
        "artifact_ids": list(extracted.artifact_ids),
        "transform_id": extracted.transform_id,
        "status": extracted.status,
        "redacted": bool(redacted),
        "redaction_policy": redaction_policy.name if redaction_policy else "",
        "context_json": json.dumps(stored_context, sort_keys=True, separators=(",", ":")),
        "retention_policy": str(retention_policy or DEFAULT_RETENTION_POLICY),
        "protected": bool(existing.get("protected")) if existing else bool(protected),
        "expires_at": _as_dt(existing.get("expires_at")) if existing else expires_at,
        "legal_hold": bool(existing.get("legal_hold")) if existing else bool(legal_hold),
        "audit_hold": bool(existing.get("audit_hold")) if existing else bool(audit_hold),
        "metadata": _kv_items(metadata) if not existing else list(existing.get("metadata") or []),
        "created_by": (str(existing.get("created_by")) if existing else (created_by or "")),
        "created_at": created_at,
        "updated_at": now,
    }
    return row, existing is not None


# --- Table IO ---------------------------------------------------------------


def _upsert_row(lake: Lake, row: Mapping[str, Any]) -> None:
    _upsert_rows(lake, [row])


def _upsert_rows(lake: Lake, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=EXTERNAL_CONTEXTS_SCHEMA)
    (
        lake.table(CATALOG_TABLE)
        .merge_insert("context_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(table)
    )


def _emit_event(
    lake: Lake,
    *,
    context_id: str,
    event_type: str,
    provider: str,
    external_run_id: str,
    detail: str,
    created_by: str | None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    now = datetime.now(UTC)
    event_id = hashlib.sha256(
        json.dumps(
            [context_id, event_type, detail, now.isoformat(), dict(metadata or {})],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    event = {
        "event_id": event_id,
        "context_id": context_id,
        "event_type": event_type,
        "provider": provider,
        "external_run_id": external_run_id,
        "detail": detail,
        "metadata": _kv_items(metadata),
        "created_by": created_by or "",
        "created_at": now,
    }
    table = pa.Table.from_pylist([event], schema=EXTERNAL_CONTEXT_EVENTS_SCHEMA)
    lake.table(EVENTS_TABLE).add(table)


def _iter_row_batches(
    lake: Lake,
    table_name: str,
    columns: Sequence[str],
    batch_size: int,
) -> Iterable[list[dict[str, Any]]]:
    dataset = lake.table(table_name).to_lance()
    available = set(dataset.schema.names)
    projected = [column for column in columns if column in available]
    for batch in dataset.to_batches(columns=projected, batch_size=batch_size):
        if batch.num_rows:
            yield batch.to_pylist()


def _partition_by_existing(
    lake: Lake,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    ids = [str(row["context_id"]) for row in rows]
    literals = ", ".join(_sql_literal(cid) for cid in ids)
    existing_rows = _load_rows_where(lake, f"context_id IN ({literals})") if ids else []
    existing_ids = {str(row.get("context_id")) for row in existing_rows}
    fresh = [cid for cid in ids if cid not in existing_ids]
    return fresh, sorted(existing_ids)


def _load_row(lake: Lake, *, context_id: str) -> dict[str, Any] | None:
    rows = _load_rows_where(lake, f"context_id = {_sql_literal(context_id)}")
    return rows[0] if rows else None


def _load_rows_where(lake: Lake, where: str | None) -> list[dict[str, Any]]:
    handle = lake.table(CATALOG_TABLE).to_lance()
    arrow = handle.to_table(filter=where) if where else handle.to_table()
    return arrow.to_pylist()


def _row_to_entry(row: Mapping[str, Any]) -> ExternalContextEntry:
    return ExternalContextEntry(
        context_id=str(row.get("context_id") or ""),
        catalog_schema_version=str(row.get("catalog_schema_version") or CATALOG_SCHEMA_VERSION),
        context_schema_version=str(row.get("context_schema_version") or CONTEXT_SCHEMA_VERSION),
        lake_uri=str(row.get("lake_uri") or ""),
        provider=str(row.get("provider") or ""),
        external_run_id=str(row.get("external_run_id") or ""),
        external_job_id=str(row.get("external_job_id") or ""),
        external_parent_run_id=str(row.get("external_parent_run_id") or ""),
        external_url=str(row.get("external_url") or ""),
        code_ref=str(row.get("code_ref") or ""),
        environment_digest=str(row.get("environment_digest") or ""),
        artifact_uris=tuple(str(v) for v in row.get("artifact_uris") or []),
        source_table=str(row.get("source_table") or ""),
        source_id=str(row.get("source_id") or ""),
        execution_id=str(row.get("execution_id") or ""),
        artifact_ids=tuple(str(v) for v in row.get("artifact_ids") or []),
        transform_id=str(row.get("transform_id") or ""),
        status=str(row.get("status") or ""),
        redacted=bool(row.get("redacted")),
        redaction_policy=str(row.get("redaction_policy") or ""),
        context=_load_json(row.get("context_json")),
        retention_policy=str(row.get("retention_policy") or DEFAULT_RETENTION_POLICY),
        protected=bool(row.get("protected")),
        expires_at=_as_dt(row.get("expires_at")),
        legal_hold=bool(row.get("legal_hold")),
        audit_hold=bool(row.get("audit_hold")),
        metadata=_kv_to_dict(row.get("metadata")),
        created_by=str(row.get("created_by") or ""),
        created_at=_as_dt(row.get("created_at")) or datetime.now(UTC),
        updated_at=_as_dt(row.get("updated_at")) or datetime.now(UTC),
    )


# --- Small shared utilities -------------------------------------------------

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _context_digest(extracted: _ExtractedContext, source_table: str, source_id: str) -> str:
    payload = [
        source_table,
        source_id,
        extracted.provider,
        extracted.external_run_id,
        extracted.external_job_id,
        extracted.external_parent_run_id,
        extracted.code_ref,
        extracted.environment_digest,
        extracted.execution_id,
        list(extracted.artifact_uris),
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _artifact_uris(payload: Mapping[str, Any]) -> tuple[str, ...]:
    uris: set[str] = set()
    for ref in payload.get("artifact_refs") or []:
        if isinstance(ref, Mapping):
            uri = ref.get("artifact_uri") or ref.get("uri")
            if uri:
                uris.add(str(uri))
    return tuple(sorted(uris))


def _env_digest(environment_json: Any) -> str:
    env = _load_json(environment_json)
    return str(env.get("digest") or "")


def _load_json(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _json_ready_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["created_at"] = _iso_or_none(_as_dt(item.get("created_at")))
        item["metadata"] = _kv_to_dict(item.get("metadata"))
        out.append(item)
    return out


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
        raise ExternalContextError(f"invalid cursor {cursor!r}") from exc
