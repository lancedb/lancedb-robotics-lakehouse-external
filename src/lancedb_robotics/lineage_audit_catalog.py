"""Durable lineage-audit report catalog (backlog 0112).

Backlog 0067 added an in-process audit report for unresolved graph references,
missing sources/table versions, stale external links, retained versions, and
cleanup candidates. Production cleanup/release gates need those audits to be
durable: record a report once, reload it later by digest, list recent pass/fail
history, and export findings in bounded pages without re-running traversal.

This module adds a content-addressed catalog table:

- ``lineage_audit_reports`` stores one idempotent row per report digest.
- Summary counts are surfaced as scalar columns for cheap filtering/listing.
- The report payload stays inline in ``report_json`` for reload and finding
  export. Follow-on work can move large bodies to chunked/offloaded storage
  without changing the catalog identity.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import LineageAuditReport, LineageError
from lancedb_robotics.schemas import LINEAGE_AUDIT_REPORTS_SCHEMA

CATALOG_TABLE = "lineage_audit_reports"

CATALOG_SCHEMA_VERSION = "lancedb-robotics/lineage-audit-catalog/v1"
SUPPORTED_REPORT_SCHEMAS: tuple[str, ...] = ("lancedb-robotics/lineage-audit/v1",)
DEFAULT_AUDIT_MAX_AGE = timedelta(hours=24)

FINDING_FIELDS: tuple[str, ...] = (
    "unresolved_references",
    "missing_sources",
    "missing_table_versions",
    "stale_external_links",
    "retained_versions",
    "retention_holds",
    "cleanup_candidates",
)

BLOCKING_FINDING_FIELDS: tuple[str, ...] = (
    "unresolved_references",
    "missing_sources",
    "missing_table_versions",
    "stale_external_links",
)


class LineageAuditCatalogError(LineageError):
    """A lineage-audit catalog operation could not be completed."""


@dataclass(frozen=True)
class LineageAuditCatalogEntry:
    """A single durable lineage-audit catalog row."""

    report_id: str
    report_digest: str
    catalog_schema_version: str
    report_schema_version: str
    lake_uri: str
    subject: str
    root_artifact_ids: tuple[str, ...]
    status: str
    artifact_count: int
    edge_count: int
    finding_count: int
    unresolved_reference_count: int
    missing_source_count: int
    missing_table_version_count: int
    stale_external_link_count: int
    retained_version_count: int
    retention_hold_count: int
    cleanup_candidate_count: int
    validator_statuses: tuple[dict[str, Any], ...]
    metadata: dict[str, str]
    created_by: str
    created_at: datetime
    updated_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_digest": self.report_digest,
            "catalog_schema_version": self.catalog_schema_version,
            "report_schema_version": self.report_schema_version,
            "lake_uri": self.lake_uri,
            "subject": self.subject or None,
            "root_artifact_ids": list(self.root_artifact_ids),
            "status": self.status,
            "artifact_count": self.artifact_count,
            "edge_count": self.edge_count,
            "finding_count": self.finding_count,
            "summary": {
                "unresolved_references": self.unresolved_reference_count,
                "missing_sources": self.missing_source_count,
                "missing_table_versions": self.missing_table_version_count,
                "stale_external_links": self.stale_external_link_count,
                "retained_versions": self.retained_version_count,
                "retention_holds": self.retention_hold_count,
                "cleanup_candidates": self.cleanup_candidate_count,
            },
            "validator_statuses": [dict(row) for row in self.validator_statuses],
            "metadata": dict(self.metadata),
            "created_by": self.created_by,
            "created_at": _iso_or_none(self.created_at),
            "updated_at": _iso_or_none(self.updated_at),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class LineageAuditReportListPage:
    """A bounded page of audit-report catalog entries."""

    reports: tuple[LineageAuditCatalogEntry, ...]
    next_cursor: str | None
    page_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "reports": [entry.as_dict() for entry in self.reports],
            "next_cursor": self.next_cursor,
            "page_size": self.page_size,
            "count": len(self.reports),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class LineageAuditFindingPage:
    """A bounded page of findings from a stored audit report."""

    report_id: str
    findings: tuple[dict[str, Any], ...]
    next_cursor: str | None
    page_size: int
    total_findings: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "findings": [dict(row) for row in self.findings],
            "next_cursor": self.next_cursor,
            "page_size": self.page_size,
            "count": len(self.findings),
            "total_findings": self.total_findings,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


def record_audit_report(
    lake: Lake,
    report: LineageAuditReport | Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    created_by: str | None = None,
) -> LineageAuditCatalogEntry:
    """Record a lineage audit report in the durable catalog.

    Report identity is content-addressed over the report payload and the current
    lineage graph snapshot, excluding transient ids/timestamps. Re-recording the
    same report preserves the original ``created_at`` while refreshing
    ``updated_at``.
    """

    payload = _report_payload(report)
    schema_version = _require_supported_schema(payload)
    graph_snapshot = _graph_snapshot(lake)
    stored = _stored_report(payload, graph_snapshot)
    digest = _report_digest(stored)
    now = datetime.now(UTC)
    existing = _load_row(lake, digest)
    created_at = _row_timestamp(existing, "created_at", now)
    summary = _summary_counts(stored)
    status = _report_status(stored)

    row = {
        "report_id": digest,
        "report_digest": digest,
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "report_schema_version": schema_version,
        "lake_uri": str(stored.get("lake_uri") or lake.uri),
        "subject": str(stored.get("subject") or ""),
        "root_artifact_ids": [str(v) for v in stored.get("root_artifact_ids") or []],
        "status": status,
        "artifact_count": int(stored.get("artifact_count") or 0),
        "edge_count": int(stored.get("edge_count") or 0),
        "finding_count": sum(summary.values()),
        "unresolved_reference_count": summary["unresolved_references"],
        "missing_source_count": summary["missing_sources"],
        "missing_table_version_count": summary["missing_table_versions"],
        "stale_external_link_count": summary["stale_external_links"],
        "retained_version_count": summary["retained_versions"],
        "retention_hold_count": summary["retention_holds"],
        "cleanup_candidate_count": summary["cleanup_candidates"],
        "validator_statuses_json": json.dumps(
            _json_ready(stored.get("validator_statuses") or []),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "report_json": _encode_report(
            {
                **stored,
                "report_id": digest,
                "report_digest": digest,
                "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            }
        ),
        "metadata": _kv_items(metadata),
        "created_by": created_by or "",
        "created_at": created_at,
        "updated_at": now,
    }
    _upsert_row(lake, row)
    return _row_to_entry(row)


def get_audit_report(
    lake: Lake,
    report_id_or_digest: str,
) -> tuple[LineageAuditCatalogEntry, dict[str, Any]]:
    """Reload a catalog entry and its stored report payload by id/digest."""

    row = _load_row(lake, str(report_id_or_digest))
    if row is None:
        raise LineageAuditCatalogError(
            f"no lineage audit report recorded for {report_id_or_digest!r}"
        )
    report = _decode_report(row)
    _require_supported_schema(report)
    return _row_to_entry(row), report


def audit_reports(
    lake: Lake,
    *,
    status: str | None = None,
    subject: str | None = None,
    created_by: str | None = None,
    page_size: int = 50,
    cursor: str | None = None,
) -> LineageAuditReportListPage:
    """List persisted audit reports newest-first with cursor pagination."""

    if page_size < 1:
        raise LineageAuditCatalogError("page_size must be >= 1")

    predicates: list[str] = []
    if status is not None:
        predicates.append(f"status = {_sql_literal(_normalize_status(status))}")
    if subject is not None:
        predicates.append(f"subject = {_sql_literal(subject)}")
    if created_by is not None:
        predicates.append(f"created_by = {_sql_literal(created_by)}")

    rows = _load_rows_where(lake, " AND ".join(predicates) if predicates else None)
    ordered = sorted(
        rows,
        key=lambda item: (_as_dt(item.get("created_at")), str(item.get("report_id"))),
        reverse=True,
    )
    start = _decode_cursor(cursor) if cursor else 0
    window = ordered[start : start + page_size]
    next_cursor = (
        _encode_cursor(start + page_size) if start + page_size < len(ordered) else None
    )
    return LineageAuditReportListPage(
        reports=tuple(_row_to_entry(row) for row in window),
        next_cursor=next_cursor,
        page_size=page_size,
    )


def audit_findings(
    lake: Lake,
    report_id_or_digest: str,
    *,
    finding_type: str | None = None,
    page_size: int = 100,
    cursor: str | None = None,
) -> LineageAuditFindingPage:
    """Return a bounded page of findings from a stored audit report."""

    if page_size < 1:
        raise LineageAuditCatalogError("page_size must be >= 1")
    entry, report = get_audit_report(lake, report_id_or_digest)
    records = _finding_records(report, report_id=entry.report_id)
    if finding_type is not None:
        normalized = _normalize_finding_type(finding_type)
        records = [row for row in records if row["finding_type"] == normalized]
    start = _decode_cursor(cursor) if cursor else 0
    window = records[start : start + page_size]
    next_cursor = (
        _encode_cursor(start + page_size) if start + page_size < len(records) else None
    )
    return LineageAuditFindingPage(
        report_id=entry.report_id,
        findings=tuple(window),
        next_cursor=next_cursor,
        page_size=page_size,
        total_findings=len(records),
    )


def iter_audit_findings_ndjson(
    lake: Lake,
    report_id_or_digest: str,
    *,
    finding_type: str | None = None,
    page_size: int = 512,
    include_summary: bool = False,
):
    """Yield stored audit findings as one JSON record per line."""

    cursor: str | None = None
    emitted = 0
    report_id = ""
    while True:
        page = audit_findings(
            lake,
            report_id_or_digest,
            finding_type=finding_type,
            page_size=page_size,
            cursor=cursor,
        )
        report_id = page.report_id
        for row in page.findings:
            emitted += 1
            yield json.dumps(_json_ready(row), sort_keys=True, separators=(",", ":"))
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    if include_summary:
        yield json.dumps(
            {
                "record_type": "summary",
                "report_id": report_id,
                "finding_type": finding_type,
                "record_count": emitted,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def require_recent_passed_audit_report(
    lake: Lake,
    *,
    subject: str | None = None,
    max_age: timedelta | None = DEFAULT_AUDIT_MAX_AGE,
    now: datetime | None = None,
) -> LineageAuditCatalogEntry:
    """Return the newest passed report or raise with an actionable diagnostic."""

    reference = now or datetime.now(UTC)
    predicates = ["status = 'passed'"]
    predicates.append(
        f"subject = {_sql_literal(subject)}" if subject is not None else "subject = ''"
    )
    rows = _load_rows_where(lake, " AND ".join(predicates))
    if not rows:
        raise LineageAuditCatalogError(
            "cleanup requires a recent passed lineage audit report, but none is "
            "recorded; run `lancedb-robotics lineage audit --record --lake ...` first"
        )
    rows.sort(key=lambda item: (_as_dt(item.get("created_at")), str(item.get("report_id"))))
    newest = rows[-1]
    created_at = _as_dt(newest.get("created_at"))
    if max_age is not None:
        if created_at is None:
            raise LineageAuditCatalogError(
                f"audit report {newest.get('report_id')!r} has no created_at timestamp"
            )
        age = reference - created_at
        if age > max_age:
            raise LineageAuditCatalogError(
                f"latest passed lineage audit report {newest.get('report_id')!r} "
                f"is stale ({age.total_seconds():.0f}s old; max_age="
                f"{max_age.total_seconds():.0f}s); record a new audit before cleanup"
            )
    return _row_to_entry(newest)


def _upsert_row(lake: Lake, row: Mapping[str, Any]) -> None:
    table = pa.Table.from_pylist([dict(row)], schema=LINEAGE_AUDIT_REPORTS_SCHEMA)
    (
        lake.table(CATALOG_TABLE)
        .merge_insert("report_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(table)
    )


def _load_row(lake: Lake, report_id_or_digest: str) -> dict[str, Any] | None:
    literal = _sql_literal(report_id_or_digest)
    rows = _load_rows_where(lake, f"report_id = {literal} OR report_digest = {literal}")
    return rows[0] if rows else None


def _load_rows_where(lake: Lake, where: str | None) -> list[dict[str, Any]]:
    handle = lake.table(CATALOG_TABLE).to_lance()
    arrow = handle.to_table(filter=where) if where else handle.to_table()
    return arrow.to_pylist()


def _report_payload(report: LineageAuditReport | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(report, LineageAuditReport):
        return dict(report.as_dict())
    if hasattr(report, "as_dict") and callable(report.as_dict):
        return dict(report.as_dict())
    if isinstance(report, Mapping):
        return dict(report)
    raise LineageAuditCatalogError("report must be a LineageAuditReport or mapping")


def _require_supported_schema(report: Mapping[str, Any]) -> str:
    schema = report.get("schema_version")
    if schema not in SUPPORTED_REPORT_SCHEMAS:
        raise LineageAuditCatalogError(
            f"unsupported lineage-audit schema {schema!r}; expected one of "
            f"{SUPPORTED_REPORT_SCHEMAS}"
        )
    return str(schema)


def _stored_report(
    payload: Mapping[str, Any],
    graph_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    stored = _json_ready(dict(payload))
    stored.pop("report_id", None)
    stored.pop("report_digest", None)
    stored.pop("catalog_schema_version", None)
    stored["graph_snapshot"] = _json_ready(graph_snapshot)
    stored.setdefault("status", _report_status(stored))
    stored.setdefault("summary", _summary_counts(stored))
    return stored


def _report_digest(stored: Mapping[str, Any]) -> str:
    canonical = dict(stored)
    canonical.pop("generated_at", None)
    canonical.pop("page", None)
    return _full_digest(canonical)


def _graph_snapshot(lake: Lake) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    for table_name in ("lineage_artifacts", "lineage_edges", "lineage_executions"):
        try:
            handle = lake.table(table_name)
            tables.append(
                {
                    "table": table_name,
                    "version": int(handle.version),
                    "row_count": int(handle.count_rows()),
                }
            )
        except Exception as exc:  # noqa: BLE001 - snapshot is diagnostic metadata
            tables.append({"table": table_name, "error": str(exc)})
    return {"tables": tables}


def _summary_counts(report: Mapping[str, Any]) -> dict[str, int]:
    raw = report.get("summary")
    summary = raw if isinstance(raw, Mapping) else {}
    return {
        field: int(summary.get(field, len(report.get(field) or [])) or 0)
        for field in FINDING_FIELDS
    }


def _report_status(report: Mapping[str, Any]) -> str:
    existing = str(report.get("status") or "").strip().lower()
    if existing:
        return _normalize_status(existing)
    summary = _summary_counts(report)
    if any(summary[field] for field in BLOCKING_FINDING_FIELDS):
        return "failed"
    return "passed"


def _normalize_status(status: str) -> str:
    value = str(status or "").strip().lower()
    if value not in {"passed", "failed", "partial"}:
        raise LineageAuditCatalogError(
            f"unknown lineage-audit status {status!r}; expected passed, failed, or partial"
        )
    return value


def _finding_records(report: Mapping[str, Any], *, report_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for field in FINDING_FIELDS:
        for index, finding in enumerate(report.get(field) or []):
            records.append(
                {
                    "record_type": "finding",
                    "report_id": report_id,
                    "report_digest": report.get("report_digest") or report_id,
                    "finding_type": field,
                    "finding_index": index,
                    "finding": dict(finding),
                }
            )
    return records


def _normalize_finding_type(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "unresolved": "unresolved_references",
        "source": "missing_sources",
        "sources": "missing_sources",
        "table_version": "missing_table_versions",
        "table_versions": "missing_table_versions",
        "external_link": "stale_external_links",
        "external_links": "stale_external_links",
        "retained": "retained_versions",
        "holds": "retention_holds",
        "cleanup": "cleanup_candidates",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in FINDING_FIELDS:
        raise LineageAuditCatalogError(
            f"unknown finding_type {value!r}; expected one of {', '.join(FINDING_FIELDS)}"
        )
    return normalized


def _encode_report(stored: Mapping[str, Any]) -> str:
    return json.dumps(_json_ready(stored), sort_keys=True, separators=(",", ":"), default=str)


def _decode_report(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("report_json")
    if not raw:
        raise LineageAuditCatalogError(
            f"catalog row {row.get('report_id')!r} has no stored report payload"
        )
    return json.loads(raw)


def _row_to_entry(row: Mapping[str, Any]) -> LineageAuditCatalogEntry:
    return LineageAuditCatalogEntry(
        report_id=str(row.get("report_id") or ""),
        report_digest=str(row.get("report_digest") or ""),
        catalog_schema_version=str(row.get("catalog_schema_version") or CATALOG_SCHEMA_VERSION),
        report_schema_version=str(row.get("report_schema_version") or ""),
        lake_uri=str(row.get("lake_uri") or ""),
        subject=str(row.get("subject") or ""),
        root_artifact_ids=tuple(str(v) for v in row.get("root_artifact_ids") or []),
        status=str(row.get("status") or ""),
        artifact_count=int(row.get("artifact_count") or 0),
        edge_count=int(row.get("edge_count") or 0),
        finding_count=int(row.get("finding_count") or 0),
        unresolved_reference_count=int(row.get("unresolved_reference_count") or 0),
        missing_source_count=int(row.get("missing_source_count") or 0),
        missing_table_version_count=int(row.get("missing_table_version_count") or 0),
        stale_external_link_count=int(row.get("stale_external_link_count") or 0),
        retained_version_count=int(row.get("retained_version_count") or 0),
        retention_hold_count=int(row.get("retention_hold_count") or 0),
        cleanup_candidate_count=int(row.get("cleanup_candidate_count") or 0),
        validator_statuses=tuple(_decode_validator_statuses(row)),
        metadata=_kv_to_dict(row.get("metadata")),
        created_by=str(row.get("created_by") or ""),
        created_at=_as_dt(row.get("created_at")) or datetime.now(UTC),
        updated_at=_as_dt(row.get("updated_at")) or datetime.now(UTC),
    )


def _decode_validator_statuses(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("validator_statuses_json")
    if not raw:
        return []
    try:
        decoded = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [dict(item) for item in decoded if isinstance(item, Mapping)]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _canonical_json(payload: Any) -> str:
    return json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"), default=str)


def _full_digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


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


def _row_timestamp(row: Mapping[str, Any] | None, key: str, default: datetime) -> datetime:
    if row is None:
        return default
    return _as_dt(row.get(key)) or default


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


def _sql_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"offset": int(offset)}).encode()).decode()


def _decode_cursor(cursor: str) -> int:
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        return max(0, int(decoded["offset"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise LineageAuditCatalogError(f"invalid cursor {cursor!r}") from exc
