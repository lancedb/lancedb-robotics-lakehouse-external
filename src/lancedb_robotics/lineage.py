"""Regression tracing from model checkpoints to source-log coordinates.

Backlog 0033 makes a bad-checkpoint investigation a deterministic query over
existing lake state: ``model_outputs`` identify the model/checkpoint run and
the training snapshot, ``dataset_snapshots.table_versions`` pin the slice, and
``observations.raw_*`` columns point back to the original log messages.
"""

from __future__ import annotations

import ast
import base64
import bisect
import hashlib
import json
import logging
import operator
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import (
    CANONICAL_TABLES,
    LINEAGE_ARTIFACTS_SCHEMA,
    LINEAGE_EDGES_SCHEMA,
    LINEAGE_EXECUTIONS_SCHEMA,
)

logger = logging.getLogger(__name__)

LINEAGE_REPORT_VERSION = "lancedb-robotics/lineage-report/v1"


class LineageError(Exception):
    """Raised when a regression trace cannot be resolved."""


class RebuildPlanError(LineageError):
    """Raised when a rebuild plan cannot be built or a guardrail is violated."""


class RebuildPlanTooLarge(RebuildPlanError):
    """Raised when a rebuild plan exceeds a configured size guardrail.

    The message names the guardrail, the observed size, and the remediation
    (paginate with ``page_size``, narrow the scope, or raise the cap). Backlog
    0110 plan-size guardrails raise this so large-lake callers fail fast with an
    actionable error instead of materializing an unbounded plan.
    """


# Canonical rebuild/invalidation action vocabulary (backlog 0066). Action
# policies (backlog 0110) may only emit values from this set; anything else is
# rejected with an actionable error at construction or plan time.
KNOWN_REBUILD_ACTIONS = frozenset(
    {
        "quarantine",
        "recompute",
        "resnapshot",
        "retrain",
        "re-export",
        "re-evaluate",
        "notify-only",
    }
)


@dataclass(frozen=True)
class LineageArtifact:
    """Stable, version-aware handle for a lake artifact."""

    artifact_id: str
    kind: str
    name: str | None = None
    table_name: str | None = None
    table_version: int | None = None
    table_tag: str | None = None
    row_grain: str | None = None
    row_ids: tuple[str, ...] = ()
    source_uri: str | None = None
    source_id: str | None = None
    digest: str | None = None
    producer_execution_id: str | None = None
    metadata: dict[str, str] | None = None

    def as_row(self, created_at: datetime) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "name": self.name,
            "table_name": self.table_name,
            "table_version": self.table_version,
            "table_tag": self.table_tag,
            "row_grain": self.row_grain,
            "row_ids": list(self.row_ids),
            "source_uri": self.source_uri,
            "source_id": self.source_id,
            "digest": self.digest,
            "producer_execution_id": self.producer_execution_id,
            "metadata": _kv_items(self.metadata),
            "created_at": created_at,
        }


@dataclass(frozen=True)
class LineageExecution:
    """Execution node with provider/code/env provenance and artifact pins."""

    execution_id: str
    kind: str
    name: str | None = None
    transform_id: str | None = None
    status: str | None = None
    params: dict[str, Any] | None = None
    params_json: str | None = None
    code_ref: str | None = None
    provider: str | None = None
    environment: dict[str, Any] | None = None
    environment_json: str | None = None
    input_artifact_ids: tuple[str, ...] = ()
    output_artifact_ids: tuple[str, ...] = ()
    input_table_versions: tuple[dict[str, Any], ...] = ()
    output_table_versions: tuple[dict[str, Any], ...] = ()
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_by: str | None = None
    metadata: dict[str, str] | None = None

    def as_row(self, created_at: datetime) -> dict[str, Any]:
        params_json = self.params_json
        if params_json is None and self.params is not None:
            params_json = _json_dumps(self.params)
        environment_json = self.environment_json
        if environment_json is None and self.environment is not None:
            environment_json = _json_dumps(self.environment)
        return {
            "execution_id": self.execution_id,
            "kind": self.kind,
            "name": self.name,
            "transform_id": self.transform_id,
            "status": self.status,
            "params_json": params_json,
            "code_ref": self.code_ref,
            "provider": self.provider,
            "environment_json": environment_json,
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifact_ids": list(self.output_artifact_ids),
            "input_table_versions": list(self.input_table_versions),
            "output_table_versions": list(self.output_table_versions),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "created_by": self.created_by,
            "metadata": _kv_items(self.metadata),
            "created_at": created_at,
        }


@dataclass(frozen=True)
class LineageEdge:
    """Directed dependency edge: upstream artifact -> downstream artifact."""

    edge_id: str
    edge_type: str
    from_artifact_id: str
    to_artifact_id: str
    execution_id: str | None = None
    metadata: dict[str, str] | None = None

    def as_row(self, created_at: datetime) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "from_artifact_id": self.from_artifact_id,
            "to_artifact_id": self.to_artifact_id,
            "execution_id": self.execution_id,
            "metadata": _kv_items(self.metadata),
            "created_at": created_at,
        }


@dataclass(frozen=True)
class LineageGraph:
    """A deterministic subgraph returned by upstream/downstream traversal.

    When a caller passes ``page_size`` the graph is a bounded *page* of the full
    traversal: ``artifacts``/``edges`` carry only this page's slice while
    ``total_artifacts``/``total_edges`` stay stable across pages and
    ``next_page_token`` is an opaque continuation handle (``None`` on the last
    page). ``page_size`` is ``None`` for an unbounded traversal.
    """

    root_artifact_id: str
    direction: str
    artifacts: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    executions: tuple[dict[str, Any], ...]
    root_artifact_ids: tuple[str, ...] = ()
    resolved_handle: str | None = None
    page_size: int | None = None
    total_artifacts: int | None = None
    total_edges: int | None = None
    next_page_token: str | None = None
    truncated: bool = False
    controls: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(
        self,
        *,
        include_evidence: bool = True,
        max_artifacts: int | None = None,
        max_edges: int | None = None,
        max_executions: int | None = None,
    ) -> dict[str, Any]:
        root_ids = self.root_artifact_ids or (self.root_artifact_id,)
        artifacts = _lineage_report_artifacts(
            _limit_lineage_report_rows(self.artifacts, max_artifacts, "max_artifacts"),
            include_evidence=include_evidence,
        )
        edges = list(_limit_lineage_report_rows(self.edges, max_edges, "max_edges"))
        executions = list(
            _limit_lineage_report_rows(self.executions, max_executions, "max_executions")
        )
        warnings = _lineage_report_warnings(
            graph=self,
            max_artifacts=max_artifacts,
            max_edges=max_edges,
            max_executions=max_executions,
        )
        payload = {
            "report_version": LINEAGE_REPORT_VERSION,
            "report_type": "lineage-graph",
            "root_artifact_id": self.root_artifact_id,
            "root_artifact_ids": list(root_ids),
            "resolved_handle": self.resolved_handle or self.root_artifact_id,
            "direction": self.direction,
            "controls": {
                "traversal": dict(sorted(dict(self.controls).items())),
                "report": {
                    "include_evidence": include_evidence,
                    "max_artifacts": max_artifacts,
                    "max_edges": max_edges,
                    "max_executions": max_executions,
                },
            },
            "warnings": warnings,
            "evidence": _lineage_report_evidence(self.artifacts, include_evidence=include_evidence),
            "artifacts": artifacts,
            "edges": edges,
            "executions": executions,
        }
        if self.page_size is not None:
            payload["page"] = {
                "page_size": self.page_size,
                "total_artifacts": self.total_artifacts,
                "total_edges": self.total_edges,
                "returned_artifacts": len(self.artifacts),
                "next_page_token": self.next_page_token,
                "truncated": self.truncated,
            }
        return payload

    def iter_ndjson_records(
        self,
        *,
        include_evidence: bool = True,
        max_artifacts: int | None = None,
        max_edges: int | None = None,
        max_executions: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        payload = self.as_dict(
            include_evidence=include_evidence,
            max_artifacts=max_artifacts,
            max_edges=max_edges,
            max_executions=max_executions,
        )
        header = {
            "record_type": "report",
            "report_version": payload["report_version"],
            "report_type": payload["report_type"],
            "root_artifact_id": payload["root_artifact_id"],
            "root_artifact_ids": payload["root_artifact_ids"],
            "resolved_handle": payload["resolved_handle"],
            "direction": payload["direction"],
            "controls": payload["controls"],
            "warnings": payload["warnings"],
            "evidence": payload["evidence"],
        }
        if "page" in payload:
            header["page"] = payload["page"]
        records: list[dict[str, Any]] = [header]
        records.extend({"record_type": "artifact", **row} for row in payload["artifacts"])
        records.extend({"record_type": "edge", **row} for row in payload["edges"])
        records.extend({"record_type": "execution", **row} for row in payload["executions"])
        return tuple(records)


_LINEAGE_ARTIFACT_EVIDENCE_FIELDS = {
    "table_name",
    "table_version",
    "table_tag",
    "row_grain",
    "row_ids",
    "source_uri",
    "source_id",
    "digest",
}


def _limit_lineage_report_rows(
    rows: Sequence[dict[str, Any]],
    limit: int | None,
    label: str,
) -> tuple[dict[str, Any], ...]:
    if limit is None:
        return tuple(rows)
    if limit < 0:
        raise LineageError(f"{label} must be >= 0")
    return tuple(rows[:limit])


def _lineage_report_artifacts(
    rows: Sequence[dict[str, Any]],
    *,
    include_evidence: bool,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for row in rows:
        if include_evidence:
            artifacts.append(dict(row))
            continue
        artifacts.append(
            {
                key: value
                for key, value in row.items()
                if key not in _LINEAGE_ARTIFACT_EVIDENCE_FIELDS
            }
        )
    return artifacts


def _lineage_report_evidence(
    artifacts: Sequence[dict[str, Any]],
    *,
    include_evidence: bool,
) -> dict[str, Any]:
    if not include_evidence:
        return {
            "included": False,
            "source_coordinates": [],
            "table_versions": [],
        }
    source_coordinates: list[dict[str, Any]] = []
    table_versions: list[dict[str, Any]] = []
    for row in sorted(artifacts, key=lambda item: str(item.get("artifact_id") or "")):
        artifact_id = row.get("artifact_id")
        if row.get("source_uri") or row.get("source_id") or row.get("digest"):
            source_coordinate = {
                "artifact_id": artifact_id,
                "kind": row.get("kind"),
            }
            for key in ("source_uri", "source_id", "digest", "row_grain", "row_ids"):
                value = row.get(key)
                if value not in (None, "", []):
                    source_coordinate[key] = value
            source_coordinates.append(source_coordinate)
        if row.get("table_name") or row.get("table_version") is not None or row.get("table_tag"):
            table_version = {
                "artifact_id": artifact_id,
                "kind": row.get("kind"),
            }
            for key in ("table_name", "table_version", "table_tag", "row_grain", "row_ids"):
                value = row.get(key)
                if value not in (None, "", []):
                    table_version[key] = value
            table_versions.append(table_version)
    return {
        "included": True,
        "source_coordinates": source_coordinates,
        "table_versions": table_versions,
    }


def _lineage_report_warnings(
    *,
    graph: LineageGraph,
    max_artifacts: int | None,
    max_edges: int | None,
    max_executions: int | None,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for label, limit, total in (
        ("artifacts", max_artifacts, len(graph.artifacts)),
        ("edges", max_edges, len(graph.edges)),
        ("executions", max_executions, len(graph.executions)),
    ):
        if limit is not None and limit < total:
            warnings.append(
                {
                    "code": f"{label}-truncated",
                    "message": f"{label} rows truncated by report max_{label}",
                    "limit": limit,
                    "total": total,
                }
            )
    if graph.truncated:
        warnings.append(
            {
                "code": "page-truncated",
                "message": "lineage traversal page has additional rows",
                "next_page_token": graph.next_page_token,
            }
        )
    return warnings


@dataclass(frozen=True)
class LineageRefreshTableStatus:
    """Per-source-table version status compared against the last-refresh watermark."""

    table: str
    previous_version: int | None
    current_version: int
    changed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "previous_version": self.previous_version,
            "current_version": self.current_version,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class LineageRefreshPlan:
    """Auditable plan for one idempotent graph refresh (backlog 0097).

    Records the source table versions inspected against the last-refresh
    watermark, whether a full re-projection is required (and why), the resulting
    graph-row counts, and the stale graph rows retired or held. ``dry_run`` plans
    describe what a refresh *would* do without mutating the graph.
    """

    lake_uri: str
    action: str
    full_scan: bool
    full_scan_reason: str | None
    source_tables: tuple[LineageRefreshTableStatus, ...]
    changed_tables: tuple[str, ...]
    artifacts: int
    executions: int
    edges: int
    retired_artifacts: int = 0
    retired_edges: int = 0
    stale_artifacts: tuple[dict[str, Any], ...] = ()
    held_stale_artifacts: tuple[dict[str, Any], ...] = ()
    missing_indexes: tuple[dict[str, Any], ...] = ()
    previous_refreshed_at: str | None = None
    refreshed_at: str | None = None
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "lancedb-robotics/lineage-refresh-plan/v1",
            "lake_uri": self.lake_uri,
            "action": self.action,
            "dry_run": self.dry_run,
            "full_scan": self.full_scan,
            "full_scan_reason": self.full_scan_reason,
            "source_tables": [row.as_dict() for row in self.source_tables],
            "changed_tables": list(self.changed_tables),
            "artifacts": self.artifacts,
            "executions": self.executions,
            "edges": self.edges,
            "retired_artifacts": self.retired_artifacts,
            "retired_edges": self.retired_edges,
            "stale_artifacts": list(self.stale_artifacts),
            "held_stale_artifacts": list(self.held_stale_artifacts),
            "missing_indexes": list(self.missing_indexes),
            "previous_refreshed_at": self.previous_refreshed_at,
            "refreshed_at": self.refreshed_at,
        }


@dataclass(frozen=True)
class LineageRefreshReport:
    """Summary of one idempotent graph refresh, with its audit plan attached."""

    lake_uri: str
    artifacts: int
    executions: int
    edges: int
    plan: LineageRefreshPlan | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "artifacts": self.artifacts,
            "executions": self.executions,
            "edges": self.edges,
            "plan": self.plan.as_dict() if self.plan else None,
        }


@dataclass(frozen=True)
class EmittedLineage:
    """Graph rows emitted inline for one canonical write path (backlog 0098).

    Every SDK/CLI write path that records a ``transform_runs`` row emits its
    lineage slice -- one execution, its input/output artifacts, and the edges
    between them -- as part of the same operation, so callers do not have to
    remember a separate ``refresh_graph()`` step. ``produced_outputs`` is ``False``
    when the transform failed/aborted: the execution and its consumed inputs are
    still recorded, but no produced-output artifacts or edges are asserted.
    """

    lake_uri: str
    transform_id: str
    execution_id: str
    kind: str
    status: str | None
    artifact_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    produced_outputs: bool = True

    @property
    def artifact_count(self) -> int:
        return len(self.artifact_ids)

    @property
    def edge_count(self) -> int:
        return len(self.edge_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "transform_id": self.transform_id,
            "execution_id": self.execution_id,
            "kind": self.kind,
            "status": self.status,
            "produced_outputs": self.produced_outputs,
            "artifact_ids": list(self.artifact_ids),
            "edge_ids": list(self.edge_ids),
            "artifact_count": self.artifact_count,
            "edge_count": self.edge_count,
        }


@dataclass(frozen=True)
class LineageEmissionDivergence:
    """Divergence between the materialized graph and a fresh full projection.

    Inline emission (backlog 0098) and ``refresh_graph()`` share the same
    per-transform projection, so a full refresh always reconciles the graph. This
    report makes that auditable (AC#4):

    - ``missing_from_graph``: ids the current projection produces that are not yet
      materialized. After inline-only emission these are the entity-level rows a
      refresh backfills; after a full refresh this is empty.
    - ``changed``: ids present in both whose materialized content differs from the
      current projection. Present after inline-only emission for the few nodes two
      projectors write with different metadata (a refresh resolves them via
      last-writer-wins); empty after a full refresh.
    - ``extra_in_graph``: materialized ids the *current* projection does not
      reproduce. These are older version-pinned artifacts/edges retained as
      reproducibility history plus emit-time metadata a refresh will overwrite.
      They are informational -- a full refresh accumulates them too -- and do NOT
      count as an inconsistency.

    ``consistent`` is ``True`` when ``missing_from_graph`` and ``changed`` are both
    empty: everything the current projection needs is materialized and matches.
    """

    lake_uri: str
    missing_from_graph: dict[str, tuple[str, ...]]
    changed: dict[str, tuple[str, ...]]
    extra_in_graph: dict[str, tuple[str, ...]]

    @property
    def consistent(self) -> bool:
        return not (self.missing_from_graph or self.changed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "consistent": self.consistent,
            "missing_from_graph": {k: list(v) for k, v in self.missing_from_graph.items()},
            "changed": {k: list(v) for k, v in self.changed.items()},
            "extra_in_graph": {k: list(v) for k, v in self.extra_in_graph.items()},
        }


@dataclass(frozen=True)
class LineageRetentionHold:
    """Generic retention metadata attached to one or more lineage artifacts."""

    lake_uri: str
    artifact_ids: tuple[str, ...]
    retain_until: datetime | None = None
    legal_hold: bool = False
    audit_hold: bool = False
    promotion_hold: bool = False
    owner: str | None = None
    reason: str | None = None
    active: bool = False
    created_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "artifact_ids": list(self.artifact_ids),
            "retain_until": self.retain_until.isoformat() if self.retain_until else None,
            "legal_hold": self.legal_hold,
            "audit_hold": self.audit_hold,
            "promotion_hold": self.promotion_hold,
            "owner": self.owner,
            "reason": self.reason,
            "active": self.active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass(frozen=True)
class LineageAuditReport:
    """Lineage retention and resolvability audit for an artifact scope."""

    lake_uri: str
    subject: str | None
    root_artifact_ids: tuple[str, ...]
    artifact_count: int
    edge_count: int
    unresolved_references: tuple[dict[str, Any], ...] = ()
    missing_sources: tuple[dict[str, Any], ...] = ()
    missing_table_versions: tuple[dict[str, Any], ...] = ()
    stale_external_links: tuple[dict[str, Any], ...] = ()
    retained_versions: tuple[dict[str, Any], ...] = ()
    retention_holds: tuple[dict[str, Any], ...] = ()
    cleanup_candidates: tuple[dict[str, Any], ...] = ()
    refreshed: bool = False
    status: str = "passed"
    generated_at: datetime | None = None
    summary_counts: Mapping[str, int] = field(default_factory=dict)
    validator_statuses: tuple[dict[str, Any], ...] = ()
    report_id: str | None = None
    report_digest: str | None = None
    page_size: int | None = None
    total_findings: int | None = None
    returned_findings: int | None = None
    next_page_token: str | None = None
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        summary = {
            "unresolved_references": len(self.unresolved_references),
            "missing_sources": len(self.missing_sources),
            "missing_table_versions": len(self.missing_table_versions),
            "stale_external_links": len(self.stale_external_links),
            "retained_versions": len(self.retained_versions),
            "retention_holds": len(self.retention_holds),
            "cleanup_candidates": len(self.cleanup_candidates),
        }
        summary.update({str(key): int(value) for key, value in self.summary_counts.items()})
        payload = {
            "schema_version": "lancedb-robotics/lineage-audit/v1",
            "lake_uri": self.lake_uri,
            "subject": self.subject,
            "root_artifact_ids": list(self.root_artifact_ids),
            "status": self.status,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "artifact_count": self.artifact_count,
            "edge_count": self.edge_count,
            "summary": summary,
            "unresolved_references": list(self.unresolved_references),
            "missing_sources": list(self.missing_sources),
            "missing_table_versions": list(self.missing_table_versions),
            "stale_external_links": list(self.stale_external_links),
            "retained_versions": list(self.retained_versions),
            "retention_holds": list(self.retention_holds),
            "cleanup_candidates": list(self.cleanup_candidates),
            "validator_statuses": list(self.validator_statuses),
            "refreshed": self.refreshed,
        }
        if self.report_id is not None:
            payload["report_id"] = self.report_id
        if self.report_digest is not None:
            payload["report_digest"] = self.report_digest
        if self.page_size is not None:
            payload["page"] = {
                "page_size": self.page_size,
                "total_findings": self.total_findings,
                "returned_findings": self.returned_findings,
                "next_page_token": self.next_page_token,
                "truncated": self.truncated,
            }
        return payload


@dataclass(frozen=True)
class LineageInvalidation:
    """A durable invalidation marker attached to one or more lineage artifacts."""

    lake_uri: str
    invalidation_id: str
    invalidation_artifact_id: str
    target_artifact_ids: tuple[str, ...]
    reason: str
    severity: str
    discovered_by: str | None = None
    actor: str | None = None
    replacement_artifact_id: str | None = None
    created_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "invalidation_id": self.invalidation_id,
            "invalidation_artifact_id": self.invalidation_artifact_id,
            "target_artifact_ids": list(self.target_artifact_ids),
            "reason": self.reason,
            "severity": self.severity,
            "discovered_by": self.discovered_by,
            "actor": self.actor,
            "replacement_artifact_id": self.replacement_artifact_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass(frozen=True)
class RebuildPlanAction:
    """One ordered rebuild/invalidation action for an impacted artifact."""

    step: int
    action: str
    artifact_id: str
    kind: str
    name: str | None = None
    table_name: str | None = None
    table_version: int | None = None
    table_tag: str | None = None
    row_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    reason: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "name": self.name,
            "table_name": self.table_name,
            "table_version": self.table_version,
            "table_tag": self.table_tag,
            "row_ids": list(self.row_ids),
            "depends_on": list(self.depends_on),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ActionContext:
    """Inputs an action policy sees for one impacted artifact (backlog 0110).

    ``default_action`` is the built-in OSS classification (backlog 0066) for this
    artifact, so a policy that returns it -- or ``None`` -- leaves 0066 behaviour
    intact. ``incoming_edge_types`` are the plan-dependency edge types that reach
    this artifact within the impact graph, letting a policy key on *how* an
    artifact was affected (edge paths), not just what it is. ``severity`` is the
    plan's severity label.
    """

    artifact_id: str
    kind: str
    table_name: str | None
    default_action: str
    severity: str | None = None
    incoming_edge_types: frozenset[str] = frozenset()
    metadata: Mapping[str, str] = field(default_factory=dict)
    artifact: Mapping[str, Any] = field(default_factory=dict)


class ActionPolicy:
    """Maps an impacted artifact to a rebuild action (backlog 0110).

    The default policy returns the built-in 0066 classification unchanged. Custom
    policies subclass this or use :class:`MappingActionPolicy` /
    :class:`CallableActionPolicy`. A policy only *relabels* actions; it never
    changes which artifacts are impacted, so traversal results stay identical no
    matter which policy runs.
    """

    name: str = "default"

    def resolve(self, context: ActionContext) -> str | None:
        """Return the action for ``context`` (or ``None`` to keep the default)."""

        return context.default_action

    def action_for(self, context: ActionContext) -> str:
        """Resolve and validate the action, defaulting on ``None``."""

        action = self.resolve(context)
        if action is None:
            action = context.default_action
        action = str(action)
        if action not in KNOWN_REBUILD_ACTIONS:
            raise RebuildPlanError(
                f"action policy {getattr(self, 'name', 'custom')!r} emitted unknown "
                f"action {action!r} for artifact {context.artifact_id!r}; allowed "
                f"actions are {sorted(KNOWN_REBUILD_ACTIONS)}"
            )
        return action


DEFAULT_ACTION_POLICY = ActionPolicy()


@dataclass(frozen=True)
class MappingActionPolicy(ActionPolicy):
    """Declarative action policy keyed by severity, edge type, table, or kind.

    Precedence (first match wins):

    1. ``severity_actions[severity]`` -- a blanket action applied to every
       artifact at that plan severity (e.g. ``{"low": "notify-only"}`` to only
       notify on low-severity invalidations).
    2. ``edge_actions`` -- any incoming plan-edge type matches.
    3. ``table_actions[table_name]``.
    4. ``kind_actions[kind]``.
    5. ``fallback`` if set, else the built-in default action (0066 behaviour).

    Every configured action is validated against :data:`KNOWN_REBUILD_ACTIONS`
    at construction, so a typo fails fast rather than at plan time.
    """

    kind_actions: Mapping[str, str] = field(default_factory=dict)
    table_actions: Mapping[str, str] = field(default_factory=dict)
    edge_actions: Mapping[str, str] = field(default_factory=dict)
    severity_actions: Mapping[str, str] = field(default_factory=dict)
    fallback: str | None = None
    name: str = "mapping"

    def __post_init__(self) -> None:
        configured = [
            *self.kind_actions.values(),
            *self.table_actions.values(),
            *self.edge_actions.values(),
            *self.severity_actions.values(),
        ]
        if self.fallback is not None:
            configured.append(self.fallback)
        unknown = sorted({a for a in configured if a not in KNOWN_REBUILD_ACTIONS})
        if unknown:
            raise RebuildPlanError(
                f"action policy {self.name!r} configured with unknown action(s) "
                f"{unknown}; allowed actions are {sorted(KNOWN_REBUILD_ACTIONS)}"
            )

    def resolve(self, context: ActionContext) -> str | None:
        if context.severity is not None and context.severity in self.severity_actions:
            return self.severity_actions[context.severity]
        for edge_type in sorted(context.incoming_edge_types):
            if edge_type in self.edge_actions:
                return self.edge_actions[edge_type]
        if context.table_name and context.table_name in self.table_actions:
            return self.table_actions[context.table_name]
        if context.kind in self.kind_actions:
            return self.kind_actions[context.kind]
        if self.fallback is not None:
            return self.fallback
        return context.default_action


@dataclass(frozen=True)
class CallableActionPolicy(ActionPolicy):
    """Wrap a user callable ``(ActionContext) -> str | None`` as a policy.

    Returning ``None`` (or the default action) keeps the built-in classification.
    The returned action is validated like any other policy output.
    """

    func: Callable[[ActionContext], str | None]
    name: str = "callable"

    def resolve(self, context: ActionContext) -> str | None:
        return self.func(context)


def _coerce_action_policy(action_policy: Any) -> ActionPolicy:
    """Normalize an API argument into an :class:`ActionPolicy` instance."""

    if action_policy is None:
        return DEFAULT_ACTION_POLICY
    if isinstance(action_policy, ActionPolicy):
        return action_policy
    if callable(action_policy):
        return CallableActionPolicy(action_policy)
    raise RebuildPlanError(
        "action_policy must be an ActionPolicy, a callable (ActionContext -> action), "
        "or None"
    )


@dataclass(frozen=True)
class RebuildPlan:
    """A deterministic downstream-impact plan for invalidated lineage roots.

    Backlog 0110 adds bounded output on top of the 0066 plan: when ``page_size``
    is set, ``actions`` is a stable, non-overlapping page of the full ordered
    plan and ``next_page_token`` resumes it; ``summary_only`` returns the
    aggregates with no per-action rows. ``affected_artifact_count``,
    ``action_count``, ``actions_by_type``, and ``affected_by_kind`` are always
    computed over the *full* impact set so the summary stays stable across pages.
    """

    lake_uri: str
    root_artifact_ids: tuple[str, ...]
    actions: tuple[RebuildPlanAction, ...]
    reason: str | None = None
    severity: str | None = None
    invalidation: LineageInvalidation | None = None
    graph: LineageGraph | None = None
    policy_name: str | None = None
    affected_artifact_count: int | None = None
    action_count: int | None = None
    actions_by_type: Mapping[str, int] = field(default_factory=dict)
    affected_by_kind: Mapping[str, int] = field(default_factory=dict)
    page_size: int | None = None
    total_actions: int | None = None
    next_page_token: str | None = None
    truncated: bool = False
    summary_only: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "lancedb-robotics/rebuild-plan/v1",
            "lake_uri": self.lake_uri,
            "root_artifact_ids": list(self.root_artifact_ids),
            "reason": self.reason,
            "severity": self.severity,
            "invalidation": self.invalidation.as_dict() if self.invalidation else None,
            "actions": [action.as_dict() for action in self.actions],
            "graph": self.graph.as_dict() if self.graph else None,
        }
        # Backlog 0110 additive fields. Excluded from the 0109 catalog digest
        # (see rebuild_catalog._stored_plan), so recording is unaffected.
        payload["summary"] = {
            "policy": self.policy_name,
            "affected_artifact_count": self.affected_artifact_count,
            "action_count": self.action_count,
            "actions_by_type": dict(sorted(self.actions_by_type.items())),
            "affected_by_kind": dict(sorted(self.affected_by_kind.items())),
        }
        if self.page_size is not None or self.summary_only:
            payload["page"] = {
                "page_size": self.page_size,
                "total_actions": self.total_actions,
                "returned_actions": len(self.actions),
                "next_page_token": self.next_page_token,
                "truncated": self.truncated,
                "summary_only": self.summary_only,
            }
        return payload


@dataclass(frozen=True)
class SourceLogCoordinate:
    """A source-log coordinate recovered from an offending training row.

    ``offset`` is the existing lake provenance offset: the per-topic
    ``raw_sequence`` captured at ingest. ``log_time_ns`` is carried in the full
    report so MCAP callers can re-resolve and sanity-check the tuple.
    """

    uri: str
    channel: str
    offset: int | None
    log_time_ns: int | None
    observation_id: str
    run_id: str
    scenario_id: str

    def as_tuple(self) -> tuple[str, str, int | None]:
        """Return the compact ``(uri, channel, offset)`` source-log tuple."""

        return (self.uri, self.channel, self.offset)

    def as_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "channel": self.channel,
            "offset": self.offset,
            "log_time_ns": self.log_time_ns,
            "observation_id": self.observation_id,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
        }


@dataclass(frozen=True)
class RegressionTrace:
    """Resolved checkpoint -> snapshot -> offending rows -> source logs."""

    lake_uri: str
    model_run_id: str
    dataset_snapshot: dict[str, Any]
    table_versions: tuple[dict[str, Any], ...]
    model_outputs: tuple[dict[str, Any], ...]
    rows: tuple[dict[str, Any], ...]
    source_logs: tuple[SourceLogCoordinate, ...]
    transform_runs: tuple[dict[str, Any], ...]
    where: str | None = None
    training_run: dict[str, Any] | None = None
    model_artifacts: tuple[dict[str, Any], ...] = ()
    _lake: Lake | None = field(default=None, repr=False, compare=False)

    def to_source_logs(self) -> tuple[tuple[str, str, int | None], ...]:
        """Emit re-derivable ``(uri, channel, offset)`` tuples for the slice."""

        return tuple(coord.as_tuple() for coord in self.source_logs)

    def evidence_pack(
        self,
        *,
        output_dir: str | None = None,
        materialize: bool = False,
        include_payloads: bool = False,
        include_attachments: bool = False,
        include_video: bool = False,
        redaction_policy: Any | None = None,
    ):
        """Build a deterministic evidence pack for this resolved trace."""

        if self._lake is None:
            raise LineageError(
                "this RegressionTrace is detached from a lake; call "
                "lake.lineage.trace_checkpoint(...).evidence_pack() or "
                "lake.lineage.evidence_pack(...)"
            )
        from lancedb_robotics.evidence import evidence_pack_from_trace

        return evidence_pack_from_trace(
            self._lake,
            self,
            output_dir=output_dir,
            materialize=materialize,
            include_payloads=include_payloads,
            include_attachments=include_attachments,
            include_video=include_video,
            redaction_policy=redaction_policy,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report payload."""

        return {
            "lake_uri": self.lake_uri,
            "model_run_id": self.model_run_id,
            "where": self.where,
            "dataset_snapshot": _snapshot_report(self.dataset_snapshot),
            "table_versions": list(self.table_versions),
            "model_outputs": list(self.model_outputs),
            "training_run": self.training_run,
            "model_artifacts": list(self.model_artifacts),
            "rows": list(self.rows),
            "source_logs": [coord.as_dict() for coord in self.source_logs],
            "source_log_tuples": list(self.to_source_logs()),
            "transform_runs": list(self.transform_runs),
        }


@dataclass(frozen=True)
class LineageResolutionCandidate:
    """One artifact a handle resolves to, with the fields it matched on (backlog 0102).

    ``in_graph`` is ``True`` when the candidate is materialized in
    ``lineage_artifacts`` today. ``in_graph=False`` marks a candidate that is known
    in a canonical source table but not yet projected into the graph, i.e. one a
    ``refresh_graph`` would add.
    """

    artifact_id: str
    kind: str | None = None
    name: str | None = None
    table_name: str | None = None
    table_version: int | None = None
    table_tag: str | None = None
    row_grain: str | None = None
    source_uri: str | None = None
    source_id: str | None = None
    digest: str | None = None
    matched_on: tuple[str, ...] = ()
    in_graph: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "name": self.name,
            "table_name": self.table_name,
            "table_version": self.table_version,
            "table_tag": self.table_tag,
            "row_grain": self.row_grain,
            "source_uri": self.source_uri,
            "source_id": self.source_id,
            "digest": self.digest,
            "matched_on": list(self.matched_on),
            "in_graph": self.in_graph,
            "evidence": dict(sorted(self.evidence.items())),
        }


@dataclass(frozen=True)
class LineageResolution:
    """Structured diagnostic for resolving a human handle to lineage roots (backlog 0102).

    ``status`` is one of:

    - ``resolved``: the handle maps to a single materialized artifact, or to several
      coordinate roots of one logical handle (e.g. a source URI fanning out to many
      ``source`` coordinates -- ``multi_root`` is then ``True`` and ``artifact_ids``
      carries every root, matching the 0063 decision that source-URI impact merges
      across all matching roots).
    - ``ambiguous``: the handle maps to several distinct materialized entities the
      caller must choose between; ``disambiguation_hints`` and ``suggested_commands``
      say how.
    - ``stale``: the handle is known in a canonical source table but is not yet in
      the lineage graph (or its source table changed since the last refresh), so a
      ``refresh_graph`` is required before trace/impact can see it.
    - ``unknown``: the handle matches nothing in the graph or any canonical table.
    - ``unsupported-kind``: an explicit ``kind`` hint is not a recognized handle kind.

    Resolution is read-only: it never refreshes or records graph rows. ``graph_fresh``
    is ``True`` only when a refresh watermark exists and no source table changed since
    it; ``fresh_through_versions`` reports the per-table versions the graph is fresh
    through and ``stale_tables`` names tables that changed since.
    """

    lake_uri: str
    handle: str
    status: str
    requested_kind: str | None
    candidates: tuple[LineageResolutionCandidate, ...]
    artifact_ids: tuple[str, ...]
    root_count: int
    multi_root: bool
    graph_fresh: bool
    refreshed_at: str | None
    fresh_through_versions: dict[str, int]
    stale_tables: tuple[str, ...]
    pending_refresh_artifact_ids: tuple[str, ...]
    disambiguation_hints: tuple[dict[str, Any], ...]
    suggested_commands: tuple[str, ...]
    message: str | None = None

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "lancedb-robotics/lineage-resolution/v1",
            "lake_uri": self.lake_uri,
            "handle": self.handle,
            "requested_kind": self.requested_kind,
            "status": self.status,
            "root_count": self.root_count,
            "multi_root": self.multi_root,
            "artifact_ids": list(self.artifact_ids),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "graph_fresh": self.graph_fresh,
            "refreshed_at": self.refreshed_at,
            "fresh_through_versions": dict(sorted(self.fresh_through_versions.items())),
            "stale_tables": list(self.stale_tables),
            "pending_refresh_artifact_ids": list(self.pending_refresh_artifact_ids),
            "disambiguation_hints": list(self.disambiguation_hints),
            "suggested_commands": list(self.suggested_commands),
            "message": self.message,
        }


class LakeLineage:
    """Lineage graph recording, traversal, and checkpoint trace queries."""

    def __init__(self, lake: Lake) -> None:
        self._lake = lake

    def artifact_id(self, kind: str, **identity: Any) -> str:
        """Return the stable LanceDB Robotics artifact id for ``identity``."""

        return artifact_id(kind, **identity)

    def resolve_artifacts(
        self,
        handle: str,
        *,
        kind: str | None = None,
        table_version: int | None = None,
    ) -> tuple[str, ...]:
        """Resolve a human artifact handle to one or more graph artifact ids.

        Handles may be exact lineage artifact ids or domain identities such as
        snapshot tags, run/scenario/observation ids, source URIs, checksums,
        transform ids, training run ids, checkpoint/model artifact ids, or
        projection manifest paths. Source URIs can map to many source-coordinate
        artifacts, so callers should be prepared for multiple roots.
        """

        return _resolve_artifact_ids(
            self._lake,
            handle,
            kind=kind,
            table_version=table_version,
        )

    def resolve(
        self,
        handle: str,
        *,
        kind: str | None = None,
        table_version: int | None = None,
    ) -> LineageResolution:
        """Resolve a handle to lineage roots with structured diagnostics (backlog 0102).

        Unlike :meth:`resolve_artifacts` (which returns ids or raises), this returns a
        :class:`LineageResolution` describing the resolution status
        (resolved/ambiguous/stale/unknown/unsupported-kind), candidate evidence, graph
        freshness against the refresh watermark, and ready-to-run disambiguation or
        refresh commands. Resolution rides bounded/indexed reads (point lookups on the
        ``artifact_id`` index plus predicate-pushed equality queries) and is read-only:
        it never refreshes or records graph rows.
        """

        return _resolve_diagnostics(
            self._lake, handle, kind=kind, table_version=table_version
        )

    def export_openlineage(
        self,
        *,
        refresh: bool = True,
        dry_run: bool = True,
        producer: str | None = None,
    ):
        """Project canonical lineage graph executions to OpenLineage events."""

        from lancedb_robotics.lineage_integrations import (
            DEFAULT_PRODUCER,
            export_openlineage,
        )

        return export_openlineage(
            self._lake,
            refresh=refresh,
            dry_run=dry_run,
            producer=producer or DEFAULT_PRODUCER,
        )

    def export_datahub(
        self,
        *,
        refresh: bool = True,
        dry_run: bool = True,
    ):
        """Project canonical lineage graph edges to DataHub-style lineage."""

        from lancedb_robotics.lineage_integrations import export_datahub

        return export_datahub(self._lake, refresh=refresh, dry_run=dry_run)

    def emit_openlineage(
        self,
        *,
        refresh: bool = True,
        producer: str | None = None,
        target: str | None = None,
        endpoint_url: str | None = None,
        auth_ref: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: Any | None = None,
        adapter: str = "openlineage",
        retry: bool = False,
        created_by: str | None = None,
    ):
        """Deliver OpenLineage events and record idempotent delivery attempts."""

        from lancedb_robotics.lineage_integrations import (
            DEFAULT_PRODUCER,
            emit_openlineage,
        )

        return emit_openlineage(
            self._lake,
            refresh=refresh,
            producer=producer or DEFAULT_PRODUCER,
            target=target,
            endpoint_url=endpoint_url,
            auth_ref=auth_ref,
            headers=headers,
            client=client,
            adapter=adapter,
            retry=retry,
            created_by=created_by,
        )

    def emit_datahub(
        self,
        *,
        refresh: bool = True,
        target: str | None = None,
        endpoint_url: str | None = None,
        auth_ref: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: Any | None = None,
        adapter: str = "datahub",
        retry: bool = False,
        created_by: str | None = None,
    ):
        """Deliver DataHub-style lineage edges and record delivery attempts."""

        from lancedb_robotics.lineage_integrations import emit_datahub

        return emit_datahub(
            self._lake,
            refresh=refresh,
            target=target,
            endpoint_url=endpoint_url,
            auth_ref=auth_ref,
            headers=headers,
            client=client,
            adapter=adapter,
            retry=retry,
            created_by=created_by,
        )

    def retry_lineage_delivery(
        self,
        backend: str,
        *,
        refresh: bool = True,
        target: str | None = None,
        endpoint_url: str | None = None,
        auth_ref: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: Any | None = None,
        adapter: str | None = None,
        producer: str | None = None,
        created_by: str | None = None,
    ):
        """Retry OpenLineage or DataHub delivery, skipping delivered digests."""

        from lancedb_robotics.lineage_integrations import (
            DEFAULT_PRODUCER,
            retry_lineage_delivery,
        )

        return retry_lineage_delivery(
            self._lake,
            backend,
            refresh=refresh,
            target=target,
            endpoint_url=endpoint_url,
            auth_ref=auth_ref,
            headers=headers,
            client=client,
            adapter=adapter,
            producer=producer or DEFAULT_PRODUCER,
            created_by=created_by,
        )

    def lineage_delivery_attempts(
        self,
        *,
        backend: str | None = None,
        target: str | None = None,
        status: str | None = None,
    ):
        """Read persisted external lineage delivery attempts."""

        from lancedb_robotics.lineage_integrations import lineage_delivery_attempts

        return lineage_delivery_attempts(
            self._lake,
            backend=backend,
            target=target,
            status=status,
        )

    def external_urn(self, artifact_id: str, *, backend: str = "openlineage") -> str:
        """Return a stable external URN for a canonical lineage artifact."""

        from lancedb_robotics.lineage_integrations import external_artifact_urn

        return external_artifact_urn(artifact_id, backend=backend)

    def resolve_external_urn(self, urn: str) -> str:
        """Resolve an exported external URN back to a canonical artifact id."""

        from lancedb_robotics.lineage_integrations import resolve_external_urn

        return resolve_external_urn(self._lake, urn)

    def require_integration_adapter(self, adapter: str) -> dict[str, str]:
        """Validate that an optional external-system adapter package is installed."""

        from lancedb_robotics.lineage_integrations import require_integration_adapter

        return require_integration_adapter(adapter)

    def context(self, lineage_context: Any | None, *, inherit: bool = True):
        """Scope external lineage context across nested SDK operations."""

        from lancedb_robotics.lineage_hooks import lineage_context_scope

        return lineage_context_scope(lineage_context, inherit=inherit)

    def current_context(self):
        """Return the external lineage context active in this execution scope."""

        from lancedb_robotics.lineage_hooks import current_lineage_context

        return current_lineage_context()

    def worker_context(
        self,
        lineage_context: Any | None = None,
        *,
        include_keys: Sequence[str] | None = None,
        redact_keys: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return an explicit, redacted lineage context payload for workers."""

        from lancedb_robotics.lineage_hooks import lineage_context_for_worker

        return lineage_context_for_worker(
            lineage_context,
            include_keys=include_keys,
            redact_keys=redact_keys,
        )

    def worker_env(
        self,
        lineage_context: Any | None = None,
        *,
        include_keys: Sequence[str] | None = None,
        redact_keys: Sequence[str] | None = None,
    ) -> dict[str, str]:
        """Return env vars that explicitly propagate lineage context to workers."""

        from lancedb_robotics.lineage_hooks import lineage_context_env_for_worker

        return lineage_context_env_for_worker(
            lineage_context,
            include_keys=include_keys,
            redact_keys=redact_keys,
        )

    def hook_adapters(self):
        """Return known dependency-light lineage hook adapter specs."""

        from lancedb_robotics.lineage_hooks import lineage_hook_adapter_specs

        return lineage_hook_adapter_specs()

    def require_hook_adapter(self, adapter: str) -> dict[str, str]:
        """Validate that an optional lineage hook adapter package is installed."""

        from lancedb_robotics.lineage_hooks import require_lineage_hook_adapter

        return require_lineage_hook_adapter(adapter)

    def check_hook_conformance(
        self,
        hook: Any,
        *,
        operation: str = "lineage-hook-conformance",
        params: Mapping[str, Any] | None = None,
    ):
        """Run the dependency-free hook plugin conformance harness."""

        from lancedb_robotics.lineage_hooks import check_lineage_hook_conformance

        return check_lineage_hook_conformance(
            hook,
            operation=operation,
            params=params,
        )

    @property
    def plugins(self):
        """Metadata-integration plugin registry + conformance suite (backlog 0106).

        Exposes the stable adapter contract: list/probe registered emitters,
        reference importers, and manifest-sync providers, register in-house
        adapters, and run the conformance suite. The registry never imports an
        optional dependency at base import time -- availability is a metadata
        probe, not an import.
        """

        from lancedb_robotics.metadata_plugins import LakeMetadataPlugins

        return LakeMetadataPlugins(self._lake)

    def export_artifact_urns(
        self,
        *,
        backend: str = "openlineage",
        page_size: int | None = None,
        page_token: str | None = None,
        artifact_kind: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        table_versions: Any | None = None,
        refresh: bool = True,
    ):
        """Return a bounded page of the bulk artifact-URN catalog (backlog 0105).

        Pass ``page_size`` for a bounded page plus a ``next_page_token``
        continuation handle; resume with the same filters. Refresh is skipped
        automatically when ``page_token`` is set so continuation reads the same
        snapshot the first page saw.
        """

        from lancedb_robotics.lineage_integrations import (
            LineageExportFilters,
            export_artifact_urn_catalog,
        )

        filters = LineageExportFilters.build(
            artifact_kind=artifact_kind,
            created_after=created_after,
            created_before=created_before,
            table_versions=table_versions,
        )
        return export_artifact_urn_catalog(
            self._lake,
            backend=backend,
            page_size=page_size,
            page_token=page_token,
            filters=filters,
            refresh=refresh,
        )

    def export_openlineage_page(
        self,
        *,
        page_size: int | None = None,
        page_token: str | None = None,
        execution_kind: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        refresh: bool = True,
        dry_run: bool = True,
        producer: str | None = None,
    ):
        """Return a bounded page of OpenLineage RunEvents (backlog 0105)."""

        from lancedb_robotics.lineage_integrations import (
            DEFAULT_PRODUCER,
            LineageExportFilters,
            export_openlineage_page,
        )

        filters = LineageExportFilters.build(
            execution_kind=execution_kind,
            created_after=created_after,
            created_before=created_before,
        )
        return export_openlineage_page(
            self._lake,
            page_size=page_size,
            page_token=page_token,
            filters=filters,
            refresh=refresh,
            dry_run=dry_run,
            producer=producer or DEFAULT_PRODUCER,
        )

    def export_datahub_page(
        self,
        *,
        page_size: int | None = None,
        page_token: str | None = None,
        artifact_kind: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        table_versions: Any | None = None,
        refresh: bool = True,
        dry_run: bool = True,
    ):
        """Return a bounded page of DataHub-style lineage edges (backlog 0105)."""

        from lancedb_robotics.lineage_integrations import (
            LineageExportFilters,
            export_datahub_page,
        )

        filters = LineageExportFilters.build(
            artifact_kind=artifact_kind,
            created_after=created_after,
            created_before=created_before,
            table_versions=table_versions,
        )
        return export_datahub_page(
            self._lake,
            page_size=page_size,
            page_token=page_token,
            filters=filters,
            refresh=refresh,
            dry_run=dry_run,
        )

    def iter_openlineage_ndjson(
        self,
        *,
        page_size: int = 512,
        execution_kind: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        refresh: bool = True,
        dry_run: bool = True,
        producer: str | None = None,
        include_summary: bool = False,
    ):
        """Stream OpenLineage RunEvents one record at a time (backlog 0105)."""

        from lancedb_robotics.lineage_integrations import (
            DEFAULT_PRODUCER,
            LineageExportFilters,
            iter_openlineage_ndjson,
        )

        filters = LineageExportFilters.build(
            execution_kind=execution_kind,
            created_after=created_after,
            created_before=created_before,
        )
        return iter_openlineage_ndjson(
            self._lake,
            page_size=page_size,
            filters=filters,
            refresh=refresh,
            dry_run=dry_run,
            producer=producer or DEFAULT_PRODUCER,
            include_summary=include_summary,
        )

    def iter_datahub_ndjson(
        self,
        *,
        page_size: int = 512,
        artifact_kind: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        table_versions: Any | None = None,
        refresh: bool = True,
        dry_run: bool = True,
        include_summary: bool = False,
    ):
        """Stream DataHub-style edges one record at a time (backlog 0105)."""

        from lancedb_robotics.lineage_integrations import (
            LineageExportFilters,
            iter_datahub_ndjson,
        )

        filters = LineageExportFilters.build(
            artifact_kind=artifact_kind,
            created_after=created_after,
            created_before=created_before,
            table_versions=table_versions,
        )
        return iter_datahub_ndjson(
            self._lake,
            page_size=page_size,
            filters=filters,
            refresh=refresh,
            dry_run=dry_run,
            include_summary=include_summary,
        )

    def iter_artifact_urn_ndjson(
        self,
        *,
        backend: str = "openlineage",
        page_size: int = 512,
        artifact_kind: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        table_versions: Any | None = None,
        refresh: bool = True,
        include_summary: bool = False,
    ):
        """Stream bulk URN-catalog records one at a time (backlog 0105)."""

        from lancedb_robotics.lineage_integrations import (
            LineageExportFilters,
            iter_artifact_urn_ndjson,
        )

        filters = LineageExportFilters.build(
            artifact_kind=artifact_kind,
            created_after=created_after,
            created_before=created_before,
            table_versions=table_versions,
        )
        return iter_artifact_urn_ndjson(
            self._lake,
            backend=backend,
            page_size=page_size,
            filters=filters,
            refresh=refresh,
            include_summary=include_summary,
        )

    def record_artifact(
        self,
        *,
        kind: str,
        artifact_id: str | None = None,
        name: str | None = None,
        table_name: str | None = None,
        table_version: int | None = None,
        table_tag: str | None = None,
        row_grain: str | None = None,
        row_ids: Iterable[str] = (),
        source_uri: str | None = None,
        source_id: str | None = None,
        digest: str | None = None,
        producer_execution_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LineageArtifact:
        """Upsert one lineage artifact row.

        ``artifact_id`` may be supplied by integrations that already use the
        LanceDB Robotics URN. When omitted, it is derived from stable identity
        fields, excluding lake-local paths unless they are the only identity.
        """

        normalized_rows = tuple(sorted(str(row_id) for row_id in row_ids if row_id is not None))
        artifact = LineageArtifact(
            artifact_id=artifact_id
            or _artifact_id_for_fields(
                kind,
                table_name=table_name,
                table_version=table_version,
                table_tag=table_tag,
                row_grain=row_grain,
                row_ids=normalized_rows,
                source_uri=source_uri,
                source_id=source_id,
                digest=digest,
                name=name,
            ),
            kind=kind,
            name=name,
            table_name=table_name,
            table_version=table_version,
            table_tag=table_tag,
            row_grain=row_grain,
            row_ids=normalized_rows,
            source_uri=source_uri,
            source_id=source_id,
            digest=digest,
            producer_execution_id=producer_execution_id,
            metadata=_string_metadata(metadata),
        )
        _replace_rows(
            self._lake,
            "lineage_artifacts",
            "artifact_id",
            [artifact.as_row(datetime.now(UTC))],
            LINEAGE_ARTIFACTS_SCHEMA,
        )
        return artifact

    def record_execution(
        self,
        *,
        kind: str,
        execution_id: str | None = None,
        name: str | None = None,
        transform_id: str | None = None,
        status: str | None = None,
        params: dict[str, Any] | None = None,
        params_json: str | None = None,
        code_ref: str | None = None,
        provider: str | None = None,
        environment: dict[str, Any] | None = None,
        environment_json: str | None = None,
        input_artifact_ids: Iterable[str] = (),
        output_artifact_ids: Iterable[str] = (),
        input_table_versions: Iterable[dict[str, Any]] = (),
        output_table_versions: Iterable[dict[str, Any]] = (),
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        created_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LineageExecution:
        """Upsert one execution row with input/output artifact pins."""

        normalized_inputs = tuple(sorted(str(value) for value in input_artifact_ids if value))
        normalized_outputs = tuple(sorted(str(value) for value in output_artifact_ids if value))
        execution = LineageExecution(
            execution_id=execution_id
            or _execution_id_for_fields(kind, transform_id=transform_id, name=name, params=params),
            kind=kind,
            name=name,
            transform_id=transform_id,
            status=status,
            params=params,
            params_json=params_json,
            code_ref=code_ref,
            provider=provider,
            environment=environment,
            environment_json=environment_json,
            input_artifact_ids=normalized_inputs,
            output_artifact_ids=normalized_outputs,
            input_table_versions=tuple(_version_row(row) for row in input_table_versions),
            output_table_versions=tuple(_version_row(row) for row in output_table_versions),
            started_at=started_at,
            finished_at=finished_at,
            created_by=created_by,
            metadata=_string_metadata(metadata),
        )
        _replace_rows(
            self._lake,
            "lineage_executions",
            "execution_id",
            [execution.as_row(datetime.now(UTC))],
            LINEAGE_EXECUTIONS_SCHEMA,
        )
        return execution

    def record_edge(
        self,
        *,
        edge_type: str,
        from_artifact_id: str,
        to_artifact_id: str,
        execution_id: str | None = None,
        edge_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LineageEdge:
        """Upsert one directed upstream -> downstream artifact edge."""

        edge = LineageEdge(
            edge_id=edge_id
            or _edge_id(edge_type, from_artifact_id, to_artifact_id, execution_id),
            edge_type=edge_type,
            from_artifact_id=from_artifact_id,
            to_artifact_id=to_artifact_id,
            execution_id=execution_id,
            metadata=_string_metadata(metadata),
        )
        _replace_rows(
            self._lake,
            "lineage_edges",
            "edge_id",
            [edge.as_row(datetime.now(UTC))],
            LINEAGE_EDGES_SCHEMA,
        )
        return edge

    def retain(
        self,
        artifact: str,
        *,
        kind: str | None = None,
        retain_until: datetime | str | None = None,
        legal_hold: bool = False,
        audit_hold: bool = False,
        promotion_hold: bool = False,
        owner: str | None = None,
        reason: str | None = None,
        refresh: bool = True,
    ) -> LineageRetentionHold:
        """Attach generic retention/audit-hold metadata to artifact handle(s).

        Holds are intentionally policy-light OSS metadata. A future
        ``retain_until`` pins until that timestamp; legal/audit/promotion holds
        are indefinite until cleared or overwritten.
        """

        if refresh:
            self.refresh_graph()
        artifact_ids = _resolve_artifact_ids(self._lake, artifact, kind=kind)
        now = datetime.now(UTC)
        retain_until_dt = _normalize_datetime(retain_until, "retain_until")
        hold = LineageRetentionHold(
            lake_uri=self._lake.uri,
            artifact_ids=tuple(artifact_ids),
            retain_until=_utc_datetime(retain_until_dt),
            legal_hold=bool(legal_hold),
            audit_hold=bool(audit_hold),
            promotion_hold=bool(promotion_hold),
            owner=owner,
            reason=reason,
            active=_retention_active(
                retain_until_dt,
                legal_hold=legal_hold,
                audit_hold=audit_hold,
                promotion_hold=promotion_hold,
                now=now,
            ),
            created_at=now,
        )
        _update_artifact_retention_metadata(self._lake, hold)
        return hold

    def clear_retention(
        self,
        artifact: str,
        *,
        kind: str | None = None,
        refresh: bool = True,
    ) -> LineageRetentionHold:
        """Remove retention metadata from artifact handle(s)."""

        if refresh:
            self.refresh_graph()
        artifact_ids = _resolve_artifact_ids(self._lake, artifact, kind=kind)
        _clear_artifact_retention_metadata(self._lake, artifact_ids)
        return LineageRetentionHold(
            lake_uri=self._lake.uri,
            artifact_ids=tuple(artifact_ids),
            active=False,
            created_at=datetime.now(UTC),
        )

    def audit(
        self,
        artifact: str | None = None,
        *,
        kind: str | None = None,
        refresh: bool = True,
        check_sources: bool = True,
        check_remote_sources: bool = False,
        source_auth_ref: str | None = None,
        source_storage_options: Mapping[str, Any] | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
        now: datetime | None = None,
    ) -> LineageAuditReport:
        """Audit lineage references, retention holds, and cleanup eligibility."""

        if refresh:
            self.refresh_graph()
        root_ids = _resolve_artifact_ids(self._lake, artifact, kind=kind) if artifact else ()
        return _audit_lineage(
            self._lake,
            subject=artifact,
            root_artifact_ids=root_ids,
            check_sources=check_sources,
            check_remote_sources=check_remote_sources,
            source_auth_ref=source_auth_ref,
            source_storage_options=source_storage_options,
            page_size=page_size,
            page_token=page_token,
            now=now,
            refreshed=refresh,
        )

    def record_audit_report(self, report, **kwargs):
        """Record a lineage audit report in the durable catalog (0112)."""

        from lancedb_robotics.lineage_audit_catalog import record_audit_report

        return record_audit_report(self._lake, report, **kwargs)

    def get_audit_report(self, report_id_or_digest: str):
        """Reload a persisted lineage audit report by id or digest (0112)."""

        from lancedb_robotics.lineage_audit_catalog import get_audit_report

        return get_audit_report(self._lake, report_id_or_digest)

    def audit_reports(self, **kwargs):
        """List persisted lineage audit reports newest-first (0112)."""

        from lancedb_robotics.lineage_audit_catalog import audit_reports

        return audit_reports(self._lake, **kwargs)

    def audit_findings(self, report_id_or_digest: str, **kwargs):
        """Return a bounded page of findings from a persisted audit report (0112)."""

        from lancedb_robotics.lineage_audit_catalog import audit_findings

        return audit_findings(self._lake, report_id_or_digest, **kwargs)

    def iter_audit_findings_ndjson(self, report_id_or_digest: str, **kwargs):
        """Yield persisted audit findings as NDJSON records (0112)."""

        from lancedb_robotics.lineage_audit_catalog import iter_audit_findings_ndjson

        return iter_audit_findings_ndjson(self._lake, report_id_or_digest, **kwargs)

    def refresh_graph(
        self,
        *,
        incremental: bool = True,
        force_full: bool = False,
        dry_run: bool = False,
        reconcile: bool = True,
    ) -> LineageRefreshReport:
        """Project existing canonical tables into the lineage graph tables.

        The operation is idempotent: each artifact/execution/edge id is stable,
        and existing graph rows with those ids are replaced rather than
        duplicated.

        With ``incremental=True`` (the default) the refresh compares each source
        table's current version against the watermark recorded at the last
        successful refresh. If nothing changed the projection is skipped entirely
        (backlog 0097): trace/impact/audit that call ``refresh_graph`` first stay
        cheap on an unchanged lake. When a source table did change, a full
        re-projection runs and the returned :class:`LineageRefreshPlan` records
        which tables changed and why a full scan was required. ``force_full``
        forces a full re-projection; ``dry_run`` computes the plan without writing
        (see :meth:`plan_refresh`); ``reconcile`` retires graph rows whose source
        canonical rows were deleted/superseded (respecting retention holds).
        """

        plan_only = _plan_refresh(
            self._lake, incremental=incremental, force_full=force_full
        )
        if dry_run:
            return LineageRefreshReport(
                lake_uri=self._lake.uri,
                artifacts=plan_only.artifacts,
                executions=plan_only.executions,
                edges=plan_only.edges,
                plan=plan_only,
            )
        if not plan_only.full_scan:
            # Nothing changed since the watermark: touch zero graph rows.
            plan = replace(plan_only, dry_run=False)
            return LineageRefreshReport(
                lake_uri=self._lake.uri,
                artifacts=plan.artifacts,
                executions=plan.executions,
                edges=plan.edges,
                plan=plan,
            )

        now = datetime.now(UTC)
        # Capture active retention holds before the projection: a projection can
        # resurrect a deleted entity's artifact and drop its hold metadata, so the
        # pre-projection snapshot is what reconciliation restores from.
        pre_holds = _active_hold_ids(self._lake) if reconcile else {}
        counts = self._project_full_graph(now)
        reconciliation = (
            _reconcile_stale_graph_rows(self._lake, dry_run=False, pre_holds=pre_holds)
            if reconcile
            else _StaleReconciliation()
        )
        # Build/refresh the lineage predicate indexes the trace/impact frontier
        # expansion rides (backlog 0181 / BUG-15): BTREE on the edge endpoints +
        # artifact resolution keys. The graph is now materialized, so this is the
        # natural point to index it; coverage of merge_insert-touched fragments is
        # folded back in via optimize_indices.
        _refresh_lineage_predicate_indexes(self._lake)
        _write_refresh_watermark(self._lake, plan_only.source_tables, counts, now)
        # ``counts`` are the rows this refresh projected; retired stale rows are
        # reported separately on the plan (they were never part of the projection).
        artifacts_count, executions_count, edges_count = counts
        plan = replace(
            plan_only,
            artifacts=artifacts_count,
            executions=executions_count,
            edges=edges_count,
            retired_artifacts=reconciliation.retired_artifacts,
            retired_edges=reconciliation.retired_edges,
            stale_artifacts=reconciliation.stale_artifacts,
            held_stale_artifacts=reconciliation.held_stale_artifacts,
            missing_indexes=_lineage_index_plan(self._lake)["missing"],
            refreshed_at=now.isoformat(),
            dry_run=False,
        )
        return LineageRefreshReport(
            lake_uri=self._lake.uri,
            artifacts=artifacts_count,
            executions=executions_count,
            edges=edges_count,
            plan=plan,
        )

    def plan_refresh(
        self, *, incremental: bool = True, force_full: bool = False
    ) -> LineageRefreshPlan:
        """Return the refresh plan without mutating the graph (backlog 0097).

        The plan records source table versions against the last-refresh
        watermark, whether a full re-projection is required and why, current
        graph-row counts, and any missing lineage predicate indexes.
        """

        return _plan_refresh(self._lake, incremental=incremental, force_full=force_full)

    def index_plan(self) -> dict[str, Any]:
        """Return the required/present/missing lineage predicate index plan.

        Traversal stays correct without the indexes (predicate pushdown), so this
        is an actionable diagnostic (backlog 0097 / 0181): every ``missing`` entry
        names the table, column, index type, and the build call that closes the
        gap.
        """

        return _lineage_index_plan(self._lake)

    def emit_transform(
        self,
        transform: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> EmittedLineage:
        """Emit the lineage graph slice for one just-written ``transform_runs`` row.

        Backlog 0098: canonical SDK write paths call this immediately after
        appending their ``transform_runs`` row so lineage is recorded as part of
        the same operation, rather than only when someone later calls
        :meth:`refresh_graph`. The projection reuses the same per-transform logic
        as the full refresh (:func:`_project_transform_row`), so the emitted rows
        are byte-identical to what a subsequent refresh would produce -- refresh
        stays a pure reconciliation/backfill tool.

        Emission is idempotent: artifact/execution/edge ids are stable and rows are
        upserted, so retrying a deterministic operation never duplicates graph
        rows. A failed/aborted transform records its execution and consumed inputs
        but no produced-output artifacts or edges.

        This method may raise; :func:`emit_transform_lineage` is the best-effort
        wrapper write paths should use so a projection error never fails the write.
        """

        now = now or datetime.now(UTC)
        transform = dict(transform)
        if not transform.get("transform_id"):
            raise LineageError("transform row is missing a transform_id")

        table_versions = _current_table_versions(self._lake)
        # Only ``dataset_snapshots`` is read from ``rows_by_table`` by the
        # per-transform projector; keep the read scoped rather than loading every
        # canonical table on every write.
        rows_by_table = {
            "dataset_snapshots": self._lake.table("dataset_snapshots").to_arrow().to_pylist()
        }

        artifacts: dict[str, LineageArtifact] = {}
        executions: dict[str, LineageExecution] = {}
        edges: dict[str, LineageEdge] = {}
        add_artifact, add_execution, add_edge = _graph_accumulators(
            artifacts, executions, edges
        )

        execution_id = _project_transform_row(
            self._lake,
            transform,
            table_versions,
            rows_by_table,
            add_artifact,
            add_execution,
            add_edge,
        )

        if artifacts:
            _replace_rows(
                self._lake,
                "lineage_artifacts",
                "artifact_id",
                [artifact.as_row(now) for artifact in artifacts.values()],
                LINEAGE_ARTIFACTS_SCHEMA,
            )
        if executions:
            _replace_rows(
                self._lake,
                "lineage_executions",
                "execution_id",
                [execution.as_row(now) for execution in executions.values()],
                LINEAGE_EXECUTIONS_SCHEMA,
            )
        if edges:
            _replace_rows(
                self._lake,
                "lineage_edges",
                "edge_id",
                [edge.as_row(now) for edge in edges.values()],
                LINEAGE_EDGES_SCHEMA,
            )

        return EmittedLineage(
            lake_uri=self._lake.uri,
            transform_id=str(transform["transform_id"]),
            execution_id=execution_id,
            kind=str(transform.get("kind") or "transform"),
            status=transform.get("status"),
            artifact_ids=tuple(sorted(artifacts)),
            edge_ids=tuple(sorted(edges)),
            produced_outputs=_transform_produced_outputs(transform),
        )

    def emission_divergence(self) -> LineageEmissionDivergence:
        """Compare the materialized graph against a fresh full projection (0098 AC#4).

        Because inline emission and :meth:`refresh_graph` share the same
        projection, emitted rows can never contradict the projection. This report
        makes that auditable: after inline-only emission ``missing_from_graph``
        lists the entity-level rows a refresh backfills, while
        ``unexpected_in_graph`` and ``changed`` stay empty; after a full refresh
        all three are empty (``consistent``).
        """

        projected_artifacts, projected_executions, projected_edges = _compute_full_graph(
            self._lake
        )
        now = datetime.now(UTC)
        projected = {
            "lineage_artifacts": {
                artifact_id: artifact.as_row(now)
                for artifact_id, artifact in projected_artifacts.items()
            },
            "lineage_executions": {
                execution_id: execution.as_row(now)
                for execution_id, execution in projected_executions.items()
            },
            "lineage_edges": {
                edge_id: edge.as_row(now) for edge_id, edge in projected_edges.items()
            },
        }
        id_columns = {
            "lineage_artifacts": "artifact_id",
            "lineage_executions": "execution_id",
            "lineage_edges": "edge_id",
        }
        missing: dict[str, tuple[str, ...]] = {}
        extra: dict[str, tuple[str, ...]] = {}
        changed: dict[str, tuple[str, ...]] = {}
        for table_name, id_column in id_columns.items():
            materialized = {
                row[id_column]: row
                for row in self._lake.table(table_name).to_arrow().to_pylist()
                # The refresh watermark sentinel lives in lineage_artifacts but is
                # not part of the projection; never count it as divergent.
                if row.get("kind") != "lineage-refresh-state"
            }
            expected = projected[table_name]
            missing_ids = tuple(sorted(set(expected) - set(materialized)))
            extra_ids = tuple(sorted(set(materialized) - set(expected)))
            changed_ids = tuple(
                sorted(
                    row_id
                    for row_id in set(expected) & set(materialized)
                    if _graph_row_digest(expected[row_id])
                    != _graph_row_digest(materialized[row_id])
                )
            )
            if missing_ids:
                missing[table_name] = missing_ids
            if extra_ids:
                extra[table_name] = extra_ids
            if changed_ids:
                changed[table_name] = changed_ids
        return LineageEmissionDivergence(
            lake_uri=self._lake.uri,
            missing_from_graph=missing,
            changed=changed,
            extra_in_graph=extra,
        )

    def emitted_transform_summary(self, transform_id: str) -> EmittedLineage | None:
        """Read back the inline-emitted lineage summary for a transform id.

        Lets CLI write commands surface the emitted lineage ids without a refresh
        (backlog 0098 AC#5). The execution is fetched by id via the indexed lineage
        read; returns ``None`` if the transform has not (yet) emitted lineage.
        """

        if not transform_id:
            return None
        execution_id = execution_artifact_id(str(transform_id))
        found = _fetch_rows_by_id_in(
            self._lake, "lineage_executions", "execution_id", [execution_id]
        )
        row = found.get(execution_id)
        if row is None:
            return None
        artifact_ids = tuple(
            sorted(
                set(row.get("input_artifact_ids") or [])
                | set(row.get("output_artifact_ids") or [])
            )
        )
        return EmittedLineage(
            lake_uri=self._lake.uri,
            transform_id=str(transform_id),
            execution_id=execution_id,
            kind=str(row.get("kind") or "transform"),
            status=row.get("status"),
            artifact_ids=artifact_ids,
            edge_ids=(),
            produced_outputs=bool(row.get("output_artifact_ids")),
        )

    def _project_full_graph(self, now: datetime) -> tuple[int, int, int]:
        """Project every canonical table into the graph tables; return row counts.

        The derivation is factored into :func:`_compute_full_graph` so that
        per-write emission (backlog 0098) and this full projection share the exact
        same per-transform logic and can never diverge.
        """

        artifacts, executions, edges = _compute_full_graph(self._lake)
        _replace_rows(
            self._lake,
            "lineage_artifacts",
            "artifact_id",
            [artifact.as_row(now) for artifact in sorted(artifacts.values(), key=lambda item: item.artifact_id)],
            LINEAGE_ARTIFACTS_SCHEMA,
        )
        _replace_rows(
            self._lake,
            "lineage_executions",
            "execution_id",
            [
                execution.as_row(now)
                for execution in sorted(executions.values(), key=lambda item: item.execution_id)
            ],
            LINEAGE_EXECUTIONS_SCHEMA,
        )
        _replace_rows(
            self._lake,
            "lineage_edges",
            "edge_id",
            [edge.as_row(now) for edge in sorted(edges.values(), key=lambda item: item.edge_id)],
            LINEAGE_EDGES_SCHEMA,
        )
        # Report the materialized graph size, not just the rows this projection
        # produced. Inline emission (backlog 0098) can leave rows the current
        # projection does not reproduce (emit-time table versions / row-sets); a
        # full projection upserts over them via merge_insert but does not remove
        # them, so the authoritative count is what is now on disk. This keeps the
        # count identical to the skipped-refresh path (which reads
        # ``_graph_row_counts``) so consecutive refreshes report the same totals.
        return _graph_row_counts(self._lake)

    def trace(
        self,
        artifact: str,
        *,
        kind: str | None = None,
        max_depth: int | None = None,
        edge_types: Iterable[str] = (),
        target_kinds: Iterable[str] = (),
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        table_versions: Mapping[str, int] | Iterable[str | tuple[str, int]] = (),
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> LineageGraph:
        """Traverse upstream provenance for an artifact id or domain handle.

        Pass ``page_size`` to return a bounded page with stable total counts and a
        ``next_page_token`` continuation handle (backlog 0097).
        """

        return _traverse_graph(
            self._lake,
            artifact,
            direction="upstream",
            kind=kind,
            max_depth=max_depth,
            edge_types=set(edge_types),
            target_kinds=set(target_kinds),
            created_after=created_after,
            created_before=created_before,
            table_versions=table_versions,
            page_size=page_size,
            page_token=page_token,
        )

    def impact(
        self,
        artifact: str,
        *,
        kind: str | None = None,
        max_depth: int | None = None,
        edge_types: Iterable[str] = (),
        target_kinds: Iterable[str] = (),
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        table_versions: Mapping[str, int] | Iterable[str | tuple[str, int]] = (),
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> LineageGraph:
        """Traverse downstream dependents for an artifact id or domain handle.

        Pass ``page_size`` to return a bounded page with stable total counts and a
        ``next_page_token`` continuation handle (backlog 0097).
        """

        return _traverse_graph(
            self._lake,
            artifact,
            direction="downstream",
            kind=kind,
            max_depth=max_depth,
            edge_types=set(edge_types),
            target_kinds=set(target_kinds),
            created_after=created_after,
            created_before=created_before,
            table_versions=table_versions,
            page_size=page_size,
            page_token=page_token,
        )

    def cypher_backend(self):
        """Return the optional ``lance-graph`` Cypher backend (backlog 0099).

        This is an optional, expert-only surface: canonical lineage state stays in
        the Lance tables and :meth:`trace`/:meth:`impact`/:meth:`audit` remain the
        stable API. The backend maps ``lineage_artifacts``/``lineage_executions``/
        ``lineage_edges`` into a lance-graph property graph so Cypher audit and
        blast-radius queries run over the same rows. Raises
        :class:`~lancedb_robotics.lineage_graph.LineageGraphExtraMissing` (a
        ``LineageError`` subclass) with an actionable message when the optional
        ``graph`` extra is not installed.
        """

        from lancedb_robotics.lineage_graph import LineageGraphBackend

        return LineageGraphBackend(self._lake)

    def cypher(
        self,
        query_text: str,
        *,
        parameters: dict[str, Any] | None = None,
        strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a Cypher query over the lineage property graph (backlog 0099).

        Convenience wrapper returning result rows as dicts. Requires the optional
        ``graph`` extra; the graph is read from the already-materialized lineage
        tables (no implicit refresh), exactly like :meth:`trace`/:meth:`impact`.
        """

        result = self.cypher_backend().query(
            query_text, parameters=parameters, strategy=strategy
        )
        return result.to_pylist()

    def compare_cypher_traversal(
        self,
        artifact: str,
        *,
        direction: str = "upstream",
        kind: str | None = None,
        max_depth: int | None = None,
    ):
        """Compare SDK traversal against Cypher reachability for one handle (0099).

        Returns a
        :class:`~lancedb_robotics.lineage_graph.CypherParityReport`. Requires the
        optional ``graph`` extra.
        """

        return self.cypher_backend().compare_traversal(
            artifact, direction=direction, kind=kind, max_depth=max_depth
        )

    def invalidate(
        self,
        artifact: str,
        *,
        kind: str | None = None,
        reason: str,
        severity: str = "high",
        discovered_by: str | None = None,
        actor: str | None = None,
        replacement: str | None = None,
        refresh: bool = True,
    ) -> LineageInvalidation:
        """Record a durable invalidation marker for an artifact or handle."""

        if refresh:
            self.refresh_graph()
        root_ids = _resolve_artifact_ids(self._lake, artifact, kind=kind)
        replacement_id = None
        if replacement:
            replacement_id = _resolve_artifact_ids(self._lake, replacement)[0]
        return _record_invalidation(
            self._lake,
            root_ids,
            reason=reason,
            severity=severity,
            discovered_by=discovered_by,
            actor=actor,
            replacement_artifact_id=replacement_id,
        )

    def rebuild_plan(
        self,
        artifact: str | None = None,
        *,
        kind: str | None = None,
        provider: str | None = None,
        provider_version: str | None = None,
        embedding_column: str | None = None,
        reason: str | None = None,
        severity: str = "high",
        discovered_by: str | None = None,
        actor: str | None = None,
        replacement: str | None = None,
        record_invalidation: bool = False,
        refresh: bool = True,
        max_depth: int | None = None,
        action_policy: ActionPolicy | Callable[[ActionContext], str | None] | None = None,
        max_affected_artifacts: int | None = None,
        max_actions: int | None = None,
        require_indexes: bool = False,
        page_size: int | None = None,
        page_token: str | None = None,
        summary: bool = False,
    ) -> RebuildPlan:
        """Return an ordered downstream rebuild plan for an invalidated handle.

        Pass ``artifact``/``kind`` for a concrete source, row, table version, or
        model handle. Pass ``provider`` with optional ``provider_version`` and
        ``embedding_column`` to plan from matching provider-backed transforms.

        Backlog 0110 -- scale and configurability:

        - ``action_policy`` remaps artifacts to actions without changing which
          artifacts are impacted. ``None`` keeps the built-in 0066 classification.
          Pass an :class:`ActionPolicy` (e.g. :class:`MappingActionPolicy`) or a
          callable ``(ActionContext) -> action | None``.
        - ``max_affected_artifacts`` / ``max_actions`` are hard guardrails; a plan
          that exceeds either raises :class:`RebuildPlanTooLarge` with remediation.
        - ``require_indexes=True`` fails fast (rather than full-scanning) when the
          lineage traversal indexes are missing.
        - ``page_size``/``page_token`` return a stable, bounded page of the ordered
          actions with a continuation handle. ``summary=True`` returns the
          aggregates only (no per-action rows) for a summary-first drill-down.
        """

        if refresh:
            self.refresh_graph()
        if require_indexes:
            _check_rebuild_index_readiness(self._lake)
        policy = _coerce_action_policy(action_policy)
        if artifact is not None:
            root_ids = _resolve_artifact_ids(self._lake, artifact, kind=kind)
            resolved_handle = str(artifact)
        else:
            root_ids = _provider_rebuild_roots(
                self._lake,
                provider=provider,
                provider_version=provider_version,
                embedding_column=embedding_column,
            )
            resolved_handle = _provider_handle(
                provider=provider,
                provider_version=provider_version,
                embedding_column=embedding_column,
            )

        graph = _impact_graph_for_roots(
            self._lake,
            root_ids,
            max_depth=max_depth,
            resolved_handle=resolved_handle,
        )
        affected_count = len(graph.artifacts)
        if max_affected_artifacts is not None and affected_count > max_affected_artifacts:
            raise RebuildPlanTooLarge(
                f"rebuild plan affects {affected_count} artifacts, exceeding "
                f"max_affected_artifacts={max_affected_artifacts}. Narrow the scope "
                "(max_depth, a more specific root), raise the cap, or use "
                "summary=True / page_size to review it in bounded pages."
            )
        actions = _rebuild_actions_for_graph(
            graph, reason=reason, policy=policy, severity=severity
        )
        if max_actions is not None and len(actions) > max_actions:
            raise RebuildPlanTooLarge(
                f"rebuild plan has {len(actions)} actions, exceeding "
                f"max_actions={max_actions}. Narrow the scope, raise the cap, or use "
                "summary=True / page_size to review it in bounded pages."
            )

        actions_by_type, affected_by_kind = _rebuild_plan_aggregates(graph, actions)
        total_actions = len(actions)
        invalidation = None
        if record_invalidation:
            replacement_id = None
            if replacement:
                replacement_id = _resolve_artifact_ids(self._lake, replacement)[0]
            invalidation = _record_invalidation(
                self._lake,
                root_ids,
                reason=reason or "unspecified invalidation",
                severity=severity,
                discovered_by=discovered_by,
                actor=actor,
                replacement_artifact_id=replacement_id,
            )

        bounded = summary or page_size is not None
        if summary:
            page_actions: tuple[RebuildPlanAction, ...] = ()
            next_token = None
            truncated = total_actions > 0
        else:
            digest = _rebuild_page_digest(
                root_ids, reason=reason, severity=severity, policy_name=policy.name
            )
            page_actions, next_token, truncated = _paginate_actions(
                actions, page_size=page_size, page_token=page_token, digest=digest
            )

        return RebuildPlan(
            lake_uri=self._lake.uri,
            root_artifact_ids=tuple(root_ids),
            actions=page_actions,
            reason=reason,
            severity=severity,
            invalidation=invalidation,
            # Keep the full graph on unbounded plans for backward compatibility;
            # drop it in bounded/summary modes so the output stays bounded.
            graph=None if bounded else graph,
            policy_name=policy.name,
            affected_artifact_count=affected_count,
            action_count=total_actions,
            actions_by_type=actions_by_type,
            affected_by_kind=affected_by_kind,
            page_size=page_size,
            total_actions=total_actions,
            next_page_token=next_token,
            truncated=truncated,
            summary_only=summary,
        )

    def rebuild_plan_summary(self, artifact: str | None = None, **kwargs) -> RebuildPlan:
        """Return a summary-first rebuild plan (aggregates only, no action rows).

        Convenience wrapper for ``rebuild_plan(..., summary=True)``. Drill down by
        re-calling :meth:`rebuild_plan` with ``page_size`` (and the resulting
        ``next_page_token``).
        """

        kwargs.pop("summary", None)
        return self.rebuild_plan(artifact, summary=True, **kwargs)

    def benchmark_rebuild_plan(
        self,
        artifact: str | None = None,
        *,
        repeat: int = 1,
        measure_memory: bool = True,
        refresh: bool = True,
        **plan_kwargs,
    ) -> dict[str, Any]:
        """Benchmark rebuild planning: affected/action counts, time, peak memory.

        Runs :meth:`rebuild_plan` ``repeat`` times and returns a report of the
        affected-artifact count, action count, per-run and best/mean traversal
        seconds, and (when ``measure_memory``) the peak allocation observed via
        ``tracemalloc``. When ``refresh`` is true the graph is refreshed once up
        front and the timed runs use ``refresh=False`` so timing measures planning,
        not the refresh. For synthetic-graph scaling fixtures (backlog 0110), not
        the hot path.
        """

        import time
        import tracemalloc

        plan_kwargs["refresh"] = False
        if refresh:
            self.refresh_graph()
        durations: list[float] = []
        peak_bytes: int | None = None
        plan: RebuildPlan | None = None
        runs = max(1, int(repeat))
        for _ in range(runs):
            if measure_memory:
                tracemalloc.start()
            start = time.perf_counter()
            plan = self.rebuild_plan(artifact, **plan_kwargs)
            durations.append(time.perf_counter() - start)
            if measure_memory:
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_bytes = peak if peak_bytes is None else max(peak_bytes, peak)
        assert plan is not None
        return {
            "schema_version": "lancedb-robotics/rebuild-plan-benchmark/v1",
            "lake_uri": self._lake.uri,
            "policy": plan.policy_name,
            "page_size": plan.page_size,
            "affected_artifact_count": plan.affected_artifact_count,
            "action_count": plan.action_count,
            "actions_by_type": dict(sorted(plan.actions_by_type.items())),
            "repeat": runs,
            "traversal_seconds": durations,
            "traversal_seconds_best": min(durations),
            "traversal_seconds_mean": sum(durations) / len(durations),
            "peak_memory_bytes": peak_bytes,
        }

    def evidence_pack(
        self,
        artifact: str,
        *,
        kind: str | None = None,
        checkpoint: bool = False,
        where: str | None = None,
        limit: int | None = None,
        max_depth: int | None = None,
        edge_types: Iterable[str] = (),
        target_kinds: Iterable[str] = (),
        refresh: bool = True,
        output_dir: str | None = None,
        materialize: bool = False,
        include_payloads: bool = False,
        include_attachments: bool = False,
        include_video: bool = False,
        redaction_policy: Any | None = None,
    ):
        """Build a source evidence pack for a checkpoint, model output, or artifact.

        ``checkpoint=True`` (or ``where``/``limit``) uses the legacy checkpoint
        regression trace. Otherwise the canonical lineage graph is refreshed and
        traced upstream from ``artifact``/``kind``. When ``redaction_policy`` is a
        :class:`~lancedb_robotics.redaction.ContextRedactionPolicy`, denied/secret
        context and environment keys are stripped from the manifest before the pack
        digest is computed and any bytes are materialized.
        """

        if checkpoint or where is not None or limit is not None:
            return self.trace_checkpoint(artifact, where=where, limit=limit).evidence_pack(
                output_dir=output_dir,
                materialize=materialize,
                include_payloads=include_payloads,
                include_attachments=include_attachments,
                include_video=include_video,
                redaction_policy=redaction_policy,
            )

        from lancedb_robotics.evidence import evidence_pack_from_graph

        if refresh:
            self.refresh_graph()
        graph = self.trace(
            artifact,
            kind=kind,
            max_depth=max_depth,
            edge_types=edge_types,
            target_kinds=target_kinds,
        )
        return evidence_pack_from_graph(
            self._lake,
            graph,
            output_dir=output_dir,
            materialize=materialize,
            include_payloads=include_payloads,
            include_attachments=include_attachments,
            include_video=include_video,
            redaction_policy=redaction_policy,
        )

    def replay_bundle(
        self,
        artifact: str,
        *,
        output_dir: str,
        kind: str | None = None,
        checkpoint: bool = False,
        where: str | None = None,
        limit: int | None = None,
        max_depth: int | None = None,
        edge_types: Iterable[str] = (),
        target_kinds: Iterable[str] = (),
        refresh: bool = True,
        include_mcap: bool = True,
        include_video: bool = False,
        include_gops: bool = True,
        viewer_formats: Iterable[str] = ("foxglove", "rerun"),
        max_bytes: int | None = None,
        max_files: int | None = None,
        storage_options: Mapping[str, Any] | None = None,
        auth_ref: str | None = None,
    ):
        """Export a replayable bundle (MCAP slices, video clip/GOP bytes) for an artifact.

        Builds a plan-only evidence pack for ``artifact`` (backlog 0065), then
        reconstructs deterministic, viewer-openable bundles from its source
        coordinates and codec-aware video refs (backlog 0107). The evidence pack
        stays the metadata source of truth; this is a downstream projection.
        """

        from lancedb_robotics.replay import build_replay_bundle

        pack = self.evidence_pack(
            artifact,
            kind=kind,
            checkpoint=checkpoint,
            where=where,
            limit=limit,
            max_depth=max_depth,
            edge_types=edge_types,
            target_kinds=target_kinds,
            refresh=refresh,
        )
        return build_replay_bundle(
            self._lake,
            pack.manifest,
            output_dir=output_dir,
            include_mcap=include_mcap,
            include_video=include_video,
            include_gops=include_gops,
            viewer_formats=tuple(viewer_formats),
            max_bytes=max_bytes,
            max_files=max_files,
            storage_options=storage_options,
            auth_ref=auth_ref,
        )

    # --- Durable evidence-pack catalog (backlog 0108) -----------------------

    def record_evidence_pack(self, pack, **kwargs):
        """Record an evidence pack in the durable catalog (idempotent by digest)."""
        from lancedb_robotics.evidence_catalog import record_evidence_pack

        return record_evidence_pack(self._lake, pack, **kwargs)

    def load_evidence_pack(self, *, digest: str | None = None, subject: str | None = None):
        """Reload a catalog entry and its manifest by digest or subject handle."""
        from lancedb_robotics.evidence_catalog import load_evidence_pack

        return load_evidence_pack(self._lake, digest=digest, subject=subject)

    def list_evidence_packs(self, **kwargs):
        """List catalog entries newest-first, filtered and bounded by page_size."""
        from lancedb_robotics.evidence_catalog import list_evidence_packs

        return list_evidence_packs(self._lake, **kwargs)

    def materialize_evidence_pack(self, pack, **kwargs):
        """Materialize a pack's blobs to a local/object-store destination (chunked, resumable)."""
        from lancedb_robotics.evidence_catalog import materialize_evidence_pack

        return materialize_evidence_pack(self._lake, pack, **kwargs)

    def plan_evidence_materialization(self, pack, **kwargs):
        """Build a bounded, chunked copy plan and enforce limits before any copy."""
        from lancedb_robotics.evidence_catalog import plan_materialization

        return plan_materialization(self._lake, pack, **kwargs)

    def set_evidence_retention(self, digest: str, **kwargs):
        """Update retention/protection metadata for a recorded pack."""
        from lancedb_robotics.evidence_catalog import set_evidence_retention

        return set_evidence_retention(self._lake, digest, **kwargs)

    def evidence_retention_plan(self, **kwargs):
        """Report which recorded packs are protected vs safe to expire."""
        from lancedb_robotics.evidence_catalog import evidence_retention_plan

        return evidence_retention_plan(self._lake, **kwargs)

    def expire_evidence_pack(self, digest: str, **kwargs):
        """Delete a catalog entry, refused if protected unless force=True."""
        from lancedb_robotics.evidence_catalog import expire_evidence_pack

        return expire_evidence_pack(self._lake, digest, **kwargs)

    def evidence_pack_events(self, **kwargs):
        """Return evidence-pack audit events, optionally filtered."""
        from lancedb_robotics.evidence_catalog import evidence_pack_events

        return evidence_pack_events(self._lake, **kwargs)

    # --- Durable rebuild-plan catalog + orchestrator handoff (backlog 0109) --

    def record_rebuild_plan(self, plan, **kwargs):
        """Record a rebuild plan in the durable catalog (idempotent by digest)."""
        from lancedb_robotics.rebuild_catalog import record_rebuild_plan

        return record_rebuild_plan(self._lake, plan, **kwargs)

    def get_rebuild_plan(self, plan_id_or_digest: str):
        """Reload a catalog entry and its stored plan payload by id or digest."""
        from lancedb_robotics.rebuild_catalog import get_rebuild_plan

        return get_rebuild_plan(self._lake, plan_id_or_digest)

    def rebuild_plans(self, **kwargs):
        """List recorded rebuild plans newest-first, filtered and bounded."""
        from lancedb_robotics.rebuild_catalog import rebuild_plans

        return rebuild_plans(self._lake, **kwargs)

    def update_rebuild_plan_status(self, plan_id: str, status: str, **kwargs):
        """Move a recorded plan to a new lifecycle status (optimistically)."""
        from lancedb_robotics.rebuild_catalog import update_rebuild_plan_status

        return update_rebuild_plan_status(self._lake, plan_id, status, **kwargs)

    def export_rebuild_plan_dispatch(self, plan_id: str, **kwargs):
        """Build a deterministic orchestrator handoff payload for a recorded plan."""
        from lancedb_robotics.rebuild_catalog import export_rebuild_plan_dispatch

        return export_rebuild_plan_dispatch(self._lake, plan_id, **kwargs)

    def rebuild_plan_events(self, **kwargs):
        """Return rebuild-plan audit events, optionally filtered."""
        from lancedb_robotics.rebuild_catalog import rebuild_plan_events

        return rebuild_plan_events(self._lake, **kwargs)

    # --- Durable retention-policy catalog + governance hooks (backlog 0111) --

    def record_retention_policy(self, policy, **kwargs):
        """Record a retention policy in the durable catalog (idempotent by digest)."""
        from lancedb_robotics.retention_catalog import record_retention_policy

        return record_retention_policy(self._lake, policy, **kwargs)

    def get_retention_policy(self, policy_id_or_digest: str):
        """Reload a catalog entry and its stored policy definition by id or digest."""
        from lancedb_robotics.retention_catalog import get_retention_policy

        return get_retention_policy(self._lake, policy_id_or_digest)

    def retention_policies(self, **kwargs):
        """List recorded retention policies newest-first, filtered and bounded."""
        from lancedb_robotics.retention_catalog import retention_policies

        return retention_policies(self._lake, **kwargs)

    def update_retention_policy_status(self, policy_id: str, status: str, **kwargs):
        """Move a recorded policy to a new lifecycle status (optimistically)."""
        from lancedb_robotics.retention_catalog import update_retention_policy_status

        return update_retention_policy_status(self._lake, policy_id, status, **kwargs)

    def apply_retention_policy(self, policy_id: str, **kwargs):
        """Expand an active policy into explicit artifact holds (idempotent, dry-run default)."""
        from lancedb_robotics.retention_catalog import apply_retention_policy

        return apply_retention_policy(self._lake, policy_id, **kwargs)

    def release_retention_policy(self, policy_id: str, **kwargs):
        """Clear the holds a policy applied, leaving manual/other-policy holds."""
        from lancedb_robotics.retention_catalog import release_retention_policy

        return release_retention_policy(self._lake, policy_id, **kwargs)

    def resolve_retention_holds(self, **kwargs):
        """Merge policy-applied + artifact-local holds into maintenance's pin shape."""
        from lancedb_robotics.retention_catalog import resolve_retention_holds

        return resolve_retention_holds(self._lake, **kwargs)

    def retention_expiration_notices(self, **kwargs):
        """Report holds whose retain_until has passed or is upcoming (append-safe notices)."""
        from lancedb_robotics.retention_catalog import retention_expiration_notices

        return retention_expiration_notices(self._lake, **kwargs)

    def export_retention_policy_state(self, **kwargs):
        """Project policy + resolved-hold state out for external governance systems."""
        from lancedb_robotics.retention_catalog import export_retention_policy_state

        return export_retention_policy_state(self._lake, **kwargs)

    def project_retention_state(self, sink, **kwargs):
        """Build a governance projection and hand it to a caller-supplied sink."""
        from lancedb_robotics.retention_catalog import project_retention_state

        return project_retention_state(self._lake, sink, **kwargs)

    def retention_policy_events(self, **kwargs):
        """Return retention-policy audit events, optionally filtered."""
        from lancedb_robotics.retention_catalog import retention_policy_events

        return retention_policy_events(self._lake, **kwargs)

    # --- Queryable external-context catalog (backlog 0114) ------------------

    def backfill_external_contexts(self, **kwargs):
        """Index external run/job context from canonical rows into the catalog."""
        from lancedb_robotics.external_context_catalog import backfill_external_contexts

        return backfill_external_contexts(self._lake, **kwargs)

    def record_external_context(self, **kwargs):
        """Record one external-context row directly (idempotent by content digest)."""
        from lancedb_robotics.external_context_catalog import record_external_context

        return record_external_context(self._lake, **kwargs)

    def find_external_context(self, **kwargs):
        """Resolve external run/job handles to canonical executions/artifacts (paged)."""
        from lancedb_robotics.external_context_catalog import find_external_context

        return find_external_context(self._lake, **kwargs)

    def get_external_context(self, context_id: str):
        """Reload a single external-context catalog entry by id."""
        from lancedb_robotics.external_context_catalog import get_external_context

        return get_external_context(self._lake, context_id)

    def set_external_context_retention(self, context_id: str, **kwargs):
        """Update retention/protection/hold metadata for a recorded context."""
        from lancedb_robotics.external_context_catalog import set_external_context_retention

        return set_external_context_retention(self._lake, context_id, **kwargs)

    def external_context_retention_plan(self, **kwargs):
        """Report which recorded contexts are held vs safe to expire."""
        from lancedb_robotics.external_context_catalog import external_context_retention_plan

        return external_context_retention_plan(self._lake, **kwargs)

    def expire_external_context(self, context_id: str, **kwargs):
        """Delete a recorded context row, refused if held unless force=True."""
        from lancedb_robotics.external_context_catalog import expire_external_context

        return expire_external_context(self._lake, context_id, **kwargs)

    def external_context_events(self, **kwargs):
        """Return external-context audit events, optionally filtered."""
        from lancedb_robotics.external_context_catalog import external_context_events

        return external_context_events(self._lake, **kwargs)

    def trace_checkpoint(
        self,
        model_run_id: str,
        *,
        where: str | None = None,
        limit: int | None = None,
    ) -> RegressionTrace:
        """Trace a model/checkpoint run to a training snapshot and source rows.

        ``model_run_id`` is matched against ``model_outputs.producer_run_id`` and
        compatible metadata aliases (``model_run_id``, ``checkpoint_id``, and
        ``training_run_id``). The linked snapshot is resolved from
        ``dataset_id`` first, then from metadata keys such as ``dataset_tag`` or
        ``snapshot_name`` when an external trainer recorded only a tag/name.
        """

        if not model_run_id:
            raise LineageError("model_run_id is required")
        if limit is not None and limit < 1:
            raise LineageError("limit must be >= 1")

        training_run = None
        model_artifacts: list[dict[str, Any]] = []
        try:
            model_outputs = _model_outputs_for_run(self._lake, model_run_id)
            snapshot = _resolve_snapshot(self._lake, model_outputs)
        except LineageError as model_output_error:
            model_outputs = []
            model_artifacts = _model_artifacts_for_run(self._lake, model_run_id)
            if not model_artifacts:
                raise LineageError(
                    f"no model_outputs or model_artifacts rows linked to "
                    f"model_run_id {model_run_id!r}"
                ) from model_output_error
            training_run = _training_run_for_model_artifacts(self._lake, model_artifacts)
            snapshot = _resolve_snapshot_from_training_run(self._lake, training_run)
        snapshot_versions = _table_versions(snapshot)
        context = _snapshot_context(self._lake, snapshot)
        rows = _trace_rows(context)
        rows = _filter_rows(rows, where)
        if limit is not None:
            rows = rows[:limit]

        source_logs = _source_logs(rows)
        transform_runs = _transform_lineage(
            self._lake,
            snapshot,
            model_outputs,
            rows,
            training_run=training_run,
            model_artifacts=model_artifacts,
        )
        return RegressionTrace(
            lake_uri=self._lake.uri,
            model_run_id=model_run_id,
            dataset_snapshot=snapshot,
            table_versions=snapshot_versions,
            model_outputs=tuple(model_outputs),
            rows=tuple(rows),
            source_logs=tuple(source_logs),
            transform_runs=tuple(transform_runs),
            where=where,
            training_run=training_run,
            model_artifacts=tuple(model_artifacts),
            _lake=self._lake,
        )


@dataclass(frozen=True)
class _SnapshotContext:
    snapshot: dict[str, Any]
    scenario_ids: tuple[str, ...]
    scenarios: dict[str, dict[str, Any]]
    observations: dict[str, dict[str, Any]]
    runs: dict[str, dict[str, Any]]


_COMPARISON_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(==|=|!=|<>|<=|>=|<|>)\s*(.+?)\s*$"
)
_IN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s+IN\s*\((.+)\)\s*$", re.I)
_IS_NULL_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s+IS\s+(NOT\s+)?NULL\s*$", re.I
)
_AND_RE = re.compile(r"\s+AND\s+", re.I)

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def trace_checkpoint(
    lake: Lake,
    model_run_id: str,
    *,
    where: str | None = None,
    limit: int | None = None,
) -> RegressionTrace:
    """Convenience wrapper for ``lake.lineage.trace_checkpoint(...)``."""

    return LakeLineage(lake).trace_checkpoint(model_run_id, where=where, limit=limit)


def refresh_graph(
    lake: Lake,
    *,
    incremental: bool = True,
    force_full: bool = False,
    dry_run: bool = False,
    reconcile: bool = True,
) -> LineageRefreshReport:
    """Convenience wrapper for ``lake.lineage.refresh_graph()``."""

    return LakeLineage(lake).refresh_graph(
        incremental=incremental,
        force_full=force_full,
        dry_run=dry_run,
        reconcile=reconcile,
    )


def plan_refresh(
    lake: Lake, *, incremental: bool = True, force_full: bool = False
) -> LineageRefreshPlan:
    """Convenience wrapper for ``lake.lineage.plan_refresh()``."""

    return LakeLineage(lake).plan_refresh(incremental=incremental, force_full=force_full)


def trace(
    lake: Lake,
    artifact: str,
    *,
    kind: str | None = None,
    max_depth: int | None = None,
    edge_types: Iterable[str] = (),
    target_kinds: Iterable[str] = (),
    created_after: datetime | str | None = None,
    created_before: datetime | str | None = None,
    table_versions: Mapping[str, int] | Iterable[str | tuple[str, int]] = (),
    page_size: int | None = None,
    page_token: str | None = None,
) -> LineageGraph:
    """Convenience wrapper for upstream artifact traversal."""

    return LakeLineage(lake).trace(
        artifact,
        kind=kind,
        max_depth=max_depth,
        edge_types=edge_types,
        target_kinds=target_kinds,
        created_after=created_after,
        created_before=created_before,
        table_versions=table_versions,
        page_size=page_size,
        page_token=page_token,
    )


def impact(
    lake: Lake,
    artifact: str,
    *,
    kind: str | None = None,
    max_depth: int | None = None,
    edge_types: Iterable[str] = (),
    target_kinds: Iterable[str] = (),
    created_after: datetime | str | None = None,
    created_before: datetime | str | None = None,
    table_versions: Mapping[str, int] | Iterable[str | tuple[str, int]] = (),
    page_size: int | None = None,
    page_token: str | None = None,
) -> LineageGraph:
    """Convenience wrapper for downstream artifact traversal."""

    return LakeLineage(lake).impact(
        artifact,
        kind=kind,
        max_depth=max_depth,
        edge_types=edge_types,
        target_kinds=target_kinds,
        created_after=created_after,
        created_before=created_before,
        table_versions=table_versions,
        page_size=page_size,
        page_token=page_token,
    )


def invalidate(
    lake: Lake,
    artifact: str,
    *,
    kind: str | None = None,
    reason: str,
    severity: str = "high",
    discovered_by: str | None = None,
    actor: str | None = None,
    replacement: str | None = None,
    refresh: bool = True,
) -> LineageInvalidation:
    """Convenience wrapper for ``lake.lineage.invalidate(...)``."""

    return LakeLineage(lake).invalidate(
        artifact,
        kind=kind,
        reason=reason,
        severity=severity,
        discovered_by=discovered_by,
        actor=actor,
        replacement=replacement,
        refresh=refresh,
    )


def retain(
    lake: Lake,
    artifact: str,
    *,
    kind: str | None = None,
    retain_until: datetime | str | None = None,
    legal_hold: bool = False,
    audit_hold: bool = False,
    promotion_hold: bool = False,
    owner: str | None = None,
    reason: str | None = None,
    refresh: bool = True,
) -> LineageRetentionHold:
    """Convenience wrapper for ``lake.lineage.retain(...)``."""

    return LakeLineage(lake).retain(
        artifact,
        kind=kind,
        retain_until=retain_until,
        legal_hold=legal_hold,
        audit_hold=audit_hold,
        promotion_hold=promotion_hold,
        owner=owner,
        reason=reason,
        refresh=refresh,
    )


def clear_retention(
    lake: Lake,
    artifact: str,
    *,
    kind: str | None = None,
    refresh: bool = True,
) -> LineageRetentionHold:
    """Convenience wrapper for ``lake.lineage.clear_retention(...)``."""

    return LakeLineage(lake).clear_retention(artifact, kind=kind, refresh=refresh)


def audit(
    lake: Lake,
    artifact: str | None = None,
    *,
    kind: str | None = None,
    refresh: bool = True,
    check_sources: bool = True,
    check_remote_sources: bool = False,
    source_auth_ref: str | None = None,
    source_storage_options: Mapping[str, Any] | None = None,
    page_size: int | None = None,
    page_token: str | None = None,
    now: datetime | None = None,
) -> LineageAuditReport:
    """Convenience wrapper for ``lake.lineage.audit(...)``."""

    return LakeLineage(lake).audit(
        artifact,
        kind=kind,
        refresh=refresh,
        check_sources=check_sources,
        check_remote_sources=check_remote_sources,
        source_auth_ref=source_auth_ref,
        source_storage_options=source_storage_options,
        page_size=page_size,
        page_token=page_token,
        now=now,
    )


def record_audit_report(lake: Lake, report, **kwargs):
    """Convenience wrapper for ``lake.lineage.record_audit_report(...)`` (0112)."""

    return LakeLineage(lake).record_audit_report(report, **kwargs)


def get_audit_report(lake: Lake, report_id_or_digest: str):
    """Convenience wrapper for ``lake.lineage.get_audit_report(...)`` (0112)."""

    return LakeLineage(lake).get_audit_report(report_id_or_digest)


def audit_reports(lake: Lake, **kwargs):
    """Convenience wrapper for ``lake.lineage.audit_reports(...)`` (0112)."""

    return LakeLineage(lake).audit_reports(**kwargs)


def audit_findings(lake: Lake, report_id_or_digest: str, **kwargs):
    """Convenience wrapper for ``lake.lineage.audit_findings(...)`` (0112)."""

    return LakeLineage(lake).audit_findings(report_id_or_digest, **kwargs)


def iter_audit_findings_ndjson(lake: Lake, report_id_or_digest: str, **kwargs):
    """Convenience wrapper for ``lake.lineage.iter_audit_findings_ndjson(...)`` (0112)."""

    return LakeLineage(lake).iter_audit_findings_ndjson(report_id_or_digest, **kwargs)


def rebuild_plan(
    lake: Lake,
    artifact: str | None = None,
    *,
    kind: str | None = None,
    provider: str | None = None,
    provider_version: str | None = None,
    embedding_column: str | None = None,
    reason: str | None = None,
    severity: str = "high",
    discovered_by: str | None = None,
    actor: str | None = None,
    replacement: str | None = None,
    record_invalidation: bool = False,
    refresh: bool = True,
    max_depth: int | None = None,
    action_policy: ActionPolicy | Callable[[ActionContext], str | None] | None = None,
    max_affected_artifacts: int | None = None,
    max_actions: int | None = None,
    require_indexes: bool = False,
    page_size: int | None = None,
    page_token: str | None = None,
    summary: bool = False,
) -> RebuildPlan:
    """Convenience wrapper for ``lake.lineage.rebuild_plan(...)``."""

    return LakeLineage(lake).rebuild_plan(
        artifact,
        kind=kind,
        provider=provider,
        provider_version=provider_version,
        embedding_column=embedding_column,
        reason=reason,
        severity=severity,
        discovered_by=discovered_by,
        actor=actor,
        replacement=replacement,
        record_invalidation=record_invalidation,
        refresh=refresh,
        max_depth=max_depth,
        action_policy=action_policy,
        max_affected_artifacts=max_affected_artifacts,
        max_actions=max_actions,
        require_indexes=require_indexes,
        page_size=page_size,
        page_token=page_token,
        summary=summary,
    )


def rebuild_plan_summary(lake: Lake, artifact: str | None = None, **kwargs) -> RebuildPlan:
    """Convenience wrapper for ``lake.lineage.rebuild_plan_summary(...)`` (0110)."""

    return LakeLineage(lake).rebuild_plan_summary(artifact, **kwargs)


def benchmark_rebuild_plan(lake: Lake, artifact: str | None = None, **kwargs) -> dict[str, Any]:
    """Convenience wrapper for ``lake.lineage.benchmark_rebuild_plan(...)`` (0110)."""

    return LakeLineage(lake).benchmark_rebuild_plan(artifact, **kwargs)


def evidence_pack(
    lake: Lake,
    artifact: str,
    *,
    kind: str | None = None,
    checkpoint: bool = False,
    where: str | None = None,
    limit: int | None = None,
    max_depth: int | None = None,
    edge_types: Iterable[str] = (),
    target_kinds: Iterable[str] = (),
    refresh: bool = True,
    output_dir: str | None = None,
    materialize: bool = False,
    include_payloads: bool = False,
    include_attachments: bool = False,
    include_video: bool = False,
    redaction_policy: Any | None = None,
):
    """Convenience wrapper for ``lake.lineage.evidence_pack(...)``."""

    return LakeLineage(lake).evidence_pack(
        artifact,
        kind=kind,
        checkpoint=checkpoint,
        where=where,
        limit=limit,
        max_depth=max_depth,
        edge_types=edge_types,
        target_kinds=target_kinds,
        refresh=refresh,
        output_dir=output_dir,
        materialize=materialize,
        include_payloads=include_payloads,
        include_attachments=include_attachments,
        include_video=include_video,
        redaction_policy=redaction_policy,
    )


def replay_bundle(
    lake: Lake,
    artifact: str,
    *,
    output_dir: str,
    kind: str | None = None,
    checkpoint: bool = False,
    where: str | None = None,
    limit: int | None = None,
    max_depth: int | None = None,
    edge_types: Iterable[str] = (),
    target_kinds: Iterable[str] = (),
    refresh: bool = True,
    include_mcap: bool = True,
    include_video: bool = False,
    include_gops: bool = True,
    viewer_formats: Iterable[str] = ("foxglove", "rerun"),
    max_bytes: int | None = None,
    max_files: int | None = None,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
):
    """Convenience wrapper for ``lake.lineage.replay_bundle(...)``."""

    return LakeLineage(lake).replay_bundle(
        artifact,
        output_dir=output_dir,
        kind=kind,
        checkpoint=checkpoint,
        where=where,
        limit=limit,
        max_depth=max_depth,
        edge_types=edge_types,
        target_kinds=target_kinds,
        refresh=refresh,
        include_mcap=include_mcap,
        include_video=include_video,
        include_gops=include_gops,
        viewer_formats=viewer_formats,
        max_bytes=max_bytes,
        max_files=max_files,
        storage_options=storage_options,
        auth_ref=auth_ref,
    )


def record_evidence_pack(lake: Lake, pack, **kwargs):
    """Convenience wrapper for ``lake.lineage.record_evidence_pack(...)`` (0108)."""

    return LakeLineage(lake).record_evidence_pack(pack, **kwargs)


def load_evidence_pack(lake: Lake, *, digest: str | None = None, subject: str | None = None):
    """Convenience wrapper for ``lake.lineage.load_evidence_pack(...)`` (0108)."""

    return LakeLineage(lake).load_evidence_pack(digest=digest, subject=subject)


def list_evidence_packs(lake: Lake, **kwargs):
    """Convenience wrapper for ``lake.lineage.list_evidence_packs(...)`` (0108)."""

    return LakeLineage(lake).list_evidence_packs(**kwargs)


def materialize_evidence_pack(lake: Lake, pack, **kwargs):
    """Convenience wrapper for ``lake.lineage.materialize_evidence_pack(...)`` (0108)."""

    return LakeLineage(lake).materialize_evidence_pack(pack, **kwargs)


def set_evidence_retention(lake: Lake, digest: str, **kwargs):
    """Convenience wrapper for ``lake.lineage.set_evidence_retention(...)`` (0108)."""

    return LakeLineage(lake).set_evidence_retention(digest, **kwargs)


def evidence_retention_plan(lake: Lake, **kwargs):
    """Convenience wrapper for ``lake.lineage.evidence_retention_plan(...)`` (0108)."""

    return LakeLineage(lake).evidence_retention_plan(**kwargs)


def expire_evidence_pack(lake: Lake, digest: str, **kwargs):
    """Convenience wrapper for ``lake.lineage.expire_evidence_pack(...)`` (0108)."""

    return LakeLineage(lake).expire_evidence_pack(digest, **kwargs)


def record_rebuild_plan(lake: Lake, plan, **kwargs):
    """Convenience wrapper for ``lake.lineage.record_rebuild_plan(...)`` (0109)."""

    return LakeLineage(lake).record_rebuild_plan(plan, **kwargs)


def get_rebuild_plan(lake: Lake, plan_id_or_digest: str):
    """Convenience wrapper for ``lake.lineage.get_rebuild_plan(...)`` (0109)."""

    return LakeLineage(lake).get_rebuild_plan(plan_id_or_digest)


def rebuild_plans(lake: Lake, **kwargs):
    """Convenience wrapper for ``lake.lineage.rebuild_plans(...)`` (0109)."""

    return LakeLineage(lake).rebuild_plans(**kwargs)


def update_rebuild_plan_status(lake: Lake, plan_id: str, status: str, **kwargs):
    """Convenience wrapper for ``lake.lineage.update_rebuild_plan_status(...)`` (0109)."""

    return LakeLineage(lake).update_rebuild_plan_status(plan_id, status, **kwargs)


def export_rebuild_plan_dispatch(lake: Lake, plan_id: str, **kwargs):
    """Convenience wrapper for ``lake.lineage.export_rebuild_plan_dispatch(...)`` (0109)."""

    return LakeLineage(lake).export_rebuild_plan_dispatch(plan_id, **kwargs)


def record_retention_policy(lake: Lake, policy, **kwargs):
    """Convenience wrapper for ``lake.lineage.record_retention_policy(...)`` (0111)."""

    return LakeLineage(lake).record_retention_policy(policy, **kwargs)


def get_retention_policy(lake: Lake, policy_id_or_digest: str):
    """Convenience wrapper for ``lake.lineage.get_retention_policy(...)`` (0111)."""

    return LakeLineage(lake).get_retention_policy(policy_id_or_digest)


def retention_policies(lake: Lake, **kwargs):
    """Convenience wrapper for ``lake.lineage.retention_policies(...)`` (0111)."""

    return LakeLineage(lake).retention_policies(**kwargs)


def update_retention_policy_status(lake: Lake, policy_id: str, status: str, **kwargs):
    """Convenience wrapper for ``lake.lineage.update_retention_policy_status(...)`` (0111)."""

    return LakeLineage(lake).update_retention_policy_status(policy_id, status, **kwargs)


def apply_retention_policy(lake: Lake, policy_id: str, **kwargs):
    """Convenience wrapper for ``lake.lineage.apply_retention_policy(...)`` (0111)."""

    return LakeLineage(lake).apply_retention_policy(policy_id, **kwargs)


def release_retention_policy(lake: Lake, policy_id: str, **kwargs):
    """Convenience wrapper for ``lake.lineage.release_retention_policy(...)`` (0111)."""

    return LakeLineage(lake).release_retention_policy(policy_id, **kwargs)


def resolve_retention_holds(lake: Lake, **kwargs):
    """Convenience wrapper for ``lake.lineage.resolve_retention_holds(...)`` (0111)."""

    return LakeLineage(lake).resolve_retention_holds(**kwargs)


def export_retention_policy_state(lake: Lake, **kwargs):
    """Convenience wrapper for ``lake.lineage.export_retention_policy_state(...)`` (0111)."""

    return LakeLineage(lake).export_retention_policy_state(**kwargs)


def rebuild_plan_events(lake: Lake, **kwargs):
    """Convenience wrapper for ``lake.lineage.rebuild_plan_events(...)`` (0109)."""

    return LakeLineage(lake).rebuild_plan_events(**kwargs)


def artifact_id(kind: str, **identity: Any) -> str:
    """Stable LanceDB Robotics artifact id for an external or internal entity."""

    return _artifact_id_for_fields(kind, **identity)


def snapshot_artifact_id(dataset_id: str) -> str:
    return f"lancedb-robotics:snapshot:{dataset_id}"


def table_version_artifact_id(table_name: str, version: int, tag: str | None = None) -> str:
    suffix = f"tag:{tag}" if tag else f"v:{int(version)}"
    return f"lancedb-robotics:table:{table_name}:{suffix}"


def execution_artifact_id(execution_id: str) -> str:
    return f"lancedb-robotics:execution:{execution_id}"


def training_run_artifact_id(training_run_id: str) -> str:
    return f"lancedb-robotics:training-run:{training_run_id}"


def model_artifact_lineage_id(model_artifact_id: str) -> str:
    return f"lancedb-robotics:model:{model_artifact_id}"


def evaluation_run_artifact_id(eval_run_id: str) -> str:
    return f"lancedb-robotics:evaluation-run:{eval_run_id}"


_GRAPH_TABLES = {"lineage_artifacts", "lineage_executions", "lineage_edges"}
_GRAPH_PROJECTION_EXCLUDED_TABLES = {
    *_GRAPH_TABLES,
    "lineage_delivery_attempts",
    "evidence_packs",
    "evidence_pack_events",
    "retention_policies",
    "retention_policy_events",
    "external_contexts",
    "external_context_events",
    "training_reports",
}
_GRAPH_SOURCE_TABLES = tuple(
    name for name in CANONICAL_TABLES if name not in _GRAPH_PROJECTION_EXCLUDED_TABLES
)
_ID_COLUMNS = {
    "integration_sources": "source_id",
    "runs": "run_id",
    "episodes": "episode_id",
    "observations": "observation_id",
    "videos": "video_id",
    "video_encodings": "encoding_id",
    "attachments": "attachment_id",
    "events": "event_id",
    "scenarios": "scenario_id",
    "dataset_snapshots": "dataset_id",
    "training_runs": "training_run_id",
    "model_artifacts": "model_artifact_id",
    "evaluation_runs": "eval_run_id",
    "curation_views": "view_id",
    "curation_memberships": "membership_id",
    "curation_review_queues": "queue_item_id",
    "curation_materializations": "materialization_id",
    "curation_comparisons": "comparison_id",
    "labels": "label_id",
    "model_outputs": "model_output_id",
    "feedback": "feedback_id",
    "alignment_jobs": "alignment_id",
    "aligned_frames": "aligned_frame_id",
    "aligned_ticks": "aligned_tick_id",
    "transform_runs": "transform_id",
}
_HANDLE_KIND_ALIASES = {
    "artifact": "artifact",
    "artifact-id": "artifact",
    "raw": "source",
    "raw-uri": "source",
    "source": "source",
    "checksum": "source",
    "snapshot": "dataset-snapshot",
    "dataset": "dataset-snapshot",
    "dataset-id": "dataset-snapshot",
    "dataset-snapshot": "dataset-snapshot",
    "snapshot-tag": "dataset-snapshot",
    "run": "run",
    "observation": "observation",
    "scenario": "scenario",
    "episode": "episode",
    "aligned-frame": "aligned-frame",
    "aligned_frame": "aligned-frame",
    "aligned-tick": "aligned-tick",
    "aligned_tick": "aligned-tick",
    "model-output": "model-output",
    "model_output": "model-output",
    "label": "label",
    "labels": "label",
    "feedback": "feedback",
    "feedback-event": "feedback",
    "checkpoint": "model",
    "model": "model",
    "model-artifact": "model",
    "training-run": "training-run",
    "training_run": "training-run",
    "evaluation-run": "evaluation-run",
    "eval-run": "evaluation-run",
    "transform": "transform",
    "transform-run": "transform",
    "projection": "projection-manifest",
    "projection-manifest": "projection-manifest",
}
_HANDLE_KIND_TABLES = {
    "run": ("runs", "run_id"),
    "observation": ("observations", "observation_id"),
    "scenario": ("scenarios", "scenario_id"),
    "episode": ("episodes", "episode_id"),
    "aligned-frame": ("aligned_frames", "aligned_frame_id"),
    "aligned-tick": ("aligned_ticks", "aligned_tick_id"),
    "model-output": ("model_outputs", "model_output_id"),
    "label": ("labels", "label_id"),
    "feedback": ("feedback", "feedback_id"),
}
_RETENTION_METADATA_KEYS = (
    "retain_until",
    "legal_hold",
    "audit_hold",
    "promotion_hold",
    "owner",
    "reason",
    "retention_created_at",
)


def _stable_digest(payload: Any) -> str:
    encoded = _json_dumps(payload).encode()
    return hashlib.sha1(encoded).hexdigest()[:20]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _artifact_id_for_fields(kind: str, **identity: Any) -> str:
    if not kind:
        raise LineageError("artifact kind is required")
    cleaned = {
        key: value
        for key, value in sorted(identity.items())
        if value is not None and value != "" and value != [] and value != ()
    }
    if not cleaned:
        raise LineageError("artifact identity requires at least one stable field")
    return f"lancedb-robotics:{kind}:{_stable_digest(cleaned)}"


def _execution_id_for_fields(
    kind: str,
    *,
    transform_id: str | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    if transform_id:
        return execution_artifact_id(transform_id)
    return f"lancedb-robotics:execution:{_stable_digest({'kind': kind, 'name': name, 'params': params})}"


def _edge_id(
    edge_type: str,
    from_artifact_id: str,
    to_artifact_id: str,
    execution_id: str | None = None,
) -> str:
    if not edge_type:
        raise LineageError("edge_type is required")
    if not from_artifact_id or not to_artifact_id:
        raise LineageError("lineage edges require from_artifact_id and to_artifact_id")
    return "lancedb-robotics:edge:" + _stable_digest(
        {
            "edge_type": edge_type,
            "from": from_artifact_id,
            "to": to_artifact_id,
            "execution_id": execution_id,
        }
    )


def _kv_items(metadata: dict[str, Any] | None) -> list[dict[str, str]]:
    if not metadata:
        return []
    return [
        {"key": str(key), "value": _metadata_value(value)}
        for key, value in sorted(metadata.items())
        if value is not None
    ]


def _metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _json_dumps(value)


def _string_metadata(metadata: dict[str, Any] | None) -> dict[str, str] | None:
    if metadata is None:
        return None
    return {str(key): _metadata_value(value) for key, value in metadata.items() if value is not None}


def _version_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "table": str(row.get("table") or ""),
        "version": int(row.get("version") or 0),
        "tag": "" if row.get("tag") is None else str(row.get("tag")),
    }


# Lance 7.0.0's mini-block encoder packs list/struct columns into 32 KiB chunks
# and panics at the Rust layer (``primitive.rs`` ``assert!(chunk_bytes <=
# max_chunk_size)``) when a single page accumulates more rep/def levels than fit.
# That happens once a list column is mostly empty across a huge number of rows --
# exactly the shape of the ``metadata``/``row_ids`` columns when a full
# ``refresh_graph`` projects hundreds of thousands of per-row artifacts/edges on a
# large corpus. Writing in bounded batches caps the rows per fragment so a chunk
# can never overflow; 16 Ki stays well under the ~64 Ki all-empty-list ceiling.
_GRAPH_WRITE_BATCH_ROWS = 16_384


def _refresh_lineage_predicate_indexes(lake: Lake) -> None:
    """Ensure the lineage predicate indexes exist and cover the freshly written rows.

    Builds the BTREE/BITMAP indexes on the edge endpoints + artifact resolution
    keys (backlog 0181 / BUG-15) if absent, then folds any merge_insert-touched
    fragments back into coverage via ``optimize_indices``. Index support varies by
    backend, so failures degrade silently to predicate pushdown -- ``trace`` /
    ``impact`` stay correct, just unindexed. Imported locally to avoid an
    import-time ``indexing`` <-> ``lineage`` coupling.
    """
    from lancedb_robotics.indexing import build_lineage_predicate_indexes

    build_lineage_predicate_indexes(lake, replace=False)
    for table_name in ("lineage_edges", "lineage_artifacts"):
        handle = lake.table(table_name)
        try:
            handle.to_lance().optimize.optimize_indices()
        except Exception:  # noqa: BLE001 - optimize is best-effort; pushdown still correct
            pass


def _replace_rows(
    lake: Lake,
    table_name: str,
    id_column: str,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
) -> None:
    """Idempotently upsert ``rows`` keyed on ``id_column``.

    Uses batched ``merge_insert`` rather than a single delete-by-id + add so that
    (a) a full graph refresh never builds a multi-megabyte ``IN (...)`` predicate,
    and (b) the Lance mini-block encoder never receives a fragment large enough to
    overflow its chunk limit and panic (see ``_GRAPH_WRITE_BATCH_ROWS``). Rows not
    present in ``rows`` (e.g. out-of-band invalidations) are left untouched.
    """
    keyed = [row for row in rows if row.get(id_column)]
    if not keyed:
        return
    table = lake.table(table_name)
    try:
        for start in range(0, len(keyed), _GRAPH_WRITE_BATCH_ROWS):
            batch = keyed[start : start + _GRAPH_WRITE_BATCH_ROWS]
            data = pa.Table.from_pylist(batch, schema=schema)
            (
                table.merge_insert(id_column)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(data)
            )
    except Exception as exc:  # lancedb can surface a RustPanic on pathological data
        raise LineageError(
            f"failed to upsert {len(keyed)} row(s) into {table_name!r}: {exc}"
        ) from exc


# --- shared per-transform projection + inline emission (backlog 0098) -----------

_FAILED_TRANSFORM_STATUSES = frozenset(
    {"failed", "failure", "error", "aborted", "cancelled", "canceled"}
)


def _transform_produced_outputs(transform: Mapping[str, Any]) -> bool:
    """Whether a transform's status permits produced-output artifacts/edges.

    A failed/aborted transform records its execution and consumed inputs but must
    not assert produced outputs (backlog 0098 AC#2).
    """

    status = str(transform.get("status") or "").strip().lower()
    return status not in _FAILED_TRANSFORM_STATUSES


def _graph_accumulators(
    artifacts: dict[str, LineageArtifact],
    executions: dict[str, LineageExecution],
    edges: dict[str, LineageEdge],
) -> tuple[
    Callable[[LineageArtifact], str],
    Callable[[LineageExecution], str],
    Callable[..., None],
]:
    """Return id-keyed ``add_artifact``/``add_execution``/``add_edge`` closures.

    Shared by the full projection and inline emission so both accumulate graph
    rows identically.
    """

    def add_artifact(artifact: LineageArtifact) -> str:
        artifacts[artifact.artifact_id] = artifact
        return artifact.artifact_id

    def add_execution(execution: LineageExecution) -> str:
        executions[execution.execution_id] = execution
        return execution.execution_id

    def add_edge(
        edge_type: str,
        from_artifact_id: str,
        to_artifact_id: str,
        execution_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        edge = LineageEdge(
            edge_id=_edge_id(edge_type, from_artifact_id, to_artifact_id, execution_id),
            edge_type=edge_type,
            from_artifact_id=from_artifact_id,
            to_artifact_id=to_artifact_id,
            execution_id=execution_id,
            metadata=_string_metadata(metadata),
        )
        edges[edge.edge_id] = edge

    return add_artifact, add_execution, add_edge


def _project_transform_row(
    lake: Lake,
    transform: Mapping[str, Any],
    table_versions: dict[str, int],
    rows_by_table: dict[str, list[dict[str, Any]]],
    add_artifact: Callable[[LineageArtifact], str],
    add_execution: Callable[[LineageExecution], str],
    add_edge: Callable[..., None],
) -> str:
    """Project one ``transform_runs`` row into its execution + I/O artifacts + edges.

    Shared by the full projection (:func:`_compute_full_graph`) and per-write
    emission (:meth:`LakeLineage.emit_transform`) so emitted rows can never diverge
    from a later refresh. A failed/aborted transform records its execution and
    consumed inputs but no produced outputs (backlog 0098 AC#2). Returns the
    execution id.
    """

    emit_outputs = _transform_produced_outputs(transform)
    input_ids = _input_artifacts_for_transform(transform, add_artifact)
    output_ids = (
        _output_artifacts_for_transform(
            lake, transform, table_versions, rows_by_table, add_artifact
        )
        if emit_outputs
        else set()
    )
    execution_id = execution_artifact_id(str(transform["transform_id"]))
    params = _json_dict(transform.get("params"))
    lineage_context = (
        params.get("lineage_context")
        if isinstance(params.get("lineage_context"), dict)
        else {}
    )
    external_refs = (
        params.get("external_refs") if isinstance(params.get("external_refs"), dict) else {}
    )
    context_refs = (
        lineage_context.get("external_refs")
        if isinstance(lineage_context.get("external_refs"), dict)
        else {}
    )
    environment = params.get("environment") if isinstance(params.get("environment"), dict) else {}
    context_environment = (
        lineage_context.get("environment")
        if isinstance(lineage_context.get("environment"), dict)
        else {}
    )
    execution = LineageExecution(
        execution_id=execution_id,
        kind=str(transform.get("kind") or "transform"),
        name=transform.get("kind"),
        transform_id=transform.get("transform_id"),
        status=transform.get("status"),
        params=params,
        code_ref=(
            _first_present(params, "code_ref", "git_sha", "model_version")
            or _first_present(lineage_context, "code_ref")
        ),
        provider=(
            _first_present(params, "provider", "embedding_provider", "caption_provider", "source")
            or _first_present(lineage_context, "provider")
        ),
        environment={**context_environment, **environment} or None,
        input_artifact_ids=tuple(sorted(input_ids)),
        output_artifact_ids=tuple(sorted(output_ids)),
        input_table_versions=tuple(
            _version_row(row) for row in transform.get("input_table_versions") or []
        ),
        output_table_versions=(
            tuple(
                _version_row({"table": table, "version": table_versions[table], "tag": ""})
                for table in sorted(transform.get("output_tables") or [])
                if table in table_versions
            )
            if emit_outputs
            else ()
        ),
        started_at=transform.get("started_at"),
        finished_at=transform.get("finished_at"),
        created_by=transform.get("created_by"),
        metadata={
            "source_id": transform.get("source_id") or "",
            **context_refs,
            **external_refs,
        },
    )
    add_execution(execution)
    transform_artifact_id = add_artifact(_transform_artifact(dict(transform)))
    for input_id in input_ids:
        add_edge("consumed-by", input_id, transform_artifact_id, execution_id)
    if emit_outputs:
        for output_id in output_ids:
            add_edge("produced", transform_artifact_id, output_id, execution_id)
        for input_id in input_ids:
            for output_id in output_ids:
                add_edge("produced", input_id, output_id, execution_id)
    return execution_id


def _compute_full_graph(
    lake: Lake,
) -> tuple[
    dict[str, LineageArtifact],
    dict[str, LineageExecution],
    dict[str, LineageEdge],
]:
    """Derive the full lineage graph from canonical tables without writing.

    Shared by :meth:`LakeLineage._project_full_graph` (which materializes the
    result) and :meth:`LakeLineage.emission_divergence` (which compares it against
    the materialized rows). Returns id-keyed maps of artifacts, executions, edges.
    """

    artifacts: dict[str, LineageArtifact] = {}
    executions: dict[str, LineageExecution] = {}
    edges: dict[str, LineageEdge] = {}
    add_artifact, add_execution, add_edge = _graph_accumulators(artifacts, executions, edges)

    table_versions = _current_table_versions(lake)
    for table_name, version in table_versions.items():
        add_artifact(_table_artifact(table_name, version))

    rows_by_table = _canonical_rows(lake)
    for transform in rows_by_table.get("transform_runs", []):
        _project_transform_row(
            lake,
            transform,
            table_versions,
            rows_by_table,
            add_artifact,
            add_execution,
            add_edge,
        )

    _add_canonical_entity_graph(rows_by_table, table_versions, add_artifact, add_edge)
    _add_dataset_snapshot_graph(rows_by_table, table_versions, add_artifact, add_edge)
    _add_model_output_graph(rows_by_table, table_versions, add_artifact, add_edge)
    _add_feedback_graph(rows_by_table, table_versions, add_artifact, add_edge)
    _add_run_manifest_graph(rows_by_table, table_versions, add_artifact, add_execution, add_edge)
    return artifacts, executions, edges


def _graph_row_digest(row: Mapping[str, Any]) -> str:
    """Stable content digest of a graph row, ignoring the write timestamp.

    Used by :meth:`LakeLineage.emission_divergence` to detect rows whose
    materialized content differs from a fresh projection. ``created_at`` is
    excluded because it legitimately differs between an emit and a later refresh.
    """

    def _norm(value: Any) -> Any:
        if isinstance(value, datetime):
            value = value if value.tzinfo else value.replace(tzinfo=UTC)
            return value.astimezone(UTC).isoformat()
        if isinstance(value, Mapping):
            return {str(key): _norm(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            normalized = [_norm(item) for item in value]
            return sorted(
                normalized,
                key=lambda item: json.dumps(item, sort_keys=True, default=str),
            )
        return value

    payload = {key: _norm(value) for key, value in row.items() if key != "created_at"}
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def emit_transform_lineage(
    lake: Lake,
    transform: Mapping[str, Any],
    *,
    strict: bool = False,
) -> EmittedLineage | None:
    """Best-effort inline lineage emission for a canonical write path (backlog 0098).

    Write paths call this right after appending their ``transform_runs`` row. It
    never raises into the caller unless ``strict`` is set: a lineage projection
    error must never fail an ingest/curation/export, and ``refresh_graph()`` remains
    the reconciliation/backfill safety net. Returns the :class:`EmittedLineage`
    summary, or ``None`` when emission was skipped/failed.
    """

    try:
        return lake.lineage.emit_transform(transform)
    except Exception as exc:  # noqa: BLE001 - emission must not break the write path
        if strict:
            raise
        logger.warning(
            "inline lineage emission failed for transform %s: %s",
            transform.get("transform_id") if isinstance(transform, Mapping) else transform,
            exc,
        )
        return None


def _update_artifact_retention_metadata(lake: Lake, hold: LineageRetentionHold) -> None:
    rows = lake.table("lineage_artifacts").to_arrow().to_pylist()
    wanted = set(hold.artifact_ids)
    replacements: list[dict[str, Any]] = []
    for row in rows:
        if row.get("artifact_id") not in wanted:
            continue
        metadata = _metadata_map(row)
        metadata.update(
            {
                "retain_until": hold.retain_until.isoformat() if hold.retain_until else "",
                "legal_hold": str(hold.legal_hold).lower(),
                "audit_hold": str(hold.audit_hold).lower(),
                "promotion_hold": str(hold.promotion_hold).lower(),
                "owner": hold.owner or "",
                "reason": hold.reason or "",
                "retention_created_at": (
                    hold.created_at.isoformat() if hold.created_at else datetime.now(UTC).isoformat()
                ),
            }
        )
        row["metadata"] = _kv_items(metadata)
        replacements.append(row)
    if len(replacements) != len(wanted):
        missing = sorted(wanted - {row["artifact_id"] for row in replacements})
        raise LineageError(f"cannot retain unknown artifact(s): {missing}")
    _replace_rows(
        lake,
        "lineage_artifacts",
        "artifact_id",
        replacements,
        LINEAGE_ARTIFACTS_SCHEMA,
    )


def _clear_artifact_retention_metadata(lake: Lake, artifact_ids: Iterable[str]) -> None:
    wanted = {str(artifact_id) for artifact_id in artifact_ids if artifact_id}
    rows = lake.table("lineage_artifacts").to_arrow().to_pylist()
    replacements: list[dict[str, Any]] = []
    for row in rows:
        if row.get("artifact_id") not in wanted:
            continue
        metadata = _metadata_map(row)
        for key in _RETENTION_METADATA_KEYS:
            metadata.pop(key, None)
        row["metadata"] = _kv_items(metadata)
        replacements.append(row)
    if replacements:
        _replace_rows(
            lake,
            "lineage_artifacts",
            "artifact_id",
            replacements,
            LINEAGE_ARTIFACTS_SCHEMA,
        )


def _retention_hold_from_artifact_row(
    lake_uri: str,
    row: dict[str, Any],
    *,
    now: datetime | None = None,
) -> LineageRetentionHold | None:
    metadata = _metadata_map(row)
    if not any(key in metadata for key in _RETENTION_METADATA_KEYS):
        return None
    retain_until = _normalize_datetime(metadata.get("retain_until"), "retain_until")
    created_at = _normalize_datetime(metadata.get("retention_created_at"), "retention_created_at")
    legal_hold = _bool_metadata(metadata.get("legal_hold"))
    audit_hold = _bool_metadata(metadata.get("audit_hold"))
    promotion_hold = _bool_metadata(metadata.get("promotion_hold"))
    return LineageRetentionHold(
        lake_uri=lake_uri,
        artifact_ids=(str(row["artifact_id"]),),
        retain_until=_utc_datetime(retain_until),
        legal_hold=legal_hold,
        audit_hold=audit_hold,
        promotion_hold=promotion_hold,
        owner=metadata.get("owner") or None,
        reason=metadata.get("reason") or None,
        active=_retention_active(
            retain_until,
            legal_hold=legal_hold,
            audit_hold=audit_hold,
            promotion_hold=promotion_hold,
            now=now,
        ),
        created_at=_utc_datetime(created_at),
    )


def _retention_active(
    retain_until: datetime | str | None,
    *,
    legal_hold: bool,
    audit_hold: bool,
    promotion_hold: bool,
    now: datetime | None = None,
) -> bool:
    if legal_hold or audit_hold or promotion_hold:
        return True
    retain_until_dt = _normalize_datetime(retain_until, "retain_until")
    if retain_until_dt is None:
        return False
    now_dt = _utc_datetime(now) or datetime.now(UTC)
    return _utc_datetime(retain_until_dt) > now_dt


def _bool_metadata(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _current_table_versions(lake: Lake) -> dict[str, int]:
    return {table: int(lake.table(table).version) for table in _GRAPH_SOURCE_TABLES}


# --- Incremental refresh watermark (backlog 0097) -------------------------------
#
# The last successful full projection stores the source-table versions it saw as a
# sentinel row in ``lineage_artifacts`` (kept out of the graph itself: no edges, so
# traversal never reaches it, and audit/report counts exclude its kind). A later
# refresh compares the current source versions against this watermark and skips the
# whole projection when nothing changed -- the win that makes the refresh_graph()
# that trace/impact/audit/evidence call first cheap on an unchanged lake.
_REFRESH_STATE_ARTIFACT_ID = "lancedb-robotics:lineage:refresh-state"
_REFRESH_STATE_KIND = "lineage-refresh-state"


def _read_refresh_watermark(lake: Lake) -> dict[str, Any] | None:
    """Return the last-refresh watermark, or ``None`` if the graph was never refreshed."""

    found = _fetch_rows_by_id_in(
        lake, "lineage_artifacts", "artifact_id", [_REFRESH_STATE_ARTIFACT_ID]
    )
    row = found.get(_REFRESH_STATE_ARTIFACT_ID)
    if not row:
        return None
    metadata = _metadata_map(row)
    try:
        versions = json.loads(metadata.get("versions_json") or "{}")
    except (TypeError, ValueError):
        versions = {}
    return {
        "versions": {str(table): int(version) for table, version in versions.items()},
        "refreshed_at": metadata.get("refreshed_at") or None,
        "artifacts": int(metadata.get("artifacts") or 0),
        "executions": int(metadata.get("executions") or 0),
        "edges": int(metadata.get("edges") or 0),
    }


def _write_refresh_watermark(
    lake: Lake,
    source_tables: Sequence[LineageRefreshTableStatus],
    counts: tuple[int, int, int],
    now: datetime,
) -> None:
    """Persist the source-table versions + graph-row counts seen by this refresh."""

    versions = {row.table: row.current_version for row in source_tables}
    artifacts, executions, edges = counts
    sentinel = LineageArtifact(
        artifact_id=_REFRESH_STATE_ARTIFACT_ID,
        kind=_REFRESH_STATE_KIND,
        name="lineage refresh watermark",
        metadata=_string_metadata(
            {
                "schema": "lancedb-robotics/lineage-refresh-state/v1",
                "versions_json": _json_dumps(versions),
                "refreshed_at": now.isoformat(),
                "artifacts": str(artifacts),
                "executions": str(executions),
                "edges": str(edges),
            }
        ),
    )
    _replace_rows(
        lake,
        "lineage_artifacts",
        "artifact_id",
        [sentinel.as_row(now)],
        LINEAGE_ARTIFACTS_SCHEMA,
    )


def _graph_row_counts(lake: Lake) -> tuple[int, int, int]:
    """Current materialized (artifacts-excluding-sentinel, executions, edges) counts."""

    artifacts = lake.table("lineage_artifacts").count_rows()
    if _fetch_rows_by_id_in(
        lake, "lineage_artifacts", "artifact_id", [_REFRESH_STATE_ARTIFACT_ID]
    ):
        artifacts -= 1
    return (
        artifacts,
        lake.table("lineage_executions").count_rows(),
        lake.table("lineage_edges").count_rows(),
    )


def _plan_refresh(
    lake: Lake, *, incremental: bool = True, force_full: bool = False
) -> LineageRefreshPlan:
    """Compute the refresh plan against the watermark without mutating the graph."""

    current = _current_table_versions(lake)
    watermark = _read_refresh_watermark(lake)
    prior_versions = watermark["versions"] if watermark else {}
    statuses: list[LineageRefreshTableStatus] = []
    for table in sorted(current):
        previous = prior_versions.get(table)
        changed = previous is None or int(previous) != int(current[table])
        statuses.append(
            LineageRefreshTableStatus(
                table=table,
                previous_version=None if previous is None else int(previous),
                current_version=int(current[table]),
                changed=changed,
            )
        )
    changed_tables = tuple(status.table for status in statuses if status.changed)
    graph_counts = _graph_row_counts(lake)

    if force_full:
        full_scan, action, reason = True, "full-refresh", "forced full re-projection"
    elif watermark is None:
        full_scan, action, reason = (
            True,
            "initial-refresh",
            "no prior refresh watermark; projecting the full graph",
        )
    elif graph_counts[0] == 0:
        full_scan, action, reason = (
            True,
            "initial-refresh",
            "lineage graph is empty; projecting the full graph",
        )
    elif not incremental:
        full_scan, action, reason = (
            True,
            "full-refresh",
            "incremental refresh disabled; projecting the full graph",
        )
    elif changed_tables:
        full_scan, action, reason = (
            True,
            "full-refresh",
            "source tables changed ("
            + ", ".join(changed_tables)
            + "); v1 re-projects the full graph because lineage edges span tables "
            "(row-level incremental projection is a tracked follow-on)",
        )
    else:
        full_scan, action, reason = False, "skipped-unchanged", None

    return LineageRefreshPlan(
        lake_uri=lake.uri,
        action=action,
        full_scan=full_scan,
        full_scan_reason=reason,
        source_tables=tuple(statuses),
        changed_tables=changed_tables,
        artifacts=graph_counts[0],
        executions=graph_counts[1],
        edges=graph_counts[2],
        missing_indexes=_lineage_index_plan(lake)["missing"],
        previous_refreshed_at=(watermark or {}).get("refreshed_at"),
        refreshed_at=None,
        dry_run=True,
    )


# --- Missing-index plan (backlog 0097 / 0181) -----------------------------------
#
# Traversal stays correct without the lineage predicate indexes (it falls back to
# predicate pushdown), so absent indexes are a *diagnostic*, not an error: the plan
# names every missing (table, column, index type) and the build call that closes
# the gap. ``lineage_edges`` (BTREE endpoints + BITMAP edge_type) and
# ``lineage_artifacts`` (BTREE resolution keys) are the graph-traversal tables.
_LINEAGE_INDEX_TABLES = ("lineage_edges", "lineage_artifacts")


def _present_index_columns(lake: Lake, table: str) -> dict[str, str] | None:
    """Map indexed column -> index type for ``table``; ``None`` if unintrospectable."""

    try:
        handle = lake.table(table)
        infos = handle.list_indices()
    except Exception:  # noqa: BLE001 - backends without index introspection
        return None
    present: dict[str, str] = {}
    for info in infos:
        columns = getattr(info, "columns", None) or []
        if columns:
            present[str(columns[0])] = str(getattr(info, "index_type", ""))
    return present


def _lineage_index_plan(lake: Lake) -> dict[str, Any]:
    """Return the required/present/missing lineage predicate index plan."""

    from lancedb_robotics.indexing import PREDICATE_INDEX_COLUMNS_BY_TABLE

    tables: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    present_rows: list[dict[str, Any]] = []
    for table in _LINEAGE_INDEX_TABLES:
        required = PREDICATE_INDEX_COLUMNS_BY_TABLE.get(table, ())
        present = _present_index_columns(lake, table)
        if present is None:
            tables[table] = {"introspectable": False}
            continue
        table_missing: list[dict[str, Any]] = []
        table_present: list[dict[str, Any]] = []
        for column, index_type in required:
            entry = {"table": table, "column": column, "index_type": index_type}
            if column in present:
                table_present.append({**entry, "actual_type": present[column]})
                present_rows.append(table_present[-1])
            else:
                actionable = {
                    **entry,
                    "build_action": (
                        "lake.lineage.refresh_graph() or "
                        "build_lineage_predicate_indexes(lake)"
                    ),
                }
                table_missing.append(actionable)
                missing.append(actionable)
        tables[table] = {
            "introspectable": True,
            "required": [
                {"table": table, "column": column, "index_type": index_type}
                for column, index_type in required
            ],
            "present": table_present,
            "missing": table_missing,
        }
    return {
        "tables": tables,
        "missing": missing,
        "present": present_rows,
        "all_present": not missing,
    }


# --- Stale graph-row reconciliation (backlog 0097) ------------------------------


@dataclass(frozen=True)
class _StaleReconciliation:
    retired_artifacts: int = 0
    retired_edges: int = 0
    stale_artifacts: tuple[dict[str, Any], ...] = ()
    held_stale_artifacts: tuple[dict[str, Any], ...] = ()


# Whole-entity artifact kinds whose ``(row_grain, row_ids)`` are guaranteed to be
# the primary keys of a canonical source table, so their existence can be checked
# by a point lookup. Other kinds (``row-set`` groupings, ``source`` URIs,
# ``table-version`` pins, invalidation markers, the refresh-state sentinel) either
# aren't row-keyed or are durable out-of-band records, so reconciliation leaves
# them alone.
_RECONCILE_KINDS = frozenset({"row", "dataset-snapshot", "model-output"})
_STALE_SAMPLE_LIMIT = 50


def _active_hold_ids(lake: Lake) -> dict[str, dict[str, str]]:
    """Map artifact_id -> retained metadata for artifacts under an active hold.

    Captured *before* a projection so a hold survives even when the projection
    resurrects a deleted entity's artifact (dropping its metadata): reconciliation
    restores the captured hold onto the preserved row.
    """

    holds: dict[str, dict[str, str]] = {}
    for row in lake.table("lineage_artifacts").to_arrow().to_pylist():
        artifact_id = row.get("artifact_id")
        if not artifact_id:
            continue
        hold = _retention_hold_from_artifact_row(lake.uri, row)
        if hold and hold.active:
            metadata = _metadata_map(row)
            holds[str(artifact_id)] = {
                key: metadata[key] for key in _RETENTION_METADATA_KEYS if key in metadata
            }
    return holds


def _reconcile_stale_graph_rows(
    lake: Lake, *, dry_run: bool, pre_holds: Mapping[str, dict[str, str]] | None = None
) -> _StaleReconciliation:
    """Retire graph rows whose source canonical rows were deleted/superseded.

    A projection upserts current rows but never deletes rows for canonical rows
    that vanished, so a full refresh leaves orphaned artifacts/edges behind. This
    pass finds whole-entity artifacts pinned to a ``(table, primary-key)`` whose
    source row no longer exists, retires them and their incident edges, and reports
    them -- except artifacts under an active retention hold (now or captured before
    the projection), which are preserved, re-stamped with the hold, and reported
    separately (backlog 0097 acceptance: retire "safely").
    """

    pre_holds = dict(pre_holds or {})
    artifact_rows = lake.table("lineage_artifacts").to_arrow().to_pylist()
    # Group candidate whole-entity artifacts by their source table.
    by_table: dict[str, list[dict[str, Any]]] = {}
    for row in artifact_rows:
        if row.get("kind") not in _RECONCILE_KINDS:
            continue
        table = row.get("row_grain") or row.get("table_name")
        row_ids = [str(value) for value in (row.get("row_ids") or []) if value]
        if not table or table not in _ID_COLUMNS or not row_ids:
            continue
        by_table.setdefault(str(table), []).append(row)

    stale_rows: list[dict[str, Any]] = []
    for table, candidates in by_table.items():
        referenced = {
            str(value)
            for row in candidates
            for value in (row.get("row_ids") or [])
            if value
        }
        id_column = _ID_COLUMNS[table]
        existing = set(
            _fetch_rows_by_id_in(
                lake, table, id_column, referenced, columns=[id_column]
            )
        )
        for row in candidates:
            row_ids = [str(value) for value in (row.get("row_ids") or []) if value]
            # Stale only when every source row this artifact pins is gone, so a
            # partially-valid row-set is never retired out from under live rows.
            if row_ids and all(row_id not in existing for row_id in row_ids):
                stale_rows.append(row)

    retire_ids: list[str] = []
    held_samples: list[dict[str, Any]] = []
    stale_samples: list[dict[str, Any]] = []
    restore_rows: list[dict[str, Any]] = []
    for row in stale_rows:
        artifact_id = str(row["artifact_id"])
        hold = _retention_hold_from_artifact_row(lake.uri, row)
        held = (hold is not None and hold.active) or artifact_id in pre_holds
        sample = {
            "artifact_id": artifact_id,
            "kind": row.get("kind"),
            "table_name": row.get("table_name") or row.get("row_grain"),
            "row_ids": list(row.get("row_ids") or []),
        }
        if held:
            if len(held_samples) < _STALE_SAMPLE_LIMIT:
                held_samples.append({**sample, "reason": "retention hold"})
            # Re-stamp the hold the projection may have dropped when resurrecting
            # a deleted entity's tombstone artifact.
            captured = pre_holds.get(artifact_id)
            if captured:
                metadata = _metadata_map(row)
                metadata.update(captured)
                restored = dict(row)
                restored["metadata"] = _kv_items(metadata)
                restore_rows.append(restored)
            continue
        retire_ids.append(artifact_id)
        if len(stale_samples) < _STALE_SAMPLE_LIMIT:
            stale_samples.append({**sample, "reason": "source rows deleted"})

    retired_edge_ids: list[str] = []
    if retire_ids:
        incident: dict[str, dict[str, Any]] = {}
        for key in ("from_artifact_id", "to_artifact_id"):
            incident.update(
                {
                    edge["edge_id"]: edge
                    for edge in _fetch_edges_incident(
                        lake, key=key, ids=retire_ids, edge_types=set(), after=None, before=None
                    )
                }
            )
        retired_edge_ids = sorted(incident)

    if not dry_run:
        if retire_ids:
            _delete_rows_by_id(lake, "lineage_artifacts", "artifact_id", retire_ids)
            if retired_edge_ids:
                _delete_rows_by_id(lake, "lineage_edges", "edge_id", retired_edge_ids)
        if restore_rows:
            _replace_rows(
                lake,
                "lineage_artifacts",
                "artifact_id",
                restore_rows,
                LINEAGE_ARTIFACTS_SCHEMA,
            )

    return _StaleReconciliation(
        retired_artifacts=len(retire_ids),
        retired_edges=len(retired_edge_ids),
        stale_artifacts=tuple(stale_samples),
        held_stale_artifacts=tuple(held_samples),
    )


def _delete_rows_by_id(
    lake: Lake, table_name: str, id_column: str, ids: Sequence[str]
) -> None:
    """Delete rows whose ``id_column`` is in ``ids`` via chunked indexed predicates."""

    table = lake.table(table_name)
    for chunk in _chunk_values(sorted({str(value) for value in ids if value}), _LINEAGE_FRONTIER_CHUNK):
        table.delete(_sql_in_predicate(id_column, chunk))


# --- Bounded / paged traversal (backlog 0097) -----------------------------------


def _page_query_digest(root_ids: Sequence[str], direction: str) -> str:
    """Stable digest binding a page token to the traversal that minted it."""

    return _stable_digest({"roots": sorted(root_ids), "direction": direction})[:16]


def _encode_page_token(after_id: str, digest: str) -> str:
    raw = json.dumps({"after": after_id, "q": digest}, sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_page_token(token: str, digest: str) -> str:
    try:
        payload = json.loads(base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8"))
        after_id = str(payload["after"])
        token_digest = str(payload["q"])
    except (ValueError, KeyError, TypeError) as exc:
        raise LineageError(f"invalid lineage page token: {token!r}") from exc
    if token_digest != digest:
        raise LineageError(
            "lineage page token does not match this query; a continuation handle is "
            "only valid for the traversal (root + direction) that produced it"
        )
    return after_id


def _paginate_graph(
    *,
    root_ids: Sequence[str],
    resolved_handle: str,
    direction: str,
    artifacts: tuple[dict[str, Any], ...],
    edges: tuple[dict[str, Any], ...],
    executions: tuple[dict[str, Any], ...],
    page_size: int | None,
    page_token: str | None,
    controls: Mapping[str, Any],
) -> LineageGraph:
    """Return the full graph, or a stable bounded page of it, from a completed traversal.

    Pages are consecutive, non-overlapping slices of the artifacts sorted by
    ``artifact_id`` (already sorted by the caller), so total counts stay stable
    across pages and the union of pages equals the full result. A page carries the
    edges incident to its artifacts as a bounded preview.
    """

    if page_size is None:
        if page_token is not None:
            raise LineageError("page_token requires page_size")
        return LineageGraph(
            root_artifact_id=root_ids[0],
            root_artifact_ids=tuple(root_ids),
            resolved_handle=resolved_handle,
            direction=direction,
            artifacts=artifacts,
            edges=edges,
            executions=executions,
            controls=controls,
        )

    digest = _page_query_digest(root_ids, direction)
    all_ids = [row["artifact_id"] for row in artifacts]
    start = 0
    if page_token is not None:
        after_id = _decode_page_token(page_token, digest)
        start = bisect.bisect_right(all_ids, after_id)
    page_artifacts = artifacts[start : start + page_size]
    page_id_set = {row["artifact_id"] for row in page_artifacts}
    has_next = start + page_size < len(artifacts) and bool(page_artifacts)
    next_token = (
        _encode_page_token(page_artifacts[-1]["artifact_id"], digest) if has_next else None
    )
    page_edges = tuple(
        edge
        for edge in edges
        if edge["from_artifact_id"] in page_id_set or edge["to_artifact_id"] in page_id_set
    )
    page_execution_ids = {
        edge.get("execution_id") for edge in page_edges if edge.get("execution_id")
    }
    page_executions = tuple(
        execution
        for execution in executions
        if execution.get("execution_id") in page_execution_ids
    )
    return LineageGraph(
        root_artifact_id=root_ids[0],
        root_artifact_ids=tuple(root_ids),
        resolved_handle=resolved_handle,
        direction=direction,
        artifacts=page_artifacts,
        edges=page_edges,
        executions=page_executions,
        page_size=page_size,
        total_artifacts=len(artifacts),
        total_edges=len(edges),
        next_page_token=next_token,
        truncated=next_token is not None,
        controls=controls,
    )


def lineage_retention_pin_details(
    lake: Lake,
    *,
    now: datetime | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Return table-version pins implied by lineage edges and active holds."""

    artifacts = lake.table("lineage_artifacts").to_arrow().to_pylist()
    edges = lake.table("lineage_edges").to_arrow().to_pylist()
    artifacts_by_id = {row["artifact_id"]: row for row in artifacts if row.get("artifact_id")}
    referenced_ids = {
        artifact_id
        for edge in edges
        for artifact_id in (edge.get("from_artifact_id"), edge.get("to_artifact_id"))
        if artifact_id
    }
    pins: dict[str, dict[int, dict[str, Any]]] = {}
    for row in artifacts:
        if row.get("kind") != "table-version":
            continue
        artifact_id = row.get("artifact_id")
        table = row.get("table_name")
        version = row.get("table_version")
        if not artifact_id or not table or version is None:
            continue

        detail = _empty_pin_detail()
        if artifact_id in referenced_ids:
            _merge_pin_detail(detail, _reference_pin_detail(str(artifact_id), artifacts_by_id, edges))

        hold = _retention_hold_from_artifact_row(lake.uri, row, now=now)
        if hold and hold.active:
            detail["reasons"].add(_hold_reason(hold))
            detail["categories"].update(_hold_categories(hold))
            detail["artifact_ids"].add(str(artifact_id))
            detail["holds"].append(hold.as_dict())

        if detail["reasons"] or detail["holds"]:
            pins.setdefault(str(table), {}).setdefault(int(version), _empty_pin_detail())
            _merge_pin_detail(pins[str(table)][int(version)], detail)
    return pins


def snapshot_retention_pin_details(lake: Lake) -> dict[str, dict[int, dict[str, Any]]]:
    """Return dataset-snapshot table-version pins independent of graph refresh."""

    pins: dict[str, dict[int, dict[str, Any]]] = {}
    for row in lake.table("dataset_snapshots").to_arrow().to_pylist():
        snapshot_id = snapshot_artifact_id(str(row.get("dataset_id") or ""))
        label = row.get("name") or row.get("dataset_id") or "dataset-snapshot"
        for version in row.get("table_versions") or []:
            table = version.get("table")
            if not table or version.get("version") is None:
                continue
            detail = _empty_pin_detail()
            detail["reasons"].add(f"dataset-snapshot:{label}")
            detail["categories"].update({"dataset-snapshot", "training-reproducibility"})
            detail["artifact_ids"].add(snapshot_id)
            pins.setdefault(str(table), {}).setdefault(int(version["version"]), _empty_pin_detail())
            _merge_pin_detail(pins[str(table)][int(version["version"])], detail)
    return pins


def merge_retention_pin_details(
    *pin_maps: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[int, dict[str, Any]]]:
    merged: dict[str, dict[int, dict[str, Any]]] = {}
    for pin_map in pin_maps:
        for table, versions in pin_map.items():
            for version, detail in versions.items():
                merged.setdefault(table, {}).setdefault(version, _empty_pin_detail())
                _merge_pin_detail(merged[table][version], detail)
    return merged


def retention_pin_rows(pin_map: dict[str, dict[int, dict[str, Any]]]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for table, versions in sorted(pin_map.items()):
        for version, detail in sorted(versions.items()):
            rows.append(
                {
                    "table": table,
                    "version": int(version),
                    "categories": sorted(detail["categories"]),
                    "reasons": sorted(detail["reasons"]),
                    "artifact_ids": sorted(detail["artifact_ids"]),
                    "holds": sorted(
                        detail["holds"],
                        key=lambda hold: (hold.get("artifact_ids") or [""], hold.get("reason") or ""),
                    ),
                }
            )
    return tuple(rows)


def _empty_pin_detail() -> dict[str, Any]:
    return {
        "reasons": set(),
        "categories": set(),
        "artifact_ids": set(),
        "holds": [],
    }


def _merge_pin_detail(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["reasons"].update(source.get("reasons") or ())
    target["categories"].update(source.get("categories") or ())
    target["artifact_ids"].update(source.get("artifact_ids") or ())
    existing = {_json_dumps(hold) for hold in target["holds"]}
    for hold in source.get("holds") or ():
        encoded = _json_dumps(hold)
        if encoded not in existing:
            target["holds"].append(hold)
            existing.add(encoded)


def _reference_pin_detail(
    artifact_id: str,
    artifacts_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    detail = _empty_pin_detail()
    for edge in edges:
        from_id = edge.get("from_artifact_id")
        to_id = edge.get("to_artifact_id")
        if artifact_id not in {from_id, to_id}:
            continue
        other_id = to_id if from_id == artifact_id else from_id
        other = artifacts_by_id.get(str(other_id), {})
        other_label = other.get("kind") or other.get("name") or other_id or "unknown"
        detail["reasons"].add(f"{edge.get('edge_type') or 'lineage'}:{other_label}")
        detail["categories"].update(_reference_categories(edge, other))
        if other_id:
            detail["artifact_ids"].add(str(other_id))
    if not detail["reasons"]:
        detail["reasons"].add("lineage-reference")
        detail["categories"].add("lineage-reference")
    return detail


def _reference_categories(edge: dict[str, Any], other: dict[str, Any]) -> set[str]:
    kind = str(other.get("kind") or "").lower()
    edge_type = str(edge.get("edge_type") or "").lower()
    metadata = _metadata_map(edge)
    categories = {"lineage-reference"}
    if kind in {"dataset-snapshot", "training-run", "model", "evaluation-run", "projection-manifest"}:
        categories.add("training-reproducibility")
    if kind == "dataset-snapshot":
        categories.add("dataset-snapshot")
    if kind in {"evidence-pack", "invalidation"} or "evidence" in kind:
        categories.add("audit-evidence")
    if "audit" in kind or "audit" in metadata.get("reason", "").lower():
        categories.add("audit-evidence")
    if "external" in kind or "openlineage" in kind or "datahub" in kind:
        categories.update({"external-lineage", "audit-evidence"})
    if edge_type in {"invalidates", "superseded-by"}:
        categories.add("audit-evidence")
    return categories


def _hold_reason(hold: LineageRetentionHold) -> str:
    if hold.reason:
        return f"retention-hold:{hold.reason}"
    if hold.legal_hold:
        return "retention-hold:legal"
    if hold.audit_hold:
        return "retention-hold:audit"
    if hold.promotion_hold:
        return "retention-hold:promotion"
    if hold.retain_until:
        return f"retention-hold:until:{hold.retain_until.isoformat()}"
    return "retention-hold"


def _hold_categories(hold: LineageRetentionHold) -> set[str]:
    categories = {"retention-hold"}
    reason = (hold.reason or "").lower()
    if hold.legal_hold or hold.audit_hold or "audit" in reason or "legal" in reason:
        categories.add("audit-evidence")
    if hold.promotion_hold or "deploy" in reason or "promotion" in reason:
        categories.add("training-reproducibility")
    return categories


def _canonical_rows(lake: Lake) -> dict[str, list[dict[str, Any]]]:
    return {table: lake.table(table).to_arrow().to_pylist() for table in _GRAPH_SOURCE_TABLES}


def _table_artifact(table_name: str, version: int, tag: str | None = None) -> LineageArtifact:
    return LineageArtifact(
        artifact_id=table_version_artifact_id(table_name, version, tag),
        kind="table-version",
        name=f"{table_name}@{tag or version}",
        table_name=table_name,
        table_version=int(version),
        table_tag=tag or "",
    )


def _row_artifact(
    table_name: str,
    row_id: str,
    *,
    table_version: int | None = None,
    producer_execution_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LineageArtifact:
    row_id = str(row_id)
    return LineageArtifact(
        artifact_id=artifact_id(
            "row",
            table_name=table_name,
            row_id=row_id,
            table_version=table_version,
        ),
        kind="row",
        name=f"{table_name}:{row_id}",
        table_name=table_name,
        table_version=table_version,
        row_grain=table_name,
        row_ids=(row_id,),
        producer_execution_id=producer_execution_id,
        metadata=_string_metadata(metadata),
    )


def _row_set_artifact(
    table_name: str,
    row_ids: Iterable[str],
    *,
    table_version: int | None = None,
    name: str | None = None,
    producer_execution_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LineageArtifact | None:
    normalized = tuple(sorted(str(row_id) for row_id in row_ids if row_id))
    if not normalized:
        return None
    return LineageArtifact(
        artifact_id=artifact_id(
            "row-set",
            table_name=table_name,
            row_ids=normalized,
            table_version=table_version,
        ),
        kind="row-set",
        name=name or f"{table_name}:{len(normalized)} rows",
        table_name=table_name,
        table_version=table_version,
        row_grain=table_name,
        row_ids=normalized,
        digest=_stable_digest({"table_name": table_name, "row_ids": normalized}),
        producer_execution_id=producer_execution_id,
        metadata=_string_metadata(metadata),
    )


def _transform_artifact(transform: dict[str, Any]) -> LineageArtifact:
    transform_id = str(transform["transform_id"])
    params = _json_dict(transform.get("params"))
    return LineageArtifact(
        artifact_id=execution_artifact_id(transform_id),
        kind="transform",
        name=transform_id,
        table_name="transform_runs",
        row_grain="transform_runs",
        row_ids=(transform_id,),
        metadata=_string_metadata(
            {
                "kind": transform.get("kind"),
                "status": transform.get("status"),
                "source_id": transform.get("source_id"),
                "dataset_id": params.get("dataset_id") or params.get("source_snapshot_id"),
            }
        ),
    )


def _snapshot_artifact(snapshot: dict[str, Any]) -> LineageArtifact:
    execution_id = execution_artifact_id(snapshot["transform_id"]) if snapshot.get("transform_id") else None
    return LineageArtifact(
        artifact_id=snapshot_artifact_id(str(snapshot["dataset_id"])),
        kind="dataset-snapshot",
        name=snapshot.get("name"),
        table_name="dataset_snapshots",
        row_grain="dataset_snapshots",
        row_ids=(str(snapshot["dataset_id"]),),
        producer_execution_id=execution_id,
        metadata=_string_metadata({"tag": snapshot.get("tag"), "kind": snapshot.get("kind")}),
    )


def _source_artifact(
    *,
    source_uri: str,
    source_id: str | None = None,
    channel: str | None = None,
    offset: int | None = None,
    log_time_ns: int | None = None,
) -> LineageArtifact:
    return LineageArtifact(
        artifact_id=artifact_id(
            "source",
            source_id=source_id,
            source_uri=source_uri if source_id is None else None,
            channel=channel,
            offset=offset,
            log_time_ns=log_time_ns,
        ),
        kind="source",
        name=source_uri,
        source_uri=source_uri,
        source_id=source_id,
        metadata=_string_metadata({"channel": channel, "offset": offset, "log_time_ns": log_time_ns}),
    )


def _input_artifacts_for_transform(
    transform: dict[str, Any],
    add_artifact: Callable[[LineageArtifact], str],
) -> set[str]:
    artifact_ids: set[str] = set()
    params = _json_dict(transform.get("params"))
    dataset_candidates = [params.get("dataset_id"), params.get("source_snapshot_id")]
    if str(transform.get("kind") or "") in {"projection", "dataset-export"} or str(
        transform.get("source_id") or ""
    ).startswith("ds-"):
        dataset_candidates.append(transform.get("source_id"))
    for dataset_id in dataset_candidates:
        if dataset_id:
            artifact_ids.add(
                add_artifact(
                    LineageArtifact(
                        artifact_id=snapshot_artifact_id(str(dataset_id)),
                        kind="dataset-snapshot",
                        name=str(dataset_id),
                        table_name="dataset_snapshots",
                        row_grain="dataset_snapshots",
                        row_ids=(str(dataset_id),),
                    )
                )
            )
    for uri in transform.get("input_uris") or []:
        artifact_ids.add(add_artifact(_source_artifact(source_uri=str(uri))))
    for item in transform.get("input_table_versions") or []:
        version = _version_row(item)
        if version["table"]:
            artifact_ids.add(
                add_artifact(
                    _table_artifact(version["table"], version["version"], version.get("tag") or None)
                )
            )
    return artifact_ids


def _output_artifacts_for_transform(
    lake: Lake,
    transform: dict[str, Any],
    table_versions: dict[str, int],
    rows_by_table: dict[str, list[dict[str, Any]]],
    add_artifact: Callable[[LineageArtifact], str],
) -> set[str]:
    artifact_ids: set[str] = set()
    execution_id = execution_artifact_id(str(transform["transform_id"]))
    output_tables = [table for table in transform.get("output_tables") or [] if table in table_versions]
    for table in output_tables:
        artifact_ids.add(add_artifact(_table_artifact(table, table_versions[table])))

    params = _json_dict(transform.get("params"))
    if params.get("dataset_id"):
        snapshot = _row_by_id(rows_by_table, "dataset_snapshots", "dataset_id", str(params["dataset_id"]))
        if snapshot:
            artifact_ids.add(add_artifact(_snapshot_artifact(snapshot)))
    if str(transform.get("kind") or "") == "projection":
        manifest_paths = tuple(str(path) for path in params.get("output_paths") or [] if path)
        manifest_path = next(
            (path for path in manifest_paths if path.endswith("projection_manifest.json")),
            manifest_paths[-1] if manifest_paths else None,
        )
        artifact_ids.add(
            add_artifact(
                LineageArtifact(
                    artifact_id=artifact_id(
                        "projection-manifest",
                        transform_id=transform.get("transform_id"),
                        source_snapshot_id=params.get("source_snapshot_id"),
                        format=params.get("format"),
                        mode=params.get("mode"),
                    ),
                    kind="projection-manifest",
                    name=":".join(
                        str(part)
                        for part in (
                            params.get("format"),
                            params.get("mode"),
                            params.get("snapshot_name"),
                        )
                        if part
                    )
                    or str(transform.get("transform_id")),
                    source_uri=manifest_path,
                    digest=(params.get("content_hashes") or {}).get("dataset"),
                    producer_execution_id=execution_artifact_id(str(transform["transform_id"])),
                    metadata=_string_metadata(
                        {
                            "source_snapshot_id": params.get("source_snapshot_id"),
                            "format": params.get("format"),
                            "mode": params.get("mode"),
                            "output_paths": manifest_paths,
                        }
                    ),
                )
            )
        )

    for search_index in _search_index_artifacts_for_transform(transform, params):
        artifact_ids.add(add_artifact(search_index))

    scenario_ids = params.get("scenario_ids") or []
    if scenario_ids and "scenarios" in table_versions:
        row_set = _row_set_artifact(
            "scenarios",
            scenario_ids,
            table_version=table_versions["scenarios"],
            name=f"scenario-windowing:{len(scenario_ids)} scenarios",
            producer_execution_id=execution_id,
        )
        if row_set:
            artifact_ids.add(add_artifact(row_set))

    row_ids = params.get("row_ids") or []
    for table in output_tables:
        id_column = _ID_COLUMNS.get(table)
        if not id_column or not row_ids:
            continue
        row_set = _row_set_artifact(
            table,
            row_ids,
            table_version=int(lake.table(table).version),
            name=f"{table}:{len(row_ids)} rows",
            producer_execution_id=execution_id,
        )
        if row_set:
            artifact_ids.add(add_artifact(row_set))
    return artifact_ids


def _search_index_artifacts_for_transform(
    transform: dict[str, Any],
    params: dict[str, Any],
) -> tuple[LineageArtifact, ...]:
    artifacts: list[LineageArtifact] = []
    payloads: list[dict[str, Any]] = []
    for key in ("index", "fts_index"):
        payload = params.get(key)
        if isinstance(payload, dict):
            payloads.append(payload | {"source_param": key})
    if params.get("index_type") and params.get("table") and params.get("column"):
        payloads.append(params | {"source_param": "index"})

    for payload in payloads:
        table_name = payload.get("table")
        column = payload.get("column")
        if not table_name or not column:
            continue
        index_type = payload.get("index_type") or payload.get("type") or "INDEX"
        provider = (
            params.get("embedding_provider")
            or params.get("caption_provider")
            or params.get("provider")
            or ""
        )
        provider_version = (
            params.get("embedding_provider_version")
            or params.get("caption_provider_version")
            or params.get("provider_version")
            or ""
        )
        artifacts.append(
            LineageArtifact(
                artifact_id=artifact_id(
                    "search-index",
                    table_name=table_name,
                    column=column,
                    index_type=index_type,
                    provider=provider,
                    provider_version=provider_version,
                    transform_id=transform.get("transform_id"),
                ),
                kind="search-index",
                name=f"{table_name}.{column}:{index_type}",
                table_name=str(table_name),
                producer_execution_id=execution_artifact_id(str(transform["transform_id"])),
                metadata=_string_metadata(
                    {
                        **payload,
                        "provider": provider,
                        "provider_version": provider_version,
                        "transform_id": transform.get("transform_id"),
                    }
                ),
            )
        )
    return tuple(artifacts)


def _add_canonical_entity_graph(
    rows_by_table: dict[str, list[dict[str, Any]]],
    table_versions: dict[str, int],
    add_artifact: Callable[[LineageArtifact], str],
    add_edge: Callable[[str, str, str, str | None, dict[str, Any] | None], None],
) -> None:
    runs: dict[str, str] = {}
    observations: dict[str, str] = {}
    scenarios: dict[str, str] = {}

    for run in rows_by_table.get("runs", []):
        run_id = add_artifact(
            _row_artifact(
                "runs",
                run["run_id"],
                table_version=table_versions.get("runs"),
                producer_execution_id=(
                    execution_artifact_id(run["transform_id"]) if run.get("transform_id") else None
                ),
                metadata={
                    "raw_uri": run.get("raw_uri"),
                    "source_id": run.get("source_id"),
                    "task_id": run.get("task_id"),
                },
            )
        )
        runs[str(run["run_id"])] = run_id
        if run.get("raw_uri"):
            source_id = add_artifact(
                _source_artifact(
                    source_uri=str(run["raw_uri"]),
                    source_id=run.get("source_id") or run.get("run_id"),
                )
            )
            add_edge(
                "ingested-from",
                source_id,
                run_id,
                execution_artifact_id(run["transform_id"]) if run.get("transform_id") else None,
            )

    for observation in rows_by_table.get("observations", []):
        observation_id = add_artifact(
            _row_artifact(
                "observations",
                observation["observation_id"],
                table_version=table_versions.get("observations"),
                producer_execution_id=(
                    execution_artifact_id(observation["transform_id"])
                    if observation.get("transform_id")
                    else None
                ),
                metadata={
                    "run_id": observation.get("run_id"),
                    "topic": observation.get("topic"),
                    "raw_uri": observation.get("raw_uri"),
                },
            )
        )
        observations[str(observation["observation_id"])] = observation_id
        run_id = runs.get(str(observation.get("run_id")))
        if run_id:
            add_edge(
                "contains-observation",
                run_id,
                observation_id,
                execution_artifact_id(observation["transform_id"])
                if observation.get("transform_id")
                else None,
            )
        if observation.get("raw_uri"):
            source_id = add_artifact(
                _source_artifact(
                    source_uri=str(observation["raw_uri"]),
                    source_id=observation.get("run_id"),
                    channel=observation.get("raw_channel") or observation.get("topic"),
                    offset=observation.get("raw_sequence"),
                    log_time_ns=observation.get("raw_log_time_ns"),
                )
            )
            add_edge(
                "source-coordinate",
                source_id,
                observation_id,
                execution_artifact_id(observation["transform_id"])
                if observation.get("transform_id")
                else None,
            )

    for episode in rows_by_table.get("episodes", []):
        episode_id = add_artifact(
            _row_artifact(
                "episodes",
                episode["episode_id"],
                table_version=table_versions.get("episodes"),
                producer_execution_id=(
                    execution_artifact_id(episode["transform_id"])
                    if episode.get("transform_id")
                    else None
                ),
                metadata={"run_id": episode.get("run_id"), "task_id": episode.get("task_id")},
            )
        )
        run_id = runs.get(str(episode.get("run_id")))
        if run_id:
            add_edge(
                "contains-episode",
                run_id,
                episode_id,
                execution_artifact_id(episode["transform_id"]) if episode.get("transform_id") else None,
            )

    for scenario in rows_by_table.get("scenarios", []):
        scenario_id = add_artifact(
            _row_artifact(
                "scenarios",
                scenario["scenario_id"],
                table_version=table_versions.get("scenarios"),
                producer_execution_id=(
                    execution_artifact_id(scenario["transform_id"])
                    if scenario.get("transform_id")
                    else None
                ),
                metadata={"run_id": scenario.get("run_id"), "task_id": scenario.get("task_id")},
            )
        )
        scenarios[str(scenario["scenario_id"])] = scenario_id
        run_id = runs.get(str(scenario.get("run_id")))
        if run_id:
            add_edge(
                "windowed-run",
                run_id,
                scenario_id,
                execution_artifact_id(scenario["transform_id"])
                if scenario.get("transform_id")
                else None,
            )
        for observation_key in scenario.get("observation_ids") or []:
            observation_id = observations.get(str(observation_key))
            if observation_id:
                add_edge(
                    "windowed-into",
                    observation_id,
                    scenario_id,
                    execution_artifact_id(scenario["transform_id"])
                    if scenario.get("transform_id")
                    else None,
                )

    alignment_jobs = {
        str(row["alignment_id"]): add_artifact(
            _row_artifact(
                "alignment_jobs",
                row["alignment_id"],
                table_version=table_versions.get("alignment_jobs"),
                producer_execution_id=(
                    execution_artifact_id(row["transform_id"]) if row.get("transform_id") else None
                ),
            )
        )
        for row in rows_by_table.get("alignment_jobs", [])
    }
    for aligned in rows_by_table.get("aligned_frames", []):
        aligned_id = add_artifact(
            _row_artifact(
                "aligned_frames",
                aligned["aligned_frame_id"],
                table_version=table_versions.get("aligned_frames"),
                producer_execution_id=(
                    execution_artifact_id(aligned["transform_id"])
                    if aligned.get("transform_id")
                    else None
                ),
                metadata={"run_id": aligned.get("run_id"), "stream": aligned.get("stream")},
            )
        )
        job_id = alignment_jobs.get(str(aligned.get("alignment_id")))
        if job_id:
            add_edge(
                "alignment-output",
                job_id,
                aligned_id,
                execution_artifact_id(aligned["transform_id"])
                if aligned.get("transform_id")
                else None,
            )
        for observation_key in (
            aligned.get("observation_id"),
            *(aligned.get("source_observation_ids") or ()),
        ):
            observation_id = observations.get(str(observation_key))
            if observation_id:
                add_edge(
                    "aligned-from",
                    observation_id,
                    aligned_id,
                    execution_artifact_id(aligned["transform_id"])
                    if aligned.get("transform_id")
                    else None,
                )
    for tick in rows_by_table.get("aligned_ticks", []):
        tick_id = add_artifact(
            _row_artifact(
                "aligned_ticks",
                tick["aligned_tick_id"],
                table_version=table_versions.get("aligned_ticks"),
                producer_execution_id=(
                    execution_artifact_id(tick["transform_id"])
                    if tick.get("transform_id")
                    else None
                ),
                metadata={
                    "run_id": tick.get("run_id"),
                    "tick_index": tick.get("tick_index"),
                    "storage_backend": "aligned_ticks-jsonb",
                },
            )
        )
        job_id = alignment_jobs.get(str(tick.get("alignment_id")))
        if job_id:
            add_edge(
                "alignment-output",
                job_id,
                tick_id,
                execution_artifact_id(tick["transform_id"])
                if tick.get("transform_id")
                else None,
            )
        lineage = _json_dict(tick.get("lineage_json"))
        source_ids = lineage.get("source_observation_ids") or {}
        if isinstance(source_ids, Mapping):
            observation_keys = {
                str(observation_key)
                for values in source_ids.values()
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
                for observation_key in values
            }
            for observation_key in observation_keys:
                observation_id = observations.get(observation_key)
                if observation_id:
                    add_edge(
                        "aligned-from",
                        observation_id,
                        tick_id,
                        execution_artifact_id(tick["transform_id"])
                        if tick.get("transform_id")
                        else None,
                    )


def _add_dataset_snapshot_graph(
    rows_by_table: dict[str, list[dict[str, Any]]],
    table_versions: dict[str, int],
    add_artifact: Callable[[LineageArtifact], str],
    add_edge: Callable[[str, str, str, str | None, dict[str, Any] | None], None],
) -> None:
    scenarios_by_id = {row["scenario_id"]: row for row in rows_by_table.get("scenarios", [])}
    for snapshot in rows_by_table.get("dataset_snapshots", []):
        snapshot_id = add_artifact(_snapshot_artifact(snapshot))
        execution_id = execution_artifact_id(snapshot["transform_id"]) if snapshot.get("transform_id") else None
        for item in snapshot.get("table_versions") or []:
            version = _version_row(item)
            if not version["table"]:
                continue
            table_id = add_artifact(
                _table_artifact(version["table"], version["version"], version.get("tag") or None)
            )
            add_edge("version-pinned", table_id, snapshot_id, execution_id, {"table": version["table"]})

        scenario_ids = _snapshot_scenario_ids(snapshot)
        scenario_version = _snapshot_table_version(snapshot, "scenarios") or table_versions.get("scenarios")
        row_set = _row_set_artifact(
            "scenarios",
            scenario_ids,
            table_version=scenario_version,
            name=f"{snapshot.get('name') or snapshot['dataset_id']} scenarios",
            producer_execution_id=execution_id,
        )
        if row_set:
            row_set_id = add_artifact(row_set)
            add_edge("selected-from", row_set_id, snapshot_id, execution_id)
        for scenario_id in scenario_ids:
            scenario_artifact_id = add_artifact(
                _row_artifact("scenarios", scenario_id, table_version=scenario_version)
            )
            add_edge("selected-from", scenario_artifact_id, snapshot_id, execution_id)
            scenario = scenarios_by_id.get(scenario_id)
            if scenario and scenario.get("run_id"):
                run_artifact_id = add_artifact(
                    _row_artifact("runs", scenario["run_id"], table_version=table_versions.get("runs"))
                )
                add_edge(
                    "windowed-run",
                    run_artifact_id,
                    scenario_artifact_id,
                    execution_artifact_id(scenario["transform_id"])
                    if scenario.get("transform_id")
                    else None,
                )

        for source in _source_artifacts_for_snapshot(snapshot, rows_by_table):
            source_id = add_artifact(source)
            add_edge("source-coordinate", source_id, snapshot_id, execution_id)


def _add_model_output_graph(
    rows_by_table: dict[str, list[dict[str, Any]]],
    table_versions: dict[str, int],
    add_artifact: Callable[[LineageArtifact], str],
    add_edge: Callable[[str, str, str, str | None, dict[str, Any] | None], None],
) -> None:
    version = table_versions.get("model_outputs")
    for row in rows_by_table.get("model_outputs", []):
        output_id = add_artifact(
            _row_artifact(
                "model_outputs",
                row["model_output_id"],
                table_version=version,
                producer_execution_id=(
                    execution_artifact_id(row["producer_run_id"]) if row.get("producer_run_id") else None
                ),
                metadata={"model_version": row.get("model_version")},
            )
        )
        if row.get("dataset_id"):
            snapshot_id = add_artifact(
                LineageArtifact(
                    artifact_id=snapshot_artifact_id(str(row["dataset_id"])),
                    kind="dataset-snapshot",
                    name=str(row["dataset_id"]),
                    table_name="dataset_snapshots",
                    row_grain="dataset_snapshots",
                    row_ids=(str(row["dataset_id"]),),
                )
            )
            add_edge("evaluated-on", snapshot_id, output_id, row.get("producer_run_id"))
        if row.get("model_version") or row.get("producer_run_id"):
            model_id = add_artifact(
                LineageArtifact(
                    artifact_id=artifact_id(
                        "model",
                        model_version=row.get("model_version"),
                        producer_run_id=row.get("producer_run_id"),
                    ),
                    kind="model",
                    name=row.get("model_version") or row.get("producer_run_id"),
                    metadata=_string_metadata({"producer_run_id": row.get("producer_run_id")}),
                )
            )
            add_edge("produced-output", model_id, output_id, row.get("producer_run_id"))
        _add_target_row_edges(row, output_id, table_versions, add_artifact, add_edge, "scored-row")


def _add_feedback_graph(
    rows_by_table: dict[str, list[dict[str, Any]]],
    table_versions: dict[str, int],
    add_artifact: Callable[[LineageArtifact], str],
    add_edge: Callable[[str, str, str, str | None, dict[str, Any] | None], None],
) -> None:
    version = table_versions.get("feedback")
    for row in rows_by_table.get("feedback", []):
        feedback_id = add_artifact(
            _row_artifact(
                "feedback",
                row["feedback_id"],
                table_version=version,
                metadata={"feedback_type": row.get("feedback_type"), "severity": row.get("severity")},
            )
        )
        if row.get("model_output_id"):
            model_output_id = add_artifact(
                _row_artifact(
                    "model_outputs",
                    row["model_output_id"],
                    table_version=table_versions.get("model_outputs"),
                )
            )
            add_edge("feedback-on", model_output_id, feedback_id, row.get("transform_id"))
        if row.get("label_id"):
            label_id = add_artifact(
                _row_artifact("labels", row["label_id"], table_version=table_versions.get("labels"))
            )
            add_edge("feedback-on", label_id, feedback_id, row.get("transform_id"))
        _add_target_row_edges(row, feedback_id, table_versions, add_artifact, add_edge, "feedback-target")


def _add_run_manifest_graph(
    rows_by_table: dict[str, list[dict[str, Any]]],
    table_versions: dict[str, int],
    add_artifact: Callable[[LineageArtifact], str],
    add_execution: Callable[[LineageExecution], str],
    add_edge: Callable[[str, str, str, str | None, dict[str, Any] | None], None],
) -> None:
    training_version = table_versions.get("training_runs")
    artifact_version = table_versions.get("model_artifacts")
    eval_version = table_versions.get("evaluation_runs")

    for row in rows_by_table.get("training_runs", []):
        training_id = add_artifact(
            LineageArtifact(
                artifact_id=training_run_artifact_id(row["training_run_id"]),
                kind="training-run",
                name=row["training_run_id"],
                table_name="training_runs",
                table_version=training_version,
                row_grain="training_runs",
                row_ids=(row["training_run_id"],),
                digest=row.get("manifest_digest"),
                producer_execution_id=execution_artifact_id(row["training_run_id"]),
                metadata=_string_metadata(
                    {
                        "dataset_id": row.get("dataset_id"),
                        "snapshot_name": row.get("snapshot_name"),
                        "snapshot_tag": row.get("snapshot_tag"),
                        "row_plan_id": row.get("row_plan_id"),
                        "epoch_plan_id": row.get("epoch_plan_id"),
                        "status": row.get("status"),
                    }
                ),
            )
        )
        snapshot_id = add_artifact(
            LineageArtifact(
                artifact_id=snapshot_artifact_id(row["dataset_id"]),
                kind="dataset-snapshot",
                name=row.get("snapshot_name"),
                table_name="dataset_snapshots",
                row_grain="dataset_snapshots",
                row_ids=(row["dataset_id"],),
                metadata=_string_metadata({"tag": row.get("snapshot_tag")}),
            )
        )
        execution_id = add_execution(
            LineageExecution(
                execution_id=execution_artifact_id(row["training_run_id"]),
                kind="training-run",
                name=row["training_run_id"],
                transform_id=row.get("transform_id"),
                status=row.get("status"),
                params={
                    "hyperparameters": _json_dict(row.get("hyperparameters_json")),
                    "random_seeds": _json_dict(row.get("random_seeds_json")),
                    "split_policy": _json_dict(row.get("split_policy_json")),
                },
                code_ref=row.get("code_ref"),
                environment_json=_json_dumps(
                    {
                        "packages": _json_dict(row.get("package_versions_json")),
                        "environment": _json_dict(row.get("environment_json")),
                        "hardware": _json_dict(row.get("hardware_json")),
                        "runtime": _json_dict(row.get("runtime_json")),
                    }
                ),
                input_artifact_ids=(snapshot_id,),
                output_artifact_ids=(training_id,),
                input_table_versions=tuple(_version_row(item) for item in row.get("table_versions") or []),
                created_by=row.get("created_by"),
                metadata=_metadata_map(row, column="external_refs")
                | {
                    "row_plan_id": row.get("row_plan_id") or "",
                    "epoch_plan_id": row.get("epoch_plan_id") or "",
                },
            )
        )
        add_edge("trained-on", snapshot_id, training_id, execution_id)
        for item in row.get("table_versions") or []:
            version = _version_row(item)
            if not version["table"]:
                continue
            table_id = add_artifact(
                _table_artifact(version["table"], version["version"], version.get("tag") or None)
            )
            add_edge("version-pinned", table_id, training_id, execution_id, {"table": version["table"]})

    for row in rows_by_table.get("model_artifacts", []):
        model_id = add_artifact(
            LineageArtifact(
                artifact_id=model_artifact_lineage_id(row["model_artifact_id"]),
                kind="model",
                name=(row.get("aliases") or [row["model_artifact_id"]])[0],
                table_name="model_artifacts",
                table_version=artifact_version,
                row_grain="model_artifacts",
                row_ids=(row["model_artifact_id"],),
                source_uri=row.get("artifact_uri"),
                digest=row.get("checksum") or row.get("manifest_digest"),
                producer_execution_id=execution_artifact_id(row["training_run_id"]),
                metadata=_string_metadata(
                    {
                        "training_run_id": row.get("training_run_id"),
                        "aliases": row.get("aliases") or [],
                        "framework": row.get("framework"),
                        "epoch": row.get("epoch"),
                        "step": row.get("step"),
                    }
                    | _metadata_map(row, column="external_refs")
                ),
            )
        )
        training_id = add_artifact(
            LineageArtifact(
                artifact_id=training_run_artifact_id(row["training_run_id"]),
                kind="training-run",
                name=row["training_run_id"],
                table_name="training_runs",
                row_grain="training_runs",
                row_ids=(row["training_run_id"],),
            )
        )
        add_edge(
            "produced-model",
            training_id,
            model_id,
            execution_artifact_id(row["training_run_id"]),
        )

    for row in rows_by_table.get("evaluation_runs", []):
        eval_id = add_artifact(
            LineageArtifact(
                artifact_id=evaluation_run_artifact_id(row["eval_run_id"]),
                kind="evaluation-run",
                name=row["eval_run_id"],
                table_name="evaluation_runs",
                table_version=eval_version,
                row_grain="evaluation_runs",
                row_ids=(row["eval_run_id"],),
                digest=row.get("manifest_digest"),
                producer_execution_id=execution_artifact_id(row["eval_run_id"]),
                metadata=_string_metadata(
                    {
                        "model_artifact_id": row.get("model_artifact_id"),
                        "dataset_id": row.get("dataset_id"),
                        "snapshot_name": row.get("snapshot_name"),
                        "status": row.get("status"),
                    }
                ),
            )
        )
        model_id = add_artifact(
            LineageArtifact(
                artifact_id=model_artifact_lineage_id(row["model_artifact_id"]),
                kind="model",
                name=row["model_artifact_id"],
                table_name="model_artifacts",
                row_grain="model_artifacts",
                row_ids=(row["model_artifact_id"],),
            )
        )
        snapshot_id = add_artifact(
            LineageArtifact(
                artifact_id=snapshot_artifact_id(row["dataset_id"]),
                kind="dataset-snapshot",
                name=row.get("snapshot_name"),
                table_name="dataset_snapshots",
                row_grain="dataset_snapshots",
                row_ids=(row["dataset_id"],),
                metadata=_string_metadata({"tag": row.get("snapshot_tag")}),
            )
        )
        execution_id = add_execution(
            LineageExecution(
                execution_id=execution_artifact_id(row["eval_run_id"]),
                kind="evaluation-run",
                name=row["eval_run_id"],
                transform_id=row.get("transform_id"),
                status=row.get("status"),
                params={
                    "metrics": _json_dict(row.get("metrics_json")),
                    "slice_metrics": _json_dict(row.get("slice_metrics_json")),
                    "failure_outputs": _json_dict(row.get("failure_outputs_json")),
                },
                code_ref=row.get("code_ref"),
                environment_json=_json_dumps(
                    {
                        "packages": _json_dict(row.get("package_versions_json")),
                        "environment": _json_dict(row.get("environment_json")),
                        "hardware": _json_dict(row.get("hardware_json")),
                        "runtime": _json_dict(row.get("runtime_json")),
                    }
                ),
                input_artifact_ids=(model_id, snapshot_id),
                output_artifact_ids=(eval_id,),
                input_table_versions=tuple(_version_row(item) for item in row.get("table_versions") or []),
                created_by=row.get("created_by"),
                metadata=_metadata_map(row, column="external_refs"),
            )
        )
        add_edge("evaluated-model", model_id, eval_id, execution_id)
        add_edge("evaluated-on", snapshot_id, eval_id, execution_id)
        for item in row.get("table_versions") or []:
            version = _version_row(item)
            if not version["table"]:
                continue
            table_id = add_artifact(
                _table_artifact(version["table"], version["version"], version.get("tag") or None)
            )
            add_edge("version-pinned", table_id, eval_id, execution_id, {"table": version["table"]})


def _add_target_row_edges(
    row: dict[str, Any],
    target_artifact_id: str,
    table_versions: dict[str, int],
    add_artifact: Callable[[LineageArtifact], str],
    add_edge: Callable[[str, str, str, str | None, dict[str, Any] | None], None],
    edge_type: str,
) -> None:
    for table_name, id_column in (
        ("runs", "run_id"),
        ("observations", "observation_id"),
        ("scenarios", "scenario_id"),
        ("events", "event_id"),
    ):
        if not row.get(id_column):
            continue
        source_id = add_artifact(
            _row_artifact(table_name, row[id_column], table_version=table_versions.get(table_name))
        )
        add_edge(edge_type, source_id, target_artifact_id, row.get("transform_id"))


def _snapshot_scenario_ids(snapshot: dict[str, Any]) -> list[str]:
    query_spec = _json_dict(snapshot.get("query_spec"))
    return sorted(str(value) for value in query_spec.get("scenario_ids") or [] if value)


def _snapshot_table_version(snapshot: dict[str, Any], table_name: str) -> int | None:
    for item in snapshot.get("table_versions") or []:
        if item.get("table") == table_name and item.get("version") is not None:
            return int(item["version"])
    return None


def _source_artifacts_for_snapshot(
    snapshot: dict[str, Any],
    rows_by_table: dict[str, list[dict[str, Any]]],
) -> list[LineageArtifact]:
    scenarios = {row["scenario_id"]: row for row in rows_by_table.get("scenarios", [])}
    observations = {row["observation_id"]: row for row in rows_by_table.get("observations", [])}
    sources: dict[str, LineageArtifact] = {}
    for scenario_id in _snapshot_scenario_ids(snapshot):
        scenario = scenarios.get(scenario_id)
        if not scenario:
            continue
        for observation_id in scenario.get("observation_ids") or []:
            observation = observations.get(observation_id)
            if not observation or not observation.get("raw_uri"):
                continue
            source = _source_artifact(
                source_uri=str(observation["raw_uri"]),
                source_id=observation.get("run_id"),
                channel=observation.get("raw_channel") or observation.get("topic"),
                offset=observation.get("raw_sequence"),
                log_time_ns=observation.get("raw_log_time_ns"),
            )
            sources[source.artifact_id] = source
    return [sources[key] for key in sorted(sources)]


def _row_by_id(
    rows_by_table: dict[str, list[dict[str, Any]]],
    table_name: str,
    id_column: str,
    row_id: str,
) -> dict[str, Any] | None:
    for row in rows_by_table.get(table_name, []):
        if row.get(id_column) == row_id:
            return row
    return None


def _json_dict(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _first_present(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is not None and value != "":
            return str(value)
    return None


# Frontier ids per indexed `IN` lookup. Kept small so each `WHERE endpoint IN (...)`
# stays a cheap indexed `IsIn` (lance BTREE `pages_in` is O(list length), and a giant
# `IN` literal blows up the planner -- the BUG-06 lesson). The visited subgraph is
# read level-by-level in chunks of this size.
_LINEAGE_FRONTIER_CHUNK = 512
_EDGE_PROJECTION = (
    "edge_id",
    "edge_type",
    "from_artifact_id",
    "to_artifact_id",
    "execution_id",
    "created_at",
)


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in_predicate(column: str, values: Sequence[str]) -> str:
    unique = tuple(dict.fromkeys(str(value) for value in values if str(value)))
    if not unique:
        return "FALSE"
    if len(unique) == 1:
        return f"{column} = {_sql_literal(unique[0])}"
    return f"{column} IN ({', '.join(_sql_literal(value) for value in unique)})"


def _chunk_values(values: Iterable[str], size: int) -> Iterable[list[str]]:
    items = list(values)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _fetch_edges_incident(
    lake: Lake,
    *,
    key: str,
    ids: Iterable[str],
    edge_types: set[str],
    after: datetime | None,
    before: datetime | None,
) -> list[dict[str, Any]]:
    """Edges whose ``key`` endpoint is in ``ids``, via chunked indexed ``IN`` reads.

    The read seam behind the BUG-13 fix: each BFS level fetches only the edges
    incident to the current frontier, pushing ``endpoint IN (chunk)`` (BUG-15 BTREE
    on the endpoint) and ``edge_type IN (...)`` (BITMAP) into the scan, streaming
    via ``to_batches``. Work is proportional to the visited subgraph, never the
    whole edge table. ``created_at`` range is applied while streaming. Uses the
    remote-safe lancedb query API (not ``.to_lance()``), so it works over db://.
    """
    distinct = sorted({str(item) for item in ids if item is not None and str(item)})
    if not distinct:
        return []
    table = lake.table("lineage_edges")
    edge_clause = _sql_in_predicate("edge_type", sorted(edge_types)) if edge_types else None
    found: dict[str, dict[str, Any]] = {}
    for chunk in _chunk_values(distinct, _LINEAGE_FRONTIER_CHUNK):
        where = _sql_in_predicate(key, chunk)
        if edge_clause:
            where = f"({where}) AND ({edge_clause})"
        query = table.search().select(list(_EDGE_PROJECTION)).where(where)
        for batch in query.to_batches(batch_size=4096):
            for row in batch.to_pylist():
                if (after is not None or before is not None) and not _created_in_range(
                    row.get("created_at"), after, before
                ):
                    continue
                found[row["edge_id"]] = row
    return list(found.values())


def _fetch_rows_by_id_in(
    lake: Lake,
    table_name: str,
    id_column: str,
    ids: Iterable[str],
    *,
    columns: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch ``{id: row}`` for rows whose ``id_column`` is in ``ids`` (chunked indexed ``IN``).

    Reads only the referenced rows -- the on-demand artifact/execution fetch behind
    the BUG-13 frontier expansion, replacing whole-table ``to_pylist`` loads.
    """
    distinct = sorted({str(item) for item in ids if item is not None and str(item)})
    if not distinct:
        return {}
    table = lake.table(table_name)
    found: dict[str, dict[str, Any]] = {}
    for chunk in _chunk_values(distinct, _LINEAGE_FRONTIER_CHUNK):
        query = table.search()
        if columns:
            query = query.select(list(columns))
        query = query.where(_sql_in_predicate(id_column, chunk))
        for batch in query.to_batches(batch_size=4096):
            for row in batch.to_pylist():
                found[str(row[id_column])] = row
    return found


def _traverse_graph(
    lake: Lake,
    artifact: str,
    *,
    direction: str,
    kind: str | None,
    max_depth: int | None,
    edge_types: set[str],
    target_kinds: set[str],
    created_after: datetime | str | None,
    created_before: datetime | str | None,
    table_versions: Mapping[str, int] | Iterable[str | tuple[str, int]],
    page_size: int | None = None,
    page_token: str | None = None,
) -> LineageGraph:
    if max_depth is not None and max_depth < 0:
        raise LineageError("max_depth must be >= 0")
    if page_size is not None and page_size < 1:
        raise LineageError("page_size must be >= 1")
    root_ids = _resolve_artifact_ids(lake, artifact, kind=kind)
    normalized_target_kinds = {_normalize_handle_kind(item) or str(item) for item in target_kinds}
    after = _normalize_datetime(created_after, "created_after")
    before = _normalize_datetime(created_before, "created_before")
    version_filters = _normalize_table_version_filters(table_versions)

    # BUG-13 frontier expansion. The whole 1.29M-edge / 704K-artifact tables used to
    # be loaded and indexed in memory on every call (O(total edges), ~22s on a real
    # graph, OOM at didi scale). Instead, expand the graph one BFS level at a time,
    # fetching only the edges incident to the current frontier via a chunked indexed
    # ``endpoint IN (frontier)`` read (BUG-15 BTREE), and fetch artifacts/executions
    # on demand by id. Work is proportional to the *visited* subgraph.
    source_key = "to_artifact_id" if direction == "upstream" else "from_artifact_id"
    next_key = "from_artifact_id" if direction == "upstream" else "to_artifact_id"

    artifact_cache: dict[str, dict[str, Any]] = {}

    def ensure_artifacts(ids: Iterable[str]) -> None:
        missing = [
            str(item)
            for item in ids
            if item is not None and str(item) and str(item) not in artifact_cache
        ]
        if missing:
            artifact_cache.update(
                _fetch_rows_by_id_in(lake, "lineage_artifacts", "artifact_id", missing)
            )

    def expandable(artifact_id: str) -> bool:
        # Mirror the original per-node guard: a node absent from lineage_artifacts is
        # still traversable; a present node must satisfy the version/time filters.
        row = artifact_cache.get(artifact_id)
        return row is None or _artifact_matches_filters(row, version_filters, after, before)

    ensure_artifacts(root_ids)
    seen_artifacts: set[str] = set(root_ids)
    seen_edges: dict[str, dict[str, Any]] = {}
    parents: dict[str, set[tuple[str, str]]] = {}
    frontier: list[str] = list(root_ids)
    depth = 0
    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        sources = [artifact_id for artifact_id in frontier if expandable(artifact_id)]
        if not sources:
            break
        # One indexed, chunked, streaming read per level for the incident edges.
        level_edges = _fetch_edges_incident(
            lake,
            key=source_key,
            ids=sources,
            edge_types=edge_types,
            after=after,
            before=before,
        )
        level_edges.sort(
            key=lambda row: (
                row["edge_type"],
                row["from_artifact_id"],
                row["to_artifact_id"],
                row["edge_id"],
            )
        )
        ensure_artifacts(edge[next_key] for edge in level_edges)
        next_frontier: list[str] = []
        queued: set[str] = set()
        for edge in level_edges:
            next_id = edge[next_key]
            if next_id not in artifact_cache:
                continue
            if not _artifact_matches_filters(
                artifact_cache[next_id], version_filters, after, before
            ):
                continue
            seen_edges[edge["edge_id"]] = edge
            parents.setdefault(next_id, set()).add((edge[source_key], edge["edge_id"]))
            if next_id in seen_artifacts:
                continue
            seen_artifacts.add(next_id)
            if next_id not in queued:
                queued.add(next_id)
                next_frontier.append(next_id)
        frontier = next_frontier
        depth += 1

    if normalized_target_kinds:
        kept_artifacts, kept_edges = _target_paths(
            root_ids,
            seen_artifacts,
            seen_edges,
            parents,
            artifact_cache,
            normalized_target_kinds,
        )
    else:
        kept_artifacts = seen_artifacts
        kept_edges = set(seen_edges)

    execution_ids = {
        seen_edges[edge_id].get("execution_id")
        for edge_id in kept_edges
        if seen_edges[edge_id].get("execution_id")
    }
    executions = _fetch_rows_by_id_in(
        lake, "lineage_executions", "execution_id", [eid for eid in execution_ids if eid]
    )
    full_artifacts = tuple(
        artifact_cache[item] for item in sorted(kept_artifacts) if item in artifact_cache
    )
    full_edges = tuple(seen_edges[item] for item in sorted(kept_edges))
    full_executions = tuple(
        executions[item] for item in sorted(execution_ids) if item in executions
    )
    return _paginate_graph(
        root_ids=root_ids,
        resolved_handle=str(artifact),
        direction=direction,
        artifacts=full_artifacts,
        edges=full_edges,
        executions=full_executions,
        page_size=page_size,
        page_token=page_token,
        controls=_lineage_traversal_controls(
            artifact=str(artifact),
            kind=kind,
            direction=direction,
            max_depth=max_depth,
            edge_types=edge_types,
            target_kinds=normalized_target_kinds,
            created_after=after,
            created_before=before,
            table_versions=version_filters,
            page_size=page_size,
            page_token=page_token,
        ),
    )


def _lineage_traversal_controls(
    *,
    artifact: str,
    kind: str | None,
    direction: str,
    max_depth: int | None,
    edge_types: Iterable[str],
    target_kinds: Iterable[str],
    created_after: datetime | None,
    created_before: datetime | None,
    table_versions: Mapping[str, int],
    page_size: int | None,
    page_token: str | None,
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "kind": kind,
        "direction": direction,
        "max_depth": max_depth,
        "edge_types": sorted(str(item) for item in edge_types),
        "target_kinds": sorted(str(item) for item in target_kinds),
        "created_after": created_after.isoformat() if created_after else None,
        "created_before": created_before.isoformat() if created_before else None,
        "table_versions": {
            str(table): int(version) for table, version in sorted(table_versions.items())
        },
        "page_size": page_size,
        "page_token": page_token,
    }


def _audit_lineage(
    lake: Lake,
    *,
    subject: str | None,
    root_artifact_ids: Sequence[str],
    check_sources: bool,
    check_remote_sources: bool,
    source_auth_ref: str | None,
    source_storage_options: Mapping[str, Any] | None,
    page_size: int | None,
    page_token: str | None,
    now: datetime | None,
    refreshed: bool,
) -> LineageAuditReport:
    if page_size is not None and page_size < 1:
        raise LineageError("page_size must be >= 1")
    artifacts = {
        row["artifact_id"]: row
        for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
        if row.get("artifact_id") and row.get("kind") != _REFRESH_STATE_KIND
    }
    edges = lake.table("lineage_edges").to_arrow().to_pylist()
    scoped_artifact_ids, scoped_edges = _audit_scope(artifacts, edges, root_artifact_ids)
    scoped_artifacts = {
        artifact_id: artifacts[artifact_id]
        for artifact_id in scoped_artifact_ids
        if artifact_id in artifacts
    }
    retention_pins = merge_retention_pin_details(
        snapshot_retention_pin_details(lake),
        lineage_retention_pin_details(lake, now=now),
    )
    missing_sources, source_validator_statuses = _source_validation_results(
        scoped_artifacts,
        check_sources=check_sources,
        check_remote_sources=check_remote_sources,
        storage_options=source_storage_options,
        auth_ref=source_auth_ref,
    )
    findings = {
        "unresolved_references": _unresolved_edge_references(scoped_edges, artifacts),
        "missing_sources": missing_sources,
        "missing_table_versions": _missing_table_version_rows(lake, scoped_artifacts),
        "stale_external_links": _stale_external_link_rows(scoped_artifacts, artifacts),
        "retained_versions": retention_pin_rows(retention_pins),
        "retention_holds": _retention_hold_rows(lake.uri, artifacts.values(), now=now),
        "cleanup_candidates": _cleanup_candidate_rows(lake, retention_pins),
    }
    summary_counts = {category: len(rows) for category, rows in findings.items()}
    status = _audit_status(findings)
    validator_statuses = tuple(
        sorted(
            (
                *source_validator_statuses,
                _external_link_validator_status(findings["stale_external_links"]),
            ),
            key=lambda row: str(row.get("validator") or ""),
        )
    )
    paged_findings, next_token, total_findings, returned_findings, truncated = (
        _paginate_audit_findings(findings, page_size=page_size, page_token=page_token)
    )
    return LineageAuditReport(
        lake_uri=lake.uri,
        subject=subject,
        root_artifact_ids=tuple(root_artifact_ids),
        artifact_count=len(scoped_artifacts),
        edge_count=len(scoped_edges),
        unresolved_references=paged_findings["unresolved_references"],
        missing_sources=paged_findings["missing_sources"],
        missing_table_versions=paged_findings["missing_table_versions"],
        stale_external_links=paged_findings["stale_external_links"],
        retained_versions=paged_findings["retained_versions"],
        retention_holds=paged_findings["retention_holds"],
        cleanup_candidates=paged_findings["cleanup_candidates"],
        refreshed=refreshed,
        status=status,
        generated_at=now or datetime.now(UTC),
        summary_counts=summary_counts,
        validator_statuses=validator_statuses,
        page_size=page_size,
        total_findings=total_findings if page_size is not None else None,
        returned_findings=returned_findings if page_size is not None else None,
        next_page_token=next_token,
        truncated=truncated,
    )


def _audit_scope(
    artifacts: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    root_artifact_ids: Sequence[str],
) -> tuple[set[str], tuple[dict[str, Any], ...]]:
    if not root_artifact_ids:
        return set(artifacts), tuple(edges)
    scoped = set(root_artifact_ids)
    scoped_edges: dict[str, dict[str, Any]] = {}
    frontier = list(root_artifact_ids)
    while frontier:
        current = frontier.pop(0)
        for edge in edges:
            endpoints = {edge.get("from_artifact_id"), edge.get("to_artifact_id")}
            if current not in endpoints:
                continue
            scoped_edges[str(edge["edge_id"])] = edge
            for endpoint in endpoints:
                if endpoint in artifacts and endpoint not in scoped:
                    scoped.add(str(endpoint))
                    frontier.append(str(endpoint))
    return scoped, tuple(scoped_edges[key] for key in sorted(scoped_edges))


def _unresolved_edge_references(
    edges: Sequence[dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    unresolved: list[dict[str, Any]] = []
    for edge in sorted(edges, key=lambda row: (row["edge_type"], row["edge_id"])):
        for endpoint in ("from_artifact_id", "to_artifact_id"):
            artifact_id = edge.get(endpoint)
            if artifact_id and artifact_id not in artifacts:
                unresolved.append(
                    {
                        "edge_id": edge["edge_id"],
                        "edge_type": edge.get("edge_type"),
                        "endpoint": endpoint,
                        "missing_artifact_id": artifact_id,
                    }
                )
    return tuple(unresolved)


_AUDIT_FINDING_ORDER = (
    "unresolved_references",
    "missing_sources",
    "missing_table_versions",
    "stale_external_links",
    "retained_versions",
    "retention_holds",
    "cleanup_candidates",
)

_AUDIT_BLOCKING_FINDINGS = frozenset(
    {
        "unresolved_references",
        "missing_sources",
        "missing_table_versions",
        "stale_external_links",
    }
)


def _audit_status(findings: Mapping[str, Sequence[dict[str, Any]]]) -> str:
    return (
        "failed"
        if any(findings.get(category) for category in _AUDIT_BLOCKING_FINDINGS)
        else "passed"
    )


def _paginate_audit_findings(
    findings: Mapping[str, Sequence[dict[str, Any]]],
    *,
    page_size: int | None,
    page_token: str | None,
) -> tuple[dict[str, tuple[dict[str, Any], ...]], str | None, int, int, bool]:
    if page_size is None and page_token is not None:
        raise LineageError("page_token requires page_size")

    flattened: list[tuple[str, dict[str, Any]]] = []
    for category in _AUDIT_FINDING_ORDER:
        for row in findings.get(category) or ():
            flattened.append((category, dict(row)))
    total = len(flattened)

    if page_size is None:
        return (
            {
                category: tuple(dict(row) for row in findings.get(category) or ())
                for category in _AUDIT_FINDING_ORDER
            },
            None,
            total,
            total,
            False,
        )

    digest = _stable_digest(
        {
            category: [dict(row) for row in findings.get(category) or ()]
            for category in _AUDIT_FINDING_ORDER
        }
    )
    offset = _decode_audit_page_token(page_token, digest) if page_token else 0
    window = flattened[offset : offset + page_size]
    next_offset = offset + page_size
    next_token = (
        _encode_audit_page_token(next_offset, digest) if next_offset < total else None
    )
    grouped: dict[str, list[dict[str, Any]]] = {category: [] for category in _AUDIT_FINDING_ORDER}
    for category, row in window:
        grouped[category].append(row)
    return (
        {category: tuple(rows) for category, rows in grouped.items()},
        next_token,
        total,
        len(window),
        next_token is not None,
    )


def _encode_audit_page_token(offset: int, digest: str) -> str:
    return base64.urlsafe_b64encode(
        json.dumps({"offset": int(offset), "digest": digest}, sort_keys=True).encode()
    ).decode()


def _decode_audit_page_token(page_token: str, digest: str) -> int:
    try:
        decoded = json.loads(base64.urlsafe_b64decode(page_token.encode()).decode())
        if decoded.get("digest") != digest:
            raise LineageError(
                "audit page_token does not match the current finding set; rerun from the first page"
            )
        return max(0, int(decoded["offset"]))
    except LineageError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise LineageError(f"invalid audit page_token {page_token!r}") from exc


def _source_validation_results(
    artifacts: Mapping[str, dict[str, Any]],
    *,
    check_sources: bool,
    check_remote_sources: bool,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if not check_sources:
        return (), (
            {
                "validator": "source-existence",
                "status": "skipped",
                "reason": "source checks disabled",
            },
        )

    local_missing: list[dict[str, Any]] = []
    remote_missing: list[dict[str, Any]] = []
    remote_errors: list[str] = []
    local_count = 0
    remote_count = 0

    for artifact_id, row in sorted(artifacts.items()):
        source_uri = row.get("source_uri")
        if row.get("kind") != "source" or not source_uri:
            continue
        uri = str(source_uri)
        if _is_remote_source_uri(uri):
            remote_count += 1
            if check_remote_sources:
                try:
                    if not _remote_source_exists(
                        uri,
                        storage_options=storage_options,
                        auth_ref=auth_ref,
                    ):
                        remote_missing.append(
                            {
                                "artifact_id": artifact_id,
                                "source_uri": uri,
                                "reason": "remote source does not exist",
                            }
                        )
                except Exception as exc:  # noqa: BLE001 - validator diagnostics are report data
                    remote_errors.append(f"{uri}: {exc}")
            continue
        local_count += 1
        if _local_source_missing(uri):
            local_missing.append(
                {
                    "artifact_id": artifact_id,
                    "source_uri": uri,
                    "reason": "local source does not exist",
                }
            )

    statuses = [
        {
            "validator": "local-source-existence",
            "status": "failed" if local_missing else "passed",
            "source_count": local_count,
            "finding_count": len(local_missing),
        }
    ]
    if remote_count:
        if not check_remote_sources:
            statuses.append(
                {
                    "validator": "object-store-source-existence",
                    "status": "skipped",
                    "source_count": remote_count,
                    "reason": "remote validators are opt-in",
                    "install_hint": "pass check_remote_sources=True and install lancedb-robotics[object-store]",
                }
            )
        elif remote_errors:
            statuses.append(
                {
                    "validator": "object-store-source-existence",
                    "status": "unavailable",
                    "source_count": remote_count,
                    "finding_count": len(remote_missing),
                    "reason": "; ".join(remote_errors[:3]),
                    "install_hint": "install lancedb-robotics[object-store] and configure storage credentials",
                }
            )
        else:
            statuses.append(
                {
                    "validator": "object-store-source-existence",
                    "status": "failed" if remote_missing else "passed",
                    "source_count": remote_count,
                    "finding_count": len(remote_missing),
                }
            )
    return tuple([*local_missing, *remote_missing]), tuple(statuses)


def _external_link_validator_status(
    stale_external_links: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "validator": "external-link-freshness",
        "status": "failed" if stale_external_links else "passed",
        "finding_count": len(stale_external_links),
        "reason": (
            "deterministic reversible URN checks only; live metadata-system validation is optional plugin work"
            if not stale_external_links
            else "one or more reversible external URNs no longer resolve to graph artifacts"
        ),
    }


def _is_remote_source_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme.lower() not in {"", "file"}


def _remote_source_exists(
    uri: str,
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> bool:
    from lancedb_robotics.storage import uri_exists

    return uri_exists(uri, storage_options=storage_options, auth_ref=auth_ref)


def _missing_source_rows(
    artifacts: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for artifact_id, row in sorted(artifacts.items()):
        source_uri = row.get("source_uri")
        if row.get("kind") == "source" and source_uri and _local_source_missing(str(source_uri)):
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "source_uri": source_uri,
                    "reason": "local source does not exist",
                }
            )
    return tuple(rows)


def _local_source_missing(uri: str) -> bool:
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        return False
    path = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(uri)
    return not path.exists()


def _missing_table_version_rows(
    lake: Lake,
    artifacts: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for artifact_id, row in sorted(artifacts.items()):
        if row.get("kind") != "table-version":
            continue
        table = row.get("table_name")
        version = row.get("table_version")
        if not table or version is None:
            continue
        if not _table_version_readable(lake, str(table), int(version)):
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "table": str(table),
                    "version": int(version),
                    "reason": "table version is not readable",
                }
            )
    return tuple(rows)


def _table_version_readable(lake: Lake, table_name: str, version: int) -> bool:
    try:
        table = lake.table(table_name)
    except Exception:
        return False
    try:
        table.checkout(int(version))
        table.to_arrow()
        return True
    except Exception:
        return False
    finally:
        try:
            table.checkout_latest()
        except Exception:
            pass


def _stale_external_link_rows(
    scoped_artifacts: Mapping[str, dict[str, Any]],
    artifacts: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    stale: list[dict[str, Any]] = []
    for artifact_id, row in sorted(scoped_artifacts.items()):
        metadata = _metadata_map(row)
        for key, value in sorted(metadata.items()):
            if not _looks_like_external_urn(key, value):
                continue
            resolved_id = _artifact_id_from_external_link(value)
            if resolved_id is not None and resolved_id not in artifacts:
                stale.append(
                    {
                        "artifact_id": artifact_id,
                        "metadata_key": key,
                        "external_ref": value,
                        "missing_artifact_id": resolved_id,
                    }
                )
    return tuple(stale)


def _looks_like_external_urn(key: str, value: str) -> bool:
    lowered_key = key.lower()
    lowered_value = str(value).lower()
    return (
        "urn" in lowered_key
        or lowered_value.startswith("urn:lancedb-robotics:")
        or lowered_value.startswith("urn:li:dataset:")
    )


def _artifact_id_from_external_link(value: str) -> str | None:
    from lancedb_robotics.lineage_integrations import (
        LineageIntegrationError,
        artifact_id_from_external_urn,
    )

    try:
        return artifact_id_from_external_urn(value)
    except LineageIntegrationError:
        return None


def _retention_hold_rows(
    lake_uri: str,
    artifacts: Iterable[dict[str, Any]],
    *,
    now: datetime | None,
) -> tuple[dict[str, Any], ...]:
    holds = [
        hold.as_dict()
        for row in artifacts
        if (hold := _retention_hold_from_artifact_row(lake_uri, row, now=now)) is not None
    ]
    return tuple(
        sorted(
            holds,
            key=lambda hold: (hold.get("artifact_ids") or [""], hold.get("reason") or ""),
        )
    )


def _cleanup_candidate_rows(
    lake: Lake,
    retained_versions: dict[str, dict[int, dict[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    candidates: list[dict[str, Any]] = []
    for table_name in lake.table_names():
        try:
            dataset = lake.table(table_name).to_lance()
            current_version = int(dataset.version)
            versions = dataset.versions()
        except Exception:
            continue
        pinned = set(retained_versions.get(table_name, {}))
        for item in versions:
            version = int(item.get("version") or 0)
            if version == current_version or version in pinned:
                continue
            candidates.append(
                {
                    "table": table_name,
                    "version": version,
                    "timestamp": (
                        item.get("timestamp").isoformat()
                        if hasattr(item.get("timestamp"), "isoformat")
                        else item.get("timestamp")
                    ),
                    "reason": "not current and not retained by snapshot, lineage, or active hold",
                }
            )
    return tuple(
        sorted(candidates, key=lambda row: (row["table"], row["version"]))
    )


def _record_invalidation(
    lake: Lake,
    target_artifact_ids: Iterable[str],
    *,
    reason: str,
    severity: str,
    discovered_by: str | None,
    actor: str | None,
    replacement_artifact_id: str | None,
) -> LineageInvalidation:
    normalized_targets = tuple(sorted(str(value) for value in target_artifact_ids if value))
    if not normalized_targets:
        raise LineageError("invalidation requires at least one target artifact")
    if not reason:
        raise LineageError("invalidation reason is required")
    now = datetime.now(UTC)
    invalidation_id = "inv-" + _stable_digest(
        {
            "target_artifact_ids": normalized_targets,
            "reason": reason,
            "severity": severity,
            "discovered_by": discovered_by,
            "actor": actor,
            "replacement_artifact_id": replacement_artifact_id,
            "created_at": now.isoformat(),
        }
    )
    invalidation_artifact_id = f"lancedb-robotics:invalidation:{invalidation_id}"
    artifact = LineageArtifact(
        artifact_id=invalidation_artifact_id,
        kind="invalidation",
        name=invalidation_id,
        row_ids=(invalidation_id,),
        metadata=_string_metadata(
            {
                "reason": reason,
                "severity": severity,
                "discovered_by": discovered_by,
                "actor": actor,
                "target_artifact_ids": normalized_targets,
                "replacement_artifact_id": replacement_artifact_id,
            }
        ),
    )
    _replace_rows(
        lake,
        "lineage_artifacts",
        "artifact_id",
        [artifact.as_row(now)],
        LINEAGE_ARTIFACTS_SCHEMA,
    )
    edges: list[dict[str, Any]] = []
    for target_id in normalized_targets:
        edges.append(
            LineageEdge(
                edge_id=_edge_id("invalidates", target_id, invalidation_artifact_id),
                edge_type="invalidates",
                from_artifact_id=target_id,
                to_artifact_id=invalidation_artifact_id,
                metadata={
                    "reason": reason,
                    "severity": severity,
                    "discovered_by": discovered_by or "",
                    "actor": actor or "",
                },
            ).as_row(now)
        )
        if replacement_artifact_id:
            edges.append(
                LineageEdge(
                    edge_id=_edge_id("superseded-by", target_id, replacement_artifact_id),
                    edge_type="superseded-by",
                    from_artifact_id=target_id,
                    to_artifact_id=replacement_artifact_id,
                    metadata={"invalidation_id": invalidation_id},
                ).as_row(now)
            )
    _replace_rows(lake, "lineage_edges", "edge_id", edges, LINEAGE_EDGES_SCHEMA)
    return LineageInvalidation(
        lake_uri=lake.uri,
        invalidation_id=invalidation_id,
        invalidation_artifact_id=invalidation_artifact_id,
        target_artifact_ids=normalized_targets,
        reason=reason,
        severity=severity,
        discovered_by=discovered_by,
        actor=actor,
        replacement_artifact_id=replacement_artifact_id,
        created_at=now,
    )


def _provider_handle(
    *,
    provider: str | None,
    provider_version: str | None,
    embedding_column: str | None,
) -> str:
    parts = []
    if provider:
        parts.append(f"provider={provider}")
    if provider_version:
        parts.append(f"provider_version={provider_version}")
    if embedding_column:
        parts.append(f"embedding_column={embedding_column}")
    return ",".join(parts) or "provider"


def _provider_rebuild_roots(
    lake: Lake,
    *,
    provider: str | None,
    provider_version: str | None,
    embedding_column: str | None,
) -> tuple[str, ...]:
    if not (provider or provider_version or embedding_column):
        raise LineageError("rebuild_plan requires an artifact handle or provider filter")
    artifacts = {
        row["artifact_id"]
        for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
        if row.get("artifact_id")
    }
    roots: set[str] = set()
    for row in lake.table("transform_runs").to_arrow().to_pylist():
        params = _json_dict(row.get("params"))
        if not _provider_params_match(
            params,
            provider=provider,
            provider_version=provider_version,
            embedding_column=embedding_column,
        ):
            continue
        artifact_id = execution_artifact_id(str(row["transform_id"]))
        if artifact_id in artifacts:
            roots.add(artifact_id)
    if not roots:
        raise LineageError(f"no provider-backed lineage roots matched {_provider_handle(provider=provider, provider_version=provider_version, embedding_column=embedding_column)!r}")
    return tuple(sorted(roots))


def _provider_params_match(
    params: dict[str, Any],
    *,
    provider: str | None,
    provider_version: str | None,
    embedding_column: str | None,
) -> bool:
    if provider and provider not in _nested_param_values(
        params,
        {"provider", "embedding_provider", "caption_provider"},
    ):
        return False
    if provider_version and provider_version not in _nested_param_values(
        params,
        {
            "provider_version",
            "embedding_provider_version",
            "caption_provider_version",
            "version",
        },
    ):
        return False
    if embedding_column and embedding_column not in _nested_param_values(
        params,
        {"column", "embedding_column"},
    ):
        return False
    return True


def _nested_param_values(payload: Any, keys: set[str]) -> set[str]:
    values: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                if key in keys and inner is not None:
                    values.add(str(inner))
                visit(inner)
        elif isinstance(value, (list, tuple)):
            for inner in value:
                visit(inner)

    visit(payload)
    return values


def _impact_graph_for_roots(
    lake: Lake,
    root_ids: Sequence[str],
    *,
    max_depth: int | None,
    resolved_handle: str,
) -> LineageGraph:
    if not root_ids:
        raise LineageError("rebuild plan requires at least one root artifact")
    artifacts: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    executions: dict[str, dict[str, Any]] = {}
    for root_id in root_ids:
        graph = _traverse_graph(
            lake,
            root_id,
            direction="downstream",
            kind=None,
            max_depth=max_depth,
            edge_types=set(),
            target_kinds=set(),
            created_after=None,
            created_before=None,
            table_versions={},
        )
        artifacts.update({row["artifact_id"]: row for row in graph.artifacts})
        edges.update({row["edge_id"]: row for row in graph.edges})
        executions.update({row["execution_id"]: row for row in graph.executions})
    return LineageGraph(
        root_artifact_id=str(root_ids[0]),
        root_artifact_ids=tuple(root_ids),
        resolved_handle=resolved_handle,
        direction="downstream",
        artifacts=tuple(artifacts[key] for key in sorted(artifacts)),
        edges=tuple(
            edges[key]
            for key in sorted(
                edges,
                key=lambda edge_id: (
                    edges[edge_id]["edge_type"],
                    edges[edge_id]["from_artifact_id"],
                    edges[edge_id]["to_artifact_id"],
                    edge_id,
                ),
            )
        ),
        executions=tuple(executions[key] for key in sorted(executions)),
        controls=_lineage_traversal_controls(
            artifact=resolved_handle,
            kind=None,
            direction="downstream",
            max_depth=max_depth,
            edge_types=(),
            target_kinds=(),
            created_after=None,
            created_before=None,
            table_versions={},
            page_size=None,
            page_token=None,
        ),
    )


def _rebuild_actions_for_graph(
    graph: LineageGraph,
    *,
    reason: str | None,
    policy: ActionPolicy = DEFAULT_ACTION_POLICY,
    severity: str | None = None,
) -> tuple[RebuildPlanAction, ...]:
    artifacts = {row["artifact_id"]: row for row in graph.artifacts}
    edges = [row for row in graph.edges if row["from_artifact_id"] in artifacts and row["to_artifact_id"] in artifacts]
    plan_edges = [edge for edge in edges if _is_plan_dependency_edge(edge)]
    order = _topological_artifact_order(
        artifacts,
        plan_edges,
        roots=graph.root_artifact_ids or (graph.root_artifact_id,),
    )
    # Precompute the incoming plan-edge types and dependency parents per artifact
    # in one pass so per-artifact policy resolution stays O(edges), not O(V*E).
    incoming_edge_types: dict[str, set[str]] = {}
    dependency_parents: dict[str, set[str]] = {}
    for edge in plan_edges:
        upstream = edge["from_artifact_id"]
        downstream = edge["to_artifact_id"]
        if upstream == downstream:
            continue
        incoming_edge_types.setdefault(downstream, set()).add(str(edge.get("edge_type") or ""))
        dependency_parents.setdefault(downstream, set()).add(upstream)
    actions: list[RebuildPlanAction] = []
    for artifact_id in order:
        row = artifacts[artifact_id]
        dependencies = tuple(sorted(dependency_parents.get(artifact_id, set())))
        metadata = _metadata_map(row)
        context = ActionContext(
            artifact_id=artifact_id,
            kind=str(row.get("kind") or ""),
            table_name=row.get("table_name") or row.get("row_grain"),
            default_action=_action_for_artifact(row),
            severity=severity,
            incoming_edge_types=frozenset(incoming_edge_types.get(artifact_id, set())),
            metadata=metadata,
            artifact=row,
        )
        actions.append(
            RebuildPlanAction(
                step=len(actions) + 1,
                action=policy.action_for(context),
                artifact_id=artifact_id,
                kind=str(row.get("kind") or ""),
                name=row.get("name"),
                table_name=row.get("table_name"),
                table_version=row.get("table_version"),
                table_tag=row.get("table_tag"),
                row_ids=tuple(str(value) for value in row.get("row_ids") or []),
                depends_on=dependencies,
                reason=reason,
                metadata=metadata,
            )
        )
    return tuple(actions)


# --- Rebuild-plan aggregates, guardrails, and action pagination (backlog 0110) ---


def _rebuild_plan_aggregates(
    graph: LineageGraph,
    actions: Sequence[RebuildPlanAction],
) -> tuple[dict[str, int], dict[str, int]]:
    """Return ``(actions_by_type, affected_by_kind)`` counts over the full set."""

    actions_by_type: dict[str, int] = {}
    for action in actions:
        actions_by_type[action.action] = actions_by_type.get(action.action, 0) + 1
    affected_by_kind: dict[str, int] = {}
    for row in graph.artifacts:
        kind = str(row.get("kind") or "")
        affected_by_kind[kind] = affected_by_kind.get(kind, 0) + 1
    return actions_by_type, affected_by_kind


def _rebuild_page_digest(
    root_ids: Sequence[str],
    *,
    reason: str | None,
    severity: str | None,
    policy_name: str | None,
) -> str:
    """Stable digest binding an action page token to the plan that minted it."""

    return _stable_digest(
        {
            "roots": sorted(str(r) for r in root_ids),
            "reason": reason,
            "severity": severity,
            "policy": policy_name,
        }
    )[:16]


def _encode_action_page_token(after_step: int, digest: str) -> str:
    raw = json.dumps({"after": int(after_step), "q": digest}, sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_action_page_token(token: str, digest: str) -> int:
    try:
        payload = json.loads(base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8"))
        after_step = int(payload["after"])
        token_digest = str(payload["q"])
    except (ValueError, KeyError, TypeError) as exc:
        raise RebuildPlanError(f"invalid rebuild-plan page token: {token!r}") from exc
    if token_digest != digest:
        raise RebuildPlanError(
            "rebuild-plan page token does not match this plan; a continuation handle "
            "is only valid for the plan (roots + reason + severity + policy) that "
            "produced it"
        )
    return after_step


def _paginate_actions(
    actions: tuple[RebuildPlanAction, ...],
    *,
    page_size: int | None,
    page_token: str | None,
    digest: str,
) -> tuple[tuple[RebuildPlanAction, ...], str | None, bool]:
    """Slice ``actions`` (already ordered by ``step``) into a stable page.

    Returns ``(page_actions, next_page_token, truncated)``. Pages are consecutive,
    non-overlapping slices keyed by the deterministic ``step``, so the union of
    pages equals the full plan and totals stay stable across pages.
    """

    if page_size is None:
        if page_token is not None:
            raise RebuildPlanError("page_token requires page_size")
        return actions, None, False
    if page_size < 1:
        raise RebuildPlanError("page_size must be >= 1")
    start = 0
    if page_token is not None:
        after_step = _decode_action_page_token(page_token, digest)
        # ``step`` is 1-based and contiguous; resume just past the last returned.
        start = after_step
    page = actions[start : start + page_size]
    has_next = start + page_size < len(actions) and bool(page)
    next_token = (
        _encode_action_page_token(page[-1].step, digest) if has_next else None
    )
    return page, next_token, has_next


def _check_rebuild_index_readiness(lake: Lake) -> None:
    """Raise an actionable error if lineage traversal indexes are missing.

    Traversal stays correct without the indexes (predicate pushdown), but at large
    scale the missing ``lineage_edges`` endpoint indexes turn each frontier read
    into a full scan. Callers that opt into ``require_indexes`` want to fail fast
    with the exact build call rather than silently degrade (backlog 0110).
    """

    plan = _lineage_index_plan(lake)
    missing = plan.get("missing") or []
    if missing:
        cols = ", ".join(
            f"{entry['table']}.{entry['column']} ({entry['index_type']})" for entry in missing
        )
        raise RebuildPlanError(
            "rebuild plan requires lineage traversal indexes that are missing: "
            f"{cols}. Build them with lake.lineage.refresh_graph() or "
            "build_lineage_predicate_indexes(lake), or pass require_indexes=False to "
            "plan anyway (slower on large graphs)."
        )


def _is_plan_dependency_edge(edge: dict[str, Any]) -> bool:
    return str(edge.get("edge_type") or "") in {
        "ingested-from",
        "contains-observation",
        "contains-episode",
        "windowed-run",
        "windowed-into",
        "alignment-output",
        "aligned-from",
        "selected-from",
        "source-coordinate",
        "version-pinned",
        "trained-on",
        "produced-model",
        "evaluated-model",
        "evaluated-on",
        "produced-output",
        "scored-row",
        "feedback-on",
        "feedback-target",
        "invalidates",
        "superseded-by",
    }


def _topological_artifact_order(
    artifacts: dict[str, dict[str, Any]],
    edges: Sequence[dict[str, Any]],
    *,
    roots: Sequence[str],
) -> list[str]:
    nodes = set(artifacts)
    root_rank = {root: index for index, root in enumerate(roots)}
    adjacency = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for edge in edges:
        upstream = edge["from_artifact_id"]
        downstream = edge["to_artifact_id"]
        if upstream not in nodes or downstream not in nodes or upstream == downstream:
            continue
        if downstream in adjacency[upstream]:
            continue
        adjacency[upstream].add(downstream)
        indegree[downstream] += 1

    def sort_key(artifact_id: str) -> tuple[int, int, str]:
        return (0 if artifact_id in root_rank else 1, root_rank.get(artifact_id, 10**9), artifact_id)

    ready = sorted((node for node, degree in indegree.items() if degree == 0), key=sort_key)
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for downstream in sorted(adjacency[current], key=sort_key):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                ready.append(downstream)
        ready.sort(key=sort_key)
    if len(ordered) < len(nodes):
        ordered.extend(sorted(nodes - set(ordered), key=sort_key))
    return ordered


def _action_for_artifact(row: dict[str, Any]) -> str:
    kind = str(row.get("kind") or "")
    table_name = str(row.get("table_name") or row.get("row_grain") or "")
    metadata = _metadata_map(row)
    if kind == "source":
        return "quarantine"
    if kind == "dataset-snapshot":
        return "resnapshot"
    if kind == "projection-manifest":
        return "re-export"
    if kind in {"training-run", "model"}:
        return "retrain"
    if kind == "evaluation-run":
        return "re-evaluate"
    if kind == "search-index":
        return "recompute"
    if kind == "invalidation":
        return "notify-only"
    if kind == "transform":
        transform_kind = metadata.get("kind") or str(row.get("name") or "")
        if transform_kind in {"projection", "dataset-export"}:
            return "re-export"
        if transform_kind in {"training-run", "model-artifact"}:
            return "retrain"
        if transform_kind == "evaluation-run":
            return "re-evaluate"
        if transform_kind == "maintenance":
            return "notify-only"
        return "recompute"
    if kind == "table-version":
        if table_name == "dataset_snapshots":
            return "resnapshot"
        if table_name in {"training_runs", "model_artifacts"}:
            return "retrain"
        if table_name in {"evaluation_runs", "model_outputs"}:
            return "re-evaluate"
        return "recompute"
    if kind == "row":
        if table_name in {"runs", "observations", "attachments", "videos", "video_encodings"}:
            return "quarantine"
        if table_name == "dataset_snapshots":
            return "resnapshot"
        if table_name in {"training_runs", "model_artifacts"}:
            return "retrain"
        if table_name in {"evaluation_runs", "model_outputs"}:
            return "re-evaluate"
        if table_name.startswith("curation_") or table_name in {
            "scenarios",
            "episodes",
            "events",
            "alignment_jobs",
            "aligned_frames",
            "aligned_ticks",
            "labels",
        }:
            return "recompute"
    return "notify-only"


def _target_paths(
    root_ids: Sequence[str],
    seen_artifacts: set[str],
    seen_edges: dict[str, dict[str, Any]],
    parents: dict[str, set[tuple[str, str]]],
    artifacts: dict[str, dict[str, Any]],
    target_kinds: set[str],
) -> tuple[set[str], set[str]]:
    kept_artifacts = set(root_ids)
    kept_edges: set[str] = set()
    targets = {
        artifact_id
        for artifact_id in seen_artifacts
        if _artifact_matches_kind(artifacts.get(artifact_id, {}), target_kinds)
    }
    processed: set[str] = set()
    pending = list(sorted(targets))
    while pending:
        artifact_id = pending.pop()
        if artifact_id in processed:
            continue
        processed.add(artifact_id)
        kept_artifacts.add(artifact_id)
        for parent_id, edge_id in sorted(parents.get(artifact_id, ())):
            if edge_id not in seen_edges:
                continue
            kept_edges.add(edge_id)
            if parent_id not in kept_artifacts:
                pending.append(parent_id)
    return kept_artifacts, kept_edges


def _artifact_matches_kind(row: dict[str, Any], target_kinds: set[str]) -> bool:
    row_kind = str(row.get("kind") or "")
    if row_kind in target_kinds:
        return True
    for target in target_kinds:
        if target in _HANDLE_KIND_TABLES:
            table_name, _id_column = _HANDLE_KIND_TABLES[target]
            if row.get("table_name") == table_name or row.get("row_grain") == table_name:
                return True
        if target == "dataset-snapshot" and row_kind == "dataset-snapshot":
            return True
        if target == "training-run" and row_kind == "training-run":
            return True
        if target == "model" and row_kind == "model":
            return True
        if target == "projection-manifest" and row_kind == "projection-manifest":
            return True
    return False


def _resolve_artifact_ids(
    lake: Lake,
    handle: str,
    *,
    kind: str | None = None,
    table_version: int | None = None,
) -> tuple[str, ...]:
    value = str(handle or "").strip()
    if not value:
        raise LineageError("artifact handle is required")
    normalized_kind = _normalize_handle_kind(kind)
    # Indexed fast path (BUG-13): a direct artifact_id handle -- the common case, and
    # BUG-13's own example -- resolves via a point lookup on the BUG-15 artifact_id
    # index instead of loading all ~700K lineage_artifacts rows. Exotic handles
    # (name/source_uri/metadata matches) still fall through to the full scan below.
    direct = _fetch_rows_by_id_in(lake, "lineage_artifacts", "artifact_id", [value]).get(value)
    if direct is not None and normalized_kind in {None, "artifact", direct.get("kind")}:
        return (value,)
    artifacts = lake.table("lineage_artifacts").to_arrow().to_pylist()
    by_id = {row["artifact_id"]: row for row in artifacts}
    if value in by_id and normalized_kind in {None, "artifact", by_id[value].get("kind")}:
        return (value,)

    candidates: set[str] = set()
    for row in artifacts:
        if table_version is not None and row.get("table_version") not in (None, table_version):
            continue
        if _artifact_matches_handle(row, value, normalized_kind):
            candidates.add(str(row["artifact_id"]))

    for artifact_id in _candidate_artifact_ids_from_canonical_rows(
        lake,
        value,
        normalized_kind,
        table_version=table_version,
    ):
        if artifact_id in by_id:
            candidates.add(artifact_id)

    if candidates:
        return tuple(sorted(candidates))

    kind_hint = f" with kind {kind!r}" if kind else ""
    raise LineageError(
        f"unknown lineage artifact handle {value!r}{kind_hint}; run "
        "lake.lineage.refresh_graph() or record the artifact first"
    )


# --- Handle resolution catalog and ambiguity diagnostics (backlog 0102) ---------
#
# The recognized normalized handle kinds, derived from the alias map so the two
# never drift. An explicit ``--kind`` outside this set is reported as
# ``unsupported-kind`` rather than silently matching nothing.
_KNOWN_HANDLE_KINDS = frozenset(_HANDLE_KIND_ALIASES.values())
# Kinds whose many-per-handle fan-out is expected and NOT ambiguous: one source URI
# legitimately maps to many source-coordinate roots (0063 decision). Any other kind
# matching >1 distinct entity is a genuine ambiguity the caller must disambiguate.
_MULTI_ROOT_KINDS = frozenset({"source"})
# The bounded equality columns the resolver probes on ``lineage_artifacts``. The
# first four carry a BTREE scalar index (indexing.LINEAGE_ARTIFACT_PREDICATE_INDEX_
# COLUMNS); ``name``/``table_tag`` ride predicate pushdown. Every probe reads only
# matching rows, never the whole table.
_RESOLUTION_MATCH_COLUMNS = (
    "source_id",
    "digest",
    "producer_execution_id",
    "source_uri",
    "name",
    "table_tag",
)


def _query_table_by_column(
    lake: Lake,
    table_name: str,
    column: str,
    value: str,
) -> list[dict[str, Any]]:
    """Rows where ``column`` equals ``value``, via a predicate-pushed equality scan.

    Bounded by construction: the ``column = value`` predicate is pushed into the
    scan (indexed when the column carries a scalar index), so only matching rows are
    materialized -- never the whole table. Uses the remote-safe lancedb query API so
    it works over ``db://``.
    """

    if value is None or str(value) == "":
        return []
    try:
        table = lake.table(table_name)
    except Exception:  # noqa: BLE001 - table may not exist in this lake
        return []
    rows: list[dict[str, Any]] = []
    query = table.search().where(f"{column} = {_sql_literal(str(value))}")
    for batch in query.to_batches(batch_size=4096):
        rows.extend(batch.to_pylist())
    return rows


def _kind_allows(kind: str | None, row: dict[str, Any]) -> bool:
    """Whether a graph row satisfies a normalized ``--kind`` gate (mirrors resolve)."""

    if kind is None or kind == "artifact":
        return True
    if kind in _HANDLE_KIND_TABLES:
        table_name, _id_column = _HANDLE_KIND_TABLES[kind]
        return row.get("table_name") == table_name or row.get("row_grain") == table_name
    return kind == str(row.get("kind") or "")


def _candidate_derived_ids(
    handle: str,
    kind: str | None,
    table_version: int | None,
    current_versions: Mapping[str, int],
) -> dict[str, str]:
    """Derived graph artifact ids a handle could be, keyed to the field they encode.

    Point-lookup targets (checked against the ``artifact_id`` index), so a manifest
    or row handle resolves without scanning ``lineage_artifacts``.
    """

    derived: dict[str, str] = {}
    if kind in (None, "dataset-snapshot"):
        derived[snapshot_artifact_id(handle)] = "dataset_id"
    if kind in (None, "training-run"):
        derived[training_run_artifact_id(handle)] = "training_run_id"
    if kind in (None, "model"):
        derived[model_artifact_lineage_id(handle)] = "model_artifact_id"
    if kind in (None, "evaluation-run"):
        derived[evaluation_run_artifact_id(handle)] = "eval_run_id"
    if kind in (None, "transform"):
        derived[execution_artifact_id(handle)] = "transform_id"
    row_kinds = (
        _HANDLE_KIND_TABLES
        if kind is None
        else ({kind: _HANDLE_KIND_TABLES[kind]} if kind in _HANDLE_KIND_TABLES else {})
    )
    for _row_kind, (table_name, _id_column) in row_kinds.items():
        version = table_version if table_version is not None else current_versions.get(table_name)
        derived[
            artifact_id("row", table_name=table_name, row_id=handle, table_version=version)
        ] = table_name
    return derived


def _canonical_probe(
    lake: Lake,
    handle: str,
    kind: str | None,
    table_version: int | None,
    current_versions: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Bounded probe of canonical source tables for artifacts a refresh would create.

    Returns one entry per matching canonical row: the artifact id it would project
    to, its kind, source table, and match evidence. Every probe is a point lookup on
    an id column or a predicate-pushed equality scan -- no full canonical scans. Used
    both to add derived candidates (snapshot name/tag, model uri/checksum) and to
    detect stale graph state (known canonically but not yet materialized).
    """

    entries: list[dict[str, Any]] = []

    def known(table: str) -> bool:
        return table in current_versions

    if kind in (None, "dataset-snapshot") and known("dataset_snapshots"):
        rows: dict[str, dict[str, Any]] = dict(
            _fetch_rows_by_id_in(lake, "dataset_snapshots", "dataset_id", [handle])
        )
        for column in ("name", "tag"):
            for row in _query_table_by_column(lake, "dataset_snapshots", column, handle):
                rows[str(row["dataset_id"])] = row
        for dataset_id, row in rows.items():
            entries.append(
                {
                    "artifact_id": snapshot_artifact_id(str(dataset_id)),
                    "kind": "dataset-snapshot",
                    "table_name": "dataset_snapshots",
                    "matched_on": "dataset_snapshots",
                    "evidence": {
                        "dataset_id": str(dataset_id),
                        "name": row.get("name"),
                        "tag": row.get("tag"),
                    },
                }
            )
    if kind in (None, "training-run") and known("training_runs"):
        for _tid, _row in _fetch_rows_by_id_in(
            lake, "training_runs", "training_run_id", [handle]
        ).items():
            entries.append(
                {
                    "artifact_id": training_run_artifact_id(handle),
                    "kind": "training-run",
                    "table_name": "training_runs",
                    "matched_on": "training_run_id",
                    "evidence": {"training_run_id": handle},
                }
            )
    if kind in (None, "model") and known("model_artifacts"):
        rows = dict(_fetch_rows_by_id_in(lake, "model_artifacts", "model_artifact_id", [handle]))
        for column in ("training_run_id", "artifact_uri", "checksum"):
            for row in _query_table_by_column(lake, "model_artifacts", column, handle):
                rows[str(row["model_artifact_id"])] = row
        for model_artifact_id, row in rows.items():
            entries.append(
                {
                    "artifact_id": model_artifact_lineage_id(str(model_artifact_id)),
                    "kind": "model",
                    "table_name": "model_artifacts",
                    "matched_on": "model_artifacts",
                    "evidence": {
                        "model_artifact_id": str(model_artifact_id),
                        "training_run_id": row.get("training_run_id"),
                    },
                }
            )
    if kind in (None, "evaluation-run") and known("evaluation_runs"):
        for _eid, _row in _fetch_rows_by_id_in(
            lake, "evaluation_runs", "eval_run_id", [handle]
        ).items():
            entries.append(
                {
                    "artifact_id": evaluation_run_artifact_id(handle),
                    "kind": "evaluation-run",
                    "table_name": "evaluation_runs",
                    "matched_on": "eval_run_id",
                    "evidence": {"eval_run_id": handle},
                }
            )
    if kind in (None, "transform") and known("transform_runs"):
        for _xid, _row in _fetch_rows_by_id_in(
            lake, "transform_runs", "transform_id", [handle]
        ).items():
            entries.append(
                {
                    "artifact_id": execution_artifact_id(handle),
                    "kind": "transform",
                    "table_name": "transform_runs",
                    "matched_on": "transform_id",
                    "evidence": {"transform_id": handle},
                }
            )
    row_kinds = (
        _HANDLE_KIND_TABLES
        if kind is None
        else ({kind: _HANDLE_KIND_TABLES[kind]} if kind in _HANDLE_KIND_TABLES else {})
    )
    for _row_kind, (table_name, id_column) in row_kinds.items():
        if not known(table_name):
            continue
        for _rid, _row in _fetch_rows_by_id_in(lake, table_name, id_column, [handle]).items():
            version = table_version if table_version is not None else current_versions.get(table_name)
            entries.append(
                {
                    "artifact_id": artifact_id(
                        "row", table_name=table_name, row_id=handle, table_version=version
                    ),
                    "kind": "row",
                    "table_name": table_name,
                    "matched_on": table_name,
                    "evidence": {id_column: handle, "table_version": version},
                }
            )
    return entries


def _candidate_from_graph_row(
    row: dict[str, Any], matched_on: Sequence[str]
) -> LineageResolutionCandidate:
    metadata = _metadata_map(row)
    evidence: dict[str, Any] = {}
    if row.get("producer_execution_id"):
        evidence["producer_execution_id"] = str(row["producer_execution_id"])
    for key in ("tag", "channel", "offset", "log_time_ns"):
        if metadata.get(key) not in (None, ""):
            evidence[key] = metadata[key]
    row_ids = [str(value) for value in (row.get("row_ids") or []) if value]
    if row_ids:
        evidence["row_ids"] = row_ids[:5]
    return LineageResolutionCandidate(
        artifact_id=str(row["artifact_id"]),
        kind=row.get("kind"),
        name=row.get("name"),
        table_name=row.get("table_name"),
        table_version=row.get("table_version"),
        table_tag=row.get("table_tag"),
        row_grain=row.get("row_grain"),
        source_uri=row.get("source_uri"),
        source_id=row.get("source_id"),
        digest=row.get("digest"),
        matched_on=tuple(matched_on),
        in_graph=True,
        evidence=evidence,
    )


def _candidate_from_canonical_entry(entry: dict[str, Any]) -> LineageResolutionCandidate:
    evidence = {
        key: value for key, value in (entry.get("evidence") or {}).items() if value is not None
    }
    return LineageResolutionCandidate(
        artifact_id=str(entry["artifact_id"]),
        kind=entry.get("kind"),
        table_name=entry.get("table_name"),
        matched_on=(str(entry.get("matched_on") or "canonical"),),
        in_graph=False,
        evidence=evidence,
    )


def _resolve_diagnostics(
    lake: Lake,
    handle: str,
    *,
    kind: str | None = None,
    table_version: int | None = None,
) -> LineageResolution:
    value = str(handle or "").strip()
    if not value:
        raise LineageError("artifact handle is required")
    normalized_kind = _normalize_handle_kind(kind)

    # Freshness snapshot: which source tables the graph is fresh through, and which
    # changed since. Read-only -- never triggers a refresh.
    watermark = _read_refresh_watermark(lake)
    current_versions = _safe_current_table_versions(lake)
    fresh_through = (
        {str(table): int(version) for table, version in (watermark or {}).get("versions", {}).items()}
        if watermark
        else {}
    )
    refreshed_at = (watermark or {}).get("refreshed_at")
    stale_tables = tuple(
        sorted(
            table
            for table, version in current_versions.items()
            if int(fresh_through.get(table, -1)) != int(version)
        )
    )
    graph_fresh = watermark is not None and not stale_tables
    refresh_command = f"lancedb-robotics lineage refresh --lake {lake.uri}"

    def build(
        status: str,
        *,
        candidates: Sequence[LineageResolutionCandidate] = (),
        artifact_ids: Sequence[str] = (),
        multi_root: bool = False,
        hints: Sequence[dict[str, Any]] = (),
        commands: Sequence[str] = (),
        pending: Sequence[str] = (),
        message: str | None = None,
    ) -> LineageResolution:
        return LineageResolution(
            lake_uri=lake.uri,
            handle=value,
            status=status,
            requested_kind=normalized_kind,
            candidates=tuple(candidates),
            artifact_ids=tuple(artifact_ids),
            root_count=len(tuple(artifact_ids)),
            multi_root=multi_root,
            graph_fresh=graph_fresh,
            refreshed_at=refreshed_at,
            fresh_through_versions=fresh_through,
            stale_tables=stale_tables,
            pending_refresh_artifact_ids=tuple(pending),
            disambiguation_hints=tuple(hints),
            suggested_commands=tuple(commands),
            message=message,
        )

    if kind is not None and str(kind).strip() != "" and normalized_kind not in _KNOWN_HANDLE_KINDS:
        supported = ", ".join(sorted(_KNOWN_HANDLE_KINDS))
        return build(
            "unsupported-kind",
            message=(
                f"unsupported handle kind {kind!r}; supported kinds: {supported}"
            ),
        )

    graph_rows: dict[str, dict[str, Any]] = {}
    matched: dict[str, set[str]] = {}

    def add_graph(row: dict[str, Any], field_name: str) -> None:
        raw_id = row.get("artifact_id")
        if not raw_id or str(raw_id) == _REFRESH_STATE_ARTIFACT_ID:
            return
        if table_version is not None and row.get("table_version") not in (None, table_version):
            return
        if not _kind_allows(normalized_kind, row):
            return
        aid = str(raw_id)
        graph_rows[aid] = row
        matched.setdefault(aid, set()).add(field_name)

    # 1. Exact id + derived-id point lookups (indexed).
    derived_ids = _candidate_derived_ids(value, normalized_kind, table_version, current_versions)
    for aid, row in _fetch_rows_by_id_in(
        lake, "lineage_artifacts", "artifact_id", [*derived_ids, value]
    ).items():
        add_graph(row, "artifact_id" if aid == value else derived_ids.get(aid, "artifact_id"))

    # 2. Bounded equality-column matches (skipped for kind=artifact: exact id only).
    if normalized_kind != "artifact":
        for column in _RESOLUTION_MATCH_COLUMNS:
            for row in _query_table_by_column(lake, "lineage_artifacts", column, value):
                add_graph(row, column)

    # 3. Canonical probe: derived candidates + stale detection.
    canonical_entries = _canonical_probe(lake, value, normalized_kind, table_version, current_versions)
    canonical_extra = [
        entry["artifact_id"] for entry in canonical_entries if entry["artifact_id"] not in graph_rows
    ]
    if canonical_extra:
        for _aid, row in _fetch_rows_by_id_in(
            lake, "lineage_artifacts", "artifact_id", canonical_extra
        ).items():
            add_graph(row, "canonical")

    resolved_ids = tuple(sorted(graph_rows))
    graph_candidates = [
        _candidate_from_graph_row(graph_rows[aid], sorted(matched.get(aid, {"canonical"})))
        for aid in resolved_ids
    ]
    pending = tuple(
        sorted({entry["artifact_id"] for entry in canonical_entries if entry["artifact_id"] not in graph_rows})
    )

    if resolved_ids:
        distinct_kinds = {str(row.get("kind") or "") for row in graph_rows.values()}
        multi_root = len(resolved_ids) > 1 and distinct_kinds <= _MULTI_ROOT_KINDS
        common_kind = next(iter(distinct_kinds)) if len(distinct_kinds) == 1 else None
        if len(resolved_ids) == 1 or multi_root:
            status = "resolved"
            hints: list[dict[str, Any]] = []
            kind_flag = f" --kind {common_kind}" if common_kind else ""
            commands = [
                f"lancedb-robotics lineage impact {value}{kind_flag} --lake {lake.uri}",
                f"lancedb-robotics lineage trace {value}{kind_flag} --lake {lake.uri}",
            ]
            message = (
                f"{len(resolved_ids)} source-coordinate root(s) for one handle"
                if multi_root
                else None
            )
            if pending:
                message = (
                    (message + "; " if message else "")
                    + f"{len(pending)} additional match(es) pending graph refresh"
                )
                commands.append(refresh_command)
        else:
            status = "ambiguous"
            hints = _ambiguity_hints(graph_candidates)
            commands = []
            for candidate in graph_candidates:
                commands.append(
                    f"lancedb-robotics lineage impact {candidate.artifact_id} --lake {lake.uri}"
                )
            message = (
                f"{len(resolved_ids)} distinct artifacts match {value!r}; "
                "pass a disambiguating --kind/--table-version or use an exact artifact id"
            )
        return build(
            status,
            candidates=graph_candidates,
            artifact_ids=resolved_ids,
            multi_root=multi_root,
            hints=hints,
            commands=commands,
            pending=pending,
            message=message,
        )

    if canonical_entries:
        canonical_candidates = [_candidate_from_canonical_entry(entry) for entry in canonical_entries]
        stale_hint = (
            f"changed since refresh: {', '.join(stale_tables)}" if stale_tables else "graph not yet refreshed"
        )
        return build(
            "stale",
            candidates=canonical_candidates,
            pending=pending,
            commands=[refresh_command],
            message=(
                f"{value!r} is known in canonical tables but not yet in the lineage "
                f"graph ({stale_hint}); run refresh_graph() before trace/impact"
            ),
        )

    kind_hint = f" with kind {normalized_kind!r}" if normalized_kind else ""
    return build(
        "unknown",
        message=(
            f"no lineage artifact or canonical row matches handle {value!r}{kind_hint}"
        ),
    )


def _ambiguity_hints(
    candidates: Sequence[LineageResolutionCandidate],
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    kinds = sorted({candidate.kind for candidate in candidates if candidate.kind})
    if len(kinds) > 1:
        hints.append({"flag": "--kind", "values": kinds})
    table_versions = sorted(
        {
            f"{candidate.table_name}={candidate.table_version}"
            for candidate in candidates
            if candidate.table_name and candidate.table_version is not None
        }
    )
    if len(table_versions) > 1:
        hints.append({"flag": "--table-version", "values": table_versions})
    # The always-available disambiguator: re-run against an exact artifact id. Listed
    # last so kind/table-version (the cheaper hints) take precedence when they apply.
    hints.append(
        {"flag": "artifact-id", "values": [candidate.artifact_id for candidate in candidates]}
    )
    return hints


def _normalize_handle_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    normalized = str(kind).strip().lower().replace("_", "-")
    if not normalized:
        return None
    return _HANDLE_KIND_ALIASES.get(normalized, normalized)


def _artifact_matches_handle(
    row: dict[str, Any],
    handle: str,
    kind: str | None,
) -> bool:
    row_kind = str(row.get("kind") or "")
    if kind and kind != "artifact":
        if kind in _HANDLE_KIND_TABLES:
            table_name, _id_column = _HANDLE_KIND_TABLES[kind]
            if row.get("table_name") != table_name and row.get("row_grain") != table_name:
                return False
        elif kind != row_kind:
            return False
    metadata = _metadata_map(row)
    row_ids = {str(value) for value in row.get("row_ids") or [] if value}
    values = {
        str(value)
        for value in (
            row.get("artifact_id"),
            row.get("name"),
            row.get("source_uri"),
            row.get("source_id"),
            row.get("digest"),
            row.get("producer_execution_id"),
            row.get("table_tag"),
            *metadata.values(),
        )
        if value is not None and value != ""
    }
    if handle in values or handle in row_ids:
        return True
    if row_kind == "dataset-snapshot" and (
        row.get("artifact_id") == snapshot_artifact_id(handle)
        or metadata.get("tag") == handle
    ):
        return True
    if row_kind == "training-run" and row.get("artifact_id") == training_run_artifact_id(handle):
        return True
    if row_kind == "model" and row.get("artifact_id") == model_artifact_lineage_id(handle):
        return True
    if row_kind == "evaluation-run" and row.get("artifact_id") == evaluation_run_artifact_id(handle):
        return True
    if row_kind == "transform" and row.get("artifact_id") == execution_artifact_id(handle):
        return True
    return False


def _candidate_artifact_ids_from_canonical_rows(
    lake: Lake,
    handle: str,
    kind: str | None,
    *,
    table_version: int | None,
) -> tuple[str, ...]:
    candidates: list[str] = []
    current_versions = _safe_current_table_versions(lake)

    def version_for(table_name: str) -> int | None:
        return table_version if table_version is not None else current_versions.get(table_name)

    if kind in {None, "dataset-snapshot"}:
        for row in lake.table("dataset_snapshots").to_arrow().to_pylist():
            metadata_tag = row.get("tag")
            if handle in {row.get("dataset_id"), row.get("name"), metadata_tag}:
                candidates.append(snapshot_artifact_id(str(row["dataset_id"])))
    if kind in {None, "training-run"}:
        for row in lake.table("training_runs").to_arrow().to_pylist():
            if handle == row.get("training_run_id"):
                candidates.append(training_run_artifact_id(handle))
    if kind in {None, "model"}:
        for row in lake.table("model_artifacts").to_arrow().to_pylist():
            aliases = {str(value) for value in row.get("aliases") or [] if value}
            refs = _metadata_map(row, column="external_refs")
            values = {
                row.get("model_artifact_id"),
                row.get("training_run_id"),
                row.get("artifact_uri"),
                row.get("checksum"),
                *refs.values(),
            }
            if handle in aliases or handle in {str(value) for value in values if value}:
                candidates.append(model_artifact_lineage_id(str(row["model_artifact_id"])))
    if kind in {None, "evaluation-run"}:
        for row in lake.table("evaluation_runs").to_arrow().to_pylist():
            if handle == row.get("eval_run_id"):
                candidates.append(evaluation_run_artifact_id(handle))
    if kind in {None, "transform"}:
        for row in lake.table("transform_runs").to_arrow().to_pylist():
            if handle == row.get("transform_id"):
                candidates.append(execution_artifact_id(handle))
    row_kinds = _HANDLE_KIND_TABLES if kind is None else {kind: _HANDLE_KIND_TABLES[kind]} if kind in _HANDLE_KIND_TABLES else {}
    for _row_kind, (table_name, id_column) in row_kinds.items():
        if table_name not in current_versions:
            continue
        for row in lake.table(table_name).to_arrow().to_pylist():
            if handle == row.get(id_column):
                candidates.append(
                    artifact_id(
                        "row",
                        table_name=table_name,
                        row_id=handle,
                        table_version=version_for(table_name),
                    )
                )
    return tuple(dict.fromkeys(candidates))


def _safe_current_table_versions(lake: Lake) -> dict[str, int]:
    versions: dict[str, int] = {}
    for table in _GRAPH_SOURCE_TABLES:
        try:
            versions[table] = int(lake.table(table).version)
        except Exception:
            continue
    return versions


def _normalize_datetime(value: datetime | str | None, label: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise LineageError(f"{label} must be an ISO-8601 datetime") from exc


def _created_in_range(
    value: datetime | None,
    after: datetime | None,
    before: datetime | None,
) -> bool:
    if value is None:
        return True
    if after is not None and value < after:
        return False
    if before is not None and value > before:
        return False
    return True


def _normalize_table_version_filters(
    filters: Mapping[str, int] | Iterable[str | tuple[str, int]],
) -> dict[str, int]:
    if isinstance(filters, Mapping):
        return {str(table): int(version) for table, version in filters.items()}
    normalized: dict[str, int] = {}
    for item in filters:
        if isinstance(item, str):
            if "=" in item:
                table, version = item.split("=", 1)
            elif ":" in item:
                table, version = item.split(":", 1)
            else:
                raise LineageError("table version filters must look like table=version")
            normalized[table.strip()] = int(version)
            continue
        table, version = item
        normalized[str(table)] = int(version)
    return normalized


def _artifact_matches_filters(
    row: dict[str, Any],
    table_versions: dict[str, int],
    created_after: datetime | None,
    created_before: datetime | None,
) -> bool:
    if not _created_in_range(row.get("created_at"), created_after, created_before):
        return False
    table_name = row.get("table_name")
    if table_name and table_name in table_versions:
        return int(row.get("table_version") or -1) == table_versions[table_name]
    return True


def _model_outputs_for_run(lake: Lake, model_run_id: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in lake.table("model_outputs").to_arrow().to_pylist()
        if _model_run_alias(row) == model_run_id
    ]
    if not rows:
        raise LineageError(f"no model_outputs rows linked to model_run_id {model_run_id!r}")
    return sorted(rows, key=lambda row: row["model_output_id"])


def _model_run_alias(row: dict[str, Any]) -> str | None:
    if row.get("producer_run_id"):
        return row["producer_run_id"]
    metadata = _metadata_map(row)
    for key in ("model_run_id", "checkpoint_id", "training_run_id"):
        if metadata.get(key):
            return metadata[key]
    return None


def _model_artifacts_for_run(lake: Lake, model_run_id: str) -> list[dict[str, Any]]:
    rows = []
    for row in lake.table("model_artifacts").to_arrow().to_pylist():
        aliases = {str(value) for value in row.get("aliases") or [] if value}
        refs = _metadata_map(row, column="external_refs")
        candidates = {
            row.get("model_artifact_id"),
            row.get("training_run_id"),
            row.get("artifact_uri"),
            row.get("checksum"),
            refs.get("model_run_id"),
            refs.get("checkpoint_id"),
            refs.get("training_run_id"),
            refs.get("mlflow_run_id"),
            refs.get("wandb_run_id"),
        }
        if model_run_id in aliases or model_run_id in {value for value in candidates if value}:
            rows.append(row)
    return sorted(rows, key=lambda row: row["model_artifact_id"])


def _training_run_for_model_artifacts(
    lake: Lake,
    model_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    training_run_ids = {
        row.get("training_run_id") for row in model_artifacts if row.get("training_run_id")
    }
    if not training_run_ids:
        raise LineageError("model artifacts are missing training_run_id")
    if len(training_run_ids) != 1:
        raise LineageError(
            "model artifacts reference multiple training runs: "
            f"{sorted(training_run_ids)}"
        )
    training_run_id = next(iter(training_run_ids))
    for row in lake.table("training_runs").to_arrow().to_pylist():
        if row["training_run_id"] == training_run_id:
            return row
    raise LineageError(
        f"model artifact references unknown training_run_id {training_run_id!r}"
    )


def _resolve_snapshot_from_training_run(lake: Lake, training_run: dict[str, Any]) -> dict[str, Any]:
    dataset_id = training_run.get("dataset_id")
    if not dataset_id:
        raise LineageError(
            f"training run {training_run.get('training_run_id')!r} is missing dataset_id"
        )
    for row in lake.table("dataset_snapshots").to_arrow().to_pylist():
        if row["dataset_id"] == dataset_id:
            return row
    raise LineageError(
        f"training run references unknown dataset_id {dataset_id!r}"
    )


def _resolve_snapshot(lake: Lake, model_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    snapshots = lake.table("dataset_snapshots").to_arrow().to_pylist()
    by_id = {row["dataset_id"]: row for row in snapshots}

    dataset_ids = {row.get("dataset_id") for row in model_outputs if row.get("dataset_id")}
    metadata = [_metadata_map(row) for row in model_outputs]
    dataset_ids.update(item.get("dataset_id") for item in metadata if item.get("dataset_id"))
    dataset_ids.update(item.get("snapshot_dataset_id") for item in metadata if item.get("snapshot_dataset_id"))
    dataset_ids = {dataset_id for dataset_id in dataset_ids if dataset_id}

    if len(dataset_ids) > 1:
        raise LineageError(f"model run references multiple dataset snapshots: {sorted(dataset_ids)}")
    if len(dataset_ids) == 1:
        dataset_id = next(iter(dataset_ids))
        if dataset_id not in by_id:
            raise LineageError(f"model run references unknown dataset_id {dataset_id!r}")
        return by_id[dataset_id]

    tags = {
        value
        for item in metadata
        for value in (item.get("dataset_tag"), item.get("snapshot_tag"))
        if value
    }
    names = {
        value
        for item in metadata
        for value in (item.get("snapshot_name"), item.get("dataset_name"))
        if value
    }
    candidates = [
        row
        for row in snapshots
        if (row.get("tag") in tags and row.get("tag")) or (row.get("name") in names and row.get("name"))
    ]
    if not candidates:
        raise LineageError(
            "model run is not linked to a dataset snapshot; include dataset_id or "
            "metadata.dataset_tag/snapshot_name on model_outputs"
        )
    matches = _latest_by_key(candidates, key=lambda row: (row["created_at"], row["dataset_id"]))
    if len({row["dataset_id"] for row in matches}) > 1:
        raise LineageError(
            "model run metadata matches multiple latest dataset snapshots: "
            f"{sorted(row['dataset_id'] for row in matches)}"
        )
    return matches[0]


def _latest_by_key(
    rows: Iterable[dict[str, Any]],
    *,
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    rows = list(rows)
    if not rows:
        return []
    maximum = max(key(row) for row in rows)
    return [row for row in rows if key(row) == maximum]


def _metadata_map(row: dict[str, Any], *, column: str = "metadata") -> dict[str, str]:
    metadata = row.get(column) or []
    result: dict[str, str] = {}
    for item in metadata:
        if not isinstance(item, dict) or item.get("key") is None:
            continue
        result[str(item["key"])] = "" if item.get("value") is None else str(item["value"])
    return result


def _table_versions(snapshot: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    versions = snapshot.get("table_versions") or []
    return tuple(
        {
            "table": item["table"],
            "version": int(item["version"]),
            "tag": item.get("tag") or "",
        }
        for item in versions
    )


def _snapshot_context(lake: Lake, snapshot: dict[str, Any]) -> _SnapshotContext:
    query_spec = _json_object(snapshot.get("query_spec"), "query_spec")
    scenario_ids = tuple(sorted(query_spec.get("scenario_ids") or []))
    if not scenario_ids:
        raise LineageError(f"snapshot {snapshot['dataset_id']!r} has no scenario_ids in query_spec")

    versions = {item["table"]: int(item["version"]) for item in snapshot.get("table_versions") or []}
    scenario_rows = _table_rows_as_of(lake, "scenarios", versions.get("scenarios"))
    observation_rows = _table_rows_as_of(lake, "observations", versions.get("observations"))
    run_rows = _table_rows_as_of(lake, "runs", versions.get("runs"))

    scenarios = {row["scenario_id"]: row for row in scenario_rows}
    observations = {row["observation_id"]: row for row in observation_rows}
    runs = {row["run_id"]: row for row in run_rows}

    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in scenarios]
    if missing:
        raise LineageError(
            f"snapshot {snapshot['name']!r} references scenarios missing at the "
            f"pinned version: {missing}"
        )

    return _SnapshotContext(
        snapshot=snapshot,
        scenario_ids=scenario_ids,
        scenarios=scenarios,
        observations=observations,
        runs=runs,
    )


def _json_object(payload: str | None, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        raise LineageError(f"snapshot has invalid {label}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise LineageError(f"snapshot {label} must be a JSON object")
    return value


def _table_rows_as_of(lake: Lake, table_name: str, version: int | None) -> list[dict[str, Any]]:
    table = lake.table(table_name)
    if version is None:
        return table.to_arrow().to_pylist()
    table.checkout(version)
    try:
        return table.to_arrow().to_pylist()
    finally:
        table.checkout_latest()


def _trace_rows(context: _SnapshotContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_id in context.scenario_ids:
        scenario = context.scenarios[scenario_id]
        run = context.runs.get(scenario["run_id"], {})
        for observation_id in sorted(scenario.get("observation_ids") or []):
            observation = context.observations.get(observation_id)
            if observation is None:
                raise LineageError(
                    f"scenario {scenario_id!r} references observation {observation_id!r} "
                    "missing at the pinned version"
                )
            rows.append(_joined_row(context.snapshot["dataset_id"], observation, scenario, run))
    return sorted(rows, key=lambda row: (row["scenario_id"], row["timestamp_ns"], row["observation_id"]))


def _joined_row(
    dataset_id: str,
    observation: dict[str, Any],
    scenario: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    task_id = observation.get("task_id") or scenario.get("task_id") or run.get("task_id")
    robot_id = observation.get("robot_id") or run.get("robot_id")
    site_id = observation.get("site_id") or run.get("site_id")
    return {
        "dataset_id": dataset_id,
        "scenario_id": scenario["scenario_id"],
        "observation_id": observation["observation_id"],
        "run_id": observation["run_id"],
        "timestamp_ns": observation["timestamp_ns"],
        "sensor_id": observation.get("sensor_id"),
        "topic": observation.get("topic"),
        "modality": observation.get("modality"),
        "task_id": task_id,
        "robot_id": robot_id,
        "site_id": site_id,
        "scenario_type": scenario.get("scenario_type"),
        "scenario_start_time_ns": scenario.get("start_time_ns"),
        "scenario_end_time_ns": scenario.get("end_time_ns"),
        "raw_uri": observation.get("raw_uri"),
        "raw_channel": observation.get("raw_channel"),
        "raw_log_time_ns": observation.get("raw_log_time_ns"),
        "raw_sequence": observation.get("raw_sequence"),
        "observation_transform_id": observation.get("transform_id"),
        "scenario_transform_id": scenario.get("transform_id"),
        "run_transform_id": run.get("transform_id"),
    }


def _filter_rows(rows: list[dict[str, Any]], where: str | None) -> list[dict[str, Any]]:
    if where is None or not where.strip():
        return rows
    predicate = _compile_where(where, set().union(*(row.keys() for row in rows)) if rows else set())
    return [row for row in rows if predicate(row)]


def _compile_where(where: str, columns: set[str]) -> Callable[[dict[str, Any]], bool]:
    clauses = [clause.strip() for clause in _AND_RE.split(where) if clause.strip()]
    if not clauses:
        raise LineageError("where predicate is empty")
    checks = [_compile_clause(clause, columns) for clause in clauses]
    return lambda row: all(check(row) for check in checks)


def _compile_clause(clause: str, columns: set[str]) -> Callable[[dict[str, Any]], bool]:
    match = _IS_NULL_RE.match(clause)
    if match:
        field, negated = match.groups()
        _ensure_column(field, columns)
        return lambda row: (row.get(field) is not None) if negated else (row.get(field) is None)

    match = _IN_RE.match(clause)
    if match:
        field, raw_values = match.groups()
        _ensure_column(field, columns)
        values = tuple(_parse_literal(value) for value in _split_values(raw_values))
        return lambda row: row.get(field) in values

    match = _COMPARISON_RE.match(clause)
    if match:
        field, op, raw_value = match.groups()
        _ensure_column(field, columns)
        expected = _parse_literal(raw_value)
        return lambda row: _compare(row.get(field), op, expected)

    raise LineageError(
        f"unsupported where clause {clause!r}; use field = value, field IN (...), "
        "field IS NULL, and AND"
    )


def _ensure_column(field: str, columns: set[str]) -> None:
    if columns and field not in columns:
        raise LineageError(f"unknown where field {field!r}; available fields: {', '.join(sorted(columns))}")


def _split_values(raw: str) -> list[str]:
    values: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for char in raw:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
            continue
        if char == ",":
            values.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if quote:
        raise LineageError("unterminated string literal in where predicate")
    values.append("".join(current).strip())
    return [value for value in values if value]


def _parse_literal(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith(("'", '"')):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise LineageError(f"invalid string literal {value!r}") from exc
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if actual is None or expected is None:
        return _OPS[op](actual, expected) if op in {"=", "==", "!=", "<>"} else False
    actual_value, expected_value = _coerce_pair(actual, expected)
    try:
        return _OPS[op](actual_value, expected_value)
    except TypeError:
        return _OPS[op](str(actual), str(expected))


def _coerce_pair(actual: Any, expected: Any) -> tuple[Any, Any]:
    if isinstance(expected, bool):
        return bool(actual), expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual), expected
        except (TypeError, ValueError):
            return actual, expected
    if isinstance(expected, float):
        try:
            return float(actual), expected
        except (TypeError, ValueError):
            return actual, expected
    return actual, expected


def _source_logs(rows: list[dict[str, Any]]) -> list[SourceLogCoordinate]:
    seen: set[tuple[str, str, int | None]] = set()
    refs: list[SourceLogCoordinate] = []
    for row in rows:
        uri = row.get("raw_uri")
        channel = row.get("raw_channel") or row.get("topic")
        if not uri or not channel:
            continue
        offset = row.get("raw_sequence")
        if offset is not None:
            offset = int(offset)
        key = (str(uri), str(channel), offset)
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            SourceLogCoordinate(
                uri=str(uri),
                channel=str(channel),
                offset=offset,
                log_time_ns=row.get("raw_log_time_ns"),
                observation_id=row["observation_id"],
                run_id=row["run_id"],
                scenario_id=row["scenario_id"],
            )
        )
    return refs


def _transform_lineage(
    lake: Lake,
    snapshot: dict[str, Any],
    model_outputs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    training_run: dict[str, Any] | None = None,
    model_artifacts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    transforms = {row["transform_id"]: row for row in lake.table("transform_runs").to_arrow().to_pylist()}
    wanted: set[str] = {snapshot.get("transform_id")}
    wanted.update(row.get("transform_id") for row in model_outputs)
    wanted.update(row.get("transform_id") for row in (model_artifacts or []))
    if training_run is not None:
        wanted.add(training_run.get("transform_id"))
    for row in rows:
        wanted.update(
            (
                row.get("observation_transform_id"),
                row.get("scenario_transform_id"),
                row.get("run_transform_id"),
            )
        )

    expanded: set[str] = set()
    pending = [transform_id for transform_id in wanted if transform_id]
    while pending:
        transform_id = pending.pop()
        if transform_id in expanded:
            continue
        expanded.add(transform_id)
        row = transforms.get(transform_id)
        if row is None:
            continue
        for source_id in _source_transform_ids(row):
            if source_id not in expanded:
                pending.append(source_id)

    return [transforms[transform_id] for transform_id in sorted(expanded) if transform_id in transforms]


def _source_transform_ids(transform: dict[str, Any]) -> list[str]:
    try:
        params = json.loads(transform.get("params") or "{}")
    except json.JSONDecodeError:
        return []
    ids = params.get("source_transform_ids") or []
    return sorted(str(value) for value in ids if value)


def _snapshot_report(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": snapshot["dataset_id"],
        "name": snapshot["name"],
        "tag": snapshot.get("tag"),
        "kind": snapshot.get("kind"),
        "query_spec": snapshot.get("query_spec"),
        "table_versions": list(_table_versions(snapshot)),
        "transform_id": snapshot.get("transform_id"),
        "created_at": snapshot.get("created_at"),
    }
