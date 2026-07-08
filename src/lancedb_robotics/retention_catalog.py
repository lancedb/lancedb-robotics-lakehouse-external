"""Durable retention-policy catalog and governance hooks (backlog 0111).

Backlog 0067 stores generic retention hold facets (``retain_until``,
``legal_hold``, ``audit_hold``, ``promotion_hold``, ``owner``, ``reason``)
directly on lineage artifacts via :meth:`lake.lineage.retain`. That is enough for
OSS mechanics, but production robotics teams need durable, reusable *policy
definitions* -- scoped by artifact kind / table / owner / source / dataset /
model / deployment / project -- that can be versioned, approved, and *expanded*
to explicit artifact holds without manual per-artifact metadata edits, plus an
append-safe application/release history and a way to project policy + resolved
hold state out to enterprise governance / DLP / SIEM / records systems without
pulling those systems into OSS core. This module adds:

- a persisted catalog table (:data:`CATALOG_TABLE`) with one idempotent row per
  policy, keyed by ``policy_digest`` (== ``policy_id``), content-addressed over
  the policy's name / version / scope selectors / rules / owner / reason
  template. The full definition is stored inline (``policy_json``) so a policy
  reloads, applies, and projects without a separate definition store;
- a generic approval-style lifecycle (``draft`` -> ``active`` -> ``suspended``,
  plus terminal ``archived``) guarded by a ``revision`` optimistic-concurrency
  counter, so a status update made against a stale view is rejected with an
  actionable diagnostic. Activating a policy (making it enforce holds) requires
  an ``approver``;
- policy *application*: :func:`apply_retention_policy` resolves a policy's scope
  against ``lineage_artifacts`` and expands it into explicit 0067 holds, reusing
  the exact metadata contract maintenance already consumes. Artifact-local
  (manually set) holds are the lowest-level override: an artifact that already
  carries a hold from a human or another policy is a *conflict* -- reported
  deterministically and left untouched, never clobbered;
- :func:`resolve_retention_holds`: a read-only inspector that merges
  policy-applied and artifact-local holds into the SAME table-version pin shape
  maintenance consumes (via :func:`lineage_retention_pin_details` /
  :func:`snapshot_retention_pin_details`), classified by source, with
  deterministic shadowing diagnostics for active policies whose scope is
  overridden by an artifact-local hold;
- an append-only audit log (:data:`EVENTS_TABLE`) for recording, status
  transitions, hold application / release, and expiration notifications that
  survives status churn and policy archival;
- governance projection: :func:`export_retention_policy_state` emits policy +
  resolved-hold state (dict or NDJSON), and :func:`project_retention_state`
  hands it to a caller-supplied :class:`RetentionGovernanceSink` -- no mandatory
  dependency, no secrets persisted. Auth / endpoints live on the sink at runtime.

Org-specific authorization stays out of OSS core: ``actor`` / ``approver`` /
``note`` are free-form metadata, and enforcing *who* may approve or which
external system receives a projection belongs in an enterprise / plugin layer.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import (
    LINEAGE_ARTIFACTS_SCHEMA,
    LineageError,
    LineageRetentionHold,
    _kv_items,
    _metadata_map,
    _normalize_datetime,
    _replace_rows,
    _retention_hold_from_artifact_row,
    _utc_datetime,
    lineage_retention_pin_details,
    merge_retention_pin_details,
    retention_pin_rows,
    snapshot_retention_pin_details,
)
from lancedb_robotics.schemas import (
    RETENTION_POLICIES_SCHEMA,
    RETENTION_POLICY_EVENTS_SCHEMA,
)

CATALOG_TABLE = "retention_policies"
EVENTS_TABLE = "retention_policy_events"

#: Catalog row contract version (independent of the policy payload schema).
CATALOG_SCHEMA_VERSION = "lancedb-robotics/retention-policy-catalog/v1"

#: Policy payloads this catalog can record / reload.
SUPPORTED_POLICY_SCHEMAS: tuple[str, ...] = ("lancedb-robotics/retention-policy/v1",)
DEFAULT_POLICY_SCHEMA = SUPPORTED_POLICY_SCHEMAS[0]

#: Governance projection contract version.
GOVERNANCE_SCHEMA_VERSION = "lancedb-robotics/retention-governance/v1"

DEFAULT_STATUS = "draft"
DEFAULT_POLICY_VERSION = "1"

#: The scope selector categories, in canonical order. Empty categories are
#: unconstrained; within a category selectors OR; across categories they AND.
SCOPE_KEYS: tuple[str, ...] = (
    "kinds",
    "tables",
    "owners",
    "sources",
    "datasets",
    "models",
    "deployments",
    "projects",
    "name_prefixes",
    "artifact_ids",
)

#: Scope categories matched against per-artifact ``metadata`` values.
_METADATA_SCOPE_KEYS: dict[str, str] = {
    "owners": "owner",
    "datasets": "dataset",
    "models": "model",
    "deployments": "deployment",
    "projects": "project",
}

#: Lifecycle states in canonical order.
STATUSES: tuple[str, ...] = ("draft", "active", "suspended", "archived")

#: Allowed status transitions. ``archived`` is terminal.
_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"active", "archived"}),
    "active": frozenset({"suspended", "archived"}),
    "suspended": frozenset({"active", "archived"}),
    "archived": frozenset(),
}

#: Retention metadata keys written onto an artifact (must match backlog 0067).
_RETENTION_KEYS = (
    "retain_until",
    "legal_hold",
    "audit_hold",
    "promotion_hold",
    "owner",
    "reason",
    "retention_created_at",
)

#: Marker embedded in a hold's ``reason`` so applied holds are attributable to a
#: policy and cleanly releasable. Format: ``[policy:<policy_id>] <template>``.
_POLICY_MARKER_PREFIX = "policy:"

#: Bounded scan batch for scope resolution over ``lineage_artifacts``.
_SCAN_BATCH_ROWS = 4096


class RetentionPolicyError(LineageError):
    """A retention-policy catalog operation could not be completed."""


class RetentionPolicyConflict(RetentionPolicyError):
    """An optimistic-concurrency check rejected a stale status update."""


class RetentionPolicyTooLarge(RetentionPolicyError):
    """A policy's scope matched more artifacts than the configured guardrail."""


# --- Catalog entry / reports ------------------------------------------------


@dataclass(frozen=True)
class RetentionPolicyCatalogEntry:
    """A single durable retention-policy catalog row (definition kept inline)."""

    policy_id: str
    policy_digest: str
    catalog_schema_version: str
    policy_schema_version: str
    lake_uri: str
    name: str
    version: str
    scope_summary: str
    retain_until: datetime | None
    retain_for_days: int | None
    legal_hold: bool
    audit_hold: bool
    promotion_hold: bool
    owner: str
    status: str
    revision: int
    actor: str
    approver: str
    activated_at: datetime | None
    archived_at: datetime | None
    note: str
    metadata: dict[str, str]
    created_by: str
    created_at: datetime
    updated_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "catalog_schema_version": self.catalog_schema_version,
            "policy_schema_version": self.policy_schema_version,
            "lake_uri": self.lake_uri,
            "name": self.name,
            "version": self.version,
            "scope_summary": self.scope_summary,
            "retain_until": _iso_or_none(self.retain_until),
            "retain_for_days": self.retain_for_days,
            "legal_hold": self.legal_hold,
            "audit_hold": self.audit_hold,
            "promotion_hold": self.promotion_hold,
            "owner": self.owner,
            "status": self.status,
            "revision": self.revision,
            "actor": self.actor,
            "approver": self.approver,
            "activated_at": _iso_or_none(self.activated_at),
            "archived_at": _iso_or_none(self.archived_at),
            "note": self.note,
            "metadata": dict(self.metadata),
            "created_by": self.created_by,
            "created_at": _iso_or_none(self.created_at),
            "updated_at": _iso_or_none(self.updated_at),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class RetentionPolicyListPage:
    """A bounded page of catalog entries plus an opaque continuation cursor."""

    policies: tuple[RetentionPolicyCatalogEntry, ...]
    next_cursor: str | None
    page_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "policies": [entry.as_dict() for entry in self.policies],
            "next_cursor": self.next_cursor,
            "page_size": self.page_size,
            "count": len(self.policies),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class RetentionPolicyApplication:
    """The result (or dry-run preview) of applying or releasing a policy."""

    policy_id: str
    policy_digest: str
    name: str
    version: str
    operation: str  # "apply" | "release"
    dry_run: bool
    evaluated_at: datetime | None
    matched_count: int
    applied_count: int
    conflict_count: int
    applied_artifact_ids: tuple[str, ...]
    conflicts: tuple[dict[str, Any], ...]
    holds: tuple[dict[str, Any], ...]
    scope_bounded: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "name": self.name,
            "version": self.version,
            "operation": self.operation,
            "dry_run": self.dry_run,
            "evaluated_at": _iso_or_none(self.evaluated_at),
            "matched_count": self.matched_count,
            "applied_count": self.applied_count,
            "conflict_count": self.conflict_count,
            "applied_artifact_ids": list(self.applied_artifact_ids),
            "conflicts": [dict(row) for row in self.conflicts],
            "holds": [dict(row) for row in self.holds],
            "scope_bounded": self.scope_bounded,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class RetentionHoldResolution:
    """Resolved active holds in the exact pin shape maintenance consumes."""

    lake_uri: str
    hold_count: int
    policy_hold_count: int
    artifact_local_count: int
    holds: tuple[dict[str, Any], ...]
    pins: tuple[dict[str, Any], ...]
    conflicts: tuple[dict[str, Any], ...]
    policies_considered: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "lancedb-robotics/retention-hold-resolution/v1",
            "lake_uri": self.lake_uri,
            "hold_count": self.hold_count,
            "policy_hold_count": self.policy_hold_count,
            "artifact_local_count": self.artifact_local_count,
            "holds": [dict(row) for row in self.holds],
            "pins": [dict(row) for row in self.pins],
            "conflicts": [dict(row) for row in self.conflicts],
            "policies_considered": list(self.policies_considered),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class RetentionGovernanceProjection:
    """A dependency-free projection of policy + hold state for external systems."""

    schema_version: str
    lake_uri: str
    policies: tuple[dict[str, Any], ...]
    holds: tuple[dict[str, Any], ...]
    pins: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lake_uri": self.lake_uri,
            "policies": [dict(row) for row in self.policies],
            "holds": [dict(row) for row in self.holds],
            "pins": [dict(row) for row in self.pins],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()

    def iter_ndjson(self):
        """Yield one self-describing JSON line per policy, then per hold."""
        for policy in self.policies:
            yield json.dumps(
                {"record": "policy", "lake_uri": self.lake_uri, "policy": dict(policy)},
                sort_keys=True,
                separators=(",", ":"),
            )
        for hold in self.holds:
            yield json.dumps(
                {"record": "hold", "lake_uri": self.lake_uri, "hold": dict(hold)},
                sort_keys=True,
                separators=(",", ":"),
            )

    def as_ndjson(self) -> str:
        return "\n".join(self.iter_ndjson())


@runtime_checkable
class RetentionGovernanceSink(Protocol):
    """A caller-supplied hook that receives a governance projection.

    Implementations own their own transport, endpoint, and credentials at
    runtime; nothing from a sink is ever written into lake rows. This keeps
    enterprise governance / DLP / SIEM / records integrations out of OSS core.
    """

    def project(
        self, projection: RetentionGovernanceProjection
    ) -> Mapping[str, Any] | None:
        ...


@dataclass
class CollectingGovernanceSink:
    """A trivial in-memory sink for tests and previews (stores what it received)."""

    projections: list[RetentionGovernanceProjection] = field(default_factory=list)

    def project(
        self, projection: RetentionGovernanceProjection
    ) -> Mapping[str, Any]:
        self.projections.append(projection)
        return {
            "policies": len(projection.policies),
            "holds": len(projection.holds),
        }


# --- Policy construction ----------------------------------------------------


def build_retention_policy(
    *,
    name: str,
    version: str = DEFAULT_POLICY_VERSION,
    kinds: Iterable[str] = (),
    tables: Iterable[str] = (),
    owners: Iterable[str] = (),
    sources: Iterable[str] = (),
    datasets: Iterable[str] = (),
    models: Iterable[str] = (),
    deployments: Iterable[str] = (),
    projects: Iterable[str] = (),
    name_prefixes: Iterable[str] = (),
    artifact_ids: Iterable[str] = (),
    retain_until: datetime | str | None = None,
    retain_for_days: int | None = None,
    legal_hold: bool = False,
    audit_hold: bool = False,
    promotion_hold: bool = False,
    owner: str | None = None,
    reason_template: str | None = None,
) -> dict[str, Any]:
    """Build a canonical retention-policy definition dict (validated on record)."""

    scope = {
        "kinds": list(kinds),
        "tables": list(tables),
        "owners": list(owners),
        "sources": list(sources),
        "datasets": list(datasets),
        "models": list(models),
        "deployments": list(deployments),
        "projects": list(projects),
        "name_prefixes": list(name_prefixes),
        "artifact_ids": list(artifact_ids),
    }
    rules: dict[str, Any] = {
        "retain_until": _normalize_datetime(retain_until, "retain_until"),
        "retain_for_days": int(retain_for_days) if retain_for_days is not None else None,
        "legal_hold": bool(legal_hold),
        "audit_hold": bool(audit_hold),
        "promotion_hold": bool(promotion_hold),
    }
    return {
        "schema_version": DEFAULT_POLICY_SCHEMA,
        "name": name,
        "version": str(version),
        "scope": scope,
        "rules": rules,
        "owner": owner,
        "reason_template": reason_template,
    }


# --- Record / load / list ---------------------------------------------------


def record_retention_policy(
    lake: Lake,
    policy: Mapping[str, Any],
    *,
    status: str = DEFAULT_STATUS,
    actor: str | None = None,
    approver: str | None = None,
    note: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_by: str | None = None,
) -> RetentionPolicyCatalogEntry:
    """Record a retention policy in the durable catalog, idempotent by digest.

    Policy identity is the content digest of its name / version / scope / rules /
    owner / reason template, so re-recording an unchanged definition is a no-op
    that preserves the existing lifecycle (status, revision, approver,
    ``created_at``). Changing any of those fields yields a *new* immutable policy
    id -- an audit-friendly new revision rather than an in-place mutation.
    """

    canonical, digest = _canonicalize_policy(policy)

    existing = _load_row(lake, digest)
    if existing is not None:
        return _row_to_entry(existing)

    initial = _normalize_status(status)
    if initial == "active" and not (approver or "").strip():
        raise RetentionPolicyError("recording a policy as active requires approver=")

    now = datetime.now(UTC)
    rules = canonical["rules"]
    row = {
        "policy_id": digest,
        "policy_digest": digest,
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "policy_schema_version": canonical["schema_version"],
        "lake_uri": str(lake.uri),
        "name": canonical["name"],
        "version": canonical["version"],
        "scope_summary": _scope_summary(canonical["scope"]),
        "retain_until": _as_dt(rules.get("retain_until")),
        "retain_for_days": rules.get("retain_for_days"),
        "legal_hold": bool(rules.get("legal_hold")),
        "audit_hold": bool(rules.get("audit_hold")),
        "promotion_hold": bool(rules.get("promotion_hold")),
        "owner": canonical.get("owner") or "",
        "status": initial,
        "revision": 0,
        "actor": actor or "",
        "approver": approver or "",
        "activated_at": now if initial == "active" else None,
        "archived_at": None,
        "note": note or "",
        "policy_json": _encode_policy(canonical),
        "metadata": _kv_pairs(metadata),
        "created_by": created_by or "",
        "created_at": now,
        "updated_at": now,
    }
    _upsert_row(lake, row)
    _emit_event(
        lake,
        policy_id=digest,
        policy_digest=digest,
        event_type="recorded",
        from_status="",
        to_status=initial,
        revision=0,
        actor=actor,
        approver=approver,
        detail=f"recorded policy {canonical['name']!r} v{canonical['version']}",
        created_by=created_by,
    )
    return _row_to_entry(row)


def get_retention_policy(
    lake: Lake,
    policy_id_or_digest: str,
) -> tuple[RetentionPolicyCatalogEntry, dict[str, Any]]:
    """Reload a catalog entry and its stored policy definition by id or digest."""

    row = _load_row(lake, str(policy_id_or_digest))
    if row is None:
        raise RetentionPolicyError(
            f"no retention policy recorded for {policy_id_or_digest!r}"
        )
    return _row_to_entry(row), _decode_policy(row)


def retention_policies(
    lake: Lake,
    *,
    status: str | None = None,
    name: str | None = None,
    owner: str | None = None,
    page_size: int = 50,
    cursor: str | None = None,
) -> RetentionPolicyListPage:
    """List catalog entries newest-first, filtered and bounded by ``page_size``.

    ``status`` / ``name`` / ``owner`` push down to SQL. Results are ordered by
    ``(created_at desc, policy_id desc)`` and paged with an opaque cursor so a
    large catalog never loads in one shot.
    """

    if page_size < 1:
        raise RetentionPolicyError("page_size must be >= 1")

    predicates: list[str] = []
    if status is not None:
        predicates.append(f"status = {_sql_literal(_normalize_status(status))}")
    if name is not None:
        predicates.append(f"name = {_sql_literal(name)}")
    if owner is not None:
        predicates.append(f"owner = {_sql_literal(owner)}")

    rows = _load_rows_where(lake, " AND ".join(predicates) if predicates else None)
    ordered = sorted(
        rows,
        key=lambda item: (_as_dt(item.get("created_at")), str(item.get("policy_id"))),
        reverse=True,
    )
    start = _decode_cursor(cursor) if cursor else 0
    window = ordered[start : start + page_size]
    next_cursor = (
        _encode_cursor(start + page_size) if start + page_size < len(ordered) else None
    )
    return RetentionPolicyListPage(
        policies=tuple(_row_to_entry(row) for row in window),
        next_cursor=next_cursor,
        page_size=page_size,
    )


# --- Lifecycle / approvals --------------------------------------------------


def update_retention_policy_status(
    lake: Lake,
    policy_id: str,
    status: str,
    *,
    expected_status: str | None = None,
    expected_revision: int | None = None,
    actor: str | None = None,
    approver: str | None = None,
    note: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_by: str | None = None,
) -> RetentionPolicyCatalogEntry:
    """Move a recorded policy to a new lifecycle ``status`` (optimistically).

    Pass ``expected_revision`` and/or ``expected_status`` to guard against a
    concurrent update: a mismatch raises :class:`RetentionPolicyConflict`. The
    transition must be allowed (see :data:`STATUSES`); moving to ``active``
    requires an ``approver`` (here or already on the row). Each accepted update
    bumps ``revision`` and appends an audit event.
    """

    target = _normalize_status(status)
    row = _load_row(lake, str(policy_id))
    if row is None:
        raise RetentionPolicyError(f"no retention policy recorded for {policy_id!r}")

    current_status = str(row.get("status") or DEFAULT_STATUS)
    current_revision = int(row.get("revision") or 0)

    if expected_revision is not None and int(expected_revision) != current_revision:
        raise RetentionPolicyConflict(
            f"stale retention-policy update: {row['policy_id']} is at revision "
            f"{current_revision}, but expected_revision={expected_revision}; "
            "reload the policy and retry"
        )
    if expected_status is not None and _normalize_status(expected_status) != current_status:
        raise RetentionPolicyConflict(
            f"stale retention-policy update: {row['policy_id']} is in status "
            f"{current_status!r}, but expected_status={expected_status!r}; "
            "reload the policy and retry"
        )

    allowed = _TRANSITIONS.get(current_status, frozenset())
    if target == current_status:
        raise RetentionPolicyError(
            f"retention policy {row['policy_id']} is already in status {target!r}"
        )
    if target not in allowed:
        allowed_text = ", ".join(sorted(allowed)) or "(terminal state, no transitions)"
        raise RetentionPolicyError(
            f"illegal retention-policy transition {current_status!r} -> {target!r}; "
            f"allowed from {current_status!r}: {allowed_text}"
        )
    if target == "active" and not ((approver or "").strip() or str(row.get("approver") or "").strip()):
        raise RetentionPolicyError("activating a retention policy requires approver=")

    now = datetime.now(UTC)
    row["status"] = target
    row["revision"] = current_revision + 1
    if actor is not None:
        row["actor"] = actor
    if approver is not None:
        row["approver"] = approver
    if note is not None:
        row["note"] = note
    if metadata is not None:
        row["metadata"] = _kv_pairs(metadata)
    if target == "active":
        row["activated_at"] = now
    elif target == "archived":
        row["archived_at"] = now
    row["updated_at"] = now

    _upsert_row(lake, row)
    _emit_event(
        lake,
        policy_id=str(row["policy_id"]),
        policy_digest=str(row.get("policy_digest") or row["policy_id"]),
        event_type="status-changed",
        from_status=current_status,
        to_status=target,
        revision=int(row["revision"]),
        actor=actor if actor is not None else str(row.get("actor") or ""),
        approver=approver if approver is not None else str(row.get("approver") or ""),
        detail=note or f"{current_status} -> {target}",
        created_by=created_by,
    )
    return _row_to_entry(row)


# --- Apply / release --------------------------------------------------------


def apply_retention_policy(
    lake: Lake,
    policy_id: str,
    *,
    dry_run: bool = True,
    now: datetime | None = None,
    max_artifacts: int | None = None,
    actor: str | None = None,
    created_by: str | None = None,
    refresh: bool = True,
) -> RetentionPolicyApplication:
    """Expand an active policy into explicit artifact holds (idempotent).

    The policy's scope is resolved against ``lineage_artifacts`` and each matching
    artifact receives the policy's hold via the exact backlog 0067 metadata
    contract, so maintenance and reconciliation treat a policy-applied hold
    identically to a manual one. Artifacts that already carry a hold from a human
    or a *different* policy are conflicts: reported and left untouched (the
    artifact-local override wins). Re-applying an unchanged policy is a no-op.

    ``dry_run=True`` (the default) previews matches / conflicts without writing.
    ``dry_run=False`` requires the policy to be ``active``. ``max_artifacts``
    bounds the matched set, raising :class:`RetentionPolicyTooLarge` before any
    write.
    """

    entry, policy = get_retention_policy(lake, policy_id)
    if not dry_run and entry.status != "active":
        raise RetentionPolicyError(
            f"retention policy {entry.policy_id} must be active to apply "
            f"(current status: {entry.status!r}); activate it first or pass dry_run=True"
        )
    if refresh:
        lake.lineage.refresh_graph()

    reference = _utc_datetime(now) or datetime.now(UTC)
    marker = _policy_marker(entry.policy_id)
    reason = _reason_with_marker(entry.policy_id, policy.get("reason_template"))
    rules = policy.get("rules") or {}
    scope = policy.get("scope") or {}

    matched, bounded = _resolve_scope_artifacts(lake, scope, max_artifacts=max_artifacts)

    to_write: list[dict[str, Any]] = []
    applied_holds: list[dict[str, Any]] = []
    applied_ids: list[str] = []
    conflicts: list[dict[str, Any]] = []
    for row in matched:
        artifact_id = str(row.get("artifact_id") or "")
        existing = _retention_hold_from_artifact_row(lake.uri, row, now=reference)
        owning_policy = _policy_id_in_reason(existing.reason) if existing else None
        if existing is not None and owning_policy != entry.policy_id:
            conflicts.append(
                {
                    "artifact_id": artifact_id,
                    "kind": row.get("kind"),
                    "table_name": row.get("table_name"),
                    "existing_source": (
                        f"policy:{owning_policy}" if owning_policy else "artifact-local"
                    ),
                    "existing_reason": existing.reason,
                    "resolution": "artifact-local override wins; policy hold not applied",
                }
            )
            continue
        retain_until = _retain_until_for(row, rules, reference)
        hold = LineageRetentionHold(
            lake_uri=lake.uri,
            artifact_ids=(artifact_id,),
            retain_until=retain_until,
            legal_hold=bool(rules.get("legal_hold")),
            audit_hold=bool(rules.get("audit_hold")),
            promotion_hold=bool(rules.get("promotion_hold")),
            owner=policy.get("owner") or entry.owner or None,
            reason=reason,
            active=True,
            created_at=reference,
        )
        applied_ids.append(artifact_id)
        applied_holds.append(hold.as_dict())
        if not dry_run:
            to_write.append(_row_with_hold(row, hold))

    if not dry_run and to_write:
        _replace_rows(lake, "lineage_artifacts", "artifact_id", to_write, LINEAGE_ARTIFACTS_SCHEMA)
        _emit_event(
            lake,
            policy_id=entry.policy_id,
            policy_digest=entry.policy_digest,
            event_type="applied",
            from_status=entry.status,
            to_status=entry.status,
            revision=entry.revision,
            actor=actor,
            approver=None,
            artifact_count=len(applied_ids),
            detail=f"applied hold to {len(applied_ids)} artifact(s); {len(conflicts)} conflict(s)",
            created_by=created_by,
            metadata={"marker": marker},
        )

    return RetentionPolicyApplication(
        policy_id=entry.policy_id,
        policy_digest=entry.policy_digest,
        name=entry.name,
        version=entry.version,
        operation="apply",
        dry_run=dry_run,
        evaluated_at=reference,
        matched_count=len(matched),
        applied_count=len(applied_ids),
        conflict_count=len(conflicts),
        applied_artifact_ids=tuple(sorted(applied_ids)),
        conflicts=tuple(sorted(conflicts, key=lambda item: item["artifact_id"])),
        holds=tuple(applied_holds),
        scope_bounded=bounded,
    )


def release_retention_policy(
    lake: Lake,
    policy_id: str,
    *,
    dry_run: bool = True,
    actor: str | None = None,
    created_by: str | None = None,
    refresh: bool = True,
) -> RetentionPolicyApplication:
    """Clear the holds this policy applied, leaving manual / other-policy holds.

    Only artifacts whose current hold ``reason`` carries this policy's marker are
    released, so a manually set hold or another policy's hold is never touched.
    ``dry_run=True`` previews the release without writing.
    """

    entry, _policy = get_retention_policy(lake, policy_id)
    if refresh:
        lake.lineage.refresh_graph()

    released_ids: list[str] = []
    to_write: list[dict[str, Any]] = []
    for row in _iter_artifact_rows(lake, where=None):
        existing = _retention_hold_from_artifact_row(lake.uri, row)
        if existing is None or _policy_id_in_reason(existing.reason) != entry.policy_id:
            continue
        artifact_id = str(row.get("artifact_id") or "")
        released_ids.append(artifact_id)
        if not dry_run:
            to_write.append(_row_without_hold(row))

    if not dry_run and to_write:
        _replace_rows(lake, "lineage_artifacts", "artifact_id", to_write, LINEAGE_ARTIFACTS_SCHEMA)
        _emit_event(
            lake,
            policy_id=entry.policy_id,
            policy_digest=entry.policy_digest,
            event_type="released",
            from_status=entry.status,
            to_status=entry.status,
            revision=entry.revision,
            actor=actor,
            approver=None,
            artifact_count=len(released_ids),
            detail=f"released hold from {len(released_ids)} artifact(s)",
            created_by=created_by,
        )

    return RetentionPolicyApplication(
        policy_id=entry.policy_id,
        policy_digest=entry.policy_digest,
        name=entry.name,
        version=entry.version,
        operation="release",
        dry_run=dry_run,
        evaluated_at=datetime.now(UTC),
        matched_count=len(released_ids),
        applied_count=0 if dry_run else len(released_ids),
        conflict_count=0,
        applied_artifact_ids=tuple(sorted(released_ids)),
        conflicts=(),
        holds=(),
        scope_bounded=True,
    )


# --- Resolution / expiration ------------------------------------------------


def resolve_retention_holds(
    lake: Lake,
    *,
    now: datetime | None = None,
    include_snapshot: bool = True,
    detect_conflicts: bool = True,
) -> RetentionHoldResolution:
    """Merge policy-applied + artifact-local holds into maintenance's pin shape.

    The returned ``pins`` are exactly what maintenance pins: the merge of
    :func:`lineage_retention_pin_details` (materialized artifact holds) and,
    when ``include_snapshot`` is set, :func:`snapshot_retention_pin_details`.
    Each active hold is also classified by source (``policy:<id>`` vs
    ``artifact-local``). When ``detect_conflicts`` is set, active catalog
    policies whose scope covers an artifact-local hold are reported as
    deterministic shadowing diagnostics.
    """

    reference = _utc_datetime(now) or datetime.now(UTC)

    holds: list[dict[str, Any]] = []
    policy_count = 0
    local_count = 0
    local_by_artifact: dict[str, dict[str, Any]] = {}
    for row in _iter_artifact_rows(lake, where=None):
        hold = _retention_hold_from_artifact_row(lake.uri, row, now=reference)
        if hold is None or not hold.active:
            continue
        owning_policy = _policy_id_in_reason(hold.reason)
        source = f"policy:{owning_policy}" if owning_policy else "artifact-local"
        if owning_policy:
            policy_count += 1
        else:
            local_count += 1
            local_by_artifact[str(row.get("artifact_id") or "")] = row
        holds.append(
            {
                "artifact_id": str(row.get("artifact_id") or ""),
                "kind": row.get("kind"),
                "table_name": row.get("table_name"),
                "table_version": row.get("table_version"),
                "source": source,
                "policy_id": owning_policy,
                "hold": hold.as_dict(),
            }
        )

    pin_map = lineage_retention_pin_details(lake, now=reference)
    if include_snapshot:
        pin_map = merge_retention_pin_details(pin_map, snapshot_retention_pin_details(lake))
    pins = retention_pin_rows(pin_map)

    conflicts: list[dict[str, Any]] = []
    considered: list[str] = []
    if detect_conflicts and local_by_artifact:
        active_policies = _load_rows_where(lake, f"status = {_sql_literal('active')}")
        for policy_row in active_policies:
            considered.append(str(policy_row.get("policy_id") or ""))
            scope = (_decode_policy(policy_row).get("scope")) or {}
            for artifact_id, row in local_by_artifact.items():
                if _artifact_matches_scope(row, _metadata_map(row), scope):
                    conflicts.append(
                        {
                            "artifact_id": artifact_id,
                            "policy_id": str(policy_row.get("policy_id") or ""),
                            "policy_name": str(policy_row.get("name") or ""),
                            "resolution": (
                                "artifact-local hold overrides active policy; "
                                "apply will skip this artifact"
                            ),
                        }
                    )

    return RetentionHoldResolution(
        lake_uri=lake.uri,
        hold_count=len(holds),
        policy_hold_count=policy_count,
        artifact_local_count=local_count,
        holds=tuple(sorted(holds, key=lambda item: item["artifact_id"])),
        pins=pins,
        conflicts=tuple(sorted(conflicts, key=lambda item: (item["artifact_id"], item["policy_id"]))),
        policies_considered=tuple(sorted(set(considered))),
    )


def retention_expiration_notices(
    lake: Lake,
    *,
    within: timedelta | None = None,
    now: datetime | None = None,
    notify: bool = False,
    created_by: str | None = None,
) -> list[dict[str, Any]]:
    """Report holds whose ``retain_until`` has passed or falls within ``within``.

    Read-only by default. Legal/audit/promotion (indefinite) holds never expire
    and are excluded. When ``notify`` is set, an append-safe
    ``expiration-notified`` audit event is emitted per owning policy.
    """

    reference = _utc_datetime(now) or datetime.now(UTC)
    horizon = reference + within if within is not None else None
    notices: list[dict[str, Any]] = []
    for row in _iter_artifact_rows(lake, where=None):
        hold = _retention_hold_from_artifact_row(lake.uri, row, now=reference)
        if hold is None or hold.retain_until is None:
            continue
        if hold.legal_hold or hold.audit_hold or hold.promotion_hold:
            continue
        retain_until = _utc_datetime(hold.retain_until)
        expired = retain_until <= reference
        upcoming = horizon is not None and reference < retain_until <= horizon
        if not (expired or upcoming):
            continue
        notices.append(
            {
                "artifact_id": str(row.get("artifact_id") or ""),
                "table_name": row.get("table_name"),
                "table_version": row.get("table_version"),
                "retain_until": retain_until.isoformat(),
                "expired": expired,
                "policy_id": _policy_id_in_reason(hold.reason),
                "owner": hold.owner,
            }
        )

    notices.sort(key=lambda item: (item["retain_until"], item["artifact_id"]))
    if notify and notices:
        by_policy: dict[str, int] = {}
        for notice in notices:
            by_policy[notice.get("policy_id") or ""] = by_policy.get(notice.get("policy_id") or "", 0) + 1
        for policy_id, count in sorted(by_policy.items()):
            if not policy_id:
                continue
            row = _load_row(lake, policy_id)
            if row is None:
                continue
            _emit_event(
                lake,
                policy_id=policy_id,
                policy_digest=str(row.get("policy_digest") or policy_id),
                event_type="expiration-notified",
                from_status=str(row.get("status") or ""),
                to_status=str(row.get("status") or ""),
                revision=int(row.get("revision") or 0),
                actor=None,
                approver=None,
                artifact_count=count,
                detail=f"{count} hold(s) expiring/expired",
                created_by=created_by,
            )
    return notices


# --- Governance projection --------------------------------------------------


def export_retention_policy_state(
    lake: Lake,
    *,
    policy_id: str | None = None,
    status: str | None = None,
    include_holds: bool = True,
    now: datetime | None = None,
) -> RetentionGovernanceProjection:
    """Project policy + resolved-hold state out for external governance systems.

    Deterministic given lake state (no injected timestamp), carries no secrets,
    and adds no mandatory dependency. Filter by ``policy_id`` or ``status``.
    """

    if policy_id is not None:
        rows = [row for row in [_load_row(lake, str(policy_id))] if row is not None]
    else:
        predicate = f"status = {_sql_literal(_normalize_status(status))}" if status else None
        rows = _load_rows_where(lake, predicate)
    ordered = sorted(
        rows,
        key=lambda item: (_as_dt(item.get("created_at")), str(item.get("policy_id"))),
        reverse=True,
    )
    policies = tuple(_row_to_entry(row).as_dict() for row in ordered)

    holds: tuple[dict[str, Any], ...] = ()
    pins: tuple[dict[str, Any], ...] = ()
    if include_holds:
        resolution = resolve_retention_holds(lake, now=now, detect_conflicts=False)
        holds = resolution.holds
        pins = resolution.pins

    return RetentionGovernanceProjection(
        schema_version=GOVERNANCE_SCHEMA_VERSION,
        lake_uri=lake.uri,
        policies=policies,
        holds=holds,
        pins=pins,
    )


def project_retention_state(
    lake: Lake,
    sink: RetentionGovernanceSink,
    *,
    policy_id: str | None = None,
    status: str | None = None,
    include_holds: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a governance projection and hand it to a caller-supplied ``sink``.

    The sink owns its transport and credentials at runtime; nothing from it is
    persisted. Returns a small receipt describing what was projected.
    """

    try:
        project = sink.project
    except AttributeError as exc:
        raise RetentionPolicyError(
            "governance sink must implement project(projection)"
        ) from exc
    if not callable(project):
        raise RetentionPolicyError("governance sink must implement project(projection)")
    projection = export_retention_policy_state(
        lake,
        policy_id=policy_id,
        status=status,
        include_holds=include_holds,
        now=now,
    )
    receipt = sink.project(projection)
    return {
        "projected_policies": len(projection.policies),
        "projected_holds": len(projection.holds),
        "receipt": dict(receipt) if isinstance(receipt, Mapping) else receipt,
    }


def retention_policy_events(
    lake: Lake,
    *,
    policy_id: str | None = None,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return audit events, optionally filtered by policy id / event type."""

    predicates: list[str] = []
    if policy_id is not None:
        predicates.append(f"policy_id = {_sql_literal(policy_id)}")
    if event_type is not None:
        predicates.append(f"event_type = {_sql_literal(event_type)}")
    handle = lake.table(EVENTS_TABLE).to_lance()
    where = " AND ".join(predicates) if predicates else None
    arrow = handle.to_table(filter=where) if where else handle.to_table()
    rows = arrow.to_pylist()
    rows.sort(key=lambda item: (_as_dt(item.get("created_at")), str(item.get("event_id"))))
    return _json_ready(rows)


# --- Scope resolution -------------------------------------------------------


def _resolve_scope_artifacts(
    lake: Lake,
    scope: Mapping[str, Any],
    *,
    max_artifacts: int | None,
) -> tuple[list[dict[str, Any]], bool]:
    """Return artifact rows matching ``scope`` (bounded scan, guardrailed).

    A coarse predicate on kind / table / source is pushed down so a scoped
    policy never scans the whole graph; the remaining selectors are matched in
    Python. Returns ``(matched_rows, scope_bounded)`` where ``scope_bounded`` is
    True when a pushdown predicate limited the scan. Raises
    :class:`RetentionPolicyTooLarge` if the match exceeds ``max_artifacts``.
    """

    where, bounded = _scope_pushdown(scope)
    matched: list[dict[str, Any]] = []
    for row in _iter_artifact_rows(lake, where=where):
        metadata = _metadata_map(row)
        if not _artifact_matches_scope(row, metadata, scope):
            continue
        matched.append(row)
        if max_artifacts is not None and len(matched) > max_artifacts:
            raise RetentionPolicyTooLarge(
                f"policy scope matched more than max_artifacts={max_artifacts} "
                "artifacts; narrow the scope or raise the guardrail"
            )
    return matched, bounded


def _scope_pushdown(scope: Mapping[str, Any]) -> tuple[str | None, bool]:
    """Build a coarse SQL predicate for the scope's kind/table/artifact-id sets."""

    clauses: list[str] = []
    kinds = _string_list(scope.get("kinds"))
    tables = _string_list(scope.get("tables"))
    artifact_ids = _string_list(scope.get("artifact_ids"))
    if kinds:
        clauses.append("kind IN (" + ", ".join(_sql_literal(v) for v in kinds) + ")")
    if tables:
        clauses.append("table_name IN (" + ", ".join(_sql_literal(v) for v in tables) + ")")
    if artifact_ids:
        clauses.append("artifact_id IN (" + ", ".join(_sql_literal(v) for v in artifact_ids) + ")")
    if not clauses:
        return None, False
    # OR the coarse clauses so the pushdown is a superset of the Python match
    # (which ANDs categories); the Python pass then applies the exact semantics.
    return " OR ".join(clauses), True


def _artifact_matches_scope(
    row: Mapping[str, Any],
    metadata: Mapping[str, str],
    scope: Mapping[str, Any],
) -> bool:
    """AND across specified selector categories; OR within each category."""

    kinds = _string_list(scope.get("kinds"))
    if kinds and str(row.get("kind") or "") not in kinds:
        return False
    tables = _string_list(scope.get("tables"))
    if tables and str(row.get("table_name") or "") not in tables:
        return False
    artifact_ids = _string_list(scope.get("artifact_ids"))
    if artifact_ids and str(row.get("artifact_id") or "") not in artifact_ids:
        return False
    sources = _string_list(scope.get("sources"))
    if sources:
        source_uri = str(row.get("source_uri") or "")
        if not source_uri or not any(
            source_uri == value or source_uri.startswith(value) for value in sources
        ):
            return False
    name_prefixes = _string_list(scope.get("name_prefixes"))
    if name_prefixes:
        name = str(row.get("name") or "")
        if not any(name.startswith(prefix) for prefix in name_prefixes):
            return False
    name = str(row.get("name") or "")
    for scope_key, meta_key in _METADATA_SCOPE_KEYS.items():
        values = _string_list(scope.get(scope_key))
        if not values:
            continue
        candidate = metadata.get(meta_key) or ""
        if candidate in values:
            continue
        # datasets/models/... also match the artifact name for convenience.
        if scope_key in {"datasets", "models", "deployments", "projects"} and name in values:
            continue
        return False
    return True


def _iter_artifact_rows(lake: Lake, *, where: str | None):
    """Yield ``lineage_artifacts`` rows in bounded batches (all columns)."""

    dataset = lake.table("lineage_artifacts").to_lance()
    scanner = dataset.scanner(filter=where, batch_size=_SCAN_BATCH_ROWS)
    for batch in scanner.to_batches():
        yield from batch.to_pylist()


# --- Hold materialization helpers -------------------------------------------


def _retain_until_for(
    row: Mapping[str, Any],
    rules: Mapping[str, Any],
    reference: datetime,
) -> datetime | None:
    """Compute an artifact's ``retain_until`` from the policy rules.

    An absolute ``retain_until`` wins; otherwise ``retain_for_days`` is measured
    from the artifact's ``created_at`` (falling back to ``reference``).
    """

    absolute = _as_dt(rules.get("retain_until"))
    if absolute is not None:
        return absolute
    days = rules.get("retain_for_days")
    if days is None:
        return None
    base = _as_dt(row.get("created_at")) or reference
    return base + timedelta(days=int(days))


def _row_with_hold(row: Mapping[str, Any], hold: LineageRetentionHold) -> dict[str, Any]:
    """Return a copy of ``row`` with the hold's retention metadata merged in."""

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
    updated = dict(row)
    updated["metadata"] = _kv_items(metadata)
    return updated


def _row_without_hold(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``row`` with all retention metadata keys removed."""

    metadata = _metadata_map(row)
    for key in _RETENTION_KEYS:
        metadata.pop(key, None)
    updated = dict(row)
    updated["metadata"] = _kv_items(metadata)
    return updated


# --- Reason markers ---------------------------------------------------------


def _policy_marker(policy_id: str) -> str:
    return f"[{_POLICY_MARKER_PREFIX}{policy_id}]"


def _reason_with_marker(policy_id: str, template: str | None) -> str:
    marker = _policy_marker(policy_id)
    body = (template or "").strip()
    return f"{marker} {body}".strip() if body else marker


def _policy_id_in_reason(reason: str | None) -> str | None:
    text = str(reason or "")
    prefix = f"[{_POLICY_MARKER_PREFIX}"
    if not text.startswith(prefix):
        return None
    end = text.find("]")
    if end == -1:
        return None
    return text[len(prefix):end] or None


# --- Policy canonicalization ------------------------------------------------


def _canonicalize_policy(policy: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Validate + normalize a policy definition and return (canonical, digest)."""

    if hasattr(policy, "as_dict") and callable(policy.as_dict):
        policy = dict(policy.as_dict())
    if not isinstance(policy, Mapping):
        raise RetentionPolicyError("policy must be a mapping or expose as_dict()")

    schema = policy.get("schema_version") or DEFAULT_POLICY_SCHEMA
    if schema not in SUPPORTED_POLICY_SCHEMAS:
        raise RetentionPolicyError(
            f"unsupported retention-policy schema {schema!r}; expected one of "
            f"{SUPPORTED_POLICY_SCHEMAS}"
        )

    name = str(policy.get("name") or "").strip()
    if not name:
        raise RetentionPolicyError("policy name is required")
    version = str(policy.get("version") or DEFAULT_POLICY_VERSION).strip()

    raw_scope = policy.get("scope") or {}
    scope = {key: _string_list(raw_scope.get(key)) for key in SCOPE_KEYS}
    if not any(scope[key] for key in SCOPE_KEYS):
        raise RetentionPolicyError(
            "policy scope must specify at least one selector "
            f"(one of: {', '.join(SCOPE_KEYS)})"
        )

    raw_rules = policy.get("rules") or {}
    retain_until = _normalize_datetime(raw_rules.get("retain_until"), "retain_until")
    retain_for_days = raw_rules.get("retain_for_days")
    if retain_for_days is not None:
        retain_for_days = int(retain_for_days)
        if retain_for_days <= 0:
            raise RetentionPolicyError("retain_for_days must be a positive integer")
    legal_hold = bool(raw_rules.get("legal_hold"))
    audit_hold = bool(raw_rules.get("audit_hold"))
    promotion_hold = bool(raw_rules.get("promotion_hold"))
    if not (retain_until or retain_for_days or legal_hold or audit_hold or promotion_hold):
        raise RetentionPolicyError(
            "policy rules must define at least one hold (retain_until, "
            "retain_for_days, or a legal/audit/promotion hold)"
        )

    canonical = {
        "schema_version": str(schema),
        "name": name,
        "version": version,
        "scope": scope,
        "rules": {
            "retain_until": _utc_datetime(retain_until).isoformat() if retain_until else None,
            "retain_for_days": retain_for_days,
            "legal_hold": legal_hold,
            "audit_hold": audit_hold,
            "promotion_hold": promotion_hold,
        },
        "owner": (str(policy.get("owner")).strip() or None) if policy.get("owner") else None,
        "reason_template": (
            str(policy.get("reason_template")) if policy.get("reason_template") else None
        ),
    }
    digest = _full_digest(canonical)
    return canonical, digest


def _scope_summary(scope: Mapping[str, Any]) -> str:
    parts = []
    for key in SCOPE_KEYS:
        values = _string_list(scope.get(key))
        if values:
            parts.append(f"{key}=[{','.join(values)}]")
    return ";".join(parts)


def _encode_policy(canonical: Mapping[str, Any]) -> str:
    return json.dumps(_json_ready(canonical), sort_keys=True, separators=(",", ":"))


def _decode_policy(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("policy_json")
    if not raw:
        raise RetentionPolicyError(
            f"catalog row {row.get('policy_id')!r} has no stored policy definition"
        )
    return json.loads(raw)


# --- Table IO ---------------------------------------------------------------


def _upsert_row(lake: Lake, row: Mapping[str, Any]) -> None:
    table = pa.Table.from_pylist([dict(row)], schema=RETENTION_POLICIES_SCHEMA)
    (
        lake.table(CATALOG_TABLE)
        .merge_insert("policy_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(table)
    )


def _emit_event(
    lake: Lake,
    *,
    policy_id: str,
    policy_digest: str,
    event_type: str,
    from_status: str,
    to_status: str,
    revision: int,
    actor: str | None,
    approver: str | None,
    detail: str,
    created_by: str | None,
    artifact_count: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    now = datetime.now(UTC)
    event_id = hashlib.sha256(
        json.dumps(
            [policy_id, event_type, from_status, to_status, int(revision), int(artifact_count), now.isoformat()],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    event = {
        "event_id": event_id,
        "policy_id": policy_id,
        "policy_digest": policy_digest,
        "event_type": event_type,
        "from_status": from_status,
        "to_status": to_status,
        "revision": int(revision),
        "artifact_count": int(artifact_count),
        "actor": actor or "",
        "approver": approver or "",
        "detail": detail,
        "metadata": _kv_pairs(metadata),
        "created_by": created_by or "",
        "created_at": now,
    }
    table = pa.Table.from_pylist([event], schema=RETENTION_POLICY_EVENTS_SCHEMA)
    lake.table(EVENTS_TABLE).add(table)


def _load_row(lake: Lake, policy_id_or_digest: str) -> dict[str, Any] | None:
    literal = _sql_literal(policy_id_or_digest)
    rows = _load_rows_where(lake, f"policy_id = {literal} OR policy_digest = {literal}")
    return rows[0] if rows else None


def _load_rows_where(lake: Lake, where: str | None) -> list[dict[str, Any]]:
    handle = lake.table(CATALOG_TABLE).to_lance()
    arrow = handle.to_table(filter=where) if where else handle.to_table()
    return arrow.to_pylist()


def _row_to_entry(row: Mapping[str, Any]) -> RetentionPolicyCatalogEntry:
    return RetentionPolicyCatalogEntry(
        policy_id=str(row.get("policy_id") or ""),
        policy_digest=str(row.get("policy_digest") or ""),
        catalog_schema_version=str(row.get("catalog_schema_version") or CATALOG_SCHEMA_VERSION),
        policy_schema_version=str(row.get("policy_schema_version") or DEFAULT_POLICY_SCHEMA),
        lake_uri=str(row.get("lake_uri") or ""),
        name=str(row.get("name") or ""),
        version=str(row.get("version") or DEFAULT_POLICY_VERSION),
        scope_summary=str(row.get("scope_summary") or ""),
        retain_until=_as_dt(row.get("retain_until")),
        retain_for_days=(int(row["retain_for_days"]) if row.get("retain_for_days") is not None else None),
        legal_hold=bool(row.get("legal_hold")),
        audit_hold=bool(row.get("audit_hold")),
        promotion_hold=bool(row.get("promotion_hold")),
        owner=str(row.get("owner") or ""),
        status=str(row.get("status") or DEFAULT_STATUS),
        revision=int(row.get("revision") or 0),
        actor=str(row.get("actor") or ""),
        approver=str(row.get("approver") or ""),
        activated_at=_as_dt(row.get("activated_at")),
        archived_at=_as_dt(row.get("archived_at")),
        note=str(row.get("note") or ""),
        metadata=_kv_to_dict(row.get("metadata")),
        created_by=str(row.get("created_by") or ""),
        created_at=_as_dt(row.get("created_at")) or datetime.now(UTC),
        updated_at=_as_dt(row.get("updated_at")) or datetime.now(UTC),
    )


# --- Small shared utilities -------------------------------------------------


def _normalize_status(status: str) -> str:
    value = str(status or "").strip().lower()
    if value not in STATUSES:
        raise RetentionPolicyError(
            f"unknown retention-policy status {status!r}; expected one of {', '.join(STATUSES)}"
        )
    return value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return result
    return []


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


def _kv_pairs(metadata: Mapping[str, Any] | None) -> list[dict[str, str]]:
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
        raise RetentionPolicyError(f"invalid cursor {cursor!r}") from exc
