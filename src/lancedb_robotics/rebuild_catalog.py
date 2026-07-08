"""Durable rebuild-plan catalog, approvals, and orchestrator handoff (0109).

Backlog 0066 emits deterministic :class:`~lancedb_robotics.lineage.RebuildPlan`
reports and can record invalidation markers, but the plans themselves are
transient: once the process exits, the ordered action list is gone. Production
platform teams need to look a plan up later, move it through an approval
lifecycle, and hand its actions to an external orchestrator (Airflow, Dagster,
Ray, Batch, Slurm, ...) with stable, retry-safe ids. This module adds:

- a persisted catalog table (:data:`CATALOG_TABLE`) with one idempotent row per
  plan, keyed by ``plan_digest`` (== ``plan_id``). The digest is content-addressed
  over the plan's roots, reason, severity, and ordered actions -- and is
  independent of any invalidation timestamp -- so re-planning the same
  invalidation on the same lake state re-records to the same row. The ordered
  action list is stored inline (``plan_json``), so a plan reloads and exports for
  orchestrators without re-tracing the lineage graph;
- a generic approval lifecycle (``draft`` -> ``approved`` -> ``dispatched`` ->
  ``completed`` / ``failed``, plus ``abandoned``) guarded by a ``revision``
  optimistic-concurrency counter, so a status update made against a stale view is
  rejected with an actionable diagnostic instead of silently clobbering;
- an append-only audit log (:data:`EVENTS_TABLE`) for recording, status
  transitions, and dispatch exports that survives status churn;
- deterministic orchestrator handoff: :func:`export_rebuild_plan_dispatch` emits a
  payload (dict or NDJSON) with stable per-action ids, dependency ids resolved to
  action ids, target artifact ids, table-version pins, and retry-safe external run
  references. Re-exporting an unchanged plan yields byte-identical payloads.

Org-specific authorization stays out of OSS core: the ``actor`` / ``approver`` /
``note`` columns are free-form metadata, and enforcing *who* may approve belongs
in an enterprise/plugin layer.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import LineageError, RebuildPlan
from lancedb_robotics.schemas import (
    REBUILD_PLAN_EVENTS_SCHEMA,
    REBUILD_PLANS_SCHEMA,
)

CATALOG_TABLE = "rebuild_plans"
EVENTS_TABLE = "rebuild_plan_events"

#: Catalog row contract version (independent of the plan payload schema).
CATALOG_SCHEMA_VERSION = "lancedb-robotics/rebuild-plan-catalog/v1"

#: Plan payloads this catalog can record / reload.
SUPPORTED_PLAN_SCHEMAS: tuple[str, ...] = ("lancedb-robotics/rebuild-plan/v1",)

#: Dispatch payload contract version.
DISPATCH_SCHEMA_VERSION = "lancedb-robotics/rebuild-dispatch/v1"

DEFAULT_STATUS = "draft"
DEFAULT_RUN_REF_PREFIX = "rebuild"

#: Lifecycle states in canonical order.
STATUSES: tuple[str, ...] = (
    "draft",
    "approved",
    "dispatched",
    "completed",
    "failed",
    "abandoned",
)

#: Allowed status transitions. Terminal states (``completed`` / ``abandoned``)
#: have no outgoing edges; ``failed`` can be retried back to ``dispatched``.
_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"approved", "abandoned"}),
    "approved": frozenset({"dispatched", "draft", "abandoned"}),
    "dispatched": frozenset({"completed", "failed", "abandoned"}),
    "failed": frozenset({"dispatched", "abandoned"}),
    "completed": frozenset(),
    "abandoned": frozenset(),
}


class RebuildPlanCatalogError(LineageError):
    """A rebuild-plan catalog operation could not be completed."""


class RebuildPlanConflict(RebuildPlanCatalogError):
    """An optimistic-concurrency check rejected a stale status update."""


# --- Catalog entry / reports ------------------------------------------------


@dataclass(frozen=True)
class RebuildPlanCatalogEntry:
    """A single durable rebuild-plan catalog row (ordered actions kept inline)."""

    plan_id: str
    plan_digest: str
    catalog_schema_version: str
    plan_schema_version: str
    lake_uri: str
    invalidation_id: str
    root_artifact_ids: tuple[str, ...]
    action_count: int
    reason: str
    severity: str
    status: str
    revision: int
    actor: str
    approver: str
    approved_at: datetime | None
    dispatched_at: datetime | None
    completed_at: datetime | None
    note: str
    table_version_pins: tuple[dict[str, Any], ...]
    metadata: dict[str, str]
    created_by: str
    created_at: datetime
    updated_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "catalog_schema_version": self.catalog_schema_version,
            "plan_schema_version": self.plan_schema_version,
            "lake_uri": self.lake_uri,
            "invalidation_id": self.invalidation_id,
            "root_artifact_ids": list(self.root_artifact_ids),
            "action_count": self.action_count,
            "reason": self.reason,
            "severity": self.severity,
            "status": self.status,
            "revision": self.revision,
            "actor": self.actor,
            "approver": self.approver,
            "approved_at": _iso_or_none(self.approved_at),
            "dispatched_at": _iso_or_none(self.dispatched_at),
            "completed_at": _iso_or_none(self.completed_at),
            "note": self.note,
            "table_version_pins": [dict(row) for row in self.table_version_pins],
            "metadata": dict(self.metadata),
            "created_by": self.created_by,
            "created_at": _iso_or_none(self.created_at),
            "updated_at": _iso_or_none(self.updated_at),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class RebuildPlanListPage:
    """A bounded page of catalog entries plus an opaque continuation cursor."""

    plans: tuple[RebuildPlanCatalogEntry, ...]
    next_cursor: str | None
    page_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "plans": [entry.as_dict() for entry in self.plans],
            "next_cursor": self.next_cursor,
            "page_size": self.page_size,
            "count": len(self.plans),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class RebuildPlanDispatch:
    """A deterministic orchestrator handoff payload for a recorded plan."""

    schema_version: str
    plan_id: str
    plan_digest: str
    lake_uri: str
    orchestrator: str
    invalidation_id: str
    root_artifact_ids: tuple[str, ...]
    reason: str
    severity: str
    actions: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "lake_uri": self.lake_uri,
            "orchestrator": self.orchestrator,
            "invalidation_id": self.invalidation_id,
            "root_artifact_ids": list(self.root_artifact_ids),
            "reason": self.reason,
            "severity": self.severity,
            "actions": [dict(action) for action in self.actions],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()

    def iter_ndjson(self):
        """Yield one self-describing JSON line per action (stable order)."""
        for action in self.actions:
            line = {
                "plan_id": self.plan_id,
                "plan_digest": self.plan_digest,
                "orchestrator": self.orchestrator,
                "action": dict(action),
            }
            yield json.dumps(line, sort_keys=True, separators=(",", ":"))

    def as_ndjson(self) -> str:
        return "\n".join(self.iter_ndjson())


# --- Record / load / list ---------------------------------------------------


def record_rebuild_plan(
    lake: Lake,
    plan: RebuildPlan | Mapping[str, Any],
    *,
    status: str = DEFAULT_STATUS,
    actor: str | None = None,
    approver: str | None = None,
    note: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_by: str | None = None,
) -> RebuildPlanCatalogEntry:
    """Record a rebuild plan in the durable catalog, idempotent by digest.

    ``plan`` is a :class:`~lancedb_robotics.lineage.RebuildPlan` or a plan mapping
    (as produced by :meth:`RebuildPlan.as_dict`). Plan identity is the content
    digest of its roots/reason/severity/actions, so re-recording an unchanged plan
    is a no-op that preserves the existing lifecycle (status, revision, approver,
    ``created_at``) instead of resetting an already-approved plan back to draft.
    """

    plan_dict = _plan_of(plan)
    schema_version = _require_supported_schema(plan_dict)
    stored, digest, roots = _stored_plan(plan_dict, schema_version)

    existing = _load_row(lake, digest)
    if existing is not None:
        # Idempotent: an identical plan keeps its lifecycle untouched.
        return _row_to_entry(existing)

    initial = _normalize_status(status)
    if initial == "approved" and not (approver or "").strip():
        raise RebuildPlanCatalogError("recording a plan as approved requires approver=")

    now = datetime.now(UTC)
    invalidation = plan_dict.get("invalidation") or {}
    row = {
        "plan_id": digest,
        "plan_digest": digest,
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "plan_schema_version": schema_version,
        "lake_uri": str(plan_dict.get("lake_uri") or lake.uri),
        "invalidation_id": str(invalidation.get("invalidation_id") or ""),
        "root_artifact_ids": list(roots),
        "action_count": len(stored["actions"]),
        "reason": str(plan_dict.get("reason") or ""),
        "severity": str(plan_dict.get("severity") or ""),
        "status": initial,
        "revision": 0,
        "actor": actor or "",
        "approver": approver or "",
        "approved_at": now if initial == "approved" else None,
        "dispatched_at": None,
        "completed_at": None,
        "note": note or "",
        "plan_json": _encode_plan(stored),
        "table_version_pins": _version_pins(stored["actions"]),
        "metadata": _kv_items(metadata),
        "created_by": created_by or "",
        "created_at": now,
        "updated_at": now,
    }
    _upsert_row(lake, row)
    _emit_event(
        lake,
        plan_id=digest,
        plan_digest=digest,
        event_type="recorded",
        from_status="",
        to_status=initial,
        revision=0,
        actor=actor,
        approver=approver,
        detail=f"recorded {len(stored['actions'])} action(s)",
        created_by=created_by,
        metadata={"invalidation_id": row["invalidation_id"]} if row["invalidation_id"] else {},
    )
    return _row_to_entry(row)


def get_rebuild_plan(
    lake: Lake,
    plan_id_or_digest: str,
) -> tuple[RebuildPlanCatalogEntry, dict[str, Any]]:
    """Reload a catalog entry and its stored plan payload by id or digest.

    The plan payload (roots, ordered actions, any invalidation marker) is stored
    inline, so this never re-traces the lineage graph.
    """

    row = _load_row(lake, str(plan_id_or_digest))
    if row is None:
        raise RebuildPlanCatalogError(
            f"no rebuild plan recorded for {plan_id_or_digest!r}"
        )
    plan_dict = _decode_plan(row)
    _require_supported_schema(plan_dict)
    return _row_to_entry(row), plan_dict


def rebuild_plans(
    lake: Lake,
    *,
    status: str | None = None,
    invalidation_id: str | None = None,
    root_artifact_id: str | None = None,
    page_size: int = 50,
    cursor: str | None = None,
) -> RebuildPlanListPage:
    """List catalog entries newest-first, filtered and bounded by ``page_size``.

    ``status`` / ``invalidation_id`` push down to SQL; ``root_artifact_id`` is
    matched against the inline root list in Python. Results are ordered by
    ``(created_at desc, plan_id desc)`` and paged with an opaque cursor so large
    catalogs never load in one shot.
    """

    if page_size < 1:
        raise RebuildPlanCatalogError("page_size must be >= 1")

    predicates: list[str] = []
    if status is not None:
        predicates.append(f"status = {_sql_literal(_normalize_status(status))}")
    if invalidation_id is not None:
        predicates.append(f"invalidation_id = {_sql_literal(invalidation_id)}")

    rows = _load_rows_where(lake, " AND ".join(predicates) if predicates else None)
    if root_artifact_id is not None:
        wanted = str(root_artifact_id)
        rows = [row for row in rows if wanted in {str(v) for v in row.get("root_artifact_ids") or []}]

    ordered = sorted(
        rows,
        key=lambda item: (_as_dt(item.get("created_at")), str(item.get("plan_id"))),
        reverse=True,
    )
    start = _decode_cursor(cursor) if cursor else 0
    window = ordered[start : start + page_size]
    next_cursor = (
        _encode_cursor(start + page_size) if start + page_size < len(ordered) else None
    )
    return RebuildPlanListPage(
        plans=tuple(_row_to_entry(row) for row in window),
        next_cursor=next_cursor,
        page_size=page_size,
    )


# --- Lifecycle / approvals --------------------------------------------------


def update_rebuild_plan_status(
    lake: Lake,
    plan_id: str,
    status: str,
    *,
    expected_status: str | None = None,
    expected_revision: int | None = None,
    actor: str | None = None,
    approver: str | None = None,
    note: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_by: str | None = None,
) -> RebuildPlanCatalogEntry:
    """Move a recorded plan to a new lifecycle ``status`` (optimistically).

    Pass ``expected_revision`` and/or ``expected_status`` to guard against a
    concurrent update: a mismatch raises :class:`RebuildPlanConflict` telling the
    caller the current revision/status so they can reload and retry. The
    transition itself must be allowed (see :data:`STATUSES`); moving to
    ``approved`` requires an ``approver`` (here or already on the row). Each
    accepted update bumps ``revision`` and appends an audit event.
    """

    target = _normalize_status(status)
    row = _load_row(lake, str(plan_id))
    if row is None:
        raise RebuildPlanCatalogError(f"no rebuild plan recorded for {plan_id!r}")

    current_status = str(row.get("status") or DEFAULT_STATUS)
    current_revision = int(row.get("revision") or 0)

    if expected_revision is not None and int(expected_revision) != current_revision:
        raise RebuildPlanConflict(
            f"stale rebuild-plan update: {row['plan_id']} is at revision "
            f"{current_revision}, but expected_revision={expected_revision}; "
            "reload the plan and retry"
        )
    if expected_status is not None and _normalize_status(expected_status) != current_status:
        raise RebuildPlanConflict(
            f"stale rebuild-plan update: {row['plan_id']} is in status "
            f"{current_status!r}, but expected_status={expected_status!r}; "
            "reload the plan and retry"
        )

    allowed = _TRANSITIONS.get(current_status, frozenset())
    if target != current_status and target not in allowed:
        allowed_text = ", ".join(sorted(allowed)) or "(terminal state, no transitions)"
        raise RebuildPlanCatalogError(
            f"illegal rebuild-plan transition {current_status!r} -> {target!r}; "
            f"allowed from {current_status!r}: {allowed_text}"
        )
    if target == current_status:
        raise RebuildPlanCatalogError(
            f"rebuild plan {row['plan_id']} is already in status {target!r}"
        )

    if target == "approved" and not ((approver or "").strip() or str(row.get("approver") or "").strip()):
        raise RebuildPlanCatalogError("approving a rebuild plan requires approver=")

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
        row["metadata"] = _kv_items(metadata)
    if target == "approved":
        row["approved_at"] = now
    elif target == "dispatched":
        row["dispatched_at"] = now
    elif target in {"completed", "failed"}:
        row["completed_at"] = now
    row["updated_at"] = now

    _upsert_row(lake, row)
    _emit_event(
        lake,
        plan_id=str(row["plan_id"]),
        plan_digest=str(row.get("plan_digest") or row["plan_id"]),
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


# --- Orchestrator handoff ---------------------------------------------------


def export_rebuild_plan_dispatch(
    lake: Lake,
    plan_id: str,
    *,
    orchestrator: str | None = None,
    run_ref_prefix: str = DEFAULT_RUN_REF_PREFIX,
    dry_run: bool = True,
    actor: str | None = None,
    created_by: str | None = None,
) -> RebuildPlanDispatch:
    """Build a deterministic orchestrator handoff payload for a recorded plan.

    The payload gives every action a stable ``action_id`` (content-addressed on
    the plan digest + action), maps each dependency to the depended-on action id,
    and derives a retry-safe ``external_run_ref`` per action, so re-exporting an
    unchanged plan yields byte-identical payloads. ``dry_run=True`` (the default)
    is a pure validation/preview with no state change. ``dry_run=False`` requires
    the plan to be ``approved`` (or already ``dispatched``) and transitions it to
    ``dispatched`` once -- re-dispatching an already-dispatched plan is an
    idempotent no-op that returns the same payload.
    """

    row = _load_row(lake, str(plan_id))
    if row is None:
        raise RebuildPlanCatalogError(f"no rebuild plan recorded for {plan_id!r}")
    plan_dict = _decode_plan(row)
    dispatch = _build_dispatch(
        row,
        plan_dict,
        orchestrator=orchestrator,
        run_ref_prefix=run_ref_prefix,
    )

    if dry_run:
        return dispatch

    current_status = str(row.get("status") or DEFAULT_STATUS)
    if current_status == "dispatched":
        return dispatch  # idempotent re-dispatch
    if current_status != "approved":
        raise RebuildPlanCatalogError(
            f"rebuild plan {row['plan_id']} must be approved before dispatch "
            f"(current status: {current_status!r}); approve it first or pass dry_run=True"
        )

    updated = update_rebuild_plan_status(
        lake,
        str(row["plan_id"]),
        "dispatched",
        expected_status="approved",
        expected_revision=int(row.get("revision") or 0),
        actor=actor,
        created_by=created_by,
    )
    _emit_event(
        lake,
        plan_id=updated.plan_id,
        plan_digest=updated.plan_digest,
        event_type="dispatch-exported",
        from_status="approved",
        to_status="dispatched",
        revision=updated.revision,
        actor=actor,
        approver=None,
        detail=f"orchestrator={dispatch.orchestrator} actions={len(dispatch.actions)}",
        created_by=created_by,
        metadata={"orchestrator": dispatch.orchestrator, "run_ref_prefix": run_ref_prefix},
    )
    return dispatch


def rebuild_plan_events(
    lake: Lake,
    *,
    plan_id: str | None = None,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return audit events, optionally filtered by plan id / event type."""

    predicates: list[str] = []
    if plan_id is not None:
        predicates.append(f"plan_id = {_sql_literal(plan_id)}")
    if event_type is not None:
        predicates.append(f"event_type = {_sql_literal(event_type)}")
    handle = lake.table(EVENTS_TABLE).to_lance()
    where = " AND ".join(predicates) if predicates else None
    arrow = handle.to_table(filter=where) if where else handle.to_table()
    rows = arrow.to_pylist()
    rows.sort(key=lambda item: (_as_dt(item.get("created_at")), str(item.get("event_id"))))
    return _json_ready(rows)


# --- Dispatch construction --------------------------------------------------


def _build_dispatch(
    row: Mapping[str, Any],
    plan_dict: Mapping[str, Any],
    *,
    orchestrator: str | None,
    run_ref_prefix: str,
) -> RebuildPlanDispatch:
    plan_digest = str(row.get("plan_digest") or row.get("plan_id") or "")
    orchestrator_name = str(orchestrator or "generic")
    prefix = str(run_ref_prefix or DEFAULT_RUN_REF_PREFIX)
    actions = list(plan_dict.get("actions") or [])

    action_id_by_artifact: dict[str, str] = {}
    for action in actions:
        artifact = str(action.get("artifact_id") or "")
        action_id_by_artifact[artifact] = _action_id(plan_digest, action)

    dispatch_actions: list[dict[str, Any]] = []
    for action in actions:
        artifact = str(action.get("artifact_id") or "")
        action_id = action_id_by_artifact[artifact]
        depends_on_artifacts = [str(dep) for dep in action.get("depends_on") or []]
        depends_on_action_ids = [
            action_id_by_artifact[dep]
            for dep in depends_on_artifacts
            if dep in action_id_by_artifact
        ]
        dispatch_actions.append(
            {
                "action_id": action_id,
                "step": int(action.get("step") or 0),
                "action": str(action.get("action") or ""),
                "target_artifact_id": artifact,
                "kind": str(action.get("kind") or ""),
                "name": action.get("name"),
                "table_name": action.get("table_name"),
                "table_version": action.get("table_version"),
                "table_tag": action.get("table_tag"),
                "row_ids": [str(v) for v in action.get("row_ids") or []],
                "depends_on": depends_on_action_ids,
                "depends_on_artifact_ids": depends_on_artifacts,
                "external_run_ref": f"{prefix}:{plan_digest[:12]}:{action_id[len('act-'):]}",
                "reason": action.get("reason"),
                "metadata": dict(action.get("metadata") or {}),
            }
        )

    return RebuildPlanDispatch(
        schema_version=DISPATCH_SCHEMA_VERSION,
        plan_id=str(row.get("plan_id") or plan_digest),
        plan_digest=plan_digest,
        lake_uri=str(plan_dict.get("lake_uri") or row.get("lake_uri") or ""),
        orchestrator=orchestrator_name,
        invalidation_id=str(row.get("invalidation_id") or ""),
        root_artifact_ids=tuple(str(v) for v in plan_dict.get("root_artifact_ids") or []),
        reason=str(plan_dict.get("reason") or ""),
        severity=str(plan_dict.get("severity") or ""),
        actions=tuple(dispatch_actions),
    )


def _action_id(plan_digest: str, action: Mapping[str, Any]) -> str:
    payload = {
        "plan_digest": plan_digest,
        "step": int(action.get("step") or 0),
        "artifact_id": str(action.get("artifact_id") or ""),
        "action": str(action.get("action") or ""),
    }
    return "act-" + _short_digest(payload)


# --- Table IO ---------------------------------------------------------------


def _upsert_row(lake: Lake, row: Mapping[str, Any]) -> None:
    table = pa.Table.from_pylist([dict(row)], schema=REBUILD_PLANS_SCHEMA)
    (
        lake.table(CATALOG_TABLE)
        .merge_insert("plan_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(table)
    )


def _emit_event(
    lake: Lake,
    *,
    plan_id: str,
    plan_digest: str,
    event_type: str,
    from_status: str,
    to_status: str,
    revision: int,
    actor: str | None,
    approver: str | None,
    detail: str,
    created_by: str | None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    now = datetime.now(UTC)
    event_id = hashlib.sha256(
        json.dumps(
            [plan_id, event_type, from_status, to_status, int(revision), now.isoformat()],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    event = {
        "event_id": event_id,
        "plan_id": plan_id,
        "plan_digest": plan_digest,
        "event_type": event_type,
        "from_status": from_status,
        "to_status": to_status,
        "revision": int(revision),
        "actor": actor or "",
        "approver": approver or "",
        "detail": detail,
        "metadata": _kv_items(metadata),
        "created_by": created_by or "",
        "created_at": now,
    }
    table = pa.Table.from_pylist([event], schema=REBUILD_PLAN_EVENTS_SCHEMA)
    lake.table(EVENTS_TABLE).add(table)


def _load_row(lake: Lake, plan_id_or_digest: str) -> dict[str, Any] | None:
    literal = _sql_literal(plan_id_or_digest)
    rows = _load_rows_where(lake, f"plan_id = {literal} OR plan_digest = {literal}")
    return rows[0] if rows else None


def _load_rows_where(lake: Lake, where: str | None) -> list[dict[str, Any]]:
    handle = lake.table(CATALOG_TABLE).to_lance()
    arrow = handle.to_table(filter=where) if where else handle.to_table()
    return arrow.to_pylist()


# --- Plan / row parsing helpers ---------------------------------------------


def _plan_of(plan: RebuildPlan | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, RebuildPlan):
        return plan.as_dict()
    if hasattr(plan, "as_dict") and callable(plan.as_dict):
        return dict(plan.as_dict())
    if isinstance(plan, Mapping):
        return dict(plan)
    raise RebuildPlanCatalogError("plan must be a RebuildPlan or a plan mapping")


def _require_supported_schema(plan_dict: Mapping[str, Any]) -> str:
    schema = plan_dict.get("schema_version")
    if schema not in SUPPORTED_PLAN_SCHEMAS:
        raise RebuildPlanCatalogError(
            f"unsupported rebuild-plan schema {schema!r}; expected one of {SUPPORTED_PLAN_SCHEMAS}"
        )
    return str(schema)


def _stored_plan(
    plan_dict: Mapping[str, Any],
    schema_version: str,
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    """Return (bounded plan payload, content digest, sorted root ids).

    The digest excludes the invalidation marker (whose id embeds a timestamp) and
    the impact graph (derived/large), so the same plan content always hashes the
    same. The stored payload drops the graph but keeps the invalidation marker for
    reference.
    """

    roots = tuple(sorted(str(v) for v in plan_dict.get("root_artifact_ids") or []))
    actions = [dict(action) for action in plan_dict.get("actions") or []]
    canonical = {
        "schema_version": schema_version,
        "lake_uri": str(plan_dict.get("lake_uri") or ""),
        "root_artifact_ids": list(roots),
        "reason": plan_dict.get("reason"),
        "severity": plan_dict.get("severity"),
        "actions": actions,
    }
    digest = _full_digest(canonical)
    stored = {
        "schema_version": schema_version,
        "lake_uri": canonical["lake_uri"],
        "root_artifact_ids": list(roots),
        "reason": plan_dict.get("reason"),
        "severity": plan_dict.get("severity"),
        "invalidation": plan_dict.get("invalidation"),
        "actions": actions,
    }
    return stored, digest, roots


def _encode_plan(stored: Mapping[str, Any]) -> str:
    return json.dumps(_json_ready(stored), sort_keys=True, separators=(",", ":"))


def _decode_plan(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("plan_json")
    if not raw:
        raise RebuildPlanCatalogError(
            f"catalog row {row.get('plan_id')!r} has no stored plan payload"
        )
    return json.loads(raw)


def _version_pins(actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, int, str], dict[str, Any]] = {}
    for action in actions:
        table = action.get("table_name")
        version = action.get("table_version")
        if table is None or version is None:
            continue
        tag = str(action.get("table_tag") or "")
        key = (str(table), int(version), tag)
        seen.setdefault(key, {"table": str(table), "version": int(version), "tag": tag})
    return [seen[key] for key in sorted(seen)]


def _row_to_entry(row: Mapping[str, Any]) -> RebuildPlanCatalogEntry:
    return RebuildPlanCatalogEntry(
        plan_id=str(row.get("plan_id") or ""),
        plan_digest=str(row.get("plan_digest") or ""),
        catalog_schema_version=str(row.get("catalog_schema_version") or CATALOG_SCHEMA_VERSION),
        plan_schema_version=str(row.get("plan_schema_version") or ""),
        lake_uri=str(row.get("lake_uri") or ""),
        invalidation_id=str(row.get("invalidation_id") or ""),
        root_artifact_ids=tuple(str(v) for v in row.get("root_artifact_ids") or []),
        action_count=int(row.get("action_count") or 0),
        reason=str(row.get("reason") or ""),
        severity=str(row.get("severity") or ""),
        status=str(row.get("status") or DEFAULT_STATUS),
        revision=int(row.get("revision") or 0),
        actor=str(row.get("actor") or ""),
        approver=str(row.get("approver") or ""),
        approved_at=_as_dt(row.get("approved_at")),
        dispatched_at=_as_dt(row.get("dispatched_at")),
        completed_at=_as_dt(row.get("completed_at")),
        note=str(row.get("note") or ""),
        table_version_pins=tuple(dict(v) for v in row.get("table_version_pins") or []),
        metadata=_kv_to_dict(row.get("metadata")),
        created_by=str(row.get("created_by") or ""),
        created_at=_as_dt(row.get("created_at")) or datetime.now(UTC),
        updated_at=_as_dt(row.get("updated_at")) or datetime.now(UTC),
    )


# --- Small shared utilities -------------------------------------------------


def _normalize_status(status: str) -> str:
    value = str(status or "").strip().lower()
    if value not in STATUSES:
        raise RebuildPlanCatalogError(
            f"unknown rebuild-plan status {status!r}; expected one of {', '.join(STATUSES)}"
        )
    return value


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


def _short_digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:16]


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
        raise RebuildPlanCatalogError(f"invalid cursor {cursor!r}") from exc
