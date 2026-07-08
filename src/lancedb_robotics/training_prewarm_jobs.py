"""Durable, deduplicated prewarm JobRun lifecycle for Enterprise training.

Backlog 0121 (epic 0069, PRD G6/G10/G12). Backlog 0072 gave the SDK a
deterministic ``prewarm-*`` request/status envelope plus optional
``lake.plan_executor_prewarm`` / ``lake.plan_executor_prewarm_status`` hooks. That
envelope is *process-local*: every worker that opens the same snapshot/epoch
recomputes the same request and, if each submitted independently, would fan out
duplicate cache-warm work to the plan executors.

This module backs the 0072 envelope with a durable **JobRun** keyed by the
deterministic ``prewarm_id``. The ``prewarm_id`` is a content digest of the
policy/scope/plan-ids/columns/table-version, so every worker that would submit
the same warm request produces the *same* key. The first worker to claim the key
owns the remote submit; every other worker (across retries and repeated job
starts) attaches to the same JobRun and waits/proceeds according to policy. The
JobRun persists its full status history (submitted -> active -> complete /
skipped / failed / canceled / expired), caller/job labels, table versions,
projected columns, limits, routing, timestamps, terminal reason, PE fanout, and
retry count, and can be polled, retried, and canceled by id.

The module is coordination-plane only. It never opens the snapshot tables and
holds no lake reference: it operates over an already-built prewarm request
envelope and a pluggable :class:`PrewarmJobRunStore`. Capability gating and the
typed ``PrewarmUnavailableError`` diagnostic live in
:mod:`lancedb_robotics.training`, which owns the Enterprise capability matrix and
imports this file (the coupling runs one way).

Design decisions (recorded in the task record / decision file):

* **The dedup key is the 0072 ``prewarm_id``, unchanged.** No new identity is
  invented: the JobRun store is a durable index keyed by the request digest that
  0072 already computes. Two workers, a retry, and a re-run all collapse to one
  JobRun for free.
* **Attach, do not resubmit, while a JobRun is in-flight or warm-within-TTL.**
  A ``complete`` JobRun is reused until its TTL expires or cache invalidation
  marks it stale; a ``submitted``/``active`` JobRun is attached to (the caller
  waits on the shared run). Only ``failed``/``canceled``/``expired`` JobRuns are
  re-submitted, incrementing ``retry_count``.
* **Records are small and secret-free.** ``to_dict`` is asserted secret-free; the
  store persists scalar query dimensions plus a full ``record_json`` reload blob,
  never API keys or raw credentials (only opaque auth references and ids).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

PREWARM_JOB_KIND = "lancedb-robotics/prewarm-job-run/v1"
PREWARM_JOB_TABLE = "__lancedb_robotics_prewarm_job_runs"
DEFAULT_PREWARM_JOB_TTL_S = 3600.0

STATUS_SUBMITTED = "submitted"
STATUS_ACTIVE = "active"
STATUS_COMPLETE = "complete"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_CANCELED = "canceled"
STATUS_EXPIRED = "expired"

#: Statuses that will not change without an explicit retry / re-submit.
TERMINAL_STATUSES = frozenset(
    {STATUS_COMPLETE, STATUS_SKIPPED, STATUS_FAILED, STATUS_CANCELED, STATUS_EXPIRED}
)
#: Statuses that count as a usable warm cache (reused until TTL / invalidation).
WARM_STATUSES = frozenset({STATUS_COMPLETE})
#: Statuses that mean "already running" -- attach and wait, do not resubmit.
IN_FLIGHT_STATUSES = frozenset({STATUS_SUBMITTED, STATUS_ACTIVE})
#: Terminal-but-not-warm statuses that a fresh open should re-submit.
RESUBMITTABLE_STATUSES = frozenset({STATUS_FAILED, STATUS_CANCELED, STATUS_EXPIRED})
#: Statuses treated as an error by fail-fast callers.
ERROR_STATUSES = frozenset({STATUS_FAILED})

_SECRET_KEY_TOKENS = (
    "authorization",
    "api_key",
    "apikey",
    "access_key",
    "secret_key",
    "session_token",
    "password",
    "secret",
    "credential",
    "bearer",
    "token",
)


class PrewarmJobRunError(Exception):
    """Raised for prewarm JobRun store/lifecycle misuse (unknown id, bad state)."""


# --------------------------------------------------------------------------- #
# Serialization / digest / secret helpers (kept local so this module has no    #
# dependency on training.py -- training.py imports this file, not vice versa). #
# --------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _secret_key(key: str) -> bool:
    lowered = str(key).lower().replace("-", "_")
    if "auth_ref" in lowered:
        return False
    return any(token in lowered for token in _SECRET_KEY_TOKENS)


def _secret_value(value: str) -> bool:
    stripped = value.strip().lower()
    return stripped.startswith(("bearer ", "basic "))


def _assert_secret_free(payload: Any, *, path: str = "prewarm_job") -> None:
    if isinstance(payload, Mapping):
        for key, item in payload.items():
            if _secret_key(str(key)):
                raise PrewarmJobRunError(
                    f"prewarm JobRun record would leak a secret-like field at {path}.{key}"
                )
            _assert_secret_free(item, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            _assert_secret_free(item, path=f"{path}[{index}]")
    elif isinstance(payload, str) and _secret_value(payload):
        raise PrewarmJobRunError(
            f"prewarm JobRun record would leak a bearer/basic credential at {path}"
        )


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_iso(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #
# The durable JobRun record                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PrewarmJobRun:
    """A durable, deduplicated cache-prewarm operation keyed by ``prewarm_id``.

    Immutable: lifecycle transitions return a new record (``with_status`` /
    ``attach`` / ``retrying``) so the store owns the single source of truth and
    history is append-only.
    """

    prewarm_id: str
    status: str
    policy: str
    scope: str
    job_label: str
    caller_label: str | None
    snapshot_name: str | None
    dataset_id: str | None
    alignment_id: str | None
    alignment_name: str | None
    row_plan_id: str | None
    epoch_plan_id: str | None
    tick_plan_id: str | None
    table_versions: tuple[dict[str, Any], ...]
    projected_columns: tuple[str, ...]
    logical_columns: tuple[str, ...]
    excluded_columns: tuple[dict[str, Any], ...]
    row_count: int
    estimated_bytes: int
    limits: dict[str, Any]
    routing: dict[str, Any]
    table_uri: str | None
    workers: tuple[str, ...]
    submitted_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    updated_at: datetime
    ttl_s: float | None
    expires_at: datetime | None
    terminal_reason: str | None
    pe_fanout: int | None
    completed_executors: int | None
    failed_executors: int | None
    cache_hits: int | None
    cache_misses: int | None
    warm_bytes: int | None
    cold_bytes: int | None
    duration_ms: float | None
    retry_count: int
    attach_count: int
    status_history: tuple[dict[str, Any], ...]
    owner_nonce: str
    content_digest: str
    kind: str = PREWARM_JOB_KIND
    store_kind: str = "in-memory"
    store_ref: str = ""

    # -- predicates -------------------------------------------------------- #

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_in_flight(self) -> bool:
        return self.status in IN_FLIGHT_STATUSES

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and now >= self.expires_at

    def is_warm(self, now: datetime) -> bool:
        """A complete JobRun is warm until its TTL elapses."""
        return self.status in WARM_STATUSES and not self.is_expired(now)

    # -- transitions ------------------------------------------------------- #

    def _append_history(self, status: str, now: datetime, reason: str | None) -> tuple[dict[str, Any], ...]:
        entry = {"status": status, "at": _iso(now), "reason": reason}
        return (*self.status_history, entry)

    def attach(self, worker_label: str, now: datetime) -> PrewarmJobRun:
        """Record another worker attaching to this shared JobRun."""
        workers = self.workers if worker_label in self.workers else (*self.workers, worker_label)
        return replace(
            self,
            workers=workers,
            attach_count=self.attach_count + 1,
            updated_at=now,
        )

    def with_status(
        self,
        status: str,
        now: datetime,
        *,
        reason: str | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> PrewarmJobRun:
        """Return a new record transitioned to ``status`` with appended history."""
        started_at = self.started_at
        completed_at = self.completed_at
        if status == STATUS_ACTIVE and started_at is None:
            started_at = now
        if status in TERMINAL_STATUSES:
            completed_at = now
        terminal_reason = reason if status in TERMINAL_STATUSES else self.terminal_reason
        updates: dict[str, Any] = {
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "updated_at": now,
            "terminal_reason": terminal_reason,
            "status_history": self._append_history(status, now, reason),
        }
        if metrics:
            for source_key, target in (
                ("pe_fanout", "pe_fanout"),
                ("completed_executors", "completed_executors"),
                ("failed_executors", "failed_executors"),
                ("cache_hits", "cache_hits"),
                ("cache_misses", "cache_misses"),
                ("warm_bytes", "warm_bytes"),
                ("cold_bytes", "cold_bytes"),
                ("duration_ms", "duration_ms"),
            ):
                if metrics.get(source_key) is not None:
                    updates[target] = metrics[source_key]
        return replace(self, **updates)

    def retrying(self, now: datetime, *, ttl_s: float | None) -> PrewarmJobRun:
        """Return a fresh ``submitted`` attempt with an incremented retry count."""
        return replace(
            self,
            status=STATUS_SUBMITTED,
            started_at=None,
            completed_at=None,
            updated_at=now,
            submitted_at=now,
            expires_at=(now + _timedelta(ttl_s)) if ttl_s else None,
            ttl_s=ttl_s,
            terminal_reason=None,
            pe_fanout=None,
            completed_executors=None,
            failed_executors=None,
            retry_count=self.retry_count + 1,
            owner_nonce=uuid.uuid4().hex,
            status_history=self._append_history(STATUS_SUBMITTED, now, "retry"),
        )

    # -- serialization ----------------------------------------------------- #

    def status_dict(self) -> dict[str, Any]:
        """The status envelope returned to a prewarm caller (0072-compatible shape)."""
        payload = {
            "status": self.status,
            "prewarm_id": self.prewarm_id,
            "policy": self.policy,
            "scope": self.scope,
            "job_run_id": self.prewarm_id,
            "reason": self.terminal_reason,
            "retry_count": self.retry_count,
            "attach_count": self.attach_count,
            "submitted_at": _iso(self.submitted_at),
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "expires_at": _iso(self.expires_at),
            "pe_fanout": self.pe_fanout,
            "completed_executors": self.completed_executors,
            "failed_executors": self.failed_executors,
        }
        for key in ("cache_hits", "cache_misses", "warm_bytes", "cold_bytes", "duration_ms"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return {key: value for key, value in payload.items() if value is not None}

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "prewarm_id": self.prewarm_id,
            "status": self.status,
            "policy": self.policy,
            "scope": self.scope,
            "job_label": self.job_label,
            "caller_label": self.caller_label,
            "snapshot_name": self.snapshot_name,
            "dataset_id": self.dataset_id,
            "alignment_id": self.alignment_id,
            "alignment_name": self.alignment_name,
            "row_plan_id": self.row_plan_id,
            "epoch_plan_id": self.epoch_plan_id,
            "tick_plan_id": self.tick_plan_id,
            "table_versions": [dict(item) for item in self.table_versions],
            "projected_columns": list(self.projected_columns),
            "logical_columns": list(self.logical_columns),
            "excluded_columns": [dict(item) for item in self.excluded_columns],
            "row_count": self.row_count,
            "estimated_bytes": self.estimated_bytes,
            "limits": _jsonable(self.limits),
            "routing": _jsonable(self.routing),
            "table_uri": self.table_uri,
            "workers": list(self.workers),
            "submitted_at": _iso(self.submitted_at),
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "updated_at": _iso(self.updated_at),
            "ttl_s": self.ttl_s,
            "expires_at": _iso(self.expires_at),
            "terminal_reason": self.terminal_reason,
            "pe_fanout": self.pe_fanout,
            "completed_executors": self.completed_executors,
            "failed_executors": self.failed_executors,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "warm_bytes": self.warm_bytes,
            "cold_bytes": self.cold_bytes,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
            "attach_count": self.attach_count,
            "status_history": [dict(item) for item in self.status_history],
            "owner_nonce": self.owner_nonce,
            "content_digest": self.content_digest,
            "store_kind": self.store_kind,
            "store_ref": self.store_ref,
        }
        _assert_secret_free(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PrewarmJobRun:
        return cls(
            prewarm_id=str(payload["prewarm_id"]),
            status=str(payload.get("status", STATUS_SUBMITTED)),
            policy=str(payload.get("policy", "")),
            scope=str(payload.get("scope", "")),
            job_label=str(payload.get("job_label", "default")),
            caller_label=payload.get("caller_label"),
            snapshot_name=payload.get("snapshot_name"),
            dataset_id=payload.get("dataset_id"),
            alignment_id=payload.get("alignment_id"),
            alignment_name=payload.get("alignment_name"),
            row_plan_id=payload.get("row_plan_id"),
            epoch_plan_id=payload.get("epoch_plan_id"),
            tick_plan_id=payload.get("tick_plan_id"),
            table_versions=tuple(dict(item) for item in payload.get("table_versions", [])),
            projected_columns=tuple(str(c) for c in payload.get("projected_columns", [])),
            logical_columns=tuple(str(c) for c in payload.get("logical_columns", [])),
            excluded_columns=tuple(dict(item) for item in payload.get("excluded_columns", [])),
            row_count=int(payload.get("row_count", 0)),
            estimated_bytes=int(payload.get("estimated_bytes", 0)),
            limits=dict(payload.get("limits", {})),
            routing=dict(payload.get("routing", {})),
            table_uri=payload.get("table_uri"),
            workers=tuple(str(w) for w in payload.get("workers", [])),
            submitted_at=_parse_iso(payload.get("submitted_at")) or _utcnow(),
            started_at=_parse_iso(payload.get("started_at")),
            completed_at=_parse_iso(payload.get("completed_at")),
            updated_at=_parse_iso(payload.get("updated_at")) or _utcnow(),
            ttl_s=payload.get("ttl_s"),
            expires_at=_parse_iso(payload.get("expires_at")),
            terminal_reason=payload.get("terminal_reason"),
            pe_fanout=payload.get("pe_fanout"),
            completed_executors=payload.get("completed_executors"),
            failed_executors=payload.get("failed_executors"),
            cache_hits=payload.get("cache_hits"),
            cache_misses=payload.get("cache_misses"),
            warm_bytes=payload.get("warm_bytes"),
            cold_bytes=payload.get("cold_bytes"),
            duration_ms=payload.get("duration_ms"),
            retry_count=int(payload.get("retry_count", 0)),
            attach_count=int(payload.get("attach_count", 0)),
            status_history=tuple(dict(item) for item in payload.get("status_history", [])),
            owner_nonce=str(payload.get("owner_nonce", "")),
            content_digest=str(payload.get("content_digest", "")),
            kind=str(payload.get("kind", PREWARM_JOB_KIND)),
            store_kind=str(payload.get("store_kind", "in-memory")),
            store_ref=str(payload.get("store_ref", "")),
        )


def _timedelta(seconds: float):
    from datetime import timedelta

    return timedelta(seconds=float(seconds))


def build_prewarm_job_run(
    request: Mapping[str, Any],
    *,
    job_label: str,
    caller_label: str | None,
    now: datetime,
    ttl_s: float | None,
    store_kind: str,
    store_ref: str,
) -> PrewarmJobRun:
    """Build the initial ``submitted`` JobRun from a 0072 prewarm request envelope."""
    tables = request.get("tables") or []
    table_versions = tuple(
        {
            "table": str(item.get("table")),
            "version": item.get("version"),
            "label": item.get("label"),
        }
        for item in tables
    )
    content_digest = _stable_digest(
        {
            "prewarm_id": request.get("prewarm_id"),
            "policy": request.get("policy"),
            "scope": request.get("scope"),
            "table_versions": [dict(item) for item in table_versions],
            "projected_columns": list(request.get("projected_columns") or []),
        }
    )
    expires_at = (now + _timedelta(ttl_s)) if ttl_s else None
    return PrewarmJobRun(
        prewarm_id=str(request["prewarm_id"]),
        status=STATUS_SUBMITTED,
        policy=str(request.get("policy", "")),
        scope=str(request.get("scope", "")),
        job_label=job_label,
        caller_label=caller_label,
        snapshot_name=request.get("snapshot_name"),
        dataset_id=request.get("dataset_id"),
        alignment_id=request.get("alignment_id"),
        alignment_name=request.get("alignment_name"),
        row_plan_id=request.get("row_plan_id"),
        epoch_plan_id=request.get("epoch_plan_id"),
        tick_plan_id=request.get("tick_plan_id"),
        table_versions=table_versions,
        projected_columns=tuple(str(c) for c in request.get("projected_columns") or []),
        logical_columns=tuple(str(c) for c in request.get("logical_columns") or []),
        excluded_columns=tuple(dict(item) for item in request.get("excluded_columns") or []),
        row_count=int(request.get("row_count") or 0),
        estimated_bytes=int(request.get("estimated_bytes") or 0),
        limits=dict(request.get("limits") or {}),
        routing=dict(request.get("routing") or {}),
        table_uri=request.get("table_uri"),
        workers=(),
        submitted_at=now,
        started_at=None,
        completed_at=None,
        updated_at=now,
        ttl_s=ttl_s,
        expires_at=expires_at,
        terminal_reason=None,
        pe_fanout=None,
        completed_executors=None,
        failed_executors=None,
        cache_hits=None,
        cache_misses=None,
        warm_bytes=None,
        cold_bytes=None,
        duration_ms=None,
        retry_count=0,
        attach_count=0,
        status_history=({"status": STATUS_SUBMITTED, "at": _iso(now), "reason": None},),
        owner_nonce=uuid.uuid4().hex,
        content_digest=content_digest,
        store_kind=store_kind,
        store_ref=store_ref,
    )


# --------------------------------------------------------------------------- #
# Stores                                                                       #
# --------------------------------------------------------------------------- #


class PrewarmJobRunStore:
    """Where prewarm JobRuns are persisted and deduplicated, keyed by ``prewarm_id``.

    ``claim`` is the deduplication primitive: it inserts a JobRun only if the key
    is free and returns ``(winning_record, created)`` -- when ``created`` is
    ``False`` another caller already owns the key and this caller must attach.
    """

    kind = "abstract"

    def claim(self, record: PrewarmJobRun) -> tuple[PrewarmJobRun, bool]:
        raise NotImplementedError

    def get(self, prewarm_id: str) -> PrewarmJobRun | None:
        raise NotImplementedError

    def put(self, record: PrewarmJobRun) -> None:
        raise NotImplementedError

    def list(
        self,
        *,
        status: str | None = None,
        policy: str | None = None,
        limit: int | None = None,
    ) -> list[PrewarmJobRun]:
        raise NotImplementedError

    def store_ref(self, prewarm_id: str) -> str:
        raise NotImplementedError


class InMemoryPrewarmJobRunStore(PrewarmJobRunStore):
    """In-process store used by tests, the single-process loader, and simulation."""

    kind = "in-memory"

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}

    def claim(self, record: PrewarmJobRun) -> tuple[PrewarmJobRun, bool]:
        existing = self._runs.get(record.prewarm_id)
        if existing is not None:
            return PrewarmJobRun.from_dict(existing), False
        stored = replace(record, store_kind=self.kind, store_ref=self.store_ref(record.prewarm_id))
        self._runs[record.prewarm_id] = stored.to_dict()
        return stored, True

    def get(self, prewarm_id: str) -> PrewarmJobRun | None:
        payload = self._runs.get(prewarm_id)
        return PrewarmJobRun.from_dict(payload) if payload is not None else None

    def put(self, record: PrewarmJobRun) -> None:
        stored = replace(record, store_kind=self.kind, store_ref=self.store_ref(record.prewarm_id))
        self._runs[record.prewarm_id] = stored.to_dict()

    def list(
        self,
        *,
        status: str | None = None,
        policy: str | None = None,
        limit: int | None = None,
    ) -> list[PrewarmJobRun]:
        records = [PrewarmJobRun.from_dict(payload) for payload in self._runs.values()]
        records = _filter_records(records, status=status, policy=policy)
        records.sort(key=lambda r: (r.updated_at, r.prewarm_id), reverse=True)
        return records[:limit] if limit is not None else records

    def store_ref(self, prewarm_id: str) -> str:
        return f"memory://{PREWARM_JOB_TABLE}/{prewarm_id}"


class LanceTablePrewarmJobRunStore(PrewarmJobRunStore):
    """Durable store: one internal LanceDB table of JobRuns keyed by ``prewarm_id``.

    Survives process restarts so a worker in a fresh process (a spawned
    DataLoader worker, a re-run, a CLI invocation) sees the same JobRun another
    worker submitted. Upserts and claims go through ``merge_insert`` on
    ``prewarm_id``; reads are bounded (``select``/``where``/``limit``) and reload
    the full record from the ``record_json`` blob.
    """

    kind = "lancedb-table"

    def __init__(self, db: Any) -> None:
        if db is None:
            raise PrewarmJobRunError("LanceTablePrewarmJobRunStore requires a LanceDB connection")
        self._db = db

    # -- table plumbing ---------------------------------------------------- #

    def _schema(self):
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("prewarm_id", pa.string()),
                pa.field("status", pa.string()),
                pa.field("policy", pa.string()),
                pa.field("scope", pa.string()),
                pa.field("job_label", pa.string()),
                pa.field("snapshot_name", pa.string()),
                pa.field("retry_count", pa.int64()),
                pa.field("attach_count", pa.int64()),
                pa.field("submitted_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("expires_at", pa.string()),
                pa.field("terminal_reason", pa.string()),
                pa.field("record_json", pa.string()),
            ]
        )

    def _table_names(self) -> set[str]:
        response = self._db.list_tables()
        tables = getattr(response, "tables", response)
        return {str(name) for name in (tables or [])}

    def _row(self, record: PrewarmJobRun) -> dict[str, Any]:
        stored = replace(record, store_kind=self.kind, store_ref=self.store_ref(record.prewarm_id))
        payload = stored.to_dict()
        return {
            "prewarm_id": stored.prewarm_id,
            "status": stored.status,
            "policy": stored.policy,
            "scope": stored.scope,
            "job_label": stored.job_label,
            "snapshot_name": stored.snapshot_name or "",
            "retry_count": stored.retry_count,
            "attach_count": stored.attach_count,
            "submitted_at": _iso(stored.submitted_at) or "",
            "updated_at": _iso(stored.updated_at) or "",
            "expires_at": _iso(stored.expires_at) or "",
            "terminal_reason": stored.terminal_reason or "",
            "record_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        }

    def _ensure_table(self):
        import pyarrow as pa

        name = PREWARM_JOB_TABLE
        if name not in self._table_names():
            empty = pa.Table.from_pylist([], schema=self._schema())
            self._db.create_table(name, data=empty, mode="create")
        return self._db.open_table(name)

    def _upsert(self, record: PrewarmJobRun) -> None:
        import pyarrow as pa

        table = self._ensure_table()
        data = pa.Table.from_pylist([self._row(record)], schema=self._schema())
        (
            table.merge_insert("prewarm_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(data)
        )

    def _read_row(self, prewarm_id: str) -> dict[str, Any] | None:
        if PREWARM_JOB_TABLE not in self._table_names():
            return None
        table = self._db.open_table(PREWARM_JOB_TABLE)
        rows = (
            table.search()
            .where(f"prewarm_id = '{_escape(prewarm_id)}'")
            .select(["record_json"])
            .limit(1)
            .to_arrow()
            .to_pylist()
        )
        if not rows:
            return None
        return json.loads(rows[0]["record_json"])

    # -- store API --------------------------------------------------------- #

    def claim(self, record: PrewarmJobRun) -> tuple[PrewarmJobRun, bool]:
        existing = self._read_row(record.prewarm_id)
        if existing is not None:
            return PrewarmJobRun.from_dict(existing), False
        self._upsert(record)
        winner = self._read_row(record.prewarm_id)
        if winner is None:
            # Should not happen; treat as our own claim.
            return replace(record, store_kind=self.kind, store_ref=self.store_ref(record.prewarm_id)), True
        won = str(winner.get("owner_nonce")) == record.owner_nonce
        return PrewarmJobRun.from_dict(winner), won

    def get(self, prewarm_id: str) -> PrewarmJobRun | None:
        payload = self._read_row(prewarm_id)
        return PrewarmJobRun.from_dict(payload) if payload is not None else None

    def put(self, record: PrewarmJobRun) -> None:
        self._upsert(record)

    def list(
        self,
        *,
        status: str | None = None,
        policy: str | None = None,
        limit: int | None = None,
    ) -> list[PrewarmJobRun]:
        if PREWARM_JOB_TABLE not in self._table_names():
            return []
        table = self._db.open_table(PREWARM_JOB_TABLE)
        query = table.search().select(["record_json"])
        predicates = []
        if status is not None:
            predicates.append(f"status = '{_escape(status)}'")
        if policy is not None:
            predicates.append(f"policy = '{_escape(policy)}'")
        if predicates:
            query = query.where(" AND ".join(predicates))
        rows = query.to_arrow().to_pylist()
        records = [PrewarmJobRun.from_dict(json.loads(row["record_json"])) for row in rows]
        records = _filter_records(records, status=status, policy=policy)
        records.sort(key=lambda r: (r.updated_at, r.prewarm_id), reverse=True)
        return records[:limit] if limit is not None else records

    def store_ref(self, prewarm_id: str) -> str:
        return f"lancedb://{PREWARM_JOB_TABLE}/{prewarm_id}"


def _escape(value: str) -> str:
    return str(value).replace("'", "''")


def _filter_records(
    records: Iterable[PrewarmJobRun],
    *,
    status: str | None,
    policy: str | None,
) -> list[PrewarmJobRun]:
    result = []
    for record in records:
        if status is not None and record.status != status:
            continue
        if policy is not None and record.policy != policy:
            continue
        result.append(record)
    return result


# --------------------------------------------------------------------------- #
# Coordinator                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PrewarmJobResult:
    """Outcome of ``submit_or_attach``: the JobRun plus whether we owned the submit."""

    record: PrewarmJobRun
    created: bool
    attached: bool
    reused_warm: bool

    @property
    def status(self) -> dict[str, Any]:
        return self.record.status_dict()


class PrewarmJobCoordinator:
    """Deduplicated submit / poll / retry / cancel over a :class:`PrewarmJobRunStore`.

    ``submit_fn`` / ``status_fn`` are the underlying (0072) plan-executor hooks:
    ``submit_fn(request) -> response`` and
    ``status_fn(prewarm_id, request, wait, timeout_s) -> response``. When they are
    ``None`` the coordinator still records a durable JobRun (``submitted`` / a
    ``planned`` terminal reason) so the lifecycle is inspectable even before a
    live plan-executor client is attached.
    """

    def __init__(
        self,
        store: PrewarmJobRunStore,
        *,
        submit_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        status_fn: Callable[..., Any] | None = None,
        now_fn: Callable[[], datetime] = _utcnow,
        ttl_s: float | None = DEFAULT_PREWARM_JOB_TTL_S,
    ) -> None:
        self.store = store
        self._submit_fn = submit_fn
        self._status_fn = status_fn
        self._now = now_fn
        self._ttl_s = ttl_s

    # -- submit / attach --------------------------------------------------- #

    def submit_or_attach(
        self,
        request: Mapping[str, Any],
        *,
        job_label: str = "default",
        caller_label: str | None = None,
        worker_label: str = "worker-0/1",
        wait: bool = False,
        timeout_s: float | None = None,
    ) -> PrewarmJobResult:
        now = self._now()
        prewarm_id = str(request["prewarm_id"])
        existing = self.store.get(prewarm_id)
        if existing is not None and self._maybe_expire(existing, now) is not existing:
            existing = self._maybe_expire(existing, now)

        if existing is not None and existing.is_warm(now):
            attached = existing.attach(worker_label, now)
            self.store.put(attached)
            return PrewarmJobResult(attached, created=False, attached=True, reused_warm=True)

        if existing is not None and existing.is_in_flight:
            attached = existing.attach(worker_label, now)
            self.store.put(attached)
            if wait:
                polled = self.poll(prewarm_id, request, wait=True, timeout_s=timeout_s)
                return PrewarmJobResult(polled, created=False, attached=True, reused_warm=False)
            return PrewarmJobResult(attached, created=False, attached=True, reused_warm=False)

        if existing is not None and existing.status in RESUBMITTABLE_STATUSES:
            record = existing.retrying(now, ttl_s=self._ttl_s)
            self.store.put(record)
            return self._run_submit(record, request, wait=wait, timeout_s=timeout_s)

        # No usable JobRun -- claim the key.
        fresh = build_prewarm_job_run(
            request,
            job_label=job_label,
            caller_label=caller_label,
            now=now,
            ttl_s=self._ttl_s,
            store_kind=self.store.kind,
            store_ref=self.store.store_ref(prewarm_id),
        )
        winner, created = self.store.claim(fresh)
        if not created:
            attached = winner.attach(worker_label, now)
            self.store.put(attached)
            if wait and attached.is_in_flight:
                polled = self.poll(prewarm_id, request, wait=True, timeout_s=timeout_s)
                return PrewarmJobResult(polled, created=False, attached=True, reused_warm=attached.is_warm(now))
            return PrewarmJobResult(attached, created=False, attached=True, reused_warm=winner.is_warm(now))
        owned = winner.attach(worker_label, now)
        return self._run_submit(owned, request, wait=wait, timeout_s=timeout_s)

    def _run_submit(
        self,
        record: PrewarmJobRun,
        request: Mapping[str, Any],
        *,
        wait: bool,
        timeout_s: float | None,
    ) -> PrewarmJobResult:
        now = self._now()
        if self._submit_fn is None:
            planned = record.with_status(
                STATUS_SUBMITTED,
                now,
                reason="no live plan-executor prewarm client is attached",
            )
            self.store.put(planned)
            return PrewarmJobResult(planned, created=True, attached=True, reused_warm=False)
        try:
            response = self._submit_fn(request)
        except Exception as exc:  # pragma: no cover - exercised by tests with hooks.
            failed = record.with_status(STATUS_FAILED, now, reason=str(exc))
            self.store.put(failed)
            return PrewarmJobResult(failed, created=True, attached=True, reused_warm=False)
        normalized = _normalize_response(response)
        status = normalized.get("status", STATUS_ACTIVE)
        active = record.with_status(status, now, reason=normalized.get("reason"), metrics=normalized)
        self.store.put(active)
        if wait and not active.is_terminal:
            polled = self.poll(str(request["prewarm_id"]), request, wait=True, timeout_s=timeout_s)
            return PrewarmJobResult(polled, created=True, attached=True, reused_warm=False)
        return PrewarmJobResult(active, created=True, attached=True, reused_warm=False)

    # -- poll -------------------------------------------------------------- #

    def poll(
        self,
        prewarm_id: str,
        request: Mapping[str, Any] | None = None,
        *,
        wait: bool = False,
        timeout_s: float | None = None,
    ) -> PrewarmJobRun:
        now = self._now()
        record = self.store.get(prewarm_id)
        if record is None:
            raise PrewarmJobRunError(f"no prewarm JobRun for id {prewarm_id!r}")
        record = self._maybe_expire(record, now)
        if record.is_terminal or self._status_fn is None:
            return record
        try:
            response = self._status_fn(
                prewarm_id=prewarm_id,
                request=dict(request or {}),
                wait=wait,
                timeout_s=timeout_s,
            )
        except TypeError:
            try:
                response = self._status_fn(prewarm_id, dict(request or {}), wait, timeout_s)
            except TypeError:
                response = self._status_fn(prewarm_id)
        except Exception as exc:  # pragma: no cover - exercised by tests with hooks.
            failed = record.with_status(STATUS_FAILED, now, reason=str(exc))
            self.store.put(failed)
            return failed
        normalized = _normalize_response(response)
        status = normalized.get("status", record.status)
        updated = record.with_status(status, now, reason=normalized.get("reason"), metrics=normalized)
        self.store.put(updated)
        return updated

    # -- retry / cancel / expire ------------------------------------------ #

    def retry(self, prewarm_id: str, request: Mapping[str, Any] | None = None) -> PrewarmJobRun:
        now = self._now()
        record = self.store.get(prewarm_id)
        if record is None:
            raise PrewarmJobRunError(f"no prewarm JobRun for id {prewarm_id!r}")
        if not record.is_terminal or record.is_warm(now):
            raise PrewarmJobRunError(
                f"prewarm JobRun {prewarm_id!r} is {record.status!r}; only failed/canceled/expired "
                "JobRuns can be retried"
            )
        retried = record.retrying(now, ttl_s=self._ttl_s)
        self.store.put(retried)
        result = self._run_submit(retried, request or _request_from_record(record), wait=False, timeout_s=None)
        return result.record

    def cancel(self, prewarm_id: str, *, reason: str | None = None) -> PrewarmJobRun:
        now = self._now()
        record = self.store.get(prewarm_id)
        if record is None:
            raise PrewarmJobRunError(f"no prewarm JobRun for id {prewarm_id!r}")
        if record.status in {STATUS_COMPLETE, STATUS_CANCELED}:
            return record
        canceled = record.with_status(STATUS_CANCELED, now, reason=reason or "canceled by caller")
        self.store.put(canceled)
        return canceled

    def expire_due(self) -> list[PrewarmJobRun]:
        now = self._now()
        expired: list[PrewarmJobRun] = []
        for record in self.store.list():
            updated = self._maybe_expire(record, now)
            if updated is not record:
                expired.append(updated)
        return expired

    def _maybe_expire(self, record: PrewarmJobRun, now: datetime) -> PrewarmJobRun:
        if record.status in {STATUS_CANCELED, STATUS_EXPIRED, STATUS_FAILED, STATUS_SKIPPED}:
            return record
        if record.is_expired(now):
            expired = record.with_status(STATUS_EXPIRED, now, reason="ttl elapsed")
            self.store.put(expired)
            return expired
        return record


def _normalize_response(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        status = dict(response)
    elif response is None:
        status = {"status": STATUS_ACTIVE}
    else:
        status = {"status": STATUS_ACTIVE}
    if status.get("pe_fanout") is None:
        completed = int(status.get("completed_executors") or 0)
        failed = int(status.get("failed_executors") or 0)
        if completed or failed:
            status["pe_fanout"] = completed + failed
    return status


def _request_from_record(record: PrewarmJobRun) -> dict[str, Any]:
    """Reconstruct a minimal prewarm request envelope from a stored JobRun.

    Used by CLI/API retry when the caller has only the ``prewarm_id`` -- enough
    for the plan-executor hook to re-issue the warm request (identity + tables +
    projection + routing), not a full re-plan of the row order.
    """
    return {
        "kind": "lancedb-robotics/training-prewarm/v1",
        "requested": True,
        "prewarm_id": record.prewarm_id,
        "policy": record.policy,
        "scope": record.scope,
        "snapshot_name": record.snapshot_name,
        "dataset_id": record.dataset_id,
        "alignment_id": record.alignment_id,
        "alignment_name": record.alignment_name,
        "row_plan_id": record.row_plan_id,
        "epoch_plan_id": record.epoch_plan_id,
        "tick_plan_id": record.tick_plan_id,
        "routing": dict(record.routing),
        "table_uri": record.table_uri,
        "tables": [dict(item) for item in record.table_versions],
        "projected_columns": list(record.projected_columns),
        "logical_columns": list(record.logical_columns),
        "row_count": record.row_count,
        "limits": dict(record.limits),
    }


def worker_label_from_request(request: Mapping[str, Any]) -> str:
    worker = request.get("worker") or {}
    return f"worker-{worker.get('id', 0)}/{worker.get('num_workers', 1)}"


def resolve_prewarm_job_store(lake: Any) -> PrewarmJobRunStore | None:
    """Return the JobRun store attached to ``lake``, or ``None`` for the 0072 path.

    The durable JobRun lifecycle is opt-in so that, with nothing attached, prewarm
    is byte-identical to the 0072 process-local envelope. Resolution order:

    1. ``lake.prewarm_job_store`` set to a :class:`PrewarmJobRunStore` -- used as-is.
    2. ``lake.prewarm_jobs_durable`` truthy and a LanceDB connection present --
       a durable :class:`LanceTablePrewarmJobRunStore`.
    3. otherwise ``None``.
    """
    explicit = getattr(lake, "prewarm_job_store", None)
    if isinstance(explicit, PrewarmJobRunStore):
        return explicit
    if getattr(lake, "prewarm_jobs_durable", False):
        db = getattr(lake, "_db", None)
        if db is not None and all(
            hasattr(db, name) for name in ("create_table", "open_table", "list_tables")
        ):
            return LanceTablePrewarmJobRunStore(db)
    return None


def open_prewarm_job_store(lake: Any) -> PrewarmJobRunStore | None:
    """Return a store for read/retry/cancel APIs: the attached store, else durable."""
    store = resolve_prewarm_job_store(lake)
    if store is not None:
        return store
    db = getattr(lake, "_db", None)
    if db is not None and all(
        hasattr(db, name) for name in ("create_table", "open_table", "list_tables")
    ):
        durable = LanceTablePrewarmJobRunStore(db)
        if PREWARM_JOB_TABLE in durable._table_names():
            return durable
    return None
