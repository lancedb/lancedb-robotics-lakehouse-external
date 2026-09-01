"""Durable, deduplicated scalar predicate index job lifecycle (backlog 0136).

Backlog 0079 (`indexing.py`) added *synchronous* scalar predicate index helpers:
``build_scalar_index`` calls ``create_scalar_index`` inline and returns a
``ScalarIndexResult`` immediately, recording ``skipped``/``failed`` when a backend
cannot build one. That is enough for local LanceDB and object-store OSS lakes,
where an index build is a fast, in-process, permitted operation.

Enterprise-scale ``db://`` lakes are different. A scalar index over a large
``aligned_ticks`` table may be submitted to a remote build service, take minutes
to hours, need progress polling, and be permission-gated. Blocking a training
read on such a build (or fanning out duplicate builds from every worker) is
exactly the failure this module prevents. It backs the 0079 helper with a durable
**job** keyed by a content digest of ``{table, table_version, column,
index_type}`` so that:

* **repeated requests are idempotent** -- two workers, a retry, and a re-run of
  ``lake maintain`` all collapse to one job. A ``complete`` job is reused; an
  in-flight job is attached to; a ``failed``/``canceled``/``expired`` job is
  resubmitted with an incremented ``retry_count``;
* **the read path never mutates the lake** -- ``aligned_dataset`` only *reads*
  persisted jobs to surface pending/failed references in its manifest, and never
  creates one (decision: index creation is an operator/maintenance action, not a
  training-read side effect -- consistent with 0079);
* **the build mode is honest** -- capability probing distinguishes
  ``synchronous`` (inline local build), ``asynchronous`` (remote submit + poll),
  ``unsupported`` (backend cannot build scalar indexes), and
  ``permission_denied`` (backend could, but the caller is not authorized), so a
  build never silently degrades and pushdown-only reads are never surprised.

The module is coordination-plane only past the synchronous builder seam: the
``ScalarIndexJobCoordinator`` operates over a pluggable
:class:`ScalarIndexJobStore` plus optional ``build_fn`` (sync) / ``submit_fn`` +
``status_fn`` (async remote) callables. It mirrors the 0121 prewarm JobRun
lifecycle (immutable record, ``with_status``/``retrying`` transitions,
``owner_nonce`` race arbitration, ``status_history``) and adopts the 0131
permutation catalog's merge_insert + bounded commit-conflict-retry write path,
which is the repo's mandated concurrent-writer standard (SKILLS.md s1).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from lancedb_robotics.capability_gates import INDEX, lake_capability_reason
from lancedb_robotics.indexing import (
    ALIGNED_FRAME_PREDICATE_INDEX_COLUMNS,
    ALIGNED_TICK_PREDICATE_INDEX_COLUMNS,
    SCALAR_INDEX_TYPE,
    ScalarIndexResult,
)

SCALAR_INDEX_JOB_KIND = "lancedb-robotics/scalar-index-job/v1"
SCALAR_INDEX_JOB_TABLE = "__lancedb_robotics_scalar_index_jobs"
#: Staleness deadline for a job that is *stuck* submitted/active (a lost remote
#: build). Complete jobs never expire on a timer -- a built index stays valid
#: until the table version changes (which mints a new job id). Default 6h.
DEFAULT_SCALAR_INDEX_JOB_TTL_S = 6 * 3600.0

#: Bounded retry on retryable Lance commit conflicts (mirrors 0131 catalog).
_JOB_COMMIT_RETRIES = 8
_SCAN_BATCH_SIZE = 2048

# --- build modes reported by the capability probe -------------------------- #
MODE_SYNCHRONOUS = "synchronous"
MODE_ASYNCHRONOUS = "asynchronous"
MODE_UNSUPPORTED = "unsupported"
MODE_PERMISSION_DENIED = "permission_denied"
BUILD_MODES = (MODE_SYNCHRONOUS, MODE_ASYNCHRONOUS, MODE_UNSUPPORTED, MODE_PERMISSION_DENIED)
#: Modes where a build is actually attempted (sync inline or async submit).
BUILDABLE_MODES = frozenset({MODE_SYNCHRONOUS, MODE_ASYNCHRONOUS})

# --- job lifecycle statuses (mirror 0121) ----------------------------------- #
STATUS_SUBMITTED = "submitted"
STATUS_ACTIVE = "active"
STATUS_COMPLETE = "complete"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_CANCELED = "canceled"
STATUS_EXPIRED = "expired"

TERMINAL_STATUSES = frozenset(
    {STATUS_COMPLETE, STATUS_SKIPPED, STATUS_FAILED, STATUS_CANCELED, STATUS_EXPIRED}
)
#: A complete job means the index exists; reused until the table version (and so
#: the job id) changes. Unlike a prewarm cache, it does not go cold on a timer.
COMPLETE_STATUSES = frozenset({STATUS_COMPLETE})
#: "Already running" -- attach and (optionally) wait, do not resubmit.
IN_FLIGHT_STATUSES = frozenset({STATUS_SUBMITTED, STATUS_ACTIVE})
#: Terminal-but-not-complete: a fresh request resubmits these.
RESUBMITTABLE_STATUSES = frozenset({STATUS_FAILED, STATUS_CANCELED, STATUS_EXPIRED})
#: Statuses a fail-fast/strict caller treats as "not index-backed".
ERROR_STATUSES = frozenset({STATUS_FAILED})

#: 0079 ScalarIndexResult.status -> job status.
_INDEX_RESULT_TO_JOB_STATUS = {
    "built": STATUS_COMPLETE,
    "already_present": STATUS_COMPLETE,
    "skipped": STATUS_SKIPPED,
    "failed": STATUS_FAILED,
}

_PERMISSION_ERROR_FRAGMENTS = (
    "permission denied",
    "permissiondenied",
    "not authorized",
    "unauthorized",
    "access denied",
    "accessdenied",
    "forbidden",
    "insufficient privilege",
    "not permitted",
)


class ScalarIndexJobError(Exception):
    """Raised for scalar-index job store/lifecycle misuse (unknown id, bad state)."""


class ScalarIndexRequiredError(Exception):
    """Raised in strict mode when a hot filter predicate has no index or job.

    Carries the missing columns and any known pending job ids so a caller can
    render actionable guidance (e.g. ``lake maintain`` to build them).
    """

    def __init__(self, table: str, missing: Sequence[Mapping[str, Any]]) -> None:
        self.table = table
        self.missing = [dict(item) for item in missing]
        cols = ", ".join(sorted({str(item.get("column")) for item in self.missing}))
        super().__init__(
            f"strict predicate indexing requested for {table!r} but these filter "
            f"columns are not index-backed and have no completed index job: {cols}. "
            "Run `lake maintain` (or lake.training.request_aligned_index_jobs(...)) "
            "to build them, or call without require_predicate_indexes=True to read "
            "through predicate pushdown."
        )


# --------------------------------------------------------------------------- #
# Serialization / digest / error-classification helpers (self-contained so the #
# module has no dependency cycle back into training.py).                       #
# --------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


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


def _timedelta(ttl_s: float) -> timedelta:
    return timedelta(seconds=float(ttl_s))


def _escape(value: str) -> str:
    return str(value).replace("'", "''")


def _is_permission_denied_error(exc: BaseException) -> bool:
    """True when ``exc`` reads as an authorization failure, not a transient error."""
    if isinstance(exc, PermissionError):
        return True
    message = str(exc).lower()
    return any(fragment in message for fragment in _PERMISSION_ERROR_FRAGMENTS)


def _is_retryable_commit_conflict(exc: BaseException) -> bool:
    """True when ``exc`` is a Lance optimistic-concurrency commit conflict.

    Matches ``enrich._is_retryable_commit_conflict`` / the 0131 catalog so every
    shared write target treats concurrent-writer preemption identically.
    """
    return "commit conflict" in str(exc).lower()


def _already_exists(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "already exists" in message or "alreadyexists" in message


# --------------------------------------------------------------------------- #
# Capability descriptor                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScalarIndexCapability:
    """How the active backend can build a scalar predicate index.

    ``mode`` is one of :data:`BUILD_MODES`. ``synchronous`` and ``asynchronous``
    are buildable; ``unsupported`` and ``permission_denied`` are not, and carry a
    ``reason`` plus recommended ``fallbacks`` so a caller renders guidance rather
    than a bare stack trace.
    """

    mode: str
    backend_kind: str
    reason: str | None = None
    fallbacks: tuple[str, ...] = ()

    @property
    def buildable(self) -> bool:
        return self.mode in BUILDABLE_MODES

    @property
    def is_async(self) -> bool:
        return self.mode == MODE_ASYNCHRONOUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "backend_kind": self.backend_kind,
            "reason": self.reason,
            "fallbacks": list(self.fallbacks),
        }


def _async_hook(lake: Any) -> Callable[..., Any] | None:
    hook = getattr(lake, "scalar_index_job_submit", None)
    return hook if callable(hook) else None


def probe_scalar_index_capability(lake: Any) -> ScalarIndexCapability:
    """Classify how ``lake``'s backend builds scalar predicate indexes.

    Ordering (positive signals win, matching the repo's capability-driven gates):

    1. no resolved connection spec (bare/local/in-process lake) -> ``synchronous``
       -- unchanged 0079 behavior;
    2. an explicit permission-denied signal -> ``permission_denied``;
    3. an attached async submit hook or advertised async capability ->
       ``asynchronous``;
    4. the ``index_management`` control-plane capability advertised ->
       ``synchronous``;
    5. otherwise -> ``unsupported`` (with the 0128 capability-gate guidance).
    """
    spec = getattr(lake, "connection_spec", None)
    if spec is None:
        return ScalarIndexCapability(mode=MODE_SYNCHRONOUS, backend_kind="in-process")
    backend_kind = getattr(spec, "kind", "unknown")
    capabilities = getattr(spec, "capabilities", None)

    permission_denied = bool(getattr(lake, "scalar_index_permission_denied", False)) or bool(
        getattr(capabilities, "index_permission_denied", False)
    )
    if permission_denied:
        return ScalarIndexCapability(
            mode=MODE_PERMISSION_DENIED,
            backend_kind=backend_kind,
            reason=(
                f"backend {backend_kind!r} reports the caller is not authorized to "
                "build scalar indexes; predicate pushdown remains available. Grant "
                "index-management permission, or retry with authorized credentials."
            ),
            fallbacks=("object_store_lancedb_oss", "pylance_direct_namespace"),
        )

    if _async_hook(lake) is not None or bool(getattr(capabilities, "async_index_builds", False)):
        return ScalarIndexCapability(mode=MODE_ASYNCHRONOUS, backend_kind=backend_kind)

    # Reuse the 0128 gate: index_management advertised => synchronous; else the
    # gate hands us the actionable "not advertised" guidance for unsupported.
    reason = lake_capability_reason(lake, INDEX)
    if reason is None:
        return ScalarIndexCapability(mode=MODE_SYNCHRONOUS, backend_kind=backend_kind)
    return ScalarIndexCapability(
        mode=MODE_UNSUPPORTED,
        backend_kind=backend_kind,
        reason=reason,
        fallbacks=("object_store_lancedb_oss", "pylance_direct_namespace"),
    )


# --------------------------------------------------------------------------- #
# Job request + durable record                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScalarIndexJobRequest:
    """The identity of one scalar-index build: the digest inputs, verbatim."""

    table: str
    column: str
    index_type: str = SCALAR_INDEX_TYPE
    table_version: int | None = None
    replace: bool = False

    def job_id(self) -> str:
        # table_version IS part of identity (task 0136: "same table/version/column/
        # index-type request should reuse"). replace is a per-attempt flag, NOT
        # identity -- excluded so a forced rebuild reuses the same job id.
        return "scix-" + _stable_digest(
            {
                "table": self.table,
                "table_version": self.table_version,
                "column": self.column,
                "index_type": self.index_type,
            }
        )


@dataclass(frozen=True)
class ScalarIndexJob:
    """A durable, deduplicated scalar predicate index build keyed by ``job_id``.

    Immutable: lifecycle transitions return a new record so the store owns the
    single source of truth and ``status_history`` is append-only.
    """

    job_id: str
    status: str
    table: str
    column: str
    index_type: str
    build_mode: str
    backend_kind: str
    table_version: int | None
    num_rows: int | None
    submitted_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    updated_at: datetime
    ttl_s: float | None
    expires_at: datetime | None
    terminal_reason: str | None
    retry_count: int
    request_count: int
    status_history: tuple[dict[str, Any], ...]
    owner_nonce: str
    content_digest: str
    kind: str = SCALAR_INDEX_JOB_KIND
    store_kind: str = "in-memory"
    store_ref: str = ""

    # -- predicates -------------------------------------------------------- #

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_in_flight(self) -> bool:
        return self.status in IN_FLIGHT_STATUSES

    @property
    def is_complete(self) -> bool:
        return self.status in COMPLETE_STATUSES

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and now >= self.expires_at

    # -- transitions ------------------------------------------------------- #

    def _append_history(
        self, status: str, now: datetime, reason: str | None
    ) -> tuple[dict[str, Any], ...]:
        return (*self.status_history, {"status": status, "at": _iso(now), "reason": reason})

    def touch_request(self, now: datetime) -> ScalarIndexJob:
        """Record that the same job was requested again (dedup hit)."""
        return replace(self, request_count=self.request_count + 1, updated_at=now)

    def with_status(
        self,
        status: str,
        now: datetime,
        *,
        reason: str | None = None,
        num_rows: int | None = None,
    ) -> ScalarIndexJob:
        started_at = self.started_at
        completed_at = self.completed_at
        if status == STATUS_ACTIVE and started_at is None:
            started_at = now
        if status in TERMINAL_STATUSES:
            completed_at = now
        terminal_reason = reason if status in TERMINAL_STATUSES else self.terminal_reason
        return replace(
            self,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            updated_at=now,
            terminal_reason=terminal_reason,
            num_rows=num_rows if num_rows is not None else self.num_rows,
            status_history=self._append_history(status, now, reason),
        )

    def retrying(
        self,
        now: datetime,
        *,
        ttl_s: float | None,
        build_mode: str | None = None,
        backend_kind: str | None = None,
    ) -> ScalarIndexJob:
        """A fresh ``submitted`` attempt with an incremented retry count."""
        return replace(
            self,
            status=STATUS_SUBMITTED,
            build_mode=build_mode or self.build_mode,
            backend_kind=backend_kind or self.backend_kind,
            started_at=None,
            completed_at=None,
            submitted_at=now,
            updated_at=now,
            expires_at=(now + _timedelta(ttl_s)) if ttl_s else None,
            ttl_s=ttl_s,
            terminal_reason=None,
            retry_count=self.retry_count + 1,
            owner_nonce=uuid.uuid4().hex,
            status_history=self._append_history(STATUS_SUBMITTED, now, "retry"),
        )

    # -- serialization ----------------------------------------------------- #

    def status_dict(self) -> dict[str, Any]:
        """Compact status envelope returned to a caller / CLI."""
        payload = {
            "job_id": self.job_id,
            "status": self.status,
            "table": self.table,
            "column": self.column,
            "index_type": self.index_type,
            "build_mode": self.build_mode,
            "backend_kind": self.backend_kind,
            "table_version": self.table_version,
            "num_rows": self.num_rows,
            "reason": self.terminal_reason,
            "retry_count": self.retry_count,
            "request_count": self.request_count,
            "submitted_at": _iso(self.submitted_at),
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "expires_at": _iso(self.expires_at),
        }
        return {key: value for key, value in payload.items() if value is not None}

    def manifest_ref(self) -> dict[str, Any]:
        """The per-column reference folded into an aligned training manifest."""
        return {
            "job_id": self.job_id,
            "job_status": self.status,
            "job_build_mode": self.build_mode,
            "job_reason": self.terminal_reason,
            "job_retry_count": self.retry_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "job_id": self.job_id,
            "status": self.status,
            "table": self.table,
            "column": self.column,
            "index_type": self.index_type,
            "build_mode": self.build_mode,
            "backend_kind": self.backend_kind,
            "table_version": self.table_version,
            "num_rows": self.num_rows,
            "submitted_at": _iso(self.submitted_at),
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "updated_at": _iso(self.updated_at),
            "ttl_s": self.ttl_s,
            "expires_at": _iso(self.expires_at),
            "terminal_reason": self.terminal_reason,
            "retry_count": self.retry_count,
            "request_count": self.request_count,
            "status_history": [dict(item) for item in self.status_history],
            "owner_nonce": self.owner_nonce,
            "content_digest": self.content_digest,
            "store_kind": self.store_kind,
            "store_ref": self.store_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScalarIndexJob:
        return cls(
            job_id=str(payload["job_id"]),
            status=str(payload["status"]),
            table=str(payload["table"]),
            column=str(payload["column"]),
            index_type=str(payload.get("index_type") or SCALAR_INDEX_TYPE),
            build_mode=str(payload.get("build_mode") or MODE_SYNCHRONOUS),
            backend_kind=str(payload.get("backend_kind") or "unknown"),
            table_version=payload.get("table_version"),
            num_rows=payload.get("num_rows"),
            submitted_at=_parse_iso(payload.get("submitted_at")) or _utcnow(),
            started_at=_parse_iso(payload.get("started_at")),
            completed_at=_parse_iso(payload.get("completed_at")),
            updated_at=_parse_iso(payload.get("updated_at")) or _utcnow(),
            ttl_s=payload.get("ttl_s"),
            expires_at=_parse_iso(payload.get("expires_at")),
            terminal_reason=payload.get("terminal_reason") or None,
            retry_count=int(payload.get("retry_count") or 0),
            request_count=int(payload.get("request_count") or 0),
            status_history=tuple(dict(item) for item in (payload.get("status_history") or ())),
            owner_nonce=str(payload.get("owner_nonce") or ""),
            content_digest=str(payload.get("content_digest") or ""),
            kind=str(payload.get("kind") or SCALAR_INDEX_JOB_KIND),
            store_kind=str(payload.get("store_kind") or "in-memory"),
            store_ref=str(payload.get("store_ref") or ""),
        )


def build_scalar_index_job(
    request: ScalarIndexJobRequest,
    *,
    build_mode: str,
    backend_kind: str,
    now: datetime,
    ttl_s: float | None,
    store_kind: str,
    store_ref: str,
) -> ScalarIndexJob:
    """Mint the initial ``submitted`` record for a new job claim."""
    content_digest = _stable_digest(
        {
            "table": request.table,
            "table_version": request.table_version,
            "column": request.column,
            "index_type": request.index_type,
        }
    )
    return ScalarIndexJob(
        job_id=request.job_id(),
        status=STATUS_SUBMITTED,
        table=request.table,
        column=request.column,
        index_type=request.index_type,
        build_mode=build_mode,
        backend_kind=backend_kind,
        table_version=request.table_version,
        num_rows=None,
        submitted_at=now,
        started_at=None,
        completed_at=None,
        updated_at=now,
        ttl_s=ttl_s,
        expires_at=(now + _timedelta(ttl_s)) if ttl_s else None,
        terminal_reason=None,
        retry_count=0,
        request_count=1,
        status_history=({"status": STATUS_SUBMITTED, "at": _iso(now), "reason": "requested"},),
        owner_nonce=uuid.uuid4().hex,
        content_digest=content_digest,
        store_kind=store_kind,
        store_ref=store_ref,
    )


@dataclass(frozen=True)
class ScalarIndexJobResult:
    """Outcome of a ``request``: the job plus what happened to it."""

    job: ScalarIndexJob
    created: bool
    reused: bool
    index_result: ScalarIndexResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "reused": self.reused,
            "job": self.job.to_dict(),
            "index_result": (self.index_result.to_params() if self.index_result else None),
        }


# --------------------------------------------------------------------------- #
# Stores                                                                       #
# --------------------------------------------------------------------------- #


class ScalarIndexJobStore:
    """Abstract durable store keyed by ``job_id``."""

    kind = "abstract"

    def claim(self, record: ScalarIndexJob) -> tuple[ScalarIndexJob, bool]:
        raise NotImplementedError

    def get(self, job_id: str) -> ScalarIndexJob | None:
        raise NotImplementedError

    def put(self, record: ScalarIndexJob) -> None:
        raise NotImplementedError

    def list(
        self,
        *,
        status: str | None = None,
        table: str | None = None,
        limit: int | None = None,
    ) -> list[ScalarIndexJob]:
        raise NotImplementedError

    def store_ref(self, job_id: str) -> str:
        raise NotImplementedError


class InMemoryScalarIndexJobStore(ScalarIndexJobStore):
    """In-process store used by tests, single-process builds, and simulation."""

    kind = "in-memory"

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def claim(self, record: ScalarIndexJob) -> tuple[ScalarIndexJob, bool]:
        existing = self._jobs.get(record.job_id)
        if existing is not None:
            return ScalarIndexJob.from_dict(existing), False
        stored = replace(record, store_kind=self.kind, store_ref=self.store_ref(record.job_id))
        self._jobs[record.job_id] = stored.to_dict()
        return stored, True

    def get(self, job_id: str) -> ScalarIndexJob | None:
        payload = self._jobs.get(job_id)
        return ScalarIndexJob.from_dict(payload) if payload is not None else None

    def put(self, record: ScalarIndexJob) -> None:
        stored = replace(record, store_kind=self.kind, store_ref=self.store_ref(record.job_id))
        self._jobs[record.job_id] = stored.to_dict()

    def list(
        self,
        *,
        status: str | None = None,
        table: str | None = None,
        limit: int | None = None,
    ) -> list[ScalarIndexJob]:
        records = [ScalarIndexJob.from_dict(payload) for payload in self._jobs.values()]
        records = _filter_records(records, status=status, table=table)
        records.sort(key=lambda r: (r.updated_at, r.job_id), reverse=True)
        return records[:limit] if limit is not None else records

    def store_ref(self, job_id: str) -> str:
        return f"memory://{SCALAR_INDEX_JOB_TABLE}/{job_id}"


class LanceTableScalarIndexJobStore(ScalarIndexJobStore):
    """Durable store: one internal LanceDB table of jobs keyed by ``job_id``.

    Survives process restarts so a worker / re-run / CLI invocation in a fresh
    process sees the same job. Upserts go through ``merge_insert`` on ``job_id``
    inside a bounded commit-conflict retry loop (0131 catalog standard); reads are
    bounded (``select``/``where``/``limit``) and reload the full record from the
    ``record_json`` blob.
    """

    kind = "lancedb-table"

    def __init__(self, db: Any) -> None:
        if db is None:
            raise ScalarIndexJobError("LanceTableScalarIndexJobStore requires a LanceDB connection")
        self._db = db

    def _schema(self):
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("job_id", pa.string()),
                pa.field("status", pa.string()),
                pa.field("table_name", pa.string()),
                pa.field("column_name", pa.string()),
                pa.field("index_type", pa.string()),
                pa.field("build_mode", pa.string()),
                pa.field("backend_kind", pa.string()),
                pa.field("table_version", pa.int64()),
                pa.field("retry_count", pa.int64()),
                pa.field("request_count", pa.int64()),
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

    def _row(self, record: ScalarIndexJob) -> dict[str, Any]:
        stored = replace(record, store_kind=self.kind, store_ref=self.store_ref(record.job_id))
        payload = stored.to_dict()
        return {
            "job_id": stored.job_id,
            "status": stored.status,
            "table_name": stored.table,
            "column_name": stored.column,
            "index_type": stored.index_type,
            "build_mode": stored.build_mode,
            "backend_kind": stored.backend_kind,
            "table_version": stored.table_version if stored.table_version is not None else -1,
            "retry_count": stored.retry_count,
            "request_count": stored.request_count,
            "submitted_at": _iso(stored.submitted_at) or "",
            "updated_at": _iso(stored.updated_at) or "",
            "expires_at": _iso(stored.expires_at) or "",
            "terminal_reason": stored.terminal_reason or "",
            "record_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        }

    def _ensure_table(self):
        import pyarrow as pa

        if SCALAR_INDEX_JOB_TABLE not in self._table_names():
            empty = pa.Table.from_pylist([], schema=self._schema())
            try:
                self._db.create_table(SCALAR_INDEX_JOB_TABLE, data=empty, mode="create")
            except Exception as exc:  # noqa: BLE001
                # Lost a create race: converge on the winner's table.
                if not _already_exists(exc):
                    raise
        return self._db.open_table(SCALAR_INDEX_JOB_TABLE)

    def _execute_merge(self, record: ScalarIndexJob, *, update_matched: bool) -> None:
        import pyarrow as pa

        data = pa.Table.from_pylist([self._row(record)], schema=self._schema())
        # Bounded retry on retryable commit conflicts: the job table is one shared
        # write target, so concurrent maintenance/build requests race here. Re-opening
        # the latest version and retrying converges instead of dropping a row
        # (SKILLS.md s1).
        #
        # ``update_matched`` selects the semantics:
        #  - False (claim): insert-if-absent. A losing racer whose retry now *matches*
        #    the winner's row no-ops, so the first writer's owner_nonce survives and
        #    exactly one caller reads back its own nonce -> a real first-writer-wins
        #    CAS, preventing duplicate builds/submits.
        #  - True (put): last-writer-wins upsert for lifecycle status transitions,
        #    where the caller already owns the job and is advancing its state.
        last_exc: BaseException | None = None
        for _attempt in range(_JOB_COMMIT_RETRIES + 1):
            table = self._ensure_table()
            try:
                builder = table.merge_insert("job_id")
                if update_matched:
                    builder = builder.when_matched_update_all()
                builder.when_not_matched_insert_all().execute(data)
                return
            except Exception as exc:  # noqa: BLE001
                if not _is_retryable_commit_conflict(exc):
                    raise
                last_exc = exc
        raise ScalarIndexJobError(
            f"scalar-index job write for {record.job_id!r} lost "
            f"{_JOB_COMMIT_RETRIES} consecutive commit races: {last_exc}"
        )

    def _upsert(self, record: ScalarIndexJob) -> None:
        self._execute_merge(record, update_matched=True)

    def _insert_if_absent(self, record: ScalarIndexJob) -> None:
        self._execute_merge(record, update_matched=False)

    def _read_row(self, job_id: str) -> dict[str, Any] | None:
        if SCALAR_INDEX_JOB_TABLE not in self._table_names():
            return None
        table = self._db.open_table(SCALAR_INDEX_JOB_TABLE)
        rows = (
            table.search()
            .where(f"job_id = '{_escape(job_id)}'")
            .select(["record_json"])
            .limit(1)
            .to_arrow()
            .to_pylist()
        )
        if not rows:
            return None
        return json.loads(rows[0]["record_json"])

    def claim(self, record: ScalarIndexJob) -> tuple[ScalarIndexJob, bool]:
        existing = self._read_row(record.job_id)
        if existing is not None:
            return ScalarIndexJob.from_dict(existing), False
        # Insert-if-absent (NOT update-all): if a concurrent racer already inserted
        # this job_id, our retry matches its row and no-ops, so the read-back reflects
        # the true first writer's owner_nonce -- exactly one claimer wins.
        self._insert_if_absent(record)
        winner = self._read_row(record.job_id)
        if winner is None:  # pragma: no cover - should not happen
            return replace(record, store_kind=self.kind, store_ref=self.store_ref(record.job_id)), True
        won = str(winner.get("owner_nonce")) == record.owner_nonce
        return ScalarIndexJob.from_dict(winner), won

    def get(self, job_id: str) -> ScalarIndexJob | None:
        payload = self._read_row(job_id)
        return ScalarIndexJob.from_dict(payload) if payload is not None else None

    def put(self, record: ScalarIndexJob) -> None:
        self._upsert(record)

    def list(
        self,
        *,
        status: str | None = None,
        table: str | None = None,
        limit: int | None = None,
    ) -> list[ScalarIndexJob]:
        if SCALAR_INDEX_JOB_TABLE not in self._table_names():
            return []
        handle = self._db.open_table(SCALAR_INDEX_JOB_TABLE)
        query = handle.search().select(["record_json"])
        predicates = []
        if status is not None:
            predicates.append(f"status = '{_escape(status)}'")
        if table is not None:
            predicates.append(f"table_name = '{_escape(table)}'")
        if predicates:
            query = query.where(" AND ".join(predicates))
        records: list[ScalarIndexJob] = []
        for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
            for row in batch.to_pylist():
                records.append(ScalarIndexJob.from_dict(json.loads(row["record_json"])))
        records = _filter_records(records, status=status, table=table)
        records.sort(key=lambda r: (r.updated_at, r.job_id), reverse=True)
        return records[:limit] if limit is not None else records

    def store_ref(self, job_id: str) -> str:
        return f"lancedb://{SCALAR_INDEX_JOB_TABLE}/{job_id}"


def _filter_records(
    records: Iterable[ScalarIndexJob],
    *,
    status: str | None,
    table: str | None,
) -> list[ScalarIndexJob]:
    result = list(records)
    if status is not None:
        result = [r for r in result if r.status == status]
    if table is not None:
        result = [r for r in result if r.table == table]
    return result


# --------------------------------------------------------------------------- #
# Coordinator                                                                  #
# --------------------------------------------------------------------------- #


def _normalize_response(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return dict(response)
    if response is None:
        return {"status": STATUS_ACTIVE}
    return {"status": STATUS_ACTIVE}


def _request_dict(record: ScalarIndexJob, request: ScalarIndexJobRequest) -> dict[str, Any]:
    """Secret-free envelope handed to an async submit/status hook."""
    return {
        "kind": SCALAR_INDEX_JOB_KIND,
        "job_id": record.job_id,
        "table": request.table,
        "column": request.column,
        "index_type": request.index_type,
        "table_version": request.table_version,
        "replace": request.replace,
    }


class ScalarIndexJobCoordinator:
    """Owns the dedup/submit/poll/retry/cancel/reconcile lifecycle over a store.

    ``build_fn`` runs a synchronous inline build and returns a 0079
    ``ScalarIndexResult``. ``submit_fn``/``status_fn`` drive an asynchronous
    remote build. With neither attached, an async-mode request records a planned
    ``submitted`` job (reason: no client) so it is still tracked and resumable.
    """

    def __init__(
        self,
        store: ScalarIndexJobStore,
        *,
        build_fn: Callable[[ScalarIndexJobRequest], ScalarIndexResult] | None = None,
        submit_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        status_fn: Callable[..., Any] | None = None,
        ttl_s: float | None = DEFAULT_SCALAR_INDEX_JOB_TTL_S,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self._build_fn = build_fn
        self._submit_fn = submit_fn
        self._status_fn = status_fn
        self._ttl_s = ttl_s
        self._now_fn = now_fn or _utcnow

    def _now(self) -> datetime:
        return self._now_fn()

    # -- request ----------------------------------------------------------- #

    def request(
        self,
        request: ScalarIndexJobRequest,
        *,
        build_mode: str,
        backend_kind: str,
        wait: bool = False,
    ) -> ScalarIndexJobResult:
        now = self._now()
        job_id = request.job_id()
        existing = self.store.get(job_id)
        if existing is not None:
            existing = self._maybe_expire(existing, now)

        if existing is not None and existing.is_complete:
            bumped = existing.touch_request(now)
            self.store.put(bumped)
            return ScalarIndexJobResult(bumped, created=False, reused=True)

        if existing is not None and existing.is_in_flight:
            bumped = existing.touch_request(now)
            self.store.put(bumped)
            if wait and self._status_fn is not None:
                polled = self.poll(job_id, request)
                return ScalarIndexJobResult(polled, created=False, reused=True)
            return ScalarIndexJobResult(bumped, created=False, reused=True)

        if existing is not None and existing.status in RESUBMITTABLE_STATUSES:
            retried = existing.retrying(
                now, ttl_s=self._ttl_s, build_mode=build_mode, backend_kind=backend_kind
            )
            self.store.put(retried)
            return self._run(retried, request, wait=wait)

        # No usable job -- claim the key (race-safe via owner_nonce).
        fresh = build_scalar_index_job(
            request,
            build_mode=build_mode,
            backend_kind=backend_kind,
            now=now,
            ttl_s=self._ttl_s,
            store_kind=self.store.kind,
            store_ref=self.store.store_ref(job_id),
        )
        winner, created = self.store.claim(fresh)
        if not created:
            winner = self._maybe_expire(winner, now)
            bumped = winner.touch_request(now)
            self.store.put(bumped)
            return ScalarIndexJobResult(bumped, created=False, reused=True)
        return self._run(winner, request, wait=wait)

    def _run(
        self, record: ScalarIndexJob, request: ScalarIndexJobRequest, *, wait: bool
    ) -> ScalarIndexJobResult:
        now = self._now()
        if self._build_fn is not None:
            active = record.with_status(STATUS_ACTIVE, now)
            self.store.put(active)
            try:
                result = self._build_fn(request)
            except Exception as exc:  # noqa: BLE001 - backend build errors vary
                reason = f"permission denied: {exc}" if _is_permission_denied_error(exc) else str(exc)
                failed = active.with_status(STATUS_FAILED, self._now(), reason=reason)
                self.store.put(failed)
                return ScalarIndexJobResult(failed, created=True, reused=False)
            status = _INDEX_RESULT_TO_JOB_STATUS.get(result.status, STATUS_FAILED)
            done = active.with_status(
                status, self._now(), reason=result.reason, num_rows=result.num_rows
            )
            self.store.put(done)
            return ScalarIndexJobResult(done, created=True, reused=False, index_result=result)

        if self._submit_fn is not None:
            try:
                response = self._submit_fn(_request_dict(record, request))
            except Exception as exc:  # noqa: BLE001 - remote submit errors vary
                reason = f"permission denied: {exc}" if _is_permission_denied_error(exc) else str(exc)
                failed = record.with_status(STATUS_FAILED, self._now(), reason=reason)
                self.store.put(failed)
                return ScalarIndexJobResult(failed, created=True, reused=False)
            normalized = _normalize_response(response)
            status = normalized.get("status", STATUS_ACTIVE)
            updated = record.with_status(
                status, now, reason=normalized.get("reason"), num_rows=normalized.get("num_rows")
            )
            self.store.put(updated)
            if wait and not updated.is_terminal and self._status_fn is not None:
                polled = self.poll(record.job_id, request)
                return ScalarIndexJobResult(polled, created=True, reused=False)
            return ScalarIndexJobResult(updated, created=True, reused=False)

        planned = record.with_status(
            STATUS_SUBMITTED, now, reason="no async scalar-index-job client is attached"
        )
        self.store.put(planned)
        return ScalarIndexJobResult(planned, created=True, reused=False)

    # -- poll / reconcile -------------------------------------------------- #

    def _invoke_status_fn(self, job_id: str, envelope: Mapping[str, Any]) -> Any:
        """Call ``status_fn`` tolerating three known arities; other errors propagate.

        Only a ``TypeError`` (arity mismatch) falls through to the next shape; a
        ``TypeError`` from the final shape -- or any other error -- propagates to
        ``poll``'s handler, which records the job failed rather than silently
        swallowing it.
        """
        try:
            return self._status_fn(job_id=job_id, request=dict(envelope))
        except TypeError:
            pass
        try:
            return self._status_fn(job_id, dict(envelope))
        except TypeError:
            pass
        return self._status_fn(job_id)

    def poll(self, job_id: str, request: ScalarIndexJobRequest | None = None) -> ScalarIndexJob:
        now = self._now()
        record = self.store.get(job_id)
        if record is None:
            raise ScalarIndexJobError(f"no scalar-index job for id {job_id!r}")
        record = self._maybe_expire(record, now)
        if record.is_terminal or self._status_fn is None:
            return record
        envelope = _request_dict(record, request) if request is not None else {"job_id": job_id}
        try:
            response = self._invoke_status_fn(job_id, envelope)
        except Exception as exc:  # noqa: BLE001 - remote status errors vary
            reason = f"permission denied: {exc}" if _is_permission_denied_error(exc) else str(exc)
            failed = record.with_status(STATUS_FAILED, self._now(), reason=reason)
            self.store.put(failed)
            return failed
        normalized = _normalize_response(response)
        status = normalized.get("status", record.status)
        updated = record.with_status(
            status, now, reason=normalized.get("reason"), num_rows=normalized.get("num_rows")
        )
        self.store.put(updated)
        return updated

    def _in_flight_jobs(self) -> list[ScalarIndexJob]:
        """Only submitted/active jobs -- the sole rows reconcile/expiry can change.

        Pushes the status filter into the store query so terminal rows (which
        ``_maybe_expire`` short-circuits and ``poll`` returns unchanged) are never
        materialized. Terminal rows dominate a long-lived job table, so scanning
        only in-flight rows keeps reconcile bounded as the table grows.
        """
        jobs: list[ScalarIndexJob] = []
        for status in (STATUS_SUBMITTED, STATUS_ACTIVE):
            jobs.extend(self.store.list(status=status))
        return jobs

    def reconcile(self) -> list[ScalarIndexJob]:
        """Poll every in-flight job and expire stale ones; return changed records."""
        now = self._now()
        changed: list[ScalarIndexJob] = []
        for record in self._in_flight_jobs():
            updated = self._maybe_expire(record, now)
            if updated.is_in_flight and self._status_fn is not None:
                polled = self.poll(updated.job_id)
                if polled is not updated:
                    updated = polled
            if updated is not record:
                changed.append(updated)
        return changed

    # -- retry / cancel / expire ------------------------------------------- #

    def retry(self, job_id: str, request: ScalarIndexJobRequest | None = None) -> ScalarIndexJob:
        now = self._now()
        record = self.store.get(job_id)
        if record is None:
            raise ScalarIndexJobError(f"no scalar-index job for id {job_id!r}")
        if record.status not in RESUBMITTABLE_STATUSES:
            raise ScalarIndexJobError(
                f"scalar-index job {job_id!r} is {record.status!r}; only failed/canceled/"
                "expired jobs can be retried"
            )
        retried = record.retrying(now, ttl_s=self._ttl_s)
        self.store.put(retried)
        req = request or _request_from_record(record)
        return self._run(retried, req, wait=False).job

    def cancel(self, job_id: str, *, reason: str | None = None) -> ScalarIndexJob:
        now = self._now()
        record = self.store.get(job_id)
        if record is None:
            raise ScalarIndexJobError(f"no scalar-index job for id {job_id!r}")
        if record.status in {STATUS_COMPLETE, STATUS_CANCELED}:
            return record
        canceled = record.with_status(STATUS_CANCELED, now, reason=reason or "canceled by caller")
        self.store.put(canceled)
        return canceled

    def expire_due(self) -> list[ScalarIndexJob]:
        now = self._now()
        expired: list[ScalarIndexJob] = []
        for record in self._in_flight_jobs():  # only in-flight jobs can expire
            updated = self._maybe_expire(record, now)
            if updated is not record:
                expired.append(updated)
        return expired

    def _maybe_expire(self, record: ScalarIndexJob, now: datetime) -> ScalarIndexJob:
        if record.is_terminal:
            return record
        if record.is_expired(now):
            expired = record.with_status(STATUS_EXPIRED, now, reason="ttl elapsed")
            self.store.put(expired)
            return expired
        return record


def _request_from_record(record: ScalarIndexJob) -> ScalarIndexJobRequest:
    return ScalarIndexJobRequest(
        table=record.table,
        column=record.column,
        index_type=record.index_type,
        table_version=record.table_version,
        replace=True,
    )


# --------------------------------------------------------------------------- #
# Lake resolution + orchestration                                              #
# --------------------------------------------------------------------------- #


def _lake_db(lake: Any) -> Any | None:
    db = getattr(lake, "_db", None)
    if db is None:
        return None
    if all(hasattr(db, name) for name in ("create_table", "open_table", "list_tables")):
        return db
    return None


def resolve_scalar_index_job_store(lake: Any) -> ScalarIndexJobStore | None:
    """Store used by *write* paths (request/reconcile): durable when possible.

    Unlike the 0121 prewarm store (opt-in), the scalar-index job table is created
    automatically when a build is requested, so the default researcher experience
    stays automatic. Set ``lake.scalar_index_jobs_durable = False`` to opt out, or
    attach ``lake.scalar_index_job_store`` to override.
    """
    explicit = getattr(lake, "scalar_index_job_store", None)
    if isinstance(explicit, ScalarIndexJobStore):
        return explicit
    if not getattr(lake, "scalar_index_jobs_durable", True):
        return None
    db = _lake_db(lake)
    if db is not None:
        return LanceTableScalarIndexJobStore(db)
    return None


def open_scalar_index_job_store(lake: Any) -> ScalarIndexJobStore | None:
    """Store used by *read* paths (manifest/list): durable ONLY if it exists.

    The aligned-dataset read path must never create the table (reads stay
    read-only). Returns ``None`` until a write path has created the table.
    """
    explicit = getattr(lake, "scalar_index_job_store", None)
    if isinstance(explicit, ScalarIndexJobStore):
        return explicit
    db = _lake_db(lake)
    if db is None:
        return None
    store = LanceTableScalarIndexJobStore(db)
    if SCALAR_INDEX_JOB_TABLE in store._table_names():
        return store
    return None


def _resolve_table_version(lake: Any, table: str) -> int | None:
    try:
        handle = lake.table(table)
    except Exception:  # noqa: BLE001 - unknown/absent table -> no version pin
        return None
    version = getattr(handle, "version", None)
    try:
        return int(version) if version is not None else None
    except (TypeError, ValueError):
        return None


def _coordinator_for(
    lake: Any,
    store: ScalarIndexJobStore,
    capability: ScalarIndexCapability,
    *,
    ttl_s: float | None,
    now_fn: Callable[[], datetime] | None,
) -> ScalarIndexJobCoordinator:
    build_fn: Callable[[ScalarIndexJobRequest], ScalarIndexResult] | None = None
    submit_fn = None
    status_fn = None
    if capability.mode == MODE_SYNCHRONOUS:
        def build_fn(req: ScalarIndexJobRequest, _lake=lake) -> ScalarIndexResult:
            from lancedb_robotics.indexing import build_scalar_index

            return build_scalar_index(
                _lake,
                table=req.table,
                column=req.column,
                replace=req.replace,
                index_type=req.index_type,
            )
    elif capability.mode == MODE_ASYNCHRONOUS:
        submit_hook = _async_hook(lake)
        status_hook = getattr(lake, "scalar_index_job_status", None)
        submit_fn = (lambda req: submit_hook(request=dict(req))) if submit_hook else None
        status_fn = status_hook if callable(status_hook) else None
    return ScalarIndexJobCoordinator(
        store,
        build_fn=build_fn,
        submit_fn=submit_fn,
        status_fn=status_fn,
        ttl_s=ttl_s,
        now_fn=now_fn,
    )


def request_scalar_index(
    lake: Any,
    *,
    table: str,
    column: str,
    index_type: str = SCALAR_INDEX_TYPE,
    replace: bool = False,
    store: ScalarIndexJobStore | None = None,
    capability: ScalarIndexCapability | None = None,
    ttl_s: float | None = DEFAULT_SCALAR_INDEX_JOB_TTL_S,
    now_fn: Callable[[], datetime] | None = None,
) -> ScalarIndexJobResult:
    """Request a scalar predicate index build as a durable, idempotent job.

    On a synchronous backend this builds inline and records a ``complete`` job; on
    an asynchronous backend it submits (or plans) a remote build and records a
    pending job. On an ``unsupported``/``permission_denied`` backend it does NOT
    persist a job -- it returns an ephemeral terminal ``skipped`` result carrying
    the capability reason, so predicate pushdown continues and no dead job rows
    accumulate that would block recovery once the backend changes.
    """
    capability = capability or probe_scalar_index_capability(lake)
    request = ScalarIndexJobRequest(
        table=table,
        column=column,
        index_type=index_type,
        table_version=_resolve_table_version(lake, table),
        replace=replace,
    )

    if not capability.buildable:
        now = (now_fn or _utcnow)()
        ephemeral = ScalarIndexJob(
            job_id=request.job_id(),
            status=STATUS_SKIPPED,
            table=table,
            column=column,
            index_type=index_type,
            build_mode=capability.mode,
            backend_kind=capability.backend_kind,
            table_version=request.table_version,
            num_rows=None,
            submitted_at=now,
            started_at=None,
            completed_at=now,
            updated_at=now,
            ttl_s=None,
            expires_at=None,
            terminal_reason=capability.reason,
            retry_count=0,
            request_count=1,
            status_history=({"status": STATUS_SKIPPED, "at": _iso(now), "reason": capability.reason},),
            owner_nonce="",
            content_digest="",
        )
        index_result = ScalarIndexResult(
            table=table,
            column=column,
            status="skipped",
            index_type=index_type,
            reason=capability.reason,
        )
        return ScalarIndexJobResult(ephemeral, created=False, reused=False, index_result=index_result)

    resolved_store = store or resolve_scalar_index_job_store(lake) or InMemoryScalarIndexJobStore()
    coordinator = _coordinator_for(
        lake, resolved_store, capability, ttl_s=ttl_s, now_fn=now_fn
    )
    return coordinator.request(
        request, build_mode=capability.mode, backend_kind=capability.backend_kind
    )


#: Recommended hot-predicate columns per aligned table (reuse the 0079 sets).
ALIGNED_PREDICATE_INDEX_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "aligned_ticks": ALIGNED_TICK_PREDICATE_INDEX_COLUMNS,
    "aligned_frames": ALIGNED_FRAME_PREDICATE_INDEX_COLUMNS,
}


def request_aligned_predicate_index_jobs(
    lake: Any,
    *,
    include_frames: bool = True,
    replace: bool = False,
    store: ScalarIndexJobStore | None = None,
    ttl_s: float | None = DEFAULT_SCALAR_INDEX_JOB_TTL_S,
    now_fn: Callable[[], datetime] | None = None,
) -> list[ScalarIndexJobResult]:
    """Request jobs for every recommended aligned hot-predicate column.

    This is the automatic path: alignment materialization and ``lake maintain``
    call it so researchers using default APIs get indexes built for them without
    ever naming a column. One capability probe is shared across all columns.
    """
    capability = probe_scalar_index_capability(lake)
    resolved_store = store or resolve_scalar_index_job_store(lake) or InMemoryScalarIndexJobStore()
    results: list[ScalarIndexJobResult] = []
    tables = ["aligned_ticks"] + (["aligned_frames"] if include_frames else [])
    for table in tables:
        for column in ALIGNED_PREDICATE_INDEX_COLUMNS_BY_TABLE[table]:
            results.append(
                request_scalar_index(
                    lake,
                    table=table,
                    column=column,
                    replace=replace,
                    store=resolved_store,
                    capability=capability,
                    ttl_s=ttl_s,
                    now_fn=now_fn,
                )
            )
    return results


def reconcile_scalar_index_jobs(
    lake: Any,
    *,
    store: ScalarIndexJobStore | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> list[dict[str, Any]]:
    """Poll pending jobs and expire stale ones (maintenance hook).

    Returns the changed job records (``to_dict``). A no-op returning ``[]`` when no
    job table exists, so it is safe to call on every ``lake maintain`` run.
    """
    resolved_store = store or open_scalar_index_job_store(lake)
    if resolved_store is None:
        return []
    capability = probe_scalar_index_capability(lake)
    coordinator = _coordinator_for(
        lake, resolved_store, capability, ttl_s=DEFAULT_SCALAR_INDEX_JOB_TTL_S, now_fn=now_fn
    )
    return [record.to_dict() for record in coordinator.reconcile()]


def scalar_index_jobs_for_table(
    lake: Any,
    table: str,
    *,
    store: ScalarIndexJobStore | None = None,
) -> dict[str, ScalarIndexJob]:
    """Read-only map of ``column -> latest job`` for ``table`` (manifest surfacing).

    Uses :func:`open_scalar_index_job_store` so it never creates the table. Returns
    the most-recently-updated job per column.
    """
    resolved_store = store if store is not None else open_scalar_index_job_store(lake)
    if resolved_store is None:
        return {}
    latest: dict[str, ScalarIndexJob] = {}
    for record in resolved_store.list(table=table):
        current = latest.get(record.column)
        if current is None or record.updated_at > current.updated_at:
            latest[record.column] = record
    return latest
