"""Lifecycle catalog for internal epoch-permutation artifacts (backlog 0131).

Backlog 0077 persists a deterministic epoch order as an internal LanceDB
``(row_id, split_id)`` permutation table. The table is deterministically named by
a content digest of ``(row_plan_id, snapshot_id, table_versions, shuffle_seed,
epoch, ordered_row_ids)`` and reused whenever the same order is requested again.
That reuse is correct, but nothing tracked *which* internal tables exist, who
still needs them, or when they were last used -- so a lake that ran many
experiments accumulated unbounded ``__lancedb_robotics_epoch_perm_*`` tables with
no safe way to reclaim them.

This module is the lifecycle layer 0131 adds on top of those tables. It is
**additive**: 0077 still owns the deterministic naming/reuse; the public
``lake.training.dataset(...)`` arguments are unchanged. What this adds is:

* a durable **catalog** (one internal, non-canonical LanceDB table keyed by the
  permutation table name) recording each artifact's owner row/epoch plan ids,
  snapshot id, pinned table versions, seed, epoch, worker partition, row count,
  created/last-used timestamps, use count, and retention policy;
* **discovery** of every internal permutation table -- cataloged ones with full
  metadata, plus any stray ``__lancedb_robotics_epoch_perm_*`` table left by an
  older writer (discovered by prefix, reported as ``cataloged=False``);
* **cleanup** that drops unreferenced internal permutation tables while
  preserving any table still referenced by a retained training run/report (by
  ``row_plan_id``/``epoch_plan_id``) or pinned by a ``keep`` retention policy,
  reports exactly what it changed, and is safe to run repeatedly;
* **accounting** of the artifact count / rows / estimated bytes for training and
  benchmark reporting.

Design notes (recorded in the task record / decision file):

* **The catalog key is the 0077 permutation table name.** No new identity is
  invented; two workers / a re-run that build the same order collapse to one
  catalog row via ``merge_insert`` on ``permutation_table``.
* **Reference model = plan-id linkage, not embedded-manifest scanning.** A
  retained ``training_runs`` / ``training_reports`` row records the
  ``row_plan_id`` and ``epoch_plan_id`` it trained against; an artifact is
  "referenced" iff its owner row/epoch plan id appears among those. This reads
  only two projected scalar columns per manifest table (bounded by run count,
  which is orders of magnitude smaller than the observation grain) -- no
  full-manifest hydration and no coupling to each catalog's JSON shape.
* **``use_count`` / ``last_used_at`` are advisory.** The correctness-critical
  fields (identity, plan ids, retention policy) converge under ``merge_insert``;
  the reuse counters are best-effort under concurrent writers and are never used
  to decide deletion.
* **The catalog table name deliberately does not share the
  ``__lancedb_robotics_epoch_perm_`` prefix**, so the prefix scan that finds
  artifacts never mistakes the catalog for one of its own rows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

#: Prefix every internal 0077 epoch-permutation table shares. A local copy (not
#: imported) so this module has no ``training.py`` dependency; kept identical to
#: ``lancedb_robotics.training.EPOCH_PERMUTATION_TABLE_PREFIX`` and pinned equal by
#: ``tests/test_training_permutation_catalog.py``.
EPOCH_PERMUTATION_TABLE_PREFIX = "__lancedb_robotics_epoch_perm_"

PERMUTATION_CATALOG_KIND = "lancedb-robotics/epoch-permutation-artifact/v1"
#: Note: intentionally *not* under ``EPOCH_PERMUTATION_TABLE_PREFIX`` so the
#: artifact prefix scan never treats the catalog as an artifact.
PERMUTATION_CATALOG_TABLE = "__lancedb_robotics_perm_catalog"

RETENTION_AUTO = "auto"
RETENTION_KEEP = "keep"
RETENTION_POLICIES = (RETENTION_AUTO, RETENTION_KEEP)

#: Two ``uint64`` columns (``row_id``, ``split_id``) => 16 bytes/row. Used for a
#: bounded, dependency-free ordered-row-id byte estimate in accounting; this is
#: the logical size of the ordering, not the on-disk file size.
PERMUTATION_ROW_BYTES = 16

_SCAN_BATCH_SIZE = 2048

#: The catalog table is a single shared write target: every local-lake dataset
#: build (including forked/spawned DataLoader workers and parallel experiments)
#: upserts into it, so concurrent writers race at commit time. Lance arbitrates
#: with a *retryable* commit conflict for the loser; because the upsert is an
#: idempotent ``merge_insert`` on the table name, re-reading and retrying
#: converges (mirrors ``enrich.py`` / BUG-04). Bounded so a pathological race
#: fails loudly instead of spinning.
_CATALOG_COMMIT_RETRIES = 8


class PermutationCatalogError(Exception):
    """Raised for permutation-artifact catalog store/lifecycle misuse."""


def _is_retryable_commit_conflict(exc: BaseException) -> bool:
    """True when ``exc`` is a Lance optimistic-concurrency commit conflict.

    Kept local (this module imports no ``enrich``/``training`` code) but matches
    ``enrich._is_retryable_commit_conflict`` exactly, so the two write paths treat
    concurrent-writer preemption identically.
    """
    return "commit conflict" in str(exc).lower()


def _already_exists(exc: BaseException) -> bool:
    """True when ``exc`` is a "table already exists" race on ``create_table``."""
    message = str(exc).lower()
    return "already exists" in message or "alreadyexists" in message


# --------------------------------------------------------------------------- #
# Serialization helpers (kept local; this module imports no training.py types).#
# --------------------------------------------------------------------------- #


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


def _escape(value: str) -> str:
    return str(value).replace("'", "''")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


# --------------------------------------------------------------------------- #
# The durable catalog record                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PermutationArtifactRecord:
    """One internal epoch-permutation artifact tracked for lifecycle/cleanup.

    Immutable: reuse updates return a new record (``mark_used`` /
    ``with_retention_policy``) so the store owns the single source of truth.
    """

    permutation_table: str
    permutation_ref: str
    backend_kind: str
    permutation_source: str
    row_plan_id: str | None
    epoch_plan_id: str | None
    dataset_id: str | None
    snapshot_name: str | None
    table_versions: tuple[dict[str, Any], ...]
    shuffle_seed: int | None
    epoch: int | None
    worker_id: int | None
    num_workers: int | None
    resume_from: int | None
    row_count: int
    created_at: datetime
    last_used_at: datetime
    use_count: int
    retention_policy: str
    kind: str = PERMUTATION_CATALOG_KIND
    store_kind: str = "in-memory"
    store_ref: str = ""

    @property
    def estimated_bytes(self) -> int:
        return max(0, int(self.row_count)) * PERMUTATION_ROW_BYTES

    def mark_used(self, now: datetime, *, source: str | None = None) -> PermutationArtifactRecord:
        """Return a copy reflecting one more reuse (advisory counters only)."""
        return replace(
            self,
            last_used_at=now,
            use_count=self.use_count + 1,
            permutation_source=source or self.permutation_source,
        )

    def with_retention_policy(self, policy: str) -> PermutationArtifactRecord:
        if policy not in RETENTION_POLICIES:
            raise PermutationCatalogError(
                f"unknown retention policy {policy!r}; expected one of {RETENTION_POLICIES}"
            )
        return replace(self, retention_policy=policy)

    def is_referenced_by(self, protected_plan_ids: frozenset[str]) -> bool:
        """True iff a retained run/report pins this artifact's row/epoch plan."""
        return (
            (self.row_plan_id is not None and self.row_plan_id in protected_plan_ids)
            or (self.epoch_plan_id is not None and self.epoch_plan_id in protected_plan_ids)
        )

    def to_dict(self, *, cataloged: bool = True) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "cataloged": cataloged,
            "permutation_table": self.permutation_table,
            "permutation_ref": self.permutation_ref,
            "backend_kind": self.backend_kind,
            "permutation_source": self.permutation_source,
            "row_plan_id": self.row_plan_id,
            "epoch_plan_id": self.epoch_plan_id,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "table_versions": [dict(item) for item in self.table_versions],
            "shuffle_seed": self.shuffle_seed,
            "epoch": self.epoch,
            "worker": {
                "id": self.worker_id,
                "num_workers": self.num_workers,
                "resume_from": self.resume_from,
            },
            "row_count": self.row_count,
            "estimated_bytes": self.estimated_bytes,
            "created_at": _iso(self.created_at),
            "last_used_at": _iso(self.last_used_at),
            "use_count": self.use_count,
            "retention_policy": self.retention_policy,
            "store_kind": self.store_kind,
            "store_ref": self.store_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PermutationArtifactRecord:
        worker = payload.get("worker") or {}
        created = _parse_iso(payload.get("created_at")) or _utcnow()
        return cls(
            permutation_table=str(payload["permutation_table"]),
            permutation_ref=str(payload.get("permutation_ref", "")),
            backend_kind=str(payload.get("backend_kind", "")),
            permutation_source=str(payload.get("permutation_source", "")),
            row_plan_id=payload.get("row_plan_id"),
            epoch_plan_id=payload.get("epoch_plan_id"),
            dataset_id=payload.get("dataset_id"),
            snapshot_name=payload.get("snapshot_name"),
            table_versions=tuple(dict(item) for item in payload.get("table_versions", [])),
            shuffle_seed=payload.get("shuffle_seed"),
            epoch=payload.get("epoch"),
            worker_id=worker.get("id", payload.get("worker_id")),
            num_workers=worker.get("num_workers", payload.get("num_workers")),
            resume_from=worker.get("resume_from", payload.get("resume_from")),
            row_count=int(payload.get("row_count", 0)),
            created_at=created,
            last_used_at=_parse_iso(payload.get("last_used_at")) or created,
            use_count=int(payload.get("use_count", 1)),
            retention_policy=str(payload.get("retention_policy", RETENTION_AUTO)),
            kind=str(payload.get("kind", PERMUTATION_CATALOG_KIND)),
            store_kind=str(payload.get("store_kind", "in-memory")),
            store_ref=str(payload.get("store_ref", "")),
        )


def build_permutation_artifact_record(
    *,
    permutation_table: str,
    permutation_ref: str,
    backend_kind: str,
    permutation_source: str,
    row_plan_id: str | None,
    epoch_plan_id: str | None,
    dataset_id: str | None,
    snapshot_name: str | None,
    table_versions: Sequence[Mapping[str, Any]],
    shuffle_seed: int | None,
    epoch: int | None,
    worker_id: int | None,
    num_workers: int | None,
    resume_from: int | None,
    row_count: int,
    now: datetime,
    retention_policy: str = RETENTION_AUTO,
) -> PermutationArtifactRecord:
    """Build a fresh (``use_count=1``) catalog record for a permutation table."""
    return PermutationArtifactRecord(
        permutation_table=permutation_table,
        permutation_ref=permutation_ref,
        backend_kind=backend_kind,
        permutation_source=permutation_source,
        row_plan_id=row_plan_id,
        epoch_plan_id=epoch_plan_id,
        dataset_id=dataset_id,
        snapshot_name=snapshot_name,
        table_versions=tuple(dict(item) for item in table_versions),
        shuffle_seed=shuffle_seed,
        epoch=epoch,
        worker_id=worker_id,
        num_workers=num_workers,
        resume_from=resume_from,
        row_count=int(row_count),
        created_at=now,
        last_used_at=now,
        use_count=1,
        retention_policy=retention_policy,
    )


# --------------------------------------------------------------------------- #
# Stores                                                                       #
# --------------------------------------------------------------------------- #


class PermutationArtifactStore:
    """Where permutation-artifact records live, keyed by ``permutation_table``."""

    kind = "abstract"

    def get(self, permutation_table: str) -> PermutationArtifactRecord | None:
        raise NotImplementedError

    def put(self, record: PermutationArtifactRecord) -> None:
        raise NotImplementedError

    def delete(self, permutation_table: str) -> None:
        raise NotImplementedError

    def list(self) -> list[PermutationArtifactRecord]:
        raise NotImplementedError

    def store_ref(self, permutation_table: str) -> str:
        raise NotImplementedError


class InMemoryPermutationArtifactStore(PermutationArtifactStore):
    """In-process store for tests and single-process loaders."""

    kind = "in-memory"

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def get(self, permutation_table: str) -> PermutationArtifactRecord | None:
        payload = self._records.get(permutation_table)
        return PermutationArtifactRecord.from_dict(payload) if payload is not None else None

    def put(self, record: PermutationArtifactRecord) -> None:
        stored = replace(
            record, store_kind=self.kind, store_ref=self.store_ref(record.permutation_table)
        )
        self._records[record.permutation_table] = stored.to_dict()

    def delete(self, permutation_table: str) -> None:
        self._records.pop(permutation_table, None)

    def list(self) -> list[PermutationArtifactRecord]:
        records = [PermutationArtifactRecord.from_dict(p) for p in self._records.values()]
        records.sort(key=lambda r: (r.last_used_at, r.permutation_table), reverse=True)
        return records

    def store_ref(self, permutation_table: str) -> str:
        return f"memory://{PERMUTATION_CATALOG_TABLE}/{permutation_table}"


class LanceTablePermutationArtifactStore(PermutationArtifactStore):
    """Durable store: one internal LanceDB catalog table keyed by table name.

    Survives process restarts and multiple workers. Upserts go through
    ``merge_insert`` on ``permutation_table``; reads are bounded (projected
    ``select`` + streamed ``to_batches``) and reload the full record from the
    ``record_json`` blob.
    """

    kind = "lancedb-table"

    def __init__(self, db: Any) -> None:
        if db is None:
            raise PermutationCatalogError(
                "LanceTablePermutationArtifactStore requires a LanceDB connection"
            )
        self._db = db

    def _schema(self):
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("permutation_table", pa.string()),
                pa.field("backend_kind", pa.string()),
                pa.field("row_plan_id", pa.string()),
                pa.field("epoch_plan_id", pa.string()),
                pa.field("dataset_id", pa.string()),
                pa.field("snapshot_name", pa.string()),
                pa.field("shuffle_seed", pa.int64()),
                pa.field("epoch", pa.int64()),
                pa.field("row_count", pa.int64()),
                pa.field("use_count", pa.int64()),
                pa.field("retention_policy", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("last_used_at", pa.string()),
                pa.field("record_json", pa.string()),
            ]
        )

    def _table_names(self) -> set[str]:
        response = self._db.list_tables()
        tables = getattr(response, "tables", response)
        return {str(name) for name in (tables or [])}

    def _ensure_table(self):
        import pyarrow as pa

        if PERMUTATION_CATALOG_TABLE not in self._table_names():
            empty = pa.Table.from_pylist([], schema=self._schema())
            try:
                self._db.create_table(PERMUTATION_CATALOG_TABLE, data=empty, mode="create")
            except Exception as exc:  # noqa: BLE001
                # Another worker created it first between the check and the create;
                # converge by opening the winner's table rather than failing.
                if not _already_exists(exc):
                    raise
        return self._db.open_table(PERMUTATION_CATALOG_TABLE)

    def _row(self, record: PermutationArtifactRecord) -> dict[str, Any]:
        stored = replace(
            record, store_kind=self.kind, store_ref=self.store_ref(record.permutation_table)
        )
        payload = stored.to_dict()
        return {
            "permutation_table": stored.permutation_table,
            "backend_kind": stored.backend_kind,
            "row_plan_id": stored.row_plan_id or "",
            "epoch_plan_id": stored.epoch_plan_id or "",
            "dataset_id": stored.dataset_id or "",
            "snapshot_name": stored.snapshot_name or "",
            "shuffle_seed": stored.shuffle_seed if stored.shuffle_seed is not None else -1,
            "epoch": stored.epoch if stored.epoch is not None else -1,
            "row_count": stored.row_count,
            "use_count": stored.use_count,
            "retention_policy": stored.retention_policy,
            "created_at": _iso(stored.created_at) or "",
            "last_used_at": _iso(stored.last_used_at) or "",
            "record_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        }

    def get(self, permutation_table: str) -> PermutationArtifactRecord | None:
        if PERMUTATION_CATALOG_TABLE not in self._table_names():
            return None
        table = self._db.open_table(PERMUTATION_CATALOG_TABLE)
        rows = (
            table.search()
            .where(f"permutation_table = '{_escape(permutation_table)}'")
            .select(["record_json"])
            .limit(1)
            .to_arrow()
            .to_pylist()
        )
        if not rows:
            return None
        return PermutationArtifactRecord.from_dict(json.loads(rows[0]["record_json"]))

    def put(self, record: PermutationArtifactRecord) -> None:
        import pyarrow as pa

        data = pa.Table.from_pylist([self._row(record)], schema=self._schema())
        # Bounded retry on retryable commit conflicts: the catalog is one shared
        # write target, so concurrent dataset builds race here. merge_insert on the
        # table name is idempotent, so re-opening the latest version and retrying
        # converges instead of silently dropping the loser's row (SKILLS.md §1).
        last_exc: BaseException | None = None
        for _attempt in range(_CATALOG_COMMIT_RETRIES + 1):
            table = self._ensure_table()
            try:
                (
                    table.merge_insert("permutation_table")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(data)
                )
                return
            except Exception as exc:  # noqa: BLE001
                if not _is_retryable_commit_conflict(exc):
                    raise
                last_exc = exc
        raise PermutationCatalogError(
            f"permutation catalog upsert for {record.permutation_table!r} lost "
            f"{_CATALOG_COMMIT_RETRIES} consecutive commit races: {last_exc}"
        )

    def delete(self, permutation_table: str) -> None:
        if PERMUTATION_CATALOG_TABLE not in self._table_names():
            return
        table = self._db.open_table(PERMUTATION_CATALOG_TABLE)
        table.delete(f"permutation_table = '{_escape(permutation_table)}'")

    def list(self) -> list[PermutationArtifactRecord]:
        if PERMUTATION_CATALOG_TABLE not in self._table_names():
            return []
        table = self._db.open_table(PERMUTATION_CATALOG_TABLE)
        records: list[PermutationArtifactRecord] = []
        query = table.search().select(["record_json"])
        for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
            for row in batch.to_pylist():
                records.append(
                    PermutationArtifactRecord.from_dict(json.loads(row["record_json"]))
                )
        records.sort(key=lambda r: (r.last_used_at, r.permutation_table), reverse=True)
        return records

    def store_ref(self, permutation_table: str) -> str:
        return f"lancedb://{PERMUTATION_CATALOG_TABLE}/{permutation_table}"


# --------------------------------------------------------------------------- #
# Reference gathering                                                          #
# --------------------------------------------------------------------------- #

#: Manifest tables that pin a permutation artifact via its row/epoch plan ids.
#: Evidence packs, retention holds, and model artifacts do not currently record
#: a permutation or plan-id linkage, so they cannot reference an artifact today;
#: extending reference tracking to them is a filed follow-on (see task record).
_REFERENCE_SOURCES = (
    ("training_runs", ("row_plan_id", "epoch_plan_id")),
    ("training_reports", ("row_plan_id", "epoch_plan_id")),
)


def _db_table_names(db: Any) -> set[str]:
    response = db.list_tables()
    tables = getattr(response, "tables", response)
    return {str(name) for name in (tables or [])}


def gather_referenced_plan_ids(db: Any) -> frozenset[str]:
    """Collect the row/epoch plan ids pinned by retained runs/reports.

    Reads only the two projected plan-id columns per manifest table and streams
    batches, so cost scales with the (small) number of manifests, never with the
    observation grain (SKILLS.md bounded-read discipline).
    """
    if db is None:
        return frozenset()
    present = _db_table_names(db)
    protected: set[str] = set()
    for table_name, columns in _REFERENCE_SOURCES:
        if table_name not in present:
            continue
        table = db.open_table(table_name)
        available = [c for c in columns if c in set(table.schema.names)]
        if not available:
            continue
        # Project only the plan-id columns and stream batches (same bounded read
        # as run_manifests._project_rows). Any failure here propagates so cleanup
        # fails loudly rather than proceeding with an under-counted protected set
        # and dropping a referenced artifact (SKILLS.md: converge or fail loudly).
        query = table.search().select(available)
        for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
            for column in available:
                for value in batch.column(column).to_pylist():
                    if value:
                        protected.add(str(value))
    return frozenset(protected)


# --------------------------------------------------------------------------- #
# Catalog coordinator                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PermutationCleanupReport:
    """Outcome of a cleanup sweep (safe-to-serialize, idempotent to recompute)."""

    dry_run: bool
    scanned: int
    kept: tuple[str, ...]
    removed: tuple[str, ...]
    reclaimed_bytes: int
    protected_plan_ids: int
    skipped_uncataloged: tuple[str, ...] = ()
    errors: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "scanned": self.scanned,
            "kept": list(self.kept),
            "removed": list(self.removed),
            "kept_count": len(self.kept),
            "removed_count": len(self.removed),
            "reclaimed_bytes": self.reclaimed_bytes,
            "protected_plan_ids": self.protected_plan_ids,
            "skipped_uncataloged": list(self.skipped_uncataloged),
            "skipped_uncataloged_count": len(self.skipped_uncataloged),
            "errors": [dict(item) for item in self.errors],
        }


class PermutationArtifactCatalog:
    """Record / discover / account / clean up internal epoch-permutation tables.

    Holds the durable :class:`PermutationArtifactStore` plus the LanceDB
    connection needed to enumerate/drop the physical ``__lancedb_robotics_epoch_perm_*``
    tables and to gather references from the manifest catalogs. ``now_fn`` is
    injectable for tests.
    """

    def __init__(
        self,
        store: PermutationArtifactStore,
        db: Any,
        *,
        now_fn=_utcnow,
    ) -> None:
        self.store = store
        self._db = db
        self._now = now_fn

    # -- write path -------------------------------------------------------- #

    def record(
        self,
        *,
        permutation_table: str,
        permutation_ref: str,
        backend_kind: str,
        permutation_source: str,
        row_plan_id: str | None,
        epoch_plan_id: str | None,
        dataset_id: str | None,
        snapshot_name: str | None,
        table_versions: Sequence[Mapping[str, Any]],
        shuffle_seed: int | None,
        epoch: int | None,
        worker_id: int | None,
        num_workers: int | None,
        resume_from: int | None,
        row_count: int,
    ) -> PermutationArtifactRecord:
        """Upsert the catalog entry for a just-created-or-reused permutation table.

        Insert-vs-update is keyed on whether a catalog *row* already exists (not on
        whether the physical table was newly written): a first sighting inserts a
        fresh record; a later sighting bumps the advisory ``use_count`` /
        ``last_used_at`` and preserves ``created_at`` and any operator-set retention
        policy. ``permutation_source`` carries the physical materialized/reused
        signal for reporting.
        """
        now = self._now()
        existing = self.store.get(permutation_table)
        if existing is None:
            record = build_permutation_artifact_record(
                permutation_table=permutation_table,
                permutation_ref=permutation_ref,
                backend_kind=backend_kind,
                permutation_source=permutation_source,
                row_plan_id=row_plan_id,
                epoch_plan_id=epoch_plan_id,
                dataset_id=dataset_id,
                snapshot_name=snapshot_name,
                table_versions=table_versions,
                shuffle_seed=shuffle_seed,
                epoch=epoch,
                worker_id=worker_id,
                num_workers=num_workers,
                resume_from=resume_from,
                row_count=row_count,
                now=now,
            )
        else:
            record = existing.mark_used(now, source=permutation_source)
            # Backfill identity if the prior record was a bare prefix-discovered
            # orphan promoted with fuller metadata now.
            if existing.row_plan_id is None and row_plan_id is not None:
                record = replace(
                    record,
                    row_plan_id=row_plan_id,
                    epoch_plan_id=epoch_plan_id,
                    dataset_id=dataset_id,
                    snapshot_name=snapshot_name,
                    table_versions=tuple(dict(item) for item in table_versions),
                    shuffle_seed=shuffle_seed,
                    epoch=epoch,
                    worker_id=worker_id,
                    num_workers=num_workers,
                    resume_from=resume_from,
                    row_count=row_count,
                )
        self.store.put(record)
        return record

    def set_retention_policy(self, permutation_table: str, policy: str) -> PermutationArtifactRecord:
        """Pin (``keep``) or release (``auto``) an artifact from auto-cleanup."""
        record = self.store.get(permutation_table)
        if record is None:
            raise PermutationCatalogError(
                f"no cataloged permutation artifact named {permutation_table!r}"
            )
        updated = record.with_retention_policy(policy)
        self.store.put(updated)
        return updated

    # -- discovery --------------------------------------------------------- #

    def _physical_permutation_tables(self) -> set[str]:
        if self._db is None:
            return set()
        return {
            name
            for name in _db_table_names(self._db)
            if name.startswith(EPOCH_PERMUTATION_TABLE_PREFIX)
        }

    def get(self, permutation_table: str) -> dict[str, Any] | None:
        record = self.store.get(permutation_table)
        if record is not None:
            return record.to_dict(cataloged=True)
        if permutation_table in self._physical_permutation_tables():
            return _uncataloged_summary(permutation_table)
        return None

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Every internal permutation artifact: cataloged (full) + orphan (prefix).

        ``limit`` bounds the user-facing listing (a first-draft page cap; a
        stable cursor for fully resumable paging at fleet scale is tracked as a
        0131 follow-on). ``accounting``/``cleanup`` intentionally fold over the
        full set (bounded by the number of internal permutation tables, not the
        observation grain), so they do not pass ``limit``.
        """
        cataloged = {r.permutation_table: r for r in self.store.list()}
        physical = self._physical_permutation_tables()
        results: list[dict[str, Any]] = []
        for name, record in cataloged.items():
            # Drop catalog entries whose physical table is gone unless it is a
            # store-only in-memory record (db unavailable) -- keep those visible.
            if self._db is not None and name not in physical:
                continue
            results.append(record.to_dict(cataloged=True))
        for name in physical:
            if name not in cataloged:
                results.append(_uncataloged_summary(name))
        results.sort(key=lambda item: (item.get("last_used_at") or "", item["permutation_table"]))
        if limit is not None:
            return results[:limit]
        return results

    def accounting(self) -> dict[str, Any]:
        """Artifact count / rows / estimated bytes for training/benchmark reports."""
        artifacts = self.list()
        by_backend: dict[str, int] = {}
        total_rows = 0
        total_bytes = 0
        cataloged = 0
        for item in artifacts:
            total_rows += int(item.get("row_count") or 0)
            total_bytes += int(item.get("estimated_bytes") or 0)
            if item.get("cataloged"):
                cataloged += 1
            backend = item.get("backend_kind") or "unknown"
            by_backend[backend] = by_backend.get(backend, 0) + 1
        return {
            "artifact_count": len(artifacts),
            "cataloged_count": cataloged,
            "uncataloged_count": len(artifacts) - cataloged,
            "total_rows": total_rows,
            "estimated_bytes": total_bytes,
            "estimated_bytes_basis": f"{PERMUTATION_ROW_BYTES} bytes/ordered-row-id",
            "by_backend": by_backend,
        }

    # -- cleanup ----------------------------------------------------------- #

    def cleanup(
        self, *, dry_run: bool = False, include_uncataloged: bool = False
    ) -> PermutationCleanupReport:
        """Drop unreferenced internal permutation tables; keep referenced/pinned.

        A **cataloged** artifact is reclaimed only when it can be *proven*
        unreferenced: its ``row_plan_id``/``epoch_plan_id`` is not pinned by any
        retained ``training_runs``/``training_reports`` row and its retention
        policy is not ``keep``.

        An **uncataloged** orphan (a physical ``__lancedb_robotics_epoch_perm_*``
        table with no catalog row -- e.g. left by a pre-catalog writer, or by a
        crash/lost race between the physical-table write and the catalog write)
        carries no plan-id linkage, so it *cannot* be proven unreferenced. By
        default such orphans are **kept** (reported under ``skipped_uncataloged``)
        rather than risk dropping a table a retained run references. Pass
        ``include_uncataloged=True`` to also reclaim orphans -- an explicit
        operator choice; those cannot be reference-checked, so only use it when
        stray artifacts are known to be safe to remove.

        Idempotent: a re-run over the same lake with the same retained records
        removes nothing new.
        """
        protected = gather_referenced_plan_ids(self._db)
        physical = self._physical_permutation_tables()
        cataloged = {r.permutation_table: r for r in self.store.list()}

        # Union of everything we might act on: physical tables + catalog entries.
        candidates = sorted(set(physical) | set(cataloged))
        kept: list[str] = []
        removed: list[str] = []
        skipped_uncataloged: list[str] = []
        reclaimed = 0
        errors: list[dict[str, str]] = []

        for name in candidates:
            record = cataloged.get(name)
            table_exists = name in physical
            # Orphan physical table with no catalog metadata: conservative by
            # default -- keep it, since we cannot prove it is unreferenced.
            if record is None and table_exists and not include_uncataloged:
                skipped_uncataloged.append(name)
                kept.append(name)
                continue
            if self._should_keep(record, protected, table_exists):
                kept.append(name)
                continue
            est_bytes = record.estimated_bytes if record is not None else 0
            if dry_run:
                removed.append(name)
                reclaimed += est_bytes
                continue
            try:
                if table_exists:
                    self._db.drop_table(name)
                if record is not None:
                    self.store.delete(name)
            except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                errors.append({"permutation_table": name, "error": str(exc)})
                kept.append(name)
                continue
            removed.append(name)
            reclaimed += est_bytes

        return PermutationCleanupReport(
            dry_run=dry_run,
            scanned=len(candidates),
            kept=tuple(kept),
            removed=tuple(removed),
            reclaimed_bytes=reclaimed,
            protected_plan_ids=len(protected),
            skipped_uncataloged=tuple(skipped_uncataloged),
            errors=tuple(errors),
        )

    def _should_keep(
        self,
        record: PermutationArtifactRecord | None,
        protected: frozenset[str],
        table_exists: bool,
    ) -> bool:
        # A catalog entry whose physical table already vanished is dropped (its
        # bookkeeping is stale), never "kept".
        if record is None:
            return False  # orphan reached here only under include_uncataloged
        if not table_exists:
            return False
        if record.retention_policy == RETENTION_KEEP:
            return True
        return record.is_referenced_by(protected)


def _uncataloged_summary(permutation_table: str) -> dict[str, Any]:
    """Minimal summary for a prefix-discovered table with no catalog metadata."""
    return {
        "kind": PERMUTATION_CATALOG_KIND,
        "cataloged": False,
        "permutation_table": permutation_table,
        "permutation_ref": f"lancedb://{permutation_table}",
        "backend_kind": "unknown",
        "permutation_source": "prefix-discovered",
        "row_plan_id": None,
        "epoch_plan_id": None,
        "dataset_id": None,
        "snapshot_name": None,
        "table_versions": [],
        "shuffle_seed": None,
        "epoch": None,
        "worker": {"id": None, "num_workers": None, "resume_from": None},
        "row_count": 0,
        "estimated_bytes": 0,
        "created_at": None,
        "last_used_at": None,
        "use_count": 0,
        "retention_policy": RETENTION_AUTO,
    }


# --------------------------------------------------------------------------- #
# Resolution                                                                   #
# --------------------------------------------------------------------------- #


def _lake_db(lake: Any) -> Any | None:
    db = getattr(lake, "_db", None)
    if db is None:
        return None
    if not all(hasattr(db, name) for name in ("create_table", "open_table", "list_tables")):
        return None
    return db


def resolve_permutation_catalog(lake: Any) -> PermutationArtifactCatalog | None:
    """Return the durable catalog for ``lake``'s local LanceDB connection.

    A caller may attach an explicit store/catalog via
    ``lake.permutation_catalog`` (a :class:`PermutationArtifactCatalog`) or
    ``lake.permutation_artifact_store`` (a :class:`PermutationArtifactStore`);
    otherwise a durable :class:`LanceTablePermutationArtifactStore` over the
    lake's connection is used. Returns ``None`` when no LanceDB connection is
    available (e.g. a purely remote/enterprise lake using server-side plans).
    """
    explicit = getattr(lake, "permutation_catalog", None)
    if isinstance(explicit, PermutationArtifactCatalog):
        return explicit
    db = _lake_db(lake)
    if db is None:
        return None
    store = getattr(lake, "permutation_artifact_store", None)
    if not isinstance(store, PermutationArtifactStore):
        store = LanceTablePermutationArtifactStore(db)
    return PermutationArtifactCatalog(store, db)


def open_permutation_catalog(lake: Any) -> PermutationArtifactCatalog | None:
    """Alias of :func:`resolve_permutation_catalog` for read/cleanup callers."""
    return resolve_permutation_catalog(lake)


def record_permutation_artifact(
    lake: Any,
    *,
    permutation_table: str,
    permutation_ref: str,
    backend_kind: str,
    permutation_source: str,
    row_plan: Any,
    epoch_plan_id: str | None,
    shuffle_seed: int | None,
    epoch: int | None,
    worker_id: int | None,
    num_workers: int | None,
    resume_from: int | None,
    row_count: int,
) -> PermutationArtifactRecord | None:
    """Best-effort catalog upsert from an epoch-permutation write path.

    Never raises into the training write path: catalog bookkeeping is additive
    lifecycle metadata, so a store hiccup must not fail a dataset build. Returns
    the stored record, or ``None`` when no catalog is available / the write was
    skipped.
    """
    catalog = resolve_permutation_catalog(lake)
    if catalog is None:
        return None
    try:
        return catalog.record(
            permutation_table=permutation_table,
            permutation_ref=permutation_ref,
            backend_kind=backend_kind,
            permutation_source=permutation_source,
            row_plan_id=getattr(row_plan, "plan_id", None),
            epoch_plan_id=epoch_plan_id,
            dataset_id=getattr(row_plan, "dataset_id", None),
            snapshot_name=getattr(row_plan, "snapshot_name", None),
            table_versions=tuple(dict(item) for item in getattr(row_plan, "table_versions", ())),
            shuffle_seed=shuffle_seed,
            epoch=epoch,
            worker_id=worker_id,
            num_workers=num_workers,
            resume_from=resume_from,
            row_count=row_count,
        )
    except Exception:  # noqa: BLE001 - lifecycle bookkeeping is never fatal
        return None
