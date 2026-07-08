"""Server-side / query-node-backed row-plan artifacts for Enterprise training.

Backlog 0117 (epic 0069, PRD G6/G10/G12). The local planner (0070) assembles the
full row order in the SDK process and every worker slices it ``[worker_id::P]`` --
each trainer materializes the whole order before iteration starts. Fleet-scale
snapshots need a *version-pinned* plan artifact that a query node builds **once**
and workers page through, claiming deterministic non-overlapping pages without
any worker holding the full row order.

This module is data-plane only. It never opens the snapshot tables and holds no
lake reference: it operates over an already-selected, already-ordered set of row
ids / frame ids (plus optional aligned-tick ids) and a pluggable page store.
Capability gating and the typed ``ServerSidePlanUnavailableError`` diagnostic live
in :mod:`lancedb_robotics.training`, which owns the Enterprise capability matrix.

Design decisions (recorded in the task record):

* **Worker partition is page-strided, not sample-round-robin.** Worker ``W`` of
  ``P`` claims whole storage pages ``p`` where ``p % P == W`` -- non-overlapping,
  covering every page exactly once. ``resume_from`` is a *global sample offset*
  applied by trimming within each claimed page. The union of all workers' pages,
  in ascending page order, is exactly the resumed global order ``G[resume_from:]``.
* **Equivalence is defined at the epoch level.** With ``num_workers == 1`` the
  artifact yields the byte-identical ordered ids the local planner produced for
  the same snapshot/seed/epoch/filters; multi-worker output covers that epoch
  exactly once with no duplicates. Per-worker page sets differ from the in-memory
  ``[W::P]`` round-robin by construction, and that is intentional.
* **Handles are small and secret-free.** The handle is JSON-serializable metadata
  (identity + pagination shape + backend display URI); the ordered ids live in the
  store, fetched a page at a time. ``to_dict`` is asserted secret-free.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

SERVER_SIDE_PLAN_KIND = "lancedb-robotics/server-side-row-plan/v1"
DEFAULT_PLAN_PAGE_SIZE = 1024
PLAN_PAGE_TABLE_PREFIX = "__lancedb_robotics_row_plan_pages_"

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


class ServerSidePlanError(Exception):
    """Raised for server-side plan artifact misuse (bad token, page out of range)."""


# --------------------------------------------------------------------------- #
# Serialization / digest helpers (kept local so this module has no dependency  #
# on training.py -- the coupling runs one way: training.py imports this file). #
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


def _sequence_digest(values: Sequence[Any]) -> str:
    """Stream a (possibly large) ordered sequence into one sha1 without buffering JSON."""
    hasher = hashlib.sha1()
    for value in values:
        hasher.update(repr(value).encode())
        hasher.update(b"\x1f")
    return hasher.hexdigest()[:16]


def _secret_key(key: str) -> bool:
    lowered = str(key).lower().replace("-", "_")
    if "auth_ref" in lowered:
        return False
    return any(token in lowered for token in _SECRET_KEY_TOKENS)


def _secret_value(value: str) -> bool:
    stripped = value.strip().lower()
    return stripped.startswith(("bearer ", "basic "))


def _assert_secret_free(payload: Any, *, path: str = "handle") -> None:
    if isinstance(payload, Mapping):
        for key, item in payload.items():
            if _secret_key(str(key)):
                raise ServerSidePlanError(
                    f"server-side plan handle would leak a secret-like field at {path}.{key}"
                )
            _assert_secret_free(item, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            _assert_secret_free(item, path=f"{path}[{index}]")
    elif isinstance(payload, str) and _secret_value(payload):
        raise ServerSidePlanError(
            f"server-side plan handle would leak a bearer/basic credential at {path}"
        )


# --------------------------------------------------------------------------- #
# Page stores                                                                  #
# --------------------------------------------------------------------------- #


class ServerSidePlanStore:
    """Where a query node persists the ordered row-plan pages for a handle.

    A store maps ``plan_handle_id`` to an ordered list of pages; each page is a
    mapping with ``page_index``, ``start_offset``, ``row_ids``, ``frame_ids``, and
    optional ``aligned_tick_ids``. Writes are idempotent by content ``digest`` so a
    retry/rebuild for the same handle is a no-op.
    """

    kind = "abstract"

    def write(self, plan_handle_id: str, digest: str, pages: Sequence[Mapping[str, Any]]) -> None:
        raise NotImplementedError

    def exists(self, plan_handle_id: str, digest: str) -> bool:
        raise NotImplementedError

    def page_count(self, plan_handle_id: str) -> int:
        raise NotImplementedError

    def read_page(self, plan_handle_id: str, page_index: int) -> dict[str, Any]:
        raise NotImplementedError

    def store_ref(self, plan_handle_id: str) -> str:
        raise NotImplementedError


class InMemoryServerSidePlanStore(ServerSidePlanStore):
    """In-process store used by the query-node simulation and by serialize/reload.

    ``dump``/``load`` round-trip the pages as plain JSON so a worker can rebuild the
    reader from serialized bytes with no local Lance table object.
    """

    kind = "in-memory"

    def __init__(self) -> None:
        self._plans: dict[str, dict[str, Any]] = {}

    def write(self, plan_handle_id: str, digest: str, pages: Sequence[Mapping[str, Any]]) -> None:
        existing = self._plans.get(plan_handle_id)
        if existing is not None and existing.get("digest") == digest:
            return
        self._plans[plan_handle_id] = {
            "digest": digest,
            "pages": [dict(page) for page in pages],
        }

    def exists(self, plan_handle_id: str, digest: str) -> bool:
        record = self._plans.get(plan_handle_id)
        return record is not None and (digest == "" or record.get("digest") == digest)

    def page_count(self, plan_handle_id: str) -> int:
        record = self._plans.get(plan_handle_id)
        if record is None:
            raise ServerSidePlanError(f"no server-side plan pages for handle {plan_handle_id!r}")
        return len(record["pages"])

    def read_page(self, plan_handle_id: str, page_index: int) -> dict[str, Any]:
        record = self._plans.get(plan_handle_id)
        if record is None:
            raise ServerSidePlanError(f"no server-side plan pages for handle {plan_handle_id!r}")
        pages = record["pages"]
        if page_index < 0 or page_index >= len(pages):
            raise ServerSidePlanError(
                f"page {page_index} is out of range for handle {plan_handle_id!r} "
                f"({len(pages)} pages)"
            )
        return dict(pages[page_index])

    def store_ref(self, plan_handle_id: str) -> str:
        return f"memory://{plan_handle_id}"

    def dump(self) -> dict[str, Any]:
        return {"kind": self.kind, "plans": _jsonable(self._plans)}

    @classmethod
    def load(cls, payload: Mapping[str, Any]) -> InMemoryServerSidePlanStore:
        store = cls()
        for handle_id, record in dict(payload.get("plans", {})).items():
            store._plans[str(handle_id)] = {
                "digest": str(record.get("digest", "")),
                "pages": [dict(page) for page in record.get("pages", [])],
            }
        return store


class LanceTablePlanPageStore(ServerSidePlanStore):
    """Durable store: one internal LanceDB table per handle, addressed by page index.

    Survives process restarts so a worker in a fresh process can page a handle it
    only holds as serialized metadata -- the durable analog of the in-memory store,
    used by the CLI/API cross-invocation path and the enterprise dataset wiring.
    """

    kind = "lancedb-table"

    def __init__(self, db: Any) -> None:
        if db is None:
            raise ServerSidePlanError("LanceTablePlanPageStore requires a LanceDB connection")
        self._db = db

    def _table_name(self, plan_handle_id: str) -> str:
        return PLAN_PAGE_TABLE_PREFIX + plan_handle_id

    def _table_names(self) -> set[str]:
        response = self._db.list_tables()
        tables = getattr(response, "tables", response)
        return {str(name) for name in (tables or [])}

    def write(self, plan_handle_id: str, digest: str, pages: Sequence[Mapping[str, Any]]) -> None:
        name = self._table_name(plan_handle_id)
        names = self._table_names()
        if name in names and self._stored_digest(name) == digest:
            return
        rows = [
            {
                "page_index": int(page["page_index"]),
                "start_offset": int(page["start_offset"]),
                "row_ids": [int(value) for value in page["row_ids"]],
                "frame_ids": [str(value) for value in page["frame_ids"]],
                "aligned_tick_ids": (
                    [str(value) for value in page["aligned_tick_ids"]]
                    if page.get("aligned_tick_ids") is not None
                    else []
                ),
                "has_aligned_ticks": page.get("aligned_tick_ids") is not None,
                "digest": digest,
            }
            for page in pages
        ]
        schema = pa.schema(
            [
                pa.field("page_index", pa.int64()),
                pa.field("start_offset", pa.int64()),
                pa.field("row_ids", pa.list_(pa.int64())),
                pa.field("frame_ids", pa.list_(pa.string())),
                pa.field("aligned_tick_ids", pa.list_(pa.string())),
                pa.field("has_aligned_ticks", pa.bool_()),
                pa.field("digest", pa.string()),
            ]
        )
        data = pa.Table.from_pylist(rows, schema=schema)
        mode = "overwrite" if name in names else "create"
        self._db.create_table(name, data=data, mode=mode)

    def _stored_digest(self, table_name: str) -> str | None:
        table = self._db.open_table(table_name)
        rows = table.search().select(["digest"]).limit(1).to_arrow().to_pylist()
        if not rows:
            return None
        return str(rows[0]["digest"])

    def exists(self, plan_handle_id: str, digest: str) -> bool:
        name = self._table_name(plan_handle_id)
        if name not in self._table_names():
            return False
        if digest == "":
            return True
        return self._stored_digest(name) == digest

    def page_count(self, plan_handle_id: str) -> int:
        name = self._table_name(plan_handle_id)
        if name not in self._table_names():
            raise ServerSidePlanError(f"no server-side plan pages for handle {plan_handle_id!r}")
        return int(self._db.open_table(name).count_rows())

    def read_page(self, plan_handle_id: str, page_index: int) -> dict[str, Any]:
        name = self._table_name(plan_handle_id)
        if name not in self._table_names():
            raise ServerSidePlanError(f"no server-side plan pages for handle {plan_handle_id!r}")
        table = self._db.open_table(name)
        rows = (
            table.search()
            .where(f"page_index = {int(page_index)}")
            .limit(1)
            .to_arrow()
            .to_pylist()
        )
        if not rows:
            raise ServerSidePlanError(
                f"page {page_index} is out of range for handle {plan_handle_id!r}"
            )
        row = rows[0]
        return {
            "page_index": int(row["page_index"]),
            "start_offset": int(row["start_offset"]),
            "row_ids": [int(value) for value in row["row_ids"]],
            "frame_ids": [str(value) for value in row["frame_ids"]],
            "aligned_tick_ids": (
                [str(value) for value in row["aligned_tick_ids"]]
                if row.get("has_aligned_ticks")
                else None
            ),
        }

    def store_ref(self, plan_handle_id: str) -> str:
        return f"lancedb://{self._table_name(plan_handle_id)}"


# --------------------------------------------------------------------------- #
# Plan pages and the plan handle                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlanPage:
    """One bounded, worker-scoped page of the row-plan order."""

    plan_handle_id: str
    page_index: int
    page_token: str
    next_page_token: str | None
    worker_id: int
    num_workers: int
    resume_from: int
    start_offset: int
    row_ids: tuple[int, ...]
    frame_ids: tuple[str, ...]
    aligned_tick_ids: tuple[str, ...] | None
    size: int

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "plan_handle_id": self.plan_handle_id,
            "page_index": self.page_index,
            "page_token": self.page_token,
            "next_page_token": self.next_page_token,
            "worker": {"id": self.worker_id, "num_workers": self.num_workers},
            "resume_from": self.resume_from,
            "start_offset": self.start_offset,
            "row_ids": list(self.row_ids),
            "frame_ids": list(self.frame_ids),
            "size": self.size,
        }
        if self.aligned_tick_ids is not None:
            result["aligned_tick_ids"] = list(self.aligned_tick_ids)
        return result


@dataclass(frozen=True)
class ServerSidePlanArtifact:
    """A version-pinned, paginated, serializable row-plan handle.

    Built once by the query node from an ordered row/frame id set; workers claim
    deterministic non-overlapping pages through :meth:`iter_pages` without holding
    the full order. The handle carries no snapshot table reference and no secrets.
    """

    plan_handle_id: str
    row_plan_id: str
    snapshot_id: str
    snapshot_name: str
    table_versions: tuple[dict[str, Any], ...]
    columns: tuple[str, ...]
    pushed_filters: dict[str, str]
    logical_predicates: tuple[str, ...]
    display_uri: str
    connection_kind: str
    ordering_policy: str
    shuffle: bool
    shuffle_seed: int | None
    epoch: int
    total_rows: int
    page_size: int
    num_pages: int
    has_aligned_ticks: bool
    content_digest: str
    store_kind: str
    store_ref: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    kind: str = SERVER_SIDE_PLAN_KIND
    _store: ServerSidePlanStore | None = field(default=None, compare=False, repr=False)

    # -- serialization ----------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "plan_handle_id": self.plan_handle_id,
            "row_plan_id": self.row_plan_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_name": self.snapshot_name,
            "table_versions": [dict(item) for item in self.table_versions],
            "columns": list(self.columns),
            "pushed_filters": dict(self.pushed_filters),
            "logical_predicates": list(self.logical_predicates),
            "display_uri": self.display_uri,
            "connection_kind": self.connection_kind,
            "ordering_policy": self.ordering_policy,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "epoch": self.epoch,
            "total_rows": self.total_rows,
            "page_size": self.page_size,
            "num_pages": self.num_pages,
            "has_aligned_ticks": self.has_aligned_ticks,
            "content_digest": self.content_digest,
            "store_kind": self.store_kind,
            "store_ref": self.store_ref,
            "capabilities": _jsonable(self.capabilities),
        }
        _assert_secret_free(payload)
        return payload

    def summary(self) -> dict[str, Any]:
        """A lightweight, non-persisting descriptor for manifests / loader reports."""
        return {
            "kind": self.kind,
            "plan_handle_id": self.plan_handle_id,
            "row_plan_id": self.row_plan_id,
            "snapshot_id": self.snapshot_id,
            "ordering_policy": self.ordering_policy,
            "total_rows": self.total_rows,
            "page_size": self.page_size,
            "num_pages": self.num_pages,
            "store_kind": self.store_kind,
            "store_ref": self.store_ref,
            "display_uri": self.display_uri,
        }

    def bind(self, store: ServerSidePlanStore) -> ServerSidePlanArtifact:
        return self._replace_store(store)

    def _replace_store(self, store: ServerSidePlanStore | None) -> ServerSidePlanArtifact:
        from dataclasses import replace

        return replace(self, _store=store)

    # -- page tokens ------------------------------------------------------- #

    def page_token(self, page_index: int) -> str:
        raw = json.dumps(
            {"h": self.plan_handle_id, "p": int(page_index)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return base64.urlsafe_b64encode(raw).decode()

    def parse_page_token(self, token: str) -> int:
        try:
            decoded = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
        except Exception as exc:  # noqa: BLE001 - surface as a typed plan error
            raise ServerSidePlanError(f"invalid plan page token: {exc}") from exc
        if decoded.get("h") != self.plan_handle_id:
            raise ServerSidePlanError(
                "plan page token does not belong to this plan handle "
                f"(token handle {decoded.get('h')!r}, artifact {self.plan_handle_id!r})"
            )
        return int(decoded["p"])

    # -- worker partitioning ---------------------------------------------- #

    def worker_page_indices(self, worker_id: int, num_workers: int) -> list[int]:
        _validate_partition(worker_id, num_workers)
        return [index for index in range(self.num_pages) if index % num_workers == worker_id]

    def _page_end_offset(self, page_index: int) -> int:
        return min((page_index + 1) * self.page_size, self.total_rows)

    def _next_worker_page(
        self, page_index: int, worker_id: int, num_workers: int, resume_from: int
    ) -> int | None:
        candidate = page_index + num_workers
        while candidate < self.num_pages:
            if self._page_end_offset(candidate) > resume_from:
                return candidate
            candidate += num_workers
        return None

    def _materialize_page(
        self,
        page_index: int,
        *,
        worker_id: int,
        num_workers: int,
        resume_from: int,
    ) -> PlanPage | None:
        if self._store is None:
            raise ServerSidePlanError(
                "server-side plan artifact is detached from its page store; reopen it "
                "with open_server_side_row_plan(handle, store=...)"
            )
        raw = self._store.read_page(self.plan_handle_id, page_index)
        start = int(raw["start_offset"])
        row_ids = [int(value) for value in raw["row_ids"]]
        frame_ids = [str(value) for value in raw["frame_ids"]]
        aligned = raw.get("aligned_tick_ids")
        aligned = [str(value) for value in aligned] if aligned is not None else None
        # resume_from is a global sample offset: drop the prefix of this page that
        # falls before it. Pages entirely before resume_from contribute nothing.
        if resume_from > start:
            cut = min(resume_from - start, len(row_ids))
            row_ids = row_ids[cut:]
            frame_ids = frame_ids[cut:]
            if aligned is not None:
                aligned = aligned[cut:]
            page_start = start + cut
        else:
            page_start = start
        if not row_ids:
            return None
        next_index = self._next_worker_page(page_index, worker_id, num_workers, resume_from)
        return PlanPage(
            plan_handle_id=self.plan_handle_id,
            page_index=page_index,
            page_token=self.page_token(page_index),
            next_page_token=self.page_token(next_index) if next_index is not None else None,
            worker_id=worker_id,
            num_workers=num_workers,
            resume_from=resume_from,
            start_offset=page_start,
            row_ids=tuple(row_ids),
            frame_ids=tuple(frame_ids),
            aligned_tick_ids=tuple(aligned) if aligned is not None else None,
            size=len(row_ids),
        )

    def iter_pages(
        self,
        *,
        worker_id: int = 0,
        num_workers: int = 1,
        resume_from: int = 0,
    ) -> Iterator[PlanPage]:
        """Yield this worker's bounded pages of the resumed global order, in page order."""
        if resume_from < 0:
            raise ServerSidePlanError("resume_from must be non-negative")
        for page_index in self.worker_page_indices(worker_id, num_workers):
            page = self._materialize_page(
                page_index,
                worker_id=worker_id,
                num_workers=num_workers,
                resume_from=resume_from,
            )
            if page is not None:
                yield page

    def page_from_token(self, token: str, *, resume_from: int = 0) -> PlanPage:
        """Fetch a single page by its token (worker-agnostic, num_workers=1)."""
        page_index = self.parse_page_token(token)
        page = self._materialize_page(
            page_index, worker_id=0, num_workers=1, resume_from=resume_from
        )
        if page is None:
            raise ServerSidePlanError(
                f"page {page_index} is empty after applying resume_from={resume_from}"
            )
        return page

    # -- convenience readers ---------------------------------------------- #

    def worker_row_ids(
        self, *, worker_id: int = 0, num_workers: int = 1, resume_from: int = 0
    ) -> list[int]:
        ids: list[int] = []
        for page in self.iter_pages(
            worker_id=worker_id, num_workers=num_workers, resume_from=resume_from
        ):
            ids.extend(page.row_ids)
        return ids

    def worker_frame_ids(
        self, *, worker_id: int = 0, num_workers: int = 1, resume_from: int = 0
    ) -> list[str]:
        ids: list[str] = []
        for page in self.iter_pages(
            worker_id=worker_id, num_workers=num_workers, resume_from=resume_from
        ):
            ids.extend(page.frame_ids)
        return ids

    def global_frame_ids(self, *, resume_from: int = 0) -> list[str]:
        """The full resumed global order as a single worker would see it."""
        return self.worker_frame_ids(worker_id=0, num_workers=1, resume_from=resume_from)


def _validate_partition(worker_id: int, num_workers: int) -> None:
    if num_workers < 1:
        raise ServerSidePlanError("num_workers must be at least 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ServerSidePlanError("worker_id must be between 0 and num_workers - 1")


# --------------------------------------------------------------------------- #
# Build / open                                                                 #
# --------------------------------------------------------------------------- #


def build_server_side_row_plan(
    *,
    row_plan_id: str,
    snapshot_id: str,
    snapshot_name: str,
    table_versions: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    display_uri: str,
    connection_kind: str,
    ordered_row_ids: Sequence[int],
    ordered_frame_ids: Sequence[str],
    ordering_policy: str,
    shuffle: bool,
    shuffle_seed: int | None,
    epoch: int,
    ordered_aligned_tick_ids: Sequence[str] | None = None,
    pushed_filters: Mapping[str, str] | None = None,
    logical_predicates: Sequence[str] | None = None,
    page_size: int = DEFAULT_PLAN_PAGE_SIZE,
    store: ServerSidePlanStore | None = None,
    capabilities: Mapping[str, Any] | None = None,
) -> ServerSidePlanArtifact:
    """Persist ``ordered_row_ids``/``ordered_frame_ids`` as a paginated plan artifact.

    Idempotent by content digest: rebuilding the same order for the same handle is a
    store no-op. ``store`` defaults to an in-memory store (the query-node simulation).
    """

    if int(page_size) < 1:
        raise ServerSidePlanError("page_size must be at least 1")
    page_size = int(page_size)
    row_ids = [int(value) for value in ordered_row_ids]
    frame_ids = [str(value) for value in ordered_frame_ids]
    if len(row_ids) != len(frame_ids):
        raise ServerSidePlanError(
            f"ordered_row_ids ({len(row_ids)}) and ordered_frame_ids ({len(frame_ids)}) "
            "must be the same length"
        )
    aligned_ids: list[str] | None = None
    if ordered_aligned_tick_ids is not None:
        aligned_ids = [str(value) for value in ordered_aligned_tick_ids]
        if len(aligned_ids) != len(row_ids):
            raise ServerSidePlanError(
                "ordered_aligned_tick_ids must match the ordered row id length"
            )
    total_rows = len(row_ids)
    num_pages = (total_rows + page_size - 1) // page_size

    content_digest = _stable_digest(
        {
            "rows": _sequence_digest(row_ids),
            "frames": _sequence_digest(frame_ids),
            "aligned": _sequence_digest(aligned_ids) if aligned_ids is not None else None,
            "page_size": page_size,
        }
    )
    table_versions_tuple = tuple(
        {
            "table": str(item.get("table")),
            "version": int(item.get("version")) if item.get("version") is not None else None,
            "tag": item.get("tag") or "",
        }
        for item in table_versions
    )
    plan_handle_id = "srvplan-" + _stable_digest(
        {
            "row_plan_id": row_plan_id,
            "snapshot_id": snapshot_id,
            "table_versions": table_versions_tuple,
            "ordering_policy": ordering_policy,
            "shuffle": shuffle,
            "shuffle_seed": shuffle_seed if shuffle else None,
            "epoch": epoch,
            "page_size": page_size,
            "total_rows": total_rows,
            "content_digest": content_digest,
        }
    )

    if store is None:
        store = InMemoryServerSidePlanStore()

    pages: list[dict[str, Any]] = []
    for page_index in range(num_pages):
        start = page_index * page_size
        stop = min(start + page_size, total_rows)
        pages.append(
            {
                "page_index": page_index,
                "start_offset": start,
                "row_ids": row_ids[start:stop],
                "frame_ids": frame_ids[start:stop],
                "aligned_tick_ids": aligned_ids[start:stop] if aligned_ids is not None else None,
            }
        )
    store.write(plan_handle_id, content_digest, pages)

    artifact = ServerSidePlanArtifact(
        plan_handle_id=plan_handle_id,
        row_plan_id=row_plan_id,
        snapshot_id=snapshot_id,
        snapshot_name=snapshot_name,
        table_versions=table_versions_tuple,
        columns=tuple(columns),
        pushed_filters=dict(pushed_filters or {}),
        logical_predicates=tuple(logical_predicates or ()),
        display_uri=display_uri,
        connection_kind=connection_kind,
        ordering_policy=ordering_policy,
        shuffle=bool(shuffle),
        shuffle_seed=shuffle_seed if shuffle else None,
        epoch=int(epoch),
        total_rows=total_rows,
        page_size=page_size,
        num_pages=num_pages,
        has_aligned_ticks=aligned_ids is not None,
        content_digest=content_digest,
        store_kind=store.kind,
        store_ref=store.store_ref(plan_handle_id),
        capabilities=dict(capabilities or {}),
        _store=store,
    )
    # Fail fast if any identity field somehow carried a secret.
    artifact.to_dict()
    return artifact


def open_server_side_row_plan(
    handle: Mapping[str, Any],
    *,
    store: ServerSidePlanStore,
) -> ServerSidePlanArtifact:
    """Rebuild an artifact from a serialized handle plus a store holding its pages.

    Used by a worker in a fresh process: the handle is small serialized metadata and
    the store serves pages -- no snapshot table object is required.
    """

    plan_handle_id = str(handle["plan_handle_id"])
    if not store.exists(plan_handle_id, ""):
        raise ServerSidePlanError(
            f"store does not hold pages for plan handle {plan_handle_id!r}"
        )
    stored_pages = store.page_count(plan_handle_id)
    declared_pages = int(handle.get("num_pages", stored_pages))
    if stored_pages != declared_pages:
        raise ServerSidePlanError(
            f"handle declares {declared_pages} pages but store holds {stored_pages}"
        )
    return ServerSidePlanArtifact(
        plan_handle_id=plan_handle_id,
        row_plan_id=str(handle.get("row_plan_id", "")),
        snapshot_id=str(handle.get("snapshot_id", "")),
        snapshot_name=str(handle.get("snapshot_name", "")),
        table_versions=tuple(dict(item) for item in handle.get("table_versions", ())),
        columns=tuple(handle.get("columns", ())),
        pushed_filters=dict(handle.get("pushed_filters", {})),
        logical_predicates=tuple(handle.get("logical_predicates", ())),
        display_uri=str(handle.get("display_uri", "")),
        connection_kind=str(handle.get("connection_kind", "")),
        ordering_policy=str(handle.get("ordering_policy", "")),
        shuffle=bool(handle.get("shuffle", False)),
        shuffle_seed=handle.get("shuffle_seed"),
        epoch=int(handle.get("epoch", 0)),
        total_rows=int(handle.get("total_rows", 0)),
        page_size=int(handle.get("page_size", DEFAULT_PLAN_PAGE_SIZE)),
        num_pages=declared_pages,
        has_aligned_ticks=bool(handle.get("has_aligned_ticks", False)),
        content_digest=str(handle.get("content_digest", "")),
        store_kind=str(handle.get("store_kind", store.kind)),
        store_ref=str(handle.get("store_ref", store.store_ref(plan_handle_id))),
        capabilities=dict(handle.get("capabilities", {})),
        _store=store,
    )
