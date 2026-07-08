"""Query-driven cache-warm planner (backlog 0348).

An alternative to the whole-table page-cache admin prewarm (0122): instead of asking
the query node to warm every row of a table's columns, **replay the loader's own read
queries ahead of time** so the query node warms *exactly* the rows a training run
reads — no more, no less.

Rationale (grounded in the code):

- The remote LanceDB API has no random-access ``take`` (``to_arrow``/``to_pandas``
  raise ``NotImplementedError`` on cloud); reads go through ``search()`` /
  ``query().where(...)``. Our own remote read path already filters on a **stable
  data-id column** (``observation_id``, ``episode_id``, ``run_id``), never the
  internal ``_rowid``.
- A query served by the query node pulls exactly the data pages it touches, plus the
  scalar-index pages used to resolve the predicate, into the page cache. So issuing the
  loader's read queries is itself a precise, subset-scoped cache warm.
- Enterprise cache is bounded, so warming the whole table (0122) for a training subset
  is expensive and self-evicting. Query-driven warming avoids that over-warm.

This module emits bounded ``<id_col> IN (<chunk>)`` warm queries over the epoch's
stable ids, checks that the id column is scalar-indexed (else the warm degrades to a
full scan — see BUG-15), and optionally executes them. It never references a plan
executor, never emits placement, and never uses ``_rowid``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Default number of ids per ``IN (...)`` predicate. Bounded to avoid the BUG-06
#: multi-MB predicate that overwhelms the planner; each chunk is one warm query.
DEFAULT_QUERY_WARM_CHUNK_SIZE = 512

#: Kind marker for the emitted plan.
QUERY_WARM_PLAN_KIND = "lancedb-robotics/query-warm-plan/v1"

#: Stable data-id column per canonical training table. Deliberately a real data column
#: (queryable + stable across compaction), never the internal ``_rowid``.
WARM_ID_COLUMNS: dict[str, str] = {
    "observations": "observation_id",
    "episodes": "episode_id",
    "aligned_frames": "aligned_frame_id",
    "aligned_ticks": "aligned_tick_id",
    "runs": "run_id",
    "scenarios": "scenario_id",
}


class QueryWarmError(Exception):
    """Raised when a query-warm plan cannot be constructed or executed."""


def warm_id_column(table: str) -> str | None:
    """Return the stable data-id column to filter on for ``table`` (never ``_rowid``)."""
    return WARM_ID_COLUMNS.get(table)


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _in_predicate(column: str, values: Sequence[Any]) -> str:
    return f"{column} IN ({', '.join(_sql_literal(value) for value in values)})"


def _chunk(values: Sequence[Any], size: int) -> list[tuple[Any, ...]]:
    if size < 1:
        raise QueryWarmError(f"chunk_size must be >= 1, got {size}")
    return [tuple(values[i : i + size]) for i in range(0, len(values), size)]


@dataclass(frozen=True)
class TableIndexPrecondition:
    """Whether the id column a warm query filters on is scalar-indexed.

    When ``indexed`` is False, ``<id> IN (...)`` degrades to a full table scan on the
    query node — it still warms the touched pages but at scan cost and without warming a
    useful index (see BUG-15: grain tables often lack scalar indexes).
    """

    table: str
    id_column: str
    indexed: bool
    status: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "id_column": self.id_column,
            "indexed": self.indexed,
            "status": self.status,
            "note": self.note,
        }


#: ``(table, id_column) -> TableIndexPrecondition | None``.
IndexChecker = Callable[[str, str], "TableIndexPrecondition | None"]


@dataclass(frozen=True)
class QueryWarmQuery:
    """One bounded warm query mirroring a loader read: ``where(<id> IN chunk)``."""

    table: str
    version: int | None
    id_column: str
    columns: tuple[str, ...]
    id_values: tuple[Any, ...]
    chunk_index: int
    chunk_count: int

    def where_clause(self) -> str:
        return _in_predicate(self.id_column, self.id_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "version": self.version,
            "id_column": self.id_column,
            "select": list(self.columns),
            "where": self.where_clause(),
            "id_count": len(self.id_values),
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
        }


@dataclass(frozen=True)
class QueryWarmTable:
    table: str
    version: int | None
    id_column: str
    columns: tuple[str, ...]
    total_ids: int
    queries: tuple[QueryWarmQuery, ...]
    precondition: TableIndexPrecondition | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "version": self.version,
            "id_column": self.id_column,
            "columns": list(self.columns),
            "total_ids": self.total_ids,
            "queries": [query.to_dict() for query in self.queries],
            "precondition": self.precondition.to_dict() if self.precondition else None,
        }


@dataclass(frozen=True)
class QueryWarmTableSpec:
    """Resolved input for one table: which stable ids + columns the loader reads."""

    table: str
    version: int | None
    id_column: str
    id_values: Sequence[Any]
    columns: Sequence[str]


@dataclass(frozen=True)
class QueryWarmPlan:
    """A set of bounded warm queries that replay a training run's reads."""

    snapshot_name: str | None
    scope: str
    chunk_size: int
    tables: tuple[QueryWarmTable, ...]
    warnings: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    kind: str = QUERY_WARM_PLAN_KIND

    def all_queries(self) -> list[QueryWarmQuery]:
        return [query for table in self.tables for query in table.queries]

    def where_clauses(self) -> list[str]:
        return [query.where_clause() for query in self.all_queries()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "snapshot_name": self.snapshot_name,
            "scope": self.scope,
            "chunk_size": self.chunk_size,
            "tables": [table.to_dict() for table in self.tables],
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


def build_query_warm_plan(
    *,
    snapshot_name: str | None,
    scope: str,
    specs: Sequence[QueryWarmTableSpec],
    chunk_size: int = DEFAULT_QUERY_WARM_CHUNK_SIZE,
    index_checker: IndexChecker | None = None,
) -> QueryWarmPlan:
    """Build bounded ``where(<id> IN chunk)`` warm queries from resolved table specs.

    Deterministic for fixed inputs: id order is preserved, deduplicated, and chunked in
    order. Never references ``_rowid`` or a plan executor.
    """
    if chunk_size < 1:
        raise QueryWarmError(f"chunk_size must be >= 1, got {chunk_size}")

    tables: list[QueryWarmTable] = []
    warnings: list[str] = []
    total_queries = 0
    total_ids = 0
    unindexed = 0

    for spec in specs:
        if spec.id_column == "_rowid":
            raise QueryWarmError(
                "query-warm must filter on a stable data-id column, not the internal "
                "'_rowid'"
            )
        columns = tuple(dict.fromkeys(str(column) for column in spec.columns))
        ids = tuple(dict.fromkeys(value for value in spec.id_values if value is not None))
        chunks = _chunk(ids, chunk_size)
        queries = tuple(
            QueryWarmQuery(
                table=spec.table,
                version=spec.version,
                id_column=spec.id_column,
                columns=columns,
                id_values=chunk,
                chunk_index=index,
                chunk_count=len(chunks),
            )
            for index, chunk in enumerate(chunks)
        )

        precondition: TableIndexPrecondition | None = None
        if index_checker is not None:
            try:
                precondition = index_checker(spec.table, spec.id_column)
            except Exception as exc:  # advisory only; never fail the plan on it.
                warnings.append(
                    f"index precondition check for {spec.table}.{spec.id_column} "
                    f"failed: {exc}"
                )
        if precondition is not None and not precondition.indexed and ids:
            unindexed += 1
            warnings.append(
                f"{spec.table}.{spec.id_column} is not scalar-indexed; warm queries "
                "will full-scan (build a scalar index on the id column to warm the "
                "index and avoid a scan)"
            )

        tables.append(
            QueryWarmTable(
                table=spec.table,
                version=spec.version,
                id_column=spec.id_column,
                columns=columns,
                total_ids=len(ids),
                queries=queries,
                precondition=precondition,
            )
        )
        total_queries += len(queries)
        total_ids += len(ids)

    metrics = {
        "tables": len(tables),
        "queries": total_queries,
        "total_ids": total_ids,
        "chunk_size": chunk_size,
        "unindexed_tables": unindexed,
    }
    return QueryWarmPlan(
        snapshot_name=snapshot_name,
        scope=scope,
        chunk_size=chunk_size,
        tables=tuple(tables),
        warnings=tuple(warnings),
        metrics=metrics,
    )


def warm_query_cache(
    lake: Any,
    plan: QueryWarmPlan,
    *,
    row_limit_per_query: int | None = None,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Execute the warm queries to pull their pages into the query-node cache.

    Bounded-memory: each query is drained via ``to_batches`` (never a full collect) and
    only the touched-row count is retained. Best-effort per query — a failed warm query
    is recorded and skipped, never fatal (warming must self-right, not block training).
    Runs through the sanctioned ``search().where(...)`` path; never a plan executor.
    """
    queries_run = 0
    queries_failed = 0
    rows_touched = 0
    errors: list[str] = []

    for table in plan.tables:
        if not table.queries:
            continue
        handle = lake.table(table.table)
        pinned = table.version is not None
        if pinned:
            try:
                handle.checkout(int(table.version))
            except Exception as exc:
                errors.append(f"{table.table}@v{table.version}: checkout failed: {exc}")
                pinned = False
        try:
            for query in table.queries:
                try:
                    builder = handle.search().select(list(query.columns)).where(
                        query.where_clause()
                    )
                    seen = 0
                    for batch in builder.to_batches(batch_size=batch_size):
                        seen += batch.num_rows
                        if row_limit_per_query is not None and seen >= row_limit_per_query:
                            break
                    rows_touched += seen
                    queries_run += 1
                except Exception as exc:
                    queries_failed += 1
                    errors.append(
                        f"{table.table} chunk {query.chunk_index}: {exc}"
                    )
        finally:
            if pinned:
                try:
                    handle.checkout_latest()
                except Exception:
                    pass

    return {
        "queries_run": queries_run,
        "queries_failed": queries_failed,
        "rows_touched": rows_touched,
        "errors": errors,
    }
