"""Enterprise page-cache prewarm request planner (backlog 0122).

This module turns a training column projection + snapshot into a *valid* LanceDB
Enterprise **query-node** page-cache prewarm request, plus advisory client-side
cost estimates.

Architecture rule enforced here (see the 0122 task record and decision): a client
application such as ``lancedb-robotics-lakehouse`` talks only to **object storage**
or the **query node** — never to a plan executor. The wire request mirrors the real
Enterprise contract, sophon ``PageCacheBeginPrewarmRequest`` ::

    { id?, db, table, columns[], table_version?, concurrency? }

submitted to the query node's ``POST /admin/page_cache/prewarm`` (or the durable
``/admin/page_cache/begin_prewarm_job``). The query node internally fans the work out
to plan executors using server-side consistent hashing (``sophon-consistent-hashing``);
fragment and plan-executor placement are therefore **not** the client's concern.
Consequently this planner emits:

- NO plan-executor placement hints, NO ``pe_fanout``, NO PE addresses.
- NO row-id / fragment / row-group ranges (the API is whole-table, per-column; it has
  no such parameters, and fragment/PE fanout is reported by the query node in the
  prewarm *response*, never supplied by the client).

Row / byte estimates are computed **only** from query-node-safe metadata (Arrow schema
and whole-table row count) to help a researcher gauge prewarm cost and the whole-table
over-warm ratio versus the epoch subset actually read during training. Advisory fields
are kept in a separate ``estimate`` block and are never part of the wire request.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import pyarrow as pa

#: Kind marker for a single query-node page-cache prewarm wire request envelope.
PREWARM_REQUEST_KIND = "lancedb-robotics/page-cache-prewarm-request/v1"

#: Kind marker for the aggregate plan a training dataset emits for inspection.
PREWARM_PLAN_KIND = "lancedb-robotics/page-cache-prewarm-plan/v1"

#: Mirror of sophon ``CACHE_PREWARM_SPEC_SCHEMA_VERSION`` so a query node that
#: validates ``schema_version`` accepts requests emitted here.
PAGE_CACHE_PREWARM_SPEC_SCHEMA_VERSION = 1

#: Default per-row byte estimate for variable-width metadata columns (string / json
#: / list) when only the Arrow schema (not the data) is available.
DEFAULT_PREWARM_VARIABLE_WIDTH_BYTES = 64

#: Column-metadata source markers used by the advisory estimate. Everything here
#: comes from the query node (schema, row count) -- never from fragment or
#: plan-executor internals.
ESTIMATE_SOURCE_QUERY_NODE = "query-node"
ESTIMATE_SOURCE_SCHEMA_ONLY = "schema-only"
ESTIMATE_SOURCE_UNAVAILABLE = "unavailable"


class PrewarmPlannerError(Exception):
    """Raised when a page-cache prewarm plan cannot be constructed."""


@dataclass(frozen=True)
class TableMetadata:
    """Cheap, data-free table metadata used only for advisory estimates.

    Both fields are query-node-safe metadata calls: ``schema`` is the Arrow schema
    and ``total_rows`` is the whole-table row count that prewarm warms. The planner
    deliberately does not read fragments, row groups, or any plan-executor internals
    -- the query node owns fragment/PE fanout and reports it in the prewarm response.
    """

    schema: pa.Schema | None = None
    total_rows: int | None = None


#: A metadata accessor: ``(table_name, table_version) -> TableMetadata | None``.
#: The planner never calls anything else on the lake, so it can never read payload
#: data or reach a plan executor.
MetadataFn = Callable[[str, "int | None"], "TableMetadata | None"]


@dataclass(frozen=True)
class PrewarmPlannerOptions:
    """Knobs for the page-cache prewarm planner (all advisory-side only)."""

    concurrency: int | None = None
    variable_width_bytes: int = DEFAULT_PREWARM_VARIABLE_WIDTH_BYTES
    heavy_bytes_per_row: int | None = None
    estimate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "variable_width_bytes": self.variable_width_bytes,
            "heavy_bytes_per_row": self.heavy_bytes_per_row,
            "estimate": self.estimate,
        }


@dataclass(frozen=True)
class PrewarmColumnEstimate:
    """Advisory per-column byte estimate for the *whole* table (what prewarm warms)."""

    column: str
    kind: str  # "metadata" | "heavy"
    arrow_type: str | None
    bytes_per_row: float | None
    estimated_bytes: int
    basis: str  # schema-width | configured-variable | configured-heavy | unavailable

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "kind": self.kind,
            "arrow_type": self.arrow_type,
            "bytes_per_row": self.bytes_per_row,
            "estimated_bytes": self.estimated_bytes,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class PrewarmTableEstimate:
    """Advisory cost estimate for one table. Never sent to the query node.

    Reasoning is at the table/column/row level only: ``over_warm_ratio`` compares the
    whole-table rows prewarm warms against the epoch subset training reads. Fragment
    and plan-executor fanout are the query node's job and appear only in the prewarm
    response, so they are intentionally absent here.
    """

    table: str
    version: int | None
    selected_rows: int
    total_rows: int | None
    over_warm_ratio: float | None
    metadata_bytes: int
    heavy_bytes: int
    estimated_bytes: int
    columns: tuple[PrewarmColumnEstimate, ...]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "version": self.version,
            "selected_rows": self.selected_rows,
            "total_rows": self.total_rows,
            "over_warm_ratio": self.over_warm_ratio,
            "metadata_bytes": self.metadata_bytes,
            "heavy_bytes": self.heavy_bytes,
            "estimated_bytes": self.estimated_bytes,
            "columns": [column.to_dict() for column in self.columns],
            "source": self.source,
        }


@dataclass(frozen=True)
class PrewarmTableRequest:
    """One query-node page-cache prewarm request for a single canonical table.

    :meth:`wire_request` returns exactly the JSON body the query node accepts
    (sophon ``PageCacheBeginPrewarmRequest``). Advisory estimates live in
    :attr:`estimate` and are deliberately excluded from the wire body.
    """

    db: str
    table: str
    columns: tuple[str, ...]
    table_version: int | None
    concurrency: int | None
    logical_columns: tuple[str, ...]
    estimate: PrewarmTableEstimate | None = None

    def wire_request(self, *, request_id: str | None = None) -> dict[str, Any]:
        """Return the query-node request body — no advisory / routing fields."""
        body: dict[str, Any] = {
            "schema_version": PAGE_CACHE_PREWARM_SPEC_SCHEMA_VERSION,
            "db": self.db,
            "table": self.table,
            "columns": list(self.columns),
        }
        if request_id is not None:
            body["id"] = request_id
        if self.table_version is not None:
            body["table_version"] = self.table_version
        if self.concurrency is not None:
            body["concurrency"] = self.concurrency
        return body

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": PREWARM_REQUEST_KIND,
            "db": self.db,
            "table": self.table,
            "columns": list(self.columns),
            "logical_columns": list(self.logical_columns),
            "table_version": self.table_version,
            "concurrency": self.concurrency,
            "wire_request": self.wire_request(),
            "estimate": self.estimate.to_dict() if self.estimate is not None else None,
        }


@dataclass(frozen=True)
class PrewarmPlan:
    """Aggregate page-cache prewarm plan for a training snapshot / epoch."""

    prewarm_id: str
    scope: str
    policy: str
    database: str | None
    applicable: bool
    reason: str | None
    tables: tuple[PrewarmTableRequest, ...]
    excluded_columns: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    kind: str = PREWARM_PLAN_KIND

    def wire_requests(self, *, request_id_prefix: str | None = None) -> list[dict[str, Any]]:
        """Return the query-node request bodies for every table in this plan.

        When ``request_id_prefix`` is given each request gets a stable, deterministic
        ``id`` (``<prefix>-<table>``) so repeated planning yields identical bodies.
        """
        bodies: list[dict[str, Any]] = []
        for table in self.tables:
            request_id = (
                f"{request_id_prefix}-{table.table}" if request_id_prefix is not None else None
            )
            bodies.append(table.wire_request(request_id=request_id))
        return bodies

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "prewarm_id": self.prewarm_id,
            "scope": self.scope,
            "policy": self.policy,
            "database": self.database,
            "applicable": self.applicable,
            "reason": self.reason,
            "tables": [table.to_dict() for table in self.tables],
            "wire_requests": self.wire_requests(request_id_prefix=self.prewarm_id),
            "excluded_columns": [dict(item) for item in self.excluded_columns],
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


def resolve_prewarm_database(
    *,
    uri: str | None,
    connection_kind: str | None,
    namespace_properties: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve the query-node database name for a prewarm request.

    Returns ``None`` for local / object-store lakes (there is no query node to
    prewarm against). Advisory estimates may still be produced for those lakes,
    but no wire request is emitted.
    """
    if connection_kind == "lancedb_remote_db":
        parsed = urlparse(str(uri or ""))
        database = parsed.netloc or parsed.path.lstrip("/").split("/", 1)[0]
        return database or None
    if connection_kind in {"rest_namespace_lancedb", "namespace_lancedb"}:
        props = dict(namespace_properties or {})
        for key in ("database", "db", "namespace"):
            value = props.get(key)
            if value:
                return str(value)
        parsed = urlparse(str(uri or ""))
        return parsed.netloc or None
    return None


def _fixed_width_bytes(dtype: pa.DataType) -> int | None:
    """Return the fixed per-value byte width of an Arrow type, or ``None`` if variable."""
    if pa.types.is_boolean(dtype):
        return 1
    if (
        pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_temporal(dtype)
    ):
        bit_width = getattr(dtype, "bit_width", None)
        if bit_width:
            return max(1, bit_width // 8)
        return 8
    if pa.types.is_decimal(dtype):
        byte_width = getattr(dtype, "byte_width", None)
        if byte_width:
            return int(byte_width)
    if pa.types.is_fixed_size_binary(dtype):
        return int(dtype.byte_width)
    return None


def _column_estimate(
    column: str,
    dtype: pa.DataType | None,
    *,
    rows: int,
    heavy: bool,
    options: PrewarmPlannerOptions,
) -> PrewarmColumnEstimate:
    arrow_type = str(dtype) if dtype is not None else None
    if heavy:
        bpr = options.heavy_bytes_per_row
        if bpr is None:
            return PrewarmColumnEstimate(column, "heavy", arrow_type, None, 0, "unavailable")
        return PrewarmColumnEstimate(
            column, "heavy", arrow_type, float(bpr), int(bpr) * rows, "configured-heavy"
        )
    width = _fixed_width_bytes(dtype) if dtype is not None else None
    if width is not None:
        return PrewarmColumnEstimate(
            column, "metadata", arrow_type, float(width), width * rows, "schema-width"
        )
    variable = options.variable_width_bytes
    return PrewarmColumnEstimate(
        column, "metadata", arrow_type, float(variable), variable * rows, "configured-variable"
    )


def estimate_table_prewarm(
    *,
    table: str,
    version: int | None,
    source_columns: Sequence[str],
    logical_columns: Sequence[str],
    selected_rows: int,
    heavy_columns: Sequence[str],
    metadata: TableMetadata | None,
    options: PrewarmPlannerOptions,
) -> PrewarmTableEstimate:
    """Build the advisory cost estimate for one prewarm table.

    Byte estimates are computed for the *whole* table (``total_rows``) because
    prewarm warms every row of the requested columns. When ``total_rows`` is
    unknown the selected-row count is used as a floor and the source is downgraded.
    """
    heavy_set = set(heavy_columns)
    schema = metadata.schema if metadata is not None else None
    total_rows = metadata.total_rows if metadata is not None else None

    rows_for_bytes = total_rows if total_rows is not None else selected_rows
    schema_field_types: dict[str, pa.DataType] = {}
    if schema is not None:
        for name in schema.names:
            schema_field_types[name] = schema.field(name).type

    column_estimates: list[PrewarmColumnEstimate] = []
    metadata_bytes = 0
    heavy_bytes = 0
    for column in source_columns:
        dtype = schema_field_types.get(column)
        heavy = column in heavy_set
        estimate = _column_estimate(
            column, dtype, rows=rows_for_bytes, heavy=heavy, options=options
        )
        column_estimates.append(estimate)
        if estimate.kind == "heavy":
            heavy_bytes += estimate.estimated_bytes
        else:
            metadata_bytes += estimate.estimated_bytes

    over_warm_ratio: float | None = None
    if total_rows is not None and selected_rows > 0:
        over_warm_ratio = round(total_rows / selected_rows, 4)

    if total_rows is not None:
        source = ESTIMATE_SOURCE_QUERY_NODE
    elif schema is not None:
        source = ESTIMATE_SOURCE_SCHEMA_ONLY
    else:
        source = ESTIMATE_SOURCE_UNAVAILABLE

    return PrewarmTableEstimate(
        table=table,
        version=version,
        selected_rows=selected_rows,
        total_rows=total_rows,
        over_warm_ratio=over_warm_ratio,
        metadata_bytes=metadata_bytes,
        heavy_bytes=heavy_bytes,
        estimated_bytes=metadata_bytes + heavy_bytes,
        columns=tuple(column_estimates),
        source=source,
    )


def build_page_cache_prewarm_plan(
    request: Mapping[str, Any],
    *,
    database: str | None,
    heavy_columns: Sequence[str] = (),
    metadata_fn: MetadataFn | None = None,
    options: PrewarmPlannerOptions | None = None,
    not_applicable_reason: str | None = None,
) -> PrewarmPlan:
    """Build a query-node page-cache prewarm plan from a 0072 prewarm request.

    ``request`` is the existing ``lancedb-robotics/training-prewarm/v1`` envelope
    (it already carries per-table source columns, logical columns, pinned versions
    and the selected row count). This function projects it onto the real Enterprise
    ``PageCacheBeginPrewarmRequest`` shape and attaches advisory estimates. It never
    reads payload data and never references a plan executor.
    """
    options = options or PrewarmPlannerOptions()
    prewarm_id = str(request.get("prewarm_id") or "")
    scope = str(request.get("scope") or "")
    policy = str(request.get("policy") or "")
    excluded = tuple(dict(item) for item in request.get("excluded_columns") or ())
    warnings: list[str] = []

    applicable = database is not None
    reason = not_applicable_reason
    if not applicable and reason is None:
        reason = "page-cache prewarm requires a query-node (db://) connection"

    tables: list[PrewarmTableRequest] = []
    total_selected = 0
    total_est_bytes = 0
    total_heavy_bytes = 0
    for table_entry in request.get("tables") or ():
        table_name = str(table_entry.get("table"))
        version = table_entry.get("version")
        version_int = int(version) if version is not None else None
        source_columns = tuple(str(c) for c in table_entry.get("projected_columns") or ())
        logical_columns = tuple(str(c) for c in table_entry.get("logical_columns") or ())
        selected_rows = int(table_entry.get("row_count") or 0)
        total_selected += selected_rows

        estimate: PrewarmTableEstimate | None = None
        if options.estimate:
            metadata: TableMetadata | None = None
            if metadata_fn is not None:
                try:
                    metadata = metadata_fn(table_name, version_int)
                except Exception as exc:  # advisory only: never fail the plan on it.
                    warnings.append(
                        f"advisory metadata for {table_name!r} unavailable: {exc}"
                    )
                    metadata = None
            estimate = estimate_table_prewarm(
                table=table_name,
                version=version_int,
                source_columns=source_columns,
                logical_columns=logical_columns,
                selected_rows=selected_rows,
                heavy_columns=heavy_columns,
                metadata=metadata,
                options=options,
            )
            total_est_bytes += estimate.estimated_bytes
            total_heavy_bytes += estimate.heavy_bytes

        tables.append(
            PrewarmTableRequest(
                db=database or "",
                table=table_name,
                columns=source_columns,
                table_version=version_int,
                concurrency=options.concurrency,
                logical_columns=logical_columns,
                estimate=estimate,
            )
        )

    over_warm_ratio: float | None = None
    est_total_rows = sum(
        (t.estimate.total_rows or 0)
        for t in tables
        if t.estimate is not None and t.estimate.total_rows is not None
    )
    if est_total_rows and total_selected:
        over_warm_ratio = round(est_total_rows / total_selected, 4)

    metrics = {
        "tables": len(tables),
        "columns": sum(len(t.columns) for t in tables),
        "selected_rows": total_selected,
        "estimated_bytes": total_est_bytes,
        "estimated_heavy_bytes": total_heavy_bytes,
        "estimated_metadata_bytes": total_est_bytes - total_heavy_bytes,
        "over_warm_ratio": over_warm_ratio,
        "skipped_heavy_columns": len(
            [item for item in excluded if "heavy" in str(item.get("reason", ""))]
        ),
    }

    return PrewarmPlan(
        prewarm_id=prewarm_id,
        scope=scope,
        policy=policy,
        database=database,
        applicable=applicable,
        reason=reason if not applicable else None,
        tables=tuple(tables),
        excluded_columns=excluded,
        warnings=tuple(warnings),
        metrics=metrics,
    )
