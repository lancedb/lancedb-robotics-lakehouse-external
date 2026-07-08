"""Persistent ANN vector, FTS, and scalar predicate indexes.

Vector and hybrid search were brute-force: ``search.py`` fetched the *entire*
candidate set (``fetch_k = total``) and sorted in Python, with no
``create_index``. That is fine for fixtures and fatal for multi-GB corpora. This
module builds a persistent ANN index over an embedding column so the vector path
hits an index instead of scanning every row -- the "Lance is the *index* and the
fast-access layer" half of decision 0024. The index type defaults to IVF_PQ but
is configurable per build via ``IndexSpec.index_type`` (any of
:data:`SUPPORTED_VECTOR_INDEX_TYPES`), with the quantizer/HNSW params each type
needs.

Re-embedding with a new model is a *new column* with its *own* index (decision
0025: many vector columns side by side, each independently indexed), so building
an index is always scoped to ``(table, column)`` and never disturbs other
columns or the snapshots pinned to them.

IVF_PQ training needs a floor of rows (lance requires >= 256 to train the PQ
codebook). Below that floor we **do not** force an index: brute force over a few
hundred vectors is exact and cheap, and ``search.py`` keeps using it. The build
result records ``skipped`` with the reason so the choice is auditable, never
silent.

Backlog 0022 applies the same managed-index rule to full-text search:
``scenarios.summary`` gets a persistent Lance FTS index at enrich/index time,
and search reuses it instead of rebuilding on every query.

Backlog 0079 adds the same explicit lifecycle for scalar predicates used by
aligned training. Unsupported backends report skipped outcomes so scans still
use predicate pushdown, while supported backends build BTree indexes on typed
columns rather than assuming JSON-path support.

Backlog 0081 extends that lifecycle to curation saved-view membership and
membership-decision hot paths.

Backlog 0090 applies the same lifecycle to review queues. Queue reopen,
pagination, export, and outcome import all filter on a small set of queue-item
columns, so those predicates are indexable when the backend supports scalar
indexes and explicitly reported as skipped when it does not.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.schemas import TRANSFORM_RUNS_SCHEMA

# IVF_PQ PQ-codebook training floor in lance. Tables below this stay brute-force.
MIN_INDEX_ROWS = 256
# Below this row count PQ compression buys nothing (the full-vector matrix fits in
# RAM) and only costs recall -- lance's own guidance flags IVF indexes under ~64k
# rows as not yet meaningful, and at 57k the default IVF_PQ returned 0/10 recall
# (backlog 0186 spike evidence). Scale-aware *defaults* pick IVF_FLAT below this
# floor; an explicit index_type always wins.
PQ_MEANINGFUL_ROWS = 65_536
DEFAULT_METRIC = "cosine"
INDEX_TYPE = "IVF_PQ"
DEFAULT_VECTOR_INDEX_TYPE = INDEX_TYPE
# Lance vector index types wired through ``create_index``. Every one partitions
# with IVF, so ``num_partitions`` (build) and ``nprobes`` (query, see ``search``)
# apply across the whole family; the quantizer and optional HNSW graph layers are
# what the per-type params tune. ``IVF_RQ`` is intentionally excluded pending
# stable residual-quantizer param semantics.
SUPPORTED_VECTOR_INDEX_TYPES = (
    "IVF_FLAT",
    "IVF_SQ",
    "IVF_PQ",
    "IVF_HNSW_FLAT",
    "IVF_HNSW_SQ",
    "IVF_HNSW_PQ",
)
FTS_INDEX_TYPE = "FTS"
SCALAR_INDEX_TYPE = "BTREE"
INDEX_TRANSFORM_KIND = "index"
_SCALAR_INDEX_TYPE_KEYS = frozenset({"BTREE", "BITMAP", "LABEL"})
ALIGNED_TICK_PREDICATE_INDEX_COLUMNS = (
    "alignment_id",
    "run_id",
    "recipe_digest",
    "tick_index",
    "timestamp_ns",
    "has_missing",
    "has_out_of_tolerance",
    "min_confidence",
)
ALIGNED_FRAME_PREDICATE_INDEX_COLUMNS = (
    "alignment_id",
    "tick_index",
    "stream",
    "status",
    "confidence",
)
CURATION_MEMBERSHIP_PREDICATE_INDEX_COLUMNS = (
    "view_id",
    "target_grain",
    "target_id",
    "scenario_id",
    "decision",
    "queue",
    "created_at",
)
CURATION_VIEW_CHUNK_PREDICATE_INDEX_COLUMNS = (
    "view_id",
    "chunk_index",
    "start_ordinal",
    "end_ordinal",
)
CURATION_REVIEW_QUEUE_PREDICATE_INDEX_COLUMNS = (
    "queue_id",
    "queue_name",
    "target_grain",
    "target_id",
    "scenario_id",
    "source_operation",
    "status",
    "assignee",
    "priority",
    "created_at",
)
# Grain + lineage predicate indexes (backlog 0181 / BUG-15). The scalar-index
# lifecycle previously stopped at aligned_*/curation_* and left the two largest
# tables in any real lake unindexed, so the filters behind BUG-10 (run_id on the
# 352K-row observations grain) and BUG-13 (edge endpoints on the 1.29M-row
# lineage_edges graph) were full scans. Index type per column follows lance's
# cardinality guidance (lancedb index.py:48,68; lance btree.md:281-298): BTREE for
# high-cardinality id/range columns (sorted, range-friendly, low build memory --
# the external sort spills to disk), BITMAP only for genuinely low-cardinality
# columns (< ~a few thousand distinct: topic, edge_type). High-cardinality
# endpoints (from/to artifact ids) MUST be BTREE so 0182's chunked
# ``WHERE from_artifact_id IN (frontier)`` rides an indexed ``IsIn`` lookup
# instead of a per-value bitmap blowup. Columns paired with an explicit index type.
OBSERVATION_PREDICATE_INDEX_COLUMNS = (
    ("observation_id", "BTREE"),
    ("run_id", "BTREE"),
    ("topic", "BITMAP"),
    ("timestamp_ns", "BTREE"),
)
LINEAGE_EDGE_PREDICATE_INDEX_COLUMNS = (
    ("from_artifact_id", "BTREE"),
    ("to_artifact_id", "BTREE"),
    ("edge_type", "BITMAP"),
)
LINEAGE_ARTIFACT_PREDICATE_INDEX_COLUMNS = (
    ("artifact_id", "BTREE"),
    ("source_id", "BTREE"),
    ("digest", "BTREE"),
    ("producer_execution_id", "BTREE"),
)
# Eval metric catalog hot predicates (backlog 0095). Metric lookups filter by
# snapshot/run/model/metric/slice identity columns (high-cardinality → BTREE)
# and by lifecycle/regression classification (two-to-three distinct values →
# BITMAP). ``series_key`` is the supersession group (snapshot_name |
# model_version | metric | slice_label digest) so latest-per-series resolution
# and supersede updates ride one indexed equality predicate.
EVAL_METRIC_CATALOG_PREDICATE_INDEX_COLUMNS = (
    ("model_output_id", "BTREE"),
    ("series_key", "BTREE"),
    ("dataset_id", "BTREE"),
    ("snapshot_name", "BTREE"),
    ("training_run_id", "BTREE"),
    ("model_version", "BTREE"),
    ("evaluation_run_id", "BTREE"),
    ("metric", "BTREE"),
    ("slice_label", "BTREE"),
    ("state", "BITMAP"),
    ("regressed", "BITMAP"),
    ("created_at", "BTREE"),
)

# Run-manifest hot predicates (backlog 0100). The training/evaluation manifest
# query and paged-list surfaces filter and order on a small set of identity,
# provenance, and lifecycle columns. High-cardinality id/ref/range columns take
# BTREE (sorted, range-friendly cursor pagination on ``created_at``); the
# low-cardinality lifecycle/classification columns (``status``, ``framework``,
# ``scope``) take BITMAP. ``aliases`` is a list column and is intentionally left
# out -- alias lookup uses a projected scan rather than a per-value bitmap
# blowup. Unsupported backends report ``skipped`` so predicate pushdown still
# applies (see ``build_scalar_index``).
TRAINING_RUN_PREDICATE_INDEX_COLUMNS = (
    ("training_run_id", "BTREE"),
    ("dataset_id", "BTREE"),
    ("snapshot_name", "BTREE"),
    ("code_ref", "BTREE"),
    ("status", "BITMAP"),
    ("created_at", "BTREE"),
)
MODEL_ARTIFACT_PREDICATE_INDEX_COLUMNS = (
    ("model_artifact_id", "BTREE"),
    ("training_run_id", "BTREE"),
    ("artifact_uri", "BTREE"),
    ("checksum", "BTREE"),
    ("framework", "BITMAP"),
    ("created_at", "BTREE"),
)
EVALUATION_RUN_PREDICATE_INDEX_COLUMNS = (
    ("eval_run_id", "BTREE"),
    ("model_artifact_id", "BTREE"),
    ("training_run_id", "BTREE"),
    ("dataset_id", "BTREE"),
    ("snapshot_name", "BTREE"),
    ("status", "BITMAP"),
    ("created_at", "BTREE"),
)
# Materialized eval-metric surface hot predicates (backlog 0100). ``metric_key``
# is the single handle a slice-metric lookup filters on ("night/rain.success_rate");
# ``score`` is BTREE so a value-range predicate ("< 0.8") pushes down.
EVALUATION_RUN_METRICS_PREDICATE_INDEX_COLUMNS = (
    ("eval_run_id", "BTREE"),
    ("model_artifact_id", "BTREE"),
    ("training_run_id", "BTREE"),
    ("dataset_id", "BTREE"),
    ("snapshot_name", "BTREE"),
    ("metric", "BTREE"),
    ("metric_key", "BTREE"),
    ("slice_label", "BTREE"),
    ("score", "BTREE"),
    ("scope", "BITMAP"),
    ("created_at", "BTREE"),
)


class IndexingError(Exception):
    """Raised when a vector index cannot be built as requested."""


@dataclass(frozen=True)
class IndexSpec:
    """Requested ANN index parameters; ``None`` fields are auto-tuned for size.

    ``index_type`` selects the Lance vector index; see
    :data:`SUPPORTED_VECTOR_INDEX_TYPES`. Left ``None`` (the default) the type is
    **scale-aware** (backlog 0186): ``IVF_FLAT`` below
    :data:`PQ_MEANINGFUL_ROWS` vectors -- where PQ's lossy codebook only costs
    recall -- and ``IVF_PQ`` at or above it. An explicit type always wins. The
    quantizer/graph params apply only to the types that have them and are ignored
    otherwise: ``num_sub_vectors``/``num_bits`` for the PQ family
    (``IVF_PQ``/``IVF_HNSW_PQ``), ``m``/``ef_construction`` for the HNSW family
    (``IVF_HNSW_*``). When left ``None`` they fall back to the Lance engine
    defaults. ``num_partitions`` applies to every IVF type and is size-tuned when
    ``None``.
    """

    index_type: str | None = None
    metric: str = DEFAULT_METRIC
    num_partitions: int | None = None
    num_sub_vectors: int | None = None
    num_bits: int | None = None
    m: int | None = None
    ef_construction: int | None = None


@dataclass(frozen=True)
class IndexResult:
    """Outcome of one index build, recorded in lineage for reproducibility."""

    table: str
    column: str
    status: str  # "built" | "skipped"
    index_type: str = INDEX_TYPE
    metric: str | None = None
    num_partitions: int | None = None
    num_sub_vectors: int | None = None
    num_bits: int | None = None
    m: int | None = None
    ef_construction: int | None = None
    num_rows: int | None = None
    dimension: int | None = None
    reason: str | None = None

    def to_params(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FtsIndexResult:
    """Outcome of one FTS index build/refresh."""

    table: str
    column: str
    status: str = "built"
    index_type: str = FTS_INDEX_TYPE
    num_rows: int | None = None
    reason: str | None = None

    def to_params(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ScalarIndexResult:
    """Outcome of one scalar predicate index build, refresh, or probe."""

    table: str
    column: str
    status: str  # "built" | "already_present" | "skipped" | "failed"
    index_type: str = SCALAR_INDEX_TYPE
    num_rows: int | None = None
    reason: str | None = None

    def to_params(self) -> dict:
        return asdict(self)


def _largest_divisor_at_most(dimension: int, cap: int) -> int:
    """Largest divisor of ``dimension`` that is ``<= cap`` (PQ needs dim % subs == 0)."""
    for candidate in range(min(cap, dimension), 0, -1):
        if dimension % candidate == 0:
            return candidate
    return 1


def _auto_params(num_rows: int, dimension: int, spec: IndexSpec) -> tuple[int, int]:
    """Pick sane IVF (partition) + PQ (sub-vector) params when the spec leaves them open.

    ``num_partitions`` applies to every IVF type; ``num_sub_vectors`` is only used
    by the PQ family (the caller drops it for non-PQ index types).
    """
    num_partitions = spec.num_partitions or max(1, min(256, int(math.isqrt(num_rows))))
    # num_sub_vectors must divide the dimension (PQ requirement). Keep each
    # sub-vector at least ~4-D so tiny demo dimensions (e.g. 16) stay valid while
    # real model dimensions (384/512) still split into a useful 16 sub-vectors.
    cap = max(1, min(16, dimension // 4))
    num_sub_vectors = spec.num_sub_vectors or _largest_divisor_at_most(dimension, cap)
    return num_partitions, num_sub_vectors


def scale_aware_index_type(num_rows: int) -> str:
    """Default vector index *type* for a corpus of ``num_rows`` vectors (backlog 0186).

    ``IVF_PQ`` below :data:`PQ_MEANINGFUL_ROWS` is a silent-recall footgun (PQ's
    lossy codebook needs a non-default ``refine_factor`` to rank correctly), so the
    scale-aware default is ``IVF_FLAT`` until the corpus is large enough for PQ
    compression to pay for itself. Only a *default*: callers pass an explicit
    ``IndexSpec.index_type`` to override, and :func:`build_vector_index` still
    skips entirely below :data:`MIN_INDEX_ROWS`.
    """
    return DEFAULT_VECTOR_INDEX_TYPE if num_rows >= PQ_MEANINGFUL_ROWS else "IVF_FLAT"


def _normalize_vector_index_type(index_type: str | None) -> str:
    """Validate and canonicalize a requested vector index type.

    Raises :class:`IndexingError` for anything outside
    :data:`SUPPORTED_VECTOR_INDEX_TYPES` so a typo fails loudly at build time
    rather than silently picking a default index.
    """
    if index_type is None:
        return DEFAULT_VECTOR_INDEX_TYPE
    normalized = str(index_type).strip().upper()
    if normalized not in SUPPORTED_VECTOR_INDEX_TYPES:
        raise IndexingError(
            f"unsupported vector index type {index_type!r}; expected one of "
            f"{', '.join(SUPPORTED_VECTOR_INDEX_TYPES)}"
        )
    return normalized


def _is_pq_index(index_type: str) -> bool:
    """PQ-quantized types carry a product-quantization codebook (num_sub_vectors/num_bits)."""
    return index_type.endswith("PQ")


def _is_hnsw_index(index_type: str) -> bool:
    """HNSW types build a navigable-graph layer over partitions (m/ef_construction)."""
    return "HNSW" in index_type


def _vector_dimension(table: Any, column: str) -> int:
    if column not in table.schema.names:
        raise IndexingError(f"no {column!r} column to index in table {table.name!r}")
    field_type = table.schema.field(column).type
    list_size = getattr(field_type, "list_size", None)
    if list_size is None or list_size <= 0:
        raise IndexingError(
            f"column {column!r} is {field_type}, not a fixed-size-list vector; "
            "enrich/embed it before indexing"
        )
    return list_size


def _index_type_key(index_type: Any) -> str:
    return "".join(char for char in str(index_type).upper() if char.isalnum())


def vector_index_columns(table: Any) -> set[str]:
    """Columns of ``table`` that currently carry a vector index."""
    columns: set[str] = set()
    for index in table.list_indices():
        index_type = _index_type_key(getattr(index, "index_type", ""))
        if index_type == FTS_INDEX_TYPE or index_type in _SCALAR_INDEX_TYPE_KEYS:
            continue
        for column in getattr(index, "columns", None) or []:
            columns.add(column)
    return columns


def fts_index_columns(table: Any) -> set[str]:
    """Columns of ``table`` that currently carry a Lance FTS index."""
    columns: set[str] = set()
    for index in table.list_indices():
        if str(getattr(index, "index_type", "")).upper() != FTS_INDEX_TYPE:
            continue
        for column in getattr(index, "columns", None) or []:
            columns.add(column)
    return columns


def scalar_index_columns(table: Any) -> set[str]:
    """Columns of ``table`` that currently carry a scalar predicate index."""
    columns: set[str] = set()
    try:
        indices = table.list_indices()
    except Exception:
        return columns
    for index in indices:
        if _index_type_key(getattr(index, "index_type", "")) not in _SCALAR_INDEX_TYPE_KEYS:
            continue
        for column in getattr(index, "columns", None) or []:
            columns.add(column)
    return columns


def has_vector_index(table: Any, column: str) -> bool:
    """True if ``column`` has a persistent vector index (so search can use it)."""
    return column in vector_index_columns(table)


def has_fts_index(table: Any, column: str) -> bool:
    """True if ``column`` has a persistent FTS index."""
    return column in fts_index_columns(table)


def has_scalar_index(table: Any, column: str) -> bool:
    """True if ``column`` has a persistent scalar predicate index."""
    return column in scalar_index_columns(table)


def _table_column_names(table: Any) -> set[str]:
    return set(getattr(getattr(table, "schema", None), "names", ()) or ())


def _table_row_count(table: Any) -> int | None:
    try:
        return int(table.count_rows())
    except Exception:
        return None


def _scalar_index_unsupported_reason(table: Any) -> str | None:
    if not callable(getattr(table, "create_scalar_index", None)):
        return (
            "backend does not expose create_scalar_index; predicate pushdown "
            "remains available"
        )
    if not callable(getattr(table, "list_indices", None)):
        return (
            "backend does not expose list_indices; predicate-index availability "
            "cannot be confirmed"
        )
    return None


def _is_unsupported_scalar_index_error(exc: Exception) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    message = str(exc).lower()
    return any(
        fragment in message
        for fragment in (
            "not implemented",
            "not supported",
            "unsupported",
            "unimplemented",
            "not available",
        )
    )


def describe_scalar_indexes(
    lake: Lake,
    *,
    table: str,
    columns: Sequence[str],
) -> tuple[ScalarIndexResult, ...]:
    """Describe scalar-index availability without mutating the table."""
    handle = lake.table(table)
    names = _table_column_names(handle)
    num_rows = _table_row_count(handle)
    unsupported_reason = _scalar_index_unsupported_reason(handle)
    indexed_columns = scalar_index_columns(handle) if unsupported_reason is None else set()
    results: list[ScalarIndexResult] = []
    for column in columns:
        if column not in names:
            results.append(
                ScalarIndexResult(
                    table=table,
                    column=column,
                    status="failed",
                    num_rows=num_rows,
                    reason=f"no {column!r} column to index in table {table!r}",
                )
            )
        elif unsupported_reason is not None:
            results.append(
                ScalarIndexResult(
                    table=table,
                    column=column,
                    status="skipped",
                    num_rows=num_rows,
                    reason=unsupported_reason,
                )
            )
        elif column in indexed_columns:
            results.append(
                ScalarIndexResult(
                    table=table,
                    column=column,
                    status="already_present",
                    num_rows=num_rows,
                )
            )
        else:
            results.append(
                ScalarIndexResult(
                    table=table,
                    column=column,
                    status="skipped",
                    num_rows=num_rows,
                    reason=(
                        "no scalar index is present for this predicate; "
                        "predicate pushdown remains available"
                    ),
                )
            )
    return tuple(results)


def build_scalar_index(
    lake: Lake,
    *,
    table: str,
    column: str,
    replace: bool = False,
    index_type: str = SCALAR_INDEX_TYPE,
) -> ScalarIndexResult:
    """Build or refresh a scalar predicate index when the backend supports it."""
    handle = lake.table(table)
    names = _table_column_names(handle)
    num_rows = _table_row_count(handle)
    if column not in names:
        return ScalarIndexResult(
            table=table,
            column=column,
            status="failed",
            index_type=index_type,
            num_rows=num_rows,
            reason=f"no {column!r} column to index in table {table!r}",
        )
    unsupported_reason = _scalar_index_unsupported_reason(handle)
    if unsupported_reason is not None:
        return ScalarIndexResult(
            table=table,
            column=column,
            status="skipped",
            index_type=index_type,
            num_rows=num_rows,
            reason=unsupported_reason,
        )
    if not replace and has_scalar_index(handle, column):
        return ScalarIndexResult(
            table=table,
            column=column,
            status="already_present",
            index_type=index_type,
            num_rows=num_rows,
        )
    try:
        try:
            handle.create_scalar_index(column, replace=replace, index_type=index_type)
        except TypeError:
            handle.create_scalar_index(column, replace=replace)
    except Exception as exc:  # noqa: BLE001 - backend capability varies by deployment
        status = "skipped" if _is_unsupported_scalar_index_error(exc) else "failed"
        return ScalarIndexResult(
            table=table,
            column=column,
            status=status,
            index_type=index_type,
            num_rows=num_rows,
            reason=(
                f"failed to build {index_type} scalar index on {table}.{column} "
                f"(rows={num_rows}): {exc}"
            ),
        )
    return ScalarIndexResult(
        table=table,
        column=column,
        status="built",
        index_type=index_type,
        num_rows=num_rows,
    )


def build_scalar_indexes(
    lake: Lake,
    *,
    table: str,
    columns: Sequence[str],
    replace: bool = False,
    index_type: str = SCALAR_INDEX_TYPE,
) -> tuple[ScalarIndexResult, ...]:
    """Build scalar predicate indexes for several columns."""
    return tuple(
        build_scalar_index(
            lake,
            table=table,
            column=column,
            replace=replace,
            index_type=index_type,
        )
        for column in columns
    )


def build_aligned_training_predicate_indexes(
    lake: Lake,
    *,
    include_frames: bool = True,
    replace: bool = False,
) -> tuple[ScalarIndexResult, ...]:
    """Build hot scalar predicate indexes for aligned training tables."""
    results = list(
        build_scalar_indexes(
            lake,
            table="aligned_ticks",
            columns=ALIGNED_TICK_PREDICATE_INDEX_COLUMNS,
            replace=replace,
        )
    )
    if include_frames:
        results.extend(
            build_scalar_indexes(
                lake,
                table="aligned_frames",
                columns=ALIGNED_FRAME_PREDICATE_INDEX_COLUMNS,
                replace=replace,
            )
        )
    return tuple(results)


def _build_typed_predicate_indexes(
    lake: Lake,
    *,
    table: str,
    columns: Sequence[tuple[str, str]],
    replace: bool,
) -> tuple[ScalarIndexResult, ...]:
    """Build scalar indexes for ``(column, index_type)`` pairs on one table.

    Unlike :func:`build_scalar_indexes` (one index type for every column), each
    column carries its own type so low-cardinality columns get a BITMAP and
    high-cardinality id/range columns get a BTREE.
    """
    return tuple(
        build_scalar_index(
            lake,
            table=table,
            column=column,
            replace=replace,
            index_type=index_type,
        )
        for column, index_type in columns
    )


def build_observation_predicate_indexes(
    lake: Lake,
    *,
    replace: bool = False,
) -> tuple[ScalarIndexResult, ...]:
    """Build hot scalar predicate indexes for the ``observations`` grain (BUG-15).

    Indexes ``run_id``/``observation_id``/``timestamp_ns`` (BTREE) and ``topic``
    (BITMAP) so ``run_id``-filtered reads (``derive_scenarios``,
    ``downstream_for_run``; spike BUG-10) push down to an indexed scan of one run
    instead of materializing the whole grain. Mirrors
    :func:`build_aligned_training_predicate_indexes`; unsupported backends report
    ``skipped`` so predicate pushdown still applies.
    """
    return _build_typed_predicate_indexes(
        lake,
        table="observations",
        columns=OBSERVATION_PREDICATE_INDEX_COLUMNS,
        replace=replace,
    )


def build_lineage_predicate_indexes(
    lake: Lake,
    *,
    include_artifacts: bool = True,
    replace: bool = False,
) -> tuple[ScalarIndexResult, ...]:
    """Build hot scalar predicate indexes for the lineage graph tables (BUG-15).

    Indexes the ``lineage_edges`` endpoints ``from_artifact_id``/``to_artifact_id``
    (BTREE) and ``edge_type`` (BITMAP), plus the ``lineage_artifacts`` resolution
    keys ``artifact_id``/``source_id``/``digest``/``producer_execution_id``
    (BTREE). This is what lets 0182's frontier expansion fetch only the edges
    incident to the visited frontier via chunked indexed ``IN`` lookups instead of
    loading all 1.29M edges per ``lineage trace``/``impact`` call (BUG-13).
    """
    results = list(
        _build_typed_predicate_indexes(
            lake,
            table="lineage_edges",
            columns=LINEAGE_EDGE_PREDICATE_INDEX_COLUMNS,
            replace=replace,
        )
    )
    if include_artifacts:
        results.extend(
            _build_typed_predicate_indexes(
                lake,
                table="lineage_artifacts",
                columns=LINEAGE_ARTIFACT_PREDICATE_INDEX_COLUMNS,
                replace=replace,
            )
        )
    return tuple(results)


# Grain/lineage tables that carry a managed predicate-index column set. Used by
# `lake maintain` to create (not just refresh) these indexes on existing lakes.
PREDICATE_INDEX_COLUMNS_BY_TABLE: dict[str, tuple[tuple[str, str], ...]] = {
    "observations": OBSERVATION_PREDICATE_INDEX_COLUMNS,
    "lineage_edges": LINEAGE_EDGE_PREDICATE_INDEX_COLUMNS,
    "lineage_artifacts": LINEAGE_ARTIFACT_PREDICATE_INDEX_COLUMNS,
    "eval_metric_catalog": EVAL_METRIC_CATALOG_PREDICATE_INDEX_COLUMNS,
    "training_runs": TRAINING_RUN_PREDICATE_INDEX_COLUMNS,
    "model_artifacts": MODEL_ARTIFACT_PREDICATE_INDEX_COLUMNS,
    "evaluation_runs": EVALUATION_RUN_PREDICATE_INDEX_COLUMNS,
    "evaluation_run_metrics": EVALUATION_RUN_METRICS_PREDICATE_INDEX_COLUMNS,
}


def build_run_manifest_predicate_indexes(
    lake: Lake,
    *,
    include_metrics: bool = True,
    replace: bool = False,
) -> tuple[ScalarIndexResult, ...]:
    """Build hot scalar predicate indexes for the run-manifest tables (backlog 0100).

    Covers ``training_runs``/``model_artifacts``/``evaluation_runs`` and, when
    ``include_metrics`` is set, the materialized ``evaluation_run_metrics`` surface.
    Called by ``run_manifests.sync_evaluation_run_metrics`` after a metric rebuild
    and by ``lake maintain`` via :data:`PREDICATE_INDEX_COLUMNS_BY_TABLE`.
    Unsupported backends report ``skipped``; predicate pushdown still applies.
    """
    results = list(
        _build_typed_predicate_indexes(
            lake,
            table="training_runs",
            columns=TRAINING_RUN_PREDICATE_INDEX_COLUMNS,
            replace=replace,
        )
    )
    results.extend(
        _build_typed_predicate_indexes(
            lake,
            table="model_artifacts",
            columns=MODEL_ARTIFACT_PREDICATE_INDEX_COLUMNS,
            replace=replace,
        )
    )
    results.extend(
        _build_typed_predicate_indexes(
            lake,
            table="evaluation_runs",
            columns=EVALUATION_RUN_PREDICATE_INDEX_COLUMNS,
            replace=replace,
        )
    )
    if include_metrics:
        results.extend(
            _build_typed_predicate_indexes(
                lake,
                table="evaluation_run_metrics",
                columns=EVALUATION_RUN_METRICS_PREDICATE_INDEX_COLUMNS,
                replace=replace,
            )
        )
    return tuple(results)


def build_eval_metric_catalog_predicate_indexes(
    lake: Lake,
    *,
    replace: bool = False,
) -> tuple[ScalarIndexResult, ...]:
    """Build hot scalar predicate indexes for ``eval_metric_catalog`` (backlog 0095).

    Called by ``curate sync-eval-metrics`` after a catalog rebuild and by
    ``lake maintain`` via :data:`PREDICATE_INDEX_COLUMNS_BY_TABLE`. Unsupported
    backends report ``skipped``; predicate pushdown still applies.
    """
    return _build_typed_predicate_indexes(
        lake,
        table="eval_metric_catalog",
        columns=EVAL_METRIC_CATALOG_PREDICATE_INDEX_COLUMNS,
        replace=replace,
    )


def build_predicate_indexes_for_table(
    lake: Lake,
    table: str,
    *,
    replace: bool = False,
) -> tuple[ScalarIndexResult, ...]:
    """Build the managed predicate indexes for ``table`` if it has any (else ``()``).

    Lets a per-table caller (``lake maintain``) create the grain/lineage predicate
    indexes without knowing each table's column set.
    """
    columns = PREDICATE_INDEX_COLUMNS_BY_TABLE.get(table)
    if not columns:
        return ()
    return _build_typed_predicate_indexes(lake, table=table, columns=columns, replace=replace)


def describe_curation_predicate_indexes(
    lake: Lake,
    *,
    include_view_chunks: bool = True,
) -> tuple[ScalarIndexResult, ...]:
    """Describe scalar-index availability for curation hot predicates."""
    results = list(
        describe_scalar_indexes(
            lake,
            table="curation_memberships",
            columns=CURATION_MEMBERSHIP_PREDICATE_INDEX_COLUMNS,
        )
    )
    if include_view_chunks:
        results.extend(
            describe_scalar_indexes(
                lake,
                table="curation_view_membership_chunks",
                columns=CURATION_VIEW_CHUNK_PREDICATE_INDEX_COLUMNS,
            )
        )
    return tuple(results)


def build_curation_predicate_indexes(
    lake: Lake,
    *,
    include_view_chunks: bool = True,
    replace: bool = False,
) -> tuple[ScalarIndexResult, ...]:
    """Build hot scalar predicate indexes for curation view/decision lookups."""
    results = list(
        build_scalar_indexes(
            lake,
            table="curation_memberships",
            columns=CURATION_MEMBERSHIP_PREDICATE_INDEX_COLUMNS,
            replace=replace,
        )
    )
    if include_view_chunks:
        results.extend(
            build_scalar_indexes(
                lake,
                table="curation_view_membership_chunks",
                columns=CURATION_VIEW_CHUNK_PREDICATE_INDEX_COLUMNS,
                replace=replace,
            )
        )
    return tuple(results)


def describe_review_queue_predicate_indexes(lake: Lake) -> tuple[ScalarIndexResult, ...]:
    """Describe scalar-index availability for review queue hot predicates."""
    return describe_scalar_indexes(
        lake,
        table="curation_review_queues",
        columns=CURATION_REVIEW_QUEUE_PREDICATE_INDEX_COLUMNS,
    )


def build_review_queue_predicate_indexes(
    lake: Lake,
    *,
    replace: bool = False,
) -> tuple[ScalarIndexResult, ...]:
    """Build hot scalar predicate indexes for curation review queue lookups."""
    return build_scalar_indexes(
        lake,
        table="curation_review_queues",
        columns=CURATION_REVIEW_QUEUE_PREDICATE_INDEX_COLUMNS,
        replace=replace,
    )


def build_vector_index(
    lake: Lake,
    *,
    table: str,
    column: str,
    spec: IndexSpec | None = None,
) -> IndexResult:
    """Build a persistent ANN index over ``table.column``; return what happened.

    ``spec.index_type`` selects the Lance vector index;
    ``IVF_FLAT``/``IVF_SQ``/``IVF_HNSW_*`` are supported too (see
    :data:`SUPPORTED_VECTOR_INDEX_TYPES`). Left ``None`` the type default is
    **scale-aware** (backlog 0186): :func:`scale_aware_index_type` over the count
    of actual *vectors* (non-null rows of ``column`` -- a mostly-NULL column like
    an observation image embedding has far fewer vectors than table rows), so
    small/medium corpora get ``IVF_FLAT`` (no PQ recall footgun) and only
    PQ-meaningful corpora default to ``IVF_PQ``. The auto choice is recorded in
    the result ``reason`` so callers can echo it; an explicit type always wins.
    Per-type params are passed only for the types that accept them --
    ``num_sub_vectors``/``num_bits`` for the PQ family, ``m``/``ef_construction``
    for the HNSW family -- and unset params fall back to the Lance engine
    defaults. ``num_partitions`` is size-tuned when unset and applies to every
    IVF type.

    Tables below :data:`MIN_INDEX_ROWS` are left brute-force and reported as
    ``skipped`` (with the reason) rather than failed -- forcing an index there
    would only error in IVF/PQ training. Existing indexes on the column are
    replaced.
    """
    spec = spec or IndexSpec()
    handle = lake.table(table)
    dimension = _vector_dimension(handle, column)
    num_rows = handle.count_rows()

    auto_note: str | None = None
    if spec.index_type is None:
        try:
            vector_rows = handle.count_rows(f"{column} IS NOT NULL")
        except Exception:  # noqa: BLE001 - backends without filtered counts: fall back
            vector_rows = num_rows
        index_type = scale_aware_index_type(vector_rows)
        if index_type != DEFAULT_VECTOR_INDEX_TYPE:
            auto_note = (
                f"scale-aware default: {vector_rows} vectors < {PQ_MEANINGFUL_ROWS} "
                f"PQ-meaningful floor -> {index_type}; pass an explicit index type "
                "to override"
            )
    else:
        index_type = _normalize_vector_index_type(spec.index_type)

    if num_rows < MIN_INDEX_ROWS:
        return IndexResult(
            table=table,
            column=column,
            status="skipped",
            index_type=index_type,
            metric=spec.metric,
            num_rows=num_rows,
            dimension=dimension,
            reason=(
                f"{num_rows} rows < {MIN_INDEX_ROWS} required to train {index_type}; "
                "search stays brute-force (exact at this size)"
            ),
        )

    num_partitions, auto_sub_vectors = _auto_params(num_rows, dimension, spec)
    is_pq = _is_pq_index(index_type)
    is_hnsw = _is_hnsw_index(index_type)
    # Sub-vectors are a PQ-only concept; the auto value is irrelevant for FLAT/SQ.
    num_sub_vectors = auto_sub_vectors if is_pq else None

    create_kwargs: dict[str, Any] = {
        "metric": spec.metric,
        "vector_column_name": column,
        "num_partitions": num_partitions,
        "index_type": index_type,
        "replace": True,
    }
    if is_pq:
        create_kwargs["num_sub_vectors"] = num_sub_vectors
        if spec.num_bits is not None:
            create_kwargs["num_bits"] = spec.num_bits
    if is_hnsw:
        if spec.m is not None:
            create_kwargs["m"] = spec.m
        if spec.ef_construction is not None:
            create_kwargs["ef_construction"] = spec.ef_construction

    try:
        handle.create_index(**create_kwargs)
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable failure
        raise IndexingError(
            f"failed to build {index_type} index on {table}.{column} "
            f"(rows={num_rows}, dim={dimension}, partitions={num_partitions}, "
            f"sub_vectors={num_sub_vectors}): {exc}"
        ) from exc

    return IndexResult(
        table=table,
        column=column,
        status="built",
        index_type=index_type,
        metric=spec.metric,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        num_bits=spec.num_bits if is_pq else None,
        m=spec.m if is_hnsw else None,
        ef_construction=spec.ef_construction if is_hnsw else None,
        num_rows=num_rows,
        dimension=dimension,
        reason=auto_note,  # visible when the scale-aware default downgraded to FLAT
    )


def build_fts_index(
    lake: Lake,
    *,
    table: str,
    column: str,
    replace: bool = True,
) -> FtsIndexResult:
    """Build or refresh a persistent Lance FTS index over ``table.column``."""
    handle = lake.table(table)
    if column not in handle.schema.names:
        raise IndexingError(f"no {column!r} column to index in table {table!r}")
    num_rows = handle.count_rows()
    try:
        handle.create_fts_index(column, replace=replace)
    except Exception as exc:  # noqa: BLE001 - wrap the engine error with context
        raise IndexingError(
            f"failed to build {FTS_INDEX_TYPE} index on {table}.{column} "
            f"(rows={num_rows}): {exc}"
        ) from exc
    return FtsIndexResult(table=table, column=column, num_rows=num_rows)


def record_index_transform(
    lake: Lake,
    result: IndexResult,
    *,
    created_by: str = "lancedb-robotics",
) -> str:
    """Record a standalone index build as a ``kind="index"`` transform row.

    Used by the CLI ``scenarios index`` step. The inline enrich-time path folds
    the same params into its own enrichment row instead (see ``enrich``).
    """
    transform_id = f"tfm-index-{result.table}-{result.column}"
    now = datetime.now(UTC)
    transform_row = {
        "transform_id": transform_id,
        "kind": INDEX_TRANSFORM_KIND,
        "input_uris": [],
        "input_table_versions": [],
        "output_tables": [result.table],
        "params": json.dumps(result.to_params(), sort_keys=True),
        "status": "completed",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{transform_id}'")
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): record the vector index build's lineage
    # slice without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
    return transform_id


def record_fts_index_transform(
    lake: Lake,
    result: FtsIndexResult,
    *,
    created_by: str = "lancedb-robotics",
) -> str:
    """Record a standalone FTS index build as a ``kind="index"`` transform row."""
    transform_id = f"tfm-index-{result.table}-{result.column}-fts"
    now = datetime.now(UTC)
    transform_row = {
        "transform_id": transform_id,
        "kind": INDEX_TRANSFORM_KIND,
        "input_uris": [],
        "input_table_versions": [],
        "output_tables": [result.table],
        "params": json.dumps(result.to_params(), sort_keys=True),
        "status": "completed",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{transform_id}'")
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): record the FTS index build's lineage
    # slice without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
    return transform_id


def record_scalar_index_transform(
    lake: Lake,
    result: ScalarIndexResult,
    *,
    created_by: str = "lancedb-robotics",
) -> str:
    """Record a standalone scalar predicate index build/probe result."""
    transform_id = f"tfm-index-{result.table}-{result.column}-scalar"
    now = datetime.now(UTC)
    transform_row = {
        "transform_id": transform_id,
        "kind": INDEX_TRANSFORM_KIND,
        "input_uris": [],
        "input_table_versions": [],
        "output_tables": [result.table],
        "params": json.dumps(result.to_params(), sort_keys=True),
        "status": "completed" if result.status != "failed" else "failed",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{transform_id}'")
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): record the scalar index build's lineage
    # slice without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
    return transform_id


def _iter_recorded_fts_index_params(lake: Lake, *, table: str, column: str) -> list[dict]:
    recorded: list[dict] = []
    for row in lake.table("transform_runs").to_arrow().to_pylist():
        if not row.get("params"):
            continue
        try:
            params = json.loads(row["params"])
        except json.JSONDecodeError:
            continue
        candidates = []
        if row.get("kind") == INDEX_TRANSFORM_KIND:
            candidates.append(params)
        if row.get("kind") == "enrichment" and isinstance(params.get("fts_index"), dict):
            candidates.append(params["fts_index"])
        if row.get("kind") == "maintenance":
            table_params = (params.get("tables") or {}).get(table) or {}
            candidates.extend(table_params.get("indexes_refreshed") or [])
        for candidate in candidates:
            if (
                candidate.get("index_type") == FTS_INDEX_TYPE
                and candidate.get("table") == table
                and candidate.get("column") == column
            ):
                recorded.append(candidate)
    return recorded


def is_fts_index_stale(lake: Lake, *, table: str, column: str) -> bool:
    """Return True when lineage says the FTS index predates current row count."""
    recorded = _iter_recorded_fts_index_params(lake, table=table, column=column)
    if not recorded:
        return False
    indexed_rows = recorded[-1].get("num_rows")
    if indexed_rows is None:
        return False
    return int(indexed_rows) != lake.table(table).count_rows()
