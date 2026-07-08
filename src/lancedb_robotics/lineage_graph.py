"""Optional ``lance-graph`` Cypher query backend for the canonical lineage graph.

Backlog 0099. This is an **optional, expert-only** surface. Canonical lineage
state stays in the Lance-backed ``lineage_artifacts`` / ``lineage_executions`` /
``lineage_edges`` tables, and the default :meth:`LakeLineage.trace` /
:meth:`LakeLineage.impact` / :meth:`LakeLineage.audit` APIs remain the stable
contract. When the optional ``lance-graph`` extra is installed, this module maps
those same three tables into a ``lance-graph`` property graph so operators can run
Cypher audit / blast-radius queries over the exact same rows -- with no copy to
Neo4j, Kùzu, or DataHub.

Design decisions:

- ``lance-graph`` is never a hard dependency. The backend degrades with an
  actionable missing-extra error, mirroring the embeddings/integration adapter
  pattern. Install with ``pip install 'lancedb-robotics[graph]'``.
- The graph is read from the already-materialized lineage tables. The backend
  does **not** refresh the graph; it reads whatever ``refresh_graph`` /
  automatic emission (backlog 0097/0098) has committed, exactly like the SDK
  traversal APIs.
- ``lineage_edges`` carries every dependency type in one table keyed by
  ``edge_type``. We map it to a single ``DEPENDS_ON`` relationship whose direction
  is ``from_artifact_id`` (upstream/producer) -> ``to_artifact_id``
  (downstream/consumer), matching the SDK edge convention. ``edge_type`` is a
  relationship property, so Cypher can filter on it.
- Node/edge tables are projected to scalar columns before they are handed to the
  planner: the DataFusion strategy does not need the ``list<struct>`` metadata /
  pin columns, and dropping them keeps the property-graph schema small and the
  planner happy.

The variable-length reachability helpers (:meth:`LineageGraphBackend.trace_ids`
and :meth:`LineageGraphBackend.impact_ids`) exist so the spike can prove Cypher
parity with the SDK traversal on the fixture graph and benchmark high-fan-out
graphs; they are not a replacement for the SDK APIs.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lancedb_robotics.lineage import LineageError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa

    from lancedb_robotics.lake import Lake

# --- optional dependency plumbing -------------------------------------------

_EXTRA = "graph"
_MODULE = "lance_graph"

# Property-graph vocabulary. Node labels double as the dataset keys handed to the
# lance-graph engine, so they must be stable.
ARTIFACT_LABEL = "Artifact"
EXECUTION_LABEL = "Execution"
DEPENDS_ON = "DEPENDS_ON"

_ARTIFACTS_TABLE = "lineage_artifacts"
_EXECUTIONS_TABLE = "lineage_executions"
_EDGES_TABLE = "lineage_edges"

# The refresh watermark sentinel lives in lineage_artifacts but is not a real
# graph node; it must never appear as a Cypher node (mirrors lineage.py).
_REFRESH_STATE_KIND = "lineage-refresh-state"

# Scalar columns projected into the property graph. Dropping the list<struct>
# metadata / pin columns keeps the DataFusion planner input simple; every column
# here is queryable as a Cypher property.
_ARTIFACT_NODE_COLUMNS = (
    "artifact_id",
    "kind",
    "name",
    "table_name",
    "table_version",
    "table_tag",
    "row_grain",
    "source_uri",
    "source_id",
    "digest",
    "producer_execution_id",
)
_EXECUTION_NODE_COLUMNS = (
    "execution_id",
    "kind",
    "name",
    "transform_id",
    "status",
    "provider",
    "code_ref",
    "created_by",
)
_EDGE_COLUMNS = (
    "edge_id",
    "edge_type",
    "from_artifact_id",
    "to_artifact_id",
    "execution_id",
)

# A variable-length Cypher path needs a bounded upper hop count, and the
# DataFusion strategy expands each hop into a self-join, so the bound is a real
# cost knob (unlike the SDK's unbounded BFS). This default comfortably covers the
# lineage graph's depth (source -> ... -> model/eval is < 10 hops) while keeping
# the join expansion small. Callers pass an explicit ``max_depth`` to widen it.
_DEFAULT_MAX_DEPTH = 16

# lance-graph 0.5.4 hard-caps variable-length paths at 20 hops
# (``NotImplementedError: Variable-length paths with max length > 20``). The SDK
# trace/impact traversal is unbounded, so deeper reachability must use it; we
# raise an actionable error rather than let the query fail mid-plan.
_MAX_VARLEN_PATH = 20


class LineageGraphError(LineageError):
    """Raised when the optional lance-graph backend cannot serve a query."""


class LineageGraphExtraMissing(LineageGraphError):
    """Raised when the optional ``lance-graph`` extra is not installed."""


def lance_graph_available() -> bool:
    """Return ``True`` when the optional ``lance_graph`` module can be imported."""

    try:
        return importlib.util.find_spec(_MODULE) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def require_lance_graph() -> Any:
    """Import and return the ``lance_graph`` module or raise an actionable error.

    Mirrors the embeddings ``_import_extra`` contract: a genuinely missing extra
    and a present-but-unimportable build both surface as
    :class:`LineageGraphExtraMissing`, never a raw ``ImportError``.
    """

    if importlib.util.find_spec(_MODULE) is None:
        raise LineageGraphExtraMissing(
            "the optional lance-graph lineage query backend needs the "
            f"{_MODULE!r} package; install it with "
            f"`pip install 'lancedb-robotics[{_EXTRA}]'`"
        )
    try:
        return importlib.import_module(_MODULE)
    except (ImportError, OSError) as exc:  # pragma: no cover - env-specific
        raise LineageGraphExtraMissing(
            f"the optional {_MODULE!r} package is installed but failed to import "
            f"({type(exc).__name__}: {exc}); the lance-graph lineage backend is "
            "unavailable on this runtime"
        ) from exc


def _resolve_strategy(module: Any, strategy: str | None) -> Any | None:
    """Map a friendly strategy name to a ``lance_graph.ExecutionStrategy`` value."""

    if strategy is None:
        return None
    normalized = str(strategy).strip().lower().replace("-", "_")
    mapping = {
        "datafusion": "DataFusion",
        "": "DataFusion",
        "native": "LanceNative",
        "lance_native": "LanceNative",
    }
    attr = mapping.get(normalized)
    if attr is None:
        raise LineageGraphError(
            f"unknown execution strategy {strategy!r}; expected 'datafusion' or 'native'"
        )
    return getattr(module.ExecutionStrategy, attr)


# --- results -----------------------------------------------------------------


@dataclass(frozen=True)
class NativeStrategyProbe:
    """Whether the installed lance-graph exposes a working native traversal path.

    The native (CSR/Lance-native) execution strategy is the only reason 0097 would
    reuse lance-graph instead of the SDK's adjacency-indexed BFS, so the spike
    probes it explicitly. ``available`` is ``False`` when the strategy exists as an
    enum value but raises ``NotImplementedError`` at execution.
    """

    version: str
    strategy_names: tuple[str, ...]
    available: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "strategy_names": list(self.strategy_names),
            "native_available": self.available,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CypherParityReport:
    """Reachability parity between the SDK traversal and the Cypher backend.

    Roots are excluded from both sides so the report isolates *traversal*
    reachability (the SDK seeds its result with the root artifacts; the Cypher
    reachability query returns only the far endpoint of each path).
    """

    root_artifact_ids: tuple[str, ...]
    direction: str
    max_depth: int
    sdk_ids: tuple[str, ...]
    cypher_ids: tuple[str, ...]
    only_in_sdk: tuple[str, ...]
    only_in_cypher: tuple[str, ...]

    @property
    def matches(self) -> bool:
        return not self.only_in_sdk and not self.only_in_cypher

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_artifact_ids": list(self.root_artifact_ids),
            "direction": self.direction,
            "max_depth": self.max_depth,
            "matches": self.matches,
            "sdk_count": len(self.sdk_ids),
            "cypher_count": len(self.cypher_ids),
            "only_in_sdk": list(self.only_in_sdk),
            "only_in_cypher": list(self.only_in_cypher),
        }


# --- backend -----------------------------------------------------------------


class LineageGraphBackend:
    """Optional Cypher/property-graph view over the canonical lineage tables."""

    def __init__(self, lake: Lake) -> None:
        self._lake = lake
        self._module = require_lance_graph()

    @property
    def module(self) -> Any:
        return self._module

    def _materialize(self) -> tuple[Any, dict[str, Any]]:
        """Build a matching ``(GraphConfig, datasets)`` pair from the lineage tables.

        A node label / relationship is declared only when its backing table has
        rows: the DataFusion strategy raises ``RuntimeError('Table has no data')``
        for an empty declared table, so config and datasets must stay in lockstep.
        """

        import pyarrow.compute as pc

        artifacts = _select_columns(
            self._lake.table(_ARTIFACTS_TABLE).to_arrow(), _ARTIFACT_NODE_COLUMNS
        )
        if "kind" in artifacts.schema.names:
            kind = artifacts["kind"]
            keep = pc.or_kleene(
                pc.is_null(kind), pc.not_equal(kind, _REFRESH_STATE_KIND)
            )
            artifacts = artifacts.filter(keep)
        executions = _select_columns(
            self._lake.table(_EXECUTIONS_TABLE).to_arrow(), _EXECUTION_NODE_COLUMNS
        )
        edges = _select_columns(
            self._lake.table(_EDGES_TABLE).to_arrow(), _EDGE_COLUMNS
        )

        builder = self._module.GraphConfig.builder()
        datasets: dict[str, Any] = {}
        if artifacts.num_rows:
            builder = builder.with_node_label(ARTIFACT_LABEL, "artifact_id")
            datasets[ARTIFACT_LABEL] = artifacts
        if executions.num_rows:
            builder = builder.with_node_label(EXECUTION_LABEL, "execution_id")
            datasets[EXECUTION_LABEL] = executions
        if edges.num_rows:
            builder = builder.with_relationship(
                DEPENDS_ON, "from_artifact_id", "to_artifact_id"
            )
            datasets[DEPENDS_ON] = edges
        return builder.build(), datasets

    def config(self) -> Any:
        """Return the ``GraphConfig`` mapping the (non-empty) lineage tables."""

        return self._materialize()[0]

    def datasets(self) -> dict[str, Any]:
        """Materialize the property-graph datasets keyed by node label / rel type."""

        return self._materialize()[1]

    def query(
        self,
        query_text: str,
        *,
        parameters: dict[str, Any] | None = None,
        strategy: str | None = None,
        graph: tuple[Any, dict[str, Any]] | None = None,
    ) -> pa.Table:
        """Run a Cypher query over the lineage property graph.

        Returns the raw ``pyarrow.Table`` result. ``parameters`` are bound as
        Cypher query parameters (use ``$name`` in the text). ``strategy`` selects
        the execution strategy: ``"datafusion"`` (default) or ``"native"``; the
        native path currently raises an actionable error because the installed
        lance-graph has not implemented it yet. ``graph`` reuses an already
        materialized ``(config, datasets)`` pair (e.g. across multiple roots).
        """

        config, payload = graph if graph is not None else self._materialize()
        query = self._module.CypherQuery(query_text).with_config(config)
        for key, value in (parameters or {}).items():
            query = query.with_parameter(key, value)
        resolved = _resolve_strategy(self._module, strategy)
        try:
            if resolved is None:
                return query.execute(payload)
            return query.execute(payload, strategy=resolved)
        except NotImplementedError as exc:
            raise LineageGraphError(
                f"lance-graph {getattr(self._module, '__version__', '?')} cannot serve "
                f"this query with the requested strategy: {exc}"
            ) from exc

    def probe_native_strategy(self) -> NativeStrategyProbe:
        """Check whether the installed lance-graph has a working native traversal."""

        import pyarrow as pa

        version = str(getattr(self._module, "__version__", "?"))
        names = tuple(
            name
            for name in dir(self._module.ExecutionStrategy)
            if not name.startswith("_")
        )
        probe_nodes = pa.table({"artifact_id": ["__a", "__b"], "kind": ["x", "y"]})
        probe_edges = pa.table(
            {
                "edge_id": ["__e"],
                "edge_type": ["probe"],
                "from_artifact_id": ["__a"],
                "to_artifact_id": ["__b"],
                "execution_id": ["__x"],
            }
        )
        config = (
            self._module.GraphConfig.builder()
            .with_node_label(ARTIFACT_LABEL, "artifact_id")
            .with_relationship(DEPENDS_ON, "from_artifact_id", "to_artifact_id")
            .build()
        )
        payload = {ARTIFACT_LABEL: probe_nodes, DEPENDS_ON: probe_edges}
        query = self._module.CypherQuery(
            "MATCH (a:Artifact {artifact_id:'__a'})-[:DEPENDS_ON*1..2]->(b:Artifact) "
            "RETURN b.artifact_id AS id"
        ).with_config(config)
        try:
            query.execute(payload, strategy=self._module.ExecutionStrategy.LanceNative)
        except NotImplementedError as exc:
            return NativeStrategyProbe(version, names, False, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            return NativeStrategyProbe(
                version, names, False, f"{type(exc).__name__}: {exc}"
            )
        return NativeStrategyProbe(
            version, names, True, "native execution strategy executed successfully"
        )

    def trace_ids(
        self,
        artifact: str,
        *,
        kind: str | None = None,
        max_depth: int | None = None,
        graph: tuple[Any, dict[str, Any]] | None = None,
    ) -> tuple[str, ...]:
        """Cypher upstream reachability (parity target for ``lineage.trace``)."""

        return self._reachable_ids(
            artifact,
            direction="upstream",
            kind=kind,
            max_depth=max_depth,
            graph=graph,
        )

    def impact_ids(
        self,
        artifact: str,
        *,
        kind: str | None = None,
        max_depth: int | None = None,
        graph: tuple[Any, dict[str, Any]] | None = None,
    ) -> tuple[str, ...]:
        """Cypher downstream reachability (parity target for ``lineage.impact``)."""

        return self._reachable_ids(
            artifact,
            direction="downstream",
            kind=kind,
            max_depth=max_depth,
            graph=graph,
        )

    def compare_traversal(
        self,
        artifact: str,
        *,
        direction: str = "upstream",
        kind: str | None = None,
        max_depth: int | None = None,
    ) -> CypherParityReport:
        """Compare SDK traversal and Cypher reachability for one handle (0099 AC)."""

        from lancedb_robotics.lineage import _resolve_artifact_ids, _traverse_graph

        if direction not in {"upstream", "downstream"}:
            raise LineageGraphError("direction must be 'upstream' or 'downstream'")
        roots = _resolve_artifact_ids(self._lake, artifact, kind=kind)
        if not roots:
            raise LineageGraphError(f"no lineage artifact resolved for {artifact!r}")
        depth = _DEFAULT_MAX_DEPTH if max_depth is None else int(max_depth)
        graph = _traverse_graph(
            self._lake,
            artifact,
            direction=direction,
            kind=kind,
            max_depth=max_depth,
            edge_types=set(),
            target_kinds=set(),
            created_after=None,
            created_before=None,
            table_versions=(),
        )
        root_set = set(roots)
        sdk_ids = {
            row["artifact_id"] for row in graph.artifacts if row.get("artifact_id")
        } - root_set
        cypher_ids = set(
            self._reachable_ids(
                artifact, direction=direction, kind=kind, max_depth=depth
            )
        ) - root_set
        return CypherParityReport(
            root_artifact_ids=tuple(sorted(root_set)),
            direction=direction,
            max_depth=depth,
            sdk_ids=tuple(sorted(sdk_ids)),
            cypher_ids=tuple(sorted(cypher_ids)),
            only_in_sdk=tuple(sorted(sdk_ids - cypher_ids)),
            only_in_cypher=tuple(sorted(cypher_ids - sdk_ids)),
        )

    def _reachable_ids(
        self,
        artifact: str,
        *,
        direction: str,
        kind: str | None,
        max_depth: int | None,
        graph: tuple[Any, dict[str, Any]] | None = None,
    ) -> tuple[str, ...]:
        from lancedb_robotics.lineage import _resolve_artifact_ids

        roots = _resolve_artifact_ids(self._lake, artifact, kind=kind)
        if not roots:
            return ()
        depth = _DEFAULT_MAX_DEPTH if max_depth is None else max(1, int(max_depth))
        if depth > _MAX_VARLEN_PATH:
            raise LineageGraphError(
                f"lance-graph caps variable-length paths at {_MAX_VARLEN_PATH} hops "
                f"(requested max_depth={depth}); the SDK lake.lineage."
                f"{'trace' if direction == 'upstream' else 'impact'}(...) traversal is "
                "unbounded -- use it for deeper reachability"
            )
        materialized = graph if graph is not None else self._materialize()
        _config, payload = materialized
        # No edges materialized -> nothing is reachable and DEPENDS_ON is not
        # declared, so short-circuit rather than run an undefined-relationship query.
        if DEPENDS_ON not in payload:
            return ()
        if direction == "upstream":
            text = (
                f"MATCH (a:Artifact)-[:DEPENDS_ON*1..{depth}]->"
                "(root:Artifact {artifact_id:$root}) "
                "RETURN DISTINCT a.artifact_id AS id"
            )
        else:
            text = (
                "MATCH (root:Artifact {artifact_id:$root})"
                f"-[:DEPENDS_ON*1..{depth}]->(b:Artifact) "
                "RETURN DISTINCT b.artifact_id AS id"
            )
        found: set[str] = set()
        for root in roots:
            result = self.query(text, parameters={"root": root}, graph=materialized)
            if result.num_rows:
                found.update(
                    value for value in result.column("id").to_pylist() if value
                )
        return tuple(sorted(found))


def _select_columns(table: pa.Table, columns: tuple[str, ...]) -> pa.Table:
    """Project a lineage table to the scalar columns the property graph needs."""

    names = [name for name in columns if name in table.schema.names]
    return table.select(names)
