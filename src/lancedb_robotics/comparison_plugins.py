"""Extension hooks for optional curation-comparison metric sections.

Backlog 0094: domain-specific comparison metrics (regression tracing, benchmark
outcomes, review-queue SLA, partner-specific evals) can be contributed as
plugins instead of being hard-coded into ``curate.py``. A plugin

- declares the source tables it reads (so the comparison plan can estimate scan
  cost and required table versions),
- estimates its scan size and whether it can run locally or needs an external
  executor,
- computes a report section, and
- can thread its own transform ids into the comparison's lineage.

The interface is intentionally small and decoupled from the curation module so
that partner code can register a metric without importing curation internals.
``curate.compare(...)`` binds a bounded-memory ``stream`` helper onto the context
so plugins do not have to materialize whole tables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lancedb_robotics.lake import Lake

_LOCAL_EXECUTION = "local"
_EXTERNAL_EXECUTION = "external"
_EXECUTIONS = (_LOCAL_EXECUTION, _EXTERNAL_EXECUTION)


class ComparisonPluginError(Exception):
    """Raised when a comparison metric plugin is unknown or misconfigured."""


@dataclass(frozen=True)
class ComparisonMetricContext:
    """Read-only inputs handed to a comparison metric plugin.

    Prefer :meth:`stream` for scans: it pushes column projection and an optional
    SQL predicate into Lance and yields bounded row batches instead of
    materializing a whole table in Python.
    """

    lake: Lake
    left: str
    right: str
    left_snapshot: Mapping[str, Any]
    right_snapshot: Mapping[str, Any]
    left_dataset_id: str
    right_dataset_id: str
    left_ids: tuple[str, ...]
    right_ids: tuple[str, ...]
    dimensions: tuple[str, ...]
    membership: Mapping[str, Any]
    batch_size: int = 4096
    _scanner: Callable[..., Iterator[list[dict[str, Any]]]] | None = field(
        default=None, repr=False
    )

    def stream(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        where_sql: str | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield bounded row batches from ``table`` with projection/filter pushdown."""
        if self._scanner is None:
            raise ComparisonPluginError(
                "comparison context has no scan helper bound; build the context "
                "through curate.compare(...) or pass a scanner"
            )
        yield from self._scanner(
            self.lake,
            table,
            columns=columns,
            where_sql=where_sql,
            batch_size=self.batch_size,
        )


class ComparisonMetricPlugin(ABC):
    """Base class for an optional comparison report section.

    Subclasses set a unique ``name`` and implement :meth:`compute`. The other
    hooks are optional and only feed the comparison plan / lineage.
    """

    #: Unique metric name; also the report-section key under ``report["plugins"]``.
    name: str = ""

    def required_tables(self, ctx: ComparisonMetricContext) -> tuple[str, ...]:
        """Source tables this metric reads (for plan cost + table versions)."""
        return ()

    def estimate_rows(self, ctx: ComparisonMetricContext) -> int:
        """Estimated rows scanned to compute the section (for plan cost)."""
        return 0

    def execution(self, ctx: ComparisonMetricContext) -> str:
        """``"local"`` (default) or ``"external"`` when it needs an executor."""
        return _LOCAL_EXECUTION

    @abstractmethod
    def compute(self, ctx: ComparisonMetricContext) -> dict[str, Any]:
        """Return the JSON-able report section for this metric."""

    def lineage_transform_ids(self, section: Mapping[str, Any]) -> tuple[str, ...]:
        """Transform ids the section should thread into comparison lineage."""
        return ()


_REGISTRY: dict[str, ComparisonMetricPlugin] = {}


def register_comparison_plugin(
    plugin: ComparisonMetricPlugin, *, replace: bool = False
) -> ComparisonMetricPlugin:
    """Register a plugin under its ``name`` so it can be selected by name."""
    name = getattr(plugin, "name", "") or ""
    if not name:
        raise ComparisonPluginError("comparison plugin must define a non-empty name")
    if name in _REGISTRY and not replace:
        raise ComparisonPluginError(
            f"comparison plugin {name!r} already registered; pass replace=True to override"
        )
    _REGISTRY[name] = plugin
    return plugin


def unregister_comparison_plugin(name: str) -> None:
    _REGISTRY.pop(name, None)


def clear_comparison_plugins() -> None:
    """Drop all registered plugins (intended for tests)."""
    _REGISTRY.clear()


def registered_comparison_plugins() -> dict[str, ComparisonMetricPlugin]:
    return dict(_REGISTRY)


def resolve_comparison_plugins(
    plugins: Sequence[ComparisonMetricPlugin | str] | None,
) -> tuple[ComparisonMetricPlugin, ...]:
    """Normalize a mix of plugin instances and registered names to instances."""
    resolved: list[ComparisonMetricPlugin] = []
    seen: set[str] = set()
    for item in plugins or ():
        if isinstance(item, ComparisonMetricPlugin):
            plugin = item
        elif isinstance(item, str):
            if item not in _REGISTRY:
                raise ComparisonPluginError(
                    f"unknown comparison plugin {item!r}; registered: {sorted(_REGISTRY)}"
                )
            plugin = _REGISTRY[item]
        else:
            raise ComparisonPluginError(
                "comparison plugins must be ComparisonMetricPlugin instances or "
                "registered plugin names"
            )
        name = getattr(plugin, "name", "") or ""
        if not name:
            raise ComparisonPluginError("comparison plugin must define a non-empty name")
        if name in seen:
            continue
        seen.add(name)
        resolved.append(plugin)
    return tuple(resolved)
