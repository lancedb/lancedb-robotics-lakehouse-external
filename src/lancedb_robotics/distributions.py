"""Distribution specs, reports, comparisons, and gap findings over lake slices."""

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa

from lancedb_robotics.dataset import SnapshotManifest
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.schemas import DISTRIBUTION_CATALOG_SCHEMA, TRANSFORM_RUNS_SCHEMA

_SOURCE_TABLES = (
    "scenarios",
    "runs",
    "observations",
    "events",
    "labels",
    "model_outputs",
    "feedback",
    "dataset_snapshots",
    "curation_views",
    "curation_memberships",
    "transform_runs",
)
_GAP_KINDS = ("missing", "underrepresented", "overrepresented")
_CARDINALITY_ACTIONS = ("error", "bucket")
_DEFAULT_BATCH_SIZE = 4096
_DEFAULT_FILTER_CHUNK_SIZE = 1000
_DEFAULT_RARE_SLICE_LABEL = "__bucket__=rare"
_DEFAULT_OVERFLOW_SLICE_LABEL = "__bucket__=overflow"


class DistributionError(Exception):
    """Raised when a distribution spec, source, or comparison is invalid."""


@dataclass(frozen=True)
class DistributionSpec:
    """Versionable definition of dimensions and minimum coverage targets."""

    name: str
    dimensions: tuple[str, ...]
    min_count_per_slice: int = 0
    slice_min_counts: tuple[tuple[str, int], ...] = ()
    weights: tuple[tuple[str, float], ...] = ()
    scope: dict[str, Any] = field(default_factory=dict)
    spec_id: str = ""
    transform_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec_id": self.spec_id,
            "dimensions": list(self.dimensions),
            "min_count_per_slice": self.min_count_per_slice,
            "slice_min_counts": dict(self.slice_min_counts),
            "weights": dict(self.weights),
            "scope": _jsonable(self.scope),
            "transform_id": self.transform_id,
        }

    def min_count_for(self, label: str) -> int:
        return dict(self.slice_min_counts).get(label, self.min_count_per_slice)


@dataclass(frozen=True)
class DistributionSlice:
    """Observed metrics for one dimension slice."""

    label: str
    values: tuple[tuple[str, str], ...]
    count: int
    percentage: float
    scenario_ids: tuple[str, ...] = ()
    quality_stats: dict[str, Any] = field(default_factory=dict)
    label_completeness: dict[str, Any] = field(default_factory=dict)
    duplicate_pressure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "values": dict(self.values),
            "count": self.count,
            "percentage": self.percentage,
            "scenario_ids": list(self.scenario_ids),
            "quality_stats": _jsonable(self.quality_stats),
            "label_completeness": _jsonable(self.label_completeness),
            "duplicate_pressure": _jsonable(self.duplicate_pressure),
        }


@dataclass(frozen=True)
class DistributionExecutionOptions:
    """Controls for bounded distribution-report execution."""

    batch_size: int = _DEFAULT_BATCH_SIZE
    filter_chunk_size: int = _DEFAULT_FILTER_CHUNK_SIZE
    max_slice_count: int | None = None
    overflow: str = "error"
    rare_slice_min_count: int = 0
    rare_slice_label: str = _DEFAULT_RARE_SLICE_LABEL
    overflow_slice_label: str = _DEFAULT_OVERFLOW_SLICE_LABEL
    top_k_overflow: int = 10
    max_scenario_ids_per_slice: int | None = None
    slice_bins: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "filter_chunk_size": self.filter_chunk_size,
            "max_slice_count": self.max_slice_count,
            "overflow": self.overflow,
            "rare_slice_min_count": self.rare_slice_min_count,
            "rare_slice_label": self.rare_slice_label,
            "overflow_slice_label": self.overflow_slice_label,
            "top_k_overflow": self.top_k_overflow,
            "max_scenario_ids_per_slice": self.max_scenario_ids_per_slice,
            "slice_bins": dict(self.slice_bins),
        }


@dataclass(frozen=True)
class DistributionReport:
    """Measured distribution for one source, with source table-version evidence."""

    lake: Lake
    report_id: str
    spec: DistributionSpec
    source: dict[str, Any]
    slices: tuple[DistributionSlice, ...]
    total_count: int
    table_versions: tuple[tuple[str, int], ...]
    transform_id: str
    created_at: datetime
    execution: dict[str, Any] = field(default_factory=dict)

    @property
    def slice_counts(self) -> dict[str, int]:
        return {item.label: item.count for item in self.slices}

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for item in self.slices:
            ids.extend(item.scenario_ids)
        return tuple(dict.fromkeys(ids))

    def slice(self, label: str) -> DistributionSlice | None:
        return next((item for item in self.slices if item.label == label), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": "distribution-report",
            "report_id": self.report_id,
            "spec": self.spec.to_dict(),
            "source": _jsonable(self.source),
            "total_count": self.total_count,
            "slice_counts": self.slice_counts,
            "slices": [item.to_dict() for item in self.slices],
            "table_versions": [
                {"table": table, "version": version, "tag": ""}
                for table, version in self.table_versions
            ],
            "transform_id": self.transform_id,
            "created_at": self.created_at.isoformat(),
            "execution": _jsonable(self.execution),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Distribution Report: {self.spec.name}",
            "",
            f"- report: `{self.report_id}`",
            f"- source: `{self.source.get('kind', 'unknown')}`",
            f"- total: {self.total_count}",
            "",
            "| slice | count | percentage |",
            "| --- | ---: | ---: |",
        ]
        for item in self.slices:
            lines.append(f"| `{item.label}` | {item.count} | {item.percentage:.2%} |")
        return "\n".join(lines)


@dataclass(frozen=True)
class GapFinding:
    """Actionable distribution difference that can feed curation or collection."""

    kind: str
    label: str
    values: tuple[tuple[str, str], ...]
    observed_count: int
    target_count: int
    needed_count: int
    observed_percentage: float
    target_percentage: float
    delta_count: int
    delta_percentage: float
    severity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "values": dict(self.values),
            "observed_count": self.observed_count,
            "target_count": self.target_count,
            "needed_count": self.needed_count,
            "observed_percentage": self.observed_percentage,
            "target_percentage": self.target_percentage,
            "delta_count": self.delta_count,
            "delta_percentage": self.delta_percentage,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class DistributionComparison:
    """Observed-vs-target distribution comparison with sorted gap findings."""

    comparison_id: str
    observed: DistributionReport
    target: DistributionReport
    gap_findings: tuple[GapFinding, ...]
    summary: dict[str, Any]
    transform_id: str
    created_at: datetime

    def top(self, limit: int) -> tuple[GapFinding, ...]:
        if limit <= 0:
            return ()
        return self.gap_findings[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": "distribution-comparison",
            "comparison_id": self.comparison_id,
            "observed_report_id": self.observed.report_id,
            "target_report_id": self.target.report_id,
            "spec": self.observed.spec.to_dict(),
            "summary": _jsonable(self.summary),
            "gap_findings": [item.to_dict() for item in self.gap_findings],
            "transform_id": self.transform_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class DistributionCatalogEntry:
    """Queryable catalog metadata for a persisted distribution artifact."""

    catalog_id: str
    kind: str
    name: str
    spec_id: str
    report_id: str
    comparison_id: str
    finding_id: str
    source_kind: str
    source_name: str
    source_id: str
    dataset_id: str
    view_id: str
    source_digest: str
    summary: dict[str, Any]
    body: dict[str, Any] | None
    body_sha1: str
    body_bytes: int
    body_compacted: bool
    retention_policy: dict[str, Any]
    expires_at: datetime | None
    compacted_at: datetime | None
    table_versions: tuple[tuple[str, int], ...]
    source_transform_ids: tuple[str, ...]
    created_by: str
    transform_id: str
    created_at: datetime

    def to_dict(self, *, include_body: bool = False) -> dict[str, Any]:
        payload = {
            "catalog_id": self.catalog_id,
            "kind": self.kind,
            "name": self.name,
            "spec_id": self.spec_id,
            "report_id": self.report_id,
            "comparison_id": self.comparison_id,
            "finding_id": self.finding_id,
            "source_kind": self.source_kind,
            "source_name": self.source_name,
            "source_id": self.source_id,
            "dataset_id": self.dataset_id,
            "view_id": self.view_id,
            "source_digest": self.source_digest,
            "summary": _jsonable(self.summary),
            "body_sha1": self.body_sha1,
            "body_bytes": self.body_bytes,
            "body_compacted": self.body_compacted,
            "retention_policy": _jsonable(self.retention_policy),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "compacted_at": self.compacted_at.isoformat() if self.compacted_at else None,
            "table_versions": [
                {"table": table, "version": version, "tag": ""}
                for table, version in self.table_versions
            ],
            "source_transform_ids": list(self.source_transform_ids),
            "created_by": self.created_by,
            "transform_id": self.transform_id,
            "created_at": self.created_at.isoformat(),
        }
        if include_body:
            payload["body"] = _jsonable(self.body)
        return payload


@dataclass(frozen=True)
class DistributionRetentionReport:
    """Result of applying distribution-catalog body retention rules."""

    compacted_catalog_ids: tuple[str, ...]
    retained_catalog_ids: tuple[str, ...]
    dry_run: bool
    body_bytes_before: int
    body_bytes_after: int
    policy: dict[str, Any]
    transform_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def compacted_count(self) -> int:
        return len(self.compacted_catalog_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": "distribution-catalog-retention",
            "compacted_catalog_ids": list(self.compacted_catalog_ids),
            "retained_catalog_ids": list(self.retained_catalog_ids),
            "dry_run": self.dry_run,
            "body_bytes_before": self.body_bytes_before,
            "body_bytes_after": self.body_bytes_after,
            "policy": _jsonable(self.policy),
            "transform_id": self.transform_id,
            "created_at": self.created_at.isoformat(),
        }


class LakeDistributions:
    """Facade exposed as ``lake.distributions``."""

    def __init__(self, lake: Lake) -> None:
        self._lake = lake

    def define(
        self,
        *,
        name: str,
        dimensions: Sequence[str],
        min_count_per_slice: int = 0,
        slice_min_counts: Mapping[str, int] | None = None,
        weights: Mapping[str, float] | None = None,
        scope: Any = None,
        created_by: str = "lancedb-robotics",
    ) -> DistributionSpec:
        """Define the dimensions and minimum counts for a distribution report."""
        spec_name = str(name).strip()
        if not spec_name:
            raise DistributionError("distribution spec name must not be empty")
        normalized_dimensions = tuple(dict.fromkeys(str(item) for item in dimensions if str(item)))
        if not normalized_dimensions:
            raise DistributionError("distribution spec requires at least one dimension")
        if min_count_per_slice < 0:
            raise DistributionError("min_count_per_slice must be non-negative")
        normalized_slice_mins = tuple(
            sorted(
                (str(label), int(count))
                for label, count in (slice_min_counts or {}).items()
                if str(label)
            )
        )
        if any(count < 0 for _, count in normalized_slice_mins):
            raise DistributionError("slice minimum counts must be non-negative")
        normalized_weights = tuple(
            sorted((str(label), float(weight)) for label, weight in (weights or {}).items())
        )
        normalized_scope = _scope_payload(scope)
        spec_payload = {
            "name": spec_name,
            "dimensions": list(normalized_dimensions),
            "min_count_per_slice": min_count_per_slice,
            "slice_min_counts": dict(normalized_slice_mins),
            "weights": dict(normalized_weights),
            "scope": normalized_scope,
        }
        spec_id = "dist-spec-" + _digest(spec_payload)
        transform_id = _record_distribution_transform(
            self._lake,
            operation="spec",
            payload={**spec_payload, "spec_id": spec_id},
            input_table_versions=_table_versions(self._lake),
            output_tables=("distribution_catalog",),
            created_by=created_by,
        )
        spec = DistributionSpec(
            name=spec_name,
            dimensions=normalized_dimensions,
            min_count_per_slice=min_count_per_slice,
            slice_min_counts=normalized_slice_mins,
            weights=normalized_weights,
            scope=normalized_scope,
            spec_id=spec_id,
            transform_id=transform_id,
        )
        _persist_distribution_catalog(
            self._lake,
            kind="spec",
            name=spec.name,
            spec_id=spec.spec_id,
            report_id="",
            comparison_id="",
            finding_id="",
            source={},
            summary={
                "dimension_count": len(spec.dimensions),
                "min_count_per_slice": spec.min_count_per_slice,
            },
            body=spec.to_dict(),
            table_versions=tuple(
                (str(item["table"]), int(item["version"]))
                for item in _table_versions(self._lake)
            ),
            transform_id=transform_id,
            created_by=created_by,
        )
        return spec

    def measure(
        self,
        spec: DistributionSpec | Mapping[str, Any],
        *,
        source: Any = None,
        name: str | None = None,
        created_by: str = "lancedb-robotics",
        execution: DistributionExecutionOptions | Mapping[str, Any] | None = None,
        batch_size: int | None = None,
        max_slice_count: int | None = None,
        overflow: str | None = None,
        rare_slice_min_count: int | None = None,
        top_k_overflow: int | None = None,
        max_scenario_ids_per_slice: int | None = None,
        slice_bins: Mapping[str, str] | None = None,
    ) -> DistributionReport:
        """Measure counts, percentages, quality, labels, and duplicate pressure."""
        normalized_spec = ensure_distribution_spec(spec)
        options = ensure_distribution_execution_options(
            execution,
            batch_size=batch_size,
            max_slice_count=max_slice_count,
            overflow=overflow,
            rare_slice_min_count=rare_slice_min_count,
            top_k_overflow=top_k_overflow,
            max_scenario_ids_per_slice=max_scenario_ids_per_slice,
            slice_bins=slice_bins,
        )
        normalized_source = _resolve_source(self._lake, source, normalized_spec)
        if normalized_source.get("external"):
            slices = _external_slices(normalized_source, normalized_spec)
            total_count = sum(item.count for item in slices)
            execution_report = {
                "strategy": "external-manifest",
                "options": options.to_dict(),
            }
        else:
            slices, total_count, execution_report = _measure_distribution_slices(
                self._lake,
                normalized_source,
                normalized_spec.dimensions,
                options,
            )

        table_versions = tuple(
            (str(item["table"]), int(item["version"]))
            for item in normalized_source.get("table_versions") or _table_versions(self._lake)
        )
        report_payload = {
            "name": name or normalized_spec.name,
            "spec": normalized_spec.to_dict(),
            "source": normalized_source,
            "total_count": total_count,
            "slice_counts": {item.label: item.count for item in slices},
            "slices": [item.to_dict() for item in slices],
            "execution": execution_report,
            "table_versions": [
                {"table": table, "version": version, "tag": ""}
                for table, version in table_versions
            ],
        }
        report_id = "dist-report-" + _digest(report_payload)
        now = datetime.now(UTC)
        transform_id = _record_distribution_transform(
            self._lake,
            operation="report",
            payload={**report_payload, "report_id": report_id},
            input_table_versions=[
                {"table": table, "version": version, "tag": ""}
                for table, version in table_versions
            ],
            output_tables=("distribution_catalog",),
            created_by=created_by,
        )
        report = DistributionReport(
            lake=self._lake,
            report_id=report_id,
            spec=normalized_spec,
            source=normalized_source,
            slices=slices,
            total_count=total_count,
            table_versions=table_versions,
            transform_id=transform_id,
            created_at=now,
            execution=execution_report,
        )
        _persist_distribution_catalog(
            self._lake,
            kind="report",
            name=str(report_payload["name"]),
            spec_id=normalized_spec.spec_id,
            report_id=report_id,
            comparison_id="",
            finding_id="",
            source=normalized_source,
            summary={
                "total_count": total_count,
                "slice_count": len(slices),
                "slice_counts": report.slice_counts,
                "execution_strategy": execution_report.get("strategy", ""),
            },
            body=report.to_dict(),
            table_versions=table_versions,
            transform_id=transform_id,
            created_by=created_by,
            created_at=now,
        )
        return report

    def compare(
        self,
        *,
        observed: DistributionReport | Mapping[str, Any],
        target: DistributionReport | Mapping[str, Any],
        overrepresented_threshold: float = 0.05,
        created_by: str = "lancedb-robotics",
    ) -> DistributionComparison:
        """Compare an observed report against a target report and emit gaps."""
        observed_report = ensure_distribution_report(observed, self._lake)
        target_report = ensure_distribution_report(target, self._lake)
        if observed_report.spec.dimensions != target_report.spec.dimensions:
            raise DistributionError("observed and target reports must use the same dimensions")
        if overrepresented_threshold < 0:
            raise DistributionError("overrepresented_threshold must be non-negative")

        findings = _gap_findings(
            observed_report,
            target_report,
            overrepresented_threshold=overrepresented_threshold,
        )
        summary = {
            "observed_total_count": observed_report.total_count,
            "target_total_count": target_report.total_count,
            "finding_count": len(findings),
            "missing_count": sum(1 for item in findings if item.kind == "missing"),
            "underrepresented_count": sum(
                1 for item in findings if item.kind == "underrepresented"
            ),
            "overrepresented_count": sum(
                1 for item in findings if item.kind == "overrepresented"
            ),
        }
        payload = {
            "observed_report_id": observed_report.report_id,
            "target_report_id": target_report.report_id,
            "spec": observed_report.spec.to_dict(),
            "summary": summary,
            "gap_findings": [item.to_dict() for item in findings],
        }
        comparison_id = "dist-compare-" + _digest(payload)
        now = datetime.now(UTC)
        input_versions = _merge_versions(observed_report.table_versions, target_report.table_versions)
        transform_id = _record_distribution_transform(
            self._lake,
            operation="comparison",
            payload={**payload, "comparison_id": comparison_id},
            input_table_versions=[
                {"table": table, "version": version, "tag": ""}
                for table, version in input_versions
            ],
            output_tables=("distribution_catalog",),
            created_by=created_by,
        )
        comparison = DistributionComparison(
            comparison_id=comparison_id,
            observed=observed_report,
            target=target_report,
            gap_findings=findings,
            summary=summary,
            transform_id=transform_id,
            created_at=now,
        )
        _persist_distribution_catalog(
            self._lake,
            kind="comparison",
            name=observed_report.spec.name,
            spec_id=observed_report.spec.spec_id,
            report_id="",
            comparison_id=comparison_id,
            finding_id="",
            source={
                "kind": "distribution-comparison",
                "observed_report_id": observed_report.report_id,
                "target_report_id": target_report.report_id,
                "source_transform_ids": [
                    observed_report.transform_id,
                    target_report.transform_id,
                ],
            },
            summary=summary,
            body=comparison.to_dict(),
            table_versions=input_versions,
            transform_id=transform_id,
            created_by=created_by,
            created_at=now,
        )
        for index, finding in enumerate(findings):
            finding_id = f"{comparison_id}-finding-{index:04d}"
            _persist_distribution_catalog(
                self._lake,
                kind="gap-finding",
                name=observed_report.spec.name,
                spec_id=observed_report.spec.spec_id,
                report_id="",
                comparison_id=comparison_id,
                finding_id=finding_id,
                source={
                    "kind": "distribution-comparison",
                    "comparison_id": comparison_id,
                    "observed_report_id": observed_report.report_id,
                    "target_report_id": target_report.report_id,
                    "source_transform_ids": [
                        observed_report.transform_id,
                        target_report.transform_id,
                    ],
                },
                summary={
                    "kind": finding.kind,
                    "label": finding.label,
                    "needed_count": finding.needed_count,
                    "severity": finding.severity,
                },
                body=finding.to_dict(),
                table_versions=input_versions,
                transform_id=transform_id,
                created_by=created_by,
                created_at=now,
            )
        return comparison

    def list_catalog(
        self,
        *,
        kind: str | None = None,
        name: str | None = None,
        spec_id: str | None = None,
        report_id: str | None = None,
        comparison_id: str | None = None,
        source_kind: str | None = None,
        source_id: str | None = None,
        source_name: str | None = None,
        dataset_id: str | None = None,
        view_id: str | None = None,
        created_by: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        include_compacted: bool = True,
        limit: int | None = None,
    ) -> tuple[DistributionCatalogEntry, ...]:
        """List persisted distribution artifacts from the catalog table."""
        return _list_distribution_catalog(
            self._lake,
            kind=kind,
            name=name,
            spec_id=spec_id,
            report_id=report_id,
            comparison_id=comparison_id,
            source_kind=source_kind,
            source_id=source_id,
            source_name=source_name,
            dataset_id=dataset_id,
            view_id=view_id,
            created_by=created_by,
            since=since,
            until=until,
            include_compacted=include_compacted,
            limit=limit,
        )

    def list_reports(
        self,
        *,
        name: str | None = None,
        spec_id: str | None = None,
        source_kind: str | None = None,
        source_id: str | None = None,
        source_name: str | None = None,
        dataset_id: str | None = None,
        view_id: str | None = None,
        created_by: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        include_compacted: bool = True,
        limit: int | None = None,
    ) -> tuple[DistributionCatalogEntry, ...]:
        """List persisted distribution reports without scanning transform JSON."""
        return self.list_catalog(
            kind="report",
            name=name,
            spec_id=spec_id,
            source_kind=source_kind,
            source_id=source_id,
            source_name=source_name,
            dataset_id=dataset_id,
            view_id=view_id,
            created_by=created_by,
            since=since,
            until=until,
            include_compacted=include_compacted,
            limit=limit,
        )

    def get_report(self, report_id: str) -> DistributionReport:
        """Fetch a persisted report body by id."""
        matches = self.list_catalog(kind="report", report_id=report_id, limit=1)
        if not matches:
            raise DistributionError(f"no distribution report {report_id!r} in catalog")
        entry = matches[0]
        if entry.body is None:
            raise DistributionError(
                f"distribution report {report_id!r} body was compacted; "
                "catalog audit metadata is still available"
            )
        return ensure_distribution_report(entry.body, self._lake)

    def latest_report(
        self,
        *,
        name: str | None = None,
        spec_id: str | None = None,
        source_kind: str | None = None,
        source_id: str | None = None,
        source_name: str | None = None,
        dataset_id: str | None = None,
        view_id: str | None = None,
        created_by: str | None = None,
    ) -> DistributionReport:
        """Fetch the newest persisted report matching a name/spec/source filter."""
        matches = self.list_reports(
            name=name,
            spec_id=spec_id,
            source_kind=source_kind,
            source_id=source_id,
            source_name=source_name,
            dataset_id=dataset_id,
            view_id=view_id,
            created_by=created_by,
            include_compacted=False,
            limit=1,
        )
        if not matches:
            raise DistributionError("no non-compacted distribution report matched the filters")
        return self.get_report(matches[0].report_id)

    def get_comparison(self, comparison_id: str) -> dict[str, Any]:
        """Fetch a persisted distribution comparison body by id."""
        matches = self.list_catalog(kind="comparison", comparison_id=comparison_id, limit=1)
        if not matches:
            raise DistributionError(f"no distribution comparison {comparison_id!r} in catalog")
        entry = matches[0]
        if entry.body is None:
            raise DistributionError(
                f"distribution comparison {comparison_id!r} body was compacted; "
                "catalog audit metadata is still available"
            )
        return entry.body

    def compact_reports(
        self,
        *,
        older_than: datetime | timedelta | None = None,
        retain_latest_per_name: int = 1,
        kinds: Sequence[str] = ("report", "comparison"),
        dry_run: bool = False,
        created_by: str = "lancedb-robotics",
    ) -> DistributionRetentionReport:
        """Compact expired report/comparison bodies while keeping audit metadata."""
        return _compact_distribution_catalog(
            self._lake,
            older_than=older_than,
            retain_latest_per_name=retain_latest_per_name,
            kinds=kinds,
            dry_run=dry_run,
            created_by=created_by,
        )


def ensure_distribution_spec(spec: DistributionSpec | Mapping[str, Any]) -> DistributionSpec:
    """Coerce an SDK/dataclass/dict spec into ``DistributionSpec``."""
    if isinstance(spec, DistributionSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise DistributionError("distribution spec must be a DistributionSpec or dict")
    dimensions = tuple(dict.fromkeys(str(item) for item in spec.get("dimensions", ()) if str(item)))
    if not dimensions:
        raise DistributionError("distribution spec requires dimensions")
    slice_min_counts = tuple(
        sorted(
            (str(label), int(count))
            for label, count in (spec.get("slice_min_counts") or {}).items()
            if str(label)
        )
    )
    weights = tuple(
        sorted(
            (str(label), float(weight))
            for label, weight in (spec.get("weights") or {}).items()
            if str(label)
        )
    )
    payload = {
        "name": str(spec.get("name") or "distribution"),
        "dimensions": list(dimensions),
        "min_count_per_slice": int(spec.get("min_count_per_slice") or 0),
        "slice_min_counts": dict(slice_min_counts),
        "weights": dict(weights),
        "scope": _jsonable(dict(spec.get("scope") or {})),
    }
    return DistributionSpec(
        name=payload["name"],
        dimensions=dimensions,
        min_count_per_slice=payload["min_count_per_slice"],
        slice_min_counts=slice_min_counts,
        weights=weights,
        scope=payload["scope"],
        spec_id=str(spec.get("spec_id") or "dist-spec-" + _digest(payload)),
        transform_id=str(spec.get("transform_id") or ""),
    )


def ensure_distribution_execution_options(
    options: DistributionExecutionOptions | Mapping[str, Any] | None = None,
    *,
    batch_size: int | None = None,
    max_slice_count: int | None = None,
    overflow: str | None = None,
    rare_slice_min_count: int | None = None,
    top_k_overflow: int | None = None,
    max_scenario_ids_per_slice: int | None = None,
    slice_bins: Mapping[str, str] | None = None,
) -> DistributionExecutionOptions:
    """Normalize additive execution knobs for scalable distribution reports."""
    if isinstance(options, DistributionExecutionOptions):
        payload = options.to_dict()
    elif options is None:
        payload = {}
    elif isinstance(options, Mapping):
        payload = dict(options)
    else:
        raise DistributionError("execution must be DistributionExecutionOptions, dict, or None")

    overrides = {
        "batch_size": batch_size,
        "max_slice_count": max_slice_count,
        "overflow": overflow,
        "rare_slice_min_count": rare_slice_min_count,
        "top_k_overflow": top_k_overflow,
        "max_scenario_ids_per_slice": max_scenario_ids_per_slice,
        "slice_bins": slice_bins,
    }
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value

    normalized_overflow = str(payload.get("overflow") or "error").strip().lower()
    if normalized_overflow not in _CARDINALITY_ACTIONS:
        raise DistributionError(
            f"overflow must be one of {', '.join(_CARDINALITY_ACTIONS)}"
        )
    normalized_bins = payload.get("slice_bins") or payload.get("bins") or {}
    if isinstance(normalized_bins, Mapping):
        bins = tuple(
            sorted((str(raw), str(bucket)) for raw, bucket in normalized_bins.items() if str(raw))
        )
    else:
        bins = tuple(
            sorted(
                (str(item[0]), str(item[1]))
                for item in normalized_bins
                if len(item) >= 2 and str(item[0])
            )
        )

    parsed = DistributionExecutionOptions(
        batch_size=_positive_int(payload.get("batch_size"), "batch_size", _DEFAULT_BATCH_SIZE),
        filter_chunk_size=_positive_int(
            payload.get("filter_chunk_size"),
            "filter_chunk_size",
            _DEFAULT_FILTER_CHUNK_SIZE,
        ),
        max_slice_count=_optional_positive_int(payload.get("max_slice_count"), "max_slice_count"),
        overflow=normalized_overflow,
        rare_slice_min_count=_nonnegative_int(
            payload.get("rare_slice_min_count"),
            "rare_slice_min_count",
        ),
        rare_slice_label=str(payload.get("rare_slice_label") or _DEFAULT_RARE_SLICE_LABEL),
        overflow_slice_label=str(
            payload.get("overflow_slice_label") or _DEFAULT_OVERFLOW_SLICE_LABEL
        ),
        top_k_overflow=_nonnegative_int(payload.get("top_k_overflow"), "top_k_overflow", 10),
        max_scenario_ids_per_slice=_optional_nonnegative_int(
            payload.get("max_scenario_ids_per_slice"),
            "max_scenario_ids_per_slice",
        ),
        slice_bins=bins,
    )
    if parsed.max_slice_count is not None and parsed.max_slice_count < 1:
        raise DistributionError("max_slice_count must be positive")
    if parsed.overflow == "bucket" and parsed.max_slice_count == 1:
        raise DistributionError("bucket overflow requires max_slice_count greater than 1")
    return parsed


def ensure_distribution_report(
    report: DistributionReport | Mapping[str, Any],
    lake: Lake,
) -> DistributionReport:
    """Coerce a report dataclass or serialized report payload into a report."""
    if isinstance(report, DistributionReport):
        return report
    if not isinstance(report, Mapping):
        raise DistributionError("distribution report must be a DistributionReport or dict")
    spec = ensure_distribution_spec(report.get("spec") or {})
    slices = tuple(
        DistributionSlice(
            label=str(item["label"]),
            values=tuple(sorted((str(k), str(v)) for k, v in (item.get("values") or {}).items())),
            count=int(item.get("count") or 0),
            percentage=float(item.get("percentage") or 0.0),
            scenario_ids=tuple(str(sid) for sid in item.get("scenario_ids") or ()),
            quality_stats=dict(item.get("quality_stats") or {}),
            label_completeness=dict(item.get("label_completeness") or {}),
            duplicate_pressure=dict(item.get("duplicate_pressure") or {}),
        )
        for item in report.get("slices") or ()
    )
    table_versions = tuple(
        (str(item["table"]), int(item["version"]))
        for item in report.get("table_versions") or ()
    )
    created_at_raw = report.get("created_at")
    created_at = (
        datetime.fromisoformat(created_at_raw)
        if isinstance(created_at_raw, str) and created_at_raw
        else datetime.now(UTC)
    )
    return DistributionReport(
        lake=lake,
        report_id=str(report.get("report_id") or "dist-report-" + _digest(dict(report))),
        spec=spec,
        source=dict(report.get("source") or {}),
        slices=slices,
        total_count=int(report.get("total_count") or sum(item.count for item in slices)),
        table_versions=table_versions,
        transform_id=str(report.get("transform_id") or ""),
        created_at=created_at,
        execution=dict(report.get("execution") or {}),
    )


def gap_findings_from(value: Any) -> tuple[GapFinding, ...]:
    """Coerce a comparison, finding, dict, or sequence into gap findings."""
    if isinstance(value, DistributionComparison):
        return value.gap_findings
    if isinstance(value, GapFinding):
        return (value,)
    if isinstance(value, Mapping):
        if "gap_findings" in value:
            return gap_findings_from(value["gap_findings"])
        return (_gap_from_mapping(value),)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        findings: list[GapFinding] = []
        for item in value:
            findings.extend(gap_findings_from(item))
        return tuple(findings)
    raise DistributionError("expected a distribution comparison or gap findings")


def _gap_from_mapping(item: Mapping[str, Any]) -> GapFinding:
    kind = str(item.get("kind") or "").strip().lower().replace("_", "-")
    if kind not in _GAP_KINDS:
        raise DistributionError(f"unknown gap kind {kind!r}; expected one of {', '.join(_GAP_KINDS)}")
    values = item.get("values") or _label_values(str(item.get("label") or ""))
    return GapFinding(
        kind=kind,
        label=str(item.get("label") or _label_from_values(values)),
        values=tuple(sorted((str(k), str(v)) for k, v in dict(values).items())),
        observed_count=int(item.get("observed_count") or 0),
        target_count=int(item.get("target_count") or 0),
        needed_count=int(item.get("needed_count") or 0),
        observed_percentage=float(item.get("observed_percentage") or 0.0),
        target_percentage=float(item.get("target_percentage") or 0.0),
        delta_count=int(item.get("delta_count") or 0),
        delta_percentage=float(item.get("delta_percentage") or 0.0),
        severity=float(item.get("severity") or 0.0),
    )


def _persist_distribution_catalog(
    lake: Lake,
    *,
    kind: str,
    name: str,
    spec_id: str,
    report_id: str,
    comparison_id: str,
    finding_id: str,
    source: Mapping[str, Any],
    summary: Mapping[str, Any],
    body: Mapping[str, Any],
    table_versions: Sequence[tuple[str, int]] | Sequence[Mapping[str, Any]],
    transform_id: str,
    created_by: str,
    created_at: datetime | None = None,
) -> None:
    normalized_kind = _catalog_kind(kind)
    source_fields = _catalog_source_fields(source)
    body_json = json.dumps(_jsonable(body), sort_keys=True)
    body_bytes = len(body_json.encode())
    base_id = finding_id or comparison_id or report_id or spec_id or _digest(body)
    row = {
        "catalog_id": f"dist-catalog-{normalized_kind}-{base_id}",
        "kind": normalized_kind,
        "name": str(name),
        "spec_id": str(spec_id),
        "report_id": str(report_id),
        "comparison_id": str(comparison_id),
        "finding_id": str(finding_id),
        "summary_json": json.dumps(_jsonable(dict(summary)), sort_keys=True),
        "body_json": body_json,
        "body_sha1": hashlib.sha1(body_json.encode()).hexdigest(),
        "body_bytes": body_bytes,
        "body_compacted": False,
        "retention_policy_json": "",
        "expires_at": None,
        "compacted_at": None,
        "table_versions": _catalog_table_version_rows(table_versions),
        "source_transform_ids": source_fields.pop("source_transform_ids"),
        "created_by": created_by,
        "transform_id": transform_id,
        "created_at": created_at or datetime.now(UTC),
        **source_fields,
    }
    table = lake.table("distribution_catalog")
    table.delete(f"catalog_id = '{row['catalog_id']}'")
    table.add(pa.Table.from_pylist([row], schema=DISTRIBUTION_CATALOG_SCHEMA))


def _list_distribution_catalog(
    lake: Lake,
    *,
    kind: str | None,
    name: str | None,
    spec_id: str | None,
    report_id: str | None,
    comparison_id: str | None,
    source_kind: str | None,
    source_id: str | None,
    source_name: str | None,
    dataset_id: str | None,
    view_id: str | None,
    created_by: str | None,
    since: datetime | None,
    until: datetime | None,
    include_compacted: bool,
    limit: int | None,
) -> tuple[DistributionCatalogEntry, ...]:
    wanted_kind = _catalog_kind(kind) if kind else None
    rows = lake.table("distribution_catalog").to_arrow().to_pylist()
    entries = [_catalog_entry_from_row(row) for row in rows]
    filtered = [
        entry for entry in entries
        if (wanted_kind is None or entry.kind == wanted_kind)
        and _matches_optional(entry.name, name)
        and _matches_optional(entry.spec_id, spec_id)
        and _matches_optional(entry.report_id, report_id)
        and _matches_optional(entry.comparison_id, comparison_id)
        and _matches_optional(entry.source_kind, source_kind)
        and _matches_optional(entry.source_id, source_id)
        and _matches_optional(entry.source_name, source_name)
        and _matches_optional(entry.dataset_id, dataset_id)
        and _matches_optional(entry.view_id, view_id)
        and _matches_optional(entry.created_by, created_by)
        and (include_compacted or not entry.body_compacted)
        and (since is None or entry.created_at >= _as_utc(since))
        and (until is None or entry.created_at <= _as_utc(until))
    ]
    filtered.sort(key=_catalog_sort_key, reverse=True)
    if limit is not None:
        if limit < 0:
            raise DistributionError("catalog limit must be non-negative")
        filtered = filtered[:limit]
    return tuple(filtered)


def _catalog_entry_from_row(row: Mapping[str, Any]) -> DistributionCatalogEntry:
    body_json = str(row.get("body_json") or "")
    body_compacted = bool(row.get("body_compacted"))
    body = None if body_compacted or not body_json else _loads_json(body_json, {})
    return DistributionCatalogEntry(
        catalog_id=str(row.get("catalog_id") or ""),
        kind=_catalog_kind(row.get("kind") or "report"),
        name=str(row.get("name") or ""),
        spec_id=str(row.get("spec_id") or ""),
        report_id=str(row.get("report_id") or ""),
        comparison_id=str(row.get("comparison_id") or ""),
        finding_id=str(row.get("finding_id") or ""),
        source_kind=str(row.get("source_kind") or ""),
        source_name=str(row.get("source_name") or ""),
        source_id=str(row.get("source_id") or ""),
        dataset_id=str(row.get("dataset_id") or ""),
        view_id=str(row.get("view_id") or ""),
        source_digest=str(row.get("source_digest") or ""),
        summary=dict(_loads_json(row.get("summary_json"), {})),
        body=body,
        body_sha1=str(row.get("body_sha1") or ""),
        body_bytes=int(row.get("body_bytes") or 0),
        body_compacted=body_compacted,
        retention_policy=dict(_loads_json(row.get("retention_policy_json"), {})),
        expires_at=_as_optional_utc(row.get("expires_at")),
        compacted_at=_as_optional_utc(row.get("compacted_at")),
        table_versions=_catalog_table_versions(row.get("table_versions") or ()),
        source_transform_ids=tuple(
            str(item) for item in row.get("source_transform_ids") or () if str(item)
        ),
        created_by=str(row.get("created_by") or ""),
        transform_id=str(row.get("transform_id") or ""),
        created_at=_as_optional_utc(row.get("created_at")) or datetime.now(UTC),
    )


def _compact_distribution_catalog(
    lake: Lake,
    *,
    older_than: datetime | timedelta | None,
    retain_latest_per_name: int,
    kinds: Sequence[str],
    dry_run: bool,
    created_by: str,
) -> DistributionRetentionReport:
    if retain_latest_per_name < 0:
        raise DistributionError("retain_latest_per_name must be non-negative")
    normalized_kinds = {_catalog_kind(kind) for kind in kinds}
    if not normalized_kinds:
        raise DistributionError("at least one catalog kind must be selected")
    now = datetime.now(UTC)
    cutoff = _retention_cutoff(older_than, now)
    entries = _list_distribution_catalog(
        lake,
        kind=None,
        name=None,
        spec_id=None,
        report_id=None,
        comparison_id=None,
        source_kind=None,
        source_id=None,
        source_name=None,
        dataset_id=None,
        view_id=None,
        created_by=None,
        since=None,
        until=None,
        include_compacted=True,
        limit=None,
    )
    candidates_by_group: dict[tuple[str, ...], list[DistributionCatalogEntry]] = {}
    for entry in entries:
        if entry.kind not in normalized_kinds or entry.body_compacted:
            continue
        candidates_by_group.setdefault(_catalog_retention_group(entry), []).append(entry)
    retained: set[str] = set()
    for group_entries in candidates_by_group.values():
        group_entries.sort(key=_catalog_sort_key, reverse=True)
        retained.update(entry.catalog_id for entry in group_entries[:retain_latest_per_name])
    compacted = tuple(
        entry for group_entries in candidates_by_group.values()
        for entry in group_entries
        if entry.catalog_id not in retained
        and (cutoff is None or entry.created_at < cutoff)
        and entry.body is not None
    )
    body_bytes_before = sum(
        entry.body_bytes
        for group_entries in candidates_by_group.values()
        for entry in group_entries
        if entry.body is not None
    )
    compacted_bytes = sum(entry.body_bytes for entry in compacted)
    policy = {
        "older_than": cutoff.isoformat() if cutoff else None,
        "retain_latest_per_name": retain_latest_per_name,
        "kinds": sorted(normalized_kinds),
        "compacted_at": now.isoformat(),
    }
    transform_id = ""
    if compacted and not dry_run:
        rows_by_id = {
            str(row.get("catalog_id") or ""): row
            for row in lake.table("distribution_catalog").to_arrow().to_pylist()
        }
        updated_rows = []
        for entry in compacted:
            row = dict(rows_by_id[entry.catalog_id])
            row["body_json"] = ""
            row["body_compacted"] = True
            row["retention_policy_json"] = json.dumps(policy, sort_keys=True)
            row["compacted_at"] = now
            updated_rows.append(row)
        table = lake.table("distribution_catalog")
        for row in updated_rows:
            table.delete(f"catalog_id = '{row['catalog_id']}'")
        table.add(pa.Table.from_pylist(updated_rows, schema=DISTRIBUTION_CATALOG_SCHEMA))
        transform_id = _record_distribution_transform(
            lake,
            operation="catalog-retention",
            payload={
                "compacted_catalog_ids": [entry.catalog_id for entry in compacted],
                "retained_catalog_ids": sorted(retained),
                "body_bytes_before": body_bytes_before,
                "body_bytes_after": body_bytes_before - compacted_bytes,
                "policy": policy,
            },
            input_table_versions=_table_versions(lake, tables=("distribution_catalog",)),
            output_tables=("distribution_catalog",),
            created_by=created_by,
        )
    return DistributionRetentionReport(
        compacted_catalog_ids=tuple(entry.catalog_id for entry in compacted),
        retained_catalog_ids=tuple(sorted(retained)),
        dry_run=dry_run,
        body_bytes_before=body_bytes_before,
        body_bytes_after=body_bytes_before - compacted_bytes,
        policy=policy,
        transform_id=transform_id,
        created_at=now,
    )


def _catalog_source_fields(source: Mapping[str, Any]) -> dict[str, Any]:
    source_kind = str(source.get("kind") or "").strip().lower().replace("_", "-")
    source_name = str(source.get("name") or source.get("operation") or "")
    dataset_id = str(source.get("dataset_id") or "")
    view_id = str(source.get("view_id") or "")
    source_id = str(
        source.get("source_id")
        or dataset_id
        or view_id
        or source.get("comparison_id")
        or source.get("observed_report_id")
        or ""
    )
    transform_ids = []
    for key in ("transform_id",):
        if source.get(key):
            transform_ids.append(str(source[key]))
    for key in ("operation_transform_ids", "source_transform_ids"):
        transform_ids.extend(str(item) for item in source.get(key) or () if str(item))
    return {
        "source_kind": source_kind,
        "source_name": source_name,
        "source_id": source_id,
        "dataset_id": dataset_id,
        "view_id": view_id,
        "source_digest": _digest(source) if source else "",
        "source_transform_ids": sorted(dict.fromkeys(transform_ids)),
    }


def _catalog_table_versions(
    table_versions: Sequence[tuple[str, int]] | Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, int], ...]:
    pairs: list[tuple[str, int]] = []
    for item in table_versions:
        if isinstance(item, Mapping):
            pairs.append((str(item.get("table") or ""), int(item.get("version") or 0)))
            continue
        table, version = item[:2]
        pairs.append((str(table), int(version)))
    return tuple((table, version) for table, version in pairs if table)


def _catalog_table_version_rows(
    table_versions: Sequence[tuple[str, int]] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"table": table, "version": version, "tag": ""}
        for table, version in _catalog_table_versions(table_versions)
    ]


def _catalog_kind(kind: Any) -> str:
    normalized = str(kind or "").strip().lower().replace("_", "-")
    if normalized not in {"spec", "report", "comparison", "gap-finding"}:
        raise DistributionError(
            "distribution catalog kind must be one of spec, report, comparison, gap-finding"
        )
    return normalized


def _catalog_sort_key(entry: DistributionCatalogEntry) -> tuple[datetime, str]:
    return (
        entry.created_at,
        entry.report_id or entry.comparison_id or entry.finding_id or entry.catalog_id,
    )


def _catalog_retention_group(entry: DistributionCatalogEntry) -> tuple[str, ...]:
    return (
        entry.kind,
        entry.name,
        entry.spec_id,
        entry.source_kind,
        entry.source_name,
        entry.source_id,
        entry.dataset_id,
        entry.view_id,
    )


def _retention_cutoff(value: datetime | timedelta | None, now: datetime) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, timedelta):
        return now - value
    return _as_utc(value)


def _matches_optional(actual: str, expected: str | None) -> bool:
    return expected is None or actual == expected


def _loads_json(raw: Any, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _as_optional_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, str) and value:
        normalized = value.replace("Z", "+00:00")
        try:
            return _as_utc(datetime.fromisoformat(normalized))
        except ValueError:
            return None
    return None


def _resolve_source(
    lake: Lake,
    source: Any,
    spec: DistributionSpec,
) -> dict[str, Any]:
    if source is None:
        return _source_from_scope(lake, spec.scope or None, kind="scope")
    if hasattr(source, "scenario_ids") and hasattr(source, "operation"):
        return {
            "kind": "curation-selection",
            "operation": str(getattr(source, "operation", "")),
            "scenario_ids": [str(item) for item in source.scenario_ids],
            "operation_transform_ids": list(getattr(source, "operation_transform_ids", ())),
            "report": _jsonable(getattr(source, "report", {})),
            "table_versions": _table_versions(lake),
        }
    if isinstance(source, SnapshotManifest):
        return {
            "kind": "dataset-snapshot",
            "name": source.name,
            "dataset_id": source.dataset_id,
            "scenario_ids": list(source.scenario_ids),
            "table_versions": [
                {"table": table, "version": version, "tag": ""}
                for table, version in source.table_versions
            ],
        }
    if isinstance(source, str):
        snapshot = _latest_snapshot_row(lake, source)
        if snapshot is not None:
            return _source_from_snapshot_row(snapshot)
        view = _latest_view_row(lake, source)
        if view is not None:
            return _source_from_view_row(view)
        raise DistributionError(f"no dataset snapshot or curation view named {source!r}")
    if isinstance(source, Mapping):
        kind = str(source.get("kind") or "").strip().lower().replace("_", "-")
        if not kind:
            return _source_from_scope(lake, source, kind="scope")
        if kind in {"snapshot", "dataset-snapshot"}:
            name = str(source.get("name") or source.get("snapshot") or source.get("snapshot_name") or "")
            row = _latest_snapshot_row(lake, name)
            if row is None:
                raise DistributionError(f"no dataset snapshot named {name!r}")
            return _source_from_snapshot_row(row)
        if kind in {"view", "curation-view", "saved-view"}:
            name = str(source.get("name") or source.get("view") or source.get("view_name") or "")
            row = _latest_view_row(lake, name)
            if row is None:
                raise DistributionError(f"no curation view named {name!r}")
            return _source_from_view_row(row)
        if kind in {"scenarios", "scenario-ids"}:
            scenario_ids = source.get("scenario_ids") or source.get("scenario_id") or ()
            return {
                "kind": "scenario-ids",
                "scenario_ids": [str(item) for item in _as_tuple(scenario_ids)],
                "table_versions": _table_versions(lake),
            }
        if kind in {"scope", "scenario-scope"}:
            scope = source.get("scope") or {
                key: value for key, value in source.items() if key != "kind"
            }
            return _source_from_scope(lake, scope, kind=kind)
        if kind in {"episode-scope", "episodes"}:
            return _source_from_episode_scope(lake, source)
        if kind in {"feedback-window", "feedback"}:
            return _source_from_row_window(lake, "feedback", source)
        if kind in {"model-output-window", "model-outputs"}:
            return _source_from_row_window(lake, "model_outputs", source)
        if kind in {"external-manifest", "deployment-manifest", "manifest"}:
            return _source_from_external_manifest(source, spec)
        raise DistributionError(f"unknown distribution source kind {kind!r}")
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return {
            "kind": "scenario-ids",
            "scenario_ids": [str(item) for item in source],
            "table_versions": _table_versions(lake),
        }
    raise DistributionError("unsupported distribution source")


def _source_from_scope(lake: Lake, scope: Any, *, kind: str) -> dict[str, Any]:
    from lancedb_robotics.curate import CurationError

    payload = _scope_payload(scope)
    if not payload:
        return {
            "kind": kind,
            "scope": {},
            "scan": "all-scenarios",
            "table_versions": _table_versions(lake),
        }
    try:
        selection = lake.curate.workbench(scope=scope)
    except CurationError as exc:
        raise DistributionError(str(exc)) from exc
    return {
        "kind": kind,
        "scope": payload,
        "scenario_ids": list(selection.scenario_ids),
        "operation_transform_ids": list(selection.operation_transform_ids),
        "table_versions": _table_versions(lake),
    }


def _source_from_snapshot_row(row: dict[str, Any]) -> dict[str, Any]:
    query_spec = json.loads(row["query_spec"] or "{}")
    return {
        "kind": "dataset-snapshot",
        "name": str(row["name"]),
        "dataset_id": str(row["dataset_id"]),
        "scenario_ids": [str(item) for item in query_spec.get("scenario_ids") or ()],
        "query_spec": query_spec,
        "table_versions": row.get("table_versions") or (),
        "transform_id": str(row.get("transform_id") or ""),
    }


def _source_from_view_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "curation-view",
        "name": str(row["name"]),
        "view_id": str(row["view_id"]),
        "scenario_ids": [str(item) for item in row.get("scenario_ids") or ()],
        "table_versions": row.get("table_versions") or (),
        "transform_id": str(row.get("transform_id") or ""),
    }


def _source_from_episode_scope(lake: Lake, source: Mapping[str, Any]) -> dict[str, Any]:
    wanted = {str(item) for item in _as_tuple(source.get("episode_ids") or source.get("episode_id"))}
    episodes = [
        row for row in lake.table("episodes").to_arrow().to_pylist()
        if not wanted or str(row.get("episode_id") or "") in wanted
    ]
    scenario_ids = _scenario_ids_for_windows(
        lake,
        (
            (
                str(row.get("run_id") or ""),
                int(row.get("from_timestamp_ns") or 0),
                int(row.get("to_timestamp_ns") or 0),
            )
            for row in episodes
        ),
    )
    return {
        "kind": "episode-scope",
        "episode_ids": [str(row.get("episode_id") or "") for row in episodes],
        "scenario_ids": scenario_ids,
        "table_versions": _table_versions(lake),
    }


def _source_from_row_window(
    lake: Lake,
    table_name: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _filtered_rows(lake.table(table_name).to_arrow().to_pylist(), source)
    scenario_ids = _scenario_ids_for_references(lake, rows)
    return {
        "kind": table_name.replace("_", "-"),
        "filters": {
            str(key): _jsonable(value)
            for key, value in source.items()
            if key not in {"kind", "scope"}
        },
        "row_count": len(rows),
        "scenario_ids": scenario_ids,
        "table_versions": _table_versions(lake),
    }


def _source_from_external_manifest(
    source: Mapping[str, Any],
    spec: DistributionSpec,
) -> dict[str, Any]:
    raw_slices = source.get("slices") or source.get("slice_counts") or {}
    if isinstance(raw_slices, Mapping):
        slices = {
            str(label): _external_count(value)
            for label, value in raw_slices.items()
        }
    else:
        slices = {
            str(item.get("label") or _label_from_values(item.get("values") or {})): int(
                item.get("count") or 0
            )
            for item in raw_slices
        }
    dimensions = tuple(str(item) for item in source.get("dimensions") or spec.dimensions)
    if dimensions != spec.dimensions:
        raise DistributionError("external manifest dimensions must match the distribution spec")
    total = int(source.get("total_count") or sum(slices.values()))
    return {
        "kind": "external-manifest",
        "external": True,
        "name": str(source.get("name") or "external"),
        "dimensions": list(dimensions),
        "slice_counts": dict(sorted(slices.items())),
        "total_count": total,
        "table_versions": _table_versions_from_source(source),
    }


def _external_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return int(value.get("count") or 0)
    return int(value or 0)


def _external_slices(
    source: Mapping[str, Any],
    spec: DistributionSpec,
) -> tuple[DistributionSlice, ...]:
    counts = {
        str(label): int(count)
        for label, count in (source.get("slice_counts") or {}).items()
    }
    total = int(source.get("total_count") or sum(counts.values()))
    slices = []
    for label, count in sorted(counts.items()):
        slices.append(
            DistributionSlice(
                label=label,
                values=tuple(sorted(_label_values(label).items())),
                count=count,
                percentage=count / total if total else 0.0,
                quality_stats={},
                label_completeness={},
                duplicate_pressure={},
            )
        )
    for label, _ in spec.slice_min_counts:
        if label not in counts:
            slices.append(
                DistributionSlice(
                    label=label,
                    values=tuple(sorted(_label_values(label).items())),
                    count=0,
                    percentage=0.0,
                    quality_stats={},
                    label_completeness={},
                    duplicate_pressure={},
                )
            )
    return tuple(sorted(slices, key=lambda item: item.label))


@dataclass
class _SliceAccumulator:
    label: str
    count: int = 0
    scenario_refs: list[tuple[int, str]] = field(default_factory=list)
    omitted_scenario_ids: int = 0
    quality_score_count: int = 0
    quality_score_sum: float = 0.0
    quality_score_min: float | None = None
    quality_score_max: float | None = None
    run_quality_flagged_count: int = 0
    labeled_count: int = 0
    duplicate_decision_count: int = 0

    def add_scenario(
        self,
        row: Mapping[str, Any],
        run: Mapping[str, Any],
        *,
        scenario_id_cap: int | None,
    ) -> None:
        self.count += 1
        scenario_id = str(row["scenario_id"])
        if scenario_id_cap is None or len(self.scenario_refs) < scenario_id_cap:
            self.scenario_refs.append((int(row.get("start_time_ns") or 0), scenario_id))
        else:
            self.omitted_scenario_ids += 1
        if run.get("quality_flags"):
            self.run_quality_flagged_count += 1
        score = _numeric_value(dict(row), "quality_score")
        if score is not None:
            self.quality_score_count += 1
            self.quality_score_sum += score
            self.quality_score_min = (
                score if self.quality_score_min is None else min(self.quality_score_min, score)
            )
            self.quality_score_max = (
                score if self.quality_score_max is None else max(self.quality_score_max, score)
            )

    def merge(self, other: "_SliceAccumulator", *, scenario_id_cap: int | None) -> None:
        self.count += other.count
        self.quality_score_count += other.quality_score_count
        self.quality_score_sum += other.quality_score_sum
        if other.quality_score_min is not None:
            self.quality_score_min = (
                other.quality_score_min
                if self.quality_score_min is None
                else min(self.quality_score_min, other.quality_score_min)
            )
        if other.quality_score_max is not None:
            self.quality_score_max = (
                other.quality_score_max
                if self.quality_score_max is None
                else max(self.quality_score_max, other.quality_score_max)
            )
        self.run_quality_flagged_count += other.run_quality_flagged_count
        self.labeled_count += other.labeled_count
        self.duplicate_decision_count += other.duplicate_decision_count
        merged_refs = sorted(self.scenario_refs + other.scenario_refs)
        if scenario_id_cap is None:
            self.scenario_refs = merged_refs
            self.omitted_scenario_ids += other.omitted_scenario_ids
            return
        self.scenario_refs = merged_refs[:scenario_id_cap]
        omitted_refs = max(0, len(merged_refs) - scenario_id_cap)
        self.omitted_scenario_ids += other.omitted_scenario_ids + omitted_refs

    def to_slice(self, *, total_count: int) -> DistributionSlice:
        scenario_ids = tuple(scenario_id for _, scenario_id in sorted(self.scenario_refs))
        quality_mean = (
            self.quality_score_sum / self.quality_score_count
            if self.quality_score_count
            else None
        )
        return DistributionSlice(
            label=self.label,
            values=tuple(sorted(_label_values(self.label).items())),
            count=self.count,
            percentage=self.count / total_count if total_count else 0.0,
            scenario_ids=scenario_ids,
            quality_stats={
                "quality_score_count": self.quality_score_count,
                "quality_score_min": self.quality_score_min,
                "quality_score_max": self.quality_score_max,
                "quality_score_mean": quality_mean,
                "run_quality_flagged_count": self.run_quality_flagged_count,
                "scenario_ids_omitted": self.omitted_scenario_ids,
            },
            label_completeness={
                "labeled_count": self.labeled_count,
                "unlabeled_count": self.count - self.labeled_count,
                "completeness": self.labeled_count / self.count if self.count else 0.0,
            },
            duplicate_pressure={
                "duplicate_decision_count": self.duplicate_decision_count,
                "duplicate_pressure": (
                    self.duplicate_decision_count / self.count if self.count else 0.0
                ),
            },
        )


def _measure_distribution_slices(
    lake: Lake,
    source: Mapping[str, Any],
    dimensions: Sequence[str],
    options: DistributionExecutionOptions,
) -> tuple[tuple[DistributionSlice, ...], int, dict[str, Any]]:
    version_map = _source_version_map(source)
    run_rows = _run_rows(lake, version=version_map.get("runs"), dimensions=dimensions)
    slice_bins = dict(options.slice_bins)
    accumulators: dict[str, _SliceAccumulator] = {}
    scenario_to_slice: dict[str, str] = {}
    observation_to_scenario: dict[str, str] = {}
    wanted_ids = (
        tuple(dict.fromkeys(str(item) for item in source.get("scenario_ids") or ()))
        if "scenario_ids" in source
        else None
    )
    scan_stats = {
        "strategy": "lance-streaming",
        "scenario_rows": 0,
        "scenario_batches": 0,
        "run_rows": len(run_rows),
        "labels_rows": 0,
        "curation_membership_rows": 0,
        "batch_size": options.batch_size,
    }

    found_ids: set[str] = set()
    for batch_rows in _scan_selected_scenario_batches(
        lake,
        wanted_ids=wanted_ids,
        dimensions=dimensions,
        options=options,
        version=version_map.get("scenarios"),
    ):
        scan_stats["scenario_batches"] += 1
        scan_stats["scenario_rows"] += len(batch_rows)
        for row in batch_rows:
            scenario_id = str(row.get("scenario_id") or "")
            if not scenario_id:
                continue
            found_ids.add(scenario_id)
            run = run_rows.get(str(row.get("run_id") or ""), {})
            raw_label = _slice_label(row, run, dimensions)
            label = slice_bins.get(raw_label, raw_label)
            accumulator = accumulators.setdefault(label, _SliceAccumulator(label=label))
            accumulator.add_scenario(
                row,
                run,
                scenario_id_cap=options.max_scenario_ids_per_slice,
            )
            scenario_to_slice[scenario_id] = label
            for observation_id in row.get("observation_ids") or ():
                if observation_id:
                    observation_to_scenario[str(observation_id)] = scenario_id

    if wanted_ids is not None and len(found_ids) != len(wanted_ids):
        missing = sorted(set(wanted_ids) - found_ids)
        raise DistributionError(f"selected scenarios are missing from the lake: {missing}")

    label_stats = _add_label_completeness(
        lake,
        accumulators,
        scenario_to_slice=scenario_to_slice,
        observation_to_scenario=observation_to_scenario,
        options=options,
        version=version_map.get("labels"),
        all_source=wanted_ids is None,
    )
    duplicate_stats = _add_duplicate_pressure(
        lake,
        accumulators,
        scenario_to_slice=scenario_to_slice,
        options=options,
        version=version_map.get("curation_memberships"),
        all_source=wanted_ids is None,
    )
    scan_stats["labels_rows"] = label_stats["rows"]
    scan_stats["curation_membership_rows"] = duplicate_stats["rows"]

    accumulators, cardinality = _apply_cardinality_controls(accumulators, options)
    total_count = sum(accumulator.count for accumulator in accumulators.values())
    slices = tuple(
        accumulator.to_slice(total_count=total_count)
        for accumulator in sorted(accumulators.values(), key=lambda item: item.label)
    )
    execution = {
        "strategy": scan_stats["strategy"],
        "options": options.to_dict(),
        "scan": scan_stats,
        "related_scan": {
            "labels": label_stats,
            "curation_memberships": duplicate_stats,
        },
        "cardinality": cardinality,
        "memory_bound": {
            "tracked_scenario_ids": len(scenario_to_slice),
            "tracked_observation_ids": len(observation_to_scenario),
            "stored_scenario_ids_per_slice": options.max_scenario_ids_per_slice,
            "slice_accumulators": len(accumulators),
        },
    }
    return slices, total_count, execution


def _measured_slices(
    lake: Lake,
    rows: Sequence[dict[str, Any]],
    run_rows: Mapping[str, dict[str, Any]],
    dimensions: Sequence[str],
) -> tuple[DistributionSlice, ...]:
    total = len(rows)
    labels_by_scenario = _label_counts(lake)
    duplicate_decisions = _duplicate_decision_counts(lake)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = _slice_label(row, run_rows.get(str(row.get("run_id") or ""), {}), dimensions)
        by_label.setdefault(label, []).append(row)

    slices: list[DistributionSlice] = []
    for label in sorted(by_label):
        members = sorted(by_label[label], key=_scenario_sort_key)
        scenario_ids = tuple(str(row["scenario_id"]) for row in members)
        quality_scores = [
            score for row in members
            if (score := _numeric_value(row, "quality_score")) is not None
        ]
        run_flagged = sum(
            1
            for row in members
            if run_rows.get(str(row.get("run_id") or ""), {}).get("quality_flags")
        )
        labeled_count = sum(1 for sid in scenario_ids if labels_by_scenario.get(sid, 0) > 0)
        duplicate_count = sum(duplicate_decisions.get(sid, 0) for sid in scenario_ids)
        count = len(members)
        slices.append(
            DistributionSlice(
                label=label,
                values=tuple(sorted(_label_values(label).items())),
                count=count,
                percentage=count / total if total else 0.0,
                scenario_ids=scenario_ids,
                quality_stats={
                    "quality_score_count": len(quality_scores),
                    "quality_score_min": min(quality_scores) if quality_scores else None,
                    "quality_score_max": max(quality_scores) if quality_scores else None,
                    "quality_score_mean": (
                        sum(quality_scores) / len(quality_scores) if quality_scores else None
                    ),
                    "run_quality_flagged_count": run_flagged,
                },
                label_completeness={
                    "labeled_count": labeled_count,
                    "unlabeled_count": count - labeled_count,
                    "completeness": labeled_count / count if count else 0.0,
                },
                duplicate_pressure={
                    "duplicate_decision_count": duplicate_count,
                    "duplicate_pressure": duplicate_count / count if count else 0.0,
                },
            )
        )
    return tuple(slices)


def _gap_findings(
    observed: DistributionReport,
    target: DistributionReport,
    *,
    overrepresented_threshold: float,
) -> tuple[GapFinding, ...]:
    observed_by_label = {item.label: item for item in observed.slices}
    target_by_label = {item.label: item for item in target.slices}
    labels = sorted(
        set(observed_by_label)
        | set(target_by_label)
        | {label for label, _ in observed.spec.slice_min_counts}
    )
    findings: list[GapFinding] = []
    for label in labels:
        observed_slice = observed_by_label.get(label)
        target_slice = target_by_label.get(label)
        observed_count = observed_slice.count if observed_slice else 0
        target_count = target_slice.count if target_slice else 0
        observed_pct = observed_slice.percentage if observed_slice else 0.0
        target_pct = target_slice.percentage if target_slice else 0.0
        min_count = observed.spec.min_count_for(label)
        expected_from_target = (
            math.ceil(target_pct * observed.total_count)
            if observed.total_count and target.total_count
            else target_count
        )
        required_count = max(min_count, expected_from_target)
        needed = max(0, required_count - observed_count)
        values = (
            observed_slice.values
            if observed_slice
            else target_slice.values
            if target_slice
            else tuple(sorted(_label_values(label).items()))
        )
        if needed:
            kind = "missing" if observed_count == 0 else "underrepresented"
            findings.append(
                GapFinding(
                    kind=kind,
                    label=label,
                    values=values,
                    observed_count=observed_count,
                    target_count=target_count,
                    needed_count=needed,
                    observed_percentage=observed_pct,
                    target_percentage=target_pct,
                    delta_count=observed_count - target_count,
                    delta_percentage=observed_pct - target_pct,
                    severity=max(float(needed), target_pct - observed_pct),
                )
            )
            continue
        if observed_count <= 0:
            continue
        over = False
        if target_count == 0 and target.total_count:
            over = True
        elif target.total_count and observed_pct > target_pct + overrepresented_threshold:
            over = True
        if over:
            findings.append(
                GapFinding(
                    kind="overrepresented",
                    label=label,
                    values=values,
                    observed_count=observed_count,
                    target_count=target_count,
                    needed_count=0,
                    observed_percentage=observed_pct,
                    target_percentage=target_pct,
                    delta_count=observed_count - target_count,
                    delta_percentage=observed_pct - target_pct,
                    severity=observed_pct - target_pct,
                )
            )
    rank = {"missing": 0, "underrepresented": 1, "overrepresented": 2}
    return tuple(
        sorted(findings, key=lambda item: (rank[item.kind], -item.severity, item.label))
    )


def _scan_selected_scenario_batches(
    lake: Lake,
    *,
    wanted_ids: Sequence[str] | None,
    dimensions: Sequence[str],
    options: DistributionExecutionOptions,
    version: int | None,
) -> Iterable[list[dict[str, Any]]]:
    columns = _scenario_scan_columns(lake, dimensions, version=version)
    if wanted_ids is None:
        yield from _scan_table_batches(
            lake,
            "scenarios",
            columns=columns,
            where_sql=None,
            version=version,
            batch_size=options.batch_size,
        )
        return
    if not wanted_ids:
        return
    for chunk in _chunks(wanted_ids, options.filter_chunk_size):
        yield from _scan_table_batches(
            lake,
            "scenarios",
            columns=columns,
            where_sql=_sql_in("scenario_id", chunk),
            version=version,
            batch_size=options.batch_size,
        )


def _add_label_completeness(
    lake: Lake,
    accumulators: Mapping[str, _SliceAccumulator],
    *,
    scenario_to_slice: Mapping[str, str],
    observation_to_scenario: Mapping[str, str],
    options: DistributionExecutionOptions,
    version: int | None,
    all_source: bool,
) -> dict[str, Any]:
    if not accumulators:
        return {"rows": 0, "matched_scenarios": 0, "filter": "none"}
    columns = _table_columns(
        lake,
        "labels",
        ("label_id", "scenario_id", "observation_id"),
        version=version,
    )
    seen_label_ids: set[str] = set()
    labeled_scenarios: set[str] = set()
    rows = 0

    def consume(row: Mapping[str, Any]) -> None:
        nonlocal rows
        label_id = str(row.get("label_id") or "")
        if label_id and label_id in seen_label_ids:
            return
        if label_id:
            seen_label_ids.add(label_id)
        rows += 1
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            scenario_id = observation_to_scenario.get(str(row.get("observation_id") or ""), "")
        if scenario_id and scenario_id in scenario_to_slice:
            labeled_scenarios.add(scenario_id)

    if all_source:
        for batch in _scan_table_batches(
            lake,
            "labels",
            columns=columns,
            where_sql=None,
            version=version,
            batch_size=options.batch_size,
        ):
            for row in batch:
                consume(row)
    else:
        for scenario_chunk in _chunks(tuple(scenario_to_slice), options.filter_chunk_size):
            for batch in _scan_table_batches(
                lake,
                "labels",
                columns=columns,
                where_sql=_sql_in("scenario_id", scenario_chunk),
                version=version,
                batch_size=options.batch_size,
            ):
                for row in batch:
                    consume(row)
        observation_ids = tuple(observation_to_scenario)
        for observation_chunk in _chunks(observation_ids, options.filter_chunk_size):
            for batch in _scan_table_batches(
                lake,
                "labels",
                columns=columns,
                where_sql=_sql_in("observation_id", observation_chunk),
                version=version,
                batch_size=options.batch_size,
            ):
                for row in batch:
                    consume(row)

    for scenario_id in labeled_scenarios:
        label = scenario_to_slice.get(scenario_id)
        if label in accumulators:
            accumulators[label].labeled_count += 1
    return {
        "rows": rows,
        "matched_scenarios": len(labeled_scenarios),
        "filter": "all-labels" if all_source else "selected-scenario-or-observation-ids",
    }


def _add_duplicate_pressure(
    lake: Lake,
    accumulators: Mapping[str, _SliceAccumulator],
    *,
    scenario_to_slice: Mapping[str, str],
    options: DistributionExecutionOptions,
    version: int | None,
    all_source: bool,
) -> dict[str, Any]:
    if not accumulators:
        return {"rows": 0, "matched_scenarios": 0, "filter": "none"}
    columns = _table_columns(
        lake,
        "curation_memberships",
        ("membership_id", "scenario_id", "target_id", "source", "decision"),
        version=version,
    )
    seen_membership_ids: set[str] = set()
    rows = 0
    matched = 0

    def consume(row: Mapping[str, Any]) -> None:
        nonlocal rows, matched
        membership_id = str(row.get("membership_id") or "")
        if membership_id and membership_id in seen_membership_ids:
            return
        if membership_id:
            seen_membership_ids.add(membership_id)
        rows += 1
        if row.get("source") != "dedup" or row.get("decision") != "exclude":
            return
        scenario_id = str(row.get("scenario_id") or row.get("target_id") or "")
        label = scenario_to_slice.get(scenario_id)
        if label in accumulators:
            accumulators[label].duplicate_decision_count += 1
            matched += 1

    if all_source:
        predicate = "source = 'dedup' AND decision = 'exclude'"
        for batch in _scan_table_batches(
            lake,
            "curation_memberships",
            columns=columns,
            where_sql=predicate,
            version=version,
            batch_size=options.batch_size,
        ):
            for row in batch:
                consume(row)
    else:
        for scenario_chunk in _chunks(tuple(scenario_to_slice), options.filter_chunk_size):
            predicates = (
                f"source = 'dedup' AND decision = 'exclude' AND {_sql_in('scenario_id', scenario_chunk)}",
                f"source = 'dedup' AND decision = 'exclude' AND {_sql_in('target_id', scenario_chunk)}",
            )
            for predicate in predicates:
                for batch in _scan_table_batches(
                    lake,
                    "curation_memberships",
                    columns=columns,
                    where_sql=predicate,
                    version=version,
                    batch_size=options.batch_size,
                ):
                    for row in batch:
                        consume(row)

    return {
        "rows": rows,
        "matched_scenarios": matched,
        "filter": "dedup-exclude" if all_source else "dedup-exclude-selected-scenarios",
    }


def _apply_cardinality_controls(
    accumulators: Mapping[str, _SliceAccumulator],
    options: DistributionExecutionOptions,
) -> tuple[dict[str, _SliceAccumulator], dict[str, Any]]:
    cardinality: dict[str, Any] = {
        "raw_slice_count": len(accumulators),
        "slice_count_before_overflow": len(accumulators),
        "slice_count": len(accumulators),
        "actions": [],
    }
    controlled = dict(accumulators)
    if options.rare_slice_min_count > 0:
        rare = [
            accumulator
            for accumulator in controlled.values()
            if accumulator.count < options.rare_slice_min_count
            and accumulator.label != options.rare_slice_label
        ]
        if rare:
            controlled = {
                label: accumulator
                for label, accumulator in controlled.items()
                if accumulator not in rare
            }
            rare_accumulator = controlled.setdefault(
                options.rare_slice_label,
                _SliceAccumulator(label=options.rare_slice_label),
            )
            for accumulator in sorted(rare, key=lambda item: item.label):
                rare_accumulator.merge(
                    accumulator,
                    scenario_id_cap=options.max_scenario_ids_per_slice,
                )
            cardinality["actions"].append(
                {
                    "action": "bucket-rare",
                    "reason": (
                        f"{len(rare)} slices had count below "
                        f"rare_slice_min_count={options.rare_slice_min_count}"
                    ),
                    "bucket_label": options.rare_slice_label,
                    "bucketed_slice_count": len(rare),
                    "bucketed_total_count": sum(item.count for item in rare),
                    "examples": [
                        {"label": item.label, "count": item.count}
                        for item in sorted(rare, key=lambda item: (-item.count, item.label))[
                            : options.top_k_overflow
                        ]
                    ],
                }
            )
    cardinality["slice_count_before_overflow"] = len(controlled)
    if options.max_slice_count is not None and len(controlled) > options.max_slice_count:
        reason = (
            f"distribution produced {len(controlled)} slices, exceeding "
            f"max_slice_count={options.max_slice_count}"
        )
        if options.overflow == "error":
            raise DistributionError(
                reason + "; set overflow='bucket' or add explicit slice_bins"
            )
        keep_count = options.max_slice_count - 1
        ranked = sorted(controlled.values(), key=lambda item: (-item.count, item.label))
        keep = ranked[:keep_count]
        overflow = ranked[keep_count:]
        controlled = {item.label: item for item in keep}
        overflow_accumulator = controlled.setdefault(
            options.overflow_slice_label,
            _SliceAccumulator(label=options.overflow_slice_label),
        )
        for accumulator in overflow:
            overflow_accumulator.merge(
                accumulator,
                scenario_id_cap=options.max_scenario_ids_per_slice,
            )
        cardinality["actions"].append(
            {
                "action": "bucket-overflow",
                "reason": reason,
                "bucket_label": options.overflow_slice_label,
                "bucketed_slice_count": len(overflow),
                "bucketed_total_count": sum(item.count for item in overflow),
                "top_overflow_slices": [
                    {"label": item.label, "count": item.count}
                    for item in overflow[: options.top_k_overflow]
                ],
            }
        )
    cardinality["slice_count"] = len(controlled)
    cardinality["reason"] = "; ".join(
        str(action["reason"]) for action in cardinality["actions"]
    )
    return controlled, cardinality


def _filtered_rows(
    rows: Sequence[dict[str, Any]],
    source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    filters = {
        str(key): value
        for key, value in source.items()
        if key not in {"kind", "scope", "start_time_ns", "end_time_ns"}
    }
    start = source.get("start_time_ns")
    end = source.get("end_time_ns")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if start is not None and _row_time(row) < int(start):
            continue
        if end is not None and _row_time(row) >= int(end):
            continue
        if not _row_matches(row, filters):
            continue
        selected.append(row)
    return selected


def _row_matches(row: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        if key not in row:
            continue
        actual = row.get(key)
        expected_values = {str(item) for item in _as_tuple(expected)}
        if str(actual) not in expected_values:
            return False
    return True


def _row_time(row: Mapping[str, Any]) -> int:
    for key in ("timestamp_ns", "start_time_ns", "created_at"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, datetime):
            return int(value.timestamp() * 1_000_000_000)
        return int(value)
    return 0


def _scenario_ids_for_references(
    lake: Lake,
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    scenarios = lake.table("scenarios").to_arrow().to_pylist()
    by_observation: dict[str, str] = {}
    for scenario in scenarios:
        for observation_id in scenario.get("observation_ids") or ():
            by_observation[str(observation_id)] = str(scenario["scenario_id"])
    event_rows = {str(row["event_id"]): row for row in lake.table("events").to_arrow().to_pylist()}
    scenario_ids: list[str] = []
    windows = []
    for row in rows:
        scenario_id = str(row.get("scenario_id") or "")
        if scenario_id:
            scenario_ids.append(scenario_id)
            continue
        observation_id = str(row.get("observation_id") or "")
        if observation_id and observation_id in by_observation:
            scenario_ids.append(by_observation[observation_id])
            continue
        event_id = str(row.get("event_id") or "")
        event = event_rows.get(event_id) if event_id else None
        if event:
            timestamp_ns = int(event.get("timestamp_ns") or 0)
            windows.append((str(event.get("run_id") or ""), timestamp_ns, timestamp_ns))
            continue
        run_id = str(row.get("run_id") or "")
        if run_id:
            windows.append((run_id, 0, 2**63 - 1))
    scenario_ids.extend(_scenario_ids_for_windows(lake, windows))
    return sorted(dict.fromkeys(scenario_ids))


def _scenario_ids_for_windows(
    lake: Lake,
    windows: Iterable[tuple[str, int, int]],
) -> list[str]:
    scenarios = lake.table("scenarios").to_arrow().to_pylist()
    selected = []
    for run_id, start, end in windows:
        for row in scenarios:
            if str(row.get("run_id") or "") != run_id:
                continue
            row_start = int(row.get("start_time_ns") or 0)
            row_end = int(row.get("end_time_ns") or 0)
            if row_start <= end and row_end >= start:
                selected.append(str(row["scenario_id"]))
    return sorted(dict.fromkeys(selected))


def _source_version_map(source: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(item.get("table") or ""): int(item.get("version") or 0)
        for item in source.get("table_versions") or ()
        if item.get("table")
    }


def _scenario_scan_columns(
    lake: Lake,
    dimensions: Sequence[str],
    *,
    version: int | None,
) -> tuple[str, ...]:
    wanted = [
        "scenario_id",
        "run_id",
        "start_time_ns",
        "observation_ids",
        "coverage_tags",
        "quality_score",
    ]
    for dimension in dimensions:
        wanted.extend(_candidate_keys(str(dimension)))
    return _table_columns(lake, "scenarios", wanted, version=version)


def _run_scan_columns(
    lake: Lake,
    dimensions: Sequence[str],
    *,
    version: int | None,
) -> tuple[str, ...]:
    wanted = ["run_id", "quality_flags"]
    for dimension in dimensions:
        wanted.extend(_candidate_keys(str(dimension)))
    return _table_columns(lake, "runs", wanted, version=version)


def _table_columns(
    lake: Lake,
    table_name: str,
    wanted: Sequence[str],
    *,
    version: int | None,
) -> tuple[str, ...]:
    table = lake.table(table_name)
    if version is not None:
        table.checkout(version)
    try:
        available = set(table.schema.names)
        return tuple(dict.fromkeys(column for column in wanted if column in available))
    finally:
        if version is not None:
            table.checkout_latest()


def _scan_table_batches(
    lake: Lake,
    table_name: str,
    *,
    columns: Sequence[str],
    where_sql: str | None,
    version: int | None,
    batch_size: int,
) -> Iterable[list[dict[str, Any]]]:
    if not columns:
        return
    table = lake.table(table_name)
    if version is not None:
        table.checkout(version)
    try:
        query = table.search().select(list(columns))
        if where_sql:
            query = query.where(where_sql)
        for batch in query.to_batches(batch_size=batch_size):
            yield batch.to_pylist()
    finally:
        if version is not None:
            table.checkout_latest()


def _selected_rows(lake: Lake, scenario_ids: Sequence[str]) -> list[dict[str, Any]]:
    wanted = set(str(item) for item in scenario_ids)
    rows = [
        row for row in lake.table("scenarios").to_arrow().to_pylist()
        if str(row["scenario_id"]) in wanted
    ]
    if len(rows) != len(wanted):
        found = {str(row["scenario_id"]) for row in rows}
        missing = sorted(wanted - found)
        raise DistributionError(f"selected scenarios are missing from the lake: {missing}")
    return sorted(rows, key=_scenario_sort_key)


def _run_rows(
    lake: Lake,
    *,
    version: int | None = None,
    dimensions: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    columns = _run_scan_columns(lake, dimensions, version=version)
    rows: list[dict[str, Any]] = []
    for batch in _scan_table_batches(
        lake,
        "runs",
        columns=columns,
        where_sql=None,
        version=version,
        batch_size=_DEFAULT_BATCH_SIZE,
    ):
        rows.extend(batch)
    return {str(row["run_id"]): row for row in rows if row.get("run_id")}


def _chunks(values: Sequence[str], size: int) -> Iterable[tuple[str, ...]]:
    for index in range(0, len(values), size):
        yield tuple(values[index:index + size])


def _sql_in(column: str, values: Sequence[str]) -> str:
    escaped = ", ".join(_sql_string(value) for value in values)
    return f"{column} IN ({escaped})"


def _sql_string(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _label_counts(lake: Lake) -> dict[str, int]:
    scenarios = lake.table("scenarios").to_arrow().to_pylist()
    by_observation: dict[str, str] = {}
    for scenario in scenarios:
        for observation_id in scenario.get("observation_ids") or ():
            by_observation[str(observation_id)] = str(scenario["scenario_id"])
    counts: dict[str, int] = {}
    for row in lake.table("labels").to_arrow().to_pylist():
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            observation_id = str(row.get("observation_id") or "")
            scenario_id = by_observation.get(observation_id, "")
        if scenario_id:
            counts[scenario_id] = counts.get(scenario_id, 0) + 1
    return counts


def _duplicate_decision_counts(lake: Lake) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in lake.table("curation_memberships").to_arrow().to_pylist():
        if row.get("source") != "dedup" or row.get("decision") != "exclude":
            continue
        scenario_id = str(row.get("scenario_id") or row.get("target_id") or "")
        if scenario_id:
            counts[scenario_id] = counts.get(scenario_id, 0) + 1
    return counts


def _latest_snapshot_row(lake: Lake, name: str) -> dict[str, Any] | None:
    if not name:
        return None
    rows = [
        row for row in lake.table("dataset_snapshots").to_arrow().to_pylist()
        if row["name"] == name
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: (row["created_at"], row["dataset_id"]))


def _latest_view_row(lake: Lake, name: str) -> dict[str, Any] | None:
    if not name:
        return None
    rows = [
        row for row in lake.table("curation_views").to_arrow().to_pylist()
        if row["name"] == name
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: (row["created_at"], row["view_id"]))


def _scope_payload(scope: Any) -> dict[str, Any]:
    if scope is None:
        return {}
    if hasattr(scope, "to_dict"):
        return _jsonable(scope.to_dict())
    if isinstance(scope, Mapping):
        return _jsonable(dict(scope))
    if isinstance(scope, Sequence) and not isinstance(scope, (str, bytes)):
        return {"scenario_ids": [str(item) for item in scope]}
    return {"value": str(scope)}


def _slice_label(row: dict[str, Any], run: dict[str, Any], dimensions: Sequence[str]) -> str:
    return "|".join(f"{dimension}={_dimension_value(row, run, dimension)}" for dimension in dimensions)


def _label_values(label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    if not label:
        return values
    for part in str(label).split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return values


def _label_from_values(values: Mapping[str, Any]) -> str:
    return "|".join(f"{key}={value}" for key, value in sorted(values.items()))


def _dimension_value(row: dict[str, Any], run: dict[str, Any], dimension: str) -> str:
    for key in _candidate_keys(dimension):
        for source in (row, run):
            if key in source and source[key] not in (None, ""):
                value = source[key]
                if isinstance(value, list):
                    return ",".join(str(item) for item in value)
                return str(value)
        tag = _coverage_value(row.get("coverage_tags") or (), key)
        if tag is not None:
            return tag
    return "unknown"


def _candidate_keys(name: str) -> tuple[str, ...]:
    keys = [name]
    if name.endswith("_id"):
        keys.append(name[:-3])
    else:
        keys.append(f"{name}_id")
    return tuple(dict.fromkeys(keys))


def _coverage_value(tags: Sequence[str], key: str) -> str | None:
    for tag in tags:
        for sep in (":", "="):
            prefix = f"{key}{sep}"
            if str(tag).startswith(prefix):
                return str(tag)[len(prefix):]
    return None


def _numeric_value(row: dict[str, Any], column: str) -> float | None:
    if column in row and row[column] is not None:
        try:
            return float(row[column])
        except (TypeError, ValueError):
            return None
    value = _coverage_value(row.get("coverage_tags") or (), column)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _scenario_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    return (int(row.get("start_time_ns") or 0), str(row["scenario_id"]))


def _table_versions(lake: Lake, tables: Sequence[str] = _SOURCE_TABLES) -> list[dict[str, Any]]:
    return [
        {"table": table, "version": int(lake.table(table).version), "tag": ""}
        for table in tables
    ]


def _table_versions_from_source(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "table": str(item.get("table") or "external"),
            "version": int(item.get("version") or 0),
            "tag": str(item.get("tag") or ""),
        }
        for item in source.get("table_versions") or ()
    ]


def _merge_versions(
    *versions: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    merged: dict[str, int] = {}
    for group in versions:
        for table, version in group:
            merged[table] = max(merged.get(table, 0), int(version))
    return tuple(sorted(merged.items()))


def _record_distribution_transform(
    lake: Lake,
    *,
    operation: str,
    payload: dict[str, Any],
    input_table_versions: Sequence[Mapping[str, Any]],
    output_tables: Sequence[str],
    created_by: str,
) -> str:
    params = {
        "operation": f"distribution-{operation}",
        **_jsonable(payload),
        "input_table_versions": _jsonable(input_table_versions),
    }
    transform_id = "tfm-distribution-" + operation.replace("-", "_") + "-" + _digest(params)
    now = datetime.now(UTC)
    transform_row = {
        "transform_id": transform_id,
        "kind": f"distribution-{operation}",
        "input_uris": [],
        "input_table_versions": list(input_table_versions),
        "output_tables": list(output_tables),
        "params": json.dumps(params, sort_keys=True),
        "status": "completed",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{transform_id}'")
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): distribution transform provenance without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
    return transform_id


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(dict(payload)), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(inner) for key, inner in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(inner) for inner in value]
    return value


def _positive_int(value: Any, name: str, default: int) -> int:
    if value is None:
        parsed = default
    else:
        parsed = int(value)
    if parsed <= 0:
        raise DistributionError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: Any, name: str, default: int = 0) -> int:
    if value is None:
        parsed = default
    else:
        parsed = int(value)
    if parsed < 0:
        raise DistributionError(f"{name} must be non-negative")
    return parsed


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise DistributionError(f"{name} must be positive")
    return parsed


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise DistributionError(f"{name} must be non-negative")
    return parsed


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)
