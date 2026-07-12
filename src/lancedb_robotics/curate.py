"""Thin curation / mining workbench over version-pinned scenario slices.

The workbench intentionally does not create a new curation table. It composes
scenario ID selections, records each operation in ``transform_runs``, and uses
``dataset_snapshots`` for named, reproducible branch/snapshot artifacts.
"""

import base64
import hashlib
import heapq
import json
import math
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa

from lancedb_robotics.capability_gates import VERSIONING, require_lake_capability
from lancedb_robotics.comparison_plugins import (
    ComparisonMetricContext,
    ComparisonMetricPlugin,
    resolve_comparison_plugins,
)
from lancedb_robotics.dataset import SPLIT_BY_RUN, SnapshotManifest, create_snapshot
from lancedb_robotics.enrich import DEFAULT_EMBEDDING_COLUMN
from lancedb_robotics.indexing import (
    MIN_INDEX_ROWS,
    build_curation_predicate_indexes,
    build_eval_metric_catalog_predicate_indexes,
    build_review_queue_predicate_indexes,
    describe_curation_predicate_indexes,
    describe_review_queue_predicate_indexes,
    has_vector_index,
    vector_index_columns,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.lineage_hooks import (
    attach_lineage_context_to_params,
    begin_lineage_execution,
)
from lancedb_robotics.materialization import (
    ProjectionAccounting,
    json_metadata_bytes,
    normalize_table_versions,
)
from lancedb_robotics.review_connectors import (
    ReviewConnectorResult,
    ReviewConnectorTask,
    ReviewToolConnector,
)
from lancedb_robotics.schemas import (
    CURATION_COMPARISONS_SCHEMA,
    CURATION_MATERIALIZATIONS_SCHEMA,
    CURATION_MEMBERSHIPS_SCHEMA,
    CURATION_REVIEW_QUEUES_SCHEMA,
    CURATION_VIEW_MEMBERSHIP_CHUNKS_SCHEMA,
    CURATION_VIEWS_SCHEMA,
    EVAL_METRIC_CATALOG_SCHEMA,
    MODEL_OUTPUTS_SCHEMA,
    TRANSFORM_RUNS_SCHEMA,
)
from lancedb_robotics.scoring import (
    ActiveLearningScorer,
    Calibration,
    ModelOutputSignal,
    ScoringCandidate,
    resolve_calibration,
    resolve_scorer,
)

_SOURCE_TABLES = (
    "scenarios",
    "runs",
    "events",
    "episodes",
    "observations",
    "curation_views",
    "curation_view_membership_chunks",
    "curation_memberships",
    "curation_review_queues",
    "curation_materializations",
)
_DEDUP_SOURCE_TABLES = ("scenarios", "runs", "labels")
_MEMBERSHIP_DECISIONS = (
    "include",
    "exclude",
    "defer",
    "needs-review",
    "label",
    "relabel",
    "promote",
    "reject",
)
_EXCLUDING_DECISIONS = ("exclude", "defer", "needs-review")
_DECISION_SOURCES = ("human", "model", "rule", "dedup", "active-learning", "gap-analysis")
_TARGET_GRAINS = ("scenario", "episode", "observation", "aligned-frame", "snapshot-row")
_REVIEW_QUEUE_SOURCE_OPERATIONS = (
    "failure-mining",
    "active-learning",
    "gap-finding",
    "dedup-review",
    "eval-regression",
    "manual",
)
_REVIEW_QUEUE_SOURCE_ALIASES = {
    "mine-failures": "failure-mining",
    "failure": "failure-mining",
    "failures": "failure-mining",
    "uncertainty": "active-learning",
    "distribution-gap-analysis": "gap-finding",
    "from-gaps": "gap-finding",
    "dedup": "dedup-review",
    "semantic-dedup": "dedup-review",
    "training-eval": "eval-regression",
    "eval": "eval-regression",
    "regression": "eval-regression",
}
_REVIEW_QUEUE_STATUSES = ("open", "assigned", "exported", "completed", "skipped")
_DEFAULT_REPRESENTATIVE_POLICY = ("quality", "labels", "rarity", "earliest", "scenario_id")
_REPRESENTATIVE_POLICY_ALIASES = {
    "label": "labels",
    "labeled": "labels",
    "label-completeness": "labels",
    "label_completeness": "labels",
    "newest": "latest",
    "recency": "latest",
    "scenario-id": "scenario_id",
}
_REPRESENTATIVE_POLICY_TOKENS = (
    "quality",
    "labels",
    "rarity",
    "earliest",
    "latest",
    "scenario_id",
)
_DIVERSITY_METHODS = ("farthest-first", "cluster-representative")
_DIVERSITY_OPTIMIZATION_BACKENDS = ("deterministic-greedy",)
_BALANCE_REPORT_OPERATIONS = {
    "stratified-sample",
    "distribution-gap-analysis",
    "diversity-sample",
    "diversity-optimization",
    "from-gaps",
}
_DEFAULT_COMPARISON_METRICS = (
    "membership",
    "coverage",
    "source-overlap",
    "duplicate-pressure",
    "quality",
    "label-completeness",
    "payload",
    "materialization",
    "training-eval",
)
_COMPARISON_METRIC_ALIASES = {
    "all": "all",
    "diff": "membership",
    "snapshot-diff": "membership",
    "distribution": "coverage",
    "distributions": "coverage",
    "duplicates": "duplicate-pressure",
    "dedup": "duplicate-pressure",
    "labels": "label-completeness",
    "label": "label-completeness",
    "label-completeness": "label-completeness",
    "label_completeness": "label-completeness",
    "projection": "materialization",
    "projections": "materialization",
    "materializations": "materialization",
    "training": "training-eval",
    "eval": "training-eval",
    "evaluation": "training-eval",
    "model": "training-eval",
    "model-outputs": "training-eval",
    "source": "source-overlap",
    "sources": "source-overlap",
    "source_overlap": "source-overlap",
}
# Backlog 0094: scalable comparison execution + metric planning knobs.
_COMPARISON_DEFAULT_PREVIEW_LIMIT = 100
_COMPARISON_DEFAULT_BATCH_SIZE = 4096
_COMPARISON_LOCAL_ROW_BUDGET = 5_000_000
_COMPARISON_MEMBERSHIP_FIELDS = ("added", "removed", "shared")
# Built-in metrics that execute through the bounded-memory streaming path
# (projection + filter pushdown, no full-table Python materialization).
_COMPARISON_STREAMED_METRICS = frozenset(
    {
        "membership",
        "coverage",
        "quality",
        "label-completeness",
        "training-eval",
        "materialization",
    }
)
# Source tables each built-in metric reads, for the comparison plan's cost and
# table-version evidence. ``membership`` reads snapshot id sets, not a table.
_COMPARISON_METRIC_TABLES = {
    "membership": (),
    "coverage": ("scenarios", "runs"),
    "source-overlap": ("scenarios", "runs", "observations"),
    "duplicate-pressure": ("curation_memberships",),
    "quality": ("scenarios", "runs"),
    "label-completeness": ("scenarios", "labels"),
    "payload": ("scenarios", "observations"),
    "materialization": ("curation_materializations",),
    "training-eval": ("model_outputs", "feedback"),
}
_LOCAL_EXECUTION = "local"
_EXTERNAL_EXECUTION = "external"
_DEFAULT_DEDUP_NEIGHBOR_LIMIT = 64
_PROMOTION_DECISIONS = ("promote", "reject")
_ROW_PLAN_TARGET_GRAINS = ("episode", "observation", "aligned-frame", "snapshot-row")
_ROW_PLAN_INCLUDE_DECISIONS = ("include", "promote")
_ROW_PLAN_EXCLUDE_DECISIONS = (*_EXCLUDING_DECISIONS, "reject")
_ROW_PLAN_INTENT_DECISIONS = ("label", "relabel")
_TARGET_ID_COLUMNS = {
    "scenario": ("scenarios", "scenario_id"),
    "episode": ("episodes", "episode_id"),
    "observation": ("observations", "observation_id"),
    "aligned-frame": ("aligned_frames", "aligned_frame_id"),
    "aligned-tick": ("aligned_ticks", "aligned_tick_id"),
}
_FEEDBACK_LOOP_TABLES = (
    "dataset_snapshots",
    "scenarios",
    "runs",
    "model_outputs",
    "feedback",
    "curation_views",
    "curation_memberships",
    "curation_review_queues",
    "curation_comparisons",
    "transform_runs",
)
_EVAL_METRIC_STATE_ACTIVE = "active"
_EVAL_METRIC_STATE_SUPERSEDED = "superseded"
_EVAL_METRIC_STATE_PRUNED = "pruned"
_EVAL_METRIC_STATES = (
    _EVAL_METRIC_STATE_ACTIVE,
    _EVAL_METRIC_STATE_SUPERSEDED,
    _EVAL_METRIC_STATE_PRUNED,
)
# Source tables a stale eval metric is checked against: the tables whose
# advance invalidates "is this metric still about the current lake state"
# (snapshot membership, curation decisions, model outputs, human feedback).
_EVAL_METRIC_STALENESS_TABLES = (
    "dataset_snapshots",
    "curation_memberships",
    "model_outputs",
    "feedback",
)
_EVAL_METRIC_DEFAULT_PREVIEW_LIMIT = 100
# Backlog 0096: scalable feedback candidate generation. ``route`` carries the
# search.py 0185/0187 contract (auto rides a persistent vector index when one
# exists, exact bypasses it, ann requires it). Below the exact-scan limit an
# unindexed pool may still fall back to the in-memory exact scan; above it the
# plan marks the vector-index requirement unmet and apply refuses with the
# index-build remedy. Preview output is bounded per regression slice; complete
# candidate counts are always recorded from the full (limit-bounded) selection.
_FEEDBACK_CANDIDATE_ROUTE_AUTO = "auto"
_FEEDBACK_CANDIDATE_ROUTE_EXACT = "exact"
_FEEDBACK_CANDIDATE_ROUTE_ANN = "ann"
_FEEDBACK_CANDIDATE_ROUTES = (
    _FEEDBACK_CANDIDATE_ROUTE_AUTO,
    _FEEDBACK_CANDIDATE_ROUTE_EXACT,
    _FEEDBACK_CANDIDATE_ROUTE_ANN,
)
_FEEDBACK_CANDIDATE_EXACT_SCAN_LIMIT = 10_000
_FEEDBACK_CANDIDATE_PREVIEW_LIMIT = 100
_FEEDBACK_CANDIDATE_OVERFETCH = 64
_FEEDBACK_CANDIDATE_DEFAULT_NPROBES = 64
_FEEDBACK_CANDIDATE_DEFAULT_REFINE_FACTOR = 50
# Regression fields that define the *identity* of a regression input for plan
# digests. Deliberately excludes transform ids and table versions, so
# re-planning the same regression after unrelated table churn yields the same
# ``plan_id``.
_FEEDBACK_REGRESSION_IDENTITY_KEYS = (
    "source_model_output_id",
    "model_output_id",
    "metric",
    "output_type",
    "slice",
    "score",
    "baseline_score",
    "improvement",
    "severity",
    "scenario_ids",
    "scenario_id",
    "snapshot_name",
    "dataset_id",
    "training_run_id",
    "evaluation_run_id",
    "model_version",
)
_VIEW_INLINE_SCENARIO_ID_LIMIT = 1024
_VIEW_MEMBERSHIP_CHUNK_SIZE = 1024
_VIEW_MEMBERSHIP_CHUNK_TABLE = "curation_view_membership_chunks"
_VIEW_STORAGE_INLINE = "inline"
_VIEW_STORAGE_CHUNKED = "chunked"
_ROW_ID_COLUMN = "_rowid"
_REVIEW_QUEUE_DEFAULT_PAGE_LIMIT = 100
_REVIEW_QUEUE_MAX_PAGE_LIMIT = 10_000
_REVIEW_QUEUE_BATCH_SIZE = 4096
_REVIEW_QUEUE_ORDER_COLUMNS = ("priority", "target_id", "queue_item_id")
_REVIEW_CONNECTOR_RESULT_STATUSES = (
    "queued",
    "exported",
    "already-present",
    "skipped",
    "failed",
)
_REVIEW_CONNECTOR_COMPLETED_STATUSES = ("completed", "complete", "done", "reviewed", "accepted")
_REVIEW_CONNECTOR_EXPORTED_STATUSES = (
    "exported",
    "in-progress",
    "in_progress",
    "pending",
    "reviewing",
)


class CurationError(Exception):
    """Raised when a curation operation cannot be evaluated."""


@dataclass(frozen=True)
class CurationScope:
    """Python-filterable scope for a curation workbench."""

    filters: dict[str, Any] = field(default_factory=dict)
    scenario_ids: tuple[str, ...] = ()
    coverage_tags: tuple[str, ...] = ()

    @classmethod
    def from_filters(cls, **filters: Any) -> "CurationScope":
        scenario_ids = tuple(_as_tuple(filters.pop("scenario_ids", ())))
        coverage_tags = tuple(_as_tuple(filters.pop("coverage_tags", ())))
        if "scenario_id" in filters:
            scenario_ids += tuple(_as_tuple(filters.pop("scenario_id")))
        if "coverage_tag" in filters:
            coverage_tags += tuple(_as_tuple(filters.pop("coverage_tag")))
        return cls(filters={k: v for k, v in filters.items() if v is not None},
                   scenario_ids=scenario_ids,
                   coverage_tags=coverage_tags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filters": _jsonable(self.filters),
            "scenario_ids": list(self.scenario_ids),
            "coverage_tags": list(self.coverage_tags),
        }


@dataclass(frozen=True)
class CurationView:
    """Durable logical curation view over scenario IDs and pinned table versions."""

    view_id: str
    name: str
    scenario_ids: tuple[str, ...]
    table_versions: tuple[tuple[str, int], ...]
    transform_id: str
    owner: str = ""
    tags: tuple[str, ...] = ()
    description: str = ""
    status: str = "active"
    membership_storage: str = _VIEW_STORAGE_INLINE
    membership_count: int = 0


@dataclass(frozen=True)
class CurationDecisionSet:
    """Membership decisions written for one saved view."""

    view: CurationView
    target_grain: str
    decision: str
    target_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    membership_ids: tuple[str, ...]
    transform_id: str


@dataclass(frozen=True)
class CurationComparison:
    """Coverage and membership delta between two curation branches/snapshots."""

    comparison_id: str
    left: str
    right: str
    report: dict[str, Any]
    transform_id: str


@dataclass(frozen=True)
class CurationComparisonEntry:
    """Queryable catalog metadata for a persisted curation comparison report.

    Light by design: listing many reports never materializes the full report
    JSON. ``report_available`` is ``True`` while the body can still be reloaded
    via ``lake.curate.comparison(...)`` and ``False`` once the body has been
    pruned (audit metadata, digest, and lineage survive either way).
    """

    comparison_id: str
    pair_alias: str
    state: str
    left_snapshot_name: str
    right_snapshot_name: str
    left_dataset_id: str
    right_dataset_id: str
    metrics: tuple[str, ...]
    dimensions: tuple[str, ...]
    added_scenario_count: int
    removed_scenario_count: int
    shared_scenario_count: int
    report_sha1: str
    report_bytes: int
    report_available: bool
    retention_policy: dict[str, Any]
    table_versions: tuple[tuple[str, int], ...]
    created_by: str
    transform_id: str
    created_at: datetime
    archived_at: datetime | None = None
    pruned_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "pair_alias": self.pair_alias,
            "state": self.state,
            "left_snapshot_name": self.left_snapshot_name,
            "right_snapshot_name": self.right_snapshot_name,
            "left_dataset_id": self.left_dataset_id,
            "right_dataset_id": self.right_dataset_id,
            "metrics": list(self.metrics),
            "dimensions": list(self.dimensions),
            "added_scenario_count": self.added_scenario_count,
            "removed_scenario_count": self.removed_scenario_count,
            "shared_scenario_count": self.shared_scenario_count,
            "report_sha1": self.report_sha1,
            "report_bytes": self.report_bytes,
            "report_available": self.report_available,
            "retention_policy": _jsonable(self.retention_policy),
            "table_versions": [
                {"table": table, "version": version, "tag": ""}
                for table, version in self.table_versions
            ],
            "created_by": self.created_by,
            "transform_id": self.transform_id,
            "created_at": self.created_at.isoformat(),
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "pruned_at": self.pruned_at.isoformat() if self.pruned_at else None,
        }


@dataclass(frozen=True)
class CurationComparisonRetentionReport:
    """Result of applying the curation-comparison retention lifecycle.

    Pruning clears report bodies to bound the operational table; archiving only
    flags superseded reports while keeping them reloadable. Snapshot ids, table
    versions, transform ids, counts, and body digests survive both, so lineage
    and snapshot evidence remain queryable after retention runs.
    """

    archived_comparison_ids: tuple[str, ...]
    pruned_comparison_ids: tuple[str, ...]
    retained_comparison_ids: tuple[str, ...]
    dry_run: bool
    body_bytes_before: int
    body_bytes_after: int
    policy: dict[str, Any]
    transform_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def archived_count(self) -> int:
        return len(self.archived_comparison_ids)

    @property
    def pruned_count(self) -> int:
        return len(self.pruned_comparison_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": "curation-comparison-retention",
            "archived_comparison_ids": list(self.archived_comparison_ids),
            "pruned_comparison_ids": list(self.pruned_comparison_ids),
            "retained_comparison_ids": list(self.retained_comparison_ids),
            "dry_run": self.dry_run,
            "body_bytes_before": self.body_bytes_before,
            "body_bytes_after": self.body_bytes_after,
            "policy": _jsonable(self.policy),
            "transform_id": self.transform_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class ComparisonMetricPlanEntry:
    """Planned cost/feasibility line for one comparison metric (backlog 0094)."""

    metric: str
    kind: str  # "builtin" | "plugin"
    required_tables: tuple[str, ...]
    table_versions: tuple[tuple[str, int], ...]
    estimated_rows: int
    execution: str  # "local" | "external"
    streamed: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "kind": self.kind,
            "required_tables": list(self.required_tables),
            "table_versions": [
                {"table": table, "version": version}
                for table, version in self.table_versions
            ],
            "estimated_rows": self.estimated_rows,
            "execution": self.execution,
            "streamed": self.streamed,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CurationComparisonPlan:
    """Cost/feasibility plan for a comparison before it executes (backlog 0094).

    Separates metric selection, snapshot evidence, required tables/versions, and
    an estimated scan cost so callers (and external Ray/Batch/Slurm executors)
    can size the work before running it. ``requires_external_executor`` is True
    when any selected metric's estimated scan exceeds the local row budget.
    """

    left: str
    right: str
    left_dataset_id: str
    right_dataset_id: str
    left_count: int
    right_count: int
    metrics: tuple[str, ...]
    plugin_metrics: tuple[str, ...]
    dimensions: tuple[str, ...]
    entries: tuple[ComparisonMetricPlanEntry, ...]
    table_versions: tuple[tuple[str, int], ...]
    batch_size: int
    local_row_budget: int
    estimated_scan_rows: int
    estimated_peak_rows: int
    requires_external_executor: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": "compare-branches-plan",
            "left": self.left,
            "right": self.right,
            "left_dataset_id": self.left_dataset_id,
            "right_dataset_id": self.right_dataset_id,
            "left_count": self.left_count,
            "right_count": self.right_count,
            "metrics": list(self.metrics),
            "plugin_metrics": list(self.plugin_metrics),
            "dimensions": list(self.dimensions),
            "entries": [entry.to_dict() for entry in self.entries],
            "table_versions": [
                {"table": table, "version": version}
                for table, version in self.table_versions
            ],
            "batch_size": self.batch_size,
            "local_row_budget": self.local_row_budget,
            "estimated_scan_rows": self.estimated_scan_rows,
            "estimated_peak_rows": self.estimated_peak_rows,
            "requires_external_executor": self.requires_external_executor,
        }


@dataclass(frozen=True)
class CurationComparisonMembershipPage:
    """A deterministic, bounded page of one membership-delta id list (0094).

    Comparison reports cap inline id previews; full added/removed/shared lists
    are paged through this handle so a million-row diff never inflates the
    report body. Pages are reproducible: the snapshots are version-pinned, so
    paging recomputes the same sorted diff every time.
    """

    comparison_id: str
    field: str  # added | removed | shared
    offset: int
    limit: int
    total: int
    scenario_ids: tuple[str, ...]
    next_page_token: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "field": self.field,
            "offset": self.offset,
            "limit": self.limit,
            "total": self.total,
            "scenario_ids": list(self.scenario_ids),
            "next_page_token": self.next_page_token,
        }


@dataclass(frozen=True)
class CurationComparisonStaleness:
    """Whether a persisted comparison's source tables have advanced (0094)."""

    comparison_id: str
    stale: bool
    advanced_tables: tuple[dict[str, Any], ...]
    recorded_table_versions: tuple[tuple[str, int], ...]
    current_table_versions: tuple[tuple[str, int], ...]
    checked_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "stale": self.stale,
            "advanced_tables": [dict(item) for item in self.advanced_tables],
            "recorded_table_versions": [
                {"table": table, "version": version}
                for table, version in self.recorded_table_versions
            ],
            "current_table_versions": [
                {"table": table, "version": version}
                for table, version in self.current_table_versions
            ],
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass(frozen=True)
class CurationMaterializationReport:
    """Copy-accounting row for a logical snapshot projected across a boundary."""

    materialization_id: str
    dataset_id: str
    snapshot_name: str
    target_format: str
    total_payload_bytes: int
    copied_payload_bytes: int
    logical_reference_bytes: int
    metadata_bytes_written: int
    transform_id: str
    report: dict[str, Any]
    planned_payload_bytes: int = 0


@dataclass(frozen=True)
class CurationFeedbackReport:
    """Imported training/eval metrics linked to one curated dataset snapshot."""

    snapshot_name: str
    dataset_id: str
    training_run_id: str
    model_version: str
    evaluation_run_id: str
    metric_output_ids: tuple[str, ...]
    regressions: tuple[dict[str, Any], ...]
    transform_id: str
    table_versions: tuple[tuple[str, int], ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class EvalMetricEntry:
    """Queryable catalog row for one imported feedback-loop eval metric (0095).

    Light by design: listing many metrics reads promoted identity/score columns
    from ``eval_metric_catalog`` and never parses ``model_outputs.output_json``.
    ``source_available`` is ``True`` while the full source-of-record row can
    still be fetched from ``model_outputs`` and ``False`` once retention pruned
    it (the catalog row survives as audit metadata either way).
    """

    model_output_id: str
    series_key: str
    state: str
    dataset_id: str
    snapshot_name: str
    snapshot_tag: str
    training_run_id: str
    model_version: str
    evaluation_run_id: str
    metric: str
    output_type: str
    slice_label: str
    slice_values: dict[str, str]
    score: float
    baseline_score: float | None
    improvement: float | None
    higher_is_better: bool
    regressed: bool
    regression_threshold: float
    scenario_count: int
    retention_policy: dict[str, Any]
    table_versions: tuple[tuple[str, int], ...]
    created_by: str
    transform_id: str
    created_at: datetime
    superseded_by: str = ""
    superseded_at: datetime | None = None
    pruned_at: datetime | None = None

    @property
    def source_available(self) -> bool:
        return self.state != _EVAL_METRIC_STATE_PRUNED

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_output_id": self.model_output_id,
            "series_key": self.series_key,
            "state": self.state,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "snapshot_tag": self.snapshot_tag,
            "training_run_id": self.training_run_id,
            "model_version": self.model_version,
            "evaluation_run_id": self.evaluation_run_id,
            "metric": self.metric,
            "output_type": self.output_type,
            "slice_label": self.slice_label,
            "slice_values": dict(self.slice_values),
            "score": self.score,
            "baseline_score": self.baseline_score,
            "improvement": self.improvement,
            "higher_is_better": self.higher_is_better,
            "regressed": self.regressed,
            "regression_threshold": self.regression_threshold,
            "scenario_count": self.scenario_count,
            "source_available": self.source_available,
            "retention_policy": _jsonable(self.retention_policy),
            "table_versions": [
                {"table": table, "version": version}
                for table, version in self.table_versions
            ],
            "created_by": self.created_by,
            "transform_id": self.transform_id,
            "created_at": self.created_at.isoformat(),
            "superseded_by": self.superseded_by,
            "superseded_at": self.superseded_at.isoformat() if self.superseded_at else None,
            "pruned_at": self.pruned_at.isoformat() if self.pruned_at else None,
        }


@dataclass(frozen=True)
class EvalMetricListing:
    """Bounded eval-metric listing: stable total count plus a capped preview."""

    entries: tuple[EvalMetricEntry, ...]
    total_count: int
    preview_limit: int

    @property
    def truncated(self) -> bool:
        return self.total_count > len(self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "count": self.total_count,
            "preview_limit": self.preview_limit,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class EvalMetricStaleness:
    """Whether an eval metric's recorded source tables have advanced (0095)."""

    model_output_id: str
    evaluation_run_id: str
    stale: bool
    advanced_tables: tuple[dict[str, Any], ...]
    recorded_table_versions: tuple[tuple[str, int], ...]
    current_table_versions: tuple[tuple[str, int], ...]
    checked_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_output_id": self.model_output_id,
            "evaluation_run_id": self.evaluation_run_id,
            "stale": self.stale,
            "advanced_tables": [dict(item) for item in self.advanced_tables],
            "recorded_table_versions": [
                {"table": table, "version": version}
                for table, version in self.recorded_table_versions
            ],
            "current_table_versions": [
                {"table": table, "version": version}
                for table, version in self.current_table_versions
            ],
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass(frozen=True)
class EvalMetricRetentionReport:
    """Result of applying the eval-metric retention lifecycle (0095).

    Superseding only flips catalog state (source rows stay). Pruning deletes
    the superseded ``model_outputs`` source rows to bound the operational
    table; the catalog rows survive as audit metadata (identity, scores,
    table-version pins, transform id). Entries protected by promotion evidence
    or still active in their series are never pruned.
    """

    superseded_ids: tuple[str, ...]
    pruned_ids: tuple[str, ...]
    protected_ids: tuple[str, ...]
    retained_ids: tuple[str, ...]
    dry_run: bool
    policy: dict[str, Any]
    transform_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def superseded_count(self) -> int:
        return len(self.superseded_ids)

    @property
    def pruned_count(self) -> int:
        return len(self.pruned_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": "eval-metric-retention",
            "superseded_ids": list(self.superseded_ids),
            "pruned_ids": list(self.pruned_ids),
            "protected_ids": list(self.protected_ids),
            "retained_ids": list(self.retained_ids),
            "dry_run": self.dry_run,
            "policy": _jsonable(self.policy),
            "transform_id": self.transform_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class EvalMetricCatalogSyncReport:
    """Result of rebuilding ``eval_metric_catalog`` from ``model_outputs`` (0095)."""

    scanned_model_outputs: int
    cataloged: int
    active: int
    superseded: int
    preserved_pruned: int
    index_results: tuple[dict[str, Any], ...]
    transform_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": "eval-metric-catalog-sync",
            "scanned_model_outputs": self.scanned_model_outputs,
            "cataloged": self.cataloged,
            "active": self.active,
            "superseded": self.superseded,
            "preserved_pruned": self.preserved_pruned,
            "index_results": [dict(item) for item in self.index_results],
            "transform_id": self.transform_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class FeedbackCandidatePreviewPage:
    """One deterministic, bounded page of planned feedback candidates (0096)."""

    plan_id: str
    rows: tuple[dict[str, Any], ...]
    limit: int
    cursor: str = ""
    next_cursor: str = ""
    has_more: bool = False
    total_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "limit": self.limit,
            "cursor": self.cursor,
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
            "total_count": self.total_count,
            "row_count": len(self.rows),
            "rows": [dict(row) for row in self.rows],
        }


@dataclass(frozen=True)
class FeedbackCandidatePlan:
    """Replayable explain plan for regression-seeded candidate generation (0096).

    ``plan_id`` digests only the *identity* inputs — regression identities,
    source-scope identity, stage configuration (limit, embedding column,
    requested route and ANN knobs), and artifact names — and deliberately
    excludes incidental table versions and the display-only ``preview_limit``,
    so re-planning after unrelated table churn (including this loop's own
    prior writes) yields the same plan id. ``table_versions`` is a separate
    audit snapshot of the feedback-loop tables at plan time.

    ``selected_by_regression`` carries the complete (limit-bounded) candidate
    selection per regression key; ``preview`` is a bounded per-slice sample of
    it and never alters the recorded complete counts. When a required vector
    index is missing over a large pool the plan is not ``runnable``: the
    requirement appears unmet in ``index_requirements`` with the exact
    index-build remedy, and candidate fields stay empty.
    """

    plan_id: str
    regressions: tuple[dict[str, Any], ...]
    regression_keys: tuple[str, ...]
    regression_identities: tuple[dict[str, Any], ...]
    source_scope: dict[str, Any]
    source_scenario_count: int
    stages: tuple[dict[str, Any], ...]
    index_requirements: tuple[dict[str, Any], ...]
    table_versions: tuple[dict[str, Any], ...]
    route: str
    effective_route: str
    route_reason: str
    nprobes: int | None
    refine_factor: int | None
    limit_per_regression: int
    preview_limit: int
    embedding_column: str
    expected_artifacts: dict[str, Any]
    selected_by_regression: dict[str, tuple[str, ...]]
    selected: tuple[str, ...]
    candidate_counts: dict[str, int]
    total_candidate_count: int
    candidate_digest: str
    preview: tuple[dict[str, Any], ...]
    runnable: bool

    @property
    def unmet_index_requirements(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            req for req in self.index_requirements if req.get("required") and not req.get("met")
        )

    def to_dict(self) -> dict[str, Any]:
        """Bounded JSON payload: counts, stages, and preview — not the full selection."""
        return {
            "operation": "feedback-regression-candidates-plan",
            "plan_id": self.plan_id,
            "regression_count": len(self.regressions),
            "regression_keys": list(self.regression_keys),
            "regression_identities": [dict(item) for item in self.regression_identities],
            "source_scope": dict(self.source_scope),
            "source_scenario_count": self.source_scenario_count,
            "stages": [dict(stage) for stage in self.stages],
            "index_requirements": [dict(req) for req in self.index_requirements],
            "table_versions": [dict(item) for item in self.table_versions],
            "route": self.route,
            "effective_route": self.effective_route,
            "route_reason": self.route_reason,
            "nprobes": self.nprobes,
            "refine_factor": self.refine_factor,
            "limit_per_regression": self.limit_per_regression,
            "preview_limit": self.preview_limit,
            "embedding_column": self.embedding_column,
            "expected_artifacts": _jsonable(self.expected_artifacts),
            "candidate_counts": dict(self.candidate_counts),
            "total_candidate_count": self.total_candidate_count,
            "candidate_digest": self.candidate_digest,
            "preview": [dict(row) for row in self.preview],
            "runnable": self.runnable,
        }


@dataclass(frozen=True)
class CurationCandidateReport:
    """Regression-seeded next curation candidate selection and optional artifacts."""

    selection: "CurationSelection"
    regressions: tuple[dict[str, Any], ...]
    transform_id: str
    report: dict[str, Any]
    queue: "CurationReviewQueue | None" = None
    view: CurationView | None = None
    snapshot: SnapshotManifest | None = None
    plan: FeedbackCandidatePlan | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class CurationPromotionDecision:
    """Promotion or rejection decision recorded against a snapshot row."""

    snapshot_name: str
    dataset_id: str
    decision: str
    reason: str
    view: CurationView
    membership_ids: tuple[str, ...]
    transform_id: str
    report: dict[str, Any]


@dataclass(frozen=True)
class CurationDecisionResolution:
    """As-of membership decision replay for one view/target scope."""

    view: CurationView | None
    target_grain: str
    target_ids: tuple[str, ...]
    as_of: datetime | None
    latest_decisions: tuple[dict[str, Any], ...]
    membership_history: tuple[dict[str, Any], ...]
    superseded_decisions: tuple[dict[str, Any], ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class CurationCompiledRowPlan:
    """Version-pinned row-grain membership plan compiled from curation decisions."""

    plan_id: str
    target_grain: str
    view: CurationView
    target_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    lance_row_ids: tuple[int | None, ...]
    table_versions: tuple[tuple[str, int], ...]
    membership_transform_ids: tuple[str, ...]
    transform_id: str
    report: dict[str, Any]
    artifact_id: str = ""
    frozen: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "target_grain": self.target_grain,
            "view": _view_report(self.view),
            "target_ids": list(self.target_ids),
            "scenario_ids": list(self.scenario_ids),
            "lance_row_ids": list(self.lance_row_ids),
            "table_versions": [
                {"table": table, "version": version, "tag": ""}
                for table, version in self.table_versions
            ],
            "membership_transform_ids": list(self.membership_transform_ids),
            "transform_id": self.transform_id,
            "artifact_id": self.artifact_id,
            "frozen": self.frozen,
            "report": self.report,
        }


@dataclass(frozen=True)
class CurationMembershipTrace:
    """Audit explanation for one scenario's membership in a dataset snapshot."""

    snapshot_name: str
    dataset_id: str
    scenario_id: str
    final_result: str
    included_in_snapshot: bool
    resolution: CurationDecisionResolution
    report: dict[str, Any]


@dataclass(frozen=True)
class CurationReviewOutcomeReport:
    """Writeback summary after importing human/tool outcomes for a review queue."""

    queue: "CurationReviewQueue"
    outcome_count: int
    decision_transform_ids: tuple[str, ...] = ()
    label_transform_id: str = ""
    label_transform_ids: tuple[str, ...] = ()
    label_ids: tuple[str, ...] = ()
    feedback_transform_id: str = ""
    feedback_transform_ids: tuple[str, ...] = ()
    feedback_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CurationReviewConnectorTaskReport:
    """Summary for connector task export or status-sync operations."""

    queue: "CurationReviewQueue"
    operation: str
    tool: str
    project_id: str
    item_count: int
    counts_by_status: dict[str, int]
    results: tuple[ReviewConnectorResult, ...]
    dry_run: bool = False
    plan_only: bool = False
    transform_id: str = ""
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "queue_id": self.queue.queue_id,
            "queue_name": self.queue.name,
            "tool": self.tool,
            "project_id": self.project_id,
            "item_count": self.item_count,
            "counts_by_status": dict(sorted(self.counts_by_status.items())),
            "dry_run": self.dry_run,
            "plan_only": self.plan_only,
            "transform_id": self.transform_id,
            "results": [result.to_dict() for result in self.results],
            "report": self.report,
        }


@dataclass(frozen=True)
class CurationReviewConnectorOutcomeImportReport:
    """Summary for importing reviewed outcomes from an external connector."""

    queue: "CurationReviewQueue"
    tool: str
    project_id: str
    outcome_count: int
    outcome_report: CurationReviewOutcomeReport | None = None
    dry_run: bool = False
    transform_id: str = ""
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        outcome_report = self.outcome_report
        return {
            "operation": "review-queue-connector-outcome-import",
            "queue_id": self.queue.queue_id,
            "queue_name": self.queue.name,
            "tool": self.tool,
            "project_id": self.project_id,
            "outcome_count": self.outcome_count,
            "dry_run": self.dry_run,
            "transform_id": self.transform_id,
            "decision_transform_ids": list(
                outcome_report.decision_transform_ids if outcome_report else ()
            ),
            "label_transform_ids": list(outcome_report.label_transform_ids if outcome_report else ()),
            "label_ids": list(outcome_report.label_ids if outcome_report else ()),
            "feedback_transform_ids": list(
                outcome_report.feedback_transform_ids if outcome_report else ()
            ),
            "feedback_ids": list(outcome_report.feedback_ids if outcome_report else ()),
            "report": self.report,
        }


@dataclass(frozen=True)
class CurationReviewQueuePage:
    """One deterministic page of review queue items."""

    queue: "CurationReviewQueue"
    rows: tuple[dict[str, Any], ...]
    limit: int
    cursor: str = ""
    next_cursor: str = ""
    has_more: bool = False

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(str(row["queue_item_id"]) for row in self.rows)

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(str(row["target_id"]) for row in self.rows)

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(row["scenario_id"]) for row in self.rows if row["scenario_id"]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue.queue_id,
            "queue_name": self.queue.name,
            "limit": self.limit,
            "cursor": self.cursor,
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
            "item_count": len(self.rows),
            "items": list(self.rows),
        }


@dataclass(frozen=True)
class CurationReviewQueue:
    """Logical review/labeling queue backed by canonical queue-item rows."""

    lake: Lake
    queue_id: str
    name: str
    target_grain: str
    item_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    source_operation: str
    transform_id: str
    source_transform_ids: tuple[str, ...]
    table_versions: tuple[tuple[str, int], ...]
    item_count: int | None = None
    #: "written" when this call persisted rows, "unchanged" when an identical
    #: row set already existed and the write was skipped (backlog 0096).
    write_status: str = "written"

    def rows(self) -> tuple[dict[str, Any], ...]:
        """Return queue item rows in priority order."""
        rows = _review_queue_rows(self.lake, self.queue_id)
        return tuple(sorted(rows, key=_review_queue_sort_key))

    def page(
        self,
        *,
        limit: int = _REVIEW_QUEUE_DEFAULT_PAGE_LIMIT,
        cursor: str | None = None,
        status: str | None = None,
        assignee: str | None = None,
        batch_size: int = _REVIEW_QUEUE_BATCH_SIZE,
    ) -> CurationReviewQueuePage:
        """Return a deterministic cursor page ordered by priority and target id."""
        normalized_limit = _normalize_review_queue_page_limit(limit)
        rows = _review_queue_page_rows(
            self.lake,
            self.queue_id,
            limit=normalized_limit + 1,
            cursor=cursor,
            status=status,
            assignee=assignee,
            batch_size=batch_size,
        )
        page_rows = tuple(rows[:normalized_limit])
        has_more = len(rows) > normalized_limit
        next_cursor = _review_queue_cursor(page_rows[-1]) if has_more and page_rows else ""
        return CurationReviewQueuePage(
            queue=self,
            rows=page_rows,
            limit=normalized_limit,
            cursor=str(cursor or ""),
            next_cursor=next_cursor,
            has_more=has_more,
        )

    def summary(
        self,
        *,
        batch_size: int = _REVIEW_QUEUE_BATCH_SIZE,
    ) -> dict[str, Any]:
        """Summarize queue status and assignment counts without materializing rows."""
        return _review_queue_summary(self.lake, self.queue_id, batch_size=batch_size)

    def export_manifest(
        self,
        *,
        tool: str = "generic",
        output_uri: str = "",
        created_by: str = "lancedb-robotics",
        lineage_context: Any | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Build a projection manifest for an external review tool.

        The manifest is logical: it records source versions and transforms but
        does not copy payload bytes or create external tasks by itself.
        """
        if limit is None and not cursor:
            rows = self.rows()
            next_cursor = ""
            has_more = False
            page_limit = len(rows)
        else:
            page = self.page(
                limit=limit or _REVIEW_QUEUE_DEFAULT_PAGE_LIMIT,
                cursor=cursor,
            )
            rows = page.rows
            next_cursor = page.next_cursor
            has_more = page.has_more
            page_limit = page.limit
        handle = begin_lineage_execution(
            lineage_context,
            operation="curation-review-queue-export",
            params={"queue_id": self.queue_id, "tool": tool, "output_uri": output_uri},
        )
        context = handle.finish(status="completed")
        source_transform_ids = tuple(
            dict.fromkeys((*self.source_transform_ids, self.transform_id))
        )
        source_table_versions = [
            {"table": table, "version": version, "tag": ""}
            for table, version in self.table_versions
        ]
        items = [
            {
                "queue_item_id": row["queue_item_id"],
                "target_grain": row["target_grain"],
                "target_id": row["target_id"],
                "scenario_id": row["scenario_id"],
                "priority": row["priority"],
                "priority_score": row["priority_score"],
                "priority_reason": row["priority_reason"],
                "status": row["status"],
                "assignee": row["assignee"],
                "external_task_id": row["external_task_id"],
                "external_url": row["external_url"],
                "metadata": _metadata_dict(row.get("metadata") or ()),
            }
            for row in rows
        ]
        page_scenario_ids = tuple(
            dict.fromkeys(str(row["scenario_id"]) for row in rows if row["scenario_id"])
        )
        if limit is None and not cursor and self.scenario_ids:
            selected_scenario_ids = self.scenario_ids
        else:
            selected_scenario_ids = page_scenario_ids
        payload_summary = _payload_summary(self.lake, selected_scenario_ids)
        summary = self.summary()
        report = {
            "operation": "review-queue-export",
            "queue_id": self.queue_id,
            "queue_name": self.name,
            "tool": tool,
            "output_uri": output_uri,
            "target_grain": self.target_grain,
            "item_count": len(items),
            "total_item_count": summary["item_count"],
            "page": {
                "limit": page_limit,
                "cursor": str(cursor or ""),
                "next_cursor": next_cursor,
                "has_more": has_more,
            },
            "source_table_versions": source_table_versions,
            "source_transform_ids": list(source_transform_ids),
            "queue_transform_id": self.transform_id,
        }
        if context:
            report["lineage_context"] = context.to_dict()
        report["projection_accounting"] = ProjectionAccounting(
            logical_row_count=len(items),
            selected_scenario_count=len(self.scenario_ids),
            selected_observation_count=int(payload_summary["observation_count"]),
            payload_bytes_referenced=int(payload_summary["total_payload_bytes"]),
            payload_bytes_copied=0,
            metadata_bytes_written=json_metadata_bytes({**report, "items": items}),
            target_format=f"labeling-manifest:{tool}",
            target_path=output_uri,
            projection_transform_id="",
            source_snapshot_id="",
            source_snapshot_name=self.name,
            source_table_versions=normalize_table_versions(source_table_versions),
            mode="plan",
            payload_copy_policy="logical-reference",
            dry_run=True,
        ).to_dict()
        export_transform_id = _record_curation_transform(
            self.lake,
            operation="review-queue-export",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=self.scenario_ids,
            report=report,
            prior_transform_ids=source_transform_ids,
            created_by=created_by,
            lineage_context=context,
        )
        projection_accounting = {
            **report["projection_accounting"],
            "projection_transform_id": export_transform_id,
        }
        return {
            "kind": "curation-review-queue-export",
            **report,
            "projection_accounting": projection_accounting,
            "export_transform_id": export_transform_id,
            "items": items,
        }

    def export_to_connector(
        self,
        connector: ReviewToolConnector | None = None,
        *,
        project_id: str,
        tool: str | None = None,
        output_uri: str = "",
        created_by: str = "lancedb-robotics",
        lineage_context: Any | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        status: str | None = None,
        assignee: str | None = None,
        dry_run: bool = False,
        plan_only: bool = False,
    ) -> CurationReviewConnectorTaskReport:
        """Create/upsert external tasks for queue items through a connector.

        The lake owns idempotency: rows with existing external ids are reported
        as already present and are not sent to the connector again.
        """
        resolved_tool = _review_connector_tool_name(connector, tool)
        normalized_project = _normalize_review_connector_text(project_id, "project id")
        rows, page_report = _review_connector_rows(
            self,
            limit=limit,
            cursor=cursor,
            status=status,
            assignee=assignee,
        )
        tasks = tuple(
            _review_connector_task(row, tool=resolved_tool, project_id=normalized_project)
            for row in rows
        )
        existing_results: list[ReviewConnectorResult] = []
        pending_tasks: list[ReviewConnectorTask] = []
        for row, task in zip(rows, tasks, strict=True):
            if row.get("external_task_id") or row.get("external_url"):
                existing_results.append(
                    ReviewConnectorResult(
                        queue_item_id=task.queue_item_id,
                        idempotency_key=task.idempotency_key,
                        status="already-present",
                        external_task_id=str(row.get("external_task_id") or ""),
                        external_url=str(row.get("external_url") or ""),
                        metadata={
                            "tool": resolved_tool,
                            "project_id": normalized_project,
                            "idempotency_key": task.idempotency_key,
                        },
                    )
                )
            else:
                pending_tasks.append(task)

        if dry_run or plan_only:
            connector_results = [
                ReviewConnectorResult(
                    queue_item_id=task.queue_item_id,
                    idempotency_key=task.idempotency_key,
                    status="queued",
                    metadata={
                        "tool": resolved_tool,
                        "project_id": normalized_project,
                        "target_grain": task.target_grain,
                        "target_id": task.target_id,
                    },
                )
                for task in pending_tasks
            ]
        elif not pending_tasks:
            connector_results = []
        else:
            if connector is None:
                raise CurationError("connector is required unless dry_run or plan_only is set")
            try:
                raw_results = connector.upsert_tasks(pending_tasks, project_id=normalized_project)
            except Exception as exc:  # noqa: BLE001 - connectors are external adapters
                raw_results = [
                    ReviewConnectorResult(
                        queue_item_id=task.queue_item_id,
                        idempotency_key=task.idempotency_key,
                        status="failed",
                        error=str(exc),
                    )
                    for task in pending_tasks
                ]
            connector_results = _normalize_review_connector_results(
                pending_tasks,
                raw_results,
                operation="export",
            )

        results = tuple(
            sorted(
                [*existing_results, *connector_results],
                key=lambda result: _review_connector_result_sort_key(result, rows),
            )
        )
        counts = _review_connector_result_counts(results)
        report = _review_connector_task_report(
            operation="review-queue-connector-export",
            queue=self,
            tool=resolved_tool,
            project_id=normalized_project,
            output_uri=output_uri,
            page=page_report,
            tasks=tasks,
            results=results,
            dry_run=dry_run,
            plan_only=plan_only,
        )
        transform_id = ""
        if not dry_run and not plan_only:
            transform_id = _record_curation_transform(
                self.lake,
                operation="review-queue-connector-export",
                input_scenario_ids=self.scenario_ids,
                output_scenario_ids=self.scenario_ids,
                report=report,
                prior_transform_ids=tuple(
                    dict.fromkeys((*self.source_transform_ids, self.transform_id))
                ),
                created_by=created_by,
                output_tables=("curation_review_queues",),
                lineage_context=lineage_context,
            )
            _apply_review_connector_export_results(
                self.lake,
                queue_id=self.queue_id,
                results=results,
                tool=resolved_tool,
                project_id=normalized_project,
                output_uri=output_uri,
                transform_id=transform_id,
            )
        return CurationReviewConnectorTaskReport(
            queue=self,
            operation="review-queue-connector-export",
            tool=resolved_tool,
            project_id=normalized_project,
            item_count=len(tasks),
            counts_by_status=counts,
            results=results,
            dry_run=dry_run,
            plan_only=plan_only,
            transform_id=transform_id,
            report=report,
        )

    def sync_connector_status(
        self,
        connector: ReviewToolConnector | None = None,
        *,
        project_id: str,
        tool: str | None = None,
        created_by: str = "lancedb-robotics",
        lineage_context: Any | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        dry_run: bool = False,
    ) -> CurationReviewConnectorTaskReport:
        """Sync external task status back to queue rows."""
        resolved_tool = _review_connector_tool_name(connector, tool)
        normalized_project = _normalize_review_connector_text(project_id, "project id")
        rows, page_report = _review_connector_rows(self, limit=limit, cursor=cursor)
        tasks = tuple(
            _review_connector_task(row, tool=resolved_tool, project_id=normalized_project)
            for row in rows
        )
        sync_tasks: list[ReviewConnectorTask] = []
        skipped: list[ReviewConnectorResult] = []
        for row, task in zip(rows, tasks, strict=True):
            if row.get("external_task_id") or row.get("external_url"):
                sync_tasks.append(task)
            else:
                skipped.append(
                    ReviewConnectorResult(
                        queue_item_id=task.queue_item_id,
                        idempotency_key=task.idempotency_key,
                        status="skipped",
                        error="queue item has no external task mapping",
                    )
                )
        if dry_run:
            connector_results = [
                ReviewConnectorResult(
                    queue_item_id=task.queue_item_id,
                    idempotency_key=task.idempotency_key,
                    status="queued",
                    metadata={"tool": resolved_tool, "project_id": normalized_project},
                )
                for task in sync_tasks
            ]
        elif not sync_tasks:
            connector_results = []
        else:
            if connector is None:
                raise CurationError("connector is required unless dry_run is set")
            try:
                raw_results = connector.sync_task_status(sync_tasks, project_id=normalized_project)
            except Exception as exc:  # noqa: BLE001 - connectors are external adapters
                raw_results = [
                    ReviewConnectorResult(
                        queue_item_id=task.queue_item_id,
                        idempotency_key=task.idempotency_key,
                        status="failed",
                        error=str(exc),
                    )
                    for task in sync_tasks
                ]
            connector_results = _normalize_review_connector_results(
                sync_tasks,
                raw_results,
                operation="status-sync",
            )
        results = tuple(
            sorted(
                [*skipped, *connector_results],
                key=lambda result: _review_connector_result_sort_key(result, rows),
            )
        )
        counts = _review_connector_result_counts(results)
        report = _review_connector_task_report(
            operation="review-queue-connector-status-sync",
            queue=self,
            tool=resolved_tool,
            project_id=normalized_project,
            output_uri="",
            page=page_report,
            tasks=tasks,
            results=results,
            dry_run=dry_run,
            plan_only=False,
        )
        transform_id = ""
        if not dry_run:
            transform_id = _record_curation_transform(
                self.lake,
                operation="review-queue-connector-status-sync",
                input_scenario_ids=self.scenario_ids,
                output_scenario_ids=self.scenario_ids,
                report=report,
                prior_transform_ids=tuple(
                    dict.fromkeys((*self.source_transform_ids, self.transform_id))
                ),
                created_by=created_by,
                output_tables=("curation_review_queues",),
                lineage_context=lineage_context,
            )
            _apply_review_connector_status_results(
                self.lake,
                queue_id=self.queue_id,
                results=results,
                tool=resolved_tool,
                project_id=normalized_project,
                transform_id=transform_id,
            )
        return CurationReviewConnectorTaskReport(
            queue=self,
            operation="review-queue-connector-status-sync",
            tool=resolved_tool,
            project_id=normalized_project,
            item_count=len(tasks),
            counts_by_status=counts,
            results=results,
            dry_run=dry_run,
            plan_only=False,
            transform_id=transform_id,
            report=report,
        )

    def import_connector_outcomes(
        self,
        connector: ReviewToolConnector | None = None,
        *,
        project_id: str,
        tool: str | None = None,
        view_name: str | None = None,
        created_by: str = "lancedb-robotics",
        lineage_context: Any | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        dry_run: bool = False,
    ) -> CurationReviewConnectorOutcomeImportReport:
        """Import connector-reviewed outcomes through the local writeback API."""
        resolved_tool = _review_connector_tool_name(connector, tool)
        normalized_project = _normalize_review_connector_text(project_id, "project id")
        rows, page_report = _review_connector_rows(self, limit=limit, cursor=cursor)
        tasks = tuple(
            _review_connector_task(row, tool=resolved_tool, project_id=normalized_project)
            for row in rows
            if row.get("external_task_id") or row.get("external_url")
        )
        if dry_run:
            outcomes: tuple[Mapping[str, Any], ...] = ()
        elif not tasks:
            outcomes = ()
        else:
            if connector is None:
                raise CurationError("connector is required unless dry_run is set")
            outcomes = tuple(connector.import_outcomes(tasks, project_id=normalized_project))
        outcome_rows = [
            _review_connector_outcome_row(row, tool=resolved_tool)
            for row in outcomes
        ]
        report = {
            "operation": "review-queue-connector-outcome-import",
            "queue_id": self.queue_id,
            "queue_name": self.name,
            "tool": resolved_tool,
            "project_id": normalized_project,
            "dry_run": dry_run,
            "page": page_report,
            "task_count": len(tasks),
            "outcome_count": len(outcome_rows),
            "idempotency_keys": [task.idempotency_key for task in tasks],
        }
        outcome_report: CurationReviewOutcomeReport | None = None
        transform_id = ""
        if not dry_run and outcome_rows:
            outcome_report = self.import_outcomes(
                outcome_rows,
                view_name=view_name,
                source=f"{resolved_tool}:{normalized_project}",
                created_by=created_by,
            )
            report.update(
                {
                    "decision_transform_ids": list(outcome_report.decision_transform_ids),
                    "label_transform_ids": list(outcome_report.label_transform_ids),
                    "label_ids": list(outcome_report.label_ids),
                    "feedback_transform_ids": list(outcome_report.feedback_transform_ids),
                    "feedback_ids": list(outcome_report.feedback_ids),
                }
            )
        if not dry_run:
            output_tables = []
            if outcome_report and outcome_report.decision_transform_ids:
                output_tables.append("curation_memberships")
            if outcome_report and outcome_report.label_ids:
                output_tables.append("labels")
            if outcome_report and outcome_report.feedback_ids:
                output_tables.append("feedback")
            transform_id = _record_curation_transform(
                self.lake,
                operation="review-queue-connector-outcome-import",
                input_scenario_ids=self.scenario_ids,
                output_scenario_ids=self.scenario_ids,
                report=report,
                prior_transform_ids=tuple(
                    dict.fromkeys(
                        (
                            *self.source_transform_ids,
                            self.transform_id,
                            *(outcome_report.decision_transform_ids if outcome_report else ()),
                            *(outcome_report.label_transform_ids if outcome_report else ()),
                            *(outcome_report.feedback_transform_ids if outcome_report else ()),
                        )
                    )
                ),
                created_by=created_by,
                output_tables=tuple(output_tables),
                lineage_context=lineage_context,
            )
        return CurationReviewConnectorOutcomeImportReport(
            queue=self,
            tool=resolved_tool,
            project_id=normalized_project,
            outcome_count=len(outcome_rows),
            outcome_report=outcome_report,
            dry_run=dry_run,
            transform_id=transform_id,
            report=report,
        )

    def import_outcomes(
        self,
        outcomes: Iterable[dict[str, Any]] | dict[str, Any],
        *,
        view_name: str | None = None,
        source: str = "review-queue",
        created_by: str = "lancedb-robotics",
        writeback_batch_size: int = 1000,
    ) -> CurationReviewOutcomeReport:
        """Import review outcomes as membership decisions, labels, and feedback."""
        selection_scope = list(self.scenario_ids) if self.scenario_ids else None
        selection = self.lake.curate.workbench(scope=selection_scope)
        resolved_view = view_name or self.name
        label_rows: list[dict[str, Any]] = []
        feedback_rows: list[dict[str, Any]] = []
        decision_transform_ids: list[str] = []
        queue_row_cache: dict[tuple[str, str], dict[str, Any]] = {}
        outcome_count = 0
        label_transform_ids: list[str] = []
        label_ids: list[str] = []
        feedback_transform_ids: list[str] = []
        feedback_ids: list[str] = []

        def flush_labels() -> None:
            if not label_rows:
                return
            from lancedb_robotics.writeback import import_labels

            label_report = import_labels(
                self.lake,
                label_rows,
                source=f"{source}:{self.queue_id}",
                created_by=created_by,
            )
            label_transform_ids.append(label_report.transform_id)
            label_ids.extend(label_report.row_ids)
            label_rows.clear()

        def flush_feedback() -> None:
            if not feedback_rows:
                return
            from lancedb_robotics.writeback import record_feedback

            feedback_report = record_feedback(
                self.lake,
                feedback_rows,
                source=f"{source}:{self.queue_id}",
                created_by=created_by,
            )
            feedback_transform_ids.append(feedback_report.transform_id)
            feedback_ids.extend(feedback_report.row_ids)
            feedback_rows.clear()

        for index, raw in _iter_outcome_rows(outcomes):
            outcome_count += 1
            try:
                queue_row = _queue_row_for_outcome_lookup(
                    self.lake,
                    self.queue_id,
                    raw,
                    row_index=index,
                    cache=queue_row_cache,
                )
                target_grain = str(raw.get("target_grain") or queue_row["target_grain"])
                target_id = str(
                    raw.get("target_id")
                    or raw.get("scenario_id")
                    or raw.get("observation_id")
                    or queue_row["target_id"]
                )
                scenario_id = str(raw.get("scenario_id") or queue_row["scenario_id"] or "")
                outcome_metadata = {
                    **_metadata_dict(raw.get("metadata") or ()),
                    "queue_id": self.queue_id,
                    "queue_name": self.name,
                    "queue_item_id": queue_row["queue_item_id"],
                    "review_outcome_row_index": index,
                }
                if raw.get("decision"):
                    decisions = selection.record_decisions(
                        view_name=resolved_view,
                        decision=str(raw["decision"]),
                        target_grain=target_grain,
                        target_ids=[target_id],
                        scenario_ids=[scenario_id] if scenario_id else None,
                        reason=str(raw.get("reason") or raw.get("reason_code") or ""),
                        reason_code=str(raw.get("reason_code") or ""),
                        note=str(raw.get("note") or raw.get("reason") or ""),
                        reviewer=str(raw.get("reviewer") or raw.get("assignee") or ""),
                        queue=self.name,
                        priority=int(raw.get("priority") or queue_row["priority"] or 0),
                        score=_optional_float_value(raw.get("score"), queue_row["priority_score"]),
                        metadata=outcome_metadata,
                        source=str(raw.get("source") or "human"),
                        created_by=created_by,
                    )
                    decision_transform_ids.append(decisions.transform_id)
                if raw.get("label_type") and (
                    raw.get("label") is not None or raw.get("value") is not None
                ):
                    label_rows.append(
                        _writeback_row_from_outcome(
                            raw,
                            queue_row,
                            target_grain=target_grain,
                            target_id=target_id,
                            scenario_id=scenario_id,
                            metadata=outcome_metadata,
                            fields=(
                                "label_type",
                                "label",
                                "value",
                                "label_value",
                                "label_spec",
                                "status",
                            ),
                        )
                    )
                if raw.get("feedback_type"):
                    feedback = _writeback_row_from_outcome(
                        raw,
                        queue_row,
                        target_grain=target_grain,
                        target_id=target_id,
                        scenario_id=scenario_id,
                        metadata=outcome_metadata,
                        fields=(
                            "feedback_type",
                            "severity",
                            "notes",
                            "status",
                            "linked_incident_id",
                        ),
                    )
                    feedback["severity"] = str(feedback.get("severity") or "medium")
                    feedback_rows.append(feedback)
                if len(label_rows) >= writeback_batch_size:
                    flush_labels()
                if len(feedback_rows) >= writeback_batch_size:
                    flush_feedback()
            except CurationError:
                raise
            except Exception as exc:  # noqa: BLE001 - annotate mixed writeback errors
                raise CurationError(
                    f"review outcome row {index} failed for queue {self.queue_id!r}: {exc}"
                ) from exc

        if outcome_count == 0:
            raise CurationError("no review outcomes supplied")
        flush_labels()
        flush_feedback()

        return CurationReviewOutcomeReport(
            queue=self,
            outcome_count=outcome_count,
            decision_transform_ids=tuple(decision_transform_ids),
            label_transform_id=label_transform_ids[0] if label_transform_ids else "",
            label_transform_ids=tuple(label_transform_ids),
            label_ids=tuple(label_ids),
            feedback_transform_id=feedback_transform_ids[0] if feedback_transform_ids else "",
            feedback_transform_ids=tuple(feedback_transform_ids),
            feedback_ids=tuple(feedback_ids),
        )


@dataclass(frozen=True)
class SemanticDedupPlan:
    """Auditable semantic duplicate groups and deterministic representative choices."""

    lake: Lake
    input_scenario_ids: tuple[str, ...]
    representative_ids: tuple[str, ...]
    dropped_scenario_ids: tuple[str, ...]
    groups: tuple[dict[str, Any], ...]
    report: dict[str, Any]
    transform_id: str


@dataclass(frozen=True)
class DistributedDedupShard:
    """One completed shard in a resumable semantic dedup job."""

    shard_id: str
    label: str
    scenario_ids: tuple[str, ...]
    edges: tuple[tuple[str, str, float], ...]
    comparisons: int
    possible_pairs: int
    mode: str
    transform_id: str
    report: dict[str, Any]
    resumed: bool = False


@dataclass(frozen=True)
class CurationSelection:
    """A composable set of scenario IDs selected by the curation workbench."""

    lake: Lake
    scenario_ids: tuple[str, ...]
    scope: dict[str, Any]
    operation: str = "scope"
    transform_id: str = ""
    report: dict[str, Any] = field(default_factory=dict)
    operation_transform_ids: tuple[str, ...] = ()

    def plan_dedup(
        self,
        *,
        near_duplicate_threshold: float = 0.98,
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        representative_policy: Sequence[str] | str = _DEFAULT_REPRESENTATIVE_POLICY,
        shard_by: Sequence[str] = (),
        neighbor_limit: int = _DEFAULT_DEDUP_NEIGHBOR_LIMIT,
        require_index: bool = False,
        index_min_rows: int = MIN_INDEX_ROWS,
        created_by: str = "lancedb-robotics",
    ) -> SemanticDedupPlan:
        """Plan semantic duplicate groups without deleting source rows.

        Large candidate sets require a persistent LanceDB vector index on the
        selected embedding column. Small unindexed fixtures stay exact and
        brute-force, which keeps local tests deterministic while preventing an
        accidental all-pairs scan on real corpora.
        """
        if not 0.0 <= near_duplicate_threshold <= 1.0:
            raise CurationError("near_duplicate_threshold must be between 0 and 1")
        if neighbor_limit <= 0:
            raise CurationError("neighbor_limit must be positive")
        if index_min_rows <= 0:
            raise CurationError("index_min_rows must be positive")
        rows = _selected_rows(self.lake, self.scenario_ids)
        _require_embedding_column(self.lake, embedding_column)
        normalized_policy = _normalize_representative_policy(representative_policy)
        normalized_shards = tuple(dict.fromkeys(str(dim) for dim in shard_by if str(dim)))
        plan_payload = _semantic_dedup_payload(
            self.lake,
            rows=rows,
            embedding_column=embedding_column,
            near_duplicate_threshold=near_duplicate_threshold,
            representative_policy=normalized_policy,
            shard_by=normalized_shards,
            neighbor_limit=neighbor_limit,
            require_index=require_index,
            index_min_rows=index_min_rows,
        )
        report = plan_payload["report"]
        transform_id = _record_curation_transform(
            self.lake,
            operation="dedup",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=plan_payload["representative_ids"],
            report=report,
            prior_transform_ids=self.operation_transform_ids,
            created_by=created_by,
        )
        return SemanticDedupPlan(
            lake=self.lake,
            input_scenario_ids=self.scenario_ids,
            representative_ids=tuple(plan_payload["representative_ids"]),
            dropped_scenario_ids=tuple(plan_payload["dropped_scenario_ids"]),
            groups=tuple(plan_payload["groups"]),
            report=report,
            transform_id=transform_id,
        )

    def plan_distributed_dedup(
        self,
        *,
        job_id: str | None = None,
        near_duplicate_threshold: float = 0.98,
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        representative_policy: Sequence[str] | str = _DEFAULT_REPRESENTATIVE_POLICY,
        shard_by: Sequence[str] = (),
        neighbor_limit: int = _DEFAULT_DEDUP_NEIGHBOR_LIMIT,
        max_neighbor_limit: int | None = None,
        adaptive_neighbor_limit: bool = True,
        recall_audit_sample_size: int = 0,
        recall_audit_seed: int = 0,
        require_index: bool = False,
        index_min_rows: int = MIN_INDEX_ROWS,
        max_shards: int | None = None,
        created_by: str = "lancedb-robotics",
    ) -> SemanticDedupPlan:
        """Run a resumable shard-level semantic dedup job and return its plan.

        Completed shard transforms are reused on retry when ``job_id`` and the
        input table/version fingerprint match. ``max_shards`` is primarily a
        batching/interruption control: when it leaves pending shards, the method
        records completed shard lineage and raises with a resume hint.
        """
        if not 0.0 <= near_duplicate_threshold <= 1.0:
            raise CurationError("near_duplicate_threshold must be between 0 and 1")
        if neighbor_limit <= 0:
            raise CurationError("neighbor_limit must be positive")
        if max_neighbor_limit is not None and max_neighbor_limit < neighbor_limit:
            raise CurationError("max_neighbor_limit must be >= neighbor_limit")
        if recall_audit_sample_size < 0:
            raise CurationError("recall_audit_sample_size must be non-negative")
        if index_min_rows <= 0:
            raise CurationError("index_min_rows must be positive")
        if max_shards is not None and max_shards <= 0:
            raise CurationError("max_shards must be positive when provided")

        rows = _selected_rows(self.lake, self.scenario_ids)
        _require_embedding_column(self.lake, embedding_column)
        normalized_policy = _normalize_representative_policy(representative_policy)
        normalized_shards = tuple(dict.fromkeys(str(dim) for dim in shard_by if str(dim)))
        plan_payload = _distributed_semantic_dedup_payload(
            self.lake,
            rows=rows,
            input_scenario_ids=self.scenario_ids,
            job_id=str(job_id or "").strip(),
            embedding_column=embedding_column,
            near_duplicate_threshold=near_duplicate_threshold,
            representative_policy=normalized_policy,
            shard_by=normalized_shards,
            neighbor_limit=neighbor_limit,
            max_neighbor_limit=max_neighbor_limit,
            adaptive_neighbor_limit=adaptive_neighbor_limit,
            recall_audit_sample_size=recall_audit_sample_size,
            recall_audit_seed=recall_audit_seed,
            require_index=require_index,
            index_min_rows=index_min_rows,
            max_shards=max_shards,
            prior_transform_ids=self.operation_transform_ids,
            created_by=created_by,
        )
        report = plan_payload["report"]
        transform_id = _record_stable_curation_transform(
            self.lake,
            operation="distributed-dedup",
            stable_payload=plan_payload["stable_transform_payload"],
            stable_input_versions=plan_payload["stable_input_versions"],
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=plan_payload["representative_ids"],
            report=report,
            prior_transform_ids=self.operation_transform_ids
            + tuple(plan_payload["shard_transform_ids"]),
            created_by=created_by,
        )
        return SemanticDedupPlan(
            lake=self.lake,
            input_scenario_ids=self.scenario_ids,
            representative_ids=tuple(plan_payload["representative_ids"]),
            dropped_scenario_ids=tuple(plan_payload["dropped_scenario_ids"]),
            groups=tuple(plan_payload["groups"]),
            report=report,
            transform_id=transform_id,
        )

    def distributed_dedup(
        self,
        *,
        job_id: str | None = None,
        near_duplicate_threshold: float = 0.98,
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        representative_policy: Sequence[str] | str = _DEFAULT_REPRESENTATIVE_POLICY,
        shard_by: Sequence[str] = (),
        neighbor_limit: int = _DEFAULT_DEDUP_NEIGHBOR_LIMIT,
        max_neighbor_limit: int | None = None,
        adaptive_neighbor_limit: bool = True,
        recall_audit_sample_size: int = 0,
        recall_audit_seed: int = 0,
        require_index: bool = False,
        index_min_rows: int = MIN_INDEX_ROWS,
        max_shards: int | None = None,
        record_decisions: bool = True,
        view_name: str | None = None,
        created_by: str = "lancedb-robotics",
    ) -> "CurationSelection":
        """Apply a resumable distributed semantic dedup plan."""
        plan = self.plan_distributed_dedup(
            job_id=job_id,
            near_duplicate_threshold=near_duplicate_threshold,
            embedding_column=embedding_column,
            representative_policy=representative_policy,
            shard_by=shard_by,
            neighbor_limit=neighbor_limit,
            max_neighbor_limit=max_neighbor_limit,
            adaptive_neighbor_limit=adaptive_neighbor_limit,
            recall_audit_sample_size=recall_audit_sample_size,
            recall_audit_seed=recall_audit_seed,
            require_index=require_index,
            index_min_rows=index_min_rows,
            max_shards=max_shards,
            created_by=created_by,
        )
        decision_transform_id = ""
        if record_decisions:
            decision_transform_id = _persist_semantic_dedup_decisions(
                self,
                plan,
                view_name=view_name,
                created_by=created_by,
            )
        transform_ids = self.operation_transform_ids + (plan.transform_id,)
        if decision_transform_id:
            transform_ids += (decision_transform_id,)
        return CurationSelection(
            lake=self.lake,
            scenario_ids=plan.representative_ids,
            scope=self.scope,
            operation="distributed-dedup",
            transform_id=plan.transform_id,
            report={
                **plan.report,
                "decision_transform_id": decision_transform_id,
            },
            operation_transform_ids=transform_ids,
        )

    def dedup(
        self,
        *,
        near_duplicate_threshold: float = 0.98,
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        representative_policy: Sequence[str] | str = _DEFAULT_REPRESENTATIVE_POLICY,
        shard_by: Sequence[str] = (),
        neighbor_limit: int = _DEFAULT_DEDUP_NEIGHBOR_LIMIT,
        require_index: bool = False,
        index_min_rows: int = MIN_INDEX_ROWS,
        record_decisions: bool = True,
        view_name: str | None = None,
        created_by: str = "lancedb-robotics",
    ) -> "CurationSelection":
        """Collapse near-duplicate scenarios by semantic plan representatives."""
        plan = self.plan_dedup(
            near_duplicate_threshold=near_duplicate_threshold,
            embedding_column=embedding_column,
            representative_policy=representative_policy,
            shard_by=shard_by,
            neighbor_limit=neighbor_limit,
            require_index=require_index,
            index_min_rows=index_min_rows,
            created_by=created_by,
        )
        decision_transform_id = ""
        if record_decisions:
            decision_transform_id = _persist_semantic_dedup_decisions(
                self,
                plan,
                view_name=view_name,
                created_by=created_by,
            )
        transform_ids = self.operation_transform_ids + (plan.transform_id,)
        if decision_transform_id:
            transform_ids += (decision_transform_id,)
        return CurationSelection(
            lake=self.lake,
            scenario_ids=plan.representative_ids,
            scope=self.scope,
            operation="dedup",
            transform_id=plan.transform_id,
            report={
                **plan.report,
                "decision_transform_id": decision_transform_id,
            },
            operation_transform_ids=transform_ids,
        )

    def stratified_sample(
        self,
        *,
        by: Sequence[str] | None = None,
        per_slice: int | None = None,
        spec: Any | None = None,
        created_by: str = "lancedb-robotics",
    ) -> "CurationSelection":
        """Select up to ``per_slice`` deterministic scenarios per dimension slice."""
        distribution_spec = None
        if spec is not None:
            from lancedb_robotics.distributions import ensure_distribution_spec

            distribution_spec = ensure_distribution_spec(spec)
        dimensions_source = by if by is not None else (
            distribution_spec.dimensions if distribution_spec is not None else ()
        )
        dimensions = tuple(dict.fromkeys(str(dim) for dim in dimensions_source if str(dim)))
        if not dimensions:
            raise CurationError("stratified_sample requires at least one dimension")
        if per_slice is None and distribution_spec is not None:
            per_slice = distribution_spec.min_count_per_slice
        if per_slice is None or per_slice <= 0:
            raise CurationError("per_slice must be positive")

        rows = _selected_rows(self.lake, self.scenario_ids)
        run_rows = _run_rows(self.lake)
        groups: dict[str, list[dict]] = {}
        for row in rows:
            label = _slice_label(row, run_rows.get(row["run_id"], {}), dimensions)
            groups.setdefault(label, []).append(row)

        selected: list[str] = []
        slice_counts: dict[str, int] = {}
        for label in sorted(groups):
            chosen = sorted(groups[label], key=_scenario_sort_key)[:per_slice]
            selected.extend(row["scenario_id"] for row in chosen)
            slice_counts[label] = len(chosen)

        report = {
            "operation": "stratified-sample",
            "by": list(dimensions),
            "per_slice": per_slice,
            "distribution_spec": distribution_spec.to_dict() if distribution_spec else None,
            "slice_counts": slice_counts,
            "input_count": len(rows),
            "output_count": len(selected),
        }
        transform_id = _record_curation_transform(
            self.lake,
            operation="stratified-sample",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=selected,
            report=report,
            prior_transform_ids=self.operation_transform_ids,
            created_by=created_by,
        )
        return self._next("stratified-sample", selected, transform_id, report)

    def diversity_sample(
        self,
        *,
        limit: int,
        method: str = "farthest-first",
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        by: Sequence[str] = (),
        min_per_slice: int = 0,
        max_per_duplicate_group: int = 1,
        duplicate_threshold: float = 0.98,
        representative_policy: Sequence[str] | str = _DEFAULT_REPRESENTATIVE_POLICY,
        shard_by: Sequence[str] = (),
        neighbor_limit: int = _DEFAULT_DEDUP_NEIGHBOR_LIMIT,
        require_index: bool = False,
        index_min_rows: int = MIN_INDEX_ROWS,
        created_by: str = "lancedb-robotics",
    ) -> "CurationSelection":
        """Select a less-redundant semantic sample from the current curation slice."""
        normalized_method = method.strip().lower().replace("_", "-")
        if normalized_method not in _DIVERSITY_METHODS:
            raise CurationError(
                f"unknown diversity method {method!r}; expected one of "
                f"{', '.join(_DIVERSITY_METHODS)}"
            )
        if limit <= 0:
            raise CurationError("limit must be positive")
        if min_per_slice < 0:
            raise CurationError("min_per_slice must be non-negative")
        if max_per_duplicate_group <= 0:
            raise CurationError("max_per_duplicate_group must be positive")

        rows = _selected_rows(self.lake, self.scenario_ids)
        if not rows:
            raise CurationError("cannot diversity-sample an empty selection")
        dedup_plan = self.plan_dedup(
            near_duplicate_threshold=duplicate_threshold,
            embedding_column=embedding_column,
            representative_policy=representative_policy,
            shard_by=shard_by,
            neighbor_limit=neighbor_limit,
            require_index=require_index,
            index_min_rows=index_min_rows,
            created_by=created_by,
        )
        vectors = {row["scenario_id"]: _vector(row, embedding_column) for row in rows}
        row_by_id = {row["scenario_id"]: row for row in rows}
        run_rows = _run_rows(self.lake)
        dimensions = tuple(dict.fromkeys(str(dim) for dim in by if str(dim)))
        if normalized_method == "cluster-representative":
            candidate_ids = list(dedup_plan.representative_ids)
        else:
            candidate_ids = [row["scenario_id"] for row in rows]

        selected = _select_diverse_ids(
            candidate_ids,
            limit=min(limit, len(candidate_ids)),
            vectors=vectors,
            rows=row_by_id,
            runs=run_rows,
            dimensions=dimensions,
            min_per_slice=min_per_slice,
            max_per_duplicate_group=max_per_duplicate_group,
            duplicate_group_by_scenario=_duplicate_group_by_scenario(dedup_plan.groups),
            representative_policy=_normalize_representative_policy(representative_policy),
        )
        group_usage = _duplicate_group_usage(selected, dedup_plan.groups)
        report = {
            "operation": "diversity-sample",
            "method": normalized_method,
            "embedding_column": embedding_column,
            "limit": limit,
            "by": list(dimensions),
            "min_per_slice": min_per_slice,
            "max_per_duplicate_group": max_per_duplicate_group,
            "duplicate_threshold": duplicate_threshold,
            "duplicate_plan_transform_id": dedup_plan.transform_id,
            "input_count": len(rows),
            "output_count": len(selected),
            "selected_scenario_ids": selected,
            "mean_pairwise_similarity": _mean_pairwise_similarity(selected, vectors),
            "duplicate_group_usage": group_usage,
            "slice_counts": _slice_counts(selected, row_by_id, run_rows, dimensions)
            if dimensions
            else {},
        }
        transform_id = _record_curation_transform(
            self.lake,
            operation="diversity-sample",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=selected,
            report=report,
            prior_transform_ids=self.operation_transform_ids + (dedup_plan.transform_id,),
            created_by=created_by,
        )
        return self._next("diversity-sample", selected, transform_id, report)

    def optimize_diversity(
        self,
        *,
        limit: int,
        constraint_spec: Mapping[str, Any],
        backend: str = "deterministic-greedy",
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        max_per_duplicate_group: int = 1,
        duplicate_threshold: float = 0.98,
        representative_policy: Sequence[str] | str = _DEFAULT_REPRESENTATIVE_POLICY,
        shard_by: Sequence[str] = (),
        neighbor_limit: int = _DEFAULT_DEDUP_NEIGHBOR_LIMIT,
        require_index: bool = False,
        index_min_rows: int = MIN_INDEX_ROWS,
        created_by: str = "lancedb-robotics",
    ) -> "CurationSelection":
        """Select a constraint-aware, explainable diversity sample.

        The base implementation is deliberately dependency-free: it uses a
        deterministic greedy optimizer with hard maximum/duplicate caps and
        soft minimum coverage goals. Reports classify unsatisfied minimums as
        ``violated`` when the corpus cannot provide enough candidates, or
        ``relaxed`` when competing limits left a feasible goal unmet.
        """
        normalized_backend = backend.strip().lower().replace("_", "-")
        if normalized_backend not in _DIVERSITY_OPTIMIZATION_BACKENDS:
            raise CurationError(
                f"unknown diversity optimization backend {backend!r}; expected one of "
                f"{', '.join(_DIVERSITY_OPTIMIZATION_BACKENDS)}"
            )
        if limit <= 0:
            raise CurationError("limit must be positive")
        if max_per_duplicate_group <= 0:
            raise CurationError("max_per_duplicate_group must be positive")

        rows = _selected_rows(self.lake, self.scenario_ids)
        if not rows:
            raise CurationError("cannot optimize an empty curation selection")
        normalized_spec = _normalize_diversity_constraint_spec(constraint_spec)
        dedup_plan = self.plan_dedup(
            near_duplicate_threshold=duplicate_threshold,
            embedding_column=embedding_column,
            representative_policy=representative_policy,
            shard_by=shard_by,
            neighbor_limit=neighbor_limit,
            require_index=require_index,
            index_min_rows=index_min_rows,
            created_by=created_by,
        )
        vectors = {row["scenario_id"]: _vector(row, embedding_column) for row in rows}
        row_by_id = {row["scenario_id"]: row for row in rows}
        run_rows = _run_rows(self.lake)
        normalized_policy = _normalize_representative_policy(representative_policy)
        duplicate_group_by_scenario = _duplicate_group_by_scenario(dedup_plan.groups)
        candidate_ids = [row["scenario_id"] for row in rows]
        greedy_selected = _select_diverse_ids(
            candidate_ids,
            limit=min(limit, len(candidate_ids)),
            vectors=vectors,
            rows=row_by_id,
            runs=run_rows,
            dimensions=normalized_spec["dimensions"],
            min_per_slice=0,
            max_per_duplicate_group=max_per_duplicate_group,
            duplicate_group_by_scenario=duplicate_group_by_scenario,
            representative_policy=normalized_policy,
        )
        selected, optimization_report = _optimize_diversity_ids(
            candidate_ids,
            limit=limit,
            vectors=vectors,
            rows=row_by_id,
            runs=run_rows,
            constraint_spec=normalized_spec,
            duplicate_group_by_scenario=duplicate_group_by_scenario,
            max_per_duplicate_group=max_per_duplicate_group,
            representative_policy=normalized_policy,
            lake=self.lake,
        )
        greedy_constraints = _diversity_constraint_report(
            greedy_selected,
            candidate_ids=candidate_ids,
            rows=row_by_id,
            runs=run_rows,
            constraint_spec=normalized_spec,
        )
        report = {
            "operation": "diversity-optimization",
            "backend": normalized_backend,
            "embedding_column": embedding_column,
            "limit": limit,
            "constraint_spec": normalized_spec,
            "max_per_duplicate_group": max_per_duplicate_group,
            "duplicate_threshold": duplicate_threshold,
            "duplicate_plan_transform_id": dedup_plan.transform_id,
            "input_count": len(rows),
            "output_count": len(selected),
            "selected_scenario_ids": selected,
            "mean_pairwise_similarity": _mean_pairwise_similarity(selected, vectors),
            "duplicate_group_usage": _duplicate_group_usage(selected, dedup_plan.groups),
            "slice_counts": _slice_counts(
                selected,
                row_by_id,
                run_rows,
                normalized_spec["dimensions"],
            )
            if normalized_spec["dimensions"]
            else {},
            "constraints": optimization_report["constraints"],
            "constraint_summary": optimization_report["constraint_summary"],
            "selection_trace": optimization_report["selection_trace"],
            "greedy_baseline": {
                "operation": "diversity-sample",
                "selected_scenario_ids": greedy_selected,
                "output_count": len(greedy_selected),
                "mean_pairwise_similarity": _mean_pairwise_similarity(greedy_selected, vectors),
                "constraints": greedy_constraints["constraints"],
                "constraint_summary": greedy_constraints["constraint_summary"],
                "duplicate_group_usage": _duplicate_group_usage(greedy_selected, dedup_plan.groups),
            },
        }
        transform_id = _record_curation_transform(
            self.lake,
            operation="diversity-optimization",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=selected,
            report=report,
            prior_transform_ids=self.operation_transform_ids + (dedup_plan.transform_id,),
            created_by=created_by,
        )
        return self._next("diversity-optimization", selected, transform_id, report)

    def mine_failures(
        self,
        *,
        seed_event: str | None = None,
        seed_embedding: Sequence[float] | str | None = None,
        seed_scenario: str | None = None,
        limit: int = 500,
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        created_by: str = "lancedb-robotics",
    ) -> "CurationSelection":
        """Return nearest scenario neighbors of a failure seed."""
        if limit <= 0:
            raise CurationError("limit must be positive")
        seed_count = sum(value is not None for value in (seed_event, seed_embedding, seed_scenario))
        if seed_count != 1:
            raise CurationError("provide exactly one of seed_event, seed_embedding, or seed_scenario")

        rows = _selected_rows(self.lake, self.scenario_ids)
        _require_embedding_column(self.lake, embedding_column)
        vectors = {row["scenario_id"]: _vector(row, embedding_column) for row in rows}
        by_id = {row["scenario_id"]: row for row in rows}

        if seed_event is not None:
            seed_vector, seed = _seed_from_event(
                self.lake, rows, vectors, seed_event=seed_event
            )
        elif seed_scenario is not None:
            if seed_scenario not in vectors:
                raise CurationError(f"seed scenario {seed_scenario!r} is not in the workbench scope")
            seed_vector = vectors[seed_scenario]
            seed = {"kind": "scenario", "scenario_id": seed_scenario}
        else:
            seed_vector = _parse_seed_embedding(seed_embedding)
            seed = {"kind": "embedding", "dimension": len(seed_vector)}

        neighbors = []
        for scenario_id, vector in vectors.items():
            similarity = _cosine(seed_vector, vector)
            neighbors.append(
                {
                    "scenario_id": scenario_id,
                    "similarity": similarity,
                    "distance": 1.0 - similarity,
                }
            )
        neighbors.sort(key=lambda row: (-row["similarity"], row["scenario_id"]))
        selected = [row["scenario_id"] for row in neighbors[:limit]]

        report = {
            "operation": "mine-failures",
            "embedding_column": embedding_column,
            "seed": seed,
            "limit": limit,
            "neighbors": neighbors[:limit],
            "input_count": len(by_id),
            "output_count": len(selected),
        }
        transform_id = _record_curation_transform(
            self.lake,
            operation="mine-failures",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=selected,
            report=report,
            prior_transform_ids=self.operation_transform_ids,
            created_by=created_by,
        )
        return self._next("mine-failures", selected, transform_id, report)

    def failure_review_queue(
        self,
        name: str,
        *,
        seed_event: str | None = None,
        seed_embedding: Sequence[float] | str | None = None,
        seed_scenario: str | None = None,
        limit: int = 500,
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        assignee: str = "",
        status: str = "open",
        export_uri: str = "",
        metadata: dict[str, Any] | None = None,
        created_by: str = "lancedb-robotics",
    ) -> CurationReviewQueue:
        """Seed a review queue from nearest neighbors of a failure event/embedding."""
        mined = self.mine_failures(
            seed_event=seed_event,
            seed_embedding=seed_embedding,
            seed_scenario=seed_scenario,
            limit=limit,
            embedding_column=embedding_column,
            created_by=created_by,
        )
        neighbors = {row["scenario_id"]: row for row in mined.report["neighbors"]}
        priority_scores = {
            scenario_id: float(neighbors[scenario_id]["similarity"])
            for scenario_id in mined.scenario_ids
        }
        priority_reasons = {
            scenario_id: (
                f"nearest-neighbor similarity={neighbors[scenario_id]['similarity']:.6f} "
                f"distance={neighbors[scenario_id]['distance']:.6f}"
            )
            for scenario_id in mined.scenario_ids
        }
        return mined.to_review_queue(
            name,
            source_operation="failure-mining",
            priority_scores=priority_scores,
            priority_reasons=priority_reasons,
            source_ref={
                "operation": "failure-mining",
                "seed": mined.report["seed"],
                "embedding_column": embedding_column,
                "limit": limit,
            },
            assignee=assignee,
            status=status,
            export_uri=export_uri,
            metadata=metadata,
            created_by=created_by,
        )

    def _distribution_slice_context(
        self, gap_by: Sequence[str], gap_min_per_slice: int
    ) -> "_SliceContext":
        """Compute per-scenario distribution-slice deficits for gap-aware scoring."""
        dimensions = tuple(dict.fromkeys(str(dim) for dim in gap_by if str(dim)))
        if not dimensions:
            return _SliceContext()
        if gap_min_per_slice <= 0:
            raise CurationError("gap_min_per_slice must be positive")
        rows = _selected_rows(self.lake, self.scenario_ids)
        run_rows = _run_rows(self.lake)
        slice_members: dict[str, list[str]] = {}
        slice_by_scenario: dict[str, str] = {}
        for row in rows:
            label = _slice_label(row, run_rows.get(row["run_id"], {}), dimensions)
            slice_members.setdefault(label, []).append(row["scenario_id"])
            slice_by_scenario[row["scenario_id"]] = label
        needed_by_scenario: dict[str, int] = {}
        for members in slice_members.values():
            needed = max(0, gap_min_per_slice - len(members))
            for scenario_id in members:
                needed_by_scenario[scenario_id] = needed
        return _SliceContext(
            dimensions=dimensions,
            slice_by_scenario=slice_by_scenario,
            needed_by_scenario=needed_by_scenario,
        )

    def active_learning_queue(
        self,
        name: str,
        *,
        limit: int = 500,
        model_version: str = "",
        output_type: str = "",
        required_label_types: Sequence[str] = (),
        max_confidence_score: float | None = None,
        scorer: "ActiveLearningScorer | str | Sequence[Any] | None" = None,
        scorer_params: Mapping[str, Any] | None = None,
        calibration: "Calibration | Mapping[str, Any] | str | None" = None,
        gap_by: Sequence[str] = (),
        gap_min_per_slice: int = 1,
        max_per_duplicate_group: int = 1,
        duplicate_threshold: float = 0.98,
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        representative_policy: Sequence[str] | str = _DEFAULT_REPRESENTATIVE_POLICY,
        assignee: str = "",
        status: str = "open",
        export_uri: str = "",
        metadata: dict[str, Any] | None = None,
        created_by: str = "lancedb-robotics",
    ) -> CurationReviewQueue:
        """Create an active-learning queue from a pluggable scorer.

        ``scorer`` selects the scoring policy: ``None`` reproduces the 0056
        max-over-signals heuristic; a built-in/registered name, a scorer
        instance, or a sequence of ``(spec, weight)`` (a composite) are all
        accepted. ``calibration`` explicitly fixes how raw ``model_outputs.score``
        is interpreted. ``gap_by`` enriches candidates with distribution-slice
        deficits so a gap-aware scorer can boost underrepresented slices. The
        resolved scorer descriptor and calibration are recorded in transform
        lineage and in every queue row's source ref.
        """
        if limit <= 0:
            raise CurationError("limit must be positive")
        if max_per_duplicate_group < 0:
            raise CurationError("max_per_duplicate_group must be non-negative")
        resolved_calibration = resolve_calibration(calibration)
        if scorer is None and max_confidence_score is not None:
            scorer_params = {**dict(scorer_params or {}), "max_confidence_score": max_confidence_score}
        try:
            resolved_scorer = resolve_scorer(scorer, params=scorer_params)
        except ValueError as exc:
            raise CurationError(str(exc)) from exc
        slice_context = self._distribution_slice_context(gap_by, gap_min_per_slice)
        candidates = _active_learning_candidates(
            self.lake,
            self.scenario_ids,
            model_version=model_version,
            output_type=output_type,
            required_label_types=required_label_types,
            scorer=resolved_scorer,
            calibration=resolved_calibration,
            slice_context=slice_context,
        )
        prior_transform_ids = self.operation_transform_ids
        duplicate_plan_transform_id = ""
        duplicate_group_usage: dict[str, int] = {}
        if max_per_duplicate_group:
            plan = self.plan_dedup(
                near_duplicate_threshold=duplicate_threshold,
                embedding_column=embedding_column,
                representative_policy=representative_policy,
                created_by=created_by,
            )
            duplicate_plan_transform_id = plan.transform_id
            prior_transform_ids = prior_transform_ids + (plan.transform_id,)
            candidates, duplicate_group_usage = _cap_ranked_candidates_by_duplicate_group(
                candidates,
                group_by_scenario=_duplicate_group_by_scenario(plan.groups),
                max_per_group=max_per_duplicate_group,
                limit=limit,
            )
        else:
            candidates = candidates[:limit]
        selected = [candidate["scenario_id"] for candidate in candidates[:limit]]
        if not selected:
            raise CurationError("active-learning selection produced no queue candidates")
        report = {
            "operation": "active-learning",
            "model_version": model_version,
            "output_type": output_type,
            "required_label_types": list(required_label_types),
            "max_confidence_score": max_confidence_score,
            "scorer": resolved_scorer.descriptor(),
            "calibration": resolved_calibration.as_dict(),
            "gap_by": list(slice_context.dimensions),
            "gap_min_per_slice": gap_min_per_slice if slice_context.dimensions else 0,
            "limit": limit,
            "max_per_duplicate_group": max_per_duplicate_group,
            "duplicate_threshold": duplicate_threshold,
            "duplicate_plan_transform_id": duplicate_plan_transform_id,
            "duplicate_group_usage": duplicate_group_usage,
            "candidates": candidates[:limit],
            "input_count": len(self.scenario_ids),
            "output_count": len(selected),
        }
        transform_id = _record_curation_transform(
            self.lake,
            operation="active-learning",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=selected,
            report=report,
            prior_transform_ids=prior_transform_ids,
            created_by=created_by,
        )
        selection = CurationSelection(
            lake=self.lake,
            scenario_ids=tuple(selected),
            scope=self.scope,
            operation="active-learning",
            transform_id=transform_id,
            report=report,
            operation_transform_ids=prior_transform_ids + (transform_id,),
        )
        return selection.to_review_queue(
            name,
            source_operation="active-learning",
            priority_scores={
                candidate["scenario_id"]: float(candidate["priority_score"])
                for candidate in candidates[:limit]
            },
            priority_reasons={
                candidate["scenario_id"]: str(candidate["priority_reason"])
                for candidate in candidates[:limit]
            },
            source_refs={
                candidate["scenario_id"]: candidate["source_ref"]
                for candidate in candidates[:limit]
            },
            assignee=assignee,
            status=status,
            export_uri=export_uri,
            metadata=metadata,
            created_by=created_by,
        )

    def evaluate_active_learning_selection(
        self,
        *,
        limit: int = 100,
        scorer: "ActiveLearningScorer | str | Sequence[Any] | None" = None,
        scorer_params: Mapping[str, Any] | None = None,
        calibration: "Calibration | Mapping[str, Any] | str | None" = None,
        model_version: str = "",
        output_type: str = "",
        required_label_types: Sequence[str] = (),
        gap_by: Sequence[str] = (),
        gap_min_per_slice: int = 1,
        baselines: Sequence[str] = ("random", "low-confidence", "diversity-only"),
        seed: int = 0,
        duplicate_threshold: float = 0.98,
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        representative_policy: Sequence[str] | str = _DEFAULT_REPRESENTATIVE_POLICY,
        created_by: str = "lancedb-robotics",
        record: bool = True,
    ) -> dict[str, Any]:
        """Compare the pluggable scorer's selection against baseline strategies.

        Builds one candidate pool, then selects up to ``limit`` scenarios under
        each strategy (``scored`` plus the requested ``baselines``: ``random``,
        ``low-confidence``, ``diversity-only``) and reports comparable metrics:
        mean informativeness under the pluggable scorer, duplicate-group and
        distribution-slice coverage, gap hits, and Jaccard overlap with the
        scored selection. Deterministic for a fixed ``seed`` and inputs.
        """
        if limit <= 0:
            raise CurationError("limit must be positive")
        resolved_calibration = resolve_calibration(calibration)
        try:
            resolved_scorer = resolve_scorer(scorer, params=scorer_params)
        except ValueError as exc:
            raise CurationError(str(exc)) from exc
        slice_context = self._distribution_slice_context(gap_by, gap_min_per_slice)
        candidates = _collect_scoring_candidates(
            self.lake,
            self.scenario_ids,
            model_version=model_version,
            output_type=output_type,
            required_label_types=required_label_types,
            calibration=resolved_calibration,
            slice_context=slice_context,
        )
        if not candidates:
            raise CurationError("no active-learning candidates available to benchmark")

        from lancedb_robotics.scoring import ConfidenceMarginScorer

        pool_ids = [candidate.scenario_id for candidate in candidates]
        scored_lookup = {
            candidate.scenario_id: resolved_scorer.score(candidate) for candidate in candidates
        }
        score_by_scenario = {
            scenario_id: (result.score if result is not None else 0.0)
            for scenario_id, result in scored_lookup.items()
        }
        confidence_scorer = ConfidenceMarginScorer()
        confidence_by_scenario = {
            candidate.scenario_id: (
                result.score if (result := confidence_scorer.score(candidate)) is not None else None
            )
            for candidate in candidates
        }

        # Best-effort duplicate grouping for the diversity baseline / metric.
        group_by_scenario: dict[str, str] = {sid: sid for sid in pool_ids}
        diversity_degraded = False
        dedup_transform_id = ""
        try:
            plan = self.plan_dedup(
                near_duplicate_threshold=duplicate_threshold,
                embedding_column=embedding_column,
                representative_policy=representative_policy,
                created_by=created_by,
            )
            dedup_transform_id = plan.transform_id
            mapped = _duplicate_group_by_scenario(plan.groups)
            for sid in pool_ids:
                group_by_scenario[sid] = mapped.get(sid, sid)
        except CurationError:
            diversity_degraded = True

        def _scored_selection() -> list[str]:
            ranked = sorted(
                (sid for sid in pool_ids if scored_lookup[sid] is not None),
                key=lambda sid: (-float(score_by_scenario[sid]), sid),
            )
            return ranked[:limit]

        def _low_confidence_selection() -> list[str]:
            ranked = sorted(
                (sid for sid in pool_ids if confidence_by_scenario[sid] is not None),
                key=lambda sid: (-float(confidence_by_scenario[sid]), sid),
            )
            return ranked[:limit]

        def _random_selection() -> list[str]:
            ordered = sorted(
                pool_ids,
                key=lambda sid: hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest(),
            )
            return ordered[:limit]

        def _diversity_selection() -> list[str]:
            seen: set[str] = set()
            chosen: list[str] = []
            for sid in sorted(pool_ids):
                group = group_by_scenario.get(sid, sid)
                if group in seen:
                    continue
                seen.add(group)
                chosen.append(sid)
                if len(chosen) >= limit:
                    break
            return chosen

        strategy_fns = {
            "scored": _scored_selection,
            "random": _random_selection,
            "low-confidence": _low_confidence_selection,
            "diversity-only": _diversity_selection,
        }
        requested = ["scored", *[name for name in baselines if name != "scored"]]
        unknown = [name for name in requested if name not in strategy_fns]
        if unknown:
            raise CurationError(
                f"unknown benchmark baselines {unknown}; expected subset of "
                f"{sorted(name for name in strategy_fns if name != 'scored')}"
            )

        scored_selection = set(strategy_fns["scored"]())

        def _metrics(selection: Sequence[str]) -> dict[str, Any]:
            selection = list(selection)
            count = len(selection)
            mean_score = (
                sum(score_by_scenario[sid] for sid in selection) / count if count else 0.0
            )
            distinct_groups = len({group_by_scenario.get(sid, sid) for sid in selection})
            distinct_slices = len(
                {slice_context.slice_by_scenario.get(sid, "") for sid in selection}
            )
            gap_hits = sum(
                1 for sid in selection if slice_context.needed_by_scenario.get(sid, 0) > 0
            )
            overlap = len(scored_selection & set(selection))
            union = len(scored_selection | set(selection)) or 1
            return {
                "count": count,
                "mean_scored_priority": mean_score,
                "distinct_duplicate_groups": distinct_groups,
                "distinct_slices": distinct_slices,
                "gap_slice_hits": gap_hits,
                "overlap_with_scored": overlap,
                "jaccard_with_scored": overlap / union,
                "scenario_ids": selection,
            }

        strategies = {name: _metrics(strategy_fns[name]()) for name in requested}
        report = {
            "operation": "active-learning-benchmark",
            "limit": limit,
            "seed": seed,
            "scorer": resolved_scorer.descriptor(),
            "calibration": resolved_calibration.as_dict(),
            "gap_by": list(slice_context.dimensions),
            "baselines": list(requested),
            "pool_size": len(pool_ids),
            "diversity_degraded": diversity_degraded,
            "dedup_transform_id": dedup_transform_id,
            "strategies": strategies,
            "input_count": len(self.scenario_ids),
            "output_count": len(scored_selection),
        }
        if record:
            prior = self.operation_transform_ids
            if dedup_transform_id:
                prior = prior + (dedup_transform_id,)
            transform_id = _record_curation_transform(
                self.lake,
                operation="active-learning-benchmark",
                input_scenario_ids=self.scenario_ids,
                output_scenario_ids=sorted(scored_selection),
                report=report,
                prior_transform_ids=prior,
                created_by=created_by,
            )
            report["transform_id"] = transform_id
        return report

    def to_review_queue(
        self,
        name: str,
        *,
        target_grain: str = "scenario",
        target_ids: Sequence[str] | None = None,
        source_operation: str | None = None,
        priority_scores: dict[str, float] | None = None,
        priority_reasons: dict[str, str] | None = None,
        source_ref: dict[str, Any] | None = None,
        source_refs: dict[str, dict[str, Any]] | None = None,
        assignee: str = "",
        status: str = "open",
        export_uri: str = "",
        external_task_ids: dict[str, str] | None = None,
        external_urls: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        identity_payload: dict[str, Any] | None = None,
        created_by: str = "lancedb-robotics",
    ) -> CurationReviewQueue:
        """Persist this selection as logical review queue items.

        ``identity_payload`` (backlog 0096) pins the queue identity to an
        explicit stable payload (e.g. a feedback plan id + queue name + target
        grain) instead of the default digest over ``source_transform_ids``,
        which churns with table versions. On that path item identities are
        also transform-free, and an identical existing row set skips the
        delete+add entirely (``write_status == "unchanged"``). Left ``None``
        the behavior is unchanged for existing callers.
        """
        queue_name = str(name).strip()
        if not queue_name:
            raise CurationError("review queue name must not be empty")
        normalized_grain = _normalize_target_grain(target_grain)
        normalized_status = _normalize_review_queue_status(status)
        normalized_source = _normalize_review_queue_source_operation(
            source_operation or self.operation
        )
        selected_targets = _decision_target_ids(
            target_grain=normalized_grain,
            target_ids=target_ids,
            scenario_ids=self.scenario_ids,
            default_scenario_ids=self.scenario_ids,
        )
        _validate_targets(self.lake, target_grain=normalized_grain, target_ids=selected_targets)
        scenario_context = _target_scenario_context(
            self.lake,
            target_grain=normalized_grain,
            target_ids=selected_targets,
            scenario_ids=self.scenario_ids if normalized_grain != "scenario" else None,
        )
        contextual_scenario_ids = tuple(
            dict.fromkeys(sid for sid in scenario_context.values() if sid)
        )
        outside_selection = sorted(set(contextual_scenario_ids) - set(self.scenario_ids))
        if outside_selection:
            raise CurationError(
                f"review queue targets are outside selection scope: {outside_selection}"
            )
        source_transform_ids = tuple(
            dict.fromkeys(str(item) for item in self.operation_transform_ids if str(item))
        )
        table_versions = _table_versions(self.lake)
        if identity_payload is not None:
            queue_id = "queue-" + _digest(_jsonable(identity_payload))
        else:
            queue_id = "queue-" + _digest(
                {
                    "name": queue_name,
                    "target_grain": normalized_grain,
                    "target_ids": list(selected_targets),
                    "source_operation": normalized_source,
                    "source_transform_ids": list(source_transform_ids),
                }
            )
        stable_item_ids: dict[str, str] | None = None
        if identity_payload is not None:
            # Transform-free item identity: stable across table-version churn,
            # so an unchanged target set can skip the write entirely.
            stable_item_ids = {
                target_id: "qitem-"
                + _digest(
                    {
                        "queue_id": queue_id,
                        "target_grain": normalized_grain,
                        "target_id": target_id,
                        "scenario_id": scenario_context.get(target_id, ""),
                    }
                )
                for target_id in selected_targets
            }
            existing_rows = _review_queue_query_rows(
                self.lake,
                queue_id=queue_id,
                batch_size=_REVIEW_QUEUE_BATCH_SIZE,
            )
            existing_item_ids = sorted(str(row["queue_item_id"]) for row in existing_rows)
            if existing_rows and existing_item_ids == sorted(stable_item_ids.values()):
                return _review_queue_from_rows(
                    self.lake, existing_rows, write_status="unchanged"
                )
        report = {
            "operation": "review-queue",
            "queue_id": queue_id,
            "queue_name": queue_name,
            "target_grain": normalized_grain,
            "source_operation": normalized_source,
            "status": normalized_status,
            "assignee": assignee,
            "export_uri": export_uri,
            "source_transform_ids": list(source_transform_ids),
            "source_table_versions": table_versions,
            "input_count": len(self.scenario_ids),
            "output_count": len(selected_targets),
        }
        transform_id = _record_curation_transform(
            self.lake,
            operation="review-queue",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=contextual_scenario_ids or self.scenario_ids,
            report=report,
            prior_transform_ids=source_transform_ids,
            created_by=created_by,
            output_tables=("curation_review_queues",),
        )
        now = datetime.now(UTC)
        default_source_ref = source_ref or self._source_payload()
        scores = priority_scores or {}
        reasons = priority_reasons or {}
        refs = source_refs or {}
        external_ids = external_task_ids or {}
        urls = external_urls or {}
        rows: list[dict[str, Any]] = []
        for index, target_id in enumerate(selected_targets, start=1):
            scenario_id = scenario_context.get(target_id, "")
            priority_score = _priority_score(
                scores,
                target_id=target_id,
                scenario_id=scenario_id,
                fallback=float(len(selected_targets) - index + 1),
            )
            source_ref_payload = refs.get(target_id) or refs.get(scenario_id) or default_source_ref
            row_metadata = {
                **(metadata or {}),
                "selection_operation": self.operation,
                "selection_transform_id": self.transform_id,
            }
            if stable_item_ids is not None:
                queue_item_id = stable_item_ids[target_id]
            else:
                queue_item_id = "qitem-" + _digest(
                    {
                        "queue_id": queue_id,
                        "target_grain": normalized_grain,
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "source_transform_ids": list(source_transform_ids),
                        "transform_id": transform_id,
                    }
                )
            rows.append(
                {
                    "queue_item_id": queue_item_id,
                    "queue_id": queue_id,
                    "queue_name": queue_name,
                    "target_grain": normalized_grain,
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "source_operation": normalized_source,
                    "source_ref": json.dumps(_jsonable(source_ref_payload), sort_keys=True),
                    "priority": index,
                    "priority_score": priority_score,
                    "priority_reason": reasons.get(target_id)
                    or reasons.get(scenario_id)
                    or normalized_source,
                    "assignee": assignee,
                    "status": normalized_status,
                    "export_uri": export_uri,
                    "external_task_id": external_ids.get(target_id)
                    or external_ids.get(scenario_id)
                    or "",
                    "external_url": urls.get(target_id) or urls.get(scenario_id) or "",
                    "metadata": _metadata_items(row_metadata),
                    "table_versions": table_versions,
                    "source_transform_ids": list(source_transform_ids),
                    "created_by": created_by,
                    "transform_id": transform_id,
                    "created_at": now,
                }
            )
        table = self.lake.table("curation_review_queues")
        table.delete(f"queue_id = '{queue_id}'")
        table.add(pa.Table.from_pylist(rows, schema=CURATION_REVIEW_QUEUES_SCHEMA))
        return _review_queue_from_rows(self.lake, rows)

    def filter_quality(
        self,
        *,
        min_score: float | None = None,
        score_column: str = "quality_score",
        include_flags: Sequence[str] = (),
        exclude_flags: Sequence[str] = ("quarantined",),
        created_by: str = "lancedb-robotics",
    ) -> "CurationSelection":
        """Filter by run quality flags and optional scenario quality score."""
        rows = _selected_rows(self.lake, self.scenario_ids)
        run_rows = _run_rows(self.lake)
        include = set(include_flags)
        exclude = set(exclude_flags)
        selected: list[str] = []
        rejected: dict[str, str] = {}
        for row in rows:
            run = run_rows.get(row["run_id"], {})
            flags = set(run.get("quality_flags") or ())
            if include and not include <= flags:
                rejected[row["scenario_id"]] = "missing-required-quality-flag"
                continue
            if exclude and flags & exclude:
                rejected[row["scenario_id"]] = "excluded-quality-flag"
                continue
            if min_score is not None:
                score = _numeric_value(row, score_column)
                if score is None or score < min_score:
                    rejected[row["scenario_id"]] = "quality-score-below-threshold"
                    continue
            selected.append(row["scenario_id"])

        report = {
            "operation": "quality-filter",
            "min_score": min_score,
            "score_column": score_column,
            "include_flags": sorted(include),
            "exclude_flags": sorted(exclude),
            "rejected": rejected,
            "input_count": len(rows),
            "output_count": len(selected),
        }
        transform_id = _record_curation_transform(
            self.lake,
            operation="quality-filter",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=selected,
            report=report,
            prior_transform_ids=self.operation_transform_ids,
            created_by=created_by,
        )
        return self._next("quality-filter", selected, transform_id, report)

    def distribution_gap(
        self,
        *,
        by: Sequence[str],
        min_per_slice: int = 1,
        required_slices: Sequence[str] = (),
        created_by: str = "lancedb-robotics",
    ) -> "CurationSelection":
        """Record distribution coverage and underrepresented slice gaps."""
        dimensions = tuple(dict.fromkeys(str(dim) for dim in by if str(dim)))
        if not dimensions:
            raise CurationError("distribution_gap requires at least one dimension")
        if min_per_slice <= 0:
            raise CurationError("min_per_slice must be positive")

        rows = _selected_rows(self.lake, self.scenario_ids)
        run_rows = _run_rows(self.lake)
        slice_members: dict[str, list[str]] = {}
        for row in rows:
            label = _slice_label(row, run_rows.get(row["run_id"], {}), dimensions)
            slice_members.setdefault(label, []).append(row["scenario_id"])

        for label in required_slices:
            slice_members.setdefault(str(label), [])

        slice_counts = {
            label: len(ids)
            for label, ids in sorted(slice_members.items())
        }
        gaps = {
            label: {
                "count": count,
                "needed": max(0, min_per_slice - count),
            }
            for label, count in slice_counts.items()
            if count < min_per_slice
        }
        report = {
            "operation": "distribution-gap-analysis",
            "by": list(dimensions),
            "min_per_slice": min_per_slice,
            "required_slices": list(required_slices),
            "slice_counts": slice_counts,
            "gaps": gaps,
            "input_count": len(rows),
            "output_count": len(rows),
        }
        transform_id = _record_curation_transform(
            self.lake,
            operation="distribution-gap-analysis",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=self.scenario_ids,
            report=report,
            prior_transform_ids=self.operation_transform_ids,
            created_by=created_by,
        )
        return self._next("distribution-gap-analysis", self.scenario_ids, transform_id, report)

    def save_view(
        self,
        name: str,
        *,
        owner: str | None = None,
        tags: Sequence[str] = (),
        description: str = "",
        status: str = "active",
        inline_scenario_limit: int = _VIEW_INLINE_SCENARIO_ID_LIMIT,
        membership_chunk_size: int = _VIEW_MEMBERSHIP_CHUNK_SIZE,
        created_by: str = "lancedb-robotics",
    ) -> CurationView:
        """Persist this logical scenario selection as a named curation view."""
        view_name = str(name).strip()
        if not view_name:
            raise CurationError("view name must not be empty")
        if inline_scenario_limit < 0:
            raise CurationError("inline_scenario_limit must be non-negative")
        if membership_chunk_size <= 0:
            raise CurationError("membership_chunk_size must be positive")
        view_owner = str(owner or created_by).strip()
        view_tags = tuple(dict.fromkeys(str(tag).strip() for tag in tags if str(tag).strip()))
        membership_storage = _view_membership_storage_payload(
            self.scenario_ids,
            inline_scenario_limit=inline_scenario_limit,
            chunk_size=membership_chunk_size,
        )
        stored_scenario_ids = (
            list(self.scenario_ids)
            if membership_storage["kind"] == _VIEW_STORAGE_INLINE
            else []
        )
        query_spec = {
            "source": self._source_payload(),
            "scenario_ids": stored_scenario_ids,
            "membership_storage": membership_storage,
        }
        predicate_indexes = _curation_predicate_index_params(self.lake)
        report = {
            "operation": "save-view",
            "name": view_name,
            "owner": view_owner,
            "tags": list(view_tags),
            "description": description,
            "status": status,
            "scenario_ids": stored_scenario_ids,
            "membership_storage": membership_storage,
            "predicate_indexes": predicate_indexes,
            "query_spec": query_spec,
            "input_count": len(self.scenario_ids),
            "output_count": len(self.scenario_ids),
        }
        view_id = "view-" + _digest(
            {
                "name": view_name,
                "owner": view_owner,
                "tags": list(view_tags),
                "scenario_ids": list(self.scenario_ids),
                "query_spec": query_spec,
            }
        )
        transform_id = _record_curation_transform(
            self.lake,
            operation="save-view",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=self.scenario_ids,
            report={**report, "view_id": view_id},
            prior_transform_ids=self.operation_transform_ids,
            created_by=created_by,
            output_tables=(
                ("curation_views", _VIEW_MEMBERSHIP_CHUNK_TABLE)
                if membership_storage["kind"] == _VIEW_STORAGE_CHUNKED
                else ("curation_views",)
            ),
        )
        now = datetime.now(UTC)
        table_versions = _table_versions(self.lake)
        views = self.lake.table("curation_views")
        views.delete(f"view_id = '{view_id}'")
        views.add(
            pa.Table.from_pylist(
                [
                    {
                        "view_id": view_id,
                        "name": view_name,
                        "owner": view_owner,
                        "tags": list(view_tags),
                        "description": description,
                        "source_kind": "curation-workbench",
                        "scope": json.dumps(self.scope, sort_keys=True),
                        "query_spec": json.dumps(query_spec, sort_keys=True),
                        "scenario_ids": stored_scenario_ids,
                        "table_versions": table_versions,
                        "parent_transform_ids": list(self.operation_transform_ids),
                        "status": status,
                        "created_by": created_by,
                        "transform_id": transform_id,
                        "created_at": now,
                    }
                ],
                schema=CURATION_VIEWS_SCHEMA,
            )
        )
        chunk_table = self.lake.table(_VIEW_MEMBERSHIP_CHUNK_TABLE)
        chunk_table.delete(f"view_id = {_sql_literal(view_id)}")
        if membership_storage["kind"] == _VIEW_STORAGE_CHUNKED:
            chunk_table.add(
                pa.Table.from_pylist(
                    _view_membership_chunk_rows(
                        view_id=view_id,
                        scenario_ids=self.scenario_ids,
                        chunk_size=membership_chunk_size,
                        created_by=created_by,
                        transform_id=transform_id,
                        created_at=now,
                    ),
                    schema=CURATION_VIEW_MEMBERSHIP_CHUNKS_SCHEMA,
                )
            )
        return CurationView(
            view_id=view_id,
            name=view_name,
            scenario_ids=self.scenario_ids,
            table_versions=tuple((tv["table"], tv["version"]) for tv in table_versions),
            transform_id=transform_id,
            owner=view_owner,
            tags=view_tags,
            description=description,
            status=status,
            membership_storage=str(membership_storage["kind"]),
            membership_count=len(self.scenario_ids),
        )

    def record_decisions(
        self,
        *,
        decision: str,
        scenario_ids: Sequence[str] | None = None,
        target_grain: str = "scenario",
        target_ids: Sequence[str] | None = None,
        view_name: str | None = None,
        reason: str = "",
        reason_code: str = "",
        note: str = "",
        reviewer: str = "",
        queue: str = "",
        priority: int = 0,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "human",
        created_by: str = "lancedb-robotics",
    ) -> CurationDecisionSet:
        """Record append-only saved-view membership decisions at a target grain."""
        normalized_decision = _normalize_decision(decision)
        normalized_grain = _normalize_target_grain(target_grain)
        normalized_source = _normalize_decision_source(source)
        selected_targets = _decision_target_ids(
            target_grain=normalized_grain,
            target_ids=target_ids,
            scenario_ids=scenario_ids,
            default_scenario_ids=self.scenario_ids,
        )
        _validate_targets(self.lake, target_grain=normalized_grain, target_ids=selected_targets)
        scenario_context = _target_scenario_context(
            self.lake,
            target_grain=normalized_grain,
            target_ids=selected_targets,
            scenario_ids=scenario_ids,
        )

        if view_name:
            try:
                view = _view_from_row(_latest_view_row(self.lake, view_name), lake=self.lake)
            except CurationError:
                view = self.save_view(view_name, created_by=created_by)
        else:
            view = self.save_view(self.operation, created_by=created_by)
        contextual_scenario_ids = tuple(
            dict.fromkeys(sid for sid in scenario_context.values() if sid)
        )
        outside_view = sorted(set(contextual_scenario_ids) - set(view.scenario_ids))
        if outside_view:
            raise CurationError(
                f"decision scenario ids are outside saved view {view.name!r}: {outside_view}"
            )
        existing = _membership_rows(
            self.lake,
            view_id=view.view_id,
            target_grain=normalized_grain,
            target_ids=selected_targets,
        )
        latest_by_target = _latest_membership_by_target(existing)
        resolved_reason_code = str(reason_code or reason).strip()
        resolved_note = str(note or reason).strip()
        transform_report = {
            "operation": "record-decisions",
            "view_id": view.view_id,
            "view_name": view.name,
            "target_grain": normalized_grain,
            "target_ids": list(selected_targets),
            "decision": normalized_decision,
            "scenario_ids": list(contextual_scenario_ids),
            "reason_code": resolved_reason_code,
            "reason": reason,
            "note": resolved_note,
            "reviewer": reviewer,
            "queue": queue,
            "priority": priority,
            "score": score,
            "metadata": _jsonable(metadata or {}),
            "source": normalized_source,
            "input_count": len(selected_targets),
            "output_count": len(selected_targets),
        }
        transform_id = _record_curation_transform(
            self.lake,
            operation="record-decisions",
            input_scenario_ids=contextual_scenario_ids or self.scenario_ids,
            output_scenario_ids=contextual_scenario_ids or self.scenario_ids,
            report=transform_report,
            prior_transform_ids=self.operation_transform_ids + (view.transform_id,),
            created_by=created_by,
            output_tables=("curation_memberships",),
        )
        now = datetime.now(UTC)
        records: list[dict[str, Any]] = []
        membership_ids: list[str] = []
        for target_id in selected_targets:
            scenario_id = scenario_context.get(target_id, "")
            previous = latest_by_target.get((normalized_grain, target_id), {})
            membership_id = "mbr-" + _digest(
                {
                    "view_id": view.view_id,
                    "target_grain": normalized_grain,
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "decision": normalized_decision,
                    "reason_code": resolved_reason_code,
                    "reason": reason,
                    "note": resolved_note,
                    "reviewer": reviewer,
                    "queue": queue,
                    "priority": priority,
                    "score": score,
                    "metadata": _jsonable(metadata or {}),
                    "source": normalized_source,
                    "supersedes_membership_id": previous.get("membership_id", ""),
                    "transform_id": transform_id,
                }
            )
            membership_ids.append(membership_id)
            records.append(
                {
                    "membership_id": membership_id,
                    "view_id": view.view_id,
                    "target_grain": normalized_grain,
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "decision": normalized_decision,
                    "reason_code": resolved_reason_code,
                    "reason": reason,
                    "note": resolved_note,
                    "reviewer": reviewer,
                    "queue": queue,
                    "priority": int(priority),
                    "score": score,
                    "metadata": _metadata_items(metadata or {}),
                    "source": normalized_source,
                    "supersedes_membership_id": previous.get("membership_id", ""),
                    "created_by": created_by,
                    "transform_id": transform_id,
                    "created_at": now,
                }
            )

        memberships = self.lake.table("curation_memberships")
        memberships.add(pa.Table.from_pylist(records, schema=CURATION_MEMBERSHIPS_SCHEMA))
        return CurationDecisionSet(
            view=view,
            target_grain=normalized_grain,
            decision=normalized_decision,
            target_ids=selected_targets,
            scenario_ids=contextual_scenario_ids,
            membership_ids=tuple(membership_ids),
            transform_id=transform_id,
        )

    def apply_decisions(
        self,
        *,
        view_name: str | None = None,
        excluding_decisions: Sequence[str] = _EXCLUDING_DECISIONS,
        created_by: str = "lancedb-robotics",
    ) -> "CurationSelection":
        """Apply latest saved membership decisions before freezing a snapshot."""
        excluded = tuple(_normalize_decision(decision) for decision in excluding_decisions)
        view_row = _latest_view_row(self.lake, view_name) if view_name else None
        view_id = str(view_row["view_id"]) if view_row else None
        view = _view_from_row(view_row, lake=self.lake) if view_row else None
        decision_rows = _membership_rows(self.lake, view_id=view_id)
        latest = _latest_membership_by_scenario(decision_rows)
        decision_transform_ids = tuple(
            dict.fromkeys(
                str(row["transform_id"])
                for row in latest.values()
                if row.get("transform_id")
            )
        )

        base_order = list(dict.fromkeys(self.scenario_ids))
        selected: list[str] = []
        included_by_decision: list[str] = []
        removed: dict[str, str] = {}
        for scenario_id in base_order:
            decision = latest.get(scenario_id, {}).get("decision")
            if decision in excluded:
                removed[scenario_id] = decision
                continue
            selected.append(scenario_id)

        if view:
            for scenario_id in view.scenario_ids:
                decision = latest.get(scenario_id, {}).get("decision")
                if decision == "include" and scenario_id not in selected:
                    selected.append(scenario_id)
                    included_by_decision.append(scenario_id)

        if not selected:
            raise CurationError("membership decisions removed every scenario from the selection")

        report = {
            "operation": "apply-decisions",
            "view_id": view_id or "",
            "view_name": view_name or "",
            "excluding_decisions": list(excluded),
            "included_by_decision": included_by_decision,
            "removed": removed,
            "decision_transform_ids": list(decision_transform_ids),
            "input_count": len(self.scenario_ids),
            "output_count": len(selected),
        }
        transform_id = _record_curation_transform(
            self.lake,
            operation="apply-decisions",
            input_scenario_ids=self.scenario_ids,
            output_scenario_ids=selected,
            report=report,
            prior_transform_ids=self.operation_transform_ids + decision_transform_ids,
            created_by=created_by,
        )
        return CurationSelection(
            lake=self.lake,
            scenario_ids=tuple(selected),
            scope=self.scope,
            operation="apply-decisions",
            transform_id=transform_id,
            report=report,
            operation_transform_ids=(
                self.operation_transform_ids + decision_transform_ids + (transform_id,)
            ),
        )

    def compile_row_plan(
        self,
        *,
        view_name: str | None = None,
        target_grain: str = "observation",
        source_snapshot_name: str | None = None,
        include_decisions: Sequence[str] = _ROW_PLAN_INCLUDE_DECISIONS,
        excluding_decisions: Sequence[str] = _ROW_PLAN_EXCLUDE_DECISIONS,
        freeze: bool = False,
        created_by: str = "lancedb-robotics",
    ) -> CurationCompiledRowPlan:
        """Compile latest saved-view decisions into a row-grain training plan.

        Scenario-grain decisions keep their existing snapshot behavior. This
        compiler produces an explicit row plan for narrower grains so a curation
        review can exclude or re-include one observation/aligned row without
        mutating ``dataset_snapshots.scenario_ids`` or copying payload bytes.
        """
        return _compile_curation_row_plan(
            self,
            view_name=view_name,
            target_grain=target_grain,
            source_snapshot_name=source_snapshot_name,
            include_decisions=include_decisions,
            excluding_decisions=excluding_decisions,
            freeze=freeze,
            created_by=created_by,
        )

    def branch(
        self,
        name: str,
        *,
        tag: str | None = None,
        split_by: str = SPLIT_BY_RUN,
        created_by: str = "lancedb-robotics",
        lineage_context: Any | None = None,
    ) -> "CurationBranch":
        """Create a named candidate slice without copying payload rows."""
        manifest = self.snapshot(
            name=name,
            tag=tag or name,
            split_by=split_by,
            created_by=created_by,
            lineage_context=lineage_context,
        )
        return CurationBranch(name=name, selection=self, manifest=manifest)

    def snapshot(
        self,
        *,
        name: str | None = None,
        tag: str | None = None,
        split_by: str = SPLIT_BY_RUN,
        created_by: str = "lancedb-robotics",
        lineage_context: Any | None = None,
    ) -> SnapshotManifest:
        """Freeze this selection as a reproducible dataset snapshot."""
        snapshot_name = name or self.operation
        return create_snapshot(
            self.lake,
            name=snapshot_name,
            scenario_ids=list(self.scenario_ids),
            source=self._source_payload(),
            split_by=split_by,
            tag=tag,
            balance_report=self.report
            if self.operation in _BALANCE_REPORT_OPERATIONS else None,
            coverage_report=self.report
            if self.operation in {"quality-filter", "apply-decisions", "diversity-optimization"}
            else None,
            created_by=created_by,
            lineage_context=lineage_context,
        )

    def _next(
        self,
        operation: str,
        scenario_ids: Sequence[str],
        transform_id: str,
        report: dict[str, Any],
    ) -> "CurationSelection":
        return CurationSelection(
            lake=self.lake,
            scenario_ids=tuple(scenario_ids),
            scope=self.scope,
            operation=operation,
            transform_id=transform_id,
            report=report,
            operation_transform_ids=self.operation_transform_ids + (transform_id,),
        )

    def _source_payload(self) -> dict[str, Any]:
        return {
            "kind": "curation-workbench",
            "operation": self.operation,
            "scope": self.scope,
            "operation_transform_ids": list(self.operation_transform_ids),
            "scenario_count": len(self.scenario_ids),
            "report": _jsonable(self.report),
        }


@dataclass
class CurationBranch:
    """A named curation candidate backed by a dataset snapshot row."""

    name: str
    selection: CurationSelection
    manifest: SnapshotManifest

    def snapshot(
        self,
        *,
        tag: str | None = None,
        split_by: str = SPLIT_BY_RUN,
        created_by: str = "lancedb-robotics",
        lineage_context: Any | None = None,
    ) -> SnapshotManifest:
        """Freeze or retag the candidate slice."""
        self.manifest = self.selection.snapshot(
            name=self.name,
            tag=tag or self.name,
            split_by=split_by,
            created_by=created_by,
            lineage_context=lineage_context,
        )
        return self.manifest


class LakeCurate:
    """Facade exposed as ``lake.curate``."""

    def __init__(self, lake: Lake) -> None:
        self._lake = lake

    def workbench(
        self,
        *,
        scope: CurationScope | CurationView | dict | str | Sequence[str] | None = None,
        apply_decisions: bool = False,
    ) -> CurationSelection:
        """Open a composable curation workbench over matching scenarios."""
        if isinstance(scope, CurationView):
            selection = _selection_from_view(self._lake, scope)
            return selection.apply_decisions(view_name=scope.name) if apply_decisions else selection
        if isinstance(scope, str):
            selection = self.view(scope)
            return selection.apply_decisions(view_name=scope) if apply_decisions else selection
        if isinstance(scope, dict):
            view_name = scope.get("view") or scope.get("view_name") or scope.get("saved_view")
            if view_name:
                selection = self.view(str(view_name))
                consume = bool(scope.get("apply_decisions", apply_decisions))
                return selection.apply_decisions(view_name=str(view_name)) if consume else selection

        normalized = _normalize_scope(scope)
        rows = sorted(self._lake.table("scenarios").to_arrow().to_pylist(), key=_scenario_sort_key)
        run_rows = _run_rows(self._lake)
        selected = [
            row["scenario_id"]
            for row in rows
            if _matches_scope(row, run_rows.get(row["run_id"], {}), normalized)
        ]
        if not selected:
            raise CurationError(f"no scenarios match curation scope in {self._lake.uri}")
        selection = CurationSelection(
            lake=self._lake,
            scenario_ids=tuple(selected),
            scope=normalized.to_dict(),
            report={"operation": "scope", "input_count": len(rows), "output_count": len(selected)},
        )
        return selection.apply_decisions() if apply_decisions else selection

    def view(self, name: str) -> CurationSelection:
        """Reopen the latest saved curation view by name."""
        row = _latest_view_row(self._lake, name)
        return _selection_from_view(self._lake, _view_from_row(row, lake=self._lake), row=row)

    def predicate_index_status(
        self,
        *,
        include_view_chunks: bool = True,
    ) -> tuple[dict[str, Any], ...]:
        """Describe scalar predicate-index availability for curation hot paths."""
        return tuple(
            result.to_params() | _curation_predicate_index_role(result.table, result.column)
            for result in describe_curation_predicate_indexes(
                self._lake,
                include_view_chunks=include_view_chunks,
            )
        )

    def index_predicates(
        self,
        *,
        include_view_chunks: bool = True,
        refresh: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Build scalar indexes for saved-view and membership-decision lookups."""
        return tuple(
            result.to_params() | _curation_predicate_index_role(result.table, result.column)
            for result in build_curation_predicate_indexes(
                self._lake,
                include_view_chunks=include_view_chunks,
                replace=refresh,
            )
        )

    def review_queue_predicate_index_status(self) -> tuple[dict[str, Any], ...]:
        """Describe scalar predicate-index availability for review queue lookups."""
        return tuple(
            result.to_params() | _review_queue_predicate_index_role(result.column)
            for result in describe_review_queue_predicate_indexes(self._lake)
        )

    def index_review_queue_predicates(
        self,
        *,
        refresh: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Build scalar indexes for review queue reopen, pagination, and import."""
        return tuple(
            result.to_params() | _review_queue_predicate_index_role(result.column)
            for result in build_review_queue_predicate_indexes(
                self._lake,
                replace=refresh,
            )
        )

    def queue(self, name_or_id: str) -> CurationReviewQueue:
        """Reopen the latest review queue by queue name or queue id."""
        return _review_queue_lookup(self._lake, name_or_id)

    def resolve_membership(
        self,
        *,
        view_name: str | None = None,
        view_id: str | None = None,
        target_grain: str | None = "scenario",
        target_ids: Sequence[str] = (),
        scenario_ids: Sequence[str] = (),
        as_of: datetime | str | None = None,
        transform_id: str | None = None,
        snapshot_name: str | None = None,
        superseded_policy: str = "latest",
    ) -> CurationDecisionResolution:
        """Resolve latest membership decisions as of a time, transform, or snapshot."""
        return _resolve_membership(
            self._lake,
            view_name=view_name,
            view_id=view_id,
            target_grain=target_grain,
            target_ids=target_ids,
            scenario_ids=scenario_ids,
            as_of=as_of,
            transform_id=transform_id,
            snapshot_name=snapshot_name,
            superseded_policy=superseded_policy,
        )

    def trace_membership(
        self,
        snapshot_name: str,
        scenario_id: str,
        *,
        superseded_policy: str = "history",
    ) -> CurationMembershipTrace:
        """Explain why a scenario was included in or excluded from a snapshot."""
        return _trace_membership(
            self._lake,
            snapshot_name=snapshot_name,
            scenario_id=scenario_id,
            superseded_policy=superseded_policy,
        )

    def compile_row_plan(
        self,
        *,
        view_name: str,
        target_grain: str = "observation",
        source_snapshot_name: str | None = None,
        include_decisions: Sequence[str] = _ROW_PLAN_INCLUDE_DECISIONS,
        excluding_decisions: Sequence[str] = _ROW_PLAN_EXCLUDE_DECISIONS,
        freeze: bool = False,
        created_by: str = "lancedb-robotics",
    ) -> CurationCompiledRowPlan:
        """Compile a saved view's latest membership decisions into a row plan."""
        return self.view(view_name).compile_row_plan(
            view_name=view_name,
            target_grain=target_grain,
            source_snapshot_name=source_snapshot_name,
            include_decisions=include_decisions,
            excluding_decisions=excluding_decisions,
            freeze=freeze,
            created_by=created_by,
        )

    def from_gaps(
        self,
        gaps: Any,
        *,
        source: CurationScope | CurationView | dict | str | Sequence[str] | None = None,
        limit_per_gap: int | None = None,
        exclude_scenario_ids: Sequence[str] = (),
        created_by: str = "lancedb-robotics",
    ) -> CurationSelection:
        """Create a candidate selection from missing or underrepresented slices."""
        from lancedb_robotics.distributions import DistributionComparison, gap_findings_from

        if limit_per_gap is not None and limit_per_gap <= 0:
            raise CurationError("limit_per_gap must be positive")
        findings = tuple(
            finding
            for finding in gap_findings_from(gaps)
            if finding.kind in {"missing", "underrepresented"}
        )
        if not findings:
            raise CurationError("no missing or underrepresented gap findings to select from")

        source_selection = (
            source
            if isinstance(source, CurationSelection)
            else self.workbench(scope=source)
        )
        excluded = set(str(item) for item in exclude_scenario_ids if str(item))
        prior_transform_ids = source_selection.operation_transform_ids
        comparison_id = ""
        if isinstance(gaps, DistributionComparison):
            comparison_id = gaps.comparison_id
            prior_transform_ids = prior_transform_ids + (gaps.transform_id,)
            excluded.update(gaps.observed.scenario_ids)

        rows = _selected_rows(self._lake, source_selection.scenario_ids)
        run_rows = _run_rows(self._lake)
        selected: list[str] = []
        selected_by_gap: dict[str, list[str]] = {}
        for finding in findings:
            dimensions = _dimensions_from_gap_label(finding.label)
            if not dimensions:
                dimensions = tuple(key for key, _ in finding.values)
            matches = [
                row
                for row in rows
                if row["scenario_id"] not in excluded
                and _slice_label(row, run_rows.get(row["run_id"], {}), dimensions) == finding.label
            ]
            chosen = [
                row["scenario_id"]
                for row in sorted(matches, key=_scenario_sort_key)[:limit_per_gap]
            ]
            selected_by_gap[finding.label] = chosen
            for scenario_id in chosen:
                if scenario_id not in selected:
                    selected.append(scenario_id)

        if not selected:
            raise CurationError("gap selection found no candidate scenarios")

        report = {
            "operation": "from-gaps",
            "comparison_id": comparison_id,
            "gap_findings": [finding.to_dict() for finding in findings],
            "selected_by_gap": selected_by_gap,
            "limit_per_gap": limit_per_gap,
            "excluded_scenario_ids": sorted(excluded),
            "input_count": len(source_selection.scenario_ids),
            "output_count": len(selected),
        }
        transform_id = _record_curation_transform(
            self._lake,
            operation="from-gaps",
            input_scenario_ids=source_selection.scenario_ids,
            output_scenario_ids=selected,
            report=report,
            prior_transform_ids=prior_transform_ids,
            created_by=created_by,
        )
        return CurationSelection(
            lake=self._lake,
            scenario_ids=tuple(selected),
            scope=source_selection.scope,
            operation="from-gaps",
            transform_id=transform_id,
            report=report,
            operation_transform_ids=prior_transform_ids + (transform_id,),
        )

    def plan_comparison(
        self,
        left: str,
        right: str,
        *,
        by: Sequence[str] = (),
        metrics: Sequence[str] | None = None,
        plugins: Sequence[ComparisonMetricPlugin | str] | None = None,
        batch_size: int | None = None,
        local_row_budget: int | None = None,
    ) -> CurationComparisonPlan:
        """Plan a comparison before running it (backlog 0094).

        Returns required tables, current table versions, an estimated scan size,
        and whether each selected metric (built-in or plugin) can run locally or
        needs an external Ray/Batch/Slurm-style executor. No metric is computed
        and no report is persisted.
        """
        return _build_comparison_plan(
            self._lake,
            left,
            right,
            by=by,
            metrics=metrics,
            plugins=plugins,
            batch_size=batch_size,
            local_row_budget=local_row_budget,
        )

    def comparison_membership(
        self,
        id_or_name: str,
        *,
        field: str | None = None,
        offset: int = 0,
        limit: int | None = None,
        page_token: str | None = None,
    ) -> CurationComparisonMembershipPage:
        """Page one membership-delta id list for a persisted comparison (0094).

        Comparison reports cap inline id previews; full ``added``/``removed``/
        ``shared`` lists are paged through this deterministic handle (the
        snapshots are version-pinned, so pages reproduce exactly). Pass a
        ``page_token`` from a prior page or report preview, or an explicit
        ``field`` + ``offset``.
        """
        return _comparison_membership_page(
            self._lake,
            id_or_name,
            field_name=field,
            offset=offset,
            limit=limit,
            page_token=page_token,
        )

    def comparison_staleness(self, id_or_name: str) -> CurationComparisonStaleness:
        """Report whether a persisted comparison's source tables have advanced (0094)."""
        return _comparison_staleness(self._lake, id_or_name)

    def compare(
        self,
        left: str,
        right: str,
        *,
        by: Sequence[str] = (),
        metrics: Sequence[str] | None = None,
        plugins: Sequence[ComparisonMetricPlugin | str] | None = None,
        batch_size: int | None = None,
        preview_limit: int | None = None,
        created_by: str = "lancedb-robotics",
    ) -> CurationComparison:
        """Compare two dataset snapshots by membership, curation metrics, and lineage.

        Supported metrics execute through a bounded-memory streaming path
        (projection + filter pushdown, no full-table Python materialization).
        Long membership id lists are emitted as bounded previews with paging
        handles (see :meth:`comparison_membership`). ``plugins`` contribute
        optional report sections and thread their own lineage.
        """
        left_snapshot = _latest_snapshot_row(self._lake, left)
        right_snapshot = _latest_snapshot_row(self._lake, right)
        left_ids = _scenario_ids_from_snapshot_row(left_snapshot)
        right_ids = _scenario_ids_from_snapshot_row(right_snapshot)
        left_set = set(left_ids)
        right_set = set(right_ids)
        all_ids = tuple(sorted(left_set | right_set))
        normalized_metrics = _normalize_comparison_metrics(metrics)
        dimensions = tuple(dict.fromkeys(str(dim) for dim in by if str(dim)))
        resolved_plugins = resolve_comparison_plugins(plugins)
        resolved_batch_size = _comparison_batch_size(batch_size)
        resolved_preview_limit = _comparison_preview_limit(preview_limit)
        stats = _ComparisonExecutionStats(batch_size=resolved_batch_size)

        membership = _membership_diff(left_ids, right_ids)
        bounded_membership = _bounded_membership(membership, preview_limit=resolved_preview_limit)
        membership_preview = _membership_preview(membership, preview_limit=resolved_preview_limit)
        table_versions = _comparison_table_versions(self._lake, left_snapshot, right_snapshot)
        left_metrics = _model_output_metrics(
            self._lake, left_snapshot["dataset_id"], stats=stats, batch_size=resolved_batch_size
        )
        right_metrics = _model_output_metrics(
            self._lake, right_snapshot["dataset_id"], stats=stats, batch_size=resolved_batch_size
        )
        report: dict[str, Any] = {
            "operation": "compare-branches",
            "left": left,
            "right": right,
            "metrics": list(normalized_metrics),
            "dimensions": list(dimensions),
            "left_snapshot": _snapshot_report(left_snapshot, left_ids),
            "right_snapshot": _snapshot_report(right_snapshot, right_ids),
            "left_dataset_id": left_snapshot["dataset_id"],
            "right_dataset_id": right_snapshot["dataset_id"],
            "left_count": len(left_ids),
            "right_count": len(right_ids),
            "shared_count": membership["shared_count"],
            "left_only": bounded_membership["removed_scenario_ids"],
            "right_only": bounded_membership["added_scenario_ids"],
            "membership": bounded_membership,
            "membership_preview": membership_preview,
            "left_metrics": left_metrics,
            "right_metrics": right_metrics,
            "table_versions": table_versions,
        }

        need_coverage = "coverage" in normalized_metrics
        need_quality = "quality" in normalized_metrics
        need_label = "label-completeness" in normalized_metrics
        if need_coverage or need_quality or need_label:
            runs_lookup: dict[str, dict[str, Any]] = {}
            if need_coverage or need_quality:
                runs_lookup = _comparison_run_lookup(
                    self._lake, dimensions, stats=stats, batch_size=resolved_batch_size
                )
            scenario_pass = _comparison_scenario_pass(
                self._lake,
                union_ids=all_ids,
                left_set=left_set,
                right_set=right_set,
                runs=runs_lookup,
                dimensions=dimensions,
                need_coverage=need_coverage,
                need_quality=need_quality,
                need_label_map=need_label,
                stats=stats,
                batch_size=resolved_batch_size,
            )
        else:
            scenario_pass = None

        if need_coverage:
            coverage = _coverage_from_slices(
                scenario_pass["left_slices"],
                scenario_pass["right_slices"],
                len(left_ids),
                len(right_ids),
                dimensions,
            )
            report["coverage"] = coverage
            report["by"] = list(dimensions)
            report["left_slices"] = coverage["left_slices"]
            report["right_slices"] = coverage["right_slices"]
        if "source-overlap" in normalized_metrics:
            report["source_overlap"] = _source_overlap(self._lake, left_ids, right_ids)
        if "duplicate-pressure" in normalized_metrics:
            left_duplicate = _duplicate_pressure_summary(self._lake, left_ids)
            right_duplicate = _duplicate_pressure_summary(self._lake, right_ids)
            report["duplicate_pressure"] = {
                "left": left_duplicate,
                "right": right_duplicate,
                "delta": _numeric_delta(left_duplicate, right_duplicate),
            }
        if need_quality:
            left_quality = scenario_pass["quality_left"].to_summary()
            right_quality = scenario_pass["quality_right"].to_summary()
            report["quality"] = {
                "left": left_quality,
                "right": right_quality,
                "delta": _numeric_delta(left_quality, right_quality),
            }
        if need_label:
            observation_to_scenario = scenario_pass["observation_to_scenario"]
            left_labels = _streaming_label_completeness(
                self._lake,
                left_ids,
                observation_to_scenario,
                stats=stats,
                batch_size=resolved_batch_size,
                preview_limit=resolved_preview_limit,
            )
            right_labels = _streaming_label_completeness(
                self._lake,
                right_ids,
                observation_to_scenario,
                stats=stats,
                batch_size=resolved_batch_size,
                preview_limit=resolved_preview_limit,
            )
            report["label_completeness"] = {
                "left": left_labels,
                "right": right_labels,
                "delta": _numeric_delta(left_labels, right_labels),
            }
        if "payload" in normalized_metrics:
            left_payload = _payload_summary(self._lake, left_ids)
            right_payload = _payload_summary(self._lake, right_ids)
            report["payload"] = {
                "left": left_payload,
                "right": right_payload,
                "delta": _numeric_delta(left_payload, right_payload),
            }
        if "materialization" in normalized_metrics:
            left_materialization = _materialization_summary(
                self._lake, left_snapshot["dataset_id"], stats=stats, batch_size=resolved_batch_size
            )
            right_materialization = _materialization_summary(
                self._lake, right_snapshot["dataset_id"], stats=stats, batch_size=resolved_batch_size
            )
            report["materialization"] = {
                "left": left_materialization,
                "right": right_materialization,
                "delta": _numeric_delta(left_materialization, right_materialization),
            }
        if "training-eval" in normalized_metrics:
            left_eval = _downstream_training_eval(
                self._lake, left_snapshot["dataset_id"], stats=stats, batch_size=resolved_batch_size
            )
            right_eval = _downstream_training_eval(
                self._lake, right_snapshot["dataset_id"], stats=stats, batch_size=resolved_batch_size
            )
            report["training_eval"] = {
                "left": left_eval,
                "right": right_eval,
                "delta": _eval_metric_delta(left_eval, right_eval),
            }

        plugin_transform_ids: list[str] = []
        if resolved_plugins:
            ctx = ComparisonMetricContext(
                lake=self._lake,
                left=left,
                right=right,
                left_snapshot=left_snapshot,
                right_snapshot=right_snapshot,
                left_dataset_id=str(left_snapshot["dataset_id"]),
                right_dataset_id=str(right_snapshot["dataset_id"]),
                left_ids=tuple(left_ids),
                right_ids=tuple(right_ids),
                dimensions=dimensions,
                membership=membership,
                batch_size=resolved_batch_size,
                _scanner=_comparison_scanner_for(stats),
            )
            plugin_sections: dict[str, Any] = {}
            for plugin in resolved_plugins:
                section = plugin.compute(ctx)
                plugin_sections[plugin.name] = section
                plugin_transform_ids.extend(
                    str(item) for item in plugin.lineage_transform_ids(section) if str(item)
                )
            report["plugins"] = plugin_sections
            report["plugin_metrics"] = [plugin.name for plugin in resolved_plugins]

        non_streamed_metrics = [
            metric for metric in normalized_metrics if metric not in _COMPARISON_STREAMED_METRICS
        ]
        execution_report = stats.to_report()
        execution_report["non_streamed_metrics"] = non_streamed_metrics
        execution_report["preview_limit"] = resolved_preview_limit
        execution_report["bounded"] = (
            not non_streamed_metrics and not stats.materialized_tables
        )
        report["execution"] = execution_report

        comparison_id = "cmp-" + _digest(report)
        report["comparison_id"] = comparison_id
        for field_name in _COMPARISON_MEMBERSHIP_FIELDS:
            token = membership_preview[field_name]["next_page_token"]
            if token:
                membership_preview[field_name]["comparison_id"] = comparison_id
        prior_transform_ids = list(
            _comparison_prior_transform_ids(left_snapshot, right_snapshot, report)
        )
        prior_transform_ids.extend(plugin_transform_ids)
        prior_transform_ids = list(dict.fromkeys(item for item in prior_transform_ids if item))
        transform_id = _record_curation_transform(
            self._lake,
            operation="compare-branches",
            input_scenario_ids=tuple(left_ids) + tuple(right_ids),
            output_scenario_ids=all_ids,
            report=report,
            prior_transform_ids=tuple(prior_transform_ids),
            created_by=created_by,
            output_tables=("curation_comparisons",),
        )
        report["transform_id"] = transform_id
        _persist_comparison_report(
            self._lake,
            comparison_id=comparison_id,
            left_snapshot=left_snapshot,
            right_snapshot=right_snapshot,
            metrics=normalized_metrics,
            dimensions=dimensions,
            membership=membership,
            report=report,
            table_versions=table_versions,
            transform_id=transform_id,
            created_by=created_by,
        )
        return CurationComparison(
            comparison_id=comparison_id,
            left=left,
            right=right,
            report=report,
            transform_id=transform_id,
        )

    def diff_snapshots(
        self,
        left: str,
        right: str,
        *,
        created_by: str = "lancedb-robotics",
    ) -> CurationComparison:
        """Return and record the reproducible selected-ID diff between two snapshots."""
        return self.compare(
            left,
            right,
            metrics=("membership",),
            created_by=created_by,
        )

    def list_comparisons(
        self,
        *,
        snapshot: str | None = None,
        left: str | None = None,
        right: str | None = None,
        metric: str | None = None,
        state: str | None = None,
        include_archived: bool = True,
        include_pruned: bool = True,
        since: datetime | None = None,
        until: datetime | None = None,
        created_by: str | None = None,
        limit: int | None = None,
    ) -> tuple[CurationComparisonEntry, ...]:
        """List persisted comparison reports from the catalog, newest first.

        Filter by ``snapshot`` (matches either side by name or dataset id),
        explicit ``left``/``right`` snapshot, requested ``metric``, lifecycle
        ``state``, ``created_by``, or a ``since``/``until`` window. This reads
        the catalog directly instead of scanning transform params.
        """
        return _list_comparison_catalog(
            self._lake,
            snapshot=snapshot,
            left=left,
            right=right,
            metric=metric,
            state=state,
            include_archived=include_archived,
            include_pruned=include_pruned,
            since=since,
            until=until,
            created_by=created_by,
            limit=limit,
        )

    def comparison(self, id_or_name: str) -> CurationComparison:
        """Reload a persisted comparison by ``comparison_id`` or snapshot-pair alias.

        A ``comparison_id`` returns that exact report. Any other token is treated
        as a snapshot-pair alias — either ``"<left>..<right>"`` or a snapshot name
        present on one side — and resolves to the newest non-pruned comparison
        for that pair, so callers can reopen "the latest comparison" without
        scanning transform params. The returned report JSON matches the shape
        emitted by ``curate compare --json``. Reloading a pruned report raises a
        :class:`CurationError` that points at the surviving audit metadata.
        """
        return _resolve_comparison(self._lake, id_or_name)

    def prune_comparisons(
        self,
        *,
        retain_latest: int = 1,
        older_than: datetime | timedelta | None = None,
        dry_run: bool = False,
        created_by: str = "lancedb-robotics",
    ) -> CurationComparisonRetentionReport:
        """Apply the comparison retention lifecycle (active -> archived -> pruned).

        Within each snapshot pair the newest ``retain_latest`` reports stay
        ``active``. Older reports are ``archived`` (body retained and still
        reloadable) unless they predate ``older_than``, in which case the body is
        ``pruned`` to bound the operational table. Pruning preserves the
        comparison id, snapshot ids, table versions, transform id, counts, and
        body digest, and records an audit transform. With no ``older_than`` no
        body is dropped, so the default call is a safe soft-retire.
        """
        return _prune_comparison_catalog(
            self._lake,
            retain_latest=retain_latest,
            older_than=older_than,
            dry_run=dry_run,
            created_by=created_by,
        )

    def list_eval_metrics(
        self,
        *,
        snapshot: str | None = None,
        evaluation_run: str | None = None,
        training_run: str | None = None,
        model_version: str | None = None,
        metric: str | None = None,
        slice_label: str | None = None,
        regressed_only: bool = False,
        state: str | None = None,
        latest_only: bool = False,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> EvalMetricListing:
        """List imported eval metrics from the catalog, newest first (0095).

        Filters push down to indexed ``eval_metric_catalog`` columns — listing
        never scans ``model_outputs`` or parses metric JSON in Python.
        ``snapshot`` matches the snapshot name or dataset id; ``latest_only``
        keeps the newest import per (snapshot, model version, metric, slice)
        series. Results carry a stable ``total_count`` and are capped at
        ``limit`` (default 100) for bounded previews.
        """
        return _list_eval_metric_catalog(
            self._lake,
            snapshot=snapshot,
            evaluation_run=evaluation_run,
            training_run=training_run,
            model_version=model_version,
            metric=metric,
            slice_label=slice_label,
            regressed_only=regressed_only,
            state=state,
            latest_only=latest_only,
            since=since,
            until=until,
            limit=limit,
        )

    def eval_metric_staleness(self, id_or_run: str) -> EvalMetricStaleness:
        """Report whether an eval metric's source tables have advanced (0095).

        ``id_or_run`` is a ``model_output_id`` (checks that exact metric) or an
        ``evaluation_run_id`` (checks the newest metric imported for that run —
        all metrics of one import share the same recorded table versions).
        Staleness compares the snapshot/curation/model-output/feedback versions
        recorded at import time against the current lake.
        """
        return _eval_metric_staleness(self._lake, id_or_run)

    def prune_eval_metrics(
        self,
        *,
        retain_latest: int = 1,
        older_than: datetime | timedelta | None = None,
        dry_run: bool = False,
        created_by: str = "lancedb-robotics",
    ) -> EvalMetricRetentionReport:
        """Apply the eval-metric retention lifecycle (active -> superseded -> pruned).

        Within each (snapshot, model version, metric, slice) series the newest
        ``retain_latest`` imports stay ``active``. Older imports are marked
        ``superseded`` (source rows retained). Superseded imports that predate
        ``older_than`` are ``pruned``: the ``model_outputs`` source row is
        deleted and the catalog row survives as audit metadata. Metrics
        referenced by a snapshot promotion decision (by ``model_output_id`` or
        ``evaluation_run_id``) are protected and never pruned. With no
        ``older_than`` nothing is deleted, so the default call is a safe
        soft-retire; ``dry_run`` reports the same sets without writing.
        """
        return _prune_eval_metric_catalog(
            self._lake,
            retain_latest=retain_latest,
            older_than=older_than,
            dry_run=dry_run,
            created_by=created_by,
        )

    def sync_eval_metric_catalog(
        self,
        *,
        build_indexes: bool = True,
        created_by: str = "lancedb-robotics",
    ) -> EvalMetricCatalogSyncReport:
        """Rebuild ``eval_metric_catalog`` from ``model_outputs`` (0095).

        Streams the eval-import metric rows in bounded batches, re-derives the
        catalog rows deterministically, recomputes series lifecycle state, and
        preserves pruned-entry audit metadata for source rows that retention
        already deleted. Run once on a lake that imported metrics before 0095,
        or any time to repair catalog drift; imports via
        :meth:`feedback_from_eval` keep the catalog current automatically.
        """
        return _sync_eval_metric_catalog(
            self._lake,
            build_indexes=build_indexes,
            created_by=created_by,
        )

    def feedback_from_eval(
        self,
        snapshot_name: str,
        *,
        metrics: Iterable[dict[str, Any]] | dict[str, Any],
        training_run_id: str = "",
        model_version: str = "",
        evaluation_run_id: str = "",
        regression_threshold: float = 0.0,
        created_by: str = "lancedb-robotics",
    ) -> CurationFeedbackReport:
        """Import evaluation metrics and link them to an exact curated snapshot.

        Metrics are stored as deterministic ``model_outputs`` rows with the
        snapshot ``dataset_id``, training/eval identifiers, slice metadata, and
        regression classification. Re-running the same import replaces the same
        metric rows rather than creating duplicate eval records.
        """
        if regression_threshold < 0:
            raise CurationError("regression_threshold must be non-negative")
        snapshot = _latest_snapshot_row(self._lake, snapshot_name)
        scenario_ids = _scenario_ids_from_snapshot_row(snapshot)
        metric_rows, regressions, metric_payloads = _eval_metric_rows(
            self._lake,
            snapshot=snapshot,
            metrics=metrics,
            training_run_id=training_run_id,
            model_version=model_version,
            evaluation_run_id=evaluation_run_id,
            regression_threshold=regression_threshold,
        )
        metric_output_ids = tuple(row["model_output_id"] for row in metric_rows)
        prior_transform_ids = _snapshot_prior_transform_ids(snapshot)
        report = {
            "operation": "feedback-from-eval",
            "snapshot_name": snapshot["name"],
            "dataset_id": snapshot["dataset_id"],
            "snapshot_tag": snapshot.get("tag") or "",
            "training_run_id": training_run_id,
            "model_version": model_version,
            "evaluation_run_id": evaluation_run_id,
            "regression_threshold": regression_threshold,
            "metric_output_ids": list(metric_output_ids),
            "metric_count": len(metric_output_ids),
            "metrics": metric_payloads,
            "regression_count": len(regressions),
            "regressions": list(regressions),
        }
        transform_id = _record_feedback_loop_transform(
            self._lake,
            operation="feedback-from-eval",
            input_scenario_ids=scenario_ids,
            output_scenario_ids=scenario_ids,
            report=report,
            prior_transform_ids=prior_transform_ids,
            output_tables=("model_outputs", "eval_metric_catalog"),
            created_by=created_by,
        )
        metric_payloads = [{**payload, "transform_id": transform_id} for payload in metric_payloads]
        regressions = tuple({**regression, "transform_id": transform_id} for regression in regressions)
        now = datetime.now(UTC)
        for row in metric_rows:
            row["transform_id"] = transform_id
            row["created_at"] = now
        _replace_model_output_rows(self._lake, metric_rows)
        table_versions = tuple(
            (str(item["table"]), int(item["version"]))
            for item in _table_versions(self._lake, tables=_FEEDBACK_LOOP_TABLES)
        )
        _upsert_eval_metric_catalog(
            self._lake,
            payloads=metric_payloads,
            snapshot_tag=str(snapshot.get("tag") or ""),
            transform_id=transform_id,
            created_by=created_by,
            now=now,
            table_versions=table_versions,
        )
        return CurationFeedbackReport(
            snapshot_name=str(snapshot["name"]),
            dataset_id=str(snapshot["dataset_id"]),
            training_run_id=training_run_id,
            model_version=model_version,
            evaluation_run_id=evaluation_run_id,
            metric_output_ids=metric_output_ids,
            regressions=regressions,
            transform_id=transform_id,
            table_versions=table_versions,
            report={
                **report,
                "metrics": metric_payloads,
                "regressions": list(regressions),
                "transform_id": transform_id,
            },
        )

    def plan_next_candidates(
        self,
        *,
        from_regressions: Any,
        source: CurationScope | CurationView | dict | str | Sequence[str] | None = None,
        queue_name: str | None = None,
        view_name: str | None = None,
        snapshot_name: str | None = None,
        snapshot_tag: str | None = None,
        limit_per_regression: int = 50,
        create_queue: bool = True,
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        route: str = _FEEDBACK_CANDIDATE_ROUTE_AUTO,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        preview_limit: int = _FEEDBACK_CANDIDATE_PREVIEW_LIMIT,
    ) -> FeedbackCandidatePlan:
        """Explain regression-seeded candidate generation without writing anything (0096).

        Accepts the same candidate-selection kwargs as :meth:`next_candidates`
        and returns the :class:`FeedbackCandidatePlan` that an apply would
        execute: regression identities, source scope, ordered stages with row
        counts, index requirements (met/unmet with the exact build remedy),
        pinned source table versions, expected artifact identities, complete
        candidate counts, and a bounded preview. No queue/view/snapshot rows,
        membership rows, or lineage transforms are written.
        """
        plan, _, _ = self._plan_feedback_candidates(
            from_regressions=from_regressions,
            source=source,
            queue_name=queue_name,
            view_name=view_name,
            snapshot_name=snapshot_name,
            snapshot_tag=snapshot_tag,
            limit_per_regression=limit_per_regression,
            create_queue=create_queue,
            embedding_column=embedding_column,
            route=route,
            nprobes=nprobes,
            refine_factor=refine_factor,
            preview_limit=preview_limit,
        )
        return plan

    def preview_candidates(
        self,
        plan: FeedbackCandidatePlan,
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> FeedbackCandidatePreviewPage:
        """Page a plan's complete candidate selection deterministically (0096).

        Ordering is (regression order in the plan, per-regression candidate
        ordinal); the cursor is a stable reload handle in the review-queue
        cursor style, so identical plans (same ``plan_id``) page identically
        across processes. Paging never changes the plan's complete counts.
        """
        page_limit = int(limit if limit is not None else plan.preview_limit)
        if page_limit <= 0:
            raise CurationError("preview limit must be positive")
        entries: list[tuple[int, dict[str, Any]]] = []
        for key_index, (regression_key, scenario_ids) in enumerate(
            plan.selected_by_regression.items()
        ):
            for ordinal, scenario_id in enumerate(scenario_ids):
                entries.append(
                    (
                        key_index,
                        {
                            "regression_key": regression_key,
                            "ordinal": ordinal,
                            "scenario_id": scenario_id,
                        },
                    )
                )
        start = 0
        decoded = _decode_feedback_preview_cursor(cursor)
        if decoded is not None:
            key_order = {key: idx for idx, key in enumerate(plan.selected_by_regression)}
            if decoded["regression_key"] not in key_order:
                raise CurationError("feedback preview cursor does not match this plan")
            resume_after = (key_order[decoded["regression_key"]], decoded["ordinal"])
            start = len(entries)
            for index, (key_index, row) in enumerate(entries):
                if (key_index, row["ordinal"]) > resume_after:
                    start = index
                    break
        rows = tuple(row for _, row in entries[start : start + page_limit])
        has_more = len(entries) > start + page_limit
        next_cursor = _encode_feedback_preview_cursor(rows[-1]) if has_more and rows else ""
        return FeedbackCandidatePreviewPage(
            plan_id=plan.plan_id,
            rows=rows,
            limit=page_limit,
            cursor=str(cursor or ""),
            next_cursor=next_cursor,
            has_more=has_more,
            total_count=len(entries),
        )

    def _plan_feedback_candidates(
        self,
        *,
        from_regressions: Any,
        source: CurationScope | CurationView | dict | str | Sequence[str] | None,
        queue_name: str | None,
        view_name: str | None,
        snapshot_name: str | None,
        snapshot_tag: str | None,
        limit_per_regression: int,
        create_queue: bool,
        embedding_column: str,
        route: str,
        nprobes: int | None,
        refine_factor: int | None,
        preview_limit: int,
    ) -> tuple[FeedbackCandidatePlan, tuple[dict[str, Any], ...], CurationSelection]:
        if limit_per_regression <= 0:
            raise CurationError("limit_per_regression must be positive")
        if preview_limit <= 0:
            raise CurationError("preview_limit must be positive")
        regressions = _regressions_from(from_regressions, self._lake)
        if not regressions:
            raise CurationError("no eval regressions supplied")
        source_selection = self.workbench(scope=source)
        _require_embedding_column(self._lake, embedding_column)
        route_state = _resolve_feedback_candidate_route(
            self._lake,
            embedding_column=embedding_column,
            pool_size=len(source_selection.scenario_ids),
            route=route,
            nprobes=nprobes,
            refine_factor=refine_factor,
        )
        source_set = set(source_selection.scenario_ids)

        regression_keys = tuple(
            _regression_key(regression, index) for index, regression in enumerate(regressions)
        )
        selected: list[str] = []
        selected_by_regression: dict[str, tuple[str, ...]] = {}
        stage_infos: list[dict[str, Any]] = []
        if route_state["runnable"]:
            for index, regression in enumerate(regressions):
                candidates, stage_info = _candidate_ids_for_regression(
                    self._lake,
                    source_selection,
                    regression,
                    limit=limit_per_regression,
                    embedding_column=embedding_column,
                    use_index=route_state["use_index"],
                    nprobes=route_state["nprobes"],
                    refine_factor=route_state["refine_factor"],
                )
                candidates = [
                    scenario_id for scenario_id in candidates if scenario_id in source_set
                ]
                stage_infos.append(
                    {
                        **stage_info,
                        "regression_key": regression_keys[index],
                        "candidate_count": len(candidates),
                    }
                )
                if not candidates:
                    continue
                selected_by_regression[regression_keys[index]] = tuple(candidates)
                for scenario_id in candidates:
                    if scenario_id not in selected:
                        selected.append(scenario_id)

        candidate_counts = {key: len(ids) for key, ids in selected_by_regression.items()}
        total_candidate_count = sum(candidate_counts.values())
        candidate_digest = (
            _digest(
                {
                    "selected_by_regression": {
                        key: list(ids) for key, ids in selected_by_regression.items()
                    },
                    "selected": list(selected),
                }
            )
            if route_state["runnable"] and selected
            else ""
        )
        membership_digest = (
            _digest({"scenario_ids": sorted(selected)}) if selected else ""
        )

        regression_identities = tuple(
            _regression_identity(regression) for regression in regressions
        )
        resolved_queue_name = (
            (queue_name or _default_regression_queue_name(regressions)) if create_queue else ""
        )
        resolved_snapshot_tag = (
            str(snapshot_tag or snapshot_name or "") if snapshot_name else ""
        )
        scope_identity = {
            "scope": _jsonable(source_selection.scope),
            "operation": source_selection.operation,
            "scenario_count": len(source_selection.scenario_ids),
            "scenario_ids_digest": _digest(
                {"scenario_ids": sorted(source_set)}
            ),
        }
        # Identity deliberately excludes table versions (audit-only) and the
        # display-only preview limit: re-planning after unrelated churn or with
        # a different preview cap yields the same plan_id.
        plan_id = "fplan-" + _digest(
            {
                "operation": "feedback-regression-candidates",
                "regressions": [dict(item) for item in regression_identities],
                "source": scope_identity,
                "stage_config": {
                    "limit_per_regression": limit_per_regression,
                    "embedding_column": embedding_column,
                    "route": route,
                    "nprobes": nprobes,
                    "refine_factor": refine_factor,
                },
                "artifacts": {
                    "queue_name": resolved_queue_name,
                    "view_name": view_name or "",
                    "snapshot_name": snapshot_name or "",
                    "snapshot_tag": resolved_snapshot_tag,
                    "create_queue": bool(create_queue),
                },
            }
        )

        expected_artifacts: dict[str, Any] = {
            "queue": (
                {
                    "name": resolved_queue_name,
                    "target_grain": "scenario",
                    "queue_id": "queue-"
                    + _digest(
                        _jsonable(_feedback_queue_identity(plan_id, resolved_queue_name))
                    ),
                }
                if create_queue
                else None
            ),
            "view": (
                {"name": view_name, "membership_digest": membership_digest}
                if view_name
                else None
            ),
            "snapshot": (
                {
                    "name": snapshot_name,
                    "tag": resolved_snapshot_tag,
                    "membership_digest": membership_digest,
                }
                if snapshot_name
                else None
            ),
        }

        pool_size = len(source_selection.scenario_ids)
        estimated = not route_state["runnable"]
        if route_state["runnable"]:
            mining_rows = sum(
                info["candidate_count"] for info in stage_infos if not info["slice_fill"]
            )
            slice_rows = sum(
                info["candidate_count"] for info in stage_infos if info["slice_fill"]
            )
            dedup_rows = len(selected)
        else:
            mining_rows = min(limit_per_regression, pool_size) * len(regressions)
            slice_rows = 0
            dedup_rows = mining_rows
        stages = (
            {
                "stage": "failure-mining",
                "strategy": "seed-neighbor-search",
                "route": route_state["effective"],
                "estimated_rows": mining_rows,
                "estimated": estimated,
                "regression_count": len(regressions),
            },
            {
                "stage": "gap-analysis",
                "strategy": "slice-fill-fallback",
                "estimated_rows": slice_rows,
                "estimated": estimated,
                "regression_count": sum(
                    1 for info in stage_infos if info.get("slice_fill")
                ),
            },
            {
                "stage": "dedup-diversity",
                "strategy": "cross-regression-first-seen-dedup",
                "estimated_rows": dedup_rows,
                "estimated": estimated,
                "regression_count": len(selected_by_regression),
            },
        )
        index_requirements = (
            {
                "table": "scenarios",
                "column": embedding_column,
                "kind": "vector",
                "required": route_state["index_required"],
                "met": route_state["index_met"],
                "remedy": route_state["remedy"],
            },
        )
        preview_rows: list[dict[str, Any]] = []
        for regression_key, scenario_ids in selected_by_regression.items():
            for ordinal, scenario_id in enumerate(scenario_ids[:preview_limit]):
                preview_rows.append(
                    {
                        "regression_key": regression_key,
                        "ordinal": ordinal,
                        "scenario_id": scenario_id,
                    }
                )
        plan = FeedbackCandidatePlan(
            plan_id=plan_id,
            regressions=regressions,
            regression_keys=regression_keys,
            regression_identities=regression_identities,
            source_scope=scope_identity,
            source_scenario_count=pool_size,
            stages=stages,
            index_requirements=index_requirements,
            table_versions=tuple(
                _table_versions(self._lake, tables=_FEEDBACK_LOOP_TABLES)
            ),
            route=route,
            effective_route=route_state["effective"],
            route_reason=route_state["reason"],
            nprobes=route_state["nprobes"],
            refine_factor=route_state["refine_factor"],
            limit_per_regression=limit_per_regression,
            preview_limit=preview_limit,
            embedding_column=embedding_column,
            expected_artifacts=expected_artifacts,
            selected_by_regression=selected_by_regression,
            selected=tuple(selected),
            candidate_counts=candidate_counts,
            total_candidate_count=total_candidate_count,
            candidate_digest=candidate_digest,
            preview=tuple(preview_rows),
            runnable=route_state["runnable"],
        )
        return plan, regressions, source_selection

    def next_candidates(
        self,
        *,
        from_regressions: Any,
        source: CurationScope | CurationView | dict | str | Sequence[str] | None = None,
        queue_name: str | None = None,
        view_name: str | None = None,
        snapshot_name: str | None = None,
        snapshot_tag: str | None = None,
        limit_per_regression: int = 50,
        create_queue: bool = True,
        embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
        assignee: str = "",
        status: str = "open",
        route: str = _FEEDBACK_CANDIDATE_ROUTE_AUTO,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        preview_limit: int = _FEEDBACK_CANDIDATE_PREVIEW_LIMIT,
        dry_run: bool = False,
        created_by: str = "lancedb-robotics",
    ) -> CurationCandidateReport:
        """Turn eval regressions into the next review queue, saved view, or snapshot.

        Backlog 0096 additions: candidate search follows ``route`` (``auto``
        rides a persistent vector index on ``scenarios.embedding_column`` when
        one exists, ``exact`` forces the in-memory scan, ``ann`` requires the
        index); over an unindexed pool larger than the exact-scan limit the
        call refuses with the index-build remedy from the plan. Artifact
        writes are idempotent — the queue identity is keyed on the plan id
        (stable across table-version churn), and an unchanged queue/view/
        snapshot is reused instead of rewritten. ``dry_run`` computes the full
        plan, candidate selection, and artifact identities but writes nothing.
        """
        plan, regressions, source_selection = self._plan_feedback_candidates(
            from_regressions=from_regressions,
            source=source,
            queue_name=queue_name,
            view_name=view_name,
            snapshot_name=snapshot_name,
            snapshot_tag=snapshot_tag,
            limit_per_regression=limit_per_regression,
            create_queue=create_queue,
            embedding_column=embedding_column,
            route=route,
            nprobes=nprobes,
            refine_factor=refine_factor,
            preview_limit=preview_limit,
        )
        unmet = plan.unmet_index_requirements
        if unmet:
            raise CurationError(unmet[0]["remedy"])

        selected = list(plan.selected)
        selected_by_regression = {
            key: list(ids) for key, ids in plan.selected_by_regression.items()
        }
        if not selected:
            raise CurationError("eval regressions did not resolve to candidate scenarios")

        source_refs: dict[str, dict[str, Any]] = {}
        priority_scores: dict[str, float] = {}
        priority_reasons: dict[str, str] = {}
        for index, regression in enumerate(regressions):
            regression_key = plan.regression_keys[index]
            for scenario_id in selected_by_regression.get(regression_key, ()):
                score = _regression_priority_score(regression)
                priority_scores[scenario_id] = max(priority_scores.get(scenario_id, score), score)
                priority_reasons.setdefault(scenario_id, _regression_priority_reason(regression))
                source_refs.setdefault(
                    scenario_id,
                    {
                        "kind": "eval-regression",
                        "scenario_id": scenario_id,
                        "regression": _jsonable(regression),
                        "snapshot_name": regression.get("snapshot_name") or "",
                        "dataset_id": regression.get("dataset_id") or "",
                        "training_run_id": regression.get("training_run_id") or "",
                        "evaluation_run_id": regression.get("evaluation_run_id") or "",
                        "model_version": regression.get("model_version") or "",
                        "source_model_output_id": regression.get("source_model_output_id") or "",
                    },
                )

        prior_transform_ids = tuple(
            dict.fromkeys(
                (
                    *source_selection.operation_transform_ids,
                    *_regression_transform_ids(regressions),
                )
            )
        )
        # Transform-digest report: stable inputs only (no table versions, no
        # dry_run flag), so dry-run and apply share the same transform id and
        # identical re-runs replace rather than duplicate the lineage row.
        report = {
            "operation": "feedback-regression-candidates",
            "regression_count": len(regressions),
            "regressions": [dict(regression) for regression in regressions],
            "selected_by_regression": selected_by_regression,
            "limit_per_regression": limit_per_regression,
            "source_operation": source_selection.operation,
            "input_count": len(source_selection.scenario_ids),
            "output_count": len(selected),
            "plan_id": plan.plan_id,
            "candidate_digest": plan.candidate_digest,
            "route": plan.effective_route,
        }
        if dry_run:
            transform_id = _feedback_loop_transform_id(
                operation="feedback-regression-candidates",
                input_scenario_ids=source_selection.scenario_ids,
                output_scenario_ids=selected,
                report=report,
                prior_transform_ids=prior_transform_ids,
                output_tables=(),
            )
        else:
            transform_id = _record_feedback_loop_transform(
                self._lake,
                operation="feedback-regression-candidates",
                input_scenario_ids=source_selection.scenario_ids,
                output_scenario_ids=selected,
                report=report,
                prior_transform_ids=prior_transform_ids,
                output_tables=(),
                created_by=created_by,
            )
        selection = CurationSelection(
            lake=self._lake,
            scenario_ids=tuple(selected),
            scope=source_selection.scope,
            operation="feedback-regressions",
            transform_id=transform_id,
            report={**report, "transform_id": transform_id},
            operation_transform_ids=prior_transform_ids + (transform_id,),
        )

        artifact_writes: dict[str, str] = {}
        view = None
        snapshot = None
        queue = None
        resolved_queue_name = (
            (plan.expected_artifacts["queue"] or {}).get("name", "") if create_queue else ""
        )
        if dry_run:
            if create_queue:
                artifact_writes["queue"] = "skipped"
            if view_name:
                artifact_writes["view"] = "skipped"
            if snapshot_name:
                artifact_writes["snapshot"] = "skipped"
        else:
            if view_name:
                view = _reuse_feedback_view(self._lake, view_name, selected)
                if view is not None:
                    artifact_writes["view"] = "unchanged"
                else:
                    view = selection.save_view(
                        view_name,
                        tags=("eval-regression",),
                        description="Evaluation regression candidates for the next curation pass.",
                        created_by=created_by,
                    )
                    artifact_writes["view"] = "written"
            if snapshot_name:
                resolved_tag = str(snapshot_tag or snapshot_name)
                snapshot = _reuse_feedback_snapshot(
                    self._lake,
                    name=snapshot_name,
                    tag=resolved_tag,
                    scenario_ids=selected,
                )
                if snapshot is not None:
                    artifact_writes["snapshot"] = "unchanged"
                else:
                    snapshot = selection.snapshot(
                        name=snapshot_name, tag=snapshot_tag, created_by=created_by
                    )
                    artifact_writes["snapshot"] = "written"
            if create_queue:
                queue = selection.to_review_queue(
                    resolved_queue_name,
                    source_operation="eval-regression",
                    priority_scores=priority_scores,
                    priority_reasons=priority_reasons,
                    source_refs=source_refs,
                    assignee=assignee,
                    status=status,
                    metadata={
                        "regression_count": len(regressions),
                        "candidate_transform_id": transform_id,
                    },
                    identity_payload=_feedback_queue_identity(
                        plan.plan_id, resolved_queue_name
                    ),
                    created_by=created_by,
                )
                artifact_writes["queue"] = queue.write_status

        full_report = {
            **report,
            "transform_id": transform_id,
            "dry_run": dry_run,
            "expected_artifacts": _jsonable(plan.expected_artifacts),
            "artifact_writes": artifact_writes,
            "plan": plan.to_dict(),
        }
        return CurationCandidateReport(
            selection=selection,
            regressions=regressions,
            transform_id=transform_id,
            report=full_report,
            queue=queue,
            view=view,
            snapshot=snapshot,
            plan=plan,
            dry_run=dry_run,
        )

    def promote_snapshot(
        self,
        snapshot_name: str,
        *,
        decision: str = "promote",
        reason: str,
        reason_code: str = "",
        reviewer: str = "",
        evaluation_run_id: str = "",
        training_run_id: str = "",
        model_version: str = "",
        metrics: Iterable[dict[str, Any]] | dict[str, Any] | None = None,
        comparison_id: str = "",
        created_by: str = "lancedb-robotics",
    ) -> CurationPromotionDecision:
        """Record a promotion or rejection decision for a candidate snapshot."""
        normalized_decision = _normalize_promotion_decision(decision)
        if not str(reason).strip():
            raise CurationError("promotion/rejection reason must not be empty")
        snapshot = _latest_snapshot_row(self._lake, snapshot_name)
        scenario_ids = _scenario_ids_from_snapshot_row(snapshot)
        metric_payloads = _promotion_metric_payload(metrics)
        prior_transform_ids = _snapshot_prior_transform_ids(snapshot) + tuple(
            _collect_transform_ids(metric_payloads)
        )
        selection = CurationSelection(
            lake=self._lake,
            scenario_ids=scenario_ids,
            scope={"snapshot_name": snapshot_name, "dataset_id": snapshot["dataset_id"]},
            operation="snapshot-promotion",
            transform_id=str(snapshot.get("transform_id") or ""),
            report=_snapshot_report(snapshot, scenario_ids),
            operation_transform_ids=tuple(dict.fromkeys(prior_transform_ids)),
        )
        view_name = f"snapshot-decision-{snapshot_name}"
        metadata = {
            "snapshot_name": snapshot["name"],
            "dataset_id": snapshot["dataset_id"],
            "snapshot_tag": snapshot.get("tag") or "",
            "decision": normalized_decision,
            "evaluation_run_id": evaluation_run_id,
            "training_run_id": training_run_id,
            "model_version": model_version,
            "comparison_id": comparison_id,
            "metrics": metric_payloads,
            "snapshot_transform_id": snapshot.get("transform_id") or "",
            "snapshot_table_versions": _version_rows(snapshot.get("table_versions") or ()),
        }
        decisions = selection.record_decisions(
            decision=normalized_decision,
            target_grain="snapshot-row",
            target_ids=[str(snapshot["dataset_id"])],
            view_name=view_name,
            reason=reason,
            reason_code=reason_code or normalized_decision,
            note=reason,
            reviewer=reviewer,
            metadata=metadata,
            source="human",
            created_by=created_by,
        )
        report = {
            "operation": "promote-snapshot",
            "snapshot_name": snapshot["name"],
            "dataset_id": snapshot["dataset_id"],
            "decision": normalized_decision,
            "reason": reason,
            "evaluation_run_id": evaluation_run_id,
            "training_run_id": training_run_id,
            "model_version": model_version,
            "comparison_id": comparison_id,
            "metrics": metric_payloads,
            "membership_ids": list(decisions.membership_ids),
            "transform_id": decisions.transform_id,
        }
        return CurationPromotionDecision(
            snapshot_name=str(snapshot["name"]),
            dataset_id=str(snapshot["dataset_id"]),
            decision=normalized_decision,
            reason=reason,
            view=decisions.view,
            membership_ids=decisions.membership_ids,
            transform_id=decisions.transform_id,
            report=report,
        )

    def materialization_report(
        self,
        snapshot_name: str,
        *,
        target_format: str,
        output_uri: str = "",
        mode: str = "projection",
        copied_payload_bytes: int = 0,
        metadata_bytes_written: int = 0,
        planned_payload_bytes: int = 0,
        projection_transform_id: str = "",
        created_by: str = "lancedb-robotics",
    ) -> CurationMaterializationReport:
        """Record byte/copy accounting for a logical snapshot boundary projection."""
        if copied_payload_bytes < 0:
            raise CurationError("copied_payload_bytes must be non-negative")
        if metadata_bytes_written < 0:
            raise CurationError("metadata_bytes_written must be non-negative")
        if planned_payload_bytes < 0:
            raise CurationError("planned_payload_bytes must be non-negative")
        snapshot = _latest_snapshot_row(self._lake, snapshot_name)
        scenario_ids = _scenario_ids_from_snapshot_row(snapshot)
        observation_rows = _observations_for_scenarios(self._lake, scenario_ids)
        total_payload_bytes = sum(_payload_size(row.get("payload_blob")) for row in observation_rows)
        logical_reference_bytes = max(0, total_payload_bytes - copied_payload_bytes)
        copy_ratio = (
            copied_payload_bytes / total_payload_bytes
            if total_payload_bytes
            else 0.0
        )
        payload_copy_policy = (
            "would-copy-payloads"
            if mode in {"plan", "dry-run"} and planned_payload_bytes
            else (
                "materialized-copy"
                if copied_payload_bytes
                else "logical-reference"
            )
        )
        materialization_id = "mat-" + _digest(
            {
                "dataset_id": snapshot["dataset_id"],
                "target_format": target_format,
                "output_uri": output_uri,
                "mode": mode,
                "copied_payload_bytes": copied_payload_bytes,
                "metadata_bytes_written": metadata_bytes_written,
                "projection_transform_id": projection_transform_id,
            }
        )
        source_table_versions = normalize_table_versions(snapshot.get("table_versions") or ())
        accounting = ProjectionAccounting(
            logical_row_count=len(observation_rows),
            selected_scenario_count=len(scenario_ids),
            selected_observation_count=len(observation_rows),
            payload_bytes_referenced=total_payload_bytes,
            payload_bytes_copied=copied_payload_bytes,
            metadata_bytes_written=metadata_bytes_written,
            target_format=target_format,
            target_path=output_uri,
            projection_transform_id=projection_transform_id,
            source_snapshot_id=str(snapshot["dataset_id"]),
            source_snapshot_name=str(snapshot["name"]),
            source_table_versions=source_table_versions,
            mode=mode,
            payload_copy_policy=payload_copy_policy,
            dry_run=mode in {"plan", "dry-run"},
            payload_bytes_planned=planned_payload_bytes,
        ).to_dict()
        report = {
            "operation": "materialization-report",
            "dataset_id": snapshot["dataset_id"],
            "snapshot_name": snapshot["name"],
            "target_format": target_format,
            "output_uri": output_uri,
            "mode": mode,
            "logical_row_count": len(observation_rows),
            "selected_scenario_count": len(scenario_ids),
            "selected_observation_count": len(observation_rows),
            "total_payload_bytes": total_payload_bytes,
            "payload_bytes_referenced": total_payload_bytes,
            "copied_payload_bytes": copied_payload_bytes,
            "payload_bytes_copied": copied_payload_bytes,
            "planned_payload_bytes": planned_payload_bytes,
            "payload_bytes_planned": planned_payload_bytes,
            "logical_reference_bytes": logical_reference_bytes,
            "metadata_bytes_written": metadata_bytes_written,
            "copy_ratio": copy_ratio,
            "projection_transform_id": projection_transform_id,
            "source_table_versions": list(source_table_versions),
            "accounting": accounting,
        }
        transform_id = _record_curation_transform(
            self._lake,
            operation="materialization-report",
            input_scenario_ids=scenario_ids,
            output_scenario_ids=scenario_ids,
            report={**report, "materialization_id": materialization_id},
            prior_transform_ids=(snapshot.get("transform_id") or "",),
            created_by=created_by,
            output_tables=("curation_materializations",),
        )
        now = datetime.now(UTC)
        table = self._lake.table("curation_materializations")
        table.delete(f"materialization_id = '{materialization_id}'")
        table.add(
            pa.Table.from_pylist(
                [
                    {
                        "materialization_id": materialization_id,
                        "dataset_id": snapshot["dataset_id"],
                        "snapshot_name": snapshot["name"],
                        "target_format": target_format,
                        "output_uri": output_uri,
                        "mode": mode,
                        "selected_scenario_count": len(scenario_ids),
                        "selected_observation_count": len(observation_rows),
                        "total_payload_bytes": total_payload_bytes,
                        "copied_payload_bytes": copied_payload_bytes,
                        "logical_reference_bytes": logical_reference_bytes,
                        "metadata_bytes_written": metadata_bytes_written,
                        "copy_ratio": copy_ratio,
                        "source_table_versions": list(source_table_versions),
                        "report_json": json.dumps(report, sort_keys=True),
                        "projection_transform_id": projection_transform_id,
                        "created_by": created_by,
                        "transform_id": transform_id,
                        "created_at": now,
                    }
                ],
                schema=CURATION_MATERIALIZATIONS_SCHEMA,
            )
        )
        return CurationMaterializationReport(
            materialization_id=materialization_id,
            dataset_id=str(snapshot["dataset_id"]),
            snapshot_name=str(snapshot["name"]),
            target_format=target_format,
            total_payload_bytes=total_payload_bytes,
            copied_payload_bytes=copied_payload_bytes,
            logical_reference_bytes=logical_reference_bytes,
            metadata_bytes_written=metadata_bytes_written,
            transform_id=transform_id,
            report=report,
            planned_payload_bytes=planned_payload_bytes,
        )


def _metric_rows(metrics: Iterable[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(metrics, dict):
        if "metrics" in metrics and isinstance(metrics["metrics"], list):
            rows = metrics["metrics"]
        else:
            rows = [metrics]
    elif isinstance(metrics, (str, bytes)):
        raise CurationError("eval metrics must be dict rows, not a string")
    else:
        rows = list(metrics)
    if not rows:
        raise CurationError("no eval metrics supplied")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CurationError(f"eval metric row {index} must be a JSON object")
    return [dict(row) for row in rows]


def _eval_metric_rows(
    lake: Lake,
    *,
    snapshot: dict[str, Any],
    metrics: Iterable[dict[str, Any]] | dict[str, Any],
    training_run_id: str,
    model_version: str,
    evaluation_run_id: str,
    regression_threshold: float,
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    metric_payloads: list[dict[str, Any]] = []
    snapshot_name = str(snapshot["name"])
    dataset_id = str(snapshot["dataset_id"])
    snapshot_scenario_ids = set(_scenario_ids_from_snapshot_row(snapshot))
    for raw in _metric_rows(metrics):
        metric_name = str(
            raw.get("metric")
            or raw.get("name")
            or raw.get("metric_name")
            or raw.get("output_type")
            or "eval_metric"
        )
        output_type = str(raw.get("output_type") or metric_name)
        score = _required_metric_score(raw)
        baseline_score = _optional_float_value(
            raw.get("baseline_score"),
            raw.get("baseline", raw.get("previous_score", raw.get("reference_score"))),
        )
        higher_is_better = _higher_is_better(raw)
        improvement = None
        if baseline_score is not None:
            improvement = score - baseline_score if higher_is_better else baseline_score - score
        regressed = _metric_regressed(raw, improvement, regression_threshold)
        slice_label = _metric_slice_label(raw)
        scenario_ids = tuple(
            dict.fromkeys(str(item) for item in _as_tuple(raw.get("scenario_ids")) if str(item))
        )
        if raw.get("scenario_id"):
            scenario_ids = tuple(dict.fromkeys((*scenario_ids, str(raw["scenario_id"]))))
        if not scenario_ids and slice_label:
            scenario_ids = tuple(
                sid
                for sid in _scenario_ids_for_slice(lake, snapshot_scenario_ids, slice_label)
                if sid in snapshot_scenario_ids
            )
        unknown = sorted(set(scenario_ids) - snapshot_scenario_ids)
        if unknown:
            raise CurationError(
                f"eval metric scenarios are not in snapshot {snapshot_name!r}: {unknown}"
            )
        row_training_run_id = str(raw.get("training_run_id") or training_run_id)
        row_model_version = str(raw.get("model_version") or model_version)
        row_evaluation_run_id = str(raw.get("evaluation_run_id") or evaluation_run_id)
        row_payload = {
            "snapshot_name": snapshot_name,
            "dataset_id": dataset_id,
            "training_run_id": row_training_run_id,
            "model_version": row_model_version,
            "evaluation_run_id": row_evaluation_run_id,
            "metric": metric_name,
            "output_type": output_type,
            "score": score,
            "baseline_score": baseline_score,
            "higher_is_better": higher_is_better,
            "improvement": improvement,
            "regressed": regressed,
            "regression_threshold": regression_threshold,
            "slice": slice_label,
            "slice_values": _slice_values(slice_label),
            "scenario_ids": list(scenario_ids),
            "metadata": _jsonable(raw.get("metadata") or {}),
        }
        model_output_id = str(raw.get("model_output_id") or "eval-" + _digest(row_payload))
        row_payload["source_model_output_id"] = model_output_id
        row = {
            "model_output_id": model_output_id,
            "run_id": str(raw.get("run_id") or ""),
            "observation_id": str(raw.get("observation_id") or ""),
            "scenario_id": str(raw.get("scenario_id") or (scenario_ids[0] if len(scenario_ids) == 1 else "")),
            "dataset_id": dataset_id,
            "model_version": row_model_version,
            "output_type": output_type,
            "prediction": str(raw.get("prediction") or metric_name),
            "output_json": json.dumps(row_payload, sort_keys=True),
            "score": score,
            "producer_run_id": row_training_run_id,
            "source": str(raw.get("source") or "eval-import"),
            "metadata": _metadata_items(row_payload),
            "transform_id": "",
            "created_at": datetime.now(UTC),
        }
        rows.append(row)
        metric_payloads.append(dict(row_payload))
        if regressed:
            regressions.append(dict(row_payload))
    return rows, tuple(regressions), metric_payloads


def _required_metric_score(row: dict[str, Any]) -> float:
    value = _optional_float_value(
        row.get("score"),
        row.get("value", row.get("metric_value", row.get("mean"))),
    )
    if value is None:
        raise CurationError("eval metric row is missing numeric score/value")
    return value


def _higher_is_better(row: dict[str, Any]) -> bool:
    if row.get("higher_is_better") is not None:
        return bool(row["higher_is_better"])
    if row.get("lower_is_better") is not None:
        return not bool(row["lower_is_better"])
    direction = str(row.get("direction") or "").strip().lower().replace("_", "-")
    if direction in {"lower", "lower-is-better", "minimize"}:
        return False
    return True


def _metric_regressed(
    row: dict[str, Any],
    improvement: float | None,
    regression_threshold: float,
) -> bool:
    if row.get("regressed") is not None:
        return bool(row["regressed"])
    if improvement is None:
        return False
    return improvement < -regression_threshold


def _metric_slice_label(row: dict[str, Any]) -> str:
    label = str(row.get("slice") or row.get("slice_label") or "").strip()
    if label:
        return label
    values = row.get("slice_values") or row.get("values") or {}
    if isinstance(values, dict) and values:
        return "|".join(f"{key}={values[key]}" for key in sorted(values))
    return ""


def _slice_values(label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in str(label).split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key:
            values[key] = value
    return values


def _replace_model_output_rows(lake: Lake, rows: Sequence[dict[str, Any]]) -> None:
    ids = [str(row["model_output_id"]) for row in rows]
    if not ids:
        return
    lake.table("model_outputs").delete(
        f"model_output_id IN ({', '.join(_sql_literal(row_id) for row_id in ids)})"
    )
    lake.table("model_outputs").add(pa.Table.from_pylist(list(rows), schema=MODEL_OUTPUTS_SCHEMA))


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _eval_metric_series_key(
    snapshot_name: str, model_version: str, metric: str, slice_label: str
) -> str:
    """Supersession group: re-imports of the same logical metric replace each other.

    Keyed on the snapshot *name* (not ``dataset_id``) so a re-created snapshot
    with the same name continues the same metric series.
    """
    return "ems-" + _digest(
        {
            "snapshot_name": str(snapshot_name),
            "model_version": str(model_version),
            "metric": str(metric),
            "slice_label": str(slice_label),
        }
    )


def _eval_catalog_row_from_payload(
    payload: Mapping[str, Any],
    *,
    snapshot_tag: str,
    state: str,
    transform_id: str,
    created_by: str,
    now: datetime,
    table_versions: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    slice_label = str(payload.get("slice") or payload.get("slice_label") or "")
    slice_values = payload.get("slice_values") or _slice_values(slice_label)
    baseline = payload.get("baseline_score")
    improvement = payload.get("improvement")
    return {
        "model_output_id": str(payload["source_model_output_id"]),
        "series_key": _eval_metric_series_key(
            str(payload.get("snapshot_name") or ""),
            str(payload.get("model_version") or ""),
            str(payload.get("metric") or ""),
            slice_label,
        ),
        "state": state,
        "dataset_id": str(payload.get("dataset_id") or ""),
        "snapshot_name": str(payload.get("snapshot_name") or ""),
        "snapshot_tag": snapshot_tag,
        "training_run_id": str(payload.get("training_run_id") or ""),
        "model_version": str(payload.get("model_version") or ""),
        "evaluation_run_id": str(payload.get("evaluation_run_id") or ""),
        "metric": str(payload.get("metric") or ""),
        "output_type": str(payload.get("output_type") or ""),
        "slice_label": slice_label,
        "slice_values": [
            {"key": str(key), "value": str(value)}
            for key, value in sorted(dict(slice_values).items())
        ],
        "score": float(payload["score"]),
        "baseline_score": float(baseline) if baseline is not None else None,
        "improvement": float(improvement) if improvement is not None else None,
        "higher_is_better": bool(payload.get("higher_is_better", True)),
        "regressed": bool(payload.get("regressed", False)),
        "regression_threshold": float(payload.get("regression_threshold") or 0.0),
        "scenario_count": len(payload.get("scenario_ids") or ()),
        "superseded_by": "",
        "superseded_at": None,
        "pruned_at": None,
        "retention_policy_json": "",
        "table_versions": [
            {"table": table, "version": int(version), "tag": ""}
            for table, version in table_versions
            if table in _EVAL_METRIC_STALENESS_TABLES
        ],
        "created_by": created_by,
        "transform_id": transform_id,
        "created_at": now,
    }


def _eval_metric_entry_from_row(row: Mapping[str, Any]) -> EvalMetricEntry:
    state = str(row.get("state") or _EVAL_METRIC_STATE_ACTIVE)
    baseline = row.get("baseline_score")
    improvement = row.get("improvement")
    return EvalMetricEntry(
        model_output_id=str(row.get("model_output_id") or ""),
        series_key=str(row.get("series_key") or ""),
        state=state,
        dataset_id=str(row.get("dataset_id") or ""),
        snapshot_name=str(row.get("snapshot_name") or ""),
        snapshot_tag=str(row.get("snapshot_tag") or ""),
        training_run_id=str(row.get("training_run_id") or ""),
        model_version=str(row.get("model_version") or ""),
        evaluation_run_id=str(row.get("evaluation_run_id") or ""),
        metric=str(row.get("metric") or ""),
        output_type=str(row.get("output_type") or ""),
        slice_label=str(row.get("slice_label") or ""),
        slice_values={
            str(item.get("key")): str(item.get("value"))
            for item in row.get("slice_values") or ()
            if isinstance(item, Mapping) and item.get("key") is not None
        },
        score=float(row.get("score") or 0.0),
        baseline_score=float(baseline) if baseline is not None else None,
        improvement=float(improvement) if improvement is not None else None,
        higher_is_better=bool(row.get("higher_is_better", True)),
        regressed=bool(row.get("regressed", False)),
        regression_threshold=float(row.get("regression_threshold") or 0.0),
        scenario_count=int(row.get("scenario_count") or 0),
        retention_policy=dict(_loads_json(row.get("retention_policy_json"), {})),
        table_versions=_comparison_table_version_pairs(row.get("table_versions") or ()),
        created_by=str(row.get("created_by") or ""),
        transform_id=str(row.get("transform_id") or ""),
        created_at=_as_optional_utc(row.get("created_at")) or datetime.now(UTC),
        superseded_by=str(row.get("superseded_by") or ""),
        superseded_at=_as_optional_utc(row.get("superseded_at")),
        pruned_at=_as_optional_utc(row.get("pruned_at")),
    )


def _eval_metric_sort_key(entry: EvalMetricEntry) -> tuple[datetime, str]:
    return (entry.created_at, entry.model_output_id)


def _eval_catalog_delete_ids(table: Any, ids: Sequence[str], *, chunk: int = 512) -> None:
    for start in range(0, len(ids), chunk):
        block = ids[start : start + chunk]
        table.delete(
            "model_output_id IN (" + ", ".join(_sql_literal(item) for item in block) + ")"
        )


def _upsert_eval_metric_catalog(
    lake: Lake,
    *,
    payloads: Sequence[Mapping[str, Any]],
    snapshot_tag: str = "",
    transform_id: str,
    created_by: str,
    now: datetime,
    table_versions: Sequence[tuple[str, int]],
) -> None:
    """Mirror one eval import into the catalog and supersede older series entries."""
    rows = [
        _eval_catalog_row_from_payload(
            payload,
            snapshot_tag=snapshot_tag,
            state=_EVAL_METRIC_STATE_ACTIVE,
            transform_id=transform_id,
            created_by=created_by,
            now=now,
            table_versions=table_versions,
        )
        for payload in payloads
    ]
    if not rows:
        return
    table = lake.table("eval_metric_catalog")
    new_ids = [row["model_output_id"] for row in rows]
    newest_by_series = {row["series_key"]: row["model_output_id"] for row in rows}
    superseded_rows: list[dict[str, Any]] = []
    series_sql = ", ".join(_sql_literal(key) for key in sorted(newest_by_series))
    for batch in _stream_table_rows(
        lake,
        "eval_metric_catalog",
        where_sql=(
            f"series_key IN ({series_sql}) AND state = {_sql_literal(_EVAL_METRIC_STATE_ACTIVE)}"
        ),
    ):
        for row in batch:
            if str(row.get("model_output_id")) in new_ids:
                continue
            updated = dict(row)
            updated["state"] = _EVAL_METRIC_STATE_SUPERSEDED
            updated["superseded_by"] = newest_by_series[str(row.get("series_key"))]
            updated["superseded_at"] = now
            superseded_rows.append(updated)
    _eval_catalog_delete_ids(
        table, new_ids + [row["model_output_id"] for row in superseded_rows]
    )
    table.add(
        pa.Table.from_pylist(rows + superseded_rows, schema=EVAL_METRIC_CATALOG_SCHEMA)
    )


def _eval_metric_where_sql(
    *,
    snapshot: str | None,
    evaluation_run: str | None,
    training_run: str | None,
    model_version: str | None,
    metric: str | None,
    slice_label: str | None,
    regressed_only: bool,
    state: str | None,
) -> str | None:
    clauses: list[str] = []
    if snapshot:
        literal = _sql_literal(snapshot)
        clauses.append(f"(snapshot_name = {literal} OR dataset_id = {literal})")
    if evaluation_run:
        clauses.append(f"evaluation_run_id = {_sql_literal(evaluation_run)}")
    if training_run:
        clauses.append(f"training_run_id = {_sql_literal(training_run)}")
    if model_version:
        clauses.append(f"model_version = {_sql_literal(model_version)}")
    if metric:
        clauses.append(f"metric = {_sql_literal(metric)}")
    if slice_label:
        clauses.append(f"slice_label = {_sql_literal(slice_label)}")
    if regressed_only:
        clauses.append("regressed = true")
    if state:
        clauses.append(f"state = {_sql_literal(state)}")
    return " AND ".join(clauses) if clauses else None


def _list_eval_metric_catalog(
    lake: Lake,
    *,
    snapshot: str | None,
    evaluation_run: str | None,
    training_run: str | None,
    model_version: str | None,
    metric: str | None,
    slice_label: str | None,
    regressed_only: bool,
    state: str | None,
    latest_only: bool,
    since: datetime | None,
    until: datetime | None,
    limit: int | None,
) -> EvalMetricListing:
    if state is not None and state not in _EVAL_METRIC_STATES:
        raise CurationError(
            f"unknown eval metric state {state!r}; expected one of {_EVAL_METRIC_STATES}"
        )
    if limit is not None and limit <= 0:
        raise CurationError("eval metric listing limit must be positive")
    preview_limit = limit or _EVAL_METRIC_DEFAULT_PREVIEW_LIMIT
    where_sql = _eval_metric_where_sql(
        snapshot=snapshot,
        evaluation_run=evaluation_run,
        training_run=training_run,
        model_version=model_version,
        metric=metric,
        slice_label=slice_label,
        regressed_only=regressed_only,
        state=state,
    )

    def matches() -> Iterable[EvalMetricEntry]:
        for batch in _stream_table_rows(lake, "eval_metric_catalog", where_sql=where_sql):
            for row in batch:
                entry = _eval_metric_entry_from_row(row)
                if where_sql and not _eval_metric_row_matches(
                    entry,
                    snapshot=snapshot,
                    evaluation_run=evaluation_run,
                    training_run=training_run,
                    model_version=model_version,
                    metric=metric,
                    slice_label=slice_label,
                    regressed_only=regressed_only,
                    state=state,
                ):
                    # The streaming scan can fall back to a full materialized
                    # read; re-apply the predicate so the fallback stays correct.
                    continue
                if since is not None and entry.created_at < _as_utc(since):
                    continue
                if until is not None and entry.created_at > _as_utc(until):
                    continue
                yield entry

    if latest_only:
        newest_by_series: dict[str, EvalMetricEntry] = {}
        for entry in matches():
            current = newest_by_series.get(entry.series_key)
            if current is None or _eval_metric_sort_key(entry) > _eval_metric_sort_key(current):
                newest_by_series[entry.series_key] = entry
        candidates = list(newest_by_series.values())
        total = len(candidates)
        candidates.sort(key=_eval_metric_sort_key, reverse=True)
        return EvalMetricListing(
            entries=tuple(candidates[:preview_limit]),
            total_count=total,
            preview_limit=preview_limit,
        )

    total = 0

    def counted() -> Iterable[EvalMetricEntry]:
        nonlocal total
        for entry in matches():
            total += 1
            yield entry

    top = heapq.nlargest(preview_limit, counted(), key=_eval_metric_sort_key)
    return EvalMetricListing(
        entries=tuple(top), total_count=total, preview_limit=preview_limit
    )


def _eval_metric_row_matches(
    entry: EvalMetricEntry,
    *,
    snapshot: str | None,
    evaluation_run: str | None,
    training_run: str | None,
    model_version: str | None,
    metric: str | None,
    slice_label: str | None,
    regressed_only: bool,
    state: str | None,
) -> bool:
    if snapshot and snapshot not in (entry.snapshot_name, entry.dataset_id):
        return False
    if evaluation_run and entry.evaluation_run_id != evaluation_run:
        return False
    if training_run and entry.training_run_id != training_run:
        return False
    if model_version and entry.model_version != model_version:
        return False
    if metric and entry.metric != metric:
        return False
    if slice_label and entry.slice_label != slice_label:
        return False
    if regressed_only and not entry.regressed:
        return False
    if state and entry.state != state:
        return False
    return True


def _eval_metric_staleness(lake: Lake, id_or_run: str) -> EvalMetricStaleness:
    token = str(id_or_run or "").strip()
    if not token:
        raise CurationError("an eval metric model_output_id or evaluation_run_id is required")
    entry: EvalMetricEntry | None = None
    literal = _sql_literal(token)
    for batch in _stream_table_rows(
        lake,
        "eval_metric_catalog",
        where_sql=f"model_output_id = {literal} OR evaluation_run_id = {literal}",
    ):
        for row in batch:
            candidate = _eval_metric_entry_from_row(row)
            if candidate.model_output_id != token and candidate.evaluation_run_id != token:
                continue
            if candidate.model_output_id == token:
                entry = candidate
                break
            if entry is None or _eval_metric_sort_key(candidate) > _eval_metric_sort_key(entry):
                entry = candidate
        if entry is not None and entry.model_output_id == token:
            break
    if entry is None:
        raise CurationError(f"no eval metric {token!r} in catalog")
    recorded = {
        table: version
        for table, version in entry.table_versions
        if table in _EVAL_METRIC_STALENESS_TABLES
    }
    if recorded:
        current_pairs = _table_versions(lake, tables=tuple(sorted(recorded)))
        current = {row["table"]: int(row["version"]) for row in current_pairs}
    else:
        current = {}
    advanced: list[dict[str, Any]] = []
    for table in sorted(recorded):
        recorded_version = int(recorded[table])
        current_version = int(current.get(table, recorded_version))
        if current_version > recorded_version:
            advanced.append(
                {
                    "table": table,
                    "recorded_version": recorded_version,
                    "current_version": current_version,
                }
            )
    return EvalMetricStaleness(
        model_output_id=entry.model_output_id,
        evaluation_run_id=entry.evaluation_run_id,
        stale=bool(advanced),
        advanced_tables=tuple(advanced),
        recorded_table_versions=tuple(sorted(recorded.items())),
        current_table_versions=tuple(sorted(current.items())),
        checked_at=datetime.now(UTC),
    )


def _protected_eval_metric_refs(lake: Lake) -> tuple[set[str], set[str]]:
    """Eval evidence pinned by snapshot promotion decisions (never pruned).

    Promotion decisions land in ``curation_memberships`` at the ``snapshot-row``
    grain with the decision's ``evaluation_run_id`` and metric payloads in kv
    metadata; both the referenced ``model_output_id``s and every metric of the
    referenced evaluation run stay protected.
    """
    protected_ids: set[str] = set()
    protected_runs: set[str] = set()
    for batch in _stream_table_rows(
        lake,
        "curation_memberships",
        columns=["target_grain", "metadata"],
        where_sql="target_grain = 'snapshot-row'",
    ):
        for row in batch:
            if str(row.get("target_grain") or "") != "snapshot-row":
                continue
            for item in row.get("metadata") or ():
                if not isinstance(item, Mapping):
                    continue
                key = str(item.get("key") or "")
                value = _loads_json(item.get("value"), item.get("value"))
                if key == "evaluation_run_id" and value:
                    protected_runs.add(str(value))
                elif key == "metrics" and isinstance(value, list):
                    for payload in value:
                        if not isinstance(payload, Mapping):
                            continue
                        for id_key in ("source_model_output_id", "model_output_id"):
                            if payload.get(id_key):
                                protected_ids.add(str(payload[id_key]))
                        if payload.get("evaluation_run_id"):
                            protected_runs.add(str(payload["evaluation_run_id"]))
    return protected_ids, protected_runs


def _prune_eval_metric_catalog(
    lake: Lake,
    *,
    retain_latest: int,
    older_than: datetime | timedelta | None,
    dry_run: bool,
    created_by: str,
) -> EvalMetricRetentionReport:
    if retain_latest < 1:
        raise CurationError("retain_latest must be at least 1")
    now = datetime.now(UTC)
    cutoff = _retention_cutoff(older_than, now)
    rows_by_id: dict[str, dict[str, Any]] = {}
    entries: list[EvalMetricEntry] = []
    for batch in _stream_table_rows(lake, "eval_metric_catalog"):
        for row in batch:
            entry = _eval_metric_entry_from_row(row)
            rows_by_id[entry.model_output_id] = dict(row)
            entries.append(entry)
    groups: dict[str, list[EvalMetricEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.series_key, []).append(entry)
    protected_ids, protected_runs = _protected_eval_metric_refs(lake)
    retained: list[str] = []
    supersede_targets: list[EvalMetricEntry] = []
    prune_targets: list[EvalMetricEntry] = []
    protected: list[str] = []
    for group in groups.values():
        group.sort(key=_eval_metric_sort_key, reverse=True)
        retained.extend(entry.model_output_id for entry in group[:retain_latest])
        for entry in group[retain_latest:]:
            if entry.state == _EVAL_METRIC_STATE_PRUNED:
                continue
            is_protected = (
                entry.model_output_id in protected_ids
                or (entry.evaluation_run_id and entry.evaluation_run_id in protected_runs)
            )
            if entry.state == _EVAL_METRIC_STATE_ACTIVE:
                supersede_targets.append(entry)
            if cutoff is not None and entry.created_at < cutoff:
                if is_protected:
                    protected.append(entry.model_output_id)
                else:
                    prune_targets.append(entry)
    policy = {
        "retain_latest": retain_latest,
        "older_than": cutoff.isoformat() if cutoff else None,
        "applied_at": now.isoformat(),
    }
    transform_id = ""
    if (supersede_targets or prune_targets) and not dry_run:
        policy_json = json.dumps(policy, sort_keys=True)
        updated_rows: list[dict[str, Any]] = []
        prune_ids = {entry.model_output_id for entry in prune_targets}
        for entry in supersede_targets:
            if entry.model_output_id in prune_ids:
                continue
            row = dict(rows_by_id[entry.model_output_id])
            row["state"] = _EVAL_METRIC_STATE_SUPERSEDED
            newer = next(
                (
                    item.model_output_id
                    for item in groups[entry.series_key]
                    if _eval_metric_sort_key(item) > _eval_metric_sort_key(entry)
                ),
                "",
            )
            row["superseded_by"] = newer
            row["superseded_at"] = now
            row["retention_policy_json"] = policy_json
            updated_rows.append(row)
        for entry in prune_targets:
            row = dict(rows_by_id[entry.model_output_id])
            row["state"] = _EVAL_METRIC_STATE_PRUNED
            row["pruned_at"] = now
            if row.get("superseded_at") is None:
                row["superseded_at"] = now
            row["retention_policy_json"] = policy_json
            updated_rows.append(row)
        table = lake.table("eval_metric_catalog")
        _eval_catalog_delete_ids(table, [row["model_output_id"] for row in updated_rows])
        table.add(pa.Table.from_pylist(updated_rows, schema=EVAL_METRIC_CATALOG_SCHEMA))
        if prune_targets:
            _eval_catalog_delete_ids(
                lake.table("model_outputs"), sorted(prune_ids)
            )
        transform_id = _record_feedback_loop_transform(
            lake,
            operation="eval-metric-retention",
            input_scenario_ids=(),
            output_scenario_ids=(),
            report={
                "operation": "eval-metric-retention",
                "superseded_ids": [entry.model_output_id for entry in supersede_targets],
                "pruned_ids": sorted(prune_ids),
                "protected_ids": sorted(set(protected)),
                "retained_ids": sorted(retained),
                "policy": policy,
            },
            prior_transform_ids=tuple(
                dict.fromkeys(
                    entry.transform_id
                    for entry in (*supersede_targets, *prune_targets)
                    if entry.transform_id
                )
            ),
            output_tables=("model_outputs", "eval_metric_catalog"),
            created_by=created_by,
        )
    return EvalMetricRetentionReport(
        superseded_ids=tuple(entry.model_output_id for entry in supersede_targets),
        pruned_ids=tuple(entry.model_output_id for entry in prune_targets),
        protected_ids=tuple(sorted(set(protected))),
        retained_ids=tuple(sorted(retained)),
        dry_run=dry_run,
        policy=policy,
        transform_id=transform_id,
        created_at=now,
    )


def _eval_metric_payload_from_output_json(raw: Any) -> dict[str, Any] | None:
    payload = _loads_json(raw, None)
    if not isinstance(payload, dict):
        return None
    required = ("source_model_output_id", "snapshot_name", "metric", "score")
    if any(payload.get(key) in (None, "") for key in required):
        return None
    return payload


def _sync_eval_metric_catalog(
    lake: Lake,
    *,
    build_indexes: bool,
    created_by: str,
) -> EvalMetricCatalogSyncReport:
    now = datetime.now(UTC)
    existing_by_id: dict[str, dict[str, Any]] = {}
    for batch in _stream_table_rows(lake, "eval_metric_catalog"):
        for row in batch:
            existing_by_id[str(row.get("model_output_id") or "")] = dict(row)
    snapshot_tags: dict[str, str] = {}
    for batch in _stream_table_rows(
        lake, "dataset_snapshots", columns=["dataset_id", "tag"]
    ):
        for row in batch:
            snapshot_tags[str(row.get("dataset_id") or "")] = str(row.get("tag") or "")
    current_versions = tuple(
        (str(item["table"]), int(item["version"]))
        for item in _table_versions(lake, tables=_EVAL_METRIC_STALENESS_TABLES)
    )
    scanned = 0
    rebuilt: dict[str, dict[str, Any]] = {}
    for batch in _stream_table_rows(
        lake,
        "model_outputs",
        columns=["model_output_id", "output_json", "transform_id", "created_at"],
    ):
        for row in batch:
            scanned += 1
            payload = _eval_metric_payload_from_output_json(row.get("output_json"))
            if payload is None:
                continue
            model_output_id = str(row.get("model_output_id") or "")
            existing = existing_by_id.get(model_output_id)
            catalog_row = _eval_catalog_row_from_payload(
                payload,
                snapshot_tag=snapshot_tags.get(str(payload.get("dataset_id") or ""), ""),
                state=_EVAL_METRIC_STATE_ACTIVE,
                transform_id=str(row.get("transform_id") or ""),
                created_by=created_by,
                now=_as_optional_utc(row.get("created_at")) or now,
                table_versions=current_versions,
            )
            catalog_row["model_output_id"] = model_output_id
            if existing is not None:
                # Keep import-time facts the source row cannot re-derive.
                catalog_row["table_versions"] = existing.get("table_versions") or catalog_row[
                    "table_versions"
                ]
                catalog_row["created_by"] = str(existing.get("created_by") or created_by)
                catalog_row["retention_policy_json"] = str(
                    existing.get("retention_policy_json") or ""
                )
            rebuilt[model_output_id] = catalog_row
    preserved = 0
    for model_output_id, row in existing_by_id.items():
        if model_output_id in rebuilt:
            continue
        if str(row.get("state") or "") == _EVAL_METRIC_STATE_PRUNED:
            rebuilt[model_output_id] = dict(row)
            preserved += 1
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rebuilt.values():
        groups.setdefault(str(row.get("series_key") or ""), []).append(row)
    active = 0
    superseded = 0
    for group in groups.values():
        live = [
            row for row in group if str(row.get("state")) != _EVAL_METRIC_STATE_PRUNED
        ]
        live.sort(
            key=lambda row: (
                _as_optional_utc(row.get("created_at")) or now,
                str(row.get("model_output_id") or ""),
            ),
            reverse=True,
        )
        for index, row in enumerate(live):
            if index == 0:
                row["state"] = _EVAL_METRIC_STATE_ACTIVE
                row["superseded_by"] = ""
                row["superseded_at"] = None
                active += 1
            else:
                row["state"] = _EVAL_METRIC_STATE_SUPERSEDED
                row["superseded_by"] = str(live[0].get("model_output_id") or "")
                if row.get("superseded_at") is None:
                    row["superseded_at"] = now
                superseded += 1
    table = lake.table("eval_metric_catalog")
    _eval_catalog_delete_ids(table, sorted(existing_by_id))
    if rebuilt:
        table.add(
            pa.Table.from_pylist(list(rebuilt.values()), schema=EVAL_METRIC_CATALOG_SCHEMA)
        )
    index_results: tuple[dict[str, Any], ...] = ()
    if build_indexes:
        index_results = tuple(
            {
                "table": result.table,
                "column": result.column,
                "status": result.status,
                "index_type": result.index_type,
                "reason": result.reason,
            }
            for result in build_eval_metric_catalog_predicate_indexes(lake)
        )
    transform_id = _record_feedback_loop_transform(
        lake,
        operation="eval-metric-catalog-sync",
        input_scenario_ids=(),
        output_scenario_ids=(),
        report={
            "operation": "eval-metric-catalog-sync",
            "scanned_model_outputs": scanned,
            "cataloged": len(rebuilt),
            "active": active,
            "superseded": superseded,
            "preserved_pruned": preserved,
        },
        prior_transform_ids=(),
        output_tables=("eval_metric_catalog",),
        created_by=created_by,
    )
    return EvalMetricCatalogSyncReport(
        scanned_model_outputs=scanned,
        cataloged=len(rebuilt),
        active=active,
        superseded=superseded,
        preserved_pruned=preserved,
        index_results=index_results,
        transform_id=transform_id,
        created_at=now,
    )


def _feedback_loop_transform_id(
    *,
    operation: str,
    input_scenario_ids: Sequence[str],
    output_scenario_ids: Sequence[str],
    report: dict[str, Any],
    prior_transform_ids: Sequence[str],
    output_tables: Sequence[str],
) -> str:
    """Stable feedback-loop transform id (content digest, no table versions).

    Shared by :func:`_record_feedback_loop_transform` and the 0096 dry-run
    path, so a dry run reports the exact transform id a real apply would
    record.
    """
    stable_payload = {
        "operation": operation,
        "input_scenario_ids": list(input_scenario_ids),
        "output_scenario_ids": list(output_scenario_ids),
        "report": _jsonable(report),
        "prior_operation_transform_ids": [str(item) for item in prior_transform_ids if str(item)],
        "output_tables": list(output_tables),
    }
    return "tfm-curate-" + operation.replace("-", "_") + "-" + _digest(stable_payload)


def _record_feedback_loop_transform(
    lake: Lake,
    *,
    operation: str,
    input_scenario_ids: Sequence[str],
    output_scenario_ids: Sequence[str],
    report: dict[str, Any],
    prior_transform_ids: Sequence[str],
    output_tables: Sequence[str],
    created_by: str,
) -> str:
    transform_id = _feedback_loop_transform_id(
        operation=operation,
        input_scenario_ids=input_scenario_ids,
        output_scenario_ids=output_scenario_ids,
        report=report,
        prior_transform_ids=prior_transform_ids,
        output_tables=output_tables,
    )
    input_versions = _table_versions(lake, tables=_FEEDBACK_LOOP_TABLES)
    params = {
        **_jsonable(report),
        "input_scenario_ids": list(input_scenario_ids),
        "output_scenario_ids": list(output_scenario_ids),
        "prior_operation_transform_ids": [
            str(item) for item in prior_transform_ids if str(item)
        ],
        "input_table_versions": input_versions,
    }
    now = datetime.now(UTC)
    transform_row = {
        "transform_id": transform_id,
        "kind": f"curation-{operation}",
        "input_uris": [],
        "input_table_versions": input_versions,
        "output_tables": list(output_tables),
        "params": json.dumps(params, sort_keys=True),
        "status": "completed",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = {_sql_literal(transform_id)}")
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): feedback-loop transform provenance without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
    return transform_id


def _snapshot_prior_transform_ids(snapshot: dict[str, Any]) -> tuple[str, ...]:
    ids = [str(snapshot.get("transform_id") or "")]
    try:
        query_spec = json.loads(snapshot.get("query_spec") or "{}")
    except json.JSONDecodeError:
        query_spec = {}
    source = query_spec.get("source") or {}
    ids.extend(str(item) for item in source.get("operation_transform_ids") or ())
    return tuple(dict.fromkeys(item for item in ids if item))


def _regressions_from(value: Any, lake: Lake) -> tuple[dict[str, Any], ...]:
    if isinstance(value, CurationFeedbackReport):
        return value.regressions
    if isinstance(value, Mapping):
        return _regressions_from_mapping(value, lake)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        regressions: list[dict[str, Any]] = []
        for item in value:
            regressions.extend(_regressions_from(item, lake))
        return tuple(regressions)
    raise CurationError("from_regressions must be a feedback report, dict, or regression rows")


def _regressions_from_mapping(value: Mapping[str, Any], lake: Lake) -> tuple[dict[str, Any], ...]:
    if "regressions" in value:
        return tuple(dict(item) for item in value.get("regressions") or ())
    if value.get("regressed") is not None or value.get("source_model_output_id"):
        return (dict(value),)
    if value.get("evaluation_run_id") or value.get("dataset_id") or value.get("snapshot_name"):
        return _regressions_from_model_outputs(lake, value)
    return (dict(value),)


def _regressions_from_model_outputs(lake: Lake, query: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    dataset_id = str(query.get("dataset_id") or "")
    if not dataset_id and query.get("snapshot_name"):
        dataset_id = str(_latest_snapshot_row(lake, str(query["snapshot_name"]))["dataset_id"])
    evaluation_run_id = str(query.get("evaluation_run_id") or "")
    output_rows = lake.table("model_outputs").to_arrow().to_pylist()
    regressions: list[dict[str, Any]] = []
    for row in output_rows:
        if dataset_id and row.get("dataset_id") != dataset_id:
            continue
        metadata = _metadata_dict(row.get("metadata") or ())
        if evaluation_run_id and metadata.get("evaluation_run_id") != evaluation_run_id:
            continue
        if not metadata.get("regressed"):
            continue
        regression = dict(metadata)
        regression.setdefault("source_model_output_id", row.get("model_output_id") or "")
        regression.setdefault("transform_id", row.get("transform_id") or "")
        regressions.append(regression)
    return tuple(sorted(regressions, key=lambda item: _regression_key(item, 0)))


def _candidate_ids_for_regression(
    lake: Lake,
    source_selection: CurationSelection,
    regression: dict[str, Any],
    *,
    limit: int,
    embedding_column: str,
    use_index: bool = False,
    nprobes: int | None = None,
    refine_factor: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Candidates for one regression, plus stage attribution for the plan (0096).

    ``use_index=True`` routes the per-seed neighbor search through the
    persistent LanceDB vector index (:func:`_nearest_scenarios_indexed`)
    instead of the in-memory exact scan. The returned stage info records how
    many seeds were searched and whether the slice-fill fallback ran.
    """
    seed_ids = [
        str(item)
        for item in _as_tuple(regression.get("scenario_ids") or regression.get("scenario_id"))
        if str(item)
    ]
    selected: list[str] = []
    seeds_searched = 0
    for seed_id in seed_ids:
        try:
            if use_index:
                candidates = _nearest_scenarios_indexed(
                    lake,
                    source_selection.scenario_ids,
                    seed_scenario=seed_id,
                    limit=limit,
                    embedding_column=embedding_column,
                    nprobes=nprobes,
                    refine_factor=refine_factor,
                )
            else:
                candidates = _nearest_scenarios(
                    lake,
                    source_selection.scenario_ids,
                    seed_scenario=seed_id,
                    limit=limit,
                    embedding_column=embedding_column,
                )
        except CurationError:
            if seed_id in source_selection.scenario_ids:
                candidates = [seed_id]
            else:
                continue
        seeds_searched += 1
        for scenario_id in candidates:
            if scenario_id not in selected:
                selected.append(scenario_id)
        if len(selected) >= limit:
            break
    slice_fill = False
    if not selected and regression.get("slice"):
        selected = _scenario_ids_for_slice(lake, source_selection.scenario_ids, str(regression["slice"]))[:limit]
        slice_fill = bool(selected)
    stage_info = {
        "seed_count": len(seed_ids),
        "seeds_searched": seeds_searched,
        "slice_fill": slice_fill,
    }
    return selected[:limit], stage_info


def _resolve_feedback_candidate_route(
    lake: Lake,
    *,
    embedding_column: str,
    pool_size: int,
    route: str,
    nprobes: int | None,
    refine_factor: int | None,
) -> dict[str, Any]:
    """Resolve the candidate-search route (0096; mirrors the search.py contract).

    ``auto`` rides the persistent vector index when one exists on
    ``scenarios.embedding_column`` and otherwise allows the exact in-memory
    scan only while the pool stays within ``_FEEDBACK_CANDIDATE_EXACT_SCAN_LIMIT``
    — above it the plan carries an unmet index requirement (with the exact
    build remedy) instead of silently scanning. ``exact`` is an explicit
    operator override that always brute-forces; ``ann`` requires the index and
    errors when absent.
    """
    if route not in _FEEDBACK_CANDIDATE_ROUTES:
        raise CurationError(
            f"unknown route {route!r}; expected one of {', '.join(_FEEDBACK_CANDIDATE_ROUTES)}"
        )
    has_index = has_vector_index(lake.table("scenarios"), embedding_column)
    if route == _FEEDBACK_CANDIDATE_ROUTE_ANN and not has_index:
        raise CurationError(
            f"route='ann' requested but scenarios.{embedding_column} has no vector index; "
            f"build one with `lancedb-robotics scenarios index --column {embedding_column}` "
            "or use route='auto'"
        )
    exact_scan_limit = _FEEDBACK_CANDIDATE_EXACT_SCAN_LIMIT
    use_index = has_index and route != _FEEDBACK_CANDIDATE_ROUTE_EXACT
    state = {
        "requested": route,
        "has_index": has_index,
        "use_index": use_index,
        "nprobes": nprobes if use_index else None,
        "refine_factor": refine_factor if use_index else None,
        "pool_size": pool_size,
        "exact_scan_limit": exact_scan_limit,
        "index_met": has_index,
        "remedy": "",
    }
    if use_index:
        return {
            **state,
            "effective": _FEEDBACK_CANDIDATE_ROUTE_ANN,
            "reason": f"persistent vector index on scenarios.{embedding_column}",
            "index_required": True,
            "runnable": True,
        }
    if route == _FEEDBACK_CANDIDATE_ROUTE_EXACT:
        return {
            **state,
            "effective": _FEEDBACK_CANDIDATE_ROUTE_EXACT,
            "reason": "explicit exact route requested; any index is bypassed",
            "index_required": False,
            "runnable": True,
        }
    if pool_size <= exact_scan_limit:
        return {
            **state,
            "effective": _FEEDBACK_CANDIDATE_ROUTE_EXACT,
            "reason": (
                f"no vector index on scenarios.{embedding_column}; pool of {pool_size} "
                f"scenarios is within the {exact_scan_limit}-row exact-scan limit"
            ),
            "index_required": False,
            "runnable": True,
        }
    remedy = (
        f"feedback candidate generation over {pool_size} scenarios requires a persistent "
        f"vector index on scenarios.{embedding_column} (pool exceeds the "
        f"{exact_scan_limit}-row exact-scan limit); run `lancedb-robotics scenarios index "
        f"--column {embedding_column}` (SDK: build_vector_index(lake, table='scenarios', "
        f"column='{embedding_column}')) or pass route='exact' to force an explicit "
        "brute-force scan"
    )
    return {
        **state,
        "effective": _FEEDBACK_CANDIDATE_ROUTE_ANN,
        "reason": remedy,
        "index_required": True,
        "runnable": False,
        "remedy": remedy,
    }


def _nearest_scenarios_indexed(
    lake: Lake,
    scenario_ids: Sequence[str],
    *,
    seed_scenario: str,
    limit: int,
    embedding_column: str,
    nprobes: int | None = None,
    refine_factor: int | None = None,
) -> list[str]:
    """Indexed nearest-neighbor search restricted to the source selection (0096).

    Rides ``table.search()`` over the persistent vector index — probing with
    ``nprobes``/``refine_factor`` in the search.py idiom — with a prefiltering
    ``scenario_id`` where-clause when the source scope is smaller than the
    table, and a bounded over-fetch (never the whole table). Results are
    re-sorted by (distance, scenario_id) so ranking stays deterministic.
    """
    source_ids = tuple(dict.fromkeys(str(item) for item in scenario_ids if str(item)))
    source_set = set(source_ids)
    if seed_scenario not in source_set:
        raise CurationError(f"seed scenario {seed_scenario!r} is not in the workbench scope")
    table = lake.table("scenarios")
    seed_rows = (
        table.search()
        .where(f"scenario_id = {_sql_literal(seed_scenario)}")
        .select(["scenario_id", embedding_column])
        .limit(1)
        .to_list()
    )
    if not seed_rows:
        raise CurationError(f"seed scenario {seed_scenario!r} is not in the workbench scope")
    seed_vector = _vector(seed_rows[0], embedding_column)
    builder = table.search(
        seed_vector, vector_column_name=embedding_column, query_type="vector"
    )
    builder = builder.nprobes(nprobes or _FEEDBACK_CANDIDATE_DEFAULT_NPROBES).refine_factor(
        refine_factor or _FEEDBACK_CANDIDATE_DEFAULT_REFINE_FACTOR
    )
    if len(source_set) < table.count_rows():
        builder = builder.where(
            _sql_in_predicate("scenario_id", source_ids), prefilter=True
        )
    fetch_k = max(limit, _FEEDBACK_CANDIDATE_OVERFETCH)
    rows = builder.select(["scenario_id"]).limit(fetch_k).to_list()
    neighbors = [
        (
            row["_distance"] if row.get("_distance") is not None else float("inf"),
            str(row["scenario_id"]),
        )
        for row in rows
        if str(row.get("scenario_id") or "") in source_set
    ]
    neighbors.sort()
    return [scenario_id for _, scenario_id in neighbors[:limit]]


def _regression_identity(regression: Mapping[str, Any]) -> dict[str, Any]:
    """Stable identity fields of one regression input for plan digests (0096)."""
    return {
        key: _jsonable(regression.get(key))
        for key in _FEEDBACK_REGRESSION_IDENTITY_KEYS
        if regression.get(key) is not None
    }


def _feedback_queue_identity(
    plan_id: str,
    name: str,
    target_grain: str = "scenario",
) -> dict[str, Any]:
    """Stable review-queue identity for feedback-loop queues (0096).

    Keyed on the plan id + artifact name + target grain — NOT on
    ``source_transform_ids``, which drift with table-version churn.
    """
    return {
        "kind": "feedback-candidate-queue",
        "plan_id": str(plan_id),
        "name": str(name),
        "target_grain": str(target_grain),
    }


def _encode_feedback_preview_cursor(row: dict[str, Any]) -> str:
    payload = {
        "regression_key": str(row.get("regression_key") or ""),
        "ordinal": int(row.get("ordinal") or 0),
        "scenario_id": str(row.get("scenario_id") or ""),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return encoded.decode("ascii")


def _decode_feedback_preview_cursor(cursor: str | None) -> dict[str, Any] | None:
    if not cursor:
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(str(cursor).encode("ascii")).decode("utf-8"))
        return {
            "regression_key": str(payload["regression_key"]),
            "ordinal": int(payload["ordinal"]),
            "scenario_id": str(payload["scenario_id"]),
        }
    except Exception as exc:
        raise CurationError("invalid feedback candidate preview cursor") from exc


def _reuse_feedback_view(
    lake: Lake,
    name: str,
    scenario_ids: Sequence[str],
) -> CurationView | None:
    """Latest saved view by name when its membership already matches, else None."""
    try:
        row = _latest_view_row(lake, name)
    except CurationError:
        return None
    view = _view_from_row(row, lake=lake)
    if sorted(view.scenario_ids) == sorted(str(item) for item in scenario_ids):
        return view
    return None


def _reuse_feedback_snapshot(
    lake: Lake,
    *,
    name: str,
    tag: str,
    scenario_ids: Sequence[str],
) -> SnapshotManifest | None:
    """Latest snapshot with the same name+tag+membership, else None (0096).

    ``create_snapshot`` digests table versions into ``dataset_id``, so
    re-creating an identical selection after unrelated churn would add a
    second ``dataset_snapshots`` row under the same name. Reusing the existing
    row keeps repeated feedback-loop runs duplicate-free.
    """
    rows = [
        row
        for row in lake.table("dataset_snapshots").to_arrow().to_pylist()
        if row["name"] == name
    ]
    if not rows:
        return None
    latest = max(rows, key=lambda row: (row["created_at"], row["dataset_id"]))
    if str(latest.get("tag") or "") != str(tag):
        return None
    if sorted(_scenario_ids_from_snapshot_row(latest)) != sorted(
        str(item) for item in scenario_ids
    ):
        return None
    return _snapshot_manifest_from_row(lake, latest)


def _snapshot_manifest_from_row(lake: Lake, row: dict[str, Any]) -> SnapshotManifest:
    """Rehydrate a ``SnapshotManifest`` from a ``dataset_snapshots`` row."""
    query_spec = json.loads(row.get("query_spec") or "{}")
    split = json.loads(row.get("split") or "{}")
    return SnapshotManifest(
        lake_uri=lake.uri,
        dataset_id=str(row["dataset_id"]),
        name=str(row["name"]),
        tag=str(row.get("tag") or ""),
        transform_id=str(row.get("transform_id") or ""),
        scenario_ids=tuple(str(item) for item in query_spec.get("scenario_ids") or ()),
        split_by=str(split.get("by") or SPLIT_BY_RUN),
        split_ratios=dict(split.get("ratios") or {}),
        split_counts=dict(split.get("counts") or {}),
        split_assignments=dict(split.get("assignments") or {}),
        table_versions=tuple(
            (str(item["table"]), int(item["version"]))
            for item in row.get("table_versions") or ()
        ),
        source=query_spec.get("source") or {},
        balance_report=json.loads(row["balance_report"]) if row.get("balance_report") else None,
        coverage_report=(
            json.loads(row["coverage_report"]) if row.get("coverage_report") else None
        ),
    )


def _nearest_scenarios(
    lake: Lake,
    scenario_ids: Sequence[str],
    *,
    seed_scenario: str,
    limit: int,
    embedding_column: str,
) -> list[str]:
    rows = _selected_rows(lake, scenario_ids)
    _require_embedding_column(lake, embedding_column)
    vectors = {row["scenario_id"]: _vector(row, embedding_column) for row in rows}
    if seed_scenario not in vectors:
        raise CurationError(f"seed scenario {seed_scenario!r} is not in the workbench scope")
    seed_vector = vectors[seed_scenario]
    neighbors = [
        {
            "scenario_id": scenario_id,
            "similarity": _cosine(seed_vector, vector),
        }
        for scenario_id, vector in vectors.items()
    ]
    neighbors.sort(key=lambda row: (-row["similarity"], row["scenario_id"]))
    return [str(row["scenario_id"]) for row in neighbors[:limit]]


def _scenario_ids_for_slice(
    lake: Lake,
    scenario_ids: Iterable[str],
    slice_label: str,
) -> list[str]:
    values = _slice_values(slice_label)
    if not values:
        return []
    wanted = set(str(item) for item in scenario_ids)
    rows = _selected_rows(lake, tuple(wanted))
    runs = _run_rows(lake)
    dimensions = tuple(values)
    return [
        str(row["scenario_id"])
        for row in rows
        if _slice_label(row, runs.get(row["run_id"], {}), dimensions) == slice_label
    ]


def _regression_key(regression: dict[str, Any], index: int) -> str:
    return str(
        regression.get("source_model_output_id")
        or regression.get("model_output_id")
        or regression.get("slice")
        or regression.get("metric")
        or f"regression-{index}"
    )


def _regression_priority_score(regression: dict[str, Any]) -> float:
    if regression.get("severity") is not None:
        value = _optional_float_value(regression.get("severity"))
        if value is not None:
            return value
    improvement = _optional_float_value(regression.get("improvement"))
    if improvement is not None:
        return max(0.0, -improvement)
    baseline = _optional_float_value(regression.get("baseline_score"))
    score = _optional_float_value(regression.get("score"))
    if baseline is not None and score is not None:
        return abs(baseline - score)
    return 1.0


def _regression_priority_reason(regression: dict[str, Any]) -> str:
    metric = str(regression.get("metric") or "metric")
    slice_label = str(regression.get("slice") or "all")
    score = regression.get("score")
    baseline = regression.get("baseline_score")
    return f"eval-regression {metric} slice={slice_label} score={score} baseline={baseline}"


def _regression_transform_ids(regressions: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    ids: list[str] = []
    for regression in regressions:
        for key in ("transform_id", "source_transform_id"):
            value = regression.get(key)
            if value:
                ids.append(str(value))
    return tuple(dict.fromkeys(ids))


def _default_regression_queue_name(regressions: Sequence[dict[str, Any]]) -> str:
    for key in ("evaluation_run_id", "training_run_id", "dataset_id"):
        values = sorted({str(row.get(key) or "") for row in regressions if row.get(key)})
        if values:
            return "eval-regressions-" + values[0]
    return "eval-regressions-" + _digest({"regressions": list(regressions)})


def _normalize_promotion_decision(decision: str) -> str:
    normalized = str(decision or "").strip().lower().replace("_", "-")
    aliases = {"promoted": "promote", "accepted": "promote", "rejected": "reject"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in _PROMOTION_DECISIONS:
        raise CurationError(
            f"unknown snapshot promotion decision {decision!r}; expected promote or reject"
        )
    return normalized


def _promotion_metric_payload(
    metrics: Iterable[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if metrics is None:
        return []
    if isinstance(metrics, dict):
        if "metrics" in metrics and isinstance(metrics["metrics"], list):
            rows = metrics["metrics"]
        elif "regressions" in metrics and isinstance(metrics["regressions"], list):
            rows = metrics["regressions"]
        else:
            rows = [metrics]
    elif isinstance(metrics, (str, bytes)):
        raise CurationError("promotion metrics must be dict rows, not a string")
    else:
        rows = list(metrics)
    return [dict(_jsonable(row)) for row in rows if isinstance(row, dict)]


def _semantic_dedup_payload(
    lake: Lake,
    *,
    rows: Sequence[dict[str, Any]],
    embedding_column: str,
    near_duplicate_threshold: float,
    representative_policy: Sequence[str],
    shard_by: Sequence[str],
    neighbor_limit: int,
    require_index: bool,
    index_min_rows: int,
) -> dict[str, Any]:
    table = lake.table("scenarios")
    indexed = has_vector_index(table, embedding_column)
    if len(rows) > 1 and (require_index or len(rows) >= index_min_rows) and not indexed:
        raise CurationError(
            f"semantic dedup over {len(rows)} candidates requires a persistent vector index "
            f"on scenarios.{embedding_column}; run `lancedb-robotics scenarios index "
            f"--column {embedding_column}` or lower index_min_rows for an explicit small scan"
        )

    vectors = {row["scenario_id"]: _vector(row, embedding_column) for row in rows}
    row_by_id = {row["scenario_id"]: row for row in rows}
    run_rows = _run_rows(lake)
    label_counts = _label_counts(lake)
    rarity_scores = _rarity_scores(rows, run_rows, shard_by)
    groups: list[dict[str, Any]] = []
    total_pairs_considered = 0
    total_possible_pairs = 0
    indexed_shards = 0
    exact_shards = 0

    for shard_label, shard_rows in _sharded_rows(rows, run_rows, shard_by).items():
        shard_ids = [row["scenario_id"] for row in shard_rows]
        possible_pairs = len(shard_ids) * (len(shard_ids) - 1) // 2
        total_possible_pairs += possible_pairs
        if indexed and len(shard_ids) > 1:
            edges, comparisons = _duplicate_edges_indexed(
                lake,
                shard_ids=shard_ids,
                vectors=vectors,
                embedding_column=embedding_column,
                threshold=near_duplicate_threshold,
                neighbor_limit=neighbor_limit,
            )
            indexed_shards += 1
        else:
            edges, comparisons = _duplicate_edges_exact(
                shard_ids=shard_ids,
                vectors=vectors,
                threshold=near_duplicate_threshold,
            )
            exact_shards += 1
        total_pairs_considered += comparisons
        for component_ids in _duplicate_components(shard_ids, edges, row_by_id):
            representative_id = _representative_id(
                component_ids,
                rows=row_by_id,
                runs=run_rows,
                label_counts=label_counts,
                rarity_scores=rarity_scores,
                policy=representative_policy,
            )
            members = sorted(component_ids, key=lambda sid: _scenario_sort_key(row_by_id[sid]))
            dropped = [sid for sid in members if sid != representative_id]
            group_id = "dup-" + _digest(
                {
                    "embedding_column": embedding_column,
                    "threshold": near_duplicate_threshold,
                    "shard": shard_label,
                    "representative": representative_id,
                    "members": members,
                }
            )
            member_scores = []
            for rank, scenario_id in enumerate(members):
                similarity = (
                    1.0
                    if scenario_id == representative_id
                    else _cosine(vectors[representative_id], vectors[scenario_id])
                )
                member_scores.append(
                    {
                        "scenario_id": scenario_id,
                        "role": "representative"
                        if scenario_id == representative_id
                        else "duplicate",
                        "rank": rank,
                        "similarity_to_representative": similarity,
                        "distance_to_representative": 1.0 - similarity,
                        "quality_score": _numeric_value(row_by_id[scenario_id], "quality_score"),
                        "label_count": label_counts.get(scenario_id, 0),
                        "rarity_score": rarity_scores.get(scenario_id, 0.0),
                    }
                )
            groups.append(
                {
                    "group_id": group_id,
                    "shard": shard_label,
                    "representative": representative_id,
                    "members": members,
                    "dropped": dropped,
                    "scores": member_scores,
                }
            )

    groups.sort(key=lambda group: _scenario_sort_key(row_by_id[group["representative"]]))
    representative_set = {group["representative"] for group in groups}
    representative_ids = [
        row["scenario_id"]
        for row in sorted(rows, key=_scenario_sort_key)
        if row["scenario_id"] in representative_set
    ]
    dropped_scenario_ids = [
        scenario_id
        for group in groups
        for scenario_id in group["dropped"]
    ]
    search_strategy = (
        "indexed-neighbor"
        if indexed_shards
        else "exact-small"
    )
    report = {
        "operation": "dedup",
        "embedding_column": embedding_column,
        "embedding_provenance": _embedding_provenance(lake, embedding_column),
        "near_duplicate_threshold": near_duplicate_threshold,
        "representative_policy": list(representative_policy),
        "shard_by": list(shard_by),
        "neighbor_limit": neighbor_limit,
        "input_count": len(rows),
        "output_count": len(representative_ids),
        "kept_scenario_ids": representative_ids,
        "dropped_scenario_ids": dropped_scenario_ids,
        "duplicate_groups": groups,
        # Backward-compatible shape consumed by existing tests and notebooks.
        "clusters": [
            {
                "representative": group["representative"],
                "members": group["members"],
                "dropped": group["dropped"],
            }
            for group in groups
        ],
        "index": _dedup_index_metadata(
            table,
            embedding_column=embedding_column,
            present=indexed,
            required=require_index or len(rows) >= index_min_rows,
            index_min_rows=index_min_rows,
        ),
        "search_strategy": search_strategy,
        "planner": {
            "indexed_shards": indexed_shards,
            "exact_shards": exact_shards,
            "pairs_considered": total_pairs_considered,
            "possible_pairs": total_possible_pairs,
            "all_pairs_scanned": bool(total_possible_pairs and not indexed_shards),
        },
    }
    return {
        "representative_ids": representative_ids,
        "dropped_scenario_ids": dropped_scenario_ids,
        "groups": groups,
        "report": report,
    }


def _distributed_semantic_dedup_payload(
    lake: Lake,
    *,
    rows: Sequence[dict[str, Any]],
    input_scenario_ids: Sequence[str],
    job_id: str,
    embedding_column: str,
    near_duplicate_threshold: float,
    representative_policy: Sequence[str],
    shard_by: Sequence[str],
    neighbor_limit: int,
    max_neighbor_limit: int | None,
    adaptive_neighbor_limit: bool,
    recall_audit_sample_size: int,
    recall_audit_seed: int,
    require_index: bool,
    index_min_rows: int,
    max_shards: int | None,
    prior_transform_ids: Sequence[str],
    created_by: str,
) -> dict[str, Any]:
    table = lake.table("scenarios")
    indexed = has_vector_index(table, embedding_column)
    if len(rows) > 1 and (require_index or len(rows) >= index_min_rows) and not indexed:
        raise CurationError(
            f"distributed semantic dedup over {len(rows)} candidates requires a persistent "
            f"vector index on scenarios.{embedding_column}; run `lancedb-robotics scenarios "
            f"index --column {embedding_column}` or lower index_min_rows for an explicit small scan"
        )

    input_versions = _dedup_source_table_versions(lake)
    vectors = {row["scenario_id"]: _vector(row, embedding_column) for row in rows}
    run_rows = _run_rows(lake)
    shards_by_label = _sharded_rows(rows, run_rows, shard_by)
    resolved_job_id = job_id or _distributed_dedup_job_id(
        input_scenario_ids=input_scenario_ids,
        embedding_column=embedding_column,
        near_duplicate_threshold=near_duplicate_threshold,
        representative_policy=representative_policy,
        shard_by=shard_by,
        neighbor_limit=neighbor_limit,
        max_neighbor_limit=max_neighbor_limit,
        adaptive_neighbor_limit=adaptive_neighbor_limit,
        recall_audit_sample_size=recall_audit_sample_size,
        input_versions=input_versions,
    )
    job_fingerprint = _distributed_dedup_job_fingerprint(
        job_id=resolved_job_id,
        input_scenario_ids=input_scenario_ids,
        embedding_column=embedding_column,
        near_duplicate_threshold=near_duplicate_threshold,
        representative_policy=representative_policy,
        shard_by=shard_by,
        neighbor_limit=neighbor_limit,
        max_neighbor_limit=max_neighbor_limit,
        adaptive_neighbor_limit=adaptive_neighbor_limit,
        recall_audit_sample_size=recall_audit_sample_size,
        input_versions=input_versions,
    )
    completed = _completed_distributed_dedup_shards(
        lake,
        job_id=resolved_job_id,
        job_fingerprint=job_fingerprint,
    )
    shard_results: list[DistributedDedupShard] = []
    pending_labels: list[str] = []
    executed_count = 0

    for shard_label, shard_rows in shards_by_label.items():
        shard_ids = tuple(row["scenario_id"] for row in shard_rows)
        shard_id = _distributed_dedup_shard_id(
            job_id=resolved_job_id,
            job_fingerprint=job_fingerprint,
            label=shard_label,
            scenario_ids=shard_ids,
        )
        if shard_id in completed:
            shard_results.append(completed[shard_id])
            continue
        if max_shards is not None and executed_count >= max_shards:
            pending_labels.append(shard_label)
            continue
        shard_results.append(
            _execute_distributed_dedup_shard(
                lake,
                job_id=resolved_job_id,
                job_fingerprint=job_fingerprint,
                shard_id=shard_id,
                shard_label=shard_label,
                shard_ids=shard_ids,
                vectors=vectors,
                embedding_column=embedding_column,
                near_duplicate_threshold=near_duplicate_threshold,
                representative_policy=representative_policy,
                neighbor_limit=neighbor_limit,
                max_neighbor_limit=max_neighbor_limit,
                adaptive_neighbor_limit=adaptive_neighbor_limit,
                recall_audit_sample_size=recall_audit_sample_size,
                recall_audit_seed=recall_audit_seed,
                indexed=indexed,
                prior_transform_ids=prior_transform_ids,
                created_by=created_by,
            )
        )
        executed_count += 1

    if pending_labels:
        raise CurationError(
            f"distributed dedup job {resolved_job_id!r} has {len(pending_labels)} pending "
            f"shard(s); rerun with job_id={resolved_job_id!r} to resume"
        )

    return _dedup_payload_from_shards(
        lake,
        rows=rows,
        input_scenario_ids=input_scenario_ids,
        job_id=resolved_job_id,
        job_fingerprint=job_fingerprint,
        embedding_column=embedding_column,
        near_duplicate_threshold=near_duplicate_threshold,
        representative_policy=representative_policy,
        shard_by=shard_by,
        neighbor_limit=neighbor_limit,
        max_neighbor_limit=max_neighbor_limit,
        adaptive_neighbor_limit=adaptive_neighbor_limit,
        recall_audit_sample_size=recall_audit_sample_size,
        require_index=require_index,
        index_min_rows=index_min_rows,
        indexed=indexed,
        shard_results=tuple(shard_results),
        executed_count=executed_count,
        input_versions=input_versions,
    )


def _distributed_dedup_job_id(
    *,
    input_scenario_ids: Sequence[str],
    embedding_column: str,
    near_duplicate_threshold: float,
    representative_policy: Sequence[str],
    shard_by: Sequence[str],
    neighbor_limit: int,
    max_neighbor_limit: int | None,
    adaptive_neighbor_limit: bool,
    recall_audit_sample_size: int,
    input_versions: Sequence[dict[str, Any]],
) -> str:
    return "dedup-job-" + _digest(
        {
            "input_scenario_ids": list(input_scenario_ids),
            "embedding_column": embedding_column,
            "near_duplicate_threshold": near_duplicate_threshold,
            "representative_policy": list(representative_policy),
            "shard_by": list(shard_by),
            "neighbor_limit": neighbor_limit,
            "max_neighbor_limit": max_neighbor_limit,
            "adaptive_neighbor_limit": adaptive_neighbor_limit,
            "recall_audit_sample_size": recall_audit_sample_size,
            "input_table_versions": list(input_versions),
        }
    )


def _distributed_dedup_job_fingerprint(
    *,
    job_id: str,
    input_scenario_ids: Sequence[str],
    embedding_column: str,
    near_duplicate_threshold: float,
    representative_policy: Sequence[str],
    shard_by: Sequence[str],
    neighbor_limit: int,
    max_neighbor_limit: int | None,
    adaptive_neighbor_limit: bool,
    recall_audit_sample_size: int,
    input_versions: Sequence[dict[str, Any]],
) -> str:
    return _digest(
        {
            "job_id": job_id,
            "input_scenario_ids": list(input_scenario_ids),
            "embedding_column": embedding_column,
            "near_duplicate_threshold": near_duplicate_threshold,
            "representative_policy": list(representative_policy),
            "shard_by": list(shard_by),
            "neighbor_limit": neighbor_limit,
            "max_neighbor_limit": max_neighbor_limit,
            "adaptive_neighbor_limit": adaptive_neighbor_limit,
            "recall_audit_sample_size": recall_audit_sample_size,
            "input_table_versions": list(input_versions),
        }
    )


def _distributed_dedup_shard_id(
    *,
    job_id: str,
    job_fingerprint: str,
    label: str,
    scenario_ids: Sequence[str],
) -> str:
    return "dedup-shard-" + _digest(
        {
            "job_id": job_id,
            "job_fingerprint": job_fingerprint,
            "label": label,
            "scenario_ids": list(scenario_ids),
        }
    )


def _completed_distributed_dedup_shards(
    lake: Lake,
    *,
    job_id: str,
    job_fingerprint: str,
) -> dict[str, DistributedDedupShard]:
    completed: dict[str, DistributedDedupShard] = {}
    for row in lake.table("transform_runs").to_arrow().to_pylist():
        if row.get("kind") != "curation-distributed-dedup-shard":
            continue
        if row.get("status") != "completed":
            continue
        try:
            params = json.loads(row.get("params") or "{}")
        except json.JSONDecodeError:
            continue
        if params.get("operation") != "distributed-dedup-shard":
            continue
        if params.get("job_id") != job_id or params.get("job_fingerprint") != job_fingerprint:
            continue
        shard = params.get("shard") or {}
        shard_id = str(shard.get("id") or "")
        if not shard_id:
            continue
        completed[shard_id] = DistributedDedupShard(
            shard_id=shard_id,
            label=str(shard.get("label") or ""),
            scenario_ids=tuple(str(item) for item in shard.get("scenario_ids") or ()),
            edges=tuple(
                (
                    str(edge["left"]),
                    str(edge["right"]),
                    float(edge["similarity"]),
                )
                for edge in params.get("duplicate_edges") or ()
            ),
            comparisons=int(params.get("comparisons") or 0),
            possible_pairs=int(params.get("possible_pairs") or 0),
            mode=str(params.get("mode") or ""),
            transform_id=str(row.get("transform_id") or ""),
            report=params,
            resumed=True,
        )
    return completed


def _execute_distributed_dedup_shard(
    lake: Lake,
    *,
    job_id: str,
    job_fingerprint: str,
    shard_id: str,
    shard_label: str,
    shard_ids: Sequence[str],
    vectors: dict[str, list[float]],
    embedding_column: str,
    near_duplicate_threshold: float,
    representative_policy: Sequence[str],
    neighbor_limit: int,
    max_neighbor_limit: int | None,
    adaptive_neighbor_limit: bool,
    recall_audit_sample_size: int,
    recall_audit_seed: int,
    indexed: bool,
    prior_transform_ids: Sequence[str],
    created_by: str,
) -> DistributedDedupShard:
    started = time.perf_counter()
    possible_pairs = len(shard_ids) * (len(shard_ids) - 1) // 2
    if indexed and len(shard_ids) > 1:
        edge_result = _duplicate_edges_indexed_adaptive(
            lake,
            shard_ids=shard_ids,
            vectors=vectors,
            embedding_column=embedding_column,
            threshold=near_duplicate_threshold,
            neighbor_limit=neighbor_limit,
            max_neighbor_limit=max_neighbor_limit,
            adaptive=adaptive_neighbor_limit,
        )
        edges = edge_result["edges"]
        comparisons = int(edge_result["comparisons"])
        mode = "indexed-neighbor"
    else:
        edges, comparisons = _duplicate_edges_exact(
            shard_ids=shard_ids,
            vectors=vectors,
            threshold=near_duplicate_threshold,
        )
        edge_result = {
            "initial_neighbor_limit": neighbor_limit,
            "final_neighbor_limit": None,
            "max_neighbor_limit": max_neighbor_limit,
            "expansions": 0,
            "saturated": False,
            "index_query": {
                "strategy": "exact",
                "embedding_column": embedding_column,
                "neighbor_limit": None,
            },
        }
        mode = "exact-small"
    recall_audit = _recall_audit(
        shard_ids=shard_ids,
        vectors=vectors,
        threshold=near_duplicate_threshold,
        found_edges=edges,
        sample_size=recall_audit_sample_size,
        seed=recall_audit_seed,
        shard_id=shard_id,
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    report = {
        "operation": "distributed-dedup-shard",
        "job_id": job_id,
        "job_fingerprint": job_fingerprint,
        "shard": {
            "id": shard_id,
            "label": shard_label,
            "scenario_ids": list(shard_ids),
            "scenario_count": len(shard_ids),
        },
        "embedding_column": embedding_column,
        "near_duplicate_threshold": near_duplicate_threshold,
        "representative_policy": list(representative_policy),
        "mode": mode,
        "comparisons": comparisons,
        "possible_pairs": possible_pairs,
        "duplicate_edge_count": len(edges),
        "duplicate_edges": [
            {"left": left, "right": right, "similarity": similarity}
            for left, right, similarity in edges
        ],
        "neighbor_limit": neighbor_limit,
        "max_neighbor_limit": max_neighbor_limit,
        "adaptive_neighbor_limit": adaptive_neighbor_limit,
        "initial_neighbor_limit": edge_result["initial_neighbor_limit"],
        "final_neighbor_limit": edge_result["final_neighbor_limit"],
        "neighbor_expansions": edge_result["expansions"],
        "neighbor_window_saturated": edge_result["saturated"],
        "index_query": edge_result["index_query"],
        "recall_audit": recall_audit,
        "timings": {"duration_ms": duration_ms},
        "input_count": len(shard_ids),
        "output_count": len(shard_ids),
    }
    transform_id = _record_stable_curation_transform(
        lake,
        operation="distributed-dedup-shard",
        stable_payload={
            "operation": "distributed-dedup-shard",
            "job_id": job_id,
            "job_fingerprint": job_fingerprint,
            "shard_id": shard_id,
        },
        stable_input_versions=_dedup_source_table_versions(lake),
        input_scenario_ids=shard_ids,
        output_scenario_ids=shard_ids,
        report=report,
        prior_transform_ids=prior_transform_ids,
        created_by=created_by,
    )
    return DistributedDedupShard(
        shard_id=shard_id,
        label=shard_label,
        scenario_ids=tuple(shard_ids),
        edges=tuple(edges),
        comparisons=comparisons,
        possible_pairs=possible_pairs,
        mode=mode,
        transform_id=transform_id,
        report=report,
        resumed=False,
    )


def _dedup_payload_from_shards(
    lake: Lake,
    *,
    rows: Sequence[dict[str, Any]],
    input_scenario_ids: Sequence[str],
    job_id: str,
    job_fingerprint: str,
    embedding_column: str,
    near_duplicate_threshold: float,
    representative_policy: Sequence[str],
    shard_by: Sequence[str],
    neighbor_limit: int,
    max_neighbor_limit: int | None,
    adaptive_neighbor_limit: bool,
    recall_audit_sample_size: int,
    require_index: bool,
    index_min_rows: int,
    indexed: bool,
    shard_results: Sequence[DistributedDedupShard],
    executed_count: int,
    input_versions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    table = lake.table("scenarios")
    row_by_id = {row["scenario_id"]: row for row in rows}
    vectors = {row["scenario_id"]: _vector(row, embedding_column) for row in rows}
    run_rows = _run_rows(lake)
    label_counts = _label_counts(lake)
    rarity_scores = _rarity_scores(rows, run_rows, shard_by)
    groups: list[dict[str, Any]] = []
    total_pairs_considered = 0
    total_possible_pairs = 0
    indexed_shards = 0
    exact_shards = 0
    neighbor_expansions = 0

    for shard in sorted(shard_results, key=lambda item: item.label):
        total_pairs_considered += shard.comparisons
        total_possible_pairs += shard.possible_pairs
        if shard.mode == "indexed-neighbor":
            indexed_shards += 1
        else:
            exact_shards += 1
        neighbor_expansions += int(shard.report.get("neighbor_expansions") or 0)
        for component_ids in _duplicate_components(shard.scenario_ids, shard.edges, row_by_id):
            representative_id = _representative_id(
                component_ids,
                rows=row_by_id,
                runs=run_rows,
                label_counts=label_counts,
                rarity_scores=rarity_scores,
                policy=representative_policy,
            )
            members = sorted(component_ids, key=lambda sid: _scenario_sort_key(row_by_id[sid]))
            dropped = [sid for sid in members if sid != representative_id]
            group_id = "dup-" + _digest(
                {
                    "embedding_column": embedding_column,
                    "threshold": near_duplicate_threshold,
                    "job_id": job_id,
                    "shard": shard.label,
                    "representative": representative_id,
                    "members": members,
                }
            )
            member_scores = []
            for rank, scenario_id in enumerate(members):
                similarity = (
                    1.0
                    if scenario_id == representative_id
                    else _cosine(vectors[representative_id], vectors[scenario_id])
                )
                member_scores.append(
                    {
                        "scenario_id": scenario_id,
                        "role": "representative"
                        if scenario_id == representative_id
                        else "duplicate",
                        "rank": rank,
                        "similarity_to_representative": similarity,
                        "distance_to_representative": 1.0 - similarity,
                        "quality_score": _numeric_value(row_by_id[scenario_id], "quality_score"),
                        "label_count": label_counts.get(scenario_id, 0),
                        "rarity_score": rarity_scores.get(scenario_id, 0.0),
                    }
                )
            groups.append(
                {
                    "group_id": group_id,
                    "shard": shard.label,
                    "representative": representative_id,
                    "members": members,
                    "dropped": dropped,
                    "scores": member_scores,
                }
            )

    groups.sort(key=lambda group: _scenario_sort_key(row_by_id[group["representative"]]))
    representative_set = {group["representative"] for group in groups}
    representative_ids = [
        row["scenario_id"]
        for row in sorted(rows, key=_scenario_sort_key)
        if row["scenario_id"] in representative_set
    ]
    dropped_scenario_ids = [
        scenario_id
        for group in groups
        for scenario_id in group["dropped"]
    ]
    search_strategy = "distributed-indexed-neighbor" if indexed_shards else "distributed-exact-small"
    shard_summaries = [_distributed_shard_summary(shard) for shard in shard_results]
    recall_audit = _aggregate_recall_audits(shard_results)
    report = {
        "operation": "distributed-dedup",
        "job_id": job_id,
        "job_fingerprint": job_fingerprint,
        "job_status": "completed",
        "embedding_column": embedding_column,
        "embedding_provenance": _embedding_provenance(lake, embedding_column),
        "near_duplicate_threshold": near_duplicate_threshold,
        "representative_policy": list(representative_policy),
        "shard_by": list(shard_by),
        "neighbor_limit": neighbor_limit,
        "max_neighbor_limit": max_neighbor_limit,
        "adaptive_neighbor_limit": adaptive_neighbor_limit,
        "recall_audit_sample_size": recall_audit_sample_size,
        "input_count": len(rows),
        "output_count": len(representative_ids),
        "kept_scenario_ids": representative_ids,
        "dropped_scenario_ids": dropped_scenario_ids,
        "duplicate_groups": groups,
        "clusters": [
            {
                "representative": group["representative"],
                "members": group["members"],
                "dropped": group["dropped"],
            }
            for group in groups
        ],
        "index": _dedup_index_metadata(
            table,
            embedding_column=embedding_column,
            present=indexed,
            required=require_index or len(rows) >= index_min_rows,
            index_min_rows=index_min_rows,
        ),
        "search_strategy": search_strategy,
        "planner": {
            "shard_count": len(shard_results),
            "completed_shard_count": len(shard_results),
            "executed_shard_count": executed_count,
            "resumed_shard_count": sum(1 for shard in shard_results if shard.resumed),
            "indexed_shards": indexed_shards,
            "exact_shards": exact_shards,
            "pairs_considered": total_pairs_considered,
            "possible_pairs": total_possible_pairs,
            "neighbor_expansions": neighbor_expansions,
            "all_pairs_scanned": bool(total_possible_pairs and not indexed_shards),
        },
        "recall_audit": recall_audit,
        "shards": shard_summaries,
        "shard_transform_ids": [shard.transform_id for shard in shard_results],
    }
    stable_transform_payload = {
        "operation": "distributed-dedup",
        "job_id": job_id,
        "job_fingerprint": job_fingerprint,
        "input_scenario_ids": list(input_scenario_ids),
        "output_scenario_ids": representative_ids,
        "shard_transform_ids": [shard.transform_id for shard in shard_results],
        "input_table_versions": list(input_versions),
    }
    return {
        "representative_ids": representative_ids,
        "dropped_scenario_ids": dropped_scenario_ids,
        "groups": groups,
        "report": report,
        "shard_transform_ids": [shard.transform_id for shard in shard_results],
        "stable_transform_payload": stable_transform_payload,
        "stable_input_versions": list(input_versions),
    }


def _distributed_shard_summary(shard: DistributedDedupShard) -> dict[str, Any]:
    recall_audit = shard.report.get("recall_audit") or {}
    return {
        "shard_id": shard.shard_id,
        "label": shard.label,
        "scenario_count": len(shard.scenario_ids),
        "mode": shard.mode,
        "comparisons": shard.comparisons,
        "possible_pairs": shard.possible_pairs,
        "duplicate_edge_count": len(shard.edges),
        "transform_id": shard.transform_id,
        "resumed": shard.resumed,
        "timings": shard.report.get("timings") or {},
        "index_query": shard.report.get("index_query") or {},
        "neighbor_limit": shard.report.get("neighbor_limit"),
        "final_neighbor_limit": shard.report.get("final_neighbor_limit"),
        "neighbor_expansions": shard.report.get("neighbor_expansions", 0),
        "recall_audit": {
            "sampled_pairs": recall_audit.get("sampled_pairs", 0),
            "exact_duplicate_edges": recall_audit.get("exact_duplicate_edges", 0),
            "found_duplicate_edges": recall_audit.get("found_duplicate_edges", 0),
            "missed_edge_count": recall_audit.get("missed_edge_count", 0),
            "estimated_recall": recall_audit.get("estimated_recall"),
        },
    }


def _aggregate_recall_audits(shards: Sequence[DistributedDedupShard]) -> dict[str, Any]:
    audits = [shard.report.get("recall_audit") or {} for shard in shards]
    sampled_pairs = sum(int(audit.get("sampled_pairs") or 0) for audit in audits)
    exact_edges = sum(int(audit.get("exact_duplicate_edges") or 0) for audit in audits)
    found_edges = sum(int(audit.get("found_duplicate_edges") or 0) for audit in audits)
    missed_edges = sum(int(audit.get("missed_edge_count") or 0) for audit in audits)
    missed_samples: list[dict[str, Any]] = []
    for audit in audits:
        missed_samples.extend(audit.get("missed_edges") or [])
    estimated_recall = 1.0 if exact_edges == 0 else found_edges / exact_edges
    return {
        "sampled_pairs": sampled_pairs,
        "exact_duplicate_edges": exact_edges,
        "found_duplicate_edges": found_edges,
        "missed_edge_count": missed_edges,
        "estimated_recall": estimated_recall,
        "missed_edges": missed_samples[:20],
    }


def _sharded_rows(
    rows: Sequence[dict[str, Any]],
    run_rows: dict[str, dict[str, Any]],
    shard_by: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    if not shard_by:
        return {"all": sorted(rows, key=_scenario_sort_key)}
    shards: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = _slice_label(row, run_rows.get(row["run_id"], {}), shard_by)
        shards.setdefault(label, []).append(row)
    return {
        label: sorted(shard_rows, key=_scenario_sort_key)
        for label, shard_rows in sorted(shards.items())
    }


def _duplicate_edges_exact(
    *,
    shard_ids: Sequence[str],
    vectors: dict[str, list[float]],
    threshold: float,
) -> tuple[list[tuple[str, str, float]], int]:
    edges: list[tuple[str, str, float]] = []
    comparisons = 0
    for index, left_id in enumerate(shard_ids):
        for right_id in shard_ids[index + 1:]:
            comparisons += 1
            similarity = _cosine(vectors[left_id], vectors[right_id])
            if similarity >= threshold:
                edges.append((left_id, right_id, similarity))
    return edges, comparisons


def _duplicate_edges_indexed(
    lake: Lake,
    *,
    shard_ids: Sequence[str],
    vectors: dict[str, list[float]],
    embedding_column: str,
    threshold: float,
    neighbor_limit: int,
) -> tuple[list[tuple[str, str, float]], int]:
    result = _duplicate_edges_indexed_scan(
        lake,
        shard_ids=shard_ids,
        vectors=vectors,
        embedding_column=embedding_column,
        threshold=threshold,
        neighbor_limit=neighbor_limit,
    )
    return result["edges"], int(result["comparisons"])


def _duplicate_edges_indexed_adaptive(
    lake: Lake,
    *,
    shard_ids: Sequence[str],
    vectors: dict[str, list[float]],
    embedding_column: str,
    threshold: float,
    neighbor_limit: int,
    max_neighbor_limit: int | None,
    adaptive: bool,
) -> dict[str, Any]:
    max_possible = max(0, len(shard_ids) - 1)
    if max_possible == 0:
        return {
            "edges": [],
            "comparisons": 0,
            "initial_neighbor_limit": neighbor_limit,
            "final_neighbor_limit": 0,
            "max_neighbor_limit": 0,
            "expansions": 0,
            "saturated": False,
            "index_query": {
                "strategy": "indexed-neighbor",
                "embedding_column": embedding_column,
                "neighbor_limit": 0,
                "nprobes": 64,
                "refine_factor": 50,
            },
        }
    resolved_max = (
        max_neighbor_limit
        if max_neighbor_limit is not None
        else min(max_possible, max(neighbor_limit, neighbor_limit * 4))
    )
    resolved_max = max(1, min(max_possible, resolved_max))
    current_limit = max(1, min(neighbor_limit, max_possible))
    expansions = 0
    result: dict[str, Any] | None = None
    while True:
        result = _duplicate_edges_indexed_scan(
            lake,
            shard_ids=shard_ids,
            vectors=vectors,
            embedding_column=embedding_column,
            threshold=threshold,
            neighbor_limit=current_limit,
        )
        if not adaptive or not result["saturated"] or current_limit >= resolved_max:
            break
        next_limit = min(resolved_max, max_possible, max(current_limit + 1, current_limit * 2))
        if next_limit <= current_limit:
            break
        current_limit = next_limit
        expansions += 1
    assert result is not None
    return {
        **result,
        "initial_neighbor_limit": neighbor_limit,
        "final_neighbor_limit": current_limit,
        "max_neighbor_limit": resolved_max,
        "expansions": expansions,
    }


def _duplicate_edges_indexed_scan(
    lake: Lake,
    *,
    shard_ids: Sequence[str],
    vectors: dict[str, list[float]],
    embedding_column: str,
    threshold: float,
    neighbor_limit: int,
) -> dict[str, Any]:
    table = lake.table("scenarios")
    shard_set = set(shard_ids)
    seen_pairs: set[tuple[str, str]] = set()
    edges: list[tuple[str, str, float]] = []
    comparisons = 0
    fetch_limit = min(table.count_rows(), neighbor_limit + 1)
    effective_limit = min(max(0, len(shard_ids) - 1), neighbor_limit)
    saturated_queries = 0
    candidate_counts: dict[str, int] = {}
    for scenario_id in shard_ids:
        builder = table.search(vectors[scenario_id], query_type="vector")
        try:
            builder = builder.nprobes(64).refine_factor(50)
        except AttributeError:
            pass
        try:
            builder = builder.where(_sql_in_predicate("scenario_id", shard_ids))
        except Exception:
            pass
        valid_neighbor_count = 0
        for neighbor in builder.limit(fetch_limit).to_list():
            neighbor_id = str(neighbor.get("scenario_id") or "")
            if not neighbor_id or neighbor_id == scenario_id or neighbor_id not in shard_set:
                continue
            valid_neighbor_count += 1
            pair = tuple(sorted((scenario_id, neighbor_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            comparisons += 1
            similarity = _cosine(vectors[scenario_id], vectors[neighbor_id])
            if similarity >= threshold:
                edges.append((scenario_id, neighbor_id, similarity))
        candidate_counts[scenario_id] = valid_neighbor_count
        if effective_limit and valid_neighbor_count >= effective_limit and effective_limit < len(shard_ids) - 1:
            saturated_queries += 1
    return {
        "edges": edges,
        "comparisons": comparisons,
        "saturated": saturated_queries > 0,
        "saturated_queries": saturated_queries,
        "candidate_counts": candidate_counts,
        "initial_neighbor_limit": neighbor_limit,
        "final_neighbor_limit": neighbor_limit,
        "max_neighbor_limit": neighbor_limit,
        "expansions": 0,
        "index_query": {
            "strategy": "indexed-neighbor",
            "embedding_column": embedding_column,
            "neighbor_limit": neighbor_limit,
            "fetch_limit": fetch_limit,
            "nprobes": 64,
            "refine_factor": 50,
            "saturated_queries": saturated_queries,
        },
    }


def _recall_audit(
    *,
    shard_ids: Sequence[str],
    vectors: dict[str, list[float]],
    threshold: float,
    found_edges: Sequence[tuple[str, str, float]],
    sample_size: int,
    seed: int,
    shard_id: str,
) -> dict[str, Any]:
    if sample_size <= 0 or len(shard_ids) < 2:
        return {
            "sampled_pairs": 0,
            "exact_duplicate_edges": 0,
            "found_duplicate_edges": 0,
            "missed_edge_count": 0,
            "estimated_recall": 1.0,
            "missed_edges": [],
        }
    pairs = [
        (left_id, right_id)
        for index, left_id in enumerate(shard_ids)
        for right_id in shard_ids[index + 1:]
    ]
    sampled = _sample_recall_pairs(
        pairs,
        sample_size=sample_size,
        seed=seed,
        shard_id=shard_id,
    )
    found = {tuple(sorted((left, right))) for left, right, _ in found_edges}
    exact_duplicate_edges = 0
    found_duplicate_edges = 0
    missed_edges: list[dict[str, Any]] = []
    for left_id, right_id in sampled:
        similarity = _cosine(vectors[left_id], vectors[right_id])
        if similarity < threshold:
            continue
        exact_duplicate_edges += 1
        pair = tuple(sorted((left_id, right_id)))
        if pair in found:
            found_duplicate_edges += 1
        else:
            missed_edges.append(
                {
                    "left": left_id,
                    "right": right_id,
                    "similarity": similarity,
                }
            )
    estimated_recall = (
        1.0
        if exact_duplicate_edges == 0
        else found_duplicate_edges / exact_duplicate_edges
    )
    return {
        "sampled_pairs": len(sampled),
        "exact_duplicate_edges": exact_duplicate_edges,
        "found_duplicate_edges": found_duplicate_edges,
        "missed_edge_count": len(missed_edges),
        "estimated_recall": estimated_recall,
        "missed_edges": missed_edges[:20],
    }


def _sample_recall_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    sample_size: int,
    seed: int,
    shard_id: str,
) -> list[tuple[str, str]]:
    if sample_size >= len(pairs):
        return list(pairs)
    return sorted(
        pairs,
        key=lambda pair: _digest(
            {
                "seed": seed,
                "shard_id": shard_id,
                "left": pair[0],
                "right": pair[1],
            }
        ),
    )[:sample_size]


def _duplicate_components(
    scenario_ids: Sequence[str],
    edges: Sequence[tuple[str, str, float]],
    row_by_id: dict[str, dict[str, Any]],
) -> list[tuple[str, ...]]:
    parent = {scenario_id: scenario_id for scenario_id in scenario_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right, _ in edges:
        union(left, right)

    components: dict[str, list[str]] = {}
    for scenario_id in scenario_ids:
        components.setdefault(find(scenario_id), []).append(scenario_id)
    ordered = [
        tuple(sorted(member_ids, key=lambda sid: _scenario_sort_key(row_by_id[sid])))
        for member_ids in components.values()
    ]
    return sorted(ordered, key=lambda ids: _scenario_sort_key(row_by_id[ids[0]]))


def _representative_id(
    scenario_ids: Sequence[str],
    *,
    rows: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    label_counts: dict[str, int],
    rarity_scores: dict[str, float],
    policy: Sequence[str],
) -> str:
    return sorted(
        scenario_ids,
        key=lambda sid: _representative_policy_key(
            rows[sid],
            runs.get(rows[sid]["run_id"], {}),
            label_counts=label_counts,
            rarity_scores=rarity_scores,
            policy=policy,
        ),
    )[0]


def _normalize_representative_policy(policy: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(policy, str):
        raw_tokens = [part.strip() for part in policy.replace(";", ",").split(",")]
    else:
        raw_tokens = [str(part).strip() for part in policy]
    normalized: list[str] = []
    for token in raw_tokens:
        if not token:
            continue
        lowered = token.lower().replace(" ", "-")
        lowered = _REPRESENTATIVE_POLICY_ALIASES.get(lowered, lowered)
        if lowered not in _REPRESENTATIVE_POLICY_TOKENS:
            raise CurationError(
                f"unknown representative policy token {token!r}; expected one of "
                f"{', '.join(_REPRESENTATIVE_POLICY_TOKENS)}"
            )
        normalized.append(lowered)
    if not normalized:
        normalized = list(_DEFAULT_REPRESENTATIVE_POLICY)
    if "scenario_id" not in normalized:
        normalized.append("scenario_id")
    return tuple(dict.fromkeys(normalized))


def _representative_policy_key(
    row: dict[str, Any],
    run: dict[str, Any],
    *,
    label_counts: dict[str, int],
    rarity_scores: dict[str, float],
    policy: Sequence[str],
) -> tuple[Any, ...]:
    scenario_id = str(row["scenario_id"])
    key: list[Any] = []
    for token in policy:
        if token == "quality":
            key.append(-(_numeric_value(row, "quality_score") or 0.0))
        elif token == "labels":
            key.append(-label_counts.get(scenario_id, 0))
        elif token == "rarity":
            key.append(-rarity_scores.get(scenario_id, 0.0))
        elif token == "earliest":
            key.append(_scenario_time_value(row))
        elif token == "latest":
            key.append(-_scenario_time_value(row))
        elif token == "scenario_id":
            key.append(scenario_id)
    return tuple(key)


def _scenario_time_value(row: dict[str, Any]) -> int:
    value = row.get("start_time_ns")
    if value is not None:
        return int(value)
    created_at = row.get("created_at")
    if isinstance(created_at, datetime):
        return int(created_at.timestamp() * 1_000_000_000)
    return 0


def _label_counts(lake: Lake) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in lake.table("labels").to_arrow().to_pylist():
        scenario_id = str(row.get("scenario_id") or "")
        if scenario_id:
            counts[scenario_id] = counts.get(scenario_id, 0) + 1
    return counts


def _rarity_scores(
    rows: Sequence[dict[str, Any]],
    run_rows: dict[str, dict[str, Any]],
    dimensions: Sequence[str],
) -> dict[str, float]:
    rarity_dimensions = tuple(dimensions) or ("site_id", "task_id", "scenario_type")
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for row in rows:
        label = _slice_label(row, run_rows.get(row["run_id"], {}), rarity_dimensions)
        labels[row["scenario_id"]] = label
        counts[label] = counts.get(label, 0) + 1
    return {
        scenario_id: 1.0 / counts[label]
        for scenario_id, label in labels.items()
        if counts.get(label)
    }


def _dedup_index_metadata(
    table: Any,
    *,
    embedding_column: str,
    present: bool,
    required: bool,
    index_min_rows: int,
) -> dict[str, Any]:
    return {
        "table": "scenarios",
        "column": embedding_column,
        "present": present,
        "required": required,
        "index_min_rows": index_min_rows,
        "indexed_columns": sorted(vector_index_columns(table)),
        "table_version": int(table.version),
    }


def _embedding_provenance(lake: Lake, column: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for row in lake.table("transform_runs").to_arrow().to_pylist():
        try:
            params = json.loads(row["params"] or "{}")
        except json.JSONDecodeError:
            continue
        if params.get("embedding_column") != column and params.get("column") != column:
            continue
        provider = params.get("embedding_provider") or params.get("provider") or ""
        matches.append(
            {
                "column": column,
                "provider": provider,
                "provider_version": params.get("embedding_provider_version")
                or params.get("provider_version")
                or "",
                "dimension": params.get("embedding_dimension") or params.get("dimension"),
                "transform_id": row.get("transform_id") or "",
                "created_at": row.get("created_at"),
            }
        )
    if not matches:
        return {"column": column, "provider": "", "provider_version": "", "transform_id": ""}
    latest = max(
        matches,
        key=lambda item: (
            item.get("created_at") or datetime.min.replace(tzinfo=UTC),
            item["transform_id"],
        ),
    )
    latest.pop("created_at", None)
    return latest


def _persist_semantic_dedup_decisions(
    selection: CurationSelection,
    plan: SemanticDedupPlan,
    *,
    view_name: str | None,
    created_by: str,
) -> str:
    audited_groups = [group for group in plan.groups if group["dropped"]]
    if not audited_groups:
        return ""
    resolved_view_name = view_name or f"semantic-dedup-{plan.transform_id[-16:]}"
    view = selection.save_view(
        resolved_view_name,
        tags=("semantic-dedup",),
        description="Automatic semantic dedup representative and duplicate decisions.",
        created_by=created_by,
    )
    report = {
        "operation": "dedup-decisions",
        "view_id": view.view_id,
        "view_name": view.name,
        "dedup_transform_id": plan.transform_id,
        "group_count": len(audited_groups),
        "representative_count": len(audited_groups),
        "dropped_count": len(plan.dropped_scenario_ids),
        "input_count": len(plan.input_scenario_ids),
        "output_count": len(plan.representative_ids),
    }
    transform_id = _record_curation_transform(
        selection.lake,
        operation="dedup-decisions",
        input_scenario_ids=plan.input_scenario_ids,
        output_scenario_ids=plan.representative_ids,
        report=report,
        prior_transform_ids=selection.operation_transform_ids + (plan.transform_id, view.transform_id),
        created_by=created_by,
        output_tables=("curation_memberships",),
    )
    existing = _membership_rows(selection.lake, view_id=view.view_id)
    latest_by_target = _latest_membership_by_target(existing)
    now = datetime.now(UTC)
    records: list[dict[str, Any]] = []
    membership_ids: list[str] = []
    for group in audited_groups:
        representative_id = str(group["representative"])
        for score in group["scores"]:
            scenario_id = str(score["scenario_id"])
            role = str(score["role"])
            decision = "include" if role == "representative" else "exclude"
            reason_code = (
                "semantic-dedup-representative"
                if role == "representative"
                else "semantic-duplicate"
            )
            previous = latest_by_target.get(("scenario", scenario_id), {})
            membership_id = "mbr-" + _digest(
                {
                    "view_id": view.view_id,
                    "target_id": scenario_id,
                    "decision": decision,
                    "reason_code": reason_code,
                    "group_id": group["group_id"],
                    "dedup_transform_id": plan.transform_id,
                }
            )
            supersedes_membership_id = previous.get("membership_id", "")
            if supersedes_membership_id == membership_id:
                supersedes_membership_id = ""
            membership_ids.append(membership_id)
            records.append(
                {
                    "membership_id": membership_id,
                    "view_id": view.view_id,
                    "target_grain": "scenario",
                    "target_id": scenario_id,
                    "scenario_id": scenario_id,
                    "decision": decision,
                    "reason_code": reason_code,
                    "reason": "semantic dedup",
                    "note": f"{role} in duplicate group {group['group_id']}",
                    "reviewer": "",
                    "queue": "",
                    "priority": 0,
                    "score": float(score["similarity_to_representative"]),
                    "metadata": _metadata_items(
                        {
                            "group_id": group["group_id"],
                            "representative_id": representative_id,
                            "role": role,
                            "rank": score["rank"],
                            "embedding_column": plan.report["embedding_column"],
                            "threshold": plan.report["near_duplicate_threshold"],
                            "representative_policy": plan.report["representative_policy"],
                            "dedup_transform_id": plan.transform_id,
                        }
                    ),
                    "source": "dedup",
                    "supersedes_membership_id": supersedes_membership_id,
                    "created_by": created_by,
                    "transform_id": transform_id,
                    "created_at": now,
                }
            )
    memberships = selection.lake.table("curation_memberships")
    for membership_id in membership_ids:
        memberships.delete(f"membership_id = '{membership_id}'")
    memberships.add(pa.Table.from_pylist(records, schema=CURATION_MEMBERSHIPS_SCHEMA))
    return transform_id


def _select_diverse_ids(
    candidate_ids: Sequence[str],
    *,
    limit: int,
    vectors: dict[str, list[float]],
    rows: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    dimensions: Sequence[str],
    min_per_slice: int,
    max_per_duplicate_group: int,
    duplicate_group_by_scenario: dict[str, str],
    representative_policy: Sequence[str],
) -> list[str]:
    candidates = list(dict.fromkeys(candidate_ids))
    label_counts: dict[str, int] = {}
    rarity_scores = _rarity_scores(list(rows.values()), runs, dimensions)
    selected: list[str] = []
    group_counts: dict[str, int] = {}

    def select(scenario_id: str) -> None:
        selected.append(scenario_id)
        group_id = duplicate_group_by_scenario.get(scenario_id, scenario_id)
        group_counts[group_id] = group_counts.get(group_id, 0) + 1

    if dimensions and min_per_slice:
        by_slice: dict[str, list[str]] = {}
        for scenario_id in candidates:
            row = rows[scenario_id]
            label = _slice_label(row, runs.get(row["run_id"], {}), dimensions)
            by_slice.setdefault(label, []).append(scenario_id)
        for label in sorted(by_slice):
            while (
                len(selected) < limit
                and sum(1 for sid in selected if sid in by_slice[label]) < min_per_slice
            ):
                pool = [
                    sid
                    for sid in by_slice[label]
                    if sid not in selected
                    and _can_select_duplicate_group(
                        sid,
                        group_counts=group_counts,
                        duplicate_group_by_scenario=duplicate_group_by_scenario,
                        max_per_group=max_per_duplicate_group,
                    )
                ]
                if not pool:
                    break
                select(
                    _farthest_candidate_id(
                        pool,
                        selected,
                        vectors=vectors,
                        rows=rows,
                        runs=runs,
                        label_counts=label_counts,
                        rarity_scores=rarity_scores,
                        policy=representative_policy,
                    )
                )

    while len(selected) < limit:
        pool = [
            sid
            for sid in candidates
            if sid not in selected
            and _can_select_duplicate_group(
                sid,
                group_counts=group_counts,
                duplicate_group_by_scenario=duplicate_group_by_scenario,
                max_per_group=max_per_duplicate_group,
            )
        ]
        if not pool:
            break
        select(
            _farthest_candidate_id(
                pool,
                selected,
                vectors=vectors,
                rows=rows,
                runs=runs,
                label_counts=label_counts,
                rarity_scores=rarity_scores,
                policy=representative_policy,
            )
        )
    return selected


def _can_select_duplicate_group(
    scenario_id: str,
    *,
    group_counts: dict[str, int],
    duplicate_group_by_scenario: dict[str, str],
    max_per_group: int,
) -> bool:
    group_id = duplicate_group_by_scenario.get(scenario_id, scenario_id)
    return group_counts.get(group_id, 0) < max_per_group


def _farthest_candidate_id(
    candidates: Sequence[str],
    selected: Sequence[str],
    *,
    vectors: dict[str, list[float]],
    rows: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    label_counts: dict[str, int],
    rarity_scores: dict[str, float],
    policy: Sequence[str],
) -> str:
    if not selected:
        return sorted(
            candidates,
            key=lambda sid: _representative_policy_key(
                rows[sid],
                runs.get(rows[sid]["run_id"], {}),
                label_counts=label_counts,
                rarity_scores=rarity_scores,
                policy=policy,
            ),
        )[0]
    return sorted(
        candidates,
        key=lambda sid: (
            -min(1.0 - _cosine(vectors[sid], vectors[selected_id]) for selected_id in selected),
            _representative_policy_key(
                rows[sid],
                runs.get(rows[sid]["run_id"], {}),
                label_counts=label_counts,
                rarity_scores=rarity_scores,
                policy=policy,
            ),
        ),
    )[0]


def _duplicate_group_by_scenario(groups: Sequence[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group in groups:
        for scenario_id in group["members"]:
            mapping[str(scenario_id)] = str(group["group_id"])
    return mapping


def _duplicate_group_usage(
    scenario_ids: Sequence[str],
    groups: Sequence[dict[str, Any]],
) -> dict[str, int]:
    group_by_scenario = _duplicate_group_by_scenario(groups)
    usage: dict[str, int] = {}
    for scenario_id in scenario_ids:
        group_id = group_by_scenario.get(scenario_id, scenario_id)
        usage[group_id] = usage.get(group_id, 0) + 1
    return dict(sorted(usage.items()))


def _normalize_diversity_constraint_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise CurationError("constraint_spec must be a mapping")
    minimum = _normalize_constraint_count_map(
        spec.get("minimum") or spec.get("min") or spec.get("min_counts") or {}
    )
    maximum = _normalize_constraint_count_map(
        spec.get("maximum")
        or spec.get("max")
        or spec.get("budgets")
        or spec.get("per_slice_budgets")
        or {}
    )
    for label, count in _normalize_required_constraints(spec.get("required") or {}).items():
        minimum[label] = max(minimum.get(label, 0), count)
    weights = _normalize_constraint_weight_map(spec.get("weights") or spec.get("weighted") or {})
    raw_dimensions = spec.get("dimensions") or spec.get("by") or ()
    dimensions = [str(dimension) for dimension in _as_tuple(raw_dimensions) if str(dimension)]
    dimensions.extend(_dimensions_from_constraint_labels([*minimum, *maximum, *weights]))

    quality_spec = spec.get("quality", {})
    quality_column = "quality_score"
    quality_weight = 1.0
    if isinstance(quality_spec, Mapping):
        quality_column = str(quality_spec.get("column") or quality_column)
        quality_weight = _nonnegative_float(quality_spec.get("weight", quality_weight), "quality.weight")
    elif quality_spec not in (None, False):
        quality_weight = _nonnegative_float(quality_spec, "quality")

    label_spec = spec.get("label_completeness", spec.get("labels", {}))
    label_weight = 1.0
    if isinstance(label_spec, Mapping):
        label_weight = _nonnegative_float(label_spec.get("weight", label_weight), "label_completeness.weight")
    elif label_spec not in (None, False):
        label_weight = _nonnegative_float(label_spec, "label_completeness")

    return {
        "dimensions": list(dict.fromkeys(dimensions)),
        "minimum": dict(sorted(minimum.items())),
        "maximum": dict(sorted(maximum.items())),
        "weights": dict(sorted(weights.items())),
        "quality": {"column": quality_column, "weight": quality_weight},
        "label_completeness": {"weight": label_weight},
    }


def _normalize_constraint_count_map(raw: Any) -> dict[str, int]:
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        raise CurationError("constraint count maps must be mappings")
    normalized: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping):
            dimension = str(key)
            for member, count in value.items():
                label = _constraint_label(dimension, member)
                normalized[label] = _nonnegative_int(count, f"constraint {label}")
        else:
            label = str(key)
            normalized[label] = _nonnegative_int(value, f"constraint {label}")
    return normalized


def _normalize_required_constraints(raw: Any) -> dict[str, int]:
    if not raw:
        return {}
    if isinstance(raw, Mapping):
        required: dict[str, int] = {}
        for key, value in raw.items():
            if isinstance(value, Mapping):
                for label, count in _normalize_constraint_count_map({key: value}).items():
                    required[label] = max(1, count)
            elif isinstance(value, bool):
                required[str(key)] = 1 if value else 0
            elif isinstance(value, (int, float)):
                required[str(key)] = max(1, _nonnegative_int(value, f"required {key}"))
            else:
                required[_constraint_label(str(key), value)] = 1
        return {label: count for label, count in required.items() if count > 0}
    return {str(label): 1 for label in _as_tuple(raw) if str(label)}


def _normalize_constraint_weight_map(raw: Any) -> dict[str, float]:
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        raise CurationError("constraint weights must be a mapping")
    normalized: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping):
            dimension = str(key)
            for member, weight in value.items():
                label = _constraint_label(dimension, member)
                normalized[label] = _nonnegative_float(weight, f"weight {label}")
        else:
            label = str(key)
            normalized[label] = _nonnegative_float(value, f"weight {label}")
    return normalized


def _constraint_label(dimension: str, member: Any) -> str:
    member_label = str(member)
    if "=" in member_label:
        return member_label
    return f"{dimension}={member_label}"


def _nonnegative_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CurationError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise CurationError(f"{field} must be non-negative")
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CurationError(f"{field} must be numeric") from exc
    if parsed < 0:
        raise CurationError(f"{field} must be non-negative")
    return parsed


def _dimensions_from_constraint_labels(labels: Iterable[str]) -> list[str]:
    dimensions: list[str] = []
    for label in labels:
        for dimension in _constraint_label_dimensions(label):
            dimensions.append(dimension)
    return list(dict.fromkeys(dimensions))


def _constraint_label_dimensions(label: str) -> tuple[str, ...]:
    dimensions: list[str] = []
    for part in str(label).split("|"):
        if "=" not in part:
            continue
        dimension, _ = part.split("=", 1)
        if dimension:
            dimensions.append(dimension)
    return tuple(dimensions)


def _constraint_labels_for_row(
    row: dict[str, Any],
    run: dict[str, Any],
    *,
    dimensions: Sequence[str],
    requested_labels: Sequence[str],
) -> set[str]:
    labels = {
        f"{dimension}={_dimension_value(row, run, dimension)}"
        for dimension in dimensions
    }
    for requested in requested_labels:
        requested_dimensions = _constraint_label_dimensions(requested)
        if len(requested_dimensions) > 1:
            labels.add(_slice_label(row, run, requested_dimensions))
        elif len(requested_dimensions) == 1:
            dimension = requested_dimensions[0]
            labels.add(f"{dimension}={_dimension_value(row, run, dimension)}")
    return labels


def _constraint_label_maps(
    candidate_ids: Sequence[str],
    *,
    rows: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    constraint_spec: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, set[str]]]:
    requested_labels = tuple(
        dict.fromkeys(
            [
                *constraint_spec["minimum"].keys(),
                *constraint_spec["maximum"].keys(),
                *constraint_spec["weights"].keys(),
            ]
        )
    )
    labels_by_candidate = {
        scenario_id: _constraint_labels_for_row(
            rows[scenario_id],
            runs.get(rows[scenario_id]["run_id"], {}),
            dimensions=constraint_spec["dimensions"],
            requested_labels=requested_labels,
        )
        for scenario_id in candidate_ids
    }
    return requested_labels, labels_by_candidate


def _optimize_diversity_ids(
    candidate_ids: Sequence[str],
    *,
    limit: int,
    vectors: dict[str, list[float]],
    rows: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    constraint_spec: Mapping[str, Any],
    duplicate_group_by_scenario: dict[str, str],
    max_per_duplicate_group: int,
    representative_policy: Sequence[str],
    lake: Lake,
) -> tuple[list[str], dict[str, Any]]:
    candidates = list(dict.fromkeys(candidate_ids))
    _, labels_by_candidate = _constraint_label_maps(
        candidates,
        rows=rows,
        runs=runs,
        constraint_spec=constraint_spec,
    )
    minimum = constraint_spec["minimum"]
    maximum = constraint_spec["maximum"]
    weights = constraint_spec["weights"]
    selected: list[str] = []
    group_counts: dict[str, int] = {}
    selected_label_counts: dict[str, int] = {}
    scenario_label_counts = _label_counts(lake)
    rarity_scores = _rarity_scores(list(rows.values()), runs, constraint_spec["dimensions"])
    trace: list[dict[str, Any]] = []

    def select(scenario_id: str, phase: str) -> None:
        selected.append(scenario_id)
        group_id = duplicate_group_by_scenario.get(scenario_id, scenario_id)
        group_counts[group_id] = group_counts.get(group_id, 0) + 1
        for label in labels_by_candidate[scenario_id]:
            selected_label_counts[label] = selected_label_counts.get(label, 0) + 1
        row = rows[scenario_id]
        trace.append(
            {
                "scenario_id": scenario_id,
                "phase": phase,
                "constraint_labels": sorted(
                    label
                    for label in labels_by_candidate[scenario_id]
                    if label in minimum or label in maximum or label in weights
                ),
                "quality_score": _numeric_value(row, constraint_spec["quality"]["column"]),
                "label_count": scenario_label_counts.get(scenario_id, 0),
            }
        )

    def can_select(scenario_id: str) -> bool:
        if scenario_id in selected:
            return False
        group_id = duplicate_group_by_scenario.get(scenario_id, scenario_id)
        if group_counts.get(group_id, 0) >= max_per_duplicate_group:
            return False
        for label in labels_by_candidate[scenario_id]:
            if label in maximum and selected_label_counts.get(label, 0) + 1 > maximum[label]:
                return False
        return True

    while len(selected) < limit:
        unmet = {
            label
            for label, target in minimum.items()
            if selected_label_counts.get(label, 0) < target
        }
        if not unmet:
            break
        pool = [
            scenario_id
            for scenario_id in candidates
            if can_select(scenario_id) and labels_by_candidate[scenario_id] & unmet
        ]
        if not pool:
            break
        select(
            sorted(
                pool,
                key=lambda sid: _diversity_optimization_key(
                    sid,
                    selected,
                    labels_by_candidate=labels_by_candidate,
                    selected_label_counts=selected_label_counts,
                    unmet_labels=unmet,
                    weights=weights,
                    minimum=minimum,
                    vectors=vectors,
                    rows=rows,
                    runs=runs,
                    scenario_label_counts=scenario_label_counts,
                    rarity_scores=rarity_scores,
                    representative_policy=representative_policy,
                    quality_column=constraint_spec["quality"]["column"],
                    quality_weight=constraint_spec["quality"]["weight"],
                    label_weight=constraint_spec["label_completeness"]["weight"],
                ),
            )[0],
            "minimum-coverage",
        )

    while len(selected) < limit:
        unmet = {
            label
            for label, target in minimum.items()
            if selected_label_counts.get(label, 0) < target
        }
        pool = [scenario_id for scenario_id in candidates if can_select(scenario_id)]
        if not pool:
            break
        select(
            sorted(
                pool,
                key=lambda sid: _diversity_optimization_key(
                    sid,
                    selected,
                    labels_by_candidate=labels_by_candidate,
                    selected_label_counts=selected_label_counts,
                    unmet_labels=unmet,
                    weights=weights,
                    minimum=minimum,
                    vectors=vectors,
                    rows=rows,
                    runs=runs,
                    scenario_label_counts=scenario_label_counts,
                    rarity_scores=rarity_scores,
                    representative_policy=representative_policy,
                    quality_column=constraint_spec["quality"]["column"],
                    quality_weight=constraint_spec["quality"]["weight"],
                    label_weight=constraint_spec["label_completeness"]["weight"],
                ),
            )[0],
            "fill-budget",
        )

    report = _diversity_constraint_report(
        selected,
        candidate_ids=candidates,
        rows=rows,
        runs=runs,
        constraint_spec=constraint_spec,
    )
    report["selection_trace"] = trace
    return selected, report


def _diversity_optimization_key(
    scenario_id: str,
    selected: Sequence[str],
    *,
    labels_by_candidate: dict[str, set[str]],
    selected_label_counts: dict[str, int],
    unmet_labels: set[str],
    weights: Mapping[str, float],
    minimum: Mapping[str, int],
    vectors: dict[str, list[float]],
    rows: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    scenario_label_counts: Mapping[str, int],
    rarity_scores: Mapping[str, float],
    representative_policy: Sequence[str],
    quality_column: str,
    quality_weight: float,
    label_weight: float,
) -> tuple[Any, ...]:
    row = rows[scenario_id]
    labels = labels_by_candidate[scenario_id]
    unmet_score = 0.0
    for label in labels & unmet_labels:
        deficit = minimum[label] - selected_label_counts.get(label, 0)
        unmet_score += max(0, deficit) * (10.0 + weights.get(label, 1.0))
    weighted_score = sum(weights.get(label, 0.0) for label in labels)
    quality_score = (_numeric_value(row, quality_column) or 0.0) * quality_weight
    label_score = float(scenario_label_counts.get(scenario_id, 0)) * label_weight
    distance = 0.0
    if selected:
        distance = min(1.0 - _cosine(vectors[scenario_id], vectors[selected_id]) for selected_id in selected)
    return (
        -unmet_score,
        -weighted_score,
        -quality_score,
        -label_score,
        -distance,
        _representative_policy_key(
            row,
            runs.get(row["run_id"], {}),
            label_counts=dict(scenario_label_counts),
            rarity_scores=dict(rarity_scores),
            policy=representative_policy,
        ),
    )


def _diversity_constraint_report(
    selected: Sequence[str],
    *,
    candidate_ids: Sequence[str],
    rows: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    constraint_spec: Mapping[str, Any],
) -> dict[str, Any]:
    requested_labels, labels_by_candidate = _constraint_label_maps(
        candidate_ids,
        rows=rows,
        runs=runs,
        constraint_spec=constraint_spec,
    )
    feasible_counts: dict[str, int] = {label: 0 for label in requested_labels}
    actual_counts: dict[str, int] = {label: 0 for label in requested_labels}
    for scenario_id in candidate_ids:
        for label in labels_by_candidate[scenario_id]:
            if label in feasible_counts:
                feasible_counts[label] += 1
    for scenario_id in selected:
        for label in labels_by_candidate.get(scenario_id, set()):
            if label in actual_counts:
                actual_counts[label] += 1

    constraints: list[dict[str, Any]] = []
    for label, target in constraint_spec["minimum"].items():
        actual = actual_counts.get(label, 0)
        feasible = feasible_counts.get(label, 0)
        if actual >= target:
            status = "satisfied"
        elif feasible < target:
            status = "violated"
        else:
            status = "relaxed"
        constraints.append(
            {
                "kind": "minimum",
                "label": label,
                "target": target,
                "actual": actual,
                "feasible_count": feasible,
                "status": status,
                "relaxed_by": max(0, target - actual) if status == "relaxed" else 0,
                "violated_by": max(0, target - feasible) if status == "violated" else 0,
            }
        )
    for label, target in constraint_spec["maximum"].items():
        actual = actual_counts.get(label, 0)
        feasible = feasible_counts.get(label, 0)
        status = "satisfied" if actual <= target else "violated"
        constraints.append(
            {
                "kind": "maximum",
                "label": label,
                "target": target,
                "actual": actual,
                "feasible_count": feasible,
                "status": status,
                "relaxed_by": 0,
                "violated_by": max(0, actual - target),
            }
        )
    constraints.sort(key=lambda item: (item["status"] != "violated", item["kind"], item["label"]))
    summary = {"satisfied": 0, "relaxed": 0, "violated": 0}
    for item in constraints:
        summary[item["status"]] = summary.get(item["status"], 0) + 1
    return {
        "constraints": constraints,
        "constraint_summary": summary,
        "coverage_counts": dict(sorted(actual_counts.items())),
        "feasible_counts": dict(sorted(feasible_counts.items())),
    }


def _mean_pairwise_similarity(
    scenario_ids: Sequence[str],
    vectors: dict[str, list[float]],
) -> float:
    if len(scenario_ids) < 2:
        return 0.0
    total = 0.0
    count = 0
    for index, left_id in enumerate(scenario_ids):
        for right_id in scenario_ids[index + 1:]:
            total += _cosine(vectors[left_id], vectors[right_id])
            count += 1
    return total / count if count else 0.0


@dataclass(frozen=True)
class _SliceContext:
    """Per-scenario distribution-slice deficits for gap-aware scoring."""

    dimensions: tuple[str, ...] = ()
    slice_by_scenario: Mapping[str, str] = field(default_factory=dict)
    needed_by_scenario: Mapping[str, int] = field(default_factory=dict)


def _collect_scoring_candidates(
    lake: Lake,
    scenario_ids: Sequence[str],
    *,
    model_version: str,
    output_type: str,
    required_label_types: Sequence[str],
    calibration: Calibration,
    slice_context: _SliceContext,
) -> list[ScoringCandidate]:
    """Assemble immutable per-scenario signal bundles for the scorer.

    Bundles every matching ``model_outputs`` row per scenario (an ensemble
    scorer needs to see all of them), the present label types, and the
    distribution-slice deficit, so scorers stay pure and storage-agnostic.
    """
    wanted = set(scenario_ids)
    observation_to_scenario: dict[str, str] = {}
    for row in lake.table("scenarios").to_arrow().to_pylist():
        if row["scenario_id"] not in wanted:
            continue
        for observation_id in row.get("observation_ids") or ():
            observation_to_scenario[str(observation_id)] = str(row["scenario_id"])

    outputs_by_scenario: dict[str, list[ModelOutputSignal]] = {}
    for output in lake.table("model_outputs").to_arrow().to_pylist():
        if model_version and output.get("model_version") != model_version:
            continue
        if output_type and output.get("output_type") != output_type:
            continue
        scenario_id = str(output.get("scenario_id") or "")
        if not scenario_id and output.get("observation_id"):
            scenario_id = observation_to_scenario.get(str(output["observation_id"]), "")
        if scenario_id not in wanted:
            continue
        signal = ModelOutputSignal(
            model_output_id=str(output.get("model_output_id") or ""),
            model_version=str(output.get("model_version") or ""),
            output_type=str(output.get("output_type") or ""),
            prediction=str(output.get("prediction") or ""),
            score=_optional_float_value(output.get("score")),
            metadata=_metadata_dict(output.get("metadata") or ()),
        )
        outputs_by_scenario.setdefault(scenario_id, []).append(signal)

    required = tuple(dict.fromkeys(str(item) for item in required_label_types if str(item)))
    labels_by_scenario: dict[str, set[str]] = {}
    if required:
        for row in lake.table("labels").to_arrow().to_pylist():
            scenario_id = str(row.get("scenario_id") or "")
            if scenario_id in wanted:
                labels_by_scenario.setdefault(scenario_id, set()).add(str(row.get("label_type") or ""))

    # Scenarios with model outputs are always candidates. When required labels
    # or a distribution-gap context is in play, every selected scenario becomes
    # a candidate so the missing-labels / gap scorers can fire on rows that have
    # no model output at all.
    include_all = bool(required) or bool(slice_context.dimensions)
    candidate_ids = set(outputs_by_scenario)
    if include_all:
        candidate_ids.update(str(sid) for sid in scenario_ids)

    candidates: list[ScoringCandidate] = []
    for scenario_id in sorted(candidate_ids):
        outputs = tuple(
            sorted(outputs_by_scenario.get(scenario_id, ()), key=lambda s: s.model_output_id)
        )
        candidates.append(
            ScoringCandidate(
                scenario_id=scenario_id,
                outputs=outputs,
                label_types=frozenset(labels_by_scenario.get(scenario_id, set())),
                required_label_types=required,
                slice_label=str(slice_context.slice_by_scenario.get(scenario_id, "")),
                slice_needed=int(slice_context.needed_by_scenario.get(scenario_id, 0)),
                calibration=calibration,
            )
        )
    return candidates


def _active_learning_candidates(
    lake: Lake,
    scenario_ids: Sequence[str],
    *,
    model_version: str,
    output_type: str,
    required_label_types: Sequence[str],
    scorer: ActiveLearningScorer,
    calibration: Calibration,
    slice_context: _SliceContext | None = None,
) -> list[dict[str, Any]]:
    slice_context = slice_context or _SliceContext()
    candidates = _collect_scoring_candidates(
        lake,
        scenario_ids,
        model_version=model_version,
        output_type=output_type,
        required_label_types=required_label_types,
        calibration=calibration,
        slice_context=slice_context,
    )
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        result = scorer.score(candidate)
        if result is None:
            continue
        ranked.append(
            {
                "scenario_id": candidate.scenario_id,
                "priority_score": float(result.score),
                "priority_reason": str(result.reason),
                "model_output_id": str(result.model_output_id or ""),
                "source_ref": {
                    "kind": "active-learning-score",
                    "scorer": scorer.name,
                    "scorer_version": scorer.version,
                    "metric": result.metric,
                    "calibration": calibration.as_dict(),
                    "model_output_id": str(result.model_output_id or ""),
                    "components": {key: float(value) for key, value in result.components.items()},
                    "detail": dict(result.detail),
                },
            }
        )
    if not ranked:
        raise CurationError(
            f"scorer {scorer.name!r} selected no active-learning candidates; "
            "no uncertainty, loss, disagreement, severity, gap, or missing-label signal matched"
        )
    return sorted(ranked, key=_active_learning_sort_key)


def _active_learning_sort_key(candidate: dict[str, Any]) -> tuple[float, str, str]:
    return (
        -float(candidate["priority_score"]),
        str(candidate["scenario_id"]),
        str(candidate.get("model_output_id") or ""),
    )


def _cap_ranked_candidates_by_duplicate_group(
    candidates: Sequence[dict[str, Any]],
    *,
    group_by_scenario: dict[str, str],
    max_per_group: int,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    for candidate in candidates:
        scenario_id = str(candidate["scenario_id"])
        group_id = group_by_scenario.get(scenario_id, scenario_id)
        if usage.get(group_id, 0) >= max_per_group:
            continue
        selected.append(candidate)
        usage[group_id] = usage.get(group_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected, dict(sorted(usage.items()))


def _normalize_review_queue_source_operation(source_operation: str) -> str:
    normalized = str(source_operation or "manual").strip().lower().replace("_", "-")
    normalized = _REVIEW_QUEUE_SOURCE_ALIASES.get(normalized, normalized)
    if normalized not in _REVIEW_QUEUE_SOURCE_OPERATIONS:
        raise CurationError(
            f"unknown review queue source operation {source_operation!r}; expected one of "
            f"{', '.join(_REVIEW_QUEUE_SOURCE_OPERATIONS)}"
        )
    return normalized


def _normalize_review_queue_status(status: str) -> str:
    normalized = str(status or "open").strip().lower().replace("_", "-")
    if normalized not in _REVIEW_QUEUE_STATUSES:
        raise CurationError(
            f"unknown review queue status {status!r}; expected one of "
            f"{', '.join(_REVIEW_QUEUE_STATUSES)}"
        )
    return normalized


def _priority_score(
    scores: dict[str, float],
    *,
    target_id: str,
    scenario_id: str,
    fallback: float,
) -> float:
    value = scores.get(target_id, scores.get(scenario_id, fallback))
    return float(value)


def _optional_float_value(value: Any, fallback: Any | None = None) -> float | None:
    if value is None:
        value = fallback
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_decision(decision: str) -> str:
    normalized = str(decision).strip().lower().replace("_", "-")
    if normalized not in _MEMBERSHIP_DECISIONS:
        raise CurationError(
            f"unknown membership decision {decision!r}; expected one of "
            f"{', '.join(_MEMBERSHIP_DECISIONS)}"
        )
    return normalized


def _normalize_target_grain(target_grain: str) -> str:
    normalized = str(target_grain).strip().lower().replace("_", "-")
    if normalized not in _TARGET_GRAINS:
        raise CurationError(
            f"unknown membership target grain {target_grain!r}; expected one of "
            f"{', '.join(_TARGET_GRAINS)}"
        )
    return normalized


def _normalize_decision_source(source: str) -> str:
    normalized = str(source or "human").strip().lower().replace("_", "-")
    if normalized == "manual":
        normalized = "human"
    if normalized not in _DECISION_SOURCES:
        raise CurationError(
            f"unknown membership decision source {source!r}; expected one of "
            f"{', '.join(_DECISION_SOURCES)}"
        )
    return normalized


def _decision_target_ids(
    *,
    target_grain: str,
    target_ids: Sequence[str] | None,
    scenario_ids: Sequence[str] | None,
    default_scenario_ids: Sequence[str],
) -> tuple[str, ...]:
    if target_ids is not None:
        selected = tuple(dict.fromkeys(str(item) for item in target_ids if str(item)))
    elif target_grain == "scenario":
        selected = tuple(dict.fromkeys(str(item) for item in (scenario_ids or default_scenario_ids)))
    else:
        selected = ()
    if not selected:
        if target_grain == "scenario":
            raise CurationError("at least one scenario_id is required")
        raise CurationError(f"at least one target_id is required for {target_grain} decisions")
    return selected


def _validate_targets(
    lake: Lake,
    *,
    target_grain: str,
    target_ids: Sequence[str],
) -> None:
    if target_grain == "snapshot-row":
        return
    table_name, id_column = _TARGET_ID_COLUMNS[target_grain]
    known = {str(row[id_column]) for row in lake.table(table_name).to_arrow().to_pylist()}
    unknown = sorted(set(target_ids) - known)
    if unknown:
        raise CurationError(f"unknown {target_grain} target ids: {unknown}")


def _target_scenario_context(
    lake: Lake,
    *,
    target_grain: str,
    target_ids: Sequence[str],
    scenario_ids: Sequence[str] | None,
) -> dict[str, str]:
    explicit = tuple(dict.fromkeys(str(item) for item in (scenario_ids or ()) if str(item)))
    if target_grain == "scenario":
        return {target_id: target_id for target_id in target_ids}
    if explicit:
        if len(explicit) == 1:
            return {target_id: explicit[0] for target_id in target_ids}
        if len(explicit) == len(target_ids):
            return dict(zip(target_ids, explicit, strict=True))
        raise CurationError(
            "scenario_ids must be omitted, contain one context scenario, or match target_ids"
        )
    if target_grain == "observation":
        by_observation: dict[str, str] = {}
        for row in lake.table("scenarios").to_arrow().to_pylist():
            for observation_id in row.get("observation_ids") or ():
                by_observation.setdefault(str(observation_id), str(row["scenario_id"]))
        return {target_id: by_observation.get(target_id, "") for target_id in target_ids}
    if target_grain == "episode":
        return _episode_scenario_context(lake, target_ids)
    if target_grain == "aligned-frame":
        return _aligned_frame_scenario_context(lake, target_ids)
    return {target_id: "" for target_id in target_ids}


def _episode_scenario_context(lake: Lake, episode_ids: Sequence[str]) -> dict[str, str]:
    episodes = {
        str(row["episode_id"]): row
        for row in lake.table("episodes").to_arrow().to_pylist()
        if str(row["episode_id"]) in set(episode_ids)
    }
    scenarios = lake.table("scenarios").to_arrow().to_pylist()
    context: dict[str, str] = {}
    for episode_id in episode_ids:
        episode = episodes.get(episode_id)
        if not episode:
            context[episode_id] = ""
            continue
        context[episode_id] = _scenario_for_window(
            scenarios,
            run_id=str(episode.get("run_id") or ""),
            start_time_ns=int(episode.get("from_timestamp_ns") or 0),
            end_time_ns=int(episode.get("to_timestamp_ns") or 0),
        )
    return context


def _aligned_frame_scenario_context(lake: Lake, aligned_frame_ids: Sequence[str]) -> dict[str, str]:
    frames = {
        str(row["aligned_frame_id"]): row
        for row in lake.table("aligned_frames").to_arrow().to_pylist()
        if str(row["aligned_frame_id"]) in set(aligned_frame_ids)
    }
    scenarios = lake.table("scenarios").to_arrow().to_pylist()
    context: dict[str, str] = {}
    for aligned_frame_id in aligned_frame_ids:
        frame = frames.get(aligned_frame_id)
        if not frame:
            context[aligned_frame_id] = ""
            continue
        timestamp_ns = int(frame.get("timestamp_ns") or 0)
        context[aligned_frame_id] = _scenario_for_window(
            scenarios,
            run_id=str(frame.get("run_id") or ""),
            start_time_ns=timestamp_ns,
            end_time_ns=timestamp_ns,
        )
    return context


def _scenario_for_window(
    scenarios: Sequence[dict[str, Any]],
    *,
    run_id: str,
    start_time_ns: int,
    end_time_ns: int,
) -> str:
    matches = [
        row
        for row in scenarios
        if str(row.get("run_id") or "") == run_id
        and int(row.get("start_time_ns") or 0) <= end_time_ns
        and int(row.get("end_time_ns") or 0) >= start_time_ns
    ]
    if not matches:
        return ""
    return str(sorted(matches, key=_scenario_sort_key)[0]["scenario_id"])


def _metadata_items(metadata: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"key": str(key), "value": json.dumps(_jsonable(value), sort_keys=True)}
        for key, value in sorted(metadata.items())
    ]


def _metadata_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return {str(key): _jsonable(value) for key, value in sorted(metadata.items())}
    if isinstance(metadata, list):
        decoded: dict[str, Any] = {}
        for item in metadata:
            if not isinstance(item, dict) or "key" not in item:
                continue
            value = item.get("value")
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            decoded[str(item["key"])] = _jsonable(value)
        return decoded
    return {}


def _view_membership_storage_payload(
    scenario_ids: Sequence[str],
    *,
    inline_scenario_limit: int,
    chunk_size: int,
) -> dict[str, Any]:
    count = len(scenario_ids)
    if count <= inline_scenario_limit:
        return {
            "kind": _VIEW_STORAGE_INLINE,
            "table": "curation_views",
            "scenario_count": count,
            "inline_scenario_count": count,
            "chunk_count": 0,
            "chunk_size": 0,
            "order": "scenario_ids",
        }
    return {
        "kind": _VIEW_STORAGE_CHUNKED,
        "table": _VIEW_MEMBERSHIP_CHUNK_TABLE,
        "scenario_count": count,
        "inline_scenario_count": 0,
        "chunk_count": int(math.ceil(count / chunk_size)),
        "chunk_size": chunk_size,
        "order": "start_ordinal",
        "scenario_ids_digest": _digest({"scenario_ids": list(scenario_ids)}),
    }


def _view_membership_storage_from_row(row: dict[str, Any]) -> dict[str, Any]:
    try:
        query_spec = json.loads(row.get("query_spec") or "{}")
    except (TypeError, json.JSONDecodeError):
        query_spec = {}
    storage = query_spec.get("membership_storage")
    if isinstance(storage, dict) and storage.get("kind"):
        return {
            "kind": str(storage.get("kind") or _VIEW_STORAGE_INLINE),
            "table": str(storage.get("table") or "curation_views"),
            "scenario_count": int(storage.get("scenario_count") or 0),
            "inline_scenario_count": int(storage.get("inline_scenario_count") or 0),
            "chunk_count": int(storage.get("chunk_count") or 0),
            "chunk_size": int(storage.get("chunk_size") or 0),
            "order": str(storage.get("order") or "scenario_ids"),
            **(
                {"scenario_ids_digest": str(storage["scenario_ids_digest"])}
                if storage.get("scenario_ids_digest")
                else {}
            ),
        }
    scenario_count = len(row.get("scenario_ids") or ())
    return {
        "kind": _VIEW_STORAGE_INLINE,
        "table": "curation_views",
        "scenario_count": scenario_count,
        "inline_scenario_count": scenario_count,
        "chunk_count": 0,
        "chunk_size": 0,
        "order": "scenario_ids",
    }


def _view_membership_chunk_rows(
    *,
    view_id: str,
    scenario_ids: Sequence[str],
    chunk_size: int,
    created_by: str,
    transform_id: str,
    created_at: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, len(scenario_ids), chunk_size)):
        chunk_scenario_ids = list(scenario_ids[start : start + chunk_size])
        end = start + len(chunk_scenario_ids)
        chunk_digest = _digest(
            {
                "view_id": view_id,
                "chunk_index": chunk_index,
                "start_ordinal": start,
                "scenario_ids": chunk_scenario_ids,
            }
        )
        rows.append(
            {
                "chunk_id": f"viewchunk-{chunk_digest}",
                "view_id": view_id,
                "chunk_index": chunk_index,
                "start_ordinal": start,
                "end_ordinal": end,
                "scenario_ids": chunk_scenario_ids,
                "scenario_count": len(chunk_scenario_ids),
                "chunk_digest": chunk_digest,
                "created_by": created_by,
                "transform_id": transform_id,
                "created_at": created_at,
            }
        )
    return rows


def _rows_where(
    lake: Lake,
    table: str,
    *,
    where_sql: str | None,
    fallback_filter: Any,
) -> list[dict[str, Any]]:
    handle = lake.table(table)
    if where_sql:
        try:
            return handle.search().where(where_sql).to_arrow().to_pylist()
        except Exception:
            pass
    return [row for row in handle.to_arrow().to_pylist() if fallback_filter(row)]


def _sql_in_predicate(column: str, values: Sequence[str]) -> str:
    unique = tuple(dict.fromkeys(str(value) for value in values if str(value)))
    if not unique:
        return "FALSE"
    if len(unique) == 1:
        return f"{column} = {_sql_literal(unique[0])}"
    return f"{column} IN ({', '.join(_sql_literal(value) for value in unique)})"


def _view_membership_chunk_rows_for(lake: Lake, view_id: str) -> list[dict[str, Any]]:
    return _rows_where(
        lake,
        _VIEW_MEMBERSHIP_CHUNK_TABLE,
        where_sql=f"view_id = {_sql_literal(view_id)}",
        fallback_filter=lambda row: row["view_id"] == view_id,
    )


def _scenario_ids_from_view_chunks(lake: Lake, view_id: str) -> tuple[str, ...]:
    rows = sorted(
        _view_membership_chunk_rows_for(lake, view_id),
        key=lambda row: (
            int(row.get("start_ordinal") or 0),
            int(row.get("chunk_index") or 0),
            str(row.get("chunk_id") or ""),
        ),
    )
    scenario_ids: list[str] = []
    expected_start = 0
    for row in rows:
        start = int(row.get("start_ordinal") or 0)
        chunk_ids = [str(item) for item in row.get("scenario_ids") or ()]
        end = int(row.get("end_ordinal") or start + len(chunk_ids))
        if start != expected_start or end != start + len(chunk_ids):
            raise CurationError(
                f"curation view {view_id!r} has non-contiguous membership chunks"
            )
        if int(row.get("scenario_count") or len(chunk_ids)) != len(chunk_ids):
            raise CurationError(
                f"curation view {view_id!r} has an invalid membership chunk count"
            )
        scenario_ids.extend(chunk_ids)
        expected_start = end
    return tuple(scenario_ids)


def _membership_rows(
    lake: Lake,
    *,
    view_id: str | None = None,
    target_grain: str | None = None,
    target_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    if view_id:
        clauses.append(f"view_id = {_sql_literal(view_id)}")
    if target_grain:
        clauses.append(f"target_grain = {_sql_literal(target_grain)}")
    if target_ids:
        clauses.append(_sql_in_predicate("target_id", target_ids))
    where_sql = " AND ".join(clauses)
    target_id_set = {str(item) for item in target_ids}

    def matches(row: dict[str, Any]) -> bool:
        if view_id and row["view_id"] != view_id:
            return False
        if target_grain and row["target_grain"] != target_grain:
            return False
        if target_id_set and str(row["target_id"]) not in target_id_set:
            return False
        return True

    return _rows_where(
        lake,
        "curation_memberships",
        where_sql=where_sql or None,
        fallback_filter=matches,
    )


def _curation_predicate_index_role(table: str, column: str) -> dict[str, str]:
    if table == "curation_memberships":
        roles = {
            "view_id": "view-decision-scope",
            "target_grain": "target-resolution",
            "target_id": "target-resolution",
            "scenario_id": "scenario-resolution",
            "decision": "decision-filter",
            "queue": "review-queue-filter",
            "created_at": "latest-decision-order",
        }
        return {"predicate_role": roles.get(column, "curation-decision")}
    if table == _VIEW_MEMBERSHIP_CHUNK_TABLE:
        roles = {
            "view_id": "view-membership-scope",
            "chunk_index": "membership-order",
            "start_ordinal": "membership-range",
            "end_ordinal": "membership-range",
        }
        return {"predicate_role": roles.get(column, "view-membership")}
    return {"predicate_role": "curation"}


def _review_queue_predicate_index_role(column: str) -> dict[str, str]:
    roles = {
        "queue_id": "queue-identity",
        "queue_name": "queue-reopen",
        "target_grain": "target-filter",
        "target_id": "target-lookup",
        "scenario_id": "scenario-lookup",
        "source_operation": "source-filter",
        "status": "status-filter",
        "assignee": "assignee-filter",
        "priority": "page-order",
        "created_at": "latest-queue-order",
    }
    return {"predicate_role": roles.get(column, "review-queue")}


def _curation_predicate_index_params(lake: Lake) -> list[dict[str, Any]]:
    return [
        result.to_params() | _curation_predicate_index_role(result.table, result.column)
        for result in describe_curation_predicate_indexes(lake)
    ]


def _membership_target_key(row: dict[str, Any]) -> tuple[str, str]:
    target_grain = str(row.get("target_grain") or "scenario")
    target_id = str(row.get("target_id") or row.get("scenario_id") or "")
    return target_grain, target_id


def _latest_membership_by_target(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (item["created_at"], item["membership_id"])):
        key = _membership_target_key(row)
        if key[1]:
            latest[key] = row
    return latest


def _latest_membership_by_scenario(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for (target_grain, target_id), row in _latest_membership_by_target(rows).items():
        if target_grain == "scenario":
            latest[target_id] = row
    return latest


def _latest_view_row(lake: Lake, name: str | None) -> dict[str, Any]:
    if not name:
        raise CurationError("view name is required")
    rows = [
        row for row in lake.table("curation_views").to_arrow().to_pylist()
        if row["name"] == name
    ]
    if not rows:
        raise CurationError(f"no curation view named {name!r} in {lake.uri}")
    return max(rows, key=lambda row: (row["created_at"], row["view_id"]))


def _normalize_review_queue_page_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError) as exc:
        raise CurationError("review queue page limit must be an integer") from exc
    if value <= 0:
        raise CurationError("review queue page limit must be positive")
    if value > _REVIEW_QUEUE_MAX_PAGE_LIMIT:
        raise CurationError(
            f"review queue page limit {value} exceeds maximum {_REVIEW_QUEUE_MAX_PAGE_LIMIT}"
        )
    return value


def _review_queue_rows(lake: Lake, name_or_id: str) -> list[dict[str, Any]]:
    key = str(name_or_id).strip()
    if not key:
        raise CurationError("review queue name or id is required")
    queue_id = _review_queue_id_for(lake, key)
    rows = _review_queue_query_rows(
        lake,
        queue_id=queue_id,
        batch_size=_REVIEW_QUEUE_BATCH_SIZE,
    )
    if not rows:
        raise CurationError(f"no review queue named/id {key!r} in {lake.uri}")
    return rows


def _review_queue_lookup(lake: Lake, name_or_id: str) -> CurationReviewQueue:
    queue_id = _review_queue_id_for(lake, name_or_id)
    rows = _review_queue_page_rows(lake, queue_id, limit=1)
    if not rows:
        raise CurationError(f"no review queue named/id {name_or_id!r} in {lake.uri}")
    summary = _review_queue_summary(lake, queue_id)
    return _review_queue_from_metadata(lake, rows[0], item_count=int(summary["item_count"]))


def _review_queue_id_for(lake: Lake, name_or_id: str) -> str:
    key = str(name_or_id).strip()
    if not key:
        raise CurationError("review queue name or id is required")
    direct = _review_queue_find_rows(
        lake,
        clauses=[f"queue_id = {_sql_literal(key)}"],
        fallback_filter=lambda row: row["queue_id"] == key,
        limit=1,
    )
    if direct:
        return key
    rows = _review_queue_find_rows(
        lake,
        clauses=[f"queue_name = {_sql_literal(key)}"],
        fallback_filter=lambda row: row["queue_name"] == key,
        order_by_created=True,
        limit=1,
    )
    if not rows:
        raise CurationError(f"no review queue named/id {key!r} in {lake.uri}")
    return str(rows[0]["queue_id"])


def _review_queue_page_rows(
    lake: Lake,
    queue_id: str,
    *,
    limit: int,
    cursor: str | None = None,
    status: str | None = None,
    assignee: str | None = None,
    batch_size: int = _REVIEW_QUEUE_BATCH_SIZE,
) -> list[dict[str, Any]]:
    normalized_limit = _normalize_review_queue_page_limit(limit)
    return _review_queue_query_rows(
        lake,
        queue_id=queue_id,
        limit=normalized_limit,
        cursor=cursor,
        status=status,
        assignee=assignee,
        batch_size=batch_size,
    )


def _review_queue_query_rows(
    lake: Lake,
    *,
    queue_id: str,
    limit: int | None = None,
    cursor: str | None = None,
    status: str | None = None,
    assignee: str | None = None,
    columns: Sequence[str] | None = None,
    ordered: bool = True,
    batch_size: int = _REVIEW_QUEUE_BATCH_SIZE,
) -> list[dict[str, Any]]:
    cursor_values = _decode_review_queue_cursor(cursor)
    clauses = [f"queue_id = {_sql_literal(queue_id)}"]
    if status:
        clauses.append(f"status = {_sql_literal(_normalize_review_queue_status(status))}")
    if assignee:
        clauses.append(f"assignee = {_sql_literal(assignee)}")
    if cursor_values is not None:
        clauses.append(_review_queue_cursor_sql(cursor_values))

    def matches(row: dict[str, Any]) -> bool:
        if row["queue_id"] != queue_id:
            return False
        if status and row["status"] != _normalize_review_queue_status(status):
            return False
        if assignee and row["assignee"] != assignee:
            return False
        if cursor_values is not None and not _review_queue_after_cursor(row, cursor_values):
            return False
        return True

    return _review_queue_find_rows(
        lake,
        clauses=clauses,
        fallback_filter=matches,
        columns=columns,
        order_by_queue=ordered,
        limit=limit,
        batch_size=batch_size,
    )


def _review_queue_find_rows(
    lake: Lake,
    *,
    clauses: Sequence[str],
    fallback_filter: Any,
    columns: Sequence[str] | None = None,
    order_by_queue: bool = False,
    order_by_created: bool = False,
    limit: int | None = None,
    batch_size: int = _REVIEW_QUEUE_BATCH_SIZE,
) -> list[dict[str, Any]]:
    where_sql = " AND ".join(clauses)
    handle = lake.table("curation_review_queues")
    try:
        query = handle.search().where(where_sql)
        if columns:
            query = query.select(list(columns))
        if order_by_queue:
            from lancedb.query import ColumnOrdering

            query = query.order_by(
                [
                    ColumnOrdering(column_name="priority", ascending=True),
                    ColumnOrdering(column_name="target_id", ascending=True),
                    ColumnOrdering(column_name="queue_item_id", ascending=True),
                ]
            )
        elif order_by_created:
            from lancedb.query import ColumnOrdering

            query = query.order_by(
                [
                    ColumnOrdering(column_name="created_at", ascending=False),
                    ColumnOrdering(column_name="queue_id", ascending=False),
                ]
            )
        if limit is not None:
            query = query.limit(limit)
        rows: list[dict[str, Any]] = []
        for batch in query.to_batches(batch_size=batch_size):
            rows.extend(batch.to_pylist())
            if limit is not None and len(rows) >= limit:
                return rows[:limit]
        return rows
    except Exception:
        rows = [row for row in handle.to_arrow().to_pylist() if fallback_filter(row)]
        if order_by_queue:
            rows = sorted(rows, key=_review_queue_sort_key)
        elif order_by_created:
            rows = sorted(rows, key=lambda row: (row["created_at"], row["queue_id"]), reverse=True)
        if limit is not None:
            rows = rows[:limit]
        if columns:
            wanted = set(columns)
            rows = [{key: value for key, value in row.items() if key in wanted} for row in rows]
        return rows


def _review_queue_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row.get("priority") or 0),
        str(row.get("target_id") or ""),
        str(row.get("queue_item_id") or ""),
    )


def _review_queue_from_rows(
    lake: Lake,
    rows: Sequence[dict[str, Any]],
    *,
    item_count: int | None = None,
    write_status: str = "written",
) -> CurationReviewQueue:
    if not rows:
        raise CurationError("review queue has no rows")
    ordered = sorted(rows, key=_review_queue_sort_key)
    first = ordered[0]
    table_versions = tuple(
        (str(item["table"]), int(item["version"]))
        for item in (first.get("table_versions") or ())
    )
    source_transform_ids = tuple(
        dict.fromkeys(
            str(item)
            for row in ordered
            for item in (row.get("source_transform_ids") or ())
            if str(item)
        )
    )
    return CurationReviewQueue(
        lake=lake,
        queue_id=str(first["queue_id"]),
        name=str(first["queue_name"]),
        target_grain=str(first["target_grain"]),
        item_ids=tuple(str(row["queue_item_id"]) for row in ordered),
        target_ids=tuple(str(row["target_id"]) for row in ordered),
        scenario_ids=tuple(
            dict.fromkeys(str(row["scenario_id"]) for row in ordered if row["scenario_id"])
        ),
        source_operation=str(first["source_operation"]),
        transform_id=str(first["transform_id"]),
        source_transform_ids=source_transform_ids,
        table_versions=table_versions,
        item_count=item_count if item_count is not None else len(ordered),
        write_status=write_status,
    )


def _review_queue_from_metadata(
    lake: Lake,
    row: dict[str, Any],
    *,
    item_count: int,
) -> CurationReviewQueue:
    table_versions = tuple(
        (str(item["table"]), int(item["version"]))
        for item in (row.get("table_versions") or ())
    )
    source_transform_ids = tuple(
        dict.fromkeys(str(item) for item in (row.get("source_transform_ids") or ()) if str(item))
    )
    return CurationReviewQueue(
        lake=lake,
        queue_id=str(row["queue_id"]),
        name=str(row["queue_name"]),
        target_grain=str(row["target_grain"]),
        item_ids=(),
        target_ids=(),
        scenario_ids=(),
        source_operation=str(row["source_operation"]),
        transform_id=str(row["transform_id"]),
        source_transform_ids=source_transform_ids,
        table_versions=table_versions,
        item_count=item_count,
    )


def _review_queue_summary(
    lake: Lake,
    queue_id: str,
    *,
    batch_size: int = _REVIEW_QUEUE_BATCH_SIZE,
) -> dict[str, Any]:
    where_sql = f"queue_id = {_sql_literal(queue_id)}"
    count_from_table: int | None
    try:
        count_from_table = int(lake.table("curation_review_queues").count_rows(where_sql))
    except Exception:
        count_from_table = None
    rows = _review_queue_query_rows(
        lake,
        queue_id=queue_id,
        columns=("status", "assignee", "source_operation", "priority"),
        ordered=False,
        batch_size=batch_size,
    )
    by_status: Counter[str] = Counter()
    by_assignee: Counter[str] = Counter()
    by_source_operation: Counter[str] = Counter()
    by_priority_band: Counter[str] = Counter()
    for row in rows:
        by_status[str(row.get("status") or "")] += 1
        by_assignee[str(row.get("assignee") or "unassigned")] += 1
        by_source_operation[str(row.get("source_operation") or "")] += 1
        by_priority_band[_review_queue_priority_band(row.get("priority"))] += 1
    return {
        "queue_id": queue_id,
        "item_count": count_from_table if count_from_table is not None else len(rows),
        "counts_by_status": dict(sorted(by_status.items())),
        "counts_by_assignee": dict(sorted(by_assignee.items())),
        "counts_by_source_operation": dict(sorted(by_source_operation.items())),
        "counts_by_priority_band": dict(sorted(by_priority_band.items())),
        "batch_size": batch_size,
    }


def _review_queue_priority_band(value: Any) -> str:
    priority = int(value or 0)
    if priority <= 1:
        return "1"
    if priority <= 5:
        return "2-5"
    if priority <= 10:
        return "6-10"
    if priority <= 50:
        return "11-50"
    return "51+"


def _review_queue_cursor(row: dict[str, Any]) -> str:
    payload = {
        "priority": int(row.get("priority") or 0),
        "target_id": str(row.get("target_id") or ""),
        "queue_item_id": str(row.get("queue_item_id") or ""),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return encoded.decode("ascii")


def _decode_review_queue_cursor(cursor: str | None) -> dict[str, Any] | None:
    if not cursor:
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(str(cursor).encode("ascii")).decode("utf-8"))
        return {
            "priority": int(payload["priority"]),
            "target_id": str(payload["target_id"]),
            "queue_item_id": str(payload["queue_item_id"]),
        }
    except Exception as exc:
        raise CurationError("invalid review queue cursor") from exc


def _review_queue_cursor_sql(cursor: dict[str, Any]) -> str:
    priority = int(cursor["priority"])
    target_id = _sql_literal(str(cursor["target_id"]))
    queue_item_id = _sql_literal(str(cursor["queue_item_id"]))
    return (
        "("
        f"priority > {priority} OR "
        f"(priority = {priority} AND target_id > {target_id}) OR "
        f"(priority = {priority} AND target_id = {target_id} AND queue_item_id > {queue_item_id})"
        ")"
    )


def _review_queue_after_cursor(row: dict[str, Any], cursor: dict[str, Any]) -> bool:
    return _review_queue_sort_key(row) > (
        int(cursor["priority"]),
        str(cursor["target_id"]),
        str(cursor["queue_item_id"]),
    )


def _review_connector_tool_name(
    connector: ReviewToolConnector | None,
    tool: str | None,
) -> str:
    if tool:
        return _normalize_review_connector_text(tool, "tool")
    connector_tool = getattr(connector, "tool", "")
    if connector_tool:
        return _normalize_review_connector_text(connector_tool, "tool")
    return "generic"


def _normalize_review_connector_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CurationError(f"review connector {label} is required")
    return text


def _review_connector_rows(
    queue: CurationReviewQueue,
    *,
    limit: int | None,
    cursor: str | None,
    status: str | None = None,
    assignee: str | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    if limit is None and not cursor and not status and not assignee:
        rows = queue.rows()
        return rows, {
            "limit": len(rows),
            "cursor": "",
            "next_cursor": "",
            "has_more": False,
            "item_count": len(rows),
        }
    page = queue.page(
        limit=limit or _REVIEW_QUEUE_DEFAULT_PAGE_LIMIT,
        cursor=cursor,
        status=status,
        assignee=assignee,
    )
    return page.rows, {
        "limit": page.limit,
        "cursor": page.cursor,
        "next_cursor": page.next_cursor,
        "has_more": page.has_more,
        "item_count": len(page.rows),
    }


def _review_connector_task(
    row: dict[str, Any],
    *,
    tool: str,
    project_id: str,
) -> ReviewConnectorTask:
    metadata = _metadata_dict(row.get("metadata") or ())
    source_transform_ids = _review_connector_source_transform_ids(row)
    idempotency_key = _review_connector_idempotency_key(
        row,
        tool=tool,
        project_id=project_id,
        source_transform_ids=source_transform_ids,
    )
    payload = {
        "queue_id": row["queue_id"],
        "queue_name": row["queue_name"],
        "queue_item_id": row["queue_item_id"],
        "target_grain": row["target_grain"],
        "target_id": row["target_id"],
        "scenario_id": row["scenario_id"],
        "priority": row["priority"],
        "priority_score": row["priority_score"],
        "priority_reason": row["priority_reason"],
        "assignee": row["assignee"],
        "status": row["status"],
        "source_operation": row["source_operation"],
        "source_ref": _json_payload(row.get("source_ref") or ""),
        "source_transform_ids": list(source_transform_ids),
        "table_versions": _version_rows(row.get("table_versions") or ()),
        "metadata": metadata,
    }
    return ReviewConnectorTask(
        queue_id=str(row["queue_id"]),
        queue_name=str(row["queue_name"]),
        queue_item_id=str(row["queue_item_id"]),
        target_grain=str(row["target_grain"]),
        target_id=str(row["target_id"]),
        scenario_id=str(row.get("scenario_id") or ""),
        tool=tool,
        project_id=project_id,
        idempotency_key=idempotency_key,
        payload=payload,
    )


def _review_connector_idempotency_key(
    row: dict[str, Any],
    *,
    tool: str,
    project_id: str,
    source_transform_ids: Sequence[str],
) -> str:
    return "review-task-" + _digest(
        {
            "queue_id": row["queue_id"],
            "queue_item_id": row["queue_item_id"],
            "target_grain": row["target_grain"],
            "target_id": row["target_id"],
            "scenario_id": row.get("scenario_id") or "",
            "source_transform_ids": list(source_transform_ids),
            "tool": tool,
            "project_id": project_id,
        }
    )


def _review_connector_source_transform_ids(row: dict[str, Any]) -> tuple[str, ...]:
    values = [str(item) for item in (row.get("source_transform_ids") or ()) if str(item)]
    if row.get("transform_id"):
        values.append(str(row["transform_id"]))
    return tuple(dict.fromkeys(values))


def _normalize_review_connector_results(
    tasks: Sequence[ReviewConnectorTask],
    raw_results: Sequence[Any],
    *,
    operation: str,
) -> list[ReviewConnectorResult]:
    by_queue_item: dict[str, ReviewConnectorResult] = {}
    by_key: dict[str, ReviewConnectorResult] = {}
    for raw in raw_results or ():
        result = _coerce_review_connector_result(raw)
        if result.queue_item_id:
            by_queue_item[result.queue_item_id] = result
        if result.idempotency_key:
            by_key[result.idempotency_key] = result
    normalized: list[ReviewConnectorResult] = []
    for task in tasks:
        result = by_queue_item.get(task.queue_item_id) or by_key.get(task.idempotency_key)
        if result is None:
            normalized.append(
                ReviewConnectorResult(
                    queue_item_id=task.queue_item_id,
                    idempotency_key=task.idempotency_key,
                    status="failed",
                    error="connector did not return a result for this task",
                )
            )
            continue
        normalized.append(
            ReviewConnectorResult(
                queue_item_id=task.queue_item_id,
                idempotency_key=task.idempotency_key,
                status=_normalize_review_connector_status(result.status, operation=operation),
                external_task_id=result.external_task_id,
                external_url=result.external_url,
                metadata=_jsonable(result.metadata),
                error=result.error,
            )
        )
    return normalized


def _coerce_review_connector_result(raw: Any) -> ReviewConnectorResult:
    if isinstance(raw, ReviewConnectorResult):
        return raw
    if isinstance(raw, Mapping):
        return ReviewConnectorResult(
            queue_item_id=str(raw.get("queue_item_id") or ""),
            idempotency_key=str(raw.get("idempotency_key") or ""),
            status=str(raw.get("status") or "failed"),
            external_task_id=str(raw.get("external_task_id") or ""),
            external_url=str(raw.get("external_url") or ""),
            metadata=_jsonable(raw.get("metadata") or {}),
            error=str(raw.get("error") or ""),
        )
    return ReviewConnectorResult(
        queue_item_id="",
        idempotency_key="",
        status="failed",
        error=f"connector returned unsupported result type {type(raw).__name__}",
    )


def _normalize_review_connector_status(status: Any, *, operation: str) -> str:
    text = str(status or "").strip().lower().replace("_", "-")
    if operation == "status-sync":
        return _normalize_external_review_status(text)
    aliases = {
        "created": "exported",
        "upserted": "exported",
        "ok": "exported",
        "present": "already-present",
        "already-present": "already-present",
        "already-present-task": "already-present",
        "exists": "already-present",
        "error": "failed",
    }
    normalized = aliases.get(text, text)
    if normalized in _REVIEW_CONNECTOR_RESULT_STATUSES:
        return normalized
    return "failed" if normalized == "failure" else "exported"


def _normalize_external_review_status(status: Any) -> str:
    text = str(status or "").strip().lower().replace("_", "-")
    if text in _REVIEW_CONNECTOR_COMPLETED_STATUSES:
        return "completed"
    if text in _REVIEW_CONNECTOR_EXPORTED_STATUSES:
        return "exported"
    if text in {"open", "assigned", "skipped", "failed"}:
        return text
    if text in {"canceled", "cancelled", "closed", "rejected"}:
        return "skipped"
    if text in {"error", "failure"}:
        return "failed"
    return "exported" if text else "failed"


def _review_connector_result_counts(
    results: Sequence[ReviewConnectorResult],
) -> dict[str, int]:
    counts = Counter(str(result.status or "failed") for result in results)
    return dict(sorted(counts.items()))


def _review_connector_result_sort_key(
    result: ReviewConnectorResult,
    rows: Sequence[dict[str, Any]],
) -> tuple[int, str]:
    positions = {str(row["queue_item_id"]): index for index, row in enumerate(rows)}
    return positions.get(result.queue_item_id, len(rows)), result.queue_item_id


def _review_connector_task_report(
    *,
    operation: str,
    queue: CurationReviewQueue,
    tool: str,
    project_id: str,
    output_uri: str,
    page: dict[str, Any],
    tasks: Sequence[ReviewConnectorTask],
    results: Sequence[ReviewConnectorResult],
    dry_run: bool,
    plan_only: bool,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "queue_id": queue.queue_id,
        "queue_name": queue.name,
        "tool": tool,
        "project_id": project_id,
        "output_uri": output_uri,
        "target_grain": queue.target_grain,
        "dry_run": dry_run,
        "plan_only": plan_only,
        "item_count": len(tasks),
        "page": page,
        "counts_by_status": _review_connector_result_counts(results),
        "idempotency_keys": [task.idempotency_key for task in tasks],
        "tasks": [task.to_dict() for task in tasks],
        "results": [result.to_dict() for result in results],
    }


def _apply_review_connector_export_results(
    lake: Lake,
    *,
    queue_id: str,
    results: Sequence[ReviewConnectorResult],
    tool: str,
    project_id: str,
    output_uri: str,
    transform_id: str,
) -> None:
    by_item = {result.queue_item_id: result for result in results}
    rows = _review_queue_query_rows(lake, queue_id=queue_id, ordered=False)
    updated: list[dict[str, Any]] = []
    for row in rows:
        result = by_item.get(str(row["queue_item_id"]))
        if result is None:
            updated.append(row)
            continue
        next_row = dict(row)
        if result.external_task_id:
            next_row["external_task_id"] = result.external_task_id
        if result.external_url:
            next_row["external_url"] = result.external_url
        if output_uri:
            next_row["export_uri"] = output_uri
        next_row["status"] = _review_queue_status_after_connector_export(row, result)
        next_row["metadata"] = _metadata_items(
            _review_connector_metadata(
                row,
                tool=tool,
                project_id=project_id,
                transform_id=transform_id,
                result=result,
                operation="export",
            )
        )
        updated.append(next_row)
    _replace_review_queue_rows(lake, queue_id=queue_id, rows=updated)


def _apply_review_connector_status_results(
    lake: Lake,
    *,
    queue_id: str,
    results: Sequence[ReviewConnectorResult],
    tool: str,
    project_id: str,
    transform_id: str,
) -> None:
    by_item = {result.queue_item_id: result for result in results}
    rows = _review_queue_query_rows(lake, queue_id=queue_id, ordered=False)
    updated: list[dict[str, Any]] = []
    for row in rows:
        result = by_item.get(str(row["queue_item_id"]))
        if result is None:
            updated.append(row)
            continue
        next_row = dict(row)
        if result.external_task_id:
            next_row["external_task_id"] = result.external_task_id
        if result.external_url:
            next_row["external_url"] = result.external_url
        next_row["status"] = _review_queue_status_after_connector_sync(row, result)
        next_row["metadata"] = _metadata_items(
            _review_connector_metadata(
                row,
                tool=tool,
                project_id=project_id,
                transform_id=transform_id,
                result=result,
                operation="status-sync",
            )
        )
        updated.append(next_row)
    _replace_review_queue_rows(lake, queue_id=queue_id, rows=updated)


def _review_queue_status_after_connector_export(
    row: dict[str, Any],
    result: ReviewConnectorResult,
) -> str:
    current = str(row.get("status") or "open")
    if result.status in {"exported", "already-present"}:
        return "completed" if current == "completed" else "exported"
    if result.status == "skipped":
        return "skipped"
    return current


def _review_queue_status_after_connector_sync(
    row: dict[str, Any],
    result: ReviewConnectorResult,
) -> str:
    current = str(row.get("status") or "open")
    if result.status in _REVIEW_QUEUE_STATUSES:
        return result.status
    return current


def _review_connector_metadata(
    row: dict[str, Any],
    *,
    tool: str,
    project_id: str,
    transform_id: str,
    result: ReviewConnectorResult,
    operation: str,
) -> dict[str, Any]:
    metadata = _metadata_dict(row.get("metadata") or ())
    connectors = metadata.get("review_connectors")
    if not isinstance(connectors, dict):
        connectors = {}
    key = f"{tool}:{project_id}"
    connectors[key] = {
        "tool": tool,
        "project_id": project_id,
        "operation": operation,
        "status": result.status,
        "idempotency_key": result.idempotency_key,
        "external_task_id": result.external_task_id,
        "external_url": result.external_url,
        "error": result.error,
        "metadata": _jsonable(result.metadata),
        "transform_id": transform_id,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    metadata["review_connectors"] = connectors
    return metadata


def _replace_review_queue_rows(
    lake: Lake,
    *,
    queue_id: str,
    rows: Sequence[dict[str, Any]],
) -> None:
    table = lake.table("curation_review_queues")
    table.delete(f"queue_id = {_sql_literal(queue_id)}")
    if rows:
        table.add(pa.Table.from_pylist(list(rows), schema=CURATION_REVIEW_QUEUES_SCHEMA))


def _json_payload(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return _jsonable(value)
    try:
        return _jsonable(json.loads(value))
    except json.JSONDecodeError:
        return value


def _review_connector_outcome_row(row: Mapping[str, Any], *, tool: str) -> dict[str, Any]:
    payload = dict(row)
    source = str(payload.get("source") or "")
    if not source:
        payload["source"] = "human"
        return payload
    if source in _DECISION_SOURCES:
        return payload
    metadata = _metadata_dict(payload.get("metadata") or {})
    metadata.setdefault("connector_source", source)
    metadata.setdefault("connector_tool", tool)
    payload["metadata"] = metadata
    payload["source"] = "human"
    return payload


def _iter_outcome_rows(
    outcomes: Iterable[dict[str, Any]] | dict[str, Any],
) -> Iterable[tuple[int, dict[str, Any]]]:
    if isinstance(outcomes, dict):
        if "outcomes" in outcomes and isinstance(outcomes["outcomes"], list):
            rows: Iterable[Any] = outcomes["outcomes"]
        else:
            rows = [outcomes]
    elif isinstance(outcomes, (str, bytes)):
        raise CurationError("review outcomes must be dict rows, not a string")
    else:
        rows = outcomes
    seen = False
    for index, row in enumerate(rows):
        seen = True
        if not isinstance(row, dict):
            raise CurationError(
                f"review outcome row {index} must be a JSON object; "
                "retry after removing or correcting that row"
            )
        yield index, dict(row)
    if not seen:
        raise CurationError("no review outcomes supplied")


def _outcome_rows(outcomes: Iterable[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    return [row for _, row in _iter_outcome_rows(outcomes)]


def _queue_row_for_outcome_lookup(
    lake: Lake,
    queue_id: str,
    outcome: dict[str, Any],
    *,
    row_index: int,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    attempted: list[str] = []
    for key in ("queue_item_id", "target_id", "scenario_id", "observation_id"):
        value = outcome.get(key)
        if not value:
            continue
        text = str(value)
        attempted.append(f"{key}={text!r}")
        cache_key = (key, text)
        if cache_key in cache:
            return cache[cache_key]
        column = "target_id" if key == "observation_id" else key
        rows = _review_queue_find_rows(
            lake,
            clauses=[
                f"queue_id = {_sql_literal(queue_id)}",
                f"{column} = {_sql_literal(text)}",
            ],
            fallback_filter=lambda row, column=column, text=text: (
                row["queue_id"] == queue_id and str(row.get(column) or "") == text
            ),
            order_by_queue=True,
            limit=1,
        )
        if rows:
            cache[cache_key] = rows[0]
            return rows[0]
    if attempted:
        raise CurationError(
            f"review outcome row {row_index} target {'; '.join(attempted)} "
            f"is not in queue {queue_id!r}; retry after correcting that row"
        )
    raise CurationError(
        f"review outcome row {row_index} does not reference a target in queue "
        f"{queue_id!r}; include queue_item_id, target_id, scenario_id, or observation_id"
    )


def _queue_row_for_outcome(
    outcome: dict[str, Any],
    queue_by_target: dict[str, dict[str, Any]],
    queue_by_scenario: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for key in ("queue_item_id", "target_id", "scenario_id", "observation_id"):
        value = outcome.get(key)
        if not value:
            continue
        text = str(value)
        if key == "queue_item_id":
            match = next(
                (
                    row
                    for row in queue_by_target.values()
                    if str(row["queue_item_id"]) == text
                ),
                None,
            )
            if match:
                return match
        if text in queue_by_target:
            return queue_by_target[text]
        if text in queue_by_scenario:
            return queue_by_scenario[text]
    raise CurationError("review outcome does not reference a target in the queue")


def _writeback_row_from_outcome(
    outcome: dict[str, Any],
    queue_row: dict[str, Any],
    *,
    target_grain: str,
    target_id: str,
    scenario_id: str,
    metadata: dict[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    normalized_grain = _normalize_target_grain(target_grain)
    row: dict[str, Any] = {
        key: outcome[key]
        for key in fields
        if key in outcome and outcome[key] is not None
    }
    if normalized_grain == "observation":
        row["observation_id"] = target_id
        if scenario_id:
            row["scenario_id"] = scenario_id
    else:
        row["scenario_id"] = scenario_id or target_id
    row["source"] = str(outcome.get("source") or "review-queue")
    if outcome.get("reviewer") or outcome.get("assignee") or queue_row.get("assignee"):
        row["reviewer"] = str(
            outcome.get("reviewer") or outcome.get("assignee") or queue_row.get("assignee") or ""
        )
    if outcome.get("confidence") is not None:
        row["confidence"] = outcome["confidence"]
    elif outcome.get("score") is not None:
        row["confidence"] = outcome["score"]
    row["metadata"] = metadata
    return row


def _view_from_row(row: dict[str, Any], *, lake: Lake | None = None) -> CurationView:
    storage = _view_membership_storage_from_row(row)
    inline_ids = tuple(str(item) for item in row["scenario_ids"] or ())
    scenario_ids = inline_ids
    if storage["kind"] == _VIEW_STORAGE_CHUNKED:
        if lake is None:
            scenario_ids = ()
        else:
            scenario_ids = _scenario_ids_from_view_chunks(lake, str(row["view_id"]))
        expected = int(storage.get("scenario_count") or 0)
        if expected and len(scenario_ids) != expected:
            raise CurationError(
                f"curation view {row['name']!r} expected {expected} chunked scenarios, "
                f"found {len(scenario_ids)}"
            )
    return CurationView(
        view_id=str(row["view_id"]),
        name=str(row["name"]),
        scenario_ids=scenario_ids,
        table_versions=tuple(
            (str(item["table"]), int(item["version"]))
            for item in row["table_versions"] or ()
        ),
        transform_id=str(row["transform_id"] or ""),
        owner=str(row.get("owner") or row.get("created_by") or ""),
        tags=tuple(str(item) for item in row.get("tags") or ()),
        description=str(row.get("description") or ""),
        status=str(row.get("status") or "active"),
        membership_storage=str(storage["kind"]),
        membership_count=len(scenario_ids),
    )


def _selection_from_view(
    lake: Lake,
    view: CurationView,
    *,
    row: dict[str, Any] | None = None,
) -> CurationSelection:
    if row is None:
        row = next(
            (
                item
                for item in lake.table("curation_views").to_arrow().to_pylist()
                if item["view_id"] == view.view_id
            ),
            None,
        )
    if row:
        query_spec = json.loads(row["query_spec"] or "{}")
        scope = json.loads(row["scope"] or "{}")
        membership_storage = _view_membership_storage_from_row(row)
    else:
        query_spec = {"source": {}, "scenario_ids": list(view.scenario_ids)}
        scope = {}
        membership_storage = _view_membership_storage_payload(
            view.scenario_ids,
            inline_scenario_limit=_VIEW_INLINE_SCENARIO_ID_LIMIT,
            chunk_size=_VIEW_MEMBERSHIP_CHUNK_SIZE,
        )
    source = query_spec.get("source") or {}
    transform_ids = tuple(str(item) for item in source.get("operation_transform_ids") or ())
    if view.transform_id:
        transform_ids = transform_ids + (view.transform_id,)
    scenario_ids = tuple(str(item) for item in view.scenario_ids)
    if not scenario_ids:
        raise CurationError(f"saved curation view {view.name!r} does not contain scenarios")
    return CurationSelection(
        lake=lake,
        scenario_ids=scenario_ids,
        scope=scope,
        operation="saved-view",
        transform_id=view.transform_id,
        report={
            "operation": "saved-view",
            "view_id": view.view_id,
            "view_name": view.name,
            "owner": view.owner,
            "tags": list(view.tags),
            "table_versions": [
                {"table": table, "version": version, "tag": ""}
                for table, version in view.table_versions
            ],
            "membership_storage": membership_storage,
            "predicate_indexes": _curation_predicate_index_params(lake),
            "input_count": len(scenario_ids),
            "output_count": len(scenario_ids),
        },
        operation_transform_ids=transform_ids,
    )


def _compile_curation_row_plan(
    selection: CurationSelection,
    *,
    view_name: str | None,
    target_grain: str,
    source_snapshot_name: str | None,
    include_decisions: Sequence[str],
    excluding_decisions: Sequence[str],
    freeze: bool,
    created_by: str,
) -> CurationCompiledRowPlan:
    normalized_grain = _normalize_row_plan_target_grain(target_grain)
    normalized_includes = tuple(_normalize_decision(decision) for decision in include_decisions)
    normalized_excludes = tuple(_normalize_decision(decision) for decision in excluding_decisions)
    view = _row_plan_view(selection, view_name=view_name, created_by=created_by)
    candidate_rows = _row_plan_candidates(
        selection.lake,
        target_grain=normalized_grain,
        scenario_ids=selection.scenario_ids,
        source_snapshot_name=source_snapshot_name,
    )
    if not candidate_rows:
        raise CurationError(
            f"saved view {view.name!r} has no {normalized_grain} rows to compile"
        )
    membership_rows = _membership_rows(selection.lake, view_id=view.view_id)
    latest_by_target = _latest_membership_by_target(membership_rows)
    superseded_ids = {
        str(row["membership_id"])
        for row in membership_rows
        if row.get("membership_id")
    } - {
        str(row["membership_id"])
        for row in latest_by_target.values()
        if row.get("membership_id")
    }
    membership_transform_ids = tuple(
        dict.fromkeys(
            str(row["transform_id"])
            for row in membership_rows
            if row.get("transform_id")
        )
    )
    selected, rejected, conflicts, label_intents = _resolve_row_plan_membership(
        candidate_rows,
        latest_by_target=latest_by_target,
        target_grain=normalized_grain,
        include_decisions=normalized_includes,
        excluding_decisions=normalized_excludes,
    )
    if not selected:
        raise CurationError(
            f"membership decisions removed every {normalized_grain} row from the plan"
        )
    selected_target_ids = tuple(str(item["target_id"]) for item in selected)
    selected_scenario_ids = tuple(
        dict.fromkeys(
            scenario_id
            for item in selected
            for scenario_id in item.get("scenario_ids", ())
            if scenario_id
        )
    )
    table_versions = _row_plan_table_versions(selection.lake, normalized_grain)
    supersession_chains = [
        {
            "target_grain": str(row.get("target_grain") or ""),
            "target_id": str(row.get("target_id") or ""),
            "latest_membership_id": str(row.get("membership_id") or ""),
            "chain": _supersession_chain(membership_rows, row),
        }
        for row in sorted(latest_by_target.values(), key=_membership_sort_key)
        if row.get("supersedes_membership_id")
    ]
    base_policy = (
        "row-include-decisions"
        if any(
            str(row.get("decision") or "") in normalized_includes
            for (grain, _), row in latest_by_target.items()
            if grain == normalized_grain
        )
        else "view-candidate-rows"
    )
    read_versions = [
        {"table": table, "version": version, "tag": ""}
        for table, version in table_versions
    ]
    plan_payload = {
        "schema_version": "lancedb-robotics/curation-row-plan/v1",
        "lake_uri": selection.lake.uri,
        "view_id": view.view_id,
        "target_grain": normalized_grain,
        "source_snapshot_name": source_snapshot_name or "",
        "base_policy": base_policy,
        "table_versions": read_versions,
        "target_ids": list(selected_target_ids),
        "membership_transform_ids": list(membership_transform_ids),
        "superseded_membership_ids": sorted(superseded_ids),
    }
    plan_id = "curation-rowplan-" + _digest(plan_payload)
    artifact_id = f"lancedb-robotics:curation-row-plan:{plan_id}" if freeze else ""
    selected_rows = [
        {
            "target_id": str(item["target_id"]),
            "scenario_id": str(item.get("scenario_id") or ""),
            "scenario_ids": list(item.get("scenario_ids") or ()),
            "lance_row_id": item.get("row_id"),
            "table": _row_plan_target_table(normalized_grain),
            "decision": item.get("decision") or "",
            "scenario_decision": item.get("scenario_decision") or "",
            "resolution": item.get("resolution") or "included",
        }
        for item in selected
    ]
    report = {
        **plan_payload,
        "operation": "compile-row-plan",
        "plan_id": plan_id,
        "artifact_id": artifact_id,
        "frozen": bool(freeze),
        "view": _view_report(view),
        "target_table": _row_plan_target_table(normalized_grain),
        "source_snapshot_name": source_snapshot_name or "",
        "include_decisions": list(normalized_includes),
        "excluding_decisions": list(normalized_excludes),
        "conflict_policy": "scenario-exclude-unless-later-row-include",
        "candidate_count": len(candidate_rows),
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "selected_rows": selected_rows,
        "selected_target_ids": list(selected_target_ids),
        "selected_scenario_ids": list(selected_scenario_ids),
        "lance_row_ids": [item.get("row_id") for item in selected],
        "rejected": rejected,
        "conflicts": conflicts,
        "label_intents": label_intents,
        "latest_membership_ids": [
            str(row.get("membership_id") or "")
            for row in sorted(latest_by_target.values(), key=_membership_sort_key)
            if row.get("membership_id")
        ],
        "superseded_membership_ids": sorted(superseded_ids),
        "supersession_chains": supersession_chains,
        "membership_transform_ids": list(membership_transform_ids),
        "source_table_versions": read_versions,
        "source_view_transform_id": view.transform_id,
        "source_snapshot_transform_id": _source_snapshot_transform_id(
            selection.lake,
            source_snapshot_name,
        ),
        "payload_copy_policy": "logical-reference",
        "copied_payload_bytes": 0,
        "metadata_only": True,
        "input_count": len(candidate_rows),
        "output_count": len(selected),
    }
    prior_transform_ids = tuple(
        dict.fromkeys(
            (
                *selection.operation_transform_ids,
                view.transform_id,
                *membership_transform_ids,
                _source_snapshot_transform_id(selection.lake, source_snapshot_name),
            )
        )
    )
    transform_id = _record_curation_transform(
        selection.lake,
        operation="compile-row-plan",
        input_scenario_ids=selection.scenario_ids,
        output_scenario_ids=selected_scenario_ids,
        report=report,
        prior_transform_ids=tuple(item for item in prior_transform_ids if item),
        created_by=created_by,
        output_tables=("lineage_artifacts",) if freeze else (),
    )
    report = {**report, "transform_id": transform_id}
    if freeze:
        selection.lake.lineage.record_artifact(
            kind="curation-row-plan",
            artifact_id=artifact_id,
            name=plan_id,
            table_name=_row_plan_target_table(normalized_grain),
            table_version=_version_for_table(table_versions, _row_plan_target_table(normalized_grain)),
            row_grain=normalized_grain,
            row_ids=selected_target_ids,
            source_uri=f"{selection.lake.uri}#curation-row-plan/{plan_id}",
            source_id=source_snapshot_name or view.view_id,
            digest=_digest({**plan_payload, "plan_id": plan_id}),
            producer_execution_id=transform_id,
            metadata={
                "view_id": view.view_id,
                "view_name": view.name,
                "target_grain": normalized_grain,
                "source_snapshot_name": source_snapshot_name or "",
                "membership_transform_ids": list(membership_transform_ids),
                "selected_count": len(selected),
                "copied_payload_bytes": 0,
                "payload_copy_policy": "logical-reference",
            },
        )
    return CurationCompiledRowPlan(
        plan_id=plan_id,
        target_grain=normalized_grain,
        view=view,
        target_ids=selected_target_ids,
        scenario_ids=selected_scenario_ids,
        lance_row_ids=tuple(item.get("row_id") for item in selected),
        table_versions=table_versions,
        membership_transform_ids=membership_transform_ids,
        transform_id=transform_id,
        report=report,
        artifact_id=artifact_id,
        frozen=bool(freeze),
    )


def _row_plan_view(
    selection: CurationSelection,
    *,
    view_name: str | None,
    created_by: str,
) -> CurationView:
    if view_name:
        return _view_from_row(_latest_view_row(selection.lake, view_name), lake=selection.lake)
    view_id = str(selection.report.get("view_id") or "")
    if view_id:
        return _view_from_row(_latest_view_row_by_id(selection.lake, view_id), lake=selection.lake)
    generated = "row-plan-" + _digest({"scenario_ids": list(selection.scenario_ids)})
    return selection.save_view(generated, created_by=created_by)


def _latest_view_row_by_id(lake: Lake, view_id: str) -> dict[str, Any]:
    rows = [
        row
        for row in lake.table("curation_views").to_arrow().to_pylist()
        if str(row["view_id"]) == view_id
    ]
    if not rows:
        raise CurationError(f"no curation view id {view_id!r} in {lake.uri}")
    return max(rows, key=lambda row: (row["created_at"], row["view_id"]))


def _normalize_row_plan_target_grain(target_grain: str) -> str:
    normalized = _normalize_target_grain(target_grain)
    if normalized == "aligned-tick":
        normalized = "aligned-frame"
    if normalized not in _ROW_PLAN_TARGET_GRAINS:
        raise CurationError(
            f"row plans support {', '.join(_ROW_PLAN_TARGET_GRAINS)}; got {target_grain!r}"
        )
    return normalized


def _row_plan_target_table(target_grain: str) -> str:
    if target_grain == "episode":
        return "episodes"
    if target_grain == "observation":
        return "observations"
    if target_grain == "aligned-frame":
        return "aligned_frames"
    return "dataset_snapshots"


def _row_plan_table_versions(lake: Lake, target_grain: str) -> tuple[tuple[str, int], ...]:
    tables = tuple(
        dict.fromkeys(
            (
                "scenarios",
                "curation_views",
                _VIEW_MEMBERSHIP_CHUNK_TABLE,
                "curation_memberships",
                "dataset_snapshots",
                _row_plan_target_table(target_grain),
            )
        )
    )
    return tuple((row["table"], int(row["version"])) for row in _table_versions(lake, tables))


def _version_for_table(table_versions: Sequence[tuple[str, int]], table: str) -> int:
    for name, version in table_versions:
        if name == table:
            return int(version)
    return 0


def _source_snapshot_transform_id(lake: Lake, snapshot_name: str | None) -> str:
    if not snapshot_name:
        return ""
    return str(_latest_snapshot_row(lake, snapshot_name).get("transform_id") or "")


def _row_plan_candidates(
    lake: Lake,
    *,
    target_grain: str,
    scenario_ids: Sequence[str],
    source_snapshot_name: str | None,
) -> list[dict[str, Any]]:
    if target_grain == "observation":
        return _observation_row_plan_candidates(lake, scenario_ids)
    if target_grain == "episode":
        return _episode_row_plan_candidates(lake, scenario_ids)
    if target_grain == "aligned-frame":
        return _aligned_frame_row_plan_candidates(lake, scenario_ids)
    return _snapshot_row_plan_candidates(
        lake,
        scenario_ids,
        source_snapshot_name=source_snapshot_name,
    )


def _observation_row_plan_candidates(
    lake: Lake,
    scenario_ids: Sequence[str],
) -> list[dict[str, Any]]:
    scenario_order = {scenario_id: index for index, scenario_id in enumerate(scenario_ids)}
    scenario_rows = {row["scenario_id"]: row for row in _selected_rows(lake, scenario_ids)}
    observation_to_scenario: dict[str, str] = {}
    observation_order: dict[str, tuple[int, int, str]] = {}
    for scenario_id in scenario_ids:
        scenario = scenario_rows.get(scenario_id)
        if not scenario:
            continue
        for ordinal, observation_id in enumerate(scenario.get("observation_ids") or ()):
            obs_id = str(observation_id)
            observation_to_scenario.setdefault(obs_id, scenario_id)
            observation_order.setdefault(
                obs_id,
                (scenario_order.get(scenario_id, 10**9), ordinal, obs_id),
            )
    candidates: list[dict[str, Any]] = []
    rows = _table_rows_with_row_id(lake, "observations")
    rows_by_id = {str(row["observation_id"]): row for row in rows}
    for observation_id, scenario_id in observation_to_scenario.items():
        row = rows_by_id.get(observation_id)
        if row is None:
            continue
        candidates.append(
            _row_plan_candidate(
                "observation",
                "observations",
                row,
                target_id=observation_id,
                scenario_ids=(scenario_id,),
                sort_key=observation_order[observation_id],
            )
        )
    if observation_to_scenario:
        return sorted(candidates, key=lambda item: item["sort_key"])
    known = set(observation_to_scenario)
    scenario_set = set(scenario_ids)
    scenario_windows = list(scenario_rows.values())
    for row in rows:
        observation_id = str(row.get("observation_id") or "")
        if not observation_id or observation_id in known:
            continue
        timestamp_ns = int(row.get("timestamp_ns") or row.get("raw_log_time_ns") or 0)
        scenario_id = _scenario_for_window(
            scenario_windows,
            run_id=str(row.get("run_id") or ""),
            start_time_ns=timestamp_ns,
            end_time_ns=timestamp_ns,
        )
        if scenario_id not in scenario_set:
            continue
        candidates.append(
            _row_plan_candidate(
                "observation",
                "observations",
                row,
                target_id=observation_id,
                scenario_ids=(scenario_id,),
                sort_key=(
                    scenario_order.get(scenario_id, 10**9),
                    int(row.get("timestamp_ns") or 0),
                    observation_id,
                ),
            )
        )
    return sorted(candidates, key=lambda item: item["sort_key"])


def _episode_row_plan_candidates(
    lake: Lake,
    scenario_ids: Sequence[str],
) -> list[dict[str, Any]]:
    scenario_order = {scenario_id: index for index, scenario_id in enumerate(scenario_ids)}
    scenario_rows = _selected_rows(lake, scenario_ids)
    scenario_set = set(scenario_ids)
    candidates = []
    for row in _table_rows_with_row_id(lake, "episodes"):
        episode_id = str(row.get("episode_id") or "")
        scenario_id = _scenario_for_window(
            scenario_rows,
            run_id=str(row.get("run_id") or ""),
            start_time_ns=int(row.get("from_timestamp_ns") or 0),
            end_time_ns=int(row.get("to_timestamp_ns") or 0),
        )
        if not episode_id or scenario_id not in scenario_set:
            continue
        candidates.append(
            _row_plan_candidate(
                "episode",
                "episodes",
                row,
                target_id=episode_id,
                scenario_ids=(scenario_id,),
                sort_key=(
                    scenario_order.get(scenario_id, 10**9),
                    int(row.get("episode_index") or 0),
                    episode_id,
                ),
            )
        )
    return sorted(candidates, key=lambda item: item["sort_key"])


def _aligned_frame_row_plan_candidates(
    lake: Lake,
    scenario_ids: Sequence[str],
) -> list[dict[str, Any]]:
    scenario_order = {scenario_id: index for index, scenario_id in enumerate(scenario_ids)}
    scenario_rows = _selected_rows(lake, scenario_ids)
    scenario_set = set(scenario_ids)
    candidates = []
    for row in _table_rows_with_row_id(lake, "aligned_frames"):
        aligned_frame_id = str(row.get("aligned_frame_id") or "")
        timestamp_ns = int(row.get("timestamp_ns") or row.get("source_time_ns") or 0)
        scenario_id = _scenario_for_window(
            scenario_rows,
            run_id=str(row.get("run_id") or ""),
            start_time_ns=timestamp_ns,
            end_time_ns=timestamp_ns,
        )
        if not aligned_frame_id or scenario_id not in scenario_set:
            continue
        candidates.append(
            _row_plan_candidate(
                "aligned-frame",
                "aligned_frames",
                row,
                target_id=aligned_frame_id,
                scenario_ids=(scenario_id,),
                sort_key=(
                    scenario_order.get(scenario_id, 10**9),
                    int(row.get("tick_index") or 0),
                    str(row.get("stream") or ""),
                    aligned_frame_id,
                ),
            )
        )
    return sorted(candidates, key=lambda item: item["sort_key"])


def _snapshot_row_plan_candidates(
    lake: Lake,
    scenario_ids: Sequence[str],
    *,
    source_snapshot_name: str | None,
) -> list[dict[str, Any]]:
    scenario_set = set(scenario_ids)
    candidates = []
    for row in _table_rows_with_row_id(lake, "dataset_snapshots"):
        if source_snapshot_name and str(row.get("name") or "") != source_snapshot_name:
            continue
        snapshot_scenario_ids = _scenario_ids_from_snapshot_row(row)
        if snapshot_scenario_ids and not set(snapshot_scenario_ids) <= scenario_set:
            continue
        if not snapshot_scenario_ids and not source_snapshot_name:
            continue
        dataset_id = str(row.get("dataset_id") or "")
        if not dataset_id:
            continue
        candidates.append(
            _row_plan_candidate(
                "snapshot-row",
                "dataset_snapshots",
                row,
                target_id=dataset_id,
                scenario_ids=snapshot_scenario_ids,
                sort_key=(str(row.get("name") or ""), dataset_id),
            )
        )
    return sorted(candidates, key=lambda item: item["sort_key"])


def _row_plan_candidate(
    target_grain: str,
    table: str,
    row: dict[str, Any],
    *,
    target_id: str,
    scenario_ids: Sequence[str],
    sort_key: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        "target_grain": target_grain,
        "table": table,
        "target_id": str(target_id),
        "scenario_id": str(next((item for item in scenario_ids if item), "")),
        "scenario_ids": tuple(str(item) for item in scenario_ids if str(item)),
        "row_id": row.get(_ROW_ID_COLUMN),
        "row": row,
        "sort_key": sort_key,
    }


def _table_rows_with_row_id(lake: Lake, table_name: str) -> list[dict[str, Any]]:
    query = lake.table(table_name).search()
    rows: list[dict[str, Any]] = []
    for batch in query.with_row_id(True).to_batches(batch_size=4096):
        for row in batch.to_pylist():
            row_id = row.get(_ROW_ID_COLUMN)
            rows.append(
                {
                    **row,
                    _ROW_ID_COLUMN: int(row_id) if row_id is not None else None,
                }
            )
    return rows


def _resolve_row_plan_membership(
    candidates: Sequence[dict[str, Any]],
    *,
    latest_by_target: Mapping[tuple[str, str], dict[str, Any]],
    target_grain: str,
    include_decisions: Sequence[str],
    excluding_decisions: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_latest = {
        target_id: row
        for (grain, target_id), row in latest_by_target.items()
        if grain == "scenario"
    }
    row_latest = {
        target_id: row
        for (grain, target_id), row in latest_by_target.items()
        if grain == target_grain
    }
    include_mode = any(
        str(row.get("decision") or "") in include_decisions
        for row in row_latest.values()
    )
    selected: list[dict[str, Any]] = []
    rejected: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    label_intents: list[dict[str, Any]] = []
    for candidate in candidates:
        target_id = str(candidate["target_id"])
        scenario_id = str(candidate.get("scenario_id") or "")
        row_decision = row_latest.get(target_id)
        scenario_decision = scenario_latest.get(scenario_id) if scenario_id else None
        include, reason = _initial_row_plan_membership(include_mode)
        scenario_value = str((scenario_decision or {}).get("decision") or "")
        row_value = str((row_decision or {}).get("decision") or "")
        if scenario_value in excluding_decisions:
            include = False
            reason = f"scenario-{scenario_value}"
        if row_value in _ROW_PLAN_INTENT_DECISIONS:
            label_intents.append(_row_plan_decision_summary(row_decision, candidate))
        elif row_value in excluding_decisions:
            include = False
            reason = f"row-{row_value}"
            if scenario_value in excluding_decisions:
                conflicts.append(
                    _row_plan_conflict(
                        candidate,
                        scenario_decision=scenario_decision,
                        row_decision=row_decision,
                        resolution=reason,
                    )
                )
        elif row_value in include_decisions:
            if scenario_value in excluding_decisions and not _decision_is_later_or_equal(
                row_decision,
                scenario_decision,
            ):
                include = False
                reason = f"scenario-{scenario_value}-after-row-{row_value}"
            else:
                include = True
                reason = f"row-{row_value}"
            if scenario_value in excluding_decisions:
                conflicts.append(
                    _row_plan_conflict(
                        candidate,
                        scenario_decision=scenario_decision,
                        row_decision=row_decision,
                        resolution=reason,
                    )
                )
        if include:
            selected.append(
                {
                    **candidate,
                    "decision": row_value,
                    "scenario_decision": scenario_value,
                    "resolution": reason,
                }
            )
        else:
            rejected[target_id] = {
                "target_grain": target_grain,
                "scenario_id": scenario_id,
                "scenario_ids": list(candidate.get("scenario_ids") or ()),
                "decision": row_value,
                "scenario_decision": scenario_value,
                "reason": reason,
                "row_membership_id": str((row_decision or {}).get("membership_id") or ""),
                "scenario_membership_id": str(
                    (scenario_decision or {}).get("membership_id") or ""
                ),
            }
    return selected, rejected, conflicts, label_intents


def _initial_row_plan_membership(include_mode: bool) -> tuple[bool, str]:
    if include_mode:
        return False, "not-row-included"
    return True, "view-candidate"


def _decision_is_later_or_equal(
    row_decision: dict[str, Any] | None,
    scenario_decision: dict[str, Any] | None,
) -> bool:
    if not row_decision:
        return False
    if not scenario_decision:
        return True
    return _decision_order(row_decision) >= _decision_order(scenario_decision)


def _decision_order(row: Mapping[str, Any]) -> tuple[Any, str]:
    return row.get("created_at") or datetime.min.replace(tzinfo=UTC), str(
        row.get("membership_id") or ""
    )


def _row_plan_conflict(
    candidate: Mapping[str, Any],
    *,
    scenario_decision: Mapping[str, Any] | None,
    row_decision: Mapping[str, Any] | None,
    resolution: str,
) -> dict[str, Any]:
    return {
        "target_grain": str(candidate.get("target_grain") or ""),
        "target_id": str(candidate.get("target_id") or ""),
        "scenario_id": str(candidate.get("scenario_id") or ""),
        "scenario_membership_id": str((scenario_decision or {}).get("membership_id") or ""),
        "scenario_decision": str((scenario_decision or {}).get("decision") or ""),
        "row_membership_id": str((row_decision or {}).get("membership_id") or ""),
        "row_decision": str((row_decision or {}).get("decision") or ""),
        "resolution": resolution,
    }


def _row_plan_decision_summary(
    decision: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "membership_id": str((decision or {}).get("membership_id") or ""),
        "target_grain": str(candidate.get("target_grain") or ""),
        "target_id": str(candidate.get("target_id") or ""),
        "scenario_id": str(candidate.get("scenario_id") or ""),
        "decision": str((decision or {}).get("decision") or ""),
        "reason_code": str((decision or {}).get("reason_code") or ""),
        "queue": str((decision or {}).get("queue") or ""),
        "reviewer": str((decision or {}).get("reviewer") or ""),
    }


def _latest_snapshot_row(lake: Lake, name: str) -> dict[str, Any]:
    rows = [
        row for row in lake.table("dataset_snapshots").to_arrow().to_pylist()
        if row["name"] == name
    ]
    if not rows:
        raise CurationError(f"no dataset snapshot named {name!r} in {lake.uri}")
    return max(rows, key=lambda row: (row["created_at"], row["dataset_id"]))


def _scenario_ids_from_snapshot_row(row: dict[str, Any]) -> tuple[str, ...]:
    query_spec = json.loads(row["query_spec"] or "{}")
    return tuple(str(item) for item in query_spec.get("scenario_ids") or ())


def _snapshot_scenario_ids(lake: Lake, name: str) -> tuple[str, ...]:
    return _scenario_ids_from_snapshot_row(_latest_snapshot_row(lake, name))


@dataclass(frozen=True)
class _ReplayContext:
    as_of: datetime | None
    as_of_transform_id: str
    snapshot_row: dict[str, Any] | None
    table_versions: dict[str, int]
    validation: dict[str, Any]


def _resolve_membership(
    lake: Lake,
    *,
    view_name: str | None,
    view_id: str | None,
    target_grain: str | None,
    target_ids: Sequence[str],
    scenario_ids: Sequence[str],
    as_of: datetime | str | None,
    transform_id: str | None,
    snapshot_name: str | None,
    superseded_policy: str,
) -> CurationDecisionResolution:
    policy = _normalize_superseded_policy(superseded_policy)
    context = _replay_context(
        lake,
        as_of=as_of,
        transform_id=transform_id,
        snapshot_name=snapshot_name,
    )
    normalized_grain = _normalize_target_grain(target_grain) if target_grain else ""
    normalized_targets = tuple(dict.fromkeys(str(item) for item in target_ids if str(item)))
    normalized_scenarios = tuple(
        dict.fromkeys(str(item) for item in scenario_ids if str(item))
    )
    view_version = context.table_versions.get("curation_views")
    membership_version = context.table_versions.get("curation_memberships")
    view_row = _view_row_for_replay(
        lake,
        view_name=view_name,
        view_id=view_id,
        as_of=context.as_of,
        table_version=view_version,
    )
    resolved_view_id = str(view_row["view_id"]) if view_row else (view_id or None)
    raw_rows = _rows_at_version(lake, "curation_memberships", membership_version)
    scoped_rows = _filter_replay_membership_rows(
        raw_rows,
        view_id=resolved_view_id,
        target_grain=normalized_grain,
        target_ids=normalized_targets,
        scenario_ids=normalized_scenarios,
        as_of=context.as_of,
    )
    latest_by_target = _latest_membership_by_target(scoped_rows)
    latest_rows = tuple(sorted(latest_by_target.values(), key=_membership_sort_key))
    latest_ids = {str(row["membership_id"]) for row in latest_rows}
    superseded_rows = tuple(
        row for row in scoped_rows if str(row["membership_id"]) not in latest_ids
    )
    history_rows = tuple(scoped_rows if policy == "history" else latest_rows)
    view = _view_from_row(view_row, lake=lake) if view_row else None
    latest_decisions = tuple(_membership_audit_row(row) for row in latest_rows)
    membership_history = tuple(_membership_audit_row(row) for row in history_rows)
    superseded_decisions = tuple(_membership_audit_row(row) for row in superseded_rows)
    snapshot_row = context.snapshot_row
    report = {
        "schema_version": "lancedb-robotics/curation-replay/v1",
        "lake_uri": lake.uri,
        "view": _view_report(view),
        "target_grain": normalized_grain or "all",
        "target_ids": list(normalized_targets),
        "scenario_ids": list(normalized_scenarios),
        "as_of": _jsonable(context.as_of),
        "as_of_transform_id": context.as_of_transform_id,
        "snapshot_name": str(snapshot_row["name"]) if snapshot_row else "",
        "snapshot_dataset_id": str(snapshot_row["dataset_id"]) if snapshot_row else "",
        "superseded_policy": policy,
        "latest_decisions": list(latest_decisions),
        "membership_history": list(membership_history),
        "superseded_decisions": list(superseded_decisions),
        "latest_count": len(latest_decisions),
        "history_count": len(membership_history),
        "superseded_count": len(superseded_decisions),
        "table_version_validation": context.validation,
        "read_table_versions": _replay_read_versions(lake, context.table_versions),
    }
    return CurationDecisionResolution(
        view=view,
        target_grain=normalized_grain or "all",
        target_ids=normalized_targets,
        as_of=context.as_of,
        latest_decisions=latest_decisions,
        membership_history=membership_history,
        superseded_decisions=superseded_decisions,
        report=report,
    )


def _trace_membership(
    lake: Lake,
    *,
    snapshot_name: str,
    scenario_id: str,
    superseded_policy: str,
) -> CurationMembershipTrace:
    scenario = str(scenario_id).strip()
    if not scenario:
        raise CurationError("scenario_id is required")
    context = _replay_context(lake, as_of=None, transform_id=None, snapshot_name=snapshot_name)
    if context.snapshot_row is None:
        raise CurationError(f"no dataset snapshot named {snapshot_name!r} in {lake.uri}")
    snapshot_row = context.snapshot_row
    snapshot_ids = _scenario_ids_from_snapshot_row(snapshot_row)
    snapshot_report = _snapshot_report(snapshot_row, snapshot_ids)
    source = snapshot_report["query_spec"].get("source") or {}
    source_report = source.get("report") or {}
    view_id = str(source_report.get("view_id") or "")
    view_name = str(source_report.get("view_name") or "")
    resolution = _resolve_membership(
        lake,
        view_name=view_name or None,
        view_id=view_id or None,
        target_grain=None,
        target_ids=(),
        scenario_ids=(scenario,),
        as_of=None,
        transform_id=None,
        snapshot_name=snapshot_name,
        superseded_policy=superseded_policy,
    )
    included = scenario in set(snapshot_ids)
    scenario_latest = _latest_scenario_decision(resolution.membership_history, scenario)
    final_result = _final_membership_result(
        included_in_snapshot=included,
        latest_decision=scenario_latest,
    )
    transform_ids = set(_transform_ids_from_payload(source))
    transform_ids.add(str(snapshot_row.get("transform_id") or ""))
    for row in resolution.membership_history:
        transform_ids.add(str(row.get("transform_id") or ""))
    transforms = _transform_audit_rows(lake, transform_ids, as_of=context.as_of)
    chain = _supersession_chain(resolution.membership_history, scenario_latest)
    report = {
        "schema_version": "lancedb-robotics/curation-membership-trace/v1",
        "lake_uri": lake.uri,
        "snapshot": snapshot_report,
        "scenario_id": scenario,
        "included_in_snapshot": included,
        "final_result": final_result,
        "saved_view": resolution.report["view"],
        "source": source,
        "automatic_transform_decisions": [
            row
            for row in resolution.membership_history
            if row.get("source") in {"dedup", "model", "rule", "active-learning", "gap-analysis"}
        ],
        "membership_history": list(resolution.membership_history),
        "latest_decision": scenario_latest,
        "supersession_chain": chain,
        "transforms": transforms,
        "table_version_validation": context.validation,
    }
    return CurationMembershipTrace(
        snapshot_name=str(snapshot_row["name"]),
        dataset_id=str(snapshot_row["dataset_id"]),
        scenario_id=scenario,
        final_result=final_result,
        included_in_snapshot=included,
        resolution=resolution,
        report=report,
    )


def _normalize_superseded_policy(policy: str) -> str:
    normalized = str(policy or "latest").strip().lower().replace("_", "-")
    aliases = {
        "exclude": "latest",
        "excluded": "latest",
        "latest-only": "latest",
        "include": "history",
        "included": "history",
        "include-superseded": "history",
        "all": "history",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"latest", "history"}:
        raise CurationError("superseded_policy must be latest or history")
    return normalized


def _replay_context(
    lake: Lake,
    *,
    as_of: datetime | str | None,
    transform_id: str | None,
    snapshot_name: str | None,
) -> _ReplayContext:
    provided = [
        value is not None and str(value).strip() != ""
        for value in (as_of, transform_id, snapshot_name)
    ]
    if sum(provided) > 1:
        raise CurationError("provide only one of as_of, transform_id, or snapshot_name")
    if snapshot_name:
        snapshot = _latest_snapshot_row(lake, snapshot_name)
        validation = _validate_replay_table_versions(lake, snapshot)
        return _ReplayContext(
            as_of=_normalize_replay_datetime(snapshot.get("created_at"), "snapshot.created_at"),
            as_of_transform_id=str(snapshot.get("transform_id") or ""),
            snapshot_row=snapshot,
            table_versions=_pinned_version_map(snapshot.get("table_versions") or ()),
            validation=validation,
        )
    if transform_id:
        transform = _transform_row(lake, transform_id)
        snapshot = _snapshot_row_for_transform(lake, transform_id)
        if snapshot:
            validation = _validate_replay_table_versions(lake, snapshot)
            return _ReplayContext(
                as_of=_normalize_replay_datetime(
                    snapshot.get("created_at"),
                    "snapshot.created_at",
                ),
                as_of_transform_id=str(transform_id),
                snapshot_row=snapshot,
                table_versions=_pinned_version_map(snapshot.get("table_versions") or ()),
                validation=validation,
            )
        return _ReplayContext(
            as_of=_normalize_replay_datetime(
                transform.get("finished_at") or transform.get("created_at"),
                "transform.finished_at",
            ),
            as_of_transform_id=str(transform_id),
            snapshot_row=None,
            table_versions={},
            validation={"status": "not-applicable", "checked": []},
        )
    return _ReplayContext(
        as_of=_normalize_replay_datetime(as_of, "as_of"),
        as_of_transform_id="",
        snapshot_row=None,
        table_versions={},
        validation={"status": "not-applicable", "checked": []},
    )


def _normalize_replay_datetime(value: datetime | str | None, label: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        normalized = value
    else:
        raw = str(value).strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            normalized = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise CurationError(f"{label} must be an ISO-8601 datetime") from exc
    if normalized.tzinfo is None:
        return normalized.replace(tzinfo=UTC)
    return normalized.astimezone(UTC)


def _snapshot_row_for_transform(lake: Lake, transform_id: str) -> dict[str, Any] | None:
    rows = [
        row
        for row in lake.table("dataset_snapshots").to_arrow().to_pylist()
        if row.get("transform_id") == transform_id
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: (row["created_at"], row["dataset_id"]))


def _transform_row(lake: Lake, transform_id: str) -> dict[str, Any]:
    rows = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row.get("transform_id") == transform_id
    ]
    if not rows:
        raise CurationError(f"no transform_run {transform_id!r} in {lake.uri}")
    return max(rows, key=lambda row: (row.get("created_at"), row["transform_id"]))


def _pinned_version_map(rows: Sequence[Any]) -> dict[str, int]:
    versions: dict[str, int] = {}
    for row in rows:
        table = str(row.get("table") or "")
        if table and row.get("version") is not None:
            versions[table] = int(row["version"])
    return versions


def _validate_replay_table_versions(lake: Lake, snapshot: dict[str, Any]) -> dict[str, Any]:
    versions = _pinned_version_map(snapshot.get("table_versions") or ())
    required = {"scenarios", "curation_views", "curation_memberships"}
    missing = sorted(required - set(versions))
    if missing:
        raise CurationError(
            f"snapshot {snapshot['name']!r} does not pin required replay tables: {missing}"
        )
    require_lake_capability(lake, VERSIONING, operation="snapshot version replay validation")
    checked = []
    for table_name, pinned_version in sorted(versions.items()):
        try:
            table = lake.table(table_name)
            current_version = int(table.version)
        except Exception as exc:
            raise CurationError(
                f"snapshot {snapshot['name']!r} pins missing table {table_name!r}"
            ) from exc
        checked_out = False
        try:
            table.checkout(pinned_version)
            checked_out = True
        except Exception as exc:
            raise CurationError(
                f"snapshot {snapshot['name']!r} pins unreadable table version "
                f"{table_name}@{pinned_version}"
            ) from exc
        finally:
            if checked_out:
                table.checkout_latest()
        checked.append(
            {
                "table": table_name,
                "pinned_version": pinned_version,
                "current_version": current_version,
                "matches_current": current_version == pinned_version,
                "status": "readable",
            }
        )
    return {"status": "passed", "checked": checked}


def _rows_at_version(
    lake: Lake,
    table_name: str,
    version: int | None,
) -> list[dict[str, Any]]:
    table = lake.table(table_name)
    if version is None:
        return table.to_arrow().to_pylist()
    require_lake_capability(lake, VERSIONING, operation="as-of table version checkout")
    table.checkout(int(version))
    try:
        return table.to_arrow().to_pylist()
    finally:
        table.checkout_latest()


def _view_row_for_replay(
    lake: Lake,
    *,
    view_name: str | None,
    view_id: str | None,
    as_of: datetime | None,
    table_version: int | None,
) -> dict[str, Any] | None:
    if not view_name and not view_id:
        return None
    rows = _rows_at_version(lake, "curation_views", table_version)
    candidates = []
    for row in rows:
        if view_id and row.get("view_id") != view_id:
            continue
        if view_name and row.get("name") != view_name:
            continue
        created_at = _normalize_replay_datetime(row.get("created_at"), "view.created_at")
        if as_of is not None and created_at is not None and created_at > as_of:
            continue
        candidates.append(row)
    if not candidates:
        label = view_id or view_name
        raise CurationError(f"no curation view {label!r} visible in replay scope")
    return max(candidates, key=lambda row: (row["created_at"], row["view_id"]))


def _filter_replay_membership_rows(
    rows: Sequence[dict[str, Any]],
    *,
    view_id: str | None,
    target_grain: str,
    target_ids: Sequence[str],
    scenario_ids: Sequence[str],
    as_of: datetime | None,
) -> tuple[dict[str, Any], ...]:
    target_set = {str(item) for item in target_ids}
    scenario_set = {str(item) for item in scenario_ids}
    filtered = []
    for row in rows:
        if view_id and row.get("view_id") != view_id:
            continue
        if target_grain and row.get("target_grain") != target_grain:
            continue
        if target_set and str(row.get("target_id") or "") not in target_set:
            continue
        if scenario_set and not _membership_row_matches_scenario(row, scenario_set):
            continue
        created_at = _normalize_replay_datetime(row.get("created_at"), "membership.created_at")
        if as_of is not None and created_at is not None and created_at > as_of:
            continue
        filtered.append(row)
    return tuple(sorted(filtered, key=_membership_sort_key))


def _membership_row_matches_scenario(
    row: dict[str, Any],
    scenario_ids: set[str],
) -> bool:
    scenario_id = str(row.get("scenario_id") or "")
    target_grain = str(row.get("target_grain") or "")
    target_id = str(row.get("target_id") or "")
    return scenario_id in scenario_ids or (
        target_grain == "scenario" and target_id in scenario_ids
    )


def _membership_sort_key(row: dict[str, Any]) -> tuple[Any, str]:
    return row.get("created_at"), str(row.get("membership_id") or "")


def _membership_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "membership_id": str(row.get("membership_id") or ""),
        "view_id": str(row.get("view_id") or ""),
        "target_grain": str(row.get("target_grain") or ""),
        "target_id": str(row.get("target_id") or ""),
        "scenario_id": str(row.get("scenario_id") or ""),
        "decision": str(row.get("decision") or ""),
        "reason_code": str(row.get("reason_code") or ""),
        "reason": str(row.get("reason") or ""),
        "note": str(row.get("note") or ""),
        "reviewer": str(row.get("reviewer") or ""),
        "queue": str(row.get("queue") or ""),
        "priority": int(row.get("priority") or 0),
        "score": row.get("score"),
        "metadata": _metadata_dict(row.get("metadata") or ()),
        "source": str(row.get("source") or ""),
        "supersedes_membership_id": str(row.get("supersedes_membership_id") or ""),
        "created_by": str(row.get("created_by") or ""),
        "transform_id": str(row.get("transform_id") or ""),
        "created_at": _jsonable(row.get("created_at")),
    }


def _view_report(view: CurationView | None) -> dict[str, Any] | None:
    if view is None:
        return None
    return {
        "view_id": view.view_id,
        "name": view.name,
        "owner": view.owner,
        "tags": list(view.tags),
        "description": view.description,
        "status": view.status,
        "membership_storage": view.membership_storage,
        "membership_count": view.membership_count,
        "transform_id": view.transform_id,
        "table_versions": [
            {"table": table, "version": version, "tag": ""}
            for table, version in view.table_versions
        ],
    }


def _replay_read_versions(lake: Lake, pinned_versions: Mapping[str, int]) -> list[dict[str, Any]]:
    tables = ("curation_views", "curation_memberships")
    rows = []
    for table_name in tables:
        version = pinned_versions.get(table_name)
        if version is None:
            version = int(lake.table(table_name).version)
        rows.append({"table": table_name, "version": int(version), "tag": ""})
    return rows


def _latest_scenario_decision(
    rows: Sequence[dict[str, Any]],
    scenario_id: str,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("target_grain") == "scenario" and row.get("target_id") == scenario_id
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row.get("created_at") or "", row["membership_id"]))


def _final_membership_result(
    *,
    included_in_snapshot: bool,
    latest_decision: dict[str, Any] | None,
) -> str:
    decision = str((latest_decision or {}).get("decision") or "")
    if not included_in_snapshot and decision in _EXCLUDING_DECISIONS:
        return decision
    if included_in_snapshot:
        return "include"
    return decision or "exclude"


def _supersession_chain(
    rows: Sequence[dict[str, Any]],
    latest_decision: dict[str, Any] | None,
) -> list[str]:
    if latest_decision is None:
        return []
    by_id = {str(row["membership_id"]): row for row in rows}
    chain = []
    cursor: dict[str, Any] | None = latest_decision
    while cursor:
        membership_id = str(cursor.get("membership_id") or "")
        if not membership_id or membership_id in chain:
            break
        chain.append(membership_id)
        previous_id = str(cursor.get("supersedes_membership_id") or "")
        cursor = by_id.get(previous_id)
    return list(reversed(chain))


def _transform_ids_from_payload(value: Any) -> set[str]:
    transform_ids: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.endswith("transform_id") and isinstance(item, str) and item:
                transform_ids.add(item)
            elif (
                key.endswith("transform_ids")
                and isinstance(item, Sequence)
                and not isinstance(item, (str, bytes, bytearray))
            ):
                transform_ids.update(str(inner) for inner in item if str(inner))
            else:
                transform_ids.update(_transform_ids_from_payload(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            transform_ids.update(_transform_ids_from_payload(item))
    return transform_ids


def _transform_audit_rows(
    lake: Lake,
    transform_ids: set[str],
    *,
    as_of: datetime | None,
) -> list[dict[str, Any]]:
    requested = {item for item in transform_ids if item}
    rows = []
    for row in lake.table("transform_runs").to_arrow().to_pylist():
        transform_id = str(row.get("transform_id") or "")
        if transform_id not in requested:
            continue
        created_at = _normalize_replay_datetime(row.get("created_at"), "transform.created_at")
        if as_of is not None and created_at is not None and created_at > as_of:
            continue
        rows.append(_transform_audit_row(row))
    return sorted(rows, key=lambda row: (row.get("created_at") or "", row["transform_id"]))


def _transform_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    params = _json_object(row.get("params"))
    return {
        "transform_id": str(row.get("transform_id") or ""),
        "kind": str(row.get("kind") or ""),
        "operation": str(params.get("operation") or str(row.get("kind") or "")),
        "status": str(row.get("status") or ""),
        "input_table_versions": _version_rows(row.get("input_table_versions") or ()),
        "output_tables": list(row.get("output_tables") or ()),
        "prior_operation_transform_ids": list(params.get("prior_operation_transform_ids") or ()),
        "input_scenario_ids": list(params.get("input_scenario_ids") or ()),
        "output_scenario_ids": list(params.get("output_scenario_ids") or ()),
        "created_by": str(row.get("created_by") or ""),
        "created_at": _jsonable(row.get("created_at")),
    }


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _normalize_comparison_metrics(metrics: Sequence[str] | None) -> tuple[str, ...]:
    if metrics is None:
        return _DEFAULT_COMPARISON_METRICS
    normalized: list[str] = []
    for item in metrics:
        token = str(item).strip().lower().replace("_", "-")
        if not token:
            continue
        metric = _COMPARISON_METRIC_ALIASES.get(token, token)
        if metric == "all":
            return _DEFAULT_COMPARISON_METRICS
        if metric not in _DEFAULT_COMPARISON_METRICS:
            raise CurationError(
                f"unknown comparison metric {item!r}; expected one of "
                f"{', '.join(_DEFAULT_COMPARISON_METRICS)}"
            )
        normalized.append(metric)
    return tuple(dict.fromkeys(normalized)) or _DEFAULT_COMPARISON_METRICS


# ---------------------------------------------------------------------------
# Backlog 0094: bounded-memory streaming execution + comparison planning.
# ---------------------------------------------------------------------------


def _comparison_batch_size(batch_size: int | None) -> int:
    if batch_size is None:
        return _COMPARISON_DEFAULT_BATCH_SIZE
    try:
        value = int(batch_size)
    except (TypeError, ValueError):
        return _COMPARISON_DEFAULT_BATCH_SIZE
    return value if value > 0 else _COMPARISON_DEFAULT_BATCH_SIZE


def _comparison_preview_limit(preview_limit: int | None) -> int:
    if preview_limit is None:
        return _COMPARISON_DEFAULT_PREVIEW_LIMIT
    value = int(preview_limit)
    if value < 0:
        raise CurationError("comparison preview_limit must be non-negative")
    return value


@dataclass
class _ComparisonExecutionStats:
    """Instrumentation proving the streamed path never materializes a full table.

    ``peak_batch_rows`` is the largest row count handed to Python at once;
    ``materialized_tables`` records any table that fell back to a full
    ``to_arrow().to_pylist()`` scan. For supported metrics both stay bounded.
    """

    batch_size: int
    peak_batch_rows: int = 0
    total_scanned_rows: int = 0
    streamed_tables: set[str] = field(default_factory=set)
    materialized_tables: set[str] = field(default_factory=set)

    def record_batch(self, table: str, count: int) -> None:
        self.streamed_tables.add(table)
        self.total_scanned_rows += int(count)
        if count > self.peak_batch_rows:
            self.peak_batch_rows = int(count)

    def record_full_scan(self, table: str, count: int) -> None:
        self.materialized_tables.add(table)
        self.total_scanned_rows += int(count)
        if count > self.peak_batch_rows:
            self.peak_batch_rows = int(count)

    def to_report(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "peak_batch_rows": self.peak_batch_rows,
            "total_scanned_rows": self.total_scanned_rows,
            "streamed_tables": sorted(self.streamed_tables),
            "materialized_tables": sorted(self.materialized_tables),
        }


def _stream_table_rows(
    lake: Lake,
    table: str,
    *,
    columns: Sequence[str] | None = None,
    where_sql: str | None = None,
    batch_size: int = _COMPARISON_DEFAULT_BATCH_SIZE,
    stats: _ComparisonExecutionStats | None = None,
) -> Iterable[list[dict[str, Any]]]:
    """Yield bounded row batches with column projection + filter pushdown.

    Falls back to a single materialized scan only when the streaming query
    cannot be built (recorded in ``stats.materialized_tables`` for honesty).
    Callers must re-apply their own row predicate so the fallback stays correct.
    """
    handle = lake.table(table)
    available = set(handle.schema.names)
    projected = [column for column in (columns or ()) if column in available]
    batches = None
    try:
        query = handle.search()
        if projected:
            query = query.select(projected)
        if where_sql:
            query = query.where(where_sql)
        batches = query.to_batches(batch_size=batch_size)
    except Exception:
        batches = None
    if batches is not None:
        for batch in batches:
            rows = batch.to_pylist()
            if stats is not None:
                stats.record_batch(table, len(rows))
            yield rows
        return
    rows = handle.to_arrow().to_pylist()
    if projected:
        wanted = set(projected)
        rows = [{key: value for key, value in row.items() if key in wanted} for row in rows]
    if stats is not None:
        stats.record_full_scan(table, len(rows))
    yield rows


def _comparison_scanner_for(stats: _ComparisonExecutionStats | None):
    """Bind a bounded-memory scan helper for ``ComparisonMetricContext.stream``."""

    def scanner(
        lake: Lake,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        where_sql: str | None = None,
        batch_size: int = _COMPARISON_DEFAULT_BATCH_SIZE,
    ) -> Iterable[list[dict[str, Any]]]:
        yield from _stream_table_rows(
            lake,
            table,
            columns=columns,
            where_sql=where_sql,
            batch_size=batch_size,
            stats=stats,
        )

    return scanner


class _QualityAccumulator:
    """Streaming accumulator reproducing ``_quality_summary`` without row lists."""

    def __init__(self) -> None:
        self.scenario_count = 0
        self.score_count = 0
        self.score_sum = 0.0
        self.score_min: float | None = None
        self.score_max: float | None = None
        self.flag_counts: dict[str, int] = {}
        self.flagged_count = 0

    def add(self, row: dict[str, Any], run: dict[str, Any]) -> None:
        self.scenario_count += 1
        score = _numeric_value(row, "quality_score")
        if score is not None:
            self.score_count += 1
            self.score_sum += score
            self.score_min = score if self.score_min is None else min(self.score_min, score)
            self.score_max = score if self.score_max is None else max(self.score_max, score)
        run_flags = list(run.get("quality_flags") or ())
        row_flags = list(row.get("quality_flags") or ())
        for flag in run_flags + row_flags:
            self.flag_counts[str(flag)] = self.flag_counts.get(str(flag), 0) + 1
        if run_flags or row_flags:
            self.flagged_count += 1

    def to_summary(self) -> dict[str, Any]:
        return {
            "scenario_count": self.scenario_count,
            "quality_score_count": self.score_count,
            "quality_score_mean": self.score_sum / self.score_count if self.score_count else None,
            "quality_score_min": self.score_min,
            "quality_score_max": self.score_max,
            "quality_flag_counts": dict(sorted(self.flag_counts.items())),
            "flagged_scenario_count": self.flagged_count,
        }


def _comparison_run_lookup(
    lake: Lake,
    dimensions: Sequence[str],
    *,
    stats: _ComparisonExecutionStats | None,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    columns = {"run_id", "quality_flags", "raw_uri"}
    for dimension in dimensions:
        columns.update(_candidate_keys(str(dimension)))
    runs: dict[str, dict[str, Any]] = {}
    for rows in _stream_table_rows(
        lake, "runs", columns=sorted(columns), batch_size=batch_size, stats=stats
    ):
        for row in rows:
            run_id = str(row.get("run_id") or "")
            if run_id:
                runs[run_id] = row
    return runs


def _comparison_scenario_pass(
    lake: Lake,
    *,
    union_ids: Sequence[str],
    left_set: set[str],
    right_set: set[str],
    runs: dict[str, dict[str, Any]],
    dimensions: Sequence[str],
    need_coverage: bool,
    need_quality: bool,
    need_label_map: bool,
    stats: _ComparisonExecutionStats | None,
    batch_size: int,
) -> dict[str, Any]:
    """Single streamed pass over scenarios feeding coverage/quality/label map."""
    union_set = set(union_ids)
    columns = {"scenario_id", "run_id", "coverage_tags", "observation_ids"}
    if need_quality:
        columns.update({"quality_score", "quality_flags"})
    for dimension in dimensions:
        columns.update(_candidate_keys(str(dimension)))
    left_slices: dict[str, int] = {}
    right_slices: dict[str, int] = {}
    quality_left = _QualityAccumulator()
    quality_right = _QualityAccumulator()
    observation_to_scenario: dict[str, str] = {}
    found: set[str] = set()
    for rows in _stream_table_rows(
        lake, "scenarios", columns=sorted(columns), batch_size=batch_size, stats=stats
    ):
        for row in rows:
            scenario_id = str(row.get("scenario_id") or "")
            if scenario_id not in union_set:
                continue
            found.add(scenario_id)
            run = runs.get(str(row.get("run_id") or ""), {})
            if need_label_map:
                for observation_id in row.get("observation_ids") or ():
                    observation_to_scenario[str(observation_id)] = scenario_id
            if need_coverage and dimensions:
                label = _slice_label(row, run, dimensions)
                if scenario_id in left_set:
                    left_slices[label] = left_slices.get(label, 0) + 1
                if scenario_id in right_set:
                    right_slices[label] = right_slices.get(label, 0) + 1
            if need_quality:
                if scenario_id in left_set:
                    quality_left.add(row, run)
                if scenario_id in right_set:
                    quality_right.add(row, run)
    missing = sorted(union_set - found)
    if missing:
        raise CurationError(f"selected scenarios are missing from the lake: {missing}")
    return {
        "left_slices": dict(sorted(left_slices.items())),
        "right_slices": dict(sorted(right_slices.items())),
        "quality_left": quality_left,
        "quality_right": quality_right,
        "observation_to_scenario": observation_to_scenario,
    }


def _coverage_from_slices(
    left_slices: dict[str, int],
    right_slices: dict[str, int],
    left_count: int,
    right_count: int,
    dimensions: Sequence[str],
) -> dict[str, Any]:
    if not dimensions:
        return {
            "dimensions": [],
            "left_slices": {},
            "right_slices": {},
            "deltas": {},
            "left_total_count": left_count,
            "right_total_count": right_count,
        }
    labels = sorted(set(left_slices) | set(right_slices))
    left_total = left_count or 1
    right_total = right_count or 1
    deltas = {
        label: {
            "left_count": left_slices.get(label, 0),
            "right_count": right_slices.get(label, 0),
            "delta_count": right_slices.get(label, 0) - left_slices.get(label, 0),
            "left_percentage": left_slices.get(label, 0) / left_total,
            "right_percentage": right_slices.get(label, 0) / right_total,
            "delta_percentage": (right_slices.get(label, 0) / right_total)
            - (left_slices.get(label, 0) / left_total),
        }
        for label in labels
    }
    return {
        "dimensions": list(dimensions),
        "left_slices": left_slices,
        "right_slices": right_slices,
        "deltas": deltas,
        "left_total_count": left_count,
        "right_total_count": right_count,
    }


def _streaming_label_completeness(
    lake: Lake,
    scenario_ids: Sequence[str],
    observation_to_scenario: dict[str, str],
    *,
    stats: _ComparisonExecutionStats | None,
    batch_size: int,
    preview_limit: int,
) -> dict[str, Any]:
    scenario_set = set(scenario_ids)
    labeled: set[str] = set()
    label_count = 0
    label_type_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for rows in _stream_table_rows(
        lake,
        "labels",
        columns=["scenario_id", "observation_id", "label_type", "status"],
        batch_size=batch_size,
        stats=stats,
    ):
        for row in rows:
            scenario_id = str(row.get("scenario_id") or "")
            if not scenario_id:
                scenario_id = observation_to_scenario.get(str(row.get("observation_id") or ""), "")
            if scenario_id not in scenario_set:
                continue
            labeled.add(scenario_id)
            label_count += 1
            label_type = str(row.get("label_type") or "label")
            status = str(row.get("status") or "unknown")
            label_type_counts[label_type] = label_type_counts.get(label_type, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
    labeled_sorted = sorted(labeled)
    missing_sorted = sorted(scenario_set - labeled)
    return {
        "scenario_count": len(scenario_ids),
        "label_count": label_count,
        "labeled_scenario_count": len(labeled_sorted),
        "missing_label_scenario_count": len(missing_sorted),
        "completeness": len(labeled_sorted) / len(scenario_ids) if scenario_ids else 0.0,
        "labeled_scenario_ids": labeled_sorted[:preview_limit],
        "missing_label_scenario_ids": missing_sorted[:preview_limit],
        "labeled_scenario_ids_truncated": len(labeled_sorted) > preview_limit,
        "missing_label_scenario_ids_truncated": len(missing_sorted) > preview_limit,
        "label_type_counts": dict(sorted(label_type_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
    }


def _membership_page_token(field_name: str, offset: int) -> str:
    return f"{field_name}:{offset}"


def _membership_field_key(field_name: str) -> str:
    token = str(field_name or "").strip().lower()
    mapping = {
        "added": "added_scenario_ids",
        "removed": "removed_scenario_ids",
        "shared": "shared_scenario_ids",
        "right_only": "added_scenario_ids",
        "left_only": "removed_scenario_ids",
    }
    if token not in mapping:
        raise CurationError(
            f"membership field must be one of {', '.join(_COMPARISON_MEMBERSHIP_FIELDS)}"
        )
    return mapping[token]


def _membership_preview(
    membership: dict[str, Any], *, preview_limit: int
) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for field_name in _COMPARISON_MEMBERSHIP_FIELDS:
        ids = list(membership.get(_membership_field_key(field_name)) or [])
        total = len(ids)
        truncated = total > preview_limit
        preview[field_name] = {
            "count": total,
            "preview": ids[:preview_limit],
            "truncated": truncated,
            "preview_limit": preview_limit,
            "next_page_token": _membership_page_token(field_name, preview_limit)
            if truncated
            else None,
        }
    return preview


def _bounded_membership(
    membership: dict[str, Any], *, preview_limit: int
) -> dict[str, Any]:
    bounded = dict(membership)
    for key in ("shared_scenario_ids", "removed_scenario_ids", "added_scenario_ids"):
        ids = list(membership.get(key) or [])
        bounded[key] = ids[:preview_limit]
        bounded[f"{key}_truncated"] = len(ids) > preview_limit
    return bounded


def _comparison_count_rows(lake: Lake, table: str, where_sql: str | None = None) -> int:
    handle = lake.table(table)
    try:
        return int(handle.count_rows(where_sql) if where_sql else handle.count_rows())
    except Exception:
        try:
            return int(handle.count_rows())
        except Exception:
            return 0


_COMPARISON_DATASET_SCOPED_TABLES = frozenset({"model_outputs", "curation_materializations"})


def _comparison_metric_estimate(
    lake: Lake,
    metric: str,
    *,
    left_count: int,
    right_count: int,
    left_dataset_id: str,
    right_dataset_id: str,
) -> int:
    if metric == "membership":
        return left_count + right_count
    total = 0
    for table in _COMPARISON_METRIC_TABLES.get(metric, ()):
        if table in _COMPARISON_DATASET_SCOPED_TABLES:
            total += _comparison_count_rows(
                lake, table, f"dataset_id = {_sql_literal(left_dataset_id)}"
            )
            total += _comparison_count_rows(
                lake, table, f"dataset_id = {_sql_literal(right_dataset_id)}"
            )
        else:
            total += _comparison_count_rows(lake, table)
    return total


def _build_comparison_plan(
    lake: Lake,
    left: str,
    right: str,
    *,
    by: Sequence[str],
    metrics: Sequence[str] | None,
    plugins: Sequence[ComparisonMetricPlugin | str] | None,
    batch_size: int | None,
    local_row_budget: int | None,
) -> CurationComparisonPlan:
    left_snapshot = _latest_snapshot_row(lake, left)
    right_snapshot = _latest_snapshot_row(lake, right)
    left_ids = _scenario_ids_from_snapshot_row(left_snapshot)
    right_ids = _scenario_ids_from_snapshot_row(right_snapshot)
    normalized_metrics = _normalize_comparison_metrics(metrics)
    dimensions = tuple(dict.fromkeys(str(dim) for dim in by if str(dim)))
    resolved_plugins = resolve_comparison_plugins(plugins)
    resolved_batch_size = _comparison_batch_size(batch_size)
    budget = int(local_row_budget) if local_row_budget is not None else _COMPARISON_LOCAL_ROW_BUDGET
    membership = _membership_diff(left_ids, right_ids)
    ctx = ComparisonMetricContext(
        lake=lake,
        left=left,
        right=right,
        left_snapshot=left_snapshot,
        right_snapshot=right_snapshot,
        left_dataset_id=str(left_snapshot["dataset_id"]),
        right_dataset_id=str(right_snapshot["dataset_id"]),
        left_ids=tuple(left_ids),
        right_ids=tuple(right_ids),
        dimensions=dimensions,
        membership=membership,
        batch_size=resolved_batch_size,
        _scanner=_comparison_scanner_for(None),
    )

    required_tables: set[str] = set()
    for metric in normalized_metrics:
        required_tables.update(_COMPARISON_METRIC_TABLES.get(metric, ()))
    for plugin in resolved_plugins:
        required_tables.update(str(table) for table in plugin.required_tables(ctx))
    version_pairs = _table_versions(lake, tables=tuple(sorted(required_tables))) if required_tables else []
    version_by_table = {row["table"]: int(row["version"]) for row in version_pairs}

    entries: list[ComparisonMetricPlanEntry] = []
    for metric in normalized_metrics:
        tables = tuple(_COMPARISON_METRIC_TABLES.get(metric, ()))
        estimate = _comparison_metric_estimate(
            lake,
            metric,
            left_count=len(left_ids),
            right_count=len(right_ids),
            left_dataset_id=ctx.left_dataset_id,
            right_dataset_id=ctx.right_dataset_id,
        )
        streamed = metric in _COMPARISON_STREAMED_METRICS
        execution = _EXTERNAL_EXECUTION if estimate > budget else _LOCAL_EXECUTION
        notes = "" if streamed else "v1 materializes selected rows; streaming tracked as follow-up"
        entries.append(
            ComparisonMetricPlanEntry(
                metric=metric,
                kind="builtin",
                required_tables=tables,
                table_versions=tuple(
                    (table, version_by_table[table]) for table in tables if table in version_by_table
                ),
                estimated_rows=estimate,
                execution=execution,
                streamed=streamed,
                notes=notes,
            )
        )
    plugin_metric_names: list[str] = []
    for plugin in resolved_plugins:
        plugin_metric_names.append(plugin.name)
        tables = tuple(str(table) for table in plugin.required_tables(ctx))
        estimate = int(plugin.estimate_rows(ctx))
        execution = str(plugin.execution(ctx) or _LOCAL_EXECUTION)
        if execution not in (_LOCAL_EXECUTION, _EXTERNAL_EXECUTION):
            execution = _LOCAL_EXECUTION
        if estimate > budget:
            execution = _EXTERNAL_EXECUTION
        entries.append(
            ComparisonMetricPlanEntry(
                metric=plugin.name,
                kind="plugin",
                required_tables=tables,
                table_versions=tuple(
                    (table, version_by_table[table]) for table in tables if table in version_by_table
                ),
                estimated_rows=estimate,
                execution=execution,
                streamed=False,
                notes="plugin-provided metric section",
            )
        )

    estimated_scan_rows = sum(entry.estimated_rows for entry in entries)
    peak_candidates = [
        resolved_batch_size if entry.streamed else entry.estimated_rows for entry in entries
    ]
    estimated_peak_rows = max(peak_candidates) if peak_candidates else 0
    requires_external = any(entry.execution == _EXTERNAL_EXECUTION for entry in entries)
    return CurationComparisonPlan(
        left=left,
        right=right,
        left_dataset_id=ctx.left_dataset_id,
        right_dataset_id=ctx.right_dataset_id,
        left_count=len(left_ids),
        right_count=len(right_ids),
        metrics=tuple(normalized_metrics),
        plugin_metrics=tuple(plugin_metric_names),
        dimensions=dimensions,
        entries=tuple(entries),
        table_versions=tuple(sorted(version_by_table.items())),
        batch_size=resolved_batch_size,
        local_row_budget=budget,
        estimated_scan_rows=estimated_scan_rows,
        estimated_peak_rows=estimated_peak_rows,
        requires_external_executor=requires_external,
    )


def _snapshot_row_by_dataset_id(lake: Lake, dataset_id: str) -> dict[str, Any]:
    for row in lake.table("dataset_snapshots").to_arrow().to_pylist():
        if str(row.get("dataset_id") or "") == str(dataset_id):
            return row
    raise CurationError(f"no dataset snapshot with dataset id {dataset_id!r} in {lake.uri}")


def _resolve_comparison_entry(lake: Lake, id_or_name: str) -> CurationComparisonEntry:
    token = str(id_or_name or "").strip()
    if not token:
        raise CurationError("a comparison id or snapshot-pair alias is required")
    rows = lake.table("curation_comparisons").to_arrow().to_pylist()
    entries = sorted(
        (_comparison_entry_from_row(row) for row in rows),
        key=_comparison_sort_key,
        reverse=True,
    )
    for entry in entries:
        if entry.comparison_id == token:
            return entry
    for entry in entries:
        if token in (entry.pair_alias, entry.left_snapshot_name, entry.right_snapshot_name):
            return entry
    raise CurationError(f"no curation comparison {token!r} in catalog")


def _comparison_membership_page(
    lake: Lake,
    id_or_name: str,
    *,
    field_name: str | None,
    offset: int,
    limit: int | None,
    page_token: str | None,
) -> CurationComparisonMembershipPage:
    if page_token:
        token = str(page_token)
        if ":" not in token:
            raise CurationError(f"invalid membership page token {page_token!r}")
        field_part, _, offset_part = token.partition(":")
        field_name = field_part
        try:
            offset = int(offset_part)
        except ValueError as exc:
            raise CurationError(f"invalid membership page token {page_token!r}") from exc
    if field_name is None:
        raise CurationError("a membership field or page token is required")
    if offset < 0:
        raise CurationError("membership page offset must be non-negative")
    page_limit = _COMPARISON_DEFAULT_PREVIEW_LIMIT if limit is None else int(limit)
    if page_limit < 0:
        raise CurationError("membership page limit must be non-negative")
    entry = _resolve_comparison_entry(lake, id_or_name)
    membership_key = _membership_field_key(field_name)
    normalized_field = {
        "added_scenario_ids": "added",
        "removed_scenario_ids": "removed",
        "shared_scenario_ids": "shared",
    }[membership_key]
    left_ids = _scenario_ids_from_snapshot_row(
        _snapshot_row_by_dataset_id(lake, entry.left_dataset_id)
    )
    right_ids = _scenario_ids_from_snapshot_row(
        _snapshot_row_by_dataset_id(lake, entry.right_dataset_id)
    )
    membership = _membership_diff(left_ids, right_ids)
    ids = list(membership.get(membership_key) or [])
    total = len(ids)
    page = tuple(ids[offset : offset + page_limit]) if page_limit else ()
    next_offset = offset + page_limit
    next_token = (
        _membership_page_token(normalized_field, next_offset)
        if page_limit and next_offset < total
        else None
    )
    return CurationComparisonMembershipPage(
        comparison_id=entry.comparison_id,
        field=normalized_field,
        offset=offset,
        limit=page_limit,
        total=total,
        scenario_ids=page,
        next_page_token=next_token,
    )


# Tables the comparison itself mutates while recording, so their advance is not
# evidence that the comparison's *source data* changed.
_COMPARISON_STALENESS_IGNORE_TABLES = frozenset({"transform_runs", "curation_comparisons"})


def _comparison_staleness(lake: Lake, id_or_name: str) -> CurationComparisonStaleness:
    entry = _resolve_comparison_entry(lake, id_or_name)
    recorded = {
        table: version
        for table, version in entry.table_versions
        if table not in _COMPARISON_STALENESS_IGNORE_TABLES
    }
    if recorded:
        current_pairs = _table_versions(lake, tables=tuple(sorted(recorded)))
        current = {row["table"]: int(row["version"]) for row in current_pairs}
    else:
        current = {}
    advanced: list[dict[str, Any]] = []
    for table in sorted(recorded):
        recorded_version = int(recorded[table])
        current_version = int(current.get(table, recorded_version))
        if current_version > recorded_version:
            advanced.append(
                {
                    "table": table,
                    "recorded_version": recorded_version,
                    "current_version": current_version,
                }
            )
    return CurationComparisonStaleness(
        comparison_id=entry.comparison_id,
        stale=bool(advanced),
        advanced_tables=tuple(advanced),
        recorded_table_versions=tuple(sorted(recorded.items())),
        current_table_versions=tuple(sorted(current.items())),
        checked_at=datetime.now(UTC),
    )


def _snapshot_report(row: dict[str, Any], scenario_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "name": str(row["name"]),
        "dataset_id": str(row["dataset_id"]),
        "kind": str(row.get("kind") or ""),
        "tag": str(row.get("tag") or ""),
        "split": str(row.get("split") or ""),
        "scenario_ids": list(scenario_ids),
        "scenario_count": len(scenario_ids),
        "query_spec": json.loads(row.get("query_spec") or "{}"),
        "table_versions": _version_rows(row.get("table_versions") or ()),
        "transform_id": str(row.get("transform_id") or ""),
        "created_at": _jsonable(row.get("created_at")),
    }


def _membership_diff(left_ids: Sequence[str], right_ids: Sequence[str]) -> dict[str, Any]:
    left_set = set(left_ids)
    right_set = set(right_ids)
    shared = sorted(left_set & right_set)
    removed = sorted(left_set - right_set)
    added = sorted(right_set - left_set)
    return {
        "shared_scenario_ids": shared,
        "removed_scenario_ids": removed,
        "added_scenario_ids": added,
        "shared_count": len(shared),
        "removed_count": len(removed),
        "added_count": len(added),
    }


def _coverage_comparison(
    left_ids: Sequence[str],
    right_ids: Sequence[str],
    scenarios: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    dimensions: Sequence[str],
) -> dict[str, Any]:
    if not dimensions:
        return {
            "dimensions": [],
            "left_slices": {},
            "right_slices": {},
            "deltas": {},
            "left_total_count": len(left_ids),
            "right_total_count": len(right_ids),
        }
    left_slices = _slice_counts(left_ids, scenarios, runs, dimensions)
    right_slices = _slice_counts(right_ids, scenarios, runs, dimensions)
    labels = sorted(set(left_slices) | set(right_slices))
    left_total = len(left_ids) or 1
    right_total = len(right_ids) or 1
    deltas = {
        label: {
            "left_count": left_slices.get(label, 0),
            "right_count": right_slices.get(label, 0),
            "delta_count": right_slices.get(label, 0) - left_slices.get(label, 0),
            "left_percentage": left_slices.get(label, 0) / left_total,
            "right_percentage": right_slices.get(label, 0) / right_total,
            "delta_percentage": (right_slices.get(label, 0) / right_total)
            - (left_slices.get(label, 0) / left_total),
        }
        for label in labels
    }
    return {
        "dimensions": list(dimensions),
        "left_slices": left_slices,
        "right_slices": right_slices,
        "deltas": deltas,
        "left_total_count": len(left_ids),
        "right_total_count": len(right_ids),
    }


def _source_overlap(lake: Lake, left_ids: Sequence[str], right_ids: Sequence[str]) -> dict[str, Any]:
    left = _source_summary(lake, left_ids)
    right = _source_summary(lake, right_ids)
    return {
        "left": left,
        "right": right,
        "shared_run_ids": sorted(set(left["run_ids"]) & set(right["run_ids"])),
        "shared_episode_ids": sorted(set(left["episode_ids"]) & set(right["episode_ids"])),
        "shared_raw_uris": sorted(set(left["raw_uris"]) & set(right["raw_uris"])),
        "shared_run_count": len(set(left["run_ids"]) & set(right["run_ids"])),
        "shared_episode_count": len(set(left["episode_ids"]) & set(right["episode_ids"])),
        "shared_raw_uri_count": len(set(left["raw_uris"]) & set(right["raw_uris"])),
    }


def _source_summary(lake: Lake, scenario_ids: Sequence[str]) -> dict[str, Any]:
    rows = _selected_rows(lake, scenario_ids)
    runs = _run_rows(lake)
    observations = _observations_for_scenarios(lake, scenario_ids)
    run_ids = sorted({str(row.get("run_id") or "") for row in rows if row.get("run_id")})
    episode_ids = sorted(
        {str(row.get("episode_id") or "") for row in observations if row.get("episode_id")}
    )
    raw_uris = sorted(
        {
            str(value)
            for value in (
                [runs.get(run_id, {}).get("raw_uri") for run_id in run_ids]
                + [row.get("raw_uri") for row in observations]
            )
            if value
        }
    )
    return {
        "scenario_count": len(scenario_ids),
        "run_ids": run_ids,
        "run_count": len(run_ids),
        "episode_ids": episode_ids,
        "episode_count": len(episode_ids),
        "raw_uris": raw_uris,
        "raw_uri_count": len(raw_uris),
    }


def _duplicate_pressure_summary(lake: Lake, scenario_ids: Sequence[str]) -> dict[str, Any]:
    scenario_set = set(scenario_ids)
    latest = _latest_membership_by_scenario(_membership_rows(lake))
    groups: dict[str, dict[str, Any]] = {}
    dedup_decision_count = 0
    excluded_duplicate_count = 0
    for row in latest.values():
        metadata = _metadata_dict(row.get("metadata") or ())
        source = str(row.get("source") or "")
        reason_code = str(row.get("reason_code") or "")
        if source != "dedup" and reason_code != "semantic-duplicate":
            continue
        scenario_id = str(row.get("scenario_id") or row.get("target_id") or "")
        representative_id = str(metadata.get("representative_id") or scenario_id)
        if scenario_id not in scenario_set and representative_id not in scenario_set:
            continue
        group = groups.setdefault(
            representative_id,
            {
                "representative_id": representative_id,
                "selected_scenario_ids": [],
                "excluded_scenario_ids": [],
            },
        )
        if scenario_id in scenario_set:
            group["selected_scenario_ids"].append(scenario_id)
        if row.get("decision") == "exclude":
            group["excluded_scenario_ids"].append(scenario_id)
            excluded_duplicate_count += 1
        dedup_decision_count += 1

    normalized_groups = []
    for group in groups.values():
        selected = sorted(set(group["selected_scenario_ids"]))
        excluded = sorted(set(group["excluded_scenario_ids"]))
        normalized_groups.append(
            {
                "representative_id": group["representative_id"],
                "selected_scenario_ids": selected,
                "excluded_scenario_ids": excluded,
                "selected_count": len(selected),
                "excluded_count": len(excluded),
            }
        )
    normalized_groups.sort(key=lambda item: item["representative_id"])
    return {
        "scenario_count": len(scenario_ids),
        "duplicate_group_count": len(normalized_groups),
        "dedup_decision_count": dedup_decision_count,
        "excluded_duplicate_count": excluded_duplicate_count,
        "duplicate_pressure": excluded_duplicate_count / len(scenario_ids)
        if scenario_ids
        else 0.0,
        "groups": normalized_groups,
    }


def _quality_summary(rows: Sequence[dict[str, Any]], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores = [
        score
        for score in (_numeric_value(row, "quality_score") for row in rows)
        if score is not None
    ]
    flag_counts: dict[str, int] = {}
    for row in rows:
        run = runs.get(row["run_id"], {})
        flags = list(run.get("quality_flags") or ()) + list(row.get("quality_flags") or ())
        for flag in flags:
            flag_counts[str(flag)] = flag_counts.get(str(flag), 0) + 1
    return {
        "scenario_count": len(rows),
        "quality_score_count": len(scores),
        "quality_score_mean": sum(scores) / len(scores) if scores else None,
        "quality_score_min": min(scores) if scores else None,
        "quality_score_max": max(scores) if scores else None,
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "flagged_scenario_count": sum(
            1
            for row in rows
            if (runs.get(row["run_id"], {}).get("quality_flags") or ())
            or (row.get("quality_flags") or ())
        ),
    }


def _label_completeness_summary(
    lake: Lake,
    scenario_ids: Sequence[str],
    scenario_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    scenario_set = set(scenario_ids)
    observation_to_scenario = {
        str(observation_id): str(row["scenario_id"])
        for row in scenario_rows
        for observation_id in (row.get("observation_ids") or ())
    }
    labels_by_scenario: dict[str, list[dict[str, Any]]] = {scenario_id: [] for scenario_id in scenario_ids}
    label_type_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in lake.table("labels").to_arrow().to_pylist():
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            scenario_id = observation_to_scenario.get(str(row.get("observation_id") or ""), "")
        if scenario_id not in scenario_set:
            continue
        labels_by_scenario.setdefault(scenario_id, []).append(row)
        label_type = str(row.get("label_type") or "label")
        status = str(row.get("status") or "unknown")
        label_type_counts[label_type] = label_type_counts.get(label_type, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
    labeled = sorted(sid for sid, rows in labels_by_scenario.items() if rows)
    missing = sorted(scenario_set - set(labeled))
    return {
        "scenario_count": len(scenario_ids),
        "label_count": sum(len(rows) for rows in labels_by_scenario.values()),
        "labeled_scenario_count": len(labeled),
        "missing_label_scenario_count": len(missing),
        "completeness": len(labeled) / len(scenario_ids) if scenario_ids else 0.0,
        "labeled_scenario_ids": labeled,
        "missing_label_scenario_ids": missing,
        "label_type_counts": dict(sorted(label_type_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
    }


def _payload_summary(lake: Lake, scenario_ids: Sequence[str]) -> dict[str, Any]:
    observations = _observations_for_scenarios(lake, scenario_ids)
    total_bytes = sum(_payload_size(row.get("payload_blob")) for row in observations)
    return {
        "scenario_count": len(scenario_ids),
        "observation_count": len(observations),
        "total_payload_bytes": total_bytes,
        "average_payload_bytes": total_bytes / len(observations) if observations else 0.0,
    }


def _dataset_scoped_rows(
    lake: Lake,
    table: str,
    dataset_id: str,
    *,
    columns: Sequence[str] | None = None,
    stats: _ComparisonExecutionStats | None = None,
    batch_size: int = _COMPARISON_DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Stream a table filtered to ``dataset_id`` (pushdown + Python re-check)."""
    where_sql = f"dataset_id = {_sql_literal(dataset_id)}"
    collected: list[dict[str, Any]] = []
    for rows in _stream_table_rows(
        lake, table, columns=columns, where_sql=where_sql, batch_size=batch_size, stats=stats
    ):
        for row in rows:
            if str(row.get("dataset_id") or "") == str(dataset_id):
                collected.append(row)
    return collected


def _materialization_summary(
    lake: Lake,
    dataset_id: str,
    *,
    stats: _ComparisonExecutionStats | None = None,
    batch_size: int = _COMPARISON_DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    rows = _dataset_scoped_rows(
        lake, "curation_materializations", dataset_id, stats=stats, batch_size=batch_size
    )
    rows.sort(key=lambda row: (row["created_at"], row["materialization_id"]))
    copied = sum(int(row.get("copied_payload_bytes") or 0) for row in rows)
    logical = sum(int(row.get("logical_reference_bytes") or 0) for row in rows)
    planned = sum(_planned_payload_bytes_from_materialization_row(row) for row in rows)
    total = sum(int(row.get("total_payload_bytes") or 0) for row in rows)
    metadata = sum(int(row.get("metadata_bytes_written") or 0) for row in rows)
    return {
        "materialization_count": len(rows),
        "total_payload_bytes": total,
        "copied_payload_bytes": copied,
        "logical_reference_bytes": logical,
        "planned_payload_bytes": planned,
        "metadata_bytes_written": metadata,
        "copy_ratio": copied / total if total else 0.0,
        "reports": [
            {
                "materialization_id": row["materialization_id"],
                "target_format": row["target_format"],
                "output_uri": row["output_uri"],
                "mode": row["mode"],
                "copied_payload_bytes": int(row.get("copied_payload_bytes") or 0),
                "logical_reference_bytes": int(row.get("logical_reference_bytes") or 0),
                "planned_payload_bytes": _planned_payload_bytes_from_materialization_row(row),
                "metadata_bytes_written": int(row.get("metadata_bytes_written") or 0),
                "copy_ratio": float(row.get("copy_ratio") or 0.0),
                "transform_id": row.get("transform_id") or "",
            }
            for row in rows
        ],
    }


def _planned_payload_bytes_from_materialization_row(row: dict[str, Any]) -> int:
    try:
        report = json.loads(row.get("report_json") or "{}")
    except json.JSONDecodeError:
        return 0
    accounting = report.get("accounting") or {}
    return int(
        accounting.get("payload_bytes_planned")
        or report.get("payload_bytes_planned")
        or report.get("planned_payload_bytes")
        or 0
    )


def _downstream_training_eval(
    lake: Lake,
    dataset_id: str,
    *,
    stats: _ComparisonExecutionStats | None = None,
    batch_size: int = _COMPARISON_DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    output_rows = _model_output_rows_for_dataset(
        lake, dataset_id, stats=stats, batch_size=batch_size
    )
    output_ids = {str(row["model_output_id"]) for row in output_rows}
    feedback_rows: list[dict[str, Any]] = []
    for rows in _stream_table_rows(lake, "feedback", batch_size=batch_size, stats=stats):
        for row in rows:
            if str(row.get("model_output_id") or "") in output_ids:
                feedback_rows.append(row)
    by_type: dict[str, list[dict[str, Any]]] = {}
    by_slice: dict[str, dict[str, list[dict[str, Any]]]] = {}
    regressions: list[dict[str, Any]] = []
    for row in output_rows:
        by_type.setdefault(str(row.get("output_type") or "score"), []).append(row)
        metadata = _metadata_dict(row.get("metadata") or ())
        slice_label = str(metadata.get("slice") or "all")
        metric_name = str(metadata.get("metric") or row.get("output_type") or "score")
        by_slice.setdefault(slice_label, {}).setdefault(metric_name, []).append(row)
        if metadata.get("regressed"):
            regression = dict(metadata)
            regression.setdefault("source_model_output_id", row.get("model_output_id") or "")
            regression.setdefault("transform_id", row.get("transform_id") or "")
            regressions.append(regression)
    output_types = {
        output_type: _score_summary(rows)
        for output_type, rows in sorted(by_type.items())
    }
    slice_metrics = {
        slice_label: {
            metric_name: _score_summary(rows)
            for metric_name, rows in sorted(metrics.items())
        }
        for slice_label, metrics in sorted(by_slice.items())
    }
    return {
        "dataset_id": dataset_id,
        "training_run_ids": sorted(
            {str(row.get("producer_run_id") or "") for row in output_rows if row.get("producer_run_id")}
        ),
        "model_versions": sorted(
            {str(row.get("model_version") or "") for row in output_rows if row.get("model_version")}
        ),
        "model_output_ids": sorted(output_ids),
        "model_output_count": len(output_rows),
        "output_types": output_types,
        "slice_metrics": slice_metrics,
        "regression_count": len(regressions),
        "regressions": sorted(regressions, key=lambda item: _regression_key(item, 0)),
        "source_transform_ids": sorted(
            {str(row.get("transform_id") or "") for row in output_rows if row.get("transform_id")}
        ),
        "feedback": _feedback_summary(feedback_rows),
    }


def _score_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["score"]) for row in rows if row.get("score") is not None]
    return {
        "count": len(rows),
        "score_count": len(scores),
        "avg_score": sum(scores) / len(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "producer_run_ids": sorted(
            {str(row.get("producer_run_id") or "") for row in rows if row.get("producer_run_id")}
        ),
        "model_versions": sorted(
            {str(row.get("model_version") or "") for row in rows if row.get("model_version")}
        ),
        "transform_ids": sorted(
            {str(row.get("transform_id") or "") for row in rows if row.get("transform_id")}
        ),
    }


def _feedback_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in rows:
        feedback_type = str(row.get("feedback_type") or "feedback")
        severity = str(row.get("severity") or "unknown")
        status = str(row.get("status") or "unknown")
        type_counts[feedback_type] = type_counts.get(feedback_type, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "feedback_count": len(rows),
        "feedback_type_counts": dict(sorted(type_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "source_transform_ids": sorted(
            {str(row.get("transform_id") or "") for row in rows if row.get("transform_id")}
        ),
    }


def _model_output_rows_for_dataset(
    lake: Lake,
    dataset_id: str,
    *,
    stats: _ComparisonExecutionStats | None = None,
    batch_size: int = _COMPARISON_DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    return _dataset_scoped_rows(
        lake, "model_outputs", dataset_id, stats=stats, batch_size=batch_size
    )


def _numeric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in sorted(set(left) | set(right)):
        left_value = left.get(key)
        right_value = right.get(key)
        if (
            isinstance(left_value, (int, float))
            and not isinstance(left_value, bool)
            and isinstance(right_value, (int, float))
            and not isinstance(right_value, bool)
        ):
            delta[key] = right_value - left_value
    return delta


def _eval_metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {
        "model_output_count": right.get("model_output_count", 0)
        - left.get("model_output_count", 0),
        "regression_count": right.get("regression_count", 0)
        - left.get("regression_count", 0),
    }
    output_delta: dict[str, Any] = {}
    left_types = left.get("output_types") or {}
    right_types = right.get("output_types") or {}
    for output_type in sorted(set(left_types) | set(right_types)):
        output_delta[output_type] = _numeric_delta(
            left_types.get(output_type, {}),
            right_types.get(output_type, {}),
        )
    deltas["output_types"] = output_delta
    slice_delta: dict[str, Any] = {}
    left_slices = left.get("slice_metrics") or {}
    right_slices = right.get("slice_metrics") or {}
    for slice_label in sorted(set(left_slices) | set(right_slices)):
        metric_delta: dict[str, Any] = {}
        left_metrics = left_slices.get(slice_label, {})
        right_metrics = right_slices.get(slice_label, {})
        for metric_name in sorted(set(left_metrics) | set(right_metrics)):
            metric_delta[metric_name] = _numeric_delta(
                left_metrics.get(metric_name, {}),
                right_metrics.get(metric_name, {}),
            )
        slice_delta[slice_label] = metric_delta
    deltas["slice_metrics"] = slice_delta
    return deltas


def _version_rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                "table": str(row.get("table") or ""),
                "version": int(row.get("version") or 0),
                "tag": str(row.get("tag") or ""),
            }
        )
    return normalized


def _merge_version_rows(*row_groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    versions: dict[str, dict[str, Any]] = {}
    for rows in row_groups:
        for row in _version_rows(rows):
            table = row["table"]
            if not table:
                continue
            current = versions.get(table)
            if current is None or row["version"] > current["version"]:
                versions[table] = row
    return [versions[table] for table in sorted(versions)]


def _comparison_table_versions(
    lake: Lake,
    left_snapshot: dict[str, Any],
    right_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    metric_versions = _table_versions(
        lake,
        tables=(
            "labels",
            "model_outputs",
            "feedback",
            "curation_memberships",
            "curation_review_queues",
            "curation_materializations",
            "transform_runs",
        ),
    )
    return _merge_version_rows(
        left_snapshot.get("table_versions") or (),
        right_snapshot.get("table_versions") or (),
        metric_versions,
    )


def _comparison_prior_transform_ids(
    left_snapshot: dict[str, Any],
    right_snapshot: dict[str, Any],
    report: dict[str, Any],
) -> tuple[str, ...]:
    ids = [left_snapshot.get("transform_id") or "", right_snapshot.get("transform_id") or ""]
    ids.extend(_collect_transform_ids(report))
    return tuple(dict.fromkeys(str(item) for item in ids if str(item)))


def _collect_transform_ids(value: Any) -> list[str]:
    ids: list[str] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            if key in {"transform_id", "transform_ids", "source_transform_ids"}:
                if isinstance(inner, str) and inner:
                    ids.append(inner)
                elif isinstance(inner, list):
                    ids.extend(str(item) for item in inner if str(item))
            ids.extend(_collect_transform_ids(inner))
    elif isinstance(value, list):
        for item in value:
            ids.extend(_collect_transform_ids(item))
    return ids


def _persist_comparison_report(
    lake: Lake,
    *,
    comparison_id: str,
    left_snapshot: dict[str, Any],
    right_snapshot: dict[str, Any],
    metrics: Sequence[str],
    dimensions: Sequence[str],
    membership: dict[str, Any],
    report: dict[str, Any],
    table_versions: Sequence[dict[str, Any]],
    transform_id: str,
    created_by: str,
) -> None:
    now = datetime.now(UTC)
    report_json = json.dumps(_jsonable(report), sort_keys=True)
    table = lake.table("curation_comparisons")
    table.delete(f"comparison_id = '{comparison_id}'")
    table.add(
        pa.Table.from_pylist(
            [
                {
                    "comparison_id": comparison_id,
                    "pair_alias": _comparison_pair_alias(
                        left_snapshot["name"], right_snapshot["name"]
                    ),
                    "state": _COMPARISON_STATE_ACTIVE,
                    "left_dataset_id": left_snapshot["dataset_id"],
                    "right_dataset_id": right_snapshot["dataset_id"],
                    "left_snapshot_name": left_snapshot["name"],
                    "right_snapshot_name": right_snapshot["name"],
                    "metrics": list(metrics),
                    "dimensions": list(dimensions),
                    "added_scenario_count": int(membership["added_count"]),
                    "removed_scenario_count": int(membership["removed_count"]),
                    "shared_scenario_count": int(membership["shared_count"]),
                    "report_json": report_json,
                    "report_sha1": hashlib.sha1(report_json.encode()).hexdigest(),
                    "report_bytes": len(report_json.encode()),
                    "retention_policy_json": "",
                    "archived_at": None,
                    "pruned_at": None,
                    "table_versions": list(table_versions),
                    "created_by": created_by,
                    "transform_id": transform_id,
                    "created_at": now,
                }
            ],
            schema=CURATION_COMPARISONS_SCHEMA,
        )
    )


_COMPARISON_STATE_ACTIVE = "active"
_COMPARISON_STATE_ARCHIVED = "archived"
_COMPARISON_STATE_PRUNED = "pruned"
_COMPARISON_STATES = (
    _COMPARISON_STATE_ACTIVE,
    _COMPARISON_STATE_ARCHIVED,
    _COMPARISON_STATE_PRUNED,
)


def _comparison_pair_alias(left_snapshot_name: str, right_snapshot_name: str) -> str:
    """Stable snapshot-pair handle (``"<left>..<right>"``) for latest resolution."""
    return f"{str(left_snapshot_name)}..{str(right_snapshot_name)}"


def _normalize_comparison_state(state: Any) -> str:
    normalized = str(state or "").strip().lower()
    if normalized not in _COMPARISON_STATES:
        raise CurationError(
            "comparison state must be one of "
            f"{', '.join(_COMPARISON_STATES)}"
        )
    return normalized


def _comparison_state(row: Mapping[str, Any]) -> str:
    raw = str(row.get("state") or "").strip().lower()
    return raw if raw in _COMPARISON_STATES else _COMPARISON_STATE_ACTIVE


def _comparison_table_version_pairs(
    table_versions: Sequence[Any],
) -> tuple[tuple[str, int], ...]:
    pairs: list[tuple[str, int]] = []
    for item in table_versions or ():
        if isinstance(item, Mapping):
            table = str(item.get("table") or "")
            if table:
                pairs.append((table, int(item.get("version") or 0)))
    return tuple(pairs)


def _comparison_entry_from_row(row: Mapping[str, Any]) -> CurationComparisonEntry:
    state = _comparison_state(row)
    report_json = str(row.get("report_json") or "")
    left_name = str(row.get("left_snapshot_name") or "")
    right_name = str(row.get("right_snapshot_name") or "")
    pair_alias = str(row.get("pair_alias") or "") or _comparison_pair_alias(
        left_name, right_name
    )
    return CurationComparisonEntry(
        comparison_id=str(row.get("comparison_id") or ""),
        pair_alias=pair_alias,
        state=state,
        left_snapshot_name=left_name,
        right_snapshot_name=right_name,
        left_dataset_id=str(row.get("left_dataset_id") or ""),
        right_dataset_id=str(row.get("right_dataset_id") or ""),
        metrics=tuple(str(item) for item in row.get("metrics") or () if str(item)),
        dimensions=tuple(str(item) for item in row.get("dimensions") or () if str(item)),
        added_scenario_count=int(row.get("added_scenario_count") or 0),
        removed_scenario_count=int(row.get("removed_scenario_count") or 0),
        shared_scenario_count=int(row.get("shared_scenario_count") or 0),
        report_sha1=str(row.get("report_sha1") or ""),
        report_bytes=int(row.get("report_bytes") or 0),
        report_available=bool(report_json) and state != _COMPARISON_STATE_PRUNED,
        retention_policy=dict(_loads_json(row.get("retention_policy_json"), {})),
        table_versions=_comparison_table_version_pairs(row.get("table_versions") or ()),
        created_by=str(row.get("created_by") or ""),
        transform_id=str(row.get("transform_id") or ""),
        created_at=_as_optional_utc(row.get("created_at")) or datetime.now(UTC),
        archived_at=_as_optional_utc(row.get("archived_at")),
        pruned_at=_as_optional_utc(row.get("pruned_at")),
    )


def _comparison_sort_key(entry: CurationComparisonEntry) -> tuple[datetime, str]:
    return (entry.created_at, entry.comparison_id)


def _list_comparison_catalog(
    lake: Lake,
    *,
    snapshot: str | None,
    left: str | None,
    right: str | None,
    metric: str | None,
    state: str | None,
    include_archived: bool,
    include_pruned: bool,
    since: datetime | None,
    until: datetime | None,
    created_by: str | None,
    limit: int | None,
) -> tuple[CurationComparisonEntry, ...]:
    state_filter = _normalize_comparison_state(state) if state else None
    wanted_metrics = set(_normalize_comparison_metrics((metric,))) if metric else None
    rows = lake.table("curation_comparisons").to_arrow().to_pylist()
    entries = [_comparison_entry_from_row(row) for row in rows]
    filtered: list[CurationComparisonEntry] = []
    for entry in entries:
        if state_filter is not None and entry.state != state_filter:
            continue
        if not include_archived and entry.state == _COMPARISON_STATE_ARCHIVED:
            continue
        if not include_pruned and entry.state == _COMPARISON_STATE_PRUNED:
            continue
        if left is not None and left not in (entry.left_snapshot_name, entry.left_dataset_id):
            continue
        if right is not None and right not in (entry.right_snapshot_name, entry.right_dataset_id):
            continue
        if snapshot is not None and snapshot not in (
            entry.left_snapshot_name,
            entry.right_snapshot_name,
            entry.left_dataset_id,
            entry.right_dataset_id,
        ):
            continue
        if wanted_metrics is not None and not (wanted_metrics & set(entry.metrics)):
            continue
        if not _matches_optional(entry.created_by, created_by):
            continue
        if since is not None and entry.created_at < _as_utc(since):
            continue
        if until is not None and entry.created_at > _as_utc(until):
            continue
        filtered.append(entry)
    filtered.sort(key=_comparison_sort_key, reverse=True)
    if limit is not None:
        if limit < 0:
            raise CurationError("comparison catalog limit must be non-negative")
        filtered = filtered[:limit]
    return tuple(filtered)


def _comparison_from_row(row: Mapping[str, Any]) -> CurationComparison:
    comparison_id = str(row.get("comparison_id") or "")
    state = _comparison_state(row)
    report_json = str(row.get("report_json") or "")
    if state == _COMPARISON_STATE_PRUNED or not report_json:
        raise CurationError(
            f"curation comparison {comparison_id!r} body was pruned; "
            "catalog audit metadata (snapshot ids, table versions, transform id) "
            "is still available via list_comparisons"
        )
    return CurationComparison(
        comparison_id=comparison_id,
        left=str(row.get("left_snapshot_name") or ""),
        right=str(row.get("right_snapshot_name") or ""),
        report=_loads_json(report_json, {}),
        transform_id=str(row.get("transform_id") or ""),
    )


def _resolve_comparison(lake: Lake, id_or_name: str) -> CurationComparison:
    token = str(id_or_name or "").strip()
    if not token:
        raise CurationError("a comparison id or snapshot-pair alias is required")
    rows = lake.table("curation_comparisons").to_arrow().to_pylist()
    rows.sort(
        key=lambda row: (
            _as_optional_utc(row.get("created_at")) or datetime.min.replace(tzinfo=UTC),
            str(row.get("comparison_id") or ""),
        ),
        reverse=True,
    )
    for row in rows:
        if str(row.get("comparison_id") or "") == token:
            return _comparison_from_row(row)
    for row in rows:
        if _comparison_state(row) == _COMPARISON_STATE_PRUNED:
            continue
        names = (
            str(row.get("pair_alias") or "")
            or _comparison_pair_alias(
                str(row.get("left_snapshot_name") or ""),
                str(row.get("right_snapshot_name") or ""),
            ),
            str(row.get("left_snapshot_name") or ""),
            str(row.get("right_snapshot_name") or ""),
        )
        if token in names:
            return _comparison_from_row(row)
    raise CurationError(f"no curation comparison {token!r} in catalog")


def _prune_comparison_catalog(
    lake: Lake,
    *,
    retain_latest: int,
    older_than: datetime | timedelta | None,
    dry_run: bool,
    created_by: str,
) -> CurationComparisonRetentionReport:
    if retain_latest < 0:
        raise CurationError("retain_latest must be non-negative")
    now = datetime.now(UTC)
    cutoff = _retention_cutoff(older_than, now)
    rows = lake.table("curation_comparisons").to_arrow().to_pylist()
    rows_by_id = {str(row.get("comparison_id") or ""): row for row in rows}
    entries = [_comparison_entry_from_row(row) for row in rows]
    groups: dict[str, list[CurationComparisonEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.pair_alias, []).append(entry)
    retained_ids: set[str] = set()
    archive_targets: list[CurationComparisonEntry] = []
    prune_targets: list[CurationComparisonEntry] = []
    for group in groups.values():
        group.sort(key=_comparison_sort_key, reverse=True)
        retained_ids.update(entry.comparison_id for entry in group[:retain_latest])
        for entry in group[retain_latest:]:
            if entry.state == _COMPARISON_STATE_PRUNED:
                continue
            if cutoff is not None and entry.created_at < cutoff:
                prune_targets.append(entry)
            elif entry.state != _COMPARISON_STATE_ARCHIVED:
                archive_targets.append(entry)
    body_bytes_before = sum(entry.report_bytes for entry in entries if entry.report_available)
    pruned_bytes = sum(entry.report_bytes for entry in prune_targets)
    policy = {
        "retain_latest": retain_latest,
        "older_than": cutoff.isoformat() if cutoff else None,
        "applied_at": now.isoformat(),
    }
    transform_id = ""
    if (archive_targets or prune_targets) and not dry_run:
        policy_json = json.dumps(policy, sort_keys=True)
        updated_rows: list[dict[str, Any]] = []
        for entry in archive_targets:
            row = dict(rows_by_id[entry.comparison_id])
            row["state"] = _COMPARISON_STATE_ARCHIVED
            row["archived_at"] = now
            row["retention_policy_json"] = policy_json
            updated_rows.append(row)
        for entry in prune_targets:
            row = dict(rows_by_id[entry.comparison_id])
            row["state"] = _COMPARISON_STATE_PRUNED
            row["report_json"] = ""
            row["pruned_at"] = now
            if row.get("archived_at") is None:
                row["archived_at"] = now
            row["retention_policy_json"] = policy_json
            updated_rows.append(row)
        table = lake.table("curation_comparisons")
        for row in updated_rows:
            table.delete(f"comparison_id = '{row['comparison_id']}'")
        table.add(pa.Table.from_pylist(updated_rows, schema=CURATION_COMPARISONS_SCHEMA))
        transform_id = _record_curation_transform(
            lake,
            operation="comparison-retention",
            input_scenario_ids=(),
            output_scenario_ids=(),
            report={
                "operation": "comparison-retention",
                "archived_comparison_ids": [entry.comparison_id for entry in archive_targets],
                "pruned_comparison_ids": [entry.comparison_id for entry in prune_targets],
                "retained_comparison_ids": sorted(retained_ids),
                "body_bytes_before": body_bytes_before,
                "body_bytes_after": body_bytes_before - pruned_bytes,
                "policy": policy,
            },
            prior_transform_ids=tuple(
                dict.fromkeys(
                    entry.transform_id
                    for entry in (*archive_targets, *prune_targets)
                    if entry.transform_id
                )
            ),
            created_by=created_by,
            output_tables=("curation_comparisons",),
        )
    return CurationComparisonRetentionReport(
        archived_comparison_ids=tuple(entry.comparison_id for entry in archive_targets),
        pruned_comparison_ids=tuple(entry.comparison_id for entry in prune_targets),
        retained_comparison_ids=tuple(sorted(retained_ids)),
        dry_run=dry_run,
        body_bytes_before=body_bytes_before,
        body_bytes_after=body_bytes_before - pruned_bytes,
        policy=policy,
        transform_id=transform_id,
        created_at=now,
    )


def _retention_cutoff(
    value: datetime | timedelta | None, now: datetime
) -> datetime | None:
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


def _slice_counts(
    scenario_ids: Sequence[str],
    scenarios: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    dimensions: Sequence[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scenario_id in scenario_ids:
        row = scenarios.get(scenario_id)
        if not row:
            continue
        label = _slice_label(row, runs.get(row["run_id"], {}), dimensions)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _model_output_metrics(
    lake: Lake,
    dataset_id: str,
    *,
    stats: _ComparisonExecutionStats | None = None,
    batch_size: int = _COMPARISON_DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    rows = [
        row
        for row in _dataset_scoped_rows(
            lake, "model_outputs", dataset_id, stats=stats, batch_size=batch_size
        )
        if row.get("score") is not None
    ]
    by_type: dict[str, list[float]] = {}
    for row in rows:
        output_type = str(row.get("output_type") or "score")
        by_type.setdefault(output_type, []).append(float(row["score"]))
    return {
        key: {
            "count": len(values),
            "avg_score": sum(values) / len(values) if values else None,
            "min_score": min(values) if values else None,
            "max_score": max(values) if values else None,
        }
        for key, values in sorted(by_type.items())
    }


def _observations_for_scenarios(lake: Lake, scenario_ids: Sequence[str]) -> list[dict[str, Any]]:
    scenarios = {
        row["scenario_id"]: row
        for row in lake.table("scenarios").to_arrow().to_pylist()
        if row["scenario_id"] in set(scenario_ids)
    }
    observation_ids: set[str] = set()
    for row in scenarios.values():
        observation_ids.update(str(item) for item in row.get("observation_ids") or ())
    if observation_ids:
        return [
            row for row in lake.table("observations").to_arrow().to_pylist()
            if row["observation_id"] in observation_ids
        ]
    run_windows = [
        (row["run_id"], int(row["start_time_ns"]), int(row["end_time_ns"]))
        for row in scenarios.values()
    ]
    rows = []
    for observation in lake.table("observations").to_arrow().to_pylist():
        timestamp = int(observation.get("timestamp_ns") or 0)
        if any(
            observation["run_id"] == run_id and start <= timestamp <= end
            for run_id, start, end in run_windows
        ):
            rows.append(observation)
    return rows


def _payload_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, dict):
        size = value.get("size")
        if size is not None:
            return int(size)
    return 0


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _normalize_scope(scope: CurationScope | dict | Sequence[str] | None) -> CurationScope:
    if scope is None:
        return CurationScope()
    if isinstance(scope, CurationScope):
        return scope
    if isinstance(scope, dict):
        return CurationScope.from_filters(**dict(scope))
    if isinstance(scope, Sequence) and not isinstance(scope, str):
        return CurationScope(scenario_ids=tuple(str(item) for item in scope))
    raise CurationError("scope must be None, a CurationScope, a dict, or scenario-id sequence")


def _selected_rows(lake: Lake, scenario_ids: Sequence[str]) -> list[dict]:
    wanted = set(scenario_ids)
    rows = [
        row for row in lake.table("scenarios").to_arrow().to_pylist()
        if row["scenario_id"] in wanted
    ]
    if len(rows) != len(wanted):
        found = {row["scenario_id"] for row in rows}
        missing = sorted(wanted - found)
        raise CurationError(f"selected scenarios are missing from the lake: {missing}")
    return sorted(rows, key=_scenario_sort_key)


def _run_rows(lake: Lake) -> dict[str, dict]:
    return {row["run_id"]: row for row in lake.table("runs").to_arrow().to_pylist()}


def _scenario_sort_key(row: dict) -> tuple[int, str]:
    return (int(row.get("start_time_ns") or 0), row["scenario_id"])


def _require_embedding_column(lake: Lake, column: str) -> None:
    table = lake.table("scenarios")
    if column not in table.schema.names:
        raise CurationError(
            f"no scenarios.{column} column in {lake.uri}; run enrichment before curation"
        )


def _vector(row: dict, column: str) -> list[float]:
    raw = row.get(column)
    if raw is None:
        raise CurationError(f"scenario {row['scenario_id']} has no {column!r} embedding")
    vector = [float(value) for value in raw]
    if not vector:
        raise CurationError(f"scenario {row['scenario_id']} has an empty {column!r} embedding")
    return vector


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise CurationError("embedding vectors must have matching dimensions")
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _seed_from_event(
    lake: Lake,
    rows: list[dict],
    vectors: dict[str, list[float]],
    *,
    seed_event: str,
) -> tuple[list[float], dict[str, Any]]:
    event = next(
        (row for row in lake.table("events").to_arrow().to_pylist() if row["event_id"] == seed_event),
        None,
    )
    if event is None:
        raise CurationError(f"unknown seed event {seed_event!r}")
    candidates = [
        row
        for row in rows
        if (
            row.get("trigger_event_id") == seed_event
            or (
                row["run_id"] == event["run_id"]
                and row["start_time_ns"] <= event["timestamp_ns"] < row["end_time_ns"]
            )
        )
    ]
    if not candidates:
        raise CurationError(f"seed event {seed_event!r} does not fall inside the workbench scope")
    scenario = sorted(
        candidates,
        key=lambda row: (row["end_time_ns"] - row["start_time_ns"], row["scenario_id"]),
    )[0]
    return vectors[scenario["scenario_id"]], {
        "kind": "event",
        "event_id": seed_event,
        "scenario_id": scenario["scenario_id"],
    }


def _parse_seed_embedding(seed_embedding: Sequence[float] | str | None) -> list[float]:
    if seed_embedding is None:
        raise CurationError("seed_embedding is required")
    if isinstance(seed_embedding, str):
        try:
            values = [float(part.strip()) for part in seed_embedding.split(",") if part.strip()]
        except ValueError as exc:
            raise CurationError("--seed-vector must be a comma-separated list of floats") from exc
    else:
        values = [float(value) for value in seed_embedding]
    if not values:
        raise CurationError("seed_embedding must not be empty")
    return values


def _matches_scope(row: dict, run: dict, scope: CurationScope) -> bool:
    if scope.scenario_ids and row["scenario_id"] not in set(scope.scenario_ids):
        return False
    tags = set(row.get("coverage_tags") or ())
    if scope.coverage_tags and not set(scope.coverage_tags) <= tags:
        return False
    for key, expected in scope.filters.items():
        actual = _dimension_value(row, run, key)
        expected_values = {str(value) for value in _as_tuple(expected)}
        if str(actual) not in expected_values:
            return False
    return True


def _candidate_keys(name: str) -> tuple[str, ...]:
    keys = [name]
    if name.endswith("_id"):
        keys.append(name[:-3])
    else:
        keys.append(f"{name}_id")
    return tuple(dict.fromkeys(keys))


def _dimension_value(row: dict, run: dict, dimension: str) -> str:
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


def _coverage_value(tags: Sequence[str], key: str) -> str | None:
    for tag in tags:
        for sep in (":", "="):
            prefix = f"{key}{sep}"
            if tag.startswith(prefix):
                return tag[len(prefix):]
    return None


def _slice_label(row: dict, run: dict, dimensions: Sequence[str]) -> str:
    return "|".join(f"{dimension}={_dimension_value(row, run, dimension)}" for dimension in dimensions)


def _dimensions_from_gap_label(label: str) -> tuple[str, ...]:
    dimensions: list[str] = []
    for part in str(label).split("|"):
        if "=" not in part:
            continue
        dimension, _ = part.split("=", 1)
        if dimension:
            dimensions.append(dimension)
    return tuple(dimensions)


def _numeric_value(row: dict, column: str) -> float | None:
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


def _table_versions(lake: Lake, tables: Sequence[str] = _SOURCE_TABLES) -> list[dict[str, Any]]:
    return [
        {"table": table, "version": int(lake.table(table).version), "tag": ""}
        for table in tables
    ]


def _dedup_source_table_versions(lake: Lake) -> list[dict[str, Any]]:
    return _table_versions(lake, tables=_DEDUP_SOURCE_TABLES)


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _record_curation_transform(
    lake: Lake,
    *,
    operation: str,
    input_scenario_ids: Sequence[str],
    output_scenario_ids: Sequence[str],
    report: dict[str, Any],
    prior_transform_ids: Sequence[str],
    created_by: str,
    output_tables: Sequence[str] = (),
    lineage_context: Any | None = None,
) -> str:
    input_versions = _table_versions(lake)
    params = {
        **report,
        "input_scenario_ids": list(input_scenario_ids),
        "output_scenario_ids": list(output_scenario_ids),
        "prior_operation_transform_ids": list(prior_transform_ids),
        "input_table_versions": input_versions,
    }
    params = attach_lineage_context_to_params(params, lineage_context)
    output_table_list = list(output_tables)
    context_params = attach_lineage_context_to_params({}, lineage_context)
    digest_payload = {
        "operation": operation,
        "input_scenario_ids": list(input_scenario_ids),
        "output_scenario_ids": list(output_scenario_ids),
        "report": report,
        "input_table_versions": input_versions,
        "prior_operation_transform_ids": list(prior_transform_ids),
        "output_tables": output_table_list,
    }
    if context_params:
        digest_payload["lineage_context"] = context_params
    transform_id = "tfm-curate-" + operation.replace("-", "_") + "-" + _digest(digest_payload)
    now = datetime.now(UTC)
    transform_row = {
        "transform_id": transform_id,
        "kind": f"curation-{operation}",
        "input_uris": [],
        "input_table_versions": input_versions,
        "output_tables": output_table_list,
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
    # Emit lineage inline (backlog 0098): curation transform provenance without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
    return transform_id


def _record_stable_curation_transform(
    lake: Lake,
    *,
    operation: str,
    stable_payload: dict[str, Any],
    input_scenario_ids: Sequence[str],
    output_scenario_ids: Sequence[str],
    report: dict[str, Any],
    prior_transform_ids: Sequence[str],
    created_by: str,
    output_tables: Sequence[str] = (),
    stable_input_versions: Sequence[dict[str, Any]] | None = None,
) -> str:
    input_versions = _table_versions(lake)
    digest_input_versions = list(stable_input_versions or input_versions)
    output_table_list = list(output_tables)
    transform_id = "tfm-curate-" + operation.replace("-", "_") + "-" + _digest(
        {
            **_jsonable(stable_payload),
            "input_table_versions": digest_input_versions,
            "output_tables": output_table_list,
        }
    )
    params = {
        **_jsonable(report),
        "input_scenario_ids": list(input_scenario_ids),
        "output_scenario_ids": list(output_scenario_ids),
        "prior_operation_transform_ids": [
            str(item) for item in prior_transform_ids if str(item)
        ],
        "input_table_versions": input_versions,
    }
    now = datetime.now(UTC)
    transform_row = {
        "transform_id": transform_id,
        "kind": f"curation-{operation}",
        "input_uris": [],
        "input_table_versions": input_versions,
        "output_tables": output_table_list,
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
    # Emit lineage inline (backlog 0098): stable curation transform provenance without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
    return transform_id


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(inner) for key, inner in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(inner) for inner in value]
    return value
