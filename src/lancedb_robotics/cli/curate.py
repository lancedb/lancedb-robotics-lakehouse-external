"""`lancedb-robotics curate` subcommands."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import typer

from lancedb_robotics.cli.lineage_context import LINEAGE_CONTEXT_OPTION, load_lineage_context
from lancedb_robotics.review_connectors import JsonFileReviewToolConnector

curate_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
review_queue_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
curate_app.add_typer(
    review_queue_app,
    name="review-queue",
    help="Create and export logical review queues.",
)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_SNAPSHOT_OPTION = typer.Option(..., "--snapshot", help="Snapshot name to create.")
_OPTIONAL_SNAPSHOT_OPTION = typer.Option(None, "--snapshot", help="Snapshot name to create.")
_TAG_OPTION = typer.Option(None, "--tag", help="Snapshot tag (defaults to the name).")
_SCENARIO_ID_OPTION = typer.Option(
    None, "--scenario-id", help="Limit scope to an explicit scenario id; repeat for several."
)
_COVERAGE_TAG_OPTION = typer.Option(
    None, "--coverage-tag", help="Limit scope to scenarios carrying this coverage tag."
)
_EMBEDDING_COLUMN_OPTION = typer.Option(
    "embedding", "--embedding-column", help="Scenario embedding column to compare."
)
_THRESHOLD_OPTION = typer.Option(0.98, "--threshold", help="Cosine threshold for near duplicates.")
_REPRESENTATIVE_POLICY_OPTION = typer.Option(
    None,
    "--representative-policy",
    help="Representative policy token; repeat for quality, labels, rarity, earliest, latest, scenario-id.",
)
_SHARD_BY_OPTION = typer.Option(
    None, "--shard-by", help="Dimension used to shard dedup candidates; repeat for several."
)
_NEIGHBOR_LIMIT_OPTION = typer.Option(
    64, "--neighbor-limit", help="Indexed nearest neighbors to inspect per candidate."
)
_MAX_NEIGHBOR_LIMIT_OPTION = typer.Option(
    None,
    "--max-neighbor-limit",
    help="Maximum adaptive neighbors to inspect per candidate for distributed dedup.",
)
_NO_ADAPTIVE_NEIGHBOR_OPTION = typer.Option(
    False,
    "--no-adaptive-neighbor-expansion",
    help="Disable adaptive neighbor expansion for saturated distributed dedup shards.",
)
_RECALL_AUDIT_SAMPLE_SIZE_OPTION = typer.Option(
    0,
    "--recall-audit-sample-size",
    help="Sampled exact pairs per shard for distributed dedup recall audits.",
)
_RECALL_AUDIT_SEED_OPTION = typer.Option(
    0, "--recall-audit-seed", help="Deterministic seed for distributed recall-audit sampling."
)
_DEDUP_JOB_ID_OPTION = typer.Option(
    None, "--job-id", help="Stable distributed dedup job id used for resume."
)
_MAX_SHARDS_OPTION = typer.Option(
    None,
    "--max-shards",
    help="Run at most this many pending distributed dedup shards before stopping.",
)
_REQUIRE_INDEX_OPTION = typer.Option(
    False, "--require-index", help="Fail unless scenarios.<embedding-column> has a vector index."
)
_NO_RECORD_DECISIONS_OPTION = typer.Option(
    False, "--no-record-decisions", help="Do not write automatic curation membership decisions."
)
_BY_OPTION = typer.Option(..., "--by", help="Dimension to balance; repeat for several.")
_PER_SLICE_OPTION = typer.Option(..., "--per-slice", help="Maximum scenarios per slice.")
_DIVERSITY_BY_OPTION = typer.Option(None, "--by", help="Coverage dimension; repeat for several.")
_DIVERSITY_LIMIT_OPTION = typer.Option(..., "--limit", help="Maximum scenarios to sample.")
_DIVERSITY_METHOD_OPTION = typer.Option(
    "farthest-first", "--method", help="farthest-first or cluster-representative."
)
_DIVERSITY_CONSTRAINT_OPTION = typer.Option(
    None, "--constraint", help="JSON diversity constraint spec for optimize action."
)
_DIVERSITY_BACKEND_OPTION = typer.Option(
    "deterministic-greedy", "--backend", help="Constraint optimizer backend."
)
_DIVERSITY_MIN_PER_SLICE_OPTION = typer.Option(
    0, "--min-per-slice", help="Minimum scenarios to try to cover per configured slice."
)
_MAX_PER_DUPLICATE_GROUP_OPTION = typer.Option(
    1, "--max-per-duplicate-group", help="Maximum selected scenarios per semantic duplicate group."
)
_SEED_EVENT_OPTION = typer.Option(None, "--seed-event", help="Event id to mine around.")
_SEED_SCENARIO_OPTION = typer.Option(None, "--seed-scenario", help="Scenario id to mine around.")
_SEED_VECTOR_OPTION = typer.Option(None, "--seed-vector", help="Comma-separated embedding vector.")
_LIMIT_OPTION = typer.Option(500, "--limit", help="Maximum neighbor scenarios to keep.")
_MIN_SCORE_OPTION = typer.Option(None, "--min-score", help="Minimum scenario quality score.")
_SCORE_COLUMN_OPTION = typer.Option("quality_score", "--score-column", help="Score column/tag key.")
_INCLUDE_FLAG_OPTION = typer.Option(
    None, "--include-flag", help="Require a run quality flag; repeat for several."
)
_EXCLUDE_FLAG_OPTION = typer.Option(
    None, "--exclude-flag", help="Exclude a run quality flag; repeat for several."
)
_VIEW_OPTION = typer.Option(..., "--view", help="Saved curation view name.")
_OPTIONAL_VIEW_OPTION = typer.Option(None, "--view", help="Saved curation view name.")
_OWNER_OPTION = typer.Option(None, "--owner", help="Saved view owner.")
_VIEW_TAG_OPTION = typer.Option(None, "--view-tag", help="Saved view tag; repeat for several.")
_DESCRIPTION_OPTION = typer.Option("", "--description", help="Saved view description.")
_DECISION_OPTION = typer.Option(
    ..., "--decision", help="include, exclude, defer, needs-review, label, or relabel."
)
_TARGET_GRAIN_OPTION = typer.Option(
    "scenario",
    "--target-grain",
    help="Decision target grain: scenario, episode, observation, aligned-frame, or snapshot-row.",
)
_ROW_PLAN_TARGET_GRAIN_OPTION = typer.Option(
    "observation",
    "--target-grain",
    help="Row-plan grain: episode, observation, aligned-frame, or snapshot-row.",
)
_TARGET_ID_OPTION = typer.Option(
    None, "--target-id", help="Decision target id; repeat for several non-scenario decisions."
)
_REASON_OPTION = typer.Option("", "--reason", help="Decision rationale.")
_REASON_CODE_OPTION = typer.Option("", "--reason-code", help="Machine-readable decision reason.")
_NOTE_OPTION = typer.Option("", "--note", help="Freeform decision note.")
_REVIEWER_OPTION = typer.Option("", "--reviewer", help="Reviewer or actor name.")
_QUEUE_OPTION = typer.Option("", "--queue", help="Review/labeling queue name.")
_PRIORITY_OPTION = typer.Option(0, "--priority", help="Queue priority.")
_SCORE_OPTION = typer.Option(None, "--score", help="Optional model/manual confidence score.")
_SOURCE_OPTION = typer.Option(
    "human", "--source", help="Decision source: human, model, rule, dedup, active-learning, or gap-analysis."
)
_MIN_PER_SLICE_OPTION = typer.Option(1, "--min-per-slice", help="Minimum desired rows per slice.")
_GAP_BY_OPTION = typer.Option(None, "--by", help="Optional gap-analysis dimension.")
_COMPARE_BY_OPTION = typer.Option(None, "--by", help="Optional coverage dimension.")
_COMPARE_METRIC_OPTION = typer.Option(
    None,
    "--metric",
    help="Comparison metric to compute; repeat or use all, membership, coverage, quality, labels, duplicates, payload, materialization, eval.",
)
_LEFT_OPTION = typer.Option(..., "--left", help="Left snapshot name.")
_RIGHT_OPTION = typer.Option(..., "--right", help="Right snapshot name.")
_JSON_OPTION = typer.Option(False, "--json", help="Emit the machine-readable comparison report.")
_TARGET_FORMAT_OPTION = typer.Option(..., "--format", help="Projection/materialization format.")
_OUTPUT_URI_OPTION = typer.Option("", "--output-uri", help="Output URI or directory.")
_COPIED_BYTES_OPTION = typer.Option(0, "--copied-payload-bytes", help="Payload bytes copied.")
_METADATA_BYTES_OPTION = typer.Option(0, "--metadata-bytes", help="Metadata/control bytes written.")
_MODE_OPTION = typer.Option("projection", "--mode", help="logical, plan, projection, or export.")
_PROJECTION_TRANSFORM_OPTION = typer.Option(
    "", "--projection-transform", help="Projection/export transform id, when known."
)
_REVIEW_QUEUE_NAME_OPTION = typer.Option(..., "--queue", help="Review queue name.")
_REVIEW_QUEUE_LOOKUP_OPTION = typer.Option(..., "--queue", help="Review queue name or id.")
_ASSIGNEE_OPTION = typer.Option("", "--assignee", help="Initial queue assignee.")
_STATUS_OPTION = typer.Option("open", "--status", help="Initial queue item status.")
_MODEL_VERSION_OPTION = typer.Option("", "--model-version", help="Restrict model outputs.")
_FEEDBACK_MODEL_VERSION_OPTION = typer.Option("", "--model-version", help="Model version under evaluation.")
_TRAINING_RUN_OPTION = typer.Option("", "--training-run", help="Training/model run id.")
_EVALUATION_RUN_OPTION = typer.Option("", "--evaluation-run", help="Evaluation run id.")
_REGRESSION_THRESHOLD_OPTION = typer.Option(
    0.0, "--regression-threshold", help="Minimum negative delta before a metric is a regression."
)
_INPUT_JSON_OPTION = typer.Option(..., "--input", help="JSON metrics/regressions input file.")
_OPTIONAL_INPUT_JSON_OPTION = typer.Option(None, "--input", help="Optional JSON metrics input file.")
_PROMOTION_DECISION_OPTION = typer.Option(
    "promote", "--decision", help="Snapshot decision: promote or reject."
)
_OUTPUT_TYPE_OPTION = typer.Option("", "--output-type", help="Restrict model output type.")
_REQUIRED_LABEL_TYPE_OPTION = typer.Option(
    None, "--required-label-type", help="Required label type; repeat for several."
)
_MAX_CONFIDENCE_SCORE_OPTION = typer.Option(
    None, "--max-confidence-score", help="Only queue model outputs at or below this score."
)
_SCORER_OPTION = typer.Option(
    None,
    "--scorer",
    help="Built-in/registered scorer name (e.g. confidence-margin, entropy, "
    "ensemble-disagreement, high-loss, missing-labels, failure-severity, "
    "distribution-gap-boost). Defaults to the 0056 uncertainty heuristic.",
)
_SCORER_PARAM_OPTION = typer.Option(
    None,
    "--scorer-param",
    help="Scorer parameter as key=value (JSON-parsed value); repeat for several.",
)
_CALIBRATION_OPTION = typer.Option(
    None,
    "--calibration",
    help="Score calibration mode: confidence, probability, loss, or raw.",
)
_GAP_BY_OPTION = typer.Option(
    None, "--gap-by", help="Distribution slice dimension for gap-aware scoring; repeat for several."
)
_GAP_MIN_PER_SLICE_OPTION = typer.Option(
    1, "--gap-min-per-slice", help="Target samples per slice; deeper deficits rank higher."
)
_BENCHMARK_BASELINE_OPTION = typer.Option(
    None,
    "--baseline",
    help="Benchmark baseline strategy (random, low-confidence, diversity-only); repeat for several.",
)
_BENCHMARK_SEED_OPTION = typer.Option(0, "--seed", help="Deterministic seed for the random baseline.")
_TOOL_OPTION = typer.Option("generic", "--tool", help="External review tool name.")
_PROJECT_OPTION = typer.Option(..., "--project", help="External review tool project id.")
_CONNECTOR_STATE_OPTION = typer.Option(
    None,
    "--connector-state",
    help="JSON-file connector state path for local connector execution.",
)
_DRY_RUN_OPTION = typer.Option(False, "--dry-run", help="Plan without connector contact.")
_PLAN_ONLY_OPTION = typer.Option(False, "--plan-only", help="Emit a connector plan only.")
_REVIEW_QUEUE_PAGE_LIMIT_OPTION = typer.Option(
    None, "--page-limit", help="Maximum review queue items to emit in this page."
)
_REVIEW_QUEUE_CURSOR_OPTION = typer.Option(
    None, "--cursor", help="Cursor returned by a prior review queue page."
)
_REFRESH_INDEX_OPTION = typer.Option(False, "--refresh", help="Rebuild existing scalar indexes.")
_INDEX_STATUS_ONLY_OPTION = typer.Option(
    False, "--status-only", help="Report curation predicate-index status without building."
)
_AS_OF_OPTION = typer.Option(None, "--as-of", help="Resolve decisions as of an ISO-8601 time.")
_AS_OF_TRANSFORM_OPTION = typer.Option(
    None,
    "--as-of-transform",
    help="Resolve decisions as of a transform id; snapshot transforms use pinned table versions.",
)
_REPLAY_SNAPSHOT_OPTION = typer.Option(
    None,
    "--snapshot",
    help="Resolve decisions against a snapshot's pinned table versions.",
)
_SOURCE_SNAPSHOT_OPTION = typer.Option(
    None,
    "--source-snapshot",
    help="Optional source snapshot name for snapshot-row plan compilation.",
)
_FREEZE_ROW_PLAN_OPTION = typer.Option(
    False,
    "--freeze",
    help="Persist the compiled row plan as a lineage artifact.",
)
# 0146: row-plan catalog inspection, paging, validation, and retention controls.
_ROW_PLAN_ID_OPTION = typer.Option(
    ..., "--plan", help="Compiled row-plan id (curation-rowplan-<digest>)."
)
_OPTIONAL_ROW_PLAN_ID_OPTION = typer.Option(
    "", "--plan", help="Limit to one compiled row-plan id; omit to check every chunked plan."
)
_ROW_PLAN_LIST_GRAIN_OPTION = typer.Option(
    "",
    "--target-grain",
    help="Filter plans by grain: episode, observation, aligned-frame, or snapshot-row.",
)
_ROW_PLAN_STORAGE_KIND_OPTION = typer.Option(
    "", "--storage", help="Filter plans by target storage: inline or chunked."
)
_ROW_PLAN_STATE_OPTION = typer.Option(
    "", "--state", help="Filter plans by lifecycle state: active, superseded, or pruned."
)
_ROW_PLAN_FROZEN_OPTION = typer.Option(
    None, "--frozen/--not-frozen", help="Filter plans by whether they were frozen as a lineage artifact."
)
_ROW_PLAN_PAGE_SIZE_OPTION = typer.Option(
    None, "--page-size", help="Row-plan headers per page (bounded; default 100)."
)
_ROW_PLAN_TARGET_PAGE_SIZE_OPTION = typer.Option(
    None, "--page-size", help="Plan targets per page (bounded; default 1000)."
)
_ROW_PLAN_CURSOR_OPTION = typer.Option(
    None, "--cursor", help="Resume token from a previous page's next_cursor."
)
_ROW_PLAN_ALL_OPTION = typer.Option(
    False, "--all", help="Stream every page from --cursor forward instead of just the first."
)
_ROW_PLAN_RETAIN_LATEST_OPTION = typer.Option(
    1,
    "--retain-latest",
    help="Newest plans to keep active per (view, grain, source snapshot, base policy) series.",
)
_ROW_PLAN_OLDER_THAN_DAYS_OPTION = typer.Option(
    0,
    "--older-than-days",
    help="Prune superseded plans older than N days (0 = mark superseded, prune nothing).",
)
_ROW_PLAN_DRY_RUN_OPTION = typer.Option(
    False, "--dry-run", help="Preview the outcome without writing."
)
_ROW_PLAN_GRACE_HOURS_OPTION = typer.Option(
    6,
    "--grace-hours",
    help=(
        "Leave orphan chunk rows younger than N hours alone; they may belong to a "
        "compile that has not yet published its header."
    ),
)
_TRACE_SNAPSHOT_OPTION = typer.Option(
    ...,
    "--snapshot",
    help="Snapshot name whose membership should be explained.",
)
_TRACE_SCENARIO_ID_OPTION = typer.Option(
    ...,
    "--scenario-id",
    help="Scenario id whose snapshot membership should be explained.",
)
_SUPERSEDED_POLICY_OPTION = typer.Option(
    "latest",
    "--superseded-policy",
    help="Use latest to hide superseded decisions or history to include them.",
)
# 0142: bounded paged membership-history audit-slice filters + paging controls.
_MEMBERSHIP_HISTORY_SOURCE_OPTION = typer.Option(
    None, "--source", help="Filter by decision source; repeat for several (enables paged mode)."
)
_MEMBERSHIP_HISTORY_DECISION_OPTION = typer.Option(
    None, "--decision", help="Filter by decision; repeat for several (enables paged mode)."
)
_MEMBERSHIP_HISTORY_REVIEWER_OPTION = typer.Option(
    None, "--reviewer", help="Filter by reviewer; repeat for several (enables paged mode)."
)
_MEMBERSHIP_HISTORY_PAGE_SIZE_OPTION = typer.Option(
    None,
    "--page-size",
    help="Enable bounded paged replay with this many decision rows per page.",
)
_MEMBERSHIP_HISTORY_CURSOR_OPTION = typer.Option(
    None, "--cursor", help="Resume token from a previous page's next_cursor (paged mode)."
)
_MEMBERSHIP_HISTORY_ALL_OPTION = typer.Option(
    False, "--all", help="Stream every page of the history instead of just the first (paged mode)."
)


@curate_app.command("dedup")
def dedup(
    action: str = typer.Argument(
        "snapshot",
        help="Use plan/apply, distributed-plan/distributed-apply, or omit for legacy snapshot creation.",
    ),
    lake: str = _LAKE_OPTION,
    snapshot: str | None = _OPTIONAL_SNAPSHOT_OPTION,
    tag: str = _TAG_OPTION,
    threshold: float = _THRESHOLD_OPTION,
    embedding_column: str = _EMBEDDING_COLUMN_OPTION,
    representative_policy: list[str] = _REPRESENTATIVE_POLICY_OPTION,
    shard_by: list[str] = _SHARD_BY_OPTION,
    neighbor_limit: int = _NEIGHBOR_LIMIT_OPTION,
    max_neighbor_limit: int | None = _MAX_NEIGHBOR_LIMIT_OPTION,
    no_adaptive_neighbor_expansion: bool = _NO_ADAPTIVE_NEIGHBOR_OPTION,
    recall_audit_sample_size: int = _RECALL_AUDIT_SAMPLE_SIZE_OPTION,
    recall_audit_seed: int = _RECALL_AUDIT_SEED_OPTION,
    job_id: str | None = _DEDUP_JOB_ID_OPTION,
    max_shards: int | None = _MAX_SHARDS_OPTION,
    require_index: bool = _REQUIRE_INDEX_OPTION,
    view: str = _OPTIONAL_VIEW_OPTION,
    no_record_decisions: bool = _NO_RECORD_DECISIONS_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    coverage_tag: list[str] = _COVERAGE_TAG_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Plan/apply semantic dedup over scenario windows."""
    try:
        normalized_action = action.strip().lower().replace("_", "-")
        workbench = _workbench(lake, scenario_id=scenario_id, coverage_tag=coverage_tag)
        if normalized_action == "plan":
            plan = workbench.plan_dedup(
                near_duplicate_threshold=threshold,
                embedding_column=embedding_column,
                representative_policy=representative_policy or (),
                shard_by=shard_by or (),
                neighbor_limit=neighbor_limit,
                require_index=require_index,
            )
            _echo_dedup_plan(lake, plan)
            return
        if normalized_action == "distributed-plan":
            plan = workbench.plan_distributed_dedup(
                job_id=job_id,
                near_duplicate_threshold=threshold,
                embedding_column=embedding_column,
                representative_policy=representative_policy or (),
                shard_by=shard_by or (),
                neighbor_limit=neighbor_limit,
                max_neighbor_limit=max_neighbor_limit,
                adaptive_neighbor_limit=not no_adaptive_neighbor_expansion,
                recall_audit_sample_size=recall_audit_sample_size,
                recall_audit_seed=recall_audit_seed,
                require_index=require_index,
                max_shards=max_shards,
            )
            _echo_dedup_plan(lake, plan)
            return
        if normalized_action not in {"snapshot", "apply", "distributed-apply"}:
            raise ValueError(
                "dedup action must be plan, apply, distributed-plan, distributed-apply, or snapshot"
            )
        if normalized_action == "distributed-apply":
            selection = workbench.distributed_dedup(
                job_id=job_id,
                near_duplicate_threshold=threshold,
                embedding_column=embedding_column,
                representative_policy=representative_policy or (),
                shard_by=shard_by or (),
                neighbor_limit=neighbor_limit,
                max_neighbor_limit=max_neighbor_limit,
                adaptive_neighbor_limit=not no_adaptive_neighbor_expansion,
                recall_audit_sample_size=recall_audit_sample_size,
                recall_audit_seed=recall_audit_seed,
                require_index=require_index,
                max_shards=max_shards,
                record_decisions=not no_record_decisions,
                view_name=view,
            )
        else:
            selection = workbench.dedup(
                near_duplicate_threshold=threshold,
                embedding_column=embedding_column,
                representative_policy=representative_policy or (),
                shard_by=shard_by or (),
                neighbor_limit=neighbor_limit,
                require_index=require_index,
                record_decisions=not no_record_decisions,
                view_name=view,
            )
        if snapshot:
            context = load_lineage_context(lineage_context)
            manifest = selection.snapshot(name=snapshot, tag=tag, lineage_context=context)
            _echo_result(selection, manifest)
        else:
            if normalized_action == "snapshot":
                raise ValueError("--snapshot is required when dedup action is omitted")
            _echo_selection(lake, selection)
    except Exception as exc:
        _exit(exc)


@curate_app.command("diversity")
def diversity(
    action: str = typer.Argument(
        "sample", help="Use `sample` for greedy diversity or `optimize` for constraints."
    ),
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    limit: int = _DIVERSITY_LIMIT_OPTION,
    tag: str = _TAG_OPTION,
    method: str = _DIVERSITY_METHOD_OPTION,
    constraint: Path | None = _DIVERSITY_CONSTRAINT_OPTION,
    backend: str = _DIVERSITY_BACKEND_OPTION,
    embedding_column: str = _EMBEDDING_COLUMN_OPTION,
    threshold: float = _THRESHOLD_OPTION,
    by: list[str] = _DIVERSITY_BY_OPTION,
    min_per_slice: int = _DIVERSITY_MIN_PER_SLICE_OPTION,
    max_per_duplicate_group: int = _MAX_PER_DUPLICATE_GROUP_OPTION,
    representative_policy: list[str] = _REPRESENTATIVE_POLICY_OPTION,
    shard_by: list[str] = _SHARD_BY_OPTION,
    neighbor_limit: int = _NEIGHBOR_LIMIT_OPTION,
    require_index: bool = _REQUIRE_INDEX_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    coverage_tag: list[str] = _COVERAGE_TAG_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Sample a semantically diverse scenario snapshot."""
    try:
        normalized_action = action.strip().lower().replace("_", "-")
        if normalized_action not in {"sample", "optimize"}:
            raise ValueError("diversity action must be sample or optimize")
        workbench = _workbench(lake, scenario_id=scenario_id, coverage_tag=coverage_tag)
        if normalized_action == "optimize":
            if constraint is None:
                raise ValueError("--constraint is required for diversity optimize")
            selection = workbench.optimize_diversity(
                limit=limit,
                constraint_spec=_load_json(constraint),
                backend=backend,
                embedding_column=embedding_column,
                max_per_duplicate_group=max_per_duplicate_group,
                duplicate_threshold=threshold,
                representative_policy=representative_policy or (),
                shard_by=shard_by or (),
                neighbor_limit=neighbor_limit,
                require_index=require_index,
            )
        else:
            selection = workbench.diversity_sample(
                limit=limit,
                method=method,
                embedding_column=embedding_column,
                by=by or (),
                min_per_slice=min_per_slice,
                max_per_duplicate_group=max_per_duplicate_group,
                duplicate_threshold=threshold,
                representative_policy=representative_policy or (),
                shard_by=shard_by or (),
                neighbor_limit=neighbor_limit,
                require_index=require_index,
            )
        manifest = selection.snapshot(
            name=snapshot,
            tag=tag,
            lineage_context=load_lineage_context(lineage_context),
        )
    except Exception as exc:
        _exit(exc)
    _echo_result(selection, manifest)


@curate_app.command("stratified-sample")
def stratified_sample(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    by: list[str] = _BY_OPTION,
    per_slice: int = _PER_SLICE_OPTION,
    tag: str = _TAG_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    coverage_tag: list[str] = _COVERAGE_TAG_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Create a balanced per-slice scenario snapshot."""
    try:
        workbench = _workbench(lake, scenario_id=scenario_id, coverage_tag=coverage_tag)
        selection = workbench.stratified_sample(by=by, per_slice=per_slice)
        manifest = selection.snapshot(
            name=snapshot,
            tag=tag,
            lineage_context=load_lineage_context(lineage_context),
        )
    except Exception as exc:
        _exit(exc)
    _echo_result(selection, manifest)


@curate_app.command("mine-failures")
def mine_failures(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    seed_event: str = _SEED_EVENT_OPTION,
    seed_scenario: str = _SEED_SCENARIO_OPTION,
    seed_vector: str = _SEED_VECTOR_OPTION,
    limit: int = _LIMIT_OPTION,
    tag: str = _TAG_OPTION,
    embedding_column: str = _EMBEDDING_COLUMN_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    coverage_tag: list[str] = _COVERAGE_TAG_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Mine nearest failure/outlier neighbors into a snapshot."""
    try:
        workbench = _workbench(lake, scenario_id=scenario_id, coverage_tag=coverage_tag)
        selection = workbench.mine_failures(
            seed_event=seed_event,
            seed_scenario=seed_scenario,
            seed_embedding=seed_vector,
            limit=limit,
            embedding_column=embedding_column,
        )
        manifest = selection.snapshot(
            name=snapshot,
            tag=tag,
            lineage_context=load_lineage_context(lineage_context),
        )
    except Exception as exc:
        _exit(exc)
    _echo_result(selection, manifest)


@review_queue_app.command("failure")
def review_queue_failure(
    lake: str = _LAKE_OPTION,
    queue: str = _REVIEW_QUEUE_NAME_OPTION,
    seed_event: str = _SEED_EVENT_OPTION,
    seed_scenario: str = _SEED_SCENARIO_OPTION,
    seed_vector: str = _SEED_VECTOR_OPTION,
    limit: int = _LIMIT_OPTION,
    embedding_column: str = _EMBEDDING_COLUMN_OPTION,
    assignee: str = _ASSIGNEE_OPTION,
    status: str = _STATUS_OPTION,
    export_uri: str = _OUTPUT_URI_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    coverage_tag: list[str] = _COVERAGE_TAG_OPTION,
) -> None:
    """Seed a logical review queue from nearest failure neighbors."""
    try:
        workbench = _workbench(lake, scenario_id=scenario_id, coverage_tag=coverage_tag)
        review_queue = workbench.failure_review_queue(
            queue,
            seed_event=seed_event,
            seed_scenario=seed_scenario,
            seed_embedding=seed_vector,
            limit=limit,
            embedding_column=embedding_column,
            assignee=assignee,
            status=status,
            export_uri=export_uri,
        )
    except Exception as exc:
        _exit(exc)
    _echo_review_queue(lake, review_queue)


def _parse_scorer_params(raw: list[str] | None) -> dict[str, object]:
    """Parse ``key=value`` scorer params; values are JSON-parsed, else strings."""
    params: dict[str, object] = {}
    for item in raw or ():
        if "=" not in item:
            raise typer.BadParameter(f"--scorer-param must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"--scorer-param key must not be empty in {item!r}")
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    return params


@review_queue_app.command("active-learning")
def review_queue_active_learning(
    lake: str = _LAKE_OPTION,
    queue: str = _REVIEW_QUEUE_NAME_OPTION,
    limit: int = _LIMIT_OPTION,
    model_version: str = _MODEL_VERSION_OPTION,
    output_type: str = _OUTPUT_TYPE_OPTION,
    required_label_type: list[str] = _REQUIRED_LABEL_TYPE_OPTION,
    max_confidence_score: float = _MAX_CONFIDENCE_SCORE_OPTION,
    scorer: str = _SCORER_OPTION,
    scorer_param: list[str] = _SCORER_PARAM_OPTION,
    calibration: str = _CALIBRATION_OPTION,
    gap_by: list[str] = _GAP_BY_OPTION,
    gap_min_per_slice: int = _GAP_MIN_PER_SLICE_OPTION,
    max_per_duplicate_group: int = _MAX_PER_DUPLICATE_GROUP_OPTION,
    threshold: float = _THRESHOLD_OPTION,
    embedding_column: str = _EMBEDDING_COLUMN_OPTION,
    assignee: str = _ASSIGNEE_OPTION,
    status: str = _STATUS_OPTION,
    export_uri: str = _OUTPUT_URI_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    coverage_tag: list[str] = _COVERAGE_TAG_OPTION,
) -> None:
    """Create an active-learning queue from a pluggable scorer."""
    try:
        workbench = _workbench(lake, scenario_id=scenario_id, coverage_tag=coverage_tag)
        review_queue = workbench.active_learning_queue(
            queue,
            limit=limit,
            model_version=model_version,
            output_type=output_type,
            required_label_types=tuple(required_label_type or ()),
            max_confidence_score=max_confidence_score,
            scorer=scorer or None,
            scorer_params=_parse_scorer_params(scorer_param),
            calibration=calibration or None,
            gap_by=tuple(gap_by or ()),
            gap_min_per_slice=gap_min_per_slice,
            max_per_duplicate_group=max_per_duplicate_group,
            duplicate_threshold=threshold,
            embedding_column=embedding_column,
            assignee=assignee,
            status=status,
            export_uri=export_uri,
        )
    except Exception as exc:
        _exit(exc)
    _echo_review_queue(lake, review_queue)


@review_queue_app.command("benchmark")
def review_queue_benchmark(
    lake: str = _LAKE_OPTION,
    limit: int = _LIMIT_OPTION,
    model_version: str = _MODEL_VERSION_OPTION,
    output_type: str = _OUTPUT_TYPE_OPTION,
    required_label_type: list[str] = _REQUIRED_LABEL_TYPE_OPTION,
    scorer: str = _SCORER_OPTION,
    scorer_param: list[str] = _SCORER_PARAM_OPTION,
    calibration: str = _CALIBRATION_OPTION,
    gap_by: list[str] = _GAP_BY_OPTION,
    gap_min_per_slice: int = _GAP_MIN_PER_SLICE_OPTION,
    baseline: list[str] = _BENCHMARK_BASELINE_OPTION,
    seed: int = _BENCHMARK_SEED_OPTION,
    threshold: float = _THRESHOLD_OPTION,
    embedding_column: str = _EMBEDDING_COLUMN_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    coverage_tag: list[str] = _COVERAGE_TAG_OPTION,
) -> None:
    """Compare scored vs random/low-confidence/diversity-only queue selection."""
    try:
        workbench = _workbench(lake, scenario_id=scenario_id, coverage_tag=coverage_tag)
        baselines = tuple(baseline) if baseline else ("random", "low-confidence", "diversity-only")
        report = workbench.evaluate_active_learning_selection(
            limit=limit,
            model_version=model_version,
            output_type=output_type,
            required_label_types=tuple(required_label_type or ()),
            scorer=scorer or None,
            scorer_params=_parse_scorer_params(scorer_param),
            calibration=calibration or None,
            gap_by=tuple(gap_by or ()),
            gap_min_per_slice=gap_min_per_slice,
            baselines=baselines,
            seed=seed,
            duplicate_threshold=threshold,
            embedding_column=embedding_column,
        )
    except Exception as exc:
        _exit(exc)
    typer.echo(json.dumps(report, sort_keys=True))


@review_queue_app.command("export")
def review_queue_export(
    lake: str = _LAKE_OPTION,
    queue: str = _REVIEW_QUEUE_LOOKUP_OPTION,
    tool: str = _TOOL_OPTION,
    output_uri: str = _OUTPUT_URI_OPTION,
    page_limit: int | None = _REVIEW_QUEUE_PAGE_LIMIT_OPTION,
    cursor: str | None = _REVIEW_QUEUE_CURSOR_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Emit a logical external-tool queue manifest with lineage."""
    try:
        from lancedb_robotics.lake import Lake

        manifest = Lake.open(lake).curate.queue(queue).export_manifest(
            tool=tool,
            output_uri=output_uri,
            limit=page_limit,
            cursor=cursor,
            lineage_context=load_lineage_context(lineage_context),
        )
    except Exception as exc:
        _exit(exc)
    typer.echo(json.dumps(manifest, sort_keys=True))


@review_queue_app.command("connector-export")
def review_queue_connector_export(
    lake: str = _LAKE_OPTION,
    queue: str = _REVIEW_QUEUE_LOOKUP_OPTION,
    tool: str = _TOOL_OPTION,
    project_id: str = _PROJECT_OPTION,
    connector_state: Path | None = _CONNECTOR_STATE_OPTION,
    output_uri: str = _OUTPUT_URI_OPTION,
    page_limit: int | None = _REVIEW_QUEUE_PAGE_LIMIT_OPTION,
    cursor: str | None = _REVIEW_QUEUE_CURSOR_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
    plan_only: bool = _PLAN_ONLY_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Create/upsert external review tasks through a connector."""
    try:
        from lancedb_robotics.lake import Lake

        connector = _review_connector(tool, connector_state, dry_run=dry_run or plan_only)
        report = (
            Lake.open(lake)
            .curate.queue(queue)
            .export_to_connector(
                connector,
                tool=tool,
                project_id=project_id,
                output_uri=output_uri,
                limit=page_limit,
                cursor=cursor,
                dry_run=dry_run,
                plan_only=plan_only,
                lineage_context=load_lineage_context(lineage_context),
            )
        )
    except Exception as exc:
        _exit(exc)
    typer.echo(json.dumps(report.to_dict(), sort_keys=True))


@review_queue_app.command("sync-status")
def review_queue_sync_status(
    lake: str = _LAKE_OPTION,
    queue: str = _REVIEW_QUEUE_LOOKUP_OPTION,
    tool: str = _TOOL_OPTION,
    project_id: str = _PROJECT_OPTION,
    connector_state: Path | None = _CONNECTOR_STATE_OPTION,
    page_limit: int | None = _REVIEW_QUEUE_PAGE_LIMIT_OPTION,
    cursor: str | None = _REVIEW_QUEUE_CURSOR_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Sync external review task status back to queue rows."""
    try:
        from lancedb_robotics.lake import Lake

        connector = _review_connector(tool, connector_state, dry_run=dry_run)
        report = (
            Lake.open(lake)
            .curate.queue(queue)
            .sync_connector_status(
                connector,
                tool=tool,
                project_id=project_id,
                limit=page_limit,
                cursor=cursor,
                dry_run=dry_run,
                lineage_context=load_lineage_context(lineage_context),
            )
        )
    except Exception as exc:
        _exit(exc)
    typer.echo(json.dumps(report.to_dict(), sort_keys=True))


@review_queue_app.command("connector-import")
def review_queue_connector_import(
    lake: str = _LAKE_OPTION,
    queue: str = _REVIEW_QUEUE_LOOKUP_OPTION,
    tool: str = _TOOL_OPTION,
    project_id: str = _PROJECT_OPTION,
    connector_state: Path | None = _CONNECTOR_STATE_OPTION,
    view_name: str | None = _OPTIONAL_VIEW_OPTION,
    page_limit: int | None = _REVIEW_QUEUE_PAGE_LIMIT_OPTION,
    cursor: str | None = _REVIEW_QUEUE_CURSOR_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Import reviewed connector outcomes as labels, feedback, and decisions."""
    try:
        from lancedb_robotics.lake import Lake

        connector = _review_connector(tool, connector_state, dry_run=dry_run)
        report = (
            Lake.open(lake)
            .curate.queue(queue)
            .import_connector_outcomes(
                connector,
                tool=tool,
                project_id=project_id,
                view_name=view_name,
                limit=page_limit,
                cursor=cursor,
                dry_run=dry_run,
                lineage_context=load_lineage_context(lineage_context),
            )
        )
    except Exception as exc:
        _exit(exc)
    typer.echo(json.dumps(report.to_dict(), sort_keys=True))


@review_queue_app.command("summary")
def review_queue_summary(
    lake: str = _LAKE_OPTION,
    queue: str = _REVIEW_QUEUE_LOOKUP_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Summarize review queue counts without exporting every item."""
    try:
        from lancedb_robotics.lake import Lake

        summary = Lake.open(lake).curate.queue(queue).summary()
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(summary, sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"queue: {summary['queue_id']}")
    typer.echo(f"items: {summary['item_count']}")
    for group in (
        "counts_by_status",
        "counts_by_assignee",
        "counts_by_source_operation",
        "counts_by_priority_band",
    ):
        values = summary[group]
        rendered = ", ".join(f"{key}={value}" for key, value in values.items())
        typer.echo(f"{group}: {rendered}")


@review_queue_app.command("index-predicates")
def review_queue_index_predicates(
    lake: str = _LAKE_OPTION,
    refresh: bool = _REFRESH_INDEX_OPTION,
    status_only: bool = _INDEX_STATUS_ONLY_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Build or inspect scalar indexes for review queue hot paths."""
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        results = (
            opened.curate.review_queue_predicate_index_status()
            if status_only
            else opened.curate.index_review_queue_predicates(refresh=refresh)
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(list(results), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    for result in results:
        reason = f" ({result['reason']})" if result.get("reason") else ""
        typer.echo(
            f"{result['table']}.{result['column']}: "
            f"{result['status']} [{result['predicate_role']}]{reason}"
        )


@curate_app.command("quality-filter")
def quality_filter(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    min_score: float = _MIN_SCORE_OPTION,
    score_column: str = _SCORE_COLUMN_OPTION,
    include_flag: list[str] = _INCLUDE_FLAG_OPTION,
    exclude_flag: list[str] = _EXCLUDE_FLAG_OPTION,
    tag: str = _TAG_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    coverage_tag: list[str] = _COVERAGE_TAG_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Filter scenarios by quality flags and optional quality score."""
    try:
        workbench = _workbench(lake, scenario_id=scenario_id, coverage_tag=coverage_tag)
        selection = workbench.filter_quality(
            min_score=min_score,
            score_column=score_column,
            include_flags=include_flag or (),
            exclude_flags=tuple(exclude_flag) if exclude_flag is not None else ("quarantined",),
        )
        manifest = selection.snapshot(
            name=snapshot,
            tag=tag,
            lineage_context=load_lineage_context(lineage_context),
        )
    except Exception as exc:
        _exit(exc)
    _echo_result(selection, manifest)


@curate_app.command("save-view")
def save_view(
    lake: str = _LAKE_OPTION,
    view: str = _VIEW_OPTION,
    owner: str = _OWNER_OPTION,
    view_tag: list[str] = _VIEW_TAG_OPTION,
    description: str = _DESCRIPTION_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    coverage_tag: list[str] = _COVERAGE_TAG_OPTION,
) -> None:
    """Persist a logical curation view without copying payload rows."""
    try:
        selection = _workbench(lake, scenario_id=scenario_id, coverage_tag=coverage_tag)
        saved = selection.save_view(view, owner=owner, tags=tuple(view_tag or ()), description=description)
    except Exception as exc:
        _exit(exc)
    typer.echo(f"lake: {lake}")
    typer.echo(f"view: {saved.name} ({saved.view_id})")
    typer.echo(f"owner: {saved.owner}")
    if saved.tags:
        typer.echo(f"tags: {','.join(saved.tags)}")
    typer.echo(f"scenarios: {len(saved.scenario_ids)}")
    typer.echo(f"membership_storage: {saved.membership_storage}")
    typer.echo(f"transform: {saved.transform_id}")


@curate_app.command("index-predicates")
def index_predicates(
    lake: str = _LAKE_OPTION,
    refresh: bool = _REFRESH_INDEX_OPTION,
    status_only: bool = _INDEX_STATUS_ONLY_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Build or inspect scalar indexes for curation saved-view hot paths."""
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        results = (
            opened.curate.predicate_index_status()
            if status_only
            else opened.curate.index_predicates(refresh=refresh)
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(list(results), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    for result in results:
        reason = f" ({result['reason']})" if result.get("reason") else ""
        typer.echo(
            f"{result['table']}.{result['column']}: "
            f"{result['status']} [{result['predicate_role']}]{reason}"
        )


@curate_app.command("migrate-views")
def migrate_views(
    lake: str = _LAKE_OPTION,
    view: str | None = _OPTIONAL_VIEW_OPTION,
    inline_limit: int = typer.Option(
        1024, "--inline-limit", help="Inline scenario-id count above which a view is chunked."
    ),
    chunk_size: int = typer.Option(
        1024, "--chunk-size", help="Scenario ids stored per membership chunk row."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan the migration without writing."),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Migrate legacy inline saved views into chunked membership storage."""
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        report = opened.curate.migrate_view_membership(
            view_name=view,
            inline_scenario_limit=inline_limit,
            chunk_size=chunk_size,
            dry_run=dry_run,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.to_params(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"dry_run: {report.dry_run}")
    typer.echo(f"migrated: {report.migrated}")
    for result in report.results:
        typer.echo(
            f"{result.view_name} ({result.view_id}): {result.status} "
            f"scenarios={result.scenario_count} chunks={result.chunk_count}"
        )


@curate_app.command("validate-chunks")
def validate_chunks(
    lake: str = _LAKE_OPTION,
    view: str | None = _OPTIONAL_VIEW_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Validate chunk integrity for chunked saved views and report repairs."""
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        report = opened.curate.validate_view_membership(view_name=view)
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.to_params(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"status: {report.status}")
    typer.echo(f"chunked_views: {report.chunked_views}")
    typer.echo(f"healthy_views: {report.healthy_views}")
    if report.orphan_view_ids:
        typer.echo(f"orphan_view_ids: {','.join(report.orphan_view_ids)}")
    for issue in report.issues:
        typer.echo(f"[{issue.code}] {issue.view_name} ({issue.view_id}): {issue.detail}")
        typer.echo(f"  repair: {issue.repair}")


@curate_app.command("compact-chunks")
def compact_chunks(
    lake: str = _LAKE_OPTION,
    include_superseded: bool = typer.Option(
        False,
        "--include-superseded",
        help="Also reclaim chunks for non-latest revisions of live view names.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="List candidates without deleting."),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Reclaim orphaned (and opt-in superseded) view membership chunk rows."""
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        report = opened.curate.compact_view_membership(
            include_superseded=include_superseded,
            dry_run=dry_run,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.to_params(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"dry_run: {report.dry_run}")
    typer.echo(f"orphan_chunks_removed: {report.orphan_chunks_removed}")
    typer.echo(f"superseded_chunks_removed: {report.superseded_chunks_removed}")
    if report.orphan_view_ids:
        typer.echo(f"orphan_view_ids: {','.join(report.orphan_view_ids)}")
    if report.superseded_view_ids:
        typer.echo(f"superseded_view_ids: {','.join(report.superseded_view_ids)}")
    if report.retained_snapshot_pinned_view_ids:
        typer.echo(
            "retained_snapshot_pinned_view_ids: "
            + ",".join(report.retained_snapshot_pinned_view_ids)
        )


@curate_app.command("view-membership")
def view_membership(
    lake: str = _LAKE_OPTION,
    view: str = _VIEW_OPTION,
    page_size: int = typer.Option(
        1000, "--page-size", help="Scenario ids per page (bounded lazy read)."
    ),
    cursor: str | None = typer.Option(
        None, "--cursor", help="Resume token from a previous page's next_cursor."
    ),
    all_pages: bool = typer.Option(
        False, "--all", help="Stream every page instead of just the first."
    ),
    show_ids: bool = typer.Option(
        False, "--show-ids", help="Print the scenario ids in each page."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Page a saved view's membership lazily and report scale diagnostics.

    Reads the chunk store by bounded ordinal pages without materializing the whole
    membership; ``--cursor`` resumes a prior page, ``--all`` streams to the end.
    """
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        membership = opened.curate.view_membership(view, page_size=page_size)
        diagnostics = membership.diagnostics()
        pages = (
            list(membership.iter_pages(cursor=cursor))
            if all_pages
            else [membership.page(cursor=cursor)]
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "diagnostics": diagnostics,
                    "pages": [page.to_dict() for page in pages],
                },
                sort_keys=True,
            )
        )
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"view: {membership.view_name} ({membership.view_id})")
    typer.echo(f"storage: {diagnostics['storage_kind']} lazy={diagnostics['lazy']}")
    typer.echo(
        f"row_count: {diagnostics['row_count']} "
        f"chunk_count: {diagnostics['chunk_count']} "
        f"chunk_size: {diagnostics['chunk_size']}"
    )
    typer.echo(
        f"page_size: {diagnostics['page_size']} "
        f"estimated_membership_bytes: {diagnostics['estimated_membership_bytes']} "
        f"materialize_recommended: {diagnostics['materialize_recommended']}"
    )
    for page in pages:
        typer.echo(
            f"page[{page.start_ordinal}:{page.end_ordinal}] "
            f"count={len(page.scenario_ids)} has_more={page.has_more} "
            f"next_cursor={page.next_cursor or '-'}"
        )
        if show_ids:
            for scenario_id in page.scenario_ids:
                typer.echo(f"  {scenario_id}")


@curate_app.command("decide")
def decide(
    lake: str = _LAKE_OPTION,
    view: str = _VIEW_OPTION,
    decision: str = _DECISION_OPTION,
    target_grain: str = _TARGET_GRAIN_OPTION,
    target_id: list[str] = _TARGET_ID_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    reason: str = _REASON_OPTION,
    reason_code: str = _REASON_CODE_OPTION,
    note: str = _NOTE_OPTION,
    reviewer: str = _REVIEWER_OPTION,
    queue: str = _QUEUE_OPTION,
    priority: int = _PRIORITY_OPTION,
    score: float = _SCORE_OPTION,
    source: str = _SOURCE_OPTION,
) -> None:
    """Record saved-view membership decisions."""
    try:
        selection = _selection_for_view_or_scope(lake, view=view, scenario_id=scenario_id)
        decisions = selection.record_decisions(
            decision=decision,
            target_grain=target_grain,
            target_ids=target_id or None,
            scenario_ids=scenario_id or None,
            view_name=view,
            reason=reason,
            reason_code=reason_code,
            note=note,
            reviewer=reviewer,
            queue=queue,
            priority=priority,
            score=score,
            source=source,
        )
    except Exception as exc:
        _exit(exc)
    typer.echo(f"lake: {lake}")
    typer.echo(f"view: {decisions.view.name} ({decisions.view.view_id})")
    typer.echo(f"target_grain: {decisions.target_grain}")
    typer.echo(f"decision: {decisions.decision}")
    typer.echo(f"targets: {len(decisions.target_ids)}")
    typer.echo(f"scenarios: {len(decisions.scenario_ids)}")
    typer.echo(f"transform: {decisions.transform_id}")


@curate_app.command("membership-history")
def membership_history(
    lake: str = _LAKE_OPTION,
    view: str = _OPTIONAL_VIEW_OPTION,
    target_grain: str = _TARGET_GRAIN_OPTION,
    target_id: list[str] = _TARGET_ID_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    as_of: str = _AS_OF_OPTION,
    as_of_transform: str = _AS_OF_TRANSFORM_OPTION,
    snapshot: str = _REPLAY_SNAPSHOT_OPTION,
    superseded_policy: str = _SUPERSEDED_POLICY_OPTION,
    source: list[str] = _MEMBERSHIP_HISTORY_SOURCE_OPTION,
    decision: list[str] = _MEMBERSHIP_HISTORY_DECISION_OPTION,
    reviewer: list[str] = _MEMBERSHIP_HISTORY_REVIEWER_OPTION,
    page_size: int | None = _MEMBERSHIP_HISTORY_PAGE_SIZE_OPTION,
    cursor: str | None = _MEMBERSHIP_HISTORY_CURSOR_OPTION,
    all_pages: bool = _MEMBERSHIP_HISTORY_ALL_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Inspect saved-view membership decisions with as-of replay semantics.

    Without any paging flag this prints the in-memory resolution (small-history
    convenience, backlog 0082). Passing ``--page-size``/``--cursor``/``--all`` or
    a ``--source``/``--decision``/``--reviewer`` filter switches to the bounded,
    resumable audit replay (backlog 0142): scope predicates are pushed into Lance
    and rows are paged deterministically by ``(created_at, membership_id)`` in
    O(page_size) client memory without materializing unrelated targets. Scope the
    replay (``--view``/``--target-id``/filters) for the bounded-scan fast path;
    deep paging over a broad scope re-scans the prefix per page (follow-up 0475).
    """
    sources = tuple(source or ())
    decisions = tuple(decision or ())
    reviewers = tuple(reviewer or ())
    paged = (
        page_size is not None
        or bool(cursor)
        or all_pages
        or bool(sources)
        or bool(decisions)
        or bool(reviewers)
    )
    try:
        from lancedb_robotics.lake import Lake

        target_ids = tuple(target_id or ())
        scenario_ids = tuple(scenario_id or ())
        if not target_ids and target_grain == "scenario":
            target_ids = scenario_ids
            scenario_ids = ()
        opened = Lake.open(lake)
        if paged:
            history = opened.curate.resolve_membership_pages(
                view_name=view,
                target_grain=target_grain if target_ids else None,
                target_ids=target_ids,
                scenario_ids=scenario_ids,
                as_of=as_of,
                transform_id=as_of_transform,
                snapshot_name=snapshot,
                sources=sources,
                decisions=decisions,
                reviewers=reviewers,
                page_size=page_size,
            )
            manifest = history.manifest()
            pages = (
                list(history.iter_pages(cursor=cursor))
                if all_pages
                else [history.page(cursor=cursor)]
            )
        else:
            resolution = opened.curate.resolve_membership(
                view_name=view,
                target_grain=target_grain if target_ids else None,
                target_ids=target_ids,
                scenario_ids=scenario_ids,
                as_of=as_of,
                transform_id=as_of_transform,
                snapshot_name=snapshot,
                superseded_policy=superseded_policy,
            )
    except Exception as exc:
        _exit(exc)
    if paged:
        if json_output:
            typer.echo(
                json.dumps(
                    {"manifest": manifest, "pages": [page.to_dict() for page in pages]},
                    sort_keys=True,
                    default=str,
                )
            )
            return
        typer.echo(f"lake: {lake}")
        scope = manifest["scope"]
        if manifest["view"]:
            typer.echo(f"view: {manifest['view']['name']} ({manifest['view']['view_id']})")
        typer.echo(f"target_grain: {scope['target_grain']}")
        typer.echo(f"as_of: {scope['as_of'] or '-'}")
        typer.echo(f"page_size: {manifest['page_size']} ordering: {manifest['ordering']}")
        for page in pages:
            typer.echo(
                f"page[{page.page_index}] count={len(page.records)} "
                f"has_more={page.has_more} next_cursor={page.next_cursor or '-'}"
            )
            for row in page.records:
                supersedes = (
                    f" supersedes={row['supersedes_membership_id']}"
                    if row["supersedes_membership_id"]
                    else ""
                )
                typer.echo(
                    f"  {row['created_at']} {row['target_grain']}:{row['target_id']} "
                    f"{row['decision']} source={row['source']} reviewer={row['reviewer']}"
                    f"{supersedes}"
                )
        return
    if json_output:
        typer.echo(json.dumps(resolution.report, sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    if resolution.view:
        typer.echo(f"view: {resolution.view.name} ({resolution.view.view_id})")
    typer.echo(f"target_grain: {resolution.target_grain}")
    typer.echo(f"latest: {len(resolution.latest_decisions)}")
    typer.echo(f"history: {len(resolution.membership_history)}")
    typer.echo(f"superseded: {len(resolution.superseded_decisions)}")
    for row in resolution.membership_history:
        supersedes = (
            f" supersedes={row['supersedes_membership_id']}"
            if row["supersedes_membership_id"]
            else ""
        )
        typer.echo(
            f"{row['created_at']} {row['target_grain']}:{row['target_id']} "
            f"{row['decision']} source={row['source']}{supersedes}"
        )


@curate_app.command("trace-membership")
def trace_membership(
    lake: str = _LAKE_OPTION,
    snapshot: str = _TRACE_SNAPSHOT_OPTION,
    scenario_id: str = _TRACE_SCENARIO_ID_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Explain why a scenario is included in or excluded from a snapshot."""
    try:
        from lancedb_robotics.lake import Lake

        trace = Lake.open(lake).curate.trace_membership(snapshot, scenario_id)
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(trace.report, sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"snapshot: {trace.snapshot_name} ({trace.dataset_id})")
    typer.echo(f"scenario: {trace.scenario_id}")
    typer.echo(f"final_result: {trace.final_result}")
    typer.echo(f"included_in_snapshot: {trace.included_in_snapshot}")
    typer.echo(f"history: {len(trace.resolution.membership_history)}")
    typer.echo(f"transforms: {len(trace.report['transforms'])}")


_REPLAY_READINESS_SNAPSHOT_OPTION = typer.Option(
    None,
    "--snapshot",
    help="Scope readiness to one dataset snapshot (default: all active snapshots).",
)


@curate_app.command("replay-readiness")
def replay_readiness(
    lake: str = _LAKE_OPTION,
    snapshot: str | None = _REPLAY_READINESS_SNAPSHOT_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Report whether snapshot-pinned curation versions are still replayable (0143)."""
    try:
        from lancedb_robotics.lake import Lake

        report = Lake.open(lake).curate.replay_readiness(snapshot_name=snapshot)
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {report.lake_uri}")
    typer.echo(f"backend: {report.backend.backend_kind} ({report.backend.status})")
    typer.echo(f"status: {report.status}")
    typer.echo(f"ready: {report.ready}")
    typer.echo(f"snapshots_checked: {report.snapshots_checked}")
    typer.echo(f"pins: {len(report.pins)}")
    for pin in report.pins:
        if pin.status != "protected":
            snapshots = ", ".join(pin.snapshots)
            typer.echo(f"  {pin.table}@{pin.version}: {pin.status} ({snapshots})")
    for action in report.suggested_actions:
        typer.echo(f"  action: {action}")


@curate_app.command("compile-row-plan")
def compile_row_plan(
    lake: str = _LAKE_OPTION,
    view: str = _VIEW_OPTION,
    target_grain: str = _ROW_PLAN_TARGET_GRAIN_OPTION,
    source_snapshot: str = _SOURCE_SNAPSHOT_OPTION,
    freeze: bool = _FREEZE_ROW_PLAN_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Compile saved-view decisions into a row-grain training plan.

    The plan is persisted to the ``curation_row_plans`` catalog; above the inline
    threshold its ordered targets live in bounded ``curation_row_plan_chunks``
    rows (backlog 0146). Inspect them with ``curate row-plan-targets``.
    """
    try:
        from lancedb_robotics.lake import Lake

        plan = Lake.open(lake).curate.compile_row_plan(
            view_name=view,
            target_grain=target_grain,
            source_snapshot_name=source_snapshot,
            freeze=freeze,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(plan.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"view: {plan.view.name} ({plan.view.view_id})")
    typer.echo(f"row_plan: {plan.plan_id}")
    typer.echo(f"target_grain: {plan.target_grain}")
    # Counts come from the plan header, not the bounded report samples.
    typer.echo(f"rows: {plan.target_count}")
    typer.echo(f"scenarios: {len(plan.scenario_ids)}")
    typer.echo(f"conflicts: {plan.report['conflict_count']}")
    typer.echo(f"storage: {plan.storage_kind}")
    if plan.storage_kind == "chunked":
        typer.echo(f"chunks: {plan.storage.get('chunk_count', 0)}")
    if plan.artifact_id:
        typer.echo(f"artifact: {plan.artifact_id}")
    typer.echo(f"transform: {plan.transform_id}")


@curate_app.command("row-plans")
def row_plans(
    lake: str = _LAKE_OPTION,
    view: str = _OPTIONAL_VIEW_OPTION,
    target_grain: str = _ROW_PLAN_LIST_GRAIN_OPTION,
    source_snapshot: str = _SOURCE_SNAPSHOT_OPTION,
    storage_kind: str = _ROW_PLAN_STORAGE_KIND_OPTION,
    state: str = _ROW_PLAN_STATE_OPTION,
    frozen: bool | None = _ROW_PLAN_FROZEN_OPTION,
    page_size: int | None = _ROW_PLAN_PAGE_SIZE_OPTION,
    cursor: str | None = _ROW_PLAN_CURSOR_OPTION,
    all_pages: bool = _ROW_PLAN_ALL_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """List compiled row plans, newest key last, in bounded pages (backlog 0146).

    Filters push down to indexed header columns and rows are ordered by
    ``(created_at, plan_id)``; client memory is bounded to one page regardless of
    catalog size. ``--all`` streams every page from ``--cursor`` forward.

    Pages are emitted as they arrive. In ``--json`` mode that is one JSON object per
    line (JSONL): a header line, then one line per page, then a trailing summary.
    """
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        if json_output:
            typer.echo(json.dumps({"lake": lake}, sort_keys=True))
        else:
            typer.echo(f"lake: {lake}")

        def _fetch(token: str | None) -> Any:
            return opened.curate.row_plans(
                view_name=view or None,
                target_grain=target_grain or None,
                source_snapshot_name=source_snapshot or None,
                storage_kind=storage_kind or None,
                state=state or None,
                frozen=frozen,
                page_size=page_size,
                cursor=token,
            )

        def _render(page: Any) -> None:
            for record in page.records:
                typer.echo(
                    f"{record.plan_id}  grain={record.target_grain} "
                    f"view={record.view_name} targets={record.target_count} "
                    f"storage={record.storage_kind} state={record.state} "
                    f"frozen={'yes' if record.frozen else 'no'}"
                )

        _stream_row_plan_pages(
            _cursor_pages(_fetch, cursor=cursor, all_pages=all_pages),
            _render,
            json_output=json_output,
        )
    except Exception as exc:
        _exit(exc)

def _cursor_pages(
    fetch: Callable[[str | None], Any],
    *,
    cursor: str | None,
    all_pages: bool,
) -> Iterator[Any]:
    """Walk pages by re-fetching from each ``next_cursor``."""
    token = cursor
    while True:
        page = fetch(token)
        yield page
        if not all_pages or not page.has_more or not page.next_cursor:
            return
        token = page.next_cursor


def _stream_row_plan_pages(
    pages: Iterable[Any],
    render: Callable[[Any], None],
    *,
    json_output: bool,
) -> None:
    """Emit paged output page-by-page, then a trailing summary (backlog 0146).

    Shared by ``row-plans`` and ``row-plan-targets``: both stream rather than
    accumulate, because holding every page to serialize at the end would put the
    whole catalog (or the whole plan) in client memory and negate the paging. In
    ``--json`` mode each page is one JSONL line, followed by a summary line.

    Takes an *iterable of pages* rather than a fetch callback so a caller can supply
    a single-pass generator where re-fetching would be expensive -- see
    ``CurationRowPlanTargets.iter_pages``.
    """
    page_count = 0
    last = None
    for page in pages:
        page_count += 1
        last = page
        if json_output:
            typer.echo(json.dumps(page.to_dict(), sort_keys=True))
        else:
            render(page)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "page_count": page_count,
                    "has_more": last.has_more,
                    "next_cursor": last.next_cursor or "",
                },
                sort_keys=True,
            )
        )
        return
    typer.echo(f"pages: {page_count}")
    typer.echo(f"has_more: {last.has_more}")
    if last.next_cursor:
        typer.echo(f"next_cursor: {last.next_cursor}")


@curate_app.command("row-plan")
def row_plan(
    lake: str = _LAKE_OPTION,
    plan: str = _ROW_PLAN_ID_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Show one compiled row plan's header plus its bounded compile summary.

    Conflict / rejected / label-intent samples are capped with explicit
    ``*_truncated`` flags; the counts alongside them are exact (backlog 0146).
    """
    try:
        from lancedb_robotics.lake import Lake

        summary = Lake.open(lake).curate.row_plan_summary(plan)
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(summary, sort_keys=True, default=str))
        return
    typer.echo(f"lake: {lake}")
    for key in (
        "plan_id",
        "view_name",
        "target_grain",
        "target_table",
        "storage_kind",
        "target_count",
        "chunk_count",
        "candidate_count",
        "rejected_count",
        "conflict_count",
        "label_intent_count",
        "state",
        "frozen",
        "artifact_id",
        "transform_id",
    ):
        typer.echo(f"{key}: {summary.get(key)}")
    body = summary.get("summary") or {}
    if body.get("conflicts_truncated") or body.get("rejected_truncated"):
        typer.echo("samples: truncated (see counts above for exact totals)")


@curate_app.command("row-plan-targets")
def row_plan_targets(
    lake: str = _LAKE_OPTION,
    plan: str = _ROW_PLAN_ID_OPTION,
    page_size: int | None = _ROW_PLAN_TARGET_PAGE_SIZE_OPTION,
    cursor: str | None = _ROW_PLAN_CURSOR_OPTION,
    all_pages: bool = _ROW_PLAN_ALL_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Page a compiled plan's targets in stable ordinal order (backlog 0146).

    Reads bounded windows of ``curation_row_plan_chunks``, so a multi-million
    target plan pages in O(page_size) client memory. Each target carries its
    resolved Lance row id for a ``take_row_ids`` handoff. ``--all`` streams every
    page from ``--cursor`` forward.

    Output is emitted page by page as each page arrives -- accumulating them to
    serialize at the end would hold the whole plan in client memory and negate the
    paging. In ``--json`` mode that means one JSON object per line (JSONL): the
    first line is the plan handle, then one line per page.
    """
    try:
        from lancedb_robotics.lake import Lake

        handle = Lake.open(lake).curate.row_plan_targets(plan, page_size=page_size)
        if json_output:
            typer.echo(json.dumps({"lake": lake, "plan": handle.to_dict()}, sort_keys=True))
        else:
            typer.echo(f"lake: {lake}")
            typer.echo(f"row_plan: {plan}")
            typer.echo(f"storage: {handle.storage_kind}")
            typer.echo(f"target_count: {handle.target_count}")
            if not handle.row_ids_trustworthy:
                typer.echo(
                    "warning: lance_row_ids may have been remapped by compaction "
                    "since this plan was compiled; resolve by target_id"
                )
        def _render(page: Any) -> None:
            for target in page.targets:
                typer.echo(f"{target.ordinal}\t{target.target_id}\t{target.lance_row_id}")

        # ``iter_pages`` walks the chunk table in ONE pass; re-fetching per page
        # would re-read and re-sort a whole ordinal window each time.
        pages = (
            handle.iter_pages(page_size=page_size, cursor=cursor)
            if all_pages
            else [handle.page(page_size=page_size, cursor=cursor)]
        )
        _stream_row_plan_pages(pages, _render, json_output=json_output)
    except Exception as exc:
        _exit(exc)


@curate_app.command("validate-row-plans")
def validate_row_plans(
    lake: str = _LAKE_OPTION,
    plan: str = _OPTIONAL_ROW_PLAN_ID_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Check chunked row-plan storage against its header (backlog 0146).

    Recomputes each plan's target-id digest, chunk count, ordinal contiguity, and
    per-chunk bounds through the bounded windowed reader, and reports orphan chunk
    rows left behind by an interrupted write.
    """
    try:
        from lancedb_robotics.lake import Lake

        report = Lake.open(lake).curate.validate_row_plans(plan_id=plan or None)
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"scanned_plans: {len(report.scanned_plan_ids)}")
    typer.echo(f"issues: {len(report.issues)}")
    for issue in report.issues:
        typer.echo(f"  {issue.plan_id} {issue.code}: {issue.detail}")
    typer.echo(f"orphan_chunk_plans: {len(report.orphan_chunk_plan_ids)}")
    typer.echo(f"ok: {report.ok}")


@curate_app.command("compact-row-plans")
def compact_row_plans(
    lake: str = _LAKE_OPTION,
    grace_hours: int = _ROW_PLAN_GRACE_HOURS_OPTION,
    dry_run: bool = _ROW_PLAN_DRY_RUN_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Reclaim orphan row-plan chunk rows whose header never landed (0146).

    Chunks younger than ``--grace-hours`` are reported as in-flight and left
    alone: a compile that has written its chunks but not yet published its header
    looks exactly like a crashed one, and a large compile stays in that state for
    minutes.
    """
    try:
        from lancedb_robotics.lake import Lake

        grace = timedelta(hours=grace_hours) if grace_hours >= 0 else None
        report = Lake.open(lake).curate.compact_row_plans(dry_run=dry_run, grace=grace)
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report, sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    for key in (
        "dry_run",
        "grace_seconds",
        "orphan_plan_count",
        "skipped_in_flight_count",
        "chunks_deleted",
    ):
        typer.echo(f"{key}: {report.get(key)}")
    if report.get("delete_failed_plan_ids"):
        typer.echo(f"delete_failed: {len(report['delete_failed_plan_ids'])}")


@curate_app.command("prune-row-plans")
def prune_row_plans(
    lake: str = _LAKE_OPTION,
    retain_latest: int = _ROW_PLAN_RETAIN_LATEST_OPTION,
    older_than_days: int = _ROW_PLAN_OLDER_THAN_DAYS_OPTION,
    dry_run: bool = _ROW_PLAN_DRY_RUN_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Compact superseded, unreferenced compiled row plans (backlog 0146).

    Keeps the newest ``--retain-latest`` plans per
    ``(view, grain, source snapshot, base policy)`` series active. Frozen plans and
    plans referenced by a training run or training report are protected and never
    pruned. Without ``--older-than-days`` nothing is deleted (safe soft-retire).
    """
    try:
        from lancedb_robotics.lake import Lake

        cutoff = timedelta(days=older_than_days) if older_than_days > 0 else None
        report = Lake.open(lake).curate.prune_row_plans(
            retain_latest=retain_latest,
            older_than=cutoff,
            dry_run=dry_run,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    for key, value in (
        ("dry_run", report.dry_run),
        ("retain_latest", report.retain_latest),
        ("scanned", report.scanned_count),
        ("superseded", report.superseded_count),
        ("pruned", report.pruned_count),
        ("protected", report.protected_count),
        ("chunks_deleted", report.chunks_deleted),
    ):
        typer.echo(f"{key}: {value}")


@curate_app.command("compose-snapshot")
def compose_snapshot(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    view: str = _OPTIONAL_VIEW_OPTION,
    by: list[str] = _GAP_BY_OPTION,
    min_per_slice: int = _MIN_PER_SLICE_OPTION,
    tag: str = _TAG_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    coverage_tag: list[str] = _COVERAGE_TAG_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Apply decisions and freeze a curated training snapshot."""
    try:
        if view:
            from lancedb_robotics.lake import Lake

            selection = Lake.open(lake).curate.workbench(scope=view, apply_decisions=True)
        else:
            selection = _workbench(lake, scenario_id=scenario_id, coverage_tag=coverage_tag)
            selection = selection.apply_decisions()
        if by:
            selection = selection.distribution_gap(by=by, min_per_slice=min_per_slice)
        manifest = selection.snapshot(
            name=snapshot,
            tag=tag,
            lineage_context=load_lineage_context(lineage_context),
        )
    except Exception as exc:
        _exit(exc)
    _echo_result(selection, manifest)


_PREVIEW_LIMIT_OPTION = typer.Option(
    None,
    "--preview-limit",
    help="Cap inline membership/label id previews (counts/deltas stay complete).",
)
_BATCH_SIZE_OPTION = typer.Option(
    None, "--batch-size", help="Streaming scan batch size for bounded-memory execution."
)
_LOCAL_ROW_BUDGET_OPTION = typer.Option(
    None,
    "--local-row-budget",
    help="Max estimated rows a metric may scan locally before the plan flags an executor.",
)


@curate_app.command("compare")
def compare(
    lake: str = _LAKE_OPTION,
    left: str = _LEFT_OPTION,
    right: str = _RIGHT_OPTION,
    by: list[str] = _COMPARE_BY_OPTION,
    metric: list[str] = _COMPARE_METRIC_OPTION,
    preview_limit: int = _PREVIEW_LIMIT_OPTION,
    batch_size: int = _BATCH_SIZE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Compare two curated snapshot branches."""
    try:
        from lancedb_robotics.lake import Lake

        comparison = Lake.open(lake).curate.compare(
            left,
            right,
            by=tuple(by or ()),
            metrics=tuple(metric or ()) or None,
            preview_limit=preview_limit,
            batch_size=batch_size,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(comparison.report, sort_keys=True))
        return
    membership = comparison.report.get("membership", {})
    execution = comparison.report.get("execution", {})
    typer.echo(f"lake: {lake}")
    typer.echo(f"left: {comparison.left}")
    typer.echo(f"right: {comparison.right}")
    typer.echo(f"comparison: {comparison.comparison_id}")
    typer.echo(f"shared: {comparison.report['shared_count']}")
    typer.echo(f"removed: {membership.get('removed_count', len(comparison.report['left_only']))}")
    typer.echo(f"added: {membership.get('added_count', len(comparison.report['right_only']))}")
    typer.echo(
        f"execution: bounded={execution.get('bounded')} "
        f"peak_batch_rows={execution.get('peak_batch_rows')} "
        f"scanned={execution.get('total_scanned_rows')}"
    )
    if comparison.report.get("coverage", {}).get("deltas"):
        typer.echo(f"coverage_slices: {len(comparison.report['coverage']['deltas'])}")
    if comparison.report.get("training_eval"):
        left_runs = len(comparison.report["training_eval"]["left"]["training_run_ids"])
        right_runs = len(comparison.report["training_eval"]["right"]["training_run_ids"])
        typer.echo(f"training_runs: left={left_runs} right={right_runs}")
    if comparison.report.get("plugin_metrics"):
        typer.echo(f"plugins: {','.join(comparison.report['plugin_metrics'])}")
    typer.echo(f"transform: {comparison.transform_id}")


@curate_app.command("compare-plan")
def compare_plan(
    lake: str = _LAKE_OPTION,
    left: str = _LEFT_OPTION,
    right: str = _RIGHT_OPTION,
    by: list[str] = _COMPARE_BY_OPTION,
    metric: list[str] = _COMPARE_METRIC_OPTION,
    batch_size: int = _BATCH_SIZE_OPTION,
    local_row_budget: int = _LOCAL_ROW_BUDGET_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Plan a comparison: required tables, versions, estimated scan, executor need."""
    try:
        from lancedb_robotics.lake import Lake

        plan = Lake.open(lake).curate.plan_comparison(
            left,
            right,
            by=tuple(by or ()),
            metrics=tuple(metric or ()) or None,
            batch_size=batch_size,
            local_row_budget=local_row_budget,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(plan.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"left: {plan.left} ({plan.left_count} scenarios)")
    typer.echo(f"right: {plan.right} ({plan.right_count} scenarios)")
    typer.echo(f"estimated_scan_rows: {plan.estimated_scan_rows}")
    typer.echo(f"estimated_peak_rows: {plan.estimated_peak_rows}")
    typer.echo(f"requires_external_executor: {plan.requires_external_executor}")
    for entry in plan.entries:
        typer.echo(
            f"metric: {entry.metric} kind={entry.kind} execution={entry.execution} "
            f"streamed={entry.streamed} estimated_rows={entry.estimated_rows} "
            f"tables={','.join(entry.required_tables)}"
        )


@curate_app.command("comparison-members")
def comparison_members(
    lake: str = _LAKE_OPTION,
    comparison: str = typer.Argument(
        ..., help="comparison_id or snapshot-pair alias (e.g. 'left..right')."
    ),
    field: str = typer.Option(
        None, "--field", help="Membership field to page: added, removed, or shared."
    ),
    offset: int = typer.Option(0, "--offset", help="Start offset into the sorted id list."),
    limit: int = typer.Option(0, "--limit", help="Page size (0 = default preview limit)."),
    page_token: str = typer.Option(
        None, "--page-token", help="Resume token from a prior page or report preview."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Page a comparison's full added/removed/shared id list (bounded reload)."""
    try:
        from lancedb_robotics.lake import Lake

        page = Lake.open(lake).curate.comparison_membership(
            comparison,
            field=field,
            offset=offset,
            limit=limit or None,
            page_token=page_token,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(page.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"comparison: {page.comparison_id}")
    typer.echo(f"field: {page.field}")
    typer.echo(f"offset: {page.offset} limit: {page.limit} total: {page.total}")
    typer.echo(f"scenario_ids: {len(page.scenario_ids)}")
    for scenario_id in page.scenario_ids:
        typer.echo(f"  {scenario_id}")
    if page.next_page_token:
        typer.echo(f"next_page_token: {page.next_page_token}")


@curate_app.command("comparison-staleness")
def comparison_staleness(
    lake: str = _LAKE_OPTION,
    comparison: str = typer.Argument(
        ..., help="comparison_id or snapshot-pair alias (e.g. 'left..right')."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Report whether a persisted comparison's source tables have advanced."""
    try:
        from lancedb_robotics.lake import Lake

        result = Lake.open(lake).curate.comparison_staleness(comparison)
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(result.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"comparison: {result.comparison_id}")
    typer.echo(f"stale: {result.stale}")
    for advanced in result.advanced_tables:
        typer.echo(
            f"advanced: {advanced['table']} "
            f"{advanced['recorded_version']} -> {advanced['current_version']}"
        )


@curate_app.command("comparisons")
def list_comparisons(
    lake: str = _LAKE_OPTION,
    snapshot: str = typer.Option(
        None, "--snapshot", help="Filter by a snapshot on either side (name or dataset id)."
    ),
    left: str = typer.Option(None, "--left", help="Filter by left snapshot (name or dataset id)."),
    right: str = typer.Option(None, "--right", help="Filter by right snapshot (name or dataset id)."),
    metric: str = typer.Option(None, "--metric", help="Filter to reports that requested this metric."),
    state: str = typer.Option(
        None, "--state", help="Filter by lifecycle state: active, archived, or pruned."
    ),
    include_pruned: bool = typer.Option(
        True, "--include-pruned/--no-include-pruned", help="Include pruned reports in the listing."
    ),
    created_by: str = typer.Option(None, "--created-by", help="Filter by report creator."),
    limit: int = typer.Option(0, "--limit", help="Maximum reports to return (0 = all)."),
    json_output: bool = _JSON_OPTION,
) -> None:
    """List persisted curation comparison reports from the catalog (newest first)."""
    try:
        from lancedb_robotics.lake import Lake

        entries = Lake.open(lake).curate.list_comparisons(
            snapshot=snapshot,
            left=left,
            right=right,
            metric=metric,
            state=state,
            include_pruned=include_pruned,
            created_by=created_by,
            limit=limit or None,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps([entry.to_dict() for entry in entries], sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"comparisons: {len(entries)}")
    for entry in entries:
        typer.echo(
            f"comparison: {entry.comparison_id} state={entry.state} pair={entry.pair_alias} "
            f"metrics={','.join(entry.metrics)} added={entry.added_scenario_count} "
            f"removed={entry.removed_scenario_count} shared={entry.shared_scenario_count} "
            f"bytes={entry.report_bytes} reloadable={entry.report_available} "
            f"created={entry.created_at.isoformat()}"
        )


@curate_app.command("comparison")
def show_comparison(
    lake: str = _LAKE_OPTION,
    comparison: str = typer.Argument(
        ..., help="comparison_id or snapshot-pair alias (e.g. 'left..right')."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Reload a persisted comparison report by id or snapshot-pair alias."""
    try:
        from lancedb_robotics.lake import Lake

        result = Lake.open(lake).curate.comparison(comparison)
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(result.report, sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"comparison: {result.comparison_id}")
    typer.echo(f"left: {result.left}")
    typer.echo(f"right: {result.right}")
    typer.echo(f"shared: {result.report['shared_count']}")
    typer.echo(f"removed: {len(result.report['left_only'])}")
    typer.echo(f"added: {len(result.report['right_only'])}")
    typer.echo(f"transform: {result.transform_id}")


@curate_app.command("prune-comparisons")
def prune_comparisons(
    lake: str = _LAKE_OPTION,
    retain_latest: int = typer.Option(
        1, "--retain-latest", help="Newest reports to keep active per snapshot pair."
    ),
    older_than_days: int = typer.Option(
        0,
        "--older-than-days",
        help="Prune bodies older than N days (0 = archive superseded reports, prune nothing).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview retention without writing."),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Apply the comparison retention lifecycle (active -> archived -> pruned)."""
    try:
        from lancedb_robotics.lake import Lake

        older_than = timedelta(days=older_than_days) if older_than_days > 0 else None
        report = Lake.open(lake).curate.prune_comparisons(
            retain_latest=retain_latest,
            older_than=older_than,
            dry_run=dry_run,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"archived: {report.archived_count}")
    typer.echo(f"pruned: {report.pruned_count}")
    typer.echo(f"retained: {len(report.retained_comparison_ids)}")
    typer.echo(f"body_bytes_before: {report.body_bytes_before}")
    typer.echo(f"body_bytes_after: {report.body_bytes_after}")
    typer.echo(f"dry_run: {report.dry_run}")
    if report.transform_id:
        typer.echo(f"transform: {report.transform_id}")


@curate_app.command("eval-metrics")
def eval_metrics(
    lake: str = _LAKE_OPTION,
    snapshot: str = typer.Option(
        None, "--snapshot", help="Filter by snapshot name or dataset id."
    ),
    evaluation_run: str = typer.Option(None, "--evaluation-run", help="Filter by eval run id."),
    training_run: str = typer.Option(None, "--training-run", help="Filter by training run id."),
    model_version: str = typer.Option(None, "--model-version", help="Filter by model version."),
    metric: str = typer.Option(None, "--metric", help="Filter by metric name."),
    slice_label: str = typer.Option(
        None, "--slice", help="Filter by slice label (e.g. 'site_id=site-b|object_category=cup')."
    ),
    regressed_only: bool = typer.Option(
        False, "--regressed-only", help="Only metrics classified as regressions."
    ),
    state: str = typer.Option(
        None, "--state", help="Filter by lifecycle state: active, superseded, or pruned."
    ),
    latest_only: bool = typer.Option(
        False,
        "--latest-only",
        help="Keep the newest import per (snapshot, model version, metric, slice) series.",
    ),
    limit: int = typer.Option(
        0, "--limit", help="Preview rows to print (0 = bounded default of 100)."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """List imported eval metrics from the indexed catalog (newest first)."""
    try:
        from lancedb_robotics.lake import Lake

        listing = Lake.open(lake).curate.list_eval_metrics(
            snapshot=snapshot,
            evaluation_run=evaluation_run,
            training_run=training_run,
            model_version=model_version,
            metric=metric,
            slice_label=slice_label,
            regressed_only=regressed_only,
            state=state,
            latest_only=latest_only,
            limit=limit or None,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(listing.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"metrics: {listing.total_count}")
    typer.echo(f"showing: {len(listing.entries)} (preview_limit={listing.preview_limit})")
    for entry in listing.entries:
        slice_part = f" slice={entry.slice_label}" if entry.slice_label else ""
        baseline_part = (
            f" baseline={entry.baseline_score} improvement={entry.improvement}"
            if entry.baseline_score is not None
            else ""
        )
        typer.echo(
            f"metric: {entry.metric} snapshot={entry.snapshot_name} "
            f"model={entry.model_version} eval_run={entry.evaluation_run_id}{slice_part} "
            f"score={entry.score}{baseline_part} regressed={entry.regressed} "
            f"state={entry.state} source_available={entry.source_available} "
            f"created={entry.created_at.isoformat()}"
        )


@curate_app.command("eval-metric-staleness")
def eval_metric_staleness(
    lake: str = _LAKE_OPTION,
    metric: str = typer.Argument(
        ..., help="model_output_id of a metric or an evaluation_run_id."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Report whether an eval metric's recorded source tables have advanced."""
    try:
        from lancedb_robotics.lake import Lake

        result = Lake.open(lake).curate.eval_metric_staleness(metric)
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(result.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"metric: {result.model_output_id}")
    typer.echo(f"evaluation_run: {result.evaluation_run_id}")
    typer.echo(f"stale: {result.stale}")
    for advanced in result.advanced_tables:
        typer.echo(
            f"advanced: {advanced['table']} "
            f"{advanced['recorded_version']} -> {advanced['current_version']}"
        )


@curate_app.command("prune-eval-metrics")
def prune_eval_metrics(
    lake: str = _LAKE_OPTION,
    retain_latest: int = typer.Option(
        1, "--retain-latest", help="Newest imports to keep active per metric series."
    ),
    older_than_days: int = typer.Option(
        0,
        "--older-than-days",
        help="Prune superseded source rows older than N days (0 = supersede only, delete nothing).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview retention without writing."),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Apply the eval-metric retention lifecycle (active -> superseded -> pruned)."""
    try:
        from lancedb_robotics.lake import Lake

        older_than = timedelta(days=older_than_days) if older_than_days > 0 else None
        report = Lake.open(lake).curate.prune_eval_metrics(
            retain_latest=retain_latest,
            older_than=older_than,
            dry_run=dry_run,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"superseded: {report.superseded_count}")
    typer.echo(f"pruned: {report.pruned_count}")
    typer.echo(f"protected: {len(report.protected_ids)}")
    typer.echo(f"retained: {len(report.retained_ids)}")
    typer.echo(f"dry_run: {report.dry_run}")
    if report.transform_id:
        typer.echo(f"transform: {report.transform_id}")


@curate_app.command("sync-eval-metrics")
def sync_eval_metrics(
    lake: str = _LAKE_OPTION,
    build_indexes: bool = typer.Option(
        True,
        "--build-indexes/--no-build-indexes",
        help="Build the catalog's scalar predicate indexes after the rebuild.",
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Rebuild the eval metric catalog from model_outputs (pre-0095 lakes, drift repair)."""
    try:
        from lancedb_robotics.lake import Lake

        report = Lake.open(lake).curate.sync_eval_metric_catalog(build_indexes=build_indexes)
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"scanned_model_outputs: {report.scanned_model_outputs}")
    typer.echo(f"cataloged: {report.cataloged}")
    typer.echo(f"active: {report.active}")
    typer.echo(f"superseded: {report.superseded}")
    typer.echo(f"preserved_pruned: {report.preserved_pruned}")
    for result in report.index_results:
        typer.echo(
            f"index: {result['column']} type={result['index_type']} status={result['status']}"
        )
    typer.echo(f"transform: {report.transform_id}")


@curate_app.command("feedback-from-eval")
def feedback_from_eval(
    lake: str = _LAKE_OPTION,
    snapshot: str = typer.Option(..., "--snapshot", help="Snapshot name that produced the eval run."),
    input_file: Path = _INPUT_JSON_OPTION,
    training_run: str = _TRAINING_RUN_OPTION,
    model_version: str = _FEEDBACK_MODEL_VERSION_OPTION,
    evaluation_run: str = _EVALUATION_RUN_OPTION,
    regression_threshold: float = _REGRESSION_THRESHOLD_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Import eval metrics and link them to a curated snapshot."""
    try:
        from lancedb_robotics.lake import Lake

        payload = _load_json(input_file)
        metrics = payload.get("metrics", payload) if isinstance(payload, dict) else payload
        report = Lake.open(lake).curate.feedback_from_eval(
            snapshot,
            metrics=metrics,
            training_run_id=training_run,
            model_version=model_version,
            evaluation_run_id=evaluation_run,
            regression_threshold=regression_threshold,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.report, sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"snapshot: {report.snapshot_name} ({report.dataset_id})")
    typer.echo(f"training_run: {report.training_run_id}")
    typer.echo(f"evaluation_run: {report.evaluation_run_id}")
    typer.echo(f"metrics: {len(report.metric_output_ids)}")
    typer.echo(f"regressions: {len(report.regressions)}")
    typer.echo(f"transform: {report.transform_id}")


@curate_app.command("next-candidates")
def next_candidates(
    lake: str = _LAKE_OPTION,
    input_file: Path = _INPUT_JSON_OPTION,
    queue: str = typer.Option(None, "--queue", help="Review queue name to create."),
    view: str = _OPTIONAL_VIEW_OPTION,
    snapshot: str = _OPTIONAL_SNAPSHOT_OPTION,
    tag: str = _TAG_OPTION,
    limit: int = _LIMIT_OPTION,
    no_queue: bool = typer.Option(False, "--no-queue", help="Do not create a review queue."),
    assignee: str = _ASSIGNEE_OPTION,
    status: str = _STATUS_OPTION,
    route: str = typer.Option(
        "auto", "--route", help="Candidate search routing: auto, exact, or ann."
    ),
    nprobes: int = typer.Option(
        None, "--nprobes", help="ANN partitions to probe on the indexed route."
    ),
    refine_factor: int = typer.Option(
        None, "--refine-factor", help="ANN exact re-rank multiplier on the indexed route."
    ),
    preview_limit: int = typer.Option(
        100, "--preview-limit", help="Bounded candidate preview rows per regression slice."
    ),
    explain: bool = typer.Option(
        False, "--explain", help="Print the candidate plan without writing anything."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Compute candidates and artifact identities without writing any rows.",
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Create next curation candidates from eval regressions."""
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        if explain:
            plan = opened.curate.plan_next_candidates(
                from_regressions=_load_json(input_file),
                queue_name=queue,
                view_name=view,
                snapshot_name=snapshot,
                snapshot_tag=tag,
                limit_per_regression=limit,
                create_queue=not no_queue,
                route=route,
                nprobes=nprobes,
                refine_factor=refine_factor,
                preview_limit=preview_limit,
            )
        else:
            report = opened.curate.next_candidates(
                from_regressions=_load_json(input_file),
                queue_name=queue,
                view_name=view,
                snapshot_name=snapshot,
                snapshot_tag=tag,
                limit_per_regression=limit,
                create_queue=not no_queue,
                assignee=assignee,
                status=status,
                route=route,
                nprobes=nprobes,
                refine_factor=refine_factor,
                preview_limit=preview_limit,
                dry_run=dry_run,
            )
    except Exception as exc:
        _exit(exc)
    if explain:
        if json_output:
            typer.echo(json.dumps(plan.to_dict(), sort_keys=True))
            return
        typer.echo(f"lake: {lake}")
        typer.echo(f"plan: {plan.plan_id}")
        typer.echo(f"route: {plan.effective_route} ({plan.route_reason})")
        typer.echo(f"source_scenarios: {plan.source_scenario_count}")
        for stage in plan.stages:
            typer.echo(
                f"stage: {stage['stage']} strategy={stage['strategy']} "
                f"rows={stage['estimated_rows']}"
            )
        for requirement in plan.index_requirements:
            state = "met" if requirement["met"] else "unmet"
            required = "required" if requirement["required"] else "optional"
            typer.echo(
                f"index: {requirement['table']}.{requirement['column']} "
                f"kind={requirement['kind']} {required} {state}"
            )
            if requirement["remedy"]:
                typer.echo(f"  remedy: {requirement['remedy']}")
        typer.echo(f"candidates: {plan.total_candidate_count}")
        for name, artifact in plan.expected_artifacts.items():
            if artifact:
                typer.echo(f"artifact: {name} {json.dumps(artifact, sort_keys=True)}")
        return
    if json_output:
        typer.echo(json.dumps(report.report, sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"operation: {report.selection.operation}")
    typer.echo(f"scenarios: {len(report.selection.scenario_ids)}")
    if report.plan:
        typer.echo(f"plan: {report.plan.plan_id} route={report.plan.effective_route}")
    if report.dry_run:
        typer.echo("dry-run: no queue/view/snapshot rows written")
    if report.queue:
        typer.echo(f"queue: {report.queue.name} ({report.queue.queue_id})")
    if report.view:
        typer.echo(f"view: {report.view.name} ({report.view.view_id})")
    if report.snapshot:
        typer.echo(f"snapshot: {report.snapshot.name} ({report.snapshot.dataset_id})")
    typer.echo(f"transform: {report.transform_id}")


@curate_app.command("promote-snapshot")
def promote_snapshot(
    lake: str = _LAKE_OPTION,
    snapshot: str = typer.Option(..., "--snapshot", help="Snapshot branch to promote or reject."),
    decision: str = _PROMOTION_DECISION_OPTION,
    reason: str = _REASON_OPTION,
    input_file: Path | None = _OPTIONAL_INPUT_JSON_OPTION,
    training_run: str = _TRAINING_RUN_OPTION,
    model_version: str = _FEEDBACK_MODEL_VERSION_OPTION,
    evaluation_run: str = _EVALUATION_RUN_OPTION,
    reviewer: str = _REVIEWER_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Record a promote/reject decision for a curated snapshot branch."""
    try:
        from lancedb_robotics.lake import Lake

        payload = _load_json(input_file) if input_file else None
        metrics = payload.get("regressions", payload.get("metrics")) if isinstance(payload, dict) else payload
        report = Lake.open(lake).curate.promote_snapshot(
            snapshot,
            decision=decision,
            reason=reason,
            evaluation_run_id=evaluation_run,
            training_run_id=training_run,
            model_version=model_version,
            metrics=metrics,
            reviewer=reviewer,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.report, sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"snapshot: {report.snapshot_name} ({report.dataset_id})")
    typer.echo(f"decision: {report.decision}")
    typer.echo(f"memberships: {len(report.membership_ids)}")
    typer.echo(f"transform: {report.transform_id}")


@curate_app.command("materialization-report")
def materialization_report(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    target_format: str = _TARGET_FORMAT_OPTION,
    output_uri: str = _OUTPUT_URI_OPTION,
    mode: str = _MODE_OPTION,
    copied_payload_bytes: int = _COPIED_BYTES_OPTION,
    metadata_bytes: int = _METADATA_BYTES_OPTION,
    projection_transform: str = _PROJECTION_TRANSFORM_OPTION,
) -> None:
    """Record copy accounting for a boundary projection/export."""
    try:
        from lancedb_robotics.lake import Lake

        report = Lake.open(lake).curate.materialization_report(
            snapshot,
            target_format=target_format,
            output_uri=output_uri,
            mode=mode,
            copied_payload_bytes=copied_payload_bytes,
            metadata_bytes_written=metadata_bytes,
            projection_transform_id=projection_transform,
        )
    except Exception as exc:
        _exit(exc)
    typer.echo(f"lake: {lake}")
    typer.echo(f"snapshot: {report.snapshot_name} ({report.dataset_id})")
    typer.echo(f"format: {report.target_format}")
    typer.echo(f"total_payload_bytes: {report.total_payload_bytes}")
    typer.echo(f"copied_payload_bytes: {report.copied_payload_bytes}")
    typer.echo(f"logical_reference_bytes: {report.logical_reference_bytes}")
    typer.echo(f"metadata_bytes_written: {report.metadata_bytes_written}")
    typer.echo(f"transform: {report.transform_id}")


@curate_app.command("materialization-history")
def materialization_history(
    lake: str = _LAKE_OPTION,
    dataset_id: str = typer.Option(None, "--dataset", help="Filter by dataset id."),
    snapshot: str = typer.Option(None, "--snapshot", help="Filter by snapshot name (branch)."),
    target_format: str = typer.Option(None, "--format", help="Filter by target projection format."),
    mode: str = typer.Option(None, "--mode", help="Filter by mode (projection/export/plan/dry-run/live)."),
    payload_copy_policy: str = typer.Option(
        None,
        "--policy",
        help="Filter by copy policy (logical-reference/materialized-copy/would-copy-payloads).",
    ),
    state: str = typer.Option(
        None, "--state", help="Filter by lifecycle state: active, superseded, or pruned."
    ),
    page_size: int = typer.Option(
        0, "--page-size", help="Rows per page (0 = bounded default of 50, capped at 1000)."
    ),
    cursor: str = typer.Option(None, "--cursor", help="Resume from a prior page's next_cursor."),
    all_pages: bool = typer.Option(False, "--all", help="Stream every page to exhaustion."),
    json_output: bool = _JSON_OPTION,
) -> None:
    """List materialization history from the indexed rollup catalog, bounded + paged.

    Filters push down to scalar-indexed columns and rows are keyset-paged by
    ``(created_at, materialization_id)`` in O(page_size) client memory -- history
    queries never scan or parse every ``report_json`` blob (backlog 0145).
    """
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        kwargs = dict(
            dataset_id=dataset_id or None,
            snapshot=snapshot or None,
            target_format=target_format or None,
            mode=mode or None,
            payload_copy_policy=payload_copy_policy or None,
            state=state or None,
            page_size=page_size or None,
        )
        if all_pages:
            pages = []
            current = cursor or None
            while True:
                page = opened.curate.materialization_history(cursor=current, **kwargs)
                pages.append(page)
                if not page.has_more or not page.next_cursor:
                    break
                current = page.next_cursor
        else:
            pages = [opened.curate.materialization_history(cursor=cursor or None, **kwargs)]
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps({"pages": [page.to_dict() for page in pages]}, sort_keys=True, default=str))
        return
    typer.echo(f"lake: {lake}")
    for page in pages:
        typer.echo(
            f"page[{page.page_index}] count={len(page.records)} "
            f"has_more={page.has_more} next_cursor={page.next_cursor or '-'}"
        )
        for entry in page.records:
            typer.echo(
                f"  {entry.created_at} {entry.snapshot_name} {entry.target_format} "
                f"{entry.mode} state={entry.state} copied={entry.copied_payload_bytes} "
                f"planned={entry.planned_payload_bytes} ratio={entry.copy_ratio:.3f} "
                f"{entry.materialization_id}"
            )


@curate_app.command("materialization-rollup")
def materialization_rollup(
    lake: str = _LAKE_OPTION,
    dataset_id: str = typer.Option(None, "--dataset", help="Scope to a dataset id."),
    snapshot: str = typer.Option(None, "--snapshot", help="Scope to a snapshot name (branch)."),
    group_by: str = typer.Option(
        None,
        "--group-by",
        help="Bucket copy-cost by: snapshot|branch|format|mode|policy|transform|day.",
    ),
    include_pruned: bool = typer.Option(
        True, "--include-pruned/--exclude-pruned", help="Include pruned rollup rows in the totals."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Aggregate copy-cost over the indexed rollup catalog (branch/format trends).

    Reads only promoted numeric/identity columns -- never parses ``report_json``.
    The ``--group-by`` buckets are the copy-cost trend surface branch comparison
    and benchmark reports reuse (backlog 0145).
    """
    try:
        from lancedb_robotics.lake import Lake

        rollup = Lake.open(lake).curate.materialization_rollup(
            dataset_id=dataset_id or None,
            snapshot=snapshot or None,
            include_pruned=include_pruned,
            group_by=group_by or None,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(rollup.to_dict(), sort_keys=True, default=str))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"materialization_count: {rollup.materialization_count}")
    typer.echo(f"total_payload_bytes: {rollup.total_payload_bytes}")
    typer.echo(f"copied_payload_bytes: {rollup.copied_payload_bytes}")
    typer.echo(f"logical_reference_bytes: {rollup.logical_reference_bytes}")
    typer.echo(f"planned_payload_bytes: {rollup.planned_payload_bytes}")
    typer.echo(f"metadata_bytes_written: {rollup.metadata_bytes_written}")
    typer.echo(f"copy_ratio: {rollup.copy_ratio:.4f}")
    typer.echo(f"output_file_count: {rollup.output_file_count}")
    if rollup.group_by:
        typer.echo(f"group_by: {rollup.group_by}")
        for bucket in rollup.buckets:
            typer.echo(
                f"  {bucket.key}: count={bucket.materialization_count} "
                f"copied={bucket.copied_payload_bytes} planned={bucket.planned_payload_bytes} "
                f"ratio={bucket.copy_ratio:.3f}"
            )


@curate_app.command("sync-materialization-rollups")
def sync_materialization_rollups(
    lake: str = _LAKE_OPTION,
    skip_indexes: bool = typer.Option(
        False, "--skip-indexes", help="Do not build the rollup predicate indexes after rebuild."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Rebuild the rollup catalog from ``curation_materializations`` (backlog 0145).

    Run once on a lake that recorded materializations before 0145, or any time to
    repair catalog drift; ``materialization-report`` keeps it current otherwise.
    """
    try:
        from lancedb_robotics.lake import Lake

        report = Lake.open(lake).curate.sync_materialization_rollups(
            build_indexes=not skip_indexes,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"source_count: {report.source_count}")
    typer.echo(f"rebuilt_count: {report.rebuilt_count}")
    typer.echo(f"preserved_pruned_count: {report.preserved_pruned_count}")
    typer.echo(f"indexes_built: {len(report.indexes_built)}")


@curate_app.command("prune-materializations")
def prune_materializations(
    lake: str = _LAKE_OPTION,
    retain_latest: int = typer.Option(
        1, "--retain-latest", help="Newest plan reports to keep active per (dataset, format, uri, mode) series."
    ),
    older_than_days: int = typer.Option(
        0,
        "--older-than-days",
        help="Prune plan bodies older than N days (0 = mark superseded, prune nothing).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview retention without writing."),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Compact superseded plan/dry-run reports (active -> superseded -> pruned).

    Completed export evidence is never eligible. Pruning is safe-delete: the
    source ``report_json`` and per-file chunks are cleared while the promoted
    rollup row + digest survive as audit evidence (backlog 0145).
    """
    try:
        from lancedb_robotics.lake import Lake

        older_than = timedelta(days=older_than_days) if older_than_days > 0 else None
        report = Lake.open(lake).curate.prune_materializations(
            retain_latest=retain_latest,
            older_than=older_than,
            dry_run=dry_run,
        )
    except Exception as exc:
        _exit(exc)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), sort_keys=True))
        return
    typer.echo(f"lake: {lake}")
    typer.echo(f"scanned: {report.scanned_count}")
    typer.echo(f"protected: {report.protected_count}")
    typer.echo(f"superseded: {report.superseded_count}")
    typer.echo(f"pruned: {report.pruned_count}")
    typer.echo(f"body_bytes_before: {report.body_bytes_before}")
    typer.echo(f"body_bytes_after: {report.body_bytes_after}")
    typer.echo(f"file_chunks_deleted: {report.file_chunks_deleted}")
    typer.echo(f"dry_run: {report.dry_run}")


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc.msg}") from exc


def _workbench(lake: str, *, scenario_id: list[str] | None, coverage_tag: list[str] | None):
    from lancedb_robotics.curate import CurationScope
    from lancedb_robotics.lake import Lake

    opened = Lake.open(lake)
    scope = CurationScope(
        scenario_ids=tuple(scenario_id or ()),
        coverage_tags=tuple(coverage_tag or ()),
    )
    return opened.curate.workbench(scope=scope)


def _selection_for_view_or_scope(
    lake: str,
    *,
    view: str,
    scenario_id: list[str] | None,
):
    from lancedb_robotics.curate import CurationError
    from lancedb_robotics.lake import Lake

    opened = Lake.open(lake)
    try:
        return opened.curate.view(view)
    except CurationError:
        if not scenario_id:
            raise
        scope = opened.scope(scenario_ids=tuple(scenario_id))
        return opened.curate.workbench(scope=scope)


def _echo_result(selection, manifest) -> None:
    typer.echo(f"lake: {manifest.lake_uri}")
    typer.echo(f"operation: {selection.operation}")
    typer.echo(f"scenarios: {len(selection.scenario_ids)}")
    typer.echo(f"snapshot: {manifest.name} ({manifest.dataset_id})")
    typer.echo(f"tag: {manifest.tag}")
    typer.echo(f"transform: {selection.transform_id}")
    typer.echo(f"snapshot transform: {manifest.transform_id}")


def _echo_selection(lake: str, selection) -> None:
    typer.echo(f"lake: {lake}")
    typer.echo(f"operation: {selection.operation}")
    typer.echo(f"scenarios: {len(selection.scenario_ids)}")
    typer.echo(f"transform: {selection.transform_id}")


def _echo_review_queue(lake: str, queue) -> None:
    typer.echo(f"lake: {lake}")
    typer.echo(f"queue: {queue.name} ({queue.queue_id})")
    typer.echo(f"source: {queue.source_operation}")
    typer.echo(f"target_grain: {queue.target_grain}")
    typer.echo(f"items: {queue.item_count if queue.item_count is not None else len(queue.item_ids)}")
    typer.echo(f"scenarios: {len(queue.scenario_ids)}")
    typer.echo(f"transform: {queue.transform_id}")


def _echo_dedup_plan(lake: str, plan) -> None:
    typer.echo(f"lake: {lake}")
    operation = (
        "distributed-dedup-plan"
        if plan.report.get("operation") == "distributed-dedup"
        else "dedup-plan"
    )
    typer.echo(f"operation: {operation}")
    typer.echo(f"groups: {len(plan.groups)}")
    typer.echo(f"representatives: {len(plan.representative_ids)}")
    typer.echo(f"dropped: {len(plan.dropped_scenario_ids)}")
    typer.echo(f"search_strategy: {plan.report['search_strategy']}")
    if plan.report.get("job_id"):
        typer.echo(f"job: {plan.report['job_id']}")
    if plan.report.get("recall_audit"):
        recall = plan.report["recall_audit"]
        typer.echo(f"recall_estimate: {recall.get('estimated_recall')}")
    typer.echo(f"transform: {plan.transform_id}")


def _review_connector(
    tool: str,
    connector_state: Path | None,
    *,
    dry_run: bool,
):
    if dry_run:
        return None
    normalized = str(tool or "").strip().lower()
    if normalized in {"json-file", "json", "local-json"}:
        if connector_state is None:
            raise ValueError("--connector-state is required for the json-file connector")
        return JsonFileReviewToolConnector(connector_state, tool="json-file")
    raise ValueError(
        f"no built-in review connector for {tool!r}; use --tool json-file "
        "or call the Python SDK with a connector implementation"
    )


def _exit(exc: Exception) -> None:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=1) from exc
