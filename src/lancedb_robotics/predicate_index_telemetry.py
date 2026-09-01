"""Predicate-index selectivity telemetry and recommendations (backlog 0137).

Backlog 0079 made scalar predicate index *status* visible on aligned training
reads; backlog 0136 gave those indexes a durable, capability-aware *job*
lifecycle. This module closes the loop: it turns the per-read predicate status
that already rides along in aligned training loader reports into **evidence** for
which columns deserve an index as a corpus grows, and routes the safe, high-
confidence recommendations back into the 0136 job lifecycle so default users get
fast reads without ever naming a column.

Design (see the ``predicate-index-recommender-stateless-over-durable-reports``
decision):

* **No new canonical table.** The aligned loader report is already persisted by
  the 0115 ``training_reports`` catalog. Telemetry is emitted as an additive
  ``predicate_telemetry`` section *inside* that report (:func:`build_predicate_telemetry`),
  carrying only identifiers and shape -- table, per-column index status/role,
  backend kind, buildability, and read-level selectivity -- **never literal
  predicate values**, so nothing leaks and there is no read-path write.
* **Stateless aggregation.** :func:`extract_predicate_observations` normalizes a
  supplied report/manifest into :class:`PredicateObservation`s;
  :func:`aggregate_predicate_observations` groups them by
  ``(table, column, column_kind)``. The *filter shape* is the sorted set of
  filter *columns*, so two reads filtering ``alignment_id='a'`` and
  ``alignment_id='b'`` group together (correct for "repeated use") and no literal
  value is ever recorded.
* **Guardrailed recommendations.** :func:`recommend_from_aggregates` emits typed
  ``create_scalar_index`` / ``refresh_stale_index`` recommendations and advisory
  ``promote_jsonb_column`` recommendations, suppressing small tables, low-
  selectivity predicates, unsupported backends, and rarely-seen predicates by
  default. JSONB-path predicates are never auto-indexed (they need a schema
  change, not an index).
* **Auditable, reversible apply.** :func:`apply_index_recommendations` routes only
  safe, auto-applicable typed recommendations into the durable, idempotent 0136
  ``request_scalar_index`` job store; it returns a per-recommendation outcome
  record and never mutates data. Operators can ``force`` guardrail-suppressed
  columns and ``suppress`` specific columns.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import fields as _dc_fields
from typing import Any

# --------------------------------------------------------------------------- #
# Normalized per-read index status.
#
# ``describe_scalar_indexes`` (0079) reports status ``skipped`` for TWO very
# different situations, distinguished only by reason:
#   1. the backend cannot build scalar indexes at all, or
#   2. the backend can, but this column simply is not indexed yet.
# Case (2) is the primary recommendation candidate; case (1) must *suppress*
# recommendations. We split them here so downstream code never has to re-parse
# a reason string.
# --------------------------------------------------------------------------- #

#: Normalized index-status vocabulary emitted by :func:`normalize_index_status`.
INDEX_PRESENT = "present"  # built | already_present
INDEX_UNINDEXED = "unindexed"  # buildable backend, no index yet -> candidate
INDEX_UNSUPPORTED = "unsupported"  # backend cannot build scalar indexes
INDEX_ABSENT_COLUMN = "absent_column"  # no such column to index
INDEX_FAILED = "failed"  # a real build failure (stale/broken)

#: Substrings in a ``describe_scalar_indexes`` reason that mean the *backend*
#: cannot build scalar indexes (as opposed to a column merely being unindexed).
_UNSUPPORTED_REASON_MARKERS = (
    "does not expose create_scalar_index",
    "does not expose list_indices",
    "predicate pushdown remains available; backend",
)
#: Substring that marks a "there is no such column" failure (cannot be indexed).
_ABSENT_COLUMN_REASON_MARKER = "column to index in table"

#: Terminal job states (0136) that mean an index the read expected is not usable.
_STALE_JOB_STATES = frozenset({"failed", "expired", "canceled", "cancelled"})

#: JSONB columns whose sub-paths cannot carry a scalar index and should be
#: promoted to a typed column first. ``aligned_ticks`` stores per-stream status
#: and confidence inside ``stream_detail_json``; a per-stream ``status`` filter is
#: therefore a JSONB-path predicate, not a typed-column predicate.
JSONB_PREDICATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "aligned_ticks": ("stream_detail_json.status",),
}


def _is_unsupported_reason(reason: Any) -> bool:
    text = str(reason or "")
    return any(marker in text for marker in _UNSUPPORTED_REASON_MARKERS)


def normalize_index_status(payload: Mapping[str, Any]) -> str:
    """Collapse a raw predicate-index payload into the normalized vocabulary."""
    status = str(payload.get("status") or "").strip().lower()
    reason = payload.get("reason")
    if status in {"built", "already_present"}:
        return INDEX_PRESENT
    if status == "failed":
        if _ABSENT_COLUMN_REASON_MARKER in str(reason or ""):
            return INDEX_ABSENT_COLUMN
        return INDEX_FAILED
    if status == "skipped":
        if _is_unsupported_reason(reason):
            return INDEX_UNSUPPORTED
        return INDEX_UNINDEXED
    # Unknown/empty status: treat as unindexed but harmless (never a hard error).
    return INDEX_UNINDEXED


def _job_backed(payload: Mapping[str, Any]) -> bool:
    return payload.get("job_status") == "complete"


def _index_backed(normalized: str, payload: Mapping[str, Any]) -> bool:
    return normalized == INDEX_PRESENT or _job_backed(payload)


# --------------------------------------------------------------------------- #
# Observation model.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PredicateObservation:
    """One column's predicate-planning telemetry from a single training read.

    Read-level ``total_rows``/``selected_rows`` are the whole read's row count and
    post-filter selection; ``selectivity_fraction`` is ``selected/total`` (the
    fraction *retained* -- lower is more selective). They are attributed to every
    predicate in the read, since the SDK plan only records a read-level selection,
    not a per-column one.
    """

    table: str
    column: str
    column_kind: str  # "typed" | "jsonb_path"
    predicate_role: str  # "filter" | "quality-diagnostic" | "hot-column"
    used_in_filter: bool
    index_status: str  # normalized vocabulary above
    index_backed: bool
    index_type: str | None = None
    job_status: str | None = None
    backend_kind: str | None = None
    backend_supports_index: bool = True
    total_rows: int | None = None
    selected_rows: int | None = None
    reason: str | None = None
    sequence: int = 0  # caller-assigned chronological order (higher = newer)

    @property
    def selectivity_fraction(self) -> float | None:
        if self.total_rows in (None, 0) or self.selected_rows is None:
            return None
        return max(0.0, min(1.0, float(self.selected_rows) / float(self.total_rows)))

    @property
    def is_jsonb(self) -> bool:
        return self.column_kind == "jsonb_path"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selectivity_fraction"] = self.selectivity_fraction
        return payload


# --------------------------------------------------------------------------- #
# Telemetry section (emitted into the aligned loader report).
# --------------------------------------------------------------------------- #


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _jsonb_filter_paths(table: str, quality_policy: Mapping[str, Any]) -> tuple[str, ...]:
    """JSONB-path predicates a read applied (derived from the quality policy).

    Only ``aligned_ticks`` per-stream ``status`` lives in JSONB today (see
    :data:`JSONB_PREDICATE_COLUMNS`): a read that constrains per-stream ``statuses``
    post-filters ``stream_detail_json`` rather than a typed column, so it surfaces
    as ``stream_detail_json.status``.
    """
    known = JSONB_PREDICATE_COLUMNS.get(table, ())
    if "stream_detail_json.status" in known and (quality_policy.get("statuses") or ()):
        return ("stream_detail_json.status",)
    return ()


def build_predicate_telemetry(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the additive ``predicate_telemetry`` report section from a manifest.

    Pure over the manifest dict (no lake handle, no I/O): it reads the
    ``predicate_indexes`` the aligned plan already recorded, the read-level
    selection (``total_ticks``/``selected_ticks``), the resolved backend kind, and
    the quality policy (to surface JSONB-path predicates). Emits only identifiers,
    shape, and counts -- never literal predicate values.
    """
    output_table = manifest.get("output_table")
    backend = manifest.get("backend")
    backend_kind = None
    if isinstance(backend, Mapping):
        backend_kind = backend.get("resolved_backend") or backend.get("requested_backend")
    total_rows = _int_or_none(manifest.get("total_ticks"))
    selected_rows = _int_or_none(manifest.get("selected_ticks"))
    raw = manifest.get("predicate_indexes") or ()

    supports_index = True
    normalized_predicates: list[dict[str, Any]] = []
    filter_columns: list[str] = []
    for payload in raw:
        if not isinstance(payload, Mapping):
            continue
        normalized = normalize_index_status(payload)
        if normalized == INDEX_UNSUPPORTED:
            supports_index = False
        used_in_filter = bool(payload.get("used_in_filter"))
        column = payload.get("column")
        if used_in_filter and column is not None:
            filter_columns.append(str(column))
        normalized_predicates.append(
            {
                "table": payload.get("table", output_table),
                "column": column,
                "column_kind": "typed",
                "predicate_role": payload.get("predicate_role", "hot-column"),
                "used_in_filter": used_in_filter,
                "index_status": normalized,
                "index_backed": _index_backed(normalized, payload),
                "index_type": payload.get("index_type"),
                "job_status": payload.get("job_status"),
                "num_rows": _int_or_none(payload.get("num_rows")),
                "reason": payload.get("reason"),
            }
        )

    quality_policy = manifest.get("quality_policy")
    if isinstance(quality_policy, Mapping) and output_table is not None:
        for jsonb_path in _jsonb_filter_paths(str(output_table), quality_policy):
            filter_columns.append(jsonb_path)
            normalized_predicates.append(
                {
                    "table": output_table,
                    "column": jsonb_path,
                    "column_kind": "jsonb_path",
                    "predicate_role": "filter",
                    "used_in_filter": True,
                    "index_status": INDEX_UNINDEXED,
                    "index_backed": False,
                    "index_type": None,
                    "job_status": None,
                    "num_rows": total_rows,
                    "reason": (
                        "JSONB-path predicate; a scalar index cannot be built on a "
                        "JSONB sub-path -- promote it to a typed column first"
                    ),
                }
            )

    selectivity = None
    if total_rows not in (None, 0) and selected_rows is not None:
        selectivity = max(0.0, min(1.0, float(selected_rows) / float(total_rows)))

    return {
        "output_table": output_table,
        "backend_kind": backend_kind,
        "backend_supports_scalar_index": supports_index,
        "total_rows": total_rows,
        "selected_rows": selected_rows,
        "selectivity_fraction": selectivity,
        "filter_columns": sorted(set(filter_columns)),
        "predicates": normalized_predicates,
    }


# --------------------------------------------------------------------------- #
# Extraction from reports / manifests.
# --------------------------------------------------------------------------- #


def _telemetry_section(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Find the ``predicate_telemetry`` section in a report or manifest.

    Accepts a stored loader report (section at the top level), a manifest that
    already embedded it, or a raw manifest (built on the fly from
    ``predicate_indexes``).
    """
    section = payload.get("predicate_telemetry")
    if isinstance(section, Mapping):
        return dict(section)
    # A raw aligned manifest (or its ``loader_report`` wrapper) -- build it.
    if "predicate_indexes" in payload:
        return build_predicate_telemetry(payload)
    loader_report = payload.get("loader_report")
    if isinstance(loader_report, Mapping):
        return _telemetry_section(loader_report)
    return None


def extract_predicate_observations(
    payload: Mapping[str, Any],
    *,
    sequence: int = 0,
) -> list[PredicateObservation]:
    """Normalize one report/manifest into :class:`PredicateObservation`s.

    ``sequence`` is a caller-assigned chronological rank (higher = newer) used by
    :func:`aggregate_predicate_observations` to pick the latest index status.
    Returns ``[]`` when the payload carries no predicate telemetry.
    """
    section = _telemetry_section(payload)
    if not section:
        return []
    backend_kind = section.get("backend_kind")
    supports = bool(section.get("backend_supports_scalar_index", True))
    total_rows = _int_or_none(section.get("total_rows"))
    selected_rows = _int_or_none(section.get("selected_rows"))
    observations: list[PredicateObservation] = []
    for predicate in section.get("predicates") or ():
        if not isinstance(predicate, Mapping):
            continue
        column = predicate.get("column")
        table = predicate.get("table") or section.get("output_table")
        if column is None or table is None:
            continue
        status = str(predicate.get("index_status") or INDEX_UNINDEXED)
        observations.append(
            PredicateObservation(
                table=str(table),
                column=str(column),
                column_kind=str(predicate.get("column_kind") or "typed"),
                predicate_role=str(predicate.get("predicate_role") or "hot-column"),
                used_in_filter=bool(predicate.get("used_in_filter")),
                index_status=status,
                index_backed=bool(predicate.get("index_backed")),
                index_type=predicate.get("index_type"),
                job_status=predicate.get("job_status"),
                backend_kind=backend_kind,
                backend_supports_index=supports and status != INDEX_UNSUPPORTED,
                total_rows=_int_or_none(predicate.get("num_rows")) or total_rows,
                selected_rows=selected_rows,
                reason=predicate.get("reason"),
                sequence=sequence,
            )
        )
    return observations


def recent_aligned_reports(lake: Any, *, limit: int = 200) -> list[dict[str, Any]]:
    """Load the newest ``limit`` aligned loader-report bodies (oldest-first).

    Read-only and bounded: delegates to ``run_manifests.recent_training_reports``,
    which selects the newest reports by ``created_at`` in a single streaming scan
    with a bounded min-heap -- it never fans out into one scan per report id and
    never holds more than ``limit`` bodies in memory. Returns ``[]`` when the
    catalog table is absent so a fresh lake degrades to "no recommendations",
    never an error. The ``run_manifests`` import is deferred so this module carries
    no import-time dependency on it.
    """
    from lancedb_robotics.run_manifests import recent_training_reports

    try:
        records = recent_training_reports(lake, loader_kind="aligned-training", limit=limit)
    except Exception:  # noqa: BLE001 - a missing/empty catalog is not an error
        return []
    return [dict(record.report) for record in records if record.report]


def extract_observations_from_reports(
    reports: Iterable[Mapping[str, Any]],
) -> list[PredicateObservation]:
    """Extract observations from an ordered iterable of reports (oldest first).

    Each report is assigned an increasing ``sequence`` so the aggregate can tell
    which index status is the most recent.
    """
    observations: list[PredicateObservation] = []
    for index, report in enumerate(reports):
        if isinstance(report, Mapping):
            observations.extend(extract_predicate_observations(report, sequence=index))
    return observations


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PredicateAggregate:
    """Repeated-use rollup of one ``(table, column, column_kind)`` predicate."""

    table: str
    column: str
    column_kind: str
    observations: int
    filter_observations: int
    index_backed_observations: int
    ever_index_backed: bool
    latest_index_status: str
    latest_index_backed: bool
    latest_job_status: str | None
    min_selectivity_fraction: float | None
    mean_selectivity_fraction: float | None
    max_total_rows: int | None
    estimated_unindexed_scan_rows: int
    backend_supports_index: bool
    backend_kinds: tuple[str, ...]
    stale: bool

    @property
    def is_jsonb(self) -> bool:
        return self.column_kind == "jsonb_path"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_predicate_observations(
    observations: Iterable[PredicateObservation],
) -> list[PredicateAggregate]:
    """Group observations by ``(table, column, column_kind)`` into aggregates.

    Deterministic and order-independent for correctness: the "latest" status is
    the observation with the greatest ``sequence`` (ties broken by input order),
    not merely the last one seen.
    """
    grouped: OrderedDict[tuple[str, str, str], list[PredicateObservation]] = OrderedDict()
    for obs in observations:
        grouped.setdefault((obs.table, obs.column, obs.column_kind), []).append(obs)

    aggregates: list[PredicateAggregate] = []
    for (table, column, kind), group in grouped.items():
        latest = max(group, key=lambda o: o.sequence)
        fractions = [o.selectivity_fraction for o in group if o.selectivity_fraction is not None]
        totals = [o.total_rows for o in group if o.total_rows is not None]
        filter_obs = [o for o in group if o.used_in_filter]
        backed = [o for o in group if o.index_backed]
        ever_backed = bool(backed)
        latest_backed = latest.index_backed
        latest_job = latest.job_status
        stale = (ever_backed and not latest_backed) or (
            latest_job in _STALE_JOB_STATES
        ) or latest.index_status == INDEX_FAILED
        estimated = sum(
            (o.total_rows or 0) for o in filter_obs if not o.index_backed
        )
        supports = all(o.backend_supports_index for o in group) and (
            latest.index_status != INDEX_UNSUPPORTED
        )
        backend_kinds = tuple(
            sorted({o.backend_kind for o in group if o.backend_kind is not None})
        )
        aggregates.append(
            PredicateAggregate(
                table=table,
                column=column,
                column_kind=kind,
                observations=len(group),
                filter_observations=len(filter_obs),
                index_backed_observations=len(backed),
                ever_index_backed=ever_backed,
                latest_index_status=latest.index_status,
                latest_index_backed=latest_backed,
                latest_job_status=latest_job,
                min_selectivity_fraction=min(fractions) if fractions else None,
                mean_selectivity_fraction=(sum(fractions) / len(fractions)) if fractions else None,
                max_total_rows=max(totals) if totals else None,
                estimated_unindexed_scan_rows=int(estimated),
                backend_supports_index=supports,
                backend_kinds=backend_kinds,
                stale=stale,
            )
        )
    return aggregates


# --------------------------------------------------------------------------- #
# Recommendation.
# --------------------------------------------------------------------------- #

#: Recommendation actions.
ACTION_CREATE = "create_scalar_index"
ACTION_REFRESH = "refresh_stale_index"
ACTION_PROMOTE_JSONB = "promote_jsonb_column"
ACTION_NONE = "none"

_ACTION_RANK = {
    ACTION_REFRESH: 0,
    ACTION_CREATE: 1,
    ACTION_PROMOTE_JSONB: 2,
    ACTION_NONE: 3,
}

#: Confidence tiers, ordered.
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class RecommendationPolicy:
    """Guardrail thresholds for :func:`recommend_from_aggregates`.

    Defaults are deliberately conservative so a fresh lake never gets spammed with
    index recommendations for small or unselective predicates. ``max_selectivity_fraction``
    is the largest *retained* fraction a predicate may keep and still count as
    selective (0.5 = must drop at least half the rows).
    """

    min_observations: int = 3
    min_total_rows: int = 100_000
    max_selectivity_fraction: float = 0.5
    high_confidence_observations: int = 6
    scalar_index_type: str = "BTREE"

    def __post_init__(self) -> None:
        if self.min_observations < 1:
            raise ValueError("min_observations must be >= 1")
        if self.min_total_rows < 0:
            raise ValueError("min_total_rows must be >= 0")
        if not 0.0 < self.max_selectivity_fraction <= 1.0:
            raise ValueError("max_selectivity_fraction must be in (0, 1]")


@dataclass(frozen=True)
class IndexRecommendation:
    """One recommendation about a predicate, with the evidence behind it."""

    table: str
    column: str
    column_kind: str
    action: str
    confidence: str  # "high" | "medium" | "low"
    auto_applicable: bool
    forceable: bool
    reason: str
    index_type: str | None
    observations: int
    filter_observations: int
    selectivity_fraction: float | None
    estimated_unindexed_scan_rows: int
    backend_kinds: tuple[str, ...]
    guardrails: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tripped_guardrails(
    aggregate: PredicateAggregate, policy: RecommendationPolicy
) -> list[str]:
    tripped: list[str] = []
    if aggregate.observations < policy.min_observations:
        tripped.append("insufficient-observations")
    if (aggregate.max_total_rows or 0) < policy.min_total_rows:
        tripped.append("small-table")
    fraction = aggregate.min_selectivity_fraction
    if fraction is None or fraction > policy.max_selectivity_fraction:
        tripped.append("low-selectivity")
    if aggregate.filter_observations < 1:
        tripped.append("never-used-in-filter")
    return tripped


def _confidence(aggregate: PredicateAggregate, policy: RecommendationPolicy) -> str:
    fraction = aggregate.min_selectivity_fraction
    very_selective = fraction is not None and fraction <= policy.max_selectivity_fraction / 2
    very_repeated = aggregate.observations >= policy.high_confidence_observations
    if very_selective and very_repeated:
        return "high"
    if very_selective or very_repeated:
        return "medium"
    return "low"


def _recommend_one(
    aggregate: PredicateAggregate, policy: RecommendationPolicy
) -> IndexRecommendation:
    common = dict(
        table=aggregate.table,
        column=aggregate.column,
        column_kind=aggregate.column_kind,
        index_type=None if aggregate.is_jsonb else policy.scalar_index_type,
        observations=aggregate.observations,
        filter_observations=aggregate.filter_observations,
        selectivity_fraction=aggregate.min_selectivity_fraction,
        estimated_unindexed_scan_rows=aggregate.estimated_unindexed_scan_rows,
        backend_kinds=aggregate.backend_kinds,
    )
    guardrails = tuple(_tripped_guardrails(aggregate, policy))

    # Backend cannot build scalar indexes at all: never recommend, never forceable.
    if not aggregate.backend_supports_index:
        return IndexRecommendation(
            action=ACTION_NONE,
            confidence="low",
            auto_applicable=False,
            forceable=False,
            reason=(
                "backend cannot build scalar indexes; predicate pushdown remains "
                "the only path (no index recommendation possible)"
            ),
            guardrails=("unsupported-backend", *guardrails),
            **common,
        )

    # Column does not exist -- cannot be indexed.
    if aggregate.latest_index_status == INDEX_ABSENT_COLUMN:
        return IndexRecommendation(
            action=ACTION_NONE,
            confidence="low",
            auto_applicable=False,
            forceable=False,
            reason="column is absent from the table; nothing to index",
            guardrails=("absent-column",),
            **common,
        )

    # JSONB-path predicate: advise promotion, never auto-index.
    if aggregate.is_jsonb:
        if guardrails:
            return IndexRecommendation(
                action=ACTION_NONE,
                confidence="low",
                auto_applicable=False,
                forceable=False,
                reason=(
                    "JSONB-path predicate seen but not hot/selective enough to "
                    "justify a typed-column promotion: " + ", ".join(guardrails)
                ),
                guardrails=guardrails,
                **common,
            )
        return IndexRecommendation(
            action=ACTION_PROMOTE_JSONB,
            confidence=_confidence(aggregate, policy),
            auto_applicable=False,
            forceable=False,
            reason=(
                f"hot, selective JSONB-path predicate {aggregate.column!r}: promote "
                "it to a typed column, then index the typed column (a scalar index "
                "cannot be built on a JSONB sub-path)"
            ),
            guardrails=guardrails,
            **common,
        )

    # Typed column, already covered by a present index or completed job.
    if aggregate.latest_index_backed and not aggregate.stale:
        return IndexRecommendation(
            action=ACTION_NONE,
            confidence="high",
            auto_applicable=False,
            forceable=False,
            reason="already served by a scalar index or a completed index job",
            guardrails=(),
            **common,
        )

    # Typed column whose index went missing / a job failed: refresh it.
    if aggregate.stale:
        return IndexRecommendation(
            action=ACTION_REFRESH,
            confidence=_confidence(aggregate, policy),
            auto_applicable=True,
            forceable=True,
            reason=(
                "an index this predicate previously used is no longer present "
                f"(latest status {aggregate.latest_index_status!r}"
                + (
                    f", job {aggregate.latest_job_status!r}"
                    if aggregate.latest_job_status
                    else ""
                )
                + "); refresh it"
            ),
            guardrails=guardrails,
            **common,
        )

    # Typed, unindexed, buildable: recommend an index unless a guardrail trips.
    if guardrails:
        return IndexRecommendation(
            action=ACTION_NONE,
            confidence="low",
            auto_applicable=False,
            forceable=True,  # an index CAN be built; it just is not recommended
            reason="not recommended by default: " + ", ".join(guardrails),
            guardrails=guardrails,
            **common,
        )
    return IndexRecommendation(
        action=ACTION_CREATE,
        confidence=_confidence(aggregate, policy),
        auto_applicable=True,
        forceable=True,
        reason=(
            f"hot ({aggregate.filter_observations} filtered reads), selective "
            f"(retains {aggregate.min_selectivity_fraction:.3f} of rows), unindexed "
            "typed predicate on a buildable backend"
        ),
        guardrails=(),
        **common,
    )


def _recommendation_sort_key(rec: IndexRecommendation) -> tuple[Any, ...]:
    return (
        _ACTION_RANK.get(rec.action, 9),
        _CONFIDENCE_RANK.get(rec.confidence, 9),
        -rec.estimated_unindexed_scan_rows,
        -rec.observations,
        rec.table,
        rec.column,
    )


def recommend_from_aggregates(
    aggregates: Iterable[PredicateAggregate],
    *,
    policy: RecommendationPolicy | None = None,
) -> list[IndexRecommendation]:
    """Turn aggregates into ranked recommendations (most actionable first)."""
    resolved = policy or RecommendationPolicy()
    recommendations = [_recommend_one(aggregate, resolved) for aggregate in aggregates]
    recommendations.sort(key=_recommendation_sort_key)
    return recommendations


def recommend_from_reports(
    reports: Iterable[Mapping[str, Any]],
    *,
    policy: RecommendationPolicy | None = None,
) -> list[IndexRecommendation]:
    """Convenience: extract -> aggregate -> recommend over an ordered report set."""
    observations = extract_observations_from_reports(reports)
    aggregates = aggregate_predicate_observations(observations)
    return recommend_from_aggregates(aggregates, policy=policy)


_RECOMMENDATION_FIELDS = {f.name for f in _dc_fields(IndexRecommendation)}


def recommendation_from_dict(payload: Mapping[str, Any]) -> IndexRecommendation:
    """Rebuild an :class:`IndexRecommendation` from its ``to_dict`` form.

    Tolerant of extra keys and missing optional fields so a recommendation that
    round-tripped through JSON (CLI, stored report) can be re-applied. Requires at
    least ``table``, ``column``, and ``action``.
    """
    for required in ("table", "column", "action"):
        if payload.get(required) in (None, ""):
            raise ValueError(f"recommendation is missing required field {required!r}")
    kwargs: dict[str, Any] = {}
    for name in _RECOMMENDATION_FIELDS:
        if name in payload:
            kwargs[name] = payload[name]
    kwargs.setdefault("column_kind", "typed")
    kwargs.setdefault("confidence", "low")
    kwargs.setdefault("auto_applicable", False)
    kwargs.setdefault("forceable", False)
    kwargs.setdefault("reason", "")
    kwargs.setdefault("index_type", "BTREE")
    kwargs.setdefault("observations", 0)
    kwargs.setdefault("filter_observations", 0)
    kwargs.setdefault("selectivity_fraction", None)
    kwargs.setdefault("estimated_unindexed_scan_rows", 0)
    kwargs.setdefault("guardrails", ())
    kwargs.setdefault("backend_kinds", ())
    kwargs["guardrails"] = tuple(kwargs["guardrails"])
    kwargs["backend_kinds"] = tuple(kwargs["backend_kinds"])
    return IndexRecommendation(**kwargs)


# --------------------------------------------------------------------------- #
# Apply (route safe recommendations into the 0136 job lifecycle).
# --------------------------------------------------------------------------- #

#: Per-recommendation apply outcomes.
OUTCOME_APPLIED = "applied"  # an index job was requested
OUTCOME_SUPPRESSED = "suppressed"  # explicitly suppressed by the caller
OUTCOME_ADVISORY = "advisory"  # surfaced only (e.g. JSONB promotion)
OUTCOME_SKIPPED = "skipped"  # not auto-applicable / below confidence bar


def _should_apply(
    rec: IndexRecommendation,
    *,
    force: bool,
    min_confidence: str,
) -> bool:
    if rec.action in {ACTION_CREATE, ACTION_REFRESH} and rec.auto_applicable:
        if _CONFIDENCE_RANK.get(rec.confidence, 9) <= _CONFIDENCE_RANK.get(min_confidence, 0):
            return True
        return force
    # A guardrail-suppressed typed candidate can be forced.
    if force and rec.forceable and rec.action == ACTION_NONE:
        return True
    return False


def apply_index_recommendations(
    lake: Any,
    recommendations: Iterable[IndexRecommendation],
    *,
    force: bool = False,
    suppress: Sequence[str] | None = None,
    min_confidence: str = "high",
    replace: bool = False,
) -> list[dict[str, Any]]:
    """Route safe recommendations into the durable 0136 scalar-index job store.

    Only typed ``create_scalar_index`` / ``refresh_stale_index`` recommendations
    are ever applied, via :func:`lancedb_robotics.scalar_index_jobs.request_scalar_index`
    (idempotent, capability-aware, reversible). JSONB promotions are advisory. A
    refresh applies with ``replace=True`` (rebuild the stale index). ``force``
    lifts the confidence bar and applies guardrail-suppressed *forceable* columns;
    ``suppress`` (``"table.column"`` or ``"column"``) always wins.

    Returns one auditable outcome record per recommendation -- never mutates data
    and never raises for a normal per-column failure (that becomes the job's
    ``skipped``/``failed`` status).
    """
    from lancedb_robotics.scalar_index_jobs import request_scalar_index

    suppressed = set(suppress or ())
    outcomes: list[dict[str, Any]] = []
    for rec in recommendations:
        keys = {f"{rec.table}.{rec.column}", rec.column}
        if keys & suppressed:
            outcomes.append(_outcome(rec, OUTCOME_SUPPRESSED, reason="suppressed by caller"))
            continue
        if rec.action == ACTION_PROMOTE_JSONB:
            outcomes.append(
                _outcome(
                    rec,
                    OUTCOME_ADVISORY,
                    reason="JSONB-path promotion is advisory; requires a schema change",
                )
            )
            continue
        if not _should_apply(rec, force=force, min_confidence=min_confidence):
            outcomes.append(
                _outcome(
                    rec,
                    OUTCOME_SKIPPED,
                    reason=(
                        "not auto-applied (action "
                        f"{rec.action!r}, confidence {rec.confidence!r}; "
                        "pass force=True to override)"
                    ),
                )
            )
            continue
        try:
            result = request_scalar_index(
                lake,
                table=rec.table,
                column=rec.column,
                index_type=rec.index_type or "BTREE",
                replace=replace or rec.action == ACTION_REFRESH,
            )
        except Exception as exc:  # noqa: BLE001 - one column's failure never aborts the batch
            outcomes.append(
                _outcome(rec, OUTCOME_SKIPPED, reason=f"index-job request failed: {exc}")
            )
            continue
        outcomes.append(
            _outcome(
                rec,
                OUTCOME_APPLIED,
                reason=f"requested scalar-index job ({result.job.status})",
                job=result.to_dict(),
            )
        )
    return outcomes


def _outcome(
    rec: IndexRecommendation,
    outcome: str,
    *,
    reason: str,
    job: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "reason": reason,
        "recommendation": rec.to_dict(),
        "job": dict(job) if job is not None else None,
    }


__all__ = [
    "ACTION_CREATE",
    "ACTION_NONE",
    "ACTION_PROMOTE_JSONB",
    "ACTION_REFRESH",
    "INDEX_ABSENT_COLUMN",
    "INDEX_FAILED",
    "INDEX_PRESENT",
    "INDEX_UNINDEXED",
    "INDEX_UNSUPPORTED",
    "IndexRecommendation",
    "OUTCOME_ADVISORY",
    "OUTCOME_APPLIED",
    "OUTCOME_SKIPPED",
    "OUTCOME_SUPPRESSED",
    "PredicateAggregate",
    "PredicateObservation",
    "RecommendationPolicy",
    "aggregate_predicate_observations",
    "apply_index_recommendations",
    "build_predicate_telemetry",
    "extract_observations_from_reports",
    "extract_predicate_observations",
    "normalize_index_status",
    "recent_aligned_reports",
    "recommend_from_aggregates",
    "recommend_from_reports",
    "recommendation_from_dict",
]
