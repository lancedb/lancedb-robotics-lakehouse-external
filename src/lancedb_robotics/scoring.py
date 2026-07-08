"""Pluggable active-learning scorers and score calibration.

Backlog 0092 builds on the deterministic active-learning queue from 0056. The
0056 queue hard-codes a max-over-signals heuristic (low confidence, entropy,
loss, disagreement, missing labels). Real curation teams want calibrated,
task-specific scoring policies that combine model uncertainty, failure impact,
label cost, diversity, and distribution gaps without rewriting queue creation.

This module is intentionally storage-agnostic. ``curate.py`` reads Lance tables
and assembles immutable :class:`ScoringCandidate` bundles; scorers only ever see
those bundles, so they are pure, deterministic, and unit-testable without a lake.

Public surface:

- :class:`Calibration` -- how ``model_outputs.score`` is interpreted (confidence,
  probability, loss, or raw) before any scorer runs.
- :class:`ScoringCandidate` / :class:`ModelOutputSignal` -- per-scenario signals.
- :class:`ScorerResult` -- a scorer's score, reason, and component breakdown.
- :class:`ActiveLearningScorer` -- the scorer base class.
- Built-in scorers: :class:`DefaultUncertaintyScorer` (reproduces 0056),
  :class:`ConfidenceMarginScorer`, :class:`EntropyScorer`,
  :class:`EnsembleDisagreementScorer`, :class:`HighLossScorer`,
  :class:`MissingLabelsScorer`, :class:`FailureSeverityScorer`,
  :class:`DistributionGapBoostScorer`, :class:`LabelCostPenaltyScorer`.
- :class:`CompositeScorer` -- a weighted blend of sub-scorers.
- :func:`register_scorer` / :func:`resolve_scorer` / :func:`builtin_scorer_names`
  -- a process-local registry so a scorer can be created and used without
  editing core queue code.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class ScoringError(ValueError):
    """Raised when a scorer or calibration is mis-configured."""


_CALIBRATION_MODES = ("confidence", "probability", "loss", "raw")
_DEFAULT_SEVERITY_MAP: dict[str, float] = {
    "none": 0.0,
    "info": 0.1,
    "low": 0.25,
    "minor": 0.25,
    "medium": 0.5,
    "moderate": 0.5,
    "high": 0.75,
    "major": 0.75,
    "critical": 1.0,
    "severe": 1.0,
    "blocker": 1.0,
}


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """Explicitly interpret a raw ``model_outputs.score`` as an uncertainty.

    Calibration is a queue-level, recorded parameter. Switching ``mode`` changes
    how a numeric score maps to "how informative is this candidate" *explicitly*
    (the mode is persisted in transform lineage and queue source refs) rather
    than silently re-ranking behind an unchanged knob.

    Modes:

    - ``confidence`` (default): ``score`` is the model's confidence in its top
      prediction; uncertainty is ``1 - clamp01(score)`` (the 0056 behavior).
    - ``probability``: ``score`` is the probability of the predicted class;
      uncertainty peaks at ``0.5`` via ``1 - |2*clamp01(score) - 1|``.
    - ``loss``: ``score`` is an error/loss where larger is more informative;
      uncertainty is ``score / loss_scale`` (or ``score`` when no scale is set).
    - ``raw``: ``score`` is already a scorer output; it is used unchanged.
    """

    mode: str = "confidence"
    score_field: str = "score"
    loss_scale: float | None = None
    reference: str = ""

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower().replace("_", "-")
        if mode not in _CALIBRATION_MODES:
            raise ScoringError(
                f"unknown calibration mode {self.mode!r}; expected one of "
                f"{', '.join(_CALIBRATION_MODES)}"
            )
        object.__setattr__(self, "mode", mode)
        if self.loss_scale is not None:
            scale = float(self.loss_scale)
            if scale <= 0.0:
                raise ScoringError("calibration loss_scale must be positive")
            object.__setattr__(self, "loss_scale", scale)

    def uncertainty(self, score: Any) -> float | None:
        """Map a raw score to an informativeness value, or ``None`` if absent."""
        numeric = _optional_float(score)
        if numeric is None:
            return None
        if self.mode == "confidence":
            return 1.0 - _clamp01(numeric)
        if self.mode == "probability":
            return 1.0 - abs(2.0 * _clamp01(numeric) - 1.0)
        if self.mode == "loss":
            return numeric / self.loss_scale if self.loss_scale else numeric
        return numeric

    def reason(self, score: Any) -> str:
        numeric = _optional_float(score)
        rendered = "n/a" if numeric is None else f"{numeric:.6f}"
        if self.mode == "confidence":
            return f"low-confidence score={rendered}"
        if self.mode == "probability":
            return f"calibrated-probability score={rendered}"
        if self.mode == "loss":
            return f"calibrated-loss score={rendered}"
        return f"calibrated-raw score={rendered}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "score_field": self.score_field,
            "loss_scale": self.loss_scale,
            "reference": self.reference,
        }


def resolve_calibration(spec: Calibration | Mapping[str, Any] | str | None) -> Calibration:
    """Coerce a calibration spec (mode string / mapping / instance) to a Calibration."""
    if spec is None:
        return Calibration()
    if isinstance(spec, Calibration):
        return spec
    if isinstance(spec, str):
        return Calibration(mode=spec)
    if isinstance(spec, Mapping):
        allowed = {"mode", "score_field", "loss_scale", "reference"}
        unknown = set(spec) - allowed
        if unknown:
            raise ScoringError(
                f"unknown calibration fields {sorted(unknown)}; expected subset of {sorted(allowed)}"
            )
        return Calibration(**{key: spec[key] for key in spec})
    raise ScoringError(f"cannot interpret calibration spec {spec!r}")


# ---------------------------------------------------------------------------
# Candidate / result data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelOutputSignal:
    """One ``model_outputs`` row reduced to the fields scorers read."""

    model_output_id: str
    model_version: str = ""
    output_type: str = ""
    prediction: str = ""
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoringCandidate:
    """Immutable per-scenario signal bundle handed to every scorer.

    ``calibration`` is attached so scorers interpret raw scores consistently
    without each one re-deriving the mode.
    """

    scenario_id: str
    outputs: tuple[ModelOutputSignal, ...] = ()
    label_types: frozenset[str] = frozenset()
    required_label_types: tuple[str, ...] = ()
    slice_label: str = ""
    slice_needed: int = 0
    calibration: Calibration = field(default_factory=Calibration)

    @property
    def missing_label_types(self) -> tuple[str, ...]:
        return tuple(lt for lt in self.required_label_types if lt not in self.label_types)


@dataclass(frozen=True)
class ScorerResult:
    """A scorer's verdict for one candidate."""

    score: float
    reason: str
    metric: str = ""
    model_output_id: str = ""
    components: Mapping[str, float] = field(default_factory=dict)
    detail: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scorer base class + built-ins
# ---------------------------------------------------------------------------


class ActiveLearningScorer:
    """Base class for active-learning scorers.

    Subclasses set ``name``/``version`` and implement :meth:`score`. ``score``
    returns a :class:`ScorerResult` or ``None`` to abstain (the candidate is
    dropped unless another scorer in a composite contributes).
    """

    name: str = "scorer"
    version: str = "0"

    def params(self) -> dict[str, Any]:
        return {}

    def score(self, candidate: ScoringCandidate) -> ScorerResult | None:  # pragma: no cover
        raise NotImplementedError

    def descriptor(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "params": self.params()}


def _best_over_outputs(
    candidate: ScoringCandidate,
    value_fn: Callable[[ModelOutputSignal], tuple[float, str] | None],
) -> tuple[float, str, str] | None:
    """Pick the highest (value, reason, model_output_id) across outputs.

    Outputs are visited in ``model_output_id`` order and ties favor the
    lexicographically smallest id, so the result is fully deterministic.
    """

    best: tuple[float, str, str] | None = None
    for output in sorted(candidate.outputs, key=lambda o: o.model_output_id):
        produced = value_fn(output)
        if produced is None:
            continue
        value, reason = produced
        contender = (float(value), reason, output.model_output_id)
        if best is None or contender[0] > best[0] or (
            contender[0] == best[0] and contender[2] < best[2]
        ):
            best = contender
    return best


class ConfidenceMarginScorer(ActiveLearningScorer):
    """Score by model uncertainty derived from the calibrated score."""

    name = "confidence-margin"
    version = "1"

    def __init__(self, *, max_confidence_score: float | None = None) -> None:
        self.max_confidence_score = (
            None if max_confidence_score is None else float(max_confidence_score)
        )

    def params(self) -> dict[str, Any]:
        return {"max_confidence_score": self.max_confidence_score}

    def score(self, candidate: ScoringCandidate) -> ScorerResult | None:
        cap = self.max_confidence_score

        def value_fn(output: ModelOutputSignal) -> tuple[float, str] | None:
            if output.score is None:
                return None
            if cap is not None and float(output.score) > cap:
                return None
            uncertainty = candidate.calibration.uncertainty(output.score)
            if uncertainty is None:
                return None
            return uncertainty, candidate.calibration.reason(output.score)

        best = _best_over_outputs(candidate, value_fn)
        if best is None:
            return None
        return ScorerResult(
            score=best[0], reason=best[1], metric="score", model_output_id=best[2]
        )


class _MetadataMaxScorer(ActiveLearningScorer):
    """Shared base for scorers that read one numeric metadata key per output."""

    metadata_key = "value"

    def __init__(self, *, metadata_key: str | None = None) -> None:
        if metadata_key:
            self.metadata_key = str(metadata_key)

    def params(self) -> dict[str, Any]:
        return {"metadata_key": self.metadata_key}

    def score(self, candidate: ScoringCandidate) -> ScorerResult | None:
        key = self.metadata_key

        def value_fn(output: ModelOutputSignal) -> tuple[float, str] | None:
            numeric = _optional_float(output.metadata.get(key))
            if numeric is None:
                return None
            return numeric, f"{key}={numeric:.6f}"

        best = _best_over_outputs(candidate, value_fn)
        if best is None:
            return None
        return ScorerResult(
            score=best[0], reason=best[1], metric=key, model_output_id=best[2]
        )


class EntropyScorer(_MetadataMaxScorer):
    """Prioritize high predictive entropy recorded in output metadata."""

    name = "entropy"
    version = "1"
    metadata_key = "entropy"


class HighLossScorer(_MetadataMaxScorer):
    """Prioritize high training/eval loss recorded in output metadata.

    ``loss_scale`` normalizes raw loss into a comparable range; the unscaled
    loss is used when no scale is provided.
    """

    name = "high-loss"
    version = "1"
    metadata_key = "loss"

    def __init__(self, *, metadata_key: str | None = None, loss_scale: float | None = None) -> None:
        super().__init__(metadata_key=metadata_key)
        if loss_scale is not None:
            scale = float(loss_scale)
            if scale <= 0.0:
                raise ScoringError("high-loss loss_scale must be positive")
            self.loss_scale = scale
        else:
            self.loss_scale = None

    def params(self) -> dict[str, Any]:
        return {"metadata_key": self.metadata_key, "loss_scale": self.loss_scale}

    def score(self, candidate: ScoringCandidate) -> ScorerResult | None:
        key = self.metadata_key
        scale = self.loss_scale

        def value_fn(output: ModelOutputSignal) -> tuple[float, str] | None:
            numeric = _optional_float(output.metadata.get(key))
            if numeric is None:
                return None
            value = numeric / scale if scale else numeric
            return value, f"{key}={numeric:.6f}"

        best = _best_over_outputs(candidate, value_fn)
        if best is None:
            return None
        return ScorerResult(
            score=best[0], reason=best[1], metric=key, model_output_id=best[2]
        )


class EnsembleDisagreementScorer(ActiveLearningScorer):
    """Score by disagreement across multiple ``model_outputs`` rows.

    Two complementary signals, whichever yields a value:

    1. A per-member numeric field (``member_field``, default ``member_score``)
       read from each output's metadata, aggregated across the scenario's
       outputs into a ``range`` or ``variance`` statistic. This is what
       "consume metadata fields from multiple model_outputs rows" means.
    2. An explicit precomputed ``disagreement`` metadata value on any output.

    The larger of the two is reported, so a single row carrying a precomputed
    disagreement still works while ensembles get true cross-row spread.
    """

    name = "ensemble-disagreement"
    version = "1"

    def __init__(
        self,
        *,
        member_field: str = "member_score",
        statistic: str = "range",
        disagreement_key: str = "disagreement",
    ) -> None:
        statistic = str(statistic).strip().lower()
        if statistic not in ("range", "variance"):
            raise ScoringError("ensemble disagreement statistic must be 'range' or 'variance'")
        self.member_field = str(member_field)
        self.statistic = statistic
        self.disagreement_key = str(disagreement_key)

    def params(self) -> dict[str, Any]:
        return {
            "member_field": self.member_field,
            "statistic": self.statistic,
            "disagreement_key": self.disagreement_key,
        }

    def score(self, candidate: ScoringCandidate) -> ScorerResult | None:
        members: list[float] = []
        for output in sorted(candidate.outputs, key=lambda o: o.model_output_id):
            numeric = _optional_float(output.metadata.get(self.member_field))
            if numeric is not None:
                members.append(numeric)

        spread: float | None = None
        if len(members) >= 2:
            if self.statistic == "range":
                spread = max(members) - min(members)
            else:
                mean = sum(members) / len(members)
                spread = sum((value - mean) ** 2 for value in members) / len(members)

        precomputed = _best_over_outputs(
            candidate,
            lambda output: (
                (value, f"{self.disagreement_key}={value:.6f}")
                if (value := _optional_float(output.metadata.get(self.disagreement_key)))
                is not None
                else None
            ),
        )

        if spread is None and precomputed is None:
            return None
        if spread is not None and (precomputed is None or spread >= precomputed[0]):
            return ScorerResult(
                score=spread,
                reason=f"ensemble-disagreement {self.statistic}={spread:.6f} members={len(members)}",
                metric="ensemble-disagreement",
                detail={"members": len(members), "statistic": self.statistic},
            )
        return ScorerResult(
            score=precomputed[0],
            reason=precomputed[1],
            metric="ensemble-disagreement",
            model_output_id=precomputed[2],
        )


class MissingLabelsScorer(ActiveLearningScorer):
    """Prioritize scenarios missing required label types."""

    name = "missing-labels"
    version = "1"

    def __init__(self, *, weight: float = 1.0, proportional: bool = False) -> None:
        self.weight = float(weight)
        self.proportional = bool(proportional)

    def params(self) -> dict[str, Any]:
        return {"weight": self.weight, "proportional": self.proportional}

    def score(self, candidate: ScoringCandidate) -> ScorerResult | None:
        required = candidate.required_label_types
        if not required:
            return None
        missing = candidate.missing_label_types
        if not missing:
            return None
        if self.proportional:
            value = self.weight * (len(missing) / len(required))
        else:
            value = self.weight
        return ScorerResult(
            score=value,
            reason="missing-labels:" + ",".join(missing),
            metric="missing-labels",
            detail={"missing_label_types": list(missing)},
        )


class FailureSeverityScorer(ActiveLearningScorer):
    """Prioritize by failure severity / incident impact in output metadata.

    Textual severities (``low``/``high``/``critical`` ...) map through
    ``severity_map``; a numeric ``impact_key`` value is used directly when
    present and overrides the textual mapping when larger.
    """

    name = "failure-severity"
    version = "1"

    def __init__(
        self,
        *,
        severity_key: str = "severity",
        impact_key: str = "incident_impact",
        severity_map: Mapping[str, float] | None = None,
    ) -> None:
        self.severity_key = str(severity_key)
        self.impact_key = str(impact_key)
        self.severity_map = {
            str(key).strip().lower(): float(value)
            for key, value in (severity_map or _DEFAULT_SEVERITY_MAP).items()
        }

    def params(self) -> dict[str, Any]:
        return {
            "severity_key": self.severity_key,
            "impact_key": self.impact_key,
            "severity_map": dict(sorted(self.severity_map.items())),
        }

    def score(self, candidate: ScoringCandidate) -> ScorerResult | None:
        def value_fn(output: ModelOutputSignal) -> tuple[float, str] | None:
            best_value: float | None = None
            reason = ""
            raw_severity = output.metadata.get(self.severity_key)
            if raw_severity is not None:
                key = str(raw_severity).strip().lower()
                if key in self.severity_map:
                    best_value = self.severity_map[key]
                    reason = f"severity={key}({best_value:.3f})"
            impact = _optional_float(output.metadata.get(self.impact_key))
            if impact is not None and (best_value is None or impact > best_value):
                best_value = impact
                reason = f"incident-impact={impact:.6f}"
            if best_value is None:
                return None
            return best_value, reason

        best = _best_over_outputs(candidate, value_fn)
        if best is None:
            return None
        return ScorerResult(
            score=best[0], reason=best[1], metric="failure-severity", model_output_id=best[2]
        )


class DistributionGapBoostScorer(ActiveLearningScorer):
    """Boost candidates that fall in underrepresented distribution slices.

    The boost is supplied per-candidate via ``slice_needed`` (how many more
    samples the candidate's slice needs). ``normalize`` divides the deficit by
    ``target`` so a deeper gap ranks higher; otherwise a flat ``boost`` applies.
    """

    name = "distribution-gap-boost"
    version = "1"

    def __init__(self, *, boost: float = 1.0, normalize: bool = False, target: float = 1.0) -> None:
        self.boost = float(boost)
        self.normalize = bool(normalize)
        target = float(target)
        if target <= 0.0:
            raise ScoringError("distribution-gap-boost target must be positive")
        self.target = target

    def params(self) -> dict[str, Any]:
        return {"boost": self.boost, "normalize": self.normalize, "target": self.target}

    def score(self, candidate: ScoringCandidate) -> ScorerResult | None:
        if candidate.slice_needed <= 0:
            return None
        if self.normalize:
            value = self.boost * min(1.0, candidate.slice_needed / self.target)
        else:
            value = self.boost
        slice_label = candidate.slice_label or "(unlabeled-slice)"
        return ScorerResult(
            score=value,
            reason=f"distribution-gap slice={slice_label} needed={candidate.slice_needed}",
            metric="distribution-gap-boost",
            detail={"slice": candidate.slice_label, "needed": candidate.slice_needed},
        )


class LabelCostPenaltyScorer(ActiveLearningScorer):
    """A penalty term: returns a negative score proportional to labeling cost.

    Useful only inside a :class:`CompositeScorer`, where it subtracts budget
    cost from the blended priority. Cost comes from output metadata
    (``cost_key``) or the constant ``cost`` fallback.
    """

    name = "label-cost"
    version = "1"

    def __init__(self, *, cost: float = 1.0, weight: float = 1.0, cost_key: str = "label_cost") -> None:
        self.cost = float(cost)
        self.weight = float(weight)
        self.cost_key = str(cost_key)

    def params(self) -> dict[str, Any]:
        return {"cost": self.cost, "weight": self.weight, "cost_key": self.cost_key}

    def score(self, candidate: ScoringCandidate) -> ScorerResult | None:
        cost: float | None = None
        model_output_id = ""
        for output in sorted(candidate.outputs, key=lambda o: o.model_output_id):
            numeric = _optional_float(output.metadata.get(self.cost_key))
            if numeric is not None:
                cost = numeric
                model_output_id = output.model_output_id
                break
        if cost is None:
            cost = self.cost
        penalty = -self.weight * cost
        return ScorerResult(
            score=penalty,
            reason=f"label-cost penalty={penalty:.6f}",
            metric="label-cost",
            model_output_id=model_output_id,
        )


class DefaultUncertaintyScorer(ActiveLearningScorer):
    """Reproduce the 0056 max-over-signals heuristic.

    For each output, the calibrated score plus the ``uncertainty``/``entropy``/
    ``loss``/``disagreement`` metadata values are candidate signals; missing
    required labels contribute a flat ``1.0``. The highest signal wins, with
    deterministic tie-breaks. This is the default so existing active-learning
    queues keep their behavior when no explicit scorer is requested.
    """

    name = "uncertainty"
    version = "0056"
    _METADATA_KEYS = ("uncertainty", "entropy", "loss", "disagreement")

    def __init__(self, *, max_confidence_score: float | None = None) -> None:
        self.max_confidence_score = (
            None if max_confidence_score is None else float(max_confidence_score)
        )

    def params(self) -> dict[str, Any]:
        return {"max_confidence_score": self.max_confidence_score}

    def score(self, candidate: ScoringCandidate) -> ScorerResult | None:
        cap = self.max_confidence_score
        # (value, metric, reason, model_output_id); ties favor larger metric
        # name then smaller model_output_id, matching the 0056 ordering.
        signals: list[tuple[float, str, str, str]] = []
        for output in sorted(candidate.outputs, key=lambda o: o.model_output_id):
            if output.score is not None and (cap is None or float(output.score) <= cap):
                uncertainty = candidate.calibration.uncertainty(output.score)
                if uncertainty is not None:
                    signals.append(
                        (uncertainty, "score", candidate.calibration.reason(output.score), output.model_output_id)
                    )
            for key in self._METADATA_KEYS:
                numeric = _optional_float(output.metadata.get(key))
                if numeric is not None:
                    signals.append((numeric, key, f"{key}={numeric:.6f}", output.model_output_id))

        missing = candidate.missing_label_types
        if candidate.required_label_types and missing:
            signals.append((1.0, "missing-labels", "missing-labels:" + ",".join(missing), ""))

        if not signals:
            return None
        best = max(signals, key=lambda s: (s[0], s[1], _NegStr(s[3])))
        return ScorerResult(
            score=best[0], reason=best[2], metric=best[1], model_output_id=best[3]
        )


class _NegStr:
    """Reverse-ordering wrapper so a ``max`` tie favors the smaller string."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __lt__(self, other: _NegStr) -> bool:
        return self.value > other.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _NegStr) and self.value == other.value


class CompositeScorer(ActiveLearningScorer):
    """Blend multiple scorers with weights.

    ``combine='sum'`` (default) adds weighted contributions; ``combine='max'``
    takes the largest weighted contribution. A candidate is scored only if at
    least one component contributes (non-``None``).
    """

    version = "1"

    def __init__(
        self,
        components: Sequence[tuple[ActiveLearningScorer, float] | ActiveLearningScorer],
        *,
        name: str = "composite",
        combine: str = "sum",
    ) -> None:
        combine = str(combine).strip().lower()
        if combine not in ("sum", "max"):
            raise ScoringError("composite combine must be 'sum' or 'max'")
        normalized: list[tuple[ActiveLearningScorer, float]] = []
        for component in components:
            if isinstance(component, tuple):
                scorer, weight = component
            else:
                scorer, weight = component, 1.0
            if not isinstance(scorer, ActiveLearningScorer):
                raise ScoringError(f"composite component is not a scorer: {scorer!r}")
            normalized.append((scorer, float(weight)))
        if not normalized:
            raise ScoringError("composite scorer requires at least one component")
        self.name = name
        self.combine = combine
        self.components = tuple(normalized)

    def params(self) -> dict[str, Any]:
        return {
            "combine": self.combine,
            "components": [
                {**scorer.descriptor(), "weight": weight}
                for scorer, weight in self.components
            ],
        }

    def score(self, candidate: ScoringCandidate) -> ScorerResult | None:
        contributions: dict[str, float] = {}
        reasons: list[str] = []
        total: float | None = None
        for scorer, weight in self.components:
            result = scorer.score(candidate)
            if result is None:
                continue
            contribution = weight * result.score
            contributions[scorer.name] = contribution
            reasons.append(f"{scorer.name}={result.score:.6f}*{weight:g}")
            if total is None:
                total = contribution
            elif self.combine == "sum":
                total += contribution
            else:
                total = max(total, contribution)
        if total is None:
            return None
        return ScorerResult(
            score=total,
            reason="; ".join(reasons),
            metric=self.name,
            components=contributions,
            detail={"combine": self.combine},
        )


# ---------------------------------------------------------------------------
# Registry + resolution
# ---------------------------------------------------------------------------


_BUILTIN_FACTORIES: dict[str, Callable[..., ActiveLearningScorer]] = {
    "uncertainty": DefaultUncertaintyScorer,
    "confidence-margin": ConfidenceMarginScorer,
    "entropy": EntropyScorer,
    "high-loss": HighLossScorer,
    "ensemble-disagreement": EnsembleDisagreementScorer,
    "missing-labels": MissingLabelsScorer,
    "failure-severity": FailureSeverityScorer,
    "distribution-gap-boost": DistributionGapBoostScorer,
    "label-cost": LabelCostPenaltyScorer,
}

_REGISTRY: dict[str, Callable[..., ActiveLearningScorer]] = dict(_BUILTIN_FACTORIES)


def builtin_scorer_names() -> tuple[str, ...]:
    """Names of the built-in scorers shipped with the SDK."""
    return tuple(sorted(_BUILTIN_FACTORIES))


def registered_scorer_names() -> tuple[str, ...]:
    """All currently-resolvable scorer names (built-in plus registered)."""
    return tuple(sorted(_REGISTRY))


def register_scorer(
    name: str,
    factory: Callable[..., ActiveLearningScorer] | ActiveLearningScorer,
    *,
    overwrite: bool = False,
) -> None:
    """Register a scorer factory (or a singleton instance) under ``name``.

    ``factory`` may be a callable taking keyword params, an
    :class:`ActiveLearningScorer` subclass, or a ready instance (which is then
    returned ignoring params). Built-in names cannot be overwritten.
    """
    key = str(name).strip().lower()
    if not key:
        raise ScoringError("scorer name must not be empty")
    if key in _BUILTIN_FACTORIES and not overwrite:
        raise ScoringError(f"cannot overwrite built-in scorer {key!r}")
    if key in _REGISTRY and key not in _BUILTIN_FACTORIES and not overwrite:
        raise ScoringError(f"scorer {key!r} already registered; pass overwrite=True to replace")
    if isinstance(factory, ActiveLearningScorer):
        instance = factory

        def _singleton(**params: Any) -> ActiveLearningScorer:
            if params:
                raise ScoringError(
                    f"scorer {key!r} is registered as a fixed instance and takes no params"
                )
            return instance

        _REGISTRY[key] = _singleton
    elif callable(factory):
        _REGISTRY[key] = factory
    else:
        raise ScoringError(f"scorer factory for {key!r} must be callable or a scorer instance")


def unregister_scorer(name: str) -> None:
    """Remove a non-built-in scorer from the registry (mainly for tests)."""
    key = str(name).strip().lower()
    if key in _BUILTIN_FACTORIES:
        raise ScoringError(f"cannot unregister built-in scorer {key!r}")
    _REGISTRY.pop(key, None)


ScorerSpec = "ActiveLearningScorer | str | Sequence[Any] | None"


def resolve_scorer(
    spec: Any = None,
    *,
    params: Mapping[str, Any] | None = None,
) -> ActiveLearningScorer:
    """Resolve a scorer spec into an :class:`ActiveLearningScorer`.

    Accepts:

    - ``None`` -> :class:`DefaultUncertaintyScorer` (plus any ``params``).
    - an :class:`ActiveLearningScorer` instance -> returned unchanged.
    - a ``str`` -> looked up in the registry and built with ``params``.
    - a sequence of ``(spec, weight)`` / ``spec`` / scorer -> a
      :class:`CompositeScorer` (composite-level ``params`` may set ``combine``
      and ``name``).
    """
    params = dict(params or {})
    if spec is None:
        return DefaultUncertaintyScorer(**params)
    if isinstance(spec, ActiveLearningScorer):
        if params:
            raise ScoringError("scorer instance was supplied; per-name params are not applied")
        return spec
    if isinstance(spec, str):
        key = spec.strip().lower()
        factory = _REGISTRY.get(key)
        if factory is None:
            raise ScoringError(
                f"unknown scorer {spec!r}; known scorers: {', '.join(registered_scorer_names())}"
            )
        return factory(**params)
    if isinstance(spec, Sequence):
        combine = str(params.pop("combine", "sum"))
        name = str(params.pop("name", "composite"))
        if params:
            raise ScoringError(
                f"composite params only accept 'combine'/'name'; got {sorted(params)}"
            )
        components: list[tuple[ActiveLearningScorer, float]] = []
        for item in spec:
            if isinstance(item, tuple):
                inner, weight = item[0], (item[1] if len(item) > 1 else 1.0)
            else:
                inner, weight = item, 1.0
            components.append((resolve_scorer(inner), float(weight)))
        return CompositeScorer(components, name=name, combine=combine)
    raise ScoringError(f"cannot resolve scorer spec {spec!r}")


__all__ = [
    "ScoringError",
    "Calibration",
    "resolve_calibration",
    "ModelOutputSignal",
    "ScoringCandidate",
    "ScorerResult",
    "ActiveLearningScorer",
    "ConfidenceMarginScorer",
    "EntropyScorer",
    "HighLossScorer",
    "EnsembleDisagreementScorer",
    "MissingLabelsScorer",
    "FailureSeverityScorer",
    "DistributionGapBoostScorer",
    "LabelCostPenaltyScorer",
    "DefaultUncertaintyScorer",
    "CompositeScorer",
    "register_scorer",
    "unregister_scorer",
    "resolve_scorer",
    "builtin_scorer_names",
    "registered_scorer_names",
]
