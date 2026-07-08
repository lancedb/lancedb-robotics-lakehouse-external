"""Pure unit tests for the pluggable active-learning scoring module (0092)."""

import pytest

from lancedb_robotics.scoring import (
    ActiveLearningScorer,
    Calibration,
    CompositeScorer,
    ConfidenceMarginScorer,
    DefaultUncertaintyScorer,
    DistributionGapBoostScorer,
    EnsembleDisagreementScorer,
    EntropyScorer,
    FailureSeverityScorer,
    HighLossScorer,
    LabelCostPenaltyScorer,
    MissingLabelsScorer,
    ModelOutputSignal,
    ScorerResult,
    ScoringCandidate,
    ScoringError,
    builtin_scorer_names,
    register_scorer,
    resolve_calibration,
    resolve_scorer,
    unregister_scorer,
)


def _candidate(scenario_id="scn", outputs=(), **kwargs):
    return ScoringCandidate(scenario_id=scenario_id, outputs=tuple(outputs), **kwargs)


def _output(model_output_id="out", *, score=None, metadata=None):
    return ModelOutputSignal(model_output_id=model_output_id, score=score, metadata=metadata or {})


def test_calibration_modes_interpret_score_differently():
    assert Calibration("confidence").uncertainty(0.1) == pytest.approx(0.9)
    assert Calibration("probability").uncertainty(0.5) == pytest.approx(1.0)
    assert Calibration("probability").uncertainty(1.0) == pytest.approx(0.0)
    assert Calibration("loss").uncertainty(2.0) == pytest.approx(2.0)
    assert Calibration("loss", loss_scale=4.0).uncertainty(2.0) == pytest.approx(0.5)
    assert Calibration("raw").uncertainty(-3.0) == pytest.approx(-3.0)
    assert Calibration("confidence").uncertainty(None) is None


def test_calibration_validation_and_resolution():
    with pytest.raises(ScoringError):
        Calibration("nonsense")
    with pytest.raises(ScoringError):
        Calibration("loss", loss_scale=0.0)
    assert resolve_calibration("loss").mode == "loss"
    assert resolve_calibration({"mode": "raw"}).mode == "raw"
    assert resolve_calibration(None).mode == "confidence"
    with pytest.raises(ScoringError):
        resolve_calibration({"bogus": 1})


def test_default_uncertainty_scorer_reproduces_max_over_signals():
    scorer = DefaultUncertaintyScorer()
    candidate = _candidate(
        outputs=[
            _output("out-1", score=0.2),
            _output("out-2", metadata={"entropy": 0.95, "loss": 0.30}),
        ]
    )
    result = scorer.score(candidate)
    # entropy 0.95 beats 1-0.2=0.8 confidence and 0.30 loss
    assert result.score == pytest.approx(0.95)
    assert result.metric == "entropy"


def test_default_uncertainty_scorer_missing_labels_wins():
    scorer = DefaultUncertaintyScorer()
    candidate = _candidate(
        outputs=[_output("out-1", score=0.4)],
        required_label_types=("bbox", "mask"),
        label_types=frozenset({"bbox"}),
    )
    result = scorer.score(candidate)
    assert result.score == pytest.approx(1.0)
    assert result.reason == "missing-labels:mask"


def test_confidence_margin_respects_calibration_and_cap():
    candidate = _candidate(
        outputs=[_output("out-1", score=0.1)],
        calibration=Calibration("loss"),
    )
    assert ConfidenceMarginScorer().score(candidate).score == pytest.approx(0.1)
    capped = _candidate(outputs=[_output("out-1", score=0.9)])
    assert ConfidenceMarginScorer(max_confidence_score=0.5).score(capped) is None


def test_entropy_and_high_loss_scorers_read_metadata():
    candidate = _candidate(
        outputs=[
            _output("out-1", metadata={"entropy": 0.3, "loss": 8.0}),
            _output("out-2", metadata={"entropy": 0.7, "loss": 2.0}),
        ]
    )
    assert EntropyScorer().score(candidate).score == pytest.approx(0.7)
    assert HighLossScorer().score(candidate).score == pytest.approx(8.0)
    assert HighLossScorer(loss_scale=16.0).score(candidate).score == pytest.approx(0.5)


def test_ensemble_disagreement_aggregates_across_outputs():
    candidate = _candidate(
        outputs=[
            _output("m-1", metadata={"member_score": 0.1}),
            _output("m-2", metadata={"member_score": 0.5}),
            _output("m-3", metadata={"member_score": 0.9}),
        ]
    )
    ranged = EnsembleDisagreementScorer().score(candidate)
    assert ranged.score == pytest.approx(0.8)
    assert "members=3" in ranged.reason
    variance = EnsembleDisagreementScorer(statistic="variance").score(candidate)
    assert variance.score == pytest.approx(((0.1 - 0.5) ** 2 + 0 + (0.9 - 0.5) ** 2) / 3)
    # single member -> no spread, abstain
    assert EnsembleDisagreementScorer().score(
        _candidate(outputs=[_output("m-1", metadata={"member_score": 0.4})])
    ) is None


def test_failure_severity_maps_text_and_numeric_impact():
    candidate = _candidate(outputs=[_output("out-1", metadata={"severity": "high"})])
    assert FailureSeverityScorer().score(candidate).score == pytest.approx(0.75)
    numeric = _candidate(
        outputs=[_output("out-1", metadata={"severity": "low", "incident_impact": 0.9})]
    )
    assert FailureSeverityScorer().score(numeric).score == pytest.approx(0.9)


def test_missing_labels_and_distribution_gap_boost():
    missing = MissingLabelsScorer(proportional=True).score(
        _candidate(required_label_types=("a", "b", "c", "d"), label_types=frozenset({"a"}))
    )
    assert missing.score == pytest.approx(0.75)
    gap = DistributionGapBoostScorer(normalize=True, target=4).score(
        _candidate(slice_label="site=b", slice_needed=2)
    )
    assert gap.score == pytest.approx(0.5)
    assert DistributionGapBoostScorer().score(_candidate(slice_needed=0)) is None


def test_label_cost_penalty_is_negative():
    result = LabelCostPenaltyScorer(weight=2.0).score(
        _candidate(outputs=[_output("out-1", metadata={"label_cost": 3.0})])
    )
    assert result.score == pytest.approx(-6.0)


def test_composite_sum_and_max_blend_components():
    candidate = _candidate(
        outputs=[_output("out-1", score=0.2, metadata={"entropy": 0.9, "label_cost": 1.0})]
    )
    composite = CompositeScorer(
        [
            (ConfidenceMarginScorer(), 1.0),
            (EntropyScorer(), 0.5),
            (LabelCostPenaltyScorer(), 1.0),
        ]
    )
    result = composite.score(candidate)
    # 0.8*1 + 0.9*0.5 - 1.0*1 = 0.25
    assert result.score == pytest.approx(0.25)
    assert set(result.components) == {"confidence-margin", "entropy", "label-cost"}

    max_blend = CompositeScorer(
        [(ConfidenceMarginScorer(), 1.0), (EntropyScorer(), 0.5)], combine="max"
    )
    assert max_blend.score(candidate).score == pytest.approx(0.8)


def test_composite_abstains_when_all_components_abstain():
    composite = CompositeScorer([(EntropyScorer(), 1.0)])
    assert composite.score(_candidate(outputs=[_output("out-1")])) is None


def test_registry_resolution_and_custom_registration():
    assert "confidence-margin" in builtin_scorer_names()
    assert isinstance(resolve_scorer(None), DefaultUncertaintyScorer)
    assert isinstance(resolve_scorer("entropy"), EntropyScorer)
    instance = ConfidenceMarginScorer()
    assert resolve_scorer(instance) is instance
    with pytest.raises(ScoringError):
        resolve_scorer("does-not-exist")
    with pytest.raises(ScoringError):
        register_scorer("confidence-margin", EntropyScorer)  # cannot overwrite built-in

    class _Custom(ActiveLearningScorer):
        name = "unit-custom"
        version = "9"

        def score(self, candidate):
            return ScorerResult(score=1.0, reason="custom")

    register_scorer("unit-custom", _Custom)
    try:
        assert isinstance(resolve_scorer("unit-custom"), _Custom)
    finally:
        unregister_scorer("unit-custom")
    with pytest.raises(ScoringError):
        resolve_scorer("unit-custom")


def test_resolve_scorer_builds_composite_from_sequence():
    composite = resolve_scorer(
        [("confidence-margin", 1.0), ("entropy", 0.5)], params={"combine": "max"}
    )
    assert isinstance(composite, CompositeScorer)
    assert composite.combine == "max"
    assert len(composite.components) == 2


def test_scorer_descriptor_is_deterministic_and_serializable():
    descriptor = CompositeScorer(
        [(ConfidenceMarginScorer(max_confidence_score=0.5), 2.0), (EntropyScorer(), 1.0)]
    ).descriptor()
    assert descriptor["name"] == "composite"
    assert [component["name"] for component in descriptor["params"]["components"]] == [
        "confidence-margin",
        "entropy",
    ]
    assert descriptor["params"]["components"][0]["weight"] == 2.0
