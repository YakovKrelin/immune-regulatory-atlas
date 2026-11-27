import pytest
from pydantic import ValidationError

from hypothesis_engine_v0_1.scoring_rules import (
    ScoringPolicy,
    WeightedObservation,
    evaluate_weighted_hypothesis,
)


def test_empty_observations_returns_zero_trace() -> None:
    trace = evaluate_weighted_hypothesis([])
    assert trace.raw_support == 0.0
    assert trace.raw_contradict == 0.0
    assert trace.conflict_penalty == 0.0
    assert trace.net_score == 0.0
    assert trace.passed_threshold is False
    assert trace.triggered_rules == ["empty_observations"]
    assert trace.contributions == []


def test_support_neutral_and_contradict_with_conflict_penalty() -> None:
    policy = ScoringPolicy(
        source_weights={"assay": 2.0, "literature": 1.0},
        default_source_weight=1.0,
        conflict_penalty_factor=0.5,
        min_support_threshold=0.2,
    )
    observations = [
        WeightedObservation(
            observation_id="obs-1",
            direction="support",
            confidence=0.8,
            source_type="assay",
        ),
        WeightedObservation(
            observation_id="obs-2",
            direction="contradict",
            confidence=0.6,
            source_type="literature",
        ),
        WeightedObservation(
            observation_id="obs-3",
            direction="neutral",
            confidence=0.9,
            source_type="other",
        ),
    ]

    trace = evaluate_weighted_hypothesis(observations, policy=policy)
    assert trace.raw_support == pytest.approx(1.6)
    assert trace.raw_contradict == pytest.approx(0.6)
    assert trace.conflict_penalty == pytest.approx(0.3)
    assert trace.net_score == pytest.approx(0.7 / 4.0)
    assert trace.passed_threshold is False
    assert "conflict_penalty_applied" in trace.triggered_rules
    assert "threshold_failed" in trace.triggered_rules


def test_threshold_passed_and_zero_total_weight() -> None:
    policy = ScoringPolicy(
        source_weights={"zero_src": 0.0},
        default_source_weight=0.0,
        min_support_threshold=0.0,
    )
    observations = [
        WeightedObservation(
            observation_id="obs-zero",
            direction="support",
            confidence=1.0,
            source_type="zero_src",
        )
    ]

    trace = evaluate_weighted_hypothesis(observations, policy=policy)
    assert trace.net_score == 0.0
    assert trace.passed_threshold is True
    assert "threshold_passed" in trace.triggered_rules
    assert "conflict_penalty_applied" not in trace.triggered_rules


def test_validation_rejects_invalid_confidence_and_weights() -> None:
    with pytest.raises(ValidationError):
        WeightedObservation(
            observation_id="bad-obs",
            direction="support",
            confidence=1.5,
            source_type="assay",
        )

    with pytest.raises(ValidationError):
        ScoringPolicy(default_source_weight=-0.2)
