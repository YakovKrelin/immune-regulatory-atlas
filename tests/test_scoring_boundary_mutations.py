import pytest

from hypothesis_engine_v0_1.scoring_rules import (
    ScoringPolicy,
    WeightedObservation,
    evaluate_weighted_hypothesis,
)


def test_threshold_exact_boundary_and_just_below():
    policy = ScoringPolicy(
        source_weights={"assay": 1.0, "literature": 0.5},
        min_support_threshold=0.5,
    )

    obs_exact = [
        WeightedObservation(
            observation_id="obs-1",
            direction="support",
            confidence=0.5,
            source_type="assay",
        )
    ]
    trace_exact = evaluate_weighted_hypothesis(observations=obs_exact, policy=policy)
    assert trace_exact.raw_support == pytest.approx(0.5)
    assert trace_exact.raw_contradict == pytest.approx(0.0)
    assert trace_exact.passed_threshold is True

    obs_below = [
        WeightedObservation(
            observation_id="obs-2",
            direction="support",
            confidence=0.49,
            source_type="assay",
        )
    ]
    trace_below = evaluate_weighted_hypothesis(observations=obs_below, policy=policy)
    assert trace_below.raw_support == pytest.approx(0.49)
    assert trace_below.passed_threshold is False


def test_source_weights_and_directions_are_not_swapped():
    policy = ScoringPolicy(
        source_weights={"assay": 1.0, "literature": 0.25},
        min_support_threshold=0.6,
    )
    observations = [
        WeightedObservation(
            observation_id="obs-sup",
            direction="support",
            confidence=0.6,
            source_type="assay",
        ),
        WeightedObservation(
            observation_id="obs-con",
            direction="contradict",
            confidence=0.8,
            source_type="literature",
        ),
    ]

    trace = evaluate_weighted_hypothesis(observations=observations, policy=policy)
    assert trace.raw_support == pytest.approx(0.6)
    assert trace.raw_contradict == pytest.approx(0.2)
    assert trace.conflict_penalty == pytest.approx(0.1)
    assert trace.net_score == pytest.approx(0.24)
    assert trace.passed_threshold is False
    assert "conflict_penalty_applied" in trace.triggered_rules
    assert "threshold_failed" in trace.triggered_rules
    assert len(trace.contributions) == 2
    assert trace.contributions[0].signed_contribution == pytest.approx(0.6)
    assert trace.contributions[1].signed_contribution == pytest.approx(-0.2)



import pytest

from hypothesis_engine_v0_1.scoring_rules import (
    ScoringPolicy,
    WeightedObservation,
    evaluate_weighted_hypothesis,
)


def test_zero_conflict_and_exact_boundary_operators():
    policy = ScoringPolicy(
        source_weights={"assay": 1.0, "literature": 0.5},
        min_support_threshold=0.5,
    )

    # Exact threshold equality (0.5 == 0.5)
    obs_on_threshold = [
        WeightedObservation(
            observation_id="obs-eq",
            direction="support",
            confidence=0.5,
            source_type="assay",
        )
    ]
    trace_eq = evaluate_weighted_hypothesis(observations=obs_on_threshold, policy=policy)
    assert trace_eq.raw_support == pytest.approx(0.5)
    assert trace_eq.raw_contradict == pytest.approx(0.0)
    assert trace_eq.conflict_penalty == pytest.approx(0.0)
    assert trace_eq.passed_threshold is True

    # Zero confidence on boundary (0.0)
    obs_zero_conf = [
        WeightedObservation(
            observation_id="obs-zero",
            direction="support",
            confidence=0.0,
            source_type="assay",
        )
    ]
    trace_zero = evaluate_weighted_hypothesis(observations=obs_zero_conf, policy=policy)
    assert trace_zero.raw_support == pytest.approx(0.0)
    assert trace_zero.conflict_penalty == pytest.approx(0.0)
    assert trace_zero.passed_threshold is False

    # Just above and just below threshold (0.5001 vs 0.4999)
    obs_above = [
        WeightedObservation(
            observation_id="obs-hi",
            direction="support",
            confidence=0.5001,
            source_type="assay",
        )
    ]
    obs_below = [
        WeightedObservation(
            observation_id="obs-lo",
            direction="support",
            confidence=0.4999,
            source_type="assay",
        )
    ]
    assert evaluate_weighted_hypothesis(observations=obs_above, policy=policy).passed_threshold is True
    assert evaluate_weighted_hypothesis(observations=obs_below, policy=policy).passed_threshold is False

def test_zero_conflict_and_exact_boundary_operators():
    policy = ScoringPolicy(
        source_weights={"assay": 1.0, "literature": 0.5},
        min_support_threshold=0.5,
    )

    obs_on_threshold = [
        WeightedObservation(
            observation_id="obs-eq",
            direction="support",
            confidence=0.5,
            source_type="assay",
        )
    ]
    trace_eq = evaluate_weighted_hypothesis(observations=obs_on_threshold, policy=policy)
    assert trace_eq.raw_support == pytest.approx(0.5)
    assert trace_eq.raw_contradict == pytest.approx(0.0)
    assert trace_eq.conflict_penalty == pytest.approx(0.0)
    assert trace_eq.passed_threshold is True

    obs_zero_conf = [
        WeightedObservation(
            observation_id="obs-zero",
            direction="support",
            confidence=0.0,
            source_type="assay",
        )
    ]
    trace_zero = evaluate_weighted_hypothesis(observations=obs_zero_conf, policy=policy)
    assert trace_zero.raw_support == pytest.approx(0.0)
    assert trace_zero.conflict_penalty == pytest.approx(0.0)
    assert trace_zero.passed_threshold is False

    obs_above = [
        WeightedObservation(
            observation_id="obs-hi",
            direction="support",
            confidence=0.5001,
            source_type="assay",
        )
    ]
    obs_below = [
        WeightedObservation(
            observation_id="obs-lo",
            direction="support",
            confidence=0.4999,
            source_type="assay",
        )
    ]
    assert evaluate_weighted_hypothesis(observations=obs_above, policy=policy).passed_threshold is True
    assert evaluate_weighted_hypothesis(observations=obs_below, policy=policy).passed_threshold is False

def test_zero_support_zero_contradict_and_zero_weight_boundaries():
    from hypothesis_engine_v0_1.scoring_rules import calculate_conflict_penalty

    assert calculate_conflict_penalty(0.0, 0.8, 0.5) == pytest.approx(0.0)
    assert calculate_conflict_penalty(0.8, 0.0, 0.5) == pytest.approx(0.0)
    assert calculate_conflict_penalty(0.0, 0.0, 0.5) == pytest.approx(0.0)

    zero_weight_policy = ScoringPolicy(
        source_weights={"assay": 0.0, "literature": 0.5},
        min_support_threshold=0.5,
    )
    obs_zero_contrib = [
        WeightedObservation(
            observation_id="obs-w0",
            direction="support",
            confidence=0.9,
            source_type="assay",
        )
    ]
    trace_w0 = evaluate_weighted_hypothesis(observations=obs_zero_contrib, policy=zero_weight_policy)
    assert trace_w0.raw_support == pytest.approx(0.0)
    assert trace_w0.raw_contradict == pytest.approx(0.0)
    assert trace_w0.conflict_penalty == pytest.approx(0.0)
    assert trace_w0.net_score == pytest.approx(0.0)
    assert trace_w0.passed_threshold is False
    assert "conflict_penalty_applied" not in trace_w0.triggered_rules

    only_contradict_policy = ScoringPolicy(
        source_weights={"assay": 1.0, "literature": 0.5},
        min_support_threshold=0.5,
    )
    obs_only_con = [
        WeightedObservation(
            observation_id="obs-con-only",
            direction="contradict",
            confidence=0.7,
            source_type="assay",
        )
    ]
    trace_con = evaluate_weighted_hypothesis(observations=obs_only_con, policy=only_contradict_policy)
    assert trace_con.raw_support == pytest.approx(0.0)
    assert trace_con.raw_contradict == pytest.approx(0.7)
    assert trace_con.conflict_penalty == pytest.approx(0.0)
    assert "conflict_penalty_applied" not in trace_con.triggered_rules
