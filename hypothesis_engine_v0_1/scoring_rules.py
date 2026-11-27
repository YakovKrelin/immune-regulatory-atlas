from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

Direction = Literal["support", "contradict", "neutral"]


class WeightedObservation(BaseModel):
    observation_id: str = Field(..., min_length=1)
    direction: Direction
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_type: str = Field(..., min_length=1)


class ScoringPolicy(BaseModel):
    source_weights: dict[str, float] = Field(default_factory=dict)
    default_source_weight: float = Field(default=1.0, ge=0.0)
    conflict_penalty_factor: float = Field(default=0.5, ge=0.0, le=1.0)
    min_support_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ObservationContribution(BaseModel):
    observation_id: str
    direction: Direction
    applied_weight: float
    signed_contribution: float


class ScoringTrace(BaseModel):
    raw_support: float
    raw_contradict: float
    conflict_penalty: float
    net_score: float
    passed_threshold: bool
    triggered_rules: list[str]
    contributions: list[ObservationContribution]


def resolve_source_weight(source_type: str, policy: ScoringPolicy) -> float:
    """озвращает вес для source_type из policy.source_weights или default_source_weight."""
    if source_type in policy.source_weights:
        return policy.source_weights[source_type]
    return policy.default_source_weight


def compute_observation_contribution(
    obs: WeightedObservation,
    policy: ScoringPolicy,
) -> ObservationContribution:
    """ычисляет вклад одного наблюдения."""
    weight = resolve_source_weight(obs.source_type, policy)
    if obs.direction == "support":
        signed = obs.confidence * weight
    elif obs.direction == "contradict":
        signed = -(obs.confidence * weight)
    else:
        signed = 0.0

    return ObservationContribution(
        observation_id=obs.observation_id,
        direction=obs.direction,
        applied_weight=weight,
        signed_contribution=signed,
    )


def calculate_conflict_penalty(
    raw_support: float,
    raw_contradict: float,
    penalty_factor: float,
) -> float:
    """ычисляет штраф за конфликт при одновременной поддержке и противоречии."""
    if raw_support > 0.0 and raw_contradict > 0.0:
        return min(raw_support, raw_contradict) * penalty_factor
    return 0.0


def evaluate_weighted_hypothesis(
    observations: list[WeightedObservation],
    policy: ScoringPolicy | None = None,
) -> ScoringTrace:
    """ыполняет детерминированный расчёт скоринга и формирует трассировку."""
    active_policy = policy if policy is not None else ScoringPolicy()

    if not observations:
        return ScoringTrace(
            raw_support=0.0,
            raw_contradict=0.0,
            conflict_penalty=0.0,
            net_score=0.0,
            passed_threshold=False,
            triggered_rules=["empty_observations"],
            contributions=[],
        )

    contributions: list[ObservationContribution] = []
    raw_support = 0.0
    raw_contradict = 0.0
    total_weight = 0.0

    for obs in observations:
        contrib = compute_observation_contribution(obs, active_policy)
        contributions.append(contrib)
        total_weight += contrib.applied_weight
        if contrib.signed_contribution > 0.0:
            raw_support += contrib.signed_contribution
        elif contrib.signed_contribution < 0.0:
            raw_contradict += abs(contrib.signed_contribution)

    conflict_penalty = calculate_conflict_penalty(
        raw_support=raw_support,
        raw_contradict=raw_contradict,
        penalty_factor=active_policy.conflict_penalty_factor,
    )

    if total_weight == 0.0:
        net_score = 0.0
    else:
        raw_net = (raw_support - raw_contradict - conflict_penalty) / total_weight
        net_score = max(0.0, min(1.0, raw_net))

    passed = net_score >= active_policy.min_support_threshold
    rules: list[str] = ["weighted_aggregation"]
    if conflict_penalty > 0.0:
        rules.append("conflict_penalty_applied")
    rules.append("threshold_passed" if passed else "threshold_failed")

    return ScoringTrace(
        raw_support=raw_support,
        raw_contradict=raw_contradict,
        conflict_penalty=conflict_penalty,
        net_score=net_score,
        passed_threshold=passed,
        triggered_rules=rules,
        contributions=contributions,
    )
