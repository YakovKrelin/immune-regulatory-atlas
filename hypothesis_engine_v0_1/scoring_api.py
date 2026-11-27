from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from hypothesis_engine_v0_1.scoring_rules import (
    ScoringPolicy,
    ScoringTrace,
    WeightedObservation,
    evaluate_weighted_hypothesis,
)

router = APIRouter(prefix="/scoring", tags=["scoring"])


class HypothesisEvaluationRequest(BaseModel):
    observations: list[WeightedObservation] = Field(default_factory=list)
    policy: ScoringPolicy | None = None


@router.post("/evaluate", response_model=ScoringTrace)
def evaluate_hypothesis_endpoint(payload: HypothesisEvaluationRequest) -> ScoringTrace:
    """ценивает список наблюдений по заданной политике скоринга."""
    return evaluate_weighted_hypothesis(
        observations=payload.observations,
        policy=payload.policy,
    )
