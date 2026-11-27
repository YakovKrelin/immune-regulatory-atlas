from __future__ import annotations

import json
from pathlib import Path

import pytest

import process_module_contract as pmc
from hypothesis_engine_v0_1.scoring_rules import (
    ScoringPolicy,
    WeightedObservation,
    evaluate_weighted_hypothesis,
)

CASES_DIR = Path(__file__).resolve().parent / "biological_cases"
VALID_MODES = {m.value for m in pmc.InferenceMode}


def _load_cases() -> list[dict[str, object]]:
    case_files = sorted(CASES_DIR.glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8-sig")) for path in case_files]


@pytest.mark.parametrize(
    "case",
    _load_cases(),
    ids=lambda c: str(c["case_id"]),
)
def test_biological_regression_case(case: dict[str, object]) -> None:
    assert {"case_id", "category", "input", "expected", "forbidden_claims"}.issubset(case.keys())

    payload = case["input"]
    expected = case["expected"]
    forbidden = case["forbidden_claims"]

    assert isinstance(payload, dict)
    assert isinstance(expected, dict)
    assert isinstance(forbidden, list)
    assert expected.get("mode") in VALID_MODES

    raw_observations = payload.get("observations")
    if isinstance(raw_observations, list):
        policy_data = payload.get("policy")
        policy = ScoringPolicy.model_validate(policy_data) if isinstance(policy_data, dict) else None
        observations = [WeightedObservation.model_validate(item) for item in raw_observations]
        trace = evaluate_weighted_hypothesis(observations=observations, policy=policy)

        if "raw_support" in expected:
            assert trace.raw_support == pytest.approx(float(expected["raw_support"]))
        if "raw_contradict" in expected:
            assert trace.raw_contradict == pytest.approx(float(expected["raw_contradict"]))
        if "passed_threshold" in expected:
            assert trace.passed_threshold is bool(expected["passed_threshold"])

        serialized_trace = trace.model_dump_json().lower()
        for claim in forbidden:
            assert str(claim).lower() not in serialized_trace
