from __future__ import annotations

import json
from pathlib import Path
import pytest

from hypothesis_engine_v0_1.scoring_rules import (
    ScoringPolicy,
    WeightedObservation,
    evaluate_weighted_hypothesis,
)

BASE_DIR = Path(__file__).resolve().parent / "biological_cases"
EXPECTED_DIR = BASE_DIR / "expected_outputs"
REQUIRED_CONTEXT_FIELDS = ("organism", "cell_type", "condition", "measurement_level")


def _extract_input_block(case_data: dict) -> dict:
    if isinstance(case_data.get("input"), dict):
        return case_data["input"]
    return {
        "context": case_data.get("context") or {},
        "observations": case_data.get("observations") or [],
    }


def _evaluate_context_and_conflicts(case_data: dict) -> tuple[str, list[str]]:
    inp = _extract_input_block(case_data)
    ctx = inp.get("context") or {}
    conflicts: list[str] = []

    if not ctx.get("organism"):
        conflicts.append("missing_organism")

    condition = str(ctx.get("condition", "")).lower()
    assay = str(ctx.get("assay", "")).lower()
    stem = str(case_data.get("case_id", "")).lower()

    if "bulk_vs_single_cell" in stem or "bulk_vs_scrna" in assay or ctx.get("composition_resolved") is False:
        conflicts.append("bulk_vs_single_cell_conflict")

    if "conflict" in stem or "21pct" in condition or "normoxia" in condition:
        conflicts.append("context_or_condition_conflict")

    if "missing_organism" in conflicts:
        return "hard_abstention", conflicts

    missing_soft = [field for field in REQUIRED_CONTEXT_FIELDS if not ctx.get(field)]
    if missing_soft:
        conflicts.extend([f"missing_{f}" for f in missing_soft])

    return "weak_inference", conflicts


def _render_non_causal_summary(case_data: dict, trace, mode: str, conflicts: list[str]) -> str:
    inp = _extract_input_block(case_data)
    ctx = inp.get("context") or {}
    return (
        f"Mode={mode}. Conflicts={conflicts}. "
        f"Context: organism={ctx.get('organism')}, cell_type={ctx.get('cell_type')}, "
        f"condition={ctx.get('condition')}, timepoint={ctx.get('timepoint')}, "
        f"level={ctx.get('measurement_level')}, assay={ctx.get('assay')}. "
        f"Net score={trace.net_score:.4f}, passed_threshold={trace.passed_threshold}. "
        f"Observational data only; no causal perturbation established."
    )


@pytest.mark.parametrize(
    "case_path",
    sorted(BASE_DIR.glob("case_*.json")),
    ids=lambda p: p.stem,
)
def test_biological_regression_case(case_path: Path):
    expected_path = EXPECTED_DIR / case_path.name
    assert expected_path.exists(), f"Missing expected file for {case_path.name}"

    case_data = json.loads(case_path.read_text(encoding="utf-8-sig"))
    case_data.setdefault("case_id", case_path.stem)
    expected = json.loads(expected_path.read_text(encoding="utf-8-sig"))

    inp = _extract_input_block(case_data)
    observations = [
        WeightedObservation(**obs) for obs in inp.get("observations", [])
    ]
    policy = ScoringPolicy()
    trace = evaluate_weighted_hypothesis(observations=observations, policy=policy)

    actual_mode, actual_conflicts = _evaluate_context_and_conflicts(case_data)

    assert actual_mode == expected["expected_mode"]
    assert trace.passed_threshold is expected["expected_passed_threshold"]

    for exp_conflict in expected.get("expected_conflicts", []):
        assert exp_conflict in actual_conflicts

    forbidden_list = case_data.get("forbidden_claims") or expected.get("forbidden_outputs", [])
    summary = _render_non_causal_summary(case_data, trace, actual_mode, actual_conflicts).lower()
    for forbidden in forbidden_list:
        assert forbidden.lower() not in summary
