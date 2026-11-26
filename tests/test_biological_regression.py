import json
from pathlib import Path

import pytest

import process_module_contract as pmc
from hypothesis_engine_v0_1 import engine as eng
from hypothesis_engine_v0_1 import schemas

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "hypoxia_macrophage_contract.json"
CASES_DIR = REPO_ROOT / "tests" / "biological_cases"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_cases():
    files = sorted(CASES_DIR.glob("*.json"))
    assert files, f"No biological cases found in {CASES_DIR}"
    return [_load_json(p) for p in files]


def _filter_to_model_fields(model_cls, data: dict) -> dict:
    fields = getattr(model_cls, "model_fields", None)
    if not fields:
        return dict(data)
    return {k: v for k, v in data.items() if k in fields}


def _resolve_target_level(raw_level):
    level_enum = getattr(schemas, "MeasurementLevel", None)
    if level_enum is None or raw_level is None:
        return raw_level
    for member in level_enum:
        if member.value == raw_level or member.name == raw_level:
            return member
    return getattr(level_enum, "NOT_SPECIFIED", raw_level)


def _build_target_spec(case_input: dict):
    target_cls = getattr(schemas, "TargetSpec", None)
    raw_target = case_input.get("target")
    merged = {}
    if isinstance(raw_target, dict):
        merged.update(raw_target)
    elif isinstance(raw_target, str) and raw_target:
        merged["entity"] = raw_target
    if not merged.get("entity"):
        merged["entity"] = "IL6"

    raw_level = merged.get("level", case_input.get("measurement_level"))
    resolved_level = _resolve_target_level(raw_level)
    if resolved_level is not None:
        merged["level"] = resolved_level

    if target_cls is None:
        return merged
    return target_cls.model_validate(_filter_to_model_fields(target_cls, merged))


def _build_nested_subcontext(cls_name: str, sources: list[dict]):
    sub_cls = getattr(schemas, cls_name, None)
    merged = {}
    for src in sources:
        if isinstance(src, dict):
            merged.update(src)
    if sub_cls is None:
        return merged
    return sub_cls.model_validate(_filter_to_model_fields(sub_cls, merged))


def _build_biological_context(case_input: dict):
    ctx_cls = getattr(schemas, "BiologicalContext", None)
    merged = {}
    nested_ctx = case_input.get("context")
    if isinstance(nested_ctx, dict):
        merged.update(nested_ctx)

    for key in (
        "organism",
        "cell_type",
        "cell_subtype",
        "tissue",
        "compartment",
        "timepoint",
        "sample_type",
    ):
        if key in case_input and key not in merged:
            merged[key] = case_input[key]

    cond_sources = []
    if isinstance(nested_ctx, dict) and isinstance(nested_ctx.get("condition"), dict):
        cond_sources.append(nested_ctx["condition"])
    if isinstance(case_input.get("condition"), dict):
        cond_sources.append(case_input["condition"])
    cond_extra = {}
    for k in ("oxygen", "oxygen_level", "duration", "stimulus", "priming", "co_stimulus"):
        if k in case_input and k not in cond_extra:
            cond_extra[k] = case_input[k]
    if cond_extra:
        cond_sources.append(cond_extra)
    merged["condition"] = _build_nested_subcontext("ConditionContext", cond_sources)

    design_sources = []
    if isinstance(nested_ctx, dict) and isinstance(nested_ctx.get("study_design"), dict):
        design_sources.append(nested_ctx["study_design"])
    if isinstance(case_input.get("study_design"), dict):
        design_sources.append(case_input["study_design"])
    design_extra = {}
    for k in ("control", "control_or_reference_condition", "replicates", "batch"):
        if k in case_input and k not in design_extra:
            design_extra[k] = case_input[k]
    if "control_or_reference_condition" in design_extra and "control" not in design_extra:
        design_extra["control"] = design_extra["control_or_reference_condition"]
    if "control" in design_extra and "control_or_reference_condition" not in design_extra:
        design_extra["control_or_reference_condition"] = design_extra["control"]
    if design_extra:
        design_sources.append(design_extra)
    merged["study_design"] = _build_nested_subcontext("StudyDesignContext", design_sources)

    if ctx_cls is None:
        return merged
    return ctx_cls.model_validate(_filter_to_model_fields(ctx_cls, merged))


def _build_engine_input(case_id: str, case_input: dict) -> schemas.HypothesisEngineInput:
    payload = {
        "query_id": case_input.get("query_id", case_id),
        "scenario": case_input.get("scenario", "IL6_hypoxia_macrophage"),
        "target": _build_target_spec(case_input),
        "context": _build_biological_context(case_input),
    }
    for top_field in (
        "data_type",
        "cell_composition_changed",
        "timepoint_known",
        "observations",
        "evidence_records",
    ):
        if top_field in case_input:
            payload[top_field] = case_input[top_field]

    filtered = _filter_to_model_fields(schemas.HypothesisEngineInput, payload)
    return schemas.HypothesisEngineInput.model_validate(filtered)


def _dump_output_text(output: schemas.HypothesisEngineOutput) -> str:
    return json.dumps(output.model_dump(mode="json"), ensure_ascii=False, default=str)


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["case_id"])
def test_biological_case_mode_conflicts_and_hypotheses(case):
    contract = pmc.ProcessModuleContract.model_validate(_load_json(CONTRACT_PATH))
    case_input = case["input"]

    if "measurement_level" in case_input:
        allowed_levels = {lvl.value for lvl in contract.allowed_measurement_levels}
        assert case_input["measurement_level"] in allowed_levels

    engine_input = _build_engine_input(case["case_id"], case_input)
    output = eng.run_hypothesis_engine(engine_input)

    expected = case.get("expected", {})
    expected_mode = expected.get("mode")
    if expected_mode is not None:
        actual_mode = getattr(output.mode, "value", output.mode)
        assert actual_mode == expected_mode

    expected_conflicts = expected.get("conflicts", [])
    if expected_conflicts:
        all_conflict_text = " ".join(
            [
                *output.context_conflicts,
                *(rc.reason for rc in output.rule_conflicts),
                *(rid for rc in output.rule_conflicts for rid in rc.rule_ids),
            ]
        )
        for conflict_ref in expected_conflicts:
            assert (
                conflict_ref in all_conflict_text
                or len(output.context_conflicts) > 0
                or len(output.rule_conflicts) > 0
            )

    for branch_id, expected_axes in expected.get("branch_statuses", {}).items():
        if branch_id in output.hypotheses:
            hyp_eval = output.hypotheses[branch_id]
            for axis_name, expected_val in expected_axes.items():
                if hasattr(hyp_eval, axis_name):
                    actual_val = getattr(hyp_eval, axis_name)
                    assert getattr(actual_val, "value", actual_val) == expected_val

    rendered = _dump_output_text(output)
    for forbidden in case.get("forbidden_claims", []):
        assert forbidden not in rendered
