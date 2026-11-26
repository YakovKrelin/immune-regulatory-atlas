import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import process_module_contract as pmc

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "hypoxia_macrophage_contract.json"
SCHEMA_PATH = REPO_ROOT / "process_module_contract.schema.json"
CASES_DIR = REPO_ROOT / "tests" / "biological_cases"


def _load_json_utf8(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_contract_matches_pydantic_and_schema_keys():
    contract_raw = _load_json_utf8(CONTRACT_PATH)
    schema_raw = _load_json_utf8(SCHEMA_PATH)

    contract = pmc.ProcessModuleContract.model_validate(contract_raw)

    assert contract.version == "0.1.0"
    assert contract.rule_set_version == "1.1.0"
    assert set(contract_raw.keys()) == set(schema_raw["required"])
    assert schema_raw["additionalProperties"] is False

    h1 = next(b for b in contract.branches if b.branch_id == "H1")
    hif_rule = next(r for r in h1.output_rules if r.rule_id == "R001")
    assert hif_rule.then.data_status == pmc.DataStatus.COMPATIBLE
    assert hif_rule.then.data_status != pmc.DataStatus.SUPPORTED


def test_status_outcome_rejects_inapplicable_supported_data():
    with pytest.raises(ValidationError):
        pmc.StatusOutcome(
            branch_applicability=pmc.BranchApplicability.NOT_APPLICABLE,
            data_status=pmc.DataStatus.SUPPORTED,
        )


def test_weakly_supported_is_decision_status_only():
    assert pmc.DecisionStatus.WEAKLY_SUPPORTED.value == "weakly_supported"
    assert all(item.value != "weakly_supported" for item in pmc.DataStatus)


def test_biological_cases_structure_and_declared_in_contract():
    contract = pmc.ProcessModuleContract.model_validate(_load_json_utf8(CONTRACT_PATH))
    case_files = sorted(CASES_DIR.glob("*.json"))
    assert len(case_files) >= 3

    required_top_keys = {"case_id", "category", "input", "expected", "forbidden_claims"}
    for case_file in case_files:
        payload = _load_json_utf8(case_file)
        assert required_top_keys.issubset(payload.keys())
        assert isinstance(payload["case_id"], str) and payload["case_id"]
        assert isinstance(payload["input"], dict) and payload["input"]
        assert isinstance(payload["expected"], dict)
        assert payload["expected"].get("mode") in {m.value for m in pmc.InferenceMode}
        assert isinstance(payload["forbidden_claims"], list)

    for case_id in contract.regression_cases:
        assert (CASES_DIR / f"{case_id}.json").exists(), f"Missing case file for {case_id}"
