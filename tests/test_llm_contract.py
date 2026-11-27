from __future__ import annotations

import pytest
from pydantic import ValidationError

from hypothesis_engine_v0_1.llm_contract import (
    GroundedClaim,
    LLMExplanationPayload,
    get_llm_response_json_schema,
    validate_llm_explanation,
)


def test_json_schema_structure():
    schema = get_llm_response_json_schema()
    assert schema["type"] == "object"
    assert "claims" in schema["properties"]
    assert "introduced_hypothesis_ids" in schema["properties"]


def test_valid_non_causal_cited_payload_passes():
    payload = LLMExplanationPayload(
        query_id="q-001",
        mode="weak_inference",
        has_perturbation_evidence=False,
        claims=[
            GroundedClaim(
                statement="Higher IL6 mRNA abundance is observed under 1% O2 compared with control.",
                claim_kind="observation_summary",
                hypothesis_id="H1",
                cited_ids=["OBS-1"],
            )
        ],
        limitations=["Observational transcriptomic data without causal perturbation."],
    )

    res = validate_llm_explanation(
        payload,
        allowed_source_ids={"OBS-1", "EV-1"},
        allowed_hypothesis_ids={"H1", "H2"},
        expected_mode="weak_inference",
        has_perturbation_evidence=False,
    )
    assert res.is_valid is True
    assert res.issues == []
    assert res.sanitized_payload is not None


def test_rejects_unverified_citation_causal_claim_and_new_hypothesis():
    bad_claim_payload = LLMExplanationPayload(
        query_id="q-002",
        mode="weak_inference",
        has_perturbation_evidence=False,
        claims=[
            GroundedClaim(
                statement="HIF1A causes IL6 upregulation in macrophages.",
                claim_kind="hypothesis_status",
                hypothesis_id="H99",
                cited_ids=["OBS-FAKE"],
            )
        ],
    )

    res = validate_llm_explanation(
        bad_claim_payload,
        allowed_source_ids={"OBS-1"},
        allowed_hypothesis_ids={"H1", "H2"},
        expected_mode="weak_inference",
        has_perturbation_evidence=False,
    )
    assert res.is_valid is False
    codes = {i.code for i in res.issues}
    assert "missing_or_unknown_citation" in codes
    assert "forbidden_causal_claim_without_perturbation" in codes
    assert "unauthorized_hypothesis" in codes

    with pytest.raises(ValidationError):
        LLMExplanationPayload(
            query_id="q-003",
            mode="weak_inference",
            introduced_hypothesis_ids=["H_NEW"],
            claims=[],
        )
