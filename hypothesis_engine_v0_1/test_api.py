from __future__ import annotations

from fastapi.testclient import TestClient

from hypothesis_engine_v0_1.api import create_app
from hypothesis_engine_v0_1.schemas import (
    BiologicalContext,
    ConditionContext,
    DataType,
    HypothesisEngineInput,
    MeasurementLevel,
    Observation,
    SignalDirection,
    StudyDesignContext,
    TargetSpec,
)


class _CleanOverrideLLM:
    def render_markdown(self, *, system_prompt: str, user_payload_json: str) -> str:
        assert '"query_id": "Q-API-001"' in user_payload_json
        return (
            "### Route Summary\n"
            "- H2 is compatible with the observed HIF markers under hypoxia."
        )


class _CausalOverrideLLM:
    def render_markdown(self, *, system_prompt: str, user_payload_json: str) -> str:
        return "Under hypoxia, IL6 is elevated because of HIF1A stabilization."


def _sample_payload() -> dict:
    engine_input = HypothesisEngineInput(
        query_id="Q-API-001",
        data_type=DataType.PURIFIED_MACROPHAGE_CULTURE,
        cell_composition_changed=False,
        timepoint_known=True,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        context=BiologicalContext(
            organism="Homo sapiens",
            cell_type="Macrophage",
            timepoint="12h",
            condition=ConditionContext(
                oxygen="Hypoxia",
                oxygen_level="1% O2",
                duration="12h",
            ),
            study_design=StudyDesignContext(control="Normoxia"),
        ),
        observations=[
            Observation(
                observation_id="OBS-1",
                entity="VEGFA",
                assay="RNA-seq",
                level=MeasurementLevel.MRNA,
                direction=SignalDirection.UP,
                timepoint="12h",
            ),
            Observation(
                observation_id="OBS-2",
                entity="SLC2A1",
                assay="RNA-seq",
                level=MeasurementLevel.MRNA,
                direction=SignalDirection.UP,
                timepoint="12h",
            ),
        ],
    )
    return {
        "engine_input": engine_input.model_dump(mode="json"),
        "max_bullets_per_section": 5,
        "strict_postprocess": True,
    }


def test_evaluate_and_render_deterministic_default() -> None:
    client = TestClient(create_app(llm_client=None))
    response = client.post(
        "/v0.1/hypothesis-engine/evaluate-and-render",
        json=_sample_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["used_llm"] is False
    assert body["validation_passed"] is True
    assert body["violations"] == []
    assert body["engine_output"]["query_id"] == "Q-API-001"
    assert "Q-API-001" in body["markdown"]
    assert len(body["input_hash"]) == 64
    assert len(body["output_hash"]) == 64
    assert body["explanation_hash"] is not None


def test_evaluate_and_render_with_llm_client_override() -> None:
    client = TestClient(create_app(llm_client=_CleanOverrideLLM()))
    response = client.post(
        "/v0.1/hypothesis-engine/evaluate-and-render",
        json=_sample_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["used_llm"] is True
    assert body["validation_passed"] is True
    assert "Route Summary" in body["markdown"]


def test_evaluate_and_render_strict_postprocess_returns_422_on_causal_claim() -> None:
    client = TestClient(create_app(llm_client=_CausalOverrideLLM()))
    payload = _sample_payload()
    payload["strict_postprocess"] = True

    response = client.post(
        "/v0.1/hypothesis-engine/evaluate-and-render",
        json=payload,
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("elevated because of" in v.lower() for v in detail["violations"])
    assert len(detail["input_hash"]) == 64
    assert len(detail["output_hash"]) == 64
