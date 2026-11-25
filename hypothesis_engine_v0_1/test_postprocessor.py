from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hypothesis_engine_v0_1.engine import run_hypothesis_engine
from hypothesis_engine_v0_1.postprocessor import (
    PostProcessorRequest,
    router as postprocessor_router,
    validate_and_postprocess,
)
from hypothesis_engine_v0_1.schemas import (
    BiologicalContext,
    ConditionContext,
    DataType,
    EvidenceRecord,
    HypothesisEngineInput,
    MeasurementLevel,
    Observation,
    SignalDirection,
    StudyDesignContext,
    TargetSpec,
)


def _sample_input_and_output() -> tuple[HypothesisEngineInput, dict]:
    engine_input = HypothesisEngineInput(
        query_id="Q-PP-001",
        data_type=DataType.PURIFIED_MACROPHAGE_CULTURE,
        cell_composition_changed=False,
        timepoint_known=True,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        context=BiologicalContext(
            organism="Homo sapiens",
            cell_type="Macrophage",
            timepoint="12h",
            condition=ConditionContext(oxygen="Hypoxia", oxygen_level="1% O2", duration="12h"),
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
        evidence_records=[
            EvidenceRecord(
                evidence_id="EV-101",
                source_id="SRC-101",
                claim_summary="HIF targets induced under hypoxia.",
                supports_hypothesis_ids=["H2"],
            )
        ],
    )
    engine_output = run_hypothesis_engine(engine_input).model_dump(mode="json")
    return engine_input, engine_output


def test_valid_output_and_clean_explanation_pass() -> None:
    engine_input, engine_output = _sample_input_and_output()
    req = PostProcessorRequest(
        engine_input=engine_input,
        engine_output=engine_output,
        rendered_explanation=(
            "Under hypoxia in purified macrophages, H2 is compatible with the observed HIF program, "
            "while causality is not established without perturbation."
        ),
    )
    res = validate_and_postprocess(req)
    assert res.valid is True
    assert res.violations == []
    assert len(res.input_hash) == 64
    assert len(res.output_hash) == 64
    assert res.explanation_hash is not None


def test_rejects_forbidden_numeric_fields_and_unknown_evidence_ids() -> None:
    engine_input, engine_output = _sample_input_and_output()
    engine_output["hypotheses"]["H2"]["supporting_evidence"] = ["EV-UNKNOWN"]
    engine_output["hypotheses"]["H2"]["posterior_probability"] = 0.85

    req = PostProcessorRequest(
        engine_input=engine_input,
        engine_output=engine_output,
        rendered_explanation="IL6 is elevated because of HIF1A.",
    )
    res = validate_and_postprocess(req)
    assert res.valid is False
    assert any("posterior_probability" in v for v in res.violations)
    assert any("EV-UNKNOWN" in v for v in res.violations)
    assert any("elevated because of" in v.lower() for v in res.violations)
    assert res.sanitized_explanation is not None
    assert "[removed_unsupported_causal_claim]" in res.sanitized_explanation


def test_fastapi_postprocess_endpoint_returns_422_on_violation() -> None:
    engine_input, engine_output = _sample_input_and_output()
    engine_output["query_id"] = "WRONG-ID"

    app = FastAPI()
    app.include_router(postprocessor_router)
    client = TestClient(app)

    response = client.post(
        "/v0.1/hypothesis-engine/postprocess",
        json={
            "engine_input": engine_input.model_dump(mode="json"),
            "engine_output": engine_output,
            "rendered_explanation": "Consistent with observed data.",
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert any("query_id mismatch" in v for v in body["detail"]["violations"])

def test_postprocessor_remaining_validation_and_non_strict_branches() -> None:
    engine_input, engine_output = _sample_input_and_output()

    bad_schema_output = dict(engine_output)
    bad_schema_output["hypotheses"] = "not-a-dict"
    res_bad_schema = validate_and_postprocess(
        PostProcessorRequest(
            engine_input=engine_input,
            engine_output=bad_schema_output,
            rendered_explanation="Neutral summary.",
            strict=False,
        )
    )
    assert any("Schema validation error" in v for v in res_bad_schema.violations)

    rich_output = dict(engine_output)
    h1_dict = dict(rich_output["hypotheses"]["H1"])
    h1_dict["supporting_evidence"] = ["EV-EXP-1"]
    h1_dict["contradicting_evidence"] = ["EV-EXP-2"]
    h1_dict["evidence_conflict"] = False
    h1_dict["resolution"] = "IL6 is elevated because of HIF1A activation."
    rich_output["hypotheses"] = dict(rich_output["hypotheses"])
    rich_output["hypotheses"]["H1"] = h1_dict

    if rich_output.get("pairwise_relations"):
        rel0 = dict(rich_output["pairwise_relations"][0])
        rel0["discriminating_measurement_id"] = "M-MISSING-ID"
        rich_output["pairwise_relations"] = [rel0] + list(rich_output["pairwise_relations"][1:])

    rich_output["model_prior_notes"] = [
        {
            "note_id": "NOTE-1",
            "description": "Prior note",
            "status": "model_prior_knowledge",
            "citable_as_evidence": False,
        }
    ]
    rich_output["observations"] = [{"entity": "UNKNOWN_MARKER_XYZ"}]

    res_rich = validate_and_postprocess(
        PostProcessorRequest(
            engine_input=engine_input,
            engine_output=rich_output,
            rendered_explanation="Neutral summary.",
            allowed_marker_ids={"UNKNOWN_MARKER_XYZ"},
            allowed_evidence_ids={"EV-EXP-1", "EV-EXP-2"},
            strict=False,
        )
    )
    assert not any("UNKNOWN_MARKER_XYZ" in v for v in res_rich.violations)
    assert any("evidence_conflict" in v.lower() or "contradicting" in v.lower() for v in res_rich.violations)
    assert any("M-MISSING-ID" in v for v in res_rich.violations)

    unknown_contra_output = dict(engine_output)
    h1_contra = dict(unknown_contra_output["hypotheses"]["H1"])
    h1_contra["contradicting_evidence"] = ["EV-UNKNOWN-CONTRA"]
    unknown_contra_output["hypotheses"] = {
        "H1": h1_contra,
        **{k: v for k, v in unknown_contra_output["hypotheses"].items() if k != "H1"},
    }
    res_contra = validate_and_postprocess(
        PostProcessorRequest(
            engine_input=engine_input,
            engine_output=unknown_contra_output,
            rendered_explanation="Neutral summary.",
            strict=False,
        )
    )
    assert any("EV-UNKNOWN-CONTRA" in v for v in res_contra.violations)

    valid_in, valid_out = _sample_input_and_output()
    app = FastAPI()
    app.include_router(postprocessor_router)
    client = TestClient(app)
    ok_resp = client.post(
        "/v0.1/hypothesis-engine/postprocess",
        json={
            "engine_input": valid_in.model_dump(mode="json"),
            "engine_output": valid_out,
            "rendered_explanation": "Consistent with observed data.",
        },
    )
    assert ok_resp.status_code == 200
    assert ok_resp.json()["valid"] is True
    assert any("causal" in v.lower() or "elevated because of" in v.lower() for v in res_rich.violations)

    res_unknown_obs = validate_and_postprocess(
        PostProcessorRequest(
            engine_input=engine_input,
            engine_output=rich_output,
            rendered_explanation="Neutral summary.",
            allowed_evidence_ids={"EV-EXP-1", "EV-EXP-2"},
            strict=False,
        )
    )
    assert any("UNKNOWN_MARKER_XYZ" in v for v in res_unknown_obs.violations)