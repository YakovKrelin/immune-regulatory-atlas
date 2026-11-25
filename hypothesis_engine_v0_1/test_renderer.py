from __future__ import annotations

from hypothesis_engine_v0_1.engine import run_hypothesis_engine
from hypothesis_engine_v0_1.renderer import (
    RenderReportRequest,
    render_hypothesis_report,
)
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


class _FakeSuccessLLM:
    def render_markdown(self, *, system_prompt: str, user_payload_json: str) -> str:
        assert "Do not change mode" in system_prompt
        assert '"query_id": "Q-REN-001"' in user_payload_json
        return (
            "### Concise Summary\n"
            "- H2 is compatible with the observed HIF program under hypoxia.\n"
            "- H1 is not supported by the current data at this timepoint."
        )


class _FakeCausalLLM:
    def render_markdown(self, *, system_prompt: str, user_payload_json: str) -> str:
        return "Under hypoxia, IL6 is elevated because of HIF1A stabilization."


class _FakeFailingLLM:
    def render_markdown(self, *, system_prompt: str, user_payload_json: str) -> str:
        raise RuntimeError("Local inference timeout")


def _sample_input() -> HypothesisEngineInput:
    return HypothesisEngineInput(
        query_id="Q-REN-001",
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
    )


def test_deterministic_fallback_when_no_llm_client() -> None:
    engine_input = _sample_input()
    engine_output = run_hypothesis_engine(engine_input)

    res = render_hypothesis_report(
        RenderReportRequest(engine_input=engine_input, engine_output=engine_output),
        llm_client=None,
    )

    assert res.used_llm is False
    assert res.validation_passed is True
    assert res.violations == []
    assert "Q-REN-001" in res.markdown
    assert res.explanation_hash is not None


def test_mock_llm_client_renders_clean_markdown() -> None:
    engine_input = _sample_input()
    engine_output = run_hypothesis_engine(engine_input)

    res = render_hypothesis_report(
        RenderReportRequest(engine_input=engine_input, engine_output=engine_output),
        llm_client=_FakeSuccessLLM(),
    )

    assert res.used_llm is True
    assert res.validation_passed is True
    assert res.violations == []
    assert "H2 is compatible" in res.markdown


def test_postprocessor_sanitizes_causal_claims_from_llm() -> None:
    engine_input = _sample_input()
    engine_output = run_hypothesis_engine(engine_input)

    res = render_hypothesis_report(
        RenderReportRequest(engine_input=engine_input, engine_output=engine_output),
        llm_client=_FakeCausalLLM(),
    )

    assert res.used_llm is True
    assert res.validation_passed is False
    assert any("elevated because of" in v.lower() for v in res.violations)
    assert "[removed_unsupported_causal_claim]" in res.markdown


def test_failing_llm_falls_back_to_deterministic_markdown() -> None:
    engine_input = _sample_input()
    engine_output = run_hypothesis_engine(engine_input)

    res = render_hypothesis_report(
        RenderReportRequest(engine_input=engine_input, engine_output=engine_output),
        llm_client=_FakeFailingLLM(),
    )

    assert res.used_llm is False
    assert res.validation_passed is True
    assert "Hypothesis Engine Summary: `Q-REN-001`" in res.markdown

def test_renderer_abstention_missing_fields_and_conflicts_sections() -> None:
    from hypothesis_engine_v0_1.schemas import DataStatus, RuleConflict

    engine_input = _sample_input()
    engine_output = run_hypothesis_engine(engine_input)
    engine_output.abstention_reason = "Missing mandatory fields."
    engine_output.missing_mandatory_fields = ["context.organism"]
    engine_output.context_conflicts = ["Oxygen mismatch"]
    engine_output.rule_conflicts = [
        RuleConflict(
            rule_ids=["R002", "R004"],
            affected_hypothesis_ids=["H1"],
            selected_status=DataStatus.COMPATIBLE,
            discarded_status=DataStatus.NOT_SUPPORTED_BY_DATA,
            reason="Test conflict",
        )
    ]

    res = render_hypothesis_report(
        RenderReportRequest(engine_input=engine_input, engine_output=engine_output),
        llm_client=None,
    )
    assert "Missing mandatory fields." in res.markdown
    assert "context.organism" in res.markdown
    assert "### Conflicts" in res.markdown
    assert "Oxygen mismatch" in res.markdown