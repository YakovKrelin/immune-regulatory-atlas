from hypothesis_engine_v0_1.engine import run_hypothesis_engine
from hypothesis_engine_v0_1.renderer import RenderReportRequest, render_hypothesis_report
from hypothesis_engine_v0_1.schemas import (
    BiologicalContext,
    ConditionContext,
    DataType,
    HypothesisEngineInput,
    Observation,
    SignalDirection,
    TargetSpec,
)


def test_report_generation_with_mocked_ollama() -> None:
    target = TargetSpec(entity="IL6")
    payload = HypothesisEngineInput(
        query_id="test_mock_report",
        data_type=DataType.PURIFIED_MACROPHAGE_CULTURE,
        cell_composition_changed=False,
        timepoint_known=True,
        target=target,
        context=BiologicalContext(
            organism="Homo sapiens",
            cell_type="macrophage",
            timepoint="6h",
            condition=ConditionContext(
                stimulus="hypoxia",
                oxygen="1% O2",
                duration="6h",
            ),
        ),
        observations=[
            Observation(observation_id="obs_1", entity="IL6", assay="RNA-seq", direction=SignalDirection.UP, value=2.4, fdr=0.001),
            Observation(observation_id="obs_2", entity="VEGFA", assay="RNA-seq", direction=SignalDirection.UP, value=2.0, fdr=0.002),
            Observation(observation_id="obs_3", entity="SLC2A1", assay="RNA-seq", direction=SignalDirection.UP, value=1.3, fdr=0.005),
            Observation(observation_id="obs_4", entity="NFKBIA", assay="RNA-seq", direction=SignalDirection.UNCHANGED, value=0.05, fdr=0.80),
            Observation(observation_id="obs_5", entity="TNFAIP3", assay="RNA-seq", direction=SignalDirection.UNCHANGED, value=-0.04, fdr=0.85),
        ],
    )
    engine_out = run_hypothesis_engine(payload)

    captured_calls: list[tuple[str, str]] = []

    class _MockLLMClient:
        def render_markdown(self, *, system_prompt: str, user_payload_json: str) -> str:
            captured_calls.append((system_prompt, user_payload_json))
            return (
                "### Summary\n"
                "- Under hypoxia (1% O2, 6h) in purified macrophages, H2 is compatible with current data "
                "and H1 is not supported by data.\n"
                "- This evaluation is hypothesis-generating and does not establish causal proof."
            )

    request = RenderReportRequest(
        engine_input=payload,
        engine_output=engine_out,
    )
    report = render_hypothesis_report(request, llm_client=_MockLLMClient())

    assert len(captured_calls) == 1
    assert captured_calls[0][0].strip()
    assert "test_mock_report" in captured_calls[0][1]
    assert report.used_llm is True
    assert report.validation_passed is True
    assert "H2" in report.markdown
