from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from hypothesis_engine_v0_1.postprocessor import (
    PostProcessorRequest,
    validate_and_postprocess,
)
from hypothesis_engine_v0_1.schemas import HypothesisEngineInput, HypothesisEngineOutput


class ExplanationLLMClient(Protocol):
    def render_markdown(self, *, system_prompt: str, user_payload_json: str) -> str:
        """Render concise Markdown strictly from the provided JSON payload."""


RENDERER_SYSTEM_PROMPT = (
    "You are a concise scientific report renderer for BIOOS Hypothesis Engine v0.1. "
    "Format the provided computed JSON into a short Markdown summary. "
    "Do not change mode, evidence_status, data_status, decision_status, or pairwise relations. "
    "Do not invent evidence IDs, markers, or numeric probabilities. "
    "Do not state causal claims unless explicitly supported by perturbation evidence in the payload. "
    "Keep model_prior_notes clearly labeled as non-citable prior notes."
)


class RenderReportRequest(BaseModel):
    engine_input: HypothesisEngineInput
    engine_output: HypothesisEngineOutput
    max_bullets_per_section: int = Field(default=5, ge=1, le=20)


class RenderReportResponse(BaseModel):
    markdown: str
    used_llm: bool
    validation_passed: bool
    violations: list[str] = Field(default_factory=list)
    input_hash: str
    output_hash: str
    explanation_hash: str | None = None


def render_deterministic_markdown(
    output: HypothesisEngineOutput,
    *,
    max_bullets: int = 5,
) -> str:
    lines: list[str] = [
        f"## Hypothesis Engine Summary: `{output.query_id}`",
        f"- **Scenario:** `{output.scenario}` (`{output.rule_set_version}`)",
        f"- **Inference Mode:** `{output.mode.value}`",
    ]

    if output.abstention_reason:
        lines.append(f"- **Abstention Reason:** {output.abstention_reason}")

    if output.missing_mandatory_fields:
        lines.append(
            "- **Missing Mandatory Fields:** " + ", ".join(output.missing_mandatory_fields)
        )

    if output.program_statuses:
        prog_items = [
            f"`{k}`={v.value}" for k, v in list(output.program_statuses.items())[:max_bullets]
        ]
        lines.append("- **Program Statuses:** " + ", ".join(prog_items))

    if output.hypotheses:
        lines.append("### Hypotheses")
        for h_id in sorted(output.hypotheses.keys())[:max_bullets]:
            h = output.hypotheses[h_id]
            lines.append(
                f"- **{h.hypothesis_id} ({h.name}):** "
                f"data=`{h.data_status.value}`, decision=`{h.decision_status.value}`, "
                f"evidence=`{h.evidence_status.value}`"
            )

    if output.pairwise_relations:
        lines.append("### Pairwise Relations")
        for rel in output.pairwise_relations[:max_bullets]:
            pair = f"{rel.hypotheses[0]} vs {rel.hypotheses[1]}"
            lines.append(f"- **{pair}:** `{rel.relation.value}` — {rel.reason}")

    if output.rule_conflicts or output.context_conflicts:
        lines.append("### Conflicts")
        for c in output.context_conflicts[:max_bullets]:
            lines.append(f"- Context conflict: {c}")
        for rc in output.rule_conflicts[:max_bullets]:
            lines.append(
                f"- Rule conflict on {', '.join(rc.affected_hypothesis_ids)}: "
                f"selected `{rc.selected_status.value}`, discarded `{rc.discarded_status.value}`"
            )

    if output.discriminating_measurements:
        lines.append("### Priority Measurements")
        for m in output.discriminating_measurements[:max_bullets]:
            lines.append(
                f"- **P{m.priority} (`{m.measurement_id}`):** {m.measurement} "
                f"({', '.join(m.recommended_assays)})"
            )

    if output.limitations:
        lines.append("### Limitations")
        for lim in output.limitations[:max_bullets]:
            lines.append(f"- {lim}")

    return "\n".join(lines)


def render_hypothesis_report(
    request: RenderReportRequest,
    *,
    llm_client: ExplanationLLMClient | None = None,
) -> RenderReportResponse:
    used_llm = False
    raw_markdown: str

    if llm_client is not None:
        try:
            raw_markdown = llm_client.render_markdown(
                system_prompt=RENDERER_SYSTEM_PROMPT,
                user_payload_json=request.engine_output.model_dump_json(indent=2),
            )
            used_llm = True
        except Exception:
            raw_markdown = render_deterministic_markdown(
                request.engine_output,
                max_bullets=request.max_bullets_per_section,
            )
    else:
        raw_markdown = render_deterministic_markdown(
            request.engine_output,
            max_bullets=request.max_bullets_per_section,
        )

    post_res = validate_and_postprocess(
        PostProcessorRequest(
            engine_input=request.engine_input,
            engine_output=request.engine_output.model_dump(mode="json"),
            rendered_explanation=raw_markdown,
        )
    )

    final_markdown = (
        post_res.sanitized_explanation
        if post_res.sanitized_explanation is not None
        else raw_markdown
    )

    return RenderReportResponse(
        markdown=final_markdown,
        used_llm=used_llm,
        validation_passed=post_res.valid,
        violations=post_res.violations,
        input_hash=post_res.input_hash,
        output_hash=post_res.output_hash,
        explanation_hash=post_res.explanation_hash,
    )
