from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from hypothesis_engine_v0_1.engine import run_hypothesis_engine
from hypothesis_engine_v0_1.postprocessor import compute_canonical_sha256
from hypothesis_engine_v0_1.renderer import (
    ExplanationLLMClient,
    RenderReportRequest,
    render_hypothesis_report,
)
from hypothesis_engine_v0_1.schemas import HypothesisEngineInput, HypothesisEngineOutput

router = APIRouter(prefix="/v0.1/hypothesis-engine", tags=["hypothesis-engine"])


class EvaluateAndRenderRequest(BaseModel):
    engine_input: HypothesisEngineInput
    max_bullets_per_section: int = Field(default=5, ge=1, le=20)
    strict_postprocess: bool = Field(default=True)


class EvaluateAndRenderResponse(BaseModel):
    engine_output: HypothesisEngineOutput
    markdown: str
    used_llm: bool
    validation_passed: bool
    violations: list[str] = Field(default_factory=list)
    input_hash: str
    output_hash: str
    explanation_hash: str | None = None


def get_llm_client() -> ExplanationLLMClient | None:
    return None


@router.post("/evaluate-and-render", response_model=EvaluateAndRenderResponse)
def evaluate_and_render_route(
    request: EvaluateAndRenderRequest,
    llm_client: ExplanationLLMClient | None = Depends(get_llm_client),
) -> EvaluateAndRenderResponse:
    engine_output = run_hypothesis_engine(request.engine_input)

    render_res = render_hypothesis_report(
        RenderReportRequest(
            engine_input=request.engine_input,
            engine_output=engine_output,
            max_bullets_per_section=request.max_bullets_per_section,
        ),
        llm_client=llm_client,
    )

    input_hash = compute_canonical_sha256(request.engine_input)
    output_hash = compute_canonical_sha256(engine_output)

    if request.strict_postprocess and not render_res.validation_passed:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Post-processor validation failed after rendering.",
                "violations": render_res.violations,
                "input_hash": input_hash,
                "output_hash": output_hash,
            },
        )

    return EvaluateAndRenderResponse(
        engine_output=engine_output,
        markdown=render_res.markdown,
        used_llm=render_res.used_llm,
        validation_passed=render_res.validation_passed,
        violations=render_res.violations,
        input_hash=input_hash,
        output_hash=output_hash,
        explanation_hash=render_res.explanation_hash,
    )


def create_app(llm_client: ExplanationLLMClient | None = None) -> FastAPI:
    app = FastAPI(title="BIOOS Hypothesis Engine v0.1")
    if llm_client is not None:
        app.dependency_overrides[get_llm_client] = lambda: llm_client
    app.include_router(router)
    return app
