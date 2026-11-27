from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable
from pydantic import BaseModel, Field, ValidationError

from hypothesis_engine_v0_1.llm_contract import (
    GroundedClaim,
    LLMExplanationPayload,
    PostValidationIssue,
    get_llm_response_json_schema,
    validate_llm_explanation,
)

DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"


class RenderedReportResult(BaseModel):
    query_id: str
    model_used: str
    used_fallback: bool
    payload: LLMExplanationPayload
    validation_issues: list[PostValidationIssue] = Field(default_factory=list)


def get_configured_ollama_model() -> str:
    val = (os.getenv("IRA_OLLAMA_MODEL") or "").strip()
    return val if val else DEFAULT_OLLAMA_MODEL


def build_deterministic_fallback(
    *,
    query_id: str,
    mode: str,
    allowed_source_ids: list[str] | set[str],
    allowed_hypothesis_ids: list[str] | set[str],
    has_perturbation_evidence: bool = False,
    summary_note: str = "Observed data only without causal perturbation.",
) -> LLMExplanationPayload:
    valid_mode: str = (
        mode if mode in {"conditional_inference", "weak_inference", "hard_abstention"} else "weak_inference"
    )
    src_list = sorted(s for s in allowed_source_ids if s)
    hyp_list = sorted(h for h in allowed_hypothesis_ids if h)
    default_hyp = hyp_list[0] if hyp_list else None

    claims: list[GroundedClaim] = []
    if src_list:
        claims.append(
            GroundedClaim(
                statement=f"{summary_note} Referenced verified sources: {', '.join(src_list)}.",
                claim_kind="observation_summary",
                hypothesis_id=default_hyp,
                cited_ids=src_list,
            )
        )

    return LLMExplanationPayload(
        query_id=query_id,
        mode=valid_mode,  # type: ignore[arg-type]
        has_perturbation_evidence=has_perturbation_evidence,
        claims=claims,
        introduced_hypothesis_ids=[],
        limitations=[
            "Deterministic non-causal fallback summary generated from verified engine inputs.",
            "No causal mechanism is claimed without perturbation evidence.",
        ],
    )


def default_ollama_chat_caller(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    timeout_sec: float = 30.0,
) -> str:
    body = {
        "model": model,
        "stream": False,
        "format": schema,
        "messages": messages,
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        url=base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw_text = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Ollama request failed for model {model!r}: {exc}") from exc

    parsed = json.loads(raw_text)
    msg = parsed.get("message") if isinstance(parsed, dict) else None
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama returned empty message content.")
    return content


def render_with_ollama_guardrail(
    *,
    query_id: str,
    mode: str,
    context_summary: str,
    allowed_source_ids: set[str],
    allowed_hypothesis_ids: set[str],
    has_perturbation_evidence: bool = False,
    model: str | None = None,
    ollama_url: str | None = None,
    chat_caller: Callable[..., str] | None = None,
) -> RenderedReportResult:
    selected_model = (model or "").strip() or get_configured_ollama_model()
    endpoint = (ollama_url or "").strip() or os.getenv("IRA_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    caller = chat_caller or default_ollama_chat_caller
    schema = get_llm_response_json_schema()

    system_prompt = (
        "You are a factual scientific renderer for Immune Regulatory Atlas. "
        "Output ONLY JSON matching the provided schema. "
        f"Allowed source IDs: {sorted(allowed_source_ids)}. "
        f"Allowed hypothesis IDs: {sorted(allowed_hypothesis_ids)}. "
        f"Perturbation evidence present: {has_perturbation_evidence}. "
        "Never state causal claims when perturbation evidence is False. "
        "Never introduce new hypotheses."
    )
    user_prompt = f"query_id={query_id}; mode={mode}; context={context_summary}"

    fallback_payload = build_deterministic_fallback(
        query_id=query_id,
        mode=mode,
        allowed_source_ids=allowed_source_ids,
        allowed_hypothesis_ids=allowed_hypothesis_ids,
        has_perturbation_evidence=has_perturbation_evidence,
    )

    try:
        raw_content = caller(
            base_url=endpoint,
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema=schema,
        )
        raw_obj = json.loads(raw_content)
        val_res = validate_llm_explanation(
            raw_obj,
            allowed_source_ids=allowed_source_ids,
            allowed_hypothesis_ids=allowed_hypothesis_ids,
            expected_mode=mode,
            has_perturbation_evidence=has_perturbation_evidence,
        )
        if val_res.is_valid and val_res.sanitized_payload is not None:
            return RenderedReportResult(
                query_id=query_id,
                model_used=selected_model,
                used_fallback=False,
                payload=val_res.sanitized_payload,
                validation_issues=[],
            )
        return RenderedReportResult(
            query_id=query_id,
            model_used=selected_model,
            used_fallback=True,
            payload=fallback_payload,
            validation_issues=val_res.issues,
        )
    except (RuntimeError, ValueError, TypeError, json.JSONDecodeError, ValidationError) as exc:
        return RenderedReportResult(
            query_id=query_id,
            model_used=selected_model,
            used_fallback=True,
            payload=fallback_payload,
            validation_issues=[
                PostValidationIssue(
                    code="missing_or_unknown_citation",
                    severity="error",
                    message=f"Fallback triggered due to renderer error: {exc}",
                )
            ],
        )
