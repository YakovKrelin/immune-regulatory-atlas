from __future__ import annotations

import re
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class GroundedClaim(BaseModel):
    statement: str = Field(..., min_length=1)
    claim_kind: Literal["observation_summary", "evidence_summary", "prior_note", "hypothesis_status"]
    hypothesis_id: str | None = None
    cited_ids: list[str] = Field(default_factory=list, min_length=1)


class LLMExplanationPayload(BaseModel):
    query_id: str = Field(..., min_length=1)
    mode: Literal["conditional_inference", "weak_inference", "hard_abstention"]
    has_perturbation_evidence: bool = False
    claims: list[GroundedClaim] = Field(default_factory=list)
    introduced_hypothesis_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_no_unapproved_new_hypotheses_field(self) -> LLMExplanationPayload:
        if self.introduced_hypothesis_ids:
            raise ValueError(
                f"LLM is prohibited from introducing new hypotheses: {self.introduced_hypothesis_ids}"
            )
        return self


class PostValidationIssue(BaseModel):
    code: Literal[
        "missing_or_unknown_citation",
        "forbidden_causal_claim_without_perturbation",
        "unauthorized_hypothesis",
        "mode_mismatch",
    ]
    severity: Literal["error", "warning"] = "error"
    message: str


class PostValidationResult(BaseModel):
    is_valid: bool
    issues: list[PostValidationIssue] = Field(default_factory=list)
    sanitized_payload: LLMExplanationPayload | None = None


_CAUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcauses?\b", re.IGNORECASE),
    re.compile(r"\bcausing\b", re.IGNORECASE),
    re.compile(r"\bcausal\b", re.IGNORECASE),
    re.compile(r"\bdrives?\b", re.IGNORECASE),
    re.compile(r"\binduces?\b", re.IGNORECASE),
    re.compile(r"\bleads?\s+to\b", re.IGNORECASE),
    re.compile(r"\bresults?\s+in\b", re.IGNORECASE),
    re.compile(r"\bbecause\s+of\b", re.IGNORECASE),
    re.compile(r"\bвызывает\b", re.IGNORECASE),
    re.compile(r"\bприводит\s+к\b", re.IGNORECASE),
    re.compile(r"\bиз-за\b", re.IGNORECASE),
)


def get_llm_response_json_schema() -> dict[str, Any]:
    """озвращает строгую JSON Schema для structured output LLM-рендерера."""
    return LLMExplanationPayload.model_json_schema()


def _contains_causal_language(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CAUSAL_PATTERNS)


def validate_llm_explanation(
    payload: LLMExplanationPayload | dict[str, Any],
    *,
    allowed_source_ids: set[str],
    allowed_hypothesis_ids: set[str],
    expected_mode: str | None = None,
    has_perturbation_evidence: bool = False,
) -> PostValidationResult:
    parsed = (
        payload
        if isinstance(payload, LLMExplanationPayload)
        else LLMExplanationPayload.model_validate(payload)
    )
    issues: list[PostValidationIssue] = []

    if expected_mode is not None and parsed.mode != expected_mode:
        issues.append(
            PostValidationIssue(
                code="mode_mismatch",
                severity="error",
                message=f"LLM mode {parsed.mode!r} does not match engine mode {expected_mode!r}.",
            )
        )

    effective_perturbation = has_perturbation_evidence and parsed.has_perturbation_evidence

    for idx, claim in enumerate(parsed.claims):
        unknown_ids = [cid for cid in claim.cited_ids if cid not in allowed_source_ids]
        if not claim.cited_ids or unknown_ids:
            issues.append(
                PostValidationIssue(
                    code="missing_or_unknown_citation",
                    severity="error",
                    message=(
                        f"Claim[{idx}] has missing or unverified cited_ids: {unknown_ids} "
                        f"(provided={claim.cited_ids})."
                    ),
                )
            )

        if claim.hypothesis_id is not None and claim.hypothesis_id not in allowed_hypothesis_ids:
            issues.append(
                PostValidationIssue(
                    code="unauthorized_hypothesis",
                    severity="error",
                    message=(
                        f"Claim[{idx}] references hypothesis_id {claim.hypothesis_id!r} "
                        f"not in allowed hypotheses {sorted(allowed_hypothesis_ids)}."
                    ),
                )
            )

        if not effective_perturbation and _contains_causal_language(claim.statement):
            issues.append(
                PostValidationIssue(
                    code="forbidden_causal_claim_without_perturbation",
                    severity="error",
                    message=(
                        f"Claim[{idx}] states or implies causation without perturbation evidence: "
                        f"{claim.statement!r}"
                    ),
                )
            )

    is_valid = len(issues) == 0
    return PostValidationResult(
        is_valid=is_valid,
        issues=issues,
        sanitized_payload=parsed if is_valid else None,
    )
