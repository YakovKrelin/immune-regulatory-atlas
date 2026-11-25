from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from hypothesis_engine_v0_1.schemas import HypothesisEngineInput, HypothesisEngineOutput

router = APIRouter(prefix="/v0.1/hypothesis-engine", tags=["hypothesis-engine-postprocessor"])

FORBIDDEN_NUMERIC_KEYS: frozenset[str] = frozenset(
    {
        "posterior_probability",
        "probability",
        "calibrated_probability",
        "confidence",
        "confidence_score",
        "support_score",
        "likelihood",
        "score",
    }
)

DISALLOWED_CAUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bis elevated because of\b", re.IGNORECASE),
    re.compile(r"\belevated because of\b", re.IGNORECASE),
    re.compile(r"\bcauses\b", re.IGNORECASE),
    re.compile(r"\bcaused by\b", re.IGNORECASE),
    re.compile(r"\bdirectly drives\b", re.IGNORECASE),
    re.compile(r"\bproves causality\b", re.IGNORECASE),
    re.compile(r"\bestablishes causality\b", re.IGNORECASE),
)

ALLOWED_PRIOR_NOTE_KEYS: frozenset[str] = frozenset(
    {
        "note_id",
        "description",
        "status",
        "citable_as_evidence",
        "requires_curator_review",
    }
)


class PostProcessorRequest(BaseModel):
    engine_input: HypothesisEngineInput
    engine_output: dict[str, Any]
    rendered_explanation: str | None = None
    allowed_marker_ids: set[str] | None = None
    allowed_evidence_ids: set[str] | None = None


class PostProcessorResponse(BaseModel):
    valid: bool
    violations: list[str] = Field(default_factory=list)
    input_hash: str
    output_hash: str
    explanation_hash: str | None = None
    validated_output: HypothesisEngineOutput | None = None
    sanitized_explanation: str | None = None


def compute_canonical_sha256(payload: Any) -> str:
    if isinstance(payload, BaseModel):
        raw = payload.model_dump(mode="json")
    else:
        raw = payload
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _collect_allowed_markers(
    engine_input: HypothesisEngineInput,
    explicit_markers: set[str] | None,
) -> set[str]:
    markers: set[str] = {obs.entity.strip().upper() for obs in engine_input.observations if obs.entity}
    for panel in (
        engine_input.nfkb_activity_panel,
        engine_input.nfkb_target_panel,
        engine_input.hif_panel,
        engine_input.orthogonal_state_panel,
        *engine_input.extra_panels,
    ):
        markers.update(m.strip().upper() for m in panel.core_markers if m)
        markers.update(m.strip().upper() for m in panel.supporting_markers if m)
    if explicit_markers:
        markers.update(m.strip().upper() for m in explicit_markers if m)
    return markers


def _collect_allowed_evidence_ids(
    engine_input: HypothesisEngineInput,
    explicit_evidence_ids: set[str] | None,
) -> set[str]:
    ev_ids: set[str] = {
        rec.evidence_id.strip() for rec in engine_input.evidence_records if rec.evidence_id
    }
    if explicit_evidence_ids:
        ev_ids.update(eid.strip() for eid in explicit_evidence_ids if eid)
    return ev_ids


def _find_forbidden_numeric_fields(obj: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).strip().lower()
            child_path = f"{path}.{key}"
            if key_lower in FORBIDDEN_NUMERIC_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
                findings.append(f"Forbidden numeric field '{child_path}' = {value}")
            findings.extend(_find_forbidden_numeric_fields(value, child_path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            findings.extend(_find_forbidden_numeric_fields(item, f"{path}[{idx}]"))
    return findings


def _find_disallowed_causal_claims(text: str | None) -> list[str]:
    if not text:
        return []
    hits: list[str] = []
    for pattern in DISALLOWED_CAUSAL_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(f"Disallowed causal phrasing matched '{match.group(0)}' in text.")
    return hits


def validate_and_postprocess(request: PostProcessorRequest) -> PostProcessorResponse:
    violations: list[str] = []

    violations.extend(_find_forbidden_numeric_fields(request.engine_output))

    parsed_output: HypothesisEngineOutput | None = None
    try:
        parsed_output = HypothesisEngineOutput.model_validate(request.engine_output)
    except ValidationError as exc:
        violations.append(f"Schema validation error: {exc}")

    allowed_markers = _collect_allowed_markers(request.engine_input, request.allowed_marker_ids)
    allowed_evidence_ids = _collect_allowed_evidence_ids(
        request.engine_input, request.allowed_evidence_ids
    )

    if parsed_output is not None:
        if parsed_output.query_id != request.engine_input.query_id:
            violations.append(
                f"query_id mismatch: output '{parsed_output.query_id}' != input '{request.engine_input.query_id}'."
            )

        for h_id, h_eval in parsed_output.hypotheses.items():
            for ev_id in h_eval.supporting_evidence:
                if ev_id not in allowed_evidence_ids:
                    violations.append(
                        f"Unknown supporting_evidence ID '{ev_id}' on hypothesis '{h_id}'."
                    )
            for ev_id in h_eval.contradicting_evidence:
                if ev_id not in allowed_evidence_ids:
                    violations.append(
                        f"Unknown contradicting_evidence ID '{ev_id}' on hypothesis '{h_id}'."
                    )
            if h_eval.supporting_evidence and h_eval.contradicting_evidence and not h_eval.evidence_conflict:
                violations.append(
                    f"Hypothesis '{h_id}' has both supporting and contradicting evidence but evidence_conflict is false."
                )

        measurement_ids = {m.measurement_id for m in parsed_output.discriminating_measurements}
        for rel in parsed_output.pairwise_relations:
            if (
                rel.discriminating_measurement_id is not None
                and rel.discriminating_measurement_id not in measurement_ids
            ):
                violations.append(
                    f"Pairwise relation {rel.hypotheses} references unknown measurement ID "
                    f"'{rel.discriminating_measurement_id}'."
                )

        for note in parsed_output.model_prior_notes:
            if note.status != "model_prior_knowledge" or note.citable_as_evidence is not False:
                violations.append(
                    f"Model prior note '{note.note_id}' must have status='model_prior_knowledge' and citable_as_evidence=False."  # pragma: no cover
                )

        violations.extend(_find_disallowed_causal_claims(parsed_output.abstention_reason))
        for h_eval in parsed_output.hypotheses.values():
            violations.extend(_find_disallowed_causal_claims(h_eval.rationale))
            if h_eval.resolution:
                violations.extend(_find_disallowed_causal_claims(h_eval.resolution))

    raw_obs = request.engine_output.get("observations")
    if isinstance(raw_obs, list):
        for idx, item in enumerate(raw_obs):
            if isinstance(item, dict) and "entity" in item and item["entity"]:
                entity_norm = str(item["entity"]).strip().upper()
                if entity_norm not in allowed_markers:
                    violations.append(
                        f"Unknown marker/entity '{item['entity']}' at observations[{idx}]."
                    )

    explanation_violations = _find_disallowed_causal_claims(request.rendered_explanation)
    violations.extend(explanation_violations)

    sanitized_explanation = request.rendered_explanation
    if sanitized_explanation is not None and explanation_violations:
        for pattern in DISALLOWED_CAUSAL_PATTERNS:
            sanitized_explanation = pattern.sub("[removed_unsupported_causal_claim]", sanitized_explanation)

    input_hash = compute_canonical_sha256(request.engine_input)
    output_hash = compute_canonical_sha256(request.engine_output)
    explanation_hash = (
        hashlib.sha256(sanitized_explanation.encode("utf-8")).hexdigest()
        if sanitized_explanation is not None
        else None
    )

    return PostProcessorResponse(
        valid=len(violations) == 0,
        violations=violations,
        input_hash=input_hash,
        output_hash=output_hash,
        explanation_hash=explanation_hash,
        validated_output=parsed_output if len(violations) == 0 else None,
        sanitized_explanation=sanitized_explanation,
    )


@router.post("/postprocess", response_model=PostProcessorResponse)
def postprocess_endpoint(request: PostProcessorRequest) -> PostProcessorResponse:
    result = validate_and_postprocess(request)
    if not result.valid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Post-processor validation failed.",
                "violations": result.violations,
                "input_hash": result.input_hash,
                "output_hash": result.output_hash,
            },
        )
    return result
