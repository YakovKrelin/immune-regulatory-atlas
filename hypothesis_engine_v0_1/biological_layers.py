from __future__ import annotations

import re
from pydantic import BaseModel, Field

from hypothesis_engine_v0_1.schemas import (
    BiologicalContext,
    DataType,
    InferenceMode,
    MeasurementLevel,
    Observation,
    TargetSpec,
    parse_oxygen_percent,
)

ENTITY_SYNONYMS: dict[str, tuple[str, str]] = {
    "il6": ("IL6", "gene_or_protein"),
    "il-6": ("IL6", "gene_or_protein"),
    "interleukin-6": ("IL6", "gene_or_protein"),
    "interleukin 6": ("IL6", "gene_or_protein"),
    "hif1a": ("HIF1A", "gene_or_protein"),
    "hif-1a": ("HIF1A", "gene_or_protein"),
    "hif1alpha": ("HIF1A", "gene_or_protein"),
    "hif-1alpha": ("HIF1A", "gene_or_protein"),
    "il1b": ("IL1B", "gene_or_protein"),
    "il-1b": ("IL1B", "gene_or_protein"),
    "il1beta": ("IL1B", "gene_or_protein"),
    "il-1beta": ("IL1B", "gene_or_protein"),
    "tnf": ("TNF", "gene_or_protein"),
    "tnf-alpha": ("TNF", "gene_or_protein"),
    "tnfa": ("TNF", "gene_or_protein"),
    "nfkb1": ("NFKB1", "gene_or_protein"),
    "nf-kb": ("NFKB_COMPLEX", "pathway_or_complex"),
    "nfkb": ("NFKB_COMPLEX", "pathway_or_complex"),
    "macrophage": ("macrophage", "cell_type"),
    "macrophages": ("macrophage", "cell_type"),
    "mphi": ("macrophage", "cell_type"),
    "mφ": ("macrophage", "cell_type"),
    "epithelial cell": ("epithelial_cell", "cell_type"),
    "epithelial cells": ("epithelial_cell", "cell_type"),
    "endothelial cell": ("endothelial_cell", "cell_type"),
    "endothelial cells": ("endothelial_cell", "cell_type"),
    "hypoxia": ("hypoxia", "condition"),
    "low oxygen": ("hypoxia", "condition"),
    "normoxia": ("normoxia", "condition"),
}


class ResolvedEntity(BaseModel):
    raw_value: str
    canonical_id: str
    entity_kind: str
    matched_alias: bool


class MeasurementCompatibilityResult(BaseModel):
    target_level: MeasurementLevel
    observation_level: MeasurementLevel
    status: str
    is_compatible: bool
    reason: str


class ContextValidationResult(BaseModel):
    mode: InferenceMode
    missing_mandatory_fields: list[str] = Field(default_factory=list)
    missing_context_fields: list[str] = Field(default_factory=list)
    context_conflicts: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None


def _normalize_key(raw: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw.strip().lower())
    return cleaned


def resolve_biological_entity(raw_value: str) -> ResolvedEntity:
    stripped = (raw_value or "").strip()
    key = _normalize_key(stripped)
    if key in ENTITY_SYNONYMS:
        canonical, kind = ENTITY_SYNONYMS[key]
        return ResolvedEntity(
            raw_value=stripped,
            canonical_id=canonical,
            entity_kind=kind,
            matched_alias=True,
        )
    fallback_id = re.sub(r"[^A-Za-z0-9_]+", "_", stripped).strip("_").upper() or "UNKNOWN"
    return ResolvedEntity(
        raw_value=stripped,
        canonical_id=fallback_id,
        entity_kind="unknown",
        matched_alias=False,
    )


def validate_measurement_compatibility(
    target_level: MeasurementLevel,
    observation_level: MeasurementLevel,
) -> MeasurementCompatibilityResult:
    if target_level == MeasurementLevel.NOT_SPECIFIED or observation_level == MeasurementLevel.NOT_SPECIFIED:
        return MeasurementCompatibilityResult(
            target_level=target_level,
            observation_level=observation_level,
            status="unspecified_level",
            is_compatible=False,
            reason="Measurement level is not specified on target or observation.",
        )

    if target_level == observation_level:
        return MeasurementCompatibilityResult(
            target_level=target_level,
            observation_level=observation_level,
            status="exact_match",
            is_compatible=True,
            reason=f"Measurement level matches ({target_level.value}).",
        )

    if {target_level, observation_level} == {
        MeasurementLevel.TOTAL_PROTEIN,
        MeasurementLevel.SECRETED_PROTEIN,
    }:
        return MeasurementCompatibilityResult(
            target_level=target_level,
            observation_level=observation_level,
            status="compartment_shift",
            is_compatible=False,
            reason="Intracellular/total protein differs from secreted protein without secretion assay.",
        )

    return MeasurementCompatibilityResult(
        target_level=target_level,
        observation_level=observation_level,
        status="incompatible_level",
        is_compatible=False,
        reason=(
            f"Observation level ({observation_level.value}) does not match "
            f"target level ({target_level.value})."
        ),
    )


def evaluate_mandatory_context(
    context: BiologicalContext,
    *,
    target: TargetSpec | None = None,
    observations: list[Observation] | None = None,
    data_type: DataType | None = None,
    cell_composition_changed: bool | None = None,
) -> ContextValidationResult:
    missing_mandatory: list[str] = []
    missing_soft: list[str] = []
    conflicts: list[str] = []

    if not (context.organism or "").strip():
        missing_mandatory.append("organism")
    if not (context.cell_type or "").strip():
        missing_mandatory.append("cell_type")

    cond = context.condition
    has_condition = any(
        bool((v or "").strip())
        for v in (cond.oxygen, cond.oxygen_level, cond.stimulus, cond.priming, cond.co_stimulus)
    )
    if not has_condition:
        missing_mandatory.append("condition")

    control = (context.study_design.control or "").strip()
    if not control:
        missing_mandatory.append("study_design.control")

    if target is not None and target.level == MeasurementLevel.NOT_SPECIFIED:
        missing_mandatory.append("target.level")

    if not (context.timepoint or "").strip() and not (cond.duration or "").strip():
        missing_soft.append("timepoint_or_duration")
    if not (context.tissue or "").strip():
        missing_soft.append("tissue")
    if context.study_design.replicates is None:
        missing_soft.append("study_design.replicates")

    ox_pct = parse_oxygen_percent(cond.oxygen_level or cond.oxygen)
    cond_text = " ".join(
        part.strip().lower()
        for part in (cond.oxygen, cond.oxygen_level, cond.stimulus)
        if part and part.strip()
    )
    if ox_pct is not None and ox_pct < 5.0 and "normox" in cond_text:
        conflicts.append("hypoxic_oxygen_percent_with_normoxia_label")
    if ox_pct is not None and ox_pct >= 18.0 and "hypox" in cond_text:
        conflicts.append("normoxic_oxygen_percent_with_hypoxia_label")

    if data_type in {DataType.BULK_RNA_SEQ, DataType.BULK_TISSUE_RNA_SEQ} and cell_composition_changed is True:
        conflicts.append("bulk_signal_with_changed_cell_composition")

    if observations and target is not None and target.level != MeasurementLevel.NOT_SPECIFIED:
        mismatched = [
            obs.observation_id
            for obs in observations
            if not validate_measurement_compatibility(target.level, obs.level).is_compatible
        ]
        if mismatched:
            conflicts.append(f"measurement_level_mismatch:{','.join(mismatched)}")

    if missing_mandatory or conflicts:
        reasons: list[str] = []
        if missing_mandatory:
            reasons.append(f"missing mandatory fields: {', '.join(missing_mandatory)}")
        if conflicts:
            reasons.append(f"context/measurement conflicts: {', '.join(conflicts)}")
        return ContextValidationResult(
            mode=InferenceMode.HARD_ABSTENTION,
            missing_mandatory_fields=missing_mandatory,
            missing_context_fields=missing_soft,
            context_conflicts=conflicts,
            abstention_reason="; ".join(reasons),
        )

    if missing_soft or data_type in {DataType.BULK_RNA_SEQ, DataType.BULK_TISSUE_RNA_SEQ}:
        return ContextValidationResult(
            mode=InferenceMode.WEAK_INFERENCE,
            missing_mandatory_fields=[],
            missing_context_fields=missing_soft,
            context_conflicts=[],
            abstention_reason=None,
        )

    return ContextValidationResult(
        mode=InferenceMode.CONDITIONAL_INFERENCE,
        missing_mandatory_fields=[],
        missing_context_fields=[],
        context_conflicts=[],
        abstention_reason=None,
    )
