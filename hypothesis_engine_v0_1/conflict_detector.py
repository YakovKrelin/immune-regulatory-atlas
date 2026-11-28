from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field

from hypothesis_engine_v0_1.biological_layers import (
    resolve_biological_entity,
    validate_measurement_compatibility,
)
from hypothesis_engine_v0_1.schemas import (
    BiologicalContext,
    DataType,
    EvidenceRecord,
    MeasurementLevel,
    ModelPriorKnowledgeNote,
    Observation,
    SignalDirection,
    TargetSpec,
    parse_oxygen_percent,
)


class NegativeEvidenceStatus(str, Enum):
    NO_EFFECT_OBSERVED = "no_effect_observed"
    NOT_REPLICATED = "not_replicated"
    REFUTED = "refuted"
    RETRACTED = "retracted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DetectedConflict(BaseModel):
    conflict_type: Literal[
        "context_conflict",
        "evidence_conflict",
        "measurement_level_conflict",
        "direction_conflict",
        "species_conflict",
        "cell_type_conflict",
        "timepoint_conflict",
        "rule_conflict",
    ]
    severity: Literal["low", "medium", "high"] = "medium"
    auto_resolved: Literal[False] = False
    involved_ids: list[str] = Field(default_factory=list)
    description: str


class ProvenanceRecord(BaseModel):
    claim: str
    source_type: Literal["user_observation", "evidence_record", "model_prior_knowledge"]
    source_id: str
    generated_by: str
    citable_as_evidence: bool
    requires_curator_review: bool = False
    negative_evidence_status: NegativeEvidenceStatus | None = None


class ConflictAndProvenanceReport(BaseModel):
    conflicts: list[DetectedConflict] = Field(default_factory=list)
    negative_evidence_items: list[ProvenanceRecord] = Field(default_factory=list)
    provenance_records: list[ProvenanceRecord] = Field(default_factory=list)


def classify_negative_evidence(
    evidence: EvidenceRecord,
    *,
    replicated: bool | None = None,
    effect_observed: bool | None = None,
) -> NegativeEvidenceStatus:
    text = f"{evidence.claim_summary} {evidence.evidence_type} {evidence.assertion_method}".lower()
    if "retract" in text:
        return NegativeEvidenceStatus.RETRACTED
    if effect_observed is False or "no effect" in text or "null effect" in text:
        return NegativeEvidenceStatus.NO_EFFECT_OBSERVED
    if replicated is False or "not replicated" in text or "failed to replicate" in text:
        return NegativeEvidenceStatus.NOT_REPLICATED
    if "refut" in text or "contradict" in text or bool(evidence.contradicts_hypothesis_ids):
        return NegativeEvidenceStatus.REFUTED
    return NegativeEvidenceStatus.INSUFFICIENT_EVIDENCE


def detect_conflicts(
    *,
    context: BiologicalContext,
    target: TargetSpec,
    observations: list[Observation],
    evidence_records: list[EvidenceRecord] | None = None,
    data_type: DataType | None = None,
    cell_composition_changed: bool | None = None,
) -> list[DetectedConflict]:
    conflicts: list[DetectedConflict] = []
    ev_list = evidence_records or []

    by_entity: dict[str, list[Observation]] = {}
    for obs in observations:
        resolved = resolve_biological_entity(obs.entity)
        by_entity.setdefault(resolved.canonical_id, []).append(obs)

        compat = validate_measurement_compatibility(target.level, obs.level)
        if not compat.is_compatible and compat.status != "unspecified_level":
            conflicts.append(
                DetectedConflict(
                    conflict_type="measurement_level_conflict",
                    severity="high",
                    involved_ids=[obs.observation_id],
                    description=f"{obs.observation_id} ({resolved.canonical_id}): {compat.reason}",
                )
            )

    for canonical_id, group in by_entity.items():
        if len(group) < 2:
            continue
        directions = {obs.resolve_direction() for obs in group if obs.resolve_direction() != SignalDirection.NOT_MEASURED}
        if SignalDirection.UP in directions and SignalDirection.DOWN in directions:
            conflicts.append(
                DetectedConflict(
                    conflict_type="direction_conflict",
                    severity="high",
                    involved_ids=[o.observation_id for o in group],
                    description=(
                        f"Conflicting UP and DOWN directions for canonical entity {canonical_id} "
                        f"across {[o.observation_id for o in group]}."
                    ),
                )
            )

        levels = {obs.level for obs in group if obs.level != MeasurementLevel.NOT_SPECIFIED}
        if len(levels) > 1:
            conflicts.append(
                DetectedConflict(
                    conflict_type="measurement_level_conflict",
                    severity="medium",
                    involved_ids=[o.observation_id for o in group],
                    description=(
                        f"Multiple measurement levels ({sorted(l.value for l in levels)}) "
                        f"observed for {canonical_id} without harmonization."
                    ),
                )
            )

        timepoints = {(obs.timepoint or "").strip() for obs in group if (obs.timepoint or "").strip()}
        if len(timepoints) > 1:
            conflicts.append(
                DetectedConflict(
                    conflict_type="timepoint_conflict",
                    severity="low",
                    involved_ids=[o.observation_id for o in group],
                    description=(
                        f"Observations for {canonical_id} span multiple timepoints "
                        f"({sorted(timepoints)}) without longitudinal alignment."
                    ),
                )
            )

    ox_pct = parse_oxygen_percent(context.condition.oxygen_level or context.condition.oxygen)
    cond_text = " ".join(
        part.strip().lower()
        for part in (context.condition.oxygen, context.condition.oxygen_level, context.condition.stimulus)
        if part and part.strip()
    )
    if (ox_pct is not None and ox_pct < 5.0 and "normox" in cond_text) or (
        ox_pct is not None and ox_pct >= 18.0 and "hypox" in cond_text
    ):
        conflicts.append(
            DetectedConflict(
                conflict_type="context_conflict",
                severity="high",
                involved_ids=[context.organism or "unknown_organism"],
                description=(
                    f"Oxygen percentage ({ox_pct}%) contradicts condition text "
                    f"({context.condition.oxygen_level or context.condition.oxygen})."
                ),
            )
        )

    if data_type in {DataType.BULK_RNA_SEQ, DataType.BULK_TISSUE_RNA_SEQ} and cell_composition_changed is True:
        conflicts.append(
            DetectedConflict(
                conflict_type="rule_conflict",
                severity="high",
                involved_ids=[o.observation_id for o in observations],
                description=(
                    "Bulk expression shift coincides with changed cell composition; "
                    "cannot attribute solely to per-cell transcriptional activation."
                ),
            )
        )

    for ev in ev_list:
        overlap = set(ev.supports_hypothesis_ids) & set(ev.contradicts_hypothesis_ids)
        if overlap:
            conflicts.append(
                DetectedConflict(
                    conflict_type="evidence_conflict",
                    severity="medium",
                    involved_ids=[ev.evidence_id],
                    description=(
                        f"EvidenceRecord {ev.evidence_id} lists conflicting support and "
                        f"contradiction for hypotheses: {sorted(overlap)}."
                    ),
                )
            )

    return conflicts


def build_provenance_report(
    *,
    context: BiologicalContext,
    target: TargetSpec,
    observations: list[Observation],
    evidence_records: list[EvidenceRecord] | None = None,
    model_prior_notes: list[ModelPriorKnowledgeNote] | None = None,
    data_type: DataType | None = None,
    cell_composition_changed: bool | None = None,
) -> ConflictAndProvenanceReport:
    ev_list = evidence_records or []
    prior_list = model_prior_notes or []
    conflicts = detect_conflicts(
        context=context,
        target=target,
        observations=observations,
        evidence_records=ev_list,
        data_type=data_type,
        cell_composition_changed=cell_composition_changed,
    )

    records: list[ProvenanceRecord] = []
    neg_items: list[ProvenanceRecord] = []

    for obs in observations:
        resolved = resolve_biological_entity(obs.entity)
        direction = obs.resolve_direction().value
        records.append(
            ProvenanceRecord(
                claim=f"{resolved.canonical_id} ({obs.level.value}) direction={direction}",
                source_type="user_observation",
                source_id=obs.observation_id,
                generated_by=obs.source or "user_upload",
                citable_as_evidence=True,
                requires_curator_review=False,
            )
        )

    for ev in ev_list:
        neg_status: NegativeEvidenceStatus | None = None
        is_neg = bool(ev.contradicts_hypothesis_ids) or any(
            token in f"{ev.claim_summary} {ev.evidence_type}".lower()
            for token in ("no effect", "null", "refut", "retract", "not replicated")
        )
        if is_neg:
            neg_status = classify_negative_evidence(ev)
        rec = ProvenanceRecord(
            claim=ev.claim_summary,
            source_type="evidence_record",
            source_id=ev.evidence_id,
            generated_by=ev.assertion_method,
            citable_as_evidence=ev.curated and neg_status != NegativeEvidenceStatus.RETRACTED,
            requires_curator_review=not ev.curated or neg_status == NegativeEvidenceStatus.RETRACTED,
            negative_evidence_status=neg_status,
        )
        records.append(rec)
        if neg_status is not None:
            neg_items.append(rec)

    for note in prior_list:
        records.append(
            ProvenanceRecord(
                claim=note.description,
                source_type="model_prior_knowledge",
                source_id=note.note_id,
                generated_by="llm_prior",
                citable_as_evidence=False,
                requires_curator_review=True,
            )
        )

    return ConflictAndProvenanceReport(
        conflicts=conflicts,
        negative_evidence_items=neg_items,
        provenance_records=records,
    )
