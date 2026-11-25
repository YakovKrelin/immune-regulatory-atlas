from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DataStatus(str, Enum):
    SUPPORTED = "supported"
    COMPATIBLE = "compatible"
    NOT_SUPPORTED_BY_DATA = "not_supported_by_data"
    CONTRADICTED = "contradicted"
    NOT_ASSESSED = "not_assessed"


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    PLAUSIBLE = "plausible"
    CONTRADICTED = "contradicted"
    NOT_ASSESSED = "not_assessed"


class DecisionStatus(str, Enum):
    SELECTABLE = "selectable"
    NOT_SUPPORTED_BY_DATA = "not_supported_by_data"
    NOT_ASSESSED = "not_assessed"
    WEAKLY_SUPPORTED = "weakly_supported"
    INDISTINGUISHABLE_FROM_OTHER_HYPOTHESES = "indistinguishable_from_other_hypotheses"
    NON_EXCLUSIVE_NOT_SEPARABLE = "non_exclusive_not_separable"
    CONTRADICTED = "contradicted"


class InferenceMode(str, Enum):
    CONDITIONAL_INFERENCE = "conditional_inference"
    WEAK_INFERENCE = "weak_inference"
    HARD_ABSTENTION = "hard_abstention"


class PairwiseRelationType(str, Enum):
    NON_EXCLUSIVE_NOT_SEPARABLE = "non_exclusive_not_separable"
    DISTINGUISHABLE_WITH_CURRENT_DATA = "distinguishable_with_current_data"
    INDISTINGUISHABLE_WITH_CURRENT_DATA = "indistinguishable_with_current_data"
    NOT_ASSESSED = "not_assessed"


class SignalDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    UNCHANGED = "unchanged"
    AMBIGUOUS = "ambiguous"
    NOT_MEASURED = "not_measured"


class ProgramStatus(str, Enum):
    UP = "up"
    DOWN = "down"
    UNCHANGED = "unchanged"
    INDETERMINATE = "indeterminate"
    NOT_MEASURED = "not_measured"


class MeasurementLevel(str, Enum):
    MRNA = "mRNA"
    TOTAL_PROTEIN = "total_protein"
    SECRETED_PROTEIN = "secreted_protein"
    ACTIVITY = "activity"
    NOT_SPECIFIED = "not_specified"


class DataType(str, Enum):
    BULK_RNA_SEQ = "bulk_rna_seq"
    BULK_TISSUE_RNA_SEQ = "bulk_tissue_rna_seq"
    PURIFIED_MACROPHAGE_CULTURE = "purified_macrophage_culture"
    PURIFIED_CELL_CULTURE = "purified_cell_culture"
    SINGLE_CELL_RNA_SEQ = "single_cell_rna_seq"
    OTHER = "other"


class ThresholdSpec(BaseModel):
    min_abs_log2fc_up: float = Field(default=0.58, ge=0.0)
    max_abs_log2fc_unchanged: float = Field(default=0.25, ge=0.0)
    max_fdr: float = Field(default=0.05, gt=0.0, le=1.0)


class Observation(BaseModel):
    observation_id: str
    entity: str
    assay: str
    level: MeasurementLevel = MeasurementLevel.NOT_SPECIFIED
    direction: SignalDirection | None = None
    value: float | None = None
    unit: str | None = None
    fdr: float | None = Field(default=None, ge=0.0, le=1.0)
    timepoint: str | None = None
    source: str = "user_upload"
    threshold: ThresholdSpec = Field(default_factory=ThresholdSpec)

    def resolve_direction(self) -> SignalDirection:
        if self.direction is not None:
            return self.direction
        if self.value is None:
            return SignalDirection.NOT_MEASURED
        if self.unit == "log2FC":
            if self.fdr is None or self.fdr > self.threshold.max_fdr:
                return SignalDirection.AMBIGUOUS
            if self.value >= self.threshold.min_abs_log2fc_up:
                return SignalDirection.UP
            if self.value <= -self.threshold.min_abs_log2fc_up:
                return SignalDirection.DOWN
            if abs(self.value) <= self.threshold.max_abs_log2fc_unchanged:
                return SignalDirection.UNCHANGED
            return SignalDirection.AMBIGUOUS
        return SignalDirection.NOT_MEASURED


class ConditionContext(BaseModel):
    oxygen: str | None = None
    oxygen_level: str | None = None
    duration: str | None = None
    stimulus: str | None = None
    priming: str | None = None
    co_stimulus: str | None = None


class StudyDesignContext(BaseModel):
    design: str | None = None
    control: str | None = None
    replicates: int | None = Field(default=None, ge=0)
    batch: str | None = None


class BiologicalContext(BaseModel):
    organism: str | None = None
    cell_type: str | None = None
    cell_subtype: str | None = None
    tissue: str | None = None
    compartment: str | None = None
    timepoint: str | None = None
    sample_type: str | None = None
    condition: ConditionContext = Field(default_factory=ConditionContext)
    study_design: StudyDesignContext = Field(default_factory=StudyDesignContext)


class TargetSpec(BaseModel):
    entity: str
    level: MeasurementLevel = MeasurementLevel.NOT_SPECIFIED


class ProgramPanel(BaseModel):
    program_id: str
    core_markers: list[str] = Field(default_factory=list)
    supporting_markers: list[str] = Field(default_factory=list)
    min_core_up: int = Field(default=1, ge=0)
    min_total_up: int = Field(default=1, ge=0)
    min_measured_for_unchanged: int = Field(default=2, ge=1)


class EvidenceRecord(BaseModel):
    evidence_id: str
    source_id: str
    claim_summary: str
    evidence_type: str = "literature"
    assertion_method: str = "curated_extraction"
    context_summary: str | None = None
    supports_hypothesis_ids: list[str] = Field(default_factory=list)
    contradicts_hypothesis_ids: list[str] = Field(default_factory=list)
    curated: bool = True


class ModelPriorKnowledgeNote(BaseModel):
    note_id: str
    description: str
    status: Literal["model_prior_knowledge"] = "model_prior_knowledge"
    citable_as_evidence: Literal[False] = False
    requires_curator_review: Literal[True] = True


def _default_nfkb_activity_panel() -> ProgramPanel:
    return ProgramPanel(
        program_id="NFKB_ACTIVITY",
        core_markers=["RELA_NUCLEAR", "P65_PHOSPHO", "IKBA_DEGRADATION", "NFKB_DNA_BINDING"],
        supporting_markers=[],
        min_core_up=1,
        min_total_up=1,
        min_measured_for_unchanged=1,
    ).model_copy(deep=True)


def _default_nfkb_target_panel() -> ProgramPanel:
    return ProgramPanel(
        program_id="NFKB_TARGET_PROGRAM",
        core_markers=["NFKBIA", "TNFAIP3"],
        supporting_markers=["CXCL8", "CCL2", "CXCL10"],
        min_core_up=1,
        min_total_up=2,
        min_measured_for_unchanged=2,
    ).model_copy(deep=True)


def _default_hif_panel() -> ProgramPanel:
    return ProgramPanel(
        program_id="HIF_PROGRAM",
        core_markers=["VEGFA", "SLC2A1", "LDHA", "PDK1", "CA9"],
        supporting_markers=[],
        min_core_up=2,
        min_total_up=2,
        min_measured_for_unchanged=2,
    ).model_copy(deep=True)


def _default_orthogonal_state_panel() -> ProgramPanel:
    return ProgramPanel(
        program_id="ORTHOGONAL_MACROPHAGE_STATE_PROGRAM",
        core_markers=["MRC1", "CD163", "APOE", "TREM2", "FOLR2"],
        supporting_markers=["MERTK", "ITGAX", "SELENOP"],
        min_core_up=2,
        min_total_up=2,
        min_measured_for_unchanged=2,
    ).model_copy(deep=True)


class HypothesisEngineInput(BaseModel):
    query_id: str
    scenario: str = "IL6_hypoxia_macrophage"
    data_type: DataType | None = None
    cell_composition_changed: bool | None = None
    timepoint_known: bool | None = None
    target: TargetSpec
    context: BiologicalContext = Field(default_factory=BiologicalContext)
    observations: list[Observation] = Field(default_factory=list)
    evidence_records: list[EvidenceRecord] = Field(default_factory=list)
    model_prior_notes: list[ModelPriorKnowledgeNote] = Field(default_factory=list)
    nfkb_activity_panel: ProgramPanel = Field(default_factory=_default_nfkb_activity_panel)
    nfkb_target_panel: ProgramPanel = Field(default_factory=_default_nfkb_target_panel)
    hif_panel: ProgramPanel = Field(default_factory=_default_hif_panel)
    orthogonal_state_panel: ProgramPanel = Field(default_factory=_default_orthogonal_state_panel)

    @model_validator(mode="after")
    def validate_panels_and_target_circularity(self) -> HypothesisEngineInput:
        panels = [
            self.nfkb_activity_panel,
            self.nfkb_target_panel,
            self.hif_panel,
            self.orthogonal_state_panel,
        ]
        for panel in panels:
            all_markers = panel.core_markers + panel.supporting_markers
            if not all_markers:
                raise ValueError(f"Panel '{panel.program_id}' must define at least one marker.")
            if self.target.entity and self.target.entity.strip().upper() in {
                m.strip().upper() for m in all_markers
            }:
                raise ValueError(
                    f"Target entity '{self.target.entity}' cannot appear in panel '{panel.program_id}'."
                )

        forbidden = (
            {m.upper() for m in self.hif_panel.core_markers + self.hif_panel.supporting_markers}
            | {
                m.upper()
                for m in self.nfkb_target_panel.core_markers + self.nfkb_target_panel.supporting_markers
            }
            | {m.upper() for m in self.nfkb_activity_panel.core_markers + self.nfkb_activity_panel.supporting_markers}
        )
        orth_markers = {
            m.upper()
            for m in self.orthogonal_state_panel.core_markers + self.orthogonal_state_panel.supporting_markers
        }
        overlap = forbidden & orth_markers
        if overlap:
            raise ValueError(
                f"Orthogonal state program overlaps with H1/H2 markers: {sorted(overlap)}"
            )
        return self


class HypothesisEvaluation(BaseModel):
    hypothesis_id: str
    name: str
    evidence_status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    evidence_conflict: bool = False
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    data_status: DataStatus = DataStatus.NOT_ASSESSED
    decision_status: DecisionStatus = DecisionStatus.NOT_ASSESSED
    rule_conflict: bool = False
    applied_rules: list[str] = Field(default_factory=list)
    resolution: str | None = None
    rationale: str = ""


class RuleConflict(BaseModel):
    rule_ids: list[str]
    affected_hypothesis_ids: list[str]
    selected_status: DataStatus
    discarded_status: DataStatus
    reason: str


class PairwiseRelation(BaseModel):
    hypotheses: tuple[str, str]
    relation: PairwiseRelationType
    reason: str
    discriminating_measurement_id: str | None = None


class DiscriminatingMeasurement(BaseModel):
    measurement_id: str
    priority: int = Field(ge=1)
    measurement: str
    recommended_assays: list[str]
    discriminates: list[str]
    rationale: str


class HypothesisEngineOutput(BaseModel):
    query_id: str
    scenario: str
    rule_set_version: str = "IL6_hypoxia_macrophage_v0.1"
    mode: InferenceMode
    abstention_reason: str | None = None
    missing_mandatory_fields: list[str] = Field(default_factory=list)
    missing_context_fields: list[str] = Field(default_factory=list)
    context_conflicts: list[str] = Field(default_factory=list)
    program_statuses: dict[str, ProgramStatus] = Field(default_factory=dict)
    hypotheses: dict[str, HypothesisEvaluation] = Field(default_factory=dict)
    rule_conflicts: list[RuleConflict] = Field(default_factory=list)
    pairwise_relations: list[PairwiseRelation] = Field(default_factory=list)
    discriminating_measurements: list[DiscriminatingMeasurement] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    model_prior_notes: list[ModelPriorKnowledgeNote] = Field(default_factory=list)


def parse_oxygen_percent(oxygen_level: str | None) -> float | None:
    if not oxygen_level:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", oxygen_level)
    return float(match.group(1)) if match else None
