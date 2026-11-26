from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MeasurementLevel(str, Enum):
    MRNA = "mRNA"
    TOTAL_PROTEIN = "total_protein"
    SECRETED_PROTEIN = "secreted_protein"
    PATHWAY_ACTIVITY = "pathway_activity"
    CELL_PROPORTION = "cell_proportion"


class BranchApplicability(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    PLAUSIBLE = "plausible"
    CONTRADICTED = "contradicted"
    NOT_ASSESSED = "not_assessed"


class DataStatus(str, Enum):
    SUPPORTED = "supported"
    COMPATIBLE = "compatible"
    NOT_SUPPORTED_BY_DATA = "not_supported_by_data"
    CONTRADICTED = "contradicted"
    NOT_ASSESSED = "not_assessed"


class DecisionStatus(str, Enum):
    SELECTABLE = "selectable"
    WEAKLY_SUPPORTED = "weakly_supported"
    NOT_SUPPORTED_BY_DATA = "not_supported_by_data"
    NOT_ASSESSED = "not_assessed"
    INDISTINGUISHABLE_FROM_OTHER_BRANCHES = "indistinguishable_from_other_branches"


class ExpectedDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    UNCHANGED = "unchanged"


class ContextOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    MISSING = "missing"
    PRESENT = "present"


class ConflictType(str, Enum):
    CONTEXT_CONFLICT = "context_conflict"
    MEASUREMENT_LEVEL_CONFLICT = "measurement_level_conflict"
    DIRECTION_CONFLICT = "direction_conflict"
    COMPOSITION_CONFLICT = "composition_conflict"
    COMPETING_BRANCH_CONFLICT = "competing_branch_conflict"
    EVIDENCE_CONFLICT = "evidence_conflict"
    RULE_CONFLICT = "rule_conflict"


class ConflictDecision(str, Enum):
    HARD_ABSTENTION = "hard_abstention"
    PRESERVE_CONFLICT = "preserve_conflict"
    WEAK_INFERENCE = "weak_inference"
    CONDITIONAL_INFERENCE = "conditional_inference"


class InferenceMode(str, Enum):
    HARD_ABSTENTION = "hard_abstention"
    WEAK_INFERENCE = "weak_inference"
    CONDITIONAL_INFERENCE = "conditional_inference"


class PairwiseRelationType(str, Enum):
    NON_EXCLUSIVE_NOT_SEPARABLE = "non_exclusive_not_separable"
    INDISTINGUISHABLE_WITH_CURRENT_DATA = "indistinguishable_with_current_data"
    DISTINGUISHABLE_WITH_CURRENT_DATA = "distinguishable_with_current_data"
    NOT_ASSESSED = "not_assessed"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextRule(StrictModel):
    field: str = Field(..., min_length=1)
    operator: ContextOperator
    value: Optional[Any] = None

    @model_validator(mode="after")
    def validate_value_presence(self) -> "ContextRule":
        if self.operator in (ContextOperator.MISSING, ContextOperator.PRESENT):
            if self.value is not None:
                raise ValueError(f"Operator '{self.operator.value}' must not specify a value.")
        elif self.value is None:
            raise ValueError(f"Operator '{self.operator.value}' requires a value.")
        return self


class MeasurementRequirement(StrictModel):
    entity: str = Field(..., min_length=1)
    level: MeasurementLevel
    requirement: str = Field(..., pattern="^(required|optional|forbidden)$")


class DirectionExpectation(StrictModel):
    entity: str = Field(..., min_length=1)
    expected_direction: ExpectedDirection
    interpretation: str = Field(..., min_length=1)


class EvidenceRequirement(StrictModel):
    evidence_type: str = Field(..., min_length=1)
    required: bool
    minimum_replication: int = Field(default=1, ge=0)
    translational_status: str = Field(default="any", min_length=1)


class StatusOutcome(StrictModel):
    branch_applicability: Optional[BranchApplicability] = None
    evidence_status: Optional[EvidenceStatus] = None
    data_status: Optional[DataStatus] = None
    decision_status: Optional[DecisionStatus] = None

    @model_validator(mode="after")
    def validate_axis_consistency(self) -> "StatusOutcome":
        if (
            self.branch_applicability == BranchApplicability.NOT_APPLICABLE
            and self.data_status in (DataStatus.SUPPORTED, DataStatus.COMPATIBLE)
        ):
            raise ValueError(
                "Inapplicable branch cannot have 'supported' or 'compatible' data_status."
            )
        return self


class OutputRule(StrictModel):
    rule_id: str = Field(..., min_length=1)
    condition_expr: str = Field(..., min_length=1)
    then: StatusOutcome
    rationale: str = Field(..., min_length=1)


class BranchDefinition(StrictModel):
    branch_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    biological_claim: str = Field(..., min_length=1)
    applies_when: List[ContextRule] = Field(default_factory=list)
    measurement_requirements: List[MeasurementRequirement] = Field(default_factory=list)
    direction_expectations: List[DirectionExpectation] = Field(default_factory=list)
    evidence_requirements: List[EvidenceRequirement] = Field(default_factory=list)
    output_rules: List[OutputRule] = Field(default_factory=list)


class ConflictRule(StrictModel):
    conflict_id: str = Field(..., min_length=1)
    conflict_type: ConflictType
    trigger: List[ContextRule] = Field(..., min_length=1)
    affected_branches: List[str] = Field(default_factory=list)
    decision: ConflictDecision
    resolution_required: List[str] = Field(default_factory=list)


class AbstentionRule(StrictModel):
    rule_id: str = Field(..., min_length=1)
    condition: List[ContextRule] = Field(..., min_length=1)
    mode: InferenceMode
    reason: str = Field(..., min_length=1)
    required_measurements: List[str] = Field(default_factory=list)


class PairwiseRelation(StrictModel):
    branch_ids: List[str] = Field(..., min_length=2, max_length=2)
    relation: PairwiseRelationType
    reason: str = Field(..., min_length=1)
    discriminating_measurement: Optional[str] = None


class DiscriminatingMeasurement(StrictModel):
    measurement_id: str = Field(..., min_length=1)
    priority: int = Field(..., ge=1)
    measurement: str = Field(..., min_length=1)
    method: str = Field(..., min_length=1)
    discriminates: List[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=1)
    data_type_condition: str = Field(default="any", min_length=1)


class ProcessModuleContract(StrictModel):
    module_id: str = Field(..., min_length=1)
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    rule_set_id: str = Field(..., min_length=1)
    rule_set_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    description: str = Field(..., min_length=1)
    scenario: str = Field(..., min_length=1)

    required_context: List[str] = Field(default_factory=list)
    recommended_context: List[str] = Field(default_factory=list)
    allowed_measurement_levels: List[MeasurementLevel] = Field(default_factory=list)

    branches: List[BranchDefinition] = Field(default_factory=list)
    conflict_rules: List[ConflictRule] = Field(default_factory=list)
    abstention_rules: List[AbstentionRule] = Field(default_factory=list)
    pairwise_relations: List[PairwiseRelation] = Field(default_factory=list)
    discriminating_measurements: List[DiscriminatingMeasurement] = Field(default_factory=list)
    regression_cases: List[str] = Field(default_factory=list)
