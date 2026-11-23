from typing import Literal
from pydantic import BaseModel, Field


class AssertionContext(BaseModel):
    organism: str
    cell_type: str
    tissue: str
    pathology: str
    physiology: str
    stimulus: str
    timepoint: str
    method: str
    measurement_level: str | None = None
    phase: str | None = None
    dose_level: str | None = None
    duration_pattern: str | None = None
    cell_circuit: str | None = None


class Assertion(BaseModel):
    id: str
    subject: str
    predicate: str
    relation_label: str
    object: str
    direction: Literal["up", "down", "bidirectional", "unknown"]
    effect_size: str
    confidence: str
    status: str
    version: int
    eco: Literal["A", "B", "C", "D", "E", "F", "G"]
    conflict: bool
    context: AssertionContext
    source: str
    notes: str


class Entity(BaseModel):
    id: str
    bioos_id: str
    name: str
    full_name: str
    category: str
    sub_type: str
    organism: str
    description: str
    functions: list[str]
    pathways: list[str]
    cells: list[str]
    activators: list[str]
    inhibitors: list[str]
    contexts: list[str]
    pathologies: list[str]
    evidence_summary: str


class Module(BaseModel):
    id: str
    name: str
    module_type: str
    inputs: list[str]
    core_nodes: list[str]
    outputs: list[str]
    states: list[str]
    limitations: str


class BiologicalState(BaseModel):
    id: str
    name: str
    rule: str
    causes: list[str]
    drivers: list[str]
    biomarkers: list[str]
    anchor_markers: list[str] = Field(default_factory=list)
    counter_markers: list[str] = Field(default_factory=list)
    activated_pathways: list[str]
    suppressed_pathways: list[str]
    consequences: list[str]


class ExpressionMeasurement(BaseModel):
    symbol: str
    direction: Literal["up", "down", "unchanged"] = "up"
    log2fc: float | None = None
    padj: float | None = None


class DatasetUploadRequest(BaseModel):
    dataset_name: str = Field(default="uploaded_expression_set")
    modality: Literal["RNA-seq", "Proteomics", "Cytokines", "Metabolomics"] = "RNA-seq"
    measurement_level: str | None = None
    organism: str = "Homo sapiens"
    tissue: str | None = None
    cell_type: str | None = None
    measurements: list[ExpressionMeasurement]


class EvaluatedStateScore(BaseModel):
    state_id: str
    name: str
    matched_drivers: list[str]
    matched_biomarkers: list[str]
    matched_anchors: list[str] = Field(default_factory=list)
    matched_counter_markers: list[str] = Field(default_factory=list)
    counter_penalty: float = 0.0
    coverage_ratio: float
    diagnostic_disclaimer: str = "Not a clinical diagnosis"


class EvaluatedModuleScore(BaseModel):
    module_id: str
    name: str
    matched_core_nodes: list[str]
    matched_inputs: list[str]
    coverage_ratio: float
    limitations: str


class SnapshotRequest(BaseModel):
    title: str = Field(default="BIOOS Analytical Snapshot")
    signals: str = Field(default="TNF up, IL6 up, IFNG up, CXCL10 up")
    start_node: str = Field(default="TNF")
    max_hops: int = Field(default=2, ge=1, le=5)
    actor: str = Field(default="BIOOS-Analyst-Pipeline")


class ProvProvenance(BaseModel):
    context: dict[str, str]
    activity_id: str
    generated_at: str
    was_associated_with: dict[str, str]
    used_entities: list[dict[str, str]]
    generated_entity: dict[str, str]


class ReportSnapshot(BaseModel):
    snapshot_id: str
    title: str
    created_at: str
    canonical_payload_sha256: str
    knowledge_base_sha256: str
    analytical_scope_note: str
    summary: dict[str, object]
    prov_o: ProvProvenance
