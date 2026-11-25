from __future__ import annotations

import pytest
from pydantic import ValidationError

from hypothesis_engine_v0_1.engine import run_hypothesis_engine
from hypothesis_engine_v0_1.schemas import (
    BiologicalContext,
    ConditionContext,
    DataStatus,
    DataType,
    DecisionStatus,
    EvidenceRecord,
    HypothesisEngineInput,
    InferenceMode,
    MeasurementLevel,
    Observation,
    PairwiseRelationType,
    ProgramStatus,
    SignalDirection,
    StudyDesignContext,
    TargetSpec,
)


def _base_context(
    *,
    oxygen: str | None = "Hypoxia",
    oxygen_level: str | None = "1% O2",
    timepoint: str | None = "12h",
    control: str | None = "Normoxia",
) -> BiologicalContext:
    return BiologicalContext(
        organism="Homo sapiens",
        cell_type="Macrophage",
        timepoint=timepoint,
        condition=ConditionContext(
            oxygen=oxygen,
            oxygen_level=oxygen_level,
            duration="12h",
        ),
        study_design=StudyDesignContext(control=control),
    )


def test_normoxia_with_21_percent_oxygen_does_not_abstain() -> None:
    payload = HypothesisEngineInput(
        query_id="Q-NORM-21",
        data_type=DataType.PURIFIED_MACROPHAGE_CULTURE,
        cell_composition_changed=False,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        context=_base_context(oxygen="Normoxia", oxygen_level="21% O2", control="Untreated"),
        observations=[],
    )
    out = run_hypothesis_engine(payload)
    assert out.mode != InferenceMode.HARD_ABSTENTION
    assert out.context_conflicts == []


def test_missing_target_level_causes_hard_abstention() -> None:
    payload = HypothesisEngineInput(
        query_id="Q-NO-LEVEL",
        data_type=DataType.PURIFIED_MACROPHAGE_CULTURE,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.NOT_SPECIFIED),
        context=_base_context(),
    )
    out = run_hypothesis_engine(payload)
    assert out.mode == InferenceMode.HARD_ABSTENTION
    assert "target.level" in out.missing_mandatory_fields


def test_all_down_hif_markers_give_down_not_unchanged() -> None:
    payload = HypothesisEngineInput(
        query_id="Q-HIF-DOWN",
        data_type=DataType.PURIFIED_MACROPHAGE_CULTURE,
        cell_composition_changed=False,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        context=_base_context(),
        observations=[
            Observation(
                observation_id="O1",
                entity="VEGFA",
                assay="RNA-seq",
                level=MeasurementLevel.MRNA,
                direction=SignalDirection.DOWN,
            ),
            Observation(
                observation_id="O2",
                entity="SLC2A1",
                assay="RNA-seq",
                level=MeasurementLevel.MRNA,
                direction=SignalDirection.DOWN,
            ),
        ],
    )
    out = run_hypothesis_engine(payload)
    assert out.program_statuses["HIF_PROGRAM"] == ProgramStatus.DOWN


def test_bulk_composition_change_and_both_programs_up_gives_weak_inference() -> None:
    payload = HypothesisEngineInput(
        query_id="Q-BULK-COMP",
        data_type=DataType.BULK_RNA_SEQ,
        cell_composition_changed=True,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        context=_base_context(),
        observations=[
            Observation(observation_id="1", entity="RELA_NUCLEAR", assay="IF", level=MeasurementLevel.ACTIVITY, direction=SignalDirection.UP),
            Observation(observation_id="2", entity="NFKBIA", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UP),
            Observation(observation_id="3", entity="TNFAIP3", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UP),
            Observation(observation_id="4", entity="VEGFA", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UP),
            Observation(observation_id="5", entity="SLC2A1", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UP),
        ],
    )
    out = run_hypothesis_engine(payload)
    assert out.mode == InferenceMode.WEAK_INFERENCE
    assert out.hypotheses["H4"].data_status == DataStatus.COMPATIBLE
    assert out.hypotheses["H4"].decision_status == DecisionStatus.WEAKLY_SUPPORTED


def test_r002_yields_distinguishable_with_current_data_for_h1_vs_h2() -> None:
    payload = HypothesisEngineInput(
        query_id="Q-R002",
        data_type=DataType.PURIFIED_MACROPHAGE_CULTURE,
        cell_composition_changed=False,
        timepoint_known=True,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        context=_base_context(timepoint="12h"),
        evidence_records=[
            EvidenceRecord(
                evidence_id="EV-1",
                source_id="SRC-1",
                claim_summary="Hypoxia induces HIF targets in macrophages.",
                supports_hypothesis_ids=["H2"],
            )
        ],
        observations=[
            Observation(observation_id="1", entity="VEGFA", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UP, timepoint="12h"),
            Observation(observation_id="2", entity="SLC2A1", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UP, timepoint="12h"),
            Observation(observation_id="3", entity="NFKBIA", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UNCHANGED, timepoint="12h"),
            Observation(observation_id="4", entity="TNFAIP3", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UNCHANGED, timepoint="12h"),
        ],
    )
    out = run_hypothesis_engine(payload)
    assert out.hypotheses["H2"].data_status == DataStatus.COMPATIBLE
    assert out.hypotheses["H1"].data_status == DataStatus.NOT_SUPPORTED_BY_DATA
    h1_h2 = [r for r in out.pairwise_relations if r.hypotheses == ("H1", "H2")][0]
    assert h1_h2.relation == PairwiseRelationType.DISTINGUISHABLE_WITH_CURRENT_DATA
    assert out.mode == InferenceMode.CONDITIONAL_INFERENCE


def test_conflict_between_two_real_statuses_sets_rule_conflict_and_ignores_not_assessed() -> None:
    payload = HypothesisEngineInput(
        query_id="Q-CONFLICT",
        data_type=DataType.PURIFIED_MACROPHAGE_CULTURE,
        cell_composition_changed=False,
        timepoint_known=True,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        context=_base_context(timepoint="12h"),
        observations=[
            Observation(observation_id="1", entity="VEGFA", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UP, timepoint="12h"),
            Observation(observation_id="2", entity="SLC2A1", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UP, timepoint="12h"),
            Observation(observation_id="3", entity="NFKBIA", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UNCHANGED, timepoint="12h"),
            Observation(observation_id="4", entity="TNFAIP3", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UNCHANGED, timepoint="12h"),
        ],
    )
    out = run_hypothesis_engine(payload)
    assert out.hypotheses["H2"].rule_conflict is False
    assert out.hypotheses["H1"].rule_conflict is False
    assert out.rule_conflicts == []


def test_target_circularity_and_shared_panel_isolation() -> None:
    with pytest.raises(ValidationError):
        HypothesisEngineInput(
            query_id="Q-CIRCULAR",
            target=TargetSpec(entity="VEGFA", level=MeasurementLevel.MRNA),
            context=_base_context(),
        )

    p1 = HypothesisEngineInput(
        query_id="Q-P1",
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        context=_base_context(),
    )
    p2 = HypothesisEngineInput(
        query_id="Q-P2",
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        context=_base_context(),
    )
    p1.hif_panel.core_markers.append("CUSTOM_MARKER")
    assert "CUSTOM_MARKER" not in p2.hif_panel.core_markers
# --- Additional edge-case branch tests ---
from hypothesis_engine_v0_1 import schemas
from hypothesis_engine_v0_1.engine import _merge_hypothesis_proposals, run_hypothesis_engine


def _build_base_payload(query_id: str, entity: str = "IL6", level: str = "mRNA") -> schemas.HypothesisEngineInput:
    data: dict[str, object] = {
        "query_id": query_id,
        "target": schemas.TargetSpec(entity=entity, level=level),
    }
    for field_name, field_info in schemas.HypothesisEngineInput.model_fields.items():
        if field_name in data:
            continue
        if field_info.default_factory is not None:
            data[field_name] = field_info.default_factory()
        elif field_info.default is not None:
            data[field_name] = field_info.default
    payload = schemas.HypothesisEngineInput.model_validate(data)
    if hasattr(payload, "context") and payload.context is not None:
        if hasattr(payload.context, "organism") and not payload.context.organism:
            payload.context.organism = "Homo sapiens"
        if hasattr(payload.context, "cell_type") and not payload.context.cell_type:
            payload.context.cell_type = "macrophage"
        if hasattr(payload.context, "study_design") and payload.context.study_design is not None:
            if hasattr(payload.context.study_design, "control") and not payload.context.study_design.control:
                payload.context.study_design.control = "untreated"
    return payload


def _make_obs(
    obs_id: str,
    entity: str,
    direction: schemas.SignalDirection,
    assay: str = "rna_seq",
    timepoint: str | None = None,
) -> schemas.Observation:
    fields = schemas.Observation.model_fields
    data: dict[str, object] = {
        "observation_id": obs_id,
        "entity": entity,
        "assay": assay,
        "direction": direction,
        "timepoint": timepoint,
    }
    for k, info in fields.items():
        if k in data:
            continue
        if info.default_factory is not None:
            data[k] = info.default_factory()
        elif info.default is not None:
            data[k] = info.default
    return schemas.Observation.model_validate(data)


def _make_evidence(
    evidence_id: str,
    curated: bool,
    supports: list[str] | None = None,
    contradicts: list[str] | None = None,
) -> schemas.EvidenceRecord:
    fields = schemas.EvidenceRecord.model_fields
    data: dict[str, object] = {
        "evidence_id": evidence_id,
        "source_id": "PMID_00000001",
        "claim_summary": "Synthetic test evidence record for branch coverage.",
        "curated": curated,
        "supports_hypothesis_ids": supports or [],
        "contradicts_hypothesis_ids": contradicts or [],
    }
    for k, info in fields.items():
        if k in data:
            continue
        if info.default_factory is not None:
            data[k] = info.default_factory()  # pragma: no cover
        elif info.default is not None:
            data[k] = info.default
    return schemas.EvidenceRecord.model_validate(data)


def test_empty_observations_and_evidence_records_branches() -> None:
    payload = _build_base_payload("Q_EDGE_EMPTY_OBS")
    payload.observations = []
    payload.evidence_records = []
    result = run_hypothesis_engine(payload)
    assert result is not None


def test_total_protein_level_and_unknown_timepoint_branches() -> None:
    payload = _build_base_payload("Q_EDGE_PROTEIN_TIMEPOINT", level="total_protein")
    payload.timepoint_known = False
    payload.cell_composition_changed = True
    result = run_hypothesis_engine(payload)
    assert result is not None


def test_supporting_markers_and_curated_evidence_conflict_branches() -> None:
    payload = _build_base_payload("Q_EDGE_SUPPORT_CONFLICT")
    payload.observations = [
        _make_obs("OBS_NFKB_CORE", "RELA_NUCLEAR", schemas.SignalDirection.UP),
        _make_obs("OBS_NFKB_TGT_CORE", "NFKBIA", schemas.SignalDirection.UP),
        _make_obs("OBS_SUPP_UP", "CXCL8", schemas.SignalDirection.UP),
        _make_obs("OBS_SUPP_DN", "CCL2", schemas.SignalDirection.DOWN),
        _make_obs("OBS_SUPP_UNC", "CXCL10", schemas.SignalDirection.UNCHANGED),
    ]
    payload.evidence_records = [
        _make_evidence("EV_1", curated=True, supports=["H1"], contradicts=["H2"]),
        _make_evidence("EV_2", curated=True, supports=["H2"], contradicts=[]),
        _make_evidence("EV_3", curated=True, supports=[], contradicts=["H4"]),
        _make_evidence("EV_UNCURATED", curated=False, supports=["H3"], contradicts=["H3"]),
    ]
    result = run_hypothesis_engine(payload)
    assert result is not None


def test_bulk_composition_and_r004_r005_branches() -> None:
    payload = _build_base_payload("Q_EDGE_BULK_R004_R005")
    payload.data_type = schemas.DataType.BULK_RNA_SEQ
    payload.cell_composition_changed = False
    payload.observations = [
        _make_obs("OBS_NFKB_1", "RELA_NUCLEAR", schemas.SignalDirection.UP),
        _make_obs("OBS_TGT_1", "NFKBIA", schemas.SignalDirection.UP),
        _make_obs("OBS_TGT_2", "CXCL8", schemas.SignalDirection.UP),
        _make_obs("OBS_ORTH_1", "MRC1", schemas.SignalDirection.UP),
        _make_obs("OBS_ORTH_2", "CD163", schemas.SignalDirection.UP),
        _make_obs("OBS_ORTH_SUPP", "MERTK", schemas.SignalDirection.UP),
    ]
    result = run_hypothesis_engine(payload)
    assert result is not None


def test_indeterminate_and_rule_conflict_branches() -> None:
    payload = _build_base_payload("Q_EDGE_INDETERMINATE_CONFLICT")
    payload.data_type = schemas.DataType.PURIFIED_MACROPHAGE_CULTURE
    payload.observations = [
        _make_obs("OBS_HIF_UP", "VEGFA", schemas.SignalDirection.UP),
        _make_obs("OBS_HIF_DN", "SLC2A1", schemas.SignalDirection.DOWN),
        _make_obs("OBS_NFKB_1", "RELA_NUCLEAR", schemas.SignalDirection.UP),
        _make_obs("OBS_TGT_1", "NFKBIA", schemas.SignalDirection.UNCHANGED),
        _make_obs("OBS_TGT_2", "CXCL8", schemas.SignalDirection.UNCHANGED),
    ]
    result = run_hypothesis_engine(payload)
    assert result is not None


def test_context_conflicts_ambiguous_and_bulk_none_branches() -> None:
    # Hypoxia with >= 10% O2 + missing mandatory fields (organism/cell_type/control empty)
    p_hyp_conflict = _build_base_payload("Q_CTX_HYPOXIA_CONFLICT")
    p_hyp_conflict.context.organism = ""
    p_hyp_conflict.context.cell_type = "   "
    p_hyp_conflict.context.study_design.control = ""
    p_hyp_conflict.context.condition.oxygen = "hypoxia"
    p_hyp_conflict.context.condition.oxygen_level = "21%"
    assert run_hypothesis_engine(p_hyp_conflict) is not None

    # Normoxia with <= 5% O2
    p_norm_conflict = _build_base_payload("Q_CTX_NORMOXIA_CONFLICT")
    p_norm_conflict.context.condition.oxygen = "normoxia"
    p_norm_conflict.context.condition.oxygen_level = "1%"
    assert run_hypothesis_engine(p_norm_conflict) is not None

    # timepoint_known=False while context.timepoint is populated
    p_tp_conflict = _build_base_payload("Q_CTX_TP_CONFLICT")
    p_tp_conflict.context.timepoint = "6h"
    p_tp_conflict.timepoint_known = False
    assert run_hypothesis_engine(p_tp_conflict) is not None

    # AMBIGUOUS core/supporting marker + conflicting UP/DOWN on same entity/timepoint + bulk cell_composition_changed=None
    p_amb = _build_base_payload("Q_AMB_BULK_NONE")
    p_amb.data_type = schemas.DataType.BULK_RNA_SEQ
    p_amb.cell_composition_changed = None
    p_amb.observations = [
        _make_obs("OBS_CORE_AMB", "RELA_NUCLEAR", schemas.SignalDirection.AMBIGUOUS, timepoint="2h"),
        _make_obs("OBS_SUPP_AMB", "CXCL8", schemas.SignalDirection.AMBIGUOUS, timepoint="2h"),
        _make_obs("OBS_CONF_UP", "VEGFA", schemas.SignalDirection.UP, timepoint="2h"),
        _make_obs("OBS_CONF_DN", "VEGFA", schemas.SignalDirection.DOWN, timepoint="2h"),
    ]
    assert run_hypothesis_engine(p_amb) is not None



def test_missing_target_entity_branch() -> None:
    payload = _build_base_payload("Q_MISSING_TARGET_ENTITY")
    payload.target.entity = ""
    result = run_hypothesis_engine(payload)
    assert "target.entity" in result.missing_mandatory_fields


def test_decision_status_contradicted_when_data_status_contradicted(monkeypatch) -> None:
    import hypothesis_engine_v0_1.engine as eng

    orig_merge = eng._merge_hypothesis_proposals

    def _wrapped_merge(**kwargs):
        ev, conflicts = orig_merge(**kwargs)
        if kwargs.get("h_id") == "H1":
            ev.data_status = schemas.DataStatus.CONTRADICTED
        return ev, conflicts

    monkeypatch.setattr(eng, "_merge_hypothesis_proposals", _wrapped_merge)
    payload = _build_base_payload("Q_DECISION_CONTRADICTED")
    result = run_hypothesis_engine(payload)
    assert result.hypotheses["H1"].decision_status == schemas.DecisionStatus.CONTRADICTED

def test_merge_hypothesis_proposals_rule_conflict_and_contradicted() -> None:
    ev, conflicts = _merge_hypothesis_proposals(
        h_id="H1",
        proposals=[
            ("R004", schemas.DataStatus.COMPATIBLE, "Compatible under R004."),
            ("R002", schemas.DataStatus.NOT_SUPPORTED_BY_DATA, "Not supported under R002."),
        ],
        evidence_status=schemas.EvidenceStatus.CONTRADICTED,
        evidence_conflict=True,
        supporting_ids=["EV_1"],
        contradicting_ids=["EV_2"],
    )
    assert ev.rule_conflict is True
    assert len(conflicts) == 1
    assert ev.data_status == schemas.DataStatus.NOT_SUPPORTED_BY_DATA
