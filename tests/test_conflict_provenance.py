from __future__ import annotations

from hypothesis_engine_v0_1.conflict_detector import (
    NegativeEvidenceStatus,
    build_provenance_report,
    classify_negative_evidence,
    detect_conflicts,
)
from hypothesis_engine_v0_1.schemas import (
    BiologicalContext,
    ConditionContext,
    DataType,
    EvidenceRecord,
    MeasurementLevel,
    ModelPriorKnowledgeNote,
    Observation,
    SignalDirection,
    StudyDesignContext,
    TargetSpec,
)


def test_detect_conflicts_without_auto_resolution():
    ctx = BiologicalContext(
        organism="Homo sapiens",
        cell_type="macrophage",
        condition=ConditionContext(oxygen="hypoxia", oxygen_level="21% O2"),
        study_design=StudyDesignContext(control="normoxia"),
    )
    target = TargetSpec(entity="IL6", level=MeasurementLevel.MRNA)
    obs = [
        Observation(
            observation_id="obs_1",
            entity="IL-6",
            assay="RNA-seq",
            level=MeasurementLevel.MRNA,
            direction=SignalDirection.UP,
            timepoint="6h",
        ),
        Observation(
            observation_id="obs_2",
            entity="interleukin-6",
            assay="RNA-seq",
            level=MeasurementLevel.MRNA,
            direction=SignalDirection.DOWN,
            timepoint="12h",
        ),
    ]

    conflicts = detect_conflicts(
        context=ctx,
        target=target,
        observations=obs,
        data_type=DataType.BULK_RNA_SEQ,
        cell_composition_changed=True,
    )
    c_types = {c.conflict_type for c in conflicts}
    assert "direction_conflict" in c_types
    assert "timepoint_conflict" in c_types
    assert "context_conflict" in c_types
    assert "rule_conflict" in c_types
    assert all(c.auto_resolved is False for c in conflicts)


def test_negative_evidence_statuses_and_provenance_separation():
    ev_no_eff = EvidenceRecord(
        evidence_id="EV-1",
        source_id="PMID:10001",
        claim_summary="No effect observed on secreted IL6 under hypoxia.",
        contradicts_hypothesis_ids=["H2"],
    )
    assert classify_negative_evidence(ev_no_eff) == NegativeEvidenceStatus.NO_EFFECT_OBSERVED

    ev_retracted = EvidenceRecord(
        evidence_id="EV-2",
        source_id="PMID:10002",
        claim_summary="Study retracted due to assay artifact.",
        curated=False,
    )
    assert classify_negative_evidence(ev_retracted) == NegativeEvidenceStatus.RETRACTED

    prior = ModelPriorKnowledgeNote(
        note_id="PRIOR-1",
        description="HIF1A may influence IL6 expression in inflammatory macrophages.",
    )
    ctx = BiologicalContext(organism="Homo sapiens", cell_type="macrophage")
    target = TargetSpec(entity="IL6", level=MeasurementLevel.MRNA)
    obs = [
        Observation(
            observation_id="OBS-1",
            entity="IL6",
            assay="RNA-seq",
            level=MeasurementLevel.MRNA,
            direction=SignalDirection.UP,
        )
    ]

    report = build_provenance_report(
        context=ctx,
        target=target,
        observations=obs,
        evidence_records=[ev_no_eff, ev_retracted],
        model_prior_notes=[prior],
    )

    by_id = {r.source_id: r for r in report.provenance_records}
    assert by_id["OBS-1"].source_type == "user_observation"
    assert by_id["OBS-1"].citable_as_evidence is True

    assert by_id["EV-1"].source_type == "evidence_record"
    assert by_id["EV-1"].negative_evidence_status == NegativeEvidenceStatus.NO_EFFECT_OBSERVED

    assert by_id["EV-2"].negative_evidence_status == NegativeEvidenceStatus.RETRACTED
    assert by_id["EV-2"].citable_as_evidence is False
    assert by_id["EV-2"].requires_curator_review is True

    assert by_id["PRIOR-1"].source_type == "model_prior_knowledge"
    assert by_id["PRIOR-1"].citable_as_evidence is False
    assert by_id["PRIOR-1"].requires_curator_review is True
    assert len(report.negative_evidence_items) == 2
