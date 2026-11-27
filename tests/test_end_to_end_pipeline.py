from __future__ import annotations

import importlib.util
from pathlib import Path

from hypothesis_engine_v0_1.biological_layers import (
    evaluate_mandatory_context,
    resolve_biological_entity,
    validate_measurement_compatibility,
)
from hypothesis_engine_v0_1.conflict_detector import (
    NegativeEvidenceStatus,
    build_provenance_report,
)
from hypothesis_engine_v0_1.llm_contract import (
    GroundedClaim,
    LLMExplanationPayload,
    validate_llm_explanation,
)
from hypothesis_engine_v0_1.schemas import (
    BiologicalContext,
    ConditionContext,
    DataType,
    EvidenceRecord,
    InferenceMode,
    MeasurementLevel,
    ModelPriorKnowledgeNote,
    Observation,
    SignalDirection,
    StudyDesignContext,
    TargetSpec,
)
from hypothesis_engine_v0_1.scoring_rules import ScoringPolicy, evaluate_weighted_hypothesis


def _load_csv_module():
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "csv_to_observations.py"
    spec = importlib.util.spec_from_file_location("csv_to_observations", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_end_to_end_observational_pipeline_with_llm_guardrail(tmp_path: Path):
    csv_mod = _load_csv_module()
    csv_path = tmp_path / "weighted_obs.csv"
    csv_path.write_text(
        "observation_id,direction,confidence,source_type\n"
        "OBS-1,support,0.75,transcriptomics\n"
        "OBS-2,support,0.65,transcriptomics\n",
        encoding="utf-8",
    )

    weighted_obs = csv_mod.load_weighted_observations_from_csv(csv_path)
    trace = evaluate_weighted_hypothesis(weighted_obs, policy=ScoringPolicy())
    assert trace.passed_threshold is True

    entity = resolve_biological_entity("interleukin-6")
    assert entity.canonical_id == "IL6"

    target = TargetSpec(entity=entity.canonical_id, level=MeasurementLevel.MRNA)
    context = BiologicalContext(
        organism="Homo sapiens",
        cell_type="macrophage",
        timepoint="12h",
        tissue="bone_marrow",
        condition=ConditionContext(oxygen="hypoxia", oxygen_level="1% O2"),
        study_design=StudyDesignContext(control="normoxia_21pct", replicates=3),
    )
    observations = [
        Observation(
            observation_id="OBS-1",
            entity="IL-6",
            assay="bulk_rna_seq",
            level=MeasurementLevel.MRNA,
            direction=SignalDirection.UP,
            source="csv_upload",
        ),
        Observation(
            observation_id="OBS-2",
            entity="IL6",
            assay="bulk_rna_seq",
            level=MeasurementLevel.MRNA,
            direction=SignalDirection.UP,
            source="csv_upload",
        ),
    ]

    assert all(
        validate_measurement_compatibility(target.level, obs.level).is_compatible
        for obs in observations
    )

    ctx_res = evaluate_mandatory_context(
        context,
        target=target,
        observations=observations,
        data_type=DataType.BULK_RNA_SEQ,
        cell_composition_changed=False,
    )
    assert ctx_res.mode == InferenceMode.WEAK_INFERENCE
    assert ctx_res.context_conflicts == []

    prior = ModelPriorKnowledgeNote(
        note_id="PRIOR-HIF-IL6",
        description="HIF1A and IL6 co-occur in hypoxic macrophage signatures.",
    )
    ev_null = EvidenceRecord(
        evidence_id="EV-NULL-1",
        source_id="PMID:99999",
        claim_summary="No effect observed on secreted IL6 in unstimulated baseline.",
        contradicts_hypothesis_ids=["H2"],
    )
    prov_report = build_provenance_report(
        context=context,
        target=target,
        observations=observations,
        evidence_records=[ev_null],
        model_prior_notes=[prior],
        data_type=DataType.BULK_RNA_SEQ,
        cell_composition_changed=False,
    )
    assert prov_report.conflicts == []
    assert prov_report.negative_evidence_items[0].negative_evidence_status == NegativeEvidenceStatus.NO_EFFECT_OBSERVED

    allowed_ids = {"OBS-1", "OBS-2", "EV-NULL-1", "PRIOR-HIF-IL6"}
    valid_llm = LLMExplanationPayload(
        query_id="e2e-001",
        mode=ctx_res.mode.value,
        has_perturbation_evidence=False,
        claims=[
            GroundedClaim(
                statement="Higher IL6 mRNA signal is observed under 1% O2 relative to control.",
                claim_kind="observation_summary",
                hypothesis_id="H1",
                cited_ids=["OBS-1", "OBS-2"],
            ),
            GroundedClaim(
                statement="No effect observed on secreted IL6 in unstimulated baseline.",
                claim_kind="evidence_summary",
                hypothesis_id="H2",
                cited_ids=["EV-NULL-1"],
            ),
        ],
        limitations=["Observational data only; perturbation evidence is absent."],
    )
    ok_res = validate_llm_explanation(
        valid_llm,
        allowed_source_ids=allowed_ids,
        allowed_hypothesis_ids={"H1", "H2"},
        expected_mode=ctx_res.mode.value,
        has_perturbation_evidence=False,
    )
    assert ok_res.is_valid is True

    causal_llm = LLMExplanationPayload(
        query_id="e2e-002",
        mode=ctx_res.mode.value,
        has_perturbation_evidence=False,
        claims=[
            GroundedClaim(
                statement="Hypoxia causes IL6 elevation in macrophages.",
                claim_kind="hypothesis_status",
                hypothesis_id="H1",
                cited_ids=["OBS-1"],
            )
        ],
    )
    blocked_res = validate_llm_explanation(
        causal_llm,
        allowed_source_ids=allowed_ids,
        allowed_hypothesis_ids={"H1", "H2"},
        expected_mode=ctx_res.mode.value,
        has_perturbation_evidence=False,
    )
    assert blocked_res.is_valid is False
    assert any(
        i.code == "forbidden_causal_claim_without_perturbation" for i in blocked_res.issues
    )


def test_end_to_end_conflict_and_level_mismatch_abstains():
    target = TargetSpec(entity="IL6", level=MeasurementLevel.SECRETED_PROTEIN)
    context = BiologicalContext(
        organism="Homo sapiens",
        cell_type="macrophage",
        condition=ConditionContext(oxygen="hypoxia", oxygen_level="1% O2"),
        study_design=StudyDesignContext(control="normoxia"),
    )
    observations = [
        Observation(
            observation_id="OBS-MRNA-1",
            entity="IL6",
            assay="bulk_rna_seq",
            level=MeasurementLevel.MRNA,
            direction=SignalDirection.UP,
        )
    ]

    ctx_res = evaluate_mandatory_context(
        context,
        target=target,
        observations=observations,
        data_type=DataType.BULK_RNA_SEQ,
        cell_composition_changed=True,
    )
    assert ctx_res.mode == InferenceMode.HARD_ABSTENTION

    prov_report = build_provenance_report(
        context=context,
        target=target,
        observations=observations,
        data_type=DataType.BULK_RNA_SEQ,
        cell_composition_changed=True,
    )
    conflict_types = {c.conflict_type for c in prov_report.conflicts}
    assert "measurement_level_conflict" in conflict_types
    assert "rule_conflict" in conflict_types
    assert all(c.auto_resolved is False for c in prov_report.conflicts)


def test_additional_branches_across_new_modules():
    from hypothesis_engine_v0_1.biological_layers import (
        evaluate_mandatory_context,
        validate_measurement_compatibility,
    )
    from hypothesis_engine_v0_1.conflict_detector import (
        NegativeEvidenceStatus,
        classify_negative_evidence,
        detect_conflicts,
    )
    from hypothesis_engine_v0_1.llm_contract import (
        GroundedClaim,
        LLMExplanationPayload,
        validate_llm_explanation,
    )
    from hypothesis_engine_v0_1.schemas import (
        BiologicalContext,
        ConditionContext,
        DataType,
        EvidenceRecord,
        MeasurementLevel,
        Observation,
        SignalDirection,
        StudyDesignContext,
        TargetSpec,
    )

    unspec = validate_measurement_compatibility(MeasurementLevel.NOT_SPECIFIED, MeasurementLevel.MRNA)
    assert unspec.status == "unspecified_level"

    hypoxic_normoxia_label = evaluate_mandatory_context(
        BiologicalContext(
            organism="Homo sapiens",
            cell_type="macrophage",
            timepoint="4h",
            tissue="lung",
            condition=ConditionContext(oxygen="normoxia", oxygen_level="1% O2"),
            study_design=StudyDesignContext(control="ctrl", replicates=2),
        ),
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        data_type=DataType.BULK_RNA_SEQ,
        cell_composition_changed=True,
    )
    assert hypoxic_normoxia_label.mode == "hard_abstention"

    weak_bulk = evaluate_mandatory_context(
        BiologicalContext(
            organism="Homo sapiens",
            cell_type="macrophage",
            condition=ConditionContext(oxygen="1% O2"),
            study_design=StudyDesignContext(control="ctrl"),
        ),
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        data_type=DataType.BULK_RNA_SEQ,
        cell_composition_changed=False,
    )
    assert weak_bulk.mode == "weak_inference"

    assert classify_negative_evidence(
        EvidenceRecord(evidence_id="E-NR", source_id="S1", claim_summary="Not replicated in cohort"),
        replicated=False,
    ) == NegativeEvidenceStatus.NOT_REPLICATED
    assert classify_negative_evidence(
        EvidenceRecord(evidence_id="E-REF", source_id="S2", claim_summary="Refuted by follow-up"),
    ) == NegativeEvidenceStatus.REFUTED
    assert classify_negative_evidence(
        EvidenceRecord(evidence_id="E-INS", source_id="S3", claim_summary="Ambiguous assay"),
    ) == NegativeEvidenceStatus.INSUFFICIENT_EVIDENCE

    ev_conf = detect_conflicts(
        context=BiologicalContext(organism="Homo sapiens", cell_type="macrophage"),
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        observations=[
            Observation(
                observation_id="O1",
                entity="IL6",
                assay="RNA-seq",
                level=MeasurementLevel.MRNA,
                direction=SignalDirection.UP,
            )
        ],
        evidence_records=[
            EvidenceRecord(
                evidence_id="EV-DUP-POL",
                source_id="PMID:1",
                claim_summary="Mixed claim",
                supports_hypothesis_ids=["H1"],
                contradicts_hypothesis_ids=["H1"],
            )
        ],
    )
    assert any(c.conflict_type == "evidence_conflict" for c in ev_conf)

    mm = validate_llm_explanation(
        LLMExplanationPayload(
            query_id="q-mm",
            mode="conditional_inference",
            claims=[
                GroundedClaim(
                    statement="Observed signal in dataset.",
                    claim_kind="observation_summary",
                    cited_ids=["O1"],
                )
            ],
        ),
        allowed_source_ids={"O1"},
        allowed_hypothesis_ids={"H1"},
        expected_mode="weak_inference",
    )
    assert any(i.code == "mode_mismatch" for i in mm.issues)


def test_remaining_branches_biological_and_conflict():
    from hypothesis_engine_v0_1.biological_layers import evaluate_mandatory_context
    from hypothesis_engine_v0_1.conflict_detector import NegativeEvidenceStatus, build_provenance_report, detect_conflicts
    from hypothesis_engine_v0_1.schemas import BiologicalContext, ConditionContext, EvidenceRecord, MeasurementLevel, Observation, SignalDirection, StudyDesignContext, TargetSpec

    ctx = BiologicalContext(
        organism="Homo sapiens",
        cell_type="macrophage",
        timepoint="24h",
        tissue="lung",
        condition=ConditionContext(oxygen="normoxia", oxygen_level="21% O2", stimulus="LPS"),
        study_design=StudyDesignContext(control="wt_vehicle", replicates=3),
    )
    res_ctx = evaluate_mandatory_context(
        ctx,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        observations=[Observation(observation_id="O-PERT", entity="IL6", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UP)],
    )
    assert res_ctx.mode in {"weak_inference", "conditional_inference"}

    lvl_c = detect_conflicts(
        context=BiologicalContext(organism="Homo sapiens", cell_type="macrophage"),
        target=TargetSpec(entity="IL6", level=MeasurementLevel.SECRETED_PROTEIN),
        observations=[Observation(observation_id="O-LVL-1", entity="IL6", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UP)],
    )
    assert any(c.conflict_type == "measurement_level_conflict" for c in lvl_c)

    prov = build_provenance_report(
        context=BiologicalContext(organism="Homo sapiens", cell_type="macrophage"),
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        observations=[Observation(observation_id="O-P1", entity="IL6", assay="RNA-seq", level=MeasurementLevel.MRNA, direction=SignalDirection.UP)],
        evidence_records=[
            EvidenceRecord(evidence_id="EV-OK", source_id="PMID:555", claim_summary="IL6 transcript rises.", supports_hypothesis_ids=["H1"], curated=True),
            EvidenceRecord(evidence_id="EV-RET", source_id="PMID:556", claim_summary="Retracted observation of IL6.", curated=True),
        ],
    )
    by_id = {r.source_id: r for r in prov.provenance_records}
    assert by_id["EV-OK"].negative_evidence_status is None and by_id["EV-OK"].citable_as_evidence is True
    assert by_id["EV-RET"].negative_evidence_status == NegativeEvidenceStatus.RETRACTED and by_id["EV-RET"].citable_as_evidence is False
