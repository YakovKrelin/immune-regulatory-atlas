from __future__ import annotations

from hypothesis_engine_v0_1.biological_layers import (
    evaluate_mandatory_context,
    resolve_biological_entity,
    validate_measurement_compatibility,
)
from hypothesis_engine_v0_1.schemas import (
    BiologicalContext,
    ConditionContext,
    DataType,
    InferenceMode,
    MeasurementLevel,
    Observation,
    StudyDesignContext,
    TargetSpec,
)


def test_resolve_biological_entity_synonyms_and_unknown():
    il6 = resolve_biological_entity("interleukin-6")
    assert il6.canonical_id == "IL6"
    assert il6.entity_kind == "gene_or_protein"
    assert il6.matched_alias is True

    mphi = resolve_biological_entity("Mphi")
    assert mphi.canonical_id == "macrophage"
    assert mphi.entity_kind == "cell_type"
    assert mphi.matched_alias is True

    custom = resolve_biological_entity("novel_factor_x")
    assert custom.canonical_id == "NOVEL_FACTOR_X"
    assert custom.entity_kind == "unknown"
    assert custom.matched_alias is False


def test_validate_measurement_compatibility():
    same = validate_measurement_compatibility(MeasurementLevel.MRNA, MeasurementLevel.MRNA)
    assert same.is_compatible is True
    assert same.status == "exact_match"

    mrna_vs_secreted = validate_measurement_compatibility(
        MeasurementLevel.SECRETED_PROTEIN,
        MeasurementLevel.MRNA,
    )
    assert mrna_vs_secreted.is_compatible is False
    assert mrna_vs_secreted.status == "incompatible_level"

    prot_vs_secreted = validate_measurement_compatibility(
        MeasurementLevel.SECRETED_PROTEIN,
        MeasurementLevel.TOTAL_PROTEIN,
    )
    assert prot_vs_secreted.is_compatible is False
    assert prot_vs_secreted.status == "compartment_shift"


def test_evaluate_mandatory_context_modes_and_abstention():
    missing_ctx = BiologicalContext(
        organism="",
        cell_type="macrophage",
        condition=ConditionContext(oxygen_level="1% O2"),
        study_design=StudyDesignContext(control="normoxia_vehicle"),
    )
    res_abstain = evaluate_mandatory_context(
        missing_ctx,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
    )
    assert res_abstain.mode == InferenceMode.HARD_ABSTENTION
    assert "organism" in res_abstain.missing_mandatory_fields

    conflict_ctx = BiologicalContext(
        organism="Homo sapiens",
        cell_type="macrophage",
        timepoint="12h",
        tissue="bone_marrow",
        condition=ConditionContext(oxygen="hypoxia", oxygen_level="21% O2"),
        study_design=StudyDesignContext(control="vehicle", replicates=3),
    )
    res_conflict = evaluate_mandatory_context(
        conflict_ctx,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
    )
    assert res_conflict.mode == InferenceMode.HARD_ABSTENTION
    assert "normoxic_oxygen_percent_with_hypoxia_label" in res_conflict.context_conflicts

    obs_mismatch = [
        Observation(
            observation_id="obs_1",
            entity="IL6",
            assay="ELISA",
            level=MeasurementLevel.SECRETED_PROTEIN,
        )
    ]
    valid_ctx = BiologicalContext(
        organism="Homo sapiens",
        cell_type="macrophage",
        timepoint="12h",
        tissue="bone_marrow",
        condition=ConditionContext(oxygen_level="1% O2", stimulus="hypoxia"),
        study_design=StudyDesignContext(control="21%_o2_vehicle", replicates=3),
    )
    res_level_conflict = evaluate_mandatory_context(
        valid_ctx,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        observations=obs_mismatch,
    )
    assert res_level_conflict.mode == InferenceMode.HARD_ABSTENTION
    assert any(c.startswith("measurement_level_mismatch:") for c in res_level_conflict.context_conflicts)

    res_cond = evaluate_mandatory_context(
        valid_ctx,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.MRNA),
        observations=[
            Observation(
                observation_id="obs_2",
                entity="IL6",
                assay="RNA-seq",
                level=MeasurementLevel.MRNA,
            )
        ],
        data_type=DataType.SINGLE_CELL_RNA_SEQ,
    )
    assert res_cond.mode == InferenceMode.CONDITIONAL_INFERENCE


def test_evaluate_mandatory_context_missing_mandatory_fields_and_unspecified_target_level():
    from hypothesis_engine_v0_1.biological_layers import evaluate_mandatory_context
    from hypothesis_engine_v0_1.schemas import (
        BiologicalContext,
        ConditionContext,
        MeasurementLevel,
        StudyDesignContext,
        TargetSpec,
    )

    sparse_ctx = BiologicalContext(
        organism="Homo sapiens",
        cell_type="",
        condition=ConditionContext(),
        study_design=StudyDesignContext(control=""),
    )
    res = evaluate_mandatory_context(
        sparse_ctx,
        target=TargetSpec(entity="IL6", level=MeasurementLevel.NOT_SPECIFIED),
    )
    assert "cell_type" in res.missing_mandatory
    assert "condition" in res.missing_mandatory
    assert "study_design.control" in res.missing_mandatory
    assert "target.level" in res.missing_mandatory
