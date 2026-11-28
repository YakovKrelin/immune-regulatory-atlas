from bioos.experimental.bifurcation_pilot import (
    AbsenceStatus,
    ContextField,
    ContextSpec,
    Direction,
    EvidenceItem,
    EvidenceStatus,
    InferenceMode,
    Observation,
    evaluate_pilot_scenario,
)


def test_evaluate_pilot_hard_abstention_and_missing_context():
    empty_ctx = ContextSpec(
        species=ContextField(None, AbsenceStatus.UNKNOWN),
        tissue_or_cell=ContextField(None, AbsenceStatus.UNKNOWN),
        condition_or_stimulus=ContextField(None, AbsenceStatus.UNKNOWN),
        timepoint=ContextField(None, AbsenceStatus.UNKNOWN),
        duration=ContextField(None, AbsenceStatus.UNKNOWN),
        measurement_level=None,
    )
    rep_empty = evaluate_pilot_scenario([], empty_ctx, [], snapshot_ref="sha256:empty")
    assert rep_empty.mode == InferenceMode.HARD_ABSTENTION
    assert rep_empty.hypotheses == []

    obs = [Observation("IL6", Direction.UP, measurement_level="RNA")]
    ev = [
        EvidenceItem("e1", "src:1", "TNF", Direction.DOWN, "IL6", EvidenceStatus.CONFLICTED),
        EvidenceItem("e2", "src:2", "IFNG", Direction.DOWN, "IL6", EvidenceStatus.REVIEWED),
    ]
    rep_missing = evaluate_pilot_scenario(obs, empty_ctx, ev, snapshot_ref="sha256:miss")
    assert len(rep_missing.missing_context_or_data) == 5
    assert len(rep_missing.conflicts) == 3
    assert rep_missing.mode == InferenceMode.WEAK_INFERENCE


def test_evaluate_pilot_conditional_inference_and_high_compatibility():
    full_ctx = ContextSpec(
        species=ContextField.from_value("Homo sapiens"),
        tissue_or_cell=ContextField.from_value("T cell"),
        condition_or_stimulus=ContextField.from_value("LPS"),
        timepoint=ContextField.from_value("6h"),
        duration=ContextField.from_value("6h"),
        measurement_level="protein",
    )
    obs_single = [Observation("TNF", Direction.UP, measurement_level="protein")]
    ev_ok = [EvidenceItem("e_ok", "src:ok", "TNF", Direction.UP, "TNF", EvidenceStatus.REVIEWED)]
    rep_cond = evaluate_pilot_scenario(obs_single, full_ctx, ev_ok, snapshot_ref="sha256:cond")
    assert rep_cond.missing_context_or_data == []
    assert rep_cond.conflicts == []
    assert rep_cond.mode == InferenceMode.CONDITIONAL_INFERENCE
    assert rep_cond.hypotheses[0].compatibility_level == "moderate"
    assert rep_cond.hypotheses[1].compatibility_level == "low"
    assert rep_cond.hypotheses[2].compatibility_level == "low"

    pbmc_ctx = ContextSpec(
        species=ContextField.from_value("Homo sapiens"),
        tissue_or_cell=ContextField.from_value("PBMC / macrophage"),
        condition_or_stimulus=ContextField.from_value("hypoxia"),
        timepoint=ContextField.from_value("12h"),
        duration=ContextField(None, AbsenceStatus.UNKNOWN),
        measurement_level="RNA-seq",
    )
    obs_all = [
        Observation("TNF", Direction.UP),
        Observation("IL6", Direction.UP),
        Observation("IFNG", Direction.UP),
        Observation("CXCL10", Direction.UP),
    ]
    rep_high = evaluate_pilot_scenario(obs_all, pbmc_ctx, [], snapshot_ref="sha256:high")
    assert rep_high.hypotheses[0].compatibility_level == "high"
    assert rep_high.hypotheses[1].compatibility_level == "moderate"
    assert rep_high.hypotheses[2].compatibility_level == "moderate"

def test_context_field_empty_and_mixed_direction_branch():
    from bioos.experimental.bifurcation_pilot import ContextField

    empty_field = ContextField.from_value("   ", fallback=AbsenceStatus.MISSING)
    assert empty_field.value is None
    assert empty_field.status == AbsenceStatus.MISSING

    ctx = ContextSpec(
        species=ContextField.from_value("Homo sapiens"),
        tissue_or_cell=ContextField.from_value("T cell"),
        condition_or_stimulus=ContextField.from_value("LPS"),
        timepoint=ContextField.from_value("6h"),
        duration=ContextField.from_value("6h"),
        measurement_level="protein",
    )
    obs = [Observation("IL6", Direction.UP)]
    ev_mixed = [EvidenceItem("e_mix", "src:mix", "TNF", Direction.MIXED, "IL6", EvidenceStatus.REVIEWED)]
    rep = evaluate_pilot_scenario(obs, ctx, ev_mixed, snapshot_ref="sha256:mix")
    assert rep.conflicts == []
