from bioos.engine.graph_snapshot import RegulatoryGraphSnapshot
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
from bioos.experimental.snapshot_adapter import run_bifurcation_from_snapshot


def test_weak_inference_bifurcation_for_cytokine_panel():
    ctx = ContextSpec(
        species=ContextField.from_value("Homo sapiens"),
        tissue_or_cell=ContextField.from_value("macrophage"),
        condition_or_stimulus=ContextField.from_value("hypoxia"),
        timepoint=ContextField.from_value("12h"),
        duration=ContextField(None, AbsenceStatus.UNKNOWN),
    )
    obs = [
        Observation("TNF", Direction.UP),
        Observation("IL6", Direction.UP),
        Observation("IFNG", Direction.UP),
        Observation("CXCL10", Direction.UP),
    ]
    ev = [
        EvidenceItem("ev_1", "pmid:test_1", "TNF", Direction.UP, "IL6", EvidenceStatus.REVIEWED),
        EvidenceItem("ev_2", "pmid:test_2", "IFNG", Direction.UP, "CXCL10", EvidenceStatus.REVIEWED),
    ]

    report = evaluate_pilot_scenario(obs, ctx, ev, snapshot_ref="sha256:test_snap")

    assert report.mode == InferenceMode.WEAK_INFERENCE
    assert len(report.hypotheses) >= 2
    assert any(h.hypothesis_id == "H1" for h in report.hypotheses)
    assert any(h.hypothesis_id == "H2" for h in report.hypotheses)
    assert len(report.recommended_next_tests) > 0
    assert any("measurement_level" in item for item in report.missing_context_or_data)


def test_hard_abstention_when_no_observations():
    ctx = ContextSpec(
        species=ContextField.from_value("Homo sapiens"),
        tissue_or_cell=ContextField.from_value("macrophage"),
        condition_or_stimulus=ContextField.from_value("hypoxia"),
    )

    report = evaluate_pilot_scenario(observations=[], context=ctx, evidence_items=[])

    assert report.mode == InferenceMode.HARD_ABSTENTION
    assert report.hypotheses == []


def test_snapshot_adapter_attaches_sha256_and_paths_per_branch():
    snap = RegulatoryGraphSnapshot(
        snapshot_id="snap_pilot_01",
        target_context={
            "organism": "Homo sapiens",
            "cell_type": "macrophage",
            "stimulus": "hypoxia",
            "timepoint": "12h",
        },
        candidates=[
            {
                "path_id": "p_tnf_il6",
                "edges": [{"id": "e1", "subject": "TNF", "object": "IL6", "sign": "+"}],
                "organism": "Homo sapiens",
                "cell_type": "macrophage",
                "stimulus": "hypoxia",
                "timepoint": "12h",
            },
            {
                "path_id": "p_tlr3_cxcl10",
                "edges": [{"id": "e2", "subject": "TLR3", "object": "CXCL10", "sign": "+"}],
                "organism": "Homo sapiens",
                "cell_type": "macrophage",
                "stimulus": "hypoxia",
                "timepoint": "12h",
            },
            {
                "path_id": "p_cd14_shift",
                "edges": [{"id": "e3", "subject": "CD14", "object": "HLA-DRA", "sign": "+"}],
                "organism": "Homo sapiens",
                "cell_type": "macrophage",
                "stimulus": "hypoxia",
                "timepoint": "12h",
            },
        ],
    )
    obs = [
        Observation("TNF", Direction.UP),
        Observation("IL6", Direction.UP),
        Observation("IFNG", Direction.UP),
        Observation("CXCL10", Direction.UP),
    ]

    report = run_bifurcation_from_snapshot(snap, obs, expected_sign="+")

    assert report.snapshot_ref == f"snap_pilot_01:sha256:{snap.sha256}"
    assert report.mode == InferenceMode.WEAK_INFERENCE
    by_id = {h.hypothesis_id: h for h in report.hypotheses}
    assert any(ref.startswith("p_tnf_il6:") for ref in by_id["H1"].related_path_refs)
    assert any(ref.startswith("p_tlr3_cxcl10:") for ref in by_id["H2"].related_path_refs)
    assert any(ref.startswith("p_cd14_shift:") for ref in by_id["H3"].related_path_refs)


def test_conflicted_evidence_triggers_weak_inference_and_records_conflicts():
    ctx = ContextSpec(
        species=ContextField.from_value("Homo sapiens"),
        tissue_or_cell=ContextField.from_value("macrophage"),
        condition_or_stimulus=ContextField.from_value("hypoxia"),
        timepoint=ContextField.from_value("12h"),
        duration=ContextField.from_value("12h"),
        measurement_level="rna",
    )
    obs = [
        Observation("TNF", Direction.UP),
        Observation("IL6", Direction.UP),
    ]
    ev = [
        EvidenceItem("ev_conf", "pmid:conflict_1", "TNF", Direction.DOWN, "TNF", EvidenceStatus.CONFLICTED),
        EvidenceItem("ev_opp", "pmid:conflict_2", "IFNG", Direction.DOWN, "IL6", EvidenceStatus.REVIEWED),
    ]

    report = evaluate_pilot_scenario(obs, ctx, ev, snapshot_ref="sha256:conflict_snap")

    assert report.mode == InferenceMode.WEAK_INFERENCE
    assert len(report.conflicts) >= 2
    assert any(h.hypothesis_id == "H1" and len(h.contradicting_evidence_ids) > 0 for h in report.hypotheses)


def test_snapshot_counter_sign_conflict_preserves_weak_inference():
    snap = RegulatoryGraphSnapshot(
        snapshot_id="snap_countersign_01",
        target_context={
            "organism": "Homo sapiens",
            "cell_type": "macrophage",
            "stimulus": "hypoxia",
            "timepoint": "12h",
        },
        candidates=[
            {
                "path_id": "p_counter_1",
                "edges": [{"id": "e_tnf_il6", "subject": "TNF", "object": "IL6", "sign": "-"}],
                "organism": "Homo sapiens",
                "cell_type": "macrophage",
                "stimulus": "hypoxia",
                "timepoint": "12h",
            },
        ],
    )
    obs = [Observation("TNF", Direction.UP)]

    report = run_bifurcation_from_snapshot(snap, obs, expected_sign="+")

    assert report.snapshot_ref == f"snap_countersign_01:sha256:{snap.sha256}"
    assert report.mode == InferenceMode.WEAK_INFERENCE
    assert any("p_counter_1" in c for c in report.conflicts)
