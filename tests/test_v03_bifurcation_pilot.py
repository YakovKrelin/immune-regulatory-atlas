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


def test_snapshot_adapter_attaches_sha256_and_paths():
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
                "nodes": ["TNF", "NFKB1", "IL6"],
                "sign": "+",
                "organism": "Homo sapiens",
                "cell_type": "macrophage",
                "stimulus": "hypoxia",
                "timepoint": "12h",
            },
            {
                "path_id": "p_ifng_cxcl10",
                "nodes": ["IFNG", "STAT1", "CXCL10"],
                "sign": "+",
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
    assert len(report.hypotheses) >= 2
    assert all(len(h.related_path_refs) > 0 for h in report.hypotheses)
