from bioos.data.knowledge import ASSERTIONS, BIOLOGICAL_STATES, MODULES
from bioos.engine.dataset_engine import evaluate_dataset
from bioos.engine.graph_causality import (
    analyze_why_elevated,
    get_neighborhood,
    simulate_perturbation,
)
from bioos.engine.hypotheses_export import build_hypotheses_from_signals
from bioos.models import DatasetUploadRequest, ExpressionMeasurement


def test_anchor_markers_and_counter_marker_penalty():
    base_measurements = [
        ExpressionMeasurement(symbol="TNF", direction="up"),
        ExpressionMeasurement(symbol="IL6", direction="up"),
        ExpressionMeasurement(symbol="CRP", direction="up"),
    ]

    without_counter = evaluate_dataset(
        DatasetUploadRequest(
            dataset_name="infl_anchors_only",
            modality="Cytokines",
            measurement_level="secreted",
            organism="Homo sapiens",
            tissue="Blood",
            cell_type="Macrophage",
            measurements=base_measurements,
        ),
        ASSERTIONS,
        BIOLOGICAL_STATES,
        MODULES,
    )

    with_counter = evaluate_dataset(
        DatasetUploadRequest(
            dataset_name="infl_with_counter",
            modality="Cytokines",
            measurement_level="secreted",
            organism="Homo sapiens",
            tissue="Blood",
            cell_type="Macrophage",
            measurements=[
                *base_measurements,
                ExpressionMeasurement(symbol="IL10", direction="up"),
                ExpressionMeasurement(symbol="TGFB1", direction="up"),
            ],
        ),
        ASSERTIONS,
        BIOLOGICAL_STATES,
        MODULES,
    )

    state_clean = next(s for s in without_counter["all_evaluated_states"] if s["state_id"] == "BIOOS:S000001")
    state_penalized = next(s for s in with_counter["all_evaluated_states"] if s["state_id"] == "BIOOS:S000001")

    assert state_clean["matched_anchors"] == ["CRP", "IL6", "TNF"]
    assert state_clean["counter_penalty"] == 0.0

    assert state_penalized["matched_counter_markers"] == ["IL10", "TGFB1"]
    assert state_penalized["counter_penalty"] > 0.0
    assert state_penalized["coverage_ratio"] < state_clean["coverage_ratio"]


def test_graph_outputs_include_conflict_fields():
    hood = get_neighborhood(ASSERTIONS, start_node="HIF1A", max_hops=1)
    assert hood["conflict_count"] >= 1
    assert any(edge.get("id") == "BIOOS:A000004" for edge in hood["conflicting_edges"])

    why = analyze_why_elevated(ASSERTIONS, target_entity="IFNG")
    assert "conflict_count" in why
    assert "conflicting_edges" in why
    assert why["conflict_count"] >= 1
    assert any(edge.get("id") == "BIOOS:A000004" for edge in why["conflicting_edges"])

    pert = simulate_perturbation(ASSERTIONS, target_entity="HIF1A", action="suppress")
    assert pert["conflict_count"] >= 1
    assert any(edge.get("id") == "BIOOS:A000004" for edge in pert["conflicting_edges"])


def test_hypotheses_output_includes_conflict_and_counter_marker_fields():
    hyp = build_hypotheses_from_signals("TNF up, IL6 up, IFNG up, CXCL10 up, HIF1A up", ASSERTIONS)
    assert hyp["conflict_count"] >= 1
    assert any(c.get("id") == "BIOOS:A000004" for c in hyp["conflicts"])
    assert "IL10" in hyp["counter_markers_to_check"]
    assert "bifurcation_note" in hyp
