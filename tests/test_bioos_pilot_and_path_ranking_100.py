import sys
import runpy

import bioos.engine.path_ranking as pr
import bioos.main as bioos_main
from bioos.regulatory_graph import _normalize_counter_signs


def test_bifurcation_pilot_dunder_main(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "bioos.experimental.bifurcation_pilot", raising=False)
    runpy.run_module("bioos.experimental.bifurcation_pilot", run_name="__main__")
    out = capsys.readouterr().out
    assert "Mode:" in out


def test_path_ranking_low_level_helpers():
    assert pr._norm_sign(True) == 0
    assert pr._norm_sign(5) == 1
    assert pr._norm_sign(-3) == -1
    assert pr._norm_sign(0) == 0

    assert pr._clamp01(-0.5) == 0.0
    assert pr._clamp01(1.5) == 1.0
    assert pr._clamp01(0.25) == 0.25

    assert pr._jaccard(set(), set()) == 0.0
    assert pr._jaccard({"A"}, {"A", "B"}) == 0.5


def test_path_ranking_empty_cohorts_and_include_rejected_variants():
    assert pr.summarize_consensus_net_score_cohorts_v02([])["candidate_count"] == 0
    assert pr.summarize_consensus_confidence_cohorts_v02([])["candidate_count"] == 0
    assert pr.summarize_consensus_conflict_cohorts_v02([])["candidate_count"] == 0
    assert pr.summarize_consensus_retention_cohorts_v02([])["candidate_count"] == 0
    assert pr.summarize_consensus_retention_dispersion_v02([])["candidate_count"] == 0
    assert pr.summarize_consensus_rank_shifts_v02([])["candidate_count"] == 0
    assert pr.summarize_consensus_stability_v02([])["candidate_count"] == 0

    rows = [
        {"path_id": "", "conflict_rate": 0.5, "max_rank_shift": 5, "conflict_count": 1, "context_coverage": 0.0},
        {
            "path_id": "p1",
            "mean_net_score": 0.1,
            "mean_normalized_confidence": 0.1,
            "conflict_rate": 0.9,
            "zero_conflict_contexts": 0,
            "retention_rate": 0.1,
            "stability_score": 0.1,
            "conflict_count": 0,
            "context_coverage": 0.0,
        },
    ]
    assert pr.filter_consensus_by_net_score_v02(rows, min_mean_net_score=0.8, include_rejected=False) == []
    assert pr.filter_consensus_by_confidence_v02(rows, min_mean_normalized_confidence=0.8, include_rejected=False) == []
    assert pr.filter_consensus_by_conflict_rate_v02(rows, max_conflict_rate=0.1, include_rejected=False) == []

    stab_rows = [
        {
            "path_id": "p1",
            "retention_rate": 0.1,
            "consensus_retention_rate": 0.1,
            "max_rank_shift": 5,
            "stability_confidence": 0.1,
            "stability_score": 0.1,
        }
    ]
    assert pr.filter_consensus_by_rank_shift_v02([rows[1]], stab_rows, max_rank_shift=1, include_rejected=False) == []
    assert isinstance(pr.filter_consensus_by_retention_v02(rows, stab_rows, min_retention_rate=0.9, include_rejected=False), list)
    assert isinstance(pr.filter_consensus_by_retention_v02(rows, stab_rows, min_retention_rate=0.9, include_rejected=True), list)
    assert (
        pr.filter_consensus_by_stability_confidence_v02(
            rows, stab_rows, min_stability_adjusted_confidence=0.9, include_rejected=False
        )
        == []
    )
    assert isinstance(
        pr.filter_consensus_by_stability_v02(
            rows, stab_rows, min_retention_rate=0.9, max_rank_shift=1, include_rejected=False
        ),
        list,
    )
    assert isinstance(
        pr.filter_consensus_by_stability_v02(
            rows, stab_rows, min_retention_rate=0.9, max_rank_shift=1, include_rejected=True
        ),
        list,
    )


def test_path_ranking_margins_correlation_and_thresholds():
    m = pr.compute_consensus_rank_margins_v02(
        [
            {"path_id": "p_a", "mean_net_score": 0.90},
            {"path_id": "p_b", "mean_net_score": 0.90},
            {"path_id": "p_c", "mean_net_score": 0.85},
            {"path_id": "p_d", "mean_net_score": 0.20},
        ],
        narrow_threshold=0.10,
    )
    assert len(m) == 4

    corr = pr.compute_consensus_rank_correlation_v02(
        [{"path_id": "p1", "rank": 1}],
        [{"path_id": "p2", "rank": 1}],
    )
    assert corr["shared_path_count"] == 0

    f_rows = [
        {"path_id": "p1", "mean_net_score": 0.2, "mean_normalized_confidence": 0.2, "conflict_rate": 0.8},
        {"path_id": "p2", "mean_net_score": 0.9, "mean_normalized_confidence": 0.9, "conflict_rate": 0.0},
    ]
    res_f = pr.filter_consensus_by_thresholds_v02(
        f_rows,
        min_mean_net_score=0.5,
        min_mean_normalized_confidence=0.5,
        max_conflict_rate=0.2,
        include_rejected=False,
    )
    assert len(res_f) == 1


def test_path_ranking_dispersion_coverage_and_scoring():
    sample_edges = [{"source": "A", "target": "B", "sign": 1}]
    assert pr.compute_conflict_aware_confidence_dispersion_v02("p_empty", sample_edges, [])["contexts_evaluated"] == 0
    assert pr.compute_path_conflict_intensity_v02("p_empty", [])["edge_count"] == 0
    c_no_conf = pr.compute_path_conflict_intensity_v02(
        "p_clean",
        [{"source": "A", "target": "B", "sign": 1}, {"source": "B", "target": "C", "sign": 1}],
    )
    assert isinstance(c_no_conf["conflicted_support_ratio"], float)

    assert pr.compute_path_confidence_dispersion_v02("p_empty", sample_edges, [])["contexts_evaluated"] == 0
    assert pr.compute_path_evidence_coverage_v02("p_empty", [])["edge_count"] == 0
    assert pr.compute_path_sign_polarity_coherence_v02("p_empty", [])["edge_count"] == 0
    assert pr.compute_context_edge_support_density_v02("p_empty", [])["edge_count"] == 0
    assert pr.compute_context_coverage_dispersion_v02("p_empty", sample_edges, [])["contexts_evaluated"] == 0

    scored = pr.score_candidate_path_v02(
        path_id="p1",
        edges=sample_edges,
        active_markers=["A"],
        anchor_markers=["B"],
        counter_markers=[],
    )
    bal = pr.score_marker_coverage_balance_v02(scored, expected_anchor_count=None)
    assert isinstance(bal, dict)

    assert pr.profile_context_field_sensitivity_v02("p1", sample_edges, target_context={}) == []


def test_path_ranking_explanations_tiers_gates_and_counter_signs():
    edges = [
        {"subject": "A", "object": "B", "sign": 1, "counter_sign": True, "context": {"tissue": "blood"}},
        {"subject": "B", "object": "C", "sign": -1, "context": {"tissue": "blood"}},
        {"subject": "C", "object": "D", "sign": 0},
    ]
    attr = pr.attribute_path_evidence_v02(edges, target_context={"tissue": "blood"})
    assert isinstance(attr, dict)

    overlap = pr.compute_path_overlap_metrics_v02(
        [
            {"subject": "A", "object": ""},
            {"subject": "", "object": "B"},
            {"subject": "", "object": ""},
        ]
    )
    assert overlap["path_token_count"] == 2

    assert pr.explain_path_ranking_v02([], index=0) is None
    ranked = [
        {"path_id": "p_a", "net_score": 0.5, "conflict_count": 0, "matched_counter_markers": [], "score_breakdown": {"x": 1}},
        {"path_id": "p_b", "net_score": 0.5, "conflict_count": 2, "matched_counter_markers": ["X", "Y"]},
        {"path_id": "p_c", "net_score": 0.5, "conflict_count": 2, "matched_counter_markers": ["X"]},
    ]
    exp0 = pr.explain_path_ranking_v02(ranked, index=0)
    assert exp0 is not None
    factors0 = exp0["comparison_to_next"]["tie_breaker_factors"]
    assert "fewer_conflicts" in factors0
    assert "fewer_counter_markers" in factors0
    assert "path_id_lexicographic" in factors0

    exp1 = pr.explain_path_ranking_v02(ranked, index=1)
    assert exp1 is not None and "path_id_lexicographic" in exp1["comparison_to_next"]["tie_breaker_factors"]

    exp2 = pr.explain_path_ranking_v02(ranked, index=2)
    assert exp2 is not None and exp2["comparison_to_next"] is None

    serialized = pr.serialize_ranked_paths_v02(ranked, include_breakdown=True)
    assert "score_breakdown" in serialized[0]
    serialized_plain = pr.serialize_ranked_paths_v02(ranked, include_breakdown=False)
    assert "score_breakdown" not in serialized_plain[0]

    margins = pr.classify_rank_margins_v02(
        [
            {"path_id": "p1", "net_score": 0.5},
            {"path_id": "p2", "net_score": 0.5},
            {"path_id": "p3", "net_score": 0.45},
            {"path_id": "p4", "net_score": 0.1},
        ],
        narrow_threshold=0.1,
    )
    assert len(margins) == 4

    tiers = pr.partition_paths_by_confidence_tier_v02(
        [
            {"path_id": "h", "normalized_confidence": 0.9},
            {"path_id": "m", "normalized_confidence": 0.6},
            {"path_id": "l", "normalized_confidence": 0.1},
        ]
    )
    assert len(tiers["high"]) == 1 and len(tiers["moderate"]) == 1

    cand = [
        {"path_id": "", "edges": [{"source": "A", "target": "B", "sign": 1}]},
        {"path_id": "p1", "edges": [{"source": "A", "target": "B", "sign": 1}]},
    ]
    assert len(pr.filter_and_rank_paths_v02(cand, min_normalized_confidence=0.0, max_counter_hits=0)) >= 1
    assert pr.filter_and_rank_paths_v02(cand, min_normalized_confidence=0.99) == []
    assert pr.filter_consensus_by_thresholds_v02([], min_mean_net_score=0.0, min_mean_normalized_confidence=0.0, max_conflict_rate=1.0) == []

    single_ctx = [{"context_id": "c0", "context": {"tissue": "lung"}}]
    assert pr.rank_differential_contexts_v02("p1", cand[1]["edges"], []) == []
    diff_single = pr.rank_differential_contexts_v02("p1", cand[1]["edges"], single_ctx)
    assert len(diff_single) == 1 and diff_single[0]["mean_other_context_score"] == 0.0

    ctx_candidates = [
        {"context_id": "c0", "context": {"tissue": "lung"}},
        {"context_id": "c1", "weight": 0.7, "context": {"tissue": "spleen"}},
        {"context_id": "c2", "weight": 0.4, "context": {"tissue": "liver"}},
    ]
    assert len(pr.gate_paths_by_required_context_v02(cand, {})) >= 1
    assert len(pr.rank_paths_by_weighted_context_v02(cand, ctx_candidates)) >= 1
    assert len(pr.rank_paths_by_context_weighted_consensus_v02(cand, ctx_candidates)) >= 1
    assert len(pr.summarize_context_breadth_v02(pr.rank_paths_across_contexts_v02(cand, ctx_candidates))) >= 1

    zero_conf_cand = [
        {"path_id": "", "edges": [{"source": "X", "target": "Y", "sign": 0}]},
        {"path_id": "p_zc", "edges": [{"source": "X", "target": "Y", "sign": 0}]},
    ]
    c_rows = pr.rank_consensus_by_confidence_v02(zero_conf_cand, single_ctx)
    assert len(c_rows) == 2 and all(r["confidence_weighted_mean_score"] == r["mean_net_score"] for r in c_rows)
    assert len(pr.rank_consensus_with_conflict_penalty_v02(zero_conf_cand, single_ctx)) == 2
    assert len(pr.rank_consensus_by_score_stability_v02(zero_conf_cand, single_ctx)) == 2

    agg = pr.aggregate_weight_variant_consensus_v02(
        {
            "v1": [{"path_id": "", "rank": 1, "net_score": 0.5}, {"path_id": "p1", "rank": 1, "net_score": 0.8}],
        }
    )
    assert len(agg) == 1

    assert pr.summarize_ranked_paths_v02([])["count"] == 0
    aligned = pr.compute_path_context_alignment_v02("p1", [{"source": "A", "target": "B"}], target_context={})
    assert aligned["context_alignment_score"] == 0.5

    cs_pos = pr.score_counter_sign_conflict_v02("p1", [{"source": "A", "target": "B", "sign": 1}], expected_sign=None)
    assert cs_pos["expected_sign"] == 1

    neg_edges = [{"source": "A", "target": "B", "sign": -1}, {"source": "B", "target": "C", "sign": -1}]
    cs_neg = pr.score_counter_sign_conflict_v02("p2", neg_edges, expected_sign=None)
    assert cs_neg["expected_sign"] == -1

    assert pr.summarize_context_counter_sign_cohort_v02([])["candidate_count"] == 0


def test_main_and_regulatory_graph_edge_branches(monkeypatch):
    tokens = _normalize_counter_signs(["", "  IL6_PATHWAY  ", "TNF_SIGNAL"])
    assert "IL6_PATHWAY" in tokens and "TNF_SIGNAL" in tokens

    assertions = [
        {"id": "e1", "source": "A", "target": "B", "relation": "regulates", "eco": "ECO:0000314"},
        {"id": "e2", "source": "B", "target": "A", "relation": "regulates", "eco": "ECO:0000314"},
        {"id": "e3", "source": "B", "target": "C", "relation": "regulates", "eco": "ECO:0000314"},
    ]
    monkeypatch.setattr(bioos_main, "_query_assertions", lambda **kwargs: assertions)
    res = bioos_main._traverse_graph(start="A", max_hops=2)
    assert res["start"] == "A"


def test_path_ranking_remaining_branches_closure():
    ranked_tie_desc = [
        {"path_id": "p_z", "net_score": 0.5, "conflict_count": 1, "matched_counter_markers": ["X"]},
        {"path_id": "p_a", "net_score": 0.5, "conflict_count": 1, "matched_counter_markers": ["X"]},
    ]
    exp = pr.explain_path_ranking_v02(ranked_tie_desc, index=0)
    assert exp is not None
    assert "path_id_lexicographic" not in exp["comparison_to_next"]["tie_breaker_factors"]

    cands = [
        {
            "path_id": "p_conf",
            "edges": [
                {"source": "A", "target": "B", "sign": 1},
                {"source": "A", "target": "B", "sign": -1},
            ],
        }
    ]
    ctxs = [{"context_id": "c0", "context": {"tissue": "lung"}}]
    assert len(pr.filter_and_rank_paths_v02(cands, min_normalized_confidence=0.0, max_counter_hits=0)) == 1
    w_rows = pr.rank_paths_by_context_weighted_consensus_v02(cands, ctxs)
    assert len(w_rows) == 1 and w_rows[0]["zero_conflict_contexts"] == 1

    cs_edges = [
        {"source": "A", "target": "B", "sign": 1, "counter_sign": True},
        {"source": "B", "target": "C", "sign": -1},
        {"source": "C", "target": "D", "sign": 0},
    ]
    res_cs = pr.score_counter_sign_conflict_v02("p_mix", cs_edges, expected_sign=None)
    assert res_cs["expected_sign"] == 0


def test_path_ranking_1460_and_2620_exact_paths():
    zero_conf_only = [{"path_id": "p_zero_conf", "edges": []}]
    ctx = [{"context_id": "c0", "context": {"tissue": "lung"}}]
    cw_rows = pr.rank_consensus_by_confidence_v02(zero_conf_only, ctx)
    assert len(cw_rows) == 1
    assert cw_rows[0]["confidence_weighted_mean_score"] == cw_rows[0]["mean_net_score"]

    counter_cand = [
        {
            "path_id": "p_counter",
            "edges": [{"subject": "A", "object": "B", "sign": 1}],
        }
    ]
    assert (
        pr.filter_and_rank_paths_v02(
            counter_cand,
            counter_markers=["B"],
            min_normalized_confidence=0.0,
            max_counter_hits=0,
        )
        == []
    )


def test_path_ranking_2986_nonzero_conflict_branch():
    conflicted_cand = [
        {
            "path_id": "p_with_conflict",
            "edges": [
                {"subject": "A", "object": "B", "sign": 1, "conflict": True},
            ],
        }
    ]
    ctxs = [{"context_id": "c0", "context": {"tissue": "lung"}}]
    rows = pr.rank_paths_by_context_weighted_consensus_v02(conflicted_cand, ctxs)
    assert len(rows) == 1
    assert rows[0]["zero_conflict_contexts"] == 0


def test_summarize_context_breadth_empty_path_id_branch():
    rows = [
        {"path_id": "", "context_id": "c0", "net_score": 0.5},
        {"path_id": "p_ok", "context_id": "c0", "net_score": 0.8},
    ]
    out = pr.summarize_context_breadth_v02(rows)
    assert len(out) == 1
    assert out[0]["path_id"] == "p_ok"
