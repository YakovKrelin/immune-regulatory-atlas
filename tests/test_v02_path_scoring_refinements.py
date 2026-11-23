from bioos.engine.path_ranking import (
    PathScoringWeights,
    aggregate_weight_variant_consensus_v02,
    attach_snapshot_provenance,
    attribute_marker_group_contributions_v02,
    attribute_path_evidence_v02,
    audit_context_score_shifts_v02,
    audit_path_rank_shifts_v02,
    build_path_score_delta_matrix_v02,
    build_ranked_hypothesis_payload_v02,
    classify_rank_margins_v02,
    compare_context_rankings_v02,
    compare_path_scores_v02,
    compute_context_field_coverage_v02,
    compute_context_specificity_v02,
    compute_path_overlap_metrics_v02,
    compute_score_breakdown_shares_v02,
    evaluate_context_perturbation_stability_v02,
    evaluate_weight_sensitivity_v02,
    explain_path_ranking_v02,
    filter_and_rank_paths_v02,
    gate_paths_by_required_context_v02,
    group_ranked_paths_by_context_v02,
    partition_paths_by_confidence_tier_v02,
    profile_context_field_sensitivity_v02,
    rank_contexts_for_path_v02,
    rank_differential_contexts_v02,
    rank_paths_across_contexts_v02,
    rank_paths_by_consensus_v02,
    rank_paths_by_context_weighted_consensus_v02,
    rank_paths_by_weighted_context_v02,
    rank_paths_with_context_v02,
    score_candidate_path_v02,
    score_conflict_context_tradeoff_v02,
    select_top_paths_v02,
    serialize_ranked_paths_v02,
    summarize_context_breadth_v02,
    summarize_ranked_paths_v02,
)


def test_context_coverage_and_evidence_boost_preferred_path():
    full_ctx_path = {
        "path_id": "P_FULL_CTX",
        "edges": [
            {
                "id": "BIOOS:A000001",
                "subject": "TNF",
                "object": "IL6",
                "conflict": False,
                "evidence_score": 0.9,
                "context": {"tissue": "Blood", "cell_type": "Macrophage", "eco": "A"},
            }
        ],
    }
    partial_ctx_path = {
        "path_id": "P_PARTIAL_CTX",
        "edges": [
            {
                "id": "BIOOS:A000002",
                "subject": "TNF",
                "object": "IL6",
                "conflict": False,
                "context": {"tissue": "Blood", "cell_type": "T Cell", "eco": "C"},
            }
        ],
    }

    ranked = rank_paths_with_context_v02(
        candidates=[partial_ctx_path, full_ctx_path],
        active_markers=["TNF", "IL6"],
        anchor_markers=["TNF"],
        counter_markers=[],
        target_context={"tissue": "Blood", "cell_type": "Macrophage", "eco": "A"},
    )

    assert [r["path_id"] for r in ranked] == ["P_FULL_CTX", "P_PARTIAL_CTX"]
    assert ranked[0]["context_coverage"] == 1.0
    assert ranked[0]["evidence_bonus"] > 0.0
    assert ranked[0]["net_score"] > ranked[1]["net_score"]
    assert 0.0 <= ranked[0]["normalized_confidence"] <= 1.0
    assert "bonuses" in ranked[0]["score_breakdown"]

    shares = compute_score_breakdown_shares_v02(ranked[0])
    assert shares["total_magnitude"] > 0.0
    assert round(sum(shares["component_shares"].values()), 4) <= 1.0004
    assert shares["dominant_component"] != "none"

    marker_contrib = attribute_marker_group_contributions_v02(ranked[0])
    assert marker_contrib["matched_anchor_count"] == 1
    assert marker_contrib["matched_active_count"] == 2
    assert marker_contrib["net_marker_effect"] > 0.0

    margins = classify_rank_margins_v02(ranked, narrow_threshold=0.15)
    assert margins[0]["margin_class"] == "decisive"
    assert margins[1]["margin_class"] == "tail"


def test_context_perturbation_stability():
    p_a = {
        "path_id": "P_A",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}}],
    }
    p_b = {
        "path_id": "P_B",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Liver"}}],
    }

    stab = evaluate_context_perturbation_stability_v02(
        candidates=[p_a, p_b],
        baseline_context={"tissue": "Blood"},
        perturbed_contexts=[{"tissue": "Blood"}, {"tissue": "Liver"}],
    )
    by_id = {row["path_id"]: row for row in stab}
    assert by_id["P_A"]["perturbations_evaluated"] == 2
    assert by_id["P_A"]["max_abs_score_swing"] > 0.0
    assert 0.0 <= by_id["P_A"]["stability_index"] <= 1.0


def test_context_pair_ranking_comparison():
    p_blood = {
        "path_id": "P_BLOOD",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}}],
    }
    p_liver = {
        "path_id": "P_LIVER",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Liver"}}],
    }

    comp = compare_context_rankings_v02(
        candidates=[p_blood, p_liver],
        context_a={"tissue": "Blood"},
        context_b={"tissue": "Liver"},
        top_k=1,
    )
    assert comp["top_k_jaccard"] == 0.0
    assert comp["top_path_context_a"] == "P_BLOOD"
    assert comp["top_path_context_b"] == "P_LIVER"
    assert comp["top_path_preserved"] is False
    assert comp["mean_abs_rank_shift"] == 1.0


def test_context_field_coverage_sensitivity_and_conflict_tradeoff():
    edges = [
        {"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood", "cell_type": "Macrophage"}},
        {"id": "E2", "subject": "IL6", "object": "STAT3", "conflict": True, "context": {"tissue": "Blood", "cell_type": "T Cell"}},
    ]
    field_cov = compute_context_field_coverage_v02(
        edges=edges,
        target_context={"tissue": "Blood", "cell_type": "Macrophage"},
    )
    assert field_cov["requested_fields"] == ["tissue", "cell_type"]
    assert field_cov["field_match_counts"] == {"tissue": 2, "cell_type": 1}
    assert field_cov["field_match_ratios"] == {"tissue": 1.0, "cell_type": 0.5}
    assert field_cov["fully_matched_fields"] == ["tissue"]

    sensitivity = profile_context_field_sensitivity_v02(
        path_id="P_SENS",
        edges=edges,
        target_context={"tissue": "Blood", "cell_type": "Macrophage"},
    )
    assert [row["field"] for row in sensitivity] == ["tissue", "cell_type"]
    assert sensitivity[0]["score_drop"] > sensitivity[1]["score_drop"]
    assert round(sum(row["relative_contribution"] for row in sensitivity), 4) > 0.0

    tradeoff = score_conflict_context_tradeoff_v02(
        path_id="P_TRADE",
        edges=edges,
        target_context={"tissue": "Blood", "cell_type": "Macrophage"},
    )
    assert tradeoff["conflict_count"] == 1
    assert tradeoff["context_bonus"] > 0.0
    assert tradeoff["conflict_penalty"] > 0.0
    assert 0.0 <= tradeoff["tradeoff_ratio"] <= 1.0


def test_sign_consistency_and_context_ranking():
    edges = [
        {"id": "E1", "subject": "TNF", "object": "IL6", "sign": "+", "conflict": False, "context": {"tissue": "Blood"}},
        {"id": "E2", "subject": "IL6", "object": "CRP", "sign": "+", "conflict": False, "context": {"tissue": "Blood"}},
    ]
    res = score_candidate_path_v02(path_id="P_SIGN", edges=edges)
    assert res["sign_consistency_score"] == 1.0
    assert res["sign_bonus"] > 0.0
    assert res["context_agreement_rate"] == 1.0
    assert res["conflict_resolution_ratio"] == 1.0

    ctx_ranked = rank_contexts_for_path_v02(
        path_id="P_SIGN",
        edges=edges,
        context_candidates=[
            {"context_id": "CTX_LOW", "context": {"tissue": "Liver", "cell_type": "Hepatocyte"}},
            {"context_id": "CTX_HIGH", "context": {"tissue": "Blood"}},
        ],
    )
    assert [c["context_id"] for c in ctx_ranked] == ["CTX_HIGH", "CTX_LOW"]
    assert [c["rank"] for c in ctx_ranked] == [1, 2]

    diff_ctx = rank_differential_contexts_v02(
        path_id="P_SIGN",
        edges=edges,
        context_candidates=[
            {"context_id": "CTX_LOW", "context": {"tissue": "Liver", "cell_type": "Hepatocyte"}},
            {"context_id": "CTX_HIGH", "context": {"tissue": "Blood"}},
        ],
    )
    assert diff_ctx[0]["context_id"] == "CTX_HIGH"
    assert diff_ctx[0]["differential_score"] > 0.0


def test_context_specificity_and_context_shift_audit():
    edges = [
        {"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood", "cell_type": "Macrophage"}},
    ]
    spec = compute_context_specificity_v02(
        path_id="P_SPEC",
        edges=edges,
        target_context={"tissue": "Blood", "cell_type": "Macrophage"},
        background_contexts=[
            {"tissue": "Liver", "cell_type": "Hepatocyte"},
            {"tissue": "Skin", "cell_type": "Keratinocyte"},
        ],
    )
    assert spec["target_context_score"] > 0.0
    assert spec["mean_background_score"] == 0.0
    assert spec["specificity_delta"] > 0.0

    shifts = audit_context_score_shifts_v02(
        candidates=[{"path_id": "P_SPEC", "edges": edges}],
        baseline_context={"tissue": "Blood", "cell_type": "Macrophage"},
        shifted_context={"tissue": "Liver", "cell_type": "Hepatocyte"},
    )
    assert len(shifts) == 1
    assert shifts[0]["delta_net_score"] < 0.0
    assert shifts[0]["delta_context_bonus"] < 0.0


def test_active_marker_coverage_and_context_agreement():
    high_cov = {
        "path_id": "P_HIGH_COV",
        "edges": [
            {"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}},
            {"id": "E2", "subject": "IL6", "object": "STAT3", "conflict": False, "context": {"tissue": "Blood"}},
        ],
    }
    low_cov = {
        "path_id": "P_LOW_COV",
        "edges": [
            {"id": "E3", "subject": "TNF", "object": "OTHER", "conflict": False, "context": {"tissue": "Blood"}},
            {"id": "E4", "subject": "OTHER", "object": "NOISE", "conflict": False, "context": {"tissue": "Liver"}},
        ],
    }

    ranked = rank_paths_with_context_v02(
        candidates=[low_cov, high_cov],
        active_markers=["TNF", "IL6", "STAT3"],
    )
    assert [r["path_id"] for r in ranked] == ["P_HIGH_COV", "P_LOW_COV"]
    assert ranked[0]["active_marker_coverage"] == 1.0
    assert ranked[0]["context_agreement_rate"] == 1.0
    assert ranked[1]["context_agreement_rate"] == 0.0


def test_counter_marker_penalty_and_multi_context_ranking():
    path_clean = {
        "path_id": "P_CLEAN",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}}],
    }
    path_counter = {
        "path_id": "P_COUNTER",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL10", "conflict": False, "context": {"tissue": "Blood"}}],
    }

    ranked = rank_paths_across_contexts_v02(
        candidates=[path_counter, path_clean],
        context_candidates=[
            {"context_id": "CTX_BLOOD", "context": {"tissue": "Blood"}},
            {"context_id": "CTX_LIVER", "context": {"tissue": "Liver"}},
        ],
        counter_markers=["IL10"],
    )
    assert ranked[0]["path_id"] == "P_CLEAN"
    assert ranked[0]["context_id"] == "CTX_BLOOD"
    assert ranked[0]["rank"] == 1

    breadth = summarize_context_breadth_v02(ranked)
    by_pid = {b["path_id"]: b for b in breadth}
    assert by_pid["P_CLEAN"]["contexts_matched"] == 1
    assert by_pid["P_CLEAN"]["matched_context_ids"] == ["CTX_BLOOD"]

    consensus = rank_paths_by_consensus_v02(
        candidates=[path_counter, path_clean],
        context_candidates=[
            {"context_id": "CTX_BLOOD", "context": {"tissue": "Blood"}},
            {"context_id": "CTX_LIVER", "context": {"tissue": "Liver"}},
        ],
        counter_markers=["IL10"],
    )
    assert consensus[0]["path_id"] == "P_CLEAN"
    assert consensus[0]["contexts_evaluated"] == 2
    assert consensus[0]["rank"] == 1

    weighted = rank_paths_by_weighted_context_v02(
        candidates=[path_counter, path_clean],
        context_candidates=[
            {"context_id": "CTX_BLOOD", "context": {"tissue": "Blood"}},
            {"context_id": "CTX_LIVER", "context": {"tissue": "Liver"}},
        ],
        context_weights={"CTX_BLOOD": 3.0, "CTX_LIVER": 1.0},
        counter_markers=["IL10"],
    )
    assert weighted[0]["path_id"] == "P_CLEAN"
    assert weighted[0]["weighted_mean_net_score"] > weighted[1]["weighted_mean_net_score"]

    weighted_consensus = rank_paths_by_context_weighted_consensus_v02(
        candidates=[path_counter, path_clean],
        context_candidates=[
            {"context_id": "CTX_BLOOD", "context": {"tissue": "Blood"}},
            {"context_id": "CTX_LIVER", "context": {"tissue": "Liver"}},
        ],
        context_weights={"CTX_BLOOD": 3.0, "CTX_LIVER": 1.0},
        counter_markers=["IL10"],
    )
    assert weighted_consensus[0]["path_id"] == "P_CLEAN"
    assert weighted_consensus[0]["matched_context_count"] == 1
    assert weighted_consensus[0]["weighted_mean_score"] > weighted_consensus[1]["weighted_mean_score"]

    grouped = group_ranked_paths_by_context_v02(ranked)
    assert sorted(grouped.keys()) == ["CTX_BLOOD", "CTX_LIVER"]
    assert grouped["CTX_BLOOD"][0]["context_rank"] == 1

    s_clean = score_candidate_path_v02("P_CLEAN", path_clean["edges"], counter_markers=["IL10"])
    s_counter = score_candidate_path_v02("P_COUNTER", path_counter["edges"], counter_markers=["IL10"])
    diff = compare_path_scores_v02(s_clean, s_counter)
    assert diff["delta_net_score"] > 0.0
    assert diff["delta_counter_penalty"] < 0.0

    matrix = build_path_score_delta_matrix_v02([s_clean, s_counter])
    assert len(matrix) == 2


def test_filter_and_required_context_gating_v02():
    clean_p = {
        "path_id": "P_CLEAN",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood", "cell_type": "Macrophage"}}],
    }
    tissue_only_p = {
        "path_id": "P_TISSUE_ONLY",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}}],
    }
    conflicted_p = {
        "path_id": "P_CONFLICT",
        "edges": [{"id": "E3", "subject": "TNF", "object": "IL6", "conflict": True, "context": {"tissue": "Blood", "cell_type": "Macrophage"}}],
    }

    surviving = filter_and_rank_paths_v02(
        candidates=[conflicted_p, clean_p],
        target_context={"tissue": "Blood"},
        max_conflicts=0,
        min_normalized_confidence=0.5,
    )
    assert [r["path_id"] for r in surviving] == ["P_CLEAN"]
    assert surviving[0]["rank"] == 1

    gated = gate_paths_by_required_context_v02(
        candidates=[tissue_only_p, clean_p],
        required_context={"tissue": "Blood", "cell_type": "Macrophage"},
    )
    assert [r["path_id"] for r in gated] == ["P_CLEAN"]
    assert gated[0]["rank"] == 1


def test_weight_sensitivity_and_rank_shifts():
    p_ctx = {
        "path_id": "P_CTX",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}}],
    }
    p_plain = {
        "path_id": "P_PLAIN",
        "edges": [{"id": "E2", "subject": "A", "object": "B", "conflict": False, "context": {"tissue": "Liver"}}],
    }
    variants = evaluate_weight_sensitivity_v02(
        candidates=[p_plain, p_ctx],
        weight_variants={"heavy_ctx": PathScoringWeights(context_match_bonus=1.0, context_coverage_bonus=1.5)},
        target_context={"tissue": "Blood"},
    )
    assert "baseline" in variants and "heavy_ctx" in variants
    shifts = audit_path_rank_shifts_v02(variants["baseline"], variants["heavy_ctx"])
    by_id = {row["path_id"]: row for row in shifts}
    assert by_id["P_CTX"]["perturbed_rank"] == 1
    assert by_id["P_CTX"]["delta_net_score"] > 0.0

    variant_consensus = aggregate_weight_variant_consensus_v02(variants)
    assert variant_consensus[0]["path_id"] == "P_CTX"
    assert variant_consensus[0]["variants_evaluated"] == 2


def test_evidence_attribution_and_ranking_explanation():
    edges_a = [
        {"id": "E1", "subject": "TNF", "object": "IL6", "sign": "+", "conflict": False, "evidence_score": 0.8, "context": {"tissue": "Blood"}}
    ]
    edges_b = [
        {"id": "E2", "subject": "TNF", "object": "IL6", "sign": "+", "conflict": False, "context": {"tissue": "Liver"}}
    ]
    attr = attribute_path_evidence_v02(edges_a, target_context={"tissue": "Blood"})
    assert attr["total_edges"] == 1
    assert attr["supported_edges"] == 1
    assert attr["context_matched_edges"] == 1
    assert attr["sign_counts"]["positive"] == 1

    overlap = compute_path_overlap_metrics_v02(
        edges_a,
        active_markers=["TNF", "IL6", "STAT3"],
        anchor_markers=["TNF", "IFNG"],
        counter_markers=["IL10"],
    )
    assert overlap["active_jaccard"] > 0.0
    assert overlap["unmatched_active_markers"] == ["STAT3"]
    assert overlap["unmatched_anchor_markers"] == ["IFNG"]

    ranked = rank_paths_with_context_v02(
        candidates=[{"path_id": "P_B", "edges": edges_b}, {"path_id": "P_A", "edges": edges_a}],
        target_context={"tissue": "Blood"},
    )
    exp = explain_path_ranking_v02(
        ranked,
        index=0,
        edges_by_path_id={"P_A": edges_a, "P_B": edges_b},
        target_context={"tissue": "Blood"},
    )
    assert exp is not None
    assert exp["path_id"] == "P_A"
    assert exp["evidence_attribution"]["supported_edges"] == 1
    assert exp["comparison_to_next"]["next_path_id"] == "P_B"
    assert exp["comparison_to_next"]["score_lead_over_next"] > 0.0


def test_top_k_selection_and_summary():
    p1 = {"path_id": "P1", "edges": [{"id": "E1", "subject": "A", "object": "B", "context": {"tissue": "Blood"}}]}
    p2 = {"path_id": "P2", "edges": [{"id": "E2", "subject": "B", "object": "C", "context": {"tissue": "Blood"}}]}
    p3 = {"path_id": "P3", "edges": [{"id": "E3", "subject": "C", "object": "D", "context": {"tissue": "Blood"}}]}

    ranked = rank_paths_with_context_v02(
        candidates=[p3, p1, p2],
        target_context={"tissue": "Blood"},
    )
    top2 = select_top_paths_v02(ranked, top_k=2)
    assert len(top2) == 2
    assert [r["rank"] for r in top2] == [1, 2]

    compact = serialize_ranked_paths_v02(top2, include_breakdown=True)
    assert len(compact) == 2
    assert "score_breakdown" in compact[0]

    tiers = partition_paths_by_confidence_tier_v02(ranked, high_threshold=0.70, moderate_threshold=0.40)
    assert len(tiers["high"]) + len(tiers["moderate"]) + len(tiers["low"]) == 3

    summary = summarize_ranked_paths_v02(ranked)
    assert summary["count"] == 3
    assert summary["top_path_id"] == ranked[0]["path_id"]
    assert summary["total_conflicts"] == 0
    assert summary["common_context_fields"] == ["tissue"]


def test_snapshot_provenance_envelope():
    sample = [{"path_id": "P1", "rank": 1, "net_score": 1.5}]
    env = attach_snapshot_provenance(sample, snapshot_id="snap_01", sha256="abc123")
    assert env["count"] == 1
    assert env["snapshot"] == {"snapshot_id": "snap_01", "sha256": "abc123"}

    payload = build_ranked_hypothesis_payload_v02(sample, snapshot_id="snap_01", sha256="abc123")
    assert payload["summary"]["count"] == 1
    assert payload["snapshot"]["snapshot_id"] == "snap_01"
