import pytest

from bioos.engine.path_ranking import (
    PathScoringWeights,
    aggregate_weight_variant_consensus_v02,
    attach_snapshot_provenance,
    attribute_marker_group_contributions_v02,
    attribute_path_evidence_v02,
    audit_context_score_shifts_v02,
    audit_path_rank_shifts_v02,
    build_consensus_report_v02,
    build_context_consensus_envelope_v02,
    build_path_score_delta_matrix_v02,
    build_ranked_hypothesis_payload_v02,
    calibrate_conflict_adjusted_confidence_v02,
    classify_rank_margins_v02,
    compare_context_rankings_v02,
    compare_path_scores_v02,
    compute_conflict_aware_confidence_dispersion_v02,
    compute_consensus_rank_correlation_v02,
    compute_consensus_rank_margins_v02,
    compute_consensus_rank_stability_v02,
    compute_context_coverage_dispersion_v02,
    compute_context_edge_support_density_v02,
    compute_context_field_coverage_v02,
    compute_context_rank_stability_v02,
    compute_context_specificity_v02,
    compute_path_confidence_dispersion_v02,
    compute_path_conflict_intensity_v02,
    compute_path_evidence_coverage_v02,
    compute_path_overlap_metrics_v02,
    compute_path_sign_polarity_coherence_v02,
    compute_score_breakdown_shares_v02,
    evaluate_context_perturbation_stability_v02,
    evaluate_weight_sensitivity_v02,
    explain_path_ranking_v02,
    filter_and_rank_paths_v02,
    filter_consensus_by_confidence_v02,
    filter_consensus_by_conflict_rate_v02,
    filter_consensus_by_net_score_v02,
    filter_consensus_by_rank_shift_v02,
    filter_consensus_by_retention_v02,
    filter_consensus_by_stability_confidence_v02,
    filter_consensus_by_stability_v02,
    filter_consensus_by_thresholds_v02,
    gate_paths_by_required_context_v02,
    group_ranked_paths_by_context_v02,
    partition_paths_by_confidence_tier_v02,
    profile_context_field_sensitivity_v02,
    rank_consensus_by_confidence_v02,
    rank_consensus_by_conflict_aware_confidence_v02,
    rank_consensus_by_score_stability_v02,
    rank_consensus_by_stability_confidence_v02,
    rank_consensus_by_stability_v02,
    rank_consensus_with_conflict_penalty_v02,
    rank_contexts_for_path_v02,
    rank_differential_contexts_v02,
    rank_conflict_free_consensus_v02,
    rank_paths_across_contexts_v02,
    rank_paths_by_consensus_v02,
    rank_paths_by_context_weighted_consensus_v02,
    rank_paths_by_weighted_context_v02,
    rank_paths_with_context_v02,
    score_candidate_path_v02,
    score_conflict_context_tradeoff_v02,
    score_marker_coverage_balance_v02,
    select_top_consensus_paths_v02,
    select_top_paths_v02,
    serialize_ranked_paths_v02,
    summarize_consensus_confidence_cohorts_v02,
    summarize_consensus_conflict_cohorts_v02,
    summarize_consensus_net_score_cohorts_v02,
    summarize_consensus_rank_shifts_v02,
    summarize_consensus_retention_cohorts_v02,
    summarize_consensus_retention_dispersion_v02,
    summarize_consensus_stability_v02,
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

    density = compute_context_edge_support_density_v02(
        path_id="P_FULL_CTX",
        edges=full_ctx_path["edges"],
        target_context={"tissue": "Blood", "cell_type": "Macrophage", "eco": "A"},
    )
    assert density["edge_count"] == 1
    assert density["supported_edge_ratio"] == 1.0
    assert density["context_matched_edge_ratio"] == 1.0
    assert density["joint_support_density"] == 1.0

    balance = score_marker_coverage_balance_v02(ranked[0], expected_anchor_count=1)
    assert balance["active_coverage"] == 1.0
    assert balance["anchor_hit_ratio"] == 1.0
    assert balance["counter_clean_ratio"] == 1.0
    assert 0.0 <= balance["marker_balance_score"] <= 1.0

    cal = calibrate_conflict_adjusted_confidence_v02(ranked[0])
    assert cal["raw_confidence"] == ranked[0]["normalized_confidence"]
    assert cal["conflict_discount"] == 0.0
    assert 0.0 <= cal["calibrated_confidence"] <= 1.0

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


def test_net_score_filtering_and_net_score_cohorts_summary():
    consensus_rows = [
        {
            "path_id": "P_MOD",
            "contexts_evaluated": 4,
            "zero_conflict_contexts": 3,
            "mean_net_score": 1.1,
            "mean_normalized_confidence": 0.7,
        },
        {
            "path_id": "P_STRONG",
            "contexts_evaluated": 4,
            "zero_conflict_contexts": 4,
            "mean_net_score": 2.1,
            "mean_normalized_confidence": 0.9,
        },
        {
            "path_id": "P_WEAK",
            "contexts_evaluated": 4,
            "zero_conflict_contexts": 1,
            "mean_net_score": 0.25,
            "mean_normalized_confidence": 0.3,
        },
    ]

    filtered = filter_consensus_by_net_score_v02(
        consensus_rows=consensus_rows,
        min_mean_net_score=1.0,
        min_zero_conflict_contexts=3,
        include_rejected=True,
    )
    accepted = [r for r in filtered if r.get("accepted")]
    rejected = [r for r in filtered if not r.get("accepted")]
    assert [r["path_id"] for r in accepted] == ["P_STRONG", "P_MOD"]
    assert [r["rank"] for r in accepted] == [1, 2]
    assert [r["path_id"] for r in rejected] == ["P_WEAK"]
    assert "below_min_mean_net_score" in rejected[0]["rejection_reasons"]
    assert "below_min_zero_conflict_contexts" in rejected[0]["rejection_reasons"]

    cohorts = summarize_consensus_net_score_cohorts_v02(consensus_rows)
    assert cohorts["candidate_count"] == 3
    assert cohorts["strong_score_count"] == 1
    assert cohorts["moderate_score_count"] == 1
    assert cohorts["weak_score_count"] == 1
    assert (
        cohorts["strong_score_share"]
        + cohorts["moderate_score_share"]
        + cohorts["weak_score_share"]
        == pytest.approx(1.0, abs=1e-3)
    )


def test_confidence_filtering_and_confidence_cohorts_summary():
    consensus_rows = [
        {
            "path_id": "P_MOD",
            "contexts_evaluated": 4,
            "zero_conflict_contexts": 3,
            "mean_net_score": 2.0,
            "mean_normalized_confidence": 0.55,
        },
        {
            "path_id": "P_HIGH",
            "contexts_evaluated": 4,
            "zero_conflict_contexts": 4,
            "mean_net_score": 1.6,
            "mean_normalized_confidence": 0.85,
        },
        {
            "path_id": "P_LOW",
            "contexts_evaluated": 4,
            "zero_conflict_contexts": 1,
            "mean_net_score": 1.2,
            "mean_normalized_confidence": 0.25,
        },
    ]

    filtered = filter_consensus_by_confidence_v02(
        consensus_rows=consensus_rows,
        min_mean_normalized_confidence=0.5,
        min_zero_conflict_contexts=3,
        include_rejected=True,
    )
    accepted = [r for r in filtered if r.get("accepted")]
    rejected = [r for r in filtered if not r.get("accepted")]
    assert [r["path_id"] for r in accepted] == ["P_HIGH", "P_MOD"]
    assert [r["rank"] for r in accepted] == [1, 2]
    assert [r["path_id"] for r in rejected] == ["P_LOW"]
    assert "below_min_mean_normalized_confidence" in rejected[0]["rejection_reasons"]
    assert "below_min_zero_conflict_contexts" in rejected[0]["rejection_reasons"]

    cohorts = summarize_consensus_confidence_cohorts_v02(consensus_rows)
    assert cohorts["candidate_count"] == 3
    assert cohorts["high_confidence_count"] == 1
    assert cohorts["moderate_confidence_count"] == 1
    assert cohorts["low_confidence_count"] == 1
    assert (
        cohorts["high_confidence_share"]
        + cohorts["moderate_confidence_share"]
        + cohorts["low_confidence_share"]
        == pytest.approx(1.0, abs=1e-3)
    )


def test_conflict_rate_filtering_and_conflict_cohorts_summary():
    consensus_rows = [
        {
            "path_id": "P_PARTIAL",
            "contexts_evaluated": 4,
            "zero_conflict_contexts": 3,
            "conflict_rate": 0.25,
            "mean_net_score": 2.0,
            "mean_normalized_confidence": 0.9,
        },
        {
            "path_id": "P_CLEAN",
            "contexts_evaluated": 4,
            "zero_conflict_contexts": 4,
            "conflict_rate": 0.0,
            "mean_net_score": 1.6,
            "mean_normalized_confidence": 0.85,
        },
        {
            "path_id": "P_HIGH_CONF",
            "contexts_evaluated": 4,
            "zero_conflict_contexts": 0,
            "conflict_rate": 0.75,
            "mean_net_score": 1.2,
            "mean_normalized_confidence": 0.5,
        },
    ]

    filtered = filter_consensus_by_conflict_rate_v02(
        consensus_rows=consensus_rows,
        max_conflict_rate=0.4,
        min_zero_conflict_contexts=3,
        include_rejected=True,
    )
    accepted = [r for r in filtered if r.get("accepted")]
    rejected = [r for r in filtered if not r.get("accepted")]
    assert [r["path_id"] for r in accepted] == ["P_CLEAN", "P_PARTIAL"]
    assert [r["rank"] for r in accepted] == [1, 2]
    assert [r["path_id"] for r in rejected] == ["P_HIGH_CONF"]
    assert "above_max_conflict_rate" in rejected[0]["rejection_reasons"]
    assert "below_min_zero_conflict_contexts" in rejected[0]["rejection_reasons"]

    cohorts = summarize_consensus_conflict_cohorts_v02(consensus_rows)
    assert cohorts["candidate_count"] == 3
    assert cohorts["conflict_free_count"] == 1
    assert cohorts["partial_conflict_count"] == 1
    assert cohorts["high_conflict_count"] == 1
    assert (
        cohorts["conflict_free_share"]
        + cohorts["partial_conflict_share"]
        + cohorts["high_conflict_share"]
        == pytest.approx(1.0, abs=1e-3)
    )


def test_rank_shift_filtering_and_retention_cohorts_summary():
    consensus_rows = [
        {"path_id": "P_SHIFT_2", "mean_net_score": 2.0, "mean_normalized_confidence": 0.9},
        {"path_id": "P_SHIFT_0", "mean_net_score": 1.5, "mean_normalized_confidence": 0.8},
        {"path_id": "P_SHIFT_3", "mean_net_score": 1.2, "mean_normalized_confidence": 0.5},
    ]
    stability_rows = [
        {"path_id": "P_SHIFT_2", "consensus_retention_rate": 0.6, "max_rank_shift": 2},
        {"path_id": "P_SHIFT_0", "consensus_retention_rate": 1.0, "max_rank_shift": 0},
        {"path_id": "P_SHIFT_3", "consensus_retention_rate": 0.25, "max_rank_shift": 3},
    ]

    filtered = filter_consensus_by_rank_shift_v02(
        consensus_rows=consensus_rows,
        stability_rows=stability_rows,
        max_rank_shift=2,
        min_retention_rate=0.5,
        include_rejected=True,
    )
    accepted = [r for r in filtered if r.get("accepted")]
    rejected = [r for r in filtered if not r.get("accepted")]
    assert [r["path_id"] for r in accepted] == ["P_SHIFT_0", "P_SHIFT_2"]
    assert [r["rank"] for r in accepted] == [1, 2]
    assert [r["path_id"] for r in rejected] == ["P_SHIFT_3"]
    assert "above_max_rank_shift" in rejected[0]["rejection_reasons"]
    assert "below_min_retention_rate" in rejected[0]["rejection_reasons"]

    cohorts = summarize_consensus_retention_cohorts_v02(stability_rows)
    assert cohorts["candidate_count"] == 3
    assert cohorts["high_retention_count"] == 1
    assert cohorts["moderate_retention_count"] == 1
    assert cohorts["low_retention_count"] == 1
    assert cohorts["high_retention_share"] + cohorts["moderate_retention_share"] + cohorts["low_retention_share"] == pytest.approx(1.0, abs=1e-3)


def test_retention_filtering_and_retention_dispersion_summary():
    consensus_rows = [
        {"path_id": "P_PARTIAL", "mean_net_score": 2.0, "mean_normalized_confidence": 0.9},
        {"path_id": "P_FULL", "mean_net_score": 1.5, "mean_normalized_confidence": 0.8},
        {"path_id": "P_ZERO", "mean_net_score": 1.2, "mean_normalized_confidence": 0.5},
    ]
    stability_rows = [
        {"path_id": "P_PARTIAL", "consensus_retention_rate": 0.5, "max_rank_shift": 2},
        {"path_id": "P_FULL", "consensus_retention_rate": 1.0, "max_rank_shift": 0},
        {"path_id": "P_ZERO", "consensus_retention_rate": 0.0, "max_rank_shift": 3},
    ]

    filtered = filter_consensus_by_retention_v02(
        consensus_rows=consensus_rows,
        stability_rows=stability_rows,
        min_retention_rate=0.5,
        max_rank_shift=2,
        include_rejected=True,
    )
    accepted = [r for r in filtered if r.get("accepted")]
    rejected = [r for r in filtered if not r.get("accepted")]
    assert [r["path_id"] for r in accepted] == ["P_FULL", "P_PARTIAL"]
    assert [r["rank"] for r in accepted] == [1, 2]
    assert [r["path_id"] for r in rejected] == ["P_ZERO"]
    assert "below_min_retention_rate" in rejected[0]["rejection_reasons"]
    assert "above_max_rank_shift" in rejected[0]["rejection_reasons"]

    disp = summarize_consensus_retention_dispersion_v02(stability_rows)
    assert disp["candidate_count"] == 3
    assert disp["full_retention_count"] == 1
    assert disp["partial_retention_count"] == 1
    assert disp["zero_retention_count"] == 1
    assert disp["min_retention_rate"] == 0.0
    assert disp["max_retention_rate"] == 1.0
    assert disp["retention_spread"] == 1.0
    assert disp["mean_retention_rate"] == 0.5


def test_stability_filtered_confidence_and_shift_summary():
    consensus_rows = [
        {"path_id": "P_LOW_STAB", "mean_net_score": 2.0, "mean_normalized_confidence": 0.9},
        {"path_id": "P_STRONG", "mean_net_score": 1.6, "mean_normalized_confidence": 0.8},
    ]
    stability_rows = [
        {"path_id": "P_LOW_STAB", "consensus_retention_rate": 0.25, "max_rank_shift": 2},
        {"path_id": "P_STRONG", "consensus_retention_rate": 1.0, "max_rank_shift": 0},
    ]

    filtered = filter_consensus_by_stability_confidence_v02(
        consensus_rows=consensus_rows,
        stability_rows=stability_rows,
        min_stability_adjusted_confidence=0.5,
        min_retention_rate=0.5,
        max_rank_shift=1,
        include_rejected=True,
    )
    accepted = [r for r in filtered if r.get("accepted")]
    rejected = [r for r in filtered if not r.get("accepted")]
    assert [r["path_id"] for r in accepted] == ["P_STRONG"]
    assert accepted[0]["stability_adjusted_confidence"] == 0.8
    assert accepted[0]["rank"] == 1
    assert [r["path_id"] for r in rejected] == ["P_LOW_STAB"]
    assert "below_min_stability_adjusted_confidence" in rejected[0]["rejection_reasons"]
    assert "below_min_retention_rate" in rejected[0]["rejection_reasons"]
    assert "above_max_rank_shift" in rejected[0]["rejection_reasons"]

    shift_summary = summarize_consensus_rank_shifts_v02(stability_rows)
    assert shift_summary["candidate_count"] == 2
    assert shift_summary["zero_shift_count"] == 1
    assert shift_summary["min_max_rank_shift"] == 0
    assert shift_summary["max_max_rank_shift"] == 2
    assert shift_summary["rank_shift_spread"] == 2
    assert shift_summary["mean_max_rank_shift"] == 1.0


def test_consensus_stability_confidence_ranking_and_summary():
    consensus_rows = [
        {"path_id": "P_UNSTABLE_CONF", "mean_net_score": 2.0, "mean_normalized_confidence": 0.9},
        {"path_id": "P_STABLE_CONF", "mean_net_score": 1.6, "mean_normalized_confidence": 0.75},
    ]
    stability_rows = [
        {"path_id": "P_UNSTABLE_CONF", "consensus_retention_rate": 0.25, "max_rank_shift": 2},
        {"path_id": "P_STABLE_CONF", "consensus_retention_rate": 1.0, "max_rank_shift": 0},
    ]

    ranked = rank_consensus_by_stability_confidence_v02(consensus_rows, stability_rows)
    assert [r["path_id"] for r in ranked] == ["P_STABLE_CONF", "P_UNSTABLE_CONF"]
    assert ranked[0]["stability_adjusted_confidence"] == 0.75
    assert ranked[1]["stability_adjusted_confidence"] == 0.225
    assert ranked[0]["rank"] == 1

    summary = summarize_consensus_stability_v02(stability_rows)
    assert summary["candidate_count"] == 2
    assert summary["min_retention_rate"] == 0.25
    assert summary["max_retention_rate"] == 1.0
    assert summary["retention_spread"] == 0.75
    assert summary["max_observed_rank_shift"] == 2
    assert summary["full_retention_count"] == 1


def test_consensus_stability_filtering_and_ranking():
    consensus_rows = [
        {"path_id": "P_UNSTABLE_HIGH", "mean_net_score": 2.0, "zero_conflict_contexts": 1},
        {"path_id": "P_STABLE_STEADY", "mean_net_score": 1.6, "zero_conflict_contexts": 2},
    ]
    stability_rows = [
        {"path_id": "P_UNSTABLE_HIGH", "consensus_retention_rate": 0.25, "max_rank_shift": 2},
        {"path_id": "P_STABLE_STEADY", "consensus_retention_rate": 1.0, "max_rank_shift": 0},
    ]

    ranked = rank_consensus_by_stability_v02(consensus_rows, stability_rows)
    assert [r["path_id"] for r in ranked] == ["P_STABLE_STEADY", "P_UNSTABLE_HIGH"]
    assert ranked[0]["stability_adjusted_score"] == 1.6
    assert ranked[1]["stability_adjusted_score"] == 0.5
    assert ranked[0]["rank"] == 1

    filtered = filter_consensus_by_stability_v02(
        consensus_rows=consensus_rows,
        stability_rows=stability_rows,
        min_retention_rate=0.5,
        max_rank_shift=1,
        include_rejected=True,
    )
    accepted = [r for r in filtered if r.get("accepted")]
    rejected = [r for r in filtered if not r.get("accepted")]
    assert [r["path_id"] for r in accepted] == ["P_STABLE_STEADY"]
    assert accepted[0]["rank"] == 1
    assert [r["path_id"] for r in rejected] == ["P_UNSTABLE_HIGH"]
    assert "below_min_retention_rate" in rejected[0]["rejection_reasons"]
    assert "above_max_rank_shift" in rejected[0]["rejection_reasons"]


def test_consensus_rank_margins_and_top_k_selection():
    p_blood = {
        "path_id": "P_BLOOD",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9, "context": {"tissue": "Blood"}}],
    }
    p_liver = {
        "path_id": "P_LIVER",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9, "context": {"tissue": "Liver"}}],
    }
    ctx_blood = {"context_id": "C1", "context": {"tissue": "Blood"}}
    ctx_liver = {"context_id": "C2", "context": {"tissue": "Liver"}}

    consensus = rank_paths_by_consensus_v02(
        candidates=[p_blood, p_liver],
        context_candidates=[ctx_blood, ctx_liver],
    )
    margins = compute_consensus_rank_margins_v02(consensus, narrow_threshold=0.15)
    assert len(margins) == 2
    assert margins[0]["margin_class"] == "tied"
    assert margins[0]["margin_to_next"] == 0.0
    assert margins[1]["margin_class"] == "tail"

    top1 = select_top_consensus_paths_v02(consensus, top_k=1)
    assert len(top1) == 1
    assert top1[0]["rank"] == 1


def test_consensus_rank_correlation_and_envelope():
    p_blood = {
        "path_id": "P_BLOOD",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9, "context": {"tissue": "Blood"}}],
    }
    p_liver = {
        "path_id": "P_LIVER",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9, "context": {"tissue": "Liver"}}],
    }
    ctx_blood = {"context_id": "C1", "context": {"tissue": "Blood"}}
    ctx_liver = {"context_id": "C2", "context": {"tissue": "Liver"}}

    primary = rank_paths_by_consensus_v02(
        candidates=[p_blood, p_liver],
        context_candidates=[ctx_blood, ctx_liver],
    )
    secondary = rank_consensus_by_confidence_v02(
        candidates=[p_blood, p_liver],
        context_candidates=[ctx_blood, ctx_liver],
    )

    corr = compute_consensus_rank_correlation_v02(primary, secondary)
    assert corr["shared_path_count"] == 2
    assert corr["max_abs_rank_shift"] >= 0
    assert 0.0 <= corr["exact_rank_agreement_rate"] <= 1.0
    assert 0.0 <= corr["rank_alignment_score"] <= 1.0

    stability = compute_consensus_rank_stability_v02(
        candidates=[p_blood, p_liver],
        context_subsets=[[ctx_blood], [ctx_liver]],
        baseline_contexts=[ctx_blood, ctx_liver],
    )
    envelope = build_context_consensus_envelope_v02(
        consensus_rows=primary,
        stability_rows=stability,
        snapshot_id="snap_v02",
        sha256="deadbeef",
    )
    assert envelope["summary"]["candidate_count"] == 2
    assert len(envelope["stability_diagnostics"]) == 2
    assert envelope["snapshot"] == {"snapshot_id": "snap_v02", "sha256": "deadbeef"}


def test_consensus_report_and_subset_rank_stability():
    p_blood = {
        "path_id": "P_BLOOD",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9, "context": {"tissue": "Blood"}}],
    }
    p_liver = {
        "path_id": "P_LIVER",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9, "context": {"tissue": "Liver"}}],
    }
    ctx_blood = {"context_id": "C1", "context": {"tissue": "Blood"}}
    ctx_liver = {"context_id": "C2", "context": {"tissue": "Liver"}}

    consensus = rank_paths_by_consensus_v02(
        candidates=[p_blood, p_liver],
        context_candidates=[ctx_blood, ctx_liver],
    )
    report = build_consensus_report_v02(
        consensus_rows=consensus,
        min_mean_net_score=1.0,
        snapshot_id="snap_v02",
        sha256="deadbeef",
    )
    assert report["total_candidates"] == 2
    assert report["accepted_count"] == 2
    assert report["rejected_count"] == 0
    assert report["snapshot"] == {"snapshot_id": "snap_v02", "sha256": "deadbeef"}

    stability = compute_consensus_rank_stability_v02(
        candidates=[p_blood, p_liver],
        context_subsets=[[ctx_blood], [ctx_liver]],
        baseline_contexts=[ctx_blood, ctx_liver],
    )
    assert len(stability) == 2
    for row in stability:
        assert row["subsets_evaluated"] == 2
        assert 0.0 <= row["consensus_retention_rate"] <= 1.0
        assert row["max_rank_shift"] >= 0


def test_consensus_threshold_filtering_and_conflict_aware_dispersion():
    p_clean = {
        "path_id": "P_CLEAN",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9, "context": {"tissue": "Blood"}}],
    }
    p_conflicted = {
        "path_id": "P_CONFLICTED",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": True, "evidence_score": 0.9, "context": {"tissue": "Liver"}}],
    }
    consensus_rows = rank_consensus_by_conflict_aware_confidence_v02(
        candidates=[p_conflicted, p_clean],
        context_candidates=[{"context_id": "C1", "context": {"tissue": "Blood"}}],
    )
    filtered = filter_consensus_by_thresholds_v02(
        consensus_rows=consensus_rows,
        max_conflict_rate=0.0,
        min_zero_conflict_contexts=1,
        include_rejected=True,
    )
    accepted = [r for r in filtered if r.get("accepted")]
    rejected = [r for r in filtered if not r.get("accepted")]
    assert [r["path_id"] for r in accepted] == ["P_CLEAN"]
    assert accepted[0]["rank"] == 1
    assert [r["path_id"] for r in rejected] == ["P_CONFLICTED"]
    assert "above_max_conflict_rate" in rejected[0]["rejection_reasons"]

    disp = compute_conflict_aware_confidence_dispersion_v02(
        path_id="P_CLEAN",
        edges=p_clean["edges"],
        context_candidates=[
            {"context_id": "C1", "context": {"tissue": "Blood"}},
            {"context_id": "C2", "context": {"tissue": "Liver"}},
        ],
    )
    assert disp["contexts_evaluated"] == 2
    assert disp["max_adjusted_confidence"] >= disp["min_adjusted_confidence"]
    assert 0.0 <= disp["adjusted_confidence_consistency"] <= 1.0


def test_conflict_intensity_and_conflict_aware_confidence_consensus():
    edges = [
        {"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9},
        {"id": "E2", "subject": "IL6", "object": "CRP", "conflict": True, "evidence_score": 0.4},
    ]
    intensity = compute_path_conflict_intensity_v02(path_id="P_CI", edges=edges)
    assert intensity["edge_count"] == 2
    assert intensity["conflict_edge_count"] == 1
    assert intensity["conflict_edge_ratio"] == 0.5
    assert intensity["conflict_resilience_index"] == 0.5
    assert 0.0 <= intensity["conflicted_support_ratio"] <= 1.0

    p_clean = {
        "path_id": "P_CLEAN",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9, "context": {"tissue": "Blood"}}],
    }
    p_conflicted = {
        "path_id": "P_CONFLICTED",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": True, "evidence_score": 0.9, "context": {"tissue": "Blood"}}],
    }
    ranked = rank_consensus_by_conflict_aware_confidence_v02(
        candidates=[p_conflicted, p_clean],
        context_candidates=[{"context_id": "C1", "context": {"tissue": "Blood"}}],
    )
    assert [r["path_id"] for r in ranked] == ["P_CLEAN", "P_CONFLICTED"]
    assert ranked[0]["conflict_aware_confidence_score"] > ranked[1]["conflict_aware_confidence_score"]


def test_confidence_dispersion_and_confidence_weighted_consensus():
    edges = [
        {"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9, "context": {"tissue": "Blood"}},
    ]
    disp = compute_path_confidence_dispersion_v02(
        path_id="P_CONF",
        edges=edges,
        context_candidates=[
            {"context_id": "C1", "context": {"tissue": "Blood"}},
            {"context_id": "C2", "context": {"tissue": "Liver"}},
        ],
    )
    assert disp["contexts_evaluated"] == 2
    assert disp["max_confidence"] >= disp["min_confidence"]
    assert round(disp["max_confidence"] - disp["min_confidence"], 4) == disp["confidence_range"]
    assert 0.0 <= disp["confidence_consistency"] <= 1.0

    p_high = {
        "path_id": "P_HIGH",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9, "context": {"tissue": "Blood"}}],
    }
    p_low = {
        "path_id": "P_LOW",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": True, "evidence_score": 0.9, "context": {"tissue": "Liver"}}],
    }
    cw_ranked = rank_consensus_by_confidence_v02(
        candidates=[p_low, p_high],
        context_candidates=[{"context_id": "C1", "context": {"tissue": "Blood"}}],
    )
    assert [r["path_id"] for r in cw_ranked] == ["P_HIGH", "P_LOW"]
    assert cw_ranked[0]["confidence_weighted_mean_score"] > cw_ranked[1]["confidence_weighted_mean_score"]


def test_evidence_coverage_and_conflict_penalized_consensus_ranking():
    edges = [
        {"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.8, "context": {"tissue": "Blood"}},
        {"id": "E2", "subject": "IL6", "object": "CRP", "conflict": False, "evidence_score": 0.6, "context": {"tissue": "Blood"}},
    ]
    cov = compute_path_evidence_coverage_v02(path_id="P_EV", edges=edges)
    assert cov["edge_count"] == 2
    assert cov["conflict_free_ratio"] == 1.0
    assert cov["supported_ratio"] == 1.0
    assert cov["clean_support_index"] == 1.0

    p_clean = {
        "path_id": "P_CLEAN",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "evidence_score": 0.9, "context": {"tissue": "Blood"}}],
    }
    p_conflicted = {
        "path_id": "P_CONFLICTED",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": True, "evidence_score": 0.9, "context": {"tissue": "Blood"}}],
    }
    pen_ranked = rank_consensus_with_conflict_penalty_v02(
        candidates=[p_conflicted, p_clean],
        context_candidates=[{"context_id": "C1", "context": {"tissue": "Blood"}}],
        conflict_penalty_weight=0.50,
    )
    assert [r["path_id"] for r in pen_ranked] == ["P_CLEAN", "P_CONFLICTED"]
    assert pen_ranked[0]["conflict_adjusted_mean_score"] > pen_ranked[1]["conflict_adjusted_mean_score"]
    assert pen_ranked[1]["conflict_rate"] == 1.0


def test_sign_polarity_coherence_and_consensus_score_stability_ranking():
    edges = [
        {"id": "E1", "subject": "TNF", "object": "IL6", "sign": "+", "conflict": False},
        {"id": "E2", "subject": "IL6", "object": "CRP", "sign": "+", "conflict": False},
    ]
    pol = compute_path_sign_polarity_coherence_v02(path_id="P_POS", edges=edges)
    assert pol["edge_count"] == 2
    assert pol["positive_ratio"] == 1.0
    assert pol["negative_ratio"] == 0.0
    assert pol["net_polarity_bias"] == 1.0
    assert pol["polarity_coherence_score"] == 1.0

    p_stable = {
        "path_id": "P_STABLE",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}}],
    }
    p_variable = {
        "path_id": "P_VAR",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood", "cell_type": "Macrophage"}}],
    }
    stab_ranked = rank_consensus_by_score_stability_v02(
        candidates=[p_variable, p_stable],
        context_candidates=[
            {"context_id": "C1", "context": {"tissue": "Blood", "cell_type": "Macrophage"}},
            {"context_id": "C2", "context": {"tissue": "Liver", "cell_type": "Hepatocyte"}},
        ],
    )
    assert [r["path_id"] for r in stab_ranked] == ["P_VAR", "P_STABLE"]
    assert stab_ranked[0]["score_spread"] > stab_ranked[1]["score_spread"]
    assert stab_ranked[1]["score_stability_index"] > stab_ranked[0]["score_stability_index"]


def test_conflict_free_consensus_ranking():
    clean_p = {
        "path_id": "P_CLEAN",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}}],
    }
    conflict_p = {
        "path_id": "P_CONFLICT",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": True, "context": {"tissue": "Blood"}}],
    }
    ranked_clean = rank_conflict_free_consensus_v02(
        candidates=[conflict_p, clean_p],
        context_candidates=[{"context_id": "C1", "context": {"tissue": "Blood"}}],
    )
    assert [r["path_id"] for r in ranked_clean] == ["P_CLEAN"]
    assert ranked_clean[0]["rank"] == 1


def test_context_rank_stability():
    p_a = {
        "path_id": "P_A",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}}],
    }
    p_b = {
        "path_id": "P_B",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Liver"}}],
    }

    stab = compute_context_rank_stability_v02(
        candidates=[p_a, p_b],
        baseline_context={"tissue": "Blood"},
        perturbed_contexts=[{"tissue": "Blood"}, {"tissue": "Liver"}],
    )
    by_id = {row["path_id"]: row for row in stab}
    assert by_id["P_A"]["perturbed_contexts_evaluated"] == 2
    assert by_id["P_A"]["unchanged_rank_count"] == 1
    assert by_id["P_A"]["rank_retention_rate"] == 0.5
    assert by_id["P_A"]["mean_rank_deviation"] == 0.5


def test_context_coverage_dispersion():
    edges = [
        {"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood", "cell_type": "Macrophage"}},
    ]
    disp = compute_context_coverage_dispersion_v02(
        path_id="P_DISP",
        edges=edges,
        context_candidates=[
            {"context_id": "C1", "context": {"tissue": "Blood", "cell_type": "Macrophage"}},
            {"context_id": "C2", "context": {"tissue": "Blood", "cell_type": "T Cell"}},
        ],
    )
    assert disp["contexts_evaluated"] == 2
    assert disp["min_coverage"] == 0.5
    assert disp["max_coverage"] == 1.0
    assert disp["coverage_range"] == 0.5
    assert disp["coverage_uniformity"] == 0.5


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


def test_context_alignment_and_counter_sign_weighting_v02():
    from bioos.engine.path_ranking import rank_paths_by_context_and_counter_sign_v02, summarize_context_counter_sign_cohort_v02
    aligned_clean = {
        "path_id": "P_ALIGNED_CLEAN",
        "edges": [
            {"id": "E1", "subject": "TNF", "object": "IL6", "sign": "+", "context": {"tissue": "Blood"}},
            {"id": "E2", "subject": "IL6", "object": "STAT3", "sign": "+", "context": {"tissue": "Blood"}},
        ],
    }
    counter_sign_path = {
        "path_id": "P_COUNTER_SIGN",
        "edges": [
            {"id": "E3", "subject": "TNF", "object": "IL6", "sign": "+", "context": {"tissue": "Blood"}},
            {"id": "E4", "subject": "IL6", "object": "STAT3", "sign": "-", "context": {"tissue": "Blood"}},
        ],
    }
    mismatched_path = {
        "path_id": "P_MISMATCHED",
        "edges": [
            {"id": "E5", "subject": "TNF", "object": "IL6", "sign": "+", "context": {"tissue": "Liver"}},
        ],
    }

    ranked = rank_paths_by_context_and_counter_sign_v02(
        candidates=[counter_sign_path, mismatched_path, aligned_clean],
        target_context={"tissue": "Blood"},
        expected_sign="+",
    )

    assert [r["path_id"] for r in ranked] == [
        "P_ALIGNED_CLEAN",
        "P_COUNTER_SIGN",
        "P_MISMATCHED",
    ]
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    assert ranked[0]["context_alignment_score"] == 1.0
    assert ranked[0]["counter_sign_count"] == 0
    assert ranked[1]["counter_sign_count"] == 1
    assert ranked[1]["adjusted_net_score"] < ranked[1]["base_net_score"]

    summary = summarize_context_counter_sign_cohort_v02(ranked)
    assert summary["candidate_count"] == 3
    assert summary["total_counter_sign_edges"] == 1
    assert summary["counter_sign_free_count"] == 2
