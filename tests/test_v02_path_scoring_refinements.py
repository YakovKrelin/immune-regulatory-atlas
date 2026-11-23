from bioos.engine.path_ranking import (
    attach_snapshot_provenance,
    build_ranked_hypothesis_payload_v02,
    compare_path_scores_v02,
    filter_and_rank_paths_v02,
    group_ranked_paths_by_context_v02,
    rank_contexts_for_path_v02,
    rank_paths_across_contexts_v02,
    rank_paths_by_consensus_v02,
    rank_paths_with_context_v02,
    score_candidate_path_v02,
    select_top_paths_v02,
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

    grouped = group_ranked_paths_by_context_v02(ranked)
    assert sorted(grouped.keys()) == ["CTX_BLOOD", "CTX_LIVER"]
    assert grouped["CTX_BLOOD"][0]["context_rank"] == 1

    s_clean = score_candidate_path_v02("P_CLEAN", path_clean["edges"], counter_markers=["IL10"])
    s_counter = score_candidate_path_v02("P_COUNTER", path_counter["edges"], counter_markers=["IL10"])
    diff = compare_path_scores_v02(s_clean, s_counter)
    assert diff["delta_net_score"] > 0.0
    assert diff["delta_counter_penalty"] < 0.0


def test_filter_and_rank_paths_v02():
    clean_p = {
        "path_id": "P_CLEAN",
        "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}}],
    }
    conflicted_p = {
        "path_id": "P_CONFLICT",
        "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": True, "context": {"tissue": "Blood"}}],
    }

    surviving = filter_and_rank_paths_v02(
        candidates=[conflicted_p, clean_p],
        target_context={"tissue": "Blood"},
        max_conflicts=0,
        min_normalized_confidence=0.5,
    )
    assert [r["path_id"] for r in surviving] == ["P_CLEAN"]
    assert surviving[0]["rank"] == 1


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
