from bioos.engine.path_ranking import (
    attach_snapshot_provenance,
    rank_contexts_for_path_v02,
    rank_paths_with_context_v02,
    score_candidate_path_v02,
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


def test_sign_consistency_and_context_ranking():
    edges = [
        {"id": "E1", "subject": "TNF", "object": "IL6", "sign": "+", "conflict": False, "context": {"tissue": "Blood"}},
        {"id": "E2", "subject": "IL6", "object": "CRP", "sign": "+", "conflict": False, "context": {"tissue": "Blood"}},
    ]
    res = score_candidate_path_v02(path_id="P_SIGN", edges=edges)
    assert res["sign_consistency_score"] == 1.0
    assert res["sign_bonus"] > 0.0

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


def test_snapshot_provenance_envelope():
    sample = [{"path_id": "P1", "rank": 1, "net_score": 1.5}]
    env = attach_snapshot_provenance(sample, snapshot_id="snap_01", sha256="abc123")
    assert env["count"] == 1
    assert env["snapshot"] == {"snapshot_id": "snap_01", "sha256": "abc123"}
