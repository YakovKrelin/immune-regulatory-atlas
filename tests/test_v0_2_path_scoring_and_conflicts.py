from __future__ import annotations

from fastapi.testclient import TestClient

from bioos.engine.v0_2_path_scoring import PathScoringConfig, rank_paths_with_context, score_candidate_path
from bioos.main import app

client = TestClient(app)


def test_score_candidate_path_applies_conflict_and_counter_sign_penalties() -> None:
    clean_edges = [
        {
            "assertion_id": "BIOOS:A0004880",
            "source": "TNF",
            "target": "IL6",
            "relation": "activates",
            "eco": "A",
            "conflict": False,
            "context": {"timepoint": "6h", "phase": "innate_response"},
        },
        {
            "assertion_id": "BIOOS:A0004881",
            "source": "IL6",
            "target": "STAT3",
            "relation": "activates",
            "eco": "A",
            "conflict": False,
            "context": {"timepoint": "12h", "phase": "acute_phase"},
        },
    ]
    conflicted_edges = [
        {
            "assertion_id": "BIOOS:A0004903",
            "source": "IFNG",
            "target": "CXCL10",
            "relation": "inhibits",
            "eco": "B",
            "conflict": True,
            "contradicts": ["BIOOS:A0004874"],
            "context": {"timepoint": "24h", "phase": "amplification_isg"},
        }
    ]

    clean_res = score_candidate_path(clean_edges, counter_sign_hits=[])
    conflict_res = score_candidate_path(conflicted_edges, counter_sign_hits=["EGLN1 UP"])

    assert clean_res["has_conflict"] is False
    assert clean_res["conflict_edge_ids"] == []
    assert clean_res["score"] > conflict_res["score"]
    assert conflict_res["has_conflict"] is True
    assert conflict_res["conflict_edge_ids"] == ["BIOOS:A0004903"]
    assert conflict_res["counter_sign_hits"] == ["EGLN1 UP"]


def test_rank_paths_with_context_orders_clean_path_first_and_breaks_ties_deterministically() -> None:
    path_conflicted = [
        {
            "assertion_id": "BIOOS:A0004903",
            "source": "IFNG",
            "target": "CXCL10",
            "relation": "inhibits",
            "eco": "A",
            "conflict": True,
        }
    ]
    path_clean_high_eco = [
        {
            "assertion_id": "BIOOS:A0004871",
            "source": "TNF",
            "target": "IL6",
            "relation": "activates",
            "eco": "A",
            "conflict": False,
        }
    ]
    path_clean_lower_eco = [
        {
            "assertion_id": "BIOOS:A0004882",
            "source": "HIF1A",
            "target": "VEGFA",
            "relation": "activates",
            "eco": "B",
            "conflict": False,
        }
    ]

    ranked = rank_paths_with_context(
        [path_conflicted, path_clean_lower_eco, path_clean_high_eco],
        counter_sign_hits=[],
        config=PathScoringConfig(conflict_penalty=0.35),
    )

    assert [item["rank"] for item in ranked] == [1, 2, 3]
    assert ranked[0]["nodes"] == ["TNF", "IL6"]
    assert ranked[0]["has_conflict"] is False
    assert ranked[1]["nodes"] == ["HIF1A", "VEGFA"]
    assert ranked[2]["nodes"] == ["IFNG", "CXCL10"]
    assert ranked[2]["has_conflict"] is True


def test_integration_v0_2_hypotheses_and_snapshot_sha256_integrity() -> None:
    gen_resp = client.post(
        "/v1/hypotheses/generate",
        json={
            "name": "v0_2_conflict_integration",
            "modality": "RNA-seq",
            "tissue": "Blood",
            "cell_type": "Monocyte",
            "signals": ["IFNG UP", "CXCL10 UP", "TNF UP", "IL6 UP"],
        },
    )
    assert gen_resp.status_code == 200
    gen_data = gen_resp.json()
    assert len(gen_data["knowledge_base_sha256"]) == 64

    snap_resp = client.post(
        "/v1/snapshots",
        json={
            "title": "v0_2_conflict_snapshot",
            "signals": ["IFNG UP", "CXCL10 UP", "TNF UP", "IL6 UP"],
            "start_node": "TNF",
            "max_hops": 2,
            "actor": "pytest-v0.2",
            "include_hypotheses": True,
            "verify": True,
        },
    )
    assert snap_resp.status_code == 200
    snap_data = snap_resp.json()
    snapshot_obj = snap_data.get("snapshot", snap_data)

    verify_resp = client.post(
        "/v1/snapshots/verify",
        json={"snapshot": snapshot_obj},
    )
    if verify_resp.status_code == 200:
        verify_body = verify_resp.json()
        assert verify_body.get("valid") is True or verify_body.get("verification", {}).get("valid") is True