from fastapi import FastAPI
from fastapi.testclient import TestClient

from bioos.engine.path_ranking import rank_paths_with_context_v02, score_candidate_path_v02
from bioos.main import app as _v02_target


def test_score_candidate_path_applies_conflict_context_and_counter_adjustments():
    clean_edges = [
        {
            "id": "BIOOS:A000001",
            "subject": "TNF",
            "object": "IL6",
            "conflict": False,
            "context": {"tissue": "Blood", "cell_type": "Macrophage", "eco": "A"},
        },
        {
            "id": "BIOOS:A000002",
            "subject": "IL6",
            "object": "CRP",
            "conflict": False,
            "context": {"tissue": "Liver", "cell_type": "Hepatocyte", "eco": "A"},
        },
    ]
    conflicting_edges = [
        {
            "id": "BIOOS:A000004",
            "subject": "HIF1A",
            "object": "IFNG",
            "conflict": True,
            "context": {"tissue": "Tumor Microenvironment", "cell_type": "T Cell", "eco": "F"},
        }
    ]

    clean_res = score_candidate_path_v02(
        path_id="P_CLEAN",
        edges=clean_edges,
        active_markers=["TNF", "IL6", "CRP"],
        anchor_markers=["TNF", "IL6", "CRP"],
        counter_markers=["IL10"],
        target_context={"tissue": "Blood", "cell_type": "Macrophage", "eco": "A"},
    )
    conflict_res = score_candidate_path_v02(
        path_id="P_CONFLICT",
        edges=conflicting_edges,
        active_markers=["HIF1A", "IFNG", "IL10"],
        anchor_markers=["TNF", "IL6", "CRP"],
        counter_markers=["IL10"],
        target_context={"tissue": "Blood", "cell_type": "Macrophage", "eco": "A"},
    )

    assert clean_res["conflict_count"] == 0
    assert clean_res["context_bonus"] > 0.0
    assert clean_res["anchor_bonus"] > 0.0
    assert clean_res["counter_penalty"] == 0.0

    assert conflict_res["conflict_count"] == 1
    assert conflict_res["conflict_penalty"] > 0.0
    assert conflict_res["counter_penalty"] > 0.0
    assert clean_res["net_score"] > conflict_res["net_score"]


def test_rank_paths_with_context_is_deterministic():
    candidates = [
        {
            "path_id": "P_B",
            "edges": [{"id": "E2", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}}],
        },
        {
            "path_id": "P_A",
            "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False, "context": {"tissue": "Blood"}}],
        },
        {
            "path_id": "P_CONF",
            "edges": [{"id": "E3", "subject": "HIF1A", "object": "IFNG", "conflict": True, "context": {"tissue": "Blood"}}],
        },
    ]

    ranked = rank_paths_with_context_v02(
        candidates=candidates,
        active_markers=["TNF", "IL6"],
        anchor_markers=["TNF", "IL6"],
        counter_markers=[],
        target_context={"tissue": "Blood"},
    )

    assert [r["path_id"] for r in ranked] == ["P_A", "P_B", "P_CONF"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]


def test_v02_paths_rank_endpoint_integration():
    if hasattr(_v02_target, "include_router"):
        client = TestClient(_v02_target)
    else:
        app = FastAPI()
        app.include_router(_v02_target)
        client = TestClient(app)

    response = client.post(
        "/v0.2/paths/rank",
        json={
            "candidates": [
                {
                    "path_id": "P_CONFLICT",
                    "edges": [{"id": "E2", "subject": "HIF1A", "object": "IFNG", "conflict": True}],
                },
                {
                    "path_id": "P_CLEAN",
                    "edges": [{"id": "E1", "subject": "TNF", "object": "IL6", "conflict": False}],
                },
            ],
            "active_markers": ["TNF", "IL6"],
            "anchor_markers": ["TNF", "IL6"],
            "target_context": {"tissue": "Blood"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert [row["path_id"] for row in data["ranked_paths"]] == ["P_CLEAN", "P_CONFLICT"]
