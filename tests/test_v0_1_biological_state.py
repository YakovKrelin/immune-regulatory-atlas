from __future__ import annotations

from fastapi.testclient import TestClient

from bioos.main import _evaluate_dataset_payload, app


client = TestClient(app)


def test_biological_state_coverage_metrics() -> None:
    result = _evaluate_dataset_payload(
        name="hypoxia_coverage_run",
        modality="RNA-seq",
        tissue="Lung",
        cell_type="Macrophage",
        signals=["HIF1A UP", "VEGFA UP", "BNIP3 UP"],
    )
    states_by_name = {item["name"]: item for item in result["active_states"]}
    assert "Hypoxia" in states_by_name

    hypoxia_state = states_by_name["Hypoxia"]
    assert hypoxia_state["anchor_hits"] == ["HIF1A UP", "VEGFA UP"]
    assert hypoxia_state["supporting_hits"] == ["BNIP3 UP"]
    assert hypoxia_state["counter_sign_hits"] == []
    assert hypoxia_state["counter_sign_penalty"] == 0.0
    assert hypoxia_state["coverage_score"] > 0.0
    assert hypoxia_state["coverage_breakdown"]["anchors_matched"] == 2
    assert hypoxia_state["coverage_breakdown"]["supporting_matched"] == 1


def test_biological_state_counter_signs_reduce_score_via_api() -> None:
    clean_resp = client.post(
        "/v1/datasets/evaluate",
        json={
            "name": "clean_inflammation",
            "modality": "proteomics",
            "tissue": "Blood",
            "cell_type": "Monocyte",
            "signals": ["TNF UP", "IL6 UP"],
        },
    )
    assert clean_resp.status_code == 200
    clean_infl = {s["name"]: s for s in clean_resp.json()["active_states"]}["Acute Inflammation"]

    opposed_resp = client.post(
        "/v1/datasets/evaluate",
        json={
            "name": "opposed_inflammation",
            "modality": "proteomics",
            "tissue": "Blood",
            "cell_type": "Monocyte",
            "signals": ["TNF UP", "IL6 UP", "IL10 UP", "TGFB1 UP"],
        },
    )
    assert opposed_resp.status_code == 200
    opposed_infl = {s["name"]: s for s in opposed_resp.json()["active_states"]}["Acute Inflammation"]

    assert opposed_infl["counter_sign_hits"] == ["IL10 UP", "TGFB1 UP"]
    assert opposed_infl["counter_sign_penalty"] == 0.3
    assert opposed_infl["score"] < clean_infl["score"]
    assert opposed_infl["raw_anchor_score"] == clean_infl["raw_anchor_score"]