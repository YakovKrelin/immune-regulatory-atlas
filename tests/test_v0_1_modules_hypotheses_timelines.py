from __future__ import annotations

from fastapi.testclient import TestClient

from bioos.main import app

client = TestClient(app)


def test_modules_include_temporal_and_translational_metadata() -> None:
    resp = client.get("/v1/modules", params={"axis": "Hypoxia"})
    assert resp.status_code == 200
    mods = resp.json()
    assert len(mods) == 1
    hypoxia_mod = mods[0]
    assert "6h" in hypoxia_mod["timepoints"]
    assert "24h" in hypoxia_mod["timepoints"]
    assert "innate_response" in hypoxia_mod["phases"]
    assert "primary_human_cells" in hypoxia_mod["translational_statuses"]


def test_hypotheses_include_coverage_counter_signs_and_timeline_window() -> None:
    resp = client.post(
        "/v1/hypotheses/generate",
        json={
            "name": "hypoxia_timeline_check",
            "modality": "RNA-seq",
            "tissue": "Lung",
            "cell_type": "Macrophage",
            "signals": ["HIF1A UP", "VEGFA UP", "EGLN1 UP"],
            "axes": ["Hypoxia"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["hypotheses"]) == 1
    hyp = data["hypotheses"][0]
    assert hyp["hypothesis_id"] == "HYP:V0_1:HYPOXIA"
    assert hyp["counter_sign_hits"] == ["EGLN1 UP"]
    assert hyp["counter_sign_penalty"] == 0.15
    assert hyp["coverage_score"] > 0.0
    assert "timeline_window" in hyp
    assert "6h" in hyp["timeline_window"]["observed_timepoints"]


def test_timelines_endpoint_filters_by_axis_and_timepoint() -> None:
    resp = client.get("/v1/timelines", params={"axis": "Interferon", "timepoint": "12h"})
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["timelines"]) == 1
    tl = payload["timelines"][0]
    assert tl["axis"] == "Interferon"
    assert len(tl["events"]) == 1
    assert tl["events"][0]["assertion_id"] == "BIOOS:A0004902"