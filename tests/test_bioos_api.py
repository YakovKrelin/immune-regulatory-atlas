from __future__ import annotations
import copy
import json
from pathlib import Path
from fastapi.testclient import TestClient
from bioos.main import app

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["assertions_count"] == 5
    assert len(body["knowledge_base_sha256"]) == 64

def test_filtered_assertions_blood_eco_a():
    r = client.get("/v1/assertions", params={"tissue": "Blood", "eco_levels": ["A"]})
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 2
    assert {x["id"] for x in items} == {"BIOOS:A0004871", "BIOOS:A0004874"}

def test_filtered_graph_traversal_liver_from_tnf():
    r = client.get("/v1/graph", params={"start": "TNF", "max_hops": 2, "min_eco": "A", "tissue": "Liver"})
    assert r.status_code == 200
    body = r.json()
    assert set(body["nodes"]) == {"IL6", "STAT3"}
    assert len(body["edges"]) == 2

def test_dataset_evaluation_scoring():
    payload = {"name": "pbmc_viral_pilot", "modality": "RNA-seq", "tissue": "Blood", "cell_type": "Monocyte", "signals": "TNF up, IL6 up, IFNG up, CXCL10 up"}
    r = client.post("/v1/datasets/evaluate", json=payload)
    assert r.status_code == 200
    states = {s["name"]: s for s in r.json()["active_states"]}
    assert states["Acute Inflammation"]["score"] == 0.5
    assert states["Interferon State"]["score"] == 0.5

def test_snapshot_creation_and_sha256_verification(tmp_path: Path):
    snap_file = tmp_path / "snapshot_check.json"
    req = {"title": "Pytest Snapshot Verification", "signals": "TNF up, IL6 up, IFNG up, CXCL10 up", "start_node": "TNF", "max_hops": 2, "actor": "pytest-runner", "tissue": "Blood", "eco_levels": ["A"], "verify": True}
    created = client.post("/v1/snapshots", json=req).json()
    assert created["verification"]["valid"] is True
    snap_file.write_text(json.dumps(created, indent=2), encoding="utf-8")
    assert client.post("/v1/snapshots/verify", json={"file_path": str(snap_file)}).json()["valid"] is True
    tampered = copy.deepcopy(created["snapshot"])
    tampered["payload"]["title"] = "Tampered"
    assert client.post("/v1/snapshots/verify", json={"snapshot": tampered}).json()["valid"] is False


def test_dataset_evaluation_hypoxia_and_warnings() -> None:
    resp = client.post(
        "/v1/datasets/evaluate",
        json={
            "name": "hypoxia_check",
            "modality": "RNA-seq",
            "tissue": "Lung",
            "cell_type": "Macrophage",
            "signals": ["HIF1A UP", "VEGFA UP", "IL10 UP"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    names = [s["name"] for s in body["active_states"]]
    assert "Hypoxia" in names
    assert body["modality_warnings"]
    assert "not a clinical diagnosis" in body["disclaimer"].lower()


def test_hypoxia_graph_and_conflict_filtering() -> None:
    from bioos.engine.filters import filter_assertions
    from bioos.main import V0_1_ASSERTIONS

    lung_hypoxia = filter_assertions(V0_1_ASSERTIONS, tissue="Lung")
    assert any(item["id"] == "BIOOS:A0004901" and item["target"] == "VEGFA" for item in lung_hypoxia)

    conflicts = filter_assertions(V0_1_ASSERTIONS, conflict_only=True)
    assert [item["id"] for item in conflicts] == ["BIOOS:A0004903"]


def test_v0_1_modules_endpoints() -> None:
    resp = client.get("/v1/modules")
    assert resp.status_code == 200
    modules = resp.json()
    axes = {m["axis"] for m in modules}
    assert {"Inflammation", "Interferon", "Hypoxia"}.issubset(axes)

    hypoxia_resp = client.get("/v1/modules/MOD:V0_1:HYPOXIA")
    assert hypoxia_resp.status_code == 200
    hypoxia_mod = hypoxia_resp.json()
    assert "HIF1A" in hypoxia_mod["nodes"]
    assert "BIOOS:A0004903" in hypoxia_mod["conflict_assertion_ids"]

    missing_resp = client.get("/v1/modules/MOD:UNKNOWN")
    assert missing_resp.status_code == 404


def test_v0_1_hypotheses_generation() -> None:
    resp = client.post(
        "/v1/hypotheses/generate",
        json={
            "name": "tri_axis_check",
            "modality": "RNA-seq",
            "tissue": "Lung",
            "cell_type": "Macrophage",
            "signals": ["TNF UP", "IFNG UP", "HIF1A UP", "VEGFA UP"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    axes = {h["axis"] for h in body["hypotheses"]}
    assert {"Inflammation", "Interferon", "Hypoxia"}.issubset(axes)
    hypoxia_hyp = next(h for h in body["hypotheses"] if h["axis"] == "Hypoxia")
    assert "BIOOS:A0004903" in hypoxia_hyp["conflict_assertion_ids"]
    assert body["modality_warnings"]
    assert body["disclaimer"]


def test_snapshot_with_hypotheses_sha256_integrity() -> None:
    create_resp = client.post(
        "/v1/snapshots",
        json={
            "title": "v0_1_hypotheses_snapshot",
            "signals": ["TNF UP", "IFNG UP", "HIF1A UP", "VEGFA UP"],
            "tissue": "Lung",
            "cell_type": "Macrophage",
            "verify": True,
            "include_hypotheses": True,
            "hypotheses_request": {
                "name": "snap_hyp_run",
                "modality": "RNA-seq",
                "tissue": "Lung",
                "cell_type": "Macrophage",
                "signals": ["TNF UP", "IFNG UP", "HIF1A UP", "VEGFA UP"],
            },
        },
    )
    assert create_resp.status_code in (200, 201)
    created = create_resp.json()
    assert created["verification"]["valid"] is True

    snap = created["snapshot"]
    assert snap["payload"].get("hypotheses") is not None
    assert snap["payload"]["hypotheses"]["hypotheses"]

    verify_resp = client.post("/v1/snapshots/verify", json={"snapshot": snap})
    assert verify_resp.status_code == 200
    assert verify_resp.json()["valid"] is True

    tampered = copy.deepcopy(snap)
    tampered["payload"]["hypotheses"]["hypotheses"][0]["statement"] = "TAMPERED_STATEMENT"
    tampered_resp = client.post("/v1/snapshots/verify", json={"snapshot": tampered})
    assert tampered_resp.status_code == 200
    assert tampered_resp.json()["valid"] is False
