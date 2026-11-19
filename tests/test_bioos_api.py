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
    assert set(body["nodes"]) == {"IL6", "STAT3", "CRP"}
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
