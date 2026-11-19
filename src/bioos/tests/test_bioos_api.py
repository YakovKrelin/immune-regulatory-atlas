from __future__ import annotations

import copy
import json
from pathlib import Path

from fastapi.testclient import TestClient

from bioos.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["assertions_count"] == 5
    assert len(body["knowledge_base_sha256"]) == 64


def test_filtered_assertions_blood_eco_a():
    response = client.get(
        "/v1/assertions",
        params={"tissue": "Blood", "eco_levels": ["A"]},
    )
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 2
    ids = {item["id"] for item in items}
    assert ids == {"BIOOS:A0004871", "BIOOS:A0004874"}


def test_filtered_graph_traversal_liver_from_tnf():
    response = client.get(
        "/v1/graph",
        params={
            "start": "TNF",
            "max_hops": 2,
            "min_eco": "A",
            "tissue": "Liver",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["start"] == "TNF"
    assert body["max_hops"] == 2
    assert set(body["nodes"]) == {"IL6", "STAT3", "CRP"}
    assert len(body["edges"]) == 2


def test_dataset_evaluation_scoring():
    payload = {
        "name": "pbmc_viral_pilot",
        "modality": "RNA-seq",
        "tissue": "Blood",
        "cell_type": "Monocyte",
        "signals": "TNF up, IL6 up, IFNG up, CXCL10 up",
    }
    response = client.post("/v1/datasets/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_name"] == "pbmc_viral_pilot"

    states_by_name = {s["name"]: s for s in body["active_states"]}
    assert "Acute Inflammation" in states_by_name
    assert "Interferon State" in states_by_name
    assert states_by_name["Acute Inflammation"]["score"] == 0.5
    assert states_by_name["Interferon State"]["score"] == 0.5
    assert len(body["knowledge_base_sha256"]) == 64


def test_snapshot_creation_and_sha256_verification(tmp_path: Path):
    snap_file = tmp_path / "snapshot_check.json"
    req_payload = {
        "title": "Pytest Snapshot Verification",
        "signals": "TNF up, IL6 up, IFNG up, CXCL10 up",
        "start_node": "TNF",
        "max_hops": 2,
        "actor": "pytest-runner",
        "tissue": "Blood",
        "eco_levels": ["A"],
        "verify": True,
    }
    create_resp = client.post("/v1/snapshots", json=req_payload)
    assert create_resp.status_code == 200
    created = create_resp.json()

    assert created["verification"]["valid"] is True
    assert created["verification"]["payload_sha256_valid"] is True
    assert created["verification"]["knowledge_base_unchanged"] is True

    snap_record = created["snapshot"]
    assert snap_record["snapshot_id"].startswith("snap_")
    assert len(snap_record["canonical_payload_sha256"]) == 64
    assert len(snap_record["knowledge_base_sha256"]) == 64

    snap_file.write_text(json.dumps(created, indent=2), encoding="utf-8")
    verify_file_resp = client.post(
        "/v1/snapshots/verify",
        json={"file_path": str(snap_file)},
    )
    assert verify_file_resp.status_code == 200
    assert verify_file_resp.json()["valid"] is True

    tampered = copy.deepcopy(snap_record)
    tampered["payload"]["title"] = "Tampered Snapshot Title"
    verify_tampered_resp = client.post(
        "/v1/snapshots/verify",
        json={"snapshot": tampered},
    )
    assert verify_tampered_resp.status_code == 200
    assert verify_tampered_resp.json()["valid"] is False
    assert verify_tampered_resp.json()["payload_sha256_valid"] is False