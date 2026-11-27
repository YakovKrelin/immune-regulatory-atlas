from fastapi import FastAPI
from fastapi.testclient import TestClient

from hypothesis_engine_v0_1.scoring_api import router


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_evaluate_endpoint_with_policy_and_observations() -> None:
    client = _build_client()
    response = client.post(
        "/scoring/evaluate",
        json={
            "observations": [
                {
                    "observation_id": "obs-1",
                    "direction": "support",
                    "confidence": 0.9,
                    "source_type": "assay",
                },
                {
                    "observation_id": "obs-2",
                    "direction": "contradict",
                    "confidence": 0.2,
                    "source_type": "literature",
                },
            ],
            "policy": {
                "source_weights": {"assay": 2.0},
                "default_source_weight": 1.0,
                "conflict_penalty_factor": 0.5,
                "min_support_threshold": 0.4,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["passed_threshold"] is True
    assert "weighted_aggregation" in data["triggered_rules"]
    assert len(data["contributions"]) == 2


def test_evaluate_endpoint_defaults_on_empty_payload() -> None:
    client = _build_client()
    response = client.post("/scoring/evaluate", json={})

    assert response.status_code == 200
    data = response.json()
    assert data["net_score"] == 0.0
    assert data["passed_threshold"] is False
    assert data["triggered_rules"] == ["empty_observations"]


def test_evaluate_endpoint_rejects_invalid_observation() -> None:
    client = _build_client()
    response = client.post(
        "/scoring/evaluate",
        json={
            "observations": [
                {
                    "observation_id": "bad-obs",
                    "direction": "support",
                    "confidence": 1.7,
                    "source_type": "assay",
                }
            ]
        },
    )

    assert response.status_code == 422
