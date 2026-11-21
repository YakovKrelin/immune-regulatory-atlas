from __future__ import annotations

from bioos.graph_hypotheses import (
    RegulatoryGraphSnapshot,
    build_graph_hypotheses,
    compute_snapshot_sha256,
    create_regulatory_graph_snapshot,
)
from bioos.regulatory_graph import (
    ConflictPolicy,
    RegulationDirection,
    RegulatoryEdge,
    SubgraphQuery,
    build_regulatory_graph_result,
)


def _edges_for_hypotheses() -> tuple[RegulatoryEdge, ...]:
    return (
        RegulatoryEdge(
            edge_id="e1",
            source="TNF",
            target="NFKB1",
            direction=RegulationDirection.ACTIVATION,
            tissue="liver",
            context="hypoxia",
            evidence_codes=("ECO:0000314",),
            confidence_score=0.90,
        ),
        RegulatoryEdge(
            edge_id="e2",
            source="NFKB1",
            target="HIF1A",
            direction=RegulationDirection.ACTIVATION,
            tissue="liver",
            context="hypoxia",
            evidence_codes=("ECO:0000314",),
            confidence_score=0.80,
        ),
    )


def test_build_hypotheses_and_snapshot_sha256() -> None:
    query = SubgraphQuery(
        seed_entities=("TNF",),
        max_hops=2,
        tissue="liver",
        context="hypoxia",
        min_confidence_score=0.5,
        conflict_policy=ConflictPolicy.KEEP_FLAGGED,
    )
    graph_result = build_regulatory_graph_result(_edges_for_hypotheses(), query, targets=("HIF1A",))

    hypotheses = build_graph_hypotheses(graph_result, max_hypotheses=3)
    assert len(hypotheses) == 1
    assert hypotheses[0].chain == ("TNF", "NFKB1", "HIF1A")
    assert hypotheses[0].has_conflicts is False

    snapshot_a = create_regulatory_graph_snapshot(graph_result, max_hypotheses=3)
    snapshot_b = create_regulatory_graph_snapshot(graph_result, max_hypotheses=3)

    assert len(snapshot_a.sha256) == 64
    assert snapshot_a.sha256 == snapshot_b.sha256
    assert compute_snapshot_sha256(graph_result, hypotheses) == snapshot_a.sha256

    restored = RegulatoryGraphSnapshot.model_validate(snapshot_a.model_dump(mode="json"))
    assert restored == snapshot_a
    assert compute_snapshot_sha256(restored.graph_result, restored.hypotheses) == restored.sha256

def test_v2_regulatory_graph_snapshot_verify_endpoint() -> None:
    from fastapi.testclient import TestClient
    from bioos.main import app

    client = TestClient(app)
    request_payload = {
        "edges": [
            {
                "edge_id": "e1",
                "source": "TNF",
                "target": "NFKB1",
                "direction": "activation",
                "tissue": "liver",
                "context": "hypoxia",
                "evidence_codes": ["ECO:0000314"],
                "confidence_score": 0.9,
            },
            {
                "edge_id": "e2",
                "source": "NFKB1",
                "target": "HIF1A",
                "direction": "activation",
                "tissue": "liver",
                "context": "hypoxia",
                "evidence_codes": ["ECO:0000314"],
                "confidence_score": 0.8,
            },
        ],
        "query": {
            "seed_entities": ["TNF"],
            "max_hops": 2,
            "tissue": "liver",
            "context": "hypoxia",
            "min_confidence_score": 0.5,
            "conflict_policy": "keep_flagged",
        },
        "targets": ["HIF1A"],
        "max_hypotheses": 3,
    }

    created = client.post("/v2/regulatory-graph/snapshots", json=request_payload)
    assert created.status_code == 200
    snapshot_json = created.json()

    valid_resp = client.post("/v2/regulatory-graph/snapshots/verify", json=snapshot_json)
    assert valid_resp.status_code == 200
    valid_data = valid_resp.json()
    assert valid_data["valid"] is True
    assert valid_data["provided_sha256"] == valid_data["computed_sha256"]

    tampered = dict(snapshot_json)
    tampered["sha256"] = "0" * 64
    invalid_resp = client.post("/v2/regulatory-graph/snapshots/verify", json=tampered)
    assert invalid_resp.status_code == 200
    invalid_data = invalid_resp.json()
    assert invalid_data["valid"] is False
    assert invalid_data["computed_sha256"] == snapshot_json["sha256"]
