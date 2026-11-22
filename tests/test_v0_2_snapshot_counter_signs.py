from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from bioos.graph_hypotheses import compute_snapshot_sha256, create_regulatory_graph_snapshot
from bioos.main import app
from bioos.regulatory_graph import (
    ConflictPolicy,
    RegulationDirection,
    RegulatoryEdge,
    SubgraphQuery,
    build_regulatory_graph_result,
)

client = TestClient(app)


def _sample_edges() -> tuple[RegulatoryEdge, ...]:
    return (
        RegulatoryEdge(
            edge_id="E_CLEAN_1",
            source="TNF",
            target="IL6",
            direction=RegulationDirection.ACTIVATION,
            tissue="Blood",
            context="acute",
            evidence_codes=("ECO:0000314",),
            confidence_score=0.9,
            conflict_flag=False,
        ),
        RegulatoryEdge(
            edge_id="E_CLEAN_2",
            source="IL6",
            target="STAT3",
            direction=RegulationDirection.ACTIVATION,
            tissue="Blood",
            context="acute",
            evidence_codes=("ECO:0000314",),
            confidence_score=0.85,
            conflict_flag=False,
        ),
        RegulatoryEdge(
            edge_id="E_COUNTER_1",
            source="IFNG",
            target="CXCL10",
            direction=RegulationDirection.ACTIVATION,
            tissue="Blood",
            context="acute",
            evidence_codes=("ECO:0000314",),
            confidence_score=0.9,
            conflict_flag=False,
        ),
        RegulatoryEdge(
            edge_id="E_COUNTER_2",
            source="CXCL10",
            target="STAT1",
            direction=RegulationDirection.ACTIVATION,
            tissue="Blood",
            context="acute",
            evidence_codes=("ECO:0000314",),
            confidence_score=0.85,
            conflict_flag=False,
        ),
    )


def _tamper_test_payload() -> dict[str, object]:
    edges = [
        RegulatoryEdge(
            edge_id="E_TAMPER_1",
            source="TNF",
            target="IL6",
            direction=RegulationDirection.ACTIVATION,
            tissue="Blood",
            context="acute",
            evidence_codes=("ECO:0000314",),
            confidence_score=0.9,
            conflict_flag=False,
        ),
        RegulatoryEdge(
            edge_id="E_TAMPER_2",
            source="IL6",
            target="STAT3",
            direction=RegulationDirection.ACTIVATION,
            tissue="Blood",
            context="acute",
            evidence_codes=("ECO:0000314",),
            confidence_score=0.85,
            conflict_flag=False,
        ),
    ]
    query = SubgraphQuery(
        seed_entities=("TNF",),
        max_hops=2,
        tissue="Blood",
        context="acute",
        conflict_policy=ConflictPolicy.KEEP_FLAGGED,
    )
    return {
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "query": query.model_dump(mode="json"),
        "targets": ["STAT3"],
        "max_hypotheses": 3,
        "counter_sign_hits": ["IL6 DOWN"],
    }


def test_v0_2_snapshot_with_counter_sign_hits_and_sha256_verification() -> None:
    edges = _sample_edges()
    query = SubgraphQuery(
        seed_entities=("TNF", "IFNG"),
        max_hops=2,
        tissue="Blood",
        context="acute",
        conflict_policy=ConflictPolicy.KEEP_FLAGGED,
    )

    graph_without_hits = build_regulatory_graph_result(
        edges=edges,
        query=query,
        targets=("STAT3", "STAT1"),
    )
    graph_with_hits = build_regulatory_graph_result(
        edges=edges,
        query=query,
        targets=("STAT3", "STAT1"),
        counter_sign_hits=("CXCL10 UP",),
    )

    scores_no_hits = {p.target: p.path_score for p in graph_without_hits.candidate_paths}
    scores_with_hits = {p.target: p.path_score for p in graph_with_hits.candidate_paths}
    conflicts_with_hits = {p.target: p.has_conflicts for p in graph_with_hits.candidate_paths}

    assert scores_with_hits["STAT1"] < scores_no_hits["STAT1"]
    assert scores_with_hits["STAT3"] == scores_no_hits["STAT3"]
    assert conflicts_with_hits["STAT1"] is True
    assert conflicts_with_hits["STAT3"] is False

    snapshot_obj = create_regulatory_graph_snapshot(graph_result=graph_with_hits, max_hypotheses=5)
    assert snapshot_obj.sha256 == compute_snapshot_sha256(
        snapshot_obj.graph_result,
        snapshot_obj.hypotheses,
    )

    hypotheses_by_target = {h.target: h for h in snapshot_obj.hypotheses}
    assert "[contains regulatory conflict or counter-sign hit]" in hypotheses_by_target["STAT1"].summary
    assert "[contains regulatory conflict or counter-sign hit]" not in hypotheses_by_target["STAT3"].summary

    create_resp = client.post(
        "/v2/regulatory-graph/snapshots",
        json={
            "edges": [edge.model_dump(mode="json") for edge in edges],
            "query": query.model_dump(mode="json"),
            "targets": ["STAT3", "STAT1"],
            "max_hypotheses": 5,
            "counter_sign_hits": ["CXCL10 UP"],
        },
    )
    assert create_resp.status_code == 200
    created_snapshot = create_resp.json()
    assert len(created_snapshot["sha256"]) == 64

    verify_resp = client.post(
        "/v2/regulatory-graph/snapshots/verify",
        json=created_snapshot,
    )
    assert verify_resp.status_code == 200
    verify_body = verify_resp.json()
    assert verify_body["valid"] is True
    assert verify_body["provided_sha256"] == created_snapshot["sha256"]
    assert verify_body["computed_sha256"] == created_snapshot["sha256"]


def test_snapshot_verify_rejects_tampered_path_score_or_hypothesis() -> None:
    create_resp = client.post(
        "/v2/regulatory-graph/snapshots",
        json=_tamper_test_payload(),
    )
    assert create_resp.status_code == 200
    snapshot = create_resp.json()
    assert snapshot["graph_result"]["candidate_paths"], "Expected at least one candidate path"
    assert snapshot["hypotheses"], "Expected at least one generated hypothesis"

    tampered_score_snapshot = copy.deepcopy(snapshot)
    current_score = tampered_score_snapshot["graph_result"]["candidate_paths"][0]["path_score"]
    tampered_score_snapshot["graph_result"]["candidate_paths"][0]["path_score"] = (
        round(float(current_score) + 0.123456, 6)
    )

    score_verify_resp = client.post(
        "/v2/regulatory-graph/snapshots/verify",
        json=tampered_score_snapshot,
    )
    assert score_verify_resp.status_code == 200
    score_verify_body = score_verify_resp.json()
    assert score_verify_body["valid"] is False
    assert score_verify_body["provided_sha256"] == snapshot["sha256"]
    assert score_verify_body["computed_sha256"] != snapshot["sha256"]

    tampered_hypothesis_snapshot = copy.deepcopy(snapshot)
    tampered_hypothesis_snapshot["hypotheses"][0]["summary"] = (
        f"{tampered_hypothesis_snapshot['hypotheses'][0]['summary']} [tampered]"
    )

    hyp_verify_resp = client.post(
        "/v2/regulatory-graph/snapshots/verify",
        json=tampered_hypothesis_snapshot,
    )
    assert hyp_verify_resp.status_code == 200
    hyp_verify_body = hyp_verify_resp.json()
    assert hyp_verify_body["valid"] is False
    assert hyp_verify_body["provided_sha256"] == snapshot["sha256"]
    assert hyp_verify_body["computed_sha256"] != snapshot["sha256"]
