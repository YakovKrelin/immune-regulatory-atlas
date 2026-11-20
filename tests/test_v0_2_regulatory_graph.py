from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from bioos.regulatory_graph import (
    ConflictPolicy,
    RegulationDirection,
    RegulatoryEdge,
    RegulatoryGraphResult,
    SubgraphQuery,
    build_regulatory_graph_result,
    detect_and_resolve_conflicts,
    extract_contextual_subgraph,
    rank_candidate_paths,
)


def _sample_edges() -> tuple[RegulatoryEdge, ...]:
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
            evidence_codes=("ECO:0000314", "ECO:0000269"),
            confidence_score=0.80,
        ),
        RegulatoryEdge(
            edge_id="e3",
            source="TNF",
            target="HIF1A",
            direction=RegulationDirection.ACTIVATION,
            tissue="liver",
            context="hypoxia",
            evidence_codes=("ECO:0000314",),
            confidence_score=0.75,
        ),
        RegulatoryEdge(
            edge_id="e4",
            source="TNF",
            target="NFKB1",
            direction=RegulationDirection.INHIBITION,
            tissue="liver",
            context="hypoxia",
            evidence_codes=("ECO:0000304",),
            confidence_score=0.40,
        ),
        RegulatoryEdge(
            edge_id="e5",
            source="TNF",
            target="IL6",
            direction=RegulationDirection.ACTIVATION,
            tissue="blood",
            context="normoxia",
            evidence_codes=("ECO:0000314",),
            confidence_score=0.95,
        ),
    )


def _canonical_sha256(result: RegulatoryGraphResult) -> str:
    payload = json.dumps(
        result.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_pydantic_models_strict_validation() -> None:
    with pytest.raises(ValidationError):
        RegulatoryEdge(
            edge_id="",
            source="TNF",
            target="NFKB1",
            confidence_score=1.5,
        )

    with pytest.raises(ValidationError):
        SubgraphQuery(seed_entities=(), max_hops=0)


def test_extract_contextual_subgraph_filters_tissue_context_eco_and_hops() -> None:
    edges = _sample_edges()
    query = SubgraphQuery(
        seed_entities=("TNF",),
        max_hops=2,
        tissue="liver",
        context="hypoxia",
        required_evidence_codes=("ECO:0000314",),
        min_confidence_score=0.5,
    )

    nodes, selected = extract_contextual_subgraph(edges, query)

    assert nodes == ("HIF1A", "NFKB1", "TNF")
    assert [edge.edge_id for edge in selected] == ["e1", "e2", "e3"]
    assert all(edge.tissue == "liver" and edge.context == "hypoxia" for edge in selected)


def test_extract_contextual_subgraph_respects_max_hops() -> None:
    edges = _sample_edges()
    query = SubgraphQuery(
        seed_entities=("TNF",),
        max_hops=1,
        tissue="liver",
        context="hypoxia",
        min_confidence_score=0.5,
    )

    nodes, selected = extract_contextual_subgraph(edges, query)

    assert nodes == ("HIF1A", "NFKB1", "TNF")
    assert {edge.edge_id for edge in selected} == {"e1", "e3"}


def test_detect_and_resolve_conflicts_policies() -> None:
    conflicting_pair = (
        RegulatoryEdge(
            edge_id="act_1",
            source="TNF",
            target="NFKB1",
            direction=RegulationDirection.ACTIVATION,
            tissue="liver",
            context="hypoxia",
            confidence_score=0.85,
        ),
        RegulatoryEdge(
            edge_id="inh_1",
            source="TNF",
            target="NFKB1",
            direction=RegulationDirection.INHIBITION,
            tissue="liver",
            context="hypoxia",
            confidence_score=0.55,
        ),
    )

    kept, count_keep = detect_and_resolve_conflicts(
        conflicting_pair, policy=ConflictPolicy.KEEP_FLAGGED
    )
    assert count_keep == 1
    assert len(kept) == 2
    assert all(edge.conflict_flag is True for edge in kept)

    preferred, count_pref = detect_and_resolve_conflicts(
        conflicting_pair, policy=ConflictPolicy.PREFER_HIGHER_SCORE
    )
    assert count_pref == 1
    assert len(preferred) == 1
    assert preferred[0].edge_id == "act_1"
    assert preferred[0].conflict_flag is False

    dropped, count_drop = detect_and_resolve_conflicts(
        conflicting_pair, policy=ConflictPolicy.DROP_CONFLICTING
    )
    assert count_drop == 1
    assert dropped == ()


def test_rank_candidate_paths_orders_by_score_and_flags_conflicts() -> None:
    edge_a = RegulatoryEdge(
        edge_id="a",
        source="TNF",
        target="NFKB1",
        direction=RegulationDirection.ACTIVATION,
        confidence_score=0.9,
        conflict_flag=True,
    )
    edge_b = RegulatoryEdge(
        edge_id="b",
        source="NFKB1",
        target="HIF1A",
        direction=RegulationDirection.ACTIVATION,
        confidence_score=0.9,
        conflict_flag=False,
    )
    edge_direct = RegulatoryEdge(
        edge_id="d",
        source="TNF",
        target="HIF1A",
        direction=RegulationDirection.ACTIVATION,
        confidence_score=0.6,
        conflict_flag=False,
    )

    paths = rank_candidate_paths(
        edges=(edge_a, edge_b, edge_direct),
        sources=("TNF",),
        targets=("HIF1A",),
        max_depth=2,
        top_k=5,
    )

    assert len(paths) == 2
    assert paths[0].nodes == ("TNF", "NFKB1", "HIF1A")
    assert paths[0].path_score == round(0.9 * 0.9 / 1.1, 6)
    assert paths[0].has_conflicts is True
    assert paths[1].nodes == ("TNF", "HIF1A")
    assert paths[1].path_score == round(0.6 / 1.0, 6)
    assert paths[1].has_conflicts is False


def test_build_regulatory_graph_result_end_to_end() -> None:
    query = SubgraphQuery(
        seed_entities=("TNF",),
        max_hops=2,
        tissue="liver",
        context="hypoxia",
        min_confidence_score=0.3,
        conflict_policy=ConflictPolicy.PREFER_HIGHER_SCORE,
    )

    result = build_regulatory_graph_result(_sample_edges(), query, targets=("HIF1A",))

    assert result.nodes == ("HIF1A", "NFKB1", "TNF")
    assert result.resolved_conflicts_count == 1
    assert len(result.candidate_paths) == 2
    assert result.warnings == ()


def test_regulatory_graph_result_snapshot_sha256_integrity(tmp_path) -> None:
    edges = _sample_edges()
    query = SubgraphQuery(
        seed_entities=("TNF",),
        max_hops=2,
        tissue="liver",
        context="hypoxia",
        min_confidence_score=0.3,
        conflict_policy=ConflictPolicy.PREFER_HIGHER_SCORE,
    )

    result_first = build_regulatory_graph_result(edges, query, targets=("HIF1A",))
    result_second = build_regulatory_graph_result(edges, query, targets=("HIF1A",))

    digest_first = _canonical_sha256(result_first)
    digest_second = _canonical_sha256(result_second)

    assert len(digest_first) == 64
    assert digest_first == digest_second

    snapshot_file = tmp_path / "regulatory_graph_snapshot.json"
    snapshot_record = {
        "sha256": digest_first,
        "result": result_first.model_dump(mode="json"),
    }
    snapshot_file.write_text(
        json.dumps(snapshot_record, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    loaded = json.loads(snapshot_file.read_text(encoding="utf-8"))
    restored = RegulatoryGraphResult.model_validate(loaded["result"])
    restored_digest = _canonical_sha256(restored)

    assert loaded["sha256"] == digest_first
    assert restored_digest == loaded["sha256"]
    assert restored == result_first

    alt_query = query.model_copy(update={"conflict_policy": ConflictPolicy.DROP_CONFLICTING})
    alt_result = build_regulatory_graph_result(edges, alt_query, targets=("HIF1A",))
    assert _canonical_sha256(alt_result) != digest_first