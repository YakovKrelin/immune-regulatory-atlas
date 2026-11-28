from types import SimpleNamespace

from bioos.engine.dataset_engine import _build_measurement_warnings
from bioos.engine.filters import _extract_eco_code, filter_assertions
from bioos.engine.graph_snapshot import rank_snapshot_paths_v02
from bioos.regulatory_graph import (
    RegulatoryEdge,
    SubgraphQuery,
    _normalize_counter_signs,
    extract_contextual_subgraph,
    rank_candidate_paths,
)


def test_graph_snapshot_dict_and_list_branches():
    snap_dict = {
        "candidates": [{"path_id": "P1", "nodes": ["A", "B"], "score": 0.5}],
        "target_context": {"tissue": "blood"},
    }
    res_dict = rank_snapshot_paths_v02(snap_dict, target_context=None)
    assert "ranked_paths" in res_dict

    res_list = rank_snapshot_paths_v02(
        snap_dict,
        candidates=[{"path_id": "P2", "nodes": ["A", "C"], "score": 0.6}],
        target_context={"tissue": "blood"},
    )
    assert "ranked_paths" in res_list


def test_dataset_engine_measurement_warnings():
    req_prot = SimpleNamespace(
        modality="rna-seq",
        measurement_level="protein",
        cell_type=None,
        tissue=None,
    )
    w_prot = _build_measurement_warnings(req_prot)
    assert len(w_prot) == 3

    req_rna = SimpleNamespace(
        modality="RNA-seq",
        measurement_level="RNA",
        cell_type="T_cell",
        tissue="lung",
    )
    w_rna = _build_measurement_warnings(req_rna)
    assert len(w_rna) == 1


def test_filters_eco_without_keys_and_extra_context_fields():
    assert _extract_eco_code({"evidence": {"other": "x"}}) is None

    items = [
        {
            "id": "1",
            "context": {
                "physiology": "homeostasis",
                "phase": "early",
                "timepoint": "2h",
                "translational_status": "translated",
            },
        }
    ]
    assert len(filter_assertions(items, physiology="homeostasis")) == 1
    assert filter_assertions(items, physiology="other") == []
    assert filter_assertions(items, phase="late") == []
    assert filter_assertions(items, timepoint="24h") == []
    assert filter_assertions(items, translational_status="untranslated") == []


def test_regulatory_graph_filters_tokens_and_cycle():
    tokens = _normalize_counter_sign_tokens_or_none(["", "   ", "TNF", "TNF alpha"])
    assert "TNF" in tokens
    assert "TNF ALPHA" in tokens

    e1 = RegulatoryEdge.model_construct(
        edge_id="e1",
        source="A",
        target="B",
        relation="activates",
        direction="up",
        tissue="blood",
        context="ctx1",
        evidence_codes=("ECO:0000314",),
        confidence_score=0.8,
        conflict_flag=False,
    )
    e2 = RegulatoryEdge.model_construct(
        edge_id="e2",
        source="B",
        target="A",
        relation="activates",
        direction="up",
        tissue="blood",
        context="ctx2",
        evidence_codes=("ECO:0000001",),
        confidence_score=0.7,
        conflict_flag=False,
    )
    q_ctx = SubgraphQuery.model_construct(
        seed_entities=("A",),
        tissue="blood",
        context="ctx2",
        max_hops=2,
    )
    sub_ctx = extract_contextual_subgraph([e1, e2], q_ctx)
    assert sub_ctx is not None

    q_eco = SubgraphQuery.model_construct(
        seed_entities=("A",),
        tissue="blood",
        required_evidence_codes=("ECO:0000999",),
        max_hops=2,
    )
    sub_eco = extract_contextual_subgraph([e1, e2], q_eco)
    assert sub_eco is not None

    ranked = rank_candidate_paths([e1, e2], sources=("A",), max_depth=3)
    assert isinstance(ranked, tuple)
    assert len(ranked) >= 1


def _normalize_counter_sign_tokens_or_none(vals):
    return _normalize_counter_signs(vals)
