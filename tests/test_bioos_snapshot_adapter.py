from unittest.mock import patch

from bioos.engine.graph_snapshot import RegulatoryGraphSnapshot
from bioos.experimental.bifurcation_pilot import Direction, InferenceMode, Observation
from bioos.experimental.snapshot_adapter import (
    _build_context_from_snapshot,
    _candidates_to_evidence,
    _classify_path_for_branches,
    _norm_sign,
    run_bifurcation_from_snapshot,
)


def _make_snapshot(target_context=None, candidates=None):
    return RegulatoryGraphSnapshot(
        snapshot_id="SNAP:TEST:01",
        candidates=candidates if candidates is not None else [],
        target_context=target_context if target_context is not None else {},
    )


def test_norm_sign_and_path_classification():
    assert _norm_sign(2) == 1
    assert _norm_sign(-0.5) == -1
    assert _norm_sign(0) == 0
    assert _norm_sign("UP") == 1
    assert _norm_sign("down") == -1
    assert _norm_sign("ambiguous") == 0
    assert _classify_path_for_branches({"nodes": ["TNF", "STAT1"]}, 0) == ["H1"]
    assert _classify_path_for_branches({"path": ["TLR4", "STING1"]}, 0) == ["H2"]
    assert _classify_path_for_branches({"genes": ["CD14", "HLA-DRA"]}, 0) == ["H3"]
    assert _classify_path_for_branches({"path_id": "CUSTOM_X_Y"}, 1) == ["H2"]
    assert _classify_path_for_branches({"path_id": "CUSTOM_Z"}, 2) == ["H3"]


def test_context_and_candidates_conversion():
    snap = _make_snapshot(
        target_context={
            "organism": "Homo sapiens",
            "cell_type": "Macrophage",
            "stimulus": "Hypoxia",
            "timepoint": "12h",
            "duration": "12h",
            "measurement_level": "RNA-seq",
        },
        candidates=[
            "invalid-candidate",
            {
                "path_id": "P_EDGES",
                "edges": [
                    "bad-edge",
                    {"source": "TNF", "target": "IL6", "sign": "-1"},
                    {"subject": "TLR4", "object": "IL6", "polarity": "up"},
                ],
            },
            {
                "id": "P_NODES",
                "nodes": ["CD14", "CD68"],
                "sign": "down",
            },
        ],
    )
    ctx = _build_context_from_snapshot(snap)
    assert ctx.species.value == "Homo sapiens"
    assert ctx.tissue_or_cell.value == "Macrophage"
    assert ctx.condition_or_stimulus.value == "Hypoxia"
    assert ctx.measurement_level == "RNA-seq"
    ev = _candidates_to_evidence(snap)
    assert len(ev) == 3
    assert ev[0].relation_or_direction == Direction.DOWN
    assert ev[1].relation_or_direction == Direction.UP
    assert ev[2].evidence_id == "P_NODES"


def test_run_bifurcation_from_snapshot_with_conflicts():
    snap = _make_snapshot(
        target_context={"species": "human", "tissue": "Blood", "condition": "LPS"},
        candidates=[{"path_id": "TNF_IL6", "nodes": ["TNF", "IL6"], "sign": "+"}],
    )
    obs = [Observation(marker_or_entity="IL6", direction=Direction.UP, measurement_level="RNA-seq")]
    fake_ranking = {
        "ranked_paths": [
            "not-a-dict",
            {
                "path_id": "TNF_IL6",
                "adjusted_net_score": 0.85,
                "counter_sign_count": 1,
                "counter_sign_ratio": 0.25,
                "sign": "-",
            },
        ]
    }
    with patch("bioos.experimental.snapshot_adapter.rank_snapshot_paths_v02", return_value=fake_ranking):
        report = run_bifurcation_from_snapshot(snap, obs, expected_sign="+")
    assert any("TNF_IL6:0.85" in (h.related_path_refs or []) for h in report.hypotheses)
    assert any("counter-sign conflict in TNF_IL6" in c for c in report.conflicts)
    assert report.mode in {InferenceMode.WEAK_INFERENCE, InferenceMode.CONDITIONAL_INFERENCE}


def test_adapter_extra_edge_and_duplicate_branches():
    from bioos.experimental.snapshot_adapter import _extract_path_tokens

    tokens = _extract_path_tokens(
        {
            "edges": ["not-dict", {"subject": "TNF", "object": "IFNG"}],
            "path_id": "TNF_IFNG",
        }
    )
    tokens_mixed = _extract_path_tokens({"nodes": [None, 42, "TNF"], "path_id": "TNF_IFNG"})
    assert {"TNF", "IFNG"}.issubset(tokens)
    assert "TNF" in tokens_mixed

    snap = _make_snapshot(
        target_context={},
        candidates=[{"path_id": "SHORT_CAND", "nodes": ["ONLY_ONE"]}],
    )
    assert _candidates_to_evidence(snap) == []

    obs = [Observation("IL6", Direction.UP)]
    dup_ranking = {
        "ranked_paths": [
            {"path_id": "TNF_IL6", "adjusted_net_score": 0.5, "counter_sign_count": 1, "counter_sign_ratio": 0.1, "sign": "-"},
            {"path_id": "TNF_IL6", "adjusted_net_score": 0.5, "counter_sign_count": 1, "counter_sign_ratio": 0.1, "sign": "-"},
        ]
    }
    with patch("bioos.experimental.snapshot_adapter.rank_snapshot_paths_v02", return_value=dup_ranking):
        report = run_bifurcation_from_snapshot(snap, obs, expected_sign="+")
    assert report.conflicts.count("counter-sign conflict in TNF_IL6 (count=1, ratio=0.1)") == 1
