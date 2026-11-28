from bioos.engine.graph_causality import analyze_why_elevated, get_neighborhood, simulate_perturbation
from bioos.models import Assertion


def _make_assertion(aid, subj, obj, direction="up", conflict=False):
    return Assertion.model_construct(
        id=aid,
        subject=subj,
        object=obj,
        direction=direction,
        conflict=conflict,
    )


def test_get_neighborhood_empty_start_and_hops():
    empty = get_neighborhood([], "", max_hops=2)
    assert empty == {"nodes": [], "edges": [], "conflict_count": 0, "conflicting_edges": []}

    assertions = [
        _make_assertion("a1", "TNF", "IL6", direction="up", conflict=True),
        _make_assertion("a2", "IL6", "STAT3", direction="up", conflict=False),
    ]
    hood = get_neighborhood(assertions, "TNF", max_hops=2)
    assert hood["nodes"] == ["TNF", "IL6", "STAT3"]
    assert hood["conflict_count"] == 1
    assert len(hood["edges"]) == 2


def test_analyze_why_elevated_and_simulate_perturbation():
    assertions = [
        _make_assertion("e1", "NFKB", "IL6", direction="up", conflict=True),
        _make_assertion("e2", "TLR4", "NFKB", direction="bidirectional", conflict=False),
    ]
    why = analyze_why_elevated(assertions, "IL6")
    assert why["inference_type"] == "observational_graph_association"
    assert len(why["direct_causes"]) == 1
    assert len(why["upstream_causes"]) == 1
    assert why["conflict_count"] == 1

    sim_il6 = simulate_perturbation(assertions, "IL6", action="suppress")
    assert sim_il6["resulting_direction"] == "down"
    assert "CRP down" in sim_il6["trade_offs"]
    assert sim_il6["conflict_count"] == 1

    sim_other = simulate_perturbation(assertions, "STAT3", action="activate")
    assert sim_other["resulting_direction"] == "up"
    assert sim_other["trade_offs"] == ["Depends on cell type, dose, concentration, and exposure duration"]
