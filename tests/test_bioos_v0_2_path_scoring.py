from bioos.engine.v0_2_path_scoring import PathScoringConfig, rank_paths_with_context, score_candidate_path


def test_weight_for_eco_missing_and_known_tiers():
    cfg = PathScoringConfig()
    assert cfg.weight_for_eco(None) == 0.25
    assert cfg.weight_for_eco("") == 0.25
    assert cfg.weight_for_eco(" a ") == 1.0
    assert cfg.weight_for_eco("unknown") == 0.25


def test_score_empty_edges_and_ranked_paths():
    empty_res = score_candidate_path([], counter_sign_hits=[" il6 ", ""])
    assert empty_res["score"] == 0.0
    assert empty_res["has_conflict"] is False
    assert empty_res["counter_sign_hits"] == ["IL6"]

    ranked = rank_paths_with_context(
        [
            [],
            [{"source": "TNF", "target": "IL6", "eco": "A", "context": {"timepoint": "1h", "phase": "early"}}],
        ],
        counter_sign_hits=["TNF"],
    )
    assert [r["rank"] for r in ranked] == [1, 2]
    assert ranked[0]["score"] > ranked[1]["score"]
