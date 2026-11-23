from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


def _norm_set(values: list[str] | set[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()
    return {str(v).strip().upper() for v in values if str(v).strip()}


def _norm_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _norm_sign(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0
    raw = _norm_str(value)
    if raw in {"+", "pos", "positive", "act", "activates", "activating", "up", "upregulates"}:
        return 1
    if raw in {"-", "neg", "negative", "inh", "inhibits", "inhibiting", "rep", "represses", "down", "downregulates"}:
        return -1
    return 0


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(value, 4)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return round(len(left & right) / len(union), 4)


@dataclass(frozen=True)
class PathScoringWeights:
    base_edge_score: float = 1.0
    anchor_bonus_per_match: float = 0.25
    active_coverage_bonus: float = 0.30
    context_match_bonus: float = 0.20
    context_coverage_bonus: float = 0.35
    evidence_bonus_per_edge: float = 0.10
    sign_consistency_bonus: float = 0.15
    conflict_penalty_per_edge: float = 0.45
    counter_penalty_per_hit: float = 0.30


def score_candidate_path_v02(
    path_id: str,
    edges: list[dict[str, Any]],
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    target_context: dict[str, Any] | None = None,
    weights: PathScoringWeights = PathScoringWeights(),
) -> dict[str, Any]:
    active = _norm_set(active_markers)
    anchors = _norm_set(anchor_markers)
    counters = _norm_set(counter_markers)
    ctx = target_context or {}

    context_fields = ("organism", "tissue", "cell_type", "stimulus", "timepoint", "eco")
    requested_ctx_fields = [
        field for field in context_fields
        if _norm_str(ctx.get(field))
    ]

    edge_count = len(edges)
    conflict_edges = [e for e in edges if bool(e.get("conflict"))]
    conflict_count = len(conflict_edges)

    path_tokens: set[str] = set()
    matched_context_fields: set[str] = set()
    context_matches = 0
    supported_edges = 0
    signs: list[int] = []
    field_values: dict[str, list[str]] = {field: [] for field in context_fields}

    for edge in edges:
        subj = str(edge.get("subject", "")).strip().upper()
        obj = str(edge.get("object", "")).strip().upper()
        if subj:
            path_tokens.add(subj)
        if obj:
            path_tokens.add(obj)

        sign_val = _norm_sign(edge.get("sign") or edge.get("direction") or edge.get("polarity"))
        if sign_val != 0:
            signs.append(sign_val)

        edge_ctx = edge.get("context") if isinstance(edge.get("context"), dict) else edge
        for field in context_fields:
            expected = _norm_str(ctx.get(field))
            actual = _norm_str(edge_ctx.get(field))
            if actual:
                field_values[field].append(actual)
            if expected and actual and expected == actual:
                context_matches += 1
                matched_context_fields.add(field)

        eco_val = _norm_str(edge_ctx.get("eco") or edge.get("eco"))
        ev_score = edge.get("evidence_score")
        has_strong_eco = eco_val in {"a", "b", "eco:0000314", "eco:0000353"}
        has_positive_ev = isinstance(ev_score, (int, float)) and float(ev_score) > 0.0
        if (has_strong_eco or has_positive_ev) and not bool(edge.get("conflict")):
            supported_edges += 1

    matched_active = sorted(path_tokens & active)
    matched_anchors = sorted((path_tokens | active) & anchors)
    matched_counters = sorted((path_tokens | active) & counters)

    active_marker_coverage = (
        round(len(matched_active) / len(active), 4)
        if active
        else 0.0
    )

    context_coverage = (
        round(len(matched_context_fields) / len(requested_ctx_fields), 4)
        if requested_ctx_fields
        else 0.0
    )

    conflict_resolution_ratio = (
        1.0
        if conflict_count == 0
        else round(supported_edges / max(1, edge_count), 4)
    )

    agreed_fields = 0
    populated_fields = 0
    for field in context_fields:
        vals = field_values[field]
        if len(vals) >= 2:
            populated_fields += 1
            top_count = Counter(vals).most_common(1)[0][1]
            if top_count == len(vals):
                agreed_fields += 1

    context_agreement_rate = (
        round(agreed_fields / populated_fields, 4)
        if populated_fields > 0
        else (1.0 if edge_count > 0 else 0.0)
    )

    sign_flips = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            sign_flips += 1

    if len(signs) >= 2 and sign_flips == 0 and conflict_count == 0:
        sign_consistency_score = 1.0
    elif len(signs) >= 2:
        sign_consistency_score = round(max(0.0, 1.0 - (sign_flips / (len(signs) - 1))), 4)
    elif len(signs) == 1 and conflict_count == 0:
        sign_consistency_score = 0.5
    else:
        sign_consistency_score = 0.0

    raw_score = edge_count * weights.base_edge_score
    anchor_bonus = round(len(matched_anchors) * weights.anchor_bonus_per_match, 4)
    active_coverage_bonus = round(active_marker_coverage * weights.active_coverage_bonus, 4)
    context_bonus = round(
        context_matches * weights.context_match_bonus
        + context_coverage * weights.context_coverage_bonus,
        4,
    )
    evidence_bonus = round(supported_edges * weights.evidence_bonus_per_edge, 4)
    sign_bonus = round(sign_consistency_score * weights.sign_consistency_bonus, 4)
    conflict_penalty = round(conflict_count * weights.conflict_penalty_per_edge, 4)
    counter_penalty = round(len(matched_counters) * weights.counter_penalty_per_hit, 4)

    net_score = round(
        raw_score
        + anchor_bonus
        + active_coverage_bonus
        + context_bonus
        + evidence_bonus
        + sign_bonus
        - conflict_penalty
        - counter_penalty,
        4,
    )

    max_possible_score = max(
        1.0,
        edge_count
        * (
            weights.base_edge_score
            + weights.anchor_bonus_per_match
            + weights.context_match_bonus
            + weights.evidence_bonus_per_edge
            + weights.sign_consistency_bonus
        )
        + weights.active_coverage_bonus
        + weights.context_coverage_bonus,
    )
    normalized_confidence = _clamp01(net_score / max_possible_score)

    score_breakdown = {
        "bonuses": {
            "anchor": anchor_bonus,
            "active_coverage": active_coverage_bonus,
            "context": context_bonus,
            "evidence": evidence_bonus,
            "sign": sign_bonus,
        },
        "penalties": {
            "conflict": conflict_penalty,
            "counter_marker": counter_penalty,
        },
    }

    return {
        "path_id": path_id,
        "edge_count": edge_count,
        "conflict_count": conflict_count,
        "conflict_resolution_ratio": conflict_resolution_ratio,
        "conflicting_edge_ids": [str(e.get("id", "")) for e in conflict_edges if e.get("id")],
        "matched_active_markers": matched_active,
        "active_marker_coverage": active_marker_coverage,
        "matched_anchors": matched_anchors,
        "matched_counter_markers": matched_counters,
        "matched_context_fields": sorted(matched_context_fields),
        "context_coverage": context_coverage,
        "context_agreement_rate": context_agreement_rate,
        "supported_edges": supported_edges,
        "sign_consistency_score": sign_consistency_score,
        "sign_flips": sign_flips,
        "raw_score": round(raw_score, 4),
        "anchor_bonus": anchor_bonus,
        "active_coverage_bonus": active_coverage_bonus,
        "context_bonus": context_bonus,
        "evidence_bonus": evidence_bonus,
        "sign_bonus": sign_bonus,
        "conflict_penalty": conflict_penalty,
        "counter_penalty": counter_penalty,
        "net_score": net_score,
        "normalized_confidence": normalized_confidence,
        "score_breakdown": score_breakdown,
    }


def compute_context_specificity_v02(
    path_id: str,
    edges: list[dict[str, Any]],
    target_context: dict[str, Any],
    background_contexts: list[dict[str, Any]],
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    weights: PathScoringWeights = PathScoringWeights(),
) -> dict[str, Any]:
    target_scored = score_candidate_path_v02(
        path_id=path_id,
        edges=edges,
        active_markers=active_markers,
        anchor_markers=anchor_markers,
        counter_markers=counter_markers,
        target_context=target_context,
        weights=weights,
    )
    target_ctx_bonus = float(target_scored.get("context_bonus", 0.0))

    bg_bonuses: list[float] = []
    for bg_ctx in background_contexts:
        payload = bg_ctx.get("context") if isinstance(bg_ctx.get("context"), dict) else bg_ctx
        bg_scored = score_candidate_path_v02(
            path_id=path_id,
            edges=edges,
            active_markers=active_markers,
            anchor_markers=anchor_markers,
            counter_markers=counter_markers,
            target_context=payload,
            weights=weights,
        )
        bg_bonuses.append(float(bg_scored.get("context_bonus", 0.0)))

    mean_bg = round(sum(bg_bonuses) / len(bg_bonuses), 4) if bg_bonuses else 0.0
    specificity_delta = round(target_ctx_bonus - mean_bg, 4)
    specificity_ratio = (
        round(target_ctx_bonus / mean_bg, 4)
        if mean_bg > 0.0
        else (float(target_ctx_bonus > 0.0) if target_ctx_bonus > 0.0 else 0.0)
    )

    return {
        "path_id": path_id,
        "target_context_score": round(target_ctx_bonus, 4),
        "background_contexts_evaluated": len(bg_bonuses),
        "mean_background_score": mean_bg,
        "specificity_delta": specificity_delta,
        "specificity_ratio": specificity_ratio,
    }


def compute_path_overlap_metrics_v02(
    edges: list[dict[str, Any]],
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    path_tokens: set[str] = set()
    for edge in edges:
        subj = str(edge.get("subject", "")).strip().upper()
        obj = str(edge.get("object", "")).strip().upper()
        if subj:
            path_tokens.add(subj)
        if obj:
            path_tokens.add(obj)

    active = _norm_set(active_markers)
    anchors = _norm_set(anchor_markers)
    counters = _norm_set(counter_markers)

    return {
        "path_token_count": len(path_tokens),
        "active_jaccard": _jaccard(path_tokens, active),
        "anchor_jaccard": _jaccard(path_tokens, anchors),
        "counter_jaccard": _jaccard(path_tokens, counters),
        "unmatched_active_markers": sorted(active - path_tokens),
        "unmatched_anchor_markers": sorted(anchors - path_tokens),
    }


def attribute_path_evidence_v02(
    edges: list[dict[str, Any]],
    target_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx = target_context or {}
    context_fields = ("organism", "tissue", "cell_type", "stimulus", "timepoint", "eco")
    requested_ctx = {
        field: _norm_str(ctx.get(field))
        for field in context_fields
        if _norm_str(ctx.get(field))
    }

    total_edges = len(edges)
    conflict_free_edges = 0
    supported_edges = 0
    context_matched_edges = 0
    pos_signs = 0
    neg_signs = 0
    neutral_signs = 0

    for edge in edges:
        is_conflict = bool(edge.get("conflict"))
        if not is_conflict:
            conflict_free_edges += 1

        sign_val = _norm_sign(edge.get("sign") or edge.get("direction") or edge.get("polarity"))
        if sign_val > 0:
            pos_signs += 1
        elif sign_val < 0:
            neg_signs += 1
        else:
            neutral_signs += 1

        edge_ctx = edge.get("context") if isinstance(edge.get("context"), dict) else edge
        eco_val = _norm_str(edge_ctx.get("eco") or edge.get("eco"))
        ev_score = edge.get("evidence_score")
        has_strong_eco = eco_val in {"a", "b", "eco:0000314", "eco:0000353"}
        has_positive_ev = isinstance(ev_score, (int, float)) and float(ev_score) > 0.0
        if (has_strong_eco or has_positive_ev) and not is_conflict:
            supported_edges += 1

        if requested_ctx and any(
            _norm_str(edge_ctx.get(field)) == expected
            for field, expected in requested_ctx.items()
        ):
            context_matched_edges += 1

    return {
        "total_edges": total_edges,
        "conflict_free_edges": conflict_free_edges,
        "supported_edges": supported_edges,
        "context_matched_edges": context_matched_edges,
        "sign_counts": {
            "positive": pos_signs,
            "negative": neg_signs,
            "neutral": neutral_signs,
        },
    }


def explain_path_ranking_v02(
    ranked_paths: list[dict[str, Any]],
    index: int = 0,
    edges_by_path_id: dict[str, list[dict[str, Any]]] | None = None,
    target_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not ranked_paths or index < 0 or index >= len(ranked_paths):
        return None

    current = ranked_paths[index]
    pid = str(current.get("path_id", ""))
    cur_score = float(current.get("net_score", 0.0))

    explanation: dict[str, Any] = {
        "path_id": pid,
        "rank": current.get("rank", index + 1),
        "net_score": round(cur_score, 4),
        "normalized_confidence": round(float(current.get("normalized_confidence", 0.0)), 4),
        "conflict_count": int(current.get("conflict_count", 0)),
        "counter_marker_hits": len(current.get("matched_counter_markers") or []),
        "context_coverage": round(float(current.get("context_coverage", 0.0)), 4),
        "score_breakdown": current.get("score_breakdown", {}),
    }

    if edges_by_path_id and pid in edges_by_path_id:
        explanation["evidence_attribution"] = attribute_path_evidence_v02(
            edges=edges_by_path_id[pid],
            target_context=target_context,
        )

    if index + 1 < len(ranked_paths):
        nxt = ranked_paths[index + 1]
        nxt_score = float(nxt.get("net_score", 0.0))
        lead = round(cur_score - nxt_score, 4)
        factors: list[str] = []
        if lead == 0.0:
            if int(current.get("conflict_count", 0)) < int(nxt.get("conflict_count", 0)):
                factors.append("fewer_conflicts")
            if len(current.get("matched_counter_markers") or []) < len(nxt.get("matched_counter_markers") or []):
                factors.append("fewer_counter_markers")
            if pid < str(nxt.get("path_id", "")):
                factors.append("path_id_lexicographic")
        else:
            factors.append("higher_net_score")

        explanation["comparison_to_next"] = {
            "next_path_id": nxt.get("path_id"),
            "score_lead_over_next": lead,
            "tie_breaker_factors": factors,
        }
    else:
        explanation["comparison_to_next"] = None

    return explanation


def classify_rank_margins_v02(
    ranked_paths: list[dict[str, Any]],
    narrow_threshold: float = 0.15,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(ranked_paths)
    for idx, cur in enumerate(ranked_paths):
        pid = str(cur.get("path_id", ""))
        cur_score = float(cur.get("net_score", 0.0))
        if idx + 1 < n:
            nxt = ranked_paths[idx + 1]
            nxt_score = float(nxt.get("net_score", 0.0))
            margin = round(cur_score - nxt_score, 4)
            if margin == 0.0:
                status = "tied"
            elif margin < float(narrow_threshold):
                status = "narrow"
            else:
                status = "decisive"
            next_id: str | None = str(nxt.get("path_id", ""))
        else:
            margin = None
            status = "tail"
            next_id = None

        rows.append(
            {
                "path_id": pid,
                "rank": int(cur.get("rank", idx + 1)),
                "next_path_id": next_id,
                "margin_to_next": margin,
                "margin_class": status,
            }
        )
    return rows


def compare_path_scores_v02(
    primary_score: dict[str, Any],
    baseline_score: dict[str, Any],
) -> dict[str, Any]:
    return {
        "primary_path_id": primary_score.get("path_id"),
        "baseline_path_id": baseline_score.get("path_id"),
        "delta_net_score": round(
            float(primary_score.get("net_score", 0.0))
            - float(baseline_score.get("net_score", 0.0)),
            4,
        ),
        "delta_normalized_confidence": round(
            float(primary_score.get("normalized_confidence", 0.0))
            - float(baseline_score.get("normalized_confidence", 0.0)),
            4,
        ),
        "delta_context_bonus": round(
            float(primary_score.get("context_bonus", 0.0))
            - float(baseline_score.get("context_bonus", 0.0)),
            4,
        ),
        "delta_evidence_bonus": round(
            float(primary_score.get("evidence_bonus", 0.0))
            - float(baseline_score.get("evidence_bonus", 0.0)),
            4,
        ),
        "delta_conflict_penalty": round(
            float(primary_score.get("conflict_penalty", 0.0))
            - float(baseline_score.get("conflict_penalty", 0.0)),
            4,
        ),
        "delta_counter_penalty": round(
            float(primary_score.get("counter_penalty", 0.0))
            - float(baseline_score.get("counter_penalty", 0.0)),
            4,
        ),
    }


def audit_context_score_shifts_v02(
    candidates: list[dict[str, Any]],
    baseline_context: dict[str, Any] | None,
    shifted_context: dict[str, Any] | None,
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    weights: PathScoringWeights = PathScoringWeights(),
) -> list[dict[str, Any]]:
    base_ranked = rank_paths_with_context_v02(
        candidates=candidates,
        active_markers=active_markers,
        anchor_markers=anchor_markers,
        counter_markers=counter_markers,
        target_context=baseline_context,
        weights=weights,
    )
    shifted_ranked = rank_paths_with_context_v02(
        candidates=candidates,
        active_markers=active_markers,
        anchor_markers=anchor_markers,
        counter_markers=counter_markers,
        target_context=shifted_context,
        weights=weights,
    )

    base_by_id = {str(r.get("path_id", "")): r for r in base_ranked if r.get("path_id")}
    shifted_by_id = {str(r.get("path_id", "")): r for r in shifted_ranked if r.get("path_id")}
    all_ids = sorted(set(base_by_id.keys()) | set(shifted_by_id.keys()))

    rows: list[dict[str, Any]] = []
    for pid in all_ids:
        b = base_by_id.get(pid, {})
        s = shifted_by_id.get(pid, {})
        b_score = float(b.get("net_score", 0.0))
        s_score = float(s.get("net_score", 0.0))
        b_ctx_bonus = float(b.get("context_bonus", 0.0))
        s_ctx_bonus = float(s.get("context_bonus", 0.0))
        b_cov = float(b.get("context_coverage", 0.0))
        s_cov = float(s.get("context_coverage", 0.0))

        rows.append(
            {
                "path_id": pid,
                "baseline_net_score": round(b_score, 4),
                "shifted_net_score": round(s_score, 4),
                "delta_net_score": round(s_score - b_score, 4),
                "delta_context_bonus": round(s_ctx_bonus - b_ctx_bonus, 4),
                "context_coverage_delta": round(s_cov - b_cov, 4),
            }
        )

    rows.sort(key=lambda item: (-abs(float(item["delta_net_score"])), str(item["path_id"])))
    return rows


def build_path_score_delta_matrix_v02(
    scored_paths: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    n = len(scored_paths)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append(compare_path_scores_v02(scored_paths[i], scored_paths[j]))
    return pairs


def partition_paths_by_confidence_tier_v02(
    ranked_paths: list[dict[str, Any]],
    high_threshold: float = 0.70,
    moderate_threshold: float = 0.40,
) -> dict[str, list[dict[str, Any]]]:
    tiers: dict[str, list[dict[str, Any]]] = {
        "high": [],
        "moderate": [],
        "low": [],
    }
    for item in ranked_paths:
        conf = float(item.get("normalized_confidence", 0.0))
        entry = dict(item)
        if conf >= float(high_threshold):
            entry["confidence_tier"] = "high"
            tiers["high"].append(entry)
        elif conf >= float(moderate_threshold):
            entry["confidence_tier"] = "moderate"
            tiers["moderate"].append(entry)
        else:
            entry["confidence_tier"] = "low"
            tiers["low"].append(entry)
    return tiers


def rank_paths_with_context_v02(
    candidates: list[dict[str, Any]],
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    target_context: dict[str, Any] | None = None,
    weights: PathScoringWeights = PathScoringWeights(),
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for idx, item in enumerate(candidates):
        path_id = str(item.get("path_id") or f"path_{idx}")
        edges = item.get("edges") if isinstance(item.get("edges"), list) else []
        entry = score_candidate_path_v02(
            path_id=path_id,
            edges=edges,
            active_markers=active_markers,
            anchor_markers=anchor_markers,
            counter_markers=counter_markers,
            target_context=target_context,
            weights=weights,
        )
        scored.append(entry)

    scored.sort(
        key=lambda row: (
            -float(row["net_score"]),
            int(row["conflict_count"]),
            len(row["matched_counter_markers"]),
            str(row["path_id"]),
        )
    )
    for rank_idx, row in enumerate(scored, start=1):
        row["rank"] = rank_idx
    return scored


def filter_and_rank_paths_v02(
    candidates: list[dict[str, Any]],
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    target_context: dict[str, Any] | None = None,
    max_conflicts: int | None = None,
    max_counter_hits: int | None = None,
    min_normalized_confidence: float = 0.0,
    weights: PathScoringWeights = PathScoringWeights(),
) -> list[dict[str, Any]]:
    ranked = rank_paths_with_context_v02(
        candidates=candidates,
        active_markers=active_markers,
        anchor_markers=anchor_markers,
        counter_markers=counter_markers,
        target_context=target_context,
        weights=weights,
    )
    filtered: list[dict[str, Any]] = []
    for row in ranked:
        conflicts = int(row.get("conflict_count", 0))
        counter_hits = len(row.get("matched_counter_markers") or [])
        confidence = float(row.get("normalized_confidence", 0.0))

        if max_conflicts is not None and conflicts > max_conflicts:
            continue
        if max_counter_hits is not None and counter_hits > max_counter_hits:
            continue
        if confidence < float(min_normalized_confidence):
            continue
        filtered.append(dict(row))

    for rank_idx, row in enumerate(filtered, start=1):
        row["rank"] = rank_idx
    return filtered


def gate_paths_by_required_context_v02(
    candidates: list[dict[str, Any]],
    required_context: dict[str, Any],
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    weights: PathScoringWeights = PathScoringWeights(),
) -> list[dict[str, Any]]:
    required_fields = {
        str(k).strip().lower()
        for k, v in (required_context or {}).items()
        if _norm_str(v)
    }
    ranked_all = rank_paths_with_context_v02(
        candidates=candidates,
        active_markers=active_markers,
        anchor_markers=anchor_markers,
        counter_markers=counter_markers,
        target_context=required_context,
        weights=weights,
    )
    if not required_fields:
        return ranked_all

    gated: list[dict[str, Any]] = []
    for row in ranked_all:
        matched = {str(f).strip().lower() for f in (row.get("matched_context_fields") or [])}
        if required_fields.issubset(matched):
            entry = dict(row)
            gated.append(entry)

    for idx, entry in enumerate(gated, start=1):
        entry["rank"] = idx
    return gated


def rank_contexts_for_path_v02(
    path_id: str,
    edges: list[dict[str, Any]],
    context_candidates: list[dict[str, Any]],
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    weights: PathScoringWeights = PathScoringWeights(),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, ctx_item in enumerate(context_candidates):
        ctx_id = str(ctx_item.get("context_id") or f"ctx_{idx}")
        ctx_payload = (
            ctx_item.get("context")
            if isinstance(ctx_item.get("context"), dict)
            else ctx_item
        )
        scored = score_candidate_path_v02(
            path_id=path_id,
            edges=edges,
            active_markers=active_markers,
            anchor_markers=anchor_markers,
            counter_markers=counter_markers,
            target_context=ctx_payload,
            weights=weights,
        )
        rows.append(
            {
                "context_id": ctx_id,
                "path_id": path_id,
                "context": ctx_payload,
                "context_score": scored["context_bonus"],
                "context_coverage": scored["context_coverage"],
                "matched_context_fields": scored["matched_context_fields"],
                "net_score": scored["net_score"],
            }
        )

    rows.sort(
        key=lambda item: (
            -float(item["context_score"]),
            -float(item["context_coverage"]),
            str(item["context_id"]),
        )
    )
    for rank_idx, item in enumerate(rows, start=1):
        item["rank"] = rank_idx
    return rows


def rank_differential_contexts_v02(
    path_id: str,
    edges: list[dict[str, Any]],
    context_candidates: list[dict[str, Any]],
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    weights: PathScoringWeights = PathScoringWeights(),
) -> list[dict[str, Any]]:
    base_rows = rank_contexts_for_path_v02(
        path_id=path_id,
        edges=edges,
        context_candidates=context_candidates,
        active_markers=active_markers,
        anchor_markers=anchor_markers,
        counter_markers=counter_markers,
        weights=weights,
    )
    if not base_rows:
        return []

    n = len(base_rows)
    total_ctx_score = sum(float(r.get("context_score", 0.0)) for r in base_rows)
    diff_rows: list[dict[str, Any]] = []
    for row in base_rows:
        cur_score = float(row.get("context_score", 0.0))
        if n > 1:
            mean_other = (total_ctx_score - cur_score) / (n - 1)
        else:
            mean_other = 0.0
        diff_score = round(cur_score - mean_other, 4)
        enriched = dict(row)
        enriched["mean_other_context_score"] = round(mean_other, 4)
        enriched["differential_score"] = diff_score
        diff_rows.append(enriched)

    diff_rows.sort(
        key=lambda item: (
            -float(item["differential_score"]),
            -float(item["context_coverage"]),
            str(item["context_id"]),
        )
    )
    for idx, item in enumerate(diff_rows, start=1):
        item["rank"] = idx
    return diff_rows


def rank_paths_across_contexts_v02(
    candidates: list[dict[str, Any]],
    context_candidates: list[dict[str, Any]],
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    weights: PathScoringWeights = PathScoringWeights(),
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for idx, ctx_item in enumerate(context_candidates):
        ctx_id = str(ctx_item.get("context_id") or f"ctx_{idx}")
        ctx_payload = (
            ctx_item.get("context")
            if isinstance(ctx_item.get("context"), dict)
            else ctx_item
        )
        ranked_for_ctx = rank_paths_with_context_v02(
            candidates=candidates,
            active_markers=active_markers,
            anchor_markers=anchor_markers,
            counter_markers=counter_markers,
            target_context=ctx_payload,
            weights=weights,
        )
        for path_row in ranked_for_ctx:
            enriched = dict(path_row)
            enriched["context_id"] = ctx_id
            enriched["target_context"] = ctx_payload
            combined.append(enriched)

    combined.sort(
        key=lambda row: (
            -float(row["net_score"]),
            int(row["conflict_count"]),
            len(row["matched_counter_markers"]),
            str(row["context_id"]),
            str(row["path_id"]),
        )
    )
    for rank_idx, row in enumerate(combined, start=1):
        row["rank"] = rank_idx
    return combined


def rank_paths_by_consensus_v02(
    candidates: list[dict[str, Any]],
    context_candidates: list[dict[str, Any]],
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    weights: PathScoringWeights = PathScoringWeights(),
) -> list[dict[str, Any]]:
    rows = rank_paths_across_contexts_v02(
        candidates=candidates,
        context_candidates=context_candidates,
        active_markers=active_markers,
        anchor_markers=anchor_markers,
        counter_markers=counter_markers,
        weights=weights,
    )
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[str(row.get("path_id", ""))].append(row)

    consensus_rows: list[dict[str, Any]] = []
    for path_id, items in by_path.items():
        n_ctx = len(items)
        scores = [float(i.get("net_score", 0.0)) for i in items]
        confs = [float(i.get("normalized_confidence", 0.0)) for i in items]
        zero_conflict = sum(1 for i in items if int(i.get("conflict_count", 0)) == 0)
        total_counter_hits = sum(len(i.get("matched_counter_markers") or []) for i in items)

        consensus_rows.append(
            {
                "path_id": path_id,
                "contexts_evaluated": n_ctx,
                "zero_conflict_contexts": zero_conflict,
                "total_counter_hits": total_counter_hits,
                "mean_net_score": round(sum(scores) / max(1, n_ctx), 4),
                "max_net_score": round(max(scores), 4) if scores else 0.0,
                "mean_normalized_confidence": round(sum(confs) / max(1, n_ctx), 4),
            }
        )

    consensus_rows.sort(
        key=lambda item: (
            -float(item["mean_net_score"]),
            -int(item["zero_conflict_contexts"]),
            int(item["total_counter_hits"]),
            str(item["path_id"]),
        )
    )
    for idx, item in enumerate(consensus_rows, start=1):
        item["rank"] = idx
    return consensus_rows


def rank_paths_by_weighted_context_v02(
    candidates: list[dict[str, Any]],
    context_candidates: list[dict[str, Any]],
    context_weights: dict[str, float] | None = None,
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    weights: PathScoringWeights = PathScoringWeights(),
) -> list[dict[str, Any]]:
    ctx_w_map = {str(k): float(v) for k, v in (context_weights or {}).items()}
    rows = rank_paths_across_contexts_v02(
        candidates=candidates,
        context_candidates=context_candidates,
        active_markers=active_markers,
        anchor_markers=anchor_markers,
        counter_markers=counter_markers,
        weights=weights,
    )

    candidate_weight_lookup: dict[str, float] = {}
    for idx, ctx_item in enumerate(context_candidates):
        cid = str(ctx_item.get("context_id") or f"ctx_{idx}")
        if cid in ctx_w_map:
            candidate_weight_lookup[cid] = max(0.0, float(ctx_w_map[cid]))
        elif isinstance(ctx_item.get("weight"), (int, float)):
            candidate_weight_lookup[cid] = max(0.0, float(ctx_item["weight"]))
        else:
            candidate_weight_lookup[cid] = 1.0

    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[str(row.get("path_id", ""))].append(row)

    weighted_rows: list[dict[str, Any]] = []
    for path_id, items in by_path.items():
        n_ctx = len(items)
        scores = [float(i.get("net_score", 0.0)) for i in items]
        w_sum = 0.0
        w_total = 0.0
        for item in items:
            cid = str(item.get("context_id", ""))
            cw = candidate_weight_lookup.get(cid, 1.0)
            w_sum += float(item.get("net_score", 0.0)) * cw
            w_total += cw

        weighted_mean = round(w_sum / w_total, 4) if w_total > 0.0 else round(sum(scores) / max(1, n_ctx), 4)
        zero_conflict = sum(1 for i in items if int(i.get("conflict_count", 0)) == 0)
        total_counter_hits = sum(len(i.get("matched_counter_markers") or []) for i in items)

        weighted_rows.append(
            {
                "path_id": path_id,
                "contexts_evaluated": n_ctx,
                "zero_conflict_contexts": zero_conflict,
                "total_counter_hits": total_counter_hits,
                "mean_net_score": round(sum(scores) / max(1, n_ctx), 4),
                "weighted_mean_net_score": weighted_mean,
            }
        )

    weighted_rows.sort(
        key=lambda item: (
            -float(item["weighted_mean_net_score"]),
            -int(item["zero_conflict_contexts"]),
            int(item["total_counter_hits"]),
            str(item["path_id"]),
        )
    )
    for idx, item in enumerate(weighted_rows, start=1):
        item["rank"] = idx
    return weighted_rows


def summarize_context_breadth_v02(
    across_context_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in across_context_rows:
        pid = str(row.get("path_id", ""))
        if pid:
            by_path[pid].append(row)

    breadth_rows: list[dict[str, Any]] = []
    for pid in sorted(by_path.keys()):
        items = by_path[pid]
        evaluated = len(items)
        matched_ids = sorted(
            str(item.get("context_id", ""))
            for item in items
            if float(item.get("context_coverage", 0.0)) > 0.0 and item.get("context_id")
        )
        matched_count = len(matched_ids)
        ratio = round(matched_count / evaluated, 4) if evaluated > 0 else 0.0
        breadth_rows.append(
            {
                "path_id": pid,
                "contexts_evaluated": evaluated,
                "contexts_matched": matched_count,
                "context_breadth_ratio": ratio,
                "matched_context_ids": matched_ids,
            }
        )

    breadth_rows.sort(
        key=lambda item: (
            -float(item["context_breadth_ratio"]),
            -int(item["contexts_matched"]),
            str(item["path_id"]),
        )
    )
    return breadth_rows


def group_ranked_paths_by_context_v02(
    ranked_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_rows:
        ctx_id = str(row.get("context_id") or "default_context")
        grouped[ctx_id].append(dict(row))

    result: dict[str, list[dict[str, Any]]] = {}
    for ctx_id in sorted(grouped.keys()):
        items = grouped[ctx_id]
        items.sort(
            key=lambda item: (
                -float(item.get("net_score", 0.0)),
                int(item.get("conflict_count", 0)),
                len(item.get("matched_counter_markers") or []),
                str(item.get("path_id", "")),
            )
        )
        for idx, item in enumerate(items, start=1):
            item["context_rank"] = idx
        result[ctx_id] = items
    return result


def select_top_paths_v02(
    ranked_paths: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    limit = max(1, int(top_k))
    sliced = [dict(item) for item in ranked_paths[:limit]]
    for idx, row in enumerate(sliced, start=1):
        row["rank"] = idx
    return sliced


def serialize_ranked_paths_v02(
    ranked_paths: list[dict[str, Any]],
    include_breakdown: bool = False,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked_paths, start=1):
        record: dict[str, Any] = {
            "rank": int(row.get("rank", idx)),
            "path_id": str(row.get("path_id", "")),
            "net_score": round(float(row.get("net_score", 0.0)), 4),
            "normalized_confidence": round(float(row.get("normalized_confidence", 0.0)), 4),
            "conflict_count": int(row.get("conflict_count", 0)),
            "context_coverage": round(float(row.get("context_coverage", 0.0)), 4),
            "matched_active_markers": list(row.get("matched_active_markers") or []),
            "matched_counter_markers": list(row.get("matched_counter_markers") or []),
        }
        if include_breakdown:
            record["score_breakdown"] = row.get("score_breakdown", {})
        serialized.append(record)
    return serialized


def evaluate_weight_sensitivity_v02(
    candidates: list[dict[str, Any]],
    weight_variants: dict[str, PathScoringWeights],
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    target_context: dict[str, Any] | None = None,
    base_weights: PathScoringWeights = PathScoringWeights(),
) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {
        "baseline": rank_paths_with_context_v02(
            candidates=candidates,
            active_markers=active_markers,
            anchor_markers=anchor_markers,
            counter_markers=counter_markers,
            target_context=target_context,
            weights=base_weights,
        )
    }
    for variant_name in sorted(weight_variants.keys()):
        results[variant_name] = rank_paths_with_context_v02(
            candidates=candidates,
            active_markers=active_markers,
            anchor_markers=anchor_markers,
            counter_markers=counter_markers,
            target_context=target_context,
            weights=weight_variants[variant_name],
        )
    return results


def aggregate_weight_variant_consensus_v02(
    variant_rankings: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_path_ranks: dict[str, list[int]] = defaultdict(list)
    by_path_scores: dict[str, list[float]] = defaultdict(list)

    for _variant_name, ranked_list in variant_rankings.items():
        for row in ranked_list:
            pid = str(row.get("path_id", ""))
            if not pid:
                continue
            by_path_ranks[pid].append(int(row.get("rank", 999999)))
            by_path_scores[pid].append(float(row.get("net_score", 0.0)))

    summary_rows: list[dict[str, Any]] = []
    for pid in sorted(by_path_ranks.keys()):
        ranks = by_path_ranks[pid]
        scores = by_path_scores[pid]
        best_r = min(ranks)
        worst_r = max(ranks)
        mean_r = round(sum(ranks) / len(ranks), 4)
        mean_s = round(sum(scores) / len(scores), 4)
        summary_rows.append(
            {
                "path_id": pid,
                "variants_evaluated": len(ranks),
                "mean_rank": mean_r,
                "best_rank": best_r,
                "worst_rank": worst_r,
                "rank_volatility": worst_r - best_r,
                "mean_net_score": mean_s,
            }
        )

    summary_rows.sort(
        key=lambda item: (
            float(item["mean_rank"]),
            int(item["rank_volatility"]),
            str(item["path_id"]),
        )
    )
    for idx, item in enumerate(summary_rows, start=1):
        item["rank"] = idx
    return summary_rows


def audit_path_rank_shifts_v02(
    baseline_ranked: list[dict[str, Any]],
    perturbed_ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_by_id = {str(r.get("path_id", "")): r for r in baseline_ranked if r.get("path_id")}
    pert_by_id = {str(r.get("path_id", "")): r for r in perturbed_ranked if r.get("path_id")}
    all_ids = sorted(set(base_by_id.keys()) | set(pert_by_id.keys()))

    audit_rows: list[dict[str, Any]] = []
    for pid in all_ids:
        b = base_by_id.get(pid)
        p = pert_by_id.get(pid)
        b_rank = int(b["rank"]) if b and "rank" in b else None
        p_rank = int(p["rank"]) if p and "rank" in p else None
        rank_delta = (b_rank - p_rank) if (b_rank is not None and p_rank is not None) else None
        b_score = float(b.get("net_score", 0.0)) if b else 0.0
        p_score = float(p.get("net_score", 0.0)) if p else 0.0

        audit_rows.append(
            {
                "path_id": pid,
                "baseline_rank": b_rank,
                "perturbed_rank": p_rank,
                "rank_delta": rank_delta,
                "rank_changed": b_rank != p_rank,
                "delta_net_score": round(p_score - b_score, 4),
            }
        )

    audit_rows.sort(
        key=lambda row: (
            row["perturbed_rank"] if row["perturbed_rank"] is not None else 999999,
            str(row["path_id"]),
        )
    )
    return audit_rows


def summarize_ranked_paths_v02(
    ranked_paths: list[dict[str, Any]],
) -> dict[str, Any]:
    if not ranked_paths:
        return {
            "count": 0,
            "top_path_id": None,
            "top_net_score": 0.0,
            "mean_net_score": 0.0,
            "mean_normalized_confidence": 0.0,
            "total_conflicts": 0,
            "common_context_fields": [],
        }

    count = len(ranked_paths)
    net_scores = [float(item.get("net_score", 0.0)) for item in ranked_paths]
    confidences = [float(item.get("normalized_confidence", 0.0)) for item in ranked_paths]
    total_conflicts = sum(int(item.get("conflict_count", 0)) for item in ranked_paths)

    field_counts: Counter[str] = Counter()
    for item in ranked_paths:
        for field in item.get("matched_context_fields") or []:
            field_counts[str(field)] += 1

    common_fields = sorted(
        field for field, freq in field_counts.items() if freq == count
    )

    return {
        "count": count,
        "top_path_id": ranked_paths[0].get("path_id"),
        "top_net_score": round(net_scores[0], 4),
        "mean_net_score": round(sum(net_scores) / count, 4),
        "mean_normalized_confidence": round(sum(confidences) / count, 4),
        "total_conflicts": total_conflicts,
        "common_context_fields": common_fields,
    }


def attach_snapshot_provenance(
    ranked_paths: list[dict[str, Any]],
    snapshot_id: str | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "ranked_paths": ranked_paths,
        "count": len(ranked_paths),
        "snapshot": {
            "snapshot_id": snapshot_id,
            "sha256": sha256,
        }
        if (snapshot_id or sha256)
        else None,
    }


def build_ranked_hypothesis_payload_v02(
    ranked_paths: list[dict[str, Any]],
    snapshot_id: str | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    envelope = attach_snapshot_provenance(
        ranked_paths=ranked_paths,
        snapshot_id=snapshot_id,
        sha256=sha256,
    )
    envelope["summary"] = summarize_ranked_paths_v02(ranked_paths)
    return envelope
