from collections import Counter
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


def select_top_paths_v02(
    ranked_paths: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    limit = max(1, int(top_k))
    sliced = [dict(item) for item in ranked_paths[:limit]]
    for idx, row in enumerate(sliced, start=1):
        row["rank"] = idx
    return sliced


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
