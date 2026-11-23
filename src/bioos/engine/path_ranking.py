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


@dataclass(frozen=True)
class PathScoringWeights:
    base_edge_score: float = 1.0
    anchor_bonus_per_match: float = 0.25
    context_match_bonus: float = 0.20
    context_coverage_bonus: float = 0.35
    evidence_bonus_per_edge: float = 0.10
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

    requested_ctx_fields = [
        field for field in ("organism", "tissue", "cell_type", "stimulus", "timepoint", "eco")
        if _norm_str(ctx.get(field))
    ]

    edge_count = len(edges)
    conflict_edges = [e for e in edges if bool(e.get("conflict"))]
    conflict_count = len(conflict_edges)

    path_tokens: set[str] = set()
    matched_context_fields: set[str] = set()
    context_matches = 0
    supported_edges = 0

    for edge in edges:
        subj = str(edge.get("subject", "")).strip().upper()
        obj = str(edge.get("object", "")).strip().upper()
        if subj:
            path_tokens.add(subj)
        if obj:
            path_tokens.add(obj)

        edge_ctx = edge.get("context") if isinstance(edge.get("context"), dict) else edge
        for field in ("organism", "tissue", "cell_type", "stimulus", "timepoint", "eco"):
            expected = _norm_str(ctx.get(field))
            actual = _norm_str(edge_ctx.get(field))
            if expected and actual and expected == actual:
                context_matches += 1
                matched_context_fields.add(field)

        eco_val = _norm_str(edge_ctx.get("eco") or edge.get("eco"))
        ev_score = edge.get("evidence_score")
        has_strong_eco = eco_val in {"a", "b", "eco:0000314", "eco:0000353"}
        has_positive_ev = isinstance(ev_score, (int, float)) and float(ev_score) > 0.0
        if (has_strong_eco or has_positive_ev) and not bool(edge.get("conflict")):
            supported_edges += 1

    matched_anchors = sorted((path_tokens | active) & anchors)
    matched_counters = sorted((path_tokens | active) & counters)

    context_coverage = (
        round(len(matched_context_fields) / len(requested_ctx_fields), 4)
        if requested_ctx_fields
        else 0.0
    )

    raw_score = edge_count * weights.base_edge_score
    anchor_bonus = round(len(matched_anchors) * weights.anchor_bonus_per_match, 4)
    context_bonus = round(
        context_matches * weights.context_match_bonus
        + context_coverage * weights.context_coverage_bonus,
        4,
    )
    evidence_bonus = round(supported_edges * weights.evidence_bonus_per_edge, 4)
    conflict_penalty = round(conflict_count * weights.conflict_penalty_per_edge, 4)
    counter_penalty = round(len(matched_counters) * weights.counter_penalty_per_hit, 4)

    net_score = round(
        raw_score
        + anchor_bonus
        + context_bonus
        + evidence_bonus
        - conflict_penalty
        - counter_penalty,
        4,
    )

    return {
        "path_id": path_id,
        "edge_count": edge_count,
        "conflict_count": conflict_count,
        "conflicting_edge_ids": [str(e.get("id", "")) for e in conflict_edges if e.get("id")],
        "matched_anchors": matched_anchors,
        "matched_counter_markers": matched_counters,
        "matched_context_fields": sorted(matched_context_fields),
        "context_coverage": context_coverage,
        "supported_edges": supported_edges,
        "raw_score": round(raw_score, 4),
        "anchor_bonus": anchor_bonus,
        "context_bonus": context_bonus,
        "evidence_bonus": evidence_bonus,
        "conflict_penalty": conflict_penalty,
        "counter_penalty": counter_penalty,
        "net_score": net_score,
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
