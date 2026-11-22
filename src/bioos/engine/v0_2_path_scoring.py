from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PathScoringConfig:
    eco_weights: dict[str, float] | None = None
    conflict_penalty: float = 0.25
    counter_sign_penalty_weight: float = 0.20
    temporal_bonus: float = 0.05

    def weight_for_eco(self, eco: str | None) -> float:
        weights = self.eco_weights or {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}
        if not eco:
            return 0.25
        return weights.get(str(eco).strip().upper(), 0.25)


def score_candidate_path(
    edges: list[dict[str, Any]],
    counter_sign_hits: list[str] | None = None,
    config: PathScoringConfig | None = None,
) -> dict[str, Any]:
    """Compute a deterministic score, conflict flags, and temporal notes for a path."""
    cfg = config or PathScoringConfig()
    counter_hits = sorted({s.strip().upper() for s in (counter_sign_hits or []) if s and s.strip()})

    if not edges:
        return {
            "score": 0.0,
            "has_conflict": False,
            "conflict_edge_ids": [],
            "counter_sign_hits": counter_hits,
            "timepoints": [],
            "phases": [],
        }

    edge_scores: list[float] = []
    conflict_edge_ids: list[str] = []
    timepoints: list[str] = []
    phases: list[str] = []

    for edge in edges:
        eco_weight = cfg.weight_for_eco(edge.get("eco"))
        has_edge_conflict = bool(edge.get("conflict")) or bool(edge.get("contradicts"))
        if has_edge_conflict:
            edge_id = str(edge.get("assertion_id") or edge.get("id") or "unknown")
            conflict_edge_ids.append(edge_id)
            eco_weight = max(0.0, eco_weight - cfg.conflict_penalty)
        edge_scores.append(eco_weight)

        ctx = edge.get("context") if isinstance(edge.get("context"), dict) else {}
        if ctx.get("timepoint"):
            timepoints.append(str(ctx["timepoint"]))
        if ctx.get("phase"):
            phases.append(str(ctx["phase"]))

    base_avg = sum(edge_scores) / len(edge_scores)
    temporal_alignment_bonus = cfg.temporal_bonus if len(set(timepoints)) > 1 or len(phases) > 1 else 0.0
    counter_penalty = min(0.5, cfg.counter_sign_penalty_weight * len(counter_hits))

    final_score = round(max(0.0, min(1.0, base_avg + temporal_alignment_bonus - counter_penalty)), 4)

    return {
        "score": final_score,
        "has_conflict": bool(conflict_edge_ids or counter_hits),
        "conflict_edge_ids": conflict_edge_ids,
        "counter_sign_hits": counter_hits,
        "timepoints": timepoints,
        "phases": phases,
    }


def rank_paths_with_context(
    candidate_paths: list[list[dict[str, Any]]],
    counter_sign_hits: list[str] | None = None,
    config: PathScoringConfig | None = None,
) -> list[dict[str, Any]]:
    """Rank candidate paths deterministically by score descending, then fewer conflicts."""
    scored: list[dict[str, Any]] = []
    for idx, path_edges in enumerate(candidate_paths):
        metrics = score_candidate_path(path_edges, counter_sign_hits=counter_sign_hits, config=config)
        scored.append(
            {
                "rank": 0,
                "path_index": idx,
                "nodes": [e.get("source") for e in path_edges[:1]] + [e.get("target") for e in path_edges],
                "edges": path_edges,
                **metrics,
            }
        )

    scored.sort(
        key=lambda item: (
            -item["score"],
            len(item["conflict_edge_ids"]),
            len(item["edges"]),
            item["path_index"],
        )
    )
    for rank_idx, item in enumerate(scored, start=1):
        item["rank"] = rank_idx
    return scored