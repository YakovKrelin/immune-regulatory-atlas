from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from bioos.engine.path_ranking import (
    rank_paths_by_context_and_counter_sign_v02,
    summarize_context_counter_sign_cohort_v02,
)


@dataclass
class RegulatoryGraphSnapshot:
    snapshot_id: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    target_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""

    def __post_init__(self) -> None:
        if not self.sha256:
            self.sha256 = self.compute_sha256()

    def compute_sha256(self) -> str:
        payload = {
            "snapshot_id": self.snapshot_id,
            "candidates": self.candidates,
            "target_context": self.target_context,
            "metadata": self.metadata,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def rank_snapshot_paths_v02(
    snapshot: RegulatoryGraphSnapshot | dict[str, Any] | Any,
    *,
    candidates: list[dict[str, Any]] | None = None,
    target_context: dict[str, Any] | None = None,
    expected_sign: Any = None,
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
    context_alignment_bonus: float = 0.25,
    counter_sign_penalty_weight: float = 0.35,
) -> dict[str, Any]:
    if isinstance(candidates, list):
        cand_rows = candidates
    elif isinstance(snapshot, dict):
        cand_rows = list(snapshot.get("candidates") or snapshot.get("paths") or [])
    else:
        cand_rows = list(getattr(snapshot, "candidates", None) or getattr(snapshot, "paths", None) or [])

    if isinstance(target_context, dict):
        ctx = target_context
    elif isinstance(snapshot, dict):
        ctx = dict(snapshot.get("target_context") or snapshot.get("context") or {})
    else:
        ctx = dict(getattr(snapshot, "target_context", None) or getattr(snapshot, "context", None) or {})

    ranked_paths = rank_paths_by_context_and_counter_sign_v02(
        candidates=cand_rows,
        active_markers=active_markers,
        anchor_markers=anchor_markers,
        counter_markers=counter_markers,
        target_context=ctx,
        expected_sign=expected_sign,
        context_alignment_bonus=context_alignment_bonus,
        counter_sign_penalty_weight=counter_sign_penalty_weight,
    )
    cohort_summary = summarize_context_counter_sign_cohort_v02(ranked_paths)

    snap_id = snapshot.get("snapshot_id") if isinstance(snapshot, dict) else getattr(snapshot, "snapshot_id", None)
    snap_sha = snapshot.get("sha256") if isinstance(snapshot, dict) else getattr(snapshot, "sha256", None)

    return {
        "snapshot_id": snap_id,
        "sha256": snap_sha,
        "target_context": ctx,
        "expected_sign": expected_sign,
        "ranked_paths": ranked_paths,
        "cohort_summary": cohort_summary,
    }