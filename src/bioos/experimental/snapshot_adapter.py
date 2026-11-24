from __future__ import annotations

from typing import Any

from bioos.engine.graph_snapshot import RegulatoryGraphSnapshot, rank_snapshot_paths_v02
from bioos.engine.path_ranking import _norm_sign
from bioos.experimental.bifurcation_pilot import (
    ContextField,
    ContextSpec,
    Direction,
    EvidenceItem,
    EvidenceStatus,
    InferenceMode,
    Observation,
    ReasoningReport,
    evaluate_pilot_scenario,
)


def context_spec_from_dict(ctx: dict[str, Any] | None) -> ContextSpec:
    data = ctx or {}
    species = data.get("organism") or data.get("species")
    tissue_cell = data.get("cell_type") or data.get("tissue") or data.get("tissue_or_cell")
    stimulus = data.get("stimulus") or data.get("condition") or data.get("condition_or_stimulus")
    timepoint = data.get("timepoint")
    duration = data.get("duration")

    return ContextSpec(
        species=ContextField.from_value(species),
        tissue_or_cell=ContextField.from_value(tissue_cell),
        condition_or_stimulus=ContextField.from_value(stimulus),
        timepoint=ContextField.from_value(timepoint),
        duration=ContextField.from_value(duration),
        dose_raw=str(data["dose"]) if data.get("dose") is not None else None,
        assay=str(data["assay"]) if data.get("assay") is not None else None,
        measurement_level=str(data["measurement_level"]) if data.get("measurement_level") is not None else None,
        donor_or_batch=str(data["donor_or_batch"]) if data.get("donor_or_batch") is not None else None,
    )


def candidates_to_evidence(candidates: list[dict[str, Any]], snapshot_id: str | None) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    prefix = snapshot_id or "snapshot"
    for idx, cand in enumerate(candidates):
        path_id = str(cand.get("path_id") or cand.get("id") or f"path_{idx}")
        nodes = cand.get("nodes") or cand.get("path") or []
        edges = cand.get("edges") or []

        if edges and isinstance(edges, list):
            for e_idx, edge in enumerate(edges):
                subj = str(edge.get("subject", "")).strip()
                obj = str(edge.get("object", "")).strip()
                if not subj or not obj:
                    continue
                is_conflict = bool(edge.get("conflict")) or bool(cand.get("counter_sign_conflict"))
                sign = _norm_sign(edge.get("sign") or edge.get("direction") or edge.get("polarity"))
                direction = (
                    Direction.CONFLICTED
                    if is_conflict
                    else (Direction.UP if sign > 0 else Direction.DOWN if sign < 0 else Direction.UNKNOWN)
                )
                status = EvidenceStatus.CONFLICTED if is_conflict else EvidenceStatus.EXTRACTED
                items.append(
                    EvidenceItem(
                        evidence_id=f"{prefix}:{path_id}:e{e_idx}",
                        source_ref=str(edge.get("source_ref") or f"{prefix}:{path_id}"),
                        subject=subj,
                        relation_or_direction=direction,
                        target=obj,
                        status=status,
                    )
                )
        elif isinstance(nodes, (list, tuple)) and len(nodes) >= 2:
            subj = str(nodes[0]).strip()
            obj = str(nodes[-1]).strip()
            sign = _norm_sign(cand.get("sign") or cand.get("expected_sign") or cand.get("polarity"))
            is_conflict = bool(cand.get("counter_sign_conflict")) or bool(cand.get("conflict"))
            direction = (
                Direction.CONFLICTED
                if is_conflict
                else (Direction.UP if sign > 0 else Direction.DOWN if sign < 0 else Direction.UNKNOWN)
            )
            status = EvidenceStatus.CONFLICTED if is_conflict else EvidenceStatus.EXTRACTED
            items.append(
                EvidenceItem(
                    evidence_id=f"{prefix}:{path_id}",
                    source_ref=f"{prefix}:{path_id}",
                    subject=subj,
                    relation_or_direction=direction,
                    target=obj,
                    status=status,
                )
            )
    return items


def run_bifurcation_from_snapshot(
    snapshot: RegulatoryGraphSnapshot | dict[str, Any],
    observations: list[Observation],
    context: ContextSpec | None = None,
    *,
    expected_sign: Any = None,
    active_markers: list[str] | set[str] | None = None,
    anchor_markers: list[str] | set[str] | None = None,
    counter_markers: list[str] | set[str] | None = None,
) -> ReasoningReport:
    obs_up = [o.marker_or_entity for o in observations if o.direction == Direction.UP]
    effective_active = list(active_markers) if active_markers is not None else obs_up

    snap_result = rank_snapshot_paths_v02(
        snapshot,
        expected_sign=expected_sign,
        active_markers=effective_active,
        anchor_markers=anchor_markers,
        counter_markers=counter_markers,
    )

    ctx_spec = context if context is not None else context_spec_from_dict(snap_result.get("target_context"))
    ranked_paths = snap_result.get("ranked_paths") or []
    cohort_summary = snap_result.get("cohort_summary") or {}

    snap_id = snap_result.get("snapshot_id") or "snapshot"
    snap_sha = snap_result.get("sha256") or "missing_sha256"
    snapshot_ref = f"{snap_id}:sha256:{snap_sha}"

    evidence = candidates_to_evidence(ranked_paths, snap_id)
    report = evaluate_pilot_scenario(
        observations=observations,
        context=ctx_spec,
        evidence_items=evidence,
        snapshot_ref=snapshot_ref,
    )

    top_refs: list[str] = []
    for item in ranked_paths[:5]:
        pid = item.get("path_id") or item.get("id") or "path"
        score_val = (
            item.get("final_score")
            or item.get("adjusted_score")
            or item.get("net_score")
            or item.get("score")
        )
        top_refs.append(f"{pid}:{score_val}" if score_val is not None else str(pid))

    if top_refs or cohort_summary:
        cohort_note = f"cohort_summary={cohort_summary}" if cohort_summary else ""
        for branch in report.hypotheses:
            for ref in top_refs:
                if ref not in branch.related_path_refs:
                    branch.related_path_refs.append(ref)
            if cohort_note:
                branch.uncertainty_notes.append(cohort_note)

    has_snap_conflict = any(
        bool(row.get("counter_sign_conflict")) or bool(row.get("conflict"))
        for row in ranked_paths
    )
    if has_snap_conflict and report.mode == InferenceMode.CONDITIONAL_INFERENCE:
        report.mode = InferenceMode.WEAK_INFERENCE

    return report
