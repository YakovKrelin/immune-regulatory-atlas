from __future__ import annotations

from typing import Any

from bioos.engine.graph_snapshot import RegulatoryGraphSnapshot, rank_snapshot_paths_v02
from bioos.experimental.bifurcation_pilot import (
    AbsenceStatus,
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

H1_MARKERS = {"TNF", "IL6", "IL1B", "NFKB1", "IFNG", "STAT1", "CXCL10", "ISG15", "MX1", "OAS1"}
H2_MARKERS = {"TLR3", "TLR4", "TLR7", "TLR9", "DDX58", "RIGI", "IFIH1", "CGAS", "STING1", "IFNB1", "IFNG", "CXCL10", "IL6"}
H3_MARKERS = {"CD14", "CD16", "FCGR3A", "CD4", "CD8A", "HLA-DRA", "CCR7", "CD68", "MKI67"}


def _norm_sign(value: Any) -> int:
    if isinstance(value, (int, float)):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"+", "1", "up", "activate", "activating", "positive", "pos"}:
            return 1
        if s in {"-", "-1", "down", "inhibit", "inhibiting", "negative", "neg"}:
            return -1
    return 0


def _extract_path_tokens(row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("nodes", "path", "genes", "markers"):
        val = row.get(key)
        if isinstance(val, (list, tuple, set)):
            for item in val:
                if isinstance(item, str) and item.strip():
                    tokens.add(item.strip().upper())
    edges = row.get("edges")
    if isinstance(edges, (list, tuple)):
        for edge in edges:
            if isinstance(edge, dict):
                for ek in ("subject", "source", "object", "target"):
                    ev = edge.get(ek)
                    if isinstance(ev, str) and ev.strip():
                        tokens.add(ev.strip().upper())
    pid = str(row.get("path_id") or row.get("id") or "")
    for part in pid.replace("->", "_").replace("-", "_").split("_"):
        if part.strip():
            tokens.add(part.strip().upper())
    return tokens


def _classify_path_for_branches(row: dict[str, Any], index: int) -> list[str]:
    tokens = _extract_path_tokens(row)
    assigned: list[str] = []
    if tokens & H1_MARKERS:
        assigned.append("H1")
    if tokens & H2_MARKERS:
        assigned.append("H2")
    if tokens & H3_MARKERS:
        assigned.append("H3")
    if not assigned:
        fallback_order = ["H1", "H2", "H3"]
        assigned.append(fallback_order[index % len(fallback_order)])
    return assigned


def _build_context_from_snapshot(snapshot: RegulatoryGraphSnapshot) -> ContextSpec:
    target = snapshot.target_context or {}
    species_val = target.get("organism") or target.get("species")
    tissue_val = target.get("cell_type") or target.get("tissue") or target.get("tissue_or_cell")
    cond_val = target.get("stimulus") or target.get("condition") or target.get("condition_or_stimulus")
    time_val = target.get("timepoint")
    dur_val = target.get("duration")
    meas_val = target.get("measurement_level") or target.get("assay")

    return ContextSpec(
        species=ContextField.from_value(species_val) if species_val else ContextField(None, AbsenceStatus.UNKNOWN),
        tissue_or_cell=ContextField.from_value(tissue_val) if tissue_val else ContextField(None, AbsenceStatus.UNKNOWN),
        condition_or_stimulus=ContextField.from_value(cond_val) if cond_val else ContextField(None, AbsenceStatus.UNKNOWN),
        timepoint=ContextField.from_value(time_val) if time_val else ContextField(None, AbsenceStatus.UNKNOWN),
        duration=ContextField.from_value(dur_val) if dur_val else ContextField(None, AbsenceStatus.UNKNOWN),
        measurement_level=str(meas_val) if meas_val else None,
    )


def _candidates_to_evidence(snapshot: RegulatoryGraphSnapshot) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    candidates = getattr(snapshot, "candidates", None) or []
    for c_idx, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            continue
        pid = str(cand.get("path_id") or cand.get("id") or f"cand_{c_idx}")
        edges = cand.get("edges") or []
        if isinstance(edges, list) and edges:
            for e_idx, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    continue
                subj = str(edge.get("subject") or edge.get("source") or "UNKNOWN_SRC")
                obj = str(edge.get("object") or edge.get("target") or "UNKNOWN_TGT")
                sign = _norm_sign(edge.get("sign") or edge.get("polarity"))
                direction = Direction.DOWN if sign < 0 else Direction.UP
                status = EvidenceStatus.CONFLICTED if sign < 0 else EvidenceStatus.REVIEWED
                items.append(
                    EvidenceItem(
                        evidence_id=f"{pid}:e{e_idx}",
                        source_ref=f"snapshot:{snapshot.snapshot_id}",
                        subject=subj,
                        relation_or_direction=direction,
                        target=obj,
                        status=status,
                    )
                )
        else:
            nodes = cand.get("nodes") or []
            if isinstance(nodes, (list, tuple)) and len(nodes) >= 2:
                subj = str(nodes[0])
                obj = str(nodes[-1])
                sign = _norm_sign(cand.get("sign") or cand.get("polarity"))
                direction = Direction.DOWN if sign < 0 else Direction.UP
                status = EvidenceStatus.CONFLICTED if sign < 0 else EvidenceStatus.REVIEWED
                items.append(
                    EvidenceItem(
                        evidence_id=pid,
                        source_ref=f"snapshot:{snapshot.snapshot_id}",
                        subject=subj,
                        relation_or_direction=direction,
                        target=obj,
                        status=status,
                    )
                )
    return items


def run_bifurcation_from_snapshot(
    snapshot: RegulatoryGraphSnapshot,
    observations: list[Observation],
    expected_sign: str = "+",
) -> ReasoningReport:
    ctx = _build_context_from_snapshot(snapshot)
    ev_items = _candidates_to_evidence(snapshot)
    snap_ref = f"{snapshot.snapshot_id}:sha256:{snapshot.sha256}"

    report = evaluate_pilot_scenario(
        observations=observations,
        context=ctx,
        evidence_items=ev_items,
        snapshot_ref=snap_ref,
    )

    ranking_out = rank_snapshot_paths_v02(snapshot, expected_sign=expected_sign)
    ranked_paths = ranking_out.get("ranked_paths") or []

    branch_path_refs: dict[str, list[str]] = {"H1": [], "H2": [], "H3": []}
    for idx, row in enumerate(ranked_paths):
        if not isinstance(row, dict):
            continue
        pid = str(row.get("path_id") or row.get("id") or f"path_{idx}")
        score = row.get("adjusted_net_score", row.get("net_score", row.get("raw_score", 0.0)))
        ref_label = f"{pid}:{score}"
        target_branches = _classify_path_for_branches(row, idx)
        for b_id in target_branches:
            if b_id in branch_path_refs and ref_label not in branch_path_refs[b_id]:
                branch_path_refs[b_id].append(ref_label)

        cs_count = int(row.get("counter_sign_count") or 0)
        cs_ratio = float(row.get("counter_sign_ratio") or 0.0)
        conf_count = int(row.get("conflict_count") or 0)
        has_flag = bool(row.get("counter_sign_conflict")) or bool(row.get("conflict"))
        row_sign = _norm_sign(row.get("sign") or row.get("expected_sign") or row.get("polarity"))
        exp_sign = _norm_sign(expected_sign)
        sign_mismatch = exp_sign != 0 and row_sign != 0 and row_sign != exp_sign

        if cs_count > 0 or cs_ratio > 0.0 or conf_count > 0 or has_flag or sign_mismatch:
            msg = f"counter-sign conflict in {pid} (count={cs_count}, ratio={cs_ratio})"
            if msg not in report.conflicts:
                report.conflicts.append(msg)

    for hyp in report.hypotheses:
        if hyp.hypothesis_id in branch_path_refs and branch_path_refs[hyp.hypothesis_id]:
            hyp.related_path_refs = branch_path_refs[hyp.hypothesis_id]

    if report.conflicts and report.mode == InferenceMode.CONDITIONAL_INFERENCE:
        report.mode = InferenceMode.WEAK_INFERENCE

    return report
