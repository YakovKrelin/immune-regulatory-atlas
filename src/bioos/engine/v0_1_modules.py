"""BIOOS v0.1 regulatory module aggregation, timelines, and hypothesis generation."""

from __future__ import annotations

from typing import Any


def _as_item_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "__dict__"):
        return dict(vars(item))
    return {}


def _get_runtime_context() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, Any]:
    import bioos.main as _m

    assertions = _as_item_dict_list(getattr(_m, "ASSERTIONS", []))
    v0_1_assertions = _as_item_dict_list(getattr(_m, "V0_1_ASSERTIONS", []))
    module_specs = getattr(_m, "_V0_1_MODULE_SPECS", [])
    evaluate_fn = getattr(_m, "_evaluate_dataset_payload")
    kb_hash_fn = getattr(_m, "_knowledge_base_sha256")

    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in [*assertions, *v0_1_assertions]:
        item = _as_item_dict(raw)
        item_id = str(item.get("id") or "").strip()
        if item_id:
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
        merged.append(item)

    return merged, module_specs, evaluate_fn, kb_hash_fn


def _as_item_dict_list(items: Any) -> list[dict[str, Any]]:
    if not items:
        return []
    return [_as_item_dict(item) for item in items if _as_item_dict(item)]


_AXIS_EXPECTED_TIMELINE: dict[str, dict[str, Any]] = {
    "Inflammation": {
        "expected_phases": ["innate_response", "acute_phase"],
        "expected_timepoints": ["1h", "6h", "12h", "24h"],
        "transition_note": "Track early cytokine induction against downstream acute-phase and counter-regulatory markers.",
    },
    "Interferon": {
        "expected_phases": ["innate_response", "amplification_isg"],
        "expected_timepoints": ["6h", "12h", "24h"],
        "transition_note": "Verify progression from initial interferon trigger to secondary ISG amplification.",
    },
    "Hypoxia": {
        "expected_phases": ["innate_response", "hypoxic_adaptation"],
        "expected_timepoints": ["6h", "24h"],
        "transition_note": "Compare early HIF1A/mRNA adaptation against sustained hypoxic adaptation and secreted/conflict signals.",
    },
}


def _build_v0_1_modules() -> list[dict[str, Any]]:
    pool, module_specs, _, kb_hash_fn = _get_runtime_context()
    kb_hash = kb_hash_fn()
    modules: list[dict[str, Any]] = []

    for spec in module_specs:
        anchors = {str(node).upper() for node in spec.get("anchor_nodes", [])}
        matched_assertions: list[dict[str, Any]] = []
        nodes_set: set[str] = set(spec.get("anchor_nodes", []))
        modalities_set: set[str] = set()
        phases_set: set[str] = set()
        timepoints_set: set[str] = set()
        translational_set: set[str] = set()
        conflict_ids: list[str] = []

        for item in pool:
            src = str(item.get("source", "")).strip()
            tgt = str(item.get("target", "")).strip()
            ctx = item.get("context") if isinstance(item.get("context"), dict) else {}
            stimulus = str(ctx.get("stimulus", "")).strip().upper()
            pathology = str(ctx.get("pathology", "")).strip().upper()
            axis_upper = str(spec.get("axis", "")).upper()

            is_match = (
                src.upper() in anchors
                or tgt.upper() in anchors
                or stimulus == axis_upper
                or pathology == axis_upper
            )
            if not is_match:
                continue

            matched_assertions.append(item)
            if src:
                nodes_set.add(src)
            if tgt:
                nodes_set.add(tgt)

            mod_val = ctx.get("modality") or ctx.get("measurement_level")
            if mod_val:
                modalities_set.add(str(mod_val))
            if ctx.get("phase"):
                phases_set.add(str(ctx["phase"]))
            if ctx.get("timepoint"):
                timepoints_set.add(str(ctx["timepoint"]))
            if ctx.get("translational_status"):
                translational_set.add(str(ctx["translational_status"]))

            has_conflict = bool(item.get("conflict")) or str(item.get("relation", "")).lower() == "contradicts"
            if has_conflict and item.get("id"):
                conflict_ids.append(str(item["id"]))

        axis_name = str(spec["axis"])
        timeline_meta = _AXIS_EXPECTED_TIMELINE.get(axis_name, {})

        modules.append(
            {
                "module_id": spec["module_id"],
                "version": "0.1",
                "name": spec["name"],
                "axis": axis_name,
                "description": spec["description"],
                "anchor_nodes": spec["anchor_nodes"],
                "nodes": sorted(nodes_set),
                "assertions_count": len(matched_assertions),
                "assertion_ids": [a.get("id") for a in matched_assertions if a.get("id")],
                "conflict_assertion_ids": conflict_ids,
                "modalities": sorted(modalities_set),
                "phases": sorted(phases_set),
                "timepoints": sorted(timepoints_set),
                "translational_statuses": sorted(translational_set),
                "expected_timepoints": timeline_meta.get("expected_timepoints", []),
                "knowledge_base_sha256": kb_hash,
            }
        )

    return modules


def _build_v0_1_timelines(axis: str | None = None, timepoint: str | None = None) -> dict[str, Any]:
    pool, _, _, kb_hash_fn = _get_runtime_context()
    modules = _build_v0_1_modules()
    norm_axis = axis.strip().lower() if axis else None
    norm_tp = timepoint.strip().lower() if timepoint else None

    timelines: list[dict[str, Any]] = []
    for mod in modules:
        axis_name = str(mod["axis"])
        if norm_axis and axis_name.lower() != norm_axis:
            continue

        mod_assertion_ids = set(mod.get("assertion_ids", []))
        events: list[dict[str, Any]] = []

        for item in pool:
            if item.get("id") not in mod_assertion_ids:
                continue
            ctx = item.get("context") if isinstance(item.get("context"), dict) else {}
            tp_val = str(ctx.get("timepoint") or "unspecified")
            if norm_tp and tp_val.lower() != norm_tp:
                continue
            events.append(
                {
                    "assertion_id": item.get("id"),
                    "source": item.get("source"),
                    "target": item.get("target"),
                    "relation": item.get("relation"),
                    "eco": item.get("eco"),
                    "conflict": bool(item.get("conflict")),
                    "timepoint": tp_val,
                    "phase": ctx.get("phase"),
                    "modality": ctx.get("modality") or ctx.get("measurement_level"),
                    "translational_status": ctx.get("translational_status"),
                }
            )

        meta = _AXIS_EXPECTED_TIMELINE.get(axis_name, {})
        timelines.append(
            {
                "axis": axis_name,
                "module_id": mod["module_id"],
                "expected_timepoints": meta.get("expected_timepoints", []),
                "observed_timepoints": mod.get("timepoints", []),
                "observed_phases": mod.get("phases", []),
                "transition_note": meta.get("transition_note", ""),
                "events": events,
            }
        )

    return {
        "version": "0.1",
        "axis_filter": axis,
        "timepoint_filter": timepoint,
        "timelines": timelines,
        "knowledge_base_sha256": kb_hash_fn(),
    }


def _build_v0_1_hypotheses(req: Any) -> dict[str, Any]:
    pool, _, evaluate_fn, _ = _get_runtime_context()
    evaluation = evaluate_fn(
        name=req.name,
        modality=req.modality,
        tissue=req.tissue,
        cell_type=req.cell_type,
        signals=req.signals,
    )
    all_modules = _build_v0_1_modules()
    requested_axes = {a.strip().lower() for a in (getattr(req, "axes", None) or []) if a.strip()}
    modules = (
        [m for m in all_modules if str(m["axis"]).lower() in requested_axes]
        if requested_axes
        else all_modules
    )

    state_to_axis = {
        "Acute Inflammation": "Inflammation",
        "Interferon State": "Interferon",
        "Hypoxia": "Hypoxia",
    }
    active_state_map = {s["name"]: s for s in evaluation["active_states"]}
    active_axes = {state_to_axis[name] for name in active_state_map if name in state_to_axis}

    hypotheses: list[dict[str, Any]] = []
    for mod in modules:
        axis_name = str(mod["axis"])
        matching_state = next(
            (s for s, ax in state_to_axis.items() if ax == axis_name and s in active_state_map),
            None,
        )
        state_obj = active_state_map[matching_state] if matching_state else None
        matched_signals = state_obj["matched_signals"] if state_obj else []
        state_score = state_obj["score"] if state_obj else 0.0
        coverage_score = state_obj.get("coverage_score", 0.0) if state_obj else 0.0
        counter_sign_hits = state_obj.get("counter_sign_hits", []) if state_obj else []
        counter_sign_penalty = state_obj.get("counter_sign_penalty", 0.0) if state_obj else 0.0

        supporting_assertions = [
            item
            for item in pool
            if item.get("id") in mod["assertion_ids"] and not item.get("conflict")
        ]
        conflict_assertions = [
            item
            for item in pool
            if item.get("id") in mod["conflict_assertion_ids"]
        ]

        if matching_state or mod["conflict_assertion_ids"]:
            checks = list(state_obj["missing_context_checks"]) if state_obj else ["context & stimulus confirmation"]
            if not mod.get("timepoints"):
                checks.append("explicit experimental timepoint coverage in KB module")

            hypotheses.append(
                {
                    "hypothesis_id": f"HYP:V0_1:{axis_name.upper()}",
                    "version": "0.1",
                    "axis": axis_name,
                    "module_id": mod["module_id"],
                    "active_state": matching_state,
                    "state_score": state_score,
                    "coverage_score": coverage_score,
                    "counter_sign_hits": counter_sign_hits,
                    "counter_sign_penalty": counter_sign_penalty,
                    "matched_signals": matched_signals,
                    "supporting_assertion_ids": [a.get("id") for a in supporting_assertions if a.get("id")],
                    "conflict_assertion_ids": mod["conflict_assertion_ids"],
                    "timeline_window": {
                        "expected_timepoints": mod.get("expected_timepoints", []),
                        "observed_timepoints": mod.get("timepoints", []),
                        "observed_phases": mod.get("phases", []),
                    },
                    "context_checks_required": checks,
                    "statement": (
                        f"{axis_name} axis ({mod['module_id']}) shows active state '{matching_state}' "
                        f"(score={state_score}, coverage={coverage_score}, penalty={counter_sign_penalty}) "
                        f"with {len(supporting_assertions)} supporting assertions and "
                        f"{len(conflict_assertions)} context-dependent/conflicting edge(s)."
                    ),
                    "not_a_diagnosis": True,
                }
            )

    crosstalk_notes: list[str] = []
    if {"Inflammation", "Hypoxia"}.issubset(active_axes) or "Hypoxia" in active_axes:
        crosstalk_notes.append(
            "Evaluate HIF1A-driven hypoxia adaptation alongside inflammatory/interferon signaling and context-dependent conflict edges."
        )
    if {"Inflammation", "Interferon"}.issubset(active_axes):
        crosstalk_notes.append(
            "Concurrent Inflammation and Interferon signals detected; verify cell-type specificity and temporal phase before inferring synergy or antagonism."
        )

    return {
        "run_name": req.name,
        "version": "0.1",
        "modality": req.modality,
        "tissue": req.tissue,
        "cell_type": req.cell_type,
        "active_states": evaluation["active_states"],
        "hypotheses": hypotheses,
        "crosstalk_notes": crosstalk_notes,
        "modality_warnings": evaluation["modality_warnings"],
        "disclaimer": evaluation["disclaimer"],
        "knowledge_base_sha256": evaluation["knowledge_base_sha256"],
    }