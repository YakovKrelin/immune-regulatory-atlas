"""BIOOS v0.1 regulatory module aggregation and hypothesis generation."""

from __future__ import annotations

from typing import Any

from bioos.data.knowledge import ASSERTIONS



def _as_item_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "__dict__"):
        return dict(vars(item))
    return {}

def _build_v0_1_modules() -> list[dict[str, Any]]:

    import bioos.main as _m
    HypothesisRequest = getattr(_m, 'HypothesisRequest', globals().get('HypothesisRequest'))
    V0_1_ASSERTIONS = getattr(_m, 'V0_1_ASSERTIONS', globals().get('V0_1_ASSERTIONS'))
    _V0_1_MODULE_SPECS = getattr(_m, '_V0_1_MODULE_SPECS', globals().get('_V0_1_MODULE_SPECS'))
    _evaluate_dataset_payload = getattr(_m, '_evaluate_dataset_payload', globals().get('_evaluate_dataset_payload'))
    _knowledge_base_sha256 = getattr(_m, '_knowledge_base_sha256', globals().get('_knowledge_base_sha256'))
    pool = [*ASSERTIONS, *V0_1_ASSERTIONS]
    kb_hash = _knowledge_base_sha256()
    modules: list[dict[str, Any]] = []

    for spec in _V0_1_MODULE_SPECS:
        anchors = {node.upper() for node in spec["anchor_nodes"]}
        matched_assertions: list[dict[str, Any]] = []
        nodes_set: set[str] = set(spec["anchor_nodes"])
        modalities_set: set[str] = set()
        conflict_ids: list[str] = []

        for item in pool:
            item = _as_item_dict(item)
            src = str(_as_item_dict(item).get("source", "")).strip()
            tgt = str(_as_item_dict(item).get("target", "")).strip()
            ctx = _as_item_dict(item).get("context") if isinstance(_as_item_dict(item).get("context"), dict) else {}
            stimulus = str(ctx.get("stimulus", "")).strip().upper()
            pathology = str(ctx.get("pathology", "")).strip().upper()
            axis_upper = spec["axis"].upper()

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

            has_conflict = bool(_as_item_dict(item).get("conflict")) or str(_as_item_dict(item).get("relation", "")).lower() == "contradicts"
            if has_conflict and _as_item_dict(item).get("id"):
                conflict_ids.append(str(item["id"]))

        modules.append(
            {
                "module_id": spec["module_id"],
                "version": "0.1",
                "name": spec["name"],
                "axis": spec["axis"],
                "description": spec["description"],
                "anchor_nodes": spec["anchor_nodes"],
                "nodes": sorted(nodes_set),
                "assertions_count": len(matched_assertions),
                "assertion_ids": [a.get("id") for a in matched_assertions if a.get("id")],
                "conflict_assertion_ids": conflict_ids,
                "modalities": sorted(modalities_set),
                "knowledge_base_sha256": kb_hash,
            }
        )

    return modules

def _build_v0_1_hypotheses(req: HypothesisRequest) -> dict[str, Any]:

    import bioos.main as _m
    HypothesisRequest = getattr(_m, 'HypothesisRequest', globals().get('HypothesisRequest'))
    V0_1_ASSERTIONS = getattr(_m, 'V0_1_ASSERTIONS', globals().get('V0_1_ASSERTIONS'))
    _V0_1_MODULE_SPECS = getattr(_m, '_V0_1_MODULE_SPECS', globals().get('_V0_1_MODULE_SPECS'))
    _evaluate_dataset_payload = getattr(_m, '_evaluate_dataset_payload', globals().get('_evaluate_dataset_payload'))
    _knowledge_base_sha256 = getattr(_m, '_knowledge_base_sha256', globals().get('_knowledge_base_sha256'))
    evaluation = _evaluate_dataset_payload(
        name=req.name,
        modality=req.modality,
        tissue=req.tissue,
        cell_type=req.cell_type,
        signals=req.signals,
    )
    all_modules = _build_v0_1_modules()
    requested_axes = {a.strip().lower() for a in (req.axes or []) if a.strip()}
    if requested_axes:
        modules = [m for m in all_modules if str(m["axis"]).lower() in requested_axes]
    else:
        modules = all_modules

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
        matched_signals = active_state_map[matching_state]["matched_signals"] if matching_state else []
        state_score = active_state_map[matching_state]["score"] if matching_state else 0.0

        supporting_assertions = [
            item
            for item in [*ASSERTIONS, *V0_1_ASSERTIONS]
            if _as_item_dict(item).get("id") in mod["assertion_ids"] and not _as_item_dict(item).get("conflict")
        ]
        conflict_assertions = [
            item
            for item in [*ASSERTIONS, *V0_1_ASSERTIONS]
            if _as_item_dict(item).get("id") in mod["conflict_assertion_ids"]
        ]

        if matching_state or mod["conflict_assertion_ids"]:
            hypotheses.append(
                {
                    "hypothesis_id": f"HYP:V0_1:{axis_name.upper()}",
                    "version": "0.1",
                    "axis": axis_name,
                    "module_id": mod["module_id"],
                    "active_state": matching_state,
                    "state_score": state_score,
                    "matched_signals": matched_signals,
                    "supporting_assertion_ids": [a.get("id") for a in supporting_assertions if a.get("id")],
                    "conflict_assertion_ids": mod["conflict_assertion_ids"],
                    "context_checks_required": (
                        active_state_map[matching_state]["missing_context_checks"]
                        if matching_state
                        else ["context & stimulus confirmation"]
                    ),
                    "statement": (
                        f"{axis_name} axis ({mod['module_id']}) shows active state '{matching_state}' "
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
