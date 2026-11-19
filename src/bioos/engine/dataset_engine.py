from bioos.models import (
    Assertion,
    BiologicalState,
    DatasetUploadRequest,
    EvaluatedModuleScore,
    EvaluatedStateScore,
    Module,
)


def _normalize_token(value: str) -> str:
    return value.strip().upper().replace(" UP", "").replace(" DOWN", "")


def evaluate_dataset(
    payload: DatasetUploadRequest,
    assertions: list[Assertion],
    states: list[BiologicalState],
    modules: list[Module],
) -> dict[str, object]:
    active_symbols = {
        _normalize_token(m.symbol)
        for m in payload.measurements
        if m.direction in {"up", "down"} and _normalize_token(m.symbol)
    }

    evaluated_states: list[dict[str, object]] = []
    for state in states:
        driver_tokens = {_normalize_token(d) for d in state.drivers}
        biomarker_tokens = {_normalize_token(b) for b in state.biomarkers}

        matched_drivers = sorted(active_symbols & driver_tokens)
        matched_biomarkers = sorted(active_symbols & biomarker_tokens)
        denominator = max(len(driver_tokens), 1)
        coverage = round(len(matched_drivers) / denominator, 3)

        score_obj = EvaluatedStateScore(
            state_id=state.id,
            name=state.name,
            matched_drivers=matched_drivers,
            matched_biomarkers=matched_biomarkers,
            coverage_ratio=coverage,
        )
        evaluated_states.append(score_obj.model_dump())

    evaluated_modules: list[dict[str, object]] = []
    for mod in modules:
        core_tokens = {_normalize_token(n) for n in mod.core_nodes}
        input_tokens = {_normalize_token(i) for i in mod.inputs}

        matched_core = sorted(active_symbols & core_tokens)
        matched_inputs = sorted(active_symbols & input_tokens)
        denominator = max(len(core_tokens), 1)
        coverage = round(len(matched_core) / denominator, 3)

        mod_obj = EvaluatedModuleScore(
            module_id=mod.id,
            name=mod.name,
            matched_core_nodes=matched_core,
            matched_inputs=matched_inputs,
            coverage_ratio=coverage,
            limitations=mod.limitations,
        )
        evaluated_modules.append(mod_obj.model_dump())

    matched_assertions = [
        a.model_dump()
        for a in assertions
        if _normalize_token(a.subject) in active_symbols
        or _normalize_token(a.object) in active_symbols
    ]
    conflicts = [a for a in matched_assertions if a["conflict"]]

    evaluated_states.sort(key=lambda s: s["coverage_ratio"], reverse=True)
    evaluated_modules.sort(key=lambda m: m["coverage_ratio"], reverse=True)

    return {
        "dataset_name": payload.dataset_name,
        "modality": payload.modality,
        "organism": payload.organism,
        "tissue": payload.tissue,
        "cell_type": payload.cell_type,
        "normalized_features": sorted(active_symbols),
        "active_states": [s for s in evaluated_states if s["coverage_ratio"] > 0],
        "all_evaluated_states": evaluated_states,
        "active_modules": [m for m in evaluated_modules if m["coverage_ratio"] > 0],
        "all_evaluated_modules": evaluated_modules,
        "matched_assertions": matched_assertions,
        "conflicts": conflicts,
        "causality_note": (
            "Feature overlap indicates pathway/state alignment; it does not establish "
            "direct causality or clinical diagnosis."
        ),
    }