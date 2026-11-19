from __future__ import annotations

from typing import Any

ECO_RANK: dict[str, int] = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
}


def _norm_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text else None


def _norm_eco(value: Any) -> str | None:
    if value is None:
        return None
    code = str(value).strip().upper()
    return code if code else None


def _extract_eco_code(assertion: dict[str, Any]) -> str | None:
    for key in ("eco", "eco_level", "evidence_level"):
        val = _norm_eco(assertion.get(key))
        if val:
            return val
    evidence = assertion.get("evidence")
    if isinstance(evidence, dict):
        for key in ("eco", "level", "code"):
            val = _norm_eco(evidence.get(key))
            if val:
                return val
    return None


def _extract_context_field(assertion: dict[str, Any], field_name: str) -> str | None:
    context = assertion.get("context")
    if isinstance(context, dict):
        val = _norm_str(context.get(field_name))
        if val is not None:
            return val
    return _norm_str(assertion.get(field_name))


def _extract_conflict_flag(assertion: dict[str, Any]) -> bool:
    for key in ("conflict", "has_conflict", "conflicting"):
        if key in assertion:
            return bool(assertion.get(key))
    return False


def filter_assertions(
    assertions: list[dict[str, Any]],
    eco_levels: list[str] | None = None,
    min_eco: str | None = None,
    organism: str | None = None,
    cell_type: str | None = None,
    tissue: str | None = None,
    pathology: str | None = None,
    physiology: str | None = None,
    stimulus: str | None = None,
    conflict_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter regulatory assertions by ECO tier, biological context, and conflict flag."""
    allowed_eco_codes: set[str] = set()
    for raw_eco in eco_levels or []:
        for part in str(raw_eco).split(","):
            code = _norm_eco(part)
            if code:
                allowed_eco_codes.add(code)

    min_eco_code = _norm_eco(min_eco)
    max_allowed_rank = ECO_RANK.get(min_eco_code) if min_eco_code else None

    f_organism = _norm_str(organism)
    f_cell_type = _norm_str(cell_type)
    f_tissue = _norm_str(tissue)
    f_pathology = _norm_str(pathology)
    f_physiology = _norm_str(physiology)
    f_stimulus = _norm_str(stimulus)

    matched: list[dict[str, Any]] = []
    for item in assertions:
        if not isinstance(item, dict):
            continue

        eco_code = _extract_eco_code(item)
        if allowed_eco_codes and (eco_code is None or eco_code not in allowed_eco_codes):
            continue

        if max_allowed_rank is not None:
            item_rank = ECO_RANK.get(eco_code) if eco_code else None
            if item_rank is None or item_rank > max_allowed_rank:
                continue

        if conflict_only and not _extract_conflict_flag(item):
            continue

        if f_organism is not None and _extract_context_field(item, "organism") != f_organism:
            continue
        if f_cell_type is not None and _extract_context_field(item, "cell_type") != f_cell_type:
            continue
        if f_tissue is not None and _extract_context_field(item, "tissue") != f_tissue:
            continue
        if f_pathology is not None and _extract_context_field(item, "pathology") != f_pathology:
            continue
        if f_physiology is not None and _extract_context_field(item, "physiology") != f_physiology:
            continue
        if f_stimulus is not None and _extract_context_field(item, "stimulus") != f_stimulus:
            continue

        matched.append(item)

    return matched