from datetime import datetime, timezone
import json
import re
from bioos.models import Assertion


def build_hypotheses_from_signals(signals_text: str, assertions: list[Assertion]) -> dict:
    tokens = re.split(r"[\s,;\n]+", signals_text.upper())
    signals: list[str] = []
    for tok in [re.sub(r"(UP|DOWN|[↑↓])", "", t).strip() for t in tokens]:
        if tok and tok not in signals:
            signals.append(tok)
    matched = [a for a in assertions if a.subject.upper() in signals or a.object.upper() in signals]
    has_ifn = any(s in {"IFNG", "CXCL10", "IFNA", "IFNB", "STAT1"} for s in signals)
    has_infl = any(s in {"TNF", "IL6", "IL1B", "NFKB1", "CRP"} for s in signals)
    primary = (
        "H1: Coordinated activation of pro-inflammatory and interferon responses."
        if (has_ifn and has_infl)
        else f"H1: Cascade activation involving {', '.join(signals) or 'selected factors'}."
    )
    conflict_items = [a.model_dump() for a in matched if a.conflict]
    has_conflicts = bool(conflict_items)
    return {
        "inference_type": "hypothesis_compatibility_not_certainty",
        "signals": signals,
        "primary_hypothesis": primary,
        "alternatives": [
            "H2: Viral or intracellular infectious stimulus.",
            "H3: Autoimmune or hyperinflammatory activation.",
            "H4: Tissue damage / microenvironmental stress.",
            "H5: Cell-composition artifact.",
        ],
        "missing_markers": ["ISG15", "MX1", "OAS1", "IL1B", "CRP", "Cell subset fractions"],
        "counter_markers_to_check": ["IL10", "TGFB1"],
        "uncertainty_breakdown": {
            "data": "moderate" if matched else "high",
            "model": "moderate",
            "context": "high",
            "translational_transfer": "moderate",
        },
        "conflict_count": len(conflict_items),
        "conflicts": conflict_items,
        "bifurcation_note": (
            "Conflicting or multi-axis signals present; distinguish competing hypotheses with targeted assays and cell composition."
            if has_conflicts or (has_ifn and has_infl)
            else "Single-axis alignment; verify context, measurement level, and counter-markers."
        ),
    }


def export_assertions(assertions: list[Assertion], fmt: str = "jsonld") -> str:
    if fmt == "csv":
        rows = [f'"{a.id}","{a.subject}","{a.predicate}","{a.object}","{a.direction}","{a.eco}","{a.source}"' for a in assertions]
        return "\n".join(["id,subject,predicate,object,direction,eco,source", *rows])
    if fmt == "sif":
        return "\n".join(f"{a.subject}\t{a.predicate}\t{a.object}" for a in assertions)
    if fmt == "turtle":
        lines = ["@prefix bioos: <https://bioos.org/ontology/> .", ""]
        for a in assertions:
            lines.extend([
                f'bioos:{a.id.replace(":", "_")} a bioos:Assertion ;',
                f'  bioos:subject "{a.subject}" ; bioos:object "{a.object}" ; bioos:eco "{a.eco}" .',
            ])
        return "\n".join(lines)
    return json.dumps(
        {
            "@context": "https://bioos.org/contexts/assertion-v0.1.jsonld",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "assertions": [a.model_dump() for a in assertions],
        },
        indent=2,
    )
