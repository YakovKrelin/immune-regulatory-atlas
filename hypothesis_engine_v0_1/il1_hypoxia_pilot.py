from __future__ import annotations

from typing import Any


def evaluate_il1_pathway(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate IL1A and IL1B priming, processing, and secretion independently."""
    il1b_mrna_up = False
    il1b_mrna_seen = False
    pro_il1b_seen = False
    mature_il1b_seen = False
    secreted_il1b_seen = False
    caspase1_seen = False
    il1a_up = False
    il1a_seen = False

    for obs in observations:
        entity = str(obs.get("entity", "")).upper()
        level = str(obs.get("level", "")).lower()
        assay = str(obs.get("assay", "")).lower()
        direction = str(obs.get("direction", "")).lower()
        is_up = direction in {"increase", "up", "elevated"}

        if entity == "IL1B":
            if level == "mrna" or "rna" in assay or "qpcr" in assay:
                il1b_mrna_seen = True
                if is_up:
                    il1b_mrna_up = True
            if "pro" in level or "pro" in entity.lower():
                pro_il1b_seen = True
            if "mature" in level:
                mature_il1b_seen = True
            if "secret" in level or "supernatant" in assay or "elisa" in assay:
                secreted_il1b_seen = True
        elif entity in {"CASP1", "CASPASE1", "CASPASE-1"} or "caspase-1" in assay or "caspase1" in assay:
            caspase1_seen = True
        elif entity in {"IL1A", "IL1-ALPHA", "IL1_ALPHA"}:
            il1a_seen = True
            if is_up:
                il1a_up = True

    il1b_priming_status = (
        "observed_increase" if il1b_mrna_up else ("observed" if il1b_mrna_seen else "not_assessed")
    )
    has_processing_or_secretion = any(
        [pro_il1b_seen, mature_il1b_seen, secreted_il1b_seen, caspase1_seen]
    )
    if secreted_il1b_seen or mature_il1b_seen:
        il1b_processing_status = "observed"
    elif il1b_mrna_up or pro_il1b_seen or caspase1_seen:
        il1b_processing_status = "unassessed_processing"
    else:
        il1b_processing_status = "not_assessed"

    il1b_secretion_status = "observed" if secreted_il1b_seen else "not_assessed"
    il1_alpha_status = (
        "observed_increase" if il1a_up else ("observed" if il1a_seen else "not_assessed")
    )

    if il1a_up and (secreted_il1b_seen or mature_il1b_seen):
        h3_data_status = "supported"
    elif il1b_mrna_up or il1a_seen or has_processing_or_secretion:
        h3_data_status = "partial_unconfirmed"
    else:
        h3_data_status = "not_assessed"

    return {
        "il1b_priming_status": il1b_priming_status,
        "il1b_processing_status": il1b_processing_status,
        "il1b_secretion_status": il1b_secretion_status,
        "il1_alpha_status": il1_alpha_status,
        "h3_data_status": h3_data_status,
    }


def evaluate_il6_hypoxia_pilot(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate H1-H5 hypotheses, mode, pairwise non-separability, and next measurements."""
    context = payload.get("context") or {}
    condition = context.get("condition") or {}
    observations = payload.get("observations") or []
    data_type = str(payload.get("data_type", "")).lower()

    target = payload.get("target") or {}
    target_entity = target.get("entity", "") if isinstance(target, dict) else str(target)
    target_level = target.get("level", "unknown") if isinstance(target, dict) else "unknown"

    has_organism = bool(context.get("organism"))
    has_cell_type = bool(context.get("cell_type"))
    has_condition = bool(
        condition.get("oxygen")
        or condition.get("stimulus")
        or context.get("condition")
    )
    il6_obs = [
        obs for obs in observations if str(obs.get("entity", "")).upper() == "IL6"
    ]

    if not (target_entity and has_organism and has_cell_type and has_condition and il6_obs):
        missing = []
        if not target_entity:
            missing.append("target.entity")
        if not has_organism:
            missing.append("context.organism")
        if not has_cell_type:
            missing.append("context.cell_type")
        if not has_condition:
            missing.append("context.condition")
        if not il6_obs:
            missing.append("observations[IL6]")
        return {
            "mode": "hard_abstention",
            "missing_requirements": missing,
            "hypotheses": {},
            "pairwise_relations": [],
            "discriminating_measurements": [],
            "limitations": ["Required context or IL6 observation is missing."],
        }

    hif_markers = {"HIF1A", "VEGFA", "SLC2A1", "LDHA", "PDK1", "CA9"}
    nfkb_markers = {"NFKBIA", "TNFAIP3", "NFKB1", "RELA"}

    hif_up_count = sum(
        1
        for obs in observations
        if str(obs.get("entity", "")).upper() in hif_markers
        and str(obs.get("direction", "")).lower() in {"increase", "up", "elevated"}
    )
    nfkb_up_count = sum(
        1
        for obs in observations
        if str(obs.get("entity", "")).upper() in nfkb_markers
        and str(obs.get("direction", "")).lower() in {"increase", "up", "elevated"}
    )

    stimulus = str(condition.get("stimulus", "unknown")).lower()
    co_stimulus = str(condition.get("co_stimulus", "unknown")).lower()
    has_tlr_stimulus = any(
        token in f"{stimulus} {co_stimulus}" for token in ("lps", "tlr", "pam3", "cpg", "poly")
    )

    h1_status = (
        "compatible"
        if (str(condition.get("oxygen", "")).lower() == "hypoxia" and hif_up_count >= 1)
        else "not_assessed"
    )
    h2_status = (
        "compatible"
        if (has_tlr_stimulus or nfkb_up_count >= 1)
        else "not_supported_by_data"
    )

    il1_pathway = evaluate_il1_pathway(observations)
    h3_status = il1_pathway["h3_data_status"]

    is_bulk = "bulk" in data_type or "bulk" in str(context.get("sample_type", "")).lower()
    h4_status = "unconfirmed_cell_composition" if is_bulk else "not_assessed"

    secreted_il6_measured = any(
        str(obs.get("entity", "")).upper() == "IL6"
        and (
            "secret" in str(obs.get("level", "")).lower()
            or "elisa" in str(obs.get("assay", "")).lower()
        )
        for obs in observations
    )
    unresolved_flag = (not secreted_il6_measured) or (
        il1_pathway["il1b_priming_status"] == "observed_increase"
        and il1_pathway["il1b_secretion_status"] != "observed"
    )

    mode = "weak_inference" if unresolved_flag else "conditional_inference"

    pairwise_relations: list[dict[str, Any]] = []
    if h1_status == "compatible" and h3_status in {"partial_unconfirmed", "not_assessed"}:
        pairwise_relations.append(
            {
                "hypotheses": ["H1", "H3"],
                "relation": "non_exclusive_not_separable",
                "reason": (
                    "HIF-associated and IL1-associated pathways may both contribute, "
                    "but mature/secreted IL1B and IL1A protein are not assessed."
                ),
            }
        )

    discriminating_measurements = [
        {
            "priority": 1,
            "measurement": "Secreted IL6 protein",
            "method": "ELISA",
            "discriminates": ["H1", "H3", "H4"],
            "rationale": "Checks whether IL6 mRNA elevation translates into secreted IL6.",
        },
        {
            "priority": 2,
            "measurement": "Mature and secreted IL1B, caspase-1 activity",
            "method": "ELISA, Western blot, caspase-1 activity assay",
            "discriminates": ["H3"],
            "rationale": "IL1B mRNA alone does not establish processing or secretion of mature IL-1beta.",
        },
        {
            "priority": 3,
            "measurement": "IL1-alpha protein",
            "method": "ELISA",
            "discriminates": ["H3"],
            "rationale": "Evaluates IL1-alpha separately from IL1-beta.",
        },
        {
            "priority": 4,
            "measurement": "HIF program panel",
            "method": "Targeted qPCR or RNA-seq",
            "discriminates": ["H1"],
            "rationale": "Confirms HIF-associated program compatibility without proving causality.",
        },
        {
            "priority": 5,
            "measurement": "Cell composition",
            "method": "scRNA-seq or flow cytometry",
            "discriminates": ["H4"],
            "rationale": "Separates per-cell response from population shifts in bulk data.",
        },
    ]

    limitations = [
        "Compatibility with a pathway or HIF targets does not establish causal drive of IL6.",
        "Bulk measurements cannot separate per-cell regulation from cell-composition shifts.",
        "Absence of NF-kB target increase at a single timepoint does not rule out transient activation.",
    ]
    if not secreted_il6_measured:
        limitations.insert(0, "IL6 is assessed at mRNA level; secreted IL6 is not measured.")
    if (
        il1_pathway["il1b_priming_status"] == "observed_increase"
        and il1_pathway["il1b_secretion_status"] != "observed"
    ):
        limitations.insert(
            1, "IL1B mRNA is elevated without mature/secreted IL1B or caspase-1 measurements."
        )

    return {
        "query_id": payload.get("query_id"),
        "mode": mode,
        "target_level": target_level,
        "hypotheses": {
            "H1": {
                "name": "HIF-associated IL6 regulation",
                "data_status": h1_status,
            },
            "H2": {
                "name": "TLR / NF-kB-associated IL6 regulation",
                "data_status": h2_status,
            },
            "H3": {
                "name": "IL1-alpha / IL1-beta-dependent IL6 regulation",
                "data_status": h3_status,
            },
            "H4": {
                "name": "Cell-composition shift",
                "data_status": h4_status,
            },
            "H5": {
                "name": "Unresolved or insufficiently specified mechanism",
                "active": unresolved_flag,
            },
        },
        "il1_pathway": il1_pathway,
        "pairwise_relations": pairwise_relations,
        "discriminating_measurements": discriminating_measurements,
        "limitations": limitations,
    }