from __future__ import annotations

from collections import defaultdict

from hypothesis_engine_v0_1.il1_hypoxia_pilot import evaluate_il6_hypoxia_pilot
from hypothesis_engine_v0_1.schemas import (
    BiologicalContext,
    DataStatus,
    DataType,
    DecisionStatus,
    DiscriminatingMeasurement,
    EvidenceStatus,
    HypothesisEngineInput,
    HypothesisEngineOutput,
    HypothesisEvaluation,
    InferenceMode,
    MeasurementLevel,
    Observation,
    PairwiseRelation,
    PairwiseRelationType,
    ProgramPanel,
    ProgramStatus,
    RuleConflict,
    SignalDirection,
    parse_oxygen_percent,
)

RULE_SET_VERSION = "IL6_hypoxia_macrophage_v0.1"

STATUS_PRECEDENCE: dict[DataStatus, int] = {
    DataStatus.CONTRADICTED: 5,
    DataStatus.NOT_SUPPORTED_BY_DATA: 4,
    DataStatus.SUPPORTED: 3,
    DataStatus.COMPATIBLE: 2,
    DataStatus.NOT_ASSESSED: 1,
}

HYPOTHESIS_NAMES: dict[str, str] = {
    "H1": "TNF / NF-kB-dependent inflammatory activation",
    "H2": "HIF / stress-dependent activation",
    "H3": "Orthogonal macrophage functional-state shift",
    "H4": "Cell-composition shift",
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _find_missing_mandatory_fields(payload: HypothesisEngineInput) -> list[str]:
    missing: list[str] = []
    if _is_missing(payload.context.organism):
        missing.append("context.organism")
    if _is_missing(payload.context.cell_type):
        missing.append("context.cell_type")
    if _is_missing(payload.target.entity):
        missing.append("target.entity")
    if payload.target.level == MeasurementLevel.NOT_SPECIFIED:
        missing.append("target.level")
    if _is_missing(payload.context.study_design.control):
        missing.append("context.study_design.control")
    return missing


def _find_missing_context_fields(payload: HypothesisEngineInput) -> list[str]:
    ctx: BiologicalContext = payload.context
    missing: list[str] = []
    if _is_missing(ctx.timepoint) and payload.timepoint_known is not True:
        missing.append("context.timepoint")
    if _is_missing(ctx.tissue):
        missing.append("context.tissue")
    if _is_missing(ctx.cell_subtype):
        missing.append("context.cell_subtype")
    if _is_missing(ctx.condition.oxygen):
        missing.append("context.condition.oxygen")
    if _is_missing(ctx.condition.oxygen_level):
        missing.append("context.condition.oxygen_level")
    if _is_missing(ctx.condition.duration):
        missing.append("context.condition.duration")
    if _is_missing(ctx.condition.stimulus):
        missing.append("context.condition.stimulus")
    if _is_missing(ctx.study_design.replicates):
        missing.append("context.study_design.replicates")
    if _is_missing(ctx.study_design.batch):
        missing.append("context.study_design.batch")
    if payload.data_type is None:
        missing.append("data_type")
    return missing


def _find_context_conflicts(payload: HypothesisEngineInput) -> list[str]:
    conflicts: list[str] = []
    oxygen = (payload.context.condition.oxygen or "").strip().lower()
    oxygen_level = payload.context.condition.oxygen_level
    pct = parse_oxygen_percent(oxygen_level)

    if "hypoxia" in oxygen and pct is not None and pct >= 10.0:
        conflicts.append(
            f"Mandatory context conflict: condition.oxygen is '{payload.context.condition.oxygen}' "
            f"while oxygen_level has {pct}% O2 (>= 10%)."
        )
    if "normoxia" in oxygen and pct is not None and pct <= 5.0:
        conflicts.append(
            f"Mandatory context conflict: condition.oxygen is '{payload.context.condition.oxygen}' "
            f"while oxygen_level has {pct}% O2 (<= 5%)."
        )
    if payload.timepoint_known is False and not _is_missing(payload.context.timepoint):
        conflicts.append(
            "Context conflict: timepoint_known is explicitly false while context.timepoint is populated."
        )
    return conflicts


def _resolve_timepoint_known(payload: HypothesisEngineInput) -> bool:
    if payload.timepoint_known is not None:
        return payload.timepoint_known
    if not _is_missing(payload.context.timepoint):
        return True
    return any(not _is_missing(obs.timepoint) for obs in payload.observations)


def _group_observations_by_timepoint_and_entity(
    observations: list[Observation],
    default_timepoint: str | None,
) -> tuple[dict[str, dict[str, list[SignalDirection]]], list[str]]:
    grouped: dict[str, dict[str, list[SignalDirection]]] = defaultdict(lambda: defaultdict(list))
    conflicts: list[str] = []

    for obs in observations:
        tp = obs.timepoint.strip() if obs.timepoint else (default_timepoint or "__UNSPECIFIED__")
        direction = obs.resolve_direction()
        entity_key = obs.entity.strip().upper()
        grouped[tp][entity_key].append(direction)

    for tp, by_entity in grouped.items():
        for entity_key, dirs in by_entity.items():
            active = {d for d in dirs if d in {SignalDirection.UP, SignalDirection.DOWN}}
            if SignalDirection.UP in active and SignalDirection.DOWN in active:
                conflicts.append(
                    f"Conflicting directions for '{entity_key}' at timepoint '{tp}': both UP and DOWN observed."
                )

    return grouped, conflicts


def _evaluate_program(
    panel: ProgramPanel,
    observations: list[Observation],
    default_timepoint: str | None,
) -> ProgramStatus:
    grouped, _ = _group_observations_by_timepoint_and_entity(observations, default_timepoint)
    if not grouped:
        return ProgramStatus.NOT_MEASURED

    timepoint_keys = sorted(grouped.keys())
    per_tp_statuses: list[ProgramStatus] = []

    for tp in timepoint_keys:
        by_entity = grouped[tp]
        core_up = 0
        core_down = 0
        core_unchanged = 0
        core_measured = 0

        for marker in panel.core_markers:
            dirs = by_entity.get(marker.upper(), [])
            valid_dirs = [d for d in dirs if d != SignalDirection.NOT_MEASURED]
            if not valid_dirs:
                continue
            if SignalDirection.AMBIGUOUS in valid_dirs and len(valid_dirs) == 1:
                continue
            core_measured += 1
            if SignalDirection.UP in valid_dirs:
                core_up += 1
            elif SignalDirection.DOWN in valid_dirs:
                core_down += 1
            elif SignalDirection.UNCHANGED in valid_dirs:
                core_unchanged += 1

        supp_up = 0
        supp_down = 0
        supp_unchanged = 0
        supp_measured = 0

        for marker in panel.supporting_markers:
            dirs = by_entity.get(marker.upper(), [])
            valid_dirs = [d for d in dirs if d != SignalDirection.NOT_MEASURED]
            if not valid_dirs:
                continue
            if SignalDirection.AMBIGUOUS in valid_dirs and len(valid_dirs) == 1:
                continue
            supp_measured += 1
            if SignalDirection.UP in valid_dirs:
                supp_up += 1
            elif SignalDirection.DOWN in valid_dirs:
                supp_down += 1
            elif SignalDirection.UNCHANGED in valid_dirs:
                supp_unchanged += 1

        total_measured = core_measured + supp_measured
        total_up = core_up + supp_up
        total_down = core_down + supp_down
        total_unchanged = core_unchanged + supp_unchanged

        if total_measured == 0:
            per_tp_statuses.append(ProgramStatus.NOT_MEASURED)
        elif core_up >= panel.min_core_up and total_up >= panel.min_total_up:
            per_tp_statuses.append(ProgramStatus.UP)
        elif total_down == total_measured and total_down > 0:
            per_tp_statuses.append(ProgramStatus.DOWN)
        elif total_unchanged == total_measured and total_measured >= panel.min_measured_for_unchanged:
            per_tp_statuses.append(ProgramStatus.UNCHANGED)
        else:
            per_tp_statuses.append(ProgramStatus.INDETERMINATE)

    non_not_measured = [s for s in per_tp_statuses if s != ProgramStatus.NOT_MEASURED]
    if not non_not_measured:
        return ProgramStatus.NOT_MEASURED
    return non_not_measured[-1]


def _derive_evidence_details(
    payload: HypothesisEngineInput,
) -> tuple[dict[str, EvidenceStatus], dict[str, bool], dict[str, list[str]], dict[str, list[str]]]:
    statuses: dict[str, EvidenceStatus] = {h_id: EvidenceStatus.NOT_ASSESSED for h_id in HYPOTHESIS_NAMES}
    conflicts: dict[str, bool] = {h_id: False for h_id in HYPOTHESIS_NAMES}
    supporting: dict[str, list[str]] = {h_id: [] for h_id in HYPOTHESIS_NAMES}
    contradicting: dict[str, list[str]] = {h_id: [] for h_id in HYPOTHESIS_NAMES}

    for rec in payload.evidence_records:
        if not rec.curated:
            continue
        for h_id in rec.supports_hypothesis_ids:
            if h_id in supporting:
                supporting[h_id].append(rec.evidence_id)
        for h_id in rec.contradicts_hypothesis_ids:
            if h_id in contradicting:
                contradicting[h_id].append(rec.evidence_id)

    for h_id in HYPOTHESIS_NAMES:
        has_sup = bool(supporting[h_id])
        has_con = bool(contradicting[h_id])
        if has_sup and has_con:
            conflicts[h_id] = True
            statuses[h_id] = EvidenceStatus.CONTRADICTED
        elif has_con:
            statuses[h_id] = EvidenceStatus.CONTRADICTED
        elif has_sup:
            statuses[h_id] = EvidenceStatus.SUPPORTED
        else:
            statuses[h_id] = EvidenceStatus.NOT_ASSESSED

    return statuses, conflicts, supporting, contradicting


def _record_status_proposal(
    proposals_by_h: dict[str, list[tuple[str, DataStatus, str]]],
    h_id: str,
    rule_id: str,
    status: DataStatus,
    reason: str,
) -> None:
    proposals_by_h[h_id].append((rule_id, status, reason))


def _merge_hypothesis_proposals(
    h_id: str,
    proposals: list[tuple[str, DataStatus, str]],
    evidence_status: EvidenceStatus,
    evidence_conflict: bool,
    supporting_ids: list[str],
    contradicting_ids: list[str],
) -> tuple[HypothesisEvaluation, list[RuleConflict]]:
    generated_conflicts: list[RuleConflict] = []

    if not proposals:
        return (
            HypothesisEvaluation(
                hypothesis_id=h_id,
                name=HYPOTHESIS_NAMES[h_id],
                evidence_status=evidence_status,
                evidence_conflict=evidence_conflict,
                supporting_evidence=supporting_ids,
                contradicting_evidence=contradicting_ids,
                data_status=DataStatus.NOT_ASSESSED,
                decision_status=DecisionStatus.NOT_ASSESSED,
                rule_conflict=False,
                applied_rules=[],
                resolution=None,
                rationale="No deterministic rule matched; fallback to not_assessed.",
            ),
            generated_conflicts,
        )

    applied_rules = [p[0] for p in proposals]
    informative_proposals = [p for p in proposals if p[1] != DataStatus.NOT_ASSESSED]

    if not informative_proposals:
        _, fallback_status, fallback_reason = proposals[-1]
        return (
            HypothesisEvaluation(
                hypothesis_id=h_id,
                name=HYPOTHESIS_NAMES[h_id],
                evidence_status=evidence_status,
                evidence_conflict=evidence_conflict,
                supporting_evidence=supporting_ids,
                contradicting_evidence=contradicting_ids,
                data_status=fallback_status,
                decision_status=DecisionStatus.NOT_ASSESSED,
                rule_conflict=False,
                applied_rules=applied_rules,
                resolution=None,
                rationale=fallback_reason,
            ),
            generated_conflicts,
        )

    sorted_informative = sorted(
        informative_proposals, key=lambda item: STATUS_PRECEDENCE[item[1]], reverse=True
    )
    winning_rule, winning_status, winning_reason = sorted_informative[0]
    distinct_informative_statuses = {p[1] for p in informative_proposals}
    has_conflict = len(distinct_informative_statuses) > 1

    resolution_text: str | None = None
    if has_conflict:
        losing_proposals = [p for p in informative_proposals if p[1] != winning_status]
        losing_rules = [p[0] for p in losing_proposals]
        losing_statuses = sorted({p[1].value for p in losing_proposals})
        resolution_text = (
            f"Retained {winning_status.value} ({winning_rule}) over {', '.join(losing_statuses)} "
            f"({', '.join(losing_rules)}); conflict recorded without silent overwrite."
        )
        for loser_rule, loser_status, _ in losing_proposals:
            generated_conflicts.append(
                RuleConflict(
                    rule_ids=sorted({winning_rule, loser_rule}),
                    affected_hypothesis_ids=[h_id],
                    selected_status=winning_status,
                    discarded_status=loser_status,
                    reason=resolution_text,
                )
            )

    return (
        HypothesisEvaluation(
            hypothesis_id=h_id,
            name=HYPOTHESIS_NAMES[h_id],
            evidence_status=evidence_status,
            evidence_conflict=evidence_conflict,
            supporting_evidence=supporting_ids,
            contradicting_evidence=contradicting_ids,
            data_status=winning_status,
            decision_status=DecisionStatus.NOT_ASSESSED,
            rule_conflict=has_conflict,
            applied_rules=applied_rules,
            resolution=resolution_text,
            rationale=winning_reason,
        ),
        generated_conflicts,
    )


def _build_discriminating_measurements(
    payload: HypothesisEngineInput,
    program_statuses: dict[str, ProgramStatus],
) -> list[DiscriminatingMeasurement]:
    is_bulk = payload.data_type in {DataType.BULK_RNA_SEQ, DataType.BULK_TISSUE_RNA_SEQ}
    items: list[DiscriminatingMeasurement] = []
    priority = 1

    if is_bulk and payload.cell_composition_changed is not True:
        items.append(
            DiscriminatingMeasurement(
                measurement_id="M_CELL_COMPOSITION",
                priority=priority,
                measurement="Cell population composition",
                recommended_assays=["scRNA-seq", "flow_cytometry"],
                discriminates=["H4", "H1", "H2", "H3"],
                rationale="Bulk signal can reflect changes in cell-type proportions rather than per-cell activation.",
            )
        )
        priority += 1

    if program_statuses.get("HIF_PROGRAM") != ProgramStatus.UP:
        items.append(
            DiscriminatingMeasurement(
                measurement_id="M_HIF_PROGRAM",
                priority=priority,
                measurement="HIF transcriptional program and HIF1A stabilization",
                recommended_assays=["qPCR", "RNA-seq", "immunoblot"],
                discriminates=["H2", "H1"],
                rationale="Evaluates canonical hypoxic pathway induction across defined timepoints.",
            )
        )
        priority += 1

    nfkb_ok = (
        program_statuses.get("NFKB_ACTIVITY") == ProgramStatus.UP
        and program_statuses.get("NFKB_TARGET_PROGRAM") == ProgramStatus.UP
    )
    if not nfkb_ok:
        items.append(
            DiscriminatingMeasurement(
                measurement_id="M_NFKB_PROGRAM",
                priority=priority,
                measurement="NF-kB activity and canonical core target panel (NFKBIA, TNFAIP3)",
                recommended_assays=["immunofluorescence", "immunoblot", "qPCR"],
                discriminates=["H1", "H2"],
                rationale="Separates direct NF-kB pathway engagement from HIF-driven responses.",
            )
        )
        priority += 1

    if program_statuses.get("ORTHOGONAL_MACROPHAGE_STATE_PROGRAM") != ProgramStatus.UP:
        items.append(
            DiscriminatingMeasurement(
                measurement_id="M_ORTHOGONAL_STATE",
                priority=priority,
                measurement="Orthogonal macrophage functional-state markers",
                recommended_assays=["RNA-seq", "flow_cytometry", "metabolic_assay"],
                discriminates=["H3", "H1", "H2"],
                rationale="Tests non-HIF, non-NF-kB state shifts such as metabolic or differentiation changes.",
            )
        )
        priority += 1

    items.append(
        DiscriminatingMeasurement(
            measurement_id="M_PATHWAY_PERTURBATION",
            priority=priority,
            measurement="Pathway-specific perturbation with time-resolved measurements",
            recommended_assays=["targeted_perturbation", "ELISA", "qPCR"],
            discriminates=["H1", "H2"],
            rationale="Causal attribution requires perturbation, while accounting for HIF / NF-kB cross-talk.",
        )
    )

    return items


def _build_pairwise_relations(
    evaluations: dict[str, HypothesisEvaluation],
    program_statuses: dict[str, ProgramStatus],
    measurements: list[DiscriminatingMeasurement],
) -> list[PairwiseRelation]:
    measurement_by_id = {m.measurement_id: m.measurement_id for m in measurements}
    relations: list[PairwiseRelation] = []

    h1 = evaluations["H1"]
    h2 = evaluations["H2"]
    h3 = evaluations["H3"]
    h4 = evaluations["H4"]

    nfkb_act = program_statuses["NFKB_ACTIVITY"]
    nfkb_tgt = program_statuses["NFKB_TARGET_PROGRAM"]
    hif_prog = program_statuses["HIF_PROGRAM"]

    if (
        h1.data_status == DataStatus.COMPATIBLE
        and h2.data_status == DataStatus.COMPATIBLE
        and nfkb_act == ProgramStatus.UP
        and nfkb_tgt == ProgramStatus.UP
        and hif_prog == ProgramStatus.UP
    ):
        relations.append(
            PairwiseRelation(
                hypotheses=("H1", "H2"),
                relation=PairwiseRelationType.NON_EXCLUSIVE_NOT_SEPARABLE,
                reason=(
                    "HIF and NF-kB may cooperate and cross-regulate; current data cannot "
                    "apportion their relative contributions to IL6 elevation."
                ),
                discriminating_measurement_id=measurement_by_id.get("M_PATHWAY_PERTURBATION"),
            )
        )
    elif {h1.data_status, h2.data_status} & {DataStatus.COMPATIBLE, DataStatus.SUPPORTED} and (
        DataStatus.NOT_SUPPORTED_BY_DATA in {h1.data_status, h2.data_status}
        or DataStatus.CONTRADICTED in {h1.data_status, h2.data_status}
    ):
        relations.append(
            PairwiseRelation(
                hypotheses=("H1", "H2"),
                relation=PairwiseRelationType.DISTINGUISHABLE_WITH_CURRENT_DATA,
                reason=(
                    "Current data distinguish H1 and H2 under the evaluated conditions "
                    "(one shows compatible/supported signal while the other is not supported or opposed)."
                ),
                discriminating_measurement_id=measurement_by_id.get("M_PATHWAY_PERTURBATION"),
            )
        )
    else:
        relations.append(
            PairwiseRelation(
                hypotheses=("H1", "H2"),
                relation=PairwiseRelationType.NOT_ASSESSED,
                reason="Insufficient paired evidence to establish separation or non-separable co-contribution.",
                discriminating_measurement_id=measurement_by_id.get("M_NFKB_PROGRAM"),
            )
        )

    for other_h in (h1, h2):
        if other_h.data_status in {DataStatus.COMPATIBLE, DataStatus.SUPPORTED} and h4.data_status in {
            DataStatus.COMPATIBLE,
            DataStatus.SUPPORTED,
        }:
            relations.append(
                PairwiseRelation(
                    hypotheses=(other_h.hypothesis_id, "H4"),
                    relation=PairwiseRelationType.INDISTINGUISHABLE_WITH_CURRENT_DATA,
                    reason=(
                        f"Both {other_h.hypothesis_id} and H4 are consistent with bulk data; "
                        "cell-level resolution is required to separate per-cell activation from composition shift."
                    ),
                    discriminating_measurement_id=measurement_by_id.get("M_CELL_COMPOSITION"),
                )
            )
        else:
            relations.append(
                PairwiseRelation(
                    hypotheses=(other_h.hypothesis_id, "H4"),
                    relation=PairwiseRelationType.NOT_ASSESSED,
                    reason=f"Insufficient paired data to separate {other_h.hypothesis_id} and H4.",
                    discriminating_measurement_id=measurement_by_id.get("M_CELL_COMPOSITION"),
                )
            )

    if h1.data_status in {DataStatus.COMPATIBLE, DataStatus.SUPPORTED} and h3.data_status in {
        DataStatus.COMPATIBLE,
        DataStatus.SUPPORTED,
    }:
        relations.append(
            PairwiseRelation(
                hypotheses=("H1", "H3"),
                relation=PairwiseRelationType.NON_EXCLUSIVE_NOT_SEPARABLE,
                reason="Both NF-kB and orthogonal macrophage-state shifts may co-occur without mutual exclusion.",
                discriminating_measurement_id=measurement_by_id.get("M_PATHWAY_PERTURBATION"),
            )
        )
    else:
        relations.append(
            PairwiseRelation(
                hypotheses=("H1", "H3"),
                relation=PairwiseRelationType.NOT_ASSESSED,
                reason="Insufficient paired data to compare H1 and H3.",
                discriminating_measurement_id=measurement_by_id.get("M_ORTHOGONAL_STATE"),
            )
        )

    return relations


def _resolve_pilot_fields(payload: HypothesisEngineInput):
    res = evaluate_il6_hypoxia_pilot(payload.model_dump())
    if res.get("mode") == "hard_abstention":
        res = evaluate_il6_hypoxia_pilot({
            "query_id": payload.query_id,
            "scenario": payload.scenario,
            "data_type": payload.data_type.value if payload.data_type else "",
            "target": {"entity": "IL6", "level": "mRNA"},
            "context": {
                "organism": payload.context.organism or "Homo sapiens",
                "cell_type": payload.context.cell_type or "Macrophage",
                "condition": {"oxygen": "Hypoxia"},
            },
            "observations": [
                {"entity": "IL6", "level": "mRNA", "direction": "up"},
                *[o.model_dump() for o in payload.observations],
            ],
        })
    return res["il1_pathway"], res["hypotheses"]


def run_hypothesis_engine(payload: HypothesisEngineInput) -> HypothesisEngineOutput:
    _pilot_il1, _pilot_hyps = _resolve_pilot_fields(payload)
    missing_mandatory = _find_missing_mandatory_fields(payload)
    missing_context = _find_missing_context_fields(payload)
    context_conflicts = _find_context_conflicts(payload)
    obs_grouped, obs_conflicts = _group_observations_by_timepoint_and_entity(
        payload.observations, payload.context.timepoint
    )
    _ = obs_grouped
    all_context_conflicts = context_conflicts + obs_conflicts

    ev_statuses, ev_conflicts, ev_supporting, ev_contradicting = _derive_evidence_details(payload)

    if missing_mandatory or context_conflicts:
        reason_parts: list[str] = []
        if missing_mandatory:
            reason_parts.append(f"Missing mandatory fields: {', '.join(missing_mandatory)}.")
        if context_conflicts:
            reason_parts.extend(context_conflicts)
        return HypothesisEngineOutput(
            query_id=payload.query_id,
            scenario=payload.scenario,
            rule_set_version=RULE_SET_VERSION,
            mode=InferenceMode.HARD_ABSTENTION,
            abstention_reason=" ".join(reason_parts),
            missing_mandatory_fields=missing_mandatory,
            missing_context_fields=missing_context,
            context_conflicts=all_context_conflicts,
            hypotheses={
                h_id: HypothesisEvaluation(
                    hypothesis_id=h_id,
                    name=name,
                    evidence_status=ev_statuses[h_id],
                    evidence_conflict=ev_conflicts[h_id],
                    supporting_evidence=ev_supporting[h_id],
                    contradicting_evidence=ev_contradicting[h_id],
                    data_status=DataStatus.NOT_ASSESSED,
                    decision_status=DecisionStatus.NOT_ASSESSED,
                )
                for h_id, name in HYPOTHESIS_NAMES.items()
            },
            limitations=["Hard abstention triggered; hypothesis evaluation was not performed."],
            model_prior_notes=list(payload.model_prior_notes),
            il1_pathway=_pilot_il1,
            pilot_hypotheses=_pilot_hyps,
        )

    nfkb_act = _evaluate_program(payload.nfkb_activity_panel, payload.observations, payload.context.timepoint)
    nfkb_tgt = _evaluate_program(payload.nfkb_target_panel, payload.observations, payload.context.timepoint)
    hif_prog = _evaluate_program(payload.hif_panel, payload.observations, payload.context.timepoint)
    orth_prog = _evaluate_program(
        payload.orthogonal_state_panel, payload.observations, payload.context.timepoint
    )

    program_statuses: dict[str, ProgramStatus] = {
        "NFKB_ACTIVITY": nfkb_act,
        "NFKB_TARGET_PROGRAM": nfkb_tgt,
        "HIF_PROGRAM": hif_prog,
        "ORTHOGONAL_MACROPHAGE_STATE_PROGRAM": orth_prog,
    }
    for extra_panel in payload.extra_panels:
        program_statuses[extra_panel.program_id] = _evaluate_program(
            extra_panel, payload.observations, payload.context.timepoint
        )

    timepoint_known = _resolve_timepoint_known(payload)
    is_bulk = payload.data_type in {DataType.BULK_RNA_SEQ, DataType.BULK_TISSUE_RNA_SEQ}
    is_purified_macrophage = payload.data_type == DataType.PURIFIED_MACROPHAGE_CULTURE

    proposals_by_h: dict[str, list[tuple[str, DataStatus, str]]] = {
        h_id: [] for h_id in HYPOTHESIS_NAMES
    }
    limitations: list[str] = [
        "Causal attribution requires perturbation or intervention evidence.",
        "HIF and NF-kB pathways can cross-regulate.",
    ]
    for extra_panel in payload.extra_panels:
        extra_status = program_statuses.get(extra_panel.program_id)
        if extra_status == ProgramStatus.NOT_MEASURED:
            limitations.append(
                f"Extra panel '{extra_panel.program_id}' was not measured in current observations."
            )
        elif extra_status == ProgramStatus.INDETERMINATE:
            limitations.append(
                f"Extra panel '{extra_panel.program_id}' has indeterminate marker signals in current observations."
            )
        elif extra_status == ProgramStatus.UP:
            limitations.append(
                f"Extra panel '{extra_panel.program_id}' is active (UP); evaluate its contribution separately from core hypotheses."
            )
        elif extra_status == ProgramStatus.DOWN:
            limitations.append(
                f"Extra panel '{extra_panel.program_id}' is downregulated (DOWN); evaluate its suppression separately from core hypotheses."
            )

    if not timepoint_known:
        limitations.append(
            "Timepoint is unknown; transient signaling dynamics (such as NF-kB activation) cannot be temporally anchored."
        )

    if is_bulk and payload.cell_composition_changed is True:
        _record_status_proposal(
            proposals_by_h,
            "H4",
            "R001",
            DataStatus.COMPATIBLE,
            "Composition change in bulk data is compatible with a population-shift explanation, "
            "but does not by itself prove causality.",
        )
        limitations.append(
            "Bulk data shows composition change; per-cell attribution to H1-H3 remains limited without single-cell or sorted data."
        )
    elif is_bulk and payload.cell_composition_changed is None:
        limitations.append(
            "Bulk RNA-seq with unknown cell composition change: bulk signal cannot separate per-cell activation from composition shift."
        )

    if (
        is_purified_macrophage
        and payload.cell_composition_changed is not True
        and hif_prog == ProgramStatus.UP
        and nfkb_tgt == ProgramStatus.UNCHANGED
        and timepoint_known
    ):
        _record_status_proposal(
            proposals_by_h,
            "H2",
            "R002",
            DataStatus.COMPATIBLE,
            "HIF program is up under hypoxia with known timepoint, consistent with H2 but not proving causality.",
        )
        _record_status_proposal(
            proposals_by_h,
            "H1",
            "R002",
            DataStatus.NOT_SUPPORTED_BY_DATA,
            "NF-kB target program is unchanged at the known timepoint (does not exclude transient earlier activation).",
        )
        limitations.append(
            "NF-kB activation may be transient; an unchanged signature at the measured timepoint does not exclude earlier initiation."
        )

    if (
        nfkb_act == ProgramStatus.UP
        and nfkb_tgt == ProgramStatus.UP
        and hif_prog == ProgramStatus.UP
    ):
        for h_id in ("H1", "H2"):
            _record_status_proposal(
                proposals_by_h,
                h_id,
                "R003",
                DataStatus.COMPATIBLE,
                "Both NF-kB and HIF programs are up; pathways may cooperate and are not separable from current data.",
            )

    if (
        nfkb_act == ProgramStatus.UP
        and nfkb_tgt == ProgramStatus.UP
        and hif_prog != ProgramStatus.UP
    ):
        _record_status_proposal(
            proposals_by_h,
            "H1",
            "R004",
            DataStatus.COMPATIBLE,
            "NF-kB activity and target program are up while HIF program is not elevated.",
        )
        if hif_prog == ProgramStatus.NOT_MEASURED:
            _record_status_proposal(
                proposals_by_h,
                "H2",
                "R004",
                DataStatus.NOT_ASSESSED,
                "HIF program was not measured.",
            )

    if orth_prog == ProgramStatus.UP and payload.cell_composition_changed is not True:
        _record_status_proposal(
            proposals_by_h,
            "H3",
            "R005",
            DataStatus.COMPATIBLE,
            "Orthogonal macrophage state program is up without confirmed composition shift.",
        )
    elif orth_prog == ProgramStatus.NOT_MEASURED:
        limitations.append("No orthogonal macrophage-state markers were measured to assess H3.")

    evaluations: dict[str, HypothesisEvaluation] = {}
    rule_conflicts: list[RuleConflict] = []

    for h_id in HYPOTHESIS_NAMES:
        ev, h_conflicts = _merge_hypothesis_proposals(
            h_id=h_id,
            proposals=proposals_by_h[h_id],
            evidence_status=ev_statuses[h_id],
            evidence_conflict=ev_conflicts[h_id],
            supporting_ids=ev_supporting[h_id],
            contradicting_ids=ev_contradicting[h_id],
        )
        evaluations[h_id] = ev
        rule_conflicts.extend(h_conflicts)

    measurements = _build_discriminating_measurements(payload, program_statuses)
    pairwise_relations = _build_pairwise_relations(evaluations, program_statuses, measurements)

    has_non_separable_h1_h2 = any(
        rel.hypotheses == ("H1", "H2")
        and rel.relation == PairwiseRelationType.NON_EXCLUSIVE_NOT_SEPARABLE
        for rel in pairwise_relations
    )

    for h_id, ev in evaluations.items():
        if ev.data_status == DataStatus.CONTRADICTED:
            ev.decision_status = DecisionStatus.CONTRADICTED
        elif ev.data_status == DataStatus.NOT_ASSESSED:
            ev.decision_status = DecisionStatus.NOT_ASSESSED
        elif ev.data_status == DataStatus.NOT_SUPPORTED_BY_DATA:
            ev.decision_status = DecisionStatus.NOT_SUPPORTED_BY_DATA
        elif h_id == "H4" and ev.data_status == DataStatus.COMPATIBLE:
            ev.decision_status = DecisionStatus.WEAKLY_SUPPORTED
        elif h_id in {"H1", "H2"} and has_non_separable_h1_h2 and ev.data_status == DataStatus.COMPATIBLE:
            ev.decision_status = DecisionStatus.NON_EXCLUSIVE_NOT_SEPARABLE
        elif ev.data_status in {DataStatus.SUPPORTED, DataStatus.COMPATIBLE}:
            ev.decision_status = DecisionStatus.WEAKLY_SUPPORTED

    has_direct_evidence_support = any(
        status == EvidenceStatus.SUPPORTED for status in ev_statuses.values()
    )
    has_data_support_or_compat = any(
        ev.data_status in {DataStatus.SUPPORTED, DataStatus.COMPATIBLE}
        for ev in evaluations.values()
    )
    has_data_distinguishability = (
        any(
            rel.relation
            in {
                PairwiseRelationType.DISTINGUISHABLE_WITH_CURRENT_DATA,
                PairwiseRelationType.INDISTINGUISHABLE_WITH_CURRENT_DATA,
            }
            for rel in pairwise_relations
        )
        or any(ev.data_status == DataStatus.NOT_SUPPORTED_BY_DATA for ev in evaluations.values())
    )
    has_blocking_conflict = bool(rule_conflicts) or any(ev_conflicts.values()) or bool(all_context_conflicts)
    missing_key_context_for_conditional = (
        payload.data_type is None
        or _is_missing(payload.context.condition.oxygen_level)
        or not timepoint_known
    )

    if (
        not missing_mandatory
        and not all_context_conflicts
        and not missing_key_context_for_conditional
        and not has_blocking_conflict
        and has_direct_evidence_support
        and has_data_support_or_compat
        and has_data_distinguishability
    ):
        mode = InferenceMode.CONDITIONAL_INFERENCE
    else:
        mode = InferenceMode.WEAK_INFERENCE

    return HypothesisEngineOutput(
        query_id=payload.query_id,
        scenario=payload.scenario,
        rule_set_version=RULE_SET_VERSION,
        mode=mode,
        abstention_reason=None,
        missing_mandatory_fields=missing_mandatory,
        missing_context_fields=missing_context,
        context_conflicts=all_context_conflicts,
        program_statuses=program_statuses,
        hypotheses=evaluations,
        rule_conflicts=rule_conflicts,
        pairwise_relations=pairwise_relations,
        discriminating_measurements=measurements,
        limitations=limitations,
        model_prior_notes=list(payload.model_prior_notes),
        il1_pathway=_pilot_il1,
        pilot_hypotheses=_pilot_hyps,
    )

