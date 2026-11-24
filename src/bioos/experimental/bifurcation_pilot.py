from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class AbsenceStatus(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    NO_CHANGE = "no_change"
    MIXED = "mixed"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    EXTRACTED = "extracted"
    REVIEWED = "reviewed"
    CONFLICTED = "conflicted"
    REPRODUCED = "reproduced"
    DEPRECATED = "deprecated"


class InferenceMode(str, Enum):
    CONDITIONAL_INFERENCE = "conditional_inference"
    WEAK_INFERENCE = "weak_inference"
    HARD_ABSTENTION = "hard_abstention"


CompatibilityLevel = Literal["high", "moderate", "low", "indeterminate"]


@dataclass(frozen=True)
class ContextField:
    value: str | None
    status: AbsenceStatus = AbsenceStatus.PRESENT

    @classmethod
    def from_value(
        cls,
        value: str | None,
        fallback: AbsenceStatus = AbsenceStatus.MISSING,
    ) -> "ContextField":
        if value is None or not str(value).strip():
            return cls(value=None, status=fallback)
        return cls(value=str(value).strip(), status=AbsenceStatus.PRESENT)


@dataclass(frozen=True)
class ContextSpec:
    species: ContextField
    tissue_or_cell: ContextField
    condition_or_stimulus: ContextField
    timepoint: ContextField = field(default_factory=lambda: ContextField(None, AbsenceStatus.MISSING))
    duration: ContextField = field(default_factory=lambda: ContextField(None, AbsenceStatus.MISSING))
    dose_raw: str | None = None
    dose_normalized: str | None = None
    assay: str | None = None
    measurement_level: str | None = None # rna | protein | phenotype | other
    donor_or_batch: str | None = None


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source_ref: str
    subject: str
    relation_or_direction: Direction
    target: str
    status: EvidenceStatus = EvidenceStatus.EXTRACTED

    study_type: str | None = None
    measurement_level: str | None = None
    sample_size: int | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Observation:
    marker_or_entity: str
    direction: Direction
    measurement_level: str | None = None


@dataclass
class HypothesisBranch:
    hypothesis_id: str
    title: str
    summary: str
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    related_path_refs: list[str] = field(default_factory=list)
    compatibility_level: CompatibilityLevel = "indeterminate"
    uncertainty_notes: list[str] = field(default_factory=list)
    distinguishing_measurements: list[str] = field(default_factory=list)


@dataclass
class ReasoningReport:
    mode: InferenceMode
    hypotheses: list[HypothesisBranch]
    conflicts: list[str]
    missing_context_or_data: list[str]
    recommended_next_tests: list[str]
    snapshot_ref: str


def evaluate_pilot_scenario(
    observations: list[Observation],
    context: ContextSpec,
    evidence_items: list[EvidenceItem],
    snapshot_ref: str = "sha256:unpinned",
) -> ReasoningReport:
    conflicts: list[str] = []
    missing: list[str] = []

    if not observations:
        return ReasoningReport(
            mode=InferenceMode.HARD_ABSTENTION,
            hypotheses=[],
            conflicts=[],
            missing_context_or_data=["observations (no input observations or markers provided)"],
            recommended_next_tests=["Provide observed markers or changes along with the experimental context."],
            snapshot_ref=snapshot_ref,
        )

    if context.species.status != AbsenceStatus.PRESENT:
        missing.append(f"context.species ({context.species.status.value})")
    if context.tissue_or_cell.status != AbsenceStatus.PRESENT:
        missing.append(f"context.tissue_or_cell ({context.tissue_or_cell.status.value})")
    if context.condition_or_stimulus.status != AbsenceStatus.PRESENT:
        missing.append(f"context.condition_or_stimulus ({context.condition_or_stimulus.status.value})")
    if context.timepoint.status != AbsenceStatus.PRESENT and context.duration.status != AbsenceStatus.PRESENT:
        missing.append("context.timepoint / context.duration (neither timepoint nor duration provided)")
    if context.measurement_level is None:
        missing.append("measurement_level (measurement level not specified: RNA, protein, or phenotype)")

    obs_map = {obs.marker_or_entity.upper(): obs.direction for obs in observations}
    up_markers = {k for k, v in obs_map.items() if v == Direction.UP}

    for ev in evidence_items:
        if ev.status == EvidenceStatus.CONFLICTED:
            conflicts.append(f"Conflict in {ev.source_ref} ({ev.evidence_id}): {ev.subject} -> {ev.target}")
        marker_key = ev.target.upper()
        if (
            marker_key in obs_map
            and obs_map[marker_key] != Direction.UNKNOWN
            and ev.relation_or_direction != obs_map[marker_key]
        ):
            if ev.relation_or_direction not in (Direction.MIXED, Direction.CONFLICTED):
                conflicts.append(
                    f"Conflict on {marker_key}: observed={obs_map[marker_key].value}, "
                    f"evidence {ev.evidence_id}={ev.relation_or_direction.value}"
                )

    has_tnf_il6 = "TNF" in up_markers and "IL6" in up_markers
    has_ifng_cxcl10 = "IFNG" in up_markers and "CXCL10" in up_markers

    h1 = HypothesisBranch(
        hypothesis_id="H1",
        title="Mixed inflammatory and interferon response",
        summary="Concurrent inflammatory (TNF/IL6) and interferon (IFNG/CXCL10) activation under hypoxia or stress.",
        supporting_evidence_ids=[e.evidence_id for e in evidence_items if e.relation_or_direction == Direction.UP],
        contradicting_evidence_ids=[e.evidence_id for e in evidence_items if e.status == EvidenceStatus.CONFLICTED],
        related_path_refs=["path:TNF->NFKB->IL6", "path:IFNG->STAT1->CXCL10"],
        compatibility_level="high" if (has_tnf_il6 and has_ifng_cxcl10) else "moderate",
        uncertainty_notes=[
            "Bulk measurements do not separate intracellular signaling from shifts in population composition.",
            "Baseline or prior activation state may confound stimulus attribution.",
        ],
        distinguishing_measurements=[
            "Measure ISG15, MX1, and OAS1 expression",
            "Assess cellular composition and protein abundance (not RNA alone)",
        ],
    )

    h2 = HypothesisBranch(
        hypothesis_id="H2",
        title="Infectious or pathogen-driven stimulus",
        summary="Profile aligns with pathogen or nucleic acid sensing driving an interferon and cytokine response.",
        supporting_evidence_ids=[e.evidence_id for e in evidence_items if e.target.upper() in {"IFNG", "CXCL10", "IL6"}],
        contradicting_evidence_ids=[],
        related_path_refs=["path:PRR/IFN->CXCL10", "path:inflammatory->IL6"],
        compatibility_level="moderate" if has_ifng_cxcl10 else "low",
        uncertainty_notes=[
            "Pathogen nucleic acid or viral stimulus has not been directly assayed.",
        ],
        distinguishing_measurements=[
            "Viral PCR or check for infectious stimulus",
            "Measure ISG15, MX1, and OAS1 expression",
        ],
    )

    h3 = HypothesisBranch(
        hypothesis_id="H3",
        title="Cellular composition or activation-state shift",
        summary="Observed signal may reflect shifts in activated subpopulation fractions or prior activation state.",
        supporting_evidence_ids=[e.evidence_id for e in evidence_items if e.target.upper() in {"TNF", "IL6", "CXCL10"}],
        contradicting_evidence_ids=[],
        related_path_refs=["path:population_shift->bulk_signal"],
        compatibility_level="moderate" if context.tissue_or_cell.value in {"PBMC", "macrophage", "PBMC / macrophage"} else "low",
        uncertainty_notes=[
            "Bulk measurements do not separate intracellular signaling from shifts in population composition.",
        ],
        distinguishing_measurements=[
            "Assess cellular composition and subpopulation phenotype",
            "Control for baseline or prior activation state and protein level",
        ],
    )

    hypotheses = [h1, h2, h3]

    compatible_count = sum(1 for h in hypotheses if h.compatibility_level in {"high", "moderate"})
    if compatible_count >= 2 or conflicts:
        mode = InferenceMode.WEAK_INFERENCE
    else:
        mode = InferenceMode.CONDITIONAL_INFERENCE

    recommended_tests = [
        "Viral PCR or check for infectious stimulus",
        "Measure ISG15, MX1, and OAS1 expression",
        "Assess cellular composition and protein abundance (not RNA alone)",
        "Repeat time-course measurements (separating timepoint and duration)",
    ]

    return ReasoningReport(
        mode=mode,
        hypotheses=hypotheses,
        conflicts=conflicts,
        missing_context_or_data=missing,
        recommended_next_tests=recommended_tests,
        snapshot_ref=snapshot_ref,
    )


if __name__ == "__main__":
    ctx = ContextSpec(
        species=ContextField.from_value("Homo sapiens"),
        tissue_or_cell=ContextField.from_value("PBMC / macrophage"),
        condition_or_stimulus=ContextField.from_value("hypoxia"),
        timepoint=ContextField.from_value("12h"),
        duration=ContextField(None, AbsenceStatus.UNKNOWN),
    )
    obs = [
        Observation("TNF", Direction.UP),
        Observation("IL6", Direction.UP),
        Observation("IFNG", Direction.UP),
        Observation("CXCL10", Direction.UP),
    ]
    ev = [
        EvidenceItem("ev_1", "pmid:example_1", "TNF", Direction.UP, "IL6", EvidenceStatus.REVIEWED),
        EvidenceItem("ev_2", "pmid:example_2", "IFNG", Direction.UP, "CXCL10", EvidenceStatus.REVIEWED),
    ]

    report = evaluate_pilot_scenario(obs, ctx, ev, snapshot_ref="sha256:7299b2f_v0.2.6")
    print("Mode:", report.mode.value)
    print("Missing context/data:", report.missing_context_or_data)
    print("Conflicts:", report.conflicts)
    for h in report.hypotheses:
        print(f"- [{h.hypothesis_id}] {h.title} | compatibility={h.compatibility_level}")
    print("Next tests:", report.recommended_next_tests)
