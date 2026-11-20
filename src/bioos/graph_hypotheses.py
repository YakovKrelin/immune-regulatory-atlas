from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from bioos.regulatory_graph import CausalCandidatePath, RegulatoryGraphResult


class GraphHypothesisCandidate(BaseModel):
    """Кандидат гипотезы, построенный из ранжированного регуляторного пути."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    hypothesis_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    chain: tuple[str, ...] = Field(..., min_length=2)
    path_score: float = Field(..., ge=0.0)
    has_conflicts: bool = False
    summary: str = Field(..., min_length=1)


class RegulatoryGraphSnapshot(BaseModel):
    """Снапшот результата графа и гипотез с проверкой целостности SHA-256."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "v0.2"
    sha256: str = Field(..., min_length=64, max_length=64)
    graph_result: RegulatoryGraphResult
    hypotheses: tuple[GraphHypothesisCandidate, ...] = Field(default_factory=tuple)


GraphHypothesisCandidate.model_rebuild()
RegulatoryGraphSnapshot.model_rebuild()


def _format_chain_summary(path: CausalCandidatePath) -> str:
    steps = [
        f"{edge.source} -[{edge.direction.value}:{edge.edge_id}]-> {edge.target}"
        for edge in path.edges
    ]
    conflict_note = " [contains regulatory conflict]" if path.has_conflicts else ""
    return f"{' ; '.join(steps)} (score={path.path_score}){conflict_note}"


def build_graph_hypotheses(
    graph_result: RegulatoryGraphResult,
    max_hypotheses: int = 5,
) -> tuple[GraphHypothesisCandidate, ...]:
    """Строит кандидаты гипотез из `candidate_paths` результата графа."""
    hypotheses: list[GraphHypothesisCandidate] = []
    for idx, path in enumerate(graph_result.candidate_paths[:max_hypotheses], start=1):
        hypotheses.append(
            GraphHypothesisCandidate(
                hypothesis_id=f"hyp_{path.source}_{path.target}_{idx:02d}",
                source=path.source,
                target=path.target,
                chain=path.nodes,
                path_score=path.path_score,
                has_conflicts=path.has_conflicts,
                summary=_format_chain_summary(path),
            )
        )
    return tuple(hypotheses)


def compute_snapshot_sha256(
    graph_result: RegulatoryGraphResult,
    hypotheses: tuple[GraphHypothesisCandidate, ...],
) -> str:
    """Вычисляет детерминированный SHA-256 для связки графа и гипотез."""
    canonical_body = {
        "schema_version": "v0.2",
        "graph_result": graph_result.model_dump(mode="json"),
        "hypotheses": [item.model_dump(mode="json") for item in hypotheses],
    }
    encoded = json.dumps(
        canonical_body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_regulatory_graph_snapshot(
    graph_result: RegulatoryGraphResult,
    max_hypotheses: int = 5,
) -> RegulatoryGraphSnapshot:
    """Создаёт снапшот `v0.2` с контрольной суммой SHA-256."""
    hypotheses = build_graph_hypotheses(graph_result, max_hypotheses=max_hypotheses)
    digest = compute_snapshot_sha256(graph_result, hypotheses)
    return RegulatoryGraphSnapshot(
        schema_version="v0.2",
        sha256=digest,
        graph_result=graph_result,
        hypotheses=hypotheses,
    )