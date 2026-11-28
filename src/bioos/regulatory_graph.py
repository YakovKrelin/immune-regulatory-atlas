from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RegulationDirection(str, Enum):
    """Direction of influence for a regulatory edge."""

    ACTIVATION = "activation"
    INHIBITION = "inhibition"
    MODULATION = "modulation"


class ConflictPolicy(str, Enum):
    """Resolution policy for conflicting regulatory edges."""

    KEEP_FLAGGED = "keep_flagged"
    PREFER_HIGHER_SCORE = "prefer_higher_score"
    DROP_CONFLICTING = "drop_conflicting"


class RegulatoryEdge(BaseModel):
    """Directed regulatory edge between two biological entities."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    edge_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    direction: RegulationDirection = RegulationDirection.MODULATION
    tissue: str | None = None
    context: str | None = None
    evidence_codes: tuple[str, ...] = Field(default_factory=tuple)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    conflict_flag: bool = False


class SubgraphQuery(BaseModel):
    """Parameters for filtering and extracting a contextual subgraph."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    seed_entities: tuple[str, ...] = Field(..., min_length=1)
    max_hops: int = Field(default=2, ge=1, le=6)
    tissue: str | None = None
    context: str | None = None
    required_evidence_codes: tuple[str, ...] = Field(default_factory=tuple)
    min_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    conflict_policy: ConflictPolicy = ConflictPolicy.KEEP_FLAGGED


class CausalCandidatePath(BaseModel):
    """Directed candidate path for causal inference and hypothesis generation."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    nodes: tuple[str, ...] = Field(..., min_length=2)
    edges: tuple[RegulatoryEdge, ...] = Field(..., min_length=1)
    path_score: float = Field(default=0.0, ge=0.0)
    has_conflicts: bool = False


class RegulatoryGraphResult(BaseModel):
    """Aggregated subgraph analysis result, conflicts, and candidate paths."""

    model_config = ConfigDict(frozen=True)

    query: SubgraphQuery
    nodes: tuple[str, ...] = Field(default_factory=tuple)
    edges: tuple[RegulatoryEdge, ...] = Field(default_factory=tuple)
    candidate_paths: tuple[CausalCandidatePath, ...] = Field(default_factory=tuple)
    resolved_conflicts_count: int = Field(default=0, ge=0)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


RegulatoryEdge.model_rebuild()
SubgraphQuery.model_rebuild()
CausalCandidatePath.model_rebuild()
RegulatoryGraphResult.model_rebuild()


def _normalize_counter_signs(
    counter_sign_hits: list[str] | tuple[str, ...] | None,
) -> set[str]:
    """Normalize counter-sign labels and leading entity symbols to uppercase."""
    if not counter_sign_hits:
        return set()
    normalized: set[str] = set()
    for raw in counter_sign_hits:
        cleaned = str(raw or "").strip().upper()
        if not cleaned:
            continue
        normalized.add(cleaned)
        token = cleaned.split()[0]
        normalized.add(token)
    return normalized


def _compute_adjusted_path_score(
    current_edges: list[RegulatoryEdge],
    score_acc: float,
    counter_signs: set[str],
) -> float:
    """Compute length-penalized path score adjusted for counter-sign hits."""
    length_penalty = 1.0 / (1.0 + 0.1 * (len(current_edges) - 1))
    base_score = score_acc * length_penalty

    if not counter_signs:
        return round(max(0.0, base_score), 6)

    matched_counter = 0
    for edge in current_edges:
        src = edge.source.upper()
        tgt = edge.target.upper()
        if src in counter_signs or tgt in counter_signs:
            matched_counter += 1

    counter_multiplier = max(0.25, 1.0 - 0.15 * matched_counter)
    return round(max(0.0, base_score * counter_multiplier), 6)


def extract_contextual_subgraph(
    edges: list[RegulatoryEdge] | tuple[RegulatoryEdge, ...],
    query: SubgraphQuery,
) -> tuple[tuple[str, ...], tuple[RegulatoryEdge, ...]]:
    """
    Filter edges by tissue, context, evidence codes, and confidence threshold,
    then traverse from `seed_entities` within `max_hops`.
    """
    filtered: list[RegulatoryEdge] = []
    required_eco = set(query.required_evidence_codes)

    for edge in edges:
        if edge.confidence_score < query.min_confidence_score:
            continue
        if query.tissue is not None and edge.tissue != query.tissue:
            continue
        if query.context is not None and edge.context != query.context:
            continue
        if required_eco and not required_eco.issubset(edge.evidence_codes):
            continue
        filtered.append(edge)

    visited_nodes: set[str] = set(query.seed_entities)
    frontier: set[str] = set(query.seed_entities)
    selected_edges: dict[str, RegulatoryEdge] = {}

    for _ in range(query.max_hops):
        next_frontier: set[str] = set()
        for edge in filtered:
            if edge.source in frontier:
                selected_edges[edge.edge_id] = edge
                if edge.target not in visited_nodes:
                    visited_nodes.add(edge.target)
                    next_frontier.add(edge.target)
        frontier = next_frontier
        if not frontier:
            break

    sorted_nodes = tuple(sorted(visited_nodes))
    sorted_edges = tuple(
        sorted(selected_edges.values(), key=lambda item: (-item.confidence_score, item.edge_id))
    )
    return sorted_nodes, sorted_edges


def detect_and_resolve_conflicts(
    edges: list[RegulatoryEdge] | tuple[RegulatoryEdge, ...],
    policy: ConflictPolicy = ConflictPolicy.KEEP_FLAGGED,
) -> tuple[tuple[RegulatoryEdge, ...], int]:
    """
    Detect edge groups sharing `(source, target, tissue, context)` with opposing
    regulation directions (`activation` vs `inhibition`) and apply `policy`.
    """
    grouped: dict[tuple[str, str, str | None, str | None], list[RegulatoryEdge]] = {}
    for edge in edges:
        key = (edge.source, edge.target, edge.tissue, edge.context)
        grouped.setdefault(key, []).append(edge)

    output: list[RegulatoryEdge] = []
    resolved_count = 0

    for group in grouped.values():
        directions = {item.direction for item in group}
        is_conflicting = (
            RegulationDirection.ACTIVATION in directions
            and RegulationDirection.INHIBITION in directions
        )

        if not is_conflicting:
            output.extend(group)
            continue

        resolved_count += 1
        if policy == ConflictPolicy.DROP_CONFLICTING:
            continue

        if policy == ConflictPolicy.PREFER_HIGHER_SCORE:
            best = max(group, key=lambda item: (item.confidence_score, item.edge_id))
            output.append(best.model_copy(update={"conflict_flag": False}))
            continue

        for item in group:
            output.append(item.model_copy(update={"conflict_flag": True}))

    sorted_output = tuple(
        sorted(output, key=lambda item: (-item.confidence_score, item.edge_id))
    )
    return sorted_output, resolved_count


def rank_candidate_paths(
    edges: list[RegulatoryEdge] | tuple[RegulatoryEdge, ...],
    sources: tuple[str, ...],
    targets: tuple[str, ...] | None = None,
    max_depth: int = 3,
    top_k: int = 10,
    counter_sign_hits: list[str] | tuple[str, ...] | None = None,
) -> tuple[CausalCandidatePath, ...]:
    """
    Build simple directed paths from `sources` and rank them by adjusted
    `confidence_score` product accounting for path length and `counter_sign_hits`.
    """
    adjacency: dict[str, list[RegulatoryEdge]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge)

    target_set = set(targets) if targets else None
    counter_signs = _normalize_counter_signs(counter_sign_hits)
    paths: list[CausalCandidatePath] = []

    def dfs(
        current: str,
        current_nodes: list[str],
        current_edges: list[RegulatoryEdge],
        score_acc: float,
    ) -> None:
        if current_edges:
            is_valid_endpoint = target_set is None or current in target_set
            if is_valid_endpoint:
                adjusted_score = _compute_adjusted_path_score(
                    current_edges=current_edges,
                    score_acc=score_acc,
                    counter_signs=counter_signs,
                )
                has_counter_hit = any(
                    edge.source.upper() in counter_signs or edge.target.upper() in counter_signs
                    for edge in current_edges
                )
                paths.append(
                    CausalCandidatePath(
                        source=current_nodes[0],
                        target=current,
                        nodes=tuple(current_nodes),
                        edges=tuple(current_edges),
                        path_score=adjusted_score,
                        has_conflicts=any(e.conflict_flag for e in current_edges) or has_counter_hit,
                    )
                )

        if len(current_edges) >= max_depth:
            return

        for edge in adjacency.get(current, ()):
            if edge.target in current_nodes:
                continue
            dfs(
                edge.target,
                [*current_nodes, edge.target],
                [*current_edges, edge],
                score_acc * edge.confidence_score,
            )

    for seed in sources:
        dfs(seed, [seed], [], 1.0)

    paths.sort(
        key=lambda p: (
            -p.path_score,
            p.has_conflicts,
            len(p.edges),
            p.source,
            p.target,
        )
    )
    return tuple(paths[:top_k])


def build_regulatory_graph_result(
    edges: list[RegulatoryEdge] | tuple[RegulatoryEdge, ...],
    query: SubgraphQuery,
    targets: tuple[str, ...] | None = None,
    counter_sign_hits: list[str] | tuple[str, ...] | None = None,
) -> RegulatoryGraphResult:
    """Assemble subgraph analysis result for API routes, hypotheses, and snapshots."""
    nodes, subgraph_edges = extract_contextual_subgraph(edges, query)
    resolved_edges, resolved_conflicts_count = detect_and_resolve_conflicts(
        subgraph_edges,
        policy=query.conflict_policy,
    )
    candidate_paths = rank_candidate_paths(
        resolved_edges,
        sources=query.seed_entities,
        targets=targets,
        max_depth=query.max_hops,
        counter_sign_hits=counter_sign_hits,
    )

    warnings: list[str] = []
    if not resolved_edges:
        warnings.append("No regulatory edges matched the query and conflict policy.")

    return RegulatoryGraphResult(
        query=query,
        nodes=nodes,
        edges=resolved_edges,
        candidate_paths=candidate_paths,
        resolved_conflicts_count=resolved_conflicts_count,
        warnings=tuple(warnings),
    )