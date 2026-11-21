from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from bioos.graph_hypotheses import (
    RegulatoryGraphSnapshot,
    create_regulatory_graph_snapshot,
)
from bioos.regulatory_graph import (
    RegulatoryEdge,
    SubgraphQuery,
    build_regulatory_graph_result,
)

router = APIRouter(prefix="/v2", tags=["v0.2"])


class RegulatoryGraphSnapshotRequest(BaseModel):
    """Запрос на построение регуляторного подграфа, гипотез и снапшота v0.2."""

    model_config = ConfigDict(frozen=True)

    edges: tuple[RegulatoryEdge, ...] = Field(default_factory=tuple)
    query: SubgraphQuery
    targets: tuple[str, ...] | None = None
    max_hypotheses: int = Field(default=5, ge=1, le=50)


RegulatoryGraphSnapshotRequest.model_rebuild()


@router.post("/regulatory-graph/snapshots", response_model=RegulatoryGraphSnapshot)
def create_v2_regulatory_graph_snapshot(
    payload: RegulatoryGraphSnapshotRequest,
) -> RegulatoryGraphSnapshot:
    graph_result = build_regulatory_graph_result(
        edges=payload.edges,
        query=payload.query,
        targets=payload.targets,
    )
    return create_regulatory_graph_snapshot(
        graph_result=graph_result,
        max_hypotheses=payload.max_hypotheses,
    )

import hmac

from bioos.graph_hypotheses import compute_snapshot_sha256


class SnapshotVerificationResponse(BaseModel):
    """SHA-256 integrity verification result for a v0.2 snapshot."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    schema_version: str
    provided_sha256: str
    computed_sha256: str


SnapshotVerificationResponse.model_rebuild()


@router.post(
    "/regulatory-graph/snapshots/verify",
    response_model=SnapshotVerificationResponse,
)
def verify_v2_regulatory_graph_snapshot(
    snapshot: RegulatoryGraphSnapshot,
) -> SnapshotVerificationResponse:
    computed = compute_snapshot_sha256(
        snapshot.graph_result,
        snapshot.hypotheses,
    )
    provided_normalized = snapshot.sha256.strip().lower()
    is_valid = hmac.compare_digest(provided_normalized, computed.lower())
    return SnapshotVerificationResponse(
        valid=is_valid,
        schema_version=snapshot.schema_version,
        provided_sha256=snapshot.sha256,
        computed_sha256=computed,
    )
