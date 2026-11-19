from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from bioos.engine.graph_causality import get_neighborhood
from bioos.engine.hypotheses_export import build_hypotheses_from_signals
from bioos.models import (
    Assertion,
    BiologicalState,
    Module,
    ProvProvenance,
    ReportSnapshot,
    SnapshotRequest,
)


def _sha256_canonical(obj: object) -> str:
    canonical_bytes = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def build_report_snapshot(
    request: SnapshotRequest,
    assertions: list[Assertion],
    states: list[BiologicalState],
    modules: list[Module],
) -> ReportSnapshot:
    now_iso = datetime.now(timezone.utc).isoformat()
    snapshot_uuid = str(uuid4())
    snapshot_id = f"BIOOS:SNAP:{snapshot_uuid[:12]}"
    activity_id = f"bioos:activity:{snapshot_uuid[:12]}"
    report_entity_id = f"bioos:entity:{snapshot_uuid[:12]}"

    kb_bundle = {
        "assertions": [a.model_dump() for a in assertions],
        "states": [s.model_dump() for s in states],
        "modules": [m.model_dump() for m in modules],
    }
    kb_hash = _sha256_canonical(kb_bundle)

    neighborhood = get_neighborhood(assertions, request.start_node, request.max_hops)
    hypotheses_result = build_hypotheses_from_signals(request.signals, assertions)

    core_analytical_body = {
        "title": request.title,
        "parameters": {
            "signals": request.signals,
            "start_node": request.start_node,
            "max_hops": request.max_hops,
        },
        "knowledge_base_sha256": kb_hash,
        "neighborhood": neighborhood,
        "hypotheses": hypotheses_result,
    }
    payload_hash = _sha256_canonical(core_analytical_body)

    prov = ProvProvenance(
        context={
            "prov": "http://www.w3.org/ns/prov#",
            "bioos": "https://bioos.org/ontology/",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        },
        activity_id=activity_id,
        generated_at=now_iso,
        was_associated_with={
            "agent_id": f"bioos:agent:{request.actor.replace(' ', '_')}",
            "agent_type": "prov:SoftwareAgent",
            "label": request.actor,
        },
        used_entities=[
            {
                "entity_id": "bioos:kb:assertions_states_modules",
                "entity_type": "prov:Entity",
                "sha256": kb_hash,
                "assertion_count": str(len(assertions)),
            },
            {
                "entity_id": "bioos:input:signals_and_graph_params",
                "entity_type": "prov:Entity",
                "sha256": _sha256_canonical(core_analytical_body["parameters"]),
            },
        ],
        generated_entity={
            "entity_id": report_entity_id,
            "entity_type": "bioos:ReproducibilitySnapshot",
            "sha256": payload_hash,
            "was_generated_by": activity_id,
        },
    )

    summary = {
        "graph_nodes_count": len(neighborhood.get("nodes", [])),
        "graph_edges_count": len(neighborhood.get("edges", [])),
        "parsed_signals": hypotheses_result.get("signals", []),
        "primary_hypothesis": hypotheses_result.get("primary_hypothesis", ""),
        "conflicts_detected": len(hypotheses_result.get("conflicts", [])),
    }

    return ReportSnapshot(
        snapshot_id=snapshot_id,
        title=request.title,
        created_at=now_iso,
        canonical_payload_sha256=payload_hash,
        knowledge_base_sha256=kb_hash,
        analytical_scope_note=(
            "Analytical reproducibility snapshot with SHA-256 verification and PROV-O lineage; "
            "intended for research analysis, not clinical diagnosis."
        ),
        summary=summary,
        prov_o=prov,
    )