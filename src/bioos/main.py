from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from bioos.engine.filters import filter_assertions

app = FastAPI(
    title="BIOOS Human Regulatory Atlas API",
    version="1.0.0",
    description="Verifiable regulatory assertions, causal graph traversal, dataset evaluation, and SHA-256 snapshots.",
)

ECO_RANK: dict[str, int] = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
}


class DatasetEvalRequest(BaseModel):
    name: str
    modality: str = "RNA-seq"
    tissue: str | None = None
    cell_type: str | None = None
    signals: list[str] | str = Field(default_factory=list)


class SnapshotCreateRequest(BaseModel):
    title: str
    signals: list[str] | str = Field(default_factory=list)
    start_node: str = "TNF"
    max_hops: int = 2
    actor: str = "BIOOS-Check-Runner"
    tissue: str | None = None
    cell_type: str | None = None
    pathology: str | None = None
    physiology: str | None = None
    stimulus: str | None = None
    eco_levels: list[str] | None = None
    min_eco: str | None = None
    verify: bool = True


class SnapshotVerifyRequest(BaseModel):
    snapshot: dict[str, Any] | None = None
    file_path: str | None = None


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_obj(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _split_csv_param(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    expanded: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            cleaned = part.strip()
            if cleaned:
                expanded.append(cleaned)
    return expanded or None


def _parse_signals(signals: list[str] | str | None) -> list[str]:
    if signals is None:
        return []
    if isinstance(signals, str):
        return [s.strip() for s in signals.split(",") if s.strip()]
    parsed: list[str] = []
    for item in signals:
        for part in str(item).split(","):
            cleaned = part.strip()
            if cleaned:
                parsed.append(cleaned)
    return parsed


def _as_assertion_dicts(items: Any) -> list[dict[str, Any]]:
    if items is None:
        return []
    result: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            result.append(item)
        elif hasattr(item, "model_dump"):
            result.append(item.model_dump())
        elif hasattr(item, "dict"):
            result.append(item.dict())
        elif hasattr(item, "__dict__"):
            result.append(dict(vars(item)))
    return result


def _load_initial_assertions() -> list[dict[str, Any]]:
    """Loads assertions from a JSON store if present, otherwise seeds reference assertions."""
    candidate_paths = [
        Path(__file__).resolve().parents[2] / "data" / "assertions.json",
        Path(__file__).resolve().parents[2] / "src" / "bioos" / "data" / "assertions.json",
    ]
    for path in candidate_paths:
        if path.exists():
            raw_data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw_data, dict) and "assertions" in raw_data:
                return _as_assertion_dicts(raw_data["assertions"])
            if isinstance(raw_data, list):
                return _as_assertion_dicts(raw_data)

    return [
        {
            "id": "BIOOS:A0004871",
            "source": "TNF",
            "target": "IL6",
            "relation": "activates",
            "eco": "A",
            "conflict": False,
            "context": {
                "organism": "Homo sapiens",
                "tissue": "Blood",
                "cell_type": "Monocyte",
                "pathology": "Viral infection",
                "physiology": "Acute immune response",
                "stimulus": "TNF",
            },
            "provenance": {
                "prov_type": "prov:Entity",
                "derived_from": "curated_immunology_kb_v1",
            },
        },
        {
            "id": "BIOOS:A0004874",
            "source": "IFNG",
            "target": "CXCL10",
            "relation": "induces",
            "eco": "A",
            "conflict": False,
            "context": {
                "organism": "Homo sapiens",
                "tissue": "Blood",
                "cell_type": "Monocyte",
                "pathology": "Viral infection",
                "physiology": "Antiviral signaling",
                "stimulus": "IFNG",
            },
            "provenance": {
                "prov_type": "prov:Entity",
                "derived_from": "curated_immunology_kb_v1",
            },
        },
        {
            "id": "BIOOS:A0004880",
            "source": "TNF",
            "target": "IL6",
            "relation": "activates",
            "eco": "A",
            "conflict": False,
            "context": {
                "organism": "Homo sapiens",
                "tissue": "Liver",
                "cell_type": "Kupffer cell",
                "pathology": "Inflammation",
                "physiology": "Acute phase response",
                "stimulus": "TNF",
            },
            "provenance": {
                "prov_type": "prov:Entity",
                "derived_from": "curated_immunology_kb_v1",
            },
        },
        {
            "id": "BIOOS:A0004881",
            "source": "IL6",
            "target": "STAT3",
            "relation": "activates",
            "eco": "A",
            "conflict": False,
            "context": {
                "organism": "Homo sapiens",
                "tissue": "Liver",
                "cell_type": "Hepatocyte",
                "pathology": "Inflammation",
                "physiology": "Acute phase response",
                "stimulus": "IL6",
            },
            "provenance": {
                "prov_type": "prov:Entity",
                "derived_from": "curated_immunology_kb_v1",
            },
        },
        {
            "id": "BIOOS:A0004882",
            "source": "STAT3",
            "target": "CRP",
            "relation": "transactivates",
            "eco": "A",
            "conflict": False,
            "context": {
                "organism": "Homo sapiens",
                "tissue": "Liver",
                "cell_type": "Hepatocyte",
                "pathology": "Inflammation",
                "physiology": "Acute phase response",
                "stimulus": "STAT3",
            },
            "provenance": {
                "prov_type": "prov:Entity",
                "derived_from": "curated_immunology_kb_v1",
            },
        },
    ]


ASSERTIONS: list[dict[str, Any]] = _load_initial_assertions()
SNAPSHOT_STORE: dict[str, dict[str, Any]] = {}


def _knowledge_base_sha256() -> str:
    return _sha256_obj(_as_assertion_dicts(ASSERTIONS))


def _query_assertions(
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
    normalized_ecos = _split_csv_param(eco_levels)
    return filter_assertions(
        assertions=_as_assertion_dicts(ASSERTIONS),
        eco_levels=normalized_ecos,
        min_eco=min_eco,
        organism=organism,
        cell_type=cell_type,
        tissue=tissue,
        pathology=pathology,
        physiology=physiology,
        stimulus=stimulus,
        conflict_only=conflict_only,
    )


def _traverse_graph(
    start: str,
    max_hops: int = 2,
    eco_levels: list[str] | None = None,
    min_eco: str | None = None,
    organism: str | None = None,
    cell_type: str | None = None,
    tissue: str | None = None,
    pathology: str | None = None,
    physiology: str | None = None,
    stimulus: str | None = None,
    conflict_only: bool = False,
) -> dict[str, Any]:
    filtered = _query_assertions(
        eco_levels=eco_levels,
        min_eco=min_eco,
        organism=organism,
        cell_type=cell_type,
        tissue=tissue,
        pathology=pathology,
        physiology=physiology,
        stimulus=stimulus,
        conflict_only=conflict_only,
    )

    start_clean = start.strip()
    visited: set[str] = {start_clean}
    frontier: set[str] = {start_clean}
    discovered_nodes: list[str] = []
    edges: list[dict[str, Any]] = []
    seen_edge_keys: set[tuple[str, str, str]] = set()

    for _hop in range(max(0, max_hops)):
        next_frontier: set[str] = set()
        for assertion in filtered:
            src = str(assertion.get("source") or "").strip()
            tgt = str(assertion.get("target") or "").strip()
            rel = str(assertion.get("relation") or "regulates").strip()
            if not src or not tgt:
                continue
            if src in frontier:
                edge_key = (src, tgt, assertion.get("id") or rel)
                if edge_key not in seen_edge_keys:
                    seen_edge_keys.add(edge_key)
                    edges.append(
                        {
                            "assertion_id": assertion.get("id"),
                            "source": src,
                            "target": tgt,
                            "relation": rel,
                            "eco": assertion.get("eco"),
                        }
                    )
                if tgt not in visited:
                    visited.add(tgt)
                    next_frontier.add(tgt)
                    if tgt != start_clean and tgt not in discovered_nodes:
                        discovered_nodes.append(tgt)
        frontier = next_frontier
        if not frontier:
            break

    return {
        "start": start_clean,
        "max_hops": max_hops,
        "nodes": discovered_nodes,
        "all_visited_nodes": [start_clean] + discovered_nodes,
        "edges": edges,
        "knowledge_base_sha256": _knowledge_base_sha256(),
    }


def _evaluate_dataset_payload(
    name: str,
    modality: str,
    tissue: str | None,
    cell_type: str | None,
    signals: list[str],
) -> dict[str, Any]:
    upper_tokens = {token.strip().upper() for token in signals if token.strip()}
    active_states: list[dict[str, Any]] = []

    inflammation_markers = {"TNF UP", "IL6 UP", "TNF", "IL6"}
    if upper_tokens & inflammation_markers:
        matched = sorted(upper_tokens & inflammation_markers)
        active_states.append(
            {
                "name": "Acute Inflammation",
                "score": round(len(matched) / len(inflammation_markers), 3),
                "matched_signals": matched,
            }
        )

    interferon_markers = {"IFNG UP", "CXCL10 UP", "IFNG", "CXCL10"}
    if upper_tokens & interferon_markers:
        matched = sorted(upper_tokens & interferon_markers)
        active_states.append(
            {
                "name": "Interferon State",
                "score": round(len(matched) / len(interferon_markers), 3),
                "matched_signals": matched,
            }
        )

    return {
        "dataset_name": name,
        "modality": modality,
        "tissue": tissue,
        "cell_type": cell_type,
        "signals": signals,
        "active_states": active_states,
        "knowledge_base_sha256": _knowledge_base_sha256(),
    }


def _build_snapshot(req: SnapshotCreateRequest) -> dict[str, Any]:
    signals_list = _parse_signals(req.signals)
    kb_hash = _knowledge_base_sha256()
    filtered_assertions = _query_assertions(
        eco_levels=req.eco_levels,
        min_eco=req.min_eco,
        organism=None,
        cell_type=req.cell_type,
        tissue=req.tissue,
        pathology=req.pathology,
        physiology=req.physiology,
        stimulus=req.stimulus,
    )
    subgraph = _traverse_graph(
        start=req.start_node,
        max_hops=req.max_hops,
        eco_levels=req.eco_levels,
        min_eco=req.min_eco,
        cell_type=req.cell_type,
        tissue=req.tissue,
        pathology=req.pathology,
        physiology=req.physiology,
        stimulus=req.stimulus,
    )
    dataset_eval = _evaluate_dataset_payload(
        name=f"{req.title}_evaluation",
        modality="RNA-seq",
        tissue=req.tissue,
        cell_type=req.cell_type,
        signals=signals_list,
    )

    canonical_payload = {
        "title": req.title,
        "signals": signals_list,
        "start_node": req.start_node,
        "max_hops": req.max_hops,
        "filters": {
            "tissue": req.tissue,
            "cell_type": req.cell_type,
            "pathology": req.pathology,
            "physiology": req.physiology,
            "stimulus": req.stimulus,
            "eco_levels": _split_csv_param(req.eco_levels),
            "min_eco": req.min_eco,
        },
        "assertion_ids": [a.get("id") for a in filtered_assertions],
        "subgraph": subgraph,
        "dataset_evaluation": dataset_eval,
        "knowledge_base_sha256": kb_hash,
    }
    payload_hash = _sha256_obj(canonical_payload)
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot_id = f"snap_{payload_hash[:12]}"

    snapshot_record = {
        "snapshot_id": snapshot_id,
        "title": req.title,
        "created_at": timestamp,
        "knowledge_base_sha256": kb_hash,
        "canonical_payload_sha256": payload_hash,
        "payload": canonical_payload,
        "provenance": {
            "@context": "http://www.w3.org/ns/prov#",
            "prov:type": "prov:Entity",
            "prov:wasGeneratedBy": {
                "prov:type": "prov:Activity",
                "prov:wasAssociatedWith": {
                    "prov:type": "prov:Agent",
                    "name": req.actor,
                },
                "prov:endedAtTime": timestamp,
            },
            "prov:wasDerivedFrom": f"kb:sha256:{kb_hash}",
        },
    }
    SNAPSHOT_STORE[snapshot_id] = snapshot_record
    return snapshot_record


def _verify_snapshot_record(snapshot: dict[str, Any]) -> dict[str, Any]:
    payload = snapshot.get("payload")
    recorded_payload_hash = snapshot.get("canonical_payload_sha256")
    recorded_kb_hash = snapshot.get("knowledge_base_sha256")
    current_kb_hash = _knowledge_base_sha256()

    recomputed_payload_hash = _sha256_obj(payload) if payload is not None else ""
    payload_matches = bool(recorded_payload_hash) and (recomputed_payload_hash == recorded_payload_hash)
    kb_matches_current = bool(recorded_kb_hash) and (recorded_kb_hash == current_kb_hash)

    return {
        "valid": payload_matches,
        "payload_sha256_valid": payload_matches,
        "recomputed_payload_sha256": recomputed_payload_hash,
        "recorded_payload_sha256": recorded_payload_hash,
        "recorded_knowledge_base_sha256": recorded_kb_hash,
        "current_knowledge_base_sha256": current_kb_hash,
        "knowledge_base_unchanged": kb_matches_current,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "assertions_count": len(ASSERTIONS),
        "knowledge_base_sha256": _knowledge_base_sha256(),
    }


@app.get("/v1/assertions")
def get_assertions(
    eco_levels: list[str] | None = Query(default=None, description="ECO codes, e.g. A, B, C"),
    min_eco: str | None = Query(default=None, description="Minimum ECO tier (A through G)"),
    organism: str | None = Query(default=None, description="Organism name"),
    cell_type: str | None = Query(default=None, description="Cell type name"),
    tissue: str | None = Query(default=None, description="Tissue name, e.g. Blood or Liver"),
    pathology: str | None = Query(default=None, description="Pathology category"),
    physiology: str | None = Query(default=None, description="Physiological state"),
    stimulus: str | None = Query(default=None, description="Stimulus or perturbation"),
    conflict_only: bool = Query(default=False, description="Return only conflicting assertions"),
) -> list[dict[str, Any]]:
    return _query_assertions(
        eco_levels=eco_levels,
        min_eco=min_eco,
        organism=organism,
        cell_type=cell_type,
        tissue=tissue,
        pathology=pathology,
        physiology=physiology,
        stimulus=stimulus,
        conflict_only=conflict_only,
    )


@app.get("/v1/graph")
def get_graph(
    start: str = Query(..., description="Starting regulator or gene symbol"),
    max_hops: int = Query(default=2, ge=1, le=8, description="Maximum traversal hops"),
    eco_levels: list[str] | None = Query(default=None),
    min_eco: str | None = Query(default=None),
    organism: str | None = Query(default=None),
    cell_type: str | None = Query(default=None),
    tissue: str | None = Query(default=None),
    pathology: str | None = Query(default=None),
    physiology: str | None = Query(default=None),
    stimulus: str | None = Query(default=None),
    conflict_only: bool = Query(default=False),
) -> dict[str, Any]:
    return _traverse_graph(
        start=start,
        max_hops=max_hops,
        eco_levels=eco_levels,
        min_eco=min_eco,
        organism=organism,
        cell_type=cell_type,
        tissue=tissue,
        pathology=pathology,
        physiology=physiology,
        stimulus=stimulus,
        conflict_only=conflict_only,
    )


@app.post("/v1/datasets/evaluate")
def evaluate_dataset(req: DatasetEvalRequest) -> dict[str, Any]:
    return _evaluate_dataset_payload(
        name=req.name,
        modality=req.modality,
        tissue=req.tissue,
        cell_type=req.cell_type,
        signals=_parse_signals(req.signals),
    )


@app.post("/v1/snapshots")
def create_snapshot(req: SnapshotCreateRequest) -> dict[str, Any]:
    snap = _build_snapshot(req)
    verification = _verify_snapshot_record(snap) if req.verify else None
    return {
        "snapshot": snap,
        "verification": verification,
    }


@app.get("/v1/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str) -> dict[str, Any]:
    snap = SNAPSHOT_STORE.get(snapshot_id)
    if not snap:
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found")
    return snap


@app.post("/v1/snapshots/verify")
def verify_snapshot(req: SnapshotVerifyRequest) -> dict[str, Any]:
    if req.snapshot is not None:
        return _verify_snapshot_record(req.snapshot)
    if req.file_path:
        path = Path(req.file_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Snapshot file not found: {req.file_path}")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        snap_obj = loaded.get("snapshot", loaded) if isinstance(loaded, dict) else {}
        return _verify_snapshot_record(snap_obj)
    raise HTTPException(status_code=400, detail="Provide either 'snapshot' or 'file_path'")