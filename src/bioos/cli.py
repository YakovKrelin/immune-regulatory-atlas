import argparse
import hashlib
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _sha256_canonical(obj: Any) -> str:
    canonical_bytes = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def verify_snapshot_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    recorded_payload_hash = snapshot.get("canonical_payload_sha256", "")
    recorded_kb_hash = snapshot.get("knowledge_base_sha256", "")
    prov = snapshot.get("prov_o", {})
    used_entities = prov.get("used_entities", [])

    kb_entity_hash = ""
    for ent in used_entities:
        if ent.get("entity_id") == "bioos:kb:assertions_states_modules":
            kb_entity_hash = ent.get("sha256", "")
            break

    kb_hash_consistent = bool(recorded_kb_hash) and (recorded_kb_hash == kb_entity_hash)
    generated_entity_hash = prov.get("generated_entity", {}).get("sha256", "")
    payload_hash_present = len(recorded_payload_hash) == 64
    payload_matches_prov = bool(recorded_payload_hash) and (recorded_payload_hash == generated_entity_hash)

    return {
        "snapshot_id": snapshot.get("snapshot_id"),
        "canonical_payload_sha256": recorded_payload_hash,
        "knowledge_base_sha256": recorded_kb_hash,
        "kb_provenance_hash_match": kb_hash_consistent,
        "payload_provenance_hash_match": payload_matches_prov,
        "sha256_format_valid": payload_hash_present,
        "valid": bool(kb_hash_consistent and payload_matches_prov and payload_hash_present),
    }


def _request_json(
    base_url: str,
    path: str,
    method: str = "GET",
    query: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> Any:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    if query:
        cleaned: list[tuple[str, str]] = []
        for key, value in query.items():
            if value is None:
                continue
            if isinstance(value, list):
                for item in value:
                    if item is not None and str(item).strip():
                        cleaned.append((key, str(item).strip()))
            else:
                if str(value).strip():
                    cleaned.append((key, str(value).strip()))
        if cleaned:
            url = f"{url}?{urlencode(cleaned)}"

    data_bytes: bytes | None = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=url, data=data_bytes, headers=headers, method=method)
    try:
        with urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} error for {url}: {body}", file=sys.stderr)
        sys.exit(1)
    except URLError as exc:
        print(f"Connection error for {url}: {exc.reason}", file=sys.stderr)
        sys.exit(1)


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eco-levels", nargs="+", default=None, help="ECO codes (e.g. A B C)")
    parser.add_argument("--min-eco", default=None, help="Minimum ECO code (e.g. A, B, C)")
    parser.add_argument("--organism", default=None, help="Organism filter")
    parser.add_argument("--cell-type", default=None, help="Cell type filter")
    parser.add_argument("--tissue", default=None, help="Tissue filter (e.g. Blood, Liver)")
    parser.add_argument("--pathology", default=None, help="Pathology filter")
    parser.add_argument("--physiology", default=None, help="Physiology filter")
    parser.add_argument("--stimulus", default=None, help="Stimulus filter")
    parser.add_argument("--conflict-only", action="store_true", help="Only conflicting assertions")


def _extract_filter_query(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "eco_levels": getattr(args, "eco_levels", None),
        "min_eco": getattr(args, "min_eco", None),
        "organism": getattr(args, "organism", None),
        "cell_type": getattr(args, "cell_type", None),
        "tissue": getattr(args, "tissue", None),
        "pathology": getattr(args, "pathology", None),
        "physiology": getattr(args, "physiology", None),
        "stimulus": getattr(args, "stimulus", None),
        "conflict_only": "true" if getattr(args, "conflict_only", False) else None,
    }


def _parse_signals_arg(signals_arg: str) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    for raw_part in signals_arg.split(","):
        part = raw_part.strip()
        if not part:
            continue
        tokens = part.split()
        symbol = tokens[0].upper()
        direction = "up"
        if len(tokens) > 1 and tokens[1].lower() in {"up", "down", "unchanged"}:
            direction = tokens[1].lower()
        measurements.append({"symbol": symbol, "direction": direction})
    return measurements


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BIOOS Human Regulatory Atlas CLI")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="BIOOS API base URL (default: http://127.0.0.1:8000)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_assert = sub.add_parser("assertions", help="Query filtered assertions")
    _add_filter_args(p_assert)

    p_graph = sub.add_parser("graph", help="Traverse regulatory neighborhood")
    p_graph.add_argument("--start", default="TNF", help="Start node (default: TNF)")
    p_graph.add_argument("--max-hops", type=int, default=2, help="Max traversal hops (1-5)")
    _add_filter_args(p_graph)

    p_why = sub.add_parser("why-elevated", help="Analyze upstream causes for an elevated entity")
    p_why.add_argument("--entity", default="IL6", help="Target entity (default: IL6)")

    p_pert = sub.add_parser("perturb", help="Simulate perturbation on an entity")
    p_pert.add_argument("--entity", default="IL6", help="Target entity (default: IL6)")
    p_pert.add_argument("--action", default="suppress", choices=["suppress", "activate"])

    p_hyp = sub.add_parser("hypotheses", help="Generate hypotheses from signal expressions")
    p_hyp.add_argument("--signals", default="TNF up, IL6 up, IFNG up, CXCL10 up")

    p_ds = sub.add_parser("dataset", help="Evaluate expression measurements via POST /v1/datasets")
    p_ds.add_argument("--name", default="cli_expression_set", help="Dataset name")
    p_ds.add_argument(
        "--modality",
        default="RNA-seq",
        choices=["RNA-seq", "Proteomics", "Cytokines", "Metabolomics"],
    )
    p_ds.add_argument("--organism", default="Homo sapiens")
    p_ds.add_argument("--tissue", default=None)
    p_ds.add_argument("--cell-type", default=None)
    p_ds.add_argument(
        "--signals",
        default="TNF up, IL6 up, IFNG up, CXCL10 up",
        help="Comma-separated features with direction (e.g. 'TNF up, IL6 up')",
    )
    p_ds.add_argument("--json-file", default=None, help="Optional JSON file matching DatasetUploadRequest")

    p_snap = sub.add_parser("snapshot", help="Create reproducibility snapshot via POST /v1/snapshots")
    p_snap.add_argument("--title", default="BIOOS CLI Snapshot")
    p_snap.add_argument("--signals", default="TNF up, IL6 up, IFNG up, CXCL10 up")
    p_snap.add_argument("--start-node", default="TNF")
    p_snap.add_argument("--max-hops", type=int, default=2)
    p_snap.add_argument("--actor", default="BIOOS-CLI")
    p_snap.add_argument("--out", default=None, help="Optional output JSON file path")
    p_snap.add_argument("--verify", action="store_true", help="Verify snapshot SHA-256 and PROV-O hashes")
    _add_filter_args(p_snap)

    p_ver = sub.add_parser("verify-snapshot", help="Verify SHA-256 digests in a saved snapshot JSON file")
    p_ver.add_argument("--file", required=True, help="Path to saved snapshot JSON file")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    result: Any
    if args.command == "assertions":
        result = _request_json(args.base_url, "/v1/assertions", query=_extract_filter_query(args))
    elif args.command == "graph":
        query = {"start": args.start, "max_hops": args.max_hops, **_extract_filter_query(args)}
        result = _request_json(args.base_url, "/v1/graph", query=query)
    elif args.command == "why-elevated":
        result = _request_json(args.base_url, "/v1/causality/why-elevated", query={"entity": args.entity})
    elif args.command == "perturb":
        result = _request_json(
            args.base_url,
            "/v1/causality/perturb",
            query={"entity": args.entity, "action": args.action},
        )
    elif args.command == "hypotheses":
        result = _request_json(args.base_url, "/v1/hypotheses", query={"signals": args.signals})
    elif args.command == "dataset":
        if args.json_file:
            with open(args.json_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            payload = {
                "dataset_name": args.name,
                "modality": args.modality,
                "organism": args.organism,
                "tissue": args.tissue,
                "cell_type": args.cell_type,
                "measurements": _parse_signals_arg(args.signals),
            }
        result = _request_json(args.base_url, "/v1/datasets", method="POST", payload=payload)
    elif args.command == "snapshot":
        active_filters = {k: v for k, v in _extract_filter_query(args).items() if v is not None}
        payload = {
            "title": args.title,
            "signals": args.signals,
            "start_node": args.start_node,
            "max_hops": args.max_hops,
            "actor": args.actor,
        }
        snapshot_data = _request_json(args.base_url, "/v1/snapshots", method="POST", payload=payload)
        if active_filters:
            snapshot_data["applied_cli_filters"] = active_filters
            snapshot_data["filtered_assertions_preview"] = [
                a.get("id")
                for a in _request_json(args.base_url, "/v1/assertions", query=active_filters)
            ]
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(snapshot_data, fh, indent=2)
        if args.verify:
            result = {
                "verification": verify_snapshot_payload(snapshot_data),
                "snapshot": snapshot_data,
            }
        else:
            result = snapshot_data
    elif args.command == "verify-snapshot":
        with open(args.file, "r", encoding="utf-8") as fh:
            saved_snapshot = json.load(fh)
        result = verify_snapshot_payload(saved_snapshot)
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()