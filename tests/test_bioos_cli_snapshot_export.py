import hashlib
import io
import json
from typing import Any, Literal, get_args, get_origin
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from bioos.cli import main, verify_snapshot_payload
from bioos.engine.hypotheses_export import export_assertions
from bioos.engine.snapshot_engine import build_report_snapshot
from bioos.models import Assertion, BiologicalState, Module, SnapshotRequest


def _sample_for_annotation(ann: Any, name: str, overrides: dict[str, Any]) -> Any:
    if name in overrides:
        val = overrides[name]
        if get_origin(ann) is Literal:
            allowed = get_args(ann)
            if val not in allowed and allowed:
                return allowed[0]
        return val
    if get_origin(ann) is Literal:
        return get_args(ann)[0]
    if hasattr(ann, "model_fields"):
        sub_data: dict[str, Any] = {}
        for sub_name, sub_field in ann.model_fields.items():
            if sub_field.is_required():
                sub_data[sub_name] = _sample_for_annotation(sub_field.annotation, sub_name, overrides)
        return ann(**sub_data)
    if ann is str:
        return f"{name}_val"
    if ann is float:
        return 1.0
    if ann is int:
        return 1
    if ann is bool:
        return False
    if ann is dict or get_origin(ann) is dict:
        return {}
    if ann is list or get_origin(ann) is list:
        return []
    return f"{name}_val"


def _build_model_sample(model_cls: Any, overrides: dict[str, Any]) -> Any:
    data: dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        if not field.is_required():
            continue
        data[name] = _sample_for_annotation(field.annotation, name, overrides)
    return model_cls(**data)


def _build_sample_assertion(
    assertion_id: str = "ASSERT:1",
    subject: str = "TNF",
    obj: str = "IL6",
    conflict: bool = False,
) -> Assertion:
    overrides: dict[str, Any] = {
        "id": assertion_id,
        "subject": subject,
        "predicate": "activates",
        "relation_label": "activates",
        "object": obj,
        "direction": "up",
        "eco": "A",
        "effect_size": "1.5",
        "confidence": "0.9",
        "status": "active",
        "source": "PMID:12345",
        "version": 1,
        "organism": "Homo sapiens",
        "cell_type": "macrophage",
        "tissue": "Blood",
        "pathology": "inflammation",
        "physiology": "immune_response",
        "stimulus": "hypoxia",
        "timepoint": "24h",
        "method": "RNA-seq",
        "notes": "test assertion",
        "conflict": conflict,
    }
    data: dict[str, Any] = {}
    for name, field in Assertion.model_fields.items():
        if not field.is_required():
            continue
        data[name] = _sample_for_annotation(field.annotation, name, overrides)
    return Assertion(**data)


def test_export_assertions_formats():
    assertions = [_build_sample_assertion()]

    csv_out = export_assertions(assertions, fmt="csv")
    assert "id,subject,predicate,object,direction,eco,source" in csv_out
    assert "ASSERT:1" in csv_out

    sif_out = export_assertions(assertions, fmt="sif")
    assert "\t" in sif_out

    ttl_out = export_assertions(assertions, fmt="turtle")
    assert "@prefix bioos:" in ttl_out
    assert "bioos:ASSERT_1 a bioos:Assertion ;" in ttl_out

    jsonld_out = json.loads(export_assertions(assertions, fmt="jsonld"))
    assert jsonld_out["@context"] == "https://bioos.org/contexts/assertion-v0.1.jsonld"
    assert len(jsonld_out["assertions"]) == 1


def test_build_report_snapshot_and_verify_payload():
    req = SnapshotRequest(
        title="Hypoxia IL6 Snapshot",
        signals="TNF up, IL6 up, IFNG up",
        start_node="TNF",
        max_hops=2,
        actor="BIOOS Tester",
    )
    assertions = [_build_sample_assertion(conflict=True)]
    states = [_build_model_sample(BiologicalState, {"id": "STATE:1", "name": "Inflamed", "description": "Pro-inflammatory state"})]
    modules = [_build_model_sample(Module, {"id": "MOD:1", "name": "NFkB", "core_nodes": ["TNF", "IL6"]})]

    snap = build_report_snapshot(req, assertions, states, modules)
    assert snap.snapshot_id.startswith("BIOOS:SNAP:")
    assert len(snap.canonical_payload_sha256) == 64
    assert len(snap.knowledge_base_sha256) == 64
    assert snap.prov_o.was_associated_with["agent_id"] == "bioos:agent:BIOOS_Tester"

    valid_check = verify_snapshot_payload(
        {
            "snapshot_id": snap.snapshot_id,
            "canonical_payload_sha256": snap.canonical_payload_sha256,
            "knowledge_base_sha256": snap.knowledge_base_sha256,
            "provenance": {"prov:wasDerivedFrom": f"kb:sha256:{snap.knowledge_base_sha256}"},
        }
    )
    assert valid_check["valid"] is True
    assert valid_check["prov_derived_from_valid"] is True

    mismatch = verify_snapshot_payload(
        {
            "canonical_payload_sha256": snap.canonical_payload_sha256,
            "knowledge_base_sha256": snap.knowledge_base_sha256,
            "payload": {"tampered": True},
        }
    )
    assert mismatch["payload_sha256_valid"] is False
    assert mismatch["valid"] is False


class _FakeResponse:
    def __init__(self, body: Any):
        self._raw = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_cli_subcommands_and_errors(tmp_path, capsys):
    with patch("bioos.cli.urlopen", return_value=_FakeResponse([{"id": "A1"}])):
        main(["assertions", "--eco-levels", "A", "B", "--conflict-only"])
    assert json.loads(capsys.readouterr().out) == [{"id": "A1"}]

    with patch("bioos.cli.urlopen", return_value=_FakeResponse({"nodes": [], "edges": []})):
        main(["graph", "--start", "IL6", "--max-hops", "1"])
    assert "nodes" in json.loads(capsys.readouterr().out)

    for cmd in [
        ["why-elevated", "--entity", "IL6"],
        ["perturb", "--entity", "IL6", "--action", "suppress"],
        ["hypotheses", "--signals", "IL6 up"],
        ["dataset", "--name", "ds1", "--modality", "RNA-seq"],
    ]:
        with patch("bioos.cli.urlopen", return_value=_FakeResponse({"ok": True})):
            main(cmd)
        assert json.loads(capsys.readouterr().out) == {"ok": True}

    snap_file = tmp_path / "snap.json"
    canon = json.dumps({"x": 1}, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    snap_payload = {
        "snapshot_id": "BIOOS:SNAP:abc",
        "canonical_payload_sha256": hashlib.sha256(canon).hexdigest(),
        "knowledge_base_sha256": "b" * 64,
        "payload": {"x": 1},
    }

    with patch("bioos.cli.urlopen", side_effect=[_FakeResponse(snap_payload), _FakeResponse([{"id": "A1"}])]):
        main(["snapshot", "--tissue", "Blood", "--out", str(snap_file), "--verify"])
    out_snap = json.loads(capsys.readouterr().out)
    assert out_snap["verification"]["valid"] is True
    assert snap_file.exists()

    main(["verify-snapshot", "--file", str(snap_file)])
    assert json.loads(capsys.readouterr().out)["valid"] is True

    http_err = HTTPError("http://127.0.0.1:8000/v1/assertions", 500, "Server Error", {}, io.BytesIO(b"boom"))
    with patch("bioos.cli.urlopen", side_effect=http_err):
        with pytest.raises(SystemExit) as exc_info:
            main(["assertions"])
        assert exc_info.value.code == 1
    capsys.readouterr()

    with patch("bioos.cli.urlopen", side_effect=URLError("refused")):
        with pytest.raises(SystemExit) as exc_info:
            main(["assertions"])
        assert exc_info.value.code == 1
    capsys.readouterr()


def test_verify_snapshot_edge_cases_and_dataset_json_file(tmp_path, capsys):
    assert verify_snapshot_payload("not-a-dict")["valid"] is False
    assert verify_snapshot_payload({"snapshot": "not-a-dict"})["valid"] is False

    bad_prov = verify_snapshot_payload(
        {
            "canonical_payload_sha256": "a" * 64,
            "knowledge_base_sha256": "b" * 64,
            "provenance": {"prov:wasDerivedFrom": f"kb:sha256:{'c' * 64}"},
        }
    )
    assert bad_prov["prov_derived_from_valid"] is False
    assert bad_prov["valid"] is False

    ds_file = tmp_path / "ds.json"
    ds_file.write_text('{"name": "from_file", "modality": "Cytokines", "signals": "IL6 up"}', encoding="utf-8")
    with patch("bioos.cli.urlopen", return_value=_FakeResponse({"evaluated": True})):
        main(["dataset", "--json-file", str(ds_file)])
    assert json.loads(capsys.readouterr().out) == {"evaluated": True}
