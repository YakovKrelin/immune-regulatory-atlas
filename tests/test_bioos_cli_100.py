import json
import runpy
import sys
import urllib.request
from io import BytesIO
from types import SimpleNamespace

import pytest

import bioos.cli as bioos_cli
from bioos.cli import _request_json, main, verify_snapshot_payload


class _DummyResp(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_verify_snapshot_non_kb_derived_from():
    snap = {
        "snapshot_id": "s1",
        "payload_sha256": "0" * 64,
        "knowledge_base_sha256": "1" * 64,
        "provenance": {"prov:wasDerivedFrom": "external:source"},
    }
    res = verify_snapshot_payload(snap)
    assert res["prov_derived_from_valid"] is True


def test_request_json_query_filtering(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout=30):
        captured["url"] = req.full_url
        return _DummyResp(b'{"ok": true}')

    monkeypatch.setattr(bioos_cli, "urlopen", _fake_urlopen)
    out = _request_json(
        "http://localhost:8000/",
        "/v1/items",
        query={"skip": None, "empty_str": "   ", "items": [None, "  ", "IL6"]},
    )
    assert out == {"ok": True}
    assert "items=IL6" in captured["url"]
    assert "empty_str" not in captured["url"]


def test_cli_snapshot_no_filters_no_out_no_verify(monkeypatch, capsys):
    monkeypatch.setattr(
        bioos_cli,
        "_request_json",
        lambda base_url, path, method="GET", query=None, payload=None: {"snapshot_id": "snap-1"},
    )
    monkeypatch.setattr(sys, "argv", ["bioos", "snapshot"])
    main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["snapshot_id"] == "snap-1"


def test_cli_no_command_exits_one(monkeypatch):
    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self, args=None, namespace=None: SimpleNamespace(command=None),
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_cli_dunder_main(monkeypatch, capsys):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=15: _DummyResp(b'{"snapshot_id": "snap-main"}'),
    )
    monkeypatch.setattr(sys, "argv", ["bioos", "snapshot"])
    monkeypatch.delitem(sys.modules, "bioos.cli", raising=False)
    runpy.run_module("bioos.cli", run_name="__main__")
    printed = json.loads(capsys.readouterr().out)
    assert printed["snapshot_id"] == "snap-main"
