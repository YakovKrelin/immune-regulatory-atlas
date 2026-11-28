import json
import pathlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import bioos.main as bioos_main
from bioos.main import (
    _RankPathsPayload,
    _as_assertion_dicts,
    _load_initial_assertions,
    _merge_assertions_with_v0_1,
    _parse_signals,
    _split_csv_param,
    _traverse_graph,
    bioos_v02_rank_paths,
    get_snapshot,
    verify_snapshot,
)


class _ModelDumpObj:
    def model_dump(self):
        return {"id": "md1", "source": "A", "target": "B"}


class _LegacyDictObj:
    def dict(self):
        return {"id": "ld1", "source": "B", "target": "C"}


class _SlottedObj:
    __slots__ = ("val",)

    def __init__(self, val):
        self.val = val


def test_helpers_and_load_initial_assertions(monkeypatch, tmp_path):
    assert _split_csv_param(["TNF, , IL6"]) == ["TNF", "IL6"]
    assert _parse_signals(None) == []
    assert _parse_signals(["  ", "STAT1"]) == ["STAT1"]

    assert _as_assertion_dicts(None) == []
    assert len(
        _as_assertion_dicts([_ModelDumpObj(), _LegacyDictObj(), SimpleNamespace(id="ns1"), _SlottedObj(5)])
    ) == 3

    json_file = tmp_path / "data" / "assertions.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)

    class _FakePath:
        def __init__(self, p):
            self._p = pathlib.Path(p)

        def resolve(self):
            return self

        @property
        def parents(self):
            return {2: tmp_path}

        def __truediv__(self, other):
            return self._p / other

    monkeypatch.setattr(bioos_main, "Path", _FakePath)

    json_file.write_text(json.dumps({"other": []}), encoding="utf-8")
    assert isinstance(_load_initial_assertions(), list)

    json_file.write_text(json.dumps({"assertions": [{"id": "A1"}]}), encoding="utf-8")
    assert _load_initial_assertions() == [{"id": "A1"}]

    json_file.write_text(json.dumps([{"id": "A2"}]), encoding="utf-8")
    assert _load_initial_assertions() == [{"id": "A2"}]


def test_merge_and_traverse_graph_branches(monkeypatch):
    merged = _merge_assertions_with_v0_1(
        [
            {"id": "", "source": "X", "target": "Y"},
            {"id": "dup1", "source": "A", "target": "B"},
            {"id": "c1", "source": "B", "target": "C", "conflict": True},
        ],
        [
            {"id": "dup1", "source": "A", "target": "B"},
        ],
    )
    assert len(merged) == 3
    conflict_item = [x for x in merged if x.get("id") == "c1"][0]
    assert conflict_item["conflict_status"] == "context_dependent"

    sample_assertions = [
        {"id": "", "source": "", "target": "B"},
        {"id": "e1", "source": "A", "target": "B", "relation": "regulates"},
        {"id": "e1", "source": "A", "target": "B", "relation": "regulates"},
        {"id": "e2", "source": "A", "target": "B", "relation": "activates"},
        {"id": "e3", "source": "B", "target": "C", "relation": "regulates"},
    ]
    monkeypatch.setattr(bioos_main, "_query_assertions", lambda **kwargs: sample_assertions)
    traversed = _traverse_graph(start="A", max_hops=2)
    assert traversed["start"] == "A"
    assert isinstance(traversed["edges"], list)


def test_snapshots_and_v02_rank_paths_endpoints(monkeypatch, tmp_path):
    with pytest.raises(HTTPException) as exc_404:
        get_snapshot("missing-snap-id")
    assert exc_404.value.status_code == 404

    monkeypatch.setattr(bioos_main, "SNAPSHOT_STORE", {"snap-ok": {"snapshot_id": "snap-ok"}}, raising=False)
    assert get_snapshot("snap-ok") == {"snapshot_id": "snap-ok"}

    missing_file = tmp_path / "does_not_exist.json"
    with pytest.raises(HTTPException) as exc_file_404:
        verify_snapshot(SimpleNamespace(snapshot=None, file_path=str(missing_file)))
    assert exc_file_404.value.status_code == 404

    with pytest.raises(HTTPException) as exc_400:
        verify_snapshot(SimpleNamespace(snapshot=None, file_path=None))
    assert exc_400.value.status_code == 400

    payload = _RankPathsPayload.model_construct(
        candidates=[],
        active_markers=[],
        anchor_markers=[],
        counter_markers=[],
        target_context={},
    )
    ranked_res = bioos_v02_rank_paths(payload)
    assert ranked_res["count"] == 0
