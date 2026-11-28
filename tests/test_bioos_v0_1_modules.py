from types import SimpleNamespace

import bioos.main as bioos_main
from bioos.engine.v0_1_modules import (
    _as_item_dict,
    _as_item_dict_list,
    _build_v0_1_hypotheses,
    _build_v0_1_modules,
    _get_runtime_context,
)


class _SlottedOnly:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x


class _DumpedObj:
    def model_dump(self):
        return {"id": "d1", "source": "S1", "target": "T1"}


def test_item_dict_conversions_and_empty_list():
    assert _as_item_dict_list([]) == []
    assert _as_item_dict(_DumpedObj())["id"] == "d1"
    assert _as_item_dict(SimpleNamespace(id="ns1"))["id"] == "ns1"
    assert _as_item_dict(_SlottedOnly(1)) == {}


def test_runtime_context_and_v01_modules_branches(monkeypatch):
    monkeypatch.setattr(
        bioos_main,
        "ASSERTIONS",
        [
            {"id": "", "source": "", "target": "", "context": {"stimulus": "INFLAMMATION"}},
            {"id": "a1", "source": "TNF", "target": "IL6", "context": {"stimulus": "INFLAMMATION"}},
        ],
        raising=False,
    )
    monkeypatch.setattr(bioos_main, "V0_1_ASSERTIONS", [], raising=False)
    monkeypatch.setattr(
        bioos_main,
        "_V0_1_MODULE_SPECS",
        [
            {
                "module_id": "m_inf",
                "name": "Inflammation Module",
                "description": "Inflammation axis",
                "axis": "Inflammation",
                "anchor_nodes": ["TNF"],
            },
            {
                "module_id": "m_hyp",
                "name": "Hypoxia Module",
                "description": "Hypoxia axis",
                "axis": "Hypoxia",
                "anchor_nodes": ["HIF1A"],
            },
        ],
        raising=False,
    )
    monkeypatch.setattr(
        bioos_main,
        "_evaluate_dataset_payload",
        lambda **kwargs: {
            "active_states": [
                {
                    "name": "Acute Inflammation",
                    "score": 0.8,
                    "matched_signals": ["TNF"],
                    "missing_context_checks": [],
                }
            ],
            "modality_warnings": [],
            "disclaimer": "Test disclaimer",
            "knowledge_base_sha256": "deadbeef",
        },
        raising=False,
    )
    monkeypatch.setattr(bioos_main, "_knowledge_base_sha256", lambda: "deadbeef", raising=False)

    merged, specs, _, kb_hash = _get_runtime_context()
    assert len(merged) == 2
    assert len(specs) == 2
    assert kb_hash() == "deadbeef"

    mods = _build_v0_1_modules()
    assert len(mods) == 2

    req = SimpleNamespace(
        name="run1",
        modality="RNA-seq",
        tissue="blood",
        cell_type="PBMC",
        signals=["TNF"],
        axes=None,
    )
    res = _build_v0_1_hypotheses(req)
    assert isinstance(res, dict)
    assert len(res["hypotheses"]) == 1
    assert res["hypotheses"][0]["axis"] == "Inflammation"

    monkeypatch.setattr(
        bioos_main,
        "_evaluate_dataset_payload",
        lambda **kwargs: {"active_states": [], "modality_warnings": [], "disclaimer": "Test disclaimer", "knowledge_base_sha256": "deadbeef"},
        raising=False,
    )
    res_empty = _build_v0_1_hypotheses(req)
    assert res_empty["hypotheses"] == []
