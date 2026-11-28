from __future__ import annotations

import json
from hypothesis_engine_v0_1.ollama_renderer import (
    DEFAULT_OLLAMA_MODEL,
    get_configured_ollama_model,
    render_with_ollama_guardrail,
)


def test_default_model_is_lightweight_llama32_1b_and_env_override(monkeypatch):
    monkeypatch.delenv("IRA_OLLAMA_MODEL", raising=False)
    assert get_configured_ollama_model() == DEFAULT_OLLAMA_MODEL == "llama3.2:1b"

    monkeypatch.setenv("IRA_OLLAMA_MODEL", "qwen2.5:latest")
    assert get_configured_ollama_model() == "qwen2.5:latest"


def test_ollama_renderer_accepts_valid_grounded_json(monkeypatch):
    monkeypatch.delenv("IRA_OLLAMA_MODEL", raising=False)

    def fake_caller(**kwargs):
        assert kwargs["model"] == "llama3.2:1b"
        assert "properties" in kwargs["schema"]
        return json.dumps(
            {
                "query_id": "q-100",
                "mode": "weak_inference",
                "has_perturbation_evidence": False,
                "claims": [
                    {
                        "statement": "IL6 transcript level is elevated in hypoxia sample compared to control.",
                        "claim_kind": "observation_summary",
                        "hypothesis_id": "H1",
                        "cited_ids": ["OBS-1"],
                    }
                ],
                "introduced_hypothesis_ids": [],
                "limitations": ["No causal perturbation performed."],
            }
        )

    res = render_with_ollama_guardrail(
        query_id="q-100",
        mode="weak_inference",
        context_summary="human macrophage 1% O2",
        allowed_source_ids={"OBS-1"},
        allowed_hypothesis_ids={"H1", "H2"},
        has_perturbation_evidence=False,
        chat_caller=fake_caller,
    )
    assert res.model_used == "llama3.2:1b"
    assert res.used_fallback is False
    assert len(res.payload.claims) == 1
    assert res.payload.claims[0].cited_ids == ["OBS-1"]


def test_ollama_renderer_falls_back_safely_on_causal_violation_or_outage():
    def causal_violating_caller(**kwargs):
        return json.dumps(
            {
                "query_id": "q-101",
                "mode": "weak_inference",
                "has_perturbation_evidence": False,
                "claims": [
                    {
                        "statement": "Hypoxia causes IL6 elevation in all macrophages.",
                        "claim_kind": "hypothesis_status",
                        "hypothesis_id": "H1",
                        "cited_ids": ["OBS-1"],
                    }
                ],
            }
        )

    res_bad = render_with_ollama_guardrail(
        query_id="q-101",
        mode="weak_inference",
        context_summary="human macrophage 1% O2",
        allowed_source_ids={"OBS-1"},
        allowed_hypothesis_ids={"H1"},
        has_perturbation_evidence=False,
        chat_caller=causal_violating_caller,
    )
    assert res_bad.used_fallback is True
    assert any(i.code == "forbidden_causal_claim_without_perturbation" for i in res_bad.validation_issues)
    assert "causes" not in res_bad.payload.claims[0].statement.lower()
    assert res_bad.payload.claims[0].cited_ids == ["OBS-1"]

    def offline_caller(**kwargs):
        raise RuntimeError("connection refused")

    res_offline = render_with_ollama_guardrail(
        query_id="q-102",
        mode="weak_inference",
        context_summary="human macrophage 1% O2",
        allowed_source_ids={"OBS-1"},
        allowed_hypothesis_ids={"H1"},
        chat_caller=offline_caller,
    )
    assert res_offline.used_fallback is True
    assert res_offline.payload.claims[0].cited_ids == ["OBS-1"]


def test_ollama_renderer_rejects_unknown_ids_and_mode_mismatch():
    def bad_ids_and_mode_caller(**kwargs):
        return json.dumps(
            {
                "query_id": "q-103",
                "mode": "conditional_inference",
                "has_perturbation_evidence": False,
                "claims": [
                    {
                        "statement": "IL6 transcript is reported in observed sample.",
                        "claim_kind": "observation_summary",
                        "hypothesis_id": "H_UNAPPROVED",
                        "cited_ids": ["OBS-UNKNOWN"],
                    }
                ],
            }
        )

    res = render_with_ollama_guardrail(
        query_id="q-103",
        mode="weak_inference",
        context_summary="human macrophage 1% O2",
        allowed_source_ids={"OBS-1"},
        allowed_hypothesis_ids={"H1"},
        has_perturbation_evidence=False,
        chat_caller=bad_ids_and_mode_caller,
    )
    codes = {i.code for i in res.validation_issues}
    assert res.used_fallback is True
    assert {"mode_mismatch", "missing_or_unknown_citation", "unauthorized_hypothesis"}.issubset(codes)


def test_default_ollama_chat_caller_success_empty_and_network_error(monkeypatch):
    import io
    import urllib.error
    import pytest
    from hypothesis_engine_v0_1.ollama_renderer import build_deterministic_fallback, default_ollama_chat_caller

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30.0: _R(json.dumps({"message": {"content": "{\"query_id\":\"q1\"}"}}).encode("utf-8")),
    )
    assert '"q1"' in default_ollama_chat_caller(
        base_url="http://localhost:11434/api/chat",
        model="llama3.2:1b",
        messages=[{"role": "user", "content": "hi"}],
        schema={"type": "object"},
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30.0: _R(json.dumps({"message": {"content": "   "}}).encode("utf-8")),
    )
    with pytest.raises(RuntimeError, match="empty message content"):
        default_ollama_chat_caller(
            base_url="http://localhost:11434/api/chat",
            model="llama3.2:1b",
            messages=[{"role": "user", "content": "hi"}],
            schema={"type": "object"},
        )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30.0: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    with pytest.raises(RuntimeError, match="Ollama request failed"):
        default_ollama_chat_caller(
            base_url="http://localhost:11434/api/chat",
            model="llama3.2:1b",
            messages=[{"role": "user", "content": "hi"}],
            schema={"type": "object"},
        )
    fb = build_deterministic_fallback(
        query_id="q-empty",
        mode="hard_abstention",
        allowed_source_ids=set(),
        allowed_hypothesis_ids={"H1"},
    )
    assert fb.claims == [] and fb.mode == "hard_abstention"
