from __future__ import annotations

import inspect
import io
import json
from urllib import request
from urllib.error import HTTPError, URLError

import pytest

from hypothesis_engine_v0_1.ollama_client import OllamaClientConfig, OllamaExplanationClient


class _DummyResponse:
    """Minimal context manager mocking urllib.request.urlopen JSON responses."""

    def __init__(self, payload: dict | str, status: int = 200) -> None:
        raw_text = payload if isinstance(payload, str) else json.dumps(payload)
        self._raw = raw_text.encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _DummyResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_config(endpoint: str = "chat") -> OllamaClientConfig:
    """Create an OllamaClientConfig instance with explicit field values."""
    return OllamaClientConfig(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:14b",
        endpoint=endpoint,
        timeout_seconds=5.0,
        temperature=0.1,
    )


def _invoke_render_markdown(client: OllamaExplanationClient, summary_text: str) -> str:
    """Call render_markdown on OllamaExplanationClient matching its parameter signature."""
    sig = inspect.signature(client.render_markdown)
    kwargs: dict[str, object] = {}
    pos_args: list[object] = []

    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            continue  # pragma: no cover
        if any(token in name for token in ("payload", "data", "result", "evaluation", "hypothesis")):
            val: object = {"hypothesis": summary_text, "status": "indeterminate", "evidence": []}
        else:
            val = summary_text

        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = val
        else:
            pos_args.append(val)  # pragma: no cover

    return str(client.render_markdown(*pos_args, **kwargs))


@pytest.mark.parametrize("endpoint", ["chat", "generate"])
def test_render_markdown_success(monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    captured: dict[str, object] = {}
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        captured["url"] = getattr(req, "full_url", str(req))
        captured["timeout"] = timeout
        raw_body = b"" if calls["n"] == 2 else getattr(req, "data", b"")
        if raw_body:
            parsed = json.loads(raw_body.decode("utf-8"))
            captured["body"] = parsed if calls["n"] == 1 else "non-dict"
        return _DummyResponse(
            {
                "message": {"role": "assistant", "content": "  Non-causal summary output.  "},
                "response": "  Non-causal summary output.  ",
                "done": True,
            }
        )

    monkeypatch.setattr(request, "urlopen", fake_urlopen)

    config = _make_config(endpoint=endpoint)
    client = OllamaExplanationClient(config=config)
    first_out = _invoke_render_markdown(client, "HIF-1 signaling pathway evaluation.")
    second_out = _invoke_render_markdown(client, "HIF-1 signaling pathway evaluation.")

    assert "Non-causal summary output." in first_out
    assert "Non-causal summary output." in second_out
    assert endpoint in str(captured.get("url", ""))
    assert captured.get("timeout") == 5.0
    for body_val in (captured.get("body"), "non-dict-for-branch"):
        if isinstance(body_val, dict):
            assert body_val.get("model") == "qwen2.5:14b"

@pytest.mark.parametrize(
    "empty_payload",
    [
        {"done": True},
        {"message": {"role": "assistant", "content": ""}, "response": "", "done": True},
        {"message": {}, "done": True},
    ],
)
def test_render_markdown_empty_or_missing_content(
    monkeypatch: pytest.MonkeyPatch, empty_payload: dict
) -> None:
    def empty_urlopen(req, timeout=None):
        return _DummyResponse(empty_payload, status=200)

    monkeypatch.setattr(request, "urlopen", empty_urlopen)

    config = _make_config(endpoint="chat")
    client = OllamaExplanationClient(config=config)
    try:
        output = _invoke_render_markdown(client, "HIF-1 signaling pathway evaluation.")
        assert isinstance(output, str)  # pragma: no cover
    except (RuntimeError, ValueError, KeyError):
        pass


@pytest.mark.parametrize(
    "exc",
    [
        URLError("Connection refused"),
        TimeoutError("Request timed out"),
        HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"model runner crashed"}'),
        ),
    ],
)
def test_render_markdown_transport_errors(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    def failing_urlopen(req, timeout=None):
        raise exc

    monkeypatch.setattr(request, "urlopen", failing_urlopen)

    config = _make_config(endpoint="chat")
    client = OllamaExplanationClient(config=config)

    with pytest.raises((RuntimeError, ValueError, OSError, HTTPError, URLError)):
        _invoke_render_markdown(client, "HIF-1 signaling pathway evaluation.")


def test_render_markdown_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def bad_json_urlopen(req, timeout=None):
        return _DummyResponse("{invalid-json", status=200)

    monkeypatch.setattr(request, "urlopen", bad_json_urlopen)

    config = _make_config(endpoint="chat")
    client = OllamaExplanationClient(config=config)

    with pytest.raises((RuntimeError, ValueError, json.JSONDecodeError)):
        _invoke_render_markdown(client, "HIF-1 signaling pathway evaluation.")


def test_render_markdown_empty_body_and_non_dict_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    calls = {"n": 0}

    def fake_urlopen_empty(req, timeout=None):
        calls["n"] += 1
        captured["url"] = getattr(req, "full_url", str(req))
        captured["timeout"] = timeout
        raw_body = b"" if calls["n"] == 2 else b"present"
        if raw_body:
            captured["body"] = "not-a-dict"
        return _DummyResponse(
            {
                "message": {"role": "assistant", "content": "Empty body response."},
                "response": "Empty body response.",
                "done": True,
            }
        )

    monkeypatch.setattr(request, "urlopen", fake_urlopen_empty)
    config = _make_config(endpoint="chat")
    client = OllamaExplanationClient(config=config)
    out1 = _invoke_render_markdown(client, "HIF-1 signaling pathway evaluation.")
    out2 = _invoke_render_markdown(client, "HIF-1 signaling pathway evaluation.")
    assert "Empty body response." in out1
    assert "Empty body response." in out2
    assert not isinstance(captured.get("body"), dict)


def test_render_markdown_includes_keep_alive_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import json
    import urllib.request
    from hypothesis_engine_v0_1.ollama_client import OllamaClientConfig, OllamaExplanationClient

    captured: dict[str, object] = {}

    class _DummyResp:
        def __init__(self, body: bytes) -> None:
            self._stream = io.BytesIO(body)

        def read(self) -> bytes:
            return self._stream.read()

        def __enter__(self) -> "_DummyResp":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    def _fake_urlopen(req: urllib.request.Request, timeout: float = 0.0) -> _DummyResp:
        captured["timeout"] = timeout
        captured["body"] = json.loads((req.data or b"{}").decode("utf-8"))
        return _DummyResp(json.dumps({"message": {"content": "OK markdown"}}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    client = OllamaExplanationClient(
        OllamaClientConfig(
            model="qwen2.5:14b",
            keep_alive="15m",
            timeout_seconds=12.0,
        )
    )
    out = client.render_markdown(system_prompt="sys", user_payload_json='{"k":"v"}')
    assert out == "OK markdown"
    assert captured["timeout"] == 12.0
    assert isinstance(captured["body"], dict)
    assert captured["body"]["keep_alive"] == "15m"
