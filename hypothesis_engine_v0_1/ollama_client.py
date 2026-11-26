from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Literal

from pydantic import BaseModel, Field


class OllamaClientConfig(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="qwen2.5:14b")
    endpoint: Literal["chat", "generate"] = Field(default="chat")
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    num_predict: int | None = Field(default=None, ge=1)
    num_ctx: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None)
    keep_alive: str | int | None = Field(default=None)

    @classmethod
    def for_qwen25_14b(
        cls,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 60.0,
        temperature: float = 0.1,
        top_p: float | None = 0.9,
        num_predict: int | None = 1024,
        num_ctx: int | None = 4096,
        seed: int | None = 42,
        keep_alive: str | int | None = "10m",
    ) -> "OllamaClientConfig":
        return cls(
            base_url=base_url,
            model="qwen2.5:14b",
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            num_ctx=num_ctx,
            seed=seed,
            keep_alive=keep_alive,
        )


class OllamaExplanationClient:
    """Adapter implementing ExplanationLLMClient against a local Ollama server."""

    def __init__(self, config: OllamaClientConfig | None = None) -> None:
        self.config = config or OllamaClientConfig()

    def _build_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if self.config.endpoint == "generate":
            return f"{base}/api/generate"
        return f"{base}/api/chat"

    def _build_payload(self, *, system_prompt: str, user_payload_json: str) -> dict[str, object]:
        user_prompt = (
            "Render a concise Markdown summary strictly from this computed engine JSON.\n"
            "Do not add new facts, statuses, probabilities, or causal claims.\n\n"
            f"```json\n{user_payload_json}\n```"
        )
        options: dict[str, object] = {"temperature": self.config.temperature}
        if self.config.top_p is not None:
            options["top_p"] = self.config.top_p
        if self.config.num_predict is not None:
            options["num_predict"] = self.config.num_predict
        if self.config.num_ctx is not None:
            options["num_ctx"] = self.config.num_ctx
        if self.config.seed is not None:
            options["seed"] = self.config.seed

        if self.config.endpoint == "generate":
            payload: dict[str, object] = {
                "model": self.config.model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": options,
            }
        else:
            payload = {
                "model": self.config.model,
                "stream": False,
                "options": options,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

        if self.config.keep_alive is not None:
            payload["keep_alive"] = self.config.keep_alive

        return payload

    def render_markdown(self, *, system_prompt: str, user_payload_json: str) -> str:
        url = self._build_url()
        body = json.dumps(
            self._build_payload(system_prompt=system_prompt, user_payload_json=user_payload_json)
        ).encode("utf-8")

        req = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                raw_text = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach Ollama at {url}: {exc.reason}") from exc

        data = json.loads(raw_text)
        if self.config.endpoint == "generate":
            content = data.get("response")
        else:
            message = data.get("message") or {}
            content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty response payload.")

        return content.strip()
