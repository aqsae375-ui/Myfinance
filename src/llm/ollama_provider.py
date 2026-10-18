"""Ollama (fully local) provider, via its REST API /api/chat.

JSON is forced with Ollama's `format` option set to our JSON schema (constrained decoding; needs
Ollama >= 0.5), temperature 0 and a fixed seed for repeatable output. Uses only the stdlib HTTP
client, so the local mode has no extra dependency.

Env: OLLAMA_HOST (default http://localhost:11434), OLLAMA_MODEL, OLLAMA_TIMEOUT (seconds),
OLLAMA_BATCH_SIZE (small local models are more accurate with smaller batches).
"""
from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Sequence

from dotenv import load_dotenv

from src.llm.base import (RESPONSE_SCHEMA, ChatResponse, Completion, LLMProvider, LLMProviderError, ToolCall,
                          ToolSpec)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"


class OllamaProvider(LLMProvider):
    def __init__(self, model: str | None = None, host: str | None = None,
                 timeout: float | None = None, batch_size: int | None = None) -> None:
        super().__init__()
        load_dotenv()
        self.model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL
        self.host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.timeout = float(timeout or os.getenv("OLLAMA_TIMEOUT") or 120)
        self.batch_size = int(batch_size or os.getenv("OLLAMA_BATCH_SIZE") or 10)
        self.name = f"ollama:{self.model}"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self.host}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code == 404 and "not found" in detail.lower():
                raise LLMProviderError(
                    f"Ollama model {self.model!r} is not installed. Run: ollama pull {self.model}") from e
            raise LLMProviderError(f"Ollama returned HTTP {e.code} for {path}: {detail}") from e
        except (TimeoutError, socket.timeout) as e:
            raise self._timeout_error(url) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)):
                raise self._timeout_error(url) from e
            raise LLMProviderError(
                f"Cannot reach Ollama at {self.host} ({e.reason}). Is it running? Start it with: ollama serve") from e
        except json.JSONDecodeError as e:
            raise LLMProviderError(f"Ollama returned a non-JSON HTTP body for {path}.") from e

    def _timeout_error(self, url: str) -> LLMProviderError:
        return LLMProviderError(
            f"Ollama timed out after {self.timeout:.0f}s ({url}). The first call loads the model "
            "and can be slow: raise OLLAMA_TIMEOUT or lower OLLAMA_BATCH_SIZE.")

    def ensure_ready(self) -> None:
        """Fail fast with a clear message if the server is down or the model is not pulled."""
        tags = self._request("GET", "/api/tags")
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        wanted = {self.model, f"{self.model}:latest"}
        if not names & wanted:
            raise LLMProviderError(f"Ollama model {self.model!r} is not installed. Run: ollama pull {self.model}")

    def _complete(self, system: str, user: str) -> Completion:
        data = self._request("POST", "/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "options": {"temperature": 0, "seed": 42},
        })
        if "error" in data:
            raise LLMProviderError(f"Ollama error: {data['error']}")
        text = (data.get("message") or {}).get("content", "")
        return Completion(text, data.get("prompt_eval_count"), data.get("eval_count"))

    def chat(self, system: str, messages: Sequence[dict], tools: Sequence[ToolSpec]) -> ChatResponse:
        data = self._request("POST", "/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *to_ollama_messages(messages)],
            "tools": [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                        "parameters": t.parameters}} for t in tools],
            "stream": False,
            "options": {"temperature": 0, "seed": 42},
        })
        if "error" in data:
            raise LLMProviderError(f"Ollama error: {data['error']}")
        msg = data.get("message") or {}
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):  # some models return the arguments as a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            args = args if isinstance(args, dict) else {}
            calls.append(ToolCall(c.get("id") or f"call_{i}", fn.get("name", ""), args))
        return ChatResponse((msg.get("content") or "").strip(), calls,
                            data.get("prompt_eval_count"), data.get("eval_count"))


def to_ollama_messages(messages: Sequence[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"], "tool_name": m["name"]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content", ""),
                        "tool_calls": [{"function": {"name": c.name, "arguments": c.args}} for c in m["tool_calls"]]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out
