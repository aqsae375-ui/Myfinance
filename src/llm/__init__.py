"""LLM providers behind one interface. Pick one with LLM_PROVIDER=ollama|fake."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMProviderError

PROVIDERS = ("ollama", "fake")


def get_provider(name: str | None = None, **kwargs) -> LLMProvider:
    """Build the provider named by `name` or LLM_PROVIDER. Defaults to 'fake' so nothing
    reaches a network model unless explicitly configured."""
    load_dotenv()
    name = (name or os.getenv("LLM_PROVIDER") or "fake").strip().lower()
    if name == "fake":
        from src.llm.fake_provider import FakeProvider
        return FakeProvider(**kwargs)
    if name == "ollama":
        from src.llm.ollama_provider import OllamaProvider
        return OllamaProvider(**kwargs)
    raise LLMProviderError(f"Unknown LLM_PROVIDER {name!r}. Available providers: {', '.join(PROVIDERS)}.")


__all__ = ["LLMProvider", "LLMProviderError", "get_provider", "PROVIDERS"]
