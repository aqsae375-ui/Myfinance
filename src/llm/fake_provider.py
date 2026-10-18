"""Deterministic, offline provider for tests and dry runs. No network.

It reads the descriptions back out of the same delimited prompt real providers receive, so tests
exercise the real prompt building and response validation. It records everything it was sent.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

from src.categories import FALLBACK
from src.llm.base import ChatResponse, Completion, LLMProvider, ToolSpec, parse_prompt_items

INJECTION_MARKER = "ignore previous instructions"
HINT_PREFIX = "a rule suggested "
_FACTS_RE = re.compile(r"<facts>\n(.*?)\n</facts>", re.DOTALL)


class FakeProvider(LLMProvider):
    """`mapping`: case-insensitive substring -> category (or "Unsure"); unmatched descriptions follow
    the rule hint when `follow_hints` is set (a model that agrees with the hint), else get `default`.
    Every answer carries `confidence`. With `obey_injections=True` it simulates a model that follows an
    injected instruction and answers `injection_answer` (off-taxonomy by default), so tests can check
    that the validator and the categorizer contain the damage.

    Chat (the agent): by default a deterministic keyword planner (src/agent/fake_policy.py) that also
    obeys injected instructions when `obey_injections` is set. `chat_script` replaces it with fixed
    responses returned in order, to simulate specific model behavior in tests.

    Write (the advisor): rephrases the <facts> block of the prompt, copying its numbers verbatim;
    `write_script` replaces it with fixed texts returned in order (e.g. a tampered number)."""

    def __init__(self, mapping: dict[str, str] | None = None, default: str = FALLBACK,
                 batch_size: int = 20, obey_injections: bool = True,
                 chat_script: Sequence[ChatResponse] | None = None,
                 write_script: Sequence[str] | None = None, follow_hints: bool = True,
                 confidence: float = 0.9, injection_answer: str = "essential") -> None:
        super().__init__()
        self.name = "fake"
        self.mapping = {k.casefold(): v for k, v in (mapping or {}).items()}
        self.default = default
        self.batch_size = batch_size
        self.obey_injections = obey_injections
        self.follow_hints = follow_hints
        self.confidence = confidence
        self.injection_answer = injection_answer
        self.prompts: list[tuple[str, str]] = []  # (system, user) per call
        self.seen_descriptions: list[str] = []
        self.seen_items: list[dict] = []           # full prompt objects (description, direction, rule_hint)
        self.chat_script = list(chat_script) if chat_script is not None else None
        self.chat_calls: list[tuple[str, list[dict]]] = []  # (system, messages) per chat call
        self.write_script = list(write_script) if write_script is not None else None
        self.write_calls: list[tuple[str, str]] = []  # (system, user) per write call

    def _answer(self, item: dict) -> str:
        text = item["description"].casefold()
        if self.obey_injections and INJECTION_MARKER in text:
            return self.injection_answer
        for key, category in self.mapping.items():
            if key in text:
                return category
        hint = item.get("rule_hint") or ""
        if self.follow_hints and hint.startswith(HINT_PREFIX):
            return hint[len(HINT_PREFIX):]
        return self.default

    def _complete(self, system: str, user: str) -> Completion:
        self.prompts.append((system, user))
        items = parse_prompt_items(user)
        self.seen_items.extend(items)
        self.seen_descriptions.extend(it["description"] for it in items)
        payload = {"results": [{"id": it["id"], "category": self._answer(it), "confidence": self.confidence}
                               for it in items]}
        return Completion(json.dumps(payload))  # token usage left None -> estimated

    def chat(self, system: str, messages: Sequence[dict], tools: Sequence[ToolSpec]) -> ChatResponse:
        self.chat_calls.append((system, [dict(m) for m in messages]))
        if self.chat_script is not None:
            return self.chat_script.pop(0) if self.chat_script else ChatResponse("")
        from src.agent.fake_policy import respond  # lazy: src.llm must not import the agent at load time
        return respond(system, messages, tools, obey_injections=self.obey_injections)

    def write(self, system: str, user: str) -> str:
        self.write_calls.append((system, user))
        if self.write_script is not None:
            return self.write_script.pop(0) if self.write_script else ""
        m = _FACTS_RE.search(user)
        try:
            facts = json.loads(m.group(1)) if m else {}
        except json.JSONDecodeError:
            facts = {}
        if not {"action", "monthly_saving", "annual_saving"} <= set(facts):
            return ""
        return (f"{facts['action']}: that frees up {facts['monthly_saving']} every month, "
                f"which is {facts['annual_saving']} over a year.")
