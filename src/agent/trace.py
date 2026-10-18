"""Per-turn log of every tool call (name, args, result, source rows), for the "How I got this" panel.

A TurnTrace is created for each question; the agent appends one ToolCallRecord per call, including
calls it makes itself for context and calls that returned an error. `TraceLog` keeps all turns and can
append them to a JSONL file.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agent.tools import ToolResult

log = logging.getLogger("finance_agent.trace")


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]            # exactly what the model saw
    txn_ids: tuple[int, ...]          # source rows behind the result
    source: str                       # analytics function used
    requested_by: str = "model"       # "model" or "agent" (context the agent fetched itself)
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class TurnTrace:
    question: str
    provider: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    answer: str = ""
    model_answer: str | None = None   # the model's own text, kept even when it was replaced
    refused: bool = False             # out-of-scope: fixed refusal, no model or tool call
    verified: bool = True             # every number in the answer was found in a tool result
    unverified_numbers: list[str] = field(default_factory=list)
    fallback_used: bool = False       # answer replaced by a deterministic rendering of the results
    retried: bool = False             # model answered with numbers but no tool call, and was told to retry
    rejected_answers: list[str] = field(default_factory=list)  # model text discarded before the final answer
    injection_flags: list[str] = field(default_factory=list)
    model_calls: int = 0

    def record(self, result: ToolResult, args: dict, requested_by: str = "model", duration_ms: float = 0.0) -> None:
        rec = ToolCallRecord(name=result.name, args=dict(args), result=result.data, txn_ids=result.txn_ids,
                             source=result.source, requested_by=requested_by,
                             error=result.data.get("error"), duration_ms=round(duration_ms, 2))
        self.tool_calls.append(rec)
        log.info("tool %s(%s) -> %s", rec.name, json.dumps(rec.args), "error: " + rec.error if rec.error else "ok")

    @property
    def model_tool_calls(self) -> list[ToolCallRecord]:
        return [c for c in self.tool_calls if c.requested_by == "model"]

    def to_dict(self) -> dict:
        d = asdict(self)
        for c in d["tool_calls"]:
            c["txn_ids"] = list(c["txn_ids"])
        return d


class TraceLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.turns: list[TurnTrace] = []
        self.path = Path(path) if path else None

    def add(self, turn: TurnTrace) -> None:
        self.turns.append(turn)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(turn.to_dict(), ensure_ascii=False) + "\n")

    @property
    def last(self) -> TurnTrace | None:
        return self.turns[-1] if self.turns else None
