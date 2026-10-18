"""Stage 4 conversational agent: typed tools over src/analytics, a provider-agnostic tool-calling loop,
a per-turn trace of every tool call, and a number verifier that keeps the model from stating figures
it did not get from a tool."""
from src.agent.chat import REFUSAL, AgentReply, FinanceAgent, in_scope
from src.agent.tools import TOOL_SPECS, TOOLS, ToolResult, run_tool
from src.agent.trace import ToolCallRecord, TraceLog, TurnTrace

__all__ = ["FinanceAgent", "AgentReply", "REFUSAL", "in_scope", "TOOLS", "TOOL_SPECS", "ToolResult", "run_tool",
           "TurnTrace", "ToolCallRecord", "TraceLog"]
