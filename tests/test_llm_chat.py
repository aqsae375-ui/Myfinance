"""Provider chat (tool calling) wire formats. No network: a patched Ollama _request."""

from src.llm.base import ChatResponse, ToolCall, ToolSpec

TOOLS = [ToolSpec("get_monthly_summary", "Totals for a month.",
                  {"type": "object", "properties": {"month": {"type": "string"}}, "required": ["month"]})]
CALL = ToolCall("toolu_1", "get_monthly_summary", {"month": "2026-05"})
HISTORY = [
    {"role": "user", "content": "Spending in May?"},
    {"role": "assistant", "content": "", "tool_calls": [CALL, ToolCall("toolu_2", "get_subscriptions", {})]},
    {"role": "tool", "tool_call_id": "toolu_1", "name": "get_monthly_summary", "content": '{"total_spent": 1.0}'},
    {"role": "tool", "tool_call_id": "toolu_2", "name": "get_subscriptions", "content": '{"count": 0}'},
]


def test_ollama_chat_request_and_tool_calls(monkeypatch):
    from src.llm import ollama_provider
    sent = {}

    def fake_request(self, method, path, body=None):
        sent.update(method=method, path=path, body=body)
        return {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_monthly_summary", "arguments": {"month": "2026-05"}}},
            {"function": {"name": "get_subscriptions", "arguments": "{}"}}]},  # string args are tolerated
            "prompt_eval_count": 80, "eval_count": 12}

    monkeypatch.setattr(ollama_provider.OllamaProvider, "_request", fake_request)
    resp = ollama_provider.OllamaProvider(model="qwen2.5:7b").chat("SYS", HISTORY, TOOLS)
    assert [(c.name, c.args) for c in resp.tool_calls] == [("get_monthly_summary", {"month": "2026-05"}),
                                                           ("get_subscriptions", {})]
    assert (resp.input_tokens, resp.output_tokens) == (80, 12)

    body = sent["body"]
    assert sent["path"] == "/api/chat" and body["model"] == "qwen2.5:7b" and "format" not in body
    assert body["tools"][0] == {"type": "function", "function": {
        "name": "get_monthly_summary", "description": "Totals for a month.", "parameters": TOOLS[0].parameters}}
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool"]
    assert body["messages"][2]["tool_calls"][0] == {"function": {"name": "get_monthly_summary",
                                                                 "arguments": {"month": "2026-05"}}}
    assert body["messages"][3] == {"role": "tool", "content": '{"total_spent": 1.0}',
                                   "tool_name": "get_monthly_summary"}


def test_fake_provider_chat_script_and_recording():
    from src.llm.fake_provider import FakeProvider
    p = FakeProvider(chat_script=[ChatResponse("one")])
    assert p.chat("S", [{"role": "user", "content": "q"}], TOOLS).text == "one"
    assert p.chat("S", [{"role": "user", "content": "q"}], TOOLS).text == ""  # script exhausted
    assert len(p.chat_calls) == 2 and p.chat_calls[0][0] == "S"
