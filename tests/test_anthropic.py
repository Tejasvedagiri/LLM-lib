import asyncio
from types import SimpleNamespace as NS

import pytest

from llm_lib import LLM, LLMRefusal, Settings
from llm_lib.providers.anthropic import build_kwargs, split_system, translate_messages, translate_tools
from llm_lib.types import Request


def test_split_and_translate_history():
    system, rest = split_system(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "price of KO?"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "quote", "arguments": '{"symbol": "KO"}'}},
                    {"id": "t2", "type": "function", "function": {"name": "news", "arguments": "not json"}},
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "61.2"},
            {"role": "tool", "tool_call_id": "t2", "content": "none"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]},
        ]
    )
    assert system == "be brief"
    out = translate_messages(rest)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "user"]
    assert out[1]["content"][1] == {"type": "tool_use", "id": "t1", "name": "quote", "input": {"symbol": "KO"}}
    assert out[1]["content"][2]["input"] == {}
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["t1", "t2"]  # one user turn for both results
    assert out[3]["content"][0]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}


def test_translate_tools():
    tools = [{"type": "function", "function": {"name": "quote", "description": "d", "parameters": {"type": "object"}}}]
    assert translate_tools(tools) == [{"name": "quote", "description": "d", "input_schema": {"type": "object"}}]


def test_current_models_get_effort_fallbacks_and_no_sampling():
    s = Settings(provider="anthropic", temperature=0.7, effort="high")
    kwargs, beta = build_kwargs(Request(messages=[{"role": "user", "content": "q"}], json_schema={"type": "object"}), s)
    assert beta and kwargs["fallbacks"] == "default" and kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in kwargs  # rejected with a 400 on Opus 5.5
    assert kwargs["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    assert kwargs["max_tokens"] == 64000


def test_older_models_keep_sampling():
    kwargs, beta = build_kwargs(
        Request(messages=[{"role": "user", "content": "q"}]),
        Settings(provider="anthropic", model="claude-haiku-4-5", temperature=0.2, effort="low"),
    )
    assert not beta and kwargs["temperature"] == 0.2 and "output_config" not in kwargs


def _events():
    return [
        NS(type="message_start", message=NS(usage=NS(input_tokens=11))),
        NS(type="content_block_start", index=0, content_block=NS(type="thinking")),
        NS(type="content_block_delta", index=0, delta=NS(type="thinking_delta", thinking="plan")),
        NS(type="content_block_start", index=1, content_block=NS(type="text")),
        NS(type="content_block_delta", index=1, delta=NS(type="text_delta", text="Hi")),
        NS(type="content_block_start", index=2, content_block=NS(type="tool_use", id="tu1", name="quote")),
        NS(type="content_block_delta", index=2, delta=NS(type="input_json_delta", partial_json='{"symbol":')),
        NS(type="content_block_delta", index=2, delta=NS(type="input_json_delta", partial_json=' "KO"}')),
        NS(type="message_delta", usage=NS(output_tokens=7)),
    ]


class FakeStream:
    def __init__(self, events, stop_reason):
        self.events, self.stop_reason = events, stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return NS(stop_reason=self.stop_reason, stop_details=NS(category="cyber"))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e

        return gen()


class AsyncFakeStream(FakeStream):
    async def get_final_message(self):
        return super().get_final_message()


class FakeMessages:
    def __init__(self, stream_cls, stop_reason="tool_use"):
        self.calls, self.stream_cls, self.stop_reason = [], stream_cls, stop_reason

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self.stream_cls(_events(), self.stop_reason)


def fake_client(stream_cls=FakeStream, stop_reason="tool_use"):
    messages = FakeMessages(stream_cls, stop_reason)
    return NS(messages=messages, beta=NS(messages=messages)), messages


def test_stream_maps_sdk_events():
    pytest.importorskip("anthropic")
    client, messages = fake_client()
    llm = LLM(Settings(provider="anthropic", api_key="k"), anthropic_client=client)
    reply = llm.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}])
    assert (reply.text, reply.reasoning, reply.finish_reason) == ("Hi", "plan", "tool_calls")
    assert [(c.id, c.name, c.args()) for c in reply.tool_calls] == [("tu1", "quote", {"symbol": "KO"})]
    assert (reply.usage.prompt_tokens, reply.usage.completion_tokens) == (11, 7)
    assert messages.calls[0]["system"] == "s"


def test_refusal_raises():
    pytest.importorskip("anthropic")
    client, _ = fake_client(stop_reason="refusal")
    with pytest.raises(LLMRefusal) as info:
        LLM(Settings(provider="anthropic", api_key="k"), anthropic_client=client).complete("s", "q")
    assert info.value.category == "cyber"


def test_async_stream():
    pytest.importorskip("anthropic")
    client, _ = fake_client(AsyncFakeStream, stop_reason="end_turn")
    llm = LLM(Settings(provider="anthropic", api_key="k"), async_anthropic_client=client)
    reply = asyncio.run(llm.achat([{"role": "user", "content": "q"}]))
    assert reply.text == "Hi" and reply.finish_reason == "stop"
