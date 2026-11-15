import asyncio
import threading

import httpx2
import pytest
from conftest import Server, delta, sse

from llm_lib import LLMCancelled, LLMContextExceeded, LLMStatusError, LLMUnavailable, Settings

LMS = Settings(provider="lmstudio", model="qwen", retry_waits=(0,))


def ok(body: bytes) -> httpx2.Response:
    return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})


def test_chat_collects_content_reasoning_tools_and_usage():
    server = Server(
        ok(
            sse(
                delta(reasoning_content="hmm "),
                delta(content="Hello"),
                delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "quote", "arguments": '{"sym'}}]),
                delta(tool_calls=[{"index": 0, "function": {"arguments": 'bol": "KO"}'}}]),
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}},
            )
        )
    )
    reply = server.client(LMS).chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert reply.text == "Hello"
    assert reply.reasoning == "hmm "
    assert [(c.id, c.name, c.args()) for c in reply.tool_calls] == [("c1", "quote", {"symbol": "KO"})]
    assert reply.finish_reason == "tool_calls"
    assert reply.usage.total_tokens == 17
    assert reply.to_message()["tool_calls"][0]["function"]["name"] == "quote"
    body = server.bodies()[0]
    assert body["model"] == "qwen" and body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert server.requests[0].url == "http://127.0.0.1:1234/v1/chat/completions"
    assert server.requests[0].headers["authorization"] == "Bearer lm-studio"


def test_inline_think_is_split_off():
    server = Server(ok(sse(delta(content="<think>plan"), delta(content="</think>\n\nAnswer"))))
    reply = server.client(LMS).chat([{"role": "user", "content": "q"}])
    assert (reply.text, reply.reasoning) == ("Answer", "plan")


def test_sampling_settings_and_request_overrides():
    server = Server(ok(sse(delta(content="x"))))
    llm = server.client(LMS.with_(temperature=0.8, max_tokens=400, repetition_penalty=1.1))
    llm.chat([{"role": "user", "content": "q"}], temperature=0.2, stop=list("abcdef"), keep_alive="5m")
    body = server.bodies()[0]
    assert body["temperature"] == 0.2 and body["max_tokens"] == 400
    assert body["repeat_penalty"] == body["repetition_penalty"] == 1.1
    assert body["stop"] == list("abcd")  # OpenAI allows 4
    assert body["keep_alive"] == "5m"  # unknown keywords go into the body


def test_stream_options_rejected_once_then_never_sent_again():
    server = Server(
        httpx2.Response(400, text="Unrecognized request argument: stream_options"),
        ok(sse(delta(content="a"))),
    )
    llm = server.client(LMS)
    assert llm.complete("s", "p") == "a"
    assert llm.complete("s", "p") == "a"
    assert ["stream_options" in b for b in server.bodies()] == [True, False, False]


def test_passing_5xx_is_sent_again():
    server = Server(httpx2.Response(500, text="Internal Server Error"), ok(sse(delta(content="ok"))))
    lines = []
    assert server.client(LMS, log=lines.append).complete("s", "p") == "ok"
    assert len(server.requests) == 2
    assert any("sending it again" in line for line in lines)


def test_context_exceeded_is_not_retried():
    server = Server(httpx2.Response(500, text='{"error": "Context size has been exceeded"}'))
    with pytest.raises(LLMContextExceeded):
        server.client(LMS).complete("s", "p")
    assert len(server.requests) == 1


def test_4xx_raises_with_html_pages_called_out():
    server = Server(httpx2.Response(502, text="<!DOCTYPE html><html><body>Bad gateway</body></html>"))
    with pytest.raises(LLMStatusError, match="HTML error page") as info:
        server.client(LMS.with_(retry_waits=())).complete("s", "p")
    assert info.value.status_code == 502


def test_unreachable_server():
    def refuse(request):
        raise httpx2.ConnectError("Connection refused", request=request)

    with pytest.raises(LLMUnavailable, match="cannot reach LM Studio"):
        Server(refuse).client(LMS).complete("s", "p")


def test_json_schema_retried_without_response_format_when_reply_is_empty():
    server = Server(ok(sse(delta(content=""))), ok(sse(delta(content='{"ok": true}'))))
    text = server.client(LMS).complete("s", "p", json_schema={"type": "object"})
    assert text == '{"ok": true}'
    first, second = server.bodies()
    assert first["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert "response_format" not in second


def test_json_schema_is_prompt_only_where_not_enforced():
    server = Server(ok(sse(delta(content="{}"))))
    server.client(Settings(provider="koboldcpp", model="m")).complete("s", "p", json_schema={"type": "object"})
    assert "response_format" not in server.bodies()[0]


def test_text_completion_mode():
    server = Server(ok(sse({"choices": [{"text": "Once upon"}]}, {"choices": [{"text": " a time"}]})))
    assert server.client(LMS).complete_text("### Story:\n", stop=["###"]) == "Once upon a time"
    assert server.requests[0].url.path == "/v1/completions"
    assert server.bodies()[0]["prompt"] == "### Story:\n"


def test_no_think_switch_on_last_user_message():
    server = Server(ok(sse(delta(content="x"))))
    server.client(LMS.with_(no_think=True)).complete("s", "p")
    assert server.bodies()[0]["messages"][-1]["content"] == "p\n\n/no_think"


def test_cancel_stops_the_stream():
    cancel = threading.Event()
    server = Server(ok(sse(*[delta(content=f"{i} ") for i in range(10)])))
    seen = []
    with pytest.raises(LLMCancelled):
        for event in server.client(LMS).stream([{"role": "user", "content": "q"}], cancel=cancel):
            seen.append(event.text)
            if len(seen) == 2:
                cancel.set()
    assert seen == ["0 ", "1 "]


def test_server_error_event_in_stream():
    server = Server(ok(sse(delta(content="a"), {"error": {"message": "model crashed"}})))
    with pytest.raises(Exception, match="model crashed"):
        server.client(LMS).complete("s", "p")


def test_async_chat_and_retry():
    server = Server(httpx2.Response(503, text="loading"), ok(sse(delta(content="<think>x</think>hi"))))

    async def main():
        async with server.client(LMS) as llm:
            events = [e async for e in llm.astream([{"role": "user", "content": "q"}])]
            return events

    events = asyncio.run(main())
    assert [(e.type, e.text) for e in events if e.type in ("content", "reasoning")] == [
        ("reasoning", "x"),
        ("content", "hi"),
    ]
    assert events[-1].type == "done"


def test_list_models_and_available():
    server = Server(httpx2.Response(200, json={"data": [{"id": "qwen"}, {"id": "gemma"}]}))
    llm = server.client(LMS)
    assert llm.list_models() == ["gemma", "qwen"]
    assert llm.available()
    assert not server.client(LMS.with_(model="other")).available()
