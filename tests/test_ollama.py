import asyncio
import json

import httpx2
import pytest
from conftest import Server, ndjson

from llm_lib import LLMError, Settings
from llm_lib.providers.ollama import to_ollama_messages

OLLAMA = Settings(provider="ollama", model="qwen3", context=16384, base_url="http://gpu:11434")


def test_chat_streams_and_sends_native_options():
    server = Server(
        httpx2.Response(
            200,
            content=ndjson(
                {"message": {"role": "assistant", "thinking": "hmm"}},
                {"message": {"role": "assistant", "content": "Hel"}},
                {"message": {"role": "assistant", "content": "lo"}},
                {"done": True, "done_reason": "stop", "prompt_eval_count": 30, "eval_count": 4},
            ),
        )
    )
    llm = server.client(OLLAMA.with_(no_think=True, max_tokens=256, temperature=0.3))
    reply = llm.chat([{"role": "user", "content": "hi"}], json_schema={"type": "object"})
    assert (reply.text, reply.reasoning, reply.finish_reason) == ("Hello", "hmm", "stop")
    assert reply.usage.prompt_tokens == 30
    body = server.bodies()[0]
    assert server.requests[0].url == "http://gpu:11434/api/chat"
    assert body["options"] == {"num_ctx": 16384, "temperature": 0.3, "num_predict": 256}
    assert body["think"] is False and body["format"] == {"type": "object"}
    assert "stream_options" not in body


def test_tool_calls():
    server = Server(
        httpx2.Response(
            200,
            content=ndjson(
                {
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "quote", "arguments": {"symbol": "KO"}}}],
                    }
                },
                {"done": True, "done_reason": "stop"},
            ),
        )
    )
    reply = server.client(OLLAMA).chat([{"role": "user", "content": "price?"}], tools=[{"type": "function"}])
    assert [(c.id, c.name, c.args()) for c in reply.tool_calls] == [("call_0", "quote", {"symbol": "KO"})]
    assert reply.finish_reason == "tool_calls"


def test_history_translated_to_ollama_shape():
    out = to_ollama_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "quote", "arguments": '{"symbol": "KO"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "61.2"},
        ]
    )
    assert out[0] == {"role": "user", "content": "look", "images": ["QUJD"]}
    assert out[1]["tool_calls"] == [{"function": {"name": "quote", "arguments": {"symbol": "KO"}}}]
    assert out[2] == {"role": "tool", "content": "61.2", "tool_name": "quote"}


def test_missing_model_says_how_to_fix():
    server = Server(httpx2.Response(404, json={"error": "model 'qwen3' not found"}))
    with pytest.raises(LLMError, match="ollama pull qwen3"):
        server.client(OLLAMA).complete("s", "p")


def test_unload_and_list_models():
    server = Server(httpx2.Response(200, json={"models": [{"name": "qwen3:latest"}]}))
    llm = server.client(OLLAMA)
    assert llm.available()  # "qwen3" matches "qwen3:latest"
    llm.unload()
    assert json.loads(server.requests[-1].content) == {"model": "qwen3", "keep_alive": 0}


def test_async():
    server = Server(httpx2.Response(200, content=ndjson({"message": {"content": "hi"}}, {"done": True})))
    assert asyncio.run(server.client(OLLAMA).acomplete("s", "p")) == "hi"
