import json
import subprocess

import httpx2
import pytest
from conftest import Server, delta, sse

from llm_lib import FailoverLLM, LLMError, LLMUnavailable, Settings, is_retryable
from llm_lib.errors import LLMContextExceeded, LLMStatusError
from llm_lib.servers.capacity import effective_parallel, is_local_host, server_parallel
from llm_lib.servers.lmstudio import AsyncLMStudio, LMStudio


def routes(table: dict[str, httpx2.Response]):
    """A handler answering by "METHOD path"; records what it was asked."""
    seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        key = f"{request.method} {request.url.path}"
        seen.append((key, json.loads(request.content) if request.content else None))
        return table.get(key, httpx2.Response(404, json={"error": "nope"}))

    return handler, seen


V1_MODELS = {
    "models": [
        {
            "key": "qwen",
            "type": "llm",
            "loaded_instances": [{"id": "qwen:1", "config": {"context_length": 40960, "parallel": 4}}],
        },
        {"key": "gemma", "type": "llm", "loaded_instances": []},
    ]
}


def test_lmstudio_rest_load_unloads_others_first():
    handler, seen = routes(
        {
            "GET /api/v1/models": httpx2.Response(200, json=V1_MODELS),
            "POST /api/v1/models/unload": httpx2.Response(200, json={}),
            "POST /api/v1/models/load": httpx2.Response(200, json={"load_time_seconds": 3.2}),
        }
    )
    lms = LMStudio("http://h:1234/v1", http=httpx2.Client(transport=httpx2.MockTransport(handler)))
    assert lms.loaded_context("qwen") == 40960
    assert lms.load("gemma", 32768)["load_time_seconds"] == 3.2
    assert ("POST /api/v1/models/unload", {"instance_id": "qwen:1"}) in seen
    assert seen[-1] == ("POST /api/v1/models/load", {"model": "gemma", "context_length": 32768, "parallel": 1})
    assert lms.load("qwen", 40960, parallel=4)["already"]


def test_async_lmstudio_loads_unloads_and_reads_the_context():
    import asyncio

    handler, seen = routes(
        {
            "GET /api/v1/models": httpx2.Response(200, json=V1_MODELS),
            "GET /api/v0/models": httpx2.Response(
                200, json={"data": [{"id": "gemma", "state": "not-loaded"}, {"id": "qwen", "state": "loaded"}]}
            ),
            "POST /api/v1/models/unload": httpx2.Response(200, json={}),
            "POST /api/v1/models/load": httpx2.Response(200, json={"load_time_seconds": 3.2}),
        }
    )

    async def run():
        async with AsyncLMStudio("http://h:1234/v1", transport=httpx2.MockTransport(handler)) as lms:
            assert [m["id"] for m in await lms.models()] == ["qwen", "gemma"]
            assert await lms.loaded_context("qwen") == 40960
            assert (await lms.load("qwen", 40960, parallel=4))["already"]
            assert (await lms.load("gemma", 32768))["load_time_seconds"] == 3.2
            assert ("POST /api/v1/models/unload", {"instance_id": "qwen:1"}) in seen
            assert seen[-1] == ("POST /api/v1/models/load", {"model": "gemma", "context_length": 32768, "parallel": 1})
            assert await lms.unload_all() == ["qwen"]

    asyncio.run(run())


def test_async_lmstudio_without_rest_api():
    import asyncio

    handler, _ = routes({})

    async def run():
        async with AsyncLMStudio("http://h:8080/v1", transport=httpx2.MockTransport(handler)) as lms:
            assert await lms.models() == [] and await lms.loaded_context("m") is None
            with pytest.raises(LLMError, match="REST API"):
                await lms.load("m", 4096)
            with pytest.raises(LLMError, match="REST API"):
                await lms.unload_all()

    asyncio.run(run())


def test_client_loads_lmstudio_model_with_load_context_before_first_request():
    handler, seen = routes(
        {
            "GET /api/v0/models/qwen": httpx2.Response(200, json={"state": "not-loaded"}),
            "GET /api/v1/models": httpx2.Response(200, json=V1_MODELS),
            "POST /api/v1/models/load": httpx2.Response(200, json={}),
            "POST /v1/chat/completions": httpx2.Response(200, content=sse(delta(content="hi"))),
        }
    )
    llm = Server(handler).client(Settings(provider="lmstudio", model="qwen", load_context=65536, parallel=2))
    assert llm.complete("s", "p") == "hi"
    assert llm.complete("s", "p") == "hi"
    keys = [k for k, _ in seen]
    assert keys.count("POST /api/v1/models/load") == 1  # checked once, not per request
    assert ("POST /api/v1/models/load", {"model": "qwen", "context_length": 65536, "parallel": 2}) in seen


def test_models_unloaded_reloads_exactly_as_before_even_on_error():
    calls = []
    snapshot = [
        {
            "type": "llm",
            "modelKey": "qwen/qwen3.8-27b",
            "identifier": "q",
            "contextLength": 40960,
            "parallel": 4,
            "ttlMs": 600000,
        }
    ]

    def runner(args, **kw):
        calls.append(args[1:])
        out = json.dumps(snapshot) if args[1] == "ps" else ""
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

    lms = LMStudio(lms_path="/bin/lms", runner=runner)
    with pytest.raises(RuntimeError), lms.models_unloaded(log=lambda _: None):
        raise RuntimeError("image step failed")
    assert calls == [
        ["ps", "--json"],
        ["unload", "--all"],
        [
            "load",
            "qwen/qwen3.8-27b",
            "-y",
            "--context-length",
            "40960",
            "--parallel",
            "4",
            "--identifier",
            "q",
            "--ttl",
            "600",
        ],
    ]


def test_capacity():
    assert is_local_host("localhost") and is_local_host("192.168.1.9") and is_local_host("gpu-box")
    assert not is_local_host("api.openai.com")
    handler, _ = routes(
        {"GET /props": httpx2.Response(200, json={"total_slots": 4, "default_generation_settings": {"n_ctx": 40960}})}
    )
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    s = Settings(provider="llamacpp", model="m")
    assert server_parallel(s, http=http)[0] == 4
    assert server_parallel(s, window=16384, http=http)[0] == 2  # the shared context holds two whole windows
    assert effective_parallel(8, Settings(provider="openrouter", model="m", api_key="k"))[0] == 8
    assert server_parallel(Settings(provider="ollama", model="m"))[0] == 1

    handler, _ = routes({"GET /api/v1/models": httpx2.Response(200, json=V1_MODELS)})
    lms = LMStudio(http=httpx2.Client(transport=httpx2.MockTransport(handler)))
    assert server_parallel(Settings(provider="lmstudio", model="qwen"), lmstudio=lms)[0] == 4


def test_failover_when_primary_is_unreachable():
    def refuse(request):
        raise httpx2.ConnectError("refused", request=request)

    primary = Server(refuse)
    backup = Server(httpx2.Response(200, content=sse(delta(content="from backup"))))
    llm = FailoverLLM(
        primary.client(Settings(provider="lmstudio", model="big")),
        backup.client(Settings(provider="llamacpp", model="small")),
        log=lambda _: None,
    )
    assert llm.complete("s", "p") == "from backup"
    assert llm.on_fallback and llm.active.label == "llama.cpp"
    assert llm.complete("s", "p") == "from backup"
    assert len(primary.requests) == 1  # not asked again within retry_after


def test_failover_respects_primary_off():
    primary = Server(httpx2.Response(200, content=sse(delta(content="main"))))
    backup = Server(httpx2.Response(200, content=sse(delta(content="backup"))))
    off = [True]
    llm = FailoverLLM(
        primary.client(Settings(provider="lmstudio")),
        backup.client(Settings(provider="llamacpp")),
        primary_off=lambda: off[0],
    )
    assert llm.complete("s", "p") == "backup"
    off[0] = False
    assert llm.complete("s", "p") == "main"


def test_is_retryable():
    assert is_retryable(LLMUnavailable("down"))
    assert is_retryable(LLMStatusError("x", status_code=503))
    assert not is_retryable(LLMStatusError("x", status_code=401))
    assert not is_retryable(LLMContextExceeded("x", status_code=500))
    assert not is_retryable(ValueError())


def test_mock_provider():
    from llm_lib import LLM

    reply = LLM(Settings(provider="mock")).chat([{"role": "user", "content": "hello"}])
    assert 'You said: "hello"' in reply.text and reply.finish_reason == "stop"
