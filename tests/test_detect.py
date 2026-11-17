import asyncio

import httpx2

from llm_lib.servers.detect import adetect_server, detect_server, models_url, server_kind

COLIBRI = {"object": "list", "data": [{"id": "glm-5.3-colibri", "object": "model", "owned_by": "colibri"}]}
LM_STUDIO = {"data": [{"id": "gemma-4-31b", "object": "model", "owned_by": "organization_owner"}]}


def answering(body, status=200, seen=None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(request)
        return httpx2.Response(status, json=body)

    return handler


def test_models_url():
    assert models_url("http://h:1222/v1/") == "http://h:1222/v1/models"
    assert models_url("http://h:1222") == "http://h:1222/v1/models"


def test_server_kind():
    assert server_kind(COLIBRI) == "colibri"
    assert server_kind(LM_STUDIO) == "lmstudio"
    assert server_kind({"data": [{"id": "x", "owned_by": "openai"}]}) == "openai-compatible"
    assert server_kind("nonsense") == "openai-compatible"


def test_detect_sends_the_key_and_reads_the_owner():
    seen = []
    kind = detect_server("http://h:1222/v1", api_key="k", transport=httpx2.MockTransport(answering(COLIBRI, seen=seen)))
    assert kind == "colibri"
    assert seen[0].url == "http://h:1222/v1/models" and seen[0].headers["Authorization"] == "Bearer k"


def test_async_detect_and_failures_are_generic():
    async def run():
        assert await adetect_server("http://h/v1", transport=httpx2.MockTransport(answering(LM_STUDIO))) == "lmstudio"
        refused = httpx2.MockTransport(answering({"error": "Invalid or missing API key."}, 401))
        assert await adetect_server("http://h/v1", transport=refused) == "openai-compatible"

        def down(request):
            raise httpx2.ConnectError("refused")

        assert await adetect_server("http://h/v1", transport=httpx2.MockTransport(down)) == "openai-compatible"

    asyncio.run(run())
