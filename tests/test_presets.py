import httpx2
import pytest
from conftest import Server, delta, sse

from llm_lib import LLM, LLMError, Settings
from llm_lib.servers.capacity import server_parallel
from llm_lib.settings import PRESETS

HOSTED = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "huggingface": ("https://router.huggingface.co/v1", "HF_TOKEN"),
}

LOCAL = {
    "lmstudio": 1234,
    "llamacpp": 8080,
    "vllm": 8000,
    "sglang": 30000,
    "colibri": 8000,
    "mlx": 8080,
    "koboldcpp": 5001,
    "textgen": 5000,
    "tabbyapi": 5000,
    "localai": 8080,
    "jan": 1337,
}


@pytest.mark.parametrize(("provider", "expected"), HOSTED.items())
def test_hosted_presets(provider, expected, monkeypatch):
    url, key_env = expected
    for preset in PRESETS.values():
        for name in preset.key_env:
            monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMError, match=key_env):
        LLM(Settings(provider=provider, model="m"))
    monkeypatch.setenv(key_env, "sk-test")
    server = Server(httpx2.Response(200, content=sse(delta(content="hi"))))
    assert server.client(Settings(provider=provider, model="m")).complete("s", "p") == "hi"
    request = server.requests[0]
    assert str(request.url) == f"{url}/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert server_parallel(Settings(provider=provider, model="m"))[0] is None  # hosted: no limit


@pytest.mark.parametrize(("provider", "port"), LOCAL.items())
def test_local_presets(provider, port):
    s = Settings(provider=provider, model="m")
    assert s.resolved_base_url({}) == f"http://127.0.0.1:{port}/v1"
    assert s.preset.local and s.resolved_api_key({})


@pytest.mark.parametrize(
    ("alias", "provider"),
    [
        ("grok", "xai"),
        ("google", "gemini"),
        ("oobabooga", "textgen"),
        ("mlx-lm", "mlx"),
        ("hf", "huggingface"),
        ("kobold", "koboldcpp"),
        ("coli", "colibri"),
        ("tabby", "tabbyapi"),
        ("nim", "nvidia"),
    ],
)
def test_aliases(alias, provider):
    assert Settings(provider=alias).provider == provider


def test_base_url_env_override(monkeypatch):
    monkeypatch.setenv("SGLANG_BASE_URL", "http://gpu-box:30000/v1")
    assert Settings(provider="sglang").resolved_base_url() == "http://gpu-box:30000/v1"


def test_stream_options_rejected_with_422():
    server = Server(
        httpx2.Response(422, json={"detail": "Extra inputs are not permitted: stream_options"}),
        httpx2.Response(200, content=sse(delta(content="ok"))),
    )
    assert server.client(Settings(provider="mistral", model="m", api_key="k")).complete("s", "p") == "ok"
    assert "stream_options" not in server.bodies()[-1]


def test_hosted_429_waits_for_retry_after():
    server = Server(
        httpx2.Response(429, headers={"retry-after": "0"}, json={"error": "rate limited"}),
        httpx2.Response(200, content=sse(delta(content="ok"))),
    )
    assert server.client(Settings(provider="groq", model="m", api_key="k")).complete("s", "p") == "ok"


def test_effort_goes_out_as_reasoning_effort():
    server = Server(httpx2.Response(200, content=sse(delta(content="ok"))))
    server.client(Settings(provider="openrouter", model="m", api_key="k", effort="low")).complete("s", "p")
    assert server.bodies()[0]["reasoning_effort"] == "low"


def test_ollama_thinking_switches():
    from llm_lib.providers.ollama import OllamaProvider
    from llm_lib.types import Request

    req = Request(messages=[{"role": "user", "content": "q"}])
    build = OllamaProvider().build
    assert build(req, Settings(provider="ollama", model="gpt-oss", effort="high")).json["think"] == "high"
    assert build(req, Settings(provider="ollama", model="q", thinking=True)).json["think"] is True
    assert "think" not in build(req, Settings(provider="ollama", model="q")).json
