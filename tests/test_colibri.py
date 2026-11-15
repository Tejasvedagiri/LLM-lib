import base64
import json

import httpx2
import pytest
from conftest import Server, delta, sse

from llm_lib import LLMStatusError, Settings
from llm_lib.servers.capacity import server_parallel
from llm_lib.servers.colibri import Colibri, choice, noul, score

MODELS = httpx2.Response(200, json={"object": "list", "data": [{"id": "glm-5.3-flash-colibri"}]})


def chat_ok(text="ok"):
    return httpx2.Response(200, content=sse(delta(content=text)))


def router(table):
    seen = []

    def handler(request):
        key = f"{request.method} {request.url.path}"
        seen.append((key, request))
        answer = table[key]
        return answer.pop(0) if isinstance(answer, list) else answer

    return handler, seen


def test_model_is_taken_from_the_server_when_unset():
    handler, seen = router({"GET /v1/models": MODELS, "POST /v1/chat/completions": chat_ok()})
    lines = []
    llm = Server(handler).client(Settings(provider="colibri"), log=lines.append)
    assert llm.complete("s", "p") == "ok"
    assert llm.complete("s", "p") == "ok"
    assert [k for k, _ in seen].count("GET /v1/models") == 1
    bodies = [json.loads(r.content) for k, r in seen if k.startswith("POST")]
    assert all(b["model"] == "glm-5.3-flash-colibri" for b in bodies)
    assert any("serves one model" in line for line in lines)


def test_penalties_are_left_out_and_said_once():
    server = Server(chat_ok())
    lines = []
    llm = server.client(Settings(provider="colibri", model="glm", frequency_penalty=0.4), log=lines.append)
    llm.complete("s", "p", repetition_penalty=1.1)
    llm.complete("s", "p")
    for body in server.bodies():
        assert not {"frequency_penalty", "repeat_penalty", "repetition_penalty"} & body.keys()
    assert sum("refuses" in line for line in lines) == 1


def test_queue_full_429_is_resent_after_retry_after_but_5xx_is_not():
    busy = httpx2.Response(429, headers={"Retry-After": "0"}, json={"error": {"code": "queue_full"}})
    server = Server(busy, busy, chat_ok("done"))
    assert server.client(Settings(provider="colibri", model="glm")).complete("s", "p") == "done"
    assert len(server.requests) == 3

    server = Server(httpx2.Response(500, json={"error": {"code": "engine_error"}}))
    with pytest.raises(LLMStatusError):
        server.client(Settings(provider="colibri", model="glm")).complete("s", "p")
    assert len(server.requests) == 1  # hours of generation aren't started over


def test_thinking_and_effort():
    server = Server(chat_ok())
    server.client(Settings(provider="colibri", model="glm", thinking=True, effort="high")).complete("s", "p")
    server.client(Settings(provider="colibri", model="glm", thinking=False, effort="high")).complete("s", "p")
    server.client(Settings(provider="vllm", model="q", no_think=True)).complete("s", "p")
    on, off, vllm = server.bodies()
    assert (on["enable_thinking"], on["reasoning_effort"]) == (True, "high")
    assert (off["enable_thinking"], off["reasoning_effort"]) == (False, "none")
    assert vllm["chat_template_kwargs"] == {"enable_thinking": False}


def test_api_key_from_coli_env(monkeypatch):
    monkeypatch.setenv("COLI_API_KEY", "local-secret")
    server = Server(chat_ok())
    server.client(Settings(provider="colibri", model="glm")).complete("s", "p")
    assert server.requests[0].headers["authorization"] == "Bearer local-secret"


def colibri_with(handler):
    return Colibri("http://127.0.0.1:8000/v1", api_key="k", http=httpx2.Client(transport=httpx2.MockTransport(handler)))


def test_decide():
    reply = {
        "model": "qwen36",
        "answers": {
            "urgency": {"type": "noul", "noul": 0.85},
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": {"billing": 0.06, "technical": 0.94},
                "confidence": 0.91,
            },
            "severity": {
                "type": "score",
                "score": 2.83,
                "legend": {"0": "none", "1": "minor", "2": "blocked", "3": "losing money"},
                "probabilities": {"0": 0.04, "1": 0.03, "2": 0.05, "3": 0.88},
                "confidence": 0.8,
            },
        },
        "usage": {"input_tokens": 86, "output_tokens": 15},
    }
    handler, seen = router({"POST /v1/systemone": httpx2.Response(200, json=reply)})
    d = colibri_with(handler).decide(
        "Stripe integration failing for 3 days, losing sales",
        {
            "urgency": noul("Does this message express urgency?"),
            "department": choice("Which team?", {"billing": "payments", "technical": "bugs"}),
            "severity": score("How severe?", ["none", "minor", "blocked", "losing money"]),
        },
        pin_state=True,
    )
    assert d["urgency"].value == 0.85
    assert (d["department"].choice, d["department"].confidence) == ("technical", 0.91)
    assert d["severity"].level == "losing money" and d["severity"].score == 2.83
    body = json.loads(seen[0][1].content)
    assert body["questions"]["severity"] == {
        "type": "score",
        "instructions": "How severe?",
        "criteria": ["none", "minor", "blocked", "losing money"],
    }
    assert body["pin_state"] is True and seen[0][1].headers["authorization"] == "Bearer k"
    with pytest.raises(ValueError):
        colibri_with(handler).decide("x", {})


def test_generate_image():
    png = b"\x89PNG\r\n fake"
    answer = {
        "data": [{"b64_json": base64.b64encode(png).decode()}],
        "colibri": {"width": 1024, "height": 576, "seed": 42, "steps": 8},
    }
    handler, seen = router(
        {
            "GET /v1/models": httpx2.Response(200, json={"data": [{"id": "qwen-image-2.1-colibri"}]}),
            "POST /v1/images/generations": httpx2.Response(200, json=answer),
        }
    )
    image = colibri_with(handler).generate_image("a red fox in the snow", size="1024x576", steps=8)
    assert (image.png, image.seed, image.width) == (png, 42, 1024)
    body = json.loads(seen[-1][1].content)
    assert body["model"] == "qwen-image-2.1-colibri" and body["size"] == "1024x576" and "seed" not in body


def test_health_and_capacity():
    health = {"status": "ok", "arch": "glm", "kv_slots": 2, "scheduler": {"capacity": 2, "active": 1, "queued": 0}}
    handler, _ = router({"GET /health": httpx2.Response(200, json=health)})
    assert colibri_with(handler).health()["arch"] == "glm"
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    assert server_parallel(Settings(provider="colibri", model="glm"), http=http) == (2, "colibri serves 2 KV slot(s)")


def test_model_lookup_is_retried_after_the_server_was_down():
    state = {"up": False}

    def handler(request):
        if not state["up"]:
            raise httpx2.ConnectError("refused", request=request)
        return MODELS if request.method == "GET" else chat_ok()

    from llm_lib import LLMUnavailable

    llm = Server(handler).client(Settings(provider="colibri"))
    with pytest.raises(LLMUnavailable):
        llm.complete("s", "p")
    state["up"] = True
    assert llm.complete("s", "p") == "ok"
    assert llm.model == "glm-5.3-flash-colibri"
