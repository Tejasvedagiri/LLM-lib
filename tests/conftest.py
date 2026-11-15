import json
from collections.abc import Callable

import httpx2
import pytest

from llm_lib import LLM, Settings


def sse(*chunks: dict | str) -> bytes:
    """A server-sent-events body from chunk dicts (or raw strings), ending in [DONE]."""
    lines = [f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def delta(**fields) -> dict:
    return {"choices": [{"index": 0, "delta": fields}]}


def ndjson(*parts: dict) -> bytes:
    return "".join(json.dumps(p) + "\n" for p in parts).encode()


class Server:
    """A MockTransport that answers each request with the next canned response and records requests."""

    def __init__(self, *responses: httpx2.Response | Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return response(request) if callable(response) else response

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.content]

    def client(self, settings: Settings, **kwargs) -> LLM:
        return LLM(settings, transport=httpx2.MockTransport(self), async_transport=httpx2.MockTransport(self), **kwargs)


@pytest.fixture
def lines() -> list[str]:
    return []
