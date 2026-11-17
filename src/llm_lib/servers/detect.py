"""Which server is behind an OpenAI-protocol URL, from its GET /v1/models, so an app with one "URL"
setting can pick the PRESETS entry that fits it (colibri's timeout and refused fields, LM Studio's
load and unload) without asking the user.

    provider = await adetect_server("http://192.168.86.244:1222/v1", api_key="...")   # "colibri"
    llm = LLM(Settings(provider=provider, base_url=..., ...))

colibri says `owned_by: "colibri"` on its model; LM Studio says `owned_by: "organization_owner"`.
Anything else (or no answer) is "openai-compatible", the generic preset.
"""

from __future__ import annotations

import re
from typing import Any

import httpx2

GENERIC = "openai-compatible"
DETECT_TIMEOUT_S = 10.0
_OWNERS = {"colibri": "colibri", "organization_owner": "lmstudio"}


def models_url(base_url: str) -> str:
    """http://host:1234 or http://host:1234/v1 -> http://host:1234/v1/models."""
    base = base_url.strip().rstrip("/")
    return f"{base}/models" if re.search(r"/v\d+$", base) else f"{base}/v1/models"


def server_kind(models: Any) -> str:
    """The provider name for a /v1/models answer (the parsed JSON)."""
    items = models.get("data") if isinstance(models, dict) else None
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and (kind := _OWNERS.get(str(item.get("owned_by") or ""))):
            return kind
    return GENERIC


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _kind_of(resp: httpx2.Response) -> str:
    try:
        return server_kind(resp.json()) if resp.status_code < 400 else GENERIC
    except ValueError:
        return GENERIC


def detect_server(base_url: str, *, api_key: str | None = None, transport: httpx2.BaseTransport | None = None) -> str:
    """The provider name: "colibri", "lmstudio" or "openai-compatible" (also when there's no answer)."""
    try:
        with httpx2.Client(timeout=DETECT_TIMEOUT_S, transport=transport) as http:
            return _kind_of(http.get(models_url(base_url), headers=_headers(api_key)))
    except httpx2.HTTPError:
        return GENERIC


async def adetect_server(
    base_url: str, *, api_key: str | None = None, transport: httpx2.AsyncBaseTransport | None = None
) -> str:
    """detect_server on an async client."""
    try:
        async with httpx2.AsyncClient(timeout=DETECT_TIMEOUT_S, transport=transport) as http:
            return _kind_of(await http.get(models_url(base_url), headers=_headers(api_key)))
    except httpx2.HTTPError:
        return GENERIC
