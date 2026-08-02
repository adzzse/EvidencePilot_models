import asyncio

import httpx
import pytest

from app.generation import (
    GenerationConfigurationError,
    GenerationInvalidResponseError,
    GenerationUnavailableError,
    select_generation_provider,
)
from app.settings import Settings


@pytest.mark.parametrize(
    ("configured", "api_key", "expected"),
    [
        ("auto", "", "ollama"),
        ("auto", "secret", "gemini"),
        ("ollama", "secret", "ollama"),
        ("gemini", "secret", "gemini"),
    ],
)
def test_provider_selection_matrix(configured, api_key, expected):
    provider = select_generation_provider(Settings(
        generation_provider=configured,
        gemini_api_key=api_key,
    ))

    assert provider.name == expected


def test_forced_gemini_requires_api_key():
    with pytest.raises(GenerationConfigurationError, match="GEMINI_API_KEY"):
        select_generation_provider(Settings(
            generation_provider="gemini",
            gemini_api_key="",
        ))


def test_gemini_generation_uses_official_rest_contract(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [{
                    "content": {"parts": [{"text": '{"quality":"GOOD"}'}]},
                }],
                "modelVersion": "gemini-3.6-flash",
            }

    class Client:
        def __init__(self, **kwargs):
            request["client"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, headers, json):
            request.update(url=url, headers=headers, json=json)
            return Response()

    monkeypatch.setattr("app.generation.httpx.AsyncClient", Client)
    provider = select_generation_provider(Settings(
        generation_provider="gemini",
        gemini_api_key="secret",
        gemini_model="gemini-3.6-flash",
    ))

    result = asyncio.run(provider.generate(
        "Judge claim quality",
        '{"claim":"A"}',
    ))

    assert result.model == "gemini-3.6-flash"
    assert result.provider == "gemini"
    assert result.response == '{"quality":"GOOD"}'
    assert request["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.6-flash:generateContent"
    )
    assert request["headers"] == {"x-goog-api-key": "secret"}
    assert request["json"] == {
        "systemInstruction": {"parts": [{"text": "Judge claim quality"}]},
        "contents": [{
            "role": "user",
            "parts": [{"text": '{"claim":"A"}'}],
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 8192,
        },
    }


def test_gemini_failure_does_not_fall_back_or_expose_key(monkeypatch):
    local_called = False

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, *_args, **_kwargs):
            raise httpx.ConnectError("offline")

    async def local_generation(*_):
        nonlocal local_called
        local_called = True

    monkeypatch.setattr("app.generation.httpx.AsyncClient", Client)
    monkeypatch.setattr("app.generation.generate_with_ollama", local_generation)
    provider = select_generation_provider(Settings(
        generation_provider="auto",
        gemini_api_key="very-secret-key",
    ))

    with pytest.raises(GenerationUnavailableError) as failure:
        asyncio.run(provider.generate("system", "prompt"))

    assert local_called is False
    assert "very-secret-key" not in str(failure.value)


def test_gemini_rejects_malformed_upstream_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"candidates": []}

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("app.generation.httpx.AsyncClient", Client)
    provider = select_generation_provider(Settings(
        generation_provider="gemini",
        gemini_api_key="secret",
    ))

    with pytest.raises(GenerationInvalidResponseError):
        asyncio.run(provider.generate("system", "prompt"))


def test_gemini_rejects_non_text_model_version(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                "modelVersion": 123,
            }

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("app.generation.httpx.AsyncClient", Client)
    provider = select_generation_provider(Settings(
        generation_provider="gemini",
        gemini_api_key="secret",
    ))

    with pytest.raises(GenerationInvalidResponseError):
        asyncio.run(provider.generate("system", "prompt"))


def test_gemini_health_checks_configured_model(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"name": "models/gemini-3.6-flash"}

    class Client:
        def __init__(self, **kwargs):
            request["client"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url, headers):
            request.update(url=url, headers=headers)
            return Response()

    monkeypatch.setattr("app.generation.httpx.AsyncClient", Client)
    provider = select_generation_provider(Settings(
        generation_provider="gemini",
        gemini_api_key="secret",
        gemini_model="gemini-3.6-flash",
    ))

    result = asyncio.run(provider.health())

    assert result == {
        "ok": True,
        "provider": "gemini",
        "model": "gemini-3.6-flash",
    }
    assert request["client"] == {"timeout": 5.0}
    assert request["url"].endswith("/v1beta/models/gemini-3.6-flash")
    assert request["headers"] == {"x-goog-api-key": "secret"}
