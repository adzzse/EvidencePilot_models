import asyncio

import httpx
import pytest

from app.generation import (
    GenerationConfigurationError,
    GenerationInvalidResponseError,
    GenerationRateLimitError,
    GenerationUnavailableError,
    select_generation_provider,
)
from app.settings import Settings


@pytest.mark.parametrize(
    ("configured", "compatible_key", "gemini_key", "expected"),
    [
        ("auto", "", "", "ollama"),
        ("auto", "zen-secret", "", "openai_compatible"),
        ("auto", "", "gemini-secret", "ollama"),
        ("ollama", "zen-secret", "gemini-secret", "ollama"),
        ("openai_compatible", "zen-secret", "", "openai_compatible"),
        ("gemini", "", "gemini-secret", "gemini"),
    ],
)
def test_provider_selection_matrix(
    configured, compatible_key, gemini_key, expected
):
    provider = select_generation_provider(Settings(
        generation_provider=configured,
        openai_compatible_api_key=compatible_key,
        gemini_api_key=gemini_key,
    ))

    assert provider.name == expected


def test_forced_openai_compatible_requires_api_key():
    with pytest.raises(
        GenerationConfigurationError,
        match="OPENAI_COMPATIBLE_API_KEY",
    ):
        select_generation_provider(Settings(
            generation_provider="openai_compatible",
            openai_compatible_api_key="",
        ))


def test_forced_gemini_requires_api_key():
    with pytest.raises(GenerationConfigurationError, match="GEMINI_API_KEY"):
        select_generation_provider(Settings(
            generation_provider="gemini",
            gemini_api_key="",
        ))


def test_openai_compatible_generation_uses_chat_completions(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "deepseek-v4-flash-free",
                "choices": [{"message": {"content": '{"quality":"GOOD"}'}}],
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
        generation_provider="openai_compatible",
        openai_compatible_api_key="zen-secret",
    ))

    result = asyncio.run(provider.generate("Judge claim quality", '{"claim":"A"}'))

    assert result.provider == "openai_compatible"
    assert result.model == "deepseek-v4-flash-free"
    assert result.response == '{"quality":"GOOD"}'
    assert request == {
        "client": {"timeout": 120.0},
        "url": "https://opencode.ai/zen/v1/chat/completions",
        "headers": {"Authorization": "Bearer zen-secret"},
        "json": {
            "model": "deepseek-v4-flash-free",
            "messages": [
                {"role": "system", "content": "Judge claim quality"},
                {"role": "user", "content": '{"claim":"A"}'},
            ],
            "max_tokens": 8192,
            "temperature": 0,
            "stream": False,
            "thinking": {"type": "disabled"},
        },
    }


def test_openai_compatible_rejects_malformed_response_once(monkeypatch, caplog):
    calls = 0

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = '{"unexpected":"raw-provider-body"}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            return Response()

    caplog.set_level("WARNING", logger="app.generation")
    monkeypatch.setattr("app.generation.httpx.AsyncClient", Client)
    provider = select_generation_provider(Settings(
        generation_provider="openai_compatible",
        openai_compatible_api_key="zen-secret",
    ))

    with pytest.raises(GenerationInvalidResponseError):
        asyncio.run(provider.generate("system", "prompt"))

    assert calls == 1
    assert "Invalid provider response: IndexError" in caplog.text
    assert '{"unexpected":"raw-provider-body"}' in caplog.text


def test_openai_compatible_preserves_rate_limit(monkeypatch):
    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, *_args, **_kwargs):
            return httpx.Response(
                429,
                request=httpx.Request("POST", "https://opencode.ai/zen/v1/chat/completions"),
            )

    monkeypatch.setattr("app.generation.httpx.AsyncClient", Client)
    provider = select_generation_provider(Settings(
        generation_provider="openai_compatible",
        openai_compatible_api_key="zen-secret",
    ))

    with pytest.raises(GenerationRateLimitError, match="rate limit"):
        asyncio.run(provider.generate("system", "prompt"))


def test_openai_compatible_failure_does_not_fall_back_or_expose_key(monkeypatch):
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
        openai_compatible_api_key="zen-secret",
    ))

    with pytest.raises(GenerationUnavailableError) as failure:
        asyncio.run(provider.generate("system", "prompt"))

    assert local_called is False
    assert "zen-secret" not in str(failure.value)


def test_openai_compatible_health_checks_configured_model(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "deepseek-v4-flash-free"}]}

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
        generation_provider="openai_compatible",
        openai_compatible_api_key="zen-secret",
    ))

    result = asyncio.run(provider.health())

    assert result == {
        "ok": True,
        "provider": "openai_compatible",
        "model": "deepseek-v4-flash-free",
    }
    assert request == {
        "client": {"timeout": 5.0},
        "url": "https://opencode.ai/zen/v1/models",
        "headers": {"Authorization": "Bearer zen-secret"},
    }


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
        generation_provider="gemini",
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
