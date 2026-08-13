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


REMOTE_SETTINGS = Settings(
    generation_provider="remote",
    generation_api_key="router-secret",
    generation_base_url="https://openrouter.ai/api/v1",
    generation_model="nvidia/nemotron-3-ultra-550b-a55b:free",
    generation_extra_body={"reasoning": {"effort": "none"}},
)


@pytest.mark.parametrize(
    ("configured", "api_key", "expected"),
    [
        ("auto", "", "ollama"),
        ("auto", "secret", "remote"),
        ("ollama", "secret", "ollama"),
        ("remote", "secret", "remote"),
    ],
)
def test_provider_selection_matrix(configured, api_key, expected):
    settings = REMOTE_SETTINGS.model_copy(update={
        "generation_provider": configured,
        "generation_api_key": api_key,
    })

    assert select_generation_provider(settings).name == expected


def test_forced_remote_requires_complete_configuration():
    with pytest.raises(GenerationConfigurationError, match="GENERATION_API_KEY"):
        select_generation_provider(Settings(generation_provider="remote"))


def test_remote_generation_uses_configured_openai_contract(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
                "choices": [{"message": {"content": '{"ok":true}'}}],
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
    provider = select_generation_provider(REMOTE_SETTINGS)

    result = asyncio.run(provider.generate("Return JSON", "Reply OK"))

    assert result.provider == "remote"
    assert result.response == '{"ok":true}'
    assert request == {
        "client": {"timeout": 600.0},
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {"Authorization": "Bearer router-secret"},
        "json": {
            "reasoning": {"effort": "none"},
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "messages": [
                {"role": "system", "content": "Return JSON"},
                {"role": "user", "content": "Reply OK"},
            ],
            "max_tokens": 8192,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "stream": False,
        },
    }
    assert "thinking" not in request["json"]


@pytest.mark.parametrize(
    "data",
    [
        {"choices": []},
        {
            "model": 123,
            "choices": [{"message": {"content": "{}"}}],
        },
    ],
)
def test_remote_rejects_malformed_response_once(monkeypatch, caplog, data):
    calls = 0

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = '{"unexpected":"raw-provider-body"}'

        def raise_for_status(self):
            return None

        def json(self):
            return data

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
    provider = select_generation_provider(REMOTE_SETTINGS)

    with pytest.raises(GenerationInvalidResponseError):
        asyncio.run(provider.generate("system", "prompt"))

    assert calls == 1
    assert "Invalid provider response" in caplog.text
    assert '{"unexpected":"raw-provider-body"}' in caplog.text


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, GenerationRateLimitError),
        (502, GenerationInvalidResponseError),
        (503, GenerationUnavailableError),
        (504, GenerationUnavailableError),
    ],
)
def test_remote_preserves_http_error_status(monkeypatch, status, expected):
    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, *_args, **_kwargs):
            return httpx.Response(
                status,
                request=httpx.Request(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                ),
            )

    monkeypatch.setattr("app.generation.httpx.AsyncClient", Client)
    provider = select_generation_provider(REMOTE_SETTINGS)

    with pytest.raises(expected):
        asyncio.run(provider.generate("system", "prompt"))


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (429, GenerationRateLimitError),
        (502, GenerationInvalidResponseError),
        (503, GenerationUnavailableError),
        (504, GenerationUnavailableError),
    ],
)
def test_remote_preserves_error_envelope_status(
    monkeypatch, caplog, code, expected
):
    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = '{"error":{"code":%d}}' % code

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "error": {
                    "code": code,
                    "message": "provider failed",
                    "metadata": {"error_type": "upstream_error"},
                }
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

    caplog.set_level("WARNING", logger="app.generation")
    monkeypatch.setattr("app.generation.httpx.AsyncClient", Client)
    provider = select_generation_provider(REMOTE_SETTINGS)

    with pytest.raises(expected):
        asyncio.run(provider.generate("system", "prompt"))

    assert f"code={code}" in caplog.text
    assert "error-type='upstream_error'" in caplog.text


def test_remote_failure_does_not_fall_back_or_expose_key(monkeypatch):
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
    settings = REMOTE_SETTINGS.model_copy(update={"generation_provider": "auto"})
    provider = select_generation_provider(settings)

    with pytest.raises(GenerationUnavailableError) as failure:
        asyncio.run(provider.generate("system", "prompt"))

    assert local_called is False
    assert "router-secret" not in str(failure.value)


def test_remote_health_checks_configured_model(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [{
                    "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
                }]
            }

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
    provider = select_generation_provider(REMOTE_SETTINGS)

    result = asyncio.run(provider.health())

    assert result == {
        "ok": True,
        "provider": "remote",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    }
    assert request == {
        "client": {"timeout": 5.0},
        "url": "https://openrouter.ai/api/v1/models",
        "headers": {"Authorization": "Bearer router-secret"},
    }
