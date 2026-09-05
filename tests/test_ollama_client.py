import asyncio

import pytest

from app.ollama_client import (
    OllamaInvalidResponseError,
    check_ollama,
    generate_embeddings,
    generate_text,
)
from app.settings import Settings


def test_generation_sends_runtime_system_and_json_options(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"model": "qwen3.5:9b", "response": "{}", "done": True}

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, json):
            request.update(url=url, json=json)
            return Response()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", Client)

    result = asyncio.run(generate_text(
        "Judge claim quality",
        '{"claim":"A"}',
        Settings(ollama_model="qwen3.5:9b"),
    ))

    assert result.provider == "ollama"
    assert request["url"].endswith("/api/generate")
    assert request["json"] == {
        "model": "qwen3.5:9b",
        "system": "Judge claim quality",
        "prompt": '{"claim":"A"}',
        "format": "json",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_ctx": 262144,
        },
    }

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    asyncio.run(generate_text(
        "",
        "Reply OK",
        Settings(ollama_model="qwen3.5:9b"),
        {
            "type": "json_schema",
            "json_schema": {"name": "result", "strict": True, "schema": schema},
        },
    ))

    assert request["json"]["format"] == schema


@pytest.mark.parametrize("model", [123, ""])
def test_generation_rejects_malformed_ollama_metadata(monkeypatch, model):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"model": model, "response": "{}", "done": True}

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", Client)

    with pytest.raises(OllamaInvalidResponseError):
        asyncio.run(generate_text("system", "prompt", Settings()))


def test_batch_embeddings_use_ollama_embed_api(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[1, 2], [3, 4]]}

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, json):
            request.update(url=url, json=json)
            return Response()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", Client)
    settings = Settings(ollama_embedding_model="nomic-embed-text")

    result = asyncio.run(generate_embeddings(["one", "two"], settings))

    assert result == [[1.0, 2.0], [3.0, 4.0]]
    assert request["url"].endswith("/api/embed")
    assert request["json"] == {"model": "nomic-embed-text", "input": ["one", "two"]}


def test_health_degrades_when_a_required_model_is_missing(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "qwen3.5:9b"}]}

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, _):
            return Response()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", Client)

    result = asyncio.run(check_ollama(Settings()))

    assert result["ok"] is False


def test_embedding_only_health_does_not_advertise_local_generation(monkeypatch):
    import httpx
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        "models": [{"name": "nomic-embed-text:latest"}],
    }))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    result = asyncio.run(check_ollama(Settings(), require_generation=False))
    assert result == {"ok": True, "embedding_model_available": True}
