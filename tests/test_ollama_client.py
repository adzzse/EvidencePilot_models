import asyncio

from app.ollama_client import check_ollama, generate_embeddings
from app.settings import Settings


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
            return {"models": [{"name": "evidencopilot:latest"}]}

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
