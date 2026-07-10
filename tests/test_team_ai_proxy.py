import pytest
from fastapi.testclient import TestClient

from app.main import app, get_settings
from app.models import GenerateResponse
from app.ollama_client import OllamaUnavailableError
from app.settings import Settings


@pytest.fixture(autouse=True)
def reset_overrides():
    app.dependency_overrides[get_settings] = lambda: Settings()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_ai_generate_defaults_model_and_stream_false(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    async def fake_generate(prompt, settings):
        captured["prompt"] = prompt
        captured["settings"] = settings
        return GenerateResponse(
            model="evidencopilot:latest",
            response="RAG combines retrieval with generation.",
            done=True,
        )

    monkeypatch.setattr("app.main.generate_text", fake_generate)

    response = client.post("/ai/generate", json={"prompt": "Explain RAG."})

    assert response.status_code == 200
    assert response.json() == {
        "model": "evidencopilot:latest",
        "response": "RAG combines retrieval with generation.",
        "done": True,
    }
    assert captured["prompt"] == "Explain RAG."
    assert captured["settings"].ollama_model == "evidencopilot:latest"


def test_ai_generate_rejects_model_override(client: TestClient):
    response = client.post(
        "/ai/generate",
        json={
            "prompt": "Use a custom model.",
            "model": "llama3.2:latest",
        },
    )

    assert response.status_code == 422


def test_ai_generate_rejects_empty_prompt(client: TestClient):
    response = client.post("/ai/generate", json={"prompt": "   "})

    assert response.status_code == 422


def test_ai_generate_rejects_stream_true_for_simple_api(client: TestClient):
    response = client.post(
        "/ai/generate",
        json={"prompt": "Explain RAG.", "stream": True},
    )

    assert response.status_code == 422


def test_ai_generate_rejects_stream_false_in_public_body(client: TestClient):
    response = client.post(
        "/ai/generate",
        json={"prompt": "Explain RAG.", "stream": False},
    )

    assert response.status_code == 422


def test_ai_models_returns_ollama_models(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def fake_models(settings):
        return {"models": [{"name": "evidencopilot:latest"}, {"name": "llama3.2:latest"}]}

    monkeypatch.setattr("app.main.list_models", fake_models)

    response = client.get("/ai/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {"name": "evidencopilot:latest"},
            {"name": "llama3.2:latest"},
        ]
    }


def test_ai_generate_returns_503_when_ollama_is_unavailable(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def fake_generate(prompt, settings):
        raise OllamaUnavailableError("Ollama generation failed")

    monkeypatch.setattr("app.main.generate_text", fake_generate)

    response = client.post("/ai/generate", json={"prompt": "Explain RAG."})

    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama generation failed"
