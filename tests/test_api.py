import asyncio
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import app.extraction as extraction
import app.main as main
from app.extraction import ExtractedMarkdown, extract_from_url
from app.models import ExtractRequest, GenerateResponse
from app.settings import Settings


SETTINGS = Settings(
    model_api_key="test-key",
    extraction_allowed_hosts=("storage.test",),
)
HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture(autouse=True)
def settings_override():
    main.app.dependency_overrides[main.get_settings] = lambda: SETTINGS
    yield
    main.app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


def test_health_degrades_without_taking_api_offline(client: TestClient, monkeypatch):
    async def unavailable(_):
        raise main.OllamaUnavailableError("offline")

    monkeypatch.setattr(main, "check_ollama", unavailable)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_post_endpoints_require_api_key(client: TestClient):
    assert client.post("/ai/embeddings", json={"text": "claim"}).status_code == 401


def test_generate_uses_configured_model(client: TestClient, monkeypatch):
    async def generate(prompt, settings):
        assert prompt == "Review this"
        assert settings.ollama_model == SETTINGS.ollama_model
        return GenerateResponse(model=settings.ollama_model, response="done", done=True)

    monkeypatch.setattr(main, "generate_text", generate)
    response = client.post("/ai/generate", headers=HEADERS, json={"prompt": "Review this"})

    assert response.status_code == 200
    assert response.json()["response"] == "done"


def test_single_and_batch_embeddings_preserve_order(client: TestClient, monkeypatch):
    async def embed(texts, _):
        return [[float(index)] for index, _text in enumerate(texts)]

    monkeypatch.setattr(main, "generate_embeddings", embed)

    single = client.post("/ai/embeddings", headers=HEADERS, json={"text": "one"})
    batch = client.post(
        "/ai/embeddings/batch",
        headers=HEADERS,
        json={"texts": ["one", "two"]},
    )

    assert single.json() == {"embedding": [0.0]}
    assert batch.json() == {"embeddings": [[0.0], [1.0]]}


def test_batch_rejects_more_than_64_texts(client: TestClient):
    response = client.post(
        "/ai/embeddings/batch",
        headers=HEADERS,
        json={"texts": ["chunk"] * 65},
    )
    assert response.status_code == 422


def test_extract_returns_markdown(client: TestClient, monkeypatch):
    async def extract(payload, _):
        assert payload.filename == "paper.pdf"
        return ExtractedMarkdown("paper.pdf", "mineru", "# Paper")

    monkeypatch.setattr(main, "extract_from_url", extract)
    response = client.post(
        "/extract",
        headers=HEADERS,
        json={
            "document_id": str(uuid4()),
            "filename": "paper.pdf",
            "content_type": "application/pdf",
            "download_url": "https://storage.test/paper.pdf?signature=test",
        },
    )

    assert response.status_code == 200
    assert response.json()["method"] == "mineru"


def test_extract_rejects_untrusted_download_host(client: TestClient):
    response = client.post(
        "/extract",
        headers=HEADERS,
        json={
            "document_id": str(uuid4()),
            "filename": "paper.pdf",
            "content_type": "application/pdf",
            "download_url": "https://evil.test/paper.pdf",
        },
    )
    assert response.status_code == 422


def test_extraction_routes_pdf_to_configured_mineru(monkeypatch):
    received = {}

    async def download(*_):
        return None

    async def mineru(*args):
        received["args"] = args
        return "# Extracted"

    monkeypatch.setattr("app.extraction._download", download)
    monkeypatch.setattr("app.extraction.extract_with_mineru", mineru)
    payload = ExtractRequest(
        document_id=uuid4(),
        filename="paper.pdf",
        content_type="application/pdf",
        download_url="https://storage.test/paper.pdf",
    )

    settings = SETTINGS.model_copy(update={
        "mineru_command": "C:/tools/mineru.exe",
        "mineru_backend": "pipeline",
    })
    result = asyncio.run(extract_from_url(payload, settings))
    assert result.method == "mineru"
    assert result.markdown == "# Extracted"
    assert received["args"][-2:] == ("C:/tools/mineru.exe", "pipeline")


def test_extraction_routes_docx_to_liteparse(monkeypatch):
    async def download(*_):
        return None

    monkeypatch.setattr("app.extraction._download", download)
    monkeypatch.setattr("app.extraction.extract_with_liteparse", lambda _: "# Extracted")
    payload = ExtractRequest(
        document_id=uuid4(),
        filename="paper.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        download_url="https://storage.test/paper.docx",
    )

    result = asyncio.run(extract_from_url(payload, SETTINGS))
    assert result.method == "liteparse"
    assert result.markdown == "# Extracted"


def test_mineru_reads_markdown_from_current_output_layout(tmp_path):
    markdown_path = tmp_path / "input" / "hybrid_auto" / "input.md"
    markdown_path.parent.mkdir(parents=True)
    markdown_path.write_text("# Extracted", encoding="utf-8")

    read_output = getattr(extraction, "_read_mineru_markdown", None)
    assert read_output is not None
    assert read_output(tmp_path, "input") == "# Extracted"


def test_settings_reads_mineru_command(monkeypatch):
    monkeypatch.setenv("MINERU_COMMAND", "C:/tools/mineru.exe")
    monkeypatch.setenv("MINERU_BACKEND", "pipeline")

    settings = Settings.from_env()

    assert getattr(settings, "mineru_command", None) == "C:/tools/mineru.exe"
    assert settings.mineru_backend == "pipeline"
