import pytest
from fastapi.testclient import TestClient

from app.main import app, get_settings, run_claim_analysis
from app.models import ClaimAnalysisRequest, ClaimAnalysisResponse
from app.ollama_client import (
    OllamaInvalidResponseError,
    OllamaUnavailableError,
    analyze_claim,
)
from app.settings import Settings


def override_settings() -> Settings:
    return Settings()


@pytest.fixture(autouse=True)
def reset_overrides():
    app.dependency_overrides[get_settings] = override_settings
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def sample_payload() -> dict:
    return {
        "claim": "Traceable evidence improves review quality.",
        "source_id": "source-1",
        "title": "agile-risk-management.pdf",
        "excerpt": "Evidence traceability links claims to source material so reviewers can evaluate support quality.",
    }


def test_process_claim_accepts_missing_api_key(client: TestClient):
    async def fake_analysis():
        return ClaimAnalysisResponse(
            verdict="supported",
            confidence=0.92,
            matched_source_ids=["source-1"],
            missing_evidence=[],
            explanation="The provided source directly supports the claim.",
        )

    app.dependency_overrides[run_claim_analysis] = fake_analysis

    response = client.post("/process/claim", json=sample_payload())

    assert response.status_code == 200
    assert response.json()["verdict"] == "supported"


def test_process_claim_rejects_empty_claim(client: TestClient):
    payload = sample_payload()
    payload["claim"] = "   "

    response = client.post("/process/claim", json=payload)

    assert response.status_code == 422


def test_process_claim_rejects_empty_source_id(client: TestClient):
    payload = sample_payload()
    payload["source_id"] = "   "

    response = client.post("/process/claim", json=payload)

    assert response.status_code == 422


def test_process_claim_rejects_missing_excerpt(client: TestClient):
    payload = sample_payload()
    del payload["excerpt"]

    response = client.post("/process/claim", json=payload)

    assert response.status_code == 422


def test_process_claim_rejects_oversized_excerpt(client: TestClient):
    payload = sample_payload()
    payload["excerpt"] = "x" * 4001

    response = client.post("/process/claim", json=payload)

    assert response.status_code == 422


def test_process_claim_rejects_old_sources_shape(client: TestClient):
    payload = {
        "claim": "Traceable evidence improves review quality.",
        "sources": [
            {
                "id": "source-1",
                "title": "agile-risk-management.pdf",
                "excerpt": "Evidence traceability links claims to source material.",
            }
        ],
    }

    response = client.post("/process/claim", json=payload)

    assert response.status_code == 422


def test_process_claim_returns_structured_model_result(client: TestClient):
    async def fake_analysis():
        return ClaimAnalysisResponse(
            verdict="supported",
            confidence=0.92,
            matched_source_ids=["source-1"],
            missing_evidence=[],
            explanation="The provided source directly supports the claim.",
        )

    app.dependency_overrides[run_claim_analysis] = fake_analysis

    response = client.post("/process/claim", json=sample_payload())

    assert response.status_code == 200
    assert response.json() == {
        "verdict": "supported",
        "confidence": 0.92,
        "matched_source_ids": ["source-1"],
        "missing_evidence": [],
        "explanation": "The provided source directly supports the claim.",
    }


def test_process_claim_returns_503_when_ollama_is_unavailable(client: TestClient):
    async def fake_analysis():
        raise OllamaUnavailableError("Ollama generation failed")

    app.dependency_overrides[run_claim_analysis] = fake_analysis

    response = client.post("/process/claim", json=sample_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama generation failed"


def test_process_claim_returns_502_when_model_json_is_invalid(client: TestClient):
    async def fake_analysis():
        raise OllamaInvalidResponseError("Ollama returned invalid claim-analysis JSON")

    app.dependency_overrides[run_claim_analysis] = fake_analysis

    response = client.post("/process/claim", json=sample_payload())

    assert response.status_code == 502
    assert response.json()["detail"] == "Ollama returned invalid claim-analysis JSON"


def test_health_returns_ollama_status(client: TestClient):
    async def fake_health():
        return {
            "ok": True,
            "model_available": True,
            "embedding_model_available": True,
        }

    def fake_liteparse_health():
        return {
            "ok": True,
            "package": "liteparse",
        }

    from app.main import check_liteparse_health, check_ollama_health

    app.dependency_overrides[check_ollama_health] = fake_health
    app.dependency_overrides[check_liteparse_health] = fake_liteparse_health

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "evidencopilot:latest",
        "embedding_model": "nomic-embed-text",
        "ollama": {
            "ok": True,
            "model_available": True,
            "embedding_model_available": True,
        },
        "liteparse": {
            "ok": True,
            "package": "liteparse",
        },
    }


def test_health_reports_degraded_when_dependency_is_unavailable():
    import asyncio
    from app.main import health

    result = asyncio.run(
        health(
            Settings(),
            {"ok": True, "model_available": True, "embedding_model_available": True},
            {"ok": False, "package": "liteparse"},
        )
    )

    assert result["status"] == "degraded"


def test_ollama_health_checks_embedding_model_alias(monkeypatch):
    import asyncio
    from app.ollama_client import check_ollama

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "models": [
                    {"name": "evidencopilot:latest"},
                    {"name": "nomic-embed-text:latest"},
                ]
            }

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def get(self, url):
            assert url.endswith("/api/tags")
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)

    result = asyncio.run(check_ollama(Settings()))

    assert result == {
        "ok": True,
        "model_available": True,
        "embedding_model_available": True,
    }


def test_ollama_health_rejects_invalid_response(monkeypatch):
    import asyncio
    import httpx
    from app.ollama_client import check_ollama

    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"models": [{"size": 123}]})
    )
    monkeypatch.setattr(
        "app.ollama_client.httpx.AsyncClient",
        lambda timeout: async_client(timeout=timeout, transport=transport),
    )

    with pytest.raises(OllamaInvalidResponseError):
        asyncio.run(check_ollama(Settings()))


def test_liteparse_health_reports_package_availability(monkeypatch):
    from app.extraction import check_liteparse

    monkeypatch.setattr("app.extraction.importlib_util.find_spec", lambda package: object())

    assert check_liteparse() == {
        "ok": True,
        "package": "liteparse",
    }


def test_extract_pdf_returns_liteparse_markdown_without_storing_source(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_extract_with_liteparse(filename: str, raw: bytes) -> str:
        assert filename == "original.pdf"
        assert raw == b"%PDF not a real file"
        return "# Extracted document\n\nLiteParse markdown."

    monkeypatch.setattr("app.extraction.extract_with_liteparse", fake_extract_with_liteparse)

    response = client.post(
        "/extract",
        files={"file": ("original.pdf", b"%PDF not a real file", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "original.pdf",
        "method": "liteparse",
        "markdown": "# Extracted document\n\nLiteParse markdown.",
    }


def test_demo_storage_routes_are_not_exposed(client: TestClient):
    post_sources = client.post(
        "/sources",
        files={"files": ("source.txt", b"source text", "text/plain")},
    )
    match_claim = client.post("/match/claim", json={"claim": "A claim."})
    post_papers = client.post(
        "/papers",
        files={"file": ("paper.txt", b"paper text", "text/plain")},
    )
    review_paper = client.post("/review/paper", json={"paper_id": "paper-1"})
    process_document = client.post(
        "/ai/process-document",
        files={"file": ("source.pdf", b"%PDF test", "application/pdf")},
    )

    assert post_sources.status_code == 404
    assert client.get("/sources").status_code == 404
    assert client.get("/sources/references").status_code == 404
    assert match_claim.status_code == 404
    assert post_papers.status_code == 404
    assert review_paper.status_code == 404
    assert process_document.status_code == 404


def test_sparse_embed_returns_weighted_token_counts(client: TestClient, monkeypatch):
    import math

    class FakeTokenizer:
        def __call__(self, text):
            assert text == "repeat repeat once"
            return {"input_ids": [101, 10, 10, 11, 102, 0]}

    monkeypatch.setattr("app.sparse._get_tokenizer", lambda: FakeTokenizer())

    response = client.post("/api/sparse-embed", json={"text": "repeat repeat once"})

    assert response.status_code == 200
    assert response.json()["indices"] == [10, 11]
    assert response.json()["values"] == pytest.approx([math.log(3), math.log(2)])


def test_extract_returns_json_error_when_liteparse_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from app.extraction import LiteParseExtractionError

    def fake_extract_with_liteparse(filename: str, raw: bytes) -> str:
        raise LiteParseExtractionError("LiteParse extraction failed: bad document")

    monkeypatch.setattr("app.extraction.extract_with_liteparse", fake_extract_with_liteparse)

    response = client.post(
        "/extract",
        files={"file": ("broken.pdf", b"%PDF broken", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "LiteParse extraction failed: bad document"


def test_generate_text_success(client: TestClient):
    from app.main import run_generate_text
    from app.models import GenerateResponse

    async def fake_run_generate_text():
        return GenerateResponse(model="evidencopilot:latest", response="Generated answer.", done=True)

    app.dependency_overrides[run_generate_text] = fake_run_generate_text

    response = client.post("/ai/generate", json={"prompt": "Explain traceability."})

    assert response.status_code == 200
    assert response.json() == {
        "model": "evidencopilot:latest",
        "response": "Generated answer.",
        "done": True,
    }


def test_generate_text_client_success(monkeypatch):
    import asyncio
    from app.ollama_client import generate_text

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "evidencopilot:latest",
                "response": "Generated answer.",
                "done": True,
            }

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            assert url.endswith("/api/generate")
            assert json["prompt"] == "Explain traceability."
            assert json["model"] == "evidencopilot:latest"
            assert json["stream"] is False
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)

    result = asyncio.run(generate_text("Explain traceability.", Settings()))

    assert result.model == "evidencopilot:latest"
    assert result.response == "Generated answer."
    assert result.done is True


def test_analyze_claim_logs_ai_verdict_and_explanation(monkeypatch, caplog):
    import asyncio
    import json
    import logging

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": json.dumps(
                    {
                        "verdict": "supported",
                        "confidence": 0.87,
                        "matched_source_ids": ["source-1"],
                        "missing_evidence": [],
                        "reasoning_summary": "I compared the claim terms against source-1 and found direct support.",
                        "explanation": "The source directly links traceability to review quality.",
                    }
                )
            }

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            assert json["think"] is False
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)
    caplog.set_level(logging.DEBUG, logger="app")

    result = asyncio.run(
        analyze_claim(
            ClaimAnalysisRequest.model_validate(sample_payload()),
            Settings(),
        )
    )

    assert result.verdict == "supported"
    messages = [record.getMessage() for record in caplog.records]
    assert any("ai verdict verdict=supported confidence=0.87" in message for message in messages)
    assert any(
        "ai reasoning summary I compared the claim terms against source-1 and found direct support." in message
        for message in messages
    )
    assert any(
        "ai explanation The source directly links traceability to review quality." in message
        for message in messages
    )


def test_analyze_claim_rejects_unknown_source_id(monkeypatch):
    import asyncio
    import httpx
    import json

    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "response": json.dumps(
                    {
                        "verdict": "supported",
                        "confidence": 0.9,
                        "matched_source_ids": ["invented-source"],
                        "missing_evidence": [],
                        "explanation": "The source supports the claim.",
                    }
                )
            },
        )
    )
    monkeypatch.setattr(
        "app.ollama_client.httpx.AsyncClient",
        lambda timeout: async_client(timeout=timeout, transport=transport),
    )

    with pytest.raises(OllamaInvalidResponseError):
        asyncio.run(
            analyze_claim(
                ClaimAnalysisRequest.model_validate(sample_payload()),
                Settings(),
            )
        )


def test_get_embeddings_success(client: TestClient):
    from app.models import EmbeddingResponse

    async def fake_run_generate_embeddings():
        return EmbeddingResponse(embedding=[0.1, -0.2, 0.35])

    from app.main import run_generate_embeddings
    app.dependency_overrides[run_generate_embeddings] = fake_run_generate_embeddings

    response = client.post("/ai/embeddings", json={"text": "hello world"})

    assert response.status_code == 200
    assert response.json() == {"embedding": [0.1, -0.2, 0.35]}


def test_get_embeddings_validation_error_empty_text(client: TestClient):
    response = client.post("/ai/embeddings", json={"text": "   "})
    assert response.status_code == 422


def test_get_embeddings_validation_error_missing_field(client: TestClient):
    response = client.post("/ai/embeddings", json={})
    assert response.status_code == 422


def test_generate_embeddings_client_success(monkeypatch):
    import asyncio
    from app.ollama_client import generate_embeddings

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": [0.01, -0.02, 0.03]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            assert url.endswith("/api/embeddings")
            assert json["prompt"] == "test text"
            assert json["model"] == "nomic-embed-text"
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)

    res = asyncio.run(generate_embeddings("test text", Settings()))
    assert res == [0.01, -0.02, 0.03]


def test_generate_embeddings_client_fallback_embeddings(monkeypatch):
    import asyncio
    from app.ollama_client import generate_embeddings

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.99, -0.88]]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)

    res = asyncio.run(generate_embeddings("test text", Settings()))
    assert res == [0.99, -0.88]


def test_generate_embeddings_client_unavailable(monkeypatch):
    import asyncio
    import httpx
    from app.ollama_client import generate_embeddings

    class FakeAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)

    with pytest.raises(OllamaUnavailableError):
        asyncio.run(generate_embeddings("test text", Settings()))


def test_generate_embeddings_client_invalid_response(monkeypatch):
    import asyncio
    from app.ollama_client import generate_embeddings

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"invalid_key": "not a list"}

    class FakeAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)

    with pytest.raises(OllamaInvalidResponseError):
        asyncio.run(generate_embeddings("test text", Settings()))


def test_generate_embeddings_client_rejects_empty_embedding(monkeypatch):
    import asyncio
    from app.ollama_client import generate_embeddings

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": []}

    class FakeAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr("app.ollama_client.httpx.AsyncClient", FakeAsyncClient)

    with pytest.raises(OllamaInvalidResponseError):
        asyncio.run(generate_embeddings("test text", Settings()))
