import asyncio
import io
import json
import logging
import zipfile
from types import SimpleNamespace
from uuid import uuid4
from zipfile import ZipFile

import pytest
from docx import Document as WordDocument
from fastapi.testclient import TestClient

import app.extraction as extraction
import app.main as main
from app.extraction import (
    ExtractedDocument,
    ExtractionError,
    ExtractionUnavailableError,
    ExtractionWorkProduct,
    extract_from_url,
)
from app.models import ExtractRequest, ExtractionBlock, GenerateResponse
from app.generation import (
    GeminiGenerationProvider,
    GenerationConfigurationError,
    GenerationInvalidResponseError,
    GenerationRateLimitError,
    GenerationUnavailableError,
    OllamaGenerationProvider,
)
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

    monkeypatch.setattr(OllamaGenerationProvider, "health", unavailable)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_health_with_gemini_requires_only_local_embedding(
    client: TestClient, monkeypatch
):
    required_generation = []
    settings = SETTINGS.model_copy(update={
        "generation_provider": "gemini",
        "gemini_api_key": "secret",
    })
    main.app.dependency_overrides[main.get_settings] = lambda: settings

    async def local_health(_, require_generation=True):
        required_generation.append(require_generation)
        return {
            "ok": True,
            "model_available": False,
            "embedding_model_available": True,
        }

    async def gemini_health(_):
        return {
            "ok": True,
            "provider": "gemini",
            "model": settings.gemini_model,
        }

    monkeypatch.setattr(main, "check_ollama", local_health)
    monkeypatch.setattr(GeminiGenerationProvider, "health", gemini_health)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["generation_provider"] == "gemini"
    assert response.json()["generation"] == {
        "ok": True,
        "provider": "gemini",
        "model": settings.gemini_model,
    }
    assert required_generation == [False]


def test_post_endpoints_require_api_key(client: TestClient):
    assert client.post("/ai/embeddings", json={"text": "claim"}).status_code == 401


def test_generate_accepts_system_and_returns_provider_metadata(
    client: TestClient, monkeypatch
):
    async def generate(system, prompt, settings):
        assert system == "Judge claim quality"
        assert prompt == "Review this"
        assert settings.ollama_model == SETTINGS.ollama_model
        return GenerateResponse(
            provider="ollama",
            model=settings.ollama_model,
            response="done",
            done=True,
        )

    monkeypatch.setattr(main, "generate_text", generate)
    response = client.post(
        "/ai/generate",
        headers=HEADERS,
        json={"system": "Judge claim quality", "prompt": "Review this"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "provider": "ollama",
        "model": SETTINGS.ollama_model,
        "response": "done",
        "done": True,
    }


def test_generate_keeps_prompt_only_request_compatible(client: TestClient, monkeypatch):
    async def generate(system, prompt, settings):
        assert system == ""
        return GenerateResponse(
            provider="ollama",
            model=settings.ollama_model,
            response=prompt,
            done=True,
        )

    monkeypatch.setattr(main, "generate_text", generate)

    response = client.post(
        "/ai/generate",
        headers=HEADERS,
        json={"prompt": "Review this"},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Review this"


def test_generate_accepts_whole_paper_prompt(client: TestClient, monkeypatch):
    async def generate(_system, prompt, settings):
        return GenerateResponse(
            provider="ollama",
            model=settings.ollama_model,
            response=str(len(prompt)),
            done=True,
        )

    monkeypatch.setattr(main, "generate_text", generate)

    accepted = client.post(
        "/ai/generate",
        headers=HEADERS,
        json={"prompt": "x" * 48000},
    )
    rejected = client.post(
        "/ai/generate",
        headers=HEADERS,
        json={"prompt": "x" * 48001},
    )

    assert accepted.status_code == 200
    assert accepted.json()["response"] == "48000"
    assert rejected.status_code == 422


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (GenerationConfigurationError("bad config"), 503),
        (GenerationUnavailableError("offline"), 503),
        (GenerationRateLimitError("rate limited"), 429),
        (GenerationInvalidResponseError("malformed"), 502),
    ],
)
def test_generation_errors_map_to_gateway_status(failure, expected_status, monkeypatch):
    async def generate(*_):
        raise failure

    monkeypatch.setattr(main, "generate_text", generate)
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/ai/generate",
            headers=HEADERS,
            json={"prompt": "Review this"},
        )

    assert response.status_code == expected_status


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


def test_extract_returns_bundle(monkeypatch):
    async def create_bundle(payload, _, destination):
        assert payload.filename == "paper.pdf"
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr(
                "extraction.json",
                json.dumps({
                    "blocks": [
                        {"type": "heading", "text": "Paper", "level": 1, "caption": None}
                    ],
                    "images": ["images/figure.jpg"],
                }),
            )
            archive.writestr("document.md", "# Paper\n\n![](images/figure.jpg)")
            archive.writestr("images/figure.jpg", b"jpeg")

    monkeypatch.setattr(main, "create_extraction_bundle", create_bundle)
    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/extract",
            headers=HEADERS,
            json={
                "filename": "paper.pdf",
                "download_url": "https://storage.test/paper.pdf?signature=test",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == [
            "extraction.json",
            "document.md",
            "images/figure.jpg",
        ]
        assert archive.read("document.md") == b"# Paper\n\n![](images/figure.jpg)"
        assert archive.read("images/figure.jpg") == b"jpeg"


def test_extract_rejects_removed_request_fields(client: TestClient):
    response = client.post(
        "/extract",
        headers=HEADERS,
        json={
            "document_id": str(uuid4()),
            "filename": "paper.pdf",
            "content_type": "application/pdf",
            "download_url": "https://storage.test/paper.pdf",
        },
    )

    assert response.status_code == 422


def test_extract_rejects_untrusted_download_host(client: TestClient):
    response = client.post(
        "/extract",
        headers=HEADERS,
        json={
            "filename": "paper.pdf",
            "download_url": "https://evil.test/paper.pdf",
        },
    )
    assert response.status_code == 422


def test_extraction_routes_pdf_to_configured_mineru(monkeypatch):
    received = {}
    blocks = (SimpleNamespace(type="paragraph", text="Extracted", level=None, caption=None),)

    async def download(*_):
        return None

    async def mineru(*args):
        received["args"] = args
        return ExtractionWorkProduct(
            ExtractedDocument("# Extracted", blocks),
        )

    monkeypatch.setattr("app.extraction._download", download)
    monkeypatch.setattr("app.extraction.extract_with_mineru", mineru)
    payload = ExtractRequest(
        filename="paper.pdf",
        download_url="https://storage.test/paper.pdf",
    )

    settings = SETTINGS.model_copy(update={
        "mineru_command": "C:/tools/mineru.exe",
        "mineru_backend": "pipeline",
    })
    result = asyncio.run(extract_from_url(payload, settings))
    assert result.markdown == "# Extracted"
    assert result.blocks == blocks
    assert received["args"][-2:] == ("C:/tools/mineru.exe", "pipeline")


def test_mineru_logs_output_before_stream_ends(caplog):
    async def stream() -> bytes:
        reader = asyncio.StreamReader()
        task = asyncio.create_task(extraction._stream_mineru_output(reader))
        reader.feed_data(b"Processing page 1\n")
        await asyncio.sleep(0)
        assert "MinerU: Processing page 1" in caplog.text
        reader.feed_eof()
        return await task

    with caplog.at_level(logging.INFO, logger=extraction.__name__):
        assert asyncio.run(stream()) == b"Processing page 1\n"


def test_extraction_routes_markdown_to_structured_blocks(monkeypatch):
    async def download(_, destination, __):
        destination.write_text(
            "# Methods\n\n"
            "A result paragraph.\n\n"
            "| Metric | Value |\n"
            "| --- | --- |\n"
            "| Recall | 0.91 |\n\n"
            "# References\n\n"
            "Smith 2024",
            encoding="utf-8",
        )

    monkeypatch.setattr("app.extraction._download", download)
    payload = ExtractRequest(
        filename="source.markdown",
        download_url="https://storage.test/source.markdown",
    )

    result = asyncio.run(extract_from_url(payload, SETTINGS))

    assert result.markdown.startswith("# Methods")
    assert [block.type for block in result.blocks] == [
        "heading",
        "paragraph",
        "table",
        "reference",
        "reference",
    ]
    assert result.blocks[2].text.endswith("| Recall | 0.91 |")


def test_extraction_rejects_non_utf8_markdown(monkeypatch):
    async def download(_, destination, __):
        destination.write_bytes(b"\xff\xfe\x00")

    monkeypatch.setattr("app.extraction._download", download)
    payload = ExtractRequest(
        filename="source.md",
        download_url="https://storage.test/source.md",
    )

    with pytest.raises(ExtractionError, match="UTF-8"):
        asyncio.run(extract_from_url(payload, SETTINGS))


def test_markdown_parser_preserves_setext_lists_and_fenced_code():
    result = extraction._document_from_markdown(
        "Results\n"
        "=======\n\n"
        "- first\n"
        "- second\n\n"
        "```python\n"
        "print('ok')\n"
        "```"
    )

    assert [block.type for block in result.blocks] == ["heading", "list", "code"]
    assert result.blocks[0].level == 1
    assert result.blocks[1].text == "- first\n- second"
    assert result.blocks[2].text == "print('ok')"


def test_extraction_routes_docx_to_structured_blocks(monkeypatch):
    async def download(_, destination, __):
        document = WordDocument()
        document.add_heading("Methods", level=1)
        document.add_paragraph("A result paragraph.")
        document.add_paragraph("First item", style="List Bullet")
        table = document.add_table(rows=2, cols=2)
        table.rows[0].cells[0].text = "Metric"
        table.rows[0].cells[1].text = "Value"
        table.rows[1].cells[0].text = "Recall"
        table.rows[1].cells[1].text = "0.91"
        document.add_heading("References", level=1)
        document.add_paragraph("Smith 2024")
        document.save(destination)

    monkeypatch.setattr("app.extraction._download", download)
    payload = ExtractRequest(
        filename="source.docx",
        download_url="https://storage.test/source.docx",
    )

    result = asyncio.run(extract_from_url(payload, SETTINGS))

    assert result.markdown.startswith("# Methods")
    assert [block.type for block in result.blocks] == [
        "heading",
        "paragraph",
        "list",
        "table",
        "reference",
        "reference",
    ]
    assert "| Recall | 0.91 |" in result.blocks[3].text


def test_extraction_rejects_invalid_docx(monkeypatch):
    async def download(_, destination, __):
        destination.write_bytes(b"not-a-docx")

    monkeypatch.setattr("app.extraction._download", download)
    payload = ExtractRequest(
        filename="source.docx",
        download_url="https://storage.test/source.docx",
    )

    with pytest.raises(ExtractionError, match="DOCX file is invalid"):
        asyncio.run(extract_from_url(payload, SETTINGS))


def test_extraction_rejects_docx_expanding_beyond_limit(monkeypatch):
    async def download(_, destination, __):
        with ZipFile(destination, "w") as archive:
            archive.writestr("[Content_Types].xml", b"x" * 101)

    monkeypatch.setattr("app.extraction._download", download)
    payload = ExtractRequest(
        filename="source.docx",
        download_url="https://storage.test/source.docx",
    )
    settings = SETTINGS.model_copy(update={"max_download_bytes": 100})

    with pytest.raises(ExtractionError, match="DOCX content exceeds"):
        asyncio.run(extract_from_url(payload, settings))


def test_extract_rejects_tex_with_422(client: TestClient):
    response = client.post(
        "/extract",
        headers=HEADERS,
        json={
            "filename": "source.tex",
            "download_url": "https://storage.test/source.tex",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "only PDF, DOCX, and Markdown files are supported"
    }


def test_mineru_reads_markdown_and_normalizes_blocks(tmp_path):
    output_dir = tmp_path / "input" / "hybrid_auto"
    markdown_path = output_dir / "input.md"
    markdown_path.parent.mkdir(parents=True)
    markdown_path.write_text("# Extracted", encoding="utf-8")
    (output_dir / "images").mkdir()
    (output_dir / "images" / "architecture.png").write_bytes(b"png")
    (output_dir / "input_content_list.json").write_text(
        json.dumps([
            {
                "type": "text", "text": "Results", "text_level": 2,
                "page_idx": 0, "page_size": [1000, 1400],
                "bbox": [100, 120, 300, 150],
            },
            {
                "type": "text", "text": "A result paragraph.",
                "page_idx": 0, "bbox": [100, 180, 480, 240],
            },
            {
                "type": "table",
                "table_caption": ["Table 1"],
                "page_idx": 0,
                "bbox": [100, 260, 900, 500],
                "table_body": (
                    "<table><thead><tr><th>A</th><th>B</th></tr></thead>"
                    "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
                ),
            },
            {
                "type": "chart", "image_caption": ["Figure 1. Architecture"],
                "img_path": "images/architecture.png", "page_idx": 0,
                "bbox": [520, 520, 900, 800],
            },
            {"type": "header", "text": "Running title", "page_idx": 0, "bbox": [100, 30, 300, 50]},
            {"type": "footer", "text": "Journal 2026", "page_idx": 0, "bbox": [100, 1350, 300, 1370]},
            {"type": "page_number", "text": "192", "page_idx": 0, "bbox": [900, 30, 930, 50]},
            {"type": "page_footnote", "text": "Funding note", "page_idx": 0, "bbox": [100, 1280, 400, 1320]},
            {"type": "text", "text": "References", "text_level": 1},
            {"type": "list", "sub_type": "ref_text", "text": "Smith 2024"},
        ]),
        encoding="utf-8",
    )

    read_output = getattr(extraction, "_read_mineru_output", None)
    assert read_output is not None
    result = read_output(tmp_path, "input")

    assert result.document.markdown == "# Extracted"
    assert [block.type for block in result.document.blocks] == [
        "heading",
        "paragraph",
        "table",
        "figure",
        "header",
        "footer",
        "page_number",
        "page_footnote",
        "reference",
        "reference",
    ]
    assert result.document.blocks[2].caption == "Table 1"
    assert result.document.blocks[2].text == "| A | B |\n| --- | --- |\n| 1 | 2 |"
    assert result.document.blocks[0].source_type == "text"
    assert result.document.blocks[0].page_index == 0
    assert result.document.blocks[0].bbox == (100.0, 120.0, 300.0, 150.0)
    assert result.document.blocks[3].asset_path == "images/architecture.png"
    assert result.document.pages[0].model_dump() == {
        "page_index": 0,
        "width": 1000.0,
        "height": 1400.0,
    }


def test_mineru_bundle_includes_referenced_image(tmp_path):
    output_dir = tmp_path / "output"
    document_dir = output_dir / "paper"
    image_path = document_dir / "images" / "figure.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"jpeg")
    (document_dir / "paper.md").write_text(
        "# Result\n\n![](images/figure.jpg)",
        encoding="utf-8",
    )
    (document_dir / "paper_content_list.json").write_text(
        json.dumps([
            {"type": "text", "text": "Result", "text_level": 1},
            {"type": "image", "img_path": "images/figure.jpg"},
        ]),
        encoding="utf-8",
    )

    product = extraction._read_mineru_output(output_dir, "paper")

    assert product.document.images == ("images/figure.jpg",)
    assert product.image_files == (("images/figure.jpg", image_path),)


def test_mineru_rejects_image_path_escape(tmp_path):
    output_dir = tmp_path / "output"
    document_dir = output_dir / "paper"
    document_dir.mkdir(parents=True)
    (document_dir / "paper.md").write_text("# Result", encoding="utf-8")
    (document_dir / "paper_content_list.json").write_text(
        json.dumps([
            {"type": "text", "text": "Result", "text_level": 1},
            {"type": "image", "img_path": "../secret.jpg"},
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ExtractionUnavailableError, match="image path"):
        extraction._read_mineru_output(output_dir, "paper")


def test_mineru_requires_content_list(tmp_path):
    output_dir = tmp_path / "input" / "hybrid_auto"
    output_dir.mkdir(parents=True)
    (output_dir / "input.md").write_text("# Extracted", encoding="utf-8")

    read_output = getattr(extraction, "_read_mineru_output", None)
    assert read_output is not None
    with pytest.raises(ExtractionUnavailableError, match="content list"):
        read_output(tmp_path, "input")


def test_extraction_block_level_is_only_valid_for_headings():
    with pytest.raises(ValueError):
        ExtractionBlock(type="heading", text="Methods")
    with pytest.raises(ValueError):
        ExtractionBlock(type="paragraph", text="Body", level=2)


def test_settings_reads_mineru_command(monkeypatch):
    monkeypatch.setenv("MINERU_COMMAND", "C:/tools/mineru.exe")
    monkeypatch.setenv("MINERU_BACKEND", "pipeline")

    settings = Settings.from_env()

    assert getattr(settings, "mineru_command", None) == "C:/tools/mineru.exe"
    assert settings.mineru_backend == "pipeline"


def test_settings_reads_generation_provider_configuration(monkeypatch):
    monkeypatch.setenv("GENERATION_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "zen-secret")
    monkeypatch.setenv(
        "OPENAI_COMPATIBLE_BASE_URL",
        "https://gateway.test/v1/",
    )
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "deepseek-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:9b")

    settings = Settings.from_env()

    assert settings.generation_provider == "openai_compatible"
    assert settings.openai_compatible_api_key == "zen-secret"
    assert settings.openai_compatible_base_url == "https://gateway.test/v1"
    assert settings.openai_compatible_model == "deepseek-test"
    assert settings.gemini_api_key == "gemini-secret"
    assert settings.gemini_model == "gemini-3.6-flash"
    assert settings.ollama_model == "qwen3.5:9b"
