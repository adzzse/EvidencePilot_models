import asyncio
import json
import logging
from types import SimpleNamespace
from uuid import uuid4
from zipfile import ZipFile

import pytest
from docx import Document as WordDocument
from fastapi.testclient import TestClient

import app.extraction as extraction
import app.main as main
from app.extraction import (
    ExtractionBlock as MinerUExtractionBlock,
    ExtractionError,
    ExtractionUnavailableError,
    extract_from_url,
)
from app.models import ExtractRequest, ExtractionBlock, GenerateResponse
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


def test_extract_serializes_mineru_blocks(monkeypatch):
    async def extract(payload, _):
        assert payload.filename == "paper.pdf"
        return SimpleNamespace(
            markdown="# Paper",
            blocks=(MinerUExtractionBlock("heading", "Paper", level=1),),
        )

    monkeypatch.setattr(main, "extract_from_url", extract)
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
    assert response.json() == {
        "markdown": "# Paper",
        "blocks": [{"type": "heading", "text": "Paper", "level": 1, "caption": None}],
    }


def test_extract_rejects_removed_request_fields(client: TestClient, monkeypatch):
    async def extract(*_):
        return SimpleNamespace(
            markdown="# Paper",
            blocks=[{"type": "heading", "text": "Paper", "level": 1}],
        )

    monkeypatch.setattr(main, "extract_from_url", extract)
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
        return SimpleNamespace(
            markdown="# Extracted",
            blocks=blocks,
            replace=lambda *args: "# Extracted".replace(*args),
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
    (output_dir / "input_content_list.json").write_text(
        json.dumps([
            {"type": "text", "text": "Results", "text_level": 2},
            {"type": "text", "text": "A result paragraph."},
            {
                "type": "table",
                "table_caption": ["Table 1"],
                "table_body": (
                    "<table><thead><tr><th>A</th><th>B</th></tr></thead>"
                    "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
                ),
            },
            {"type": "image", "image_caption": ["Figure 1. Architecture"]},
            {"type": "text", "text": "References", "text_level": 1},
            {"type": "list", "sub_type": "ref_text", "text": "Smith 2024"},
        ]),
        encoding="utf-8",
    )

    read_output = getattr(extraction, "_read_mineru_output", None)
    assert read_output is not None
    result = read_output(tmp_path, "input")

    assert result.markdown == "# Extracted"
    assert [block.type for block in result.blocks] == [
        "heading",
        "paragraph",
        "table",
        "figure_caption",
        "reference",
        "reference",
    ]
    assert result.blocks[2].caption == "Table 1"
    assert result.blocks[2].text == "| A | B |\n| --- | --- |\n| 1 | 2 |"


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
