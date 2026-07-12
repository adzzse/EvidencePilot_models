import asyncio
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.models import ExtractRequest
from app.settings import Settings


logger = logging.getLogger(__name__)


class ExtractionError(ValueError):
    pass


class ExtractionUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractedMarkdown:
    filename: str
    method: str
    markdown: str


async def extract_from_url(payload: ExtractRequest, settings: Settings) -> ExtractedMarkdown:
    host = (urlparse(str(payload.download_url)).hostname or "").lower()
    if not settings.extraction_allowed_hosts or host not in settings.extraction_allowed_hosts:
        raise ExtractionError("download_url host is not allowed")

    filename = Path(payload.filename).name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise ExtractionError("only PDF and DOCX files are supported")

    with tempfile.TemporaryDirectory(prefix="evidencepilot-extract-") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        await _download(str(payload.download_url), input_path, settings.max_download_bytes)

        if suffix == ".pdf":
            markdown = await extract_with_mineru(
                input_path,
                Path(temp_dir) / "output",
                settings.mineru_timeout_seconds,
                settings.mineru_command,
                settings.mineru_backend,
            )
            method = "mineru"
        else:
            markdown = await asyncio.to_thread(extract_with_liteparse, input_path)
            method = "liteparse"

    return ExtractedMarkdown(filename, method, _clean(markdown))


async def _download(url: str, destination: Path, max_bytes: int) -> None:
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=False) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                size = 0
                with destination.open("wb") as output:
                    async for block in response.aiter_bytes():
                        size += len(block)
                        if size > max_bytes:
                            raise ExtractionError("download exceeds the 52 MB limit")
                        output.write(block)
    except ExtractionError:
        raise
    except httpx.HTTPError as exc:
        raise ExtractionUnavailableError("could not download the source document") from exc


async def extract_with_mineru(
    pdf_path: Path,
    output_dir: Path,
    timeout_seconds: int,
    command: str,
    backend: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        process = await asyncio.create_subprocess_exec(
            command,
            "-p",
            str(pdf_path),
            "-o",
            str(output_dir),
            "--backend",
            backend,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise ExtractionUnavailableError(f"MinerU executable is not available: {command}") from exc

    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise ExtractionUnavailableError(f"MinerU timed out after {timeout_seconds}s") from exc

    if process.returncode != 0:
        detail = stderr.decode(errors="replace")[-2000:]
        raise ExtractionUnavailableError(f"MinerU failed: {detail}")

    return _read_mineru_markdown(output_dir, pdf_path.stem)


def _read_mineru_markdown(output_dir: Path, document_stem: str) -> str:
    markdown_path = next(output_dir.rglob(f"{document_stem}.md"), None)
    if markdown_path is None:
        markdown_path = next(output_dir.rglob("result.md"), None)
    if markdown_path is None:
        raise ExtractionUnavailableError("MinerU produced no Markdown output")
    return markdown_path.read_text(encoding="utf-8")


def extract_with_liteparse(document_path: Path) -> str:
    try:
        from liteparse import LiteParse

        result = LiteParse(output_format="markdown").parse(str(document_path))
        markdown = getattr(result, "text", None)
    except Exception as exc:
        raise ExtractionUnavailableError("LiteParse extraction failed") from exc
    if not isinstance(markdown, str):
        raise ExtractionUnavailableError("LiteParse produced no Markdown")
    return markdown


def _clean(markdown: str) -> str:
    cleaned = markdown.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        raise ExtractionError("no text could be extracted from the document")
    return cleaned
