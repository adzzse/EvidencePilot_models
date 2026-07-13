import asyncio
import json
import logging
import re
import tempfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
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
class ExtractionBlock:
    type: str
    text: str
    level: int | None = None
    caption: str | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    markdown: str
    blocks: tuple[ExtractionBlock, ...]


async def extract_from_url(payload: ExtractRequest, settings: Settings) -> ExtractedDocument:
    host = (urlparse(str(payload.download_url)).hostname or "").lower()
    if settings.extraction_allowed_hosts and host not in settings.extraction_allowed_hosts:
        raise ExtractionError("download_url host is not allowed")

    filename = Path(payload.filename).name
    suffix = Path(filename).suffix.lower()
    if suffix != ".pdf":
        raise ExtractionError("only PDF files are supported")

    with tempfile.TemporaryDirectory(prefix="evidencepilot-extract-") as temp_dir:
        input_path = Path(temp_dir) / "input.pdf"
        await _download(str(payload.download_url), input_path, settings.max_download_bytes)
        result = await extract_with_mineru(
            input_path,
            Path(temp_dir) / "output",
            settings.mineru_timeout_seconds,
            settings.mineru_command,
            settings.mineru_backend,
        )

    return ExtractedDocument(_clean(result.markdown), tuple(result.blocks))


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
) -> ExtractedDocument:
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

    return _read_mineru_output(output_dir, pdf_path.stem)


def _read_mineru_output(output_dir: Path, document_stem: str) -> ExtractedDocument:
    markdown_path = next(output_dir.rglob(f"{document_stem}.md"), None)
    if markdown_path is None:
        markdown_path = next(output_dir.rglob("result.md"), None)
    if markdown_path is None:
        raise ExtractionUnavailableError("MinerU produced no Markdown output")

    content_list_path = markdown_path.with_name(f"{markdown_path.stem}_content_list.json")
    if not content_list_path.is_file():
        content_list_path = next(output_dir.rglob(f"{document_stem}_content_list.json"), None)
    if content_list_path is None or not content_list_path.is_file():
        raise ExtractionUnavailableError("MinerU produced no content list output")

    try:
        content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractionUnavailableError("MinerU content list is invalid") from exc
    if not isinstance(content_list, list):
        raise ExtractionUnavailableError("MinerU content list is invalid")

    blocks = tuple(_normalize_mineru_blocks(content_list))
    if not blocks:
        raise ExtractionUnavailableError("MinerU produced no content blocks")
    return ExtractedDocument(_clean(markdown_path.read_text(encoding="utf-8")), blocks)


def _normalize_mineru_blocks(content_list: list[dict[str, Any]]) -> list[ExtractionBlock]:
    blocks: list[ExtractionBlock] = []
    reference_level: int | None = None

    for item in content_list:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).lower()
        level = item.get("text_level")
        level = level if isinstance(level, int) and 1 <= level <= 6 else None
        text = _item_text(item)

        if item_type == "text" and level is not None:
            is_reference_heading = _is_reference_heading(text)
            if reference_level is not None and level <= reference_level and not is_reference_heading:
                reference_level = None
            if is_reference_heading:
                reference_level = level
            if text:
                blocks.append(ExtractionBlock(
                    "reference" if reference_level is not None else "heading",
                    text,
                    None if reference_level is not None else level,
                ))
            continue

        is_reference = reference_level is not None or item.get("sub_type") == "ref_text"
        if is_reference:
            if text:
                blocks.append(ExtractionBlock("reference", text))
            continue

        if item_type == "text" and text:
            blocks.append(ExtractionBlock("paragraph", text))
        elif item_type == "list" and text:
            blocks.append(ExtractionBlock("list", text))
        elif item_type == "table":
            table = _table_to_markdown(str(item.get("table_body", "")))
            if table:
                blocks.append(ExtractionBlock("table", table, caption=_caption(item, "table_caption")))
        elif item_type in {"image", "figure"}:
            caption = _caption(item, "image_caption")
            if caption:
                blocks.append(ExtractionBlock("figure_caption", caption))
        elif item_type in {"equation", "interline_equation"} and text:
            blocks.append(ExtractionBlock("equation", text))
        elif item_type == "code" and text:
            blocks.append(ExtractionBlock("code", text, caption=_caption(item, "code_caption")))

    return blocks


def _item_text(item: dict[str, Any]) -> str:
    value = item.get("text") or item.get("code_body") or item.get("equation_body")
    if value is None:
        value = item.get("list_items")
    if isinstance(value, list):
        parts = []
        for entry in value:
            raw = entry.get("text", "") if isinstance(entry, dict) else entry
            cleaned = _clean_block_text(str(raw))
            if cleaned:
                parts.append(f"- {cleaned}")
        return "\n".join(parts)
    return _clean_block_text(str(value or ""))


def _caption(item: dict[str, Any], key: str) -> str | None:
    value = item.get(key)
    if isinstance(value, list):
        value = " ".join(str(part) for part in value)
    cleaned = _clean_block_text(str(value or ""))
    return cleaned or None


def _clean_block_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return "\n".join(line.strip() for line in text.split("\n") if line.strip()).strip()


def _is_reference_heading(text: str) -> bool:
    normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    return normalized in {"references", "bibliography", "works cited"}


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[list[str], bool]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._has_header = False

    def handle_starttag(self, tag: str, _attrs) -> None:
        if tag == "tr":
            self._row = []
            self._has_header = False
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []
            self._has_header = self._has_header or tag == "th"

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._row is not None and self._cell is not None:
            self._row.append(_markdown_cell(" ".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row:
            self.rows.append((self._row, self._has_header))
            self._row = None


def _markdown_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().replace("|", "\\|")


def _table_to_markdown(html: str) -> str:
    parser = _TableParser()
    parser.feed(html)
    if not parser.rows:
        return ""

    rows = [row for row, _ in parser.rows]
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header_index = next((index for index, (_, header) in enumerate(parser.rows) if header), 0)
    header = rows.pop(header_index)
    rendered = [_markdown_row(header), _markdown_row(["---"] * width)]
    rendered.extend(_markdown_row(row) for row in rows)
    return "\n".join(rendered)


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _clean(markdown: str) -> str:
    cleaned = markdown.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        raise ExtractionError("no text could be extracted from the document")
    return cleaned
