# EvidencePilot Python Model Service

Stateless FastAPI service called by the Java backend. It has no RabbitMQ, MinIO,
Qdrant, or application database access.

- PDF extraction: MinerU (`mineru` CLI)
- DOCX extraction: `python-docx`, normalized to Markdown and structured blocks
- Markdown extraction: direct UTF-8 normalization to structured blocks
- Text generation: Ollama
- Single and batch embeddings: Ollama `nomic-embed-text`

Java remains responsible for upload state, queue consumption, Markdown/chunk
persistence, vector indexing, retries, and the final `READY` status.

## Setup

```powershell
cd E:\Code\SEP490\EvidencePilot_models
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Install MinerU separately, then set `MINERU_COMMAND` to its executable. For a
separate Windows virtual environment, for example:

```dotenv
MINERU_COMMAND=.venv-mineru\Scripts\mineru.exe
MINERU_BACKEND=pipeline
```

Create the generation model once:

```powershell
ollama create evidencopilot -f Modelfile
```

Set `MODEL_API_KEY` to the same value as Java's `AI_MODEL_API_KEY`. Set
`EXTRACTION_ALLOWED_HOSTS` to the hostname used by Java's presigned MinIO URLs;
use a comma-separated list when more than one hostname is required. That MinIO
hostname must be reachable from this machine, so a Railway-private hostname is
not suitable for the presigned download URL.

## Run

```powershell
cd E:\Code\SEP490\EvidencePilot_models
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`GET /health` is public. Every `POST` route requires `X-API-Key`.

## API contract

### Extract a document

`POST /extract`

```json
{
  "filename": "paper.pdf",
  "download_url": "https://storage.example.com/presigned-object"
}
```

The service downloads only an allowlisted PDF, DOCX, or Markdown file and
returns Markdown with normalized content blocks without persisting them:

```json
{
  "markdown": "# Extracted document",
  "blocks": [
    {"type": "heading", "text": "Extracted document", "level": 1},
    {"type": "paragraph", "text": "First paragraph."}
  ]
}
```

Supported suffixes are `.pdf`, `.docx`, `.md`, and `.markdown`; `.tex` is
unsupported. Blocks use the types `heading`, `paragraph`, `list`, `table`,
`figure_caption`, `equation`, `code`, and `reference`.

### Generate text

`POST /ai/generate`

```json
{"prompt": "Explain evidence traceability."}
```

### Embed one text

`POST /ai/embeddings`

```json
{"text": "Evidence traceability links claims to sources."}
```

### Embed a batch

`POST /ai/embeddings/batch`

```json
{"texts": ["First chunk", "Second chunk"]}
```

The batch endpoint accepts 1-64 texts and preserves input order.

## Ngrok

Expose the local service when Java runs remotely on Railway:

```powershell
python scripts\start_ngrok_tunnel.py
```

Configure Railway's `AI_MODEL_BASE_URL` with the HTTPS tunnel URL and use the
same API key on both services.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
