# EvidencePilot Python Model Service

Stateless FastAPI service called by the Java backend. It has no RabbitMQ, MinIO,
Qdrant, or application database access.

- PDF extraction: MinerU (`mineru` CLI)
- DOCX extraction: `python-docx`, normalized to Markdown and structured blocks
- Markdown extraction: direct UTF-8 normalization to structured blocks
- Text generation: OpenAI-compatible API, Gemini API, or local Ollama
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

Pull the local generation and embedding models:

```powershell
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

Generation provider configuration:

```dotenv
GENERATION_PROVIDER=auto
OPENAI_COMPATIBLE_API_KEY=
OPENAI_COMPATIBLE_BASE_URL=https://opencode.ai/zen/v1
OPENAI_COMPATIBLE_MODEL=deepseek-v4-flash-free
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.6-flash
OLLAMA_MODEL=qwen3.5:9b
```

With `auto`, a non-empty `OPENAI_COMPATIBLE_API_KEY` selects the compatible
provider; otherwise generation uses local Ollama. Gemini is selected only with
`GENERATION_PROVIDER=gemini`, which requires `GEMINI_API_KEY`. Set the provider
to `ollama` to force local generation or `openai_compatible` to require the
compatible API key. Once a provider is selected, an upstream failure is
returned to the caller without failover. Embeddings and document extraction
always remain local. Generation context, including Claims, source chunks,
paper sections, and feedback, is sent to the selected remote provider.

OpenCode's free DeepSeek V4 Flash endpoint may retain and use submitted data
for model improvement. Do not send personal or confidential data through it.

Set `MODEL_API_KEY` to the same value as Java's `AI_MODEL_API_KEY`. Set
`EXTRACTION_ALLOWED_HOSTS` to the hostname used by Java's presigned MinIO URLs;
use a comma-separated list when more than one hostname is required. That MinIO
hostname must be reachable from this machine, so a Railway-private hostname is
not suitable for the presigned download URL.

`MODEL_API_KEY` authenticates Java requests to this worker. It is unrelated to
`OPENAI_COMPATIBLE_API_KEY` or Google's `GEMINI_API_KEY`.

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
{
  "system": "Return one JSON object describing evidence traceability.",
  "prompt": "{\"claim\":\"Evidence traceability links claims to sources.\"}"
}
```

`system` is optional for backward compatibility. The response identifies the
provider and actual model used:

```json
{
  "provider": "gemini",
  "model": "gemini-3.6-flash",
  "response": "{\"supported\":true}",
  "done": true
}
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
