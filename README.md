# EvidencePilot Python Model Service

Stateless FastAPI service called by the Java backend. It has no RabbitMQ, MinIO,
Qdrant, or application database access.

- PDF extraction: MinerU (`mineru` CLI), with configured-provider hierarchy repair for flat headings
- DOCX extraction: `python-docx`, normalized to Markdown and structured blocks
- Markdown extraction: direct UTF-8 normalization to structured blocks
- Text generation: any OpenAI-compatible API or local Ollama
- Single and batch embeddings: Ollama `nomic-embed-text`

Java remains responsible for upload state, queue consumption, Markdown/chunk
persistence, vector indexing, retries, and the final `READY` status.

## Setup

```powershell
cd E:\Code\SEP490\EvidencePilot_models
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements-dev.txt
Copy-Item .env.example .env
```

Install MinerU separately in another Python 3.13 virtual environment, then set
`MINERU_COMMAND` to its executable. For example:

```dotenv
MINERU_COMMAND=.venv-mineru\Scripts\mineru.exe
MINERU_BACKEND=pipeline
```

Pull the local generation and embedding models:

```powershell
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

Generation can use local Ollama or one remote OpenAI-compatible endpoint:

```dotenv
GENERATION_PROVIDER=auto
GENERATION_API_KEY=
GENERATION_BASE_URL=https://openrouter.ai/api/v1
GENERATION_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
GENERATION_EXTRA_BODY={"reasoning":{"effort":"none"}}
OLLAMA_MODEL=qwen3.5:9b
```

With `auto`, a non-empty `GENERATION_API_KEY` selects `remote`; otherwise
generation uses local Ollama. Use `GENERATION_PROVIDER=remote` or `ollama` to
force either path. Switching services only changes the generic remote values:

| Service | `GENERATION_BASE_URL` | Example model | `GENERATION_EXTRA_BODY` |
| --- | --- | --- | --- |
| OpenRouter | `https://openrouter.ai/api/v1` | `nvidia/nemotron-3-ultra-550b-a55b:free` | `{"reasoning":{"effort":"none"}}` |
| OpenCode Zen | `https://opencode.ai/zen/v1` | `deepseek-v4-flash-free` | `{"thinking":{"type":"disabled"}}` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-3.6-flash` | `{"reasoning_effort":"minimal"}` |

`GENERATION_EXTRA_BODY` defaults to `{}` and holds provider-specific compatible
options. Gemini 3 models support reducing thinking to `minimal`, but do not
support fully disabling it; Gemini 2.5 models that allow disabling thinking can
use `{"reasoning_effort":"none"}`.

Every remote request uses `temperature=0`, JSON-object output, and non-streaming
mode. Once selected, an upstream failure is returned without failover.
Embeddings and document extraction always remain local. Generation context,
including Claims, source chunks, paper sections, and feedback, is sent to the
selected remote service. Review that service's data policy and do not send
personal or confidential data through a free endpoint.

Set `MODEL_API_KEY` to the same value as Java's `AI_MODEL_API_KEY`. Set
`EXTRACTION_ALLOWED_HOSTS` to the hostname used by Java's presigned MinIO URLs;
use a comma-separated list when more than one hostname is required. That MinIO
hostname must be reachable from this machine, so a Railway-private hostname is
not suitable for the presigned download URL.

`MODEL_API_KEY` authenticates Java requests to this worker. It is unrelated to
`GENERATION_API_KEY`.

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
  "provider": "remote",
  "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
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
