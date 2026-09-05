# EvidencePilot Python Model Service

Stateless FastAPI service called by the Java backend. It has no RabbitMQ, MinIO,
Qdrant, or application database access.

- PDF extraction: MinerU (`mineru` CLI), with configured-provider hierarchy repair for flat headings
- DOCX extraction: `python-docx`, normalized to Markdown and structured blocks
- Markdown extraction: direct UTF-8 normalization to structured blocks
- Text generation: OpenRouter, with ordered remote model fallback
- Single and batch embeddings: Ollama `nomic-embed-text`

Java remains responsible for upload state, queue consumption, Markdown/chunk
persistence, vector indexing, business validation, job retries, and the final
`READY` status. Python owns generation transport retries and model fallback.

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

Pull the local embedding model:

```powershell
ollama pull nomic-embed-text
```

Configure the OpenRouter generation chain:

```dotenv
GENERATION_PROVIDER=remote
GENERATION_API_KEY=
GENERATION_BASE_URL=https://openrouter.ai/api/v1
GENERATION_MODEL=minimax/minimax-m3:free
GENERATION_FALLBACK_MODELS=["google/gemma-4-31b-it:free","nvidia/nemotron-3-super-120b-a12b:free"]
GENERATION_EXTRA_BODY={}
```

Remote is the default and missing credentials fail explicitly. Local LLM
generation is disabled in this setup; the legacy `ollama` and `auto` modes remain
available only through explicit configuration. They are never remote fallbacks.
An omitted `GENERATION_FALLBACK_MODELS` means primary only. Do not put `models`
or other managed request parameters in `GENERATION_EXTRA_BODY`.

Each call uses one model, `temperature=0`, `max_tokens=8192`, and non-streaming
JSON output. MiniMax and Gemma receive JSON object mode plus the requested schema
in the system instruction. Nemotron Super receives native JSON Schema mode when
requested. Python validates complete output, JSON and the supplied schema before
returning success. Invalid output gets one regeneration on the same model, then
the next model; transport failures go directly to the next model. Refusals,
request/authentication errors, and shared or ambiguous quota limits stop the call.
Explicit upstream rate limits can fall through. `Retry-After` on upstream 429
and temporary 503 responses is honored within the batch budget.

The batch budget is at most 300 seconds including queueing, pacing and all
attempts (at most two per model, six total); each remote HTTP attempt is capped
at 60 seconds. Flat PDF heading repair has a separate 30-second budget and keeps
the MinerU result on failure.
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

Run one Python worker: `MODEL_MAX_CONCURRENT_REQUESTS` caps each of two independent
process-local pools, remote generation and local extraction/embedding.
`MODEL_MIN_INTERVAL_MS` spaces every remote attempt, including fallback and
heading repair. Local calls have no remote pacing delay.

Before activating the full chain for Java traffic, update Java to consume the
continuation fields below and remove its duplicate generation retries. The
300-second/six-attempt bound applies to one Python request until Java shares the
same deadline across semantic validation attempts. Existing `.env` files are not
changed automatically by a code update.

## Run

```powershell
cd E:\Code\SEP490\EvidencePilot_models
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`GET /health` is public. Every `POST` route requires `X-API-Key`.
Remote health checks catalog coverage for the configured chain; it does not prove
inference availability or remaining quota (`inference_verified=false`). Local
generation availability is not required or advertised in remote mode.

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
returns a ZIP containing `document.md`, `extraction.json`, and any referenced
`images/` files. The manifest in `extraction.json` contains normalized blocks:

```json
{
  "blocks": [
    {"type": "heading", "text": "Extracted document", "level": 1},
    {"type": "paragraph", "text": "First paragraph."}
  ],
  "images": []
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
  "model": "minimax/minimax-m3:free",
  "response": "{\"supported\":true}",
  "done": true,
  "model_index": 0,
  "attempt": 1,
  "next_model_index": 1
}
```

Optional request fields are `response_format` (`json_object` or `json_schema`),
`model_index` (zero-based; default 0), `attempt` (1 or 2; default 1), `budget_ms`
(1–300000; default 300000), and `validation_feedback` (at most 2000 characters).
Schemas may use inline/local references; remote and filesystem references are
disabled. Without a schema, the result must be a JSON object.

When Java business validation rejects a result from attempt 1, continue at the
returned `model_index` with attempt 2. After attempt 2, use `next_model_index`
and attempt 1; null means exhausted. Always subtract elapsed time from the same
logical batch deadline. `done=true` means technical validation passed; Java must
still validate IDs, exact source quotes, permissions, and persistence rules.
Errors retain `detail` and add a stable `code`; final Python errors must not
restart the chain in Java. The API specification is also available at `/docs`.

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
