# EvidencePilot Python Model Service

Stateless FastAPI service called by the Java backend. It has no RabbitMQ, MinIO,
Qdrant, or application database access.

- PDF extraction: MinerU (`mineru` CLI); Paper requests retain configured-provider hierarchy repair, Source requests disable it
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

Configure OpenRouter model choices:

```dotenv
GENERATION_PROVIDER=remote
GENERATION_API_KEY=
GENERATION_BASE_URL=https://openrouter.ai/api/v1
GENERATION_MODEL=
GENERATION_FALLBACK_MODELS=[]
GENERATION_EXTRA_BODY={}
```

With both model fields empty, the Python service fetches each configured key's
OpenRouter `/models/user` catalog and offers their common, compatible free
text models in the existing Admin model selector. Admins select one primary and
up to two fallbacks; the service does not choose a new saved selection silently.
It requires zero-price `:free` variants with `response_format` support and
enough context/output tokens for this service's request cap. The catalog is
cached for five minutes; a transient fetch failure can reuse a successful
snapshot for up to one hour. An absent selected model requires the admin to
reload and choose again. Catalog metadata does not prove current inference
availability or remaining quota. Preview variants remain selectable but are
not placed in the default chain when stable variants exist. For a non-OpenRouter gateway, set
`GENERATION_MODEL` and optionally `GENERATION_FALLBACK_MODELS` explicitly.

For optional OpenRouter key failover, leave `GENERATION_API_KEY` empty and set
`GENERATION_API_KEYS` to a JSON array of one to three distinct keys in local
secret configuration. The two settings cannot be used together. The service
tries every selected model on the first key before repeating the model chain
on the next key. A timeout, transport error, or 429 skips to the next model on
the same key, including daily free-model quota errors. Remote models enter a
process-local cooldown only when a retryable provider error includes a valid,
positive `Retry-After` (seconds or HTTP date). Timeout, transport errors, and
responses without that header do not create a cooldown. Each timer belongs to
the provider endpoint, API key identity, and exact outbound model name; another
key may still try the same model, and other models on the same key remain eligible.
Timers survive between requests and key/model reordering, but not service restarts.
A cooling pair is skipped before queueing and checked again before sending;
after expiry it can be tried again. No task sleeps while waiting for a timer.
Each call has a 60-second timeout within the remaining batch budget.
Terminal errors and exhausted batch budgets stop the chain. Each new request
starts with the first configured key. [OpenRouter says](https://openrouter.ai/docs/api_reference/limits)
extra accounts or keys do not guarantee more capacity, and switching keys can
reduce [response-cache hits](https://openrouter.ai/docs/guides/features/response-caching).

Remote is the default and missing credentials fail explicitly. Local LLM
generation is disabled in this setup; the legacy `ollama` and `auto` modes remain
available only through explicit configuration. They are never remote fallbacks.
In fixed-model mode, an omitted `GENERATION_FALLBACK_MODELS` means primary only. Do not put `models`
or other managed request parameters in `GENERATION_EXTRA_BODY`.

Each call uses one model, `temperature=0`, `max_tokens=8192`, and non-streaming
JSON output. Catalog models advertising `structured_outputs` receive native
JSON Schema mode when requested; models supporting only `response_format` receive
JSON object mode, and models supporting neither receive formatting instructions
without a native response format. The requested schema remains in the system
instruction. Python validates complete output, JSON and the supplied schema before
returning success. Invalid output gets one regeneration on the same model, then
the next model; transport failures go directly to the next model. Refusals,
request/authentication errors stop the call. Model/key pairs on cooldown are skipped
without waiting for their `Retry-After`, including temporary 503 responses.
If every remaining model/key pair is on cooldown, the service returns the failure
for the pair that becomes available first, with its remaining `Retry-After`,
without making another provider request. If any remaining pair has no active
timer, the final error does not advertise a chain-wide `Retry-After` delay.

The batch budget is at most 300 seconds including queueing, pacing and all
attempts (at most two per key/model pair, 18 total with three keys and three models);
each remote HTTP attempt is capped
at 60 seconds. Paper PDF heading repair has a separate 30-second budget and keeps
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
`GENERATION_API_KEY` and `GENERATION_API_KEYS`.

Run one Python worker: `MODEL_MAX_CONCURRENT_REQUESTS` caps each of two independent
process-local pools, remote generation and local extraction/embedding.
`MODEL_MIN_INTERVAL_MS` spaces every remote attempt, including fallback and
Paper heading repair. Local calls have no remote pacing delay.

Before activating the full chain for Java traffic, update Java to consume the
continuation fields below and remove its duplicate generation retries. The
300-second/18-attempt bound applies to one Python request until Java shares the
same deadline across semantic validation attempts. Existing `.env` files are not
changed automatically by a code update.

## Run

```powershell
cd E:\Code\SEP490\EvidencePilot_models
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Python logs and Uvicorn access/error logs are also saved to `logs/model-service.log`,
including timestamps and concise error summaries. Errors show their type and
generation failures identify the reason, model, key slot, elapsed time and next
action; handled HTTP errors include the response status and error code. Console
and file output use the same redacted format without stack traces. The file rotates
at 10 MiB and retains five backups. Configured API keys and URL query strings are
redacted in both outputs; request prompts and model responses are not added to logs. This applies
to both direct Uvicorn runs and `scripts/start_ngrok_tunnel.py`; `logs/` is Git-ignored.

`GET /health` is public. Every `POST` route requires `X-API-Key`.
Remote health checks catalog coverage for the configured or discovered models; it does not prove
inference availability or remaining quota (`inference_verified=false`). Local
generation availability is not required or advertised in remote mode.

## API contract

### Extract a document

`POST /extract`

```json
{
  "filename": "source.pdf",
  "download_url": "https://storage.example.com/presigned-object",
  "enrich_hierarchy": false
}
```

Java sends `enrich_hierarchy=false` for Source documents and `true` for Papers.
Source extraction returns MinerU headings without invoking a generation provider;
Paper extraction retains hierarchy repair for flat PDF headings. Omitted values
default to `true` for compatibility with older callers. DOCX and Markdown do not
use generation regardless of this flag. Source and Paper PDF bundles use separate
cache entries in Java; existing per-document checkpoints remain reusable.
Roll out the Python service first, then the Java backend: older Python versions
reject the new request field, while this version still accepts older Java requests.

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
  "model": "nex-agi/nex-n2.5-pro:free",
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

The reserved tunnel endpoint is `https://scoff-difficult-said.ngrok-free.dev`.
The launcher pins ngrok to this domain; configure Railway's
`AI_MODEL_BASE_URL` with that exact origin and use the same API key on both
services.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
