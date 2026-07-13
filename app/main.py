import logging
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from app.extraction import ExtractionError, ExtractionUnavailableError, extract_from_url
from app.models import (
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ExtractRequest,
    ExtractResponse,
    GenerateRequest,
    GenerateResponse,
)
from app.ollama_client import (
    OllamaInvalidResponseError,
    OllamaUnavailableError,
    check_ollama,
    generate_embeddings,
    generate_text,
)
from app.settings import Settings, load_settings


logging.basicConfig(level=logging.INFO)
app = FastAPI(title="EvidencePilot AI Worker", version="1.0.0")


def get_settings() -> Settings:
    return load_settings()


def require_api_key(
    settings: Settings = Depends(get_settings),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if not settings.model_api_key:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "MODEL_API_KEY is not configured")
    if api_key is None or not secrets.compare_digest(api_key, settings.model_api_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")


@app.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> dict:
    try:
        ollama = await check_ollama(settings)
    except (OllamaUnavailableError, OllamaInvalidResponseError) as exc:
        ollama = {"ok": False, "error": str(exc)}
    return {
        "status": "ok" if ollama.get("ok") else "degraded",
        "model": settings.ollama_model,
        "embedding_model": settings.ollama_embedding_model,
        "ollama": ollama,
    }


@app.post("/extract", response_model=ExtractResponse, dependencies=[Depends(require_api_key)])
async def extract_document(
    payload: ExtractRequest,
    settings: Settings = Depends(get_settings),
) -> ExtractResponse:
    result = await extract_from_url(payload, settings)
    return ExtractResponse(markdown=result.markdown, blocks=result.blocks)


@app.post("/ai/generate", response_model=GenerateResponse, dependencies=[Depends(require_api_key)])
async def generate(
    payload: GenerateRequest,
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    return await generate_text(payload.prompt, settings)


@app.post("/ai/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(require_api_key)])
async def embedding(
    payload: EmbeddingRequest,
    settings: Settings = Depends(get_settings),
) -> EmbeddingResponse:
    vectors = await generate_embeddings([payload.text], settings)
    return EmbeddingResponse(embedding=vectors[0])


@app.post(
    "/ai/embeddings/batch",
    response_model=BatchEmbeddingResponse,
    dependencies=[Depends(require_api_key)],
)
async def batch_embeddings(
    payload: BatchEmbeddingRequest,
    settings: Settings = Depends(get_settings),
) -> BatchEmbeddingResponse:
    return BatchEmbeddingResponse(embeddings=await generate_embeddings(payload.texts, settings))


@app.exception_handler(ExtractionError)
async def extraction_error_handler(_, exc: ExtractionError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ExtractionUnavailableError)
@app.exception_handler(OllamaUnavailableError)
async def unavailable_handler(_, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(OllamaInvalidResponseError)
async def invalid_upstream_handler(_, exc: OllamaInvalidResponseError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})
