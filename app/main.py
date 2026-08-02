import logging
import os
import secrets
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from app.extraction import (
    ExtractionError,
    ExtractionUnavailableError,
    create_extraction_bundle,
)
from app.generation import (
    GenerationConfigurationError,
    GenerationInvalidResponseError,
    GenerationUnavailableError,
    generate_text,
    select_generation_provider,
)
from app.models import (
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ExtractRequest,
    GenerateRequest,
    GenerateResponse,
)
from app.ollama_client import (
    OllamaInvalidResponseError,
    OllamaUnavailableError,
    check_ollama,
    generate_embeddings,
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
    generation_provider = settings.generation_provider
    try:
        provider = select_generation_provider(settings)
        generation_provider = provider.name
        try:
            generation = await provider.health()
        except (
            GenerationUnavailableError,
            GenerationInvalidResponseError,
            OllamaUnavailableError,
            OllamaInvalidResponseError,
        ) as exc:
            generation = {
                "ok": False,
                "provider": generation_provider,
                "error": str(exc),
            }
    except GenerationConfigurationError as exc:
        generation = {
            "ok": False,
            "provider": generation_provider,
            "error": str(exc),
        }

    if generation_provider == "ollama":
        ollama = generation
    else:
        try:
            ollama = await check_ollama(settings, require_generation=False)
        except (OllamaUnavailableError, OllamaInvalidResponseError) as exc:
            ollama = {"ok": False, "error": str(exc)}
    return {
        "status": "ok" if generation.get("ok") and ollama.get("ok") else "degraded",
        "model": settings.ollama_model,
        "embedding_model": settings.ollama_embedding_model,
        "ollama": ollama,
        "generation_provider": generation_provider,
        "generation": generation,
    }


@app.post("/extract", dependencies=[Depends(require_api_key)])
async def extract_document(
    payload: ExtractRequest,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    descriptor, raw_path = tempfile.mkstemp(
        prefix="evidencepilot-extraction-",
        suffix=".zip",
    )
    os.close(descriptor)
    bundle_path = Path(raw_path)
    try:
        await create_extraction_bundle(payload, settings, bundle_path)
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise
    return FileResponse(
        bundle_path,
        media_type="application/zip",
        filename="extraction.zip",
        background=BackgroundTask(bundle_path.unlink, missing_ok=True),
    )


@app.post("/ai/generate", response_model=GenerateResponse, dependencies=[Depends(require_api_key)])
async def generate(
    payload: GenerateRequest,
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    return await generate_text(payload.system, payload.prompt, settings)


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
@app.exception_handler(GenerationConfigurationError)
@app.exception_handler(GenerationUnavailableError)
async def unavailable_handler(_, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(OllamaInvalidResponseError)
@app.exception_handler(GenerationInvalidResponseError)
async def invalid_upstream_handler(_, exc: RuntimeError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})
