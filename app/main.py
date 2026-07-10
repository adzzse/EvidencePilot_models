import logging

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from app.extraction import ExtractionError, check_liteparse, extract_upload_markdown
from app.logging_config import configure_app_logging
from app.models import (
    ClaimAnalysisRequest,
    ClaimAnalysisResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ExtractResponse,
    GenerateRequest,
    GenerateResponse,
    ModelsResponse,
    SparseEmbedRequest,
    SparseEmbedResponse,
)
from app.ollama_client import (
    OllamaInvalidResponseError,
    OllamaUnavailableError,
    analyze_claim,
    check_ollama,
    generate_embeddings,
    generate_text,
    list_models,
)
from app.settings import Settings, load_settings
from app.sparse import encode_sparse

configure_app_logging()
app = FastAPI(title="Ollama Claim Analysis API", version="0.2.0")
logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    return load_settings()


async def check_ollama_health(
    settings: Settings = Depends(get_settings),
) -> dict:
    return await check_ollama(settings)


def check_liteparse_health() -> dict:
    return check_liteparse()


async def run_claim_analysis(
    payload: ClaimAnalysisRequest,
    settings: Settings = Depends(get_settings),
) -> ClaimAnalysisResponse:
    return await analyze_claim(payload, settings)


@app.get("/health")
async def health(
    settings: Settings = Depends(get_settings),
    ollama_status: dict = Depends(check_ollama_health),
    liteparse_status: dict = Depends(check_liteparse_health),
) -> dict:
    healthy = (
        ollama_status.get("ok") is True
        and ollama_status.get("model_available") is True
        and ollama_status.get("embedding_model_available") is True
        and liteparse_status.get("ok") is True
    )
    return {
        "status": "ok" if healthy else "degraded",
        "model": settings.ollama_model,
        "embedding_model": settings.ollama_embedding_model,
        "ollama": ollama_status,
        "liteparse": liteparse_status,
    }


@app.get("/ai/models", response_model=ModelsResponse)
async def ai_models(
    settings: Settings = Depends(get_settings),
) -> ModelsResponse:
    return await list_models(settings)


@app.post("/process/claim", response_model=ClaimAnalysisResponse)
async def process_claim(
    result: ClaimAnalysisResponse = Depends(run_claim_analysis),
) -> ClaimAnalysisResponse:
    return result


@app.post("/extract", response_model=ExtractResponse)
async def extract_document(
    file: UploadFile = File(...),
) -> ExtractResponse:
    result = await extract_upload_markdown(file)
    return ExtractResponse(
        filename=result.filename,
        method=result.method,
        markdown=result.markdown,
    )


async def run_generate_embeddings(
    payload: EmbeddingRequest,
    settings: Settings = Depends(get_settings),
) -> EmbeddingResponse:
    logger.debug(
        "embeddings start text_chars=%s model=%s",
        len(payload.text),
        settings.ollama_embedding_model,
    )
    vector = await generate_embeddings(payload.text, settings)
    logger.debug(
        "embeddings complete text_chars=%s vector_dim=%s",
        len(payload.text),
        len(vector),
    )
    return EmbeddingResponse(embedding=vector)


async def run_generate_text(
    payload: GenerateRequest,
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    logger.debug(
        "generate start prompt_chars=%s model=%s",
        len(payload.prompt),
        settings.ollama_model,
    )
    result = await generate_text(payload.prompt, settings)
    logger.debug(
        "generate complete prompt_chars=%s response_chars=%s",
        len(payload.prompt),
        len(result.response),
    )
    return result


@app.post("/ai/generate", response_model=GenerateResponse)
async def generate(
    result: GenerateResponse = Depends(run_generate_text),
) -> GenerateResponse:
    return result


@app.post("/ai/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(
    result: EmbeddingResponse = Depends(run_generate_embeddings),
) -> EmbeddingResponse:
    return result


@app.post("/api/sparse-embed", response_model=SparseEmbedResponse)
async def sparse_embed(payload: SparseEmbedRequest) -> SparseEmbedResponse:
    return SparseEmbedResponse.model_validate(encode_sparse(payload.text))


@app.exception_handler(OllamaUnavailableError)
async def ollama_unavailable_handler(_, exc: OllamaUnavailableError):
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(exc),
    )


@app.exception_handler(OllamaInvalidResponseError)
async def ollama_invalid_response_handler(_, exc: OllamaInvalidResponseError):
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=str(exc),
    )


@app.exception_handler(ExtractionError)
async def extraction_error_handler(_, exc: ExtractionError):
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=str(exc),
    )
