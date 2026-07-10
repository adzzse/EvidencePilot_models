import json
import logging
from typing import Any

import httpx
from pydantic import ValidationError

from app.models import ClaimAnalysisRequest, ClaimAnalysisResponse, GenerateResponse, ModelsResponse
from app.settings import Settings


logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaInvalidResponseError(RuntimeError):
    pass


def build_claim_prompt(payload: ClaimAnalysisRequest) -> str:
    source = (
        f"- id: {payload.source_id}\n"
        f"  title: {payload.title or 'Untitled source'}\n"
        f"  excerpt: {payload.excerpt}"
    )
    return f"""You are an evidence reviewer for academic writing.
Judge whether the claim is supported only by the provided source excerpts.
Do not use outside knowledge. Do not invent sources.

Return strict JSON only, with exactly these keys:
- verdict: one of "supported", "partially_supported", "unsupported", "unclear"
- confidence: number from 0.0 to 1.0
- matched_source_ids: array of source ids that support the verdict
- missing_evidence: array of concise notes about gaps or needed evidence
- reasoning_summary: 2-4 concise sentences explaining the evidence comparison process, without hidden chain-of-thought
- explanation: concise explanation under 120 words

Claim:
{payload.claim}

Sources:
{source}
"""


async def check_ollama(settings: Settings) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        raise OllamaUnavailableError("Ollama is not reachable") from exc

    try:
        models = ModelsResponse.model_validate(response.json()).models
    except (TypeError, ValueError, ValidationError) as exc:
        raise OllamaInvalidResponseError("Ollama returned invalid models response") from exc

    model_names = {model.name for model in models}
    return {
        "ok": True,
        "model_available": _model_available(settings.ollama_model, model_names),
        "embedding_model_available": _model_available(settings.ollama_embedding_model, model_names),
    }


def _model_available(configured_model: str, available_models: set[str | None]) -> bool:
    normalized = {model for model in available_models if model}
    if configured_model in normalized:
        return True
    if ":" not in configured_model and f"{configured_model}:latest" in normalized:
        return True
    return any(model.split(":", 1)[0] == configured_model for model in normalized)


async def analyze_claim(payload: ClaimAnalysisRequest, settings: Settings) -> ClaimAnalysisResponse:
    prompt = build_claim_prompt(payload)
    request_body = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0},
    }
    logger.info(
        "ai request start model=%s claim_chars=%s source_count=%s",
        settings.ollama_model,
        len(payload.claim),
        1,
    )
    logger.debug("ai prompt %s", prompt)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json=request_body,
            )
            response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.info("ai request failed model=%s reason=%s", settings.ollama_model, exc)
        raise OllamaUnavailableError("Ollama generation failed") from exc

    try:
        response_data = response.json()
        raw_model_text = response_data["response"]
        logger.debug("ai raw response %s", raw_model_text)
        if not raw_model_text.strip():
            logger.info(
                "ai response empty done_reason=%s thinking_chars=%s",
                response_data.get("done_reason"),
                len(response_data.get("thinking") or ""),
            )
        model_json = json.loads(raw_model_text)
        reasoning_summary = model_json.get("reasoning_summary")
        if reasoning_summary:
            logger.debug("ai reasoning summary %s", reasoning_summary)
        result = ClaimAnalysisResponse.model_validate(model_json)
        if any(source_id != payload.source_id for source_id in result.matched_source_ids):
            raise ValueError("matched_source_ids contains an unknown source id")
        logger.info(
            "ai verdict verdict=%s confidence=%s matched_source_ids=%s missing_evidence=%s",
            result.verdict,
            result.confidence,
            result.matched_source_ids,
            result.missing_evidence,
        )
        logger.info("ai explanation %s", result.explanation)
        return result
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        logger.info("ai response invalid reason=%s", exc)
        raise OllamaInvalidResponseError("Ollama returned invalid claim-analysis JSON") from exc


async def list_models(settings: Settings) -> ModelsResponse:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.info("models request failed reason=%s", exc)
        raise OllamaUnavailableError("Ollama models request failed") from exc

    try:
        return ModelsResponse.model_validate(response.json())
    except (TypeError, ValueError, ValidationError) as exc:
        logger.info("models response invalid reason=%s", exc)
        raise OllamaInvalidResponseError("Ollama returned invalid models response") from exc


async def generate_text(prompt: str, settings: Settings) -> GenerateResponse:
    request_body = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "think": False,
    }
    logger.info(
        "generate request start model=%s prompt_chars=%s",
        settings.ollama_model,
        len(prompt),
    )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json=request_body,
            )
            response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.info("generate request failed model=%s reason=%s", settings.ollama_model, exc)
        raise OllamaUnavailableError("Ollama generation failed") from exc

    try:
        response_data = response.json()
        result = GenerateResponse(
            model=response_data["model"],
            response=response_data["response"],
            done=response_data["done"],
        )
        logger.info("generate request complete response_chars=%s", len(result.response))
        return result
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        logger.info("generate response invalid reason=%s", exc)
        raise OllamaInvalidResponseError("Ollama returned invalid generation response") from exc
async def generate_embeddings(text: str, settings: Settings) -> list[float]:
    request_body = {
        "model": settings.ollama_embedding_model,
        "prompt": text,
    }
    logger.info(
        "embeddings request start model=%s text_chars=%s",
        settings.ollama_embedding_model,
        len(text),
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/embeddings",
                json=request_body,
            )
            response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.info(
            "embeddings request failed model=%s reason=%s",
            settings.ollama_embedding_model,
            exc,
        )
        raise OllamaUnavailableError("Ollama embeddings generation failed") from exc

    try:
        response_data = response.json()
        embedding = response_data.get("embedding")
        if embedding is None:
            embeddings = response_data.get("embeddings")
            if embeddings and isinstance(embeddings, list):
                embedding = embeddings[0]
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("No embedding vector found in response")
        logger.info("embeddings request complete dimensions=%s", len(embedding))
        return embedding
    except (KeyError, TypeError, ValueError) as exc:
        logger.info("embeddings response invalid reason=%s", exc)
        raise OllamaInvalidResponseError("Ollama returned invalid embeddings response") from exc
