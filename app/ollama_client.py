import logging
from typing import Any

import httpx

from app.models import GenerateResponse
from app.settings import Settings


logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaInvalidResponseError(RuntimeError):
    pass


async def check_ollama(
    settings: Settings,
    require_generation: bool = True,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
        models = response.json().get("models", [])
        names = {model.get("name", "") for model in models if isinstance(model, dict)}
        if not names:
            raise ValueError("missing models")
    except httpx.HTTPError as exc:
        raise OllamaUnavailableError("Ollama is not reachable") from exc
    except (TypeError, ValueError) as exc:
        raise OllamaInvalidResponseError("Ollama returned an invalid models response") from exc

    model_available = _available(settings.ollama_model, names)
    embedding_model_available = _available(settings.ollama_embedding_model, names)
    return {
        "ok": embedding_model_available and (
            model_available or not require_generation
        ),
        "model_available": model_available,
        "embedding_model_available": embedding_model_available,
    }


async def generate_text(
    system: str,
    prompt: str,
    settings: Settings,
    response_format: dict[str, Any] | None = None,
) -> GenerateResponse:
    output_format = (
        response_format["json_schema"]["schema"]
        if response_format and response_format["type"] == "json_schema"
        else "json"
    )
    payload = {
        "model": settings.ollama_model,
        "system": system,
        "prompt": prompt,
        "format": output_format,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_ctx": 262144,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
            response.raise_for_status()
        data = response.json()
        return GenerateResponse(
            provider="ollama",
            model=data["model"],
            response=data["response"],
            done=data["done"],
        )
    except httpx.HTTPError as exc:
        raise OllamaUnavailableError("Ollama generation failed") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise OllamaInvalidResponseError("Ollama returned an invalid generation response") from exc


async def generate_embeddings(texts: list[str], settings: Settings) -> list[list[float]]:
    payload = {"model": settings.ollama_embedding_model, "input": texts}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{settings.ollama_base_url}/api/embed", json=payload)
            response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ValueError("embedding count mismatch")
        if any(not isinstance(vector, list) or not vector for vector in embeddings):
            raise ValueError("empty embedding")
        return [[float(value) for value in vector] for vector in embeddings]
    except httpx.HTTPError as exc:
        raise OllamaUnavailableError("Ollama embeddings generation failed") from exc
    except (TypeError, ValueError) as exc:
        raise OllamaInvalidResponseError("Ollama returned invalid embeddings") from exc


def _available(configured: str, names: set[str]) -> bool:
    return configured in names or f"{configured}:latest" in names or any(
        name.split(":", 1)[0] == configured for name in names
    )
