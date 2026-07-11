import logging
import os

import httpx

logger = logging.getLogger(__name__)


async def generate_embeddings(text: str) -> list[float]:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    request_body = {
        "model": model,
        "prompt": text,
    }
    logger.info("embeddings request start model=%s text_chars=%s", model, len(text))

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/api/embeddings",
                json=request_body,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.info("embeddings request failed model=%s reason=%s", model, exc)
        raise RuntimeError("Ollama embeddings generation failed") from exc

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
        raise RuntimeError("Ollama returned invalid embeddings response") from exc
