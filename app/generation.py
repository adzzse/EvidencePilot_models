import logging
from typing import Any, Protocol

import httpx

from app.models import GenerateResponse
from app.ollama_client import check_ollama, generate_text as generate_with_ollama
from app.settings import Settings


logger = logging.getLogger(__name__)


class GenerationConfigurationError(RuntimeError):
    pass


class GenerationUnavailableError(RuntimeError):
    pass


class GenerationRateLimitError(RuntimeError):
    pass


class GenerationInvalidResponseError(RuntimeError):
    pass


class GenerationProvider(Protocol):
    name: str

    async def generate(self, system: str, prompt: str) -> GenerateResponse: ...

    async def health(self) -> dict[str, Any]: ...


class OllamaGenerationProvider:
    name = "ollama"

    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(self, system: str, prompt: str) -> GenerateResponse:
        return await generate_with_ollama(system, prompt, self.settings)

    async def health(self) -> dict[str, Any]:
        result = await check_ollama(self.settings)
        return {
            **result,
            "provider": self.name,
            "model": self.settings.ollama_model,
        }


class OpenAICompatibleGenerationProvider:
    name = "remote"

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.generation_api_key}"
        }

    async def generate(self, system: str, prompt: str) -> GenerateResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            **self.settings.generation_extra_body,
            "model": self.settings.generation_model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.settings.generation_base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()
                try:
                    data = response.json()
                    error = data.get("error") if isinstance(data, dict) else None
                    if error is not None:
                        if not isinstance(error, dict):
                            raise TypeError("invalid provider error")
                        code = error.get("code")
                        if isinstance(code, str) and code.isdigit():
                            code = int(code)
                        message = error.get("message")
                        temporarily_overloaded = (
                            isinstance(message, str)
                            and "temporarily overloaded" in message.casefold()
                        )
                        if code == 429:
                            classification = "rate_limited"
                        elif temporarily_overloaded:
                            classification = "temporarily_overloaded"
                        elif isinstance(code, int) and 500 <= code <= 599:
                            classification = "temporarily_unavailable"
                        else:
                            classification = "invalid_response"
                        logger.warning(
                            "Provider error response: code=%r, classification=%s "
                            "(status=%s, content-type=%s)",
                            code,
                            classification,
                            response.status_code,
                            response.headers.get("content-type"),
                        )
                        if classification == "rate_limited":
                            raise GenerationRateLimitError(
                                "Provider rate limit exceeded"
                            )
                        if classification == "temporarily_overloaded":
                            raise GenerationUnavailableError(
                                "Generation provider is temporarily overloaded"
                            )
                        if classification == "temporarily_unavailable":
                            raise GenerationUnavailableError(
                                "Generation provider is temporarily unavailable"
                            )
                        raise GenerationInvalidResponseError(
                            "Provider returned an invalid generation response"
                        )
                    text = data["choices"][0]["message"]["content"]
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("missing generated text")
                    returned_model = data.get("model")
                    if returned_model is not None and not isinstance(
                        returned_model, str
                    ):
                        raise TypeError("invalid model")
                    return GenerateResponse(
                        provider=self.name,
                        model=returned_model or self.settings.generation_model,
                        response=text.strip(),
                        done=True,
                    )
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Invalid provider response: %s: %s (status=%s, content-type=%s)",
                        type(exc).__name__,
                        exc,
                        response.status_code,
                        response.headers.get("content-type"),
                    )
                    raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise GenerationRateLimitError(
                    "Provider rate limit exceeded"
                ) from exc
            if 500 <= exc.response.status_code <= 599:
                raise GenerationUnavailableError(
                    "Generation provider is temporarily unavailable"
                ) from exc
            raise GenerationUnavailableError(
                "Generation failed"
            ) from exc
        except httpx.HTTPError as exc:
            raise GenerationUnavailableError(
                "Generation failed"
            ) from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise GenerationInvalidResponseError(
                "Provider returned an invalid generation response"
            ) from exc

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.settings.generation_base_url}/models",
                    headers=self.headers,
                )
                response.raise_for_status()
            models = response.json()["data"]
            if not isinstance(models, list) or not any(
                isinstance(model, dict)
                and model.get("id") == self.settings.generation_model
                for model in models
            ):
                raise ValueError("configured model is unavailable")
            return {
                "ok": True,
                "provider": self.name,
                "model": self.settings.generation_model,
            }
        except httpx.HTTPError as exc:
            raise GenerationUnavailableError(
                "Health check failed"
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise GenerationInvalidResponseError(
                "Provider returned an invalid model response"
            ) from exc


def select_generation_provider(settings: Settings) -> GenerationProvider:
    configured = settings.generation_provider
    selected = (
        "remote"
        if settings.generation_api_key
        else "ollama"
    ) if configured == "auto" else configured
    if selected == "ollama":
        return OllamaGenerationProvider(settings)
    if selected == "remote":
        if not all((
            settings.generation_api_key,
            settings.generation_base_url,
            settings.generation_model,
        )):
            raise GenerationConfigurationError(
                "GENERATION_API_KEY, GENERATION_BASE_URL, and GENERATION_MODEL "
                "are required when GENERATION_PROVIDER=remote"
            )
        return OpenAICompatibleGenerationProvider(settings)
    raise GenerationConfigurationError(
        "GENERATION_PROVIDER must be auto, ollama, or remote"
    )


async def generate_text(
    system: str,
    prompt: str,
    settings: Settings,
) -> GenerateResponse:
    return await select_generation_provider(settings).generate(system, prompt)
