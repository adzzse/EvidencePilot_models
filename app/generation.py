from typing import Any, Protocol

import httpx

from app.models import GenerateResponse
from app.ollama_client import check_ollama, generate_text as generate_with_ollama
from app.settings import Settings


class GenerationConfigurationError(RuntimeError):
    pass


class GenerationUnavailableError(RuntimeError):
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


class GeminiGenerationProvider:
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.settings.gemini_api_key}

    async def generate(self, system: str, prompt: str) -> GenerateResponse:
        payload: dict[str, Any] = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}],
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": 8192,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/{self.settings.gemini_model}:generateContent",
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()
            data = response.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(
                part["text"] for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ).strip()
            if not text:
                raise ValueError("missing generated text")
            model = data.get("modelVersion") or self.settings.gemini_model
            return GenerateResponse(
                provider=self.name,
                model=model,
                response=text,
                done=True,
            )
        except httpx.HTTPError as exc:
            raise GenerationUnavailableError("Gemini generation failed") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise GenerationInvalidResponseError(
                "Gemini returned an invalid generation response"
            ) from exc

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/{self.settings.gemini_model}",
                    headers=self.headers,
                )
                response.raise_for_status()
            data = response.json()
            if data.get("name") != f"models/{self.settings.gemini_model}":
                raise ValueError("model name mismatch")
            return {
                "ok": True,
                "provider": self.name,
                "model": self.settings.gemini_model,
            }
        except httpx.HTTPError as exc:
            raise GenerationUnavailableError("Gemini health check failed") from exc
        except (AttributeError, TypeError, ValueError) as exc:
            raise GenerationInvalidResponseError(
                "Gemini returned an invalid model response"
            ) from exc


def select_generation_provider(settings: Settings) -> GenerationProvider:
    configured = settings.generation_provider
    selected = (
        "gemini" if settings.gemini_api_key else "ollama"
    ) if configured == "auto" else configured
    if selected == "ollama":
        return OllamaGenerationProvider(settings)
    if selected == "gemini":
        if not settings.gemini_api_key:
            raise GenerationConfigurationError(
                "GEMINI_API_KEY is required when GENERATION_PROVIDER=gemini"
            )
        return GeminiGenerationProvider(settings)
    raise GenerationConfigurationError(
        "GENERATION_PROVIDER must be auto, ollama, or gemini"
    )


async def generate_text(
    system: str,
    prompt: str,
    settings: Settings,
) -> GenerateResponse:
    return await select_generation_provider(settings).generate(system, prompt)
