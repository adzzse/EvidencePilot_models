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


class OpenAICompatibleGenerationProvider:
    name = "openai_compatible"

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.openai_compatible_api_key}"
        }

    async def generate(self, system: str, prompt: str) -> GenerateResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.settings.openai_compatible_model,
            "messages": messages,
            "max_tokens": 8192,
            "stream": False,
        }
        if self.settings.openai_compatible_model.startswith("deepseek-v4"):
            payload["thinking"] = {"type": "disabled"}
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                for attempt in range(2):
                    response = await client.post(
                        f"{self.settings.openai_compatible_base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    try:
                        data = response.json()
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
                            model=returned_model
                            or self.settings.openai_compatible_model,
                            response=text.strip(),
                            done=True,
                        )
                    except (KeyError, IndexError, TypeError, ValueError):
                        if attempt == 1:
                            raise
        except httpx.HTTPError as exc:
            raise GenerationUnavailableError(
                "OpenAI-compatible generation failed"
            ) from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise GenerationInvalidResponseError(
                "OpenAI-compatible provider returned an invalid generation response"
            ) from exc

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.settings.openai_compatible_base_url}/models",
                    headers=self.headers,
                )
                response.raise_for_status()
            models = response.json()["data"]
            if not isinstance(models, list) or not any(
                isinstance(model, dict)
                and model.get("id") == self.settings.openai_compatible_model
                for model in models
            ):
                raise ValueError("configured model is unavailable")
            return {
                "ok": True,
                "provider": self.name,
                "model": self.settings.openai_compatible_model,
            }
        except httpx.HTTPError as exc:
            raise GenerationUnavailableError(
                "OpenAI-compatible health check failed"
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise GenerationInvalidResponseError(
                "OpenAI-compatible provider returned an invalid model response"
            ) from exc


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
        "openai_compatible"
        if settings.openai_compatible_api_key
        else "ollama"
    ) if configured == "auto" else configured
    if selected == "ollama":
        return OllamaGenerationProvider(settings)
    if selected == "openai_compatible":
        if not settings.openai_compatible_api_key:
            raise GenerationConfigurationError(
                "OPENAI_COMPATIBLE_API_KEY is required when "
                "GENERATION_PROVIDER=openai_compatible"
            )
        return OpenAICompatibleGenerationProvider(settings)
    if selected == "gemini":
        if not settings.gemini_api_key:
            raise GenerationConfigurationError(
                "GEMINI_API_KEY is required when GENERATION_PROVIDER=gemini"
            )
        return GeminiGenerationProvider(settings)
    raise GenerationConfigurationError(
        "GENERATION_PROVIDER must be auto, ollama, gemini, or "
        "openai_compatible"
    )


async def generate_text(
    system: str,
    prompt: str,
    settings: Settings,
) -> GenerateResponse:
    return await select_generation_provider(settings).generate(system, prompt)
