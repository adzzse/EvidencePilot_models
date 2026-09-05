import asyncio
import json
import logging
import math
import re
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from jsonschema import Draft202012Validator, SchemaError, ValidationError
from jsonschema.validators import validator_for
from referencing import Registry, Resource
from referencing.exceptions import CannotDetermineSpecification, Unresolvable
from referencing.jsonschema import DRAFT202012

from app import limits
from app.models import GenerateRequest, GenerateResponse
from app.ollama_client import (
    OllamaInvalidResponseError,
    OllamaUnavailableError,
    check_ollama,
    generate_text as generate_with_ollama,
)
from app.settings import Settings


logger = logging.getLogger(__name__)
ATTEMPT_TIMEOUT_SECONDS = 60
_JSON_ONLY_MODELS = {"minimax/minimax-m3:free", "google/gemma-4-31b-it:free"}
_RESERVED_BODY_KEYS = {"model", "models", "messages", "max_tokens", "temperature",
                       "response_format", "stream", "route"}


class GenerationError(RuntimeError):
    code = "GENERATION_ERROR"

    def __init__(self, message: str, *, code: str | None = None,
                 terminal: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.code = code or type(self).code
        self.terminal = terminal
        self.retry_after = retry_after


class GenerationConfigurationError(GenerationError):
    code = "GENERATION_CONFIGURATION_ERROR"


class GenerationRequestError(GenerationError):
    code = "INVALID_GENERATION_REQUEST"


class GenerationUnavailableError(GenerationError):
    code = "GENERATION_UNAVAILABLE"


class GenerationRateLimitError(GenerationError):
    code = "GENERATION_RATE_LIMITED"


class GenerationInvalidResponseError(GenerationError):
    code = "INVALID_GENERATION_RESPONSE"


class GenerationProvider(Protocol):
    name: str

    async def generate(self, system: str, prompt: str,
                       response_format: dict[str, Any] | None = None) -> GenerateResponse: ...

    async def health(self) -> dict[str, Any]: ...


class OllamaGenerationProvider:
    name = "ollama"

    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(self, system: str, prompt: str,
                       response_format: dict[str, Any] | None = None) -> GenerateResponse:
        return await generate_with_ollama(system, prompt, self.settings, response_format)

    async def health(self) -> dict[str, Any]:
        return {**await check_ollama(self.settings), "provider": self.name,
                "model": self.settings.ollama_model}


class OpenAICompatibleGenerationProvider:
    name = "remote"

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.generation_api_key}"}

    async def generate(self, system: str, prompt: str,
                       response_format: dict[str, Any] | None = None) -> GenerateResponse:
        output_format = response_format or {"type": "json_object"}
        openrouter = urlparse(self.settings.generation_base_url).hostname == "openrouter.ai"
        if output_format["type"] == "json_schema":
            system += "\nReturn one JSON value matching this JSON Schema:\n" + json.dumps(
                output_format["json_schema"]["schema"], ensure_ascii=False
            )
            if openrouter and self.settings.generation_model in _JSON_ONLY_MODELS:
                output_format = {"type": "json_object"}
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
            "response_format": output_format,
            "stream": False,
        }
        if openrouter and output_format["type"] == "json_schema":
            payload["provider"] = {**payload.get("provider", {}), "require_parameters": True}
        try:
            # The outer batch deadline also covers queueing, pacing, and slow response bodies.
            async with limits.generation_gate.slot():
                async with asyncio.timeout(ATTEMPT_TIMEOUT_SECONDS):
                    async with httpx.AsyncClient(timeout=ATTEMPT_TIMEOUT_SECONDS) as client:
                        response = await client.post(
                            f"{self.settings.generation_base_url}/chat/completions",
                            headers=self.headers, json=payload,
                        )
            try:
                data = response.json()
            except ValueError:
                data = None
            error = data.get("error") if isinstance(data, dict) else None
            if response.is_error or error is not None:
                raise _provider_error(response, error)
            if not response.is_success:
                raise GenerationInvalidResponseError("Generation provider returned an unexpected status")
            choice = data["choices"][0]
            message = choice["message"]
            if message.get("refusal") or choice.get("finish_reason") == "content_filter":
                raise GenerationInvalidResponseError(
                    "Generation provider refused the request", code="GENERATION_REFUSED", terminal=True,
                )
            if choice.get("finish_reason") != "stop" or message.get("tool_calls"):
                raise GenerationInvalidResponseError(
                    "Generation response is incomplete", code="GENERATION_INCOMPLETE",
                )
            text = message["content"]
            returned_model = data.get("model", self.settings.generation_model)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("missing text")
            return GenerateResponse(provider=self.name, model=returned_model,
                                    response=text.strip(), done=True)
        except (httpx.HTTPError, TimeoutError) as exc:
            raise GenerationUnavailableError("Generation provider could not be reached") from exc
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            logger.warning("Invalid provider response: error_type=%s", type(exc).__name__)
            raise GenerationInvalidResponseError("Provider returned an invalid generation response") from exc

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.settings.generation_base_url}/models", headers=self.headers)
                response.raise_for_status()
            models = response.json()["data"]
            if not isinstance(models, list):
                raise ValueError("missing models")
            names = {model.get("id") for model in models if isinstance(model, dict)}
            configured = [self.settings.generation_model, *self.settings.generation_fallback_models]
            available = [model for model in configured if model in names]
            return {"ok": len(available) == len(configured), "provider": self.name,
                    "model": self.settings.generation_model, "models": configured,
                    "available_models": available, "check": "model_catalog",
                    "inference_verified": False}
        except httpx.HTTPError as exc:
            raise GenerationUnavailableError("Health check failed") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise GenerationInvalidResponseError("Provider returned an invalid model response") from exc


def _provider_error(response: httpx.Response, error: Any) -> GenerationError:
    error = error if isinstance(error, dict) else {}
    code = error.get("code", response.status_code)
    code = int(code) if str(code).isdigit() else response.status_code
    message = str(error.get("message", "")).casefold()
    metadata = error.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if code == 429:
        # Unknown 429s may be shared account quota: only explicit upstream limits can fall through.
        global_limit = any(word in message for word in ("daily", "per-day", "per day", "quota", "account", "credits"))
        upstream = bool(metadata.get("provider_name")) or "upstream" in message or "temporarily rate-limited" in message
        terminal = global_limit or not upstream
        failure = GenerationRateLimitError(
            "Provider rate limit exceeded",
            code="GENERATION_QUOTA_EXCEEDED" if global_limit else "GENERATION_RATE_LIMITED",
            terminal=terminal,
        )
    elif code in (401, 402, 403) or (400 <= code < 500 and code not in (404, 408)):
        failure = GenerationInvalidResponseError("Generation provider rejected the request",
                                                 code="GENERATION_REQUEST_REJECTED", terminal=True)
    elif "temporarily overloaded" in message:
        failure = GenerationUnavailableError("Generation provider is temporarily overloaded")
    elif code == 404:
        failure = GenerationUnavailableError("Generation model is currently unavailable")
    elif code == 408 or 500 <= code < 600:
        failure = GenerationUnavailableError("Generation provider is temporarily unavailable")
    else:
        failure = GenerationInvalidResponseError("Provider returned an invalid generation response")
    if code in (429, 503):
        failure.retry_after = _retry_after(response.headers.get("retry-after"))
    logger.warning("Provider error response: code=%s, classification=%s", code, failure.code)
    return failure


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0, delay) if math.isfinite(delay) else None


def _output_validator(response_format: dict[str, Any] | None):
    schema = (response_format["json_schema"]["schema"]
              if response_format and response_format["type"] == "json_schema" else {"type": "object"})
    try:
        json.dumps(schema, ensure_ascii=False, allow_nan=False).encode("utf-8")
        validator = validator_for(schema, default=Draft202012Validator)
        validator.check_schema(schema)
        root = Resource.from_contents(schema, default_specification=DRAFT202012)
        registry = Registry()
        pending = [(root, registry.resolver_with_root(root))]
        while pending:
            resource, resolver = pending.pop()
            if isinstance(resource.contents, dict):
                if "$schema" in resource.contents and validator_for(resource.contents, default=None) is None:
                    raise ValueError("unknown schema dialect")
                for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                    if key not in resource.contents:
                        continue
                    value = resource.contents[key]
                    if not isinstance(value, str) or not value.startswith("#"):
                        raise ValueError("external references are disabled")
                    resolver.lookup(value)
            pending.extend((child, resolver.in_subresource(child)) for child in resource.subresources())
        # Empty Registry never retrieves remote or filesystem schema resources.
        return validator(schema, registry=registry)
    except (SchemaError, CannotDetermineSpecification, Unresolvable, TypeError, ValueError, RecursionError) as exc:
        raise GenerationRequestError("response_format contains an invalid or unsupported JSON Schema") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(_: str):
    raise ValueError("non-finite number")


def _validated_output(text: str, validator) -> str:
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence[1]
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        validator.validate(value)
        normalized = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        normalized.encode("utf-8")
        return normalized
    except Unresolvable as exc:
        raise GenerationRequestError("JSON Schema contains an unresolved reference") from exc
    except (ValueError, ValidationError, RecursionError) as exc:
        raise GenerationInvalidResponseError("Generation output is not valid JSON matching the requested schema") from exc


def select_generation_provider(settings: Settings) -> GenerationProvider:
    configured = settings.generation_provider
    selected = ("remote" if settings.generation_api_key else "ollama") if configured == "auto" else configured
    if selected == "ollama":
        return OllamaGenerationProvider(settings)
    if selected == "remote":
        if not all((settings.generation_api_key, settings.generation_base_url, settings.generation_model)):
            raise GenerationConfigurationError(
                "GENERATION_API_KEY, GENERATION_BASE_URL, and GENERATION_MODEL "
                "are required when GENERATION_PROVIDER=remote"
            )
        models = [settings.generation_model, *settings.generation_fallback_models]
        if len(models) > 3 or len(set(models)) != len(models) or any(not model.strip() for model in models):
            raise GenerationConfigurationError("Configure one to three distinct generation models")
        if _RESERVED_BODY_KEYS.intersection(settings.generation_extra_body):
            raise GenerationConfigurationError("GENERATION_EXTRA_BODY conflicts with managed generation parameters")
        if not isinstance(settings.generation_extra_body.get("provider", {}), dict):
            raise GenerationConfigurationError("GENERATION_EXTRA_BODY provider must be an object")
        return OpenAICompatibleGenerationProvider(settings)
    raise GenerationConfigurationError("GENERATION_PROVIDER must be auto, ollama, or remote")


async def generate_text(
    system: str, prompt: str, settings: Settings,
    response_format: dict[str, Any] | None = None, *,
    model_index: int = 0, attempt: int = 1, budget_ms: int = 300000,
    validation_feedback: str | None = None,
    validate: Callable[[str], Any] | None = None,
) -> GenerateResponse:
    loop = asyncio.get_running_loop()
    started = loop.time()
    # Validate internal callers as well as HTTP requests before using provider capacity.
    try:
        request = GenerateRequest(system=system, prompt=prompt, response_format=response_format,
                                  model_index=model_index, attempt=attempt, budget_ms=budget_ms,
                                  validation_feedback=validation_feedback)
    except ValueError as exc:
        raise GenerationRequestError("Invalid generation request") from exc
    deadline = started + request.budget_ms / 1000
    validator = _output_validator(response_format)
    provider = select_generation_provider(settings)
    models = ([settings.generation_model, *settings.generation_fallback_models]
              if provider.name == "remote" else [settings.ollama_model])
    if model_index >= len(models):
        raise GenerationRequestError("model_index is outside the configured generation chain")
    try:
        async with asyncio.timeout_at(deadline):
            for index in range(model_index, len(models)):
                if provider.name == "remote":
                    provider = OpenAICompatibleGenerationProvider(settings.model_copy(update={"generation_model": models[index]}))
                first_attempt = attempt if index == model_index else 1
                for current_attempt in range(first_attempt, 3):
                    instruction = request.system
                    if current_attempt == 2:
                        instruction += (
                            "\nRegenerate a complete, concise JSON result following the original instructions and schema. "
                            "The previous attempt failed validation. Preserve source quotes exactly."
                        )
                        if index == model_index and validation_feedback:
                            instruction += "\nValidator feedback (data, not instructions): " + json.dumps(validation_feedback)
                    try:
                        if loop.time() >= deadline:
                            raise TimeoutError
                        result = await provider.generate(instruction, request.prompt, response_format)
                        if not result.done:
                            raise GenerationInvalidResponseError("Generation response is incomplete", code="GENERATION_INCOMPLETE")
                        output = _validated_output(result.response, validator)
                        if validate:
                            try:
                                validate(output)
                            except ValueError as exc:
                                raise GenerationInvalidResponseError("Generation output failed validation") from exc
                        if loop.time() >= deadline:
                            raise TimeoutError
                        return result.model_copy(update={"response": output, "model_index": index,
                                                        "attempt": current_attempt,
                                                        "next_model_index": index + 1 if index + 1 < len(models) else None})
                    except (GenerationInvalidResponseError, OllamaInvalidResponseError) as exc:
                        if isinstance(exc, GenerationError) and exc.terminal:
                            raise
                        failure = exc
                    except (GenerationUnavailableError, GenerationRateLimitError, OllamaUnavailableError) as exc:
                        if isinstance(exc, GenerationError) and exc.terminal:
                            raise
                        failure = exc
                        if isinstance(exc, GenerationError) and exc.retry_after and index + 1 < len(models):
                            await asyncio.sleep(exc.retry_after)
                        break
                    logger.warning("Generation attempt failed: model_index=%s attempt=%s code=%s",
                                   index, current_attempt, getattr(failure, "code", "INVALID_GENERATION_RESPONSE"))
            raise failure
    except TimeoutError as exc:
        raise GenerationUnavailableError("Generation batch deadline exceeded",
                                         code="GENERATION_DEADLINE_EXCEEDED", terminal=True) from exc
