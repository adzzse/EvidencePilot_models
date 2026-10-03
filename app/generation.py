import asyncio
import hashlib
import json
import logging
import math
import re
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from itertools import product
from typing import Any, Protocol
from urllib.parse import urlparse, urlsplit, urlunsplit

import httpx
from jsonschema import Draft202012Validator, SchemaError, ValidationError
from jsonschema.validators import validator_for
from referencing import Registry, Resource
from referencing.exceptions import CannotDetermineSpecification, Unresolvable
from referencing.jsonschema import DRAFT202012

from app import limits
from app import openrouter_catalog
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


class GenerationConfigChangedError(GenerationError):
    code = "GENERATION_CONFIG_CHANGED"


class GenerationUnavailableError(GenerationError):
    code = "GENERATION_UNAVAILABLE"


class GenerationRateLimitError(GenerationError):
    code = "GENERATION_RATE_LIMITED"


class GenerationInvalidResponseError(GenerationError):
    code = "INVALID_GENERATION_RESPONSE"


_model_cooldowns: dict[tuple[str, str, str], tuple[float, GenerationError]] = {}


def _cooldown_key(settings: Settings, key_index: int, model: str) -> tuple[str, str, str]:
    key_hash = hashlib.sha256(configured_generation_keys(settings)[key_index].encode()).hexdigest()
    return settings.generation_base_url.rstrip("/"), key_hash, model


def _model_cooldown(settings: Settings, key_index: int, model: str) -> GenerationError | None:
    identity = _cooldown_key(settings, key_index, model)
    cooldown = _model_cooldowns.get(identity)
    if cooldown is None:
        return None
    expires, failure = cooldown
    remaining = expires - time.monotonic()
    if remaining <= 0:
        del _model_cooldowns[identity]
        return None
    logger.info("Generation model skipped: model=%s key_slot=%s cooldown_seconds=%.2f code=%s",
                model, key_index + 1, remaining, failure.code)
    return type(failure)(str(failure), code=failure.code, retry_after=remaining)


def _cool_down_model(settings: Settings, key_index: int, model: str, failure: GenerationError) -> None:
    duration = failure.retry_after
    if duration is None or duration <= 0:
        return
    identity = _cooldown_key(settings, key_index, model)
    expires = time.monotonic() + duration
    current = _model_cooldowns.get(identity)
    if current is None or expires > current[0]:
        _model_cooldowns[identity] = (expires, type(failure)(str(failure), code=failure.code))
    logger.warning("Generation model cooldown: model=%s key_slot=%s cooldown_seconds=%.2f code=%s",
                   model, key_index + 1, duration, failure.code)


def configured_generation_keys(settings: Settings) -> tuple[str, ...]:
    keys = tuple(settings.generation_api_keys)
    if settings.generation_api_key and keys:
        raise GenerationConfigurationError("Configure GENERATION_API_KEY or GENERATION_API_KEYS, not both")
    if keys:
        if len(keys) > 3 or any(not isinstance(key, str) or not key.strip() for key in keys):
            raise GenerationConfigurationError("Configure one to three nonempty generation API keys")
        keys = tuple(key.strip() for key in keys)
        if len(set(keys)) != len(keys):
            raise GenerationConfigurationError("Generation API keys must be distinct")
        if len(keys) > 1 and urlparse(settings.generation_base_url).hostname != "openrouter.ai":
            raise GenerationConfigurationError("Multiple generation API keys require OpenRouter")
    return keys or ((settings.generation_api_key,) if settings.generation_api_key else ())


def _dynamic_openrouter(settings: Settings) -> bool:
    return (settings.generation_provider != "ollama"
            and urlparse(settings.generation_base_url).hostname == "openrouter.ai"
            and not settings.generation_model and not settings.generation_fallback_models)


def _catalog_snapshot(settings: Settings) -> openrouter_catalog.CatalogSnapshot:
    try:
        return openrouter_catalog.get_catalog(settings.generation_base_url,
                                              configured_generation_keys(settings))
    except openrouter_catalog.CatalogAuthenticationError as exc:
        raise GenerationConfigurationError("OpenRouter model catalog access rejected") from exc
    except openrouter_catalog.CatalogUnavailableError as exc:
        raise GenerationUnavailableError("OpenRouter model catalog is unavailable", terminal=True) from exc


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

    def __init__(self, settings: Settings, *, schema_supported: bool | None = None,
                 json_supported: bool | None = None, key_index: int = 0):
        self.settings = settings
        self.schema_supported = schema_supported
        self.json_supported = json_supported
        self.key_index = key_index

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {configured_generation_keys(self.settings)[self.key_index]}"}

    async def _post(self, payload: dict[str, Any]) -> tuple[httpx.Response, Any, Any]:
        started = time.monotonic()
        try:
            async with asyncio.timeout(ATTEMPT_TIMEOUT_SECONDS):
                async with httpx.AsyncClient(timeout=ATTEMPT_TIMEOUT_SECONDS) as client:
                    async with limits.generation_gate.slot():
                        cooldown = _model_cooldown(self.settings, self.key_index, payload["model"])
                        if cooldown is not None:
                            raise cooldown
                        response = await client.post(
                            f"{self.settings.generation_base_url}/chat/completions",
                            headers=self.headers, json=payload,
                        )
            try:
                data = response.json()
            except ValueError:
                data = None
            error = data.get("error") if isinstance(data, dict) else None
            return response, data, error
        except (httpx.HTTPError, TimeoutError) as exc:
            details = "".join(traceback.format_exception(exc))
            for key in configured_generation_keys(self.settings):
                details = details.replace(key, "<REDACTED>")
            logger.warning("Generation provider call failed: model=%s key_slot=%s elapsed_ms=%s error_type=%s\n%s",
                           self.settings.generation_model, self.key_index + 1,
                           int((time.monotonic() - started) * 1000), type(exc).__name__, details)
            raise

    async def generate(self, system: str, prompt: str,
                       response_format: dict[str, Any] | None = None) -> GenerateResponse:
        output_format = response_format or {"type": "json_object"}
        openrouter = urlparse(self.settings.generation_base_url).hostname == "openrouter.ai"
        if output_format["type"] == "json_schema":
            system += "\nReturn one JSON value matching this JSON Schema:\n" + json.dumps(
                output_format["json_schema"]["schema"], ensure_ascii=False
            )
            if openrouter and (self.schema_supported is False or
                               self.schema_supported is None and self.settings.generation_model in _JSON_ONLY_MODELS):
                output_format = {"type": "json_object"} if self.json_supported is not False else None
        elif self.json_supported is False:
            system += "\nReturn exactly one valid JSON value."
            output_format = None
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
            "stream": False,
        }
        if output_format is not None:
            payload["response_format"] = output_format
        if openrouter and output_format is not None and (
                self.schema_supported is not None or output_format["type"] == "json_schema"):
            payload["provider"] = {**payload.get("provider", {}), "require_parameters": True}
        try:
            # The outer batch deadline also covers queueing, pacing, and slow response bodies.
            response, data, error = await self._post(payload)
            if response.is_error or error is not None:
                failure = _provider_error(response, error)
                if isinstance(failure, (GenerationUnavailableError, GenerationRateLimitError)) and not failure.terminal:
                    _cool_down_model(self.settings, self.key_index, payload["model"], failure)
                raise failure
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
        if _dynamic_openrouter(self.settings):
            snapshot = await asyncio.to_thread(_catalog_snapshot, self.settings)
            return {"ok": bool(snapshot.allowed_models), "provider": self.name,
                    "model": snapshot.default_models[0], "models": list(snapshot.default_models),
                    "available_models": list(snapshot.allowed_models), "check": "model_catalog",
                    "inference_verified": False}
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


def _is_rate_limit_response(response: httpx.Response, error: Any) -> bool:
    return (response.status_code == 429 or
            isinstance(error, dict) and str(error.get("code")) == "429")


def _provider_error(response: httpx.Response, error: Any) -> GenerationError:
    error = error if isinstance(error, dict) else {}
    code = 429 if _is_rate_limit_response(response, error) else error.get("code", response.status_code)
    code = int(code) if str(code).isdigit() else response.status_code
    message = str(error.get("message", "")).casefold()
    metadata = error.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if code == 429:
        # A quota can apply only to this model; continue the selected chain on the same key.
        upstream = (metadata.get("provider_code") is not None or bool(metadata.get("provider_name"))
                    or "upstream" in message or "temporarily rate-limited" in message)
        global_limit = not upstream and any(
            word in message for word in ("daily", "per-day", "per day", "quota", "account", "credits")
        )
        failure = GenerationRateLimitError(
            "Provider rate limit exceeded",
            code="GENERATION_QUOTA_EXCEEDED" if global_limit else "GENERATION_RATE_LIMITED",
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
    if isinstance(failure, (GenerationUnavailableError, GenerationRateLimitError)):
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
    keys = configured_generation_keys(settings)
    selected = ("remote" if keys else "ollama") if configured == "auto" else configured
    if selected == "ollama":
        return OllamaGenerationProvider(settings)
    if selected == "remote":
        if not all((keys, settings.generation_base_url)) or not (
            settings.generation_model or _dynamic_openrouter(settings)
        ):
            raise GenerationConfigurationError(
                "GENERATION_API_KEY or GENERATION_API_KEYS, GENERATION_BASE_URL, and a model "
                "or OpenRouter catalog mode are required"
            )
        if not _dynamic_openrouter(settings):
            models = [settings.generation_model, *settings.generation_fallback_models]
            if len(models) > 3 or len(set(models)) != len(models) or any(not model.strip() for model in models):
                raise GenerationConfigurationError("Configure one to three distinct generation models")
        if _RESERVED_BODY_KEYS.intersection(settings.generation_extra_body):
            raise GenerationConfigurationError("GENERATION_EXTRA_BODY conflicts with managed generation parameters")
        if not isinstance(settings.generation_extra_body.get("provider", {}), dict):
            raise GenerationConfigurationError("GENERATION_EXTRA_BODY provider must be an object")
        return OpenAICompatibleGenerationProvider(settings)
    raise GenerationConfigurationError("GENERATION_PROVIDER must be auto, ollama, or remote")


def generation_catalog(settings: Settings, *,
                       snapshot: openrouter_catalog.CatalogSnapshot | None = None) -> dict[str, Any]:
    provider = select_generation_provider(settings)
    dynamic = provider.name == "remote" and _dynamic_openrouter(settings)
    if dynamic:
        snapshot = snapshot or _catalog_snapshot(settings)
        models = list(snapshot.allowed_models)
        defaults = list(snapshot.default_models)
    else:
        models = ([settings.generation_model, *settings.generation_fallback_models]
                  if provider.name == "remote" else [settings.ollama_model])
        defaults = models
    endpoint = settings.generation_base_url if provider.name == "remote" else settings.ollama_base_url
    parsed = urlsplit(endpoint)
    normalized_endpoint = urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(),
                                      parsed.path.rstrip("/"), "", ""))
    canonical = {
        "protocol_version": 1,
        "provider": provider.name,
        "endpoint": normalized_endpoint,
        "allowed_models": "openrouter-free-json-v1" if dynamic else models,
        "generation": {"max_tokens": 8192, "temperature": 0, "stream": False,
                       "attempts_per_model": 2, "extra_body": settings.generation_extra_body},
    }
    fingerprint = hashlib.sha256(json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "protocol_version": 1,
        "provider": provider.name,
        "allowed_models": models,
        "default_models": defaults,
        "json_models": sorted(snapshot.json_models) if dynamic else [],
        "schema_models": sorted(snapshot.schema_models) if dynamic else [],
        "catalog_fingerprint": fingerprint,
        "limits": {"system_chars": 8000, "prompt_chars": 48000, "chain_length": 3},
    }


def resolve_model_chain(settings: Settings, model_ids: list[str] | None,
                        catalog_fingerprint: str | None,
                        catalog: dict[str, Any] | None = None) -> tuple[list[str], str | None]:
    catalog = catalog or generation_catalog(settings)
    if model_ids is None:
        return catalog["default_models"], None
    if catalog_fingerprint != catalog["catalog_fingerprint"]:
        raise GenerationConfigChangedError(
            "Generation configuration changed; reload before retrying", terminal=True
        )
    if any(model not in catalog["allowed_models"] for model in model_ids):
        if _dynamic_openrouter(settings):
            raise GenerationConfigChangedError(
                "Selected model is no longer available; reload before retrying", terminal=True
            )
        raise GenerationRequestError("model_ids contains a model outside the configured allowlist")
    return model_ids, catalog_fingerprint


async def generate_text(
    system: str, prompt: str, settings: Settings,
    response_format: dict[str, Any] | None = None, *,
    model_index: int = 0, attempt: int = 1, budget_ms: int = 300000,
    validation_feedback: str | None = None,
    model_ids: list[str] | None = None,
    catalog_fingerprint: str | None = None,
    validate: Callable[[str], Any] | None = None,
) -> GenerateResponse:
    loop = asyncio.get_running_loop()
    started = loop.time()
    # Validate internal callers as well as HTTP requests before using provider capacity.
    try:
        request = GenerateRequest(system=system, prompt=prompt, response_format=response_format,
                                  model_index=model_index, attempt=attempt, budget_ms=budget_ms,
                                  validation_feedback=validation_feedback, model_ids=model_ids,
                                  catalog_fingerprint=catalog_fingerprint)
    except ValueError as exc:
        raise GenerationRequestError("Invalid generation request") from exc
    deadline = started + request.budget_ms / 1000
    validator = _output_validator(response_format)
    provider = select_generation_provider(settings)
    snapshot = None
    if _dynamic_openrouter(settings):
        try:
            snapshot = await asyncio.wait_for(
                asyncio.to_thread(_catalog_snapshot, settings),
                timeout=max(0, deadline - loop.time()),
            )
        except TimeoutError as exc:
            raise GenerationUnavailableError("Generation batch deadline exceeded",
                                             code="GENERATION_DEADLINE_EXCEEDED", terminal=True) from exc
    catalog = generation_catalog(settings, snapshot=snapshot)
    models, selection_fingerprint = resolve_model_chain(
        settings, request.model_ids, request.catalog_fingerprint, catalog
    )
    if model_index >= len(models):
        raise GenerationRequestError("model_index is outside the configured generation chain")
    key_count = len(configured_generation_keys(settings)) if provider.name == "remote" else 1
    try:
        async with asyncio.timeout_at(deadline):
            for key_index, index in product(range(key_count), range(model_index, len(models))):
                if key_index and index == model_index:
                    logger.warning("Generation key retry: key_slot=%s", key_index + 1)
                if provider.name == "remote":
                    cooldown = _model_cooldown(settings, key_index, models[index])
                    if cooldown is not None:
                        failure = cooldown
                        continue
                    provider = OpenAICompatibleGenerationProvider(
                        settings.model_copy(update={"generation_model": models[index]}),
                        schema_supported=(models[index] in snapshot.schema_models if snapshot else None),
                        json_supported=(models[index] in snapshot.json_models if snapshot else None),
                        key_index=key_index,
                    )
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
                                                        "next_model_index": index + 1 if index + 1 < len(models) else None,
                                                        "catalog_fingerprint": selection_fingerprint})
                    except (GenerationInvalidResponseError, OllamaInvalidResponseError) as exc:
                        if isinstance(exc, GenerationError) and exc.terminal:
                            raise
                        failure = exc
                    except (GenerationUnavailableError, GenerationRateLimitError, OllamaUnavailableError) as exc:
                        logger.warning("Generation attempt failed: key_slot=%s model_index=%s attempt=%s code=%s",
                                       key_index + 1, index, current_attempt, getattr(exc, "code", "GENERATION_UNAVAILABLE"))
                        if isinstance(exc, GenerationError) and exc.terminal:
                            raise
                        failure = exc
                        break
                    logger.warning("Generation attempt failed: model_index=%s attempt=%s code=%s",
                                   index, current_attempt, getattr(failure, "code", "INVALID_GENERATION_RESPONSE"))
            if provider.name == "remote":
                cooldowns = [_model_cooldown(settings, key, models[index])
                             for key, index in product(range(key_count), range(model_index, len(models)))]
                if all(cooldown is not None for cooldown in cooldowns):
                    failure = min(cooldowns, key=lambda cooldown: cooldown.retry_after)
                elif isinstance(failure, GenerationError):
                    failure.retry_after = None
            raise failure
    except TimeoutError as exc:
        raise GenerationUnavailableError("Generation batch deadline exceeded",
                                         code="GENERATION_DEADLINE_EXCEEDED", terminal=True) from exc
