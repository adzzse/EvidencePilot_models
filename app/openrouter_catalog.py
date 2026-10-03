import threading
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import httpx


_FRESH_SECONDS = 300
_STALE_SECONDS = 3600


class CatalogUnavailableError(RuntimeError):
    pass


class CatalogAuthenticationError(CatalogUnavailableError):
    pass


class CatalogNoModelsError(CatalogUnavailableError):
    pass


@dataclass(frozen=True)
class CatalogSnapshot:
    allowed_models: tuple[str, ...]
    schema_models: frozenset[str]
    default_models: tuple[str, ...]
    json_models: frozenset[str] = frozenset()


def _zero(value) -> bool:
    try:
        return Decimal(str(value)) == 0
    except (InvalidOperation, TypeError, ValueError):
        return False


def _free(model: object) -> bool:
    if not isinstance(model, dict) or not isinstance(model.get("id"), str):
        return False
    model_id = model["id"]
    pricing = model.get("pricing")
    return (bool(model_id) and model_id.strip() == model_id and len(model_id) <= 255
            and isinstance(pricing, dict) and _zero(pricing.get("prompt"))
            and _zero(pricing.get("completion")) and _zero(pricing.get("request") or 0))


def _eligible(model: object) -> tuple[str, bool, int, int] | None:
    if not isinstance(model, dict):
        return None
    model_id = model.get("id")
    pricing = model.get("pricing")
    architecture = model.get("architecture")
    provider = model.get("top_provider")
    parameters = model.get("supported_parameters")
    context = model.get("context_length")
    if (not isinstance(model_id, str) or not model_id.endswith(":free")
            or not model_id or model_id.strip() != model_id or len(model_id) > 255
            or not isinstance(pricing, dict) or not isinstance(architecture, dict)
            or not isinstance(provider, dict) or not isinstance(parameters, list)
            or not {"response_format", "temperature", "max_tokens"}.issubset(parameters)
            or not _zero(pricing.get("prompt")) or not _zero(pricing.get("completion"))
            or not _zero(pricing.get("request") or 0)
            or "text" not in (architecture.get("input_modalities") or [])
            or "text" not in (architecture.get("output_modalities") or [])
            or not isinstance(context, int) or context < 65536):
        return None
    completion = provider.get("max_completion_tokens")
    if not isinstance(completion, int) or completion < 8192:
        return None
    return model_id, "structured_outputs" in parameters, context, completion


def _fetch(base_url: str, keys: tuple[str, ...]) -> CatalogSnapshot:
    catalogs = []
    try:
        with httpx.Client(timeout=5.0) as client:
            for key in keys:
                response = client.get(f"{base_url}/models/user",
                                      headers={"Authorization": f"Bearer {key}"})
                if response.status_code in (401, 403):
                    raise CatalogAuthenticationError("OpenRouter catalog access rejected")
                response.raise_for_status()
                models = response.json()["data"]
                if not isinstance(models, list):
                    raise ValueError("invalid catalog")
                catalogs.append({model["id"]: model for model in models if _free(model)})
    except CatalogAuthenticationError:
        raise
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise CatalogUnavailableError("OpenRouter model catalog is unavailable") from None
    common = set.intersection(*(set(catalog) for catalog in catalogs))
    if not common:
        raise CatalogNoModelsError("No free OpenRouter models are available")
    json_models = frozenset(model_id for model_id in common if all(
        "response_format" in (catalog[model_id].get("supported_parameters") or [])
        for catalog in catalogs
    ))
    schema = frozenset(model_id for model_id in json_models if all(
        "structured_outputs" in (catalog[model_id].get("supported_parameters") or [])
        for catalog in catalogs
    ))
    eligible = {}
    for model_id in common:
        details = tuple(_eligible(catalog[model_id]) for catalog in catalogs)
        if all(details):
            eligible[model_id] = details
    ranked = sorted(eligible, key=lambda model_id: (
        -min(item[2] for item in eligible[model_id]),
        model_id not in schema,
        -min(item[3] for item in eligible[model_id]),
        model_id,
    ))
    defaults = [model_id for model_id in ranked if "preview" not in model_id.casefold()]
    fallback = sorted(json_models) or sorted(common)
    return CatalogSnapshot(tuple(sorted(common)), schema, tuple((defaults or ranked or fallback)[:3]),
                           json_models)


class _CatalogCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.key = None
        self.snapshot = None
        self.fetched_at = 0.0

    def get(self, base_url: str, keys: tuple[str, ...]) -> CatalogSnapshot:
        cache_key = (base_url, keys)
        now = time.monotonic()
        with self.lock:
            if self.key == cache_key and self.snapshot and now - self.fetched_at < _FRESH_SECONDS:
                return self.snapshot
            stale = (self.snapshot if self.key == cache_key and self.snapshot
                     and now - self.fetched_at < _STALE_SECONDS else None)
        try:
            snapshot = _fetch(base_url, keys)
        except (CatalogAuthenticationError, CatalogNoModelsError):
            raise
        except CatalogUnavailableError:
            if stale:
                return stale
            raise
        with self.lock:
            self.key = cache_key
            self.snapshot = snapshot
            self.fetched_at = time.monotonic()
        return snapshot


_cache = _CatalogCache()


def get_catalog(base_url: str, keys: tuple[str, ...]) -> CatalogSnapshot:
    return _cache.get(base_url, keys)
