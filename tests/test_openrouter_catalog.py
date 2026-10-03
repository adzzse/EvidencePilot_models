import httpx
import pytest

from app import openrouter_catalog


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(openrouter_catalog, "_cache", openrouter_catalog._CatalogCache())


def model(model_id, *, schema=True, price="0", response_format=True):
    parameters = ["temperature", "max_tokens"]
    if response_format:
        parameters.append("response_format")
    if schema:
        parameters.append("structured_outputs")
    return {
        "id": model_id,
        "pricing": {"prompt": price, "completion": price},
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "context_length": 65536,
        "top_provider": {"max_completion_tokens": 8192},
        "supported_parameters": parameters,
    }


def mock_catalog(monkeypatch, responses):
    calls = []
    real_client = httpx.Client
    responses = iter(responses)

    def respond(request):
        calls.append(request)
        response = next(responses)
        return response if isinstance(response, httpx.Response) else httpx.Response(200, json=response)

    monkeypatch.setattr(openrouter_catalog.httpx, "Client",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    return calls


def test_catalog_lists_all_free_models_and_marks_json_support(monkeypatch):
    keys = ("secret-a", "secret-b", "secret-c")
    common = model("provider/common:free")
    json_only = model("provider/json-only:free", schema=False)
    missing = model("provider/missing:free")
    preview = model("provider/preview-model:free")
    paid = model("provider/paid", price="1")
    plain = model("provider/plain:free", response_format=False)
    free_without_suffix = model("provider/promo")
    short_context = model("provider/short:free")
    short_context["context_length"] = 8192
    calls = mock_catalog(monkeypatch, [
        {"data": [common, json_only, missing, preview, paid, plain, free_without_suffix, short_context]},
        {"data": [common, json_only, missing, preview, plain, free_without_suffix, short_context]},
        {"data": [common, json_only, preview, plain, free_without_suffix, short_context]},
    ])

    snapshot = openrouter_catalog.get_catalog("https://openrouter.ai/api/v1", keys)

    assert snapshot.allowed_models == ("provider/common:free", "provider/json-only:free",
                                       "provider/plain:free", "provider/preview-model:free",
                                       "provider/promo", "provider/short:free")
    assert snapshot.json_models == frozenset({"provider/common:free", "provider/json-only:free",
                                              "provider/preview-model:free", "provider/promo",
                                              "provider/short:free"})
    assert snapshot.schema_models == frozenset({"provider/common:free", "provider/preview-model:free",
                                                "provider/promo", "provider/short:free"})
    assert snapshot.default_models == ("provider/common:free", "provider/json-only:free")
    assert len(calls) == 3
    assert all(key not in str(snapshot) for key in keys)
    assert [request.headers["Authorization"] for request in calls] == [
        "Bearer secret-a", "Bearer secret-b", "Bearer secret-c",
    ]


def test_catalog_cache_reuses_snapshot_during_short_outage(monkeypatch):
    keys = ("secret-a",)
    now = [1000.0]
    monkeypatch.setattr(openrouter_catalog.time, "monotonic", lambda: now[0])
    calls = mock_catalog(monkeypatch, [
        {"data": [model("provider/common:free")]},
        httpx.Response(503, json={"error": {"message": "unavailable"}}),
        httpx.Response(503, json={"error": {"message": "unavailable"}}),
    ])

    first = openrouter_catalog.get_catalog("https://openrouter.ai/api/v1", keys)
    assert openrouter_catalog.get_catalog("https://openrouter.ai/api/v1", keys) is first
    assert len(calls) == 1
    now[0] += 301
    assert openrouter_catalog.get_catalog("https://openrouter.ai/api/v1", keys) is first
    now[0] += 3600
    with pytest.raises(openrouter_catalog.CatalogUnavailableError) as failure:
        openrouter_catalog.get_catalog("https://openrouter.ai/api/v1", keys)
    assert len(calls) == 3
    assert "secret-a" not in str(failure.value)


def test_catalog_authentication_failure_never_uses_stale_snapshot(monkeypatch):
    keys = ("secret-a",)
    now = [1000.0]
    monkeypatch.setattr(openrouter_catalog.time, "monotonic", lambda: now[0])
    mock_catalog(monkeypatch, [
        {"data": [model("provider/common:free")]},
        httpx.Response(401, json={"error": {"message": "secret-a"}}),
    ])
    openrouter_catalog.get_catalog("https://openrouter.ai/api/v1", keys)
    now[0] += 301
    with pytest.raises(openrouter_catalog.CatalogAuthenticationError) as failure:
        openrouter_catalog.get_catalog("https://openrouter.ai/api/v1", keys)
    assert "secret-a" not in str(failure.value)
