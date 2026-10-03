import asyncio
import io
import json
import logging
from logging.handlers import RotatingFileHandler

import httpx
import pytest

from app import generation, limits, logging_config
from app.settings import Settings


@pytest.fixture
def file_logging(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "model-service.log"
    monkeypatch.setattr(logging_config, "LOG_PATH", log_path)
    monkeypatch.setattr(logging_config, "load_settings", lambda: Settings(
        generation_api_keys=("provider-secret-a", "provider-secret-b"), model_api_key="worker-secret",
    ))
    loggers = [logging.getLogger(name) for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access")]
    original = [(logger, list(logger.handlers), logger.level, logger.propagate) for logger in loggers]
    formatters = {handler: handler.formatter for logger in loggers for handler in logger.handlers}
    for logger in loggers:
        logger.setLevel(logging.INFO)
    logging.getLogger("uvicorn").propagate = False
    logging.getLogger("uvicorn.error").propagate = True
    logging.getLogger("uvicorn.access").propagate = False
    logging_config.configure_logging()
    yield log_path
    added = set()
    for logger, handlers, level, propagate in original:
        added.update(handler for handler in logger.handlers if handler not in handlers)
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate
    for handler in added:
        handler.close()
    for handler, formatter in formatters.items():
        handler.setFormatter(formatter)


@pytest.fixture
def console_logging(file_logging):
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    for name in ("", "uvicorn", "uvicorn.access"):
        logging.getLogger(name).addHandler(handler)
    logging_config.configure_logging()
    return output


def test_file_keeps_generation_and_uvicorn_logs_without_duplicates(file_logging):
    logging_config.configure_logging()
    logging.getLogger("app.generation").warning("Generation attempt failed: model_index=0 code=GENERATION_UNAVAILABLE")
    logging.getLogger("uvicorn.error").info("Worker startup verification")
    logging.getLogger("uvicorn.access").info('127.0.0.1 - "GET /health HTTP/1.1" 200')
    output = file_logging.read_text(encoding="utf-8")
    assert output.count("Generation attempt failed") == 1
    assert output.count("Worker startup verification") == 1
    assert output.count("GET /health") == 1
    assert "WARNING app.generation" in output and "INFO uvicorn.access" in output
    assert output[:4].isdigit() and "private-prompt" not in output


@pytest.mark.parametrize("logger_name", ["app.generation", "uvicorn.error"])
def test_console_and_file_summarize_errors_and_redact_secrets(file_logging, console_logging, logger_name):
    try:
        raise RuntimeError("provider-secret-a provider-secret-b worker-secret https://storage.test/file?signature=private-query")
    except RuntimeError:
        logging.getLogger(logger_name).exception("Provider call failed")
    logging.getLogger("uvicorn.access").info('GET /health?token=private-access-query HTTP/1.1 200')
    for output in (file_logging.read_text(encoding="utf-8"), console_logging.getvalue()):
        assert all(secret not in output for secret in (
            "provider-secret-a", "provider-secret-b", "worker-secret", "private-query", "private-access-query",
        ))
        assert "Traceback" not in output and "RuntimeError" in output
        assert output.count("Provider call failed") == 1
        assert len(output.splitlines()) == 2
        assert "https://storage.test/file?<REDACTED>" in output
        assert "/health?<REDACTED>" in output


@pytest.mark.parametrize("failure", [
    "attempt_timeout", httpx.ReadTimeout("provider-secret-a private-prompt"),
    httpx.ConnectError("provider-secret-a private-prompt"), httpx.Response(429), httpx.Response(503),
])
def test_provider_fallback_errors_are_clear_in_console_and_file(file_logging, console_logging, monkeypatch, failure):
    calls = []
    real_client = httpx.AsyncClient

    async def respond(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "primary":
            if failure == "attempt_timeout":
                await asyncio.sleep(1)
            if isinstance(failure, Exception):
                raise failure
            return failure
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]})

    monkeypatch.setattr(generation.httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setattr(generation, "ATTEMPT_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(generation, "_model_cooldowns", {})
    monkeypatch.setattr(limits, "generation_gate", limits.ModelCallGate(4))
    settings = Settings(generation_provider="remote", generation_api_key="provider-secret-a",
                        generation_base_url="https://openrouter.ai/api/v1",
                        generation_model="primary", generation_fallback_models=["fallback"])
    result = asyncio.run(generation.generate_text("private-system", "private-prompt", settings, budget_ms=1000))
    assert result.done and calls == ["primary", "fallback"]
    for output in (file_logging.read_text(encoding="utf-8"), console_logging.getvalue()):
        assert "model=primary" in output and "key_slot=1" in output and "action=try_next_model" in output
        assert all(value not in output for value in (
            "Traceback", "CancelledError", "provider-secret-a", "private-system", "private-prompt",
        ))


def test_file_rotation_keeps_bounded_history(file_logging):
    handler = next(handler for handler in logging.getLogger().handlers
                   if isinstance(handler, RotatingFileHandler) and handler.baseFilename == str(file_logging))
    assert handler.maxBytes == 10 * 1024 * 1024 and handler.backupCount == 5
    handler.maxBytes = 250
    handler.backupCount = 2
    for index in range(20):
        logging.getLogger("app.generation").warning("Rotation entry %s %s", index, "x" * 60)
    files = list(file_logging.parent.glob("model-service.log*"))
    assert len(files) == 3
    assert "Rotation entry 19" in file_logging.read_text(encoding="utf-8")
