from copy import copy
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re

from app.settings import load_settings


LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "model-service.log"


class _LogFormatter(logging.Formatter):
    def formatException(self, exc_info) -> str:
        return f"{exc_info[0].__name__}: {exc_info[1]}"

    def format(self, record: logging.LogRecord) -> str:
        record = copy(record)
        record.exc_text = None
        record.stack_info = None
        output = super().format(record)
        settings = load_settings()
        for key in (settings.model_api_key, settings.generation_api_key, *settings.generation_api_keys):
            if key:
                output = output.replace(key, "<REDACTED>")
        output = re.sub(r"((?:https?://|/)[^\s?]*)\?[^\s\"']+", r"\1?<REDACTED>", output)
        return " | ".join(output.splitlines())


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    root = logging.getLogger()
    formatter = _LogFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        for existing in logging.getLogger(name).handlers:
            if isinstance(existing, logging.StreamHandler) and not isinstance(existing, logging.FileHandler):
                existing.setFormatter(formatter)
    if any(isinstance(handler, RotatingFileHandler) and handler.baseFilename == str(LOG_PATH)
           for handler in root.handlers):
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    # Uvicorn's default loggers do not propagate to root; keep their console handlers.
    logging.getLogger("uvicorn").addHandler(handler)
    logging.getLogger("uvicorn.access").addHandler(handler)
