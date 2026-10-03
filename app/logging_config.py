import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re

from app.settings import load_settings


LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "model-service.log"


class _FileFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        output = super().format(record)
        settings = load_settings()
        for key in (settings.model_api_key, settings.generation_api_key, *settings.generation_api_keys):
            if key:
                output = output.replace(key, "<REDACTED>")
        return re.sub(r"((?:https?://|/)[^\s?]*)\?[^\s\"']+", r"\1?<REDACTED>", output)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    root = logging.getLogger()
    if any(isinstance(handler, RotatingFileHandler) and handler.baseFilename == str(LOG_PATH)
           for handler in root.handlers):
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(_FileFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    # Uvicorn's default loggers do not propagate to root; keep their console handlers.
    logging.getLogger("uvicorn").addHandler(handler)
    logging.getLogger("uvicorn.access").addHandler(handler)
