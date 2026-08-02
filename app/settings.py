import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel


class Settings(BaseModel):
    generation_provider: str = "auto"
    openai_compatible_api_key: str = ""
    openai_compatible_base_url: str = "https://opencode.ai/zen/v1"
    openai_compatible_model: str = "deepseek-v4-flash-free"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:9b"
    ollama_embedding_model: str = "nomic-embed-text"
    model_api_key: str = ""
    extraction_allowed_hosts: tuple[str, ...] = ()
    mineru_command: str = "mineru"
    mineru_backend: str = "pipeline"
    mineru_timeout_seconds: int = 600
    max_download_bytes: int = 52 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        hosts = tuple(
            host.strip().lower()
            for host in os.getenv("EXTRACTION_ALLOWED_HOSTS", "").split(",")
            if host.strip()
        )
        return cls(
            generation_provider=os.getenv("GENERATION_PROVIDER", "auto").strip().lower(),
            openai_compatible_api_key=os.getenv(
                "OPENAI_COMPATIBLE_API_KEY", ""
            ).strip(),
            openai_compatible_base_url=os.getenv(
                "OPENAI_COMPATIBLE_BASE_URL",
                "https://opencode.ai/zen/v1",
            ).rstrip("/"),
            openai_compatible_model=os.getenv(
                "OPENAI_COMPATIBLE_MODEL",
                "deepseek-v4-flash-free",
            ).strip(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3.5:9b"),
            ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            model_api_key=os.getenv("MODEL_API_KEY", ""),
            extraction_allowed_hosts=hosts,
            mineru_command=os.getenv("MINERU_COMMAND", "mineru"),
            mineru_backend=os.getenv("MINERU_BACKEND", "pipeline"),
            mineru_timeout_seconds=int(os.getenv("MINERU_TIMEOUT_SECONDS", "600")),
        )


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings.from_env()
