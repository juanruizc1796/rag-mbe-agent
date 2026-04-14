"""
Central configuration module.
Loads all settings from environment variables via .env file.
"""

import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "RAG MBE Agent"
    APP_ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    API_PORT: int = Field(default=8000)

    # ── Ollama / LLM ─────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = Field(default="http://ollama:11434")
    OLLAMA_MODEL: str = Field(default="llama3.1:8b")
    OLLAMA_TIMEOUT: int = Field(default=120)
    OLLAMA_MAX_RETRIES: int = Field(default=3)
    OLLAMA_RETRY_DELAY: float = Field(default=2.0)

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = Field(default="dmis-lab/biobert-base-cased-v1.2")
    EMBEDDING_DEVICE: str = Field(default="cpu")
    EMBEDDING_BATCH_SIZE: int = Field(default=32)

    # ── FAISS ─────────────────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = Field(default="/app/data/faiss_index")
    FAISS_TOP_K: int = Field(default=5)
    FAISS_SIMILARITY_THRESHOLD: float = Field(default=0.30)

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    POSTGRES_HOST: str = Field(default="postgres")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="mbe_agent")
    POSTGRES_USER: str = Field(default="mbe_user")
    POSTGRES_PASSWORD: str = Field(default="mbe_pass")
    DB_MAX_RETRIES: int = Field(default=5)
    DB_RETRY_DELAY: float = Field(default=3.0)
    DB_POOL_SIZE: int = Field(default=5)
    DB_MAX_OVERFLOW: int = Field(default=10)

    # ── Memory ────────────────────────────────────────────────────────────────
    SHORT_TERM_HISTORY_N: int = Field(default=10)

    # ── Agent ─────────────────────────────────────────────────────────────────
    DEFAULT_LANGUAGE: str = Field(default="es")
    MBE_CLASSIFIER_THRESHOLD: float = Field(default=0.6)
    MAX_RETRIES_AGENT: int = Field(default=3)

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn_sync(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
