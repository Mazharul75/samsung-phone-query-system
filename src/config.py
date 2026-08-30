"""Central configuration for the Samsung Phone Query and Review System.

Every tunable value lives here so the scraper, database layer, RAG pipeline,
agents and API all read from a single source of truth.  Values are loaded from
environment variables (optionally via a local ``.env`` file), which keeps
credentials out of the source tree.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: <root>/src/config.py -> parents[1] is <root>
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_STORE_DIR = DATA_DIR / "vectorstore"
MODEL_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"


class Settings(BaseSettings):
    """Application settings, overridable through environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Database (MySQL / MariaDB)
    # ------------------------------------------------------------------
    db_host: str = Field(default="127.0.0.1")
    db_port: int = Field(default=3307)
    db_user: str = Field(default="root")
    db_password: str = Field(default="")
    db_name: str = Field(default="samsung_phones")
    db_echo: bool = Field(default=False)

    # ------------------------------------------------------------------
    # Scraper
    # ------------------------------------------------------------------
    scraper_base_url: str = Field(default="https://www.gsmarena.com")
    scraper_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    )
    #: Seconds to wait between two consecutive requests (politeness).
    scraper_delay_seconds: float = Field(default=2.0)
    scraper_timeout_seconds: int = Field(default=30)
    scraper_max_retries: int = Field(default=4)
    #: How many brand-listing pages to walk while resolving target models.
    scraper_max_listing_pages: int = Field(default=15)
    #: Cache downloaded HTML so re-parsing never re-hits the network.
    scraper_use_cache: bool = Field(default=True)

    # ------------------------------------------------------------------
    # RAG / models (all open source, run locally on CPU)
    # ------------------------------------------------------------------
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    llm_model: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct")
    llm_max_new_tokens: int = Field(default=512)
    llm_temperature: float = Field(default=0.3)
    #: Number of documents pulled from the vector store per query.
    retrieval_top_k: int = Field(default=6)
    #: Weight of dense (embedding) scores in the hybrid retriever; the
    #: remainder is given to the BM25 lexical scores.
    retrieval_dense_weight: float = Field(default=0.6)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """SQLAlchemy connection URL for the application database."""
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def server_url(self) -> str:
        """Connection URL without a database, used to run ``CREATE DATABASE``."""
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/?charset=utf8mb4"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()

# Make sure the directories the application writes to always exist.
for _directory in (DATA_DIR, RAW_DATA_DIR, VECTOR_STORE_DIR, MODEL_CACHE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# Keep the Hugging Face model cache inside the project unless the user has
# already chosen a location.  The models are several gigabytes, and defaulting
# to the home directory silently fills the system drive.
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
