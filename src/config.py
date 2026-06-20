"""Environment-driven settings — the single place all tunables live.

Nodes and services receive a Settings instance (or values from it) via NodeContext;
they never import this module directly, keeping them testable with fake values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="Set via OPENAI_API_KEY env var.")
    llm_enabled: bool = Field(default=True, description="False → rules-only (offline/test mode).")
    llm_model: str = Field(default="gpt-5-mini")
    llm_timeout_s: float = Field(default=20.0)
    llm_max_retries: int = Field(default=1)

    # ── Pricing (USD / 1M tokens) — used for live cost tracking ────────────
    price_in: float = Field(default=0.25)
    price_out: float = Field(default=2.00)

    # ── Extraction thresholds ──────────────────────────────────────────────
    pattern_coverage_threshold: float = Field(
        default=0.95,
        description="Min fraction of the query covered by patterns/rules before calling the LLM.",
    )

    # ── Input hygiene ──────────────────────────────────────────────────────
    max_input_chars: int = Field(default=512)

    # ── Cache ──────────────────────────────────────────────────────────────
    cache_size: int = Field(default=10_000, description="In-process LRU capacity.")

    # ── Data ───────────────────────────────────────────────────────────────
    taxonomy_path: Path = Field(default=_REPO_ROOT / "yad2_search_taxonomy.json")

    # ── Observability ──────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    """Process-singleton accessor — re-reads env only on first call."""
    return Settings()
