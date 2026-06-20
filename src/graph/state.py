"""Graph state — the typed contract that flows through every node.

Using Pydantic with extra="forbid" means any node returning an unexpected key
raises immediately at state-merge time, making bugs visible during development.

Each field has a clear owner:
  raw_q           → set by the API, never mutated
  clean_q         → sanitize node, then rewritten by normalize
  security_flags  → sanitize node
  query_words     → normalize node
  category        → extract node (inferred from the segments found)
  params          → extract node, then schema-filtered by validate
  cache_hit       → cache_lookup node
  llm_used        → extract node
  confidence      → validate node (final value)
  errors          → any node can append
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GraphState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ── Input ──────────────────────────────────────────────────────────────
    raw_q: str = ""

    # ── After sanitize ─────────────────────────────────────────────────────
    clean_q: str = ""
    security_flags: list[str] = Field(default_factory=list)

    # ── After security_check ───────────────────────────────────────────────
    injection_confirmed: bool = False   # True → graph exits, API returns 400

    # ── After normalize ────────────────────────────────────────────────────
    query_words: list[str] = Field(default_factory=list)
    normalization_applied: dict[str, str] = Field(default_factory=dict)  # {original_word: corrected_word}

    # ── After cache_lookup ─────────────────────────────────────────────────
    cache_hit: bool = False
    cache_key: str = ""

    # ── After extract (category is inferred from the segments) ──────────────
    category: str = ""               # "נדל״ן" | "רכב" | "יד_שנייה"
    classification_certainty: float = 0.0   # winner dominance in [0,1] (feeds confidence)

    # ── After extract ──────────────────────────────────────────────────────
    params: dict[str, Any] = Field(default_factory=dict)
    segments: list[dict[str, Any]] = Field(default_factory=list)  # resolved meaningful chunks
    extraction_coverage: float = 0.0
    llm_used: bool = False
    taxonomy_hints: dict[str, list[str]] = Field(default_factory=dict)  # FAISS hits sent to LLM

    # ── After validate ─────────────────────────────────────────────────────
    confidence: float = 0.0
    notes: list[str] = Field(default_factory=list)

    # ── Errors (any node) ──────────────────────────────────────────────────
    errors: list[str] = Field(default_factory=list)
