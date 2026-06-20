"""Shared test fixtures.

Every node test gets a real taxonomy + NoOp LLM + in-memory Cache — offline,
fast, deterministic.  NodeContext is built from these so nodes are tested in
isolation without touching real services.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = REPO_ROOT / "yad2_search_taxonomy.json"


@pytest.fixture(scope="session")
def taxonomy():
    from taxonomy.loader import load
    return load(TAXONOMY_PATH)


@pytest.fixture
def noop_llm():
    from llm.client import LLMResult, NoOpLLMClient, TokenUsage
    client = NoOpLLMClient()
    return client


@pytest.fixture
def mock_cache():
    from cache.cache import Cache
    return Cache(capacity=100)


@pytest.fixture
def settings():
    from config import Settings
    return Settings(openai_api_key="", llm_enabled=False, taxonomy_path=TAXONOMY_PATH)


@pytest.fixture
def ctx(taxonomy, noop_llm, mock_cache, settings):
    from observability import metrics as m
    from graph.context import NodeContext
    from patterns.library import PatternLibrary
    from patterns.normalizations import NormalizationDB
    return NodeContext(taxonomy=taxonomy, llm=noop_llm, cache=mock_cache,
                       settings=settings, metrics=m,
                       pattern_library=PatternLibrary(), normalization_db=NormalizationDB())


@pytest.fixture
def base_state():
    from graph.state import GraphState
    return GraphState()
