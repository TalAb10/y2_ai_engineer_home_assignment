"""NodeContext — dependency container injected into every node.

Built once at startup in main.py and passed to all nodes via the closure in
graph/build.py.  Nodes call ctx.llm, ctx.cache, etc. — never constructing
services themselves — so they stay unit-testable with simple fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cache.cache import Cache
from config import Settings
from observability import metrics as m
from patterns.library import PatternLibrary
from patterns.normalizations import NormalizationDB
from taxonomy.loader import TaxonomyIndex


@dataclass
class NodeContext:
    taxonomy: TaxonomyIndex
    llm: Any          # LLMClient | NoOpLLMClient — typed loosely to avoid circular import
    cache: Cache
    settings: Settings
    metrics: type[m]
    pattern_library: PatternLibrary     # learned query shapes → segment types
    normalization_db: NormalizationDB   # LLM-learned typo corrections
    semantic_index: Any = None          # SemanticTaxonomyIndex | None — optional, loaded from disk
