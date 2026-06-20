"""Build the semantic taxonomy FAISS index.

Embeds every canonical taxonomy value with text-embedding-3-small and writes the
two artifacts the service loads at startup:
    taxonomy.faiss       — FAISS index (repo root)
    taxonomy.meta.json   — parallel metadata (repo root)

Lives in the taxonomy package because it is the offline counterpart to
semantic_index.py — it produces exactly what SemanticTaxonomyIndex.load() reads.

Run once, or whenever yad2_search_taxonomy.json changes, from the repo root:
    python -m taxonomy.build_index      # with src on PYTHONPATH
    python src/taxonomy/build_index.py  # standalone

Cost: ~$0.00002 per 1k tokens — a full taxonomy build is typically < $0.01.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Repo root is three levels up: src/taxonomy/build_index.py → src/taxonomy → src → root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

_BATCH_SIZE = 100
_EMBED_MODEL = "text-embedding-3-small"


async def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    import openai

    from config import Settings
    from taxonomy.loader import load as load_taxonomy
    from taxonomy.semantic_index import (
        DEFAULT_INDEX_PATH, DEFAULT_META_PATH, SemanticTaxonomyIndex, collect_entries,
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set — cannot embed taxonomy values.")
        sys.exit(1)

    settings = Settings(taxonomy_path=REPO_ROOT / "yad2_search_taxonomy.json")

    print("Loading taxonomy...")
    tax = load_taxonomy(settings.taxonomy_path)
    entries = collect_entries(tax)
    print(f"Collected {len(entries)} unique taxonomy values to embed.")

    print("Embedding (this calls the OpenAI embeddings API)...")
    client = openai.AsyncOpenAI(api_key=api_key)
    embeddings: list[list[float]] = []
    for i in range(0, len(entries), _BATCH_SIZE):
        batch = entries[i : i + _BATCH_SIZE]
        response = await client.embeddings.create(
            input=[e["value"] for e in batch],
            model=_EMBED_MODEL,
        )
        embeddings.extend(d.embedding for d in sorted(response.data, key=lambda x: x.index))
        print(f"  embedded {min(i + _BATCH_SIZE, len(entries))}/{len(entries)}")

    print("Building FAISS index...")
    index = SemanticTaxonomyIndex.build(entries, embeddings)
    index.save(DEFAULT_INDEX_PATH, DEFAULT_META_PATH)
    print(f"Done. Index size: {index.size} entries")
    print(f"  {DEFAULT_INDEX_PATH}")
    print(f"  {DEFAULT_META_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
