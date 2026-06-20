"""Build the semantic taxonomy FAISS index.

Embeds every canonical taxonomy value with text-embedding-3-small and saves:
    taxonomy.faiss       — FAISS index (project root)
    taxonomy.meta.json   — parallel metadata (project root)

Run once (or whenever the taxonomy JSON changes):
    python scripts/build_taxonomy_index.py

Cost: ~$0.00002 per 1k tokens — a full taxonomy build is typically < $0.01.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


async def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    from taxonomy.loader import load as load_taxonomy
    from taxonomy.semantic_index import (
        DEFAULT_INDEX_PATH, DEFAULT_META_PATH, SemanticTaxonomyIndex, collect_entries,
    )
    from llm.client import create_llm_client
    from config import Settings

    settings = Settings(taxonomy_path=REPO_ROOT / "yad2_search_taxonomy.json")
    llm = create_llm_client(settings)

    if not llm.is_available():
        print("ERROR: OPENAI_API_KEY not set — cannot embed taxonomy values.")
        sys.exit(1)

    print("Loading taxonomy...")
    tax = load_taxonomy(settings.taxonomy_path)
    entries = collect_entries(tax)
    print(f"Collected {len(entries)} unique taxonomy values to embed.")

    print("Embedding (this calls the OpenAI embeddings API)...")
    embeddings: list[list[float]] = []
    batch_size = 100

    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        texts = [e["value"] for e in batch]

        # Call embeddings API for the whole batch.
        import openai
        from dotenv import load_dotenv
        import os
        load_dotenv(REPO_ROOT / ".env")
        client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        response = await client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
        )
        batch_embeddings = [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
        embeddings.extend(batch_embeddings)
        print(f"  embedded {min(i + batch_size, len(entries))}/{len(entries)}")

    print("Building FAISS index...")
    index = SemanticTaxonomyIndex.build(entries, embeddings)
    index.save(DEFAULT_INDEX_PATH, DEFAULT_META_PATH)
    print(f"Done. Index size: {index.size} entries")
    print(f"  {DEFAULT_INDEX_PATH}")
    print(f"  {DEFAULT_META_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
