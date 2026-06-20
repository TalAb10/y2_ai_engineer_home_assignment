# Architecture

## Overview

The service is a stateless FastAPI application that wraps a LangGraph processing pipeline. Every request to `POST /parse` flows through a fixed sequence of nodes; each node does one thing and passes a typed state object to the next.

The core design principle is **pattern-first, LLM-last**: deterministic rules handle the majority of queries entirely in-process with no network call. The LLM is only invoked to fill gaps that rules cannot cover, and the result is learned so future identical queries skip the LLM.

The reason is latency: an LLM call cannot be made reliably fast — it is bounded by the API round-trip (network + time-to-first-token + generation), and TTFT alone is variable and can exceed the latency target under load. Rather than trying to make a single call fast, the design keeps the slow LLM path **rare**, so the latency target is met on the common (rules/cache) path by *avoiding* the LLM, not by speeding it up. Cost follows the same logic — every query the rules resolve is a query that costs zero tokens.

---

## Pipeline

```
POST /parse
     │
     ▼
┌──────────────────────────────────────────────────────────────────┐
│ sanitize                                                         │
│   NFKC normalise → strip emojis/control/RTL-overrides →          │
│   truncate to 512 chars → scan injection markers                 │
└──────────────────────┬───────────────────────────────────────────┘
                       │ security_flags non-empty?
                ┌──────▼──────┐
                │  yes        │  no
                ▼             │
      ┌─────────────────┐     │
      │ security_check  │     │
      │  LLM binary     │     │
      │  classifier:    │     │
      │  legitimate?    │     │
      └────┬────────────┘     │
           │ injection        │
           ▼                  │
          END (400)           │
                              ▼
              ┌───────────────────────────┐
              │ normalize                 │
              │  apply typo_map → resolve │
              │  unit aliases → tokenise  │
              └──────────────┬────────────┘
                             ▼
              ┌───────────────────────────┐
              │ cache_lookup              │──── HIT ──▶ END
              │  SHA-256(NFKC(query))     │
              └──────────────┬────────────┘
                           MISS
                             ▼
              ┌───────────────────────────┐
              │ extract                   │
              │  1. taxonomy span lookup  │
              │  2. learned PatternLibrary│
              │  3. deterministic numerics│
              │  4. LLM gap-fill (if      │
              │     coverage < 90%)       │
              │  5. learn new segments    │
              └──────────────┬────────────┘
                             ▼
              ┌───────────────────────────┐
              │ validate                  │
              │  Pydantic schema per      │
              │  vertical (extra=forbid)  │
              │  compute confidence       │
              └──────────────┬────────────┘
                             ▼
              ┌───────────────────────────┐
              │ cache_store               │
              └──────────────┬────────────┘
                             ▼
                            END
```

---

## Components

### sanitize
Stateless. Pure string transformation: NFKC → strip control chars → truncate → regex-scan for injection markers. Sets `security_flags` if any marker matched. No network call.

The scanner checks for four categories of injection signal:
- `instruction_override` — phrases like *"ignore previous instructions"*, *"התעלם"*
- `role_injection` — phrases like *"you are now"*, *"act as"*, *"pretend"*
- `prompt_extraction` — phrases like *"print your system prompt"*, *"reveal"*
- `delimiter_injection` — structural tokens like `<system>`, `[INST]`, `<|...|>`

A match does **not** immediately block the query — it raises a flag that triggers the `security_check` LLM deepcheck. This two-step design avoids blocking legitimate queries that happen to contain a flagged word (e.g. a furniture brand named "Pretend").

### security_check
Only runs when `security_flags` is non-empty. Calls the LLM with a fixed binary classification prompt: *is this a legitimate search or an injection attempt?* A flagged query is let through **only** when the LLM explicitly judges it legitimate; whenever the deepcheck cannot clear it — the LLM is unavailable, errors, or refuses — the node fails closed and blocks. This matches the classifier's own rule (false positives acceptable, false negatives not). On a block, the state is routed to END and the caller returns 400.

### normalize
Applies typo corrections and unit aliases, then tokenises the query into word spans for coverage tracking.

Typo correction merges two sources at runtime: the static `typo_map` from the taxonomy JSON (curated, shipped with the repo) and the dynamic `NormalizationDB` (corrections the LLM has learned from previous requests). Learned entries win on conflict — they reflect real misspellings the LLM caught that weren't in the original taxonomy. The merged map is applied word-by-word with Hebrew clitic-prefix awareness (`ב/ל/מ/ה/ו/ש/כ` attach to words, so `"בירושליים"` is corrected to `"בירושלים"`).

Unit normalisation collapses variants before pattern matching: `מ׳/מטר/מטרים → מ״ר`, `קמ/ק"מ → ק״מ`, `שקל/₪ → שח`, `מליון → 1000000`, `X אלף → X000`.

### cache_lookup
Computes `SHA-256(NFKC(clean_q))`. On a hit, populates all result fields from the cached value and routes directly to END — extract, validate, and cache_store are skipped entirely.

### extract
The main extraction node. Two parallel tracks:

1. **Non-numeric segments** — taxonomy span lookup + learned `PatternLibrary` scan. Segments are typed (city, manufacturer, model, amenity, condition, …). Overlapping spans are merged by priority.
2. **Numeric fields** — deterministic regex for price, rooms, area, floor, km, year, storage. Year is only read in a vehicle context and never when preceded by a price operator (`עד 9000` stays a price).

If pattern coverage of the query is below the `PATTERN_COVERAGE_THRESHOLD` (default 0.90), the LLM is called for the uncovered portions only. The LLM receives:
- The full query for context
- Already-identified parts annotated inline as `[text](type)`
- Only the uncovered gap text to label
- FAISS semantic search suggestions (nearest canonical taxonomy values)

The LLM returns typed segment labels and optional typo corrections. Each labelled segment is kept only if its value-extractor resolves to a real taxonomy value — the same `valid_types` gate the deterministic track applies. An unresolvable enum label (e.g. the LLM tagging `במבצע`/"on sale" as a real-estate `transaction_mode`, which has no matching taxonomy value) is dropped, so it cannot cast a category vote while contributing no value. Free-text types (city, model, brand, condition) always resolve. Surviving segments are written into the in-process `PatternLibrary` so the same surface form skips the LLM on future requests.

### validate
Runs the extracted params through the per-vertical Pydantic model. `extra="forbid"` means any key not explicitly declared in the schema causes a validation error — so even if the LLM returns a hallucinated field, it is structurally impossible for it to appear in the API response. Enum membership, numeric bounds, and range consistency are also enforced here.

Computes the final `confidence` score from two signals:

```
confidence = 0.4 × classification_certainty + 0.6 × extraction_coverage
```

- `classification_certainty` — how dominant the winning vertical was, as a ratio in [0, 1]: `top_votes / (top_votes + runner_up_votes)`. An uncontested winner → 1.0; a near-tie → ~0.5.
- `extraction_coverage` — the fraction of the query's characters that were matched by rules or the LLM. A high coverage means most of the query was understood.

### cache_store
Writes the validated result to the LRU cache under the key computed by cache_lookup.

---

## Self-Learning Pattern Library

The `PatternLibrary` (in-process, per-instance) maps raw surface forms to segment types. When the LLM labels a new segment (e.g. `פנדר → brand`), the raw text is validated against the taxonomy and then written to the library. On the next request with the same surface form, the library scan matches before coverage falls below the LLM threshold — the LLM is not called.

This means the LLM call rate decays naturally over time as the service sees more queries. A fresh instance starts cold; a warm instance handles common queries entirely offline.

**Persistence:** The pattern library is currently in-process only. Across restarts or instances it resets. For persistent learning, serialise the library to a shared store (Redis, S3) on a background interval. The interface is intentionally simple (`learn(text, type)`, `scan(query)`) to make this a drop-in addition.

**Safety of what gets learned — and used:** The same `valid_types` check guards both what the LLM segment *does* in this request and what enters the library. A segment whose type does not resolve to a taxonomy value is neither used for classification nor learned, so a plausible-but-wrong label can neither misclassify the current query nor poison future ones. This matters because the library is self-reinforcing — a bad pattern would be reused on every future query of the same shape. The `_model` extractor's bare-number guard is part of this: it stops a mislabelled price/year from being learned as the numeric "model" shape `????`, which would otherwise match any 4-digit number forever.

---

## Evaluation & Known Risks (what I would do next)

The current correctness signal is the offline test suite plus a manual stress run across ~60 queries. That is enough to validate the happy paths and the security defences, but a production deployment needs a deeper, continuous evaluation loop. Specifically, I would:

- **Build a labelled evaluation set** (hundreds of real Hebrew queries with expected category + params) and run it on every change to measure precision/recall per field and per vertical — not just pass/fail on a handful of golden cases. `tests/dataset/cases.json` (run as parametrized pytest cases in `test_cases.py`) is the seed of this.
- **Evaluate the LLM segmentation step in isolation**: how often does it mislabel a segment, drop a meaningful token, or hallucinate a value? Each failure mode (e.g. labelling a bare number as a model) should be a tracked metric with a regression guard, not a bug found by chance.
- **Check whether the LLM has the context it needs to label safely.** The segmenter only sees the query, the already-identified spans, and the FAISS taxonomy hints. Some queries are genuinely ambiguous without more context (e.g. brand vs. model, sector disambiguation). I would measure where the LLM guesses without sufficient signal and decide, per case, whether to feed it more taxonomy context, add few-shot examples, or fall back to "unknown" rather than guess.
- **Validate the self-learning loop continuously.** Because learned patterns are reused, a single bad learned pattern compounds. I would log every learned pattern, periodically audit them against the taxonomy, and add a confidence threshold / human-review step before a pattern is promoted to the shared persistent store.
- **A/B the coverage threshold and model choice** against the labelled set to find the real cost/quality frontier, rather than picking the threshold by hand.

These are deliberately framed as evaluation and observability work: the architecture already isolates the LLM to a single, schema-constrained step, so the open question is not "is it wired correctly" but "how often is it wrong, and where is it missing the context to be right."

---

## Semantic FAISS Index

At build time (`src/taxonomy/build_index.py`), every canonical taxonomy value is embedded with `text-embedding-3-small` and stored in a FAISS `IndexFlatIP` (cosine via L2-normalised vectors). At runtime the index is loaded once and injected into `NodeContext`.

During LLM gap-fill, two concurrent embedding calls are made (gap text + full query) and the top-8 nearest taxonomy values per field type are passed to the LLM as hints. This bridges Hebrew morphological variants (e.g. `עגלת תינוק → עגלות`) that exact lookup would miss.

The index is generated once, offline, by `src/taxonomy/build_index.py` (the only place embeddings are computed — the request path makes no embedding call to build it). It is then loaded from disk at startup if present. If the index files are absent, the service logs a warning and runs without semantic hints: exact taxonomy lookup, learned patterns, and numeric rules all still work — only the morphological-variant bridging is disabled. To enable semantic hints in a fresh clone or container, run the build script (or ship the generated `taxonomy.faiss` / `taxonomy.meta.json` alongside the image).

---

## Taxonomy

`yad2_search_taxonomy.json` defines the allowed values for every field across all three verticals, plus:
- `typo_map` — known misspellings and their corrections
- `unit_aliases` — unit normalisation rules
- Numeric range limits (used to validate extracted values)

The Pydantic schemas in `src/taxonomy/schemas.py` are derived from this taxonomy. The LLM's output enum (`SEGMENT_TYPE_NAMES`) is also derived from it, so the LLM can never label a segment with a type that doesn't exist in the schema.

---

## Security

### Threat model

| Threat | Mitigation |
|---|---|
| Prompt injection via search query | Two-layer detection: keyword scanner + LLM deepcheck |
| Off-taxonomy keys in LLM output | Pydantic `extra="forbid"` + allowlist filtering in validate node |
| Hallucinated enum values | Strict JSON schema passed to OpenAI structured outputs |
| Instruction override via RTL/unicode tricks | NFKC normalisation + RTL-override character strip in sanitize |
| Long inputs causing token overflow | Hard truncation at 512 chars before any LLM call |
| LLM output with arbitrary JSON | JSON schema enforced at the OpenAI API level (`strict: true`) |
| System prompt leakage | Query always injected as `<query>…</query>` user-role content, never in system prompt |

---

## Scaling

### Single instance
- Target: ≥12 QPS (≈1M queries/day)
- Cache / rules path: ≤150 ms p95 — achievable, no network calls
- LLM path: ≤600 ms p95 is a target, **not a guarantee** — this path makes a network call, so it is dominated by the API round-trip (network + time-to-first-token + generation), and TTFT alone can exceed 600 ms under load. The design does **not** try to make this call fast; it makes it **rare**. Patterns, the cache, and the self-learning library resolve the large majority of queries with no LLM call at all, so this slow path barely moves the overall p95 — the lower the LLM-call rate (which keeps dropping as the library warms up), the less the slow tail matters. We bound the worst case with a 20 s timeout + 1 retry, and read the real distribution from the `yad2_request_latency_seconds` Prometheus histogram (exposed at `/metrics`, queried with `histogram_quantile`) rather than assuming it.

### Horizontal scaling
The service is fully stateless — all mutable state is:
1. The in-process LRU cache → swap for Redis (same `Cache` interface)
2. The in-process PatternLibrary → swap for a shared persistent store

With Redis-backed cache, any number of instances can be placed behind a load balancer. The FAISS index is read-only and loaded from disk at startup — no coordination required.

### Throughput math
Target: ≥ 12 QPS per instance (~1M queries/day).

At 12 QPS with a 10% LLM rate the instance makes ~1.2 LLM calls/s:

- **RPM:** 1.2 calls/s × 60 = **~72 RPM**. OpenAI Tier 1 limit is 500 RPM — comfortable headroom.
- **TPM:** Each segmentation call is ~400 input + ~80 output tokens. At 72 calls/min: 72 × 480 = **~34,600 TPM**. OpenAI Tier 1 limit is 200K TPM — well within range.

These are estimates based on assumed 10% LLM rate and ~400 input token average per call. Actual usage should be confirmed from live metrics before committing to an API tier.

---

## Technology choices

| Choice | Rationale |
|---|---|
| **FastAPI** | Async-native, automatic OpenAPI docs, Pydantic integration |
| **LangGraph** | Explicit node/edge pipeline with typed state; routing logic is one function, not scattered conditionals |
| **Pydantic v2 with `extra="forbid"`** | Schema enforcement at the Python level independent of the LLM; unknown keys are structurally impossible to return |
| **In-process LRU over Redis** | Zero infrastructure for single-instance deployment; swappable interface for scale-out |
| **Pattern-first, LLM for gaps** | Deterministic rules are fast, free, and auditable; LLM is reserved for ambiguous cases. Self-learning collapses LLM rate over time |
| **FAISS + text-embedding-3-small** | Cheap to build ($<0.01), fast to query, bridges Hebrew morphological variants without a dictionary lookup per query |
| **OpenAI structured outputs** | JSON schema enforced at the API level — no post-processing parser, no hallucinated keys |
