# Yad2 Hebrew Search Parser

A production-ready service that converts free-text Hebrew search queries into structured Yad2 marketplace parameters across three verticals: **נדל״ן** (Real Estate), **רכב** (Vehicles), and **יד_שנייה** (Second-hand).

---

## Quick Start

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# (Optional) Build the semantic FAISS index — requires API key, costs < $0.01
python scripts/build_taxonomy_index.py
```

### Option A — Backend only (Dockerfile)

The assignment requires a Dockerfile; this is the minimal way to run the service:

```bash
docker build -t yad2-parser .
docker run --env-file .env -p 8000:8000 yad2-parser
```

Service available at `http://localhost:8000`. The `/metrics` endpoint exposes raw Prometheus text — use Option B to get Prometheus + Grafana scraping it automatically.

### Option B — Full stack (Docker Compose)

Starts the backend, Prometheus, and Grafana together with everything pre-wired:

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Parser API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

The Yad2 dashboard loads automatically in Grafana. You can also import `grafana_dashboard.json` manually into any existing Grafana instance — go to Dashboards → Import → Upload JSON file, then select your Prometheus data source.

---

## API

### `POST /parse`

Convert a free-text Hebrew query into structured search parameters.

**Request**
```json
{ "q": "דירת 3 חדרים בירושלים עד מליון שח" }
```

Add `"debug": true` to receive internal extraction details (segments, coverage, LLM usage).

**Response**
```json
{
  "category": "נדל״ן",
  "params": {
    "מס׳_חדרים": 3,
    "עיר": "ירושלים",
    "מחיר": { "max": 1000000 }
  },
  "confidence": 0.82,
  "notes": []
}
```

| Field | Type | Description |
|---|---|---|
| `category` | `"נדל״ן" \| "רכב" \| "יד_שנייה"` | Detected marketplace vertical |
| `params` | object | Extracted filters — keys and value shapes follow the taxonomy schema |
| `confidence` | float [0, 1] | Extraction confidence (0.4 × classification certainty + 0.6 × coverage) |
| `notes` | string[] | Normalization or assumption notes |

**Error responses**

| Status | Body | Cause |
|---|---|---|
| 400 | `{"error": "blocked_query"}` | Injection attempt confirmed by security pipeline |

---

### `GET /health`

```json
{ "status": "ok", "llm_available": true, "cache": { "ok": true } }
```

### `GET /metrics`

Prometheus text format. See [Observability](#observability) for the full metric list.

> FastAPI also serves interactive API docs automatically at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

---

## Examples

### Real Estate

| Query | category | Key params |
|---|---|---|
| `דירת 3 חדרים בירושלים עד מליון שח` | `נדל״ן` | `סוגי_נכס: [דירה]`, `מס׳_חדרים: 3`, `עיר: ירושלים`, `מחיר: {max: 1000000}` |
| `דירת סטודיו להשכרה בתל אביב עד 5000 שח` | `נדל״ן` | `סוגי_נכס: [דירת סטודיו]`, `מצבי_עסקה: [השכרה]`, `עיר: תל אביב-יפו`, `מחיר: {max: 5000}` |
| `דירה 4 חדרים עם מעלית וחניה בחיפה` | `נדל״ן` | `סוגי_נכס: [דירה]`, `מס׳_חדרים: 4`, `עיר: חיפה`, `מעלית: true`, `חניה: 1` |
| `פנטהאוז עם נוף לים 5 חדרים בנתניה עד 4 מיליון` | `נדל״ן` | `סוגי_נכס: [פנטהאוז]`, `מס׳_חדרים: 5`, `עיר: נתניה`, `מחיר: {max: 4000000}` |
| `דירה בין 1.5 מיליון ל-2 מיליון` | `נדל״ן` | `סוגי_נכס: [דירה]`, `מחיר: {min: 1500000, max: 2000000}` |

**Full example response:**
```json
{
  "category": "נדל״ן",
  "params": {
    "סוגי_נכס": ["דירה"],
    "מס׳_חדרים": 3,
    "עיר": "ירושלים",
    "מחיר": { "max": 1000000 }
  },
  "confidence": 0.84,
  "notes": []
}
```

### Vehicles

| Query | category | Key params |
|---|---|---|
| `טויוטה קורולה 2018-2021 עד 70 אלף שח צבע לבן` | `רכב` | `יצרן: טויוטה`, `דגם: קורולה`, `שנה: {min:2018, max:2021}`, `מחיר: {max: 70000}`, `צבע: לבן` |
| `טסלה מודל 3 חשמלי עד 150000 שח` | `רכב` | `יצרן: טסלה`, `דגם: מודל 3`, `סוג_דלק: חשמלי`, `מחיר: {max: 150000}` |
| `יונדאי טוסון עד 80000 ק״מ` | `רכב` | `יצרן: יונדאי`, `דגם: טוסון`, `ק״מ: {max: 80000}` |
| `יונדי טוסון 2020` *(typo)* | `רכב` | `יצרן: יונדאי` *(corrected from יונדי)*, `דגם: טוסון`, `שנה: {min:2020, max:2020}` |

**Full example response:**
```json
{
  "category": "רכב",
  "params": {
    "יצרן": "טויוטה",
    "דגם": "קורולה",
    "שנה": { "min": 2018, "max": 2021 },
    "מחיר": { "max": 70000 },
    "צבע": "לבן"
  },
  "confidence": 0.91,
  "notes": []
}
```

### Second-hand

| Query | category | Key params |
|---|---|---|
| `אייפון 13 פרו 256 ג׳יגה כמו חדש עד 2500` | `יד_שנייה` | `סקטור: אלקטרוניקה`, `תת_קטגוריה: טלפונים_סלולריים`, `נפח_אחסון: 256GB`, `מצב: כמו חדש`, `מחיר: {max: 2500}` |
| `מחשב נייד HP i7 16 גיגה RAM עד 3000 שח` | `יד_שנייה` | `סקטור: אלקטרוניקה`, `תת_קטגוריה: מחשבים_ניידים`, `מותג: HP`, `דגם: i7`, `מחיר: {max: 3000}` |
| `ספה פינתית עד 2000 שח` | `יד_שנייה` | `סקטור: ריהוט`, `תת_קטגוריה: סלון`, `מחיר: {max: 2000}` |

**Full example response:**
```json
{
  "category": "יד_שנייה",
  "params": {
    "תת_קטגוריה": "טלפונים_סלולריים",
    "נפח_אחסון": "256GB",
    "מצב": "כמו חדש",
    "מחיר": { "max": 2500 }
  },
  "confidence": 0.78,
  "notes": []
}
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document.

**Pipeline overview:**

```
POST /parse
     │
     ▼
┌─────────┐   ┌────────────────┐   ┌───────────┐   ┌──────────────┐
│sanitize │──▶│ security_check │──▶│ normalize │──▶│ cache_lookup │──▶ END (cache hit)
│         │   │ (if flagged)   │   │           │   │              │
└─────────┘   └────────────────┘   └───────────┘   └──────┬───────┘
                      │ (injection)                         │ miss
                      ▼                                     ▼
                     END                             ┌─────────────┐
                  400 blocked                        │   extract   │
                                                     │ (patterns + │
                                                     │  LLM gaps)  │
                                                     └──────┬──────┘
                                                            ▼
                                                     ┌─────────────┐
                                                     │  validate   │
                                                     │ (Pydantic)  │
                                                     └──────┬──────┘
                                                            ▼
                                                     ┌─────────────┐
                                                     │ cache_store │
                                                     └──────┬──────┘
                                                            ▼
                                                           END
```

---

## Design & Methodology

### Why pattern-first, LLM for gaps?

Many Yad2 queries follow predictable templates: a known city name, a number of rooms, a price range. These can be parsed entirely by deterministic rules — in-process, with no network call and no token cost. An LLM call, by contrast, adds the round-trip: network latency, time-to-first-token (the model's queue + prompt-processing delay before any output appears), then the generation time itself — and it bills per token. So the pipeline runs the rules first and only calls the LLM when pattern coverage of the query falls below `PATTERN_COVERAGE_THRESHOLD` (default 0.95) — and even then, only for the uncovered fragments, not the whole query. The exact savings depend on the model, prompt size, and query mix; they should be measured from the live `/metrics` counters rather than assumed.

### Taxonomy span lookup

Every canonical value in `yad2_search_taxonomy.json` (cities, manufacturers, property types, conditions, …) is pre-indexed as exact or prefix spans. At request time, the query is scanned in a single pass and matching spans are typed (e.g. `"ירושלים" → city`, `"טויוטה" → manufacturer`). No model call, no network, sub-millisecond.

### Typo dictionary

The taxonomy ships with a `typo_map` that covers common Hebrew misspellings and slang variants (e.g. `יונדי → יונדאי`, `ירושליים → ירושלים`, `מיליון → מיליון`). The normalize node applies this map before any lookup, so misspelled queries hit the cache and the pattern index the same as correctly-spelled ones. When the LLM encounters a new typo it returns a correction in its `normalizations` output, and the service learns it for future requests.

### Semantic FAISS hints fed to the LLM

For queries that still have uncovered gaps after rules run, the service embeds both the full query and the uncovered portion with `text-embedding-3-small` and searches the FAISS index (built offline from all taxonomy values) for the nearest canonical matches. These top-8 suggestions per field type are passed to the LLM as context alongside the query. This means the LLM does not need to hallucinate field values — it picks from a shortlist of values it knows exist in the taxonomy (e.g. `"עגלת" → "עגלות"`). This both improves accuracy and makes the LLM's output easier to validate.

### Self-learning pattern library

When the LLM labels a new segment (e.g. surface text `"פנדר"` → type `brand`), the mapping is written to an in-process `PatternLibrary`. On the next request with the same surface form, the library scan matches before coverage falls below the threshold — the LLM is not called. LLM call rate decays naturally as the service warms up.

---

## Scale & Performance

| Target | Value |
|---|---|
| Throughput | ≥ 12 QPS per instance (~1M queries/day) |
| p95 latency — cache/rules path | ≤ 150 ms |
| p95 latency — LLM path | ≤ 600 ms |

**Why the pattern-first design exists:** the LLM path is inherently too slow to meet a tight latency target — it is dominated by the API round-trip (network + time-to-first-token + generation), and TTFT alone is variable and routinely exceeds 600 ms under load (we saw multi-second responses in stress testing). We cannot make a single LLM call reliably fast. So instead of trying to, the architecture keeps the **slow path rare**: deterministic rules resolve the majority of queries entirely in-process, and the LLM is only invoked for the fraction that rules cannot cover. The latency target is met by *avoiding* the LLM on most requests, not by making the LLM fast.

**How each path relates to the targets:**
- Cache hit path: no network calls — in-process LRU lookup + JSON serialisation only. Comfortably within the 150 ms target.
- Rules-only path: taxonomy span scan + regex numerics, all in-process. Also within the 150 ms target. This is the path the pattern-first design pushes most traffic onto.
- LLM path (the rare, slow minority): the 600 ms target is **not guaranteed** — TTFT can exceed it on its own. We only reduce the controllable parts (send just the uncovered gap as a short prompt; bound the worst case with a 20 s timeout + 1 retry). The real distribution must be read from the `yad2_request_latency_seconds` histogram, not assumed. The lower the LLM-call rate (driven up by the pattern library and cache warming over time), the less this slow tail affects overall p95.

**Horizontal scaling:**
The service is stateless — the only mutable per-instance state is the in-process LRU cache and the PatternLibrary. Swap `create_cache()` for a Redis-backed implementation (same `get/set/health` interface) and the PatternLibrary for a shared persistent store to run any number of instances behind a load balancer with no coordination required. The FAISS index is read-only and loaded from disk at startup.

---

## Caching Strategy

**Implementation:** In-process thread-safe LRU (10,000 entries, `OrderedDict` + lock).

**Cache key:** SHA-256 of the NFKC-normalised, lowercased query — applied *after* the normalize node, so `"דירה 3 חדרים"` and `"דירה  3 חדרים"` (extra space) share a cache entry.

**Why exact-match and not semantic caching:** Near-miss queries like `"עד מליון"` vs `"עד 2 מליון"` share high cosine similarity but produce different numeric params. Semantic cache hits would return wrong numbers silently. Exact-normalised keys are the safe and correct choice.

**Hit path latency:** The cache_lookup node short-circuits to END — normalize, extract, validate, and cache_store are all skipped. Target p95 ≤ 150 ms on the cache path.

**Scaling the cache:** The current in-process LRU is sufficient for a single instance. For horizontal scaling, swap `create_cache()` in [src/cache/cache.py](src/cache/cache.py) for a Redis-backed implementation behind the same `get/set/health` interface — no other code changes required.

---

## Cost Model

### Token cost per LLM call
The service uses a pattern-first strategy that minimises LLM calls:

| Path | Condition | LLM calls |
|---|---|---|
| Cache hit | Seen query | 0 |
| Pattern-only | ≥95% of query covered by rules | 0 |
| LLM gap-fill | Coverage < 95% | 1 segmentation call |
| Security deepcheck | Injection marker found | +1 classification call |

**Segmentation call** (gap-fill): ~400 input tokens (fixed system prompt + annotated query) + ~80 output tokens.  
**Security call** (deepcheck): ~200 input tokens + ~10 output tokens.

### 10M queries / month estimate

| Segment | Queries | LLM calls | Input tokens | Output tokens |
|---|---|---|---|---|
| Cache hits (65%) | 6,500,000 | 0 | — | — |
| Pattern-only (25%) | 2,500,000 | 0 | — | — |
| LLM gap-fill (10%) | 1,000,000 | 1,000,000 | 400M | 80M |
| Security deepcheck (1% of all) | 100,000 | 100,000 | 20M | 1M |

**Cost at `gpt-5-mini` pricing ($0.25/1M in, $2.00/1M out):**

| | Tokens | Cost |
|---|---|---|
| Input | 420M | $105 |
| Output | 81M | $162 |
| **Total** | | **~$267 / month** |

**Cost per query:** ~$0.000027 (~$27 per million queries).

### Cost reduction levers

1. **Cache warm-up**: Pre-parse the top-N most frequent queries from search logs on deploy. A 65% hit rate is conservative — popular marketplaces typically see 80%+ repeat queries.
2. **Prompt caching** *(opportunity, not yet realised)*: The segmentation system prompt is a fixed prefix, so it is a candidate for OpenAI's automatic prompt caching. We do **not** enable anything explicitly — OpenAI auto-caches only prompts ≥1024 tokens, and our system prompt is likely below that, so in practice caching probably does not trigger today. The cost accounting already credits cached tokens (`_build_usage` bills `cached_tokens` at a reduced rate), so *if* the prompt is enlarged past the threshold or the model caches it, the saving is captured. To actually realise it, pad/restructure the prompt to cross the caching threshold and confirm via the `cached_tokens` field in `/metrics`.
3. **Lightweight classifier + selective LLM calls**: The pattern coverage score already acts as a classifier — if coverage is at or above `PATTERN_COVERAGE_THRESHOLD` (default 0.95), the LLM is skipped entirely. Raising `PATTERN_COVERAGE_THRESHOLD` tightens this gate further. A dedicated lightweight classifier (e.g. a small embedding-based model) could replace the coverage heuristic for more accurate LLM-call routing.
4. **Model downgrade for security deepcheck**: The security deepcheck is a binary classification task (legitimate vs. injection) — much simpler than segmentation. In a real system I would run an evaluation on a labelled set of flagged queries to measure whether a smaller, cheaper model matches the accuracy of `gpt-5-mini` on this task. If it does, the deepcheck cost drops significantly since the model cost dominates at scale.
5. **Prompt compression**: The segmentation system prompt lists all allowed segment types with one-line descriptions. Removing descriptions for types already matched by rules in the current query would reduce input tokens by ~30% on LLM calls. Not yet implemented — worth measuring against accuracy before applying.
6. **Embeddings + rules replace LLM for common slang**: Once the PatternLibrary has seen a surface form once, it never calls the LLM for it again. At steady state, embeddings + taxonomy rules handle the long tail; LLM calls concentrate on truly novel queries.
7. **Batch embeddings**: The FAISS index is built once offline; per-request embedding calls only happen during gap-fill and are batched via `asyncio.gather`.

---

## Observability

All metrics are exported at `GET /metrics` in Prometheus text format. A `prometheus.yml` scrape config and a Grafana dashboard ([grafana_dashboard.json](grafana_dashboard.json)) are included.

Parsing decisions and security events are written as structured JSON logs (via `python-json-logger`) so they can be ingested by any log aggregator. Key log events: `parse_decision` (query, category, confidence, cache_hit, llm_used) and `security_event` (flag_type, query_snippet).

| Metric | Type | Labels | Description |
|---|---|---|---|
| `yad2_requests_total` | Counter | `category` | Total parse requests, per vertical |
| `yad2_errors_total` | Counter | `error_type` | Parse errors and blocked queries |
| `yad2_request_latency_seconds` | Histogram | — | End-to-end /parse latency (p50/p95 via PromQL) |
| `yad2_cache_hits_total` | Counter | — | Cache hits |
| `yad2_cache_misses_total` | Counter | — | Cache misses |
| `yad2_llm_calls_total` | Counter | `model`, `status` | LLM calls by model and outcome |
| `yad2_llm_tokens_input_total` | Counter | `model` | Cumulative input tokens |
| `yad2_llm_tokens_output_total` | Counter | `model` | Cumulative output tokens |
| `yad2_llm_cost_usd_total` | Counter | `model` | Cumulative LLM cost in USD |
| `yad2_security_events_total` | Counter | `flag_type` | Injection marker triggers |
| `yad2_injections_confirmed_total` | Counter | — | Queries confirmed as injections |

**Cache hit ratio** (PromQL):
```
rate(yad2_cache_hits_total[5m]) /
  (rate(yad2_cache_hits_total[5m]) + rate(yad2_cache_misses_total[5m]))
```

**p95 latency** (PromQL):
```
histogram_quantile(0.95, rate(yad2_request_latency_seconds_bucket[5m]))
```

---

## Security

See [ARCHITECTURE.md](ARCHITECTURE.md#security) for the full threat model.

**Defenses:**
- **Input sanitization**: NFKC normalisation, emoji/control-char strip, RTL-override strip, 512-char truncation — before any LLM call.
- **Two-layer injection detection**: (1) fast keyword scanner (4 categories: instruction_override, role_injection, prompt_extraction, delimiter_injection), then (2) LLM binary deepcheck to eliminate false positives.
- **Fixed system prompts**: `SEGMENTATION_SYSTEM` and the security classifier prompt are module-level constants — never dynamically constructed from user input.
- **User query isolation**: The query is always injected as delimited `<query>…</query>` user-role content, never concatenated into the system prompt (instruction-hierarchy defence, OWASP LLM01).
- **Strict schema enforcement**: All LLM output goes through per-vertical Pydantic models with `extra="forbid"`. Unknown keys are dropped before the response is returned.
- **Allowlisted categories and fields**: `VALID_CATEGORIES` is a frozenset; field names and enum values are locked in Pydantic Literal types derived from the taxonomy.

**Security tests** (`tests/test_security_redteam.py`): classic injection, role injection, delimiter injection, prompt extraction, oversized input, RTL override, null bytes, percent-encoded payloads, empty query, gibberish query.

---

## Testing

```bash
# Offline tests (no API key required)
pytest

# Live LLM integration tests (requires OPENAI_API_KEY in .env)
pytest tests/test_llm_integration.py -v -s
```

Test suites:

| File | Coverage |
|---|---|
| `tests/test_examples.py` | Golden examples — all three verticals |
| `tests/test_security_redteam.py` | 11 red-team / abuse cases |
| `tests/test_patterns.py` | Regex and numeric extraction |
| `tests/nodes/test_sanitize.py` | Sanitize node unit tests |
| `tests/nodes/test_normalize.py` | Normalize node unit tests |
| `tests/nodes/test_validate.py` | Schema validation unit tests |
| `tests/test_llm_integration.py` | End-to-end LLM path (live) |
| `tests/eval_cases.json` | 26 labelled eval cases with expected params |

---

## Configuration

All settings are read from environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required for LLM)* | OpenAI API key |
| `LLM_MODEL` | `gpt-5-mini` | Model for segmentation and security calls |
| `LLM_ENABLED` | `true` | Set `false` for rules-only offline mode |
| `LLM_TIMEOUT_S` | `20` | Per-call timeout in seconds |
| `PATTERN_COVERAGE_THRESHOLD` | `0.95` | Min pattern coverage before calling LLM |
| `MAX_INPUT_CHARS` | `512` | Input truncation limit |
| `CACHE_SIZE` | `10000` | In-process LRU capacity |
| `LOG_LEVEL` | `INFO` | Logging level |
