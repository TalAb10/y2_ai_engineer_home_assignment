"""Prometheus metrics collectors.

All counters/histograms are registered here as module-level singletons.
The /metrics endpoint calls prometheus_client.generate_latest() to expose them.

Tracked per the assignment requirements:
  - Total requests / per category
  - Error rate
  - p50/p95 latency
  - Cache hit ratio
  - Token usage & cost/request
  - Model call success/failure rate
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ── Request counters ───────────────────────────────────────────────────────────
requests_total = Counter(
    "yad2_requests_total",
    "Total parse requests received",
    labelnames=["category"],  # label value set after classification; "unknown" until then
)

errors_total = Counter(
    "yad2_errors_total",
    "Total parse errors",
    labelnames=["error_type"],
)

# ── Latency ───────────────────────────────────────────────────────────────────
request_latency_seconds = Histogram(
    "yad2_request_latency_seconds",
    "End-to-end /parse latency in seconds",
    # Fine granularity around the latency targets (150 ms cache/rules, 600 ms LLM),
    # then coarse buckets out to ~40 s so the slow LLM tail is actually measurable.
    # The LLM call has a 20 s timeout with 1 retry, so a worst-case request can
    # approach 40 s; without buckets this high, histogram_quantile would clamp the
    # reported p95/p99 at the top bucket and hide the real tail.
    buckets=(0.05, 0.1, 0.15, 0.25, 0.5, 0.6, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0),
)

# ── Cache ─────────────────────────────────────────────────────────────────────
cache_hits_total = Counter(
    "yad2_cache_hits_total",
    "Cache hits",
)

cache_misses_total = Counter(
    "yad2_cache_misses_total",
    "Cache misses",
)

# ── LLM ──────────────────────────────────────────────────────────────────────
llm_calls_total = Counter(
    "yad2_llm_calls_total",
    "LLM API calls made",
    labelnames=["model", "status"],   # status: success | refusal | error
)

llm_tokens_input_total = Counter(
    "yad2_llm_tokens_input_total",
    "Total input tokens sent to LLM",
    labelnames=["model"],
)

llm_tokens_output_total = Counter(
    "yad2_llm_tokens_output_total",
    "Total output tokens received from LLM",
    labelnames=["model"],
)

llm_cost_usd_total = Counter(
    "yad2_llm_cost_usd_total",
    "Cumulative LLM cost in USD",
    labelnames=["model"],
)

# ── Security events ───────────────────────────────────────────────────────────
security_events_total = Counter(
    "yad2_security_events_total",
    "Input sanitization security flag triggers",
    labelnames=["flag_type"],
)

injections_confirmed_total = Counter(
    "yad2_injections_confirmed_total",
    "Queries confirmed as injection attempts by LLM deepcheck",
)

# ── Convenience helper ────────────────────────────────────────────────────────

def record_llm_usage(model: str, status: str, input_tok: int, output_tok: int, cost: float) -> None:
    """Update all LLM-related counters in one call."""
    llm_calls_total.labels(model=model, status=status).inc()
    llm_tokens_input_total.labels(model=model).inc(input_tok)
    llm_tokens_output_total.labels(model=model).inc(output_tok)
    llm_cost_usd_total.labels(model=model).inc(cost)
