"""LLM client.

Provides:
  - TokenUsage / LLMResult: value types returned by every call
  - LLMClient: OpenAI backend with strict structured outputs and cost accounting
  - NoOpLLMClient: offline / rules-only stand-in that always refuses
  - create_llm_client(settings): selects the client based on configuration

Security: the user query is always passed as delimited <query> user-role content,
never concatenated into the system prompt (instruction-hierarchy defence).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""


@dataclass(frozen=True)
class LLMResult:
    """One structured-output call result.

    Exactly one of `parsed` or `refusal` is non-None:
      parsed  → schema-conforming dict (use it)
      refusal → model declined; check security_flags before escalating
    """
    parsed: dict[str, Any] | None
    refusal: str | None
    usage: TokenUsage

    @property
    def ok(self) -> bool:
        return self.parsed is not None


class LLMClient:
    """OpenAI-backed client using strict json_schema structured outputs."""

    def __init__(
        self,
        api_key: str,
        timeout_s: float = 8.0,
        max_retries: int = 1,
        price_in: float = 0.25,
        price_out: float = 2.00,
    ) -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout_s, max_retries=max_retries)
        self._price_in = price_in
        self._price_out = price_out

    def is_available(self) -> bool:
        return True

    async def embed(self, text: str) -> list[float] | None:
        """Return an embedding vector, or None on error."""
        try:
            response = await self._client.embeddings.create(
                input=text,
                model="text-embedding-3-small",
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.warning("Embedding error: %s", exc)
            return None

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_content: str,
        json_schema: dict[str, Any],
        schema_name: str,
        model: str,
    ) -> LLMResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<query>{user_content}</query>"},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": json_schema},
        }
        try:
            import openai
            completion = await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_format=response_format,  # type: ignore[arg-type]
            )
        except Exception as exc:  # openai.APIError
            logger.warning("OpenAI error: %s", exc)
            return LLMResult(parsed=None, refusal=f"api_error:{exc}", usage=TokenUsage(model=model))

        msg = completion.choices[0].message
        usage = self._build_usage(completion.usage, model)

        if getattr(msg, "refusal", None):
            return LLMResult(parsed=None, refusal=msg.refusal, usage=usage)

        parsed: dict[str, Any] | None = None
        if hasattr(msg, "parsed") and msg.parsed is not None:
            parsed = msg.parsed  # type: ignore[assignment]
        elif msg.content:
            try:
                parsed = json.loads(msg.content)
            except json.JSONDecodeError as exc:
                return LLMResult(parsed=None, refusal=f"json_error:{exc}", usage=usage)

        return LLMResult(parsed=parsed, refusal=None, usage=usage)

    def _build_usage(self, raw: Any, model: str) -> TokenUsage:
        if raw is None:
            return TokenUsage(model=model)
        input_tokens: int = getattr(raw, "prompt_tokens", 0)
        output_tokens: int = getattr(raw, "completion_tokens", 0)
        cached_tokens: int = getattr(getattr(raw, "prompt_tokens_details", None), "cached_tokens", 0)
        # Cached input tokens bill at 25% of normal price (OpenAI prompt caching).
        cost = ((input_tokens - cached_tokens) * self._price_in
                + cached_tokens * self._price_in * 0.25
                + output_tokens * self._price_out) / 1_000_000
        return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached_tokens,
                          cost_usd=cost, model=model)


class NoOpLLMClient:
    """Returns refusal on every call — used in offline / rules-only mode."""

    def is_available(self) -> bool:
        return False

    async def embed(self, text: str) -> list[float] | None:
        return None

    async def complete_structured(self, **_kwargs: Any) -> LLMResult:
        return LLMResult(parsed=None, refusal="llm_disabled", usage=TokenUsage())


def create_llm_client(settings: Any) -> LLMClient | NoOpLLMClient:
    """Return a live client when the LLM is enabled and configured, else a no-op."""
    if not settings.llm_enabled or not settings.openai_api_key:
        return NoOpLLMClient()
    return LLMClient(
        api_key=settings.openai_api_key,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        price_in=settings.price_in,
        price_out=settings.price_out,
    )
