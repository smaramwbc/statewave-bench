"""Unified LLM client wrapping Anthropic + OpenAI.

Two roles in this benchmark:

  1. **Answer model** — every system-under-test eventually produces a
     prompt + retrieved context and needs an LLM to generate the final
     answer. The bench fixes the answer model so cross-system results
     are comparable (one system can't win because it happened to use
     a stronger model).

  2. **LLM-as-judge** — open-ended LoCoMo questions can't be string-
     matched, so we run a separate model that scores answers against
     ground truth. We deliberately use a *different* model for judging
     than for answering to reduce same-model-bias.

Defaults: Sonnet for the answer role, GPT-4o for the judge. Both can
be overridden via env vars; pricing-sensitive operators can swap to
smaller models for the pilot run.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

# Provider client imports stay top-level — these are library deps,
# always installed. The actual API calls only fire when an answer or
# judge call happens, so an `import llm` from inside a unit test
# doesn't need real keys.

# ── Default model selection ───────────────────────────────────────────────
# Override via env vars: SWB_ANSWER_MODEL, SWB_JUDGE_MODEL.
# Switch providers by changing the model string — the client
# auto-routes by name prefix.

# Default answer model is claude-haiku-4-5: ~2x faster than Sonnet at
# comparable LoCoMo accuracy on conv-26 probes. Matches the answer model
# Honcho's published bench uses, so our numbers are directly comparable
# to theirs without a model-mix confound. Override via SWB_ANSWER_MODEL
# for the (slower, more expensive) Sonnet track if needed for production
# comparison runs.
DEFAULT_ANSWER_MODEL = "claude-haiku-4-5"
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"


# ── Public types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LlmCall:
    """Result of one model call. Surfacing token counts so the bench
    runner can report cost and prompt-size efficiency alongside
    answer quality."""

    model: str
    answer: str
    input_tokens: int
    output_tokens: int


Provider = Literal["anthropic", "openai"]


# ── Retry helper (shared by answer + judge calls) ─────────────────────────


class ProviderQuotaExhausted(RuntimeError):
    """The provider reports the operator's quota / credit balance is
    exhausted. Every subsequent call will fail the same way; retries
    won't help and continuing burns spend on requests that can never
    succeed. The runner catches this at the top level and halts the
    bench so the operator can refill and re-run with --resume.

    Triggered by provider error messages containing:
      - `insufficient_quota`           (OpenAI)
      - `quota_exceeded`               (OpenAI variant)
      - `credit balance is too low`    (Anthropic)
      - `billing_quota_exceeded`       (other)
    """


# Provider error-body markers that indicate "no more credits / quota".
# Matched case-insensitively against str(exc).
_QUOTA_MARKERS: tuple[str, ...] = (
    "insufficient_quota",
    "quota_exceeded",
    "billing_quota_exceeded",
    "credit balance is too low",
)


# Patterns that indicate the failure is transient and worth retrying.
# Anthropic 529 ("overloaded_error") is the dominant pattern we saw
# during the first --limit 10 run — bursts that lasted minutes and
# burned through the runner's failure-streak breaker on every system.
# Catching it here means individual answer / judge calls retry through
# the burst instead of bubbling up as fatal failures.
_TRANSIENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bError code:\s*5\d\d\b", re.IGNORECASE),
    re.compile(r"\bError code:\s*429\b", re.IGNORECASE),
    re.compile(r"\bConnectionError\b", re.IGNORECASE),
    re.compile(r"\bRemoteProtocolError\b", re.IGNORECASE),
    re.compile(r"\bAPITimeoutError\b", re.IGNORECASE),
    re.compile(r"\bAPIConnectionError\b", re.IGNORECASE),
    re.compile(r"\boverloaded_error\b", re.IGNORECASE),  # Anthropic 529 body
    re.compile(r"\btimed?\s*out\b", re.IGNORECASE),
)


def _is_quota_error(err: BaseException) -> bool:
    s = str(err).lower()
    return any(marker.lower() in s for marker in _QUOTA_MARKERS)


def _is_transient(err: BaseException) -> bool:
    s = str(err)
    return any(p.search(s) for p in _TRANSIENT_PATTERNS)


T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    initial_backoff_sec: float = 1.0,
    backoff_multiplier: float = 4.0,
    backoff_cap_sec: float = 60.0,
) -> T:
    """Call `fn`, retrying on transient errors with capped exponential backoff.

    - Quota errors short-circuit immediately as ProviderQuotaExhausted.
    - Transient errors (5xx incl. Anthropic 529, 429-without-quota,
      connection/timeout, overloaded_error body) get up to
      `max_attempts` tries. Default backoff schedule with the defaults
      above: 1s -> 4s -> 16s -> 60s -> 60s. Total patience ~2.5 min.
    - Non-transient errors (4xx, parse failures, etc.) re-raise on the
      first occurrence — no point retrying a 400 bad request.

    Used by both `LlmClient.complete()` (so every answer + judge call
    gets the same safety net) and by the rescore command. Tunable
    backoff_cap_sec keeps the final retries from sleeping multiple
    minutes when the provider is genuinely down.
    """
    backoff = initial_backoff_sec
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if _is_quota_error(e):
                raise ProviderQuotaExhausted(str(e)) from e
            if attempt < max_attempts and _is_transient(e):
                time.sleep(backoff)
                backoff = min(backoff * backoff_multiplier, backoff_cap_sec)
                continue
            raise
    # Unreachable — the loop always either returns or raises.
    raise RuntimeError("retry loop exhausted without returning or raising")


# ── Client implementation ─────────────────────────────────────────────────


class LlmClient:
    """One client per process — providers' SDKs are HTTP-thin so a
    single instance handles both answer + judge roles. Constructed
    lazily so an `import llm` from a unit test doesn't require API
    keys to be present.
    """

    def __init__(self) -> None:
        self._anthropic: object | None = None
        self._openai: object | None = None

    def complete(
        self,
        *,
        model: str,
        system: str | None,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LlmCall:
        """One call to the answer or judge model. Wrapped in
        `call_with_retry` so transient provider errors (Anthropic 529
        overloaded bursts, OpenAI 5xx, timeouts) don't bubble up as
        fatal failures — they retry up to 5 times with capped
        exponential backoff. Quota errors short-circuit as
        `ProviderQuotaExhausted` so the bench halts cleanly instead
        of burning more spend.
        """
        provider = _provider_for(model)

        def _call() -> LlmCall:
            if provider == "anthropic":
                return self._anthropic_complete(model, system, user, max_tokens, temperature)
            return self._openai_complete(model, system, user, max_tokens, temperature)

        return call_with_retry(_call)

    # ── Anthropic ─────────────────────────────────────────────────────────

    def _anthropic_client(self) -> object:
        if self._anthropic is None:
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set — required for Anthropic models. "
                    "Export it before running the bench."
                )
            self._anthropic = Anthropic(api_key=api_key)
        return self._anthropic

    def _anthropic_complete(
        self,
        model: str,
        system: str | None,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> LlmCall:
        client = self._anthropic_client()
        # Anthropic SDK signature: messages.create(model, system?, messages, max_tokens)
        kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)  # type: ignore[attr-defined]
        # Response shape: response.content is a list of TextBlock; we
        # join all text blocks (usually one).
        text_parts = [
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        return LlmCall(
            model=model,
            answer="".join(text_parts),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    # ── OpenAI ────────────────────────────────────────────────────────────

    def _openai_client(self) -> object:
        if self._openai is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set — required for OpenAI models. "
                    "Export it before running the bench."
                )
            self._openai = OpenAI(api_key=api_key)
        return self._openai

    def _openai_complete(
        self,
        model: str,
        system: str | None,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> LlmCall:
        client = self._openai_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = response.choices[0]
        usage = response.usage
        return LlmCall(
            model=model,
            answer=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


# ── Helpers ───────────────────────────────────────────────────────────────


def _provider_for(model: str) -> Provider:
    """Route by model-name prefix. We don't need anything fancier —
    Anthropic models all start with `claude-`; OpenAI's relevant ones
    start with `gpt-` or `o1-` / `o3-`."""
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    raise ValueError(
        f"Unknown model `{model}`. Add a prefix branch in llm._provider_for "
        "or rename the model string."
    )


def resolve_answer_model() -> str:
    return os.environ.get("SWB_ANSWER_MODEL", DEFAULT_ANSWER_MODEL)


def resolve_judge_model() -> str:
    return os.environ.get("SWB_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


# ── Unified answer prompt ─────────────────────────────────────────────────
#
# Every system-under-test eventually pastes its retrieved context into an
# answer prompt. Keeping the wording identical across adapters is a
# fairness control — otherwise prompt-engineering noise leaks into the
# cross-system comparison. In particular, the naive baseline previously
# omitted the "do not fabricate" instruction, which would have inflated
# its adversarial-refusal rate relative to the other systems.


def make_qa_prompt(*, context: str, question: str) -> str:
    """Single QA prompt used by every memory-using adapter.

    Same wording, same "do not fabricate" instruction, same section
    headers — so any difference in answer quality reflects the memory
    layer, not the prompt the adapter happened to write.
    """
    return (
        "Answer the question using the context below. If the answer "
        "isn't in the context, say so honestly — do not fabricate.\n\n"
        f"--- Context ---\n{context}\n\n"
        f"--- Question ---\n{question}"
    )


# ── Live provider health checks ───────────────────────────────────────────
#
# Cost-trivial (1 generated token per provider, ~$0.0001 each) live
# probes that catch the failure modes that would otherwise blow up
# mid-bench: missing keys, wrong keys, exhausted balance, region
# restrictions. Run via `swb config-check`.


@dataclass(frozen=True)
class ProviderCheck:
    """Outcome of one live provider probe."""

    provider: str
    model: str
    ok: bool
    detail: str
    # Tokens billed by the probe itself (1-token max generation). Surfaced
    # so the cost-estimator can add it to the total when reporting.
    probe_input_tokens: int = 0
    probe_output_tokens: int = 0


def check_anthropic_live(client: LlmClient | None = None) -> ProviderCheck:
    """Hit Anthropic with a 1-token generation against the answer
    model. Distinguishes missing-key (config_error) from low-balance
    (invalid_request_error w/ specific message) so the operator sees
    exactly what to fix."""
    model = resolve_answer_model()
    if _provider_for(model) != "anthropic":
        return ProviderCheck(
            provider="anthropic",
            model=model,
            ok=True,
            detail="(answer model is not Anthropic — skipped)",
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderCheck(
            provider="anthropic",
            model=model,
            ok=False,
            detail="ANTHROPIC_API_KEY not set",
        )
    c = client or LlmClient()
    try:
        result = c.complete(
            model=model,
            system=None,
            user="hi",
            max_tokens=1,
            temperature=0.0,
        )
        return ProviderCheck(
            provider="anthropic",
            model=model,
            ok=True,
            detail="ok",
            probe_input_tokens=result.input_tokens,
            probe_output_tokens=result.output_tokens,
        )
    except Exception as e:
        return ProviderCheck(
            provider="anthropic",
            model=model,
            ok=False,
            detail=_short_error(e),
        )


def check_openai_live(client: LlmClient | None = None) -> ProviderCheck:
    """Hit OpenAI with a 1-token generation against the judge model.
    Same distinction surface as Anthropic: missing key vs. quota
    exhausted vs. wrong project."""
    model = resolve_judge_model()
    if _provider_for(model) != "openai":
        return ProviderCheck(
            provider="openai",
            model=model,
            ok=True,
            detail="(judge model is not OpenAI — skipped)",
        )
    if not os.environ.get("OPENAI_API_KEY"):
        return ProviderCheck(
            provider="openai",
            model=model,
            ok=False,
            detail="OPENAI_API_KEY not set",
        )
    c = client or LlmClient()
    try:
        result = c.complete(
            model=model,
            system=None,
            user="hi",
            max_tokens=1,
            temperature=0.0,
        )
        return ProviderCheck(
            provider="openai",
            model=model,
            ok=True,
            detail="ok",
            probe_input_tokens=result.input_tokens,
            probe_output_tokens=result.output_tokens,
        )
    except Exception as e:
        return ProviderCheck(
            provider="openai",
            model=model,
            ok=False,
            detail=_short_error(e),
        )


def _short_error(err: object) -> str:
    """Compress an SDK error into a one-line summary fit for a
    table cell. SDK exceptions carry useful body text (e.g. Anthropic's
    'credit balance is too low') but the full repr is multi-line and
    full of request_ids the operator doesn't need to see in the
    summary view."""
    s = str(err)
    # Trim very long messages — operators can re-run with verbose
    # for the full thing if needed.
    if len(s) > 200:
        s = s[:200] + "…"
    # Replace newlines so the table cell stays one line.
    return s.replace("\n", " ")
