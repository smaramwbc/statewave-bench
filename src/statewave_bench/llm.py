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
from dataclasses import dataclass
from typing import Literal

# Provider client imports stay top-level — these are library deps,
# always installed. The actual API calls only fire when an answer or
# judge call happens, so an `import llm` from inside a unit test
# doesn't need real keys.

# ── Default model selection ───────────────────────────────────────────────
# Override via env vars: SWB_ANSWER_MODEL, SWB_JUDGE_MODEL.
# Switch providers by changing the model string — the client
# auto-routes by name prefix.

DEFAULT_ANSWER_MODEL = "claude-sonnet-4-6"
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
        provider = _provider_for(model)
        if provider == "anthropic":
            return self._anthropic_complete(model, system, user, max_tokens, temperature)
        return self._openai_complete(model, system, user, max_tokens, temperature)

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
