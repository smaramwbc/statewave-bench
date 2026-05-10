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

DEFAULT_ANSWER_MODEL = "claude-3-5-sonnet-20241022"
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
