"""Per-provider price table + estimator.

The bench operator wants to see an estimated total cost BEFORE
committing to a real run. We keep the price table small + explicit so
operators can sanity-check each line — and so updating prices when
the providers rev them is a one-file change.

Prices are USD per 1M tokens, sourced from each provider's pricing
page at the times noted in the comments. The estimator surfaces a
range (low / high) because actual token counts vary per question:
some retrievals fit 1K tokens of context, some fit 4K.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Per-model prices, USD per 1M tokens, January 2026 ────────────────────
#
# Update these when the providers do. Operators reading the README's
# cost estimates should be confident the numbers are current.

PRICES_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    # model_name : (input_price, output_price)
    # Claude 4.x family — current default lineup.
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    # OpenAI judge models.
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

# ── LoCoMo characteristics, derived from the dataset itself ──────────────
#
# These are observed values from the locomo10.json snapshot we use. If
# upstream re-publishes with a different shape, the estimator's output
# will drift; the README documents how to refresh.

QUESTIONS_PER_CONVERSATION = 199  # observed
SESSIONS_PER_CONVERSATION = 19  # observed
FRACTION_OPEN_DOMAIN = 70 / 199  # observed; only these trigger judge calls

# ── Per-call token-budget envelopes (observed during smoke planning) ─────

# Answer-side: assembled context + the question itself, on average.
# Real conversations vary; the bench's actual results carry the true
# per-call counts so post-hoc operators can re-derive cost.
AVG_ANSWER_INPUT_TOKENS_LOW = 2000
AVG_ANSWER_INPUT_TOKENS_HIGH = 4000
AVG_ANSWER_OUTPUT_TOKENS = 256

# Judge-side: question + ground-truth + prediction in a short envelope.
# CORRECT/INCORRECT is one token; we round to a small output budget.
AVG_JUDGE_INPUT_TOKENS = 500
AVG_JUDGE_OUTPUT_TOKENS = 8

# Mem0 internal fact-extraction LLM cost per ingested session. Mem0
# does roughly one OpenAI call per `add()`; the model + prompt size
# isn't documented but ~1500 input + ~200 output is a reasonable
# envelope per their published examples.
MEM0_INTERNAL_INPUT_PER_SESSION = 1500
MEM0_INTERNAL_OUTPUT_PER_SESSION = 200
# Mem0 routes its internal calls through gpt-4o-mini by default in
# the cloud — operators self-hosting can override and the cost shifts.
MEM0_INTERNAL_MODEL = "gpt-4o-mini"


# ── Output type ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CostEstimate:
    """Per-provider cost ranges for a planned run. `total_low` /
    `total_high` are the sums callers usually want; the per-line
    breakdown is useful for "where is my money going?" debugging."""

    n_conversations: int
    n_systems: int
    answer_model: str
    judge_model: str
    # Per-provider USD, low + high
    anthropic_low: float
    anthropic_high: float
    openai_judge_low: float
    openai_judge_high: float
    openai_mem0_internal: float  # single-point estimate
    statewave_internal: float  # single-point estimate (LLM compile if enabled)
    total_low: float
    total_high: float


# ── Estimator ────────────────────────────────────────────────────────────


def estimate(
    *,
    n_conversations: int,
    n_systems: int,
    answer_model: str,
    judge_model: str,
    include_mem0: bool = True,
    statewave_llm_compile: bool = False,
) -> CostEstimate:
    """Return a cost range for the planned run.

    `include_mem0` defaults true — operators usually run Mem0 in the
    system set; flip false when they're benchmarking only baselines or
    only Statewave. `statewave_llm_compile` defaults false (heuristic
    compiler costs zero externally); flip true when the Statewave server
    is configured with the LLM compiler so its internal cost shows up.
    """
    # Per-system, per-conversation question count.
    questions = n_conversations * QUESTIONS_PER_CONVERSATION
    # Every system answers every question.
    answer_calls = questions * n_systems
    # Only open_domain questions trigger judge calls.
    judge_calls = int(questions * FRACTION_OPEN_DOMAIN) * n_systems

    # Answer cost — assume the answer model is on Anthropic for now.
    # If operators swap to GPT-4o for the answer, the table line shifts
    # to OpenAI; the bench's pricing dict handles both transparently.
    ans_in_low, ans_out = (
        answer_calls * AVG_ANSWER_INPUT_TOKENS_LOW,
        answer_calls * AVG_ANSWER_OUTPUT_TOKENS,
    )
    ans_in_high = answer_calls * AVG_ANSWER_INPUT_TOKENS_HIGH
    ans_price_in, ans_price_out = _price_for(answer_model)

    anthropic_low = _usd(ans_in_low, ans_price_in) + _usd(ans_out, ans_price_out)
    anthropic_high = _usd(ans_in_high, ans_price_in) + _usd(ans_out, ans_price_out)

    # Judge cost — fixed envelope so just low=high.
    judge_in = judge_calls * AVG_JUDGE_INPUT_TOKENS
    judge_out = judge_calls * AVG_JUDGE_OUTPUT_TOKENS
    judge_price_in, judge_price_out = _price_for(judge_model)
    judge_total = _usd(judge_in, judge_price_in) + _usd(judge_out, judge_price_out)

    # Mem0's internal LLM cost (operator's OpenAI bill).
    mem0_internal = 0.0
    if include_mem0:
        mem0_in = n_conversations * SESSIONS_PER_CONVERSATION * MEM0_INTERNAL_INPUT_PER_SESSION
        mem0_out = n_conversations * SESSIONS_PER_CONVERSATION * MEM0_INTERNAL_OUTPUT_PER_SESSION
        mp_in, mp_out = _price_for(MEM0_INTERNAL_MODEL)
        mem0_internal = _usd(mem0_in, mp_in) + _usd(mem0_out, mp_out)

    # Statewave LLM compile cost (server's bill against operator's
    # provider key). Heuristic compile = 0.
    statewave_internal = 0.0
    if statewave_llm_compile:
        # One compile per conversation, ~3K input + ~500 output on
        # gpt-4o-mini by default (server-configurable).
        sw_in_tok = n_conversations * 3000
        sw_out_tok = n_conversations * 500
        sp_in, sp_out = _price_for("gpt-4o-mini")
        statewave_internal = _usd(sw_in_tok, sp_in) + _usd(sw_out_tok, sp_out)

    total_low = anthropic_low + judge_total + mem0_internal + statewave_internal
    total_high = anthropic_high + judge_total + mem0_internal + statewave_internal

    return CostEstimate(
        n_conversations=n_conversations,
        n_systems=n_systems,
        answer_model=answer_model,
        judge_model=judge_model,
        anthropic_low=anthropic_low,
        anthropic_high=anthropic_high,
        openai_judge_low=judge_total,
        openai_judge_high=judge_total,
        openai_mem0_internal=mem0_internal,
        statewave_internal=statewave_internal,
        total_low=total_low,
        total_high=total_high,
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _price_for(model: str) -> tuple[float, float]:
    """Look up (input_price, output_price) per 1M tokens. Falls back to
    a deliberately conservative estimate if the model isn't in the
    table — operators see the unknown-model warning in the CLI output."""
    if model in PRICES_PER_MILLION_TOKENS:
        return PRICES_PER_MILLION_TOKENS[model]
    # Unknown model: assume premium-tier pricing so the estimate
    # overstates rather than understates. The CLI prints a warning so
    # operators can update the table when they see it.
    return (5.00, 20.00)


def _usd(tokens: int, price_per_million: float) -> float:
    return (tokens / 1_000_000) * price_per_million
