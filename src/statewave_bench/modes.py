"""Benchmark comparison mode — `vendor_defaults` vs `equal_context_budget`.

The systems expose retrieval limits in different units (Statewave: a
token budget; Zep: a character budget / edge limit; Mem0: top_k). Two
honest things a reader can ask, and the bench must label which one a
result answers:

  - `vendor_defaults`  — each system at its out-of-the-box retrieval
    config. Answers "how do they compare as shipped?" The defaults are
    wildly unequal in context size, so this is NOT an equal-cost
    comparison and the report says so.

  - `equal_context_budget` — every system targets roughly the same
    answer-context size (~2k tokens by default). Answers "at a fixed
    context cost, which retrieves best?" The per-system knobs are
    coarse proxies, so the report always shows the *actual* measured
    context size per system alongside the score — the nominal target
    is a label, the measured size is the truth.

This module only resolves the mode and, for `equal_context_budget`,
seeds the per-system knob env vars to a documented default IF the
operator has not set them explicitly. It never overrides an explicit
operator value, and `scripts/budget_sweep.py` remains the tool for
sweeping the budget across tiers.
"""

from __future__ import annotations

import os

VENDOR_DEFAULTS = "vendor_defaults"
EQUAL_CONTEXT_BUDGET = "equal_context_budget"
VALID_MODES = (VENDOR_DEFAULTS, EQUAL_CONTEXT_BUDGET)

# Default equal-context target. ~2k tokens is enough for multi-fact
# synthesis without letting context-stuffing baselines run away.
DEFAULT_EQUAL_BUDGET_TOKENS = 2048

# Proxy calibration (same mapping scripts/budget_sweep.py documents):
# the knobs are not exact — get_context returns <= max_tokens, a Mem0
# memory is ~22 tok, a naive turn ~49 tok, a Zep edge ~24 tok. We seed
# values that land near the target; the report publishes the ACTUAL
# measured per-system context size so the approximation is auditable.
_TOK_PER_MEM0_MEMORY = 22
_TOK_PER_NAIVE_TURN = 49
_TOK_PER_ZEP_EDGE = 24
_ZEP_TOKEN_CEILING = 1180  # graph.search caps ~50 edges; can't exceed this


def resolve_mode(cli_mode: str | None = None) -> str:
    """CLI flag > SWB_BENCH_MODE env > vendor_defaults."""
    mode = (cli_mode or os.environ.get("SWB_BENCH_MODE") or VENDOR_DEFAULTS).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown bench mode {mode!r}. Valid: {', '.join(VALID_MODES)}.")
    return mode


def apply_mode_env(
    mode: str, *, target_tokens: int = DEFAULT_EQUAL_BUDGET_TOKENS
) -> dict[str, str]:
    """For `equal_context_budget`, seed each system's budget knob to a
    documented default unless the operator already set it. Returns the
    env keys this call set (for logging). No-op for `vendor_defaults`.
    """
    os.environ["SWB_BENCH_MODE"] = mode
    if mode != EQUAL_CONTEXT_BUDGET:
        return {}

    seeded: dict[str, str] = {}

    def _seed(var: str, value: str) -> None:
        if not os.environ.get(var):
            os.environ[var] = value
            seeded[var] = value

    _seed("SWB_STATEWAVE_CONTEXT_MAX_TOKENS", str(target_tokens))
    _seed("MEM0_TOP_K", str(max(1, round(target_tokens / _TOK_PER_MEM0_MEMORY))))
    _seed("SWB_NAIVE_WINDOW_SIZE", str(max(1, round(target_tokens / _TOK_PER_NAIVE_TURN))))
    # Zep cannot exceed its graph.search ceiling — clamp, and the
    # report/methodology note that Zep is under-budget above ~1,180 tok.
    zep_limit = min(50, max(1, round(min(target_tokens, _ZEP_TOKEN_CEILING) / _TOK_PER_ZEP_EDGE)))
    _seed("SWB_ZEP_SEARCH_LIMIT", str(zep_limit))
    _seed("SWB_ZEP_SEARCH_MAX_CHARS", str(target_tokens * 4))
    return seeded
