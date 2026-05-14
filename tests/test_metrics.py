"""Smoke tests for the deterministic, non-LLM code paths.

We deliberately don't test the system adapters (Mem0/Zep/Statewave/
naive/no_memory) here — those need real API keys + cost real money,
so they're operator-triggered via `uv run swb run --limit 1`.

What's tested here is the harness's "knows what it's doing" core:
F1 normalization, scoring dispatch, etc. If these regress, the bench
is mis-measuring everyone.
"""

from __future__ import annotations

import pytest

from statewave_bench.llm import (
    ProviderQuotaExhausted,
    _is_quota_error,
    _is_transient,
    call_with_retry,
)
from statewave_bench.metrics import (
    LLM_JUDGE_CATEGORIES,
    f1,
    normalize_text,
)

# Tests below were written against the old metrics-side names. Keep
# them passing via aliases so the test bodies don't need rewriting.
JudgeQuotaExhausted = ProviderQuotaExhausted
_call_judge_with_retry = call_with_retry


class TestNormalizeText:
    def test_lowercases(self) -> None:
        assert normalize_text("HELLO World") == "hello world"

    def test_strips_punctuation(self) -> None:
        assert normalize_text("Hello, world!") == "hello world"

    def test_drops_articles(self) -> None:
        assert normalize_text("the cat sat on a mat") == "cat sat on mat"
        assert normalize_text("an apple") == "apple"

    def test_collapses_whitespace(self) -> None:
        assert normalize_text("hello  \t\n  world") == "hello world"

    def test_combines_all(self) -> None:
        # SQuAD's reference normalization, end to end.
        assert normalize_text("The quick, brown fox!") == "quick brown fox"


class TestF1:
    def test_exact_match_is_one(self) -> None:
        assert f1("Paris", "Paris") == 1.0

    def test_normalized_match_is_one(self) -> None:
        # Different casing + punctuation but same tokens after normalize.
        assert f1("the Eiffel Tower", "Eiffel Tower!") == 1.0

    def test_disjoint_is_zero(self) -> None:
        assert f1("apple", "orange") == 0.0

    def test_partial_overlap(self) -> None:
        # 1 shared token out of 2 in pred and 2 in truth → P=0.5, R=0.5, F1=0.5
        score = f1("hello world", "hello there")
        assert score == pytest.approx(0.5)

    def test_empty_prediction_is_zero(self) -> None:
        assert f1("", "Paris") == 0.0

    def test_both_empty_is_one(self) -> None:
        # Edge case: ground truth is genuinely empty (e.g. "no answer"
        # cases). Empty prediction matches → 1.
        assert f1("", "") == 1.0

    def test_handles_articles_and_punctuation(self) -> None:
        # "the answer is paris" vs "Paris" → after normalize:
        #   pred: ["answer", "is", "paris"]   (articles dropped)
        #   truth: ["paris"]
        # Common = 1 → P=1/3, R=1/1, F1=2*(1/3)/(1/3 + 1) = 0.5
        score = f1("the answer is paris", "Paris")
        assert score == pytest.approx(0.5)


def test_llm_judge_categories_is_immutable() -> None:
    """Frozen set so the dispatch in score_answer can't be accidentally
    mutated by a caller and silently change which questions get
    scored by which metric."""
    assert isinstance(LLM_JUDGE_CATEGORIES, frozenset)
    assert "open_domain" in LLM_JUDGE_CATEGORIES
    assert "open_ended" in LLM_JUDGE_CATEGORIES


# ── Judge retry / quota detection ─────────────────────────────────────────


class TestErrorClassification:
    """Provider error strings -> right routing decision. The actual
    messages come from real OpenAI/Anthropic SDK exceptions seen in
    bench runs."""

    @pytest.mark.parametrize(
        "msg",
        [
            "Error code: 429 - {'error': {'message': 'You exceeded your current "
            "quota...', 'type': 'insufficient_quota'}}",
            "Error code: 400 - {'error': {'message': 'Your credit balance is too "
            "low to access the Anthropic API'}}",
            "billing_quota_exceeded: see plan",
        ],
    )
    def test_quota_markers_detected(self, msg: str) -> None:
        assert _is_quota_error(RuntimeError(msg))

    def test_quota_not_falsely_flagged(self) -> None:
        # 500 server errors mention "error" but aren't quota
        assert not _is_quota_error(RuntimeError("Error code: 500 - server_error"))

    @pytest.mark.parametrize(
        "msg",
        [
            "Error code: 500 - server had an error",
            "Error code: 503 - service unavailable",
            "Error code: 429 - rate limit (no quota marker)",
            "APITimeoutError: request timed out",
            "ConnectionError: connection refused",
        ],
    )
    def test_transient_patterns_detected(self, msg: str) -> None:
        assert _is_transient(RuntimeError(msg))

    def test_non_transient_not_flagged(self) -> None:
        # 400 (bad request), 401 (auth), 404 (not found) are permanent.
        assert not _is_transient(RuntimeError("Error code: 400 - bad request"))
        assert not _is_transient(RuntimeError("Error code: 401 - unauthorized"))
        assert not _is_transient(RuntimeError("ValueError: bad input"))


class TestCallJudgeWithRetry:
    """The retry helper's behavior under the four outcomes that matter."""

    def test_success_first_try(self) -> None:
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        assert _call_judge_with_retry(fn) == "ok"
        assert call_count == 1

    def test_retries_transient_then_succeeds(self) -> None:
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Error code: 500 - server overload")
            return "ok"

        # Use 0 backoff so the test is fast.
        result = _call_judge_with_retry(fn, initial_backoff_sec=0.0)
        assert result == "ok"
        assert call_count == 3

    def test_exhausts_retries_then_reraises(self) -> None:
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Error code: 500 - persistent")

        with pytest.raises(RuntimeError, match="500"):
            _call_judge_with_retry(fn, initial_backoff_sec=0.0)
        assert call_count == 5  # max_attempts default

    def test_quota_short_circuits_no_retry(self) -> None:
        """Quota errors must NOT retry — every subsequent call would
        fail the same way and burn more answer-model spend."""
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Error code: 429 - {'error': {'type': 'insufficient_quota'}}")

        with pytest.raises(JudgeQuotaExhausted):
            _call_judge_with_retry(fn, initial_backoff_sec=0.0)
        assert call_count == 1

    def test_non_transient_does_not_retry(self) -> None:
        """4xx (bad request, auth) shouldn't be retried — they're
        permanent until something changes."""
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Error code: 400 - bad request")

        with pytest.raises(RuntimeError, match="400"):
            _call_judge_with_retry(fn, initial_backoff_sec=0.0)
        assert call_count == 1
