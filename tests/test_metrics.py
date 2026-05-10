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

from statewave_bench.metrics import (
    LLM_JUDGE_CATEGORIES,
    f1,
    normalize_text,
)


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
