"""Scoring metrics for LoCoMo answers.

LoCoMo questions split into two scoring regimes:

  1. **Exact-answer questions** — the ground-truth answer is short
     and unambiguous (a name, a date, a number). We score with token-
     level F1, the standard SQuAD-derived metric: tokens are
     normalized (lowercased, stripped of punctuation/articles), then
     precision + recall + F1 are computed against the prediction.
     Identical to LoCoMo's official evaluation script and to the
     SQuAD reference implementation.

  2. **Open-ended questions** — the answer can be phrased many ways.
     String-matching here would systematically penalize long
     paraphrases. We use LLM-as-judge: a separate model (the
     `judge_model`, deliberately different from the answer model) is
     asked "is this prediction equivalent to the ground truth, given
     the question?" and returns a 0/1 verdict.

Per-category aggregation: LoCoMo categorizes each question, so we
report F1 per-category alongside the overall score. A system that
crushes single-session questions but bombs temporal reasoning
shouldn't get to hide that under one global number.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

from .llm import LlmClient

# LoCoMo categories that should be scored with LLM-as-judge rather
# than F1. The category names match upstream's labels; categories not
# listed default to F1.
LLM_JUDGE_CATEGORIES = frozenset({"open_domain", "open_ended"})


# ── Public types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Score:
    """One question's score. `value` is in [0, 1]; `metric` records
    which scoring path produced it so the report can break results
    out per-metric (some operators want F1-only numbers; others want
    the LLM-judged subset)."""

    value: float
    metric: str  # "f1" | "llm_judge"


# ── F1 (SQuAD-style) ──────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """SQuAD's standard normalization: lowercase, drop punctuation,
    drop English articles, collapse whitespace. Verbatim from the
    SQuAD reference eval script — same code LoCoMo's official
    evaluator runs."""
    text = text.lower()
    text = "".join(c for c in text if c not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1. Returns 0 when either side is empty after
    normalization (avoids ZeroDivisionError + reflects 'system said
    nothing useful')."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        # Both empty → consider it a match (mostly relevant for QA
        # cases where the ground truth is genuinely "no answer").
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


# ── LLM-as-judge ──────────────────────────────────────────────────────────


_JUDGE_SYSTEM = (
    "You are an evaluator. The user will give you a question, a ground-truth "
    "answer, and a predicted answer. Decide whether the prediction is "
    "semantically equivalent to the ground truth as an answer to the question.\n"
    "\n"
    "Reply with EXACTLY one of:\n"
    "  CORRECT — prediction conveys the same factual content as the ground truth\n"
    "  INCORRECT — prediction misses, contradicts, or fabricates relevant facts\n"
    "\n"
    "Be strict on factual content (names, dates, numbers must match) but lenient "
    "on phrasing (paraphrases, synonyms, and added context that doesn't contradict "
    "the ground truth all count as CORRECT). If the prediction says it doesn't "
    "know AND the ground truth is non-empty, that's INCORRECT."
)

_JUDGE_USER_TEMPLATE = (
    "Question: {question}\nGround truth: {truth}\nPrediction: {prediction}\n\nVerdict:"
)


def llm_judge(
    *,
    question: str,
    prediction: str,
    ground_truth: str,
    llm: LlmClient,
    model: str,
) -> float:
    """Score an open-ended answer via a separate LLM. Returns 1.0 if
    the judge says CORRECT, 0.0 otherwise. The judge model is
    deliberately different from the answer model (config'd in
    llm.py) to reduce same-model-bias."""
    result = llm.complete(
        model=model,
        system=_JUDGE_SYSTEM,
        user=_JUDGE_USER_TEMPLATE.format(
            question=question, truth=ground_truth, prediction=prediction
        ),
        max_tokens=8,
        temperature=0.0,
    )
    verdict = result.answer.strip().upper()
    # Be permissive on the parse — if the judge tacks on punctuation
    # or wraps the verdict, look for the keyword anywhere.
    if "CORRECT" in verdict and "INCORRECT" not in verdict:
        return 1.0
    return 0.0


# ── Dispatch ──────────────────────────────────────────────────────────────


def score_answer(
    *,
    question: str,
    prediction: str,
    ground_truth: str,
    category: str,
    llm: LlmClient,
    judge_model: str,
) -> Score:
    """Score one answer, picking the right metric for the question
    category. Caller passes a shared `llm` so judge calls reuse the
    same client + cache."""
    if category in LLM_JUDGE_CATEGORIES:
        value = llm_judge(
            question=question,
            prediction=prediction,
            ground_truth=ground_truth,
            llm=llm,
            model=judge_model,
        )
        return Score(value=value, metric="llm_judge")
    return Score(value=f1(prediction, ground_truth), metric="f1")
