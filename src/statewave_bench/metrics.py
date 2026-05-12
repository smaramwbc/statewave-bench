"""Scoring metrics for LoCoMo answers.

LoCoMo questions split into three scoring regimes (aligned with the
paper's reference evaluator — Section 4.2 of *Evaluating Very Long-Term
Conversational Memory of LLM Agents*, Snap Research 2024):

  1. **Short factoid (`single_hop`)** — ground truth is a name, date,
     or number. Scored with token-level F1 (SQuAD-style normalization).
     F1 is appropriate here because the truth is unambiguous and any
     correct answer overlaps the truth tokens.

  2. **Reasoning / open-ended (`multi_hop`, `temporal`, `open_domain`,
     `open_ended`)** — ground truth is a natural-language explanation
     ("Likely no; though she likes reading…"). F1 systematically
     penalizes verbose-but-correct paraphrases here — the model can
     answer correctly and still score ~0.10 because of low token
     overlap with the reference phrasing. We use LLM-as-judge: a
     separate model decides whether the prediction is semantically
     equivalent to the ground truth.

  3. **Adversarial** — the question asks about something not in the
     conversation; the correct behavior is to refuse. Ground truth is
     an empty string. Standard F1 returns 0 against any non-empty
     refusal text, which means EVERY correct refusal scores zero —
     a metric bug that previously made all systems look bad on this
     category. We use a refusal-detection judge: 1.0 if the prediction
     refuses to answer, 0.0 if it fabricates one.

Per-category aggregation: LoCoMo categorizes each question, so we
report scores per-category alongside the overall mean. A system that
crushes single-session questions but bombs temporal reasoning
shouldn't get to hide that under one global number.
"""

from __future__ import annotations

import re
import string
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .llm import LlmCall, LlmClient


class JudgeQuotaExhausted(RuntimeError):
    """The judge provider reports the operator's quota / credit balance
    is exhausted. Retries won't help — every subsequent judge call would
    fail the same way, and burning more answer-model spend on questions
    that can't be scored is wasteful. The runner catches this at the
    top level and halts the bench cleanly so the operator can refill
    and re-run with --resume.

    Triggered by provider error messages containing:
      - `insufficient_quota`    (OpenAI)
      - `quota_exceeded`        (OpenAI variant)
      - `credit balance is too low`  (Anthropic)
      - `billing_quota_exceeded`     (other)
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
# Provider 5xx (server overload), 429 without quota markers (rate limit),
# connection-level failures, timeouts. Matched against str(exc).
_TRANSIENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bError code:\s*5\d\d\b", re.IGNORECASE),
    re.compile(r"\bError code:\s*429\b", re.IGNORECASE),
    re.compile(r"\bConnectionError\b", re.IGNORECASE),
    re.compile(r"\bRemoteProtocolError\b", re.IGNORECASE),
    re.compile(r"\bAPITimeoutError\b", re.IGNORECASE),
    re.compile(r"\bAPIConnectionError\b", re.IGNORECASE),
    re.compile(r"\btimed?\s*out\b", re.IGNORECASE),
)


def _is_quota_error(err: BaseException) -> bool:
    s = str(err).lower()
    return any(marker.lower() in s for marker in _QUOTA_MARKERS)


def _is_transient(err: BaseException) -> bool:
    s = str(err)
    return any(p.search(s) for p in _TRANSIENT_PATTERNS)


T = TypeVar("T")


def _call_judge_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    initial_backoff_sec: float = 1.0,
    backoff_multiplier: float = 4.0,
) -> T:
    """Call `fn`, retrying on transient errors with exponential backoff.

    - Quota errors short-circuit immediately as JudgeQuotaExhausted.
    - Transient errors (5xx, 429-without-quota, connection/timeout) get
      up to `max_attempts` tries with backoff 1s -> 4s -> 16s.
    - Non-transient errors (4xx, parse failures, etc.) re-raise on the
      first occurrence.
    """
    backoff = initial_backoff_sec
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if _is_quota_error(e):
                raise JudgeQuotaExhausted(str(e)) from e
            if attempt < max_attempts and _is_transient(e):
                time.sleep(backoff)
                backoff *= backoff_multiplier
                continue
            raise
    # Unreachable — the loop always either returns or raises — but mypy
    # can't see it, so close the function explicitly.
    raise RuntimeError("retry loop exhausted without returning or raising")


# Categories where the right metric is semantic equivalence (LLM-judge),
# not token overlap. Empirically, F1 systematically under-rates correct
# verbose answers on these — see the docstring above for the paper's
# methodology alignment.
LLM_JUDGE_CATEGORIES = frozenset(
    {
        "open_domain",
        "open_ended",
        "multi_hop",
        "temporal",
    }
)

# Categories where the correct behavior is to refuse — ground truth is
# empty, F1 against empty truth always returns 0 for any non-empty
# prediction. We route these through a refusal-detection judge.
REFUSAL_CATEGORIES = frozenset({"adversarial"})


# ── Public types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Score:
    """One question's score. `value` is in [0, 1]; `metric` records
    which scoring path produced it so the report can break results
    out per-metric (some operators want F1-only numbers; others want
    the LLM-judged subset)."""

    value: float
    metric: str  # "f1" | "llm_judge" | "refusal_judge"


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
    llm.py) to reduce same-model-bias.

    Wrapped in `_call_judge_with_retry` — transient OpenAI 5xx /
    Anthropic 529 / connection failures are retried with backoff;
    quota exhaustion short-circuits as JudgeQuotaExhausted.
    """

    def _call() -> LlmCall:
        return llm.complete(
            model=model,
            system=_JUDGE_SYSTEM,
            user=_JUDGE_USER_TEMPLATE.format(
                question=question, truth=ground_truth, prediction=prediction
            ),
            max_tokens=8,
            temperature=0.0,
        )

    result = _call_judge_with_retry(_call)
    verdict = result.answer.strip().upper()
    # Be permissive on the parse — if the judge tacks on punctuation
    # or wraps the verdict, look for the keyword anywhere.
    if "CORRECT" in verdict and "INCORRECT" not in verdict:
        return 1.0
    return 0.0


# ── Refusal judge (adversarial) ───────────────────────────────────────────

_REFUSAL_JUDGE_SYSTEM = (
    "You are evaluating whether an answer correctly refuses a question that "
    "the model should NOT be able to answer (the relevant facts are not in "
    "its context).\n"
    "\n"
    "Reply with EXACTLY one of:\n"
    "  REFUSAL — the prediction declines to answer, says the information "
    "isn't available, isn't in the context, or acknowledges uncertainty "
    "rather than committing to a fact.\n"
    "  FABRICATION — the prediction commits to a specific factual answer "
    "(name, date, place, action, attribute) as if it were known.\n"
    "\n"
    "Hedged refusals still count as REFUSAL (e.g. 'the context doesn't "
    "mention X, but Y is a related fact'). The key is whether the model "
    "claimed a factual answer to the question being asked."
)

_REFUSAL_JUDGE_USER_TEMPLATE = "Question: {question}\nPrediction: {prediction}\n\nVerdict:"


def llm_judge_refusal(
    *,
    question: str,
    prediction: str,
    llm: LlmClient,
    model: str,
) -> float:
    """Score an adversarial answer. Returns 1.0 if the prediction
    refuses to answer (correct behavior — the fact isn't in the
    conversation), 0.0 if it fabricates a specific answer.

    Same retry semantics as `llm_judge`.
    """

    def _call() -> LlmCall:
        return llm.complete(
            model=model,
            system=_REFUSAL_JUDGE_SYSTEM,
            user=_REFUSAL_JUDGE_USER_TEMPLATE.format(question=question, prediction=prediction),
            max_tokens=8,
            temperature=0.0,
        )

    result = _call_judge_with_retry(_call)
    verdict = result.answer.strip().upper()
    if "REFUSAL" in verdict and "FABRICATION" not in verdict:
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
    if category in REFUSAL_CATEGORIES:
        value = llm_judge_refusal(
            question=question,
            prediction=prediction,
            llm=llm,
            model=judge_model,
        )
        return Score(value=value, metric="refusal_judge")
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
