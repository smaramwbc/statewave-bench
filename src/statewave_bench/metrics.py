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

import os
import re
import string
from collections import Counter
from dataclasses import dataclass

from .llm import LlmClient, ProviderQuotaExhausted

# Backward-compat alias — runner imports this name; under the hood it's
# the provider-level exception from llm.py (raised by both answer and
# judge calls via call_with_retry). Kept so the runner's halt-on-quota
# handler stays unchanged when the answer-call retry was added.
JudgeQuotaExhausted = ProviderQuotaExhausted


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

# Permissive judge — byte-stable port of the upstream Mem0 LoCoMo judge
# template (the LongMem-style prompt that Mem0 / Honcho / Backboard /
# Memori all use to produce their published SOTA numbers). Use this mode
# (set SWB_SCORING_MODE=permissive) when reporting numbers meant to be
# compared against published SOTA.
#
# Key behaviors of this prompt that drive its leniency vs our strict
# default:
#   - explicit "be generous with your grading" instruction (repeated 4x)
#   - "touches on the same topic as the gold answer" suffices for CORRECT
#   - format differences explicitly forgiven (e.g. "May 7th" = "7 May")
#   - relative time references accepted ("last Tuesday")
# These match what locomo-audit measured: standard LongMem-style judge
# accepts up to 63% of *intentionally wrong* answers.
#
# Mem0 runs this prompt 10x with majority vote per question — we don't
# (single-call is enough to measure the prompt-vs-prompt methodology
# spread, and 10x would multiply judge cost by 10x for marginal noise
# reduction). The published-vs-us comparison is "Mem0-prompt single
# call" vs "our-prompt single call" — apples-to-apples on the prompt
# axis, with majority-vote noise reduction as a known unknown.
#
# Source: github.com/rtuosto/agent-memory-benchmark
# (byte-stable port of Mem0's upstream template — fingerprinted &
# tested against drift).
_PERMISSIVE_JUDGE_SYSTEM = ""  # Mem0's template puts everything in user msg

_PERMISSIVE_JUDGE_USER_TEMPLATE = (
    'Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given\n'
    "the following data: (1) a question (posed by one user to another user), (2) a 'gold'\n"
    "(ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.\n"
    "\n"
    "The point of the question is to ask about something one user should know about the other\n"
    "user based on their prior conversations. The gold answer will usually be a concise and\n"
    "short answer that includes the referenced topic, for example:\n"
    "\n"
    "Question: Do you remember what I got the last time I went to Hawaii?\n"
    "Gold answer: A shell necklace\n"
    "\n"
    "The generated answer might be much longer, but you should be generous with your "
    "grading -\n"
    "as long as it touches on the same topic as the gold answer, it should be counted as "
    "CORRECT.\n"
    "\n"
    "For time related questions, the gold answer will be a specific date, month, year, etc. "
    "The\n"
    "generated answer might be much longer or use relative time references (like 'last "
    "Tuesday'\n"
    "or 'next month'), but you should be generous with your grading - as long as it refers "
    "to the\n"
    "same date or time period as the gold answer, it should be counted as CORRECT. Even if "
    "the\n"
    "format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same "
    "date.\n"
    "\n"
    "Now it's time for the real question:\n"
    "\n"
    "Question: {question}\n"
    "Gold answer: {truth}\n"
    "Generated answer: {prediction}\n"
    "\n"
    "First, provide a short (one sentence) explanation of your reasoning, then finish "
    "with\n"
    "CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it "
    "will\n"
    "break the evaluation script.\n"
    "\n"
    'Just return the label CORRECT or WRONG in a json format with the key as "label".'
)


def llm_judge(
    *,
    question: str,
    prediction: str,
    ground_truth: str,
    llm: LlmClient,
    model: str,
    permissive: bool = False,
) -> float:
    """Score an open-ended answer via a separate LLM. Returns 1.0 if
    the judge says CORRECT, 0.0 otherwise. The judge model is
    deliberately different from the answer model (config'd in
    llm.py) to reduce same-model-bias.

    When `permissive=True`, swap to the verbatim Mem0/LongMem judge
    template (see `_PERMISSIVE_JUDGE_USER_TEMPLATE` docstring). The
    Mem0 template emits CORRECT/WRONG (not CORRECT/INCORRECT) and
    is instructed to provide a one-sentence explanation followed by
    a JSON `{"label": "CORRECT"}` — we parse permissively for either
    a JSON label or a bare CORRECT/WRONG token in the response.
    """
    if permissive:
        result = llm.complete(
            model=model,
            system=None,
            user=_PERMISSIVE_JUDGE_USER_TEMPLATE.format(
                question=question, truth=ground_truth, prediction=prediction
            ),
            # 200 tokens covers the "one sentence + verdict + json" the
            # Mem0 template asks for. 8 (our strict-mode budget) would
            # truncate before the verdict lands.
            max_tokens=200,
            temperature=0.0,
        )
        verdict_text = result.answer.strip().upper()
        # The template instructs the judge to emit JSON `{"label":
        # "CORRECT"}` — but in practice judges sometimes drop the JSON
        # wrapper and just write "CORRECT" / "WRONG". Accept either.
        # "WRONG" in raw text wins if both tokens appear (defensive —
        # the prompt explicitly says don't include both).
        if "WRONG" in verdict_text:
            return 0.0
        if "CORRECT" in verdict_text:
            return 1.0
        return 0.0

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

    Retry semantics inherited from `LlmClient.complete`.
    """
    result = llm.complete(
        model=model,
        system=_REFUSAL_JUDGE_SYSTEM,
        user=_REFUSAL_JUDGE_USER_TEMPLATE.format(question=question, prediction=prediction),
        max_tokens=8,
        temperature=0.0,
    )
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
    same client + cache.

    Two scoring modes (selected via SWB_SCORING_MODE):
      - `strict` (default): F1 for single_hop, strict LLM-judge for
        multi_hop/temporal/open_*, refusal-judge for adversarial. Our
        canonical methodology — defensible and slightly under-reports
        relative to lenient SOTA harnesses.
      - `permissive`: LLM-judges EVERY category (including single_hop
        and adversarial against an empty truth) with a generous prompt
        that accepts paraphrase, partial-overlap, hedged answers, and
        format variation. This matches the LongMem / public-SOTA
        scoring pattern. Use for apples-to-apples comparison against
        Mem0 91.6% / Honcho 89.9% / Backboard 90.1% / Memori 82%.
    """
    mode = os.environ.get("SWB_SCORING_MODE", "strict").lower()
    permissive = mode == "permissive"

    if category in REFUSAL_CATEGORIES:
        # Adversarial questions are intentionally unanswerable — ground
        # truth is empty, and the correct behavior is to refuse. The
        # refusal judge is the right metric regardless of strict /
        # permissive mode. (Public-SOTA harnesses drop adversarial
        # entirely; the report layer surfaces an adversarial-excluded
        # mean for apples-to-apples comparison.)
        value = llm_judge_refusal(
            question=question,
            prediction=prediction,
            llm=llm,
            model=judge_model,
        )
        return Score(value=value, metric="refusal_judge")
    if permissive or category in LLM_JUDGE_CATEGORIES:
        value = llm_judge(
            question=question,
            prediction=prediction,
            ground_truth=ground_truth,
            llm=llm,
            model=judge_model,
            permissive=permissive,
        )
        return Score(value=value, metric="llm_judge")
    return Score(value=f1(prediction, ground_truth), metric="f1")
