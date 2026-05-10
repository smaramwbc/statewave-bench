"""LoCoMo loader.

LoCoMo (Long-form Conversational Memory) is a public benchmark from
Snap Research — see https://github.com/snap-research/LoCoMo for the
upstream and the paper "Evaluating Very Long-Term Conversational
Memory of LLM Agents" (2024).

The dataset ships ~600 multi-session conversations, each paired with
factual recall questions probing memory across sessions. We load it
once via Hugging Face's `datasets` library and cache locally; the bench
runner then iterates conversations and submits each question through
each system under test.

Why HuggingFace as the source: it's where Snap's official mirror lives,
the cache is content-addressable so reruns don't re-download, and it
sidesteps the upstream repo's manual setup script.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Default Hugging Face dataset id. Override via `LOCOMO_DATASET_ID` env
# var if Snap re-publishes elsewhere or you have a private fork (e.g.
# for reproducibility-pinned snapshots).
DEFAULT_LOCOMO_DATASET_ID = "snap-stanford/LoCoMo"


@dataclass(frozen=True)
class LocomoTurn:
    """One conversational turn within a LoCoMo session."""

    speaker: str
    text: str
    # ISO date / freeform timestamp string per the upstream schema.
    # Some entries have only a session-level date; we surface whatever
    # the upstream payload provided rather than parsing into a real
    # datetime — callers that need that handle it.
    timestamp: str | None


@dataclass(frozen=True)
class LocomoQA:
    """One factual recall question paired with a ground-truth answer.

    The eval prompts each system with `question`, then scores the
    response against `answer` (string-match for exact-answer questions,
    LLM-as-judge for open-ended ones — the metric is selected by
    `category`).
    """

    question: str
    answer: str
    # LoCoMo categorizes questions: 'single_session' / 'multi_session'
    # / 'temporal_reasoning' / 'open_domain' / 'adversarial'. The bench
    # runner reports per-category scores so a system winning overall
    # but losing temporal reasoning is visible.
    category: str
    # Some entries point at the specific session(s) that contain the
    # answer; useful for audit and for upper-bound retrieval analysis.
    evidence_session_ids: tuple[int, ...] | None


@dataclass(frozen=True)
class LocomoConversation:
    """A full multi-session conversation plus its question set."""

    id: str
    sessions: tuple[tuple[LocomoTurn, ...], ...]
    qa: tuple[LocomoQA, ...]


def load_locomo(
    *,
    dataset_id: str = DEFAULT_LOCOMO_DATASET_ID,
    cache_dir: Path | None = None,
    split: str = "test",
    limit: int | None = None,
) -> Iterator[LocomoConversation]:
    """Yield LoCoMo conversations one at a time.

    `limit` caps the iteration — set it to a small number (e.g. 50)
    for the pilot run to keep API costs bounded; leave `None` for the
    publishable full run. The cache_dir defaults to `data/locomo/` so
    the dataset is downloaded once per repo clone.

    Implementation note: this function does the upstream → typed
    conversion in one place so every system adapter consumes the same
    typed shape regardless of upstream schema drift. If LoCoMo
    publishes a v2 schema, the change lives here.
    """
    # Imported lazily so `import statewave_bench.cli` doesn't pay the
    # dataset-lib import cost (it's heavy and only relevant when an
    # actual run is happening).
    from datasets import load_dataset

    raw = load_dataset(
        dataset_id,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    for yielded, row in enumerate(raw):
        if limit is not None and yielded >= limit:
            return
        yield _row_to_conversation(row)


def _row_to_conversation(row: dict[str, Any]) -> LocomoConversation:
    """Translate one raw LoCoMo row into our typed shape.

    LoCoMo's upstream schema has nested sessions + a flat QA list.
    We're permissive on missing fields — the bench should keep running
    if the dataset gains a field; the unfamiliar field just gets
    dropped on the floor.
    """
    sessions_raw = row.get("sessions") or []
    sessions = tuple(tuple(_turn(t) for t in (session or [])) for session in sessions_raw)
    qa_raw = row.get("qa") or row.get("questions") or []
    qa = tuple(_qa(q) for q in qa_raw)
    return LocomoConversation(
        id=str(row.get("id") or row.get("sample_id") or ""),
        sessions=sessions,
        qa=qa,
    )


def _turn(t: dict[str, Any]) -> LocomoTurn:
    return LocomoTurn(
        speaker=str(t.get("speaker") or t.get("role") or ""),
        text=str(t.get("text") or t.get("content") or ""),
        timestamp=t.get("timestamp") or t.get("date"),
    )


def _qa(q: dict[str, Any]) -> LocomoQA:
    evidence = q.get("evidence") or q.get("session_ids") or None
    if isinstance(evidence, list):
        evidence_t: tuple[int, ...] | None = tuple(int(e) for e in evidence)
    else:
        evidence_t = None
    return LocomoQA(
        question=str(q.get("question") or ""),
        answer=str(q.get("answer") or q.get("ground_truth") or ""),
        category=str(q.get("category") or q.get("type") or "uncategorized"),
        evidence_session_ids=evidence_t,
    )
