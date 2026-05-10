"""End-to-end bench runner.

For every (system, conversation) pair: ingest the conversation, then
ask each question, score the answer, and append a result record. The
runner streams results to a JSONL file as it goes — partial runs
are recoverable, and a full publishable run can be resumed if it
hits a transient error halfway through (Anthropic 529s, Mem0 rate
limits, etc.).

Concurrency: deliberately serial. The per-question latency dominates
the runtime, but parallelism would make rate-limit handling messy
and cost reporting unstable. A 600-conversation run takes hours;
that's fine — operators run it once before publishing, not in CI.

Resumability: every result row carries (system, conversation_id,
question_idx). On restart, the runner reads the existing JSONL,
skips work that's already done, and picks up from the next gap. Idempotent.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from .dataset import LocomoConversation, LocomoQA
from .llm import LlmClient, resolve_judge_model
from .metrics import Score, score_answer
from .systems.base import AnswerResult, MemorySystem

console = Console()

# Fast-fail circuit-breaker: if N consecutive `answer` calls fail for
# the same (system, conversation), skip the rest of that conversation
# for that system. Saves the operator from burning the whole question
# set when a provider's API is down / out-of-balance / mis-keyed.
#
# 3 is conservative — one transient blip won't trip it; three in a
# row almost always means systemic. The runner logs the abort so it's
# visible in the JSONL trailer.
FAILURE_STREAK_THRESHOLD = 3


# ── Result records ────────────────────────────────────────────────────────


def _result_record(
    *,
    system: str,
    conversation_id: str,
    question_idx: int,
    qa: LocomoQA,
    answer: AnswerResult,
    score: Score,
) -> dict[str, Any]:
    """One JSONL row. Flat shape so polars / jq / pandas can all read
    it without nested-field gymnastics."""
    return {
        "system": system,
        "conversation_id": conversation_id,
        "question_idx": question_idx,
        "question": qa.question,
        "category": qa.category,
        "ground_truth": qa.answer,
        "prediction": answer.answer,
        "score": score.value,
        "metric": score.metric,
        "elapsed_ms": answer.elapsed_ms,
        "answer_model": answer.answer_model,
        "answer_input_tokens": answer.answer_input_tokens,
        "answer_output_tokens": answer.answer_output_tokens,
        "internal_input_tokens": answer.internal_input_tokens,
        "internal_output_tokens": answer.internal_output_tokens,
    }


# ── Resumability ──────────────────────────────────────────────────────────


def _load_completed_keys(path: Path) -> set[tuple[str, str, int]]:
    """Read prior result rows and return the set of (system, conv_id,
    question_idx) tuples already done. Skipped on next run."""
    if not path.exists():
        return set()
    completed: set[tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed rows; partial-write tolerance
            completed.add((row["system"], row["conversation_id"], row["question_idx"]))
    return completed


# ── Main loop ─────────────────────────────────────────────────────────────


def run_bench(
    *,
    systems: list[MemorySystem],
    conversations: Iterable[LocomoConversation],
    output_path: Path,
    llm: LlmClient | None = None,
) -> None:
    """Execute the bench. Streams results to `output_path` (JSONL)
    as it goes, skipping any (system, conv_id, question_idx) tuples
    already present in the file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = _load_completed_keys(output_path)
    if already_done:
        console.print(
            f"[yellow]Resume mode:[/] {len(already_done)} result rows already in "
            f"{output_path} — skipping those."
        )

    judge_llm = llm or LlmClient()
    judge_model = resolve_judge_model()

    # We materialize the conversation list so the progress bar has
    # an accurate total. LoCoMo is small (~600 entries); the memory
    # cost is fine.
    convs = list(conversations)
    total_questions = sum(len(c.qa) for c in convs) * len(systems)

    with (
        output_path.open("a", encoding="utf-8") as out_fh,
        Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("· {task.fields[stage]}"),
            console=console,
        ) as progress,
    ):
        task = progress.add_task(
            "Running bench",
            total=total_questions,
            stage="starting",
        )

        for conv in convs:
            for system in systems:
                stage = f"{system.name} / {conv.id} (ingest)"
                progress.update(task, stage=stage)
                # Skip ingest if every question for this (system, conv)
                # has already been scored — avoids paying ingest cost
                # twice on resume.
                conv_questions_done = sum(
                    1 for i in range(len(conv.qa)) if (system.name, conv.id, i) in already_done
                )
                if conv_questions_done < len(conv.qa):
                    try:
                        system.ingest(conv)
                    except NotImplementedError as e:
                        console.print(f"[red]System {system.name} ingest not implemented:[/] {e}")
                        progress.update(task, advance=len(conv.qa))
                        continue
                    except Exception as e:
                        console.print(
                            f"[red]System {system.name} ingest failed for "
                            f"conversation {conv.id}:[/] {e}"
                        )
                        progress.update(task, advance=len(conv.qa))
                        continue

                # Per-question loop. The failure streak counter resets
                # at the start of every (system, conversation) pair —
                # one system's dead API shouldn't kill the next system,
                # and a conversation that's been completed for one
                # system shouldn't poison the next conversation.
                failure_streak = 0
                aborted_for_streak = False
                for q_idx, qa in enumerate(conv.qa):
                    if (system.name, conv.id, q_idx) in already_done:
                        progress.update(task, advance=1)
                        continue
                    if failure_streak >= FAILURE_STREAK_THRESHOLD:
                        # Skip the rest of this conversation for this
                        # system. The runner already logged the
                        # individual failures via _run_one_question;
                        # surface the abort once here so the operator
                        # sees the circuit-breaker fired without
                        # scrolling through 200 identical errors.
                        if not aborted_for_streak:
                            console.print(
                                f"[red]Aborting {system.name} / {conv.id}[/] "
                                f"after {FAILURE_STREAK_THRESHOLD} consecutive "
                                f"answer failures — skipping remaining "
                                f"{len(conv.qa) - q_idx} questions for this pair."
                            )
                            aborted_for_streak = True
                        progress.update(task, advance=1)
                        continue
                    progress.update(
                        task,
                        stage=f"{system.name} / {conv.id} (q {q_idx + 1}/{len(conv.qa)})",
                    )
                    record = _run_one_question(
                        system=system,
                        conv_id=conv.id,
                        q_idx=q_idx,
                        qa=qa,
                        judge_llm=judge_llm,
                        judge_model=judge_model,
                    )
                    if record is None:
                        failure_streak += 1
                    else:
                        failure_streak = 0  # one success resets the breaker
                        out_fh.write(json.dumps(record) + "\n")
                        out_fh.flush()
                    progress.update(task, advance=1)

        progress.update(task, stage="done")


def _run_one_question(
    *,
    system: MemorySystem,
    conv_id: str,
    q_idx: int,
    qa: LocomoQA,
    judge_llm: LlmClient,
    judge_model: str,
) -> dict[str, Any] | None:
    """Ask one question, score the answer, return a JSONL row.
    Returns None on failure (failure is logged; caller advances
    the progress bar and moves on rather than aborting the whole
    run on one bad question)."""
    try:
        start = time.perf_counter()
        answer = system.answer(conv_id, qa.question)
        del start
    except NotImplementedError as e:
        console.print(f"[yellow]System {system.name} answer not implemented:[/] {e}")
        return None
    except Exception as e:
        console.print(
            f"[red]System {system.name} answer failed[/] (conv={conv_id}, q={q_idx}): {e}"
        )
        return None

    try:
        score = score_answer(
            question=qa.question,
            prediction=answer.answer,
            ground_truth=qa.answer,
            category=qa.category,
            llm=judge_llm,
            judge_model=judge_model,
        )
    except Exception as e:
        console.print(f"[red]Scoring failed[/] (conv={conv_id}, q={q_idx}): {e}")
        return None

    return _result_record(
        system=system.name,
        conversation_id=conv_id,
        question_idx=q_idx,
        qa=qa,
        answer=answer,
        score=score,
    )


__all__ = ["run_bench"]


# Silence "imported but unused" for asdict — we keep it imported so
# future result-record refactors that lean on dataclass introspection
# don't have to re-add it.
_ = asdict
