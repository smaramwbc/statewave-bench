"""The contract every memory system implements.

Each adapter (Statewave, Mem0, Zep, naive baseline, no-memory baseline)
implements `MemorySystem`. The bench runner doesn't know or care which
system it's talking to — it just feeds conversations in via `ingest`,
then asks each question via `answer` and scores the response.

Two-phase shape on purpose:

  - `ingest` runs once per conversation BEFORE any questions are
    asked, so systems can compile their memory representation
    (Statewave's compile loop, Mem0's facts extraction, Zep's
    knowledge graph). This mirrors the actual production flow:
    memory is built ahead of query time.

  - `answer` runs once per question, with NO conversation context
    passed in directly — the system has to retrieve from its own
    memory. This is the test: did your memory layer actually
    capture the relevant fact?

We deliberately avoid a 'reset' between questions within a
conversation: real-world agents accumulate memory across an entire
session. To test cold-start retrieval, run the bench twice with
different conversation orderings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..dataset import LocomoConversation


@dataclass(frozen=True)
class AnswerResult:
    """One answer produced by a system under test.

    Token counts are the answer-model's usage for the final response;
    they do NOT include any LLM calls the system itself made
    internally during ingest (Mem0's fact-extraction calls, Zep's
    knowledge-graph compile, Statewave's LLM compile if configured).
    Those internal costs show up in `internal_input_tokens` /
    `internal_output_tokens` for systems that report them; for
    systems that don't, those fields stay at 0 and the bench reports
    only externally-observable cost.
    """

    answer: str
    answer_model: str
    answer_input_tokens: int
    answer_output_tokens: int
    # Wall-clock time spent in the system's `answer` call (retrieve +
    # generate). Reported to operators so the speed/quality tradeoff
    # is visible.
    elapsed_ms: float
    # Optional: the actual context the system passed to the answer
    # model. Captured for `--show-context` debugging; not used in
    # scoring. Stored as the joined string the model saw, NOT the
    # structured retrieval result (some systems expose only the
    # joined form).
    retrieved_context: str | None = None
    # Optional internal-cost reporting. Systems that issue LLM calls
    # during ingest or retrieval (Mem0's fact extractor, Statewave's
    # LLM compile) populate these so an honest "total cost" is
    # reportable. Defaults zero — the bench distinguishes 'reported
    # zero' from 'didn't report' via per-system metadata.
    internal_input_tokens: int = 0
    internal_output_tokens: int = 0
    # Optional: how many discrete memory/context items the system put in
    # front of the answer model (facts, edges, turns, chunks). Reported
    # so a reader can see "system A used 30 items, system B used 4" even
    # at the same token budget. None = the adapter didn't report it; the
    # runner still derives char/token size from `retrieved_context`.
    retrieved_items_count: int | None = None


class MemorySystem(ABC):
    """Adapter contract — one subclass per system under test."""

    #: Stable identifier used in result tables and chart legends.
    #: e.g. "statewave", "mem0", "zep", "naive", "no_memory".
    name: str

    @abstractmethod
    def ingest(self, conversation: LocomoConversation) -> None:
        """Feed every session of a conversation into the system's
        memory layer. Idempotent for the same conversation.id —
        re-running the bench in pieces shouldn't double-count."""

    @abstractmethod
    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        """Retrieve from memory + produce a final answer.

        The bench passes the question only — the system is expected
        to pull whatever it needs from its own memory layer keyed by
        `conversation_id`. If the system's API insists on session
        ids or subject ids, the adapter normalizes that internally."""

    def reset(self) -> None:
        """Optional — clear all stored memory.

        Default no-op; in-memory systems implement this so a single
        process can run many conversations cleanly. Cloud systems
        usually skip resetting (the bench scopes by conversation_id)
        but can offer a `--reset-cloud` opt-in flag for full purges."""
        return None

    def health_check(self) -> HealthResult:
        """Optional — verify the system is reachable BEFORE the bench
        burns money on a doomed run.

        Default: assume ok (operators who care about pre-flight
        validation override per-system with a cheap read call —
        Mem0 + Zep list users, Statewave probes a known-nonexistent
        subject for connectivity, baselines no-op). Feeds
        `swb config-check`'s output table."""
        return HealthResult(ok=True, detail="(no live health check implemented)")


@dataclass(frozen=True)
class HealthResult:
    """Outcome of a per-system health probe. The detail string is
    operator-facing — keep it one-line, no stack traces."""

    ok: bool
    detail: str
