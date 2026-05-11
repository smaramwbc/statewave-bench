"""'No memory' baseline — same answer model, zero history.

Establishes the floor: how well does the model do with just the
question, no context at all? Multi-session questions should bomb;
common-knowledge questions should still pass. The gap between this
floor and any real memory system is "the memory contribution."
"""

from __future__ import annotations

import time

from ..dataset import LocomoConversation
from ..llm import LlmClient, resolve_answer_model
from .base import AnswerResult, HealthResult, MemorySystem


class NoMemorySystem(MemorySystem):
    name = "no_memory"

    def __init__(self) -> None:
        self._llm = LlmClient()

    def ingest(self, conversation: LocomoConversation) -> None:
        # The whole point: no ingestion happens.
        return None

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        del conversation_id  # intentionally unused — there's no scoping
        model = resolve_answer_model()
        start = time.perf_counter()
        result = self._llm.complete(
            model=model,
            system=(
                "Answer the question concisely. If you don't know, say so. "
                "Do not fabricate details."
            ),
            user=question,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return AnswerResult(
            answer=result.answer,
            answer_model=model,
            answer_input_tokens=result.input_tokens,
            answer_output_tokens=result.output_tokens,
            elapsed_ms=elapsed_ms,
            retrieved_context=None,
        )

    def health_check(self) -> HealthResult:
        # Same story as naive: no remote state; the LLM provider check
        # covers the only failure mode.
        return HealthResult(ok=True, detail="(baseline — no remote state)")
