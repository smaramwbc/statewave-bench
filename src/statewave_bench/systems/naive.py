"""'Naive' baseline — every-developer's-first-attempt at memory.

Stuffs the last N conversation turns directly into the answer
prompt. No retrieval, no compilation, no ranking — just a sliding
window over recent history.

Two failure modes this exposes:
  1. Context blows past the model's context window for long
     conversations (LoCoMo's multi-session structure; the dataset
     has thousands of turns per conversation).
  2. The relevant fact is older than N turns and gets evicted; the
     model confabulates.

The gap between this baseline and any real memory system measures
"what the memory layer adds over naive context dumping" — which is
the actually-interesting question for adopters deciding whether to
take on the complexity of a memory runtime.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from ..dataset import LocomoConversation, LocomoTurn
from ..llm import LlmClient, make_qa_prompt, resolve_answer_model
from .base import AnswerResult, HealthResult, MemorySystem

# Tunable: how many turns to keep. 100 is a deliberately permissive
# floor — it gives the naive baseline a fair shot at single-session
# questions while still failing on multi-session ones (LoCoMo
# conversations are thousands of turns).
DEFAULT_WINDOW_SIZE = 100


class NaiveSystem(MemorySystem):
    name = "naive"

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self._llm = LlmClient()
        self._window_size = window_size
        # Per-conversation rolling window. We index by `conversation.id`
        # so the bench can interleave conversations across runs without
        # cross-contamination.
        self._windows: dict[str, deque[LocomoTurn]] = defaultdict(lambda: deque(maxlen=window_size))

    def ingest(self, conversation: LocomoConversation) -> None:
        window = self._windows[conversation.id]
        window.clear()
        window.extend(turn for session in conversation.sessions for turn in session)

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        window = self._windows.get(conversation_id) or deque()
        # Include LoCoMo timestamps in the dumped context so the naive
        # baseline isn't artificially handicapped on temporal questions
        # (the conversation uses relative phrases — "last Saturday",
        # "two days ago" — that only resolve against an absolute date).
        # Same shape every vendor adapter sees, for cross-system fairness.
        context = "\n".join(
            f"[{t.timestamp}] {t.speaker}: {t.text}" if t.timestamp else f"{t.speaker}: {t.text}"
            for t in window
        )
        model = resolve_answer_model()
        prompt = make_qa_prompt(context=context, question=question)
        start = time.perf_counter()
        result = self._llm.complete(
            model=model,
            system=None,
            user=prompt,
            max_tokens=512,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return AnswerResult(
            answer=result.answer,
            answer_model=model,
            answer_input_tokens=result.input_tokens,
            answer_output_tokens=result.output_tokens,
            elapsed_ms=elapsed_ms,
            retrieved_context=context,
        )

    def reset(self) -> None:
        self._windows.clear()

    def health_check(self) -> HealthResult:
        # The naive baseline has no remote state — it just keeps a
        # rolling deque in memory. The only thing that can fail at
        # health-probe time is the LLM client itself, which the
        # provider-level Anthropic/OpenAI checks already cover.
        return HealthResult(ok=True, detail="(baseline — no remote state)")
