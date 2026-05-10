"""Mem0 adapter — uses the official `mem0ai` Python SDK (v2+).

Mem0's data model: every memory belongs to a user_id. We map LoCoMo
conversation ids onto user ids of the form `bench-locomo-<id>` so
this run doesn't pollute any other Mem0 data the operator holds.
(Mem0 user_ids are restricted to alphanumeric + dash; we replace
colons + underscores in LoCoMo ids that don't match.)

Two deployment modes:
  - Cloud (Mem0 Platform)   — needs MEM0_API_KEY. Preferred for the
    bench: zero infra setup, free tier covers the pilot run.
  - Self-hosted             — needs vector store + LLM provider env;
    the SDK auto-detects from env. Documented but secondary.

The published bench numbers will be from the cloud setup so anyone
can reproduce by signing up for a free tier.

Internal LLM cost: Mem0's `add()` runs its own fact-extraction LLM
call against the operator's configured provider (OpenAI by default).
That cost lands on the operator's OpenAI bill, not in Mem0's API
response. The bench surfaces this via the README's cost section
rather than per-call internal-token counters because the SDK
doesn't return them.
"""

from __future__ import annotations

import contextlib
import os
import re
import time

from ..dataset import LocomoConversation
from ..llm import LlmClient, resolve_answer_model
from .base import AnswerResult, HealthResult, MemorySystem

# Mem0 user_id charset is alphanumeric + dash. LoCoMo ids are
# usually clean strings but some contain underscores / colons —
# normalize.
_UNSAFE_USER_ID = re.compile(r"[^a-zA-Z0-9-]+")


def _user_id_for(conversation_id: str) -> str:
    safe = _UNSAFE_USER_ID.sub("-", conversation_id)
    return f"bench-locomo-{safe}"


# Mem0 search default. Their docs recommend top_k=5 for most retrieval
# tasks; we match so we're benchmarking Mem0-as-recommended, not our
# tuning. Documented in the README's methodology section.
DEFAULT_TOP_K = 5


class Mem0System(MemorySystem):
    name = "mem0"

    def __init__(self) -> None:
        from mem0 import Memory, MemoryClient

        # Cloud-first. Self-hosted falls back to Memory.from_config()
        # which reads vector store + LLM settings from env (or a
        # config dict that operators can plug via MEM0_CONFIG_PATH —
        # not exercised in the published run).
        if api_key := os.environ.get("MEM0_API_KEY"):
            host = os.environ.get("MEM0_HOST")  # optional override
            self._client = MemoryClient(
                api_key=api_key,
                host=host,
            )
            self._cloud = True
        else:
            self._client = Memory()
            self._cloud = False
        self._llm = LlmClient()

    def ingest(self, conversation: LocomoConversation) -> None:
        user_id = _user_id_for(conversation.id)

        # Idempotency: clear prior memories for this user_id before
        # ingesting. 404 / empty-subject is fine; other errors surface
        # on the next call.
        with contextlib.suppress(Exception):
            self._delete_all(user_id)

        # Mem0 takes message lists in OpenAI-style {role, content}
        # shape. We feed every session as one batch so Mem0's fact
        # extractor sees the full conversational context per session,
        # not isolated turns. Sessions are added sequentially (one
        # add() per session) so Mem0 can de-duplicate / merge facts
        # across sessions the way it does in production.
        for session_idx, session in enumerate(conversation.sessions):
            if not session:
                continue
            messages = [
                {
                    "role": "user" if turn.speaker.lower() != "assistant" else "assistant",
                    "content": turn.text,
                }
                for turn in session
            ]
            try:
                self._add(
                    messages=messages,
                    user_id=user_id,
                    metadata={
                        "session_id": session_idx,
                        "bench": "locomo",
                    },
                )
            except Exception as e:
                # One failed session is recoverable — log + continue
                # so the bench produces some signal rather than zero.
                # The runner's per-question failure tolerance handles
                # the downstream consequences.
                from ..runner import console

                console.print(
                    f"[yellow]mem0 ingest failed for session {session_idx} "
                    f"of {conversation.id}:[/] {e}"
                )

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        user_id = _user_id_for(conversation_id)
        start = time.perf_counter()

        # Step 1: retrieve memories.
        search_result = self._search(query=question, user_id=user_id, top_k=DEFAULT_TOP_K)

        # The SDK returns either {"results": [...]} (cloud v2 shape)
        # or just [...] (older / self-hosted). Handle both.
        if isinstance(search_result, dict):
            memories = search_result.get("results") or search_result.get("memories") or []
        elif isinstance(search_result, list):
            memories = search_result
        else:
            memories = []

        # Format the memories into a prompt prelude. Mem0 returns
        # `{memory: "...", score: 0.x, ...}` per item; we use the
        # `memory` field, which is the human-readable fact.
        formatted = []
        for m in memories:
            if isinstance(m, dict):
                text = m.get("memory") or m.get("content") or m.get("text") or ""
            else:
                text = str(m)
            if text:
                formatted.append(f"- {text}")
        context = "\n".join(formatted) if formatted else "(no relevant memories found)"

        # Step 2: prompt the shared answer model with the memory list.
        model = resolve_answer_model()
        prompt = (
            "Answer the question using the memories below. If the answer "
            "isn't in the memories, say so honestly — do not fabricate.\n\n"
            f"--- Memories ---\n{context}\n\n"
            f"--- Question ---\n{question}"
        )
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
            # Mem0 does its own LLM calls during ingest (fact
            # extraction). Those tokens land on the operator's
            # provider account but aren't reported in the SDK
            # response — the README's cost section flags this so
            # operators don't underestimate the real bill.
        )

    def reset(self) -> None:
        # The runner doesn't call this between conversations (each
        # adapter's `ingest` already deletes its own user_id first).
        # Operators wanting a full purge across all bench user_ids
        # can iterate themselves.
        return None

    def health_check(self) -> HealthResult:
        # Cheap read on a guaranteed-empty user_id. Routes through the
        # same v2-aware helper as the real ingest/answer paths so the
        # probe fails the same way the runner would.
        try:
            self._get_all(user_id="bench-health-probe-nonexistent")
            return HealthResult(ok=True, detail="ok")
        except Exception as e:
            return HealthResult(ok=False, detail=_short(e))

    # ── Cloud-vs-self-hosted shims ────────────────────────────────────────
    # Mem0 v2's cloud `MemoryClient` rejects top-level entity kwargs
    # (`user_id=`); identity fields move into `filters={...}`. The
    # self-hosted `Memory` class kept the old signatures for write
    # methods (`add`, `delete_all`) but adopted `filters=` for reads
    # (`search`, `get_all`). These four helpers branch on `self._cloud`
    # so the caller code stays clean.

    def _add(
        self,
        *,
        messages: list[dict[str, str]],
        user_id: str,
        metadata: dict[str, object],
    ) -> object:
        if self._cloud:
            from mem0.client.types import AddMemoryOptions

            return self._client.add(
                messages,
                options=AddMemoryOptions(
                    filters={"user_id": user_id},
                    metadata=metadata,
                ),
            )
        return self._client.add(messages, user_id=user_id, metadata=metadata)

    def _search(self, *, query: str, user_id: str, top_k: int) -> object:
        if self._cloud:
            from mem0.client.types import SearchMemoryOptions

            return self._client.search(
                query,
                options=SearchMemoryOptions(
                    filters={"user_id": user_id},
                    top_k=top_k,
                ),
            )
        return self._client.search(query, top_k=top_k, filters={"user_id": user_id})

    def _get_all(self, *, user_id: str) -> object:
        if self._cloud:
            from mem0.client.types import GetAllMemoryOptions

            return self._client.get_all(
                options=GetAllMemoryOptions(filters={"user_id": user_id}),
            )
        return self._client.get_all(filters={"user_id": user_id})

    def _delete_all(self, user_id: str) -> object:
        if self._cloud:
            from mem0.client.types import DeleteAllMemoryOptions

            return self._client.delete_all(
                options=DeleteAllMemoryOptions(filters={"user_id": user_id}),
            )
        return self._client.delete_all(user_id=user_id)


def _short(err: object) -> str:
    s = str(err)
    if len(s) > 200:
        s = s[:200] + "…"
    return s.replace("\n", " ")
