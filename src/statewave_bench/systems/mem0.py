"""Mem0 adapter — uses the official `mem0ai` Python SDK.

Mem0's model: every memory belongs to a user_id. We map LoCoMo
conversation ids onto user ids of the form `bench:locomo:<id>` so
this run doesn't pollute any other Mem0 data the operator holds.

Two deployment modes Mem0 supports:
  - Cloud (Mem0 Platform) — needs MEM0_API_KEY
  - Self-hosted — needs a Postgres + a vector store

The adapter prefers cloud (lower setup cost for the bench operator)
but auto-detects self-hosted via env. The bench's published numbers
will be from the cloud setup so users can reproduce by signing up
for a free tier.

Implementation status: SCAFFOLDED. The mem0 SDK shape lives behind
the lazy import; the exact `Memory.add(...)` / `Memory.search(...)`
call signatures want one careful pass against the v0.1.30+ SDK
before publishing. Wave A2 wires it end-to-end.
"""

from __future__ import annotations

import os
import time

from ..dataset import LocomoConversation
from ..llm import LlmClient, resolve_answer_model
from .base import AnswerResult, MemorySystem


USER_ID_PREFIX = "bench:locomo:"


class Mem0System(MemorySystem):
    name = "mem0"

    def __init__(self) -> None:
        from mem0 import Memory, MemoryClient  # type: ignore[import-not-found]

        # Cloud mode — preferred for the bench because operators don't
        # need to provision Postgres + a vector store.
        if api_key := os.environ.get("MEM0_API_KEY"):
            self._client = MemoryClient(api_key=api_key)
            self._cloud = True
        else:
            # Self-hosted mode — Memory.from_config(...) reads vector
            # store + LLM provider from env / config dict. Wave A2
            # will add a `swb config check` command that verifies the
            # operator's setup before the bench burns tokens.
            self._client = Memory()
            self._cloud = False
        self._llm = LlmClient()

    def ingest(self, conversation: LocomoConversation) -> None:
        # Wave A2 will:
        #   1. Build a list of {role, content} messages from every
        #      session of the conversation
        #   2. Call client.add(messages, user_id=...) — Mem0
        #      internally extracts facts via its own LLM call (the
        #      'internal cost' we'll report alongside the answer cost
        #      so the operator sees the real bill)
        #
        # Mem0 supports per-add metadata; we'll stamp `session_id` so
        # post-hoc analysis can correlate retrieved facts back to
        # source sessions.
        raise NotImplementedError(
            "Mem0 ingest is scaffolded but not yet wired — Wave A2."
        )

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        # Wave A2 will:
        #   1. client.search(query=question, user_id=...) → list of
        #      retrieved facts (Mem0 returns ranked + scored memories)
        #   2. Format facts into a prompt prelude
        #   3. Call the shared answer model with prelude + question
        #
        # Open question: how many memories to retrieve? Mem0's search
        # has a `limit` param defaulting to 5. We'll match that for
        # apples-to-apples; document the choice in the README.
        del conversation_id, question
        model = resolve_answer_model()
        elapsed_ms = (time.perf_counter() - time.perf_counter()) * 1000
        del model, elapsed_ms
        raise NotImplementedError(
            "Mem0 answer is scaffolded but not yet wired — Wave A2."
        )

    def reset(self) -> None:
        # Cloud mode: delete by user_id is a single API call.
        # Self-hosted: same SDK call. Wave A2.
        return None
