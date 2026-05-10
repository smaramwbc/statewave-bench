"""Zep adapter — uses Zep Cloud (the SaaS) via the `zep-cloud` SDK.

Zep's architectural bet: a temporal knowledge graph (Graphiti) where
every fact has bi-temporal validity (when did the fact become true vs
when was it learned). Different shape from Mem0's flat fact store and
Statewave's compiled-memories model — comparing all three on LoCoMo
shows where each architecture's strengths actually land.

We use Zep Cloud (not the open-source Graphiti directly) because:
  - It's the path Zep promotes for adoption (matches what users
    actually try)
  - Zero-setup for the bench operator (no Neo4j to provision)
  - Cloud has the latest server release; OSS lags

Cost note: Zep's free tier covers our pilot run; the publishable full
run may need a paid tier. Documented in the README's cost section.

Implementation status: SCAFFOLDED. Wave A2 wires the real calls.
"""

from __future__ import annotations

import os
import time

from ..dataset import LocomoConversation
from ..llm import LlmClient, resolve_answer_model
from .base import AnswerResult, MemorySystem


USER_ID_PREFIX = "bench-locomo-"  # Zep restricts user_id charset; no colons


class ZepSystem(MemorySystem):
    name = "zep"

    def __init__(self) -> None:
        from zep_cloud.client import Zep  # type: ignore[import-not-found]

        api_key = os.environ.get("ZEP_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ZEP_API_KEY not set. Sign up at https://www.getzep.com, "
                "create a project, and export ZEP_API_KEY before running."
            )
        self._client = Zep(api_key=api_key)
        self._llm = LlmClient()

    def ingest(self, conversation: LocomoConversation) -> None:
        # Wave A2 will:
        #   1. client.user.add(user_id=...) — create the per-conversation user
        #   2. client.thread.create(...) — Zep's session/thread abstraction
        #   3. For each LoCoMo session, post messages via
        #      client.thread.add_messages(...). Zep extracts entities,
        #      relationships, and temporal validity into the graph
        #      automatically.
        #
        # Open question: do we use one Zep "thread" per LoCoMo session,
        # or one thread per conversation? The conversation-as-thread
        # model better matches "user has continuous memory across the
        # whole multi-session interaction" — likely the right call but
        # I want to validate against Zep's docs in Wave A2.
        raise NotImplementedError(
            "Zep ingest is scaffolded but not yet wired — Wave A2."
        )

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        # Wave A2 will:
        #   1. client.memory.get(user_id=..., last_n=N) — retrieves
        #      Zep's structured memory bundle (facts + summary +
        #      relevant_facts). The bundle is already ranked by Zep's
        #      retrieval logic; we just pass it through.
        #   2. Format the bundle into a prompt (Zep ships a recommended
        #      prompt template — we'll use it verbatim so we're
        #      benchmarking Zep-as-deployed, not our prompt-engineering)
        #   3. Call the shared answer model
        del conversation_id, question
        model = resolve_answer_model()
        elapsed_ms = (time.perf_counter() - time.perf_counter()) * 1000
        del model, elapsed_ms
        raise NotImplementedError(
            "Zep answer is scaffolded but not yet wired — Wave A2."
        )

    def reset(self) -> None:
        # Wave A2 — Zep's API has user.delete which cascades to threads
        # and graph nodes. Nice fit for bench teardown.
        return None
