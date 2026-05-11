"""Zep adapter — uses Zep Cloud (the SaaS) via the `zep-cloud` SDK.

Zep's architectural bet: a temporal knowledge graph (Graphiti) where
every fact has bi-temporal validity (when did the fact become true vs
when was it learned). Different shape from Mem0's flat fact store and
Statewave's compiled-memories model — comparing all three on LoCoMo
shows where each architecture's strengths actually land.

Data model: each LoCoMo conversation maps to one Zep `user`, with
one `thread` per conversation that holds every session's messages.
Session boundaries are preserved via per-message metadata so Zep's
graph extractor can use them as discontinuity hints.

We use Zep Cloud (not the open-source Graphiti directly) because:
  - It's the path Zep promotes for adoption (matches what users try)
  - Zero-setup for the bench operator (no Neo4j to provision)
  - Cloud has the latest server release; OSS lags

Cost note: Zep's free tier covers our pilot run; the publishable full
run may need a paid tier. Documented in the README's cost section.
"""

from __future__ import annotations

import contextlib
import os
import re
import time

from ..dataset import LocomoConversation
from ..llm import LlmClient, resolve_answer_model
from .base import AnswerResult, HealthResult, MemorySystem

# Zep restricts user_id charset (alphanumeric + dash); LoCoMo ids are
# usually clean but normalize to be safe.
_UNSAFE_USER_ID = re.compile(r"[^a-zA-Z0-9-]+")


def _user_id_for(conversation_id: str) -> str:
    safe = _UNSAFE_USER_ID.sub("-", conversation_id)
    return f"bench-locomo-{safe}"


def _thread_id_for(conversation_id: str) -> str:
    return f"thread-{_user_id_for(conversation_id)}"


# Zep cloud's add_messages endpoint enforces a 30-message-per-request
# cap (400: "messages cannot contain more than 30 items"). LoCoMo
# sessions can hit 40-60 turns, so we chunk per session.
ZEP_MAX_MESSAGES_PER_REQUEST = 30


# Zep's add_messages_batch is the fast path for bulk ingest — one
# round-trip per session keeps the API budget bounded for ~600
# conversations x ~5 sessions = ~3000 calls.
class ZepSystem(MemorySystem):
    name = "zep"

    def __init__(self) -> None:
        from zep_cloud.client import Zep

        api_key = os.environ.get("ZEP_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ZEP_API_KEY not set. Sign up at https://www.getzep.com, "
                "create a project, and export ZEP_API_KEY before running."
            )
        self._client = Zep(api_key=api_key)
        self._llm = LlmClient()

    def ingest(self, conversation: LocomoConversation) -> None:
        from zep_cloud.types.message import Message

        user_id = _user_id_for(conversation.id)
        thread_id = _thread_id_for(conversation.id)

        # Idempotency: delete the user (cascade-deletes their threads
        # and graph) before re-creating. 404 means no prior data — fine.
        with contextlib.suppress(Exception):
            self._client.user.delete(user_id=user_id)

        # Create the user + the thread. Zep's user.add takes user_id
        # by kw; thread.create needs both thread_id + user_id.
        self._client.user.add(user_id=user_id, metadata={"bench": "locomo"})
        self._client.thread.create(thread_id=thread_id, user_id=user_id)

        # Add messages session-by-session. Zep extracts entities and
        # relationships into the graph asynchronously after each call;
        # the bench's per-conversation answer phase happens immediately
        # after ingest, so questions effectively race the extraction —
        # documented in the README so it's not a hidden variable.
        for session_idx, session in enumerate(conversation.sessions):
            if not session:
                continue
            # Zep's `name` field preserves the speaker, but graph
            # extraction operates on `content`. Without the LoCoMo
            # timestamp in content, the graph never gets the absolute
            # date for relative phrases ("last Saturday", "two days
            # ago") and the bench's temporal questions are unanswerable.
            messages = [
                Message(
                    content=(f"[{turn.timestamp}] {turn.text}" if turn.timestamp else turn.text),
                    role=("assistant" if turn.speaker.lower() == "assistant" else "user"),
                    name=turn.speaker,
                    metadata={"session_id": session_idx, "bench": "locomo"},
                )
                for turn in session
            ]
            # Chunk to stay under Zep's 30-message cap. We preserve
            # in-session order across chunks so the graph extractor
            # sees the conversation flow intact — submitting chunks
            # serially is fine because add_messages is synchronous
            # (extraction is async on Zep's side, but the API call
            # itself returns when the messages are accepted).
            for chunk_start in range(0, len(messages), ZEP_MAX_MESSAGES_PER_REQUEST):
                chunk = messages[chunk_start : chunk_start + ZEP_MAX_MESSAGES_PER_REQUEST]
                try:
                    self._client.thread.add_messages(
                        thread_id,
                        messages=chunk,
                    )
                except Exception as e:
                    from ..runner import console

                    console.print(
                        f"[yellow]zep ingest failed for session {session_idx} "
                        f"(chunk {chunk_start // ZEP_MAX_MESSAGES_PER_REQUEST}) "
                        f"of {conversation.id}:[/] {e}"
                    )

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        thread_id = _thread_id_for(conversation_id)
        start = time.perf_counter()

        # Step 1: retrieve Zep's structured memory bundle. This is
        # Zep's recommended retrieval entry point — it returns a
        # ready-to-paste-into-prompt context string built from the
        # graph's relevant facts + thread summary. We use it as-is so
        # we're benchmarking Zep-as-deployed, not our prompt-engineering.
        try:
            memory = self._client.thread.get_user_context(thread_id)
            context = memory.context or "(no relevant memories found)"
        except Exception as e:
            from ..runner import console

            console.print(f"[yellow]zep get_user_context failed:[/] {e}")
            context = "(retrieval failed)"

        # Step 2: prompt the shared answer model with the bundle.
        model = resolve_answer_model()
        prompt = (
            "Answer the question using the context below. If the answer "
            "isn't in the context, say so honestly — do not fabricate.\n\n"
            f"--- Context ---\n{context}\n\n"
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
            # Zep's graph build runs LLM calls server-side that bill
            # against the operator's Zep plan, not against an external
            # provider key. Documented in README cost section.
        )

    def reset(self) -> None:
        # Per-conversation cleanup is handled in `ingest` (deletes the
        # user before re-creating). Full-bench teardown can iterate
        # users via client.user.list_ordered.
        return None

    def health_check(self) -> HealthResult:
        # Cheap read: list-ordered first page of users with page_size=1.
        # Verifies the API key + project reachable without needing any
        # specific user to exist. Returns either a tiny payload (success)
        # or raises with auth / network / quota detail (failure).
        try:
            self._client.user.list_ordered(page_size=1)
            return HealthResult(ok=True, detail="ok")
        except Exception as e:
            return HealthResult(ok=False, detail=_short(e))


def _short(err: object) -> str:
    s = str(err)
    if len(s) > 200:
        s = s[:200] + "…"
    return s.replace("\n", " ")
