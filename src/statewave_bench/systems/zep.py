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

# Zep's graph build is async with TWO async phases:
#   1. Message ingestion — completes in ~5-10s; `task.get(task_id).status`
#      flips to "completed" once messages are persisted.
#   2. Graph extraction — runs BEHIND the task's "completed" flag and
#      keeps populating entities + facts for ~30-60s afterwards.
#
# Waiting only on the task status lands us in the gap between phases:
# we resume answering questions against a half-built graph (this is
# what tanked Zep to 0.000 on multi_hop in the first run; tasks
# reported done at t+7s while the graph kept growing through t+60s).
# So we wait on task completion AND then poll `get_user_context`
# until the context length stops growing for several consecutive polls.
ZEP_TASK_TIMEOUT_SEC = 180.0
ZEP_TASK_POLL_INTERVAL_SEC = 2.0
ZEP_TERMINAL_STATUSES = frozenset({"completed", "success", "succeeded", "failed", "error"})

# Graph-extraction settle: poll get_user_context until its body length
# doesn't change for N consecutive polls. 3 polls x 5s = 15s of
# stability is empirically enough to declare the graph "stable enough"
# on a 19-session LoCoMo conversation.
ZEP_GRAPH_SETTLE_POLL_SEC = 5.0
ZEP_GRAPH_SETTLE_STABLE_COUNT = 3
# 240s empirically — a 19-session LoCoMo conversation was still growing
# at 120s in the first verification probe. Doubling the budget so the
# bench measures Zep's actual graph (not a half-built one) without
# making per-conversation wall time unreasonable.
ZEP_GRAPH_SETTLE_TIMEOUT_SEC = 240.0


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

        # Add messages session-by-session. Zep's graph extraction is
        # async — every add_messages returns a task_id, and we have to
        # wait for those tasks to complete before issuing `answer`
        # calls, otherwise retrieval hits a half-built graph (this is
        # what tanked Zep to 0.000 on multi_hop and 0.029 on
        # open_domain in the first cross-system run).
        task_ids: list[str] = []
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
                    response = self._client.thread.add_messages(
                        thread_id,
                        messages=chunk,
                    )
                    task_id = getattr(response, "task_id", None)
                    if task_id:
                        task_ids.append(task_id)
                except Exception as e:
                    from ..runner import console

                    console.print(
                        f"[yellow]zep ingest failed for session {session_idx} "
                        f"(chunk {chunk_start // ZEP_MAX_MESSAGES_PER_REQUEST}) "
                        f"of {conversation.id}:[/] {e}"
                    )

        # Block until all graph-build tasks are terminal (completed or
        # failed). Mirrors the Statewave compile-wait contract: ingest
        # only returns once memory is queryable. Without this the bench
        # would race the async build and score against an empty graph.
        self._wait_for_tasks(task_ids, conversation_id=conversation.id)

        # Tasks completing only means messages were ingested. The graph
        # extractor keeps populating facts for another ~30-60s. Poll
        # the assembled context until it stops growing for a few
        # consecutive samples before declaring the graph "settled".
        self._wait_for_graph_settle(thread_id=thread_id, conversation_id=conversation.id)

    def _wait_for_tasks(self, task_ids: list[str], *, conversation_id: str) -> None:
        if not task_ids:
            return
        from ..runner import console

        pending = list(task_ids)
        elapsed = 0.0
        while pending and elapsed < ZEP_TASK_TIMEOUT_SEC:
            still_pending: list[str] = []
            for tid in pending:
                try:
                    task = self._client.task.get(tid)
                except Exception as e:
                    # Polling failure is non-fatal — we treat it as
                    # "still pending" and try again next tick. If the
                    # whole API is down, the timeout below catches it.
                    console.print(f"[yellow]zep task poll failed for {tid}:[/] {e}")
                    still_pending.append(tid)
                    continue
                status = (task.status or "").lower()
                if status in ZEP_TERMINAL_STATUSES:
                    if status not in ("completed", "success", "succeeded"):
                        console.print(
                            f"[yellow]zep task {tid} ended {status!r} "
                            f"(conv={conversation_id}); proceeding with partial graph."
                        )
                    continue
                still_pending.append(tid)
            pending = still_pending
            if pending:
                time.sleep(ZEP_TASK_POLL_INTERVAL_SEC)
                elapsed += ZEP_TASK_POLL_INTERVAL_SEC
        if pending:
            console.print(
                f"[yellow]zep ingest: {len(pending)} task(s) still pending after "
                f"{ZEP_TASK_TIMEOUT_SEC:.0f}s for {conversation_id}; "
                f"proceeding with whatever's been built."
            )

    def _wait_for_graph_settle(self, *, thread_id: str, conversation_id: str) -> None:
        from ..runner import console

        last_length: int | None = None
        stable_count = 0
        elapsed = 0.0
        while elapsed < ZEP_GRAPH_SETTLE_TIMEOUT_SEC:
            try:
                ctx = self._client.thread.get_user_context(thread_id)
                body = ctx.context or ""
            except Exception as e:
                console.print(f"[yellow]zep settle-poll failed for {thread_id}:[/] {e}")
                return
            length = len(body)
            if last_length is not None and length == last_length:
                stable_count += 1
                if stable_count >= ZEP_GRAPH_SETTLE_STABLE_COUNT:
                    return
            else:
                stable_count = 0
            last_length = length
            time.sleep(ZEP_GRAPH_SETTLE_POLL_SEC)
            elapsed += ZEP_GRAPH_SETTLE_POLL_SEC
        console.print(
            f"[yellow]zep graph still growing after {ZEP_GRAPH_SETTLE_TIMEOUT_SEC:.0f}s "
            f"for {conversation_id}; proceeding anyway."
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
