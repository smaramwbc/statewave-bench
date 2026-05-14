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
from ..llm import LlmClient, make_qa_prompt, resolve_answer_model
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
#
# Override via `MEM0_TOP_K` env var to run a sweep without code changes
# (e.g. `MEM0_TOP_K=20 swb run --systems mem0`). Sweep results published
# alongside the headline number so readers can see Mem0's ceiling, not
# just its default-config result.
DEFAULT_TOP_K = 5

# Mem0's add() is async — the response shape is
#   {"message": "queued for background execution", "status": "PENDING",
#    "event_id": "..."}
# and the actual fact extraction runs on Mem0's side after the call
# returns. Mem0's SDK doesn't expose a public wait/poll API on
# `event_id`, so we settle by polling get_all() until the memory count
# stops growing for N consecutive polls — same pattern as Zep's graph
# settle. Without this, the bench's first answer() call races the
# extractor queue and gets empty retrieval; that silently dragged
# Mem0's multi_hop score to ~0.08 in the first --limit 10 run, where
# 3 of 3 sampled misses literally returned "no relevant memories
# found" because the queue hadn't drained.
#
# 3s poll x 3 stable polls = 9s of stability; 180s overall timeout is
# generous — even dense LoCoMo sessions drain in 30-60s when Mem0 is
# healthy. The timeout exists as a runaway-cost guard, not as the
# expected exit.
MEM0_SETTLE_POLL_SEC = 3.0
MEM0_SETTLE_STABLE_COUNT = 3
MEM0_SETTLE_TIMEOUT_SEC = 180.0


def _resolve_top_k() -> int:
    raw = os.environ.get("MEM0_TOP_K")
    if not raw:
        return DEFAULT_TOP_K
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_TOP_K


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

            # Mem0's message shape uses generic user/assistant roles
            # (the SDK's fact extractor expects OpenAI-style chat
            # messages), so the speaker name has to live in `content`.
            # Same applies for the LoCoMo timestamp — without it the
            # fact extractor can't answer "When did X happen?" because
            # the conversation only uses relative phrases like
            # "last Saturday".
            def _content_for(turn: object) -> str:
                ts = getattr(turn, "timestamp", "") or ""
                speaker = getattr(turn, "speaker", "") or ""
                text = getattr(turn, "text", "")
                prefix = f"[{ts}] " if ts else ""
                return f"{prefix}{speaker}: {text}" if speaker else f"{prefix}{text}"

            messages = [
                {
                    "role": "user" if turn.speaker.lower() != "assistant" else "assistant",
                    "content": _content_for(turn),
                }
                for turn in session
            ]
            # Per-session ingest exceptions propagate to the runner,
            # which catches them at the (system, conv) boundary and
            # drops the whole conversation for this system. Matches
            # Statewave's compile-failure contract: a partial fact
            # store would silently score against missing facts on
            # every later question, which is worse than recording
            # nothing — the leaderboard would then mix "Mem0 lost the
            # question fairly" with "Mem0 never saw the fact." Better
            # to fail loudly and let the operator rerun.
            self._add(
                messages=messages,
                user_id=user_id,
                metadata={
                    "session_id": session_idx,
                    "bench": "locomo",
                },
            )

        # Block until Mem0's async fact extractor has drained its queue
        # for this user_id. See MEM0_SETTLE_* constants for the
        # rationale — without this wait the bench's first answer() call
        # would race the extractor and retrieve nothing.
        self._wait_for_extraction_settle(user_id=user_id, conversation_id=conversation.id)

    def _wait_for_extraction_settle(self, *, user_id: str, conversation_id: str) -> None:
        from ..runner import console

        last_count: int | None = None
        stable = 0
        elapsed = 0.0
        while elapsed < MEM0_SETTLE_TIMEOUT_SEC:
            try:
                result = self._get_all(user_id=user_id)
            except Exception as e:
                console.print(f"[yellow]mem0 settle-poll failed for {conversation_id}:[/] {e}")
                return
            # Result shape on cloud v2: {"results": [...]}; on self-hosted
            # Memory: {"results": [...]} or {"memories": [...]}; sometimes
            # just a bare list. Handle all three.
            if isinstance(result, dict):
                memories = result.get("results") or result.get("memories") or []
            elif isinstance(result, list):
                memories = result
            else:
                memories = []
            count = len(memories)
            if last_count is not None and count == last_count:
                stable += 1
                if stable >= MEM0_SETTLE_STABLE_COUNT:
                    return
            else:
                stable = 0
            last_count = count
            time.sleep(MEM0_SETTLE_POLL_SEC)
            elapsed += MEM0_SETTLE_POLL_SEC
        console.print(
            f"[yellow]mem0 extraction still growing after {MEM0_SETTLE_TIMEOUT_SEC:.0f}s "
            f"for {conversation_id}; proceeding anyway."
        )

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        user_id = _user_id_for(conversation_id)
        start = time.perf_counter()

        # Step 1: retrieve memories.
        search_result = self._search(query=question, user_id=user_id, top_k=_resolve_top_k())

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
        prompt = make_qa_prompt(context=context, question=question)
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
    # Mem0's cloud API splits the contract by read vs write:
    #
    #   Writes (add, delete_all):
    #     REQUIRE top-level `user_id=`. Identity in `filters=` returns
    #     400 "At least one entity ID is required".
    #
    #   Reads (search, get_all):
    #     REJECT top-level `user_id=` with "Top-level entity parameters
    #     not supported in <method>(). Use filters={'user_id': ...}".
    #     Also require `version="v2"` — without it, search hits the v1
    #     endpoint which silently returns `{'results': []}` regardless
    #     of stored memories. That silent-empty behavior is the worst
    #     possible failure mode for the bench (zero retrieval but no
    #     error), so version="v2" is non-negotiable here.
    #
    # Self-hosted `Memory` accepts top-level `user_id=` for writes and
    # `filters=` for reads; the `version` kwarg is ignored on self-
    # hosted, so the same call works in both modes.

    def _add(
        self,
        *,
        messages: list[dict[str, str]],
        user_id: str,
        metadata: dict[str, object],
    ) -> object:
        return self._client.add(messages, user_id=user_id, metadata=metadata)

    def _search(self, *, query: str, user_id: str, top_k: int) -> object:
        return self._client.search(
            query,
            filters={"user_id": user_id},
            top_k=top_k,
            version="v2",
        )

    def _get_all(self, *, user_id: str) -> object:
        return self._client.get_all(filters={"user_id": user_id}, version="v2")

    def _delete_all(self, user_id: str) -> object:
        return self._client.delete_all(user_id=user_id)


def _short(err: object) -> str:
    s = str(err)
    if len(s) > 200:
        s = s[:200] + "…"
    return s.replace("\n", " ")
