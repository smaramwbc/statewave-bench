"""Statewave adapter — talks to a running Statewave server via the
Python SDK (`statewave-py`).

Setup: operator runs a Statewave server (e.g. via the helm chart or
docker compose recipe in `statewave-connectors`), exports
`STATEWAVE_URL` + `STATEWAVE_API_KEY`, and the adapter connects.

Bench-specific subject: every conversation gets a unique subject id
of the form `bench:locomo:<conversation_id>`. That keeps bench data
isolated from any production memories the same Statewave instance
might be holding, and makes it trivial to purge afterward
(`delete_subject(...)` per conversation, or DELETE /v1/subjects/* via
admin API).

Compile-mode comparison: `compile_memories(...)` runs whichever
compiler the server is configured for (heuristic by default, LLM if
the operator set `STATEWAVE_LITELLM_*` env vars on the server). To
compare heuristic vs LLM compile modes, run this bench twice against
two server instances — see the README's methodology section.

Per-session episode model: each LoCoMo session becomes one episode
of `type="conversation"` with payload `{messages: [...]}`. That's the
shape Statewave's existing connectors emit (Slack, Zendesk, Intercom),
so the bench exercises the same retrieval path real deployments use
rather than a synthetic shape.
"""

from __future__ import annotations

import contextlib
import os
import time

from ..dataset import LocomoConversation
from ..llm import LlmClient, resolve_answer_model
from .base import AnswerResult, HealthResult, MemorySystem

SUBJECT_PREFIX = "bench:locomo:"
EPISODE_SOURCE = "statewave-bench"
EPISODE_TYPE = "conversation"

# Cap the context bundle we ask Statewave to assemble per question.
# 2K tokens is generous — most LoCoMo questions need a few facts plus
# a session excerpt or two; 8K would just inflate cost without
# improving recall on this dataset.
DEFAULT_CONTEXT_MAX_TOKENS = 2048

# Per-conversation compile uses the wait-for-completion path so the
# bench's `answer` calls always run against fully-compiled memory.
# 60s should be enough for any LoCoMo conversation; longer ones surface
# in the logs as compile-timeout errors and the bench moves on.
COMPILE_TIMEOUT_SEC = 60.0


def _subject_for(conversation_id: str) -> str:
    return f"{SUBJECT_PREFIX}{conversation_id}"


class StatewaveSystem(MemorySystem):
    name = "statewave"

    def __init__(self) -> None:
        # Lazy SDK import so other systems can run without statewave-py
        # installed locally (operators benchmarking only Mem0 vs Zep
        # don't need a Statewave server).
        from statewave import StatewaveClient

        url = os.environ.get("STATEWAVE_URL")
        api_key = os.environ.get("STATEWAVE_API_KEY")
        if not url:
            raise RuntimeError(
                "STATEWAVE_URL not set. Point at a running Statewave server, "
                "e.g. http://localhost:8000 — see the deployment guide in "
                "statewave-connectors for setup options."
            )
        # SDK signature uses `base_url` (not `url`); api_key + tenant_id
        # are kw-only. tenant_id is optional — most bench operators run
        # against a single-tenant server.
        self._client = StatewaveClient(
            base_url=url,
            api_key=api_key,
            tenant_id=os.environ.get("STATEWAVE_TENANT_ID"),
        )
        self._llm = LlmClient()

    def ingest(self, conversation: LocomoConversation) -> None:
        subject_id = _subject_for(conversation.id)
        # Idempotency: delete any prior bench data for this subject
        # before re-ingesting so a re-run doesn't double-count
        # episodes. The server's idempotency layer would absorb exact
        # duplicates, but a partial-prior-run could leave fragments.
        # 404 is fine — the subject didn't exist yet.
        with contextlib.suppress(Exception):
            self._client.delete_subject(subject_id)

        # Batch every session into one POST. Statewave's batch limit
        # is 100 episodes per request; LoCoMo conversations have ~5
        # sessions, so one batch is always enough.
        episodes_payload = [
            {
                "subject_id": subject_id,
                "source": EPISODE_SOURCE,
                "type": EPISODE_TYPE,
                "payload": {
                    "session_id": session_idx,
                    "messages": [
                        {
                            "speaker": turn.speaker,
                            "text": turn.text,
                            "timestamp": turn.timestamp,
                        }
                        for turn in session
                    ],
                },
                "metadata": {
                    "session_id": session_idx,
                    "session_turns": len(session),
                    "bench": "locomo",
                },
            }
            for session_idx, session in enumerate(conversation.sessions)
            if session  # skip empty sessions
        ]
        if episodes_payload:
            self._client.create_episodes_batch(episodes_payload)

        # Compile synchronously — every question wants compiled
        # memory, so the bench can't proceed until it lands. We use
        # the wait-for-completion path with a generous timeout.
        #
        # `compile_memories_wait` returns a CompileJob with .status of
        # "completed" or "failed" (the SDK only raises on timeout).
        # We treat all three failure modes (timeout, status=failed,
        # completed-but-zero-memories) as ingest failures and raise.
        # The runner catches the exception and skips the question set
        # for this (system, conversation) pair — better than silently
        # scoring 199 questions against an empty bundle and burning
        # ~$2 of answer-model spend per conversation.
        try:
            job = self._client.compile_memories_wait(subject_id, timeout=COMPILE_TIMEOUT_SEC)
        except TimeoutError as e:
            raise RuntimeError(
                f"compile timed out after {COMPILE_TIMEOUT_SEC}s for {subject_id} — "
                f"memory not fully available, skipping conversation"
            ) from e
        if job.status != "completed":
            raise RuntimeError(
                f"compile failed for {subject_id} "
                f"(job_id={job.job_id}, status={job.status}, error={job.error!r})"
            )
        if job.memories_created == 0:
            raise RuntimeError(
                f"compile produced 0 memories for {subject_id} "
                f"(job_id={job.job_id}) — server-side compiler may be misconfigured "
                f"(heuristic vs LLM, LiteLLM API keys, min-evidence thresholds). "
                f"Skipping rather than scoring against an empty bundle."
            )

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        subject_id = _subject_for(conversation_id)
        start = time.perf_counter()

        # Step 1: retrieve a ranked context bundle.
        bundle = self._client.get_context(
            subject_id,
            task=question,
            max_tokens=DEFAULT_CONTEXT_MAX_TOKENS,
        )

        # Step 2: prompt the shared answer model with the bundle.
        model = resolve_answer_model()
        prompt = (
            "Answer the question using the context below. If the answer "
            "isn't in the context, say so honestly — do not fabricate.\n\n"
            f"--- Context ---\n{bundle.assembled_context}\n\n"
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
            retrieved_context=bundle.assembled_context,
            # Statewave reports its own LLM-compile cost via server
            # logs / metrics; the bundle response itself doesn't carry
            # it, so we leave the internal-cost fields at 0 and document
            # that operators benchmarking the LLM compiler should
            # cross-reference the server's `/metrics` for full cost.
        )

    def reset(self) -> None:
        # Bench-only safety net: the runner doesn't call this between
        # conversations (per-subject scoping handles isolation), but
        # operators wanting a full purge before the next run can:
        #   for subject in client.list_subjects(prefix=SUBJECT_PREFIX):
        #       client.delete_subject(subject.id)
        # The SDK doesn't currently expose a prefix-list helper, so
        # this stays a per-conversation cleanup in `ingest`.
        return None

    def health_check(self) -> HealthResult:
        # Cheap connectivity probe: get_timeline against a known-
        # nonexistent subject. A reachable server returns an empty
        # Timeline (or a clean 404 / Pydantic-validated payload, depending
        # on server version) without burning compute. Anything else —
        # auth failure, network unreachable, bad URL — surfaces as the
        # exception we catch and report.
        try:
            self._client.get_timeline("bench:health-probe:nonexistent")
            return HealthResult(ok=True, detail="ok")
        except Exception as e:
            return HealthResult(ok=False, detail=_short(e))


def _short(err: object) -> str:
    s = str(err)
    if len(s) > 200:
        s = s[:200] + "…"
    return s.replace("\n", " ")
