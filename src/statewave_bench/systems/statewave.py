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
from ..llm import LlmClient, make_qa_prompt, resolve_answer_model
from .base import AnswerResult, HealthResult, MemorySystem

SUBJECT_PREFIX = "bench:locomo:"
EPISODE_SOURCE = "statewave-bench"
EPISODE_TYPE = "conversation"

# Cap the context bundle we ask Statewave to assemble per question.
# 2K is the empirical sweet spot on LoCoMo: dropping to 512 regressed
# multi_hop, temporal, AND open_domain (the latter was the experiment
# motivation — Mem0's win there is not driven by bundle size, see
# README methodology notes), and 8K would just inflate cost without
# improving recall on this dataset.
DEFAULT_CONTEXT_MAX_TOKENS = 2048

# Per-conversation compile uses the wait-for-completion path so the
# bench's `answer` calls always run against fully-compiled memory.
#
# 180s timeout: the LLM compiler batches ~19 sessions through gpt-4o-mini
# at concurrency 4 and typically finishes in 20-40s, but Anthropic /
# OpenAI rate-limiting + retry can stretch a worst-case batch to 90s+.
# Better to give the compile real headroom than to fail-and-retry a
# whole conversation because of a transient slowdown.
#
# 2s poll: the SDK default is 500ms, which generates ~360 status GETs
# over a 3-minute compile and floods the server logs. 2s drops that
# by 4x with negligible impact on bench wall-time.
COMPILE_TIMEOUT_SEC = 180.0
COMPILE_POLL_INTERVAL_SEC = 2.0


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
        #
        # Per-message shape is `{role, content}` — Statewave's canonical
        # shape, what `extract_payload_text` reads and what both the
        # heuristic and LLM compilers feed into their extractors. Using
        # `{speaker, text}` here would still pass schema validation but
        # `extract_payload_text` would silently produce empty strings,
        # so the compiler would persist zero memories with no error.
        # We map LoCoMo's `speaker` (a name like "Caroline") to `role`
        # so attribution survives into the compiled memory text.
        episodes_payload = [
            {
                "subject_id": subject_id,
                "source": EPISODE_SOURCE,
                "type": EPISODE_TYPE,
                "payload": {
                    "session_id": session_idx,
                    "messages": [
                        {
                            "role": turn.speaker,
                            # Prepend the LoCoMo timestamp to the message
                            # text so the LLM compiler sees absolute dates
                            # (the server's `extract_payload_text` joins
                            # `{role}: {content}` and drops every other
                            # payload field — including `timestamp`). The
                            # bench's temporal + multi-hop "When did X?"
                            # questions are unanswerable otherwise: the
                            # conversation uses relative phrases like
                            # "last Saturday" that only resolve against an
                            # absolute date stamp.
                            "content": (
                                f"[{turn.timestamp}] {turn.text}" if turn.timestamp else turn.text
                            ),
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
            job = self._client.compile_memories_wait(
                subject_id,
                timeout=COMPILE_TIMEOUT_SEC,
                poll_interval=COMPILE_POLL_INTERVAL_SEC,
            )
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

        # Step 1: retrieve context. Default = full ranked `get_context`
        # bundle (~2300 tokens, blends profile_fact + episode_summary).
        #
        # Experimental opt-in (env var STATEWAVE_BENCH_DIGEST=1): build
        # a hybrid bundle = topic_conclusions + atomic facts +
        # episode_summaries. Two shapes available via
        # STATEWAVE_BENCH_DIGEST_MODE:
        #
        #   compact (default): 2 topic_conclusions + 5 facts + 2 summaries
        #     (~600 tok). Tuned for cost-sensitive deployments.
        #
        #   fat: 3 topic_conclusions + 15 facts + 6 summaries
        #     (~1500 tok). Trades budget for broader synthesis context.
        #
        # `topic_conclusion` memories are emitted once per compile by
        # the LLM compiler's clustering pass: extracted profile_facts
        # are grouped into 4-10 thematic topics ("career", "family",
        # "health", etc), and each topic gets a ~200-token essay
        # synthesizing what we know about the subject within that
        # topic. Retrieval semantically matches the question to a
        # topic and returns the pre-computed conclusion — saving
        # answer-time work and producing richer cross-fact synthesis
        # than scattered atomic facts alone.
        #
        # Phase 2 redesign note: an earlier "session_digest" variant
        # (Phase 1) was rejected because it overlapped too much with
        # episode_summary — same per-session paraphrase axis, just a
        # different prompt. topic_conclusion uses the CROSS-SESSION
        # axis (a topic spans every session it appears in), producing
        # content that doesn't exist in any other memory kind.
        if os.environ.get("STATEWAVE_BENCH_DIGEST") == "1":
            digest_mode = os.environ.get("STATEWAVE_BENCH_DIGEST_MODE", "compact").lower()
            if digest_mode == "fat":
                topic_limit, fact_limit, summary_limit, summary_trim_chars = 3, 15, 6, 500
            else:
                # "compact" or any unrecognized value — fall back to the
                # balanced bundle.
                topic_limit, fact_limit, summary_limit, summary_trim_chars = 2, 5, 2, 300

            topics = self._client.search_memories(
                subject_id,
                kind="topic_conclusion",
                query=question,
                semantic=True,
                limit=topic_limit,
            )
            facts = self._client.search_memories(
                subject_id,
                kind="profile_fact",
                query=question,
                semantic=True,
                limit=fact_limit,
            )
            summaries = self._client.search_memories(
                subject_id,
                kind="episode_summary",
                query=question,
                semantic=True,
                limit=summary_limit,
            )
            digest_sections: list[str] = []
            if topics.memories:
                topic_lines: list[str] = []
                for m in topics.memories:
                    topic_name = (m.metadata or {}).get("topic") or "topic"
                    topic_lines.append(f"### {topic_name}\n{m.content}")
                digest_sections.append("## Subject topics\n" + "\n\n".join(topic_lines))
            if facts.memories:
                fact_lines = "\n".join(f"- {m.content}" for m in facts.memories)
                digest_sections.append(f"## Relevant facts\n{fact_lines}")
            if summaries.memories:
                # Trim each summary so a verbose paraphrase doesn't
                # blow the budget. compact: 300 chars (~70 tok), fat:
                # 500 chars (~120 tok) — fat mode wants more
                # conversational detail per summary.
                summary_lines = "\n".join(
                    f"- {m.content[:summary_trim_chars]}" for m in summaries.memories
                )
                digest_sections.append(f"## Recent context\n{summary_lines}")
            context = (
                "\n\n".join(digest_sections) if digest_sections else "(no relevant memories found)"
            )
        # Experimental opt-in (env var STATEWAVE_BENCH_HYBRID=1): build
        # a tighter "hybrid" bundle from two semantic searches —
        # 15 profile_facts (granular, citable, the bulk of the signal
        # on multi_hop) PLUS 3 episode_summaries (conversational
        # paraphrase that helps the answer model bridge facts on open-
        # ended questions). Empirically ~700-1000 tokens, ~3x cheaper
        # than the default bundle while keeping the high-precision
        # facts in the prompt. Built on the same `search_memories`
        # primitive the FACTS_ONLY experiment used; the difference is
        # we no longer drop episode_summaries entirely (FACTS_ONLY
        # showed they do load-bearing work on synthesis questions).
        # Revert by unsetting the env var.
        elif os.environ.get("STATEWAVE_BENCH_HYBRID") == "1":
            # Bumped summaries from 3 to 8 after the first hybrid test
            # showed multi_hop regress 31% — 3 summaries wasn't enough
            # synthesis context. 15 + 8 should land ~1100-1300 tokens,
            # roughly half of the 2343-token default while preserving
            # enough conversational paraphrase for the model to bridge
            # facts. If multi_hop still regresses materially, the
            # hybrid approach itself is wrong, not the count.
            facts = self._client.search_memories(
                subject_id,
                kind="profile_fact",
                query=question,
                semantic=True,
                limit=15,
            )
            summaries = self._client.search_memories(
                subject_id,
                kind="episode_summary",
                query=question,
                semantic=True,
                limit=8,
            )
            fact_lines = "\n".join(f"- {m.content}" for m in facts.memories)
            # Trim each summary so a verbose paraphrase doesn't blow
            # the budget. ~300 chars ≈ 70-80 tokens.
            summary_lines = "\n".join(f"- {m.content[:300]}" for m in summaries.memories)
            sections: list[str] = []
            if fact_lines:
                sections.append(f"## Facts about the subject\n{fact_lines}")
            if summary_lines:
                sections.append(f"## Conversation context\n{summary_lines}")
            context = "\n\n".join(sections) if sections else ("(no relevant memories found)")
        else:
            bundle = self._client.get_context(
                subject_id,
                task=question,
                max_tokens=DEFAULT_CONTEXT_MAX_TOKENS,
            )
            context = bundle.assembled_context

        # Step 2: prompt the shared answer model with the context.
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
