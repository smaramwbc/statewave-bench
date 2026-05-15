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
from datetime import UTC

from dateutil import parser as _dateparser

from ..dataset import LocomoConversation
from ..llm import LlmClient, make_qa_prompt, resolve_answer_model
from .base import AnswerResult, HealthResult, MemorySystem

# Zep restricts user_id charset (alphanumeric + dash); LoCoMo ids are
# usually clean but normalize to be safe.
_UNSAFE_USER_ID = re.compile(r"[^a-zA-Z0-9-]+")


def _to_iso8601(timestamp: str | None) -> str | None:
    """Convert LoCoMo's free-form timestamp ("1:56 pm on 8 May, 2023") to
    RFC 3339 with timezone ("2023-05-08T13:56:00+00:00") for Zep's
    `Message.created_at`.

    LoCoMo's `.timestamp` is a human-readable string from the dataset
    ("11:00 am on 25 Dec, 2022", "8 May, 2023", sometimes empty). Zep's
    API rejects timestamps without a timezone with a 400 "invalid json"
    (RFC 3339 requires the offset), and treats unparseable values as
    missing — falling back to ingest time, which is exactly the bug
    this layer exists to prevent. So: parse fuzzy, attach UTC if the
    parse produced a naive datetime, then isoformat.
    """
    if not timestamp:
        return None
    try:
        dt = _dateparser.parse(timestamp, fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    # dateutil's stubs declare datetime methods as returning Any in
    # some paths; the runtime is always str. Cast to satisfy mypy.
    return str(dt.isoformat())


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

# Graph-extraction settle: poll `graph.edge.get_by_user_id` — DIRECT
# enumeration of the user's entire edge set — until the total edge
# count stops growing for N consecutive polls.
#
# CRITICAL FAIRNESS FIX (two iterations):
#
#   v1 bug: polled `thread.get_user_context()` (the USER_SUMMARY
#   rollup). That summary stabilizes within ~15s — Zep generates it
#   from an early partial extraction — while the searchable graph keeps
#   extracting for much longer. The bench scored all 199 questions
#   against a half-built graph; Zep collapsed to ~0.03 on every
#   non-adversarial category.
#
#   v2 bug: switched to polling `graph.search(limit=20)` (the actual
#   retrieval path). But `graph.search` is relevance-capped at `limit`,
#   so the returned edge count saturates at 20 almost instantly for
#   any non-empty graph. (edge_count, fact_chars) stabilized while the
#   graph was still extracting facts that mattered for SPECIFIC
#   questions — same bug class, less severe. The canonical
#   support-group question started passing (that fact extracts early)
#   but sunrise/camping facts (extract later) were still scored 0
#   while a live probe minutes later returned them correctly.
#
#   v3 (this): poll `graph.edge.get_by_user_id(user_id, limit=N)` —
#   direct enumeration of the FULL edge set, not a relevance-capped
#   search. The count grows during extraction (0 -> 50 -> 200 -> 476)
#   and plateaus when done. No saturation, no query-dependence — the
#   only honest "is the whole graph built" signal Zep's API exposes.
#   (conv-26's settled graph has 476 edges; the default page size is
#   100, so the fetch limit must be set well above any conversation's
#   true edge count.)
#
#   Stability is detected by PLATEAU WITH TOLERANCE, not exact-count
#   match and not strict-max. `get_by_user_id` returns a slightly
#   different count on consecutive calls even on a fully-settled graph
#   (Zep internal ordering / eventual-consistency wobble — observed
#   474/476/475/478 on a static graph). Two naive checks both fail:
#     - exact "N identical counts" almost never triggers on wobble
#     - strict running-max resets the streak every time wobble pokes a
#       new high (478 > 476), so it never plateaus either (observed:
#       351s+ to settle a static graph, then no settle at all).
#   So we track a `plateau_base` and only treat a poll as REAL growth
#   when it exceeds the base by more than `_settle_tolerance(base)` —
#   a band wide enough to swallow consistency wobble (a handful of
#   edges) but far below a genuine extraction burst (LoCoMo adds
#   dozens of edges per burst). Real growth resets the base + streak;
#   in-band wobble advances the streak. Settles in ~30s on a done
#   graph, waits honestly while edges are still being added.
ZEP_GRAPH_SETTLE_POLL_SEC = 5.0
# 5 (not 3): an in-band streak of 5 polls = 25s of no real growth.
# LoCoMo edge extraction arrives in bursts; gaps between bursts are
# typically <25s, so 5 avoids exiting in a mid-extraction lull while
# still settling a truly-done graph in ~30s.
ZEP_GRAPH_SETTLE_STABLE_COUNT = 5
# Plateau tolerance band. A poll only counts as REAL growth (resetting
# the plateau streak) when edge_count exceeds plateau_base by more
# than max(absolute, relative * base). Consistency wobble is a handful
# of edges; a genuine extraction burst is dozens. 15 absolute + 4%
# relative cleanly separates them across conversation sizes (conv-26:
# 4% of 476 ~= 19, so any single poll within ~19 of the base is
# treated as wobble, not growth).
ZEP_GRAPH_SETTLE_TOLERANCE_ABS = 15
ZEP_GRAPH_SETTLE_TOLERANCE_REL = 0.04
# Minimum edges before the plateau detector starts. Guards the
# degenerate case where early 0-edge polls would look like a plateau
# and exit instantly. A 19-session LoCoMo conversation produces
# hundreds of edges; 5 is a conservative "extraction has started"
# floor.
ZEP_GRAPH_SETTLE_MIN_EDGES = 5
# Edge-fetch page size for the settle poll. Must exceed any single
# conversation's true edge count or the count saturates at the page
# cap and we get the v2 saturation bug again. conv-26 settles at 476
# edges; 2000 is generous headroom for denser conversations while
# still one cheap API call per poll.
ZEP_SETTLE_EDGE_FETCH_LIMIT = 2000
# 900s — a 19-session LoCoMo conversation's edge extraction can run
# 60-180s+; the settle detector exits early once the edge count
# plateaus, so this ceiling only costs real wall time on the
# conversations that genuinely need it. Better to wait honestly than
# to score against an unsettled graph (the bug this fix exists for).
ZEP_GRAPH_SETTLE_TIMEOUT_SEC = 900.0

# graph.search retrieval budget — matches Statewave's 2048-token context
# bundle at ~4 chars per token, so the answer model sees comparable
# context sizes across systems. `limit` is a count cap; `max_characters`
# is the actual prompt budget. `reranker='mmr'` with mmr_lambda=0.5
# trades relevance for diversity 50/50 — important for multi_hop
# questions that need 3-6 distinct facts (a pure relevance reranker
# returns near-duplicates of the same fact at the top of the list).
ZEP_SEARCH_LIMIT = 20
ZEP_SEARCH_MAX_CHARS = 8192
ZEP_SEARCH_RERANKER = "mmr"
ZEP_SEARCH_MMR_LAMBDA = 0.5


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
            #
            # `created_at` is Zep's canonical temporal anchor — Graphiti
            # uses it as each message's reference_time when extracting
            # facts and setting their `valid_at`. WITHOUT it, every
            # message defaults to ingest time (today), so every fact's
            # `valid_at` lands in 2026 and Zep predicts events from a
            # 2023 conversation happened "yesterday relative to today."
            # Confirmed in the pre-fix run: Zep literally returned
            # "Caroline attended the support group on May 11, 2026"
            # for an event grounded in May 2023. The bench's `[date]`
            # content prefix is informational text Graphiti may or may
            # not parse; `created_at` is the explicit channel.
            messages = [
                Message(
                    content=(f"[{turn.timestamp}] {turn.text}" if turn.timestamp else turn.text),
                    role=("assistant" if turn.speaker.lower() == "assistant" else "user"),
                    name=turn.speaker,
                    created_at=_to_iso8601(turn.timestamp),
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
            #
            # Per-chunk exceptions propagate to the runner, which drops
            # the whole (system, conv) pair. Matches Statewave's
            # contract: a partial graph that scores against missing
            # facts is worse than recording nothing, because the
            # leaderboard would mix "Zep lost the fact fairly" with
            # "Zep never saw the fact."
            for chunk_start in range(0, len(messages), ZEP_MAX_MESSAGES_PER_REQUEST):
                chunk = messages[chunk_start : chunk_start + ZEP_MAX_MESSAGES_PER_REQUEST]
                response = self._client.thread.add_messages(
                    thread_id,
                    messages=chunk,
                )
                task_id = getattr(response, "task_id", None)
                if task_id:
                    task_ids.append(task_id)

        # Block until all graph-build tasks are terminal (completed or
        # failed). Mirrors the Statewave compile-wait contract: ingest
        # only returns once memory is queryable. Without this the bench
        # would race the async build and score against an empty graph.
        self._wait_for_tasks(task_ids, conversation_id=conversation.id)

        # Tasks completing only means messages were ingested. The graph
        # EDGE extractor keeps populating facts for another ~60-180s.
        # Poll the actual retrieval path (`graph.search` over edges,
        # exactly what `answer()` uses) until the edge set stops growing
        # for a few consecutive samples before declaring the graph
        # "settled". user_id (not thread_id) is graph.search's scope key.
        self._wait_for_graph_settle(user_id=user_id, conversation_id=conversation.id)

    def _wait_for_tasks(self, task_ids: list[str], *, conversation_id: str) -> None:
        if not task_ids:
            return

        pending = list(task_ids)
        failed: list[tuple[str, str]] = []
        elapsed = 0.0
        while pending and elapsed < ZEP_TASK_TIMEOUT_SEC:
            still_pending: list[str] = []
            for tid in pending:
                # Transient poll exceptions become "still pending" — a
                # one-off network blip shouldn't fail the whole
                # conversation. If polling is broken end-to-end the
                # timeout below catches it and raises.
                try:
                    task = self._client.task.get(tid)
                except Exception:
                    still_pending.append(tid)
                    continue
                status = (task.status or "").lower()
                if status in ZEP_TERMINAL_STATUSES:
                    if status not in ("completed", "success", "succeeded"):
                        failed.append((tid, status))
                    continue
                still_pending.append(tid)
            pending = still_pending
            if pending:
                time.sleep(ZEP_TASK_POLL_INTERVAL_SEC)
                elapsed += ZEP_TASK_POLL_INTERVAL_SEC
        if failed:
            details = ", ".join(f"{tid}={status!r}" for tid, status in failed)
            raise RuntimeError(
                f"zep ingest: {len(failed)} task(s) ended in a non-success state "
                f"for {conversation_id} ({details}). Refusing to score against a "
                f"partial graph."
            )
        if pending:
            raise RuntimeError(
                f"zep ingest: {len(pending)} task(s) still pending after "
                f"{ZEP_TASK_TIMEOUT_SEC:.0f}s for {conversation_id}. Refusing to "
                f"score against a partial graph."
            )

    def _wait_for_graph_settle(self, *, user_id: str, conversation_id: str) -> None:
        """Block until the user's edge graph stops growing.

        Polls `graph.edge.get_by_user_id` — DIRECT enumeration of the
        user's entire edge set — and waits until the total edge count
        is stable for `ZEP_GRAPH_SETTLE_STABLE_COUNT` consecutive polls.

        This is the v3 readiness signal. v1 watched the USER_SUMMARY
        rollup (stabilizes ~15s before the graph is query-ready); v2
        watched `graph.search(limit=20)` (count saturates at 20
        instantly, masking ongoing extraction). Direct edge
        enumeration is the only honest "whole graph built" signal Zep
        exposes. Stability is plateau-with-tolerance (no growth beyond
        a wobble band for N consecutive polls), NOT exact-count match
        and NOT strict-max — see the `ZEP_GRAPH_SETTLE_*` comment block
        for the full incident writeup including the count-wobble
        rationale.
        """
        plateau_base: int | None = None
        in_band_streak = 0
        elapsed = 0.0
        while elapsed < ZEP_GRAPH_SETTLE_TIMEOUT_SEC:
            try:
                edges = self._client.graph.edge.get_by_user_id(
                    user_id,
                    limit=ZEP_SETTLE_EDGE_FETCH_LIMIT,
                )
            except Exception as e:
                raise RuntimeError(
                    f"zep graph-settle poll failed for {conversation_id} (user={user_id}): {e}"
                ) from e
            edge_count = len(edges) if edges else 0
            # Non-empty floor: don't let the plateau detector start
            # while the edge set is still empty/trivial (extraction
            # hasn't really begun). Without this, an early run of
            # 0-edge polls would look like a "plateau" and exit
            # instantly — the exact premature-settle bug.
            if plateau_base is None:
                if edge_count < ZEP_GRAPH_SETTLE_MIN_EDGES:
                    time.sleep(ZEP_GRAPH_SETTLE_POLL_SEC)
                    elapsed += ZEP_GRAPH_SETTLE_POLL_SEC
                    continue
                plateau_base = edge_count
            tolerance = max(
                ZEP_GRAPH_SETTLE_TOLERANCE_ABS,
                int(ZEP_GRAPH_SETTLE_TOLERANCE_REL * plateau_base),
            )
            if edge_count > plateau_base + tolerance:
                # Real extraction growth (burst of dozens of edges):
                # rebase the plateau and reset the streak.
                plateau_base = edge_count
                in_band_streak = 0
            else:
                # Within the wobble band of the base (or below it):
                # extraction has stopped producing meaningful new
                # edges. Track the high end of the band so the base
                # reflects the true ceiling, but DON'T reset the
                # streak — wobble is not growth.
                if edge_count > plateau_base:
                    plateau_base = edge_count
                in_band_streak += 1
                if in_band_streak >= ZEP_GRAPH_SETTLE_STABLE_COUNT:
                    return
            time.sleep(ZEP_GRAPH_SETTLE_POLL_SEC)
            elapsed += ZEP_GRAPH_SETTLE_POLL_SEC
        raise RuntimeError(
            f"zep edge graph still growing after {ZEP_GRAPH_SETTLE_TIMEOUT_SEC:.0f}s "
            f"for {conversation_id}. Refusing to score against a graph that "
            f"hasn't settled."
        )

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        user_id = _user_id_for(conversation_id)
        start = time.perf_counter()

        # Step 1: retrieve question-conditioned facts from Zep's graph.
        #
        # Earlier versions of this adapter called `thread.get_user_context`
        # which returns a static, question-agnostic thread summary —
        # the wrong API for QA over memories. Zep's docs recommend
        # `graph.search` for retrieval-against-the-graph; that's what
        # production Zep-powered agents do, and it's what cross-system
        # fairness requires (Statewave + Mem0 both use question-
        # conditioned retrieval). MMR reranker at lambda=0.5 trades
        # relevance for diversity 50/50, which matters for multi_hop
        # questions that need several distinct facts (a pure-relevance
        # reranker returns near-duplicates of the same fact).
        #
        # Retrieval errors propagate to the runner, which marks the
        # question as a failure. Matches Statewave's behavior: a
        # broken retrieval shouldn't be silently scored as a wrong
        # answer (which conflates "Zep doesn't know" with "Zep API is
        # down").
        results = self._client.graph.search(
            query=question,
            user_id=user_id,
            scope="edges",
            limit=ZEP_SEARCH_LIMIT,
            max_characters=ZEP_SEARCH_MAX_CHARS,
            reranker=ZEP_SEARCH_RERANKER,
            mmr_lambda=ZEP_SEARCH_MMR_LAMBDA,
        )
        context = _format_edges_as_context(results)

        # Step 2: prompt the shared answer model with the bundle.
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


def _format_edges_as_context(results: object) -> str:
    """Format `graph.search` results into a prompt-ready context block.

    Edges carry `.fact` (the textual fact) plus optional `.valid_at`
    (when the fact became true) and `.invalid_at` (when it stopped
    being true). The temporal subset of LoCoMo needs these dates —
    Zep's `valid_at` is the canonical source — so we prepend them
    where present, falling back to plain fact text otherwise.
    """
    edges = getattr(results, "edges", None) or []
    lines: list[str] = []
    for edge in edges:
        fact = getattr(edge, "fact", None)
        if not fact:
            continue
        valid_at = getattr(edge, "valid_at", None)
        invalid_at = getattr(edge, "invalid_at", None)
        # Date prefix: [valid_at → invalid_at] fact, or just [valid_at] fact,
        # or no prefix if neither is set.
        if valid_at and invalid_at:
            prefix = f"[{valid_at} → {invalid_at}] "
        elif valid_at:
            prefix = f"[{valid_at}] "
        else:
            prefix = ""
        lines.append(f"- {prefix}{fact}")
    return "\n".join(lines) if lines else "(no relevant memories found)"


def _short(err: object) -> str:
    s = str(err)
    if len(s) > 200:
        s = s[:200] + "…"
    return s.replace("\n", " ")
