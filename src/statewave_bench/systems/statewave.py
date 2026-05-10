"""Statewave adapter — talks to a running Statewave server via the
Python SDK (`statewave-py`).

Setup: operator runs a Statewave server (e.g. via the helm chart or
docker compose recipe in `statewave-connectors`), exports
`STATEWAVE_URL` + `STATEWAVE_API_KEY`, and the adapter connects.

Bench-specific subject: every conversation gets a unique subject id
of the form `bench:locomo:<conversation_id>`. That keeps bench data
isolated from any production memories the same Statewave instance
might be holding, and makes it trivial to purge afterward
(`DELETE /v1/subjects/bench:locomo:*`).

Implementation status: SCAFFOLDED. The methods below are wired to the
SDK calls but the exact shape of (a) episode payload encoding, (b)
context-bundle assembly call, and (c) compile-mode selection wants
one careful pass under real running data. Wave A2 lands the actual
end-to-end run; this stub establishes the surface so reviewers see
the intended shape now.
"""

from __future__ import annotations

import os
import time

from ..dataset import LocomoConversation
from ..llm import LlmClient, resolve_answer_model
from .base import AnswerResult, MemorySystem


SUBJECT_PREFIX = "bench:locomo:"


class StatewaveSystem(MemorySystem):
    name = "statewave"

    def __init__(self) -> None:
        # `statewave` is the official Python SDK on PyPI. The import
        # is lazy so other systems can run without the SDK installed
        # locally (operators benchmarking only Mem0 vs Zep don't need
        # a Statewave server).
        from statewave import StatewaveClient  # type: ignore[import-not-found]

        url = os.environ.get("STATEWAVE_URL")
        api_key = os.environ.get("STATEWAVE_API_KEY")
        if not url:
            raise RuntimeError(
                "STATEWAVE_URL not set. Point at a running Statewave server, "
                "e.g. http://localhost:8000 — see the deployment guide in "
                "statewave-connectors for setup options."
            )
        self._client = StatewaveClient(url=url, api_key=api_key)
        self._llm = LlmClient()

    def ingest(self, conversation: LocomoConversation) -> None:
        # Wave A2 will:
        #   1. POST one episode per session via /v1/episodes (each
        #      session becomes one episode whose payload is the
        #      session's turns concatenated)
        #   2. Trigger /v1/memories/compile on the subject so the
        #      memory layer materializes facts/decisions/etc. before
        #      questions land
        #
        # Open question for that wave: do we use the heuristic
        # compiler (zero LLM cost during ingest, faster) or the LLM
        # compiler (more expensive but produces richer typed memories
        # — matches what a real Statewave deployment would do)? The
        # honest comparison is probably "run both, report both" —
        # operators decide which mode matches their production setup.
        raise NotImplementedError(
            "Statewave ingest is scaffolded but not yet wired — Wave A2."
        )

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        # Wave A2 will:
        #   1. GET /v1/context with subject_id + task=question to
        #      assemble the ranked, token-bounded bundle
        #   2. Pass `bundle.assembled_context` + question to the
        #      shared answer model (same model every system uses)
        #   3. Return the answer plus the bundle's token estimate
        #      so the report can show context-efficiency alongside
        #      raw quality
        del conversation_id, question
        model = resolve_answer_model()
        elapsed_ms = (time.perf_counter() - time.perf_counter()) * 1000
        del model, elapsed_ms
        raise NotImplementedError(
            "Statewave answer is scaffolded but not yet wired — Wave A2."
        )

    def reset(self) -> None:
        # Bench-only safety net: delete every subject whose id starts
        # with `bench:locomo:`. The Statewave server already exposes
        # DELETE /v1/subjects/<id> (used by the support-agent demo's
        # reset flow); we'd iterate the list once at start-of-run.
        # Not wired in the scaffold — Wave A2.
        return None
