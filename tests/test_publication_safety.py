"""Publication-safety guarantees — the properties that make a number
trustworthy: no item silently disappears, failures are explicit
zero-score rows, incomplete/unequal runs block headline reporting,
every run writes metadata, and the report presentation is neutral.

All deterministic, no LLM / no network: failure-path tests use
single_hop questions (F1 scoring, never calls the judge) and fake
systems that raise.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from statewave_bench.coverage import (
    compute_coverage,
    coverage_complete,
    dedupe_rows,
    has_incomplete,
    missing_per_system,
)
from statewave_bench.dataset import LocomoConversation, LocomoQA
from statewave_bench.metadata import load_metadata, metadata_path
from statewave_bench.modes import EQUAL_CONTEXT_BUDGET, apply_mode_env, resolve_mode
from statewave_bench.report import IncompleteResultsError, _system_colors, render_report
from statewave_bench.runner import run_bench
from statewave_bench.systems.base import AnswerResult, MemorySystem

# ── Fakes ──────────────────────────────────────────────────────────────────


def _qa(i: int) -> LocomoQA:
    # single_hop ⇒ F1 scoring ⇒ score_answer never touches the LLM.
    return LocomoQA(
        question=f"q{i}?",
        answer="alpha",
        category="single_hop",
        category_code=1,
        evidence_dia_ids=(),
    )


def _conv(cid: str, n: int) -> LocomoConversation:
    return LocomoConversation(
        id=cid, speaker_a="A", speaker_b="B", sessions=(), qa=tuple(_qa(i) for i in range(n))
    )


class _OkSystem(MemorySystem):
    name = "ok_sys"

    def ingest(self, conversation: LocomoConversation) -> None:
        return None

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        return AnswerResult(
            answer="alpha",
            answer_model="fake",
            answer_input_tokens=10,
            answer_output_tokens=2,
            elapsed_ms=1.0,
            retrieved_context="ctx " * 25,
            retrieved_items_count=3,
        )


class _IngestFails(MemorySystem):
    name = "ingest_fails"

    def ingest(self, conversation: LocomoConversation) -> None:
        raise RuntimeError("ingest boom")

    def answer(self, conversation_id: str, question: str) -> AnswerResult:  # pragma: no cover
        raise AssertionError("answer must never be called when ingest failed")


class _AnswerAlwaysFails(MemorySystem):
    name = "answer_fails"

    def ingest(self, conversation: LocomoConversation) -> None:
        return None

    def answer(self, conversation_id: str, question: str) -> AnswerResult:
        raise RuntimeError("answer boom")


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# ── 1. No item silently disappears ─────────────────────────────────────────


def test_ingest_failure_writes_zero_rows_for_all_questions(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    run_bench(
        systems=[_IngestFails()],
        conversations=[_conv("c1", 5)],
        output_path=out,
        llm=object(),  # never used: ingest fails before any answer/judge
    )
    rows = _read(out)
    assert len(rows) == 5, "every question must get an explicit failure row"
    for r in rows:
        assert r["score"] == 0.0
        assert r["metric"] == "system_failed"
        assert r["error_type"] == "ingest_failed"
        assert "ingest boom" in r["error_message"]


def test_answer_failure_writes_zero_row(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    run_bench(
        systems=[_AnswerAlwaysFails()],
        conversations=[_conv("c1", 3)],
        output_path=out,
        llm=object(),
    )
    rows = _read(out)
    assert len(rows) == 3
    assert all(r["metric"] == "system_failed" and r["score"] == 0.0 for r in rows)
    assert all(r["error_type"] == "answer_failed" for r in rows)


def test_repeated_failures_still_record_every_remaining_question(tmp_path: Path) -> None:
    # > FAILURE_STREAK_THRESHOLD questions: the circuit-breaker stops
    # *calling* the dead system but every remaining question must still
    # produce an explicit failure row (denominator never shrinks).
    out = tmp_path / "r.jsonl"
    n = 20
    run_bench(
        systems=[_AnswerAlwaysFails()],
        conversations=[_conv("c1", n)],
        output_path=out,
        llm=object(),
    )
    rows = _read(out)
    assert len(rows) == n, "circuit-breaker must not drop remaining questions"
    assert {r["question_idx"] for r in rows} == set(range(n))
    assert all(r["score"] == 0.0 and r["metric"] == "system_failed" for r in rows)
    cov = compute_coverage(rows)["answer_fails"]
    assert cov.expected_questions == n
    assert cov.completed_rows == n
    assert cov.scored_rows == 0  # all failed → none successfully scored
    assert not cov.complete


def test_metadata_sidecar_written_with_bench_affecting_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SWB_STATEWAVE_CONTEXT_MAX_TOKENS", "2048")
    monkeypatch.setenv("SWB_SCORING_MODE", "strict")
    out = tmp_path / "run.jsonl"
    run_bench(
        systems=[_OkSystem()],
        conversations=[_conv("c1", 2)],
        output_path=out,
        llm=object(),
        bench_mode="equal_context_budget",
        dataset_url="http://example/locomo.json",
    )
    meta = load_metadata(out)
    assert metadata_path(out) == tmp_path / "run.metadata.json"
    assert meta is not None
    assert meta["bench_mode"] == "equal_context_budget"
    assert meta["dataset_url"] == "http://example/locomo.json"
    assert meta["env"]["SWB_STATEWAVE_CONTEXT_MAX_TOKENS"] == "2048"
    assert meta["systems"] == ["ok_sys"]
    assert meta["n_questions"] == 2


# ── coverage module ────────────────────────────────────────────────────────


def _row(system: str, cid: str, qi: int, **kw: object) -> dict:
    # Mirror the real runner row schema so polars aggregation in
    # report.py has the columns it expects.
    base = {
        "system": system,
        "conversation_id": cid,
        "question_idx": qi,
        "question": "q?",
        "category": "single_hop",
        "ground_truth": "g",
        "prediction": "p",
        "score": 1.0,
        "metric": "f1",
        "elapsed_ms": 1.0,
        "answer_model": "fake",
        "answer_input_tokens": 1,
        "answer_output_tokens": 1,
        "internal_input_tokens": 0,
        "internal_output_tokens": 0,
        "retrieved_context_chars": 100,
        "retrieved_context_tokens_estimate": 25,
        "retrieved_items_count": 2,
    }
    base.update(kw)
    return base


def test_missing_per_system_detects_unequal_sets() -> None:
    rows = [_row("a", "c", 0), _row("a", "c", 1), _row("b", "c", 0)]
    missing = missing_per_system(rows)
    assert missing == {"b": {("c", 1)}}


def test_has_incomplete_and_dedupe_keep_best() -> None:
    rows = [
        _row("a", "c", 0, score=None, metric="judge_failed"),
        _row("a", "c", 0, score=1.0, metric="f1"),  # resume re-ran it
    ]
    assert has_incomplete([rows[0]]) is True
    deduped = dedupe_rows(rows)
    assert len(deduped) == 1 and deduped[0]["metric"] == "f1"  # best wins
    assert has_incomplete(deduped) is False


def test_coverage_complete_true_only_when_equal_and_scored() -> None:
    good = [_row("a", "c", 0), _row("a", "c", 1), _row("b", "c", 0), _row("b", "c", 1)]
    assert coverage_complete(good) is True
    bad = [*good[:-1], _row("b", "c", 1, score=None, metric="judge_failed")]
    assert coverage_complete(bad) is False


# ── report blocking + neutrality ───────────────────────────────────────────


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_report_blocks_on_judge_failed(tmp_path: Path) -> None:
    f = tmp_path / "r.jsonl"
    _write_jsonl(
        f,
        [
            _row("a", "c", 0, answer_input_tokens=1, answer_output_tokens=1, elapsed_ms=1.0),
            _row(
                "a",
                "c",
                1,
                score=None,
                metric="judge_failed",
                answer_input_tokens=1,
                answer_output_tokens=1,
                elapsed_ms=1.0,
            ),
        ],
    )
    with pytest.raises(IncompleteResultsError):
        render_report(results_path=f, output_dir=tmp_path / "o")
    # allow_incomplete renders but stamps NOT PUBLICATION-SAFE
    render_report(results_path=f, output_dir=tmp_path / "o2", allow_incomplete=True)
    md = (tmp_path / "o2" / "results-summary.md").read_text()
    assert "NOT PUBLICATION-SAFE" in md


def test_report_blocks_on_unequal_sets(tmp_path: Path) -> None:
    f = tmp_path / "r.jsonl"
    _write_jsonl(
        f,
        [
            _row("a", "c", 0, answer_input_tokens=1, answer_output_tokens=1, elapsed_ms=1.0),
            _row("a", "c", 1, answer_input_tokens=1, answer_output_tokens=1, elapsed_ms=1.0),
            _row("b", "c", 0, answer_input_tokens=1, answer_output_tokens=1, elapsed_ms=1.0),
        ],
    )
    with pytest.raises(IncompleteResultsError):
        render_report(results_path=f, output_dir=tmp_path / "o")


def test_report_palette_is_neutral_not_statewave_purple() -> None:
    colors = _system_colors(["statewave", "mem0", "zep", "naive", "no_memory"])
    # The old hand-picked Statewave purple must be gone, and the mapping
    # must be deterministic by sorted name (no vendor privileged).
    assert "#7c3aed" not in colors.values()
    assert colors == _system_colors(["no_memory", "zep", "mem0", "naive", "statewave"])


# ── modes ──────────────────────────────────────────────────────────────────


def test_equal_context_budget_seeds_knobs_without_overriding_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SWB_STATEWAVE_CONTEXT_MAX_TOKENS", raising=False)
    monkeypatch.setenv("MEM0_TOP_K", "99")  # operator-set: must be preserved
    seeded = apply_mode_env(resolve_mode("equal_context_budget"))
    import os

    assert os.environ["SWB_STATEWAVE_CONTEXT_MAX_TOKENS"] == "2048"
    assert os.environ["MEM0_TOP_K"] == "99"  # not overridden
    assert "SWB_STATEWAVE_CONTEXT_MAX_TOKENS" in seeded
    assert "MEM0_TOP_K" not in seeded
    assert resolve_mode(None) in ("vendor_defaults", EQUAL_CONTEXT_BUDGET)


# ── aggregate refuses unequal sets (real CLI behaviour) ────────────────────


def test_aggregate_runs_refuses_unequal_question_sets(tmp_path: Path) -> None:
    r1 = tmp_path / "run-1.jsonl"
    r2 = tmp_path / "run-2.jsonl"
    _write_jsonl(r1, [_row("a", "c", 0), _row("a", "c", 1), _row("b", "c", 0), _row("b", "c", 1)])
    # run-2: system b is missing question 1 → unequal set within the run
    _write_jsonl(r2, [_row("a", "c", 0), _row("a", "c", 1), _row("b", "c", 0)])
    script = Path(__file__).resolve().parent.parent / "scripts" / "aggregate_runs.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(r1), str(r2)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "not\npublication-safe" in proc.stderr or "publication-safe" in proc.stderr
    ok = subprocess.run(
        [sys.executable, str(script), str(r1), str(r2), "--allow-incomplete"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0
    assert "NOT PUBLICATION-SAFE" in ok.stdout
