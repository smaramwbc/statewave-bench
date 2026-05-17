"""Coverage + completeness accounting shared by `swb report` and
`scripts/aggregate_runs.py`.

A benchmark number is only publication-safe if every system answered
the *same* question set and no row silently vanished. This module is
the single source of truth for:

  - the canonical expected key set ((conversation_id, question_idx)),
  - per-system coverage stats (expected / completed / scored / failed /
    judge_failed and the two coverage ratios),
  - which systems are missing which keys (unequal-set detection),
  - whether a result set contains incomplete rows (null score /
    judge_failed) that must block headline reporting.

It deliberately treats the *union* of keys seen across all systems as
the expected set: with the runner now writing an explicit failure row
for every attempted item, a key missing from a system genuinely means
that item never produced a row for it — exactly what we must surface,
not paper over.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Rows the runner writes for non-success outcomes. `system_failed`
# carries score 0.0 (it counts against the system — a crash is a
# wrong answer, not a freebie). `judge_failed` carries score null
# (the answer is in hand; only the judge call failed) and must NOT
# enter a headline mean.
FAILED_METRIC = "system_failed"
JUDGE_FAILED_METRIC = "judge_failed"

Key = tuple[str, int]  # (conversation_id, question_idx)


@dataclass(frozen=True)
class CoverageStats:
    system: str
    expected_questions: int
    completed_rows: int
    scored_rows: int
    failed_rows: int
    judge_failed_rows: int

    @property
    def coverage(self) -> float:
        if self.expected_questions == 0:
            return 0.0
        return self.completed_rows / self.expected_questions

    @property
    def scored_coverage(self) -> float:
        if self.expected_questions == 0:
            return 0.0
        return self.scored_rows / self.expected_questions

    @property
    def complete(self) -> bool:
        """Publication-safe iff every expected item produced a real
        score (no missing rows, no judge_failed/null)."""
        return (
            self.completed_rows == self.expected_questions
            and self.scored_rows == self.expected_questions
        )


def _key(row: dict[str, object]) -> Key:
    return (str(row["conversation_id"]), int(str(row["question_idx"])))


def _row_rank(row: dict[str, object]) -> int:
    """Best-row preference when the same (system, key) appears more than
    once (a resume that re-ran a judge_failed item appends a fresh row
    rather than mutating the old one). Higher wins: a real score beats
    judge_failed beats an explicit failure."""
    metric = row.get("metric")
    if isinstance(row.get("score"), (int, float)) and metric not in (
        FAILED_METRIC,
        JUDGE_FAILED_METRIC,
    ):
        return 3
    if metric == JUDGE_FAILED_METRIC or row.get("score") is None:
        return 2
    if metric == FAILED_METRIC:
        return 1
    return 0


def dedupe_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse duplicate (system, conversation_id, question_idx) rows,
    keeping the most-complete one. Append-only JSONL + `--resume`
    re-running judge_failed items can legitimately produce two rows for
    one key; every consumer must agree on which one counts."""
    best: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in rows:
        k = (str(row["system"]), *_key(row))
        cur = best.get(k)
        if cur is None or _row_rank(row) >= _row_rank(cur):
            best[k] = row
    return list(best.values())


def expected_keys(rows: Iterable[dict[str, object]]) -> set[Key]:
    """Canonical expected (conversation_id, question_idx) set = the
    union across every system in the file."""
    return {_key(r) for r in rows}


def compute_coverage(rows: Iterable[dict[str, object]]) -> dict[str, CoverageStats]:
    rows = dedupe_rows(rows)
    expected = expected_keys(rows)
    by_system: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by_system.setdefault(str(r["system"]), []).append(r)

    stats: dict[str, CoverageStats] = {}
    for system, srows in by_system.items():
        scored = sum(
            1
            for r in srows
            if isinstance(r.get("score"), (int, float))
            and r.get("metric") not in (FAILED_METRIC, JUDGE_FAILED_METRIC)
        )
        failed = sum(1 for r in srows if r.get("metric") == FAILED_METRIC)
        judge_failed = sum(
            1 for r in srows if r.get("metric") == JUDGE_FAILED_METRIC or r.get("score") is None
        )
        stats[system] = CoverageStats(
            system=system,
            expected_questions=len(expected),
            completed_rows=len(srows),
            scored_rows=scored,
            failed_rows=failed,
            judge_failed_rows=judge_failed,
        )
    return stats


def missing_per_system(rows: Iterable[dict[str, object]]) -> dict[str, set[Key]]:
    """Per system: expected keys (union across systems) that the system
    has no row for. Empty everywhere ⇒ equal question sets."""
    rows = dedupe_rows(rows)
    expected = expected_keys(rows)
    seen: dict[str, set[Key]] = {}
    for r in rows:
        seen.setdefault(str(r["system"]), set()).add(_key(r))
    return {sys_: expected - keys for sys_, keys in seen.items() if expected - keys}


def has_incomplete(rows: Iterable[dict[str, object]]) -> bool:
    """True if any kept row is judge_failed / has a null score — these
    must block headline reporting unless explicitly allowed."""
    for r in dedupe_rows(rows):
        if r.get("metric") == JUDGE_FAILED_METRIC or r.get("score") is None:
            return True
    return False


def coverage_complete(rows: Iterable[dict[str, object]]) -> bool:
    """Publication-safe overall: equal sets across systems AND every
    system fully scored (no missing, no failed-as-unscored, no
    judge_failed)."""
    rows = list(dedupe_rows(rows))
    if missing_per_system(rows):
        return False
    if has_incomplete(rows):
        return False
    return all(s.complete for s in compute_coverage(rows).values())
