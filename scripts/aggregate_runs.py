"""Aggregate N independent benchmark runs into per-run, mean, and
standard-deviation tables.

`swb run` executes one pass over the dataset and `swb report` renders
one JSONL. Public benchmarking needs the variance picture across
repeated passes — a single pass can land on a lucky or unlucky
answer-model / judge sample. This tool takes the JSONL outputs of
several independent `swb run` invocations and reports:

  - per-run overall score (and excl-adversarial) for every system
  - the mean across runs
  - the population standard deviation across runs (the stability signal)

It does NOT re-score anything: it reads the `score` column each run
already wrote, so the aggregate reflects exactly the scoring mode
those runs used. Run the passes first, then point this at them:

    uv run swb run -s statewave -s mem0 -o results/run-1.json
    uv run swb run -s statewave -s mem0 -o results/run-2.json
    ...
    uv run python scripts/aggregate_runs.py results/run-*.json

Exit status is non-zero if a file is missing or unparseable so a
shell loop fails loudly rather than averaging a partial set.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ADVERSARIAL = "adversarial"


def _load(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_scores(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Per-system {overall, excl_adv, <category>...} means for one run.

    Mirrors `swb report`'s aggregation: simple mean over the `score`
    column, plus a mean that drops the adversarial category (the
    refusal-only subset every long-term-memory comparison excludes).
    """
    by_system: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_system[str(r["system"])].append(r)

    out: dict[str, dict[str, float]] = {}
    for system, srows in by_system.items():
        scored = [r for r in srows if isinstance(r.get("score"), (int, float))]
        if not scored:
            continue
        overall = sum(float(r["score"]) for r in scored) / len(scored)
        non_adv = [r for r in scored if r.get("category") != ADVERSARIAL]
        excl = sum(float(r["score"]) for r in non_adv) / len(non_adv) if non_adv else float("nan")
        per_cat: dict[str, list[float]] = defaultdict(list)
        for r in scored:
            per_cat[str(r["category"])].append(float(r["score"]))
        cat_means = {c: sum(v) / len(v) for c, v in per_cat.items()}
        out[system] = {"overall": overall, "excl_adv": excl, **cat_means}
    return out


def _fmt(x: float) -> str:
    return "—" if x != x else f"{x:.3f}"  # x != x is the NaN test


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="One JSONL per run (>=1).")
    args = ap.parse_args()

    paths = sorted(args.runs)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"ERROR: missing run file(s): {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    per_run: list[dict[str, dict[str, float]]] = []
    for p in paths:
        try:
            per_run.append(_run_scores(_load(p)))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"ERROR: {p} unparseable: {e}", file=sys.stderr)
            return 1

    systems = sorted({s for run in per_run for s in run})
    n = len(paths)

    print(f"Aggregating {n} run(s): {', '.join(p.name for p in paths)}")
    print()
    print("## Per-run overall (excl. adversarial)")
    header = "| System | " + " | ".join(f"run {i + 1}" for i in range(n)) + " | mean | stddev |"
    print(header)
    print("|" + "---|" * (n + 3))
    for s in systems:
        vals = [run.get(s, {}).get("excl_adv", float("nan")) for run in per_run]
        clean = [v for v in vals if v == v]
        mean = statistics.fmean(clean) if clean else float("nan")
        std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
        cells = " | ".join(_fmt(v) for v in vals)
        print(f"| {s} | {cells} | {_fmt(mean)} | {std:.3f} |")

    print()
    print("## Mean overall (all categories) across runs")
    print("| System | mean overall | stddev | n_runs |")
    print("|---|---|---|---|")
    for s in systems:
        vals = [run.get(s, {}).get("overall", float("nan")) for run in per_run]
        clean = [v for v in vals if v == v]
        mean = statistics.fmean(clean) if clean else float("nan")
        std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
        print(f"| {s} | {_fmt(mean)} | {std:.3f} | {len(clean)} |")

    # Per-category cross-run aggregation. Reserved keys (overall,
    # excl_adv) are not categories; everything else a run produced is.
    reserved = {"overall", "excl_adv"}
    categories = sorted(
        {c for run in per_run for sysmap in run.values() for c in sysmap if c not in reserved}
    )
    if categories:
        print()
        print("## Mean per category across runs (stddev in parentheses)")
        header = "| System | " + " | ".join(categories) + " |"
        print(header)
        print("|" + "---|" * (len(categories) + 1))
        for s in systems:
            cells: list[str] = []
            for cat in categories:
                vals = [run.get(s, {}).get(cat, float("nan")) for run in per_run]
                clean = [v for v in vals if v == v]
                if not clean:
                    cells.append("—")
                    continue
                mean = statistics.fmean(clean)
                std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
                cells.append(f"{mean:.3f} ({std:.3f})")
            print(f"| {s} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
