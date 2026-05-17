"""Context-budget sweep — the honest cross-system comparison.

Memory systems differ wildly in how much context they dump by
default (Mem0's top_k=5 ≈ 200 tok; naive's 100-turn window ≈ 5,000
tok). Comparing them at their defaults conflates *retrieval quality*
with *context budget*. This driver fixes the budget and sweeps it, so
the comparison answers the honest question: **at a fixed context cost,
which system retrieves the most useful information?**

For each target budget it sets every system's budget knob, runs one
`swb run` pass per budget to its own JSONL, then prints a sweep table
(systems x budgets) showing the score AND the *actual* average input
tokens each system delivered — because the knobs are proxies (top_k,
window, max_tokens) and the real x-axis is measured tokens, not the
nominal target.

Systems that architecturally cannot reach a budget are excluded for
that budget and the exclusion is annotated, not hidden — e.g. Zep's
`graph.search` caps at 50 edges (~1,180 tokens), so it is dropped
from budgets above that ceiling rather than silently mis-measured.

Usage:
  uv run python scripts/budget_sweep.py --limit 1
  uv run python scripts/budget_sweep.py --limit 10 --budgets 512,1024,2048,4096

`--limit 1` is the smoke/preview tier; `--limit 10` is the full
dataset. Re-running with a different --limit is the only change
needed between the preview and the full sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Zep's graph.search hard ceiling: limit <= 50 edges -> ~1,180 tokens
# of fact text. Budgets above this are unreachable for Zep through
# its documented retrieval path, so Zep is excluded (and annotated)
# for any target above it rather than mis-measured at a smaller bundle.
ZEP_TOKEN_CEILING = 1180

# Per-system knob calibration. These are *proxies* for a token budget,
# not exact: get_context returns <= its max_tokens; Mem0's per-memory
# size and naive's per-turn size vary. We pick values that land near
# each target and report the ACTUAL measured tokens — the nominal
# label is just the sweep axis. Empirical sizes observed on LoCoMo:
# Mem0 memory ~20-40 tok, naive turn ~49 tok, Zep edge ~24 tok.
_TOK_PER_MEM0_MEMORY = 22
_TOK_PER_NAIVE_TURN = 49
_TOK_PER_ZEP_EDGE = 24


def _env_for_budget(budget: int) -> tuple[dict[str, str], bool]:
    """Return (env-overrides, zep_included) for one target budget."""
    env = {
        "SWB_STATEWAVE_CONTEXT_MAX_TOKENS": str(budget),
        "MEM0_TOP_K": str(max(1, round(budget / _TOK_PER_MEM0_MEMORY))),
        "SWB_NAIVE_WINDOW_SIZE": str(max(1, round(budget / _TOK_PER_NAIVE_TURN))),
    }
    zep_ok = budget <= ZEP_TOKEN_CEILING
    if zep_ok:
        zep_limit = min(50, max(1, round(budget / _TOK_PER_ZEP_EDGE)))
        env["SWB_ZEP_SEARCH_LIMIT"] = str(zep_limit)
        env["SWB_ZEP_SEARCH_MAX_CHARS"] = str(budget * 4)
    return env, zep_ok


def _systems_for(zep_ok: bool) -> list[str]:
    base = ["statewave", "mem0", "naive", "no_memory"]
    return [*base[:2], "zep", *base[2:]] if zep_ok else base


def _run_one(limit: int, budget: int, out: Path) -> None:
    env_over, zep_ok = _env_for_budget(budget)
    systems = _systems_for(zep_ok)
    cmd = [
        "uv",
        "run",
        "swb",
        "run",
        "--limit",
        str(limit),
        "-o",
        str(out),
    ]
    for s in systems:
        cmd += ["-s", s]
    env = {**os.environ, **env_over}
    note = "" if zep_ok else "  (zep excluded — above its ~1,180 tok graph.search ceiling)"
    print(f"\n=== budget {budget} tok | systems: {', '.join(systems)}{note} ===", flush=True)
    print(f"    overrides: {env_over}", flush=True)
    subprocess.run(cmd, env=env, check=True)


def _score_file(path: Path) -> dict[str, tuple[float, float, int]]:
    """{system: (overall, excl_adv, avg_input_tokens)} for one budget file."""
    by_sys: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_sys[str(r["system"])].append(r)
    out: dict[str, tuple[float, float, int]] = {}
    for s, rows in by_sys.items():
        scored = [r for r in rows if isinstance(r.get("score"), (int, float))]
        if not scored:
            continue
        overall = sum(float(r["score"]) for r in scored) / len(scored)
        na = [r for r in scored if r.get("category") != "adversarial"]
        excl = sum(float(r["score"]) for r in na) / len(na) if na else float("nan")
        toks = sum(int(r.get("answer_input_tokens", 0)) for r in scored) // len(scored)
        out[s] = (overall, excl, toks)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--limit", type=int, default=1, help="Conversations per pass (1=smoke, 10=full)."
    )
    ap.add_argument(
        "--budgets",
        default="512,1024,2048,4096",
        help="Comma-separated target token budgets.",
    )
    ap.add_argument(
        "--out-prefix",
        default="results/sweep",
        help="Output path prefix; files become <prefix>-L<limit>-<budget>.json.",
    )
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    files: dict[int, Path] = {}
    for b in budgets:
        out = Path(f"{args.out_prefix}-L{args.limit}-{b}.json")
        _run_one(args.limit, b, out)
        files[b] = out

    # Sweep table: rows = systems, columns = budgets. Each cell shows
    # excl-adversarial score and the ACTUAL avg input tokens delivered.
    scored = {b: _score_file(f) for b, f in files.items()}
    systems = sorted({s for d in scored.values() for s in d})

    print("\n" + "=" * 72)
    print(f"## Budget sweep — --limit {args.limit} (excl-adversarial score @ actual tok)")
    header = "| System | " + " | ".join(f"{b} tok" for b in budgets) + " |"
    print(header)
    print("|" + "---|" * (len(budgets) + 1))
    for s in systems:
        cells = []
        for b in budgets:
            v = scored[b].get(s)
            cells.append(f"{v[1]:.3f} @ {v[2]}" if v else "—")
        print(f"| {s} | " + " | ".join(cells) + " |")

    print()
    print(f"## Overall (all categories) — --limit {args.limit}")
    print(header)
    print("|" + "---|" * (len(budgets) + 1))
    for s in systems:
        cells = []
        for b in budgets:
            v = scored[b].get(s)
            cells.append(f"{v[0]:.3f}" if v else "—")
        print(f"| {s} | " + " | ".join(cells) + " |")
    print("=" * 72)
    print("'—' = system excluded at that budget (see per-budget log for why).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
