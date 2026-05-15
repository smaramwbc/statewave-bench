"""`swb` — the bench CLI.

Four subcommands cover the operator workflow:

  swb config-check         — verify env vars + API reachability before
                              the bench burns tokens
  swb run [--limit N]      — execute the bench, stream results to JSONL
                              (fresh by default; pass --resume to keep
                              the prior file and skip done tuples)
  swb rescore              — recompute the score + metric columns in an
                              existing JSONL using the current metric
                              module (e.g. after expanding LLM-judge
                              categories) without re-running the bench
  swb report               — read JSONL → summary + charts

Each subcommand is independent; operators can mix-and-match (run a
small pilot, regenerate the report after a methodology tweak, run
the full set on a different machine and merge the JSONL files).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from .cost import CostEstimate
from .cost import estimate as estimate_cost
from .dataset import load_locomo
from .llm import (
    check_anthropic_live,
    check_openai_live,
    resolve_answer_model,
    resolve_judge_model,
)
from .systems.base import MemorySystem

console = Console()


# ── Top-level group ────────────────────────────────────────────────────────


@click.group(help="statewave-bench — open benchmark for AI memory runtimes.")
@click.version_option()
def main() -> None:
    """Load .env once for every subcommand. Operators export keys per
    session OR keep them in .env — both are supported."""
    load_dotenv()


# ── `swb config check` ─────────────────────────────────────────────────────


@main.command("config-check", help="Verify env vars + API reachability before running.")
@click.option(
    "--systems",
    "-s",
    multiple=True,
    help="Which systems to check (default: all). Example: -s statewave -s mem0",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help=(
        "If set, also print a cost estimate for `swb run --limit <N>` "
        "based on the requested systems. Use 1 for a smoke estimate, "
        "10 for the full set."
    ),
)
@click.option(
    "--statewave-llm-compile",
    is_flag=True,
    default=False,
    help=(
        "Flag for the cost estimator: include Statewave's LLM compile "
        "cost (operator's server is configured with the LLM compiler). "
        "Default off (heuristic compiler, no external cost)."
    ),
)
def config_check(
    systems: tuple[str, ...],
    limit: int | None,
    statewave_llm_compile: bool,
) -> None:
    """End-to-end pre-flight: live API probes for every system + the
    answer + judge models, then a cost estimate for the planned run.
    Each provider probe costs ~$0.0001 (1-token generation); the
    system probes are read-only and free.

    Three failure modes the probes catch that bare instantiation
    didn't:
      - Wrong API key (auth_failure)
      - Out of credits / quota (low-balance error)
      - Cloud project misconfigured (region / model not enabled)
    """
    requested = list(systems) if systems else _all_system_names()
    problems: list[str] = []

    # ── 1. Per-provider probes ─────────────────────────────────────────
    # We always run these because every system uses the answer model
    # for its final answer and (for open_domain questions) the judge
    # model for scoring.
    console.print("[bold]Providers:[/]")
    anthropic = check_anthropic_live()
    _print_check(anthropic.provider, anthropic.model, anthropic.ok, anthropic.detail)
    if not anthropic.ok:
        problems.append("anthropic")

    openai = check_openai_live()
    _print_check(openai.provider, openai.model, openai.ok, openai.detail)
    if not openai.ok:
        problems.append("openai")

    # ── 2. Per-system probes ───────────────────────────────────────────
    console.print("\n[bold]Systems:[/]")
    for name in requested:
        try:
            instance = _instantiate_system(name)
        except Exception as e:
            _print_check(name, "—", False, _short_err(e))
            problems.append(name)
            continue
        try:
            health = instance.health_check()
        except Exception as e:
            _print_check(name, "—", False, _short_err(e))
            problems.append(name)
            continue
        _print_check(name, "—", health.ok, health.detail)
        if not health.ok:
            problems.append(name)

    # ── 3. Cost estimate (optional) ─────────────────────────────────────
    if limit is not None:
        n_systems_for_estimate = len(requested)
        # The estimator's Mem0 line only applies if mem0 is in the
        # requested set; skip it when running baselines-only.
        include_mem0 = "mem0" in requested
        est = estimate_cost(
            n_conversations=limit,
            n_systems=n_systems_for_estimate,
            answer_model=resolve_answer_model(),
            judge_model=resolve_judge_model(),
            include_mem0=include_mem0,
            statewave_llm_compile=statewave_llm_compile,
        )
        _print_cost_estimate(est)

    if problems:
        console.print(
            f"\n[red]✗ {len(problems)} probe(s) failed:[/] {', '.join(problems)}\n"
            "Fix the issues above before running `swb run`."
        )
        sys.exit(2)
    console.print("\n[green]✓ All probes passed.[/]")


def _print_check(name: str, model: str, ok: bool, detail: str) -> None:
    icon = "[green]✓[/]" if ok else "[red]✗[/]"
    model_suffix = f"  ({model})" if model and model != "—" else ""
    console.print(f"  {icon} {name}{model_suffix}: {detail}")


def _short_err(err: object) -> str:
    s = str(err)
    if len(s) > 200:
        s = s[:200] + "…"
    return s.replace("\n", " ")


def _print_cost_estimate(est: CostEstimate) -> None:
    """Format the cost estimate as a small terminal table. Operators
    eyeball this against their API balances before kicking off a real
    run."""
    console.print(f"\n[bold]Cost estimate - `swb run --limit {est.n_conversations}`:[/]")
    console.print(
        f"  Scope:      {est.n_conversations} conversation(s) x {est.n_systems} system(s)"
    )
    console.print(f"  Answer model: {est.answer_model}")
    console.print(f"  Judge model:  {est.judge_model}\n")

    rows: list[tuple[str, str]] = [
        (
            "Anthropic (answer model)",
            f"${est.anthropic_low:.2f} - ${est.anthropic_high:.2f}",
        ),
        (
            "OpenAI (judge, ~35% of questions)",
            f"${est.openai_judge_low:.2f}",
        ),
    ]
    if est.openai_mem0_internal > 0:
        rows.append(("OpenAI (Mem0 internal fact extractor)", f"${est.openai_mem0_internal:.2f}"))
    if est.statewave_internal > 0:
        rows.append(("OpenAI (Statewave LLM compile)", f"${est.statewave_internal:.2f}"))

    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        console.print(f"  {label.ljust(width)}  {value}")
    console.print(
        f"\n  [bold]Total:      ${est.total_low:.2f} - ${est.total_high:.2f}[/]\n"
        "  (Mem0 + Zep cloud free tiers cover the smoke run. Statewave is\n"
        "  self-hosted - no external cost.)"
    )


# ── `swb run` ──────────────────────────────────────────────────────────────


@main.command("run", help="Execute the bench. Streams results to a JSONL file.")
@click.option(
    "--systems",
    "-s",
    multiple=True,
    help="Which systems to run (default: all). Example: -s statewave -s no_memory",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Cap conversations (use 50 for a pilot, omit for the full ~600 set).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("results/run.jsonl"),
    help=(
        "Where to stream JSONL results. By default, an existing file at "
        "this path is deleted before this run starts so every run "
        "exercises the full delete -> ingest -> compile -> retrieve -> "
        "answer chain. Pass --resume to keep the existing file and skip "
        "already-done (system, conv, q_idx) tuples."
    ),
)
@click.option(
    "--resume",
    "resume",
    is_flag=True,
    default=False,
    help=(
        "Skip already-completed (system, conversation, question) tuples "
        "found in --output instead of deleting and starting fresh. Useful "
        "for the full ~10-conversation run that takes hours and might hit "
        "transient errors; not what you want for iterative testing of "
        "fixes to the memory pipeline."
    ),
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/locomo"),
    help="Where the LoCoMo dataset is cached. Reusable across runs.",
)
@click.option(
    "--runs",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Number of independent full passes (the variance / stability "
        "framework: 1 = smoke, 5 = stability check, 10 = preferred "
        "public benchmark). With --runs > 1 each pass writes to its own "
        "file: --output results/run.jsonl produces results/run-1.jsonl "
        "... results/run-N.jsonl. Aggregate them with "
        "`python scripts/aggregate_runs.py results/run-*.jsonl`."
    ),
)
def run(
    systems: tuple[str, ...],
    limit: int | None,
    output: Path,
    resume: bool,
    cache_dir: Path,
    runs: int,
) -> None:
    from .runner import run_bench  # imported here so `swb --help` doesn't pay the cost

    if runs < 1:
        console.print("[red]--runs must be >= 1.[/]")
        sys.exit(2)

    requested = list(systems) if systems else _all_system_names()
    instances = []
    for name in requested:
        try:
            instances.append(_instantiate_system(name))
        except Exception as e:
            console.print(f"[red]Skipping {name}:[/] {e}")
    if not instances:
        console.print("[red]No systems instantiated. Aborting.[/]")
        sys.exit(2)

    console.print(f"Running {len(instances)} system(s): {', '.join(s.name for s in instances)}")
    if limit:
        console.print(f"[yellow]Pilot mode:[/] capping at {limit} conversations.")
    if runs > 1:
        console.print(
            f"[bold]Multi-run:[/] {runs} independent passes "
            f"({'stability check' if runs < 10 else 'public-benchmark tier'})."
        )

    # One output file per pass when runs > 1 (results/run.jsonl ->
    # results/run-1.jsonl ... run-N.jsonl). Single-run keeps the exact
    # path the caller passed so existing scripts / docs are unchanged.
    pass_paths = [
        output if runs == 1 else output.with_stem(f"{output.stem}-{i}") for i in range(1, runs + 1)
    ]

    for i, pass_path in enumerate(pass_paths, start=1):
        if runs > 1:
            console.print(f"\n[bold cyan]── Pass {i}/{runs} → {pass_path} ──[/]")

        # Fresh-by-default per pass. The full chain — delete_subject()
        # in every adapter, re-ingest, re-compile, re-retrieve,
        # re-answer, re-score — only fires when the JSONL is empty for
        # a given (system, conv) tuple. Carrying stale rows means the
        # bench short-circuits the ingest/compile path; --resume opts
        # back in to the resumable behavior (per-pass).
        if pass_path.exists() and not resume:
            pass_path.unlink()
            console.print(f"[yellow]Fresh run:[/] deleted previous {pass_path}")
        elif pass_path.exists() and resume:
            console.print(
                "[yellow]Resume mode:[/] keeping existing results; "
                "already-done (system, conv, q_idx) tuples will be skipped."
            )

        # load_locomo returns a one-shot generator — reload per pass so
        # every pass sees the full dataset.
        conversations = load_locomo(cache_dir=cache_dir, limit=limit)
        run_bench(systems=instances, conversations=conversations, output_path=pass_path)

    if runs == 1:
        console.print(f"\n[green]Done.[/] Results in {output}")
        console.print(f"Render with: [bold]swb report --input {output}[/]")
    else:
        glob = output.with_stem(f"{output.stem}-*")
        console.print(
            f"\n[green]Done.[/] {runs} passes written: {pass_paths[0]} … {pass_paths[-1]}"
        )
        console.print(f"Aggregate with: [bold]python scripts/aggregate_runs.py {glob}[/]")


# ── `swb report` ──────────────────────────────────────────────────────────


@main.command("report", help="Generate summary + charts from a JSONL results file.")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("results/run.jsonl"),
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results"),
)
def report(input_path: Path, output_dir: Path) -> None:
    from .report import render_report

    render_report(results_path=input_path, output_dir=output_dir)
    console.print("[green]Report written:[/]")
    console.print(f"  - {output_dir}/results-summary.md  (markdown table, paste into READMEs)")
    console.print(f"  - {output_dir}/results.html        (combined modern report)")


# ── `swb rescore` ─────────────────────────────────────────────────────────


@main.command("rescore", help="Recompute scores in a JSONL using the current metric module.")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("results/run.jsonl"),
    help="JSONL file to re-score. Must contain question/category/ground_truth/prediction.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to write the re-scored JSONL. Default: overwrite input.",
)
def rescore(input_path: Path, output_path: Path | None) -> None:
    """Recompute the `score` + `metric` columns for every row in a
    results JSONL using the current metric module. Useful after a
    methodology change (e.g. expanding LLM-judge categories) where
    re-running the bench would be wasteful — predictions are already
    captured, only the scoring path changed.

    Atomic write: new content goes to a temp file, then renamed over
    the destination so a Ctrl-C mid-write doesn't corrupt the file.
    """
    import json
    import tempfile

    from .llm import LlmClient, resolve_judge_model
    from .metrics import score_answer

    out = output_path or input_path
    llm = LlmClient()
    judge_model = resolve_judge_model()

    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    console.print(f"Re-scoring {len(rows)} row(s) from {input_path}…")
    judge_calls = 0
    metric_changes = 0
    for i, row in enumerate(rows, start=1):
        new_score = score_answer(
            question=str(row["question"]),
            prediction=str(row["prediction"]),
            ground_truth=str(row["ground_truth"]),
            category=str(row["category"]),
            llm=llm,
            judge_model=judge_model,
        )
        if new_score.metric != row.get("metric"):
            metric_changes += 1
        if new_score.metric in ("llm_judge", "refusal_judge"):
            judge_calls += 1
        row["score"] = new_score.value
        row["metric"] = new_score.metric
        if i % 50 == 0 or i == len(rows):
            console.print(f"  scored {i}/{len(rows)}…")

    tmp_dir = out.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=tmp_dir, delete=False, suffix=".tmp", encoding="utf-8"
    ) as tmp_fh:
        tmp_path = Path(tmp_fh.name)
        for row in rows:
            tmp_fh.write(json.dumps(row) + "\n")
    tmp_path.replace(out)

    console.print(
        f"[green]Re-scored:[/] {len(rows)} rows -> {out}\n"
        f"  judge calls made: {judge_calls}\n"
        f"  rows whose metric changed: {metric_changes}"
    )


# ── System registry ───────────────────────────────────────────────────────


def _all_system_names() -> list[str]:
    return ["statewave", "mem0", "zep", "naive", "no_memory"]


def _instantiate_system(name: str) -> MemorySystem:
    """Lazy per-system import so an operator running only `naive`
    + `no_memory` doesn't have to install Mem0 or Zep SDKs."""
    if name == "statewave":
        from .systems.statewave import StatewaveSystem

        return StatewaveSystem()
    if name == "mem0":
        from .systems.mem0 import Mem0System

        return Mem0System()
    if name == "zep":
        from .systems.zep import ZepSystem

        return ZepSystem()
    if name == "naive":
        from .systems.naive import NaiveSystem

        return NaiveSystem()
    if name == "no_memory":
        from .systems.no_memory import NoMemorySystem

        return NoMemorySystem()
    raise ValueError(f"Unknown system `{name}`. Known: {', '.join(_all_system_names())}")


# Silence unused-import warnings for `os`, `Path` if any subcommand
# above gets refactored — we keep them imported because they're
# Click's idiomatic types.
_ = os
