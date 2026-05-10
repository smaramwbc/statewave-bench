"""`swb` — the bench CLI.

Three subcommands cover the operator workflow:

  swb config check         — verify env vars + API reachability before
                              the bench burns tokens
  swb run [--limit N]      — execute the bench, stream results to JSONL
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

from .dataset import load_locomo


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
def config_check(systems: tuple[str, ...]) -> None:
    """Lightweight sanity check — verifies API keys are present and
    importable for the requested systems. Does NOT make billable
    LLM calls; just confirms the environment is wired."""
    requested = set(systems) if systems else _all_system_names()

    problems: list[str] = []
    for name in requested:
        try:
            _ = _instantiate_system(name)
            console.print(f"[green]✓[/] {name}: ready")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]✗[/] {name}: {e}")
            problems.append(name)

    if problems:
        console.print(f"\n[red]{len(problems)} system(s) not ready:[/] {', '.join(problems)}")
        sys.exit(2)
    console.print("\n[green]All requested systems ready.[/]")


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
    help="Where to stream JSONL results. Resumable across runs.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/locomo"),
    help="Where the LoCoMo dataset is cached. Reusable across runs.",
)
def run(
    systems: tuple[str, ...],
    limit: int | None,
    output: Path,
    cache_dir: Path,
) -> None:
    from .runner import run_bench  # imported here so `swb --help` doesn't pay the cost

    requested = list(systems) if systems else _all_system_names()
    instances = []
    for name in requested:
        try:
            instances.append(_instantiate_system(name))
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Skipping {name}:[/] {e}")
    if not instances:
        console.print("[red]No systems instantiated. Aborting.[/]")
        sys.exit(2)

    console.print(
        f"Running {len(instances)} system(s): {', '.join(s.name for s in instances)}"
    )
    if limit:
        console.print(f"[yellow]Pilot mode:[/] capping at {limit} conversations.")

    conversations = load_locomo(cache_dir=cache_dir, limit=limit)
    run_bench(systems=instances, conversations=conversations, output_path=output)
    console.print(f"\n[green]Done.[/] Results in {output}")
    console.print(f"Render with: [bold]swb report --input {output}[/]")


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
    console.print(f"[green]Report written:[/] {output_dir}/results-summary.md")
    console.print(f"  charts: {output_dir}/results-overall.html, results-by-category.html")


# ── System registry ───────────────────────────────────────────────────────


def _all_system_names() -> list[str]:
    return ["statewave", "mem0", "zep", "naive", "no_memory"]


def _instantiate_system(name: str) -> object:
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
    raise ValueError(
        f"Unknown system `{name}`. Known: {', '.join(_all_system_names())}"
    )


# Silence unused-import warnings for `os`, `Path` if any subcommand
# above gets refactored — we keep them imported because they're
# Click's idiomatic types.
_ = os
