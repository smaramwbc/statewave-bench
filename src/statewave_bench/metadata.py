"""Per-run benchmark metadata snapshot.

A results JSONL on its own is not reproducible: the same file can be
produced by very different configs (digest mode on, a tuned token
budget, permissive scoring, a different judge model). Every run writes
a sidecar `<results-stem>.metadata.json` capturing every input that
changes the numbers, so a published result can be audited and
reproduced — and so `swb report` can refuse to headline a run whose
metadata says it wasn't a clean, equal comparison.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METADATA_SCHEMA_VERSION = 1

# Every env var that changes retrieval, context assembly, scoring, or
# judging. Captured verbatim (None when unset) so a reviewer sees the
# exact knob state behind the numbers.
BENCH_AFFECTING_ENV_VARS = (
    "SWB_BENCH_MODE",
    "SWB_SCORING_MODE",
    "SWB_STATEWAVE_CONTEXT_MAX_TOKENS",
    "MEM0_TOP_K",
    "SWB_ZEP_SEARCH_LIMIT",
    "SWB_ZEP_SEARCH_MAX_CHARS",
    "SWB_NAIVE_WINDOW_SIZE",
    "STATEWAVE_BENCH_DIGEST",
    "STATEWAVE_BENCH_DIGEST_MODE",
    "STATEWAVE_BENCH_HYBRID",
    "LOCOMO_DATASET_URL",
)


def metadata_path(results_path: Path) -> Path:
    """`results/run.jsonl` → `results/run.metadata.json`. One sidecar
    per results file (per pass, when --runs > 1)."""
    return results_path.with_name(results_path.stem + ".metadata.json")


def _git_commit() -> str | None:
    """Best-effort short commit of the bench repo. Never raises — a
    tarball install with no .git just records null."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None


def build_metadata(
    *,
    results_path: Path,
    systems: list[str],
    n_conversations: int,
    n_questions: int,
    answer_model: str,
    judge_model: str,
    bench_mode: str,
    dataset_url: str,
    dataset_cache_path: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "dataset_url": dataset_url,
        "dataset_cache_path": dataset_cache_path,
        "n_conversations": n_conversations,
        "n_questions": n_questions,
        "systems": list(systems),
        "scoring_mode": os.environ.get("SWB_SCORING_MODE", "strict").lower(),
        "bench_mode": bench_mode,
        "answer_model": answer_model,
        "judge_model": judge_model,
        "env": {k: os.environ.get(k) for k in BENCH_AFFECTING_ENV_VARS},
    }


def write_metadata(results_path: Path, metadata: dict[str, Any]) -> Path:
    path = metadata_path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_metadata(results_path: Path) -> dict[str, Any] | None:
    """Read the sidecar if present. report uses it to surface config
    and to warn when a run predates metadata capture."""
    path = metadata_path(results_path)
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return obj if isinstance(obj, dict) else None
