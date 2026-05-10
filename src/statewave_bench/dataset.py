"""LoCoMo loader.

LoCoMo (Long-form Conversational Memory) is a public benchmark from
Snap Research — see https://github.com/snap-research/LoCoMo for the
upstream repo and the paper "Evaluating Very Long-Term Conversational
Memory of LLM Agents" (2024).

The dataset is a single JSON file (`data/locomo10.json`) containing 10
multi-session conversations. Each conversation:
  - ~19 sessions over ~6 months simulated time
  - ~18 turns per session
  - ~199 categorized recall questions

We download the file directly from GitHub raw on first use (cached
locally) and parse the upstream schema into typed shapes the rest of
the bench consumes.

Why direct download (not HuggingFace `datasets`): LoCoMo isn't on
HuggingFace as a dataset. The upstream is the GitHub repo. Using the
canonical source means our results stay reproducible if HuggingFace
mirrors come and go.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

# Pinned to a specific commit of the upstream repo would give true
# reproducibility, but the dataset file hasn't changed since the paper's
# release and a SHA-pinned URL gets unwieldy. The README documents how
# to override `LOCOMO_DATASET_URL` for fork-pinned reproducibility.
DEFAULT_LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/LoCoMo/main/data/locomo10.json"
)

# Per the LoCoMo paper, Table 1. Codes 1..5; 0 isn't used in the
# upstream data but we accept it gracefully as "unknown".
_CATEGORY_LABELS: dict[int, str] = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


@dataclass(frozen=True)
class LocomoTurn:
    """One conversational turn within a LoCoMo session."""

    speaker: str
    text: str
    # Upstream `dia_id` of the form `D<session>:<turn>`. Useful for
    # auditing retrieval ("did the system pull the right session?")
    # and for joining against the ground-truth `evidence` lists.
    dia_id: str | None
    # Session-level timestamp from `session_<N>_date_time`, propagated
    # to every turn so adapters that key on per-turn timestamps don't
    # need to look it up separately.
    timestamp: str | None


@dataclass(frozen=True)
class LocomoQA:
    """One factual recall question paired with a ground-truth answer.

    The eval prompts each system with `question`, then scores the
    response against `answer` (string-match for exact-answer questions,
    LLM-as-judge for `open_domain` — the metric is selected by
    `category`).
    """

    question: str
    answer: str
    # Human-readable category label derived from the upstream integer
    # code via the paper's Table 1 mapping.
    category: str
    # Original integer code preserved so per-category aggregation in
    # the report can sort / group consistently with upstream papers.
    category_code: int
    # `dia_id` references that contain the answer when known. Empty
    # for adversarial questions where the answer isn't in the
    # conversation.
    evidence_dia_ids: tuple[str, ...]


@dataclass(frozen=True)
class LocomoConversation:
    """A full multi-session conversation plus its question set."""

    id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[tuple[LocomoTurn, ...], ...]
    qa: tuple[LocomoQA, ...]


def load_locomo(
    *,
    url: str = DEFAULT_LOCOMO_URL,
    cache_dir: Path | None = None,
    limit: int | None = None,
) -> Iterator[LocomoConversation]:
    """Yield LoCoMo conversations one at a time.

    `limit` caps the iteration — set it to a small number for the
    pilot run; leave `None` for the full 10-conversation set.
    """
    cache_dir = cache_dir or Path("data/locomo")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "locomo10.json"

    if not cached.exists():
        _download_to(url, cached)

    with cached.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected LoCoMo to be a JSON array of conversations, got {type(data).__name__}. "
            "If the upstream schema changed, file an issue."
        )

    for yielded, row in enumerate(data):
        if limit is not None and yielded >= limit:
            return
        yield _row_to_conversation(row)


def _download_to(url: str, target: Path) -> None:
    """Download once, atomically (write-tmp → rename) so a crashed
    download doesn't leave a half-written file behind."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    req = Request(url, headers={"User-Agent": "statewave-bench/0.1"})
    with urlopen(req, timeout=60) as resp, tmp.open("wb") as f:
        while chunk := resp.read(64 * 1024):
            f.write(chunk)
    tmp.replace(target)


# ── Schema conversion ─────────────────────────────────────────────────────


def _row_to_conversation(row: dict[str, Any]) -> LocomoConversation:
    conv_payload = row.get("conversation") or {}
    speaker_a = str(conv_payload.get("speaker_a") or "")
    speaker_b = str(conv_payload.get("speaker_b") or "")

    # Upstream stores sessions as sibling keys: `session_1`, `session_2`,
    # ..., paired with `session_1_date_time` etc. Sort numerically so
    # session_2 sorts before session_10.
    session_indices = sorted(
        int(k.split("_", 1)[1])
        for k in conv_payload
        if k.startswith("session_")
        and not k.endswith("_date_time")
        and not k.endswith("_summary")
        and k.split("_", 1)[1].isdigit()
    )
    sessions: list[tuple[LocomoTurn, ...]] = []
    for idx in session_indices:
        raw_turns = conv_payload.get(f"session_{idx}") or []
        if not isinstance(raw_turns, list):
            continue
        timestamp = conv_payload.get(f"session_{idx}_date_time")
        sessions.append(
            tuple(
                LocomoTurn(
                    speaker=str(t.get("speaker") or ""),
                    text=str(t.get("text") or ""),
                    dia_id=t.get("dia_id"),
                    timestamp=timestamp,
                )
                for t in raw_turns
            )
        )

    qa = tuple(_qa(q) for q in (row.get("qa") or []))

    return LocomoConversation(
        id=str(row.get("sample_id") or row.get("id") or ""),
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=tuple(sessions),
        qa=qa,
    )


def _qa(q: dict[str, Any]) -> LocomoQA:
    raw_evidence = q.get("evidence") or []
    evidence_t = tuple(str(e) for e in raw_evidence) if isinstance(raw_evidence, list) else ()

    raw_cat = q.get("category")
    if isinstance(raw_cat, int):
        cat_code = raw_cat
    else:
        try:
            cat_code = int(raw_cat) if raw_cat is not None else 0
        except (TypeError, ValueError):
            cat_code = 0
    cat_label = _CATEGORY_LABELS.get(cat_code, "unknown")

    # Upstream `answer` can be a string OR a number for some questions
    # (e.g. counts). Coerce to string for consistent F1 scoring.
    raw_answer = q.get("answer")
    answer_str = "" if raw_answer is None else str(raw_answer)

    return LocomoQA(
        question=str(q.get("question") or ""),
        answer=answer_str,
        category=cat_label,
        category_code=cat_code,
        evidence_dia_ids=evidence_t,
    )
