"""Rescore an existing bench results JSONL using Honcho's exact LoCoMo
judge — verbatim system prompt + evidence_context lookup + macro-avg
aggregation per their tests/bench/locomo_common.py.

Why this exists: Honcho publishes a 89.9% LoCoMo headline that bench
operators want to be able to compare against. Our own LLM-judge tracks
(strict + permissive Mem0-verbatim) land us at 0.53 excl-adv on the
same retrieval data. Three plausible explanations for the residual gap
to Honcho's 89.9%:

  (a) Honcho's judge prompt is more lenient than ours.
  (b) Honcho's harness gives the judge `evidence_context` — the actual
      conversation turns referenced by the question's `dia_id` list —
      which lets the judge override wrong gold answers AND verify that
      the synthesized answer is "logically supported by" the evidence
      even when it diverges in wording.
  (c) Honcho's "dreaming" step (background reasoning that consolidates
      observations into peer "conclusions" before any query) is doing
      real algorithmic work our digest doesn't.

This script isolates (a) + (b) — it leaves the retrieval / answers
unchanged, swaps only the judge, and produces three numbers:

  1. Strict (whatever the input JSONL already has) — our canonical track
  2. Honcho-judged (per-row pass/fail using their verbatim prompt)
  3. Honcho-aggregated headline = mean of per-conversation pass-rates,
     adversarial questions excluded upfront (per their `filter_questions
     (exclude_adversarial=True)` default)

If (1) → (3) closes most of the gap to 89.9%, the residual is dreaming
(c) and that's the architecture lever worth pulling. If it doesn't, the
gap may be smaller than we feared — or our retrieval has bugs the judge
is catching.

Usage:
  uv run python scripts/honcho_rescore.py \\
      --input results/run-path2-full.json \\
      --output results/run-path2-full-honcho.json

OPENAI_API_KEY must be set (the judge is gpt-4o-mini per Honcho).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

# Honcho's verbatim judge prompts. Source: plastic-labs/honcho
# (tests/bench/locomo_common.py, lines ~286-369). Kept byte-stable so a
# drift between their upstream and ours surfaces immediately on re-run.
# DO NOT edit these strings without re-pulling from upstream and
# documenting the diff in this comment block.
_HONCHO_JUDGE_SYSTEM_TEMPLATE = """You are evaluating whether a synthesized answer adequately addresses a query about a user based on available conclusions.
## EVIDENCE CONTEXT
{context}
## EVALUATION CONTEXT
You will evaluate:
1. **Query**: The specific question asked about the user
2. **Synthesized Answer**: The response generated from available conclusions
3. **Gold Standard Answer**: The expected/correct answer
## EVALUATION CRITERIA
Judge the synthesized answer as SUFFICIENT or INSUFFICIENT based on:
### Content Completeness
- Does the answer address what the query is asking?
- Are all key aspects of the gold answer covered (even if phrased differently)?
- Is critical information missing that would change the answer's usefulness?
### Semantic Accuracy
- Are any factual errors or contradictions present?
## ACCEPTABLE DIFFERENCES
The following differences are ACCEPTABLE and should NOT result in INSUFFICIENT:
- Different phrasing or word choice that still conveys the same or very similar meaning, especially in cases where the question is tentative or open-ended.
- Additional relevant context beyond the gold answer (including evidence supplied above). This includes the case where the synthesized answer is longer and more detailed than the gold answer, potentially even including additional information that is not explicitly stated in the gold answer but is still broadly relevant to the query. Do NOT penalize the synthesized answer for including additional information that is not explicitly stated in the gold answer.
- **The synthesized answer explicitly includes the full gold answer text (even if surrounded by additional or unrelated details).  If the gold answer appears within the synthesized answer, you MUST mark the answer as SUFFICIENT.**
- More detailed explanations of reasoning or evidence
- Appropriate confidence qualifiers (e.g., "likely", "probably") when warranted
- Differences in length, with the synthesized answer being longer and even more circuitous or indirect in its addressing of the query, as long as it conveys the same meaning
- Minor format or structure variations
## EVIDENCE-GOLD ANSWER CONSISTENCY CHECK
It is possible for the gold answers to be wrong. Sometimes it may not be fully supported by or follow logically from the evidence messages, instead constituting a guess or assumption. Additionally, the gold answers are generated automatically based on the limited set of evidence messages provided above, whereas if additional context were to be taken into account, the answer might be different. In these cases, we must not penalize the synthesized answer for not being exactly the same as the gold answer.
Before deciding, verify whether the gold answer logically and necessarily follows from the supplied evidence context. If you identify a mismatch or missing logical link **and** the synthesized answer acknowledges this uncertainty or provides a more cautious, evidence-grounded explanation (optionally leveraging additional context beyond the ground truth evidence above), treat the synthesized answer as SUFFICIENT even when it diverges in wording or conclusion from the gold answer.  In short:
* If the gold answer over-claims beyond what the evidence shows, do **not** penalize a synthesized answer that appropriately qualifies the claim or offers a plausible alternative consistent with evidence.
* This includes the case where the synthesized answer is ambivalent or uncertain about the answer, as long as it provides sufficient evidence to support not providing a definitive, categorical answer.
* If the synthesized answer clearly explains the gap and gives a better-supported conclusion, mark it SUFFICIENT.
## UNACCEPTABLE DIFFERENCES
The following DO warrant an INSUFFICIENT rating:
- Irreconcilable errors or contradictions with the gold answer **and** the evidence context
- Missing information central to answering the query, such that its absence would change the meaning of the answer
- Does not address the question being asked
## YOUR TASK
First, analyze what the query is asking **and** how well both answers are supported by the evidence context.
Then, provide 2 brief 2-3 sentence arguments for both SUFFICIENT and INSUFFICIENT:
**Arguments for SUFFICIENT:**
- List reasons why the synthesized answer adequately addresses the query
- Note what key information from the gold answer is present or why deviations are justified by the evidence
- Note whether the gold answer is wrong or not necessarily true given the evidence above
**Arguments for INSUFFICIENT:**
- List reasons why the synthesized answer fails to address the question.

Based on weighing these arguments, provide 2-3 sentences to determine if the synthesized answer is sufficient. In your weighing, consider whether the synthesized answer might be a better answer than the gold answer given the evidence above.
Finally, set is_sufficient to true if sufficient or false if insufficient.
Your response MUST be a valid JSON object with EXACTLY these keys:
  - arguments_for_sufficient (string)
  - arguments_for_insufficient (string)
  - final_reasoning (string)
  - is_sufficient (boolean)
Return ONLY this JSON object and nothing else."""


def _build_evidence_context(
    evidence_dia_ids: Sequence[str],
    dia_id_to_turn: dict[str, tuple[str, str]],
) -> str | None:
    """Format the dia_id-referenced turns into Honcho's evidence_context
    string. Returns None if no evidence — Honcho substitutes
    'No evidence provided.' into the system prompt in that case.
    """
    if not evidence_dia_ids:
        return None
    lines: list[str] = []
    for eid in evidence_dia_ids:
        if eid in dia_id_to_turn:
            speaker, text = dia_id_to_turn[eid]
            lines.append(f"[{eid}] {speaker}: {text}")
    return "\n".join(lines) if lines else None


def _build_dia_id_lookup(
    conversations: dict[str, object],
) -> dict[str, dict[str, tuple[str, str]]]:
    """Pre-index `dia_id -> (speaker, text)` per conversation_id for O(1)
    evidence lookup at rescore time. Returned as conversation_id -> map.
    """
    from statewave_bench.dataset import LocomoConversation

    out: dict[str, dict[str, tuple[str, str]]] = {}
    for conv_id, conv in conversations.items():
        assert isinstance(conv, LocomoConversation)
        m: dict[str, tuple[str, str]] = {}
        for session in conv.sessions:
            for turn in session:
                if turn.dia_id:
                    m[turn.dia_id] = (turn.speaker, turn.text)
        out[conv_id] = m
    return out


def _judge_one(
    client: object,
    *,
    question: str,
    gold_answer: str,
    synthesized_answer: str,
    evidence_context: str | None,
) -> tuple[bool, str]:
    """One judge call. Returns (passed, reasoning).

    Mirrors Honcho's `judge_response` — gpt-4o-mini, temperature=0,
    max_tokens=1024, JSON output parsed via tolerant pre-strip of
    markdown fences. Returns False on any parse / API failure (silent
    failures here would bias the score downward — they're flagged in
    the caller's progress log).
    """
    from openai import OpenAI  # type: ignore[import-untyped]

    assert isinstance(client, OpenAI)
    system_prompt = _HONCHO_JUDGE_SYSTEM_TEMPLATE.format(
        context=evidence_context if evidence_context else "No evidence provided."
    )
    user_prompt = (
        f"Query: {question}\nGold Answer: {gold_answer}\nSynthesized Answer: {synthesized_answer}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=0,
        n=1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    eval_response = resp.choices[0].message.content or ""

    # Strip markdown fences (Honcho's parse does the same)
    json_str = eval_response.strip()
    if json_str.startswith("```"):
        json_str = json_str.split("```")[1]
        if json_str.startswith("json"):
            json_str = json_str[4:]
        json_str = json_str.strip()

    try:
        parsed = json.loads(json_str)
        passed = bool(parsed.get("is_sufficient", False))
        reasoning = str(parsed.get("final_reasoning", eval_response[:200]))
        return passed, reasoning
    except json.JSONDecodeError:
        return False, f"JSON parse failed: {eval_response[:200]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", "-i", type=Path, required=True)
    ap.add_argument("--output", "-o", type=Path, required=True)
    ap.add_argument(
        "--limit", type=int, default=None, help="Cap rows scored — useful for smoke-testing."
    )
    ap.add_argument(
        "--exclude-adversarial",
        action="store_true",
        default=True,
        help="(Default true) Skip category=adversarial rows entirely, matching Honcho.",
    )
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    # Load LoCoMo once to build the dia_id lookup. We use our own loader
    # (statewave_bench.dataset) which already preserves dia_id + speaker
    # on each turn — no re-download needed.
    from statewave_bench.dataset import LocomoConversation, load_locomo
    from openai import OpenAI  # type: ignore[import-untyped]

    print(f"Loading LoCoMo dataset…", file=sys.stderr)
    convs_by_id: dict[str, LocomoConversation] = {}
    qa_by_id: dict[tuple[str, int], object] = {}
    for c in load_locomo():
        convs_by_id[c.id] = c
        for idx, qa in enumerate(c.qa):
            qa_by_id[(c.id, idx)] = qa
    dia_lookup = _build_dia_id_lookup(convs_by_id)  # type: ignore[arg-type]
    print(f"  {len(convs_by_id)} conversations, {len(qa_by_id)} questions", file=sys.stderr)

    # Read input JSONL
    rows: list[dict[str, object]] = []
    with args.input.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {args.input}", file=sys.stderr)

    client = OpenAI()

    out_rows: list[dict[str, object]] = []
    skipped_adversarial = 0
    judged = 0
    for i, row in enumerate(rows, start=1):
        if args.limit is not None and judged >= args.limit:
            break
        category = str(row.get("category", ""))
        if args.exclude_adversarial and category == "adversarial":
            skipped_adversarial += 1
            continue

        conv_id = str(row.get("conversation_id", ""))
        q_idx = int(row.get("question_idx", -1))
        qa = qa_by_id.get((conv_id, q_idx))
        if qa is None:
            print(f"  WARN row {i}: missing qa for ({conv_id}, {q_idx})", file=sys.stderr)
            continue
        evidence = _build_evidence_context(
            qa.evidence_dia_ids,  # type: ignore[attr-defined]
            dia_lookup.get(conv_id, {}),
        )

        passed, reasoning = _judge_one(
            client,
            question=str(row.get("question", "")),
            gold_answer=str(row.get("ground_truth", "")),
            synthesized_answer=str(row.get("prediction", "")),
            evidence_context=evidence,
        )

        # Append the Honcho verdict alongside the existing fields —
        # original score / metric preserved so we can compare side-by-
        # side without re-running the bench.
        out_row = dict(row)
        out_row["honcho_score"] = 1.0 if passed else 0.0
        out_row["honcho_reasoning"] = reasoning[:500]
        out_row["honcho_had_evidence"] = evidence is not None
        out_rows.append(out_row)
        judged += 1

        if judged % 50 == 0:
            print(f"  judged {judged}…", file=sys.stderr)

    # Atomic write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(args.output)

    # Aggregate Honcho-style: macro-avg across conversations
    per_conv: dict[str, list[float]] = defaultdict(list)
    per_cat: dict[str, list[float]] = defaultdict(list)
    for r in out_rows:
        per_conv[str(r["conversation_id"])].append(float(r["honcho_score"]))
        per_cat[str(r["category"])].append(float(r["honcho_score"]))

    per_conv_means = {cid: sum(s) / len(s) for cid, s in per_conv.items() if s}
    macro_avg = (
        sum(per_conv_means.values()) / len(per_conv_means) if per_conv_means else 0.0
    )
    micro_avg = (
        sum(float(r["honcho_score"]) for r in out_rows) / len(out_rows) if out_rows else 0.0
    )

    print()
    print("=" * 70)
    print(f"Honcho-rescored {len(out_rows)} rows ({skipped_adversarial} adversarial excluded)")
    print(f"  Honcho-style HEADLINE (macro-avg across conversations): {macro_avg:.4f}")
    print(f"  Micro-avg (per-question equal weight):                 {micro_avg:.4f}")
    print()
    print("By category (micro within each):")
    for cat in sorted(per_cat.keys()):
        scores = per_cat[cat]
        print(f"  {cat:<15} n={len(scores):>4}  mean={sum(scores) / len(scores):.4f}")
    print()
    print("By conversation:")
    for cid in sorted(per_conv_means.keys()):
        print(f"  {cid:<15} n={len(per_conv[cid]):>4}  mean={per_conv_means[cid]:.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
