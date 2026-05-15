# LoCoMo benchmark — apples-to-apples results

Headline numbers for `statewave` versus `mem0` on the full LoCoMo 10-conversation, 1986-question dataset. Every number on this page is reproducible from the JSONL artifacts in `results/` and the scoring code in `src/statewave_bench/metrics.py` + `scripts/honcho_rescore.py`.

> **TL;DR.** Under the verbatim public-SOTA judge (Honcho's `judge_response` from `plastic-labs/honcho`), Statewave scores **0.469 macro-avg** vs Mem0's **0.295 macro-avg** on adversarial-excluded LoCoMo. That's **a 1.6× lead** on the strictest scoring methodology in common use. Mem0's *published* 91.6% number from `mem0.ai/blog/state-of-ai-agent-memory-2026` does **not reproduce** against their own cloud API through any of the four judge methodologies we tested.

## Headline table

Full 10-conversation, 1986-question LoCoMo run. Statewave config: `STATEWAVE_BENCH_DIGEST=1 STATEWAVE_BENCH_DIGEST_MODE=fat` (the `fat` bundle is digest + 15 atomic facts + 6 episode summaries, ~860 input tokens per question). Answer model: `claude-haiku-4-5` for both systems.

| Scoring methodology | Statewave macro | Mem0 macro | Δ |
|---|---:|---:|---:|
| **Honcho verbatim judge** (matches Honcho's published 89.9%) | **0.469** | **0.295** | **+17.4pp / 1.6×** |
| Our permissive judge (LongMem-style; verbatim port of Mem0's template) | 0.546 | — | — |
| Our strict mixed track (F1 for `single_hop`, LLM-judge for the rest) | 0.408 | 0.271 | +13.7pp |

Per-category under the Honcho judge:

| Category | Statewave | Mem0 | Δ |
|---|---:|---:|---:|
| `multi_hop` | 0.508 | 0.100 | **5.1×** |
| `single_hop` | 0.447 | 0.110 | 4.1× |
| `temporal` | 0.458 | 0.313 | 1.5× |
| `open_domain` | 0.465 | 0.444 | tied |

Statewave's strongest lead is on multi-hop reasoning — the category that benefits most from the LLM compiler's atomic-fact extraction + cross-fact retrieval. The two systems tie on open-domain questions, where both are bounded by what fits in a compressed memory bundle versus the full conversation.

Adversarial questions (446 of the 1986 total) are excluded from the macro-avg — they have no ground-truth answer (correct behavior is refusal), and every public-SOTA harness drops them. Our refusal-judge track records them separately: Statewave 0.924, Mem0 0.865.

## Why the headline uses Honcho's judge

Public LoCoMo numbers come from four systems, each with their own scoring choices:

| System | Published claim | Judge prompt | Token cost claim |
|---|---:|---|---|
| Mem0 (their report) | 91.6% | not published; majority-vote 10× call | not detailed |
| Backboard | 90.1% | GPT-4.1 binary CORRECT/WRONG (5-judge ensemble); LoCoMo "task set A" subset | not detailed |
| Honcho | 89.9% | GPT-4o-mini, `judge_response` in `plastic-labs/honcho/tests/bench/locomo_common.py` (open source) | retrieval bundle only |
| Memori | 81.95% | GPT-4.1-mini 4-dim judge; details in their paper | "fraction of full context" |

Honcho's harness is the only one with judge code that's fully open, version-stable, and applies to a fixed LoCoMo split (no "task set A" subsetting). We use it verbatim as the headline scoring axis so anyone can independently rerun `scripts/honcho_rescore.py` against any results JSONL and reproduce the number.

For full transparency, we also report:

- **Our strict track** — F1 on the SQuAD-normalized factoid category (`single_hop`) plus our own LLM-judge prompt for the synthesis categories. This is our canonical methodology for internal regressions; it's stricter than every public-SOTA harness.
- **Our permissive track** — verbatim port of Mem0's upstream template from `rtuosto/agent-memory-benchmark`, byte-stable fingerprinted. Closest to the prompt Mem0's own report uses (minus their 10-way majority vote).

## What "Mem0 91.6% doesn't reproduce" means

Running their `mem0ai` cloud SDK against the public LoCoMo data through this harness, with their default settings, Mem0's measured scores across our judge methodologies are:

| Judge | Mem0 macro (excl. adversarial) |
|---|---:|
| Our strict | 0.271 |
| Honcho verbatim | **0.295** |
| Our permissive (Mem0 LongMem-style) | ~0.30 (estimate; we did not run the full Mem0 rescore on this track) |

The gap between Mem0's published 0.916 and our measured 0.295 (about 60 percentage points) is too large to be sampling noise. Plausible sources, ranked by what's documented in `dial481/locomo-audit`:

1. **Judge leniency drift.** Mem0's report uses a 10-way majority-vote ensemble of their own permissive judge. `locomo-audit` measured that the standard LongMem-style judge accepts up to 63% of *intentionally wrong* answers, and the honest ceiling on LoCoMo is ~93–94% (6.4% of the answer key is wrong). That alone could explain ~10–15pp.
2. **Token-cost methodology.** Their "fraction of context" framing typically excludes the answer-model prompt; ours measures end-to-end input tokens (retrieval + question).
3. **LoCoMo split.** Backboard uses "task set A," a curated subset. We don't know if Mem0 does the same — they don't publish the question-id list.
4. **Cherry-picking across runs.** `locomo-audit` notes published numbers cluster near the honest ceiling, suggesting some are best-of-N runs.

We measured Mem0 across **four runs** on the full LoCoMo set against their live cloud API; the `multi_hop` category alone swung from 0.027 to 0.270 between runs with no code change on either side. Mem0's cloud retrieval is materially non-deterministic, which complicates any one-run "publish" claim.

## Reproducibility

Every number on this page comes from JSONL files in `results/`. The pipeline is:

```bash
# 1. Run Statewave and Mem0 (separately, against a deployed Statewave server + Mem0 cloud).
uv run swb run -s statewave --output results/run-statewave-full.json     # ~1.5h
uv run swb run -s mem0      --output results/run-mem0-full.json          # ~2h

# 2. Score each against Honcho's verbatim judge.
uv run python scripts/honcho_rescore.py \
    --input results/run-statewave-full.json \
    --output results/run-statewave-full-honcho.json

uv run python scripts/honcho_rescore.py \
    --input results/run-mem0-full.json \
    --output results/run-mem0-full-honcho.json
```

The Honcho judge prompt is pinned byte-stable to its `plastic-labs/honcho/tests/bench/locomo_common.py` upstream — diff against that file to verify our port. Atomic answer rows include `question`, `category`, `ground_truth`, `prediction`, `answer_input_tokens`, `answer_output_tokens`, `elapsed_ms` so reviewers can rescore against any other judge they like.

For the published-SOTA-comparable numbers, the harness pins:

- **Answer model:** `claude-haiku-4-5` (matches Honcho's published harness)
- **Judge model:** `gpt-4o-mini`, temperature 0 (matches Honcho's `judge_response`)
- **Statewave config:** `fat` bundle (digest + 15 facts + 6 summaries, ~860 input tokens/q)
- **Mem0 config:** their cloud SDK defaults (no operator tuning)
- **Dataset:** LoCoMo `data/locomo10.json` at the canonical GitHub HEAD commit, fetched on first use
- **Adversarial:** excluded from macro-avg (per every public-SOTA harness), reported separately under the refusal judge

## Honest caveats

- **LoCoMo has a ~6.4% wrong-answer-key noise floor** (`dial481/locomo-audit`). All scores above are noisy at the sub-7% level. Differences inside that band are meaningless; the 17pp Statewave-vs-Mem0 lead is well outside it.
- **Statewave's compression has a structural ceiling.** Honcho's `--skip-dream` baseline (full LoCoMo conversation dumped into the prompt, no memory system) scores 0.754 under the same judge. The 28pp gap between Statewave (compressed 860 tokens) and Honcho's no-memory baseline is the *cost of memory compression* on this benchmark — it can't be closed without giving up the compressed-memory product story.
- **Dreaming-style architectures didn't close the gap.** We tested two such designs (per-session digest, cross-session topic clustering); neither cleared a +3pp threshold over the fat baseline under the Honcho judge. See `git log` for the reverted branches `phase-1-session-digest` and `phase-2-topic-conclusion` if you want the full data.
- **Mem0 cloud retrieval drifts run-to-run.** We measured the same `multi_hop` category at 0.027 and 0.270 across two runs with identical code. Any single number reported for Mem0 — including their 91.6% — should be treated as one draw from a wide distribution, not a deterministic measurement.

## Where Statewave fits

Bench results put Statewave clearly ahead of Mem0 on apples-to-apples scoring. Statewave is **not** state-of-the-art on LoCoMo's absolute ceiling — Honcho's no-memory baseline beats every compressed-memory system. The product question isn't whether to chase that 0.75 ceiling on a closed benchmark; it's whether a memory system is the right architecture for your application: compile-once compression, durable storage, deterministic retrieval, transparent provenance, open source. If those product properties matter, the LoCoMo number that matters is the head-to-head against other memory systems on identical scoring. That's the number above.
