# Statewave Bench

[![CI](https://github.com/smaramwbc/statewave-bench/workflows/CI/badge.svg)](https://github.com/smaramwbc/statewave-bench/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**An open, reproducible benchmark for long-term conversational memory in AI agents.**

Statewave Bench measures how well a memory layer lets an LLM answer questions across long, multi-session conversations — episodic and semantic recall, temporal reasoning, multi-hop reasoning, retrieval quality, and grounded answer generation over conversation histories that span months of simulated time. It evaluates [Statewave](https://statewave.ai) (an open-source memory runtime for AI agents) on the public [LoCoMo](https://github.com/snap-research/LoCoMo) dataset, alongside Mem0, Zep, a naive context-stuffing baseline, and a no-memory floor. Every number is produced by code in this repo and reproducible from the documented configuration.

> 📋 **Issues & feature requests** for the entire Statewave workspace are tracked centrally on [`smaramwbc/statewave`](https://github.com/smaramwbc/statewave/issues). Issues are disabled on this repo so all reports funnel to one place.

📊 **Results:** the benchmark is actively being run and published **run by run** — see [RESULTS.md](RESULTS.md).

---

## What is being benchmarked?

LoCoMo conversations are long and multi-session: ~19 sessions over ~6 months of simulated time, ~600 turns, ~199 categorized recall questions per conversation. Answering them correctly exercises the full job a memory layer does for a production agent:

- **Long-term conversational memory** — recalling something stated many sessions ago, not just in the recent window.
- **Episodic recall** — *what happened*, and *when* it happened, in a specific past session.
- **Semantic recall** — durable facts about the subject (preferences, relationships, attributes) abstracted from individual turns.
- **Temporal reasoning** — resolving "last Saturday", "two days ago", "next month" against absolute dates, and answering "when did X happen?".
- **Multi-hop reasoning** — combining facts from different turns or different sessions to reach an answer no single turn contains.
- **Retrieval quality** — surfacing the *relevant* memories for a question out of an entire conversation history, under a token budget.
- **Grounded answer generation** — producing an answer supported by retrieved memory, and correctly declining when the answer is genuinely absent.
- **Stability across repeated runs** — LLM answer generation and LLM-as-judge scoring are both stochastic; a credible benchmark reports how much the score moves across independent passes, not just one number.

---

## Why 1 / 5 / 10 runs?

A **run** is one full pass over the LoCoMo dataset: ingest → compile/index → retrieve → answer → score, for every system under test. Because both the answer model and the judge are stochastic, a single pass is one sample from a distribution. Repeating the pass and aggregating is what turns a directional signal into a defensible number.

| Runs | Purpose | Recommended use | Confidence | Publish? |
|---|---|---|---|---|
| **1** | Smoke test | Validate setup and the full pipeline; first directional signal | Low | Internal only — label clearly as directional |
| **5** | Stability check | Early public reporting and variance inspection | Medium | Acceptable if the run count and variance are stated |
| **10** | Public benchmark | Main reported result; system-vs-system and version-vs-version comparison | Higher | Preferred public reporting format |

More runs reduce the chance of over-interpreting one lucky or unlucky answer-model / judge sample. A difference between two systems is only meaningful if it exceeds the run-to-run variance at the run count you used. The 10-run aggregate is the preferred format for any public comparison.

---

## Quickstart

```bash
# 1. Install (uv-managed Python project — handles the venv + lockfile).
git clone https://github.com/smaramwbc/statewave-bench.git
cd statewave-bench
uv sync

# 2. Configure API keys.
cp .env.example .env
$EDITOR .env

# 3. Verify each system is reachable (no billable LLM calls).
uv run swb config-check

# 4. Smoke test — 1 conversation across all systems, first signal.
uv run swb run --limit 1

# 5. Render the summary + charts.
uv run swb report

# Open results/results.html in a browser.
```

---

## How to run

The CLI is `swb`. Every command is real and listed by `uv run swb --help`.

| Command | What it does |
|---|---|
| `swb config-check` | Live, cost-trivial probe that each configured system + provider is reachable. |
| `swb run` | One benchmark pass. Streams results to a JSONL file (fresh-by-default). |
| `swb report` | Renders one JSONL into `results-summary.md` + `results.html`. |
| `swb rescore` | Recomputes the `score` column of an existing JSONL with the current metric module (no re-ingest, no re-answer). |

`swb run` options: `-s/--systems` (repeatable; default = all), `--limit N` (cap conversations; omit for the full set), `-o/--output PATH`, `--resume` (skip already-completed tuples instead of starting fresh), `--cache-dir`.

### 1 run (smoke test)

```bash
# Full dataset, one pass, all systems:
uv run swb run -o results/run-1.json
uv run swb report -i results/run-1.json

# Or a fast single-conversation smoke (pipeline validation):
uv run swb run --limit 1 -o results/smoke.json
```

### 5 runs and 10 runs (stability check / public benchmark)

`swb run --runs N` executes N independent full passes, each to its own file (`-o results/run.jsonl --runs 5` → `results/run-1.jsonl … run-5.jsonl`). Aggregate them with `scripts/aggregate_runs.py`:

```bash
# 5 runs (use --runs 10 for the preferred public benchmark):
uv run swb run --runs 5 -o results/run.jsonl

# Per-run + mean + stddev — overall, excl-adversarial, and per-category:
uv run python scripts/aggregate_runs.py results/run-*.jsonl
```

`scripts/aggregate_runs.py` reads the `score` column each pass already wrote (it does not re-score) and reports, for every system: each run's score, the mean across runs, and the population standard deviation (the stability signal) — overall, excluding-adversarial, and per-category. It exits non-zero on a missing or unparseable file so the workflow fails loudly rather than averaging a partial set.

---

## How results are scored

For every question, each system retrieves context from its own memory of the conversation, the **same answer model** generates an answer from that context, and the answer is scored against the LoCoMo reference answer using the metric appropriate to the question category:

| Category | Metric | Why |
|---|---|---|
| `single_hop` | Token-level **F1** (SQuAD normalization) | The reference is an unambiguous short factoid; token overlap is the right measure. |
| `multi_hop`, `temporal`, `open_domain` | **LLM-as-judge** (CORRECT / INCORRECT → 1.0 / 0.0) | The reference is a natural-language explanation; token F1 penalizes correct-but-paraphrased answers. The judge model is deliberately different from the answer model to reduce same-model bias. |
| `adversarial` | **Refusal judge** (REFUSAL / FABRICATION → 1.0 / 0.0) | The answer is not in the conversation; the correct behavior is to decline. F1 against an empty reference would score every correct refusal as zero. |

- **Per-question score** is in `[0, 1]`. Every row in the JSONL records the question, category, reference answer, system prediction, score, the metric used, token counts, and latency.
- **Overall score** = mean of per-question scores. `swb report` also reports a mean that **excludes the adversarial category** (the refusal-only subset; long-term-memory comparisons typically report the non-adversarial mean separately) and a per-category breakdown.
- **Across runs**, `scripts/aggregate_runs.py` averages each run's overall (and excl-adversarial) score and reports the standard deviation. Averages are simple means over independent passes; no run is weighted differently.

Default answer model: `claude-haiku-4-5`. Default judge model: `gpt-4o-2024-08-06`. Both are overridable (`SWB_ANSWER_MODEL`, `SWB_JUDGE_MODEL`) and recorded per row so a report always reflects the models actually used.

---

## Fair comparison & publication safety

A benchmark number is only trustworthy if every system answered the **same** questions and nothing silently dropped out. The harness enforces this:

- **No item disappears from the denominator.** Every attempted `(system, conversation, question)` produces a row. If ingest fails, the answer call raises, it times out, or the failure circuit-breaker aborts a conversation, the runner writes an explicit **`score: 0.0`, `metric: "system_failed"`** row carrying `error_type` (`ingest_failed | retrieval_failed | answer_failed | timeout | system_error`) and a short `error_message`. A crash counts as a wrong answer — it is never removed from the average.
- **Judge failures are incomplete, not zero.** If the answer succeeds but the judge call fails after retries, the row is written with **`score: null`, `metric: "judge_failed"`**. Null scores are excluded from means (a judge outage must not look like a system getting answers wrong) — and their presence **blocks headline reporting** (below). Use `swb rescore` to retry only the judge without re-spending on the answer model.
- **Coverage is reported, every run.** `swb report` prints a per-system table: `expected / completed / scored / failed / judge_failed`, plus `coverage = completed/expected` and `scored_coverage = scored/expected`. A result is publication-safe only at 100% scored coverage with equal question sets.
- **Incomplete or unequal runs are blocked by default.** If any system is missing questions other systems answered, or any `judge_failed`/null row exists, `swb report` and `scripts/aggregate_runs.py` **refuse to render headline rankings** and exit non-zero. Pass `--allow-incomplete` to render anyway — the output is then loudly stamped **NOT PUBLICATION-SAFE** and ranking is suppressed. `aggregate_runs.py` additionally refuses runs that don't cover the same question set.
- **Every run writes metadata.** A sidecar `results/<stem>.metadata.json` records timestamp, git commit, dataset URL, conversation/question counts, selected systems, scoring mode, answer/judge models, bench mode, and every retrieval/scoring-affecting env var. The HTML report surfaces the key fields. A run without metadata is treated as un-audited.

### Comparison modes

Systems expose retrieval limits in different units (Statewave: token budget; Zep: character/edge budget; Mem0: `top_k`). `swb run --mode` (or `SWB_BENCH_MODE`) makes the comparison axis explicit and records it in metadata:

| Mode | Question it answers |
|---|---|
| `vendor_defaults` (default) | "How do they compare as shipped?" Context sizes are wildly unequal — **not** an equal-cost comparison; the report says so. |
| `equal_context_budget` | "At a fixed context cost, which retrieves best?" Every system targets ~2k tokens (operator-set knobs are never overridden). |

Because the per-system knobs are coarse proxies, **every row records the actual delivered context** — `retrieved_context_chars`, `retrieved_context_tokens_estimate` (documented ~4-chars/token approximation, applied identically to every system), `retrieved_items_count` — and the report shows average context size per system. The nominal budget is a label; the measured size is the truth. Zep cannot exceed its `graph.search` ceiling (~1,180 tokens); above that it is under-budget and must be annotated, not silently mismeasured (`scripts/budget_sweep.py` handles this).

### Reproducing a run

```bash
# Equal-budget full pass, all systems, then a publication-safe report:
uv run swb run --mode equal_context_budget -o results/run.jsonl
uv run swb report -i results/run.jsonl          # refuses if not publication-safe
# Multi-run aggregate (the preferred public number):
uv run swb run --runs 10 -o results/run.jsonl
uv run python scripts/aggregate_runs.py results/run-*.jsonl
```

Env vars that change results (all captured in metadata): `SWB_BENCH_MODE`, `SWB_SCORING_MODE`, `SWB_ANSWER_MODEL`, `SWB_JUDGE_MODEL`, `SWB_STATEWAVE_CONTEXT_MAX_TOKENS`, `MEM0_TOP_K`, `SWB_ZEP_SEARCH_LIMIT`, `SWB_ZEP_SEARCH_MAX_CHARS`, `SWB_NAIVE_WINDOW_SIZE`, `STATEWAVE_BENCH_DIGEST`, `STATEWAVE_BENCH_DIGEST_MODE`, `STATEWAVE_BENCH_HYBRID`, `LOCOMO_DATASET_URL`.

> **Publication-safety disclaimer.** Numbers are not publication-safe unless coverage is 100% with equal question sets across systems, no `judge_failed`/null rows remain, and the run's metadata sidecar is attached. `--allow-incomplete` output is for debugging only and is stamped accordingly.

---

## Reproducibility

To interpret or reproduce any number, capture all of:

| Variable | Where it's recorded |
|---|---|
| Benchmark repo commit | `git rev-parse HEAD` in this repo |
| Statewave version / commit | The Statewave server build under test (for the `statewave` system) |
| Dataset | LoCoMo `data/locomo10.json`, fetched from the canonical [snap-research/LoCoMo](https://github.com/snap-research/LoCoMo) GitHub HEAD on first use, cached under `data/locomo/` |
| Answer model | `answer_model` column in every JSONL row (default `claude-haiku-4-5`) |
| Judge model | `SWB_JUDGE_MODEL` / default `gpt-4o-2024-08-06` |
| Embedding model | Not selected by the bench — each system handles its own embeddings internally |
| Scoring mode | `SWB_SCORING_MODE` (default `strict`) |
| Number of runs | Number of independent `swb run` passes aggregated |
| Date of run | Recorded in the rendered report metadata |
| Environment | uv-locked Python deps (`uv.lock`); systems run against their own cloud/self-hosted backends |
| Non-default config | Any `-s`, `--limit`, `SWB_*` overrides used |

Anyone can re-run with their own keys and reproduce within sampling noise at the same run count. If you can't, the numbers don't count — open an issue on [`smaramwbc/statewave`](https://github.com/smaramwbc/statewave/issues).

---

## Interpreting the results

- **Scores are benchmark signals, not absolute truth.** They measure performance on LoCoMo under one configuration, not universal capability.
- **A 1-run result is directional only.** It validates the pipeline and gives a first signal; it is not a basis for strong claims.
- **5-run and 10-run aggregates are for stability.** Trust a system-vs-system gap only when it exceeds the run-to-run standard deviation at that run count.
- **Results depend on configuration.** The answer model, judge model, prompt, retrieval/budget config, embedding model, and dataset preprocessing all move scores. Two systems are only comparable when measured with the **same dataset, same models, same judge, same scoring mode, and same run count**.
- **Cost and latency are part of the result.** A higher score at 25× the token cost is a different product decision than a comparable score at a fraction of it; the report surfaces tokens and latency alongside quality.

---

## Cost note

A full pass makes real, billable LLM calls:

- One answer-model call per question, every system (the shared answer model).
- One judge-model call per LLM-judged / refusal-judged question.
- Each system's own ingest calls (Mem0's fact extraction, Statewave's LLM compile, Zep's graph build) bill against that system's account; the bench records them under `internal_*` token columns so the full bill is visible.

A single-conversation smoke (`--limit 1`) across all systems is roughly \$10–\$15. A full pass is proportionally larger; a 10-run public benchmark is 10× a full pass. Mem0 and Zep cloud free tiers cover smoke / pilot runs.

---

## Current limitations

Documented honestly so results are not over-read:

- **`swb report` renders one JSONL** (single pass). Cross-run aggregation — overall, excluding-adversarial, and per-category mean + standard deviation across N passes — is `scripts/aggregate_runs.py`. The rendered HTML report is per-pass; the cross-run view is the script's markdown output.
- **Single-conversation smoke is not a public number.** `--limit 1` validates the pipeline on one conversation; it is not the full-dataset benchmark and is labelled as smoke wherever it appears.
- **Stochastic dependencies.** Both the answer model and the judge are non-deterministic; some systems' retrieval is also non-deterministic run-to-run. This is exactly why the run-count framework exists — report variance, don't hide it.

---

## Disclaimer

Benchmark results are intended as reproducible evaluation signals, not absolute claims of universal performance. Scores may vary depending on the model, judge, prompting strategy, retrieval configuration, dataset preprocessing, random seeds, and runtime environment. Results should only be compared against other systems when the same dataset, run count, model configuration, judge configuration, and scoring method are used. One-run results are provided for convenience and smoke testing; the 10-run aggregate is the preferred reporting format for public comparison.

---

## Contributing

Adapters for new memory systems are welcome — see `src/statewave_bench/systems/base.py` for the contract. New systems land via PR; the bench is re-run against them and the published numbers updated.

## License

Apache-2.0. See [LICENSE](LICENSE).

> Part of the [Statewave](https://statewave.ai) ecosystem — open-source memory runtime for AI agents.
