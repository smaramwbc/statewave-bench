# Statewave LoCoMo Benchmark Results

Long-term conversational memory, measured on the public [LoCoMo](https://github.com/snap-research/LoCoMo) dataset. Methodology and commands: [README.md](README.md).

---

## Summary

The current published result is **one full pass over the entire LoCoMo dataset** — all 10 conversations, 1,986 questions per system, across all five systems. This is a complete-dataset single-run result (the 1-run tier of the run-count framework below): a strong directional signal, but not yet the variance-checked 5-/10-run aggregate preferred for public comparison. At this tier, **Statewave leads overall and on both reasoning categories** (multi-hop and temporal), while Mem0 leads on open-domain questions.

---

## Benchmark Configuration

| Field | Value |
|---|---|
| Dataset | LoCoMo (`data/locomo10.json`, canonical GitHub source) |
| Scope | Full dataset — 10 conversations, 1,986 questions per system |
| Systems | statewave, mem0, zep, naive, no_memory |
| Benchmark repo commit | `8ac0e74` |
| Statewave server commit | `9d143d9` |
| Answer model | `claude-haiku-4-5` |
| Judge model | `gpt-4o-2024-08-06` |
| Embedding model | n/a — handled internally by each system |
| Scoring mode | `strict` (default) |
| Runs | 1 (full-dataset single pass) |
| Date | 2026-05-16 |

---

## Run modes

A **run** is one full pass over the dataset (ingest → retrieve → answer → score for every system). Because answer generation and judging are stochastic, the run count determines how much weight a number can carry:

| Run mode | Purpose | Confidence | Status |
|---|---|---|---|
| 1 run | Full-dataset pass, strong directional signal | Low–Medium | ✅ Reported below |
| 5 runs (avg) | Stability / variance check | Medium | ⏳ Not yet run |
| 10 runs (avg) | Preferred public benchmark | Higher | ⏳ Not yet run |

The 5- and 10-run aggregates use `swb run --runs N` and `scripts/aggregate_runs.py` (see [README.md](README.md#5-runs-and-10-runs-stability-check--public-benchmark)). They are intentionally left empty here rather than estimated.

---

## Result Overview — 1 run (full dataset, 1,986 questions/system)

Scores are mean per-question scores in `[0, 1]`. **Overall** is the mean across all categories; **excl. adversarial** drops the refusal-only category (reported separately because correctly declining an unanswerable question is a different signal from recalling a fact).

| System | Overall | Excl. adversarial | Single-hop | Multi-hop | Temporal | Open-domain | Avg input tok/q | Median latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **statewave** | **0.525** | **0.400** | 0.076 | **0.548** | **0.323** | 0.460 | 2,344 | 1.97 s |
| mem0 | 0.494 | 0.363 | 0.057 | 0.268 | 0.177 | **0.523** | 205 | 1.69 s |
| naive | 0.324 | 0.152 | 0.044 | 0.156 | 0.198 | 0.182 | 4,920 | 1.54 s |
| zep | 0.239 | 0.024 | 0.034 | 0.019 | 0.031 | 0.021 | 663 | 1.46 s |
| no_memory | 0.227 | 0.003 | 0.015 | 0.000 | 0.010 | 0.000 | 42 | 1.18 s |

Category sample sizes (per system): open_domain 841, multi_hop 321, single_hop 282, temporal 96, adversarial 446.

Single-hop scores are uniformly low across all systems: LoCoMo single-hop references are terse exact strings scored by token-F1, which penalizes any verbosity in the generated answer — this floor affects every system equally and is a property of that category's metric, not of any one system.

---

## Aggregate — 1 / 5 / 10 runs

Overall score excluding adversarial.

| Run mode | statewave | mem0 | naive | zep | no_memory |
|---|---:|---:|---:|---:|---:|
| 1 run (full dataset) | 0.400 | 0.363 | 0.152 | 0.024 | 0.003 |
| 5 runs (avg) | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending |
| 10 runs (avg) | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending |

Produced by `scripts/aggregate_runs.py` once the repeated full-dataset passes are executed.

---

## Per-run results

One full pass recorded so far. (Overall excl-adversarial per system.)

| Run | Scope | statewave | mem0 | naive | zep | no_memory |
|---|---|---:|---:|---:|---:|---:|
| 1 | Full dataset (10 conv, 1,986 q) | 0.400 | 0.363 | 0.152 | 0.024 | 0.003 |

---

## Variance / stability

**Not yet assessable.** Variance requires at least 5 independent passes; one pass has no standard deviation. Treat the gaps above as a strong directional signal on the full dataset, not statistically separated values. Until the 5-/10-run aggregates exist, the exact decimals should be expected to move run-to-run and must be reported with the standard deviation once multi-run data exists.

---

## Interpretation

- **What this shows:** across the entire LoCoMo dataset in a single pass, Statewave's compiled-memory retrieval produces the highest overall answer quality of the five systems, with its clearest advantages on multi-hop reasoning (combining facts across turns/sessions) and temporal reasoning. Mem0 leads on open-domain questions. naive (raw last-N turn dumping) underperforms structured memory despite the largest token footprint. no_memory establishes the floor.
- **What this does not show:** a variance-checked public number. One pass has no stability estimate. This is the full dataset — a strong signal — but the 5-/10-run aggregate is what should anchor a public comparison.
- **How to compare fairly:** only against results produced with the same dataset scope, the same answer and judge models, the same scoring mode, and the same run count.

---

## Limitations

- **Single pass.** This is one full-dataset run. Variance-checked 5-/10-run aggregates are pending.
- **No variance estimate yet.** Requires ≥5 passes.
- **Stochastic scoring.** Answer generation and LLM-judging are non-deterministic; some systems' retrieval is also non-deterministic between passes. The run-count framework exists precisely to quantify this — see README *Current limitations*.
- **Configuration-bound.** Scores reflect one model/judge/scoring configuration; a different configuration will produce different absolute numbers.

---

## Disclaimer

Benchmark results are intended as reproducible evaluation signals, not absolute claims of universal performance. Scores may vary depending on the model, judge, prompting strategy, retrieval configuration, dataset preprocessing, random seeds, and runtime environment. Results should only be compared against other systems when the same dataset, run count, model configuration, judge configuration, and scoring method are used. One-run results are provided for convenience and smoke testing; the 10-run aggregate is the preferred reporting format for public comparison.
