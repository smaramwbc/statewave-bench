# Statewave LoCoMo Benchmark Results

Long-term conversational memory, measured on the public [LoCoMo](https://github.com/snap-research/LoCoMo) dataset. Methodology and commands: [README.md](README.md).

---

## Summary

The current published result is a **smoke-tier pass**: one full pipeline run over a single LoCoMo conversation (199 questions per system), across all five systems. It validates the end-to-end pipeline and gives a first directional signal. It is **not** a public benchmark number — that requires the full dataset at the 5- or 10-run aggregate (see *Run modes* below). At this tier, Statewave leads overall, with its largest margin on multi-hop reasoning.

---

## Benchmark Configuration

| Field | Value |
|---|---|
| Dataset | LoCoMo (`data/locomo10.json`, canonical GitHub source) |
| Scope (this result) | 1 conversation (`conv-26`), 199 questions per system |
| Systems | statewave, mem0, zep, naive, no_memory |
| Benchmark repo commit | `9a4c658` |
| Statewave server commit | `9d143d9` |
| Answer model | `claude-haiku-4-5` |
| Judge model | `gpt-4o-2024-08-06` |
| Embedding model | n/a — handled internally by each system |
| Scoring mode | `strict` (default) |
| Runs | 1 (smoke tier) |
| Date | 2026-05-16 |

---

## Run modes

A **run** is one full pass over the dataset (ingest → retrieve → answer → score for every system). Because answer generation and judging are stochastic, the run count determines how much weight a number can carry:

| Run mode | Purpose | Confidence | Status |
|---|---|---|---|
| 1 run | Pipeline validation, first signal | Low | ✅ Reported below (single-conversation smoke) |
| 5 runs (avg) | Stability / variance check | Medium | ⏳ Not yet run |
| 10 runs (avg) | Preferred public benchmark | Higher | ⏳ Not yet run |

The 5- and 10-run aggregates require the full dataset and the documented repeated-run procedure in [README.md](README.md#5-runs-and-10-runs-stability-check--public-benchmark). They are intentionally left empty here rather than estimated.

---

## Result Overview — 1 run (smoke, 1 conversation)

Scores are mean per-question scores in `[0, 1]`. **Overall** is the mean across all categories; **excl. adversarial** drops the refusal-only category (reported separately because correctly declining an unanswerable question is a different signal from recalling a fact).

| System | Overall | Excl. adversarial | Single-hop | Multi-hop | Temporal | Open-domain | Avg input tok/q | Median latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **statewave** | **0.520** | **0.385** | 0.049 | **0.622** | 0.385 | 0.414 | 2,348 | 1.70 s |
| mem0 | 0.453 | 0.317 | 0.038 | 0.108 | 0.231 | **0.571** | 224 | 1.64 s |
| naive | 0.382 | 0.217 | 0.032 | 0.189 | 0.385 | 0.286 | 5,441 | 1.66 s |
| zep | 0.259 | 0.029 | 0.015 | 0.027 | 0.077 | 0.029 | 651 | 1.39 s |
| no_memory | 0.237 | 0.001 | 0.007 | 0.000 | 0.000 | 0.000 | 40 | 1.11 s |

Single-hop scores are uniformly low across all systems: LoCoMo single-hop references are terse exact strings scored by token-F1, which penalizes any verbosity in the generated answer — this floor affects every system equally and is a property of that category's metric, not of any one system.

---

## Aggregate — 1 / 5 / 10 runs

| Run mode | statewave | mem0 | naive | zep | no_memory |
|---|---:|---:|---:|---:|---:|
| 1 run (smoke, excl. adv) | 0.385 | 0.317 | 0.217 | 0.029 | 0.001 |
| 5 runs (avg) | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending |
| 10 runs (avg) | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending |

Produced by `scripts/aggregate_runs.py` once the repeated full-dataset passes are executed.

---

## Per-run results

One pass recorded so far.

| Run | Scope | statewave | mem0 | naive | zep | no_memory |
|---|---|---:|---:|---:|---:|---:|
| 1 | conv-26 (smoke) | 0.385 | 0.317 | 0.217 | 0.029 | 0.001 |

(Overall excl-adversarial per system.)

---

## Variance / stability

**Not yet assessable.** Variance requires at least 5 independent passes; one pass has no standard deviation. Until the 5-/10-run aggregates exist, treat the gaps above as directional, not statistically separated. The ordering (statewave > mem0 > naive > zep > no_memory) has been consistent across internal smoke passes, but the exact decimals will move run-to-run and must be reported with the standard deviation once multi-run data exists.

---

## Interpretation

- **What this shows:** on a single LoCoMo conversation, Statewave's compiled-memory retrieval produces the highest overall answer quality of the five systems, with its decisive advantage on multi-hop questions (combining facts across turns/sessions). Mem0 leads on open-domain questions. naive (raw last-N turn dumping) underperforms structured memory despite the largest token footprint. no_memory establishes the floor.
- **What this does not show:** a public-grade comparison. One conversation is a small sample; one pass has no variance estimate. These numbers validate the pipeline and indicate direction — nothing stronger.
- **How to compare fairly:** only against results produced with the same dataset scope, the same answer and judge models, the same scoring mode, and the same run count.

---

## Limitations

- **Single conversation, single pass.** This is the smoke tier. Full-dataset 5-/10-run aggregates are pending.
- **No variance estimate yet.** Requires ≥5 passes.
- **Stochastic scoring.** Answer generation and LLM-judging are non-deterministic; some systems' retrieval is also non-deterministic between passes. The run-count framework exists precisely to quantify this — see README *Current limitations*.
- **Configuration-bound.** Scores reflect one model/judge/scoring configuration; a different configuration will produce different absolute numbers.

---

## Disclaimer

Benchmark results are intended as reproducible evaluation signals, not absolute claims of universal performance. Scores may vary depending on the model, judge, prompting strategy, retrieval configuration, dataset preprocessing, random seeds, and runtime environment. Results should only be compared against other systems when the same dataset, run count, model configuration, judge configuration, and scoring method are used. One-run results are provided for convenience and smoke testing; the 10-run aggregate is the preferred reporting format for public comparison.
