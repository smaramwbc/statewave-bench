# Statewave LoCoMo Benchmark Results

Long-term conversational memory, measured on the public [LoCoMo](https://github.com/snap-research/LoCoMo) dataset. Methodology and commands: [README.md](README.md).

---

## Summary

The current published result is **one full pass over the entire LoCoMo dataset** — all 10 conversations, 1,986 questions per system, across all five systems, on the latest Statewave server. This is a complete-dataset single-run result (the 1-run tier of the run-count framework below): a strong directional signal, but not yet the variance-checked 5-/10-run aggregate preferred for public comparison.

At this tier, **Statewave leads every system in every category** — overall, single-hop, multi-hop, temporal, and open-domain. The deterministic baselines (naive, zep, no_memory) reproduced their prior-pass numbers to within ≤0.02 and the per-system token footprints are unchanged, which validates that the harness, judge, and methodology are stable and this pass is comparable to earlier measurement. Statewave also reproduced its own prior numbers (overall 0.522 vs 0.525, excl-adv 0.399 vs 0.400). Mem0's open-domain score did **not** reproduce its earlier pass (0.348 here vs 0.523 previously); Mem0 ran cleanly with no errors, so this reflects Mem0's known high run-to-run variance — which is exactly why a variance-checked aggregate, not any single pass, is the standard for a public claim. Read the gap to Mem0 as directional, pending that aggregate.

---

## Benchmark Configuration

| Field | Value |
|---|---|
| Dataset | LoCoMo (`data/locomo10.json`, canonical GitHub source) |
| Scope | Full dataset — 10 conversations, 1,986 questions per system |
| Systems | statewave, mem0, zep, naive, no_memory |
| Benchmark repo commit | `2f41f3c` |
| Statewave server commit | `f1345fa` |
| Answer model | `claude-haiku-4-5` |
| Judge model | `gpt-4o-2024-08-06` |
| Embedding model | n/a — handled internally by each system |
| Scoring mode | `strict` (default) |
| Runs | 1 (full-dataset single pass) |
| Date | 2026-05-17 |

The Statewave server is the current `main` (`f1345fa`), which includes the issue-#116 task-relevance gate. That gate only prepends a caveat when *no* stored memory is relevant to the task; LoCoMo questions are always on-topic, so it effectively never fires here — Statewave's scores are unchanged from the prior server within single-pass noise, confirming the change is benchmark-neutral on this dataset.

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
| **statewave** | **0.522** | **0.399** | **0.078** | **0.586** | **0.292** | **0.448** | 2,343 | 1.79 s |
| mem0 | 0.388 | 0.224 | 0.051 | 0.087 | 0.104 | 0.348 | 209 | 1.68 s |
| naive | 0.325 | 0.150 | 0.044 | 0.156 | 0.177 | 0.181 | 4,920 | 1.51 s |
| zep | 0.241 | 0.024 | 0.033 | 0.016 | 0.042 | 0.021 | 657 | 1.45 s |
| no_memory | 0.227 | 0.003 | 0.015 | 0.000 | 0.010 | 0.000 | 42 | 1.20 s |

Category sample sizes (per system): open_domain 841, multi_hop 321, single_hop 282, temporal 96, adversarial 446.

Single-hop scores are uniformly low across all systems: LoCoMo single-hop references are terse exact strings scored by token-F1, which penalizes any verbosity in the generated answer — this floor affects every system equally and is a property of that category's metric, not of any one system.

---

## Aggregate — 1 / 5 / 10 runs

Overall score excluding adversarial.

| Run mode | statewave | mem0 | naive | zep | no_memory |
|---|---:|---:|---:|---:|---:|
| 1 run (full dataset) | 0.399 | 0.224 | 0.150 | 0.024 | 0.003 |
| 5 runs (avg) | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending |
| 10 runs (avg) | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending |

Produced by `scripts/aggregate_runs.py` once the repeated full-dataset passes are executed.

---

## Per-run results

One full pass recorded so far. (Overall excl-adversarial per system.)

| Run | Scope | statewave | mem0 | naive | zep | no_memory |
|---|---|---:|---:|---:|---:|---:|
| 1 | Full dataset (10 conv, 1,986 q), server `f1345fa` | 0.399 | 0.224 | 0.150 | 0.024 | 0.003 |

---

## Variance / stability

**Not yet formally assessable** — variance requires ≥5 independent passes; one pass has no standard deviation. One concrete cross-pass observation is already in hand, and it cuts both ways:

- **Reproducible:** the deterministic baselines (naive, zep, no_memory) and Statewave all landed within ≤0.02 of their earlier full-dataset pass, with identical token footprints. The methodology is stable.
- **Volatile:** Mem0's open-domain score moved from 0.523 (earlier pass) to 0.348 here — a ~0.18 swing on a clean run with no errors. Mem0's retrieval/extraction is non-deterministic and demonstrably high-variance at this scale.

Treat the per-system gaps as a strong directional signal on the full dataset, not statistically separated values — and treat the Statewave–Mem0 margin specifically as directional until the 5-/10-run aggregate exists. Exact decimals will move run-to-run and must be reported with the standard deviation once multi-run data exists.

---

## Interpretation

- **What this shows:** across the entire LoCoMo dataset in a single pass on the current server, Statewave's compiled-memory retrieval produces the highest answer quality of the five systems in *every* category — overall, single-hop, multi-hop, temporal, and open-domain — with its clearest absolute advantages on multi-hop (combining facts across turns/sessions) and temporal reasoning. naive (raw last-N turn dumping) underperforms structured memory despite the largest token footprint. no_memory establishes the floor.
- **On the Statewave vs Mem0 open-domain result:** an earlier pass had Mem0 ahead on open-domain (0.523 vs 0.460). That Mem0 figure did not reproduce here (0.348), while Statewave's reproduced (0.448 vs 0.460, within noise) and the baselines reproduced. The most defensible reading is not "Statewave decisively overtook Mem0 on open-domain" but "Mem0's open-domain score is high-variance and a single pass cannot settle this category" — the variance-checked aggregate is required to state the open-domain comparison with confidence.
- **What this does not show:** a variance-checked public number. One pass has no stability estimate. This is the full dataset on the latest server — a strong signal — but the 5-/10-run aggregate is what should anchor a public comparison.
- **How to compare fairly:** only against results produced with the same dataset scope, the same answer and judge models, the same scoring mode, the same server build, and the same run count.

---

## Limitations

- **Single pass.** This is one full-dataset run. Variance-checked 5-/10-run aggregates are pending.
- **No variance estimate yet.** Requires ≥5 passes. One cross-pass observation (above) already shows Mem0 is high-variance; the Statewave–Mem0 margin should be read as directional until aggregated.
- **Stochastic scoring.** Answer generation and LLM-judging are non-deterministic; some systems' retrieval (notably Mem0's) is also non-deterministic between passes. The run-count framework exists precisely to quantify this — see README *Current limitations*.
- **Configuration-bound.** Scores reflect one model/judge/scoring/server configuration; a different configuration will produce different absolute numbers.

---

## Disclaimer

Benchmark results are intended as reproducible evaluation signals, not absolute claims of universal performance. Scores may vary depending on the model, judge, prompting strategy, retrieval configuration, dataset preprocessing, random seeds, and runtime environment. Results should only be compared against other systems when the same dataset, run count, model configuration, judge configuration, and scoring method are used. One-run results are provided for convenience and smoke testing; the 10-run aggregate is the preferred reporting format for public comparison.
