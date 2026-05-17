# Statewave LoCoMo Benchmark Results

Long-term conversational memory, measured on the public [LoCoMo](https://github.com/snap-research/LoCoMo) dataset. Methodology and commands: [README.md](README.md).

---

## Status

**The benchmark is actively being run.** Results will be published here **run by run** as each full-dataset pass completes, building toward the variance-checked multi-run aggregate that is the preferred public number.

No scores are published yet — by design. A single pass is one sample from a stochastic distribution (both the answer model and the LLM-as-judge are non-deterministic, and some systems' retrieval is non-deterministic between passes). Publishing one pass as "the" result would misrepresent a system-vs-system comparison. Numbers will appear here as they are produced and reviewed, pass by pass, with the run count and standard deviation stated alongside them.

Every number published here will satisfy the harness's **publication-safety** bar: 100% scored coverage with the *same* question set across all systems, no `judge_failed`/null rows, the comparison mode (`vendor_defaults` / `equal_context_budget`) stated, each system's actual measured context size shown, and the run's metadata sidecar attached. The report tool refuses to render headline rankings otherwise. See [README.md](README.md#fair-comparison--publication-safety) for the full methodology.

---

## Run modes

A **run** is one full pass over the LoCoMo dataset: ingest → compile/index → retrieve → answer → score, for every system under test. Because the answer model and the judge are both stochastic, the run count determines how much weight a number can carry:

| Run mode | Purpose | Confidence |
|---|---|---|
| 1 run | Full-dataset pass — directional signal | Low–Medium |
| 5 runs (avg) | Stability / variance check | Medium |
| 10 runs (avg) | Preferred public benchmark | Higher |

Multi-run aggregates use `swb run --runs N` and `scripts/aggregate_runs.py` (see [README.md](README.md#5-runs-and-10-runs-stability-check--public-benchmark)) and report, per system, each run's score plus the mean and standard deviation — overall, excluding-adversarial, and per category.

---

## Disclaimer

Benchmark results are intended as reproducible evaluation signals, not absolute claims of universal performance. Scores vary with the model, judge, prompting strategy, retrieval configuration, dataset preprocessing, random seeds, and runtime environment. Results should only be compared across systems produced with the same dataset scope, run count, model configuration, judge configuration, and scoring method. The 10-run aggregate is the preferred reporting format for public comparison.
