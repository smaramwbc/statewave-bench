# Statewave Bench

[![CI](https://github.com/smaramwbc/statewave-bench/workflows/CI/badge.svg)](https://github.com/smaramwbc/statewave-bench/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Open benchmark for AI memory runtimes.

Statewave Bench measures **how well a memory layer helps an LLM answer questions across long, multi-session conversations** — the actual job memory products do for production agents. Run it against Statewave, Mem0, Zep, naive context-stuffing, and a no-memory baseline to see which architecture wins on your axes (quality, cost, latency, or all three).

> 📋 **Issues & feature requests** for the entire Statewave workspace are tracked centrally on [`smaramwbc/statewave`](https://github.com/smaramwbc/statewave/issues) — including bench-specific ones. Issues are disabled on this repo so all reports funnel to one place.

## Why

Memory products are hard to differentiate on landing pages. Every vendor claims "long-term memory for agents." The actual question — *does this layer let my agent answer "what was the customer's onboarding date" three months later?* — is rarely answered with numbers. This repo does that, on a public dataset, with the full methodology and code so anyone can reproduce.

## What it does

- Loads [LoCoMo](https://github.com/snap-research/LoCoMo) — Snap Research's public benchmark of multi-session conversations + factual recall questions
- Ingests each conversation into every system under test
- Asks each question, scores the answer (token-level F1 for exact answers, LLM-as-judge for open-ended)
- Reports overall + per-category quality, plus token cost and latency, plus charts

## Systems benchmarked

| System | Approach | Notes |
|---|---|---|
| **statewave** | Compiled memories with provenance, ranked retrieval, deterministic context bundles | Self-hosted on Postgres |
| **mem0** | Flat fact store with LLM-extracted memories | Cloud + self-hosted modes |
| **zep** | Temporal knowledge graph (Graphiti) with bi-temporal validity | Cloud-first |
| **naive** | Last-N conversation turns dumped into the prompt | The "every developer's first attempt" baseline |
| **no_memory** | Same answer model, zero history | The floor — what does the LLM get right with nothing? |

The gap between **no_memory** and **naive** measures *"what naive context dumping adds over zero memory."* The gap between **naive** and any real memory system measures *"what the memory layer adds over naive context dumping"* — which is the actually-interesting question for adopters deciding whether to take on the complexity of a memory runtime.

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

# 4. Pilot run — caps at 50 conversations, ~$50 in API costs.
uv run swb run --limit 50

# 5. Render the summary + charts.
uv run swb report

# Open results/results-overall.html in a browser.
```

## ⚠️ Cost note

**Running this benchmark costs real money on your OpenAI / Anthropic / system-vendor accounts.** You're paying for:

- Every system's `ingest` calls (Mem0's fact extraction, Statewave's optional LLM compile, Zep's graph build)
- Every question's `answer` call (the shared LLM the bench fixes for fair comparison)
- Every LLM-as-judge call on open-ended questions (a separate model scores answers)

Approximate costs on Claude 3.5 Sonnet + GPT-4o-judge, January 2026 prices:

| Run scope | LLM calls | Estimated cost |
|---|---:|---:|
| Pilot (50 conversations) | ~6,000 | $40–$80 |
| Full set (~600 conversations) | ~70,000 | $300–$600 |
| Full set + GPT-4o cross-validation | ~140,000 | $700–$1,200 |

Plus per-system fees: Mem0 cloud free tier covers the pilot; Zep cloud free tier covers the pilot. Statewave's cost is just your own infrastructure (Postgres + the Statewave server, both self-hosted).

**Always run the pilot first** — `swb run --limit 50` — so you confirm the harness works on your environment before committing to the full spend.

## Methodology

### Dataset

[LoCoMo](https://github.com/snap-research/LoCoMo) — ~600 multi-session conversations from Snap Research's 2024 paper *"Evaluating Very Long-Term Conversational Memory of LLM Agents."* Each conversation has 5+ sessions over a simulated multi-week timeframe, plus questions categorized as:

- `single_session` — answer lives in one session
- `multi_session` — answer requires combining facts across sessions
- `temporal_reasoning` — answer requires reasoning about *when* things happened
- `open_domain` — answer is open-ended, not exact-match
- `adversarial` — questions designed to surface confabulation

We report scores per-category alongside the overall mean — a system that crushes single-session questions but bombs multi-session ones shouldn't get to hide that under one global number.

### Scoring

- **Exact-answer questions** — token-level F1 (SQuAD-style normalization: lowercase, drop punctuation, drop articles, collapse whitespace). Identical to LoCoMo's reference evaluator.
- **Open-ended questions** — LLM-as-judge. The judge model is deliberately *different* from the answer model (default: GPT-4o judge for Sonnet answers) to reduce same-model-bias. The judge returns CORRECT (1.0) or INCORRECT (0.0) per question.

### Fairness controls

- **Same answer model across systems.** Whichever model the operator chooses (default: Claude 3.5 Sonnet at temp=0), every system uses it for the final answer. A system can't win because it picked a stronger model.
- **Same judge model across systems.** Same logic.
- **Internal LLM costs reported separately.** Systems that issue their own LLM calls during ingest (Mem0's fact extractor, Statewave's optional LLM compiler) report those tokens under `internal_input_tokens` / `internal_output_tokens` so the operator sees the full bill, not just the answer-model cost.
- **Per-conversation isolation.** Every system scopes its memory by conversation id (`bench:locomo:<id>` for Statewave/Mem0, `bench-locomo-<id>` for Zep). No cross-conversation leakage.
- **Deterministic where possible.** Temperature 0, fixed seeds where SDKs expose them. LLM calls aren't perfectly deterministic but two runs should land within sampling noise.

### Resumability

Results stream to `results/run.jsonl` as the bench progresses. If a run dies halfway through (Anthropic 529, Mem0 rate-limit, kernel panic), re-running `swb run` picks up from the last completed `(system, conversation, question)` tuple. No re-doing work.

## Layout

```
statewave-bench/
├── src/statewave_bench/
│   ├── cli.py              # `swb` entry point: config-check / run / report
│   ├── dataset.py          # LoCoMo loader (HuggingFace cache)
│   ├── llm.py              # unified Anthropic + OpenAI client
│   ├── metrics.py          # F1 + LLM-as-judge
│   ├── runner.py           # main eval loop, JSONL streaming, resumability
│   ├── report.py           # markdown summary + Vega-Lite charts
│   └── systems/
│       ├── base.py         # MemorySystem protocol
│       ├── statewave.py    # uses statewave-py SDK
│       ├── mem0.py         # uses mem0ai SDK
│       ├── zep.py          # uses zep-cloud SDK
│       ├── naive.py        # last-N-turn baseline
│       └── no_memory.py    # zero-context floor
├── data/                   # LoCoMo cache (gitignored)
├── results/                # JSONL + rendered charts (gitignored)
├── tests/                  # unit tests on the harness, NOT real LLM calls
└── pyproject.toml          # uv-managed project
```

## Reproducing the published numbers

When this bench reaches a publishable run, the numbers committed to the README will include:

- The exact `uv.lock` used (so re-running gives the same SDK versions)
- The exact answer + judge model identifiers
- The HuggingFace dataset commit hash
- A `results/published/<date>.jsonl` snapshot of every result row

Anyone can re-run with `swb run` against their own keys and reproduce within sampling noise. If you can't, the numbers don't count — open an issue on `smaramwbc/statewave/issues` and we'll dig in.

## Contributing

Adapters for new memory systems are welcome — see `src/statewave_bench/systems/base.py` for the contract. New systems land via PR; we'll re-run the bench against them and update the published numbers.

## License

Apache-2.0. See [LICENSE](LICENSE).

> Part of the [Statewave](https://statewave.ai) ecosystem — open-source memory runtime for AI agents.
