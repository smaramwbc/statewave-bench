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

# 4. Smoke run — one conversation across all 5 systems, ~$10–$15.
uv run swb run --limit 1

# Or full set when you're ready: 10 conversations, ~$100–$150 total.
# uv run swb run

# 5. Render the summary + charts.
uv run swb report

# Open results/results-overall.html in a browser.
```

## ⚠️ Cost note

**Running this benchmark costs real money on your OpenAI / Anthropic / system-vendor accounts.** You're paying for:

- Every system's `ingest` calls (Mem0's fact extraction, Statewave's optional LLM compile, Zep's graph build)
- Every question's `answer` call (the shared LLM the bench fixes for fair comparison)
- One judge call per non-`single_hop` question (a separate model scores reasoning answers as CORRECT/INCORRECT; an adversarial-specific judge marks refusals as REFUSAL/FABRICATION)

LoCoMo's full dataset is **10 conversations × ~199 questions = ~1,986 questions per system**. The bench runs every question through every system, so 5 systems × 1,986 questions = ~9,930 question-runs total. Per question we make:

- 1 retrieval call (free for the bench's accounting — Mem0 / Zep handle internally; Statewave runs locally)
- 1 answer call (~3K input + ~256 output tokens on Sonnet ≈ $0.013)
- 1 judge call for every question except `single_hop` (~83% of LoCoMo: `multi_hop`, `temporal`, `open_domain`, `adversarial` are all LLM-scored; ~500 input + 8 output on GPT-4o ≈ $0.001)

Approximate costs on Claude Sonnet 4.6 + GPT-4o judge, January 2026 prices:

| Run scope | Question-runs | Judge calls | Estimated cost |
|---|---:|---:|---:|
| Smoke (1 conversation, all 5 systems) | ~995 | ~825 | $13–$18 |
| Pilot (3 conversations) | ~2,985 | ~2,475 | $40–$55 |
| Full set (10 conversations) | ~9,930 | ~8,240 | $130–$180 |

Plus internal LLM costs the bench doesn't directly observe:

- **Mem0** runs its own fact-extraction LLM call on every `add()` against the operator's OpenAI / Anthropic key — about $4–$8 across the full set
- **Statewave with the LLM compiler** runs one compile per conversation (~$0.05 each, ~$0.50 total). Statewave with the heuristic compiler is $0
- **Zep** bills graph-build LLM calls against the Zep plan, not against an external provider key — see Zep's pricing page

Mem0 cloud + Zep cloud free tiers cover the smoke + pilot runs. Statewave's cost is just your own infrastructure (Postgres + the Statewave server, both self-hosted).

**Always run the smoke first** — `swb run --limit 1` — so you confirm the harness works on your environment before committing to the full spend.

## Methodology

### Dataset

[LoCoMo](https://github.com/snap-research/LoCoMo) — 10 multi-session conversations from Snap Research's 2024 paper *"Evaluating Very Long-Term Conversational Memory of LLM Agents."* Each conversation has ~19 sessions over ~6 months simulated time, totaling ~600 turns and ~199 categorized recall questions. The bench fetches the canonical `data/locomo10.json` directly from the upstream GitHub repo (cached locally on first use).

Categories per the paper's Table 1:

- `single_hop` (code 1) — answer lives in one utterance; short factoid
- `multi_hop` (code 2) — answer requires combining facts from multiple utterances or sessions
- `temporal` (code 3) — answer requires reasoning about *when* things happened
- `open_domain` (code 4) — answer is open-ended
- `adversarial` (code 5) — answer isn't in the conversation at all; the model should refuse

We report scores per-category alongside the overall mean — a system that crushes single-session questions but bombs multi-session ones shouldn't get to hide that under one global number.

### Scoring

LoCoMo questions split into three scoring regimes (aligned with the paper's reference evaluator):

- **`single_hop` — token-level F1.** SQuAD-style normalization (lowercase, drop punctuation, drop articles, collapse whitespace). The ground truth is unambiguous and any correct answer overlaps the truth tokens, so token overlap is the right metric.
- **`multi_hop` / `temporal` / `open_domain` — LLM-as-judge.** The ground truth is a natural-language explanation; token-level F1 systematically penalizes verbose-but-correct paraphrases. A separate judge model (default: GPT-4o, deliberately different from the answer model to reduce same-model-bias) decides whether the prediction is semantically equivalent to the ground truth and returns CORRECT (1.0) or INCORRECT (0.0).
- **`adversarial` — refusal judge.** The correct behavior is refusal; the ground truth is an empty string, and F1 against empty truth always returns 0 for any non-empty refusal text. A dedicated judge prompt returns REFUSAL (1.0) when the model declines to commit to a factual answer, or FABRICATION (0.0) when it answers a question that has no answer in the conversation.

### Fairness controls

- **Same answer model across systems.** Whichever model the operator chooses (default: Claude Sonnet 4.6 at temp=0), every system uses it for the final answer. A system can't win because it picked a stronger model.
- **Same judge model across systems.** Same logic.
- **Internal LLM costs reported separately.** Systems that issue their own LLM calls during ingest (Mem0's fact extractor, Statewave's optional LLM compiler) report those tokens under `internal_input_tokens` / `internal_output_tokens` so the operator sees the full bill, not just the answer-model cost.
- **Per-conversation isolation.** Every system scopes its memory by conversation id (`bench:locomo:<id>` for Statewave/Mem0, `bench-locomo-<id>` for Zep). No cross-conversation leakage.
- **Deterministic where possible.** Temperature 0, fixed seeds where SDKs expose them. LLM calls aren't perfectly deterministic but two runs should land within sampling noise.

### Resumability

Results stream to `results/run.jsonl` as the bench progresses. `swb run` is **fresh-by-default**: an existing file at the output path is deleted at startup so every run exercises the full `delete → ingest → compile → retrieve → answer` chain (otherwise the resume optimization would skip ingest for already-scored conversations, and you'd never test fixes to those layers).

Pass `--resume` to opt back in to the legacy behavior: keep the existing file and skip already-completed `(system, conversation, question)` tuples. Useful for the multi-hour full-set run that might hit a transient error (Anthropic 529, Mem0 rate-limit, kernel panic) — re-run with `--resume` to pick up from the last gap.

## Layout

```
statewave-bench/
├── src/statewave_bench/
│   ├── cli.py              # `swb` entry point: config-check / run / rescore / report
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
