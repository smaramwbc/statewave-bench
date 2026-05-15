# Contributing to statewave-bench

Thanks for your interest — `statewave-bench` is the open benchmark for AI memory runtimes (Statewave vs Mem0 vs Zep on LoCoMo) and external contributions are very welcome.

## Where to file what

`statewave-bench` Issues are **disabled** in favour of a single central tracker. Open all issues at:

> **[smaramwbc/statewave issues](https://github.com/smaramwbc/statewave/issues)**

Use the prefix `[bench]` in the title so issues route correctly.

For questions and discussion: [GitHub Discussions on the core repo](https://github.com/smaramwbc/statewave/discussions).

## Ways to contribute

- **Bug reports** — open a [central-tracker issue](https://github.com/smaramwbc/statewave/issues) prefixed `[bench]` with reproduction steps, expected vs. actual results, and which adapter / model / dataset commit it concerns.
- **New systems / adapters** — add an adapter under `src/statewave_bench/systems/` following the `base.py` interface. Open a discussion first if the adapter requires non-trivial new dependencies.
- **New evals or datasets** — propose in [Discussions](https://github.com/smaramwbc/statewave/discussions) under **Ideas & Feature Requests** before implementing — we want the bench to stay focused.
- **Result reproductions** — rerunning the bench against your own keys and posting a PR-comment / Discussion with row-level deltas is a meaningful contribution.

## Development setup

```bash
# Install uv (https://github.com/astral-sh/uv) if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the project
uv sync

# Run tests
uv run pytest tests/

# Lint
uv run ruff check src/
```

Run the bench (small smoke test against 1 conversation, all 5 systems, ~$13–18, ~15–20 min):

```bash
uv run swb run --limit 1 -o results/run_smoke.jsonl
uv run swb report -i results/run_smoke.jsonl --output-dir results/report_smoke
```

You'll need API keys for whichever systems you exercise — see `.env.example`.

## Discussing changes before opening a PR

Use the lightest-weight venue that fits the change:

- **Small, obvious fixes** — typos, doc tweaks, isolated bug fixes — go straight to a PR. No discussion needed.
- **New adapter, new metric, new dataset** — start in [Discussions](https://github.com/smaramwbc/statewave/discussions) under **Ideas & Feature Requests** with the use case and proposed shape.
- **Confirmed bugs with a clean reproduction** — open an [issue on the central tracker](https://github.com/smaramwbc/statewave/issues) with `[bench]` prefix.
- **Security concerns** — never post publicly. Follow the coordinated disclosure process in [SECURITY.md](SECURITY.md).

## Pull request process

1. **Open an issue or discussion first** for non-trivial changes so we can align on approach before you invest in code.
2. **Branch from `main`**, keep PRs focused, prefer small commits with clear messages.
3. **Add tests** for new behavior. Adapter changes especially need a regression test in `tests/`.
4. **Update docs** — README, ADRs, or per-adapter notes if you change measured behavior.
5. **Pass CI** — `ruff check src/` clean; `pytest tests/` green.
6. **Describe the change** in the PR body: motivation, approach, any benchmarks before/after.

## Licensing of contributions

`statewave-bench` is licensed under the [Apache License, Version 2.0](LICENSE). By contributing — opening a pull request, sending a patch, or otherwise submitting work — you agree that your contribution is licensed under Apache-2.0 along with the rest of the project. You retain copyright in your work.

We use the **Developer Certificate of Origin** ([DCO](https://developercertificate.org/)) for contributions. Sign your commits with `git commit -s` (or `--signoff`) to add a `Signed-off-by:` trailer asserting that you have the right to contribute the work under the project's license.

If your employer has rights to your work, please make sure they have authorized the contribution before submitting.

## Code style

- Python 3.11+, formatted with `ruff` (settings in `pyproject.toml`).
- Type hints on public APIs (adapter interfaces, scoring functions).
- Match the surrounding adapter's conventions; new adapters should look like the existing ones in `src/statewave_bench/systems/`.

## Reporting security issues

Please **do not** open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the coordinated disclosure process.

## Questions

- General questions: [GitHub Discussions](https://github.com/smaramwbc/statewave/discussions)
- Licensing questions: [licensing@statewave.ai](mailto:licensing@statewave.ai)
- Security: [security@statewave.ai](mailto:security@statewave.ai)
