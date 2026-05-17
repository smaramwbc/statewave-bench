"""Read a results JSONL and produce a markdown summary + combined HTML report.

Outputs:
  - results-summary.md  — markdown table fit for pasting into the
    bench README + a launch post (overall score per system, plus
    per-category breakdown).
  - results.html        — single combined modern report: run metadata,
    summary table, overall ranking chart, per-category breakdown chart.
    Self-contained (CSS inline, Vega loaded from CDN); shareable as one
    file.

Token + latency reporting comes alongside the score so adopters see
the full tradeoff space, not just quality. A system that wins on F1
but burns 10x the tokens isn't necessarily the right choice for
high-volume deployments.

Reads the JSONL directly (no SQL, no pandas dependency surfacing
here) — polars handles aggregation, altair renders charts, both
zero-config.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import polars as pl

from .coverage import (
    compute_coverage,
    dedupe_rows,
    has_incomplete,
    missing_per_system,
)
from .metadata import load_metadata

# Neutral, colour-vision-deficiency-safe categorical palette (Okabe-Ito
# subset). Assigned to systems in sorted-name order so the mapping is
# deterministic and identical across every chart — no system is given a
# deliberately prominent or muted colour. `no_memory` is not special-cased.
_NEUTRAL_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#999999",
]


def _system_colors(systems: list[str]) -> dict[str, str]:
    """Deterministic colour per system by sorted name — neutral, no
    vendor is visually privileged."""
    return {s: _NEUTRAL_PALETTE[i % len(_NEUTRAL_PALETTE)] for i, s in enumerate(sorted(systems))}


class IncompleteResultsError(RuntimeError):
    """Raised when a results set is not publication-safe (judge_failed/
    null-score rows present, or systems answered unequal question sets)
    and the caller did not pass allow_incomplete=True."""


def _load_rows(results_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with results_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # tolerate a partial last line
    return rows


def _coverage_report(rows: list[dict[str, object]]) -> str:
    """Human-readable per-system coverage block — the publication-safety
    audit a reviewer reads before trusting any number."""
    stats = compute_coverage(rows)
    lines = [
        "| System | Expected | Completed | Scored | Failed | Judge-failed | "
        "Coverage | Scored cov. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(stats):
        c = stats[s]
        lines.append(
            f"| {s} | {c.expected_questions} | {c.completed_rows} | {c.scored_rows} | "
            f"{c.failed_rows} | {c.judge_failed_rows} | "
            f"{c.coverage:.1%} | {c.scored_coverage:.1%} |"
        )
    return "\n".join(lines)


def _publication_safety(rows: list[dict[str, object]]) -> tuple[bool, list[str]]:
    """(safe, problems). Safe iff equal question sets across systems AND
    no judge_failed/null rows AND every system fully scored."""
    problems: list[str] = []
    missing = missing_per_system(rows)
    if missing:
        for sys_, keys in sorted(missing.items()):
            sample = sorted(keys)[:5]
            problems.append(
                f"system '{sys_}' is missing {len(keys)} expected question(s) "
                f"(e.g. {sample}{' …' if len(keys) > 5 else ''})"
            )
    if has_incomplete(rows):
        stats = compute_coverage(rows)
        jf = {s: c.judge_failed_rows for s, c in stats.items() if c.judge_failed_rows}
        problems.append(
            "judge_failed / null-score rows present "
            f"({sum(jf.values())} across {', '.join(sorted(jf)) or 'unknown'}); "
            "run `swb rescore` to retry the judge"
        )
    return (not problems), problems


def render_report(
    *,
    results_path: Path,
    output_dir: Path,
    allow_incomplete: bool = False,
) -> None:
    """Read `results_path` (JSONL), validate publication-safety, then
    write summary + combined HTML to `output_dir`.

    Refuses (raises `IncompleteResultsError`) when the run has
    judge_failed/null rows or systems answered unequal question sets,
    unless `allow_incomplete=True` — in which case the report is still
    rendered but stamped NOT PUBLICATION-SAFE and headline ranking is
    suppressed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _load_rows(results_path)
    if not raw_rows:
        raise SystemExit(f"No rows found in {results_path}.")

    # Collapse resume-duplicate (system, conv, q_idx) rows to the most
    # complete one before anything is counted.
    rows = dedupe_rows(raw_rows)

    safe, problems = _publication_safety(rows)
    if not safe and not allow_incomplete:
        raise IncompleteResultsError(
            "\n".join(f"  - {p}" for p in problems)
            + f"\n  ({len(rows)} rows in {results_path.name})"
        )

    # Schema-union read from the deduped rows (NDJSON reader tolerates
    # ragged rows — failure rows carry error_type, judge_failed rows
    # carry judge_error, scored rows neither).
    buf = io.BytesIO("\n".join(json.dumps(r) for r in rows).encode("utf-8"))
    df = pl.read_ndjson(buf)

    overall = _aggregate_overall(df)
    by_category = _aggregate_by_category(df)
    metadata = _extract_metadata(df, results_path)
    metadata["publication_safe"] = safe
    metadata["safety_problems"] = problems
    metadata["coverage_table"] = _coverage_report(rows)
    metadata["run_metadata"] = load_metadata(results_path)

    _write_markdown_summary(
        overall=overall,
        by_category=by_category,
        metadata=metadata,
        out=output_dir / "results-summary.md",
    )
    _write_combined_html(
        overall=overall,
        by_category=by_category,
        metadata=metadata,
        out=output_dir / "results.html",
    )


# ── Aggregation ───────────────────────────────────────────────────────────


def _aggregate_overall(df: pl.DataFrame) -> pl.DataFrame:
    """One row per system with mean score + total tokens + median latency.
    `score.mean()` skips null scores (judge_failed rows), so transient
    judge failures don't drag the mean down — only the count drops.

    Two means side-by-side:
      - `mean_score` over every question (our default canonical number)
      - `mean_score_excl_adversarial` over open-domain / multi_hop /
        single_hop / temporal only — the SOTA-comparable subset that
        Mem0 / Honcho / Backboard / Memori publish numbers against.
    """
    # Per-system "exclude adversarial" mean. Done as a side aggregation
    # so the full-mean version stays a single pl expression; joined back
    # below.
    excl_mean = (
        df.filter(pl.col("category") != "adversarial")
        .group_by("system")
        .agg(pl.col("score").mean().alias("mean_score_excl_adversarial"))
    )
    aggs = [
        pl.col("score").mean().alias("mean_score"),
        pl.col("score").count().alias("n_scored"),
        pl.len().alias("n_total"),
        pl.col("answer_input_tokens").sum().alias("total_input_tokens"),
        pl.col("answer_output_tokens").sum().alias("total_output_tokens"),
        pl.col("internal_input_tokens").sum().alias("internal_input_tokens"),
        pl.col("internal_output_tokens").sum().alias("internal_output_tokens"),
        pl.col("elapsed_ms").median().alias("median_elapsed_ms"),
        pl.col("elapsed_ms").quantile(0.95).alias("p95_elapsed_ms"),
    ]
    # Context-size columns are new (added with the publication-safety
    # work); guard so older JSONL without them still renders. Mean over
    # non-null so failure rows (null context) don't drag it to zero.
    if "retrieved_context_chars" in df.columns:
        aggs.append(pl.col("retrieved_context_chars").mean().alias("avg_ctx_chars"))
    if "retrieved_context_tokens_estimate" in df.columns:
        aggs.append(pl.col("retrieved_context_tokens_estimate").mean().alias("avg_ctx_tokens_est"))
    if "retrieved_items_count" in df.columns:
        aggs.append(pl.col("retrieved_items_count").mean().alias("avg_ctx_items"))
    return (
        df.group_by("system")
        .agg(*aggs)
        .join(excl_mean, on="system", how="left")
        .sort("mean_score", descending=True, nulls_last=True)
    )


def _aggregate_by_category(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["system", "category"])
        .agg(
            pl.col("score").mean().alias("mean_score"),
            pl.col("score").count().alias("n_scored"),
        )
        .sort(["category", "mean_score"], descending=[False, True])
    )


def _extract_metadata(df: pl.DataFrame, results_path: Path) -> dict[str, object]:
    """Pull display-only metadata from the JSONL — answer model, system
    count, conversation count, generation time."""
    answer_models = df.select("answer_model").unique().to_series().to_list()
    return {
        "n_conversations": df.select("conversation_id").n_unique(),
        "n_systems": df.select("system").n_unique(),
        "n_questions": df.height,
        "answer_models": [m for m in answer_models if m],
        "results_path": results_path.name,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }


# ── Markdown summary (unchanged shape — operators paste this into READMEs) ─


def _write_markdown_summary(
    *,
    overall: pl.DataFrame,
    by_category: pl.DataFrame,
    metadata: dict[str, object],
    out: Path,
) -> None:
    safe = bool(metadata.get("publication_safe"))
    _p = metadata.get("safety_problems")
    problems: list[str] = [str(x) for x in _p] if isinstance(_p, list) else []
    lines: list[str] = ["# LoCoMo benchmark results", ""]

    if safe:
        lines += [
            "> ✅ **Publication-safe:** every system answered the same "
            "question set and every question was scored.",
            "",
        ]
    else:
        lines += [
            "> ⚠️ **NOT PUBLICATION-SAFE — do not publish or rank these "
            "numbers.** Coverage is incomplete or unequal:",
            "",
        ]
        lines += [f"> - {p}" for p in problems]
        lines += [
            "",
            "> Headline ranking is suppressed. Fix the run (`swb rescore` "
            "for judge_failed rows; re-run missing systems) then regenerate.",
            "",
        ]

    lines += ["## Coverage", ""]
    lines.append(str(metadata.get("coverage_table", "(coverage unavailable)")))
    lines += [
        "",
        "_Failed = explicit zero-score rows (ingest/answer/timeout/abort). "
        "Judge-failed = answer produced but judge call failed (score null, "
        "excluded from means — blocks publication)._",
        "",
    ]

    lines += ["## Overall", ""]
    if not safe:
        lines += [
            "_Sorted by mean score for inspection only — this is **not** a "
            "ranking; the run is not publication-safe._",
            "",
        ]
    has_ctx = "avg_ctx_tokens_est" in overall.columns
    ctx_hdr = " Avg ctx tok | Avg ctx items |" if has_ctx else ""
    ctx_sep = "---:|---:|" if has_ctx else ""
    lines.append(
        "| System | Mean score | Mean (excl. adv) | n | Avg input tok / q | "
        f"Avg output tok / q | Median latency (s) | p95 latency (s) |{ctx_hdr}"
    )
    lines.append(f"|---|---:|---:|---:|---:|---:|---:|---:|{ctx_sep}")
    for row in overall.iter_rows(named=True):
        n = row["n_scored"] or 1
        mean = row["mean_score"] if row["mean_score"] is not None else 0.0
        mean_excl = row.get("mean_score_excl_adversarial")
        mean_excl_str = f"{mean_excl:.3f}" if isinstance(mean_excl, float) else "—"
        ctx_cells = ""
        if has_ctx:
            ct = row.get("avg_ctx_tokens_est")
            ci = row.get("avg_ctx_items")
            ct_s = f"{ct:.0f}" if isinstance(ct, (int, float)) else "—"
            ci_s = f"{ci:.1f}" if isinstance(ci, (int, float)) else "—"
            ctx_cells = f" {ct_s} | {ci_s} |"
        lines.append(
            f"| {row['system']} | {mean:.3f} | {mean_excl_str} | {n} | "
            f"{row['total_input_tokens'] / n:.0f} | {row['total_output_tokens'] / n:.0f} | "
            f"{row['median_elapsed_ms'] / 1000:.2f} | {row['p95_elapsed_ms'] / 1000:.2f} |"
            f"{ctx_cells}"
        )

    lines += ["", "## By category", ""]
    pivot = by_category.pivot(on="system", index="category", values="mean_score")
    headers = ["category"] + [c for c in pivot.columns if c != "category"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---:" if h != "category" else "---" for h in headers]) + "|")
    for row in pivot.iter_rows(named=True):
        cells = [str(row["category"])]
        for h in headers[1:]:
            v = row.get(h)
            cells.append(f"{v:.3f}" if isinstance(v, float) else "—")
        lines.append("| " + " | ".join(cells) + " |")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Charts ────────────────────────────────────────────────────────────────


def _color_scale(systems: list[str]) -> alt.Scale:
    """Neutral, deterministic colour scale (by sorted system name) so
    the same system gets the same colour in every chart and no vendor
    is visually privileged."""
    colors = _system_colors(systems)
    domain = sorted(colors)
    return alt.Scale(domain=domain, range=[colors[s] for s in domain])


def _overall_chart_spec(overall: pl.DataFrame) -> dict[str, object]:
    """Horizontal-bar overall chart, sorted descending by score.
    Returns the Vega-Lite spec dict (not an altair Chart) so the
    HTML template can embed it directly."""
    systems = overall.get_column("system").to_list()
    chart = (
        alt.Chart(overall)
        .mark_bar(cornerRadiusEnd=4, height=28)
        .encode(
            y=alt.Y(
                "system:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelFontSize=14, labelFontWeight=500),
            ),
            x=alt.X(
                "mean_score:Q",
                title="Mean score (higher = better)",
                axis=alt.Axis(format=".0%", labelFontSize=12, titleFontSize=12, grid=True),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("system:N", scale=_color_scale(systems), legend=None),
            tooltip=[
                alt.Tooltip("system:N", title="System"),
                alt.Tooltip("mean_score:Q", title="Mean score", format=".3f"),
                alt.Tooltip("n_scored:Q", title="Scored"),
                alt.Tooltip("median_elapsed_ms:Q", title="Median latency (ms)", format=".0f"),
            ],
        )
        .properties(height=alt.Step(40), width="container")
    )
    labels = (
        alt.Chart(overall)
        .mark_text(align="left", dx=6, fontSize=13, fontWeight="bold", color="#1f2937")
        .encode(
            y=alt.Y("system:N", sort="-x"),
            x=alt.X("mean_score:Q"),
            text=alt.Text("mean_score:Q", format=".3f"),
        )
    )
    spec: dict[str, object] = (chart + labels).configure_view(stroke=None).to_dict()
    return spec


def _by_category_chart_spec(by_category: pl.DataFrame) -> dict[str, object]:
    """Grouped bar chart per category. Same color encoding as the
    overall chart so the eye tracks each system across both."""
    systems = by_category.get_column("system").unique().to_list()
    chart = (
        alt.Chart(by_category)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "system:N",
                title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=10, labels=False, ticks=False),
            ),
            y=alt.Y(
                "mean_score:Q",
                title="Mean score",
                axis=alt.Axis(format=".0%", labelFontSize=11, titleFontSize=12, grid=True),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "system:N",
                scale=_color_scale(systems),
                legend=alt.Legend(title=None, orient="bottom", labelFontSize=13, symbolSize=180),
            ),
            column=alt.Column(
                "category:N",
                title=None,
                header=alt.Header(labelFontSize=13, labelFontWeight=600, labelColor="#1f2937"),
                spacing=10,
            ),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("system:N", title="System"),
                alt.Tooltip("mean_score:Q", title="Score", format=".3f"),
                alt.Tooltip("n_scored:Q", title="Scored"),
            ],
        )
        .properties(width=120, height=260)
        .configure_view(stroke=None)
    )
    spec: dict[str, object] = chart.to_dict()
    return spec


# ── Combined modern HTML ──────────────────────────────────────────────────


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>statewave-bench — LoCoMo results</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5.30.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5.21.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.26.0"></script>
<style>
  :root {{
    --fg: #111827;
    --fg-muted: #6b7280;
    --bg: #ffffff;
    --bg-soft: #f9fafb;
    --border: #e5e7eb;
    --accent: #7c3aed;
    --radius: 12px;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --fg: #f3f4f6;
      --fg-muted: #9ca3af;
      --bg: #0b0d12;
      --bg-soft: #11141b;
      --border: #1f2430;
      --accent: #a78bfa;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", system-ui, sans-serif;
    font-size: 15px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}
  .container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 56px 32px 96px;
  }}
  header {{
    margin-bottom: 48px;
  }}
  .eyebrow {{
    color: var(--accent);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  h1 {{
    font-size: 36px;
    line-height: 1.15;
    margin: 0 0 12px;
    letter-spacing: -0.02em;
  }}
  .lede {{
    color: var(--fg-muted);
    font-size: 17px;
    max-width: 64ch;
    margin: 0;
  }}
  .meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    margin: 28px 0 0;
    padding: 16px 20px;
    background: var(--bg-soft);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: 13px;
  }}
  .meta dt {{
    color: var(--fg-muted);
    font-weight: 500;
    margin-bottom: 2px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .meta dd {{
    margin: 0;
    font-weight: 600;
  }}
  section {{
    margin-bottom: 56px;
  }}
  section h2 {{
    font-size: 22px;
    letter-spacing: -0.01em;
    margin: 0 0 6px;
  }}
  section .section-lede {{
    color: var(--fg-muted);
    margin: 0 0 20px;
  }}
  .card {{
    background: var(--bg-soft);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    font-variant-numeric: tabular-nums;
  }}
  th, td {{
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    font-weight: 600;
    color: var(--fg-muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  td.num, th.num {{ text-align: right; }}
  tr:last-child td {{ border-bottom: none; }}
  .system-dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
  }}
  .chart-host {{
    width: 100%;
    min-height: 320px;
  }}
  footer {{
    margin-top: 64px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    color: var(--fg-muted);
    font-size: 13px;
  }}
  footer a {{ color: var(--accent); text-decoration: none; }}
  footer a:hover {{ text-decoration: underline; }}
  .banner {{
    margin: 24px 0 0;
    padding: 14px 18px;
    border-radius: var(--radius);
    font-size: 14px;
    border: 1px solid var(--border);
  }}
  .banner.ok {{ background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }}
  .banner.warn {{ background: #fef2f2; color: #991b1b; border-color: #fecaca; }}
  .banner ul {{ margin: 8px 0 4px 18px; padding: 0; }}
  pre.coverage {{
    overflow-x: auto;
    background: var(--bg-soft);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    font-size: 12.5px;
    line-height: 1.5;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="eyebrow">statewave-bench</div>
    <h1>LoCoMo benchmark results</h1>
    <p class="lede">{lede}</p>
    {safety_banner}
    <dl class="meta">
      <div><dt>Conversations</dt><dd>{n_conversations}</dd></div>
      <div><dt>Systems</dt><dd>{n_systems}</dd></div>
      <div><dt>Question-runs</dt><dd>{n_questions}</dd></div>
      <div><dt>Answer model</dt><dd>{answer_models}</dd></div>
      <div><dt>Source</dt><dd>{results_path}</dd></div>
      <div><dt>Generated</dt><dd>{generated_at}</dd></div>
    </dl>
    <p class="section-lede" style="margin-top:12px">Run config{meta_extra}</p>
  </header>

  <section>
    <h2>Coverage</h2>
    <p class="section-lede">
      Per-system completeness audit. A number is only trustworthy if every
      system answered the same question set and every question was scored.
      Failed = explicit zero-score rows; judge-failed = answer produced but
      the judge call failed (score null, blocks publication).
    </p>
    <div class="card"><pre class="coverage">{coverage_html}</pre></div>
  </section>

  <section>
    <h2>Overall (mean score across categories)</h2>
    <p class="section-lede">Higher is better. Order is by score; it is a
      ranking only when the run is publication-safe (see banner above).</p>
    <div class="card"><div id="overall" class="chart-host"></div></div>
  </section>

  <section>
    <h2>Per-category breakdown</h2>
    <p class="section-lede">
      LoCoMo splits questions into five categories. Each tests something different —
      a system that wins overall but bombs <em>temporal</em> shouldn't hide it.
    </p>
    <div class="card"><div id="by_category" class="chart-host"></div></div>
  </section>

  <section>
    <h2>Tradeoff table</h2>
    <p class="section-lede">
      Mean score alongside per-question token cost and latency.
      Token-thin systems win on cost; token-rich systems often win on quality.
    </p>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>System</th>
            <th class="num">Mean</th>
            <th class="num">Scored</th>
            <th class="num">Input tok / q</th>
            <th class="num">Output tok / q</th>
            {ctx_header}
            <th class="num">p50 latency</th>
            <th class="num">p95 latency</th>
          </tr>
        </thead>
        <tbody>
{table_rows}
        </tbody>
      </table>
    </div>
  </section>

  <footer>
    Generated by <a href="https://github.com/smaramwbc/statewave-bench">statewave-bench</a>.
    Token-level F1 for single_hop, LLM-as-judge for the rest, refusal judge for adversarial —
    matches LoCoMo's reference methodology.
  </footer>
</div>

<script>
  const overallSpec = {overall_json};
  const byCategorySpec = {by_category_json};
  const vegaOpts = {{actions: false, renderer: "svg"}};
  vegaEmbed("#overall", overallSpec, vegaOpts);
  vegaEmbed("#by_category", byCategorySpec, vegaOpts);
</script>
</body>
</html>
"""


def _write_combined_html(
    *,
    overall: pl.DataFrame,
    by_category: pl.DataFrame,
    metadata: dict[str, object],
    out: Path,
) -> None:
    safe = bool(metadata.get("publication_safe"))
    _p = metadata.get("safety_problems")
    problems: list[str] = [str(x) for x in _p] if isinstance(_p, list) else []

    # A "leader" is only stated when the run is publication-safe. An
    # unsafe run gets a neutral, non-ranking lede.
    if safe and overall.height and overall.row(0, named=True).get("mean_score") is not None:
        leader_row = overall.row(0, named=True)
        lede = (
            f"{metadata['n_questions']} question-runs across "
            f"{metadata['n_systems']} systems. Highest mean score: "
            f"{leader_row['system']} ({leader_row['mean_score']:.3f})."
        )
    else:
        lede = f"{metadata['n_questions']} question-runs across {metadata['n_systems']} systems."

    if safe:
        safety_banner = (
            '<div class="banner ok">✅ Publication-safe: every system '
            "answered the same question set and every question was scored.</div>"
        )
    else:
        items = "".join(f"<li>{p}</li>" for p in problems)
        safety_banner = (
            '<div class="banner warn"><strong>⚠️ NOT PUBLICATION-SAFE — '
            "do not publish or rank these numbers.</strong> Coverage is "
            f"incomplete or unequal:<ul>{items}</ul>"
            "Sorted by score for inspection only; this is not a ranking.</div>"
        )

    colors = _system_colors(overall.get_column("system").to_list())
    has_ctx = "avg_ctx_tokens_est" in overall.columns
    table_rows: list[str] = []
    for row in overall.iter_rows(named=True):
        n = row["n_scored"] or 1
        color = colors.get(row["system"], "#999999")
        mean_text = f"{row['mean_score']:.3f}" if row["mean_score"] is not None else "—"
        ctx_cell = ""
        if has_ctx:
            ct = row.get("avg_ctx_tokens_est")
            ct_s = f"{ct:.0f}" if isinstance(ct, (int, float)) else "—"
            ctx_cell = f'<td class="num">{ct_s}</td>'
        table_rows.append(
            "<tr>"
            f'<td><span class="system-dot" style="background:{color}"></span>'
            f"{row['system']}</td>"
            f'<td class="num">{mean_text}</td>'
            f'<td class="num">{row["n_scored"]}</td>'
            f'<td class="num">{row["total_input_tokens"] / n:.0f}</td>'
            f'<td class="num">{row["total_output_tokens"] / n:.0f}</td>'
            f"{ctx_cell}"
            f'<td class="num">{row["median_elapsed_ms"] / 1000:.2f} s</td>'
            f'<td class="num">{row["p95_elapsed_ms"] / 1000:.2f} s</td>'
            "</tr>"
        )

    answer_models = metadata.get("answer_models") or []
    answer_models_str = (
        ", ".join(str(m) for m in answer_models)
        if isinstance(answer_models, list) and answer_models
        else "(none)"
    )
    ctx_header = '<th class="num">Avg ctx tok</th>' if has_ctx else ""

    rm = metadata.get("run_metadata") or {}
    if isinstance(rm, dict) and rm:
        meta_extra = (
            f" · mode {rm.get('bench_mode', '?')} · scoring "
            f"{rm.get('scoring_mode', '?')} · judge {rm.get('judge_model', '?')}"
            f" · commit {str(rm.get('git_commit') or '?')[:10]}"
        )
    else:
        meta_extra = " · run metadata: (none — predates metadata capture)"

    html = _HTML_TEMPLATE.format(
        lede=lede,
        safety_banner=safety_banner,
        coverage_html=str(metadata.get("coverage_table", "")),
        meta_extra=meta_extra,
        ctx_header=ctx_header,
        n_conversations=metadata["n_conversations"],
        n_systems=metadata["n_systems"],
        n_questions=metadata["n_questions"],
        answer_models=answer_models_str,
        results_path=metadata["results_path"],
        generated_at=metadata["generated_at"],
        table_rows="\n".join(table_rows),
        overall_json=json.dumps(_overall_chart_spec(overall)),
        by_category_json=json.dumps(_by_category_chart_spec(by_category)),
    )

    out.write_text(html, encoding="utf-8")
