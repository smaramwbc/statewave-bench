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

import json
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import polars as pl

# Consistent color per system across every chart so the same vendor
# isn't a different color in two side-by-side images. Statewave
# deliberately uses a strong purple — it's the project the bench
# was written for and the eye should land on it first. Other vendors
# get distinct but more neutral tones; no_memory sits in slate so it
# reads as "the floor" without competing for attention.
SYSTEM_COLORS: dict[str, str] = {
    "statewave": "#7c3aed",
    "mem0": "#0891b2",
    "zep": "#059669",
    "naive": "#d97706",
    "no_memory": "#94a3b8",
}


def render_report(
    *,
    results_path: Path,
    output_dir: Path,
) -> None:
    """Read `results_path` (JSONL), write summary + combined HTML to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_ndjson(results_path)
    if df.is_empty():
        raise SystemExit(f"No rows found in {results_path}.")

    overall = _aggregate_overall(df)
    by_category = _aggregate_by_category(df)
    metadata = _extract_metadata(df, results_path)

    _write_markdown_summary(
        overall=overall,
        by_category=by_category,
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
    judge failures don't drag the mean down — only the count drops."""
    return (
        df.group_by("system")
        .agg(
            pl.col("score").mean().alias("mean_score"),
            pl.col("score").count().alias("n_scored"),
            pl.len().alias("n_total"),
            pl.col("answer_input_tokens").sum().alias("total_input_tokens"),
            pl.col("answer_output_tokens").sum().alias("total_output_tokens"),
            pl.col("internal_input_tokens").sum().alias("internal_input_tokens"),
            pl.col("internal_output_tokens").sum().alias("internal_output_tokens"),
            pl.col("elapsed_ms").median().alias("median_elapsed_ms"),
            pl.col("elapsed_ms").quantile(0.95).alias("p95_elapsed_ms"),
        )
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
    out: Path,
) -> None:
    lines: list[str] = ["# LoCoMo benchmark results", ""]

    lines += ["## Overall", ""]
    lines.append(
        "| System | Mean score | n | Avg input tok / q | Avg output tok / q | "
        "Median latency (s) | p95 latency (s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in overall.iter_rows(named=True):
        n = row["n_scored"] or 1
        mean = row["mean_score"] if row["mean_score"] is not None else 0.0
        lines.append(
            f"| {row['system']} | {mean:.3f} | {n} | "
            f"{row['total_input_tokens'] / n:.0f} | {row['total_output_tokens'] / n:.0f} | "
            f"{row['median_elapsed_ms'] / 1000:.2f} | {row['p95_elapsed_ms'] / 1000:.2f} |"
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


def _color_scale() -> alt.Scale:
    """Shared color scale so the same system gets the same color in
    every chart. Unknown systems fall through to the default palette."""
    return alt.Scale(
        domain=list(SYSTEM_COLORS),
        range=list(SYSTEM_COLORS.values()),
    )


def _overall_chart_spec(overall: pl.DataFrame) -> dict[str, object]:
    """Horizontal-bar overall ranking, sorted descending by score.
    Returns the Vega-Lite spec dict (not an altair Chart) so the
    HTML template can embed it directly."""
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
            color=alt.Color("system:N", scale=_color_scale(), legend=None),
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
                scale=_color_scale(),
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
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="eyebrow">statewave-bench</div>
    <h1>LoCoMo benchmark results</h1>
    <p class="lede">{lede}</p>
    <dl class="meta">
      <div><dt>Conversations</dt><dd>{n_conversations}</dd></div>
      <div><dt>Systems</dt><dd>{n_systems}</dd></div>
      <div><dt>Question-runs</dt><dd>{n_questions}</dd></div>
      <div><dt>Answer model</dt><dd>{answer_models}</dd></div>
      <div><dt>Source</dt><dd>{results_path}</dd></div>
      <div><dt>Generated</dt><dd>{generated_at}</dd></div>
    </dl>
  </header>

  <section>
    <h2>Overall ranking</h2>
    <p class="section-lede">Mean score across all categories. Higher is better.</p>
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
    leader_row = overall.row(0, named=True) if overall.height else None
    if leader_row and leader_row.get("mean_score") is not None:
        lede = (
            f"<strong>{leader_row['system']}</strong> leads with a mean score of "
            f"<strong>{leader_row['mean_score']:.3f}</strong> across "
            f"{metadata['n_questions']} question-runs."
        )
    else:
        lede = f"{metadata['n_questions']} question-runs across {metadata['n_systems']} systems."

    table_rows: list[str] = []
    for row in overall.iter_rows(named=True):
        n = row["n_scored"] or 1
        color = SYSTEM_COLORS.get(row["system"], "#9ca3af")
        mean_text = f"{row['mean_score']:.3f}" if row["mean_score"] is not None else "—"
        table_rows.append(
            "<tr>"
            f'<td><span class="system-dot" style="background:{color}"></span>'
            f"{row['system']}</td>"
            f'<td class="num">{mean_text}</td>'
            f'<td class="num">{row["n_scored"]}</td>'
            f'<td class="num">{row["total_input_tokens"] / n:.0f}</td>'
            f'<td class="num">{row["total_output_tokens"] / n:.0f}</td>'
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

    html = _HTML_TEMPLATE.format(
        lede=lede,
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
