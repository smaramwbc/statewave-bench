"""Read a results JSONL and produce a markdown table + Vega-Lite charts.

Outputs:
  - results-summary.md  — markdown table fit for pasting into the
    bench README + a launch post (overall score per system, plus
    per-category breakdown)
  - results-overall.html — Vega-Lite bar chart, overall scores
  - results-by-category.html — grouped bar chart, per-category scores

Token + latency reporting comes alongside the score so adopters see
the full tradeoff space, not just quality. A system that wins on F1
but burns 10x the tokens isn't necessarily the right choice for
high-volume deployments.

Reads the JSONL directly (no SQL, no pandas dependency surfacing
here) — polars handles aggregation, altair renders charts, both
zero-config.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import polars as pl


def render_report(
    *,
    results_path: Path,
    output_dir: Path,
) -> None:
    """Read `results_path` (JSONL), write summary + charts to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_ndjson(results_path)
    if df.is_empty():
        raise SystemExit(f"No rows found in {results_path}.")

    overall = _aggregate_overall(df)
    by_category = _aggregate_by_category(df)

    _write_markdown_summary(
        overall=overall,
        by_category=by_category,
        out=output_dir / "results-summary.md",
    )
    _write_overall_chart(
        overall=overall,
        out=output_dir / "results-overall.html",
    )
    _write_by_category_chart(
        by_category=by_category,
        out=output_dir / "results-by-category.html",
    )


# ── Aggregation ───────────────────────────────────────────────────────────


def _aggregate_overall(df: pl.DataFrame) -> pl.DataFrame:
    """One row per system with mean score + total tokens + median latency."""
    return (
        df.group_by("system")
        .agg(
            pl.col("score").mean().alias("mean_score"),
            pl.col("score").count().alias("n_questions"),
            pl.col("answer_input_tokens").sum().alias("total_input_tokens"),
            pl.col("answer_output_tokens").sum().alias("total_output_tokens"),
            pl.col("internal_input_tokens").sum().alias("internal_input_tokens"),
            pl.col("internal_output_tokens").sum().alias("internal_output_tokens"),
            pl.col("elapsed_ms").median().alias("median_elapsed_ms"),
            pl.col("elapsed_ms").quantile(0.95).alias("p95_elapsed_ms"),
        )
        .sort("mean_score", descending=True)
    )


def _aggregate_by_category(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["system", "category"])
        .agg(
            pl.col("score").mean().alias("mean_score"),
            pl.col("score").count().alias("n_questions"),
        )
        .sort(["category", "mean_score"], descending=[False, True])
    )


# ── Markdown summary ──────────────────────────────────────────────────────


def _write_markdown_summary(
    *,
    overall: pl.DataFrame,
    by_category: pl.DataFrame,
    out: Path,
) -> None:
    lines: list[str] = ["# LoCoMo benchmark results", ""]

    # ── Overall ──────────────────────────────────────────────────────
    lines += ["## Overall", ""]
    lines.append(
        "| System | Mean score | n | Avg input tok / q | Avg output tok / q | "
        "Median latency (s) | p95 latency (s) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    for row in overall.iter_rows(named=True):
        n = row["n_questions"] or 1
        lines.append(
            "| {system} | {mean:.3f} | {n} | {avg_in:.0f} | {avg_out:.0f} | "
            "{p50:.2f} | {p95:.2f} |".format(
                system=row["system"],
                mean=row["mean_score"],
                n=n,
                avg_in=row["total_input_tokens"] / n,
                avg_out=row["total_output_tokens"] / n,
                p50=row["median_elapsed_ms"] / 1000,
                p95=row["p95_elapsed_ms"] / 1000,
            )
        )

    # ── Per-category ─────────────────────────────────────────────────
    lines += ["", "## By category", ""]
    pivot = by_category.pivot(
        on="system",
        index="category",
        values="mean_score",
    )
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


def _write_overall_chart(*, overall: pl.DataFrame, out: Path) -> None:
    chart = (
        alt.Chart(overall.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("mean_score:Q", title="Mean score (higher is better)"),
            y=alt.Y("system:N", sort="-x", title=None),
            color=alt.Color("system:N", legend=None),
            tooltip=["system", "mean_score", "n_questions"],
        )
        .properties(
            title="LoCoMo benchmark — overall mean score per system",
            width=600,
        )
    )
    chart.save(str(out))


def _write_by_category_chart(*, by_category: pl.DataFrame, out: Path) -> None:
    chart = (
        alt.Chart(by_category.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("system:N", title=None),
            y=alt.Y("mean_score:Q", title="Mean score"),
            color="system:N",
            column=alt.Column("category:N", title="Category"),
            tooltip=["system", "category", "mean_score", "n_questions"],
        )
        .properties(
            title="LoCoMo benchmark — per-category breakdown",
        )
    )
    chart.save(str(out))
