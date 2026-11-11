"""Plain-text and Markdown rendering of summaries and comparisons."""

from __future__ import annotations

from typing import Sequence

from evalbench.runner import RunResult
from evalbench.stats import Comparison, ModelSummary


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, markdown: bool) -> str:
    if markdown:
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(lines)
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    def line(cells):
        return "  ".join(str(cell).ljust(width) for cell, width in zip(cells, widths)).rstrip()
    return "\n".join([line(headers), line("-" * width for width in widths), *map(line, rows)])


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_summary(run: RunResult, summaries: Sequence[ModelSummary], *, markdown: bool = False) -> str:
    headers = ["model", "n", "pass rate", "95% CI", "mean score", "errors", "avg latency", "tokens"]
    rows = [
        [
            s.model,
            str(s.n),
            _pct(s.pass_rate),
            f"{_pct(s.pass_rate_ci.low)} - {_pct(s.pass_rate_ci.high)}",
            f"{s.mean_score:.3f}",
            str(s.errors),
            f"{s.mean_latency_ms:.0f} ms",
            str(s.total_tokens),
        ]
        for s in summaries
    ]
    title = f"Dataset {run.dataset} ({run.dataset_fingerprint}), scorer {run.scorer}"
    parts = [f"## {title}" if markdown else title, "", _table(headers, rows, markdown=markdown)]

    tags = sorted({tag for s in summaries for tag in s.by_tag})
    if tags:
        tag_rows = [
            [tag, *(_tag_cell(s, tag) for s in summaries)] for tag in tags
        ]
        parts += ["", "Pass rate by tag:" if not markdown else "### Pass rate by tag", ""]
        parts.append(_table(["tag", *(s.model for s in summaries)], tag_rows, markdown=markdown))
    return "\n".join(parts) + "\n"


def _tag_cell(summary: ModelSummary, tag: str) -> str:
    if tag not in summary.by_tag:
        return "-"
    n, rate = summary.by_tag[tag]
    return f"{_pct(rate)} (n={n})"


def render_comparison(result: Comparison, *, markdown: bool = False, max_listed: int = 10) -> str:
    sign = "+" if result.mean_delta >= 0 else ""
    lines = [
        f"## {result.candidate} vs {result.baseline}" if markdown else f"{result.candidate} vs {result.baseline}",
        "",
        f"Paired examples: {result.n_paired}"
        + (f" ({result.unpaired} unpaired example(s) ignored)" if result.unpaired else ""),
        f"Mean score: {result.baseline_mean:.3f} -> {result.candidate_mean:.3f} "
        f"(delta {sign}{result.mean_delta:.3f}, 95% CI {result.delta_ci.low:+.3f} to {result.delta_ci.high:+.3f})",
        f"Per-example: {result.wins} better, {result.ties} same, {result.losses} worse "
        f"(sign test p = {result.p_value:.4f})",
        f"Verdict: {result.verdict.upper()}",
    ]
    if result.regressions:
        shown = ", ".join(result.regressions[:max_listed])
        more = len(result.regressions) - max_listed
        lines.append(
            f"Regressions (passed before, fail now): {shown}" + (f" and {more} more" if more > 0 else "")
        )
    separator = "\n\n" if markdown else "\n"
    return lines[0] + "\n\n" + separator.join(lines[2:]) + "\n"
