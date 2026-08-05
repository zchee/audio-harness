"""Chart rendering for benchmark results.

Two charts, adapted from the pipecat-ai/stt-benchmark plot styling (the visual
design is theirs; every number comes from this harness' own measurements):

* **Pareto frontier** — latency versus error rate, one dot per provider, with
  the set of undominated providers highlighted. This is the decision chart: any
  provider not on the frontier is beaten on both axes by someone on it.
* **Latency range** — one row per provider showing p50 and p95, sorted fastest
  first. The row length is the size of the provider's tail.

Charts are drawn per language and per mode. Mixing languages on one chart would
place WER next to CER and invite reading one as beating the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

# Chart chrome, shared by both charts: quiet ink/grid tokens, one hue for the
# data, and a single highlight reserved for the Pareto band.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
DOT = "#2a78d6"
BAND = "#1baf7a"

RANGE_P50 = "#1c5cab"
RANGE_P95 = "#3987e5"

LATENCY_METRICS = {
    "finalize": {
        "p50": "finalize_p50_s",
        "p95": "finalize_p95_s",
        "label": "Finalize (end of speech → final transcript)",
    },
    "ttft": {
        "p50": "ttft_p50_s",
        "p95": "ttft_p95_s",
        "label": "TTFT (first audio byte → first hypothesis)",
    },
}


@dataclass(slots=True, frozen=True)
class _Point:
    """One provider's position on the latency/accuracy plane."""

    name: str
    latency_ms: float
    latency_p95_ms: float
    error_pct: float


def _points(
    frame: pl.DataFrame, *, mode: str, language: str, metric: str
) -> list[_Point]:
    """Extract plottable providers for one mode and language.

    Rows missing either coordinate are dropped rather than plotted at zero,
    which would place a failed provider at the ideal corner of the chart.
    """
    keys = LATENCY_METRICS[metric]
    points = []
    for row in frame.iter_rows(named=True):
        if row.get("mode") != mode or row.get("language") != language:
            continue
        latency = row.get(keys["p50"])
        error = row.get("error_rate")
        if latency is None or error is None:
            continue
        points.append(
            _Point(
                name=row["provider"],
                latency_ms=latency * 1000,
                latency_p95_ms=(row.get(keys["p95"]) or latency) * 1000,
                error_pct=error * 100,
            )
        )
    return points


def _pareto(points: list[_Point]) -> list[_Point]:
    """Return the undominated points, fastest first.

    A point is dominated when some other point is at least as good on both
    axes and strictly better on one.
    """
    optimal = [
        p
        for p in points
        if not any(
            q.latency_ms <= p.latency_ms
            and q.error_pct <= p.error_pct
            and (q.latency_ms < p.latency_ms or q.error_pct < p.error_pct)
            for q in points
        )
    ]
    return sorted(optimal, key=lambda p: p.latency_ms)


def _chrome(ax: object, *, xlabel: str, ylabel: str, title: str) -> None:
    """Apply the shared recessive chart chrome."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)  # type: ignore[attr-defined]
    ax.set_axisbelow(True)  # type: ignore[attr-defined]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)  # type: ignore[attr-defined]
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)  # type: ignore[attr-defined]
    ax.tick_params(colors=INK_2, labelsize=10)  # type: ignore[attr-defined]
    ax.set_xlabel(xlabel, fontsize=11, color=INK)  # type: ignore[attr-defined]
    ax.set_ylabel(ylabel, fontsize=11, color=INK)  # type: ignore[attr-defined]
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=14)  # type: ignore[attr-defined]


def plot_pareto(
    frame: pl.DataFrame,
    output: Path,
    *,
    mode: str = "stream",
    language: str,
    metric: str = "finalize",
    metric_label: str = "WER",
) -> Path | None:
    """Render the latency-versus-accuracy scatter with the frontier highlighted.

    Args:
        frame: Summary frame from :func:`audio_harness.report.stt_summary_frame`.
        output: PNG path to write.
        mode: Transport mode to plot; batch rows have no latency axis.
        language: BCP-47 tag selecting the rows.
        metric: ``finalize`` or ``ttft``.
        metric_label: Accuracy metric name for the axis (WER or CER).

    Returns:
        The written path, or ``None`` when fewer than two providers have both
        coordinates — a one-point Pareto chart would imply a comparison that
        does not exist.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    points = _points(frame, mode=mode, language=language, metric=metric)
    if len(points) < 2:
        return None
    frontier = _pareto(points)

    fig, ax = plt.subplots(figsize=(12.5, 8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    fx = [p.latency_ms for p in frontier]
    fy = [p.error_pct for p in frontier]
    if len(frontier) > 1:
        ax.plot(
            fx,
            fy,
            color=BAND,
            alpha=0.22,
            linewidth=30,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )
    ax.scatter(
        fx,
        fy,
        s=200,
        facecolors="none",
        edgecolors=BAND,
        linewidths=1.8,
        alpha=0.85,
        zorder=4,
    )

    xs = [p.latency_ms for p in points]
    ys = [p.error_pct for p in points]
    ax.scatter(xs, ys, s=55, color=DOT, edgecolors=SURFACE, linewidths=1.2, zorder=5)

    leader = {"arrowstyle": "-", "color": MUTED, "alpha": 0.6, "lw": 0.8}
    texts = [
        ax.text(p.latency_ms, p.error_pct, p.name, fontsize=7.5, color=INK_2, zorder=6)
        for p in points
    ]
    try:
        from adjustText import adjust_text

        adjust_text(
            texts,
            x=xs,
            y=ys,
            ax=ax,
            arrowprops=leader,
            expand=(1.5, 1.9),
            force_text=(0.6, 1.2),
        )
    except ImportError:
        # Fixed offsets are legible for the handful of providers we plot;
        # adjustText only becomes necessary when charts get crowded.
        for text in texts:
            text.set_position((text.get_position()[0] + 8, text.get_position()[1]))

    x_range = (max(xs) - min(xs)) or max(xs) * 0.2 or 1
    y_range = (max(ys) - min(ys)) or max(ys) * 0.2 or 1
    ax.set_xlim(min(xs) - 0.08 * x_range, max(xs) + 0.10 * x_range)
    ax.set_ylim(min(ys) - 0.08 * y_range, max(ys) + 0.12 * y_range)

    keys = LATENCY_METRICS[metric]
    _chrome(
        ax,
        xlabel=f"{keys['label']} p50 (ms) (lower is better)",
        ylabel=f"{metric_label} (%) (lower is better)",
        title=f"STT Pareto frontier — {language} {mode}",
    )
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=BAND,
                alpha=0.35,
                linewidth=12,
                solid_capstyle="round",
                label="Pareto frontier",
            )
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor=GRID,
        fontsize=11,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return output


def plot_latency_range(
    frame: pl.DataFrame,
    output: Path,
    *,
    mode: str = "stream",
    language: str,
    metric: str = "finalize",
) -> Path | None:
    """Render per-provider latency rows (p50 → p95), fastest at the top.

    Args:
        frame: Summary frame from :func:`audio_harness.report.stt_summary_frame`.
        output: PNG path to write.
        mode: Transport mode to plot.
        language: BCP-47 tag selecting the rows.
        metric: ``finalize`` or ``ttft``.

    Returns:
        The written path, or ``None`` when no provider has the metric.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    points = _points(frame, mode=mode, language=language, metric=metric)
    if not points:
        return None
    points.sort(key=lambda p: p.latency_ms)

    gradient = LinearSegmentedColormap.from_list(
        "latency", [_mix(RANGE_P50, SURFACE, 0.55), _mix(RANGE_P95, SURFACE, 0.55)]
    )

    fig, ax = plt.subplots(figsize=(12.5, 0.9 * len(points) + 2.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ys = range(len(points), 0, -1)
    max_p95 = 0.0
    for y, p in zip(ys, points, strict=True):
        max_p95 = max(max_p95, p.latency_p95_ms)
        if p.latency_p95_ms > p.latency_ms:
            gx = np.linspace(p.latency_ms, p.latency_p95_ms, 60)
            segments = [((gx[i], y), (gx[i + 1], y)) for i in range(len(gx) - 1)]
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=gradient(np.linspace(0, 1, len(segments))),
                    linewidths=3,
                    capstyle="butt",
                    zorder=2,
                )
            )
        ax.scatter(
            [p.latency_p95_ms],
            [y],
            s=55,
            color=RANGE_P95,
            edgecolors=SURFACE,
            linewidths=1.4,
            zorder=4,
        )
        ax.scatter(
            [p.latency_ms],
            [y],
            s=70,
            color=RANGE_P50,
            edgecolors=SURFACE,
            linewidths=1.4,
            zorder=5,
        )

    ax.set_yticks(list(ys))
    ax.set_yticklabels([p.name for p in points], fontsize=9, color=INK_2)
    ax.set_ylim(0.3, len(points) + 0.7)
    # Zero origin on purpose: rows never collide, and it keeps the row
    # lengths (tail sizes) honestly comparable.
    ax.set_xlim(0, max_p95 * 1.05)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)

    keys = LATENCY_METRICS[metric]
    _chrome(
        ax,
        xlabel=f"{keys['label']} (ms) (lower is better)",
        ylabel="",
        title=f"STT latency distribution — {language} {mode}",
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=SURFACE,
            label=label,
        )
        for color, label in [(RANGE_P50, "p50"), (RANGE_P95, "p95")]
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor=GRID,
        fontsize=10,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return output


def plot_stability(
    frame: pl.DataFrame,
    output: Path,
    *,
    language: str,
    mode: str = "stream",
) -> Path | None:
    """Render interim-update rate versus churn, one dot per provider.

    This is the barge-in chart. Churn alone flattens very different
    behaviours into one percentage — 40% of 0.7 updates/s is three retractions
    in a clip, 0% of 4 updates/s is a provider that revises constantly and
    never contradicts itself. Plotted together, the ideal corner is bottom
    right: frequent updates, none retracted.

    Args:
        frame: Summary frame from :func:`audio_harness.report.stt_summary_frame`.
        output: PNG path to write.
        language: BCP-47 tag selecting the rows.
        mode: Transport mode to plot.

    Returns:
        The written path, or ``None`` with fewer than two plottable providers.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        (row["provider"], row["interim_per_s"], row["churn_p50"] * 100)
        for row in frame.iter_rows(named=True)
        if row.get("mode") == mode
        and row.get("language") == language
        and row.get("interim_per_s") is not None
        and row.get("churn_p50") is not None
    ]
    if len(rows) < 2:
        return None

    fig, ax = plt.subplots(figsize=(12.5, 8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    ax.scatter(xs, ys, s=70, color=DOT, edgecolors=SURFACE, linewidths=1.2, zorder=5)
    for name, x, y in rows:
        ax.annotate(
            name,
            (x, y),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
            zorder=6,
        )

    ax.set_xlim(0, max(xs) * 1.15)
    ax.set_ylim(-4, max(max(ys) * 1.15, 10))
    _chrome(
        ax,
        xlabel="Interim hypotheses per second (higher = more responsive UI)",
        ylabel="Churn (% of interims that rewrote shown text, lower is better)",
        title=f"STT hypothesis stability — {language} {mode}",
    )
    ax.annotate(
        "ideal: frequent updates,\nnone retracted",
        xy=(max(xs) * 0.98, 0),
        ha="right",
        va="bottom",
        fontsize=9,
        color=BAND,
        fontstyle="italic",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return output


def plot_language_grid(
    frame: pl.DataFrame, output: Path, *, mode: str = "stream"
) -> Path | None:
    """Render error rate per provider grouped across languages.

    The chart for a multilingual run: it shows whether a provider is uniformly
    strong or carried by its best languages, which a per-language table makes
    hard to see at a glance. Bars are grouped by language and coloured per
    provider; WER and CER languages share the axis, so the caption names the
    metric per language rather than pretending they are one scale.

    Args:
        frame: Summary frame from :func:`audio_harness.report.stt_summary_frame`.
        output: PNG path to write.
        mode: Transport mode to plot.

    Returns:
        The written path, or ``None`` when fewer than two languages are present.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in frame.iter_rows(named=True)
        if row.get("mode") == mode and row.get("error_rate") is not None
    ]
    languages = sorted({row["language"] for row in rows})
    providers = sorted({row["provider"] for row in rows})
    if len(languages) < 2 or not providers:
        return None

    rates: dict[tuple[str, str], float] = {
        (row["provider"], row["language"]): row["error_rate"] * 100 for row in rows
    }
    metric_by_language = {row["language"]: row.get("metric", "WER") for row in rows}

    # One hue per provider, stepped around the wheel from the base dot color.
    cmap = plt.get_cmap("tab10")
    colors = {provider: cmap(i % 10) for i, provider in enumerate(providers)}

    fig, ax = plt.subplots(figsize=(1.9 * len(languages) + 4, 8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    width = 0.8 / len(providers)
    for index, provider in enumerate(providers):
        offsets = [
            lang_index + index * width - 0.4 + width / 2
            for lang_index in range(len(languages))
        ]
        values = [rates.get((provider, language)) for language in languages]
        ax.bar(
            [o for o, v in zip(offsets, values, strict=True) if v is not None],
            [v for v in values if v is not None],
            width=width * 0.92,
            color=colors[provider],
            edgecolor=SURFACE,
            linewidth=0.8,
            label=provider,
            zorder=3,
        )

    labels = [
        f"{language}\n({metric_by_language.get(language, 'WER')})"
        for language in languages
    ]
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels(labels, fontsize=9, color=INK_2)
    _chrome(
        ax,
        xlabel="",
        ylabel="Error rate (%) (lower is better)",
        title=f"STT error rate by language — {mode}",
    )
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(False, axis="x")
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor=GRID,
        fontsize=9,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return output


AMBER = "#c9822b"
"""Hue for the latency panel of the overview, kept apart from the data blue."""


@dataclass(slots=True, frozen=True)
class _Cell:
    """One provider/language cell of the overview matrix."""

    value: float | None
    display: str
    shade: float


def _overview_matrix(
    rows: list[dict[str, object]],
    providers: list[str],
    languages: list[str],
    *,
    value_key: str,
    per_language_shading: bool,
    fmt: str,
    scale: float = 1.0,
) -> list[list[_Cell]]:
    """Build the cell matrix for one overview panel.

    Shading is where the honesty lives. Accuracy shades within each language
    column, because a Japanese CER and a French WER share no scale — a cell's
    darkness may only rank providers against each other on the same corpus.
    Latency shades globally, because a second is a second in every language and
    cross-language comparison is exactly what that panel is for.

    Args:
        rows: Summary rows for one mode.
        providers: Row order.
        languages: Column order.
        value_key: Summary column to plot.
        per_language_shading: Normalize shading per column instead of globally.
        fmt: ``format()`` spec for the annotated value.
        scale: Multiplier applied before display, e.g. 100 for percentages.

    Returns:
        ``matrix[row][column]`` cells; missing lanes yield an em-dash cell.
    """
    lookup: dict[tuple[str, str], dict[str, object]] = {
        (str(row["provider"]), str(row["language"])): row for row in rows
    }

    def value_of(provider: str, language: str) -> float | None:
        row = lookup.get((provider, language))
        if row is None:
            return None
        value = row.get(value_key)
        return float(value) * scale if value is not None else None

    def shades(values: list[float | None]) -> list[float]:
        present = [v for v in values if v is not None]
        low, high = (min(present), max(present)) if present else (0.0, 0.0)
        span = high - low
        return [0.0 if v is None or span <= 0 else (v - low) / span for v in values]

    columns: list[list[float | None]] = [
        [value_of(p, language) for p in providers] for language in languages
    ]
    if per_language_shading:
        column_shades = [shades(column) for column in columns]
    else:
        flat = shades([v for column in columns for v in column])
        column_shades = [
            flat[i * len(providers) : (i + 1) * len(providers)]
            for i in range(len(languages))
        ]

    matrix: list[list[_Cell]] = []
    for row_index, provider in enumerate(providers):
        cells: list[_Cell] = []
        for col_index, language in enumerate(languages):
            value = columns[col_index][row_index]
            row = lookup.get((provider, language))
            failed = bool(row and row.get("failures"))
            display = (
                "—" if value is None else format(value, fmt) + ("*" if failed else "")
            )
            cells.append(
                _Cell(
                    value=value,
                    display=display,
                    shade=column_shades[col_index][row_index],
                )
            )
        matrix.append(cells)
    return matrix


def _draw_overview_panel(
    ax: object,
    matrix: list[list[_Cell]],
    *,
    hue: str,
    columns: list[str],
    providers: list[str] | None,
    title: str,
) -> None:
    """Paint one heatmap panel of the overview figure."""
    from matplotlib.patches import Rectangle

    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            if cell.value is None:
                face = _mix(GRID, SURFACE, 0.55)
                text_color = MUTED
            else:
                face = _mix(SURFACE, hue, 0.10 + 0.78 * cell.shade)
                text_color = SURFACE if cell.shade > 0.62 else INK
            ax.add_patch(  # type: ignore[attr-defined]
                Rectangle((x, y), 0.94, 0.88, facecolor=face, edgecolor=SURFACE, lw=1.5)
            )
            ax.text(  # type: ignore[attr-defined]
                x + 0.47,
                y + 0.44,
                cell.display,
                ha="center",
                va="center",
                fontsize=8.6,
                color=text_color,
            )

    ax.set_xlim(0, len(columns))  # type: ignore[attr-defined]
    ax.set_ylim(len(matrix), -0.6)  # type: ignore[attr-defined]
    ax.set_xticks([x + 0.47 for x in range(len(columns))])  # type: ignore[attr-defined]
    ax.set_xticklabels(columns, fontsize=8.6, color=INK_2)  # type: ignore[attr-defined]
    if providers is None:
        ax.set_yticks([])  # type: ignore[attr-defined]
    else:
        ax.set_yticks([y + 0.44 for y in range(len(providers))])  # type: ignore[attr-defined]
        ax.set_yticklabels(providers, fontsize=9.5, color=INK)  # type: ignore[attr-defined]
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)  # type: ignore[attr-defined]
    ax.tick_params(length=0)  # type: ignore[attr-defined]
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left")  # type: ignore[attr-defined]


def plot_overview(
    frame: pl.DataFrame, output: Path, *, mode: str = "stream"
) -> Path | None:
    """Render the whole multilingual run as one figure.

    Two aligned matrices share the provider rows: error rate on the left,
    finalize latency on the right. Rows are ordered by mean within-language
    accuracy rank, so the vertical order *is* the composite verdict, while the
    cells keep the per-language magnitudes that a composite hides.

    Args:
        frame: Summary frame from :func:`audio_harness.report.stt_summary_frame`.
        output: PNG path to write.
        mode: Transport mode to plot.

    Returns:
        The written path, or ``None`` with fewer than two languages.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in frame.iter_rows(named=True)
        if row.get("mode") == mode and row.get("error_rate") is not None
    ]
    languages = sorted({str(row["language"]) for row in rows})
    if len(languages) < 2:
        return None

    # Mean within-language rank orders the rows; missing lanes simply do not
    # contribute rather than counting for or against a provider.
    ranks: dict[str, list[int]] = {}
    for language in languages:
        ordered = sorted(
            (row for row in rows if row["language"] == language),
            key=lambda row: float(row["error_rate"]),  # type: ignore[arg-type]
        )
        for position, row in enumerate(ordered, start=1):
            ranks.setdefault(str(row["provider"]), []).append(position)
    providers = sorted(ranks, key=lambda p: sum(ranks[p]) / len(ranks[p]))

    metric_by_language = {
        str(row["language"]): str(row.get("metric", "WER")) for row in rows
    }
    accuracy = _overview_matrix(
        rows,
        providers,
        languages,
        value_key="error_rate",
        per_language_shading=True,
        fmt=".1f",
        scale=100.0,
    )
    latency = _overview_matrix(
        rows,
        providers,
        languages,
        value_key="finalize_p50_s",
        per_language_shading=False,
        fmt=".2f",
    )

    width = max(11.0, 1.02 * len(languages) * 2 + 3.4)
    height = 2.6 + 0.58 * len(providers)
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(width, height),
        dpi=150,
        sharey=False,
        gridspec_kw={"wspace": 0.04},
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)

    _draw_overview_panel(
        left,
        accuracy,
        hue=DOT,
        columns=[f"{lang}\n{metric_by_language[lang]}" for lang in languages],
        providers=providers,
        title="Error rate (%) — shaded within each language",
    )
    _draw_overview_panel(
        right,
        latency,
        hue=AMBER,
        columns=list(languages),
        providers=None,
        title="Finalize p50 (s) — one scale across languages",
    )

    fig.suptitle(
        f"STT multilingual overview — {mode}",
        fontsize=14,
        fontweight="bold",
        color=INK,
        x=0.125,
        ha="left",
    )
    fig.text(
        0.125,
        0.015,
        "rows ordered by mean within-language accuracy rank · "
        "darker = worse · * = lane had failures · — = not run",
        fontsize=8.6,
        color=MUTED,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return output


def render_all(
    frame: pl.DataFrame, output_dir: Path, *, metric: str = "finalize"
) -> list[Path]:
    """Render every chart the summary frame supports, one pair per language.

    Args:
        frame: Summary frame from :func:`audio_harness.report.stt_summary_frame`.
        output_dir: Directory receiving the PNGs.
        metric: Latency metric for both charts.

    Returns:
        The paths written, possibly empty when nothing is plottable.
    """
    if frame.is_empty() or "language" not in frame.columns:
        return []

    written: list[Path] = []
    for language in sorted(set(frame["language"].to_list())):
        rows = frame.filter(pl.col("language") == language)
        metric_label = (
            rows["metric"].to_list()[0] if "metric" in rows.columns else "WER"
        )
        slug = language.lower().replace("-", "_")
        pareto = plot_pareto(
            frame,
            output_dir / f"pareto_{slug}_{metric}.png",
            language=language,
            metric=metric,
            metric_label=metric_label,
        )
        if pareto is not None:
            written.append(pareto)
        latency = plot_latency_range(
            frame,
            output_dir / f"latency_{slug}_{metric}.png",
            language=language,
            metric=metric,
        )
        if latency is not None:
            written.append(latency)
        stability = plot_stability(
            frame, output_dir / f"stability_{slug}.png", language=language
        )
        if stability is not None:
            written.append(stability)

    grid = plot_language_grid(frame, output_dir / "error_by_language.png")
    if grid is not None:
        written.append(grid)
    overview = plot_overview(frame, output_dir / "overview.png")
    if overview is not None:
        written.append(overview)
    return written


def _mix(hex_color: str, other: str, t: float) -> tuple[float, float, float]:
    """Blend ``hex_color`` toward ``other`` by ``t`` in [0, 1]."""
    a = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i : i + 2], 16) for i in (1, 3, 5)]
    return tuple((av + (bv - av) * t) / 255 for av, bv in zip(a, b, strict=True))
