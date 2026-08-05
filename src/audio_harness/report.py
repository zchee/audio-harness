"""Turning raw runs into comparison tables.

Latency is reported as p50 and p95 rather than a mean. Speech APIs have long
right tails — a garbage-collection pause or a cold shard shows up as one slow
request — and a mean quietly blends that tail into the typical case, which is
the opposite of what capacity planning needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .config import STT_PRICING, TTS_PRICING
from .metrics import ProviderSummary, percentile, score_pair, summarize
from .types import SttResult, TtsResult

SECONDS_PER_HOUR = 3600.0
CHARS_PER_MILLION = 1_000_000.0


def _fmt(value: float | None, places: int = 3, suffix: str = "") -> str:
    """Render an optional float, or an em dash when it is missing."""
    return "—" if value is None else f"{value:.{places}f}{suffix}"


def _pct(value: float | None) -> str:
    """Render an optional rate as a percentage."""
    return "—" if value is None else f"{value * 100:.2f}%"


def stt_summary_frame(results: list[SttResult], language: str) -> pl.DataFrame:
    """Build a tidy table of per-provider STT metrics.

    Args:
        results: Every STT run.
        language: BCP-47 tag of the corpus, selecting WER versus CER.

    Returns:
        One row per provider and mode, sorted by accuracy then latency.
    """
    rows = [_stt_row(summary) for summary in summarize(results, language)]
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows)
    return frame.sort(["error_rate", "finalize_p50_s"], nulls_last=True)


def _stt_row(summary: ProviderSummary) -> dict[str, object]:
    """Flatten one provider summary into a table row."""
    pricing = STT_PRICING.get(summary.provider)
    rate = None
    if pricing is not None:
        rate = (
            pricing.stream_per_hour
            if summary.mode == "stream"
            else pricing.batch_per_hour
        )
    audio_hours = summary.audio_s / SECONDS_PER_HOUR

    return {
        "provider": summary.provider,
        "mode": summary.mode,
        "metric": summary.metric_name,
        "clips": summary.clips,
        "failures": summary.failures,
        "error_rate": summary.error_rate,
        "ttft_p50_s": percentile(summary.ttft_s, 50),
        "ttft_p95_s": percentile(summary.ttft_s, 95),
        "finalize_p50_s": percentile(summary.finalize_s, 50),
        "finalize_p95_s": percentile(summary.finalize_s, 95),
        "rtf_p50": percentile(summary.rtf, 50),
        "churn_p50": percentile(summary.instability, 50),
        "chunk_ms": summary.chunk_ms,
        "usd_per_hour": rate,
        "audio_hours": audio_hours,
        "est_usd": None if rate is None else rate * audio_hours,
    }


def tts_summary_frame(results: list[TtsResult], language: str) -> pl.DataFrame:
    """Build a tidy table of per-provider TTS metrics.

    Round-trip intelligibility is scored here rather than in the runner so a
    saved result file can be re-scored without re-synthesizing anything.

    Args:
        results: Every TTS run.
        language: BCP-47 tag driving round-trip normalization.

    Returns:
        One row per provider and mode, sorted by time to first byte.
    """
    grouped: dict[tuple[str, str], list[TtsResult]] = {}
    for result in results:
        grouped.setdefault((result.provider, str(result.mode)), []).append(result)

    rows = [
        _tts_row(provider, mode, runs, language)
        for (provider, mode), runs in grouped.items()
    ]
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["ttfb_p50_s"], nulls_last=True)


def _tts_row(
    provider: str, mode: str, runs: list[TtsResult], language: str
) -> dict[str, object]:
    """Flatten one provider's TTS runs into a table row."""
    ok = [run for run in runs if run.ok]
    ttfb = [run.ttfb_s for run in ok if run.ttfb_s is not None]
    rtf = [run.rtf for run in ok if run.rtf is not None]
    chars = sum(run.chars for run in ok)
    audio_s = sum(run.audio_s for run in ok)

    counts = None
    for run in ok:
        hypothesis = run.raw.get("roundtrip_text")
        reference = run.raw.get("text")
        if isinstance(hypothesis, str) and isinstance(reference, str) and reference:
            pair = score_pair(reference, hypothesis, language)
            counts = pair if counts is None else counts + pair

    pricing = TTS_PRICING.get(provider)
    est_usd: float | None = None
    per_million: float | None = None
    if pricing is not None:
        if pricing.per_million_chars is not None:
            per_million = pricing.per_million_chars
            est_usd = pricing.per_million_chars * chars / CHARS_PER_MILLION
        elif pricing.per_audio_minute is not None:
            est_usd = pricing.per_audio_minute * audio_s / 60.0

    return {
        "provider": provider,
        "mode": mode,
        "prompts": len(runs),
        "failures": len(runs) - len(ok),
        "ttfb_p50_s": percentile(ttfb, 50),
        "ttfb_p95_s": percentile(ttfb, 95),
        "rtf_p50": percentile(rtf, 50),
        "roundtrip_error_rate": None if counts is None else counts.rate,
        "chars": chars,
        "audio_s": audio_s,
        "usd_per_million_chars": per_million,
        "est_usd": est_usd,
    }


@dataclass(slots=True)
class Column:
    """One rendered markdown column.

    Attributes:
        header: Column heading.
        field: Key read from the data frame row.
        render: Formatter applied to the raw value.
    """

    header: str
    field: str
    render: object


_STT_COLUMNS = [
    Column("Provider", "provider", str),
    Column("Mode", "mode", str),
    Column("Metric", "metric", str),
    Column("Error rate", "error_rate", _pct),
    Column("TTFT p50", "ttft_p50_s", lambda v: _fmt(v, 3, "s")),
    Column("TTFT p95", "ttft_p95_s", lambda v: _fmt(v, 3, "s")),
    Column("Finalize p50", "finalize_p50_s", lambda v: _fmt(v, 3, "s")),
    Column("Finalize p95", "finalize_p95_s", lambda v: _fmt(v, 3, "s")),
    Column("RTF p50", "rtf_p50", lambda v: _fmt(v, 2, "x")),
    Column("Churn p50", "churn_p50", _pct),
    Column("Frame", "chunk_ms", lambda v: "—" if v is None else f"{v}ms"),
    Column("USD/hr", "usd_per_hour", lambda v: _fmt(v, 3)),
    Column("Est. USD", "est_usd", lambda v: _fmt(v, 4)),
    Column("Fail", "failures", str),
]

_TTS_COLUMNS = [
    Column("Provider", "provider", str),
    Column("Mode", "mode", str),
    Column("TTFB p50", "ttfb_p50_s", lambda v: _fmt(v, 3, "s")),
    Column("TTFB p95", "ttfb_p95_s", lambda v: _fmt(v, 3, "s")),
    Column("RTF p50", "rtf_p50", lambda v: _fmt(v, 2, "x")),
    Column("Round-trip err", "roundtrip_error_rate", _pct),
    Column("Chars", "chars", str),
    Column("USD/1M chars", "usd_per_million_chars", lambda v: _fmt(v, 2)),
    Column("Est. USD", "est_usd", lambda v: _fmt(v, 4)),
    Column("Fail", "failures", str),
]


def _markdown_table(frame: pl.DataFrame, columns: list[Column]) -> str:
    """Render a data frame as a GitHub-flavoured markdown table."""
    if frame.is_empty():
        return "_No results._"

    present = [column for column in columns if column.field in frame.columns]
    header = "| " + " | ".join(column.header for column in present) + " |"
    divider = "| " + " | ".join("---" for _ in present) + " |"
    lines = [header, divider]

    for row in frame.iter_rows(named=True):
        cells = []
        for column in present:
            render = column.render
            value = row[column.field]
            cells.append(render(value) if callable(render) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_stt_markdown(frame: pl.DataFrame) -> str:
    """Render the STT comparison table."""
    return _markdown_table(frame, _STT_COLUMNS)


def render_tts_markdown(frame: pl.DataFrame) -> str:
    """Render the TTS comparison table."""
    return _markdown_table(frame, _TTS_COLUMNS)


LEGEND = """
### How to read this

- **Error rate** — WER for space-delimited languages, CER for Japanese and
  other scriptio-continua languages. Corpus-level: total edits over total
  reference length, so long clips carry proportional weight.
- **TTFT** — first interim hypothesis, measured from the first audio byte.
  Governs how quickly a UI can show that it is listening.
- **Finalize** — last audio byte to final transcript. This is the number that
  sets turn-taking latency in a voice agent; optimize it before TTFT.
- **RTF** — processing seconds per audio second. Below 1.0x keeps up with live
  audio; above 1.0x falls behind and will drift on long sessions.
- **Churn** — share of interim hypotheses that rewrote already-shown text.
  High churn means visible flicker and retracted phrases.
- **Round-trip err** — synthesized audio transcribed by one fixed recognizer
  and scored against the prompt. An intelligibility proxy, not naturalness;
  only comparisons between rows are meaningful.
- **Est. USD** — list price times measured volume. Verify against the vendor's
  current pricing page before quoting it.
""".strip()
