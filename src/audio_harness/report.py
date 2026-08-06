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
from .entities import EntityClassScore
from .metrics import (
    ZERO_COUNTS,
    ErrorCounts,
    ProviderSummary,
    percentile,
    score_pair,
    summarize,
)
from .stt import family_of as stt_family
from .tts import family_of as tts_family
from .types import SttResult, TtsResult

SECONDS_PER_HOUR = 3600.0
CHARS_PER_MILLION = 1_000_000.0

JUDGE_DIVERGENCE_PTS = 0.02
"""Judges disagreeing by more than 2 WER points flag a lane for review:
that much spread means at least one judge is reacting to something other
than the voice's intelligibility."""


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
    summaries = summarize(results, language)
    entity_labels = sorted({label for s in summaries for label in s.entities})
    rows = [_stt_row(summary, entity_labels) for summary in summaries]
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows)
    # Language first: rows from different languages are not a ranking, so
    # sorting them together would invite reading one as beating another.
    return frame.sort(["language", "error_rate", "finalize_p50_s"], nulls_last=True)


def _entity_cell(score: EntityClassScore | None) -> str:
    """Render one class's entity-WER next to its exact-match rate.

    Both numbers appear because they diverge on purpose: a one-digit error
    in every ID is a low entity-WER and a 0% exact match.
    """
    if score is None or (score.error_rate is None and score.exact_match_rate is None):
        return "—"
    return f"{_pct(score.error_rate)} / EM {_pct(score.exact_match_rate)}"


def _stt_row(
    summary: ProviderSummary, entity_labels: list[str] | None = None
) -> dict[str, object]:
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

    row: dict[str, object] = {
        "provider": summary.provider,
        "mode": summary.mode,
        "language": summary.language,
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
        "interim_per_s": percentile(summary.interim_rate, 50),
        "chunk_ms": summary.chunk_ms,
        "usd_per_hour": rate,
        "audio_hours": audio_hours,
        "est_usd": None if rate is None else rate * audio_hours,
    }
    for label in entity_labels or ():
        row[f"ent[{label}]"] = _entity_cell(summary.entities.get(label))
    return row


def _roundtrip_entries(run: TtsResult) -> list[dict[str, object]]:
    """Return round-trip verdicts, reading both the list and legacy shapes.

    Runs recorded before the two-judge migration carry a scalar
    ``roundtrip_text``/``roundtrip_provider`` pair; folding them into the
    list form lets historical files render as one-judge lanes instead of
    silently losing their score.
    """
    entries = run.raw.get("roundtrip")
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    text = run.raw.get("roundtrip_text")
    if isinstance(text, str):
        provider = run.raw.get("roundtrip_provider")
        return [
            {
                "provider": provider if isinstance(provider, str) else "",
                "text": text,
                "error": None,
            }
        ]
    return []


def tts_summary_frame(results: list[TtsResult], language: str) -> pl.DataFrame:
    """Build a tidy table of per-provider TTS metrics.

    Round-trip intelligibility is scored here rather than in the runner so a
    saved result file can be re-scored without re-synthesizing anything.

    Args:
        results: Every TTS run.
        language: BCP-47 tag driving round-trip normalization.

    Returns:
        One row per provider and mode, sorted by time to first byte. Each
        round-trip judge gets its own ``rt[<judge>]`` column so the frame
        stays rectangular even when a judge failed on some lanes.
    """
    grouped: dict[tuple[str, str], list[TtsResult]] = {}
    for result in results:
        grouped.setdefault((result.provider, str(result.mode)), []).append(result)

    judges = sorted(
        {
            str(entry.get("provider"))
            for result in results
            for entry in _roundtrip_entries(result)
            if entry.get("provider")
        }
    )
    rows = [
        _tts_row(provider, mode, runs, language, judges)
        for (provider, mode), runs in grouped.items()
    ]
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["ttfb_p50_s"], nulls_last=True)


def _judge_counts(ok: list[TtsResult], language: str) -> dict[str, ErrorCounts]:
    """Accumulate corpus-level edit counts per round-trip judge."""
    per_judge: dict[str, ErrorCounts] = {}
    for run in ok:
        reference = run.raw.get("text")
        if not (isinstance(reference, str) and reference):
            continue
        for entry in _roundtrip_entries(run):
            judge = str(entry.get("provider") or "")
            hypothesis = entry.get("text")
            if not judge or entry.get("error") is not None:
                continue
            if not isinstance(hypothesis, str):
                continue
            pair = score_pair(reference, hypothesis, language)
            per_judge[judge] = per_judge.get(judge, ZERO_COUNTS) + pair
    return per_judge


def _judge_cell(counts: ErrorCounts | None, *, same_family: bool) -> str:
    """Render one judge's error rate; same-family scores are diagnostic only."""
    if counts is None or counts.reference_length == 0:
        return "—"
    rendered = f"{counts.rate * 100:.2f}%"
    return f"{rendered} †" if same_family else rendered


def _tts_row(
    provider: str,
    mode: str,
    runs: list[TtsResult],
    language: str,
    judges: list[str],
) -> dict[str, object]:
    """Flatten one provider's TTS runs into a table row.

    The ranked ``roundtrip_error_rate`` pools only judges from a different
    family than the candidate — a vendor's own recognizer decodes that
    vendor's voices best, so a same-family score would flatter the lane. The
    same-family score still appears in its judge column, marked diagnostic.
    """
    ok = [run for run in runs if run.ok]
    ttfb = [run.ttfb_s for run in ok if run.ttfb_s is not None]
    rtf = [run.rtf for run in ok if run.rtf is not None]
    chars = sum(run.chars for run in ok)
    audio_s = sum(run.audio_s for run in ok)

    per_judge = _judge_counts(ok, language)
    candidate_family = tts_family(provider)
    ranked: ErrorCounts | None = None
    for judge, judge_counts in per_judge.items():
        if stt_family(judge) != candidate_family:
            ranked = judge_counts if ranked is None else ranked + judge_counts

    rates = [
        judge_counts.rate
        for judge_counts in per_judge.values()
        if judge_counts.reference_length > 0
    ]
    diverged = len(rates) >= 2 and max(rates) - min(rates) > JUDGE_DIVERGENCE_PTS

    pricing = TTS_PRICING.get(provider)
    est_usd: float | None = None
    per_million: float | None = None
    if pricing is not None:
        if pricing.per_million_chars is not None:
            per_million = pricing.per_million_chars
            est_usd = pricing.per_million_chars * chars / CHARS_PER_MILLION
        elif pricing.per_audio_minute is not None:
            est_usd = pricing.per_audio_minute * audio_s / 60.0

    row: dict[str, object] = {
        "provider": provider,
        "mode": mode,
        "prompts": len(runs),
        "failures": len(runs) - len(ok),
        "ttfb_p50_s": percentile(ttfb, 50),
        "ttfb_p95_s": percentile(ttfb, 95),
        "rtf_p50": percentile(rtf, 50),
        "roundtrip_error_rate": (
            None if ranked is None or ranked.reference_length == 0 else ranked.rate
        ),
        "rt_divergence": diverged,
        "chars": chars,
        "audio_s": audio_s,
        "usd_per_million_chars": per_million,
        "est_usd": est_usd,
    }
    for judge in judges:
        row[f"rt[{judge}]"] = _judge_cell(
            per_judge.get(judge),
            same_family=stt_family(judge) == candidate_family,
        )
    return row


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
    Column("Lang", "language", str),
    Column("Metric", "metric", str),
    Column("Error rate", "error_rate", _pct),
    Column("TTFT p50", "ttft_p50_s", lambda v: _fmt(v, 3, "s")),
    Column("TTFT p95", "ttft_p95_s", lambda v: _fmt(v, 3, "s")),
    Column("Finalize p50", "finalize_p50_s", lambda v: _fmt(v, 3, "s")),
    Column("Finalize p95", "finalize_p95_s", lambda v: _fmt(v, 3, "s")),
    Column("RTF p50", "rtf_p50", lambda v: _fmt(v, 2, "x")),
    Column("Churn p50", "churn_p50", _pct),
    Column("Interim/s", "interim_per_s", lambda v: _fmt(v, 1)),
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
    Column("Judge Δ", "rt_divergence", lambda v: "⚠ >2pt" if v else "—"),
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
    """Render the STT comparison table.

    Entity columns are dynamic — their names depend on which classes the
    corpus annotates — so they are derived from the frame and inserted next
    to the overall error rate they break down.
    """
    columns = list(_STT_COLUMNS)
    entity_fields = sorted(field for field in frame.columns if field.startswith("ent["))
    if entity_fields:
        anchor = next(
            index
            for index, column in enumerate(columns)
            if column.field == "error_rate"
        )
        columns[anchor + 1 : anchor + 1] = [
            Column(f"Ent {field[4:-1]}", field, str) for field in entity_fields
        ]
    return _markdown_table(frame, columns)


def render_tts_markdown(frame: pl.DataFrame) -> str:
    """Render the TTS comparison table with one column per round-trip judge.

    Judge columns are dynamic — their names depend on the configured judges —
    so they are derived from the frame instead of the static column list and
    inserted next to the ranked round-trip score they explain.
    """
    columns = list(_TTS_COLUMNS)
    judge_fields = sorted(field for field in frame.columns if field.startswith("rt["))
    if judge_fields:
        anchor = next(
            index
            for index, column in enumerate(columns)
            if column.field == "roundtrip_error_rate"
        )
        columns[anchor + 1 : anchor + 1] = [
            Column(f"RT {field[3:-1]}", field, str) for field in judge_fields
        ]
    return _markdown_table(frame, columns)


LEGEND = """
### How to read this

- **Error rate** — WER for space-delimited languages, CER for Japanese and
  other scriptio-continua languages. Corpus-level: total edits over total
  reference length, so long clips carry proportional weight.
- **Ent \\<class\\>** — entity-WER and exact-match rate over reference spans
  tagged with that class (numbers, dates, currency, IDs, names). Read both:
  a one-digit error in every phone number is a low entity-WER and a 0% exact
  match, and the agent dials none of them. Only present when the corpus
  carries entity annotations.
- **TTFT** — first interim hypothesis, measured from the first audio byte.
  Governs how quickly a UI can show that it is listening.
- **Finalize** — last audio byte to final transcript. This is the number that
  sets turn-taking latency in a voice agent; optimize it before TTFT.
- **RTF** — processing seconds per audio second. Below 1.0x keeps up with live
  audio; above 1.0x falls behind and will drift on long sessions.
- **Churn** — share of interim hypotheses that rewrote already-shown text.
  High churn means visible flicker and retracted phrases.
- **Interim/s** — interim hypotheses per second of audio. Read churn against
  this, never alone: 40% churn over 0.7 updates/s is three rewrites out of
  seven, while 0% over 4.0 updates/s is a provider that revises constantly and
  never contradicts itself. The percentages are not comparable without it.
- **Round-trip err** — synthesized audio transcribed by fixed recognizers and
  scored against the prompt. An intelligibility proxy, not naturalness; only
  comparisons between rows are meaningful. The ranked figure pools only
  judges from a *different* vendor family than the row — a vendor's own
  recognizer decodes its own voices best. Per-judge columns show each judge
  separately; **†** marks a same-family score kept as a diagnostic, never
  ranked.
- **Judge Δ** — flags a row whose judges disagree by more than 2 WER points.
  That much spread means at least one judge is reacting to something other
  than the voice; read the per-judge columns before trusting the row.
- **Est. USD** — list price times measured volume. Verify against the vendor's
  current pricing page before quoting it.
""".strip()
