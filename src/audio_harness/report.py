"""Turning raw runs into comparison tables.

Latency is reported as p50 and p95 rather than a mean. Speech APIs have long
right tails — a garbage-collection pause or a cold shard shows up as one slow
request — and a mean quietly blends that tail into the typical case, which is
the opposite of what capacity planning needs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

from .audio import (
    PauseStats,
    measure_pauses,
    pcm16_to_float,
    read_audio_samples,
    wav_data_offset,
)
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


def _stt_row(summary: ProviderSummary, entity_labels: list[str] | None = None) -> dict[str, object]:
    """Flatten one provider summary into a table row."""
    pricing = STT_PRICING.get(summary.provider)
    rate = None
    if pricing is not None:
        rate = pricing.stream_per_hour if summary.mode == "stream" else pricing.batch_per_hour
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
        "unverified": summary.unverified,
        "license": ", ".join(sorted(summary.licenses)),
    }
    for label in entity_labels or ():
        score = summary.entities.get(label)
        row[f"ent[{label}]"] = _entity_cell(score)
        # Numeric twin of the display cell, for charts: strings cannot shade
        # a heatmap, and re-parsing "12.00% / EM 50.00%" would be absurd.
        row[f"ent_err[{label}]"] = None if score is None else score.error_rate
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
        One row per provider, mode and load factor, sorted by time to first
        byte. Load-pass runs (several syntheses in flight) form their own
        ``stream xN`` rows so queueing under load never blends into the
        sequential percentiles. Each round-trip judge gets its own
        ``rt[<judge>]`` column so the frame stays rectangular even when a
        judge failed on some lanes.
    """
    grouped: dict[tuple[str, str, int], list[TtsResult]] = {}
    for result in results:
        key = (result.provider, str(result.mode), _load_of(result))
        grouped.setdefault(key, []).append(result)

    judges = sorted({
        str(entry.get("provider"))
        for result in results
        for entry in _roundtrip_entries(result)
        if entry.get("provider")
    })
    rows = [_tts_row(provider, mode, load, runs, language, judges) for (provider, mode, load), runs in grouped.items()]
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["ttfb_p50_s"], nulls_last=True)


def _load_of(result: TtsResult) -> int:
    """Concurrent syntheses in flight when this run was made; 1 = sequential."""
    load = result.raw.get("load")
    if isinstance(load, int) and load > 1:
        return load
    return 1


def _ranked_counts(per_judge: dict[str, ErrorCounts], provider: str) -> ErrorCounts | None:
    """Pool the judges from outside the candidate's family into one score.

    A vendor's own recognizer decodes that vendor's voices best, so only
    cross-family judges may rank a lane; same-family scores stay diagnostic.
    """
    candidate_family = tts_family(provider)
    ranked: ErrorCounts | None = None
    for judge, judge_counts in per_judge.items():
        if stt_family(judge) != candidate_family:
            ranked = judge_counts if ranked is None else ranked + judge_counts
    return ranked


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
    load: int,
    runs: list[TtsResult],
    language: str,
    judges: list[str],
) -> dict[str, object]:
    """Flatten one provider's TTS runs into a table row.

    Latency percentiles come from warm runs only: the recorded cold runs
    carry connection-establishment cost by design and get their own column
    instead of dragging the warm tail. Cost columns count every successful
    run — cold calls billed too.

    The ranked ``roundtrip_error_rate`` pools only judges from a different
    family than the candidate — a vendor's own recognizer decodes that
    vendor's voices best, so a same-family score would flatter the lane. The
    same-family score still appears in its judge column, marked diagnostic.
    """
    ok = [run for run in runs if run.ok]
    warm = [run for run in ok if not run.cold]
    ttfb = [run.ttfb_s for run in warm if run.ttfb_s is not None]
    ttfa = [run.ttfa_s for run in warm if run.ttfa_s is not None]
    gaps = [run.gap_p99_s for run in warm if run.gap_p99_s is not None]
    ttfb_cold = [run.ttfb_s for run in ok if run.cold and run.ttfb_s is not None]
    rtf = [run.rtf for run in warm if run.rtf is not None]
    chars = sum(run.chars for run in ok)
    audio_s = sum(run.audio_s for run in ok)

    per_judge = _judge_counts(warm, language)
    candidate_family = tts_family(provider)
    ranked = _ranked_counts(per_judge, provider)

    rates = [judge_counts.rate for judge_counts in per_judge.values() if judge_counts.reference_length > 0]
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
        "mode": mode if load == 1 else f"{mode} x{load}",
        "prompts": len(runs),
        "failures": len(runs) - len(ok),
        "ttfb_p50_s": percentile(ttfb, 50),
        "ttfb_p95_s": percentile(ttfb, 95),
        "ttfa_p50_s": percentile(ttfa, 50),
        "gap_p99_s": percentile(gaps, 50),
        "ttfb_cold_s": percentile(ttfb_cold, 50),
        "rtf_p50": percentile(rtf, 50),
        "roundtrip_error_rate": (None if ranked is None or ranked.reference_length == 0 else ranked.rate),
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


def tts_mode_delta_frame(results: list[TtsResult], language: str) -> pl.DataFrame:
    """Diff batch against stream lanes of one provider on identical prompts.

    Streaming synthesis buys latency with a smaller lookahead, and whatever
    that costs — mispronunciations, drifted pacing, seams at chunk boundaries
    — never shows up in a latency table. The only fair comparison is the same
    provider speaking the same prompts through both transports, so rows
    appear only for providers with successful warm sequential runs in both
    modes, restricted to the prompt ids the two modes share.

    Args:
        results: Every TTS run.
        language: BCP-47 tag driving round-trip normalization.

    Returns:
        One row per provider: ranked cross-family round-trip error, audio
        duration p50 and internal-pause profile p50 per mode, each with its
        stream-minus-batch delta. Empty when no provider ran both modes.
    """
    by_provider: dict[str, dict[str, list[TtsResult]]] = {}
    for result in results:
        if not result.ok or result.cold or _load_of(result) > 1:
            continue
        by_provider.setdefault(result.provider, {}).setdefault(str(result.mode), []).append(result)

    rows: list[dict[str, object]] = []
    for provider, modes in sorted(by_provider.items()):
        batch = modes.get("batch")
        stream = modes.get("stream")
        if not batch or not stream:
            continue
        shared = {run.prompt_id for run in batch} & {run.prompt_id for run in stream}
        if not shared:
            continue
        rows.append(
            _mode_delta_row(
                provider,
                [run for run in batch if run.prompt_id in shared],
                [run for run in stream if run.prompt_id in shared],
                language,
            )
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("provider")


def _mode_delta_row(
    provider: str,
    batch: list[TtsResult],
    stream: list[TtsResult],
    language: str,
) -> dict[str, object]:
    """Build one provider's batch-versus-stream comparison row."""
    rt_batch = _ranked_rate(batch, provider, language)
    rt_stream = _ranked_rate(stream, provider, language)
    dur_batch = percentile([run.audio_s for run in batch if run.audio_s > 0], 50)
    dur_stream = percentile([run.audio_s for run in stream if run.audio_s > 0], 50)
    pause_batch = _pause_percentiles(batch)
    pause_stream = _pause_percentiles(stream)

    return {
        "provider": provider,
        "prompts": len({run.prompt_id for run in batch}),
        "rt_batch": rt_batch,
        "rt_stream": rt_stream,
        "rt_delta": _delta(rt_batch, rt_stream),
        "dur_batch_s": dur_batch,
        "dur_stream_s": dur_stream,
        "dur_delta_s": _delta(dur_batch, dur_stream),
        "pause_batch_s": pause_batch[0],
        "pause_stream_s": pause_stream[0],
        "pause_delta_s": _delta(pause_batch[0], pause_stream[0]),
        "longest_pause_delta_s": _delta(pause_batch[1], pause_stream[1]),
    }


def _delta(batch: float | None, stream: float | None) -> float | None:
    """Stream-minus-batch difference, or ``None`` when either side is missing."""
    if batch is None or stream is None:
        return None
    return stream - batch


def _ranked_rate(runs: list[TtsResult], provider: str, language: str) -> float | None:
    """Cross-family round-trip error rate over one lane's runs."""
    ranked = _ranked_counts(_judge_counts(runs, language), provider)
    if ranked is None or ranked.reference_length == 0:
        return None
    return ranked.rate


def _pause_percentiles(runs: list[TtsResult]) -> tuple[float | None, float | None]:
    """P50 of total and longest internal pause across one lane's runs."""
    profiles = [profile for profile in map(_pause_profile, runs) if profile is not None]
    return (
        percentile([profile.total_s for profile in profiles], 50),
        percentile([profile.longest_s for profile in profiles], 50),
    )


def _pause_profile(result: TtsResult) -> PauseStats | None:
    """Internal pauses of one run's audio, from memory or the saved WAV.

    Freshly-run results still hold their PCM; results reloaded from JSONL
    dropped the bytes but may carry an ``audio_path`` written by the runner.
    Either source works, so saved runs stay re-scorable — a run persisted
    without audio simply contributes no pause figures.
    """
    if result.audio and result.encoding.startswith("pcm"):
        header = wav_data_offset(result.audio)
        samples = pcm16_to_float(result.audio[header:])
        return measure_pauses(samples, result.sample_rate)

    path = result.raw.get("audio_path")
    if isinstance(path, str) and path:
        decoded = read_audio_samples(path)
        if decoded is not None:
            samples, rate = decoded
            return measure_pauses(samples, rate)
    return None


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
    render: Callable[[Any], str]


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
    Column("Unverified", "unverified", lambda v: str(v) if v else "—"),
    Column("License", "license", lambda v: v or "—"),
    Column("Fail", "failures", str),
]

_TTS_COLUMNS = [
    Column("Provider", "provider", str),
    Column("Mode", "mode", str),
    Column("TTFB p50", "ttfb_p50_s", lambda v: _fmt(v, 3, "s")),
    Column("TTFB p95", "ttfb_p95_s", lambda v: _fmt(v, 3, "s")),
    Column("TTFA p50", "ttfa_p50_s", lambda v: _fmt(v, 3, "s")),
    Column("Gap p99", "gap_p99_s", lambda v: _fmt(v, 3, "s")),
    Column("TTFB cold", "ttfb_cold_s", lambda v: _fmt(v, 3, "s")),
    Column("RTF p50", "rtf_p50", lambda v: _fmt(v, 2, "x")),
    Column("Round-trip err", "roundtrip_error_rate", _pct),
    Column("Judge Δ", "rt_divergence", lambda v: "⚠ >2pt" if v else "—"),
    Column("Chars", "chars", str),
    Column("USD/1M chars", "usd_per_million_chars", lambda v: _fmt(v, 2)),
    Column("Est. USD", "est_usd", lambda v: _fmt(v, 4)),
    Column("Fail", "failures", str),
]


def _signed(value: float | None, places: int = 3, suffix: str = "") -> str:
    """Render an optional delta with an explicit sign, or an em dash."""
    return "—" if value is None else f"{value:+.{places}f}{suffix}"


def _signed_pts(value: float | None) -> str:
    """Render an optional rate delta in signed percentage points."""
    return "—" if value is None else f"{value * 100:+.2f}pt"


_TTS_DELTA_COLUMNS = [
    Column("Provider", "provider", str),
    Column("Prompts", "prompts", str),
    Column("RT err batch", "rt_batch", _pct),
    Column("RT err stream", "rt_stream", _pct),
    Column("Δ RT", "rt_delta", _signed_pts),
    Column("Dur p50 batch", "dur_batch_s", lambda v: _fmt(v, 2, "s")),
    Column("Dur p50 stream", "dur_stream_s", lambda v: _fmt(v, 2, "s")),
    Column("Δ Dur", "dur_delta_s", lambda v: _signed(v, 2, "s")),
    Column("Pause p50 batch", "pause_batch_s", lambda v: _fmt(v, 2, "s")),
    Column("Pause p50 stream", "pause_stream_s", lambda v: _fmt(v, 2, "s")),
    Column("Δ Pause", "pause_delta_s", lambda v: _signed(v, 2, "s")),
    Column("Δ Longest pause", "longest_pause_delta_s", lambda v: _signed(v, 2, "s")),
]


def render_tts_mode_delta_markdown(frame: pl.DataFrame) -> str:
    """Render the batch-versus-stream degradation table."""
    return _markdown_table(frame, _TTS_DELTA_COLUMNS)


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
        anchor = next(index for index, column in enumerate(columns) if column.field == "error_rate")
        columns[anchor + 1 : anchor + 1] = [Column(f"Ent {field[4:-1]}", field, str) for field in entity_fields]
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
        anchor = next(index for index, column in enumerate(columns) if column.field == "roundtrip_error_rate")
        columns[anchor + 1 : anchor + 1] = [Column(f"RT {field[3:-1]}", field, str) for field in judge_fields]
    return _markdown_table(frame, columns)


LEGEND = """
### How to read this

- **Error rate** — WER for space-delimited languages, CER for Japanese and
  other scriptio-continua languages. Corpus-level: total edits over total
  reference length, so long clips carry proportional weight.
- **Unverified / License** — curated corpora (YODAS/Granary) carry a source
  license and a gold status on every clip. Clips whose reference is an
  unverified subtitle are **excluded from Error rate** — ranking against
  caption quality would measure the captioner — but still count for TTFT,
  Finalize, RTF and Churn, which need no transcript truth. The Unverified
  column says how many clips were held out; a row that is entirely
  unverified shows no error rate at all.
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
- **TTFA** — time to first *audible* audio: the RMS onset of the decoded
  waveform translated to wall time through per-chunk arrivals, modeling a
  client that plays from the first byte. TTFB credits a vendor for container
  bytes and leading silence; TTFA credits neither. Compare the two columns —
  a large spread is padding, not speed.
- **Gap p99** — 99th-percentile gap between successive audio chunks within a
  run (p50 across runs). Stutter: a real-time client stalls whenever a gap
  outruns its playback buffer.
- **TTFB cold** — first-request latency on a cold connection stack (DNS,
  TLS, session setup), from the recorded warmup pass. Warm columns exclude
  these runs. An agent's first utterance in a call pays this price.
- **Mode "stream xN"** — the optional load pass: N syntheses in flight
  against one adapter. Compare against the plain stream row to see queueing
  under concurrency; the rows never mix.
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
- **Batch vs stream (Δ columns)** — the same provider speaking the same
  prompts through both transports; every Δ is stream minus batch, so positive
  means streaming degraded it. Round-trip error measures intelligibility
  loss, duration catches pacing drift, and the pause columns profile silence
  *inside* the speech — chunk-boundary seams that batch synthesis of the same
  text does not produce. Cold and load-pass runs are excluded from both
  sides.
- **SDK-wrapped adapters** — Google (Chirp 3) and Azure (Speech STT / Neural
  TTS) go through the vendor's official SDK instead of a raw WebSocket, since
  both vendors' realtime wire protocols are unsupported or underdocumented
  for direct use. Every result from these adapters carries
  `raw["sdk_buffered"] = True`: the SDK owns its own buffering, batching and
  retry schedule on a native thread, so their TTFT/Finalize/TTFB/TTFA figures
  include that overhead and are not directly comparable to the
  WebSocket-native adapters' numbers.
- **Hosted-proxy adapters** — Every OpenRouter result carries
  `raw["hosted_proxy"] = True`. Same-model paired measurement (2026-08-12,
  n=10 alternating, gpt-transcribe direct-vs-proxied and
  gemini-3.1-flash-tts-preview direct-vs-proxied) put the median proxy
  overhead at roughly +0.1s — within run-to-run noise once model inference
  dominates — so OpenRouter *medians* are comparable to direct-vendor lanes.
  Tail latency is not: the proxied STT p90 ran about 1.1s worse than direct,
  so treat p90+ figures from OpenRouter lanes with caution.
- **Local-compute adapters** — Every apple-speech-stt result carries
  `raw["local_compute"] = True`: latency reflects this machine's silicon
  (Apple Neural Engine / CPU), not a hosted vendor service, so its figures are
  not directly comparable to the network lanes above.
""".strip()
