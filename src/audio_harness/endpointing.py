"""End-of-utterance endpointing metrics and calibration.

A voice agent lives or dies on one decision: has the user finished talking?
Vendors expose that decision as an explicit end-of-utterance (EOU) event —
never as a bare segment final, which is a decoding boundary. This module
scores ONLY ``EventKind.EOU`` events (the Critic-mandated scope): counting
segment finals as cutoffs would rank vendors by their transcript chunking,
which is the exact confound the event model exists to remove.

Two fairness instruments publish alongside the curves, because an event
timestamp bundles the vendor's decision with the harness' own machinery:

* the loopback floor — how late this client stack observes an event a server
  sent instantly, dominated by chunk pacing quantization; and
* the per-session WebSocket RTT measured by :func:`stt.ws.run_stream`.

Neither is subtracted from any number; both are shown so a millisecond-level
gap between vendors can be read honestly.
"""

from __future__ import annotations

import contextlib
import statistics
from dataclasses import dataclass, field
from typing import Any

import orjson
from websockets.asyncio.server import ServerConnection, serve

from .metrics import percentile
from .types import AudioClip, EventKind, SttResult

LATENCY_BUDGETS_S = (0.3, 0.6)
"""Endpoint-latency budgets the report scores against, in seconds."""


@dataclass(slots=True)
class EndpointSummary:
    """Endpointing behaviour of one provider in one language.

    Attributes:
        provider: Registry key of the adapter.
        mode: Transport mode the runs used.
        language: BCP-47 tag these runs were scored under.
        eou_source: The native signal the adapter captured (for example
            ``speech_final+utterance_end`` or ``end_of_turn``), or ``None``
            when the lane exposed no end-of-utterance decision — such lanes
            are reported descriptively and never ranked.
        endpoint_config: Effective endpointing knobs the lane ran with. A
            ranking is only comparable when each lane's configuration is
            visible next to it.
        clips: Number of clips attempted.
        failures: Number of clips that errored.
        hold_pauses: Labeled mid-turn pauses across scored clips.
        cut_pauses: Hold pauses containing at least one EOU event — the
            false cutoffs. The agent would have interrupted the speaker here.
        premature_eous: EOU events before the true end of the turn,
            wherever they fell. A superset diagnostic of ``cut_pauses``.
        eou_latency_s: Per-clip latency from true speech end to the first
            EOU at or after it. Only measurable under real-time pacing.
        finals_in_pauses: Segment finals inside hold pauses — descriptive
            context for lanes without an EOU signal, never a ranked number.
        ws_rtt_s: Per-session WebSocket round-trip times.
    """

    provider: str
    mode: str
    language: str
    eou_source: str | None = None
    endpoint_config: dict[str, Any] = field(default_factory=dict)
    clips: int = 0
    failures: int = 0
    hold_pauses: int = 0
    cut_pauses: int = 0
    premature_eous: int = 0
    eou_latency_s: list[float] = field(default_factory=list)
    finals_in_pauses: int = 0
    ws_rtt_s: list[float] = field(default_factory=list)

    @property
    def ranked(self) -> bool:
        """Whether this lane may appear in the ranked comparison.

        Only vendors whose native end-of-utterance signal was captured are
        ranked; observable final-event behaviour of the rest is reported
        descriptively (plan P2.10 scope rule).
        """
        return self.eou_source is not None

    @property
    def false_cutoff_rate(self) -> float | None:
        """Fraction of labeled mid-turn pauses the vendor endpointed in."""
        if self.hold_pauses == 0:
            return None
        return self.cut_pauses / self.hold_pauses

    def latency_p(self, q: float) -> float | None:
        """Endpoint-latency percentile in seconds."""
        return percentile(self.eou_latency_s, q)

    def within_budget(self, budget_s: float) -> float | None:
        """Fraction of measured endpoint latencies within ``budget_s``."""
        if not self.eou_latency_s:
            return None
        return sum(1 for v in self.eou_latency_s if v <= budget_s) / len(
            self.eou_latency_s
        )

    @property
    def rtt_p50_s(self) -> float | None:
        """Median per-session WebSocket round trip."""
        return percentile(self.ws_rtt_s, 50)


def summarize_endpointing(
    results: list[SttResult], language: str
) -> list[EndpointSummary]:
    """Aggregate endpointing behaviour per provider, mode and language.

    Designed to run over saved results JSONL: pause labels, speech end, EOU
    kinds, RTT and effective knob configuration all survive
    :func:`runner.write_stt_results`, so thresholds can be revisited without
    re-paying for the audio.

    Args:
        results: Per-clip streaming results, typically from the endpointing
            lane (clips carrying labeled pauses).
        language: Fallback BCP-47 tag for results that recorded none.

    Returns:
        Summaries keyed by provider, mode and language.
    """
    summaries: dict[tuple[str, str, str], EndpointSummary] = {}

    for result in results:
        recorded = result.raw.get("language")
        clip_language = recorded if isinstance(recorded, str) and recorded else language
        key = (result.provider, str(result.mode), clip_language)
        summary = summaries.setdefault(
            key,
            EndpointSummary(
                provider=result.provider,
                mode=str(result.mode),
                language=clip_language,
            ),
        )
        summary.clips += 1

        source = result.raw.get("eou_source")
        if isinstance(source, str) and source:
            summary.eou_source = source
        config = result.raw.get("endpoint_config")
        if isinstance(config, dict) and config:
            summary.endpoint_config = config

        if not result.ok:
            summary.failures += 1
            continue

        rtt = result.raw.get("ws_rtt_s")
        if isinstance(rtt, int | float):
            summary.ws_rtt_s.append(float(rtt))

        pauses = _pauses_of(result)
        eou_times = [p.t_s for p in result.partials if p.kind == EventKind.EOU]
        final_times = [
            p.t_s for p in result.partials if p.is_final and p.kind != EventKind.EOU
        ]

        summary.hold_pauses += len(pauses)
        for start, end in pauses:
            if any(start <= t < end for t in eou_times):
                summary.cut_pauses += 1
            if any(start <= t < end for t in final_times):
                summary.finals_in_pauses += 1

        speech_end = result.raw.get("speech_end_s")
        if isinstance(speech_end, int | float) and speech_end > 0:
            summary.premature_eous += sum(1 for t in eou_times if t < speech_end)
            after = [t for t in eou_times if t >= speech_end]
            if after:
                summary.eou_latency_s.append(after[0] - float(speech_end))

    return list(summaries.values())


def _pauses_of(result: SttResult) -> list[tuple[float, float]]:
    """Read the labeled mid-turn pause spans a result carries."""
    raw = result.raw.get("pauses")
    if not isinstance(raw, list):
        return []
    spans: list[tuple[float, float]] = []
    for span in raw:
        if isinstance(span, list | tuple) and len(span) == 2:
            spans.append((float(span[0]), float(span[1])))
    return spans


async def measure_loopback_floor(
    chunk_ms: int, *, clip_s: float = 0.5, rounds: int = 3
) -> float:
    """Measure the client stack's own event-latency floor for one chunk size.

    A local WebSocket server answers the end-of-input signal with a final
    transcript *immediately*, so everything measured here — chunk pacing
    quantization, event-loop scheduling, JSON decode, loopback sockets — is
    the harness' contribution, not any vendor's. Adapters with a larger
    ``min_chunk_ms`` (AssemblyAI's 50 ms floor) pay a higher floor, which is
    precisely why this number must be published next to their curves.

    Args:
        chunk_ms: Chunk size to pace audio at, matching the lane under
            calibration (``provider.effective_chunk_ms(run.chunk_ms)``).
        clip_s: Duration of the silent calibration clip.
        rounds: Measurement repetitions; the median is returned.

    Returns:
        Median seconds from end-of-audio to the echoed final's timestamp.
    """
    # Imported here to keep the module import-light for report-time use;
    # the driver pulls in the websocket client stack.
    from .stt.base import StreamTimeline
    from .stt.ws import run_stream

    clip = AudioClip(
        clip_id=f"loopback-{chunk_ms}ms",
        pcm=b"\x00\x00" * int(16000 * clip_s),
        sample_rate=16000,
        duration_s=clip_s,
        reference=None,
        language="en-US",
        source_path="<loopback>",
    )

    async def echo(socket: ServerConnection) -> None:
        try:
            async for frame in socket:
                if isinstance(frame, bytes):
                    continue
                with contextlib.suppress(orjson.JSONDecodeError):
                    if orjson.loads(frame).get("type") == "eos":
                        break
        except Exception:
            return
        with contextlib.suppress(Exception):
            await socket.send(
                orjson.dumps(
                    {"type": "transcript", "text": "floor", "final": True}
                ).decode()
            )
            await socket.send(orjson.dumps({"type": "done"}).decode())

    def handle(payload: Any, timeline: StreamTimeline) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("type") == "done":
            return True
        if payload.get("type") == "transcript":
            timeline.record(str(payload["text"]), is_final=bool(payload.get("final")))
        return False

    async def eos(socket: Any) -> None:
        await socket.send(orjson.dumps({"type": "eos"}).decode())

    floors: list[float] = []
    async with serve(echo, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        for _ in range(rounds):
            timeline = StreamTimeline()
            await run_stream(
                url=f"ws://127.0.0.1:{port}",
                headers={},
                clip=clip,
                chunk_ms=chunk_ms,
                realtime=True,
                timeline=timeline,
                handle_message=handle,
                on_input_done=eos,
                finalize_timeout_s=5.0,
            )
            if timeline.finalize_s is not None:
                floors.append(timeline.finalize_s)

    if not floors:
        raise RuntimeError("loopback calibration produced no measurements")
    return statistics.median(floors)


def render_endpointing_markdown(
    summaries: list[EndpointSummary],
    *,
    floors: dict[str, float] | None = None,
) -> str:
    """Render the endpointing bench as Markdown tables.

    Ranked and descriptive lanes are separate tables by design: a vendor
    without a captured end-of-utterance signal has nothing comparable to
    rank, and folding its final-event behaviour into the ranked table would
    resurrect the segment-final confound.

    Args:
        summaries: Output of :func:`summarize_endpointing`.
        floors: Optional per-provider client-stack floor seconds from
            :func:`measure_loopback_floor`.

    Returns:
        Markdown with a ranked EOU table, a descriptive table when any
        non-EOU lane exists, and the reading notes.
    """
    floors = floors or {}
    ranked = sorted(
        (s for s in summaries if s.ranked),
        key=lambda s: (s.language, s.false_cutoff_rate or 0.0),
    )
    descriptive = sorted(
        (s for s in summaries if not s.ranked),
        key=lambda s: (s.language, s.provider),
    )

    lines: list[str] = ["## Endpointing (EOU-capable vendors, ranked)", ""]
    if ranked:
        lines += [
            "| Provider | Lang | Clips | EOU source | False cutoff | "
            "Premature | EOU p50 | p90 | p99 | <=300ms | <=600ms | "
            "RTT p50 | Floor | Config |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s in ranked:
            lines.append(
                f"| {s.provider} | {s.language} | {s.clips} | {s.eou_source} "
                f"| {_pct(s.false_cutoff_rate)} | {s.premature_eous} "
                f"| {_sec(s.latency_p(50))} | {_sec(s.latency_p(90))} "
                f"| {_sec(s.latency_p(99))} "
                f"| {_pct(s.within_budget(LATENCY_BUDGETS_S[0]))} "
                f"| {_pct(s.within_budget(LATENCY_BUDGETS_S[1]))} "
                f"| {_sec(s.rtt_p50_s)} | {_sec(floors.get(s.provider))} "
                f"| {_config(s.endpoint_config)} |"
            )
    else:
        lines.append("_No lane captured a native end-of-utterance signal._")

    if descriptive:
        lines += [
            "",
            "## Final-event behaviour (no captured EOU signal — not ranked)",
            "",
            "| Provider | Lang | Clips | Finals in pauses | RTT p50 | Floor |",
            "|---|---|---|---|---|---|",
        ]
        for s in descriptive:
            lines.append(
                f"| {s.provider} | {s.language} | {s.clips} "
                f"| {s.finals_in_pauses} | {_sec(s.rtt_p50_s)} "
                f"| {_sec(floors.get(s.provider))} |"
            )

    lines += [
        "",
        "- **False cutoff** — share of labeled mid-turn pauses containing a "
        "vendor end-of-utterance decision; the agent would have interrupted.",
        "- **EOU latency** — true speech end to the vendor's first EOU at or "
        "after it; only measured under real-time pacing.",
        "- **Floor / RTT** — the harness' own contribution (chunk pacing, "
        "loopback scheduling) and the network path. Published, not "
        "subtracted: read vendor gaps against them.",
        "- Segment finals never count as cutoffs; they are decoding "
        "boundaries, not turn-taking decisions.",
    ]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    """Format a rate as a percentage cell."""
    return "—" if value is None else f"{value * 100:.1f}%"


def _sec(value: float | None) -> str:
    """Format seconds as a milliseconds cell."""
    return "—" if value is None else f"{value * 1000:.0f}ms"


def _config(config: dict[str, Any]) -> str:
    """Format the effective endpointing knobs compactly."""
    if not config:
        return "defaults"
    return ", ".join(f"{k}={v}" for k, v in sorted(config.items()) if v is not None)
