"""Soniox stt-rt-v5 realtime speech-to-text adapter."""

from __future__ import annotations

from typing import Any

import orjson
from websockets.asyncio.client import ClientConnection

from ..config import require_env
from ..types import AudioClip, EventKind, Mode, SttResult
from .base import StreamTimeline, SttProvider, register
from .ws import StreamProtocolError, run_stream

STREAM_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


@register
class SonioxRealtimeV5(SttProvider):
    """Soniox realtime transcription over its token-streaming WebSocket.

    Soniox streams sub-word tokens rather than sentence hypotheses: tokens
    arrive with ``is_final`` flipping from false to true as context accrues.
    The adapter therefore maintains the running transcript itself and records
    the cumulative string, so downstream code sees the same shape it gets from
    sentence-oriented vendors.

    Options:
        model: Model identifier; defaults to ``stt-rt-v5``.
        language_hints: Language codes that bias recognition.
        enable_endpoint_detection: When true, Soniox finalizes each utterance
            as it ends and emits a ``<end>`` marker token. Off by default.
        endpoint_sensitivity: How readily the model emits an endpoint.
        max_endpoint_delay_ms: Ceiling on endpoint delay after speech ends.
        endpoint_latency_adjustment_level: Vendor latency/accuracy trade-off.
    """

    key = "soniox-rt-v5"
    vendor = "soniox"
    supports_stream = True

    async def transcribe_stream(
        self, clip: AudioClip, *, chunk_ms: int, realtime: bool
    ) -> SttResult:
        """Stream PCM and reassemble the token stream into a transcript."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        hints = self.options.get("language_hints") or [clip.language.split("-")[0]]
        knobs = {
            name: self.options[name]
            for name in (
                "enable_endpoint_detection",
                "endpoint_sensitivity",
                "max_endpoint_delay_ms",
                "endpoint_latency_adjustment_level",
            )
            if name in self.options
        }

        async def configure(socket: ClientConnection) -> None:
            await socket.send(
                orjson.dumps(
                    {
                        "api_key": require_env("SONIOX_API_KEY", self.key),
                        "model": str(self.options.get("model", "stt-rt-v5")),
                        "audio_format": "pcm_s16le",
                        "sample_rate": clip.sample_rate,
                        "num_channels": 1,
                        "language_hints": list(hints),
                        **knobs,
                    }
                ).decode()
            )

        async def end_of_audio(socket: ClientConnection) -> None:
            await socket.send("")

        accumulator = _TokenAccumulator()
        await run_stream(
            url=STREAM_URL,
            headers={},
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=accumulator,
            on_open=configure,
            on_input_done=end_of_audio,
        )

        result.text = timeline.last_final() or accumulator.finalized
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        if knobs.get("enable_endpoint_detection"):
            result.raw["eou_source"] = "end_token"
            result.raw["endpoint_config"] = knobs
        return result


class _TokenAccumulator:
    """Rebuilds a running transcript from Soniox's incremental token stream.

    Final tokens are appended permanently; non-final tokens form a provisional
    tail that is discarded on the next frame. Each recorded hypothesis is the
    full transcript so far, which is why the adapter reads it back with
    :meth:`StreamTimeline.last_final` rather than concatenating.
    """

    __slots__ = ("finalized",)

    def __init__(self) -> None:
        self.finalized = ""

    def __call__(self, payload: Any, timeline: StreamTimeline) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("error_code") or payload.get("error_message"):
            raise StreamProtocolError(
                f"soniox: {payload.get('error_code')}: {payload.get('error_message')}"
            )

        tail = ""
        endpoints = 0
        for token in payload.get("tokens", []):
            text = str(token.get("text", ""))
            # Marker tokens are control flow, not transcript: <end> is the
            # endpoint decision (enable_endpoint_detection), <fin> confirms a
            # manual finalize. Splicing either into the text would corrupt
            # every accuracy metric downstream.
            if text == "<end>":
                endpoints += 1
                continue
            if text == "<fin>":
                continue
            if token.get("is_final"):
                self.finalized += text
            else:
                tail += text

        if self.finalized or tail:
            timeline.record((self.finalized + tail).strip(), is_final=not tail)
        for _ in range(endpoints):
            timeline.record("", is_final=False, kind=EventKind.EOU)

        return bool(payload.get("finished"))
