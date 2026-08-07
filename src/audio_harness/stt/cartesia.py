"""Cartesia Ink-2 speech-to-text adapter."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import orjson
from websockets.asyncio.client import ClientConnection

from audio_harness.config import require_env
from audio_harness.types import AudioClip, EventKind, Mode, SttResult

from .base import StreamTimeline, SttProvider, register
from .ws import StreamProtocolError, run_stream


STREAM_URL = "wss://api.cartesia.ai/stt/turns/websocket"
DEFAULT_VERSION = "2026-03-01"


@register
class CartesiaInk2(SttProvider):
    """Cartesia Ink-2 over the turn-based streaming WebSocket.

    This adapter deliberately uses ``/stt/turns/websocket`` rather than the
    plain streaming surface: its turn events are structurally comparable to
    Deepgram Flux's model-native turn decisions in the endpointing benchmark.
    Ink-2 accepts English audio only; ``en`` and ``en-*`` BCP-47 tags are
    accepted by this adapter.

    Options:
        turn_start_threshold: Turn-start confidence threshold from 0.5 to
            0.9; the server default is 0.8.
        turn_eager_end_threshold: Eager turn-end confidence threshold from
            0.3 to 0.6; the server default is 0.4.
        turn_end_threshold: Definitive turn-end confidence threshold from
            0.05 to 0.5; the server default is 0.2.
        turn_end_timeout_ms: Turn-end timeout from 640 to 11200 ms; the
            server default is 5600 ms.
        keyterm: A list of terms to bias. Each item is sent as a separate
            repeatable ``keyterm`` query parameter.

    End of input is the text JSON command ``{"type":"close"}``. Cartesia
    processes buffered audio into normal turn events and acknowledges the
    command by completing the WebSocket close handshake; it does not emit a
    separate ``done`` event on this API surface (verified 2026-08-07).
    """

    key = "cartesia-ink2"
    vendor = "cartesia"
    supports_stream = True

    def _auth(self) -> dict[str, str]:
        return {"X-API-Key": require_env("CARTESIA_API_KEY", self.key)}

    def _endpoint_options(self) -> dict[str, float | int]:
        ranges: dict[str, tuple[float, float]] = {
            "turn_start_threshold": (0.5, 0.9),
            "turn_eager_end_threshold": (0.3, 0.6),
            "turn_end_threshold": (0.05, 0.5),
            "turn_end_timeout_ms": (640, 11200),
        }
        knobs: dict[str, float | int] = {}
        for name, (minimum, maximum) in ranges.items():
            if name not in self.options:
                continue
            value = self.options[name]
            if name == "turn_end_timeout_ms":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{self.key}: {name} must be an integer from 640 to 11200")
                validated: float | int = value
            else:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ValueError(f"{self.key}: {name} must be a number from {minimum} to {maximum}")
                validated = float(value)
            if not minimum <= validated <= maximum:
                raise ValueError(f"{self.key}: {name} must be between {minimum} and {maximum}; got {value}")
            knobs[name] = validated
        return knobs

    def _keyterms(self) -> list[str]:
        value = self.options.get("keyterm", [])
        if value is None:
            return []
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{self.key}: keyterm must be a list of strings")
        return value

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Stream English PCM and collect Cartesia's model-native turns."""
        language = clip.language.lower()
        if language != "en" and not language.startswith("en-"):
            raise ValueError(f"{self.key}: Cartesia Ink-2 supports English only; got {clip.language!r}")

        knobs = self._endpoint_options()
        keyterms = self._keyterms()
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        handler = _TurnHandler()
        params: dict[str, str | list[str]] = {
            "model": "ink-2",
            "encoding": "pcm_s16le",
            "sample_rate": str(clip.sample_rate),
            "cartesia_version": DEFAULT_VERSION,
        }
        params.update({name: str(value) for name, value in knobs.items()})
        if keyterms:
            params["keyterm"] = keyterms

        async def close_input(socket: ClientConnection) -> None:
            await socket.send(orjson.dumps({"type": "close"}).decode())

        await run_stream(
            url=f"{STREAM_URL}?{urlencode(params, doseq=True)}",
            headers=self._auth(),
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=handler,
            on_input_done=close_input,
        )

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        result.raw["eou_source"] = "turn.end"
        result.raw["eager_events"] = handler.eager_events
        result.raw["endpoint_config"] = knobs
        return result


class _TurnHandler:
    """Map Cartesia turn events without counting eager decisions as EOU."""

    __slots__ = ("eager_events",)

    def __init__(self) -> None:
        self.eager_events: list[dict[str, object]] = []

    def __call__(self, payload: Any, timeline: StreamTimeline) -> bool:
        if not isinstance(payload, dict):
            return False
        kind = payload.get("type")
        if kind == "error" or "error" in payload:
            raise StreamProtocolError(f"cartesia: {orjson.dumps(payload).decode()}")
        if kind in {"connected", "turn.start"}:
            return False
        if kind == "turn.update":
            timeline.record(str(payload.get("transcript", "")), is_final=False)
            return False
        if kind in {"turn.eager_end", "turn.resume"}:
            event: dict[str, object] = {"event": str(kind), "t_s": timeline.elapsed()}
            for name in ("turn_id", "turn_index", "confidence", "end_of_turn_confidence"):
                if name in payload:
                    event[name] = payload[name]
            self.eager_events.append(event)
            return False
        if kind == "turn.end":
            timeline.record(
                str(payload.get("transcript", "")),
                is_final=True,
                kind=EventKind.EOU,
            )
        return False
