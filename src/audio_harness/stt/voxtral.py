"""Mistral Voxtral realtime speech-to-text adapter."""

from __future__ import annotations

import base64
from typing import Any
from urllib.parse import urlencode

import orjson
from websockets.asyncio.client import ClientConnection

from audio_harness.config import require_env
from audio_harness.types import AudioClip, Mode, SttResult

from .base import StreamTimeline, SttProvider, register
from .ws import StreamProtocolError, run_stream


STREAM_URL = "wss://api.mistral.ai/v1/audio/transcriptions/realtime"


@register
class MistralVoxtralRealtime(SttProvider):
    """Mistral Voxtral Mini realtime transcription over its WebSocket API.

    Options:
        model: Model identifier; defaults to the dated
            ``voxtral-mini-transcribe-realtime-2602`` release.
        target_streaming_delay_ms: Server-side accuracy/latency trade-off.
            Rejected once audio has started, so it only ever rides the
            initial session configuration.

    The wire protocol here was verified against Mistral's own open-source
    ``client-python`` SDK (``src/mistralai/extra/realtime``), not the prose
    docs, which describe only the SDK surface and never the raw messages.
    After the handshake the server streams incremental
    ``transcription.text.delta`` fragments that must be concatenated to
    reconstruct the growing hypothesis, occasional ``transcription.segment``
    events that finalize one timestamped span at a time, and exactly one
    terminal ``transcription.done`` carrying the authoritative whole-session
    transcript. There is no end-of-turn, end-of-speech or VAD event anywhere
    in the protocol — confirmed against both the SDK's message models and the
    pipecat connector, which layers its own client-side VAD on top rather
    than relying on any server signal — so this lane is descriptive-only for
    endpointing and never populates ``eou_source``.
    """

    key = "mistral-voxtral-realtime"
    vendor = "mistral"
    supports_stream = True

    def _model(self) -> str:
        return str(self.options.get("model", "voxtral-mini-transcribe-realtime-2602"))

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {require_env('MISTRAL_API_KEY', self.key)}"}

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Stream PCM over the realtime socket and reassemble the delta stream."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        delay_ms = self.options.get("target_streaming_delay_ms")

        async def configure(socket: ClientConnection) -> None:
            session: dict[str, Any] = {
                "audio_format": {"encoding": "pcm_s16le", "sample_rate": clip.sample_rate},
            }
            if delay_ms is not None:
                session["target_streaming_delay_ms"] = int(delay_ms)
            await socket.send(orjson.dumps({"type": "session.update", "session": session}).decode())

        async def end_of_audio(socket: ClientConnection) -> None:
            await socket.send(orjson.dumps({"type": "input_audio.flush"}).decode())
            await socket.send(orjson.dumps({"type": "input_audio.end"}).decode())

        accumulator = _DeltaAccumulator()
        params = {"model": self._model()}
        await run_stream(
            url=f"{STREAM_URL}?{urlencode(params)}",
            headers=self._auth(),
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=accumulator,
            on_open=configure,
            encode_chunk=_encode_chunk,
            on_input_done=end_of_audio,
        )

        result.text = accumulator.done_text or timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        if accumulator.language is not None:
            result.raw["detected_language"] = accumulator.language
        if accumulator.done_payload is not None:
            result.raw["response"] = accumulator.done_payload
        return result


def _encode_chunk(chunk: bytes) -> str:
    """Frame one PCM chunk as a base64 ``input_audio.append`` message."""
    return orjson.dumps({
        "type": "input_audio.append",
        "audio": base64.b64encode(chunk).decode("ascii"),
    }).decode()


class _DeltaAccumulator:
    """Reassembles Mistral's incremental delta stream into timeline events.

    ``transcription.text.delta`` fragments are appended, not restated, so the
    running hypothesis has to be tracked here; each ``transcription.segment``
    finalizes that span with its own (non-cumulative) text — recorded so
    :meth:`StreamTimeline.concat_finals` reconstructs the transcript
    correctly — and resets the tail for the next span.
    ``transcription.done`` is authoritative and terminal, arriving once for
    the whole session rather than once per utterance.
    """

    __slots__ = ("_finalized_text", "_tail", "done_payload", "done_text", "language")

    def __init__(self) -> None:
        self.done_text = ""
        self.done_payload: dict[str, Any] | None = None
        self.language: str | None = None
        self._finalized_text = ""
        self._tail = ""

    def __call__(self, payload: Any, timeline: StreamTimeline) -> bool:
        if not isinstance(payload, dict):
            return False
        kind = payload.get("type")

        if kind == "error":
            raise StreamProtocolError(f"mistral: {_error_message(payload)}")
        if kind in {"session.created", "session.updated"}:
            return False
        if kind == "transcription.language":
            self.language = payload.get("audio_language")
            return False
        if kind == "transcription.text.delta":
            self._tail += str(payload.get("text", ""))
            timeline.record(f"{self._finalized_text} {self._tail}".strip(), is_final=False)
            return False
        if kind == "transcription.segment":
            text = str(payload.get("text", ""))
            self._finalized_text = f"{self._finalized_text} {text}".strip()
            self._tail = ""
            timeline.record(text, is_final=True)
            return False
        if kind == "transcription.done":
            self.done_text = str(payload.get("text", ""))
            self.done_payload = payload
            return True
        return False


def _error_message(payload: dict[str, Any]) -> str:
    """Extract the human-readable detail from an ``error`` frame.

    Mirrors the official SDK's extraction: ``error.message`` is either a
    plain string or a ``{"detail": ...}`` mapping.
    """
    error = payload.get("error")
    message = error.get("message") if isinstance(error, dict) else error
    if isinstance(message, dict):
        return str(message.get("detail", message))
    return str(message or payload)
