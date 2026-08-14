"""Mistral Voxtral speech-to-text adapters (realtime and batch)."""

from __future__ import annotations

import base64
import time
from typing import Any
from urllib.parse import urlencode

import orjson
from websockets.asyncio.client import ClientConnection

from audio_harness.audio import wrap_wav
from audio_harness.config import require_env
from audio_harness.types import AudioClip, Mode, SttResult

from .base import StreamTimeline, SttProvider, raise_for_status, register
from .ws import StreamProtocolError, run_stream


STREAM_URL = "wss://api.mistral.ai/v1/audio/transcriptions/realtime"
TRANSCRIPTIONS_URL = "https://api.mistral.ai/v1/audio/transcriptions"
CHAT_URL = "https://api.mistral.ai/v1/chat/completions"


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


@register
class MistralVoxtralMiniBatch(SttProvider):
    """Voxtral Mini transcription through La Plateforme's batch endpoint.

    The direct-vendor counterpart of the or-voxtral-mini lane (no
    hosted-proxy hop). The transcription endpoint serves only the Mini
    transcribe lineage: every accepted model alias resolves to
    ``voxtral-mini-latest`` in the response (probe-verified 2026-08-14),
    which is why there is no small-24B variant of this class — see
    :class:`MistralVoxtralSmallBatch` for how Small is reached.

    Options:
        model: Model identifier; defaults to the dated ``voxtral-mini-2507``.
        language: ISO-639-1 hint forwarded to the endpoint.
    """

    key = "mistral-voxtral-mini"
    vendor = "mistral"
    supports_batch = True

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {require_env('MISTRAL_API_KEY', self.key)}"}

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Upload a WAV-wrapped clip and extract the JSON transcript."""
        result = self._result(clip, Mode.BATCH)
        data: dict[str, Any] = {"model": str(self.options.get("model", "voxtral-mini-2507"))}
        language = self.options.get("language")
        if language is not None:
            data["language"] = str(language)
        started = time.perf_counter()
        response = await self.http.post(
            TRANSCRIPTIONS_URL,
            headers=self._auth(),
            data=data,
            files={"file": ("audio.wav", wrap_wav(clip.pcm, clip.sample_rate), "audio/wav")},
        )
        raise_for_status(response, self.key)
        payload = response.json()
        result.total_s = time.perf_counter() - started
        result.text = str(payload.get("text", ""))
        result.raw["model"] = payload.get("model")
        result.raw["response"] = {k: v for k, v in payload.items() if k != "segments"}
        return result


TRANSCRIBE_PROMPT = "Transcribe this audio verbatim. Output only the transcript text."
"""Fixed elicitation prompt for the chat-served Small lane. Changing it
changes the lane's behaviour, so it is a constant, not an option."""


@register
class MistralVoxtralSmallBatch(SttProvider):
    """Voxtral Small 24B transcription through chat completions.

    La Plateforme's transcription endpoint rejects every Small alias
    (probe-verified 2026-08-14), so the only direct-vendor path to the
    24B model is the chat API with ``input_audio`` content and a fixed
    transcription prompt at temperature 0. That is prompted transcription
    — the same elicitation shape as an audio-LLM lane, not a dedicated
    ASR endpoint — recorded as ``raw["prompted_transcription"]`` so the
    caveat travels with every row.

    Options:
        model: Model identifier; defaults to the dated ``voxtral-small-2507``.
    """

    key = "mistral-voxtral-small"
    vendor = "mistral"
    supports_batch = True

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {require_env('MISTRAL_API_KEY', self.key)}"}

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Send base64 WAV through chat completions and read the reply text."""
        result = self._result(clip, Mode.BATCH)
        result.raw["prompted_transcription"] = True
        body = {
            "model": str(self.options.get("model", "voxtral-small-2507")),
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": base64.b64encode(wrap_wav(clip.pcm, clip.sample_rate)).decode("ascii"),
                        },
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                    ],
                }
            ],
        }
        started = time.perf_counter()
        response = await self.http.post(
            CHAT_URL,
            headers={**self._auth(), "Content-Type": "application/json"},
            content=orjson.dumps(body),
        )
        raise_for_status(response, self.key)
        payload = response.json()
        result.total_s = time.perf_counter() - started
        choices = payload.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        result.text = str(message.get("content", ""))
        result.raw["model"] = payload.get("model")
        result.raw["usage"] = payload.get("usage")
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
