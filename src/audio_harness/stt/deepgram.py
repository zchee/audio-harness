"""Deepgram Nova-3 speech-to-text adapter."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import orjson
from websockets.asyncio.client import ClientConnection

from ..audio import wrap_wav
from ..config import require_env
from ..types import AudioClip, EventKind, Mode, SttResult
from .base import StreamTimeline, SttProvider, raise_for_status, register
from .ws import run_stream

BATCH_URL = "https://api.deepgram.com/v1/listen"
STREAM_URL = "wss://api.deepgram.com/v1/listen"


@register
class DeepgramNova3(SttProvider):
    """Deepgram Nova-3 over the REST and WebSocket listen endpoints.

    Options:
        model: Model name; ``nova-3`` (monolingual) or ``nova-3-multilingual``.
        language: Language code sent to Deepgram. Use ``multi`` with the
            multilingual model to enable code-switching.
        smart_format: Whether Deepgram applies punctuation and formatting.
        endpointing: Streaming silence threshold in milliseconds before a
            ``speech_final`` result, or ``false`` to disable. Deepgram's
            server default is 10 ms when unset.
        utterance_end_ms: When set, Deepgram also sends ``UtteranceEnd``
            messages after this word-gap (>= 1000 recommended; requires
            interim results, which this adapter always enables).
    """

    key = "deepgram-nova3"
    vendor = "deepgram"
    supports_batch = True
    supports_stream = True

    def _model(self) -> str:
        return str(self.options.get("model", "nova-3"))

    def _language(self, clip: AudioClip) -> str:
        return str(self.options.get("language", clip.language.split("-")[0]))

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Token {require_env('DEEPGRAM_API_KEY', self.key)}"}

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Post the clip as a WAV file to the pre-recorded endpoint."""
        result = self._result(clip, Mode.BATCH)
        params = {
            "model": self._model(),
            "language": self._language(clip),
            "smart_format": str(self.options.get("smart_format", True)).lower(),
        }
        response = await self.http.post(
            f"{BATCH_URL}?{urlencode(params)}",
            headers={**self._auth(), "Content-Type": "audio/wav"},
            content=wrap_wav(clip.pcm, clip.sample_rate),
        )
        raise_for_status(response, self.key)
        payload = response.json()
        result.total_s = response.elapsed.total_seconds()
        result.text = _batch_transcript(payload)
        result.raw["response"] = payload
        return result

    async def transcribe_stream(
        self, clip: AudioClip, *, chunk_ms: int, realtime: bool
    ) -> SttResult:
        """Stream PCM over the listen WebSocket with interim results enabled."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        params = {
            "model": self._model(),
            "language": self._language(clip),
            "encoding": "linear16",
            "sample_rate": str(clip.sample_rate),
            "channels": "1",
            "interim_results": "true",
            "smart_format": str(self.options.get("smart_format", True)).lower(),
        }
        endpointing = self.options.get("endpointing")
        utterance_end_ms = self.options.get("utterance_end_ms")
        if endpointing is not None:
            params["endpointing"] = str(endpointing).lower()
        if utterance_end_ms:
            params["utterance_end_ms"] = str(utterance_end_ms)

        async def close_stream(socket: ClientConnection) -> None:
            await socket.send(orjson.dumps({"type": "CloseStream"}).decode())

        await run_stream(
            url=f"{STREAM_URL}?{urlencode(params)}",
            headers=self._auth(),
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=_handle_message,
            on_input_done=close_stream,
        )

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        # Deepgram's endpointing is on server-side by default (10 ms), so
        # speech_final events exist on every stream — the lane is always
        # EOU-capable. The effective knob values are recorded because a
        # ranking is only comparable when each lane's configuration is known.
        result.raw["eou_source"] = (
            "speech_final+utterance_end" if utterance_end_ms else "speech_final"
        )
        result.raw["endpoint_config"] = {
            "endpointing": endpointing if endpointing is not None else 10,
            "utterance_end_ms": utterance_end_ms,
        }
        return result


def _batch_transcript(payload: dict[str, Any]) -> str:
    """Extract the transcript from a pre-recorded response."""
    channels = payload.get("results", {}).get("channels", [])
    if not channels:
        return ""
    alternatives = channels[0].get("alternatives", [])
    return str(alternatives[0].get("transcript", "")) if alternatives else ""


def _handle_message(payload: Any, timeline: StreamTimeline) -> bool:
    """Record one listen-socket event.

    Deepgram emits one ``Results`` frame per interim update and one with
    ``is_final`` per finished segment, so the full transcript is the
    concatenation of the final frames. Two independent signals mark end of
    utterance: ``speech_final`` on a Results frame (VAD endpointing) and the
    bare ``UtteranceEnd`` message (word-gap analysis, when enabled). Both are
    recorded as EOU events; a segment final without ``speech_final`` is not
    an endpointing decision and stays ``segment_final``.
    """
    if not isinstance(payload, dict):
        return False
    kind = payload.get("type")
    if kind == "Metadata":
        return True
    if kind == "UtteranceEnd":
        # last_word_end of -1 means the result was already finalized before
        # the gap condition was met; Deepgram documents it as a stale
        # notification to ignore.
        if payload.get("last_word_end") != -1:
            timeline.record("", is_final=False, kind=EventKind.EOU)
        return False
    if kind != "Results":
        return False

    alternatives = payload.get("channel", {}).get("alternatives", [])
    if not alternatives:
        return False
    timeline.record(
        str(alternatives[0].get("transcript", "")),
        is_final=bool(payload.get("is_final")),
        kind=EventKind.EOU if payload.get("speech_final") else "",
    )
    return False
