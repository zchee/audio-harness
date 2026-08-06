"""ElevenLabs Scribe v2 speech-to-text adapter."""

from __future__ import annotations

import base64
from typing import Any
from urllib.parse import urlencode

import orjson
from websockets.asyncio.client import ClientConnection

from audio_harness.audio import wrap_wav
from audio_harness.config import require_env
from audio_harness.types import AudioClip, EventKind, Mode, SttResult

from .base import StreamTimeline, SttProvider, raise_for_status, register
from .ws import HandleMessage, StreamProtocolError, run_stream


BATCH_URL = "https://api.elevenlabs.io/v1/speech-to-text"
STREAM_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

_SUPPORTED_STREAM_RATES = frozenset({8000, 16000, 24000, 44100, 48000})


@register
class ElevenLabsScribeV2(SttProvider):
    """ElevenLabs Scribe v2, batch and realtime.

    The two transports use different model identifiers — ``scribe_v2`` for
    pre-recorded audio and ``scribe_v2_realtime`` for streaming — so each mode
    reads its own option.

    Options:
        batch_model: Pre-recorded model id; defaults to ``scribe_v2``.
        stream_model: Realtime model id; defaults to ``scribe_v2_realtime``.
        commit_strategy: ``vad`` to segment on silence, or ``manual``.
    """

    key = "elevenlabs-scribe2"
    vendor = "elevenlabs"
    supports_batch = True
    supports_stream = True

    def _auth(self) -> dict[str, str]:
        return {"xi-api-key": require_env("ELEVENLABS_API_KEY", self.key)}

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Post the clip as multipart form data to the transcription endpoint."""
        result = self._result(clip, Mode.BATCH)
        response = await self.http.post(
            BATCH_URL,
            headers=self._auth(),
            files={"file": ("audio.wav", wrap_wav(clip.pcm, clip.sample_rate))},
            data={
                "model_id": str(self.options.get("batch_model", "scribe_v2")),
                "language_code": clip.language.split("-")[0],
            },
        )
        raise_for_status(response, self.key)
        payload = response.json()
        result.total_s = response.elapsed.total_seconds()
        result.text = str(payload.get("text", ""))
        result.raw["response"] = payload
        return result

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Stream base64 PCM frames over the realtime socket.

        Raises:
            ValueError: If the clip's sample rate has no ``pcm_*`` audio format.
        """
        if clip.sample_rate not in _SUPPORTED_STREAM_RATES:
            raise ValueError(
                f"{self.key}: unsupported stream sample rate {clip.sample_rate}; "
                f"expected one of {sorted(_SUPPORTED_STREAM_RATES)}"
            )

        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        params = {
            "model_id": str(self.options.get("stream_model", "scribe_v2_realtime")),
            "audio_format": f"pcm_{clip.sample_rate}",
            "commit_strategy": str(self.options.get("commit_strategy", "vad")),
            "language_code": clip.language.split("-")[0],
        }

        def encode(chunk: bytes) -> str:
            return orjson.dumps({
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(chunk).decode("ascii"),
            }).decode()

        async def commit(socket: ClientConnection) -> None:
            await socket.send(
                orjson.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",
                    "commit": True,
                }).decode()
            )

        vad = str(params["commit_strategy"]) == "vad"
        await run_stream(
            url=f"{STREAM_URL}?{urlencode(params)}",
            headers=self._auth(),
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=_make_handler(vad_commits=vad),
            encode_chunk=encode,
            on_input_done=commit,
        )

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        if vad:
            result.raw["eou_source"] = "vad_commit"
            result.raw["endpoint_config"] = {"commit_strategy": "vad"}
        return result


def _make_handler(*, vad_commits: bool) -> HandleMessage:
    """Build the message handler for one session.

    ElevenLabs signals finality through the message type rather than a flag:
    ``partial_transcript`` is interim, ``committed_transcript`` is immutable.
    Committed frames cover one segment each, so they are concatenated. The
    API exposes no separate end-of-utterance event; under the ``vad`` commit
    strategy the commit itself *is* the vendor's end-of-speech decision, so
    those frames carry the EOU kind. Manual commits are ours, not the
    vendor's, and stay segment finals.
    """

    def handle(payload: Any, timeline: StreamTimeline) -> bool:
        if not isinstance(payload, dict):
            return False
        kind = payload.get("message_type")

        if kind in {"error", "rate_limited"}:
            raise StreamProtocolError(f"elevenlabs: {payload.get('error', kind)}")
        if kind == "session_started":
            return False
        if kind == "partial_transcript":
            timeline.record(str(payload.get("text", "")), is_final=False)
            return False
        if kind in {"committed_transcript", "committed_transcript_with_timestamps"}:
            timeline.record(
                str(payload.get("text", "")),
                is_final=True,
                kind=EventKind.EOU if vad_commits else "",
            )
        return False

    return handle
