"""xAI Grok speech-to-text adapter."""

from __future__ import annotations

from typing import Any

import orjson
from websockets.asyncio.client import ClientConnection

from audio_harness.audio import wrap_wav
from audio_harness.config import require_env
from audio_harness.types import AudioClip, Mode, SttResult

from .base import StreamTimeline, SttProvider, raise_for_status, register
from .ws import StreamProtocolError, run_stream


BATCH_URL = "https://api.x.ai/v1/stt"
STREAM_URL = "wss://api.x.ai/v1/stt"


@register
class XaiGrokStt(SttProvider):
    """Grok STT over the xAI REST and WebSocket endpoints.

    Options:
        model: Model identifier; defaults to ``grok-stt``.
        format: Whether xAI applies inverse text normalization and punctuation.

    Note:
        The REST contract is documented by xAI. The realtime frame names used
        here (``session.update``, ``audio.done``, ``transcript.delta`` and
        ``transcript.done``) were taken from third-party integrations rather
        than a first-party schema, so verify streaming results against a known
        transcript before trusting the streaming latency figures.
    """

    key = "xai-grok-stt"
    vendor = "xai"
    supports_batch = True
    supports_stream = True

    def _model(self) -> str:
        return str(self.options.get("model", "grok-stt"))

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {require_env('XAI_API_KEY', self.key)}"}

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Post the clip as multipart form data to the STT endpoint."""
        result = self._result(clip, Mode.BATCH)
        response = await self.http.post(
            BATCH_URL,
            headers=self._auth(),
            data={
                "model": self._model(),
                "language": clip.language.split("-")[0],
                "format": str(self.options.get("format", True)).lower(),
            },
            files={
                "file": (
                    "audio.wav",
                    wrap_wav(clip.pcm, clip.sample_rate),
                    "audio/wav",
                )
            },
        )
        raise_for_status(response, self.key)
        payload = response.json()
        result.total_s = response.elapsed.total_seconds()
        result.text = str(payload.get("text", ""))
        result.raw["response"] = payload
        return result

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Stream raw PCM frames over the realtime socket."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()

        async def configure(socket: ClientConnection) -> None:
            await socket.send(
                orjson.dumps({
                    "type": "session.update",
                    "model": self._model(),
                    "language": clip.language.split("-")[0],
                    "sample_rate": clip.sample_rate,
                    "interim_results": True,
                }).decode()
            )

        async def audio_done(socket: ClientConnection) -> None:
            await socket.send(orjson.dumps({"type": "audio.done"}).decode())

        await run_stream(
            url=STREAM_URL,
            headers=self._auth(),
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=_handle_message,
            on_open=configure,
            on_input_done=audio_done,
        )

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        return result


def _handle_message(payload: Any, timeline: StreamTimeline) -> bool:
    """Record one realtime event."""
    if not isinstance(payload, dict):
        return False
    kind = payload.get("type", "")

    if kind == "error" or "error" in payload:
        raise StreamProtocolError(f"xai: {payload.get('error', kind)}")
    if kind in {"session.done", "transcript.complete"}:
        return True
    if kind == "transcript.delta":
        timeline.record(str(payload.get("text", "")), is_final=False)
        return False
    if kind == "transcript.done":
        timeline.record(str(payload.get("text", "")), is_final=True)
    return False
