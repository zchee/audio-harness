"""xAI Grok speech-to-text adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode

import orjson
from websockets.asyncio.client import ClientConnection

from audio_harness.audio import wrap_wav
from audio_harness.config import require_env
from audio_harness.types import AudioClip, EventKind, Mode, SttResult

from .base import StreamTimeline, SttProvider, raise_for_status, register
from .ws import StreamProtocolError, run_stream


BATCH_URL = "https://api.x.ai/v1/stt"
STREAM_URL = "wss://api.x.ai/v1/stt"


@register
class XaiGrokStt(SttProvider):
    """Grok STT over the xAI REST and WebSocket endpoints.

    Options:
        model: Model identifier for the REST endpoint; defaults to
            ``grok-stt``. The realtime socket takes no model parameter.
        format: Whether xAI applies inverse text normalization and punctuation
            (REST only).
        endpointing: Streaming silence threshold in milliseconds (0-5000)
            before ``speech_final``; xAI's server default is 10 ms when unset.

    Note:
        The realtime socket is configured entirely through URL query
        parameters and accepts only ``audio.done``/``finalize`` control
        frames — a config message such as ``session.update`` is rejected with
        "unknown variant" (verified against the live server, 2026-08-07).
        Transcripts arrive as ``transcript`` events carrying ``transcript``,
        ``is_final`` and ``speech_final`` fields.
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
        """Stream raw PCM binary frames over the query-configured socket."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        params = {
            "sample_rate": str(clip.sample_rate),
            "encoding": "pcm",
            "language": clip.language.split("-")[0],
            "interim_results": "true",
        }
        endpointing = self.options.get("endpointing")
        if endpointing is not None:
            params["endpointing"] = str(endpointing)

        async def audio_done(socket: ClientConnection) -> None:
            await socket.send(orjson.dumps({"type": "audio.done"}).decode())

        await run_stream(
            url=f"{STREAM_URL}?{urlencode(params)}",
            headers=self._auth(),
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=make_handler(),
            on_input_done=audio_done,
        )

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        result.raw["eou_source"] = "speech_final"
        result.raw["endpoint_config"] = {
            "endpointing": endpointing if endpointing is not None else 10,
        }
        return result


def make_handler() -> Callable[[Any, StreamTimeline], bool]:
    """Build a per-session handler for ``transcript.partial`` events.

    xAI emits per-segment hypotheses: interims, then a final without
    ``speech_final``, then the same segment text again with
    ``speech_final: true`` once the endpointing decision lands (observed
    live 2026-08-07). Concatenating both finals would double every segment,
    so the restated frame is recorded as a bare EOU event instead — which
    needs one segment of state, hence a closure per session.

    The error frame is raised with its full raw payload: the server's
    message text is the only protocol documentation xAI exposes, and a
    truncated "xai: error" cost a whole 360-clip run to diagnose once.
    """
    last_final: tuple[Any, str] | None = None

    def handle(payload: Any, timeline: StreamTimeline) -> bool:
        nonlocal last_final
        if not isinstance(payload, dict):
            return False
        kind = payload.get("type", "")

        if kind == "error" or "error" in payload:
            raise StreamProtocolError(f"xai: {orjson.dumps(payload).decode()}")
        if kind == "transcript.done":
            return True
        if kind != "transcript.partial":
            return False

        text = str(payload.get("text", ""))
        if bool(payload.get("speech_final")):
            if (payload.get("start"), text) == last_final:
                timeline.record("", is_final=False, kind=EventKind.EOU)
                return False
            last_final = (payload.get("start"), text)
            timeline.record(text, is_final=True, kind=EventKind.EOU)
            return False
        if bool(payload.get("is_final")):
            last_final = (payload.get("start"), text)
            timeline.record(text, is_final=True)
            return False
        timeline.record(text, is_final=False)
        return False

    return handle
