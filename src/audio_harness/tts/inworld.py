"""Inworld TTS-2 Realtime text-to-speech adapter."""

from __future__ import annotations

import base64
import os
import time
from typing import Any

import orjson
from websockets.asyncio.client import ClientConnection, connect

from audio_harness.audio import decode_audio_duration
from audio_harness.config import require_env
from audio_harness.stt.base import raise_for_status
from audio_harness.types import Mode, TtsPrompt, TtsResult

from .base import ChunkTimeline, TtsProvider, register, stamp_stream_timing


HTTP_URL = "https://api.inworld.ai/tts/v1/voice"
WS_URL = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"
DEFAULT_VOICE_ID = "Ashley"
AUDIO_ENCODING = "PCM"
"""Headerless 16-bit PCM — the WAV/LINEAR16 variants would count their
header bytes as audio, same reasoning as Deepgram's ``container=none``."""


@register
class InworldTts2(TtsProvider):
    """Inworld TTS-2 Realtime.

    The batch endpoint returns one JSON object with the whole utterance
    base64-encoded, so it carries no useful TTFB signal. The streaming mode
    uses the bidirectional WebSocket (``voice:streamBidirectional``), the
    vendor's realtime transport, with a single-shot context: create, send the
    whole prompt with an immediate flush, close. Message shapes and the
    header-based WebSocket auth follow the vendor's own reference client
    (``inworld-api-examples/tts/python/example_websocket.py``) rather than the
    auto-generated API reference, which claims (incorrectly, per that working
    example) that auth travels as a query parameter.

    Options:
        model_id: Model identifier; defaults to ``inworld-tts-2``.
        voice_id: Voice name such as ``Ashley``. Falls back to the
            ``INWORLD_VOICE_ID`` environment variable, then a built-in voice.
        sample_rate: Output rate; defaults to 24 kHz.
    """

    key = "inworld-tts2"
    vendor = "inworld"
    supports_batch = True
    supports_stream = True

    def _model(self) -> str:
        return str(self.options.get("model_id", "inworld-tts-2"))

    def _voice_id(self) -> str:
        voice = self.options.get("voice_id")
        if voice:
            return str(voice)
        return os.environ.get("INWORLD_VOICE_ID", DEFAULT_VOICE_ID)

    def _api_key(self) -> str:
        return require_env("INWORLD_API_KEY", self.key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Basic {self._api_key()}",
            "Content-Type": "application/json",
        }

    def _audio_config(self) -> dict[str, Any]:
        return {
            "audio_encoding": AUDIO_ENCODING,
            "sample_rate_hertz": self.sample_rate,
        }

    def _body(self, prompt: TtsPrompt) -> dict[str, Any]:
        return {
            "text": prompt.text,
            "voice_id": self._voice_id(),
            "model_id": self._model(),
            "language": prompt.language,
            "audio_config": self._audio_config(),
        }

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Synthesize over HTTP; the response is one JSON blob, not chunked."""
        result = self._result(prompt, Mode.BATCH)
        started = time.perf_counter()
        response = await self.http.post(HTTP_URL, headers=self._headers(), json=self._body(prompt))
        raise_for_status(response, self.key)
        payload = response.json()
        result.ttfb_s = None
        result.total_s = time.perf_counter() - started
        return _finish(result, base64.b64decode(payload["audioContent"]))

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        """Synthesize over the bidirectional WebSocket, one context per call."""
        result = self._result(prompt, Mode.STREAM)
        context_id = f"{self.key}-{prompt.prompt_id}"
        messages = [
            {
                "context_id": context_id,
                "create": {
                    "voice_id": self._voice_id(),
                    "model_id": self._model(),
                    "language": prompt.language,
                    "audio_config": self._audio_config(),
                },
            },
            {
                "context_id": context_id,
                "send_text": {"text": prompt.text, "flush_context": {}},
            },
            {"context_id": context_id, "close_context": {}},
        ]

        timeline = ChunkTimeline()
        headers = {"Authorization": f"Basic {self._api_key()}"}
        async with connect(WS_URL, additional_headers=headers, max_size=None, open_timeout=30.0) as socket:
            for message in messages:
                await socket.send(orjson.dumps(message).decode())
            await self._consume(socket, result, timeline)

        return _finish_stream(result, timeline)

    async def _consume(self, socket: ClientConnection, result: TtsResult, timeline: ChunkTimeline) -> None:
        """Drain one WebSocket context into the timeline."""
        async for raw in socket:
            payload = orjson.loads(raw)
            error = payload.get("error")
            if error:
                result.error = str(error.get("message", "inworld stream error"))
                return
            data = payload.get("result")
            if data is None:
                if payload.get("done"):
                    return
                continue
            chunk = data.get("audioChunk")
            if chunk:
                content = chunk.get("audioContent")
                if content:
                    timeline.add(base64.b64decode(content))
            if "contextClosed" in data:
                return


def _finish(result: TtsResult, audio: bytes) -> TtsResult:
    """Attach audio to a result and derive its duration."""
    result.audio = audio
    result.encoding = "pcm_s16le"
    result.audio_s = decode_audio_duration(audio, encoding=result.encoding, sample_rate=result.sample_rate)
    return result


def _finish_stream(result: TtsResult, timeline: ChunkTimeline) -> TtsResult:
    """Attach streamed audio and stamp the chunk-timing metrics."""
    _finish(result, timeline.audio)
    stamp_stream_timing(result, timeline)
    return result
