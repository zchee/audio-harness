"""xAI Grok text-to-speech adapter."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Coroutine
import time
from typing import Any
from urllib.parse import urlencode

import orjson
from websockets.asyncio.client import ClientConnection, connect

from audio_harness.audio import decode_audio_duration
from audio_harness.config import require_env
from audio_harness.stt.base import raise_for_status
from audio_harness.types import Mode, TtsPrompt, TtsResult

from .base import (
    ChunkTimeline,
    TtsProvider,
    pace_tokens,
    register,
    stamp_stream_timing,
    token_pieces,
)


BATCH_URL = "https://api.x.ai/v1/tts"
WS_URL = "wss://api.x.ai/v1/tts"

_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}


class XaiTtsProtocolError(RuntimeError):
    """An error frame returned by the xAI TTS WebSocket."""


@register
class XaiGrokTts(TtsProvider):
    """Grok TTS over the xAI REST and WebSocket endpoints.

    xAI exposes no selectable TTS model identifier for this lane. The REST
    endpoint returns raw signed 16-bit little-endian PCM; the WebSocket emits
    the same PCM in base64-encoded ``audio.delta`` events. Both default to the
    ``eve`` voice and 24 kHz output.

    Options:
        voice_id: Voice identifier; defaults to ``eve``.
        sample_rate: PCM output rate, one of 8000, 16000, 22050, 24000,
            44100 or 48000. Defaults to 24000.

    Pricing:
        $15.00 per million input characters, verified 2026-08-08.
    """

    key = "xai-grok-tts"
    vendor = "xai"
    family = "xai"
    supports_batch = True
    supports_stream = True
    supports_input_streaming = True

    def _voice_id(self) -> str:
        return str(self.options.get("voice_id", "eve"))

    def _api_key(self) -> str:
        return require_env("XAI_API_KEY", self.key)

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key()}"}

    def _headers(self) -> dict[str, str]:
        return {**self._auth(), "Content-Type": "application/json"}

    def _sample_rate(self) -> int:
        rate = self.sample_rate
        if rate not in _SAMPLE_RATES:
            raise ValueError(f"{self.key}: unsupported sample rate {rate}; expected one of {sorted(_SAMPLE_RATES)}")
        return rate

    @staticmethod
    def _language(prompt: TtsPrompt) -> str:
        return prompt.language.split("-")[0]

    def _body(self, prompt: TtsPrompt) -> dict[str, Any]:
        return {
            "text": prompt.text,
            "voice_id": self._voice_id(),
            "language": self._language(prompt),
            "output_format": {"codec": "pcm", "sample_rate": self._sample_rate()},
        }

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Request raw PCM from the batch endpoint."""
        result = self._result(prompt, Mode.BATCH)
        result.raw["model_unspecified"] = True
        started = time.perf_counter()
        response = await self.http.post(BATCH_URL, headers=self._headers(), json=self._body(prompt))
        raise_for_status(response, self.key)
        result.ttfb_s = None
        result.total_s = time.perf_counter() - started
        return _finish(result, response.content)

    def _ws_url(self, prompt: TtsPrompt) -> str:
        params = {
            # The socket intentionally uses ``voice`` while REST uses
            # ``voice_id``; this asymmetry is part of xAI's wire contract.
            "voice": self._voice_id(),
            "language": self._language(prompt),
            "codec": "pcm",
            "sample_rate": self._sample_rate(),
        }
        return f"{WS_URL}?{urlencode(params)}"

    async def _consume(self, socket: ClientConnection, timeline: ChunkTimeline) -> None:
        """Drain audio events until ``audio.done`` or raise on an error frame."""
        async for raw in socket:
            payload = orjson.loads(raw)
            if not isinstance(payload, dict):
                continue
            kind = payload.get("type")
            if kind == "error" or "error" in payload:
                # Raise rather than returning a partial successful result: a
                # protocol error makes the received audio unusable evidence.
                raise XaiTtsProtocolError(f"xai TTS: {orjson.dumps(payload).decode()}")
            if kind == "audio.delta":
                delta = payload.get("delta")
                if delta:
                    timeline.add(base64.b64decode(delta))
                continue
            if kind == "audio.done":
                return
        raise XaiTtsProtocolError("xai TTS: WebSocket ended before audio.done")

    async def _synthesize_ws(
        self,
        prompt: TtsPrompt,
        feed: Callable[[ClientConnection], Coroutine[Any, Any, None]],
        *,
        input_streaming: bool,
    ) -> TtsResult:
        result = self._result(prompt, Mode.STREAM)
        result.raw["model_unspecified"] = True
        result.raw["input_streaming"] = input_streaming
        timeline = ChunkTimeline()

        async with connect(
            self._ws_url(prompt), additional_headers=self._auth(), max_size=None, open_timeout=30.0
        ) as socket:
            feeder = asyncio.create_task(feed(socket))
            consumer = asyncio.create_task(self._consume(socket, timeline))
            try:
                done, pending = await asyncio.wait(
                    {feeder, consumer},
                    return_when=asyncio.FIRST_EXCEPTION,
                )
            except BaseException:
                for task in (feeder, consumer):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(feeder, consumer, return_exceptions=True)
                raise

            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            # Prefer the protocol/transport failure that ended consumption if
            # both tasks fail as the socket closes. A feeder-first failure is
            # still surfaced promptly instead of waiting forever for a server
            # that leaves the receive side open.
            consumer_error = consumer.exception() if consumer in done and not consumer.cancelled() else None
            feeder_error = feeder.exception() if feeder in done and not feeder.cancelled() else None
            if consumer_error is not None:
                raise consumer_error
            if feeder_error is not None:
                raise feeder_error
            if consumer.cancelled() or feeder.cancelled():
                raise asyncio.CancelledError

        return _finish_stream(result, timeline)

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        """Send the complete prompt through one WebSocket generation."""

        async def feed(socket: ClientConnection) -> None:
            await socket.send(orjson.dumps({"type": "text.delta", "delta": prompt.text}).decode())
            await socket.send(orjson.dumps({"type": "text.done"}).decode())

        return await self._synthesize_ws(prompt, feed, input_streaming=False)

    async def synthesize_incremental(self, prompt: TtsPrompt, *, token_rate: float) -> TtsResult:
        """Feed word-sized text pieces at a simulated LLM-token cadence."""
        pieces = token_pieces(prompt.text)

        async def feed(socket: ClientConnection) -> None:
            async for piece in pace_tokens(pieces, token_rate):
                await socket.send(orjson.dumps({"type": "text.delta", "delta": piece}).decode())
            await socket.send(orjson.dumps({"type": "text.done"}).decode())

        result = await self._synthesize_ws(prompt, feed, input_streaming=True)
        result.raw["text_pieces"] = len(pieces)
        return result


def _finish(result: TtsResult, audio: bytes) -> TtsResult:
    """Attach raw s16le PCM to a result and derive its duration."""
    result.audio = audio
    result.encoding = "pcm_s16le"
    result.audio_s = decode_audio_duration(audio, encoding=result.encoding, sample_rate=result.sample_rate)
    return result


def _finish_stream(result: TtsResult, timeline: ChunkTimeline) -> TtsResult:
    """Attach streamed audio and stamp chunk timing metrics."""
    _finish(result, timeline.audio)
    stamp_stream_timing(result, timeline)
    return result
