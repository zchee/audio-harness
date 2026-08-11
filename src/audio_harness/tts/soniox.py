"""Soniox tts-rt-v2 text-to-speech adapter."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Coroutine
import time
from typing import Any

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


BATCH_URL = "https://tts-rt.soniox.com/tts"
WS_URL = "wss://tts-rt.soniox.com/tts-websocket"
MODEL = "tts-rt-v2"
STREAM_ID = "stream-1"

_SAMPLE_RATES = {8000, 16000, 24000, 44100, 48000}


class SonioxTtsProtocolError(RuntimeError):
    """An error frame returned by the Soniox TTS WebSocket."""


@register
class SonioxTts(TtsProvider):
    """Soniox tts-rt-v2 over the REST and WebSocket endpoints.

    Both transports return raw signed 16-bit little-endian PCM. WebSocket
    authentication is carried in the first client frame rather than an HTTP
    header, and the same session accepts either a complete prompt or paced
    incremental text pieces.

    Options:
        voice: Voice identifier; defaults to ``Adrian``.
        sample_rate: PCM output rate, one of 8000, 16000, 24000, 44100 or
            48000. Defaults to 24000.

    Pricing:
        $0.70 per generated audio hour, verified 2026-08-12.
    """

    key = "soniox-tts-rt-v2"
    vendor = "soniox"
    family = "soniox"
    supports_batch = True
    supports_stream = True
    supports_input_streaming = True

    def _voice(self) -> str:
        return str(self.options.get("voice", "Adrian"))

    def _api_key(self) -> str:
        return require_env("SONIOX_API_KEY", self.key)

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
            "model": MODEL,
            "language": self._language(prompt),
            "voice": self._voice(),
            "audio_format": "pcm_s16le",
            "sample_rate": self._sample_rate(),
            "text": prompt.text,
        }

    def _config(self, prompt: TtsPrompt) -> dict[str, Any]:
        return {
            "api_key": self._api_key(),
            "stream_id": STREAM_ID,
            "model": MODEL,
            "language": self._language(prompt),
            "voice": self._voice(),
            "audio_format": "pcm_s16le",
            "sample_rate": self._sample_rate(),
        }

    @staticmethod
    async def _send_config(socket: ClientConnection, config: dict[str, Any]) -> None:
        """Send the authenticated first frame before any input text."""
        await socket.send(orjson.dumps(config).decode())

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Request raw PCM from the batch endpoint."""
        result = self._result(prompt, Mode.BATCH)
        result.raw["model"] = MODEL
        started = time.perf_counter()
        response = await self.http.post(BATCH_URL, headers=self._headers(), json=self._body(prompt))
        raise_for_status(response, self.key)
        result.ttfb_s = None
        result.total_s = time.perf_counter() - started
        return _finish(result, response.content)

    async def _consume(self, socket: ClientConnection, timeline: ChunkTimeline) -> None:
        """Drain audio frames until ``terminated`` or raise on an error frame."""
        async for raw in socket:
            payload = orjson.loads(raw)
            if not isinstance(payload, dict):
                continue
            if payload.get("error_code") is not None:
                raise SonioxTtsProtocolError(f"soniox TTS: {orjson.dumps(payload).decode()}")
            audio = payload.get("audio")
            if audio:
                timeline.add(base64.b64decode(audio))
            if payload.get("terminated") is True:
                return
        raise SonioxTtsProtocolError("soniox TTS: WebSocket ended before terminated")

    async def _synthesize_ws(
        self,
        prompt: TtsPrompt,
        feed: Callable[[ClientConnection], Coroutine[Any, Any, None]],
        *,
        input_streaming: bool,
    ) -> TtsResult:
        result = self._result(prompt, Mode.STREAM)
        result.raw["model"] = MODEL
        result.raw["input_streaming"] = input_streaming
        timeline = ChunkTimeline()
        config = self._config(prompt)

        async with connect(WS_URL, max_size=None, open_timeout=30.0) as socket:

            async def configured_feed() -> None:
                await self._send_config(socket, config)
                await feed(socket)

            feeder = asyncio.create_task(configured_feed())
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
            await socket.send(orjson.dumps({"text": prompt.text, "text_end": False, "stream_id": STREAM_ID}).decode())
            await socket.send(orjson.dumps({"text": "", "text_end": True, "stream_id": STREAM_ID}).decode())

        return await self._synthesize_ws(prompt, feed, input_streaming=False)

    async def synthesize_incremental(self, prompt: TtsPrompt, *, token_rate: float) -> TtsResult:
        """Feed word-sized text pieces at a simulated LLM-token cadence."""
        pieces = token_pieces(prompt.text)

        async def feed(socket: ClientConnection) -> None:
            async for piece in pace_tokens(pieces, token_rate):
                await socket.send(orjson.dumps({"text": piece, "text_end": False, "stream_id": STREAM_ID}).decode())
            await socket.send(orjson.dumps({"text": "", "text_end": True, "stream_id": STREAM_ID}).decode())

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
