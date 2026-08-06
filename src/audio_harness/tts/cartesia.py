"""Cartesia Sonic text-to-speech adapters."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import time
from typing import Any
from urllib.parse import urlencode

import orjson
from websockets.asyncio.client import connect

from ..audio import decode_audio_duration
from ..config import require_env
from ..stt.base import raise_for_status
from ..types import Mode, TtsPrompt, TtsResult
from .base import (
    ChunkTimeline,
    TtsProvider,
    pace_tokens,
    register,
    stamp_stream_timing,
    token_pieces,
)

HTTP_URL = "https://api.cartesia.ai/tts/bytes"
WS_URL = "wss://api.cartesia.ai/tts/websocket"
DEFAULT_VERSION = "2026-03-01"


class _CartesiaBase(TtsProvider):
    """Shared transport for the Sonic model family.

    Sonic 3.0 and 3.5 differ only by ``model_id``, so both benchmark entries
    share one implementation and one API key.

    Options:
        model_id: Sonic model identifier; set by each subclass.
        voice_id: Voice UUID. Pin this across models — comparing two models on
            different voices measures the voices, not the models.
        version: Value sent as the Cartesia API version.
        sample_rate: Output rate; defaults to 24 kHz.
    """

    vendor = "cartesia"
    supports_batch = True
    supports_stream = True
    supports_input_streaming = True
    model_id = "sonic-3.5"

    def _model(self) -> str:
        return str(self.options.get("model_id", self.model_id))

    def _version(self) -> str:
        return str(self.options.get("version", DEFAULT_VERSION))

    def _voice_id(self) -> str:
        voice = self.options.get("voice_id")
        if voice:
            return str(voice)
        return require_env("CARTESIA_VOICE_ID", self.key)

    def _api_key(self) -> str:
        return require_env("CARTESIA_API_KEY", self.key)

    def _body(self, prompt: TtsPrompt) -> dict[str, Any]:
        return {
            "model_id": self._model(),
            "transcript": prompt.text,
            "voice": {"mode": "id", "id": self._voice_id()},
            "language": prompt.language.split("-")[0],
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
        }

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Synthesize over HTTP, timing the first body chunk as TTFB."""
        result = self._result(prompt, Mode.BATCH)
        started = time.perf_counter()
        chunks: list[bytes] = []

        async with self.http.stream(
            "POST",
            HTTP_URL,
            headers={
                "X-API-Key": self._api_key(),
                "Cartesia-Version": self._version(),
                "Content-Type": "application/json",
            },
            json=self._body(prompt),
        ) as response:
            raise_for_status(response, self.key)
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                if result.ttfb_s is None:
                    result.ttfb_s = time.perf_counter() - started
                chunks.append(chunk)

        result.total_s = time.perf_counter() - started
        return _finish(result, b"".join(chunks))

    def _ws_url(self) -> str:
        params = {
            "cartesia_version": self._version(),
            "api_key": self._api_key(),
        }
        return f"{WS_URL}?{urlencode(params)}"

    async def _consume(
        self, socket: object, result: TtsResult, timeline: ChunkTimeline
    ) -> None:
        """Drain one WebSocket generation into the timeline."""
        async for raw in socket:  # type: ignore[attr-defined]
            payload = orjson.loads(raw)
            kind = payload.get("type")
            if kind == "chunk":
                timeline.add(base64.b64decode(payload["data"]))
            elif kind == "error":
                result.error = str(payload.get("error", "cartesia stream error"))
                break
            elif kind == "done" or payload.get("done"):
                break

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        """Synthesize over the WebSocket, timing the first audio frame."""
        result = self._result(prompt, Mode.STREAM)
        body = self._body(prompt)
        body["context_id"] = f"{self.key}-{prompt.prompt_id}"

        url = self._ws_url()
        timeline = ChunkTimeline()
        async with connect(url, max_size=None, open_timeout=30.0) as socket:
            await socket.send(orjson.dumps(body).decode())
            await self._consume(socket, result, timeline)

        return _finish_stream(result, timeline)

    async def synthesize_incremental(
        self, prompt: TtsPrompt, *, token_rate: float
    ) -> TtsResult:
        """Feed the transcript in word pieces over one WebSocket context.

        Cartesia's WebSocket accepts continuations: messages sharing a
        ``context_id`` with ``continue: true`` append transcript to the same
        generation, closed by an empty transcript with ``continue: false``.
        That documented input-streaming path is what a voice agent uses while
        its language model is still talking, so this lane measures it.
        """
        result = self._result(prompt, Mode.STREAM)
        result.raw["input_streaming"] = True
        base = self._body(prompt)
        base["context_id"] = f"{self.key}-{prompt.prompt_id}-incremental"

        url = self._ws_url()
        pieces = token_pieces(prompt.text)
        result.raw["text_pieces"] = len(pieces)
        timeline = ChunkTimeline()
        async with connect(url, max_size=None, open_timeout=30.0) as socket:

            async def feed() -> None:
                async for piece in pace_tokens(pieces, token_rate):
                    message = {**base, "transcript": piece, "continue": True}
                    await socket.send(orjson.dumps(message).decode())
                closing = {**base, "transcript": "", "continue": False}
                await socket.send(orjson.dumps(closing).decode())

            feeder = asyncio.create_task(feed())
            try:
                await self._consume(socket, result, timeline)
            finally:
                # A server-side error ends consumption early; the feeder must
                # not keep pacing text into a dead generation, and its own
                # failure must not mask the error already recorded.
                if not feeder.done():
                    feeder.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await feeder

        return _finish_stream(result, timeline)


def _finish(result: TtsResult, audio: bytes) -> TtsResult:
    """Attach audio to a result and derive its duration."""
    result.audio = audio
    result.encoding = "pcm_s16le"
    result.audio_s = decode_audio_duration(
        audio, encoding=result.encoding, sample_rate=result.sample_rate
    )
    return result


def _finish_stream(result: TtsResult, timeline: ChunkTimeline) -> TtsResult:
    """Attach streamed audio and stamp the chunk-timing metrics."""
    _finish(result, timeline.audio)
    stamp_stream_timing(result, timeline)
    return result


@register
class CartesiaSonic3(_CartesiaBase):
    """Cartesia Sonic 3.0."""

    key = "cartesia-sonic3"
    model_id = "sonic-3"


@register
class CartesiaSonic35(_CartesiaBase):
    """Cartesia Sonic 3.5."""

    key = "cartesia-sonic35"
    model_id = "sonic-3.5"
