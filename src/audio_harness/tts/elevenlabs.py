"""ElevenLabs Flash v2.5 text-to-speech adapter."""

from __future__ import annotations

import asyncio
import base64
import contextlib
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


BATCH_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"

_PCM_FORMATS = {
    8000: "pcm_8000",
    16000: "pcm_16000",
    22050: "pcm_22050",
    24000: "pcm_24000",
    32000: "pcm_32000",
    44100: "pcm_44100",
    48000: "pcm_48000",
}
"""output_format values the vendor accepts for headerless PCM; pcm_44100 and
above require a Pro-tier account (docs: elevenlabs.io/docs/api-reference/
text-to-speech/convert)."""


@register
class ElevenLabsFlash25(TtsProvider):
    """ElevenLabs Flash v2.5, batch, HTTP streaming and WS stream-input.

    Flash v2.5 is the low-latency multilingual model (~75 ms model latency,
    32 languages including the ``eleven_multilingual_v2`` set plus Hungarian,
    Norwegian and Vietnamese). Three transports are exercised: the plain
    endpoint for :meth:`synthesize`, the chunked ``/stream`` endpoint for
    :meth:`synthesize_stream`, and the ``stream-input`` WebSocket — the one
    wire protocol that actually accepts text appended to an open generation —
    for :meth:`synthesize_incremental`.

    Options:
        voice_id: Voice id. Pin this across models; comparing two models on
            different voices measures the voices, not the models.
        model_id: Defaults to ``eleven_flash_v2_5``.
        sample_rate: Output rate; must be one of the vendor's PCM formats
            (8000, 16000, 22050, 24000, 32000, 44100, 48000). Defaults to
            24000.
    """

    key = "elevenlabs-flash25"
    vendor = "elevenlabs"
    supports_batch = True
    supports_stream = True
    supports_input_streaming = True

    def _model(self) -> str:
        return str(self.options.get("model_id", "eleven_flash_v2_5"))

    def _voice_id(self) -> str:
        voice = self.options.get("voice_id")
        if voice:
            return str(voice)
        return require_env("ELEVENLABS_VOICE_ID", self.key)

    def _api_key(self) -> str:
        return require_env("ELEVENLABS_API_KEY", self.key)

    def _auth(self) -> dict[str, str]:
        return {"xi-api-key": self._api_key()}

    def _headers(self) -> dict[str, str]:
        return {**self._auth(), "Content-Type": "application/json"}

    def _output_format(self) -> str:
        rate = self.sample_rate
        fmt = _PCM_FORMATS.get(rate)
        if fmt is None:
            raise ValueError(f"{self.key}: unsupported sample rate {rate}; expected one of {sorted(_PCM_FORMATS)}")
        return fmt

    def _body(self, prompt: TtsPrompt) -> dict[str, Any]:
        return {
            "text": prompt.text,
            "model_id": self._model(),
            # Ignored by the vendor rather than rejected when a model does not
            # support the code, so this is safe across the whole prompt set.
            "language_code": prompt.language.split("-")[0],
        }

    def _url(self, template: str) -> str:
        base = template.format(voice_id=self._voice_id())
        return f"{base}?{urlencode({'output_format': self._output_format()})}"

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Request audio from the plain endpoint and read the whole body."""
        result = self._result(prompt, Mode.BATCH)
        started = time.perf_counter()
        response = await self.http.post(self._url(BATCH_URL), headers=self._headers(), json=self._body(prompt))
        raise_for_status(response, self.key)
        result.ttfb_s = None
        result.total_s = time.perf_counter() - started
        return _finish(result, response.content)

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        """Request audio from the ``/stream`` endpoint and stamp every chunk."""
        result = self._result(prompt, Mode.STREAM)
        timeline = ChunkTimeline()

        async with self.http.stream(
            "POST", self._url(STREAM_URL), headers=self._headers(), json=self._body(prompt)
        ) as response:
            if response.status_code >= 400:
                # raise_for_status reads response.text; on a streamed response
                # that raises ResponseNotRead unless the body is buffered
                # first, which would replace the vendor's error with a
                # confusing one about our own HTTP client.
                await response.aread()
            raise_for_status(response, self.key)
            async for chunk in response.aiter_bytes():
                timeline.add(chunk)

        return _finish_stream(result, timeline)

    def _ws_url(self, prompt: TtsPrompt) -> str:
        params = {
            "model_id": self._model(),
            "output_format": self._output_format(),
            "language_code": prompt.language.split("-")[0],
        }
        return f"{WS_URL.format(voice_id=self._voice_id())}?{urlencode(params)}"

    async def _consume(self, socket: ClientConnection, result: TtsResult, timeline: ChunkTimeline) -> None:
        """Drain one stream-input session into the timeline.

        The vendor's only documented terminal messages are an audio chunk
        (optionally alongside alignment data) and a closing ``isFinal``
        marker; an ``error`` key is checked defensively since the wire error
        shape is not spelled out in the published schema.
        """
        async for raw in socket:
            payload = orjson.loads(raw)
            audio = payload.get("audio")
            if audio:
                timeline.add(base64.b64decode(audio))
            error = payload.get("error")
            if error:
                result.error = str(error)
                break
            if payload.get("isFinal"):
                break

    async def synthesize_incremental(self, prompt: TtsPrompt, *, token_rate: float) -> TtsResult:
        """Feed the transcript in word pieces over the stream-input WebSocket.

        ``stream-input`` is ElevenLabs' documented protocol for text arriving
        incrementally: each message appends a fragment to the open generation,
        and the vendor's own chunk-length schedule decides when enough text
        has accumulated to synthesize. That is what a voice agent's client
        does while its language model is still talking, so this lane measures
        it rather than simulating it client-side.
        """
        result = self._result(prompt, Mode.STREAM)
        result.raw["input_streaming"] = True
        pieces = token_pieces(prompt.text)
        result.raw["text_pieces"] = len(pieces)
        timeline = ChunkTimeline()

        async with connect(
            self._ws_url(prompt), additional_headers=self._auth(), max_size=None, open_timeout=30.0
        ) as socket:
            await socket.send(orjson.dumps({"text": " "}).decode())

            async def feed() -> None:
                async for piece in pace_tokens(pieces, token_rate):
                    await socket.send(orjson.dumps({"text": piece}).decode())
                await socket.send(orjson.dumps({"text": ""}).decode())

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
    result.audio_s = decode_audio_duration(audio, encoding=result.encoding, sample_rate=result.sample_rate)
    return result


def _finish_stream(result: TtsResult, timeline: ChunkTimeline) -> TtsResult:
    """Attach streamed audio and stamp the chunk-timing metrics."""
    _finish(result, timeline.audio)
    stamp_stream_timing(result, timeline)
    return result
