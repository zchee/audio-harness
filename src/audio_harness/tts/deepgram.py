"""Deepgram Aura-2 text-to-speech adapter."""

from __future__ import annotations

import os
import time
from urllib.parse import urlencode

from ..audio import decode_audio_duration
from ..config import require_env
from ..stt.base import raise_for_status
from ..types import Mode, TtsPrompt, TtsResult
from .base import TtsProvider, register

SPEAK_URL = "https://api.deepgram.com/v1/speak"


@register
class DeepgramAura2(TtsProvider):
    """Deepgram Aura-2 over the speak endpoint.

    The endpoint delivers audio with chunked transfer encoding, so both modes
    hit the same URL and differ only in how the body is consumed: the batch
    mode reads it to completion, the streaming mode stamps the first chunk.

    Options:
        model: Voice-qualified model id such as ``aura-2-thalia-en``.
        sample_rate: Output rate; defaults to 24 kHz.
    """

    key = "deepgram-aura2"
    vendor = "deepgram"
    supports_batch = True
    supports_stream = True

    def _model(self) -> str:
        model = self.options.get("model")
        if model:
            return str(model)
        return os.environ.get("DEEPGRAM_TTS_VOICE", "aura-2-thalia-en")

    def _url(self) -> str:
        params = {
            "model": self._model(),
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
        }
        return f"{SPEAK_URL}?{urlencode(params)}"

    def _headers(self) -> dict[str, str]:
        key = require_env("DEEPGRAM_API_KEY", self.key)
        return {"Authorization": f"Token {key}", "Content-Type": "application/json"}

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Request audio and read the whole body before returning."""
        result = self._result(prompt, Mode.BATCH)
        started = time.perf_counter()
        response = await self.http.post(
            self._url(), headers=self._headers(), json={"text": prompt.text}
        )
        raise_for_status(response, self.key)
        result.ttfb_s = None
        result.total_s = time.perf_counter() - started
        return _finish(result, response.content)

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        """Request audio and stamp the first chunk of the response body."""
        result = self._result(prompt, Mode.STREAM)
        started = time.perf_counter()
        chunks: list[bytes] = []

        async with self.http.stream(
            "POST", self._url(), headers=self._headers(), json={"text": prompt.text}
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


def _finish(result: TtsResult, audio: bytes) -> TtsResult:
    """Attach audio to a result and derive its duration."""
    result.audio = audio
    result.encoding = "pcm_s16le"
    result.audio_s = decode_audio_duration(
        audio, encoding=result.encoding, sample_rate=result.sample_rate
    )
    return result
