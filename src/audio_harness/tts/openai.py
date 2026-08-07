"""OpenAI gpt-4o-mini-tts text-to-speech adapter."""

from __future__ import annotations

import os
import time
from typing import Any

from audio_harness.audio import decode_audio_duration
from audio_harness.config import require_env
from audio_harness.stt.base import raise_for_status
from audio_harness.types import Mode, TtsPrompt, TtsResult

from .base import ChunkTimeline, TtsProvider, register, stamp_stream_timing


SPEECH_URL = "https://api.openai.com/v1/audio/speech"

DEFAULT_VOICE = "alloy"

REALTIME_SAMPLE_RATE = 24000
"""Fixed rate of the vendor's headerless ``pcm`` response format."""


@register
class OpenAiGpt4oMiniTts(TtsProvider):
    """gpt-4o-mini-tts over the ``/v1/audio/speech`` HTTP endpoint.

    The default is pinned to the dated ``gpt-4o-mini-tts-2025-12-15``
    snapshot so TTS scores remain reproducible across runs. The floating
    ``gpt-4o-mini-tts`` alias remains available through the ``model_id``
    option when tracking the vendor's current alias is desired.

    ``response_format=pcm`` returns headerless 16-bit PCM at a fixed 24 kHz —
    same reasoning as Inworld's ``PCM`` choice, so no WAV header bytes get
    counted as audio. The plain endpoint answers with the complete body in
    one response and carries no TTFB signal; ``stream_format=audio`` switches
    the same endpoint to raw chunked bytes (not the ``sse``-wrapped variant,
    which would require decoding base64 out of JSON events for no benefit
    here), matching the chunk-timing shape every other streaming adapter uses.

    Options:
        voice_id: Voice name, e.g. ``alloy``, ``ash``, ``coral``, ``sage``.
            Falls back to the ``OPENAI_VOICE_ID`` environment variable, then
            ``alloy``. Pin this across models; comparing two models on
            different voices measures the voices, not the models.
        model_id: Defaults to ``gpt-4o-mini-tts-2025-12-15``.
        instructions: Optional free-text voice-control directive (tone,
            pacing, accent). Unsupported by the legacy ``tts-1`` models.
        speed: Playback rate, 0.25 to 4.0; defaults to 1.0.
    """

    key = "openai-gpt4o-mini-tts"
    vendor = "openai"
    family = "openai"
    supports_batch = True
    supports_stream = True
    default_sample_rate = REALTIME_SAMPLE_RATE

    def _model(self) -> str:
        return str(self.options.get("model_id", "gpt-4o-mini-tts-2025-12-15"))

    def _voice(self) -> str:
        voice = self.options.get("voice_id")
        if voice:
            return str(voice)
        return os.environ.get("OPENAI_VOICE_ID", DEFAULT_VOICE)

    def _api_key(self) -> str:
        return require_env("OPENAI_API_KEY", self.key)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}

    def _body(self, prompt: TtsPrompt, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model(),
            "input": prompt.text,
            "voice": self._voice(),
            "response_format": "pcm",
        }
        instructions = self.options.get("instructions")
        if instructions:
            body["instructions"] = str(instructions)
        speed = self.options.get("speed")
        if speed is not None:
            body["speed"] = float(speed)
        if stream:
            body["stream_format"] = "audio"
        return body

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Request audio from the plain endpoint and read the whole body."""
        result = self._result(prompt, Mode.BATCH)
        started = time.perf_counter()
        response = await self.http.post(SPEECH_URL, headers=self._headers(), json=self._body(prompt, stream=False))
        raise_for_status(response, self.key)
        result.ttfb_s = None
        result.total_s = time.perf_counter() - started
        return _finish(result, response.content)

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        """Request audio with ``stream_format=audio`` and stamp every chunk."""
        result = self._result(prompt, Mode.STREAM)
        timeline = ChunkTimeline()

        async with self.http.stream(
            "POST", SPEECH_URL, headers=self._headers(), json=self._body(prompt, stream=True)
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
