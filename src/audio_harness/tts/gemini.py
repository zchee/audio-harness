"""Gemini text-to-speech adapter."""

from __future__ import annotations

import inspect
import os
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from ..audio import decode_audio_duration
from ..config import require_env
from ..types import Mode, TtsPrompt, TtsResult
from .base import TtsProvider, register

GEMINI_SAMPLE_RATE = 24000
"""Gemini TTS emits 24 kHz mono 16-bit PCM regardless of what is requested."""


@register
class GeminiTts(TtsProvider):
    """Gemini TTS through the google-genai SDK.

    Options:
        model: Model identifier; defaults to ``gemini-2.5-flash-preview-tts``.
        voice: Prebuilt voice name such as ``Kore`` or ``Puck``.
    """

    key = "gemini-tts"
    vendor = "gemini"
    supports_batch = True
    supports_stream = True
    default_sample_rate = GEMINI_SAMPLE_RATE

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        """Initialize the adapter and defer client construction."""
        super().__init__(options)
        self._client: genai.Client | None = None

    def _genai_client(self) -> genai.Client:
        """Return a lazily constructed genai client."""
        if self._client is None:
            self._client = genai.Client(api_key=require_env("GEMINI_API_KEY", self.key))
        return self._client

    def _model(self) -> str:
        return str(self.options.get("model", "gemini-2.5-flash-preview-tts"))

    def _config(self) -> genai_types.GenerateContentConfig:
        voice = str(
            self.options.get("voice") or os.environ.get("GEMINI_TTS_VOICE", "Kore")
        )
        return genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                )
            ),
        )

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Generate the whole utterance in one non-streaming call."""
        result = self._result(prompt, Mode.BATCH)
        started = time.perf_counter()
        response = await self._genai_client().aio.models.generate_content(
            model=self._model(), contents=prompt.text, config=self._config()
        )
        result.total_s = time.perf_counter() - started
        return _finish(result, b"".join(_extract_audio(response)))

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        """Generate incrementally, stamping the first audio part as TTFB."""
        result = self._result(prompt, Mode.STREAM)
        started = time.perf_counter()
        chunks: list[bytes] = []

        stream = self._genai_client().aio.models.generate_content_stream(
            model=self._model(), contents=prompt.text, config=self._config()
        )
        if inspect.isawaitable(stream):
            stream = await stream

        async for response in stream:
            for audio in _extract_audio(response):
                if result.ttfb_s is None:
                    result.ttfb_s = time.perf_counter() - started
                chunks.append(audio)

        result.total_s = time.perf_counter() - started
        return _finish(result, b"".join(chunks))


def _extract_audio(response: Any) -> list[bytes]:
    """Pull inline audio payloads out of a generate-content response.

    Gemini returns audio as inline data parts alongside optional text parts, so
    non-audio parts are skipped rather than treated as an error.
    """
    audio: list[bytes] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None)
            if data:
                audio.append(data)
    return audio


def _finish(result: TtsResult, audio: bytes) -> TtsResult:
    """Attach audio to a result and derive its duration."""
    result.audio = audio
    result.encoding = "pcm_s16le"
    result.sample_rate = GEMINI_SAMPLE_RATE
    result.audio_s = decode_audio_duration(
        audio, encoding=result.encoding, sample_rate=GEMINI_SAMPLE_RATE
    )
    return result
