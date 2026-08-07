"""Mistral Voxtral Mini text-to-speech adapter."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
import time
from typing import Any

import httpx
import orjson

from audio_harness.audio import decode_audio_duration, pcm_f32le_to_s16le
from audio_harness.config import require_env
from audio_harness.stt.base import raise_for_status
from audio_harness.types import Mode, TtsPrompt, TtsResult

from .base import ChunkTimeline, TtsProvider, register, stamp_stream_timing


SPEECH_URL = "https://api.mistral.ai/v1/audio/speech"

VOXTRAL_SAMPLE_RATE = 24000
"""Fixed rate of the vendor's float32 ``pcm`` response format."""

SUPPORTED_LANGUAGES = frozenset({"ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"})


@register
class MistralVoxtralTts(TtsProvider):
    """Voxtral Mini TTS batch and SSE streaming, for nine languages.

    The endpoint returns base64-encoded float32 little-endian mono PCM inside
    JSON, including within ``speech.audio.delta`` SSE events. The harness
    converts it to its canonical signed 16-bit PCM before measurement.

    Japanese is not supported. Calls accept only ``ar``, ``de``, ``en``,
    ``es``, ``fr``, ``hi``, ``it``, ``nl`` and ``pt`` primary language tags.

    Options:
        model: Model identifier; defaults to ``voxtral-mini-tts-2603``.
        voice_id: Voice identifier; defaults to ``en_paul_neutral``.

    Pricing as of 2026-08-08 is $16.00 per million characters ($0.016/1K).
    """

    key = "mistral-voxtral-tts"
    vendor = "mistral"
    family = "mistral"
    supports_batch = True
    supports_stream = True
    supports_input_streaming = False
    default_sample_rate = VOXTRAL_SAMPLE_RATE

    @property
    def sample_rate(self) -> int:
        """The wire format's fixed sample rate."""
        return VOXTRAL_SAMPLE_RATE

    def _model(self) -> str:
        return str(self.options.get("model", "voxtral-mini-tts-2603"))

    def _voice_id(self) -> str:
        return str(self.options.get("voice_id", "en_paul_neutral"))

    def _headers(self) -> dict[str, str]:
        key = require_env("MISTRAL_API_KEY", self.key)
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _body(self, prompt: TtsPrompt, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model(),
            "input": prompt.text,
            "voice_id": self._voice_id(),
            "response_format": "pcm",
        }
        if stream:
            body["stream"] = True
        return body

    def _validate_language(self, prompt: TtsPrompt) -> None:
        language = prompt.language.split("-", 1)[0].lower()
        if language not in SUPPORTED_LANGUAGES:
            accepted = ", ".join(sorted(SUPPORTED_LANGUAGES))
            raise ValueError(
                f"{self.key}: unsupported language {prompt.language!r}; expected a primary language tag in {accepted}"
            )

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Request and convert one complete JSON-wrapped PCM response."""
        self._validate_language(prompt)
        result = self._result(prompt, Mode.BATCH)
        started = time.perf_counter()
        response = await self.http.post(SPEECH_URL, headers=self._headers(), json=self._body(prompt, stream=False))
        raise_for_status(response, self.key)
        result.ttfb_s = None
        result.total_s = time.perf_counter() - started
        payload = orjson.loads(response.content)
        audio = pcm_f32le_to_s16le(base64.b64decode(payload["audio_data"], validate=True))
        return _finish(result, audio)

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        """Request SSE audio deltas, convert each chunk and stamp arrivals."""
        self._validate_language(prompt)
        result = self._result(prompt, Mode.STREAM)
        timeline = ChunkTimeline()

        async with self.http.stream(
            "POST", SPEECH_URL, headers=self._headers(), json=self._body(prompt, stream=True)
        ) as response:
            if response.status_code >= 400:
                await response.aread()
            raise_for_status(response, self.key)
            done = False
            async for event, data in _iter_sse_events(response):
                if event == "speech.audio.done":
                    done = True
                    break
                if event != "speech.audio.delta":
                    continue
                payload = orjson.loads(data)
                chunk = base64.b64decode(payload["audio_data"], validate=True)
                timeline.add(pcm_f32le_to_s16le(chunk))
            if not done:
                raise RuntimeError(f"{self.key}: SSE stream ended before speech.audio.done")

        return _finish_stream(result, timeline)


async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """Yield standard SSE event names and joined data fields."""
    event = "message"
    data: list[str] = []
    first_line = True
    async for line in response.aiter_lines():
        if first_line:
            line = line.removeprefix("\ufeff")
            first_line = False
        if not line:
            if data or event != "message":
                yield event, "\n".join(data)
            event = "message"
            data.clear()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data or event != "message":
        yield event, "\n".join(data)


def _finish(result: TtsResult, audio: bytes) -> TtsResult:
    """Attach converted audio and document the vendor's wire representation."""
    result.audio = audio
    result.encoding = "pcm_s16le"
    result.audio_s = decode_audio_duration(audio, encoding=result.encoding, sample_rate=result.sample_rate)
    result.raw["wire_format"] = "f32le_json"
    return result


def _finish_stream(result: TtsResult, timeline: ChunkTimeline) -> TtsResult:
    """Attach streamed audio and stamp the chunk-timing metrics."""
    _finish(result, timeline.audio)
    stamp_stream_timing(result, timeline)
    return result
