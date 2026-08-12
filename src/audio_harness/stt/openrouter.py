"""OpenRouter-hosted speech-to-text adapters."""

from __future__ import annotations

import time
from typing import Any, ClassVar

from audio_harness.audio import wrap_wav
from audio_harness.config import require_env
from audio_harness.types import AudioClip, Mode, SttResult

from .base import SttProvider, raise_for_status, register


TRANSCRIPTIONS_URL = "https://openrouter.ai/api/v1/audio/transcriptions"


class OpenRouterStt(SttProvider):
    """Batch STT through OpenRouter's OpenAI-compatible transcription endpoint.

    Subclasses pin the model slug. Optional OpenAI-compatible multipart fields
    are forwarded only when configured, leaving the default request as the two
    required parts: ``file`` and ``model``.

    Options:
        language: ISO-639-1 language code.
        response_format: ``json`` or a provider-supported alternative.
        temperature: Sampling temperature between zero and one.
        timestamp_granularities: String or list of timestamp granularities.
    """

    vendor = "openrouter"
    family = "openrouter"
    supports_batch = True
    supports_stream = False
    model: ClassVar[str]

    def _auth(self) -> dict[str, str]:
        """Build the OpenRouter bearer-authentication header."""
        return {"Authorization": f"Bearer {require_env('OPENROUTER_API_KEY', self.key)}"}

    def _form_data(self) -> dict[str, Any]:
        """Build the supported multipart text fields with exact wire names."""
        data: dict[str, Any] = {"model": self.model}
        for name in ("language", "response_format", "temperature"):
            value = self.options.get(name)
            if value is not None:
                data[name] = str(value)

        granularities = self.options.get("timestamp_granularities")
        if isinstance(granularities, str):
            data["timestamp_granularities[]"] = granularities
        elif isinstance(granularities, list | tuple):
            data["timestamp_granularities[]"] = [str(value) for value in granularities]
        return data

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Upload a WAV-wrapped clip and extract the JSON transcript."""
        result = self._result(clip, Mode.BATCH)
        result.raw["hosted_proxy"] = True
        started = time.perf_counter()
        response = await self.http.post(
            TRANSCRIPTIONS_URL,
            headers=self._auth(),
            data=self._form_data(),
            files={"file": ("audio.wav", wrap_wav(clip.pcm, clip.sample_rate), "audio/wav")},
        )
        raise_for_status(response, self.key)
        payload = response.json()
        result.total_s = time.perf_counter() - started
        result.text = str(payload.get("text", ""))
        result.raw["response"] = payload
        return result


@register
class OpenRouterParakeet(OpenRouterStt):
    """OpenRouter-hosted NVIDIA Parakeet TDT 0.6B v3."""

    key = "or-parakeet"
    model = "nvidia/parakeet-tdt-0.6b-v3"


@register
class OpenRouterMaiTranscribe(OpenRouterStt):
    """OpenRouter-hosted Microsoft MAI transcribe 1.5.

    Probe-verified 2026-08-12 ($0.0001/s in the usage payload). Hosted-proxy
    caveats apply; synthetic corpora only — OpenRouter is not on the
    real-data vendor allowlist.
    """

    key = "or-mai-transcribe"
    model = "microsoft/mai-transcribe-1.5"


@register
class OpenRouterFishTranscribe(OpenRouterStt):
    """OpenRouter-hosted Fish Audio transcribe-1.

    Probe-verified 2026-08-12 (200 on the transcription endpoint; usage
    bills per second at $0.0001/s). Hosted-proxy caveats apply as with
    every OpenRouter lane; synthetic corpora only — OpenRouter is not on
    the real-data vendor allowlist.
    """

    key = "or-fish-transcribe"
    model = "fish-audio/transcribe-1"
