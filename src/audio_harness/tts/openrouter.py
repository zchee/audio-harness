"""OpenRouter-hosted text-to-speech adapters."""

from __future__ import annotations

import time
from typing import ClassVar

from audio_harness.audio import decode_audio_duration, decode_container_pcm16
from audio_harness.config import require_env
from audio_harness.stt.base import raise_for_status
from audio_harness.types import Mode, TtsPrompt, TtsResult

from .base import TtsProvider, register


SPEECH_URL = "https://openrouter.ai/api/v1/audio/speech"


class OpenRouterTts(TtsProvider):
    """Batch TTS through OpenRouter's OpenAI-compatible speech endpoint.

    Subclasses pin a registry key, model slug, and a voice accepted by that
    model. The ``voice`` option overrides the pinned default.

    Options:
        voice: Model-specific voice identifier.
    """

    vendor = "openrouter"
    family = "openrouter"
    supports_batch = True
    supports_stream = False
    model: ClassVar[str]
    default_voice: ClassVar[str]

    def _auth(self) -> dict[str, str]:
        """Build the OpenRouter bearer-authentication header."""
        return {"Authorization": f"Bearer {require_env('OPENROUTER_API_KEY', self.key)}"}

    def _voice(self) -> str:
        """Return the configured voice or this model's supported default."""
        return str(self.options.get("voice", self.default_voice))

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Synthesize MP3 audio, then decode it to native-rate mono PCM."""
        result = self._result(prompt, Mode.BATCH)
        result.raw["hosted_proxy"] = True
        body: dict[str, str] = {
            "model": self.model,
            "input": prompt.text,
            "response_format": "mp3",
        }
        # An empty default means the model picks its own voice (Fish Audio
        # publishes no supported_voices and accepts voiceless requests);
        # models with a fixed enum always get one.
        if voice := self._voice():
            body["voice"] = voice
        started = time.perf_counter()
        response = await self.http.post(SPEECH_URL, headers=self._auth(), json=body)
        raise_for_status(response, self.key)
        result.total_s = time.perf_counter() - started
        result.ttfb_s = None

        pcm, sample_rate = decode_container_pcm16(response.content)
        result.audio = pcm
        result.encoding = "pcm_s16le"
        result.sample_rate = sample_rate
        result.audio_s = decode_audio_duration(pcm, encoding=result.encoding, sample_rate=sample_rate)
        result.raw["wire_format"] = "mp3"
        generation_id = response.headers.get("X-Generation-Id")
        if generation_id:
            result.raw["generation_id"] = generation_id
        return result


@register
class OpenRouterKokoro(OpenRouterTts):
    """OpenRouter-hosted Hexgrad Kokoro 82M."""

    key = "or-kokoro"
    model = "hexgrad/kokoro-82m"
    default_voice = "af_heart"


@register
class OpenRouterOrpheus(OpenRouterTts):
    """OpenRouter-hosted Canopy Labs Orpheus 3B."""

    key = "or-orpheus"
    model = "canopylabs/orpheus-3b-0.1-ft"
    default_voice = "tara"


@register
class OpenRouterCsm(OpenRouterTts):
    """OpenRouter-hosted Sesame CSM 1B."""

    key = "or-csm"
    model = "sesame/csm-1b"
    default_voice = "conversational_a"


@register
class OpenRouterZonos(OpenRouterTts):
    """OpenRouter-hosted Zyphra Zonos v0.1 Hybrid."""

    key = "or-zonos"
    model = "zyphra/zonos-v0.1-hybrid"
    default_voice = "american_female"


@register
class OpenRouterMiniMaxTurbo(OpenRouterTts):
    """OpenRouter-hosted MiniMax Speech-2.8 Turbo."""

    key = "or-minimax-turbo"
    model = "minimax/speech-2.8-turbo"
    default_voice = "English_expressive_narrator"


@register
class OpenRouterMiniMaxHd(OpenRouterTts):
    """OpenRouter-hosted MiniMax Speech-2.8 HD."""

    key = "or-minimax-hd"
    model = "minimax/speech-2.8-hd"
    default_voice = "English_expressive_narrator"


@register
class OpenRouterQwenTtsFlash(OpenRouterTts):
    """OpenRouter-hosted Qwen-Audio-3.0-TTS Flash.

    The direct DashScope-international lane stays blocked on an account
    model grant (Model.AccessDenied), so this hosted route is how the
    Qwen voice family gets measured at all.
    """

    key = "or-qwen-tts-flash"
    model = "qwen/qwen-audio-3.0-tts-flash"
    default_voice = "loongjohn"


@register
class OpenRouterQwenTtsPlus(OpenRouterTts):
    """OpenRouter-hosted Qwen-Audio-3.0-TTS Plus."""

    key = "or-qwen-tts-plus"
    model = "qwen/qwen-audio-3.0-tts-plus"
    default_voice = "longanlingxin"


@register
class OpenRouterFishS1(OpenRouterTts):
    """OpenRouter-hosted Fish Audio S1.

    Fish publishes no voice enum and accepts voiceless requests (verified
    live 2026-08-08), so these lanes default to the model's own voice.
    """

    key = "or-fish-s1"
    model = "fish-audio/s1"
    default_voice = ""


@register
class OpenRouterFishS2Pro(OpenRouterTts):
    """OpenRouter-hosted Fish Audio S2 Pro."""

    key = "or-fish-s2-pro"
    model = "fish-audio/s2-pro"
    default_voice = ""


@register
class OpenRouterFishS21Pro(OpenRouterTts):
    """OpenRouter-hosted Fish Audio S2.1 Pro."""

    key = "or-fish-s21-pro"
    model = "fish-audio/s2.1-pro"
    default_voice = ""


@register
class OpenRouterMaiVoice2(OpenRouterTts):
    """OpenRouter-hosted Microsoft MAI Voice 2.

    An explicit voice is mandatory (400 without one; verified live
    2026-08-12) and the accepted identifiers are Azure-style neural voice
    names — en-US-AvaNeural and ja-JP-NanamiNeural both probe 200. Japanese
    configs must override ``voice`` accordingly.
    """

    key = "or-mai-voice-2"
    model = "microsoft/mai-voice-2"
    default_voice = "en-US-AvaNeural"


@register
class OpenRouterMaiVoice2Flash(OpenRouterTts):
    """OpenRouter-hosted Microsoft MAI Voice 2 Flash."""

    key = "or-mai-voice-2-flash"
    model = "microsoft/mai-voice-2-flash"
    default_voice = "en-US-AvaNeural"
