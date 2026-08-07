"""Registration and configuration tests for Gemini TTS adapters."""

from __future__ import annotations

import os

import pytest

from audio_harness import tts
from audio_harness.tts import gemini
from audio_harness.types import TtsPrompt


def test_gemini_31_registration_and_inherited_capabilities() -> None:
    adapter = tts.create("gemini-tts-31")

    assert isinstance(adapter, gemini.GeminiTts31)
    assert isinstance(adapter, gemini.GeminiTts)
    assert adapter._model() == "gemini-3.1-flash-tts-preview"
    assert adapter.supports_batch is gemini.GeminiTts.supports_batch
    assert adapter.supports_stream is gemini.GeminiTts.supports_stream
    assert adapter.supports_input_streaming is gemini.GeminiTts.supports_input_streaming
    assert adapter.default_sample_rate == gemini.GeminiTts.default_sample_rate == gemini.GEMINI_SAMPLE_RATE


def test_gemini_31_model_option_overrides_the_default() -> None:
    adapter = tts.create("gemini-tts-31", {"model": "custom-tts-model"})

    assert isinstance(adapter, gemini.GeminiTts31)
    assert adapter._model() == "custom-tts-model"


LIVE_FLAG = "AUDIO_HARNESS_TEST_GEMINI_TTS_LIVE"


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG) or not os.environ.get("GEMINI_API_KEY"),
    reason=f"live smoke needs GEMINI_API_KEY and {LIVE_FLAG}=1",
)
class TestLiveSmoke:
    """One short real generation through the 3.1 preview model."""

    async def test_batch_en(self) -> None:
        adapter = tts.create("gemini-tts-31")
        try:
            result = await adapter.synthesize(
                TtsPrompt(prompt_id="smoke-gemini-31-en", text="Good evening.", language="en-US")
            )
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0
