"""Protocol and live-smoke tests for the Azure Neural TTS adapter.

The Azure Speech SDK talks native code through its own C bindings, so there
is no wire format to pin against a fake server the way the WebSocket-based
adapters are tested. Protocol tests instead substitute a fake
:class:`~audio_harness.tts.azure.SynthesizerSession` for the
``azure.cognitiveservices.speech`` bindings, driving the same
speak/events/close contract the live session uses -- so chunk timing and
result assembly are covered without the SDK installed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import os
from typing import Any

import numpy as np
import pytest

from audio_harness import tts
from audio_harness.tts import azure as azure_tts
from audio_harness.tts.azure import AzureSpeechError, SynthesisEvent
from audio_harness.types import TtsPrompt


RATE = 24000


def make_pcm(seconds: float, rate: int = RATE) -> bytes:
    """Mono 16-bit PCM tone, headerless."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class FakeSynthesizerSession:
    """Drives the same speak/events/close contract the live SDK session uses."""

    def __init__(self, events: list[SynthesisEvent]) -> None:
        self._queued = list(events)
        self.spoken: str | None = None
        self.closed = False

    async def speak(self, text: str) -> None:
        self.spoken = text

    async def events(self) -> AsyncIterator[SynthesisEvent]:
        for event in self._queued:
            yield event
            if event.kind in {"completed", "canceled"}:
                return

    def close(self) -> None:
        self.closed = True


def make_factory(events: list[SynthesisEvent], captured: list[dict[str, Any]], sessions: list[FakeSynthesizerSession]):
    """Build a fake ``_session_factory`` replacement recording its call kwargs."""

    def factory(**kwargs: Any) -> FakeSynthesizerSession:
        captured.append(kwargs)
        session = FakeSynthesizerSession(events)
        sessions.append(session)
        return session

    return factory


PROMPT_EN = TtsPrompt(prompt_id="p1", text="hello wonderful world", language="en-US")
PROMPT_JA = TtsPrompt(prompt_id="p2", text="こんにちは", language="ja-JP")


class TestAzureTtsProtocol:
    @pytest.fixture(autouse=True)
    def azure_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fake credentials for every test in this class.

        Class-scoped, not module-level: ``TestLiveSmoke`` below needs the
        *real* ``AZURE_SPEECH_KEY``/``AZURE_SPEECH_REGION`` from the
        environment and must never see this override.
        """
        monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")

    async def test_stream_populates_chunk_timing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pcm = make_pcm(0.2)
        half = len(pcm) // 2 // 2 * 2
        events = [
            SynthesisEvent(kind="audio", audio=pcm[:half]),
            SynthesisEvent(kind="audio", audio=pcm[half:]),
            SynthesisEvent(kind="completed"),
        ]
        captured: list[dict[str, Any]] = []
        sessions: list[FakeSynthesizerSession] = []
        monkeypatch.setattr(azure_tts, "_session_factory", make_factory(events, captured, sessions))

        adapter = tts.create("azure-neural-tts")
        result = await adapter.synthesize_stream(PROMPT_EN)

        assert result.ok, result.error
        assert len(result.chunk_t_s) == 2
        assert result.ttfb_s is not None
        assert result.audio_s > 0
        assert result.raw["sdk_buffered"] is True
        assert sessions[0].spoken == PROMPT_EN.text
        assert sessions[0].closed

    async def test_english_prompt_uses_default_en_voice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [SynthesisEvent(kind="completed")]
        captured: list[dict[str, Any]] = []
        sessions: list[FakeSynthesizerSession] = []
        monkeypatch.setattr(azure_tts, "_session_factory", make_factory(events, captured, sessions))

        adapter = tts.create("azure-neural-tts")
        result = await adapter.synthesize_stream(PROMPT_EN)

        assert captured[0]["voice"] == azure_tts.DEFAULT_VOICE_EN
        assert result.raw["voice"] == azure_tts.DEFAULT_VOICE_EN

    async def test_japanese_prompt_uses_default_ja_voice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [SynthesisEvent(kind="completed")]
        captured: list[dict[str, Any]] = []
        sessions: list[FakeSynthesizerSession] = []
        monkeypatch.setattr(azure_tts, "_session_factory", make_factory(events, captured, sessions))

        adapter = tts.create("azure-neural-tts")
        result = await adapter.synthesize_stream(PROMPT_JA)

        assert captured[0]["voice"] == azure_tts.DEFAULT_VOICE_JA
        assert result.raw["voice"] == azure_tts.DEFAULT_VOICE_JA

    async def test_voice_options_override_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [SynthesisEvent(kind="completed")]
        captured: list[dict[str, Any]] = []
        sessions: list[FakeSynthesizerSession] = []
        monkeypatch.setattr(azure_tts, "_session_factory", make_factory(events, captured, sessions))

        adapter = tts.create("azure-neural-tts", {"voice_en": "en-US-JennyNeural", "voice_ja": "ja-JP-KeitaNeural"})
        await adapter.synthesize_stream(PROMPT_EN)
        await adapter.synthesize_stream(PROMPT_JA)

        assert captured[0]["voice"] == "en-US-JennyNeural"
        assert captured[1]["voice"] == "ja-JP-KeitaNeural"

    async def test_canceled_with_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [SynthesisEvent(kind="canceled", error="AuthenticationFailure: bad key")]
        captured: list[dict[str, Any]] = []
        sessions: list[FakeSynthesizerSession] = []
        monkeypatch.setattr(azure_tts, "_session_factory", make_factory(events, captured, sessions))

        adapter = tts.create("azure-neural-tts")
        with pytest.raises(AzureSpeechError, match="AuthenticationFailure"):
            await adapter.synthesize_stream(PROMPT_EN)

        assert sessions[0].closed, "session must still be closed after an error"

    async def test_unsupported_sample_rate_rejected(self) -> None:
        adapter = tts.create("azure-neural-tts", {"sample_rate": 11025})

        with pytest.raises(ValueError, match="unsupported sample rate"):
            await adapter.synthesize_stream(PROMPT_EN)


LIVE_FLAG = "AUDIO_HARNESS_TEST_AZURE_LIVE"


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run a couple of short live syntheses against Azure Neural TTS (fractions of a cent total)",
)
class TestLiveSmoke:
    """A handful of short prompts against the real Azure Neural TTS endpoint.

    Minimal-API-testing policy: one English and one Japanese prompt only --
    no bulk runs here. Requires ``AZURE_SPEECH_KEY`` and
    ``AZURE_SPEECH_REGION`` in the environment.
    """

    async def test_stream_en(self) -> None:
        adapter = tts.create("azure-neural-tts")
        try:
            result = await adapter.synthesize_stream(
                TtsPrompt(prompt_id="live-en", text="What a lovely day for a test.", language="en-US")
            )
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0
        assert result.raw["sdk_buffered"] is True

    async def test_stream_ja(self) -> None:
        adapter = tts.create("azure-neural-tts")
        try:
            result = await adapter.synthesize_stream(
                TtsPrompt(prompt_id="live-ja", text="こんにちは、今日はいい天気ですね。", language="ja-JP")
            )
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0
        assert result.raw["sdk_buffered"] is True
