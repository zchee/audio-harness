"""Protocol and live-smoke tests for the Azure Speech STT adapter.

The Azure Speech SDK talks native code through its own C bindings, so there
is no wire format to pin against a fake server the way the WebSocket-based
adapters are tested. Protocol tests instead substitute a fake
:class:`~audio_harness.stt.azure.RecognizerSession` for the
``azure.cognitiveservices.speech`` bindings, driving the same
write/finish_input/events/close contract the live session uses -- so event
routing, EOU capture and result assembly are covered without the SDK
installed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import os
from typing import Any

import numpy as np
import pytest

from audio_harness import stt
from audio_harness.stt import azure as azure_stt
from audio_harness.stt.azure import AzureSpeechError, SdkEvent
from audio_harness.types import AudioClip, EventKind


RATE = 16000


def make_clip(
    seconds: float = 0.2,
    rate: int = RATE,
    clip_id: str = "c1",
    language: str = "en-US",
) -> AudioClip:
    """Mono 16-bit PCM tone clip, headerless."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return AudioClip(
        clip_id=clip_id,
        pcm=pcm,
        sample_rate=rate,
        duration_s=seconds,
        reference=None,
        language=language,
        source_path="",
    )


class FakeRecognizerSession:
    """Drives the same event/queue contract the live SDK-backed session uses."""

    def __init__(self, events: list[SdkEvent]) -> None:
        self._queued = list(events)
        self.written: list[bytes] = []
        self.finished = False
        self.closed = False
        self._finished_event = asyncio.Event()

    async def write(self, chunk: bytes) -> None:
        self.written.append(chunk)

    async def finish_input(self) -> None:
        self.finished = True
        self._finished_event.set()

    async def events(self) -> AsyncIterator[SdkEvent]:
        # The real recognizer never emits anything -- not even an interim
        # hypothesis -- before audio has actually reached it; gating on
        # finish_input keeps the fake honest about that ordering instead of
        # racing the concurrent sender task.
        await self._finished_event.wait()
        for event in self._queued:
            yield event
            if event.kind in {"session_stopped", "canceled"}:
                return

    async def close(self) -> None:
        self.closed = True


def make_factory(events: list[SdkEvent], captured: list[dict[str, Any]], sessions: list[FakeRecognizerSession]):
    """Build a fake ``_session_factory`` replacement recording its call kwargs."""

    async def factory(**kwargs: Any) -> FakeRecognizerSession:  # ruff: ignore[unused-async] -- must match the real Awaitable factory
        captured.append(kwargs)
        session = FakeRecognizerSession(events)
        sessions.append(session)
        return session

    return factory


class TestAzureSttProtocol:
    @pytest.fixture(autouse=True)
    def azure_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fake credentials for every test in this class.

        Class-scoped, not module-level: ``TestLiveSmoke`` below needs the
        *real* ``AZURE_SPEECH_KEY``/``AZURE_SPEECH_REGION`` from the
        environment and must never see this override.
        """
        monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "japaneast")

    async def test_recognized_and_speech_end_produce_final_and_eou(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            SdkEvent(kind="recognizing", text="hel"),
            SdkEvent(kind="recognized", text="hello world"),
            SdkEvent(kind="speech_end_detected"),
            SdkEvent(kind="session_stopped"),
        ]
        captured: list[dict[str, Any]] = []
        sessions: list[FakeRecognizerSession] = []
        monkeypatch.setattr(azure_stt, "_session_factory", make_factory(events, captured, sessions))

        adapter = stt.create("azure-speech-stt")
        result = await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)

        assert result.ok, result.error
        assert result.text == "hello world"
        assert result.raw["sdk_buffered"] is True
        assert result.raw["eou_source"] == "speech_end_detected"

        kinds = [p.kind for p in result.partials]
        assert EventKind.EOU in kinds
        assert any(p.is_final and p.text == "hello world" for p in result.partials)
        assert any(not p.is_final and p.text == "hel" for p in result.partials)

    async def test_audio_written_then_input_finished(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [SdkEvent(kind="session_stopped")]
        captured: list[dict[str, Any]] = []
        sessions: list[FakeRecognizerSession] = []
        monkeypatch.setattr(azure_stt, "_session_factory", make_factory(events, captured, sessions))

        adapter = stt.create("azure-speech-stt")
        await adapter.transcribe_stream(make_clip(seconds=0.1), chunk_ms=20, realtime=False)

        session = sessions[0]
        assert session.written, "expected at least one PCM chunk written"
        assert b"".join(session.written)  # non-empty payload
        assert session.finished
        assert session.closed

    async def test_language_defaults_to_clip_language_unmodified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [SdkEvent(kind="session_stopped")]
        captured: list[dict[str, Any]] = []
        sessions: list[FakeRecognizerSession] = []
        monkeypatch.setattr(azure_stt, "_session_factory", make_factory(events, captured, sessions))

        adapter = stt.create("azure-speech-stt")
        await adapter.transcribe_stream(make_clip(language="ja-JP"), chunk_ms=20, realtime=False)

        assert captured[0]["language"] == "ja-JP"
        assert captured[0]["subscription"] == "test-key"
        assert captured[0]["region"] == "japaneast"

    async def test_language_option_overrides_clip_language(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [SdkEvent(kind="session_stopped")]
        captured: list[dict[str, Any]] = []
        sessions: list[FakeRecognizerSession] = []
        monkeypatch.setattr(azure_stt, "_session_factory", make_factory(events, captured, sessions))

        adapter = stt.create("azure-speech-stt", {"language": "en-GB"})
        await adapter.transcribe_stream(make_clip(language="en-US"), chunk_ms=20, realtime=False)

        assert captured[0]["language"] == "en-GB"

    async def test_canceled_with_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [SdkEvent(kind="canceled", error="AuthenticationFailure: bad key")]
        captured: list[dict[str, Any]] = []
        sessions: list[FakeRecognizerSession] = []
        monkeypatch.setattr(azure_stt, "_session_factory", make_factory(events, captured, sessions))

        adapter = stt.create("azure-speech-stt")
        with pytest.raises(AzureSpeechError, match="AuthenticationFailure"):
            await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)

        assert sessions[0].closed, "session must still be closed after an error"

    async def test_clean_cancel_without_error_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [SdkEvent(kind="canceled", error=None)]
        captured: list[dict[str, Any]] = []
        sessions: list[FakeRecognizerSession] = []
        monkeypatch.setattr(azure_stt, "_session_factory", make_factory(events, captured, sessions))

        adapter = stt.create("azure-speech-stt")
        result = await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)

        assert result.ok, result.error


LIVE_FLAG = "AUDIO_HARNESS_TEST_AZURE_LIVE"


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run a couple of short live streams against Azure Speech (fractions of a cent total)",
)
class TestLiveSmoke:
    """A handful of short clips against the real Azure Speech endpoint.

    Minimal-API-testing policy: one English and one Japanese streaming clip
    only -- no bulk runs here. Requires ``AZURE_SPEECH_KEY`` and
    ``AZURE_SPEECH_REGION`` in the environment.
    """

    async def test_stream_en(self) -> None:
        adapter = stt.create("azure-speech-stt")
        clip = make_clip(seconds=1.5, clip_id="live-en", language="en-US")
        try:
            result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.raw["sdk_buffered"] is True

    async def test_stream_ja(self) -> None:
        adapter = stt.create("azure-speech-stt")
        clip = make_clip(seconds=1.5, clip_id="live-ja", language="ja-JP")
        try:
            result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.raw["sdk_buffered"] is True
