"""Azure AI Speech Neural TTS streaming text-to-speech adapter.

Like :mod:`audio_harness.stt.azure`, this goes through the vendor SDK
(``azure-cognitiveservices-speech``) rather than a raw WebSocket -- Azure's
realtime synthesis protocol is underdocumented and the SDK is the officially
supported client. The SDK owns its own buffering and delivery schedule on a
native worker thread, so every result carries ``raw["sdk_buffered"] = True``
-- read latency figures against the WebSocket-native adapters with that
caveat.

The SDK import is deferred to :func:`_import_speechsdk` so this module (and
the protocol tests below it) load without ``azure-cognitiveservices-speech``
installed; the ``azure`` optional dependency group pulls it in only for
adapters that actually run against the live service.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
import contextlib
from dataclasses import dataclass
from typing import Any, Protocol

from audio_harness.audio import decode_audio_duration
from audio_harness.config import require_env
from audio_harness.types import Mode, TtsPrompt, TtsResult

from .base import ChunkTimeline, TtsProvider, register, stamp_stream_timing


DEFAULT_VOICE_EN = "en-US-AvaMultilingualNeural"
"""Multilingual Neural voice, pinned so a model swap does not also swap the
speaker; also the voice used for every non-Japanese prompt."""

DEFAULT_VOICE_JA = "ja-JP-NanamiNeural"
"""Neural voice for Japanese prompts, verified available in the target
subscription/region (japaneast)."""

_PCM_SAMPLE_RATES = frozenset({8000, 16000, 24000, 48000})
"""Sample rates with a ``Raw<rate>Khz16BitMonoPcm`` SDK output format."""


class AzureSpeechError(RuntimeError):
    """Raised when the Azure Speech SDK cancels a synthesis on error."""


@dataclass(slots=True, frozen=True)
class SynthesisEvent:
    """One synthesizer event, with the native Azure SDK shape erased.

    Attributes:
        kind: ``"audio"``, ``"completed"`` or ``"canceled"``.
        audio: One PCM chunk; only set when ``kind`` is ``"audio"``.
        error: Cancellation detail, set only when ``kind`` is ``"canceled"``
            and the session ended on an error rather than a clean stop.
    """

    kind: str
    audio: bytes = b""
    error: str | None = None


class SynthesizerSession(Protocol):
    """Injectable seam over the Azure Speech SDK's speech synthesizer.

    Isolates every direct dependency on ``azure.cognitiveservices.speech``
    behind this shape, so protocol tests drive chunk timing and result
    assembly against a fake session -- the SDK need not be installed to run
    them.
    """

    async def speak(self, text: str) -> None:
        """Request synthesis of ``text``; resolves once the SDK call completes."""
        ...

    def events(self) -> AsyncIterator[SynthesisEvent]:
        """Yield synthesis events in arrival order, ending on a terminal one."""
        ...

    def close(self) -> None:
        """Release native resources."""
        ...


SessionFactory = Callable[..., SynthesizerSession]


def _import_speechsdk() -> Any:
    """Import the Azure Speech SDK lazily so the registry works without it.

    Every adapter must be importable at package-import time -- that is how
    the registry fills -- so the optional dependency is only touched when a
    prompt is actually synthesized.
    """
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:
        raise RuntimeError(
            "azure-neural-tts: azure-cognitiveservices-speech is not "
            "installed. Install the optional dependency group: "
            "uv sync --extra azure"
        ) from exc
    return speechsdk


def _output_format(speechsdk: Any, sample_rate: int) -> Any:
    """Map a PCM sample rate to the SDK's raw output format enum.

    Raises:
        ValueError: If ``sample_rate`` has no matching raw PCM format.
    """
    formats = {
        8000: speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm,
        16000: speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
        24000: speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
        48000: speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,
    }
    if sample_rate not in formats:
        raise ValueError(
            f"azure-neural-tts: unsupported sample rate {sample_rate}; expected one of {sorted(_PCM_SAMPLE_RATES)}"
        )
    return formats[sample_rate]


def _build_live_session(*, subscription: str, region: str, voice: str, sample_rate: int) -> SynthesizerSession:
    """Construct a real Azure SDK speech synthesizer session.

    ``audio_config=None`` drops output to the default speaker/file and routes
    every chunk through the ``synthesizing`` event instead, so this process
    -- not a native audio device -- receives the bytes.
    """
    speechsdk = _import_speechsdk()

    config = speechsdk.SpeechConfig(subscription=subscription, region=region)
    config.speech_synthesis_voice_name = voice
    config.set_speech_synthesis_output_format(_output_format(speechsdk, sample_rate))
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)

    return _LiveSynthesizerSession(synthesizer, loop=asyncio.get_running_loop())


_session_factory: SessionFactory = _build_live_session
"""Module-level seam: tests monkeypatch this to a fake factory."""


class _LiveSynthesizerSession:
    """Bridges Azure SDK callback events -- fired on a native thread -- to asyncio.

    Every event handler below runs on the SDK's own worker thread, never on
    the event loop thread, so each one only schedules a queue put via
    ``call_soon_threadsafe`` rather than touching asyncio state directly.
    """

    __slots__ = ("_events", "_loop", "_synthesizer")

    def __init__(self, synthesizer: Any, *, loop: asyncio.AbstractEventLoop) -> None:
        self._synthesizer = synthesizer
        self._loop = loop
        self._events: asyncio.Queue[SynthesisEvent] = asyncio.Queue()

        synthesizer.synthesizing.connect(self._on_audio)
        synthesizer.synthesis_completed.connect(self._on_completed)
        synthesizer.synthesis_canceled.connect(self._on_canceled)

    def _emit(self, event: SynthesisEvent) -> None:
        self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    def _on_audio(self, evt: Any) -> None:
        data = evt.result.audio_data
        if data:
            self._emit(SynthesisEvent(kind="audio", audio=data))

    def _on_completed(self, evt: Any) -> None:
        del evt
        self._emit(SynthesisEvent(kind="completed"))

    def _on_canceled(self, evt: Any) -> None:
        details = evt.result.cancellation_details
        error = None
        if details is not None:
            error = f"{details.error_code}: {details.error_details}"
        self._emit(SynthesisEvent(kind="canceled", error=error))

    async def speak(self, text: str) -> None:
        """Kick off synthesis on a worker thread; resolves once it finishes."""
        future = self._synthesizer.speak_text_async(text)
        await asyncio.to_thread(future.get)

    async def events(self) -> AsyncIterator[SynthesisEvent]:
        """Yield queued events until ``completed`` or ``canceled``."""
        while True:
            event = await self._events.get()
            yield event
            if event.kind in {"completed", "canceled"}:
                return

    def close(self) -> None:
        """Disconnect event handlers so no further events reach the queue.

        Mirrors the SDK's own ``__del__`` cleanup (see ``clean_signal`` on
        :class:`azure.cognitiveservices.speech.SpeechSynthesizer`); nothing
        here blocks, so this stays synchronous.
        """
        self._synthesizer.synthesizing.disconnect_all()
        self._synthesizer.synthesis_completed.disconnect_all()
        self._synthesizer.synthesis_canceled.disconnect_all()


@register
class AzureNeuralTts(TtsProvider):
    """Azure AI Speech Neural TTS over the vendor SDK's event-driven synthesis.

    Audio arrives incrementally through the ``synthesizing`` event while
    ``speak_text_async`` is still in flight, giving a real TTFB signal even
    though the SDK call itself only resolves once the whole utterance is
    done.

    Options:
        voice_en: Neural voice for non-Japanese prompts; defaults to
            :data:`DEFAULT_VOICE_EN` (multilingual).
        voice_ja: Neural voice for Japanese prompts; defaults to
            :data:`DEFAULT_VOICE_JA`.
        sample_rate: Output rate; must be one of 8000, 16000, 24000, 48000
            (the SDK's raw 16-bit mono PCM formats). Defaults to 24000.
    """

    key = "azure-neural-tts"
    vendor = "azure"
    family = "azure"
    supports_stream = True

    def _voice(self, language: str) -> str:
        if language.split("-")[0].lower() == "ja":
            return str(self.options.get("voice_ja", DEFAULT_VOICE_JA))
        return str(self.options.get("voice_en", DEFAULT_VOICE_EN))

    def _subscription(self) -> str:
        return require_env("AZURE_SPEECH_KEY", self.key)

    def _region(self) -> str:
        return require_env("AZURE_SPEECH_REGION", self.key)

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        """Synthesize through the SDK, collecting chunks as they arrive.

        Raises:
            ValueError: If the configured sample rate has no matching SDK
                raw PCM format.
        """
        if self.sample_rate not in _PCM_SAMPLE_RATES:
            raise ValueError(
                f"{self.key}: unsupported sample rate {self.sample_rate}; expected one of {sorted(_PCM_SAMPLE_RATES)}"
            )
        result = self._result(prompt, Mode.STREAM)
        voice = self._voice(prompt.language)
        timeline = ChunkTimeline()
        session = _session_factory(
            subscription=self._subscription(),
            region=self._region(),
            voice=voice,
            sample_rate=self.sample_rate,
        )

        try:
            speak_task = asyncio.create_task(session.speak(prompt.text))
            try:
                async for event in session.events():
                    if event.kind == "audio":
                        timeline.add(event.audio)
                    elif event.kind == "canceled" and event.error:
                        raise AzureSpeechError(f"{self.key}: {event.error}")
            finally:
                with contextlib.suppress(Exception):
                    await speak_task
        finally:
            session.close()

        result.raw["sdk_buffered"] = True
        result.raw["voice"] = voice
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
