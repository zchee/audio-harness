"""Azure AI Speech streaming speech-to-text adapter.

Like :mod:`audio_harness.stt.google`, this adapter goes through the vendor
SDK (``azure-cognitiveservices-speech``) rather than a raw WebSocket: Azure's
realtime protocol is underdocumented and the SDK is the officially supported
client. The SDK owns its own buffering, batching and retry schedule on a
native worker thread, so every result carries ``raw["sdk_buffered"] = True``
-- read latency figures against the WebSocket-native adapters with that
caveat.

The SDK import is deferred to :func:`_build_live_session` so this module (and
the protocol tests below it) load without ``azure-cognitiveservices-speech``
installed; the ``azure`` optional dependency group pulls it in only for
adapters that actually run against the live service.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
import contextlib
from dataclasses import dataclass
from typing import Any, Protocol

from audio_harness.audio import pace_chunks
from audio_harness.config import require_env
from audio_harness.types import AudioClip, EventKind, Mode, SttResult

from .base import StreamTimeline, SttProvider, register


class AzureSpeechError(RuntimeError):
    """Raised when the Azure Speech SDK cancels a recognition session on error."""


@dataclass(slots=True, frozen=True)
class SdkEvent:
    """One recognizer event, with the native Azure SDK shape erased.

    Attributes:
        kind: ``"recognizing"``, ``"recognized"``, ``"speech_end_detected"``,
            ``"session_stopped"`` or ``"canceled"``.
        text: Hypothesis text; empty for the bare marker kinds.
        error: Cancellation detail, set only when ``kind`` is ``"canceled"``
            and the session ended on an error rather than a clean stop.
    """

    kind: str
    text: str = ""
    error: str | None = None


class RecognizerSession(Protocol):
    """Injectable seam over the Azure Speech SDK's push-stream recognizer.

    Isolates every direct dependency on ``azure.cognitiveservices.speech``
    behind this shape, so protocol tests drive event routing, EOU capture and
    result assembly against a fake session -- the SDK need not be installed
    to run them.
    """

    async def write(self, chunk: bytes) -> None:
        """Push one PCM chunk into the recognizer's input stream."""
        ...

    async def finish_input(self) -> None:
        """Signal that no more audio will be written."""
        ...

    def events(self) -> AsyncIterator[SdkEvent]:
        """Yield recognizer events in arrival order, ending on a terminal one."""
        ...

    async def close(self) -> None:
        """Stop recognition and release native resources."""
        ...


SessionFactory = Callable[..., Awaitable[RecognizerSession]]


def _import_speechsdk() -> Any:
    """Import the Azure Speech SDK lazily so the registry works without it.

    Every adapter must be importable at package-import time -- that is how
    the registry fills -- so the optional dependency is only touched when a
    clip is actually transcribed.
    """
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:
        raise RuntimeError(
            "azure-speech-stt: azure-cognitiveservices-speech is not "
            "installed. Install the optional dependency group: "
            "uv sync --extra azure"
        ) from exc
    return speechsdk


async def _build_live_session(*, clip: AudioClip, subscription: str, region: str, language: str) -> RecognizerSession:
    """Construct and start a real Azure SDK push-stream recognition session."""
    speechsdk = _import_speechsdk()

    config = speechsdk.SpeechConfig(subscription=subscription, region=region)
    config.speech_recognition_language = language
    stream_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=clip.sample_rate, bits_per_sample=16, channels=1
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
    recognizer = speechsdk.SpeechRecognizer(speech_config=config, audio_config=audio_config)

    session = _LiveRecognizerSession(
        speechsdk, recognizer=recognizer, push_stream=push_stream, loop=asyncio.get_running_loop()
    )
    await session.start()
    return session


_session_factory: SessionFactory = _build_live_session
"""Module-level seam: tests monkeypatch this to a fake factory."""


class _LiveRecognizerSession:
    """Bridges Azure SDK callback events -- fired on a native thread -- to asyncio.

    Every event handler below runs on the SDK's own worker thread, never on
    the event loop thread, so each one only schedules a queue put via
    ``call_soon_threadsafe`` rather than touching asyncio state directly. The
    two blocking SDK calls (``start``/``stop`` continuous recognition, push
    stream I/O) are pushed onto a thread pool with :func:`asyncio.to_thread`
    instead of being awaited inline, so nothing here blocks the event loop.
    """

    __slots__ = ("_events", "_loop", "_push_stream", "_recognizer", "_speechsdk")

    def __init__(
        self,
        speechsdk: Any,
        *,
        recognizer: Any,
        push_stream: Any,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._speechsdk = speechsdk
        self._recognizer = recognizer
        self._push_stream = push_stream
        self._loop = loop
        self._events: asyncio.Queue[SdkEvent] = asyncio.Queue()

        recognizer.recognizing.connect(self._on_recognizing)
        recognizer.recognized.connect(self._on_recognized)
        recognizer.speech_end_detected.connect(self._on_speech_end)
        recognizer.session_stopped.connect(self._on_session_stopped)
        recognizer.canceled.connect(self._on_canceled)

    def _emit(self, event: SdkEvent) -> None:
        self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    def _on_recognizing(self, evt: Any) -> None:
        self._emit(SdkEvent(kind="recognizing", text=evt.result.text))

    def _on_recognized(self, evt: Any) -> None:
        if evt.result.reason == self._speechsdk.ResultReason.RecognizedSpeech:
            self._emit(SdkEvent(kind="recognized", text=evt.result.text))

    def _on_speech_end(self, evt: Any) -> None:
        del evt
        self._emit(SdkEvent(kind="speech_end_detected"))

    def _on_session_stopped(self, evt: Any) -> None:
        del evt
        self._emit(SdkEvent(kind="session_stopped"))

    def _on_canceled(self, evt: Any) -> None:
        details = evt.cancellation_details
        error = None
        if details is not None and details.reason == self._speechsdk.CancellationReason.Error:
            error = f"{details.code}: {details.error_details}"
        self._emit(SdkEvent(kind="canceled", error=error))

    async def start(self) -> None:
        """Begin continuous recognition; resolves once the SDK confirms it started."""
        future = self._recognizer.start_continuous_recognition_async()
        await asyncio.to_thread(future.get)

    async def write(self, chunk: bytes) -> None:
        """Write one PCM chunk to the push stream on a worker thread."""
        await asyncio.to_thread(self._push_stream.write, chunk)

    async def finish_input(self) -> None:
        """Close the push stream so the SDK finalizes the last segment."""
        await asyncio.to_thread(self._push_stream.close)

    async def events(self) -> AsyncIterator[SdkEvent]:
        """Yield queued events until ``session_stopped`` or ``canceled``."""
        while True:
            event = await self._events.get()
            yield event
            if event.kind in {"session_stopped", "canceled"}:
                return

    async def close(self) -> None:
        """Stop continuous recognition and release the native recognizer."""
        future = self._recognizer.stop_continuous_recognition_async()
        await asyncio.to_thread(future.get)


@register
class AzureSpeechStt(SttProvider):
    """Azure AI Speech continuous recognition over the vendor SDK.

    The vendor's own end-of-utterance signal (``speech_end_detected``, a
    VAD-based decision) is recorded as a bare
    :attr:`~audio_harness.types.EventKind.EOU` marker, matching how
    :mod:`audio_harness.stt.google` treats ``SPEECH_ACTIVITY_END``.
    ``recognized`` events (a finished phrase) are segment finals;
    ``recognizing`` events are interim hypotheses.

    Only the streaming transport is implemented -- Azure's pre-recorded
    transcription is a separate, job-based Batch Transcription REST API with
    its own submit/poll/fetch lifecycle, out of scope here.

    Options:
        language: BCP-47 locale passed to the SDK, e.g. ``en-US``. Azure
            requires the full locale -- unlike most vendors here it rejects a
            bare language code -- so this defaults to the clip's own
            ``language`` tag unmodified.
    """

    key = "azure-speech-stt"
    vendor = "azure"
    family = "azure"
    supports_stream = True

    def _language(self, clip: AudioClip) -> str:
        return str(self.options.get("language", clip.language))

    def _subscription(self) -> str:
        return require_env("AZURE_SPEECH_KEY", self.key)

    def _region(self) -> str:
        return require_env("AZURE_SPEECH_REGION", self.key)

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Stream PCM through the SDK's push-stream continuous recognizer."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        session = await _session_factory(
            clip=clip,
            subscription=self._subscription(),
            region=self._region(),
            language=self._language(clip),
        )

        async def send_audio() -> None:
            async for chunk in pace_chunks(clip, chunk_ms, realtime=realtime):
                await session.write(chunk)
            timeline.audio_complete()
            await session.finish_input()

        try:
            timeline.start()
            sender = asyncio.create_task(send_audio())
            try:
                async for event in session.events():
                    if event.kind == "canceled" and event.error:
                        raise AzureSpeechError(f"{self.key}: {event.error}")
                    _record_event(event, timeline)
            finally:
                if not sender.done():
                    sender.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await sender
        finally:
            await session.close()

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["sdk_buffered"] = True
        result.raw["eou_source"] = "speech_end_detected"
        return result


def _record_event(event: SdkEvent, timeline: StreamTimeline) -> None:
    """Translate one SDK event into a timeline entry.

    ``speech_end_detected`` carries no text -- the SDK's VAD end-of-utterance
    decision -- and is recorded as a bare EOU marker exactly like Deepgram's
    ``UtteranceEnd`` and Google's ``SPEECH_ACTIVITY_END``.
    """
    if event.kind == "recognizing":
        timeline.record(event.text, is_final=False)
    elif event.kind == "recognized":
        timeline.record(event.text, is_final=True)
    elif event.kind == "speech_end_detected":
        timeline.record("", is_final=False, kind=EventKind.EOU)
