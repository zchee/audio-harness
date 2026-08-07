"""On-device macOS speech-to-text adapter using Apple's Speech framework."""

from __future__ import annotations

import asyncio
import tempfile
import time
from typing import Any

from audio_harness.audio import wrap_wav
from audio_harness.types import AudioClip, Mode, Partial, SttResult

from .base import StreamTimeline, SttProvider, register


Capture = tuple[float, str, bool]


@register
class AppleSpeechStt(SttProvider):
    """File-based, on-device recognition through macOS Speech.

    The OS receives the whole WAV file at once and reports evolving decode
    hypotheses. This lane therefore measures the local engine's partial-result
    timeline, not microphone capture or a chunked network transport.

    Options:
        locale: BCP-47 recognition locale; defaults to the clip's language.
        timeout_s: Maximum time to wait for a final callback; defaults to at
            least 30 seconds and otherwise twice the clip duration.
    """

    key = "apple-speech-stt"
    vendor = ""
    family = "apple"
    supports_stream = True

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Recognize a complete WAV file while retaining interim hypotheses.

        ``chunk_ms`` and ``realtime`` are accepted for the runner contract but
        intentionally ignored: ``SFSpeechURLRecognitionRequest`` consumes a
        whole file rather than client-paced chunks. The timeline marks audio
        complete immediately after the request is submitted because the whole
        file has reached the OS API at approximately t=0; ``finalize_s`` thus
        measures the engine's complete decode tail.

        Apple exposes decoding finality here, but no turn-taking decision, so
        this adapter never emits :class:`~audio_harness.types.EventKind.EOU`.
        Its endpointing data is descriptive-only.
        """
        del chunk_ms, realtime

        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        locale = str(self.options.get("locale") or clip.language)
        timeout_s = float(self.options.get("timeout_s", max(30.0, clip.duration_s * 2.0)))
        if timeout_s <= 0:
            raise ValueError("apple-speech-stt: timeout_s must be greater than zero")

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(wrap_wav(clip.pcm, clip.sample_rate))
            audio_file.flush()
            captures, started_at = await asyncio.to_thread(
                _recognize_file,
                audio_file.name,
                locale,
                timeout_s,
                timeline,
            )

        timeline.partials = _partials_from_captures(captures, started_at)
        result.text = timeline.last_final()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw.update(_local_compute_raw(result.total_s, clip.duration_s))
        return result


def _partials_from_captures(captures: list[Capture], started_at: float) -> list[Partial]:
    """Convert absolute callback captures into timeline-relative partials.

    Empty hypotheses are discarded to match :meth:`StreamTimeline.record`.
    Finality maps only to the normal interim/segment-final kinds supplied by
    :class:`Partial`; the file API provides no genuine EOU signal.
    """
    return [
        Partial(
            t_s=max(0.0, captured_at - started_at),
            text=text,
            is_final=is_final,
        )
        for captured_at, text, is_final in captures
        if text
    ]


def _local_compute_raw(total_s: float, duration_s: float) -> dict[str, object]:
    """Build metadata that distinguishes local decoding from hosted lanes."""
    return {
        "local_compute": True,
        "on_device": True,
        "rtf": total_s / duration_s if duration_s > 0 else None,
    }


def _recognize_file(
    path: str,
    locale: str,
    timeout_s: float,
    timeline: StreamTimeline,
) -> tuple[list[Capture], float]:
    """Run one callback-driven Speech request on a dedicated worker thread."""
    speech, foundation = _import_speech()
    native_locale = foundation.NSLocale.localeWithLocaleIdentifier_(locale)
    recognizer = speech.SFSpeechRecognizer.alloc().initWithLocale_(native_locale)
    if recognizer is None:
        raise RuntimeError(f"apple-speech-stt: could not create an SFSpeechRecognizer for locale {locale!r}")
    if not recognizer.isAvailable():
        raise RuntimeError(f"apple-speech-stt: SFSpeechRecognizer.isAvailable() is false for locale {locale!r}")
    if not recognizer.supportsOnDeviceRecognition():
        raise RuntimeError(
            f"apple-speech-stt: SFSpeechRecognizer.supportsOnDeviceRecognition() is false for locale {locale!r}"
        )

    # The recognizer delivers result-handler callbacks on its ``queue``
    # property, which defaults to the process main queue. This function runs
    # in an asyncio worker thread while the main thread awaits it, so nothing
    # drains the main queue and the handler would never fire (observed
    # 2026-08-08: main-thread recognition worked, worker-thread recognition
    # timed out with zero callbacks). A private operation queue makes
    # delivery independent of the main thread entirely.
    recognizer.setQueue_(foundation.NSOperationQueue.alloc().init())

    url = foundation.NSURL.fileURLWithPath_(path)
    request = speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
    request.setRequiresOnDeviceRecognition_(True)
    request.setShouldReportPartialResults_(True)

    captures: list[Capture] = []
    finished = False
    failure: str | None = None

    def receive(result: Any, error: Any) -> None:
        nonlocal failure, finished
        if result is not None:
            transcription = result.bestTranscription()
            text = str(transcription.formattedString()) if transcription is not None else ""
            is_final = bool(result.isFinal())
            captures.append((time.perf_counter(), text, is_final))
            if is_final:
                finished = True
        if error is not None:
            description = getattr(error, "localizedDescription", None)
            failure = str(description() if callable(description) else error)
            finished = True

    timeline.start()
    started_at = time.perf_counter()
    task = recognizer.recognitionTaskWithRequest_resultHandler_(request, receive)
    timeline.audio_complete()

    # Callbacks arrive on the private operation queue above, so this thread
    # only needs to sleep-poll the completion flag — no run loop involved.
    deadline = time.perf_counter() + timeout_s
    while not finished and time.perf_counter() < deadline:
        time.sleep(0.01)

    if not finished:
        task.cancel()
        raise RuntimeError(f"apple-speech-stt: recognition timed out after {timeout_s:.1f}s for locale {locale!r}")
    if failure is not None:
        raise RuntimeError(f"apple-speech-stt: recognition failed for locale {locale!r}: {failure}")
    return captures, started_at


def _import_speech() -> tuple[Any, Any]:
    """Import PyObjC lazily so the registry works without the optional extra."""
    try:
        import Foundation
        import Speech
    except ImportError as exc:
        raise RuntimeError(
            "apple-speech-stt: pyobjc-framework-Speech is not installed. Install "
            "the optional dependency group: uv sync --extra apple-speech"
        ) from exc
    return Speech, Foundation
