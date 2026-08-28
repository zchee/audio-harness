"""Google Gemini 3.5 speech-to-text adapters.

The dedicated ``gemini-3.5-transcribe`` and ``gemini-3.5-transcribe-live``
models are public preview releases from August 2026. The preview aliases
``gemini-3.5-transcribe-preview`` and
``gemini-3.5-transcribe-live-preview`` are also accepted through the
``model`` option.

Both wire shapes were live-verified on 2026-08-28 against a spoken pipecat
clip ("How do I temper chocolate"); the opt-in ``TestLiveSmoke`` tests in
``tests/test_stt_gemini.py`` re-run that check. Where Google's announcement
docs and the live wire disagreed, the wire won — each class docstring lists
the specifics. The public REST discovery document
(``generativelanguage.googleapis.com/$discovery/rest?version=v1alpha``) was
the tie-breaker for field names.

Live interims restate the whole hypothesis for the current segment on every
frame (``"How"``, ``"How do I"``, ``"How do I temper"``, ...), so they are
the full-hypothesis partials the churn metric expects rather than append-only
deltas. Live sessions cap at ten minutes, which is irrelevant to this harness
because it opens one session per clip.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import time
from typing import Any

import numpy as np
import orjson
import soxr
from websockets.asyncio.client import ClientConnection

from audio_harness.audio import pcm16_to_float, wrap_wav
from audio_harness.config import require_env
from audio_harness.types import AudioClip, EventKind, Mode, SttResult

from .base import StreamTimeline, SttProvider, raise_for_status, register
from .ws import StreamProtocolError, run_stream


BATCH_URL = "https://generativelanguage.googleapis.com/v1alpha/models/{model}:generateContent"
"""Generate Content endpoint template for Gemini batch transcription.

v1alpha, not v1beta: only the v1alpha ``Part`` schema carries the
``audioTranscription`` field the transcribe model answers with.
"""

STREAM_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
)
"""BidiGenerateContent WebSocket endpoint for Gemini live transcription."""

GEMINI_SAMPLE_RATE = 16000
"""The Live API's required mono signed-16 PCM sample rate."""

EOU_SOURCE = "speechState:NON_SPEECH"
"""The Live API frame recorded as the end-of-utterance decision."""


def _bcp47(language: str) -> str:
    """Expand the benchmark's bare English/Japanese tags to regional tags."""
    return {"en": "en-US", "ja": "ja-JP"}.get(language, language)


def _language_codes(options: dict[str, Any], clip: AudioClip) -> list[str]:
    """Return configured language codes, or one derived from ``clip``."""
    configured = options.get("language_codes")
    if configured is None:
        return [_bcp47(clip.language)]
    if isinstance(configured, str):
        return [configured]
    return [str(code) for code in configured]


def _mode(options: dict[str, Any]) -> str:
    """Return the transcript formatting mode, preserving the vendor spelling."""
    return str(options.get("mode", "VERBATIM"))


def _model_path(model: str) -> str:
    """Add the Live API's required ``models/`` prefix exactly once."""
    return model if model.startswith("models/") else f"models/{model}"


@register
class GeminiTranscribe(SttProvider):
    """Batch Gemini 3.5 transcription through the Generate Content HTTP API.

    Live-verified 2026-08-28. Two things differ from the announcement docs:

    - The request field is ``generation_config.audio_transcription_config``
      (``AudioTranscriptionConfig`` in the discovery document), not
      ``transcription_config``: the server rejects the latter with
      ``Unknown name "transcription_config" at 'generation_config'``.
    - The transcript comes back as
      ``candidates[0].content.parts[].audioTranscription.text`` — a field
      only the v1alpha ``Part`` schema defines — not as a plain ``text`` part.
      The identical request against v1beta returns HTTP 200 with an empty part
      (``"parts": [{}]``) because that surface's serializer drops the field,
      so this adapter posts to v1alpha.

    ``VERBATIM`` is the deliberate default: ``SMART`` reformats numbers and
    punctuation, which would make word-error-rate comparisons inconsistent
    with the other lanes. The model is a public preview; use the ``model``
    option to select ``gemini-3.5-transcribe-preview`` when desired.

    Options:
        model: Model identifier; defaults to ``gemini-3.5-transcribe``.
        mode: Transcription formatting mode; defaults to ``VERBATIM`` rather
            than ``SMART`` for comparable WER.
        language_codes: List of BCP-47 language codes; defaults to the clip's
            language after bare-tag expansion.
        custom_vocabulary: Optional list passed through to the API.
    """

    key = "gemini-35-transcribe"
    vendor = "gemini"
    family = "gemini"
    supports_batch = True
    supports_stream = False

    def _model(self) -> str:
        return str(self.options.get("model", "gemini-3.5-transcribe"))

    def _body(self, clip: AudioClip) -> dict[str, Any]:
        """Build the Generate Content transcription request body."""
        transcription_config: dict[str, Any] = {
            "mode": _mode(self.options),
            "language_codes": _language_codes(self.options, clip),
        }
        if "custom_vocabulary" in self.options and self.options["custom_vocabulary"] is not None:
            transcription_config["custom_vocabulary"] = self.options["custom_vocabulary"]
        return {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(wrap_wav(clip.pcm, clip.sample_rate)).decode("ascii"),
                            }
                        }
                    ]
                }
            ],
            "generation_config": {"audio_transcription_config": transcription_config},
        }

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Send a WAV-wrapped clip to Gemini's Generate Content endpoint."""
        result = self._result(clip, Mode.BATCH)
        started = time.perf_counter()
        api_key = require_env("GEMINI_API_KEY", self.key)
        response = await self.http.post(
            BATCH_URL.format(model=self._model()),
            headers={"x-goog-api-key": api_key},
            json=self._body(clip),
        )
        raise_for_status(response, self.key)
        payload = response.json()
        result.total_s = time.perf_counter() - started
        result.text = _batch_text(payload)
        result.raw["response"] = payload
        return result


@register
class GeminiTranscribeLive(SttProvider):
    """Streaming Gemini 3.5 transcription through BidiGenerateContent.

    Live-verified 2026-08-28 on the v1alpha endpoint. The handshake accepts
    the API key as an ``x-goog-api-key`` header, so the secret never appears
    in a URL and the documented ``?key=`` query form is not used. The
    ``setup.input_audio_transcription`` object (``mode``, ``language_codes``)
    and ``audio/pcm;rate=16000`` media chunks are accepted as documented.

    The server frames observed for one spoken sentence, in order:

    1. ``setupComplete``
    2. ``serverContent.speechState: "SPEECH"`` — server VAD speech start
    3. ``serverContent.interimInputTranscription.text`` — the cumulative
       hypothesis for the current segment, restated on every frame
    4. ``serverContent.inputTranscription.text`` — the final text of one
       segment
    5. ``serverContent.generationComplete: true``
    6. ``serverContent.speechState: "NON_SPEECH"`` — server VAD speech end

    Neither ``modelTurn`` nor ``turnComplete`` — the shapes the general Live
    API docs describe — appeared at any point, so the handler does not read
    them. ``speechState: "NON_SPEECH"`` is the server's own end-of-speech
    decision and is recorded as the bare ``EventKind.EOU`` marker (the same
    treatment as Google Cloud's ``SPEECH_ACTIVITY_END``); ``inputTranscription``
    finals are decoding boundaries and stay plain segment finals. A
    ``NON_SPEECH`` received before this adapter sends its ``audio_stream_end``
    terminator is a genuine voice-activity decision; one received after it is
    the server's answer to our own forced end and is not recorded.
    ``eou_grace_ms`` defaults to 1000 ms (0 disables the hold) specifically
    to give server VAD a fair window to fire before the terminator forces the
    issue. On the 2.9 s verification clip, which ends with 0.7 s of silence,
    VAD did *not* fire inside that window: the final, ``generationComplete``
    and ``NON_SPEECH`` all arrived about 0.2 s after the terminator, so
    ``finalize_s`` carried the full grace delay. A timestamp gate against
    ``audio_end_s`` is biased because clips carry trailing silence that can
    outlast or fall short of the server's VAD tail, so genuine VAD decisions
    can land on either side of that timestamp.

    The endpoint answers ``audio_stream_end`` only with the flush of a segment
    it still holds open (final, then ``generationComplete``); a silent clip,
    or one whose last segment server VAD already flushed, gets no frame at
    all — live-verified with a 0.8 s silent clip, which otherwise sat out the
    driver's full finalize timeout. So the session ends on the first
    ``generationComplete`` after the terminator when a segment is open, and
    by closing the socket right after the terminator when none is; the
    driver's finalize timeout remains the backstop.

    Options:
        model: Model identifier; defaults to ``gemini-3.5-transcribe-live``.
        mode: Transcription formatting mode; defaults to ``VERBATIM``.
        language_codes: List of BCP-47 language codes; defaults to the clip's
            language after bare-tag expansion.
        eou_grace_ms: Milliseconds to wait for server VAD before sending the
            terminator; defaults to 1000. Set to 0 to disable the hold.
    """

    key = "gemini-35-transcribe-live"
    vendor = "gemini"
    family = "gemini"
    supports_batch = False
    supports_stream = True

    def _model(self) -> str:
        return str(self.options.get("model", "gemini-3.5-transcribe-live"))

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Stream 16 kHz PCM to Gemini's Live API and assemble its transcript.

        When VAD does not fire during ``eou_grace_ms``, the forced final's
        ``finalize_s`` includes the full grace delay. This is a conservative
        penalty against the lane, and the effective delay is published through
        ``endpoint_config`` so rankings remain interpretable instead of being
        silently distorted.
        """
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        wire_clip = clip if clip.sample_rate == GEMINI_SAMPLE_RATE else _resample_clip(clip, GEMINI_SAMPLE_RATE)
        handler = _TranscriptionHandler()
        eou_grace_ms = int(self.options.get("eou_grace_ms", 1000))

        async def configure(socket: ClientConnection) -> None:
            setup = {
                "setup": {
                    "model": _model_path(self._model()),
                    "generation_config": {"response_modalities": ["TEXT"]},
                    "input_audio_transcription": {
                        "mode": _mode(self.options),
                        "language_codes": _language_codes(self.options, clip),
                    },
                }
            }
            await socket.send(orjson.dumps(setup).decode())

        async def input_done(socket: ClientConnection) -> None:
            if eou_grace_ms > 0:
                await asyncio.sleep(eou_grace_ms / 1000)
            await socket.send(orjson.dumps({"realtime_input": {"audio_stream_end": True}}).decode())
            handler.terminator_sent = True
            if not handler.segment_open:
                # The endpoint answers audio_stream_end with nothing at all
                # once its last segment is already flushed, or when it heard
                # no speech; closing here ends the session instead of sitting
                # out the driver's finalize timeout.
                await socket.close()

        await run_stream(
            url=STREAM_URL,
            headers={"x-goog-api-key": require_env("GEMINI_API_KEY", self.key)},
            clip=wire_clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=handler,
            on_open=configure,
            encode_chunk=_encode_chunk,
            on_input_done=input_done,
        )

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        result.raw["endpoint_config"] = {"eou_grace_ms": eou_grace_ms}
        if any(partial.kind == EventKind.EOU for partial in timeline.partials):
            result.raw["eou_source"] = EOU_SOURCE
        return result


def _transcription_text(transcription: Any) -> str | None:
    """Return the ``text`` of an ``AudioTranscription`` object, or ``None``."""
    if isinstance(transcription, dict) and transcription.get("text") is not None:
        return str(transcription["text"])
    return None


def _batch_text(payload: Any) -> str:
    """Join ``audioTranscription.text`` across the first candidate's parts."""
    if not isinstance(payload, dict):
        return ""
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return ""
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    texts = (_transcription_text(part.get("audioTranscription")) for part in parts if isinstance(part, dict))
    return " ".join(text.strip() for text in texts if text is not None and text.strip())


def _encode_chunk(chunk: bytes) -> str:
    """Frame one PCM chunk as a Gemini ``realtime_input`` JSON message."""
    return orjson.dumps({
        "realtime_input": {
            "media_chunks": [
                {
                    "mime_type": "audio/pcm;rate=16000",
                    "data": base64.b64encode(chunk).decode("ascii"),
                }
            ]
        }
    }).decode()


def _resample_clip(clip: AudioClip, target_rate: int) -> AudioClip:
    """Return a copy of ``clip`` resampled to ``target_rate`` for the wire.

    Only the transmitted bytes change; duration, reference and every other
    field the result carries stay keyed to the original clip — the same
    approach ``whisper_local.py`` uses for its own fixed input rate.
    """
    mono = soxr.resample(pcm16_to_float(clip.pcm), clip.sample_rate, target_rate, quality="HQ")
    pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return replace(clip, pcm=pcm, sample_rate=target_rate)


class _TranscriptionHandler:
    """Record Live API transcription frames and track what the server still owes.

    Attributes:
        segment_open: Whether the server has started a segment (``speechState:
            SPEECH`` or any transcription frame) that no ``generationComplete``
            has closed yet — the only case in which ``audio_stream_end`` is
            answered with more frames.
        terminator_sent: Whether the adapter has sent ``audio_stream_end``.
    """

    __slots__ = ("segment_open", "terminator_sent")

    def __init__(self) -> None:
        self.segment_open = False
        self.terminator_sent = False

    def __call__(self, payload: Any, timeline: StreamTimeline) -> bool:
        """Record one decoded Gemini frame and report whether the session ends."""
        if not isinstance(payload, dict):
            return False
        if "error" in payload:
            raise StreamProtocolError(f"gemini: {_frame_message(payload, 'error')}")
        if "goAway" in payload:
            raise StreamProtocolError(f"gemini: {_frame_message(payload, 'goAway')}")

        server_content = payload.get("serverContent")
        if not isinstance(server_content, dict):
            return False

        interim = _transcription_text(server_content.get("interimInputTranscription"))
        if interim is not None:
            timeline.record(interim, is_final=False)
        final = _transcription_text(server_content.get("inputTranscription"))
        if final is not None:
            timeline.record(final, is_final=True)
        if server_content.get("speechState") == "SPEECH" or interim is not None or final is not None:
            self.segment_open = True
        if server_content.get("speechState") == "NON_SPEECH" and not self.terminator_sent:
            timeline.record("", is_final=False, kind=EventKind.EOU)
        if server_content.get("generationComplete") is not True:
            return False
        self.segment_open = False
        return self.terminator_sent


def _frame_message(payload: dict[str, Any], field: str) -> str:
    """Extract a useful detail from an error-like Gemini frame."""
    detail = payload.get(field)
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("status") or detail)
    return str(detail or payload)
