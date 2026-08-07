"""OpenAI gpt-4o-transcribe speech-to-text adapter."""

from __future__ import annotations

import base64
from dataclasses import replace
from typing import Any

import numpy as np
import orjson
import soxr
from websockets.asyncio.client import ClientConnection

from audio_harness.audio import pcm16_to_float
from audio_harness.config import require_env
from audio_harness.types import AudioClip, EventKind, Mode, SttResult

from .base import StreamTimeline, SttProvider, register
from .ws import OnInputDone, StreamProtocolError, run_stream


STREAM_URL = "wss://api.openai.com/v1/realtime?intent=transcription"

REALTIME_SAMPLE_RATE = 24000
"""Fixed rate the Realtime API's ``audio/pcm`` input format requires."""


@register
class OpenAiGpt4oTranscribe(SttProvider):
    """gpt-4o-transcribe over the Realtime WebSocket transcription session.

    OpenAI's docs are mid-transition between two shapes: a legacy
    ``input_audio_transcription``/``turn_detection`` top-level session and the
    current one used here — ``session.type: "transcription"`` with a nested
    ``audio.input.{format, transcription, turn_detection}`` object. The
    ``?intent=transcription`` query parameter (rather than a conversational
    ``model=``) selects a transcription-only session on the GA endpoint;
    published examples pairing it with an ``OpenAI-Beta: realtime=v1``
    handshake header are for the deprecated Beta API and must NOT be sent —
    live-verified 2026-08-07, the server rejects it outright with "The
    Realtime Beta API is no longer supported. Please use /v1/realtime for the
    GA API." Unlike Inworld's adapter there was no open-source reference
    client to cross-check this against, so — per the xAI adapter's precedent —
    verify streaming results against a known transcript before trusting the
    latency figures.

    Server VAD (``turn_detection.type: "server_vad"``) is always enabled, so
    ``input_audio_buffer.speech_stopped`` — a bare turn-taking decision with
    no transcript, the same shape as Deepgram's ``UtteranceEnd`` — is always
    available as the EOU signal.

    Options:
        model: Transcription model; defaults to ``gpt-4o-transcribe``.
        language: ISO-639-1 language hint; defaults to the clip's language.
        threshold: Server VAD activation threshold, 0.0-1.0 (vendor default
            0.5).
        prefix_padding_ms: Audio included before VAD speech detection
            (vendor default 300).
        silence_duration_ms: Silence duration that ends a turn (vendor
            default 500).
    """

    key = "openai-gpt4o-transcribe"
    vendor = "openai"
    family = "openai"
    supports_stream = True

    def _model(self) -> str:
        return str(self.options.get("model", "gpt-4o-transcribe"))

    def _language(self, clip: AudioClip) -> str:
        return str(self.options.get("language", clip.language.split("-")[0]))

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {require_env('OPENAI_API_KEY', self.key)}"}

    def _turn_detection(self) -> dict[str, Any] | None:
        turn_detection: dict[str, Any] = {
            "type": "server_vad",
            "create_response": False,
            "interrupt_response": False,
        }
        for name in ("threshold", "prefix_padding_ms", "silence_duration_ms"):
            if name in self.options:
                turn_detection[name] = self.options[name]
        return turn_detection

    def _on_input_done(self) -> OnInputDone | None:
        """Signal end of input, if the model needs an explicit commit.

        Server VAD auto-finalizes each turn once it detects silence, so the
        base model needs no explicit commit; :class:`OpenAiGptLiveTranscribe`
        overrides this because it disables VAD entirely.
        """
        return None

    def _transcription_config(self, clip: AudioClip) -> dict[str, Any]:
        """Build the ``audio.input.transcription`` object for the session update.

        Split out so a model variant with a different set of session knobs
        (see :class:`OpenAiGptLiveTranscribe`) can override just this piece
        instead of duplicating the whole streaming method.
        """
        return {"model": self._model(), "language": self._language(clip)}

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Stream PCM over the Realtime transcription session with server VAD."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        wire_clip = clip if clip.sample_rate == REALTIME_SAMPLE_RATE else _resample_clip(clip, REALTIME_SAMPLE_RATE)
        turn_detection = self._turn_detection()

        async def configure(socket: ClientConnection) -> None:
            session: dict[str, Any] = {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE},
                        "transcription": self._transcription_config(clip),
                        "turn_detection": turn_detection,
                    }
                },
            }
            await socket.send(orjson.dumps({"type": "session.update", "session": session}).decode())

        await run_stream(
            url=STREAM_URL,
            headers=self._auth(),
            clip=wire_clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=_TranscriptionHandler(),
            on_open=configure,
            encode_chunk=_encode_chunk,
            on_input_done=self._on_input_done(),
        )

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        if turn_detection is not None:
            # Server VAD runs on this session, so the lane is EOU-capable;
            # the effective knob values ride along so rankings stay
            # interpretable (see stt/deepgram.py's endpointing comment).
            # Left unset when VAD is disabled (see
            # OpenAiGptLiveTranscribe._turn_detection): no vendor EOU
            # decision exists to report.
            result.raw["eou_source"] = "speech_stopped"
            result.raw["endpoint_config"] = turn_detection
        return result


@register
class OpenAiGptLiveTranscribe(OpenAiGpt4oTranscribe):
    """gpt-live-transcribe: OpenAI's purpose-built low-latency live-streaming model.

    Same Realtime transcription-session wire protocol as
    :class:`OpenAiGpt4oTranscribe`, with two live-verified differences
    (2026-08-07):

    Server VAD is rejected outright -- ``session.update`` errors with "Turn
    detection is not supported for this transcription model." So this model
    always sends ``turn_detection: null`` and, per the GA docs' "continuous
    audio processing and explicit turn commits" mode, an explicit
    ``input_audio_buffer.commit`` once all audio has been written (the
    inherited VAD knobs -- ``threshold``, ``prefix_padding_ms``,
    ``silence_duration_ms`` -- do not apply and are ignored). Losing VAD also
    loses the only EOU signal this wire protocol has: unlike
    :class:`OpenAiGpt4oTranscribe`, this lane never sets
    ``raw["eou_source"]``, and a manual commit is not a vendor turn-taking
    decision, so its final stays a plain segment final rather than
    :class:`~audio_harness.types.EventKind.EOU` (same reasoning as
    :mod:`audio_harness.stt.elevenlabs`'s manual-commit mode).

    The session's ``audio.input.transcription`` object also carries this
    model's own set of fields (GA docs,
    developers.openai.com/docs/guides/realtime-transcription; these are GA
    field names, not the Beta-era ones):

    - ``languages`` (a plural array) replaces the singular ``language``
      field for this model -- sending both is a validation error, so the
      parent's ``language`` is never sent here.
    - ``keywords``: hints for acronyms, proper nouns, jargon or product
      names; a list of strings.
    - ``delay``: tunable latency, one of ``minimal``, ``low``, ``medium``,
      ``high`` or ``xhigh`` -- higher values trade latency for more audio
      context and a lower error rate.
    - ``prompt``: free-form context string describing the recording.

    Options:
        model: defaults to ``gpt-live-transcribe``.
        languages: list of language hints (ISO 639-1, ISO 639-3, or a
            regional locale like ``zh-cn``); defaults to a single-element
            list built from the clip's own language.
        keywords: list of keyword/acronym/proper-noun hints.
        delay: tunable latency knob; left unset to use the vendor default.
        prompt: free-form context string.
    """

    key = "openai-live-transcribe"

    def _model(self) -> str:
        return str(self.options.get("model", "gpt-live-transcribe"))

    def _turn_detection(self) -> dict[str, Any] | None:
        return None

    async def _commit(self, socket: ClientConnection) -> None:
        await socket.send(orjson.dumps({"type": "input_audio_buffer.commit"}).decode())

    def _on_input_done(self) -> OnInputDone | None:
        return self._commit

    def _transcription_config(self, clip: AudioClip) -> dict[str, Any]:
        config: dict[str, Any] = {
            "model": self._model(),
            "languages": list(self.options.get("languages") or [self._language(clip)]),
        }
        for name in ("keywords", "delay", "prompt"):
            if name in self.options:
                config[name] = self.options[name]
        return config


def _encode_chunk(chunk: bytes) -> str:
    """Frame one PCM chunk as a base64 ``input_audio_buffer.append`` message."""
    return orjson.dumps({
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(chunk).decode("ascii"),
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
    """Records streaming transcription events, accumulating deltas per item.

    ``delta`` events carry only the newly transcribed fragment, so this
    concatenates them per ``item_id`` before recording — :class:`Partial.text
    <audio_harness.types.Partial>` is documented as the full hypothesis at
    that instant, which is what downstream churn analysis (a growing-prefix
    check) requires. ``speech_stopped`` is server VAD's turn-taking decision
    and carries no transcript; ``completed`` is a transcription decision, not
    a turn-taking one, so it stays a plain final per the endpointing design's
    rule that only a genuine EOU signal earns ``kind=eou``.
    """

    __slots__ = ("_text",)

    def __init__(self) -> None:
        self._text: dict[str, str] = {}

    def __call__(self, payload: Any, timeline: StreamTimeline) -> bool:
        if not isinstance(payload, dict):
            return False
        kind = payload.get("type")

        if kind == "error":
            raise StreamProtocolError(f"openai: {_error_message(payload)}")
        if kind == "conversation.item.input_audio_transcription.failed":
            raise StreamProtocolError(f"openai: {_error_message(payload)}")

        if kind == "input_audio_buffer.speech_stopped":
            timeline.record("", is_final=False, kind=EventKind.EOU)
            return False

        if kind == "conversation.item.input_audio_transcription.delta":
            delta = str(payload.get("delta", ""))
            if not delta:
                return False
            item_id = str(payload.get("item_id", ""))
            self._text[item_id] = self._text.get(item_id, "") + delta
            timeline.record(self._text[item_id], is_final=False)
            return False

        if kind == "conversation.item.input_audio_transcription.completed":
            item_id = str(payload.get("item_id", ""))
            transcript = str(payload.get("transcript", "")) or self._text.get(item_id, "")
            self._text.pop(item_id, None)
            timeline.record(transcript, is_final=True)
            # No explicit end-of-stream frame exists in this protocol; the
            # first completed transcript once all audio has been written is
            # this clip's result, so the session ends here rather than
            # idling out the full finalize grace period.
            return timeline.audio_end_s is not None

        return False


def _error_message(payload: dict[str, Any]) -> str:
    """Extract the human-readable detail from an OpenAI Realtime error frame."""
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(error or payload)
