"""Gladia Solaria realtime speech-to-text adapter."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import orjson
from websockets.asyncio.client import ClientConnection

from audio_harness.config import require_env
from audio_harness.types import AudioClip, EventKind, Mode, SttResult

from .base import StreamTimeline, SttProvider, raise_for_status, register
from .ws import run_stream


INIT_URL = "https://api.gladia.io/v2/live"


@register
class GladiaSolaria1(SttProvider):
    """Gladia Solaria over the v2 live session-init-then-WebSocket pipeline.

    Unlike vendors whose streaming endpoint is a static URL, Gladia's
    protocol is two steps: a ``POST /v2/live`` call negotiates the session
    and returns a one-time WebSocket URL with the auth token embedded in it,
    which is then the only thing used to open the socket. That init round
    trip happens before :class:`StreamTimeline` starts, so it is not
    measured — the same treatment every other adapter's own connection setup
    gets.

    Options:
        model: Model identifier; ``solaria-1`` is currently the only value
            the API accepts.
        language: Pinned ISO 639-1 code; defaults to the clip's language.
            Set to an empty string to let Gladia auto-detect instead.
        code_switching: Whether the detected language is re-evaluated per
            utterance rather than pinned after the first one. Ignored when
            ``language`` pins a single code.
        audio_enhancer: Whether Gladia pre-processes the audio to improve
            quality before recognition.
        speech_threshold: Voice-activity sensitivity (0-1, server default
            0.6); values closer to 1 reject background sound more strictly.
        endpointing: Silence duration in seconds (0.01-10, server default
            0.05) that closes the current utterance.
        maximum_duration_without_endpointing: Safety ceiling in seconds
            (5-60, server default 5) that forces an utterance closed even
            without qualifying silence.
        receive_partial_transcripts: Whether interim hypotheses are sent.
            Gladia defaults this to ``false``; this adapter defaults it to
            ``true`` so TTFT is measurable, matching every other streaming
            lane in this harness.
        receive_speech_events: Whether ``speech_start``/``speech_end`` are
            sent. Gladia documents these explicitly as the signal to drive
            "agent turn-taking", so ``speech_end`` is this lane's
            end-of-utterance event; a transcript's own ``is_final`` marks
            only the endpointing-driven close of that utterance's decoding,
            the same segment-boundary role Google's per-result ``is_final``
            plays.
        region: ``us-west`` or ``eu-west`` to pin the processing region.
    """

    key = "gladia-solaria1"
    vendor = "gladia"
    supports_stream = True

    def _auth(self) -> dict[str, str]:
        return {"x-gladia-key": require_env("GLADIA_API_KEY", self.key)}

    def _languages(self, clip: AudioClip) -> list[str]:
        language = self.options.get("language", clip.language.split("-")[0])
        return [str(language)] if language else []

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Init a live session over REST, then stream PCM over its returned socket."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()

        receive_partial = bool(self.options.get("receive_partial_transcripts", True))
        receive_speech_events = bool(self.options.get("receive_speech_events", True))
        endpointing = float(self.options.get("endpointing", 0.05))
        max_duration = float(self.options.get("maximum_duration_without_endpointing", 5.0))

        body: dict[str, Any] = {
            "encoding": "wav/pcm",
            "bit_depth": 16,
            "sample_rate": clip.sample_rate,
            "channels": 1,
            "model": str(self.options.get("model", "solaria-1")),
            "language_config": {
                "languages": self._languages(clip),
                "code_switching": bool(self.options.get("code_switching", False)),
            },
            "pre_processing": {
                "audio_enhancer": bool(self.options.get("audio_enhancer", False)),
                "speech_threshold": float(self.options.get("speech_threshold", 0.6)),
            },
            "endpointing": endpointing,
            "maximum_duration_without_endpointing": max_duration,
            "messages_config": {
                "receive_partial_transcripts": receive_partial,
                "receive_final_transcripts": True,
                "receive_speech_events": receive_speech_events,
            },
        }

        init_url = INIT_URL
        region = self.options.get("region")
        if region:
            init_url = f"{INIT_URL}?{urlencode({'region': str(region)})}"

        init_response = await self.http.post(
            init_url,
            headers={**self._auth(), "Content-Type": "application/json"},
            json=body,
        )
        raise_for_status(init_response, self.key)
        session = init_response.json()
        result.raw["session_id"] = session.get("id")

        async def stop_recording(socket: ClientConnection) -> None:
            await socket.send(orjson.dumps({"type": "stop_recording"}).decode())

        await run_stream(
            url=session["url"],
            headers={},
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=_handle_message,
            on_input_done=stop_recording,
        )

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        if receive_speech_events:
            result.raw["eou_source"] = "speech_end"
            result.raw["endpoint_config"] = {
                "endpointing": endpointing,
                "maximum_duration_without_endpointing": max_duration,
            }
        return result


def _handle_message(payload: Any, timeline: StreamTimeline) -> bool:
    """Record one live-session event.

    ``speech_end`` is a bare VAD marker Gladia documents as the signal to
    drive turn-taking, so it alone carries :class:`EventKind.EOU`. A
    transcript's ``is_final`` marks the endpointing-driven close of that
    utterance's decoding — genuine, but a segment boundary rather than a
    turn decision, so it keeps the default ``segment_final``/``interim``
    kind derived from ``is_final``. ``post_final_transcript`` ends the
    session once post-processing completes; Gladia closes the socket right
    after it, so returning here only saves the finalize grace period from
    being spent waiting on a socket that is already gone.
    """
    if not isinstance(payload, dict):
        return False
    kind = payload.get("type")
    if kind == "speech_end":
        timeline.record("", is_final=False, kind=EventKind.EOU)
        return False
    if kind == "post_final_transcript":
        return True
    if kind != "transcript":
        return False

    data = payload.get("data") or {}
    utterance = data.get("utterance") or {}
    timeline.record(str(utterance.get("text", "")), is_final=bool(data.get("is_final")))
    return False
