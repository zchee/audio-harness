"""AssemblyAI Universal-3.5 pro speech-to-text adapter."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urlencode

import orjson
from websockets.asyncio.client import ClientConnection

from audio_harness.audio import wrap_wav
from audio_harness.config import require_env
from audio_harness.types import AudioClip, EventKind, Mode, SttResult

from .base import StreamTimeline, SttProvider, raise_for_status, register
from .ws import StreamProtocolError, run_stream


API_BASE = "https://api.assemblyai.com/v2"
STREAM_URL = "wss://streaming.assemblyai.com/v3/ws"
POLL_INTERVAL_S = 0.5


@register
class AssemblyAIUniversal35Pro(SttProvider):
    """AssemblyAI Universal-3.5 pro over the async and streaming APIs.

    Options:
        model: Speech model identifier; defaults to ``universal-3-5-pro``.
        format_turns: Whether streaming turns arrive punctuated and cased.
        language: Streaming language pin; defaults to the clip's language.
        end_of_turn_confidence_threshold: Confidence (0-1, server default
            0.4) above which the model may end a turn early.
        min_turn_silence: Minimum silence in ms before an end-of-turn check
            fires (server default 400).
        max_turn_silence: Silence in ms that forces a turn to end regardless
            of confidence (server default 1280).
        vad_threshold: VAD sensitivity (server default 0.4).

    The v3 socket accepts (as of 2026-08): en, es, de, fr, it, pt, tr, nl,
    sv, no, da, fi, hi, vi, ar, he, ja, ur, zh, ru, ko, multi. Anything else
    is refused at the handshake with a validation error — which is the honest
    outcome. Left unpinned, the socket auto-detects per utterance instead and
    produced Hebrew, Japanese and Arabic transcripts for one Hungarian lane,
    scoring 90-120% WER that reads as terrible accuracy rather than as the
    coverage gap it actually is.
    """

    key = "assemblyai-universal35pro"
    vendor = "assemblyai"
    supports_batch = True
    supports_stream = True
    # The v3 socket rejects frames outside 50-1000 ms with an Input Duration
    # Violation, so the 20 ms telephony default has to be widened here.
    min_chunk_ms = 50
    max_chunk_ms = 1000
    # AssemblyAI keeps counting a session against the concurrency limit for a
    # while after the socket closes. With a short gap the refusals cascade: a
    # refused clip returns immediately instead of spending its audio duration
    # streaming, so the next attempt lands even deeper inside the window.
    settle_ms = 3000

    def _model(self) -> str:
        return str(self.options.get("model", "universal-3-5-pro"))

    def _auth(self) -> dict[str, str]:
        return {"Authorization": require_env("ASSEMBLYAI_API_KEY", self.key)}

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Upload the clip, request transcription, then poll until complete."""
        result = self._result(clip, Mode.BATCH)
        started = time.perf_counter()

        upload = await self.http.post(
            f"{API_BASE}/upload",
            headers=self._auth(),
            content=wrap_wav(clip.pcm, clip.sample_rate),
        )
        raise_for_status(upload, self.key)

        created = await self.http.post(
            f"{API_BASE}/transcript",
            headers={**self._auth(), "Content-Type": "application/json"},
            json={
                "audio_url": upload.json()["upload_url"],
                # The singular `speech_model` was deprecated; the batch API now
                # takes a preference-ordered list and falls back down it.
                "speech_models": [self._model()],
                "language_code": clip.language.split("-")[0],
            },
        )
        raise_for_status(created, self.key)
        transcript_id = created.json()["id"]

        payload = await self._poll(transcript_id)
        result.total_s = time.perf_counter() - started
        result.text = str(payload.get("text") or "")
        result.raw["response"] = payload
        if payload.get("status") == "error":
            result.error = str(payload.get("error", "transcription failed"))
        return result

    async def _poll(self, transcript_id: str) -> dict[str, Any]:
        """Poll a transcript until it reaches a terminal status."""
        url = f"{API_BASE}/transcript/{transcript_id}"
        while True:
            response = await self.http.get(url, headers=self._auth())
            raise_for_status(response, self.key)
            payload: dict[str, Any] = response.json()
            if payload.get("status") in {"completed", "error"}:
                return payload
            await asyncio.sleep(POLL_INTERVAL_S)

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Stream PCM over the v3 streaming socket and collect turn events."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        params = {
            "sample_rate": str(clip.sample_rate),
            "encoding": "pcm_s16le",
            "speech_model": self._model(),
            "format_turns": str(self.options.get("format_turns", True)).lower(),
            # Without this the socket auto-detects per utterance; on Hungarian
            # audio it produced Hebrew, Japanese and Arabic transcripts in one
            # lane. The batch endpoint already pins language_code — parity.
            "language": str(self.options.get("language", clip.language.split("-")[0])),
        }
        knobs = {
            name: self.options[name]
            for name in (
                "end_of_turn_confidence_threshold",
                "min_turn_silence",
                "max_turn_silence",
                "vad_threshold",
            )
            if name in self.options
        }
        params.update({name: str(value) for name, value in knobs.items()})

        async def terminate(socket: ClientConnection) -> None:
            await socket.send(orjson.dumps({"type": "Terminate"}).decode())

        await run_stream(
            url=f"{STREAM_URL}?{urlencode(params)}",
            headers=self._auth(),
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=_TurnHandler(),
            on_input_done=terminate,
        )

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        # AssemblyAI's end_of_turn is a genuine end-of-utterance decision and
        # its turn detection is always on, so the lane is always EOU-capable.
        result.raw["eou_source"] = "end_of_turn"
        result.raw["endpoint_config"] = knobs
        return result


class _TurnHandler:
    """Records streaming events, deduplicating end-of-turn decisions.

    Each ``Turn`` frame restates the whole transcript for that turn, so only
    the ``end_of_turn`` frames are joined to form the final transcript. With
    ``format_turns`` the server may emit two ``end_of_turn`` frames for one
    ``turn_order`` (unformatted, then formatted); the turn's *decision* time
    is the first one, so only that frame carries the EOU kind.
    """

    __slots__ = ("_eou_turns",)

    def __init__(self) -> None:
        self._eou_turns: set[int] = set()

    def __call__(self, payload: Any, timeline: StreamTimeline) -> bool:
        if not isinstance(payload, dict):
            return False
        kind = payload.get("type")

        if kind == "Error" or payload.get("error"):
            raise StreamProtocolError(f"assemblyai: {payload.get('error') or payload.get('message') or payload}")
        if kind == "Termination":
            return True
        if kind != "Turn":
            return False

        end_of_turn = bool(payload.get("end_of_turn"))
        turn_order = int(payload.get("turn_order") or 0)
        first_decision = end_of_turn and turn_order not in self._eou_turns
        if first_decision:
            self._eou_turns.add(turn_order)

        timeline.record(
            str(payload.get("transcript", "")),
            is_final=end_of_turn,
            kind=EventKind.EOU if first_decision else "",
        )
        return False
