"""Soniox stt-rt-v5 realtime and stt-async-v5 batch speech-to-text adapter."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
import orjson
from websockets.asyncio.client import ClientConnection

from audio_harness.audio import wrap_wav
from audio_harness.config import require_env
from audio_harness.types import AudioClip, EventKind, Mode, SttResult

from .base import StreamTimeline, SttProvider, raise_for_status, register
from .ws import StreamProtocolError, run_stream


API_BASE = "https://api.soniox.com/v1"
STREAM_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
POLL_INTERVAL_S = 1.0

_LOGGER = logging.getLogger(__name__)


@register
class SonioxRealtimeV5(SttProvider):
    """Soniox realtime transcription over its token-streaming WebSocket.

    Soniox streams sub-word tokens rather than sentence hypotheses: tokens
    arrive with ``is_final`` flipping from false to true as context accrues.
    The adapter therefore maintains the running transcript itself and records
    the cumulative string, so downstream code sees the same shape it gets from
    sentence-oriented vendors.

    Options:
        model: Model identifier; defaults to ``stt-rt-v5``.
        language_hints: Language codes that bias recognition.
        enable_endpoint_detection: When true, Soniox finalizes each utterance
            as it ends and emits a ``<end>`` marker token. Off by default.
        endpoint_sensitivity: How readily the model emits an endpoint.
        max_endpoint_delay_ms: Ceiling on endpoint delay after speech ends.
        endpoint_latency_adjustment_level: Vendor latency/accuracy trade-off.
        batch_model: Async model identifier used by the batch path; defaults
            to ``stt-async-v5``. The realtime ``model`` option is not shared
            because Soniox versions its realtime and async lineages apart.
        poll_timeout_s: Overall timeout in seconds for the batch upload,
            create, and result polling flow; defaults to 300 seconds.

    The batch path stores assets on Soniox's side — the uploaded file and the
    transcription object both persist until deleted — so cleanup is mandatory
    and unconditional: both DELETEs are always issued for whatever assets were
    created, even when the flow fails midway. Deletion failures are logged and
    recorded in ``result.raw["deleted"]``, never raised.
    """

    key = "soniox-rt-v5"
    vendor = "soniox"
    supports_batch = True
    supports_stream = True

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {require_env('SONIOX_API_KEY', self.key)}"}

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Upload a WAV, create an async transcription, poll, fetch, then delete.

        The flow follows Soniox's async pipeline: ``POST /v1/files`` uploads
        the audio, ``POST /v1/transcriptions`` references it by ``file_id``,
        ``GET /v1/transcriptions/{id}`` is polled until a terminal status, and
        ``GET /v1/transcriptions/{id}/transcript`` returns the text. Both
        stored assets are deleted unconditionally afterwards.
        """
        result = self._result(clip, Mode.BATCH)
        started = time.perf_counter()
        timeout_s = float(self.options.get("poll_timeout_s", 300.0))
        hints = self.options.get("language_hints") or [clip.language.split("-")[0]]
        # The batch path runs the async lineage, not the realtime model this
        # adapter is named after — record what was actually used so reports
        # can label the numbers honestly.
        batch_model = str(self.options.get("batch_model", "stt-async-v5"))
        result.raw["model"] = batch_model
        file_id: str | None = None
        transcription_id: str | None = None

        try:
            async with asyncio.timeout(timeout_s):
                upload = await self.http.post(
                    f"{API_BASE}/files",
                    headers=self._auth(),
                    files={"file": ("audio.wav", wrap_wav(clip.pcm, clip.sample_rate), "audio/wav")},
                )
                raise_for_status(upload, self.key)
                file_id = str(upload.json()["id"])

                created = await self.http.post(
                    f"{API_BASE}/transcriptions",
                    headers={**self._auth(), "Content-Type": "application/json"},
                    json={
                        "model": batch_model,
                        "file_id": file_id,
                        "language_hints": list(hints),
                    },
                )
                raise_for_status(created, self.key)
                transcription_id = str(created.json()["id"])
                result.raw["transcription_id"] = transcription_id

                payload = await self._poll(transcription_id)
                result.raw["response"] = payload
                if payload.get("status") == "error":
                    result.error = str(payload.get("error_message") or "transcription failed")
                else:
                    transcript = await self.http.get(
                        f"{API_BASE}/transcriptions/{transcription_id}/transcript",
                        headers=self._auth(),
                    )
                    raise_for_status(transcript, self.key)
                    result.text = str(transcript.json().get("text") or "")
        finally:
            deleted: dict[str, bool] = {}
            if transcription_id is not None:
                deleted["transcription"] = await self._delete_asset(f"/transcriptions/{transcription_id}")
            if file_id is not None:
                deleted["file"] = await self._delete_asset(f"/files/{file_id}")
            result.raw["deleted"] = deleted

        result.total_s = time.perf_counter() - started
        return result

    async def _poll(self, transcription_id: str) -> dict[str, Any]:
        """Poll a transcription until it reaches a terminal status."""
        url = f"{API_BASE}/transcriptions/{transcription_id}"
        while True:
            response = await self.http.get(url, headers=self._auth())
            raise_for_status(response, self.key)
            payload: dict[str, Any] = response.json()
            if payload.get("status") in {"completed", "error"}:
                return payload
            await asyncio.sleep(POLL_INTERVAL_S)

    async def _delete_asset(self, path: str) -> bool:
        """Delete a stored asset, reporting success rather than raising.

        Retention hygiene must never cost the transcription that already
        succeeded, so any failure — transport or HTTP status — is logged as a
        warning and surfaces as ``False`` in ``result.raw["deleted"]``.
        """
        try:
            response = await self.http.delete(f"{API_BASE}{path}", headers=self._auth())
        except httpx.HTTPError as exc:
            _LOGGER.warning("%s: failed to delete %s: %s", self.key, path, exc)
            return False
        if response.is_success:
            return True
        _LOGGER.warning(
            "%s: failed to delete %s: HTTP %d: %s",
            self.key,
            path,
            response.status_code,
            response.text.strip()[:500],
        )
        return False

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Stream PCM and reassemble the token stream into a transcript."""
        result = self._result(clip, Mode.STREAM)
        model = str(self.options.get("model", "stt-rt-v5"))
        result.raw["model"] = model
        timeline = StreamTimeline()
        hints = self.options.get("language_hints") or [clip.language.split("-")[0]]
        knobs = {
            name: self.options[name]
            for name in (
                "enable_endpoint_detection",
                "endpoint_sensitivity",
                "max_endpoint_delay_ms",
                "endpoint_latency_adjustment_level",
            )
            if name in self.options
        }

        async def configure(socket: ClientConnection) -> None:
            await socket.send(
                orjson.dumps({
                    "api_key": require_env("SONIOX_API_KEY", self.key),
                    "model": model,
                    "audio_format": "pcm_s16le",
                    "sample_rate": clip.sample_rate,
                    "num_channels": 1,
                    "language_hints": list(hints),
                    **knobs,
                }).decode()
            )

        async def end_of_audio(socket: ClientConnection) -> None:
            await socket.send("")

        accumulator = _TokenAccumulator()
        await run_stream(
            url=STREAM_URL,
            headers={},
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=accumulator,
            on_open=configure,
            on_input_done=end_of_audio,
        )

        result.text = timeline.last_final() or accumulator.finalized
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        if knobs.get("enable_endpoint_detection"):
            result.raw["eou_source"] = "end_token"
            result.raw["endpoint_config"] = knobs
        return result


class _TokenAccumulator:
    """Rebuilds a running transcript from Soniox's incremental token stream.

    Final tokens are appended permanently; non-final tokens form a provisional
    tail that is discarded on the next frame. Each recorded hypothesis is the
    full transcript so far, which is why the adapter reads it back with
    :meth:`StreamTimeline.last_final` rather than concatenating.
    """

    __slots__ = ("finalized",)

    def __init__(self) -> None:
        self.finalized = ""

    def __call__(self, payload: Any, timeline: StreamTimeline) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("error_code") or payload.get("error_message"):
            raise StreamProtocolError(f"soniox: {payload.get('error_code')}: {payload.get('error_message')}")

        tail = ""
        endpoints = 0
        for token in payload.get("tokens", []):
            text = str(token.get("text", ""))
            # Marker tokens are control flow, not transcript: <end> is the
            # endpoint decision (enable_endpoint_detection), <fin> confirms a
            # manual finalize. Splicing either into the text would corrupt
            # every accuracy metric downstream.
            if text == "<end>":
                endpoints += 1
                continue
            if text == "<fin>":
                continue
            if token.get("is_final"):
                self.finalized += text
            else:
                tail += text

        if self.finalized or tail:
            timeline.record((self.finalized + tail).strip(), is_final=not tail)
        for _ in range(endpoints):
            timeline.record("", is_final=False, kind=EventKind.EOU)

        return bool(payload.get("finished"))
