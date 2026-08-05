"""Speechmatics real-time and batch speech-to-text adapters.

Standard and Enhanced are the same service at two accuracy settings, selected
by ``operating_point``, so both benchmark entries share one implementation and
one API key.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import orjson
from websockets.asyncio.client import ClientConnection

from ..audio import wrap_wav
from ..config import require_env
from ..types import AudioClip, Mode, SttResult
from .base import StreamTimeline, SttProvider, raise_for_status, register
from .ws import StreamProtocolError, run_stream

BATCH_URL = "https://asr.api.speechmatics.com/v2/jobs"
STREAM_URL = "wss://eu2.rt.speechmatics.com/v2"
POLL_INTERVAL_S = 1.0


class _SpeechmaticsBase(SttProvider):
    """Shared transport for both Speechmatics operating points.

    Options:
        operating_point: ``standard`` or ``enhanced``; set by each subclass.
        max_delay: Latency/accuracy trade-off in seconds, 0.7 to 4.0. Lower
            values finalize sooner at some cost in accuracy.
        stream_url: Regional realtime endpoint override.
    """

    vendor = "speechmatics"
    supports_batch = True
    supports_stream = True
    operating_point = "standard"

    def _auth(self) -> dict[str, str]:
        key = require_env("SPEECHMATICS_API_KEY", self.key)
        return {"Authorization": f"Bearer {key}"}

    def _transcription_config(self, clip: AudioClip) -> dict[str, Any]:
        return {
            "language": clip.language.split("-")[0],
            "operating_point": self.options.get(
                "operating_point", self.operating_point
            ),
        }

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Submit a batch job, then poll for the plain-text transcript."""
        result = self._result(clip, Mode.BATCH)
        started = time.perf_counter()
        config = {
            "type": "transcription",
            "transcription_config": self._transcription_config(clip),
        }

        created = await self.http.post(
            BATCH_URL,
            headers=self._auth(),
            files={"data_file": ("audio.wav", wrap_wav(clip.pcm, clip.sample_rate))},
            data={"config": orjson.dumps(config).decode()},
        )
        raise_for_status(created, self.key)
        job_id = created.json()["id"]

        text = await self._poll(job_id)
        result.total_s = time.perf_counter() - started
        result.text = text
        result.raw["job_id"] = job_id
        return result

    async def _poll(self, job_id: str) -> str:
        """Wait for a job to finish and return its plain-text transcript."""
        status_url = f"{BATCH_URL}/{job_id}"
        while True:
            status = await self.http.get(status_url, headers=self._auth())
            raise_for_status(status, self.key)
            state = status.json().get("job", {}).get("status")
            if state == "done":
                break
            if state in {"rejected", "expired"}:
                raise StreamProtocolError(f"speechmatics job {job_id}: {state}")
            await asyncio.sleep(POLL_INTERVAL_S)

        transcript = await self.http.get(
            f"{status_url}/transcript",
            headers=self._auth(),
            params={"format": "txt"},
        )
        raise_for_status(transcript, self.key)
        return transcript.text.strip()

    async def transcribe_stream(
        self, clip: AudioClip, *, chunk_ms: int, realtime: bool
    ) -> SttResult:
        """Stream PCM over the v2 realtime socket, tracking sequence numbers."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        sent_chunks = 0

        config = self._transcription_config(clip)
        config["enable_partials"] = True
        if "max_delay" in self.options:
            config["max_delay"] = float(self.options["max_delay"])

        async def start_recognition(socket: ClientConnection) -> None:
            await socket.send(
                orjson.dumps(
                    {
                        "message": "StartRecognition",
                        "audio_format": {
                            "type": "raw",
                            "encoding": "pcm_s16le",
                            "sample_rate": clip.sample_rate,
                        },
                        "transcription_config": config,
                    }
                ).decode()
            )

        def count_chunk(chunk: bytes) -> bytes:
            nonlocal sent_chunks
            sent_chunks += 1
            return chunk

        async def end_of_stream(socket: ClientConnection) -> None:
            await socket.send(
                orjson.dumps(
                    {"message": "EndOfStream", "last_seq_no": sent_chunks}
                ).decode()
            )

        await run_stream(
            url=str(self.options.get("stream_url", STREAM_URL)),
            headers=self._auth(),
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=_handle_message,
            on_open=start_recognition,
            encode_chunk=count_chunk,
            on_input_done=end_of_stream,
        )

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        return result


def _handle_message(payload: Any, timeline: StreamTimeline) -> bool:
    """Record one realtime event.

    ``AddTranscript`` frames carry successive final segments, so the full
    transcript is their concatenation; ``AddPartialTranscript`` frames are
    interim hypotheses for the segment in progress.
    """
    if not isinstance(payload, dict):
        return False
    message = payload.get("message")

    if message == "EndOfTranscript":
        return True
    if message == "Error":
        raise StreamProtocolError(
            f"speechmatics: {payload.get('type')}: {payload.get('reason')}"
        )
    if message not in {"AddTranscript", "AddPartialTranscript"}:
        return False

    text = str(payload.get("metadata", {}).get("transcript", ""))
    timeline.record(text.strip(), is_final=message == "AddTranscript")
    return False


@register
class SpeechmaticsStandard(_SpeechmaticsBase):
    """Speechmatics at the ``standard`` operating point."""

    key = "speechmatics-standard"
    operating_point = "standard"


@register
class SpeechmaticsEnhanced(_SpeechmaticsBase):
    """Speechmatics at the ``enhanced`` operating point."""

    key = "speechmatics-enhanced"
    operating_point = "enhanced"
