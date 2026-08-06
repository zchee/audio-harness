"""Google Cloud Speech-to-Text v2 (Chirp 3) adapter.

Chirp 3 is only reachable over Google's gRPC transport, so unlike every other
adapter this one goes through the vendor SDK. Timing is still taken in this
process around the SDK calls, but the numbers include a gRPC stack the
WebSocket adapters do not pay for — keep that in mind when ranking latency.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech

from ..audio import pace_chunks
from ..config import require_env
from ..types import AudioClip, EventKind, Mode, SttResult
from .base import StreamTimeline, SttProvider, register


@register
class GoogleChirp3(SttProvider):
    """Google Cloud STT v2 with the Chirp 3 model.

    Requires application default credentials and an enabled
    ``speech.googleapis.com`` on the target project.

    Options:
        model: Model identifier; defaults to ``chirp_3``.
        project: Overrides ``GOOGLE_CLOUD_PROJECT``.
        location: Overrides ``GOOGLE_CLOUD_LOCATION``. Chirp 3 is served from
            regional and multi-region endpoints, not ``global``.
        enable_voice_activity_events: When true, streaming responses include
            ``SPEECH_ACTIVITY_END`` events — the API's end-of-utterance
            signal. Off by default.
    """

    key = "google-chirp3"
    vendor = "google"
    supports_batch = True
    supports_stream = True

    def __init__(self, options: dict[str, object] | None = None) -> None:
        """Initialize the adapter and defer client construction."""
        super().__init__(options)
        self._client: SpeechAsyncClient | None = None

    def _project(self) -> str:
        override = self.options.get("project")
        if override:
            return str(override)
        return require_env("GOOGLE_CLOUD_PROJECT", self.key)

    def _location(self) -> str:
        return str(
            self.options.get("location")
            or os.environ.get("GOOGLE_CLOUD_LOCATION", "us")
        )

    def _recognizer(self) -> str:
        return f"projects/{self._project()}/locations/{self._location()}/recognizers/_"

    def _speech_client(self) -> SpeechAsyncClient:
        """Return a client bound to the configured region's endpoint."""
        if self._client is None:
            location = self._location()
            options = (
                None
                if location == "global"
                else ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
            )
            self._client = SpeechAsyncClient(client_options=options)
        return self._client

    def _config(self, clip: AudioClip) -> cloud_speech.RecognitionConfig:
        """Build a recognition config pinned to the harness' PCM format."""
        return cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=clip.sample_rate,
                audio_channel_count=1,
            ),
            language_codes=[clip.language],
            model=str(self.options.get("model", "chirp_3")),
        )

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Send the whole clip in a single unary recognize call."""
        result = self._result(clip, Mode.BATCH)
        client = self._speech_client()
        started = time.perf_counter()
        response = await client.recognize(
            request=cloud_speech.RecognizeRequest(
                recognizer=self._recognizer(),
                config=self._config(clip),
                content=clip.pcm,
            )
        )
        result.total_s = time.perf_counter() - started
        result.text = " ".join(
            item.alternatives[0].transcript
            for item in response.results
            if item.alternatives
        ).strip()
        return result

    async def transcribe_stream(
        self, clip: AudioClip, *, chunk_ms: int, realtime: bool
    ) -> SttResult:
        """Stream the clip over a bidirectional recognize call."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        client = self._speech_client()

        vad_events = bool(self.options.get("enable_voice_activity_events"))
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self._config(clip),
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=True,
                enable_voice_activity_events=vad_events,
            ),
        )

        async def requests() -> AsyncIterator[cloud_speech.StreamingRecognizeRequest]:
            yield cloud_speech.StreamingRecognizeRequest(
                recognizer=self._recognizer(),
                streaming_config=streaming_config,
            )
            timeline.start()
            async for chunk in pace_chunks(clip, chunk_ms, realtime=realtime):
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)
            timeline.audio_complete()

        stream = await client.streaming_recognize(requests=requests())
        async for response in stream:
            _record_response(response, timeline)

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        if vad_events:
            result.raw["eou_source"] = "speech_activity_end"
            result.raw["endpoint_config"] = {"enable_voice_activity_events": True}
        return result


def _record_response(
    response: cloud_speech.StreamingRecognizeResponse, timeline: StreamTimeline
) -> None:
    """Record one streaming response's transcript and voice-activity events.

    ``SPEECH_ACTIVITY_END`` arrives as a bare event — no transcript rides on
    it — and is the v2 API's end-of-utterance decision. Result frames keep
    their segment-final semantics; Google's ``is_final`` is a decoding
    boundary, not an endpointing one.
    """
    if (
        response.speech_event_type
        == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END
    ):
        timeline.record("", is_final=False, kind=EventKind.EOU)
    for item in response.results:
        if item.alternatives:
            timeline.record(item.alternatives[0].transcript, is_final=item.is_final)
