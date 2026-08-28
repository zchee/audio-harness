"""Protocol tests for the Gemini 3.5 batch and Live STT adapters.

The HTTP tests use ``httpx.MockTransport`` and the streaming tests use a real
local WebSocket server. The fake server speaks the Generate Content and
BidiGenerateContent shapes as live-verified on 2026-08-28 (see the adapter
docstrings) without requiring a Gemini API key.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Callable
from io import BytesIO
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import wave

import httpx
import numpy as np
import orjson
import pytest
import websockets
from websockets.asyncio.server import ServerConnection, serve

from audio_harness import stt
from audio_harness.audio import read_audio_samples
from audio_harness.stt import gemini
from audio_harness.stt.base import ProviderHttpError
from audio_harness.stt.ws import StreamProtocolError
from audio_harness.types import AudioClip, EventKind


def make_clip(seconds: float = 0.3, rate: int = 16000, language: str = "en-US") -> AudioClip:
    """Build a silent clip of a known duration."""
    return AudioClip(
        clip_id="c1",
        pcm=b"\x00\x00" * int(rate * seconds),
        sample_rate=rate,
        duration_s=seconds,
        reference="hello wonderful world",
        language=language,
        source_path="<memory>",
    )


@pytest.fixture
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials for mocked protocol tests, never live smoke tests."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def _batch_payload(*texts: str) -> dict[str, object]:
    """Build a Generate Content response in the v1alpha ``audioTranscription`` shape."""
    return {
        "candidates": [
            {
                "content": {"parts": [{"audioTranscription": {"text": text}} for text in texts], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {"promptTokenCount": 73, "totalTokenCount": 73},
        "modelVersion": "gemini-3.5-transcribe",
    }


class FakeGeminiWs:
    """Speak the subset of BidiGenerateContent used by the adapter.

    One spoken segment is emitted the way the live endpoint emits it:
    ``speechState: SPEECH``, cumulative ``interimInputTranscription`` frames,
    the ``inputTranscription`` final, ``generationComplete`` and finally
    ``speechState: NON_SPEECH``. By default the segment is flushed only in
    answer to the client's ``audio_stream_end``; ``vad_before_audio_end`` and
    ``vad_after_quiet_s`` make the server's own VAD flush it earlier, and
    ``silent`` hears no speech at all. Like the live endpoint, the fake never
    sends anything else after the terminator and never closes on its own, so
    every test also proves the adapter ends the session by itself.
    """

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.received_audio = bytearray()
        self.setup: dict[str, object] | None = None
        self.request_path: str | None = None
        self.auth_header: str | None = None
        self.audio_stream_end = False
        self.vad_before_audio_end = False
        self.vad_after_quiet_s: float | None = None
        self.silent = False
        self.fail_with: str | None = None
        self._sent_segment = False
        self._sent_speech_state = False

    async def __call__(self, socket: ServerConnection) -> None:
        """Receive setup/audio frames and emit deterministic transcript events."""
        if socket.request is not None:
            self.request_path = socket.request.path
            self.auth_header = socket.request.headers.get("x-goog-api-key")

        while True:
            try:
                if self.vad_after_quiet_s is None:
                    frame = await socket.recv()
                else:
                    frame = await asyncio.wait_for(socket.recv(), timeout=self.vad_after_quiet_s)
            except TimeoutError:
                if self.received_audio and not self._sent_segment and not self.silent:
                    await self._send_segment(socket)
                continue
            except websockets.ConnectionClosed:
                return

            if isinstance(frame, bytes):
                continue
            message = orjson.loads(frame)
            self.messages.append(message)

            if "setup" in message:
                self.setup = message["setup"]
                await socket.send(orjson.dumps({"setupComplete": {}}).decode())
                if self.fail_with is not None:
                    await socket.send(orjson.dumps({"error": {"message": self.fail_with}}).decode())
                    return
                continue

            realtime_input = message.get("realtime_input")
            if not isinstance(realtime_input, dict):
                continue
            media_chunks = realtime_input.get("media_chunks")
            if isinstance(media_chunks, list):
                for media in media_chunks:
                    if isinstance(media, dict) and media.get("data"):
                        self.received_audio.extend(base64.b64decode(media["data"]))
                if self.silent:
                    continue
                if self.received_audio and not self._sent_speech_state:
                    self._sent_speech_state = True
                    await _send_content(socket, {"speechState": "SPEECH"})
                if self.vad_before_audio_end and not self._sent_segment:
                    await self._send_segment(socket)

            if realtime_input.get("audio_stream_end") is True:
                self.audio_stream_end = True
                if not self._sent_segment and not self.silent:
                    await self._send_segment(socket)

    async def _send_segment(self, socket: ServerConnection) -> None:
        """Emit cumulative interims, the segment final, and the VAD end."""
        self._sent_segment = True
        hypothesis = ""
        for token in ("hello", " wonderful", " world"):
            hypothesis += token
            await _send_content(socket, {"interimInputTranscription": {"text": hypothesis}})
        await _send_content(socket, {"inputTranscription": {"text": hypothesis}})
        await _send_content(socket, {"generationComplete": True})
        await _send_content(socket, {"speechState": "NON_SPEECH"})


async def _send_content(socket: ServerConnection, content: dict[str, object]) -> None:
    """Send one ``serverContent`` frame."""
    await socket.send(orjson.dumps({"serverContent": content}).decode())


@pytest.fixture
async def gemini_ws(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeGeminiWs]:
    """Run a fake Gemini Live endpoint and point the adapter at it."""
    handler = FakeGeminiWs()
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(gemini, "STREAM_URL", f"ws://127.0.0.1:{port}")
        yield handler


Respond = Callable[[httpx.Request], httpx.Response]

PROMPT_CLOSE_S = 2.0
"""Wall-clock bound proving a session ended by itself, not by the driver's 15 s finalize timeout."""


def _mocked_batch_adapter(
    respond: Respond,
) -> tuple[gemini.GeminiTranscribe, list[httpx.Request]]:
    """Create a Gemini batch adapter whose transport records requests."""
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return respond(request)

    adapter = stt.create("gemini-35-transcribe")
    assert isinstance(adapter, gemini.GeminiTranscribe)
    adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return adapter, requests


class TestRegistration:
    """Both Gemini lanes expose the expected registry contract."""

    def test_batch_and_live_registration(self) -> None:
        batch = stt.create("gemini-35-transcribe")
        live = stt.create("gemini-35-transcribe-live")

        assert isinstance(batch, gemini.GeminiTranscribe)
        assert batch.vendor == batch.family == "gemini"
        assert batch.supports_batch is True
        assert batch.supports_stream is False
        assert isinstance(live, gemini.GeminiTranscribeLive)
        assert live.vendor == live.family == "gemini"
        assert live.supports_batch is False
        assert live.supports_stream is True
        assert stt.family_of("gemini-35-transcribe") == "gemini"
        assert stt.family_of("gemini-35-transcribe-live") == "gemini"


class TestBatchProtocol:
    """Gemini Generate Content request and response contract."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_request_shape_transcript_and_inline_wav(self) -> None:
        payload = _batch_payload("hello ", "world")
        adapter, requests = _mocked_batch_adapter(lambda request: httpx.Response(200, json=payload))
        clip = make_clip()

        result = await adapter.transcribe_batch(clip)

        assert result.ok, result.error
        assert result.text == "hello world"
        assert result.raw["response"] == payload
        assert result.total_s > 0
        assert result.ttft_s is None
        assert result.finalize_s is None

        [request] = requests
        assert request.method == "POST"
        assert str(request.url) == (
            "https://generativelanguage.googleapis.com/v1alpha/models/gemini-3.5-transcribe:generateContent"
        )
        assert request.headers["x-goog-api-key"] == "test-key"

        body = orjson.loads(request.content)
        assert body["generation_config"] == {
            "audio_transcription_config": {"mode": "VERBATIM", "language_codes": ["en-US"]},
        }
        inline = body["contents"][0]["parts"][0]["inline_data"]
        assert inline["mime_type"] == "audio/wav"
        wav_payload = base64.b64decode(inline["data"])
        with wave.open(BytesIO(wav_payload), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == clip.sample_rate
            assert handle.readframes(handle.getnframes()) == clip.pcm

    async def test_options_override_model_mode_languages_and_vocabulary(self) -> None:
        adapter, requests = _mocked_batch_adapter(lambda request: httpx.Response(200, json=_batch_payload("bonjour")))
        adapter.options.update({
            "model": "gemini-3.5-transcribe-preview",
            "mode": "SMART",
            "language_codes": ["fr-FR", "en-US"],
            "custom_vocabulary": ["Acme", "AC-42"],
        })

        result = await adapter.transcribe_batch(make_clip(language="fr-FR"))

        assert result.text == "bonjour"
        request = requests[0]
        assert request.url.path.endswith("/models/gemini-3.5-transcribe-preview:generateContent")
        body = orjson.loads(request.content)
        assert body["generation_config"]["audio_transcription_config"] == {
            "mode": "SMART",
            "language_codes": ["fr-FR", "en-US"],
            "custom_vocabulary": ["Acme", "AC-42"],
        }

    async def test_empty_part_yields_empty_transcript(self) -> None:
        # The v1beta surface answers the same request with a bare part and no
        # transcript; that must surface as an empty string, not a crash.
        payload = {"candidates": [{"content": {"parts": [{}], "role": "model"}, "finishReason": "STOP"}]}
        adapter, _ = _mocked_batch_adapter(lambda request: httpx.Response(200, json=payload))

        result = await adapter.transcribe_batch(make_clip())

        assert result.text == ""

    async def test_http_error_includes_vendor_body(self) -> None:
        adapter, _ = _mocked_batch_adapter(lambda request: httpx.Response(429, text="rate limited by Gemini"))

        with pytest.raises(ProviderHttpError, match="rate limited by Gemini"):
            await adapter.transcribe_batch(make_clip())


class TestLiveProtocol:
    """Gemini Live setup, framing, transcript assembly, and endpointing."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_setup_audio_auth_and_end_of_input(self, gemini_ws: FakeGeminiWs) -> None:
        clip = make_clip(0.2)
        adapter = stt.create("gemini-35-transcribe-live", {"eou_grace_ms": 0})

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert result.error is None, result.error
        assert gemini_ws.setup == {
            "model": "models/gemini-3.5-transcribe-live",
            "generation_config": {"response_modalities": ["TEXT"]},
            "input_audio_transcription": {"mode": "VERBATIM", "language_codes": ["en-US"]},
        }
        assert bytes(gemini_ws.received_audio) == clip.pcm
        assert gemini_ws.audio_stream_end is True
        assert gemini_ws.auth_header == "test-key"
        assert gemini_ws.request_path is not None
        assert "key=" not in urlsplit(gemini_ws.request_path).query
        assert result.raw["endpoint_config"] == {"eou_grace_ms": 0}

        media = [message for message in gemini_ws.messages if "media_chunks" in message.get("realtime_input", {})]
        assert media
        first_chunk = media[0]["realtime_input"]["media_chunks"][0]
        assert first_chunk["mime_type"] == "audio/pcm;rate=16000"
        assert base64.b64decode(first_chunk["data"])

    async def test_non_speech_while_audio_streams_is_eou(self, gemini_ws: FakeGeminiWs) -> None:
        gemini_ws.vad_before_audio_end = True
        adapter = stt.create("gemini-35-transcribe-live", {"eou_grace_ms": 0})

        started = time.perf_counter()
        result = await adapter.transcribe_stream(make_clip(0.6), chunk_ms=20, realtime=True)
        # The server sends nothing after the terminator once it has flushed
        # the segment, so the adapter must close the session itself instead
        # of waiting out the driver's finalize timeout.
        assert time.perf_counter() - started < PROMPT_CLOSE_S

        eou_markers = [partial for partial in result.partials if partial.kind == EventKind.EOU]
        assert len(eou_markers) == 1
        assert eou_markers[0].text == ""
        assert eou_markers[0].is_final is False
        finals = [partial for partial in result.partials if partial.is_final]
        assert [partial.text for partial in finals] == ["hello wonderful world"]
        assert finals[0].kind == EventKind.SEGMENT_FINAL
        assert finals[0].t_s < eou_markers[0].t_s
        assert result.text == "hello wonderful world"
        assert result.raw["eou_source"] == "speechState:NON_SPEECH"
        # generationComplete before the terminator ends a segment, not the
        # session: the rest of the audio and the terminator still go out.
        assert gemini_ws.audio_stream_end is True
        assert bytes(gemini_ws.received_audio) == make_clip(0.6).pcm

    async def test_flush_after_audio_stream_end_is_plain_final(self, gemini_ws: FakeGeminiWs) -> None:
        adapter = stt.create("gemini-35-transcribe-live")

        started = time.perf_counter()
        result = await adapter.transcribe_stream(make_clip(0.2), chunk_ms=20, realtime=False)
        assert time.perf_counter() - started < PROMPT_CLOSE_S

        interim_texts = [partial.text for partial in result.partials if not partial.is_final]
        assert interim_texts == ["hello", "hello wonderful", "hello wonderful world"]
        assert result.text == "hello wonderful world"
        finals = [partial for partial in result.partials if partial.is_final]
        assert len(finals) == 1
        assert finals[0].text == "hello wonderful world"
        assert finals[0].kind == EventKind.SEGMENT_FINAL
        assert not any(partial.kind == EventKind.EOU for partial in result.partials)
        assert "eou_source" not in result.raw
        # No hold by default: the terminator follows the last audio byte.
        assert result.raw["endpoint_config"] == {"eou_grace_ms": 0}
        assert result.ttft_s is not None
        assert result.finalize_s is not None

    async def test_non_speech_in_grace_window_is_eou(self, gemini_ws: FakeGeminiWs) -> None:
        gemini_ws.vad_after_quiet_s = 0.1
        adapter = stt.create("gemini-35-transcribe-live", {"eou_grace_ms": 500})

        result = await adapter.transcribe_stream(make_clip(0.2), chunk_ms=20, realtime=False)

        eou_markers = [partial for partial in result.partials if partial.kind == EventKind.EOU]
        assert len(eou_markers) == 1
        assert result.text == "hello wonderful world"
        assert result.raw["eou_source"] == "speechState:NON_SPEECH"
        assert result.raw["endpoint_config"] == {"eou_grace_ms": 500}
        assert result.finalize_s is not None
        assert result.finalize_s < 0.5

    async def test_silent_clip_closes_right_after_terminator(self, gemini_ws: FakeGeminiWs) -> None:
        gemini_ws.silent = True
        adapter = stt.create("gemini-35-transcribe-live", {"eou_grace_ms": 0})

        started = time.perf_counter()
        result = await adapter.transcribe_stream(make_clip(0.2), chunk_ms=20, realtime=False)

        assert time.perf_counter() - started < PROMPT_CLOSE_S
        assert result.error is None, result.error
        assert result.text == ""
        assert result.partials == []
        assert result.ttft_s is None
        assert result.finalize_s is None
        assert gemini_ws.audio_stream_end is True
        assert "eou_source" not in result.raw

    async def test_non_16khz_audio_is_resampled(self, gemini_ws: FakeGeminiWs) -> None:
        clip = make_clip(0.2, rate=8000)
        adapter = stt.create("gemini-35-transcribe-live", {"eou_grace_ms": 0})

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        expected = int(len(clip.pcm) * 16000 / 8000)
        assert len(gemini_ws.received_audio) == pytest.approx(expected, rel=0.05)
        assert bytes(gemini_ws.received_audio) != clip.pcm

    async def test_model_and_language_options_are_sent(self, gemini_ws: FakeGeminiWs) -> None:
        adapter = stt.create(
            "gemini-35-transcribe-live",
            {
                "model": "gemini-3.5-transcribe-live-preview",
                "mode": "SMART",
                "language_codes": ["ja-JP"],
                "eou_grace_ms": 0,
            },
        )

        await adapter.transcribe_stream(make_clip(language="ja-JP"), chunk_ms=20, realtime=False)

        assert gemini_ws.setup is not None
        assert gemini_ws.setup["model"] == "models/gemini-3.5-transcribe-live-preview"
        assert gemini_ws.setup["input_audio_transcription"] == {
            "mode": "SMART",
            "language_codes": ["ja-JP"],
        }

    async def test_error_frame_raises_stream_protocol_error(self, gemini_ws: FakeGeminiWs) -> None:
        gemini_ws.fail_with = "invalid audio format"
        adapter = stt.create("gemini-35-transcribe-live", {"eou_grace_ms": 0})

        with pytest.raises(StreamProtocolError, match="gemini: invalid audio format"):
            await adapter.transcribe_stream(make_clip(0.1), chunk_ms=20, realtime=False)


LIVE_FLAG = "AUDIO_HARNESS_TEST_GEMINI_STT_LIVE"
SPOKEN_WAV = "AUDIO_HARNESS_TEST_GEMINI_SPOKEN_WAV"


def _needs(*names: str) -> pytest.MarkDecorator:
    """Skip a live test when one of its real credential variables is absent."""
    missing = [name for name in names if not os.environ.get(name)]
    return pytest.mark.skipif(bool(missing), reason=f"{', '.join(names)} not set")


def _spoken_wav_available() -> bool:
    """Return whether the opt-in spoken-WAV path names an existing file."""
    path = os.environ.get(SPOKEN_WAV)
    return bool(path and Path(path).is_file())


def _spoken_wav_clip() -> AudioClip:
    """Load the configured spoken WAV and convert it to signed-16 PCM."""
    path_value = os.environ.get(SPOKEN_WAV)
    if not path_value:
        pytest.skip(f"set {SPOKEN_WAV} to run spoken-WAV smoke tests")
    path = Path(path_value)
    decoded = read_audio_samples(path)
    if decoded is None:
        pytest.skip(f"could not decode spoken WAV: {path}")
    samples, rate = decoded
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return AudioClip(
        clip_id="gemini-spoken-wav",
        pcm=pcm,
        sample_rate=rate,
        duration_s=len(samples) / rate,
        reference=None,
        language="en-US",
        source_path=str(path),
    )


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run short Gemini batch and streaming transcriptions",
)
class TestLiveSmoke:
    """Short opt-in calls against the real Gemini preview endpoints.

    The silent-clip stream test also exercises the drain path: the live
    endpoint sends nothing after ``audio_stream_end`` when it heard no
    speech, and the adapter closes the session itself once ``DRAIN_WAIT_S``
    passes.
    """

    @_needs("GEMINI_API_KEY")
    async def test_batch_en(self) -> None:
        adapter = stt.create("gemini-35-transcribe")
        try:
            result = await adapter.transcribe_batch(make_clip(0.8))
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.total_s > 0
        assert "response" in result.raw

    @_needs("GEMINI_API_KEY")
    async def test_stream_en(self) -> None:
        adapter = stt.create("gemini-35-transcribe-live")
        try:
            result = await adapter.transcribe_stream(make_clip(0.8), chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.total_s > 0

    @pytest.mark.skipif(
        not os.environ.get(LIVE_FLAG) or not _spoken_wav_available(),
        reason=f"set {LIVE_FLAG}=1 and {SPOKEN_WAV} to an existing spoken WAV",
    )
    @_needs("GEMINI_API_KEY")
    async def test_batch_spoken_wav(self) -> None:
        adapter = stt.create("gemini-35-transcribe")
        try:
            result = await adapter.transcribe_batch(_spoken_wav_clip())
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.text.strip() != ""

    @pytest.mark.skipif(
        not os.environ.get(LIVE_FLAG) or not _spoken_wav_available(),
        reason=f"set {LIVE_FLAG}=1 and {SPOKEN_WAV} to an existing spoken WAV",
    )
    @_needs("GEMINI_API_KEY")
    async def test_stream_spoken_wav(self) -> None:
        adapter = stt.create("gemini-35-transcribe-live")
        try:
            result = await adapter.transcribe_stream(_spoken_wav_clip(), chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.text.strip() != ""
        assert result.ttft_s is not None
        assert result.finalize_s is not None
        interims = [partial for partial in result.partials if not partial.is_final]
        assert interims, "the live endpoint emits interimInputTranscription frames before the final"
