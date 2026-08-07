"""Tests for the OpenAI gpt-4o-transcribe Realtime STT adapter's wire protocol.

The fake server speaks the unified 2025+ Realtime transcription-session
shape: ``session.type: "transcription"`` with nested ``audio.input.{format,
transcription, turn_detection}``, connecting via ``?intent=transcription``
(never a conversational ``model=``) with an ``OpenAI-Beta: realtime=v1``
handshake header. There is no explicit end-of-input message in this
protocol — server VAD owns turn detection — so the fake server treats a
quiet socket (no frame for ``QUIET_S``) as "all audio received", mirroring
how the real vendor has no terminator either.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
import os
from urllib.parse import parse_qsl, urlsplit

import orjson
import pytest
import websockets
from websockets.asyncio.server import ServerConnection, serve

from audio_harness import stt
from audio_harness.stt import openai
from audio_harness.stt.ws import StreamProtocolError
from audio_harness.types import AudioClip


def make_clip(seconds: float = 0.3, rate: int = 16000) -> AudioClip:
    """Build a silent clip of a known duration."""
    return AudioClip(
        clip_id="c1",
        pcm=b"\x00\x00" * int(rate * seconds),
        sample_rate=rate,
        duration_s=seconds,
        reference="hello wonderful world",
        language="en-US",
        source_path="<memory>",
    )


@pytest.fixture
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials for the mocked protocol tests.

    Not autouse: :class:`TestLiveSmoke` needs the *real* environment
    variable, and this fixture would otherwise clobber it.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class FakeOpenAiWs:
    """Speaks the Realtime transcription-session protocol the adapter expects."""

    QUIET_S = 0.1

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.received_audio = bytearray()
        self.request_path: str | None = None
        self.auth_header: str | None = None
        self.beta_header: str | None = None
        self.item_id = "item_1"
        self.speech_stopped = True
        self.text_deltas = ["hello", " wonderful", " world"]
        self.transcript = "hello wonderful world"
        self.fail_with: str | None = None

    async def __call__(self, socket: ServerConnection) -> None:
        if socket.request is not None:
            self.request_path = socket.request.path
            self.auth_header = socket.request.headers.get("Authorization")
            self.beta_header = socket.request.headers.get("OpenAI-Beta")

        while True:
            try:
                frame = await asyncio.wait_for(socket.recv(), timeout=self.QUIET_S)
            except TimeoutError, websockets.ConnectionClosed:
                break
            message = orjson.loads(frame)
            self.messages.append(message)
            if message.get("type") == "input_audio_buffer.append":
                self.received_audio.extend(base64.b64decode(message["audio"]))

        if self.fail_with is not None:
            await socket.send(orjson.dumps({"type": "error", "error": {"message": self.fail_with}}).decode())
            return

        if self.speech_stopped:
            await socket.send(
                orjson.dumps({"type": "input_audio_buffer.speech_stopped", "item_id": self.item_id}).decode()
            )
        for delta in self.text_deltas:
            await socket.send(
                orjson.dumps({
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": self.item_id,
                    "delta": delta,
                }).decode()
            )
        await socket.send(
            orjson.dumps({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": self.item_id,
                "transcript": self.transcript,
            }).decode()
        )


@pytest.fixture
async def openai_ws(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeOpenAiWs]:
    """Run a fake Realtime transcription endpoint and point the adapter at it."""
    handler = FakeOpenAiWs()
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(openai, "STREAM_URL", f"ws://127.0.0.1:{port}?intent=transcription")
        yield handler


class TestRealtimeProtocol:
    """The adapter's wire behavior against a real local WebSocket."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_session_update_precedes_audio(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.2)
        adapter = stt.create("openai-gpt4o-transcribe")

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert result.error is None, result.error
        first = openai_ws.messages[0]
        assert first["type"] == "session.update"
        session = first["session"]
        assert isinstance(session, dict)
        assert session["type"] == "transcription"
        audio_input = session["audio"]["input"]
        assert audio_input["format"] == {"type": "audio/pcm", "rate": 24000}
        assert audio_input["transcription"]["model"] == "gpt-4o-transcribe"
        assert audio_input["transcription"]["language"] == "en"
        assert audio_input["turn_detection"]["type"] == "server_vad"

    async def test_audio_is_resampled_to_24khz_before_streaming(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.2, rate=16000)
        adapter = stt.create("openai-gpt4o-transcribe")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        # Not an exact 1.5x multiply: soxr's resampler filter has edge
        # effects, so this only pins the ballpark, not a precise byte count.
        expected = int(len(clip.pcm) * 24000 / 16000)
        assert len(openai_ws.received_audio) == pytest.approx(expected, rel=0.05)
        assert bytes(openai_ws.received_audio) != clip.pcm

    async def test_native_24khz_clip_is_streamed_unmodified(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.2, rate=24000)
        adapter = stt.create("openai-gpt4o-transcribe")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert bytes(openai_ws.received_audio) == clip.pcm

    async def test_intent_query_parameter_not_a_model_parameter(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("openai-gpt4o-transcribe")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert openai_ws.request_path is not None
        query = dict(parse_qsl(urlsplit(openai_ws.request_path).query))
        assert query == {"intent": "transcription"}

    async def test_bearer_auth_header_is_sent_without_the_beta_header(self, openai_ws: FakeOpenAiWs) -> None:
        """No ``OpenAI-Beta: realtime=v1`` header: live-verified 2026-08-07,
        the GA endpoint rejects it as routing into the deprecated Beta API.
        """
        clip = make_clip(0.1)
        adapter = stt.create("openai-gpt4o-transcribe")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert openai_ws.auth_header == "Bearer test-key"
        assert openai_ws.beta_header is None

    async def test_model_option_overrides_the_default(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("openai-gpt4o-transcribe", {"model": "gpt-4o-mini-transcribe"})

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        session = openai_ws.messages[0]["session"]
        assert isinstance(session, dict)
        assert session["audio"]["input"]["transcription"]["model"] == "gpt-4o-mini-transcribe"

    async def test_vad_knobs_ride_the_turn_detection_config(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.1)
        adapter = stt.create(
            "openai-gpt4o-transcribe",
            {"threshold": 0.6, "prefix_padding_ms": 200, "silence_duration_ms": 400},
        )

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        session = openai_ws.messages[0]["session"]
        assert isinstance(session, dict)
        turn_detection = session["audio"]["input"]["turn_detection"]
        assert turn_detection["threshold"] == 0.6
        assert turn_detection["prefix_padding_ms"] == 200
        assert turn_detection["silence_duration_ms"] == 400
        assert result.raw["endpoint_config"] == turn_detection

    async def test_deltas_are_concatenated_into_the_growing_hypothesis(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.2)
        adapter = stt.create("openai-gpt4o-transcribe")

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        interim_texts = [p.text for p in result.partials if not p.is_final and p.kind != "eou"]
        assert interim_texts == ["hello", "hello wonderful", "hello wonderful world"], (
            "delta fragments are appended, not restated, so each interim hypothesis must accumulate the prior fragments"
        )

    async def test_completed_transcript_is_the_result(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.2)
        adapter = stt.create("openai-gpt4o-transcribe")

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert result.text == "hello wonderful world"
        assert result.ttft_s is not None
        assert result.finalize_s is not None

    async def test_speech_stopped_is_recorded_as_a_bare_eou_marker(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("openai-gpt4o-transcribe")

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        eou_events = [p for p in result.partials if p.kind == "eou"]
        assert len(eou_events) == 1
        assert eou_events[0].text == ""
        assert eou_events[0].is_final is False
        assert result.raw["eou_source"] == "speech_stopped"

    async def test_vendor_error_propagates(self, openai_ws: FakeOpenAiWs) -> None:
        openai_ws.fail_with = "invalid audio format"
        clip = make_clip(0.1)
        adapter = stt.create("openai-gpt4o-transcribe")

        with pytest.raises(StreamProtocolError, match="invalid audio format"):
            await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)


def _sent_transcription_config(ws: FakeOpenAiWs, index: int = 0) -> dict[str, object]:
    """Narrow one sent session update to its ``audio.input.transcription`` object."""
    session = ws.messages[index]["session"]
    assert isinstance(session, dict)
    return session["audio"]["input"]["transcription"]


class TestLiveTranscribeVariant:
    """``openai-live-transcribe``'s extra session knobs (GA field names,
    developers.openai.com/docs/guides/realtime-transcription, verified
    2026-08-07): plural ``languages`` instead of singular ``language``,
    ``keywords``, ``delay`` and ``prompt``.
    """

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_model_defaults_to_gpt_live_transcribe(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("openai-live-transcribe")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        transcription = _sent_transcription_config(openai_ws)
        assert transcription["model"] == "gpt-live-transcribe"

    async def test_languages_is_plural_and_language_is_never_sent(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("openai-live-transcribe")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        transcription = _sent_transcription_config(openai_ws)
        assert transcription["languages"] == ["en"]
        assert "language" not in transcription, (
            "gpt-live-transcribe rejects a request carrying both language and languages"
        )

    async def test_languages_option_overrides_the_clip_derived_default(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("openai-live-transcribe", {"languages": ["en", "fr"]})

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        transcription = _sent_transcription_config(openai_ws)
        assert transcription["languages"] == ["en", "fr"]

    async def test_keywords_delay_and_prompt_ride_the_transcription_config(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.1)
        adapter = stt.create(
            "openai-live-transcribe",
            {
                "keywords": ["premium plan", "AC-42", "billing"],
                "delay": "low",
                "prompt": "A customer support call about a premium plan and account AC-42.",
            },
        )

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        transcription = _sent_transcription_config(openai_ws)
        assert transcription["keywords"] == ["premium plan", "AC-42", "billing"]
        assert transcription["delay"] == "low"
        assert transcription["prompt"] == "A customer support call about a premium plan and account AC-42."

    async def test_keywords_delay_and_prompt_are_omitted_when_unset(self, openai_ws: FakeOpenAiWs) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("openai-live-transcribe")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        transcription = _sent_transcription_config(openai_ws)
        assert "keywords" not in transcription
        assert "delay" not in transcription
        assert "prompt" not in transcription

    async def test_turn_detection_is_disabled(self, openai_ws: FakeOpenAiWs) -> None:
        """Live-verified 2026-08-07: this model rejects ``session.update`` with
        a ``server_vad`` turn_detection object ("Turn detection is not
        supported for this transcription model."), so it must always send
        ``turn_detection: null``.
        """
        clip = make_clip(0.1)
        adapter = stt.create("openai-live-transcribe")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        session = openai_ws.messages[0]["session"]
        assert isinstance(session, dict)
        assert session["audio"]["input"]["turn_detection"] is None

    async def test_explicit_commit_is_sent_after_all_audio(self, openai_ws: FakeOpenAiWs) -> None:
        """No VAD means no server-side auto-finalize, so this model needs the
        client's own end-of-input signal -- an explicit
        ``input_audio_buffer.commit`` -- mirroring how Deepgram's adapter
        sends ``CloseStream``.
        """
        clip = make_clip(0.1)
        adapter = stt.create("openai-live-transcribe")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        commit_messages = [m for m in openai_ws.messages if m.get("type") == "input_audio_buffer.commit"]
        assert len(commit_messages) == 1
        last_append = max(i for i, m in enumerate(openai_ws.messages) if m.get("type") == "input_audio_buffer.append")
        commit_index = next(i for i, m in enumerate(openai_ws.messages) if m.get("type") == "input_audio_buffer.commit")
        assert commit_index > last_append, "the commit must follow every audio chunk, not race it"

    async def test_no_eou_marker_without_vendor_vad(self, openai_ws: FakeOpenAiWs) -> None:
        """A manual commit is ours, not the vendor's turn-taking decision, so
        it must stay a plain segment final -- same reasoning as
        stt/elevenlabs.py's manual-commit mode -- and this lane has no other
        EOU signal to report.
        """
        openai_ws.speech_stopped = False
        clip = make_clip(0.2)
        adapter = stt.create("openai-live-transcribe")

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert result.text == "hello wonderful world"
        assert "eou_source" not in result.raw
        assert "endpoint_config" not in result.raw
        assert not any(p.kind == "eou" for p in result.partials)
        finals = [p for p in result.partials if p.is_final]
        assert finals
        assert finals[-1].kind == "segment_final"


class TestCrossFamilyValidation:
    def test_family_is_openai(self) -> None:
        assert stt.family_of("openai-gpt4o-transcribe") == "openai"
        assert stt.family_of("openai-live-transcribe") == "openai"


LIVE_FLAG = "AUDIO_HARNESS_TEST_OPENAI_STT_LIVE"


def _needs(*names: str) -> pytest.MarkDecorator:
    missing = [name for name in names if not os.environ.get(name)]
    return pytest.mark.skipif(bool(missing), reason=f"{', '.join(names)} not set")


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run a couple of short live transcriptions (fractions of a cent total)",
)
class TestLiveSmoke:
    """A handful of short clips against the real vendor, en + ja.

    Minimal-API-testing policy: two short clips only, no bulk runs here.
    Silent PCM is enough to prove the wire round-trips (session config,
    server VAD, delta/completed events); it says nothing about accuracy.
    """

    @_needs("OPENAI_API_KEY")
    async def test_stream_en(self) -> None:
        clip = make_clip(0.8, rate=16000)
        adapter = stt.create("openai-gpt4o-transcribe")
        try:
            result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.total_s > 0

    @_needs("OPENAI_API_KEY")
    async def test_stream_ja(self) -> None:
        clip = AudioClip(
            clip_id="live-ja",
            pcm=b"\x00\x00" * int(16000 * 0.8),
            sample_rate=16000,
            duration_s=0.8,
            reference=None,
            language="ja-JP",
            source_path="<memory>",
        )
        adapter = stt.create("openai-gpt4o-transcribe")
        try:
            result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.total_s > 0


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run a couple of short live transcriptions (fractions of a cent total)",
)
class TestLiveTranscribeSmoke:
    """A handful of short clips against the real ``gpt-live-transcribe`` model, en + ja.

    Minimal-API-testing policy: two short clips only, no bulk runs here.
    """

    @_needs("OPENAI_API_KEY")
    async def test_stream_en(self) -> None:
        clip = make_clip(0.8, rate=16000)
        adapter = stt.create("openai-live-transcribe")
        try:
            result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.total_s > 0

    @_needs("OPENAI_API_KEY")
    async def test_stream_ja(self) -> None:
        clip = AudioClip(
            clip_id="live-ja",
            pcm=b"\x00\x00" * int(16000 * 0.8),
            sample_rate=16000,
            duration_s=0.8,
            reference=None,
            language="ja-JP",
            source_path="<memory>",
        )
        adapter = stt.create("openai-live-transcribe", {"languages": ["ja"]})
        try:
            result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.total_s > 0
