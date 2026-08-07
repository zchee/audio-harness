"""Protocol tests for the xAI Grok realtime speech-to-text adapter.

The fake server replays the frame sequence observed against the live
``wss://api.x.ai/v1/stt`` socket on 2026-08-07: sessions are configured
entirely through URL query parameters (a ``session.update`` config message
is rejected with "unknown variant"), every hypothesis arrives as a
``transcript.partial`` event, and each segment's final text is emitted twice
— once bare, once restated with ``speech_final: true`` when the endpointing
decision lands. The restated frame must not be concatenated a second time.
"""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import orjson
import pytest
from websockets.asyncio.server import ServerConnection, serve

from audio_harness import stt
from audio_harness.stt import xai
from audio_harness.stt.ws import StreamProtocolError
from audio_harness.types import AudioClip, EventKind


def make_clip(seconds: float = 0.1, rate: int = 16000, language: str = "en-US") -> AudioClip:
    """Build a silent clip of a known duration."""
    return AudioClip(
        clip_id="c1",
        pcm=b"\x00\x00" * int(rate * seconds),
        sample_rate=rate,
        duration_s=seconds,
        reference="how do I temper chocolate",
        language=language,
        source_path="<memory>",
    )


def _segment(text: str, start: float, *, is_final: bool, speech_final: bool) -> dict[str, Any]:
    """Shape one ``transcript.partial`` event as the live server sends it."""
    return {
        "type": "transcript.partial",
        "text": text,
        "words": [],
        "is_final": is_final,
        "speech_final": speech_final,
        "start": start,
        "duration": 1.0,
        "language": "en",
    }


class FakeXaiWs:
    """Speaks the query-configured protocol the live server exposes.

    Buffers binary audio and JSON control frames until ``audio.done``
    arrives, then replays ``self.events`` and lets the handler return so the
    library closes the socket with code 1000.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.received_audio = bytearray()
        self.received_json: list[dict[str, Any]] = []
        self.path = ""
        self.authorization = ""

    async def __call__(self, socket: ServerConnection) -> None:
        assert socket.request is not None
        self.path = socket.request.path
        self.authorization = socket.request.headers.get("Authorization", "")

        async for frame in socket:
            if isinstance(frame, bytes):
                self.received_audio.extend(frame)
                continue
            message = orjson.loads(frame)
            self.received_json.append(message)
            if message.get("type") == "audio.done":
                break

        await socket.send(orjson.dumps({"type": "transcript.created", "id": "t1"}).decode())
        for event in self.events:
            await socket.send(orjson.dumps(event).decode())
        await socket.send(orjson.dumps({"type": "transcript.done", "text": "", "words": []}).decode())


TWO_SEGMENT_EVENTS = [
    _segment("", 0.0, is_final=False, speech_final=False),
    _segment("How do I temper", 0.0, is_final=False, speech_final=False),
    _segment("How do I temper chocolate?", 0.0, is_final=True, speech_final=False),
    _segment("How do I temper chocolate?", 0.0, is_final=True, speech_final=True),
    _segment("", 3.5, is_final=False, speech_final=False),
    _segment("", 3.5, is_final=True, speech_final=False),
    _segment("It worked.", 3.5, is_final=True, speech_final=False),
    _segment("It worked.", 3.5, is_final=True, speech_final=True),
]


async def run_adapter(
    events: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any] | None = None,
) -> tuple[Any, FakeXaiWs]:
    """Drive the adapter against a fake server replaying ``events``."""
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    handler = FakeXaiWs(events)
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(xai, "STREAM_URL", f"ws://127.0.0.1:{port}")
        adapter = stt.create("xai-grok-stt", options=options or {})
        result = await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)
    return result, handler


class TestHandshake:
    """Session configuration travels in the URL, not a config message."""

    async def test_query_parameters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await run_adapter(TWO_SEGMENT_EVENTS, monkeypatch)
        params = dict(parse_qsl(urlsplit(handler.path).query))
        assert params == {
            "sample_rate": "16000",
            "encoding": "pcm",
            "language": "en",
            "interim_results": "true",
        }

    async def test_endpointing_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await run_adapter(TWO_SEGMENT_EVENTS, monkeypatch, options={"endpointing": 300})
        params = dict(parse_qsl(urlsplit(handler.path).query))
        assert params["endpointing"] == "300"

    async def test_bearer_auth_and_no_config_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await run_adapter(TWO_SEGMENT_EVENTS, monkeypatch)
        assert handler.authorization == "Bearer test-key"
        assert handler.received_json == [{"type": "audio.done"}]

    async def test_audio_sent_as_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await run_adapter(TWO_SEGMENT_EVENTS, monkeypatch)
        assert bytes(handler.received_audio) == make_clip().pcm


class TestTranscriptAssembly:
    """Restated ``speech_final`` frames must not double the transcript."""

    async def test_two_segments_concatenate_once_each(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _ = await run_adapter(TWO_SEGMENT_EVENTS, monkeypatch)
        assert result.text == "How do I temper chocolate? It worked."

    async def test_restated_final_becomes_bare_eou(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _ = await run_adapter(TWO_SEGMENT_EVENTS, monkeypatch)
        eou = [p for p in result.partials if p.kind == EventKind.EOU]
        assert len(eou) == 2
        assert all(not p.text for p in eou)

    async def test_lone_speech_final_keeps_its_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [_segment("Hello there.", 0.0, is_final=True, speech_final=True)]
        result, _ = await run_adapter(events, monkeypatch)
        assert result.text == "Hello there."
        assert [p.kind for p in result.partials if p.is_final] == [EventKind.EOU]

    async def test_eou_metadata_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _ = await run_adapter(TWO_SEGMENT_EVENTS, monkeypatch, options={"endpointing": 300})
        assert result.raw["eou_source"] == "speech_final"
        assert result.raw["endpoint_config"] == {"endpointing": 300}


class TestErrorFrames:
    """The raw payload survives into the raised error."""

    async def test_error_frame_carries_full_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [{"type": "error", "message": "Invalid message: unknown variant `session.update`"}]
        with pytest.raises(StreamProtocolError, match=re.escape("unknown variant `session.update`")):
            await run_adapter(events, monkeypatch)


@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or not os.environ.get("AUDIO_HARNESS_LIVE"),
    reason="live smoke needs XAI_API_KEY and AUDIO_HARNESS_LIVE=1",
)
class TestLiveSmoke:
    """One real socket round trip; costs a fraction of a cent."""

    async def test_stream_returns_text(self) -> None:
        adapter = stt.create("xai-grok-stt")
        clip = make_clip(seconds=1.0)
        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)
        assert result.error is None
