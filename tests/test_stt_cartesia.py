"""Protocol tests for the Cartesia Ink-2 turn-based STT adapter."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

import orjson
import polars as pl
import pytest
from websockets.asyncio.server import ServerConnection, serve

from audio_harness.audio import load_clip_bytes
from audio_harness.stt import cartesia
from audio_harness.stt.ws import StreamProtocolError
from audio_harness.types import AudioClip, EventKind


LIVE_FLAG = "AUDIO_HARNESS_TEST_CARTESIA_LIVE"
LOGGER = logging.getLogger(__name__)
LIVE_AUDIO = Path(__file__).parents[1] / "data/hf/stt-benchmark-data/data/train-00000-of-00001.parquet"


def make_clip(seconds: float = 0.1, rate: int = 16000, language: str = "en-US") -> AudioClip:
    """Build a silent clip with stable protocol-test metadata."""
    return AudioClip(
        clip_id="c1",
        pcm=b"\x00\x00" * int(rate * seconds),
        sample_rate=rate,
        duration_s=seconds,
        reference="turn based speech recognition",
        language=language,
        source_path="<memory>",
    )


def make_live_clip() -> AudioClip:
    """Load one short real English utterance from the local benchmark corpus."""
    if not LIVE_AUDIO.is_file():
        raise FileNotFoundError(f"live smoke audio corpus not found: {LIVE_AUDIO}")
    row = pl.read_parquet(LIVE_AUDIO, n_rows=1).select("sample_id", "transcription", "audio").to_dicts()[0]
    audio = row["audio"]
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
        raise TypeError(f"live smoke row has no embedded audio bytes: {LIVE_AUDIO}")
    clip = load_clip_bytes(
        audio["bytes"],
        clip_id=str(row["sample_id"]),
        reference=str(row["transcription"]),
        language="en-US",
        source_path=f"{LIVE_AUDIO}#{row['sample_id']}",
    )
    if not 0.5 <= clip.duration_s <= 6.0:
        raise ValueError(f"live smoke clip must be 0.5-6.0 seconds, got {clip.duration_s:.3f}s")
    return clip


class FakeCartesiaWs:
    """Buffer audio until ``close``, emit turns, then close naturally."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.received_frames: list[str | bytes] = []
        self.path = ""
        self.api_key = ""

    async def __call__(self, socket: ServerConnection) -> None:
        assert socket.request is not None
        self.path = socket.request.path
        self.api_key = socket.request.headers.get("X-API-Key", "")

        async for frame in socket:
            self.received_frames.append(frame)
            if isinstance(frame, str) and orjson.loads(frame).get("type") == "close":
                break

        for event in self.events:
            await socket.send(orjson.dumps(event).decode())


async def run_adapter(
    events: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    *,
    options: dict[str, Any] | None = None,
    clip: AudioClip | None = None,
) -> tuple[Any, FakeCartesiaWs]:
    """Drive Ink-2 against the local turn-protocol server."""
    monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
    handler = FakeCartesiaWs(events)
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(cartesia, "STREAM_URL", f"ws://127.0.0.1:{port}/stt/turns/websocket")
        adapter = cartesia.CartesiaInk2(options)
        result = await adapter.transcribe_stream(clip or make_clip(), chunk_ms=20, realtime=False)
    return result, handler


class TestHandshake:
    """Authentication and URL query configuration match the vendor wire."""

    async def test_required_query_and_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await run_adapter([], monkeypatch)
        split = urlsplit(handler.path)
        assert split.path == "/stt/turns/websocket"
        assert parse_qs(split.query) == {
            "model": ["ink-2"],
            "encoding": ["pcm_s16le"],
            "sample_rate": ["16000"],
            "cartesia_version": ["2026-03-01"],
        }
        assert handler.api_key == "test-key"

    async def test_optional_knobs_only_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        options = {
            "turn_start_threshold": 0.7,
            "turn_eager_end_threshold": 0.35,
            "turn_end_threshold": 0.15,
            "turn_end_timeout_ms": 1200,
        }
        result, handler = await run_adapter([], monkeypatch, options=options)
        params = parse_qs(urlsplit(handler.path).query)
        assert {name: params[name] for name in options} == {
            "turn_start_threshold": ["0.7"],
            "turn_eager_end_threshold": ["0.35"],
            "turn_end_threshold": ["0.15"],
            "turn_end_timeout_ms": ["1200"],
        }
        assert result.raw["endpoint_config"] == options

    async def test_keyterms_are_repeatable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await run_adapter([], monkeypatch, options={"keyterm": ["Codex", "WebSocket"]})
        assert parse_qs(urlsplit(handler.path).query)["keyterm"] == ["Codex", "WebSocket"]

    async def test_audio_is_binary_and_close_is_last(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clip = make_clip()
        _, handler = await run_adapter([], monkeypatch, clip=clip)
        assert b"".join(frame for frame in handler.received_frames if isinstance(frame, bytes)) == clip.pcm
        assert orjson.loads(handler.received_frames[-1]) == {"type": "close"}
        assert (
            sum(
                isinstance(frame, str) and orjson.loads(frame).get("type") == "close"
                for frame in handler.received_frames
            )
            == 1
        )


class TestEventMapping:
    """Only definitive turn ends become endpointing events and finals."""

    async def test_turn_updates_and_ends_map_to_timeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            {"type": "connected"},
            {"type": "turn.start", "turn_id": "one"},
            {"type": "turn.update", "turn_id": "one", "transcript": "Hello"},
            {"type": "turn.end", "turn_id": "one", "transcript": "Hello world."},
            {"type": "turn.update", "turn_id": "two", "transcript": "Next"},
            {"type": "turn.end", "turn_id": "two", "transcript": "Next turn."},
        ]
        result, _ = await run_adapter(events, monkeypatch)
        assert result.text == "Hello world. Next turn."
        assert [(partial.text, partial.is_final, partial.kind) for partial in result.partials] == [
            ("Hello", False, EventKind.INTERIM),
            ("Hello world.", True, EventKind.EOU),
            ("Next", False, EventKind.INTERIM),
            ("Next turn.", True, EventKind.EOU),
        ]
        assert result.raw["eou_source"] == "turn.end"

    async def test_eager_events_are_metadata_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            {
                "type": "turn.eager_end",
                "turn_id": "one",
                "confidence": 0.81,
                "transcript": "not recorded",
            },
            {"type": "turn.resume", "turn_id": "one", "transcript": "also not recorded"},
            {"type": "turn.end", "turn_id": "one", "transcript": "Definitive."},
        ]
        result, _ = await run_adapter(events, monkeypatch)
        assert [partial.text for partial in result.partials] == ["Definitive."]
        eager = result.raw["eager_events"]
        assert [event["event"] for event in eager] == ["turn.eager_end", "turn.resume"]
        assert eager[0]["turn_id"] == "one"
        assert eager[0]["confidence"] == 0.81
        assert all(isinstance(event["t_s"], float) for event in eager)

    async def test_server_close_acknowledgment_returns_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _ = await run_adapter(
            [{"type": "turn.end", "turn_id": "one", "transcript": "Buffered audio flushed."}],
            monkeypatch,
        )
        assert result.text == "Buffered audio flushed."
        assert result.error is None


class TestErrorFrames:
    """Protocol errors retain the complete Cartesia payload."""

    async def test_error_frame_carries_full_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {"type": "error", "code": "invalid_sample_rate", "message": "rate 123 is unsupported"}
        with pytest.raises(StreamProtocolError, match=re.escape(orjson.dumps(payload).decode())):
            await run_adapter([payload], monkeypatch)


class TestValidation:
    """Invalid local inputs fail before authentication or networking."""

    async def test_non_english_clip_rejected_before_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        adapter = cartesia.CartesiaInk2()
        with pytest.raises(ValueError, match=r"English only.*fr-FR"):
            await adapter.transcribe_stream(make_clip(language="fr-FR"), chunk_ms=20, realtime=False)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("turn_start_threshold", 0.49),
            ("turn_eager_end_threshold", 0.61),
            ("turn_end_threshold", 0.51),
            ("turn_end_timeout_ms", 639),
            ("turn_end_timeout_ms", 640.0),
        ],
    )
    async def test_out_of_range_knob_rejected_before_auth(
        self,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        value: object,
    ) -> None:
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        adapter = cartesia.CartesiaInk2({name: value})
        with pytest.raises(ValueError, match=name):
            await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)

    async def test_keyterm_requires_string_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        adapter = cartesia.CartesiaInk2({"keyterm": "not-a-list"})
        with pytest.raises(ValueError, match="keyterm"):
            await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)


@pytest.mark.skipif(
    not os.environ.get("CARTESIA_API_KEY") or not os.environ.get(LIVE_FLAG),
    reason=f"live smoke needs CARTESIA_API_KEY and {LIVE_FLAG}=1",
)
class TestLiveSmoke:
    """One short real English turn, enabled only by the vendor live flag."""

    async def test_stream_returns_turn(self) -> None:
        clip = make_live_clip()
        adapter = cartesia.CartesiaInk2()
        result = await adapter.transcribe_stream(clip, chunk_ms=80, realtime=True)
        evidence = {
            "transcript": result.text,
            "ttft_s": result.ttft_s,
            "finalize_s": result.finalize_s,
            "total_s": result.total_s,
            "eou_events": sum(partial.kind == EventKind.EOU for partial in result.partials),
        }
        LOGGER.info("cartesia_live_smoke=%s", json.dumps(evidence, sort_keys=True))
        assert result.error is None
        assert result.text
        assert evidence["eou_events"] >= 1
