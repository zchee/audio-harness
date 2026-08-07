"""Protocol tests for Deepgram Flux's v2 turn-based streaming socket."""

from __future__ import annotations

from dataclasses import replace
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
from audio_harness.stt import deepgram
from audio_harness.stt.ws import StreamProtocolError
from audio_harness.types import AudioClip, EventKind


LOGGER = logging.getLogger(__name__)
LIVE_FLAG = "AUDIO_HARNESS_TEST_DEEPGRAM_LIVE"
LIVE_AUDIO = Path(__file__).parents[1] / "data/hf/stt-benchmark-data/data/train-00000-of-00001.parquet"


def make_clip(seconds: float = 0.1, rate: int = 16000) -> AudioClip:
    """Build a silent clip of a known duration."""
    return AudioClip(
        clip_id="c1",
        pcm=b"\x00\x00" * int(rate * seconds),
        sample_rate=rate,
        duration_s=seconds,
        reference="hello from the Deepgram Flux protocol test",
        language="en-US",
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
    # This corpus row is truncated mid-utterance with no trailing silence, so
    # Flux's end-of-turn confidence never crosses its threshold (observed
    # live 2026-08-08). Recorded corpora normally carry trailing silence —
    # append two seconds of it so the timeout-forced EndOfTurn has both the
    # silence it needs and slack for server-side processing lag: CloseStream
    # closes without draining pending turn decisions, so a decision landing
    # too close to end-of-audio is lost with shorter padding.
    silence = b"\x00\x00" * (clip.sample_rate * 2)
    return replace(clip, pcm=clip.pcm + silence, duration_s=clip.duration_s + 2.0)


def turn(event: str, turn_index: int, transcript: str = "", confidence: float = 0.8) -> dict[str, Any]:
    """Build one representative Flux ``TurnInfo`` frame."""
    return {
        "type": "TurnInfo",
        "event": event,
        "turn_index": turn_index,
        "transcript": transcript,
        "words": [],
        "end_of_turn_confidence": confidence,
        "audio_window_start": 0.0,
        "audio_window_end": 0.1,
    }


class FakeDeepgramFluxWs:
    """Capture client frames, then replay Flux events after ``CloseStream``."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.received_audio = bytearray()
        self.received_json: list[dict[str, Any]] = []
        self.received_frames: list[bytes | dict[str, Any]] = []
        self.path = ""
        self.authorization = ""

    async def __call__(self, socket: ServerConnection) -> None:
        assert socket.request is not None
        self.path = socket.request.path
        self.authorization = socket.request.headers.get("Authorization", "")
        await socket.send(orjson.dumps({"type": "Connected", "request_id": "r1"}).decode())

        async for frame in socket:
            if isinstance(frame, bytes):
                self.received_audio.extend(frame)
                self.received_frames.append(frame)
                continue
            message = orjson.loads(frame)
            self.received_json.append(message)
            self.received_frames.append(message)
            if message.get("type") == "CloseStream":
                break

        for event in self.events:
            await socket.send(orjson.dumps(event).decode())


TWO_TURN_EVENTS = [
    turn("StartOfTurn", 0),
    turn("Update", 0, "Hello from"),
    turn("EagerEndOfTurn", 0, "Hello from Flux", 0.73),
    turn("TurnResumed", 0, "Hello from Flux again", 0.41),
    turn("EndOfTurn", 0, "Hello from Flux."),
    turn("EndOfTurn", 0, "Hello from Flux restated."),
    turn("Update", 1, "Second"),
    turn("EndOfTurn", 1, "Second turn."),
]


async def run_adapter(
    events: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any] | None = None,
) -> tuple[Any, FakeDeepgramFluxWs]:
    """Drive ``DeepgramFlux`` against a local protocol server."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    handler = FakeDeepgramFluxWs(events)
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(deepgram, "FLUX_STREAM_URL", f"ws://127.0.0.1:{port}/v2/listen")
        adapter = deepgram.DeepgramFlux(options=options)
        result = await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)
    return result, handler


class TestHandshake:
    """Flux config is encoded in the v2 URL and auth header."""

    async def test_default_query_omits_optional_parameters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await run_adapter(TWO_TURN_EVENTS, monkeypatch)
        parsed = urlsplit(handler.path)
        assert parsed.path == "/v2/listen"
        assert parse_qs(parsed.query) == {
            "model": ["flux-general-en"],
            "encoding": ["linear16"],
            "sample_rate": ["16000"],
        }

    async def test_options_and_repeatable_multi_language_hints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, handler = await run_adapter(
            TWO_TURN_EVENTS,
            monkeypatch,
            options={
                "model": "flux-general-multi",
                "language_hint": ["en", "es"],
                "eot_threshold": 0.8,
                "eager_eot_threshold": 0.6,
                "eot_timeout_ms": 1200,
            },
        )
        params = parse_qs(urlsplit(handler.path).query)
        assert params["language_hint"] == ["en", "es"]
        assert params["eot_threshold"] == ["0.8"]
        assert params["eager_eot_threshold"] == ["0.6"]
        assert params["eot_timeout_ms"] == ["1200"]
        assert result.raw["endpoint_config"] == {
            "eot_threshold": 0.8,
            "eager_eot_threshold": 0.6,
            "eot_timeout_ms": 1200,
        }

    async def test_language_hints_are_omitted_for_english_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await run_adapter(TWO_TURN_EVENTS, monkeypatch, options={"language_hint": ["en", "de"]})
        assert "language_hint" not in parse_qs(urlsplit(handler.path).query)

    async def test_token_auth_and_binary_audio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await run_adapter(TWO_TURN_EVENTS, monkeypatch)
        assert handler.authorization == "Token test-key"
        assert bytes(handler.received_audio) == make_clip().pcm
        assert all(isinstance(frame, bytes) for frame in handler.received_frames[:-1])


class TestEventMapping:
    """Only definitive turn ends contribute final transcript and EOU events."""

    async def test_updates_are_interim_and_turn_ends_are_final_eou(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _ = await run_adapter(TWO_TURN_EVENTS, monkeypatch)
        assert result.text == "Hello from Flux. Second turn."
        assert [(partial.text, partial.is_final, partial.kind) for partial in result.partials] == [
            ("Hello from", False, EventKind.INTERIM),
            ("Hello from Flux.", True, EventKind.EOU),
            ("Second", False, EventKind.INTERIM),
            ("Second turn.", True, EventKind.EOU),
        ]

    async def test_repeated_end_of_turn_is_skipped_entirely(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _ = await run_adapter(TWO_TURN_EVENTS, monkeypatch)
        assert len([partial for partial in result.partials if partial.kind == EventKind.EOU]) == 2
        assert "restated" not in result.text

    async def test_eager_events_are_metadata_not_partials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _ = await run_adapter(TWO_TURN_EVENTS, monkeypatch)
        assert [event["event"] for event in result.raw["eager_events"]] == [
            "EagerEndOfTurn",
            "TurnResumed",
        ]
        assert [event["turn_index"] for event in result.raw["eager_events"]] == [0, 0]
        assert [event["end_of_turn_confidence"] for event in result.raw["eager_events"]] == [0.73, 0.41]
        assert all(isinstance(event["t_s"], float) for event in result.raw["eager_events"])
        assert all("again" not in partial.text for partial in result.partials)

    async def test_final_less_close_falls_back_to_last_hypothesis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [
            turn("StartOfTurn", 0, "This"),
            turn("Update", 0, "This is Samantha"),
            turn("Update", 0, "This is Samantha Lee"),
        ]
        result, _ = await run_adapter(events, monkeypatch)
        assert result.text == "This is Samantha Lee"
        assert result.raw["turn_completed"] is False
        assert all(partial.kind != EventKind.EOU for partial in result.partials)

    async def test_result_timing_and_endpoint_metadata_are_populated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _ = await run_adapter(TWO_TURN_EVENTS, monkeypatch)
        assert result.ttft_s is not None
        assert result.finalize_s is not None
        assert result.total_s >= 0.0
        assert "ws_rtt_s" in result.raw
        assert result.raw["eou_source"] == "end_of_turn"
        assert result.raw["endpoint_config"] == {}


class TestCloseHandshake:
    """The v2 socket receives one text ``CloseStream`` after all binary audio."""

    async def test_close_stream_sent_once_and_last(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await run_adapter(TWO_TURN_EVENTS, monkeypatch)
        assert handler.received_json == [{"type": "CloseStream"}]
        assert handler.received_frames[-1] == {"type": "CloseStream"}
        assert all(isinstance(frame, bytes) for frame in handler.received_frames[:-1])


class TestErrorFrames:
    """Vendor error frames raise with their full payload intact."""

    async def test_error_frame_carries_full_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {"type": "Error", "code": "INVALID_QUERY", "description": "bad eot threshold"}
        with pytest.raises(StreamProtocolError, match=re.escape("bad eot threshold")) as raised:
            await run_adapter([payload], monkeypatch)
        assert "INVALID_QUERY" in str(raised.value)
        assert "'type': 'Error'" in str(raised.value)


class TestValidation:
    """Invalid Flux options fail before authentication or connection setup."""

    async def test_eager_threshold_cannot_exceed_eot_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        adapter = deepgram.DeepgramFlux(options={"eot_threshold": 0.6, "eager_eot_threshold": 0.7})
        with pytest.raises(ValueError, match=r"eager_eot_threshold.*must be <= eot_threshold"):
            await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"eot_threshold": 0.49}, "eot_threshold"),
            ({"eager_eot_threshold": 0.91}, "eager_eot_threshold"),
            ({"eot_timeout_ms": 499}, "eot_timeout_ms"),
            ({"eot_timeout_ms": 500.5}, "must be an integer"),
            ({"language_hint": "en"}, "language_hint"),
            ({"model": "nova-3"}, "model must be"),
        ],
    )
    async def test_invalid_options_raise_before_auth(
        self, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], match: str
    ) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        adapter = deepgram.DeepgramFlux(options=options)
        with pytest.raises(ValueError, match=match):
            await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)


@pytest.mark.skipif(
    not os.environ.get("DEEPGRAM_API_KEY") or not os.environ.get(LIVE_FLAG),
    reason=f"live smoke needs DEEPGRAM_API_KEY and {LIVE_FLAG}=1",
)
class TestLiveSmoke:
    """One real, short Flux socket session using the local English corpus."""

    async def test_stream_returns_end_of_turn(self) -> None:
        clip = make_live_clip()
        # The corpus row is cut off mid-sentence, so the model's end-of-turn
        # confidence correctly never crosses the default threshold (observed
        # live 2026-08-08: eou_count stayed 0 even over appended silence).
        # The documented eot_timeout_ms floor forces the decision after 500 ms
        # of the appended silence, making the EndOfTurn wire path assertable.
        adapter = deepgram.DeepgramFlux(options={"eot_timeout_ms": 500})
        result = await adapter.transcribe_stream(clip, chunk_ms=80, realtime=True)
        evidence = {
            "text": result.text,
            "ttft_s": result.ttft_s,
            "finalize_s": result.finalize_s,
            "total_s": result.total_s,
            "eou_count": len([partial for partial in result.partials if partial.kind == EventKind.EOU]),
        }
        LOGGER.info("deepgram_flux_live=%s", orjson.dumps(evidence).decode())
        assert result.error is None
        assert result.text
        assert evidence["eou_count"] >= 1
