"""End-to-end tests for the WebSocket streaming driver.

These run against a real local WebSocket server rather than a mock, because
the properties under test — when bytes actually arrive, how long finalization
takes, whether send and receive truly overlap — only exist in a real socket.
A mock would happily confirm whatever timing the driver claimed.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import orjson
import pytest
import websockets
from websockets.asyncio.server import ServerConnection, serve

from audio_harness.stt.base import StreamTimeline
from audio_harness.stt.ws import StreamProtocolError, run_stream
from audio_harness.types import AudioClip


def make_clip(seconds: float = 0.4, rate: int = 16000) -> AudioClip:
    """Build a silent clip of a known duration."""
    return AudioClip(
        clip_id="c1",
        pcm=b"\x00\x00" * int(rate * seconds),
        sample_rate=rate,
        duration_s=seconds,
        reference="hello world",
        language="en-US",
        source_path="<memory>",
    )


def handle_message(payload: object, timeline: StreamTimeline) -> bool:
    """Interpret the toy protocol the test server speaks."""
    if not isinstance(payload, dict):
        return False
    if payload.get("type") == "error":
        raise StreamProtocolError(str(payload.get("detail")))
    if payload.get("type") == "done":
        return True
    if payload.get("type") == "transcript":
        timeline.record(str(payload["text"]), is_final=bool(payload.get("final")))
    return False


class Server:
    """A configurable WebSocket transcription server for tests."""

    def __init__(self) -> None:
        self.received_audio = bytearray()
        self.received_json: list[dict] = []
        self.partial_after_s = 0.05
        self.finalize_delay_s = 0.1
        self.send_done = True
        self.fail_with: str | None = None
        self.never_finalize = False

    async def __call__(self, socket: ServerConnection) -> None:
        partial_task = asyncio.create_task(self._emit_partials(socket))
        try:
            async for frame in socket:
                if isinstance(frame, bytes):
                    self.received_audio.extend(frame)
                    continue
                with contextlib.suppress(orjson.JSONDecodeError):
                    message = orjson.loads(frame)
                    self.received_json.append(message)
                    if message.get("type") == "eos":
                        break
        except websockets.ConnectionClosed:
            return
        finally:
            partial_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await partial_task

        if self.fail_with is not None:
            await socket.send(
                orjson.dumps({"type": "error", "detail": self.fail_with}).decode()
            )
            return

        if self.never_finalize:
            return

        await asyncio.sleep(self.finalize_delay_s)
        with contextlib.suppress(websockets.ConnectionClosed):
            await socket.send(
                orjson.dumps(
                    {"type": "transcript", "text": "hello world", "final": True}
                ).decode()
            )
            if self.send_done:
                await socket.send(orjson.dumps({"type": "done"}).decode())

    async def _emit_partials(self, socket: ServerConnection) -> None:
        """Emit growing interim hypotheses while audio is still arriving."""
        await asyncio.sleep(self.partial_after_s)
        for text in ("hello", "hello wor", "hello world"):
            with contextlib.suppress(websockets.ConnectionClosed):
                await socket.send(
                    orjson.dumps(
                        {"type": "transcript", "text": text, "final": False}
                    ).decode()
                )
            await asyncio.sleep(0.05)


@pytest.fixture
async def server() -> AsyncIterator[tuple[Server, str]]:
    """Run the test server on an ephemeral port and yield its URL."""
    handler = Server()
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        yield handler, f"ws://127.0.0.1:{port}"


async def drive(
    url: str,
    clip: AudioClip,
    *,
    realtime: bool = True,
    finalize_timeout_s: float = 5.0,
) -> StreamTimeline:
    """Run the driver against the test server with the toy protocol."""
    timeline = StreamTimeline()

    async def eos(socket: object) -> None:
        await socket.send(orjson.dumps({"type": "eos"}).decode())  # type: ignore[attr-defined]

    await run_stream(
        url=url,
        headers={},
        clip=clip,
        chunk_ms=20,
        realtime=realtime,
        timeline=timeline,
        handle_message=handle_message,
        on_input_done=eos,
        finalize_timeout_s=finalize_timeout_s,
    )
    return timeline


class TestAudioDelivery:
    """The provider must receive exactly the bytes the clip contains."""

    async def test_all_audio_arrives_intact(self, server: tuple[Server, str]) -> None:
        handler, url = server
        clip = make_clip(0.2)
        await drive(url, clip)

        assert bytes(handler.received_audio) == clip.pcm

    async def test_end_of_stream_is_signalled(self, server: tuple[Server, str]) -> None:
        handler, url = server
        await drive(url, make_clip(0.1))

        assert {m.get("type") for m in handler.received_json} == {"eos"}


class TestTiming:
    """Timing semantics are the reason this harness exists."""

    async def test_ttft_measures_first_partial_not_first_final(
        self, server: tuple[Server, str]
    ) -> None:
        handler, url = server
        handler.partial_after_s = 0.08
        timeline = await drive(url, make_clip(0.4))

        assert timeline.ttft_s is not None
        assert timeline.ttft_s == pytest.approx(0.08, abs=0.06)
        assert timeline.ttft_s < 0.4, (
            "the first hypothesis arrives mid-stream, so send and receive "
            "must genuinely overlap rather than run in sequence"
        )

    async def test_finalize_measures_from_last_audio_byte(
        self, server: tuple[Server, str]
    ) -> None:
        handler, url = server
        handler.finalize_delay_s = 0.25
        timeline = await drive(url, make_clip(0.4))

        assert timeline.finalize_s is not None
        assert timeline.finalize_s == pytest.approx(0.25, abs=0.12), (
            "finalization latency must exclude the clip's own duration"
        )
        assert timeline.audio_end_s is not None
        assert timeline.audio_end_s == pytest.approx(0.4, abs=0.12)

    async def test_longer_clip_does_not_inflate_finalize(
        self, server: tuple[Server, str]
    ) -> None:
        handler, url = server
        handler.finalize_delay_s = 0.15

        short = await drive(url, make_clip(0.2))
        long = await drive(url, make_clip(0.8))

        assert short.finalize_s is not None
        assert long.finalize_s is not None
        assert abs(long.finalize_s - short.finalize_s) < 0.15, (
            "finalization latency must be independent of clip length"
        )

    async def test_timestamps_increase_monotonically(
        self, server: tuple[Server, str]
    ) -> None:
        _, url = server
        timeline = await drive(url, make_clip(0.4))
        stamps = [p.t_s for p in timeline.partials]

        assert stamps == sorted(stamps)
        assert stamps[0] >= 0.0


class TestTranscriptAssembly:
    """Interim and final hypotheses must be separable after the fact."""

    async def test_partials_and_finals_are_both_recorded(
        self, server: tuple[Server, str]
    ) -> None:
        _, url = server
        timeline = await drive(url, make_clip(0.4))

        assert [p for p in timeline.partials if not p.is_final], "expected interims"
        assert [p for p in timeline.partials if p.is_final], "expected a final"

    async def test_concat_finals_builds_the_transcript(
        self, server: tuple[Server, str]
    ) -> None:
        _, url = server
        timeline = await drive(url, make_clip(0.3))

        assert timeline.concat_finals() == "hello world"
        assert timeline.last_final() == "hello world"

    async def test_empty_hypotheses_are_ignored(self) -> None:
        timeline = StreamTimeline()
        timeline.start()
        timeline.record("", is_final=False)
        timeline.record("", is_final=True)

        assert timeline.partials == []
        assert timeline.ttft_s is None, (
            "a blank keepalive must not register as a first token"
        )


class TestFailureHandling:
    """A misbehaving vendor must produce a bounded, reportable outcome."""

    async def test_vendor_error_propagates(self, server: tuple[Server, str]) -> None:
        handler, url = server
        handler.fail_with = "quota exceeded"

        with pytest.raises(StreamProtocolError, match="quota exceeded"):
            await drive(url, make_clip(0.1))

    async def test_missing_final_stops_after_the_grace_period(
        self, server: tuple[Server, str]
    ) -> None:
        handler, url = server
        handler.never_finalize = True

        timeline = await drive(url, make_clip(0.2), finalize_timeout_s=0.4)

        assert timeline.finalize_s is None, "no final means no finalization latency"
        assert timeline.ttft_s is not None, "interims still count as measurements"

    async def test_missing_done_message_still_terminates(
        self, server: tuple[Server, str]
    ) -> None:
        handler, url = server
        handler.send_done = False
        handler.finalize_delay_s = 0.05

        timeline = await asyncio.wait_for(
            drive(url, make_clip(0.2), finalize_timeout_s=0.5), timeout=5.0
        )
        assert timeline.concat_finals() == "hello world"
