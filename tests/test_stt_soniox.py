"""Protocol tests for the Soniox batch (async transcription) STT path.

The whole HTTP pipeline — upload, create, poll, transcript fetch, and the
mandatory cleanup DELETEs — is mocked via ``httpx.MockTransport``, matching
the Solaria-3 batch pattern in ``test_stt_gladia.py``. The realtime WebSocket
path is exercised by the shared stream-driver tests and is out of scope here.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import orjson
import pytest
from websockets.asyncio.server import ServerConnection, serve

from audio_harness.stt import soniox
from audio_harness.stt.base import ProviderHttpError
from audio_harness.types import AudioClip


def make_clip(seconds: float = 0.1, rate: int = 16000, language: str = "en-US") -> AudioClip:
    """Build a silent clip with stable protocol-test metadata."""
    return AudioClip(
        clip_id="c1",
        pcm=b"\x00\x00" * int(rate * seconds),
        sample_rate=rate,
        duration_s=seconds,
        reference="async token stream",
        language=language,
        source_path="<memory>",
    )


class SonioxBatchTransport:
    """Mock Soniox's upload, create, poll, transcript, and delete surfaces."""

    def __init__(
        self,
        *,
        terminal: dict[str, Any] | None = None,
        failure_path: str | None = None,
        delete_failure: bool = False,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.poll_count = 0
        self.terminal = terminal or {"id": "t1", "status": "completed"}
        self.failure_path = failure_path
        self.delete_failure = delete_failure

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "DELETE":
            if self.delete_failure:
                return httpx.Response(409, text="cannot delete while processing")
            return httpx.Response(204)
        if path == self.failure_path:
            return httpx.Response(503, text="vendor temporarily unavailable")
        if path == "/v1/files" and request.method == "POST":
            return httpx.Response(201, json={"id": "f1", "filename": "audio.wav", "size": 3244})
        if path == "/v1/transcriptions" and request.method == "POST":
            return httpx.Response(201, json={"id": "t1", "status": "queued"})
        if path == "/v1/transcriptions/t1" and request.method == "GET":
            self.poll_count += 1
            if self.poll_count == 1:
                return httpx.Response(200, json={"id": "t1", "status": "processing"})
            return httpx.Response(200, json=self.terminal)
        if path == "/v1/transcriptions/t1/transcript" and request.method == "GET":
            return httpx.Response(200, json={"id": "t1", "text": "hello from soniox async", "tokens": []})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    def deletes(self) -> list[str]:
        """Paths of every DELETE issued, in order."""
        return [request.url.path for request in self.requests if request.method == "DELETE"]


def make_adapter(
    monkeypatch: pytest.MonkeyPatch,
    transport: SonioxBatchTransport,
    options: dict[str, Any] | None = None,
) -> soniox.SonioxRealtimeV5:
    """Build a Soniox adapter with a zero-delay mocked polling loop."""
    monkeypatch.setenv("SONIOX_API_KEY", "test-key")
    monkeypatch.setattr(soniox, "POLL_INTERVAL_S", 0.0)
    adapter = soniox.SonioxRealtimeV5(options)
    adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return adapter


class TestBatchHappyPath:
    """Upload, create, polling, transcript extraction, and cleanup."""

    async def test_full_flow_request_shapes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = SonioxBatchTransport()
        adapter = make_adapter(monkeypatch, transport)
        try:
            result = await adapter.transcribe_batch(make_clip())
        finally:
            await adapter.aclose()

        assert result.text == "hello from soniox async"
        assert result.error is None
        assert result.total_s > 0
        assert result.raw["model"] == "stt-async-v5", "batch runs the async lineage and must say so"
        assert result.raw["transcription_id"] == "t1"
        assert result.raw["deleted"] == {"transcription": True, "file": True}
        assert transport.poll_count == 2

        upload, created, first_poll, second_poll, transcript, *deletes = transport.requests
        assert upload.method == "POST"
        assert upload.url == httpx.URL("https://api.soniox.com/v1/files")
        assert upload.headers["Authorization"] == "Bearer test-key"
        assert b'name="file"' in upload.content
        assert b'filename="audio.wav"' in upload.content
        assert b"RIFF" in upload.content

        assert created.method == "POST"
        assert created.url == httpx.URL("https://api.soniox.com/v1/transcriptions")
        assert created.headers["Authorization"] == "Bearer test-key"
        assert created.headers["Content-Type"] == "application/json"
        assert orjson.loads(created.content) == {
            "model": "stt-async-v5",
            "file_id": "f1",
            "language_hints": ["en"],
        }

        assert first_poll.method == "GET"
        assert first_poll.url == httpx.URL("https://api.soniox.com/v1/transcriptions/t1")
        assert second_poll.url == first_poll.url
        assert transcript.method == "GET"
        assert transcript.url == httpx.URL("https://api.soniox.com/v1/transcriptions/t1/transcript")

        assert [request.url.path for request in deletes] == ["/v1/transcriptions/t1", "/v1/files/f1"]
        assert all(request.method == "DELETE" for request in deletes)
        assert all(request.headers["Authorization"] == "Bearer test-key" for request in deletes)

    async def test_language_hints_and_batch_model_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = SonioxBatchTransport()
        adapter = make_adapter(
            monkeypatch,
            transport,
            options={"language_hints": ["ja", "en"], "batch_model": "stt-async-v4"},
        )
        try:
            result = await adapter.transcribe_batch(make_clip(language="ja-JP"))
        finally:
            await adapter.aclose()

        body = orjson.loads(transport.requests[1].content)
        assert body["model"] == "stt-async-v4"
        assert body["language_hints"] == ["ja", "en"]
        assert result.raw["model"] == "stt-async-v4", "the recorded model must track the batch_model option"

    async def test_clip_language_seeds_default_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = SonioxBatchTransport()
        adapter = make_adapter(monkeypatch, transport)
        try:
            await adapter.transcribe_batch(make_clip(language="hu-HU"))
        finally:
            await adapter.aclose()

        assert orjson.loads(transport.requests[1].content)["language_hints"] == ["hu"]


class FakeSonioxWs:
    """Accept config and PCM frames, then replay canned token events."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.config: dict[str, Any] = {}

    async def __call__(self, socket: ServerConnection) -> None:
        async for frame in socket:
            if isinstance(frame, str) and frame:
                self.config = orjson.loads(frame)
            if frame == "":
                break
        for event in self.events:
            await socket.send(orjson.dumps(event).decode())


async def run_stream_adapter(
    events: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any] | None = None,
) -> tuple[Any, FakeSonioxWs]:
    """Drive the realtime adapter against the local token-protocol server."""
    monkeypatch.setenv("SONIOX_API_KEY", "test-key")
    handler = FakeSonioxWs(events)
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(soniox, "STREAM_URL", f"ws://127.0.0.1:{port}")
        adapter = soniox.SonioxRealtimeV5(options)
        result = await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)
    return result, handler


class TestStreamModelLabel:
    """The streaming path records the realtime model it actually configured."""

    async def test_default_model_is_recorded_and_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [{"tokens": [{"text": "hi", "is_final": True}], "finished": True}]
        result, handler = await run_stream_adapter(events, monkeypatch)
        assert result.text == "hi"
        assert result.raw["model"] == "stt-rt-v5"
        assert handler.config["model"] == "stt-rt-v5", "the recorded model must match the wire"

    async def test_model_option_is_recorded_and_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = [{"tokens": [], "finished": True}]
        result, handler = await run_stream_adapter(events, monkeypatch, options={"model": "stt-rt-v4"})
        assert result.raw["model"] == "stt-rt-v4"
        assert handler.config["model"] == "stt-rt-v4", "the recorded model must match the wire"


class TestBatchTerminalError:
    """A terminal vendor error becomes a failed result, never an exception."""

    async def test_error_status_sets_result_error_and_still_cleans_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = SonioxBatchTransport(
            terminal={
                "id": "t1",
                "status": "error",
                "error_type": "audio_error",
                "error_message": "could not decode audio",
            }
        )
        adapter = make_adapter(monkeypatch, transport)
        try:
            result = await adapter.transcribe_batch(make_clip())
        finally:
            await adapter.aclose()

        assert result.text == ""
        assert result.error == "could not decode audio"
        assert result.raw["response"] == transport.terminal
        assert "/v1/transcriptions/t1/transcript" not in [request.url.path for request in transport.requests]
        assert result.raw["deleted"] == {"transcription": True, "file": True}


class TestBatchHttpFailures:
    """Failed phases raise with the vendor body; created assets still get deleted."""

    @pytest.mark.parametrize(
        ("failure_path", "expected_deletes"),
        [
            ("/v1/files", []),
            ("/v1/transcriptions", ["/v1/files/f1"]),
            ("/v1/transcriptions/t1", ["/v1/transcriptions/t1", "/v1/files/f1"]),
            ("/v1/transcriptions/t1/transcript", ["/v1/transcriptions/t1", "/v1/files/f1"]),
        ],
    )
    async def test_http_error_raises_and_cleanup_matches_created_assets(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failure_path: str,
        expected_deletes: list[str],
    ) -> None:
        transport = SonioxBatchTransport(failure_path=failure_path)
        adapter = make_adapter(monkeypatch, transport)
        try:
            with pytest.raises(ProviderHttpError, match="vendor temporarily unavailable"):
                await adapter.transcribe_batch(make_clip())
        finally:
            await adapter.aclose()

        assert transport.deletes() == expected_deletes


class TestBatchCleanupFailure:
    """A failed DELETE warns and is recorded, but never raises."""

    async def test_delete_failure_is_recorded_not_raised(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        transport = SonioxBatchTransport(delete_failure=True)
        adapter = make_adapter(monkeypatch, transport)
        try:
            with caplog.at_level(logging.WARNING, logger="audio_harness.stt.soniox"):
                result = await adapter.transcribe_batch(make_clip())
        finally:
            await adapter.aclose()

        assert result.text == "hello from soniox async"
        assert result.error is None
        assert result.raw["deleted"] == {"transcription": False, "file": False}
        assert transport.deletes() == ["/v1/transcriptions/t1", "/v1/files/f1"]
        warnings = [record.message for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 2
        assert all("failed to delete" in message for message in warnings)
