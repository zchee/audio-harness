"""Protocol tests for the AssemblyAI batch delete-after retention option.

The batch surface (upload, create, poll, delete) is mocked via
``httpx.MockTransport``, matching the Solaria-3 batch pattern in
``test_stt_gladia.py``. The streaming path stores nothing server-side, so
``delete_after`` has no streaming equivalent to test.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from audio_harness import stt
from audio_harness.stt import assemblyai
from audio_harness.types import AudioClip


def make_clip(seconds: float = 0.1, rate: int = 16000, language: str = "en-US") -> AudioClip:
    """Build a silent clip of a known duration."""
    return AudioClip(
        clip_id="c1",
        pcm=b"\x00\x00" * int(rate * seconds),
        sample_rate=rate,
        duration_s=seconds,
        reference="hello world",
        language=language,
        source_path="<memory>",
    )


class BatchTransport:
    """Mock AssemblyAI's upload, create, polling, and delete HTTP surfaces."""

    def __init__(self, *, delete_status: int = 200, delete_error: Exception | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.poll_count = 0
        self.delete_status = delete_status
        self.delete_error = delete_error

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v2/upload":
            return httpx.Response(200, json={"upload_url": "https://cdn.assemblyai.com/upload/abc"})
        if request.url.path == "/v2/transcript" and request.method == "POST":
            return httpx.Response(200, json={"id": "t-1"})
        if request.url.path == "/v2/transcript/t-1" and request.method == "GET":
            self.poll_count += 1
            if self.poll_count == 1:
                return httpx.Response(200, json={"status": "processing"})
            return httpx.Response(200, json={"status": "completed", "text": "hello world"})
        if request.url.path == "/v2/transcript/t-1" and request.method == "DELETE":
            if self.delete_error is not None:
                raise self.delete_error
            return httpx.Response(self.delete_status, json={})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    def deletes(self) -> list[httpx.Request]:
        """Return every DELETE request the adapter issued."""
        return [request for request in self.requests if request.method == "DELETE"]


def make_adapter(
    monkeypatch: pytest.MonkeyPatch,
    transport: BatchTransport,
    options: dict[str, Any] | None = None,
) -> assemblyai.AssemblyAIUniversal35Pro:
    """Build an adapter with a zero-delay mocked polling loop."""
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
    monkeypatch.setattr(assemblyai, "POLL_INTERVAL_S", 0.0)
    adapter = stt.create("assemblyai-universal35pro", options)
    assert isinstance(adapter, assemblyai.AssemblyAIUniversal35Pro)
    adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return adapter


class TestDeleteAfter:
    """The ``delete_after`` retention option on the batch path."""

    async def test_delete_issued_with_url_and_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = BatchTransport()
        adapter = make_adapter(monkeypatch, transport, {"delete_after": True})
        try:
            result = await adapter.transcribe_batch(make_clip())
        finally:
            await adapter.aclose()

        assert result.text == "hello world"
        assert result.error is None
        assert result.raw["transcript_id"] == "t-1"
        assert result.raw["deleted"] is True

        [delete] = transport.deletes()
        assert delete.url == httpx.URL("https://api.assemblyai.com/v2/transcript/t-1")
        assert delete.headers["Authorization"] == "test-key"
        assert transport.requests[-1] is delete, "delete must come after the transcript has been fetched"

    async def test_delete_failure_records_false_without_raising(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        transport = BatchTransport(delete_status=500)
        adapter = make_adapter(monkeypatch, transport, {"delete_after": True})
        try:
            with caplog.at_level(logging.WARNING, logger="audio_harness.stt.assemblyai"):
                result = await adapter.transcribe_batch(make_clip())
        finally:
            await adapter.aclose()

        assert result.text == "hello world"
        assert result.error is None, "a failed delete must not fail the transcription"
        assert result.raw["deleted"] is False
        assert any("failed to delete transcript t-1" in record.message for record in caplog.records)

    async def test_delete_transport_error_records_false_without_raising(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        transport = BatchTransport(delete_error=httpx.ConnectError("connection refused"))
        adapter = make_adapter(monkeypatch, transport, {"delete_after": True})
        try:
            with caplog.at_level(logging.WARNING, logger="audio_harness.stt.assemblyai"):
                result = await adapter.transcribe_batch(make_clip())
        finally:
            await adapter.aclose()

        assert result.error is None
        assert result.raw["deleted"] is False
        assert any("failed to delete transcript t-1" in record.message for record in caplog.records)

    async def test_default_issues_no_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = BatchTransport()
        adapter = make_adapter(monkeypatch, transport)
        try:
            result = await adapter.transcribe_batch(make_clip())
        finally:
            await adapter.aclose()

        assert result.text == "hello world"
        assert result.raw["transcript_id"] == "t-1"
        assert "deleted" not in result.raw
        assert transport.deletes() == []
