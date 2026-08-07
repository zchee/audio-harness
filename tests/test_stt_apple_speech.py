"""Tests for the on-device Apple Speech adapter."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

import polars as pl
import pytest

from audio_harness import stt
from audio_harness.audio import load_clip_bytes
from audio_harness.stt import apple_speech
from audio_harness.types import EventKind


LIVE_AUDIO = Path(__file__).parents[1] / "data/hf/stt-benchmark-data/data/train-00000-of-00001.parquet"


class TestCaptureConversion:
    def test_callback_captures_become_relative_partials_without_eou(self) -> None:
        partials = apple_speech._partials_from_captures(
            [
                (10.125, "hello", False),
                (10.375, "hello world", False),
                (10.625, "hello world", True),
                (10.750, "", False),
            ],
            started_at=10.0,
        )

        assert [partial.t_s for partial in partials] == pytest.approx([0.125, 0.375, 0.625])
        assert [partial.text for partial in partials] == ["hello", "hello world", "hello world"]
        assert [partial.is_final for partial in partials] == [False, False, True]
        assert [partial.kind for partial in partials] == [
            EventKind.INTERIM,
            EventKind.INTERIM,
            EventKind.SEGMENT_FINAL,
        ]
        assert all(partial.kind != EventKind.EOU for partial in partials)

    @pytest.mark.parametrize(
        ("duration_s", "expected_rtf"),
        [
            (4.0, 0.125),
            (0.0, None),
        ],
    )
    def test_local_compute_metadata(self, duration_s: float, expected_rtf: float | None) -> None:
        raw = apple_speech._local_compute_raw(total_s=0.5, duration_s=duration_s)

        assert raw["local_compute"] is True
        assert raw["on_device"] is True
        if expected_rtf is None:
            assert raw["rtf"] is None
        else:
            assert raw["rtf"] == pytest.approx(expected_rtf)


def test_registration_and_capabilities() -> None:
    adapter = stt.create("apple-speech-stt")

    assert isinstance(adapter, apple_speech.AppleSpeechStt)
    assert stt.family_of("apple-speech-stt") == "apple"
    assert adapter.supports_stream is True
    assert adapter.supports_batch is False


LIVE_FLAG = "AUDIO_HARNESS_TEST_APPLE_SPEECH_LIVE"


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run real on-device Apple Speech recognition",
)
class TestLiveRecognition:
    async def test_file_recognition_reports_interims_and_local_rtf(self) -> None:
        if sys.platform != "darwin":
            pytest.skip("apple-speech-stt live recognition requires macOS")
        try:
            __import__("Speech")
        except ImportError:
            pytest.skip("pyobjc-framework-Speech is not installed")
        if not LIVE_AUDIO.is_file():
            pytest.skip(f"live recognition corpus not found: {LIVE_AUDIO}")

        # A real corpus utterance, matching every other live smoke: `say` is
        # blocked in several of this project's sandboxed shells (it emits a
        # zero-sample file), so synthesized speech is not a reliable source.
        row = pl.read_parquet(LIVE_AUDIO, n_rows=1).select("sample_id", "transcription", "audio").to_dicts()[0]
        clip = load_clip_bytes(
            row["audio"]["bytes"],
            clip_id="apple-speech-live",
            reference=str(row["transcription"]),
            language="en-US",
            source_path=f"{LIVE_AUDIO}#{row['sample_id']}",
        )

        result = await apple_speech.AppleSpeechStt().transcribe_stream(
            clip,
            chunk_ms=20,
            realtime=False,
        )

        logging.getLogger(__name__).info(
            "apple-speech live: text=%r partials=%d rtf=%s",
            result.text,
            len(result.partials),
            result.raw["rtf"],
        )

        assert result.error is None
        assert result.text.strip()
        assert len(result.partials) > 1
        assert result.raw["local_compute"] is True
        assert result.raw["on_device"] is True
        assert result.raw["rtf"] == pytest.approx(result.rtf)
