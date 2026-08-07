"""Tests for the on-device Parakeet FluidAudio sidecar adapter."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import stat
import textwrap

import polars as pl
import pytest

from audio_harness import stt
from audio_harness.audio import load_clip_bytes
from audio_harness.stt import parakeet_ane
from audio_harness.types import AudioClip


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_BINARY = REPO_ROOT / "sidecars/parakeet-ane/.build/release/parakeet-ane"
LIVE_AUDIO = REPO_ROOT / "data/hf/stt-benchmark-data/data/train-00000-of-00001.parquet"


def _clip() -> AudioClip:
    pcm = b"\x00\x00" * 320
    return AudioClip(
        clip_id="parakeet-test",
        pcm=pcm,
        sample_rate=16000,
        duration_s=0.02,
        reference="known reference",
        language="en-US",
        source_path="generated",
    )


def _write_sidecar(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


async def test_protocol_and_result_mapping(tmp_path: Path) -> None:
    argv_record = tmp_path / "argv.json"
    binary = _write_sidecar(
        tmp_path / "parakeet-ane",
        f"""
        import json
        from pathlib import Path
        import sys
        import wave

        arguments = sys.argv[1:]
        Path({str(argv_record)!r}).write_text(json.dumps(arguments))
        audio_path = Path(arguments[arguments.index("--audio") + 1])
        assert audio_path.is_file()
        with wave.open(str(audio_path), "rb") as audio:
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getframerate() == 16000
        print(json.dumps({{
            "transcript": "known transcript",
            "load_s": 1.25,
            "infer_s": 0.125,
            "audio_s": 0.02,
            "rtf": 6.25,
            "model_id": "FluidInference/parakeet-tdt-0.6b-v3-coreml",
            "compute_units": "cpuAndNeuralEngine",
        }}))
        """,
    )
    model_dir = tmp_path / "models"
    adapter = parakeet_ane.ParakeetAneStt({"binary": str(binary), "model_dir": str(model_dir), "timeout_s": 5.0})

    result = await adapter.transcribe_batch(_clip())

    arguments = json.loads(argv_record.read_text())
    assert arguments[:2] == ["transcribe", "--audio"]
    assert Path(arguments[2]).suffix == ".wav"
    assert arguments[3:] == ["--json", "--model-dir", str(model_dir)]
    assert result.text == "known transcript"
    assert result.raw["reference"] == "known reference"
    assert result.raw["language"] == "en-US"
    assert result.raw["rtf"] == 6.25
    assert result.raw["load_s"] == 1.25
    assert result.raw["infer_s"] == 0.125
    assert result.raw["model_id"] == "FluidInference/parakeet-tdt-0.6b-v3-coreml"
    assert result.raw["compute_units"] == "cpuAndNeuralEngine"
    assert result.raw["local_compute"] is True
    assert result.raw["on_device"] is True
    assert result.raw["sidecar"] == "swift-fluidaudio"
    assert result.total_s >= 0


async def test_environment_binary_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = _write_sidecar(
        tmp_path / "parakeet-ane",
        """
        import json

        print(json.dumps({
            "transcript": "from environment",
            "load_s": 0.1,
            "infer_s": 0.2,
            "audio_s": 0.02,
            "rtf": 10.0,
            "model_id": "model",
            "compute_units": "cpuAndNeuralEngine",
        }))
        """,
    )
    monkeypatch.setenv(parakeet_ane.PARAKEET_ANE_BINARY_ENV, str(binary))

    result = await parakeet_ane.ParakeetAneStt({"timeout_s": 5.0}).transcribe_batch(_clip())

    assert result.text == "from environment"


async def test_sidecar_json_error_is_surfaced(tmp_path: Path) -> None:
    binary = _write_sidecar(
        tmp_path / "parakeet-ane",
        """
        import json
        import sys

        print(json.dumps({"error": "model failed to load"}))
        sys.exit(7)
        """,
    )

    with pytest.raises(RuntimeError, match="model failed to load"):
        await parakeet_ane.ParakeetAneStt({"binary": str(binary)}).transcribe_batch(_clip())


async def test_missing_binary_includes_build_hint(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(RuntimeError, match="cd sidecars/parakeet-ane && swift build -c release"):
        await parakeet_ane.ParakeetAneStt({"binary": str(missing)}).transcribe_batch(_clip())


async def test_timeout_names_timeout_and_binary(tmp_path: Path) -> None:
    binary = _write_sidecar(
        tmp_path / "parakeet-ane",
        """
        import time

        time.sleep(5)
        """,
    )

    with pytest.raises(RuntimeError, match=rf"timed out after 0.01s: {binary}"):
        await parakeet_ane.ParakeetAneStt({"binary": str(binary), "timeout_s": 0.01}).transcribe_batch(_clip())


def test_registration_and_capabilities() -> None:
    adapter = stt.create("parakeet-ane")

    assert isinstance(adapter, parakeet_ane.ParakeetAneStt)
    assert stt.family_of("parakeet-ane") == "nvidia"
    assert adapter.supports_batch is True
    assert adapter.supports_stream is False


LIVE_FLAG = "AUDIO_HARNESS_TEST_PARAKEET_ANE_LIVE"


@pytest.mark.skipif(
    os.environ.get(LIVE_FLAG) != "1",
    reason=f"set {LIVE_FLAG}=1 to run real on-device Parakeet recognition",
)
class TestLiveRecognition:
    async def test_live_batch_recognition(self) -> None:
        if not DEFAULT_BINARY.is_file():
            pytest.skip(f"parakeet-ane sidecar not built: {DEFAULT_BINARY}")
        if not LIVE_AUDIO.is_file():
            pytest.skip(f"live recognition corpus not found: {LIVE_AUDIO}")

        row = pl.read_parquet(LIVE_AUDIO, n_rows=1).select("sample_id", "transcription", "audio").to_dicts()[0]
        clip = load_clip_bytes(
            row["audio"]["bytes"],
            clip_id="parakeet-ane-live",
            reference=str(row["transcription"]),
            language="en-US",
            source_path=f"{LIVE_AUDIO}#{row['sample_id']}",
        )

        result = await parakeet_ane.ParakeetAneStt({
            "binary": str(DEFAULT_BINARY),
            "timeout_s": 3600.0,
        }).transcribe_batch(clip)

        logging.getLogger(__name__).info(
            "parakeet-ane live: text=%r rtf=%s load_s=%s infer_s=%s",
            result.text[:200],
            result.raw["rtf"],
            result.raw["load_s"],
            result.raw["infer_s"],
        )
        rtf = result.raw["rtf"]
        assert result.error is None
        assert result.text.strip()
        assert isinstance(rtf, int | float)
        assert math.isfinite(rtf)
