"""Batch-only Parakeet TDT v3 adapter through a native Swift sidecar.

FluidAudio runs the public CoreML model on Apple Silicon without involving the
Python package ecosystem. That boundary matters here because the harness runs
on Python 3.14 while coremltools has no compatible wheel. A fresh sidecar
process per clip keeps the adapter a small, auditable subprocess client and
keeps model loading visible in each result.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, cast

from audio_harness.audio import wrap_wav
from audio_harness.types import AudioClip, Mode, SttResult

from .base import SttProvider, register


DEFAULT_TIMEOUT_S = 1800.0
PARAKEET_ANE_BINARY_ENV = "PARAKEET_ANE_BINARY"
BUILD_HINT = "cd sidecars/parakeet-ane && swift build -c release"


@register
class ParakeetAneStt(SttProvider):
    """On-device Parakeet TDT v3 through the FluidAudio Swift sidecar.

    Options:
        binary: Path to a prebuilt ``parakeet-ane`` executable.
        model_dir: FluidAudio model-cache directory override.
        timeout_s: Maximum sidecar runtime; defaults to 1800 seconds so the
            first call has time to download the model.
    """

    key = "parakeet-ane"
    vendor = ""
    family = "nvidia"
    supports_batch = True

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Transcribe a complete WAV through one FluidAudio sidecar process."""
        result = self._result(clip, Mode.BATCH)
        binary = self._binary_path()
        if not binary.is_file():
            raise RuntimeError(f"parakeet-ane sidecar not found at {binary}. Build it with: {BUILD_HINT}")

        timeout_s = float(self.options.get("timeout_s", DEFAULT_TIMEOUT_S))
        if timeout_s <= 0:
            raise ValueError("parakeet-ane: timeout_s must be greater than zero")

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(wrap_wav(clip.pcm, clip.sample_rate))
            audio_file.flush()

            arguments = [str(binary), "transcribe", "--audio", audio_file.name, "--json"]
            if model_dir := self.options.get("model_dir"):
                arguments.extend(("--model-dir", str(model_dir)))

            started = time.perf_counter()
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
            except TimeoutError as exc:
                if process.returncode is None:
                    process.kill()
                await process.communicate()
                raise RuntimeError(f"parakeet-ane sidecar timed out after {timeout_s:g}s: {binary}") from exc
            result.total_s = time.perf_counter() - started

        if process.returncode != 0:
            raise RuntimeError(_sidecar_failure(binary, process.returncode, stdout, stderr))

        payload = _parse_success(binary, stdout, stderr)
        transcript = payload.get("transcript")
        if not isinstance(transcript, str):
            raise TypeError(f"parakeet-ane sidecar returned no string transcript: {binary}")

        result.text = transcript
        result.raw.update({
            "local_compute": True,
            "on_device": True,
            "rtf": payload.get("rtf"),
            "load_s": payload.get("load_s"),
            "infer_s": payload.get("infer_s"),
            "model_id": payload.get("model_id"),
            "compute_units": payload.get("compute_units"),
            "sidecar": "swift-fluidaudio",
        })
        return result

    def _binary_path(self) -> Path:
        return resolve_binary(self.options.get("binary"))


def resolve_binary(configured: object = None) -> Path:
    """Resolve the sidecar executable: option, then env var, then repo default.

    Shared with the doctor check so both agree on which binary the lane runs.
    """
    if configured:
        return Path(str(configured)).expanduser().resolve()
    if environment_path := os.environ.get(PARAKEET_ANE_BINARY_ENV):
        return Path(environment_path).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "sidecars" / "parakeet-ane" / ".build" / "release" / "parakeet-ane"


def _parse_success(binary: Path, stdout: bytes, stderr: bytes) -> dict[str, Any]:
    try:
        payload: Any = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        detail = _stderr_tail(stderr) or stdout.decode(errors="replace")[-2000:] or "<empty output>"
        raise RuntimeError(f"parakeet-ane sidecar returned invalid JSON from {binary}: {detail}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"parakeet-ane sidecar returned a non-object JSON value from {binary}")
    return cast("dict[str, Any]", payload)


def _sidecar_failure(binary: Path, returncode: int | None, stdout: bytes, stderr: bytes) -> str:
    error_message: str | None = None
    try:
        payload: Any = json.loads(stdout)
    except UnicodeDecodeError, json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        error_message = payload["error"]
    detail = error_message or _stderr_tail(stderr) or stdout.decode(errors="replace")[-2000:] or "<no diagnostics>"
    return f"parakeet-ane sidecar failed with exit code {returncode} ({binary}): {detail}"


def _stderr_tail(stderr: bytes) -> str:
    return stderr.decode(errors="replace").strip()[-2000:]
