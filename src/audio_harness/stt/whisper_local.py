"""Local Whisper large-v3 adapter, the offline round-trip judge.

The cross-family rule needs at least one recognizer outside every TTS
candidate's vendor family, and an API judge cannot fill that role for its own
vendor's voices. A local Whisper can: it bills nobody, is OpenAI-lineage
(``family = "openai"``), and — with the model revision pinned and decoding
greedy — produces the same transcript for the same audio on every machine,
which is what makes round-trip scores comparable across runs.

Runs on Apple Silicon via mlx-whisper (the ``judge-whisper`` optional
dependency group). Batch-only by design: judging saved audio is offline work
with no latency story to measure.
"""

from __future__ import annotations

import asyncio
from functools import cache
import time
from typing import Any

import numpy as np
import soxr

from audio_harness.types import AudioClip, Mode, SttResult

from .base import SttProvider, register


DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_REVISION = "49e6aa286ad60c14352c404340ded53710378a11"
"""Model repo head as of 2026-08-06; pinned so scores are reproducible."""

WHISPER_SAMPLE_RATE = 16000
"""Whisper's fixed input rate; other rates are resampled before inference."""


@register
class WhisperLocal(SttProvider):
    """Pinned Whisper large-v3 running locally through mlx-whisper.

    Requires the optional dependency group::

        uv sync --extra judge-whisper

    The first transcription downloads the pinned model snapshot (~3 GB) into
    the Hugging Face cache; later runs are fully offline.

    Options:
        model: Hugging Face repo holding converted mlx weights.
        revision: Git revision of that repo. Changing either changes every
            round-trip score, so hold them constant across compared runs.
    """

    key = "whisper-local"
    family = "openai"
    supports_batch = True

    def _model(self) -> str:
        return str(self.options.get("model", DEFAULT_MODEL))

    def _revision(self) -> str:
        return str(self.options.get("revision", DEFAULT_REVISION))

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Transcribe a clip with the pinned local model.

        Inference is synchronous and Metal-bound, so it runs in a worker
        thread to keep the event loop serving any concurrent lanes.
        """
        result = self._result(clip, Mode.BATCH)
        started = time.perf_counter()
        output = await asyncio.to_thread(self._transcribe, clip)
        result.total_s = time.perf_counter() - started
        result.text = str(output.get("text", "")).strip()
        result.raw["model"] = self._model()
        result.raw["revision"] = self._revision()
        return result

    def _transcribe(self, clip: AudioClip) -> dict[str, Any]:
        """Run greedy decoding over the clip's PCM at Whisper's input rate."""
        mlx_whisper = _import_mlx_whisper()
        samples = np.frombuffer(clip.pcm, dtype="<i2").astype(np.float32) / 32768.0
        if clip.sample_rate != WHISPER_SAMPLE_RATE:
            samples = soxr.resample(samples, clip.sample_rate, WHISPER_SAMPLE_RATE, quality="HQ")
        language = clip.language.split("-", 1)[0].lower()
        transcription: dict[str, Any] = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=_model_path(self._model(), self._revision()),
            language=language or None,
            # A bare 0.0 disables Whisper's temperature-fallback schedule,
            # keeping decoding greedy and deterministic — the judge must never
            # sample.
            temperature=0.0,
            condition_on_previous_text=False,
        )
        return transcription


def _import_mlx_whisper() -> Any:
    """Import mlx-whisper lazily so the registry works without the extra.

    Every adapter must be importable at package-import time — that is how the
    registry fills — so the optional dependency is only touched when a clip is
    actually transcribed.
    """
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "whisper-local: mlx-whisper is not installed. Install the "
            "optional dependency group: uv sync --extra judge-whisper"
        ) from exc
    return mlx_whisper


@cache
def _model_path(repo: str, revision: str) -> str:
    """Materialize the pinned model snapshot and return its local path.

    mlx-whisper resolves a bare repo name to its head, which would let scores
    drift when the repo updates. Downloading the snapshot at an explicit
    revision is what enforces the pin.
    """
    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo_id=repo, revision=revision))
