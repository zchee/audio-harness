"""Deterministic synthetic clips for the hallucination/silence lane.

Streaming recognizers are least trustworthy exactly where WER cannot see it:
audio that contains no speech at all, or speech buried under noise. This module
fabricates those conditions on demand — pure silence, noise-only, real
utterances with extended trailing silence, and speech mixed below the noise
floor — so the lane never commits audio to the repository. Every generator is
seeded, so two runs of the same config produce byte-identical clips.

Noise comes from MUSAN (Snyder, Chen and Povey, arXiv:1510.08484), which is
licensed CC BY 4.0 (attribution required). Download it with
``tools/fetch_musan.py``; nothing here fetches from the network.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .audio import (
    BYTES_PER_SAMPLE,
    SPEECH_FRAME_MS,
    SPEECH_THRESHOLD,
    _float_to_pcm16,
    detect_speech_end_s,
)
from .config import SourceConfig
from .dataset import DatasetError, load_source
from .types import AudioClip


DEFAULT_SEED = 20260806
"""Seed used when a synthetic source pins none, so runs stay reproducible."""

DEFAULT_DURATION_S = 8.0
"""Length of generated silence and noise-only clips."""

DEFAULT_TRAILING_SILENCE_S = 6.0
"""Silence appended after a real utterance to bait post-speech fabrication."""

DEFAULT_SNR_DB = -10.0
"""Active-speech SNR for the low-SNR condition: speech below the noise."""

NOISE_LEVEL_DBFS = -26.0
"""RMS level noise-only clips are normalized to — loud enough that a vendor
cannot dismiss the clip as digital silence, quiet enough not to clip."""

CONDITIONS = ("silence", "noise", "trailing_silence", "low_snr")
"""Synthetic source kinds accepted in ``dataset.sources[].synthetic``."""

_ID_PREFIXES = {
    "silence-": "silence",
    "noise-": "noise",
    "trailsil-": "trailing_silence",
    "lowsnr-": "low_snr",
}

_MUSAN_HINT = (
    "download MUSAN (CC BY 4.0) with `uv run tools/fetch_musan.py` and point noise_dir at the extracted noise directory"
)


def condition_of(clip_id: str) -> str | None:
    """Recover the synthetic condition a clip id encodes, if any.

    Clip metadata does not survive the results JSONL, but clip ids do, so the
    condition rides in the id prefix and stays recoverable when saved runs are
    re-scored.

    Args:
        clip_id: Identifier as produced by this module.

    Returns:
        One of :data:`CONDITIONS`, or ``None`` for a non-synthetic clip.
    """
    for prefix, condition in _ID_PREFIXES.items():
        if clip_id.startswith(prefix):
            return condition
    return None


def synthesize_source(source: SourceConfig, *, sample_rate: int = 16000) -> list[AudioClip]:
    """Generate the clips a synthetic source describes.

    Args:
        source: Source with ``synthetic`` set to one of :data:`CONDITIONS`.
            ``limit`` is the clip count for generated conditions and the base
            utterance count for the derived ones; ``sample_seed`` pins the
            noise selection, offsets and pairing.
        sample_rate: Rate every clip is generated at.

    Returns:
        The generated clips, in deterministic order.

    Raises:
        DatasetError: If the kind is unknown, ``limit`` is missing, a derived
            condition names no base corpus, or noise material is unavailable.
    """
    kind = source.synthetic
    seed = source.sample_seed if source.sample_seed is not None else DEFAULT_SEED
    if source.limit is None or source.limit <= 0:
        raise DatasetError(f"synthetic source {kind!r} needs a positive limit (the clip count)")

    if kind == "silence":
        duration = source.duration_s or DEFAULT_DURATION_S
        return [
            silence_clip(
                index,
                duration_s=duration,
                sample_rate=sample_rate,
                language=source.language,
            )
            for index in range(source.limit)
        ]

    if kind == "noise":
        duration = source.duration_s or DEFAULT_DURATION_S
        files = noise_files(source.noise_dir)
        return [
            noise_clip(
                index,
                files,
                duration_s=duration,
                sample_rate=sample_rate,
                seed=seed,
                language=source.language,
            )
            for index in range(source.limit)
        ]

    if kind == "trailing_silence":
        trailing = source.trailing_silence_s or DEFAULT_TRAILING_SILENCE_S
        return [
            trailing_silence_clip(base, trailing_s=trailing) for base in base_clips(source, sample_rate=sample_rate)
        ]

    if kind == "low_snr":
        snr_db = source.snr_db if source.snr_db is not None else DEFAULT_SNR_DB
        files = noise_files(source.noise_dir)
        return [
            low_snr_clip(base, index, files, snr_db=snr_db, seed=seed)
            for index, base in enumerate(base_clips(source, sample_rate=sample_rate))
        ]

    raise DatasetError(f"unknown synthetic source kind {kind!r}; expected one of {', '.join(CONDITIONS)}")


def silence_clip(index: int, *, duration_s: float, sample_rate: int, language: str) -> AudioClip:
    """Build a clip of pure digital silence.

    The correct transcript for silence is nothing at all; the reference is the
    empty string — not ``None`` — so the metrics layer can tell "no speech
    exists" apart from "no reference was collected".

    Args:
        index: Position in the condition set, encoded in the clip id.
        duration_s: Clip length in seconds.
        sample_rate: Sample rate of the generated PCM.
        language: BCP-47 tag the run scores this clip under.

    Returns:
        The generated clip.
    """
    samples = np.zeros(int(sample_rate * duration_s), dtype=np.float32)
    return to_clip(
        samples,
        clip_id=f"silence-{index:03d}",
        reference="",
        language=language,
        sample_rate=sample_rate,
        source_path="<synthetic:silence>",
    )


def noise_clip(
    index: int,
    noise_files: list[Path],
    *,
    duration_s: float,
    sample_rate: int,
    seed: int,
    language: str,
) -> AudioClip:
    """Cut a noise-only clip from recorded noise material.

    File choice and segment offset derive from ``(seed, index)``, so clip
    ``noise-007`` is the same audio in every run of the same config.

    Args:
        index: Position in the condition set, encoded in the clip id.
        noise_files: Candidate noise recordings, in deterministic order.
        duration_s: Clip length in seconds.
        sample_rate: Sample rate of the generated PCM.
        seed: Base seed shared by the whole condition set.
        language: BCP-47 tag the run scores this clip under.

    Returns:
        The generated clip, RMS-normalized to :data:`NOISE_LEVEL_DBFS`.

    Raises:
        DatasetError: If the chosen noise file decodes to silence.
    """
    rng = np.random.default_rng([seed, index])
    path = noise_files[int(rng.integers(len(noise_files)))]
    samples = load_noise(path, sample_rate)
    segment = noise_segment(samples, int(sample_rate * duration_s), rng)

    rms = float(np.sqrt(np.mean(segment**2)))
    if rms <= 0.0:
        raise DatasetError(f"noise file decodes to silence: {path}")
    target = 10.0 ** (NOISE_LEVEL_DBFS / 20.0)
    segment = np.clip(segment * (target / rms), -1.0, 1.0)

    return to_clip(
        segment,
        clip_id=f"noise-{index:03d}",
        reference="",
        language=language,
        sample_rate=sample_rate,
        source_path=f"<synthetic:noise:{path.name}>",
    )


def trailing_silence_clip(base: AudioClip, *, trailing_s: float) -> AudioClip:
    """Append silence to a real utterance.

    A recognizer that keeps decoding after speech has ended tends to fabricate
    during exactly this window. The reference stays the base transcript: any
    extra hypothesis text is insertion by construction.

    Args:
        base: Utterance to extend.
        trailing_s: Seconds of silence appended after the audio.

    Returns:
        The extended clip; ``speech_end_s`` still marks the original speech.
    """
    samples = np.concatenate([
        pcm_to_float(base.pcm),
        np.zeros(int(base.sample_rate * trailing_s), dtype=np.float32),
    ])
    return to_clip(
        samples,
        clip_id=f"trailsil-{base.clip_id}",
        reference=base.reference,
        language=base.language,
        sample_rate=base.sample_rate,
        source_path=base.source_path,
    )


def low_snr_clip(
    base: AudioClip,
    index: int,
    noise_files: list[Path],
    *,
    snr_db: float,
    seed: int,
) -> AudioClip:
    """Mix a real utterance with noise at a fixed active-speech SNR.

    Args:
        base: Utterance supplying the speech.
        index: Position in the condition set, seeding the noise pairing.
        noise_files: Candidate noise recordings, in deterministic order.
        snr_db: Target SNR of active speech over noise; negative buries the
            speech under the noise.
        seed: Base seed shared by the whole condition set.

    Returns:
        The mixed clip, keeping the base reference and language.

    Raises:
        DatasetError: If the chosen noise file decodes to silence.
    """
    rng = np.random.default_rng([seed, index])
    path = noise_files[int(rng.integers(len(noise_files)))]
    speech = pcm_to_float(base.pcm)
    noise = noise_segment(load_noise(path, base.sample_rate), len(speech), rng)
    if float(np.sqrt(np.mean(noise**2))) <= 0.0:
        raise DatasetError(f"noise file decodes to silence: {path}")

    mixed = mix_at_snr(speech, noise, snr_db=snr_db, sample_rate=base.sample_rate)
    return to_clip(
        mixed,
        clip_id=f"lowsnr-{base.clip_id}",
        reference=base.reference,
        language=base.language,
        sample_rate=base.sample_rate,
        source_path=base.source_path,
    )


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, *, snr_db: float, sample_rate: int) -> np.ndarray:
    """Mix speech and noise so active speech sits ``snr_db`` above the noise.

    The SNR is computed against the RMS of *voiced* frames only. Scaling
    against whole-clip RMS would let a clip's leading and trailing silence
    drag the speech power down and quietly deliver an easier mix than the
    config promised.

    Args:
        speech: Mono float speech samples in [-1, 1].
        noise: Mono float noise samples, at least as long as ``speech``.
        snr_db: Target active-speech SNR in decibels.
        sample_rate: Sample rate of both signals.

    Returns:
        The mixture, peak-normalized only if it would clip — a uniform scale
        that preserves the SNR.
    """
    noise = noise[: len(speech)]
    speech_rms = _active_rms(speech, sample_rate)
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    gain = speech_rms / (noise_rms * 10.0 ** (snr_db / 20.0))

    mixed = speech + noise * gain
    peak = float(np.max(np.abs(mixed), initial=0.0))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def _active_rms(samples: np.ndarray, sample_rate: int) -> float:
    """RMS over voiced frames, using the same gate as speech-end detection."""
    step = int(sample_rate * SPEECH_FRAME_MS / 1000)
    usable = len(samples) // step * step
    if step <= 0 or usable == 0:
        return float(np.sqrt(np.mean(samples**2)))

    frames = samples[:usable].reshape(-1, step)
    rms = np.sqrt((frames**2).mean(axis=1))
    threshold = max(float(rms.max()) * SPEECH_THRESHOLD, 1e-4)
    active = rms[rms > threshold]
    if len(active) == 0:
        return float(np.sqrt(np.mean(samples**2)))
    return float(np.sqrt(np.mean(active**2)))


def base_clips(source: SourceConfig, *, sample_rate: int) -> list[AudioClip]:
    """Load the real utterances a derived condition builds on."""
    if not source.parquet and not source.manifest:
        raise DatasetError(f"synthetic source {source.synthetic!r} needs a parquet or manifest of base utterances")
    return load_source(replace(source, synthetic=None), sample_rate=sample_rate)


def noise_files(noise_dir: str | None) -> list[Path]:
    """Collect noise recordings in a deterministic order."""
    if not noise_dir:
        raise DatasetError(f"synthetic source needs noise_dir; {_MUSAN_HINT}")
    root = Path(noise_dir)
    if not root.is_dir():
        raise DatasetError(f"noise_dir not found: {root}; {_MUSAN_HINT}")
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".wav", ".flac"})
    if not files:
        raise DatasetError(f"noise_dir contains no audio files: {root}; {_MUSAN_HINT}")
    return files


def load_noise(path: Path, sample_rate: int) -> np.ndarray:
    """Decode a noise file to mono float samples at ``sample_rate``."""
    try:
        data, source_rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise DatasetError(f"cannot decode noise file {path}: {exc}") from exc
    if data.shape[0] == 0:
        raise DatasetError(f"noise file decodes to zero samples: {path}")
    mono = data.mean(axis=1)
    if source_rate != sample_rate:
        mono = soxr.resample(mono, source_rate, sample_rate, quality="HQ")
    return mono.astype(np.float32)


def noise_segment(samples: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Cut a segment of ``length`` samples, looping the material if short."""
    if len(samples) < length:
        samples = np.resize(samples, length)
    if len(samples) == length:
        return samples.copy()
    offset = int(rng.integers(len(samples) - length + 1))
    return samples[offset : offset + length].copy()


def pcm_to_float(pcm: bytes) -> np.ndarray:
    """Convert little-endian 16-bit PCM bytes back to float samples."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0


def to_clip(
    samples: np.ndarray,
    *,
    clip_id: str,
    reference: str | None,
    language: str,
    sample_rate: int,
    source_path: str,
) -> AudioClip:
    """Package float samples as a clip, mirroring the decode pipeline."""
    pcm = _float_to_pcm16(samples)
    return AudioClip(
        clip_id=clip_id,
        pcm=pcm,
        sample_rate=sample_rate,
        duration_s=len(pcm) / (sample_rate * BYTES_PER_SAMPLE),
        reference=reference,
        language=language,
        source_path=source_path,
        speech_end_s=detect_speech_end_s(samples, sample_rate),
    )
