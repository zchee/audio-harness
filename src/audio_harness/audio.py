"""Audio decoding, resampling and real-time pacing.

Streaming latency numbers are only meaningful when audio is fed to the provider
at the same rate a microphone would produce it. Blasting a whole file into a
socket measures throughput, not latency, so :func:`pace_chunks` gates every
chunk on a wall-clock deadline.
"""

from __future__ import annotations

import asyncio
import time
import wave
from collections.abc import AsyncIterator, Iterator
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .types import AudioClip

BYTES_PER_SAMPLE = 2
"""Width of one mono sample in the harness' canonical pcm_s16le format."""


def load_clip(
    path: str | Path,
    *,
    clip_id: str,
    reference: str | None,
    language: str,
    target_sample_rate: int = 16000,
) -> AudioClip:
    """Decode an audio file to mono 16-bit PCM at ``target_sample_rate``.

    Args:
        path: Audio file in any format libsndfile can read (wav, flac, ogg...).
        clip_id: Stable identifier for the resulting clip.
        reference: Ground-truth transcript, or ``None`` for latency-only clips.
        language: BCP-47 language tag for the clip.
        target_sample_rate: Sample rate every provider will receive.

    Returns:
        The decoded clip.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file decodes to zero samples.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    return _decode(
        path,
        clip_id=clip_id,
        reference=reference,
        language=language,
        target_sample_rate=target_sample_rate,
        source_path=str(path),
    )


def load_clip_bytes(
    payload: bytes,
    *,
    clip_id: str,
    reference: str | None,
    language: str,
    target_sample_rate: int = 16000,
    source_path: str = "<embedded>",
) -> AudioClip:
    """Decode in-memory audio to mono 16-bit PCM at ``target_sample_rate``.

    Corpora distributed as parquet embed the audio in the table rather than as
    files on disk. Decoding straight from those bytes avoids materializing a
    second copy of the corpus just to hand it back to the same process.

    Args:
        payload: Encoded audio in any format libsndfile can read.
        clip_id: Stable identifier for the resulting clip.
        reference: Ground-truth transcript, or ``None`` for latency-only clips.
        language: BCP-47 language tag for the clip.
        target_sample_rate: Sample rate every provider will receive.
        source_path: Provenance label recorded on the clip.

    Returns:
        The decoded clip.

    Raises:
        ValueError: If the payload is empty or decodes to zero samples.
    """
    if not payload:
        raise ValueError(f"clip {clip_id}: audio payload is empty")
    return _decode(
        BytesIO(payload),
        clip_id=clip_id,
        reference=reference,
        language=language,
        target_sample_rate=target_sample_rate,
        source_path=source_path,
    )


def _decode(
    source: Path | BytesIO,
    *,
    clip_id: str,
    reference: str | None,
    language: str,
    target_sample_rate: int,
    source_path: str,
) -> AudioClip:
    """Decode, downmix and resample audio from a path or an in-memory buffer."""
    data, source_rate = sf.read(source, dtype="float32", always_2d=True)
    if data.shape[0] == 0:
        raise ValueError(f"clip {clip_id}: audio decoded to zero samples")

    mono = data.mean(axis=1)
    if source_rate != target_sample_rate:
        mono = soxr.resample(mono, source_rate, target_sample_rate, quality="HQ")

    pcm = _float_to_pcm16(mono)
    return AudioClip(
        clip_id=clip_id,
        pcm=pcm,
        sample_rate=target_sample_rate,
        duration_s=len(pcm) / (target_sample_rate * BYTES_PER_SAMPLE),
        reference=reference,
        language=language,
        source_path=source_path,
    )


def _float_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert float samples in [-1, 1] to little-endian 16-bit PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def chunk_pcm(clip: AudioClip, chunk_ms: int) -> Iterator[bytes]:
    """Split a clip's PCM into fixed-duration chunks.

    Args:
        clip: Source clip.
        chunk_ms: Chunk duration in milliseconds.

    Yields:
        Successive PCM chunks; the final chunk may be shorter.
    """
    step = int(clip.sample_rate * chunk_ms / 1000) * BYTES_PER_SAMPLE
    for offset in range(0, len(clip.pcm), step):
        yield clip.pcm[offset : offset + step]


async def pace_chunks(
    clip: AudioClip,
    chunk_ms: int,
    *,
    realtime: bool = True,
) -> AsyncIterator[bytes]:
    """Yield PCM chunks, optionally throttled to real-time playback speed.

    Deadlines are computed from a single start instant rather than by sleeping
    a fixed interval per chunk, so scheduler jitter cannot accumulate into
    systematic drift over a long clip.

    Args:
        clip: Source clip.
        chunk_ms: Chunk duration in milliseconds.
        realtime: When ``True``, gate each chunk on its playback deadline.
            When ``False``, yield as fast as the consumer accepts, which
            measures throughput instead of latency.

    Yields:
        Successive PCM chunks.
    """
    start = time.perf_counter()
    for index, chunk in enumerate(chunk_pcm(clip, chunk_ms)):
        if realtime:
            deadline = start + (index * chunk_ms / 1000)
            delay = deadline - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        yield chunk


def pcm_duration_s(pcm: bytes, sample_rate: int) -> float:
    """Return the duration in seconds of mono 16-bit PCM bytes."""
    return len(pcm) / (sample_rate * BYTES_PER_SAMPLE)


def wrap_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw mono 16-bit PCM in a WAV container.

    Several batch endpoints reject headerless PCM, so adapters that post whole
    files upload the WAV form instead.
    """
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(BYTES_PER_SAMPLE)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def decode_audio_duration(payload: bytes, *, encoding: str, sample_rate: int) -> float:
    """Measure the duration of synthesized audio returned by a TTS provider.

    Raw PCM carries no header, so its duration is computed arithmetically.
    Container formats are probed with libsndfile, which knows their real sample
    rate — the requested rate is only a fallback for formats it cannot parse.

    Args:
        payload: Audio bytes as returned by the provider.
        encoding: Provider-declared encoding (``pcm_s16le``, ``mp3``, ...).
        sample_rate: Sample rate requested from the provider.

    Returns:
        Duration in seconds, or ``0.0`` if the payload cannot be parsed.
    """
    if not payload:
        return 0.0
    if encoding.startswith("pcm"):
        return pcm_duration_s(payload, sample_rate)
    try:
        with sf.SoundFile(BytesIO(payload)) as handle:
            return len(handle) / handle.samplerate
    except RuntimeError, sf.LibsndfileError:
        return 0.0
