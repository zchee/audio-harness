"""Audio decoding, resampling and real-time pacing.

Streaming latency numbers are only meaningful when audio is fed to the provider
at the same rate a microphone would produce it. Blasting a whole file into a
socket measures throughput, not latency, so :func:`pace_chunks` gates every
chunk on a wall-clock deadline.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
import wave

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


def decode_container_pcm16(payload: bytes) -> tuple[bytes, int]:
    """Decode container audio to native-rate mono 16-bit PCM.

    Args:
        payload: Encoded audio in a format supported by libsndfile.

    Returns:
        A pair of little-endian mono PCM bytes and the decoded sample rate.

    Raises:
        ValueError: If the payload is empty or decodes to zero samples.
    """
    if not payload:
        raise ValueError("audio payload is empty")
    data, sample_rate = sf.read(BytesIO(payload), dtype="float32", always_2d=True)
    if data.shape[0] == 0:
        raise ValueError("audio payload decoded to zero samples")
    return _float_to_pcm16(data.mean(axis=1)), int(sample_rate)


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
        speech_end_s=detect_speech_end_s(mono, target_sample_rate),
    )


def _float_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert float samples in [-1, 1] to little-endian 16-bit PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


SPEECH_FRAME_MS = 20
SPEECH_THRESHOLD = 0.02
"""Frame energy, relative to the clip's loudest frame, counted as speech."""


def detect_speech_end_s(samples: np.ndarray, sample_rate: int, *, frame_ms: int = SPEECH_FRAME_MS) -> float:
    """Return the offset of the last voiced frame, in seconds.

    Uses per-frame RMS against a threshold relative to the clip's own peak, so
    it adapts to recording level instead of assuming a fixed noise floor. This
    is deliberately simple: it locates the end of speech for latency
    accounting, it is not a VAD for segmentation.

    Args:
        samples: Mono float samples in [-1, 1].
        sample_rate: Sample rate of ``samples``.
        frame_ms: Analysis frame size in milliseconds.

    Returns:
        Seconds from the start of the clip to the end of the last voiced
        frame, or the full duration when no frame clears the threshold.
    """
    step = int(sample_rate * frame_ms / 1000)
    usable = len(samples) // step * step
    if step <= 0 or usable == 0:
        return len(samples) / sample_rate

    frames = samples[:usable].reshape(-1, step)
    rms = np.sqrt((frames**2).mean(axis=1))
    threshold = max(float(rms.max()) * SPEECH_THRESHOLD, 1e-4)
    voiced = np.nonzero(rms > threshold)[0]
    if len(voiced) == 0:
        return len(samples) / sample_rate
    return float((voiced[-1] + 1) * frame_ms / 1000)


def detect_speech_onset_s(samples: np.ndarray, sample_rate: int, *, frame_ms: int = SPEECH_FRAME_MS) -> float | None:
    """Return the offset of the first voiced frame, in seconds.

    The mirror of :func:`detect_speech_end_s`, used by the TTS lane to locate
    when synthesized audio becomes audible: vendors pad output with leading
    silence, and time-to-first-byte credits them for it. The threshold is
    relative to the clip's own peak, so recording level does not matter.

    Args:
        samples: Mono float samples in [-1, 1].
        sample_rate: Sample rate of ``samples``.
        frame_ms: Analysis frame size in milliseconds.

    Returns:
        Seconds from the start of the clip to the first voiced frame, or
        ``None`` when no frame clears the threshold — silence has no onset.
    """
    step = int(sample_rate * frame_ms / 1000)
    usable = len(samples) // step * step
    if step <= 0 or usable == 0:
        return None

    frames = samples[:usable].reshape(-1, step)
    rms = np.sqrt((frames**2).mean(axis=1))
    threshold = max(float(rms.max()) * SPEECH_THRESHOLD, 1e-4)
    voiced = np.nonzero(rms > threshold)[0]
    if len(voiced) == 0:
        return None
    return float(voiced[0] * frame_ms / 1000)


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Convert little-endian 16-bit PCM bytes to float samples in [-1, 1].

    A trailing odd byte — a truncated final sample from a cut-off stream — is
    dropped rather than raised on, because analysis of what did arrive is
    exactly what a truncated stream needs.
    """
    usable = len(pcm) // BYTES_PER_SAMPLE * BYTES_PER_SAMPLE
    return np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0


def pcm_f32le_to_s16le(payload: bytes) -> bytes:
    """Convert little-endian float32 mono PCM to little-endian int16 PCM.

    Samples are clamped to ``[-1, 1]`` before scaling. A trailing partial
    float32 sample is dropped, matching :func:`pcm16_to_float`'s handling of
    truncated input.
    """
    usable = len(payload) // 4 * 4
    samples = np.frombuffer(payload[:usable], dtype="<f4")
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def wav_data_offset(payload: bytes) -> int:
    """Byte offset of the PCM samples, skipping a RIFF/WAVE header when present.

    Some vendors wrap a nominally raw PCM response in a WAV container. The
    header bytes decode as one loud click, which would defeat RMS analysis by
    making the very first frame look voiced, so scans start after it.
    Headerless payloads return 0.
    """
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        return 0
    offset = 12
    while offset + 8 <= len(payload):
        chunk_id = payload[offset : offset + 4]
        size = int.from_bytes(payload[offset + 4 : offset + 8], "little")
        if chunk_id == b"data":
            return offset + 8
        offset += 8 + size + (size & 1)
    return 0


def read_audio_samples(path: str | Path) -> tuple[np.ndarray, int] | None:
    """Decode an audio file to mono float samples and its sample rate.

    Reporting reads saved synthesis WAVs back for pause analysis; a missing
    or corrupt file degrades to "no pause stats" (``None``) rather than
    failing the whole report over one clip.
    """
    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except OSError, RuntimeError, sf.LibsndfileError:
        return None
    if data.shape[0] == 0:
        return None
    return data.mean(axis=1), int(rate)


MIN_PAUSE_S = 0.15
"""Internal silences shorter than this are articulation gaps, not pauses."""


@dataclass(slots=True, frozen=True)
class PauseStats:
    """Internal-pause profile of one utterance.

    Leading and trailing silence are excluded — they belong to latency and
    endpointing metrics, not to phrasing. What is profiled here is the silence
    a listener hears *inside* the speech: streamed synthesis can introduce
    seams at chunk boundaries that batch synthesis of the same text does not.

    Attributes:
        total_s: Summed duration of qualifying internal pauses.
        longest_s: Duration of the single longest pause.
        count: Number of pauses at least ``MIN_PAUSE_S`` long.
    """

    total_s: float
    longest_s: float
    count: int


def measure_pauses(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: int = SPEECH_FRAME_MS,
    min_pause_s: float = MIN_PAUSE_S,
) -> PauseStats:
    """Profile the internal pauses of an utterance.

    Uses the same relative-RMS voicing test as :func:`detect_speech_end_s`,
    then measures unvoiced runs strictly between the first and last voiced
    frames. Runs shorter than ``min_pause_s`` are articulation gaps — stop
    consonants produce them in perfectly natural speech — and do not count.

    Args:
        samples: Mono float samples in [-1, 1].
        sample_rate: Sample rate of ``samples``.
        frame_ms: Analysis frame size in milliseconds.
        min_pause_s: Shortest unvoiced run counted as a pause.

    Returns:
        The pause profile; all zeros when nothing is voiced.
    """
    step = int(sample_rate * frame_ms / 1000)
    usable = len(samples) // step * step
    if step <= 0 or usable == 0:
        return PauseStats(0.0, 0.0, 0)

    frames = samples[:usable].reshape(-1, step)
    rms = np.sqrt((frames**2).mean(axis=1))
    threshold = max(float(rms.max()) * SPEECH_THRESHOLD, 1e-4)
    voiced = np.nonzero(rms > threshold)[0]
    if len(voiced) < 2:
        return PauseStats(0.0, 0.0, 0)

    total = 0.0
    longest = 0.0
    count = 0
    for gap_frames in np.diff(voiced) - 1:
        pause_s = float(gap_frames) * frame_ms / 1000
        if pause_s < min_pause_s:
            continue
        count += 1
        total += pause_s
        longest = max(longest, pause_s)
    return PauseStats(total, longest, count)


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
