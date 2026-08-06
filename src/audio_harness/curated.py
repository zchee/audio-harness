"""Loading curated YODAS/Granary manifests into benchmark clips.

The curated lane redistributes no audio: manifests carry identifiers,
offsets and transcripts (``tools/curate_yodas.py``), and this module fetches
exactly the referenced segments at load time. YODAS2 audio ships as one
``.tar.gz`` per shard — gzip forbids random access, so the shard is streamed
once and decompressed on the fly, keeping only the recordings the manifest
wants and stopping early when the last one has been seen. Extracted segments
land in an on-disk cache keyed by utterance id, so a shard is streamed at
most once per machine, and repeat benchmark runs never touch the network.

Granary rows are loadable when their audio lives in YODAS (the ``utt_id``
encodes subset, shard, video and offsets). The ``ytc`` and ``voxpopuli``
sources host no fetchable audio — YouTube-Commons audio must come from
YouTube itself — so those rows are reported as skipped rather than failing
the whole lane; their transcripts remain useful for text-only work.
"""

from __future__ import annotations

from collections.abc import Buffer, Iterator
from dataclasses import dataclass
from io import RawIOBase
from pathlib import Path
import re
import tarfile
from tempfile import TemporaryDirectory

import httpx
import numpy as np
import orjson
import soundfile as sf
import soxr

from .audio import BYTES_PER_SAMPLE, detect_speech_end_s, wrap_wav
from .config import SourceConfig
from .types import AudioClip


YODAS_AUDIO_BASE = "https://huggingface.co/datasets/espnet/yodas2/resolve/main"

CACHE_DIR = Path("data/cache/curated")
"""Extracted segments, one wav per utterance id. Bounded by the curated
manifest size (hundreds of ~10 s clips), never by shard size."""

CURATED_KEYS = frozenset({"utt_id", "gold_status"})
"""Keys that identify a curated manifest record, as written by
``tools/curate_yodas.py`` — legacy audio-path manifests have neither."""

_GRANARY_YODAS_ID = re.compile(
    r"^(?P<subset>[a-z]{2}\d{3})_(?P<shard>\d{8})_(?P<video>.+)"
    r"_(?P<start_i>\d+)_(?P<start_f>\d+)_(?P<dur_i>\d+)_(?P<dur_f>\d+)$"
)


class CuratedManifestError(RuntimeError):
    """Raised when a curated manifest cannot be loaded as written."""


@dataclass(slots=True, frozen=True)
class _Segment:
    """One audio segment to cut from a YODAS recording."""

    subset: str
    shard: str
    video_id: str
    clip_id: str
    start_s: float
    end_s: float
    reference: str
    language: str
    license: str
    gold_status: str


def is_curated_manifest(path: Path) -> bool:
    """Whether a manifest file holds curated rows rather than audio paths."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = orjson.loads(line)
            return isinstance(record, dict) and record.keys() >= CURATED_KEYS
    except OSError, orjson.JSONDecodeError:
        return False
    return False


def load_curated_clips(source: SourceConfig, *, sample_rate: int = 16000) -> list[AudioClip]:
    """Load the clips a curated manifest references.

    Cached segments load from disk; the rest are cut from their shards, one
    streaming pass per shard. Rows whose audio has no fetchable host (the
    Granary ``ytc``/``voxpopuli`` sources) are skipped, not failed: the lane
    should run on what is reachable and say what was not.

    Args:
        source: Source configuration naming the manifest.
        sample_rate: Rate every clip is resampled to.

    Returns:
        Clips in manifest order, truncated to ``source.limit``.

    Raises:
        CuratedManifestError: If the manifest is malformed or a referenced
            segment cannot be extracted from its shard.
    """
    manifest = Path(source.manifest or "")
    segments, skipped = _read_manifest(manifest, limit=source.limit)
    if not segments:
        if skipped:
            return []
        raise CuratedManifestError(f"manifest yielded no clips: {manifest}")

    _fetch_missing(segments, sample_rate=sample_rate)

    clips: list[AudioClip] = []
    for segment in segments:
        cached = _cache_path(segment.clip_id)
        if not cached.is_file():
            raise CuratedManifestError(
                f"{segment.clip_id}: segment missing after shard extraction — "
                f"the recording {segment.video_id!r} was not in shard "
                f"{segment.subset}/{segment.shard}"
            )
        data, rate = sf.read(cached, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        pcm = _pcm16(mono)
        clips.append(
            AudioClip(
                clip_id=segment.clip_id,
                pcm=pcm,
                sample_rate=rate,
                duration_s=len(pcm) / (rate * BYTES_PER_SAMPLE),
                reference=segment.reference or None,
                language=segment.language,
                source_path=(
                    f"yodas2://{segment.subset}/{segment.shard}/"
                    f"{segment.video_id}#{segment.start_s:.2f}-{segment.end_s:.2f}"
                ),
                speech_end_s=detect_speech_end_s(mono, rate),
                license=segment.license,
                gold_status=segment.gold_status,
            )
        )
    return clips


def _read_manifest(manifest: Path, *, limit: int | None) -> tuple[list[_Segment], list[str]]:
    """Parse manifest rows into fetchable segments plus skip notes."""
    if not manifest.is_file():
        raise CuratedManifestError(f"manifest not found: {manifest}")

    segments: list[_Segment] = []
    skipped: list[str] = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            raise CuratedManifestError(f"{manifest}:{number}: invalid JSON") from exc

        segment = _resolve_row(row, f"{manifest}:{number}")
        if isinstance(segment, str):
            skipped.append(segment)
            continue
        segments.append(segment)
        if limit is not None and len(segments) >= limit:
            break
    return segments, skipped


def _resolve_row(row: dict[str, object], where: str) -> _Segment | str:
    """Map one manifest row to a fetch spec, or a skip note.

    YODAS2 rows carry subset/shard/offsets directly. Granary rows built on
    YODAS encode the same coordinates in their ``utt_id``; the other Granary
    sources have no directly fetchable audio and are skipped by name.
    """
    source = str(row.get("source") or "")
    utt_id = str(row.get("utt_id") or "")
    common = {
        "clip_id": utt_id,
        "reference": str(row.get("text") or ""),
        "language": str(row.get("language") or ""),
        "license": str(row.get("license") or ""),
        "gold_status": str(row.get("gold_status") or ""),
    }
    if source == "yodas2":
        return _Segment(
            subset=str(row.get("subset") or ""),
            shard=str(row.get("shard") or ""),
            video_id=str(row.get("video_id") or ""),
            start_s=_as_float(row.get("start_s")),
            end_s=_as_float(row.get("end_s")),
            **common,
        )
    if source == "granary":
        subset = str(row.get("subset") or "")
        if subset != "yodas":
            return f"granary/{subset}"
        match = _GRANARY_YODAS_ID.match(utt_id)
        if match is None:
            raise CuratedManifestError(
                f"{where}: granary/yodas utt_id {utt_id!r} does not encode its YODAS coordinates"
            )
        start_s = float(f"{match.group('start_i')}.{match.group('start_f')}")
        duration = float(f"{match.group('dur_i')}.{match.group('dur_f')}")
        return _Segment(
            subset=match.group("subset"),
            shard=match.group("shard"),
            video_id=match.group("video"),
            start_s=start_s,
            end_s=start_s + duration,
            **common,
        )
    raise CuratedManifestError(f"{where}: unknown curated source {source!r}")


def _as_float(value: object) -> float:
    """Coerce a manifest number, treating anything else as zero."""
    return float(value) if isinstance(value, int | float) else 0.0


def _fetch_missing(segments: list[_Segment], *, sample_rate: int) -> None:
    """Extract every uncached segment, one streaming pass per shard."""
    by_shard: dict[tuple[str, str], list[_Segment]] = {}
    for segment in segments:
        if not _cache_path(segment.clip_id).is_file():
            by_shard.setdefault((segment.subset, segment.shard), []).append(segment)

    for (subset, shard), wanted in by_shard.items():
        _extract_shard(subset, shard, wanted, sample_rate=sample_rate)


def _extract_shard(subset: str, shard: str, wanted: list[_Segment], *, sample_rate: int) -> None:
    """Stream one shard tarball and cache every wanted segment.

    The tarball is read sequentially (gzip allows nothing else) and the
    stream stops as soon as the last wanted recording has been cut, so the
    average cost is a fraction of the shard. Only cut segments touch the
    disk — never the tarball, never a whole recording.
    """
    by_video: dict[str, list[_Segment]] = {}
    for segment in wanted:
        by_video.setdefault(segment.video_id, []).append(segment)

    remaining = set(by_video)
    stream = _open_shard(subset, shard)
    try:
        with tarfile.open(fileobj=stream, mode="r|*") as archive:
            for member in archive:
                stem = Path(member.name).stem
                if stem not in remaining:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                with TemporaryDirectory() as scratch:
                    recording = Path(scratch) / Path(member.name).name
                    recording.write_bytes(extracted.read())
                    for segment in by_video[stem]:
                        _cut_segment(recording, segment, sample_rate=sample_rate)
                remaining.discard(stem)
                if not remaining:
                    break
    finally:
        stream.close()


def _cut_segment(recording: Path, segment: _Segment, *, sample_rate: int) -> None:
    """Cut one segment from a recording into the cache, resampled and mono."""
    with sf.SoundFile(recording) as handle:
        native = handle.samplerate
        start = min(int(segment.start_s * native), max(len(handle) - 1, 0))
        frames = max(int((segment.end_s - segment.start_s) * native), 1)
        handle.seek(start)
        data = handle.read(frames, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if native != sample_rate:
        mono = soxr.resample(mono, native, sample_rate, quality="HQ")

    target = _cache_path(segment.clip_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(wrap_wav(_pcm16(mono), sample_rate))


def _pcm16(samples: np.ndarray) -> bytes:
    """Convert float samples in [-1, 1] to little-endian 16-bit PCM bytes."""
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _cache_path(clip_id: str) -> Path:
    """Cache location for one extracted segment."""
    return CACHE_DIR / f"{clip_id}.wav"


def _open_shard(subset: str, shard: str) -> RawIOBase:
    """Open a streaming reader over one shard tarball.

    Module-level so tests can substitute a local fixture archive; the real
    implementation streams from Hugging Face without ever storing the
    tarball.
    """
    url = f"{YODAS_AUDIO_BASE}/data/{subset}/audio/{shard}.tar.gz"
    return _HttpStream(url)


class _HttpStream(RawIOBase):
    """Minimal read-only file object over a streaming HTTP response."""

    def __init__(self, url: str) -> None:
        self._client = httpx.Client(timeout=httpx.Timeout(300.0, connect=15.0), follow_redirects=True)
        self._response = self._client.send(self._client.build_request("GET", url), stream=True)
        if self._response.status_code >= 400:
            self.close()
            raise CuratedManifestError(f"HTTP {self._response.status_code} fetching {url}")
        self._chunks: Iterator[bytes] = self._response.iter_bytes()
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def readinto(self, target: Buffer) -> int:
        view = memoryview(target)
        while len(self._buffer) < len(view):
            try:
                self._buffer += next(self._chunks)
            except StopIteration:
                break
        take = min(len(view), len(self._buffer))
        view[:take] = self._buffer[:take]
        self._buffer = self._buffer[take:]
        return take

    def close(self) -> None:
        try:
            self._response.close()
            self._client.close()
        finally:
            super().close()
