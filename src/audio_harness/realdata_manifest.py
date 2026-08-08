"""Load reference-free recordings from real-data JSONL manifests."""

from __future__ import annotations

import math
from pathlib import Path
import warnings

import orjson

from .audio import load_clip
from .config import SourceConfig
from .curated import CURATED_KEYS
from .dataset import DatasetError
from .types import AudioClip


REALDATA_KEYS = frozenset({"clip"})
"""Keys distinguishing real-data rows from ordinary and curated manifests."""

_DURATION_ABS_TOLERANCE_S = 0.1
_DURATION_REL_TOLERANCE = 0.02


def is_realdata_manifest(path: Path) -> bool:
    """Whether a manifest contains reference-free real-data rows.

    The first non-empty row must contain ``clip`` and must not contain
    ``audio``. Curated rows use ``utt_id``/``gold_status`` and ordinary local
    manifests use ``audio``, so the three dispatch shapes cannot collide.

    Args:
        path: Candidate JSONL manifest.

    Returns:
        ``True`` when the first usable row has the real-data shape.
    """
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = orjson.loads(line)
            return (
                isinstance(record, dict)
                and record.keys() >= REALDATA_KEYS
                and not record.keys() >= CURATED_KEYS
                and "audio" not in record
            )
    except OSError, orjson.JSONDecodeError:
        return False
    return False


def load_realdata_clips(source: SourceConfig, *, sample_rate: int = 16000) -> list[AudioClip]:
    """Load reference-free recordings named by a real-data manifest.

    Relative ``clip`` paths resolve against the manifest directory. Every
    resulting clip has ``reference=None`` even if an unrelated ``text`` key is
    present. ``language`` overrides the source fallback per row. ``session``
    is validated but not attached because :class:`AudioClip` has no generic
    metadata field; the true file path remains in ``source_path``. A supplied
    ``duration_s`` is checked against the decoded audio and warns when the
    difference exceeds two percent or 100 ms, whichever is larger.

    Args:
        source: Source configuration naming the JSONL manifest.
        sample_rate: Rate every clip is resampled to.

    Returns:
        Decoded clips in manifest order, truncated to ``source.limit``.

    Raises:
        DatasetError: If the manifest is missing, a row is malformed, audio
            cannot be decoded, or the manifest yields no clips.
    """
    if not source.manifest:
        raise DatasetError("real-data sources need dataset.manifest")
    manifest = Path(source.manifest)
    if not manifest.is_file():
        raise DatasetError(f"manifest not found: {manifest}")

    clips: list[AudioClip] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{manifest}:{line_number}"
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            raise DatasetError(f"{where}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise DatasetError(f"{where}: record needs a 'clip' key")

        clip_value = record.get("clip")
        if not isinstance(clip_value, str) or not clip_value.strip() or "audio" in record:
            raise DatasetError(f"{where}: record needs a non-empty 'clip' key and no 'audio' key")
        _validate_session(record.get("session"), where)
        stated_duration = _duration_hint(record.get("duration_s"), where)

        audio_path = Path(clip_value)
        if not audio_path.is_absolute():
            audio_path = manifest.parent / audio_path

        try:
            clip = load_clip(
                audio_path,
                clip_id=str(record.get("id") or audio_path.stem),
                reference=None,
                language=str(record.get("language") or source.language),
                target_sample_rate=sample_rate,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise DatasetError(f"{where}: {exc}") from exc
        _warn_on_duration_mismatch(clip, stated_duration, where)
        clips.append(clip)
        if source.limit is not None and len(clips) >= source.limit:
            break

    if not clips:
        raise DatasetError(f"manifest yielded no clips: {manifest}")
    return clips


def _validate_session(value: object, where: str) -> None:
    """Validate optional session provenance without changing ``AudioClip``."""
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise DatasetError(f"{where}: 'session' must be a non-empty string when present")


def _duration_hint(value: object, where: str) -> float | None:
    """Return a validated manifest duration hint."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DatasetError(f"{where}: 'duration_s' must be a positive number when present")
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise DatasetError(f"{where}: 'duration_s' must be a positive number when present")
    return duration


def _warn_on_duration_mismatch(clip: AudioClip, stated_duration: float | None, where: str) -> None:
    """Warn when manifest metadata disagrees materially with decoded audio."""
    if stated_duration is None:
        return
    tolerance = max(_DURATION_ABS_TOLERANCE_S, stated_duration * _DURATION_REL_TOLERANCE)
    if abs(clip.duration_s - stated_duration) <= tolerance:
        return
    warnings.warn(
        f"{where}: duration_s={stated_duration:.3f} differs from decoded audio duration {clip.duration_s:.3f}s",
        stacklevel=2,
    )
