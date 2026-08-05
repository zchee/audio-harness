"""Loading evaluation material from disk.

Clips come from a JSONL manifest so the harness stays independent of any
particular corpus layout: point it at LibriSpeech, Common Voice, or a folder of
recordings from your own product, and the records look the same.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import polars as pl
import soundfile as sf

from .audio import load_clip, load_clip_bytes
from .config import DatasetConfig
from .types import AudioClip, TtsPrompt


class DatasetError(RuntimeError):
    """Raised when evaluation material cannot be loaded."""


def load_clips(config: DatasetConfig, *, sample_rate: int = 16000) -> list[AudioClip]:
    """Load evaluation clips from whichever source the config names.

    Args:
        config: Dataset section of the benchmark configuration.
        sample_rate: Rate every clip is resampled to.

    Returns:
        The decoded clips.

    Raises:
        DatasetError: If neither or both of ``manifest`` and ``parquet`` are
            set, or the named source cannot be read.
    """
    if config.manifest and config.parquet:
        raise DatasetError(
            "set only one of dataset.manifest or dataset.parquet, not both"
        )
    if config.parquet:
        return load_clips_from_parquet(config, sample_rate=sample_rate)
    return load_clips_from_manifest(config, sample_rate=sample_rate)


def load_clips_from_parquet(
    config: DatasetConfig, *, sample_rate: int = 16000
) -> list[AudioClip]:
    """Load clips from a parquet corpus with embedded audio.

    Hugging Face audio datasets ship this way: one row per clip, with the audio
    stored as a ``{bytes, path}`` struct beside its transcript. Rows are read
    lazily and only the selected subset is decoded, so a large corpus does not
    have to fit in memory to run a 50-clip sample.

    Args:
        config: Dataset section of the benchmark configuration.
        sample_rate: Rate every clip is resampled to.

    Returns:
        The decoded clips.

    Raises:
        DatasetError: If the file is missing, a configured column is absent, or
            no clip could be decoded.
    """
    path = Path(config.parquet or "")
    if not path.is_file():
        raise DatasetError(f"parquet file not found: {path}")

    frame = pl.scan_parquet(path)
    available = set(frame.collect_schema().names())
    required = {config.id_column, config.audio_column, config.text_column}
    missing = sorted(required - available)
    if missing:
        raise DatasetError(
            f"{path}: missing column(s) {', '.join(missing)}; "
            f"available: {', '.join(sorted(available))}"
        )

    selected = frame.select(
        pl.col(config.id_column).alias("id"),
        pl.col(config.audio_column).alias("audio"),
        pl.col(config.text_column).alias("text"),
    ).collect()

    if config.limit is not None and config.limit < selected.height:
        if config.sample_seed is None:
            selected = selected.head(config.limit)
        else:
            selected = selected.sample(
                n=config.limit, seed=config.sample_seed, shuffle=True
            )

    clips: list[AudioClip] = []
    failures: list[str] = []
    for row in selected.iter_rows(named=True):
        clip_id = str(row["id"])
        try:
            clips.append(
                load_clip_bytes(
                    _audio_bytes(row["audio"], clip_id),
                    clip_id=clip_id,
                    reference=row["text"],
                    language=config.language,
                    target_sample_rate=sample_rate,
                    source_path=f"{path}#{clip_id}",
                )
            )
        except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
            failures.append(f"{clip_id}: {exc}")

    if not clips:
        raise DatasetError(f"{path}: no clip could be decoded")
    if failures:
        preview = "; ".join(failures[:3])
        raise DatasetError(
            f"{path}: {len(failures)} of {selected.height} clips failed to "
            f"decode ({preview}). Undecodable audio would be scored as a "
            f"provider failure, so the corpus is rejected instead."
        )
    return clips


def _audio_bytes(value: object, clip_id: str) -> bytes:
    """Extract encoded audio from a parquet cell.

    Accepts the Hugging Face ``{bytes, path}`` struct as well as a column of
    raw bytes, since both conventions appear in the wild.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        payload = value.get("bytes")
        if isinstance(payload, bytes):
            return payload
        location = value.get("path")
        if isinstance(location, str) and location:
            return Path(location).read_bytes()
    raise ValueError(f"clip {clip_id}: unsupported audio cell {type(value).__name__}")


def load_clips_from_manifest(
    config: DatasetConfig, *, sample_rate: int = 16000
) -> list[AudioClip]:
    """Load audio clips listed in a JSONL manifest.

    Each line is a JSON object with:

    * ``audio`` (required) — path to the audio file, absolute or relative to
      the manifest's own directory.
    * ``text`` — the reference transcript. Clips without one are still timed
      but contribute nothing to the error rate.
    * ``id`` — stable clip identifier; defaults to the file stem.
    * ``language`` — BCP-47 tag; defaults to the dataset language.

    Args:
        config: Dataset section of the benchmark configuration.
        sample_rate: Rate every clip is resampled to.

    Returns:
        Clips in manifest order, truncated to ``config.limit``.

    Raises:
        DatasetError: If no manifest is configured, the file is missing, a
            record is malformed, or the manifest yields no clips.
    """
    if not config.manifest:
        raise DatasetError("STT benchmarks need dataset.manifest or dataset.parquet")
    manifest = Path(config.manifest)
    if not manifest.is_file():
        raise DatasetError(f"manifest not found: {manifest}")

    clips: list[AudioClip] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            raise DatasetError(f"{manifest}:{line_number}: invalid JSON") from exc
        if not isinstance(record, dict) or "audio" not in record:
            raise DatasetError(f"{manifest}:{line_number}: record needs an 'audio' key")

        audio_path = Path(record["audio"])
        if not audio_path.is_absolute():
            audio_path = manifest.parent / audio_path

        try:
            clips.append(
                load_clip(
                    audio_path,
                    clip_id=str(record.get("id") or audio_path.stem),
                    reference=record.get("text"),
                    language=str(record.get("language") or config.language),
                    target_sample_rate=sample_rate,
                )
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise DatasetError(f"{manifest}:{line_number}: {exc}") from exc
        if config.limit is not None and len(clips) >= config.limit:
            break

    if not clips:
        raise DatasetError(f"manifest yielded no clips: {manifest}")
    return clips


def load_prompts(config: DatasetConfig) -> list[TtsPrompt]:
    """Load TTS prompts, one per non-empty line of a text file.

    Args:
        config: Dataset section of the benchmark configuration.

    Returns:
        Prompts in file order, truncated to ``config.limit``.

    Raises:
        DatasetError: If no prompt file is configured, it is missing, or it
            contains no usable lines.
    """
    if not config.prompts:
        raise DatasetError("dataset.prompts is required for TTS benchmarks")
    path = Path(config.prompts)
    if not path.is_file():
        raise DatasetError(f"prompt file not found: {path}")

    prompts = [
        TtsPrompt(
            prompt_id=f"p{index:04d}", text=line.strip(), language=config.language
        )
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
        if line.strip()
    ]
    if config.limit is not None:
        prompts = prompts[: config.limit]
    if not prompts:
        raise DatasetError(f"prompt file yielded no prompts: {path}")
    return prompts
