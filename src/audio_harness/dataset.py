"""Loading evaluation material from disk.

Clips come from a JSONL manifest so the harness stays independent of any
particular corpus layout: point it at LibriSpeech, Common Voice, or a folder of
recordings from your own product, and the records look the same.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from .audio import load_clip
from .config import DatasetConfig
from .types import AudioClip, TtsPrompt


class DatasetError(RuntimeError):
    """Raised when evaluation material cannot be loaded."""


def load_clips(config: DatasetConfig, *, sample_rate: int = 16000) -> list[AudioClip]:
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
        raise DatasetError("dataset.manifest is required for STT benchmarks")
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

        clips.append(
            load_clip(
                audio_path,
                clip_id=str(record.get("id") or audio_path.stem),
                reference=record.get("text"),
                language=str(record.get("language") or config.language),
                target_sample_rate=sample_rate,
            )
        )
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
