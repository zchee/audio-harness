"""Loading evaluation material from disk.

Clips come from a JSONL manifest so the harness stays independent of any
particular corpus layout: point it at LibriSpeech, Common Voice, or a folder of
recordings from your own product, and the records look the same.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import orjson
import polars as pl
import soundfile as sf

from .audio import load_clip, load_clip_bytes
from .config import DatasetConfig, SourceConfig
from .types import AudioClip, TtsPrompt


class DatasetError(RuntimeError):
    """Raised when evaluation material cannot be loaded."""


def load_clips(config: DatasetConfig, *, sample_rate: int = 16000) -> list[AudioClip]:
    """Load evaluation clips from every source the config names.

    Clips from different languages are returned together; each carries its own
    language tag, so the metrics layer scores them separately rather than
    pooling error rates that are not comparable.

    Args:
        config: Dataset section of the benchmark configuration.
        sample_rate: Rate every clip is resampled to.

    Returns:
        The decoded clips, in source order.

    Raises:
        DatasetError: If a source names neither or both of ``manifest`` and
            ``parquet``, or cannot be read.
    """
    clips: list[AudioClip] = []
    for source in config.resolved_sources():
        clips.extend(load_source(source, sample_rate=sample_rate))
    return clips


def load_source(source: SourceConfig, *, sample_rate: int = 16000) -> list[AudioClip]:
    """Load one corpus in one language, or generate a synthetic condition set.

    Raises:
        DatasetError: If the source names neither or both of ``manifest`` and
            ``parquet``, cannot be read, or describes an unusable synthetic
            condition.
    """
    if source.synthetic in {"snr", "telephony"}:
        # The SNR robustness matrix lives in its own module; dispatching here
        # keeps synthetic.py's condition set closed. Lazy import: snr.py
        # reaches back through synthetic.py into this module.
        from .snr import synthesize_snr_source

        return synthesize_snr_source(source, sample_rate=sample_rate)
    if source.synthetic:
        # Imported here, not at module top: synthetic.py loads real base
        # utterances through this module, so a top-level import would be
        # circular.
        from .synthetic import synthesize_source

        return synthesize_source(source, sample_rate=sample_rate)
    if source.manifest and source.parquet:
        raise DatasetError("set only one of dataset.manifest or dataset.parquet, not both")
    if source.parquet:
        return load_clips_from_parquet(source, sample_rate=sample_rate)
    if source.manifest:
        # Curated YODAS/Granary manifests reference remote audio by id and
        # offset instead of naming local files; detection is per file so the
        # config stays a plain `manifest:` entry either way.
        from .curated import CuratedManifestError, is_curated_manifest

        if is_curated_manifest(Path(source.manifest)):
            from .curated import load_curated_clips

            try:
                return load_curated_clips(source, sample_rate=sample_rate)
            except CuratedManifestError as exc:
                raise DatasetError(str(exc)) from exc
        # Real recordings use a local `clip` path and deliberately carry no
        # ground-truth transcript. Shape detection keeps the existing config
        # surface unchanged and leaves ordinary `audio` manifests untouched.
        from .realdata_manifest import is_realdata_manifest

        if is_realdata_manifest(Path(source.manifest)):
            from .realdata_manifest import load_realdata_clips

            return load_realdata_clips(source, sample_rate=sample_rate)
    return load_clips_from_manifest(source, sample_rate=sample_rate)


def load_clips_from_parquet(config: SourceConfig, *, sample_rate: int = 16000) -> list[AudioClip]:
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
    spec = config.parquet or ""
    path = Path(spec)
    # Corpora are often sharded (validation-00000-of-00002.parquet, ...). A
    # glob keeps the config a single line instead of a shard list, and polars
    # scans all matches as one table.
    is_glob = "*" in spec or "?" in spec
    if not path.is_file() and (not is_glob or not sorted(path.parent.glob(path.name))):
        raise DatasetError(f"parquet file not found: {spec}")

    frame = pl.scan_parquet(spec)
    available = set(frame.collect_schema().names())
    spans_column = config.silence_spans_column
    required = {config.id_column, config.audio_column}
    if spans_column:
        # Endpointing corpora (eot-bench schema) label silence spans instead
        # of shipping a plain transcript; the text column becomes optional.
        required.add(spans_column)
    else:
        required.add(config.text_column)
    if config.words_column:
        required.add(config.words_column)
    missing = sorted(required - available)
    if missing:
        raise DatasetError(f"{path}: missing column(s) {', '.join(missing)}; available: {', '.join(sorted(available))}")

    columns = [
        pl.col(config.id_column).alias("id"),
        pl.col(config.audio_column).alias("audio"),
    ]
    has_text = config.text_column in available
    if has_text:
        columns.append(pl.col(config.text_column).alias("text"))
    if config.words_column:
        columns.append(pl.col(config.words_column).alias("words"))
    if spans_column:
        columns.append(pl.col(spans_column).alias("spans"))
    projected = frame.select(columns)
    selected = _select_rows(projected, limit=config.limit, seed=config.sample_seed)

    clips: list[AudioClip] = []
    failures: list[str] = []
    for row in selected.iter_rows(named=True):
        clip_id = str(row["id"])
        if has_text:
            reference = row["text"]
        elif config.words_column:
            reference = _join_words(row.get("words"))
        else:
            reference = None
        try:
            clip = load_clip_bytes(
                _audio_bytes(row["audio"], clip_id),
                clip_id=clip_id,
                reference=reference,
                language=config.language,
                target_sample_rate=sample_rate,
                source_path=f"{path}#{clip_id}",
            )
            if spans_column:
                clip = _apply_silence_spans(clip, row.get("spans"))
            clips.append(clip)
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


def _select_rows(frame: pl.LazyFrame, *, limit: int | None, seed: int | None) -> pl.DataFrame:
    """Materialize only the rows a run will actually use.

    Audio is the overwhelming majority of a speech corpus by bytes, so
    collecting the whole table and slicing afterwards pulls gigabytes through
    memory to benchmark thirty clips. Row selection happens against the id
    column alone, and only the chosen rows are then decoded.

    Args:
        frame: Projected lazy frame with ``id``, ``audio`` and ``text``.
        limit: Maximum rows to take, or ``None`` for all of them.
        seed: When set, take a reproducible random sample instead of the head.

    Returns:
        The selected rows, in corpus order.
    """
    if limit is None:
        return frame.collect()
    if seed is None:
        return frame.head(limit).collect()

    total = frame.select(pl.len()).collect().item()
    if limit >= total:
        return frame.collect()

    chosen = pl.int_range(total, eager=True).sample(n=limit, seed=seed, shuffle=False).sort()
    # implode() makes the set-membership semantics explicit; a bare Series of
    # the same dtype is deprecated as ambiguous since polars 1.43.
    return frame.with_row_index("__row").filter(pl.col("__row").is_in(chosen.implode())).drop("__row").collect()


def _join_words(value: object) -> str | None:
    """Join word-timing structs into a plain reference transcript."""
    if not isinstance(value, list):
        return None
    words = [str(item.get("word", "")).strip() for item in value if isinstance(item, dict)]
    text = " ".join(word for word in words if word)
    return text or None


def _apply_silence_spans(clip: AudioClip, value: object) -> AudioClip:
    """Attach labeled pause spans and the labeled end of speech to a clip.

    In the eot-bench schema the *final* silence span is the true end of the
    speaker's turn, so its start is the ground-truth ``speech_end_s`` —
    strictly better than energy detection, which cannot tell a hesitation
    from the end. Every earlier span is a mid-turn pause the speaker talked
    through; an end-of-utterance decision inside one is a false cutoff.
    """
    if not isinstance(value, list) or not value:
        return clip
    spans = sorted((float(item["start"]), float(item["end"])) for item in value if isinstance(item, dict))
    if not spans:
        return clip
    return replace(clip, pauses=tuple(spans[:-1]), speech_end_s=spans[-1][0])


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


def load_clips_from_manifest(config: SourceConfig, *, sample_rate: int = 16000) -> list[AudioClip]:
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
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
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
        TtsPrompt(prompt_id=f"p{index:04d}", text=line.strip(), language=config.language)
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
        if line.strip()
    ]
    if config.limit is not None:
        prompts = prompts[: config.limit]
    if not prompts:
        raise DatasetError(f"prompt file yielded no prompts: {path}")
    return prompts
