"""Offline ingestion and preprocessing for production interview recordings.

This module is intentionally standalone. It inventories recording objects,
joins video egress metadata, cuts speech clips, performs local language
identification, and prepares transcript-free human-review selections.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
from typing import Any, Literal, cast

import orjson


REALDATA_DIR = Path("data/realdata")
INVENTORY_PATH = REALDATA_DIR / "inventory.jsonl"
JOIN_PATH = REALDATA_DIR / "join.jsonl"
CLIPS_PATH = REALDATA_DIR / "clips.jsonl"
PILOT_PATH = REALDATA_DIR / "pilot-30.jsonl"
ANCHOR_DIR = Path("data/anchors/realdata")
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

SourceTrack = Literal["user", "agent", "merged", "video"]
_JsonObject = dict[str, object]
_Segment = tuple[float, float]


@dataclass(frozen=True, slots=True)
class VideoObject:
    """A listed video object and its session grouping metadata."""

    uri: str
    size: int
    session_id: str

    @property
    def filename(self) -> str:
        """The object's basename."""
        return self.uri.rsplit("/", 1)[-1]


@dataclass(frozen=True, slots=True)
class TriageResult:
    """The chosen source track and its measured speech-time ratio."""

    source: SourceTrack
    speech_ratio: float | None


@dataclass(frozen=True, slots=True)
class ClipRecord:
    """Metadata for one generated utterance clip."""

    session_id: str
    clip_path: Path
    start_offset: float
    end_offset: float
    duration: float
    source: SourceTrack


@dataclass(frozen=True, slots=True)
class _Candidate:
    clip_path: str
    session_id: str
    language: Literal["en", "ja"]
    duration: float
    source: SourceTrack


def ingest(dest: Path, bucket_prefix: str, *, dry_run: bool = False) -> list[_JsonObject]:
    """Mirror MP3 recordings, inventory local files, and write JSON Lines.

    Args:
        dest: Local directory containing or receiving MP3 objects.
        bucket_prefix: Google Cloud Storage prefix passed to ``gcloud``.
        dry_run: Skip the network sync and inventory MP3s already under ``dest``.

    Returns:
        Inventory records in deterministic path order.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        _run_command(["gcloud", "storage", "rsync", bucket_prefix, str(dest)])

    records = [_probe_audio(path) for path in sorted(dest.rglob("*.mp3"))]
    _write_jsonl(INVENTORY_PATH, records)
    return records


def list_manifests(dest: Path, bucket_prefix: str, *, dry_run: bool = False) -> list[Path]:
    """List and download LiveKit ``EG_*.json`` manifests.

    Args:
        dest: Local video-ingest directory.
        bucket_prefix: Google Cloud Storage video prefix.
        dry_run: Read already-downloaded manifests without calling subprocesses.

    Returns:
        Local manifest paths in filename order.
    """
    manifest_dir = dest / "manifests"
    if dry_run:
        return sorted(manifest_dir.glob("EG_*.json"))

    manifest_dir.mkdir(parents=True, exist_ok=True)
    listing = _run_command(
        ["gcloud", "storage", "ls", f"{bucket_prefix.rstrip('/')}/EG_*.json"],
        capture_output=True,
    )
    uris = sorted(line.strip() for line in listing.stdout.splitlines() if line.strip().endswith(".json"))
    paths: list[Path] = []
    for uri in uris:
        local_path = manifest_dir / uri.rsplit("/", 1)[-1]
        _run_command(["gcloud", "storage", "cp", uri, str(local_path)])
        paths.append(local_path)
    return paths


def build_join(
    manifest_paths: Iterable[Path],
    *,
    output_path: Path = JOIN_PATH,
) -> list[_JsonObject]:
    """Parse egress manifests into a video-to-room join file.

    Args:
        manifest_paths: Local LiveKit egress manifests.
        output_path: JSON Lines destination.

    Returns:
        Join records sorted by video filename and room name.
    """
    records: list[_JsonObject] = []
    for manifest_path in sorted(manifest_paths):
        manifest = _read_json_object(manifest_path)
        room_name = _required_string(manifest, "room_name", manifest_path)
        egress_id = _required_string(manifest, "egress_id", manifest_path)
        started_at = _required_integer(manifest, "started_at", manifest_path)
        ended_at = _required_integer(manifest, "ended_at", manifest_path)
        files = manifest.get("files")
        if not isinstance(files, list):
            msg = f"{manifest_path}: files must be a list"
            raise TypeError(msg)
        for raw_file in files:
            file_record = _object_mapping(raw_file, f"{manifest_path}: files entry")
            raw_filename = _required_string(file_record, "filename", manifest_path)
            filename = raw_filename.rsplit("/", 1)[-1]
            records.append({
                "video_filename": filename,
                "session_id": _session_id(filename),
                "room_name": room_name,
                "egress_id": egress_id,
                "started_at": started_at,
                "ended_at": ended_at,
            })

    records.sort(key=lambda record: (str(record["video_filename"]), str(record["room_name"])))
    _write_jsonl(output_path, records)
    return records


def list_video_objects(dest: Path, bucket_prefix: str, *, dry_run: bool = False) -> list[VideoObject]:
    """List video object sizes remotely or inspect an existing local mirror.

    Args:
        dest: Local directory used when ``dry_run`` is true.
        bucket_prefix: Google Cloud Storage video prefix.
        dry_run: Inspect local MP4s without calling subprocesses.

    Returns:
        Video objects sorted by session and filename.
    """
    if dry_run:
        objects = [
            VideoObject(uri=str(path), size=path.stat().st_size, session_id=_session_id(path.name))
            for path in sorted(dest.rglob("*.mp4"))
        ]
    else:
        listing = _run_command(
            ["gcloud", "storage", "ls", "--long", f"{bucket_prefix.rstrip('/')}/*.mp4"],
            capture_output=True,
        )
        objects = _parse_video_listing(listing.stdout)
    return sorted(objects, key=lambda item: (item.session_id, item.filename, item.uri))


def dedupe_sessions(objects: Iterable[VideoObject]) -> list[VideoObject]:
    """Keep exactly one MP4 per session using object size and filename.

    The largest object is preferred because differing sizes commonly signal a
    truncated alternate. Same-size duplicates are byte-count equivalents for
    this inventory-level dedupe, so the alphabetically first filename wins.

    Args:
        objects: Listed MP4 objects with byte sizes and session identifiers.

    Returns:
        One selected object per session, ordered by session identifier.
    """
    grouped: dict[str, list[VideoObject]] = defaultdict(list)
    for item in objects:
        grouped[item.session_id].append(item)
    return [
        min(grouped[session_id], key=lambda item: (-item.size, item.filename, item.uri))
        for session_id in sorted(grouped)
    ]


def ingest_video(
    dest: Path,
    bucket_prefix: str,
    session_ids: Iterable[str] | None = None,
    *,
    full_mirror: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Join metadata, select video objects, and then download them.

    Manifest listing and download happen first, followed by join construction,
    object listing, selection, and finally video download. The default path
    copies one deduplicated MP4 per requested session; with no session filter it
    copies one per available session. ``full_mirror`` explicitly opts into a
    complete prefix mirror. ``dry_run`` performs the same local join and
    selection over files already in ``dest`` without subprocess calls.

    Args:
        dest: Local video-ingest directory.
        bucket_prefix: Google Cloud Storage video prefix.
        session_ids: Optional session identifiers to retain.
        full_mirror: Download the complete prefix instead of selected objects.
        dry_run: Select from local files without invoking subprocesses.

    Returns:
        Local paths corresponding to selected MP4 objects.
    """
    manifest_paths = list_manifests(dest, bucket_prefix, dry_run=dry_run)
    build_join(manifest_paths)
    objects = list_video_objects(dest, bucket_prefix, dry_run=dry_run)

    requested = None if session_ids is None else set(session_ids)
    if full_mirror:
        selected = objects
    else:
        selected = dedupe_sessions(objects)
        if requested is not None:
            selected = [item for item in selected if item.session_id in requested]

    local_paths = [dest / item.filename for item in selected]
    if dry_run:
        return local_paths

    dest.mkdir(parents=True, exist_ok=True)
    if full_mirror:
        _run_command(["gcloud", "storage", "rsync", bucket_prefix, str(dest)])
    else:
        for item, local_path in zip(selected, local_paths, strict=True):
            _run_command(["gcloud", "storage", "cp", item.uri, str(local_path)])
    return local_paths


def extract_video_audio(video_path: Path, audio_path: Path) -> Path:
    """Extract an MP4 audio track as 16 kHz mono PCM WAV.

    Args:
        video_path: Source MP4 recording.
        audio_path: Destination WAV path.

    Returns:
        The destination WAV path.
    """
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ])
    return audio_path


def cut_video_clips(
    video_path: Path,
    session_id: str,
    dest_dir: Path,
    *,
    metadata_path: Path = CLIPS_PATH,
) -> list[ClipRecord]:
    """Extract video audio and feed it through the shared clip cutter.

    Args:
        video_path: Source MP4 recording.
        session_id: Stable session identifier used for clip paths.
        dest_dir: Root directory for session clip directories.
        metadata_path: JSON Lines file receiving clip records.

    Returns:
        Generated video clip records.
    """
    with tempfile.TemporaryDirectory(prefix="audio-harness-realdata-") as temp_dir:
        audio_path = extract_video_audio(video_path, Path(temp_dir) / "audio.wav")
        return cut_clips(
            audio_path,
            session_id,
            dest_dir,
            source="video",
            metadata_path=metadata_path,
        )


def triage_session(
    track_path: Path | None,
    merged_path: Path,
    *,
    agent_threshold: float = 0.45,
    user_threshold: float = 0.55,
) -> TriageResult:
    """Classify a non-merged track with a local speech-time heuristic.

    ``ffmpeg`` silencedetect measures non-silent seconds in the track, which
    are divided by the merged recording's total duration. A ratio strictly
    below 0.45 is treated as the shorter agent-prompt side; a ratio strictly
    above 0.55 is treated as the longer user-answer side. The inclusive
    0.45--0.55 ambiguity band falls back to ``source="merged"``. Missing
    non-merged tracks also fall back to merged. These labels are triage hints,
    not speaker ground truth.

    Args:
        track_path: Non-merged per-track recording, when available.
        merged_path: Session-mix recording used as the duration denominator.
        agent_threshold: Exclusive upper bound for an agent classification.
        user_threshold: Exclusive lower bound for a user classification.

    Returns:
        The selected source type and measured ratio, if any.

    Raises:
        ValueError: If threshold ordering is invalid.
    """
    if not 0.0 <= agent_threshold < user_threshold <= 1.0:
        msg = "triage thresholds must satisfy 0 <= agent < user <= 1"
        raise ValueError(msg)
    if track_path is None or not track_path.is_file():
        return TriageResult(source="merged", speech_ratio=None)

    merged_duration = _probe_duration(merged_path)
    track_duration = _probe_duration(track_path)
    speech_seconds = sum(end - start for start, end in _speech_segments(track_path, track_duration))
    speech_ratio = min(max(speech_seconds / merged_duration, 0.0), 1.0)
    if speech_ratio < agent_threshold:
        source: SourceTrack = "agent"
    elif speech_ratio > user_threshold:
        source = "user"
    else:
        source = "merged"
    return TriageResult(source=source, speech_ratio=speech_ratio)


def cut_clips(
    source_path: Path,
    session_id: str,
    dest_dir: Path,
    *,
    source: SourceTrack = "merged",
    metadata_path: Path = CLIPS_PATH,
    silence_noise_db: float = -40.0,
    min_silence_s: float = 0.5,
) -> list[ClipRecord]:
    """Cut a recording into 2--30 second, 16 kHz mono utterance WAVs.

    Speech runs are the complement of silences lasting at least 0.5 seconds.
    A run under 2 seconds is merged forward across the intervening silence
    when the combined window stays at most 30 seconds, otherwise merged
    backward under the same cap, and otherwise dropped. Continuous runs over
    30 seconds split at exact 30-second boundaries. A final remainder under 2
    seconds is dropped because retaining it or merging it backward would break
    the inclusive 2--30 second contract.

    Args:
        source_path: Input audio file understood by ffmpeg.
        session_id: Stable session identifier used in the output directory.
        dest_dir: Root output directory; clips go under ``session_id``.
        source: Source-track classification written into clip metadata.
        metadata_path: JSON Lines file to append generated metadata to.
        silence_noise_db: Silencedetect noise threshold in decibels.
        min_silence_s: Minimum silence duration that separates speech runs.

    Returns:
        Generated clip records in chronological order.
    """
    duration = _probe_duration(source_path)
    segments = _speech_segments(
        source_path,
        duration,
        silence_noise_db=silence_noise_db,
        min_silence_s=min_silence_s,
    )
    bounded_segments = _split_long_segments(_merge_or_drop_short_segments(segments))
    session_dir = dest_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    records: list[ClipRecord] = []
    for index, (start, end) in enumerate(bounded_segments):
        clip_path = session_dir / f"{index:04d}.wav"
        _cut_wav(source_path, clip_path, start, end)
        records.append(
            ClipRecord(
                session_id=session_id,
                clip_path=clip_path,
                start_offset=start,
                end_offset=end,
                duration=end - start,
                source=source,
            )
        )
    _append_jsonl(metadata_path, [_clip_json(record) for record in records])
    return records


def identify_language(clip_path: Path, *, model: str = DEFAULT_WHISPER_MODEL) -> str:
    """Identify a clip as English, Japanese, or other with local Whisper.

    The optional ``mlx-whisper`` dependency is imported lazily. Whisper runs
    without a forced language argument, and only its returned ``language``
    field is used; transcript text is neither needed nor retained.

    Args:
        clip_path: Local audio clip to classify.
        model: mlx-whisper model repository or local model path.

    Returns:
        ``"en"``, ``"ja"``, or ``"other"``.
    """
    mlx_whisper = _import_mlx_whisper()
    output: object = mlx_whisper.transcribe(
        str(clip_path),
        path_or_hf_repo=model,
        temperature=0.0,
        condition_on_previous_text=False,
    )
    result = _object_mapping(output, "mlx-whisper result")
    detected = str(result.get("language", "")).lower().replace("_", "-")
    if detected == "en" or detected.startswith("en-"):
        return "en"
    if detected == "ja" or detected.startswith("ja-"):
        return "ja"
    return "other"


def select_pilot(
    n: int = 30,
    *,
    seed: int = 20260808,
    clips_path: Path = CLIPS_PATH,
    join_path: Path = JOIN_PATH,
    output_path: Path = PILOT_PATH,
) -> list[_JsonObject]:
    """Write a deterministic, stratified pilot selection without transcripts.

    Eligible clips are stratified over English/Japanese and short ``[2, 8)``,
    medium ``[8, 18)``, and long ``[18, 30]`` second buckets. Sampling cycles
    through all six strata and permits at most two clips per session. Video
    sessions are sampled first because their egress join metadata is richer;
    MP3 sessions fill any remaining slots.

    Args:
        n: Maximum number of clips to select.
        seed: Deterministic random seed.
        clips_path: Input clip metadata JSON Lines file.
        join_path: Video egress join used to recognize video sessions.
        output_path: Pilot JSON Lines destination.

    Returns:
        Selected manifest records in sampling order.

    Raises:
        ValueError: If ``n`` is negative.
    """
    selected = _select_candidates(n, seed=seed, clips_path=clips_path, join_path=join_path)
    records = [_selection_json(candidate, output_path.parent) for candidate in selected]
    _write_jsonl(output_path, records)
    return records


def select_label_candidates(
    n: int = 50,
    *,
    seed: int = 20260808,
    clips_path: Path = CLIPS_PATH,
    join_path: Path = JOIN_PATH,
    output_dir: Path = ANCHOR_DIR,
) -> list[_JsonObject]:
    """Select clips independently and write an empty-transcript review kit.

    Selection uses the same language, duration, per-session, and video-first
    policy as the pilot, but runs independently and may overlap it. The review
    sheet contains no provider axis and leaves ``true_transcript`` empty for a
    human labeler. Both generated text files use explicit LF line endings.

    Args:
        n: Maximum number of clips to select for labeling.
        seed: Deterministic random seed.
        clips_path: Input clip metadata JSON Lines file.
        join_path: Video egress join used to recognize video sessions.
        output_dir: Directory receiving the CSV and README.

    Returns:
        Selected manifest records in review-sheet order.

    Raises:
        ValueError: If ``n`` is negative.
    """
    selected = _select_candidates(n, seed=seed, clips_path=clips_path, join_path=join_path)
    records = [_selection_json(candidate, output_dir) for candidate in selected]
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / "anchor-review-sheet.csv"
    with sheet_path.open("w", encoding="utf-8", newline="") as sheet:
        writer = csv.writer(sheet, lineterminator="\n")
        writer.writerow(["n", "clip_path", "session_id", "language", "duration", "true_transcript"])
        for index, candidate in enumerate(selected, start=1):
            writer.writerow([
                index,
                candidate.clip_path,
                candidate.session_id,
                candidate.language,
                f"{candidate.duration:.3f}",
                "",
            ])

    readme = _anchor_readme(len(selected), seed)
    (output_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    return records


def _run_command(arguments: Sequence[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, check=True, capture_output=capture_output, text=True)


def _probe_audio(path: Path) -> _JsonObject:
    probe = _ffprobe(path)
    stream = _first_stream(probe, path)
    audio_format = _object_mapping(probe.get("format"), f"{path}: format")
    return {
        "path": path.as_posix(),
        "duration": _required_float(audio_format, "duration", path),
        "channels": _required_integer(stream, "channels", path),
        "bit_rate": _optional_integer(stream.get("bit_rate") or audio_format.get("bit_rate")),
        "sample_rate": _required_integer(stream, "sample_rate", path),
    }


def _probe_duration(path: Path) -> float:
    probe = _ffprobe(path)
    audio_format = _object_mapping(probe.get("format"), f"{path}: format")
    duration = _required_float(audio_format, "duration", path)
    if duration <= 0.0:
        msg = f"{path}: duration must be positive"
        raise ValueError(msg)
    return duration


def _ffprobe(path: Path) -> _JsonObject:
    completed = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=duration,bit_rate:stream=channels,sample_rate,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
    )
    payload: object = orjson.loads(completed.stdout)
    return _object_mapping(payload, f"{path}: ffprobe result")


def _first_stream(probe: _JsonObject, path: Path) -> _JsonObject:
    streams = probe.get("streams")
    if not isinstance(streams, list) or not streams:
        msg = f"{path}: ffprobe returned no audio stream"
        raise ValueError(msg)
    return _object_mapping(streams[0], f"{path}: audio stream")


def _read_json_object(path: Path) -> _JsonObject:
    payload: object = orjson.loads(path.read_bytes())
    return _object_mapping(payload, str(path))


def _object_mapping(value: object, context: str) -> _JsonObject:
    if not isinstance(value, dict):
        msg = f"{context}: expected a JSON object"
        raise TypeError(msg)
    return cast("_JsonObject", value)


def _required_string(record: _JsonObject, key: str, path: Path) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{path}: {key} must be a non-empty string"
        raise ValueError(msg)
    return value


def _required_integer(record: _JsonObject, key: str, path: Path) -> int:
    value = record.get(key)
    try:
        return int(cast("str | int", value))
    except (TypeError, ValueError) as exc:
        msg = f"{path}: {key} must be an integer"
        raise ValueError(msg) from exc


def _required_float(record: _JsonObject, key: str, path: Path) -> float:
    value = record.get(key)
    try:
        return float(cast("str | int | float", value))
    except (TypeError, ValueError) as exc:
        msg = f"{path}: {key} must be numeric"
        raise ValueError(msg) from exc


def _optional_integer(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(cast("str | int", value))
    except (TypeError, ValueError) as exc:
        msg = f"expected an integer, got {value!r}"
        raise ValueError(msg) from exc


def _write_jsonl(path: Path, records: Iterable[_JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        for record in records:
            output.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))


def _append_jsonl(path: Path, records: Iterable[_JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as output:
        for record in records:
            output.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))


def _session_id(filename: str) -> str:
    return Path(filename).stem.removesuffix("_merged")


def _parse_video_listing(output: str) -> list[VideoObject]:
    objects: list[VideoObject] = []
    for line in output.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        uri = next((part for part in parts if part.startswith("gs://") and part.endswith(".mp4")), None)
        if uri is None:
            continue
        filename = uri.rsplit("/", 1)[-1]
        objects.append(VideoObject(uri=uri, size=int(parts[0]), session_id=_session_id(filename)))
    return objects


def _detect_silences(
    source_path: Path,
    duration: float,
    *,
    silence_noise_db: float,
    min_silence_s: float,
) -> list[_Segment]:
    completed = _run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(source_path),
            "-af",
            f"silencedetect=noise={silence_noise_db:g}dB:d={min_silence_s:g}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
    )
    events = re.findall(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)", completed.stderr)
    silences: list[_Segment] = []
    active_start: float | None = None
    for event, raw_offset in events:
        offset = min(max(float(raw_offset), 0.0), duration)
        if event == "start":
            active_start = offset
        elif active_start is None:
            silences.append((0.0, offset))
        else:
            silences.append((active_start, offset))
            active_start = None
    if active_start is not None:
        silences.append((active_start, duration))
    return silences


def _speech_segments(
    source_path: Path,
    duration: float,
    *,
    silence_noise_db: float = -40.0,
    min_silence_s: float = 0.5,
) -> list[_Segment]:
    silences = _detect_silences(
        source_path,
        duration,
        silence_noise_db=silence_noise_db,
        min_silence_s=min_silence_s,
    )
    speech: list[_Segment] = []
    cursor = 0.0
    for silence_start, silence_end in silences:
        if silence_start > cursor:
            speech.append((cursor, silence_start))
        cursor = max(cursor, silence_end)
    if cursor < duration:
        speech.append((cursor, duration))
    return [(start, end) for start, end in speech if end > start]


def _merge_or_drop_short_segments(segments: list[_Segment]) -> list[_Segment]:
    pending = list(segments)
    merged: list[_Segment] = []
    index = 0
    while index < len(pending):
        start, end = pending[index]
        if end - start >= 2.0:
            merged.append((start, end))
        elif index + 1 < len(pending) and pending[index + 1][1] - start <= 30.0:
            pending[index + 1] = (start, pending[index + 1][1])
        elif merged and end - merged[-1][0] <= 30.0:
            merged[-1] = (merged[-1][0], end)
        index += 1
    return merged


def _split_long_segments(segments: Iterable[_Segment]) -> list[_Segment]:
    bounded: list[_Segment] = []
    for original_start, end in segments:
        start = original_start
        while end - start > 30.0:
            bounded.append((start, start + 30.0))
            start += 30.0
        if end - start >= 2.0:
            bounded.append((start, end))
    return bounded


def _cut_wav(source_path: Path, clip_path: Path, start: float, end: float) -> None:
    _run_command([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{end - start:.6f}",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(clip_path),
    ])


def _clip_json(record: ClipRecord) -> _JsonObject:
    return {
        "session_id": record.session_id,
        "clip_path": record.clip_path.as_posix(),
        "start_offset": record.start_offset,
        "end_offset": record.end_offset,
        "duration": record.duration,
        "source": record.source,
    }


def _import_mlx_whisper() -> Any:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "realdata language identification: mlx-whisper is not installed. "
            "Install the optional dependency group: uv sync --extra judge-whisper"
        ) from exc
    return mlx_whisper


def _load_candidates(clips_path: Path) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    if not clips_path.is_file():
        return candidates
    for line_number, line in enumerate(clips_path.read_bytes().splitlines(), start=1):
        raw: object = orjson.loads(line)
        record = _object_mapping(raw, f"{clips_path}:{line_number}")
        language = record.get("language")
        if language not in {"en", "ja"}:
            continue
        duration = _candidate_duration(record, clips_path, line_number)
        if not 2.0 <= duration <= 30.0:
            continue
        source = record.get("source", "merged")
        if source not in {"user", "agent", "merged", "video"}:
            msg = f"{clips_path}:{line_number}: invalid source {source!r}"
            raise ValueError(msg)
        candidates.append(
            _Candidate(
                clip_path=_candidate_string(record, "clip_path", clips_path, line_number),
                session_id=_candidate_string(record, "session_id", clips_path, line_number),
                language=cast("Literal['en', 'ja']", language),
                duration=duration,
                source=cast("SourceTrack", source),
            )
        )
    return candidates


def _candidate_duration(record: _JsonObject, path: Path, line_number: int) -> float:
    value = record.get("duration")
    try:
        return float(cast("str | int | float", value))
    except (TypeError, ValueError) as exc:
        msg = f"{path}:{line_number}: duration must be numeric"
        raise ValueError(msg) from exc


def _candidate_string(record: _JsonObject, key: str, path: Path, line_number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{path}:{line_number}: {key} must be a non-empty string"
        raise ValueError(msg)
    return value


def _video_sessions(join_path: Path) -> set[str]:
    sessions: set[str] = set()
    if not join_path.is_file():
        return sessions
    for line_number, line in enumerate(join_path.read_bytes().splitlines(), start=1):
        raw: object = orjson.loads(line)
        record = _object_mapping(raw, f"{join_path}:{line_number}")
        for key in ("session_id", "room_name"):
            value = record.get(key)
            if isinstance(value, str) and value:
                sessions.add(value)
    return sessions


def _select_candidates(n: int, *, seed: int, clips_path: Path, join_path: Path) -> list[_Candidate]:
    if n < 0:
        msg = "selection size must be non-negative"
        raise ValueError(msg)
    candidates = _load_candidates(clips_path)
    video_sessions = _video_sessions(join_path)
    video = [item for item in candidates if item.source == "video" or item.session_id in video_sessions]
    mp3 = [item for item in candidates if item not in video]
    rng = random.Random(seed)
    session_counts: dict[str, int] = defaultdict(int)
    selected = _stratified_take(video, n, rng=rng, session_counts=session_counts)
    if len(selected) < n:
        selected.extend(
            _stratified_take(
                mp3,
                n - len(selected),
                rng=rng,
                session_counts=session_counts,
            )
        )
    return selected


def _stratified_take(
    candidates: Iterable[_Candidate],
    limit: int,
    *,
    rng: random.Random,
    session_counts: dict[str, int],
) -> list[_Candidate]:
    strata: dict[tuple[str, str], list[_Candidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda item: (item.session_id, item.clip_path)):
        strata[candidate.language, _duration_bucket(candidate.duration)].append(candidate)
    for bucket in strata.values():
        rng.shuffle(bucket)

    order = [(language, bucket) for bucket in ("short", "medium", "long") for language in ("en", "ja")]
    selected: list[_Candidate] = []
    while len(selected) < limit:
        progress = False
        for key in order:
            bucket = strata[key]
            while bucket:
                candidate = bucket.pop()
                if session_counts[candidate.session_id] >= 2:
                    continue
                selected.append(candidate)
                session_counts[candidate.session_id] += 1
                progress = True
                break
            if len(selected) == limit:
                break
        if not progress:
            break
    return selected


def _duration_bucket(duration: float) -> str:
    if duration < 8.0:
        return "short"
    if duration < 18.0:
        return "medium"
    return "long"


def _selection_json(candidate: _Candidate, base_dir: Path) -> _JsonObject:
    # The pilot manifest is consumed by the reference-free dataset loader,
    # which resolves relative clip paths against the manifest's directory.
    clip = os.path.relpath(Path(candidate.clip_path).resolve(), base_dir.resolve())
    return {
        "clip": Path(clip).as_posix(),
        "session": candidate.session_id,
        "language": candidate.language,
        "duration_s": candidate.duration,
        "source": candidate.source,
    }


def _anchor_readme(selected_count: int, seed: int) -> str:
    return f"""# Real-recording transcript anchor ({selected_count} items)

This kit prepares offline interview clips for independent human transcription. No
provider transcript exists at this stage, so the review sheet contains only clip
identity and recording metadata plus one empty `true_transcript` field.

## Files

- `anchor-review-sheet.csv` — replay each local clip identified by row number, path,
  session, language, and duration; enter the verbatim transcript in `true_transcript`.
- `README.md` — this labeling protocol.

## Protocol

- Replay the referenced audio locally; do not upload production audio or identifiers.
- Put exactly one verbatim human transcript in `true_transcript` for every completed row.
- Preserve hesitations, repetitions, numbers, names, and code-switching as spoken.
- Leave a row empty only while it is genuinely unreviewed; do not invent placeholder text.
- Selection is stratified over English/Japanese and 2--30 second duration buckets,
  limited to two clips per session, video-first, with seed {seed}.
"""
