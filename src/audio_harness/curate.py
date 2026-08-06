"""Curation of commercial-safe multilingual STT evaluation sets.

Speech-MASSIVE is CC BY-NC-SA — internal regression only — so the public
multilingual lane needs commercially usable material: YODAS2 (CC BY 3.0,
YouTube manual subtitles) for all twelve languages and NVIDIA Granary
(CC BY 4.0 manifests) for the eight EU languages. Two rules shape everything
here:

* **Audio is never redistributed.** Manifests carry identifiers, offsets and
  transcripts; audio stays wherever the upstream host keeps it and is
  fetched at benchmark time. That is what keeps the CC BY caption license
  from being stretched over audio rights it does not cover.
* **No transcript is gold until a human says so.** Subtitles are approximate
  by nature; a language's transcripts only count as gold once a
  human-verified sample reaches subtitle-vs-audio CER < 5%. Every manifest
  row therefore records ``gold_status: unverified`` until that review
  happens — a benchmark that silently promotes subtitles to ground truth is
  measuring subtitle quality, not recognition.

Selection is heuristic and honest about it: "interview/podcast style" is
approximated by preferring long-form videos with many mid-length utterances,
which biases toward talk content and away from music and shorts. The
heuristics live in this module as pure functions so they are testable
without the network; ``tools/curate_yodas.py`` does the fetching.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

import orjson

YODAS_LICENSE = "CC-BY-3.0"
GRANARY_LICENSE = "CC-BY-4.0"

MIN_UTTERANCE_S = 3.0
MAX_UTTERANCE_S = 30.0
"""Interview turns live in this band; shorter is interjection, longer is
usually caption drift across a scene cut."""

MIN_TEXT_CHARS = 12
MIN_VIDEO_UTTERANCES = 30
"""A video with dozens of aligned utterances is long-form talk content;
music videos and shorts rarely clear this bar."""

MAX_PER_VIDEO = 3
"""Diversity cap: thirty clips from one podcast episode would measure one
speaker's microphone, not a language."""

_UTT_KEY = re.compile(r"^(?P<video>.+)-\d{5}-(?P<start>\d{8})-(?P<end>\d{8})$")
_REJECT_TEXT = re.compile(r"[\[\]♪♫]|https?://")


@dataclass(slots=True, frozen=True)
class Candidate:
    """One curated utterance, before manifest serialization.

    Attributes:
        source: Corpus the clip comes from (``yodas2`` or ``granary``).
        subset: Corpus subdivision (YODAS2 subset or Granary source set).
        shard: File within the subset holding the audio.
        video_id: Upstream recording identifier (YouTube video, session id).
        utt_id: Utterance identifier within the recording.
        language: BCP-47 tag.
        start_s: Utterance onset within the recording, seconds.
        end_s: Utterance end within the recording, seconds.
        text: Subtitle transcript, unverified.
    """

    source: str
    subset: str
    shard: str
    video_id: str
    utt_id: str
    language: str
    start_s: float
    end_s: float
    text: str


def parse_yodas_text_shard(
    payload: list[dict[str, object]], *, subset: str, shard: str, language: str
) -> list[Candidate]:
    """Extract filtered candidates from one YODAS2 ``text/*.json`` shard.

    Utterance keys encode timing as ``<video>-<index>-<start>-<end>`` in
    centiseconds, so transcripts and offsets come from the text shard alone —
    the audio tarball is never touched during curation.

    Args:
        payload: Parsed JSON shard: one entry per video.
        subset: YODAS2 subset name (``de000``).
        shard: Shard stem (``00000000``), which also names the audio tarball.
        language: BCP-47 tag recorded on every candidate.

    Returns:
        Candidates passing the utterance filters, in shard order.
    """
    candidates: list[Candidate] = []
    for entry in payload:
        video_id = str(entry.get("audio_id") or "")
        utterances = entry.get("text")
        if not video_id or not isinstance(utterances, dict):
            continue
        if len(utterances) < MIN_VIDEO_UTTERANCES:
            continue
        for key, text in utterances.items():
            match = _UTT_KEY.match(str(key))
            if match is None or not isinstance(text, str):
                continue
            start_s = int(match.group("start")) / 100.0
            end_s = int(match.group("end")) / 100.0
            if not _keep_utterance(text, end_s - start_s):
                continue
            candidates.append(
                Candidate(
                    source="yodas2",
                    subset=subset,
                    shard=shard,
                    video_id=video_id,
                    utt_id=str(key),
                    language=language,
                    start_s=start_s,
                    end_s=end_s,
                    text=text.strip(),
                )
            )
    return candidates


def parse_granary_lines(
    lines: list[str], *, subset: str, language: str
) -> list[Candidate]:
    """Extract filtered candidates from Granary manifest JSONL lines.

    Granary rows reference audio by ``utt_id``/``original_source_id``; the
    wav paths they name are reconstructed from upstream sources at fetch
    time, so the manifest is the whole corpus as far as curation goes.

    Args:
        lines: Raw JSONL lines (a bounded slice of the file is fine; a
            trailing partial line is skipped).
        subset: Granary source set (``ytc``, ``voxpopuli``).
        language: BCP-47 tag recorded on every candidate.

    Returns:
        Candidates passing the utterance filters, in input order.
    """
    candidates: list[Candidate] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        text = row.get("text")
        duration = row.get("duration")
        utt_id = row.get("utt_id")
        if not (
            isinstance(text, str)
            and isinstance(duration, int | float)
            and isinstance(utt_id, str)
        ):
            continue
        if not _keep_utterance(text, float(duration)):
            continue
        candidates.append(
            Candidate(
                source="granary",
                subset=subset,
                shard=str(row.get("dataset_source") or subset),
                video_id=str(row.get("original_source_id") or ""),
                utt_id=utt_id,
                language=language,
                start_s=0.0,
                end_s=float(duration),
                text=text.strip(),
            )
        )
    return candidates


def _keep_utterance(text: str, duration_s: float) -> bool:
    """Whether one utterance is usable evaluation material."""
    if not MIN_UTTERANCE_S <= duration_s <= MAX_UTTERANCE_S:
        return False
    stripped = text.strip()
    if len(stripped) < MIN_TEXT_CHARS:
        return False
    return not _REJECT_TEXT.search(stripped)


def sample_candidates(
    candidates: list[Candidate], *, count: int, seed: int
) -> list[Candidate]:
    """Draw a reproducible, speaker-diverse sample.

    At most :data:`MAX_PER_VIDEO` utterances per recording, chosen with a
    seeded shuffle so re-running curation yields the identical manifest —
    a moving evaluation set would make every run incomparable to the last.

    Args:
        candidates: Filtered candidates.
        count: Target sample size.
        seed: RNG seed.

    Returns:
        Up to ``count`` candidates in stable (video, utterance) order.
    """
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    per_video: dict[str, int] = {}
    chosen: list[Candidate] = []
    for candidate in shuffled:
        if per_video.get(candidate.video_id, 0) >= MAX_PER_VIDEO:
            continue
        per_video[candidate.video_id] = per_video.get(candidate.video_id, 0) + 1
        chosen.append(candidate)
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda c: (c.video_id, c.utt_id))


def manifest_row(candidate: Candidate) -> dict[str, object]:
    """Serialize one candidate as a manifest record.

    The license rides on every row rather than on the file so that merged
    manifests keep per-source attribution, and ``gold_status`` starts as
    ``unverified`` — only the human CER<5% review may change it.
    """
    row = asdict(candidate)
    row["duration_s"] = round(candidate.end_s - candidate.start_s, 2)
    row["license"] = YODAS_LICENSE if candidate.source == "yodas2" else GRANARY_LICENSE
    row["gold_status"] = "unverified"
    return row
