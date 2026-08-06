"""Shared file format for the TTS prompt suite (interview / general / entities).

The suite has three commercially-clean layers, each written per locale under
``data/prompts-<lang>/``:

* ``interview`` — self-generated interview-agent questions (own IP).
* ``general`` — long-form sentences sampled from a CC0 corpus.
* ``entities`` — templated hard cases (numbers, dates, currency, IDs, names)
  sharing :data:`audio_harness.entities.ENTITY_CLASSES`.

Each layer is written twice, for two different consumers:

* ``<category>.jsonl`` is the source of truth — one :class:`SuitePrompt` per
  line, carrying language, license and provenance metadata, plus the
  entity-tagged form for categories that have one. This is what any future
  scoring or curation tooling should read.
* ``<category>.txt`` is that same layer flattened to plain speakable text,
  one prompt per line with no metadata — the exact shape
  :func:`audio_harness.dataset.load_prompts` already reads, so a benchmark
  config's ``dataset.prompts:`` can point straight at it today with no
  changes to ``dataset.py``.

Locale directories use the bare BCP-47 primary subtag (``prompts-en``,
``prompts-ko``), matching the existing ``data/prompts-en.txt`` /
``data/prompts-ja.txt`` naming; :data:`LOCALES` records the full region-
qualified tag each one carries on its :class:`SuitePrompt` records.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import orjson

CATEGORIES = ("interview", "general", "entities")
"""The three prompt-suite layers, in the order the plan introduces them."""

LOCALES: dict[str, str] = {
    "en": "en-US",
    "ja": "ja-JP",
    "ar": "ar-SA",
    "de": "de-DE",
    "es": "es-ES",
    "fr": "fr-FR",
    "hu": "hu-HU",
    "ko": "ko-KR",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "tr": "tr-TR",
    "vi": "vi-VN",
}
"""The 14 locales the harness covers: en/ja (pipecat/FLEURS) plus the 12
Speech-MASSIVE languages, keyed by bare primary subtag -> the exact
region-qualified BCP-47 tag used throughout ``configs/*.yaml``."""


@dataclass(slots=True, frozen=True)
class SuitePrompt:
    """One prompt-suite record.

    Attributes:
        prompt_id: Stable identifier, unique within its category+language
            file.
        text: Plain speakable text — what a TTS engine should be given.
        language: Full BCP-47 tag, e.g. ``ko-KR``.
        category: One of :data:`CATEGORIES`.
        license: SPDX-style identifier (e.g. ``CC0-1.0``), or ``own-IP`` for
            hand-authored content.
        source: Provenance string — corpus name, generator name, or author.
        annotated: Entity-tagged form of ``text`` in the
            :mod:`audio_harness.entities` tag vocabulary, or ``None`` for
            categories that carry no entity tags.
        entity_class: Which :data:`audio_harness.entities.ENTITY_CLASSES`
            this hard case targets, or ``None`` outside the ``entities``
            category.
    """

    prompt_id: str
    text: str
    language: str
    category: str
    license: str
    source: str
    annotated: str | None = None
    entity_class: str | None = None


def write_jsonl(path: str | Path, prompts: Iterable[SuitePrompt]) -> Path:
    """Write suite prompts as JSONL, the layer's source of truth.

    Args:
        path: Destination file; parent directories are created if missing.
        prompts: Records to write, one per line in iteration order.

    Returns:
        ``path``, for chaining.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("wb") as handle:
        for prompt in prompts:
            handle.write(orjson.dumps(asdict(prompt)))
            handle.write(b"\n")
    return file


def read_jsonl(path: str | Path) -> list[SuitePrompt]:
    """Read suite prompts back from a JSONL file written by :func:`write_jsonl`.

    Args:
        path: File to read.

    Returns:
        Records in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is not valid JSON.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"prompt suite file not found: {file}")

    prompts: list[SuitePrompt] = []
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            raise ValueError(f"{file}:{number}: invalid JSON") from exc
        prompts.append(
            SuitePrompt(
                prompt_id=record["prompt_id"],
                text=record["text"],
                language=record["language"],
                category=record["category"],
                license=record["license"],
                source=record["source"],
                annotated=record.get("annotated"),
                entity_class=record.get("entity_class"),
            )
        )
    return prompts


def flatten_to_prompts_txt(prompts: Iterable[SuitePrompt], path: str | Path) -> Path:
    """Flatten suite prompts to the plain per-line text ``load_prompts`` reads.

    Blank or whitespace-only text is skipped: ``load_prompts`` already drops
    blank lines, and writing them here would just make the two line counts
    disagree for no benefit.

    Args:
        prompts: Records to flatten, in the order they should be spoken.
        path: Destination ``.txt`` file; parent directories are created if
            missing.

    Returns:
        ``path``, for chaining.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    lines = [prompt.text.strip() for prompt in prompts if prompt.text.strip()]
    file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return file
