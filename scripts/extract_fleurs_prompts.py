#!/usr/bin/env python3
"""Extract multilingual TTS guardrail prompts from FLEURS transcripts.

Fetches the ``test.tsv`` transcript files of ``google/fleurs`` (CC-BY-4.0)
straight from the Hugging Face hub — text only, no audio — and samples a
fixed number of sentences per language into ``data/prompts-multilang/``.
This mirrors the repository's FLEURS usage (configs/stt-fleurs-ja.yaml reads
the same dataset); only transcripts are needed here, so the TSVs are fetched
directly instead of the multi-gigabyte audio parquets.

The language set is the 12 Speech-MASSIVE languages of
``configs/stt-speech-massive-internal.yaml``. FLEURS does not publish every
regional variant, so three languages map to a different region than the
Speech-MASSIVE tag (see ``LOCALES``): Arabic ``ar-SA`` -> ``ar_eg``, Spanish
``es-ES`` -> ``es_419``, Portuguese ``pt-PT`` -> ``pt_br``. The sampled text
is the ``raw_transcription`` column, which keeps original punctuation — the
right input for a TTS engine.

Selection is deterministic: sentences are deduplicated (FLEURS records
several speakers per sentence), length-filtered to the 60-120 character
band, sorted, and sampled with ``random.Random(seed)``. The canonical seed
is 20260812; the generated README records the seed and locale mapping.

Usage:

    uv run python scripts/extract_fleurs_prompts.py
    uv run python scripts/extract_fleurs_prompts.py --languages ar de --count 10
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import random

import httpx
from rich.console import Console


console = Console(markup=False, highlight=False)

RAW_BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"

DEFAULT_SEED = 20260812
DEFAULT_COUNT = 10
DEFAULT_MIN_CHARS = 60
DEFAULT_MAX_CHARS = 120

LOCALES: dict[str, str] = {
    # harness language -> FLEURS locale. Keys are the two-letter prefixes of
    # the 12 Speech-MASSIVE languages (configs/stt-speech-massive-internal.yaml).
    "ar": "ar_eg",  # Speech-MASSIVE is ar-SA; FLEURS Arabic is Egyptian
    "de": "de_de",
    "es": "es_419",  # Speech-MASSIVE is es-ES; FLEURS Spanish is Latin American
    "fr": "fr_fr",
    "hu": "hu_hu",
    "ko": "ko_kr",
    "nl": "nl_nl",
    "pl": "pl_pl",
    "pt": "pt_br",  # Speech-MASSIVE is pt-PT; FLEURS Portuguese is Brazilian
    "ru": "ru_ru",
    "tr": "tr_tr",
    "vi": "vi_vn",
}

README_TEMPLATE = """\
# FLEURS multilingual TTS guardrail prompts

- Source: https://huggingface.co/datasets/google/fleurs (``data/<locale>/test.tsv``)
- License: CC-BY-4.0 (FLEURS; Conneau et al., arXiv:2205.12446)
- Retrieved: {retrieved} by scripts/extract_fleurs_prompts.py
- Seed: {seed} — sentences deduplicated, length-filtered to
  {min_chars}-{max_chars} characters, sorted, then sampled with
  ``random.Random(seed)``; {count} per language
- Text column: ``raw_transcription`` (original punctuation kept for TTS)

Language set: the 12 Speech-MASSIVE languages of
configs/stt-speech-massive-internal.yaml. FLEURS regional variants differ
for three of them, so the prompt text region does not always match the
Speech-MASSIVE tag:

| file | FLEURS locale | Speech-MASSIVE tag | region note |
|------|---------------|--------------------|-------------|
{rows}

Regenerate with:

    uv run python scripts/extract_fleurs_prompts.py
"""


def fetch_sentences(client: httpx.Client, locale: str) -> list[str]:
    """Return the unique ``raw_transcription`` sentences of one test split."""
    response = client.get(f"{RAW_BASE}/{locale}/test.tsv")
    response.raise_for_status()
    sentences: set[str] = set()
    for line in response.text.splitlines():
        fields = line.split("\t")
        # id, filename, raw_transcription, transcription, ...
        if len(fields) < 4:
            continue
        sentence = _unquote(fields[2].strip())
        if sentence:
            sentences.add(sentence)
    return sorted(sentences)


def _unquote(sentence: str) -> str:
    """Undo the CSV-style quoting some FLEURS TSV rows carry.

    A minority of rows arrive as ``"text with ""inner"" quotes"``; fed to a
    TTS engine verbatim, the doubled quotes would be spoken as punctuation
    noise.
    """
    if len(sentence) >= 2 and sentence.startswith('"') and sentence.endswith('"'):
        return sentence[1:-1].replace('""', '"').strip()
    return sentence


def select_prompts(sentences: list[str], *, count: int, seed: int, min_chars: int, max_chars: int) -> list[str]:
    """Deterministically sample length-banded prompts from one language."""
    candidates = [sentence for sentence in sentences if min_chars <= len(sentence) <= max_chars]
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} candidates in the {min_chars}-{max_chars} band, need {count}")
    return random.Random(seed).sample(candidates, count)


def region_note(language: str) -> str:
    """Describe the region mismatch between FLEURS and Speech-MASSIVE, if any."""
    notes = {
        "ar": "FLEURS Arabic is Egyptian, not Saudi",
        "es": "FLEURS Spanish is Latin American, not Castilian",
        "pt": "FLEURS Portuguese is Brazilian, not European",
    }
    return notes.get(language, "same region")


def massive_tag(language: str) -> str:
    """Return the Speech-MASSIVE BCP-47 tag for one two-letter language."""
    tags = {
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
    return tags[language]


def main() -> None:
    """Fetch, sample and write the per-language prompt files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="*", default=sorted(LOCALES), choices=sorted(LOCALES))
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--out", type=Path, default=Path("data/prompts-multilang"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        for language in args.languages:
            locale = LOCALES[language]
            sentences = fetch_sentences(client, locale)
            prompts = select_prompts(
                sentences,
                count=args.count,
                seed=args.seed,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )
            path = args.out / f"{language}.txt"
            path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
            rows.append(f"| {language}.txt | {locale} | {massive_tag(language)} | {region_note(language)} |")
            console.print(f"{path}: {len(prompts)} prompts from {len(sentences)} unique sentences ({locale})")

    if sorted(args.languages) == sorted(LOCALES):
        readme = README_TEMPLATE.format(
            retrieved=datetime.now(UTC).date().isoformat(),
            seed=args.seed,
            count=args.count,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            rows="\n".join(rows),
        )
        (args.out / "README.md").write_text(readme, encoding="utf-8")
        console.print(f"{args.out / 'README.md'}: locale mapping and seed recorded")


if __name__ == "__main__":
    main()
