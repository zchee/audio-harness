#!/usr/bin/env python3
"""Fetch the "general" sentence layer of the TTS prompt suite.

Downloads plain-text sentence files from
``github.com/common-voice/common-voice`` (``server/data/<locale>/*.txt``,
CC0-1.0) — text only, no audio. Selection logic lives in
``audio_harness.common_voice_prompts``; this script is the network shell.
See that module's docstring for why this source was chosen over the
Hugging Face Common Voice dataset (moved off HF in October 2025) and its
alternatives.

Usage:

    uv run python tools/gen_common_voice_prompts.py
    uv run python tools/gen_common_voice_prompts.py --locales en ja --count 50
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import httpx

from audio_harness.common_voice_prompts import (
    CATEGORY,
    LICENSE,
    build_prompts,
    filter_sentences,
    sample_sentences,
)
from audio_harness.prompt_suite import (
    LOCALES,
    SuitePrompt,
    flatten_to_prompts_txt,
    write_jsonl,
)


REPO = "common-voice/common-voice"
DATA_PATH = "server/data"
API_BASE = f"https://api.github.com/repos/{REPO}/contents/{DATA_PATH}"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/main/{DATA_PATH}"

DEFAULT_SEED = 20260806
DEFAULT_COUNT = 100

LICENSE_NOTE = """\
# Common Voice sentence corpus — {language}

- Source: https://github.com/{repo}/tree/main/{data_path}/{subtag}
- License: {license} — see {data_path}/LICENSE in that repo
- Retrieved: {retrieved} by tools/gen_common_voice_prompts.py
- Source files: {files}
- Sentences selected: {selected} of {candidates} candidates (after length
  filtering; {min_or_max} band)

These are Common Voice's own collection prompt sentences, not transcribed
speech — no audio is fetched or redistributed here.
"""


def list_locale_files(client: httpx.Client, subtag: str) -> list[str]:
    """List the sentence ``.txt`` files GitHub holds for one locale.

    Args:
        client: HTTP client to fetch through.
        subtag: Bare Common Voice locale directory name (bare BCP-47
            primary subtag for every locale this harness targets).

    Returns:
        File names, excluding ``LICENSE`` and any non-``.txt`` entries.

    Raises:
        httpx.HTTPStatusError: If the locale has no ``server/data`` directory
            or the request otherwise fails.
    """
    response = client.get(f"{API_BASE}/{subtag}")
    response.raise_for_status()
    entries = response.json()
    return sorted(entry["name"] for entry in entries if entry["type"] == "file" and entry["name"].endswith(".txt"))


def fetch_lines(client: httpx.Client, subtag: str, filename: str) -> list[str]:
    """Download one sentence file and split it into lines."""
    response = client.get(f"{RAW_BASE}/{subtag}/{filename}")
    response.raise_for_status()
    return response.text.splitlines()


def generate_locale(
    client: httpx.Client, subtag: str, *, count: int, seed: int
) -> tuple[list[SuitePrompt], list[str], int]:
    """Fetch, filter and sample one locale's general-sentence layer.

    Returns:
        The selected prompts, the source file names used, and the number of
        candidates that survived filtering before sampling.
    """
    files = list_locale_files(client, subtag)
    lines: list[str] = []
    for filename in files:
        lines.extend(fetch_lines(client, subtag, filename))

    candidates = filter_sentences(lines, language=LOCALES[subtag])
    sentences = sample_sentences(candidates, count=count, seed=seed)
    prompts = build_prompts(subtag, sentences, files)
    return prompts, files, len(candidates)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="prompt suite root (default: data)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"sentences per locale (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed for reproducible sampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--locales",
        nargs="*",
        default=None,
        help="bare subtags to fetch (default: all 14 harness locales)",
    )
    args = parser.parse_args()

    subtags = args.locales or sorted(LOCALES)
    unknown = sorted(set(subtags) - set(LOCALES))
    if unknown:
        parser.error(f"unknown locale(s): {', '.join(unknown)}")

    retrieved = datetime.now(UTC).date().isoformat()
    total = 0
    with httpx.Client(timeout=30.0, headers={"User-Agent": "audio-harness/gen_common_voice_prompts"}) as client:
        for subtag in subtags:
            try:
                prompts, files, candidate_count = generate_locale(client, subtag, count=args.count, seed=args.seed)
            except httpx.HTTPStatusError as exc:
                print(f"skip {subtag}: {exc}")
                continue

            out_dir = args.out_dir / f"prompts-{subtag}"
            jsonl_path = write_jsonl(out_dir / f"{CATEGORY}.jsonl", prompts)
            flatten_to_prompts_txt(prompts, out_dir / f"{CATEGORY}.txt")
            (out_dir / f"{CATEGORY}.LICENSE.md").write_text(
                LICENSE_NOTE.format(
                    language=LOCALES[subtag],
                    repo=REPO,
                    data_path=DATA_PATH,
                    subtag=subtag,
                    license=LICENSE,
                    retrieved=retrieved,
                    files=", ".join(files),
                    selected=len(prompts),
                    candidates=candidate_count,
                    min_or_max="4-40 words" if subtag != "ja" else "8-100 characters",
                ),
                encoding="utf-8",
            )
            total += len(prompts)
            print(
                f"wrote {jsonl_path} ({len(prompts)} sentences from {len(files)} file(s), {candidate_count} candidates)"
            )

    print(f"total: {total} sentences across {len(subtags)} locale(s)")


if __name__ == "__main__":
    main()
