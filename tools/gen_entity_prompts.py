#!/usr/bin/env python3
"""Write the entity hard-case layer of the TTS prompt suite.

Generation is pure and network-free; num2words/Babel/Faker/template logic
lives in ``audio_harness.entity_prompts``, this script is the CLI shell.

Usage:

    uv run python tools/gen_entity_prompts.py
    uv run python tools/gen_entity_prompts.py --count-per-class 10 --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_harness.entity_prompts import CATEGORY, generate_locale
from audio_harness.prompt_suite import LOCALES, flatten_to_prompts_txt, write_jsonl

DEFAULT_SEED = 20260806
DEFAULT_COUNT_PER_CLASS = 6


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
        "--count-per-class",
        type=int,
        default=DEFAULT_COUNT_PER_CLASS,
        help=f"hard cases per class per locale (default: {DEFAULT_COUNT_PER_CLASS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"base seed for reproducible generation (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    total = 0
    for subtag, language in LOCALES.items():
        prompts = generate_locale(
            subtag, count_per_class=args.count_per_class, seed=args.seed
        )
        out_dir = args.out_dir / f"prompts-{subtag}"
        jsonl_path = write_jsonl(out_dir / f"{CATEGORY}.jsonl", prompts)
        flatten_to_prompts_txt(prompts, out_dir / f"{CATEGORY}.txt")
        total += len(prompts)
        print(f"wrote {jsonl_path} ({len(prompts)} hard cases, {language})")

    print(f"total: {total} hard cases across {len(LOCALES)} locales")


if __name__ == "__main__":
    main()
