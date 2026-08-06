#!/usr/bin/env python3
"""Write the interview-question layer of the TTS prompt suite.

English question set and MACHINE-DRAFT-PENDING policy live in
``audio_harness.interview_prompts``; this script is the CLI shell.

Usage:

    uv run python tools/gen_interview_prompts.py
    uv run python tools/gen_interview_prompts.py --out-dir /tmp/prompts
"""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_harness.interview_prompts import INTERVIEW_QUESTIONS_EN, generate


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="prompt suite root (default: data)",
    )
    args = parser.parse_args()

    jsonl_path, pending = generate(args.out_dir)
    print(f"wrote {jsonl_path} ({len(INTERVIEW_QUESTIONS_EN)} questions)")
    print(f"wrote {len(pending)} MACHINE-DRAFT-PENDING marker(s)")


if __name__ == "__main__":
    main()
