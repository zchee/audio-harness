#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# ///
"""Download the MUSAN corpus for the hallucination/silence lane.

MUSAN (D. Snyder, G. Chen, D. Povey, "MUSAN: A Music, Speech, and Noise
Corpus", arXiv:1510.08484) is distributed by OpenSLR under CC BY 4.0 —
attribution is required, which this script records next to the audio. Only
the requested subset is extracted (the harness needs ``noise``); the full
tarball is ~11 GB, so it is streamed and never written to disk whole.

Usage:

    uv run tools/fetch_musan.py                    # noise subset -> data/musan
    uv run tools/fetch_musan.py --subset all
    uv run tools/fetch_musan.py --dest /tmp/musan --force
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"

SUBSETS = ("noise", "music", "speech")

ATTRIBUTION = """\
MUSAN: A Music, Speech, and Noise Corpus
David Snyder, Guoguo Chen, Daniel Povey
arXiv:1510.08484 (2015), https://www.openslr.org/17/

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
https://creativecommons.org/licenses/by/4.0/

Downloaded by tools/fetch_musan.py for the audio-harness hallucination lane
(configs/stt-hallucination.yaml). Do not commit this audio to the repository;
clips are generated lazily from it at benchmark load time.
"""


def fetch(dest: Path, subsets: tuple[str, ...], *, force: bool) -> int:
    """Stream the MUSAN tarball and extract the requested subsets.

    Args:
        dest: Directory receiving ``noise/``, ``music/`` and/or ``speech/``.
        subsets: Which top-level MUSAN subsets to extract.
        force: Re-extract even when the subset directories already exist.

    Returns:
        Number of files extracted.
    """
    existing = [name for name in subsets if (dest / name).is_dir()]
    if existing and not force:
        print(f"already present in {dest}: {', '.join(existing)} (use --force)")
        _write_attribution(dest)
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    _write_attribution(dest)
    wanted = tuple(f"musan/{name}/" for name in subsets)

    print(f"downloading {MUSAN_URL}")
    print(f"extracting {', '.join(subsets)} -> {dest} (streamed; ~11 GB transfer)")
    count = 0
    with (
        urllib.request.urlopen(MUSAN_URL) as response,
        tarfile.open(fileobj=response, mode="r|gz") as archive,
    ):
        for member in archive:
            if not member.isfile():
                continue
            if not member.name.startswith(wanted):
                continue
            # Strip the leading "musan/" so files land at <dest>/<subset>/...
            member.name = member.name.split("/", 1)[1]
            archive.extract(member, dest, filter="data")
            count += 1
            if count % 100 == 0:
                print(f"  {count} files...")

    print(f"done: {count} files under {dest}")
    return count


def _write_attribution(dest: Path) -> None:
    """Record the CC BY 4.0 attribution beside the downloaded audio."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")


def main() -> int:
    """Parse arguments and run the download."""
    parser = argparse.ArgumentParser(
        description="Download the MUSAN corpus (CC BY 4.0) for the "
        "hallucination/silence lane."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/musan"),
        help="directory to extract into (default: data/musan)",
    )
    parser.add_argument(
        "--subset",
        choices=(*SUBSETS, "all"),
        default="noise",
        help="MUSAN subset to extract (default: noise)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the subset already exists",
    )
    args = parser.parse_args()

    subsets = SUBSETS if args.subset == "all" else (args.subset,)
    try:
        fetch(args.dest, subsets, force=args.force)
    except (OSError, tarfile.TarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
