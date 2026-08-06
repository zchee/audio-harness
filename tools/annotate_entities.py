"""Annotate saved benchmark results with entity tags.

Re-running vendors is the expensive part of a benchmark; this reads saved
``stt-results.jsonl`` files, tags each record's reference (gold Speech-MASSIVE
slots where the corpus provides them, deterministic per-language rules
everywhere else) and writes the records back with ``reference_annotated``
set — after which ``audio-harness report`` renders the per-class entity
columns at zero API cost.

    uv run python tools/annotate_entities.py results/<run>/stt-results.jsonl
    uv run python tools/annotate_entities.py --in-place results/*/stt-results.jsonl

By default the annotated copy is written next to the input as
``stt-results-annotated.jsonl`` so the original stays untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import orjson

from audio_harness.autotag import annotate_record
from audio_harness.entities import parse_annotated

DEFAULT_MASSIVE_DIR = Path("data/hf/speech-massive")


def load_gold(massive_dir: Path) -> dict[tuple[str, str], str]:
    """Map ``(language, clip_id)`` to the corpus's gold slot annotation.

    Speech-MASSIVE ships one directory per locale whose name is the BCP-47
    tag the harness records on every clip, so the directory name is the join
    key — the aggregate ``all`` directory would double-count and is skipped.
    """
    import polars as pl

    gold: dict[tuple[str, str], str] = {}
    if not massive_dir.is_dir():
        return gold
    for locale_dir in sorted(massive_dir.iterdir()):
        if not locale_dir.is_dir() or locale_dir.name == "all":
            continue
        files = sorted(locale_dir.glob("*.parquet"))
        if not files:
            continue
        frame = pl.concat(
            [pl.read_parquet(file, columns=["id", "annot_utt"]) for file in files]
        )
        for clip_id, annot_utt in frame.iter_rows():
            if annot_utt:
                gold[(locale_dir.name, str(clip_id))] = annot_utt
    return gold


def annotate_file(
    path: Path,
    gold: dict[tuple[str, str], str],
    *,
    in_place: bool,
    force: bool,
) -> tuple[Path, Counter[str]]:
    """Annotate one results file and return where it was written."""
    counts: Counter[str] = Counter()
    lines: list[bytes] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = orjson.loads(line)
        if annotate_record(record, gold, force=force):
            annotated = str(record.get("reference_annotated", ""))
            for _, label in parse_annotated(annotated):
                if label is not None:
                    counts[f"{record.get('language', '?')}/{label}"] += 1
        lines.append(orjson.dumps(record))

    target = path if in_place else path.with_name(f"{path.stem}-annotated.jsonl")
    target.write_bytes(b"\n".join(lines) + b"\n")
    return target, counts


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path, help="stt-results.jsonl files")
    parser.add_argument(
        "--massive-dir",
        type=Path,
        default=DEFAULT_MASSIVE_DIR,
        help="Speech-MASSIVE corpus root for gold slot annotations",
    )
    parser.add_argument(
        "--in-place", action="store_true", help="overwrite inputs instead of copying"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-annotate already-annotated records"
    )
    args = parser.parse_args()

    gold = load_gold(args.massive_dir)
    print(f"gold annotations loaded: {len(gold)}")

    for path in args.results:
        target, counts = annotate_file(
            path, gold, in_place=args.in_place, force=args.force
        )
        total = sum(counts.values())
        print(f"{path} -> {target}: {total} tagged spans")
        for key in sorted(counts):
            print(f"  {key}: {counts[key]}")


if __name__ == "__main__":
    main()
