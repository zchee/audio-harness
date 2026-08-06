"""Curate the commercial-safe multilingual STT lane from YODAS2 + Granary.

Downloads *metadata only* — YODAS2 ``text/*.json`` shards (~0.5 MB each) and
a bounded slice of each Granary manifest — never audio, so the disk
footprint stays in the tens of megabytes and nothing license-encumbered is
redistributed. Selection heuristics and filters live in
``audio_harness.curate``; this script is the network shell around them.

    uv run python tools/curate_yodas.py --out data/curated

Writes one JSONL manifest per language and source with per-row license and
``gold_status: unverified`` fields. Transcripts stay unverified until a
human-reviewed sample reaches subtitle-vs-audio CER < 5% for that language
(the plan's gold rule); this script never promotes anything to gold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import orjson

from audio_harness.curate import (
    manifest_row,
    parse_granary_lines,
    parse_yodas_text_shard,
    sample_candidates,
)

YODAS_BASE = "https://huggingface.co/datasets/espnet/yodas2/resolve/main"
GRANARY_BASE = "https://huggingface.co/datasets/nvidia/Granary/resolve/main"

# The twelve Speech-MASSIVE languages. Subset xx000 holds manually
# subtitled videos — auto captions (xx1xx) would make the gold review fail
# before it starts.
YODAS_LANGUAGES: dict[str, str] = {
    "ar-SA": "ar000",
    "de-DE": "de000",
    "es-ES": "es000",
    "fr-FR": "fr000",
    "hu-HU": "hu000",
    "ko-KR": "ko000",
    "nl-NL": "nl000",
    "pl-PL": "pl000",
    "pt-PT": "pt000",
    "ru-RU": "ru000",
    "tr-TR": "tr000",
    "vi-VN": "vi000",
}

# The eight EU languages Granary shares with the Speech-MASSIVE set.
GRANARY_LANGUAGES: dict[str, str] = {
    "de-DE": "de",
    "es-ES": "es",
    "fr-FR": "fr",
    "hu-HU": "hu",
    "nl-NL": "nl",
    "pl-PL": "pl",
    "pt-PT": "pt",
    "ru-RU": "ru",
}
GRANARY_SOURCES = ("ytc", "voxpopuli", "yodas")
"""Preference order. ``ytc`` (YouTube-Commons, explicitly CC BY uploads) has
the cleanest provenance; ``voxpopuli`` is parliament speech; ``yodas`` is a
last resort because it shares upstream provenance with the YODAS2 lane —
Russian only exists there, and the manifest's ``subset`` field records which
source each row actually came from."""

GRANARY_SLICE_BYTES = 4_000_000
"""Bounded read per manifest: a few MB of JSONL lines yields thousands of
candidates without pulling files that can reach hundreds of MB."""


def fetch_yodas_candidates(
    client: httpx.Client, language: str, subset: str, shard: str
) -> list | None:
    """Download one text shard and return its filtered candidates.

    Returns ``None`` when the shard does not exist, which is how a small
    subset signals it has run out of shards.
    """
    url = f"{YODAS_BASE}/data/{subset}/text/{shard}.json"
    response = client.get(url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return parse_yodas_text_shard(
        orjson.loads(response.content), subset=subset, shard=shard, language=language
    )


def fetch_granary_candidates(client: httpx.Client, language: str, code: str) -> list:
    """Download a bounded slice of the best available Granary source.

    Sources are tried in provenance-preference order because languages
    differ in coverage — Russian ships only the ``yodas`` source. A ranged
    request keeps the download bounded; the last line of the slice is
    dropped because it is almost always cut mid-record.
    """
    for source in GRANARY_SOURCES:
        # Large languages shard the manifest per YODAS subset; the 000 shard
        # is the manually subtitled slice, matching the YODAS2 lane's choice.
        for name in (f"{code}_asr.jsonl", f"{code}_asr_{code}000.jsonl"):
            url = f"{GRANARY_BASE}/{code}/{source}/{name}"
            response = client.get(
                url, headers={"Range": f"bytes=0-{GRANARY_SLICE_BYTES - 1}"}
            )
            if response.status_code == 404:
                continue
            response.raise_for_status()
            lines = response.text.splitlines()
            if len(response.content) >= GRANARY_SLICE_BYTES and lines:
                lines = lines[:-1]
            return parse_granary_lines(lines, subset=source, language=language)
    return []


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/curated"))
    parser.add_argument("--per-language", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--max-shards",
        type=int,
        default=4,
        help="YODAS2 text shards to widen to when a language falls short",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    shortfalls: list[str] = []
    with httpx.Client(
        timeout=httpx.Timeout(120.0, connect=15.0), follow_redirects=True
    ) as client:
        for language, subset in YODAS_LANGUAGES.items():
            candidates: list = []
            chosen: list = []
            for index in range(args.max_shards):
                shard = fetch_yodas_candidates(client, language, subset, f"{index:08d}")
                if shard is None:
                    break
                candidates.extend(shard)
                chosen = sample_candidates(
                    candidates, count=args.per_language, seed=args.seed
                )
                if len(chosen) >= args.per_language:
                    break
            path = _write(args.out / f"yodas2-{language}.jsonl", chosen)
            print(
                f"yodas2 {language}: {len(candidates)} candidates "
                f"-> {len(chosen)} curated -> {path}"
            )
            if len(chosen) < 30:
                shortfalls.append(f"yodas2 {language}: {len(chosen)} < 30")

        for language, code in GRANARY_LANGUAGES.items():
            candidates = fetch_granary_candidates(client, language, code)
            chosen = sample_candidates(
                candidates, count=args.per_language, seed=args.seed
            )
            source = candidates[0].subset if candidates else "none"
            path = _write(args.out / f"granary-{language}.jsonl", chosen)
            print(
                f"granary/{source} {language}: {len(candidates)} candidates "
                f"-> {len(chosen)} curated -> {path}"
            )
            if len(chosen) < 30:
                shortfalls.append(f"granary {language}: {len(chosen)} < 30")

    if shortfalls:
        print("SHORTFALLS (need a wider shard or looser filters):")
        for line in shortfalls:
            print(f"  {line}")
    print(
        "gold_status is 'unverified' everywhere: transcripts only become gold "
        "after a human-reviewed sample reaches CER < 5% for that language."
    )


def _write(path: Path, candidates: list) -> Path:
    """Write one manifest, one JSON record per line."""
    with path.open("wb") as handle:
        for candidate in candidates:
            handle.write(orjson.dumps(manifest_row(candidate)))
            handle.write(b"\n")
    return path


if __name__ == "__main__":
    main()
