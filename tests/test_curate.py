"""Tests for the commercial-safe lane curation.

Curation decides what the public multilingual numbers get computed on, so
the properties under test are the ones that keep those numbers defensible:
filters that reject junk, a diversity cap, reproducible sampling, and
manifest rows that never claim more than the license and gold rule allow.
"""

from __future__ import annotations

import orjson

from audio_harness.curate import (
    Candidate,
    manifest_row,
    parse_granary_lines,
    parse_yodas_text_shard,
    sample_candidates,
)


def _yodas_shard(utterances: dict[str, str]) -> list[dict[str, object]]:
    return [{"audio_id": "vid01", "text": utterances}]


def _many(utterances: dict[str, str]) -> dict[str, str]:
    """Pad a video with filler so it clears the long-form threshold."""
    padded = dict(utterances)
    for index in range(40):
        start = 100_00 + index * 500
        padded[f"vid01-9{index:04d}-{start:08d}-{start + 450:08d}"] = (
            f"filler utterance number {index} with enough text"
        )
    return padded


class TestYodasParsing:
    """Timing comes from utterance keys; filters reject junk."""

    def test_parses_offsets_from_centisecond_keys(self) -> None:
        shard = _yodas_shard(
            _many({"vid01-00000-00000018-00000450": "elég hosszú mondat ez ide"})
        )

        candidates = parse_yodas_text_shard(
            shard, subset="hu000", shard="00000000", language="hu-HU"
        )

        first = candidates[0]
        assert first.start_s == 0.18
        assert first.end_s == 4.50
        assert first.subset == "hu000"
        assert first.language == "hu-HU"

    def test_filters_reject_unusable_utterances(self) -> None:
        tests = {
            "too short in time": {
                "key": "vid01-00001-00000000-00000100",
                "text": "rövid de szöveges mondat",
            },
            "too long in time": {
                "key": "vid01-00002-00000000-00004000",
                "text": "hosszú caption ami két jeleneten át lóg",
            },
            "too little text": {
                "key": "vid01-00003-00000000-00000500",
                "text": "rövid",
            },
            "bracketed marker": {
                "key": "vid01-00004-00000000-00000500",
                "text": "[Zene] valami hosszabb szöveg itt",
            },
            "url in caption": {
                "key": "vid01-00005-00000000-00000500",
                "text": "iratkozz fel https://example.com köszi",
            },
        }
        for name, case in tests.items():
            shard = _yodas_shard(_many({case["key"]: case["text"]}))
            candidates = parse_yodas_text_shard(
                shard, subset="hu000", shard="00000000", language="hu-HU"
            )
            assert all(c.utt_id != case["key"] for c in candidates), name

    def test_short_videos_are_not_talk_content(self) -> None:
        shard = _yodas_shard(
            {"vid01-00000-00000018-00000450": "elég hosszú mondat ez ide"}
        )
        assert (
            parse_yodas_text_shard(
                shard, subset="hu000", shard="00000000", language="hu-HU"
            )
            == []
        )


class TestGranaryParsing:
    """Granary rows are manifests already; filters still apply."""

    def test_parses_rows_and_skips_partial_lines(self) -> None:
        lines = [
            orjson.dumps(
                {
                    "utt_id": "de000_x_1",
                    "text": "ein ausreichend langer satz",
                    "duration": 4.2,
                    "original_source_id": "srcvid",
                    "dataset_source": "ytc",
                }
            ).decode(),
            '{"utt_id": "truncat',
        ]

        candidates = parse_granary_lines(lines, subset="ytc", language="de-DE")

        assert len(candidates) == 1
        assert candidates[0].video_id == "srcvid"
        assert candidates[0].end_s == 4.2

    def test_duration_filter_applies(self) -> None:
        lines = [
            orjson.dumps(
                {"utt_id": "u1", "text": "kurzer aber valider text", "duration": 0.8}
            ).decode()
        ]
        assert parse_granary_lines(lines, subset="ytc", language="de-DE") == []


def _candidate(video: str, utt: str) -> Candidate:
    return Candidate(
        source="yodas2",
        subset="de000",
        shard="00000000",
        video_id=video,
        utt_id=utt,
        language="de-DE",
        start_s=0.0,
        end_s=5.0,
        text="ein ausreichend langer beispielsatz",
    )


class TestSampling:
    """Reproducible and speaker-diverse, or the lane drifts run to run."""

    def test_same_seed_same_sample(self) -> None:
        pool = [_candidate(f"v{i % 20}", f"u{i}") for i in range(200)]

        first = sample_candidates(pool, count=30, seed=7)
        second = sample_candidates(pool, count=30, seed=7)

        assert first == second

    def test_per_video_cap_holds(self) -> None:
        pool = [_candidate("one-video", f"u{i}") for i in range(50)] + [
            _candidate(f"v{i}", f"w{i}") for i in range(40)
        ]

        chosen = sample_candidates(pool, count=40, seed=7)

        from collections import Counter

        counts = Counter(c.video_id for c in chosen)
        assert counts["one-video"] <= 3, (
            "thirty clips from one episode would measure one microphone"
        )

    def test_sample_meets_the_lane_minimum_when_the_pool_allows(self) -> None:
        pool = [_candidate(f"v{i}", f"u{i}") for i in range(120)]
        assert len(sample_candidates(pool, count=40, seed=7)) == 40


class TestManifestRows:
    """Rows carry license and never claim gold."""

    def test_row_fields(self) -> None:
        row = manifest_row(_candidate("vid", "utt"))

        assert row["license"] == "CC-BY-3.0"
        assert row["gold_status"] == "unverified", (
            "only the human CER<5% review may promote a language to gold"
        )
        assert row["duration_s"] == 5.0
        assert row["language"] == "de-DE"

    def test_granary_rows_carry_their_own_license(self) -> None:
        candidate = Candidate(
            source="granary",
            subset="ytc",
            shard="ytc",
            video_id="vid",
            utt_id="utt",
            language="de-DE",
            start_s=0.0,
            end_s=4.0,
            text="ein ausreichend langer satz",
        )
        assert manifest_row(candidate)["license"] == "CC-BY-4.0"
