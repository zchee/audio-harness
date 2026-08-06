"""Tests for loading evaluation material from manifests and parquet corpora."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import wave

import orjson
import polars as pl
import pytest

from audio_harness.config import DatasetConfig
from audio_harness.dataset import DatasetError, load_clips, load_prompts


def _wav(seconds: float, rate: int = 16000) -> bytes:
    """Build a silent WAV payload of a known duration."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def _corpus(path: Path, rows: int = 5, *, rate: int = 16000) -> Path:
    """Write a parquet corpus shaped like a Hugging Face audio dataset."""
    pl.DataFrame({
        "sample_id": [f"clip-{i:03d}" for i in range(rows)],
        "audio": [{"bytes": _wav(0.5 + i * 0.1, rate), "path": None} for i in range(rows)],
        "duration_seconds": [0.5 + i * 0.1 for i in range(rows)],
        "transcription": [f"utterance number {i}" for i in range(rows)],
    }).write_parquet(path)
    return path


class TestParquetCorpus:
    """Parquet corpora carry their audio inline rather than as files."""

    def test_loads_embedded_audio_and_references(self, tmp_path: Path) -> None:
        config = DatasetConfig(parquet=str(_corpus(tmp_path / "c.parquet", rows=4)))
        clips = load_clips(config)

        assert len(clips) == 4
        assert [c.clip_id for c in clips] == [f"clip-{i:03d}" for i in range(4)]
        assert clips[0].reference == "utterance number 0"
        assert clips[0].sample_rate == 16000
        assert clips[0].duration_s == pytest.approx(0.5, abs=0.01)
        assert all(c.pcm for c in clips), "every clip must decode to real samples"

    def test_resamples_to_the_harness_rate(self, tmp_path: Path) -> None:
        config = DatasetConfig(parquet=str(_corpus(tmp_path / "c.parquet", rows=2, rate=48000)))
        clips = load_clips(config)

        assert all(c.sample_rate == 16000 for c in clips), (
            "every provider must receive identical audio, so no corpus keeps its own sample rate"
        )
        assert clips[0].duration_s == pytest.approx(0.5, abs=0.02)

    def test_limit_without_seed_takes_the_head(self, tmp_path: Path) -> None:
        config = DatasetConfig(parquet=str(_corpus(tmp_path / "c.parquet", rows=10)), limit=3)
        clips = load_clips(config)

        assert [c.clip_id for c in clips] == ["clip-000", "clip-001", "clip-002"]

    def test_seeded_sample_is_reproducible_and_not_the_head(self, tmp_path: Path) -> None:
        path = str(_corpus(tmp_path / "c.parquet", rows=40))
        config = DatasetConfig(parquet=path, limit=6, sample_seed=1234)

        first = [c.clip_id for c in load_clips(config)]
        second = [c.clip_id for c in load_clips(config)]

        assert first == second, "a pinned seed must give the same subset"
        assert len(first) == 6
        assert first != [f"clip-{i:03d}" for i in range(6)], (
            "corpora are often ordered by length or source, so sampling must not silently return the head"
        )

    def test_different_seeds_select_different_clips(self, tmp_path: Path) -> None:
        path = str(_corpus(tmp_path / "c.parquet", rows=40))
        a = [c.clip_id for c in load_clips(DatasetConfig(parquet=path, limit=8, sample_seed=1))]
        b = [c.clip_id for c in load_clips(DatasetConfig(parquet=path, limit=8, sample_seed=2))]

        assert a != b

    def test_limit_above_corpus_size_returns_everything(self, tmp_path: Path) -> None:
        config = DatasetConfig(
            parquet=str(_corpus(tmp_path / "c.parquet", rows=3)),
            limit=99,
            sample_seed=5,
        )
        assert len(load_clips(config)) == 3

    def test_custom_column_names(self, tmp_path: Path) -> None:
        path = tmp_path / "custom.parquet"
        pl.DataFrame({
            "uid": ["a", "b"],
            "wav": [{"bytes": _wav(0.4), "path": None} for _ in range(2)],
            "sentence": ["hello there", "goodbye now"],
        }).write_parquet(path)

        clips = load_clips(
            DatasetConfig(
                parquet=str(path),
                id_column="uid",
                audio_column="wav",
                text_column="sentence",
            )
        )
        assert [c.clip_id for c in clips] == ["a", "b"]
        assert clips[1].reference == "goodbye now"

    def test_raw_bytes_column_is_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "raw.parquet"
        pl.DataFrame({
            "sample_id": ["x"],
            "audio": [_wav(0.3)],
            "transcription": ["raw bytes work"],
        }).write_parquet(path)

        clips = load_clips(DatasetConfig(parquet=str(path)))
        assert clips[0].reference == "raw bytes work"


class TestCorpusErrors:
    """A corpus problem must be reported, never silently scored as failure."""

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="parquet file not found"):
            load_clips(DatasetConfig(parquet=str(tmp_path / "nope.parquet")))

    def test_missing_column_names_what_is_available(self, tmp_path: Path) -> None:
        path = _corpus(tmp_path / "c.parquet")
        config = DatasetConfig(parquet=str(path), text_column="does_not_exist")

        with pytest.raises(DatasetError, match="does_not_exist") as excinfo:
            load_clips(config)
        assert "transcription" in str(excinfo.value), "the error should list the columns that do exist"

    def test_undecodable_audio_is_rejected_not_counted_as_provider_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.parquet"
        pl.DataFrame({
            "sample_id": ["ok", "bad"],
            "audio": [
                {"bytes": _wav(0.4), "path": None},
                {"bytes": b"not audio at all", "path": None},
            ],
            "transcription": ["fine", "broken"],
        }).write_parquet(path)

        with pytest.raises(DatasetError, match=r"failed to.*decode"):
            load_clips(DatasetConfig(parquet=str(path)))

    def test_both_sources_configured_is_an_error(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("")
        config = DatasetConfig(
            manifest=str(manifest),
            parquet=str(_corpus(tmp_path / "c.parquet")),
        )
        with pytest.raises(DatasetError, match="only one"):
            load_clips(config)

    def test_no_source_configured_is_an_error(self) -> None:
        with pytest.raises(DatasetError, match=r"manifest or dataset\.parquet"):
            load_clips(DatasetConfig())


class TestManifest:
    """The JSONL manifest remains the portable path for local audio."""

    def test_loads_relative_paths_against_the_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "clips").mkdir()
        (tmp_path / "clips" / "a.wav").write_bytes(_wav(0.6))
        manifest = tmp_path / "m.jsonl"
        manifest.write_bytes(orjson.dumps({"id": "a", "audio": "clips/a.wav", "text": "hello"}) + b"\n")

        clips = load_clips(DatasetConfig(manifest=str(manifest)))
        assert len(clips) == 1
        assert clips[0].clip_id == "a"
        assert clips[0].reference == "hello"

    def test_malformed_line_reports_its_line_number(self, tmp_path: Path) -> None:
        (tmp_path / "a.wav").write_bytes(_wav(0.3))
        manifest = tmp_path / "m.jsonl"
        manifest.write_text('{"audio": "a.wav", "text": "ok"}\nnot json\n')

        with pytest.raises(DatasetError, match=r"m\.jsonl:2"):
            load_clips(DatasetConfig(manifest=str(manifest)))

    def test_missing_audio_file_reports_its_line_number(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.jsonl"
        manifest.write_text('{"audio": "gone.wav", "text": "ok"}\n')

        with pytest.raises(DatasetError, match=r"m\.jsonl:1") as excinfo:
            load_clips(DatasetConfig(manifest=str(manifest)))
        assert "gone.wav" in str(excinfo.value), (
            "a manifest pointing at missing audio should name both the line "
            "and the file, not surface a bare FileNotFoundError"
        )


class TestPrompts:
    """TTS prompts come from a plain text file."""

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "p.txt"
        path.write_text("first line\n\n  \nsecond line\n")

        prompts = load_prompts(DatasetConfig(prompts=str(path), language="ja-JP"))
        assert [p.text for p in prompts] == ["first line", "second line"]
        assert prompts[0].language == "ja-JP"
        assert prompts[0].chars == len("first line")

    def test_empty_file_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "p.txt"
        path.write_text("\n\n")
        with pytest.raises(DatasetError, match="no prompts"):
            load_prompts(DatasetConfig(prompts=str(path)))
