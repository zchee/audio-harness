"""Tests for the curated-manifest loader.

The loader is the trust boundary between redistributable manifests and
benchmark clips: it must cut exactly the referenced segments, never hit the
network when the cache can answer, skip unfetchable sources instead of
failing the lane, and carry license and gold status through to scoring so
unverified subtitles never reach ranked accuracy.
"""

from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import orjson
import pytest
import soundfile as sf

from audio_harness import curated
from audio_harness.config import SourceConfig
from audio_harness.curated import (
    CuratedManifestError,
    is_curated_manifest,
    load_curated_clips,
)
from audio_harness.dataset import load_source
from audio_harness.metrics import summarize
from audio_harness.report import render_stt_markdown, stt_summary_frame
from audio_harness.types import Mode, SttResult

NATIVE_RATE = 24000


def _wav_bytes(duration_s: float, *, tone_s: float | None = None) -> bytes:
    """A mono tone (with optional trailing silence) as WAV bytes."""
    tone_s = duration_s if tone_s is None else tone_s
    t = np.linspace(0, tone_s, int(NATIVE_RATE * tone_s), endpoint=False)
    samples = np.concatenate(
        [
            0.5 * np.sin(2 * np.pi * 220 * t),
            np.zeros(int(NATIVE_RATE * (duration_s - tone_s))),
        ]
    ).astype("float32")
    buffer = BytesIO()
    sf.write(buffer, samples, NATIVE_RATE, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def shard_tar(tmp_path: Path) -> Path:
    """A fixture shard: two recordings, one with an underscore in its id."""
    tar_path = tmp_path / "00000000.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        for name, payload in (
            ("vid01.wav", _wav_bytes(4.0, tone_s=3.0)),
            ("ab_c-d.wav", _wav_bytes(4.0)),
        ):
            info = tarfile.TarInfo(name=f"de000/{name}")
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    return tar_path


def _manifest(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(row) for row in rows) + b"\n")
    return path


def _yodas_row(utt_id: str, start_s: float, end_s: float) -> dict[str, object]:
    return {
        "source": "yodas2",
        "subset": "de000",
        "shard": "00000000",
        "video_id": "vid01",
        "utt_id": utt_id,
        "language": "de-DE",
        "start_s": start_s,
        "end_s": end_s,
        "text": "ein hinreichend langer satz",
        "license": "CC-BY-3.0",
        "gold_status": "unverified",
    }


_GRANARY_ROW: dict[str, object] = {
    "source": "granary",
    "subset": "yodas",
    "shard": "yodas",
    "video_id": "ab_c-d",
    "utt_id": "de000_00000000_ab_c-d_1_00_1_50",
    "language": "de-DE",
    "start_s": 0.0,
    "end_s": 1.5,
    "text": "noch ein langer satz",
    "license": "CC-BY-4.0",
    "gold_status": "unverified",
}

_YTC_ROW: dict[str, object] = {
    "source": "granary",
    "subset": "ytc",
    "shard": "ytc",
    "video_id": "yt01",
    "utt_id": "ytc-utt-1",
    "language": "de-DE",
    "start_s": 0.0,
    "end_s": 3.0,
    "text": "nicht abrufbarer satz",
    "license": "CC-BY-4.0",
    "gold_status": "unverified",
}


@pytest.fixture
def patched(tmp_path: Path, shard_tar: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the loader at the fixture shard and a scratch cache."""
    monkeypatch.setattr(curated, "CACHE_DIR", tmp_path / "cache")

    def open_shard(subset: str, shard: str):
        assert (subset, shard) == ("de000", "00000000"), (
            "only the fixture shard may be streamed"
        )
        return shard_tar.open("rb")

    monkeypatch.setattr(curated, "_open_shard", open_shard)
    return monkeypatch


class TestDetection:
    """Curated manifests are recognized by shape, not by config flags."""

    def test_curated_manifest_is_detected(self, tmp_path: Path) -> None:
        path = _manifest(tmp_path, [_yodas_row("u1", 0.5, 1.5)])
        assert is_curated_manifest(path)

    def test_legacy_manifest_is_not(self, tmp_path: Path) -> None:
        path = _manifest(tmp_path, [{"audio": "clip.wav", "text": "hallo"}])
        assert not is_curated_manifest(path)


class TestLoading:
    """Segments come out cut, resampled, and fully attributed."""

    def test_segments_are_cut_to_their_offsets(
        self, tmp_path: Path, patched: pytest.MonkeyPatch
    ) -> None:
        path = _manifest(
            tmp_path,
            [_yodas_row("u1", 0.5, 1.5), _yodas_row("u2", 1.0, 3.0), _GRANARY_ROW],
        )

        clips = load_curated_clips(SourceConfig(manifest=str(path)))

        assert [c.clip_id for c in clips] == ["u1", "u2", _GRANARY_ROW["utt_id"]]
        assert clips[0].sample_rate == 16000
        assert clips[0].duration_s == pytest.approx(1.0, abs=0.02)
        assert clips[1].duration_s == pytest.approx(2.0, abs=0.02)
        assert clips[2].duration_s == pytest.approx(1.5, abs=0.02), (
            "granary offsets come from the utt_id, video underscores intact"
        )
        assert clips[0].license == "CC-BY-3.0"
        assert clips[0].gold_status == "unverified"
        assert clips[0].reference == "ein hinreichend langer satz"
        assert clips[0].source_path.startswith("yodas2://de000/00000000/vid01#")

    def test_speech_end_is_detected_on_the_segment(
        self, tmp_path: Path, patched: pytest.MonkeyPatch
    ) -> None:
        # vid01 is tone until 3.0 s then silence; a segment spanning the
        # transition must place speech-end near the transition, not the end.
        path = _manifest(tmp_path, [_yodas_row("u1", 2.5, 4.0)])

        clips = load_curated_clips(SourceConfig(manifest=str(path)))

        assert clips[0].speech_end_s == pytest.approx(0.5, abs=0.1)

    def test_unfetchable_sources_are_skipped_not_failed(
        self, tmp_path: Path, patched: pytest.MonkeyPatch
    ) -> None:
        path = _manifest(tmp_path, [_yodas_row("u1", 0.5, 1.5), _YTC_ROW])

        clips = load_curated_clips(SourceConfig(manifest=str(path)))

        assert [c.clip_id for c in clips] == ["u1"]

    def test_limit_truncates_in_manifest_order(
        self, tmp_path: Path, patched: pytest.MonkeyPatch
    ) -> None:
        path = _manifest(
            tmp_path, [_yodas_row("u1", 0.5, 1.5), _yodas_row("u2", 1.0, 2.0)]
        )

        clips = load_curated_clips(SourceConfig(manifest=str(path), limit=1))

        assert [c.clip_id for c in clips] == ["u1"]

    def test_second_load_serves_from_cache_without_streaming(
        self, tmp_path: Path, patched: pytest.MonkeyPatch
    ) -> None:
        path = _manifest(tmp_path, [_yodas_row("u1", 0.5, 1.5)])
        load_curated_clips(SourceConfig(manifest=str(path)))

        def poisoned(subset: str, shard: str):
            raise AssertionError("cache hit must not stream the shard")

        patched.setattr(curated, "_open_shard", poisoned)
        clips = load_curated_clips(SourceConfig(manifest=str(path)))

        assert clips[0].duration_s == pytest.approx(1.0, abs=0.02)

    def test_dataset_dispatch_routes_curated_manifests(
        self, tmp_path: Path, patched: pytest.MonkeyPatch
    ) -> None:
        path = _manifest(tmp_path, [_yodas_row("u1", 0.5, 1.5)])

        clips = load_source(SourceConfig(manifest=str(path)))

        assert clips[0].clip_id == "u1"

    def test_malformed_granary_utt_id_is_an_error(
        self, tmp_path: Path, patched: pytest.MonkeyPatch
    ) -> None:
        row = dict(_GRANARY_ROW)
        row["utt_id"] = "not-encoding-anything"
        path = _manifest(tmp_path, [row])

        with pytest.raises(CuratedManifestError, match="utt_id"):
            load_curated_clips(SourceConfig(manifest=str(path)))


class TestScoringGate:
    """Unverified references measure latency, never ranked accuracy."""

    def _result(self, gold_status: str) -> SttResult:
        result = SttResult(
            provider="p1", clip_id="c1", mode=Mode.STREAM, text="hallo welt"
        )
        result.audio_s = 2.0
        result.ttft_s = 0.4
        result.raw["reference"] = "hallo welt kaputt"
        result.raw["language"] = "de-DE"
        result.raw["license"] = "CC-BY-3.0"
        if gold_status:
            result.raw["gold_status"] = gold_status
        return result

    def test_unverified_reference_is_excluded_from_accuracy(self) -> None:
        summary = summarize([self._result("unverified")], "de-DE")[0]

        assert summary.error_rate is None, (
            "an unverified subtitle must not feed ranked accuracy"
        )
        assert summary.ttft_s == [0.4], "latency needs no transcript truth"
        assert summary.unverified == 1
        assert summary.licenses == {"CC-BY-3.0"}

    def test_trusted_reference_still_scores(self) -> None:
        summary = summarize([self._result("")], "de-DE")[0]

        assert summary.error_rate is not None
        assert summary.unverified == 0

    def test_report_renders_license_and_unverified_columns(self) -> None:
        markdown = render_stt_markdown(
            stt_summary_frame([self._result("unverified")], "de-DE")
        )

        assert "License" in markdown
        assert "CC-BY-3.0" in markdown
        assert "Unverified" in markdown
