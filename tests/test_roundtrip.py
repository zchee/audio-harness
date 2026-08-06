"""Tests for the two-judge round-trip schema, migration and family rules.

The round-trip score only means something when no lane is scored by its own
vendor family — a recognizer decodes its own vendor's voices best. These tests
pin the whole chain: config parsing (legacy scalar and list forms), the
per-candidate cross-family validation, verdict recording, the ranked-score
computation, and the legacy-file migration path.
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings

import orjson
import pytest

from audio_harness import report, runner, stt, tts
from audio_harness.cli import _results_kind, validate_roundtrip_judges
from audio_harness.config import BenchmarkConfig, ConfigError
from audio_harness.types import AudioClip, Mode, SttResult, TtsPrompt, TtsResult


@tts.register
class _FakeOpenAiTts(tts.TtsProvider):
    """Stand-in for the P3 OpenAI TTS lane the family rule must catch."""

    key = "fake-openai-tts"
    vendor = "fake-openai"
    family = "openai"
    supports_batch = True


@stt.register
class _EchoJudge(stt.SttProvider):
    """Judge that transcribes perfectly by echoing the reference."""

    key = "fake-echo-judge"
    vendor = "fake-echo"
    supports_batch = True

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        result = self._result(clip, Mode.BATCH)
        result.text = clip.reference or ""
        return result


@stt.register
class _BrokenJudge(stt.SttProvider):
    """Judge whose vendor call always fails."""

    key = "fake-broken-judge"
    vendor = "fake-broken"
    supports_batch = True

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        raise RuntimeError("boom")


def _config(tts_names: list[str], judges: object) -> BenchmarkConfig:
    return BenchmarkConfig.from_dict({"tts": tts_names, "roundtrip_stt": judges})


def _tts_result(provider: str, prompt_id: str = "p1") -> TtsResult:
    return TtsResult(
        provider=provider,
        prompt_id=prompt_id,
        mode=Mode.BATCH,
        audio=b"\x00\x00" * 1600,
        sample_rate=16000,
        audio_s=0.1,
        chars=11,
        ttfb_s=0.1,
        total_s=0.2,
        raw={"text": "hello world"},
    )


def _judged_result(provider: str, verdicts: list[dict[str, object]]) -> TtsResult:
    result = _tts_result(provider)
    result.raw["roundtrip"] = verdicts
    return result


class TestFamilyResolution:
    """Family — not billing vendor — drives the coupling rule."""

    def test_family_defaults_to_vendor(self) -> None:
        assert stt.family_of("deepgram-nova3") == "deepgram"
        assert tts.family_of("deepgram-aura2") == "deepgram"

    def test_explicit_family_overrides_vendor(self) -> None:
        assert stt.family_of("whisper-local") == "openai"
        assert tts.family_of("fake-openai-tts") == "openai"

    def test_unknown_key_forms_its_own_family(self) -> None:
        """Results from removed adapters must render, not crash the report."""
        assert stt.family_of("ghost-provider") == "ghost-provider"
        assert tts.family_of("ghost-provider") == "ghost-provider"


class TestRoundtripConfig:
    """The judge field accepts the legacy scalar and the new list form."""

    def test_legacy_scalar_mapping_parses_with_deprecation(self) -> None:
        with pytest.warns(DeprecationWarning, match="single-judge"):
            config = BenchmarkConfig.from_dict({
                "roundtrip_stt": {
                    "name": "deepgram-nova3",
                    "options": {"smart_format": False},
                }
            })
        assert [judge.name for judge in config.roundtrip_stt] == ["deepgram-nova3"]
        assert config.roundtrip_stt[0].options == {"smart_format": False}

    def test_legacy_bare_name_parses_with_deprecation(self) -> None:
        with pytest.warns(DeprecationWarning):
            config = BenchmarkConfig.from_dict({"roundtrip_stt": "deepgram-nova3"})
        assert [judge.name for judge in config.roundtrip_stt] == ["deepgram-nova3"]

    def test_list_form_parses_without_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = BenchmarkConfig.from_dict({"roundtrip_stt": [{"name": "deepgram-nova3"}, "whisper-local"]})
        assert [judge.name for judge in config.roundtrip_stt] == [
            "deepgram-nova3",
            "whisper-local",
        ]

    def test_absent_means_no_judges(self) -> None:
        assert BenchmarkConfig.from_dict({}).roundtrip_stt == []


class TestCrossFamilyValidation:
    """Every TTS lane needs at least one judge outside its own family."""

    def test_legacy_scalar_judge_is_accepted_for_other_families(self) -> None:
        with pytest.warns(DeprecationWarning):
            config = _config(["cartesia-sonic35"], {"name": "deepgram-nova3"})
        validate_roundtrip_judges(config)

    def test_candidate_with_no_cross_family_judge_is_rejected(self) -> None:
        config = _config(["deepgram-aura2"], [{"name": "deepgram-nova3"}])
        with pytest.raises(ConfigError, match="deepgram-aura2"):
            validate_roundtrip_judges(config)

    def test_whisper_local_cannot_solely_judge_an_openai_family_tts(self) -> None:
        """whisper-local bills nobody but is still OpenAI lineage."""
        config = _config(["fake-openai-tts"], ["whisper-local"])
        with pytest.raises(ConfigError, match="openai"):
            validate_roundtrip_judges(config)

    def test_mixed_two_by_two_is_accepted(self) -> None:
        """Judges {deepgram, openai} x candidates {deepgram, openai} are
        valid: each lane is covered by the other family's judge."""
        config = _config(
            ["deepgram-aura2", "fake-openai-tts"],
            [{"name": "deepgram-nova3"}, "whisper-local"],
        )
        validate_roundtrip_judges(config)

    def test_unknown_judge_is_rejected_before_any_synthesis(self) -> None:
        config = _config(["deepgram-aura2"], ["no-such-judge"])
        with pytest.raises(ConfigError, match="no-such-judge"):
            validate_roundtrip_judges(config)

    def test_no_judges_skips_validation(self) -> None:
        validate_roundtrip_judges(_config(["deepgram-aura2"], None))


class TestScoreRoundtrip:
    """Verdicts land under raw["roundtrip"] in config order."""

    async def test_every_judge_writes_a_verdict_in_config_order(self) -> None:
        config = BenchmarkConfig.from_dict({
            "roundtrip_stt": [
                {"name": "fake-echo-judge"},
                {"name": "fake-broken-judge"},
            ]
        })
        result = _tts_result("fake-openai-tts")
        prompt = TtsPrompt(prompt_id="p1", text="hello world", language="en-US")

        await runner.score_roundtrip(config, [result], {"p1": prompt})

        verdicts = result.raw["roundtrip"]
        assert isinstance(verdicts, list)
        assert [v["provider"] for v in verdicts] == [
            "fake-echo-judge",
            "fake-broken-judge",
        ]
        assert verdicts[0] == {
            "provider": "fake-echo-judge",
            "text": "hello world",
            "error": None,
        }
        assert verdicts[1]["text"] is None
        assert "RuntimeError: boom" in str(verdicts[1]["error"]), (
            "a judge failure must be recorded on the verdict, not lose the other judge's score"
        )

    async def test_failed_synthesis_is_never_judged(self) -> None:
        config = BenchmarkConfig.from_dict({"roundtrip_stt": ["fake-echo-judge"]})
        result = TtsResult(provider="fake-openai-tts", prompt_id="p1", mode=Mode.BATCH, error="boom")

        await runner.score_roundtrip(config, [result], {})

        assert "roundtrip" not in result.raw


class TestRankedScore:
    """Each lane is ranked only by judges outside its own family."""

    def test_same_family_judge_is_diagnostic_only(self) -> None:
        result = _judged_result(
            "deepgram-aura2",
            [
                {"provider": "deepgram-nova3", "text": "hello world", "error": None},
                {"provider": "whisper-local", "text": "hello word", "error": None},
            ],
        )

        row = report.tts_summary_frame([result], "en-US").to_dicts()[0]

        assert row["roundtrip_error_rate"] == pytest.approx(0.5), (
            "the ranked score must come from the cross-family whisper judge; "
            "the perfect same-family deepgram score would flatter the lane"
        )
        assert row["rt[deepgram-nova3]"] == "0.00% †"
        assert row["rt[whisper-local]"] == "50.00%"

    def test_openai_lane_is_ranked_by_the_deepgram_judge(self) -> None:
        result = _judged_result(
            "fake-openai-tts",
            [
                {"provider": "deepgram-nova3", "text": "hello word", "error": None},
                {"provider": "whisper-local", "text": "hello world", "error": None},
            ],
        )

        row = report.tts_summary_frame([result], "en-US").to_dicts()[0]

        assert row["roundtrip_error_rate"] == pytest.approx(0.5)
        assert row["rt[whisper-local]"].endswith("†"), "whisper-local shares OpenAI lineage with this candidate"

    def test_divergent_judges_are_flagged(self) -> None:
        result = _judged_result(
            "deepgram-aura2",
            [
                {"provider": "deepgram-nova3", "text": "hello world", "error": None},
                {"provider": "whisper-local", "text": "hello word", "error": None},
            ],
        )

        row = report.tts_summary_frame([result], "en-US").to_dicts()[0]

        assert row["rt_divergence"] is True, "0% vs 50% is far beyond 2 WER points"

    def test_agreeing_judges_are_not_flagged(self) -> None:
        result = _judged_result(
            "deepgram-aura2",
            [
                {"provider": "deepgram-nova3", "text": "hello world", "error": None},
                {"provider": "whisper-local", "text": "hello world", "error": None},
            ],
        )

        row = report.tts_summary_frame([result], "en-US").to_dicts()[0]

        assert row["rt_divergence"] is False

    def test_judge_errors_are_not_scored(self) -> None:
        result = _judged_result(
            "deepgram-aura2",
            [
                {"provider": "deepgram-nova3", "text": "hello world", "error": None},
                {"provider": "whisper-local", "text": None, "error": "boom"},
            ],
        )

        row = report.tts_summary_frame([result], "en-US").to_dicts()[0]

        assert row["rt[whisper-local]"] == "—"
        assert row["roundtrip_error_rate"] is None, (
            "with the only cross-family judge failed, the lane has no ranked score rather than a same-family one"
        )

    def test_markdown_renders_judge_columns_and_divergence(self) -> None:
        result = _judged_result(
            "deepgram-aura2",
            [
                {"provider": "deepgram-nova3", "text": "hello world", "error": None},
                {"provider": "whisper-local", "text": "hello word", "error": None},
            ],
        )

        markdown = report.render_tts_markdown(report.tts_summary_frame([result], "en-US"))

        assert "RT deepgram-nova3" in markdown
        assert "RT whisper-local" in markdown
        assert "0.00% †" in markdown
        assert "⚠ >2pt" in markdown


class TestLegacyShape:
    """Pre-migration results render as one-judge lanes, never dropped."""

    def test_legacy_scalar_renders_for_a_cross_family_lane(self) -> None:
        result = _tts_result("cartesia-sonic35")
        result.raw["roundtrip_text"] = "hello world"
        result.raw["roundtrip_provider"] = "deepgram-nova3"

        row = report.tts_summary_frame([result], "en-US").to_dicts()[0]

        assert row["rt[deepgram-nova3]"] == "0.00%"
        assert row["roundtrip_error_rate"] == pytest.approx(0.0)

    def test_legacy_same_family_lane_loses_its_ranked_score(self) -> None:
        """The historical deepgram-judges-deepgram coupling was the bug; a
        legacy file must not keep ranking that lane."""
        result = _tts_result("deepgram-aura2")
        result.raw["roundtrip_text"] = "hello world"
        result.raw["roundtrip_provider"] = "deepgram-nova3"

        row = report.tts_summary_frame([result], "en-US").to_dicts()[0]

        assert row["roundtrip_error_rate"] is None
        assert row["rt[deepgram-nova3]"] == "0.00% †"


class TestResultsFiles:
    """JSONL moves to the list shape; both shapes stay readable."""

    def test_write_then_read_preserves_the_judge_list(self, tmp_path: Path) -> None:
        verdicts: list[dict[str, object]] = [
            {"provider": "deepgram-nova3", "text": "hello world", "error": None},
            {"provider": "whisper-local", "text": "hello word", "error": None},
        ]
        result = _judged_result("deepgram-aura2", verdicts)

        path = runner.write_tts_results([result], tmp_path, save_audio=False)
        loaded = runner.read_tts_results(path)

        assert loaded[0].raw["roundtrip"] == verdicts
        assert loaded[0].ok, (
            "audio bytes are not persisted, so the recorded duration is the evidence that audio existed"
        )

    def test_legacy_file_reads_as_a_one_judge_lane(self, tmp_path: Path) -> None:
        record = {
            "provider": "deepgram-aura2",
            "prompt_id": "p1",
            "mode": "batch",
            "chars": 11,
            "audio_s": 0.1,
            "ttfb_s": 0.1,
            "total_s": 0.2,
            "rtf": 2.0,
            "error": None,
            "text": "hello world",
            "roundtrip_text": "hello world",
            "roundtrip_provider": "deepgram-nova3",
            "audio_path": None,
        }
        file = tmp_path / "tts-results.jsonl"
        file.write_bytes(orjson.dumps(record) + b"\n")

        loaded = runner.read_tts_results(file)

        assert loaded[0].raw["roundtrip"] == [{"provider": "deepgram-nova3", "text": "hello world", "error": None}]

    def test_results_kind_is_detected_from_the_first_record(self, tmp_path: Path) -> None:
        tts_file = tmp_path / "tts-results.jsonl"
        tts_file.write_bytes(orjson.dumps({"provider": "x", "prompt_id": "p"}) + b"\n")
        stt_file = tmp_path / "stt-results.jsonl"
        stt_file.write_bytes(orjson.dumps({"provider": "x", "clip_id": "c"}) + b"\n")

        assert _results_kind(tts_file) == "tts"
        assert _results_kind(stt_file) == "stt"


class TestWhisperLocalAdapter:
    """The offline judge must exist even without its optional dependency."""

    def test_registered_as_batch_only_openai_family(self) -> None:
        adapter = stt.create("whisper-local")
        assert adapter.supports_batch
        assert not adapter.supports_stream
        assert stt.family_of("whisper-local") == "openai"


@pytest.mark.skipif(
    not os.environ.get("AUDIO_HARNESS_TEST_WHISPER"),
    reason="set AUDIO_HARNESS_TEST_WHISPER=1 to run the pinned local model (~3 GB first-time download)",
)
class TestWhisperLocalLive:
    """Live inference through the pinned mlx model."""

    async def test_transcribes_a_clip_end_to_end(self) -> None:
        clip = AudioClip(
            clip_id="silence",
            pcm=b"\x00\x00" * 16000,
            sample_rate=16000,
            duration_s=1.0,
            reference=None,
            language="en-US",
            source_path="<memory>",
        )
        adapter = stt.create("whisper-local")

        result = await adapter.transcribe_batch(clip)

        assert result.ok
        assert isinstance(result.text, str)
        assert result.total_s > 0
        assert result.raw["revision"], "the model revision must be recorded"
