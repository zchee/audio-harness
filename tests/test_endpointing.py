"""Tests for the EOU-aware endpointing bench (plan AC5).

The central invariant: only genuine end-of-utterance events count as
cutoffs. A segment final inside a pause is a decoding boundary — counting it
would rank vendors by transcript chunking, the exact confound the
``Partial.kind`` event model was introduced to remove.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from google.cloud.speech_v2.types import cloud_speech

from audio_harness.config import BenchmarkConfig
from audio_harness.endpointing import (
    EndpointSummary,
    measure_loopback_floor,
    render_endpointing_markdown,
    summarize_endpointing,
)
from audio_harness.metrics import partial_instability
from audio_harness.runner import read_stt_results, write_stt_results
from audio_harness.stt.assemblyai import _TurnHandler
from audio_harness.stt.base import StreamTimeline
from audio_harness.stt.deepgram import _handle_message as deepgram_handle
from audio_harness.stt.elevenlabs import _make_handler as elevenlabs_handler
from audio_harness.stt.google import _record_response
from audio_harness.stt.soniox import _TokenAccumulator
from audio_harness.stt.speechmatics import _handle_message as speechmatics_handle
from audio_harness.types import EventKind, Mode, Partial, SttResult


def _result(
    *,
    provider: str = "vendor",
    clip_id: str = "turn-1",
    partials: list[Partial] | None = None,
    pauses: list[list[float]] | None = None,
    speech_end_s: float | None = None,
    eou_source: str | None = None,
    endpoint_config: dict | None = None,
    ws_rtt_s: float | None = None,
    error: str | None = None,
) -> SttResult:
    """Build a streaming result shaped like the runner records it."""
    result = SttResult(
        provider=provider,
        clip_id=clip_id,
        mode=Mode.STREAM,
        text="hello there",
        audio_s=6.0,
        error=error,
    )
    result.partials = partials or []
    result.raw = {"reference": "hello there", "language": "en-US"}
    if pauses is not None:
        result.raw["pauses"] = pauses
    if speech_end_s is not None:
        result.raw["speech_end_s"] = speech_end_s
    if eou_source is not None:
        result.raw["eou_source"] = eou_source
    if endpoint_config is not None:
        result.raw["endpoint_config"] = endpoint_config
    if ws_rtt_s is not None:
        result.raw["ws_rtt_s"] = ws_rtt_s
    return result


class TestEventModel:
    """Partial.kind defaults must preserve pre-migration behaviour."""

    def test_kind_defaults_from_finality(self) -> None:
        tests = {
            "interim stays interim": {
                "partial": Partial(t_s=0.1, text="a", is_final=False),
                "expected": EventKind.INTERIM,
            },
            "final becomes segment_final": {
                "partial": Partial(t_s=0.2, text="a b", is_final=True),
                "expected": EventKind.SEGMENT_FINAL,
            },
            "explicit kind wins": {
                "partial": Partial(
                    t_s=0.3, text="", is_final=False, kind=EventKind.EOU
                ),
                "expected": EventKind.EOU,
            },
        }
        for name, case in tests.items():
            assert case["partial"].kind == case["expected"], name

    def test_timeline_accepts_bare_eou_markers_only(self) -> None:
        timeline = StreamTimeline()
        timeline.start()
        timeline.record("", is_final=False)
        timeline.record("", is_final=False, kind=EventKind.EOU)
        timeline.record("hello", is_final=False)

        kinds = [p.kind for p in timeline.partials]
        assert kinds == [EventKind.EOU, EventKind.INTERIM], (
            "blank keepalives stay dropped; bare EOU markers are kept"
        )
        assert timeline.ttft_s == timeline.partials[1].t_s, (
            "a textless EOU marker must not register as the first token"
        )

    def test_churn_ignores_bare_eou_markers(self) -> None:
        partials = [
            Partial(t_s=0.1, text="he", is_final=False),
            Partial(t_s=0.2, text="", is_final=False, kind=EventKind.EOU),
            Partial(t_s=0.3, text="hell", is_final=False),
        ]
        assert partial_instability(partials) == 0.0, (
            "a textless marker between two growing partials is not a rewrite"
        )


class TestFalseCutoff:
    """AC5: cutoffs come from EOU events only, never segment finals."""

    def test_segment_finals_inside_pauses_are_not_cutoffs(self) -> None:
        partials = [
            Partial(t_s=1.0, text="hello", is_final=False),
            # A segment final INSIDE the labeled pause — the confound.
            Partial(t_s=2.2, text="hello", is_final=True),
            Partial(t_s=3.0, text="hello there", is_final=False),
            # The vendor's EOU after the true end of speech: correct.
            Partial(t_s=5.4, text="", is_final=False, kind=EventKind.EOU),
        ]
        (summary,) = summarize_endpointing(
            [
                _result(
                    partials=partials,
                    pauses=[[2.0, 2.6]],
                    speech_end_s=5.0,
                    eou_source="end_of_turn",
                )
            ],
            "en-US",
        )

        assert summary.hold_pauses == 1
        assert summary.cut_pauses == 0, (
            "the segment final at 2.2s sits inside the pause but is a "
            "decoding boundary, not an end-of-utterance decision"
        )
        assert summary.false_cutoff_rate == 0.0
        assert summary.finals_in_pauses == 1, (
            "it still appears in the descriptive final-event count"
        )
        assert summary.eou_latency_s == [pytest.approx(0.4)]
        assert summary.premature_eous == 0

    def test_eou_inside_a_pause_is_a_cutoff(self) -> None:
        partials = [
            Partial(t_s=2.2, text="", is_final=False, kind=EventKind.EOU),
            Partial(t_s=5.3, text="", is_final=False, kind=EventKind.EOU),
        ]
        (summary,) = summarize_endpointing(
            [
                _result(
                    partials=partials,
                    pauses=[[2.0, 2.6], [3.5, 3.9]],
                    speech_end_s=5.0,
                    eou_source="speech_final",
                )
            ],
            "en-US",
        )

        assert summary.hold_pauses == 2
        assert summary.cut_pauses == 1
        assert summary.false_cutoff_rate == pytest.approx(0.5)
        assert summary.premature_eous == 1, "the 2.2s EOU precedes speech end"
        assert summary.eou_latency_s == [pytest.approx(0.3)]

    def test_latency_uses_first_eou_after_speech_end(self) -> None:
        partials = [
            Partial(t_s=5.2, text="", is_final=False, kind=EventKind.EOU),
            Partial(t_s=5.9, text="", is_final=False, kind=EventKind.EOU),
        ]
        (summary,) = summarize_endpointing(
            [_result(partials=partials, speech_end_s=5.0, eou_source="x")],
            "en-US",
        )
        assert summary.eou_latency_s == [pytest.approx(0.2)]
        assert summary.within_budget(0.3) == 1.0
        assert summary.within_budget(0.1) == 0.0

    def test_failures_are_counted_but_not_scored(self) -> None:
        (summary,) = summarize_endpointing(
            [_result(partials=[], pauses=[[1.0, 2.0]], error="timeout after 180s")],
            "en-US",
        )
        assert summary.clips == 1
        assert summary.failures == 1
        assert summary.hold_pauses == 0


class TestRankedScope:
    """Only vendors with a captured EOU signal may be ranked."""

    def test_lanes_split_by_eou_source(self) -> None:
        summaries = summarize_endpointing(
            [
                _result(provider="capable", eou_source="end_of_turn"),
                _result(provider="opaque"),
            ],
            "en-US",
        )
        by_provider = {s.provider: s for s in summaries}
        assert by_provider["capable"].ranked
        assert not by_provider["opaque"].ranked

    def test_markdown_separates_ranked_and_descriptive(self) -> None:
        summaries = [
            EndpointSummary(
                provider="capable",
                mode="stream",
                language="en-US",
                eou_source="end_of_turn",
                clips=3,
                hold_pauses=4,
                cut_pauses=1,
                eou_latency_s=[0.2, 0.3, 0.5],
                ws_rtt_s=[0.02],
            ),
            EndpointSummary(
                provider="opaque",
                mode="stream",
                language="en-US",
                clips=3,
                finals_in_pauses=2,
                ws_rtt_s=[0.05],
            ),
        ]
        markdown = render_endpointing_markdown(
            summaries, floors={"capable": 0.021, "opaque": 0.021}
        )

        ranked, _, descriptive = markdown.partition("no captured EOU signal")
        assert "capable" in ranked
        assert "capable" not in descriptive
        assert "opaque" in descriptive
        assert "25.0%" in ranked, "1 of 4 pauses cut"
        assert "RTT p50" in ranked
        assert "Floor" in ranked, "AC5: loopback floor column present"
        assert "not ranked" in markdown


class TestJsonlRoundTrip:
    """Every endpointing input must survive the results JSONL."""

    def test_summaries_match_after_reload(self, tmp_path: Path) -> None:
        partials = [
            Partial(t_s=2.2, text="hello", is_final=True),
            Partial(t_s=5.4, text="", is_final=False, kind=EventKind.EOU),
        ]
        results = [
            _result(
                partials=partials,
                pauses=[[2.0, 2.6]],
                speech_end_s=5.0,
                eou_source="end_of_turn",
                endpoint_config={"min_turn_silence": 400},
                ws_rtt_s=0.021,
            )
        ]
        path = write_stt_results(results, tmp_path)
        (reloaded,) = summarize_endpointing(read_stt_results(path), "en-US")
        (direct,) = summarize_endpointing(results, "en-US")

        assert reloaded == direct, (
            "pause labels, speech end, event kinds, RTT and knob config must "
            "all survive write_stt_results so saved runs stay re-scoreable"
        )
        assert reloaded.endpoint_config == {"min_turn_silence": 400}

    def test_legacy_records_without_kind_stay_segment_finals(
        self, tmp_path: Path
    ) -> None:
        results = [_result(partials=[Partial(t_s=2.2, text="hello", is_final=True)])]
        path = write_stt_results(results, tmp_path)
        raw = path.read_text(encoding="utf-8").replace(', "kind": "segment_final"', "")
        path.write_text(raw, encoding="utf-8")

        (loaded,) = read_stt_results(path)
        assert loaded.partials[0].kind == EventKind.SEGMENT_FINAL, (
            "pre-migration records derive kind from is_final"
        )


class TestAdapterEventMapping:
    """Each adapter must map its vendor's native signals, verified against
    the vendor protocol docs (see the endpointing lane audit)."""

    def _kinds(self, timeline: StreamTimeline) -> list[str]:
        return [str(p.kind) for p in timeline.partials]

    def test_deepgram_speech_final_and_utterance_end(self) -> None:
        timeline = StreamTimeline()
        timeline.start()
        frames = [
            {
                "type": "Results",
                "is_final": False,
                "channel": {"alternatives": [{"transcript": "hi"}]},
            },
            {
                "type": "Results",
                "is_final": True,
                "channel": {"alternatives": [{"transcript": "hi there"}]},
            },
            {
                "type": "Results",
                "is_final": True,
                "speech_final": True,
                "channel": {"alternatives": [{"transcript": "bye"}]},
            },
            {"type": "UtteranceEnd", "last_word_end": 2.4},
            {"type": "UtteranceEnd", "last_word_end": -1},
        ]
        for frame in frames:
            deepgram_handle(frame, timeline)

        assert self._kinds(timeline) == [
            "interim",
            "segment_final",
            "eou",
            "eou",
        ], "plain is_final stays segment_final; stale UtteranceEnd is dropped"

    def test_assemblyai_end_of_turn_deduplicates_formatted_final(self) -> None:
        timeline = StreamTimeline()
        timeline.start()
        handler = _TurnHandler()
        frames = [
            {
                "type": "Turn",
                "turn_order": 0,
                "transcript": "hello",
                "end_of_turn": False,
            },
            {
                "type": "Turn",
                "turn_order": 0,
                "transcript": "hello.",
                "end_of_turn": True,
                "turn_is_formatted": False,
            },
            {
                "type": "Turn",
                "turn_order": 0,
                "transcript": "Hello.",
                "end_of_turn": True,
                "turn_is_formatted": True,
            },
            {
                "type": "Turn",
                "turn_order": 1,
                "transcript": "again",
                "end_of_turn": True,
            },
        ]
        for frame in frames:
            handler(frame, timeline)

        eous = [p for p in timeline.partials if p.kind == EventKind.EOU]
        assert len(eous) == 2, "one EOU per turn_order, at the first decision"
        assert eous[0].text == "hello."

    def test_speechmatics_end_of_utterance_is_a_bare_marker(self) -> None:
        timeline = StreamTimeline()
        timeline.start()
        speechmatics_handle(
            {"message": "AddTranscript", "metadata": {"transcript": "hi"}},
            timeline,
        )
        speechmatics_handle({"message": "EndOfUtterance", "metadata": {}}, timeline)

        assert self._kinds(timeline) == ["segment_final", "eou"]
        assert timeline.partials[1].text == ""

    def test_soniox_end_token_is_control_flow_not_transcript(self) -> None:
        timeline = StreamTimeline()
        timeline.start()
        accumulator = _TokenAccumulator()
        accumulator(
            {
                "tokens": [
                    {"text": "Hello", "is_final": True},
                    {"text": "<end>", "is_final": True},
                ]
            },
            timeline,
        )

        assert accumulator.finalized == "Hello", "<end> must never enter text"
        assert [str(p.kind) for p in timeline.partials] == [
            "segment_final",
            "eou",
        ]

    def test_elevenlabs_commit_kind_follows_strategy(self) -> None:
        tests = {"vad": EventKind.EOU, "manual": EventKind.SEGMENT_FINAL}
        for strategy, expected in tests.items():
            timeline = StreamTimeline()
            timeline.start()
            handler = elevenlabs_handler(vad_commits=strategy == "vad")
            handler({"message_type": "committed_transcript", "text": "hi"}, timeline)
            assert timeline.partials[0].kind == expected, strategy

    def test_google_speech_activity_end_event(self) -> None:
        timeline = StreamTimeline()
        timeline.start()
        _record_response(
            cloud_speech.StreamingRecognizeResponse(
                speech_event_type=(
                    cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END
                )
            ),
            timeline,
        )
        assert self._kinds(timeline) == ["eou"]


class TestLoopbackFloor:
    """The harness must know its own contribution to event latency."""

    async def test_floor_is_small_and_positive(self) -> None:
        floor = await measure_loopback_floor(20, clip_s=0.2, rounds=2)
        assert 0.0 <= floor < 0.25, (
            "a local echo answered instantly; anything above 250ms means the "
            "client stack itself is broken"
        )


class TestLaneConfig:
    """The endpointing lane must pin knobs and stream in real time."""

    def test_config_wires_eou_knobs_for_every_optin_vendor(self) -> None:
        config = BenchmarkConfig.from_yaml("configs/stt-endpointing.yaml")
        options = {entry.name: entry.options for entry in config.stt}

        assert options["deepgram-nova3"]["utterance_end_ms"] == 1000
        assert (
            options["speechmatics-enhanced"]["end_of_utterance_silence_trigger"] == 0.7
        )
        assert options["soniox-rt-v5"]["enable_endpoint_detection"] is True
        assert options["google-chirp3"]["enable_voice_activity_events"] is True
        assert all(entry.modes == ["stream"] for entry in config.stt)
        assert config.run.realtime, "endpoint latency is meaningless unpaced"

    def test_sources_carry_span_labels(self) -> None:
        config = BenchmarkConfig.from_yaml("configs/stt-endpointing.yaml")
        sources = config.dataset.resolved_sources()
        assert all(s.silence_spans_column == "silence_spans" for s in sources)
        assert all(s.words_column == "words" for s in sources)


LIVE_FLAG = "AUDIO_HARNESS_TEST_ENDPOINTING_LIVE"


@pytest.mark.skipif(
    not __import__("os").environ.get(LIVE_FLAG)
    or not __import__("os").environ.get("DEEPGRAM_API_KEY"),
    reason=f"live smoke needs {LIVE_FLAG}=1 and DEEPGRAM_API_KEY "
    "(one ~6s clip, fractions of a cent)",
)
class TestLiveSmoke:
    """Minimal real-vendor pass: does the EOU capture read the real wire?

    One clip against one cheap vendor, per the team's testing policy. The
    clip is synthetic (tone-pause-tone-silence) with hand-labeled pauses, so
    no corpus download is needed; assertions are about plumbing — events
    captured, kinds recorded, JSONL round trip — never about the vendor's
    endpointing quality, which is the bench's job on real data.
    """

    async def test_deepgram_stream_carries_eou_events(self, tmp_path: Path) -> None:
        import math

        import numpy as np
        import soundfile as sf

        from audio_harness.audio import load_clip
        from audio_harness.config import ProviderConfig, RunConfig
        from audio_harness.runner import run_stt
        from audio_harness.types import AudioClip

        rate = 16000

        def tone(seconds: float) -> np.ndarray:
            t = np.arange(int(rate * seconds)) / rate
            return (0.25 * np.sin(2 * math.pi * 220 * t)).astype(np.float32)

        def silence(seconds: float) -> np.ndarray:
            return np.zeros(int(rate * seconds), dtype=np.float32)

        samples = np.concatenate([tone(1.0), silence(1.5), tone(1.0), silence(2.0)])
        sf.write(tmp_path / "clip.wav", samples, rate)
        base = load_clip(
            tmp_path / "clip.wav",
            clip_id="live-endpoint",
            reference=None,
            language="en-US",
        )
        clip = AudioClip(
            clip_id=base.clip_id,
            pcm=base.pcm,
            sample_rate=base.sample_rate,
            duration_s=base.duration_s,
            reference=base.reference,
            language=base.language,
            source_path=base.source_path,
            speech_end_s=3.5,
            pauses=((1.0, 2.5),),
        )

        config = BenchmarkConfig(
            stt=[
                ProviderConfig(
                    name="deepgram-nova3",
                    modes=["stream"],
                    options={"endpointing": 300, "utterance_end_ms": 1000},
                )
            ],
            run=RunConfig(
                repeats=1,
                warmup=0,
                timeout_s=60.0,
                settle_ms=0,
                output_dir=str(tmp_path),
            ),
        )
        results = await run_stt(config, [clip])
        assert len(results) == 1
        (result,) = results
        assert result.ok, result.error
        assert result.raw.get("eou_source") == "speech_final+utterance_end"
        assert result.raw.get("pauses") == [[1.0, 2.5]]
        assert isinstance(result.raw.get("endpoint_config"), dict)

        from audio_harness.runner import read_stt_results, write_stt_results

        path = write_stt_results(results, tmp_path)
        summaries = summarize_endpointing(read_stt_results(path), "en-US")
        (summary,) = summaries
        assert summary.ranked
        assert summary.hold_pauses == 1
        assert summary.false_cutoff_rate is not None
