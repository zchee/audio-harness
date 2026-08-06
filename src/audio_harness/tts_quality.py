"""Distill-MOS perceptual regression guardrail for synthesized TTS audio.

Distill-MOS predicts a single-utterance mean opinion score (MOS) from raw
audio with no reference signal needed, which makes it useful as a regression
tripwire: score a saved run's audio, compare each provider's mean against the
last time that provider was scored, and flag a drop of more than
:data:`REGRESSION_THRESHOLD` points.

Single-utterance MOS predictors collapse out-of-domain (TTSDS2,
arXiv:2506.19441), so this module never produces a comparative ranking across
providers — only a per-provider drift signal against its own recorded
baseline. Callers must label output accordingly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import soundfile as sf
import soxr

from .runner import read_tts_results


DISTILL_MOS_SAMPLE_RATE = 16000
"""Distill-MOS's fixed input rate; other rates are resampled before scoring."""

REGRESSION_THRESHOLD = 0.3
"""MOS points a provider must drop, versus its own baseline, to alert."""

ScoreFn = Callable[[Path], float]
"""A function mapping an audio file path to a Distill-MOS score."""


@dataclass(slots=True, frozen=True)
class ClipScore:
    """One scored audio file.

    Attributes:
        path: Audio file that was scored.
        provider: Registry key the file was attributed to, or "" if it could
            not be determined.
        mos: Predicted mean opinion score, nominally in [1, 5].
    """

    path: str
    provider: str
    mos: float


@dataclass(slots=True, frozen=True)
class ProviderMosSummary:
    """Aggregated Distill-MOS scores for one provider.

    Attributes:
        provider: Registry key of the adapter.
        clips: Number of audio files scored.
        mean_mos: Arithmetic mean of every clip's score.
        baseline_mos: This provider's previously recorded mean, or ``None``
            when no baseline exists yet.
    """

    provider: str
    clips: int
    mean_mos: float
    baseline_mos: float | None = None

    @property
    def delta(self) -> float | None:
        """``mean_mos - baseline_mos``, or ``None`` without a baseline."""
        if self.baseline_mos is None:
            return None
        return self.mean_mos - self.baseline_mos

    @property
    def alert(self) -> bool:
        """Whether this provider dropped more than the regression threshold.

        This is a tripwire, not a quality judgement: it only fires against
        the provider's own history, never against another provider's score.
        """
        delta = self.delta
        return delta is not None and delta < -REGRESSION_THRESHOLD


def score_waveform(samples: np.ndarray, sample_rate: int, *, model: Any = None) -> float:
    """Score mono float32 PCM with Distill-MOS.

    Args:
        samples: Mono float32 samples in [-1, 1].
        sample_rate: Sample rate of ``samples``.
        model: Loaded ``ConvTransformerSQAModel``, or ``None`` to use the
            cached default instance.

    Returns:
        Predicted MOS, nominally in [1, 5].
    """
    import torch  # Lazy: the optional dependency is only needed to score.

    if sample_rate != DISTILL_MOS_SAMPLE_RATE:
        samples = soxr.resample(samples, sample_rate, DISTILL_MOS_SAMPLE_RATE, quality="HQ")
    net = model if model is not None else _load_model()
    x = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        y = net(x)
    return float(y.item())


def score_file(path: str | Path, *, model: Any = None) -> float:
    """Decode an audio file and score it with Distill-MOS.

    Args:
        path: Audio file in any format libsndfile can read.
        model: Loaded ``ConvTransformerSQAModel``, or ``None`` to use the
            cached default instance.

    Returns:
        Predicted MOS, nominally in [1, 5].
    """
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return score_waveform(mono, rate, model=model)


@cache
def _load_model() -> Any:
    """Load and cache the pinned Distill-MOS checkpoint.

    Imported lazily so the module works without the optional dependency
    installed until a caller actually asks to score audio.
    """
    try:
        import distillmos
    except ImportError as exc:
        raise RuntimeError(
            "tts_quality: distillmos is not installed. Install the optional "
            "dependency group: uv sync --extra guardrail-mos"
        ) from exc
    model = distillmos.ConvTransformerSQAModel()
    model.eval()
    return model


def score_directory(directory: str | Path, *, score_fn: ScoreFn | None = None) -> list[ClipScore]:
    """Score every WAV file directly inside ``directory``.

    Providers are recovered from the ``{provider}-{mode}-{prompt_id}.wav``
    naming convention written by
    :func:`audio_harness.runner.write_tts_results`. A name that does not
    contain a recognized mode marker is still scored, just attributed to an
    empty provider.

    Args:
        directory: Directory holding saved TTS audio.
        score_fn: Per-file scorer to use instead of live Distill-MOS
            inference; tests inject a stub here to avoid loading the model.

    Returns:
        One score per WAV file, in filename order.
    """
    score = score_fn or score_file
    return [
        ClipScore(
            path=str(wav_path),
            provider=_provider_from_filename(wav_path.name),
            mos=score(wav_path),
        )
        for wav_path in sorted(Path(directory).glob("*.wav"))
    ]


def score_results_file(path: str | Path, *, score_fn: ScoreFn | None = None) -> list[ClipScore]:
    """Score the audio referenced by a saved ``tts-results.jsonl``.

    Only records that succeeded and carry a recorded ``audio_path`` (the run
    used ``audio-harness tts --save-audio``) can be scored; a run without
    saved audio has nothing for a perceptual guardrail to measure. A
    referenced file that no longer exists on disk is skipped rather than
    treated as an error, since the path was recorded relative to wherever the
    harness ran and may have moved.

    Args:
        path: JSONL file written by
            :func:`audio_harness.runner.write_tts_results`.
        score_fn: Per-file scorer to use instead of live Distill-MOS
            inference; tests inject a stub here to avoid loading the model.

    Returns:
        One score per scoreable result.
    """
    score = score_fn or score_file
    scores: list[ClipScore] = []
    for result in read_tts_results(path):
        audio_path = result.raw.get("audio_path")
        if not result.ok or not isinstance(audio_path, str) or not audio_path:
            continue
        file = Path(audio_path)
        if not file.is_file():
            continue
        scores.append(ClipScore(path=str(file), provider=result.provider, mos=score(file)))
    return scores


def _provider_from_filename(name: str) -> str:
    """Recover the provider key from a ``{provider}-{mode}-{prompt_id}.wav`` name."""
    stem = name.removesuffix(".wav")
    for mode in ("batch", "stream"):
        marker = f"-{mode}-"
        if marker in stem:
            return stem.split(marker, 1)[0]
    return ""


def summarize(scores: Iterable[ClipScore]) -> list[ProviderMosSummary]:
    """Aggregate per-clip scores into one mean per provider.

    Args:
        scores: Clip scores, as returned by :func:`score_directory` or
            :func:`score_results_file`.

    Returns:
        Summaries sorted by provider name, without baseline data attached.
    """
    totals: dict[str, list[float]] = {}
    for score in scores:
        totals.setdefault(score.provider, []).append(score.mos)
    return [
        ProviderMosSummary(provider=provider, clips=len(values), mean_mos=sum(values) / len(values))
        for provider, values in sorted(totals.items())
    ]


def load_baseline(path: str | Path) -> dict[str, float]:
    """Load the provider -> mean-MOS baseline JSON, or ``{}`` if absent.

    Args:
        path: Baseline file previously written by :func:`save_baseline`.

    Returns:
        Mapping of provider key to its recorded mean MOS.
    """
    file = Path(path)
    if not file.is_file():
        return {}
    return {str(key): float(value) for key, value in orjson.loads(file.read_bytes()).items()}


def save_baseline(path: str | Path, summaries: Iterable[ProviderMosSummary]) -> Path:
    """Write the provider -> mean-MOS baseline JSON.

    Args:
        path: File to write.
        summaries: Summaries whose ``mean_mos`` becomes the new baseline.

    Returns:
        ``path``, for chaining.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    payload = {summary.provider: summary.mean_mos for summary in summaries}
    file.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return file


def apply_baseline(summaries: list[ProviderMosSummary], baseline: dict[str, float]) -> list[ProviderMosSummary]:
    """Attach recorded baseline means to freshly computed summaries.

    Args:
        summaries: Summaries from :func:`summarize`, with no baseline set.
        baseline: Mapping loaded by :func:`load_baseline`.

    Returns:
        New summaries carrying ``baseline_mos`` where the provider has one.
    """
    return [
        ProviderMosSummary(
            provider=summary.provider,
            clips=summary.clips,
            mean_mos=summary.mean_mos,
            baseline_mos=baseline.get(summary.provider),
        )
        for summary in summaries
    ]


def run_guardrail(
    source: str | Path,
    *,
    baseline_path: str | Path,
    update_baseline: bool = False,
    score_fn: ScoreFn | None = None,
) -> list[ProviderMosSummary]:
    """Score saved TTS audio and compare it against the recorded baseline.

    Args:
        source: Directory of saved WAV files, or a ``tts-results.jsonl``.
        baseline_path: JSON file mapping provider -> mean MOS from the last
            time this guardrail ran.
        update_baseline: When ``True``, overwrite ``baseline_path`` with this
            run's means after scoring — use once a drop has been triaged and
            the new level is accepted, not on every run.
        score_fn: Per-file scorer to use instead of live Distill-MOS
            inference; tests inject a stub here to avoid loading the model.

    Returns:
        One summary per provider found in ``source``, each carrying its
        baseline comparison.
    """
    source = Path(source)
    scores = (
        score_results_file(source, score_fn=score_fn)
        if source.is_file()
        else score_directory(source, score_fn=score_fn)
    )
    summaries = apply_baseline(summarize(scores), load_baseline(baseline_path))
    if update_baseline:
        save_baseline(baseline_path, summaries)
    return summaries
