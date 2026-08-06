"""SNR robustness matrix: the same speech under progressively worse noise.

A vendor ranking measured on clean audio says nothing about the call-center
acoustics a voice agent actually meets. This lane re-mixes a seeded subset of
real utterances at five active-speech SNR levels — {+20, +10, +5, 0, -5} dB —
and summarizes WER as a function of SNR, plus a single WER-vs-SNR AUC number
per provider and language so degradation curves can be ranked. An optional
telephony condition round-trips audio through 8 kHz to expose narrowband
sensitivity.

Two design points keep the matrix honest:

* **One noise draw per base clip, shared by every level.** The five mixtures
  of a clip differ only in noise gain, so a level-to-level WER delta is the
  SNR's doing — not a lucky quieter noise segment at one level.
* **Telephony is a resample round-trip (16 k → 8 k → 16 k), not native-8k
  streaming.** Adapters advertise the clip's rate to their vendor, so
  streaming at 8 kHz would measure each adapter's rate plumbing as much as
  the vendor's narrowband model; the round-trip isolates the bandwidth loss
  itself and stays identical across vendors. Native-8k lanes remain possible
  later, per adapter, once their rate handling is audited.

The SNR level rides the clip id (``snr+05-<base>``), following the synthetic
lane's convention, so saved runs regroup by condition without re-running
anything. Endpointing-vs-SNR uses the same grouping: finalize latency is
collected per level here, and once adapters emit ``EventKind.EOU`` events
(endpointing bench), those timestamps group by the identical prefix — no
schema change needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
import re

import numpy as np
import soxr

from .config import SourceConfig
from .metrics import ErrorCounts, percentile, score_pair
from .types import AudioClip, SttResult


SNR_LEVELS = (20.0, 10.0, 5.0, 0.0, -5.0)
"""Pre-registered matrix levels; the plan budgets exactly these five."""

SNR_KINDS = ("snr", "telephony")
"""Synthetic source kinds this module provides."""

TELEPHONY_RATE = 8000

_SNR_ID = re.compile(r"^snr([+-]\d{2})-")
_TELEPHONY_PREFIX = "tel8k-"


def snr_level_of(clip_id: str) -> float | None:
    """Recover the SNR level a clip id encodes, or ``None`` if not a mix."""
    match = _SNR_ID.match(clip_id)
    return float(match.group(1)) if match else None


def is_telephony(clip_id: str) -> bool:
    """Whether a clip id marks the 8 kHz telephony round-trip condition."""
    return clip_id.startswith(_TELEPHONY_PREFIX)


def synthesize_snr_source(source: SourceConfig, *, sample_rate: int = 16000) -> list[AudioClip]:
    """Generate the clips an SNR-matrix or telephony source describes.

    Args:
        source: Source with ``synthetic`` set to one of :data:`SNR_KINDS`.
            The base corpus (``parquet``/``manifest``), ``limit`` and
            ``sample_seed`` select the seeded subset; ``noise_dir`` supplies
            the MUSAN material for the matrix kind.
        sample_rate: Rate every clip is generated at.

    Returns:
        For ``snr``: every base clip at every level, base-major order. For
        ``telephony``: one narrowband round-trip per base clip.

    Raises:
        DatasetError: If the kind is unknown or the base corpus or noise
            material is unavailable.
    """
    # Imported lazily: synthetic.py reaches dataset.py, which dispatches back
    # into this module — top-level imports would be circular.
    from . import synthetic
    from .dataset import DatasetError

    kind = source.synthetic
    if kind not in SNR_KINDS:
        raise DatasetError(f"unknown SNR source kind {kind!r}; expected one of {', '.join(SNR_KINDS)}")

    bases = synthetic.base_clips(source, sample_rate=sample_rate)
    if kind == "telephony":
        return [_telephony_clip(base) for base in bases]

    seed = source.sample_seed if source.sample_seed is not None else synthetic.DEFAULT_SEED
    files = synthetic.noise_files(source.noise_dir)
    clips: list[AudioClip] = []
    for index, base in enumerate(bases):
        rng = np.random.default_rng([seed, index])
        path = files[int(rng.integers(len(files)))]
        speech = synthetic.pcm_to_float(base.pcm)
        noise = synthetic.noise_segment(synthetic.load_noise(path, base.sample_rate), len(speech), rng)
        if float(np.sqrt(np.mean(noise**2))) <= 0.0:
            raise DatasetError(f"noise file decodes to silence: {path}")
        for level in SNR_LEVELS:
            mixed = synthetic.mix_at_snr(speech, noise, snr_db=level, sample_rate=base.sample_rate)
            clips.append(
                synthetic.to_clip(
                    mixed,
                    clip_id=f"snr{int(level):+03d}-{base.clip_id}",
                    reference=base.reference,
                    language=base.language,
                    sample_rate=base.sample_rate,
                    source_path=base.source_path,
                )
            )
    return clips


def _telephony_clip(base: AudioClip) -> AudioClip:
    """Round-trip one clip through the 8 kHz telephony band."""
    # Same circularity as above: synthetic.py reaches back into this module.
    from . import synthetic

    samples = synthetic.pcm_to_float(base.pcm)
    narrow = soxr.resample(samples, base.sample_rate, TELEPHONY_RATE, quality="HQ")
    restored = soxr.resample(narrow, TELEPHONY_RATE, base.sample_rate, quality="HQ")
    return synthetic.to_clip(
        restored.astype(np.float32),
        clip_id=f"{_TELEPHONY_PREFIX}{base.clip_id}",
        reference=base.reference,
        language=base.language,
        sample_rate=base.sample_rate,
        source_path=base.source_path,
    )


@dataclass(slots=True)
class SnrSummary:
    """WER-vs-SNR behaviour of one provider/mode/language lane.

    Attributes:
        provider: Registry key of the adapter.
        mode: Transport mode the runs used.
        language: BCP-47 tag these runs were scored under.
        levels: Corpus-level edit counts per SNR level.
        finalize: Finalize latencies per SNR level — the raw material for
            endpointing-vs-SNR degradation. When the endpointing bench's
            ``EventKind.EOU`` events land, they group by the same clip-id
            prefix; nothing in this shape has to change.
        failures: Failed runs per SNR level; failure under noise is itself a
            robustness result, not noise in the data.
        telephony: Edit counts for the 8 kHz round-trip condition.
    """

    provider: str
    mode: str
    language: str
    levels: dict[float, ErrorCounts] = field(default_factory=dict)
    finalize: dict[float, list[float]] = field(default_factory=dict)
    failures: dict[float, int] = field(default_factory=dict)
    telephony: ErrorCounts | None = None

    def rate(self, level: float) -> float | None:
        """WER at one level, or ``None`` when nothing was scored there."""
        counts = self.levels.get(level)
        if counts is None or counts.reference_length == 0:
            return None
        return counts.rate

    @property
    def wer_auc(self) -> float | None:
        """Area under the WER-vs-SNR curve, normalized to the SNR span.

        Trapezoidal integral over the levels that have data, divided by the
        span, so the number reads as an average WER across the noise range —
        lower is better, and a provider that only degrades below 0 dB scores
        visibly better than one already failing at +10. Needs at least two
        levels; a single point has no curve.
        """
        points = sorted((level, rate) for level in self.levels if (rate := self.rate(level)) is not None)
        if len(points) < 2:
            return None
        area = 0.0
        for (x0, y0), (x1, y1) in pairwise(points):
            area += (y0 + y1) / 2.0 * (x1 - x0)
        span = points[-1][0] - points[0][0]
        return area / span

    @property
    def finalize_degradation_s(self) -> float | None:
        """Finalize-p50 shift from the cleanest to the noisiest level.

        Positive means the vendor takes longer to commit under noise — the
        turn-taking cost of the acoustics, invisible in WER.
        """
        with_data = sorted(level for level, values in self.finalize.items() if values)
        if len(with_data) < 2:
            return None
        clean = percentile(self.finalize[with_data[-1]], 50)
        noisy = percentile(self.finalize[with_data[0]], 50)
        if clean is None or noisy is None:
            return None
        return noisy - clean


def summarize_snr(results: list[SttResult], language: str) -> list[SnrSummary]:
    """Aggregate SNR-matrix and telephony runs per provider/mode/language.

    Non-SNR clips pass through untouched — this reads the same results file
    as the main summary and picks out its own conditions by clip-id prefix.

    Args:
        results: Every run to aggregate.
        language: Fallback BCP-47 tag for results that recorded none.

    Returns:
        One summary per lane that had SNR or telephony clips.
    """
    summaries: dict[tuple[str, str, str], SnrSummary] = {}

    for result in results:
        level = snr_level_of(result.clip_id)
        telephony = is_telephony(result.clip_id)
        if level is None and not telephony:
            continue

        recorded = result.raw.get("language")
        clip_language = recorded if isinstance(recorded, str) and recorded else language
        key = (result.provider, str(result.mode), clip_language)
        summary = summaries.setdefault(
            key,
            SnrSummary(provider=result.provider, mode=str(result.mode), language=clip_language),
        )

        reference = result.raw.get("reference")
        scored = (
            score_pair(reference, result.text, clip_language)
            if result.ok and isinstance(reference, str) and reference
            else None
        )

        if telephony:
            if scored is not None:
                summary.telephony = scored if summary.telephony is None else summary.telephony + scored
            continue

        assert level is not None
        if not result.ok:
            summary.failures[level] = summary.failures.get(level, 0) + 1
            continue
        if scored is not None:
            existing = summary.levels.get(level)
            summary.levels[level] = scored if existing is None else existing + scored
        if result.finalize_s is not None:
            summary.finalize.setdefault(level, []).append(result.finalize_s)

    return sorted(summaries.values(), key=lambda s: (s.language, s.provider, s.mode))


def render_snr_markdown(summaries: list[SnrSummary]) -> str:
    """Render the WER-vs-SNR table as GitHub-flavoured markdown."""
    if not summaries:
        return "_No SNR-matrix results._"

    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.2f}%"

    headers = (
        ["Provider", "Mode", "Lang"]
        + [f"WER {int(level):+d} dB" for level in SNR_LEVELS]
        + ["WER AUC", "Fin Δ p50", "8 kHz WER", "Fail"]
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for summary in summaries:
        degradation = summary.finalize_degradation_s
        telephony = summary.telephony
        cells = (
            [summary.provider, summary.mode, summary.language]
            + [pct(summary.rate(level)) for level in SNR_LEVELS]
            + [
                pct(summary.wer_auc),
                "—" if degradation is None else f"{degradation:+.3f}s",
                ("—" if telephony is None or telephony.reference_length == 0 else pct(telephony.rate)),
                str(sum(summary.failures.values())),
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend((
        "",
        "_WER AUC: trapezoidal mean WER across the SNR range (lower is "
        "better). Fin Δ: finalize-p50 shift from cleanest to noisiest level. "
        "8 kHz: telephony resample round-trip._",
    ))
    return "\n".join(lines)


def estimated_matrix_cost(*, clips: int, mean_clip_s: float, usd_per_hour: float) -> float:
    """Estimated spend for one vendor lane over the full matrix.

    Kept as code, not a comment, so the config header's number can be
    recomputed instead of rotting.
    """
    conditions = len(SNR_LEVELS) + 1  # five mixes plus the telephony pass
    return clips * conditions * mean_clip_s / 3600.0 * usd_per_hour
