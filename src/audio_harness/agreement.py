"""Cross-model STT transcript agreement metrics and reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import orjson

from .metrics import ZERO_COUNTS, ErrorCounts, score_pair
from .runner import read_stt_results
from .types import SttResult


type AgreementLane = tuple[str, str]
type AgreementPair = tuple[AgreementLane, AgreementLane]


@dataclass(slots=True, frozen=True)
class AgreementCell:
    """One pair's corpus-level bidirectional edit counts.

    Attributes:
        counts: Both directional edit counts pooled across overlapping clips.
        clips: Overlapping usable clips included in the counts.
    """

    counts: ErrorCounts
    clips: int

    @property
    def disagreement_rate(self) -> float | None:
        """Pooled bidirectional errors per reference token."""
        if self.counts.reference_length == 0:
            return None
        return self.counts.rate


@dataclass(slots=True, frozen=True)
class AgreementMatrix:
    """Pairwise agreement cells for one corpus slice.

    Attributes:
        cells: Comparisons keyed by two distinct provider/mode lanes.
    """

    cells: dict[AgreementPair, AgreementCell]

    def cell(self, lane_a: AgreementLane, lane_b: AgreementLane) -> AgreementCell | None:
        """Return a pair's cell independent of lane order.

        Args:
            lane_a: First provider/mode lane.
            lane_b: Second provider/mode lane.

        Returns:
            The comparison, or ``None`` for a diagonal or unknown pair.
        """
        if lane_a == lane_b:
            return None
        return self.cells.get((lane_a, lane_b)) or self.cells.get((lane_b, lane_a))


@dataclass(slots=True, frozen=True)
class AgreementReport:
    """Overall and per-language cross-model disagreement matrices.

    Attributes:
        lanes: Provider/mode lanes in stable matrix order.
        overall: Matrix aggregated across every scored clip.
        per_language: The same matrix split by each clip's own language.
    """

    lanes: tuple[AgreementLane, ...]
    overall: AgreementMatrix
    per_language: dict[str, AgreementMatrix]


def load_agreement_runs(run_dirs: Sequence[str | Path]) -> dict[AgreementLane, list[SttResult]]:
    """Load completed STT runs and group their results into lanes.

    Each input may be a run directory containing ``stt-results.jsonl`` or a
    direct path to that file. At least two run paths are required; having
    fewer cannot establish cross-run agreement.

    Args:
        run_dirs: Completed STT run directories or result-file paths.

    Returns:
        Results grouped by ``(provider, mode)`` lane.

    Raises:
        ValueError: If fewer than two run paths are provided.
        FileNotFoundError: If any run lacks its results file.
    """
    paths = tuple(Path(path) for path in run_dirs)
    if len(paths) < 2:
        raise ValueError("agreement needs at least two run directories or result files")

    runs: dict[AgreementLane, list[SttResult]] = {}
    for path in paths:
        results_path = path if path.suffix == ".jsonl" else path / "stt-results.jsonl"
        try:
            results = read_stt_results(results_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"agreement results file not found: {results_path}") from exc
        for result in results:
            runs.setdefault((result.provider, str(result.mode)), []).append(result)
    return runs


def compute_agreement(runs: Mapping[AgreementLane, Sequence[SttResult]]) -> AgreementReport:
    """Compute symmetric corpus-level transcript disagreement matrices.

    For each usable overlapping clip, the transcript pair is scored in both
    directions with :func:`score_pair`. The two directional counts are added,
    then pooled across clips before division. This removes the privileged
    reference lane while retaining corpus weighting by transcript length.

    Errored results and empty transcripts are excluded. If both lanes record a
    non-empty language for a clip, the values must agree; a missing language
    falls back to the other lane and then to ``und``. Duplicate clip ids within
    a lane use the last usable result, matching ordered run supersession.

    Args:
        runs: Results grouped by provider/mode lane.

    Returns:
        Overall and per-language pairwise matrices.

    Raises:
        ValueError: If fewer than two lanes exist or overlapping results give
            conflicting languages for the same clip.
    """
    lanes = tuple(sorted(runs))
    if len(lanes) < 2:
        raise ValueError("agreement needs at least two lanes")

    usable = {lane: _usable_by_clip(runs[lane]) for lane in lanes}
    pairs = tuple(combinations(lanes, 2))
    overall_cells: dict[AgreementPair, AgreementCell] = {}
    language_cells: dict[str, dict[AgreementPair, AgreementCell]] = {}

    for pair in pairs:
        left_lane, right_lane = pair
        left_results = usable[left_lane]
        right_results = usable[right_lane]
        counts = ZERO_COUNTS
        clips = 0
        per_language: dict[str, AgreementCell] = {}
        for clip_id in sorted(left_results.keys() & right_results.keys()):
            left = left_results[clip_id]
            right = right_results[clip_id]
            language = _clip_language(left, right)
            bidirectional = score_pair(left.text, right.text, language) + score_pair(right.text, left.text, language)
            counts = counts + bidirectional
            clips += 1
            previous = per_language.get(language, AgreementCell(ZERO_COUNTS, 0))
            per_language[language] = AgreementCell(previous.counts + bidirectional, previous.clips + 1)

        overall_cells[pair] = AgreementCell(counts, clips)
        for language, cell in per_language.items():
            language_cells.setdefault(language, {})[pair] = cell

    for cells in language_cells.values():
        for pair in pairs:
            cells.setdefault(pair, AgreementCell(ZERO_COUNTS, 0))

    return AgreementReport(
        lanes=lanes,
        overall=AgreementMatrix(overall_cells),
        per_language={language: AgreementMatrix(cells) for language, cells in sorted(language_cells.items())},
    )


def render_agreement_markdown(report: AgreementReport) -> str:
    """Render overall and per-language disagreement matrices as Markdown.

    Args:
        report: Agreement report produced by :func:`compute_agreement`.

    Returns:
        A complete Markdown document. Cells show disagreement and the number
        of overlapping clips; diagonal and zero-overlap cells use an em dash.
    """
    lines = [
        "# Cross-model agreement",
        "",
        (
            "Pooled bidirectional WER/CER disagreement: each clip is scored in both directions, "
            "then edit counts are summed before division. Lower is better."
        ),
        "",
        "## Overall",
        "",
        _render_matrix(report.overall, report.lanes),
    ]
    for language, matrix in report.per_language.items():
        lines.extend(["", f"## Language: {language}", "", _render_matrix(matrix, report.lanes)])
    return "\n".join(lines)


def write_agreement_report(report: AgreementReport, output_dir: str | Path) -> tuple[Path, Path]:
    """Write ``agreement.md`` and ``agreement.json`` into a directory.

    Args:
        report: Agreement report to serialize.
        output_dir: Destination directory, created when necessary.

    Returns:
        ``(markdown_path, json_path)``.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / "agreement.md"
    json_path = directory / "agreement.json"
    markdown_path.write_text(render_agreement_markdown(report) + "\n", encoding="utf-8")
    json_path.write_bytes(orjson.dumps(_report_payload(report), option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
    return markdown_path, json_path


def _usable_by_clip(results: Sequence[SttResult]) -> dict[str, SttResult]:
    """Index the last successful non-empty result for each clip."""
    usable: dict[str, SttResult] = {}
    for result in results:
        if result.ok and result.text.strip():
            usable[result.clip_id] = result
    return usable


def _clip_language(left: SttResult, right: SttResult) -> str:
    """Resolve one clip's language and reject contradictory metadata."""
    left_value = left.raw.get("language")
    right_value = right.raw.get("language")
    left_language = left_value if isinstance(left_value, str) and left_value else None
    right_language = right_value if isinstance(right_value, str) and right_value else None
    if left_language is not None and right_language is not None and left_language != right_language:
        raise ValueError(f"language mismatch for clip {left.clip_id!r}: {left_language!r} != {right_language!r}")
    return left_language or right_language or "und"


def _lane_label(lane: AgreementLane) -> str:
    """Format a provider/mode lane for Markdown."""
    provider, mode = lane
    return f"{provider} ({mode})".replace("|", "\\|")


def _render_matrix(matrix: AgreementMatrix, lanes: tuple[AgreementLane, ...]) -> str:
    """Render one symmetric matrix as a Markdown table."""
    labels = [_lane_label(lane) for lane in lanes]
    lines = [
        "| Lane | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---" for _ in lanes) + " |",
    ]
    for lane, label in zip(lanes, labels, strict=True):
        cells = [_render_cell(matrix.cell(lane, other)) for other in lanes]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_cell(cell: AgreementCell | None) -> str:
    """Format one matrix cell with its overlap count."""
    if cell is None or cell.disagreement_rate is None:
        return "—"
    return f"{cell.disagreement_rate * 100:.2f}% (n={cell.clips})"


def _report_payload(report: AgreementReport) -> dict[str, object]:
    """Convert a report into an orjson-compatible object."""
    return {
        "lanes": [{"provider": provider, "mode": mode} for provider, mode in report.lanes],
        "overall": _matrix_payload(report.overall, report.lanes),
        "per_language": {
            language: _matrix_payload(matrix, report.lanes) for language, matrix in report.per_language.items()
        },
    }


def _matrix_payload(matrix: AgreementMatrix, lanes: tuple[AgreementLane, ...]) -> dict[str, object]:
    """Serialize rates, clip counts and edit-count evidence for one matrix."""
    rates: list[list[float | None]] = []
    clip_counts: list[list[int | None]] = []
    for lane in lanes:
        rate_row: list[float | None] = []
        count_row: list[int | None] = []
        for other in lanes:
            cell = matrix.cell(lane, other)
            rate_row.append(None if cell is None else cell.disagreement_rate)
            count_row.append(None if cell is None else cell.clips)
        rates.append(rate_row)
        clip_counts.append(count_row)

    pairs = []
    for lane_a, lane_b in combinations(lanes, 2):
        cell = matrix.cell(lane_a, lane_b)
        if cell is None:
            continue
        pairs.append({
            "lane_a": {"provider": lane_a[0], "mode": lane_a[1]},
            "lane_b": {"provider": lane_b[0], "mode": lane_b[1]},
            "clips": cell.clips,
            "substitutions": cell.counts.substitutions,
            "deletions": cell.counts.deletions,
            "insertions": cell.counts.insertions,
            "errors": cell.counts.errors,
            "reference_length": cell.counts.reference_length,
            "disagreement_rate": cell.disagreement_rate,
        })
    return {"rates": rates, "clip_counts": clip_counts, "pairs": pairs}
