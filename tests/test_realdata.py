from __future__ import annotations

from collections import Counter
import csv
import io
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import orjson
import pytest

from audio_harness import realdata
from audio_harness.realdata_manifest import is_realdata_manifest


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
MEDIA_TOOLS_AVAILABLE = FFMPEG is not None and FFPROBE is not None
MEDIA_SKIP = pytest.mark.skipif(not MEDIA_TOOLS_AVAILABLE, reason="ffmpeg and ffprobe are required")

FILE_SESSION = "00000000-0000-4000-8000-000000000001"
DATED_FILE_SESSION = f"{FILE_SESSION}_20260808_120000"
ROOM_ID = "00000000-0000-4000-8000-000000000002"
EGRESS_ID = "EG_synthetic_0001"


def _ffmpeg(arguments: list[str]) -> None:
    assert FFMPEG is not None
    subprocess.run([FFMPEG, *arguments], check=True, capture_output=True)


def _make_audio(path: Path, parts: list[tuple[str, float]]) -> None:
    arguments: list[str] = []
    labels: list[str] = []
    for index, (kind, duration) in enumerate(parts):
        source = (
            f"sine=frequency={440 + index * 20}:sample_rate=16000:duration={duration}"
            if kind == "tone"
            else f"anullsrc=r=16000:cl=mono:d={duration}"
        )
        arguments.extend(["-f", "lavfi", "-i", source])
        labels.append(f"[{index}:a]")
    arguments.extend([
        "-filter_complex",
        f"{''.join(labels)}concat=n={len(parts)}:v=0:a=1[out]",
        "-map",
        "[out]",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
        str(path),
    ])
    _ffmpeg(arguments)


def _make_video(path: Path, duration: float = 3.0) -> None:
    _ffmpeg([
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:s=32x32:r=10:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=48000:duration={duration}",
        "-shortest",
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        "-y",
        str(path),
    ])


def _manifest_bytes(filename: str) -> bytes:
    return orjson.dumps({
        "egress_id": EGRESS_ID,
        "room_id": "room-id-not-used-as-join-key",
        "room_name": ROOM_ID,
        "started_at": 1_700_000_000_000_000_000,
        "ended_at": 1_700_000_010_000_000_000,
        "files": [{"filename": filename}],
    })


def _completed(arguments: list[str], *, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr=stderr)


@MEDIA_SKIP
def test_triage_session_classifies_and_falls_back_when_ambiguous(tmp_path: Path) -> None:
    merged = tmp_path / "session_merged.wav"
    user_track = tmp_path / "session_user.wav"
    ambiguous_track = tmp_path / "session_ambiguous.wav"
    _make_audio(merged, [("tone", 10.0)])
    _make_audio(user_track, [("tone", 6.0), ("silence", 4.0)])
    _make_audio(ambiguous_track, [("tone", 5.0), ("silence", 5.0)])

    classified = realdata.triage_session(user_track, merged)
    ambiguous = realdata.triage_session(ambiguous_track, merged)
    missing = realdata.triage_session(None, merged)

    assert classified.source == "user"
    assert classified.speech_ratio == pytest.approx(0.6, abs=0.03)
    assert ambiguous.source == "merged"
    assert ambiguous.speech_ratio == pytest.approx(0.5, abs=0.03)
    assert missing == realdata.TriageResult(source="merged", speech_ratio=None)


@MEDIA_SKIP
def test_cut_clips_merges_short_run_and_splits_long_run(tmp_path: Path) -> None:
    source = tmp_path / "segmented.wav"
    clips_dir = tmp_path / "clips"
    metadata_path = tmp_path / "clips.jsonl"
    _make_audio(
        source,
        [
            ("tone", 1.0),
            ("silence", 1.0),
            ("tone", 3.0),
            ("silence", 1.0),
            ("tone", 35.0),
        ],
    )

    records = realdata.cut_clips(
        source,
        "synthetic-session",
        clips_dir,
        source="user",
        metadata_path=metadata_path,
    )

    assert [record.clip_path.name for record in records] == ["0000.wav", "0001.wav", "0002.wav"]
    assert [record.duration for record in records] == pytest.approx([5.0, 30.0, 5.0], abs=0.04)
    assert all(2.0 <= record.duration <= 30.0 for record in records)
    assert all(record.source == "user" for record in records)
    assert all(record.clip_path.is_file() for record in records)
    metadata = [orjson.loads(line) for line in metadata_path.read_bytes().splitlines()]
    assert [item["source"] for item in metadata] == ["user", "user", "user"]


@MEDIA_SKIP
def test_video_audio_feeds_shared_clip_cutting_as_16khz_mono(tmp_path: Path) -> None:
    assert FFPROBE is not None
    video = tmp_path / "synthetic.mp4"
    metadata_path = tmp_path / "video-clips.jsonl"
    _make_video(video)

    records = realdata.cut_video_clips(
        video,
        "synthetic-video-session",
        tmp_path / "clips",
        metadata_path=metadata_path,
    )

    assert len(records) == 1
    assert records[0].source == "video"
    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(records[0].clip_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = orjson.loads(probe.stdout)["streams"][0]
    assert stream == {"sample_rate": "16000", "channels": 1}


@MEDIA_SKIP
def test_dry_run_ingest_skips_gcloud_and_inventories_local_mp3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    mp3 = recordings / "synthetic_20260808_120000.mp3"
    _make_audio(mp3, [("tone", 2.0)])
    calls: list[list[str]] = []

    def fake_run(arguments, **kwargs):
        command = [str(argument) for argument in arguments]
        calls.append(command)
        assert command[0] == "ffprobe"
        payload = {
            "streams": [{"sample_rate": "16000", "channels": 1, "bit_rate": "64000"}],
            "format": {"duration": "2.000000", "bit_rate": "65000"},
        }
        return _completed(command, stdout=orjson.dumps(payload).decode())

    monkeypatch.setattr(realdata.subprocess, "run", fake_run)
    records = realdata.ingest(recordings, "gs://synthetic-audio/", dry_run=True)

    assert len(records) == 1
    assert records[0]["duration"] == 2.0
    assert calls
    assert all(command[0] == "ffprobe" for command in calls)
    assert not any("gcloud" in command for command in calls)
    inventory = (tmp_path / "data/realdata/inventory.jsonl").read_bytes().splitlines()
    assert len(inventory) == 1


def test_ingest_downloads_only_mp3_objects_before_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    recordings = tmp_path / "recordings"
    prefix = "gs://synthetic-audio/livekit-recording-audio/"
    calls: list[list[str]] = []

    def fake_run(arguments, **kwargs):
        command = [str(argument) for argument in arguments]
        calls.append(command)
        if command[0] == "gcloud":
            assert command == [
                "gcloud",
                "storage",
                "cp",
                "gs://synthetic-audio/livekit-recording-audio/*.mp3",
                str(recordings),
            ]
            (recordings / "delivered.mp3").write_bytes(b"synthetic mp3")
            return _completed(command)

        assert command[0] == "ffprobe"
        payload = {
            "streams": [{"sample_rate": "16000", "channels": 1, "bit_rate": "64000"}],
            "format": {"duration": "2.000000", "bit_rate": "65000"},
        }
        return _completed(command, stdout=orjson.dumps(payload).decode())

    monkeypatch.setattr(realdata.subprocess, "run", fake_run)
    records = realdata.ingest(recordings, prefix)

    assert len(records) == 1
    assert records[0]["path"] == (recordings / "delivered.mp3").as_posix()
    assert calls[0] == [
        "gcloud",
        "storage",
        "cp",
        "gs://synthetic-audio/livekit-recording-audio/*.mp3",
        str(recordings),
    ]
    assert all("rsync" not in command for command in calls)


def test_manifest_listing_and_join_parsing_are_composable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_uris = [
        "gs://synthetic-video/EG_synthetic_0002.json",
        "gs://synthetic-video/EG_synthetic_0001.json",
    ]
    video_filename = f"{DATED_FILE_SESSION}_merged.mp4"
    synthetic_filename = "session-beta_merged.mp4"
    manifest_filenames = {
        manifest_uris[0]: synthetic_filename,
        manifest_uris[1]: video_filename,
    }
    calls: list[list[str]] = []
    cp_inputs: list[str] = []

    def fake_run(arguments, **kwargs):
        command = [str(argument) for argument in arguments]
        calls.append(command)
        if command[2] == "ls":
            return _completed(command, stdout="\n".join(manifest_uris) + "\n")
        assert command[2:4] == ["cp", "-I"]
        input_text = kwargs["input"]
        assert isinstance(input_text, str)
        cp_inputs.append(input_text)
        for uri in input_text.splitlines():
            Path(command[-1], uri.rsplit("/", 1)[-1]).write_bytes(_manifest_bytes(manifest_filenames[uri]))
        return _completed(command)

    monkeypatch.setattr(realdata.subprocess, "run", fake_run)
    manifests = realdata.list_manifests(tmp_path / "video", "gs://synthetic-video/")
    join_path = tmp_path / "join.jsonl"
    records = realdata.build_join(manifests, output_path=join_path)

    cp_calls = [command for command in calls if command[2] == "cp"]
    assert len(cp_calls) == 1
    assert cp_calls[0] == ["gcloud", "storage", "cp", "-I", str(tmp_path / "video/manifests")]
    assert cp_inputs == ["\n".join(sorted(manifest_uris))]
    assert [path.name for path in manifests] == ["EG_synthetic_0001.json", "EG_synthetic_0002.json"]
    assert records == [
        {
            "video_filename": video_filename,
            "session_id": FILE_SESSION,
            "room_name": ROOM_ID,
            "egress_id": EGRESS_ID,
            "started_at": 1_700_000_000_000_000_000,
            "ended_at": 1_700_000_010_000_000_000,
        },
        {
            "video_filename": synthetic_filename,
            "session_id": "session-beta",
            "room_name": ROOM_ID,
            "egress_id": EGRESS_ID,
            "started_at": 1_700_000_000_000_000_000,
            "ended_at": 1_700_000_010_000_000_000,
        },
    ]
    assert orjson.loads(join_path.read_bytes().splitlines()[0]) == records[0]


def test_manifest_listing_skips_download_when_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(arguments, **kwargs):
        command = [str(argument) for argument in arguments]
        calls.append(command)
        assert command[2] == "ls"
        return _completed(command)

    monkeypatch.setattr(realdata.subprocess, "run", fake_run)

    assert realdata.list_manifests(tmp_path / "video", "gs://synthetic-video/") == []
    assert len(calls) == 1


def test_dedupe_sessions_uses_size_then_alphabetical_tiebreak() -> None:
    objects = [
        realdata.VideoObject("gs://bucket/session-a_merged.mp4", 100, "session-a"),
        realdata.VideoObject("gs://bucket/session-a.mp4", 100, "session-a"),
        realdata.VideoObject("gs://bucket/session-b.mp4", 80, "session-b"),
        realdata.VideoObject("gs://bucket/session-b_merged.mp4", 120, "session-b"),
    ]

    selected = realdata.dedupe_sessions(objects)

    assert [item.filename for item in selected] == ["session-a.mp4", "session-b_merged.mp4"]


def test_video_listing_groups_uuid_variants_and_preserves_non_uuid_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "gs://synthetic-video/"
    listing = (
        f"100  2026-08-08T00:00:00Z  {prefix}{FILE_SESSION}_merged.mp4\n"
        f"100  2026-08-08T00:00:00Z  {prefix}{DATED_FILE_SESSION}.mp4\n"
        f"90  2026-08-08T00:00:00Z  {prefix}session-beta_merged.mp4\n"
    )

    def fake_run(arguments, **kwargs):
        command = [str(argument) for argument in arguments]
        assert command == ["gcloud", "storage", "ls", "--long", f"{prefix}*.mp4"]
        return _completed(command, stdout=listing)

    monkeypatch.setattr(realdata.subprocess, "run", fake_run)
    objects = realdata.list_video_objects(tmp_path / "video", prefix)
    selected = realdata.dedupe_sessions(objects)

    assert [item.session_id for item in selected] == [FILE_SESSION, "session-beta"]
    assert [item.filename for item in selected] == [f"{DATED_FILE_SESSION}.mp4", "session-beta_merged.mp4"]


@pytest.mark.parametrize("full_mirror", [False, True])
def test_ingest_video_joins_before_selective_or_full_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    full_mirror: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    prefix = "gs://synthetic-video/"
    dest = tmp_path / "videos"
    events: list[str] = []

    def fake_run(arguments, **kwargs):
        command = [str(argument) for argument in arguments]
        if command[2] == "ls" and "--long" not in command:
            events.append("manifest-list")
            return _completed(command, stdout=f"{prefix}EG_synthetic_0001.json\n")
        if command[2:4] == ["cp", "-I"]:
            events.append("manifest-download")
            input_text = kwargs["input"]
            assert input_text == f"{prefix}EG_synthetic_0001.json"
            Path(command[-1], "EG_synthetic_0001.json").write_bytes(_manifest_bytes(f"{DATED_FILE_SESSION}.mp4"))
            return _completed(command)
        if command[2] == "ls" and "--long" in command:
            assert (tmp_path / "data/realdata/join.jsonl").is_file()
            events.append("video-list-after-join")
            listing = (
                f"100  2026-08-08T00:00:00Z  {prefix}{DATED_FILE_SESSION}.mp4\n"
                f"100  2026-08-08T00:00:00Z  {prefix}{DATED_FILE_SESSION}_merged.mp4\n"
                f"90  2026-08-08T00:00:00Z  {prefix}session-beta.mp4\n"
            )
            return _completed(command, stdout=listing)
        assert (tmp_path / "data/realdata/join.jsonl").is_file()
        if command[2] == "cp":
            events.append("selective-download-after-join")
        else:
            assert command[2] == "rsync"
            events.append("full-download-after-join")
        return _completed(command)

    monkeypatch.setattr(realdata.subprocess, "run", fake_run)
    paths = realdata.ingest_video(
        dest,
        prefix,
        session_ids={FILE_SESSION},
        full_mirror=full_mirror,
    )

    assert events[:3] == ["manifest-list", "manifest-download", "video-list-after-join"]
    if full_mirror:
        assert events[-1] == "full-download-after-join"
        assert {path.name for path in paths} == {
            f"{DATED_FILE_SESSION}.mp4",
            f"{DATED_FILE_SESSION}_merged.mp4",
            "session-beta.mp4",
        }
    else:
        assert events[-1] == "selective-download-after-join"
        assert [path.name for path in paths] == [f"{DATED_FILE_SESSION}.mp4"]


def test_ingest_video_uses_precomputed_inputs_without_relisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    prefix = "gs://synthetic-video/"
    dest = tmp_path / "videos"
    manifest_path = tmp_path / "EG_synthetic_0001.json"
    manifest_path.write_bytes(_manifest_bytes(f"{DATED_FILE_SESSION}_merged.mp4"))
    video_object = realdata.VideoObject(
        uri=f"{prefix}{DATED_FILE_SESSION}.mp4",
        size=100,
        session_id=FILE_SESSION,
    )
    calls: list[list[str]] = []

    def fake_run(arguments, **kwargs):
        command = [str(argument) for argument in arguments]
        calls.append(command)
        assert (tmp_path / "data/realdata/join.jsonl").is_file()
        assert command == ["gcloud", "storage", "cp", video_object.uri, str(dest / video_object.filename)]
        Path(command[-1]).write_bytes(b"synthetic video")
        return _completed(command)

    monkeypatch.setattr(realdata.subprocess, "run", fake_run)
    paths = realdata.ingest_video(
        dest,
        prefix,
        session_ids={FILE_SESSION},
        manifest_paths=(manifest_path,),
        objects=(video_object,),
    )

    assert calls == [["gcloud", "storage", "cp", video_object.uri, str(dest / video_object.filename)]]
    assert paths == [dest / video_object.filename]
    join_record = orjson.loads((tmp_path / "data/realdata/join.jsonl").read_bytes().splitlines()[0])
    assert join_record["session_id"] == FILE_SESSION


def test_identify_language_does_not_force_language_or_retain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def transcribe(path: str, **kwargs):
        calls.append({"path": path, **kwargs})
        return {"language": "ja", "text": "synthetic text that must not escape"}

    monkeypatch.setattr(realdata, "_import_mlx_whisper", lambda: SimpleNamespace(transcribe=transcribe))

    assert realdata.identify_language(Path("synthetic.wav")) == "ja"
    assert "language" not in calls[0]
    assert "text" not in realdata.identify_language(Path("synthetic.wav"))


def _write_candidates(path: Path, *, video_per_stratum: int, mp3_per_stratum: int) -> None:
    records: list[dict[str, object]] = []
    durations = [4.0, 12.0, 24.0]
    for source, count in (("video", video_per_stratum), ("merged", mp3_per_stratum)):
        prefix = "video" if source == "video" else "mp3"
        for language in ("en", "ja"):
            for bucket_index, duration in enumerate(durations):
                for item_index in range(count):
                    session = f"{prefix}-{language}-{bucket_index}-{item_index}"
                    records.append({
                        "clip_path": f"data/realdata/clips/{session}/0000.wav",
                        "session_id": session,
                        "language": language,
                        "duration": duration,
                        "source": source,
                    })
    path.write_bytes(b"".join(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE) for record in records))


def test_select_pilot_is_seeded_stratified_video_first_and_session_capped(tmp_path: Path) -> None:
    clips_path = tmp_path / "clips.jsonl"
    _write_candidates(clips_path, video_per_stratum=2, mp3_per_stratum=3)

    first = realdata.select_pilot(
        14,
        seed=11,
        clips_path=clips_path,
        join_path=tmp_path / "missing-join.jsonl",
        output_path=tmp_path / "pilot-a.jsonl",
    )
    repeated = realdata.select_pilot(
        14,
        seed=11,
        clips_path=clips_path,
        join_path=tmp_path / "missing-join.jsonl",
        output_path=tmp_path / "pilot-b.jsonl",
    )
    changed = realdata.select_pilot(
        14,
        seed=12,
        clips_path=clips_path,
        join_path=tmp_path / "missing-join.jsonl",
        output_path=tmp_path / "pilot-c.jsonl",
    )

    assert first == repeated
    assert first != changed
    assert len(first) == 14
    assert all(str(record["session"]).startswith("video-") for record in first[:12])
    assert all(str(record["session"]).startswith("mp3-") for record in first[12:])
    assert {record["language"] for record in first} == {"en", "ja"}
    assert {record["duration_s"] for record in first} == {4.0, 12.0, 24.0}
    assert max(Counter(str(record["session"]) for record in first).values()) <= 2
    assert all(set(record) == {"clip", "session", "language", "duration_s", "source"} for record in first)
    assert is_realdata_manifest(tmp_path / "pilot-a.jsonl")


def test_anchor_sheet_has_empty_transcript_column_and_lf_endings(tmp_path: Path) -> None:
    clips_path = tmp_path / "clips.jsonl"
    output_dir = tmp_path / "anchors"
    _write_candidates(clips_path, video_per_stratum=1, mp3_per_stratum=1)

    selected = realdata.select_label_candidates(
        8,
        seed=23,
        clips_path=clips_path,
        join_path=tmp_path / "missing-join.jsonl",
        output_dir=output_dir,
    )

    csv_bytes = (output_dir / "anchor-review-sheet.csv").read_bytes()
    readme_bytes = (output_dir / "README.md").read_bytes()
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode())))
    assert len(selected) == 8
    assert rows
    assert all(row["true_transcript"] == "" for row in rows)
    assert "provider" not in rows[0]
    assert b"\r\n" not in csv_bytes
    assert b"\r\n" not in readme_bytes
    assert csv_bytes.endswith(b"\n")
    assert readme_bytes.endswith(b"\n")
