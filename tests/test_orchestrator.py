"""End-to-end pipeline test with a fake ASR engine (no heavy deps).

Covers the Batch 1 "done when": process a recorded meeting into a
speaker-attributed transcript with Me/Others segments, and re-export from
meeting.json without re-running the model.
"""

from __future__ import annotations

import pytest

from hearhere.config import Config
from hearhere.models import Segment
from hearhere.pipeline import artifacts
from hearhere.pipeline.orchestrator import process_meeting


class FakeASR:
    """Returns canned segments per channel and counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, wav_path: str, language: str | None):
        self.calls += 1
        if wav_path.endswith("self.wav"):
            return [Segment(start=4.0, end=10.0, text="Kurzes Update.")]
        return [
            Segment(start=11.0, end=20.0, text="Staging is green."),
            Segment(start=25.0, end=30.0, text="Ship on Friday."),
        ]


def _recorded_meeting(tmp_path):
    paths = artifacts.create_meeting_dir(tmp_path, "Weekly Sync")
    paths.self_wav.write_bytes(b"")   # dummy audio; the fake ASR ignores content
    paths.others_wav.write_bytes(b"")
    return paths


def test_process_produces_transcript_with_me_and_others(tmp_path):
    paths = _recorded_meeting(tmp_path)
    cfg = Config.model_validate({"export": {"formats": ["markdown", "json", "srt"]}})
    asr = FakeASR()

    meeting = process_meeting(paths.root, cfg, asr=asr, title="Weekly Sync")

    assert asr.calls == 2  # self + others, once each
    assert meeting.speakers == ["Me"]  # others undiarized until Batch 2
    channels = {s.channel for s in meeting.transcript.segments}
    assert channels == {"self", "others"}
    me = [s for s in meeting.transcript.segments if s.channel == "self"]
    assert me and me[0].speaker == "Me"
    assert meeting.duration == 30.0

    # Artifacts written.
    assert paths.meeting_json.is_file()
    assert (paths.root / "transcript.md").is_file()
    assert (paths.root / "transcript.json").is_file()
    assert (paths.root / "transcript.srt").is_file()
    assert paths.log.is_file()

    md = (paths.root / "transcript.md").read_text(encoding="utf-8")
    assert "**[00:00:04] Me:** Kurzes Update." in md


def test_reexport_does_not_rerun_model(tmp_path):
    from hearhere.export import export_meeting

    paths = _recorded_meeting(tmp_path)
    cfg = Config.model_validate({"export": {"formats": ["markdown"]}})
    asr = FakeASR()
    process_meeting(paths.root, cfg, asr=asr, title="Weekly Sync")
    assert asr.calls == 2

    # Re-export reads meeting.json — the model is never touched again.
    meeting = artifacts.read_meeting(paths)
    written = export_meeting(paths, meeting, ["text", "srt", "vtt"])
    assert {p.name for p in written} == {"transcript.txt", "transcript.srt", "transcript.vtt"}
    assert asr.calls == 2  # unchanged


def test_process_missing_audio_raises(tmp_path):
    paths = artifacts.create_meeting_dir(tmp_path, "No Audio")
    with pytest.raises(FileNotFoundError):
        process_meeting(paths.root, Config(), asr=FakeASR())


def test_meeting_json_roundtrips_after_process(tmp_path):
    paths = _recorded_meeting(tmp_path)
    cfg = Config.model_validate({"export": {"formats": ["json"]}})
    meeting = process_meeting(paths.root, cfg, asr=FakeASR(), title="Weekly Sync")
    reloaded = artifacts.read_meeting(paths)
    assert reloaded == meeting


# --- a failed recording points at salvageable audio, but only if it exists --


class _FailingCapture:
    """Stands in for a platform adapter whose stop() raises.

    ``writes`` mirrors the two behaviours in the tree: LinuxCapture persists
    both channels before raising, WindowsCapture raises first.
    """

    def __init__(self, self_wav, others_wav, *, writes: bool, **kwargs):
        self.self_wav, self.others_wav, self._writes = self_wav, others_wav, writes

    def start(self):
        pass

    def stop(self):
        from hearhere.capture.base import CaptureError

        if self._writes:
            self.self_wav.write_bytes(b"")
            self.others_wav.write_bytes(b"")
        raise CaptureError("the mic died mid-meeting")


def _run_failing_record(tmp_path, monkeypatch, *, writes: bool):
    from hearhere.capture import base as capture_base
    from hearhere.pipeline.orchestrator import record_meeting

    monkeypatch.setattr(
        capture_base,
        "create_audio_capture",
        lambda self_wav, others_wav, **kw: _FailingCapture(
            self_wav, others_wav, writes=writes
        ),
    )
    monkeypatch.setattr("builtins.input", lambda *a: "")
    cfg = Config()
    cfg.general.storage_dir = tmp_path
    return record_meeting(cfg, "Weekly Sync")


def test_capture_failure_reports_where_the_partial_audio_is(tmp_path, monkeypatch):
    from hearhere.capture.base import CaptureError

    with pytest.raises(CaptureError) as err:
        _run_failing_record(tmp_path, monkeypatch, writes=True)
    # Both WAVs are on disk, so the user can salvage the healthy channel.
    assert err.value.meeting_dir is not None
    # cli.record tells the user to run `hearhere process <dir>`; that only works
    # if the audio is really there.
    recovered = artifacts.resolve_meeting(err.value.meeting_dir)
    assert recovered.self_wav.is_file() and recovered.others_wav.is_file()


def test_capture_failure_promises_nothing_when_no_audio_was_written(
    tmp_path, monkeypatch
):
    # WindowsCapture.stop() raises before writing, so advertising a recoverable
    # folder here would send the user to `hearhere process` on an empty dir.
    from hearhere.capture.base import CaptureError

    with pytest.raises(CaptureError) as err:
        _run_failing_record(tmp_path, monkeypatch, writes=False)
    assert err.value.meeting_dir is None
