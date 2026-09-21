"""Pipeline orchestration: capture -> ASR -> diarize -> merge -> summarize
-> meeting.json -> exports.

The orchestrator ties the stages together and owns the meeting folder. Engines
are injectable so the pipeline is testable without the heavy ML dependencies:
pass fakes for the ASR / diarization / summarizer engines to
:func:`process_meeting` and it runs end-to-end on dummy audio. Diarization and
summary are config-gated and fail soft — if the engine is unavailable the
pipeline logs a warning and continues (undiarized / unsummarized).

With ``compute.backend = "remote"`` the compute stages run on a remote worker
instead: both WAVs are uploaded, the worker returns a finished ``meeting.json``,
and only the exports happen locally.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..config import Config
from ..export import export_meeting
from ..logging_setup import (
    add_meeting_file_handler,
    get_logger,
    remove_meeting_file_handler,
)
from ..models import UNLABELED_SPEAKER, Meeting, Summary
from . import artifacts
from .artifacts import MeetingPaths
from .merge import assign_speakers, merge_segments

log = get_logger("pipeline")


def _resolve_language(config: Config) -> str | None:
    """Config language ``"auto"`` -> ``None`` (let the model decide)."""
    lang = config.general.language
    return None if not lang or lang == "auto" else lang


def record_meeting(
    config: Config,
    title: str,
    *,
    when: datetime | None = None,
) -> MeetingPaths:
    """Create a meeting folder and record mic + system output into it.

    Returns the meeting paths; call :func:`process_meeting` next (the CLI's
    ``record`` command does both).
    """
    from ..capture.base import CaptureError, create_audio_capture  # noqa: PLC0415

    when = when or datetime.now()
    paths = artifacts.create_meeting_dir(config.general.storage_dir, title, when)
    handler = add_meeting_file_handler(paths.root)
    try:
        log.info("Recording meeting %r into %s", title, paths.root)
        log.warning(
            "Recording others may require their consent — HearHere does not "
            "obtain it for you."
        )
        capture = create_audio_capture(
            paths.self_wav,
            paths.others_wav,
            mic_device=config.capture.mic_device,
            output_device=config.capture.output_device,
            sample_rate=config.capture.sample_rate,
        )
        capture.start()
        try:
            input("Recording… press Enter to stop.\n")
        except (KeyboardInterrupt, EOFError):
            pass
        try:
            result = capture.stop()
        except CaptureError as exc:
            # Some adapters write both channels before raising, so a failure on
            # one leaves the other's audio on disk; carry the folder out with
            # the error, or the CLI only echoes "re-record" and the recovered
            # audio is unreachable. Others (WindowsCapture) raise first — check
            # the files rather than trusting the adapter to have written them.
            if paths.self_wav.is_file() and paths.others_wav.is_file():
                exc.meeting_dir = paths.root
                log.error("Recording failed; partial audio in %s: %s", paths.root, exc)
            else:
                log.error("Recording failed, no audio written: %s", exc)
            raise
        log.info("Recorded %.1fs", result.duration)
    finally:
        remove_meeting_file_handler(handler)
    return paths


def process_meeting(
    meeting_dir: str | Path,
    config: Config,
    *,
    asr: object | None = None,
    diarizer: object | None = None,
    summarizer: object | None = None,
    title: str | None = None,
) -> Meeting:
    """Transcribe an already-recorded meeting folder and write its artifacts.

    ``asr`` may be any object with a ``transcribe(wav_path, language)`` method;
    ``diarizer`` any object with ``diarize(wav_path, min_speakers, max_speakers)``;
    ``summarizer`` any object with ``summarize(text, language, tasks)``. When
    ``None`` they are built from ``config`` (diarization/summary only if enabled).
    Returns the persisted :class:`Meeting`.
    """
    paths = artifacts.resolve_meeting(meeting_dir)
    if not paths.self_wav.is_file() or not paths.others_wav.is_file():
        raise FileNotFoundError(
            f"Expected {paths.self_wav} and {paths.others_wav}; record first."
        )

    handler = add_meeting_file_handler(paths.root)
    try:
        if config.compute.backend == "remote" and asr is None:
            # Offload the whole pipeline to a remote worker; exports stay local.
            meeting = _process_remote(paths, config, title=title)
        else:
            meeting = _process_local(
                paths, config, asr, diarizer, summarizer, title=title
            )

        artifacts.write_meeting(paths, meeting)
        written = export_meeting(paths, meeting, config.export.formats)
        log.info(
            "Wrote meeting.json and %d export(s): %s",
            len(written),
            ", ".join(p.name for p in written) or "—",
        )
        return meeting
    finally:
        remove_meeting_file_handler(handler)


def _process_local(
    paths: MeetingPaths,
    config: Config,
    asr: object | None,
    diarizer: object | None,
    summarizer: object | None,
    *,
    title: str | None,
) -> Meeting:
    """Run the full pipeline on this machine and build the meeting."""
    if asr is None:
        from ..engines.asr.base import create_asr_engine  # noqa: PLC0415

        asr = create_asr_engine(config)

    language = _resolve_language(config)
    log.info("Transcribing self channel: %s", paths.self_wav)
    self_segs = asr.transcribe(str(paths.self_wav), language)  # type: ignore[attr-defined]
    log.info("Transcribing others channel: %s", paths.others_wav)
    others_segs = asr.transcribe(str(paths.others_wav), language)  # type: ignore[attr-defined]

    others_segs = _diarize_others(paths, config, others_segs, diarizer)

    transcript = merge_segments(self_segs, others_segs, language=language)
    summary = _summarize(config, transcript, summarizer)
    return _build_meeting(paths, transcript, title=title, summary=summary)


def _process_remote(
    paths: MeetingPaths, config: Config, *, title: str | None
) -> Meeting:
    """Upload both channels to the remote worker and download the meeting.

    The worker runs ASR + diarization + summary; only exports happen locally.
    The privacy warning is logged by the client on every upload — the CLI adds
    an interactive confirmation before the first one.
    """
    from ..remote.client import RemoteClient  # noqa: PLC0415

    remote = config.compute.remote
    client = RemoteClient(remote.url, token=remote.token)
    language = _resolve_language(config)
    log.info("Processing meeting remotely via %s", remote.url)
    meeting = client.process(
        paths.self_wav,
        paths.others_wav,
        title=title,
        language=language,
        on_status=lambda s: log.info("Remote job %s: %s", s.id, s.state),
    )
    # Re-anchor the remote result to this local meeting folder.
    return meeting.model_copy(
        update={"id": paths.root.name, "title": title or meeting.title}
    )


def _diarize_others(
    paths: MeetingPaths, config: Config, others_segs, diarizer: object | None
):
    """Assign diarized speaker labels to the ``others`` segments (fail-soft)."""
    if diarizer is None:
        if not config.diarization.enabled:
            return others_segs
        try:
            from ..engines.diarization.base import (  # noqa: PLC0415
                create_diarization_engine,
            )

            diarizer = create_diarization_engine(config)
        except Exception as exc:  # pragma: no cover - env-dependent
            log.warning("Diarization unavailable (%s); leaving others unlabeled", exc)
            return others_segs

    try:
        log.info("Diarizing others channel: %s", paths.others_wav)
        turns = diarizer.diarize(  # type: ignore[attr-defined]
            str(paths.others_wav),
            min_speakers=config.diarization.min_speakers,
            max_speakers=config.diarization.max_speakers,
        )
    except Exception as exc:  # pragma: no cover - env-dependent
        log.warning("Diarization failed (%s); leaving others unlabeled", exc)
        return others_segs
    return assign_speakers(others_segs, turns)


def _summarize(config: Config, transcript, summarizer: object | None) -> Summary | None:
    """Produce a meeting summary via the LLM engine (config-gated, fail-soft)."""
    if summarizer is None:
        if not config.llm.enabled:
            return None
        try:
            from ..engines.llm.base import create_summarizer_engine  # noqa: PLC0415

            summarizer = create_summarizer_engine(config)
        except Exception as exc:  # pragma: no cover - env-dependent
            log.warning("Summarizer unavailable (%s); skipping summary", exc)
            return None

    from ..engines.llm.base import transcript_to_text  # noqa: PLC0415

    text = transcript_to_text(transcript)
    if not text.strip():
        return None
    try:
        log.info("Summarizing transcript (%d chars)", len(text))
        return summarizer.summarize(  # type: ignore[attr-defined]
            text,
            language=transcript.language,
            tasks=list(config.llm.tasks),
        )
    except Exception as exc:  # pragma: no cover - env-dependent
        log.warning("Summarization failed (%s); skipping summary", exc)
        return None


def _build_meeting(
    paths: MeetingPaths, transcript, *, title: str | None, summary: Summary | None = None
) -> Meeting:
    speakers = transcript.speakers()
    return Meeting(
        id=paths.root.name,
        title=title or _title_from_id(paths.root.name),
        language=transcript.language,
        duration=transcript.duration,
        speakers=speakers,
        transcript=transcript,
        summary=summary,
    )


def rename_speakers(
    meeting_dir: str | Path,
    config: Config,
    mapping: dict[str, str],
) -> Meeting:
    """Rename speakers in a processed meeting and re-export.

    ``mapping`` maps current labels to new names, e.g.
    ``{"Speaker 1": "Anna"}``. ``"Me"`` (the mic channel) can be renamed too. The
    special key ``"Unknown"`` names the not-yet-diarized ``others`` segments as a
    single speaker — useful when diarization didn't run and every remote voice
    sits in one bucket. The change is applied to ``meeting.json`` (speaker list,
    transcript segments, and any summary text) and all exports are regenerated.
    Labels in ``mapping`` that match nothing are ignored. The models are never
    re-run.
    """
    paths = artifacts.resolve_meeting(meeting_dir)
    meeting = artifacts.read_meeting(paths)

    handler = add_meeting_file_handler(paths.root)
    try:
        applied = _apply_renames(meeting, mapping)
        if not applied:
            log.info("No matching speakers to rename in %s", paths.root)
            return meeting

        artifacts.write_meeting(paths, meeting)
        written = export_meeting(paths, meeting, config.export.formats)
        log.info(
            "Renamed %s; re-exported %d file(s)",
            ", ".join(f"{old!r}->{new!r}" for old, new in applied.items()),
            len(written),
        )
        return meeting
    finally:
        remove_meeting_file_handler(handler)


def _apply_renames(meeting: Meeting, mapping: dict[str, str]) -> dict[str, str]:
    """Apply ``mapping`` in place; return the subset actually applied."""
    present = {seg.speaker for seg in meeting.transcript.segments if seg.speaker}
    present.update(s for s in meeting.speakers if s)
    applied = {old: new for old, new in mapping.items() if old in present}

    # Special case: name the unlabeled ("Unknown") others bucket as one speaker.
    # Only when there is no real speaker literally called "Unknown" to shadow.
    unlabeled_target = mapping.get(UNLABELED_SPEAKER, "").strip()
    name_unlabeled = bool(
        unlabeled_target
        and UNLABELED_SPEAKER not in present
        and any(seg.speaker is None for seg in meeting.transcript.segments)
    )
    if not applied and not name_unlabeled:
        return {}

    for seg in meeting.transcript.segments:
        if seg.speaker in applied:
            seg.speaker = applied[seg.speaker]
        elif name_unlabeled and seg.speaker is None:
            seg.speaker = unlabeled_target
    # Rebuild the roster from the (now-renamed) transcript — the source of truth.
    meeting.speakers = meeting.transcript.speakers()

    # Only rewrite real speaker labels in the summary prose — never the literal
    # word "Unknown", which can appear in a summary meaning something else.
    if meeting.summary is not None and applied:
        meeting.summary.summary = _replace_all(meeting.summary.summary, applied)
        meeting.summary.decisions = [_replace_all(d, applied) for d in meeting.summary.decisions]
        meeting.summary.action_items = [
            _replace_all(a, applied) for a in meeting.summary.action_items
        ]

    if name_unlabeled:
        applied = {**applied, UNLABELED_SPEAKER: unlabeled_target}
    return applied


def _replace_all(text: str, mapping: dict[str, str]) -> str:
    # Longest labels first so "Speaker 1" doesn't clobber "Speaker 12".
    for old in sorted(mapping, key=len, reverse=True):
        text = text.replace(old, mapping[old])
    return text


def _title_from_id(meeting_id: str) -> str:
    """Best-effort human title from a folder id like ``2026-09-13_weekly-sync``."""
    slug = meeting_id.split("_", 1)[1] if "_" in meeting_id else meeting_id
    return slug.replace("-", " ").title() or meeting_id
