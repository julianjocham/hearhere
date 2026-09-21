"""Subtitle exports: SRT (``transcript.srt``) and WebVTT (``transcript.vtt``).

SRT ships in Batch 1; the VTT renderer is here too (wired into config as an
export format in Batch 5).
"""

from __future__ import annotations

from ..models import Meeting, Segment
from .timefmt import srt_timestamp, vtt_timestamp

SRT_FILENAME = "transcript.srt"
VTT_FILENAME = "transcript.vtt"

# Give a subtitle a minimum on-screen span when start == end.
_MIN_CUE = 1.5


def _cue_end(seg: Segment) -> float:
    return seg.end if seg.end > seg.start else seg.start + _MIN_CUE


def _caption(seg: Segment) -> str:
    text = seg.text.strip()
    return f"{seg.speaker}: {text}" if seg.speaker else text


def render_srt(meeting: Meeting) -> str:
    """Render ``meeting`` as SubRip (SRT) subtitles."""
    blocks: list[str] = []
    index = 1
    for seg in meeting.transcript.segments:
        if not seg.text.strip():
            continue
        blocks.append(
            f"{index}\n"
            f"{srt_timestamp(seg.start)} --> {srt_timestamp(_cue_end(seg))}\n"
            f"{_caption(seg)}\n"
        )
        index += 1
    return "\n".join(blocks)


def render_vtt(meeting: Meeting) -> str:
    """Render ``meeting`` as WebVTT subtitles."""
    blocks = ["WEBVTT\n"]
    for seg in meeting.transcript.segments:
        if not seg.text.strip():
            continue
        blocks.append(
            f"{vtt_timestamp(seg.start)} --> {vtt_timestamp(_cue_end(seg))}\n"
            f"{_caption(seg)}\n"
        )
    return "\n".join(blocks)


# The registry keys off ``render``; alias the SRT renderer as the default.
FILENAME = SRT_FILENAME
render = render_srt
