"""The ``AudioCapture`` interface and shared audio helpers.

Every platform adapter records two channels — the microphone (``self``) and the
system output (``others``) — and writes them as **16 kHz mono WAV** files, the
format Parakeet requires. Capturing at the device's native rate and resampling
on write keeps the adapters simple; the helpers here do the mono-downmix and
resample so each adapter only has to grab frames.

Audio libraries (``numpy``, ``soundfile``) are imported lazily so that importing
HearHere never requires them; only actually recording does.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

log = get_logger("capture")

# Parakeet requires 16 kHz mono input.
TARGET_SAMPLE_RATE = 16000


class CaptureError(RuntimeError):
    """Recording failed (device unavailable, incompatible format, …).

    Raised loudly so the caller never mistakes a failed capture for a silent
    meeting — an empty WAV would otherwise crash the ASR stage with a cryptic
    error much later.

    ``meeting_dir`` is set when the failure surfaced *after* the adapter had
    already written what it captured — a one-channel failure still leaves the
    healthy channel on disk — so the caller can point the user at the
    salvageable recording instead of telling them to start over.
    """

    meeting_dir: Path | None = None


@dataclass(frozen=True)
class CaptureResult:
    """Outcome of a completed recording."""

    self_wav: Path
    others_wav: Path
    duration: float


class AudioCapture(ABC):
    """Common interface for per-OS dual-channel capture.

    Concrete adapters record the mic into ``self_wav`` and the system output
    into ``others_wav``. Use as a context manager or call
    :meth:`start`/:meth:`stop` directly.
    """

    def __init__(
        self,
        self_wav: str | Path,
        others_wav: str | Path,
        *,
        mic_device: str = "default",
        output_device: str = "default",
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        self.self_wav = Path(self_wav)
        self.others_wav = Path(others_wav)
        self.mic_device = mic_device
        self.output_device = output_device
        self.sample_rate = sample_rate

    @abstractmethod
    def start(self) -> None:
        """Begin recording both channels."""

    @abstractmethod
    def stop(self) -> CaptureResult:
        """Stop recording, flush both WAV files, and return the result."""

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def to_mono(samples: "np.ndarray") -> "np.ndarray":
    """Downmix ``(frames, channels)`` (or 1-D) float audio to mono ``(frames,)``."""
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=1).astype(np.float32)


def resample(samples: "np.ndarray", src_rate: int, dst_rate: int) -> "np.ndarray":
    """Resample mono ``samples`` from ``src_rate`` to ``dst_rate``.

    Linear interpolation — adequate for speech at 16 kHz and dependency-free.
    Returns the input unchanged when the rates already match.
    """
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(samples, dtype=np.float32)
    if src_rate == dst_rate or arr.size == 0:
        return arr
    duration = arr.shape[0] / float(src_rate)
    dst_count = int(round(duration * dst_rate))
    if dst_count <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.arange(arr.shape[0], dtype=np.float64)
    dst_idx = np.linspace(0.0, arr.shape[0] - 1, dst_count, dtype=np.float64)
    return np.interp(dst_idx, src_idx, arr).astype(np.float32)


def write_wav_16k_mono(
    path: str | Path, samples: "np.ndarray", src_rate: int
) -> Path:
    """Write ``samples`` as a 16 kHz mono 16-bit PCM WAV, resampling if needed."""
    import soundfile as sf  # noqa: PLC0415

    mono = to_mono(samples)
    mono = resample(mono, src_rate, TARGET_SAMPLE_RATE)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mono, TARGET_SAMPLE_RATE, subtype="PCM_16")
    log.debug("wrote %s (%d frames @ %d Hz)", out, mono.shape[0], TARGET_SAMPLE_RATE)
    return out


def create_audio_capture(
    self_wav: str | Path,
    others_wav: str | Path,
    *,
    mic_device: str = "default",
    output_device: str = "default",
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> AudioCapture:
    """Return the capture adapter for the current OS.

    Development priority is Windows -> macOS -> Linux (see README); the
    non-Windows adapters arrive in later batches.
    """
    kwargs: dict[str, Any] = dict(
        mic_device=mic_device,
        output_device=output_device,
        sample_rate=sample_rate,
    )
    platform = sys.platform
    if platform.startswith("win"):
        from .windows import WindowsCapture  # noqa: PLC0415

        return WindowsCapture(self_wav, others_wav, **kwargs)
    if platform == "darwin":
        raise NotImplementedError(
            "macOS audio capture is not implemented yet "
            "(it will use BlackHole or a similar virtual device)."
        )
    if platform.startswith("linux"):
        from .linux import LinuxCapture  # noqa: PLC0415

        return LinuxCapture(self_wav, others_wav, **kwargs)
    raise NotImplementedError(f"No audio capture adapter for platform {platform!r}.")
