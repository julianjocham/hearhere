"""The ``ASREngine`` interface and engine factory.

An ASR engine turns a 16 kHz mono WAV into time-stamped :class:`Segment`s. The
protocol is deliberately small so backends can be swapped via config without
touching the pipeline. Engines run locally — HearHere never ships audio off the
machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ...models import Segment

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config import Config


@runtime_checkable
class ASREngine(Protocol):
    """Transcribes a 16 kHz mono WAV into time-stamped segments."""

    def transcribe(self, wav_path: str, language: str | None) -> list[Segment]:
        """Return time-stamped text segments for ``wav_path``.

        ``language`` is an ISO code, or ``None`` to auto-detect. Returned
        segments carry ``start``/``end``/``text``; ``speaker``/``channel`` are
        assigned later by the pipeline.
        """
        ...


def create_asr_engine(config: "Config", device: str | None = None) -> ASREngine:
    """Build the ASR engine named in ``config.asr.engine``.

    ``device`` overrides the resolved compute device (see
    :func:`hearhere.compute.select_device`).
    """
    from ...compute import select_device  # noqa: PLC0415

    resolved = device or select_device(config.compute.device)
    name = config.asr.engine

    if name == "parakeet_nemo":
        from .parakeet_nemo import ParakeetNeMoEngine  # noqa: PLC0415

        return ParakeetNeMoEngine(
            model_name=config.asr.model,
            device=resolved,
            timestamps=config.asr.timestamps,
        )
    raise ValueError(f"Unknown ASR engine {name!r}.")
