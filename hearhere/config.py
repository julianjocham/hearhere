"""Configuration schema and loader for HearHere.

The full ``config.toml`` (see ``config.example.toml``) is validated into the
:class:`Config` model. Loading searches the current working directory first,
then ``~/.config/hearhere/``; if no file is found, the built-in defaults apply.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# tomllib is stdlib on 3.11+; fall back to the tomli backport on 3.10.
try:  # pragma: no cover - trivial import shim
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


CONFIG_FILENAME = "config.toml"

Language = str  # "auto" | "en" | "de" | any of the 25 supported codes
Backend = Literal["local", "remote"]
Device = Literal["auto", "cpu", "cuda", "mps"]
Timestamps = Literal["segment", "word", "char"]
ExportFormat = Literal["markdown", "text", "json", "srt", "vtt"]
LLMTask = Literal["summary", "action_items", "decisions"]


class GeneralConfig(BaseModel):
    language: Language = "auto"
    storage_dir: Path = Path("~/HearHere")

    @field_validator("storage_dir", mode="before")
    @classmethod
    def _expand(cls, v: Any) -> Any:
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v


class RemoteConfig(BaseModel):
    url: str = "https://<your-runpod-host>:8808"
    token: str = ""  # bearer token for the worker; or set HEARHERE_REMOTE_TOKEN


class ComputeConfig(BaseModel):
    backend: Backend = "local"
    device: Device = "auto"
    remote: RemoteConfig = Field(default_factory=RemoteConfig)


class CaptureConfig(BaseModel):
    # "none" (Linux) = this machine has no microphone; record system output only.
    mic_device: str = "default"
    output_device: str = "default"
    sample_rate: int = 16000

    @field_validator("sample_rate")
    @classmethod
    def _positive_rate(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("sample_rate must be a positive integer")
        return v


class ASRConfig(BaseModel):
    engine: str = "parakeet_nemo"
    model: str = "nvidia/parakeet-tdt-0.6b-v3"
    timestamps: Timestamps = "segment"


class DiarizationConfig(BaseModel):
    enabled: bool = True
    engine: str = "pyannote"
    hf_token: str = ""
    min_speakers: int = Field(default=0, ge=0)  # 0 = auto
    max_speakers: int = Field(default=0, ge=0)  # 0 = auto

    @model_validator(mode="after")
    def _speaker_bounds(self) -> "DiarizationConfig":
        lo, hi = self.min_speakers, self.max_speakers
        if lo and hi and lo > hi:
            raise ValueError(
                f"min_speakers ({lo}) cannot exceed max_speakers ({hi})"
            )
        return self


class LLMConfig(BaseModel):
    # Off by default: the base pipeline transcribes only. Turn this on (and run
    # Ollama) to also get a summary / action items / decisions.
    enabled: bool = False
    engine: Literal["ollama", "llamacpp"] = "ollama"
    model: str = "llama3.1"
    tasks: list[LLMTask] = Field(
        default_factory=lambda: ["summary", "action_items", "decisions"]
    )


class ExportConfig(BaseModel):
    formats: list[ExportFormat] = Field(
        default_factory=lambda: ["markdown", "json", "srt"]
    )

    @field_validator("formats")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("export.formats must list at least one format")
        return v


class Config(BaseModel):
    """Validated view of the full ``config.toml``."""

    model_config = {"extra": "forbid"}

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @model_validator(mode="after")
    def _apply_env(self) -> "Config":
        # Environment overrides that don't belong in a committed config file.
        if not self.diarization.hf_token:
            token = os.environ.get("HF_TOKEN", "")
            if token:
                self.diarization.hf_token = token
        if not self.compute.remote.token:
            token = os.environ.get("HEARHERE_REMOTE_TOKEN", "")
            if token:
                self.compute.remote.token = token
        return self


def default_config_paths() -> list[Path]:
    """Search order for ``config.toml``: cwd, then user config dir."""
    return [
        Path.cwd() / CONFIG_FILENAME,
        Path.home() / ".config" / "hearhere" / CONFIG_FILENAME,
    ]


def find_config_file(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Return the config file to use, or ``None`` if none exists.

    An ``explicit`` path is required to exist; otherwise the default search
    order is used and the first existing file wins.
    """
    if explicit is not None:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p
    for candidate in default_config_paths():
        if candidate.is_file():
            return candidate
    return None


def load_config(explicit: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate configuration.

    With no file present (and no ``explicit`` path), returns defaults.
    """
    path = find_config_file(explicit)
    if path is None:
        return Config()
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return Config.model_validate(data)
