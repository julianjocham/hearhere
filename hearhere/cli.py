"""HearHere command-line interface.

``record``/``process``/``export``/``list`` run the core pipeline (Batch 1);
``speakers`` renames diarized speakers (Batch 2); ``ui`` serves the local web UI
for reviewing past meetings in the browser (Batch 6). Every command runs
entirely on this machine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from hearhere import __version__
from hearhere.config import load_config
from hearhere.extras import MissingExtraError
from hearhere.logging_setup import get_logger, setup_logging
from hearhere.pipeline import artifacts

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="HearHere — hear what was said, right here on your machine.",
)

log = get_logger("cli")

# A single --config option reused across commands.
ConfigOpt = typer.Option(
    None, "--config", "-c", help="Path to config.toml (default: search cwd then ~/.config/hearhere)."
)

# --language overrides [general].language for one run. Pin it (e.g. "de") when the
# model keeps switching language mid-clip; "auto" restores auto-detection.
LanguageOpt = typer.Option(
    None, "--language", "-l",
    help='Force the transcription language, e.g. "de" or "en" ("auto" to detect).',
)


def _with_language(cfg, language: Optional[str]):
    """Return ``cfg`` with ``[general].language`` overridden, if given."""
    if language:
        cfg.general.language = language
    return cfg


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"hearhere {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Global options."""
    setup_logging("DEBUG" if verbose else "INFO")


def _todo(feature: str, batch: str) -> None:
    raise typer.Exit(
        typer.echo(  # type: ignore[func-returns-value]
            f"'{feature}' is not implemented yet (arrives in {batch}). "
            "See TODO.md for the roadmap.",
            err=True,
        )
        or 1
    )


@app.command()
def record(
    title: str = typer.Option("Untitled Meeting", "--title", "-t", help="Meeting title."),
    no_process: bool = typer.Option(
        False, "--no-process", help="Record only; skip the transcribe step."
    ),
    language: Optional[str] = LanguageOpt,
    config: Optional[str] = ConfigOpt,
) -> None:
    """Record mic + system output until stopped, then run the pipeline."""
    from hearhere.pipeline.orchestrator import process_meeting, record_meeting

    from hearhere.capture.base import CaptureError

    cfg = _with_language(load_config(config), language)
    try:
        paths = record_meeting(cfg, title)
    except (NotImplementedError, CaptureError, MissingExtraError) as exc:
        typer.echo(str(exc), err=True)
        partial = getattr(exc, "meeting_dir", None)
        if partial is not None:
            typer.echo(f"Partial recording kept in: {partial}", err=True)
            typer.echo(f"To salvage it, run: hearhere process {partial}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Recorded meeting: {paths.root}")
    if no_process:
        typer.echo(f"Run: hearhere process {paths.root}")
        return
    try:
        meeting = process_meeting(paths.root, cfg, title=title)
    except MissingExtraError as exc:
        typer.echo(f"Recorded, but not processed. {exc}", err=True)
        typer.echo(f"Once installed, run: hearhere process {paths.root}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Processed {len(meeting.transcript.segments)} segment(s) -> {paths.root}")


@app.command()
def process(
    meeting_dir: Path = typer.Argument(..., help="Path to a recorded meeting folder."),
    language: Optional[str] = LanguageOpt,
    config: Optional[str] = ConfigOpt,
) -> None:
    """Run the pipeline on an already-recorded meeting folder."""
    from hearhere.pipeline.orchestrator import process_meeting

    cfg = _with_language(load_config(config), language)
    try:
        meeting = process_meeting(meeting_dir, cfg)
    except (FileNotFoundError, MissingExtraError) as exc:
        raise typer.Exit(typer.echo(str(exc), err=True) or 1)  # type: ignore[func-returns-value]
    typer.echo(
        f"Transcribed {len(meeting.transcript.segments)} segment(s); "
        f"speakers: {', '.join(meeting.speakers) or '—'}"
    )


@app.command()
def export(
    meeting_dir: Path = typer.Argument(..., help="Path to a processed meeting folder."),
    format: Optional[str] = typer.Option(
        None, "--format", "-f", help="Comma-separated formats, e.g. md,srt."
    ),
    config: Optional[str] = ConfigOpt,
) -> None:
    """Re-export a processed meeting into other formats (from meeting.json)."""
    from hearhere.export import export_meeting

    cfg = load_config(config)
    paths = artifacts.resolve_meeting(meeting_dir)
    try:
        meeting = artifacts.read_meeting(paths)
    except FileNotFoundError as exc:
        raise typer.Exit(typer.echo(str(exc), err=True) or 1)  # type: ignore[func-returns-value]
    formats = (
        [f for f in format.split(",") if f.strip()] if format else cfg.export.formats
    )
    written = export_meeting(paths, meeting, formats)
    typer.echo("Wrote: " + (", ".join(p.name for p in written) or "nothing"))


@app.command()
def speakers(
    meeting_dir: Path = typer.Argument(..., help="Path to a processed meeting folder."),
    set_: list[str] = typer.Option(
        None, "--set", help='Rename a speaker, e.g. --set "Speaker 1=Anna". Repeatable.'
    ),
    config: Optional[str] = ConfigOpt,
) -> None:
    """Rename speakers after the fact (rewrites meeting.json, re-exports).

    With no ``--set`` given, lists the current speakers.
    """
    from hearhere.pipeline.orchestrator import rename_speakers

    cfg = load_config(config)
    paths = artifacts.resolve_meeting(meeting_dir)
    try:
        meeting = artifacts.read_meeting(paths)
    except FileNotFoundError as exc:
        raise typer.Exit(typer.echo(str(exc), err=True) or 1)  # type: ignore[func-returns-value]

    if not set_:
        typer.echo("Speakers: " + (", ".join(meeting.speakers) or "—"))
        typer.echo('Rename with: --set "Speaker 1=Anna"')
        return

    try:
        mapping = _parse_speaker_map(set_)
    except ValueError as exc:
        raise typer.Exit(typer.echo(str(exc), err=True) or 1)  # type: ignore[func-returns-value]

    updated = rename_speakers(meeting_dir, cfg, mapping)
    typer.echo("Speakers: " + (", ".join(updated.speakers) or "—"))


def _parse_speaker_map(entries: list[str]) -> dict[str, str]:
    """Parse ``["Speaker 1=Anna", ...]`` into ``{"Speaker 1": "Anna"}``."""
    mapping: dict[str, str] = {}
    for entry in entries:
        old, sep, new = entry.partition("=")
        old, new = old.strip(), new.strip()
        if not sep or not old or not new:
            raise ValueError(f'Invalid --set {entry!r}; expected "Old Name=New Name".')
        mapping[old] = new
    return mapping


@app.command(name="list")
def list_meetings(config: Optional[str] = ConfigOpt) -> None:
    """List past meetings in the configured storage directory."""
    cfg = load_config(config)
    meetings = artifacts.list_meetings(cfg.general.storage_dir)
    if not meetings:
        typer.echo(f"No meetings found in {cfg.general.storage_dir}")
        return
    for m in meetings:
        typer.echo(m.root.name)


@app.command()
def devices(config: Optional[str] = ConfigOpt) -> None:
    """List audio devices for [capture].mic_device / output_device.

    The backend depends on the OS: on Linux both channels go through soundcard
    (PulseAudio/PipeWire), so inputs and outputs are both listed from it; on
    Windows the mic is listed via sounddevice and outputs via soundcard. Copy a
    name into config.toml.
    """
    if sys.platform.startswith("linux"):
        _devices_linux()
    else:
        _devices_default()


def _devices_linux() -> None:
    """List soundcard (PulseAudio/PipeWire) sources and sinks — the Linux backend."""
    typer.echo("Microphone inputs (for [capture].mic_device):")
    try:
        import soundcard as sc  # noqa: PLC0415

        for mic in sc.all_microphones(include_loopback=False):
            typer.echo(f"  {mic.name}")
        typer.echo('  (none listed? set mic_device = "none" for output-only)')
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        typer.echo(f"  (unavailable: {exc}; install 'hearhere[capture]')", err=True)

    typer.echo("System-output devices (for [capture].output_device):")
    try:
        import soundcard as sc  # noqa: PLC0415

        for speaker in sc.all_speakers():
            typer.echo(f"  {speaker.name}")
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        typer.echo(f"  (unavailable: {exc}; install 'hearhere[capture]')", err=True)


def _devices_default() -> None:
    """List sounddevice inputs + soundcard outputs — the Windows/macOS backend."""
    typer.echo("Microphone inputs (for [capture].mic_device):")
    try:
        import sounddevice as sd  # noqa: PLC0415

        for idx, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                typer.echo(
                    f"  {d['name']}  "
                    f"({d['max_input_channels']} ch @ {int(d.get('default_samplerate') or 0)} Hz)"
                )
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        typer.echo(f"  (unavailable: {exc}; install 'hearhere[capture]')", err=True)

    typer.echo("System-output devices (for [capture].output_device):")
    try:
        import soundcard as sc  # noqa: PLC0415

        for speaker in sc.all_speakers():
            typer.echo(f"  {speaker.name}")
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        typer.echo(f"  (unavailable: {exc}; install 'hearhere[capture]')", err=True)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Bind host (loopback by default)."),
    port: int = typer.Option(8809, help="Bind port."),
    config: Optional[str] = ConfigOpt,
) -> None:
    """Serve the local web UI to browse, rename, and re-export past meetings."""
    cfg = load_config(config)
    try:
        from hearhere.webui.server import run_ui
    except ModuleNotFoundError as exc:  # missing [webui] extra
        raise typer.Exit(  # type: ignore[func-returns-value]
            typer.echo(
                f"The web UI needs the '[webui]' extra: {exc}. "
                'Install with: pip install "hearhere[webui]".',
                err=True,
            )
            or 1
        )
    typer.echo(f"HearHere web UI on http://{host}:{port} (meetings from {cfg.general.storage_dir})")
    run_ui(cfg, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
