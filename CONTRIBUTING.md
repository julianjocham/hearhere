# Contributing to HearHere

Thanks for taking a look. HearHere is a small, local-first project — issues,
bug reports, and pull requests are all welcome.

## Getting set up

```bash
git clone https://github.com/julianjocham/hearhere.git
cd hearhere
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[webui,dev]"
pytest
```

**Python 3.12 is the recommended version** — it's what HearHere is developed and
tested against, and the one the ML dependencies (NeMo, PyTorch) are known to
work on. The test suite also runs on 3.10–3.13 in CI.

That install is deliberately light: it's enough to run the **entire** test suite
(121 tests) without downloading a single model. The heavy extras (`asr`,
`diarization`, `capture`) pull in torch, NeMo and host audio libraries, and you
only need them if you're changing those engines specifically.

## Running the tests

```bash
pytest                      # everything
pytest tests/test_merge.py  # one module
```

CI runs `pytest` on Python 3.10–3.13 for every push and pull request. Tests must
pass before a PR is merged. The suite does no network I/O and downloads no
models, so it should stay fast — please keep it that way: stub the engines
rather than hitting real models.

## How the project is laid out

The README has the [full tree](README.md#project-structure). The short version:

- `hearhere/cli.py` — the Typer entry point, one function per command.
- `hearhere/config.py` — the pydantic schema for `config.toml`. Every new
  setting starts here, and gets a matching line in `config.example.toml`.
- `hearhere/models.py` — `Segment`, `Transcript`, `Meeting`, `Summary`. This is
  the `meeting.json` shape and the contract between pipeline stages.
- `hearhere/pipeline/` — orchestration, merge/align, and the on-disk meeting
  folder layout.
- `hearhere/engines/` — the swappable ASR, diarization and LLM backends.
- `hearhere/export/` — renderers, all deriving from `meeting.json`.

## Conventions worth knowing

**Heavy imports are lazy.** Only `typer` and `pydantic` are hard dependencies.
Anything from an extra (torch, NeMo, pyannote, fastapi, soundcard…) is imported
*inside* the function that needs it, so `import hearhere` and the pure-logic
pipeline work on a bare install. If you add a dependency, put it behind an
extra in `pyproject.toml` and surface a clear "needs the `[…]` extra" message
via `hearhere/extras.py`.

**The pipeline fails soft.** If diarization or the LLM isn't available, the run
logs a warning and continues with a plain transcript rather than crashing. Keep
new optional stages in that spirit — but let genuine capture failures fail
*loudly*, so nobody ends up with a silently empty recording.

**HearHere is local-only.** There is no code path that sends audio, transcripts
or telemetry anywhere, and there shouldn't be — the only network access is the
one-time model downloads. Please don't add an upload path: if a stage needs more
compute, the answer is a local device, not a remote one.

**`meeting.json` is the source of truth.** Every export format is derived from
it, which is why re-exporting never re-runs a model. Don't add an export path
that reaches back to the audio.

**Adding an engine** means implementing the Protocol in the relevant
`engines/*/base.py`, registering it under a config name in the same file, and
adding a test that exercises it with a stub.

Match the style of the code around you. There's no enforced linter yet; the
codebase leans on type hints, short module-level docstrings explaining *why* a
module exists, and comments only where the reasoning isn't obvious from the code.

## Pull requests

- Branch off `main` and open a PR against it.
- Keep the change focused — one concern per PR is much easier to review.
- Add or update tests for anything behavioural.
- Update `README.md` / `INSTRUCTIONS.md` / `config.example.toml` if you change a
  command, a config key, or a default. These are checked against the code and
  drifting docs are treated as a bug.
- Say how you tested it, especially for audio capture, which CI cannot cover.

## Platform notes

Recording works on **Windows** (WASAPI loopback) and **Linux**
(PipeWire/PulseAudio sink monitor). **macOS capture** — BlackHole or a similar
virtual device — is the biggest open item; see [TODO.md](TODO.md). Everything
except recording is cross-platform today.

Audio capture can't be tested in CI, and WSL2 has no direct audio access, so
capture changes need a real host OS to verify. Say which OS and which devices you
tested on.

## Privacy

HearHere records meetings, which means contributors and users are handling other
people's voices. Please don't attach real recordings or transcripts to issues or
PRs — a synthetic or self-recorded sample makes the same point. Recording other
people may require their consent depending on where you are; that's the user's
responsibility, and nothing in this project should imply otherwise.

## License

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
