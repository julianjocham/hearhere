# HearHere — TODO

Ordered, batchable implementation plan. Each **Batch** is a self-contained unit that leaves the app in a working, testable state. Do them top to bottom; within a batch, tasks can mostly be done in parallel. Checkboxes track progress.

Legend: 🎯 milestone · ⛓️ depends on the batch above · 🧪 has an explicit "done when" test.

**Where things stand:** recording works on **Windows** and **Linux**; **macOS
capture (Batch 3) is the only open platform.** Everything runs on the user's own
machine — there is no remote/offload path anywhere in the plan.

---

## Batch 0 — Project scaffold 🎯 *repo builds & imports*

- [x] Create `pyproject.toml` (Python 3.10+, deps as placeholders, `hearhere` console entry point → `hearhere.cli:app`).
- [x] Create package skeleton matching the README structure (`hearhere/`, `capture/`, `engines/`, `pipeline/`, `export/`, `llm/`, `tests/`).
- [x] `config.py`: pydantic schema for the full `config.toml`; loader (cwd → `~/.config/hearhere/`); sensible defaults.
- [x] `config.example.toml` matching the README.
- [x] Define core data models: `Segment`, `SpeakerTurn`, `Transcript`, `Summary`, `Meeting` (the `meeting.json` shape).
- [x] `pipeline/artifacts.py`: meeting-folder layout, create/resolve/read/write helpers.
- [x] Logging setup + `hearhere.log` per meeting.
- [x] 🧪 `hearhere --help` runs; config loads and validates; tests for config + artifact layout pass.

---

## Batch 1 — Core pipeline on Windows 🎯 *record → transcript* ⛓️ (Phase 1)

- [x] `capture/base.py`: `AudioCapture` interface (start/stop, writes 16 kHz mono WAV, resample if needed).
- [x] `capture/windows.py`: mic + WASAPI **loopback** capture → `self.wav` / `others.wav`.
- [x] `engines/asr/base.py`: `ASREngine` protocol.
- [x] `engines/asr/parakeet_nemo.py`: load `nvidia/parakeet-tdt-0.6b-v3` via NeMo; transcribe WAV → segments with timestamps.
- [x] Compute-device selection: `auto` → `cuda` if available else `cpu` (mps stub).
- [x] `pipeline/merge.py`: align/merge `self` + `others` segments on one timeline (self labeled "Me").
- [x] `pipeline/orchestrator.py`: capture → ASR(self) → ASR(others) → merge → write `meeting.json`.
- [x] `export/markdown.py`, `export/json.py`, `export/subtitles.py` (SRT), `export/text.py`.
- [x] `cli.py`: `record`, `process`, `export`, `list`.
- [x] 🧪 Record a short meeting on Windows → get `transcript.md` with Me/Others segments; re-export works without re-running models. *(Pipeline + re-export verified by tests with a fake ASR engine; the live record → transcribe path has since been run end to end on real Windows audio hardware on two machines — see the "Robust Windows capture backend" item below.)*

---

## Batch 2 — Speakers & summaries 🎯 *who-said-what + summary* ⛓️ (Phase 2)

- [x] `engines/diarization/base.py`: `DiarizationEngine` protocol.
- [x] `engines/diarization/pyannote.py`: diarize `others.wav`; HF-token config + first-download docs; offline afterward.
- [x] Merge step: assign diarization labels to `others` ASR segments (overlap-based); stable "Speaker N" ids.
- [x] `cli.py`: `speakers --set "Speaker 1=Anna"` to rename (rewrites `meeting.json`, re-exports).
- [x] `engines/llm/base.py`: `SummarizerEngine` protocol.
- [x] `engines/llm/ollama.py`: summary / decisions / action_items via Ollama.
- [x] `llm/prompts.py`: prompt templates (EN + DE aware).
- [x] `export`: `summary.md`.
- [x] Wire diarization + summary into the orchestrator (config-gated, fail-soft if engines unavailable).
- [x] 🧪 Multi-speaker meeting yields distinct speakers + `summary.md`; renaming propagates to all exports. *(Verified by tests + a CLI smoke run using fake diarization/LLM engines on a WSL2 dev box; live pyannote/Ollama runs still to be exercised with the real models — they need `pip install "hearhere[diarization,llm]"`, an HF token, and a running Ollama server.)*

---

## Batch 3 — macOS support ⛓️ (Phase 3) — **the only open platform**

- [ ] `capture/macos.py`: mic + system output via BlackHole/virtual device.
- [ ] Setup docs: Multi-Output Device + BlackHole install steps (link from README).
- [ ] MPS device path (`device = "mps"`), with graceful CPU fallback per-op.
- [ ] 🧪 Record + process a meeting end-to-end on macOS.

---

## Batch 4 — Linux support ✅ *done* ⛓️ (Phase 4)

- [x] `capture/linux.py`: mic + default-sink `.monitor` capture via PipeWire/PulseAudio. *(Both channels go through `soundcard` (libpulse), which enumerates each sink's monitor as a loopback source — symmetric with, and simpler than, the Windows split. `create_audio_capture` now dispatches Linux to `LinuxCapture`.)*
- [x] Device discovery/selection for non-default devices. *(`hearhere devices` is now platform-aware: on Linux it lists soundcard sources + sinks — the names `[capture].mic_device` / `output_device` actually accept — instead of sounddevice's raw ALSA `hw:` names.)*
- [x] 🧪 Record + process a meeting end-to-end on Linux. *(Capture path verified on real PipeWire hardware on 2026-09-17: a 440 Hz tone played to the default sink was captured on the `others` channel at the exact expected amplitude, mic on `self`, both written as 16 kHz mono WAV. Unit tests cover the error paths and empty-recording case. Full transcribe run still depends on the `[asr]` model download.)*

---

## Batch 5 — Additional engines & formats *(optional, parallelizable)*

- [ ] `engines/llm/llamacpp.py` (GGUF via llama.cpp) as an Ollama alternative.
- [x] Export: WebVTT (`vtt`).
- [x] Language auto-detect wiring + per-meeting language override (`--language`, `[general].language`).
- [ ] Engine registry so config names resolve to implementations cleanly.

---

## Batch 6 — Local web UI ⛓️ (Phase 5)

- [x] FastAPI backend: list meetings, read `meeting.json`, trigger re-export, rename speakers. *(Pure `WebUIService` core in `hearhere/webui/service.py` — testable without the web stack — delegates to `artifacts`, `export_meeting`, and `rename_speakers`. Thin FastAPI wrapper in `webui/server.py` (lazily imported): `GET /api/meetings`, `GET /api/meetings/{id}`, `POST /api/meetings/{id}/export`, `POST /api/meetings/{id}/speakers`, `GET /api/meetings/{id}/exports/{filename}`. Meeting ids are validated against path traversal — must sit directly under `storage_dir`. Read-only w.r.t. audio: the UI never records or uploads.)*
- [x] Browser frontend: browse past meetings, view transcript + summary, rename speakers inline, export. *(Single self-contained `webui/static/index.html` — vanilla JS, no CDN/build step, light+dark. Master-detail: meeting list on the left; transcript (Me/Speaker colour-coded), summary/decisions/action-items, inline speaker-rename inputs, and per-format export checkboxes + download links on the right. Served at `/` by the FastAPI app; launched with `hearhere ui` (loopback `127.0.0.1:8809` by default).)*
- [ ] **(future) Standalone no-install viewer executable.** Emit a single double-clickable program that starts the web UI and opens the browser — so a non-technical user needs no `pip install`. **Scope: viewer only** (browse / rename / re-export meetings already on disk); it bundles just `fastapi` + `uvicorn` + `pydantic` + the HearHere core + `static/index.html` (~30 MB), NOT torch/NeMo/pyannote (multi-GB, GPU-specific) or Ollama (a separate server that can't be embedded) — recording/transcription stays a `pip install "hearhere[...]"` concern. Plan: a launcher module (pick a free port, run uvicorn in-process, auto-open the default browser, resolve `index.html` via `sys._MEIPASS` when frozen) + a PyInstaller onefile spec (`--add-data` for the static HTML) + a build script. **Open question — target OS:** PyInstaller can't cross-compile, so a Windows `.exe` must be built on Windows (or in Windows CI). Scoped to viewer-only; build deferred.
- [x] 🧪 Review, rename, and export a past meeting entirely from the browser. *(Verified by 10 tests — service list/get/export/rename, traversal + unknown-id + bad-format guards, and HTTP round-trips via Starlette `TestClient` including the full review→rename→export→download flow with the rename propagating into `transcript.md`/`summary.md` — plus a live uvicorn server on a real socket driven by a stdlib `urllib` client: frontend served, meeting listed, summary read, `Speaker 1`→`Anna` renamed and re-exported, `srt` exported, rename confirmed in the downloaded markdown. New `[webui]` extra: `pip install "hearhere[webui]"`.)*

---

## Cross-cutting (do alongside, not a blocking batch)

- [x] Tests per batch (unit for merge/align/config/export; smoke tests for capture where feasible). *(121 tests, no network or model downloads.)*
- [ ] Error handling: missing devices, no model, no GPU, Ollama/pyannote unavailable. *(Partly done: capture now fails loudly with an actionable `CaptureError` when a channel's recorder crashes — and carries the meeting folder out with it so a one-channel failure leaves the healthy channel salvageable — and ASR skips empty/near-silent WAVs instead of crashing NeMo. Found on real Windows hardware 2026-09-13.)*
- [x] **Long-audio transcription.** A single NeMo `transcribe` call truncated long audio (cut off ~32 s of a 46 s clip on real hardware 2026-09-13). Fixed: audio longer than `_CHUNK_SECONDS` (24 s) is transcribed in overlapping 24 s windows (6 s overlap), kept in full, and de-duplicated (`_dedup_segments` drops copies overlapping >50 % of the shorter, keeping the fuller one). Chunks stay well under the ~32 s truncation point so none truncates its own tail, and the overlap means any utterance straddling a cut is transcribed whole in at least one chunk — a first hard-seam scheme (30 s/5 s) left an ~8 s hole at the boundary when a chunk dropped its tail, since fixed. Segment + word timestamps are offset onto the global timeline; ASR logs a coverage warning if the transcript still ends >3 s before the audio. Window/dedup/offset logic unit-tested (incl. a no-gap-across-seam case); the NeMo call path still needs a real-hardware re-run to confirm full, gap-free coverage.
- [x] **Robust Windows capture backend.** `soundcard`'s WASAPI shared-mode capture asserted on some devices' mix format (`wFormatTag == 0xFFFE`) and failed on pro/USB interfaces like the Focusrite Scarlett (hit on real hardware 2026-09-13). Done: (a) mic is now captured via `sounddevice`/PortAudio (already a `[capture]` dep); soundcard is used only for system-output loopback (PortAudio can't do loopback, soundcard can't be swapped out for it). (b) Each channel records at its device's **native sample rate** and is downmixed + resampled to 16 kHz on write. (c) New `hearhere devices` command lists input names (`[capture].mic_device`) and output names (`[capture].output_device`). ✅ Verified on real Windows hardware on two machines; the pure param-selection logic (`_mic_stream_params`) and error paths are unit-tested on top.
- [x] Choose & add HearHere's own license file — **MIT**, see `LICENSE`.
- [x] CI: tests on push/PR via GitHub Actions (`.github/workflows/tests.yml`, Python 3.10–3.13). *(Lint not wired up yet.)*
- [ ] Packaging/distribution story per OS (later).
- [ ] Recording-consent notice surfaced in `record` output (privacy/legal).
