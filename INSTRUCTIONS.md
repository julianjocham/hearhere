# Running HearHere

How to install and run HearHere on each operating system.

> **Platform status.** Recording works on **Windows** (WASAPI loopback) and
> **Linux** (PipeWire/PulseAudio sink monitor). **macOS capture is not
> implemented yet** (see [`TODO.md`](TODO.md)); everything *except* recording —
> processing an existing recording, the web UI, exports — works there today.
>
> **Everything runs locally.** HearHere has no cloud mode and no upload path; the
> only network access is the one-time model downloads.

- [Windows](#windows--step-by-step-from-a-cloned-repo)
- [Linux](#linux--step-by-step-from-a-cloned-repo)
- [macOS](#macos-recording-not-yet-supported)
- [Troubleshooting](#troubleshooting)

---

## Windows — step by step from a cloned repo

This assumes you've just done `git clone` on a Windows machine and have nothing
else set up. Do the steps in order. Every command is run in **PowerShell**
(press Start, type "PowerShell", open it).

> **Heads-up:** the Windows record → transcribe path has been run end to end on
> real audio hardware on two machines. Most first-run friction is audio-device
> selection (pro/USB interfaces especially) — the
> [Troubleshooting](#troubleshooting) section covers what turned up.

### Step 1 — Install Python 3.12

1. Download the installer from
   [python.org/downloads](https://www.python.org/downloads/). **Use 3.12** —
   it's the version HearHere is developed and tested against, and the one the
   ML dependencies (NeMo, PyTorch) are known to work on. 3.10 and 3.11 should
   also work; avoid 3.13 for now, some ML deps still lag behind.
2. Run it and **tick "Add python.exe to PATH"** on the first screen before
   clicking Install. This is the most common thing people miss.
3. Close and reopen PowerShell, then confirm:

   ```powershell
   python --version
   ```

   You should see `Python 3.12.x`. If you get "Python was not found" or the
   Microsoft Store opens, PATH wasn't set — re-run the installer and tick the box.

### Step 2 — Go into the repo folder

Replace the path with wherever you cloned it:

```powershell
cd C:\Users\you\hearhere
```

You're in the right place if `dir` shows `pyproject.toml` and a `hearhere`
folder.

### Step 3 — Create and activate a virtual environment

A venv keeps HearHere's dependencies out of your system Python.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**If activation fails** with *"running scripts is disabled on this system"*,
Windows is blocking PowerShell scripts. Allow it for this window only, then
activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

(Or, if you prefer the classic Command Prompt instead of PowerShell, run
`.venv\Scripts\activate.bat` there — no execution-policy issue.)

Once active, your prompt shows `(.venv)` at the start. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

### Step 4 — Install HearHere and the pieces you want

Install from the repo you're standing in. `-e` makes it an *editable* install, so
`git pull` updates take effect without reinstalling.

```powershell
# Minimum to record and get a transcript:
pip install -e ".[capture,asr]"

# Or everything — speaker names, summaries, and the browser UI:
pip install -e ".[capture,asr,diarization,llm,webui]"
```

This takes a while — `asr` pulls in PyTorch and NeMo, which are large. When it
finishes, confirm the command is available:

```powershell
hearhere --version
hearhere --help
```

**GPU note.** By default pip installs the **CPU** build of PyTorch, which works
but transcribes slowly. If you have an NVIDIA GPU and want CUDA acceleration,
install a CUDA build of torch *before* the line above (check your CUDA version at
[pytorch.org](https://pytorch.org/get-started/locally/)):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 5 — (Optional) set up speaker names and summaries

Skip this if you only want a plain transcript — the pipeline **fails soft**:
if diarization or the LLM isn't available it logs a warning and keeps going.

**Speaker separation (diarization)** needs a free Hugging Face token because the
model is license-gated:

1. Create an account at [huggingface.co](https://huggingface.co).
2. Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   and click **Agree / accept** the terms on each.
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   (a "Read" token is enough).
4. Make it available to HearHere for this PowerShell window:

   ```powershell
   $env:HF_TOKEN = "hf_your_token_here"
   ```

   (Or put it in `config.toml` under `[diarization].hf_token` — see Step 6.)

**Summaries are off by default** — the base pipeline only transcribes. To also
get a summary / decisions / action items, install [Ollama](https://ollama.com)
and turn the LLM on:

1. Download and install Ollama for Windows from [ollama.com](https://ollama.com).
   It runs in the background after install.
2. Pull the model once:

   ```powershell
   ollama pull llama3.1
   ```

3. Enable it in `config.toml` (Step 6) with `[llm]` → `enabled = true`.

### Step 6 — (Optional) create a config file

Defaults work without this. To change where meetings are saved, the language, or
which engines run, copy the example and edit it:

```powershell
copy config.example.toml config.toml
notepad config.toml
```

HearHere looks for `config.toml` in the current folder first, then in
`%USERPROFILE%\.config\hearhere\`. The setting most people change is
`[general].storage_dir` (default `~/HearHere`, i.e.
`C:\Users\you\HearHere`).

### Step 7 — Record your first meeting

Start a real or test call so there's system audio to capture, then:

```powershell
hearhere record --title "Weekly Sync"
```

What happens:

- It records your **microphone** (your voice → `self`) and the **system output**
  — literally what's coming out of your speakers/headphones, via WASAPI loopback,
  so **no virtual cable is needed**.
- The terminal prints `Recording… press Enter to stop.` — leave it running for
  the meeting, then press **Enter**.
- It then transcribes both channels and separates speakers (a summary is written
  too only if you enabled the LLM in Step 5). **The very first run also downloads
  the Parakeet model (~2 GB)** from Hugging Face, so give it time and keep the
  network on.

**Force a language for accuracy.** Parakeet auto-detects the language and can
switch mid-clip (e.g. flip a German sentence to English). Pin it with `--language`
(or `[general].language` in `config.toml`):

```powershell
hearhere record --title "Weekly Sync" --language de   # de, en, … or "auto"
```

When it's done you'll see something like
`Processed 42 segment(s) -> C:\Users\you\HearHere\2026-09-13_weekly-sync`.

The mic is captured with **sounddevice** (PortAudio) and the system output with
**soundcard** (WASAPI loopback). If a device won't open, list the names and pin
them in `config.toml`:

```powershell
hearhere devices    # lists input names ([capture].mic_device) and outputs ([capture].output_device)
```

Handy variants:

```powershell
hearhere record --title "Weekly Sync" --no-process   # just record; transcribe later
hearhere process "C:\Users\you\HearHere\2026-09-13_weekly-sync"   # transcribe a saved recording
hearhere list                                         # list past meetings
```

### Step 8 — Look at the results

Open the meeting folder (`C:\Users\you\HearHere\<date>_<title>\`). Inside:

- `transcript.md` — readable transcript, "Me" vs each speaker.
- `summary.md` — summary, decisions, action items (if the LLM ran).
- `meeting.json` — the full structured record (everything else is exported from
  this).
- `transcript.srt` — subtitles. `audio/` — the two WAVs. `hearhere.log` — the run log.

### Step 9 — Review and tidy up in the browser

```powershell
hearhere ui
```

Open <http://127.0.0.1:8809> in your browser. You can browse every past meeting,
read the transcript and summary, **rename speakers inline**, and **re-export** to
other formats. Renaming rewrites `meeting.json` and re-exports automatically. You
can rename:

- **"Me"** — your own mic channel (e.g. → "Julian").
- **Each "Speaker N"** — when diarization labeled the others.
- **"Unknown"** — when diarization didn't run, every remote voice sits in one
  "Unknown" bucket; give it a name to label all of it at once.

Nothing is uploaded; it only reads what's already on your disk. Press **Ctrl+C**
in PowerShell to stop the server.

### Step 10 — Re-export to other formats anytime (optional)

```powershell
hearhere export "C:\Users\you\HearHere\2026-09-13_weekly-sync" --format md,srt,vtt
```

Exports are regenerated from `meeting.json`, so this never re-runs the models.

### The short version

Once set up, your day-to-day is just:

```powershell
cd C:\Users\you\hearhere
.\.venv\Scripts\Activate.ps1
hearhere record --title "Some Meeting"
hearhere ui
```

---

## Linux — step by step from a cloned repo

Recording on Linux captures the **`.monitor` source of your audio sink** — the
native PipeWire/PulseAudio loopback of exactly what you hear. No virtual cable,
no extra software, nothing to install beyond a system library.

> **Heads-up:** the capture path is verified on real PipeWire hardware (a tone
> played to the default sink lands on the `others` channel at the expected
> amplitude, mic on `self`, both written as 16 kHz mono WAV). The
> [Troubleshooting](#troubleshooting) section covers the device-selection traps.

### Step 1 — System prerequisites

You need Python 3.10+ (**3.12 recommended**), a running **PipeWire** (or
PulseAudio) session, and the system **libpulse** library — `soundcard` talks to
PulseAudio directly rather than going through PortAudio:

```bash
sudo apt install python3-venv libpulse0     # Debian / Ubuntu
# Fedora:  sudo dnf install python3-virtualenv pulseaudio-libs
# Arch:    sudo pacman -S libpulse
```

Check that a sound server is actually running:

```bash
pactl info      # should print a Server Name like "PulseAudio (on PipeWire …)"
```

### Step 2 — Create a venv and install

```bash
cd ~/hearhere                                  # wherever you cloned it
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Minimum to record and get a transcript:
pip install -e ".[capture,asr]"

# Or everything — speaker names, summaries, and the browser UI:
pip install -e ".[capture,asr,diarization,llm,webui]"
```

`asr` pulls in PyTorch and NeMo, which are large; give it time. Then:

```bash
hearhere --version
hearhere --help
```

**GPU note.** pip installs the **CPU** build of PyTorch by default. On an NVIDIA
machine, install a CUDA build *before* the line above (check your CUDA version at
[pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 — (Optional) speaker names and summaries

Identical to Windows [Step 5](#step-5--optional-set-up-speaker-names-and-summaries):
a Hugging Face token for pyannote (`export HF_TOKEN="hf_…"`), and Ollama plus
`[llm].enabled = true` for summaries. Both fail soft — without them you still get
a plain transcript.

### Step 4 — Pick your devices

This is the step worth doing before your first real meeting:

```bash
hearhere devices
```

On Linux this lists PulseAudio/PipeWire **sources** (for `[capture].mic_device`)
and **sinks** (for `[capture].output_device`). Two things to know:

- For `output_device`, name the **sink**, not the monitor — e.g.
  `"sof-soundwire Headphones"`, not `"…​.monitor"`. HearHere appends `.monitor`
  itself and refuses to continue if what it resolved isn't a real loopback
  source, so it can never record your microphone into `others.wav` by accident.
- If the machine has **no microphone at all** (an HDMI-only desktop, a VM), set
  `mic_device = "none"` to record system output alone rather than fail.

Copy the names into `config.toml`:

```bash
cp config.example.toml config.toml
$EDITOR config.toml
```

### Step 5 — Record

```bash
hearhere record --title "Weekly Sync"
hearhere record --title "Weekly Sync" --language de   # pin the language
```

It records your mic into `self` and the sink monitor into `others`, prints
`Recording… press Enter to stop.`, and runs the pipeline when you press Enter.
**The first run downloads the Parakeet model (~2 GB).**

Results land in `~/HearHere/<date>_<title>/` — same layout as
[Step 8](#step-8--look-at-the-results) on Windows. Review them in the browser
with `hearhere ui` (<http://127.0.0.1:8809>), or re-export with
`hearhere export <folder> --format md,srt,vtt`.

> **WSL2 note:** WSL2 has no direct audio device access, so you cannot record
> from it — the `others` channel would be silent. Record on the host OS; WSL2 is
> fine as a dev environment for everything else.

---

## macOS (recording not yet supported)

Recording on macOS is **not implemented yet** — it will arrive via BlackHole or
a similar virtual output device. Until then, `hearhere record` tells you so
instead of failing obscurely.

What already works on macOS today:

- **Review existing meetings** — `pip install ".[webui]"` then `hearhere ui`.
- **Process a recording made elsewhere** — drop a meeting folder with
  `audio/self.wav` + `audio/others.wav` under your `storage_dir` and run
  `pip install ".[asr,diarization,llm]"` + `hearhere process <folder>`.

Installation is the same as Windows minus the `capture` extra:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[asr,diarization,llm,webui]"
```

---

## Troubleshooting

- **`python` not found / the Microsoft Store opens** — Python isn't on PATH.
  Re-run the python.org installer and tick *"Add python.exe to PATH"*.
- **`running scripts is disabled on this system`** (activating the venv) — run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use
  `.venv\Scripts\activate.bat` in Command Prompt.
- **`hearhere` is not recognized** — the venv isn't active. Run
  `.\.venv\Scripts\Activate.ps1` (you should see `(.venv)` in the prompt).
- **`Recording the microphone channel failed…`** — the mic is captured with
  `sounddevice` (PortAudio), which handles most pro/USB interfaces (e.g.
  **Focusrite Scarlett**). If it still can't open, run `hearhere devices` to list
  input names, then set `[capture].mic_device` in `config.toml` to one of them
  (or change the Windows default mic in *Settings → System → Sound*) and
  re-record. HearHere fails immediately with this message instead of silently
  producing an empty recording.
- **`Recording the system-output (loopback) channel failed (AssertionError…)`** —
  the *output* device isn't compatible with `soundcard`'s WASAPI loopback. Run
  `hearhere devices`, set `[capture].output_device` to another playback device
  (or change the Windows default), and re-record.
- **`IndexError: index 0 is out of bounds…` / `PermissionError [WinError 32] …
  manifest.json` during transcription** — this used to happen when a channel was
  captured empty (see the item above); the empty channel is now skipped with a
  warning instead of crashing NeMo. If you still see it, the WAV under `audio/`
  has no audio — re-record after fixing the device.
- **Transcription ran on CPU / `CUDA is not available`** — the default PyTorch is
  CPU-only. On an NVIDIA machine, install a CUDA build of torch (Step 4's GPU
  note) for a large speed-up.
- **`UserWarning: … does not support symlinks` (Hugging Face cache)** — harmless;
  the model still downloads. To silence it, enable Windows *Developer Mode*
  (Settings → Privacy & security → For developers) so the cache can use symlinks.
- **`macOS audio capture is not implemented yet`** — expected; recording works
  on Windows and Linux. Everything else works on macOS.

**Linux only:**

- **`Could not open the … device (OSError: …)`** — usually no `libpulse`
  (`sudo apt install libpulse0`) or no sound server. Confirm with `pactl info`;
  on a headless box start `pipewire-pulse` (or `pulseaudio`) first.
- **`The default input source (… ) is a loopback monitor of system output`** —
  your default PulseAudio source is a sink monitor, so recording it as `self.wav`
  would just duplicate `others.wav`. Set `[capture].mic_device` to a real input
  (`hearhere devices`), or `"none"` if the machine genuinely has no microphone.
- **`No microphone/output device matches [capture] device '…'`** — the name in
  `config.toml` didn't match anything; HearHere refuses the fuzzy near-match
  rather than silently recording the wrong device. Copy an exact name from
  `hearhere devices`.
- **`Resolved … as the system-output monitor, but it is not a loopback source`** —
  the named sink has no monitor source. Pick another sink for
  `[capture].output_device`.
- **`others.wav` is silent** — nothing was playing to the sink you captured, or
  the app is on a different sink. Check the per-app routing in `pavucontrol` and
  set `[capture].output_device` to the sink the meeting app actually uses.
- **`Capture thread … did not stop within 5s`** — a device wedged (a yanked USB
  headset, typically). HearHere keeps whatever it captured; process the folder
  and re-record if it's short.
- **`The web UI needs the '[webui]' extra`** — run `pip install ".[webui]"`.
- **Diarization is skipped** — set a Hugging Face token (`HF_TOKEN`) and accept
  the pyannote model license; the pipeline fails soft and continues unlabeled.
- **No summary** — summaries are **off by default**. Set `[llm].enabled = true`
  in `config.toml`, install/run Ollama, and pull the model (`ollama pull llama3.1`).
  If the LLM is unavailable at run time the summary is skipped (fails soft).
- **Transcript switches language** — pin it with `--language de` (or set
  `[general].language`); auto-detection can flip mid-clip.
- **Slow transcription** — Parakeet runs on CPU but is much faster on a CUDA
  GPU; set `[compute].device = "cuda"` or leave it on `auto`. There is
  deliberately no cloud/offload option — a local GPU is the fix.
