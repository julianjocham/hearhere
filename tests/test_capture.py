"""Regression tests for real-hardware failure modes found on Windows.

Both fixes are for silent failures: a capture thread that dies leaving an empty
WAV (soundcard rejecting a device's WASAPI format), and ASR then crashing on that
empty WAV. Neither test needs numpy/soundcard/torch — the guards run before any
heavy import.
"""

from __future__ import annotations

import wave

from hearhere.capture.base import CaptureError
from hearhere.capture.linux import (
    NO_MIC,
    LinuxCapture,
    _device_label,
    _is_monitor,
    _verify_requested,
)
from hearhere.capture.windows import WindowsCapture, _mic_stream_params
from hearhere.engines.asr.parakeet_nemo import (
    ParakeetNeMoEngine,
    _wav_duration_seconds,
)


def _write_wav(path, *, frames: bytes = b"", rate: int = 16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)
    return path


# --- ASR empty-audio guard -------------------------------------------------


def test_asr_skips_empty_wav_without_loading_model(tmp_path):
    wav = _write_wav(tmp_path / "self.wav")  # 0 frames
    engine = ParakeetNeMoEngine()
    # Returns [] and never touches NeMo (self._model stays None).
    assert engine.transcribe(str(wav)) == []
    assert engine._model is None


def test_asr_skips_missing_wav(tmp_path):
    engine = ParakeetNeMoEngine()
    assert engine.transcribe(str(tmp_path / "nope.wav")) == []


def test_wav_duration_helper(tmp_path):
    assert _wav_duration_seconds(str(tmp_path / "missing.wav")) == 0.0
    wav = _write_wav(tmp_path / "empty.wav")
    assert _wav_duration_seconds(str(wav)) == 0.0
    # 16000 frames @ 16 kHz == 1 second.
    full = _write_wav(tmp_path / "one_sec.wav", frames=b"\x00\x00" * 16000)
    assert abs(_wav_duration_seconds(str(full)) - 1.0) < 1e-6


# --- capture fails loudly instead of writing an empty WAV ------------------


def test_capture_raises_actionable_error_on_device_format_failure(tmp_path):
    cap = WindowsCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    # Simulate the soundcard WASAPI assertion on the mic thread.
    cap._errors["self"] = AssertionError()
    err = cap._capture_error()
    assert isinstance(err, CaptureError)
    msg = str(err)
    assert "microphone" in msg
    assert "mic_device" in msg  # points the user at the fix


def test_capture_error_generic_for_other_failures(tmp_path):
    cap = WindowsCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    cap._errors["others"] = RuntimeError("device busy")
    msg = str(cap._capture_error())
    assert "system-output" in msg
    assert "device busy" in msg


# --- Linux capture fails loudly instead of writing an empty WAV ------------


def test_linux_capture_raises_actionable_error_on_mic_failure(tmp_path):
    cap = LinuxCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    cap._errors["self"] = RuntimeError("no default source")
    err = cap._capture_error()
    assert isinstance(err, CaptureError)
    msg = str(err)
    assert "microphone" in msg
    assert "mic_device" in msg  # points the user at the fix


def test_linux_capture_error_names_monitor_for_output_failure(tmp_path):
    cap = LinuxCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    cap._errors["others"] = RuntimeError("no monitor source")
    msg = str(cap._capture_error())
    assert "sink monitor" in msg
    assert "output_device" in msg
    assert "no monitor source" in msg


def test_linux_capture_stop_without_start_writes_empty_wavs(tmp_path):
    # No frames captured and no errors -> two empty (0-frame) 16 kHz mono WAVs,
    # rather than a crash. Mirrors a recording that captured pure silence.
    # stop() resamples + writes, so it needs the [capture] extra (numpy/soundfile);
    # skip on the light CI install that has neither.
    import pytest  # noqa: PLC0415

    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    self_wav = tmp_path / "self.wav"
    others_wav = tmp_path / "others.wav"
    cap = LinuxCapture(self_wav, others_wav)
    result = cap.stop()
    assert result.self_wav == self_wav
    assert result.others_wav == others_wav
    # Never started, so duration is 0.0 — not the machine's uptime (_t0 == 0.0).
    assert result.duration == 0.0
    for path in (self_wav, others_wav):
        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 0


# --- Linux device resolution guards ----------------------------------------


class _FakeDevice:
    """Stand-in for a soundcard _Microphone/_Speaker.

    ``name`` optionally raises to mimic a device that has disappeared (a live
    source_info() round trip that IndexErrors).
    """

    def __init__(self, *, name=None, id=None, name_raises=False, isloopback=False):
        self._name = name
        self.id = id
        self._name_raises = name_raises
        self.isloopback = isloopback

    @property
    def name(self):
        if self._name_raises:
            raise IndexError("no soundcard with id")
        return self._name


def test_verify_requested_accepts_substring_match():
    dev = _FakeDevice(name="Jabra Evolve 65 Analog Stereo", id="alsa_input.jabra")
    # A legitimate substring/exact match must not raise.
    _verify_requested("Jabra", dev, "microphone")
    _verify_requested("alsa_input.jabra", dev, "microphone")


def test_verify_requested_rejects_fuzzy_match():
    # soundcard's fuzzy fallback would resolve "Xyz" to this arbitrary device;
    # the requested string appears in neither name nor id, so we fail fast.
    dev = _FakeDevice(name="Built-in Audio Analog Stereo", id="alsa_output.pci")
    try:
        _verify_requested("Xyz", dev, "microphone")
    except CaptureError as err:
        assert "No microphone matches" in str(err)
        assert "hearhere devices" in str(err)
    else:  # pragma: no cover - the guard must raise
        raise AssertionError("expected CaptureError for a fuzzy-only match")


def test_device_label_falls_back_to_id_when_name_round_trip_fails():
    # The error path logs device.name, which IndexErrors once the device is gone;
    # _device_label must not itself throw, so the diagnostic log survives.
    gone = _FakeDevice(id="alsa_input.usb-gone", name_raises=True)
    assert _device_label(gone) == "alsa_input.usb-gone"


# --- mic stream parameter selection (sounddevice) --------------------------


def test_mic_stream_params_uses_native_rate_and_channels():
    rate, channels = _mic_stream_params(
        {"default_samplerate": 48000.0, "max_input_channels": 2}
    )
    assert rate == 48000
    assert channels == 2


def test_mic_stream_params_caps_channels_and_falls_back_on_rate():
    # 18-input console -> capped to 2; missing/zero rate -> 16 kHz fallback.
    rate, channels = _mic_stream_params(
        {"default_samplerate": 0, "max_input_channels": 18}
    )
    assert rate == 16000
    assert channels == 2


def test_mic_stream_params_defaults_for_empty_info():
    rate, channels = _mic_stream_params({})
    assert rate == 16000
    assert channels == 1


# --- monitor detection (others.wav must never be the mic, and vice versa) ---


def test_is_monitor_trusts_the_proplist_when_present():
    # The id deliberately does NOT end in .monitor, so this fails if the
    # isloopback term is ever dropped from _is_monitor.
    assert _is_monitor(_FakeDevice(id="some.virtual.loopback", isloopback=True))
    assert not _is_monitor(_FakeDevice(id="alsa_input.usb-jabra", isloopback=False))


def test_is_monitor_falls_back_to_the_id_when_proplist_is_missing():
    # pipewire-pulse / Bluetooth / virtual sinks don't always set device.class,
    # so isloopback reads False for a source we resolved by its exact monitor
    # id. Rejecting it there would refuse to record a perfectly good setup.
    quiet = _FakeDevice(id="bluez_output.AC_12_2F.1.monitor", isloopback=False)
    assert _is_monitor(quiet)


def test_is_monitor_survives_a_property_that_raises():
    # isloopback round-trips to source_info(), which IndexErrors once the device
    # is gone; the check must fall through to the id rather than blow up.
    class _Gone:
        def __init__(self, id):
            self.id = id

        @property
        def isloopback(self):
            raise IndexError("no soundcard with id")

    assert not _is_monitor(_Gone(id="alsa_input.usb-gone"))
    assert _is_monitor(_Gone(id="alsa_output.pci.monitor"))


# --- per-run capture state -------------------------------------------------


def test_stop_rebinds_frame_buffers_so_an_abandoned_thread_cannot_leak(tmp_path):
    # stop() may give up on a wedged capture thread (join timeout). That thread
    # still holds the list it was handed, so stop() must hand the next run a
    # *new* list rather than clearing the shared one in place — otherwise the
    # orphan's late frames land in the next meeting's WAV.
    import pytest  # noqa: PLC0415

    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    import numpy as np  # noqa: PLC0415

    cap = LinuxCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    orphan_list = cap._mic_frames
    orphan_list.append(np.zeros((8, 1), dtype=np.float32))
    cap.stop()
    assert cap._mic_frames is not orphan_list
    # The orphan keeps writing into the list it was handed; those late frames
    # must not show up in the buffer the next run records into. Reverting
    # stop() to `self._mic_frames.clear()` fails this.
    orphan_list.append(np.zeros((8, 1), dtype=np.float32))
    assert cap._mic_frames == []


def test_start_gives_each_run_a_fresh_stop_event(tmp_path, monkeypatch):
    # start() must not clear() a shared event: an orphaned thread from the
    # previous run would see it cleared and resume its loop into this run.
    # Device resolution is stubbed out so this never opens real hardware — the
    # rebind happens first thing in start(), before any device is touched.
    import pytest  # noqa: PLC0415

    def _no_audio_server(self):
        raise CaptureError("no PulseAudio session")

    monkeypatch.setattr(LinuxCapture, "_resolve_mic", _no_audio_server)

    cap = LinuxCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    # Stand in for a finished run whose capture thread stop() gave up on: it
    # still holds these objects, so start() must hand run 2 fresh ones.
    first = cap._stop
    mic_frames, out_frames = cap._mic_frames, cap._out_frames
    first.set()
    with pytest.raises(CaptureError):
        cap.start()
    assert cap._stop is not first
    assert first.is_set()  # an orphan holding it stays stopped, and exits
    # The frame buffers are rebound for the same reason — this half of the
    # invariant needs no numpy, so unlike the stop() test it always runs.
    assert cap._mic_frames is not mic_frames
    assert cap._out_frames is not out_frames


# --- a failed channel still leaves recoverable audio -----------------------


def test_capture_error_carries_no_meeting_dir_by_default():
    # cli.record only prints the recovery hint when the orchestrator attached a
    # folder, i.e. when stop() had already written the partial audio.
    assert CaptureError("boom").meeting_dir is None
    err = CaptureError("boom")
    err.meeting_dir = "/tmp/meetings/2026-09-18-standup"
    assert "standup" in str(err.meeting_dir)


# --- no-microphone machines ------------------------------------------------


def test_mic_device_none_skips_mic_resolution(tmp_path, monkeypatch):
    # HDMI-only desktops / VMs have no capture device at all, and the default
    # source is the sink monitor. mic_device = "none" must record system output
    # alone rather than refuse — so the mic is never resolved.
    import pytest  # noqa: PLC0415

    def _must_not_run(self):
        raise AssertionError("resolved a mic despite mic_device='none'")

    def _no_sink(self):
        raise CaptureError("stop here, after the mic would have been resolved")

    monkeypatch.setattr(LinuxCapture, "_resolve_mic", _must_not_run)
    monkeypatch.setattr(LinuxCapture, "_resolve_monitor", _no_sink)

    cap = LinuxCapture(
        tmp_path / "self.wav", tmp_path / "others.wav", mic_device=NO_MIC
    )
    with pytest.raises(CaptureError, match="stop here"):
        cap.start()


def test_default_mic_that_is_a_monitor_is_refused_with_a_way_out(tmp_path, monkeypatch):
    # The converse guard: sc.default_microphone() resolves with
    # include_loopback=True, so on a box whose default source is the sink
    # monitor it hands back system output — recording that as self.wav would
    # make both channels identical. Refuse, and name the escape hatch.
    import sys as _sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    import pytest  # noqa: PLC0415

    fake_sc = types.ModuleType("soundcard")
    fake_sc.default_microphone = lambda: _FakeDevice(
        name="Monitor of Built-in Audio", id="alsa_output.pci.monitor"
    )
    monkeypatch.setitem(_sys.modules, "soundcard", fake_sc)

    cap = LinuxCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    with pytest.raises(CaptureError) as err:
        cap._resolve_mic()
    msg = str(err.value)
    assert "Monitor of Built-in Audio" in msg
    assert f'mic_device = "{NO_MIC}"' in msg  # the advice must be actionable


def test_failed_start_resets_the_recording_clock(tmp_path, monkeypatch):
    # _t0 is set only once both devices open, so a start() that fails during
    # resolution must clear the previous run's timestamp — otherwise a later
    # stop() reports the time since the *previous* recording as this duration.
    import pytest  # noqa: PLC0415

    def _no_sink(self):
        raise CaptureError("no PulseAudio session")

    monkeypatch.setattr(LinuxCapture, "_resolve_mic", _no_sink)

    cap = LinuxCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    cap._t0 = 12345.0  # as if a previous recording had run
    with pytest.raises(CaptureError):
        cap.start()
    assert cap._t0 == 0.0


def test_failed_start_leaves_no_recording_error_behind(tmp_path, monkeypatch):
    # self._errors means "a recording died mid-flight" and drives
    # _capture_error(). A device that never opened is a different failure with
    # its own message, so it must not land there — otherwise a later stop()
    # (the start()/stop() contract AudioCapture.__exit__ advertises) overwrites
    # the meeting with empty WAVs and blames a recording that never started.
    import pytest  # noqa: PLC0415

    def _no_sink(self):
        raise OSError("cannot load library 'pulse'")

    monkeypatch.setattr(LinuxCapture, "_resolve_mic", _no_sink)

    cap = LinuxCapture(tmp_path / "self.wav", tmp_path / "others.wav")
    with pytest.raises(CaptureError) as err:
        cap.start()
    assert "libpulse0" in str(err.value)  # the environmental message, not "re-record"
    assert cap._errors == {}
