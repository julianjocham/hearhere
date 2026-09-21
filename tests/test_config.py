"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from hearhere.config import Config, find_config_file, load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.toml"


def test_defaults_are_valid():
    cfg = Config()
    assert cfg.general.language == "auto"
    assert cfg.compute.device == "auto"
    assert cfg.capture.sample_rate == 16000
    assert cfg.asr.model == "nvidia/parakeet-tdt-0.6b-v3"
    assert cfg.diarization.enabled is True
    assert cfg.llm.enabled is False  # summarization is opt-in; base run transcribes only
    assert cfg.llm.tasks == ["summary", "action_items", "decisions"]
    assert cfg.export.formats == ["markdown", "json", "srt"]


def test_no_file_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.config/hearhere either
    assert find_config_file() is None
    cfg = load_config()
    assert cfg == Config()


def test_example_config_loads_and_matches_defaults():
    assert EXAMPLE.is_file()
    cfg = load_config(EXAMPLE)
    # The example file documents the defaults; storage_dir is expanded.
    assert cfg.general.storage_dir == (Path("~/HearHere").expanduser())
    assert cfg.export.formats == ["markdown", "json", "srt"]
    assert cfg.compute.device == "auto"


def test_storage_dir_is_expanded():
    cfg = Config.model_validate({"general": {"storage_dir": "~/somewhere"}})
    assert cfg.general.storage_dir == Path("~/somewhere").expanduser()
    assert "~" not in str(cfg.general.storage_dir)


def test_cwd_precedes_user_config(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text('[general]\nlanguage = "de"\n')
    monkeypatch.chdir(tmp_path)
    found = find_config_file()
    assert found == tmp_path / "config.toml"
    assert load_config().general.language == "de"


def test_hf_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    cfg = Config()
    assert cfg.diarization.hf_token == "secret-token"


def test_explicit_token_not_overridden_by_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-token")
    cfg = Config.model_validate({"diarization": {"hf_token": "file-token"}})
    assert cfg.diarization.hf_token == "file-token"


def test_invalid_device_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({"compute": {"device": "quantum"}})


def test_unknown_top_level_key_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({"nonsense": {}})


def test_empty_export_formats_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({"export": {"formats": []}})


def test_speaker_bounds_validation():
    with pytest.raises(ValidationError):
        Config.model_validate({"diarization": {"min_speakers": 5, "max_speakers": 2}})
    # Auto (0) bounds are fine.
    Config.model_validate({"diarization": {"min_speakers": 0, "max_speakers": 0}})


def test_missing_explicit_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_config_file(tmp_path / "nope.toml")
