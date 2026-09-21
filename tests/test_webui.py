"""Batch 6 tests: local web UI service core + HTTP API + frontend.

The service core is exercised directly (no web stack); the FastAPI app is driven
in-process with Starlette's ``TestClient``. Covers the "done when": review,
rename, and export a past meeting entirely through the API the browser uses.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from hearhere.config import Config
from hearhere.models import Meeting, Segment, Summary, Transcript
from hearhere.pipeline import artifacts
from hearhere.webui.server import create_app
from hearhere.webui.service import WebUIError, WebUIService


def _config(tmp_path) -> Config:
    return Config.model_validate(
        {"general": {"storage_dir": str(tmp_path)},
         "export": {"formats": ["markdown", "json"]}}
    )


def _make_meeting(tmp_path, title="Weekly Sync") -> str:
    """Write a processed meeting (meeting.json + exports) under storage_dir."""
    paths = artifacts.create_meeting_dir(tmp_path, title)
    meeting = Meeting(
        id=paths.root.name,
        title=title,
        speakers=["Me", "Speaker 1"],
        transcript=Transcript(
            segments=[
                Segment(start=0.0, end=2.0, text="Let's start.", speaker="Me"),
                Segment(start=3.0, end=6.0, text="Staging is green.", speaker="Speaker 1"),
            ],
            language="en",
        ),
        summary=Summary(
            summary="Speaker 1 confirmed staging.",
            decisions=["Ship Friday."],
            action_items=["Speaker 1 to release."],
        ),
    )
    artifacts.write_meeting(paths, meeting)
    from hearhere.export import export_meeting

    export_meeting(paths, meeting, ["markdown", "json"])
    return paths.root.name


# --- service core ----------------------------------------------------------


def test_service_lists_and_reads(tmp_path):
    mid = _make_meeting(tmp_path)
    svc = WebUIService(_config(tmp_path))

    listed = svc.list_meetings()
    assert len(listed) == 1
    assert listed[0]["id"] == mid
    assert listed[0]["has_summary"] is True
    assert listed[0]["segments"] == 2

    meeting = svc.get_meeting(mid)
    assert meeting["title"] == "Weekly Sync"
    assert meeting["speakers"] == ["Me", "Speaker 1"]
    assert "transcript.md" in meeting["exports"]
    assert "summary.md" in meeting["exports"]


def test_service_rename_propagates(tmp_path):
    mid = _make_meeting(tmp_path)
    svc = WebUIService(_config(tmp_path))

    updated = svc.rename_speakers(mid, {"Speaker 1": "Anna"})
    assert updated["speakers"] == ["Me", "Anna"]

    # Rename rewrote meeting.json and re-exported.
    paths = artifacts.resolve_meeting(tmp_path / mid)
    on_disk = json.loads(paths.meeting_json.read_text(encoding="utf-8"))
    assert on_disk["speakers"] == ["Me", "Anna"]
    assert "Anna" in paths.export("transcript.md").read_text(encoding="utf-8")
    assert "Anna" in paths.export("summary.md").read_text(encoding="utf-8")


def test_service_export_selected_formats(tmp_path):
    mid = _make_meeting(tmp_path)
    svc = WebUIService(_config(tmp_path))
    written = svc.export(mid, ["srt"])
    assert "transcript.srt" in written
    assert (tmp_path / mid / "transcript.srt").is_file()


def test_service_rejects_bad_ids(tmp_path):
    svc = WebUIService(_config(tmp_path))
    for bad in ["../secrets", "a/b", "..", ""]:
        with pytest.raises(WebUIError):
            svc.get_meeting(bad)


def test_service_unknown_meeting_404(tmp_path):
    svc = WebUIService(_config(tmp_path))
    with pytest.raises(WebUIError) as exc:
        svc.get_meeting("2026-01-01_nope")
    assert exc.value.code == 404


def test_service_unknown_format_rejected(tmp_path):
    mid = _make_meeting(tmp_path)
    svc = WebUIService(_config(tmp_path))
    with pytest.raises(WebUIError):
        svc.export(mid, ["docx"])


# --- HTTP API (what the browser calls) -------------------------------------


def test_http_index_serves_frontend(tmp_path):
    client = TestClient(create_app(_config(tmp_path)))
    r = client.get("/")
    assert r.status_code == 200
    assert "HearHere" in r.text
    assert "text/html" in r.headers["content-type"]


def test_http_review_rename_export_end_to_end(tmp_path):
    """The done-when, over HTTP: review -> rename -> export a past meeting."""
    mid = _make_meeting(tmp_path)
    client = TestClient(create_app(_config(tmp_path)))

    # Review
    listing = client.get("/api/meetings").json()["meetings"]
    assert [m["id"] for m in listing] == [mid]
    detail = client.get(f"/api/meetings/{mid}").json()
    assert detail["summary"]["decisions"] == ["Ship Friday."]

    # Rename inline
    r = client.post(f"/api/meetings/{mid}/speakers",
                    json={"renames": {"Speaker 1": "Anna"}})
    assert r.status_code == 200
    assert r.json()["speakers"] == ["Me", "Anna"]

    # Export
    r = client.post(f"/api/meetings/{mid}/export", json={"formats": ["srt"]})
    assert r.status_code == 200
    assert "transcript.srt" in r.json()["written"]

    # Download an export and confirm the rename propagated to it
    r = client.get(f"/api/meetings/{mid}/exports/transcript.md")
    assert r.status_code == 200
    assert "Anna" in r.text


def test_http_unknown_meeting_404(tmp_path):
    client = TestClient(create_app(_config(tmp_path)))
    assert client.get("/api/meetings/2026-01-01_nope").status_code == 404


def test_http_missing_export_404(tmp_path):
    mid = _make_meeting(tmp_path)
    client = TestClient(create_app(_config(tmp_path)))
    # A valid export name that hasn't been generated for this meeting.
    assert client.get(f"/api/meetings/{mid}/exports/transcript.vtt").status_code == 404
