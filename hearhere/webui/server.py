"""FastAPI wrapper + single-page frontend for the local web UI.

This is a thin shell over :class:`~hearhere.webui.service.WebUIService`: it maps
HTTP routes onto the service and serves ``static/index.html`` at ``/``. FastAPI
and uvicorn are imported lazily so the rest of HearHere works without the
``[webui]`` extra.

.. note::
   This module deliberately avoids ``from __future__ import annotations``:
   FastAPI resolves endpoint parameter annotations at route registration and the
   web-stack types are imported lazily inside :func:`create_app`. Request-body models are ordinary module globals so
   FastAPI can resolve them. Runtime ``X | None`` unions are fine on 3.10+.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from .. import __version__
from ..config import Config
from ..logging_setup import get_logger
from .service import WebUIError, WebUIService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

log = get_logger("webui.server")

_INDEX_HTML = Path(__file__).parent / "static" / "index.html"


class RenameRequest(BaseModel):
    """Body for ``POST /api/meetings/{id}/speakers``."""

    renames: dict[str, str] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    """Body for ``POST /api/meetings/{id}/export`` (``formats`` omitted = config)."""

    formats: Optional[list[str]] = None


def create_app(config: Config, *, service: Optional[WebUIService] = None) -> "FastAPI":
    """Build the FastAPI app serving the meeting-review UI and its JSON API."""
    from fastapi import FastAPI, HTTPException  # noqa: PLC0415
    from fastapi.responses import HTMLResponse, Response  # noqa: PLC0415

    svc = service or WebUIService(config)
    app = FastAPI(title="HearHere web UI", version=__version__)

    def _guard(fn):
        try:
            return fn()
        except WebUIError as exc:
            raise HTTPException(status_code=exc.code, detail=str(exc))

    @app.get("/", response_class=HTMLResponse)
    def index() -> "HTMLResponse":
        if not _INDEX_HTML.is_file():  # pragma: no cover - packaging guard
            raise HTTPException(status_code=500, detail="frontend asset missing")
        return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))

    @app.get("/api/meetings")
    def list_meetings() -> dict:
        return {"meetings": svc.list_meetings()}

    @app.get("/api/meetings/{meeting_id}")
    def get_meeting(meeting_id: str) -> dict:
        return _guard(lambda: svc.get_meeting(meeting_id))

    @app.post("/api/meetings/{meeting_id}/export")
    def export_meeting(meeting_id: str, body: ExportRequest) -> dict:
        written = _guard(lambda: svc.export(meeting_id, body.formats))
        return {"written": written}

    @app.post("/api/meetings/{meeting_id}/speakers")
    def rename_speakers(meeting_id: str, body: RenameRequest) -> dict:
        return _guard(lambda: svc.rename_speakers(meeting_id, body.renames))

    @app.get("/api/meetings/{meeting_id}/exports/{filename}")
    def download_export(meeting_id: str, filename: str) -> "Response":
        text, media = _guard(lambda: svc.read_export(meeting_id, filename))
        return Response(content=text, media_type=media)

    return app


def run_ui(
    config: Config,
    *,
    host: str = "127.0.0.1",
    port: int = 8809,
) -> None:  # pragma: no cover - starts a server
    """Serve the web UI with uvicorn (blocking).

    Binds to loopback by default: this is a personal review tool, not a service.
    """
    import uvicorn  # noqa: PLC0415

    app = create_app(config)
    log.info("Starting HearHere web UI on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
