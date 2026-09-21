"""Local web UI for reviewing past meetings (Batch 6).

HearHere is local-first; this UI is a convenience for browsing the meetings
already on disk — it never records or uploads audio. Two layers live here:

* :class:`~hearhere.webui.service.WebUIService` — the pure core. It lists
  meetings under ``storage_dir``, reads ``meeting.json``, re-exports, and renames
  speakers, all by delegating to the existing pipeline/export helpers. It imports
  no web framework, so it is unit-testable without the ``[webui]`` extra.
* :func:`~hearhere.webui.server.create_app` / :func:`~hearhere.webui.server.run_ui`
  — a thin FastAPI wrapper serving a single-page browser frontend plus a small
  JSON API. FastAPI/uvicorn are imported lazily; install them with
  ``pip install "hearhere[webui]"``.
"""

from __future__ import annotations

from .service import WebUIService, WebUIError

__all__ = ["WebUIService", "WebUIError"]
