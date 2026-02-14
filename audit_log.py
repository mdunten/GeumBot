"""Audit logger for web requests made by GeumBot.

Every Brave search and web fetch is recorded with a timestamp, action type,
target (query or URL), status, and a short summary of the result.  Entries are
written as single-line JSON objects to make them easy to parse programmatically.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_FILE = "geumbot_audit.log"


class AuditLogger:
    """Append-only audit log for outbound web requests."""

    def __init__(self, log_file: str | Path = DEFAULT_LOG_FILE):
        self._path = Path(log_file)
        self._logger = logging.getLogger("geumbot.audit")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = logging.FileHandler(self._path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def log(
        self,
        action: str,
        target: str,
        status: str,
        result_summary: str = "",
    ) -> None:
        """Write a single audit entry.

        Parameters
        ----------
        action:
            ``"brave_search"`` or ``"web_fetch"``.
        target:
            The search query or URL.
        status:
            ``"ok"`` or ``"error"``.
        result_summary:
            A short description of the outcome (e.g. result count or error message).
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "target": target,
            "status": status,
            "summary": result_summary,
        }
        self._logger.info(json.dumps(entry))
