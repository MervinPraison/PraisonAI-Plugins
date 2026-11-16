"""Gateway audit plugin: durable SQLite trail of tool calls, tool results and errors.

Stores metadata only (never tool args, outputs or error messages). Env config:
``PRAISONAI_AUDIT_DB`` (``~/.praisonai/gateway_audit.db``), ``PRAISONAI_AUDIT_RETENTION_DAYS`` (30)."""

import contextlib
import json
import os
import sqlite3
import threading
import time
from typing import Any

from praisonaiagents._logging import get_logger
from praisonaiagents.plugins.plugin import Plugin, PluginHook, PluginInfo

logger = get_logger(__name__)

_SCHEMA = """CREATE TABLE IF NOT EXISTS audit_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, kind TEXT NOT NULL, detail TEXT);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_events(ts);"""
_PRUNE = "DELETE FROM audit_events WHERE ts < ?"


class GatewayAuditPlugin(Plugin):
    """Durable, metadata-only activity audit trail. Hooks never rewrite or block."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = os.path.expanduser(db_path or os.environ.get(
            "PRAISONAI_AUDIT_DB") or os.path.join("~", ".praisonai", "gateway_audit.db"))
        raw = os.environ.get("PRAISONAI_AUDIT_RETENTION_DAYS", "").strip()
        days = float(raw) if raw.replace(".", "", 1).isdigit() and float(raw) > 0 else 30.0
        self._retention_s = days * 86400
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._inserts = 0
        self._warned = False

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(name="gateway_audit", version="1.0.0", author="PraisonAI",
                          description="Durable metadata-only audit trail (SQLite).",
                          hooks=[PluginHook.ON_ERROR])

    def before_tool(self, tool_name: str, args: dict[str, Any]) -> None:
        self.record("tool_call", {"tool": tool_name})

    def after_tool(self, tool_name: str, result: Any) -> None:
        failed = isinstance(result, Exception) or (isinstance(result, dict) and result.get("error"))
        self.record("tool_result", {"tool": tool_name, "outcome": "error" if failed else "ok"})

    def on_error(self, error_type: str, error_message: str, context: dict[str, Any]) -> None:
        self.record("error", {"type": error_type})

    def on_shutdown(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def record(self, kind: str, detail: dict[str, Any] | None = None) -> None:
        """Append one event. Never raises -- auditing must not break a turn."""
        with self._lock:
            try:
                conn = self._connect()
                conn.execute("INSERT INTO audit_events (ts, kind, detail) VALUES (?, ?, ?)",
                             (time.time(), kind, json.dumps(detail) if detail else None))
                self._inserts += 1
                if self._inserts % 500 == 0:  # prune every N inserts, not per event
                    conn.execute(_PRUNE, (time.time() - self._retention_s,))
                conn.commit()
            except Exception as exc:  # noqa: BLE001 - auditing must never break a turn
                if self._conn is not None:  # release the write lock
                    with contextlib.suppress(sqlite3.Error):
                        self._conn.rollback()
                log = logger.debug if self._warned else logger.warning
                self._warned = True
                log(f"gateway_audit: failed to record {kind}: {exc}")

    def query(self, *, kind: str | None = None, since: float | None = None,
              limit: int = 100) -> list[dict[str, Any]]:
        """Return matching events, newest first. ``since`` is a Unix timestamp."""
        sql = ("SELECT * FROM audit_events WHERE (? IS NULL OR kind = ?) AND (? IS NULL OR ts >= ?)"
               " ORDER BY ts DESC, id DESC LIMIT ?")
        with self._lock:
            rows = self._connect().execute(sql, (kind, kind, since, since, int(limit))).fetchall()
        return [dict(r, detail=json.loads(r["detail"]) if r["detail"] else {}) for r in rows]

    def _connect(self) -> sqlite3.Connection:  # caller holds the lock; opened lazily
        if self._conn is None:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA)
            conn.execute(_PRUNE, (time.time() - self._retention_s,))
            conn.commit()
            self._conn = conn
        return self._conn
