"""
Gateway Activity Audit Plugin for PraisonAI Agents.

Persists a durable, metadata-only trail of what a gateway/bot did -- inbound
message, tool call/result, outbound delivery, undeliverable reply, error --
into a local SQLite file so operators can answer "what happened, and when?"
after a restart.

It only subscribes to hook events the SDK already fires; no core changes.
Message bodies, tool arguments and tool outputs are never stored.

Configuration (environment):
  - ``PRAISONAI_AUDIT_DB``              SQLite path (default ``~/.praisonai/gateway_audit.db``)
  - ``PRAISONAI_AUDIT_RETENTION_DAYS``  rows older than this are pruned (default 30)

Message events are conversation hooks: grant them with
``plugins.gateway_audit.allow_conversation: true``. Without the grant only
tool/error events are recorded.

Query from Python::

    from praisonai_plugins.observability.gateway_audit import GatewayAuditPlugin
    GatewayAuditPlugin().query(platform="telegram", since=time.time() - 3600)
"""
from __future__ import annotations

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

_DEFAULT_DB = os.path.join("~", ".praisonai", "gateway_audit.db")
_DEFAULT_RETENTION_DAYS = 30.0
# Prune expired rows once every N inserts so retention costs nothing per event.
_PRUNE_EVERY = 500
# Non-secret identity fields copied from message payloads; ``content`` is not.
_MESSAGE_FIELDS = ("platform", "channel_id", "channel_type", "sender_id",
                   "session_id", "message_id")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    platform TEXT,
    channel_id TEXT,
    sender_id TEXT,
    session_id TEXT,
    message_id TEXT,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_events(ts);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_events(session_id);
"""


def _retention_days() -> float:
    raw = os.environ.get("PRAISONAI_AUDIT_RETENTION_DAYS", "").strip()
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning(
                f"Invalid PRAISONAI_AUDIT_RETENTION_DAYS={raw!r}; using default 30"
            )
    return _DEFAULT_RETENTION_DAYS


class GatewayAuditPlugin(Plugin):
    """Durable, metadata-only activity audit trail for gateways and bots."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = os.path.expanduser(
            db_path or os.environ.get("PRAISONAI_AUDIT_DB") or _DEFAULT_DB
        )
        self._retention_s = _retention_days() * 86400
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._inserts = 0
        self._warned = False

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="gateway_audit",
            version="1.0.0",
            description="Durable metadata-only audit trail of gateway activity (SQLite).",
            author="PraisonAI",
            hooks=[PluginHook.ON_ERROR],
        )

    # ------------------------------------------------------------------ hooks
    # Every hook returns None so the plugin never rewrites or blocks anything.

    def before_message(self, message: dict[str, Any]) -> None:
        self._record_message("message_received", message)

    def message_sent(self, message: dict[str, Any]) -> None:
        self._record_message("message_sent", message)

    def message_undelivered(self, message: dict[str, Any]) -> None:
        self._record_message("message_undelivered", message, {
            "error": str(message.get("error", ""))[:200],
            "notice_delivered": bool(message.get("notice_delivered", False)),
        })

    def before_tool(self, tool_name: str, args: dict[str, Any]) -> None:
        self.record("tool_call", detail={"tool": tool_name})

    def after_tool(self, tool_name: str, result: Any) -> None:
        failed = isinstance(result, Exception) or (
            isinstance(result, dict) and result.get("error")
        )
        outcome = "error" if failed else "ok"
        self.record("tool_result", detail={"tool": tool_name, "outcome": outcome})

    def on_error(self, error_type: str, error_message: str, context: dict[str, Any]) -> None:
        self.record("error", detail={"type": error_type})

    def on_shutdown(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ---------------------------------------------------------------- public

    def record(self, kind: str, fields: dict[str, Any] | None = None,
               detail: dict[str, Any] | None = None) -> None:
        """Append one event. Never raises -- auditing must not break a turn."""
        fields = fields or {}
        with self._lock:
            try:
                conn = self._connect()
                conn.execute(
                    "INSERT INTO audit_events (ts, kind, platform, channel_id, sender_id,"
                    " session_id, message_id, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (time.time(), kind, fields.get("platform"), fields.get("channel_id"),
                     fields.get("sender_id"), fields.get("session_id"),
                     fields.get("message_id"), json.dumps(detail) if detail else None),
                )
                self._inserts += 1
                if self._inserts % _PRUNE_EVERY == 0:
                    self._prune(conn)
                conn.commit()
            except Exception as exc:  # noqa: BLE001 - auditing must never break a turn
                # Roll back so a failed write never leaves the DB write-locked.
                if self._conn is not None:
                    with contextlib.suppress(sqlite3.Error):
                        self._conn.rollback()
                if not self._warned:
                    self._warned = True
                    logger.warning(f"gateway_audit: audit trail write failed ({exc}); "
                                   "further failures are logged at debug level")
                else:
                    logger.debug(f"gateway_audit: failed to record {kind}: {exc}")

    def query(self, *, kind: str | None = None, platform: str | None = None,
              channel_id: str | None = None, sender_id: str | None = None,
              session_id: str | None = None, since: float | None = None,
              limit: int = 100) -> list[dict[str, Any]]:
        """Return matching events, newest first. ``since`` is a Unix timestamp."""
        filters = {"kind": kind, "platform": platform, "channel_id": channel_id,
                   "sender_id": sender_id, "session_id": session_id}
        clauses = [f"{col} = ?" for col, val in filters.items() if val is not None]
        params: list[Any] = [val for val in filters.values() if val is not None]
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                f"SELECT * FROM audit_events{where} ORDER BY ts DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
        events = [dict(row) for row in rows]
        for event in events:
            event["detail"] = json.loads(event["detail"]) if event["detail"] else {}
        return events

    # -------------------------------------------------------------- internals

    def _record_message(self, kind: str, message: dict[str, Any],
                        detail: dict[str, Any] | None = None) -> None:
        fields = {k: str(message[k]) for k in _MESSAGE_FIELDS if message.get(k) is not None}
        self.record(kind, fields, detail)

    def _connect(self) -> sqlite3.Connection:
        # Caller holds self._lock. Opened lazily so installing the plugin costs
        # nothing until the first event.
        if self._conn is None:
            parent = os.path.dirname(self._db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA)
            self._prune(conn)
            conn.commit()
            self._conn = conn
        return self._conn

    def _prune(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM audit_events WHERE ts < ?",
                     (time.time() - self._retention_s,))


def create_plugin() -> GatewayAuditPlugin:
    """Factory used by directory/module loaders."""
    return GatewayAuditPlugin()
