"""Tests for the GatewayAuditPlugin (durable metadata-only activity trail)."""

import importlib
import os
import sys
import time

import pytest

pytest.importorskip("praisonaiagents")

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

audit_mod = importlib.import_module("praisonai_plugins.observability.gateway_audit")


@pytest.fixture
def plugin(tmp_path):
    p = audit_mod.GatewayAuditPlugin(db_path=str(tmp_path / "audit.db"))
    yield p
    p.on_shutdown()


def test_tool_events_never_rewrite_or_store_args(plugin):
    assert plugin.before_tool("shell", {"cmd": "rm -rf /tmp/x"}) is None
    assert plugin.after_tool("shell", {"error": "boom"}) is None

    events = plugin.query()
    assert events[0]["detail"] == {"tool": "shell", "outcome": "error"}
    assert events[1]["detail"] == {"tool": "shell"}
    assert "rm -rf" not in str(events)


def test_error_and_filters(plugin):
    assert plugin.on_error("ValueError", "secret detail", {"k": "v"}) is None
    plugin.record("tool_call", {"tool": "x"})

    assert plugin.query(kind="error")[0]["detail"] == {"type": "ValueError"}
    assert "secret detail" not in str(plugin.query())
    assert plugin.query(kind="tool_call", limit=5)[0]["detail"] == {"tool": "x"}
    assert plugin.query(since=time.time() + 60) == []


def test_survives_restart_and_prunes_expired(tmp_path):
    db = str(tmp_path / "audit.db")
    first = audit_mod.GatewayAuditPlugin(db_path=db)
    first.record("tool_call", {"tool": "kept"})
    first._conn.execute("INSERT INTO audit_events (ts, kind) VALUES (?, 'old')",
                        (time.time() - 400 * 86400,))
    first._conn.commit()
    first.on_shutdown()

    second = audit_mod.GatewayAuditPlugin(db_path=db)
    kinds = [e["kind"] for e in second.query()]
    second.on_shutdown()
    assert kinds == ["tool_call"]


def test_wires_through_sdk_plugin_bridge(plugin):
    from praisonaiagents.hooks.registry import HookRegistry
    from praisonaiagents.hooks.types import HookEvent
    from praisonaiagents.plugins.manager import PluginManager

    manager = PluginManager()
    assert manager.register(plugin)
    registry = HookRegistry()
    manager.wire_into_hook_registry(registry)
    for event in (HookEvent.BEFORE_TOOL, HookEvent.AFTER_TOOL, HookEvent.ON_ERROR):
        assert registry.has_hooks(event)
    # This slice registers no conversation hooks.
    assert not registry.has_hooks(HookEvent.MESSAGE_RECEIVED)


def test_failed_commit_rolls_back_and_warns_once(plugin, monkeypatch):
    plugin.record("tool_call", {"tool": "warmup"})
    real = plugin._conn
    warnings = []
    monkeypatch.setattr(audit_mod.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    class _FailingCommit:
        def __getattr__(self, name):
            return getattr(real, name)

        def commit(self):
            raise audit_mod.sqlite3.OperationalError("database is locked")

    plugin._conn = _FailingCommit()
    plugin.record("tool_call", {"tool": "lost1"})
    plugin.record("tool_call", {"tool": "lost2"})
    assert len(warnings) == 1
    assert not real.in_transaction  # write lock released

    plugin._conn = real
    plugin.record("tool_call", {"tool": "after"})
    assert [e["detail"]["tool"] for e in plugin.query()] == ["after", "warmup"]
