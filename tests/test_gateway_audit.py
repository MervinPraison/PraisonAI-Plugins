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


def test_message_metadata_recorded_without_body(plugin):
    msg = {"content": "my secret prompt", "platform": "telegram",
           "sender_id": "alice", "channel_id": "c1", "session_id": "s1",
           "message_id": "m1"}
    assert plugin.before_message(msg) is None
    plugin.message_sent(dict(msg, content="reply"))

    events = plugin.query(session_id="s1")
    assert [e["kind"] for e in events] == ["message_sent", "message_received"]
    assert events[1]["sender_id"] == "alice"
    assert all("secret" not in str(e) and "reply" not in str(e) for e in events)


def test_tool_events_never_rewrite_or_store_args(plugin):
    assert plugin.before_tool("shell", {"cmd": "rm -rf /tmp/x"}) is None
    assert plugin.after_tool("shell", {"error": "boom"}) is None

    events = plugin.query()
    assert events[0]["detail"] == {"tool": "shell", "outcome": "error"}
    assert events[1]["detail"] == {"tool": "shell"}
    assert "rm -rf" not in str(events)


def test_undelivered_and_filters(plugin):
    plugin.message_undelivered({"platform": "slack", "sender_id": "bob",
                                "error": "blocked_by_user", "notice_delivered": False})
    plugin.record("tool_call", detail={"tool": "x"})

    events = plugin.query(platform="slack")
    assert len(events) == 1
    assert events[0]["detail"]["error"] == "blocked_by_user"
    assert plugin.query(kind="tool_call", limit=5)[0]["detail"] == {"tool": "x"}
    assert plugin.query(since=time.time() + 60) == []


def test_survives_restart_and_prunes_expired(tmp_path):
    db = str(tmp_path / "audit.db")
    first = audit_mod.GatewayAuditPlugin(db_path=db)
    first.record("tool_call", detail={"tool": "kept"})
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
    # Conversation hooks stay gated until the operator grants allow_conversation.
    assert not registry.has_hooks(HookEvent.MESSAGE_RECEIVED)


def test_granted_message_events_dispatch_through_bridge(plugin):
    from praisonaiagents.hooks.events import MessageReceivedInput, MessageUndeliveredInput
    from praisonaiagents.hooks.registry import HookRegistry
    from praisonaiagents.hooks.types import HookEvent
    from praisonaiagents.plugins.manager import PluginManager

    manager = PluginManager()
    manager.set_plugin_options({"gateway_audit": {"allow_conversation": True}})
    assert manager.register(plugin)
    registry = HookRegistry()
    manager.wire_into_hook_registry(registry)

    common = {"cwd": "/", "timestamp": "0", "session_id": "s9"}
    inbound = MessageReceivedInput(event_name="message_received", platform="discord",
                                   content="private text", sender_id="carol",
                                   channel_id="c9", message_id="m9", **common)
    failed = MessageUndeliveredInput(event_name="message_undelivered", platform="discord",
                                     content="private reply", channel_id="c9",
                                     error="forbidden", **common)
    for event, data in ((HookEvent.MESSAGE_RECEIVED, inbound),
                        (HookEvent.MESSAGE_UNDELIVERED, failed)):
        hooks = registry.get_hooks(event)
        assert len(hooks) == 1
        hooks[0].func(data)

    events = plugin.query(session_id="s9")
    assert [e["kind"] for e in events] == ["message_undelivered", "message_received"]
    assert events[1]["sender_id"] == "carol"
    assert events[0]["detail"]["error"] == "forbidden"
    assert "private" not in str(events)


def test_failed_commit_rolls_back_and_warns_once(plugin, monkeypatch):
    plugin.record("tool_call", detail={"tool": "warmup"})
    real = plugin._conn
    warnings = []
    monkeypatch.setattr(audit_mod.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    class _FailingCommit:
        def __getattr__(self, name):
            return getattr(real, name)

        def commit(self):
            raise audit_mod.sqlite3.OperationalError("database is locked")

    plugin._conn = _FailingCommit()
    plugin.record("tool_call", detail={"tool": "lost1"})
    plugin.record("tool_call", detail={"tool": "lost2"})
    assert len(warnings) == 1
    assert not real.in_transaction  # write lock released

    plugin._conn = real
    plugin.record("tool_call", detail={"tool": "after"})
    assert [e["detail"]["tool"] for e in plugin.query()] == ["after", "warmup"]
