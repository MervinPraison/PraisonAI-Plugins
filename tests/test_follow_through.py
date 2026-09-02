"""Self-contained tests for the FollowThroughPlugin.

These tests stub the small ``praisonaiagents`` plugin/logging surface the
plugin imports so they run without the full SDK installed. They verify:

  * the plugin never mutates the turn (``after_agent`` returns input unchanged),
  * it is off unless enabled (operator switch),
  * heuristic extraction finds agent promises and user open-loops,
  * the confidence threshold and pending cap are enforced,
  * the store dedupes so the same loop is never scheduled twice,
  * extraction failures never propagate to the response path.
"""

import sys
import types
import logging
import importlib

import pytest


def _install_sdk_stubs():
    if "praisonaiagents" in sys.modules:
        return

    root = types.ModuleType("praisonaiagents")

    logging_mod = types.ModuleType("praisonaiagents._logging")
    logging_mod.get_logger = lambda name: logging.getLogger(name)

    plugins_pkg = types.ModuleType("praisonaiagents.plugins")
    plugin_mod = types.ModuleType("praisonaiagents.plugins.plugin")

    from dataclasses import dataclass, field
    from enum import Enum
    from typing import List

    class PluginHook(str, Enum):
        BEFORE_AGENT = "before_agent"
        AFTER_AGENT = "after_agent"

    @dataclass
    class PluginInfo:
        name: str
        version: str = "1.0.0"
        description: str = ""
        author: str = ""
        hooks: List = field(default_factory=list)
        dependencies: List = field(default_factory=list)

    class Plugin:
        def on_init(self, context):
            pass

        def on_shutdown(self):
            pass

        def on_config(self, config):
            return config

    plugin_mod.Plugin = Plugin
    plugin_mod.PluginInfo = PluginInfo
    plugin_mod.PluginHook = PluginHook

    sys.modules["praisonaiagents"] = root
    sys.modules["praisonaiagents._logging"] = logging_mod
    sys.modules["praisonaiagents.plugins"] = plugins_pkg
    sys.modules["praisonaiagents.plugins.plugin"] = plugin_mod


@pytest.fixture(scope="module")
def ft_module():
    _install_sdk_stubs()
    import os

    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    mod = importlib.import_module("praisonai_plugins.hooks.follow_through")
    return mod


def test_info_declares_after_agent_hook(ft_module):
    from praisonaiagents.plugins.plugin import PluginHook

    plugin = ft_module.FollowThroughPlugin()
    info = plugin.info
    assert info.name == "follow_through"
    assert PluginHook.AFTER_AGENT in info.hooks


def test_disabled_by_default_is_noop(ft_module):
    plugin = ft_module.FollowThroughPlugin()
    plugin._enabled = False
    resp = "I'll follow up after your interview tomorrow."
    out = plugin.after_agent(resp, {"transcript": "user: I have an interview"})
    assert out == resp
    assert len(plugin.store) == 0


def test_after_agent_never_mutates_response(ft_module):
    plugin = ft_module.FollowThroughPlugin()
    plugin.on_config({"enabled": True})
    resp = "I'll check back after your interview tomorrow."
    out = plugin.after_agent(resp, {"transcript": "user: I have an interview tomorrow"})
    assert out == resp


def test_heuristic_extracts_agent_promise(ft_module):
    candidates = ft_module._heuristic_extract(
        "user: thanks",
        "Good luck! I'll check in after your interview tomorrow.",
    )
    kinds = {c["kind"] for c in candidates}
    assert ft_module.CommitmentKind.AGENT_PROMISE in kinds


def test_heuristic_extracts_user_deadline(ft_module):
    candidates = ft_module._heuristic_extract(
        "user: my deadline is Friday",
        "Noted.",
    )
    kinds = {c["kind"] for c in candidates}
    assert ft_module.CommitmentKind.DEADLINE in kinds


def test_enabled_schedules_and_persists(ft_module):
    plugin = ft_module.FollowThroughPlugin()
    plugin.on_config({"enabled": True})
    plugin.after_agent(
        "Good luck! I'll check in after your interview tomorrow.",
        {"transcript": "user: I have an interview tomorrow", "channel": "slack:#gen"},
    )
    assert len(plugin.store) >= 1


def test_store_dedupes_same_loop(ft_module):
    store = ft_module.CommitmentStore()
    c = ft_module.Commitment(
        kind=ft_module.CommitmentKind.EVENT_CHECK_IN,
        summary="interview",
        dedupe_key="abc123",
        due_at=0.0,
        confidence=0.9,
    )
    assert store.add_if_new(c) is True
    assert store.add_if_new(c) is False
    assert len(store) == 1


def test_pending_cap_enforced(ft_module):
    store = ft_module.CommitmentStore(max_pending=1)
    first = ft_module.Commitment(
        kind=ft_module.CommitmentKind.DEADLINE,
        summary="a",
        dedupe_key="k1",
        due_at=0.0,
        confidence=0.9,
    )
    second = ft_module.Commitment(
        kind=ft_module.CommitmentKind.DEADLINE,
        summary="b",
        dedupe_key="k2",
        due_at=0.0,
        confidence=0.9,
    )
    assert store.add_if_new(first) is True
    assert store.add_if_new(second) is False


def test_confidence_threshold_filters(ft_module):
    plugin = ft_module.FollowThroughPlugin()
    plugin.on_config({"enabled": True, "threshold": 0.99})
    plugin.after_agent(
        "I'll follow up after your interview tomorrow.",
        {"transcript": "user: interview tomorrow"},
    )
    assert len(plugin.store) == 0


def test_extraction_failure_isolated(ft_module, monkeypatch):
    plugin = ft_module.FollowThroughPlugin()
    plugin.on_config({"enabled": True})

    def boom(*args, **kwargs):
        raise RuntimeError("extraction exploded")

    monkeypatch.setattr(ft_module, "_heuristic_extract", boom)
    monkeypatch.setattr(ft_module, "_llm_extract", lambda *a, **k: None)
    resp = "I'll follow up tomorrow."
    out = plugin.after_agent(resp, {"transcript": "user: hi"})
    assert out == resp


def test_status_lifecycle_transitions(ft_module):
    store = ft_module.CommitmentStore()
    c = ft_module.Commitment(
        kind=ft_module.CommitmentKind.AGENT_PROMISE,
        summary="x",
        dedupe_key="life1",
        due_at=0.0,
        confidence=0.9,
    )
    store.add_if_new(c)
    assert store.set_status("life1", ft_module.CommitmentStatus.SENT) is True
    assert store.get("life1").status == ft_module.CommitmentStatus.SENT
    assert store.set_status("missing", ft_module.CommitmentStatus.SENT) is False


def test_dedupe_key_is_stable(ft_module):
    k1 = ft_module._dedupe_key("slack", "s1", "event_check_in", "interview")
    k2 = ft_module._dedupe_key("slack", "s1", "event_check_in", "interview")
    k3 = ft_module._dedupe_key("slack", "s1", "event_check_in", "meeting")
    assert k1 == k2
    assert k1 != k3


def test_parse_llm_json_extracts_array(ft_module):
    raw = 'Sure! [{"kind": "deadline", "summary": "Friday", "confidence": 0.8}]'
    out = ft_module._parse_llm_json(raw)
    assert isinstance(out, list) and out[0]["kind"] == "deadline"


def test_parse_llm_json_rejects_garbage(ft_module):
    assert ft_module._parse_llm_json("no json here") is None
    assert ft_module._parse_llm_json(None) is None
