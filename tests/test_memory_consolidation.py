"""Self-contained tests for the MemoryConsolidationPlugin.

These tests stub the small ``praisonaiagents`` plugin/logging surface the plugin
imports, so they run without the full SDK installed. They verify the loss-guard,
deterministic merge/promote/prune behaviour, dry-run, the fake memory store
mutations, and the scheduled lifecycle (start/stop) being a non-blocking daemon.
"""

import sys
import types
import time
import logging
import importlib

import pytest


def _install_sdk_stubs():
    """Install minimal praisonaiagents stubs required by the plugin import."""
    if "praisonaiagents.plugins.plugin" in sys.modules:
        return

    root = sys.modules.get("praisonaiagents") or types.ModuleType("praisonaiagents")

    logging_mod = types.ModuleType("praisonaiagents._logging")
    logging_mod.get_logger = lambda name: logging.getLogger(name)

    plugins_pkg = types.ModuleType("praisonaiagents.plugins")
    plugin_mod = types.ModuleType("praisonaiagents.plugins.plugin")

    from dataclasses import dataclass, field
    from enum import Enum
    from typing import List

    class PluginHook(str, Enum):
        GATEWAY_START = "gateway_start"
        GATEWAY_STOP = "gateway_stop"

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
def mod():
    _install_sdk_stubs()
    import os

    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module("praisonai_plugins.memory.consolidator")


class FakeMemory:
    """Minimal in-memory store matching the consolidation surface used."""

    def __init__(self, entries):
        # entries: list of dicts with id/text/metadata
        self._entries = {e["id"]: dict(e) for e in entries}

    def get_all_memories(self, **kwargs):
        return [dict(e) for e in self._entries.values()]

    def delete_memories(self, ids):
        n = 0
        for i in ids:
            if i in self._entries:
                del self._entries[i]
                n += 1
        return n

    def update_memory(self, memory_id, metadata=None, **kwargs):
        if memory_id in self._entries:
            self._entries[memory_id].setdefault("metadata", {})
            if metadata:
                self._entries[memory_id]["metadata"] = metadata
            return True
        return False


def _entry(i, text, importance=0.0):
    return {"id": i, "text": text, "metadata": {"importance": importance}}


def test_info_declares_gateway_hooks(mod):
    from praisonaiagents.plugins.plugin import PluginHook

    plugin = mod.MemoryConsolidationPlugin()
    info = plugin.info
    assert info.name == "memory_consolidation"
    assert PluginHook.GATEWAY_START in info.hooks
    assert PluginHook.GATEWAY_STOP in info.hooks


def test_config_parsing(mod):
    plugin = mod.MemoryConsolidationPlugin()
    plugin.on_config(
        {
            "enabled": True,
            "interval_hours": 24,
            "max_loss_fraction": 0.5,
            "similarity_threshold": 0.9,
            "promote_importance": 0.8,
            "curated_tag": "gold",
            "use_llm": False,
            "dry_run": True,
        }
    )
    assert plugin._enabled is True
    assert plugin._interval_hours == 24.0
    assert plugin._max_loss_fraction == 0.5
    assert plugin._similarity_threshold == 0.9
    assert plugin._promote_importance == 0.8
    assert plugin._curated_tag == "gold"
    assert plugin._dry_run is True


def test_max_loss_fraction_clamped(mod):
    plugin = mod.MemoryConsolidationPlugin()
    plugin.on_config({"max_loss_fraction": 5.0})
    assert plugin._max_loss_fraction == 1.0
    plugin.on_config({"max_loss_fraction": -1.0})
    assert plugin._max_loss_fraction == 0.0


def test_empty_store_is_safe(mod):
    plugin = mod.MemoryConsolidationPlugin()
    result = plugin.consolidate(FakeMemory([]))
    assert result.entries_before == 0
    assert result.entries_after == 0
    assert result.rejected is False


def test_merges_near_duplicates_and_prunes(mod):
    mem = FakeMemory(
        [
            _entry("a", "the sky is blue today", importance=0.9),
            _entry("b", "the sky is blue today too", importance=0.1),
            _entry("c", "cats are wonderful pets", importance=0.2),
        ]
    )
    plugin = mod.MemoryConsolidationPlugin()
    plugin.on_config({"similarity_threshold": 0.5, "promote_importance": 0.7})
    result = plugin.consolidate(mem, max_loss_fraction=0.5)

    assert result.rejected is False
    assert result.merged == 1
    assert result.pruned == 1
    # keeper is highest-importance member of the duplicate cluster
    remaining_ids = {e["id"] for e in mem.get_all_memories()}
    assert "a" in remaining_ids
    assert "b" not in remaining_ids
    assert "c" in remaining_ids


def test_promotes_high_importance_keeper(mod):
    mem = FakeMemory(
        [
            _entry("a", "user prefers dark mode", importance=0.95),
            _entry("b", "user prefers dark mode setting", importance=0.1),
        ]
    )
    plugin = mod.MemoryConsolidationPlugin()
    plugin.on_config(
        {"similarity_threshold": 0.5, "promote_importance": 0.7, "curated_tag": "curated"}
    )
    result = plugin.consolidate(mem, max_loss_fraction=0.75)
    assert result.promoted == 1
    keeper = next(e for e in mem.get_all_memories() if e["id"] == "a")
    assert keeper["metadata"].get("tier") == "curated"


def test_loss_guard_rejects_and_leaves_store_untouched(mod):
    # Two near-dup clusters; each keeps one, prunes one -> loss 0.5.
    mem = FakeMemory(
        [
            _entry("a", "alpha beta gamma delta", importance=0.5),
            _entry("b", "alpha beta gamma delta epsilon", importance=0.1),
            _entry("c", "one two three four five", importance=0.5),
            _entry("d", "one two three four five six", importance=0.1),
        ]
    )
    plugin = mod.MemoryConsolidationPlugin()
    plugin.on_config({"similarity_threshold": 0.5})
    before_ids = {e["id"] for e in mem.get_all_memories()}
    result = plugin.consolidate(mem, max_loss_fraction=0.25)

    assert result.rejected is True
    assert "loss guard" in (result.reason or "")
    # store completely untouched
    after_ids = {e["id"] for e in mem.get_all_memories()}
    assert after_ids == before_ids


def test_dry_run_does_not_mutate(mod):
    mem = FakeMemory(
        [
            _entry("a", "hello world foo bar", importance=0.9),
            _entry("b", "hello world foo bar baz", importance=0.1),
        ]
    )
    plugin = mod.MemoryConsolidationPlugin()
    plugin.on_config({"similarity_threshold": 0.5, "dry_run": True})
    before_ids = {e["id"] for e in mem.get_all_memories()}
    result = plugin.consolidate(mem, max_loss_fraction=0.9)
    assert result.pruned == 1  # reported
    assert result.context.get("dry_run") is True
    assert {e["id"] for e in mem.get_all_memories()} == before_ids  # unchanged


def test_invalid_max_loss_fraction_rejected(mod):
    mem = FakeMemory([_entry("a", "x y z", 0.1), _entry("b", "x y z w", 0.1)])
    plugin = mod.MemoryConsolidationPlugin()
    plugin.on_config({"similarity_threshold": 0.5})
    result = plugin.consolidate(mem, max_loss_fraction=float("nan"))
    assert result.rejected is True


def test_disabled_does_not_start_scheduler(mod):
    plugin = mod.MemoryConsolidationPlugin(memory=FakeMemory([]))
    plugin.on_config({"enabled": False})
    plugin.gateway_start(None)
    assert plugin._thread is None


def test_scheduler_lifecycle_is_daemon_and_nonblocking(mod):
    plugin = mod.MemoryConsolidationPlugin(memory=FakeMemory([]))
    plugin.on_config({"enabled": True, "interval_hours": 1})
    plugin.gateway_start(None)
    assert plugin._thread is not None
    assert plugin._thread.daemon is True
    assert plugin._thread.is_alive()
    start = time.time()
    plugin.gateway_stop(None)
    assert time.time() - start < 2.0
    assert plugin._thread is None


def test_double_start_is_idempotent(mod):
    plugin = mod.MemoryConsolidationPlugin(memory=FakeMemory([]))
    plugin.on_config({"enabled": True, "interval_hours": 1})
    plugin.gateway_start(None)
    first = plugin._thread
    plugin.gateway_start(None)
    assert plugin._thread is first
    plugin.gateway_stop(None)


def test_satisfies_core_protocol_structurally(mod):
    plugin = mod.MemoryConsolidationPlugin()
    assert hasattr(plugin, "consolidate")
    assert callable(plugin.consolidate)
