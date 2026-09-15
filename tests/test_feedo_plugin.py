"""Tests for the Feedo integration plugin.

Covers:
1. The memory factory is registered under ``feedo`` on ``on_init``.
2. A missing identity (usage_key/private_key) fails fast with remediation
   guidance before the SDK is imported.
3. A missing ``feedo-sdk`` raises ImportError with install guidance.
4. A missing memory registry degrades gracefully (warning, no raise).
"""

import builtins
import logging
import sys
import types

import pytest

from praisonai_plugins.integrations.feedo import (
    FeedoPlugin,
    create_feedo_memory_adapter,
)

ADAPTERS_MOD = "praisonaiagents.memory.adapters"


@pytest.fixture
def plugin():
    return FeedoPlugin()


def install_fake_adapters(monkeypatch, register):
    """Inject a fake memory.adapters module so on_init's lazy import succeeds."""
    adapters = types.ModuleType(ADAPTERS_MOD)
    adapters.register_memory_factory = register
    monkeypatch.setitem(sys.modules, ADAPTERS_MOD, adapters)
    return adapters


def fail_import(monkeypatch, module_name, missing_name):
    """Force the import of ``module_name`` to raise ImportError(missing_name)."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name:
            raise ImportError(f"No module named '{missing_name}'", name=missing_name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_on_init_registers_factory(plugin, monkeypatch, caplog):
    """Happy path: factory is registered under 'feedo' and logged."""
    calls = []
    install_fake_adapters(monkeypatch, lambda name, factory: calls.append((name, factory)))

    with caplog.at_level(logging.INFO):
        plugin.on_init({})

    assert calls == [("feedo", create_feedo_memory_adapter)]
    assert any("Feedo memory adapter registered" in r.message for r in caplog.records)


def test_on_init_warns_when_registry_unavailable(plugin, monkeypatch, caplog):
    """A missing memory registry must not crash agent startup."""
    fail_import(monkeypatch, ADAPTERS_MOD, ADAPTERS_MOD)

    with caplog.at_level(logging.WARNING):
        plugin.on_init({})  # must not raise

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings
    assert any("Feedo memory adapter not available" in r.message for r in warnings)


def test_on_init_raises_on_broken_transitive_import(plugin, monkeypatch):
    """A broken transitive import must surface, not be masked as 'absent'."""
    fail_import(monkeypatch, ADAPTERS_MOD, "some_broken_dependency")

    with pytest.raises(ImportError, match="some_broken_dependency"):
        plugin.on_init({})


def test_on_init_raises_on_version_mismatch(plugin, monkeypatch):
    """An incompatible praisonaiagents (no register_memory_factory) must surface."""
    fail_import(monkeypatch, ADAPTERS_MOD, "register_memory_factory")

    with pytest.raises(ImportError, match="register_memory_factory"):
        plugin.on_init({})


def test_create_adapter_requires_identity():
    """No usage_key/private_key -> clear ValueError with setup guidance."""
    with pytest.raises(ValueError, match="usage_key"):
        create_feedo_memory_adapter()


def test_create_adapter_import_error_with_guidance(monkeypatch):
    """Missing feedo-sdk -> ImportError with install guidance."""
    fail_import(monkeypatch, "feedo", "feedo")

    with pytest.raises(ImportError, match="pip install feedo-sdk"):
        create_feedo_memory_adapter(usage_key="0x" + "1" * 64)


def test_create_adapter_happy_path(monkeypatch):
    """With feedo-sdk present, the adapter round-trips through FeedoMemory."""
    captured = {}

    class FakeFeedoMemory:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def add_short(self, text, metadata=None):
            return "mem_123"

        def search_short(self, query, limit=5):
            return [{"id": "mem_123", "text": "hi"}]

        def add_long(self, text, metadata=None):
            return "mem_456"

        def search_long(self, query, limit=5):
            return []

        def get_all_memories(self):
            return []

        def clear_short(self):
            pass

        def clear_long(self):
            pass

    feedo_mod = types.ModuleType("feedo")
    feedo_mod.FeedoMemory = FakeFeedoMemory
    monkeypatch.setitem(sys.modules, "feedo", feedo_mod)

    adapter = create_feedo_memory_adapter(
        usage_key="0x" + "1" * 64, config={"user_id": "u1"}
    )

    assert adapter.store_short_term("hello") == "mem_123"
    assert adapter.get_all_memories() == []
    # Top-level usage_key and nested config.user_id must both be forwarded.
    assert captured["usage_key"] == "0x" + "1" * 64
    assert captured["user_id"] == "u1"
