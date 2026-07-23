"""Tests for the Perseus Vault integration plugin.

Covers the PR #13 review findings:
1. Import failure logging must preserve the exception details.
2. The backend must not remain advertised while factory registration
   silently failed.
3. A broken/transitive adapter import must not be masked as optional
   availability.
"""
import builtins
import logging
import sys
import types

import pytest

from praisonai_plugins.integrations.perseus_vault import PerseusVaultPlugin

ADAPTERS_MOD = "praisonaiagents.memory.adapters"
ADAPTER_MOD = "praisonaiagents.memory.adapters.perseus_vault_adapter"


@pytest.fixture
def plugin():
    return PerseusVaultPlugin()


def install_fake_adapters(monkeypatch, register):
    """Inject fake adapter modules so the plugin's lazy imports succeed."""
    adapters = types.ModuleType(ADAPTERS_MOD)
    adapters.register_memory_factory = register
    adapter_mod = types.ModuleType(ADAPTER_MOD)
    adapter_mod.create_perseus_vault_memory_adapter = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, ADAPTERS_MOD, adapters)
    monkeypatch.setitem(sys.modules, ADAPTER_MOD, adapter_mod)
    return adapters, adapter_mod


@pytest.fixture
def fail_import(monkeypatch):
    """Force the import of `module_name` to raise ImportError(missing_name)."""
    def _fail(module_name, missing_name):
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == module_name:
                raise ImportError(f"No module named '{missing_name}'", name=missing_name)
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    return _fail


def test_on_init_registers_factory_when_adapter_available(plugin, monkeypatch, caplog):
    """Happy path: factory is registered under 'perseus_vault' and logged."""
    calls = []
    _, adapter_mod = install_fake_adapters(monkeypatch, lambda name, factory: calls.append((name, factory)))

    with caplog.at_level(logging.INFO):
        plugin.on_init({})

    assert calls == [("perseus_vault", adapter_mod.create_perseus_vault_memory_adapter)]
    assert any("Perseus Vault memory adapter registered" in r.message for r in caplog.records)


def test_missing_optional_adapter_warning_includes_exception_details(plugin, fail_import, caplog):
    """Finding 1: a genuinely-absent optional adapter logs a warning that
    preserves the original ImportError details (the missing module)."""
    fail_import(ADAPTERS_MOD, ADAPTERS_MOD)

    with caplog.at_level(logging.WARNING):
        plugin.on_init({})  # must not raise

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a warning when the optional adapter is absent"
    assert any(ADAPTERS_MOD in r.message for r in warnings), (
        "warning must include the ImportError details so the real cause is visible"
    )


def test_broken_transitive_import_is_not_masked_as_optional(plugin, monkeypatch, fail_import, caplog):
    """Finding 3: the adapter module exists but fails mid-import because a
    transitive dependency is broken. This is a real setup error — it must be
    logged as such (with details), not masked as 'adapter not available'."""
    registered = []
    install_fake_adapters(monkeypatch, lambda name, factory: registered.append(name))
    # Simulate the adapter module raising ImportError for a missing transitive
    # dependency partway through its own import.
    fail_import(ADAPTER_MOD, "perseus_vault_sdk")

    with caplog.at_level(logging.WARNING):
        plugin.on_init({})  # must not raise

    assert registered == [], "registration must not be attempted when the adapter import broke"
    assert not any("not available" in r.message for r in caplog.records), (
        "a broken transitive import must not be reported as optional availability"
    )
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors and any("perseus_vault_sdk" in r.message for r in errors), (
        "expected an ERROR log preserving the real cause (missing transitive module)"
    )


def test_registration_failure_propagates_instead_of_masked_warning(plugin, monkeypatch, caplog):
    """Finding 2: if the imports succeed but register_memory_factory itself
    fails (even with ImportError), the failure must surface — the backend must
    not stay advertised while its factory was never registered."""
    def boom(name, factory):
        raise ImportError("memory registry backend unavailable")

    install_fake_adapters(monkeypatch, boom)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ImportError, match="memory registry backend unavailable"):
            plugin.on_init({})

    assert not any("not available" in r.message for r in caplog.records), (
        "a registration failure must not be reported as optional availability"
    )
    assert not any("registered" in r.message for r in caplog.records), (
        "success must not be logged when registration did not happen"
    )
