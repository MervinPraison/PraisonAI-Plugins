"""Self-contained tests for the MemoryWatchdogPlugin.

These tests stub the small ``praisonaiagents`` plugin/logging surface the
plugin imports, so they run without the full SDK installed. They verify the
snapshot formatting, lifecycle (start/stop) behaviour, configuration parsing,
and that the sampler thread is a non-blocking daemon.
"""

import sys
import types
import time
import logging
import importlib

import pytest


def _install_sdk_stubs():
    """Install minimal praisonaiagents stubs required by the plugin import."""
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
def watchdog_module():
    _install_sdk_stubs()
    import os

    src = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "src"
    )
    if src not in sys.path:
        sys.path.insert(0, src)
    mod = importlib.import_module(
        "praisonai_plugins.observability.memory_watchdog"
    )
    return mod


def test_format_snapshot_shape(watchdog_module):
    line = watchdog_module._format_snapshot()
    assert line.startswith("[MEMORY] ")
    assert "rss=" in line
    assert "gc=(gen0=" in line
    assert "objects=" in line


def test_format_snapshot_tag(watchdog_module):
    line = watchdog_module._format_snapshot(tag="final")
    assert "[MEMORY] final " in line


def test_info_declares_gateway_hooks(watchdog_module):
    from praisonaiagents.plugins.plugin import PluginHook

    plugin = watchdog_module.MemoryWatchdogPlugin()
    info = plugin.info
    assert info.name == "memory_watchdog"
    assert PluginHook.GATEWAY_START in info.hooks
    assert PluginHook.GATEWAY_STOP in info.hooks


def test_config_parsing(watchdog_module):
    plugin = watchdog_module.MemoryWatchdogPlugin()
    plugin.on_config({"interval": 7, "log_level": "DEBUG", "use_psutil": False})
    assert plugin._interval == 7.0
    assert plugin._log_level == logging.DEBUG
    assert plugin._use_psutil is False


def test_config_interval_floor(watchdog_module):
    plugin = watchdog_module.MemoryWatchdogPlugin()
    plugin.on_config({"interval": 0})
    assert plugin._interval >= 1.0


def test_lifecycle_start_stop_is_daemon(watchdog_module):
    plugin = watchdog_module.MemoryWatchdogPlugin()
    plugin.on_config({"interval": 1})
    plugin.gateway_start(None)
    assert plugin._thread is not None
    assert plugin._thread.daemon is True
    assert plugin._thread.is_alive()
    # Stop must not block and must clear state.
    start = time.time()
    plugin.gateway_stop(None)
    assert time.time() - start < 2.0
    assert plugin._thread is None


def test_double_start_is_idempotent(watchdog_module):
    plugin = watchdog_module.MemoryWatchdogPlugin()
    plugin.on_config({"interval": 1})
    plugin.gateway_start(None)
    first = plugin._thread
    plugin.gateway_start(None)
    assert plugin._thread is first
    plugin.gateway_stop(None)


def test_get_rss_mb_type(watchdog_module):
    plugin = watchdog_module.MemoryWatchdogPlugin()
    rss = plugin.get_rss_mb()
    assert rss is None or isinstance(rss, float)
