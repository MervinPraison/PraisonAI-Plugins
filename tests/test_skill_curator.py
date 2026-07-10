"""Self-contained tests for the SkillCuratorPlugin.

These tests use a lightweight fake ``SkillManager``/skill so they exercise the
curator's ageing, pinning, dry-run, consolidation, lifecycle (start/stop) and
configuration behaviour without any real skills on disk. The plugin imports the
real ``praisonaiagents`` plugin base (available in the test env), so no SDK
stubbing is required here.
"""

import os
import sys
import time
import importlib

import pytest


@pytest.fixture(scope="module")
def curator_module():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module("praisonai_plugins.skills.curator")


class FakeProps:
    def __init__(self, name, agent_created=False, idle=None, use_count=0):
        self.name = name
        self.agent_created = agent_created
        self.use_count = use_count
        self._idle = idle

    @property
    def idle_days(self):
        return self._idle


class FakeSkill:
    def __init__(self, name, agent_created=False, idle=None, use_count=0):
        self.properties = FakeProps(name, agent_created, idle, use_count)


class FakeManager:
    def __init__(self, skills):
        self._list = list(skills)
        self.archived = []

    @property
    def skills(self):
        return list(self._list)

    def archive_skill(self, name):
        self.archived.append(name)
        self._list = [s for s in self._list if s.properties.name != name]
        return {"success": True, "skill": name, "archive_path": f"/tmp/{name}"}


def test_info_declares_gateway_hooks(curator_module):
    from praisonaiagents.plugins.plugin import PluginHook

    plugin = curator_module.SkillCuratorPlugin()
    info = plugin.info
    assert info.name == "skill_curator"
    assert PluginHook.GATEWAY_START in info.hooks
    assert PluginHook.GATEWAY_STOP in info.hooks


def test_config_parsing(curator_module):
    plugin = curator_module.SkillCuratorPlugin()
    plugin.on_config({
        "interval_hours": 24,
        "stale_after_days": 10,
        "consolidate": True,
        "min_use_count": 5,
        "dry_run": True,
    })
    assert plugin._interval_seconds == 24 * 3600.0
    assert plugin._stale_after_days == 10.0
    assert plugin._consolidate is True
    assert plugin._min_use_count == 5
    assert plugin._dry_run is True


def test_config_interval_floor(curator_module):
    plugin = curator_module.SkillCuratorPlugin()
    plugin.on_config({"interval_hours": 0})
    assert plugin._interval_seconds >= 1.0


def test_sweep_archives_only_stale_agent_created(curator_module):
    manager = FakeManager([
        FakeSkill("stale_a", agent_created=True, idle=45),
        FakeSkill("fresh_b", agent_created=True, idle=5),
        FakeSkill("user_c", agent_created=False, idle=999),
        FakeSkill("never_used", agent_created=True, idle=None),
    ])
    plugin = curator_module.SkillCuratorPlugin(manager=manager)
    plugin.on_config({"stale_after_days": 30})
    result = plugin.sweep()

    assert manager.archived == ["stale_a"]
    assert result["archived"] == ["stale_a"]
    assert result["scanned"] == 4


def test_user_authored_skills_never_touched(curator_module):
    manager = FakeManager([
        FakeSkill("user_old", agent_created=False, idle=10000),
    ])
    plugin = curator_module.SkillCuratorPlugin(manager=manager)
    plugin.on_config({"stale_after_days": 1})
    plugin.sweep()
    assert manager.archived == []


def test_min_use_count_pins_from_ageing(curator_module):
    manager = FakeManager([
        FakeSkill("hot", agent_created=True, idle=90, use_count=50),
        FakeSkill("cold", agent_created=True, idle=90, use_count=1),
    ])
    plugin = curator_module.SkillCuratorPlugin(manager=manager)
    plugin.on_config({"stale_after_days": 30, "min_use_count": 10})
    result = plugin.sweep()
    assert manager.archived == ["cold"]
    assert "hot" in result["skipped_pinned"]


def test_dry_run_does_not_archive(curator_module):
    manager = FakeManager([
        FakeSkill("stale_a", agent_created=True, idle=45),
    ])
    plugin = curator_module.SkillCuratorPlugin(manager=manager)
    plugin.on_config({"stale_after_days": 30, "dry_run": True})
    result = plugin.sweep()
    assert manager.archived == []
    assert result["archived"] == ["stale_a"]


def test_consolidation_is_non_mutating(curator_module):
    manager = FakeManager([
        FakeSkill("report_pdf", agent_created=True, idle=1),
        FakeSkill("report_csv", agent_created=True, idle=1),
    ])
    plugin = curator_module.SkillCuratorPlugin(manager=manager)
    plugin.on_config({"stale_after_days": 30, "consolidate": True})
    plugin.sweep()
    assert manager.archived == []


def test_lifecycle_start_stop_is_daemon(curator_module):
    plugin = curator_module.SkillCuratorPlugin(manager=FakeManager([]))
    plugin.on_config({"interval_hours": 1})
    plugin.gateway_start(None)
    assert plugin._thread is not None
    assert plugin._thread.daemon is True
    assert plugin._thread.is_alive()
    start = time.time()
    plugin.gateway_stop(None)
    assert time.time() - start < 2.0
    assert plugin._thread is None


def test_double_start_is_idempotent(curator_module):
    plugin = curator_module.SkillCuratorPlugin(manager=FakeManager([]))
    plugin.on_config({"interval_hours": 1})
    plugin.gateway_start(None)
    first = plugin._thread
    plugin.gateway_start(None)
    assert plugin._thread is first
    plugin.gateway_stop(None)


def test_sweep_with_no_manager_is_safe(curator_module):
    plugin = curator_module.SkillCuratorPlugin(manager=None)
    result = plugin.sweep(manager=None)
    assert result["archived"] == []
    assert result["scanned"] == 0
