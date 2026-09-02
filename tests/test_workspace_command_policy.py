"""Tests for the WorkspaceCommandPolicy plugin.

These use the installed ``praisonaiagents`` SDK (the same dependency the repo
already requires) and drive the policy with a temporary workspace root so the
containment logic is exercised against real ``path_safety`` resolution.
"""
import os

import pytest

from praisonaiagents.plugins.plugin import PluginHook, PluginDecision
from praisonai_plugins.policies.workspace_command_policy import (
    WorkspaceCommandPolicy,
)


@pytest.fixture
def policy(tmp_path):
    p = WorkspaceCommandPolicy()
    p.on_config({"workspace_command_policy": {"workspace_root": str(tmp_path)}})
    return p


def test_info_declares_expected_hooks():
    info = WorkspaceCommandPolicy().info
    assert info.name == "workspace_command_policy"
    assert PluginHook.BEFORE_TOOL in info.hooks
    assert PluginHook.ON_PERMISSION_ASK in info.hooks


def test_allows_command_with_no_write_targets(policy):
    result = policy.before_tool("execute_command", {"command": "ls -la"})
    assert result == {"command": "ls -la"}


def test_allows_in_workspace_write(policy, tmp_path):
    cmd = f"rm {tmp_path}/inside.txt"
    result = policy.before_tool("execute_command", {"command": cmd})
    assert result == {"command": cmd}


def test_denies_escaping_rm(policy):
    result = policy.before_tool(
        "execute_command", {"command": "rm ../../thing"}
    )
    assert isinstance(result, PluginDecision)
    assert result.is_denied()
    assert "OUTSIDE" in result.reason


def test_denies_escaping_cp_to_etc(policy):
    result = policy.before_tool(
        "execute_command", {"command": "cp secret /etc/x"}
    )
    assert isinstance(result, PluginDecision)
    assert result.is_denied()


def test_denies_redirection_outside(policy):
    result = policy.before_tool(
        "execute_command", {"command": "echo hi > /etc/passwd"}
    )
    assert isinstance(result, PluginDecision)
    assert result.is_denied()


def test_denies_glued_redirection_outside(policy):
    result = policy.before_tool(
        "execute_command", {"command": "echo hi >/tmp/../etc/evil"}
    )
    assert isinstance(result, PluginDecision)
    assert result.is_denied()


def test_denies_sed_inplace_outside(policy):
    result = policy.before_tool(
        "execute_command", {"command": "sed -i s/a/b/ ../outside.txt"}
    )
    assert isinstance(result, PluginDecision)
    assert result.is_denied()


def test_denies_find_delete_escaping(policy):
    result = policy.before_tool(
        "execute_command", {"command": "find .. -delete"}
    )
    assert isinstance(result, PluginDecision)
    assert result.is_denied()


def test_ignores_non_command_tools(policy):
    args = {"path": "/etc/passwd"}
    assert policy.before_tool("read_file", args) is args


def test_disabled_is_noop(tmp_path):
    p = WorkspaceCommandPolicy()
    p.on_config(
        {"workspace_command_policy": {"enabled": False, "workspace_root": str(tmp_path)}}
    )
    args = {"command": "rm ../../thing"}
    assert p.before_tool("execute_command", args) is args


def test_parse_error_fails_closed(policy):
    # Unbalanced quote should fail closed (deny) by default.
    result = policy.before_tool(
        "execute_command", {"command": "rm 'unterminated"}
    )
    assert isinstance(result, PluginDecision)
    assert result.is_denied()


def test_parse_error_allow_config(tmp_path):
    p = WorkspaceCommandPolicy()
    p.on_config(
        {
            "workspace_command_policy": {
                "workspace_root": str(tmp_path),
                "on_parse_error": "allow",
            }
        }
    )
    args = {"command": "rm 'unterminated"}
    assert p.before_tool("execute_command", args) is args


def test_on_permission_ask_denies_escape(policy):
    assert policy.on_permission_ask("rm ../../thing", "test") is False


def test_on_permission_ask_passthrough_for_safe(policy, tmp_path):
    cmd = f"rm {tmp_path}/inside.txt"
    assert policy.on_permission_ask(cmd, "test") is None


def test_extra_mutating_binaries(tmp_path):
    p = WorkspaceCommandPolicy()
    p.on_config(
        {
            "workspace_command_policy": {
                "workspace_root": str(tmp_path),
                "extra_mutating_binaries": ["myrm"],
            }
        }
    )
    result = p.before_tool("execute_command", {"command": "myrm ../outside"})
    assert isinstance(result, PluginDecision)
    assert result.is_denied()


def test_argv_list_command(policy):
    result = policy.before_tool(
        "execute_command", {"command": ["rm", "../../thing"]}
    )
    assert isinstance(result, PluginDecision)
    assert result.is_denied()


def test_compound_command_denies_on_escape(policy, tmp_path):
    cmd = f"cd {tmp_path} && rm ../../escape"
    result = policy.before_tool("execute_command", {"command": cmd})
    assert isinstance(result, PluginDecision)
    assert result.is_denied()
