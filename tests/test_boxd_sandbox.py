"""Unit tests for boxd sandbox plugin backend."""

from __future__ import annotations

import asyncio
import importlib.machinery
import logging
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from praisonaiagents.sandbox import ResourceLimits, SandboxConfig, SandboxStatus

from praisonai_plugins.sandbox.boxd import BoxdSandbox


@pytest.fixture(autouse=True)
def _no_boxd_env(monkeypatch):
    """Keep a developer's own boxd settings out of the tests."""
    for name in ("BOXD_API_KEY", "BOXD_ORG", "BOXD_REGION"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client(monkeypatch):
    """A fake boxd SDK: ``boxd.Boxd(...)`` returns this client."""
    client = MagicMock()
    client.machines.create.return_value = SimpleNamespace(id="m-123")
    client.machines.exec.return_value = SimpleNamespace(stdout="boxd-ok\n", stderr="", exit_code=0)
    sdk = types.ModuleType("boxd")
    sdk.__spec__ = importlib.machinery.ModuleSpec("boxd", None)
    sdk.Boxd = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "boxd", sdk)
    return client


def _sandbox(**kwargs):
    return BoxdSandbox(api_key="test-key", **kwargs)


class TestAvailability:
    def test_not_available_without_key(self, client):
        assert BoxdSandbox().is_available is False

    def test_available_with_key_and_sdk(self, client, monkeypatch):
        monkeypatch.setenv("BOXD_API_KEY", "test-key")
        assert BoxdSandbox().is_available is True

    def test_not_available_without_sdk(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "boxd", None)
        assert _sandbox().is_available is False

    async def test_start_says_what_is_missing(self, client):
        with pytest.raises(RuntimeError, match="BOXD_API_KEY"):
            await BoxdSandbox().start()
        client.machines.create.assert_not_called()

    def test_entry_point_registered(self):
        from importlib.metadata import entry_points

        found = [ep for ep in entry_points(group="praisonai.sandbox") if ep.name == "boxd"]
        assert found and found[0].value == "praisonai_plugins.sandbox.boxd:BoxdSandbox"


class TestStartStop:
    async def test_start_creates_an_isolated_machine_and_waits(self, client, monkeypatch):
        monkeypatch.setenv("BOXD_ORG", "acme")
        sandbox = _sandbox()
        await sandbox.start()

        assert sandbox.machine_id == "m-123"
        sys.modules["boxd"].Boxd.assert_called_once_with(api_key="test-key")
        assert client.machines.create.call_args.args[0].startswith("praisonai-")
        assert client.machines.create.call_args.kwargs == {
            "org": "acme",
            "isolated": True,
            "delete_after": 86400,
        }
        client.machines.wait_until_ready.assert_called_once_with("m-123")

    async def test_start_twice_creates_one_machine(self, client):
        sandbox = _sandbox()
        await sandbox.start()
        await sandbox.start()
        assert client.machines.create.call_count == 1

    async def test_concurrent_starts_create_one_machine(self, client):
        client.machines.create.side_effect = lambda *a, **k: time.sleep(0.05) or SimpleNamespace(id="m-123")
        sandbox = _sandbox()
        await asyncio.gather(sandbox.start(), sandbox.start(), sandbox.start())
        assert client.machines.create.call_count == 1

    async def test_start_deletes_the_machine_when_it_does_not_become_ready(self, client):
        client.machines.wait_until_ready.side_effect = TimeoutError("not ready")
        sandbox = _sandbox()
        with pytest.raises(TimeoutError):
            await sandbox.start()
        client.machines.delete.assert_called_once_with("m-123")
        assert sandbox.machine_id is None

    async def test_stop_deletes_the_machine(self, client):
        sandbox = _sandbox()
        await sandbox.start()
        await sandbox.stop()
        await sandbox.cleanup()
        client.machines.delete.assert_called_once_with("m-123")
        assert sandbox.machine_id is None

    async def test_stop_without_start_does_nothing(self, client):
        await _sandbox().stop()
        client.machines.delete.assert_not_called()

    async def test_stop_keeps_the_machine_id_when_delete_fails(self, client):
        sandbox = _sandbox()
        await sandbox.start()
        client.machines.delete.side_effect = [ConnectionError("unavailable"), None]
        with pytest.raises(ConnectionError):
            await sandbox.stop()
        assert sandbox.machine_id == "m-123"
        await sandbox.stop()
        assert sandbox.machine_id is None

    async def test_stop_deletes_the_machine_when_the_interpreter_is_shutting_down(self, client, monkeypatch):
        sandbox = _sandbox()
        await sandbox.start()

        async def no_threads(*args, **kwargs):
            raise RuntimeError("cannot schedule new futures after interpreter shutdown")

        monkeypatch.setattr(asyncio, "to_thread", no_threads)
        await sandbox.stop()
        client.machines.delete.assert_called_once_with("m-123")
        assert sandbox.machine_id is None

    async def test_stop_during_start_waits_and_then_deletes(self, client):
        creating = threading.Event()
        release = threading.Event()

        def slow_create(*args, **kwargs):
            creating.set()
            release.wait(5)
            return SimpleNamespace(id="m-123")

        client.machines.create.side_effect = slow_create
        sandbox = _sandbox()
        start = asyncio.create_task(sandbox.start())
        await asyncio.to_thread(creating.wait, 5)
        stop = asyncio.create_task(sandbox.stop())
        await asyncio.sleep(0.05)
        assert not stop.done()
        release.set()
        await asyncio.gather(start, stop)
        client.machines.delete.assert_called_once_with("m-123")
        assert sandbox.machine_id is None


class TestExecute:
    async def test_execute_python(self, client):
        result = await _sandbox().execute("print('boxd-ok')")

        assert result.status == SandboxStatus.COMPLETED
        assert result.exit_code == 0
        assert result.stdout == "boxd-ok\n"
        assert result.duration_seconds >= 0
        args = client.machines.exec.call_args.args
        assert args[0] == "m-123"
        assert args[1] == ["bash", "-c", "python3 -c 'print('\"'\"'boxd-ok'\"'\"')'"]

    async def test_execute_bash(self, client):
        await _sandbox().execute("echo $HOME | wc -c", language="Bash")
        assert client.machines.exec.call_args.args[1] == ["bash", "-c", "echo $HOME | wc -c"]

    async def test_execute_fails_for_another_language_without_a_machine(self, client):
        result = await _sandbox().execute("puts 1", language="ruby")
        assert result.status == SandboxStatus.FAILED
        assert "ruby" in result.error
        client.machines.create.assert_not_called()

    async def test_execute_starts_the_machine_once(self, client):
        sandbox = _sandbox()
        await sandbox.execute("print(1)")
        await sandbox.execute("print(2)")
        assert client.machines.create.call_count == 1
        assert client.machines.exec.call_count == 2


class TestRunCommand:
    async def test_passes_env_cwd_and_timeout(self, client):
        sandbox = _sandbox(config=SandboxConfig(sandbox_type="boxd", env={"A": "1", "B": "1"}))
        await sandbox.run_command(
            "echo $A",
            env={"B": "2"},
            working_dir="/tmp",
            limits=ResourceLimits(timeout_seconds=7),
        )
        kwargs = client.machines.exec.call_args.kwargs
        assert kwargs == {"env": {"A": "1", "B": "2"}, "cwd": "/tmp", "timeout": 7}

    async def test_uses_the_config_timeout_and_sends_no_empty_env(self, client):
        await _sandbox().run_command("true")
        kwargs = client.machines.exec.call_args.kwargs
        assert kwargs == {"env": None, "cwd": None, "timeout": 60}

    async def test_list_command_is_quoted(self, client):
        await _sandbox().run_command(["echo", "a b"])
        assert client.machines.exec.call_args.args[1] == ["bash", "-c", "echo 'a b'"]

    async def test_nonzero_exit_is_failed(self, client):
        client.machines.exec.return_value = SimpleNamespace(stdout="", stderr="boom\n", exit_code=3)
        result = await _sandbox().run_command("exit 3")
        assert result.status == SandboxStatus.FAILED
        assert result.exit_code == 3
        assert result.stderr == "boom\n"

    @pytest.mark.parametrize("error", [TimeoutError("deadline"), RuntimeError("exec timed out after 1s")])
    async def test_timeout_is_reported_as_timeout(self, client, error):
        client.machines.exec.side_effect = error
        result = await _sandbox().run_command("sleep 5", limits=ResourceLimits(timeout_seconds=1))
        assert result.status == SandboxStatus.TIMEOUT
        assert result.error == str(error)

    async def test_other_errors_are_failed(self, client):
        client.machines.exec.side_effect = RuntimeError("connection lost")
        result = await _sandbox().run_command("true")
        assert result.status == SandboxStatus.FAILED
        assert result.error == "connection lost"

    async def test_command_after_stop_starts_a_new_machine(self, client):
        sandbox = _sandbox()
        await sandbox.run_command("true")
        await sandbox.stop()
        client.machines.create.return_value = SimpleNamespace(id="m-456")
        await sandbox.run_command("true")
        assert client.machines.exec.call_args.args[0] == "m-456"

    def test_works_from_separate_event_loops(self, client):
        sandbox = _sandbox()
        first = asyncio.run(sandbox.run_command("echo 1"))
        second = asyncio.run(sandbox.run_command("echo 2"))
        assert first.status == second.status == SandboxStatus.COMPLETED
        assert client.machines.create.call_count == 1


class TestNetworkWarning:
    async def test_warns_once_when_network_is_disabled(self, client, caplog):
        sandbox = _sandbox()
        with caplog.at_level(logging.WARNING, logger="praisonai_plugins.sandbox.boxd"):
            await sandbox.execute("print(1)")
            await sandbox.run_command("true", limits=ResourceLimits(network_enabled=False))
        warnings = [r.message for r in caplog.records if "network_enabled" in r.message]
        assert warnings == ["network_enabled is False, but the boxd machine has outbound internet access"]

    async def test_no_warning_when_network_is_enabled(self, client, caplog):
        limits = ResourceLimits(network_enabled=True)
        sandbox = _sandbox(config=SandboxConfig(sandbox_type="boxd", resource_limits=limits))
        with caplog.at_level(logging.WARNING, logger="praisonai_plugins.sandbox.boxd"):
            await sandbox.run_command("true")
        assert "network_enabled" not in caplog.text

    async def test_warns_for_a_call_that_disables_the_network(self, client, caplog):
        limits = ResourceLimits(network_enabled=True)
        sandbox = _sandbox(config=SandboxConfig(sandbox_type="boxd", resource_limits=limits))
        with caplog.at_level(logging.WARNING, logger="praisonai_plugins.sandbox.boxd"):
            await sandbox.run_command("true")
            await sandbox.run_command("true", limits=ResourceLimits())
        assert caplog.text.count("network_enabled is False") == 1
