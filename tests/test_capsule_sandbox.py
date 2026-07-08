"""Unit tests for Capsule sandbox plugin backend."""

import asyncio
import pytest
from unittest.mock import Mock, patch

from praisonai_plugins.sandbox.capsule import CapsuleSandbox
from praisonaiagents.sandbox import SandboxConfig, SandboxStatus


def _capsule_config():
    factory = getattr(SandboxConfig, "capsule", None)
    if factory is not None:
        return factory()
    return SandboxConfig(sandbox_type="capsule")


class TestCapsuleSandbox:
    def test_init_default(self):
        sandbox = CapsuleSandbox()
        assert sandbox.sandbox_type == "capsule"
        assert not sandbox._is_running

    def test_init_with_config(self):
        config = _capsule_config()
        sandbox = CapsuleSandbox(config=config)
        assert sandbox.config == config

    def test_config_capsule_factory(self):
        config = _capsule_config()
        assert config.sandbox_type == "capsule"
        if hasattr(SandboxConfig, "capsule"):
            assert config.security_policy.allow_network is False
            assert config.security_policy.allow_subprocess is False

    @patch.dict("sys.modules", {"capsule": Mock()})
    def test_is_available_true(self):
        sandbox = CapsuleSandbox()
        assert sandbox.is_available is True

    def test_is_available_false(self):
        sandbox = CapsuleSandbox()
        with patch.dict("sys.modules", {"capsule": None}):
            assert sandbox.is_available is False

    async def test_start_not_available(self):
        sandbox = CapsuleSandbox()
        with patch.object(CapsuleSandbox, "is_available", property(lambda self: False)):
            with pytest.raises(RuntimeError, match="Capsule backend not available"):
                await sandbox.start()

    async def test_start_already_running(self):
        sandbox = CapsuleSandbox()
        sandbox._is_running = True
        await sandbox.start()

    async def test_execute_python_success(self):
        sandbox = CapsuleSandbox()
        sandbox._is_running = True
        sandbox._sandbox = Mock()
        sandbox._sandbox.run.return_value = Mock(stdout="720", stderr="", exit_code=0)
        result = await sandbox.execute("factorial(6)", language="python")
        assert result.status == SandboxStatus.COMPLETED
        assert result.stdout == "720"

    async def test_execute_non_python_rejected(self):
        sandbox = CapsuleSandbox()
        sandbox._is_running = True
        result = await sandbox.execute("echo hi", language="bash")
        assert result.status == SandboxStatus.FAILED
        assert "only supports Python" in result.error

    async def test_execute_timeout_enforced(self):
        sandbox = CapsuleSandbox(timeout=1)
        sandbox._is_running = True
        sandbox._sandbox = Mock()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await sandbox.execute("while True: pass", language="python")
        assert result.status == SandboxStatus.TIMEOUT

    async def test_reset(self):
        sandbox = CapsuleSandbox()
        with patch.object(sandbox, "stop") as mock_stop:
            with patch.object(sandbox, "start") as mock_start:
                await sandbox.reset()
        mock_stop.assert_called_once()
        mock_start.assert_called_once()
