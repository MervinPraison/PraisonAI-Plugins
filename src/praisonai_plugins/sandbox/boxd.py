"""boxd sandbox backend for PraisonAI. Requires: pip install praisonai-plugins[boxd]"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import logging
import os
import shlex
import threading
import time

from praisonaiagents.sandbox import SandboxConfig, SandboxResult, SandboxStatus

logger = logging.getLogger(__name__)


class BoxdSandbox:
    """Runs code in a boxd machine (an Ubuntu VM). start() creates it, stop() deletes it."""

    def __init__(self, config: SandboxConfig | None = None, api_key: str | None = None):
        self.config = config or SandboxConfig(sandbox_type="boxd")
        self._api_key = api_key or os.environ.get("BOXD_API_KEY", "")
        self.machine_id: str | None = None
        self._warned = False
        self._lock = threading.Lock()  # start() and stop() may run on different threads

    @property
    def is_available(self) -> bool:
        return bool(self._api_key.strip()) and importlib.util.find_spec("boxd") is not None

    def _start(self) -> None:
        with self._lock:
            if self.machine_id is not None:
                return
            if not self.is_available:
                raise RuntimeError("boxd needs BOXD_API_KEY and: pip install praisonai-plugins[boxd]")
            from boxd import Boxd
            self._client = Boxd(api_key=self._api_key)
            machine = self._client.machines.create(
                f"praisonai-{os.urandom(4).hex()}", org=os.environ.get("BOXD_ORG"), isolated=True,
                delete_after=86400,  # boxd deletes the machine after a day if stop() never runs
            )
            try:
                self._client.machines.wait_until_ready(machine.id)
            except BaseException:
                self._client.machines.delete(machine.id)
                raise
            self.machine_id = machine.id

    def _stop(self) -> None:
        with self._lock:
            if self.machine_id is not None:
                self._client.machines.delete(self.machine_id)
                self.machine_id = None

    async def start(self) -> None:
        await asyncio.to_thread(self._start)

    async def stop(self) -> None:
        try:
            await asyncio.to_thread(self._stop)
        except RuntimeError:  # no worker threads are left when the interpreter shuts down
            self._stop()

    cleanup = stop  # SandboxManager calls stop() and then cleanup()

    async def execute(self, code, language="python", limits=None, env=None, working_dir=None) -> SandboxResult:
        if language.lower() in ("python", "python3"):
            code = f"python3 -c {shlex.quote(code)}"
        elif language.lower() not in ("bash", "sh", "shell"):
            return SandboxResult(status=SandboxStatus.FAILED, error=f"boxd runs python or bash, not {language!r}")
        return await self.run_command(code, limits, env, working_dir)

    async def run_command(self, command, limits=None, env=None, working_dir=None) -> SandboxResult:
        limits = limits or self.config.resource_limits
        if not limits.network_enabled and not self._warned:
            self._warned = True
            logger.warning("network_enabled is False, but the boxd machine has outbound internet access")
        argv = ["bash", "-c", command if isinstance(command, str) else shlex.join(command)]
        env = {**self.config.env, **(env or {})} or None
        await self.start()
        started_at = time.time()
        try:
            run = functools.partial(self._client.machines.exec, self.machine_id, argv, env=env, cwd=working_dir)
            done = await asyncio.to_thread(run, timeout=limits.timeout_seconds)
            result = SandboxResult(exit_code=done.exit_code, stdout=done.stdout or "", stderr=done.stderr or "")
            result.status = SandboxStatus.COMPLETED if done.exit_code == 0 else SandboxStatus.FAILED
        except Exception as exc:  # noqa: BLE001
            result = SandboxResult(status=SandboxStatus.FAILED, error=str(exc))
            if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
                result.status = SandboxStatus.TIMEOUT
        result.duration_seconds = time.time() - started_at
        return result
