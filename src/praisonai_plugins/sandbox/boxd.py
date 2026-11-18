import asyncio
import importlib.util
import os
import shlex
import threading
import time

from praisonaiagents.sandbox import SandboxConfig, SandboxResult, SandboxStatus


class BoxdSandbox:
    """Runs code in a boxd machine (an Ubuntu VM). start() creates it, stop() deletes it. Needs praisonai-plugins[boxd]."""

    def __init__(self, config: SandboxConfig | None = None, api_key: str | None = None):
        self.config = config or SandboxConfig(sandbox_type="boxd")
        self._api_key = api_key or os.environ.get("BOXD_API_KEY", "")
        self.machine_id, self._lock, self._users, self._used = None, threading.Lock(), 0, False  # see start()

    @property
    def is_available(self) -> bool:
        return bool(self._api_key.strip()) and importlib.util.find_spec("boxd") is not None

    def _start(self) -> bool:
        with self._lock:
            if self.machine_id is not None:
                return False
            if not self.is_available:
                raise RuntimeError("boxd needs BOXD_API_KEY and: pip install praisonai-plugins[boxd]")
            self._client = importlib.import_module("boxd").Boxd(api_key=self._api_key)
            machine = self._client.machines.create(
                f"praisonai-{os.urandom(4).hex()}", org=os.environ.get("BOXD_ORG"),
                spec={"isolated": True, "egress": [] if self.config.resource_limits.network_enabled else ["none"]},
                delete_after=86400)  # boxd deletes the machine after a day if stop() never runs
            try:
                self._client.machines.wait_until_ready(machine.id)
            except BaseException:
                self._client.machines.delete(machine.id)
                raise
            self.machine_id, self._used = machine.id, False
            return True

    def _stop(self, leaving: bool = False) -> None:  # leaving: delete only a machine no caller has or waits for
        with self._lock:
            if self.machine_id is not None and not (leaving and (self._users or self._used)):
                self._client.machines.delete(self.machine_id)
                self.machine_id, self._used = None, False

    async def start(self) -> None:
        self._users += 1  # callers still in start(); _used: a start() returned this machine to a caller
        task = asyncio.ensure_future(asyncio.to_thread(self._start))
        try:
            await asyncio.shield(task)
        except BaseException:  # failed or cancelled: leave in a task of its own, so a second cancel can't skip it
            await asyncio.shield(asyncio.ensure_future(self._leave(task)))
            raise
        self._users, self._used = self._users - 1, True

    async def _leave(self, task) -> None:  # the last caller out deletes a machine no caller has
        await asyncio.wait({task})  # the thread may still be creating the machine
        self._users -= 1
        await asyncio.to_thread(self._stop, True)

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
        if not limits.network_enabled and self.config.resource_limits.network_enabled:
            return SandboxResult(status=SandboxStatus.FAILED, error="this sandbox was created with network access")
        argv = ["bash", "-c", command if isinstance(command, str) else shlex.join(command)]
        env = {**self.config.env, **(env or {})} or None
        await self.start()
        started_at = time.time()
        try:
            done = await asyncio.to_thread(
                self._client.machines.exec, self.machine_id, argv, env=env, cwd=working_dir, timeout=limits.timeout_seconds)
            result = SandboxResult(exit_code=done.exit_code, stdout=done.stdout or "", stderr=done.stderr or "")
            result.status = SandboxStatus.COMPLETED if done.exit_code == 0 else SandboxStatus.FAILED
        except Exception as exc:  # noqa: BLE001
            timed_out = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
            result = SandboxResult(status=SandboxStatus.TIMEOUT if timed_out else SandboxStatus.FAILED, error=str(exc))
        result.duration_seconds = time.time() - started_at
        return result
