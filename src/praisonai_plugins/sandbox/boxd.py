"""
boxd sandbox backend for PraisonAI.

Runs agent code in a boxd machine (a full Ubuntu VM) through the boxd SDK.
Register via the ``praisonai.sandbox`` entry-point group when this package is
installed.

Requires: pip install praisonai-plugins[boxd]
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import shlex
import threading
import time
import uuid
from typing import Any

from praisonaiagents.sandbox import (
    ResourceLimits,
    SandboxConfig,
    SandboxResult,
    SandboxStatus,
)

logger = logging.getLogger(__name__)

_INSTALL_HINT = "pip install praisonai-plugins[boxd]"
_NAME_PREFIX = "praisonai-"
_DEFAULT_MAX_LIFETIME = 24 * 60 * 60
_PYTHON_LANGUAGES = ("python", "python3")
_SHELL_LANGUAGES = ("bash", "sh", "shell")
# Linux caps one argument at 128 KiB, so longer code is uploaded and run as a file.
_MAX_INLINE_COMMAND = 100_000


def _is_not_found(exc: Exception) -> bool:
    try:
        from boxd import NotFoundError
    except ImportError:
        return False
    return isinstance(exc, NotFoundError)


class BoxdSandbox:
    """boxd cloud sandbox for isolated code execution.

    Each sandbox is one boxd machine (a full Ubuntu VM). ``start()`` creates
    it with ``isolated=True``, so it joins no private network and holds no
    credentials. ``stop()`` deletes it. If ``stop()`` never runs, boxd deletes
    the machine itself once it is ``max_lifetime`` seconds old.

    ``SandboxConfig.env`` and the per-call ``env`` are sent to the machine.
    The machine has outbound internet access, whatever
    ``ResourceLimits.network_enabled`` says.

    Example:
        import os
        from praisonai_plugins.sandbox import BoxdSandbox

        os.environ["BOXD_API_KEY"] = "your-api-key"
        sandbox = BoxdSandbox()
        result = await sandbox.execute("print('Hello, World!')")
        print(result.stdout)  # Hello, World!

    Requires:
        - pip install praisonai-plugins[boxd]
        - BOXD_API_KEY environment variable (BOXD_ORG and BOXD_REGION optional)
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        api_key: str | None = None,
        org: str | None = None,
        region: str | None = None,
        vcpu: int | None = None,
        memory: str | None = None,
        from_snapshot: str | None = None,
        max_lifetime: int | None = _DEFAULT_MAX_LIFETIME,
    ):
        """Initialize the boxd sandbox.

        Args:
            config: Optional sandbox configuration
            api_key: boxd API key (defaults to BOXD_API_KEY)
            org: boxd organization (defaults to BOXD_ORG, else the key's org)
            region: "fra" or "nyc" (defaults to BOXD_REGION, else your home region)
            vcpu: vCPUs, paired with memory: 1/"4G", 2/"8G", 4/"16G", 8/"32G"
                (defaults to your org's machine size)
            memory: memory, paired with vcpu
            from_snapshot: start from a saved boxd snapshot instead of the base image
            max_lifetime: seconds after which boxd deletes the machine, whatever
                it is doing, so a process that dies before ``stop()`` leaves
                nothing behind (default 24 hours; None or 0 for no limit)
        """
        self.config = config or SandboxConfig(sandbox_type="boxd")
        self._api_key = api_key or os.environ.get("BOXD_API_KEY", "")
        self.org = org or os.environ.get("BOXD_ORG") or None
        self.region = region or os.environ.get("BOXD_REGION") or None
        self.vcpu = vcpu
        self.memory = memory
        self.from_snapshot = from_snapshot
        self.max_lifetime = max_lifetime
        self.machine_id: str | None = None
        self._client = None
        self._is_running = False
        # Counts stop() calls, so a start() that is still creating the machine
        # can tell that it was stopped in the meantime.
        self._stops = 0
        # A threading lock, not an asyncio one: a sync Agent runs each tool
        # call on its own event loop, and callers may be on different threads.
        self._start_lock = threading.Lock()
        # Guards _stops and _is_running, so a stop() from another thread cannot
        # land between start()'s last check and its "running" mark.
        self._state_lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        """Check if boxd is available."""
        if not self._api_key.strip():
            return False
        try:
            import importlib
            importlib.import_module("boxd")
            return True
        except ImportError:
            return False

    @property
    def sandbox_type(self) -> str:
        return "boxd"

    def _get_client(self):
        if self._client is None:
            from boxd import Boxd

            self._client = Boxd(api_key=self._api_key, region=self.region)
        return self._client

    async def start(self) -> None:
        """Create the machine and wait until it accepts commands."""
        if self._is_running:
            return

        if not self.is_available:
            raise RuntimeError(
                f"boxd is not available. Install the boxd SDK and set BOXD_API_KEY. {_INSTALL_HINT}"
            )

        while not self._start_lock.acquire(blocking=False):
            await asyncio.sleep(0.05)
        try:
            # Another caller may have started the machine while this one waited.
            if not self._is_running:
                await self._create_and_wait()
        finally:
            self._start_lock.release()

    async def _create_and_wait(self) -> None:
        client = self._get_client()
        if self.machine_id is not None:
            # An earlier delete failed. Retry it before making a new machine,
            # so this sandbox never loses track of one.
            leftover = self.machine_id
            await self._delete_machine()
            if self.machine_id is not None:
                raise RuntimeError(
                    f"boxd could not delete machine {leftover}. Call stop() to retry."
                )
        options: dict[str, Any] = {"isolated": True}
        if self.org:
            options["org"] = self.org
        if self.vcpu is not None:
            options["vcpu"] = self.vcpu
        if self.memory is not None:
            options["memory"] = self.memory
        if self.from_snapshot:
            options["from_snapshot"] = self.from_snapshot
        if self.max_lifetime:
            options["delete_after"] = int(self.max_lifetime)

        name = f"{_NAME_PREFIX}{uuid.uuid4().hex[:8]}"
        stops = self._stops
        machine = await self._create_machine(client, name, options)
        self.machine_id = machine.id
        try:
            if stops == self._stops:
                ready = asyncio.get_running_loop().run_in_executor(
                    None, client.machines.wait_until_ready, machine.id
                )
                ready.add_done_callback(lambda done: done.cancelled() or done.exception())
                # Poll, so a stop() ends this wait at once. Its outcome no
                # longer matters then: stop() has deleted the machine.
                while stops == self._stops and not ready.done():
                    await asyncio.wait([ready], timeout=0.05)
                if stops == self._stops:
                    ready.result()
        except BaseException:
            # Also on cancel or Ctrl-C: the machine exists, so delete it.
            await self._delete_machine()
            raise
        with self._state_lock:
            stopped = stops != self._stops
            if not stopped:
                self._is_running = True
        if stopped:
            # stop() ran while the machine was starting, so stop() wins.
            await self._delete_machine()
            return
        logger.info("boxd sandbox started: %s (%s)", name, machine.id)

    async def _create_machine(self, client, name: str, options: dict[str, Any]):
        """Create the machine without losing it to a cancel.

        The create call runs in a worker thread, and a cancel cannot stop a
        thread. So on cancel, wait for the call to finish and delete the
        machine it made before passing the cancel on.
        """
        # A plain executor future, not a task: when an event loop shuts down
        # it cancels every task, and a cancelled task would lose the id.
        pending = asyncio.get_running_loop().run_in_executor(
            None, functools.partial(client.machines.create, name, **options)
        )
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
                except Exception:  # noqa: BLE001
                    break
            if not pending.cancelled() and pending.exception() is None:
                self.machine_id = pending.result().id
                await self._delete_machine()
            raise

    async def _delete_machine(self) -> None:
        """Delete the machine. The id is kept if that fails, so stop() can retry."""
        machine_id = self.machine_id
        if machine_id is None:
            return
        delete = self._get_client().machines.delete
        try:
            try:
                await asyncio.to_thread(delete, machine_id)
            except RuntimeError as exc:
                # No worker threads left, e.g. a cleanup that runs while the
                # interpreter shuts down: delete on this thread instead.
                if "interpreter shutdown" not in str(exc):
                    raise
                delete(machine_id)
        except Exception as exc:  # noqa: BLE001
            if not _is_not_found(exc):
                logger.warning(
                    "boxd machine delete failed for %s, stop() will retry: %s", machine_id, exc
                )
                return
        if self.machine_id == machine_id:
            self.machine_id = None

    async def stop(self) -> None:
        """Delete the machine. If that fails, the next stop() or cleanup() retries."""
        with self._state_lock:
            self._stops += 1
            self._is_running = False
        if self.machine_id is None:
            return
        await self._delete_machine()
        if self.machine_id is None:
            logger.info("boxd sandbox stopped")

    async def execute(
        self,
        code: str,
        language: str = "python",
        limits: ResourceLimits | None = None,
        env: dict[str, str] | None = None,
        working_dir: str | None = None,
    ) -> SandboxResult:
        """Execute Python or shell code in the sandbox.

        ``language`` is "python" or "bash" ("python3", "sh" and "shell" also
        work; shell code runs under bash). Anything else fails.
        """
        language_key = language.lower()
        if language_key not in _PYTHON_LANGUAGES + _SHELL_LANGUAGES:
            return SandboxResult(
                execution_id=str(uuid.uuid4()),
                status=SandboxStatus.FAILED,
                error=f"boxd sandbox supports Python and bash, got {language!r}",
            )
        is_shell = language_key in _SHELL_LANGUAGES

        if not self._is_running:
            await self.start()

        limits = limits or self.config.resource_limits
        command = code if is_shell else f"python3 -c {shlex.quote(code)}"
        if len(command.encode("utf-8")) > _MAX_INLINE_COMMAND:
            return await self._run_uploaded(
                code,
                "bash" if is_shell else "python3",
                "code.sh" if is_shell else "code.py",
                limits=limits,
                env=env,
                working_dir=working_dir,
            )

        return await self._run_command(
            command,
            limits=limits,
            env=env,
            working_dir=working_dir,
            execution_id=str(uuid.uuid4()),
            started_at=time.time(),
        )

    async def run_command(
        self,
        command: str | list[str],
        limits: ResourceLimits | None = None,
        env: dict[str, str] | None = None,
        working_dir: str | None = None,
    ) -> SandboxResult:
        """Run a shell command in the sandbox."""
        if not self._is_running:
            await self.start()

        if isinstance(command, list):
            command = shlex.join(command)

        limits = limits or self.config.resource_limits
        return await self._run_command(
            command,
            limits=limits,
            env=env,
            working_dir=working_dir,
            execution_id=str(uuid.uuid4()),
            started_at=time.time(),
        )

    async def _run_command(
        self,
        command: str,
        *,
        limits: ResourceLimits,
        env: dict[str, str] | None,
        working_dir: str | None,
        execution_id: str,
        started_at: float,
    ) -> SandboxResult:
        merged_env = {**self.config.env, **(env or {})}
        if self.machine_id is None:
            return SandboxResult(
                execution_id=execution_id,
                status=SandboxStatus.FAILED,
                error="boxd sandbox is not running",
                duration_seconds=time.time() - started_at,
                started_at=started_at,
                completed_at=time.time(),
            )
        try:
            # bash -c, so pipes, redirects and $VAR expansion see the env
            # passed to this call.
            result = await asyncio.to_thread(
                self._get_client().machines.exec,
                self.machine_id,
                ["bash", "-c", command],
                env=merged_env or None,
                cwd=working_dir,
                timeout=limits.timeout_seconds,
            )
            status = SandboxStatus.COMPLETED if result.exit_code == 0 else SandboxStatus.FAILED
            return SandboxResult(
                execution_id=execution_id,
                status=status,
                exit_code=result.exit_code,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                duration_seconds=time.time() - started_at,
                started_at=started_at,
                completed_at=time.time(),
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            timed_out = isinstance(exc, TimeoutError) or any(
                word in error.lower() for word in ("timed out", "timeout")
            )
            status = SandboxStatus.TIMEOUT if timed_out else SandboxStatus.FAILED
            return SandboxResult(
                execution_id=execution_id,
                status=status,
                error=error,
                duration_seconds=time.time() - started_at,
                started_at=started_at,
                completed_at=time.time(),
            )

    async def execute_file(
        self,
        file_path: str,
        args: list[str] | None = None,
        limits: ResourceLimits | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Execute a local file in the sandbox."""
        def _read_file() -> str:
            with open(file_path, encoding="utf-8") as handle:
                return handle.read()

        try:
            code = await asyncio.to_thread(_read_file)
        except OSError as exc:
            return SandboxResult(
                execution_id=str(uuid.uuid4()),
                status=SandboxStatus.FAILED,
                error=f"Could not read {file_path}: {exc}",
            )

        language = "bash" if file_path.endswith((".sh", ".bash")) else "python"
        if args:
            return await self._run_uploaded(
                code,
                "bash" if language == "bash" else "python3",
                os.path.basename(file_path),
                args=args,
                limits=limits,
                env=env,
            )

        return await self.execute(code, language=language, limits=limits, env=env)

    async def _run_uploaded(
        self,
        code: str,
        interpreter: str,
        name: str,
        *,
        args: list[str] | None = None,
        limits: ResourceLimits | None = None,
        env: dict[str, str] | None = None,
        working_dir: str | None = None,
    ) -> SandboxResult:
        """Upload code to /tmp, run it as a file, then remove it."""
        remote_path = f"/tmp/{uuid.uuid4().hex}_{name}"
        if not await self.write_file(remote_path, code):
            return SandboxResult(
                execution_id=str(uuid.uuid4()),
                status=SandboxStatus.FAILED,
                error=f"Could not upload {name} to sandbox",
            )
        parts = [interpreter, remote_path] + list(args or [])
        try:
            return await self.run_command(parts, limits=limits, env=env, working_dir=working_dir)
        finally:
            await self._remove_file(remote_path)

    async def _remove_file(self, path: str) -> None:
        if self.machine_id is None:
            return
        try:
            await asyncio.to_thread(
                self._get_client().machines.files.delete, self.machine_id, path
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("boxd could not remove %s: %s", path, exc)

    async def write_file(self, path: str, content: str | bytes) -> bool:
        """Write a file to the sandbox. Parent directories are created."""
        if not self._is_running:
            await self.start()
        if self.machine_id is None:
            logger.error("boxd write_file failed for %s: sandbox is not running", path)
            return False
        try:
            await asyncio.to_thread(
                self._get_client().machines.files.upload, self.machine_id, path, content
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("boxd write_file failed for %s: %s", path, exc)
            return False

    async def read_file(self, path: str) -> str | bytes | None:
        """Read a file from the sandbox: text as str, anything else as bytes."""
        if not self._is_running:
            await self.start()
        if self.machine_id is None:
            logger.warning("boxd read_file failed for %s: sandbox is not running", path)
            return None
        try:
            content = await asyncio.to_thread(
                self._get_client().machines.files.download, self.machine_id, path
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("boxd read_file failed for %s: %s", path, exc)
            return None
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content

    async def list_files(self, path: str = "/") -> list[str]:
        """List the files in a sandbox directory (not recursive)."""
        if not self._is_running:
            await self.start()
        if self.machine_id is None:
            return []
        try:
            listing = await asyncio.to_thread(
                self._get_client().machines.files.list_dir, self.machine_id, path
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("boxd list_files failed for %s: %s", path, exc)
            return []
        base = path.rstrip("/")
        return [f"{base}/{entry.name}" for entry in listing.entries if not entry.is_dir]

    def get_status(self) -> dict[str, Any]:
        """Get sandbox status information."""
        return {
            "available": self.is_available,
            "type": self.sandbox_type,
            "running": self._is_running,
            "machine_id": self.machine_id,
            "region": self.region,
            "api_key_set": bool(self._api_key),
        }

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        await self.stop()

    async def reset(self) -> None:
        """Reset sandbox to a fresh machine."""
        await self.stop()
        await self.start()
