"""
Capsule sandbox backend for PraisonAI.

Lightweight WebAssembly isolation for untrusted code. Register via the
``praisonai.sandbox`` entry-point group when this package is installed.

Requires: pip install praisonai-plugins[capsule]
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any

from praisonaiagents.sandbox import ResourceLimits, SandboxResult, SandboxStatus

logger = logging.getLogger(__name__)

_INSTALL_HINT = "pip install praisonai-plugins[capsule]"


class CapsuleSandbox:
    """Capsule-based sandbox for lightweight WebAssembly code execution."""

    def __init__(self, config: Any | None = None, timeout: int = 60):
        self.config = config
        self.timeout = getattr(config, "timeout", timeout) if config is not None else timeout
        self._sandbox = None
        self._is_running = False
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        try:
            import capsule  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def sandbox_type(self) -> str:
        return "capsule"

    async def start(self) -> None:
        with self._lock:
            if self._is_running:
                return
            if not self.is_available:
                raise RuntimeError(f"Capsule backend not available. Install with: {_INSTALL_HINT}")
            try:
                import capsule

                self._sandbox = capsule.Sandbox()
                self._is_running = True
                logger.info("Capsule sandbox initialized")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Capsule sandbox: {e}") from e

    def _teardown_sandbox(self) -> None:
        sandbox = self._sandbox
        self._sandbox = None
        self._is_running = False
        if sandbox is not None:
            for method in ("close", "shutdown", "stop"):
                closer = getattr(sandbox, method, None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:  # noqa: BLE001
                        logger.warning("Capsule sandbox %s() raised during teardown", method)
                    break

    async def stop(self) -> None:
        with self._lock:
            self._teardown_sandbox()
        logger.info("Capsule sandbox stopped")

    async def execute(
        self,
        code: str,
        language: str = "python",
        limits: ResourceLimits | None = None,
        env: dict[str, str] | None = None,
        working_dir: str | None = None,
    ) -> SandboxResult:
        if not self._is_running:
            await self.start()

        execution_id = str(uuid.uuid4())
        started_at = time.time()

        if language.lower() != "python":
            completed_at = time.time()
            return SandboxResult(
                execution_id=execution_id,
                status=SandboxStatus.FAILED,
                error=f"Capsule sandbox only supports Python, got {language!r}",
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=completed_at - started_at,
                metadata={"platform": "capsule", "language": language},
            )

        timeout = self.timeout
        if limits is not None and getattr(limits, "timeout_seconds", None):
            timeout = limits.timeout_seconds

        try:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, self._run_code, code, env)
            if timeout and timeout > 0:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future

            completed_at = time.time()
            duration = completed_at - started_at
            status = SandboxStatus.COMPLETED if result["exit_code"] == 0 else SandboxStatus.FAILED
            return SandboxResult(
                execution_id=execution_id,
                status=status,
                exit_code=result["exit_code"],
                stdout=result["stdout"],
                stderr=result["stderr"],
                duration_seconds=duration,
                started_at=started_at,
                completed_at=completed_at,
                metadata={"platform": "capsule", "language": language},
            )
        except asyncio.TimeoutError:
            completed_at = time.time()
            return SandboxResult(
                execution_id=execution_id,
                status=SandboxStatus.TIMEOUT,
                error=f"Execution exceeded timeout of {timeout}s",
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=completed_at - started_at,
                metadata={"platform": "capsule", "language": language},
            )
        except Exception as e:
            completed_at = time.time()
            error_msg = str(e)
            status = SandboxStatus.TIMEOUT if "timeout" in error_msg.lower() else SandboxStatus.FAILED
            return SandboxResult(
                execution_id=execution_id,
                status=status,
                error=error_msg,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=completed_at - started_at,
                metadata={"platform": "capsule", "language": language},
            )

    def _run_code(self, code: str, env: dict[str, str] | None = None) -> dict[str, Any]:
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Capsule sandbox is not running")
        with self._lock:
            try:
                result = sandbox.run(code, env=env)
            except TypeError as e:
                if "keyword argument" in str(e):
                    result = sandbox.run(code)
                else:
                    raise

        stdout = getattr(result, "stdout", None)
        if stdout is None:
            stdout = str(result) if result is not None else ""
        stderr = getattr(result, "stderr", "") or ""
        exit_code = getattr(result, "exit_code", 0)
        return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}

    async def execute_file(
        self,
        file_path: str,
        args: list[str] | None = None,
        limits: ResourceLimits | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        started_at = time.time()
        try:
            loop = asyncio.get_running_loop()

            def _read_file() -> str:
                with open(file_path, encoding="utf-8") as fh:
                    return fh.read()

            content = await loop.run_in_executor(None, _read_file)
        except OSError as e:
            completed_at = time.time()
            return SandboxResult(
                execution_id=str(uuid.uuid4()),
                status=SandboxStatus.FAILED,
                error=f"Could not read file {file_path!r}: {e}",
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=completed_at - started_at,
                metadata={"platform": "capsule", "file": file_path},
            )
        if args:
            argv = [file_path] + list(args)
            content = f"import sys\nsys.argv = {argv!r}\n" + content
        return await self.execute(content, "python", limits, env)

    async def run_command(
        self,
        command: str | list[str],
        limits: ResourceLimits | None = None,
        env: dict[str, str] | None = None,
        working_dir: str | None = None,
    ) -> SandboxResult:
        now = time.time()
        return SandboxResult(
            execution_id=str(uuid.uuid4()),
            status=SandboxStatus.FAILED,
            error="Capsule sandbox does not support shell commands.",
            started_at=now,
            completed_at=now,
            duration_seconds=0.0,
            metadata={"platform": "capsule"},
        )

    async def write_file(self, path: str, content: str | bytes) -> bool:
        logger.warning("Capsule sandbox write_file is not supported for Wasm isolation")
        return False

    async def read_file(self, path: str) -> str | bytes | None:
        logger.warning("Capsule sandbox read_file is not supported for Wasm isolation")
        return None

    async def list_files(self, path: str = "/") -> list[str]:
        logger.warning("Capsule sandbox list_files is not supported for Wasm isolation")
        return []

    def get_status(self) -> dict[str, Any]:
        return {
            "available": self.is_available,
            "type": self.sandbox_type,
            "running": self._is_running,
            "timeout": self.timeout,
        }

    async def cleanup(self) -> None:
        with self._lock:
            self._teardown_sandbox()
        logger.info("Capsule sandbox cleanup complete")

    async def reset(self) -> None:
        await self.stop()
        await self.start()
