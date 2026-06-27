"""
Gateway Forensics & Memory-Leak Monitoring Plugin for PraisonAI Agents.

A drop-in observability plugin for long-running gateways/bots that answers two
production questions the core gateway cannot:

  1. *Why did the process die?*  On SIGTERM/SIGINT it captures a fast (<10ms,
     pure-stdlib) forensic snapshot -- signal, our PID/PPID, parent process
     name+cmdline, load average, debugger/tracer presence and any planned-stop
     marker -- and logs it immediately so a hard shutdown leaves a trail.

  2. *Is it leaking memory?*  A daemon thread logs a grep-able
     ``[MEMORY] rss=... gc=... threads=... uptime=...`` time-series every N
     minutes (``resource.getrusage``, falling back to ``psutil`` on Windows),
     with a baseline on start and a final snapshot on GATEWAY_STOP.

Everything is optional and OS-aware: ``resource``/``psutil``/``/proc`` are
lazy-imported and degrade gracefully so this never breaks the gateway hot path.
"""
from __future__ import annotations

import gc
import os
import platform
import signal
import threading
import time
from typing import Any, Dict, List, Optional

from praisonaiagents.plugins.plugin import Plugin, PluginInfo, PluginHook
from praisonaiagents._logging import get_logger

logger = get_logger(__name__)

# Environment marker an operator/deploy script can set to indicate an intended
# stop (e.g. a rolling deploy), so forensics can distinguish it from a crash.
_PLANNED_STOP_ENV = "PRAISONAI_PLANNED_STOP"


def _read_proc_rss_bytes() -> Optional[int]:
    """Best-effort resident set size in bytes. Returns None if unavailable."""
    # Linux/most-Unix: resource.getrusage (ru_maxrss is KB on Linux, bytes on macOS)
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        maxrss = usage.ru_maxrss
        if maxrss:
            # macOS reports bytes; Linux reports kilobytes.
            if platform.system() == "Darwin":
                return int(maxrss)
            return int(maxrss) * 1024
    except Exception:
        pass

    # Windows / fallback: psutil if installed (optional dependency).
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    return None


def _format_mb(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def _load_average() -> str:
    try:
        if hasattr(os, "getloadavg"):
            one, _five, _fifteen = os.getloadavg()
            return f"{one:.2f}"
    except Exception:
        pass
    return "n/a"


def _tracer_pid() -> Optional[int]:
    """Detect an attached debugger/tracer via /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("TracerPid:"):
                    return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return None


def _parent_info(ppid: int) -> str:
    """Best-effort parent process name + cmdline. Pure-stdlib on Linux."""
    name = "unknown"
    cmdline = ""
    try:
        with open(f"/proc/{ppid}/comm", "r") as fh:
            name = fh.read().strip() or name
    except Exception:
        pass
    try:
        with open(f"/proc/{ppid}/cmdline", "rb") as fh:
            raw = fh.read().replace(b"\x00", b" ").strip()
            cmdline = raw.decode("utf-8", "replace")
    except Exception:
        pass

    if not cmdline and name == "unknown":
        # Fall back to psutil if /proc is unavailable (e.g. macOS/Windows).
        try:
            import psutil

            proc = psutil.Process(ppid)
            name = proc.name()
            cmdline = " ".join(proc.cmdline())
        except Exception:
            pass

    return f"{name}({ppid})" + (f" [{cmdline}]" if cmdline else "")


class GatewayForensicsPlugin(Plugin):
    """Shutdown forensics + memory-leak monitoring for long-running gateways.

    Enabled automatically when the package is installed (via the
    ``praisonai.plugins`` entry point). Configurable through environment:

      - ``PRAISONAI_FORENSICS_INTERVAL``  memory-log interval in seconds (default 300)
      - ``PRAISONAI_PLANNED_STOP``        set truthy to mark an intended shutdown
    """

    def __init__(self) -> None:
        self._start_time: float = time.time()
        self._baseline_rss: Optional[int] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._signals_installed = False
        self._prev_handlers: Dict[int, Any] = {}

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="gateway_forensics",
            version="1.0.0",
            description="Crash/shutdown forensics and memory-leak monitoring for long-running gateways.",
            author="PraisonAI",
            hooks=[
                PluginHook.GATEWAY_START,
                PluginHook.GATEWAY_STOP,
            ],
        )

    # ------------------------------------------------------------------ hooks

    def gateway_start(self, *args: Any, **kwargs: Any) -> None:
        """GATEWAY_START hook: install signal forensics + start memory monitor."""
        self._start_time = time.time()
        self._baseline_rss = _read_proc_rss_bytes()
        logger.info(
            f"[MEMORY] baseline rss={_format_mb(self._baseline_rss)} "
            f"threads={threading.active_count()} pid={os.getpid()}"
        )
        self._install_signal_forensics()
        interval = self._interval_seconds()
        self._start_memory_monitoring(interval)

    def gateway_stop(self, *args: Any, **kwargs: Any) -> None:
        """GATEWAY_STOP hook: final memory snapshot + tear down monitor."""
        self._log_memory_usage(tag="final")
        self._stop_event.set()
        self._restore_signal_handlers()

    def on_shutdown(self) -> None:
        """Plugin lifecycle teardown (manager.unregister)."""
        self._stop_event.set()
        self._restore_signal_handlers()

    # -------------------------------------------------------------- internals

    def _interval_seconds(self) -> int:
        raw = os.environ.get("PRAISONAI_FORENSICS_INTERVAL", "").strip()
        if raw:
            try:
                value = int(float(raw))
                if value > 0:
                    return value
            except ValueError:
                logger.warning(
                    f"Invalid PRAISONAI_FORENSICS_INTERVAL={raw!r}; using default 300s"
                )
        return 300

    def _install_signal_forensics(self) -> None:
        if self._signals_installed:
            return
        # Signal handlers can only be installed from the main thread.
        if threading.current_thread() is not threading.main_thread():
            logger.debug(
                "gateway_forensics: not on main thread; skipping signal handler install"
            )
            return

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except (ValueError, OSError, RuntimeError) as exc:
                logger.debug(f"gateway_forensics: could not install handler for {sig}: {exc}")
        self._signals_installed = True

    def _restore_signal_handlers(self) -> None:
        if not self._signals_installed:
            return
        for sig, handler in self._prev_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, RuntimeError):
                pass
        self._prev_handlers.clear()
        self._signals_installed = False

    def _handle_signal(self, signum: int, frame: Any) -> None:
        # Fast, pure-stdlib forensic snapshot -- keep this under ~10ms.
        try:
            try:
                signame = signal.Signals(signum).name
            except Exception:
                signame = str(signum)
            ppid = os.getppid()
            planned = os.environ.get(_PLANNED_STOP_ENV, "").strip().lower() in (
                "1", "true", "yes", "on",
            )
            tracer = _tracer_pid()
            tracer_str = "none" if not tracer else str(tracer)
            logger.warning(
                f"[SHUTDOWN] signal={signame} pid={os.getpid()} "
                f"parent={_parent_info(ppid)} load={_load_average()} "
                f"tracer={tracer_str} planned_stop={str(planned).lower()}"
            )
            self._log_memory_usage(tag="on_signal")
        except Exception as exc:  # never let forensics block shutdown
            logger.debug(f"gateway_forensics: signal snapshot failed: {exc}")
        finally:
            self._stop_event.set()
            # Re-raise to the previously installed handler so we don't swallow
            # the gateway's own graceful-shutdown logic / default behaviour.
            prev = self._prev_handlers.get(signum)
            try:
                if callable(prev):
                    prev(signum, frame)
                elif prev == signal.SIG_DFL:
                    signal.signal(signum, signal.SIG_DFL)
                    os.kill(os.getpid(), signum)
            except Exception:
                pass

    def _start_memory_monitoring(self, interval_s: int) -> None:
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()

        def _run() -> None:
            # Wait in small slices so shutdown is responsive.
            while not self._stop_event.is_set():
                slept = 0.0
                while slept < interval_s and not self._stop_event.is_set():
                    time.sleep(min(1.0, interval_s - slept))
                    slept += 1.0
                if self._stop_event.is_set():
                    break
                self._log_memory_usage(tag="periodic")

        self._monitor_thread = threading.Thread(
            target=_run, name="praisonai-gateway-forensics", daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"[MEMORY] monitor started interval={interval_s}s")

    def _log_memory_usage(self, tag: str = "periodic") -> None:
        try:
            rss = _read_proc_rss_bytes()
            counts = gc.get_count()
            uptime = int(time.time() - self._start_time)
            delta = ""
            if rss is not None and self._baseline_rss:
                growth = rss - self._baseline_rss
                delta = f" delta={_format_mb(growth)}"
            logger.info(
                f"[MEMORY] tag={tag} rss={_format_mb(rss)}{delta} "
                f"gc={counts} threads={threading.active_count()} uptime={uptime}s"
            )
        except Exception as exc:
            logger.debug(f"gateway_forensics: memory snapshot failed: {exc}")


def create_plugin() -> GatewayForensicsPlugin:
    """Factory used by directory/module loaders."""
    return GatewayForensicsPlugin()
