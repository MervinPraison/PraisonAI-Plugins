"""
Memory Watchdog Plugin for PraisonAI Agents.

A small, opt-in lifecycle observer for long-running gateways. It attaches to
the ``GATEWAY_START`` / ``GATEWAY_STOP`` hooks and runs a passive, non-blocking
background sampler that periodically logs resident memory (RSS) and GC stats so
operators can *see* slow memory growth before the process is OOM-killed.

On ``GATEWAY_START`` it spawns a daemon thread that, every ``interval`` seconds,
emits a structured line such as::

    [MEMORY] rss=412.3MB gc=(gen0=123,gen1=4,gen2=1) objects=98765

On ``GATEWAY_STOP`` it stops the sampler and emits a final "last RSS before
exit" snapshot so a crash/OOM can be diagnosed from the log alone::

    [MEMORY] final rss=1402.7MB gc=(gen0=...,gen1=...,gen2=...) objects=...

RSS is read from ``resource.getrusage()`` where available and degrades
gracefully (optional ``psutil``) on platforms that lack it. The daemon thread is
fully non-blocking and never delays shutdown.
"""

from praisonaiagents.plugins.plugin import Plugin, PluginInfo, PluginHook
from praisonaiagents._logging import get_logger
from typing import Any, Dict, Optional
import gc
import sys
import threading

logger = get_logger(__name__)


def _read_rss_mb() -> Optional[float]:
    """Return the process resident set size in megabytes, or None if unknown.

    Tries ``resource.getrusage`` first (no extra dependency, available on
    POSIX), then optionally falls back to ``psutil`` if it is installed.
    Returns ``None`` when neither source is available (e.g. on Windows without
    ``psutil``) so callers can degrade gracefully.
    """
    try:
        import resource

        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in kilobytes on Linux and in bytes on macOS.
        if sys.platform == "darwin":
            return ru_maxrss / (1024.0 * 1024.0)
        return ru_maxrss / 1024.0
    except Exception:
        pass

    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def _format_snapshot(tag: str = "") -> str:
    """Build a single ``[MEMORY]`` log line for the current process state."""
    rss = _read_rss_mb()
    rss_str = f"{rss:.1f}MB" if rss is not None else "unknown"

    try:
        counts = gc.get_count()
        gc_str = f"(gen0={counts[0]},gen1={counts[1]},gen2={counts[2]})"
    except Exception:
        gc_str = "(unavailable)"

    try:
        objects = len(gc.get_objects())
    except Exception:
        objects = -1

    prefix = f"[MEMORY] {tag} " if tag else "[MEMORY] "
    return f"{prefix}rss={rss_str} gc={gc_str} objects={objects}"


class MemoryWatchdogPlugin(Plugin):
    """Passive resource health watchdog for long-running gateways.

    Configuration (read from ``on_config``):
        interval: Sampling interval in seconds. Default ``300`` (5 minutes).
        log_level: Logging level name for periodic samples. Default ``"INFO"``.
        use_psutil: Whether to allow the optional ``psutil`` RSS fallback.
            Default ``True`` (still degrades gracefully if not installed).
    """

    def __init__(self) -> None:
        self._interval: float = 300.0
        self._log_level: int = 20  # logging.INFO
        self._use_psutil: bool = True
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="memory_watchdog",
            version="1.0.0",
            description=(
                "Periodically samples RSS/GC for long-running gateways and "
                "logs a final last-RSS-before-exit snapshot on shutdown."
            ),
            author="PraisonAI",
            hooks=[
                PluginHook.GATEWAY_START,
                PluginHook.GATEWAY_STOP,
            ],
        )

    def on_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Read optional configuration before the gateway starts."""
        try:
            if "interval" in config:
                self._interval = max(1.0, float(config["interval"]))
            level = config.get("log_level")
            if isinstance(level, str):
                import logging

                self._log_level = logging.getLevelName(level.upper())
                if not isinstance(self._log_level, int):
                    self._log_level = logging.INFO
            elif isinstance(level, int):
                self._log_level = level
            if "use_psutil" in config:
                self._use_psutil = bool(config["use_psutil"])
        except Exception as e:  # noqa: BLE001 — config must never break the runtime
            logger.debug(f"[MEMORY] config parse error (non-fatal): {e}")
        return config

    def gateway_start(self, event: Any = None) -> Any:
        """Start the non-blocking background sampler when the gateway starts."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return event
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run_sampler,
                name="memory-watchdog",
                daemon=True,
            )
            self._thread.start()
        logger.info(
            f"[MEMORY] watchdog started (interval={self._interval:.0f}s)"
        )
        logger.log(self._log_level, _format_snapshot(tag="start"))
        return event

    def gateway_stop(self, event: Any = None) -> Any:
        """Stop the sampler and emit the final last-RSS-before-exit snapshot."""
        with self._lock:
            stop_event = self._stop_event
            thread = self._thread
            self._stop_event = None
            self._thread = None

        if stop_event is not None:
            stop_event.set()
        # Never block shutdown: only join very briefly and only if alive.
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass

        logger.info(_format_snapshot(tag="final"))
        return event

    def on_shutdown(self) -> None:
        """Ensure the sampler is stopped if the plugin is unregistered."""
        self.gateway_stop(None)

    def get_rss_mb(self) -> Optional[float]:
        """Expose current RSS in MB for the metrics ``_gauge_providers`` seam.

        A gateway can register this as a pull-style gauge provider so the value
        also appears on the metrics endpoint.
        """
        return _read_rss_mb()

    def _run_sampler(self) -> None:
        """Daemon loop: log a sample every ``interval`` seconds until stopped."""
        stop_event = self._stop_event
        if stop_event is None:
            return
        # First periodic sample fires after one full interval; the start
        # snapshot was already emitted in gateway_start.
        while not stop_event.wait(self._interval):
            try:
                logger.log(self._log_level, _format_snapshot())
            except Exception as e:  # noqa: BLE001 — sampler must never crash
                logger.debug(f"[MEMORY] sample error (non-fatal): {e}")
