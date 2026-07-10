"""
Skill-Library Curator Plugin for PraisonAI Agents.

A drop-in, opt-in lifecycle plugin that keeps a self-improving agent's
auto-authored skill library healthy without user attention. An always-on
gateway run with ``Agent(self_improve=True)`` autonomously authors new skills
after tasks and marks them ``agent_created``; the core skills subsystem ships
the *mechanism* for governance (provenance/usage telemetry + a recoverable
``archive_skill`` store) but defers the *policy* — when to age, what to
consolidate — to "lifecycle curator plugins". This is that plugin.

On ``GATEWAY_START`` it spawns a daemon thread that periodically sweeps the
skill library and, for **agent-created** skills idle longer than a configurable
window, calls the core ``SkillManager.archive_skill`` (recoverable, never a
hard delete). User-authored, bundled, and hub-installed skills are never touched
because they carry ``agent_created=False``. On ``GATEWAY_STOP`` the sweeper is
stopped cleanly and never delays shutdown.

An optional, opt-in consolidation pass proposes merging overlapping narrow
skills into a broader umbrella skill; it defaults to a dry-run report and never
mutates skills on its own.

Configuration (via ``on_config`` / the ``skill_curator:`` plugin block):
    interval_hours: Sweep interval in hours. Default ``168`` (weekly).
    stale_after_days: Idle window before an agent-created skill is archived.
        Default ``30``.
    consolidate: Whether to run the (dry-run) consolidation proposal pass.
        Default ``False``.
    min_use_count: Skills with a ``use_count`` at/above this are pinned from
        ageing regardless of idle time. Default ``0`` (disabled).
    dry_run: When ``True``, log what *would* be archived without archiving.
        Default ``False``.

Everything is safe by default: the plugin is inert unless installed/enabled,
uses long defaults, only ever archives (recoverably), and never blocks the
gateway hot path or shutdown.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from praisonaiagents.plugins.plugin import Plugin, PluginInfo, PluginHook
from praisonaiagents._logging import get_logger

logger = get_logger(__name__)


def _idle_days(skill: Any) -> Optional[float]:
    """Return days since a skill was last used (or created), or None.

    Prefers the core-provided ``idle_days`` helper on ``SkillProperties`` and
    degrades gracefully if the field/property is unavailable on an older SDK.
    """
    props = getattr(skill, "properties", skill)
    idle = getattr(props, "idle_days", None)
    try:
        return float(idle) if idle is not None else None
    except (TypeError, ValueError):
        return None


class SkillCuratorPlugin(Plugin):
    """Usage-driven ageing + optional consolidation for agent-created skills.

    Mirrors the shipped ``memory_watchdog`` structure: a daemon-thread sweeper
    started on ``GATEWAY_START`` and stopped cleanly on ``GATEWAY_STOP``.
    """

    def __init__(self, manager: Any = None) -> None:
        self._interval_seconds: float = 168.0 * 3600.0
        self._stale_after_days: float = 30.0
        self._consolidate: bool = False
        self._min_use_count: int = 0
        self._dry_run: bool = False
        self._manager: Any = manager
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def info(self) -> PluginInfo:
        # Reference hooks defensively so plugin discovery still succeeds on an
        # older praisonaiagents that lacks the GATEWAY_* enum members.
        hooks = [
            getattr(PluginHook, name)
            for name in ("GATEWAY_START", "GATEWAY_STOP")
            if hasattr(PluginHook, name)
        ]
        return PluginInfo(
            name="skill_curator",
            version="1.0.0",
            description=(
                "Ages agent-created skills active->stale->archived by usage and "
                "optionally proposes consolidation, keeping self-improving "
                "gateways' skill libraries healthy."
            ),
            author="PraisonAI",
            hooks=hooks,
        )

    def on_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Read optional configuration before the gateway starts."""
        try:
            if "interval_hours" in config:
                hours = float(config["interval_hours"])
                self._interval_seconds = max(1.0, hours * 3600.0)
            if "stale_after_days" in config:
                self._stale_after_days = max(0.0, float(config["stale_after_days"]))
            if "consolidate" in config:
                self._consolidate = bool(config["consolidate"])
            if "min_use_count" in config:
                self._min_use_count = max(0, int(config["min_use_count"]))
            if "dry_run" in config:
                self._dry_run = bool(config["dry_run"])
        except Exception as e:  # noqa: BLE001 — config must never break the runtime
            logger.debug(f"[SKILL_CURATOR] config parse error (non-fatal): {e}")
        return config

    # ------------------------------------------------------------------ hooks

    def gateway_start(self, event: Any = None) -> Any:
        """Start the non-blocking background sweeper when the gateway starts."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return event
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run_sweeper,
                name="skill-curator",
                daemon=True,
            )
            self._thread.start()
        logger.info(
            f"[SKILL_CURATOR] curator started "
            f"(interval={self._interval_seconds / 3600.0:.0f}h, "
            f"stale_after={self._stale_after_days:.0f}d, "
            f"consolidate={self._consolidate}, dry_run={self._dry_run})"
        )
        return event

    def gateway_stop(self, event: Any = None) -> Any:
        """Stop the sweeper cleanly; never block shutdown."""
        with self._lock:
            stop_event = self._stop_event
            thread = self._thread
            self._stop_event = None
            self._thread = None

        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass

        logger.info("[SKILL_CURATOR] curator stopped")
        return event

    def on_shutdown(self) -> None:
        """Ensure the sweeper is stopped if the plugin is unregistered."""
        self.gateway_stop(None)

    # -------------------------------------------------------------- internals

    def _get_manager(self) -> Any:
        """Return the injected SkillManager or lazily construct a discovered one."""
        if self._manager is not None:
            return self._manager
        try:
            from praisonaiagents.skills.manager import SkillManager

            manager = SkillManager()
            try:
                manager.discover()
            except Exception:
                logger.debug("[SKILL_CURATOR] skill discovery failed", exc_info=True)
            self._manager = manager
            return manager
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[SKILL_CURATOR] could not build SkillManager: {e}")
            return None

    def _is_pinned(self, skill: Any) -> bool:
        """Skills pinned from ageing: heavily-used entries above the threshold."""
        if self._min_use_count <= 0:
            return False
        props = getattr(skill, "properties", skill)
        use_count = getattr(props, "use_count", 0) or 0
        try:
            return int(use_count) >= self._min_use_count
        except (TypeError, ValueError):
            return False

    def sweep(self, manager: Any = None) -> Dict[str, Any]:
        """Run a single curation sweep. Returns a summary dict.

        Only ``agent_created`` skills are eligible; user/bundled/hub skills
        (``agent_created=False``) are never touched. Skills idle longer than
        ``stale_after_days`` are archived via the recoverable
        ``SkillManager.archive_skill``.
        """
        manager = manager or self._get_manager()
        archived: List[str] = []
        skipped_pinned: List[str] = []
        scanned = 0
        if manager is None:
            return {"scanned": 0, "archived": archived, "skipped_pinned": skipped_pinned}

        try:
            skills = list(getattr(manager, "skills", []) or [])
        except Exception:
            logger.debug("[SKILL_CURATOR] could not list skills", exc_info=True)
            skills = []

        for skill in skills:
            scanned += 1
            props = getattr(skill, "properties", skill)
            if not getattr(props, "agent_created", False):
                continue
            if self._is_pinned(skill):
                skipped_pinned.append(getattr(props, "name", "?"))
                continue
            idle = _idle_days(skill)
            if idle is None or idle <= self._stale_after_days:
                continue

            name = getattr(props, "name", None)
            if not name:
                continue
            if self._dry_run:
                logger.info(
                    f"[SKILL_CURATOR] would archive '{name}' (idle={idle:.1f}d)"
                )
                archived.append(name)
                continue
            try:
                result = manager.archive_skill(name)
            except Exception as e:  # noqa: BLE001 — sweeper must never crash
                logger.debug(f"[SKILL_CURATOR] archive '{name}' failed: {e}")
                continue
            if isinstance(result, dict) and result.get("success"):
                archived.append(name)
                logger.info(
                    f"[SKILL_CURATOR] archived '{name}' (idle={idle:.1f}d, recoverable)"
                )
            else:
                logger.debug(
                    f"[SKILL_CURATOR] archive '{name}' returned {result!r}"
                )

        if self._consolidate:
            try:
                self._propose_consolidation(manager, skills)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[SKILL_CURATOR] consolidation pass failed: {e}")

        if archived or skipped_pinned:
            logger.info(
                f"[SKILL_CURATOR] sweep complete: scanned={scanned} "
                f"archived={len(archived)} pinned={len(skipped_pinned)}"
            )
        return {
            "scanned": scanned,
            "archived": archived,
            "skipped_pinned": skipped_pinned,
        }

    def _propose_consolidation(self, manager: Any, skills: List[Any]) -> None:
        """Opt-in, dry-run-by-default: report overlapping narrow skills.

        Deliberately non-mutating: groups agent-created skills that share a
        leading name token and reports candidate umbrella merges for a human to
        act on. Opinionated LLM-driven merging is intentionally left out of the
        default path to respect the protocol-driven-core philosophy.
        """
        groups: Dict[str, List[str]] = {}
        for skill in skills:
            props = getattr(skill, "properties", skill)
            if not getattr(props, "agent_created", False):
                continue
            name = getattr(props, "name", "") or ""
            token = name.replace("_", "-").split("-", 1)[0].lower()
            if token:
                groups.setdefault(token, []).append(name)

        for token, names in groups.items():
            if len(names) > 1:
                logger.info(
                    f"[SKILL_CURATOR] consolidation candidate: {len(names)} "
                    f"skills share prefix '{token}': {sorted(names)} "
                    f"(dry-run; no changes made)"
                )

    def _run_sweeper(self) -> None:
        """Daemon loop: sweep every ``interval`` seconds until stopped."""
        stop_event = self._stop_event
        if stop_event is None:
            return
        while not stop_event.wait(self._interval_seconds):
            try:
                self.sweep()
            except Exception as e:  # noqa: BLE001 — sweeper must never crash
                logger.debug(f"[SKILL_CURATOR] sweep error (non-fatal): {e}")


def create_plugin() -> SkillCuratorPlugin:
    """Factory used by directory/module loaders."""
    return SkillCuratorPlugin()
