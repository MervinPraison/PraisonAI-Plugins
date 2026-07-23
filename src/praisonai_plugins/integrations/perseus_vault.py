"""Integration Plugin for PraisonAI Agents."""
from praisonaiagents.plugins.plugin import Plugin, PluginInfo, PluginHook
from praisonaiagents._logging import get_logger
from typing import Dict, Any

logger = get_logger(__name__)

# Modules imported lazily in on_init(). Used to tell a genuinely absent
# optional dependency apart from a broken transitive import.
_ADAPTER_IMPORTS = (
    "praisonaiagents.memory.adapters",
    "praisonaiagents.memory.adapters.perseus_vault_adapter",
)


def _is_optional_dependency_absent(exc: ImportError) -> bool:
    """Return True only when the ImportError means one of the adapter modules
    themselves (or a parent package) is missing.

    Any other missing module means the adapter was found but failed mid-import
    (a broken/transitive dependency), and an ImportError without a module name
    (e.g. ``cannot import name 'register_memory_factory'``) means a version
    mismatch — neither is an "optional dependency absent" condition.
    """
    missing = getattr(exc, "name", None)
    if not missing:
        return False
    return any(target == missing or target.startswith(missing + ".") for target in _ADAPTER_IMPORTS)

class PerseusVaultPlugin(Plugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="perseus_vault",
            version="1.0.0",
            description="Perseus Vault — local-first encrypted persistent memory for PraisonAI agents (single-binary MCP server, no SDK dependency)",
            author="Perseus Computing LLC",
            hooks=[PluginHook.ON_INIT],
        )

    def on_init(self, context: Dict[str, Any]) -> None:
        # Import phase: a genuinely-absent adapter module means the optional
        # dependency is not installed (warn); anything else is a broken
        # install and must be reported with its real cause, not masked.
        try:
            from praisonaiagents.memory.adapters import register_memory_factory
            from praisonaiagents.memory.adapters.perseus_vault_adapter import create_perseus_vault_memory_adapter
        except ImportError as exc:
            if _is_optional_dependency_absent(exc):
                logger.warning(
                    "[INTEGRATION] Perseus Vault adapter not available — "
                    "install praisonaiagents with memory support. (%s)",
                    exc,
                )
            else:
                logger.error(
                    "[INTEGRATION] Perseus Vault adapter is installed but failed "
                    "to import due to a broken dependency: %s",
                    exc,
                )
            return

        # Registration phase: intentionally outside the ImportError guard so a
        # registration failure propagates instead of leaving the backend
        # advertised while its factory was never registered.
        register_memory_factory("perseus_vault", create_perseus_vault_memory_adapter)
        logger.info("[INTEGRATION] Perseus Vault memory adapter registered.")
