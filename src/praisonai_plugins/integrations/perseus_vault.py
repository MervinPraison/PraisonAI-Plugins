"""Integration Plugin for PraisonAI Agents."""
from praisonaiagents.plugins.plugin import Plugin, PluginInfo, PluginHook
from praisonaiagents._logging import get_logger
from typing import Dict, Any

logger = get_logger(__name__)

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
        # Register the Perseus Vault memory adapter with the agent framework
        try:
            from praisonaiagents.memory.adapters import register_memory_factory
            from praisonaiagents.memory.adapters.perseus_vault_adapter import create_perseus_vault_memory_adapter
            register_memory_factory("perseus_vault", create_perseus_vault_memory_adapter)
            logger.info("[INTEGRATION] Perseus Vault memory adapter registered.")
        except ImportError:
            logger.warning("[INTEGRATION] Perseus Vault adapter not available — install praisonaiagents with memory support.")
