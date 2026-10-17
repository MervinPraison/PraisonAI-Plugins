"""
Integration Plugin for PraisonAI Agents.
"""
from typing import Any

from praisonaiagents._logging import get_logger
from praisonaiagents.plugins.plugin import Plugin, PluginHook, PluginInfo

logger = get_logger(__name__)

class SlackIntegrationPlugin(Plugin):
    """
    A protocol-driven plugin for external integrations.
    """
    
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="slack_integration",
            version="1.0.0",
            description="Integrates PraisonAI with Slack.",
            author="PraisonAI",
            hooks=[PluginHook.ON_INIT, PluginHook.AFTER_AGENT],
        )

    def on_init(self, context: dict[str, Any]) -> None:
        logger.info("[INTEGRATION] Slack integration initialized.")
        
    def after_agent(self, response: str, context: dict[str, Any]) -> str:
        # Example: Push to a slack channel
        # slack_client.post_message(channel="#agent-updates", text=response)
        return response
