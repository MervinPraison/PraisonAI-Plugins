"""
Integration Plugin for PraisonAI Agents.
"""
from praisonaiagents.plugins.plugin import Plugin, PluginInfo, PluginHook
from praisonaiagents._logging import get_logger
from typing import Dict, Any

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

    def on_init(self, context: Dict[str, Any]) -> None:
        logger.info("[INTEGRATION] Slack integration initialized.")
        
    def after_agent(self, response: str, context: Dict[str, Any]) -> str:
        # Example: Push to a slack channel
        # slack_client.post_message(channel="#agent-updates", text=response)
        return response
