"""
Policy Plugin for PraisonAI Agents.
"""

from praisonaiagents.plugins.plugin import Plugin, PluginHook, PluginInfo


class StrictTypingPolicyPlugin(Plugin):
    """
    A protocol-driven plugin modifying operational policy.
    """
    
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="strict_policy",
            version="1.0.0",
            description="Enforces strict operational policies and permissions.",
            author="PraisonAI",
            hooks=[PluginHook.ON_PERMISSION_ASK],
        )
        
    def on_permission_ask(self, target: str, reason: str) -> bool | None:
        """Auto-deny highly destructive targets based on policy."""
        if "rm -rf" in target:
            return False
        return None
