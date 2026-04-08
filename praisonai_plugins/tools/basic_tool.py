"""
Tool Plugin for PraisonAI Agents.
"""
from praisonaiagents.plugins.plugin import Plugin, PluginInfo
from typing import Dict, Any, List

class BasicToolPlugin(Plugin):
    """
    A protocol-driven plugin providing tools.
    """
    
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="basic_tools",
            version="1.0.0",
            description="Provides basic generic tools for agents.",
            author="PraisonAI"
        )
        
    def get_tools(self) -> List[Dict[str, Any]]:
        # Example defining an external tool provided by this plugin
        def random_number() -> int:
            """Returns a random number"""
            return 42
            
        return [random_number]
