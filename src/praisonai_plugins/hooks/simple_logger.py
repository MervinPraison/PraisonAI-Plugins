"""
Simple Logger Plugin for PraisonAI Agents.
"""
from praisonaiagents.plugins.plugin import Plugin, PluginInfo, PluginHook
from praisonaiagents._logging import get_logger
from typing import Dict, Any

logger = get_logger(__name__)

class SimpleLoggerPlugin(Plugin):
    """
    A simple protocol-driven plugin that logs when an agent starts and finishes,
    and when a tool is called and finishes.
    """
    
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="simple_logger",
            version="1.0.0",
            description="Logs agent and tool execution steps.",
            author="PraisonAI",
            hooks=[
                PluginHook.BEFORE_AGENT,
                PluginHook.AFTER_AGENT,
                PluginHook.BEFORE_TOOL,
                PluginHook.AFTER_TOOL
            ]
        )
        
    def before_agent(self, prompt: str, context: Dict[str, Any]) -> str:
        logger.info(f"Agent starting with prompt length: {len(prompt)}")
        return prompt
        
    def after_agent(self, response: str, context: Dict[str, Any]) -> str:
        logger.info(f"Agent finished. Response length: {len(response)}")
        return response
        
    def before_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Tool executing: {tool_name} with args: {args}")
        return args
        
    def after_tool(self, tool_name: str, result: Any) -> Any:
        logger.info(f"Tool {tool_name} finished. Result: {result}")
        return result
