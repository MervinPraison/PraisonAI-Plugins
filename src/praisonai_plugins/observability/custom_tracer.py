"""
Custom Tracer Plugin for PraisonAI Agents.
"""
from typing import Any

from praisonaiagents._logging import get_logger
from praisonaiagents.plugins.plugin import Plugin, PluginHook, PluginInfo

logger = get_logger(__name__)

class CustomTracerPlugin(Plugin):
    """
    A simple protocol-driven plugin that traces LLM calls.
    """
    
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="custom_tracer",
            version="1.0.0",
            description="Traces LLM completion calls.",
            author="PraisonAI",
            hooks=[
                PluginHook.BEFORE_LLM,
                PluginHook.AFTER_LLM
            ]
        )
        
    def before_llm(self, messages: list[dict], params: dict[str, Any]) -> tuple:
        logger.info(f"[Trace] Sending {len(messages)} messages to LLM model {params.get('model', 'unknown')}")
        return messages, params
        
    def after_llm(self, response: str, usage: dict[str, Any]) -> str:
        tokens = usage.get('total_tokens', 0) if usage else 0
        logger.info(f"[Trace] LLM responded. Tokens used: {tokens}")
        return response
