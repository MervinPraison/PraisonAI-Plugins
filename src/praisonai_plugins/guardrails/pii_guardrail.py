"""
Guardrails Plugin for PraisonAI Agents.
"""
from typing import Any

from praisonaiagents._logging import get_logger
from praisonaiagents.plugins.plugin import Plugin, PluginHook, PluginInfo

logger = get_logger(__name__)

class PIIGuardrailPlugin(Plugin):
    """
    A protocol-driven plugin evaluating guardrails.
    """
    
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="pii_guardrail",
            version="1.0.0",
            description="Guardrails looking for PII in responses.",
            author="PraisonAI",
            hooks=[PluginHook.AFTER_LLM]
        )
        
    def after_llm(self, response: str, usage: dict[str, Any]) -> str:
        if "social security" in response.lower():
            logger.warning("[GUARDRAIL] Potential PII detected in LLM response.")
            # Can rewrite or mask
            return "[REDACTED BY GUARDRAIL]"
        return response
