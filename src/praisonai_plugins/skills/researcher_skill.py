"""
Skills Plugin for PraisonAI Agents.
"""

from praisonaiagents.plugins.plugin import Plugin, PluginInfo
from typing import Dict, Any


class ResearcherSkillPlugin(Plugin):
    """
    A protocol-driven plugin providing higher-level agent skills.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="researcher_skill",
            version="1.0.0",
            description="Capabilities related to deep research.",
            author="PraisonAI",
        )

    def before_agent(self, prompt: str, context: Dict[str, Any]) -> str:
        # Inject standard research protocol instructions into prompt
        research_directives = (
            "\n[SKILL: Focus solely on factual information, cite sources.]\n"
        )
        return prompt + research_directives
