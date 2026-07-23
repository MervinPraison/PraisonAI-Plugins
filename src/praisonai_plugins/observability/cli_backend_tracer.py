"""CLI backend delegation tracer plugin."""

from __future__ import annotations

import os
from typing import Any, Dict

from praisonaiagents.plugins.plugin import Plugin, PluginHook, PluginInfo
from praisonaiagents._logging import get_logger

logger = get_logger(__name__)


def _tracing_enabled() -> bool:
    flag = os.environ.get("PRAISONAI_CLI_BACKEND_DEBUG", "").lower()
    if flag in ("1", "true", "yes"):
        return True
    return os.environ.get("LOGLEVEL", "").upper() == "DEBUG"


class CliBackendTracerPlugin(Plugin):
    """Logs CLI backend delegation when enabled via env."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="cli_backend_tracer",
            version="1.0.0",
            description="Logs CLI backend subprocess delegation (no PraisonAI LLM HTTP).",
            author="PraisonAI",
            hooks=[PluginHook.CLI_BACKEND_EXECUTE],
        )

    def cli_backend_execute(self, context: Dict[str, Any]) -> None:
        if not _tracing_enabled():
            return
        logger.info(
            "CLI backend delegation agent=%r backend=%r session_id=%r "
            "transport=%r praisonai_llm_http=%r command=%r error=%r",
            context.get("agent_name"),
            context.get("backend"),
            context.get("session_id"),
            context.get("transport"),
            context.get("praisonai_llm_http"),
            context.get("command"),
            context.get("error"),
        )


def create_plugin() -> Plugin:
    return CliBackendTracerPlugin()
