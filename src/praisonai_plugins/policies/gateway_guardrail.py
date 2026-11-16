"""Gateway Guardrail Plugin: inbound admission and outbound on-channel redaction.

``pii_guardrail`` redacts raw model text on ``AFTER_LLM``. This plugin works on
the gateway boundary instead: it denies unwanted inbound messages on
``message_received`` and redacts the final channel payload on
``message_sending``. Public channels also get phone-number redaction.

Core does not treat plugins from this package as first-party. Conversation
hooks stay suppressed until the operator grants access with
``plugins.gateway_guardrail.allow_conversation: true``. Without that grant the
plugin does nothing.

Optional environment configuration:
  - ``PRAISONAI_GATEWAY_DENIED_SENDERS``: comma-separated sender ids to deny.
  - ``PRAISONAI_GATEWAY_BLOCKED_TERMS``: comma-separated inbound terms to deny.
  - ``PRAISONAI_GATEWAY_MAX_INBOUND``: max inbound length (0 = unlimited).
"""
from __future__ import annotations

import os
import re
from typing import Any

from praisonaiagents._logging import get_logger
from praisonaiagents.plugins.plugin import Plugin, PluginDecision, PluginHook, PluginInfo, PluginType

logger = get_logger(__name__)

_PUBLIC_CHANNEL_TYPES = frozenset({"public", "channel", "group", "public_channel"})
_BASE_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # email
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),  # card-like digit runs
    re.compile(  # common API token prefixes
        r"\b(?:sk-[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}"
        r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
    ),
)
_PUBLIC_PATTERNS = (re.compile(r"\b\+?\d[\d ().-]{7,}\d\b"),)  # phone numbers


def _env_set(name: str) -> set[str]:
    return {s.strip().lower() for s in os.environ.get(name, "").split(",") if s.strip()}


class GatewayGuardrailPlugin(Plugin):
    """Deny unwanted inbound messages and redact the outbound channel payload."""

    def __init__(self) -> None:
        self._denied_senders = _env_set("PRAISONAI_GATEWAY_DENIED_SENDERS")
        self._blocked_terms = _env_set("PRAISONAI_GATEWAY_BLOCKED_TERMS")
        try:
            self._max_inbound = int(os.environ.get("PRAISONAI_GATEWAY_MAX_INBOUND") or 0)
        except ValueError:
            self._max_inbound = 0

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="gateway_guardrail",
            version="1.0.0",
            description="Inbound admission and outbound on-channel redaction for the gateway.",
            author="PraisonAI",
            hooks=[PluginHook.MESSAGE_RECEIVED, PluginHook.MESSAGE_SENDING],
            plugin_type=PluginType.POLICY,
        )

    def before_message(self, message: dict[str, Any]) -> PluginDecision | None:
        sender = str(message.get("sender_id") or "").lower()
        content = str(message.get("content") or "")
        reason = None
        if sender and sender in self._denied_senders:
            reason = "sender not permitted on this channel"
        elif self._max_inbound and len(content) > self._max_inbound:
            reason = "inbound message exceeds size limit"
        elif any(term in content.lower() for term in self._blocked_terms):
            reason = "inbound content is blocklisted"
        if reason is None:
            return None
        logger.warning(f"[GATEWAY] inbound denied ({reason}) platform={message.get('platform')}")
        return PluginDecision.deny(reason)

    def after_message(self, message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if not isinstance(content, str) or not content:
            return message
        patterns = _BASE_PATTERNS
        if str(message.get("channel_type") or "").lower() in _PUBLIC_CHANNEL_TYPES:
            patterns += _PUBLIC_PATTERNS
        redacted = content
        for pattern in patterns:
            redacted = pattern.sub("[REDACTED]", redacted)
        if redacted == content:
            return message
        logger.warning(f"[GATEWAY] outbound payload redacted platform={message.get('platform')}")
        return {**message, "content": redacted}
