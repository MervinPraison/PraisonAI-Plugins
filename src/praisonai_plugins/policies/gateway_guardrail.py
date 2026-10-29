"""
Gateway Guardrail Plugin for PraisonAI Agents.

A production chat-gateway lifecycle plugin that governs the *conversation
boundary*: it admits/filters inbound messages and redacts the **outbound
on-channel payload that is actually sent to the channel** -- the surface no
shipped guardrail previously covered.

Why this exists
---------------
``pii_guardrail`` redacts on ``AFTER_LLM`` (the raw model text). The text that
leaves the bot, however, is produced *after* formatting, chunking, template
wrapping and media handling -- so secrets/PII introduced by a tool result, a
template, or a quoted attachment can still reach a public channel unredacted.
This plugin runs redaction against the **final channel payload** on
``message_sending`` (``after_message``), and admits/filters inbound traffic on
``message_received`` (``before_message``).

What it does, out of the box
----------------------------
- **Inbound admission / filter** (``before_message``): deny unauthorised
  senders, blocklisted inbound content, or oversized input, with a per-channel
  decision -- fronting the gateway's admission seam.
- **Outbound on-channel redaction** (``after_message``): redact the *final*
  channel text (post-format/chunk) so PII/secrets from tool results, templates
  and quoted media are caught before they leave.
- **Per-channel policy**: stricter redaction in public channels than in DMs,
  keyed on ``platform`` / ``channel_type``.

Safe by default
---------------
As a first-party (bundled) plugin it is granted the conversation-content hooks
(``message_received`` / ``message_sending``) by core's least-privilege gate, so
enabling it gives real protection rather than a silent no-op. It also exposes
``as_guardrail()`` so a ``GUARDRAIL`` consumer can reuse the same redaction via
``validate_output`` / ``validate_tool_result``.

Everything is pure-stdlib and lazy; no third-party SDKs are imported.
"""
from __future__ import annotations

import os
import re
from typing import Any

from praisonaiagents._logging import get_logger
from praisonaiagents.plugins.plugin import (
    Plugin,
    PluginDecision,
    PluginHook,
    PluginInfo,
    PluginType,
)

logger = get_logger(__name__)


# Channel types treated as "public" (stricter redaction) when a per-channel
# policy is applied. DMs/private threads get the baseline ruleset.
_PUBLIC_CHANNEL_TYPES = frozenset({"public", "channel", "group", "public_channel"})

# Baseline redaction patterns applied on every outbound payload. Kept
# conservative and deterministic (no network, no ML) so the hot path stays fast.
_BASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # US Social Security Number: 123-45-6789
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Email address
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # Credit-card-like 13-16 digit runs (optionally space/dash grouped)
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    # Common API/secret token prefixes (OpenAI, GitHub, AWS, Slack, generic).
    ("secret", re.compile(
        r"\b(?:sk-[A-Za-z0-9]{16,}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}"
        r"|AKIA[0-9A-Z]{16}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,})\b"
    )),
)

# Extra patterns applied only on public channels (stricter than DMs).
_PUBLIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # E.164-ish phone numbers
    ("phone", re.compile(r"\b\+?\d[\d ().-]{7,}\d\b")),
)

_REDACTION_MARK = "[REDACTED]"


def _env_set(name: str) -> set:
    """Parse a comma-separated env var into a lowercased set (empty if unset)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


class GatewayGuardrailPlugin(Plugin):
    """Admit inbound, redact the outbound channel payload, per-channel policy.

    Configuration (all optional, via environment so an operator can tune it
    without code changes):

      - ``PRAISONAI_GATEWAY_DENIED_SENDERS``  comma-separated sender ids denied
        on inbound (admission).
      - ``PRAISONAI_GATEWAY_BLOCKED_TERMS``   comma-separated terms; an inbound
        message containing any is dropped.
      - ``PRAISONAI_GATEWAY_MAX_INBOUND``     max inbound content length; longer
        messages are denied (flood/oversize guard). Default 0 = unlimited.
    """

    def __init__(self) -> None:
        self._denied_senders = _env_set("PRAISONAI_GATEWAY_DENIED_SENDERS")
        self._blocked_terms = _env_set("PRAISONAI_GATEWAY_BLOCKED_TERMS")
        try:
            self._max_inbound = int(os.environ.get("PRAISONAI_GATEWAY_MAX_INBOUND", "0") or "0")
        except ValueError:
            self._max_inbound = 0

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="gateway_guardrail",
            version="1.0.0",
            description=(
                "Gateway conversation-boundary guardrail: inbound admission/filter "
                "and outbound on-channel redaction with per-channel policy."
            ),
            author="PraisonAI",
            hooks=[PluginHook.MESSAGE_RECEIVED, PluginHook.MESSAGE_SENDING],
            plugin_type=PluginType.POLICY,
        )

    # ------------------------------------------------------------- inbound

    def before_message(
        self, message: dict[str, Any]
    ) -> dict[str, Any] | PluginDecision | None:
        """Admit / filter an inbound message. Returns a deny decision or None."""
        sender = str(message.get("sender_id", "") or "").lower()
        content = message.get("content", "") or ""
        platform = message.get("platform", "unknown")
        channel = message.get("channel_id", "unknown")

        if sender and sender in self._denied_senders:
            logger.warning(
                f"[GATEWAY] inbound denied: sender={sender} "
                f"platform={platform} channel={channel}"
            )
            return PluginDecision.deny("sender not permitted on this channel")

        if self._max_inbound and len(content) > self._max_inbound:
            logger.warning(
                f"[GATEWAY] inbound denied: oversized ({len(content)} > "
                f"{self._max_inbound}) platform={platform} channel={channel}"
            )
            return PluginDecision.deny("inbound message exceeds size limit")

        if self._blocked_terms:
            lowered = content.lower()
            for term in self._blocked_terms:
                if term in lowered:
                    logger.warning(
                        f"[GATEWAY] inbound denied: blocked term "
                        f"platform={platform} channel={channel}"
                    )
                    return PluginDecision.deny("inbound content is blocklisted")

        return None

    # ------------------------------------------------------------ outbound

    def after_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Redact the FINAL outbound channel payload, per channel."""
        content = message.get("content", "")
        if not content:
            return message

        channel_type = str(message.get("channel_type", "") or "").lower()
        redacted = self._redact(content, channel_type=channel_type)
        if redacted != content:
            logger.warning(
                f"[GATEWAY] outbound redacted on-channel payload "
                f"platform={message.get('platform', 'unknown')} "
                f"channel={message.get('channel_id', 'unknown')} "
                f"channel_type={channel_type or 'unknown'}"
            )
            message = dict(message)
            message["content"] = redacted
        return message

    # ------------------------------------------------------------ redaction

    def _redact(self, text: str, channel_type: str = "") -> str:
        """Apply baseline (and, on public channels, stricter) redaction."""
        patterns: list[tuple[str, re.Pattern[str]]] = list(_BASE_PATTERNS)
        if channel_type in _PUBLIC_CHANNEL_TYPES:
            patterns += list(_PUBLIC_PATTERNS)

        result = text
        for _label, pattern in patterns:
            result = pattern.sub(_REDACTION_MARK, result)
        return result

    # ----------------------------------------------------------- guardrail

    def as_guardrail(self) -> Any | None:
        """Expose the same redaction as a ``GuardrailProtocol`` object.

        Lets a ``GUARDRAIL`` consumer reuse outbound redaction via
        ``validate_output`` and raw tool-result redaction via
        ``validate_tool_result`` -- the paths ``after_llm``-only guardrails miss.
        """
        return _GatewayRedactionGuardrail(self)


class _GatewayRedactionGuardrail:
    """Adapter exposing :class:`GatewayGuardrailPlugin` redaction as a guardrail."""

    def __init__(self, plugin: GatewayGuardrailPlugin) -> None:
        self._plugin = plugin

    def validate_input(self, content: str, **kwargs: Any) -> tuple[bool, str]:
        # Inbound text is admitted/redacted; never reject here (fail-open on text).
        return True, self._plugin._redact(content or "")

    def validate_output(self, content: str, **kwargs: Any) -> tuple[bool, str]:
        channel_type = str(kwargs.get("channel_type", "") or "").lower()
        return True, self._plugin._redact(content or "", channel_type=channel_type)

    def validate_tool_result(self, tool_name: str, result: Any, **kwargs: Any) -> tuple[bool, Any]:
        if isinstance(result, str):
            return True, self._plugin._redact(result)
        return True, result


def create_plugin() -> GatewayGuardrailPlugin:
    """Factory used by directory/module loaders."""
    return GatewayGuardrailPlugin()
