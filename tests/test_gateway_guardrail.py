"""Tests for the GatewayGuardrailPlugin.

Covers inbound admission/filter (``before_message``), outbound on-channel
redaction (``after_message``) including the per-channel policy. Uses the real ``praisonaiagents`` plugin surface.
"""

import importlib

import pytest
from praisonaiagents.plugins.plugin import PluginDecision, PluginType

from praisonai_plugins.policies.gateway_guardrail import GatewayGuardrailPlugin


@pytest.fixture
def plugin(monkeypatch):
    # Clear env so each test configures its own policy deterministically.
    for var in (
        "PRAISONAI_GATEWAY_DENIED_SENDERS",
        "PRAISONAI_GATEWAY_BLOCKED_TERMS",
        "PRAISONAI_GATEWAY_MAX_INBOUND",
    ):
        monkeypatch.delenv(var, raising=False)
    return GatewayGuardrailPlugin()


def test_info_is_policy_with_message_hooks(plugin):
    info = plugin.info
    assert info.name == "gateway_guardrail"
    assert info.plugin_type == PluginType.POLICY
    values = {h.value for h in info.hooks}
    assert "message_received" in values
    assert "message_sending" in values


def test_before_message_allows_normal_traffic(plugin):
    assert plugin.before_message({"sender_id": "alice", "content": "hello"}) is None


def test_before_message_denies_blocked_sender(monkeypatch):
    monkeypatch.setenv("PRAISONAI_GATEWAY_DENIED_SENDERS", "mallory,eve")
    p = GatewayGuardrailPlugin()
    decision = p.before_message({"sender_id": "Mallory", "content": "hi"})
    assert isinstance(decision, PluginDecision)
    assert decision.is_denied()


def test_before_message_denies_blocked_term(monkeypatch):
    monkeypatch.setenv("PRAISONAI_GATEWAY_BLOCKED_TERMS", "secretproject")
    p = GatewayGuardrailPlugin()
    decision = p.before_message({"sender_id": "bob", "content": "the SecretProject plan"})
    assert isinstance(decision, PluginDecision) and decision.is_denied()


def test_before_message_denies_oversized(monkeypatch):
    monkeypatch.setenv("PRAISONAI_GATEWAY_MAX_INBOUND", "10")
    p = GatewayGuardrailPlugin()
    decision = p.before_message({"sender_id": "bob", "content": "x" * 50})
    assert isinstance(decision, PluginDecision) and decision.is_denied()


def test_after_message_redacts_ssn(plugin):
    msg = {"content": "my ssn is 123-45-6789 ok"}
    out = plugin.after_message(msg)
    assert "123-45-6789" not in out["content"]
    assert "[REDACTED]" in out["content"]
    assert msg["content"] == "my ssn is 123-45-6789 ok"  # caller's dict untouched


def test_after_message_redacts_secret_token(plugin):
    out = plugin.after_message({"content": "token sk-abcdefghijklmnop1234 leaked"})
    assert "sk-abcdefghijklmnop1234" not in out["content"]


def test_after_message_noop_when_clean(plugin):
    msg = {"content": "nothing sensitive here"}
    assert plugin.after_message(msg)["content"] == "nothing sensitive here"


def test_per_channel_policy_phone_only_public(plugin):
    phone_text = "call +1 415 555 2671 now"
    dm = plugin.after_message({"content": phone_text, "channel_type": "dm"})
    public = plugin.after_message({"content": phone_text, "channel_type": "public"})
    # Phone only redacted on public channels (stricter policy).
    assert "+1 415 555 2671" in dm["content"]
    assert "+1 415 555 2671" not in public["content"]


def test_entry_point_registered():
    # Ensure the module import path used by pyproject resolves.
    mod = importlib.import_module("praisonai_plugins.policies.gateway_guardrail")
    assert mod.GatewayGuardrailPlugin().info.name == "gateway_guardrail"


def test_public_policy_applies_when_outbound_lacks_channel_type(plugin):
    # Core's MessageSendingInput has channel_id but no channel_type.
    plugin.before_message({"sender_id": "bob", "content": "hi", "channel_id": "C1", "channel_type": "group"})
    out = plugin.after_message({"content": "call +1 415 555 2671 now", "channel_id": "C1"})
    assert out["content"] == "call [REDACTED] now"
    dm = plugin.after_message({"content": "call +1 415 555 2671 now", "channel_id": "D9"})
    assert "+1 415 555 2671" in dm["content"]


def test_after_message_redacts_project_scoped_key(plugin):
    out = plugin.after_message({"content": "key sk-proj-abc_DEF-1234567890xyz end"})
    assert out["content"] == "key [REDACTED] end"
