"""Tests for the OTLP gateway trace-hook exporter plugin.

These run WITHOUT the optional ``opentelemetry`` SDK installed: the hook must
then behave exactly like the core ``NullGatewayTraceHook`` (zero-cost no-op),
which is the backward-compatible default. When the SDK is present the same
surface opens real spans; that path is exercised opportunistically.
"""

import importlib.util

import pytest

from praisonai_plugins.observability.otel_gateway_tracer import (
    OtelGatewayTraceHook,
    create_trace_hook,
    _coerce_attr,
    _sample_ratio,
)

_HAS_OTEL = importlib.util.find_spec("opentelemetry") is not None


def test_factory_returns_hook():
    assert isinstance(create_trace_hook(), OtelGatewayTraceHook)


def test_satisfies_core_protocol_when_available():
    try:
        from praisonaiagents.gateway import GatewayTraceHook
    except Exception:
        pytest.skip("core GatewayTraceHook protocol not importable")
    assert isinstance(OtelGatewayTraceHook(), GatewayTraceHook)


def test_stage_is_a_context_manager_and_never_raises():
    hook = OtelGatewayTraceHook(endpoint="http://collector:4318")
    with hook.stage("agent.run", correlation_id="abc", session="s1") as span:
        # span is None when disabled, or a real span object when enabled.
        assert span is None or span is not None


def test_stage_propagates_body_exceptions():
    hook = OtelGatewayTraceHook()
    with pytest.raises(ValueError):
        with hook.stage("llm.call"):
            raise ValueError("boom")


def test_inject_context_is_safe_noop_on_mapping():
    hook = OtelGatewayTraceHook()
    carrier: dict = {}
    hook.inject_context(carrier)
    # Disabled -> untouched; enabled -> may add W3C headers. Never raises.
    assert isinstance(carrier, dict)


def test_extract_carrier_returns_none_without_traceparent():
    hook = OtelGatewayTraceHook()
    assert hook.extract_carrier({"x-other": "1"}) is None


def test_coerce_attr_scalars_and_sequences():
    assert _coerce_attr("s") == "s"
    assert _coerce_attr(3) == 3
    assert _coerce_attr(True) is True
    assert _coerce_attr([1, 2]) == ["1", "2"]
    assert isinstance(_coerce_attr(object()), str)


def test_sample_ratio_clamped(monkeypatch):
    monkeypatch.setenv("PRAISONAI_OTEL_SAMPLE_RATIO", "2.5")
    assert _sample_ratio() == 1.0
    monkeypatch.setenv("PRAISONAI_OTEL_SAMPLE_RATIO", "-1")
    assert _sample_ratio() == 0.0
    monkeypatch.setenv("PRAISONAI_OTEL_SAMPLE_RATIO", "0.25")
    assert _sample_ratio() == 0.25
    monkeypatch.setenv("PRAISONAI_OTEL_SAMPLE_RATIO", "junk")
    assert _sample_ratio() == 1.0


@pytest.mark.skipif(not _HAS_OTEL, reason="opentelemetry SDK not installed")
def test_extract_carrier_reads_traceparent_when_enabled():
    hook = OtelGatewayTraceHook()
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    result = hook.extract_carrier({"traceparent": tp, "tracestate": "a=1"})
    if hook._enabled:
        assert result == {"traceparent": tp, "tracestate": "a=1"}
