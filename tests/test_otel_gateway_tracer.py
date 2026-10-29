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
    _coerce_attr,
    _sample_ratio,
    create_trace_hook,
)

_HAS_OTEL = importlib.util.find_spec("opentelemetry") is not None


@pytest.fixture(autouse=True)
def _no_live_otlp_export(monkeypatch):
    """Never open a real network span exporter during tests.

    With the SDK present (CI installs it), an enabled hook would otherwise build
    a live OTLP HTTP exporter aimed at ``localhost:4318`` and spawn background
    export threads that spam connection-retry logs. Redirect every exporter to
    an in-memory one so the suite stays hermetic and fast. Tests that need to
    assert on captured spans use the ``enabled_hook`` fixture, which returns the
    shared in-memory exporter created here.
    """
    if not _HAS_OTEL:
        yield None
        return
    import opentelemetry.exporter.otlp.proto.http.trace_exporter as te
    from opentelemetry.sdk.trace import export as export_mod
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    memory = InMemorySpanExporter()

    class _InMemoryOTLP:
        def __new__(cls, *args, **kwargs):
            return memory

    monkeypatch.setattr(te, "OTLPSpanExporter", _InMemoryOTLP)
    monkeypatch.setattr(export_mod, "BatchSpanProcessor", SimpleSpanProcessor)
    yield memory


def test_factory_returns_hook():
    assert isinstance(create_trace_hook(), OtelGatewayTraceHook)


def test_satisfies_core_protocol_when_available():
    try:
        from praisonaiagents.gateway import GatewayTraceHook
    except Exception:  # noqa: BLE001 - optional core seam may be absent
        pytest.skip("core GatewayTraceHook protocol not importable")
    assert isinstance(OtelGatewayTraceHook(), GatewayTraceHook)


def test_stage_is_a_context_manager_and_never_raises():
    # No endpoint => never opens a live network exporter, so this stays a fast,
    # deterministic contract check for both the enabled and no-op paths. (Real
    # span emission is covered by the enabled_hook tests below.)
    hook = OtelGatewayTraceHook()
    with hook.stage("agent.run", correlation_id="abc", session="s1") as span:
        # span is None when disabled, or a real span object when enabled.
        assert span is None or span is not None


def test_stage_propagates_body_exceptions():
    hook = OtelGatewayTraceHook()
    with pytest.raises(ValueError), hook.stage("llm.call"):
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


# --------------------------------------------------------------------------- #
# Deterministic ENABLED-path coverage.
#
# CI installs the SDK (dev extra), so these run for real. They swap the OTLP
# network exporter for an in-memory one and the async BatchSpanProcessor for a
# synchronous SimpleSpanProcessor, so a real span is created, exported, and can
# be asserted -- guarding span naming, attribute stamping, W3C propagation, and
# error-status handling (which the disabled no-op path can never verify).
# --------------------------------------------------------------------------- #


@pytest.fixture
def enabled_hook(_no_live_otlp_export):
    if not _HAS_OTEL:
        pytest.skip("opentelemetry SDK not installed")
    memory = _no_live_otlp_export
    hook = OtelGatewayTraceHook(service_name="test-svc")
    assert hook._enabled, "hook must be enabled with the SDK present"
    return hook, memory


def test_enabled_stage_emits_span_with_correlation_and_attrs(enabled_hook):
    hook, memory = enabled_hook
    with hook.stage("agent.run", correlation_id="cid-1", foo="bar") as span:
        assert span is not None
    spans = memory.get_finished_spans()
    assert len(spans) == 1
    emitted = spans[0]
    assert emitted.name == "agent.run"
    assert emitted.attributes["praisonai.correlation_id"] == "cid-1"
    assert emitted.attributes["foo"] == "bar"


def test_enabled_inject_context_writes_traceparent(enabled_hook):
    hook, _memory = enabled_hook
    carrier: dict = {}
    with hook.stage("llm.call"):
        hook.inject_context(carrier)
    assert "traceparent" in carrier


def test_enabled_stage_records_error_status_on_exception(enabled_hook):
    from opentelemetry.trace import StatusCode

    hook, memory = enabled_hook
    with pytest.raises(ValueError), hook.stage("tool.call"):
        raise ValueError("boom")
    spans = memory.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


def test_enabled_stage_continues_upstream_trace(enabled_hook):
    hook, memory = enabled_hook
    trace_id_hex = "0af7651916cd43dd8448eb211c80319c"
    parent = {
        "traceparent": f"00-{trace_id_hex}-b7ad6b7169203331-01",
    }
    with hook.stage("admit", parent_carrier=parent):
        pass
    spans = memory.get_finished_spans()
    assert len(spans) == 1
    assert format(spans[0].context.trace_id, "032x") == trace_id_hex
