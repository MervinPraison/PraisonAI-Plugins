"""
OpenTelemetry (OTLP) exporter for the PraisonAI gateway trace-hook seam.

The core gateway defines a dependency-free tracing *seam* -- the
``GatewayTraceHook`` protocol plus the canonical ``GATEWAY_TRACE_STAGES``
stage names -- and resolves it to a zero-cost no-op until an exporter is
attached. This plugin supplies the missing *concrete* exporter: an
``OtelGatewayTraceHook`` that opens/closes a real OpenTelemetry span around
each pipeline stage (inbound -> admit -> agent.run -> llm.call -> tool.call
-> outbox.enqueue -> delivery) and ships them over OTLP to a collector
(Jaeger / Tempo / Grafana / Datadog / Honeycomb).

Design (matches the core seam's own contract and AGENTS.md routing):

  * The heavy ``opentelemetry-sdk`` dependency is an OPTIONAL extra and is
    LAZY-imported. When it is not installed (or fails to import), the hook
    degrades to the exact no-op behaviour of ``NullGatewayTraceHook`` so the
    gateway hot path stays zero-cost and never breaks.
  * ``stage()`` returns a context manager whose scope is the span: entering
    starts it, exiting ends it, and an exception propagating out records the
    error and marks the span failed. The inbound turn's correlation id is
    attached as a span attribute so spans and logs share a key.
  * W3C trace-context propagation is honoured: ``stage(parent_carrier=...)``
    continues an upstream caller's trace, ``inject_context`` writes the active
    context onto outbound (egress) headers, and ``extract_carrier`` normalises
    an inbound header mapping into a parent carrier.

Configuration (environment, so an operator can enable it without code):

  * ``OTEL_EXPORTER_OTLP_ENDPOINT``  the collector endpoint (standard OTel var)
  * ``OTEL_SERVICE_NAME``            service.name resource attr (default
    ``praisonai-gateway``)
  * ``PRAISONAI_OTEL_SAMPLE_RATIO``  parent-based ratio sampler (default 1.0)

These reuse the standard OpenTelemetry environment variables so the plugin
inherits the ecosystem's conventions rather than inventing new ones.
"""
from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any

from praisonaiagents._logging import get_logger

logger = get_logger(__name__)


_DEFAULT_SERVICE_NAME = "praisonai-gateway"


def _sample_ratio() -> float:
    """Parent-based sampling ratio from the environment (default 1.0)."""
    raw = os.environ.get("PRAISONAI_OTEL_SAMPLE_RATIO", "").strip()
    if not raw:
        return 1.0
    try:
        ratio = float(raw)
    except ValueError:
        logger.warning(
            "Invalid PRAISONAI_OTEL_SAMPLE_RATIO=%r; using 1.0", raw
        )
        return 1.0
    # Clamp to [0, 1] so a stray value can never disable/oversample silently.
    return max(0.0, min(1.0, ratio))


class OtelGatewayTraceHook:
    """Concrete ``GatewayTraceHook`` that exports OTLP spans per stage.

    Satisfies the structural ``praisonaiagents.gateway.GatewayTraceHook``
    protocol. Construct once and pass to the gateway; when OpenTelemetry is
    unavailable it transparently behaves as the core no-op hook.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        service_name: str | None = None,
        sample_ratio: float | None = None,
    ) -> None:
        # Resolve config: an *explicit* endpoint arg is the exact signal URL and
        # wins; otherwise leave it unset so ``OTLPSpanExporter`` resolves the
        # standard OTel env vars itself. Passing the general
        # ``OTEL_EXPORTER_OTLP_ENDPOINT`` (e.g. ``http://collector:4318``)
        # straight in as ``endpoint=`` would bypass the exporter's own signal
        # path handling (appending ``/v1/traces``) and override any
        # ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` -- so we don't.
        self._endpoint = endpoint
        self._service_name = (
            service_name
            or os.environ.get("OTEL_SERVICE_NAME", "").strip()
            or _DEFAULT_SERVICE_NAME
        )
        self._sample_ratio = (
            sample_ratio if sample_ratio is not None else _sample_ratio()
        )

        # Lazily-populated OTel handles; ``None`` means "no exporter, act as
        # the null hook". Import failure (SDK not installed) is swallowed so
        # the gateway hot path is never broken by an optional dependency.
        self._tracer = None
        self._propagator = None
        self._trace = None
        self._status_cls = None
        self._status_code = None
        self._enabled = self._init_otel()

    # ------------------------------------------------------------------ setup

    def _init_otel(self) -> bool:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.propagate import get_global_textmap
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.sampling import (
                ParentBased,
                TraceIdRatioBased,
            )
            from opentelemetry.trace import Status, StatusCode
        except Exception as exc:  # noqa: BLE001 - optional dependency
            logger.debug(
                "OpenTelemetry not available (%s); gateway tracing is a no-op. "
                "Install with: pip install praisonai-plugins[otel]",
                exc,
            )
            return False

        try:
            resource = Resource.create({"service.name": self._service_name})
            provider = TracerProvider(
                resource=resource,
                sampler=ParentBased(TraceIdRatioBased(self._sample_ratio)),
            )
            exporter = (
                OTLPSpanExporter(endpoint=self._endpoint)
                if self._endpoint
                else OTLPSpanExporter()
            )
            resolved_endpoint = (
                self._endpoint
                or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
                or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                or "default"
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            self._tracer = provider.get_tracer("praisonai.gateway")
            self._propagator = get_global_textmap()
            self._trace = trace
            self._status_cls = Status
            self._status_code = StatusCode
            logger.info(
                "Gateway OTLP tracing enabled (service=%s endpoint=%s ratio=%s)",
                self._service_name,
                resolved_endpoint,
                self._sample_ratio,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - defensive, never break gateway
            logger.warning(
                "Failed to initialise OpenTelemetry gateway tracer (%s); "
                "tracing disabled.",
                exc,
            )
            return False

    # ------------------------------------------------------------------ seam

    @staticmethod
    @contextmanager
    def _null_scope() -> Iterator[None]:
        yield None

    def stage(
        self,
        name: str,
        *,
        correlation_id: str | None = None,
        parent_carrier: Mapping[str, str] | None = None,
        **attrs: Any,
    ) -> Any:
        """Open a span for pipeline stage ``name`` (no-op when disabled)."""
        if not self._enabled or self._tracer is None:
            return self._null_scope()
        return self._span_scope(
            name,
            correlation_id=correlation_id,
            parent_carrier=parent_carrier,
            attrs=attrs,
        )

    @contextmanager
    def _span_scope(
        self,
        name: str,
        *,
        correlation_id: str | None,
        parent_carrier: Mapping[str, str] | None,
        attrs: Mapping[str, Any],
    ) -> Iterator[Any]:
        # Continue an upstream trace when a W3C carrier is supplied, else start
        # a fresh root span for this stage.
        context = None
        if parent_carrier:
            try:
                context = self._propagator.extract(dict(parent_carrier))
            except Exception as exc:  # noqa: BLE001 - never break the hot path
                logger.debug("trace-context extract failed: %s", exc)
                context = None

        span_cm = self._tracer.start_as_current_span(name, context=context)
        span = span_cm.__enter__()
        try:
            if correlation_id:
                span.set_attribute("praisonai.correlation_id", correlation_id)
            for key, value in attrs.items():
                if value is not None:
                    span.set_attribute(str(key), _coerce_attr(value))
            yield span
        except Exception as exc:
            try:
                span.record_exception(exc)
                span.set_status(self._status_cls(self._status_code.ERROR, str(exc)))
            except Exception as record_exc:  # noqa: BLE001 - defensive
                logger.debug("failed to record span exception: %s", record_exc)
            span_cm.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            span_cm.__exit__(None, None, None)

    def inject_context(self, carrier: MutableMapping[str, str]) -> None:
        """Write the active span context into ``carrier`` (no-op when disabled)."""
        if not self._enabled or self._propagator is None:
            return
        try:
            self._propagator.inject(carrier)
        except Exception as exc:  # noqa: BLE001 - never break egress
            logger.debug("trace-context inject failed: %s", exc)

    def extract_carrier(
        self, carrier: Mapping[str, str]
    ) -> Mapping[str, str] | None:
        """Return a normalized parent carrier from inbound ``carrier``.

        The seam passes the returned mapping straight back to
        ``stage(parent_carrier=...)``; a no-op returning ``None`` when tracing
        is disabled or no usable ``traceparent`` is present.
        """
        if not self._enabled:
            return None
        traceparent = None
        for key in carrier:
            if str(key).lower() == "traceparent":
                traceparent = carrier[key]
                break
        if not traceparent:
            return None
        normalized: dict = {"traceparent": traceparent}
        for key in carrier:
            if str(key).lower() == "tracestate":
                normalized["tracestate"] = carrier[key]
                break
        return normalized


def _coerce_attr(value: Any) -> Any:
    """Coerce a span attribute to an OTel-supported scalar/sequence type."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return str(value)


def create_trace_hook(**kwargs: Any) -> OtelGatewayTraceHook:
    """Factory: build an ``OtelGatewayTraceHook`` (used by loaders/config)."""
    return OtelGatewayTraceHook(**kwargs)
