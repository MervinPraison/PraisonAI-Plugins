"""Optional AgentFuse middleware for pre-dispatch tool decisions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from praisonaiagents.hooks import ToolRequest, ToolResponse, wrap_tool_call


class AgentFuseToolMiddleware:
    """Apply an AgentFuse decision at PraisonAI's public tool boundary."""

    def __init__(self, guard: Any) -> None:
        try:
            from dhms_agentfuse import RuntimeGuard, ToolCallRequest
        except ImportError as exc:
            raise ImportError(
                "AgentFuse middleware requires the 'agentfuse' optional dependency: "
                "pip install 'praisonai-plugins[agentfuse]'"
            ) from exc

        if not isinstance(guard, RuntimeGuard):
            raise TypeError("guard must be a dhms_agentfuse.RuntimeGuard")

        self._guard = guard
        self._request_type = ToolCallRequest
        wrap_tool_call(self)

    def _not_executed_response(
        self,
        request: ToolRequest,
        *,
        tool_call_id: str | None,
        reason_code: str,
        policy_denied: bool,
        guard_failed: bool,
        decision: Any | None = None,
    ) -> ToolResponse:
        result = {
            "status": "blocked",
            "policy_denied": policy_denied,
            "guard_failed": guard_failed,
            "tool_failure": False,
            "reason_code": reason_code,
            "tool_call_id": tool_call_id,
            "host_execution": {
                "outcome": "not_executed",
                "handler_started": False,
            },
        }
        if decision is not None:
            result["agentfuse_decision"] = decision.to_safe_dict()

        return ToolResponse(
            tool_name=request.tool_name,
            result=result,
            context=request.context,
        )

    def __call__(
        self,
        request: ToolRequest,
        call_next: Callable[[ToolRequest], ToolResponse],
    ) -> ToolResponse:
        context = request.context
        metadata = context.metadata if context is not None else {}
        tool_call_id = metadata.get("tool_call_id")
        if not tool_call_id:
            return self._not_executed_response(
                request,
                tool_call_id=None,
                reason_code="missing_tool_call_id",
                policy_denied=False,
                guard_failed=False,
            )

        try:
            decision = self._guard.evaluate(
                self._request_type(
                    tool_call_id=tool_call_id,
                    tool_name=request.tool_name,
                    arguments=request.arguments,
                    safe_metadata={"integration": "praisonai-plugins"},
                )
            )
        except Exception:  # noqa: BLE001 - the adapter boundary must fail closed
            return self._not_executed_response(
                request,
                tool_call_id=tool_call_id,
                reason_code="guard_evaluation_exception",
                policy_denied=False,
                guard_failed=True,
            )

        action = getattr(decision, "action", None)
        if action not in ("allow", "block"):
            return self._not_executed_response(
                request,
                tool_call_id=tool_call_id,
                reason_code="invalid_guard_decision",
                policy_denied=False,
                guard_failed=True,
            )

        metadata["agentfuse_decision"] = decision
        if action == "block":
            return self._not_executed_response(
                request,
                tool_call_id=tool_call_id,
                reason_code=decision.reason_code,
                policy_denied=True,
                guard_failed=False,
                decision=decision,
            )

        return call_next(request)
