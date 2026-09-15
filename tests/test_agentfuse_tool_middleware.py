"""Tests for the optional AgentFuse tool middleware."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from dhms_agentfuse import RuntimeGuard, RuntimeGuardDecision, ToolCallRequest
from praisonaiagents import Agent
from praisonaiagents.hooks import InvocationContext, ToolRequest

from praisonai_plugins.guardrails.agentfuse import AgentFuseToolMiddleware


class RecordingRuntimeGuard(RuntimeGuard):
    """Keep completed decisions inside one test instead of production state."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.recorded_decisions: list[RuntimeGuardDecision] = []

    def evaluate(self, tool_call: ToolCallRequest) -> RuntimeGuardDecision:
        decision = super().evaluate(tool_call)
        self.recorded_decisions.append(decision)
        return decision


def _sync_agent(guard: RuntimeGuard):
    handler_calls: list[str] = []

    def protected_write(value: str) -> str:
        handler_calls.append(value)
        return "write completed"

    middleware = AgentFuseToolMiddleware(guard)
    agent = Agent(
        name="agentfuse-sync-test",
        instructions="Exercise one inert test tool.",
        tools=[protected_write],
        hooks=[middleware],
        approval=True,
    )
    return agent, middleware, handler_calls


def test_sync_allow_dispatches_once_and_preserves_identity():
    guard = RecordingRuntimeGuard(allow_tools={"protected_write"})
    agent, _, calls = _sync_agent(guard)

    result = agent.execute_tool(
        "protected_write", {"value": "synthetic-value"}, "praison-allow-001"
    )

    assert result == "write completed"
    assert calls == ["synthetic-value"]
    assert guard.recorded_decisions[0].tool_call_id == "praison-allow-001"


def test_sync_block_returns_non_execution_without_dispatch():
    guard = RecordingRuntimeGuard(deny_tools={"protected_write"})
    agent, _, calls = _sync_agent(guard)

    result = agent.execute_tool(
        "protected_write", {"value": "synthetic-value"}, "praison-block-001"
    )

    assert calls == []
    assert result["status"] == "blocked"
    assert result["tool_failure"] is False
    assert result["tool_call_id"] == "praison-block-001"
    assert result["host_execution"] == {
        "outcome": "not_executed",
        "handler_started": False,
    }
    assert guard.recorded_decisions[0].action == "block"


def test_runtime_guard_policy_error_fails_closed_without_dispatch():
    def failing_policy(tool_call):
        del tool_call
        raise RuntimeError("synthetic policy failure")

    guard = RecordingRuntimeGuard(policy=failing_policy)
    agent, _, calls = _sync_agent(guard)

    result = agent.execute_tool(
        "protected_write", {"value": "synthetic-value"}, "praison-policy-001"
    )

    assert calls == []
    assert result["reason_code"] == "policy_exception"
    assert result["host_execution"]["handler_started"] is False
    assert guard.recorded_decisions[0].reason_code == "policy_exception"


def test_adapter_evaluate_exception_fails_closed_without_dispatch():
    class RaisingGuard(RuntimeGuard):
        def evaluate(self, tool_call):
            del tool_call
            raise RuntimeError("synthetic adapter-boundary failure")

    agent, _, calls = _sync_agent(RaisingGuard())

    result = agent.execute_tool(
        "protected_write", {"value": "synthetic-value"}, "praison-raise-001"
    )

    assert calls == []
    assert result["reason_code"] == "guard_evaluation_exception"
    assert result["guard_failed"] is True
    assert result["policy_denied"] is False
    assert result["tool_call_id"] == "praison-raise-001"
    assert result["host_execution"] == {
        "outcome": "not_executed",
        "handler_started": False,
    }


def test_unknown_decision_action_fails_closed_without_dispatch():
    class UnknownActionGuard(RuntimeGuard):
        """Return a decision whose action violates the AgentFuse contract."""

        def evaluate(self, tool_call):
            del tool_call
            return SimpleNamespace(action="unexpected")

    agent, _, calls = _sync_agent(UnknownActionGuard())

    result = agent.execute_tool(
        "protected_write", {"value": "synthetic-value"}, "praison-invalid-001"
    )

    assert len(calls) == 0
    assert result["status"] == "blocked"
    assert result["reason_code"] == "invalid_guard_decision"
    assert result["policy_denied"] is False
    assert result["guard_failed"] is True
    assert result["tool_call_id"] == "praison-invalid-001"
    assert result["host_execution"] == {
        "outcome": "not_executed",
        "handler_started": False,
    }


@pytest.mark.asyncio
async def test_async_block_uses_same_guard_without_dispatch():
    handler_calls: list[str] = []

    async def protected_write(value: str) -> str:
        handler_calls.append(value)
        return "write completed"

    middleware = AgentFuseToolMiddleware(RuntimeGuard(deny_tools={"protected_write"}))
    agent = Agent(
        name="agentfuse-async-test",
        instructions="Exercise one inert test tool.",
        tools=[protected_write],
        hooks=[middleware],
        approval=True,
    )

    result = await agent.execute_tool_async(
        "protected_write", {"value": "synthetic-value"}, "praison-async-001"
    )

    assert handler_calls == []
    assert result["tool_call_id"] == "praison-async-001"
    assert result["host_execution"]["outcome"] == "not_executed"


def test_missing_identity_fails_closed_without_fabricating_id():
    agent, _, calls = _sync_agent(RuntimeGuard(allow_tools={"protected_write"}))

    result = agent.execute_tool("protected_write", {"value": "synthetic-value"})

    assert calls == []
    assert result["reason_code"] == "missing_tool_call_id"
    assert result["tool_call_id"] is None
    assert result["host_execution"]["handler_started"] is False


def test_handler_failure_after_allow_is_not_converted_to_policy_block():
    middleware = AgentFuseToolMiddleware(RuntimeGuard(allow_tools={"protected_write"}))
    request = ToolRequest(
        tool_name="protected_write",
        arguments={"value": "synthetic-value"},
        context=InvocationContext(
            agent_id="agentfuse-handler-failure",
            run_id="run-001",
            session_id="session-001",
            tool_name="protected_write",
            metadata={"tool_call_id": "praison-handler-failure-001"},
        ),
    )

    def failing_handler(tool_request):
        del tool_request
        raise RuntimeError("synthetic handler failure")

    with pytest.raises(RuntimeError, match="synthetic handler failure"):
        middleware(request, failing_handler)

    assert request.context is not None
    assert request.context.metadata["agentfuse_decision"].action == "allow"
