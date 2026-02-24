"""
Tests for cancel propagation and tool call streaming in the agent pipeline.

Covers:
- asyncio.CancelledError handling in BaseAgent.run_streaming
- Tool call/result delta emission from run_streaming
- UAMP transport CancelledError handling in _generate_response
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass


@dataclass
class _FakeHandoff:
    target: str = "mock_handoff"
    description: str = "mock"
    scope: str = "all"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def _make_agent(name):
    """Create a BaseAgent with enough internals wired for run_streaming."""
    from webagents.agents.core.base_agent import BaseAgent

    agent = BaseAgent(name=name, model="mock")

    handoff = _FakeHandoff(target="mock_handoff", metadata={"function": None, "is_generator": True})
    agent.active_handoff = handoff
    agent._registered_handoffs = [{"config": handoff}]
    agent._ensure_skills_initialized = AsyncMock()
    agent._enhance_messages_with_prompts = AsyncMock(
        return_value=[{"role": "user", "content": "test"}]
    )
    return agent


class TestBaseAgentCancelHandling:
    """Test that CancelledError triggers finalization hooks"""

    @pytest.mark.asyncio
    async def test_cancelled_error_runs_finalization(self):
        """When run_streaming is cancelled, finalize_connection hooks still run"""
        agent = _make_agent("test-cancel")

        finalization_called = False
        original_hooks = agent._execute_hooks

        async def mock_execute_hooks(hook_name, context):
            nonlocal finalization_called
            if hook_name == "finalize_connection":
                finalization_called = True
            return context

        agent._execute_hooks = mock_execute_hooks

        async def _slow_gen():
            await asyncio.sleep(100)
            yield "should not reach"

        def slow_handoff(handoff_config, messages, **kwargs):
            return _slow_gen()

        agent._execute_handoff = slow_handoff

        mock_context = MagicMock()
        mock_context.messages = [{"role": "user", "content": "hi"}]
        mock_context.get = MagicMock(return_value=None)
        mock_context.set = MagicMock()
        mock_context.stream = True
        mock_context.agent = agent

        with patch(
            "webagents.agents.core.base_agent.get_context", return_value=mock_context
        ):
            task = asyncio.create_task(
                self._drain_generator(
                    agent.run_streaming([{"role": "user", "content": "hi"}])
                )
            )
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert finalization_called

    @staticmethod
    async def _drain_generator(gen):
        async for _ in gen:
            pass


class TestToolCallDeltaForwarding:
    """Test that UAMP transport forwards tool_call/tool_result deltas via WebSocket"""

    @pytest.mark.asyncio
    async def test_tool_call_and_result_sent_as_ws_events(self):
        """_generate_response sends tool_call/tool_result deltas as WebSocket events"""
        from webagents.agents.skills.core.transport.uamp.skill import (
            UAMPTransportSkill,
        )

        skill = UAMPTransportSkill({})
        skill.logger = MagicMock()

        mock_ws = AsyncMock()

        mock_session = MagicMock()
        mock_session.id = "sess_tc"
        mock_session.payment_token = None
        mock_session.payment_balance = "1.00"
        mock_session.payment_currency = "USD"
        mock_session.conversation = []

        tool_call_chunk = {
            "type": "tool_call",
            "call_id": "tc_123",
            "name": "test_tool",
            "arguments": '{"key":"value"}',
        }
        tool_result_chunk = {
            "type": "tool_result",
            "id": "tc_123",
            "result": "tool output",
            "status": "success",
        }
        done_chunk = "Final answer"

        async def mock_handoff(messages):
            yield tool_call_chunk
            yield tool_result_chunk
            yield done_chunk

        skill.execute_handoff = mock_handoff
        skill._build_messages = MagicMock(return_value=[])
        skill.get_context = MagicMock(return_value=None)

        await skill._generate_response(mock_ws, mock_session, session_id="s1")

        sent_events = [call[0][0] for call in mock_ws.send_json.call_args_list]

        tc_events = [
            e for e in sent_events
            if isinstance(e, dict)
            and e.get("delta", {}).get("type") == "tool_call"
        ]
        tr_events = [
            e for e in sent_events
            if isinstance(e, dict)
            and e.get("delta", {}).get("type") == "tool_result"
        ]

        assert len(tc_events) >= 1, f"Expected tool_call event, got: {sent_events}"
        assert tc_events[0]["delta"]["tool_call"]["name"] == "test_tool"
        assert len(tr_events) >= 1, f"Expected tool_result event, got: {sent_events}"


class TestUAMPCancelledResponse:
    """Test UAMP transport handles CancelledError in _generate_response"""

    @pytest.mark.asyncio
    async def test_cancelled_error_does_not_send_error_event(self):
        """_generate_response catches CancelledError without sending error event"""
        from webagents.agents.skills.core.transport.uamp.skill import (
            UAMPTransportSkill,
        )

        skill = UAMPTransportSkill({})
        skill.logger = MagicMock()

        mock_ws = AsyncMock()

        mock_session = MagicMock()
        mock_session.id = "sess_1"
        mock_session.payment_token = None
        mock_session.payment_balance = "1.00"
        mock_session.payment_currency = "USD"
        mock_session.conversation = []

        async def slow_handoff(messages):
            await asyncio.sleep(100)
            yield "unreachable"

        skill.execute_handoff = slow_handoff
        skill._build_messages = MagicMock(return_value=[])
        skill.get_context = MagicMock(return_value=None)

        task = asyncio.create_task(
            skill._generate_response(mock_ws, mock_session, session_id="s1")
        )
        await asyncio.sleep(0.05)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        error_calls = [
            call
            for call in mock_ws.send_json.call_args_list
            if isinstance(call[0][0], dict)
            and call[0][0].get("type") == "response.error"
        ]
        assert len(error_calls) == 0
