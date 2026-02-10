"""
Unit tests for PortalConnectSkill (UAMP WS daemon connection).
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill


@pytest.fixture
def skill_config():
    return {
        "portal_ws_url": "wss://roborum.test/ws",
        "agents": [
            {"name": "agent-a", "token": "jwt-a"},
            {"name": "agent-b", "token": "jwt-b"},
        ],
        "auto_reconnect": False,
        "max_reconnect_attempts": 0,
    }


class TestPortalConnectInit:
    def test_config_defaults(self):
        skill = PortalConnectSkill()
        assert skill.portal_ws_url == "wss://roborum.ai/ws"
        assert skill.agents == []
        assert skill.auto_reconnect is True

    def test_config_from_dict(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        assert skill.portal_ws_url == "wss://roborum.test/ws"
        assert len(skill.agents) == 2
        assert skill.agents[0]["name"] == "agent-a"
        assert skill.auto_reconnect is False

    def test_set_agent_resolver(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        resolver = MagicMock()
        skill.set_agent_resolver(resolver)
        assert skill._agent_resolver is resolver


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_session_created_registers_agent(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        skill._ws = AsyncMock()
        skill._connected = True

        # Simulate session.created event
        message = json.dumps({
            "type": "session.created",
            "event_id": "evt_1",
            "session_id": "sess_123",
            "session": {"agent": "agent-a"},
        })
        await skill._handle_message(message)
        assert skill._session_by_id.get("sess_123") == "agent-a"

    @pytest.mark.asyncio
    async def test_session_end_unregisters(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        skill._ws = AsyncMock()
        skill._connected = True
        skill._session_by_id["sess_123"] = "agent-a"

        message = json.dumps({
            "type": "session.end",
            "event_id": "evt_2",
            "session_id": "sess_123",
        })
        await skill._handle_message(message)
        assert "sess_123" not in skill._session_by_id


class TestSessionCreate:
    @pytest.mark.asyncio
    async def test_sends_session_create_event(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        mock_ws = AsyncMock()
        skill._ws = mock_ws
        skill._connected = True

        await skill._send_session_create("agent-a", "jwt-a")

        mock_ws.send.assert_called_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "session.create"
        assert sent["session"]["agent"] == "agent-a"
        assert sent["session"]["token"] == "jwt-a"

    @pytest.mark.asyncio
    async def test_skips_when_no_ws(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        skill._ws = None

        await skill._send_session_create("agent-a", "jwt-a")
        # Should not raise

    @pytest.mark.asyncio
    async def test_skips_when_empty_name(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        mock_ws = AsyncMock()
        skill._ws = mock_ws
        skill._connected = True

        await skill._send_session_create("", "jwt-a")
        mock_ws.send.assert_not_called()


class TestInputTextHandling:
    @pytest.mark.asyncio
    async def test_input_text_runs_agent_and_sends_response(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        mock_ws = AsyncMock()
        skill._ws = mock_ws
        skill._connected = True
        skill._session_by_id["sess_123"] = "agent-a"

        # Mock agent with run_streaming
        mock_agent = MagicMock()

        async def mock_streaming(messages, tools=None):
            yield {"content": "Hello "}
            yield {"content": "world!"}

        mock_agent.run_streaming = mock_streaming
        mock_agent.name = "agent-a"
        skill.set_agent_resolver(lambda name: mock_agent if name == "agent-a" else None)

        data = {"type": "input.text", "text": "Hi!", "session_id": "sess_123", "event_id": "evt_3"}
        await skill._handle_input_text(data, "sess_123")

        # Should have sent 2 response.delta + 1 response.done = 3 calls
        assert mock_ws.send.call_count == 3
        calls = [json.loads(c[0][0]) for c in mock_ws.send.call_args_list]
        delta_types = [c["type"] for c in calls if c["type"] == "response.delta"]
        done_types = [c["type"] for c in calls if c["type"] == "response.done"]
        assert len(delta_types) == 2
        assert len(done_types) == 1

        # Check delta content
        deltas = [c for c in calls if c["type"] == "response.delta"]
        assert deltas[0]["delta"]["text"] == "Hello "
        assert deltas[1]["delta"]["text"] == "world!"

    @pytest.mark.asyncio
    async def test_input_text_no_session_id(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        mock_ws = AsyncMock()
        skill._ws = mock_ws
        skill._connected = True

        data = {"type": "input.text", "text": "Hi!", "event_id": "evt_4"}
        await skill._handle_input_text(data, None)
        mock_ws.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_input_text_unknown_session(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        mock_ws = AsyncMock()
        skill._ws = mock_ws
        skill._connected = True

        data = {"type": "input.text", "text": "Hi!", "session_id": "unknown", "event_id": "evt_5"}
        await skill._handle_input_text(data, "unknown")
        mock_ws.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_input_text_agent_error_sends_response_error(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        mock_ws = AsyncMock()
        skill._ws = mock_ws
        skill._connected = True
        skill._session_by_id["sess_err"] = "agent-a"

        mock_agent = MagicMock()

        async def mock_streaming_error(messages, tools=None):
            raise RuntimeError("Agent crashed")
            yield  # noqa: unreachable - makes this a generator

        mock_agent.run_streaming = mock_streaming_error
        mock_agent.name = "agent-a"
        skill.set_agent_resolver(lambda name: mock_agent)

        data = {"type": "input.text", "text": "fail", "session_id": "sess_err", "event_id": "evt_6"}
        await skill._handle_input_text(data, "sess_err")

        # Should send response.error
        assert mock_ws.send.call_count == 1
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "response.error"
        assert "Agent crashed" in sent["error"]["message"]


class TestResponseEvents:
    @pytest.mark.asyncio
    async def test_send_response_delta(self, skill_config):
        from webagents.uamp.types import ContentDelta

        skill = PortalConnectSkill(skill_config)
        mock_ws = AsyncMock()
        skill._ws = mock_ws
        skill._connected = True

        delta = ContentDelta(type="text", text="Hello")
        await skill._send_response_delta("sess_1", "resp_1", delta)

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "response.delta"
        assert sent["session_id"] == "sess_1"
        assert sent["response_id"] == "resp_1"
        assert sent["delta"]["type"] == "text"
        assert sent["delta"]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_send_response_done(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        mock_ws = AsyncMock()
        skill._ws = mock_ws
        skill._connected = True

        await skill._send_response_done("sess_1", "resp_1")

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "response.done"
        assert sent["session_id"] == "sess_1"
        assert sent["response_id"] == "resp_1"

    @pytest.mark.asyncio
    async def test_send_response_error(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        mock_ws = AsyncMock()
        skill._ws = mock_ws
        skill._connected = True

        await skill._send_response_error("sess_1", "resp_1", "Something failed")

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "response.error"
        assert sent["error"]["code"] == "agent_error"
        assert sent["error"]["message"] == "Something failed"


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        mock_ws = AsyncMock()
        skill._ws = mock_ws
        skill._connected = True
        skill._session_by_id["sess_1"] = "agent-a"

        await skill.disconnect()

        assert skill._shutdown is True
        assert skill._connected is False
        assert skill._ws is None
        assert len(skill._session_by_id) == 0
        mock_ws.close.assert_called_once()

    def test_is_connected_property(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        assert skill.is_connected is False
        skill._connected = True
        assert skill.is_connected is True


class TestMultiplexing:
    @pytest.mark.asyncio
    async def test_multiple_sessions_registered(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        skill._ws = AsyncMock()
        skill._connected = True

        # Register two sessions
        await skill._handle_message(json.dumps({
            "type": "session.created",
            "session_id": "sess_a",
            "session": {"agent": "agent-a"},
        }))
        await skill._handle_message(json.dumps({
            "type": "session.created",
            "session_id": "sess_b",
            "session": {"agent": "agent-b"},
        }))

        assert skill._session_by_id["sess_a"] == "agent-a"
        assert skill._session_by_id["sess_b"] == "agent-b"

    @pytest.mark.asyncio
    async def test_input_routes_to_correct_agent(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        mock_ws = AsyncMock()
        skill._ws = mock_ws
        skill._connected = True
        skill._session_by_id["sess_a"] = "agent-a"
        skill._session_by_id["sess_b"] = "agent-b"

        resolved_names = []

        def resolver(name):
            resolved_names.append(name)
            agent = MagicMock()
            agent.name = name

            async def mock_streaming(messages, tools=None):
                yield {"content": f"I am {name}"}

            agent.run_streaming = mock_streaming
            return agent

        skill.set_agent_resolver(resolver)

        await skill._handle_input_text(
            {"type": "input.text", "text": "Hello", "session_id": "sess_b"},
            "sess_b",
        )

        assert resolved_names == ["agent-b"]
        # Should have sent response.delta + response.done
        assert mock_ws.send.call_count == 2
        calls = [json.loads(c[0][0]) for c in mock_ws.send.call_args_list]
        assert calls[0]["type"] == "response.delta"
        assert calls[0]["session_id"] == "sess_b"
        assert calls[0]["delta"]["text"] == "I am agent-b"
        assert calls[1]["type"] == "response.done"
        assert calls[1]["session_id"] == "sess_b"
