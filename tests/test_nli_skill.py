"""
Tests for NLISkill - Natural Language Interface for Agent-to-Agent Communication

Tests NLI functionality including agent resolution, HTTP communication,
error handling, and statistics tracking.
"""

import pytest
try:
    import robutler
    HAS_ROBUTLER = True
except ImportError:
    HAS_ROBUTLER = False

if not HAS_ROBUTLER:
    pytest.skip("robutler not installed", allow_module_level=True)

import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from webagents.agents.skills.robutler.nli import NLISkill, NLICommunication


class MockAgent:
    """Mock agent for testing"""
    def __init__(self, name="test-agent"):
        self.name = name
        self.api_key = "test_api_key"


class MockStreamResponse:
    """Mock HTTP streaming response for SSE"""
    def __init__(self, status_code=200, lines=None, text=""):
        self.status_code = status_code
        self.text = text
        self._lines = lines or []
    
    async def aiter_lines(self):
        for line in self._lines:
            yield line


@pytest.fixture
def nli_skill():
    """NLI skill with default configuration"""
    config = {
        'timeout': 10.0,
        'max_retries': 1,
        'default_authorization': 0.05,
        'max_authorization': 1.0,
    }
    return NLISkill(config)


@pytest.fixture
async def initialized_nli_skill(nli_skill):
    """Initialized NLI skill with mock agent"""
    mock_agent = MockAgent()
    await nli_skill.initialize(mock_agent)
    yield nli_skill
    await nli_skill.cleanup()


class TestAgentResolution:
    """Test agent identifier resolution to URLs"""
    
    def test_resolve_at_username(self, nli_skill):
        """@username resolves to agent daemon URL"""
        url = nli_skill._resolve_agent_to_url("@r-banana")
        assert url == "http://localhost:2224/agents/r-banana/chat/completions"
    
    def test_resolve_plain_name(self, nli_skill):
        """Plain name resolves to agent daemon URL"""
        url = nli_skill._resolve_agent_to_url("assistant")
        assert url == "http://localhost:2224/agents/assistant/chat/completions"
    
    def test_resolve_at_username_strips_at(self, nli_skill):
        """@ prefix is stripped from username"""
        url = nli_skill._resolve_agent_to_url("@my-agent")
        assert "/agents/my-agent/" in url
    
    def test_resolve_public_url(self, nli_skill):
        """Public URLs are passed through with /chat/completions appended"""
        url = nli_skill._resolve_agent_to_url("https://example.com/agents/foo")
        assert url == "https://example.com/agents/foo/chat/completions"
    
    def test_resolve_public_url_already_complete(self, nli_skill):
        """URLs already ending in /chat/completions are unchanged"""
        url = nli_skill._resolve_agent_to_url("https://example.com/agents/foo/chat/completions")
        assert url == "https://example.com/agents/foo/chat/completions"
    
    def test_reject_localhost_url(self, nli_skill):
        """Internal localhost URLs are rejected"""
        with pytest.raises(ValueError, match="Internal URLs are not allowed"):
            nli_skill._resolve_agent_to_url("http://localhost:2224/agents/foo")
    
    def test_reject_private_ip_url(self, nli_skill):
        """Private IP URLs are rejected"""
        with pytest.raises(ValueError, match="Internal URLs are not allowed"):
            nli_skill._resolve_agent_to_url("http://192.168.1.100/agents/foo")
    
    def test_reject_127001_url(self, nli_skill):
        """127.0.0.1 URLs are rejected"""
        with pytest.raises(ValueError, match="Internal URLs are not allowed"):
            nli_skill._resolve_agent_to_url("http://127.0.0.1:8080/agents/foo")
    
    def test_custom_base_url(self):
        """Custom agent_base_url is used for name resolution"""
        skill = NLISkill({'agent_base_url': 'http://custom:9999'})
        url = skill._resolve_agent_to_url("@assistant")
        assert url == "http://custom:9999/agents/assistant/chat/completions"
    
    def test_whitespace_handling(self, nli_skill):
        """Leading/trailing whitespace is stripped"""
        url = nli_skill._resolve_agent_to_url("  @assistant  ")
        assert "/agents/assistant/" in url


class TestNLISkillInitialization:
    """Test NLI skill initialization and configuration"""
    
    def test_default_configuration(self):
        skill = NLISkill()
        assert skill.default_timeout == 600.0
        assert skill.max_retries == 2
        assert skill.default_authorization == 0.10
        assert skill.max_authorization == 5.00
    
    def test_custom_configuration(self):
        config = {
            'timeout': 15.0,
            'max_retries': 3,
            'default_authorization': 0.20,
            'max_authorization': 2.50
        }
        skill = NLISkill(config)
        assert skill.default_timeout == 15.0
        assert skill.max_retries == 3
        assert skill.default_authorization == 0.20
        assert skill.max_authorization == 2.50
    
    @pytest.mark.asyncio
    async def test_initialization(self, nli_skill):
        mock_agent = MockAgent()
        with patch('webagents.agents.skills.robutler.nli.skill.HTTPX_AVAILABLE', True):
            with patch('webagents.agents.skills.robutler.nli.skill.httpx.AsyncClient') as mock_cls:
                mock_client = AsyncMock()
                mock_cls.return_value = mock_client
                await nli_skill.initialize(mock_agent)
                assert nli_skill.agent == mock_agent
                assert nli_skill.http_client == mock_client
                await nli_skill.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialization_no_httpx(self, nli_skill):
        mock_agent = MockAgent()
        with patch('webagents.agents.skills.robutler.nli.skill.HTTPX_AVAILABLE', False):
            await nli_skill.initialize(mock_agent)
            assert nli_skill.agent == mock_agent
            assert nli_skill.http_client is None


class TestNLICommunicationTool:
    """Test the nli_tool function"""
    
    @pytest.mark.asyncio
    async def test_self_call_prevention(self, initialized_nli_skill):
        """Agent cannot call itself via NLI"""
        skill = initialized_nli_skill
        result = await skill.nli_tool(agent="@test-agent", message="hello")
        assert "Cannot use nli_tool to call yourself" in result
    
    @pytest.mark.asyncio
    async def test_empty_agent(self, initialized_nli_skill):
        result = await initialized_nli_skill.nli_tool(agent="", message="hello")
        assert "provide an agent identifier" in result
    
    @pytest.mark.asyncio
    async def test_empty_message(self, initialized_nli_skill):
        result = await initialized_nli_skill.nli_tool(agent="@other", message="")
        assert "provide a message" in result
    
    @pytest.mark.asyncio
    async def test_authorization_limit_exceeded(self, initialized_nli_skill):
        skill = initialized_nli_skill
        result = await skill.nli_tool(
            agent="@other-agent",
            message="test",
            authorized_amount=10.0  # Exceeds max_authorization of 1.0
        )
        assert "exceeds maximum allowed" in result
    
    @pytest.mark.asyncio
    async def test_internal_url_rejected(self, initialized_nli_skill):
        """Internal URLs passed directly by the LLM are rejected"""
        skill = initialized_nli_skill
        result = await skill.nli_tool(
            agent="http://localhost:2224/agents/foo",
            message="test"
        )
        assert "Internal URLs are not allowed" in result
    
    @pytest.mark.asyncio
    async def test_successful_communication(self, initialized_nli_skill):
        skill = initialized_nli_skill
        
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: [DONE]',
        ]
        mock_response = MockStreamResponse(status_code=200, lines=sse_lines)
        
        with patch.object(skill.http_client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await skill.nli_tool(agent="@other-agent", message="Hi there")
            
            assert result == "Hello world"
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            assert call_url == "http://localhost:2224/agents/other-agent/chat/completions"
            
            assert len(skill.communication_history) == 1
            comm = skill.communication_history[0]
            assert comm.success is True
            assert comm.target_agent == "@other-agent"
    
    @pytest.mark.asyncio
    async def test_http_error(self, initialized_nli_skill):
        skill = initialized_nli_skill
        
        mock_response = MockStreamResponse(status_code=500, text="Server Error")
        mock_response.aiter_lines = AsyncMock(return_value=iter([]))
        
        with patch.object(skill.http_client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await skill.nli_tool(agent="@broken", message="test")
            
            assert "Failed to communicate" in result
            assert len(skill.communication_history) == 1
            assert skill.communication_history[0].success is False
    
    @pytest.mark.asyncio
    async def test_no_httpx(self):
        skill = NLISkill()
        with patch('webagents.agents.skills.robutler.nli.skill.HTTPX_AVAILABLE', False):
            await skill.initialize(MockAgent())
            result = await skill.nli_tool(agent="@other", message="test")
            assert "HTTP client not available" in result


class TestNLIStatistics:
    """Test NLI statistics tracking"""
    
    def test_empty_statistics(self, nli_skill):
        stats = nli_skill.get_statistics()
        assert stats['total_communications'] == 0
        assert stats['success_rate'] == 0
    
    def test_statistics_with_history(self, nli_skill):
        nli_skill.communication_history = [
            NLICommunication(
                timestamp=datetime.now(),
                target_agent="@agent1",
                target_url="http://localhost:2224/agents/agent1/chat/completions",
                message="msg1",
                response="resp1",
                cost_usd=0.10,
                duration_ms=100.0,
                success=True
            ),
            NLICommunication(
                timestamp=datetime.now(),
                target_agent="@agent2",
                target_url="http://localhost:2224/agents/agent2/chat/completions",
                message="msg2",
                response="",
                cost_usd=0.0,
                duration_ms=200.0,
                success=False
            )
        ]
        
        stats = nli_skill.get_statistics()
        assert stats['total_communications'] == 2
        assert stats['successful_communications'] == 1
        assert stats['success_rate'] == 0.5
        assert stats['total_cost_usd'] == 0.10
        assert stats['average_duration_ms'] == 150.0


class TestNLIPrompt:
    """Test prompt generation"""
    
    @pytest.mark.asyncio
    async def test_prompt_includes_guidance(self, initialized_nli_skill):
        prompt = initialized_nli_skill.nli_general_prompt()
        assert "@username" in prompt
        assert "discovery_tool" in prompt
        assert "NEVER fabricate" in prompt
        assert "@test-agent" in prompt  # Agent name included
    
    @pytest.mark.asyncio
    async def test_prompt_warns_against_self_call(self, initialized_nli_skill):
        prompt = initialized_nli_skill.nli_general_prompt()
        assert "NEVER call yourself" in prompt


class TestNLIEdgeCases:
    """Test edge cases"""
    
    @pytest.mark.asyncio
    async def test_cleanup_with_no_client(self):
        skill = NLISkill()
        await skill.cleanup()  # Should not raise
