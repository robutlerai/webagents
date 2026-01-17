"""
Tests for DaemonClient
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from webagents.cli.client.daemon_client import DaemonClient


@pytest.mark.asyncio
async def test_daemon_client_is_running():
    """Test checking if daemon is running"""
    client = DaemonClient("http://localhost:8765")
    
    # Mock the httpx client
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        is_running = await client.is_running()
        assert is_running is True
        mock_get.assert_called_once_with("http://localhost:8765/health")


@pytest.mark.asyncio
async def test_daemon_client_is_not_running():
    """Test checking if daemon is not running"""
    client = DaemonClient("http://localhost:8765")
    
    # Mock the httpx client to raise exception
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        
        is_running = await client.is_running()
        assert is_running is False


@pytest.mark.asyncio
async def test_daemon_client_list_agents():
    """Test listing agents"""
    client = DaemonClient("http://localhost:8765")
    
    with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "agents": [
                {"name": "agent1", "source": "local"},
                {"name": "agent2", "source": "portal"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        agents = await client.list_agents()
        assert len(agents) == 2
        assert agents[0]["name"] == "agent1"


@pytest.mark.asyncio
async def test_daemon_client_register_agent():
    """Test registering an agent"""
    client = DaemonClient("http://localhost:8765")
    
    with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "registered", "name": "test-agent"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        result = await client.register_agent(Path("AGENT.md"))
        assert result["name"] == "test-agent"
        assert result["status"] == "registered"


@pytest.mark.asyncio
async def test_daemon_client_run_agent():
    """Test running an agent"""
    client = DaemonClient("http://localhost:8765")
    
    with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "completed",
            "agent": "test-agent",
            "trigger": "manual"
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        result = await client.run_agent("test-agent", trigger="manual")
        assert result["status"] == "completed"
        assert result["agent"] == "test-agent"
