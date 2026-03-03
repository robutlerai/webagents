"""
Pytest fixtures for core skills testing, particularly MCP skill tests.

Provides reusable mock objects, test data, and setup/teardown functionality
for comprehensive skill testing.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import the classes we're testing
from webagents.agents.skills.core.mcp import MCPSkill, MCPServerConfig, MCPTransport, MCPExecution


@pytest.fixture
def mock_agent():
    """Mock BaseAgent for skill testing"""
    agent = Mock()
    agent.name = "test-agent"
    agent.register_tool = Mock()
    agent.skills = {}
    return agent


@pytest.fixture
def mock_logger():
    """Mock logger for skill testing"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def basic_mcp_config():
    """Basic MCP skill configuration for testing"""
    return {
        'timeout': 30.0,
        'reconnect_interval': 60.0,
        'max_connection_errors': 5,
        'capability_refresh_interval': 300.0,
        'servers': []
    }


@pytest.fixture
def http_server_config():
    """HTTP MCP server configuration for testing"""
    return {
        'name': 'test-http-server',
        'url': 'http://localhost:8080/mcp',
        'transport': 'http',
        'api_key': 'test-api-key'
    }


@pytest.fixture
def sse_server_config():
    """SSE MCP server configuration for testing"""
    return {
        'name': 'test-sse-server',
        'url': 'https://api.example.com/sse',
        'transport': 'sse',
        'headers': {
            'Authorization': 'Bearer test-token',
            'X-Custom': 'test-value'
        }
    }


@pytest.fixture
def mock_mcp_server_config():
    """Mock MCPServerConfig for testing"""
    config = MCPServerConfig(
        name="mock-server",
        transport=MCPTransport.HTTP,
        url="http://localhost:8080/mcp"
    )
    config.connected = True
    config.enabled = True
    config.last_ping = datetime.utcnow()
    config.connection_errors = 0
    
    # Add some mock capabilities
    mock_tool = Mock()
    mock_tool.name = "mock_tool"
    mock_tool.description = "Mock tool for testing"
    config.available_tools = [mock_tool]
    
    mock_resource = Mock()
    mock_resource.name = "mock_resource"
    config.available_resources = [mock_resource]
    
    config.available_prompts = []
    
    return config


@pytest.fixture
def mock_mcp_tool():
    """Mock MCP tool for testing"""
    tool = Mock()
    tool.name = "test_function"
    tool.description = "Test function for MCP testing"
    tool.inputSchema = Mock()
    tool.inputSchema.model_dump.return_value = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "First test parameter"
            },
            "param2": {
                "type": "integer", 
                "description": "Second test parameter"
            }
        },
        "required": ["param1"]
    }
    return tool


@pytest.fixture
def mock_mcp_session():
    """Mock MCP session for testing"""
    session = Mock()
    
    # Mock successful tool listing
    mock_tools_result = Mock()
    mock_tools_result.tools = []
    session.list_tools = AsyncMock(return_value=mock_tools_result)
    
    # Mock successful resource listing
    mock_resources_result = Mock()
    mock_resources_result.resources = []
    session.list_resources = AsyncMock(return_value=mock_resources_result)
    
    # Mock successful prompt listing
    mock_prompts_result = Mock()
    mock_prompts_result.prompts = []
    session.list_prompts = AsyncMock(return_value=mock_prompts_result)
    
    # Mock successful tool execution
    mock_call_result = Mock()
    mock_content = Mock()
    mock_content.text = "Mock tool execution result"
    mock_call_result.content = [mock_content]
    session.call_tool = AsyncMock(return_value=mock_call_result)
    
    return session


@pytest.fixture
def sample_mcp_executions():
    """Sample MCP execution history for testing"""
    return [
        MCPExecution(
            timestamp=datetime.utcnow(),
            server_name="server1",
            operation_type="tool",
            operation_name="successful_tool",
            parameters={"param": "value"},
            result="success",
            duration_ms=150.5,
            success=True
        ),
        MCPExecution(
            timestamp=datetime.utcnow() - timedelta(minutes=1),
            server_name="server2",
            operation_type="tool",
            operation_name="failing_tool",
            parameters={},
            result=None,
            duration_ms=1000.0,
            success=False,
            error="Connection timeout"
        ),
        MCPExecution(
            timestamp=datetime.utcnow() - timedelta(minutes=2),
            server_name="server1",
            operation_type="resource",
            operation_name="get_data",
            parameters={"id": 123},
            result="resource data",
            duration_ms=75.0,
            success=True
        )
    ]


@pytest.fixture
def mcp_skill_with_servers(mock_agent, mock_logger):
    """MCPSkill instance with pre-configured mock servers"""
    skill = MCPSkill({
        'timeout': 30.0,
        'reconnect_interval': 60.0,
        'max_connection_errors': 5,
        'servers': []
    })
    
    skill.agent = mock_agent
    skill.logger = mock_logger
    
    # Add mock servers
    server1 = MCPServerConfig(
        name="server1",
        transport=MCPTransport.HTTP,
        url="http://localhost:8080/mcp"
    )
    server1.connected = True
    server1.enabled = True
    server1.available_tools = [Mock(), Mock()]  # 2 tools
    server1.available_resources = [Mock()]  # 1 resource
    server1.available_prompts = []
    
    server2 = MCPServerConfig(
        name="server2", 
        transport=MCPTransport.SSE,
        url="https://api.example.com/sse"
    )
    server2.connected = False
    server2.enabled = True
    server2.available_tools = [Mock()]  # 1 tool
    server2.available_resources = []
    server2.available_prompts = [Mock(), Mock()]  # 2 prompts
    
    skill.servers = {"server1": server1, "server2": server2}
    
    return skill


@pytest.fixture
async def initialized_mcp_skill(mock_agent, mock_logger):
    """Fully initialized MCPSkill for integration testing"""
    skill = MCPSkill({
        'timeout': 30.0,
        'reconnect_interval': 60.0,
        'max_connection_errors': 3,
        'capability_refresh_interval': 300.0,
        'servers': []
    })
    
    # Initialize with mocked components
    with patch('webagents.utils.logging.get_logger', return_value=mock_logger):
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            with patch('asyncio.create_task') as mock_create_task:
                # Mock the background tasks
                mock_monitoring_task = Mock()
                mock_capability_task = Mock()
                mock_create_task.side_effect = [mock_monitoring_task, mock_capability_task]
                
                await skill.initialize(mock_agent)
                
                # Store task mocks for cleanup
                skill._monitoring_task = mock_monitoring_task
                skill._capability_refresh_task = mock_capability_task
    
    yield skill
    
    # Cleanup
    await skill.cleanup()


@pytest.fixture
def mock_mcp_sdk_available():
    """Context manager to mock MCP SDK availability"""
    class MockMCPSDK:
        def __init__(self, available=True):
            self.available = available
            
        def __enter__(self):
            self.patcher = patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', self.available)
            return self.patcher.__enter__()
            
        def __exit__(self, *args):
            return self.patcher.__exit__(*args)
    
    return MockMCPSDK


@pytest.fixture
def mock_http_client_generator():
    """Mock for create_http_client that returns proper async generator"""
    async def mock_generator():
        receive_stream = Mock()
        send_stream = Mock()
        get_session_id = Mock()
        yield (receive_stream, send_stream, get_session_id)
    
    return mock_generator


@pytest.fixture
def mock_sse_client_context():
    """Mock for create_sse_client that returns proper context manager"""
    mock_session = Mock()
    mock_session.list_tools = AsyncMock()
    mock_session.list_resources = AsyncMock()
    mock_session.list_prompts = AsyncMock()
    mock_session.call_tool = AsyncMock()
    
    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_session)
    mock_context.__aexit__ = AsyncMock()
    
    return mock_context


# Event loop handled by pytest-asyncio automatically


# Utility fixtures for common test patterns

@pytest.fixture
def assert_server_state():
    """Helper to assert server state in tests"""
    def _assert_server_state(server: MCPServerConfig, 
                           connected: bool = None,
                           enabled: bool = None,
                           connection_errors: int = None,
                           has_tools: bool = None,
                           has_resources: bool = None,
                           has_prompts: bool = None):
        """Assert various server state conditions"""
        if connected is not None:
            assert server.connected == connected
        if enabled is not None:
            assert server.enabled == enabled
        if connection_errors is not None:
            assert server.connection_errors == connection_errors
        if has_tools is not None:
            assert (len(server.available_tools) > 0) == has_tools
        if has_resources is not None:
            assert (len(server.available_resources) > 0) == has_resources
        if has_prompts is not None:
            assert (len(server.available_prompts) > 0) == has_prompts
    
    return _assert_server_state


@pytest.fixture
def assert_execution_recorded():
    """Helper to assert execution was recorded properly"""
    def _assert_execution_recorded(skill: MCPSkill,
                                 server_name: str,
                                 operation_name: str,
                                 success: bool,
                                 parameters: Dict[str, Any] = None):
        """Assert an execution was recorded with expected values"""
        executions = [e for e in skill.execution_history 
                     if e.server_name == server_name and e.operation_name == operation_name]
        
        assert len(executions) > 0, f"No executions found for {server_name}::{operation_name}"
        
        execution = executions[-1]  # Get most recent
        assert execution.success == success
        
        if parameters is not None:
            assert execution.parameters == parameters
    
    return _assert_execution_recorded


@pytest.fixture
def create_test_skill():
    """Factory fixture to create MCPSkill instances with custom config"""
    created_skills = []
    
    def _create_skill(config: Dict[str, Any] = None, 
                     mock_agent: Mock = None,
                     mock_logger: Mock = None):
        """Create MCPSkill instance with optional config and mocks"""
        skill = MCPSkill(config or {})
        
        if mock_agent:
            skill.agent = mock_agent
        if mock_logger:
            skill.logger = mock_logger
            
        created_skills.append(skill)
        return skill
    
    yield _create_skill
    
    # Cleanup all created skills
    for skill in created_skills:
        try:
            asyncio.create_task(skill.cleanup())
        except Exception:
            pass 