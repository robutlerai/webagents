"""
Tests for MCPSkill - Model Context Protocol Integration

Comprehensive test coverage for MCP server connections, tool discovery,
execution management, and error handling scenarios.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Test imports  
from webagents.agents.skills.core.mcp import MCPSkill, MCPServerConfig, MCPTransport, MCPExecution


class TestMCPSkillInitialization:
    """Test MCPSkill initialization and configuration"""
    
    def test_mcp_skill_init_default_config(self):
        """Test MCPSkill initialization with default configuration"""
        skill = MCPSkill()
        
        assert skill.config == {}
        assert skill.default_timeout == 30.0
        assert skill.reconnect_interval == 60.0
        assert skill.max_connection_errors == 5
        assert skill.capability_refresh_interval == 300.0
        assert skill.servers == {}
        assert skill.sessions == {}
        assert skill.execution_history == []
        assert skill.dynamic_tools == {}
        assert skill._monitoring_task is None
        assert skill._capability_refresh_task is None
        
    def test_mcp_skill_init_custom_config(self):
        """Test MCPSkill initialization with custom configuration"""
        config = {
            'timeout': 45.0,
            'reconnect_interval': 120.0,
            'max_connection_errors': 10,
            'capability_refresh_interval': 600.0,
            'servers': []
        }
        
        skill = MCPSkill(config)
        
        assert skill.config == config
        assert skill.default_timeout == 45.0
        assert skill.reconnect_interval == 120.0
        assert skill.max_connection_errors == 10
        assert skill.capability_refresh_interval == 600.0
    
    @pytest.mark.asyncio
    async def test_mcp_skill_initialize_without_mcp_sdk(self):
        """Test initialization when MCP SDK is not available"""
        skill = MCPSkill()
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.name = "test-agent"
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', False):
            with patch('webagents.utils.logging.get_logger') as mock_logger:
                mock_logger.return_value.warning = Mock()
                
                await skill.initialize(mock_agent)
                
                # Should warn about missing SDK
                mock_logger.return_value.warning.assert_called_with(
                    "MCP SDK not available - install 'mcp' package for full functionality"
                )
                
                # Should not start background tasks
                assert skill._monitoring_task is None
                assert skill._capability_refresh_task is None
    
    @pytest.mark.asyncio
    async def test_mcp_skill_initialize_with_servers_config(self):
        """Test initialization with pre-configured servers"""
        config = {
            'servers': [
                {
                    'name': 'test-server',
                    'url': 'http://localhost:8080/mcp',
                    'transport': 'http'
                }
            ]
        }
        
        skill = MCPSkill(config)
        mock_agent = Mock()
        mock_agent.name = "test-agent"
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            with patch('webagents.utils.logging.get_logger'):
                with patch.object(skill, '_register_mcp_server', new_callable=AsyncMock) as mock_register:
                    mock_register.return_value = True
                    
                    with patch('asyncio.create_task') as mock_create_task:
                        await skill.initialize(mock_agent)
                        
                        # Should register servers from config
                        mock_register.assert_called_once_with({
                            'name': 'test-server',
                            'url': 'http://localhost:8080/mcp',
                            'transport': 'http'
                        })
                        
                        # Should create background tasks
                        assert mock_create_task.call_count == 2


class TestMCPServerConfig:
    """Test MCPServerConfig data structure"""
    
    def test_mcp_server_config_basic(self):
        """Test basic MCPServerConfig creation"""
        config = MCPServerConfig(
            name="test-server",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        
        assert config.name == "test-server"
        assert config.transport == MCPTransport.HTTP
        assert config.url == "http://localhost:8080/mcp"
        assert config.headers is None
        assert config.enabled is True
        assert config.connected is False
        assert config.last_ping is None
        assert config.connection_errors == 0
        assert config.available_tools == []
        assert config.available_resources == []
        assert config.available_prompts == []
    
    def test_mcp_server_config_with_headers(self):
        """Test MCPServerConfig with custom headers"""
        headers = {"Authorization": "Bearer token123", "X-Custom": "value"}
        
        config = MCPServerConfig(
            name="auth-server",
            transport=MCPTransport.SSE,
            url="https://api.example.com/mcp",
            headers=headers
        )
        
        assert config.headers == headers
        assert config.transport == MCPTransport.SSE


class TestMCPTransport:
    """Test MCPTransport enum"""
    
    def test_mcp_transport_values(self):
        """Test MCPTransport enum values"""
        assert MCPTransport.HTTP.value == "http"
        assert MCPTransport.SSE.value == "sse"
        assert MCPTransport.WEBSOCKET.value == "websocket"
    
    def test_mcp_transport_enum_creation(self):
        """Test creating MCPTransport from string values"""
        assert MCPTransport("http") == MCPTransport.HTTP
        assert MCPTransport("sse") == MCPTransport.SSE
        assert MCPTransport("websocket") == MCPTransport.WEBSOCKET


class TestMCPExecution:
    """Test MCPExecution data structure"""
    
    def test_mcp_execution_successful(self):
        """Test MCPExecution for successful operation"""
        timestamp = datetime.utcnow()
        execution = MCPExecution(
            timestamp=timestamp,
            server_name="test-server",
            operation_type="tool",
            operation_name="test_tool",
            parameters={"param1": "value1"},
            result="success",
            duration_ms=123.45,
            success=True
        )
        
        assert execution.timestamp == timestamp
        assert execution.server_name == "test-server"
        assert execution.operation_type == "tool"
        assert execution.operation_name == "test_tool"
        assert execution.parameters == {"param1": "value1"}
        assert execution.result == "success"
        assert execution.duration_ms == 123.45
        assert execution.success is True
        assert execution.error is None
    
    def test_mcp_execution_failed(self):
        """Test MCPExecution for failed operation"""
        execution = MCPExecution(
            timestamp=datetime.utcnow(),
            server_name="test-server",
            operation_type="tool",
            operation_name="failing_tool",
            parameters={},
            result=None,
            duration_ms=50.0,
            success=False,
            error="Connection timeout"
        )
        
        assert execution.success is False
        assert execution.error == "Connection timeout"
        assert execution.result is None


class TestMCPServerRegistration:
    """Test MCP server registration and connection"""
    
    @pytest.mark.asyncio
    async def test_register_mcp_server_http_success(self):
        """Test successful HTTP MCP server registration"""
        skill = MCPSkill()
        mock_agent = Mock()
        mock_agent.name = "test-agent"
        skill.agent = mock_agent
        skill.logger = Mock()
        
        server_config = {
            'name': 'test-http-server',
            'url': 'http://localhost:8080/mcp',
            'transport': 'http',
            'api_key': 'test-key'
        }
        
        with patch.object(skill, '_connect_to_server', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            result = await skill._register_mcp_server(server_config)
            
            assert result is True
            assert 'test-http-server' in skill.servers
            
            server = skill.servers['test-http-server']
            assert server.name == 'test-http-server'
            assert server.transport == MCPTransport.HTTP
            assert server.url == 'http://localhost:8080/mcp'
            assert server.headers['Authorization'] == 'Bearer test-key'
            
            mock_connect.assert_called_once_with(server)
    
    @pytest.mark.asyncio
    async def test_register_mcp_server_sse_success(self):
        """Test successful SSE MCP server registration"""
        skill = MCPSkill()
        mock_agent = Mock()
        skill.agent = mock_agent
        skill.logger = Mock()
        
        server_config = {
            'name': 'test-sse-server',
            'url': 'https://api.example.com/sse',
            'transport': 'sse',
            'headers': {'X-Custom': 'value'}
        }
        
        with patch.object(skill, '_connect_to_server', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            result = await skill._register_mcp_server(server_config)
            
            assert result is True
            server = skill.servers['test-sse-server']
            assert server.transport == MCPTransport.SSE
            assert 'X-Custom' in server.headers
    
    @pytest.mark.asyncio
    async def test_register_mcp_server_connection_failure(self):
        """Test MCP server registration with connection failure"""
        skill = MCPSkill()
        mock_agent = Mock()
        skill.agent = mock_agent
        skill.logger = Mock()
        
        server_config = {
            'name': 'failing-server',
            'url': 'http://nonexistent:8080/mcp',
            'transport': 'http'
        }
        
        with patch.object(skill, '_connect_to_server', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = False
            
            result = await skill._register_mcp_server(server_config)
            
            assert result is False
            assert 'failing-server' in skill.servers  # Server still registered
            skill.logger.warning.assert_called()
    
    @pytest.mark.asyncio
    async def test_register_mcp_server_invalid_config(self):
        """Test MCP server registration with invalid configuration"""
        skill = MCPSkill()
        mock_agent = Mock()
        skill.agent = mock_agent  
        skill.logger = Mock()
        
        # Missing required fields
        server_config = {
            'transport': 'http'
            # Missing 'name' and 'url'
        }
        
        result = await skill._register_mcp_server(server_config)
        
        assert result is False
        skill.logger.error.assert_called()


class TestMCPServerConnection:
    """Test MCP server connection management"""
    
    @pytest.mark.asyncio
    async def test_connect_to_http_server_success(self):
        """Test successful connection to HTTP MCP server"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(
            name="test-server",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        
        # Mock the MCP SDK components
        mock_session = Mock()
        mock_session.list_tools = AsyncMock()
        mock_session.list_resources = AsyncMock()
        mock_session.list_prompts = AsyncMock()
        
        with patch('webagents.agents.skills.core.mcp.skill.create_http_client') as mock_create_http:
            # Mock the async generator
            async def mock_client_generator():
                yield (Mock(), Mock(), Mock())
            
            mock_create_http.return_value = mock_client_generator()
            
            with patch.object(skill, '_discover_capabilities', new_callable=AsyncMock) as mock_discover:
                result = await skill._connect_to_server(server)
                
                assert result is True
                assert server.connected is True
                assert server.last_ping is not None
                assert server.connection_errors == 0
                assert 'test-server' in skill.sessions
                
                mock_discover.assert_called_once_with(server)
    
    @pytest.mark.asyncio
    async def test_connect_to_sse_server_success(self):
        """Test successful connection to SSE MCP server"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(
            name="sse-server",
            transport=MCPTransport.SSE,
            url="https://api.example.com/sse"
        )
        
        mock_session = Mock()
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        
        with patch('webagents.agents.skills.core.mcp.skill.create_sse_client') as mock_create_sse:
            mock_create_sse.return_value = mock_context
            
            with patch.object(skill, '_discover_capabilities', new_callable=AsyncMock) as mock_discover:
                result = await skill._connect_to_server(server)
                
                assert result is True
                assert server.connected is True
                assert 'sse-server' in skill.sessions
    
    @pytest.mark.asyncio
    async def test_connect_to_server_unsupported_transport(self):
        """Test connection to server with unsupported transport"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(
            name="unsupported-server",
            transport=MCPTransport.WEBSOCKET,  # Not implemented yet
            url="ws://localhost:8080/mcp"
        )
        
        result = await skill._connect_to_server(server)
        
        assert result is False
        assert server.connected is False
        skill.logger.error.assert_called_with("Transport MCPTransport.WEBSOCKET not implemented")
    
    @pytest.mark.asyncio
    async def test_connect_to_server_exception(self):
        """Test connection failure due to exception"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(
            name="exception-server",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        
        with patch('webagents.agents.skills.core.mcp.skill.create_http_client') as mock_create_http:
            mock_create_http.side_effect = Exception("Connection failed")
            
            result = await skill._connect_to_server(server)
            
            assert result is False
            assert server.connected is False
            assert server.connection_errors == 1
            skill.logger.error.assert_called()


class TestMCPCapabilityDiscovery:
    """Test MCP server capability discovery"""
    
    @pytest.mark.asyncio
    async def test_discover_capabilities_success(self):
        """Test successful capability discovery"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(
            name="test-server",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        
        # Mock session with capabilities
        mock_session = Mock()
        
        # Mock tools discovery
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = Mock()
        mock_tool.inputSchema.model_dump.return_value = {"type": "object", "properties": {}}
        
        mock_tools_result = Mock()
        mock_tools_result.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        
        # Mock resources discovery
        mock_resource = Mock()
        mock_resource.name = "test_resource"
        mock_resources_result = Mock()
        mock_resources_result.resources = [mock_resource]
        mock_session.list_resources = AsyncMock(return_value=mock_resources_result)
        
        # Mock prompts discovery
        mock_prompt = Mock()
        mock_prompt.name = "test_prompt"
        mock_prompts_result = Mock()
        mock_prompts_result.prompts = [mock_prompt]
        mock_session.list_prompts = AsyncMock(return_value=mock_prompts_result)
        
        skill.sessions['test-server'] = mock_session
        
        # Mock agent for dynamic tool registration
        mock_agent = Mock()
        mock_agent.register_tool = Mock()
        skill.agent = mock_agent
        
        with patch.object(skill, '_register_dynamic_tool', new_callable=AsyncMock) as mock_register:
            await skill._discover_capabilities(server)
            
            assert len(server.available_tools) == 1
            assert len(server.available_resources) == 1
            assert len(server.available_prompts) == 1
            
            mock_register.assert_called_once_with(server, mock_tool)
            skill.logger.info.assert_any_call("Discovered 1 tools from 'test-server'")
            skill.logger.info.assert_any_call("Discovered 1 resources from 'test-server'")
            skill.logger.info.assert_any_call("Discovered 1 prompts from 'test-server'")
    
    @pytest.mark.asyncio
    async def test_discover_capabilities_partial_failure(self):
        """Test capability discovery with partial failures"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(
            name="partial-server",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        
        mock_session = Mock()
        
        # Tools discovery succeeds
        mock_tools_result = Mock()
        mock_tools_result.tools = []
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        
        # Resources discovery fails
        mock_session.list_resources = AsyncMock(side_effect=Exception("Resources unavailable"))
        
        # Prompts discovery fails
        mock_session.list_prompts = AsyncMock(side_effect=Exception("Prompts unavailable"))
        
        skill.sessions['partial-server'] = mock_session
        
        await skill._discover_capabilities(server)
        
        # Should handle partial failures gracefully
        assert len(server.available_tools) == 0  # Empty but discovered
        assert len(server.available_resources) == 0  # Failed, remains empty
        assert len(server.available_prompts) == 0  # Failed, remains empty
        
        skill.logger.warning.assert_any_call("Resource discovery failed for 'partial-server': Resources unavailable")
        skill.logger.warning.assert_any_call("Prompt discovery failed for 'partial-server': Prompts unavailable")
    
    @pytest.mark.asyncio
    async def test_discover_capabilities_no_session(self):
        """Test capability discovery when session doesn't exist"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(
            name="no-session-server",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        
        # No session exists for this server
        await skill._discover_capabilities(server)
        
        # Should handle gracefully without errors
        assert len(server.available_tools) == 0
        assert len(server.available_resources) == 0
        assert len(server.available_prompts) == 0


class TestMCPDynamicToolRegistration:
    """Test dynamic tool registration from MCP servers"""
    
    @pytest.mark.asyncio
    async def test_register_dynamic_tool_success(self):
        """Test successful dynamic tool registration"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.register_tool = Mock()
        skill.agent = mock_agent
        
        server = MCPServerConfig(
            name="tool-server",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        
        # Mock MCP tool
        mock_tool = Mock()
        mock_tool.name = "test_function"
        mock_tool.description = "Test function from MCP server"
        mock_tool.inputSchema = Mock()
        mock_tool.inputSchema.model_dump.return_value = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Test parameter"}
            }
        }
        
        await skill._register_dynamic_tool(server, mock_tool)
        
        # Check dynamic tool was created and registered
        dynamic_name = "tool-server_test_function"
        assert dynamic_name in skill.dynamic_tools
        
        # Check agent registration was called
        mock_agent.register_tool.assert_called_once()
        call_args = mock_agent.register_tool.call_args
        registered_func = call_args[0][0]
        
        assert hasattr(registered_func, '_webagents_is_tool')
        assert registered_func._webagents_is_tool is True
        assert registered_func._tool_scope == "all"
        assert hasattr(registered_func, '_webagents_tool_definition')
        
        # Check OpenAI schema generation
        schema = registered_func._webagents_tool_definition
        assert schema['type'] == 'function'
        assert schema['function']['name'] == dynamic_name
        assert schema['function']['description'] == 'Test function from MCP server'
        assert 'param1' in schema['function']['parameters']['properties']
    
    @pytest.mark.asyncio
    async def test_register_dynamic_tool_no_name(self):
        """Test dynamic tool registration when tool has no name"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        mock_agent = Mock()
        skill.agent = mock_agent
        
        server = MCPServerConfig(name="test-server", transport=MCPTransport.HTTP)
        
        # Tool without name
        mock_tool = Mock()
        mock_tool.name = None
        
        await skill._register_dynamic_tool(server, mock_tool)
        
        # Should not register anything
        assert len(skill.dynamic_tools) == 0
        mock_agent.register_tool.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_register_dynamic_tool_exception(self):
        """Test dynamic tool registration with exception"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        mock_agent = Mock()
        mock_agent.register_tool = Mock(side_effect=Exception("Registration failed"))
        skill.agent = mock_agent
        
        server = MCPServerConfig(name="error-server", transport=MCPTransport.HTTP)
        
        mock_tool = Mock()
        mock_tool.name = "error_tool"
        mock_tool.description = "Tool that will fail to register"
        mock_tool.inputSchema = Mock()
        mock_tool.inputSchema.model_dump.return_value = {"type": "object"}
        
        await skill._register_dynamic_tool(server, mock_tool)
        
        # Should handle error gracefully
        skill.logger.error.assert_called()
        assert "error-server_error_tool" not in skill.dynamic_tools


class TestMCPToolExecution:
    """Test MCP tool execution"""
    
    @pytest.mark.asyncio
    async def test_execute_mcp_tool_success(self):
        """Test successful MCP tool execution"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        # Setup server and session
        server = MCPServerConfig(
            name="exec-server",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        skill.servers['exec-server'] = server
        
        # Mock session with successful tool execution
        mock_session = Mock()
        mock_result = Mock()
        mock_content = Mock()
        mock_content.text = "Tool execution successful"
        mock_result.content = [mock_content]
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        skill.sessions['exec-server'] = mock_session
        
        parameters = {"param1": "value1", "param2": 42}
        
        result = await skill._execute_mcp_tool('exec-server', 'test_tool', parameters)
        
        assert result == "Tool execution successful"
        
        # Check execution was recorded
        assert len(skill.execution_history) == 1
        execution = skill.execution_history[0]
        assert execution.server_name == 'exec-server'
        assert execution.operation_name == 'test_tool'
        assert execution.parameters == parameters
        assert execution.success is True
        assert execution.error is None
        
        skill.logger.info.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_mcp_tool_server_not_found(self):
        """Test MCP tool execution when server not found"""
        skill = MCPSkill()
        
        result = await skill._execute_mcp_tool('nonexistent-server', 'test_tool', {})
        
        assert result == "❌ MCP server 'nonexistent-server' not found"
    
    @pytest.mark.asyncio
    async def test_execute_mcp_tool_not_connected(self):
        """Test MCP tool execution when server not connected"""
        skill = MCPSkill()
        
        # Server exists but no session
        server = MCPServerConfig(name="disconnected-server", transport=MCPTransport.HTTP)
        skill.servers['disconnected-server'] = server
        
        result = await skill._execute_mcp_tool('disconnected-server', 'test_tool', {})
        
        assert result == "❌ MCP server 'disconnected-server' not connected"
    
    @pytest.mark.asyncio
    async def test_execute_mcp_tool_execution_failure(self):
        """Test MCP tool execution with execution failure"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        # Setup server and session
        server = MCPServerConfig(name="fail-server", transport=MCPTransport.HTTP)
        skill.servers['fail-server'] = server
        
        mock_session = Mock()
        mock_session.call_tool = AsyncMock(side_effect=Exception("Tool execution failed"))
        skill.sessions['fail-server'] = mock_session
        
        result = await skill._execute_mcp_tool('fail-server', 'failing_tool', {'param': 'value'})
        
        assert "❌ MCP tool execution error: Tool execution failed" in result
        
        # Check failed execution was recorded
        assert len(skill.execution_history) == 1
        execution = skill.execution_history[0]
        assert execution.success is False
        assert execution.error == "Tool execution failed"
        
        skill.logger.error.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_mcp_tool_complex_result(self):
        """Test MCP tool execution with complex result format"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(name="complex-server", transport=MCPTransport.HTTP)
        skill.servers['complex-server'] = server
        
        # Mock session with complex result
        mock_session = Mock()
        mock_result = Mock()
        
        # Multiple content items
        mock_content1 = Mock()
        mock_content1.text = "Part 1: "
        mock_content2 = Mock()
        mock_content2.data = {"key": "value"}
        mock_content2.text = None  # Only has data attribute
        mock_content3 = Mock()
        mock_content3.text = " Part 3"
        
        mock_result.content = [mock_content1, mock_content2, mock_content3]
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        skill.sessions['complex-server'] = mock_session
        
        result = await skill._execute_mcp_tool('complex-server', 'complex_tool', {})
        
        # Should concatenate all content parts
        assert "Part 1: " in result
        assert "{'key': 'value'}" in result
        assert " Part 3" in result


class TestMCPBackgroundTasks:
    """Test MCP background monitoring and capability refresh tasks"""
    
    @pytest.mark.asyncio
    async def test_monitor_connections_reconnect_success(self):
        """Test connection monitoring with successful reconnection"""
        skill = MCPSkill({'reconnect_interval': 0.1})  # Fast for testing
        skill.logger = Mock()
        
        # Server with connection errors
        server = MCPServerConfig(name="monitor-server", transport=MCPTransport.HTTP)
        server.connected = False
        server.connection_errors = 2  # Below max
        server.enabled = True
        skill.servers['monitor-server'] = server
        
        with patch.object(skill, '_connect_to_server', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            # Run one iteration of monitoring
            task = asyncio.create_task(skill._monitor_connections())
            await asyncio.sleep(0.2)  # Let it run once
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Should attempt reconnection
            mock_connect.assert_called_with(server)
    
    @pytest.mark.asyncio
    async def test_monitor_connections_disable_server(self):
        """Test connection monitoring disables server after too many errors"""
        skill = MCPSkill({'reconnect_interval': 0.1, 'max_connection_errors': 3})
        skill.logger = Mock()
        
        # Server with too many connection errors
        server = MCPServerConfig(name="failing-server", transport=MCPTransport.HTTP)
        server.connected = False
        server.connection_errors = 5  # Above max
        server.enabled = True
        skill.servers['failing-server'] = server
        
        with patch.object(skill, '_connect_to_server', new_callable=AsyncMock) as mock_connect:
            # Run one iteration
            task = asyncio.create_task(skill._monitor_connections())
            await asyncio.sleep(0.2)
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Should not attempt reconnection
            mock_connect.assert_not_called()
            # Should log warning about disabling
            skill.logger.warning.assert_called()
    
    @pytest.mark.asyncio
    async def test_monitor_connections_health_check(self):
        """Test connection monitoring performs health checks"""
        skill = MCPSkill({'reconnect_interval': 0.1})
        skill.logger = Mock()
        
        # Connected and enabled server
        server = MCPServerConfig(name="healthy-server", transport=MCPTransport.HTTP)
        server.connected = True
        server.enabled = True
        skill.servers['healthy-server'] = server
        
        with patch.object(skill, '_health_check_server', new_callable=AsyncMock) as mock_health:
            # Run one iteration
            task = asyncio.create_task(skill._monitor_connections())
            await asyncio.sleep(0.2)
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Should perform health check
            mock_health.assert_called_with(server)
    
    @pytest.mark.asyncio
    async def test_refresh_capabilities_task(self):
        """Test capability refresh background task"""
        skill = MCPSkill({'capability_refresh_interval': 0.1})
        skill.logger = Mock()
        
        # Connected server
        server = MCPServerConfig(name="refresh-server", transport=MCPTransport.HTTP)
        server.connected = True
        server.enabled = True
        skill.servers['refresh-server'] = server
        
        with patch.object(skill, '_discover_capabilities', new_callable=AsyncMock) as mock_discover:
            # Run one iteration
            task = asyncio.create_task(skill._refresh_capabilities())
            await asyncio.sleep(0.2)
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Should refresh capabilities
            mock_discover.assert_called_with(server)
    
    @pytest.mark.asyncio
    async def test_health_check_server_success(self):
        """Test successful server health check"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(name="health-server", transport=MCPTransport.HTTP)
        server.connection_errors = 1  # Has some errors
        
        # Mock successful session
        mock_session = Mock()
        mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
        skill.sessions['health-server'] = mock_session
        
        await skill._health_check_server(server)
        
        # Should reset errors and update ping time
        assert server.connection_errors == 0
        assert server.last_ping is not None
    
    @pytest.mark.asyncio
    async def test_health_check_server_failure(self):
        """Test server health check failure"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(name="unhealthy-server", transport=MCPTransport.HTTP)
        server.connection_errors = 0
        
        # Mock failing session
        mock_session = Mock()
        mock_session.list_tools = AsyncMock(side_effect=Exception("Health check failed"))
        skill.sessions['unhealthy-server'] = mock_session
        
        await skill._health_check_server(server)
        
        # Should increment errors
        assert server.connection_errors == 1
        skill.logger.warning.assert_called()
    
    @pytest.mark.asyncio
    async def test_health_check_no_session(self):
        """Test health check when session doesn't exist"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(name="no-session-server", transport=MCPTransport.HTTP)
        server.connection_errors = 0
        
        # No session exists
        await skill._health_check_server(server)
        
        # Should increment errors
        assert server.connection_errors == 1


class TestMCPSkillTools:
    """Test MCPSkill provided tools"""
    
    @pytest.mark.asyncio
    async def test_list_mcp_servers_no_sdk(self):
        """Test list_mcp_servers when SDK not available"""
        skill = MCPSkill()
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', False):
            result = await skill.list_mcp_servers()
            
            assert result == "❌ MCP SDK not available - install 'mcp' package"
    
    @pytest.mark.asyncio
    async def test_list_mcp_servers_no_servers(self):
        """Test list_mcp_servers with no configured servers"""
        skill = MCPSkill()
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            result = await skill.list_mcp_servers()
            
            assert result == "📝 No MCP servers configured"
    
    @pytest.mark.asyncio
    async def test_list_mcp_servers_with_servers(self):
        """Test list_mcp_servers with configured servers"""
        skill = MCPSkill()
        
        # Add test servers
        server1 = MCPServerConfig(
            name="server1",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        server1.connected = True
        server1.enabled = True
        server1.last_ping = datetime.utcnow()
        server1.available_tools = [Mock(), Mock()]  # 2 tools
        server1.available_resources = [Mock()]  # 1 resource
        server1.available_prompts = []  # 0 prompts
        server1.connection_errors = 0
        
        server2 = MCPServerConfig(
            name="server2",
            transport=MCPTransport.SSE,
            url="https://api.example.com/sse"
        )
        server2.connected = False
        server2.enabled = True
        server2.last_ping = None
        server2.connection_errors = 3
        
        skill.servers = {"server1": server1, "server2": server2}
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            result = await skill.list_mcp_servers()
            
            assert "📡 MCP Servers (Official SDK):" in result
            assert "**server1** (http)" in result
            assert "🟢 Connected" in result
            assert "Tools: 2" in result
            assert "Resources: 1" in result
            assert "Prompts: 0" in result
            assert "Errors: 0" in result
            
            assert "**server2** (sse)" in result
            assert "🔴 Disconnected" in result
            assert "Errors: 3" in result
    
    @pytest.mark.asyncio
    async def test_show_mcp_history_empty(self):
        """Test show_mcp_history with empty history"""
        skill = MCPSkill()
        
        result = await skill.show_mcp_history()
        
        assert "📊 MCP Execution History: Empty" in result
    
    @pytest.mark.asyncio
    async def test_show_mcp_history_with_executions(self):
        """Test show_mcp_history with execution records"""
        skill = MCPSkill()
        
        # Add test execution records
        execution1 = MCPExecution(
            timestamp=datetime.utcnow(),
            server_name="test-server",
            operation_type="tool",
            operation_name="test_tool",
            parameters={"param": "value"},
            result="success",
            duration_ms=150.5,
            success=True
        )
        
        execution2 = MCPExecution(
            timestamp=datetime.utcnow() - timedelta(minutes=1),
            server_name="other-server",
            operation_type="tool",
            operation_name="failing_tool",
            parameters={},
            result=None,
            duration_ms=1000.0,
            success=False,
            error="Connection timeout"
        )
        
        skill.execution_history = [execution1, execution2]
        
        result = await skill.show_mcp_history(limit=10)
        
        assert "📊 MCP Execution History (Last 2 operations):" in result
        assert "✅ test-server::test_tool" in result
        assert "150.5ms" in result
        assert "❌ other-server::failing_tool" in result
        assert "Connection timeout" in result
    
    @pytest.mark.asyncio
    async def test_add_mcp_server_no_sdk(self):
        """Test add_mcp_server when SDK not available"""
        skill = MCPSkill()
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', False):
            result = await skill.add_mcp_server(
                name="test-server",
                url="http://localhost:8080/mcp",
                transport="http"
            )
            
            assert result == "❌ MCP SDK not available - install 'mcp' package"
    
    @pytest.mark.asyncio
    async def test_add_mcp_server_already_exists(self):
        """Test add_mcp_server when server already exists"""
        skill = MCPSkill()
        
        # Pre-existing server
        skill.servers["existing-server"] = MCPServerConfig(
            name="existing-server",
            transport=MCPTransport.HTTP
        )
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            result = await skill.add_mcp_server(
                name="existing-server",
                url="http://localhost:8080/mcp",
                transport="http"
            )
            
            assert result == "❌ MCP server 'existing-server' already exists"
    
    @pytest.mark.asyncio
    async def test_add_mcp_server_invalid_transport(self):
        """Test add_mcp_server with invalid transport"""
        skill = MCPSkill()
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            result = await skill.add_mcp_server(
                name="invalid-transport-server",
                url="http://localhost:8080/mcp",
                transport="invalid"
            )
            
            assert "❌ Invalid transport 'invalid'" in result
    
    @pytest.mark.asyncio
    async def test_add_mcp_server_success(self):
        """Test successful add_mcp_server"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            with patch.object(skill, '_register_mcp_server', new_callable=AsyncMock) as mock_register:
                mock_register.return_value = True
                
                # Mock the server that gets created
                mock_server = MCPServerConfig(
                    name="new-server",
                    transport=MCPTransport.HTTP,
                    url="http://localhost:8080/mcp"
                )
                mock_server.available_tools = [Mock(), Mock()]  # 2 tools
                mock_server.available_resources = [Mock()]  # 1 resource  
                mock_server.available_prompts = []  # 0 prompts
                skill.servers["new-server"] = mock_server
                
                result = await skill.add_mcp_server(
                    name="new-server",
                    url="http://localhost:8080/mcp",
                    transport="http",
                    api_key="test-key"
                )
                
                assert "✅ Connected: MCP server 'new-server'" in result
                assert "Transport: http" in result
                assert "Tools: 2" in result
                assert "Resources: 1" in result
                assert "Prompts: 0" in result
                assert "Total Capabilities: 3" in result
                
                # Check registration was called with correct config
                mock_register.assert_called_once()
                call_args = mock_register.call_args[0][0]
                assert call_args['name'] == 'new-server'
                assert call_args['transport'] == 'http'
                assert call_args['url'] == 'http://localhost:8080/mcp'
                assert call_args['api_key'] == 'test-key'
    
    @pytest.mark.asyncio
    async def test_add_mcp_server_connection_failure(self):
        """Test add_mcp_server with connection failure"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            with patch.object(skill, '_register_mcp_server', new_callable=AsyncMock) as mock_register:
                mock_register.return_value = False  # Connection failed
                
                # Mock the server that gets created
                mock_server = MCPServerConfig(
                    name="failing-server",
                    transport=MCPTransport.HTTP,
                    url="http://nonexistent:8080/mcp"
                )
                skill.servers["failing-server"] = mock_server
                
                result = await skill.add_mcp_server(
                    name="failing-server",
                    url="http://nonexistent:8080/mcp",
                    transport="http"
                )
                
                assert "⚠️  Registered but connection failed: MCP server 'failing-server'" in result
    
    @pytest.mark.asyncio
    async def test_add_mcp_server_exception(self):
        """Test add_mcp_server with exception during registration"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            with patch.object(skill, '_register_mcp_server', new_callable=AsyncMock) as mock_register:
                mock_register.side_effect = Exception("Registration failed")
                
                result = await skill.add_mcp_server(
                    name="error-server",
                    url="http://localhost:8080/mcp",
                    transport="http"
                )
                
                assert "❌ Failed to add MCP server: Registration failed" in result
                skill.logger.error.assert_called()


class TestMCPSkillStatistics:
    """Test MCPSkill statistics functionality"""
    
    def test_get_statistics_empty(self):
        """Test get_statistics with no servers or executions"""
        skill = MCPSkill()
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            stats = skill.get_statistics()
            
            assert stats['total_servers'] == 0
            assert stats['connected_servers'] == 0
            assert stats['total_capabilities'] == 0
            assert stats['total_tools'] == 0
            assert stats['total_resources'] == 0
            assert stats['total_prompts'] == 0
            assert stats['dynamic_tools_registered'] == 0
            assert stats['total_operations'] == 0
            assert stats['successful_operations'] == 0
            assert stats['success_rate'] == 0
            assert stats['mcp_sdk_available'] is True
            assert stats['transport_types'] == []
    
    def test_get_statistics_with_data(self):
        """Test get_statistics with servers and executions"""
        skill = MCPSkill()
        
        # Add test servers
        server1 = MCPServerConfig(
            name="server1",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        server1.connected = True
        server1.available_tools = [Mock(), Mock()]  # 2 tools
        server1.available_resources = [Mock()]  # 1 resource
        server1.available_prompts = []  # 0 prompts
        
        server2 = MCPServerConfig(
            name="server2",
            transport=MCPTransport.SSE,
            url="https://api.example.com/sse"
        )
        server2.connected = False
        server2.available_tools = [Mock()]  # 1 tool
        server2.available_resources = []  # 0 resources
        server2.available_prompts = [Mock(), Mock()]  # 2 prompts
        
        skill.servers = {"server1": server1, "server2": server2}
        
        # Add test executions
        skill.execution_history = [
            MCPExecution(
                timestamp=datetime.utcnow(),
                server_name="server1",
                operation_type="tool",
                operation_name="tool1",
                parameters={},
                result="success",
                duration_ms=100.0,
                success=True
            ),
            MCPExecution(
                timestamp=datetime.utcnow(),
                server_name="server1",
                operation_type="tool",
                operation_name="tool2",
                parameters={},
                result=None,
                duration_ms=200.0,
                success=False,
                error="Failed"
            ),
            MCPExecution(
                timestamp=datetime.utcnow(),
                server_name="server2",
                operation_type="resource",
                operation_name="resource1",
                parameters={},
                result="data",
                duration_ms=150.0,
                success=True
            )
        ]
        
        # Add dynamic tools
        skill.dynamic_tools = {
            "server1_tool1": Mock(),
            "server1_tool2": Mock(),
            "server2_tool1": Mock()
        }
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            stats = skill.get_statistics()
            
            assert stats['total_servers'] == 2
            assert stats['connected_servers'] == 1  # Only server1
            assert stats['total_capabilities'] == 6  # 2+1+0 + 1+0+2
            assert stats['total_tools'] == 3  # 2 + 1
            assert stats['total_resources'] == 1  # 1 + 0
            assert stats['total_prompts'] == 2  # 0 + 2
            assert stats['dynamic_tools_registered'] == 3
            assert stats['total_operations'] == 3
            assert stats['successful_operations'] == 2
            assert stats['success_rate'] == 2/3  # 2 out of 3 successful
            assert stats['mcp_sdk_available'] is True
            assert set(stats['transport_types']) == {'http', 'sse'}


class TestMCPSkillCleanup:
    """Test MCPSkill cleanup and resource management"""
    
    @pytest.mark.asyncio
    async def test_cleanup_cancel_background_tasks(self):
        """Test cleanup cancels background tasks"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        # Mock running tasks
        mock_monitoring_task = Mock()
        mock_monitoring_task.cancel = Mock()
        mock_monitoring_task.cancelled.return_value = True
        skill._monitoring_task = mock_monitoring_task
        
        mock_capability_task = Mock()
        mock_capability_task.cancel = Mock()
        mock_capability_task.cancelled.return_value = True
        skill._capability_refresh_task = mock_capability_task
        
        await skill.cleanup()
        
        # Should cancel both tasks
        mock_monitoring_task.cancel.assert_called_once()
        mock_capability_task.cancel.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_close_sessions(self):
        """Test cleanup closes MCP sessions"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        # Mock sessions with __aexit__ method
        mock_session1 = Mock()
        mock_session1.__aexit__ = AsyncMock()
        
        mock_session2 = Mock()
        mock_session2.__aexit__ = AsyncMock()
        
        skill.sessions = {
            'server1': mock_session1,
            'server2': mock_session2
        }
        
        await skill.cleanup()
        
        # Should close all sessions
        mock_session1.__aexit__.assert_called_once_with(None, None, None)
        mock_session2.__aexit__.assert_called_once_with(None, None, None)
        
        # Should clear sessions dict
        assert skill.sessions == {}
    
    @pytest.mark.asyncio
    async def test_cleanup_handle_session_errors(self):
        """Test cleanup handles session closing errors gracefully"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        # Mock session that raises exception on close
        mock_session = Mock()
        mock_session.__aexit__ = AsyncMock(side_effect=Exception("Close failed"))
        
        skill.sessions = {'error-server': mock_session}
        
        await skill.cleanup()
        
        # Should handle error gracefully and log warning
        skill.logger.warning.assert_called_with("Error closing MCP session: Close failed")
        assert skill.sessions == {}
    
    @pytest.mark.asyncio
    async def test_cleanup_session_without_aexit(self):
        """Test cleanup handles sessions without __aexit__ method"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        # Mock session without __aexit__ method
        mock_session = Mock()
        # Remove __aexit__ attribute
        if hasattr(mock_session, '__aexit__'):
            delattr(mock_session, '__aexit__')
        
        skill.sessions = {'simple-server': mock_session}
        
        await skill.cleanup()
        
        # Should handle gracefully without errors
        assert skill.sessions == {}
        skill.logger.info.assert_called_with("MCP skill cleaned up")
    
    @pytest.mark.asyncio
    async def test_cleanup_no_logger(self):
        """Test cleanup when logger is not set"""
        skill = MCPSkill()
        skill.logger = None  # No logger
        
        # Should not raise exception
        await skill.cleanup()


class TestMCPSkillEdgeCases:
    """Test MCPSkill edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_initialize_with_invalid_server_config(self):
        """Test initialization with malformed server configuration"""
        config = {
            'servers': [
                {
                    # Missing required fields
                    'url': 'http://localhost:8080/mcp'
                    # Missing 'name' and 'transport'
                },
                {
                    'name': 'valid-server',
                    'url': 'http://localhost:8081/mcp',
                    'transport': 'http'
                }
            ]
        }
        
        skill = MCPSkill(config)
        mock_agent = Mock()
        mock_agent.name = "test-agent"
        
        with patch('webagents.agents.skills.core.mcp.skill.MCP_AVAILABLE', True):
            with patch('webagents.utils.logging.get_logger'):
                with patch.object(skill, '_register_mcp_server', new_callable=AsyncMock) as mock_register:
                    mock_register.side_effect = [Exception("Invalid config"), True]
                    
                    await skill.initialize(mock_agent)
                    
                    # Should attempt to register both servers
                    assert mock_register.call_count == 2
    
    def test_mcp_skill_scope_configuration(self):
        """Test MCPSkill scope configuration"""
        skill = MCPSkill()
        
        # Should default to "all" scope
        assert skill.scope == "all"
    
    @pytest.mark.asyncio 
    async def test_dynamic_tool_execution_integration(self):
        """Test that dynamic tools actually call _execute_mcp_tool"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.register_tool = Mock()
        skill.agent = mock_agent
        
        server = MCPServerConfig(
            name="integration-server",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        
        # Mock MCP tool
        mock_tool = Mock()
        mock_tool.name = "integration_tool"
        mock_tool.description = "Integration test tool"
        mock_tool.inputSchema = Mock()
        mock_tool.inputSchema.model_dump.return_value = {"type": "object", "properties": {}}
        
        with patch.object(skill, '_execute_mcp_tool', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Tool executed successfully"
            
            await skill._register_dynamic_tool(server, mock_tool)
            
            # Get the registered dynamic tool
            dynamic_name = "integration-server_integration_tool"
            dynamic_tool = skill.dynamic_tools[dynamic_name]
            
            # Execute the dynamic tool
            result = await dynamic_tool(param1="test", param2=42)
            
            # Should call _execute_mcp_tool with correct parameters
            mock_execute.assert_called_once_with(
                "integration-server", 
                "integration_tool", 
                {"param1": "test", "param2": 42}
            )
            assert result == "Tool executed successfully"
    
    def test_transport_methods_creation(self):
        """Test creation of transport-specific methods"""
        skill = MCPSkill()
        
        # Test _create_list_tools_method
        receive_stream = Mock()
        send_stream = Mock()
        
        list_tools_method = skill._create_list_tools_method(receive_stream, send_stream)
        
        # Should return a callable
        assert callable(list_tools_method)
    
    @pytest.mark.asyncio
    async def test_http_transport_session_creation(self):
        """Test HTTP transport session object creation"""
        skill = MCPSkill()
        skill.logger = Mock()
        
        server = MCPServerConfig(
            name="http-session-test",
            transport=MCPTransport.HTTP,
            url="http://localhost:8080/mcp"
        )
        
        # Mock the async generator from create_http_client
        mock_receive = Mock()
        mock_send = Mock()
        mock_get_session_id = Mock()
        
        async def mock_client_generator():
            yield (mock_receive, mock_send, mock_get_session_id)
        
        with patch('webagents.agents.skills.core.mcp.skill.create_http_client') as mock_create_http:
            mock_create_http.return_value = mock_client_generator()
            
            with patch.object(skill, '_discover_capabilities', new_callable=AsyncMock):
                result = await skill._connect_to_server(server)
                
                assert result is True
                
                # Check session was created with expected attributes
                session = skill.sessions['http-session-test']
                assert hasattr(session, 'receive_stream')
                assert hasattr(session, 'send_stream')
                assert hasattr(session, 'get_session_id')
                assert hasattr(session, 'list_tools')
                assert hasattr(session, 'list_resources')
                assert hasattr(session, 'list_prompts')
                assert hasattr(session, 'call_tool')
                
                assert session.receive_stream == mock_receive
                assert session.send_stream == mock_send
                assert session.get_session_id == mock_get_session_id


# MCP Skill tests - run directly with pytest
# Example: pytest tests/skills/core/test_mcp_skill.py -v 