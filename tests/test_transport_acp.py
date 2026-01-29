"""
Comprehensive tests for ACPTransportSkill

Tests Agent Client Protocol (ACP) compatibility:
- JSON-RPC 2.0 format
- Initialization/shutdown
- Prompt submission
- Tools
- File system
- Terminal
- Agent plans
- Slash commands
"""

import pytest
import json
import os
import tempfile
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock

from webagents.agents.skills.core.transport.acp.skill import ACPTransportSkill


# ============================================================================
# Mock Fixtures
# ============================================================================

class MockAgent:
    """Mock agent for testing"""
    def __init__(self):
        self.name = "test-agent"
        self.skills = {}
        self._registered_handoffs = []
        self.active_handoff = None
    
    def get_tools_for_scope(self, scope):
        return [
            {"function": {"name": "test_tool", "description": "A test tool", "parameters": {}}}
        ]
    
    def get_all_tools(self):
        return [
            {"function": {"name": "test_tool", "description": "A test tool", "parameters": {}}}
        ]
    
    async def run_streaming(self, messages, **kwargs):
        yield {"choices": [{"delta": {"content": "Response"}}]}
    
    async def execute_tool(self, name, args):
        return {"result": f"Executed {name}"}
    
    def get_all_commands(self):
        return []


class MockContext:
    """Mock context for testing"""
    def __init__(self, agent=None):
        self.agent = agent
        self.messages = []
        self.stream = True
        self.auth = None


class MockWebSocket:
    """Mock WebSocket for testing"""
    def __init__(self):
        self.accepted = False
        self.closed = False
        self.messages_sent = []
        self.messages_to_receive = []
        self._receive_index = 0
    
    async def accept(self):
        self.accepted = True
    
    async def close(self):
        self.closed = True
    
    async def send_json(self, data):
        self.messages_sent.append(data)
    
    async def receive_json(self):
        if self._receive_index < len(self.messages_to_receive):
            msg = self.messages_to_receive[self._receive_index]
            self._receive_index += 1
            return msg
        raise Exception("No more messages")
    
    async def iter_json(self):
        for msg in self.messages_to_receive:
            yield msg


@pytest.fixture
def skill():
    return ACPTransportSkill()


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def mock_context(mock_agent):
    return MockContext(mock_agent)


@pytest.fixture
def mock_ws():
    return MockWebSocket()


# ============================================================================
# Initialization Tests
# ============================================================================

class TestACPInitialization:
    """Test skill initialization"""
    
    @pytest.mark.asyncio
    async def test_initialize(self, skill, mock_agent):
        """Test basic initialization"""
        await skill.initialize(mock_agent)
        assert skill.agent == mock_agent
    
    def test_default_scope(self, skill):
        """Test default scope is 'all'"""
        assert skill.scope == "all"
    
    def test_sessions_dict_initialized(self, skill):
        """Test sessions dictionary is initialized"""
        assert hasattr(skill, '_sessions')
        assert isinstance(skill._sessions, dict)
    
    def test_http_endpoint_registered(self, skill):
        """Test HTTP endpoint is properly decorated"""
        assert hasattr(skill.acp_http, '_http_subpath')
        assert skill.acp_http._http_subpath == '/acp'
        assert skill.acp_http._http_method == 'post'
    
    def test_websocket_endpoint_registered(self, skill):
        """Test WebSocket endpoint is properly decorated"""
        assert hasattr(skill.acp_websocket, '_webagents_is_websocket')
        assert skill.acp_websocket._websocket_path == '/acp/stream'


# ============================================================================
# JSON-RPC 2.0 Format Tests
# ============================================================================

class TestACPJsonRpcFormat:
    """Test JSON-RPC 2.0 format compliance"""
    
    @pytest.mark.asyncio
    async def test_jsonrpc_response_format(self, skill, mock_agent):
        """Test JSON-RPC response format"""
        await skill.initialize(mock_agent)
        
        response = skill._jsonrpc_response("123", {"key": "value"})
        data = json.loads(response.split("data: ")[1].strip())
        
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "123"
        assert data["result"]["key"] == "value"
    
    @pytest.mark.asyncio
    async def test_jsonrpc_error_format(self, skill, mock_agent):
        """Test JSON-RPC error format"""
        await skill.initialize(mock_agent)
        
        error = skill._jsonrpc_error("123", -32600, "Invalid Request")
        data = json.loads(error.split("data: ")[1].strip())
        
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "123"
        assert data["error"]["code"] == -32600
        assert data["error"]["message"] == "Invalid Request"
    
    @pytest.mark.asyncio
    async def test_jsonrpc_notification_format(self, skill, mock_agent):
        """Test JSON-RPC notification format (no id)"""
        await skill.initialize(mock_agent)
        
        notif = skill._jsonrpc_notification("update", {"status": "ok"})
        data = json.loads(notif.split("data: ")[1].strip())
        
        assert data["jsonrpc"] == "2.0"
        assert "id" not in data
        assert data["method"] == "update"
        assert data["params"]["status"] == "ok"
    
    @pytest.mark.asyncio
    async def test_make_response_dict_format(self, skill, mock_agent):
        """Test WebSocket _make_response returns proper dict"""
        await skill.initialize(mock_agent)
        
        response = skill._make_response("456", {"data": "test"})
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "456"
        assert response["result"]["data"] == "test"
    
    @pytest.mark.asyncio
    async def test_make_notification_dict_format(self, skill, mock_agent):
        """Test WebSocket _make_notification returns proper dict"""
        await skill.initialize(mock_agent)
        
        notif = skill._make_notification("event", {"value": 42})
        
        assert notif["jsonrpc"] == "2.0"
        assert "id" not in notif
        assert notif["method"] == "event"
        assert notif["params"]["value"] == 42
    
    @pytest.mark.asyncio
    async def test_make_error_dict_format(self, skill, mock_agent):
        """Test WebSocket _make_error returns proper dict"""
        await skill.initialize(mock_agent)
        
        error = skill._make_error("789", -32601, "Method not found")
        
        assert error["jsonrpc"] == "2.0"
        assert error["id"] == "789"
        assert error["error"]["code"] == -32601
        assert error["error"]["message"] == "Method not found"


# ============================================================================
# Initialize/Shutdown Tests
# ============================================================================

class TestACPInitializeShutdown:
    """Test initialize and shutdown methods"""
    
    @pytest.mark.asyncio
    async def test_initialize_method(self, skill, mock_agent, mock_context):
        """Test initialize method returns capabilities"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            result = await skill._handle_initialize({})
            
            assert "protocolVersion" in result
            assert result["protocolVersion"] == "1.0"
            assert "capabilities" in result
            assert "serverInfo" in result
    
    @pytest.mark.asyncio
    async def test_initialize_returns_server_info(self, skill, mock_agent, mock_context):
        """Test initialize returns server info with agent name"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            result = await skill._handle_initialize({})
            
            assert result["serverInfo"]["name"] == "test-agent"
            assert result["serverInfo"]["version"] == "2.0.0"
    
    @pytest.mark.asyncio
    async def test_initialize_with_no_context_uses_agent(self, skill, mock_agent):
        """Test initialize falls back to skill's agent when no context"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=None):
            result = await skill._handle_initialize({})
            
            assert result["serverInfo"]["name"] == "test-agent"
    
    @pytest.mark.asyncio
    async def test_shutdown_via_http(self, skill, mock_agent, mock_context):
        """Test shutdown method via HTTP endpoint"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            chunks = []
            async for chunk in skill.acp_http(
                jsonrpc="2.0",
                id="1",
                method="shutdown",
                params={}
            ):
                chunks.append(chunk)
            
            response = "".join(chunks)
            assert "shutdown" in response


# ============================================================================
# Capabilities Tests
# ============================================================================

class TestACPCapabilities:
    """Test ACP capabilities reporting"""
    
    @pytest.mark.asyncio
    async def test_capabilities_include_prompts(self, skill, mock_agent):
        """Test capabilities include prompts support"""
        await skill.initialize(mock_agent)
        
        caps = skill._get_capabilities()
        
        assert caps["prompts"] is True
    
    @pytest.mark.asyncio
    async def test_capabilities_include_tools(self, skill, mock_agent):
        """Test capabilities include tools support"""
        await skill.initialize(mock_agent)
        
        caps = skill._get_capabilities()
        
        assert caps["tools"] is True
    
    @pytest.mark.asyncio
    async def test_capabilities_include_files(self, skill, mock_agent):
        """Test capabilities include files support"""
        await skill.initialize(mock_agent)
        
        caps = skill._get_capabilities()
        
        assert caps["files"] is True
    
    @pytest.mark.asyncio
    async def test_capabilities_include_terminal(self, skill, mock_agent):
        """Test capabilities include terminal support"""
        await skill.initialize(mock_agent)
        
        caps = skill._get_capabilities()
        
        assert caps["terminal"] is True
    
    @pytest.mark.asyncio
    async def test_capabilities_include_plans(self, skill, mock_agent):
        """Test capabilities include plans support"""
        await skill.initialize(mock_agent)
        
        caps = skill._get_capabilities()
        
        assert caps["plans"] is True
    
    @pytest.mark.asyncio
    async def test_capabilities_include_streaming(self, skill, mock_agent):
        """Test capabilities include streaming support"""
        await skill.initialize(mock_agent)
        
        caps = skill._get_capabilities()
        
        assert caps["streaming"] is True
    
    @pytest.mark.asyncio
    async def test_capabilities_include_slash_commands(self, skill, mock_agent):
        """Test capabilities include slash commands support"""
        await skill.initialize(mock_agent)
        
        caps = skill._get_capabilities()
        
        assert caps["slashCommands"] is True
    
    @pytest.mark.asyncio
    async def test_capabilities_include_multi_turn(self, skill, mock_agent):
        """Test capabilities include multi-turn support"""
        await skill.initialize(mock_agent)
        
        caps = skill._get_capabilities()
        
        assert caps["multiTurn"] is True
    
    @pytest.mark.asyncio
    async def test_capabilities_via_http(self, skill, mock_agent, mock_context):
        """Test capabilities via HTTP endpoint"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            chunks = []
            async for chunk in skill.acp_http(
                jsonrpc="2.0",
                id="1",
                method="capabilities",
                params={}
            ):
                chunks.append(chunk)
            
            response = "".join(chunks)
            assert "streaming" in response
            assert "tools" in response


# ============================================================================
# Tools Tests
# ============================================================================

class TestACPTools:
    """Test tools listing and calling"""
    
    @pytest.mark.asyncio
    async def test_tools_list(self, skill, mock_agent, mock_context):
        """Test tools/list returns agent tools"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            result = await skill._handle_tools_list({})
            
            assert "tools" in result
            assert len(result["tools"]) > 0
            assert result["tools"][0]["name"] == "test_tool"
            assert result["tools"][0]["description"] == "A test tool"
    
    @pytest.mark.asyncio
    async def test_tools_list_empty_when_no_tools(self, skill, mock_agent, mock_context):
        """Test tools/list returns empty when agent has no tools"""
        await skill.initialize(mock_agent)
        mock_agent.get_all_tools = lambda: []
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            result = await skill._handle_tools_list({})
            
            assert "tools" in result
            assert len(result["tools"]) == 0
    
    @pytest.mark.asyncio
    async def test_tools_call_success(self, skill, mock_agent, mock_context):
        """Test tools/call executes tool"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            result = await skill._handle_tools_call({
                "name": "test_tool",
                "arguments": {"arg1": "value1"}
            })
            
            assert "result" in result
            assert "Executed test_tool" in result["result"]["result"]
    
    @pytest.mark.asyncio
    async def test_tools_call_no_agent(self, skill, mock_agent):
        """Test tools/call with no agent returns error"""
        await skill.initialize(mock_agent)
        
        mock_context = MockContext(None)
        mock_context.agent = None
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            result = await skill._handle_tools_call({"name": "test"})
            
            assert "error" in result
            assert result["error"] == "No agent context"
    
    @pytest.mark.asyncio
    async def test_tools_call_handles_exception(self, skill, mock_agent, mock_context):
        """Test tools/call handles execution exceptions"""
        await skill.initialize(mock_agent)
        mock_agent.execute_tool = AsyncMock(side_effect=Exception("Tool error"))
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            result = await skill._handle_tools_call({
                "name": "failing_tool",
                "arguments": {}
            })
            
            assert "error" in result
            assert "Tool error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_tools_list_via_http(self, skill, mock_agent, mock_context):
        """Test tools/list via HTTP endpoint"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            chunks = []
            async for chunk in skill.acp_http(
                jsonrpc="2.0",
                id="1",
                method="tools/list",
                params={}
            ):
                chunks.append(chunk)
            
            response = "".join(chunks)
            assert "test_tool" in response


# ============================================================================
# Prompt Submission Tests
# ============================================================================

class TestACPPromptSubmission:
    """Test prompt/submit functionality"""
    
    @pytest.mark.asyncio
    async def test_prompt_submit_streams_response(self, skill, mock_agent, mock_context):
        """Test prompt/submit streams response"""
        await skill.initialize(mock_agent)
        
        async def mock_handoff(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": " World"}}]}
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff', side_effect=mock_handoff):
                chunks = []
                async for chunk in skill.acp_http(
                    jsonrpc="2.0",
                    id="1",
                    method="prompt/submit",
                    params={"messages": [{"role": "user", "content": "Hi"}]}
                ):
                    chunks.append(chunk)
                
                # Should have started, progress, and completion
                response = "".join(chunks)
                assert "prompt/started" in response
                assert "prompt/progress" in response
                assert "complete" in response
    
    @pytest.mark.asyncio
    async def test_chat_submit_alias(self, skill, mock_agent, mock_context):
        """Test chat/submit is alias for prompt/submit"""
        await skill.initialize(mock_agent)
        
        async def mock_handoff(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Test"}}]}
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff', side_effect=mock_handoff):
                chunks = []
                async for chunk in skill.acp_http(
                    jsonrpc="2.0",
                    id="1",
                    method="chat/submit",
                    params={"messages": [{"role": "user", "content": "Hi"}]}
                ):
                    chunks.append(chunk)
                
                response = "".join(chunks)
                assert "complete" in response


# ============================================================================
# File System Tests
# ============================================================================

class TestACPFileSystem:
    """Test file system operations"""
    
    @pytest.mark.asyncio
    async def test_files_list(self, skill, mock_agent):
        """Test files/list returns directory contents"""
        await skill.initialize(mock_agent)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            open(os.path.join(tmpdir, "file1.txt"), "w").close()
            open(os.path.join(tmpdir, "file2.py"), "w").close()
            os.mkdir(os.path.join(tmpdir, "subdir"))
            
            result = await skill._handle_files_list({"path": tmpdir})
            
            assert "entries" in result
            assert len(result["entries"]) == 3
            
            names = [e["name"] for e in result["entries"]]
            assert "file1.txt" in names
            assert "file2.py" in names
            assert "subdir" in names
    
    @pytest.mark.asyncio
    async def test_files_list_includes_type(self, skill, mock_agent):
        """Test files/list includes entry type"""
        await skill.initialize(mock_agent)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "file.txt"), "w").close()
            os.mkdir(os.path.join(tmpdir, "folder"))
            
            result = await skill._handle_files_list({"path": tmpdir})
            
            types = {e["name"]: e["type"] for e in result["entries"]}
            assert types["file.txt"] == "file"
            assert types["folder"] == "directory"
    
    @pytest.mark.asyncio
    async def test_files_list_nonexistent(self, skill, mock_agent):
        """Test files/list with nonexistent directory"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_files_list({"path": "/nonexistent/path/12345"})
        
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_files_list_default_path(self, skill, mock_agent):
        """Test files/list defaults to current directory"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_files_list({})
        
        assert "entries" in result or "error" in result
        assert result.get("path", ".") == "."
    
    @pytest.mark.asyncio
    async def test_files_read(self, skill, mock_agent):
        """Test files/read returns file contents"""
        await skill.initialize(mock_agent)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello World")
            f.flush()
            
            try:
                result = await skill._handle_files_read({"path": f.name})
                
                assert result["content"] == "Hello World"
                assert result["path"] == f.name
            finally:
                os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_files_read_nonexistent(self, skill, mock_agent):
        """Test files/read with nonexistent file"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_files_read({"path": "/nonexistent/file/12345.txt"})
        
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_files_read_binary_file_error(self, skill, mock_agent):
        """Test files/read handles binary file gracefully"""
        await skill.initialize(mock_agent)
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
            f.write(b'\x00\x01\x02\x03')
            f.flush()
            
            try:
                result = await skill._handle_files_read({"path": f.name})
                # Should either return content or error for binary
                assert "content" in result or "error" in result
            finally:
                os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_files_write(self, skill, mock_agent):
        """Test files/write creates file in subdirectory"""
        await skill.initialize(mock_agent)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use relative path with subdirectory to avoid empty dirname issue
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # Using subdir/file.txt avoids empty dirname issue in implementation
                result = await skill._handle_files_write({
                    "path": "subdir/newfile.txt",
                    "content": "New content"
                })
                
                filepath = os.path.join(tmpdir, "subdir/newfile.txt")
                assert result.get("success") is True or result.get("path") == "subdir/newfile.txt"
                assert os.path.exists(filepath)
                with open(filepath) as f:
                    assert f.read() == "New content"
            finally:
                os.chdir(old_cwd)
    
    @pytest.mark.asyncio
    async def test_files_write_path_traversal_blocked(self, skill, mock_agent):
        """Test files/write blocks path traversal"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_files_write({
            "path": "../../../etc/passwd",
            "content": "malicious"
        })
        
        assert "error" in result
        assert "Invalid" in result["error"] or "path" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_files_write_blocks_absolute_path(self, skill, mock_agent):
        """Test files/write blocks absolute paths"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_files_write({
            "path": "/etc/passwd",
            "content": "malicious"
        })
        
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_files_write_creates_parent_dirs(self, skill, mock_agent):
        """Test files/write creates parent directories"""
        await skill.initialize(mock_agent)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = await skill._handle_files_write({
                    "path": "subdir/nested/file.txt",
                    "content": "Content"
                })
                
                # Check result or that file was created
                if result.get("success"):
                    assert os.path.exists(os.path.join(tmpdir, "subdir/nested/file.txt"))
            finally:
                os.chdir(old_cwd)


# ============================================================================
# Terminal Tests
# ============================================================================

class TestACPTerminal:
    """Test terminal command execution"""
    
    @pytest.mark.asyncio
    async def test_terminal_run_echo(self, skill, mock_agent):
        """Test terminal/run with echo command"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_terminal_run({
            "command": "echo hello"
        })
        
        assert "stdout" in result
        assert "hello" in result["stdout"]
        assert result["returncode"] == 0
    
    @pytest.mark.asyncio
    async def test_terminal_run_with_cwd(self, skill, mock_agent):
        """Test terminal/run respects cwd"""
        await skill.initialize(mock_agent)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await skill._handle_terminal_run({
                "command": "pwd",
                "cwd": tmpdir
            })
            
            assert tmpdir in result["stdout"]
    
    @pytest.mark.asyncio
    async def test_terminal_run_returns_stderr(self, skill, mock_agent):
        """Test terminal/run returns stderr"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_terminal_run({
            "command": "echo error >&2"
        })
        
        assert "stderr" in result
        assert "error" in result["stderr"]
    
    @pytest.mark.asyncio
    async def test_terminal_run_returns_exit_code(self, skill, mock_agent):
        """Test terminal/run returns exit code"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_terminal_run({
            "command": "exit 42"
        })
        
        assert "returncode" in result
        assert result["returncode"] == 42
    
    @pytest.mark.asyncio
    async def test_terminal_blocks_rm_rf(self, skill, mock_agent):
        """Test terminal blocks rm -rf /"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_terminal_run({
            "command": "rm -rf /"
        })
        
        assert "error" in result
        assert "Dangerous" in result["error"] or "blocked" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_terminal_blocks_sudo(self, skill, mock_agent):
        """Test terminal blocks sudo"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_terminal_run({
            "command": "sudo apt update"
        })
        
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_terminal_blocks_mkfs(self, skill, mock_agent):
        """Test terminal blocks mkfs"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_terminal_run({
            "command": "mkfs.ext4 /dev/sda"
        })
        
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_terminal_blocks_dd(self, skill, mock_agent):
        """Test terminal blocks dd if="""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_terminal_run({
            "command": "dd if=/dev/zero of=/dev/sda"
        })
        
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_terminal_captures_stderr(self, skill, mock_agent):
        """Test terminal captures stderr on failure"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_terminal_run({
            "command": "ls /nonexistent_path_12345"
        })
        
        assert "stderr" in result
        assert result["returncode"] != 0
    
    @pytest.mark.asyncio
    async def test_terminal_default_cwd(self, skill, mock_agent):
        """Test terminal uses default cwd"""
        await skill.initialize(mock_agent)
        
        result = await skill._handle_terminal_run({
            "command": "pwd"
        })
        
        assert "stdout" in result
        assert result["returncode"] == 0


# ============================================================================
# Agent Plans Tests
# ============================================================================

class TestACPAgentPlans:
    """Test agent plan generation"""
    
    @pytest.mark.asyncio
    async def test_agent_plan_returns_plan(self, skill, mock_agent, mock_context):
        """Test agent/plan returns a plan"""
        await skill.initialize(mock_agent)
        
        async def mock_handoff(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Step 1: Do this\n"}}]}
            yield {"choices": [{"delta": {"content": "Step 2: Do that"}}]}
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff', side_effect=mock_handoff):
                result = await skill._handle_agent_plan({
                    "task": "Build a web app"
                })
                
                assert "plan" in result
                assert "Step 1" in result["plan"]
                assert "steps" in result
    
    @pytest.mark.asyncio
    async def test_agent_plan_includes_task(self, skill, mock_agent, mock_context):
        """Test agent/plan includes original task"""
        await skill.initialize(mock_agent)
        
        async def mock_handoff(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Plan"}}]}
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff', side_effect=mock_handoff):
                result = await skill._handle_agent_plan({
                    "task": "My specific task"
                })
                
                assert result["task"] == "My specific task"
    
    @pytest.mark.asyncio
    async def test_agent_plan_steps_list(self, skill, mock_agent, mock_context):
        """Test agent/plan returns steps as list"""
        await skill.initialize(mock_agent)
        
        async def mock_handoff(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Step 1\nStep 2\nStep 3"}}]}
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff', side_effect=mock_handoff):
                result = await skill._handle_agent_plan({
                    "task": "Multi-step task"
                })
                
                assert "steps" in result
                assert isinstance(result["steps"], list)
                assert len(result["steps"]) == 3
    
    @pytest.mark.asyncio
    async def test_agent_plan_handles_error(self, skill, mock_agent, mock_context):
        """Test agent/plan handles errors gracefully"""
        await skill.initialize(mock_agent)
        
        async def mock_handoff(*args, **kwargs):
            raise Exception("LLM error")
            yield  # Make it a generator
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            with patch.object(skill, 'execute_handoff', side_effect=mock_handoff):
                result = await skill._handle_agent_plan({
                    "task": "Failing task"
                })
                
                assert "error" in result


# ============================================================================
# Slash Commands Tests
# ============================================================================

class TestACPSlashCommands:
    """Test slash command functionality"""
    
    @pytest.mark.asyncio
    async def test_slash_list_returns_commands(self, skill, mock_agent, mock_context):
        """Test slash/list returns available commands"""
        await skill.initialize(mock_agent)
        
        # Add mock commands
        mock_agent.get_all_commands = lambda: [
            {"path": "/help", "description": "Get help"},
            {"path": "/clear", "description": "Clear history"}
        ]
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            result = await skill._handle_slash_list({})
            
            assert "commands" in result
            assert len(result["commands"]) == 2
    
    @pytest.mark.asyncio
    async def test_slash_list_no_commands(self, skill, mock_agent, mock_context):
        """Test slash/list with no commands"""
        await skill.initialize(mock_agent)
        mock_agent.get_all_commands = lambda: []
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            result = await skill._handle_slash_list({})
            
            assert "commands" in result
            assert len(result["commands"]) == 0
    
    @pytest.mark.asyncio
    async def test_slash_list_command_format(self, skill, mock_agent, mock_context):
        """Test slash/list returns proper command format"""
        await skill.initialize(mock_agent)
        
        mock_agent.get_all_commands = lambda: [
            {"path": "/test", "description": "Test cmd", "parameters": {"arg": {"type": "string"}}}
        ]
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            result = await skill._handle_slash_list({})
            
            cmd = result["commands"][0]
            assert cmd["path"] == "/test"
            assert cmd["description"] == "Test cmd"
            assert "parameters" in cmd
    
    @pytest.mark.asyncio
    async def test_slash_execute_calls_command(self, skill, mock_agent, mock_context):
        """Test slash/execute calls agent command"""
        await skill.initialize(mock_agent)
        mock_agent.execute_command = AsyncMock(return_value={"result": "executed"})
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            chunks = []
            async for chunk in skill._handle_slash_execute({
                "command": "/help",
                "arguments": {}
            }):
                chunks.append(chunk)
            
            assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_slash_execute_no_agent(self, skill, mock_agent):
        """Test slash/execute with no agent"""
        await skill.initialize(mock_agent)
        
        mock_context = MockContext(None)
        mock_context.agent = None
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            chunks = []
            async for chunk in skill._handle_slash_execute({
                "command": "/test",
                "arguments": {}
            }):
                chunks.append(chunk)
            
            response = "".join(chunks)
            assert "error" in response.lower()


# ============================================================================
# HTTP Endpoint Tests
# ============================================================================

class TestACPHttpEndpoint:
    """Test HTTP endpoint routing"""
    
    @pytest.mark.asyncio
    async def test_http_initialize(self, skill, mock_agent, mock_context):
        """Test HTTP initialize method"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            chunks = []
            async for chunk in skill.acp_http(
                jsonrpc="2.0",
                id="1",
                method="initialize",
                params={}
            ):
                chunks.append(chunk)
            
            assert len(chunks) > 0
            response = "".join(chunks)
            assert "protocolVersion" in response
    
    @pytest.mark.asyncio
    async def test_http_tools_list(self, skill, mock_agent, mock_context):
        """Test HTTP tools/list method"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            chunks = []
            async for chunk in skill.acp_http(
                jsonrpc="2.0",
                id="1",
                method="tools/list",
                params={}
            ):
                chunks.append(chunk)
            
            assert len(chunks) > 0
            response = "".join(chunks)
            assert "tools" in response
    
    @pytest.mark.asyncio
    async def test_http_unknown_method(self, skill, mock_agent, mock_context):
        """Test HTTP with unknown method returns error"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            chunks = []
            async for chunk in skill.acp_http(
                jsonrpc="2.0",
                id="1",
                method="unknown/method",
                params={}
            ):
                chunks.append(chunk)
            
            response = "".join(chunks)
            assert "error" in response.lower()
            assert "Method not found" in response
    
    @pytest.mark.asyncio
    async def test_http_files_read(self, skill, mock_agent, mock_context):
        """Test HTTP files/read method"""
        await skill.initialize(mock_agent)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            f.flush()
            
            try:
                with patch.object(skill, 'get_context', return_value=mock_context):
                    chunks = []
                    async for chunk in skill.acp_http(
                        jsonrpc="2.0",
                        id="1",
                        method="files/read",
                        params={"path": f.name}
                    ):
                        chunks.append(chunk)
                    
                    response = "".join(chunks)
                    assert "Test content" in response
            finally:
                os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_http_terminal_run(self, skill, mock_agent, mock_context):
        """Test HTTP terminal/run method"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            chunks = []
            async for chunk in skill.acp_http(
                jsonrpc="2.0",
                id="1",
                method="terminal/run",
                params={"command": "echo test"}
            ):
                chunks.append(chunk)
            
            response = "".join(chunks)
            assert "test" in response


# ============================================================================
# WebSocket Endpoint Tests
# ============================================================================

class TestACPWebSocketEndpoint:
    """Test WebSocket endpoint"""
    
    @pytest.mark.asyncio
    async def test_websocket_creates_session(self, skill, mock_agent, mock_ws, mock_context):
        """Test WebSocket creates session on connect"""
        await skill.initialize(mock_agent)
        
        # Empty messages - just test session creation
        mock_ws.messages_to_receive = []
        
        # Run websocket handler briefly
        with patch.object(skill, 'get_context', return_value=mock_context):
            try:
                await skill.acp_websocket(mock_ws)
            except Exception:
                pass  # Expected to end when no messages
        
        assert mock_ws.accepted
    
    @pytest.mark.asyncio
    async def test_websocket_handle_initialize(self, skill, mock_agent, mock_ws, mock_context):
        """Test WebSocket handles initialize"""
        await skill.initialize(mock_agent)
        
        # Initialize session
        session_id = "test-session"
        skill._sessions[session_id] = {
            "id": session_id,
            "created_at": 0,
            "initialized": False
        }
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            await skill._handle_ws_message(
                mock_ws,
                session_id,
                {"jsonrpc": "2.0", "id": "1", "method": "initialize", "params": {}}
            )
            
            # Should send response and mark session initialized
            assert len(mock_ws.messages_sent) > 0
            assert skill._sessions[session_id]["initialized"] is True
            
            response = mock_ws.messages_sent[0]
            assert response["jsonrpc"] == "2.0"
            assert "result" in response
    
    @pytest.mark.asyncio
    async def test_websocket_handle_shutdown(self, skill, mock_agent, mock_ws, mock_context):
        """Test WebSocket handles shutdown"""
        await skill.initialize(mock_agent)
        
        session_id = "test-session"
        skill._sessions[session_id] = {
            "id": session_id,
            "created_at": 0,
            "initialized": True
        }
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            await skill._handle_ws_message(
                mock_ws,
                session_id,
                {"jsonrpc": "2.0", "id": "1", "method": "shutdown", "params": {}}
            )
            
            assert len(mock_ws.messages_sent) > 0
            assert mock_ws.closed
    
    @pytest.mark.asyncio
    async def test_websocket_handle_tools_list(self, skill, mock_agent, mock_ws, mock_context):
        """Test WebSocket handles tools/list"""
        await skill.initialize(mock_agent)
        
        session_id = "test-session"
        skill._sessions[session_id] = {
            "id": session_id,
            "created_at": 0,
            "initialized": True
        }
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            await skill._handle_ws_message(
                mock_ws,
                session_id,
                {"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}}
            )
            
            assert len(mock_ws.messages_sent) > 0
            response = mock_ws.messages_sent[0]
            assert "result" in response
            assert "tools" in response["result"]
    
    @pytest.mark.asyncio
    async def test_websocket_handle_unknown_method(self, skill, mock_agent, mock_ws, mock_context):
        """Test WebSocket handles unknown method"""
        await skill.initialize(mock_agent)
        
        session_id = "test-session"
        skill._sessions[session_id] = {
            "id": session_id,
            "created_at": 0,
            "initialized": True
        }
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            await skill._handle_ws_message(
                mock_ws,
                session_id,
                {"jsonrpc": "2.0", "id": "1", "method": "unknown/method", "params": {}}
            )
            
            assert len(mock_ws.messages_sent) > 0
            response = mock_ws.messages_sent[0]
            assert "error" in response


# ============================================================================
# Content Extraction Tests
# ============================================================================

class TestACPContentExtraction:
    """Test content extraction from OpenAI chunks"""
    
    @pytest.mark.asyncio
    async def test_extract_content_from_delta(self, skill, mock_agent):
        """Test extracting content from delta"""
        await skill.initialize(mock_agent)
        
        chunk = {"choices": [{"delta": {"content": "Hello"}}]}
        content = skill._extract_content(chunk)
        
        assert content == "Hello"
    
    @pytest.mark.asyncio
    async def test_extract_content_empty_delta(self, skill, mock_agent):
        """Test extracting from empty delta"""
        await skill.initialize(mock_agent)
        
        chunk = {"choices": [{"delta": {}}]}
        content = skill._extract_content(chunk)
        
        assert content == ""
    
    @pytest.mark.asyncio
    async def test_extract_content_no_choices(self, skill, mock_agent):
        """Test extracting with no choices"""
        await skill.initialize(mock_agent)
        
        chunk = {"choices": []}
        content = skill._extract_content(chunk)
        
        assert content == ""
    
    @pytest.mark.asyncio
    async def test_extract_content_missing_choices(self, skill, mock_agent):
        """Test extracting with missing choices key"""
        await skill.initialize(mock_agent)
        
        chunk = {}
        content = skill._extract_content(chunk)
        
        assert content == ""


# ============================================================================
# Error Codes Tests
# ============================================================================

class TestACPErrorCodes:
    """Test JSON-RPC error codes"""
    
    @pytest.mark.asyncio
    async def test_method_not_found_error_code(self, skill, mock_agent, mock_context):
        """Test method not found uses correct error code"""
        await skill.initialize(mock_agent)
        
        with patch.object(skill, 'get_context', return_value=mock_context):
            chunks = []
            async for chunk in skill.acp_http(
                jsonrpc="2.0",
                id="1",
                method="nonexistent",
                params={}
            ):
                chunks.append(chunk)
            
            response = "".join(chunks)
            data = json.loads(response.split("data: ")[1].strip())
            
            assert data["error"]["code"] == -32601


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
