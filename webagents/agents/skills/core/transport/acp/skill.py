"""
ACP Transport Skill - WebAgents V2.0

Agent Client Protocol implementation for IDE integration.
https://agentclientprotocol.com/

Uses UAMP (Universal Agentic Message Protocol) for internal message representation.
"""

import json
import uuid
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, TYPE_CHECKING

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import http, websocket
from webagents.uamp import (
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ContentDelta,
    ResponseOutput,
)
from .uamp_adapter import ACPUAMPAdapter

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent
    from fastapi import WebSocket


class ACPTransportSkill(Skill):
    """
    Agent Client Protocol (ACP) transport for IDE integration.
    
    Implements JSON-RPC 2.0 over HTTP and WebSocket for communication
    with code editors like Cursor, Zed, and JetBrains IDEs.
    
    Endpoints:
    - POST /acp - JSON-RPC over HTTP (single request)
    - WS /acp/stream - JSON-RPC over WebSocket (streaming)
    
    Example:
        agent = BaseAgent(
            name="my-agent",
            skills=[ACPTransportSkill()]
        )
        
        # POST /agents/my-agent/acp
        # {"jsonrpc": "2.0", "method": "initialize", "id": 1}
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._adapter = ACPUAMPAdapter()
    
    async def initialize(self, agent: 'BaseAgent') -> None:
        """Initialize the ACP transport"""
        self.agent = agent
    
    @http("/acp", method="post")
    async def acp_http(
        self,
        jsonrpc: str = "2.0",
        method: str = "",
        params: Optional[Dict[str, Any]] = None,
        id: Optional[Any] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        JSON-RPC 2.0 over HTTP endpoint.
        
        Handles ACP methods and streams responses for chat/submit.
        """
        params = params or {}
        
        # Handle different methods
        if method == "initialize":
            result = await self._handle_initialize(params)
            yield self._jsonrpc_response(id, result)
        
        elif method == "shutdown":
            result = {"status": "shutdown"}
            yield self._jsonrpc_response(id, result)
        
        elif method == "prompt/submit" or method == "chat/submit":
            # Streaming response for chat via UAMP
            messages = params.get("messages", [])
            
            # Send initial notification
            yield self._jsonrpc_notification("prompt/started", {
                "requestId": str(id)
            })
            
            # Stream through handoff and convert via UAMP
            full_content = ""
            async for chunk in self.execute_handoff(messages):
                # Convert OpenAI chunk to UAMP event
                uamp_event = self._openai_chunk_to_uamp(chunk)
                if uamp_event:
                    # Convert UAMP event to ACP notification
                    acp_notification = self._adapter.from_uamp_streaming(uamp_event, request_id=id)
                    if acp_notification:
                        yield f"data: {json.dumps(acp_notification)}\n\n"
                        
                        # Accumulate content
                        if isinstance(uamp_event, ResponseDeltaEvent):
                            if uamp_event.delta and uamp_event.delta.text:
                                full_content += uamp_event.delta.text
            
            # Send completion
            yield self._jsonrpc_response(id, {
                "status": "complete",
                "content": full_content
            })
        
        elif method == "tools/list":
            result = await self._handle_tools_list(params)
            yield self._jsonrpc_response(id, result)
        
        elif method == "tools/call":
            result = await self._handle_tools_call(params)
            yield self._jsonrpc_response(id, result)
        
        elif method == "capabilities":
            result = self._get_capabilities()
            yield self._jsonrpc_response(id, result)
        
        elif method == "files/read":
            result = await self._handle_files_read(params)
            yield self._jsonrpc_response(id, result)
        
        elif method == "files/write":
            result = await self._handle_files_write(params)
            yield self._jsonrpc_response(id, result)
        
        elif method == "files/list":
            result = await self._handle_files_list(params)
            yield self._jsonrpc_response(id, result)
        
        elif method == "terminal/run":
            result = await self._handle_terminal_run(params)
            yield self._jsonrpc_response(id, result)
        
        elif method == "agent/plan":
            result = await self._handle_agent_plan(params)
            yield self._jsonrpc_response(id, result)
        
        elif method == "slash/list":
            result = await self._handle_slash_list(params)
            yield self._jsonrpc_response(id, result)
        
        elif method == "slash/execute":
            # Streaming slash command execution
            async for chunk in self._handle_slash_execute(params):
                yield chunk
            yield self._jsonrpc_response(id, {"status": "complete"})
        
        else:
            yield self._jsonrpc_error(id, -32601, f"Method not found: {method}")
    
    @websocket("/acp/stream")
    async def acp_websocket(self, ws: 'WebSocket') -> None:
        """
        JSON-RPC 2.0 over WebSocket endpoint.
        
        Provides real-time bidirectional communication for IDE integration.
        """
        await ws.accept()
        
        # Create session
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "id": session_id,
            "created_at": time.time(),
            "initialized": False
        }
        
        try:
            async for message in ws.iter_json():
                await self._handle_ws_message(ws, session_id, message)
        except Exception:
            pass
        finally:
            self._sessions.pop(session_id, None)
    
    async def _handle_ws_message(
        self,
        ws: 'WebSocket',
        session_id: str,
        message: Dict[str, Any]
    ) -> None:
        """Handle WebSocket JSON-RPC message"""
        method = message.get("method", "")
        params = message.get("params", {})
        rpc_id = message.get("id")
        
        if method == "initialize":
            result = await self._handle_initialize(params)
            self._sessions[session_id]["initialized"] = True
            await ws.send_json(self._make_response(rpc_id, result))
        
        elif method == "shutdown":
            await ws.send_json(self._make_response(rpc_id, {"status": "shutdown"}))
            await ws.close()
        
        elif method == "prompt/submit" or method == "chat/submit":
            messages = params.get("messages", [])
            
            # Send started notification
            await ws.send_json(self._make_notification("prompt/started", {
                "requestId": str(rpc_id)
            }))
            
            # Stream through handoff via UAMP
            full_content = ""
            async for chunk in self.execute_handoff(messages):
                # Convert OpenAI chunk to UAMP event
                uamp_event = self._openai_chunk_to_uamp(chunk)
                if uamp_event:
                    # Convert UAMP event to ACP notification
                    acp_notification = self._adapter.from_uamp_streaming(uamp_event, request_id=rpc_id)
                    if acp_notification:
                        await ws.send_json(acp_notification)
                        
                        # Accumulate content
                        if isinstance(uamp_event, ResponseDeltaEvent):
                            if uamp_event.delta and uamp_event.delta.text:
                                full_content += uamp_event.delta.text
            
            # Send completion
            await ws.send_json(self._make_response(rpc_id, {
                "status": "complete",
                "content": full_content
            }))
        
        elif method == "tools/list":
            result = await self._handle_tools_list(params)
            await ws.send_json(self._make_response(rpc_id, result))
        
        elif method == "tools/call":
            result = await self._handle_tools_call(params)
            await ws.send_json(self._make_response(rpc_id, result))
        
        elif method == "capabilities":
            result = self._get_capabilities()
            await ws.send_json(self._make_response(rpc_id, result))
        
        elif method == "files/read":
            result = await self._handle_files_read(params)
            await ws.send_json(self._make_response(rpc_id, result))
        
        elif method == "files/write":
            result = await self._handle_files_write(params)
            await ws.send_json(self._make_response(rpc_id, result))
        
        elif method == "files/list":
            result = await self._handle_files_list(params)
            await ws.send_json(self._make_response(rpc_id, result))
        
        elif method == "terminal/run":
            result = await self._handle_terminal_run(params)
            await ws.send_json(self._make_response(rpc_id, result))
        
        elif method == "agent/plan":
            result = await self._handle_agent_plan(params)
            await ws.send_json(self._make_response(rpc_id, result))
        
        elif method == "slash/list":
            result = await self._handle_slash_list(params)
            await ws.send_json(self._make_response(rpc_id, result))
        
        elif method == "slash/execute":
            # Streaming slash command execution
            async for chunk in self._handle_slash_execute(params):
                await ws.send_json(json.loads(chunk.replace("data: ", "").strip()))
            await ws.send_json(self._make_response(rpc_id, {"status": "complete"}))
        
        else:
            await ws.send_json(self._make_error(rpc_id, -32601, f"Method not found: {method}"))
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request"""
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        return {
            "protocolVersion": "1.0",
            "serverInfo": {
                "name": agent.name if agent else "webagents",
                "version": "2.0.0"
            },
            "capabilities": self._get_capabilities()
        }
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities"""
        return {
            "streaming": True,
            "tools": True,
            "prompts": True,
            "slashCommands": True,
            "multiTurn": True,
            "files": True,
            "terminal": True,
            "plans": True
        }
    
    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available tools"""
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        tools = []
        if agent and hasattr(agent, 'get_all_tools'):
            for tool in agent.get_all_tools():
                if 'function' in tool:
                    tools.append({
                        "name": tool['function'].get('name', ''),
                        "description": tool['function'].get('description', ''),
                        "parameters": tool['function'].get('parameters', {})
                    })
        
        return {"tools": tools}
    
    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call"""
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        if not agent:
            return {"error": "No agent context"}
        
        # Find and execute tool
        try:
            result = await agent.execute_tool(tool_name, tool_args)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
    
    async def _handle_files_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read file contents"""
        path = params.get("path", "")
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        # Try to use filesystem skill if available
        if agent:
            for skill_name, skill in getattr(agent, 'skills', {}).items():
                if hasattr(skill, 'read_file'):
                    try:
                        content = await skill.read_file(path)
                        return {"path": path, "content": content}
                    except Exception as e:
                        return {"error": str(e)}
        
        # Fallback to direct file read (with safety checks)
        import os
        if os.path.exists(path) and os.path.isfile(path):
            try:
                with open(path, 'r') as f:
                    return {"path": path, "content": f.read()}
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"File not found: {path}"}
    
    async def _handle_files_write(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Write file contents"""
        path = params.get("path", "")
        content = params.get("content", "")
        
        # Safety: don't allow writing outside workspace
        import os
        if ".." in path or path.startswith("/"):
            return {"error": "Invalid path"}
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            return {"path": path, "success": True}
        except Exception as e:
            return {"error": str(e)}
    
    async def _handle_files_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List files in directory"""
        path = params.get("path", ".")
        
        import os
        if not os.path.exists(path):
            return {"error": f"Directory not found: {path}"}
        
        try:
            entries = []
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                entries.append({
                    "name": entry,
                    "type": "directory" if os.path.isdir(full_path) else "file",
                    "path": full_path
                })
            return {"path": path, "entries": entries}
        except Exception as e:
            return {"error": str(e)}
    
    async def _handle_terminal_run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run terminal command"""
        command = params.get("command", "")
        cwd = params.get("cwd", ".")
        
        # Safety: limit dangerous commands
        dangerous = ["rm -rf /", "sudo", "mkfs", "dd if="]
        for d in dangerous:
            if d in command:
                return {"error": f"Dangerous command blocked: {d}"}
        
        import subprocess
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _handle_agent_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get or create agent plan for a task"""
        task = params.get("task", "")
        
        # Use LLM to create a plan
        messages = [
            {"role": "system", "content": "You are a planning assistant. Create a step-by-step plan for the given task."},
            {"role": "user", "content": f"Create a plan for: {task}"}
        ]
        
        plan_content = ""
        try:
            async for chunk in self.execute_handoff(messages):
                content = self._extract_content(chunk)
                if content:
                    plan_content += content
        except Exception as e:
            return {"error": str(e)}
        
        return {
            "task": task,
            "plan": plan_content,
            "steps": plan_content.split("\n")
        }
    
    async def _handle_slash_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available slash commands"""
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        commands = []
        if agent and hasattr(agent, 'get_all_commands'):
            for cmd in agent.get_all_commands():
                commands.append({
                    "path": cmd.get('path', ''),
                    "description": cmd.get('description', ''),
                    "parameters": cmd.get('parameters', {})
                })
        
        return {"commands": commands}
    
    async def _handle_slash_execute(self, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Execute a slash command"""
        command = params.get("command", "")
        args = params.get("arguments", {})
        
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        if not agent:
            yield self._jsonrpc_notification("slash/error", {"error": "No agent context"})
            return
        
        # Find and execute command
        if hasattr(agent, 'execute_command'):
            try:
                result = await agent.execute_command(command, args)
                yield self._jsonrpc_notification("slash/result", {"result": result})
            except Exception as e:
                yield self._jsonrpc_notification("slash/error", {"error": str(e)})
        else:
            yield self._jsonrpc_notification("slash/error", {"error": f"Command not found: {command}"})
    
    def _extract_content(self, chunk: Dict[str, Any]) -> str:
        """Extract text content from OpenAI chunk"""
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content", "")
        return ""
    
    def _openai_chunk_to_uamp(self, chunk: Dict[str, Any]) -> Optional[ResponseDeltaEvent]:
        """Convert OpenAI streaming chunk to UAMP ResponseDeltaEvent."""
        choices = chunk.get("choices", [])
        if not choices:
            return None
        
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        
        if not content:
            return None
        
        return ResponseDeltaEvent(
            response_id=chunk.get("id", ""),
            delta=ContentDelta(type="text", text=content)
        )
    
    def _jsonrpc_response(self, id: Any, result: Any) -> str:
        """Create JSON-RPC response string for SSE"""
        return f"data: {json.dumps({'jsonrpc': '2.0', 'id': id, 'result': result})}\n\n"
    
    def _jsonrpc_notification(self, method: str, params: Dict[str, Any]) -> str:
        """Create JSON-RPC notification string for SSE"""
        return f"data: {json.dumps({'jsonrpc': '2.0', 'method': method, 'params': params})}\n\n"
    
    def _jsonrpc_error(self, id: Any, code: int, message: str) -> str:
        """Create JSON-RPC error string for SSE"""
        return f"data: {json.dumps({'jsonrpc': '2.0', 'id': id, 'error': {'code': code, 'message': message}})}\n\n"
    
    def _make_response(self, id: Any, result: Any) -> Dict[str, Any]:
        """Create JSON-RPC response dict for WebSocket"""
        return {"jsonrpc": "2.0", "id": id, "result": result}
    
    def _make_notification(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create JSON-RPC notification dict for WebSocket"""
        return {"jsonrpc": "2.0", "method": method, "params": params}
    
    def _make_error(self, id: Any, code: int, message: str) -> Dict[str, Any]:
        """Create JSON-RPC error dict for WebSocket"""
        return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}
