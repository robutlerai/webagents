"""
ACP Transport Skill - WebAgents V2.0

Agent Client Protocol (ACP) implementation for IDE integration.
https://agentclientprotocol.com/

Fully compliant with ACP specification v1.
Uses UAMP (Universal Agentic Message Protocol) for internal message representation.
"""

import json
import uuid
import time
import asyncio
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

try:
    from webagents.agents.skills.robutler.payments.exceptions import PaymentTokenRequiredError
except ImportError:
    PaymentTokenRequiredError = None  # type: ignore[misc, assignment]


# ACP Protocol Version (integer, only bumped for breaking changes)
ACP_PROTOCOL_VERSION = 1

# ACP Error Codes
class ACPErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    AUTH_REQUIRED = -32000
    RESOURCE_NOT_FOUND = -32001
    PAYMENT_REQUIRED = -32402  # Custom: agent requires payment token (retry with payment_token in params)


class ACPSession:
    """Represents an ACP session."""
    
    def __init__(self, session_id: str, cwd: str = "."):
        self.session_id = session_id
        self.cwd = cwd
        self.created_at = time.time()
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_mode: Optional[str] = None
        self.available_modes: List[Dict[str, Any]] = []
        self.mcp_servers: List[Dict[str, Any]] = []
        self.cancelled = False
        self.active_request_id: Optional[Any] = None


class ACPTransportSkill(Skill):
    """
    Agent Client Protocol (ACP) transport for IDE integration.
    
    Fully compliant with ACP specification v1 from agentclientprotocol.com.
    
    Implements JSON-RPC 2.0 over HTTP and WebSocket for communication
    with code editors like Cursor, Zed, and JetBrains IDEs.
    
    Agent Methods (required):
    - initialize: Negotiate protocol version and capabilities
    - authenticate: Authenticate with the agent (if required)
    - session/new: Create a new conversation session
    - session/prompt: Send user prompts
    - session/cancel: Cancel ongoing operations (notification)
    
    Agent Methods (optional):
    - session/load: Load an existing session
    - session/set_mode: Switch between agent modes
    
    Client Methods (agent can call):
    - fs/readTextFile: Read file contents
    - fs/writeTextFile: Write file contents
    - session/request_permission: Request user authorization
    - terminal/*: Terminal operations
    
    Notifications:
    - session/update: Stream real-time updates to client
    
    Endpoints:
    - POST /acp - JSON-RPC over HTTP
    - WS /acp/stream - JSON-RPC over WebSocket (streaming)
    
    Example:
        agent = BaseAgent(
            name="my-agent",
            skills=[ACPTransportSkill()]
        )
        
        # Initialize connection
        # POST /agents/my-agent/acp
        # {"jsonrpc": "2.0", "method": "initialize", "params": {"protocolVersion": 1}, "id": 0}
        
        # Create session
        # {"jsonrpc": "2.0", "method": "session/new", "params": {"cwd": "/path/to/project"}, "id": 1}
        
        # Send prompt
        # {"jsonrpc": "2.0", "method": "session/prompt", "params": {"sessionId": "...", "prompt": [...]}, "id": 2}
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config, scope="all")
        self._sessions: Dict[str, ACPSession] = {}
        self._adapter = ACPUAMPAdapter()
        self._client_capabilities: Dict[str, Any] = {}
        self._initialized = False
        self._authenticated = False
        self._auth_methods: List[Dict[str, Any]] = []
    
    async def initialize(self, agent: 'BaseAgent') -> None:
        """Initialize the ACP transport"""
        self.agent = agent
    
    # =========================================================================
    # HTTP Endpoint
    # =========================================================================
    
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
        
        Handles all ACP methods per specification.
        """
        params = params or {}
        
        try:
            # === Agent Methods (required) ===
            
            if method == "initialize":
                result = await self._handle_initialize(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "authenticate":
                result = await self._handle_authenticate(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "session/new":
                result = await self._handle_session_new(params)
                if "error" in result:
                    yield self._jsonrpc_error(id, result["error"]["code"], result["error"]["message"])
                else:
                    yield self._jsonrpc_response(id, result)
            
            elif method == "session/prompt":
                # Streaming response via SSE
                async for chunk in self._handle_session_prompt_streaming(id, params):
                    yield chunk
            
            elif method == "session/cancel":
                # Notification - no response
                await self._handle_session_cancel(params)
            
            # === Agent Methods (optional) ===
            
            elif method == "session/load":
                result = await self._handle_session_load(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "session/set_mode":
                result = await self._handle_session_set_mode(params)
                yield self._jsonrpc_response(id, result)
            
            # === Legacy methods (kept for backwards compatibility) ===
            
            elif method == "prompt/submit" or method == "chat/submit":
                # Map to session/prompt for backwards compatibility
                async for chunk in self._handle_session_prompt_streaming(id, params):
                    yield chunk
            
            elif method == "tools/list":
                result = await self._handle_tools_list(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "tools/call":
                result = await self._handle_tools_call(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "capabilities":
                result = self._get_agent_capabilities()
                yield self._jsonrpc_response(id, result)
            
            elif method == "shutdown":
                result = {"status": "shutdown"}
                yield self._jsonrpc_response(id, result)
            
            # === Client Methods (agent calls client - these are stubs for testing) ===
            
            elif method == "fs/readTextFile":
                result = await self._handle_fs_read_text_file(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "fs/writeTextFile":
                result = await self._handle_fs_write_text_file(params)
                yield self._jsonrpc_response(id, result)
            
            # Legacy file methods (kept for backwards compatibility)
            elif method == "files/read":
                result = await self._handle_fs_read_text_file(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "files/write":
                result = await self._handle_fs_write_text_file(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "files/list":
                result = await self._handle_files_list(params)
                yield self._jsonrpc_response(id, result)
            
            # === Terminal Methods ===
            
            elif method == "terminal/create":
                result = await self._handle_terminal_create(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "terminal/output":
                result = await self._handle_terminal_output(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "terminal/release":
                result = await self._handle_terminal_release(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "terminal/wait_for_exit":
                result = await self._handle_terminal_wait_for_exit(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "terminal/kill":
                result = await self._handle_terminal_kill(params)
                yield self._jsonrpc_response(id, result)
            
            # Legacy terminal method
            elif method == "terminal/run":
                result = await self._handle_terminal_run(params)
                yield self._jsonrpc_response(id, result)
            
            # === Plan and Commands ===
            
            elif method == "agent/plan":
                result = await self._handle_agent_plan(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "slash/list":
                result = await self._handle_slash_list(params)
                yield self._jsonrpc_response(id, result)
            
            elif method == "slash/execute":
                async for chunk in self._handle_slash_execute(params):
                    yield chunk
                yield self._jsonrpc_response(id, {"status": "complete"})
            
            else:
                yield self._jsonrpc_error(id, ACPErrorCode.METHOD_NOT_FOUND, f"Method not found: {method}")
        
        except Exception as e:
            yield self._jsonrpc_error(id, ACPErrorCode.INTERNAL_ERROR, str(e))
    
    # =========================================================================
    # WebSocket Endpoint
    # =========================================================================
    
    @websocket("/acp/stream")
    async def acp_websocket(self, ws: 'WebSocket') -> None:
        """
        JSON-RPC 2.0 over WebSocket endpoint.
        
        Provides real-time bidirectional communication for IDE integration.
        """
        await ws.accept()
        
        try:
            async for message in ws.iter_json():
                await self._handle_ws_message(ws, message)
        except Exception:
            pass
    
    async def _handle_ws_message(
        self,
        ws: 'WebSocket',
        message: Dict[str, Any]
    ) -> None:
        """Handle WebSocket JSON-RPC message"""
        method = message.get("method", "")
        params = message.get("params", {})
        rpc_id = message.get("id")
        
        try:
            # === Agent Methods (required) ===
            
            if method == "initialize":
                result = await self._handle_initialize(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "authenticate":
                result = await self._handle_authenticate(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "session/new":
                result = await self._handle_session_new(params)
                if "error" in result:
                    await ws.send_json(self._make_error(rpc_id, result["error"]["code"], result["error"]["message"]))
                else:
                    await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "session/prompt":
                await self._handle_session_prompt_ws(ws, rpc_id, params)
            
            elif method == "session/cancel":
                await self._handle_session_cancel(params)
                # No response for notifications
            
            # === Agent Methods (optional) ===
            
            elif method == "session/load":
                result = await self._handle_session_load(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "session/set_mode":
                result = await self._handle_session_set_mode(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            # === Legacy methods ===
            
            elif method == "prompt/submit" or method == "chat/submit":
                await self._handle_session_prompt_ws(ws, rpc_id, params)
            
            elif method == "tools/list":
                result = await self._handle_tools_list(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "tools/call":
                result = await self._handle_tools_call(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "capabilities":
                result = self._get_agent_capabilities()
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "shutdown":
                await ws.send_json(self._make_response(rpc_id, {"status": "shutdown"}))
                await ws.close()
            
            # === Client Methods ===
            
            elif method == "fs/readTextFile":
                result = await self._handle_fs_read_text_file(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "fs/writeTextFile":
                result = await self._handle_fs_write_text_file(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "files/read":
                result = await self._handle_fs_read_text_file(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "files/write":
                result = await self._handle_fs_write_text_file(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "files/list":
                result = await self._handle_files_list(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            # === Terminal Methods ===
            
            elif method == "terminal/create":
                result = await self._handle_terminal_create(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "terminal/output":
                result = await self._handle_terminal_output(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "terminal/release":
                result = await self._handle_terminal_release(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "terminal/wait_for_exit":
                result = await self._handle_terminal_wait_for_exit(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "terminal/kill":
                result = await self._handle_terminal_kill(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "terminal/run":
                result = await self._handle_terminal_run(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            # === Plan and Commands ===
            
            elif method == "agent/plan":
                result = await self._handle_agent_plan(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "slash/list":
                result = await self._handle_slash_list(params)
                await ws.send_json(self._make_response(rpc_id, result))
            
            elif method == "slash/execute":
                async for chunk in self._handle_slash_execute(params):
                    await ws.send_json(json.loads(chunk.replace("data: ", "").strip()))
                await ws.send_json(self._make_response(rpc_id, {"status": "complete"}))
            
            else:
                await ws.send_json(self._make_error(rpc_id, ACPErrorCode.METHOD_NOT_FOUND, f"Method not found: {method}"))
        
        except Exception as e:
            await ws.send_json(self._make_error(rpc_id, ACPErrorCode.INTERNAL_ERROR, str(e)))
    
    # =========================================================================
    # ACP Method Handlers
    # =========================================================================
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle initialize request.
        
        Negotiates protocol version and exchanges capabilities.
        See: https://agentclientprotocol.com/protocol/initialization
        """
        client_version = params.get("protocolVersion", 1)
        self._client_capabilities = params.get("clientCapabilities", {})
        client_info = params.get("clientInfo", {})
        
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        # Determine protocol version (use client's if supported, otherwise ours)
        negotiated_version = min(client_version, ACP_PROTOCOL_VERSION)
        
        self._initialized = True
        
        return {
            "protocolVersion": negotiated_version,
            "agentCapabilities": self._get_agent_capabilities(),
            "agentInfo": {
                "name": agent.name if agent else "webagents",
                "title": agent.description if agent and hasattr(agent, 'description') else "WebAgents",
                "version": "2.0.0"
            },
            "authMethods": self._auth_methods
        }
    
    async def _handle_authenticate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle authenticate request.
        
        See: https://agentclientprotocol.com/protocol/initialization
        """
        auth_method_id = params.get("authMethodId", "")
        
        # For now, accept any authentication
        self._authenticated = True
        
        return {}
    
    async def _handle_session_new(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle session/new request.
        
        Creates a new conversation session.
        See: https://agentclientprotocol.com/protocol/session-setup
        """
        # Check if auth is required but not done
        if self._auth_methods and not self._authenticated:
            return {
                "error": {
                    "code": ACPErrorCode.AUTH_REQUIRED,
                    "message": "Authentication required"
                }
            }
        
        cwd = params.get("cwd", ".")
        mcp_servers = params.get("mcpServers", [])
        
        # Create new session
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        session = ACPSession(session_id, cwd)
        session.mcp_servers = mcp_servers
        
        # Set up available modes (optional capability)
        session.available_modes = [
            {"id": "default", "name": "Default", "description": "Standard conversation mode"},
            {"id": "code", "name": "Code", "description": "Code-focused mode"},
            {"id": "architect", "name": "Architect", "description": "Architecture and planning mode"},
            {"id": "ask", "name": "Ask", "description": "Question-answering mode"}
        ]
        session.current_mode = "default"
        
        self._sessions[session_id] = session
        
        response = {
            "sessionId": session_id
        }
        
        # Include mode state if supported
        if session.available_modes:
            response["modeState"] = {
                "availableModes": session.available_modes,
                "currentModeId": session.current_mode
            }
        
        return response
    
    async def _handle_session_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle session/load request.
        
        Loads an existing session to resume a previous conversation.
        See: https://agentclientprotocol.com/protocol/session-setup#loading-sessions
        """
        session_id = params.get("sessionId", "")
        cwd = params.get("cwd", ".")
        mcp_servers = params.get("mcpServers", [])
        
        session = self._sessions.get(session_id)
        if not session:
            return {
                "error": {
                    "code": ACPErrorCode.RESOURCE_NOT_FOUND,
                    "message": f"Session not found: {session_id}"
                }
            }
        
        # Update session config
        session.cwd = cwd
        session.mcp_servers = mcp_servers
        
        # Note: In a real implementation, we would stream back the conversation
        # history via session/update notifications here
        
        response = {}
        
        if session.available_modes:
            response["modeState"] = {
                "availableModes": session.available_modes,
                "currentModeId": session.current_mode
            }
        
        return response
    
    async def _handle_session_prompt_streaming(
        self,
        rpc_id: Any,
        params: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Handle session/prompt request with SSE streaming.
        
        Uses full UAMP flow:
        1. Convert ACP request to UAMP events via adapter
        2. Process through agent.process_uamp()
        3. Convert UAMP server events back to ACP notifications via adapter
        
        See: https://agentclientprotocol.com/protocol/prompt-turn
        """
        from webagents.uamp import (
            ResponseCreatedEvent,
            ResponseDeltaEvent,
            ResponseDoneEvent,
            ResponseErrorEvent,
            ToolCallEvent,
            ThinkingEvent,
        )
        
        session_id = params.get("sessionId", "")
        
        session = self._sessions.get(session_id)
        if not session:
            # Create implicit session for backwards compatibility
            session = ACPSession(session_id or f"sess_{uuid.uuid4().hex[:12]}", ".")
            self._sessions[session.session_id] = session

        session.active_request_id = rpc_id
        session.cancelled = False

        # Get agent reference and set transport-agnostic payment token from params
        context = self.get_context()
        agent = context.agent if context else self.agent
        if context and params.get("payment_token"):
            context.payment_token = params["payment_token"]

        if not agent:
            yield self._jsonrpc_error(rpc_id, ACPErrorCode.INTERNAL_ERROR, "No agent available")
            return
        
        # Convert ACP request to UAMP events using adapter
        acp_request = {
            "method": "session/prompt",
            "params": params
        }
        uamp_events = self._adapter.to_uamp(acp_request)
        
        # Track content for history
        full_content = ""
        
        try:
            # Process through agent's native UAMP method
            async for uamp_event in agent.process_uamp(uamp_events):
                if session.cancelled:
                    yield self._jsonrpc_response(rpc_id, {"stopReason": "cancelled"})
                    return
                
                # Convert UAMP event to ACP notification using adapter
                acp_notification = self._adapter.from_uamp_streaming(
                    uamp_event, 
                    session_id=session.session_id,
                    request_id=rpc_id
                )
                
                if acp_notification:
                    yield f"data: {json.dumps(acp_notification)}\n\n"
                
                # Track content for history
                if isinstance(uamp_event, ResponseDeltaEvent) and uamp_event.delta:
                    if uamp_event.delta.text:
                        full_content += uamp_event.delta.text
            
            # Add assistant response to history
            if full_content:
                session.conversation_history.append({
                    "role": "assistant",
                    "content": full_content
                })
            
            # Final response with stop reason
            yield self._jsonrpc_response(rpc_id, {"stopReason": "end_turn"})
        
        except asyncio.CancelledError:
            yield self._jsonrpc_response(rpc_id, {"stopReason": "cancelled"})
        except Exception as e:
            if PaymentTokenRequiredError is not None and isinstance(e, PaymentTokenRequiredError):
                err_payload = {
                    "code": ACPErrorCode.PAYMENT_REQUIRED,
                    "message": getattr(e, "user_message", None) or str(e),
                }
                if hasattr(e, "context") and isinstance(e.context, dict):
                    err_payload["data"] = e.context
                yield f"data: {json.dumps({'jsonrpc': '2.0', 'id': rpc_id, 'error': err_payload})}\n\n"
            else:
                yield self._jsonrpc_error(rpc_id, ACPErrorCode.INTERNAL_ERROR, str(e))
        finally:
            session.active_request_id = None
    
    async def _handle_session_prompt_ws(
        self,
        ws: 'WebSocket',
        rpc_id: Any,
        params: Dict[str, Any]
    ) -> None:
        """
        Handle session/prompt request over WebSocket.
        
        Uses full UAMP flow via agent.process_uamp().
        """
        from webagents.uamp import ResponseDeltaEvent
        
        session_id = params.get("sessionId", "")
        
        session = self._sessions.get(session_id)
        if not session:
            session = ACPSession(session_id or f"sess_{uuid.uuid4().hex[:12]}", ".")
            self._sessions[session.session_id] = session
        
        session.active_request_id = rpc_id
        session.cancelled = False

        # Get agent reference and set transport-agnostic payment token from params
        context = self.get_context()
        agent = context.agent if context else self.agent
        if context and params.get("payment_token"):
            context.payment_token = params["payment_token"]

        if not agent:
            await ws.send_json(self._make_error(rpc_id, ACPErrorCode.INTERNAL_ERROR, "No agent available"))
            return

        # Convert ACP request to UAMP events using adapter
        acp_request = {
            "method": "session/prompt",
            "params": params
        }
        uamp_events = self._adapter.to_uamp(acp_request)

        full_content = ""

        try:
            # Process through agent's native UAMP method
            async for uamp_event in agent.process_uamp(uamp_events):
                if session.cancelled:
                    await ws.send_json(self._make_response(rpc_id, {"stopReason": "cancelled"}))
                    return
                
                # Convert UAMP event to ACP notification using adapter
                acp_notification = self._adapter.from_uamp_streaming(
                    uamp_event,
                    session_id=session.session_id,
                    request_id=rpc_id
                )
                
                if acp_notification:
                    await ws.send_json(acp_notification)
                
                # Track content for history
                if isinstance(uamp_event, ResponseDeltaEvent) and uamp_event.delta:
                    if uamp_event.delta.text:
                        full_content += uamp_event.delta.text
            
            if full_content:
                session.conversation_history.append({
                    "role": "assistant",
                    "content": full_content
                })
            
            await ws.send_json(self._make_response(rpc_id, {"stopReason": "end_turn"}))
        
        except asyncio.CancelledError:
            await ws.send_json(self._make_response(rpc_id, {"stopReason": "cancelled"}))
        except Exception as e:
            if PaymentTokenRequiredError is not None and isinstance(e, PaymentTokenRequiredError):
                err_payload = {
                    "code": ACPErrorCode.PAYMENT_REQUIRED,
                    "message": getattr(e, "user_message", None) or str(e),
                }
                if hasattr(e, "context") and isinstance(e.context, dict):
                    err_payload["data"] = e.context
                await ws.send_json({"jsonrpc": "2.0", "id": rpc_id, "error": err_payload})
            else:
                await ws.send_json(self._make_error(rpc_id, ACPErrorCode.INTERNAL_ERROR, str(e)))
        finally:
            session.active_request_id = None

    async def _handle_session_cancel(self, params: Dict[str, Any]) -> None:
        """
        Handle session/cancel notification.
        
        Cancels ongoing operations for a session.
        See: https://agentclientprotocol.com/protocol/prompt-turn#cancellation
        """
        session_id = params.get("sessionId", "")
        
        session = self._sessions.get(session_id)
        if session:
            session.cancelled = True
    
    async def _handle_session_set_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle session/set_mode request.
        
        Sets the current mode for a session.
        See: https://agentclientprotocol.com/protocol/session-modes
        """
        session_id = params.get("sessionId", "")
        mode_id = params.get("modeId", "")
        
        session = self._sessions.get(session_id)
        if not session:
            return {
                "error": {
                    "code": ACPErrorCode.RESOURCE_NOT_FOUND,
                    "message": f"Session not found: {session_id}"
                }
            }
        
        # Validate mode
        valid_modes = [m["id"] for m in session.available_modes]
        if mode_id not in valid_modes:
            return {
                "error": {
                    "code": ACPErrorCode.INVALID_PARAMS,
                    "message": f"Invalid mode: {mode_id}. Valid modes: {valid_modes}"
                }
            }
        
        session.current_mode = mode_id
        
        return {}
    
    # =========================================================================
    # Client Methods (fs/*, terminal/*)
    # =========================================================================
    
    async def _handle_fs_read_text_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle fs/readTextFile request.
        
        See: https://agentclientprotocol.com/protocol/file-system#reading-files
        """
        session_id = params.get("sessionId", "")
        path = params.get("path", "")
        start_line = params.get("startLine")  # 1-based
        max_lines = params.get("maxLines")
        
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        # Try to use filesystem skill if available
        if agent:
            for skill_name, skill in getattr(agent, 'skills', {}).items():
                if hasattr(skill, 'read_file'):
                    try:
                        content = await skill.read_file(path)
                        
                        # Apply line limits if specified
                        if start_line is not None or max_lines is not None:
                            lines = content.split('\n')
                            start = (start_line - 1) if start_line else 0
                            end = (start + max_lines) if max_lines else len(lines)
                            content = '\n'.join(lines[start:end])
                        
                        return {"text": content}
                    except Exception as e:
                        return {"error": {"code": ACPErrorCode.INTERNAL_ERROR, "message": str(e)}}
        
        # Fallback to direct file read
        import os
        if os.path.exists(path) and os.path.isfile(path):
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    
                    if start_line is not None or max_lines is not None:
                        lines = content.split('\n')
                        start = (start_line - 1) if start_line else 0
                        end = (start + max_lines) if max_lines else len(lines)
                        content = '\n'.join(lines[start:end])
                    
                    return {"text": content}
            except Exception as e:
                return {"error": {"code": ACPErrorCode.INTERNAL_ERROR, "message": str(e)}}
        
        return {"error": {"code": ACPErrorCode.RESOURCE_NOT_FOUND, "message": f"File not found: {path}"}}
    
    async def _handle_fs_write_text_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle fs/writeTextFile request.
        
        See: https://agentclientprotocol.com/protocol/file-system#writing-files
        """
        session_id = params.get("sessionId", "")
        path = params.get("path", "")
        text = params.get("text", "")
        
        import os
        
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            with open(path, 'w') as f:
                f.write(text)
            return {}
        except Exception as e:
            return {"error": {"code": ACPErrorCode.INTERNAL_ERROR, "message": str(e)}}
    
    async def _handle_files_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List files in directory (legacy method)"""
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
    
    # Terminal methods
    _terminals: Dict[str, Dict[str, Any]] = {}
    
    async def _handle_terminal_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle terminal/create request.
        
        See: https://agentclientprotocol.com/protocol/terminals
        """
        import subprocess
        
        session_id = params.get("sessionId", "")
        command = params.get("command", "")
        args = params.get("args", [])
        cwd = params.get("cwd", ".")
        env = params.get("env", [])
        max_output_bytes = params.get("maxOutputBytes", 1024 * 1024)  # 1MB default
        
        terminal_id = f"term_{uuid.uuid4().hex[:8]}"
        
        # Build environment
        process_env = dict(os.environ)
        for var in env:
            process_env[var.get("name", "")] = var.get("value", "")
        
        try:
            full_command = [command] + args
            process = subprocess.Popen(
                full_command,
                cwd=cwd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            self._terminals[terminal_id] = {
                "process": process,
                "output": "",
                "max_output_bytes": max_output_bytes,
                "session_id": session_id
            }
            
            return {"terminalId": terminal_id}
        except Exception as e:
            return {"error": {"code": ACPErrorCode.INTERNAL_ERROR, "message": str(e)}}
    
    async def _handle_terminal_output(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle terminal/output request."""
        terminal_id = params.get("terminalId", "")
        
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            return {"error": {"code": ACPErrorCode.RESOURCE_NOT_FOUND, "message": f"Terminal not found: {terminal_id}"}}
        
        process = terminal["process"]
        
        # Read any available output
        import select
        if process.stdout:
            while select.select([process.stdout], [], [], 0)[0]:
                line = process.stdout.readline()
                if not line:
                    break
                terminal["output"] += line
        
        # Truncate if needed
        max_bytes = terminal["max_output_bytes"]
        output = terminal["output"]
        truncated = False
        if len(output) > max_bytes:
            output = output[-max_bytes:]
            truncated = True
        
        exit_status = None
        if process.poll() is not None:
            exit_status = {"exitCode": process.returncode}
        
        return {
            "output": output,
            "truncated": truncated,
            "exitStatus": exit_status
        }
    
    async def _handle_terminal_release(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle terminal/release request."""
        terminal_id = params.get("terminalId", "")
        
        terminal = self._terminals.pop(terminal_id, None)
        if not terminal:
            return {"error": {"code": ACPErrorCode.RESOURCE_NOT_FOUND, "message": f"Terminal not found: {terminal_id}"}}
        
        process = terminal["process"]
        if process.poll() is None:
            process.kill()
        
        return {}
    
    async def _handle_terminal_wait_for_exit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle terminal/wait_for_exit request."""
        terminal_id = params.get("terminalId", "")
        
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            return {"error": {"code": ACPErrorCode.RESOURCE_NOT_FOUND, "message": f"Terminal not found: {terminal_id}"}}
        
        process = terminal["process"]
        process.wait()
        
        return {"exitCode": process.returncode}
    
    async def _handle_terminal_kill(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle terminal/kill request."""
        terminal_id = params.get("terminalId", "")
        
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            return {"error": {"code": ACPErrorCode.RESOURCE_NOT_FOUND, "message": f"Terminal not found: {terminal_id}"}}
        
        process = terminal["process"]
        if process.poll() is None:
            process.kill()
        
        return {}
    
    async def _handle_terminal_run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run terminal command (legacy method)"""
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
    
    # =========================================================================
    # Tool and Plan Methods
    # =========================================================================
    
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
        
        try:
            result = await agent.execute_tool(tool_name, tool_args)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
    
    async def _handle_agent_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get or create agent plan for a task"""
        task = params.get("task", "")
        
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
                    "name": cmd.get('path', ''),
                    "description": cmd.get('description', ''),
                    "input": cmd.get('input')
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
        
        if hasattr(agent, 'execute_command'):
            try:
                result = await agent.execute_command(command, args)
                yield self._jsonrpc_notification("slash/result", {"result": result})
            except Exception as e:
                yield self._jsonrpc_notification("slash/error", {"error": str(e)})
        else:
            yield self._jsonrpc_notification("slash/error", {"error": f"Command not found: {command}"})
    
    # =========================================================================
    # Capabilities
    # =========================================================================
    
    def _get_agent_capabilities(self) -> Dict[str, Any]:
        """
        Get agent capabilities per ACP spec.
        
        See: https://agentclientprotocol.com/protocol/initialization#agent-capabilities
        """
        context = self.get_context()
        agent = context.agent if context else self.agent
        
        # Get model capabilities from LLM skills
        model_caps = self._get_model_capabilities(agent)
        
        # Determine prompt capabilities from model
        modalities = model_caps.get("modalities", ["text"])
        
        return {
            "loadSession": True,
            "promptCapabilities": {
                "image": "image" in modalities,
                "audio": "audio" in modalities,
                "embeddedContext": True
            },
            "mcpCapabilities": {
                "http": True,
                "sse": False
            },
            "sessionCapabilities": {},
            # Extension: include full model capabilities
            "modelCapabilities": model_caps
        }
    
    def _get_model_capabilities(self, agent) -> Dict[str, Any]:
        """Get UAMP model capabilities from agent's LLM skills."""
        if not agent or not hasattr(agent, 'skills'):
            return {"modalities": ["text"], "supports_streaming": True}
        
        for skill in agent.skills.values():
            if hasattr(skill, '_adapter') and hasattr(skill._adapter, 'get_capabilities'):
                caps = skill._adapter.get_capabilities()
                return {
                    "model_id": caps.model_id,
                    "provider": caps.provider,
                    "modalities": caps.modalities,
                    "supports_streaming": caps.supports_streaming,
                    "supports_thinking": caps.supports_thinking,
                    "context_window": caps.context_window,
                    "max_output_tokens": caps.max_output_tokens,
                    "image": {
                        "formats": caps.image.formats,
                        "detail_levels": caps.image.detail_levels,
                    } if caps.image else None,
                    "audio": {
                        "input_formats": caps.audio.input_formats,
                        "output_formats": caps.audio.output_formats,
                        "supports_realtime": caps.audio.supports_realtime,
                    } if caps.audio else None,
                    "file": {
                        "supports_pdf": caps.file.supports_pdf,
                        "supported_mime_types": caps.file.supported_mime_types,
                    } if caps.file else None,
                    "tools": {
                        "supports_tools": caps.tools.supports_tools,
                        "built_in_tools": caps.tools.built_in_tools,
                    } if caps.tools else None,
                }
        
        return {"modalities": ["text"], "supports_streaming": True}
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _convert_prompt_to_messages(self, prompt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert ACP content blocks to OpenAI-style messages."""
        if not prompt:
            return []
        
        # Check if already in messages format (legacy)
        if prompt and isinstance(prompt[0], dict) and "role" in prompt[0]:
            return prompt
        
        # Convert content blocks to a single user message
        content_parts = []
        for block in prompt:
            block_type = block.get("type", "text")
            
            if block_type == "text":
                content_parts.append({"type": "text", "text": block.get("text", "")})
            
            elif block_type == "image":
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.get('mimeType', 'image/png')};base64,{block.get('data', '')}"
                    }
                })
            
            elif block_type == "resource":
                resource = block.get("resource", {})
                text = resource.get("text", "")
                uri = resource.get("uri", "")
                content_parts.append({
                    "type": "text",
                    "text": f"[Resource: {uri}]\n{text}"
                })
            
            elif block_type == "resourceLink":
                uri = block.get("uri", "")
                content_parts.append({
                    "type": "text",
                    "text": f"[Resource Link: {uri}]"
                })
        
        if len(content_parts) == 1 and content_parts[0].get("type") == "text":
            return [{"role": "user", "content": content_parts[0]["text"]}]
        
        return [{"role": "user", "content": content_parts}]
    
    def _extract_content(self, chunk: Dict[str, Any]) -> str:
        """Extract text content from OpenAI chunk"""
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content", "")
        return ""
    
    def _extract_tool_call(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract tool call from OpenAI chunk"""
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            tool_calls = delta.get("tool_calls", [])
            if tool_calls:
                return tool_calls[0]
        return None
    
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
    
    def _session_update_notification(self, session_id: str, update: Dict[str, Any]) -> str:
        """Create session/update notification for SSE."""
        return self._jsonrpc_notification("session/update", {
            "sessionId": session_id,
            "update": update
        })
    
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


# Import for os module used in terminal methods
import os
