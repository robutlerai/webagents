"""
WebSocket Agents Extension for WebAgentsd.

Provides an AgentSource for WebSocket-connected remote agents.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from .interface import WebAgentsExtension, AgentSource

logger = logging.getLogger(__name__)


class RemoteAgentProxy:
    """
    Proxy that forwards requests to a WebSocket-connected agent.
    
    Acts as a stand-in for the remote agent within webagentsd.
    """
    
    def __init__(
        self,
        name: str,
        websocket: WebSocket,
        capabilities: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self._ws = websocket
        self.capabilities = capabilities or {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._is_connected = True
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    async def run_streaming(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ):
        """Forward request to remote agent and yield chunks."""
        request_id = str(uuid.uuid4())
        
        # Create future for collecting chunks
        chunk_queue = asyncio.Queue()
        self._pending_requests[request_id] = chunk_queue
        
        try:
            # Send request
            await self._ws.send_json({
                "type": "completion_request",
                "id": request_id,
                "messages": messages,
                "tools": tools
            })
            
            # Yield chunks as they arrive
            while True:
                chunk = await chunk_queue.get()
                
                if chunk.get("type") == "completion_done":
                    break
                elif chunk.get("type") == "completion_error":
                    raise Exception(chunk.get("error", "Unknown error"))
                elif chunk.get("type") == "completion_chunk":
                    yield chunk.get("chunk")
                    
        finally:
            del self._pending_requests[request_id]
    
    async def run(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Forward request to remote agent and return complete response."""
        chunks = []
        async for chunk in self.run_streaming(messages, tools):
            chunks.append(chunk)
        
        # Combine chunks into response (simple implementation)
        if not chunks:
            return {"choices": []}
            
        # Assuming OpenAI format chunks
        content_parts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                choices = chunk.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        content_parts.append(delta["content"])
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "".join(content_parts)
                },
                "finish_reason": "stop"
            }]
        }
    
    async def handle_response(self, message: Dict[str, Any]) -> None:
        """Handle incoming response message from remote agent."""
        request_id = message.get("request_id")
        if request_id and request_id in self._pending_requests:
            await self._pending_requests[request_id].put(message)
    
    def disconnect(self) -> None:
        """Mark agent as disconnected."""
        self._is_connected = False
        # Cancel all pending requests
        for queue in self._pending_requests.values():
            queue.put_nowait({
                "type": "completion_error",
                "error": "Agent disconnected"
            })


class WebSocketAgentSource(AgentSource):
    """
    Agent source for WebSocket-connected remote agents.
    
    Maintains connections to remote agents and provides them
    as if they were local agents.
    """
    
    def __init__(self):
        self.agents: Dict[str, RemoteAgentProxy] = {}
    
    async def get_agent(self, name: str) -> Optional[RemoteAgentProxy]:
        """Get a connected agent by name."""
        agent = self.agents.get(name)
        if agent and agent.is_connected:
            return agent
        return None
    
    def get_source_type(self) -> str:
        return "websocket"
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all connected agents."""
        return [
            {
                "name": name,
                "source": "websocket",
                "connected": agent.is_connected,
                "capabilities": agent.capabilities
            }
            for name, agent in self.agents.items()
            if agent.is_connected
        ]
    
    def register_agent(
        self,
        name: str,
        websocket: WebSocket,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> RemoteAgentProxy:
        """Register a new WebSocket agent."""
        proxy = RemoteAgentProxy(name, websocket, capabilities)
        self.agents[name] = proxy
        logger.info(f"Registered WebSocket agent: {name}")
        return proxy
    
    def unregister_agent(self, name: str) -> None:
        """Unregister an agent."""
        if name in self.agents:
            self.agents[name].disconnect()
            del self.agents[name]
            logger.info(f"Unregistered WebSocket agent: {name}")


class WebSocketAgentsExtension(WebAgentsExtension):
    """
    Extension adding WebSocket agent connectivity to webagentsd.
    
    Allows remote agents (from TypeScript SDK, browser, etc.) to
    connect and be served through webagentsd.
    
    Example:
        ```python
        daemon = WebAgentsDaemon()
        daemon.add_extension(WebSocketAgentsExtension())
        await daemon.start()
        ```
    """
    
    def __init__(self):
        self.source = WebSocketAgentSource()
    
    def get_name(self) -> str:
        return "websocket_agents"
    
    async def initialize(self, server: "WebAgentsServer") -> None:
        """Set up WebSocket route on the server."""
        self._setup_websocket_route(server.app)
    
    def _setup_websocket_route(self, app) -> None:
        """Add WebSocket endpoint for agent connections."""
        
        @app.websocket("/ws/agents")
        async def agent_websocket(websocket: WebSocket):
            await websocket.accept()
            agent_name = None
            proxy = None
            
            try:
                # Wait for registration message
                data = await websocket.receive_json()
                
                if data.get("type") != "register":
                    await websocket.close(code=4001, reason="Expected register message")
                    return
                
                agent_name = data.get("agent_name")
                if not agent_name:
                    await websocket.close(code=4002, reason="Missing agent_name")
                    return
                
                capabilities = data.get("capabilities", {})
                proxy = self.source.register_agent(agent_name, websocket, capabilities)
                
                # Send confirmation
                await websocket.send_json({
                    "type": "registered",
                    "agent_name": agent_name
                })
                
                # Message loop
                while True:
                    message = await websocket.receive_json()
                    msg_type = message.get("type")
                    
                    if msg_type == "pong":
                        continue
                    elif msg_type in ("completion_chunk", "completion_done", "completion_error"):
                        await proxy.handle_response(message)
                    else:
                        logger.debug(f"Unknown message type from {agent_name}: {msg_type}")
                        
            except WebSocketDisconnect:
                logger.info(f"Agent {agent_name} disconnected")
            except Exception as e:
                logger.error(f"WebSocket error for {agent_name}: {e}")
            finally:
                if agent_name:
                    self.source.unregister_agent(agent_name)
    
    def get_agent_sources(self) -> List[AgentSource]:
        """Return the WebSocket agent source."""
        return [self.source]
