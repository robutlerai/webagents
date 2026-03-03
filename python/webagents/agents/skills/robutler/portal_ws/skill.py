"""
Portal WebSocket Skill.

Connect local Python agents to portal/webagentsd via WebSocket for external access.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from webagents.agents.skills.base import Skill

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketClientProtocol = Any

logger = logging.getLogger(__name__)


class PortalWSSkill(Skill):
    """
    Connect local Python agents to portal/webagentsd via WebSocket.
    
    This skill enables local agents to be exposed via a portal or webagentsd
    server, making them accessible to external users.
    
    Example:
        ```python
        agent = BaseAgent(
            name="my-agent",
            skills={
                "portal": PortalWSSkill(
                    portal_url="wss://robutler.ai/ws/agents",
                    public=True
                )
            }
        )
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.portal_url = (config or {}).get("portal_url", "wss://robutler.ai/ws/agents")
        self.public = (config or {}).get("public", False)
        self.auto_reconnect = (config or {}).get("auto_reconnect", True)
        self.reconnect_delay = (config or {}).get("reconnect_delay", 5)
        self.max_reconnect_attempts = (config or {}).get("max_reconnect_attempts", 10)
        
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._shutdown = False
        self._connection_task: Optional[asyncio.Task] = None
    
    async def initialize(self, agent: "BaseAgent") -> None:
        """Initialize and start connection to portal."""
        if not HAS_WEBSOCKETS:
            logger.warning("websockets library not installed. Portal WS skill disabled.")
            return
            
        await super().initialize(agent)
        self._connection_task = asyncio.create_task(self._connect_and_serve())
    
    async def _connect_and_serve(self) -> None:
        """Connect to portal and handle incoming requests."""
        while not self._shutdown:
            try:
                async with websockets.connect(self.portal_url) as ws:
                    self._ws = ws
                    self._connected = True
                    self._reconnect_attempts = 0
                    
                    # Register agent
                    await self._register()
                    
                    # Handle incoming messages
                    async for message in ws:
                        await self._handle_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                self._connected = False
                self._ws = None
                
                if self._shutdown:
                    break
                    
                if self.auto_reconnect and self._reconnect_attempts < self.max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    logger.info(f"Reconnecting to portal (attempt {self._reconnect_attempts})...")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    logger.error("Max reconnect attempts reached")
                    break
                    
            except Exception as e:
                logger.error(f"Portal connection error: {e}")
                self._connected = False
                self._ws = None
                
                if self._shutdown:
                    break
                    
                if self.auto_reconnect:
                    await asyncio.sleep(self.reconnect_delay)
    
    async def _register(self) -> None:
        """Register agent with portal."""
        if not self._ws or not self.agent:
            return
            
        capabilities = self._get_capabilities()
        
        await self._ws.send(json.dumps({
            "type": "register",
            "agent_name": self.agent.name,
            "capabilities": capabilities,
            "public": self.public
        }))
        
        logger.info(f"Registered agent '{self.agent.name}' with portal")
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities for registration."""
        if not self.agent:
            return {}
            
        return {
            "name": self.agent.name,
            "tools": [t.name for t in self.agent.get_tools()],
            "endpoints": list(self.agent._http_handlers.keys()) if hasattr(self.agent, "_http_handlers") else [],
        }
    
    async def _handle_message(self, message: str) -> None:
        """Handle incoming message from portal."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "registered":
                logger.info(f"Registration confirmed: {data.get('agent_name')}")
                
            elif msg_type == "completion_request":
                await self._handle_completion_request(data)
                
            elif msg_type == "ping":
                await self._send({"type": "pong"})
                
        except Exception as e:
            logger.error(f"Error handling portal message: {e}")
    
    async def _handle_completion_request(self, request: Dict[str, Any]) -> None:
        """Process completion request from portal."""
        if not self.agent:
            return
            
        request_id = request.get("id")
        messages = request.get("messages", [])
        tools = request.get("tools")
        
        try:
            # Stream response back through portal
            async for chunk in self.agent.run_streaming(messages, tools=tools):
                await self._send({
                    "type": "completion_chunk",
                    "request_id": request_id,
                    "chunk": chunk
                })
            
            await self._send({
                "type": "completion_done",
                "request_id": request_id
            })
            
        except Exception as e:
            await self._send({
                "type": "completion_error",
                "request_id": request_id,
                "error": str(e)
            })
    
    async def _send(self, message: Dict[str, Any]) -> None:
        """Send message to portal."""
        if self._ws and self._connected:
            await self._ws.send(json.dumps(message))
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to portal."""
        return self._connected
    
    async def disconnect(self) -> None:
        """Disconnect from portal."""
        self._shutdown = True
        self.auto_reconnect = False
        
        if self._ws:
            await self._ws.close()
            self._ws = None
            
        self._connected = False
        
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
