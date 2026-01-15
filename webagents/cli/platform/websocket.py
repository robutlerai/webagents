"""
Platform WebSocket Bridge

WebSocket connection to robutler.ai for agent exposure and messaging.
"""

import asyncio
import json
from typing import Optional, Dict, Callable, Any
from datetime import datetime

import websockets
from websockets.client import WebSocketClientProtocol

from ..state.local import get_state


class PlatformWebSocket:
    """WebSocket connection to robutler.ai platform.
    
    Enables:
    - Agent exposure (incoming requests)
    - Intent notifications
    - Agent-to-agent messaging
    - Platform cron triggers
    """
    
    def __init__(
        self,
        platform_url: str = "wss://api.robutler.ai/ws",
        on_message: Optional[Callable] = None,
        on_request: Optional[Callable] = None,
    ):
        """Initialize WebSocket bridge.
        
        Args:
            platform_url: WebSocket URL
            on_message: Callback for messages
            on_request: Callback for agent requests
        """
        self.platform_url = platform_url
        self.on_message = on_message
        self.on_request = on_request
        
        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60
    
    async def connect(self):
        """Connect to platform."""
        state = get_state()
        creds = state.get_credentials()
        
        if not creds.get("access_token") and not creds.get("api_key"):
            raise ValueError("Not authenticated. Run 'webagents login' first.")
        
        # Build auth header
        headers = {}
        if creds.get("access_token"):
            headers["Authorization"] = f"Bearer {creds['access_token']}"
        elif creds.get("api_key"):
            headers["X-API-Key"] = creds["api_key"]
        
        try:
            self._ws = await websockets.connect(
                self.platform_url,
                additional_headers=headers,
            )
            self._reconnect_delay = 1
            return True
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            return False
    
    async def run(self):
        """Run WebSocket message loop."""
        self._running = True
        
        while self._running:
            try:
                if not self._ws or self._ws.closed:
                    connected = await self.connect()
                    if not connected:
                        await asyncio.sleep(self._reconnect_delay)
                        self._reconnect_delay = min(
                            self._reconnect_delay * 2,
                            self._max_reconnect_delay
                        )
                        continue
                
                # Receive messages
                async for message in self._ws:
                    await self._handle_message(message)
                    
            except websockets.ConnectionClosed:
                print("WebSocket disconnected, reconnecting...")
                await asyncio.sleep(self._reconnect_delay)
            except Exception as e:
                print(f"WebSocket error: {e}")
                await asyncio.sleep(self._reconnect_delay)
    
    async def _handle_message(self, raw_message: str):
        """Handle incoming WebSocket message.
        
        Args:
            raw_message: Raw JSON message
        """
        try:
            message = json.loads(raw_message)
            msg_type = message.get("type")
            
            if msg_type == "request":
                # Incoming agent request
                if self.on_request:
                    response = await self.on_request(message)
                    await self.send_response(message.get("request_id"), response)
            
            elif msg_type == "intent":
                # Intent notification
                if self.on_message:
                    await self.on_message(message)
            
            elif msg_type == "cron":
                # Platform cron trigger
                if self.on_message:
                    await self.on_message(message)
            
            elif msg_type == "ping":
                # Keep-alive
                await self.send({"type": "pong"})
            
            else:
                # General message
                if self.on_message:
                    await self.on_message(message)
                    
        except json.JSONDecodeError:
            print(f"Invalid WebSocket message: {raw_message}")
    
    async def send(self, message: Dict):
        """Send message to platform.
        
        Args:
            message: Message to send
        """
        if self._ws and not self._ws.closed:
            await self._ws.send(json.dumps(message))
    
    async def send_response(self, request_id: str, response: Any):
        """Send response to a request.
        
        Args:
            request_id: Original request ID
            response: Response data
        """
        await self.send({
            "type": "response",
            "request_id": request_id,
            "data": response,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    async def expose_agent(self, agent_name: str):
        """Expose agent for incoming requests.
        
        Args:
            agent_name: Agent to expose
        """
        await self.send({
            "type": "expose",
            "agent": agent_name,
        })
    
    async def hide_agent(self, agent_name: str):
        """Stop exposing agent.
        
        Args:
            agent_name: Agent to hide
        """
        await self.send({
            "type": "hide",
            "agent": agent_name,
        })
    
    async def subscribe_intent(self, intent: str, agent: str = None):
        """Subscribe to intent notifications.
        
        Args:
            intent: Intent to subscribe to
            agent: Agent to route notifications to
        """
        await self.send({
            "type": "subscribe",
            "intent": intent,
            "agent": agent,
        })
    
    async def close(self):
        """Close WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()


class WebSocketAgentHandler:
    """Handle agent requests over WebSocket."""
    
    def __init__(self, resolver=None):
        """Initialize handler.
        
        Args:
            resolver: AgentResolver for resolving agents
        """
        self.resolver = resolver
    
    async def handle_request(self, message: Dict) -> Any:
        """Handle incoming agent request.
        
        Args:
            message: Request message
            
        Returns:
            Response data
        """
        agent_name = message.get("agent")
        prompt = message.get("prompt")
        
        if not agent_name or not prompt:
            return {"error": "Missing agent or prompt"}
        
        # Resolve agent
        if self.resolver:
            agent = await self.resolver.resolve(agent_name)
            if not agent:
                return {"error": f"Agent not found: {agent_name}"}
            
            # TODO: Execute agent with prompt
            return {
                "agent": agent_name,
                "response": f"Agent {agent_name} would process: {prompt}",
            }
        
        return {"error": "No resolver configured"}
