"""
Daemon Client

Client for communicating with webagentsd over HTTP.
"""

import httpx
from typing import Optional, Dict, Any, List
from pathlib import Path


class DaemonClient:
    """Client for communicating with webagentsd
    
    The client uses a configurable agents_prefix to construct URLs.
    By default, agent routes are at /agents/{name}/...
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        agents_prefix: str = "/agents",
    ):
        """Initialize daemon client.
        
        Args:
            base_url: Base URL of the daemon server (e.g., "http://localhost:8765")
            agents_prefix: URL prefix for agent routes (default: "/agents")
        """
        self.base_url = base_url.rstrip("/")
        self.agents_prefix = agents_prefix.rstrip("/") if agents_prefix else ""
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def _agents_url(self, path: str = "") -> str:
        """Build URL for agent endpoints.
        
        Args:
            path: Path after the agents prefix (e.g., "/{name}/command")
            
        Returns:
            Full URL like "http://localhost:8765/agents/{name}/command"
        """
        if path and not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{self.agents_prefix}{path}"
    
    async def is_running(self) -> bool:
        """Check if daemon is running"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    async def health(self) -> Dict[str, Any]:
        """Get daemon health status
        
        Returns:
            Health status dict with 'status' key
            
        Raises:
            httpx.HTTPError if daemon is not reachable
        """
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def list_agents(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """List or search registered agents
        
        Args:
            query: Optional search query (name pattern)
        
        Returns:
            List of agent metadata
        """
        params = {"query": query} if query else {}
        response = await self.client.get(self._agents_url("/"), params=params)
        response.raise_for_status()
        return response.json()["agents"]
    
    async def get_agent(self, name: str) -> Dict[str, Any]:
        """Get agent details
        
        Args:
            name: Agent name
            
        Returns:
            Agent metadata dict
        """
        response = await self.client.get(self._agents_url(f"/{name}"))
        response.raise_for_status()
        return response.json()
    
    async def register_agent(self, path: Path) -> Dict[str, Any]:
        """Register an agent from file
        
        Args:
            path: Path to agent file (AGENT.md)
            
        Returns:
            Registered agent metadata
        """
        response = await self.client.post(
            self._agents_url("/"),
            json={"path": str(path)}
        )
        response.raise_for_status()
        return response.json()
    
    async def unregister_agent(self, name: str) -> Dict[str, Any]:
        """Unregister an agent
        
        Args:
            name: Agent name
            
        Returns:
            Unregistration result
        """
        response = await self.client.delete(self._agents_url(f"/{name}"))
        response.raise_for_status()
        return response.json()
    
    async def chat(self, agent_name: str, message: str, history: List[Dict]) -> Dict:
        """Send chat message to agent
        
        Args:
            agent_name: Name of the agent
            message: User message
            history: Conversation history
            
        Returns:
            Chat completion response
        """
        messages = history + [{"role": "user", "content": message}]
        
        response = await self.client.post(
            self._agents_url(f"/{agent_name}/chat/completions"),
            json={
                "messages": messages,
                "stream": False,
                "model": "gpt-4o-mini"
            }
        )
        response.raise_for_status()
        return response.json()

    async def chat_stream(self, agent_name: str, message: str, history: List[Dict]):
        """Stream chat responses from agent
        
        Args:
            agent_name: Name of the agent
            message: User message
            history: Conversation history
            
        Yields:
            SSE data chunks (without "data: " prefix)
        """
        import logging
        logger = logging.getLogger(__name__)
        import time
        start_time = time.time()
        logger.debug(f"STREAM START: {start_time}")
        
        messages = history + [{"role": "user", "content": message}]
        
        async with self.client.stream(
            "POST",
            self._agents_url(f"/{agent_name}/chat/completions"),
            json={
                "messages": messages,
                "stream": True,
                "model": "gpt-4o-mini"
            }
        ) as response:
            response.raise_for_status()
            logger.debug(f"STREAM HEADERS: {time.time() - start_time:.3f}s")
            
            async for line in response.aiter_lines():
                if line.strip():
                    logger.debug(f"STREAM CHUNK: {time.time() - start_time:.3f}s | {line[:50]}...")
                
                if line.startswith("data: "):
                    yield line[6:]  # Strip "data: " prefix
    
    async def list_commands(self, agent_name: str) -> List[Dict[str, Any]]:
        """List available commands for an agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of command info dicts
        """
        import logging
        logger = logging.getLogger(__name__)
        
        url = self._agents_url(f"/{agent_name}/command")
        logger.debug(f"Fetching commands from: {url}")
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        commands = data.get("commands", [])
        logger.debug(f"Received {len(commands)} commands for '{agent_name}'")
        return commands
    
    async def execute_command(self, agent_name: str, path: str, data: Dict[str, Any] = None) -> Any:
        """Execute a command on an agent
        
        Args:
            agent_name: Name of the agent
            path: Command path (e.g., "/checkpoint/create")
            data: Command arguments
            
        Returns:
            Command result
        """
        url_path = path.lstrip("/")
        response = await self.client.post(
            self._agents_url(f"/{agent_name}/command/{url_path}"),
            json=data or {}
        )
        response.raise_for_status()
        return response.json()
    
    async def get_command_docs(self, agent_name: str, path: str) -> Dict[str, Any]:
        """Get documentation for a specific command
        
        Args:
            agent_name: Name of the agent
            path: Command path (e.g., "/checkpoint/create")
            
        Returns:
            Command documentation dict
        """
        url_path = path.lstrip("/")
        response = await self.client.get(
            self._agents_url(f"/{agent_name}/command/{url_path}")
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()
