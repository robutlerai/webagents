"""
Daemon Client

Client for communicating with webagentsd over HTTP.
"""

import httpx
from typing import Optional, Dict, Any, List
from pathlib import Path


class DaemonClient:
    """Client for communicating with webagentsd"""
    
    def __init__(self, base_url: str = "http://localhost:8765"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def is_running(self) -> bool:
        """Check if daemon is running"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    async def run_agent(self, name: str, trigger: str = "api") -> Dict[str, Any]:
        """Run a registered agent (on-demand execution)"""
        response = await self.client.post(
            f"{self.base_url}/agents/{name}/run",
            json={"trigger": trigger}
        )
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
        response = await self.client.get(f"{self.base_url}/agents", params=params)
        response.raise_for_status()
        return response.json()["agents"]
    
    async def get_agent(self, name: str) -> Dict[str, Any]:
        """Get agent details"""
        response = await self.client.get(f"{self.base_url}/agents/{name}")
        response.raise_for_status()
        return response.json()
    
    async def register_agent(self, path: Path) -> Dict[str, Any]:
        """Register an agent from file"""
        response = await self.client.post(
            f"{self.base_url}/agents/register",
            json={"path": str(path)}
        )
        response.raise_for_status()
        return response.json()
    
    async def unregister_agent(self, name: str) -> Dict[str, Any]:
        """Unregister an agent"""
        response = await self.client.post(
            f"{self.base_url}/agents/unregister",
            json={"name": name}
        )
        response.raise_for_status()
        return response.json()
    
    async def chat(self, agent_name: str, message: str, history: List[Dict]) -> Dict:
        """Send chat message to agent running on daemon"""
        # Convert to OpenAI format
        messages = history + [{"role": "user", "content": message}]
        
        response = await self.client.post(
            f"{self.base_url}/{agent_name}/chat/completions",
            json={
                "messages": messages,
                "stream": False,
                "model": "gpt-4o-mini" # Default model if not specified
            }
        )
        response.raise_for_status()
        return response.json()

    async def chat_stream(self, agent_name: str, message: str, history: List[Dict]):
        """Stream chat responses from daemon"""
        import logging
        logger = logging.getLogger(__name__)
        import time
        start_time = time.time()
        logger.debug(f"STREAM START: {start_time}")
        
        messages = history + [{"role": "user", "content": message}]
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/{agent_name}/chat/completions",
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
        response = await self.client.get(f"{self.base_url}/{agent_name}/command")
        response.raise_for_status()
        return response.json().get("commands", [])
    
    async def execute_command(self, agent_name: str, path: str, data: Dict[str, Any] = None) -> Any:
        """Execute a command on an agent
        
        Args:
            agent_name: Name of the agent
            path: Command path (e.g., "/checkpoint/create")
            data: Command arguments
            
        Returns:
            Command result
        """
        # Strip leading slash for URL path
        url_path = path.lstrip("/")
        response = await self.client.post(
            f"{self.base_url}/{agent_name}/command/{url_path}",
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
        response = await self.client.get(f"{self.base_url}/{agent_name}/command/{url_path}")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()
