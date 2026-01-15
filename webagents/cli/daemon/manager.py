"""
Agent Lifecycle Manager

Manage starting, stopping, and monitoring agents.
"""

import asyncio
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime

from .registry import DaemonRegistry, DaemonAgent


class AgentManager:
    """Manage agent lifecycle."""
    
    def __init__(self, registry: DaemonRegistry):
        """Initialize manager.
        
        Args:
            registry: Daemon registry
        """
        self.registry = registry
        self._running_agents: Dict[str, asyncio.Task] = {}
        self._agent_logs: Dict[str, List[str]] = {}
    
    async def start(self, name: str, prompt: Optional[str] = None) -> bool:
        """Start an agent.
        
        Args:
            name: Agent name
            prompt: Optional initial prompt
            
        Returns:
            Success
        """
        agent = self.registry.get(name)
        if not agent:
            raise ValueError(f"Agent not found: {name}")
        
        if name in self._running_agents:
            raise ValueError(f"Agent already running: {name}")
        
        # Create agent task
        task = asyncio.create_task(self._run_agent(agent, prompt))
        self._running_agents[name] = task
        
        # Update status
        agent.status = "running"
        agent.started_at = datetime.utcnow()
        
        self._log(name, f"Agent started: {name}")
        return True
    
    async def stop(self, name: str) -> bool:
        """Stop a running agent.
        
        Args:
            name: Agent name
            
        Returns:
            Success
        """
        if name not in self._running_agents:
            return False
        
        # Cancel task
        task = self._running_agents[name]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        del self._running_agents[name]
        
        # Update status
        agent = self.registry.get(name)
        if agent:
            agent.status = "stopped"
        
        self._log(name, f"Agent stopped: {name}")
        return True
    
    async def restart(self, name: str) -> bool:
        """Restart an agent.
        
        Args:
            name: Agent name
            
        Returns:
            Success
        """
        await self.stop(name)
        await asyncio.sleep(0.1)
        return await self.start(name)
    
    async def stop_all(self):
        """Stop all running agents."""
        for name in list(self._running_agents.keys()):
            await self.stop(name)
    
    async def _run_agent(self, agent: DaemonAgent, prompt: Optional[str] = None):
        """Run agent loop.
        
        Args:
            agent: Agent to run
            prompt: Optional initial prompt
        """
        try:
            # Load agent
            from ..loader import AgentLoader
            loader = AgentLoader()
            merged = loader.load(Path(agent.source_path))
            
            self._log(agent.name, f"Loaded agent from: {agent.source_path}")
            
            # TODO: Create actual agent runtime
            # For now, just keep running
            while True:
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            self._log(agent.name, "Agent task cancelled")
            raise
        except Exception as e:
            self._log(agent.name, f"Agent error: {e}")
            agent.status = "error"
            raise
    
    def get_status(self, name: str) -> str:
        """Get agent status.
        
        Args:
            name: Agent name
            
        Returns:
            Status string
        """
        if name in self._running_agents:
            return "running"
        
        agent = self.registry.get(name)
        return agent.status if agent else "unknown"
    
    def get_running_agents(self) -> List[str]:
        """Get list of running agent names."""
        return list(self._running_agents.keys())
    
    def get_logs(self, name: str, lines: int = 100) -> List[str]:
        """Get agent logs.
        
        Args:
            name: Agent name
            lines: Number of lines to return
            
        Returns:
            Log lines
        """
        logs = self._agent_logs.get(name, [])
        return logs[-lines:]
    
    def _log(self, agent_name: str, message: str):
        """Add log entry for agent.
        
        Args:
            agent_name: Agent name
            message: Log message
        """
        if agent_name not in self._agent_logs:
            self._agent_logs[agent_name] = []
        
        timestamp = datetime.utcnow().isoformat()
        self._agent_logs[agent_name].append(f"[{timestamp}] {message}")
        
        # Trim to last 1000 entries
        if len(self._agent_logs[agent_name]) > 1000:
            self._agent_logs[agent_name] = self._agent_logs[agent_name][-1000:]
