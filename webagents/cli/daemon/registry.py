"""
Daemon Agent Registry

Manage agents registered with the daemon.
"""

from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

from ..loader import AgentFile


class DaemonAgent(BaseModel):
    """An agent registered with the daemon."""
    
    # Identity
    name: str
    namespace: str = "local"
    description: str = ""
    
    # Source
    source_path: str
    
    # Discovery
    intents: List[str] = []
    
    # Status
    status: str = "registered"  # registered, running, stopped, error
    started_at: Optional[datetime] = None
    
    # Triggers
    cron: Optional[str] = None
    watch_patterns: List[str] = []
    
    def to_dict(self) -> dict:
        return self.model_dump(mode='json')


class DaemonRegistry:
    """Registry for daemon-managed agents."""
    
    def __init__(self):
        """Initialize registry."""
        self.agents: Dict[str, DaemonAgent] = {}
    
    def register(self, agent_file: AgentFile) -> DaemonAgent:
        """Register an agent from file.
        
        Args:
            agent_file: Parsed agent file
            
        Returns:
            DaemonAgent
        """
        daemon_agent = DaemonAgent(
            name=agent_file.metadata.name,
            namespace=agent_file.metadata.namespace,
            description=agent_file.metadata.description,
            source_path=str(agent_file.path),
            intents=agent_file.metadata.intents,
            cron=agent_file.metadata.cron,
            watch_patterns=agent_file.metadata.watch or [],
        )
        
        self.agents[daemon_agent.name] = daemon_agent
        return daemon_agent
    
    def unregister(self, name: str) -> bool:
        """Remove agent from registry.
        
        Args:
            name: Agent name
            
        Returns:
            True if removed
        """
        if name in self.agents:
            del self.agents[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[DaemonAgent]:
        """Get agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            DaemonAgent or None
        """
        return self.agents.get(name)
    
    def find_by_path(self, path: Path) -> Optional[DaemonAgent]:
        """Find agent by source path.
        
        Args:
            path: Path to agent file
            
        Returns:
            DaemonAgent or None
        """
        path_str = str(path)
        for agent in self.agents.values():
            if agent.source_path == path_str:
                return agent
        return None
    
    def list_agents(
        self,
        namespace: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[DaemonAgent]:
        """List agents with optional filters.
        
        Args:
            namespace: Filter by namespace
            status: Filter by status
            
        Returns:
            List of agents
        """
        agents = list(self.agents.values())
        
        if namespace:
            agents = [a for a in agents if a.namespace == namespace]
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        return agents
    
    def list_agent_names(self) -> List[str]:
        """Get all agent names."""
        return list(self.agents.keys())
    
    async def scan_directory(self, path: Path, recursive: bool = True) -> int:
        """Scan directory and register agents.
        
        Args:
            path: Directory to scan
            recursive: Include subdirectories
            
        Returns:
            Number of agents registered
        """
        count = 0
        pattern = "**/" if recursive else ""
        
        # Find AGENT.md files
        for agent_path in path.glob(f"{pattern}AGENT.md"):
            try:
                agent_file = AgentFile(agent_path)
                self.register(agent_file)
                count += 1
            except Exception:
                pass
        
        # Find AGENT-*.md files
        for agent_path in path.glob(f"{pattern}AGENT-*.md"):
            try:
                agent_file = AgentFile(agent_path)
                self.register(agent_file)
                count += 1
            except Exception:
                pass
        
        return count
    
    def update_from_file(self, path: Path):
        """Update agent from file (on file change).
        
        Args:
            path: Path to changed file
        """
        try:
            agent_file = AgentFile(path)
            
            # Check if already registered
            existing = self.find_by_path(path)
            if existing:
                # Update existing agent
                existing.name = agent_file.metadata.name
                existing.namespace = agent_file.metadata.namespace
                existing.description = agent_file.metadata.description
                existing.intents = agent_file.metadata.intents
                existing.cron = agent_file.metadata.cron
                existing.watch_patterns = agent_file.metadata.watch or []
            else:
                # Register new agent
                self.register(agent_file)
        except Exception:
            pass
    
    def get_agents_with_cron(self) -> List[DaemonAgent]:
        """Get agents with cron schedules."""
        return [a for a in self.agents.values() if a.cron]
    
    def get_agents_with_watch(self) -> List[DaemonAgent]:
        """Get agents with file watch patterns."""
        return [a for a in self.agents.values() if a.watch_patterns]
