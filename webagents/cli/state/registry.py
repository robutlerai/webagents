"""
Local Agent Registry

Manage registered agents in .webagents/registry.json.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from pydantic import BaseModel


class RegisteredAgent(BaseModel):
    """A registered agent in the local registry."""
    
    # Identity
    name: str
    namespace: str = "local"
    
    # Source
    source_path: str  # Path to AGENT.md or AGENT-*.md
    source: str = "local"  # "local" or "remote"
    
    # Metadata
    description: str = ""
    intents: List[str] = []
    
    # Status
    status: str = "registered"  # registered, running, stopped, error
    pid: Optional[int] = None
    
    # Timestamps
    registered_at: datetime = None
    last_run: Optional[datetime] = None
    
    # Sync status
    sync_status: str = "local"  # local, synced, pending, conflict
    remote_version: Optional[str] = None
    
    def __init__(self, **data):
        if 'registered_at' not in data or data['registered_at'] is None:
            data['registered_at'] = datetime.utcnow()
        super().__init__(**data)


class LocalRegistry:
    """Manage local agent registry."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize registry.
        
        Args:
            registry_path: Path to registry.json file
        """
        if registry_path:
            self.path = Path(registry_path)
        else:
            from .local import get_state
            self.path = get_state().get_registry_path()
        
        self.agents: Dict[str, RegisteredAgent] = {}
        self._load()
    
    def _load(self):
        """Load registry from file."""
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                for name, agent_data in data.get("agents", {}).items():
                    self.agents[name] = RegisteredAgent(**agent_data)
            except (json.JSONDecodeError, Exception):
                self.agents = {}
    
    def _save(self):
        """Save registry to file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.utcnow().isoformat(),
            "agents": {
                name: agent.model_dump(mode='json')
                for name, agent in self.agents.items()
            }
        }
        self.path.write_text(json.dumps(data, indent=2, default=str))
    
    def register(self, agent_file) -> RegisteredAgent:
        """Register an agent from file.
        
        Args:
            agent_file: AgentFile instance
            
        Returns:
            RegisteredAgent
        """
        registered = RegisteredAgent(
            name=agent_file.metadata.name,
            namespace=agent_file.metadata.namespace,
            source_path=str(agent_file.path),
            source="local",
            description=agent_file.metadata.description,
            intents=agent_file.metadata.intents,
        )
        
        self.agents[registered.name] = registered
        self._save()
        return registered
    
    def unregister(self, name: str) -> bool:
        """Remove agent from registry.
        
        Args:
            name: Agent name
            
        Returns:
            True if removed, False if not found
        """
        if name in self.agents:
            del self.agents[name]
            self._save()
            return True
        return False
    
    def get(self, name: str) -> Optional[RegisteredAgent]:
        """Get agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            RegisteredAgent or None
        """
        return self.agents.get(name)
    
    def find_by_path(self, path: Path) -> Optional[RegisteredAgent]:
        """Find agent by source path.
        
        Args:
            path: Path to agent file
            
        Returns:
            RegisteredAgent or None
        """
        path_str = str(path)
        for agent in self.agents.values():
            if agent.source_path == path_str:
                return agent
        return None
    
    def find_by_name(self, name: str) -> Optional[RegisteredAgent]:
        """Find agent by name (alias for get)."""
        return self.get(name)
    
    def list_agents(
        self,
        namespace: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[RegisteredAgent]:
        """List registered agents with optional filters.
        
        Args:
            namespace: Filter by namespace
            status: Filter by status
            source: Filter by source (local/remote)
            
        Returns:
            List of matching agents
        """
        agents = list(self.agents.values())
        
        if namespace:
            agents = [a for a in agents if a.namespace == namespace]
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        if source:
            agents = [a for a in agents if a.source == source]
        
        return agents
    
    def list_agent_names(self) -> List[str]:
        """List all agent names."""
        return list(self.agents.keys())
    
    def update_status(self, name: str, status: str, pid: int = None):
        """Update agent status.
        
        Args:
            name: Agent name
            status: New status
            pid: Process ID (if running)
        """
        if name in self.agents:
            self.agents[name].status = status
            self.agents[name].pid = pid
            if status == "running":
                self.agents[name].last_run = datetime.utcnow()
            self._save()
    
    def scan_directory(self, path: Path, recursive: bool = True) -> List[Path]:
        """Scan directory for agent files.
        
        Args:
            path: Directory to scan
            recursive: Include subdirectories
            
        Returns:
            List of agent file paths
        """
        pattern = "**/" if recursive else ""
        agents = []
        
        # Find AGENT.md files
        agents.extend(path.glob(f"{pattern}AGENT.md"))
        
        # Find AGENT-*.md files  
        agents.extend(path.glob(f"{pattern}AGENT-*.md"))
        
        return agents
    
    def sync_from_directory(self, path: Path, recursive: bool = True) -> int:
        """Scan directory and register all found agents.
        
        Args:
            path: Directory to scan
            recursive: Include subdirectories
            
        Returns:
            Number of agents registered
        """
        from ..loader import AgentFile
        
        count = 0
        for agent_path in self.scan_directory(path, recursive):
            try:
                agent_file = AgentFile(agent_path)
                self.register(agent_file)
                count += 1
            except Exception:
                pass
        
        return count
    
    def get_running_agents(self) -> List[RegisteredAgent]:
        """Get currently running agents."""
        return [a for a in self.agents.values() if a.status == "running"]
    
    def to_dict(self) -> Dict:
        """Export registry as dictionary."""
        return {
            name: agent.model_dump()
            for name, agent in self.agents.items()
        }
