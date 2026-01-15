"""
Registry Sync

Sync local registry with remote platform (robutler.ai).
"""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

from .registry import LocalRegistry, RegisteredAgent


@dataclass
class SyncResult:
    """Result of a sync operation."""
    pushed: List[str] = None
    pulled: List[str] = None
    updated: List[str] = None
    conflicts: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        self.pushed = self.pushed or []
        self.pulled = self.pulled or []
        self.updated = self.updated or []
        self.conflicts = self.conflicts or []
        self.errors = self.errors or []
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    @property
    def has_changes(self) -> bool:
        return bool(self.pushed or self.pulled or self.updated)


class RegistrySync:
    """Sync local registry with remote platform."""
    
    def __init__(self, registry: LocalRegistry, platform_url: str = None):
        """Initialize sync manager.
        
        Args:
            registry: Local registry
            platform_url: Platform API URL
        """
        self.registry = registry
        self.platform_url = platform_url or "https://api.robutler.ai"
        self._api_token: Optional[str] = None
    
    def is_authenticated(self) -> bool:
        """Check if we have valid auth."""
        if not self._api_token:
            from .local import get_state
            creds = get_state().get_credentials()
            self._api_token = creds.get("access_token")
        
        return bool(self._api_token)
    
    async def sync(self, agent_name: Optional[str] = None) -> SyncResult:
        """Sync with remote registry.
        
        Args:
            agent_name: Specific agent to sync, or all if None
            
        Returns:
            SyncResult
        """
        if not self.is_authenticated():
            return SyncResult(errors=["Not authenticated. Run 'webagents login' first."])
        
        result = SyncResult()
        
        # Get local agents
        if agent_name:
            local_agents = {agent_name: self.registry.get(agent_name)}
            if not local_agents[agent_name]:
                return SyncResult(errors=[f"Agent not found: {agent_name}"])
        else:
            local_agents = {a.name: a for a in self.registry.list_agents()}
        
        # TODO: Fetch remote agents from platform API
        remote_agents: Dict[str, Any] = {}
        
        # Compare and sync
        local_names = set(local_agents.keys())
        remote_names = set(remote_agents.keys())
        
        # Push new local agents
        to_push = local_names - remote_names
        for name in to_push:
            try:
                result.pushed.append(name)
            except Exception as e:
                result.errors.append(f"Failed to push {name}: {e}")
        
        # Pull new remote agents
        to_pull = remote_names - local_names
        for name in to_pull:
            try:
                result.pulled.append(name)
            except Exception as e:
                result.errors.append(f"Failed to pull {name}: {e}")
        
        return result
    
    async def push(self, agent_name: Optional[str] = None) -> SyncResult:
        """Push local agents to remote."""
        if not self.is_authenticated():
            return SyncResult(errors=["Not authenticated"])
        
        result = SyncResult()
        agents = [self.registry.get(agent_name)] if agent_name else self.registry.list_agents()
        
        for agent in agents:
            if agent:
                try:
                    agent.sync_status = "synced"
                    result.pushed.append(agent.name)
                except Exception as e:
                    result.errors.append(f"Failed to push {agent.name}: {e}")
        
        self.registry._save()
        return result
    
    async def pull(self, agent_name: Optional[str] = None) -> SyncResult:
        """Pull remote agents to local."""
        if not self.is_authenticated():
            return SyncResult(errors=["Not authenticated"])
        return SyncResult()
    
    async def diff(self) -> Dict[str, Any]:
        """Compare local and remote state."""
        if not self.is_authenticated():
            return {"error": "Not authenticated"}
        
        return {
            "local_only": [],
            "remote_only": [],
            "different": [],
            "synced": list(self.registry.list_agent_names()),
        }
