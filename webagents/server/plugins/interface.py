"""
Plugin Interface

Base interfaces for WebAgents plugin system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent
    from ..core.app import WebAgentsServer


class AgentSource(ABC):
    """Interface for agent sources (file, database, API, etc.)"""
    
    @abstractmethod
    async def get_agent(self, name: str) -> Optional["BaseAgent"]:
        """Get agent by name"""
        pass
    
    @abstractmethod
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents"""
        pass
    
    @abstractmethod
    async def search_agents(self, query: str) -> List[Dict[str, Any]]:
        """Search agents by name pattern or metadata
        
        Args:
            query: Search query (supports wildcards like "customer-*")
        
        Returns:
            List of matching agent metadata
        """
        pass
    
    @abstractmethod
    async def filter_agents(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter agents by criteria
        
        Args:
            criteria: Filter criteria (e.g., {"source": "local", "tags": ["production"]})
        
        Returns:
            List of matching agent metadata
        """
        pass
    
    @abstractmethod
    async def refresh(self):
        """Refresh agent list (for caching sources)"""
        pass
    
    @abstractmethod
    def get_source_type(self) -> str:
        """Return source type (local, portal, database, etc.)"""
        pass


class WebAgentsPlugin(ABC):
    """Base plugin interface"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Plugin name"""
        pass
    
    @abstractmethod
    async def initialize(self, server: "WebAgentsServer"):
        """Initialize plugin with server instance"""
        pass
    
    @abstractmethod
    def get_agent_sources(self) -> List[AgentSource]:
        """Return agent sources provided by this plugin"""
        pass
    
    @abstractmethod
    def get_skills(self) -> Dict[str, Any]:
        """Return skill classes provided by this plugin"""
        pass
    
    def get_auth_handler(self) -> Optional[Callable]:
        """Return authentication handler (optional)"""
        return None
    
    def get_hooks(self) -> Dict[str, Callable]:
        """Return hooks for server events (optional)"""
        return {}
