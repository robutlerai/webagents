"""
Extension Interface

Base interfaces for WebAgents extension system.

Note: "Extensions" provide agent sources, skills, and hooks to webagentsd.
This was previously called "plugins" but renamed to avoid confusion with
Claude Code plugins (which are now supported via SkillPlugin).
"""

from abc import ABC, abstractmethod
import warnings
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from webagents.agents.core.base_agent import BaseAgent
    from ..core.app import WebAgentsServer


class AgentSource(ABC):
    """Interface for agent sources (file, database, API, etc.)
    
    Caching:
        Sources that cache agents should implement invalidate() and invalidate_all().
        The server will call these when it receives cache invalidation events
        (file changes, Redis pub/sub messages, webhooks, etc.)
    """
    
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
    
    # Optional caching methods - implement if source caches agents
    def invalidate(self, name: str) -> bool:
        """Invalidate cached agent by name.
        
        Called by server when cache invalidation event is received
        (file change, Redis pub/sub, webhook, etc.)
        
        Returns:
            True if agent was in cache and invalidated, False otherwise
        """
        return False
    
    def invalidate_all(self) -> int:
        """Invalidate all cached agents.
        
        Returns:
            Number of agents invalidated
        """
        return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics (optional).
        
        Returns:
            Cache stats dict with keys like 'cached_agents', 'cache_size', etc.
        """
        return {"cached_agents": [], "cache_size": 0}


class WebAgentsExtension(ABC):
    """Base extension interface for extending webagentsd.
    
    Extensions provide:
    - Agent sources (databases, APIs, file systems)
    - Skills (capabilities for agents)
    - Hooks (lifecycle event handlers)
    - Auth handlers (custom authentication)
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """Extension name"""
        pass
    
    @abstractmethod
    async def initialize(self, server: "WebAgentsServer"):
        """Initialize extension with server instance"""
        pass
    
    @abstractmethod
    def get_agent_sources(self) -> List[AgentSource]:
        """Return agent sources provided by this extension"""
        pass
    
    @abstractmethod
    def get_skills(self) -> Dict[str, Any]:
        """Return skill classes provided by this extension"""
        pass
    
    def get_auth_handler(self) -> Optional[Callable]:
        """Return authentication handler (optional)"""
        return None
    
    def get_hooks(self) -> Dict[str, Callable]:
        """Return hooks for server events (optional)"""
        return {}


# Backwards compatibility alias (deprecated)
class WebAgentsPlugin(WebAgentsExtension):
    """Deprecated: Use WebAgentsExtension instead."""
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            "WebAgentsPlugin is deprecated, use WebAgentsExtension instead",
            DeprecationWarning,
            stacklevel=2
        )
