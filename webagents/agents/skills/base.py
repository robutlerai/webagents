"""
Base Skill Interface - WebAgents V2.0

Core skill interface with unified context access, automatic registration support,
and dependency resolution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..core.base_agent import BaseAgent


class Skill(ABC):
    """Base interface for agent skills with unified context access
    
    Skills have access to everything through a single Context object:
    - During initialization: Basic agent reference for registration
    - During request processing: Full Context via get_context()
    
    The Context contains BOTH request data AND agent capabilities:
    - Request: messages, user, streaming, usage, etc.  
    - Agent: skills, tools, hooks, capabilities, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None, scope: str = "all", dependencies: List[str] = None):
        self.config = config or {}
        self.scope = scope  # "all", "owner", "admin" - controls skill availability  
        self.dependencies = dependencies or []  # List of skill names this skill depends on
        self.skill_name = self.__class__.__name__  # For tracking registration source
        self.agent = None  # BaseAgent reference - set during initialization
    
    async def initialize(self, agent: 'BaseAgent') -> None:
        """
        Initialize skill with agent reference.
        Skills should register their tools, hooks, and handoffs here.
        
        Args:
            agent: The BaseAgent instance (for registration only)
        """
        self.agent = agent
        # Subclasses implement their registration logic here
    
    def get_tools(self) -> List[Callable]:
        """Return tools that this skill provides (from agent's central registry)"""
        if not self.agent:
            return []
        return [tool['function'] for tool in self.agent.get_all_tools() 
                if tool.get('source') == self.skill_name]
    
    def register_tool(self, tool_func: Callable, scope: str = None) -> None:
        """Register a tool with the agent (central registration)
        
        Can be called during initialization or at runtime from hooks/tools.
        """
        if not self.agent:
            raise RuntimeError("Cannot register tool: skill not initialized")
        
        # Use provided scope or fall back to skill's default scope
        effective_scope = scope if scope is not None else self.scope
        
        # Allow skill to override tool scope if provided
        if scope and hasattr(tool_func, '_tool_scope'):
            tool_func._tool_scope = scope
            
        # Register with agent's central registry
        self.agent.register_tool(tool_func, source=self.skill_name, scope=effective_scope)
    
    def register_hook(self, event: str, handler: Callable, priority: int = 50) -> None:
        """Register a hook for lifecycle events (central registration)
        
        Hooks receive and return the unified Context object containing everything.
        """
        if not self.agent:
            raise RuntimeError("Cannot register hook: skill not initialized")
            
        # Register with agent's central registry
        self.agent.register_hook(event, handler, priority, source=self.skill_name)
    
    def register_handoff(self, handoff_config: 'Handoff') -> None:
        """Register a handoff configuration"""
        if not self.agent:
            raise RuntimeError("Cannot register handoff: skill not initialized")
            
        # Register with agent's central registry
        self.agent.register_handoff(handoff_config, source=self.skill_name)
    
    # Everything accessible via unified Context during request processing
    def get_context(self) -> Optional['Context']:
        """
        Get unified Context containing EVERYTHING:
        
        Request data:
        - context.messages, context.user, context.stream
        - context.track_usage(), context.get()/set()
        
        Agent capabilities:
        - context.agent_skills, context.agent_tools, context.agent_handoffs
        - context.agent (BaseAgent instance)
        """
        from ...server.context.context_vars import get_context
        return get_context()
    
    def get_dependencies(self) -> List[str]:
        """Return list of skill dependencies"""
        return self.dependencies.copy()


# Base dataclasses for handoffs and other components
@dataclass
class Handoff:
    """Configuration for handoff operations"""
    target: str
    handoff_type: str
    description: str = ""
    scope: Union[str, List[str]] = "all"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass  
class HandoffResult:
    """Result from handoff execution"""
    result: Any
    handoff_type: str
    metadata: Dict[str, Any] = None
    success: bool = True
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {} 