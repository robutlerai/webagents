"""
Context Variables and Unified Context - WebAgents V2.0

Unified context management using Python's contextvars for thread-safe,
async-compatible context handling.
"""

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


# Single ContextVar for unified context
CONTEXT: ContextVar['Context'] = ContextVar('webagents_context')


@dataclass
class Context:
    """Unified Context containing ALL request data AND agent capabilities
    
    This replaces the previous multiple context variables with a single,
    comprehensive context object that contains everything skills and
    tools need access to.
    """
    
    # Request identification
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    
    # Request data
    messages: List[Dict[str, Any]] = field(default_factory=list)
    stream: bool = False
    
    agent: Optional[Any] = None  # BaseAgent instance
    request: Optional[Any] = None  # Incoming HTTP request wrapper

    # Authentication namespace (populated by AuthSkill)
    auth: Optional[Any] = None

    # Usage data
    usage: List[Dict[str, Any]] = field(default_factory=list)
    
    # Custom data storage (for skills/tools to store temporary data)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    # --- Properties for clean access ---
    
    @property
    def skills(self) -> Dict[str, Any]:
        """Access to agent's skills"""
        return self.agent.skills if self.agent else {}

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """Access to agent's registered tools"""
        return self.agent.get_all_tools() if self.agent else []
    
    @property
    def handoffs(self) -> List[Dict[str, Any]]:
        """Access to agent's registered handoffs"""  
        return self.agent.get_all_handoffs() if self.agent else []
    
    @property
    def auth_scope(self) -> str:
        """Determine user scope based on user context"""
        return self.auth.scope.value if self.auth else 'all'
    
    # --- Methods for data manipulation ---
    
    def set(self, key: str, value: Any) -> None:
        """Store custom data"""
        self.custom_data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve custom data"""
        return self.custom_data.get(key, default)
    
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add message to context"""
        self.messages.append(message)
    
    def update_agent_context(self, agent: Any, agent_name: str) -> None:
        """Update agent-related context"""
        self.agent = agent
        self.agent_name_resolved = agent_name


# Context management functions
def get_context() -> Optional[Context]:
    """Get current context from context variable"""
    try:
        return CONTEXT.get()
    except LookupError:
        return None


def set_context(context: Context) -> None:
    """Set current context in context variable"""
    CONTEXT.set(context)


def create_context(
    request_id: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
    request: Any = None,
    agent: Any = None
) -> Context:
    """Create new context with provided data"""
    context = Context(
        request_id=request_id or str(uuid.uuid4()),
        messages=messages or [],
        stream=stream,
        agent=agent,
        request=request
    )
    return context
