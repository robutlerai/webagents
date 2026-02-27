"""
Base Skill Interface - WebAgents V2.0

Core skill interface with unified context access, automatic registration support,
and dependency resolution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union, TYPE_CHECKING, AsyncGenerator
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
    
    def _sign_response(self, response_text: str, request_payload: str) -> Optional[str]:
        """Sign an NLI response using the agent's AuthSkill keys (if available).

        Returns an RS256 JWT signature string, or None if the agent has no
        signing keys configured. Signatures are always optional.
        """
        if not self.agent:
            return None
        try:
            auth_skill = None
            for skill in getattr(self.agent, '_skills', []):
                cls_name = type(skill).__name__
                if cls_name == 'AuthSkill':
                    auth_skill = skill
                    break
            if not auth_skill or not getattr(auth_skill, '_kid', None):
                return None

            from webagents.agents.skills.robutler.nli.signing import sign_nli_response
            agent_id = getattr(auth_skill, '_agent_id', None) or getattr(self.agent, 'name', 'unknown')
            private_key = auth_skill.jwks.get_signing_key()
            kid = auth_skill._kid
            return sign_nli_response(
                response_text=response_text,
                request_payload=request_payload,
                callee_agent_id=agent_id,
                caller_agent_id=None,
                private_key=private_key,
                kid=kid,
            )
        except Exception:
            return None

    def request_handoff(self, target_name: str) -> str:
        """Return a handoff request marker for the given target
        
        This is a helper for skills that want to trigger dynamic handoff.
        Return this value from a tool, and the framework will execute the handoff.
        
        Args:
            target_name: Target name of the handoff to switch to
        
        Returns:
            Handoff request marker string
        
        Example:
            @tool
            async def use_specialist(self):
                return self.request_handoff("specialist_agent")
        """
        return f"__HANDOFF_REQUEST__:{target_name}"

    async def execute_handoff(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        handoff_name: Optional[str] = None,
        **kwargs
    ) -> 'AsyncGenerator[Dict[str, Any], None]':
        """Execute the handoff system and yield streaming chunks
        
        This is the primary method for transport skills to connect endpoints
        to LLM processing. Use in @http or @websocket handlers to route
        messages through the agent's handoff system.
        
        Args:
            messages: OpenAI-format messages to process
            tools: Optional tools to make available (merged with agent tools)
            handoff_name: Specific handoff to use (defaults to active handoff)
            **kwargs: Additional OpenAI parameters (temperature, max_tokens, etc.)
        
        Yields:
            OpenAI-format streaming chunks from the handoff
        
        Example:
            @http("/completions", method="post")
            async def completions(self, request: Request) -> AsyncGenerator[str, None]:
                body = await request.json()
                async for chunk in self.execute_handoff(body["messages"]):
                    yield f"data: {json.dumps(chunk)}\\n\\n"
            
            @websocket("/chat")
            async def chat(self, ws: WebSocket) -> None:
                await ws.accept()
                async for msg in ws.iter_json():
                    async for chunk in self.execute_handoff(msg["messages"]):
                        await ws.send_json(chunk)
        """
        from typing import AsyncGenerator
        
        context = self.get_context()
        if not context:
            raise RuntimeError("execute_handoff() requires an active request context")
        
        agent = context.agent
        if not agent:
            raise RuntimeError("execute_handoff() requires an agent in context")
        
        # Switch to specified handoff if requested
        original_handoff = None
        if handoff_name:
            # Find the requested handoff
            for registered in agent._registered_handoffs:
                if registered['config'].target == handoff_name:
                    original_handoff = agent.active_handoff
                    agent.active_handoff = registered['config']
                    break
            else:
                raise ValueError(f"Handoff '{handoff_name}' not found")
        
        try:
            # Use run_streaming to get async generator of chunks
            # Pass through all kwargs (temperature, max_tokens, etc.)
            async for chunk in agent.run_streaming(messages, tools=tools, **kwargs):
                yield chunk
        finally:
            # Restore original handoff if we switched
            if original_handoff:
                agent.active_handoff = original_handoff


# Base dataclasses for handoffs and other components
@dataclass
class Handoff:
    """Configuration for handoff operations
    
    Handoffs are chat completion handlers that can be local LLMs or remote agents.
    The first registered handoff (lowest priority) becomes the default completion handler.
    
    Attributes:
        target: Unique identifier for this handoff
        description: Human-readable description (from @handoff prompt parameter)
        scope: Access control scope(s)
        metadata: Additional data including:
            - function: The actual handoff function
            - priority: Execution priority (lower = higher priority)
            - is_generator: Whether function is async generator (streaming)
    """
    target: str
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