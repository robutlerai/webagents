"""
BaseAgent - WebAgents V2.0 Core Agent Implementation

Central agent implementation with automatic decorator registration,
unified context management, and comprehensive tool/hook/handoff execution.

Key Features:
- Agentic Loop: Automatically continues conversation after internal tool execution
  - Internal tools: Executed server-side, results fed back to LLM for continuation
  - External tools: Loop breaks, returns tool calls to client for execution
  - Mixed scenario: Internal tools executed first, then returns external tools
- Streaming support with proper chunk handling for tool calls
- Unified context management across all operations
"""

import asyncio
import os
import inspect  
import json
import threading
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Union, AsyncGenerator, Awaitable
from datetime import datetime

from ..skills.base import Skill, Handoff, HandoffResult
from ..tools.decorators import tool, hook, handoff, http
from ...server.context.context_vars import Context, set_context, get_context, create_context
from webagents.utils.logging import get_logger


from datetime import datetime

class BaseAgent:
    """
    BaseAgent - Core agent implementation with unified capabilities
    
    Features:
    - Automatic decorator registration (@tool, @hook, @handoff, @http)
    - Direct tools, hooks, handoffs, and HTTP handlers registration via __init__
    - Unified context management
    - Streaming and non-streaming execution
    - Scope-based access control
    - Comprehensive tool/handoff/HTTP execution
    - OpenAI-compatible tool call handling
    - Thread-safe central registry for all capabilities
    - FastAPI-style direct registration (@agent.tool, @agent.http, etc.)
    
    Initialization supports:
    - Tools: List of callable functions (with or without @tool decorator)
    - Hooks: Dict mapping events to hook functions or configurations
    - Handoffs: List of Handoff objects or @handoff decorated functions
    - HTTP handlers: List of @http decorated functions for custom endpoints
    - Capabilities: List of any decorated functions (auto-categorized)
    - Skills: Dict of skill instances with automatic capability registration
    
    HTTP Integration:
    - Custom endpoints: /{agent_name}/{subpath}
    - Conflict detection with core paths
    - FastAPI request handling
    """
    
    def __init__(
        self,
        name: str,
        instructions: str = "",
        model: Optional[Union[str, Any]] = None,
        skills: Optional[Dict[str, Skill]] = None,
        scopes: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None,
        hooks: Optional[Dict[str, List[Union[Callable, Dict[str, Any]]]]] = None,
        handoffs: Optional[List[Union[Handoff, Callable]]] = None,
        http_handlers: Optional[List[Callable]] = None,
        capabilities: Optional[List[Callable]] = None
    ):
        """Initialize BaseAgent with comprehensive configuration
        
        Args:
            name: Agent identifier (URL-safe)
            instructions: System instructions/prompt for the agent
            model: LLM model specification (string like "openai/gpt-4o" or skill instance)
            skills: Dictionary of skill instances to attach to agent
            scopes: List of access scopes for agent capabilities (e.g., ["all"], ["owner", "admin"])
                   If None, defaults to ["all"]. Common scopes: "all", "owner", "admin"
            tools: List of tool functions (with or without @tool decorator)
            hooks: Dict mapping event names to lists of hook functions or configurations
            handoffs: List of Handoff objects or functions with @handoff decorator
            http_handlers: List of HTTP handler functions (with @http decorator)
            capabilities: List of decorated functions that will be auto-registered based on their decorator type
            
        Tools can be:
            - Functions decorated with @tool
            - Plain functions (will auto-generate schema)
            
        Hooks format:
            {
                "on_request": [hook_func, {"handler": hook_func, "priority": 10}],
                "on_chunk": [hook_func],
                ...
            }
            
        Handoffs can be:
            - Handoff objects
            - Functions decorated with @handoff
            
        HTTP handlers can be:
            - Functions decorated with @http
            - Receive FastAPI request arguments directly
            
        Capabilities auto-registration:
            - Functions decorated with @tool, @hook, @handoff, @http
            - Automatically categorized and registered based on decorator type
            
        Scopes system:
            - Agent can have multiple scopes: ["owner", "admin"]
            - Capabilities inherit agent scopes unless explicitly overridden
            - Use scope management methods: add_scope(), remove_scope(), has_scope()
        """
        self.name = name
        self.instructions = instructions
        self.scopes = scopes if scopes is not None else ["all"]
        
        # Central registries (thread-safe)
        self._registered_tools: List[Dict[str, Any]] = []
        self._registered_hooks: Dict[str, List[Dict[str, Any]]] = {}
        self._registered_handoffs: List[Dict[str, Any]] = []
        self._registered_prompts: List[Dict[str, Any]] = []
        self._registered_widgets: List[Dict[str, Any]] = []
        self._registered_http_handlers: List[Dict[str, Any]] = []
        self._registration_lock = threading.Lock()
        
        # Track tools overridden by external tools (per request)
        self._overridden_tools: set = set()
        
        # Active handoff (completion handler) - set to lowest priority handoff after initialization
        self.active_handoff: Optional[Handoff] = None
        
        # Skills management
        self.skills: Dict[str, Skill] = {}
        
        # Structured logger setup (use agent name as subsystem for clear log attribution)
        self.logger = get_logger('base_agent', self.name)
        self._ensure_logger_handler()
        
        # Process model parameter and initialize skills
        skills = skills or {}
        if model:
            skills = self._process_model_parameter(model, skills)
        
        # Initialize all skills
        self._initialize_skills(skills)
        self.logger.debug(f"🧩 Initialized skills for agent='{name}' count={len(self.skills)}")
        
        # Register agent-level tools, hooks, handoffs, HTTP handlers, and capabilities
        self._register_agent_capabilities(tools, hooks, handoffs, http_handlers, capabilities)
        self.logger.info(f"🤖 BaseAgent created name='{self.name}' scopes={self.scopes}")

    def _ensure_logger_handler(self) -> None:
        """Ensure logger emits even in background contexts without adding duplicate handlers."""
        import logging
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, log_level, logging.INFO)
        # If using a LoggerAdapter (e.g., AgentContextAdapter), operate on the underlying logger
        base_logger = self.logger.logger if isinstance(self.logger, logging.LoggerAdapter) else self.logger
        # Set desired level and let it propagate to 'webagents' logger configured by setup_logging
        base_logger.setLevel(level)
        base_logger.propagate = True
    
    def _process_model_parameter(self, model: Union[str, Any], skills: Dict[str, Skill]) -> Dict[str, Skill]:
        """Process model parameter - if string, create appropriate LLM skill"""
        if isinstance(model, str) and "/" in model:
            # Format: "skill_type/model_name" (e.g., "openai/gpt-4o")
            skill_type, model_name = model.split("/", 1)
            
            if skill_type == "openai":
                from ..skills.core.llm.openai import OpenAISkill
                skills["primary_llm"] = OpenAISkill({"model": model_name})
                self.logger.debug(f"🧠 Model configured via skill=openai model='{model_name}'")
            elif skill_type == "litellm":
                from ..skills.core.llm.litellm import LiteLLMSkill  
                skills["primary_llm"] = LiteLLMSkill({"model": model_name})
                self.logger.debug(f"🧠 Model configured via skill=litellm model='{model_name}'")
            elif skill_type == "anthropic":
                from ..skills.core.llm.anthropic import AnthropicSkill
                skills["primary_llm"] = AnthropicSkill({"model": model_name})
                self.logger.debug(f"🧠 Model configured via skill=anthropic model='{model_name}'")
        
        return skills
    
    def _initialize_skills(self, skills: Dict[str, Skill]) -> None:
        """Initialize all skills and register their decorators"""
        self.skills = skills
        
        for skill_name, skill in skills.items():
            # Auto-register decorators from skill
            self._auto_register_skill_decorators(skill, skill_name)
            
            # Note: Actual skill initialization (with agent reference) will be done when needed
            # This avoids event loop issues during testing
    
    async def _ensure_skills_initialized(self) -> None:
        """Ensure all skills are initialized with agent reference"""
        for skill_name, skill in self.skills.items():
            # Check if skill needs initialization (most skills will have this method)
            if hasattr(skill, 'initialize') and callable(skill.initialize):
                # Check if already initialized by looking for agent attribute
                if not hasattr(skill, 'agent') or skill.agent is None:
                    self.logger.debug(f"🧪 Initializing skill='{skill_name}' for agent='{self.name}'")
                    await skill.initialize(self)
                    self.logger.debug(f"✅ Skill initialized skill='{skill_name}'")
    
    def _register_agent_capabilities(self, tools: Optional[List[Callable]] = None, 
                                   hooks: Optional[Dict[str, List[Union[Callable, Dict[str, Any]]]]] = None,
                                   handoffs: Optional[List[Union[Handoff, Callable]]] = None,
                                   http_handlers: Optional[List[Callable]] = None,
                                   capabilities: Optional[List[Callable]] = None) -> None:
        """Register agent-level tools, hooks, and handoffs"""
        
        # Register tools
        if tools:
            for tool_func in tools:
                # For agent-level tools, inheritance logic:
                # - Decorated tools (@tool) keep their own scope (even default "all")
                # - Undecorated tools inherit agent scopes
                if hasattr(tool_func, '_webagents_is_tool') and tool_func._webagents_is_tool:
                    # Tool is decorated - keep its own scope
                    scope = tool_func._tool_scope
                else:
                    # Tool is undecorated - inherit agent scopes
                    scope = self.scopes
                self.register_tool(tool_func, source="agent", scope=scope)
                self.logger.debug(f"🛠️ Registered agent-level tool name='{getattr(tool_func, '_tool_name', tool_func.__name__)}' scope={scope}")
        
        # Register hooks
        if hooks:
            for event, hook_list in hooks.items():
                for hook_item in hook_list:
                    if callable(hook_item):
                        # Simple function - use default priority and inherit agent scopes
                        priority = getattr(hook_item, '_hook_priority', 50)
                        scope = getattr(hook_item, '_hook_scope', self.scopes)
                        self.register_hook(event, hook_item, priority, source="agent", scope=scope)
                        self.logger.debug(f"🪝 Registered agent-level hook event='{event}' priority={priority} scope={scope}")
                    elif isinstance(hook_item, dict):
                        # Configuration dict
                        handler = hook_item.get('handler')
                        priority = hook_item.get('priority', 50)
                        scope = hook_item.get('scope', self.scopes)
                        if handler and callable(handler):
                            self.register_hook(event, handler, priority, source="agent", scope=scope)
                            self.logger.debug(f"🪝 Registered agent-level hook (dict) event='{event}' priority={priority} scope={scope}")
        
        # Register handoffs
        if handoffs:
            for handoff_item in handoffs:
                if isinstance(handoff_item, Handoff):
                    # Direct Handoff object
                    self.register_handoff(handoff_item, source="agent")
                    self.logger.debug(f"📨 Registered handoff target='{handoff_item.target}'")
                elif callable(handoff_item) and hasattr(handoff_item, '_webagents_is_handoff'):
                    # Function with @handoff decorator
                    handoff_config = Handoff(
                        target=getattr(handoff_item, '_handoff_name', handoff_item.__name__),
                        description=getattr(handoff_item, '_handoff_prompt', ''),
                        scope=getattr(handoff_item, '_handoff_scope', self.scopes)
                    )
                    handoff_config.metadata = {
                        'function': handoff_item,
                        'priority': getattr(handoff_item, '_handoff_priority', 50),
                        'is_generator': getattr(handoff_item, '_handoff_is_generator', False)
                    }
                    self.register_handoff(handoff_config, source="agent")
                    self.logger.debug(f"📨 Registered handoff target='{handoff_config.target}'")
        
        # Register HTTP handlers
        if http_handlers:
            for handler_func in http_handlers:
                if callable(handler_func):
                    self.register_http_handler(handler_func)
                    self.logger.debug(f"🌐 Registered HTTP handler subpath='{getattr(handler_func, '_http_subpath', '<unknown>')}' method='{getattr(handler_func, '_http_method', 'get')}'")
        
        # Register capabilities (decorated functions)
        if capabilities:
            for capability_func in capabilities:
                if callable(capability_func):
                    # Attempt to determine decorator type
                    if hasattr(capability_func, '_webagents_is_tool') and capability_func._webagents_is_tool:
                        self.register_tool(capability_func, source="agent")
                    elif hasattr(capability_func, '_webagents_is_hook') and capability_func._webagents_is_hook:
                        priority = getattr(capability_func, '_hook_priority', 50)
                        scope = getattr(capability_func, '_hook_scope', self.scopes)
                        self.register_hook(getattr(capability_func, '_hook_event_type', 'on_request'), capability_func, priority, source="agent", scope=scope)
                    elif hasattr(capability_func, '_webagents_is_handoff') and capability_func._webagents_is_handoff:
                        handoff_config = Handoff(
                            target=getattr(capability_func, '_handoff_name', capability_func.__name__),
                            description=getattr(capability_func, '_handoff_prompt', ''),
                            scope=getattr(capability_func, '_handoff_scope', self.scopes)
                        )
                        handoff_config.metadata = {
                            'function': capability_func,
                            'priority': getattr(capability_func, '_handoff_priority', 50),
                            'is_generator': getattr(capability_func, '_handoff_is_generator', False)
                        }
                        self.register_handoff(handoff_config, source="agent")
                    elif hasattr(capability_func, '_webagents_is_http') and capability_func._webagents_is_http:
                        self.register_http_handler(capability_func)
                        self.logger.debug(f"🌐 Registered HTTP capability subpath='{getattr(capability_func, '_http_subpath', '<unknown>')}' method='{getattr(capability_func, '_http_method', 'get')}'")
    
    # ===== SCOPE MANAGEMENT METHODS =====
    
    def add_scope(self, scope: str) -> None:
        """Add a scope to the agent if not already present
        
        Args:
            scope: Scope to add (e.g., "owner", "admin")
        """
        if scope not in self.scopes:
            self.scopes.append(scope)
    
    def remove_scope(self, scope: str) -> None:
        """Remove a scope from the agent
        
        Args:
            scope: Scope to remove
        """
        if scope in self.scopes:
            self.scopes.remove(scope)
    
    def has_scope(self, scope: str) -> bool:
        """Check if the agent has a specific scope
        
        Args:
            scope: Scope to check for
            
        Returns:
            True if agent has the scope, False otherwise
        """
        return scope in self.scopes
    
    def get_scopes(self) -> List[str]:
        """Get all scopes for this agent
        
        Returns:
            List of scope strings
        """
        return self.scopes.copy()
    
    def set_scopes(self, scopes: List[str]) -> None:
        """Set the agent's scopes list
        
        Args:
            scopes: New list of scopes
        """
        self.scopes = scopes.copy()
    
    def clear_scopes(self) -> None:
        """Clear all scopes from the agent"""
        self.scopes = []
    
    def _auto_register_skill_decorators(self, skill: Any, skill_name: str) -> None:
        """Auto-discover and register @hook, @tool, @prompt, and @handoff decorated methods"""
        import inspect
        
        for attr_name in dir(skill):
            if attr_name.startswith('_') and not attr_name.startswith('__'):
                continue
                
            attr = getattr(skill, attr_name)
            if not inspect.ismethod(attr) and not inspect.isfunction(attr):
                continue
            
            # Check for @hook decorator
            if hasattr(attr, '_webagents_is_hook') and attr._webagents_is_hook:
                event_type = attr._hook_event_type
                priority = getattr(attr, '_hook_priority', 50)
                scope = getattr(attr, '_hook_scope', None)
                self.register_hook(event_type, attr, priority, source=skill_name, scope=scope)
            
            # Check for @tool decorator  
            elif hasattr(attr, '_webagents_is_tool') and attr._webagents_is_tool:
                scope = getattr(attr, '_tool_scope', None)
                self.register_tool(attr, source=skill_name, scope=scope)
            
            # Check for @prompt decorator
            elif hasattr(attr, '_webagents_is_prompt') and attr._webagents_is_prompt:
                priority = getattr(attr, '_prompt_priority', 50)
                scope = getattr(attr, '_prompt_scope', None)
                self.register_prompt(attr, priority, source=skill_name, scope=scope)
            
            # Check for @handoff decorator
            elif hasattr(attr, '_webagents_is_handoff') and attr._webagents_is_handoff:
                handoff_config = Handoff(
                    target=getattr(attr, '_handoff_name', attr_name),
                    description=getattr(attr, '_handoff_prompt', ''),  # prompt becomes description
                    scope=getattr(attr, '_handoff_scope', None),
                    metadata={
                        'function': attr,
                        'priority': getattr(attr, '_handoff_priority', 50),
                        'is_generator': getattr(attr, '_handoff_is_generator', False)
                    }
                )
                self.register_handoff(handoff_config, source=skill_name)
                
                # Auto-create invocation tool if requested
                if hasattr(attr, '_handoff_auto_tool') and attr._handoff_auto_tool:
                    target_name = handoff_config.target
                    tool_desc = getattr(attr, '_handoff_auto_tool_description', f"Switch to {target_name} handoff")
                    
                    # Create tool function that returns handoff request marker
                    async def invoke_handoff_tool(skill_instance=skill):
                        return skill_instance.request_handoff(target_name)
                    
                    # Register as tool
                    invoke_handoff_tool.__name__ = f"use_{target_name}"
                    invoke_handoff_tool._webagents_is_tool = True
                    invoke_handoff_tool._tool_description = tool_desc
                    invoke_handoff_tool._tool_scope = handoff_config.scope
                    
                    self.register_tool(
                        invoke_handoff_tool,
                        source=f"{skill_name}_handoff_tool"
                    )
                    self.logger.debug(f"🔧 Auto-registered handoff invocation tool: use_{target_name}")
            
            # Check for @http decorator
            elif hasattr(attr, '_webagents_is_http') and attr._webagents_is_http:
                self.register_http_handler(attr, source=skill_name)
            
            # Check for @widget decorator
            elif hasattr(attr, '_webagents_is_widget') and attr._webagents_is_widget:
                scope = getattr(attr, '_widget_scope', None)
                self.register_widget(attr, source=skill_name, scope=scope)
    
    # Central registration methods (thread-safe)
    def register_tool(self, tool_func: Callable, source: str = "manual", scope: Union[str, List[str]] = None):
        """Register a tool function"""
        with self._registration_lock:
            tool_config = {
                'function': tool_func,
                'source': source,
                'scope': scope,
                'name': getattr(tool_func, '_tool_name', tool_func.__name__),
                'description': getattr(tool_func, '_tool_description', tool_func.__doc__ or ''),
                'definition': getattr(tool_func, '_webagents_tool_definition', {})
            }
            self._registered_tools.append(tool_config)
        self.logger.debug(f"🛠️ Tool registered name='{tool_config['name']}' source='{source}' scope={scope}")
    
    def register_widget(self, widget_func: Callable, source: str = "manual", scope: Union[str, List[str]] = None):
        """Register a widget function
        
        Widgets are registered both as widgets (for browser filtering) and as tools (for execution).
        """
        widget_name = getattr(widget_func, '_widget_name', widget_func.__name__)
        widget_definition = getattr(widget_func, '_webagents_widget_definition', {})
        
        with self._registration_lock:
            # Register as widget (for browser filtering)
            widget_config = {
                'function': widget_func,
                'source': source,
                'scope': scope,
                'name': widget_name,
                'description': getattr(widget_func, '_widget_description', widget_func.__doc__ or ''),
                'definition': widget_definition,
                'template': getattr(widget_func, '_widget_template', None)
            }
            self._registered_widgets.append(widget_config)
            
            # Also register as tool (for execution)
            # This allows _get_tool_function_by_name to find it and mark it as internal
            tool_config = {
                'function': widget_func,
                'name': widget_name,
                'description': getattr(widget_func, '_widget_description', widget_func.__doc__ or ''),
                'definition': widget_definition,
                'source': source,
                'scope': scope
            }
            self._registered_tools.append(tool_config)
            
        self.logger.debug(f"🎨 Widget registered name='{widget_name}' source='{source}' scope={scope} (also registered as tool for execution)")
    
    def register_hook(self, event: str, handler: Callable, priority: int = 50, source: str = "manual", scope: Union[str, List[str]] = None):
        """Register a hook handler for an event"""
        with self._registration_lock:
            if event not in self._registered_hooks:
                self._registered_hooks[event] = []
            
            hook_config = {
                'handler': handler,
                'priority': priority,
                'source': source,
                'scope': scope,
                'event': event
            }
            self._registered_hooks[event].append(hook_config)
            # Sort by priority (higher priority first)
            self._registered_hooks[event].sort(key=lambda x: x['priority'])
        self.logger.debug(f"🪝 Hook registered event='{event}' priority={priority} source='{source}' scope={scope}")
    
    def register_handoff(self, handoff_config: Handoff, source: str = "manual"):
        """Register a handoff configuration with priority-based default selection
        
        Args:
            handoff_config: Handoff configuration
            source: Source of registration (skill name, "agent", "manual")
        """
        with self._registration_lock:
            function = handoff_config.metadata.get('function')
            
            # Auto-detect if generator
            is_generator = inspect.isasyncgenfunction(function) if function else False
            priority = handoff_config.metadata.get('priority', 50)
            
            # Store metadata
            handoff_config.metadata.update({
                'is_generator': is_generator,
                'priority': priority
            })
            
            self._registered_handoffs.append({
                'config': handoff_config,
                'source': source
            })
            
            # Sort handoffs by priority (lower = higher priority)
            self._registered_handoffs.sort(key=lambda x: (
                x['config'].metadata.get('priority', 50),  # Primary: priority
                x['source'],  # Secondary: source name
                x['config'].target  # Tertiary: target name
            ))
            
            # Set as default if this is the highest priority handoff
            if not self.active_handoff or priority < self.active_handoff.metadata.get('priority', 50):
                self.active_handoff = handoff_config
                self.logger.info(f"📨 Set default handoff: {handoff_config.target} (priority={priority})")
        
        self.logger.debug(
            f"📨 Handoff registered target='{handoff_config.target}' "
            f"priority={priority} generator={is_generator} source='{source}'"
        )
        
        # Register handoff's prompt if present
        if handoff_config.description:
            self._register_handoff_prompt(handoff_config, source)
    
    def get_handoff_by_target(self, target_name: str) -> Optional[Handoff]:
        """Get handoff configuration by target name
        
        Args:
            target_name: Target name of the handoff (e.g., 'openai_workflow', 'specialist_agent')
        
        Returns:
            Handoff configuration if found, None otherwise
        """
        with self._registration_lock:
            for entry in self._registered_handoffs:
                if entry['config'].target == target_name:
                    return entry['config']
        return None

    def list_available_handoffs(self) -> List[Dict[str, Any]]:
        """List all registered handoffs with their metadata
        
        Returns:
            List of dicts with: target, description, priority, source, scope
        """
        with self._registration_lock:
            return [
                {
                    'target': entry['config'].target,
                    'description': entry['config'].description,
                    'priority': entry['config'].metadata.get('priority', 50),
                    'source': entry['source'],
                    'scope': entry['config'].scope
                }
                for entry in self._registered_handoffs
            ]
    
    def _register_handoff_prompt(self, handoff_config: Handoff, source: str):
        """Register handoff's prompt as dynamic prompt provider"""
        prompt_text = handoff_config.description
        priority = handoff_config.metadata.get('priority', 50)
        
        # Create prompt provider function
        def handoff_prompt_provider(context=None):
            return prompt_text
        
        # Register as prompt with same priority as handoff
        self.register_prompt(
            handoff_prompt_provider,
            priority=priority,
            source=f"{source}_handoff_prompt",
            scope=handoff_config.scope
        )
        
        self.logger.debug(f"📨 Registered handoff prompt for '{handoff_config.target}'")
    
    def register_prompt(self, prompt_func: Callable, priority: int = 50, source: str = "manual", scope: Union[str, List[str]] = None):
        """Register a prompt provider function"""
        with self._registration_lock:
            prompt_config = {
                'function': prompt_func,
                'priority': priority,
                'source': source,
                'scope': scope,
                'name': getattr(prompt_func, '__name__', 'unnamed_prompt')
            }
            self._registered_prompts.append(prompt_config)
            # Sort by priority (lower numbers execute first)
            self._registered_prompts.sort(key=lambda x: x['priority'])
        self.logger.debug(f"🧾 Prompt registered name='{prompt_config['name']}' priority={priority} source='{source}' scope={scope}")
    
    def register_http_handler(self, handler_func: Callable, source: str = "manual"):
        """Register an HTTP handler function with conflict detection"""
        if not hasattr(handler_func, '_webagents_is_http'):
            raise ValueError(f"Function {handler_func.__name__} is not decorated with @http")
        
        subpath = getattr(handler_func, '_http_subpath')
        method = getattr(handler_func, '_http_method')
        scope = getattr(handler_func, '_http_scope')
        description = getattr(handler_func, '_http_description')
        
        # Check for conflicts with core handlers
        core_paths = ['/chat/completions', '/info', '/capabilities']
        if subpath in core_paths:
            raise ValueError(f"HTTP subpath '{subpath}' conflicts with core handler. Core paths: {core_paths}")
        
        with self._registration_lock:
            # Check for conflicts with existing handlers
            for existing_handler in self._registered_http_handlers:
                existing_subpath = existing_handler.get('subpath')
                existing_method = existing_handler.get('method')
                if existing_subpath == subpath and existing_method == method:
                    raise ValueError(f"HTTP handler conflict: {method.upper()} {subpath} already registered")
            
            handler_config = {
                'function': handler_func,
                'source': source,
                'subpath': subpath,
                'method': method,
                'scope': scope,
                'description': description,
                'name': getattr(handler_func, '__name__', 'unnamed_handler')
            }
            self._registered_http_handlers.append(handler_config)
        self.logger.debug(f"🌐 HTTP handler registered method='{method}' subpath='{subpath}' scope={scope} source='{source}'")
    
    def get_all_hooks(self, event: str) -> List[Dict[str, Any]]:
        """Get all hooks for a specific event"""
        return self._registered_hooks.get(event, [])
    
    def get_prompts_for_scope(self, auth_scope: str) -> List[Dict[str, Any]]:
        """Get prompt providers filtered by user scope"""
        scope_hierarchy = {"admin": 3, "owner": 2, "all": 1}
        user_level = scope_hierarchy.get(auth_scope, 1)
        
        available_prompts = []
        with self._registration_lock:
            for prompt_config in self._registered_prompts:
                prompt_scope = prompt_config.get('scope', 'all')
                if isinstance(prompt_scope, list):
                    # If scope is a list, check if auth_scope is in it
                    if auth_scope in prompt_scope or 'all' in prompt_scope:
                        available_prompts.append(prompt_config)
                else:
                    # Single scope - check hierarchy
                    required_level = scope_hierarchy.get(prompt_scope, 1)
                    if user_level >= required_level:
                        available_prompts.append(prompt_config)
        
        return available_prompts
    
    def get_tools_for_scope(self, auth_scope: str) -> List[Dict[str, Any]]:
        """Get tools filtered by single user scope
        
        Args:
            auth_scope: Single scope to check against (e.g., "owner", "admin")
            
        Returns:
            List of tool configurations accessible to the user scope
        """
        return self.get_tools_for_scopes([auth_scope])
    
    def get_tools_for_scopes(self, auth_scopes: List[str]) -> List[Dict[str, Any]]:
        """Get tools filtered by multiple user scopes
        
        Args:
            auth_scopes: List of scopes to check against (e.g., ["owner", "admin"])
            
        Returns:
            List of tool configurations accessible to any of the user scopes
        """
        scope_hierarchy = {"admin": 3, "owner": 2, "all": 1}
        user_levels = [scope_hierarchy.get(scope, 1) for scope in auth_scopes]
        max_user_level = max(user_levels) if user_levels else 1
        
        available_tools = []
        with self._registration_lock:
            for tool_config in self._registered_tools:
                tool_scope = tool_config.get('scope', 'all')
                if isinstance(tool_scope, list):
                    # If scope is a list, check if any user scope is in it
                    if any(scope in tool_scope for scope in auth_scopes) or 'all' in tool_scope:
                        available_tools.append(tool_config)
                else:
                    # Single scope - check hierarchy against max user level
                    required_level = scope_hierarchy.get(tool_scope, 1)
                    if max_user_level >= required_level:
                        available_tools.append(tool_config)
        
        return available_tools
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all registered tools regardless of scope"""
        with self._registration_lock:
            return self._registered_tools.copy()
    
    def get_all_widgets(self) -> List[Dict[str, Any]]:
        """Get all registered widgets regardless of scope"""
        with self._registration_lock:
            return self._registered_widgets.copy()
    
    def get_all_http_handlers(self) -> List[Dict[str, Any]]:
        """Get all registered HTTP handlers"""
        with self._registration_lock:
            return self._registered_http_handlers.copy()
    
    def get_http_handlers_for_scope(self, auth_scope: str) -> List[Dict[str, Any]]:
        """Get HTTP handlers filtered by single user scope"""
        return self.get_http_handlers_for_scopes([auth_scope])
    
    def get_http_handlers_for_scopes(self, auth_scopes: List[str]) -> List[Dict[str, Any]]:
        """Get HTTP handlers filtered by multiple user scopes"""
        scope_hierarchy = {"admin": 3, "owner": 2, "all": 1}
        user_levels = [scope_hierarchy.get(scope, 1) for scope in auth_scopes]
        max_user_level = max(user_levels) if user_levels else 1
        
        available_handlers = []
        with self._registration_lock:
            for handler_config in self._registered_http_handlers:
                handler_scope = handler_config.get('scope', 'all')
                if isinstance(handler_scope, list):
                    # If scope is a list, check if any user scope is in it
                    if any(scope in handler_scope for scope in auth_scopes) or 'all' in handler_scope:
                        available_handlers.append(handler_config)
                else:
                    # Single scope - check hierarchy against max user level
                    required_level = scope_hierarchy.get(handler_scope, 1)
                    if max_user_level >= required_level:
                        available_handlers.append(handler_config)
        
        return available_handlers
    
    # Hook execution
    async def _execute_hooks(self, event: str, context: Context) -> Context:
        """Execute all hooks for a given event"""
        hooks = self.get_all_hooks(event)
        # try:
        #     self.logger.debug(f"⚙️ Executing hooks event='{event}' count={len(hooks)}")
        # except Exception:
        #     pass
        
        for hook_config in hooks:
            handler = hook_config['handler']
            try:
                if inspect.iscoroutinefunction(handler):
                    context = await handler(context)
                else:
                    context = handler(context)
            except Exception as e:
                # Re-raise structured errors (e.g., payment errors) immediately to halt execution
                # We duck-type on common attributes set by our error classes
                if hasattr(e, 'status_code') or hasattr(e, 'error_code') or hasattr(e, 'detail'):
                    raise e
                
                # Log other hook execution errors but continue
                self.logger.warning(f"⚠️ Hook execution error handler='{getattr(handler, '__name__', str(handler))}' error='{e}'")
                
        # try:
        #     self.logger.debug(f"⚙️ Completed hooks event='{event}'")
        # except Exception:
        #     pass
        return context
    
    # Prompt execution
    async def _execute_prompts(self, context: Context) -> str:
        """Execute all prompt providers and combine their outputs"""
        # Get user scope from context for filtering
        auth_scope = getattr(context, 'auth_scope', 'all')
        prompts = self.get_prompts_for_scope(auth_scope)
        self.logger.debug(f"🧾 Executing prompts scope='{auth_scope}' count={len(prompts)}")
        
        prompt_parts = []
        
        for prompt_config in prompts:
            handler = prompt_config['function']
            try:
                # Don't pass context explicitly - let the decorator wrapper handle it
                if inspect.iscoroutinefunction(handler):
                    prompt_part = await handler()
                else:
                    prompt_part = handler()
                
                if prompt_part and isinstance(prompt_part, str):
                    prompt_parts.append(prompt_part.strip())
            except Exception as e:
                # Log prompt execution error but continue
                self.logger.warning(f"⚠️ Prompt execution error handler='{getattr(handler, '__name__', str(handler))}' error='{e}'")
        
        prompt_parts.append(f"@{self.name}, time: {datetime.now().isoformat()}")
        
        # Combine all prompt parts with newlines
        return "\n\n".join(prompt_parts) if prompt_parts else ""
    
    async def _enhance_messages_with_prompts(self, messages: List[Dict[str, Any]], context: Context) -> List[Dict[str, Any]]:
        """Enhance messages by adding dynamic prompts to system message"""
        # Execute all prompt providers to get dynamic content
        dynamic_prompts = await self._execute_prompts(context)
        
        # Debug logging
        self.logger.debug(f"🔍 Enhance messages agent='{self.name}' incoming_count={len(messages)} has_instructions={bool(self.instructions)} has_dynamic_prompts={bool(dynamic_prompts)}")
        
        # If no dynamic prompts, still ensure agent instructions are in a system message
        if not dynamic_prompts:
            base_instructions = self.instructions or ""
            if not base_instructions:
                return messages

            # Find first system message
            system_index = next((i for i, m in enumerate(messages) if m.get("role") == "system"), -1)
            if system_index >= 0:
                self.logger.debug("🔧 Merging agent instructions into existing system message")
                existing = messages[system_index].get("content", "")
                merged = f"{base_instructions}\n\n{existing}".strip()
                enhanced_messages = messages.copy()
                enhanced_messages[system_index] = {**messages[system_index], "content": merged}
                return enhanced_messages
            else:
                self.logger.debug("🔧 Prepending new system message with agent instructions")
                enhanced_messages = [{
                    "role": "system",
                    "content": base_instructions
                }] + messages
                return enhanced_messages
        
        # Create enhanced messages list
        enhanced_messages = []
        system_message_found = False
        
        for message in messages:
            if message.get("role") == "system":
                # Enhance existing system message with agent instructions + prompts
                system_message_found = True
                original_content = message.get("content", "")
                base_instructions = self.instructions or ""
                parts = []
                
                # Add base instructions first (agent-specific + CORE_SYSTEM_PROMPT)
                if base_instructions:
                    parts.append(base_instructions)
                
                # Only add original_content if it's not already in base_instructions
                # (prevents duplicate CORE_SYSTEM_PROMPT)
                if original_content:
                    # Check if original_content is substantially different from base_instructions
                    # Skip if it's just the CORE_SYSTEM_PROMPT that's already in base_instructions
                    original_trimmed = original_content.strip()
                    base_trimmed = base_instructions.strip()
                    
                    # If original is not a substring of base, it's new content - add it
                    if original_trimmed and original_trimmed not in base_trimmed:
                        parts.append(original_content)
                    else:
                        self.logger.debug("🔧 Skipped duplicate original_content (already in base_instructions)")
                
                # Check if dynamic_prompts contains content already in base_instructions
                # This prevents CORE_SYSTEM_PROMPT duplication when it's included in both
                if dynamic_prompts:
                    dynamic_trimmed = dynamic_prompts.strip()
                    # Check if dynamic content is substantially overlapping with base instructions
                    # If >80% of dynamic content is already in base, skip it (likely duplicate CORE_SYSTEM_PROMPT)
                    if base_trimmed and len(dynamic_trimmed) > 100:
                        # Count how many lines from dynamic are already in base
                        dynamic_lines = set(line.strip() for line in dynamic_trimmed.split('\n') if line.strip())
                        matching_lines = sum(1 for line in dynamic_lines if line in base_trimmed)
                        overlap_ratio = matching_lines / len(dynamic_lines) if dynamic_lines else 0
                        
                        if overlap_ratio > 0.8:
                            self.logger.debug(f"🔧 Skipped duplicate dynamic_prompts ({overlap_ratio:.1%} overlap with base_instructions)")
                        else:
                            parts.append(dynamic_prompts)
                    else:
                        parts.append(dynamic_prompts)
                
                enhanced_content = "\n\n".join(parts).strip()
                enhanced_messages.append({
                    **message,
                    "content": enhanced_content
                })
                self.logger.debug("🔧 Enhanced existing system message")
                
                # Log system prompt breakdown for optimization
                breakdown = []
                if base_instructions:
                    breakdown.append(f"  - Base instructions: {len(base_instructions)} chars")
                if original_content and original_content.strip() not in base_instructions.strip():
                    breakdown.append(f"  - Original content: {len(original_content)} chars")
                if dynamic_prompts:
                    breakdown.append(f"  - Dynamic prompts: {len(dynamic_prompts)} chars")
                
                # Only log on first request (2 messages: system + first user message)
                # Skip if conversation has more history
                incoming_count = len([m for m in messages if m.get("role") in ("user", "assistant")])
                if incoming_count <= 1:  # First user message only
                    self.logger.info(f"📋 System prompt: {len(enhanced_content)} chars\n" + "\n".join(breakdown))
            else:
                enhanced_messages.append(message)
        
        # If no system message exists, create one with agent instructions + dynamic prompts
        if not system_message_found:
            base_instructions = self.instructions if self.instructions else "You are a helpful AI assistant."
            system_content = f"{base_instructions}\n\n{dynamic_prompts}".strip()
            
            # Insert system message at the beginning
            enhanced_messages.insert(0, {
                "role": "system",
                "content": system_content
            })
            self.logger.debug("🔧 Created new system message with base instructions + dynamic prompts")
            
            # Only log on first request (1 message: first user message)
            incoming_count = len([m for m in messages if m.get("role") in ("user", "assistant")])
            if incoming_count <= 1:
                self.logger.info(f"📋 System prompt: {len(system_content)} chars\n  - Base instructions: {len(base_instructions)} chars\n  - Dynamic prompts: {len(dynamic_prompts)} chars")
        
        self.logger.debug(f"📦 Enhanced messages count={len(enhanced_messages)}")
        
        return enhanced_messages
    
    # Handoff execution methods
    
    def _execute_handoff(
        self,
        handoff_config: Handoff,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union['Awaitable[Dict[str, Any]]', 'AsyncGenerator[Dict[str, Any], None]']:
        """Execute handoff - returns appropriate type based on mode
        
        Args:
            handoff_config: Handoff configuration to execute
            messages: Conversation messages
            tools: Available tools
            stream: Whether to stream response
            **kwargs: Additional arguments to pass to handoff function
        
        Returns:
            - If stream=False: Awaitable[Dict] (coroutine to await)
            - If stream=True: AsyncGenerator (async iterator - NO await!)
        
        Note: Caller must handle appropriately:
            - Non-streaming: response = await self._execute_handoff(..., stream=False)
            - Streaming: async for chunk in self._execute_handoff(..., stream=True)
        """
        function = handoff_config.metadata.get('function')
        is_generator = handoff_config.metadata.get('is_generator', False)
        
        if not function:
            raise ValueError(f"No function for handoff: {handoff_config.target}")
        
        call_kwargs = {'messages': messages, 'tools': tools, **kwargs}
        
        if stream:
            # STREAMING MODE - return AsyncGenerator
            if is_generator:
                # Generator function - return directly (NO await!)
                return function(**call_kwargs)
            else:
                # Regular async function - adapt to streaming
                return self._adapt_response_to_streaming(function, call_kwargs)
        else:
            # NON-STREAMING MODE - return Awaitable[Dict]
            if is_generator:
                # Generator function - consume all chunks to response
                return self._consume_generator_to_response(function(**call_kwargs))
            else:
                # Regular async function - return coroutine directly (NO await!)
                return function(**call_kwargs)
    
    async def _consume_generator_to_response(
        self,
        generator: 'AsyncGenerator[Dict[str, Any], None]'
    ) -> Dict[str, Any]:
        """Consume streaming generator and return final response
        
        Used when generator handoff is called in non-streaming mode.
        Reconstructs full response from chunks.
        
        Args:
            generator: Async generator yielding streaming chunks
        
        Returns:
            Full OpenAI-compatible response dict
        """
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)
        
        # Reconstruct full response from chunks
        return self._reconstruct_response_from_chunks(chunks)
    
    async def _adapt_response_to_streaming(
        self,
        function: Callable,
        call_kwargs: Dict[str, Any]
    ) -> 'AsyncGenerator[Dict[str, Any], None]':
        """Adapt non-streaming function to streaming by wrapping response as chunk
        
        Used when regular handoff is called in streaming mode.
        
        Args:
            function: The handoff function to call
            call_kwargs: Arguments to pass to function
        
        Yields:
            Single streaming chunk containing full response
        """
        # Call function
        response = await function(**call_kwargs)
        
        # Convert to streaming chunk and yield once
        chunk = self._convert_response_to_streaming_chunk(response)
        yield chunk
    
    # Tool execution methods
    def _get_tool_function_by_name(self, function_name: str) -> Optional[Callable]:
        """Get a registered tool function by name, respecting external tool overrides"""
        # If this tool was overridden by an external tool, don't return the internal function
        if function_name in self._overridden_tools:
            return None
            
        with self._registration_lock:
            for tool_config in self._registered_tools:
                if tool_config['name'] == function_name:
                    return tool_config['function']
        return None
    
    async def _execute_single_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent tool call (NOT external tools - those are executed by client)"""
        function_name = tool_call["function"]["name"]
        function_args_str = tool_call["function"]["arguments"]
        original_tool_call_id = tool_call.get("id")
        tool_call_id = original_tool_call_id or f"call_{uuid.uuid4().hex[:8]}"
        
        # Enhanced debugging: Log tool call ID handling
        if not original_tool_call_id:
            self.logger.debug(f"🔧 Generated new tool_call_id '{tool_call_id}' for {function_name} (original was missing)")
        else:
            self.logger.debug(f"🔧 Using existing tool_call_id '{tool_call_id}' for {function_name}")
        
        # Finalization runs at end-of-loop or on exception

        try:
            # Parse function arguments
            function_args = json.loads(function_args_str)
        except json.JSONDecodeError as e:
            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "content": f"Error parsing tool arguments: {str(e)}"
            }
        
        # Find the tool function (only for agent's internal @tool functions)
        tool_func = self._get_tool_function_by_name(function_name)
        if not tool_func:
            # This might be an external tool - client should handle it
            return {
                "tool_call_id": tool_call_id,
                "role": "tool", 
                "content": f"Tool '{function_name}' should be executed by client (external tool)"
            }
        
        try:
            self.logger.debug(f"🛠️ Executing tool name='{function_name}' call_id='{tool_call_id}'")
            # Execute the agent's internal tool function
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**function_args)
            else:
                result = tool_func(**function_args)

            # If tool returned (result, usage_info), log usage and unwrap result
            try:
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                    result_value, usage_payload = result
                    # Append unified usage record
                    context = get_context()
                    if context and hasattr(context, 'usage'):
                        import time as _time
                        usage_record = {
                            'type': 'tool',
                            'timestamp': _time.time(),
                            'tool': function_name,
                        }
                        try:
                            usage_record.update(usage_payload or {})
                        except Exception:
                            pass
                        context.usage.append(usage_record)
                    # Use only the actual result for tool response content
                    result = result_value
            except Exception:
                # Never fail execution due to logging issues
                pass

            # Format successful result
            result_str = str(result)
            self.logger.debug(f"🛠️ Tool success name='{function_name}' call_id='{tool_call_id}' result_preview='{result_str[:100]}...' (len={len(result_str)})")
            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "content": result_str
            }
            
        except Exception as e:
            # Format error result
            self.logger.error(f"🛠️ Tool execution error name='{function_name}' call_id='{tool_call_id}' error='{e}'")
            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "content": f"Tool execution error: {str(e)}"
            }
    
    def _has_tool_calls(self, llm_response: Dict[str, Any]) -> bool:
        """Check if LLM response contains tool calls"""
        return (llm_response.get("choices", [{}])[0]
                .get("message", {})
                .get("tool_calls") is not None)
    

    

    
    # Main execution methods
    async def run(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Run agent with messages (non-streaming) - implements agentic loop for tool calling"""
        
        # Get existing context or create new one
        context = get_context()
        if context:
            # Update existing context with new data
            context.messages = messages
            context.stream = stream
            context.agent = self
        else:
            # Create new context if none exists
            context = create_context(
                messages=messages,
                stream=stream,
                agent=self
            )
            set_context(context)
        
        # Get the default handoff (first registered handoff) to reset to at end of turn
        # Define this BEFORE the try block so it's available in the except block
        default_handoff = self._registered_handoffs[0]['config'] if self._registered_handoffs else self.active_handoff
        
        try:
            # Ensure all skills are initialized with agent reference
            await self._ensure_skills_initialized()
            
            # Execute on_connection hooks
            context = await self._execute_hooks("on_connection", context)
            
            # Merge external tools with agent tools
            all_tools = self._merge_tools(tools or [])
            
            # Ensure we have an active handoff (completion handler)
            if not self.active_handoff:
                raise ValueError(
                    f"No handoff registered for agent '{self.name}'. "
                    "Agent needs at least one skill with @handoff decorator or "
                    "manual handoff registration via register_handoff()."
                )
            
            # Enhance messages with dynamic prompts before first handoff call
            enhanced_messages = await self._enhance_messages_with_prompts(messages, context)
            
            # Maintain conversation history for agentic loop
            conversation_messages = enhanced_messages.copy()
            
            # Agentic loop - continue until no more tool calls or max iterations
            max_tool_iterations = 10  # Prevent infinite loops
            tool_iterations = 0
            response = None
            
            while tool_iterations < max_tool_iterations:
                tool_iterations += 1
                
                # Debug logging for handoff call
                handoff_name = self.active_handoff.target
                self.logger.debug(f"🚀 Calling handoff '{handoff_name}' for agent '{self.name}' (iteration {tool_iterations}) with {len(all_tools)} tools")
                
                # Enhanced debugging: Log conversation history before handoff call
                self.logger.debug(f"📝 ITERATION {tool_iterations} - Conversation history ({len(conversation_messages)} messages):")
                for i, msg in enumerate(conversation_messages):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    
                    # Truncate data URLs in content to avoid logging huge base64 strings
                    if isinstance(content, list):
                        # Multimodal content - check for image_url parts
                        content_summary = []
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'image_url':
                                url = part.get('image_url', {}).get('url', '')
                                if url.startswith('data:'):
                                    content_summary.append('[data:image]')
                                else:
                                    content_summary.append(f'[image:{url[:50]}...]')
                            elif isinstance(part, dict) and part.get('type') == 'text':
                                text = part.get('text', '')[:50]
                                content_summary.append(f'"{text}..."' if len(part.get('text', '')) > 50 else f'"{text}"')
                            else:
                                content_summary.append(str(part)[:30])
                        content_preview = ', '.join(content_summary)
                    else:
                        content_preview = str(content)[:100] + ('...' if len(str(content)) > 100 else '')
                    
                    tool_calls = msg.get('tool_calls', [])
                    tool_call_id = msg.get('tool_call_id', '')
                    
                    if role == 'system':
                        self.logger.debug(f"  [{i}] SYSTEM: {content_preview}")
                    elif role == 'user':
                        self.logger.debug(f"  [{i}] USER: {content_preview}")
                    elif role == 'assistant':
                        if tool_calls:
                            tool_names = [tc.get('function', {}).get('name', 'unknown') for tc in tool_calls]
                            self.logger.debug(f"  [{i}] ASSISTANT: {content_preview} | TOOL_CALLS: {tool_names}")
                        else:
                            self.logger.debug(f"  [{i}] ASSISTANT: {content_preview}")
                    elif role == 'tool':
                        self.logger.debug(f"  [{i}] TOOL[{tool_call_id}]: {content_preview}")
                    else:
                        self.logger.debug(f"  [{i}] {role.upper()}: {content_preview}")
                
                # Execute before_llm_call hooks to allow message preprocessing
                context.set('conversation_messages', conversation_messages)
                context.set('tools', all_tools)
                context = await self._execute_hooks("before_llm_call", context)
                conversation_messages = context.get('conversation_messages', conversation_messages)
                all_tools = context.get('tools', all_tools)
                
                # Call active handoff with current conversation history
                response = await self._execute_handoff(
                    self.active_handoff,
                    conversation_messages,
                    tools=all_tools,
                    stream=False
                )
                
                # Store LLM response in context for cost tracking
                context.set('llm_response', response)
                
                # Execute after_llm_call hooks
                context = await self._execute_hooks("after_llm_call", context)
                
                # Log LLM token usage
                self._log_llm_usage(response, streaming=False)
                
                # Enhanced debugging: Log LLM response details
                self.logger.debug(f"📤 ITERATION {tool_iterations} - LLM Response:")
                if hasattr(response, 'choices') or (isinstance(response, dict) and 'choices' in response):
                    choices = response.choices if hasattr(response, 'choices') else response['choices']
                    if choices:
                        choice = choices[0]
                        message = choice.message if hasattr(choice, 'message') else choice['message']
                        finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else choice.get('finish_reason')
                        
                        content = message.content if hasattr(message, 'content') else message.get('content', '')
                        tool_calls = message.tool_calls if hasattr(message, 'tool_calls') else message.get('tool_calls', [])
                        
                        content_preview = str(content)[:500] + ('...' if len(str(content)) > 500 else '') if content else '[None]'
                        self.logger.debug(f"  Content: {content_preview}")
                        self.logger.debug(f"  Finish reason: {finish_reason}")
                        
                        if tool_calls:
                            self.logger.debug(f"  Tool calls ({len(tool_calls)}):")
                            for j, tc in enumerate(tool_calls):
                                tc_id = tc.id if hasattr(tc, 'id') else tc.get('id', 'unknown')
                                tc_func = tc.function if hasattr(tc, 'function') else tc.get('function', {})
                                tc_name = tc_func.name if hasattr(tc_func, 'name') else tc_func.get('name', 'unknown')
                                tc_args = tc_func.arguments if hasattr(tc_func, 'arguments') else tc_func.get('arguments', '{}')
                                args_preview = tc_args[:100] + ('...' if len(tc_args) > 100 else '') if tc_args else '{}'
                                self.logger.debug(f"    [{j}] {tc_name}[{tc_id}]: {args_preview}")
                        else:
                            self.logger.debug(f"  No tool calls")
                
                # Check if response has tool calls
                if not self._has_tool_calls(response):
                    # No tool calls - LLM is done
                    self.logger.debug(f"✅ LLM finished (no tool calls) after {tool_iterations} iteration(s)")
                    break
                
                # Extract tool calls from response
                assistant_message = response["choices"][0]["message"]
                tool_calls = assistant_message.get("tool_calls", [])
                
                # More detailed logging to help diagnose tool calling loops
                tool_details = []
                for tc in tool_calls:
                    func_name = tc['function']['name']
                    func_args = tc['function'].get('arguments', '{}')
                    try:
                        args_preview = func_args[:100] if len(func_args) > 100 else func_args
                    except:
                        args_preview = str(func_args)[:100]
                    tool_details.append(f"{func_name}(args={args_preview})")
                self.logger.debug(f"🔧 Iteration {tool_iterations}: Processing {len(tool_calls)} tool call(s): {tool_details}")
                
                # Separate internal and external tools
                internal_tools = []
                external_tools = []
                
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    if self._get_tool_function_by_name(function_name):
                        internal_tools.append(tool_call)
                    else:
                        external_tools.append(tool_call)
                
                # If there are ANY external tools, we need to return to client
                if external_tools:
                    self.logger.debug(f"🔄 Found {len(external_tools)} external tool(s), breaking loop to return to client")
                    
                    # First execute any internal tools
                    if internal_tools:
                        self.logger.debug(f"⚡ Executing {len(internal_tools)} internal tool(s) first")
                        for tool_call in internal_tools:
                            # Execute hooks
                            context.set("tool_call", tool_call)
                            context = await self._execute_hooks("before_toolcall", context)
                            tool_call = context.get("tool_call", tool_call)
                            
                            # Execute tool
                            result = await self._execute_single_tool(tool_call)
                            
                            # Execute hooks
                            context.set("tool_result", result)
                            context = await self._execute_hooks("after_toolcall", context)
                    
                    # Return response with external tool calls for client
                    # Convert response to dict if needed
                    if hasattr(response, 'dict') and callable(response.dict):
                        client_response = response.dict()
                    elif hasattr(response, 'model_dump') and callable(response.model_dump):
                        client_response = response.model_dump()
                    else:
                        import copy
                        client_response = copy.deepcopy(response)
                    
                    # Keep only external tool calls in response
                    client_response["choices"][0]["message"]["tool_calls"] = external_tools
                    
                    # Mark response appropriately
                    if internal_tools:
                        client_response["_mixed_execution"] = True
                    else:
                        client_response["_external_tools_only"] = True
                    
                    # Clean up flags before returning
                    if "_mixed_execution" in client_response:
                        del client_response["_mixed_execution"]
                    if "_external_tools_only" in client_response:
                        del client_response["_external_tools_only"]
                    
                    response = client_response
                    break
                
                # All tools are internal - execute them and continue loop
                self.logger.debug(f"⚙️ Executing {len(internal_tools)} internal tool(s)")
                
                # Add assistant message with tool calls to conversation
                # IMPORTANT: Preserve the entire assistant message structure to avoid confusing the LLM
                # Only modify tool_calls if needed, but keep all original fields
                # CRITICAL: Convert message object to dict format for LLM compatibility
                original_type = type(assistant_message).__name__
                if hasattr(assistant_message, 'dict') and callable(assistant_message.dict):
                    assistant_msg_copy = assistant_message.dict()
                    self.logger.debug(f"🔄 ITERATION {tool_iterations} - Converted assistant message from {original_type} to dict via .dict()")
                elif hasattr(assistant_message, 'model_dump') and callable(assistant_message.model_dump):
                    assistant_msg_copy = assistant_message.model_dump()
                    self.logger.debug(f"🔄 ITERATION {tool_iterations} - Converted assistant message from {original_type} to dict via .model_dump()")
                else:
                    assistant_msg_copy = dict(assistant_message) if hasattr(assistant_message, 'items') else assistant_message.copy()
                    self.logger.debug(f"🔄 ITERATION {tool_iterations} - Converted assistant message from {original_type} to dict via dict() or copy()")
                
                # If we filtered tools, update the tool_calls (though for internal-only, they should be the same)
                if 'tool_calls' in assistant_msg_copy:
                    # Convert tool_calls to dict format as well
                    converted_tools = []
                    for tool_call in internal_tools:
                        if hasattr(tool_call, 'dict') and callable(tool_call.dict):
                            converted_tools.append(tool_call.dict())
                        elif hasattr(tool_call, 'model_dump') and callable(tool_call.model_dump):
                            converted_tools.append(tool_call.model_dump())
                        else:
                            converted_tools.append(dict(tool_call) if hasattr(tool_call, 'items') else tool_call)
                    assistant_msg_copy['tool_calls'] = converted_tools
                
                # Enhanced debugging: Log assistant message being added to conversation
                self.logger.debug(f"📝 ITERATION {tool_iterations} - Adding assistant message to conversation:")
                self.logger.debug(f"  Original tool_calls count: {len(tool_calls)}")
                self.logger.debug(f"  Internal tool_calls count: {len(internal_tools)}")
                self.logger.debug(f"  External tool_calls count: {len(external_tools) if external_tools else 0}")
                for i, tc in enumerate(internal_tools):
                    tc_id = tc.get('id', 'unknown')
                    tc_name = tc.get('function', {}).get('name', 'unknown')
                    self.logger.debug(f"    Internal tool[{i}]: {tc_name}[{tc_id}]")
                
                conversation_messages.append(assistant_msg_copy)
                
                # Execute each internal tool and add results
                for tool_call in internal_tools:
                    # Execute hooks
                    context.set("tool_call", tool_call)
                    context = await self._execute_hooks("before_toolcall", context)
                    tool_call = context.get("tool_call", tool_call)
                    
                    # Enhanced debugging: Log tool execution details
                    tc_name = tool_call.get('function', {}).get('name', 'unknown')
                    tc_id = tool_call.get('id', 'unknown')
                    tc_args = tool_call.get('function', {}).get('arguments', '{}')
                    self.logger.debug(f"🔧 ITERATION {tool_iterations} - Executing tool: {tc_name}[{tc_id}] with args: {tc_args}")
                    
                    # Execute tool
                    result = await self._execute_single_tool(tool_call)
                    
                    # Check if tool result is a handoff request
                    if isinstance(result.get('content', ''), str) and result.get('content', '').startswith("__HANDOFF_REQUEST__:"):
                        target_name = result.get('content', '').split(":", 1)[1]
                        self.logger.info(f"🔀 Dynamic handoff requested to: {target_name}")
                        
                        # Find the requested handoff
                        requested_handoff = self.get_handoff_by_target(target_name)
                        if not requested_handoff:
                            # Invalid target - add error to conversation and continue
                            error_msg = f"❌ Handoff target '{target_name}' not found"
                            self.logger.warning(error_msg)
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": error_msg
                            }
                            conversation_messages.append(tool_message)
                            continue
                        
                        # Switch to the requested handoff - don't execute inline
                        self.active_handoff = requested_handoff
                        self.logger.info(f"🔀 Switching active handoff to: {target_name}")
                        
                        # Add tool result to conversation
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"✓ Switching to {target_name}"
                        }
                        conversation_messages.append(tool_message)
                        
                        # Break from tool execution - the agentic loop will continue with new handoff
                        break
                    
                    # Enhanced debugging: Log tool result
                    result_content = result.get('content', '')
                    result_preview = result_content[:200] + ('...' if len(result_content) > 200 else '')
                    self.logger.debug(f"🔧 ITERATION {tool_iterations} - Tool result for {tc_name}[{tc_id}]: {result_preview}")
                    
                    # Enhanced debugging: Verify tool call ID consistency
                    result_tool_call_id = result.get('tool_call_id', 'unknown')
                    if result_tool_call_id != tc_id:
                        self.logger.warning(f"⚠️ ITERATION {tool_iterations} - Tool call ID mismatch! Expected: {tc_id}, Got: {result_tool_call_id}")
                    else:
                        self.logger.debug(f"✅ ITERATION {tool_iterations} - Tool call ID matches: {tc_id}")
                    
                    # Add tool result to conversation
                    conversation_messages.append(result)
                    
                    # Execute hooks
                    context.set("tool_result", result)
                    context = await self._execute_hooks("after_toolcall", context)
                
                # Continue loop - LLM will be called again with tool results
                self.logger.debug(f"🔄 Continuing agentic loop with tool results")
            
            if tool_iterations >= max_tool_iterations:
                self.logger.warning(f"⚠️ Reached max tool iterations ({max_tool_iterations})")
                
                # Generate a helpful explanation for the user about hitting iteration limit
                explanation_response = self._generate_iteration_limit_explanation(
                    max_iterations=max_tool_iterations,
                    conversation_messages=conversation_messages,
                    original_request=messages[0] if messages else None
                )
                
                # Set the response to the explanation
                response = explanation_response
            
            # Execute on_message hooks (payment skill will track LLM costs here)
            context = await self._execute_hooks("on_message", context)
            
            # Execute finalize_connection hooks
            context = await self._execute_hooks("finalize_connection", context)
            
            # Reset to default handoff for next turn
            if self.active_handoff != default_handoff and default_handoff is not None:
                from_target = self.active_handoff.target if self.active_handoff else 'None'
                to_target = default_handoff.target if default_handoff else 'None'
                self.logger.info(f"🔄 Resetting active handoff from '{from_target}' to default '{to_target}'")
                self.active_handoff = default_handoff
            
            return response
            
        except Exception as e:
            # Handle errors and cleanup
            self.logger.exception(f"💥 Agent execution error agent='{self.name}' error='{e}'")
            
            # Reset to default handoff even on error
            if self.active_handoff != default_handoff and default_handoff is not None:
                from_target = self.active_handoff.target if self.active_handoff else 'None'
                to_target = default_handoff.target if default_handoff else 'None'
                self.logger.info(f"🔄 Resetting active handoff from '{from_target}' to default '{to_target}' (error path)")
                self.active_handoff = default_handoff
            
            await self._execute_hooks("finalize_connection", context)
            raise
    
    def _generate_iteration_limit_explanation(
        self, 
        max_iterations: int, 
        conversation_messages: List[Dict[str, Any]], 
        original_request: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a helpful explanation when hitting the iteration limit"""
        
        # Analyze the recent tool calls to understand what went wrong
        recent_tool_calls = []
        failed_operations = []
        
        # Look at the last few messages to understand the pattern
        for msg in conversation_messages[-10:]:  # Last 10 messages
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    tool_name = tool_call.get("function", {}).get("name", "unknown")
                    recent_tool_calls.append(tool_name)
            elif msg.get("role") == "tool":
                content = msg.get("content", "")
                # Check for common failure patterns
                if any(fail_indicator in content.lower() for fail_indicator in 
                       ["failed", "error", "upload failed", "not found", "timeout"]):
                    failed_operations.append(content[:100] + "..." if len(content) > 100 else content)
        
        # Determine the original task from the first user message
        original_task = "complete your request"
        if original_request and original_request.get("role") == "user":
            user_content = original_request.get("content", "")
            if user_content:
                original_task = f'"{user_content[:100]}{"..." if len(user_content) > 100 else ""}"'
        
        # Count repeated tool calls to identify loops
        tool_call_counts = {}
        for tool in recent_tool_calls:
            tool_call_counts[tool] = tool_call_counts.get(tool, 0) + 1
        
        # Generate explanation based on analysis
        explanation_parts = [
            f"I apologize, but I encountered technical difficulties while trying to {original_task}."
        ]
        
        if failed_operations:
            explanation_parts.append(
                f"I attempted several operations but encountered repeated failures: {'; '.join(failed_operations[:3])}"
            )
        
        # Identify the most common repeated tool
        if tool_call_counts:
            most_repeated_tool = max(tool_call_counts.items(), key=lambda x: x[1])
            if most_repeated_tool[1] > 3:  # If a tool was called more than 3 times
                explanation_parts.append(
                    f"I repeatedly tried using the '{most_repeated_tool[0]}' tool ({most_repeated_tool[1]} times) but it kept failing."
                )
        
        explanation_parts.extend([
            f"After {max_iterations} attempts, I've reached my maximum number of tool iterations and need to stop here to prevent an infinite loop.",
            "",
            "This could be due to:",
            "• A temporary service issue with one of my tools",
            "• A configuration problem that's causing repeated failures", 
            "• The task requiring a different approach than I attempted",
            "",
            "Would you like to:",
            "• Try the request again (the issue might be temporary)",
            "• Rephrase your request in a different way",
            "• Break down your request into smaller, more specific tasks"
        ])
        
        explanation_text = "\n".join(explanation_parts)
        
        # Create a properly formatted OpenAI-style response
        return {
            "id": f"chatcmpl-iteration-limit-{int(datetime.now().timestamp())}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "iteration-limit-handler",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": explanation_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(explanation_text.split()),
                "total_tokens": len(explanation_text.split())
            }
        }
    
    def _log_llm_usage(self, response: Any, streaming: bool = False) -> None:
        """Helper to log LLM usage from response"""
        try:
            model_name = None
            usage_obj = None
            if hasattr(response, 'model'):
                model_name = getattr(response, 'model')
            elif isinstance(response, dict):
                model_name = response.get('model')
            if hasattr(response, 'usage'):
                usage_obj = getattr(response, 'usage')
            elif isinstance(response, dict):
                usage_obj = response.get('usage')
            if usage_obj:
                prompt_tokens = int(getattr(usage_obj, 'prompt_tokens', None) or usage_obj.get('prompt_tokens') or 0)
                completion_tokens = int(getattr(usage_obj, 'completion_tokens', None) or usage_obj.get('completion_tokens') or 0)
                total_tokens = int(getattr(usage_obj, 'total_tokens', None) or usage_obj.get('total_tokens') or (prompt_tokens + completion_tokens))
                self._append_usage_record(
                    record_type='llm',
                    payload={
                        'model': model_name or 'unknown',
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens,
                        'streaming': streaming,
                    }
                )
        except Exception:
            pass
    
    async def run_streaming(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run agent with streaming response - implements agentic loop for tool calling"""
        
        # Get existing context or create new one
        context = get_context()
        if context:
            # Update existing context with new data
            context.messages = messages
            context.stream = True
            context.agent = self
        else:
            # Create new context if none exists
            context = create_context(
                messages=messages,
                stream=True,
                agent=self
            )
            set_context(context)
        
        # Get the default handoff (first registered handoff) to reset to at end of turn
        # Define this BEFORE the try block so it's available in the except block
        default_handoff = self._registered_handoffs[0]['config'] if self._registered_handoffs else self.active_handoff
        
        try:
            # Ensure all skills are initialized with agent reference
            await self._ensure_skills_initialized()
            
            # Execute on_connection hooks
            context = await self._execute_hooks("on_connection", context)
            
            # Merge external tools
            all_tools = self._merge_tools(tools or [])
            
            # Ensure we have an active handoff (completion handler)
            if not self.active_handoff:
                raise ValueError(
                    f"No handoff registered for agent '{self.name}'. "
                    "Agent needs at least one skill with @handoff decorator or "
                    "manual handoff registration via register_handoff()."
                )
            
            # Enhance messages with dynamic prompts before first handoff call
            enhanced_messages = await self._enhance_messages_with_prompts(messages, context)
            
            # Maintain conversation history for agentic loop
            conversation_messages = enhanced_messages.copy()
            
            # Agentic loop for streaming
            max_tool_iterations = 10
            tool_iterations = 0
            pending_handoff_tag = None  # Store handoff tag to prepend to next iteration's first chunk
            in_thinking_block = False  # Track if we're currently in a <think> block
            pending_widget_html = None  # Store widget HTML from tool results to inject into next LLM response
            first_chunk_of_iteration = False  # Track if this is the first chunk after tool calls (need space)
            
            while tool_iterations < max_tool_iterations:
                tool_iterations += 1
                # Mark that we need a space at the start of this iteration if it's not the first one
                if tool_iterations > 1:
                    first_chunk_of_iteration = True
                
                # Debug logging
                handoff_name = self.active_handoff.target
                self.logger.debug(f"🚀 Streaming handoff '{handoff_name}' for agent '{self.name}' (iteration {tool_iterations}) with {len(all_tools)} tools")
                
                # Enhanced debugging: Log conversation history before streaming handoff call
                self.logger.debug(f"📝 STREAMING ITERATION {tool_iterations} - Conversation history ({len(conversation_messages)} messages):")
                for i, msg in enumerate(conversation_messages):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    
                    # Truncate data URLs in content to avoid logging huge base64 strings
                    if isinstance(content, list):
                        # Multimodal content - check for image_url parts
                        content_summary = []
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'image_url':
                                url = part.get('image_url', {}).get('url', '')
                                if url.startswith('data:'):
                                    content_summary.append('[data:image]')
                                else:
                                    content_summary.append(f'[image:{url[:50]}...]')
                            elif isinstance(part, dict) and part.get('type') == 'text':
                                text = part.get('text', '')[:50]
                                content_summary.append(f'"{text}..."' if len(part.get('text', '')) > 50 else f'"{text}"')
                            else:
                                content_summary.append(str(part)[:30])
                        content_preview = ', '.join(content_summary)
                    else:
                        content_preview = str(content)[:100] + ('...' if len(str(content)) > 100 else '')
                    
                    tool_calls = msg.get('tool_calls', [])
                    tool_call_id = msg.get('tool_call_id', '')
                    
                    if role == 'system':
                        self.logger.debug(f"  [{i}] SYSTEM: {content_preview}")
                    elif role == 'user':
                        self.logger.debug(f"  [{i}] USER: {content_preview}")
                    elif role == 'assistant':
                        if tool_calls:
                            tool_names = [tc.get('function', {}).get('name', 'unknown') for tc in tool_calls]
                            self.logger.debug(f"  [{i}] ASSISTANT: {content_preview} | TOOL_CALLS: {tool_names}")
                        else:
                            self.logger.debug(f"  [{i}] ASSISTANT: {content_preview}")
                    elif role == 'tool':
                        self.logger.debug(f"  [{i}] TOOL[{tool_call_id}]: {content_preview}")
                    else:
                        self.logger.debug(f"  [{i}] {role.upper()}: {content_preview}")
                
                # Execute before_llm_call hooks to allow message preprocessing
                context.set('conversation_messages', conversation_messages)
                context.set('tools', all_tools)
                context = await self._execute_hooks("before_llm_call", context)
                conversation_messages = context.get('conversation_messages', conversation_messages)
                all_tools = context.get('tools', all_tools)
                
                # Stream from active handoff and collect chunks
                # NOTE: NO await! _execute_handoff returns generator directly in streaming mode
                full_response_chunks = []
                held_chunks = []  # Chunks with tool fragments
                tool_calls_detected = False
                waiting_for_usage_after_tool_calls = False  # Track if we're waiting for usage chunk
                chunks_since_tool_calls = 0  # Safety counter to avoid waiting forever
                chunk_count = 0
                
                stream_gen = self._execute_handoff(
                    self.active_handoff,
                    conversation_messages,
                    tools=all_tools,
                    stream=True
                )
                
                async for chunk in stream_gen:
                    chunk_count += 1
                    
                    # Execute on_chunk hooks
                    context.set("chunk", chunk)
                    context = await self._execute_hooks("on_chunk", context)
                    modified_chunk = context.get("chunk", chunk)
                    
                    # Store chunk for potential tool processing
                    full_response_chunks.append(modified_chunk)
                    
                    # Check for tool call indicators
                    choice = modified_chunk.get("choices", [{}])[0] if isinstance(modified_chunk, dict) else {}
                    delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                    delta_tool_calls = delta.get("tool_calls")
                    finish_reason = choice.get("finish_reason")
                    
                    # Check if we have tool call fragments
                    if delta_tool_calls is not None:
                        # Before holding tool call chunks, yield any text content in this chunk
                        # This prevents cutting off mid-word/mid-sentence before tool calls
                        if delta.get('content'):
                            text_chunk = dict(modified_chunk)
                            text_chunk['choices'] = [dict(choice)]
                            text_chunk['choices'][0]['delta'] = {'content': delta['content']}
                            self.logger.debug(f"💬 STREAMING: Yielding text content before tool call: {delta['content'][:50]}...")
                            yield text_chunk
                        
                        held_chunks.append(modified_chunk)
                        self.logger.debug(f"🔧 STREAMING: Tool call fragment in chunk #{chunk_count}")
                        continue  # Don't yield tool fragments
                    
                    # Check if tool calls are complete
                    # IMPORTANT: Only break if we actually have tool call data accumulated
                    if finish_reason == "tool_calls" and held_chunks:
                        tool_calls_detected = True
                        waiting_for_usage_after_tool_calls = True
                        self.logger.debug(f"🔧 STREAMING: Tool calls complete at chunk #{chunk_count}, waiting for usage")
                        # Don't break yet - continue to get usage chunk
                        continue
                    
                    # If we're waiting for usage after tool_calls, check if this chunk has it
                    if waiting_for_usage_after_tool_calls:
                        chunks_since_tool_calls += 1
                        # Log usage if present
                        if modified_chunk.get('usage'):
                            self.logger.debug(f"💰 Got usage chunk after tool_calls at chunk #{chunk_count}, logging and breaking")
                            # Let the usage logging below handle it
                        if modified_chunk.get('usage') or chunks_since_tool_calls > 5:
                            # Break either when we get usage or after waiting too long
                            if chunks_since_tool_calls > 5 and not modified_chunk.get('usage'):
                                self.logger.debug(f"⚠️ No usage after {chunks_since_tool_calls} chunks, breaking anyway")
                            break  # Exit streaming loop to process tools
                        # Continue consuming chunks until we get usage or run out
                        continue
                    
                    # Yield content chunks
                    # - In first iteration: yield all non-tool chunks for real-time display
                    # - In subsequent iterations: yield the final response after tools
                    if not delta_tool_calls:
                        # Track thinking block state (ensure content is a string, not None)
                        content = delta.get('content') or ''
                        if '<think>' in content:
                            in_thinking_block = True
                        if '</think>' in content:
                            in_thinking_block = False
                        
                        # Handle content modifications for first chunk of iteration
                        if delta.get('content') and (pending_handoff_tag or pending_widget_html or first_chunk_of_iteration):
                            modified_chunk = dict(modified_chunk)
                            modified_chunk['choices'] = [dict(modified_chunk['choices'][0])]
                            modified_chunk['choices'][0]['delta'] = dict(modified_chunk['choices'][0].get('delta', {}))
                            
                            # Prepend widget HTML first (if present), then handoff tag
                            prepend_content = ''
                            if pending_widget_html:
                                prepend_content += pending_widget_html
                                self.logger.debug(f"🎨 Injecting widget HTML into first chunk (len={len(pending_widget_html)})")
                                pending_widget_html = None  # Clear after using
                            if pending_handoff_tag:
                                prepend_content += pending_handoff_tag
                                self.logger.debug(f"🔀 Prepended handoff tag to first chunk: {pending_handoff_tag[:50]}")
                                pending_handoff_tag = None  # Clear after using
                            
                            # Get the new content
                            new_content = modified_chunk['choices'][0]['delta'].get('content', '')
                            
                            # If this is the first chunk of a new iteration (after tool calls), ensure space
                            if first_chunk_of_iteration and new_content:
                                # Add a space at the start if the content doesn't already start with whitespace
                                if not new_content[0].isspace():
                                    new_content = ' ' + new_content
                                    self.logger.debug(f"➕ Added space to start of first chunk in iteration {tool_iterations}")
                                first_chunk_of_iteration = False  # Clear flag after first chunk
                            
                            # Ensure proper spacing between prepended content and new content
                            # Add a space if prepended content doesn't end with whitespace and new content doesn't start with whitespace
                            if prepend_content and new_content and not prepend_content[-1].isspace() and not new_content[0].isspace():
                                modified_chunk['choices'][0]['delta']['content'] = prepend_content + ' ' + new_content
                            else:
                                modified_chunk['choices'][0]['delta']['content'] = prepend_content + new_content
                        yield modified_chunk
                    
                    # Log usage if present in chunk (LiteLLM sends usage in separate chunk)
                    if modified_chunk.get('usage'):
                        self.logger.debug(f"💰 Found usage in streaming chunk #{chunk_count}, logging to context")
                        self._log_llm_usage(modified_chunk, streaming=True)
                
                # If no tool calls detected, we're done
                if not tool_calls_detected:
                    # Check if we got any content at all
                    total_content = ""
                    for chunk in full_response_chunks:
                        choice = chunk.get("choices", [{}])[0] if isinstance(chunk, dict) else {}
                        delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                        delta_content = delta.get("content", "")
                        if delta_content:
                            total_content += delta_content
                    
                    if not total_content and chunk_count > 0:
                        self.logger.warning(f"⚠️ LLM generated {chunk_count} chunks but NO content! This may be a safety filter or empty response issue.")
                        self.logger.warning(f"⚠️ First chunk details:")
                        if full_response_chunks:
                            first_chunk = full_response_chunks[0]
                            self.logger.warning(f"   - Keys: {first_chunk.keys() if isinstance(first_chunk, dict) else 'not a dict'}")
                            if isinstance(first_chunk, dict) and 'choices' in first_chunk:
                                self.logger.warning(f"   - Choices: {first_chunk['choices']}")
                        
                        # CRITICAL FIX: Yield error message to client when LLM returns no content
                        self.logger.warning(f"⚠️ Yielding error message to client due to empty LLM response")
                        error_message = "I apologize, but I encountered an issue generating a response. This might be due to content filtering or a temporary problem. Please try rephrasing your request."
                        
                        # Get metadata from first chunk if available
                        first_chunk = full_response_chunks[0] if full_response_chunks else {}
                        
                        # Yield error content chunk
                        yield {
                            "id": first_chunk.get("id", "error"),
                            "created": first_chunk.get("created", 0),
                            "model": first_chunk.get("model", "unknown"),
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": error_message},
                                "finish_reason": None
                            }]
                        }
                        
                        # Yield finish chunk
                        yield {
                            "id": first_chunk.get("id", "error"),
                            "created": first_chunk.get("created", 0),
                            "model": first_chunk.get("model", "unknown"),
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                    else:
                        self.logger.debug(f"✅ Streaming finished with content (len={len(total_content)}) after {tool_iterations} iteration(s)")
                    
                    # CRITICAL FIX: Reconstruct and store LLM response for payment tracking
                    # Even when there are no tool calls, we need to track LLM costs
                    if full_response_chunks:
                        final_response = self._reconstruct_response_from_chunks(full_response_chunks)
                        context.set('llm_response', final_response)
                        # NOTE: Usage is already logged at line 1682 when the usage chunk arrives
                    
                    self.logger.debug(f"✅ Streaming finished (no tool calls) after {tool_iterations} iteration(s)")
                    break
                
                # Reconstruct response from chunks to process tool calls
                full_response = self._reconstruct_response_from_chunks(full_response_chunks)
                
                # Store LLM response in context and execute after_llm_call hooks
                context.set('llm_response', full_response)
                # NOTE: Usage is already logged at line 1683 when the usage chunk arrives
                context = await self._execute_hooks("after_llm_call", context)
                full_response = context.get('llm_response', full_response)
                
                if not self._has_tool_calls(full_response):
                    # No tool calls after all - shouldn't happen but handle gracefully
                    self.logger.debug("🔧 STREAMING: No tool calls found in reconstructed response")
                    break
                
                # Extract tool calls
                assistant_message = full_response["choices"][0]["message"]
                tool_calls = assistant_message.get("tool_calls", [])
                
                # More detailed logging to help diagnose tool calling loops
                tool_details = []
                for tc in tool_calls:
                    func_name = tc['function']['name']
                    func_args = tc['function'].get('arguments', '{}')
                    try:
                        args_preview = func_args[:100] if len(func_args) > 100 else func_args
                    except:
                        args_preview = str(func_args)[:100]
                    tool_details.append(f"{func_name}(args={args_preview})")
                self.logger.debug(f"🔧 Iteration {tool_iterations}: Processing {len(tool_calls)} tool call(s): {tool_details}")
                
                # Separate internal and external tools
                internal_tools = []
                external_tools = []
                
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    if self._get_tool_function_by_name(function_name):
                        internal_tools.append(tool_call)
                    else:
                        external_tools.append(tool_call)
                
                # If there are ANY external tools, return to client
                if external_tools:
                    self.logger.debug(f"🔄 Found {len(external_tools)} external tool(s), returning to client")
                    
                    # First execute any internal tools
                    if internal_tools:
                        self.logger.debug(f"⚡ Executing {len(internal_tools)} internal tool(s) first")
                        for tool_call in internal_tools:
                            # Execute hooks
                            context.set("tool_call", tool_call)
                            context = await self._execute_hooks("before_toolcall", context)
                            tool_call = context.get("tool_call", tool_call)
                            
                            # Execute tool
                            result = await self._execute_single_tool(tool_call)
                            
                            # Execute hooks
                            context.set("tool_result", result)
                            context = await self._execute_hooks("after_toolcall", context)
                    
                    # Yield held chunks to let client reconstruct tool calls
                    for held_chunk in held_chunks:
                        yield held_chunk
                    
                    # Yield final chunk with external tool calls
                    if hasattr(full_response, 'dict'):
                        final_response = full_response.dict()
                    elif hasattr(full_response, 'model_dump'):
                        final_response = full_response.model_dump()
                    else:
                        import copy
                        final_response = copy.deepcopy(full_response)
                    
                    # Keep only external tool calls
                    final_response["choices"][0]["message"]["tool_calls"] = external_tools
                    
                    # Convert to streaming chunk format
                    final_chunk = self._convert_response_to_chunk(final_response)
                    yield final_chunk
                    
                    # Exit the loop; finalization runs after the loop
                    break
                
                # All tools are internal - execute and continue loop
                self.logger.debug(f"⚙️ Executing {len(internal_tools)} internal tool(s)")
                
                # Add assistant message with tool calls to conversation
                # IMPORTANT: Preserve the entire assistant message structure to avoid confusing the LLM
                # Only modify tool_calls if needed, but keep all original fields
                # CRITICAL: Convert message object to dict format for LLM compatibility
                original_type = type(assistant_message).__name__
                if hasattr(assistant_message, 'dict') and callable(assistant_message.dict):
                    assistant_msg_copy = assistant_message.dict()
                    self.logger.debug(f"🔄 ITERATION {tool_iterations} - Converted assistant message from {original_type} to dict via .dict()")
                elif hasattr(assistant_message, 'model_dump') and callable(assistant_message.model_dump):
                    assistant_msg_copy = assistant_message.model_dump()
                    self.logger.debug(f"🔄 ITERATION {tool_iterations} - Converted assistant message from {original_type} to dict via .model_dump()")
                else:
                    assistant_msg_copy = dict(assistant_message) if hasattr(assistant_message, 'items') else assistant_message.copy()
                    self.logger.debug(f"🔄 ITERATION {tool_iterations} - Converted assistant message from {original_type} to dict via dict() or copy()")
                
                # If we filtered tools, update the tool_calls (though for internal-only, they should be the same)
                if 'tool_calls' in assistant_msg_copy:
                    # Convert tool_calls to dict format as well
                    converted_tools = []
                    for tool_call in internal_tools:
                        if hasattr(tool_call, 'dict') and callable(tool_call.dict):
                            converted_tools.append(tool_call.dict())
                        elif hasattr(tool_call, 'model_dump') and callable(tool_call.model_dump):
                            converted_tools.append(tool_call.model_dump())
                        else:
                            converted_tools.append(dict(tool_call) if hasattr(tool_call, 'items') else tool_call)
                    assistant_msg_copy['tool_calls'] = converted_tools
                conversation_messages.append(assistant_msg_copy)
                
                # Execute each internal tool
                for tool_call in internal_tools:
                    # Execute hooks
                    context.set("tool_call", tool_call)
                    context = await self._execute_hooks("before_toolcall", context)
                    tool_call = context.get("tool_call", tool_call)
                    
                    # Enhanced debugging: Log streaming tool execution details
                    tc_name = tool_call.get('function', {}).get('name', 'unknown')
                    tc_id = tool_call.get('id', 'unknown')
                    tc_args = tool_call.get('function', {}).get('arguments', '{}')
                    self.logger.debug(f"🔧 STREAMING ITERATION {tool_iterations} - Executing tool: {tc_name}[{tc_id}] with args: {tc_args}")
                    
                    # Execute tool
                    result = await self._execute_single_tool(tool_call)
                    
                    # Check if tool result is a handoff request
                    if isinstance(result.get('content', ''), str) and result.get('content', '').startswith("__HANDOFF_REQUEST__:"):
                        target_name = result.get('content', '').split(":", 1)[1]
                        self.logger.info(f"🔀 Dynamic handoff requested to: {target_name}")
                        
                        # Find the requested handoff
                        requested_handoff = self.get_handoff_by_target(target_name)
                        if not requested_handoff:
                            # Invalid target - yield error and continue
                            error_msg = f"❌ Handoff target '{target_name}' not found"
                            self.logger.warning(error_msg)
                            yield {
                                "choices": [{
                                    "delta": {"content": error_msg},
                                    "finish_reason": None
                                }]
                            }
                            # Add to conversation and continue loop
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": error_msg
                            }
                            conversation_messages.append(tool_message)
                            continue
                        
                        # Switch to the requested handoff - don't execute inline
                        self.active_handoff = requested_handoff
                        self.logger.info(f"🔀 Switching active handoff to: {target_name}")
                        
                        # Build handoff tag with optional thinking closure
                        handoff_tag_parts = []
                        if in_thinking_block:
                            self.logger.debug(f"🔀 Will close open thinking block before handoff")
                            handoff_tag_parts.append("</think>\n\n")
                            in_thinking_block = False
                        handoff_tag_parts.append(f"<handoff>Handoff to {target_name}</handoff>\n\n")
                        
                        # Store handoff indicator to prepend to next iteration's first chunk
                        pending_handoff_tag = "".join(handoff_tag_parts)
                        self.logger.debug(f"🔀 Stored handoff indicator for next iteration: {pending_handoff_tag[:100]}")
                        
                        # Add tool result to conversation
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"✓ Switching to {target_name}"
                        }
                        conversation_messages.append(tool_message)
                        
                        # Break from tool execution - the agentic loop will continue with new handoff
                        break
                    
                    # Enhanced debugging: Log streaming tool result
                    result_content = result.get('content', '')
                    result_preview = result_content[:200] + ('...' if len(result_content) > 200 else '')
                    self.logger.debug(f"🔧 STREAMING ITERATION {tool_iterations} - Tool result for {tc_name}[{tc_id}]: {result_preview}")
                    
                    # Enhanced debugging: Verify streaming tool call ID consistency
                    result_tool_call_id = result.get('tool_call_id', 'unknown')
                    if result_tool_call_id != tc_id:
                        self.logger.warning(f"⚠️ STREAMING ITERATION {tool_iterations} - Tool call ID mismatch! Expected: {tc_id}, Got: {result_tool_call_id}")
                    else:
                        self.logger.debug(f"✅ STREAMING ITERATION {tool_iterations} - Tool call ID matches: {tc_id}")
                    
                    # Check if result contains widget HTML - store it to prepend to next LLM response
                    if result_content and '<widget' in result_content:
                        self.logger.debug(f"🎨 Widget detected in tool result (len={len(result_content)}), will inject into next LLM response")
                        # Store widget HTML to prepend to first chunk of next iteration
                        pending_widget_html = f"\n\n{result_content}\n\n"
                    
                    # Add result to conversation
                    conversation_messages.append(result)
                    
                    # Execute hooks
                    context.set("tool_result", result)
                    context = await self._execute_hooks("after_toolcall", context)
                
                # Continue loop - will stream next LLM response
                self.logger.debug(f"🔄 Continuing agentic loop with tool results")
            
            if tool_iterations >= max_tool_iterations:
                self.logger.warning(f"⚠️ Reached max tool iterations ({max_tool_iterations})")
            
            # Finalize after breaking out (normal end)
            self.logger.debug("🔚 Executing finalization hooks")
            try:
                context = await self._execute_hooks("on_message", context)
                context = await self._execute_hooks("finalize_connection", context)
                self.logger.debug("✅ Finalization hooks completed")
            except Exception as hook_error:
                self.logger.error(f"Error executing finalization hooks: {hook_error}")
            
            # Reset to default handoff for next turn (always, even if hooks failed)
            if self.active_handoff != default_handoff and default_handoff is not None:
                from_target = self.active_handoff.target if self.active_handoff else 'None'
                to_target = default_handoff.target if default_handoff else 'None'
                self.logger.info(f"🔄 Resetting active handoff from '{from_target}' to default '{to_target}'")
                self.active_handoff = default_handoff
            
        except Exception as e:
            self.logger.exception(f"💥 Streaming execution error agent='{self.name}' error='{e}'")
            # Finalize even on error
            self.logger.debug("🔚 Executing finalization hooks (error path)")
            try:
                context = await self._execute_hooks("on_message", context)
                context = await self._execute_hooks("finalize_connection", context)
                self.logger.debug("✅ Finalization hooks completed")
            except Exception as hook_error:
                self.logger.error(f"Error executing finalization hooks: {hook_error}")
            
            # Reset to default handoff for next turn (always, even on error)
            if self.active_handoff != default_handoff and default_handoff is not None:
                from_target = self.active_handoff.target if self.active_handoff else 'None'
                to_target = default_handoff.target if default_handoff else 'None'
                self.logger.info(f"🔄 Resetting active handoff from '{from_target}' to default '{to_target}' (error path)")
                self.active_handoff = default_handoff
            
            raise
    
    def _reconstruct_response_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reconstruct a full LLM response from streaming chunks for tool processing"""
        if not chunks:
            return {}
        
        logger = self.logger
        
        # Check if any chunk has complete tool calls in message format
        for chunk in chunks:
            message_tool_calls = chunk.get("choices", [{}])[0].get("message", {}).get("tool_calls")
            if message_tool_calls is not None:
                logger.debug(f"🔧 RECONSTRUCTION: Found complete tool calls")
                return chunk
        
        # Reconstruct from streaming delta chunks
        logger.debug(f"🔧 RECONSTRUCTION: Reconstructing from {len(chunks)} delta chunks")
        
        # Accumulate streaming data (both content and tool calls)
        accumulated_tool_calls = {}
        accumulated_content = []
        role = "assistant"
        final_chunk = chunks[-1] if chunks else {}
        finish_reason = None
        
        for i, chunk in enumerate(chunks):
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
            delta_tool_calls = delta.get("tool_calls") if isinstance(delta, dict) else None
            
            # Accumulate content from deltas
            delta_content = delta.get("content") if isinstance(delta, dict) else None
            if delta_content:
                accumulated_content.append(delta_content)
            
            # Capture role if present
            delta_role = delta.get("role") if isinstance(delta, dict) else None
            if delta_role:
                role = delta_role
            
            # Capture finish_reason
            choice_finish = choice.get("finish_reason") if isinstance(choice, dict) else None
            if choice_finish:
                finish_reason = choice_finish
            
            # Accumulate tool calls
            if delta_tool_calls:
                for tool_call in delta_tool_calls:
                    tool_index = tool_call.get("index", 0)
                    
                    # Initialize tool call if not exists
                    if tool_index not in accumulated_tool_calls:
                        accumulated_tool_calls[tool_index] = {
                            "id": None,
                            "type": "function",
                            "function": {
                                "name": None,
                                "arguments": ""
                            }
                        }
                    
                    # Accumulate data
                    if tool_call.get("id"):
                        accumulated_tool_calls[tool_index]["id"] = tool_call["id"]
                    
                    func = tool_call.get("function", {})
                    if func.get("name"):
                        accumulated_tool_calls[tool_index]["function"]["name"] = func["name"]
                    if func.get("arguments"):
                        accumulated_tool_calls[tool_index]["function"]["arguments"] += func["arguments"]
        
        # If we have accumulated tool calls, create a response
        if accumulated_tool_calls:
            tool_calls_list = list(accumulated_tool_calls.values())
            
            # Try to infer missing tool names based on arguments
            for tool_call in tool_calls_list:
                if not tool_call["function"]["name"]:
                    # Try to guess the tool name from the arguments
                    args = tool_call["function"]["arguments"]
                    # Look for scope_filter pattern -> likely list_files
                    if "scope_filter" in args or "_filter" in args:
                        tool_call["function"]["name"] = "list_files"
                        logger.debug(f"🔧 RECONSTRUCTION: Inferred tool name: list_files")
            
            # Create reconstructed response with proper streaming format
            reconstructed = {
                "id": final_chunk.get("id", "chatcmpl-reconstructed"),
                "created": final_chunk.get("created", 0),
                "model": final_chunk.get("model", "azure/gpt-4o-mini"),
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls_list
                    },
                    "delta": {},
                    "logprobs": None
                }],
                "system_fingerprint": final_chunk.get("system_fingerprint"),
                "provider_specific_fields": None,
                "stream_options": None
            }
            logger.debug(f"🔧 RECONSTRUCTION: Reconstructed {len(tool_calls_list)} tool calls")
            return reconstructed
        
        # No tool calls found - check if we have content to return
        content_text = "".join(accumulated_content) if accumulated_content else None
        
        if content_text or finish_reason:
            # Create a proper response with message format
            logger.debug(f"🔧 RECONSTRUCTION: No tool calls, reconstructing content response (content_len={len(content_text) if content_text else 0})")
            reconstructed = {
                "id": final_chunk.get("id", "chatcmpl-reconstructed"),
                "created": final_chunk.get("created", 0),
                "model": final_chunk.get("model", "unknown"),
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": finish_reason or "stop",
                    "message": {
                        "role": role,
                        "content": content_text
                    }
                }],
                "usage": final_chunk.get("usage", {})
            }
            return reconstructed
        
        # No content and no tool calls - return last chunk as-is (shouldn't happen often)
        logger.warning(f"🔧 RECONSTRUCTION: No tool calls and no content found, returning last chunk as-is")
        return final_chunk
    
    def _convert_response_to_chunk(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a processed response back to streaming chunk format"""
        # For streaming, we just return the response as-is
        # The frontend will handle it as a final chunk
        return response
    
    def _convert_response_to_streaming_chunk(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a complete LLM response to streaming chunk format"""
        if not response or not response.get("choices"):
            return response
        
        choice = response["choices"][0]
        message = choice.get("message", {})
        
        # Convert to streaming chunk format
        streaming_chunk = {
            "id": response.get("id", "chatcmpl-converted"),
            "created": response.get("created", 0),
            "model": response.get("model", "azure/gpt-4o-mini"),
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "finish_reason": choice.get("finish_reason", "stop"),
                "delta": {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content"),
                    "tool_calls": None
                },
                "logprobs": None
            }],
            "system_fingerprint": response.get("system_fingerprint"),
            "provider_specific_fields": None,
            "stream_options": None
        }
        
        return streaming_chunk

    def _append_usage_record(self, record_type: str, payload: Dict[str, Any]) -> None:
        """Append a normalized usage record to context.usage"""
        try:
            context = get_context()
            if not context or not hasattr(context, 'usage'):
                return
            import time as _time
            base_record = {
                'timestamp': _time.time(),
            }
            if record_type == 'llm':
                record = {**base_record, 'type': 'llm', **payload}
            elif record_type == 'tool':
                record = {**base_record, 'type': 'tool', **payload}
            else:
                record = {**base_record, 'type': record_type, **payload}
            context.usage.append(record)
        except Exception:
            return
    
    def _is_browser_request(self, context=None) -> bool:
        """Check if the request came from a browser based on User-Agent header
        
        Args:
            context: Optional context object (uses get_context() if not provided)
        
        Returns:
            True if User-Agent contains browser markers (Mozilla, Chrome, Safari, Firefox)
        """
        if context is None:
            context = get_context()
        
        if not context or not hasattr(context, 'request') or not context.request:
            self.logger.debug("🌐 No context or request available for browser detection")
            return False
        
        user_agent = context.request.headers.get('user-agent', '').lower() if hasattr(context.request, 'headers') else ''
        
        # Check for common browser User-Agent markers
        browser_markers = ['mozilla', 'chrome', 'safari', 'firefox', 'edge']
        is_browser = any(marker in user_agent for marker in browser_markers)
        
        self.logger.debug(f"🌐 User-Agent: {user_agent[:100] if user_agent else '(empty)'} -> is_browser: {is_browser}")
        
        return is_browser
    
    def _merge_tools(self, external_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge external tools with agent tools - external tools have priority"""
        # Clear previous overrides (fresh for each request)
        self._overridden_tools.clear()
        
        # Get agent tools based on current context user scope
        context = get_context()
        auth_scope = context.auth_scope if context else "all"
        
        # Get all agent tools (now includes widgets since they're also registered as tools)
        agent_tools = self.get_tools_for_scope(auth_scope)
        
        # Get widget names for filtering
        widget_names = {w['name'] for w in self._registered_widgets}
        
        # Filter widgets out of regular tools list (we'll add them conditionally below)
        agent_tools_no_widgets = [tool for tool in agent_tools if tool.get('name') not in widget_names]
        agent_tool_defs = [tool['definition'] for tool in agent_tools_no_widgets if tool.get('definition')]
        
        # Add widgets only for browser requests
        is_browser = self._is_browser_request(context)
        self.logger.debug(f"🌐 Browser request check: {is_browser}")
        if is_browser:
            agent_widgets = self.get_all_widgets()
            self.logger.debug(f"🎨 Found {len(agent_widgets)} registered widgets")
            scope_hierarchy = {"admin": 3, "owner": 2, "all": 1}
            user_level = scope_hierarchy.get(auth_scope, 1)
            
            # Convert widget configs to tool-like definitions for LLM context
            widgets_added = 0
            for widget in agent_widgets:
                # Filter by scope (similar to tools)
                widget_scope = widget.get('scope', 'all')
                scope_matched = False
                
                if isinstance(widget_scope, list):
                    # If scope is a list, check if user scope is in it
                    if auth_scope in widget_scope or 'all' in widget_scope:
                        scope_matched = True
                else:
                    # Single scope - check hierarchy
                    required_level = scope_hierarchy.get(widget_scope, 1)
                    if user_level >= required_level:
                        scope_matched = True
                
                if scope_matched:
                    widget_def = widget.get('definition')
                    if widget_def:
                        agent_tool_defs.append(widget_def)
                        widgets_added += 1
            
            if widgets_added > 0:
                self.logger.debug(f"🎨 Added {widgets_added} widgets for browser request")
        
        # Debug logging
        logger = self.logger
        
        external_tool_names = [tool.get('function', {}).get('name', 'unknown') for tool in external_tools] if external_tools else []
        agent_tool_names = [tool.get('function', {}).get('name', 'unknown') for tool in agent_tool_defs] if agent_tool_defs else []
        
        logger.debug(f"🔧 Tool merge for scope '{auth_scope}': External tools: {external_tool_names}, Agent tools: {agent_tool_names}")
        
        # Create a dictionary to track tools by name, with external tools taking priority
        tools_by_name = {}
        
        # First add agent tools
        for tool_def in agent_tool_defs:
            tool_name = tool_def.get('function', {}).get('name', 'unknown')
            tools_by_name[tool_name] = tool_def
            logger.debug(f"  📄 Added agent tool: {tool_name}")
        
        # Then add external tools (these override agent tools with same name)
        for tool_def in external_tools:
            tool_name = tool_def.get('function', {}).get('name', 'unknown')
            if tool_name in tools_by_name:
                logger.debug(f"  🔄 External tool '{tool_name}' overrides agent tool")
                # Track this tool as overridden so execution logic respects the override
                self._overridden_tools.add(tool_name)
            else:
                logger.debug(f"  📄 Added external tool: {tool_name}")
            tools_by_name[tool_name] = tool_def
        
        # Convert back to list with external tools having priority (appear first)
        all_tools = list(tools_by_name.values())
        
        final_tool_names = [tool.get('function', {}).get('name', 'unknown') for tool in all_tools]
        logger.debug(f"🔧 Final merged tools ({len(all_tools)}): {final_tool_names} | Overridden: {list(self._overridden_tools)}")
        
        return all_tools
    
    # ===== DIRECT REGISTRATION METHODS =====
    # FastAPI-style decorator methods for direct registration on agent instances
    
    def tool(self, func: Optional[Callable] = None, *, name: Optional[str] = None, 
             description: Optional[str] = None, scope: Union[str, List[str]] = "all"):
        """Register a tool function directly on the agent instance
        
        Usage:
            @agent.tool
            def my_tool(param: str) -> str:
                return f"Result: {param}"
            
            @agent.tool(name="custom", scope="owner")
            def another_tool(value: int) -> int:
                return value * 2
        """
        def decorator(f: Callable) -> Callable:
            from ..tools.decorators import tool as tool_decorator
            decorated_func = tool_decorator(func=f, name=name, description=description, scope=scope)
            # Pass the scope from the decorator to register_tool
            effective_scope = getattr(decorated_func, '_tool_scope', scope)
            self.register_tool(decorated_func, source="agent", scope=effective_scope)
            return decorated_func
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def http(self, subpath: str, method: str = "get", scope: Union[str, List[str]] = "all"):
        """Register an HTTP handler directly on the agent instance
        
        Usage:
            @agent.http("/weather")
            def get_weather(location: str) -> dict:
                return {"location": location, "temp": 25}
            
            @agent.http("/data", method="post", scope="owner")
            async def post_data(data: dict) -> dict:
                return {"received": data}
        """
        def decorator(func: Callable) -> Callable:
            from ..tools.decorators import http as http_decorator
            decorated_func = http_decorator(subpath=subpath, method=method, scope=scope)(func)
            self.register_http_handler(decorated_func, source="agent")
            return decorated_func
        
        return decorator
    
    def hook(self, event: str, priority: int = 50, scope: Union[str, List[str]] = "all"):
        """Register a hook directly on the agent instance
        
        Usage:
            @agent.hook("on_request", priority=10)
            async def my_hook(context):
                # Process context
                return context
        """
        def decorator(func: Callable) -> Callable:
            from ..tools.decorators import hook as hook_decorator
            decorated_func = hook_decorator(event=event, priority=priority, scope=scope)(func)
            self.register_hook(event, decorated_func, priority, source="agent", scope=scope)
            return decorated_func
        
        return decorator
    
    def handoff(self, name: Optional[str] = None, prompt: Optional[str] = None, 
                scope: Union[str, List[str]] = "all", priority: int = 50):
        """Register a handoff directly on the agent instance
        
        Usage:
            @agent.handoff(name="specialist", prompt="Hand off to specialist")
            async def escalate_to_supervisor(messages, tools=None, **kwargs):
                return {"choices": [{"message": {"role": "assistant", "content": "Escalated"}}]}
        """
        def decorator(func: Callable) -> Callable:
            from ..tools.decorators import handoff as handoff_decorator
            decorated_func = handoff_decorator(name=name, prompt=prompt, scope=scope, priority=priority)(func)
            handoff_config = Handoff(
                target=getattr(decorated_func, '_handoff_name', decorated_func.__name__),
                description=getattr(decorated_func, '_handoff_prompt', ''),
                scope=getattr(decorated_func, '_handoff_scope', scope)
            )
            handoff_config.metadata = {
                'function': decorated_func,
                'priority': getattr(decorated_func, '_handoff_priority', priority),
                'is_generator': getattr(decorated_func, '_handoff_is_generator', False)
            }
            self.register_handoff(handoff_config, source="agent")
            return decorated_func
        
        return decorator 