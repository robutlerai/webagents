"""
Tool, Hook, Handoff, and HTTP Decorators - WebAgents V2.0

Decorators for automatic registration of tools, hooks, handoffs, and HTTP handlers with BaseAgent.
Supports context injection and scope-based access control.
"""

import inspect
import functools
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass


def tool(func: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None, scope: Union[str, List[str]] = "all"):
    """Decorator to mark functions as tools for automatic registration
    
    Can be used as:
        @tool
        def my_tool(param: str) -> str: ...
    
    Or:
        @tool(name="custom", description="Custom tool", scope="owner")  
        def my_tool(param: str) -> str: ...
    
    Args:
        name: Optional override for tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        scope: Access scope - "all", "owner", "admin", or list of scopes
    
    The decorated function can optionally receive Context via dependency injection:
    
    @tool(scope="owner")
    def my_tool(self, param: str, context: Context = None) -> str:
        # Context automatically injected if parameter exists
        if context:
            user_id = context.peer_user_id
            # ... use context
        return result
    """
    def decorator(f: Callable) -> Callable:
        # Generate OpenAI-compatible tool schema
        sig = inspect.signature(f)
        parameters = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # Skip 'self' and 'context' parameters from schema
            if param_name in ('self', 'context'):
                continue
                
            param_type = "string"  # Default type
            param_desc = f"Parameter {param_name}"
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
            
            parameters[param_name] = {
                "type": param_type,
                "description": param_desc
            }
            
            # Mark as required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        # Create OpenAI tool schema
        tool_schema = {
            "type": "function",
            "function": {
                "name": name or f.__name__,
                "description": description or f.__doc__ or f"Tool: {f.__name__}",
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required
                }
            }
        }
        
        # Mark function with metadata for BaseAgent discovery
        f._webagents_is_tool = True
        f._webagents_tool_definition = tool_schema
        f._tool_scope = scope
        f._tool_scope_was_set = func is None  # If func is None, decorator was called with params
        f._tool_name = name or f.__name__
        f._tool_description = description or f.__doc__ or f"Tool: {f.__name__}"
        
        # Check if function expects context injection
        has_context_param = 'context' in sig.parameters
        
        if has_context_param:
            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                # Inject context if requested and not provided
                if 'context' not in kwargs:
                    from ...server.context.context_vars import get_context
                    context = get_context()
                    kwargs['context'] = context
                # Call original function and return directly; BaseAgent will log usage
                return await f(*args, **kwargs) if inspect.iscoroutinefunction(f) else f(*args, **kwargs)
            
            @functools.wraps(f) 
            def sync_wrapper(*args, **kwargs):
                # Inject context if requested and not provided
                if 'context' not in kwargs:
                    from ...server.context.context_vars import get_context
                    context = get_context()
                    kwargs['context'] = context
                # Call original function and return directly; BaseAgent will log usage
                return f(*args, **kwargs)
            
            # Return appropriate wrapper based on function type
            if inspect.iscoroutinefunction(f):
                wrapper = async_wrapper
            else:
                wrapper = sync_wrapper
        else:
            # No context injection needed; return function directly
            wrapper = f
        
        # Copy metadata to wrapper
        wrapper._webagents_is_tool = True
        wrapper._webagents_tool_definition = tool_schema
        wrapper._tool_scope = scope
        wrapper._tool_scope_was_set = func is None  # If func is None, decorator was called with params
        wrapper._tool_name = name or f.__name__
        wrapper._tool_description = description or f.__doc__ or f"Tool: {f.__name__}"
        
        return wrapper
    
    if func is None:
        # Called with arguments: @tool(name="...", ...)
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)


def hook(event: str, priority: int = 50, scope: Union[str, List[str]] = "all"):
    """Decorator to mark functions as lifecycle hooks for automatic registration
    
    Args:
        event: Lifecycle event name (on_connection, on_chunk, on_message, etc.)
        priority: Execution priority (lower numbers execute first)
        scope: Access scope - "all", "owner", "admin", or list of scopes
    
    Hook functions should accept and return Context:
    
    @hook("on_message", priority=10, scope="owner")
    async def my_hook(self, context: Context) -> Context:
        # Process context
        return context
    """
    def decorator(func: Callable) -> Callable:
        # Mark function with metadata for BaseAgent discovery
        func._webagents_is_hook = True
        func._hook_event_type = event
        func._hook_priority = priority
        func._hook_scope = scope
        
        return func
    
    return decorator


def prompt(priority: int = 50, scope: Union[str, List[str]] = "all"):
    """Decorator to mark functions as system prompt providers for automatic registration
    
    Args:
        priority: Execution priority (lower numbers execute first)
        scope: Access scope - "all", "owner", "admin", or list of scopes
    
    Prompt functions should accept context and return a string to be added to the system prompt:
    
    @prompt(priority=10, scope="owner")
    def my_prompt(self, context: Context) -> str:
        # Generate dynamic prompt content
        return f"Current user: {context.user_id}"
    
    @prompt(priority=20)
    async def async_prompt(self, context: Context) -> str:
        # Async prompt generation
        data = await some_async_call()
        return f"Dynamic data: {data}"
    """
    def decorator(func: Callable) -> Callable:
        # Mark function with metadata for BaseAgent discovery
        func._webagents_is_prompt = True
        func._prompt_priority = priority
        func._prompt_scope = scope
        
        # Check if function expects context injection
        sig = inspect.signature(func)
        has_context_param = 'context' in sig.parameters
        
        if has_context_param:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Inject context if requested and not provided
                if 'context' not in kwargs:
                    from ...server.context.context_vars import get_context
                    context = get_context()
                    kwargs['context'] = context
                
                # Call original function
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            @functools.wraps(func) 
            def sync_wrapper(*args, **kwargs):
                # Inject context if requested and not provided
                if 'context' not in kwargs:
                    from ...server.context.context_vars import get_context
                    context = get_context()
                    kwargs['context'] = context
                
                return func(*args, **kwargs)
            
            # Return appropriate wrapper based on function type
            if inspect.iscoroutinefunction(func):
                wrapper = async_wrapper
            else:
                wrapper = sync_wrapper
        else:
            # No context injection needed
            wrapper = func
        
        # Copy metadata to wrapper
        wrapper._webagents_is_prompt = True
        wrapper._prompt_priority = priority
        wrapper._prompt_scope = scope
        
        return wrapper
    
    return decorator


def handoff(name: Optional[str] = None, handoff_type: str = "agent", description: Optional[str] = None, 
           scope: Union[str, List[str]] = "all"):
    """Decorator to mark functions as handoffs for automatic registration
    
    Args:
        name: Optional override for handoff name (defaults to function name)
        handoff_type: Type of handoff - "agent", "llm", "pipeline", etc.
        description: Handoff description (defaults to function docstring)  
        scope: Access scope - "all", "owner", "admin", or list of scopes
    
    Handoff functions should return HandoffResult:
    
    @handoff(handoff_type="agent", scope=["admin"])
    async def escalate_to_admin(self, issue: str, context: Context = None) -> HandoffResult:
        # Process handoff
        return HandoffResult(result="escalated", handoff_type="agent")
    """
    def decorator(func: Callable) -> Callable:
        # Mark function with metadata for BaseAgent discovery
        func._webagents_is_handoff = True
        func._handoff_type = handoff_type
        func._handoff_scope = scope
        func._handoff_name = name or func.__name__
        func._handoff_description = description or func.__doc__ or f"Handoff: {func.__name__}"
        
        # Check if function expects context injection
        sig = inspect.signature(func)
        has_context_param = 'context' in sig.parameters
        
        if has_context_param:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Inject context if requested and not provided
                if 'context' not in kwargs:
                    from ...server.context.context_vars import get_context
                    context = get_context()
                    kwargs['context'] = context
                
                # Call original function
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Inject context if requested and not provided  
                if 'context' not in kwargs:
                    from ...server.context.context_vars import get_context
                    context = get_context()
                    kwargs['context'] = context
                
                return func(*args, **kwargs)
            
            # Return appropriate wrapper
            if inspect.iscoroutinefunction(func):
                wrapper = async_wrapper
            else:
                wrapper = sync_wrapper
        else:
            wrapper = func
        
        # Copy metadata to wrapper
        wrapper._webagents_is_handoff = True
        wrapper._handoff_type = handoff_type
        wrapper._handoff_scope = scope
        wrapper._handoff_name = name or func.__name__
        wrapper._handoff_description = description or func.__doc__ or f"Handoff: {func.__name__}"
        
        return wrapper
    
    return decorator 


def http(subpath: str, method: str = "get", scope: Union[str, List[str]] = "all"):
    """Decorator to mark functions as HTTP handlers for automatic registration
    
    Args:
        subpath: URL path after agent name (e.g., "/myapi" -> /{agentname}/myapi)
                 Supports dynamic parameters: "/users/{user_id}/posts/{post_id}"
        method: HTTP method - "get", "post", "put", "delete", etc. (default: "get")
        scope: Access scope - "all", "owner", "admin", or list of scopes
    
    HTTP handler functions receive FastAPI request arguments directly:
    
    @http("/weather", method="get", scope="owner")
    def get_weather(location: str, units: str = "celsius") -> dict:
        # Function receives query parameters as arguments
        return {"location": location, "temperature": 25, "units": units}
    
    @http("/data", method="post")
    async def post_data(request: Request, data: dict) -> dict:
        # Function can receive Request object and body data
        return {"received": data, "status": "success"}
    
    @http("/users/{user_id}", method="get")
    def get_user(user_id: str) -> dict:
        # Function receives path parameters as arguments
        return {"user_id": user_id, "name": f"User {user_id}"}
    
    @http("/users/{user_id}/posts/{post_id}", method="get")
    def get_user_post(user_id: str, post_id: str, include_comments: bool = False) -> dict:
        # Function receives both path parameters and query parameters
        return {
            "user_id": user_id,
            "post_id": post_id, 
            "include_comments": include_comments
        }
    
    Dynamic path parameters are automatically extracted by FastAPI and passed
    to the handler function. Query parameters and JSON body data are also
    automatically passed as function arguments.
    """
    def decorator(func: Callable) -> Callable:
        # Validate HTTP method
        valid_methods = ["get", "post", "put", "delete", "patch", "head", "options"]
        if method.lower() not in valid_methods:
            raise ValueError(f"Invalid HTTP method '{method}'. Must be one of: {valid_methods}")
        
        # Ensure subpath starts with /
        normalized_subpath = subpath if subpath.startswith('/') else f'/{subpath}'
        
        # Mark function with metadata for BaseAgent discovery
        func._webagents_is_http = True
        func._http_subpath = normalized_subpath
        func._http_method = method.lower()
        func._http_scope = scope
        func._http_description = func.__doc__ or f"HTTP {method.upper()} handler for {normalized_subpath}"
        
        # Check if function expects context injection
        sig = inspect.signature(func)
        has_context_param = 'context' in sig.parameters
        
        if has_context_param:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Inject context if requested and not provided
                if 'context' not in kwargs:
                    from ...server.context.context_vars import get_context
                    context = get_context()
                    kwargs['context'] = context
                
                # Call original function
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Inject context if requested and not provided
                if 'context' not in kwargs:
                    from ...server.context.context_vars import get_context
                    context = get_context()
                    kwargs['context'] = context
                
                return func(*args, **kwargs)
            
            # Return appropriate wrapper based on function type
            if inspect.iscoroutinefunction(func):
                wrapper = async_wrapper
            else:
                wrapper = sync_wrapper
        else:
            # No context injection needed
            wrapper = func
        
        # Copy metadata to wrapper
        wrapper._webagents_is_http = True
        wrapper._http_subpath = normalized_subpath
        wrapper._http_method = method.lower()
        wrapper._http_scope = scope
        wrapper._http_description = func.__doc__ or f"HTTP {method.upper()} handler for {normalized_subpath}"
        
        return wrapper
    
    return decorator 