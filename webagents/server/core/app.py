"""
FastAPI Server - WebAgents V2.0

Production FastAPI server with OpenAI compatibility, dynamic agent routing,
and comprehensive monitoring.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union, Awaitable
import inspect

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from ..monitoring import CONTENT_TYPE_LATEST

from .models import (
    ChatCompletionRequest, ChatCompletionResponse, AgentInfoResponse, 
    HealthResponse, AgentListResponse, ServerStatsResponse
)
from .middleware import RequestLoggingMiddleware, RateLimitMiddleware, RateLimitRule
from .monitoring import initialize_monitoring
from ..context.context_vars import Context, set_context, create_context, get_context
from ...agents.core.base_agent import BaseAgent
from ...utils.logging import get_logger


class WebAgentsServer:
    """
    FastAPI server for AI agents with OpenAI compatibility and production monitoring
    
    Features:
    - OpenAI-compatible /chat/completions endpoint
    - Streaming and non-streaming support
    - Dynamic agent routing via provided resolver function
    - Context management middleware
    - Health and discovery endpoints
    - Prometheus metrics collection
    - Structured logging and request tracing
    """
    
    def __init__(
        self, 
        agents: List[BaseAgent] = None,
        dynamic_agents: Optional[Union[Callable[[str], BaseAgent], Callable[[str], Awaitable[Optional[BaseAgent]]]]] = None,
        enable_cors: bool = True,
        title: str = "WebAgents V2 Server",
        description: str = "AI Agent Server with OpenAI Compatibility",
        version: str = "2.0.0",
        url_prefix: str = "",
        # Middleware configuration
        request_timeout: float = 300.0,
        enable_rate_limiting: bool = True,
        default_rate_limit: RateLimitRule = None,
        user_rate_limits: Dict[str, RateLimitRule] = None,
        enable_request_logging: bool = True,
        # Monitoring configuration
        enable_monitoring: bool = True,
        enable_prometheus: bool = True,
        enable_structured_logging: bool = True,
        metrics_port: int = 9090
    ):
        """
        Initialize WebAgents server
        
        Args:
            agents: List of static Agent instances (optional)
            dynamic_agents: Optional function (sync or async) that takes agent_name: str and returns 
                           BaseAgent or Optional[BaseAgent]. Server does not manage how this works internally.
            enable_cors: Whether to enable CORS middleware
            title: FastAPI app title
            description: FastAPI app description
            version: Server version
            url_prefix: URL prefix for all routes (e.g., "/agents" makes all routes "/agents/...")
            request_timeout: Request timeout in seconds (default: 300.0)
            enable_rate_limiting: Whether to enable rate limiting (default: True)
            default_rate_limit: Default rate limit rule for all clients
            user_rate_limits: Per-user rate limit overrides
            enable_request_logging: Whether to enable request logging (default: True)
            enable_monitoring: Whether to enable monitoring system (default: True)
            enable_prometheus: Whether to enable Prometheus metrics (default: True)
            enable_structured_logging: Whether to enable structured logging (default: True)
            metrics_port: Port for Prometheus metrics endpoint (default: 9090)
        """
        self.app = FastAPI(
            title=title,
            description=description,
            version=version
        )
        
        self.version = version
        self.url_prefix = url_prefix.rstrip("/")  # Remove trailing slash if present
        
        # Create API router with prefix
        self.router = APIRouter(prefix=self.url_prefix)
        
        # Store agents by name for quick lookup
        self.static_agents = {agent.name: agent for agent in (agents or [])}
        
        # Store dynamic agent resolver (server doesn't manage how it works)
        self.dynamic_agents = dynamic_agents
        
        # Store middleware configuration
        self.request_timeout = request_timeout
        self.enable_rate_limiting = enable_rate_limiting
        self.default_rate_limit = default_rate_limit or RateLimitRule()
        self.user_rate_limits = user_rate_limits or {}
        self.enable_request_logging = enable_request_logging
        
        # Initialize monitoring system
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.monitoring = initialize_monitoring(
                enable_prometheus=enable_prometheus,
                enable_structured_logging=enable_structured_logging,
                metrics_port=metrics_port
            )
            
            # Set server info in metrics
            self.monitoring.prometheus.set_server_info(
                version=version,
                agents_count=len(self.static_agents),
                dynamic_agents_enabled=str(self.dynamic_agents is not None),
                prometheus_enabled=str(enable_prometheus),
                structured_logging_enabled=str(enable_structured_logging)
            )
        else:
            self.monitoring = None
        
        # Server startup time
        self.startup_time = datetime.utcnow()
        
        # Initialize logger
        self.logger = get_logger('server.core.app')
        
        # Initialize middleware and endpoints
        self._setup_middleware()
        self._create_endpoints()
        
        print(f"🚀 WebAgents V2 Server initialized")
        print(f"   URL prefix: {self.url_prefix or '(none)'}")
        print(f"   Static agents: {len(self.static_agents)}")
        print(f"   Dynamic agents: {'✅ Enabled' if self.dynamic_agents else '❌ Disabled'}")
        print(f"   Monitoring: {'✅ Enabled' if self.monitoring else '❌ Disabled'}")
    
    def _setup_middleware(self):
        """Set up FastAPI middleware"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request timeout and logging middleware
        if self.enable_request_logging:
            self.app.add_middleware(
                RequestLoggingMiddleware,
                timeout=self.request_timeout
            )
        
        # Rate limiting middleware
        if self.enable_rate_limiting:
            self.app.add_middleware(
                RateLimitMiddleware,
                default_rule=self.default_rate_limit,
                user_rules=self.user_rate_limits
            )
    
    def _create_endpoints(self):
        """Create all FastAPI endpoints"""
        
        # Health check endpoint
        @self.router.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            uptime_seconds = (datetime.utcnow() - self.startup_time).total_seconds()
            
            return HealthResponse(
                status="healthy",
                version=self.version,
                uptime_seconds=uptime_seconds,
                agents_count=len(self.static_agents),
                dynamic_agents_enabled=self.dynamic_agents is not None
            )
        
        # Server info endpoint
        @self.router.get("/info")
        async def server_info():
            """Get server information"""
            uptime_seconds = (datetime.utcnow() - self.startup_time).total_seconds()
            
            # Build endpoints with prefix
            endpoints = {
                "health": f"{self.url_prefix}/health",
                "info": f"{self.url_prefix}/info",
                "stats": f"{self.url_prefix}/stats"
            }
            
            if self.monitoring and self.monitoring.enable_prometheus:
                endpoints["metrics"] = f"{self.url_prefix}/metrics"
            
            return {
                "name": "WebAgents V2 Server",
                "version": self.version,
                "status": "running",
                "uptime_seconds": uptime_seconds,
                "static_agents_count": len(self.static_agents),
                "dynamic_agents_enabled": self.dynamic_agents is not None,
                "monitoring_enabled": self.monitoring is not None,
                "endpoints": endpoints
            }
        
        # Server stats endpoint
        @self.router.get("/stats")
        async def server_stats():
            """Get comprehensive server statistics"""
            uptime_seconds = (datetime.utcnow() - self.startup_time).total_seconds()
            
            stats = {
                "server": {
                    "name": "WebAgents V2 Server",
                    "version": self.version,
                    "uptime_seconds": uptime_seconds,
                    "startup_time": self.startup_time.isoformat()
                },
                "agents": {
                    "static_count": len(self.static_agents),
                    "static_names": list(self.static_agents.keys())
                },
                "dynamic_agents": {
                    "enabled": self.dynamic_agents is not None
                }
            }
            
            # Add monitoring performance stats
            if self.monitoring:
                stats["performance"] = self.monitoring.get_performance_stats()
                
                # Update system metrics
                self.monitoring.update_system_metrics(
                    active_agents=len(self.static_agents),
                    dynamic_cache_size=0  # Server doesn't know about caching
                )
            
            return stats
        
        # Agents listing endpoint  
        @self.router.get("/agents")
        async def list_agents():
            """List all available agents (static and dynamic)"""
            agents_list = []
            
            # Add static agents
            for agent_name, agent in self.static_agents.items():
                agents_list.append({
                    "name": agent_name,
                    "type": "static",
                    "instructions": agent.instructions,
                    "scopes": agent.scopes,
                    "tools_count": len(agent.get_tools_for_scope("all")),
                    "http_handlers_count": len(agent.get_all_http_handlers()),
                    "status": "active"
                })
            
            # Add dynamic agents info if available  
            dynamic_agents_available = []
            if self.dynamic_agents:
                try:
                    # Try to get available dynamic agents
                    # Note: This depends on dynamic agent implementation
                    dynamic_agents_available = []  # Placeholder - would need dynamic agent resolver to list
                except Exception:
                    pass
            
            return {
                "agents": agents_list,
                "static_count": len(self.static_agents),
                "dynamic_count": len(dynamic_agents_available),
                "total_count": len(agents_list) + len(dynamic_agents_available)
            }
        
        # Prometheus metrics endpoint
        if self.monitoring and self.monitoring.enable_prometheus:
            @self.router.get("/metrics")
            async def prometheus_metrics():
                """Prometheus metrics endpoint"""
                metrics_data = self.monitoring.get_metrics_response()
                return Response(
                    content=metrics_data,
                    media_type=CONTENT_TYPE_LATEST
                )
        
        # Static agent endpoints
        for agent_name in self.static_agents.keys():
            self._create_agent_endpoints(agent_name, is_dynamic=False)
        
        # Dynamic agent endpoints (if resolver available)
        if self.dynamic_agents:
            @self.router.get("/{agent_name}", response_model=AgentInfoResponse)
            async def dynamic_agent_info(agent_name: str):
                return await self._handle_agent_info(agent_name, is_dynamic=True)
            
            @self.router.post("/{agent_name}/chat/completions")
            async def dynamic_chat_completion(agent_name: str, request: ChatCompletionRequest, raw_request: Request = None):
                return await self._handle_chat_completion(agent_name, request, raw_request, is_dynamic=True)
            
            @self.router.get("/{agent_name}/health")
            async def dynamic_agent_health(agent_name: str):
                """Dynamic agent health check"""
                try:
                    agent = await self._resolve_agent(agent_name, is_dynamic=True)
                    return {
                        "agent_name": agent.name,
                        "status": "healthy",
                        "type": "dynamic_agent",
                        "instructions_preview": agent.instructions[:100] + "..." if len(agent.instructions) > 100 else agent.instructions
                    }
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Agent health check failed: {str(e)}")

            # Generic dynamic HTTP handler dispatcher for @http handlers on dynamic agents.
            # This allows dynamic agents to expose custom HTTP endpoints without static registration.
            @self.router.api_route("/{agent_name}/{request_path:path}", methods=[
                "GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"
            ])
            async def dynamic_agent_http_dispatch(
                agent_name: str,
                request_path: str,
                request: Request
            ):
                import re
                import inspect as _inspect
                import asyncio as _asyncio

                # Avoid handling reserved built-in endpoints; let their specific routes match first
                reserved_suffixes = {
                    "chat/completions", "health", "", "info", "metrics", "stats", "agents"
                }
                if request_path in reserved_suffixes:
                    raise HTTPException(status_code=404, detail="Not found")

                # Resolve the dynamic agent
                agent = await self._resolve_agent(agent_name, is_dynamic=True)
                # Ensure skills are initialized so skill methods have agent context
                try:
                    if hasattr(agent, '_ensure_skills_initialized'):
                        await agent._ensure_skills_initialized()
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to initialize agent skills: {e}")

                # Iterate agent's http handlers and find a matching subpath and method
                try:
                    http_handlers = agent.get_all_http_handlers()
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to load HTTP handlers: {e}")

                request_method = request.method.lower()
                # Normalize request_path (no leading slash)
                normalized_path = request_path.lstrip("/")

                # Build a matcher for handler subpaths supporting path params like {id}
                def build_regex(subpath: str):
                    # Remove leading slash for comparison
                    sp = subpath.lstrip("/")
                    # Find parameter names and build regex pattern
                    param_names = re.findall(r"\{([^}]+)\}", sp)
                    # Replace {param} with a capture group that matches a single path segment
                    pattern = re.sub(r"\{[^}]+\}", r"([^/]+)", sp)
                    # Anchor full match
                    pattern = f"^{pattern}$"
                    return re.compile(pattern), param_names

                # Attempt to match in declaration order
                for handler_config in http_handlers:
                    try:
                        subpath = handler_config.get('subpath') or ''
                        method = (handler_config.get('method') or 'get').lower()
                        handler_func = handler_config.get('function')
                        handler_scope = handler_config.get('scope', 'all')

                        if method != request_method:
                            continue

                        regex, param_names = build_regex(subpath)
                        match = regex.match(normalized_path)
                        if not match:
                            continue

                        # Extract path params from capture groups
                        path_param_values = match.groups()
                        path_params = {name: value for name, value in zip(param_names, path_param_values)}

                        # Extract query params
                        query_params = dict(request.query_params)

                        # Extract body data (JSON or form-encoded) for methods that commonly have a body
                        body_data = {}
                        if request.method in ["POST", "PUT", "PATCH"]:
                            try:
                                # Try JSON first
                                body_data = await request.json()
                            except Exception:
                                # Fallback to form data (application/x-www-form-urlencoded)
                                try:
                                    form_data = await request.form()
                                    body_data = dict(form_data)
                                except Exception:
                                    body_data = {}

                        # Combine and filter parameters by handler signature
                        combined_params = {**path_params, **query_params, **body_data}
                        
                        # Remove 'token' from params (used for auth, not a handler param)
                        combined_params.pop('token', None)
                        
                        sig = _inspect.signature(handler_func)
                        filtered_params = {}
                        for param_name in sig.parameters:
                            if param_name in ('self', 'context'):
                                continue
                            if param_name in combined_params:
                                filtered_params[param_name] = combined_params[param_name]

                        # Set minimal request context for handlers that depend on it (e.g., owner scope/User ID)
                        try:
                            # Support token-based auth for localhost (cross-port authentication)
                            # In production, same origin means normal cookie/header auth works
                            token_from_url = query_params.get('token')
                            if token_from_url and ('localhost' in str(request.base_url) or '127.0.0.1' in str(request.base_url)):
                                # Inject token directly into request headers for authentication
                                # This modifies the headers in-place for this request only
                                from starlette.datastructures import MutableHeaders
                                # Access the internal _headers attribute and update it
                                if hasattr(request, '_headers'):
                                    if not isinstance(request._headers, MutableHeaders):
                                        request._headers = MutableHeaders(request._headers)
                                    request._headers['authorization'] = f'Bearer {token_from_url}'
                                ctx = create_context(messages=[], stream=False, agent=agent, request=request)
                            else:
                                ctx = create_context(messages=[], stream=False, agent=agent, request=request)
                            set_context(ctx)
                        except Exception as e:
                            # Log auth errors but don't fail the request
                            import logging
                            logging.getLogger('webagents.server').debug(f"Context creation error: {e}")

                        # Check authentication/authorization if handler has restricted scope
                        if handler_scope != 'all':
                            # Check if handler requires owner/admin scope
                            required_scopes = [handler_scope] if isinstance(handler_scope, str) else handler_scope
                            
                            # Get user's auth context
                            user_scopes = []
                            try:
                                ctx = get_context()
                                if ctx and ctx.auth and hasattr(ctx.auth, 'scope'):
                                    from ...agents.skills.robutler.auth.types import AuthScope
                                    auth_scope = ctx.auth.scope
                                    if auth_scope == AuthScope.ADMIN:
                                        user_scopes = ['admin', 'owner', 'all']
                                    elif auth_scope == AuthScope.OWNER:
                                        user_scopes = ['owner', 'all']
                                    elif auth_scope == AuthScope.USER:
                                        user_scopes = ['all']
                            except Exception:
                                # If we can't get auth context, user is unauthenticated
                                user_scopes = []
                            
                            # Fallback: For localhost, check if token in URL matches agent's API key (cross-port auth)
                            token_from_url = query_params.get('token')
                            if not user_scopes and token_from_url and ('localhost' in str(request.base_url) or '127.0.0.1' in str(request.base_url)):
                                # Validate token matches agent's API key
                                if hasattr(agent, 'api_key') and agent.api_key == token_from_url:
                                    # Grant owner scope for valid agent API key on localhost
                                    user_scopes = ['owner', 'all']
                            
                            # Check if user has required scope
                            has_access = any(scope in required_scopes or scope == 'all' for scope in user_scopes)
                            if not has_access:
                                raise HTTPException(
                                    status_code=403, 
                                    detail=f"Access denied. This endpoint requires one of: {', '.join(required_scopes)}"
                                )

                        # Call the handler function (async or sync)
                        if _inspect.iscoroutinefunction(handler_func):
                            result = await handler_func(**filtered_params)
                        else:
                            # Run sync handler directly
                            result = handler_func(**filtered_params)

                        return result
                    except HTTPException:
                        raise
                    except Exception as e:
                        # If a handler matched but failed, surface the error
                        # Otherwise continue searching other handlers
                        last_error = str(e)
                        return Response(status_code=500, content=str(last_error))

                # No matching handler found
                raise HTTPException(status_code=404, detail=f"No HTTP handler for path '/{normalized_path}' and method {request.method} on agent '{agent_name}'")
        
        # Include the router in the main app
        self.app.include_router(self.router)
    
    def _create_agent_endpoints(self, agent_name: str, is_dynamic: bool = False):
        """Create endpoints for a specific agent"""
        
        @self.router.get(f"/{agent_name}", response_model=AgentInfoResponse)
        async def agent_info():
            return await self._handle_agent_info(agent_name, is_dynamic=is_dynamic)
        
        @self.router.post(f"/{agent_name}/chat/completions")
        async def chat_completion(request: ChatCompletionRequest, raw_request: Request = None):
            return await self._handle_chat_completion(agent_name, request, raw_request, is_dynamic=is_dynamic)
        
        @self.router.get(f"/{agent_name}/health")
        async def agent_health():
            """Agent health check"""
            try:
                agent = await self._resolve_agent(agent_name, is_dynamic=is_dynamic)
                return {
                    "agent_name": agent.name,
                    "status": "healthy",
                    "type": "static_agent" if not is_dynamic else "dynamic_agent",
                    "instructions_preview": agent.instructions[:100] + "..." if len(agent.instructions) > 100 else agent.instructions
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Agent health check failed: {str(e)}")
        
        # Register HTTP handlers if agent has any
        if not is_dynamic:
            agent = self.static_agents.get(agent_name)
            if agent:
                self._register_agent_http_handlers(agent_name, agent)
    
    def _register_agent_http_handlers(self, agent_name: str, agent: BaseAgent):
        """Register agent's HTTP handlers as FastAPI routes with dynamic parameter support"""
        import inspect  # Import at method level to ensure availability
        import re
        import asyncio
        
        try:
            http_handlers = agent.get_all_http_handlers()
            
            for handler_config in http_handlers:
                subpath = handler_config['subpath']
                method = handler_config['method'].lower()
                handler_func = handler_config['function']
                scope = handler_config.get('scope', 'all')
                description = handler_config.get('description', '')
                
                # Create full path: /{agent_name}{subpath}
                full_path = f"/{agent_name}{subpath}"
                
                # Extract path parameter names from subpath (e.g., {param}, {user_id})
                path_params = re.findall(r'\{([^}]+)\}', subpath)
                
                def create_http_wrapper(handler_func, scope, method, path_param_names):
                    # Create dynamic function based on path parameters
                    if path_param_names:
                        # Create a wrapper that accepts path parameters as individual arguments
                        async def http_wrapper(request: Request, **kwargs):
                            try:
                                # Extract path parameters from kwargs (FastAPI automatically extracts them)
                                path_params = {name: kwargs.get(name) for name in path_param_names if name in kwargs}
                                
                                # Extract query parameters
                                query_params = dict(request.query_params)
                                
                                # Extract JSON body if present
                                body_data = {}
                                if request.method in ["POST", "PUT", "PATCH"]:
                                    try:
                                        body_data = await request.json()
                                    except:
                                        pass
                                
                                # Combine all parameters: path params, query params, body data
                                all_params = {**path_params, **query_params, **body_data}
                                
                                # Get function signature to handle parameters properly
                                sig = inspect.signature(handler_func)
                                
                                # Filter parameters to only include those expected by the function
                                filtered_params = {}
                                for param_name in sig.parameters:
                                    if param_name in ('self', 'context'):
                                        continue
                                    if param_name in all_params:
                                        filtered_params[param_name] = all_params[param_name]
                                
                                # Call the handler function
                                if asyncio.iscoroutinefunction(handler_func):
                                    result = await handler_func(**filtered_params)
                                else:
                                    result = handler_func(**filtered_params)
                                
                                return result
                                
                            except Exception as e:
                                raise HTTPException(status_code=500, detail=str(e))
                        
                        # Now we need to dynamically add the correct path parameters to the wrapper signature
                        # Create a new function with the correct signature
                        
                        # Build parameter list for the new function
                        params = [inspect.Parameter('request', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request)]
                        for param_name in path_param_names:
                            params.append(inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str))
                        
                        # Create new signature
                        new_sig = inspect.Signature(params)
                        
                        # Create a wrapper with the correct signature
                        def make_wrapper():
                            async def wrapper(*args, **kwargs):
                                # Convert args to kwargs based on signature
                                bound_args = new_sig.bind(*args, **kwargs)
                                bound_args.apply_defaults()
                                return await http_wrapper(**bound_args.arguments)
                            wrapper.__signature__ = new_sig
                            return wrapper
                        
                        return make_wrapper()
                    else:
                        # No path parameters, simpler wrapper
                        async def http_wrapper(request: Request):
                            try:
                                # Extract query parameters
                                query_params = dict(request.query_params)
                                
                                # Extract JSON body if present
                                body_data = {}
                                if request.method in ["POST", "PUT", "PATCH"]:
                                    try:
                                        body_data = await request.json()
                                    except:
                                        pass
                                
                                # Combine all parameters: query params, body data
                                all_params = {**query_params, **body_data}
                                
                                # Get function signature to handle parameters properly
                                sig = inspect.signature(handler_func)
                                
                                # Filter parameters to only include those expected by the function
                                filtered_params = {}
                                for param_name in sig.parameters:
                                    if param_name in ('self', 'context'):
                                        continue
                                    if param_name in all_params:
                                        filtered_params[param_name] = all_params[param_name]
                                
                                # Call the handler function
                                if asyncio.iscoroutinefunction(handler_func):
                                    result = await handler_func(**filtered_params)
                                else:
                                    result = handler_func(**filtered_params)
                                
                                return result
                                
                            except Exception as e:
                                raise HTTPException(status_code=500, detail=str(e))
                        
                        return http_wrapper
                
                # Create the wrapper
                wrapper = create_http_wrapper(handler_func, scope, method, path_params)
                
                # Register with FastAPI based on HTTP method (using router instead of app directly)
                # FastAPI will automatically handle path parameter extraction
                if method == "get":
                    self.router.get(full_path, summary=description)(wrapper)
                elif method == "post":
                    self.router.post(full_path, summary=description)(wrapper)
                elif method == "put":
                    self.router.put(full_path, summary=description)(wrapper)
                elif method == "delete":
                    self.router.delete(full_path, summary=description)(wrapper)
                elif method == "patch":
                    self.router.patch(full_path, summary=description)(wrapper)
                elif method == "head":
                    self.router.head(full_path, summary=description)(wrapper)
                elif method == "options":
                    self.router.options(full_path, summary=description)(wrapper)
                
                print(f"📡 Registered HTTP endpoint: {method.upper()} {self.url_prefix}{full_path}")
                
        except Exception as e:
            print(f"⚠️ Error registering HTTP handlers for agent '{agent_name}': {e}")
    
    async def _handle_agent_info(self, agent_name: str, is_dynamic: bool = False) -> AgentInfoResponse:
        """Handle agent info requests"""
        agent = await self._resolve_agent(agent_name, is_dynamic=is_dynamic)
        
        return AgentInfoResponse(
            name=agent.name,
            instructions=agent.instructions,
            model="webagents-v2",  # Generic model identifier
            endpoints={
                "chat_completions": f"{self.url_prefix}/{agent_name}/chat/completions",
                "health": f"{self.url_prefix}/{agent_name}/health"
            }
        )
    
    async def _handle_chat_completion(self, agent_name: str, request: ChatCompletionRequest, raw_request: Request = None, is_dynamic: bool = False):
        """Handle chat completion requests"""
        
        # Resolve the agent
        agent = await self._resolve_agent(agent_name, is_dynamic=is_dynamic)
        
        # Create context for this request
        context = create_context(
            messages=request.messages,
            stream=request.stream,
            agent=agent,
            request=raw_request
        )
        
        set_context(context)
        
        try:
            # Convert Pydantic objects to dictionaries for LiteLLM compatibility
            messages_dict = []
            for msg in request.messages:
                if hasattr(msg, 'model_dump'):
                    # Pydantic v2
                    msg_dict = msg.model_dump()
                elif hasattr(msg, 'dict'):
                    # Pydantic v1
                    msg_dict = msg.dict()
                else:
                    # Already a dict
                    msg_dict = msg
                
                # Ensure required fields exist
                if 'role' not in msg_dict:
                    msg_dict['role'] = 'user'
                if 'content' not in msg_dict:
                    msg_dict['content'] = ''
                    
                messages_dict.append(msg_dict)
            
            # Convert tools to dictionaries if present
            tools_dict = None
            if request.tools:
                tools_dict = []
                for tool in request.tools:
                    if hasattr(tool, 'model_dump'):
                        tools_dict.append(tool.model_dump())
                    elif hasattr(tool, 'dict'):
                        tools_dict.append(tool.dict())
                    else:
                        tools_dict.append(tool)
            
            if request.stream:
                # Handle streaming response
                return await self._stream_response(agent, request, messages_dict, tools_dict)
            else:
                # Handle non-streaming response
                response = await agent.run(
                    messages=messages_dict,
                    tools=tools_dict,
                    stream=False
                )
                return response
                
        except Exception as e:
            # Check if this is a payment-related error with custom status code
            self.logger.error(f"🚨 Agent execution error: {type(e).__name__}: {str(e)}")
            
            if hasattr(e, 'status_code') and hasattr(e, 'detail'):
                self.logger.error(f"   - Found status_code: {getattr(e, 'status_code', None)}, detail present: {hasattr(e, 'detail')}")
                self.logger.error(f"   - Raising HTTPException with status_code={getattr(e, 'status_code', 500)}")
                raise HTTPException(
                    status_code=getattr(e, 'status_code', 500),
                    detail=getattr(e, 'detail')
                )
            else:
                self.logger.error(f"   - No status_code/detail attributes, defaulting to 500")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error executing agent '{agent_name}': {str(e)}"
                )
    
    async def _stream_response(self, agent: BaseAgent, request: ChatCompletionRequest, messages_dict: List[Dict[str, Any]], tools_dict: Optional[List[Dict[str, Any]]]):
        """Handle streaming response"""
        from fastapi.responses import StreamingResponse
        import json
        
        # Start the generator to catch early errors (on_connection, etc.)
        generator = agent.run_streaming(messages=messages_dict, tools=tools_dict)
        
        # Try to get the first chunk - this triggers on_connection hooks
        try:
            first_chunk = await generator.__anext__()
        except StopAsyncIteration:
            # Empty stream - should not happen but handle gracefully
            first_chunk = None
        except Exception as e:
            # Error before streaming started (e.g., on_connection payment error)
            # Return as HTTP error instead of SSE chunk for better frontend handling
            if hasattr(e, 'status_code') and hasattr(e, 'detail'):
                self.logger.error(f"💳 Early error before streaming: {e}")
                raise HTTPException(
                    status_code=getattr(e, 'status_code', 500),
                    detail=getattr(e, 'detail')
                )
            raise
        
        async def generate():
            try:
                # Yield the first chunk we already fetched
                if first_chunk is not None:
                    try:
                        chunk_json = json.dumps(first_chunk)
                        yield f"data: {chunk_json}\n\n"
                    except Exception as json_error:
                        self.logger.error(f"Failed to serialize first chunk: {json_error}")
                
                # Continue with remaining chunks
                async for chunk in generator:
                    # Properly serialize chunk to JSON for SSE format
                    try:
                        chunk_json = json.dumps(chunk)
                        yield f"data: {chunk_json}\n\n"
                    except Exception as json_error:
                        self.logger.error(f"Failed to serialize streaming chunk: {json_error}, chunk: {chunk}")
                        # Skip malformed chunks instead of breaking the stream
                        continue
                yield "data: [DONE]\n\n"
            except Exception as e:
                self.logger.error(f"Streaming error: {e}")
                
                # Check if this is a payment error with status_code and detail
                if hasattr(e, 'status_code') and hasattr(e, 'detail'):
                    status_code = getattr(e, 'status_code', 500)
                    detail = getattr(e, 'detail')
                    self.logger.error(f"   - Payment/custom error: status={status_code}, detail={detail}")
                    
                    # Format error in OpenAI-compatible format for AI SDK
                    # Also include payment-specific fields for frontend handling
                    error_code = detail.get("error") if isinstance(detail, dict) else "PAYMENT_ERROR"
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "insufficient_balance" if status_code == 402 else "server_error",
                            "code": error_code,
                            "status_code": status_code,
                            "detail": detail,
                            "error_code": error_code,  # For payment token manager compatibility
                            "statusCode": status_code  # For payment token manager compatibility
                        }
                    }
                else:
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                            "code": "internal_error"
                        }
                    }
                error_json = json.dumps(error_chunk)
                yield f"data: {error_json}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
    
    async def _resolve_agent(self, agent_name: str, is_dynamic: bool = False) -> BaseAgent:
        """Resolve agent by name from static or dynamic sources"""
        
        # Try static agents first
        agent = self.static_agents.get(agent_name)
        if agent:
            return agent
        
        # If not looking for dynamic agents, stop here
        if not is_dynamic:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Try dynamic resolver if available
        if self.dynamic_agents:
            try:
                if asyncio.iscoroutinefunction(self.dynamic_agents):
                    agent = await self.dynamic_agents(agent_name)
                else:
                    agent = self.dynamic_agents(agent_name)
                
                if agent and agent is not False:
                    return agent
            except Exception as e:
                print(f"Error resolving dynamic agent '{agent_name}': {e}")
        
        # Agent not found
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    # Convenience property to access the FastAPI app
    @property
    def fastapi_app(self) -> FastAPI:
        """Get the underlying FastAPI application"""
        return self.app


# Factory function for easy server creation
def create_server(
    title: str = "WebAgents V2 Server",
    description: str = "AI Agent Server with OpenAI Compatibility", 
    version: str = "2.0.0",
    agents: List[BaseAgent] = None,
    dynamic_agents: Optional[Union[Callable[[str], BaseAgent], Callable[[str], Awaitable[Optional[BaseAgent]]]]] = None,
    url_prefix: str = "",
    **kwargs
) -> WebAgentsServer:
    """
    Create a WebAgents server instance
    
    Args:
        title: Server title
        description: Server description
        version: Server version
        agents: List of static agents
        dynamic_agents: Optional dynamic agent resolver function (sync or async)
        url_prefix: URL prefix for all routes (e.g., "/agents")
        **kwargs: Additional server configuration
        
    Returns:
        Configured WebAgentsServer instance
    """
    return WebAgentsServer(
        title=title,
        description=description,
        version=version,
        agents=agents or [],
        dynamic_agents=dynamic_agents,
        url_prefix=url_prefix,
        **kwargs
    ) 