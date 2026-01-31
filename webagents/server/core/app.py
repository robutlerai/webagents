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
from pathlib import Path
import inspect
import inspect as _inspect
import json

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from ..monitoring import CONTENT_TYPE_LATEST

from .models import (
    ChatCompletionRequest, ChatCompletionResponse, AgentInfoResponse, 
    HealthResponse, AgentListResponse, ServerStatsResponse,
    RegisterAgentRequest
)
from .middleware import RequestLoggingMiddleware, RateLimitMiddleware, RateLimitRule, WorkingDirMiddleware
from .monitoring import initialize_monitoring
from ..context.context_vars import Context, set_context, create_context, get_context
from ...agents.core.base_agent import BaseAgent
from ...utils.logging import get_logger
from ..extensions.interface import AgentSource, WebAgentsExtension, WebAgentsPlugin


class WebAgentsServer:
    """
    FastAPI server for AI agents with OpenAI compatibility and production monitoring
    
    Features:
    - OpenAI-compatible chat/completions via CompletionsTransportSkill
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
        metrics_port: int = 9090,
        # Daemon/Extension configuration
        enable_file_watching: bool = False,
        watch_dirs: Optional[List[Path]] = None,
        enable_cron: bool = False,
        extension_config: Optional[Dict[str, Any]] = None,
        plugin_config: Optional[Dict[str, Any]] = None,  # Deprecated, use extension_config
        storage_backend: str = "json"
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
            version=version,
        )
        
        self.version = version
        self.url_prefix = url_prefix.rstrip("/")  # Remove trailing slash if present
        
        # Create API router with prefix
        self.router = APIRouter(prefix=self.url_prefix)
        
        # Store agents by name for quick lookup
        self.static_agents = {agent.name: agent for agent in (agents or [])}
        
        # Store dynamic agent resolver (server doesn't manage how it works)
        self.dynamic_agents = dynamic_agents
        
        # Extension system (previously called "plugins")
        self.agent_sources: List[AgentSource] = []
        self.extensions: List[WebAgentsExtension] = []
        
        # Initialize storage backend
        if storage_backend == "json":
            from ..storage.json_store import JSONMetadataStore
            self.metadata_store = JSONMetadataStore()
        elif storage_backend == "litesql":
            from ..storage.litesql_store import LiteSQLMetadataStore
            self.metadata_store = LiteSQLMetadataStore()
        else:
            from ..storage.json_store import JSONMetadataStore
            self.metadata_store = JSONMetadataStore()
        
        # Daemon components (optional)
        self.watcher = None
        self.cron = None
        self.manager = None
        self.registry = None
        
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
        
        # Load extensions if provided (support both new and deprecated keys)
        config_to_load = extension_config or plugin_config
        if config_to_load:
            if plugin_config and not extension_config:
                import warnings
                warnings.warn(
                    "plugin_config is deprecated, use extension_config instead",
                    DeprecationWarning,
                    stacklevel=2
                )
            self._load_extensions(config_to_load)
        
        # Enable file watching if requested
        if enable_file_watching:
            from ...cli.daemon.watcher import FileWatcher
            from ...cli.daemon.registry import DaemonRegistry
            from ..extensions.local_file_source import LocalFileSource
            self.registry = DaemonRegistry()
            self.watcher = FileWatcher(
                registry=self.registry,
                watch_dirs=watch_dirs or [Path.cwd()],
                on_change=self._handle_file_change
            )
            
            # Register local source
            local_source = LocalFileSource(
                watch_dirs=watch_dirs or [Path.cwd()],
                metadata_store=self.metadata_store,
                registry=self.registry
            )
            self.agent_sources.append(local_source)
        
        # Enable cron if requested
        if enable_cron:
            from ...cli.daemon.cron import CronScheduler
            from ...cli.daemon.manager import AgentManager
            if not self.registry:
                from ...cli.daemon.registry import DaemonRegistry
                self.registry = DaemonRegistry()
            self.manager = AgentManager(self.registry)
            self.cron = CronScheduler(self.manager)
        
        # Initialize middleware and endpoints
        self._setup_middleware()
        self._create_endpoints()
        self._setup_events()
    
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
        
        # Working directory middleware (extracts X-Working-Dir header)
        self.app.add_middleware(WorkingDirMiddleware)
        
        # Request timeout and logging middleware
        if self.enable_request_logging:
            @self.app.middleware("http")
            async def log_request(request: Request, call_next):
                import time
                start_time = time.time()
                # self.logger.info(f"➜ {request.method} {request.url.path}")
                response = await call_next(request)
                duration = time.time() - start_time
                # self.logger.info(f"← {response.status_code} ({duration:.3f}s)")
                return response
        
        # Rate limiting middleware (disabled as it may cause buffering)
        if self.enable_rate_limiting and False:
            self.app.add_middleware(
                RateLimitMiddleware,
                default_rule=self.default_rate_limit,
                user_rules=self.user_rate_limits
            )
    
    def _mount_webui(self):
        """Mount WebUI static files at /ui."""
        from pathlib import Path
        import logging
        
        logger = logging.getLogger("webagents.server")
        
        # Path to compiled React app
        # webagents/server/core/app.py -> webagents/cli/webui/dist/
        dist_dir = Path(__file__).parent.parent.parent / "cli" / "webui" / "dist"
        
        if not dist_dir.exists():
            logger.debug(f"WebUI not found at {dist_dir}. Run 'webagents ui --build' to build.")
            return
        
        assets_dir = dist_dir / "assets"
        index_file = dist_dir / "index.html"
        
        if not index_file.exists():
            logger.debug(f"WebUI index.html not found at {index_file}")
            return
        
        try:
            from starlette.staticfiles import StaticFiles
            from starlette.responses import FileResponse
            
            # Capture index_file path for closure
            index_file_path = str(index_file)
            
            # Mount static assets (JS, CSS, images)
            if assets_dir.exists():
                self.app.mount(
                    "/ui/assets",
                    StaticFiles(directory=str(assets_dir)),
                    name="webui_assets"
                )
            
            # Create route handlers with captured path
            async def serve_ui_root():
                """Serve React SPA root."""
                return FileResponse(index_file_path, media_type="text/html")
            
            async def serve_ui_path(path: str):
                """Serve React SPA - all routes return index.html."""
                return FileResponse(index_file_path, media_type="text/html")
            
            # Register routes
            self.app.add_api_route("/ui", serve_ui_root, methods=["GET"])
            self.app.add_api_route("/ui/{path:path}", serve_ui_path, methods=["GET"])
            
            logger.info(f"WebUI mounted at /ui")
            
        except Exception as e:
            import traceback
            logger.debug(f"Failed to mount WebUI: {e}")
    
    def _create_endpoints(self):
        """Create all FastAPI endpoints"""
        
        # Health check endpoint - on app root, not router (accessible at /health)
        @self.app.get("/health", response_model=HealthResponse)
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
        
        @self.app.get("/health/detailed")
        async def detailed_health():
            """Detailed health check with agent status."""
            agent_status = {}
            for agent_name, agent in self.static_agents.items():
                agent_status[agent_name] = {
                    "status": "healthy",
                    "skills": list(agent.skills.keys()) if hasattr(agent, 'skills') else [],
                    "tools_count": len(agent.get_tools_for_scope("all")) if hasattr(agent, 'get_tools_for_scope') else 0
                }
            return {
                "status": "healthy",
                "agents": agent_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/ready")
        async def readiness_check():
            """Kubernetes readiness probe."""
            return {"status": "ready", "details": {"agents_loaded": len(self.static_agents)}}
        
        @self.app.get("/live")
        async def liveness_check():
            """Kubernetes liveness probe."""
            uptime_seconds = (datetime.utcnow() - self.startup_time).total_seconds()
            return {"status": "alive", "uptime_seconds": uptime_seconds}
        
        @self.app.get("/metrics")
        async def root_metrics():
            """Prometheus-compatible metrics at root level."""
            if self.monitoring and self.monitoring.enable_prometheus:
                metrics_data = self.monitoring.get_metrics_response()
                return Response(
                    content=metrics_data,
                    media_type="text/plain"
                )
            # Fallback simple metrics when monitoring is disabled
            lines = [
                "# HELP webagents_agents_total Total number of agents",
                f"webagents_agents_total {len(self.static_agents)}",
                "# HELP webagents_up Server is up",
                "webagents_up 1",
            ]
            return Response(content="\n".join(lines), media_type="text/plain")
        
        # Server info endpoint
        @self.router.get("/info")
        async def server_info():
            """Get server information"""
            uptime_seconds = (datetime.utcnow() - self.startup_time).total_seconds()
            
            # Build endpoints with prefix (health is at root)
            endpoints = {
                "health": "/health",
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
        
        # Agents listing endpoint (at prefix root, e.g., GET /agents with url_prefix="/agents")
        @self.router.get("/")
        async def list_agents(query: Optional[str] = None):
            """List or search available agents (static, dynamic, and plugin sources)
            
            Args:
                query: Optional search query (name pattern with wildcards)
            
            Returns:
                List of agent metadata from all sources
            """
            agents_list = []
            
            # Add static agents
            for agent_name, agent in self.static_agents.items():
                # Apply search filter if provided
                if query:
                    import fnmatch
                    if not fnmatch.fnmatch(agent_name.lower(), query.lower()):
                        continue
                
                agents_list.append({
                    "name": agent_name,
                    "type": "static",
                    "source": "static",
                    "instructions": agent.instructions,
                    "scopes": agent.scopes,
                    "tools_count": len(agent.get_tools_for_scope("all")),
                    "http_handlers_count": len(agent.get_all_http_handlers()),
                    "status": "active"
                })
            
            # Query plugin sources
            if query:
                # Search across all agent sources
                for source in self.agent_sources:
                    try:
                        matches = await source.search_agents(query)
                        agents_list.extend(matches)
                    except Exception as e:
                        self.logger.error(f"Error searching source {source.get_source_type()}: {e}")
            else:
                # List all agents from all sources
                for source in self.agent_sources:
                    try:
                        source_agents = await source.list_agents()
                        agents_list.extend(source_agents)
                    except Exception as e:
                        self.logger.error(f"Error listing source {source.get_source_type()}: {e}")
            
            return {
                "agents": agents_list,
                "count": len(agents_list),
                "query": query
            }
        
        # Daemon-specific endpoints (at prefix root)
        @self.router.post("/")
        async def register_agent(request: RegisterAgentRequest):
            """Register an agent from file"""
            if not self.registry:
                raise HTTPException(400, "Registry not enabled")
            
            path = Path(request.path)
            if not path.exists():
                 raise HTTPException(400, f"Path not found: {path}")

            try:
                if path.is_dir():
                    count = await self.registry.scan_directory(path)
                    return {"status": "registered", "count": count, "type": "directory"}
                else:
                    agent = self.registry.update_from_file(path)
                    return agent.to_dict()
            except Exception as e:
                raise HTTPException(400, str(e))
        
        @self.router.delete("/{name}")
        async def unregister_agent(name: str):
            """Unregister an agent"""
            if not self.registry:
                raise HTTPException(400, "Registry not enabled")
            
            if self.registry.unregister(name):
                return {"status": "unregistered", "name": name}
            else:
                raise HTTPException(404, f"Agent '{name}' not found")
        
        @self.router.get("/cron")
        async def list_cron_jobs():
            """List cron jobs"""
            if not self.cron:
                return {"jobs": []}
            
            return {
                "jobs": [j.to_dict() for j in self.cron.list_jobs()]
            }
        
        @self.router.post("/cron")
        async def add_cron_job(agent: str, schedule: str):
            """Add a cron job"""
            if not self.cron:
                raise HTTPException(400, "Cron not enabled")
            
            job = self.cron.add_job(agent, schedule)
            return job.to_dict()
        
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
        
        # Dynamic agent endpoints (if resolver available or plugins present)
        if self.dynamic_agents or self.agent_sources:
            @self.router.get("/{agent_name}", response_model=AgentInfoResponse)
            async def dynamic_agent_info(agent_name: str):
                return await self._handle_agent_info(agent_name, is_dynamic=True)
            
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

            @self.router.get("/{agent_name}/command")
            async def dynamic_list_commands(agent_name: str, request: Request):
                """List available commands for a dynamic agent.
                
                Commands are dynamically discovered from agent skills via @command decorator.
                """
                # Get working_dir from request state (set by WorkingDirMiddleware)
                working_dir = getattr(request.state, 'working_dir', None)
                agent = await self._resolve_agent(agent_name, is_dynamic=True, working_dir=working_dir)
                commands = agent.list_commands()
                return {"commands": commands}
            
            @self.router.post("/{agent_name}/command/{path:path}")
            async def dynamic_execute_command(agent_name: str, path: str, request: Request):
                """Execute a command on a dynamic agent.
                
                Commands are exposed by agent skills via @command decorator.
                """
                working_dir = getattr(request.state, 'working_dir', None)
                agent = await self._resolve_agent(agent_name, is_dynamic=True, working_dir=working_dir)
                try:
                    data = await request.json()
                except Exception:
                    data = {}
                cmd_path = f"/{path}" if not path.startswith("/") else path
                result = await agent.execute_command(cmd_path, data)
                return {"result": result}
            
            @self.router.get("/{agent_name}/command/{path:path}")
            async def dynamic_get_command_docs(agent_name: str, path: str):
                """Get documentation for a specific command on a dynamic agent.
                
                Returns command info including path, description, parameters, and completions.
                """
                agent = await self._resolve_agent(agent_name, is_dynamic=True)
                cmd_path = f"/{path}" if not path.startswith("/") else path
                command = agent.get_command(cmd_path)
                if not command:
                    raise HTTPException(status_code=404, detail=f"Command not found: {cmd_path}")
                return command

            # WebSocket endpoint for agent skills
            @self.router.websocket("/{agent_name}/{ws_path:path}")
            async def websocket_endpoint(websocket: WebSocket, agent_name: str, ws_path: str):
                """WebSocket endpoint for agent skills"""
                # Resolve agent
                try:
                    agent = await self._resolve_agent(agent_name, is_dynamic=True)
                    if hasattr(agent, '_ensure_skills_initialized'):
                        await agent._ensure_skills_initialized()
                except Exception as e:
                    await websocket.close(code=4004, reason=f"Agent not found: {agent_name}")
                    return
                
                # Find matching WebSocket handler
                ws_handlers = agent.get_all_websocket_handlers()
                normalized_path = ws_path.lstrip("/")
                
                import re
                def build_regex(subpath: str):
                    sp = subpath.lstrip("/")
                    param_names = re.findall(r"\{([^}]+)\}", sp)
                    pattern = re.sub(r"\{[^}]+\}", r"([^/]+)", sp)
                    pattern = f"^{pattern}$"
                    return re.compile(pattern), param_names
                
                matched_handler = None
                path_params = {}
                
                for handler_config in ws_handlers:
                    handler_path = handler_config.get('path', '/')
                    regex, param_names = build_regex(handler_path)
                    match = regex.match(normalized_path)
                    if match:
                        path_param_values = match.groups()
                        path_params = {name: value for name, value in zip(param_names, path_param_values)}
                        matched_handler = handler_config
                        break
                
                if not matched_handler:
                    await websocket.close(code=4004, reason=f"No WebSocket handler for path: /{ws_path}")
                    return
                
                # Extract auth from query params or headers
                # WebSocket auth can come from:
                # 1. Query param: ?token=...
                # 2. Sec-WebSocket-Protocol header with token
                query_params = dict(websocket.query_params)
                token = query_params.get('token')
                
                # Create context with auth
                from ..context.context_vars import create_context, set_context, get_context
                ctx = create_context(messages=[], stream=True, agent=agent, request=websocket)
                set_context(ctx)
                
                # Check scope-based auth for the matched handler
                handler_scope = matched_handler.get('scope', 'all')
                if handler_scope != 'all':
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
                        user_scopes = []
                    
                    # Fallback: localhost token auth
                    if not user_scopes and token and ('localhost' in str(websocket.base_url) or '127.0.0.1' in str(websocket.base_url)):
                        if hasattr(agent, 'api_key') and agent.api_key == token:
                            user_scopes = ['owner', 'all']
                    
                    # Check if user has required scope
                    required_scopes = [handler_scope] if isinstance(handler_scope, str) else handler_scope
                    has_access = any(scope in required_scopes or scope == 'all' for scope in user_scopes)
                    if not has_access:
                        await websocket.close(code=4003, reason=f"Access denied. Requires: {', '.join(required_scopes)}")
                        return
                
                # Execute handler
                handler_func = matched_handler.get('function')
                try:
                    await handler_func(websocket, **path_params)
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    try:
                        await websocket.close(code=4000, reason=str(e)[:120])
                    except:
                        pass

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

                # Check if there's a skill HTTP handler for this path before blocking
                # This allows transport skills to override default endpoints like chat/completions
                reserved_suffixes = {
                    "health", "", "info", "metrics", "stats", "agents"
                }
                # "chat/completions" and "command" are now overridable by skills

                # For truly reserved endpoints (health, metrics, etc.), block skill override
                if request_path in reserved_suffixes:
                    raise HTTPException(status_code=404, detail="Not found")

                # Get working_dir from request state (set by WorkingDirMiddleware)
                working_dir = getattr(request.state, 'working_dir', None)
                
                # Resolve the dynamic agent
                agent = await self._resolve_agent(agent_name, is_dynamic=True, working_dir=working_dir)
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

                        # Check if handler is an async generator function (for SSE streaming)
                        if _inspect.isasyncgenfunction(handler_func):
                            # Async generator = SSE streaming response
                            async def sse_stream():
                                async for chunk in handler_func(**filtered_params):
                                    if isinstance(chunk, str):
                                        yield chunk
                                    elif isinstance(chunk, dict) and chunk.get('type'):
                                        # Custom event with type field - use SSE named event
                                        # OpenAI-compatible clients ignore named events
                                        event_type = chunk['type']
                                        yield f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                                    else:
                                        # Standard OpenAI-compatible chunk
                                        yield f"data: {json.dumps(chunk)}\n\n"
                            return StreamingResponse(
                                sse_stream(),
                                media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
                            )
                        elif _inspect.iscoroutinefunction(handler_func):
                            result = await handler_func(**filtered_params)
                        else:
                            # Run sync handler directly
                            result = handler_func(**filtered_params)

                        # Check if result is an async generator (returned from handler)
                        if hasattr(result, '__anext__'):
                            async def sse_stream():
                                async for chunk in result:
                                    if isinstance(chunk, str):
                                        yield chunk
                                    elif isinstance(chunk, dict) and chunk.get('type'):
                                        # Custom event with type field - use SSE named event
                                        # OpenAI-compatible clients ignore named events
                                        event_type = chunk['type']
                                        yield f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                                    else:
                                        # Standard OpenAI-compatible chunk
                                        yield f"data: {json.dumps(chunk)}\n\n"
                            return StreamingResponse(
                                sse_stream(),
                                media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
                            )

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
        
        # Mount WebUI before including router (so /ui routes take priority)
        self._mount_webui()
        
        # Include the router in the main app
        self.app.include_router(self.router)
    
    def _create_agent_endpoints(self, agent_name: str, is_dynamic: bool = False):
        """Create endpoints for a specific agent"""
        
        @self.router.get(f"/{agent_name}", response_model=AgentInfoResponse)
        async def agent_info():
            return await self._handle_agent_info(agent_name, is_dynamic=is_dynamic)
        
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
        
        @self.router.post(f"/{agent_name}/chat/completions")
        async def chat_completion(request: Request):
            """OpenAI-compatible chat completions endpoint.
            
            Uses agent.run_streaming() for streaming responses.
            Transport skills can override this via the catch-all route.
            """
            try:
                body = await request.json()
            except:
                raise HTTPException(status_code=422, detail="Invalid JSON")
            
            messages = body.get("messages", [])
            stream = body.get("stream", True)
            model = body.get("model", "")
            tools = body.get("tools")
            
            # Get the agent
            agent = self.static_agents.get(agent_name)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")
            
            if stream:
                async def generate():
                    import json as _json
                    try:
                        async for chunk in agent.run_streaming(messages, tools=tools):
                            yield f"data: {_json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        yield f"data: {_json.dumps({'error': str(e)})}\n\n"
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
                )
            else:
                # Non-streaming: collect all chunks
                result = await agent.run(messages, tools=tools)
                return result
        
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
    
    def _load_extensions(self, config: Dict[str, Any]):
        """Load extensions from configuration"""
        from ..extensions.loader import load_extensions
        
        self.extensions = load_extensions(config)
        
        for ext in self.extensions:
            # Register agent sources (async init handled in startup event)
            self.agent_sources.extend(ext.get_agent_sources())
    
    async def _initialize_extensions(self):
        """Initialize all extensions (called during startup)"""
        for ext in self.extensions:
            await ext.initialize(self)
    
    # Backwards compatibility aliases
    def _load_plugins(self, plugin_config: Dict[str, Any]):
        """Deprecated: Use _load_extensions instead."""
        import warnings
        warnings.warn(
            "_load_plugins is deprecated, use _load_extensions",
            DeprecationWarning,
            stacklevel=2
        )
        self._load_extensions(plugin_config)
    
    async def _initialize_plugins(self):
        """Deprecated: Use _initialize_extensions instead."""
        await self._initialize_extensions()
    
    @property
    def plugins(self) -> List[WebAgentsExtension]:
        """Deprecated: Use extensions instead."""
        return self.extensions
    
    async def _handle_file_change(self, event_type: str, path: Path):
        """Handle file change event from watcher
        
        Args:
            event_type: modified, created, deleted
            path: Path to changed file
        
        Note: We only invalidate cache - running agents continue until completion.
        New requests will get the updated agent definition.
        """
        # 1. Sync cron jobs if cron is enabled
        if self.cron and self.registry:
            self.cron.sync_from_registry(self.registry)
        
        # 2. Invalidate agent cache on file change
        # Running agents continue - only new requests get updated agent
        if self.registry:
            agent = self.registry.find_by_path(path)
            if agent:
                # Invalidate cache in all LocalFileSource instances
                from ..extensions.local_file_source import LocalFileSource
                for source in self.agent_sources:
                    if isinstance(source, LocalFileSource):
                        source.invalidate(agent.name)
                self.logger.info(f"Agent '{agent.name}' cache invalidated (file: {event_type})")
    
    async def resolve_agent(self, agent_name: str, working_dir: Optional[str] = None) -> Optional[BaseAgent]:
        """Resolve agent from all sources (static, dynamic, plugins)"""
        # Try static agents first
        agent = self.static_agents.get(agent_name)
        if agent:
            return agent
        
        # Try plugin sources
        for source in self.agent_sources:
            # Pass working_dir if the source supports it (e.g., LocalFileSource)
            if hasattr(source, 'get_agent'):
                try:
                    agent = await source.get_agent(agent_name, working_dir=working_dir)
                except TypeError:
                    # Source doesn't accept working_dir parameter
                    agent = await source.get_agent(agent_name)
            else:
                agent = await source.get_agent(agent_name)
            if agent:
                return agent
        
        # Try dynamic resolver (legacy support)
        if self.dynamic_agents:
            try:
                if asyncio.iscoroutinefunction(self.dynamic_agents):
                    agent = await self.dynamic_agents(agent_name)
                else:
                    agent = self.dynamic_agents(agent_name)
                
                if agent and agent is not False:
                    return agent
            except Exception as e:
                self.logger.error(f"Error resolving dynamic agent '{agent_name}': {e}")
        
        return None
    
    async def _resolve_agent(self, agent_name: str, is_dynamic: bool = False, working_dir: Optional[str] = None) -> BaseAgent:
        """Resolve agent by name from all sources (backward compatible wrapper)"""
        agent = await self.resolve_agent(agent_name, working_dir=working_dir)
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        return agent
    
    def _setup_events(self):
        """Setup startup and shutdown events"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Server startup event"""
            # Initialize extensions
            await self._initialize_extensions()
            
            # Restore registered agents from metadata store
            if self.registry and self.metadata_store and hasattr(self.metadata_store, 'agents'):
                try:
                    count = 0
                    for name, data in self.metadata_store.agents.items():
                        source_path = data.get("path") or data.get("source_path")
                        if source_path:
                            path = Path(source_path)
                            if path.exists():
                                try:
                                    self.registry.update_from_file(path)
                                    count += 1
                                except Exception as e:
                                    self.logger.warning(f"Failed to restore agent {name} from {path}: {e}")
                    
                    if count > 0:
                        self.logger.info(f"Restored {count} agents from storage")
                        
                except Exception as e:
                    self.logger.error(f"Error restoring agents from storage: {e}")
            
            # Start file watcher if enabled
            if self.watcher:
                asyncio.create_task(self.watcher.watch())
            
            # Start cron scheduler if enabled
            if self.cron:
                asyncio.create_task(self.cron.run())
            
            # Print server status
            print(f"🚀 WebAgents V2 Server ready")
            print(f"   URL prefix: {self.url_prefix or '(none)'}")
            print(f"   Static agents: {len(self.static_agents)}")
            if self.registry:
                print(f"   Registered agents: {len(self.registry.agents)}")
            
            # Show extension/dynamic agent status
            if self.agent_sources:
                source_types = [s.get_source_type() for s in self.agent_sources if hasattr(s, 'get_source_type')]
                print(f"   Agent sources: {len(self.agent_sources)} ({', '.join(source_types) if source_types else 'unknown'})")
            elif self.dynamic_agents:
                print(f"   Dynamic agents: ✅ Enabled (legacy)")
            else:
                print(f"   Dynamic agents: ❌ Disabled")
            
            print(f"   File watching: {'✅ Enabled' if self.watcher else '❌ Disabled'}")
            print(f"   Cron scheduler: {'✅ Enabled' if self.cron else '❌ Disabled'}")
            print(f"   Monitoring: {'✅ Enabled' if self.monitoring else '❌ Disabled'}")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Server shutdown event"""
            # Stop agent manager if enabled
            if self.manager:
                await self.manager.stop_all()
    
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
    # Daemon/Plugin configuration
    enable_file_watching: bool = False,
    watch_dirs: Optional[List[Path]] = None,
    enable_cron: bool = False,
    plugin_config: Optional[Dict[str, Any]] = None,
    storage_backend: str = "json",
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
        enable_file_watching: Enable file watching for AGENT*.md files
        watch_dirs: Directories to watch for agent files
        enable_cron: Enable cron scheduler for scheduled agent runs
        plugin_config: Plugin configuration dict
        storage_backend: Storage backend ("json" or "litesql")
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
        enable_file_watching=enable_file_watching,
        watch_dirs=watch_dirs,
        enable_cron=enable_cron,
        plugin_config=plugin_config,
        storage_backend=storage_backend,
        **kwargs
    ) 