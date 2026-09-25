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
import re

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from ..monitoring import CONTENT_TYPE_LATEST

from .models import (
    ChatCompletionRequest, ChatCompletionResponse, AgentInfoResponse, 
    HealthResponse, AgentListResponse, ServerStatsResponse,
    RegisterAgentRequest
)
from .middleware import RequestLoggingMiddleware, RateLimitMiddleware, RateLimitRule, WorkingDirMiddleware
from .origin_policy import (
    CorsSetting,
    WebSocketOriginGuard,
    agent_verifies_credentials,
    cors_middleware_kwargs,
    origin_allowed,
)
# `CREDENTIAL_HEADERS`, `UNAUTHORIZED_MESSAGE` and `has_credential` are imported
# but not called here: they are re-exports, because `webagents.server.core.app`
# is where callers and tests have always imported them from. The floor itself is
# installed once, in `_setup_middleware`.
from .error_reply import is_meant_to_be_shown, reply_text
from . import endpoint_gate
from .credential_floor import (  # noqa: F401
    BILLABLE_PATHS,
    CREDENTIAL_HEADERS,
    UNAUTHORIZED_MESSAGE,
    has_credential,
    install_credential_floor,
)
from .registration import (
    HEARTBEAT_INTERVAL_S,
    build_agent_card,
    compose_principal,
    resolve_agent_token,
    resolve_portal_api_url,
    resolve_public_base_url,
    run_heartbeat_loop,
)
from ..monitoring import initialize_monitoring
from ..context.context_vars import Context, set_context, create_context, get_context
from ...agents.core.base_agent import BaseAgent
from ...utils.logging import get_logger
from ..extensions.interface import AgentSource, WebAgentsExtension, WebAgentsPlugin


#: The credential floor lives in ONE module now — see
#: `webagents/server/core/credential_floor.py` for why (four doors to the same
#: billable endpoint were found across three rounds, every one of them a route
#: someone forgot to add an `if` to). These names are re-exported here because
#: `app` is where callers and tests have always imported them from.
COMPLETIONS_PATHS = BILLABLE_PATHS  #: Deprecated alias kept for importers.


def openai_completion_body(result: Any) -> Any:
    """An OpenAI chat completion for a finished run: the shape and key order the
    OpenAI API returns, and the TypeScript server's answer byte for byte
    (`server/handler.ts`, `completionBody`, 2026-09-25). The run's own dict
    carried the client library's key order and `*_tokens_details: null`
    fields; anything that is not a completion passes through untouched."""
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    if not isinstance(result, dict) or result.get("object") != "chat.completion":
        return result
    choices = []
    for index, choice in enumerate(result.get("choices") or []):
        message = (choice or {}).get("message") or {}
        shaped: Dict[str, Any] = {"role": message.get("role") or "assistant", "content": message.get("content")}
        if message.get("tool_calls"):
            shaped["tool_calls"] = message["tool_calls"]
        choices.append({"index": choice.get("index", index), "message": shaped, "finish_reason": choice.get("finish_reason")})
    body: Dict[str, Any] = {
        "id": result.get("id"),
        "object": "chat.completion",
        "created": result.get("created"),
        "model": result.get("model"),
        "choices": choices,
    }
    usage = result.get("usage")
    if isinstance(usage, dict):
        body["usage"] = {name: usage.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return body


def attach_request_metadata(body_data: Any) -> None:
    """Put a completions request body's `metadata` on the current Context.

    THIS IS THE SENDER ATTRIBUTION PATH. The platform router posts
    `metadata: {chat_id, chat_type, platform, sender}` with every turn it
    relays, and `sender` is the only thing naming the PERSON: the bearer is a
    service token whose `sub` is `service:robutler-router`. `AuthSkill`
    (`_extract_platform_sender_id`) reads `metadata.sender.id` off the context
    to decide USER vs OWNER scope — and nothing populated it, so the scope was
    permanently USER and every `scope="owner"`/`"admin"` tool was unreachable
    on every platform-routed call.

    Written to BOTH `context.metadata` and `context.custom_data['metadata']`
    because the reader accepts either. Best-effort: never raises into a
    request.
    """
    if not isinstance(body_data, dict):
        return
    metadata = body_data.get("metadata")
    if not isinstance(metadata, dict):
        return
    try:
        ctx = get_context()
        if ctx is None:
            return
        setattr(ctx, "metadata", metadata)
        ctx.set("metadata", metadata)
    except Exception:
        pass


_PATH_PARAM = re.compile(r"\{([^}:]+)(?::([^}]+))?\}")


def _path_matcher(subpath: str):
    """`(regex, names)` matching a handler's subpath with its `{name}` and
    `{name:path}` parameters, as FastAPI reads the same path on a static
    agent (2026-09-25). `{name}` is one segment and `{name:path}` the rest of
    the path; until then a converter was read as part of the name, so a
    `{rest:path}` endpoint matched one segment and was called with a
    parameter named `rest:path`."""
    sp = subpath.lstrip("/")
    names: List[str] = []
    pattern = ""
    last = 0
    for match in _PATH_PARAM.finditer(sp):
        pattern += re.escape(sp[last:match.start()])
        names.append(match.group(1))
        pattern += "(.*)" if match.group(2) == "path" else "([^/]+)"
        last = match.end()
    pattern += re.escape(sp[last:])
    return re.compile(f"^{pattern}$"), names


def _bearer_from_query(connection: Any) -> None:
    """Read a websocket upgrade's `?token` as its bearer credential when it
    carries no Authorization header: a browser cannot set headers on an
    upgrade. Only for a scoped handler (S-242/S-243)."""
    token = connection.query_params.get("token")
    if not token or connection.headers.get("authorization"):
        return
    from starlette.datastructures import MutableHeaders

    headers = MutableHeaders(scope=connection.scope)
    headers.append("authorization", f"Bearer {token}")
    connection._headers = headers


async def _refuse_upgrade(websocket: WebSocket, refusal) -> None:
    """Refuse a websocket upgrade with the gate's status and JSON body, as the
    TypeScript server does; a server that cannot answer an upgrade with a
    body refuses it with a close, which the client sees as a 403."""
    status, body = refusal
    try:
        await websocket.send_denial_response(JSONResponse(status_code=status, content=body))
    except RuntimeError:
        message = str(body.get("error", {}).get("message", "Refused"))
        await websocket.close(code=1008, reason=message[:120])


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
        cors_origins: "CorsSetting" = None,
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
        storage_backend: str = "json",
        # Platform registration (see server/core/registration.py). ON by
        # default: these are registration REQUIREMENTS, not extras, and a
        # server that omits them serves an agent the platform can never
        # finish registering. Opt out only for a deployment that registers
        # some other way.
        agent_card: bool = True,
        heartbeat: bool = True,
        public_url: Optional[str] = None,
        keys_dir: Optional[str] = None,
        # What a caller is told when a run fails (S-228, `error_reply.py`).
        error_detail: bool = False,
        quiet: bool = False,
    ):
        """
        Initialize WebAgents server
        
        Args:
            agents: List of static Agent instances (optional)
            dynamic_agents: Optional function (sync or async) that takes agent_name: str and returns 
                           BaseAgent or Optional[BaseAgent]. Server does not manage how this works internally.
            enable_cors: Whether to enable CORS middleware at all
            cors_origins: Which browser origins may call the agents. None (default):
                any origin when every agent has an AuthSkill, loopback origins
                otherwise (S-224, see `origin_policy.py`). "*" or True for any
                origin, a list for exactly those, False for none.
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
            agent_card: Serve the self-naming agent card at
                        /{agent}/.well-known/agent.json and the key set
                        (Ed25519 signing key first) at
                        /{agent}/.well-known/jwks.json and at the origin.
                        Default True: platform registration reads both.
            heartbeat: POST /api/agents/heartbeat every 60s when a per-agent
                       token (WEBAGENTS_AGENT_TOKEN) and a portal API URL
                       (ROBUTLER_API_URL) are configured. Default True.
            public_url: The URL this server is reachable at; goes on the agent
                        card. Falls back to WEBAGENTS_PUBLIC_URL.
            keys_dir: Where the agent's signing key is persisted. Falls back to
                      ~/.webagents/keys. The key MUST survive restarts:
                      registration pins the card's public key.
            error_detail: Answer a failed run with the exception's own text.
                      Default False: callers get a fixed message and a
                      reference, and the error goes to this server's log
                      (S-228). Only the local daemon on loopback sets it, for
                      the developer's own terminal; never set it on a server
                      anyone else can reach.
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
        
        # A server's log is its output: INFO, as it always was, unless the
        # application or the CLI configured logging itself. Importing the SDK
        # as a library is quiet by default now (`utils/logging.py`).
        from ...utils.logging import logging_explicitly_configured, setup_logging

        if not logging_explicitly_configured():
            setup_logging(level="INFO")

        # Store agents by name for quick lookup
        self.static_agents = {agent.name: agent for agent in (agents or [])}
        
        # Store dynamic agent resolver (server doesn't manage how it works)
        self.dynamic_agents = dynamic_agents

        # The origin rule (S-224). Settled once, here: agents resolved later
        # (`dynamic_agents`, the daemon's file-loaded agents) cannot be known
        # yet, so a server with any of them gets the loopback-only default.
        self.cors_origins = False if not enable_cors else cors_origins
        self._verifies_credentials = bool(self.static_agents) and dynamic_agents is None and all(
            agent_verifies_credentials(agent) for agent in self.static_agents.values()
        )
        
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
        
        # Platform registration configuration
        self.agent_card_enabled = agent_card
        self.heartbeat_enabled = heartbeat
        self.public_url = public_url
        self.keys_dir = keys_dir
        self.error_detail = error_detail
        #: No startup banner and no per-route lines: `webagents serve` and
        #: `webagents daemon` say what matters in the TypeScript CLI's words.
        self.quiet = quiet
        self._heartbeat_tasks: List[Any] = []
        # Which agents already beat, so registration's handoff never starts a
        # second heartbeat for one (`register_after_startup`).
        self._heartbeat_agents: set = set()
        # Per static agent, the JWKSManager whose key set this server serves
        # and the principal it is served under: what `_start_portal_connect_skills`
        # hands a PortalConnectSkill so it can sign the `/ws` handshake
        # (2026-09-23). Filled by `_create_registration_endpoints`.
        self._agent_signers: Dict[str, Tuple[Any, str]] = {}

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

        # THE CREDENTIAL FLOOR — the single chokepoint in front of every
        # billable endpoint on this server. It is added FIRST so it ends up
        # INSIDE the CORS middleware (Starlette's `add_middleware` inserts at
        # the head of the list, so the first one added is the innermost): a
        # browser must be able to read the 401, and a CORS preflight must not
        # be refused by a floor it was never meant to reach.
        #
        # Everything else about it — why one ASGI middleware instead of three
        # per-route `if`s, and what it does and does not cover — is documented
        # in `credential_floor.py`.
        install_credential_floor(self.app)

        # CORS middleware, by the origin rule rather than "*" for everyone
        # (S-224): a server that cannot verify its callers answers loopback
        # origins only. WebSocket handshakes, which CORS never covered, get the
        # same decision from `WebSocketOriginGuard`.
        cors_kwargs = cors_middleware_kwargs(self.cors_origins, self._verifies_credentials)
        if cors_kwargs is not None:
            self.app.add_middleware(CORSMiddleware, **cors_kwargs)
        self.app.add_middleware(
            WebSocketOriginGuard,
            allowed=lambda origin: origin_allowed(self.cors_origins, self._verifies_credentials, origin),
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
        
        # Platform registration surface. Registered BEFORE the agents' own
        # @http handlers so the server's key set is what answers
        # `/{agent}/.well-known/jwks.json` (2026-09-18, W2 review). Until then
        # it came after, "so an explicit AuthSkill handler still wins", and
        # that was the defect: the local AuthSkill mounts its own handler at
        # that path, its JWKSManager starts EMPTY and is filled only by the
        # skill's lazy initialize() (a chat request; and portal mode never
        # loads an Ed25519 key at all), so a fresh agent served {"keys": []}
        # at the very URL `register_with_platform` names in Signature-Agent,
        # the platform answered key_set_invalid and cached it for 300 s. The
        # signer reads the server's `keys_dir` under the agent's name; only
        # the server's own JWKSManager is built from the same two facts, so
        # only the server may answer that URL. Still before the dynamic
        # catch-all below so the card is not swallowed by it.
        if self.agent_card_enabled:
            self._create_registration_endpoints()

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
                    raise HTTPException(
                        status_code=500,
                        detail=self._reply_text(e, f"{agent_name} health", prefix="Agent health check failed: "),
                    )

            @self.router.get("/{agent_name}/command")
            async def dynamic_list_commands(agent_name: str, request: Request):
                """List available commands for a dynamic agent.
                
                Commands are dynamically discovered from agent skills via @command decorator.
                """
                # Get working_dir from request state (set by WorkingDirMiddleware)
                working_dir = getattr(request.state, 'working_dir', None)
                agent = await self._resolve_agent(agent_name, is_dynamic=True, working_dir=working_dir)
                return self._list_commands_reply(agent)
            
            @self.router.post("/{agent_name}/command/{path:path}")
            async def dynamic_execute_command(agent_name: str, path: str, request: Request):
                """Execute a command on a dynamic agent (`_execute_command_request`)."""
                self._check_command_request(request)
                working_dir = getattr(request.state, 'working_dir', None)
                agent = await self._resolve_agent(agent_name, is_dynamic=True, working_dir=working_dir)
                return await self._execute_command_request(agent, path, request)
            
            @self.router.get("/{agent_name}/command/{path:path}")
            async def dynamic_get_command_docs(agent_name: str, path: str):
                """Get documentation for a specific command on a dynamic agent.
                
                Returns command info including path, description, parameters, and completions.
                """
                agent = await self._resolve_agent(agent_name, is_dynamic=True)
                return self._command_docs_reply(agent, path)

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
                
                matched_handler = None
                path_params = {}
                
                for handler_config in ws_handlers:
                    handler_path = handler_config.get('path', '/')
                    regex, param_names = _path_matcher(handler_path)
                    match = regex.match(normalized_path)
                    if match:
                        path_param_values = match.groups()
                        path_params = {name: value for name, value in zip(param_names, path_param_values)}
                        matched_handler = handler_config
                        break
                
                if not matched_handler:
                    await websocket.close(code=4004, reason=f"No WebSocket handler for path: /{ws_path}")
                    return
                
                # WHO MAY OPEN IT (S-242/S-243, 2026-09-25): the one gate
                # (`endpoint_gate.py`). A handler with no scope is open, as it
                # always was. A browser cannot set headers on an upgrade, so a
                # scoped handler reads a `?token` as the bearer credential.
                from ..context.context_vars import create_context, set_context
                ctx = create_context(messages=[], stream=True, agent=agent, request=websocket)
                set_context(ctx)
                handler_scope = matched_handler.get('scope', 'all')
                if not endpoint_gate.is_open(handler_scope):
                    _bearer_from_query(websocket)
                ctx, refusal = await endpoint_gate.admit(agent, handler_scope, ctx)
                if refusal is not None:
                    await _refuse_upgrade(websocket, refusal)
                    return
                set_context(ctx)
                
                # Execute handler
                handler_func = matched_handler.get('function')
                try:
                    await handler_func(websocket, **path_params)
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    try:
                        # A close reason reaches the peer like a body does (S-228).
                        reason = self._reply_text(e, f"{agent_name} websocket /{ws_path}")
                        await websocket.close(code=4000, reason=reason[:120])
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

                # NO CREDENTIAL CHECK HERE, ON PURPOSE. This used to carry its
                # own copy of the floor — door 2 of four, each closed in its own
                # `if` after someone found it. The floor now runs in
                # `CredentialFloorMiddleware`, upstream of routing, so this
                # dispatcher and every other door are the same door from up
                # there. An agent's own `@http` handlers stay public by design;
                # only `BILLABLE_PATHS` is gated, and it is gated once.

                # Get working_dir from request state (set by WorkingDirMiddleware)
                working_dir = getattr(request.state, 'working_dir', None)
                
                # Resolve the dynamic agent
                agent = await self._resolve_agent(agent_name, is_dynamic=True, working_dir=working_dir)
                # Ensure skills are initialized so skill methods have agent context
                try:
                    if hasattr(agent, '_ensure_skills_initialized'):
                        await agent._ensure_skills_initialized()
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=self._reply_text(
                            e, f"{agent_name} skill initialisation", prefix="Failed to initialize agent skills: "
                        ),
                    )

                # Iterate agent's http handlers and find a matching subpath and method
                try:
                    http_handlers = agent.get_all_http_handlers()
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=self._reply_text(
                            e, f"{agent_name} HTTP handlers", prefix="Failed to load HTTP handlers: "
                        ),
                    )

                request_method = request.method.lower()
                # Normalize request_path (no leading slash)
                normalized_path = request_path.lstrip("/")

                # Attempt to match in declaration order
                for handler_config in http_handlers:
                    if handler_config.get('source') == 'builtin':
                        # The command endpoints: only the guarded command
                        # routes above serve them (S-235), never this door.
                        continue
                    try:
                        subpath = handler_config.get('subpath') or ''
                        method = (handler_config.get('method') or 'get').lower()
                        handler_func = handler_config.get('function')
                        handler_scope = handler_config.get('scope', 'all')

                        if method != request_method:
                            continue

                        regex, param_names = _path_matcher(subpath)
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
                            # Sender attribution for platform-routed turns.
                            attach_request_metadata(body_data)
                        except Exception as e:
                            # Log auth errors but don't fail the request
                            import logging
                            logging.getLogger('webagents.server').debug(f"Context creation error: {e}")

                        # WHO MAY CALL IT (S-243, 2026-09-25): the one gate
                        # (`endpoint_gate.py`). A handler with no scope is
                        # open, as it always was. This used to import a module
                        # that does not exist, so every scoped handler here
                        # refused everyone but the localhost agent key.
                        gated, refusal = await endpoint_gate.admit(agent, handler_scope, get_context())
                        if refusal is not None:
                            return JSONResponse(status_code=refusal[0], content=refusal[1])
                        if gated is not None:
                            set_context(gated)

                        # Check if handler is an async generator function (for SSE streaming)
                        if _inspect.isasyncgenfunction(handler_func):
                            # Pre-flight: consume first chunk before committing to 200 SSE.
                            # This lets us surface PaymentError (402) and similar domain errors
                            # as proper HTTP status codes instead of crashing inside the stream.
                            gen = handler_func(**filtered_params)
                            first_chunk = _SENTINEL = object()
                            try:
                                first_chunk = await gen.__anext__()
                            except StopAsyncIteration:
                                first_chunk = _SENTINEL  # empty generator
                            except Exception as _preflight_err:
                                # If the error carries an HTTP status code (e.g. PaymentError 402),
                                # return it as a proper JSON response so callers can react to the status.
                                _sc = getattr(_preflight_err, 'status_code', None)
                                if _sc and isinstance(_sc, int) and 400 <= _sc < 600:
                                    if is_meant_to_be_shown(_preflight_err):
                                        # This SDK's payment and auth errors: written to be shown.
                                        _body = _preflight_err.to_dict() if hasattr(_preflight_err, 'to_dict') else {'error': str(_preflight_err)}
                                    else:
                                        # A provider's own status error (an OpenAI 401 quotes the
                                        # rejected key): the status stays, its text does not (S-228).
                                        _body = {'error': self._reply_text(
                                            _preflight_err, f"{agent_name} {request.method} /{normalized_path}"
                                        )}
                                    from starlette.responses import JSONResponse as _JSONResponse
                                    return _JSONResponse(status_code=_sc, content=_body)
                                raise  # re-raise non-HTTP errors normally

                            # A FINISHED RESPONSE, NOT A STREAM (2026-09-24). One
                            # async-generator handler serves both modes of an
                            # OpenAI-compatible route, and whether a request streams
                            # is decided by its body, not by the handler's type. So
                            # a handler may yield a Response as its FIRST item to
                            # mean "answer with exactly this". Without it every
                            # `stream: false` chat with a daemon-served agent came
                            # back as `text/event-stream`, one `data:` event and a
                            # `[DONE]`, which no OpenAI client parses as a
                            # completion. Checked after the preflight above, so a
                            # PaymentError still becomes a 402 in both modes.
                            if isinstance(first_chunk, Response):
                                await gen.aclose()
                                return first_chunk

                            def _format_sse_chunk(chunk):
                                if isinstance(chunk, str):
                                    return chunk
                                elif isinstance(chunk, dict) and chunk.get('type'):
                                    event_type = chunk['type']
                                    return f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                                else:
                                    return f"data: {json.dumps(chunk)}\n\n"

                            async def sse_stream():
                                if first_chunk is not _SENTINEL:
                                    yield _format_sse_chunk(first_chunk)
                                async for chunk in gen:
                                    yield _format_sse_chunk(chunk)
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
                        # A handler matched and failed. This is the route the
                        # daemon's and a deployed agent's `/chat/completions`
                        # take, and it answered with `str(e)`: a model it could
                        # not reach came back as `500 Connection error.` to
                        # whoever called (S-228). Same status, same plain-text
                        # body; the text is the fixed one unless the error is
                        # meant to be shown or this is the local daemon.
                        return Response(
                            status_code=500,
                            content=self._reply_text(e, f"{agent_name} {request.method} /{normalized_path}"),
                        )

                # No matching handler found
                raise HTTPException(status_code=404, detail=f"No HTTP handler for path '/{normalized_path}' and method {request.method} on agent '{agent_name}'")
        
        # Mount WebUI before including router (so /ui routes take priority)
        self._mount_webui()
        
        # Include the router in the main app
        self.app.include_router(self.router)
    
    def _create_registration_endpoints(self):
        """Serve the agent card and the key set the platform's registration
        reads (ADR 0038 step 5, W2 design sections 3.1, 3.3 and 9.2,
        2026-09-17).

        This used to live in a `host()` wrapper, which meant the DOCUMENTED
        server (`create_server` + `uvicorn.run`) served an agent registration
        could never complete. Registration requirements belong to the server.

        Per static agent, under its own prefix, which is the principal
        `{public_url}{url_prefix}/{agent}` the platform registers:

          * `/{agent}/.well-known/jwks.json`: the key set. The Ed25519
            signing key first (the request signer's `keyid` is its
            thumbprint), then the RSA key for the RS256 consumers.
          * `/{agent}/.well-known/agent.json`: the self-naming card,
            `client_id` equal to this very URL, `url` the principal,
            `jwks_uri` the key set. No key material on it.

        There is NO origin-level card: an origin card cannot self-name for an
        agent mounted under a path, and its only purpose was an origin-level
        fallback the platform deleted. The origin-level key set stays, for
        the first static agent, because the RS256 consumers discover keys at
        the origin (`/.well-known/openid-configuration` names it).

        And ONE origin-level signatures directory,
        `/.well-known/http-message-signatures-directory`, listing every static
        agent's Ed25519 keys under the media type a verifier requires
        (2026-09-19, `key_directory.py`): it is what a `legacy-string`
        signer's bare-origin `Signature-Agent` resolves to, and until it was
        served that form failed discovery against this server every time.
        """
        if not self.static_agents:
            return

        from ...crypto.jwks import JWKSManager
        from .key_directory import DIRECTORY_WELL_KNOWN_PATH, key_directory_response

        registered_origin = False
        directory_managers: List[Any] = []
        for agent_name, agent in self.static_agents.items():
            jwks_config: Dict[str, Any] = {}
            if self.keys_dir:
                jwks_config["keys_dir"] = self.keys_dir
            try:
                jwks = JWKSManager(jwks_config)
                jwks.ensure_keys(agent_name)
                jwks.ensure_ed25519_key(agent_name)
            except Exception as e:  # noqa: BLE001 - a card without a key set
                # is still better than no card, and the reason must be visible.
                self.logger.error(
                    f"Could not load a signing key for '{agent_name}': {e}. "
                    "The key set at /.well-known/jwks.json will answer 404, so "
                    "the platform cannot verify this agent's signed requests "
                    "and registration cannot complete."
                )
                jwks = None
            directory_managers.append(jwks)

            principal = compose_principal(
                resolve_public_base_url(self.public_url, agent_name),
                agent_name,
                self.url_prefix,
            )
            if jwks is not None:
                self._agent_signers[agent_name] = (jwks, principal)
                # The agent's own handle on it, for a skill that signs as the
                # agent (the REST tool), as TypeScript's `agent.identity`.
                from ...crypto.identity import AgentSigningIdentity

                agent.signing_identity = AgentSigningIdentity(principal, jwks)

            def _make_card(_agent=agent, _principal=principal):
                async def _card():
                    return build_agent_card(_agent, _principal)
                return _card

            def _make_jwks(_jwks=jwks):
                async def _keys():
                    if _jwks is None:
                        raise HTTPException(status_code=404, detail="No signing key")
                    return _jwks.get_jwks()
                return _keys

            self.router.add_api_route(
                f"/{agent_name}/.well-known/agent.json",
                _make_card(),
                methods=["GET"],
                name=f"agent_card_{agent_name}",
            )
            self.router.add_api_route(
                f"/{agent_name}/.well-known/jwks.json",
                _make_jwks(),
                methods=["GET"],
                name=f"agent_jwks_{agent_name}",
            )

            if not registered_origin:
                registered_origin = True
                self.app.add_api_route(
                    "/.well-known/jwks.json",
                    _make_jwks(),
                    methods=["GET"],
                    name="origin_jwks",
                )

        # No parameters on the handler: FastAPI would read one as a query field.
        async def _directory():
            return key_directory_response(directory_managers)

        self.app.add_api_route(
            DIRECTORY_WELL_KNOWN_PATH,
            _directory,
            methods=["GET"],
            name="origin_signatures_directory",
        )

    async def _start_heartbeats(self) -> None:
        """Beat presence for every static agent, or say why we are not.

        Silence here is the failure mode this exists to prevent: an agent the
        platform lists as 'unknown' looks identical to one that is simply
        idle.
        """
        portal_api_url = resolve_portal_api_url()
        for agent_name, agent in (self.static_agents or {}).items():
            token = resolve_agent_token(agent)
            if not token or not portal_api_url:
                self.logger.info(
                    f"No heartbeat for '{agent_name}': needs the agent's key "
                    "(WEBAGENTS_AGENT_TOKEN, or the one `webagents publish` stored for "
                    "this directory) and ROBUTLER_API_URL. The platform will show this "
                    "agent as unknown until it beats."
                )
                continue
            self._heartbeat_tasks.append(
                asyncio.create_task(
                    run_heartbeat_loop(
                        portal_api_url, token, agent_name, HEARTBEAT_INTERVAL_S
                    )
                )
            )
            self._heartbeat_agents.add(agent_name)
            self.logger.info(
                f"Heartbeat started for '{agent_name}' -> {portal_api_url} "
                f"every {HEARTBEAT_INTERVAL_S}s"
            )

    def _reply_text(self, error: BaseException, where: str, prefix: str = "") -> str:
        """What a caller reads for a failed run: a fixed message and a reference,
        unless the error is meant to be shown or this is the local daemon
        (`error_reply.py`, S-228). The full error is logged either way."""
        return reply_text(error, detail=self.error_detail, where=where, logger=self.logger, prefix=prefix)

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
                raise HTTPException(
                    status_code=500,
                    detail=self._reply_text(e, f"{agent_name} health", prefix="Agent health check failed: "),
                )
        
        @self.router.post(f"/{agent_name}/chat/completions")
        async def chat_completion(request: Request):
            """OpenAI-compatible chat completions endpoint.
            
            Uses agent.run_streaming() for streaming responses.
            Transport skills can override this via the catch-all route.

            AUTH: this endpoint runs the agent's model on the OWNER's credit,
            so a request carrying no credential at all never gets here —
            `CredentialFloorMiddleware` refuses it with 401 before FastAPI has
            resolved a route, and before any body is read. An `AuthSkill` on the
            agent is what actually VERIFIES a presented credential (in the run's
            `on_connection` hook); the floor only makes sure a served port is
            not an open billable endpoint for an agent that has no AuthSkill.

            The check used to be an `if` right here, which is exactly why three
            more doors to this same model call stayed open: see
            `credential_floor.py`.
            """
            try:
                body = await request.json()
            except:
                raise HTTPException(status_code=422, detail="Invalid JSON")
            
            messages = body.get("messages", [])
            # `stream` DEFAULTS TO FALSE, as the OpenAI API specifies (2026-09-23).
            #
            # This defaulted to True. The official OpenAI SDKs OMIT the field
            # unless the caller sets it, so a completely standard
            # `client.chat.completions.create(model=..., messages=...)` against
            # any Python agent got `text/event-stream` back and failed to parse
            # it. The TypeScript server (`src/server/multi.ts`, `if (body.stream)`)
            # has always defaulted to false, so the two SDKs disagreed about the
            # endpoint whose whole promise is OpenAI compatibility. Found by the
            # cross-SDK campaign: a TypeScript agent calling a Python agent got
            # SSE where it expected a completion, and returned 500. Every
            # first-party client here (`cli/client/daemon_client.py`) sends
            # `stream` explicitly, so none of them relied on the old default.
            stream = body.get("stream", False)
            model = body.get("model", "")
            tools = body.get("tools")
            
            # Get the agent
            agent = self.static_agents.get(agent_name)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")

            # One context per request, carrying the platform's request
            # metadata (sender attribution). BaseAgent.run* would create a
            # bare context otherwise, and the auth hook would see no sender.
            ctx = create_context(messages=messages, stream=bool(stream), agent=agent, request=request)
            set_context(ctx)
            attach_request_metadata(body)
            
            # A REFUSAL IS A STATUS, BEFORE ANY STREAM (S-236, 2026-09-25). The
            # auth hook runs first in the turn; its 401/403 (and a payment
            # 402) comes back as that status with the error's own JSON, as on
            # the skill route below, rather than a 500 or an error event
            # inside a 200 stream.
            from starlette.responses import JSONResponse as _JSONResponse

            from .error_reply import is_meant_to_be_shown

            def _refusal(error: Exception):
                status = getattr(error, "status_code", None)
                if isinstance(status, int) and 400 <= status < 600 and is_meant_to_be_shown(error):
                    body = error.to_dict() if hasattr(error, "to_dict") else {"error": str(error)}
                    return _JSONResponse(status_code=status, content=body)
                return None

            if stream:
                import json as _json

                chunks = agent.run_streaming(messages, tools=tools)
                first = None
                first_error: Optional[Exception] = None
                try:
                    first = await chunks.__anext__()
                except StopAsyncIteration:
                    first = None
                except Exception as e:
                    refusal = _refusal(e)
                    if refusal is not None:
                        return refusal
                    first_error = e  # anything else fails inside the stream, as it always has

                async def generate():
                    try:
                        if first_error is not None:
                            raise first_error
                        if first is not None:
                            yield f"data: {_json.dumps(first)}\n\n"
                            async for chunk in chunks:
                                yield f"data: {_json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        # The same `{"error": ...}` event, without the exception's
                        # own text for a remote caller (S-228).
                        message = self._reply_text(e, f"{agent_name} chat/completions")
                        yield f"data: {_json.dumps({'error': message})}\n\n"
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
                )
            else:
                # Non-streaming: collect all chunks
                try:
                    result = await agent.run(messages, tools=tools)
                except Exception as e:
                    refusal = _refusal(e)
                    if refusal is not None:
                        return refusal
                    raise
                return openai_completion_body(result)
        
        # Register HTTP handlers if agent has any
        if not is_dynamic:
            agent = self.static_agents.get(agent_name)
            if agent:
                self._create_command_routes(agent_name)
                self._register_agent_http_handlers(agent_name, agent)
    
    # ----- Commands: one guarded implementation for static and dynamic agents -----

    @staticmethod
    def _check_command_request(request: Request) -> None:
        """What every command request needs before the agent is even looked at.

        A JSON BODY ONLY (S-235, 2026-09-25): a web page can send a
        cross-origin `text/plain` POST with no preflight, and this parsed any
        body as JSON, so any page the person opened could run the local
        daemon's commands. And a credential, like any agent route (the floor
        itself guards only billable paths, and these are not billable).
        """
        from .credential_floor import has_credential

        if not has_credential(request):
            raise HTTPException(status_code=401, detail="Authentication required: commands need a credential in the Authorization header.")
        content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise HTTPException(status_code=415, detail="Commands take a JSON body (application/json).")

    async def _execute_command_request(self, agent: BaseAgent, path: str, request: Request):
        """Run one command for the caller of `request` (static and dynamic agents).

        AN UNKNOWN COMMAND IS A 404, NOT A 500 (2026-09-23): `execute_command`
        raises `ValueError("Command not found: ...")`, which came back as a
        bare 500, so a client could not tell a typo from an outage.

        THE CALLER IS IDENTIFIED for a scoped command (2026-09-25), by the
        agent's auth skills and access block (`endpoint_gate.identify`), and
        the command's scope is checked against who that is. Until then no
        route passed a context, so the caller was always anonymous: an
        `owner` command refused its owner too, over HTTP.
        """
        try:
            data = await request.json()
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        cmd_path = f"/{path}" if not path.startswith("/") else path
        command = agent.get_command(cmd_path, include_completions=False)
        if command is None:
            raise HTTPException(status_code=404, detail=f"Command not found: {cmd_path}")

        context = create_context(messages=[], stream=False, agent=agent, request=request)
        set_context(context)
        context, refusal = await endpoint_gate.identify(agent, command.get("scope", "all"), context)
        if refusal is not None:
            return JSONResponse(status_code=refusal[0], content=refusal[1])
        set_context(context)
        try:
            result = await agent.execute_command(cmd_path, data, context=context)
        except PermissionError as refused:
            # The command's scope is not this caller's (S-235).
            raise HTTPException(status_code=403, detail=str(refused))
        return {"result": result}

    @staticmethod
    def _list_commands_reply(agent: BaseAgent) -> Dict[str, Any]:
        return {"commands": agent.list_commands()}

    @staticmethod
    def _command_docs_reply(agent: BaseAgent, path: str) -> Dict[str, Any]:
        cmd_path = f"/{path}" if not path.startswith("/") else path
        command = agent.get_command(cmd_path)
        if not command:
            raise HTTPException(status_code=404, detail=f"Command not found: {cmd_path}")
        return command

    def _create_command_routes(self, agent_name: str) -> None:
        """A static agent's command routes, the same three the dynamic agents
        have, with the same guards (2026-09-25).

        The agent registers its command endpoints as ordinary `@http` handlers
        at `/command/{path:path}`, and the generic mount below could not read
        the `:path` converter: it printed "'path:path' is not a valid
        parameter name" at every startup, stopped there, and a static agent had
        no way to run a command at all. Reading the converter would have been
        worse: it would have mounted those handlers as plain endpoints, with
        none of the S-235 guards. So the generic mount skips them, and these
        routes serve them instead.
        """

        @self.router.get(f"/{agent_name}/command")
        async def list_commands(request: Request):
            return self._list_commands_reply(self.static_agents[agent_name])

        @self.router.post(f"/{agent_name}/command/{{path:path}}")
        async def execute_command(path: str, request: Request):
            self._check_command_request(request)
            return await self._execute_command_request(self.static_agents[agent_name], path, request)

        @self.router.get(f"/{agent_name}/command/{{path:path}}")
        async def get_command_docs(path: str):
            return self._command_docs_reply(self.static_agents[agent_name], path)

    # ----- An agent's own @http endpoints on a static agent -----

    def _register_agent_http_handlers(self, agent_name: str, agent: BaseAgent):
        """Register agent's HTTP handlers as FastAPI routes with dynamic parameter support.

        THE THIRD DOOR — and the clearest illustration of why the floor is no
        longer written here. A skill's `@http` handler is mounted as its own
        FastAPI route, so it passes through neither the dedicated
        `/{agent}/chat/completions` route nor the dynamic catch-all. For a
        STATIC agent that made `CompletionsTransportSkill`'s
        `@http("/uamp/completions")` an unauthenticated, billable model
        endpoint, and the fix at the time was a fourth copy of the same `if`.

        There is no `if` here now. `CredentialFloorMiddleware` runs above
        routing, so whatever this method mounts — today's handlers and
        tomorrow's — is behind the floor the moment it is mounted, with nobody
        having to remember. See `credential_floor.py`.

        THE ENDPOINT'S SCOPE IS CHECKED (S-243, 2026-09-25) by the same gate as
        the dynamic route (`endpoint_gate.py`); it was read here and never
        checked, so an owner-only endpoint answered anyone. One wrapper now
        serves every path (FastAPI hands it the path parameters, `{name}` and
        `{name:path}` alike), and a handler that cannot be mounted is reported
        by itself instead of stopping every handler after it. The agent's
        command endpoints are not mounted here (`_create_command_routes`).
        """
        for handler_config in agent.get_all_http_handlers():
            if handler_config.get('source') == 'builtin':
                continue
            method = (handler_config.get('method') or 'get').lower()
            subpath = handler_config.get('subpath') or '/'
            full_path = f"/{agent_name}{subpath}"
            try:
                route = getattr(self.router, method, None)
                if route is None:
                    raise ValueError(f"unsupported method {method.upper()}")
                route(full_path, summary=handler_config.get('description', ''))(
                    self._http_endpoint(agent_name, agent, handler_config)
                )
            except Exception as e:
                print(f"⚠️ Could not mount {method.upper()} {full_path} for agent '{agent_name}': {e}")
                continue
            if not self.quiet:
                print(f"📡 Registered HTTP endpoint: {method.upper()} {self.url_prefix}{full_path}")

    def _http_endpoint(self, agent_name: str, agent: BaseAgent, handler_config: Dict[str, Any]):
        """The route function for one `@http` handler of a static agent."""
        handler_func = handler_config['function']
        scope = handler_config.get('scope', 'all')

        async def http_endpoint(request: Request):
            try:
                path_params = dict(request.path_params)
                query_params = dict(request.query_params)
                # A JSON body, else a form (the OpenAI setup form posts one),
                # as on the dynamic route.
                body_data: Dict[str, Any] = {}
                if request.method in ("POST", "PUT", "PATCH"):
                    try:
                        body_data = await request.json()
                    except Exception:
                        try:
                            body_data = dict(await request.form())
                        except Exception:
                            body_data = {}
                    if not isinstance(body_data, dict):
                        body_data = {}

                # A context per request, carrying the platform's request
                # metadata: this is the route the CompletionsTransportSkill's
                # /chat/completions handler is registered on.
                context = create_context(messages=[], stream=False, agent=agent, request=request)
                set_context(context)
                attach_request_metadata(body_data)

                context, refusal = await endpoint_gate.admit(agent, scope, context)
                if refusal is not None:
                    return JSONResponse(status_code=refusal[0], content=refusal[1])
                set_context(context)

                params = {**path_params, **query_params, **body_data}
                # A `?token` is a credential (the dynamic route reads one on
                # localhost), never a handler's argument.
                params.pop('token', None)
                filtered = {
                    name: params[name]
                    for name in inspect.signature(handler_func).parameters
                    if name not in ('self', 'context') and name in params
                }
                if asyncio.iscoroutinefunction(handler_func):
                    return await handler_func(**filtered)
                return handler_func(**filtered)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=self._reply_text(e, f"{agent_name} {request.method} {request.url.path}"),
                )

        http_endpoint.__name__ = f"{agent_name}_{handler_config.get('name', 'handler')}"
        return http_endpoint
    
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
        """Resolve agent by name from all sources (backward compatible wrapper)

        A FAILURE TO BUILD THE AGENT IS AN ANSWER, NOT A BARE 500 (2026-09-24).
        Building an agent constructs its skills, and a skill whose client
        library is missing raises right there: the daemon's built-in robutler
        agent defaulted to a Google model without `google-genai` installed, the
        ImportError escaped every route's handling, and the chat said
        "Internal Server Error" and nothing else. It is a 500 with the reason
        now: the local daemon's own terminal reads that reason, anyone else
        reads a reference (S-228, `error_reply.py`).
        """
        try:
            agent = await self.resolve_agent(agent_name, working_dir=working_dir)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=self._reply_text(e, f"loading agent {agent_name}", prefix=f"Could not load agent '{agent_name}': "),
            ) from None

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

            # Open the reverse WS bridge for every static agent that carries a
            # PortalConnectSkill. The skill also starts itself from
            # initialize(), but a static agent may never be initialized until
            # its first request — and its first request is supposed to ARRIVE
            # over this socket. Starting here is what closes that circle.
            # start() is idempotent.
            await self._start_portal_connect_skills()

            # Presence. Like the agent card, this is a platform registration
            # requirement that used to live in a wrapper.
            if self.heartbeat_enabled:
                await self._start_heartbeats()
            
            # Print server status
            if self.quiet:
                return
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
            for task in self._heartbeat_tasks:
                task.cancel()
            self._heartbeat_tasks.clear()

            # Stop agent manager if enabled
            if self.manager:
                await self.manager.stop_all()

    async def _start_portal_connect_skills(self) -> None:
        """Initialize + start every attached PortalConnectSkill.

        A skill that is initialized and never started looks healthy from every
        observable (process up, /health 200, no errors) and receives nothing —
        the exact failure this lifecycle exists to prevent.

        Which is why a CREDENTIAL or CONFIG error here PROPAGATES and takes
        startup down with it. Logging those and continuing produced precisely
        the state the paragraph above describes: an owner-subject token with no
        agent binding logged one line, the server came up, /health answered
        200, and no socket was ever opened. A misconfigured bridge must be a
        crash, not a log line — the TypeScript half already behaves this way
        (`serve()` lets `PortalCredentialError` escape).

        The broad catch that remains covers TRANSIENT I/O only: the bridge's
        own reconnect loop owns a portal that is merely down, so a failure to
        schedule it is worth a loud log but not a dead process.
        """
        try:
            from ...agents.skills.robutler.portal_connect import (
                PortalConnectSkill,
                PortalConnectConfigError,
                PortalCredentialError,
            )
        except Exception:  # pragma: no cover - optional dependency
            return

        fatal = (PortalCredentialError, PortalConnectConfigError)

        for agent_name, agent in (self.static_agents or {}).items():
            for skill in (getattr(agent, 'skills', None) or {}).values():
                if not isinstance(skill, PortalConnectSkill):
                    continue
                try:
                    # Hand the skill the identity this server serves the key
                    # set for, BEFORE initialize() starts it (2026-09-23):
                    # with no WEBAGENTS_AGENT_TOKEN the skill signs the `/ws`
                    # handshake with this very key, published at
                    # `{principal}/.well-known/jwks.json`, so the stock setup
                    # needs no minted key at all. A configured identity or a
                    # configured token keeps precedence inside the skill.
                    signer = self._agent_signers.get(agent_name)
                    if signer is not None:
                        jwks, principal = signer
                        skill.adopt_identity(jwks.held_ed25519_keys(), principal)
                    if getattr(skill, 'agent', None) is None:
                        await skill.initialize(agent)
                    await skill.start()
                    self.logger.info(
                        f"Portal Connect started for agent '{agent_name}' -> {skill.portal_ws_url}"
                    )
                except fatal as e:
                    self.logger.error(
                        f"Portal Connect cannot start for agent '{agent_name}': {e}"
                    )
                    raise
                except Exception as e:
                    self.logger.error(
                        f"Portal Connect failed to start for agent '{agent_name}': {e}"
                    )
    
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
    agent_card: bool = True,
    heartbeat: bool = True,
    public_url: Optional[str] = None,
    keys_dir: Optional[str] = None,
    error_detail: bool = False,
    quiet: bool = False,
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
        agent_card: Serve the self-naming agent card under the agent prefix
                    and the key set (the Ed25519 signing key first) under the
                    prefix and at the origin. Default True: the platform's
                    registration path reads both, so the plain documented
                    server satisfies registration on its own.
        heartbeat: POST /api/agents/heartbeat every 60s when
                   WEBAGENTS_AGENT_TOKEN and ROBUTLER_API_URL are set.
        public_url: URL this server is reachable at (card `url`); falls back
                    to WEBAGENTS_PUBLIC_URL.
        keys_dir: Where the signing key is persisted (default
                  ~/.webagents/keys). It MUST survive restarts: registration
                  pins the public key from the card.
        error_detail: Answer a failed run with the exception's own text rather
                  than a fixed message and a reference (S-228). Default False;
                  only the local daemon on loopback turns it on.
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
        agent_card=agent_card,
        heartbeat=heartbeat,
        public_url=public_url,
        keys_dir=keys_dir,
        error_detail=error_detail,
        quiet=quiet,
        **kwargs
    ) 