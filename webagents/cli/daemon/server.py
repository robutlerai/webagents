"""
WebAgents Daemon Server

FastAPI-based daemon for managing local agents.
"""

import asyncio
import logging
import signal
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse
import uvicorn

from .manager import AgentManager
from .registry import DaemonRegistry
from .cron import CronScheduler
from .watcher import FileWatcher
from .resolver import local_agent_resolver
from ..loader import AgentFile

# Configure logging for webagents modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
# Enable debug for MCP and manager during development
logging.getLogger("webagents.skills.mcp").setLevel(logging.DEBUG)
logging.getLogger("webagentsd.manager").setLevel(logging.DEBUG)
logging.getLogger("webagents.agents").setLevel(logging.INFO)


class WebAgentsDaemon:
    """WebAgents daemon server.
    
    Responsibilities:
    - Manage agent registry
    - Watch for AGENT*.md file changes
    - Schedule cron jobs
    - Expose agents via HTTP
    """
    
    def __init__(
        self,
        port: int = 8765,
        watch_dirs: Optional[List[Path]] = None,
        url_prefix: str = "/agents",
    ):
        """Initialize daemon.
        
        Args:
            port: HTTP port to listen on
            watch_dirs: Directories to watch for agent files
            url_prefix: URL prefix for agent routes (default: "/agents")
        """
        self.port = port
        self.watch_dirs = watch_dirs or [Path.cwd()]
        self.url_prefix = url_prefix.rstrip("/") if url_prefix else ""
        
        # Components
        self.registry = DaemonRegistry()
        self.manager = AgentManager(self.registry)
        self.cron = CronScheduler(self.manager)
        
        # Initialize watcher with callback
        self.watcher = FileWatcher(
            registry=self.registry,
            watch_dirs=self.watch_dirs,
            on_change=self._handle_file_change
        )
        
        # FastAPI app with default settings (redirect_slashes=True handles trailing slashes)
        self.app = FastAPI(
            title="webagentsd",
            description="WebAgents Daemon - Manage local agents",
            version="1.0.0",
        )
        
        # Create router with configurable prefix for agent routes
        self.agents_router = APIRouter(prefix=self.url_prefix)
        
        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        self._setup_routes()
    
    async def _handle_file_change(self, event_type: str, path: Path):
        """Handle file change event from watcher.
        
        Args:
            event_type: modified, created, deleted
            path: Path to changed file
        """
        # 1. Registry is already updated by watcher before calling this callback
        # (FileWatcher implementation calls registry.update_from_file first)
        
        # 2. Sync cron jobs
        self.cron.sync_from_registry(self.registry)
        
        # 3. Invalidate agent cache on file change
        agent = self.registry.find_by_path(path)
        if agent:
            self.manager.invalidate_agent_cache(agent.name)

    def _setup_routes(self):
        """Set up FastAPI routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Start up event."""
            # Clean up any stale PIDs
            pass

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Shutdown event."""
            await self.stop()
        
        # Root-level endpoints (not prefixed)
        @self.app.get("/")
        async def root():
            """Daemon status."""
            return {
                "name": "webagentsd",
                "status": "running",
                "agents": len(self.registry.list_agents()),
                "cron_jobs": len(self.cron.list_jobs()),
                "url_prefix": self.url_prefix,
            }
        
        @self.app.get("/health")
        async def health():
            """Health check."""
            return {"status": "healthy"}
        
        @self.app.get("/cron")
        async def list_cron_jobs():
            """List cron jobs."""
            return {
                "jobs": [j.to_dict() for j in self.cron.list_jobs()]
            }
        
        @self.app.post("/cron")
        async def add_cron_job(agent: str, schedule: str):
            """Add a cron job."""
            job = self.cron.add_job(agent, schedule)
            return job.to_dict()
        
        @self.app.delete("/cron/{job_id}")
        async def remove_cron_job(job_id: str):
            """Remove a cron job."""
            self.cron.remove_job(job_id)
            return {"status": "removed", "job_id": job_id}
        
        @self.app.post("/scan")
        async def scan_agents(path: str = "."):
            """Scan for agent files."""
            count = await self.registry.scan_directory(Path(path))
            return {"scanned": count}
        
        @self.app.get("/logs/{agent}")
        async def get_logs(agent: str, lines: int = 100):
            """Get agent logs."""
            logs = self.manager.get_logs(agent, lines)
            return {"agent": agent, "logs": logs}
        
        # Agent routes on configurable prefix
        # GET /agents - List agents
        @self.agents_router.get("/")
        async def list_agents():
            """List registered agents."""
            return {
                "agents": [
                    a.to_dict() for a in self.registry.list_agents()
                ]
            }
        
        # POST /agents - Register agent
        @self.agents_router.post("/")
        async def register_agent(data: dict):
            """Register an agent from file."""
            path_str = data.get("path")
            if not path_str:
                raise HTTPException(400, "Path required")
            
            try:
                agent_file = AgentFile(Path(path_str))
                agent = self.registry.register(agent_file)
                return agent.to_dict()
            except Exception as e:
                raise HTTPException(400, str(e))
        
        # DELETE /{name} - Unregister agent
        @self.agents_router.delete("/{name}")
        async def unregister_agent(name: str):
            """Unregister an agent."""
            if self.registry.unregister(name):
                return {"status": "unregistered", "name": name}
            raise HTTPException(404, f"Agent not found: {name}")

        # GET /{name} - Get agent details
        @self.agents_router.get("/{name}")
        async def get_agent(name: str):
            """Get agent details."""
            agent = self.registry.get(name)
            if not agent:
                raise HTTPException(404, f"Agent not found: {name}")
            return agent.to_dict()
        
        # GET /{name}/command - List commands
        @self.agents_router.get("/{name}/command")
        async def list_commands(name: str):
            """List available commands for an agent.
            
            Commands are dynamically discovered from agent skills via @command decorator.
            """
            # Get or load the agent
            agent = await self.manager.get_or_load_agent(name)
            if not agent:
                raise HTTPException(404, f"Agent not found: {name}")
            
            commands = agent.list_commands()
            return {"commands": commands}
        
        # POST /{name}/command/{path} - Execute command
        @self.agents_router.post("/{name}/command/{path:path}")
        async def execute_command(name: str, path: str, data: dict = None):
            """Execute a command on an agent.
            
            Commands are exposed by agent skills via @command decorator.
            """
            # Get or load the agent
            agent = await self.manager.get_or_load_agent(name)
            if not agent:
                raise HTTPException(404, f"Agent not found: {name}")
            
            cmd_path = f"/{path}" if not path.startswith("/") else path
            result = await agent.execute_command(cmd_path, data or {})
            return {"result": result}
        
        # GET /{name}/command/{path} - Get command documentation
        @self.agents_router.get("/{name}/command/{path:path}")
        async def get_command_docs(name: str, path: str):
            """Get documentation for a specific command."""
            import traceback
            
            agent = await self.manager.get_or_load_agent(name)
            if not agent:
                raise HTTPException(404, f"Agent not found: {name}")
            
            cmd_path = f"/{path}" if not path.startswith("/") else path
            try:
                command = agent.get_command(cmd_path)
            except Exception as e:
                # Log full traceback for debugging
                import logging
                logging.getLogger("webagentsd").error(f"Error in get_command({cmd_path}): {traceback.format_exc()}")
                raise HTTPException(500, f"Error getting command: {e}")
            
            if not command:
                raise HTTPException(404, f"Command not found: {cmd_path}")
            return command
        
        # POST /{name}/chat/completions - Chat endpoint
        @self.agents_router.post("/{name}/chat/completions")
        async def chat_completions(name: str, request: dict):
            """OpenAI-compatible chat completions endpoint."""
            from fastapi.responses import StreamingResponse
            
            agent = await self.manager.get_or_load_agent(name)
            if not agent:
                raise HTTPException(404, f"Agent not found: {name}")
            
            messages = request.get("messages", [])
            stream = request.get("stream", False)
            model = request.get("model", "gpt-4o-mini")
            
            if stream:
                async def generate():
                    async for chunk in agent.chat_stream(messages, model=model):
                        yield f"data: {chunk}\n\n"
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream"
                )
            else:
                result = await agent.chat(messages, model=model)
                return result
        
        # Include the agents router in the app
        self.app.include_router(self.agents_router)
    
    async def start(self):
        """Start the daemon."""
        self._running = True
        
        # Discover agents
        for watch_dir in self.watch_dirs:
            await self.registry.scan_directory(watch_dir)
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self.cron.run()),
            asyncio.create_task(self.watcher.watch()),
        ]
        
        # Start HTTP server
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self):
        """Stop the daemon."""
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        # Stop all running agents
        await self.manager.stop_all()
    
    def run(self, background: bool = False):
        """Run the daemon.
        
        Args:
            background: Run in background (fork process)
        """
        if background:
            self._run_background()
        else:
            asyncio.run(self.start())
    
    def _run_background(self):
        """Run daemon in background."""
        import os
        import sys
        
        # Fork process
        pid = os.fork()
        if pid > 0:
            # Parent process
            print(f"webagentsd started with PID {pid}")
            sys.exit(0)
        
        # Child process - become daemon
        os.setsid()
        
        # Second fork to prevent zombie processes
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
        
        # Run daemon
        asyncio.run(self.start())


def create_daemon(
    port: int = 8765,
    watch_dirs: Optional[List[Path]] = None,
    url_prefix: str = "/agents",
) -> WebAgentsDaemon:
    """Create a daemon instance.
    
    Args:
        port: HTTP port
        watch_dirs: Directories to watch
        url_prefix: URL prefix for agent routes (default: "/agents")
        
    Returns:
        WebAgentsDaemon instance
    """
    return WebAgentsDaemon(port=port, watch_dirs=watch_dirs, url_prefix=url_prefix)
