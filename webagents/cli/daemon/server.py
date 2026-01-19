"""
WebAgents Daemon Server

FastAPI-based daemon for managing local agents.
"""

import asyncio
import signal
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from .manager import AgentManager
from .registry import DaemonRegistry
from .cron import CronScheduler
from .watcher import FileWatcher
from .resolver import local_agent_resolver


class WebAgentsDaemon:
    """WebAgents daemon server.
    
    Responsibilities:
    - Manage agent lifecycle (start, stop, restart)
    - Watch for AGENT*.md file changes
    - Schedule cron jobs
    - Expose agents via HTTP and WebSocket
    - Handle platform WebSocket connection
    """
    
    def __init__(
        self,
        port: int = 8765,
        watch_dirs: Optional[List[Path]] = None,
    ):
        """Initialize daemon.
        
        Args:
            port: HTTP port to listen on
            watch_dirs: Directories to watch for agent files
        """
        self.port = port
        self.watch_dirs = watch_dirs or [Path.cwd()]
        
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
        
        # FastAPI app
        self.app = FastAPI(
            title="webagentsd",
            description="WebAgents Daemon - Manage local agents",
            version="1.0.0",
        )
        
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
        
        # 3. Hot-reload agent if running
        agent = self.registry.find_by_path(path)
        if agent and agent.name in self.manager.get_running_agents():
            try:
                await self.manager.restart(agent.name)
            except Exception:
                pass

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
        
        @self.app.get("/")
        async def root():
            """Daemon status."""
            return {
                "name": "webagentsd",
                "status": "running",
                "agents": len(self.registry.list_agents()),
                "cron_jobs": len(self.cron.list_jobs()),
            }
        
        @self.app.get("/health")
        async def health():
            """Health check."""
            return {"status": "healthy"}
        
        @self.app.get("/agents")
        async def list_agents():
            """List registered agents."""
            return {
                "agents": [
                    a.to_dict() for a in self.registry.list_agents()
                ]
            }
        
        @self.app.get("/agents/{name}")
        async def get_agent(name: str):
            """Get agent details."""
            agent = self.registry.get(name)
            if not agent:
                raise HTTPException(404, f"Agent not found: {name}")
            return agent.to_dict()
        
        @self.app.post("/agents/{name}/start")
        async def start_agent(name: str):
            """Start an agent."""
            try:
                await self.manager.start(name)
                return {"status": "started", "agent": name}
            except Exception as e:
                raise HTTPException(400, str(e))
        
        @self.app.post("/agents/{name}/stop")
        async def stop_agent(name: str):
            """Stop an agent."""
            try:
                await self.manager.stop(name)
                return {"status": "stopped", "agent": name}
            except Exception as e:
                raise HTTPException(400, str(e))
        
        @self.app.post("/agents/{name}/restart")
        async def restart_agent(name: str):
            """Restart an agent."""
            try:
                await self.manager.restart(name)
                return {"status": "restarted", "agent": name}
            except Exception as e:
                raise HTTPException(400, str(e))
        
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
        
        @self.app.get("/agents/{name}/command")
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
        
        @self.app.post("/agents/{name}/command/{path:path}")
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
) -> WebAgentsDaemon:
    """Create a daemon instance.
    
    Args:
        port: HTTP port
        watch_dirs: Directories to watch
        
    Returns:
        WebAgentsDaemon instance
    """
    return WebAgentsDaemon(port=port, watch_dirs=watch_dirs)
