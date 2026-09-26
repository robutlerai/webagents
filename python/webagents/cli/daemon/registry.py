"""
Daemon Agent Registry

Manage agents registered with the daemon.
"""

import os
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

from ..loader import AgentFile


class DaemonAgent(BaseModel):
    """An agent registered with the daemon."""
    
    # Identity
    name: str
    namespace: str = "local"
    description: str = ""
    
    # Source
    source_path: str
    
    # Discovery
    intents: List[str] = []
    
    # Status
    status: str = "registered"  # registered, running, stopped, error
    started_at: Optional[datetime] = None
    
    # Triggers
    cron: Optional[str] = None
    watch_patterns: List[str] = []
    
    def to_dict(self) -> dict:
        return self.model_dump(mode='json')


#: Directory names that never contain an agent the user means to run.
#:
#: DISCOVERY USED TO DESCEND INTO ALL OF THEM (found 2026-09-23 by the CLI
#: end-to-end run). `scan_directory` globbed `**/AGENT*.md` and the watchdog
#: observer is recursive, so `checkpoint create`, which copies the project into
#: `.webagents/history/`, produced a SECOND `AGENT-demo.md` there. It declared
#: the same `name:`, the registry is keyed by name, and the snapshot replaced
#: the real agent. From the first checkpoint on, the daemon served the agent
#: FROM THE SNAPSHOT: edits to the real file silently stopped taking effect, and
#: `checkpoint list` read the snapshot's empty checkpoint directory and reported
#: none. The bad registration was also persisted and restored on the next start.
#:
#: `.git`, `node_modules` and the virtualenvs are here for the same reason:
#: vendored or generated trees routinely contain files named like agents.
IGNORED_DIRS = frozenset({
    ".webagents",
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
})
# NOT `dist` or `build`: the watcher and the restore path hand over ABSOLUTE
# paths, and a project living at `~/build/myproj/` is a real layout. Every name
# above is a tool's private directory that no one keeps their own work in.


def is_discoverable(path: Path) -> bool:
    """Whether a file may be registered as an agent.

    Two rules. The file must be an agent file BY NAME, which excludes the
    inherited-context file: `WEBAGENTS.md` is watched so an edit to it reloads
    the agents beneath it, but it is not itself an agent, and treating it as one
    registered a phantom agent called `assistant` (the schema's default name)
    for every context file on disk. And no directory on the way to it may be an
    ignored one.
    """
    name = path.name
    if not (name == "AGENT.md" or (name.startswith("AGENT-") and name.endswith(".md"))):
        return False
    return not any(part in IGNORED_DIRS for part in path.parts[:-1])


def discover_agent_files(root: Path, recursive: bool = True) -> List[Path]:
    """The agent files under ``root``, by the names on disk, in walk order.

    Ignored trees are never entered and symlinked directories never followed.
    BY THE NAMES ON DISK (2026-09-25): the scan globbed ``**/AGENT.md``, and on
    a case-insensitive disk (macOS) a glob answers the PATTERN's spelling for a
    file named ``agent.md``, so the daemon served a file that neither the chat
    (``agent_files.folder_agents``) nor the TypeScript daemon counts as an
    agent. The glob also walked all of ``node_modules`` before the filter threw
    it away. This is the TypeScript watcher's walk
    (``typescript/src/daemon/watcher.ts``, ``findAgentFiles``), pinned with it
    by ``tests/fixtures/daemon/discovery.json``.
    """
    found: List[Path] = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in IGNORED_DIRS)
        for name in sorted(files):
            candidate = Path(directory) / name
            if is_discoverable(candidate.relative_to(root)) and candidate.is_file():
                found.append(candidate)
        if not recursive:
            break
    return found


class DaemonRegistry:
    """Registry for daemon-managed agents."""
    
    def __init__(self):
        """Initialize registry."""
        self.agents: Dict[str, DaemonAgent] = {}
    
    def register(self, agent_file: AgentFile) -> DaemonAgent:
        """Register an agent from file.
        
        Args:
            agent_file: Parsed agent file
            
        Returns:
            DaemonAgent
        """
        daemon_agent = DaemonAgent(
            name=agent_file.metadata.name,
            namespace=agent_file.metadata.namespace,
            description=agent_file.metadata.description,
            source_path=str(agent_file.path),
            intents=agent_file.metadata.intents,
            cron=agent_file.metadata.cron,
            watch_patterns=agent_file.metadata.watch or [],
        )
        
        # Two files declaring the same `name:` used to replace each other
        # without a word, which is what made the snapshot bug above invisible.
        # The newer registration still wins, because a moved file is the common
        # case, but it no longer happens silently.
        existing = self.agents.get(daemon_agent.name)
        if existing is not None and existing.source_path != daemon_agent.source_path:
            import logging

            logging.getLogger(__name__).warning(
                "agent %r is declared by two files; %s now replaces %s",
                daemon_agent.name,
                daemon_agent.source_path,
                existing.source_path,
            )

        self.agents[daemon_agent.name] = daemon_agent
        return daemon_agent
    
    def unregister(self, name: str) -> bool:
        """Remove agent from registry.
        
        Args:
            name: Agent name
            
        Returns:
            True if removed
        """
        if name in self.agents:
            del self.agents[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[DaemonAgent]:
        """Get agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            DaemonAgent or None
        """
        return self.agents.get(name)
    
    def find_by_path(self, path: Path) -> Optional[DaemonAgent]:
        """Find agent by source path.
        
        Args:
            path: Path to agent file
            
        Returns:
            DaemonAgent or None
        """
        path_str = str(path)
        for agent in self.agents.values():
            if agent.source_path == path_str:
                return agent
        return None
    
    def list_agents(
        self,
        namespace: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[DaemonAgent]:
        """List agents with optional filters.
        
        Args:
            namespace: Filter by namespace
            status: Filter by status
            
        Returns:
            List of agents
        """
        agents = list(self.agents.values())
        
        if namespace:
            agents = [a for a in agents if a.namespace == namespace]
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        return agents
    
    def list_agent_names(self) -> List[str]:
        """Get all agent names."""
        return list(self.agents.keys())
    
    async def scan_directory(self, path: Path, recursive: bool = True) -> int:
        """Scan directory and register agents.
        
        Args:
            path: Directory to scan
            recursive: Include subdirectories
            
        Returns:
            Number of agents registered
        """
        count = 0
        # Relative to the scan root inside `discover_agent_files`, so a project
        # that itself lives under a directory named `build` is not excluded
        # wholesale.
        for agent_path in discover_agent_files(path, recursive):
            try:
                agent_file = AgentFile(agent_path)
                self.register(agent_file)
                count += 1
            except Exception:
                pass

        return count
    
    def update_from_file(self, path: Path):
        """Update agent from file (on file change).
        
        Args:
            path: Path to changed file
        """
        # The watcher calls this for context-file events too, and the restore
        # path calls it for whatever was persisted, including any snapshot copy
        # an earlier daemon registered by mistake. One gate for both.
        if not is_discoverable(Path(path)):
            return None

        try:
            agent_file = AgentFile(path)
            
            # Check if already registered
            existing = self.find_by_path(path)
            if existing:
                # Update existing agent
                existing.name = agent_file.metadata.name
                existing.namespace = agent_file.metadata.namespace
                existing.description = agent_file.metadata.description
                existing.intents = agent_file.metadata.intents
                existing.cron = agent_file.metadata.cron
                existing.watch_patterns = agent_file.metadata.watch or []
                return existing
            else:
                # Register new agent
                return self.register(agent_file)
        except Exception:
            raise
    
    def get_agents_with_cron(self) -> List[DaemonAgent]:
        """Get agents with cron schedules."""
        return [a for a in self.agents.values() if a.cron]
    
    def get_agents_with_watch(self) -> List[DaemonAgent]:
        """Get agents with file watch patterns."""
        return [a for a in self.agents.values() if a.watch_patterns]
