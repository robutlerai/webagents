"""
File Watcher

Watch for AGENT*.md and AGENTS.md file changes.
"""

import asyncio
from typing import List, Optional, Callable
from pathlib import Path
from datetime import datetime

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent


class AgentFileHandler(FileSystemEventHandler):
    """Handle agent file system events."""
    
    def __init__(self, callback: Callable[[str, Path], None]):
        """Initialize handler.
        
        Args:
            callback: Function to call on events (event_type, path)
        """
        super().__init__()
        self.callback = callback
    
    def _is_agent_file(self, path: str) -> bool:
        """Check if path is an agent file."""
        name = Path(path).name
        return (
            name == "AGENT.md" or
            name.startswith("AGENT-") and name.endswith(".md") or
            name == "AGENTS.md"
        )
    
    def on_modified(self, event):
        if not event.is_directory and self._is_agent_file(event.src_path):
            self.callback("modified", Path(event.src_path))
    
    def on_created(self, event):
        if not event.is_directory and self._is_agent_file(event.src_path):
            self.callback("created", Path(event.src_path))
    
    def on_deleted(self, event):
        if not event.is_directory and self._is_agent_file(event.src_path):
            self.callback("deleted", Path(event.src_path))


class FileWatcher:
    """Watch directories for agent file changes."""
    
    def __init__(
        self,
        registry,
        watch_dirs: List[Path],
        on_change: Optional[Callable] = None,
    ):
        """Initialize watcher.
        
        Args:
            registry: DaemonRegistry to update
            watch_dirs: Directories to watch
            on_change: Optional callback for changes
        """
        self.registry = registry
        self.watch_dirs = watch_dirs
        self.on_change = on_change
        
        self._observer: Optional[Observer] = None
        self._running = False
        self._event_queue: asyncio.Queue = None
    
    def _handle_event(self, event_type: str, path: Path):
        """Handle file system event.
        
        Args:
            event_type: modified, created, deleted
            path: Path to changed file
        """
        if self._event_queue:
            # Put in async queue for processing
            try:
                self._event_queue.put_nowait((event_type, path))
            except asyncio.QueueFull:
                pass
    
    async def watch(self):
        """Start watching for changes."""
        self._running = True
        self._event_queue = asyncio.Queue(maxsize=100)
        
        # Start watchdog observer
        self._observer = Observer()
        handler = AgentFileHandler(self._handle_event)
        
        for watch_dir in self.watch_dirs:
            self._observer.schedule(handler, str(watch_dir), recursive=True)
        
        self._observer.start()
        
        try:
            while self._running:
                try:
                    # Process events from queue
                    event_type, path = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0
                    )
                    
                    await self._process_event(event_type, path)
                    
                except asyncio.TimeoutError:
                    continue
        finally:
            self._observer.stop()
            self._observer.join()
    
    async def _process_event(self, event_type: str, path: Path):
        """Process a file event.
        
        Args:
            event_type: Event type
            path: File path
        """
        if event_type == "deleted":
            # Remove from registry
            agent = self.registry.find_by_path(path)
            if agent:
                self.registry.unregister(agent.name)
        else:
            # Create or update
            self.registry.update_from_file(path)
        
        # Call optional callback
        if self.on_change:
            await self.on_change(event_type, path)
    
    def stop(self):
        """Stop watching."""
        self._running = False
        if self._observer:
            self._observer.stop()


class FileWatchTrigger:
    """Watch files and trigger agent execution."""
    
    def __init__(self, agent_manager, patterns: List[str]):
        """Initialize trigger.
        
        Args:
            agent_manager: AgentManager for running agents
            patterns: File patterns to watch
        """
        self.manager = agent_manager
        self.patterns = patterns
        self._observer: Optional[Observer] = None
    
    def matches(self, path: str) -> bool:
        """Check if path matches any pattern."""
        from fnmatch import fnmatch
        name = Path(path).name
        return any(fnmatch(name, p) for p in self.patterns)
    
    async def start(self, watch_dir: Path, agent_name: str):
        """Start watching and triggering.
        
        Args:
            watch_dir: Directory to watch
            agent_name: Agent to trigger on match
        """
        # TODO: Implement file watch triggering
        pass
