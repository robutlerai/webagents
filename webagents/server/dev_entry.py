"""
Dev entry point for webagentsd with auto-reload support.
This file is used by uvicorn for --reload.
"""
from webagents.server.core.app import create_server
from pathlib import Path
import os

# Get watch dirs from env or default to cwd
watch_dirs_str = os.environ.get("WEBAGENTS_WATCH_DIRS")
watch_dirs = [Path(d) for d in watch_dirs_str.split(",")] if watch_dirs_str else [Path.cwd()]

# Create app instance with file watching enabled
# This module is imported by uvicorn workers
server = create_server(
    title="WebAgents Daemon (Dev)",
    description="Local agent daemon with auto-reload",
    version="0.2.3",
    enable_file_watching=True,
    watch_dirs=watch_dirs,
    enable_cron=True,
    storage_backend="json"
)

app = server.app
