"""
Auto-start daemon logic

Ensures webagentsd is running, starts it if needed.
"""

import asyncio
import subprocess
from typing import Optional, List
from pathlib import Path

from .daemon_client import DaemonClient


async def ensure_daemon_running(
    port: int = 8765,
    watch_dirs: Optional[List[Path]] = None
) -> DaemonClient:
    """Ensure daemon is running, start if needed
    
    Args:
        port: Port number for daemon
        watch_dirs: Directories to watch for agent files
    
    Returns:
        Connected DaemonClient instance
    
    Raises:
        RuntimeError: If daemon fails to start
    """
    client = DaemonClient(f"http://localhost:{port}")
    
    # Check if already running
    if await client.is_running():
        return client
    
    # Start daemon in background with watch/reload
    watch_args = []
    if watch_dirs:
        for dir_path in watch_dirs:
            watch_args.extend(["--watch", str(dir_path)])

    import sys
    subprocess.Popen(
        [sys.executable, "-m", "webagents", "daemon", "start", "--background", "--port", str(port), *watch_args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    # Wait for daemon to start (max 5 seconds)
    for _ in range(50):
        await asyncio.sleep(0.1)
        if await client.is_running():
            return client
    
    raise RuntimeError("Failed to start webagentsd")
