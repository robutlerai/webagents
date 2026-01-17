"""
Sandbox Skill

Provides an isolated environment for executing commands using Docker containers.
Mounts the agent's directory to /workspace for persistent access to agent files
while preventing access to the host system.
"""

import subprocess
import uuid
import shutil
import asyncio
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from ...base import Skill
from webagents.agents.tools.decorators import tool


class SandboxSkill(Skill):
    """
    Docker-based execution sandbox.
    
    Provides a secure environment for running commands isolated from the host system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.agent_name = config.get("agent_name", "unknown")
        self.agent_path = config.get("agent_path")
        
        # Default to webagents-sandbox if available, else python:3.11-slim
        default_image = "python:3.11-slim"
        # Check if webagents-sandbox exists (optimization: could cache this check)
        try:
            # Quick check if custom image exists locally
            res = subprocess.run(
                ["docker", "image", "inspect", "webagents-sandbox"], 
                capture_output=True, text=True
            )
            if res.returncode == 0:
                default_image = "webagents-sandbox"
        except Exception:
            pass
            
        self.image = config.get("image", default_image)
        
        # Unique container name per session to avoid conflicts
        self.container_name = f"webagents-sandbox-{self.agent_name}-{str(uuid.uuid4())[:8]}"
        self.is_running = False
        
        # Determine mount path (Agent's directory)
        self.mount_path = None
        if self.agent_path:
             self.mount_path = str(Path(self.agent_path).parent.resolve())
        elif config.get("base_dir"):
             self.mount_path = str(Path(config["base_dir"]).resolve())

    async def ensure_started(self):
        """Public method to ensure container is running."""
        await self._ensure_container()

    def get_container_name(self) -> str:
        """Get the active container name."""
        return self.container_name
    
    def map_path(self, path: Union[str, Path]) -> str:
        """Map a host path to a container path if within mount."""
        if not self.mount_path:
             return str(path) # No mapping possible
        
        try:
            path_obj = Path(path).resolve()
            mount_obj = Path(self.mount_path).resolve()
            
            # Check if path is inside the mount
            if mount_obj in path_obj.parents or path_obj == mount_obj:
                rel = path_obj.relative_to(mount_obj)
                # Force forward slashes for Docker/Linux
                return f"/workspace/{rel.as_posix()}"
        except Exception:
            pass
            
        return str(path)

    async def _ensure_container(self):
        """Ensure the sandbox container is running."""
        # Check if docker is installed
        if not shutil.which("docker"):
             raise RuntimeError("Docker CLI not found. Please install Docker.")

        if self.is_running:
            # Verify it's actually running
            res = subprocess.run(
                ["docker", "container", "inspect", "-f", "{{.State.Running}}", self.container_name],
                capture_output=True, text=True
            )
            if res.returncode == 0 and "true" in res.stdout.strip().lower():
                return
            self.is_running = False
            # Clean up if it exists but stopped
            subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)

        # Start container
        # -d: detach
        # --rm: remove when stopped
        # -v: mount agent dir to /workspace
        # -w: set working dir
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "--rm",
            "-w", "/workspace", # Always default to /workspace
        ]
        
        if self.mount_path:
            cmd.extend(["-v", f"{self.mount_path}:/workspace"])
        
        # Keep alive command
        cmd.extend([self.image, "sleep", "infinity"])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self.is_running = True
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Failed to start sandbox container: {stderr}")

    async def run_sandbox_command(self, command: str) -> str:
        """Run a shell command inside the secure Docker sandbox.
        
        The sandbox is isolated from the host system but has access to files
        in the agent's directory (mounted at /workspace).
        
        Args:
            command: The shell command to run (e.g., "ls -la", "python script.py").
            
        Returns:
            Command output (stdout + stderr).
        """
        try:
            await self._ensure_container()
            
            # Run command via exec
            # wrap in sh -c to support pipes/redirects
            exec_cmd = [
                "docker", "exec",
                self.container_name,
                "sh", "-c", command
            ]
            
            # Run with timeout
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nStderr: {result.stderr}"
                
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
                
            return output if output else "(No output)"
            
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 60 seconds."
        except Exception as e:
            return f"Sandbox error: {str(e)}"

    @tool
    async def reset_sandbox(self) -> str:
        """Restart the sandbox container to clear any temporary state.
        
        Note: Files in the agent's directory (mounted at /workspace) are PERSISTENT.
        Only system changes (installed packages, /tmp files) are cleared.
        """
        if self.is_running:
            subprocess.run(["docker", "stop", self.container_name], capture_output=True)
            self.is_running = False
            
        try:
            await self._ensure_container()
            return "Sandbox container restarted successfully."
        except Exception as e:
            return f"Failed to restart sandbox: {e}"

    async def cleanup(self):
        """Stop the container on shutdown."""
        if self.is_running:
            subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True
            )
            self.is_running = False
