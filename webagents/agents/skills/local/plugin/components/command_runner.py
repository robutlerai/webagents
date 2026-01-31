"""
Command Runner - Plugin Command Execution

Executes plugin commands from the commands/ directory.

Command format:
- Each .py file in commands/ directory is a command
- Commands receive arguments via environment variable PLUGIN_ARGUMENTS (JSON)
- Commands return results via stdout (JSON preferred)
"""

import asyncio
import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Command execution result.
    
    Attributes:
        success: Whether command completed successfully
        output: Command output (parsed JSON or raw string)
        error: Error message if failed
        exit_code: Process exit code
        duration_ms: Execution duration in milliseconds
    """
    success: bool
    output: Any
    error: Optional[str] = None
    exit_code: int = 0
    duration_ms: float = 0


class CommandRunner:
    """Execute plugin commands.
    
    Supports:
    - Subprocess execution (sandboxed)
    - In-process execution (faster)
    - Timeout handling
    - JSON argument passing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize command runner.
        
        Args:
            config: Configuration with optional keys:
                - timeout: Default timeout in seconds (default: 30)
                - python_path: Python interpreter path
                - in_process: Run commands in-process (default: False)
                - env: Additional environment variables
        """
        self.config = config or {}
        self.timeout = self.config.get("timeout", 30)
        self.python_path = self.config.get("python_path", sys.executable)
        self.in_process = self.config.get("in_process", False)
        self.extra_env = self.config.get("env", {})
        self._loaded_commands: Dict[str, Callable] = {}
    
    def discover_commands(self, plugin_path: Path, commands_dir: str = "./commands/") -> List[Dict[str, Any]]:
        """Discover available commands in a plugin.
        
        Args:
            plugin_path: Plugin base path
            commands_dir: Commands directory relative path
            
        Returns:
            List of command descriptors
        """
        commands_path = plugin_path / commands_dir.lstrip("./")
        if not commands_path.exists():
            return []
        
        commands = []
        for cmd_file in commands_path.glob("*.py"):
            if cmd_file.stem.startswith("_"):
                continue
            
            # Try to extract docstring for description
            description = ""
            try:
                content = cmd_file.read_text()
                # Simple docstring extraction
                if content.startswith('"""'):
                    end = content.find('"""', 3)
                    if end > 0:
                        description = content[3:end].strip().split("\n")[0]
                elif content.startswith("'''"):
                    end = content.find("'''", 3)
                    if end > 0:
                        description = content[3:end].strip().split("\n")[0]
            except Exception:
                pass
            
            commands.append({
                "name": cmd_file.stem,
                "path": str(cmd_file),
                "description": description,
            })
        
        return commands
    
    async def execute(
        self,
        command_path: Path,
        arguments: Dict[str, Any] = None,
        working_dir: Path = None,
        timeout: float = None
    ) -> CommandResult:
        """Execute a command.
        
        Args:
            command_path: Path to command .py file
            arguments: Arguments to pass to command
            working_dir: Working directory (default: command's parent)
            timeout: Execution timeout (default: self.timeout)
            
        Returns:
            CommandResult with output or error
        """
        import time
        start_time = time.time()
        
        command_path = Path(command_path)
        if not command_path.exists():
            return CommandResult(
                success=False,
                output=None,
                error=f"Command file not found: {command_path}",
                exit_code=-1,
            )
        
        arguments = arguments or {}
        working_dir = working_dir or command_path.parent
        timeout = timeout or self.timeout
        
        if self.in_process:
            result = await self._execute_in_process(command_path, arguments)
        else:
            result = await self._execute_subprocess(
                command_path, arguments, working_dir, timeout
            )
        
        result.duration_ms = (time.time() - start_time) * 1000
        return result
    
    async def _execute_subprocess(
        self,
        command_path: Path,
        arguments: Dict[str, Any],
        working_dir: Path,
        timeout: float
    ) -> CommandResult:
        """Execute command in subprocess.
        
        Args:
            command_path: Path to command file
            arguments: Command arguments
            working_dir: Working directory
            timeout: Execution timeout
            
        Returns:
            CommandResult
        """
        # Prepare environment
        env = {
            **os.environ,
            **self.extra_env,
            "PLUGIN_ARGUMENTS": json.dumps(arguments),
            "PLUGIN_WORKING_DIR": str(working_dir),
            "PYTHONPATH": str(command_path.parent),
        }
        
        try:
            proc = await asyncio.create_subprocess_exec(
                self.python_path,
                str(command_path),
                cwd=str(working_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return CommandResult(
                    success=False,
                    output=None,
                    error=f"Command timed out after {timeout}s",
                    exit_code=-1,
                )
            
            output_str = stdout.decode().strip()
            error_str = stderr.decode().strip()
            
            if proc.returncode != 0:
                return CommandResult(
                    success=False,
                    output=output_str if output_str else None,
                    error=error_str or f"Command exited with code {proc.returncode}",
                    exit_code=proc.returncode or 1,
                )
            
            # Try to parse JSON output
            output: Any
            try:
                output = json.loads(output_str) if output_str else None
            except json.JSONDecodeError:
                output = output_str
            
            return CommandResult(
                success=True,
                output=output,
                error=error_str if error_str else None,
                exit_code=0,
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                output=None,
                error=str(e),
                exit_code=-1,
            )
    
    async def _execute_in_process(
        self,
        command_path: Path,
        arguments: Dict[str, Any]
    ) -> CommandResult:
        """Execute command in current process.
        
        Faster but less isolated. Requires command to define
        a run(arguments) or main(arguments) function.
        
        Args:
            command_path: Path to command file
            arguments: Command arguments
            
        Returns:
            CommandResult
        """
        cache_key = str(command_path)
        
        try:
            # Load or get cached module
            if cache_key not in self._loaded_commands:
                spec = importlib.util.spec_from_file_location(
                    f"plugin_cmd_{command_path.stem}",
                    command_path
                )
                if spec is None or spec.loader is None:
                    return CommandResult(
                        success=False,
                        output=None,
                        error=f"Could not load command: {command_path}",
                        exit_code=-1,
                    )
                
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                
                # Find callable
                if hasattr(module, "run"):
                    self._loaded_commands[cache_key] = module.run
                elif hasattr(module, "main"):
                    self._loaded_commands[cache_key] = module.main
                else:
                    return CommandResult(
                        success=False,
                        output=None,
                        error="Command must define run() or main() function",
                        exit_code=-1,
                    )
            
            handler = self._loaded_commands[cache_key]
            
            # Execute
            if asyncio.iscoroutinefunction(handler):
                result = await handler(arguments)
            else:
                result = handler(arguments)
            
            return CommandResult(
                success=True,
                output=result,
                exit_code=0,
            )
            
        except Exception as e:
            logger.error(f"In-process command execution failed: {e}")
            return CommandResult(
                success=False,
                output=None,
                error=str(e),
                exit_code=-1,
            )
    
    def clear_cache(self) -> None:
        """Clear loaded command cache."""
        self._loaded_commands.clear()
