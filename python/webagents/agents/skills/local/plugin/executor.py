"""
Plugin Executor - Execute Plugin Components

Handles execution of plugin commands, skills, and hooks.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import Plugin
    from .components.skill_runner import SkillMD

logger = logging.getLogger(__name__)


class PluginExecutor:
    """Execute plugin components.
    
    Handles:
    - Python command execution
    - SKILL.md execution
    - Hook execution
    - Sandboxed execution (optional)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize executor.
        
        Args:
            config: Configuration with optional keys:
                - sandbox: Enable sandboxed execution (default: False)
                - timeout: Default execution timeout in seconds
                - python_path: Custom Python interpreter path
        """
        self.config = config or {}
        self.sandbox_enabled = self.config.get("sandbox", False)
        self.timeout = self.config.get("timeout", 30)
        self.python_path = self.config.get("python_path", sys.executable)
    
    async def execute_command(
        self, 
        plugin: "Plugin",
        command_name: str,
        arguments: Dict[str, Any] = None,
        context: Any = None
    ) -> Dict[str, Any]:
        """Execute a plugin command.
        
        Args:
            plugin: Plugin instance
            command_name: Name of command to execute
            arguments: Command arguments
            context: Optional execution context
            
        Returns:
            Command result dict
        """
        arguments = arguments or {}
        
        # Find command file
        commands_dir = plugin.path / plugin.manifest.commands.lstrip("./")
        command_file = commands_dir / f"{command_name}.py"
        
        if not command_file.exists():
            return {
                "error": f"Command not found: {command_name}",
                "plugin": plugin.name,
            }
        
        # Execute command
        try:
            result = await self._execute_python_file(
                command_file,
                arguments,
                plugin.path,
            )
            return {
                "status": "success",
                "plugin": plugin.name,
                "command": command_name,
                "result": result,
            }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "error": str(e),
                "plugin": plugin.name,
                "command": command_name,
            }
    
    async def execute_skill(
        self,
        plugin: "Plugin",
        skill: "SkillMD",
        arguments: Dict[str, Any] = None,
        agent: Any = None
    ) -> str:
        """Execute a plugin skill (SKILL.md).
        
        Args:
            plugin: Plugin instance
            skill: Parsed SkillMD
            arguments: Skill arguments
            agent: Agent instance for context
            
        Returns:
            Skill execution result (content to inject)
        """
        from .components.skill_runner import SkillRunner
        
        runner = SkillRunner()
        content = runner.substitute_arguments(skill.content, arguments or {})
        
        # Execute based on context mode
        if skill.context == "fork":
            return await self._execute_forked(content, skill, agent)
        else:
            return await self._execute_inline(content, skill, agent)
    
    async def execute_hook(
        self,
        plugin: "Plugin",
        hook_config: Dict[str, Any],
        event_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a plugin hook.
        
        Args:
            plugin: Plugin instance
            hook_config: Hook configuration dict
            event_data: Event data to pass to hook
            
        Returns:
            Hook execution result
        """
        handler_path = hook_config.get("handler")
        if not handler_path:
            return {"error": "No handler specified for hook"}
        
        # Resolve handler path relative to plugin
        if not handler_path.startswith("/"):
            handler_file = plugin.path / handler_path.lstrip("./")
        else:
            handler_file = Path(handler_path)
        
        if not handler_file.exists():
            return {"error": f"Hook handler not found: {handler_path}"}
        
        try:
            result = await self._execute_python_file(
                handler_file,
                event_data or {},
                plugin.path,
            )
            return {
                "status": "success",
                "event": hook_config.get("event"),
                "result": result,
            }
        except Exception as e:
            logger.error(f"Hook execution failed: {e}")
            return {
                "error": str(e),
                "event": hook_config.get("event"),
            }
    
    async def _execute_python_file(
        self,
        file_path: Path,
        arguments: Dict[str, Any],
        working_dir: Path
    ) -> Any:
        """Execute a Python file.
        
        Args:
            file_path: Path to Python file
            arguments: Arguments to pass
            working_dir: Working directory
            
        Returns:
            Execution result
        """
        import json
        import asyncio
        
        # Prepare environment
        env = {
            **dict(subprocess.os.environ),
            "PLUGIN_ARGUMENTS": json.dumps(arguments),
            "PLUGIN_WORKING_DIR": str(working_dir),
        }
        
        # Execute in subprocess
        proc = await asyncio.create_subprocess_exec(
            self.python_path,
            str(file_path),
            cwd=str(working_dir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"Command timed out after {self.timeout}s")
        
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {stderr.decode()}")
        
        # Try to parse JSON output
        output = stdout.decode().strip()
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output
    
    async def _execute_inline(
        self, 
        content: str, 
        skill: "SkillMD", 
        agent: Any
    ) -> str:
        """Execute skill inline in current agent context.
        
        The skill content is returned to be injected into the conversation.
        
        Args:
            content: Processed skill content
            skill: Skill metadata
            agent: Agent instance
            
        Returns:
            Content to inject
        """
        # For inline execution, just return the content
        # It will be added to the agent's context
        return content
    
    async def _execute_forked(
        self, 
        content: str, 
        skill: "SkillMD", 
        agent: Any
    ) -> str:
        """Execute skill in a forked subagent.
        
        Creates a subagent with restricted tools if specified.
        
        Args:
            content: Processed skill content
            skill: Skill metadata with tool restrictions
            agent: Parent agent instance
            
        Returns:
            Subagent execution result
        """
        # For now, just return content with fork indicator
        # TODO: Implement actual subagent forking
        logger.info(f"Forked execution requested for skill: {skill.name}")
        
        # If allowed_tools is specified, note the restriction
        if skill.allowed_tools:
            logger.info(f"Skill restricts tools to: {skill.allowed_tools}")
        
        return content
