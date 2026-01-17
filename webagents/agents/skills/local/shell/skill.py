"""
Shell Skill

Shell command execution with proper sandboxing using command whitelist/blacklist.
"""

import shlex
import subprocess
from typing import List, Optional, Set, Dict, Any, Tuple
from pathlib import Path

from ...base import Skill
from webagents.agents.tools.decorators import tool


class ShellSkill(Skill):
    """Shell command execution with sandboxing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Command sandboxing
        self.allowed_commands: Set[str] = self._load_allowed_commands()
        self.blocked_commands: Set[str] = self._load_blocked_commands()
        
        # Filesystem sandboxing
        if config and config.get("base_dir"):
            self.working_dir = Path(config["base_dir"]).resolve()
            self.sandbox_enabled = True
        else:
            self.working_dir = Path.cwd()
            self.sandbox_enabled = False
    
    def _load_allowed_commands(self) -> Set[str]:
        """Load whitelisted commands"""
        # Default safe commands
        allowed = {
            "ls", "cat", "grep", "find", "head", "tail", "wc",
            "echo", "date", "pwd", "which", "whereis",
            "git", "npm", "pip", "python", "node", "uvx", "curl", "wget",
        }
        
        if self.config and "allowed_commands" in self.config:
            allowed.update(self.config["allowed_commands"])
        
        return allowed
    
    def _load_blocked_commands(self) -> Set[str]:
        """Load blacklisted commands"""
        # Dangerous commands
        blocked = {
            "rm", "rmdir", "dd", "mkfs", "fdisk",
            "kill", "killall", "pkill",
            "shutdown", "reboot", "halt",
            "su", "sudo", "chmod", "chown",
            ":(){:|:&};:",  # Fork bomb
        }
        
        if self.config and "blocked_commands" in self.config:
            blocked.update(self.config["blocked_commands"])
        
        return blocked
    
    def _check_command(self, command: str) -> Tuple[bool, str]:
        """Check if command is allowed and safe"""
        try:
            # Use shlex to parse command line correctly handling quotes
            cmd_parts = shlex.split(command)
        except ValueError as e:
            return False, f"Invalid command format: {e}"
            
        if not cmd_parts:
            return False, "Empty command"
        
        # Iterate over tokens to check for allowed commands and sandbox violations
        # Note: This is a simple heuristic and not a full shell parser.
        # It assumes that the first token is the command, and subsequent tokens are args.
        # It handles simple chaining like "cmd1 && cmd2" by treating separators as boundaries.
        
        command_separators = {';', '&&', '||', '|'}
        
        current_cmd_start = True
        
        for token in cmd_parts:
            if token in command_separators:
                current_cmd_start = True
                continue
                
            if current_cmd_start:
                # This token is a command executable
                if token not in self.allowed_commands and token not in command_separators:
                    # Check if it's a relative path to a script (e.g. ./script.sh)
                    if not (token.startswith("./") or token.startswith("python")): 
                         # Loosen check slightly for common script execution, 
                         # but strictly enforce whitelist for system binaries
                         return False, f"Command not allowed: {token}"
                
                # Also check blocked commands (redundant but safe)
                for blocked in self.blocked_commands:
                    if blocked == token:
                        return False, f"Blocked command: {token}"
                        
                current_cmd_start = False
            else:
                # This token is an argument
                if self.sandbox_enabled:
                    # Sandbox check 1: No absolute paths (outside sandbox)
                    # Exception: legitimate flags/options might start with /? unlikely but possible.
                    # Usually / implies path.
                    if token.startswith("/") and Path(token).exists():
                         # Check if it resolves inside sandbox
                         try:
                             p = Path(token).resolve()
                             if not (self.working_dir in p.parents or p == self.working_dir):
                                 return False, f"Access denied: Absolute path outside sandbox: {token}"
                         except Exception:
                             pass # If not a valid path, maybe just a weird string
                             
                    # Sandbox check 2: No parent directory traversal
                    if ".." in token:
                         # This blocks ".." even in strings like "version..1", which is a trade-off.
                         # Tighter check: match path components
                         parts = Path(token).parts
                         if ".." in parts:
                             return False, f"Access denied: Parent directory traversal ('..') not allowed in sandbox: {token}"
                    
                    # Sandbox check 3: No home directory expansion
                    if token.startswith("~"):
                        return False, f"Access denied: Home directory expansion ('~') not allowed in sandbox: {token}"

        return True, ""
    
    @tool
    async def run_command(self, command: str, timeout: int = 30) -> str:
        """Run a shell command (sandboxed)
        
        Args:
            command: Command to run
            timeout: Timeout in seconds (default: 30)
        
        Returns:
            Command output or error message
        """
        # Check for Sandbox (Docker)
        if self.agent:
            sandbox_skill = None
            for skill in self.agent.skills.values():
                 # Check for SandboxSkill
                 if skill.__class__.__name__ == "SandboxSkill":
                     sandbox_skill = skill
                     break
            
            if sandbox_skill:
                # Delegate to Sandbox for ultimate security
                return await sandbox_skill.run_sandbox_command(command)

        allowed, reason = self._check_command(command)
        if not allowed:
            return f"Access denied: {reason}"
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nStderr: {result.stderr}"
            
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            
            return output if output else "(No output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error executing command: {e}"
